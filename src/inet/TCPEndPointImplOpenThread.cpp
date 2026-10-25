/*
 *
 *    Copyright (c) 2020-2021 Project CHIP Authors
 *    Copyright (c) 2018 Google LLC.
 *    Copyright (c) 2013-2018 Nest Labs, Inc.
 *
 *    Licensed under the Apache License, Version 2.0 (the "License");
 *    you may not use this file except in compliance with the License.
 *    You may obtain a copy of the License at
 *
 *        http://www.apache.org/licenses/LICENSE-2.0
 *
 *    Unless required by applicable law or agreed to in writing, software
 *    distributed under the License is distributed on an "AS IS" BASIS,
 *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *    See the License for the specific language governing permissions and
 *    limitations under the License.
 */

#include <inet/TCPEndPointImplOpenThread.h>

#include <inet/UDPEndPointImplOpenThread.h>
#include <lib/support/CodeUtils.h>
#include <lib/support/SafeInt.h>
#include <lib/support/logging/CHIPLogging.h>

#include <platform/OpenThread/OpenThreadUtils.h>

#include <system/SystemPacketBuffer.h>

#include <cstring>
#include <utility>

namespace chip {
namespace Inet {

using DeviceLayer::Internal::MapOpenThreadError;

otInstance * TCPEndPointImplOT::GetOtInstance() const
{
    return (mOTInstance != nullptr) ? mOTInstance : globalOtInstance;
}

CHIP_ERROR TCPEndPointImplOT::InitEndpoint()
{
    if (mIsEndpointInitialized)
    {
        return CHIP_NO_ERROR;
    }

    otInstance * instance = GetOtInstance();
    VerifyOrReturnError(instance != nullptr, CHIP_ERROR_INCORRECT_STATE);

    memset(&mEndpoint, 0, sizeof(mEndpoint));

    otTcpEndpointInitializeArgs args;
    memset(&args, 0, sizeof(args));
    args.mContext                  = this;
    args.mEstablishedCallback      = &TCPEndPointImplOT::HandleTcpEstablishedCallback;
    args.mSendDoneCallback         = &TCPEndPointImplOT::HandleTcpSendDoneCallback;
    args.mForwardProgressCallback  = nullptr;
    args.mReceiveAvailableCallback = &TCPEndPointImplOT::HandleTcpReceiveAvailableCallback;
    args.mDisconnectedCallback     = &TCPEndPointImplOT::HandleTcpDisconnectedCallback;
    args.mReceiveBuffer            = mReceiveBuffer;
    args.mReceiveBufferSize        = sizeof(mReceiveBuffer);

    otError otErr = otTcpEndpointInitialize(instance, &mEndpoint, &args);
    ReturnErrorOnFailure(MapOpenThreadError(otErr));

    mIsEndpointInitialized = true;
    return CHIP_NO_ERROR;
}

CHIP_ERROR TCPEndPointImplOT::GetPeerInfo(IPAddress * retAddr, uint16_t * retPort) const
{
    VerifyOrReturnError(IsConnected() && mIsEndpointInitialized, CHIP_ERROR_INCORRECT_STATE);
    const otSockAddr * peerAddr = otTcpGetPeerAddress(&mEndpoint);
    VerifyOrReturnError(peerAddr != nullptr, CHIP_ERROR_INCORRECT_STATE);

    *retAddr = IPAddress::FromOtAddr(peerAddr->mAddress);
    *retPort = peerAddr->mPort;
    return CHIP_NO_ERROR;
}

CHIP_ERROR TCPEndPointImplOT::GetLocalInfo(IPAddress * retAddr, uint16_t * retPort) const
{
    VerifyOrReturnError(IsConnected() && mIsEndpointInitialized, CHIP_ERROR_INCORRECT_STATE);
    const otSockAddr * localAddr = otTcpGetLocalAddress(&mEndpoint);
    VerifyOrReturnError(localAddr != nullptr, CHIP_ERROR_INCORRECT_STATE);

    *retAddr = IPAddress::FromOtAddr(localAddr->mAddress);
    *retPort = localAddr->mPort;
    return CHIP_NO_ERROR;
}

CHIP_ERROR TCPEndPointImplOT::GetInterfaceId(InterfaceId * retInterface)
{
    VerifyOrReturnError(IsConnected(), CHIP_ERROR_INCORRECT_STATE);
    *retInterface = mBoundIntfId;
    return CHIP_NO_ERROR;
}

CHIP_ERROR TCPEndPointImplOT::EnableNoDelay()
{
    VerifyOrReturnError(IsConnected(), CHIP_ERROR_INCORRECT_STATE);
    return CHIP_NO_ERROR;
}

CHIP_ERROR TCPEndPointImplOT::EnableKeepAlive(uint16_t interval, uint16_t timeoutCount)
{
    (void) interval;
    (void) timeoutCount;
    VerifyOrReturnError(IsConnected(), CHIP_ERROR_INCORRECT_STATE);
    return CHIP_NO_ERROR;
}

CHIP_ERROR TCPEndPointImplOT::DisableKeepAlive()
{
    VerifyOrReturnError(IsConnected(), CHIP_ERROR_INCORRECT_STATE);
    return CHIP_NO_ERROR;
}

CHIP_ERROR TCPEndPointImplOT::AckReceive(size_t len)
{
    (void) len;
    VerifyOrReturnError(IsConnected(), CHIP_ERROR_INCORRECT_STATE);
    return CHIP_NO_ERROR;
}

#if INET_CONFIG_OVERRIDE_SYSTEM_TCP_USER_TIMEOUT
void TCPEndPointImplOT::TCPUserTimeoutHandler()
{
    // Handled by OpenThread TCPlp timers.
}
#endif // INET_CONFIG_OVERRIDE_SYSTEM_TCP_USER_TIMEOUT

CHIP_ERROR TCPEndPointImplOT::BindImpl(IPAddressType addrType, const IPAddress & addr, uint16_t port, bool reuseAddr)
{
    (void) reuseAddr;
    if (addrType != IPAddressType::kIPv6 && addrType != IPAddressType::kAny)
    {
        return INET_ERROR_WRONG_ADDRESS_TYPE;
    }

    mBoundAddr = addr;
    mBoundPort = port;
    return CHIP_NO_ERROR;
}

CHIP_ERROR TCPEndPointImplOT::ListenImpl(uint16_t backlog)
{
    (void) backlog;
    otInstance * instance = GetOtInstance();
    VerifyOrReturnError(instance != nullptr, CHIP_ERROR_INCORRECT_STATE);

    LockOpenThread();
    if (!mIsListenerInitialized)
    {
        memset(&mListener, 0, sizeof(mListener));
        otTcpListenerInitializeArgs listenerArgs;
        memset(&listenerArgs, 0, sizeof(listenerArgs));
        listenerArgs.mContext             = this;
        listenerArgs.mAcceptReadyCallback = &TCPEndPointImplOT::HandleTcpAcceptReadyCallback;
        listenerArgs.mAcceptDoneCallback  = &TCPEndPointImplOT::HandleTcpAcceptDoneCallback;

        otError initErr = otTcpListenerInitialize(instance, &mListener, &listenerArgs);
        if (initErr != OT_ERROR_NONE)
        {
            UnlockOpenThread();
            return MapOpenThreadError(initErr);
        }
        mIsListenerInitialized = true;
    }

    otSockAddr listenSockAddr;
    memset(&listenSockAddr, 0, sizeof(listenSockAddr));
    listenSockAddr.mAddress = mBoundAddr.ToIPv6();
    listenSockAddr.mPort    = mBoundPort;

    otError otErr = otTcpListen(&mListener, &listenSockAddr);
    if (otErr != OT_ERROR_NONE)
    {
        (void) otTcpListenerDeinitialize(&mListener);
        mIsListenerInitialized = false;
    }
    UnlockOpenThread();

    return MapOpenThreadError(otErr);
}

CHIP_ERROR TCPEndPointImplOT::ConnectImpl(const IPAddress & addr, uint16_t port, InterfaceId intfId)
{
    if (addr.Type() != IPAddressType::kIPv6)
    {
        return INET_ERROR_WRONG_ADDRESS_TYPE;
    }

    mBoundIntfId = intfId;

    LockOpenThread();
    CHIP_ERROR err = InitEndpoint();
    if (err != CHIP_NO_ERROR)
    {
        UnlockOpenThread();
        return err;
    }

    if (mBoundAddr != IPAddress::Any)
    {
        otSockAddr bindAddr;
        memset(&bindAddr, 0, sizeof(bindAddr));
        bindAddr.mAddress = mBoundAddr.ToIPv6();
        bindAddr.mPort    = mBoundPort;
        otError bindErr   = otTcpBind(&mEndpoint, &bindAddr);
        if (bindErr != OT_ERROR_NONE)
        {
            mIsEndpointInitialized = false;
            (void) otTcpEndpointDeinitialize(&mEndpoint);
            UnlockOpenThread();
            return MapOpenThreadError(bindErr);
        }
    }

    otSockAddr peerSockAddr;
    memset(&peerSockAddr, 0, sizeof(peerSockAddr));
    peerSockAddr.mAddress = addr.ToIPv6();
    peerSockAddr.mPort    = port;

    mState        = State::kConnecting;
    otError otErr = otTcpConnect(&mEndpoint, &peerSockAddr, OT_TCP_CONNECT_NO_FAST_OPEN);
    if (otErr != OT_ERROR_NONE)
    {
        mState                 = State::kClosed;
        mIsEndpointInitialized = false;
        (void) otTcpEndpointDeinitialize(&mEndpoint);
        UnlockOpenThread();
        return MapOpenThreadError(otErr);
    }

    UnlockOpenThread();
    return CHIP_NO_ERROR;
}

CHIP_ERROR TCPEndPointImplOT::SendQueuedImpl(bool queueWasEmpty)
{
    (void) queueWasEmpty;
    return CHIP_NO_ERROR;
}

CHIP_ERROR TCPEndPointImplOT::SetUserTimeoutImpl(uint32_t userTimeoutMillis)
{
    (void) userTimeoutMillis;
    return CHIP_NO_ERROR;
}

CHIP_ERROR TCPEndPointImplOT::DriveSendingImpl()
{
    VerifyOrReturnError(mIsEndpointInitialized, CHIP_ERROR_INCORRECT_STATE);

    CHIP_ERROR err = CHIP_NO_ERROR;
    LockOpenThread();

    while (!mSendQueue.IsNull())
    {
        PendingSend * slot = nullptr;
        for (size_t i = 0; i < kMaxPendingSends; ++i)
        {
            if (!mPendingSends[i].mInUse)
            {
                slot = &mPendingSends[i];
                break;
            }
        }

        if (slot == nullptr)
        {
            // All pending send slots are currently in flight; HandleTcpSendDoneCallback will resume DriveSending.
            break;
        }

        System::PacketBufferHandle head = mSendQueue.PopHead();
        slot->mInUse                    = true;
        slot->mBuffer                   = std::move(head);
        slot->mLinkedBuffer.mData       = slot->mBuffer->Start();
        slot->mLinkedBuffer.mLength     = slot->mBuffer->DataLength();
        slot->mLinkedBuffer.mNext       = nullptr;

        uint32_t flags = (!mSendQueue.IsNull()) ? OT_TCP_SEND_MORE_TO_COME : 0;
        otError otErr  = otTcpSendByReference(&mEndpoint, &slot->mLinkedBuffer, flags);
        if (otErr != OT_ERROR_NONE)
        {
            if (mSendQueue.IsNull())
            {
                mSendQueue = std::move(slot->mBuffer);
            }
            else
            {
                slot->mBuffer->AddToEnd(std::move(mSendQueue));
                mSendQueue = std::move(slot->mBuffer);
            }
            slot->mInUse = false;
            err          = MapOpenThreadError(otErr);
            break;
        }

        MarkActive();
    }

    if (err == CHIP_NO_ERROR && mState == State::kSendShutdown && mSendQueue.IsNull())
    {
        (void) otTcpSendEndOfStream(&mEndpoint);
    }

    UnlockOpenThread();
    return err;
}

void TCPEndPointImplOT::HandleConnectCompleteImpl() {}

void TCPEndPointImplOT::DoCloseImpl(CHIP_ERROR err, State oldState)
{
    (void) oldState;
    if (mState != State::kClosed && (mState != State::kClosing || !mSendQueue.IsNull()))
    {
        return;
    }

    LockOpenThread();
    if (mIsEndpointInitialized)
    {
        mIsEndpointInitialized = false;
        if (err != CHIP_NO_ERROR)
        {
            (void) otTcpAbort(&mEndpoint);
        }
        (void) otTcpEndpointDeinitialize(&mEndpoint);
    }

    for (size_t i = 0; i < kMaxPendingSends; ++i)
    {
        mPendingSends[i].mBuffer = nullptr;
        mPendingSends[i].mInUse  = false;
    }

    if (mIsListenerInitialized)
    {
        mIsListenerInitialized = false;
        (void) otTcpListenerDeinitialize(&mListener);
    }

    mPendingAcceptEndPoint = nullptr;
    UnlockOpenThread();
}

void TCPEndPointImplOT::HandleTcpEstablishedCallback(otTcpEndpoint * aEndpoint)
{
    auto * ep = static_cast<TCPEndPointImplOT *>(otTcpEndpointGetContext(aEndpoint));
    VerifyOrReturn(ep != nullptr);

    ep->Ref();
    CHIP_ERROR err = ep->GetSystemLayer().ScheduleLambda([ep] {
        if (ep->mState == State::kConnecting)
        {
            ep->HandleConnectComplete(CHIP_NO_ERROR);
        }
        ep->Unref();
    });
    if (err != CHIP_NO_ERROR)
    {
        ep->Unref();
    }
}

void TCPEndPointImplOT::HandleTcpSendDoneCallback(otTcpEndpoint * aEndpoint, otLinkedBuffer * aData)
{
    auto * ep = static_cast<TCPEndPointImplOT *>(otTcpEndpointGetContext(aEndpoint));
    VerifyOrReturn(ep != nullptr && aData != nullptr);

    size_t lenSent = aData->mLength;
    for (size_t i = 0; i < kMaxPendingSends; ++i)
    {
        if (ep->mPendingSends[i].mInUse && &ep->mPendingSends[i].mLinkedBuffer == aData)
        {
            ep->mPendingSends[i].mBuffer = nullptr;
            ep->mPendingSends[i].mInUse  = false;
            break;
        }
    }

    ep->Ref();
    CHIP_ERROR err = ep->GetSystemLayer().ScheduleLambda([ep, lenSent] {
        TCPEndPointHandle handle(ep);
        if (ep->IsConnected())
        {
            if (ep->OnDataSent != nullptr)
            {
                ep->OnDataSent(handle, lenSent);
            }
            if (!ep->mSendQueue.IsNull() || ep->mState == State::kSendShutdown)
            {
                TEMPORARY_RETURN_IGNORED ep->DriveSending();
            }
        }
        ep->Unref();
    });
    if (err != CHIP_NO_ERROR)
    {
        ep->Unref();
    }
}

void TCPEndPointImplOT::HandleTcpReceiveAvailableCallback(otTcpEndpoint * aEndpoint, size_t aBytesAvailable, bool aEndOfStream,
                                                          size_t aBytesRemaining)
{
    (void) aBytesRemaining;
    auto * ep = static_cast<TCPEndPointImplOT *>(otTcpEndpointGetContext(aEndpoint));
    VerifyOrReturn(ep != nullptr);

    if (aBytesAvailable == 0 && !aEndOfStream)
    {
        return;
    }

    System::PacketBufferHandle rcvBuf;
    size_t bytesCopied = 0;

    if (aBytesAvailable > 0)
    {
        const otLinkedBuffer * data = nullptr;
        if (otTcpReceiveByReference(aEndpoint, &data) == OT_ERROR_NONE && data != nullptr)
        {
            rcvBuf = System::PacketBufferHandle::New(aBytesAvailable, 0);
            if (!rcvBuf.IsNull())
            {
                uint8_t * dst = rcvBuf->Start();
                size_t space  = rcvBuf->AvailableDataLength();
                for (const otLinkedBuffer * cur = data; cur != nullptr && cur->mLength > 0; cur = cur->mNext)
                {
                    size_t toCopy = (cur->mLength <= space) ? cur->mLength : space;
                    if (toCopy == 0)
                    {
                        break;
                    }
                    memcpy(dst + bytesCopied, cur->mData, toCopy);
                    bytesCopied += toCopy;
                    space -= toCopy;
                }
                rcvBuf->SetDataLength(bytesCopied);
                (void) otTcpCommitReceive(aEndpoint, bytesCopied, 0);
            }
        }
    }

    auto * rawBuf = std::move(rcvBuf).UnsafeRelease();
    ep->Ref();
    CHIP_ERROR err = ep->GetSystemLayer().ScheduleLambda([ep, rawBuf, aEndOfStream] {
        System::PacketBufferHandle buf = System::PacketBufferHandle::Adopt(rawBuf);
        TCPEndPointHandle handle(ep);

        if (ep->mState != State::kConnected && ep->mState != State::kSendShutdown)
        {
            ep->Unref();
            return;
        }

        ep->MarkActive();

        if (!buf.IsNull())
        {
            if (ep->mRcvQueue.IsNull())
            {
                ep->mRcvQueue = std::move(buf);
            }
            else
            {
                ep->mRcvQueue->AddToEnd(std::move(buf));
            }
            ep->DriveReceiving(handle);
        }

        if (aEndOfStream && (ep->mState == State::kConnected || ep->mState == State::kSendShutdown))
        {
            if (ep->mState == State::kConnected && ep->OnPeerClose != nullptr)
            {
                ep->mState = State::kReceiveShutdown;
            }
            else
            {
                ep->mState = State::kClosing;
            }
            if (ep->OnPeerClose != nullptr)
            {
                ep->OnPeerClose(handle);
            }
            ep->DriveReceiving(handle);
        }

        ep->Unref();
    });
    if (err != CHIP_NO_ERROR)
    {
        ep->Unref();
        System::PacketBufferHandle cleanup = System::PacketBufferHandle::Adopt(rawBuf);
    }
}

void TCPEndPointImplOT::HandleTcpDisconnectedCallback(otTcpEndpoint * aEndpoint, otTcpDisconnectedReason aReason)
{
    auto * ep = static_cast<TCPEndPointImplOT *>(otTcpEndpointGetContext(aEndpoint));
    VerifyOrReturn(ep != nullptr);

    CHIP_ERROR closeErr = CHIP_NO_ERROR;
    switch (aReason)
    {
    case OT_TCP_DISCONNECTED_REASON_NORMAL:
    case OT_TCP_DISCONNECTED_REASON_TIME_WAIT:
        closeErr = CHIP_NO_ERROR;
        break;
    case OT_TCP_DISCONNECTED_REASON_RESET:
        closeErr = CHIP_ERROR_CONNECTION_ABORTED;
        break;
    case OT_TCP_DISCONNECTED_REASON_TIMED_OUT:
        closeErr = CHIP_ERROR_TIMEOUT;
        break;
    default:
        closeErr = CHIP_ERROR_CONNECTION_ABORTED;
        break;
    }

    ep->Ref();
    CHIP_ERROR err = ep->GetSystemLayer().ScheduleLambda([ep, closeErr] {
        if (ep->mState != State::kClosed)
        {
            ep->DoClose(closeErr, false);
        }
        ep->Unref();
    });
    if (err != CHIP_NO_ERROR)
    {
        ep->Unref();
    }
}

otTcpIncomingConnectionAction TCPEndPointImplOT::HandleTcpAcceptReadyCallback(otTcpListener * aListener, const otSockAddr * aPeer,
                                                                              otTcpEndpoint ** aAcceptInto)
{
    (void) aPeer;
    auto * listenEp = static_cast<TCPEndPointImplOT *>(otTcpListenerGetContext(aListener));
    if (listenEp == nullptr || listenEp->mState != State::kListening || listenEp->OnConnectionReceived == nullptr)
    {
        return OT_TCP_INCOMING_CONNECTION_ACTION_REFUSE;
    }

    TCPEndPointHandle connectEndPoint;
    if (listenEp->GetEndPointManager().NewEndPoint(connectEndPoint) != CHIP_NO_ERROR)
    {
        return OT_TCP_INCOMING_CONNECTION_ACTION_REFUSE;
    }

    auto & conEP       = static_cast<TCPEndPointImplOT &>(*connectEndPoint);
    conEP.mOTInstance  = listenEp->GetOtInstance();
    conEP.mBoundIntfId = listenEp->mBoundIntfId;
    if (conEP.InitEndpoint() != CHIP_NO_ERROR)
    {
        return OT_TCP_INCOMING_CONNECTION_ACTION_REFUSE;
    }

    listenEp->mPendingAcceptEndPoint = std::move(connectEndPoint);
    *aAcceptInto                     = &conEP.mEndpoint;
    return OT_TCP_INCOMING_CONNECTION_ACTION_ACCEPT;
}

void TCPEndPointImplOT::HandleTcpAcceptDoneCallback(otTcpListener * aListener, otTcpEndpoint * aEndpoint, const otSockAddr * aPeer)
{
    (void) aEndpoint;
    auto * listenEp = static_cast<TCPEndPointImplOT *>(otTcpListenerGetContext(aListener));
    VerifyOrReturn(listenEp != nullptr && !listenEp->mPendingAcceptEndPoint.IsNull() && aPeer != nullptr);

    TCPEndPointHandle connectEndPoint = std::move(listenEp->mPendingAcceptEndPoint);
    auto & conEP                      = static_cast<TCPEndPointImplOT &>(*connectEndPoint);
    conEP.mState                      = State::kConnected;

    IPAddress peerAddr = IPAddress::FromOtAddr(aPeer->mAddress);
    uint16_t peerPort  = aPeer->mPort;

    listenEp->Ref();
    conEP.Ref();
    TCPEndPointImplOT * conEpPtr = &conEP;
    CHIP_ERROR err               = listenEp->GetSystemLayer().ScheduleLambda([listenEp, conEpPtr, peerAddr, peerPort] {
        TCPEndPointHandle listenHandle(listenEp);
        TCPEndPointHandle conHandle(conEpPtr);
        if (listenEp->mState == State::kListening && listenEp->OnConnectionReceived != nullptr)
        {
            listenEp->OnConnectionReceived(listenHandle, conHandle, peerAddr, peerPort);
        }
        conEpPtr->Unref();
        listenEp->Unref();
    });
    if (err != CHIP_NO_ERROR)
    {
        conEpPtr->Unref();
        listenEp->Unref();
    }
}

} // namespace Inet
} // namespace chip
