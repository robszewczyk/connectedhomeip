#!/usr/bin/env python3
# Copyright (c) 2026 Project CHIP Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
OTNS + Simulated Thread Border Router (OTBR) + Matter BDX OTA Benchmark Orchestrator.

Replicates the network topologies and BDX (Bulk Data Exchange / OTA Software Update Over Thread)
benchmark suites from Apple's "OTA Software Update Over Thread" presentation (Causeway #43975,
Slides 3-5 & 10) and the Matter Networking Improvements Tiger Team minutes (2026-05-12,
2026-05-19, 2026-06-02, 2026-06-09):

  - Topology 1 (`scripts/otns/otns_bdx_multihop.json`):
    4-node / 3-hop linear chain:
      Home Hub 1 (OTBR Leader) <-> Home Hub 2 (Router 1, Hop 1) <-> A1 (Router 2, Hop 2) <-> OTARequestor (Hop 3, SED or MED)
    Supports placing OTARequestor at 1, 2, or 3 hops and configuring OTNS packet loss ratio (`plr` = 0% vs 5% FER).

  - Topology 2 (`scripts/otns/otns_bdx_scenario2_router_drops.json`):
    2-hop fan-out through intermediate router A1:
      Home Hub (OTBR Leader) <-> A1 (Intermediate Router) <-> {A2, Outlet (OTARequestor), A3}
    Sibling nodes A2 and A3 (`all-devices-app`) share parent/router A1 and generate concurrent
    `SubscribeReq` / `ReportData` traffic during an OTA download to `Outlet`.

  - Benchmark Suites:
    - `suite1`: Slide 10 Lab Test Multi-Hop Comparison (1024 B vs 340 B over 3 hops, SED & MED, 0% & 5% FER)
    - `suite2`: 2026-05-19 Block-Size & 6LoWPAN Fragmentation Sweep (64..1024 B across 1, 2, 3 hops, 0% & 5% FER)
    - `suite3`: Scenario 1 Aggressive MRP / Large Block Size (1024 B vs 340 B, SAI=300ms vs 2000ms)
    - `suite4`: Scenario 2 Intermediate Router Contention (OTBR <-> A1 <-> {A2, Outlet, A3}, baseline vs concurrent traffic)
"""

import argparse
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
import glob
import json
import math
import os
import re
import shutil
import struct
import subprocess
import sys
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

OTNS_PYLIBS_DIR = "/home/otns/ot-ns/pylibs"
if os.path.isdir(OTNS_PYLIBS_DIR) and OTNS_PYLIBS_DIR not in sys.path:
    sys.path.insert(0, OTNS_PYLIBS_DIR)

for extra_bin_dir in ["/home/otns/go/bin", "/usr/local/go/bin", "/usr/sbin", "/sbin"]:
    if os.path.isdir(extra_bin_dir) and extra_bin_dir not in os.environ.get("PATH", "").split(os.pathsep):
        os.environ["PATH"] = f"{extra_bin_dir}{os.pathsep}{os.environ.get('PATH', '')}"

from otns.cli import OTNS

from otns_all_devices_env import (
    CHIP_TOOL_COMMANDS_LOG_PATH,
    INTERACTIVE_LOG_PATH,
    OT_TMP_DIR,
    OT_VERSIONS_DIR,
    WRAPPER_DIR,
    InteractiveChipTool,
    configure_border_router_routes,
    detect_backbone_interface,
    ensure_otns_patches,
    run_chip_tool,
    sanitize_slug,
    wait_for_otbr_ready,
    wait_for_routers_ready,
)

REPO_ROOT = "/workspace" if os.path.isdir("/workspace/scripts/otns") else os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
DEFAULT_SIM_LOGS_DIR = os.path.join(REPO_ROOT, "sim_logs")
DEFAULT_ALL_DEVICES_BIN = os.path.join(
    REPO_ROOT, "out/linux-x64-all-devices-no-wifi-no-ble-openthread-endpoint-otns/all-devices-app"
)
DEFAULT_OTA_REQUESTOR_BIN = os.path.join(
    REPO_ROOT, "out/linux-x64-ota-requestor-no-wifi-no-ble-openthread-endpoint-otns/chip-ota-requestor-app"
)
DEFAULT_OTA_PROVIDER_BIN = os.path.join(
    REPO_ROOT, "out/linux-x64-ota-provider/chip-ota-provider-app"
)
DEFAULT_CHIP_TOOL_BIN = os.path.join(
    REPO_ROOT, "out/linux-x64-chip-tool-no-ble/chip-tool"
)
OTA_IMAGE_TOOL_SCRIPT = os.path.join(REPO_ROOT, "src/app/ota_image_tool.py")

OTA_PROVIDER_LOG_PATH = "/tmp/ota_provider.log"
OTA_PROVIDER_KVS_PATH = "/tmp/chip_kvs_ota_provider"

SUITE2_BLOCK_SIZES = [64, 128, 224, 340, 416, 512, 608, 704, 800, 896, 1024]


@dataclass
class BdxMatterNodeSpec:
    node_id: int
    discriminator: int
    passcode: int
    name: str
    app_type: str  # "ota-requestor" or "all-devices"
    devices: List[str]
    x: int
    y: int
    radio_range: int = 180
    router_eligible: bool = False
    thread_mode: str = "med"  # "med" or "sed"
    poll_period_ms: int = 250
    max_bdx_block_size: int = 1024
    ota_download_path: str = "/tmp/ota_dl.bin"
    requestor_sai_ms: Optional[int] = None
    requestor_sii_ms: Optional[int] = None
    otns_id: Optional[int] = None
    wrapper_path: Optional[str] = None
    log_path: Optional[str] = None


@dataclass
class BdxTopologyConfig:
    topology_id: str
    description: str
    config_path: str
    otbr_x: int
    otbr_y: int
    otbr_radio_range: int
    otbr_name: str
    routers: List[Dict[str, Any]]
    matter_nodes: List[BdxMatterNodeSpec]
    provider_node_id: int = 999
    provider_discriminator: int = 2222
    provider_passcode: int = 20202021
    hops: int = 3
    plr: float = 0.0
    raw_json: Dict[str, Any] = field(default_factory=dict)


def load_bdx_topology(
    config_path: str,
    hops: int = 3,
    mode: str = "med",
    poll_period_ms: int = 250,
    plr: float = 0.0,
    max_bdx_block_size: int = 1024,
    requestor_sai_ms: Optional[int] = None,
    requestor_sii_ms: Optional[int] = None,
) -> BdxTopologyConfig:
    """Load and parameterize a BDX declarative topology JSON file."""
    with open(config_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    topo_id = str(raw.get("topology_id", os.path.splitext(os.path.basename(config_path))[0]))
    desc = str(raw.get("description", ""))
    otbr_cfg = raw.get("otbr", {})
    otbr_x = int(otbr_cfg.get("x", 100))
    otbr_y = int(otbr_cfg.get("y", 250))
    otbr_rr = int(otbr_cfg.get("radio_range", 180))
    otbr_name = str(otbr_cfg.get("name", "Home Hub 1 (OTBR Leader)"))

    prov_cfg = raw.get("ota_provider", {})
    prov_node_id = int(prov_cfg.get("node_id", 999))
    prov_disc = int(prov_cfg.get("discriminator", 2222))
    prov_pass = int(prov_cfg.get("passcode", 20202021))

    all_routers = list(raw.get("routers", []))
    hop_placements = raw.get("hop_placements") or raw.get("hop_presets") or {}
    if str(hops) in hop_placements:
        hp = hop_placements[str(hops)]
        active_rtr_count = int(hp.get("active_routers", max(0, hops - 1)))
        routers = all_routers[:active_rtr_count]
        req_override_x = int(hp["requestor_x"])
        req_override_y = int(hp["requestor_y"])
    else:
        routers = all_routers
        req_override_x = None
        req_override_y = None

    matter_nodes: List[BdxMatterNodeSpec] = []
    for idx, item in enumerate(raw.get("matter_nodes", [])):
        nid = int(item.get("node_id", 101 + idx))
        raw_app = str(item.get("app_type") or item.get("app") or "ota-requestor")
        app_type = "all-devices" if "all-devices" in raw_app else "ota-requestor"
        nx = int(item.get("x", 250))
        ny = int(item.get("y", 250))
        nmode = str(item.get("thread_mode", mode)).lower()
        npoll = int(item.get("poll_period_ms", poll_period_ms))
        if app_type == "ota-requestor":
            if req_override_x is not None and req_override_y is not None:
                nx = req_override_x
                ny = req_override_y
            nmode = mode.lower()
            npoll = poll_period_ms

        matter_nodes.append(
            BdxMatterNodeSpec(
                node_id=nid,
                discriminator=int(item.get("discriminator", 3840 + idx)),
                passcode=int(item.get("passcode", 20202021)),
                name=str(item.get("name", f"Node_{nid}")),
                app_type=app_type,
                devices=[str(d) for d in item.get("devices", ["on-off-light:1"])],
                x=nx,
                y=ny,
                radio_range=int(item.get("radio_range", 180)),
                router_eligible=bool(item.get("router_eligible", False)),
                thread_mode=nmode,
                poll_period_ms=npoll,
                max_bdx_block_size=max_bdx_block_size,
                ota_download_path=str(item.get("ota_download_path", f"/tmp/ota_dl_{nid}.bin")),
                requestor_sai_ms=requestor_sai_ms if app_type == "ota-requestor" else None,
                requestor_sii_ms=requestor_sii_ms if app_type == "ota-requestor" else None,
            )
        )

    return BdxTopologyConfig(
        topology_id=topo_id,
        description=desc,
        config_path=config_path,
        otbr_x=otbr_x,
        otbr_y=otbr_y,
        otbr_radio_range=otbr_rr,
        otbr_name=otbr_name,
        routers=routers,
        matter_nodes=matter_nodes,
        provider_node_id=prov_node_id,
        provider_discriminator=prov_disc,
        provider_passcode=prov_pass,
        hops=hops if hop_placements else 2,
        plr=plr,
        raw_json=raw,
    )


def get_saved_state_dir(topo_cfg: BdxTopologyConfig) -> str:
    """Return a topology-scoped KVS cache directory so different hop/topology runs never collide."""
    node_ids_slug = "_".join(str(n.node_id) for n in topo_cfg.matter_nodes)
    key = sanitize_slug(f"{topo_cfg.topology_id}_r{len(topo_cfg.routers)}_n{node_ids_slug}")
    return f"/tmp/saved_commissioned_state_{key}"


def _strip_persisted_subscriptions_and_ota_state(kvs_path: str) -> None:
    """Strip persisted subscriptions and any stale OTA target state from a Matter KVS file."""
    if not os.path.isfile(kvs_path):
        return
    with open(kvs_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()
    filtered = [
        line
        for line in lines
        if not (
            line.startswith("g/su/")
            or line.startswith("g/sum=")
            or line.startswith("g/o/")
        )
    ]
    if len(filtered) != len(lines):
        with open(kvs_path, "w", encoding="utf-8") as f:
            f.writelines(filtered)


def save_bdx_commissioned_state(topo_cfg: BdxTopologyConfig) -> None:
    """Save commissioned KVS files for Requestor/sibling nodes, OTA Provider, and chip-tool."""
    state_dir = get_saved_state_dir(topo_cfg)
    os.makedirs(state_dir, exist_ok=True)
    for node in topo_cfg.matter_nodes:
        src = f"/tmp/chip_kvs_otns_{node.node_id}"
        if os.path.isfile(src):
            _strip_persisted_subscriptions_and_ota_state(src)
            dst = os.path.join(state_dir, os.path.basename(src))
            shutil.copy2(src, dst)
            _strip_persisted_subscriptions_and_ota_state(dst)
    if os.path.isfile(OTA_PROVIDER_KVS_PATH):
        shutil.copy2(OTA_PROVIDER_KVS_PATH, os.path.join(state_dir, os.path.basename(OTA_PROVIDER_KVS_PATH)))
    for ini_path in glob.glob("/tmp/chip_*.ini"):
        if os.path.isfile(ini_path):
            shutil.copy2(ini_path, os.path.join(state_dir, os.path.basename(ini_path)))
    if os.path.isdir("/tmp/chip_tool_kvs"):
        dst_kvs = os.path.join(state_dir, "chip_tool_kvs")
        os.makedirs(dst_kvs, exist_ok=True)
        for item in os.listdir("/tmp/chip_tool_kvs"):
            s = os.path.join("/tmp/chip_tool_kvs", item)
            d = os.path.join(dst_kvs, item)
            if os.path.isfile(s):
                shutil.copy2(s, d)


def restore_bdx_commissioned_state(topo_cfg: BdxTopologyConfig) -> bool:
    """Restore commissioned KVS files from topology-scoped cache if complete."""
    state_dir = get_saved_state_dir(topo_cfg)
    if not os.path.isdir(state_dir):
        node_ids_slug = "_".join(str(n.node_id) for n in topo_cfg.matter_nodes)
        pattern = f"/tmp/saved_commissioned_state_{sanitize_slug(topo_cfg.topology_id)}_r*_n{sanitize_slug(node_ids_slug)}"
        candidates = sorted(glob.glob(pattern))
        if candidates:
            state_dir = candidates[0]
        else:
            return False
    for item in os.listdir(state_dir):
        s = os.path.join(state_dir, item)
        d = os.path.join("/tmp", item)
        if os.path.isdir(s):
            os.makedirs(d, exist_ok=True)
            for sub in os.listdir(s):
                sub_s = os.path.join(s, sub)
                sub_d = os.path.join(d, sub)
                if os.path.isfile(sub_s):
                    shutil.copy2(sub_s, sub_d)
        elif os.path.isfile(s):
            if item == "chip_counters.ini" and os.path.isfile(d):
                continue
            if item.startswith("chip_kvs_otns_"):
                _strip_persisted_subscriptions_and_ota_state(s)
            shutil.copy2(s, d)

    for node in topo_cfg.matter_nodes:
        _strip_persisted_subscriptions_and_ota_state(f"/tmp/chip_kvs_otns_{node.node_id}")

    return (
        os.path.isfile(OTA_PROVIDER_KVS_PATH)
        and all(os.path.isfile(f"/tmp/chip_kvs_otns_{node.node_id}") for node in topo_cfg.matter_nodes)
    )


def create_ota_image(payload_size_bytes: int, software_version: int = 2) -> Tuple[str, str]:
    """Create a deterministic binary payload and wrap it with a valid Matter OTA image header."""
    raw_path = f"/tmp/ota_payload_{payload_size_bytes}b_v{software_version}.bin"
    ota_path = f"/tmp/ota_image_{payload_size_bytes}b_v{software_version}.ota"
    # Deterministic pseudo-random firmware pattern so blocks have realistic entropy
    pattern = bytearray(payload_size_bytes)
    for i in range(payload_size_bytes):
        pattern[i] = ((i * 73) ^ (i >> 8) ^ 0xA5) & 0xFF
    with open(raw_path, "wb") as f:
        f.write(pattern)

    if os.path.isfile(ota_path):
        os.remove(ota_path)

    cmd = [
        sys.executable,
        OTA_IMAGE_TOOL_SCRIPT,
        "create",
        "-v",
        "0xFFF1",
        "-p",
        "0x8001",
        "-vn",
        str(software_version),
        "-vs",
        f"{software_version}.0",
        "-da",
        "sha256",
        raw_path,
        ota_path,
    ]
    subprocess.run(cmd, check=True)
    return raw_path, ota_path


def write_node_wrappers(
    topo_cfg: BdxTopologyConfig,
    ota_requestor_bin: str,
    all_devices_bin: str,
) -> None:
    """Write per-node wrapper scripts in /tmp/otns_matter_wrappers/ for OTNS Matter nodes."""
    os.makedirs(WRAPPER_DIR, exist_ok=True)
    for node in topo_cfg.matter_nodes:
        kvs_path = f"/tmp/chip_kvs_otns_{node.node_id}"
        log_path = f"/tmp/matter_node_{node.node_id}.log"
        wrapper_path = os.path.join(WRAPPER_DIR, f"ot-matter-node-{node.node_id}.sh")
        if node.app_type == "ota-requestor":
            mrp_flags = ""
            if node.requestor_sai_ms is not None:
                mrp_flags += f" --active-retransmit-timeout {node.requestor_sai_ms}"
            if node.requestor_sii_ms is not None:
                mrp_flags += f" --idle-retransmit-timeout {node.requestor_sii_ms}"
            wrapper_content = f"""#!/bin/bash
norm_args=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --thread)
      shift
      ;;
    --thread-args)
      for tok in $2; do
        norm_args+=(--thread-arg "$tok")
      done
      shift 2
      ;;
    --thread-args=*)
      for tok in ${{1#--thread-args=}}; do
        norm_args+=(--thread-arg "$tok")
      done
      shift
      ;;
    *)
      norm_args+=("$1")
      shift
      ;;
  esac
done
exec -a ot-matter-node {ota_requestor_bin} \\
  --discriminator {node.discriminator} \\
  --passcode {node.passcode} \\
  --KVS "{kvs_path}" \\
  --maxBDXBlockSize {node.max_bdx_block_size} \\
  --otaDownloadPath "{node.ota_download_path}" \\
  --watchdogTimeout 60 {mrp_flags} \\
  "${{norm_args[@]}}"
"""
        else:
            device_flags = " ".join(f'--device "{dev}"' for dev in node.devices)
            wrapper_content = f"""#!/bin/bash
norm_args=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --thread)
      shift
      ;;
    --thread-args)
      for tok in $2; do
        norm_args+=(--thread-arg "$tok")
      done
      shift 2
      ;;
    --thread-args=*)
      for tok in ${{1#--thread-args=}}; do
        norm_args+=(--thread-arg "$tok")
      done
      shift
      ;;
    *)
      norm_args+=("$1")
      shift
      ;;
  esac
done
exec -a ot-matter-node {all_devices_bin} \\
  {device_flags} \\
  --discriminator {node.discriminator} \\
  --passcode {node.passcode} \\
  --groupcast \\
  --KVS "{kvs_path}" \\
  "${{norm_args[@]}}"
"""
        with open(wrapper_path, "w", encoding="utf-8") as f:
            f.write(wrapper_content)
        os.chmod(wrapper_path, 0o755)
        node.wrapper_path = wrapper_path
        node.log_path = log_path


def prepare_bdx_system(
    backbone_if: str,
    topo_cfg: BdxTopologyConfig,
    ota_requestor_bin: str,
    all_devices_bin: str,
    reuse_state: bool = False,
) -> bool:
    """Prepare container daemons, sysctls, KVS state, and per-node wrappers."""
    ensure_otns_patches()
    for proc_name in [
        "otns",
        "ot-matter-node",
        "all-devices-app",
        "chip-ota-requestor-app",
        "chip-ota-provider-app",
        "otbr-agent",
        "ot-cli-ftd",
        "ot-cli-mtd",
        "ot-rcp",
        "ot-ctl",
        "chip-tool",
    ]:
        subprocess.run(["pkill", "-9", "-x", proc_name], check=False, stderr=subprocess.DEVNULL)
    time.sleep(1.0)

    subprocess.run(
        [
            "sysctl",
            "-w",
            "net.ipv6.conf.all.disable_ipv6=0",
            "net.ipv6.conf.default.disable_ipv6=0",
            f"net.ipv6.conf.{backbone_if}.disable_ipv6=0",
            "net.ipv6.conf.all.forwarding=1",
            "net.ipv6.auto_flowlabels=0",
        ],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    subprocess.run(
        ["ip", "-6", "addr", "replace", "fd00:dead:beef:1::1/64", "dev", backbone_if],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    state_reused = False
    if reuse_state:
        state_reused = restore_bdx_commissioned_state(topo_cfg)
        if state_reused:
            print(
                f"  [reuse-state] Reusing commissioned KVS state for provider {topo_cfg.provider_node_id} "
                f"and {len(topo_cfg.matter_nodes)} Matter node(s)."
            )
        else:
            print("  [reuse-state] Saved state incomplete; performing fresh commissioning.")

    cleanup_patterns = [
        "/tmp/all_devices_node_*.log",
        "/tmp/matter_node_*.log",
        "/tmp/ota_dl_*.bin",
        "/tmp/otns_matter_wrappers",
        "/tmp/mdnsd.log",
        "/tmp/otbr_stderr.log",
        os.path.join(OT_TMP_DIR, "*.flash"),
        os.path.join(OT_TMP_DIR, "*.log"),
        OTA_PROVIDER_LOG_PATH,
        INTERACTIVE_LOG_PATH,
        CHIP_TOOL_COMMANDS_LOG_PATH,
    ]
    if not state_reused:
        cleanup_patterns.extend(["/tmp/chip_*", "/tmp/chip-*"])

    for pattern in cleanup_patterns:
        for path in glob.glob(pattern):
            try:
                if os.path.isdir(path) and not os.path.islink(path):
                    subprocess.run(["rm", "-rf", path], check=False)
                else:
                    os.remove(path)
            except OSError:
                pass

    ot_br_sh_path = os.path.join(OT_VERSIONS_DIR, "ot-br.sh")
    if os.path.isfile(ot_br_sh_path):
        with open(ot_br_sh_path, "r", encoding="utf-8") as f:
            ot_br_sh = f.read()
        old_line = 'sudo otbr-agent -s -d 7 -I ${THREAD_IF_NAME} -B ${BACKBONE_IF_NAME} "${RADIO_URL}" 1>&2 &'
        new_line = (
            'sudo stdbuf -oL otbr-agent -s -d 7 -I ${THREAD_IF_NAME} -B ${BACKBONE_IF_NAME} '
            '"${RADIO_URL}" 1>&2 2>/tmp/otbr_stderr.log &'
        )
        if old_line in ot_br_sh:
            ot_br_sh = ot_br_sh.replace(old_line, new_line)
            with open(ot_br_sh_path, "w", encoding="utf-8") as f:
                f.write(ot_br_sh)
            os.chmod(ot_br_sh_path, 0o755)

    os.makedirs("/tmp/chip_tool_kvs", exist_ok=True)
    os.makedirs(WRAPPER_DIR, exist_ok=True)
    os.makedirs("/run/dbus", exist_ok=True)
    subprocess.run(["dbus-daemon", "--system", "--nopidfile"], check=False, stderr=subprocess.DEVNULL)
    subprocess.run(["rsyslogd"], check=False, stderr=subprocess.DEVNULL)

    subprocess.run(["pkill", "-9", "-x", "mdnsd"], check=False, stderr=subprocess.DEVNULL)
    time.sleep(0.2)
    with open("/tmp/mdnsd.log", "wb") as logf:
        subprocess.Popen(["mdnsd"], stdout=logf, stderr=logf)
    time.sleep(1.0)

    write_node_wrappers(topo_cfg, ota_requestor_bin=ota_requestor_bin, all_devices_bin=all_devices_bin)
    return state_reused


def wait_for_bdx_mesh_and_srp(
    ns: OTNS,
    otbr_id: int,
    matter_nodes: List[BdxMatterNodeSpec],
    timeout_s: int = 90,
) -> bool:
    """Wait until all Matter nodes attach to Thread and register SRP services on the OTBR."""
    expected_count = len(matter_nodes)
    start_time = time.time()
    deadline = start_time + timeout_s
    last_nudge = start_time
    ext_route_registered = False
    while time.time() < deadline:
        try:
            otbr_state_lines = ns.node_cmd(otbr_id, "state")
            otbr_state = otbr_state_lines[0].strip() if otbr_state_lines else "unknown"
            if otbr_state not in ("leader", "router", "child"):
                time.sleep(1.5)
                continue
            if not ext_route_registered:
                try:
                    ns.node_cmd(otbr_id, "route add fd00:dead:beef:1::/64 s med")
                    ns.node_cmd(otbr_id, "netdata register")
                    ext_route_registered = True
                except Exception:
                    pass

            attached_count = 0
            registered_client_count = 0
            role_counts: Dict[str, int] = {}
            unregistered_nodes: List[BdxMatterNodeSpec] = []
            for node in matter_nodes:
                assert node.otns_id is not None
                state_lines = ns.node_cmd(node.otns_id, "state")
                state = state_lines[0].strip() if state_lines else "unknown"
                role_counts[state] = role_counts.get(state, 0) + 1
                if state in ("leader", "router", "child"):
                    attached_count += 1
                    client_svc = ns.node_cmd(node.otns_id, "srp client service")
                    if any(("state:Registered" in line or "state:Refreshing" in line) for line in client_svc):
                        registered_client_count += 1
                    else:
                        unregistered_nodes.append(node)

            srp_services = ns.node_cmd(otbr_id, "srp server service")
            omr_prefix = ns.node_cmd(otbr_id, "br omrprefix")
            configure_border_router_routes(omr_prefix)

            active_matter_srp = [
                line
                for line in srp_services
                if ("_matterc._udp" in line or "_matter._tcp" in line)
                and ("deleted: false" in line or "deleted:" not in line)
            ]
            print(
                f"  [mesh] otbr={otbr_state} attached={attached_count}/{expected_count} "
                f"roles={role_counts} client_srp={registered_client_count}/{expected_count} "
                f"server_srp={len(active_matter_srp)}/{expected_count}"
            )
            if (
                attached_count == expected_count
                and registered_client_count == expected_count
                and len(active_matter_srp) >= expected_count
            ):
                return True

            now = time.time()
            if unregistered_nodes and (now - last_nudge) >= 25.0:
                last_nudge = now
                for node in unregistered_nodes:
                    assert node.otns_id is not None
                    try:
                        ns.node_cmd(node.otns_id, "srp client stop")
                        ns.node_cmd(node.otns_id, "srp client autostart enable")
                    except Exception:
                        pass
        except Exception as exc:
            print(f"  [mesh] polling exception: {exc}")
        time.sleep(1.5)
    return False


def verify_topology_routing(
    ns: OTNS,
    otbr_id: int,
    router_ids: List[int],
    topo_cfg: BdxTopologyConfig,
) -> Dict[str, Any]:
    """Query live RLOC16s, parents, modes, and router tables on all OTNS nodes and verify expected hops."""
    otbr_rloc = ns.get_rloc16(otbr_id)
    otbr_state_lines = ns.node_cmd(otbr_id, "state")
    otbr_state = otbr_state_lines[0].strip() if otbr_state_lines else "unknown"
    otbr_ext_lines = ns.node_cmd(otbr_id, "extaddr")
    otbr_ext = otbr_ext_lines[0].strip() if otbr_ext_lines else ""
    otbr_rt = ns.node_cmd(otbr_id, "router table")

    routers_info: List[Dict[str, Any]] = []
    for idx, r_id in enumerate(router_ids):
        r_rloc = ns.get_rloc16(r_id)
        r_state_lines = ns.node_cmd(r_id, "state")
        r_state = r_state_lines[0].strip() if r_state_lines else "unknown"
        r_ext_lines = ns.node_cmd(r_id, "extaddr")
        r_ext = r_ext_lines[0].strip() if r_ext_lines else ""
        r_rt = ns.node_cmd(r_id, "router table")
        routers_info.append(
            {
                "otns_id": r_id,
                "index": idx + 1,
                "name": topo_cfg.routers[idx].get("name", f"Router_{idx+1}"),
                "rloc16": f"0x{r_rloc:04x}",
                "rloc16_int": r_rloc,
                "extaddr": r_ext,
                "state": r_state,
                "x": topo_cfg.routers[idx]["x"],
                "y": topo_cfg.routers[idx]["y"],
                "radio_range": topo_cfg.routers[idx]["radio_range"],
                "router_table": r_rt,
            }
        )

    matter_info: List[Dict[str, Any]] = []
    for node in topo_cfg.matter_nodes:
        assert node.otns_id is not None
        m_rloc = ns.get_rloc16(node.otns_id)
        m_state_lines = ns.node_cmd(node.otns_id, "state")
        m_state = m_state_lines[0].strip() if m_state_lines else "unknown"
        m_ext_lines = ns.node_cmd(node.otns_id, "extaddr")
        m_ext = m_ext_lines[0].strip() if m_ext_lines else ""
        m_mode_lines = ns.node_cmd(node.otns_id, "mode")
        m_mode = m_mode_lines[0].strip() if m_mode_lines else "unknown"
        parent_lines = ns.node_cmd(node.otns_id, "parent")
        parent_rloc_hex = None
        for line in parent_lines:
            m = re.search(r"Rloc:\s*([0-9a-fA-F]+)", line, re.IGNORECASE)
            if m:
                parent_rloc_hex = f"0x{int(m.group(1), 16):04x}"
                break
        if parent_rloc_hex is None and (m_rloc & 0x03FF) != 0:
            parent_rloc_hex = f"0x{(m_rloc & 0xFC00):04x}"

        # Compute hop count from OTBR:
        # If child of OTBR -> 1 hop; if child of Router k -> k + 1 hops
        expected_parent_rloc = f"0x{otbr_rloc:04x}"
        verified_hops = 1
        for r_info in routers_info:
            if parent_rloc_hex == r_info["rloc16"] or (m_rloc & 0xFC00) == (r_info["rloc16_int"] & 0xFC00):
                verified_hops = r_info["index"] + 1
                expected_parent_rloc = r_info["rloc16"]
                break

        matter_info.append(
            {
                "otns_id": node.otns_id,
                "matter_node_id": node.node_id,
                "name": node.name,
                "app_type": node.app_type,
                "thread_mode_configured": node.thread_mode,
                "thread_mode_actual": m_mode,
                "poll_period_ms": node.poll_period_ms if node.thread_mode == "sed" else None,
                "state": m_state,
                "rloc16": f"0x{m_rloc:04x}",
                "rloc16_int": m_rloc,
                "extaddr": m_ext,
                "parent_rloc16": parent_rloc_hex,
                "verified_hops_from_otbr": verified_hops,
                "x": node.x,
                "y": node.y,
                "radio_range": node.radio_range,
            }
        )

    summary = {
        "topology_id": topo_cfg.topology_id,
        "configured_hops": topo_cfg.hops,
        "plr": topo_cfg.plr,
        "otbr": {
            "otns_id": otbr_id,
            "name": topo_cfg.otbr_name,
            "rloc16": f"0x{otbr_rloc:04x}",
            "rloc16_int": otbr_rloc,
            "extaddr": otbr_ext,
            "state": otbr_state,
            "x": topo_cfg.otbr_x,
            "y": topo_cfg.otbr_y,
            "radio_range": topo_cfg.otbr_radio_range,
            "router_table": otbr_rt,
        },
        "routers": routers_info,
        "matter_nodes": matter_info,
    }
    print("\n=== Programmatic Topology & RLOC16 Verification ===")
    print(f"  OTBR (otns_id={otbr_id}): state={otbr_state}, RLOC16=0x{otbr_rloc:04x}")
    for r_info in routers_info:
        print(
            f"  Router {r_info['index']} ({r_info['name']}, otns_id={r_info['otns_id']}): "
            f"state={r_info['state']}, RLOC16={r_info['rloc16']}"
        )
    for m_info in matter_info:
        print(
            f"  Matter Node {m_info['matter_node_id']} ({m_info['name']}, otns_id={m_info['otns_id']}): "
            f"state={m_info['state']}, mode={m_info['thread_mode_actual']}, RLOC16={m_info['rloc16']}, "
            f"parent={m_info['parent_rloc16']}, hops_from_otbr={m_info['verified_hops_from_otbr']}"
        )
    return summary


class OtaProviderProcess:
    """Manages the Linux `chip-ota-provider-app` process on the container host."""

    def __init__(
        self,
        provider_bin: str,
        ota_image_path: str,
        max_bdx_block_size: int,
        discriminator: int = 2222,
        passcode: int = 20202021,
        kvs_path: str = OTA_PROVIDER_KVS_PATH,
        log_path: str = OTA_PROVIDER_LOG_PATH,
    ) -> None:
        self.provider_bin = provider_bin
        self.ota_image_path = ota_image_path
        self.max_bdx_block_size = max_bdx_block_size
        self.discriminator = discriminator
        self.passcode = passcode
        self.kvs_path = kvs_path
        self.log_path = log_path
        self.log_file = open(self.log_path, "w", encoding="utf-8", buffering=1)
        self.lines: List[str] = []
        self.cv = threading.Condition()
        cmd = [
            "stdbuf",
            "-oL",
            "-eL",
            self.provider_bin,
            "--filepath",
            self.ota_image_path,
            "--maxBDXBlockSize",
            str(self.max_bdx_block_size),
            "--discriminator",
            str(self.discriminator),
            "--passcode",
            str(self.passcode),
            "--KVS",
            self.kvs_path,
        ]
        print(f"  [ota-provider] Launching: {' '.join(cmd)}")
        self.proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        self.reader_thread = threading.Thread(target=self._reader_loop, daemon=True)
        self.reader_thread.start()

    def _reader_loop(self) -> None:
        assert self.proc.stdout is not None
        for raw_line in self.proc.stdout:
            line = raw_line.rstrip("\n")
            if not line:
                continue
            self.log_file.write(line + "\n")
            with self.cv:
                self.lines.append(line)
                self.cv.notify_all()

    def wait_for_ready(self, timeout_s: float = 15.0) -> None:
        deadline = time.time() + timeout_s
        with self.cv:
            while time.time() < deadline:
                if any("CHIP task running" in l or "Server Listening" in l for l in self.lines):
                    return
                self.cv.wait(timeout=0.2)
        raise RuntimeError("Timed out waiting for chip-ota-provider-app to initialize")

    def stop(self) -> None:
        try:
            if self.proc.poll() is None:
                self.proc.terminate()
                try:
                    self.proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.proc.kill()
        finally:
            self.log_file.close()


def commission_topology_and_provider(
    chip_tool_bin: str,
    provider_bin: str,
    ota_image_path: str,
    topo_cfg: BdxTopologyConfig,
    state_reused: bool,
) -> None:
    """Commission OTA Provider and all OTNS Matter nodes onto the same fabric and configure Provider ACL."""
    if state_reused:
        print("\n=== Skipping Commissioning (reused persisted fabric & ACL state) ===")
        return

    prov = OtaProviderProcess(
        provider_bin=provider_bin,
        ota_image_path=ota_image_path,
        max_bdx_block_size=1024,
        discriminator=topo_cfg.provider_discriminator,
        passcode=topo_cfg.provider_passcode,
    )
    try:
        prov.wait_for_ready(timeout_s=15.0)
        time.sleep(1.0)
        print(f"\n=== Commissioning OTA Provider (node_id={topo_cfg.provider_node_id}) ===")
        run_chip_tool(
            chip_tool_bin,
            [
                "pairing",
                "onnetwork-long",
                str(topo_cfg.provider_node_id),
                str(topo_cfg.provider_passcode),
                str(topo_cfg.provider_discriminator),
            ],
            timeout_s=60,
            retries=3,
        )
        acl_json = json.dumps(
            [
                {
                    "fabricIndex": 1,
                    "privilege": 5,
                    "authMode": 2,
                    "subjects": [112233],
                    "targets": None,
                },
                {
                    "fabricIndex": 1,
                    "privilege": 3,
                    "authMode": 2,
                    "subjects": None,
                    "targets": [{"cluster": 41, "endpoint": 0, "deviceType": None}],
                },
            ]
        )
        print("=== Writing AccessControl ACL on OTA Provider (granting Operate on Cluster 0x0029) ===")
        run_chip_tool(
            chip_tool_bin,
            [
                "accesscontrol",
                "write",
                "acl",
                acl_json,
                str(topo_cfg.provider_node_id),
                "0",
            ],
            timeout_s=30,
            retries=2,
        )
    finally:
        prov.stop()

    for idx, node in enumerate(topo_cfg.matter_nodes, start=1):
        print(
            f"\n=== Commissioning Matter Node {idx}/{len(topo_cfg.matter_nodes)}: "
            f"node_id={node.node_id} ({node.name}, app={node.app_type}) ==="
        )
        run_chip_tool(
            chip_tool_bin,
            [
                "pairing",
                "onnetwork-long",
                str(node.node_id),
                str(node.passcode),
                str(node.discriminator),
            ],
            timeout_s=90,
            retries=3,
        )

    save_bdx_commissioned_state(topo_cfg)


def slice_pcap_preserving_advertisements(
    full_pcap_path: str,
    start_byte_offset: int,
    dst_pcap_path: str,
) -> int:
    """Write a valid PCAP file containing all early Thread Advertisement frames plus all frames from start_byte_offset."""
    if not os.path.isfile(full_pcap_path):
        return 0
    with open(full_pcap_path, "rb") as f:
        data = f.read()
    if len(data) < 24:
        return 0

    out = bytearray(data[:24])
    off = 24
    pkt_count = 0
    while off + 16 <= len(data):
        pkt_start = off
        _ts_sec, _ts_usec, incl_len, _orig_len = struct.unpack("<IIII", data[off : off + 16])
        off += 16
        frame = data[off : off + incl_len]
        off += incl_len
        if pkt_start >= start_byte_offset:
            out.extend(data[pkt_start:off])
            pkt_count += 1
        elif len(frame) >= 15:
            # Keep broadcast MLE advertisement frames prior to start_byte_offset so short_to_ext discovery works
            dst_short = struct.unpack("<H", frame[5:7])[0] if len(frame) >= 9 else 0
            if dst_short == 0xFFFF:
                out.extend(data[pkt_start:off])

    with open(dst_pcap_path, "wb") as f:
        f.write(out)
    return pkt_count


def slice_node_log_preserving_header(
    src_log_path: str,
    start_line_idx: int,
    dst_log_path: str,
) -> None:
    """Copy initial node attachment/RLOC16 header lines (first 80 lines) plus all lines from start_line_idx onward."""
    if not os.path.isfile(src_log_path):
        return
    with open(src_log_path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()
    if start_line_idx <= 80:
        selected = lines
    else:
        header = lines[:80]
        body = lines[start_line_idx:]
        selected = header + ["--- [BDX BENCHMARK WINDOW START] ---\n"] + body
    with open(dst_log_path, "w", encoding="utf-8") as f:
        f.writelines(selected)


def archive_bdx_transfer_run(
    topo_cfg: BdxTopologyConfig,
    verified_topo: Dict[str, Any],
    suite_name: str,
    block_size: int,
    image_size_bytes: int,
    contend_traffic: bool,
    state_reused: bool,
    started_epoch: float,
    completed_epoch: float,
    transfer_metrics: Dict[str, Any],
    pcap_start_offset: int,
    node_log_start_lines: Dict[int, int],
    log_dir: str,
    run_tag: Optional[str] = None,
) -> str:
    """Archive all logs, sliced PCAP, stats.csv, and manifest.json for a single BDX transfer run."""
    try:
        local_tz = ZoneInfo(os.environ.get("TZ", "America/Los_Angeles"))
        ts_str = datetime.fromtimestamp(started_epoch, tz=local_tz).strftime("%Y%m%d_%H%M%S_%f")[:19]
    except Exception:
        ts_str = datetime.fromtimestamp(started_epoch).strftime("%Y%m%d_%H%M%S_%f")[:19]

    req_node = next((n for n in topo_cfg.matter_nodes if n.app_type == "ota-requestor"), topo_cfg.matter_nodes[0])
    plr_pct = int(round(topo_cfg.plr * 100))
    sai_str = f"_sai{req_node.requestor_sai_ms}" if req_node.requestor_sai_ms else ""
    contend_str = "_contend" if contend_traffic else ""
    tag_str = f"_{sanitize_slug(run_tag)}" if run_tag else ""

    run_id = (
        f"{ts_str}_{sanitize_slug(suite_name)}_{sanitize_slug(topo_cfg.topology_id)}_"
        f"{topo_cfg.hops}hop_{req_node.thread_mode}_plr{plr_pct}_blk{block_size}{sai_str}{contend_str}{tag_str}"
    )
    run_dir = os.path.join(log_dir, run_id)
    nodes_dir = os.path.join(run_dir, "nodes")
    controller_dir = os.path.join(run_dir, "controller")
    daemons_dir = os.path.join(run_dir, "daemons")
    pcap_dir = os.path.join(run_dir, "pcap")

    for d in (nodes_dir, controller_dir, daemons_dir, pcap_dir):
        os.makedirs(d, exist_ok=True)

    # 1. Save config.json
    shutil.copy2(topo_cfg.config_path, os.path.join(run_dir, "config.json"))

    # 2. Copy/slice node logs and populate node_manifest with live verified RLOC16s
    node_manifest: List[Dict[str, Any]] = []
    otbr_info = verified_topo["otbr"]
    otbr_src = os.path.join(OT_TMP_DIR, "0_1.log")
    otbr_dst = os.path.join(nodes_dir, "0_1.log")
    if os.path.isfile(otbr_src):
        slice_node_log_preserving_header(otbr_src, node_log_start_lines.get(1, 0), otbr_dst)
        link_p = os.path.join(nodes_dir, "otns_01_otbr.log")
        if not os.path.exists(link_p):
            os.symlink("0_1.log", link_p)
    node_manifest.append(
        {
            "otns_id": 1,
            "role": "otbr",
            "name": otbr_info["name"],
            "matter_node_id": None,
            "rloc16": otbr_info["rloc16"],
            "extaddr": otbr_info.get("extaddr", ""),
            "x": otbr_info["x"],
            "y": otbr_info["y"],
            "radio_range": otbr_info["radio_range"],
            "raw_log": "nodes/0_1.log",
            "named_log": "nodes/otns_01_otbr.log",
        }
    )

    for r_info in verified_topo["routers"]:
        oid = r_info["otns_id"]
        raw_name = f"0_{oid}.log"
        friendly = f"otns_{oid:02d}_router_{r_info['index']:02d}.log"
        src_p = os.path.join(OT_TMP_DIR, raw_name)
        dst_p = os.path.join(nodes_dir, raw_name)
        if os.path.isfile(src_p):
            slice_node_log_preserving_header(src_p, node_log_start_lines.get(oid, 0), dst_p)
            link_p = os.path.join(nodes_dir, friendly)
            if not os.path.exists(link_p):
                os.symlink(raw_name, link_p)
        node_manifest.append(
            {
                "otns_id": oid,
                "role": f"router_{r_info['index']}",
                "name": r_info["name"],
                "matter_node_id": None,
                "rloc16": r_info["rloc16"],
                "extaddr": r_info.get("extaddr", ""),
                "x": r_info["x"],
                "y": r_info["y"],
                "radio_range": r_info["radio_range"],
                "raw_log": f"nodes/{raw_name}",
                "named_log": f"nodes/{friendly}",
            }
        )

    for m_info in verified_topo["matter_nodes"]:
        oid = m_info["otns_id"]
        mnid = m_info["matter_node_id"]
        raw_name = f"0_{oid}.log"
        friendly = f"otns_{oid:02d}_matter_{mnid}.log"
        src_p = os.path.join(OT_TMP_DIR, raw_name)
        dst_p = os.path.join(nodes_dir, raw_name)
        if os.path.isfile(src_p):
            slice_node_log_preserving_header(src_p, node_log_start_lines.get(oid, 0), dst_p)
            link_p = os.path.join(nodes_dir, friendly)
            if not os.path.exists(link_p):
                os.symlink(raw_name, link_p)
        node_manifest.append(
            {
                "otns_id": oid,
                "role": f"matter_{m_info['thread_mode_configured']}",
                "name": m_info["name"],
                "app_type": m_info["app_type"],
                "matter_node_id": mnid,
                "rloc16": m_info["rloc16"],
                "extaddr": m_info.get("extaddr", ""),
                "parent_rloc16": m_info["parent_rloc16"],
                "verified_hops_from_otbr": m_info["verified_hops_from_otbr"],
                "x": m_info["x"],
                "y": m_info["y"],
                "radio_range": m_info["radio_range"],
                "raw_log": f"nodes/{raw_name}",
                "named_log": f"nodes/{friendly}",
            }
        )

    # 3. Copy controller & OTA provider logs
    controller_files: List[str] = []
    if os.path.isfile(OTA_PROVIDER_LOG_PATH):
        shutil.copy2(OTA_PROVIDER_LOG_PATH, os.path.join(controller_dir, "ota_provider.log"))
        controller_files.append("controller/ota_provider.log")
    if os.path.isfile(INTERACTIVE_LOG_PATH):
        shutil.copy2(INTERACTIVE_LOG_PATH, os.path.join(controller_dir, "chip_tool_interactive.log"))
        controller_files.append("controller/chip_tool_interactive.log")
    if os.path.isfile(CHIP_TOOL_COMMANDS_LOG_PATH):
        shutil.copy2(CHIP_TOOL_COMMANDS_LOG_PATH, os.path.join(controller_dir, "chip_tool_commands.log"))
        controller_files.append("controller/chip_tool_commands.log")

    # 4. Copy daemon logs
    daemon_files: List[str] = []
    for src_daemon, dst_name in [
        (os.path.join(OT_TMP_DIR, "0.log"), "otns_dispatcher.log"),
        ("/tmp/otns.log", "otns_cli.log"),
        ("/tmp/otbr_stderr.log", "otbr_agent_stderr.log"),
        ("/tmp/mdnsd.log", "mdnsd.log"),
    ]:
        if os.path.isfile(src_daemon):
            shutil.copy2(src_daemon, os.path.join(daemons_dir, dst_name))
            daemon_files.append(f"daemons/{dst_name}")

    # 5. Slice current.pcap for this transfer window
    pcap_files: List[str] = []
    src_pcap = os.path.join(OT_VERSIONS_DIR, "current.pcap")
    if not os.path.isfile(src_pcap):
        src_pcap = os.path.join(OT_TMP_DIR, "current.pcap")
    dst_pcap = os.path.join(pcap_dir, "current.pcap")
    if os.path.isfile(src_pcap):
        slice_pcap_preserving_advertisements(src_pcap, pcap_start_offset, dst_pcap)
        pcap_files.append("pcap/current.pcap")

    src_stats_csv = os.path.join(OT_TMP_DIR, "0_stats.csv")
    stats_csv_rel = None
    if os.path.isfile(src_stats_csv):
        shutil.copy2(src_stats_csv, os.path.join(run_dir, "stats.csv"))
        stats_csv_rel = "stats.csv"

    manifest = {
        "run_id": run_id,
        "suite": suite_name,
        "topology_id": topo_cfg.topology_id,
        "hops": topo_cfg.hops,
        "thread_mode": req_node.thread_mode,
        "poll_period_ms": req_node.poll_period_ms if req_node.thread_mode == "sed" else None,
        "plr": topo_cfg.plr,
        "block_size": block_size,
        "image_size_bytes": image_size_bytes,
        "requestor_sai_ms": req_node.requestor_sai_ms if req_node.requestor_sai_ms is not None else 300,
        "requestor_sii_ms": req_node.requestor_sii_ms if req_node.requestor_sii_ms is not None else 500,
        "contend_traffic": contend_traffic,
        "state_reused": state_reused,
        "started_at_utc": datetime.fromtimestamp(started_epoch, tz=timezone.utc).isoformat(),
        "completed_at_utc": datetime.fromtimestamp(completed_epoch, tz=timezone.utc).isoformat(),
        "duration_sec": round(completed_epoch - started_epoch, 3),
        "transfer_metrics": transfer_metrics,
        "verified_topology": verified_topo,
        "files": {
            "config": "config.json",
            "stats_csv": stats_csv_rel,
            "controller": controller_files,
            "daemons": daemon_files,
            "pcap": pcap_files,
        },
        "nodes": node_manifest,
    }
    with open(os.path.join(run_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    latest_link = os.path.join(log_dir, "latest")
    try:
        if os.path.islink(latest_link) or os.path.exists(latest_link):
            os.unlink(latest_link)
        os.symlink(run_id, latest_link)
    except OSError:
        pass

    try:
        with open(os.path.join(log_dir, "runs_index.jsonl"), "a", encoding="utf-8") as idx_f:
            idx_f.write(
                json.dumps(
                    {
                        "run_id": run_id,
                        "suite": suite_name,
                        "topology_id": topo_cfg.topology_id,
                        "hops": topo_cfg.hops,
                        "thread_mode": req_node.thread_mode,
                        "plr": topo_cfg.plr,
                        "block_size": block_size,
                        "image_size_bytes": image_size_bytes,
                        "contend_traffic": contend_traffic,
                        "completed": transfer_metrics.get("completed", False),
                        "transfer_duration_s": transfer_metrics.get("transfer_duration_s"),
                        "path": run_dir,
                    }
                )
                + "\n"
            )
    except OSError:
        pass

    try:
        ws_stat = os.stat(REPO_ROOT)
        ws_uid, ws_gid = ws_stat.st_uid, ws_stat.st_gid
        for target in (log_dir, latest_link, os.path.join(log_dir, "runs_index.jsonl")):
            if os.path.lexists(target):
                os.lchown(target, ws_uid, ws_gid)
        for root, dirs, files in os.walk(run_dir):
            os.lchown(root, ws_uid, ws_gid)
            for dname in dirs:
                os.lchown(os.path.join(root, dname), ws_uid, ws_gid)
            for fname in files:
                os.lchown(os.path.join(root, fname), ws_uid, ws_gid)
    except OSError:
        pass

    print(f"  [archive] Saved run to {run_dir}")
    return run_dir


def execute_single_bdx_transfer(
    ns: OTNS,
    otbr_id: int,
    topo_cfg: BdxTopologyConfig,
    verified_topo: Dict[str, Any],
    chip_tool_bin: str,
    provider_bin: str,
    ota_image_path: str,
    image_size_bytes: int,
    block_size: int,
    suite_name: str,
    state_reused: bool,
    log_dir: str,
    contend_traffic: bool = False,
    timeout_s: float = 180.0,
    run_tag: Optional[str] = None,
) -> Tuple[str, Dict[str, Any]]:
    """Run one end-to-end BDX OTA transfer on the active OTNS mesh and archive its artifacts."""
    req_node = next((n for n in topo_cfg.matter_nodes if n.app_type == "ota-requestor"), topo_cfg.matter_nodes[0])
    assert req_node.otns_id is not None
    req_log_path = os.path.join(OT_TMP_DIR, f"0_{req_node.otns_id}.log")

    # Ensure OMR route is up on wpan0
    omr_prefix = ns.node_cmd(otbr_id, "br omrprefix")
    configure_border_router_routes(omr_prefix)

    # Record current PCAP byte offset and per-node log line indices before starting the transfer
    src_pcap = os.path.join(OT_VERSIONS_DIR, "current.pcap")
    if not os.path.isfile(src_pcap):
        src_pcap = os.path.join(OT_TMP_DIR, "current.pcap")
    pcap_start_offset = os.path.getsize(src_pcap) if os.path.isfile(src_pcap) else 24

    node_log_start_lines: Dict[int, int] = {}
    for oid in [1] + [r["otns_id"] for r in verified_topo["routers"]] + [m["otns_id"] for m in verified_topo["matter_nodes"]]:
        lp = os.path.join(OT_TMP_DIR, f"0_{oid}.log")
        if os.path.isfile(lp):
            with open(lp, "r", encoding="utf-8", errors="replace") as f:
                node_log_start_lines[oid] = sum(1 for _ in f)
        else:
            node_log_start_lines[oid] = 0

    if os.path.isfile(req_node.ota_download_path):
        try:
            os.remove(req_node.ota_download_path)
        except OSError:
            pass

    # Start OTA Provider with the target --maxBDXBlockSize
    prov = OtaProviderProcess(
        provider_bin=provider_bin,
        ota_image_path=ota_image_path,
        max_bdx_block_size=block_size,
        discriminator=topo_cfg.provider_discriminator,
        passcode=topo_cfg.provider_passcode,
    )
    ict_contend: Optional[InteractiveChipTool] = None
    contend_stop = threading.Event()
    contend_thread: Optional[threading.Thread] = None

    started_epoch = time.time()
    completed = False
    failure_reason: Optional[str] = None
    t_receive_init: Optional[float] = None
    t_ack_eof: Optional[float] = None
    blocks_sent = 0
    mrp_retries_provider = 0

    try:
        prov.wait_for_ready(timeout_s=15.0)
        time.sleep(0.5)

        # If Scenario 2 concurrent traffic is enabled, start background SubscribeReq / OnOff / ReportData traffic
        sibling_nodes = [n for n in topo_cfg.matter_nodes if n.app_type != "ota-requestor"]
        if contend_traffic and sibling_nodes:
            print(
                f"  [contend] Starting concurrent SubscribeReq & ReportData traffic on sibling nodes "
                f"{[n.node_id for n in sibling_nodes]} sharing router A1..."
            )
            ict_contend = InteractiveChipTool(chip_tool_bin)
            ict_contend.wait_for_ready(timeout_s=15.0)
            for snode in sibling_nodes:
                try:
                    ict_contend.run_and_wait(
                        f"onoff subscribe on-off 1 10 {snode.node_id} 1 --keepSubscriptions false",
                        rf"Subscription established.*Peer = \d+:0*{snode.node_id:x}\b",
                        timeout_s=30.0,
                        retries=2,
                    )
                except Exception as exc:
                    print(f"  [contend] Warning: initial subscribe to {snode.node_id} returned: {exc}")

            def _contend_worker() -> None:
                step = 0
                while not contend_stop.is_set():
                    snode = sibling_nodes[step % len(sibling_nodes)]
                    cmd = "toggle" if step % 3 != 0 else "subscribe on-off 1 5"
                    try:
                        if cmd == "toggle":
                            ict_contend.send_command(f"onoff toggle {snode.node_id} 1")
                        else:
                            ict_contend.send_command(
                                f"any subscribe-by-id 0xFFFFFFFF 0xFFFFFFFF 1 10 {snode.node_id} 0xFFFF --keepSubscriptions true"
                            )
                    except Exception:
                        pass
                    step += 1
                    contend_stop.wait(0.35)

            contend_thread = threading.Thread(target=_contend_worker, daemon=True)
            contend_thread.start()

        print(
            f"\n>>> [BDX Transfer] suite={suite_name} hops={topo_cfg.hops} mode={req_node.thread_mode} "
            f"plr={topo_cfg.plr*100:.0f}% block_size={block_size}B image={image_size_bytes}B "
            f"sai={req_node.requestor_sai_ms or 300}ms contend={contend_traffic}"
        )

        # Trigger AnnounceOTAProvider
        run_chip_tool(
            chip_tool_bin,
            [
                "otasoftwareupdaterequestor",
                "announce-otaprovider",
                str(topo_cfg.provider_node_id),
                "0",
                "0",
                "0",
                str(req_node.node_id),
                "0",
            ],
            timeout_s=45,
            retries=3,
        )

        t_announce = time.time()
        deadline = t_announce + timeout_s
        prov_cursor = 0
        last_progress_log = t_announce

        while time.time() < deadline:
            with prov.cv:
                new_prov_lines = prov.lines[prov_cursor:]
                prov_cursor = len(prov.lines)

            for line in new_prov_lines:
                if "BDX:ReceiveInit" in line and t_receive_init is None:
                    m_ts = re.search(r"\[(\d+\.\d+)\]", line)
                    t_receive_init = float(m_ts.group(1)) if m_ts else time.time()
                elif ("Type 0002:11 (BDX:Block)" in line or "Type 0002:12 (BDX:BlockEOF)" in line) and "<<<" in line:
                    blocks_sent += 1
                elif re.search(r"<<\d+\s+\[E:.*Msg Retransmission to", line) or "Retransmitting MessageCounter:" in line:
                    mrp_retries_provider += 1
                elif ("Type 0002:14 (BDX:BlockAckEOF)" in line or "Transfer completed, got AckEOF" in line) and not completed:
                    m_ts = re.search(r"\[(\d+\.\d+)\]", line)
                    t_ack_eof = float(m_ts.group(1)) if m_ts else time.time()
                    completed = True
                elif "TransferSession error" in line or ("Msg Retransmission" in line and "failure" in line):
                    failure_reason = line.strip()

            if completed:
                # Wait briefly for Requestor to flush "OTA image downloaded to ..." and Reset()
                time.sleep(1.0)
                break

            if failure_reason is not None:
                time.sleep(1.0)
                break

            now = time.time()
            if now - last_progress_log >= 5.0:
                last_progress_log = now
                print(
                    f"  [bdx-progress] elapsed={now - t_announce:.1f}s blocks_tx={blocks_sent} "
                    f"prov_mrp_retries={mrp_retries_provider}"
                )
            time.sleep(0.25)

        if not completed and failure_reason is None:
            failure_reason = f"Timed out after {timeout_s:.1f}s (blocks_sent={blocks_sent})"

    finally:
        if contend_thread is not None:
            contend_stop.set()
            contend_thread.join(timeout=2.0)
        if ict_contend is not None:
            ict_contend.close()
        prov.stop()

    completed_epoch = time.time()
    if t_receive_init is not None and t_ack_eof is not None and t_ack_eof >= t_receive_init:
        transfer_duration_s = round(t_ack_eof - t_receive_init, 4)
    else:
        transfer_duration_s = round(completed_epoch - started_epoch, 4)

    dl_size = os.path.getsize(req_node.ota_download_path) if os.path.isfile(req_node.ota_download_path) else 0
    metrics = {
        "completed": completed,
        "failure_reason": failure_reason,
        "transfer_duration_s": transfer_duration_s,
        "blocks_sent_provider": blocks_sent,
        "mrp_retries_provider": mrp_retries_provider,
        "downloaded_bytes": dl_size,
    }
    print(
        f"  [bdx-result] completed={completed} duration={transfer_duration_s:.3f}s "
        f"blocks_sent={blocks_sent} mrp_retries={mrp_retries_provider} dl_bytes={dl_size}"
    )

    run_dir = archive_bdx_transfer_run(
        topo_cfg=topo_cfg,
        verified_topo=verified_topo,
        suite_name=suite_name,
        block_size=block_size,
        image_size_bytes=image_size_bytes,
        contend_traffic=contend_traffic,
        state_reused=state_reused,
        started_epoch=started_epoch,
        completed_epoch=completed_epoch,
        transfer_metrics=metrics,
        pcap_start_offset=pcap_start_offset,
        node_log_start_lines=node_log_start_lines,
        log_dir=log_dir,
        run_tag=run_tag,
    )
    return run_dir, metrics


def run_otns_session_transfers(
    config_path: str,
    suite_name: str,
    hops: int,
    mode: str,
    poll_period_ms: int,
    plr: float,
    block_sizes: List[int],
    image_size_bytes: int,
    ota_requestor_bin: str,
    ota_provider_bin: str,
    all_devices_bin: str,
    chip_tool_bin: str,
    log_dir: str,
    reuse_state: bool = True,
    requestor_sai_ms: Optional[int] = None,
    requestor_sii_ms: Optional[int] = None,
    contend_traffic: bool = False,
    timeout_s: float = 180.0,
    run_tag: Optional[str] = None,
) -> List[Tuple[str, Dict[str, Any]]]:
    """Start an OTNS session for (config, hops, mode, plr) and execute BDX transfers for each block_size."""
    max_req_block = max(1024, max(block_sizes))
    topo_cfg = load_bdx_topology(
        config_path=config_path,
        hops=hops,
        mode=mode,
        poll_period_ms=poll_period_ms,
        plr=plr,
        max_bdx_block_size=max_req_block,
        requestor_sai_ms=requestor_sai_ms,
        requestor_sii_ms=requestor_sii_ms,
    )
    backbone_if = detect_backbone_interface()
    os.chdir(OT_VERSIONS_DIR)
    state_reused = prepare_bdx_system(
        backbone_if=backbone_if,
        topo_cfg=topo_cfg,
        ota_requestor_bin=ota_requestor_bin,
        all_devices_bin=all_devices_bin,
        reuse_state=reuse_state,
    )

    ns = OTNS(
        otns_args=[
            "-realtime",
            "-logfile",
            "info",
            "-listen",
            "0.0.0.0:9000",
            "-web=false",
            "-otbr-backbone-if",
            backbone_if,
        ]
    )
    results: List[Tuple[str, Dict[str, Any]]] = []
    try:
        ns.radiomodel = "MIDisc"
        otbr_id = ns.add(
            "otbr",
            x=topo_cfg.otbr_x,
            y=topo_cfg.otbr_y,
            radio_range=topo_cfg.otbr_radio_range,
        )
        ns.autogo = True
        if not wait_for_otbr_ready(ns, otbr_id=otbr_id, timeout_s=40):
            raise RuntimeError("Timed out waiting for OTBR to become ready")

        router_ids: List[int] = []
        for rtr in topo_cfg.routers:
            r_id = ns.add(
                "router",
                x=int(rtr["x"]),
                y=int(rtr["y"]),
                radio_range=int(rtr.get("radio_range", 180)),
            )
            ns.node_cmd(r_id, "routereligible enable")
            ns.node_cmd(r_id, "routerselectionjitter 1")
            router_ids.append(r_id)
            time.sleep(1.0)

        if router_ids and not wait_for_routers_ready(ns, router_ids, timeout_s=35):
            raise RuntimeError("Timed out waiting for Thread routers to attach")

        for node in topo_cfg.matter_nodes:
            assert node.wrapper_path is not None
            node.otns_id = ns.add(
                "matter",
                x=node.x,
                y=node.y,
                radio_range=node.radio_range,
                executable=node.wrapper_path,
            )
            if not node.router_eligible:
                ns.node_cmd(node.otns_id, "routereligible disable")
            if node.thread_mode == "sed":
                # Configure Sleepy End Device (RxOnWhenIdle=0) and poll period
                ns.node_cmd(node.otns_id, "mode n")
                ns.node_cmd(node.otns_id, f"pollperiod {node.poll_period_ms}")
            time.sleep(1.0)

        if not wait_for_bdx_mesh_and_srp(ns, otbr_id=otbr_id, matter_nodes=topo_cfg.matter_nodes, timeout_s=80):
            raise RuntimeError("Timed out waiting for Matter nodes to attach and register SRP")

        time.sleep(3.0)
        omr_prefix = ns.node_cmd(otbr_id, "br omrprefix")
        configure_border_router_routes(omr_prefix)

        # Commission during 0% PLR so commissioning itself is fast and reliable
        _, initial_ota_path = create_ota_image(image_size_bytes, software_version=2)
        commission_topology_and_provider(
            chip_tool_bin=chip_tool_bin,
            provider_bin=ota_provider_bin,
            ota_image_path=initial_ota_path,
            topo_cfg=topo_cfg,
            state_reused=state_reused,
        )

        # Apply target packet loss ratio (plr) for the benchmark transfer(s)
        if plr > 0.0:
            print(f"  [otns] Setting global packet loss ratio plr={plr:.3f} ({plr*100:.1f}% FER)")
            ns.packet_loss_ratio = plr
            time.sleep(0.5)

        verified_topo = verify_topology_routing(ns, otbr_id=otbr_id, router_ids=router_ids, topo_cfg=topo_cfg)

        for idx, bsize in enumerate(block_sizes):
            # Increment software version per transfer so Requestor always sees a newer version (> current)
            _, ota_path = create_ota_image(image_size_bytes, software_version=2 + idx)
            r_dir, metrics = execute_single_bdx_transfer(
                ns=ns,
                otbr_id=otbr_id,
                topo_cfg=topo_cfg,
                verified_topo=verified_topo,
                chip_tool_bin=chip_tool_bin,
                provider_bin=ota_provider_bin,
                ota_image_path=ota_path,
                image_size_bytes=image_size_bytes,
                block_size=bsize,
                suite_name=suite_name,
                state_reused=state_reused or (idx > 0),
                log_dir=log_dir,
                contend_traffic=contend_traffic,
                timeout_s=timeout_s,
                run_tag=run_tag,
            )
            results.append((r_dir, metrics))
            time.sleep(1.5)
    finally:
        ns.close()

    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Matter BDX over Thread OTNS benchmark suites")
    parser.add_argument(
        "--config",
        default=os.path.join(REPO_ROOT, "scripts/otns/otns_bdx_multihop.json"),
        help="Path to declarative BDX topology JSON file",
    )
    parser.add_argument(
        "--suite",
        choices=["single", "suite1", "suite2", "suite3", "suite4", "all"],
        default="single",
        help="Benchmark suite to execute",
    )
    parser.add_argument("--hops", type=int, default=3, choices=[1, 2, 3], help="Hop distance for multihop topology")
    parser.add_argument("--mode", choices=["med", "sed"], default="med", help="Thread mode for OTARequestor")
    parser.add_argument("--poll-period-ms", type=int, default=200, help="SED data poll period in ms (default: 200)")
    parser.add_argument("--plr", type=float, default=0.0, help="OTNS packet loss ratio (0.0 to 1.0)")
    parser.add_argument("--block-size", type=int, default=340, help="BDX MaxBlockSize in bytes for single run")
    parser.add_argument(
        "--block-sizes",
        default=None,
        help="Comma-separated list of BDX block sizes to sweep in one session",
    )
    parser.add_argument(
        "--image-size-bytes",
        type=int,
        default=16384,
        help="OTA payload size in bytes per transfer (default: 16384 = 16 KB)",
    )
    parser.add_argument("--requestor-sai-ms", type=int, default=None, help="Override Requestor MRP SAI (ms)")
    parser.add_argument("--requestor-sii-ms", type=int, default=None, help="Override Requestor MRP SII (ms)")
    parser.add_argument(
        "--contend-traffic",
        action="store_true",
        help="Enable concurrent SubscribeReq/ReportData traffic on sibling nodes (Scenario 2)",
    )
    parser.add_argument("--reuse-state", action="store_true", help="Reuse commissioned KVS state if available")
    parser.add_argument("--timeout-sec", type=float, default=150.0, help="Max seconds per BDX transfer")
    parser.add_argument("--log-dir", default=DEFAULT_SIM_LOGS_DIR, help="Directory to archive simulation logs")
    parser.add_argument("--run-tag", default=None, help="Optional tag appended to archived run ID")
    parser.add_argument("--ota-requestor-bin", default=DEFAULT_OTA_REQUESTOR_BIN)
    parser.add_argument("--ota-provider-bin", default=DEFAULT_OTA_PROVIDER_BIN)
    parser.add_argument("--all-devices-bin", default=DEFAULT_ALL_DEVICES_BIN)
    parser.add_argument("--chip-tool-bin", default=DEFAULT_CHIP_TOOL_BIN)
    args = parser.parse_args()

    multihop_cfg = os.path.join(REPO_ROOT, "scripts/otns/otns_bdx_multihop.json")
    scen2_cfg = os.path.join(REPO_ROOT, "scripts/otns/otns_bdx_scenario2_router_drops.json")

    if args.suite == "single":
        bsizes = (
            [int(x.strip()) for x in args.block_sizes.split(",") if x.strip()]
            if args.block_sizes
            else [args.block_size]
        )
        run_otns_session_transfers(
            config_path=args.config,
            suite_name="single",
            hops=args.hops,
            mode=args.mode,
            poll_period_ms=args.poll_period_ms,
            plr=args.plr,
            block_sizes=bsizes,
            image_size_bytes=args.image_size_bytes,
            ota_requestor_bin=args.ota_requestor_bin,
            ota_provider_bin=args.ota_provider_bin,
            all_devices_bin=args.all_devices_bin,
            chip_tool_bin=args.chip_tool_bin,
            log_dir=args.log_dir,
            reuse_state=args.reuse_state,
            requestor_sai_ms=args.requestor_sai_ms,
            requestor_sii_ms=args.requestor_sii_ms,
            contend_traffic=args.contend_traffic,
            timeout_s=args.timeout_sec,
            run_tag=args.run_tag,
        )
        return 0

    if args.suite in ("suite1", "all"):
        print("\n================ SUITE 1: Slide 10 3-Hop Comparison (1024B vs 340B, SED & MED, 0% & 5% FER) ================")
        for mode in ("med", "sed"):
            for plr in (0.0, 0.05):
                run_otns_session_transfers(
                    config_path=multihop_cfg,
                    suite_name="suite1_slide10",
                    hops=3,
                    mode=mode,
                    poll_period_ms=args.poll_period_ms,
                    plr=plr,
                    block_sizes=[340, 1024],
                    image_size_bytes=args.image_size_bytes,
                    ota_requestor_bin=args.ota_requestor_bin,
                    ota_provider_bin=args.ota_provider_bin,
                    all_devices_bin=args.all_devices_bin,
                    chip_tool_bin=args.chip_tool_bin,
                    log_dir=args.log_dir,
                    reuse_state=True,
                    timeout_s=args.timeout_sec,
                    run_tag=args.run_tag,
                )

    if args.suite in ("suite2", "all"):
        print("\n================ SUITE 2: 2026-05-19 Block-Size & 6LoWPAN Fragmentation Sweep (64B..1024B, 1-3 Hops) ================")
        sweep_blocks = (
            [int(x.strip()) for x in args.block_sizes.split(",") if x.strip()]
            if args.block_sizes
            else SUITE2_BLOCK_SIZES
        )
        done_suite2 = set()
        idx_file = os.path.join(args.log_dir, "runs_index.jsonl")
        if os.path.isfile(idx_file):
            with open(idx_file, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        rec = json.loads(line)
                        if rec.get("suite") == "suite2_blocksizes" and rec.get("completed"):
                            done_suite2.add((int(rec["hops"]), str(rec["thread_mode"]), round(float(rec["plr"]), 3), int(rec["block_size"])))
                    except Exception:
                        pass
        for hops in (1, 2, 3):
            for plr in (0.0, 0.05):
                rem_blocks = [b for b in sweep_blocks if (hops, args.mode, round(plr, 3), b) not in done_suite2]
                if not rem_blocks:
                    print(f"  [suite2] Skipping hops={hops} mode={args.mode} plr={plr} (all {len(sweep_blocks)} block sizes already completed)")
                    continue
                run_otns_session_transfers(
                    config_path=multihop_cfg,
                    suite_name="suite2_blocksizes",
                    hops=hops,
                    mode=args.mode,
                    poll_period_ms=args.poll_period_ms,
                    plr=plr,
                    block_sizes=rem_blocks,
                    image_size_bytes=args.image_size_bytes,
                    ota_requestor_bin=args.ota_requestor_bin,
                    ota_provider_bin=args.ota_provider_bin,
                    all_devices_bin=args.all_devices_bin,
                    chip_tool_bin=args.chip_tool_bin,
                    log_dir=args.log_dir,
                    reuse_state=True,
                    timeout_s=args.timeout_sec,
                    run_tag=args.run_tag,
                )

    if args.suite in ("suite3", "all"):
        print("\n================ SUITE 3: Scenario 1 Aggressive MRP (SAI=300ms vs 2000ms) & Large Block Size (1024B vs 340B) ================")
        for sai_ms in (300, 2000):
            for mode in ("med", "sed"):
                run_otns_session_transfers(
                    config_path=multihop_cfg,
                    suite_name="suite3_scenario1_mrp",
                    hops=3,
                    mode=mode,
                    poll_period_ms=args.poll_period_ms,
                    plr=0.0,
                    block_sizes=[340, 1024],
                    image_size_bytes=args.image_size_bytes,
                    ota_requestor_bin=args.ota_requestor_bin,
                    ota_provider_bin=args.ota_provider_bin,
                    all_devices_bin=args.all_devices_bin,
                    chip_tool_bin=args.chip_tool_bin,
                    log_dir=args.log_dir,
                    reuse_state=True,
                    requestor_sai_ms=sai_ms,
                    requestor_sii_ms=500 if sai_ms == 300 else 2000,
                    timeout_s=args.timeout_sec,
                    run_tag=args.run_tag,
                )

    if args.suite in ("suite4", "all"):
        print("\n================ SUITE 4: Scenario 2 Intermediate Router Contention (OTBR <-> A1 <-> {A2, Outlet, A3}) ================")
        for contend in (False, True):
            run_otns_session_transfers(
                config_path=scen2_cfg,
                suite_name="suite4_scenario2_router",
                hops=2,
                mode="med",
                poll_period_ms=args.poll_period_ms,
                plr=0.0,
                block_sizes=[340, 1024],
                image_size_bytes=args.image_size_bytes,
                ota_requestor_bin=args.ota_requestor_bin,
                ota_provider_bin=args.ota_provider_bin,
                all_devices_bin=args.all_devices_bin,
                chip_tool_bin=args.chip_tool_bin,
                log_dir=args.log_dir,
                reuse_state=True,
                contend_traffic=contend,
                timeout_s=args.timeout_sec,
                run_tag=args.run_tag,
            )

    return 0


if __name__ == "__main__":
    sys.exit(main())
