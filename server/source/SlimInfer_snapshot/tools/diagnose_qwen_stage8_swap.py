#!/usr/bin/env python3
"""Run one guarded efficiency case while attributing system and worker swap."""
from __future__ import annotations

import argparse
import csv
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

_REPO_BOOTSTRAP = Path(__file__).resolve().parents[1]
if str(_REPO_BOOTSTRAP) not in sys.path:
    sys.path.insert(0, str(_REPO_BOOTSTRAP))

from tools.qwen_pipeline_common import (
    REPO_ROOT,
    atomic_write_json,
    implementation_revision,
    load_json,
    now_iso,
)
from tools.qwen_protocol_gate import RUNTIME_FILES, run_protocol_gate
from tools.qwen_resource_guard import monitor_resources
from tools.run_qwen_unified_efficiency import _command


def _key_values(path: Path) -> Dict[str, int]:
    values: Dict[str, int] = {}
    try:
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if ":" not in line:
                continue
            key, raw = line.split(":", 1)
            token = raw.strip().split()[0] if raw.strip() else "0"
            try:
                values[key] = int(token)
            except ValueError:
                continue
    except (FileNotFoundError, ProcessLookupError):
        pass
    return values


def _vmstat() -> Dict[str, int]:
    values: Dict[str, int] = {}
    for line in Path("/proc/vmstat").read_text(encoding="utf-8").splitlines():
        key, value = line.split()
        if key in {"pswpin", "pswpout", "pgmajfault"}:
            values[key] = int(value)
    return values


def _sample(pid: int, started: float) -> Dict[str, Any]:
    memory = _key_values(Path("/proc/meminfo"))
    worker = _key_values(Path(f"/proc/{pid}/status"))
    vm = _vmstat()
    return {
        "timestamp": now_iso(),
        "elapsed_s": round(time.monotonic() - started, 3),
        "system_swap_used_kib": memory.get("SwapTotal", 0) - memory.get("SwapFree", 0),
        "system_swap_cached_kib": memory.get("SwapCached", 0),
        "mem_available_kib": memory.get("MemAvailable", 0),
        "mem_free_kib": memory.get("MemFree", 0),
        "cached_kib": memory.get("Cached", 0),
        "worker_rss_kib": worker.get("VmRSS", 0),
        "worker_hwm_kib": worker.get("VmHWM", 0),
        "worker_swap_kib": worker.get("VmSwap", 0),
        "pswpin_pages": vm.get("pswpin", 0),
        "pswpout_pages": vm.get("pswpout", 0),
        "major_faults": vm.get("pgmajfault", 0),
    }


def _write_csv(path: Path, rows: list[Dict[str, Any]]) -> None:
    fields = list(rows[0])
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def run(
    pipeline_dir: Path,
    method: str,
    length: int,
    *,
    direct_gpu_loader: bool,
) -> Dict[str, Any]:
    gate = run_protocol_gate(pipeline_dir, 8)
    if gate["decision"] != "PASS":
        raise RuntimeError("Stage 8 protocol gate failed")
    policy = load_json(pipeline_dir / "SAFETY_POLICY.json")
    resource_gate = monitor_resources(
        policy,
        int(policy["thresholds"]["resource_gate_duration_seconds"]),
        int(policy["thresholds"]["resource_gate_interval_seconds"]),
        pipeline_dir=pipeline_dir,
        allowed_pids=(os.getpid(),),
    )
    if not resource_gate["pass"]:
        raise RuntimeError(f"Resource gate failed: {resource_gate['checks'][-1]['failures']}")

    root = REPO_ROOT / "results" / f"qwen_stage8_swap_diagnosis_{datetime.now():%Y%m%d_%H%M%S}"
    case_dir = root / "case"
    case_dir.mkdir(parents=True)
    revision = implementation_revision(RUNTIME_FILES)
    command = _command(method, length, case_dir, revision)
    loader = "frozen-standard"
    if direct_gpu_loader:
        command[1] = str(REPO_ROOT / "tools/qwen_efficiency_direct_gpu_loader.py")
        loader = "direct-gpu-low-cpu-memory-v1"
    atomic_write_json(root / "COMMAND.json", {"command": command})
    atomic_write_json(root / "RESOURCE_GATE.json", resource_gate)

    log_path = root / "worker.log"
    threshold = int(policy["thresholds"]["max_swap_growth_bytes_during_gate"])
    rows: list[Dict[str, Any]] = []
    stopped = False
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.Popen(command, cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT)
        started = time.monotonic()
        while True:
            row = _sample(proc.pid, started)
            rows.append(row)
            growth_bytes = (
                row["system_swap_used_kib"] - rows[0]["system_swap_used_kib"]
            ) * 1024
            if growth_bytes > threshold and proc.poll() is None:
                stopped = True
                proc.send_signal(signal.SIGTERM)
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
            returncode = proc.poll()
            if returncode is not None:
                rows.append(_sample(proc.pid, started))
                break
            time.sleep(0.5)

    _write_csv(root / "swap_timeline.csv", rows)
    initial_swap = int(rows[0]["system_swap_used_kib"])
    max_system_swap = max(int(row["system_swap_used_kib"]) for row in rows)
    worker_swap_peak = max(int(row["worker_swap_kib"]) for row in rows)
    system_growth_bytes = (max_system_swap - initial_swap) * 1024
    result = {
        "created_at": now_iso(),
        "decision": "HARD_STOP" if stopped else "PASS",
        "diagnostic_only": True,
        "loader": loader,
        "method": method,
        "context_length": length,
        "worker_pid": proc.pid,
        "returncode": proc.returncode,
        "samples": len(rows),
        "sample_interval_seconds": 0.5,
        "system_swap_growth_bytes": system_growth_bytes,
        "worker_swap_peak_bytes": worker_swap_peak * 1024,
        "other_process_swap_growth_upper_bound_bytes": max(
            0, system_growth_bytes - worker_swap_peak * 1024
        ),
        "min_mem_available_bytes": min(int(row["mem_available_kib"]) for row in rows) * 1024,
        "pswpout_pages_delta": max(int(row["pswpout_pages"]) for row in rows) - int(rows[0]["pswpout_pages"]),
        "threshold_bytes": threshold,
        "latency_complete": (case_dir / "latency.csv").exists(),
        "root": str(root),
        "log": str(log_path),
        "timeline": str(root / "swap_timeline.csv"),
    }
    atomic_write_json(root / "DIAGNOSIS.json", result)
    atomic_write_json(pipeline_dir / "STAGE_08_SWAP_DIAGNOSIS.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipeline-dir", type=Path, required=True)
    parser.add_argument("--method", default="full_kv_fa2")
    parser.add_argument("--context-length", type=int, default=8192)
    parser.add_argument("--direct-gpu-loader", action="store_true")
    args = parser.parse_args()
    result = run(
        args.pipeline_dir.resolve(), args.method, args.context_length,
        direct_gpu_loader=args.direct_gpu_loader,
    )
    print(json.dumps(result, indent=2))
    return 0 if result["decision"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
