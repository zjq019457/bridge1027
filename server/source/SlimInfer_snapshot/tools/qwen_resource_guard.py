#!/usr/bin/env python3
"""Hardware-aware safety policy and resource gate for the Qwen machine."""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import shlex
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

_REPO_BOOTSTRAP = Path(__file__).resolve().parents[1]
if str(_REPO_BOOTSTRAP) not in sys.path:
    sys.path.insert(0, str(_REPO_BOOTSTRAP))

from tools.qwen_pipeline_common import (
    REPO_ROOT,
    atomic_write_json,
    atomic_write_text,
    load_json,
    now_iso,
    run_command,
)


GIB = 1024 ** 3
MIB = 1024 ** 2
DESKTOP_GPU_PROCESS_ALLOWLIST = {
    "/usr/libexec/gnome-remote-desktop-daemon",
    "/usr/lib/xorg/Xorg",
    "/usr/bin/gnome-shell",
}


def _number(value: str, cast=float) -> Any:
    value = value.strip()
    if value in {"", "N/A", "[N/A]"}:
        return None
    return cast(value)


def _query_gpu() -> Dict[str, Any]:
    fields = (
        "index,name,uuid,driver_version,memory.total,memory.used,memory.free,"
        "utilization.gpu,temperature.gpu,power.draw,power.limit,pstate"
    )
    result = run_command([
        "nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"
    ])
    if result["returncode"] != 0:
        return {"available": False, "error": result["output"]}
    values = [part.strip() for part in result["output"].strip().split(",")]
    if len(values) != 12:
        return {"available": False, "error": f"Unexpected nvidia-smi row: {result['output']}"}
    return {
        "available": True,
        "index": int(values[0]),
        "name": values[1],
        "uuid": values[2],
        "driver_version": values[3],
        "memory_total_mib": int(float(values[4])),
        "memory_used_mib": int(float(values[5])),
        "memory_free_mib": int(float(values[6])),
        "utilization_gpu_percent": int(float(values[7])),
        "temperature_gpu_c": int(float(values[8])),
        "power_draw_w": _number(values[9]),
        "power_limit_w": _number(values[10]),
        "pstate": values[11],
    }


def _temperature_limits() -> Dict[str, Optional[int]]:
    result = run_command(["nvidia-smi", "-q", "-d", "TEMPERATURE"])
    patterns = {
        "shutdown_c": r"GPU Shutdown Temp\s*:\s*(\d+) C",
        "slowdown_c": r"GPU Slowdown Temp\s*:\s*(\d+) C",
        "max_operating_c": r"GPU Max Operating Temp\s*:\s*(\d+) C",
        "target_c": r"GPU Target Temperature\s*:\s*(\d+) C",
    }
    return {
        key: (int(match.group(1)) if (match := re.search(pattern, result["output"])) else None)
        for key, pattern in patterns.items()
    }


def _meminfo() -> Dict[str, int]:
    values: Dict[str, int] = {}
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        key, raw = line.split(":", 1)
        number = int(raw.strip().split()[0]) * 1024
        values[key] = number
    return {
        "ram_total_bytes": values["MemTotal"],
        "ram_available_bytes": values["MemAvailable"],
        "swap_total_bytes": values["SwapTotal"],
        "swap_free_bytes": values["SwapFree"],
        "swap_used_bytes": values["SwapTotal"] - values["SwapFree"],
    }


def _gpu_processes() -> List[Dict[str, Any]]:
    result = run_command([
        "nvidia-smi",
        "--query-compute-apps=pid,process_name,used_memory",
        "--format=csv,noheader,nounits",
    ])
    rows: List[Dict[str, Any]] = []
    if result["returncode"] != 0:
        return rows
    for line in result["output"].splitlines():
        if not line.strip():
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 3:
            continue
        rows.append({
            "pid": int(parts[0]),
            "process_name": parts[1],
            "used_memory_mib": int(float(parts[2])),
            "desktop_allowlisted": parts[1] in DESKTOP_GPU_PROCESS_ALLOWLIST,
        })
    return rows


def _experiment_processes() -> List[Dict[str, Any]]:
    result = run_command(["ps", "-eo", "pid=,args="])
    rows: List[Dict[str, Any]] = []
    own_pid = os.getpid()
    ancestor_pids = {own_pid}
    cursor = own_pid
    while cursor > 1:
        status_path = Path(f"/proc/{cursor}/status")
        if not status_path.exists():
            break
        parent = None
        for line in status_path.read_text(encoding="utf-8").splitlines():
            if line.startswith("PPid:"):
                parent = int(line.split()[1])
                break
        if parent is None or parent in ancestor_pids:
            break
        ancestor_pids.add(parent)
        cursor = parent
    for line in result["output"].splitlines():
        line = line.strip()
        if not line:
            continue
        pid_text, _, command = line.partition(" ")
        try:
            pid = int(pid_text)
        except ValueError:
            continue
        if pid in ancestor_pids or "qwen_resource_guard.py" in command:
            continue
        if _is_experiment_command(command):
            rows.append({"pid": pid, "command": command})
    return rows


def _is_experiment_command(command: str) -> bool:
    """Match the executed entrypoint, not arbitrary text in shell arguments."""
    patterns = (
        "run_qwen", "longbench", "sliminfer_variants",
        "minference", "flexprefill", "lazyllm",
    )
    try:
        tokens = shlex.split(command)
    except ValueError:
        return False
    if not tokens:
        return False
    executable = Path(tokens[0]).name.lower()
    entrypoint = executable
    if executable.startswith("python"):
        entrypoint = ""
        for index, token in enumerate(tokens[1:], start=1):
            if token == "-m" and index + 1 < len(tokens):
                entrypoint = tokens[index + 1]
                break
            if token.startswith("-"):
                continue
            entrypoint = Path(token).name
            break
    elif executable in {"bash", "sh", "zsh"}:
        if len(tokens) < 2 or tokens[1] in {"-c", "-lc"}:
            return False
        entrypoint = Path(tokens[1]).name
    return any(pattern in entrypoint.lower() for pattern in patterns)


def _recent_driver_errors() -> Dict[str, Any]:
    result = run_command(["journalctl", "-k", "--since", "-5 min", "--no-pager"])
    if result["returncode"] != 0:
        return {"query_ok": False, "matches": [], "error": result["output"][-1000:]}
    matches = [
        line for line in result["output"].splitlines()
        if re.search(r"NVRM: Xid|fallen off the bus|driver communication error", line, re.IGNORECASE)
    ]
    return {"query_ok": True, "matches": matches}


def collect_snapshot(path: Path = REPO_ROOT) -> Dict[str, Any]:
    disk = shutil.disk_usage(path)
    return {
        "captured_at": now_iso(),
        "gpu": _query_gpu(),
        "gpu_temperature_limits": _temperature_limits(),
        "host_memory": _meminfo(),
        "disk": {
            "path": str(path),
            "total_bytes": disk.total,
            "used_bytes": disk.used,
            "free_bytes": disk.free,
        },
        "gpu_compute_processes": _gpu_processes(),
        "experiment_processes": _experiment_processes(),
        "recent_driver_errors": _recent_driver_errors(),
    }


def derive_policy(snapshot: Mapping[str, Any]) -> Dict[str, Any]:
    gpu = snapshot["gpu"]
    if not gpu.get("available"):
        raise RuntimeError(f"GPU unavailable: {gpu.get('error')}")
    limits = snapshot["gpu_temperature_limits"]
    candidates = [
        limits.get("target_c"),
        (limits["max_operating_c"] - 5) if limits.get("max_operating_c") else None,
        (limits["slowdown_c"] - 10) if limits.get("slowdown_c") else None,
    ]
    temperature_threshold = min(value for value in candidates if value is not None)
    memory = snapshot["host_memory"]
    disk = snapshot["disk"]
    return {
        "schema_version": 2,
        "created_at": now_iso(),
        "machine_role": "qwen_experiment_machine",
        "detected_baseline": snapshot,
        "environment": {
            "TOKENIZERS_PARALLELISM": "false",
            "OMP_NUM_THREADS": "4",
            "MKL_NUM_THREADS": "4",
            "MAX_JOBS": "2",
            "dataloader_num_workers": 0,
            "batch_size": 1,
        },
        "thresholds": {
            "resource_gate_duration_seconds": 30,
            "resource_gate_interval_seconds": 5,
            "min_gpu_free_mib_before_model": math.floor(gpu["memory_total_mib"] * 0.90),
            "max_gpu_memory_drift_mib_during_gate": max(512, math.ceil(gpu["memory_total_mib"] * 0.03)),
            "max_idle_gpu_utilization_percent": min(50, max(15, gpu["utilization_gpu_percent"] + 12)),
            "max_gpu_temperature_c": temperature_threshold,
            "min_host_available_bytes": max(8 * GIB, math.ceil(memory["ram_total_bytes"] * 0.20)),
            "max_swap_growth_bytes_during_gate": min(1 * GIB, max(256 * MIB, memory["swap_total_bytes"] // 10)),
            "min_disk_free_bytes": max(100 * GIB, math.ceil(disk["total_bytes"] * 0.10)),
            "max_pipeline_output_bytes": min(250 * GIB, math.floor(disk["free_bytes"] * 0.20)),
            "max_consecutive_sample_oom": 2,
            "runtime_swap_sustained_observations": 2,
            "runtime_post_load_stable_observations": 2,
            "runtime_post_load_stable_delta_bytes": 64 * MIB,
            "runtime_swap_growth_epsilon_bytes": 1 * MIB,
            "post_worker_cooldown_seconds": 30,
        },
        "rules": {
            "single_gpu": True,
            "single_model": True,
            "sequential_methods": True,
            "raw_cpu_offload_forbidden": True,
            "oom_fill_only_after_raw_stages": True,
            "sudo_forbidden": True,
            "system_restart_forbidden": True,
            "driver_cuda_kernel_changes_forbidden": True,
            "unrelated_process_kill_forbidden": True,
            "external_baseline_env_isolation_required": True,
            "runtime_worker_cgroup_swap_primary": True,
            "runtime_system_swap_auxiliary_only": True,
            "runtime_swap_sustained_growth_required": True,
            "allowed_background_gpu_processes": sorted(DESKTOP_GPU_PROCESS_ALLOWLIST),
        },
        "derivation": {
            "gpu_free": "90% of detected total VRAM",
            "temperature": "minimum of detected target, max-operating minus 5C, slowdown minus 10C",
            "host_ram": "maximum of 8 GiB and 20% of detected RAM",
            "disk": "maximum of 100 GiB and 10% of detected filesystem",
            "not_hardcoded_for_other_machine": True,
        },
    }


def directory_size(path: Optional[Path]) -> int:
    if path is None or not path.exists():
        return 0
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def evaluate_snapshot(
    snapshot: Mapping[str, Any],
    policy: Mapping[str, Any],
    *,
    initial_snapshot: Optional[Mapping[str, Any]] = None,
    pipeline_dir: Optional[Path] = None,
    allowed_pids: Sequence[int] = (),
) -> Dict[str, Any]:
    failures: List[Dict[str, Any]] = []
    thresholds = policy["thresholds"]
    gpu = snapshot["gpu"]
    if not gpu.get("available"):
        failures.append({"check": "gpu_available", "value": gpu.get("error")})
    else:
        checks = (
            (gpu["memory_free_mib"] >= thresholds["min_gpu_free_mib_before_model"], "gpu_free_mib", gpu["memory_free_mib"], thresholds["min_gpu_free_mib_before_model"]),
            (gpu["temperature_gpu_c"] <= thresholds["max_gpu_temperature_c"], "gpu_temperature_c", gpu["temperature_gpu_c"], thresholds["max_gpu_temperature_c"]),
            (gpu["utilization_gpu_percent"] <= thresholds["max_idle_gpu_utilization_percent"], "idle_gpu_utilization_percent", gpu["utilization_gpu_percent"], thresholds["max_idle_gpu_utilization_percent"]),
        )
        for passed, name, value, threshold in checks:
            if not passed:
                failures.append({"check": name, "value": value, "threshold": threshold})
    memory = snapshot["host_memory"]
    if memory["ram_available_bytes"] < thresholds["min_host_available_bytes"]:
        failures.append({
            "check": "host_available_bytes",
            "value": memory["ram_available_bytes"],
            "threshold": thresholds["min_host_available_bytes"],
        })
    if snapshot["disk"]["free_bytes"] < thresholds["min_disk_free_bytes"]:
        failures.append({
            "check": "disk_free_bytes",
            "value": snapshot["disk"]["free_bytes"],
            "threshold": thresholds["min_disk_free_bytes"],
        })
    if snapshot["recent_driver_errors"].get("matches"):
        failures.append({"check": "recent_driver_errors", "value": snapshot["recent_driver_errors"]["matches"]})
    allowed = set(int(pid) for pid in allowed_pids)
    unexpected_gpu = [
        row for row in snapshot["gpu_compute_processes"]
        if not row.get("desktop_allowlisted") and row["pid"] not in allowed
    ]
    if unexpected_gpu:
        failures.append({"check": "unexpected_gpu_processes", "value": unexpected_gpu})
    unexpected_experiments = [
        row for row in snapshot["experiment_processes"] if row["pid"] not in allowed
    ]
    if unexpected_experiments:
        failures.append({"check": "other_experiment_processes", "value": unexpected_experiments})
    if initial_snapshot is not None:
        gpu_drift = abs(
            int(snapshot["gpu"].get("memory_used_mib", 0))
            - int(initial_snapshot["gpu"].get("memory_used_mib", 0))
        )
        if gpu_drift > thresholds["max_gpu_memory_drift_mib_during_gate"]:
            failures.append({
                "check": "gpu_memory_drift_mib",
                "value": gpu_drift,
                "threshold": thresholds["max_gpu_memory_drift_mib_during_gate"],
            })
        swap_growth = max(
            0,
            int(snapshot["host_memory"]["swap_used_bytes"])
            - int(initial_snapshot["host_memory"]["swap_used_bytes"]),
        )
        if swap_growth > thresholds["max_swap_growth_bytes_during_gate"]:
            failures.append({
                "check": "swap_growth_bytes",
                "value": swap_growth,
                "threshold": thresholds["max_swap_growth_bytes_during_gate"],
            })
    output_bytes = directory_size(pipeline_dir)
    if output_bytes > thresholds["max_pipeline_output_bytes"]:
        failures.append({
            "check": "pipeline_output_bytes",
            "value": output_bytes,
            "threshold": thresholds["max_pipeline_output_bytes"],
        })
    return {
        "captured_at": snapshot["captured_at"],
        "pass": not failures,
        "failures": failures,
        "pipeline_output_bytes": output_bytes,
    }


def monitor_resources(
    policy: Mapping[str, Any],
    duration_seconds: int,
    interval_seconds: int,
    *,
    pipeline_dir: Optional[Path] = None,
    allowed_pids: Sequence[int] = (),
) -> Dict[str, Any]:
    started = time.monotonic()
    snapshots: List[Dict[str, Any]] = []
    checks: List[Dict[str, Any]] = []
    initial = collect_snapshot()
    while True:
        snapshot = initial if not snapshots else collect_snapshot()
        snapshots.append(snapshot)
        check = evaluate_snapshot(
            snapshot,
            policy,
            initial_snapshot=initial,
            pipeline_dir=pipeline_dir,
            allowed_pids=allowed_pids,
        )
        checks.append(check)
        if not check["pass"]:
            break
        elapsed = time.monotonic() - started
        if elapsed >= duration_seconds:
            break
        time.sleep(min(interval_seconds, max(0.0, duration_seconds - elapsed)))
    return {
        "started_at": initial["captured_at"],
        "completed_at": now_iso(),
        "requested_duration_seconds": duration_seconds,
        "observations": len(snapshots),
        "pass": all(check["pass"] for check in checks),
        "checks": checks,
        "snapshots": snapshots,
    }


def write_blocker(pipeline_dir: Path, report: Mapping[str, Any]) -> None:
    atomic_write_json(pipeline_dir / "SAFETY_STOP.json", {
        "created_at": now_iso(),
        "reason": "resource_gate_failed",
        "report": report,
    })
    atomic_write_text(
        pipeline_dir / "RESOURCE_GATE_BLOCKED.md",
        "# Resource Gate Blocked\n\nThe resource gate failed. No GPU model stage was started. "
        "See `SAFETY_STOP.json` and use the pipeline resume command after resources are safe.\n",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-policy", type=Path)
    parser.add_argument("--policy", type=Path)
    parser.add_argument("--snapshot-out", type=Path)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--duration", type=int)
    parser.add_argument("--interval", type=int)
    parser.add_argument("--pipeline-dir", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.write_policy:
        snapshot = collect_snapshot()
        policy = derive_policy(snapshot)
        atomic_write_json(args.write_policy, policy)
        if args.snapshot_out:
            atomic_write_json(args.snapshot_out, snapshot)
        print(json.dumps({"policy": str(args.write_policy), "gpu": snapshot["gpu"]}, indent=2))
        return 0
    if args.check:
        if not args.policy:
            raise SystemExit("--check requires --policy")
        policy = load_json(args.policy)
        duration = args.duration if args.duration is not None else int(policy["thresholds"]["resource_gate_duration_seconds"])
        interval = args.interval if args.interval is not None else int(policy["thresholds"]["resource_gate_interval_seconds"])
        report = monitor_resources(policy, duration, interval, pipeline_dir=args.pipeline_dir)
        if args.snapshot_out:
            atomic_write_json(args.snapshot_out, report)
        if not report["pass"] and args.pipeline_dir:
            write_blocker(args.pipeline_dir, report)
        print(json.dumps(report, indent=2))
        return 0 if report["pass"] else 2
    snapshot = collect_snapshot()
    if args.snapshot_out:
        atomic_write_json(args.snapshot_out, snapshot)
    print(json.dumps(snapshot, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
