#!/usr/bin/env python3
"""GPU idle gate and status writers for LLAMA_DRM_ON_QWEN_V1."""
from __future__ import annotations

import csv
import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional

from tools.llama_drm_on_qwen_v1.common import (
    atomic_write_csv,
    atomic_write_json,
    atomic_write_text,
    now_iso,
    run_command,
)
from tools.llama_drm_on_qwen_v1.constants import (
    DESKTOP_GPU_ALLOWLIST,
    GPU_POLL_SECONDS,
    STAGES,
)


def query_gpu() -> Dict[str, Any]:
    result = run_command([
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits",
    ])
    if result["returncode"] != 0:
        return {"available": False, "error": result["output"], "busy": True}
    values = [part.strip() for part in result["output"].strip().split(",")]
    if len(values) < 6:
        return {"available": False, "error": result["output"], "busy": True}
    apps = run_command([
        "nvidia-smi",
        "--query-compute-apps=pid,process_name,used_gpu_memory",
        "--format=csv,noheader",
    ])
    compute_apps: List[Dict[str, Any]] = []
    foreign_compute = False
    if apps["returncode"] == 0 and apps["output"].strip():
        for line in apps["output"].strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 2:
                continue
            pid = parts[0]
            name = parts[1]
            used = parts[2] if len(parts) > 2 else ""
            base = Path(name).name
            allowed = name in DESKTOP_GPU_ALLOWLIST or base in DESKTOP_GPU_ALLOWLIST
            entry = {"pid": pid, "process_name": name, "used_memory": used, "allowed": allowed}
            compute_apps.append(entry)
            if not allowed:
                foreign_compute = True
    used_mib = float(values[3])
    # Treat only foreign compute processes as busy. Desktop compositor memory is ignored.
    busy = foreign_compute
    return {
        "available": True,
        "index": int(float(values[0])),
        "name": values[1],
        "memory_total_mib": float(values[2]),
        "memory_used_mib": used_mib,
        "memory_free_mib": float(values[4]),
        "utilization_gpu_percent": float(values[5]),
        "compute_apps": compute_apps,
        "busy": busy,
        "foreign_compute": foreign_compute,
    }


def wait_for_gpu_idle(
    pipeline_dir: Path,
    *,
    poll_seconds: int = GPU_POLL_SECONDS,
    should_stop: Optional[Callable[[], bool]] = None,
) -> Dict[str, Any]:
    status_dir = pipeline_dir / "status"
    while True:
        snap = query_gpu()
        payload = {
            "updated_at": now_iso(),
            "state": "WAITING_FOR_GPU_IDLE" if snap.get("busy") else "GPU_IDLE",
            "gpu": snap,
            "poll_seconds": poll_seconds,
        }
        atomic_write_json(status_dir / "WATCHDOG_STATUS.json", payload)
        update_pipeline_status(
            pipeline_dir,
            stage=read_current_stage(pipeline_dir),
            state=payload["state"],
            gpu_busy=bool(snap.get("busy")),
            waiting_for_gpu=bool(snap.get("busy")),
            detail="waiting for foreign GPU compute to finish" if snap.get("busy") else "gpu idle",
        )
        if not snap.get("busy"):
            return snap
        if should_stop is not None and should_stop():
            return snap
        time.sleep(poll_seconds)


def read_current_stage(pipeline_dir: Path) -> str:
    path = pipeline_dir / "status" / "PROGRESS.json"
    if not path.exists():
        return "NOT_STARTED"
    try:
        data = json_load(path)
        return str(data.get("current_stage") or data.get("stage") or "UNKNOWN")
    except Exception:
        return "UNKNOWN"


def json_load(path: Path) -> Any:
    import json
    return json.loads(path.read_text(encoding="utf-8"))


def update_pipeline_status(
    pipeline_dir: Path,
    *,
    stage: str,
    state: str,
    gpu_busy: Optional[bool] = None,
    waiting_for_gpu: Optional[bool] = None,
    waiting_for_model: Optional[bool] = None,
    detail: str = "",
    hard_block: Optional[str] = None,
    offline: bool = True,
) -> None:
    status_dir = pipeline_dir / "status"
    status_dir.mkdir(parents=True, exist_ok=True)
    gpu = query_gpu()
    if gpu_busy is None:
        gpu_busy = bool(gpu.get("busy"))
    if waiting_for_gpu is None:
        waiting_for_gpu = bool(gpu_busy and state == "WAITING_FOR_GPU_IDLE")
    progress = {
        "updated_at": now_iso(),
        "protocol": "LLAMA_DRM_ON_QWEN_V1",
        "current_stage": stage,
        "state": state,
        "gpu_busy": gpu_busy,
        "waiting_for_gpu": waiting_for_gpu,
        "waiting_for_model": bool(waiting_for_model),
        "offline": offline,
        "detail": detail,
        "hard_block": hard_block,
        "pid": os.getpid(),
        "stage_order": [row["id"] for row in STAGES],
    }
    atomic_write_json(status_dir / "PROGRESS.json", progress)
    md = [
        "# PIPELINE_STATUS",
        "",
        f"- updated_at: `{progress['updated_at']}`",
        f"- current_stage: `{stage}`",
        f"- state: `{state}`",
        f"- gpu_busy: `{gpu_busy}`",
        f"- waiting_for_gpu: `{waiting_for_gpu}`",
        f"- waiting_for_model: `{bool(waiting_for_model)}`",
        f"- offline: `{offline}`",
        f"- detail: {detail}",
    ]
    if hard_block:
        md.append(f"- HARD_BLOCK: `{hard_block}`")
    md.extend([
        "",
        "## Stage order",
        "",
        *[f"1. {row['id']}" if i == 0 else f"{i+1}. {row['id']}" for i, row in enumerate(STAGES)],
        "",
        "## GPU",
        "",
        "```json",
        __import__("json").dumps(gpu, indent=2, sort_keys=True),
        "```",
        "",
    ])
    atomic_write_text(status_dir / "PIPELINE_STATUS.md", "\n".join(md) + "\n")


def set_current_worker(pipeline_dir: Path, worker: Mapping[str, Any]) -> None:
    payload = {"updated_at": now_iso(), **dict(worker)}
    atomic_write_json(pipeline_dir / "status" / "CURRENT_WORKER.json", payload)


def append_worker_registry(pipeline_dir: Path, row: Mapping[str, Any]) -> None:
    path = pipeline_dir / "status" / "WORKER_REGISTRY.csv"
    fieldnames = [
        "updated_at", "stage", "worker_id", "method", "task", "status",
        "pid", "returncode", "path", "detail",
    ]
    rows: List[Dict[str, Any]] = []
    if path.exists():
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for existing in reader:
                rows.append(dict(existing))
    new_row = {key: row.get(key, "") for key in fieldnames}
    new_row["updated_at"] = now_iso()
    rows.append(new_row)
    atomic_write_csv(path, rows, fieldnames)
