#!/usr/bin/env python3
"""BridgePack G1 dedicated helpers (do not modify shared GPU runners)."""
from __future__ import annotations

import csv
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
HIST_ROOT = REPO_ROOT / "results" / "bridgepack_exact_qwen_budget_1024_20260619_181059"
QWEN_PATH = Path(
    "/home/yu-bai/.cache/huggingface/hub/"
    "models--Qwen--Qwen2.5-7B-Instruct/snapshots/"
    "a09a35458c702b33eeacc393d103063234e8bc28"
)
TASKS = ("qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "musique", "narrativeqa")
LBP_VARIANT = "lexical_bridgepack_path_order"
BUDGET = 1024
OTHER_PID = 399011  # legacy marker; do not signal — always scan all GPU apps
# No historical peak recorded; enforce absolute free-VRAM floor for 3090 Ti.
MIN_FREE_MIB = 21504  # ~21 GiB before loading Qwen
SAFETY_MARGIN_MIB = 2048
RUNTIME_MIN_FREE_MIB = 1536  # critical headroom after our own model is loaded



def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def append_csv(path: Path, row: Mapping[str, Any], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore", lineterminator="\n")
        if not exists:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in fields})


def gpu_snapshot() -> Dict[str, Any]:
    proc = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu",
            "--format=csv,noheader,nounits",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        check=False,
    )
    if proc.returncode != 0 or not proc.stdout.strip():
        return {"error": "nvidia-smi failed"}
    total, used, free, util, temp = [x.strip() for x in proc.stdout.strip().splitlines()[0].split(",")]
    apps = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,process_name,used_gpu_memory", "--format=csv,noheader"],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        check=False,
    )
    return {
        "memory_total_mib": int(total),
        "memory_used_mib": int(used),
        "memory_free_mib": int(free),
        "utilization_gpu": int(util),
        "temperature_c": int(temp),
        "apps": apps.stdout.strip(),
    }


def mem_snapshot() -> Dict[str, Any]:
    info = {}
    with open("/proc/meminfo", encoding="utf-8") as handle:
        for line in handle:
            if ":" in line:
                k, v = line.split(":", 1)
                info[k.strip()] = v.strip()
    return {
        "mem_available": info.get("MemAvailable"),
        "swap_free": info.get("SwapFree"),
        "swap_total": info.get("SwapTotal"),
    }


def other_pid_alive(pid: int = OTHER_PID) -> bool:
    return Path(f"/proc/{pid}").exists()


def list_gpu_compute_apps() -> List[Dict[str, Any]]:
    proc = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,process_name,used_gpu_memory", "--format=csv,noheader"],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        check=False,
    )
    apps: List[Dict[str, Any]] = []
    if proc.returncode != 0:
        return apps
    for line in proc.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            continue
        pid_s, name, mem_s = parts[0], parts[1], parts[2]
        try:
            mem_mib = int(mem_s.split()[0])
        except Exception:
            mem_mib = 0
        try:
            pid = int(pid_s)
        except Exception:
            pid = -1
        apps.append({"pid": pid, "name": name, "used_gpu_memory_mib": mem_mib})
    return apps


def other_gpu_workloads(bridgepack_pid: Optional[int] = None) -> List[Dict[str, Any]]:
    """Non-BridgePack GPU compute processes with meaningful VRAM."""
    me = bridgepack_pid if bridgepack_pid is not None else os.getpid()
    out = []
    for app in list_gpu_compute_apps():
        if app["pid"] == me:
            continue
        # Ignore tiny desktop daemons (< 400 MiB) unless actively loading
        if app["used_gpu_memory_mib"] >= 400:
            out.append(app)
    return out


def safety_gate(out_dir: Path, stage: str, *, phase: str = "runtime") -> Dict[str, Any]:
    """phase: preflight|before_load|runtime

    - before_load/preflight: require free >= MIN_FREE_MIB (~21GB) when no BridgePack model held
    - runtime: model already loaded; require free >= RUNTIME_MIN_FREE_MIB and watch other GPU jobs
    - Always scan **all** GPU compute apps (not only legacy PID 399011)
    """
    g = gpu_snapshot()
    m = mem_snapshot()
    alive_399011 = other_pid_alive()
    foreign = other_gpu_workloads()
    free = int(g.get("memory_free_mib") or 0)
    util = int(g.get("utilization_gpu") or 0)
    decision = "GO"
    reasons = []
    if alive_399011:
        reasons.append(f"legacy PID {OTHER_PID} ALIVE — must not signal/modify")
    if foreign:
        foreign_desc = "; ".join(
            f"pid={a['pid']} mem={a['used_gpu_memory_mib']}MiB name={a['name']}" for a in foreign
        )
        reasons.append(f"foreign GPU workloads: {foreign_desc}")
        if any(a["used_gpu_memory_mib"] >= 1024 for a in foreign) or util >= 40:
            decision = "WAITING_FOR_SAFE_GPU"
    if phase in {"preflight", "before_load"}:
        if free < MIN_FREE_MIB:
            decision = "WAITING_FOR_SAFE_GPU"
            reasons.append(f"free_mib={free} < required {MIN_FREE_MIB}")
        if foreign and any(a["used_gpu_memory_mib"] >= 2048 for a in foreign):
            decision = "WAITING_FOR_SAFE_GPU"
            reasons.append("another paper-scale GPU job detected before load")
    else:
        if free < RUNTIME_MIN_FREE_MIB:
            decision = "WAITING_FOR_SAFE_GPU"
            reasons.append(f"runtime free_mib={free} < critical {RUNTIME_MIN_FREE_MIB}")
        # Sudden foreign VRAM growth during our run
        if foreign and any(a["used_gpu_memory_mib"] >= 2048 for a in foreign):
            decision = "WAITING_FOR_SAFE_GPU"
            reasons.append("foreign GPU job >=2GB during runtime")
    if int(g.get("temperature_c") or 0) >= 90:
        decision = "WAITING_FOR_SAFE_GPU"
        reasons.append("GPU temperature >= 90C")
    row = {
        "timestamp": utc_now(),
        "stage": stage,
        "phase": phase,
        "decision": decision,
        "other_pid_alive": alive_399011,
        "foreign_gpu_apps": json.dumps(foreign, ensure_ascii=False),
        "bridgepack_pid": os.getpid(),
        **{k: g.get(k) for k in [
            "memory_total_mib", "memory_used_mib", "memory_free_mib", "utilization_gpu", "temperature_c"
        ]},
        "mem_available": m.get("mem_available"),
        "swap_free": m.get("swap_free"),
        "reasons": "; ".join(reasons) if reasons else "ok",
        "apps": g.get("apps", ""),
    }
    append_csv(
        out_dir / "G1_GPU_RESOURCE_REPORT.csv",
        row,
        [
            "timestamp", "stage", "phase", "decision", "other_pid_alive", "foreign_gpu_apps", "bridgepack_pid",
            "memory_total_mib", "memory_used_mib", "memory_free_mib", "utilization_gpu",
            "temperature_c", "mem_available", "swap_free", "reasons", "apps",
        ],
    )
    return row


def select_fixed_sample_ids(n_per_task: int) -> Dict[str, List[str]]:
    """Deterministic first-N sample_ids from historical LBP packs."""
    out: Dict[str, List[str]] = {}
    for task in TASKS:
        path = HIST_ROOT / "packed_prompts_build" / "packed_prompts" / LBP_VARIANT / f"budget_{BUDGET}" / f"{task}.jsonl"
        rows = read_jsonl(path)
        ids = [str(r["sample_id"]) for r in rows[:n_per_task]]
        if len(ids) < n_per_task:
            raise RuntimeError(f"Need {n_per_task} samples for {task}, found {len(ids)}")
        out[task] = ids
    return out


def load_hist_lbp_rows(sample_ids_by_task: Mapping[str, Sequence[str]]) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = {}
    for task, ids in sample_ids_by_task.items():
        want = set(ids)
        path = HIST_ROOT / "packed_prompts_build" / "packed_prompts" / LBP_VARIANT / f"budget_{BUDGET}" / f"{task}.jsonl"
        rows = [r for r in read_jsonl(path) if str(r.get("sample_id")) in want]
        # preserve requested order
        by_id = {str(r["sample_id"]): r for r in rows}
        ordered = [by_id[i] for i in ids if i in by_id]
        if len(ordered) != len(ids):
            raise RuntimeError(f"Missing historical packs for {task}")
        out[task] = ordered
    return out


def load_hist_preds(sample_ids_by_task: Mapping[str, Sequence[str]]) -> Dict[str, Dict[str, Dict[str, Any]]]:
    out: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for task, ids in sample_ids_by_task.items():
        path = HIST_ROOT / "predictions" / LBP_VARIANT / f"{task}.jsonl"
        by_id = {str(r.get("sample_id")): r for r in read_jsonl(path)}
        out[task] = {i: by_id[i] for i in ids if i in by_id}
        if len(out[task]) != len(ids):
            raise RuntimeError(f"Missing historical preds for {task}")
    return out
