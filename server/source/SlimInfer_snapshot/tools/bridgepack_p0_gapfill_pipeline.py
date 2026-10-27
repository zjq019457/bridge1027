#!/usr/bin/env python3
"""P0/P1 BridgePack gap-fill orchestrator (BridgePack-only outputs).

Stages:
  wave1 — matched order/selection ablation (pack ladder + QA + stats)
  wave2 — evidence-chain metrics (CPU; may join original supporting_facts)
  wave3 — full-context common-valid retained ratio + multi-method efficiency

Hard-stops: protocol/leak/other-paper/Xid/OOM. Engineering auto-heal: max 2 rounds.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from tools.bridgepack_g1.common import append_csv, gpu_snapshot, mem_snapshot, utc_now, write_json, write_text

FREEZE_SHA = "15886731816aa2eb4e08afa121bd8ccc9631dd232474a82a8f20214cc2ce16bf"
MAX_AUTO_HEAL = 2

STAGE_ORDER = [
    ("wave1_order", "Matched order-only ablation"),
    ("wave1_selection", "Matched selection ablation"),
    ("wave1_finalize", "Wave1 tables + claim guard"),
    ("wave2_evidence", "Evidence-chain metrics"),
    ("wave3_fullctx", "Full-context retained ratio"),
    ("wave3_efficiency", "Multi-method efficiency"),
    ("finalize", "Deliverables checklist"),
]


def _ts_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def scan_gpu(out_dir: Path, stage: str, phase: str) -> Dict[str, Any]:
    g = gpu_snapshot()
    m = mem_snapshot()
    foreign: List[Dict[str, str]] = []
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,process_name,used_gpu_memory",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        me = str(os.getpid())
        for line in proc.stdout.splitlines():
            line = line.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 3:
                continue
            pid, name, mem = parts[0], parts[1], parts[2]
            if pid == me:
                continue
            if "gnome-remote-desktop" in name:
                continue
            try:
                mem_mib = int(str(mem).split()[0])
            except Exception:
                mem_mib = 0
            if mem_mib >= 400:
                foreign.append({"pid": pid, "name": name, "used_gpu_memory": mem})
    except Exception as exc:
        foreign.append({"pid": "scan_error", "name": str(exc), "used_gpu_memory": "unknown"})

    free = int(g.get("memory_free_mib") or 0)
    decision = "GO"
    reasons = []
    if foreign:
        decision = "WAITING_FOR_SAFE_GPU"
        reasons.append("foreign GPU workloads: " + json.dumps(foreign))
    if phase in {"preflight", "before_load"} and free < 21504:
        decision = "WAITING_FOR_SAFE_GPU"
        reasons.append(f"free_mib={free} < 21504")
    row = {
        "timestamp": utc_now(),
        "stage": stage,
        "phase": phase,
        "decision": decision,
        "bridgepack_pid": os.getpid(),
        "foreign_gpu_apps": json.dumps(foreign, ensure_ascii=False),
        "memory_total_mib": g.get("memory_total_mib"),
        "memory_used_mib": g.get("memory_used_mib"),
        "memory_free_mib": g.get("memory_free_mib"),
        "utilization_gpu": g.get("utilization_gpu"),
        "temperature_c": g.get("temperature_c"),
        "mem_available": m.get("mem_available"),
        "swap_free": m.get("swap_free"),
        "reasons": "; ".join(reasons) if reasons else "ok",
        "apps": g.get("apps", ""),
    }
    append_csv(
        out_dir / "GPU_RESOURCE_LOG.csv",
        row,
        [
            "timestamp",
            "stage",
            "phase",
            "decision",
            "bridgepack_pid",
            "foreign_gpu_apps",
            "memory_total_mib",
            "memory_used_mib",
            "memory_free_mib",
            "utilization_gpu",
            "temperature_c",
            "mem_available",
            "swap_free",
            "reasons",
            "apps",
        ],
    )
    return row


def initial_state(out_dir: Path) -> Dict[str, Any]:
    return {
        "created_at": utc_now(),
        "updated_at": utc_now(),
        "out_dir": str(out_dir),
        "pipeline_status": "INITIALIZING",
        "freeze_config_sha256": FREEZE_SHA,
        "budget": 1024,
        "batch_size": 1,
        "single_gpu_worker": True,
        "auto_heal_rounds": 0,
        "max_auto_heal": MAX_AUTO_HEAL,
        "hard_stop": None,
        "current_stage": STAGE_ORDER[0][0],
        "last_completed_stage": None,
        "stages": {
            key: {"name": name, "status": "pending", "notes": ""} for key, name in STAGE_ORDER
        },
        "reuse_roots": {
            "g1": "results/bridgepack_g1_qwen1024_matched_20260727_004934",
            "g2": "results/bridgepack_g2_llama1024_sourceorder_20260727_023300",
            "ind_fix": "results/bridgepack_g2_independent_fix_20260727_132221",
            "completion": "results/bridgepack_completion_pipeline_20260727_140453",
        },
        "forbidden_claims": [
            "cite 38.05 as Qwen source-order",
            "端到端更快 without full baseline",
            "method is KV-cache pruning",
        ],
        "qwen_source_order_macro": {"exact": 39.385, "paper": 39.39},
    }


def write_status(out_dir: Path, state: Mapping[str, Any]) -> None:
    lines = [
        "# PIPELINE_STATUS",
        "",
        f"Generated: {utc_now()}",
        "",
        f"- out_dir: `{out_dir}`",
        f"- pipeline_status: `{state['pipeline_status']}`",
        f"- current_stage: `{state['current_stage']}`",
        f"- last_completed_stage: `{state['last_completed_stage']}`",
        f"- hard_stop: `{state['hard_stop']}`",
        f"- auto_heal_rounds: `{state.get('auto_heal_rounds', 0)}/{MAX_AUTO_HEAL}`",
        f"- freeze_config_sha256: `{FREEZE_SHA}`",
        f"- budget: `1024`",
        "",
        "## Rules",
        "",
        "- BridgePack-only output root; do not touch shared GPU runners.",
        "- Reuse legal historical predictions; only run MISSING variants.",
        "- Single GPU worker; batch=1; scan safety before load.",
        "- Hard-stop on protocol / leak / other-paper / Xid / OOM.",
        "- Max 2 engineering auto-heal rounds; never rewrite protocol to heal.",
        "- Qwen source-order macro = 39.385 → paper 39.39; never cite 38.05.",
        "",
        "## Stages",
        "",
    ]
    for key, name in STAGE_ORDER:
        item = state["stages"][key]
        marker = "DONE" if item["status"] == "done" else item["status"].upper()
        lines.append(f"- `{key}` ({name}) — **{marker}**")
        if item.get("notes"):
            lines.append(f"  - {item['notes']}")
    write_text(out_dir / "PIPELINE_STATUS.md", "\n".join(lines) + "\n")
    write_json(out_dir / "pipeline_state.json", dict(state))


def mark_stage(out_dir: Path, state: Dict[str, Any], key: str, status: str, notes: str = "") -> None:
    state["stages"][key]["status"] = status
    if notes:
        state["stages"][key]["notes"] = notes
    state["current_stage"] = key
    state["updated_at"] = utc_now()
    if status == "done":
        state["last_completed_stage"] = key
        write_text(out_dir / f"STAGE_{key.upper()}.DONE", f"{utc_now()}\n{notes}\n")
    write_status(out_dir, state)


def hard_stop(out_dir: Path, state: Dict[str, Any], reason: str, details: Optional[Dict[str, Any]] = None) -> None:
    state["pipeline_status"] = "STOPPED"
    state["hard_stop"] = reason
    state["updated_at"] = utc_now()
    write_json(
        out_dir / "PIPELINE_BLOCKED.json",
        {"timestamp": utc_now(), "reason": reason, "details": details or {}},
    )
    write_status(out_dir, state)


def run_cmd(cmd: List[str], log_path: Path, *, env: Optional[Dict[str, str]] = None) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"\n=== {utc_now()} CMD: {' '.join(cmd)}\n")
        handle.flush()
        proc = subprocess.run(
            cmd,
            cwd=str(REPO_ROOT),
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
            check=False,
        )
        handle.write(f"\n=== exit={proc.returncode}\n")
    return int(proc.returncode)


def is_hard_error(text: str) -> bool:
    # Wave3 full-context intentionally records per-sample CUDA OOM (no silent truncate).
    # Those strings must not abort the pipeline as HARD_STOP_FROM_CHILD.
    fullctx_oom_ok = "oom recorded, no silent truncate" in text.lower()
    needles = [
        "WAITING_FOR_SAFE_GPU",
        "CUDA out of memory",
        "CUDA OOM",
        "OutOfMemoryError",
        "Xid",
        "NVRM: Xid",
        "protocol",
        "over-budget",
        "SELECTOR_READS_ANSWERS",
        "foreign-paper",
        "leak",
    ]
    low = text.lower()
    for n in needles:
        if n.lower() in low:
            # protocol mismatches in harmless log lines: keep specific
            if n.lower() == "protocol" and "protocol freeze" not in low and "protocol mismatch" not in low:
                continue
            if fullctx_oom_ok and n.lower() in {
                "cuda out of memory",
                "cuda oom",
                "outofmemoryerror",
            }:
                continue
            return True
    return False


def run_with_heal(
    out_dir: Path,
    state: Dict[str, Any],
    stage_key: str,
    cmd: List[str],
    log_name: str,
) -> bool:
    """Return True on success. Engineering heal only for non-hard failures."""
    log_path = out_dir / "logs" / log_name
    for attempt in range(MAX_AUTO_HEAL + 1):
        gate = scan_gpu(out_dir, stage_key, "before_load")
        if gate["decision"] != "GO":
            hard_stop(out_dir, state, "WAITING_FOR_SAFE_GPU", gate)
            return False
        rc = run_cmd(cmd, log_path)
        log_tail = log_path.read_text(encoding="utf-8", errors="replace")[-8000:] if log_path.exists() else ""
        if rc == 0:
            return True
        if is_hard_error(log_tail):
            hard_stop(
                out_dir,
                state,
                "HARD_STOP_FROM_CHILD",
                {"stage": stage_key, "rc": rc, "log": str(log_path), "tail": log_tail[-2000:]},
            )
            return False
        state["auto_heal_rounds"] = int(state.get("auto_heal_rounds", 0)) + 1
        write_status(out_dir, state)
        if int(state["auto_heal_rounds"]) > MAX_AUTO_HEAL:
            hard_stop(
                out_dir,
                state,
                "AUTO_HEAL_EXHAUSTED",
                {"stage": stage_key, "rc": rc, "log": str(log_path)},
            )
            return False
        write_text(
            out_dir / f"AUTO_HEAL_{stage_key}_round{state['auto_heal_rounds']}.md",
            f"Retrying engineering failure for {stage_key} (rc={rc}).\n\n```\n{log_tail[-1500:]}\n```\n",
        )
    return False


def write_deliverables_list(out_dir: Path) -> None:
    expected = [
        "PIPELINE_STATUS.md",
        "MATCHED_ORDER_RESULTS.csv",
        "MATCHED_ORDER_BOOTSTRAP.csv",
        "MATCHED_SELECTION_RESULTS.csv",
        "MATCHED_SELECTION_BOOTSTRAP.csv",
        "MECHANISM_CLAIM_GUARD.md",
        "EVIDENCE_CHAIN_METRICS.csv",
        "EVIDENCE_CHAIN_REPORT.md",
        "EVIDENCE_CLAIM_GUARD.md",
        "FULL_CONTEXT_RETAINED_RATIO.csv",
        "FULL_CONTEXT_COMMON_VALID.md",
        "EFFICIENCY_MULTIMETHOD.csv",
        "EFFICIENCY_CLAIM_GUARD.md",
        "DELIVERABLES.md",
    ]
    lines = ["# DELIVERABLES", "", f"Generated: {utc_now()}", ""]
    for name in expected:
        path = out_dir / name
        status = "PRESENT" if path.exists() else "MISSING"
        lines.append(f"- `{name}` — **{status}**")
        if path.exists():
            lines.append(f"  - `{path}`")
    # stage markers
    lines.append("")
    lines.append("## Stage markers")
    for key, _ in STAGE_ORDER:
        marker = out_dir / f"STAGE_{key.upper()}.DONE"
        lines.append(f"- `{marker.name}` — {'PRESENT' if marker.exists() else 'MISSING'}")
    write_text(out_dir / "DELIVERABLES.md", "\n".join(lines) + "\n")


def main() -> int:
    # Defend against CPU-stage helpers that set CUDA_VISIBLE_DEVICES=""
    if os.environ.get("CUDA_VISIBLE_DEVICES", None) == "":
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)

    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="", help="Defaults to results/bridgepack_p0_gapfill_<UTC_ts>/")
    parser.add_argument(
        "--stop_after",
        default="finalize",
        choices=[k for k, _ in STAGE_ORDER] + ["finalize"],
    )
    parser.add_argument("--skip_gpu", action="store_true", help="CPU-only stages (debug)")
    parser.add_argument("--resume", action="store_true", help="Resume from existing out_dir state")
    args = parser.parse_args()

    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        out_dir = REPO_ROOT / "results" / f"bridgepack_p0_gapfill_{_ts_utc()}"
    out_dir.mkdir(parents=True, exist_ok=True)

    state_path = out_dir / "pipeline_state.json"
    if args.resume and state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        # Engineering recovery: clear exhausted/blocked stop so --resume can continue
        # after a code fix. Does not erase completed STAGE_*.DONE markers.
        if state.get("pipeline_status") == "STOPPED" or state.get("hard_stop"):
            blocked = out_dir / "PIPELINE_BLOCKED.json"
            note = (
                f"Cleared hard_stop={state.get('hard_stop')!r} on --resume "
                f"(auto_heal was {state.get('auto_heal_rounds')}).\n"
            )
            if blocked.exists():
                blocked.rename(out_dir / f"PIPELINE_BLOCKED_ARCHIVED_{_ts_utc()}.json")
            state["hard_stop"] = None
            state["pipeline_status"] = "RUNNING"
            state["auto_heal_rounds"] = 0
            write_text(out_dir / "RESUME_AFTER_BLOCK.md", note + f"time={utc_now()}\n")
            write_status(out_dir, state)
    else:
        state = initial_state(out_dir)
        state["pipeline_status"] = "RUNNING"
        write_status(out_dir, state)
        write_text(
            out_dir / "README.md",
            f"""# BridgePack P0 Gap-fill

- Created: {utc_now()}
- Freeze SHA: `{FREEZE_SHA}`
- Budget: 1024
- Qwen source-order (paper): **39.39** (exact 39.385); never cite 38.05.

Stages: wave1 order/selection → wave2 evidence → wave3 full-context + efficiency.
""",
        )

    py = sys.executable
    matched = str(REPO_ROOT / "tools" / "bridgepack_p0_matched_ablation.py")
    evidence = str(REPO_ROOT / "tools" / "bridgepack_p0_evidence_chain.py")
    fullctx = str(REPO_ROOT / "tools" / "bridgepack_p0_full_context_retained.py")
    efficiency = str(REPO_ROOT / "tools" / "bridgepack_p0_efficiency_compare.py")

    def done(key: str) -> bool:
        return state["stages"][key]["status"] == "done" or (out_dir / f"STAGE_{key.upper()}.DONE").exists()

    # ---- Wave1 order ----
    if not done("wave1_order"):
        mark_stage(out_dir, state, "wave1_order", "running")
        if args.skip_gpu:
            mark_stage(out_dir, state, "wave1_order", "done", "skipped_gpu")
        else:
            ok = run_with_heal(
                out_dir,
                state,
                "wave1_order",
                [py, matched, "--out_dir", str(out_dir), "--mode", "ladder_order", "--model", "all"],
                "wave1_order.log",
            )
            if not ok:
                return 2
            mark_stage(out_dir, state, "wave1_order", "done")
    if args.stop_after == "wave1_order":
        write_deliverables_list(out_dir)
        return 0

    # ---- Wave1 selection ----
    if not done("wave1_selection"):
        mark_stage(out_dir, state, "wave1_selection", "running")
        if args.skip_gpu:
            mark_stage(out_dir, state, "wave1_selection", "done", "skipped_gpu")
        else:
            ok = run_with_heal(
                out_dir,
                state,
                "wave1_selection",
                [py, matched, "--out_dir", str(out_dir), "--mode", "ladder_selection", "--model", "all"],
                "wave1_selection.log",
            )
            if not ok:
                return 2
            mark_stage(out_dir, state, "wave1_selection", "done")
    if args.stop_after == "wave1_selection":
        write_deliverables_list(out_dir)
        return 0

    # ---- Wave1 finalize (link reuse + tables) ----
    if not done("wave1_finalize"):
        mark_stage(out_dir, state, "wave1_finalize", "running")
        ok = run_with_heal(
            out_dir,
            state,
            "wave1_finalize",
            [py, matched, "--out_dir", str(out_dir), "--mode", "finalize", "--model", "all"],
            "wave1_finalize.log",
        )
        if not ok:
            return 2
        mark_stage(out_dir, state, "wave1_finalize", "done")
    if args.stop_after == "wave1_finalize":
        write_deliverables_list(out_dir)
        return 0

    # ---- Wave2 evidence (CPU) ----
    if not done("wave2_evidence"):
        mark_stage(out_dir, state, "wave2_evidence", "running")
        rc = run_cmd(
            [py, evidence, "--out_dir", str(out_dir)],
            out_dir / "logs" / "wave2_evidence.log",
        )
        if rc != 0:
            hard_stop(out_dir, state, "WAVE2_FAILED", {"rc": rc})
            return 2
        mark_stage(out_dir, state, "wave2_evidence", "done")
    if args.stop_after == "wave2_evidence":
        write_deliverables_list(out_dir)
        return 0

    # ---- Wave3 full-context ----
    if not done("wave3_fullctx"):
        mark_stage(out_dir, state, "wave3_fullctx", "running")
        if args.skip_gpu:
            mark_stage(out_dir, state, "wave3_fullctx", "done", "skipped_gpu")
        else:
            ok = run_with_heal(
                out_dir,
                state,
                "wave3_fullctx",
                [py, fullctx, "--out_dir", str(out_dir)],
                "wave3_fullctx.log",
            )
            if not ok:
                return 2
            mark_stage(out_dir, state, "wave3_fullctx", "done")
    if args.stop_after == "wave3_fullctx":
        write_deliverables_list(out_dir)
        return 0

    # ---- Wave3 efficiency ----
    if not done("wave3_efficiency"):
        mark_stage(out_dir, state, "wave3_efficiency", "running")
        if args.skip_gpu:
            mark_stage(out_dir, state, "wave3_efficiency", "done", "skipped_gpu")
        else:
            ok = run_with_heal(
                out_dir,
                state,
                "wave3_efficiency",
                [py, efficiency, "--out_dir", str(out_dir)],
                "wave3_efficiency.log",
            )
            if not ok:
                return 2
            mark_stage(out_dir, state, "wave3_efficiency", "done")
    if args.stop_after == "wave3_efficiency":
        write_deliverables_list(out_dir)
        return 0

    # ---- Finalize ----
    mark_stage(out_dir, state, "finalize", "running")
    state["pipeline_status"] = "COMPLETE"
    mark_stage(out_dir, state, "finalize", "done", "all stages complete")
    write_text(out_dir / "PIPELINE_COMPLETE.DONE", f"{utc_now()}\n")
    # Write checklist after STAGE_FINALIZE.DONE exists so self-audit is accurate.
    write_deliverables_list(out_dir)
    print(f"[gapfill] COMPLETE out_dir={out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        out = Path(".")
        if "--out_dir" in sys.argv:
            out = Path(sys.argv[sys.argv.index("--out_dir") + 1])
        write_text(out / "PIPELINE_FATAL.md", f"{exc}\n\n{traceback.format_exc()}")
        raise
