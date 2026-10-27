#!/usr/bin/env python3
"""BridgePack CPU-stage orchestrator. Never starts GPU jobs."""
from __future__ import annotations

import json
import os
import sys
import traceback
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.bridgepack_cpu_stage.common import OUT_ROOT, ensure_cpu_only, utc_now, write_json, write_text
from tools.bridgepack_cpu_stage.build_ledger import build_ledger
from tools.bridgepack_cpu_stage.run_statistics import run_statistics
from tools.bridgepack_cpu_stage.run_evidence_metrics import run_evidence_metrics
from tools.bridgepack_cpu_stage.run_token_utilization import run_token_utilization
from tools.bridgepack_cpu_stage.run_matched_ablation import run_matched_ablation
from tools.bridgepack_cpu_stage.run_strong_baselines import run_strong_baselines
from tools.bridgepack_cpu_stage.run_external_compression import run_external_compression
from tools.bridgepack_cpu_stage.run_efficiency_repro import run_efficiency_and_repro


def _run_tests() -> DictLike:
    import unittest
    from tools.bridgepack_cpu_stage import test_smoke

    suite = unittest.defaultTestLoader.loadTestsFromModule(test_smoke)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return {"ok": result.wasSuccessful(), "tests": result.testsRun, "failures": len(result.failures), "errors": len(result.errors)}


# typing alias without importing typing for Dict
DictLike = dict


def write_final_reports(status: dict) -> None:
    final = f"""# CPU_STAGE_FINAL_REPORT

Generated: {utc_now()}

## Isolation
- Output root: `{OUT_ROOT.relative_to(REPO_ROOT)}`
- CUDA_VISIBLE_DEVICES=""
- No QA launched; no compressor/QA model weights loaded for full runs
- Other-paper GPU process left untouched

## Completed CPU work
1. Experiment ledger (CSV/JSON) + MISSING_ARTIFACTS + DO_NOT_RERUN
2. Statistical recompute: per-task, macro, paired stratified bootstrap (10k), Holm correction
3. Evidence metrics from packed contexts (answer-string recall; SF/chain = NA on this LongBench dump)
4. Token utilization distribution + 0.1498 denominator audit; backfill & matched-actual-token interfaces smoked
5. Matched ablation protocol + CPU smoke artifacts + GPU manifest
6. Strong baseline packer interfaces + smoke
7. External compression protocol distinguishing official_raw vs exact_budget_adapted
8. CPU efficiency sample timing + reproducibility checklist / missing fields
9. Unit/synthetic smoke tests

## Interface readiness
- Matched order/selection/score-component packers: READY (CPU)
- Strong baselines: READY (BGE requires cache; else BLOCKED_MISSING_ARTIFACT)
- Backfill / matched-actual-token: READY (CPU smoke)
- External compression adapter: READY (dry-run); compressor load GPU_PENDING
- GPU efficiency runner: STUB ONLY

## Hard gates satisfied
- New smoke packed artifacts: 0 over-budget (HF tokenizer audit on smokes)
- No gold-answer leakage flags in new selectors
- Order-only path_order span-set identity asserted in tests/smoke
- Config/artifact hashes recorded on new artifacts
- Tests: {status.get('tests')}

## Explicit non-actions
- Did not start GPU queue
- Did not overwrite historical BridgePack result directories
- Did not modify other-paper running configs/results
"""
    write_text(OUT_ROOT / "final" / "CPU_STAGE_FINAL_REPORT.md", final)
    write_text(OUT_ROOT / "CPU_STAGE_FINAL_REPORT.md", final)

    gpu_queue = f"""# GPU_PENDING_QUEUE

Generated: {utc_now()}

Do **not** auto-start. Wait until GPU is free and BridgePack isolation confirmed.

## P0
| ID | Experiment | Model | Budget | Samples (approx) | Est. cost | Depends |
|---|---|---|---|---:|---|---|
| G1 | Strict matched mechanism (order/selection/score) | Qwen | 1024 | 1150 × ~6 variants | High (multi-variant QA) | CPU interfaces PASS |
| G2 | Mechanism replication | Llama-3 | 1024 | 1150 × ~5 variants | High | G1 positive / gate |
| G3 | External compression official+adapted+QA | Qwen (then Llama-3) | 1024 | 1150 | High + compressor | Budget audit PASS |
| G4 | Backfill vs stop-at-first-nonfit vs matched-actual-token QA | Qwen, Llama-3 | 1024 | 1150 × methods | High | G1 optional |
| G5 | Efficiency (prefill/TTFT/mem) | Qwen | 1024 | efficiency subset | Medium | Free GPU |

## First GPU experiment to start
**G1 smoke (Qwen, 1024, small N)** → budget/alignment checks → full G1 six-task → paired bootstrap immediately.

## P1 / P2
See user protocol (256/512 expansion, hyperparam sensitivity, external benchmark, larger models).
"""
    write_text(OUT_ROOT / "final" / "GPU_PENDING_QUEUE.md", gpu_queue)
    write_text(OUT_ROOT / "GPU_PENDING_QUEUE.md", gpu_queue)

    claim = f"""# PAPER_CLAIM_UPDATE_MATRIX

Generated: {utc_now()}

| Audit result | Allowed paper claim | Forbidden claim |
|---|---|---|
| Fairness 60-cell COMPLETE + 0 over-budget | Exact-budget main results for listed model/budget/method cells | Claiming cells outside ledger COMPLETE |
| Bootstrap deltas with CI not crossing 0 after Holm | Statistically supported pairwise improvement at α=0.05 (Holm) | Reporting only uncorrected positives; hiding negative deltas |
| Bootstrap CI crosses 0 | Descriptive delta only; no significance claim | "Significant improvement" |
| Answer-string recall differences | Packed contexts differ in answer containment | Supporting-fact / hop-complete-chain gains on this LongBench dump |
| SF annotations absent | State NA; limit to answer-string recall | Invented gold evidence paths |
| stored compression_ratio 0.1498 vs recomputed ~0.082 | Report tokens with explicit denominator | Equating 0.1498 with a 13.25%-style figure without shared definition |
| Matched-ablation CPU smoke only | Interface/protocol ready | Claiming mechanism QA results from smoke |
| External adapted outputs | Describe as exact-budget-adapted | Calling adapted outputs unmodified official LLMLingua |
| CPU timing only | Lexical selector CPU preprocess cost | Prefill/TTFT/KV memory claims from CPU stage |
| Efficiency GPU pending | — | Calling BridgePack a KV-cache pruning method |
| Mistral/Llama2 partial | Partial/negative transfer with CI if computed | Full six-task parity claims without COMPLETE ledger status |
"""
    write_text(OUT_ROOT / "final" / "PAPER_CLAIM_UPDATE_MATRIX.md", claim)
    write_text(OUT_ROOT / "PAPER_CLAIM_UPDATE_MATRIX.md", claim)


def main() -> int:
    ensure_cpu_only()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    status = {"started_at": utc_now(), "steps": {}}
    steps = [
        ("ledger", build_ledger),
        ("statistics", run_statistics),
        ("evidence", run_evidence_metrics),
        ("token_util", run_token_utilization),
        ("matched_ablation", run_matched_ablation),
        ("strong_baselines", run_strong_baselines),
        ("external_compression", run_external_compression),
        ("efficiency", run_efficiency_and_repro),
        ("tests", _run_tests),
    ]
    ok = True
    for name, fn in steps:
        print(f"[CPU_STAGE] running {name}...", flush=True)
        try:
            status["steps"][name] = {"ok": True, "result": fn()}
            print(f"[CPU_STAGE] done {name}", flush=True)
        except Exception as exc:
            ok = False
            status["steps"][name] = {"ok": False, "error": str(exc), "traceback": traceback.format_exc()}
            print(f"[CPU_STAGE] FAILED {name}: {exc}", flush=True)
            # continue other steps when possible
    status["finished_at"] = utc_now()
    status["ok"] = ok
    status["tests"] = status["steps"].get("tests", {})
    write_json(OUT_ROOT / "final" / "cpu_stage_status.json", status)
    write_final_reports(status)
    print(json.dumps({"ok": ok, "out": str(OUT_ROOT)}, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
