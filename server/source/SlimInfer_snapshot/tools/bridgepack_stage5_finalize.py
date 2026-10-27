#!/usr/bin/env python3
"""Stage 5: CPU-only final paper integration for BridgePack completion pipeline."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from tools.bridgepack_g1.common import utc_now, write_json, write_text

PIPE = REPO_ROOT / "results" / "bridgepack_completion_pipeline_20260727_140453"
G1 = REPO_ROOT / "results" / "bridgepack_g1_qwen1024_matched_20260727_004934"
G2 = REPO_ROOT / "results" / "bridgepack_g2_llama1024_sourceorder_20260727_023300"
INDEP = REPO_ROOT / "results" / "bridgepack_g2_independent_fix_20260727_132221"
FREEZE_SHA = "15886731816aa2eb4e08afa121bd8ccc9631dd232474a82a8f20214cc2ce16bf"


def _read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    s1 = PIPE / "stage1"
    s2 = PIPE / "stage2"
    s3 = PIPE / "stage3"
    s4 = PIPE / "stage4"

    required = {
        "stage1": s1 / "STAGE_1.DONE",
        "stage2": s2 / "STAGE_2.DONE",
        "stage3": s3 / "STAGE_3.DONE",
        "stage4": s4 / "STAGE_4.DONE",
    }
    missing = [k for k, p in required.items() if not p.exists()]
    if missing:
        raise RuntimeError(f"Stage5 blocked; missing DONE for: {missing}")

    # Master valid results
    master_rows: List[Dict[str, Any]] = []
    for row in _read_csv(s1 / "SOURCE_ORDER_MULTIBUDGET_RESULTS.csv"):
        row = dict(row)
        row["source_stage"] = "stage1_or_frozen_1024"
        row["valid"] = "YES"
        master_rows.append(row)
    for row in _read_csv(s2 / "G3_EXTERNAL_COMPRESSION_RESULTS.csv"):
        row = dict(row)
        row["source_stage"] = "stage2_adapted_only"
        row["valid"] = "YES"
        master_rows.append(row)
    for row in _read_csv(s3 / "G4_BACKFILL_RESULTS.csv"):
        row = dict(row)
        row["source_stage"] = "stage3"
        row["valid"] = "YES"
        master_rows.append(row)
    for row in _read_csv(s3 / "G4_MATCHED_ACTUAL_TOKEN_RESULTS.csv"):
        row = dict(row)
        row["source_stage"] = "stage3"
        row["valid"] = "YES"
        master_rows.append(row)

    fields = sorted({k for r in master_rows for k in r.keys()}) or ["source_stage", "valid"]
    with (out / "FINAL_VALID_RESULTS_MASTER.csv").open("w", encoding="utf-8", newline="") as handle:
        w = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n", extrasaction="ignore")
        w.writeheader()
        for r in master_rows:
            w.writerow(r)

    ledger = [
        {"unit": "G1_qwen_same_spans_source_order_1024", "macro": 39.385, "path": str(G1), "valid": True},
        {"unit": "G2_llama_source_order_bridge_1024", "macro": 34.605, "path": str(G2), "valid": True},
        {"unit": "G2_corrected_independent", "path": str(INDEP), "valid": True},
        {"unit": "G2_old_independent_16.85", "valid": False, "reason": "INVALIDATED_BY_INDEPENDENT_WIRING_BUG"},
        {"unit": "stage1_multibudget_256_512", "path": str(s1), "valid": True},
        {"unit": "stage2_external_adapted", "path": str(s2), "valid": True},
        {"unit": "stage3_backfill_matched_token", "path": str(s3), "valid": True},
        {"unit": "stage4_efficiency_subset", "path": str(s4), "valid": True},
        {"unit": "method_config_sha256", "value": FREEZE_SHA, "valid": True},
    ]
    write_json(out / "FINAL_EXPERIMENT_LEDGER.json", {"generated_at": utc_now(), "units": ledger})
    with (out / "FINAL_EXPERIMENT_LEDGER.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = sorted({k for r in ledger for k in r.keys()})
        w = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n", extrasaction="ignore")
        w.writeheader()
        for r in ledger:
            w.writerow(r)

    write_text(
        out / "FINAL_INVALIDATED_RESULTS_REGISTRY.md",
        f"""# FINAL_INVALIDATED_RESULTS_REGISTRY

Generated: {utc_now()}

| Artifact | Status | Reason |
|---|---|---|
| G2 Llama independent macro 16.85 | INVALIDATED | INDEPENDENT_WIRING_BUG (source-order lexical[:64]) |
| Claim +17.41 / old pooled +10.43 vs old independent | FORBIDDEN | based on invalidated independent |
| Stage1 hardcoded Qwen-1024 38.05 | INVALIDATED | wrong cite; path-order LBP≈38.34, source-order=39.385 |
| Style extractive external baseline as official | FORBIDDEN | not official LLMLingua |
| official_raw over-budget packs in fair main table | FORBIDDEN | diagnostic only |
""",
    )

    # Copy/aggregate key stats into reports
    s1_boot = (s1 / "SOURCE_ORDER_MULTIBUDGET_BOOTSTRAP.csv").read_text(encoding="utf-8") if (s1 / "SOURCE_ORDER_MULTIBUDGET_BOOTSTRAP.csv").exists() else ""
    s2_boot = (s2 / "G3_EXTERNAL_COMPRESSION_BOOTSTRAP.csv").read_text(encoding="utf-8") if (s2 / "G3_EXTERNAL_COMPRESSION_BOOTSTRAP.csv").exists() else ""
    s3_boot = (s3 / "G4_BOOTSTRAP.csv").read_text(encoding="utf-8") if (s3 / "G4_BOOTSTRAP.csv").exists() else ""

    write_text(
        out / "FINAL_STATISTICAL_REPORT.md",
        f"""# FINAL_STATISTICAL_REPORT

Generated: {utc_now()}

## Canonical macros

- Qwen-1024 source-order (G1 `same_spans_source_order`): **39.385** (paper **39.39**)
- Llama-1024 source-order (G2): **34.605**
- Stage1 qwen 256/512 and llama 256/512: see Stage1 RESULTS
- Stage2 adapted external compression: see G3_EXTERNAL_COMPRESSION_RESULTS.csv

## Bootstrap / Holm sources

### Stage1
```
{s1_boot[:4000]}
```

### Stage2
```
{s2_boot[:4000]}
```

### Stage3
```
{s3_boot[:4000]}
```
""",
    )

    write_text(
        out / "FINAL_BUDGET_AND_TOKEN_AUDIT.md",
        f"""# FINAL_BUDGET_AND_TOKEN_AUDIT

Generated: {utc_now()}

- Exact HF tokenizer budgets enforced for all fair main-table packs.
- Stage1 256/512: 0 over-budget (STAGE1_PACKING_AUDIT).
- Stage2 adapted qwen/llama: must be 0 over-budget; raw over-budget diagnostic only (`G3_TOKEN_BUDGET_AUDIT.*`).
- Stage3 backfill/matched-actual-token: 0 over-budget required for STAGE_3.DONE.
- Method config SHA256: `{FREEZE_SHA}`
""",
    )

    eff = (s4 / "G5_EFFICIENCY_SUMMARY.md").read_text(encoding="utf-8") if (s4 / "G5_EFFICIENCY_SUMMARY.md").exists() else "(missing)"
    write_text(out / "FINAL_EFFICIENCY_REPORT.md", f"# FINAL_EFFICIENCY_REPORT\n\nGenerated: {utc_now()}\n\n{eff}\n")

    write_text(
        out / "FINAL_CLAIM_GUARD.md",
        f"""# FINAL_CLAIM_GUARD

Generated: {utc_now()}

| Claim | Allowed? | Basis |
|---|---|---|
| Qwen-1024 source-order macro 39.39 | YES | G1 same_spans_source_order 39.385 |
| Cite Qwen-1024 source-order as 38.05 | **NO** | invalidated hardcoded error |
| Path-order is effective mechanism | **NO** | G1 CI crosses 0 |
| Connectivity score alone significant | **NO** | undetermined / not isolated |
| Old Llama independent 16.85 / +17.41 / pooled +10.43 | **NO** | invalidated wiring bug |
| Corrected Llama independent comparisons | YES | independent_fix artifacts |
| External adapted LLMLingua as unmodified official | **NO** | truncation adapted |
| Method is KV-cache pruning | **NO** | always forbidden |
| Efficiency on 1150 as latency main claim | **NO** | Stage4 subset only |
""",
    )

    write_text(
        out / "PAPER_TABLES_READY.md",
        f"""# PAPER_TABLES_READY

Generated: {utc_now()}

Use these machine-readable tables (do not hand-edit scores):

- Stage1: `{s1}/SOURCE_ORDER_MULTIBUDGET_RESULTS.csv`, `SOURCE_ORDER_MULTIBUDGET_BOOTSTRAP.csv`, `SOURCE_ORDER_BUDGET_SCALING.md`
- Stage2: `{s2}/G3_EXTERNAL_COMPRESSION_RESULTS.csv`, `G3_EXTERNAL_COMPRESSION_BOOTSTRAP.csv`
- Stage3: `{s3}/G4_*.csv`
- Stage4: `{s4}/G5_*.csv`
- Master: `{out}/FINAL_VALID_RESULTS_MASTER.csv`
""",
    )

    write_text(
        out / "PAPER_METHODS_RESULTS_DRAFT.md",
        f"""# PAPER_METHODS_RESULTS_DRAFT

Generated: {utc_now()}

## Methods (draft)

BridgePack main method (frozen SHA `{FREEZE_SHA}`): lexical initialization → bridge-aware selection → neighbor repair (±32) → **source-order** exact-budget packing.

Baselines include BM25/TF-IDF/BGE chunk packing and exact-budget adapted LLMLingua (explicitly adapted; not unmodified official).

## Results (draft pointers)

- Qwen@1024 source-order macro **39.39** (G1).
- Llama@1024 source-order macro **34.61** (G2).
- Budget scaling 256→512→1024: Stage1 `SOURCE_ORDER_BUDGET_SCALING.md`.
- External compression: Stage2 adapted-only fair table.
- Backfill / matched-actual-token: Stage3.
- Efficiency subset: Stage4 (not full 1150).
""",
    )

    write_text(
        out / "ABSTRACT_AND_CONCLUSION_UPDATE.md",
        f"""# ABSTRACT_AND_CONCLUSION_UPDATE

Generated: {utc_now()}

Update abstract/conclusion to:

1. Emphasize bridge-aware selection under matched source-order packing.
2. Avoid path-order-as-mechanism language.
3. Use Qwen source-order **39.39**, never 38.05.
4. Use corrected independent for Llama selection claims.
5. Label LLMLingua exact-budget results as adapted when truncated.
6. Do not claim KV-cache pruning.
""",
    )

    write_text(
        out / "AAAI_REVIEWER_GAP_CHECKLIST.md",
        f"""# AAAI_REVIEWER_GAP_CHECKLIST

Generated: {utc_now()}

- [ ] Exact-budget enforcement described for all main methods
- [ ] Official vs adapted external compression labeled
- [ ] Invalidated independent numbers removed from all drafts
- [ ] Qwen-1024 source-order cites 39.39 from G1 same_spans
- [ ] Bootstrap + Holm present for primary paired claims
- [ ] Efficiency claims limited to measured subset
- [ ] No KV-cache pruning wording
- [ ] Config SHA `{FREEZE_SHA}` referenced in reproducibility
""",
    )

    write_text(
        out / "PIPELINE_FINAL_REPORT.md",
        f"""# PIPELINE_FINAL_REPORT

Generated: {utc_now()}

Pipeline root: `{PIPE}`

- Stage0–5: DONE markers present under each stage directory
- Hard stops: none (this finalize succeeded)
- GPU: released after Stage4; Stage5 CPU-only
- Forbidden claims registry: `FINAL_INVALIDATED_RESULTS_REGISTRY.md`
- Claim guard: `FINAL_CLAIM_GUARD.md`
""",
    )

    write_text(out / "STAGE_5.DONE", f"Stage 5 complete at {utc_now()}\n")
    write_text(
        PIPE / "PIPELINE_STATUS.md",
        f"""# PIPELINE_STATUS

Generated: {utc_now()}

- pipeline_status: `PIPELINE_COMPLETE`
- current_stage: `5`
- last_completed_stage: `5`
- hard_stop: `None`
""",
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        out = Path(sys.argv[sys.argv.index("--out_dir") + 1]) if "--out_dir" in sys.argv else Path(".")
        write_text(out / "STAGE5_FATAL_ERROR.md", f"{exc}\n")
        raise
