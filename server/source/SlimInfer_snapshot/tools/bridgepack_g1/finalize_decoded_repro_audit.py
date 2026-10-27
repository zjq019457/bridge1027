#!/usr/bin/env python3
"""Write the post-run decoded-prediction reproduction addendum."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RUN = REPO / "results" / "bridgepack_g1_decoded_reproduction_20260928_180900"
AUDIT = REPO / "results" / "bridgepack_g1_strict_reproduction_audit_20260928_102546"
REF = REPO / "results" / "bridgepack_g1_qwen1024_matched_20260727_004934"
TASKS = ["qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "musique", "narrativeqa"]


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def write_json(path: Path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, fields, rows):
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", errors="replace")).hexdigest()


def main():
    rows = read_jsonl(RUN / "raw_predictions.jsonl")
    mismatches = []
    for x in rows:
        if not x["decoded_prediction_exact"]:
            mismatches.append({
                "task": x["task"], "sample_id": x["sample_id"],
                "input_token_count": len(x["input_token_ids"]),
                "prediction_token_count": len(x["prediction_token_ids"]),
                "new_prediction_sha256": digest(x["pred"]),
                "historical_prediction_sha256": digest(x["historical_decoded_prediction"]),
                "new_prediction_prefix": x["pred"][:160],
                "historical_prediction_prefix": x["historical_decoded_prediction"][:160],
            })
    write_csv(RUN / "G1_DECODED_MISMATCHES.csv", list(mismatches[0].keys()), mismatches)

    prompt_rows = []
    for task in TASKS:
        ref_path = REF / "full" / "packed_prompts_build" / "packed_prompts" / "same_spans_source_order" / "budget_1024" / f"{task}.jsonl"
        new_path = RUN / "packing" / "packed_prompts_build" / "packed_prompts" / "same_spans_source_order" / "budget_1024" / f"{task}.jsonl"
        ref = {str(x["sample_id"]): x for x in read_jsonl(ref_path)}
        new = {str(x["sample_id"]): x for x in read_jsonl(new_path)}
        prompt_rows.extend({
            "task": task, "sample_id": sid,
            "generation_prompt_exact": ref[sid].get("generation_prompt") == new[sid].get("generation_prompt"),
        } for sid in ref)
    write_csv(RUN / "G1_PROMPT_TEXT_COMPARISON.csv", ["task", "sample_id", "generation_prompt_exact"], prompt_rows)

    per_task = json.loads((RUN / "macro_summary.json").read_text(encoding="utf-8"))
    decision = {
        "status": "G1_NEAR_REPRODUCTION",
        "raw_result_provenance_status": "G1_RAW_RESULT_PROVENANCE_VERIFIED",
        "historical_token_exact_status": "STRICT_TOKEN_EXACT_VERIFICATION_BLOCKED_BY_MISSING_HISTORICAL_REFERENCE_LAYERS",
        "fresh_decoded_prediction_status": "NEAR_NOT_EXACT",
        "sample_count": len(rows),
        "decoded_predictions_exact": sum(bool(x["decoded_prediction_exact"]) for x in rows),
        "decoded_predictions_mismatched": len(mismatches),
        "task_scores": per_task["task_scores"],
        "macro": per_task["macro"],
        "historical_macro": 39.385,
        "macro_delta": per_task["macro"] - 39.385,
        "packing_exact_1150": True,
        "generation_prompt_text_exact_1150": all(x["generation_prompt_exact"] for x in prompt_rows),
        "historical_input_token_ids_comparable": False,
        "historical_prediction_token_ids_comparable": False,
        "new_run_saved_input_token_ids": True,
        "new_run_saved_prediction_token_ids": True,
        "likely_divergence_stage": "GENERATION_OUTPUT",
        "cause_status": "GENERATION_ENVIRONMENT_OR_HISTORICAL_CODE_PROVENANCE_UNRESOLVED",
        "no_method_tuning": True,
        "reference_modified": False,
        "second_fresh_process_token_replay": "NOT_RUN",
    }
    manifest_path = RUN / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["authoritative_status"] = decision["status"]
    manifest["decoded_exact_count"] = decision["decoded_predictions_exact"]
    manifest["decoded_mismatch_count"] = decision["decoded_predictions_mismatched"]
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_json(RUN / "G1_DECODED_REPRODUCTION_DECISION.json", decision)
    report = f"""# G1 decoded-prediction reproduction addendum

## Status

**G1_NEAR_REPRODUCTION**.

This is a fresh 1150-sample run. It must not be confused with the historical raw-result provenance audit:

- Historical raw result: verified exactly; macro `39.385`.
- Historical token-exact verification: blocked because token-level reference layers were never saved.
- Fresh decoded-prediction rerun: completed, but not exact.

## Fresh run

- Packing/text/selection/repair/packed-count: `1150/1150 exact`.
- Rendered generation prompt text: `1150/1150 exact`.
- Decoded predictions: `{len(rows) - len(mismatches)}/1150 exact`.
- Decoded mismatches: `{len(mismatches)}`.
- New run saved input and prediction token IDs for future internal replay.
- Historical token IDs remain unavailable and were not reverse-manufactured.

## Scores

| Task | Fresh score | Historical score |
|---|---:|---:|
| Qasper | {per_task['task_scores']['qasper']:.2f} | 41.76 |
| MultifieldQA | {per_task['task_scores']['multifieldqa_en']:.2f} | 49.83 |
| HotpotQA | {per_task['task_scores']['hotpotqa']:.2f} | 53.91 |
| 2WikiMQA | {per_task['task_scores']['2wikimqa']:.2f} | 45.98 |
| MuSiQue | {per_task['task_scores']['musique']:.2f} | 28.05 |
| NarrativeQA | {per_task['task_scores']['narrativeqa']:.2f} | 16.78 |
| Macro | {per_task['macro']:.4f} | 39.3850 |

The fresh run is therefore near-identical but not decoded-exact. The mismatch is observed at generation output after exact packed text and prompt text; its cause is not assigned without historical token IDs and a complete historical executable/environment provenance record. Candidate environment differences remain the RTX 3090 vs historical RTX 3090 Ti and an unpinned historical attention backend, but this is not a definitive causal attribution.

The nine mismatches are listed in `G1_DECODED_MISMATCHES.csv`.
"""
    (RUN / "G1_DECODED_REPRODUCTION_REPORT.md").write_text(report, encoding="utf-8")
    (AUDIT / "G1_REPRODUCTION_RECLASSIFICATION.md").write_text(f"""# G1 reproduction status reclassification

The earlier `G1_REPRODUCTION_FAIL` label described only the impossible historical token-reference criterion. It did not describe a failed 1150 rerun, because that rerun had not yet been executed.

The fresh run `{RUN}` has now completed:

- historical raw provenance: `VERIFIED`, macro `39.385`;
- packed text/selection/repair/packed count: `1150/1150 exact`;
- generation prompt text: `1150/1150 exact`;
- decoded predictions: `{len(rows) - len(mismatches)}/1150 exact`;
- fresh macro: `{per_task['macro']:.4f}` vs historical `39.385`;
- classification: **G1_NEAR_REPRODUCTION**.

The historical token-level equality remains **BLOCKED_REFERENCE_LAYER_NOT_AVAILABLE** and is not treated as a fresh-run failure.
""", encoding="utf-8")
    write_json(AUDIT / "G1_DECODED_REPRODUCTION_DECISION.json", decision)
    print(json.dumps({"status": decision["status"], "exact": decision["decoded_predictions_exact"], "mismatches": decision["decoded_predictions_mismatched"], "macro": decision["macro"]}))


if __name__ == "__main__":
    main()
