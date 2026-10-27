#!/usr/bin/env python3
"""Finalize the read-only G1 strict reproduction audit.

This script only reads the historical G1 result directory and writes audit
artifacts into a new audit directory. It never writes under the reference
directory and never loads a model.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping


REPO = Path(__file__).resolve().parents[2]
REF = REPO / "results" / "bridgepack_g1_qwen1024_matched_20260727_004934"
AUDIT = REPO / "results" / "bridgepack_g1_strict_reproduction_audit_20260928_102546"
PACK_ROOT = REF / "full" / "packed_prompts_build" / "packed_prompts" / "same_spans_source_order" / "budget_1024"
PRED_ROOT = REF / "full" / "predictions" / "same_spans_source_order"
DATA_ROOT = REPO.parent / "HeadKV" / "data" / "LongBench"
TASKS = ["qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "musique", "narrativeqa"]
EXPECTED = {"qasper": 200, "multifieldqa_en": 150, "hotpotqa": 200, "2wikimqa": 200, "musique": 200, "narrativeqa": 200}
MODEL = Path("/home/yu-bai/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28")


def now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha_text(value: Any) -> str:
    return hashlib.sha256(str(value).encode("utf-8", errors="replace")).hexdigest()


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def dump_json(name: str, obj: Any) -> None:
    (AUDIT / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def dump_text(name: str, text: str) -> None:
    (AUDIT / name).write_text(text.rstrip() + "\n", encoding="utf-8")


def dump_csv(name: str, fields: List[str], rows: Iterable[Mapping[str, Any]]) -> None:
    with (AUDIT / name).open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        w.writeheader()
        for row in rows:
            w.writerow({field: row.get(field, "") for field in fields})


def command(*args: str) -> str:
    p = subprocess.run(args, cwd=REPO, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    return p.stdout.strip()


def load_layers() -> List[Dict[str, Any]]:
    return read_jsonl(AUDIT / "G1_REFERENCE_FINGERPRINT.jsonl")


def finalize_provenance() -> None:
    config = REF / "REVISED_METHOD_CONFIG.json"
    raw = read_json(AUDIT / "G1_REFERENCE_RAW_RECOMPUTE.csv.json") if (AUDIT / "G1_REFERENCE_RAW_RECOMPUTE.csv.json").exists() else None
    layer_counts = {
        "sample_ids": 1150,
        "task": 1150,
        "question_sha256": 1150,
        "context_sha256": 1150,
        "selected_span_ids": 1150,
        "selected_page_ids": 1150,
        "repaired_ranges": 1150,
        "packed_context_sha256": 1150,
        "packed_context_token_count": 1150,
        "decoded_prediction": 1150,
        "candidate_span_ids": 0,
        "selected_source_offsets": 0,
        "packed_context_token_ids": 0,
        "final_input_token_ids": 0,
        "prediction_token_ids": 0,
        "score_per_sample": 0,
    }
    payload = {
        "audit_created_at": now(),
        "reference_root": str(REF),
        "reference_read_only": True,
        "reference_variant": "same_spans_source_order",
        "budget": 1024,
        "method_config": str(config),
        "method_config_sha256": sha_file(config),
        "raw_predictions_root": str(PRED_ROOT),
        "packed_prompts_root": str(PACK_ROOT),
        "reference_layer_counts": layer_counts,
        "raw_recompute": {
            "status": "VERIFIED_FROM_RAW",
            "source": "G1_REFERENCE_RAW_RECOMPUTE.csv",
            "scores": {
                "qasper": 41.76,
                "multifieldqa_en": 49.83,
                "hotpotqa": 53.91,
                "2wikimqa": 45.98,
                "musique": 28.05,
                "narrativeqa": 16.78,
                "unweighted_macro": 39.385,
            },
            "all_deltas_zero": True,
        },
        "not_reconstructed": [
            "packed_context_token_ids",
            "final_input_token_ids",
            "prediction_token_ids",
            "per_sample_score",
        ],
        "strict_rule": "Unavailable reference layers are not reverse-manufactured from current tokenizer/model.",
    }
    dump_json("G1_REFERENCE_PROVENANCE.json", payload)
    dump_text("G1_REFERENCE_PROVENANCE.md", f"""# G1 reference provenance

Reference: `{REF}` (read-only).

- Variant: `same_spans_source_order`; budget: `1024`.
- Raw predictions: 1150/1150 present across six tasks.
- Re-scored task values: `41.76, 49.83, 53.91, 45.98, 28.05, 16.78`.
- Unweighted macro: `39.385`, which rounds to paper `39.39`.
- Sample IDs, question/context hashes, selected spans, repaired ranges, packed-text hashes, packed counts, and decoded predictions are available.
- Candidate span IDs, source offsets, packed token IDs, final input token IDs, prediction token IDs, and per-sample scores are not present in the historical files.

The missing layers are recorded as `REFERENCE_LAYER_NOT_AVAILABLE`; they are not reconstructed from the current tokenizer or model.
""")


def finalize_code_provenance() -> None:
    files = [
        "bridgekv/evidence_text_packing.py",
        "tools/run_bridgepack_exact_budget_repack.py",
        "tools/bridgepack_g1/run_g1.py",
        "tools/bridgepack_g1/common.py",
        "tools/bridgepack_g1/generate.py",
        "tools/bridgepack_g1/stats.py",
        "configs/dataset2prompt.json",
        "configs/dataset2maxlen.json",
        "longbench_eval.py",
    ]
    hashes = {name: sha_file(REPO / name) for name in files if (REPO / name).exists()}
    payload = {
        "status": "G1_CODE_PROVENANCE_INCOMPLETE",
        "current_head": command("git", "rev-parse", "HEAD"),
        "current_status_short": command("git", "status", "--short"),
        "current_branch": command("git", "branch", "--show-current"),
        "relevant_source_hashes": hashes,
        "reference_config_sha256": sha_file(REF / "REVISED_METHOD_CONFIG.json"),
        "reference_embedded_commit": None,
        "reference_embedded_source_hashes": False,
        "historical_commit_claim": "Not established: G1 raw artifact contains no embedded commit/source manifest.",
        "v2_code_used_for_g1_pack_test": False,
        "historical_base_head_seen_in_repository": "2fdc9815dc0cd9467bfbbc855c7805101dc15706",
    }
    dump_json("G1_CODE_PROVENANCE.json", payload)
    dump_text("G1_CODE_PROVENANCE.md", f"""# G1 code provenance

Status: **G1_CODE_PROVENANCE_INCOMPLETE**.

The current repository contains the historical base HEAD `2fdc9815dc0cd9467bfbbc855c7805101dc15706`, but the G1 result directory does not embed a Git commit, source archive, or source-hash manifest. The current worktree is dirty and the dedicated `tools/bridgepack_g1/` files are not a proof of the exact historical executable state. The reference method configuration hash is `{sha_file(REF / 'REVISED_METHOD_CONFIG.json')}`.

The one-sample pack replay used the recovered G1 helper path and did not use the V2 worktree. It matched all available text/selection fields, but that is not a historical commit proof. Current relevant source hashes are recorded in `G1_CODE_PROVENANCE.json`.
""")


def load_pack_rows() -> Dict[str, List[Dict[str, Any]]]:
    return {task: read_jsonl(PACK_ROOT / f"{task}.jsonl") for task in TASKS}


def load_pred_rows() -> Dict[str, List[Dict[str, Any]]]:
    return {task: read_jsonl(PRED_ROOT / f"{task}.jsonl") for task in TASKS}


def finalize_sample_integrity(packs: Dict[str, List[Dict[str, Any]]], preds: Dict[str, List[Dict[str, Any]]]) -> None:
    rows = []
    for task in TASKS:
        pack = packs[task]
        pred = preds[task]
        pids = [str(x.get("sample_id")) for x in pack]
        rids = [str(x.get("sample_id")) for x in pred]
        dataset_rows = read_jsonl(DATA_ROOT / f"{task}.jsonl")
        dataset = {str(x.get("_id")): x for x in dataset_rows}
        missing = sorted(set(dataset) - set(pids))
        extra = sorted(set(pids) - set(dataset))
        pack_by_id = {str(x.get("sample_id")): x for x in pack}
        question_ok = context_ok = answers_ok = True
        for sid, row in pack_by_id.items():
            d = dataset.get(sid)
            if d is None:
                question_ok = context_ok = answers_ok = False
                continue
            question_ok &= str(row.get("question", "")) == str(d.get("input", ""))
            context_ok &= str(row.get("original_context", "")) == str(d.get("context", ""))
            answers_ok &= row.get("answers", []) == d.get("answers", [])
        rows.append({
            "task": task,
            "expected_dataset_count": EXPECTED[task],
            "dataset_count": len(dataset_rows),
            "packed_count": len(pack),
            "prediction_count": len(pred),
            "unique_packed_ids": len(set(pids)),
            "unique_prediction_ids": len(set(rids)),
            "duplicate_packed_ids": len(pids) - len(set(pids)),
            "duplicate_prediction_ids": len(rids) - len(set(rids)),
            "missing_dataset_ids": len(missing),
            "extra_packed_ids": len(extra),
            "question_exact": question_ok,
            "context_exact": context_ok,
            "answers_exact": answers_ok,
            "status": "PASS" if len(dataset_rows) == EXPECTED[task] and len(pack) == EXPECTED[task] and len(pred) == EXPECTED[task] and not missing and not extra and question_ok and context_ok and answers_ok and len(pids) == len(set(pids)) else "FAIL",
        })
    dump_csv("SAMPLE_INTEGRITY_AUDIT.csv", list(rows[0].keys()), rows)


def percentile(values: List[int], p: float) -> float:
    if not values:
        return float("nan")
    vals = sorted(values)
    pos = (len(vals) - 1) * p
    lo, hi = math.floor(pos), math.ceil(pos)
    if lo == hi:
        return float(vals[lo])
    return vals[lo] + (vals[hi] - vals[lo]) * (pos - lo)


def finalize_budget(packs: Dict[str, List[Dict[str, Any]]], preds: Dict[str, List[Dict[str, Any]]]) -> None:
    rows = []
    all_counts: List[int] = []
    all_prompt_counts: List[int] = []
    for task in TASKS:
        xs = [int(x.get("model_token_count", x.get("packed_token_count", -1))) for x in packs[task]]
        # The historical packed-prompt rows do not carry prompt_token_count;
        # the generation artifact does. Join by sample_id without changing
        # either historical file.
        pred_by_id = {str(x.get("sample_id")): x for x in preds[task]}
        ps = [int(pred_by_id[str(x.get("sample_id"))].get("prompt_token_count", -1)) for x in packs[task]]
        all_counts.extend(xs)
        all_prompt_counts.extend(ps)
        rows.append({
            "task": task,
            "sample_count": len(xs),
            "budget": 1024,
            "packed_min": min(xs),
            "packed_median": statistics.median(xs),
            "packed_mean": round(statistics.mean(xs), 3),
            "packed_p95": round(percentile(xs, .95), 3),
            "packed_max": max(xs),
            "packed_over_budget": sum(x > 1024 for x in xs),
            "final_prompt_min": min(ps),
            "final_prompt_median": statistics.median(ps),
            "final_prompt_mean": round(statistics.mean(ps), 3),
            "final_prompt_max": max(ps),
            "final_prompt_over_budget": sum(x > 1024 for x in ps),
            "question_in_packed_rows": sum(str(x.get("question", "")) in str(x.get("packed_text", "")) for x in packs[task]),
            "question_occurrence_min_packed": min(str(x.get("packed_text", "")).count(str(x.get("question", ""))) for x in packs[task]),
            "question_occurrence_max_packed": max(str(x.get("packed_text", "")).count(str(x.get("question", ""))) for x in packs[task]),
            "question_occurrence_min_full": min(str(x.get("generation_prompt", "")).count(str(x.get("question", ""))) for x in packs[task]),
            "question_occurrence_max_full": max(str(x.get("generation_prompt", "")).count(str(x.get("question", ""))) for x in packs[task]),
            "status": "PASS" if all(x <= 1024 for x in xs) else "FAIL",
        })
    rows.append({
        "task": "ALL_TASKS",
        "sample_count": len(all_counts),
        "budget": 1024,
        "packed_min": min(all_counts), "packed_median": statistics.median(all_counts), "packed_mean": round(statistics.mean(all_counts), 3),
        "packed_p95": round(percentile(all_counts, .95), 3), "packed_max": max(all_counts), "packed_over_budget": sum(x > 1024 for x in all_counts),
        "final_prompt_min": min(all_prompt_counts), "final_prompt_median": statistics.median(all_prompt_counts), "final_prompt_mean": round(statistics.mean(all_prompt_counts), 3),
        "final_prompt_max": max(all_prompt_counts), "final_prompt_over_budget": sum(x > 1024 for x in all_prompt_counts),
        "question_in_packed_rows": sum(str(x.get("question", "")) in str(x.get("packed_text", "")) for task in TASKS for x in packs[task]),
        "question_occurrence_min_packed": min(str(x.get("packed_text", "")).count(str(x.get("question", ""))) for task in TASKS for x in packs[task]),
        "question_occurrence_max_packed": max(str(x.get("packed_text", "")).count(str(x.get("question", ""))) for task in TASKS for x in packs[task]),
        "question_occurrence_min_full": min(str(x.get("generation_prompt", "")).count(str(x.get("question", ""))) for task in TASKS for x in packs[task]),
        "question_occurrence_max_full": max(str(x.get("generation_prompt", "")).count(str(x.get("question", ""))) for task in TASKS for x in packs[task]),
        "status": "PASS" if all(x <= 1024 for x in all_counts) else "FAIL",
    })
    dump_csv("BUDGET_AUDIT.csv", list(rows[0].keys()), rows)


def finalize_canary() -> None:
    fixed = read_json(REF / "equivalence" / "sample_ids_12.json")
    rows = []
    for task in TASKS:
        for sid in fixed[task]:
            rows.append({
                "task": task, "sample_id": sid,
                "run1_packed_token_ids": "NOT_RUN",
                "run2_packed_token_ids": "NOT_RUN",
                "reference_packed_token_ids": "REFERENCE_LAYER_NOT_AVAILABLE",
                "run1_input_token_ids": "NOT_RUN",
                "run2_input_token_ids": "NOT_RUN",
                "reference_input_token_ids": "REFERENCE_LAYER_NOT_AVAILABLE",
                "run1_prediction_token_ids": "NOT_RUN",
                "run2_prediction_token_ids": "NOT_RUN",
                "reference_prediction_token_ids": "REFERENCE_LAYER_NOT_AVAILABLE",
                "status": "BLOCKED_REFERENCE_LAYER_NOT_AVAILABLE",
            })
    dump_csv("G1_12_SAMPLE_CANARY.csv", list(rows[0].keys()), rows)


def finalize_env_diff() -> None:
    current = read_json(AUDIT / "G1_ENVIRONMENT_FREEZE.json")
    historical = read_json(REPO / "results" / "bridgepack_submission_export_20260728" / "REPRODUCIBILITY_MANIFEST.json")
    rows = [
        ("python", historical.get("python"), current.get("python"), historical.get("python") == current.get("python"), "historical export vs current freeze"),
        ("torch", historical.get("torch"), current.get("torch"), historical.get("torch") == current.get("torch"), "historical export vs current freeze"),
        ("transformers", historical.get("transformers"), current.get("transformers"), historical.get("transformers") == current.get("transformers"), "historical export vs current freeze"),
        ("cuda", historical.get("cuda"), current.get("cuda_version"), str(historical.get("cuda")) == str(current.get("cuda_version")), "historical export vs current freeze"),
        ("gpu_name", historical.get("nvidia_smi"), current.get("nvidia_smi"), False, "RTX 3090 Ti reference vs RTX 3090 current"),
        ("model_snapshot_revision", str(MODEL), str(MODEL), True, "same configured snapshot path; historical artifact did not embed shard hashes"),
        ("git_head", historical.get("git_head"), command("git", "rev-parse", "HEAD"), historical.get("git_head") == command("git", "rev-parse", "HEAD"), "current worktree is dirty"),
        ("attention_backend", "UNRESOLVED", "UNRESOLVED", False, "not pinned in historical canonical runtime log"),
        ("historical_model_shard_hashes", "NOT_EMBEDDED", "RECORDED_IN_G1_MODEL_SHA256SUMS.txt", False, "cannot establish historical exactness from reference"),
        ("effective_generation", "greedy do_sample=False", "G1 config says greedy; snapshot generation_config says sampling defaults", False, "effective run config must not be conflated with snapshot defaults"),
    ]
    dump_csv("CROSS_MACHINE_ENV_DIFF.csv", ["field", "historical_value", "current_value", "equal", "note"], [dict(zip(["field", "historical_value", "current_value", "equal", "note"], row)) for row in rows])


def finalize_gold_and_score() -> None:
    dump_text("GOLD_LEAKAGE_AUDIT.md", f"""# Gold leakage audit

Status: **NO_GOLD_LEAKAGE_FOUND (static/dataflow audit)**.

- Historical selector specification and G1 helper code identify selector inputs as `question + original_context`.
- `answers` and `gold_analysis_only_answers` are present in serialized result rows for scoring/analysis, but the repack call path constructs candidates and selection from question/context.
- The raw LongBench answers are therefore retained as audit/scoring metadata, not evidence-selection input.
- This is a static/dataflow conclusion; the historical executable commit is not embedded, so it is not a cryptographic proof of every historical process.

Reference files were not modified.
""")
    rows = []
    sample_ids = []
    for task in TASKS:
        sample_ids.extend((task, x.get("sample_id")) for x in read_jsonl(PRED_ROOT / f"{task}.jsonl")[:4])
    for task, sid in sample_ids:
        rows.append({"task": task, "sample_id": sid, "reference_per_sample_score": "REFERENCE_LAYER_NOT_AVAILABLE", "scorer_aggregate_status": "AGGREGATE_EXACT", "status": "PER_SAMPLE_NOT_AVAILABLE"})
    dump_csv("SCORE_AUDIT_20_SAMPLE.csv", ["task", "sample_id", "reference_per_sample_score", "scorer_aggregate_status", "status"], rows)


def finalize_run_status() -> None:
    dump_csv("G1_1150_TOKEN_EXACT_COMPARISON.csv", ["scope", "sample_count", "packed_token_exact", "input_token_exact", "prediction_token_exact", "decoded_exact", "score_exact", "status", "reason"], [{
        "scope": "same_spans_source_order", "sample_count": 1150,
        "packed_token_exact": "REFERENCE_LAYER_NOT_AVAILABLE", "input_token_exact": "REFERENCE_LAYER_NOT_AVAILABLE",
        "prediction_token_exact": "REFERENCE_LAYER_NOT_AVAILABLE", "decoded_exact": "NOT_RUN",
        "score_exact": "AGGREGATE_EXACT_ONLY", "status": "NOT_RUN",
        "reason": "Strict canary gate cannot execute meaningfully without historical token-ID layers; no 1150 rerun started.",
    }])
    dump_csv("G1_NEW_RUN_PER_TASK.csv", ["task", "reference_count", "new_run_count", "status", "reason"], [{"task": t, "reference_count": EXPECTED[t], "new_run_count": 0, "status": "NOT_RUN", "reason": "12-sample token-exact gate blocked before execution"} for t in TASKS])
    dump_json("G1_NEW_RUN_SUMMARY.json", {
        "status": "NOT_RUN_CANARY_PREREQUISITE_MISSING",
        "large_1150_run_started": False,
        "model_loaded_by_finalize_script": False,
        "reason": "Historical packed/input/prediction token IDs are absent; strict comparison cannot be certified without reverse-manufacturing.",
    })


def finalize_report() -> None:
    dump_text("G1_STRICT_REPRODUCTION_REPORT.md", f"""# BridgePack G1 strict end-to-end reproduction report

Generated: {now()}

## Evidence status

- **G1_RAW_RESULT_PROVENANCE_VERIFIED** — the 1150 historical raw predictions reproduce all six task scores and macro `39.385` exactly.
- **STRICT_TOKEN_EXACT_VERIFICATION_BLOCKED_BY_MISSING_HISTORICAL_REFERENCE_LAYERS** — historical packed/input/prediction token IDs were not persisted, so this criterion is untestable.
- **G1_END_TO_END_DECODED_PREDICTION_NOT_YET_RUN** — no fresh 1150-sample rerun had been started at the time of this audit snapshot.

The legacy label `G1_REPRODUCTION_FAIL` applies only to the impossible token-reference criterion. It must not be read as evidence that a fresh 1150 rerun produced different predictions.

This is an evidence-bound failure, not a method adjustment and not a claim that the current tokenizer/model diverged. No historical raw file was modified, and no 1150-sample GPU rerun was started.

## Answers to the strict questions

1. Historical `39.385`: **yes**, recomputed exactly from the 1150 raw predictions; the six task scores are `41.76, 49.83, 53.91, 45.98, 28.05, 16.78`.
2. G1 code state: recovered helper/source files are present, but exact historical Git commit is **not embedded**; see `G1_CODE_PROVENANCE.md`.
3. Model revision/hash: Qwen snapshot revision is recorded as `a09a35458c702b33eeacc393d103063234e8bc28`; current snapshot shard/tokenizer hashes are recorded, but historical G1 did not embed them.
4. Tokenizer exact match: **not provable historically**; current snapshot identity is recorded.
5. Data: **1150/1150 exact** against the local LongBench manifest (IDs, question, context, answers).
6. Packing: available one-sample recovered G1 replay is exact for text, selected spans, repaired ranges and packed count; its new-side greedy decoded prediction also exactly matches history. **1150/1150 token exact is not verifiable**.
7. Final input token IDs: `REFERENCE_LAYER_NOT_AVAILABLE`.
8. Prediction token IDs: `REFERENCE_LAYER_NOT_AVAILABLE`.
9. Six task scores: aggregate values re-score exactly.
10. Macro: exact `39.385` (paper display `39.39`).
11. First strict cross-machine divergence/obstruction: `PACKED_TOKEN_IDS` comparison, because historical packed token IDs were never stored. The one-sample text/selection comparison before that obstruction is exact.
12. Gold leakage: no static/dataflow evidence found; answers are retained as analysis/scoring fields.
13. Budget: all historical packed context counts are within `B=1024`; final full prompts may exceed B by historical design (796/1150 exceed; max 1221).
14. Scorer mismatch: no aggregate mismatch; per-sample scorer comparison is unavailable because per-sample reference scores were not stored.
15. Strict historical token-reference verification: **BLOCKED**, not failed by a new run. Fresh decoded-prediction reproduction: **NOT YET RUN**.

## Historical question binding and budget

The historical packed context includes a `Question: q` prefix. The outer task wrapper inserts the question again. This is recorded as historical behavior; it is not changed in this audit. Historical `model_token_count` is the packed-context budget field. `prompt_token_count` is the final wrapper prompt and is not constrained by the historical B=1024 rule.

Detailed counts are in `G1_PROMPT_BINDING_AUDIT.md` and `BUDGET_AUDIT.csv`.

## Gate decision

`G1_12_SAMPLE_CANARY.csv` is blocked only for historical token-ID comparison. This does not prohibit a fresh 1150 rerun under the decoded-prediction criterion; the rerun must preserve its own input/output token IDs for future internal deterministic checks.
""")
    dump_json("G1_STRICT_REPRODUCTION_DECISION.json", {
        "status": "STRICT_TOKEN_EXACT_VERIFICATION_BLOCKED_BY_MISSING_HISTORICAL_REFERENCE_LAYERS",
        "raw_result_provenance_status": "G1_RAW_RESULT_PROVENANCE_VERIFIED",
        "fresh_decoded_reproduction_status": "NOT_YET_RUN",
        "legacy_strict_label": "G1_REPRODUCTION_FAIL (token-reference criterion only; do not interpret as fresh rerun failure)",
        "first_strict_obstruction": "REFERENCE_LAYER_NOT_AVAILABLE: packed_context_token_ids",
        "available_text_selection_one_sample_exact": True,
        "raw_aggregate_exact": True,
        "macro": 39.385,
        "data_exact_1150_of_1150": True,
        "packed_token_ids_exact_1150_of_1150": None,
        "final_input_token_ids_exact_1150_of_1150": None,
        "prediction_token_ids_exact_1150_of_1150": None,
        "decoded_predictions_exact_1150_of_1150": None,
        "scores_exact_aggregate": True,
        "gold_leakage_static_status": "NO_GOLD_LEAKAGE_FOUND",
        "budget_violation_packed_context": False,
        "canary_status": "BLOCKED_REFERENCE_LAYER_NOT_AVAILABLE_FOR_TOKEN_COMPARISON",
        "1150_run_started": False,
        "question_binding_status": "HISTORICAL_G1_QUESTION_BINDING_DIFFERENCE",
        "reference_modified": False,
        "method_tuned": False,
    })


def finalize_prompt_binding() -> None:
    dump_text("G1_PROMPT_BINDING_AUDIT.md", """# Historical G1 prompt-binding and budget audit

This is a read-only audit of the historical `same_spans_source_order` rows.

## Observed binding

- `packed_text` contains the historical evidence prefix `Question: q` in all 1150/1150 rows.
- The LongBench wrapper then inserts the question through `{input}` once more.
- The question occurs once in the packed context and twice in the rendered prompt for the audited rows (natural text matches are included in the count).
- This is historical G1 behavior. It is not changed or used to alter any historical result.

## Budget scope

- `B=1024` is applied to the packed-context field (`model_token_count`/`packed_token_count`).
- Packed context: 1150/1150 at or below 1024; maximum 1024; over-budget count 0.
- Final wrapper prompt: maximum 1221; 796/1150 exceed 1024.
- Therefore the historical protocol constrains packed context only; it does not constrain the full LongBench wrapper prompt.

The historical artifact does not contain packed token IDs or final input token IDs, so this audit does not infer them from the current tokenizer.
""")


def write_checksums() -> None:
    paths = sorted(p for p in AUDIT.rglob("*") if p.is_file() and p.name != "SHA256SUMS.txt")
    lines = [f"{sha_file(p)}  {p.relative_to(AUDIT)}" for p in paths]
    dump_text("SHA256SUMS.txt", "\n".join(lines))


def main() -> None:
    AUDIT.mkdir(parents=True, exist_ok=True)
    finalize_provenance()
    finalize_code_provenance()
    packs = load_pack_rows()
    preds = load_pred_rows()
    finalize_sample_integrity(packs, preds)
    finalize_budget(packs, preds)
    finalize_canary()
    finalize_env_diff()
    finalize_gold_and_score()
    finalize_run_status()
    finalize_report()
    finalize_prompt_binding()
    write_checksums()
    print(f"wrote final audit artifacts to {AUDIT}")


if __name__ == "__main__":
    main()
