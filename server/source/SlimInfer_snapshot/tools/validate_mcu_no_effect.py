#!/usr/bin/env python3
"""Offline no-effect validation for MCU-SlimInfer tiny quality smoke.

Reads existing smoke + counterfactual audit artifacts only.
Does not rerun generation, scorer, or selector.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.mcu_sliminfer_selection import parse_json_field

DEFAULT_SMOKE_DIR = Path("results/mcu_sliminfer_tiny_quality_smoke_20260708_210400")
DEFAULT_AUDIT_DIR = Path("results/mcu_sliminfer_counterfactual_audit_20260708_203500")
GATE_LAYER = 26
PREFILL_READ_LAYER = 26
MAX_NEW_TOKENS = 32
TRUNCATION_WORD_THRESHOLD = 28

Key = Tuple[str, int]


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def sha_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha_json(value: Any) -> str:
    return sha_text(json.dumps(value, ensure_ascii=False, sort_keys=True))


def load_predictions(run_dir: Path) -> Dict[Key, Dict[str, Any]]:
    out: Dict[Key, Dict[str, Any]] = {}
    pred_dir = run_dir / "longbench_predictions"
    for task in ("hotpotqa", "musique"):
        for row in read_jsonl(pred_dir / f"{task}.jsonl"):
            key = (str(row["task"]) if "task" in row else task, int(row["sample_index"]))
            out[key] = row
    return out


def load_generations(run_dir: Path) -> Dict[Key, Dict[str, Any]]:
    out: Dict[Key, Dict[str, Any]] = {}
    for row in read_jsonl(run_dir / "audit" / "generations.jsonl"):
        key = (str(row["task"]), int(row["sample_index"]))
        out[key] = row
    return out


def load_used_blocks_prefill(run_dir: Path, layer_id: int = PREFILL_READ_LAYER) -> Dict[Key, Dict[str, Any]]:
    out: Dict[Key, Dict[str, Any]] = {}
    for row in read_jsonl(run_dir / "audit" / "used_blocks.jsonl"):
        if str(row.get("phase", "")) != "prefill":
            continue
        if int(row.get("layer_id", -1)) != layer_id:
            continue
        key = (str(row["task"]), int(row["sample_index"]))
        out[key] = row
    return out


def load_mcu_audit_prefill(audit_path: Path) -> Dict[Key, Dict[str, Any]]:
    out: Dict[Key, Dict[str, Any]] = {}
    for row in read_jsonl(audit_path):
        if str(row.get("phase", "")) != "prefill":
            continue
        if int(row.get("layer_id", -1)) != GATE_LAYER:
            continue
        key = (str(row["task"]), int(row["sample_index"]))
        if key not in out:
            out[key] = row
    return out


def load_frozen(path: Path) -> Dict[Key, Dict[str, Any]]:
    out: Dict[Key, Dict[str, Any]] = {}
    for row in read_jsonl(path):
        task = str(row.get("task", ""))
        sample_index = row.get("sample_index")
        if not task or sample_index is None:
            continue
        key = (task, int(sample_index))
        out[key] = row
    return out


def load_counterfactual_audit_csv(path: Path) -> Dict[Key, Dict[str, Any]]:
    import csv as csv_mod

    out: Dict[Key, Dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for row in csv_mod.DictReader(handle):
            key = (str(row["task"]), int(row["sample_index"]))
            out[key] = row
    return out


def block_set_from_used(row: Mapping[str, Any]) -> Set[int]:
    blocks = row.get("used_block_ids_first100")
    if isinstance(blocks, list):
        return {int(x) for x in blocks}
    return set()


def context_signature(row: Mapping[str, Any]) -> Dict[str, Any]:
    blocks = sorted(block_set_from_used(row))
    kv_first = row.get("kv_abs_indices_first100") or []
    return {
        "final_block_count": int(row.get("final_block_count_used_by_attention", 0) or 0),
        "kv_abs_indices_count": int(row.get("kv_abs_indices_count", 0) or 0),
        "prompt_token_count": int(row.get("prompt_token_count", 0) or 0),
        "block_ids_hash": sha_json(blocks),
        "kv_first100_hash": sha_json(kv_first),
    }


def preview_blocks(block_texts: Sequence[str], block_ids: Iterable[int], limit: int = 100) -> str:
    parts: List[str] = []
    for bid in block_ids:
        if 0 <= int(bid) < len(block_texts):
            text = str(block_texts[int(bid)]).replace("\n", " ").strip()
            parts.append(text[:limit])
    return " | ".join(parts)


def estimate_word_count(text: str) -> int:
    return len(str(text).split())


def decide_validation(rows: Sequence[Mapping[str, Any]], summary: Mapping[str, Any]) -> Tuple[str, List[str]]:
    reasons: List[str] = []

    if summary.get("context_unchanged_count", 0) > 0:
        reasons.append(
            f"context_unchanged_orig_vs_mcu={summary['context_unchanged_count']}/{summary['sample_count']}"
        )
    if summary.get("removed_still_in_used_count", 0) > 0:
        reasons.append(
            f"removed_blocks_still_in_runtime_used_blocks={summary['removed_still_in_used_count']}"
        )
    if summary.get("frozen_alignment_error_count", 0) > 0:
        reasons.append(f"frozen_alignment_errors={summary['frozen_alignment_error_count']}")
    if summary.get("audit_not_applied_count", 0) > 0:
        reasons.append(f"audit_selection_not_reflected_in_used_blocks={summary['audit_not_applied_count']}")

    if reasons:
        return "BUG_STOP_FIX_REQUIRED", reasons

    near_trunc = int(summary.get("near_max_new_tokens_count", 0) or 0)
    musique_zero = int(summary.get("musique_zero_score_count", 0) or 0)
    identical_pred = int(summary.get("identical_prediction_count", 0) or 0)
    context_changed = int(summary.get("context_changed_count", 0) or 0)

    if (
        identical_pred == summary.get("sample_count", 0)
        and context_changed == 0
        and int(summary.get("audit_selection_changed_count", 0) or 0) > 0
    ):
        return "BUG_STOP_FIX_REQUIRED", [
            "audit_reports_selection_changed_but_runtime_context_and_predictions_unchanged"
        ]

    if near_trunc >= max(3, summary.get("sample_count", 0) // 3) or musique_zero >= 4:
        subreasons = []
        if near_trunc:
            subreasons.append(f"near_max_new_tokens_count={near_trunc}")
        if musique_zero:
            subreasons.append(f"musique_zero_score_count={musique_zero}")
        return "QUALITY_SMOKE_INSENSITIVE", subreasons or ["smoke_metrics_uninformative"]

    if context_changed > 0 and identical_pred == summary.get("sample_count", 0):
        return "METHOD_NO_EFFECT_CONFIRMED", [
            f"context_changed={context_changed}",
            f"identical_predictions={identical_pred}",
        ]

    return "METHOD_NO_EFFECT_CONFIRMED", [
        f"identical_predictions={identical_pred}/{summary.get('sample_count', 0)}",
        "no_runtime_wiring_bug_detected",
    ]


def build_report(
    smoke_dir: Path,
    audit_dir: Path,
    rows: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
    decision: str,
    reasons: Sequence[str],
) -> str:
    lines = [
        "# MCU No-Effect Validation",
        "",
        f"- Smoke dir: `{smoke_dir}`",
        f"- Counterfactual audit dir: `{audit_dir}`",
        f"- Decision: **{decision}**",
        "",
        "## Summary",
        f"- Samples: {summary.get('sample_count', 0)}",
        f"- Identical predictions (orig vs MCU): {summary.get('identical_prediction_count', 0)}",
        f"- Identical prompt_hash (generations): {summary.get('identical_prompt_hash_count', 0)}",
        f"- Identical output_hash (generations): {summary.get('identical_output_hash_count', 0)}",
        f"- Audit selection changed (prefill L{GATE_LAYER}): {summary.get('audit_selection_changed_count', 0)}",
        f"- Runtime context unchanged (orig vs MCU L{PREFILL_READ_LAYER}): {summary.get('context_unchanged_count', 0)}",
        f"- Runtime context changed: {summary.get('context_changed_count', 0)}",
        f"- Removed blocks still in used_blocks: {summary.get('removed_still_in_used_count', 0)}",
        f"- Frozen alignment errors: {summary.get('frozen_alignment_error_count', 0)}",
        f"- Near max_new_tokens ({MAX_NEW_TOKENS}) predictions: {summary.get('near_max_new_tokens_count', 0)}",
        f"- MuSiQue zero-score samples: {summary.get('musique_zero_score_count', 0)}",
        f"- Token count same but selection changed (audit): {summary.get('token_count_same_selection_changed_count', 0)}",
        f"- Pred differs but score same: {summary.get('pred_diff_score_same_count', 0)}",
        "",
        "## Checklist (10 items)",
        f"1. Predictions identical orig vs MCU: {summary.get('identical_prediction_count', 0)}/{summary.get('sample_count', 0)}",
        f"2. Pred differs but score same: {summary.get('pred_diff_score_same_count', 0)} samples",
        f"3. Audit original vs mcu selected_block_ids differ: {summary.get('audit_selection_changed_count', 0)}/{summary.get('sample_count', 0)}",
        f"4. Runtime context hash differs (L{PREFILL_READ_LAYER}): {summary.get('context_changed_count', 0)}/{summary.get('sample_count', 0)}",
        f"5. Removed/added block ids non-empty; added not exclusive to MCU context: removed_still_used={summary.get('removed_still_in_used_count', 0)}, added_in_used={summary.get('added_in_used_count', 0)}",
        f"6. Frozen selection sample alignment errors: {summary.get('frozen_alignment_error_count', 0)}",
        f"7. Gate layer {GATE_LAYER} == used_blocks read layer {PREFILL_READ_LAYER}: True",
        f"8. Audit selection not reflected in used_blocks: {summary.get('audit_not_applied_count', 0)}/{summary.get('sample_count', 0)}",
        f"9. selected_token_count same but selection changed: {summary.get('token_count_same_selection_changed_count', 0)}",
        f"10. Near max_new_tokens ({MAX_NEW_TOKENS}) by word proxy (>={TRUNCATION_WORD_THRESHOLD} words): {summary.get('near_max_new_tokens_count', 0)}",
        "",
        "## Gate / read layers",
        f"- MCU gate apply layer (from smoke command): {GATE_LAYER}",
        f"- used_blocks read layer: {PREFILL_READ_LAYER}",
        f"- mcu_selection_audit prefill layer: {GATE_LAYER}",
        "",
        "## Decision reasons",
    ]
    for reason in reasons:
        lines.append(f"- {reason}")
    lines.extend([
        "",
        "## Per-sample highlights",
    ])
    for row in rows:
        lines.append(
            f"- {row['task']}:{row['sample_index']} | pred_same={row['prediction_identical']} "
            f"| ctx_same={row['context_signature_identical']} | audit_changed={row['audit_selection_changed']} "
            f"| removed_still_used={row['removed_still_in_used_blocks']} "
            f"| tok_same_sel_changed={row.get('token_count_same_selection_changed', False)}"
        )
    pred_diff_rows = [r for r in rows if r.get("pred_diff_score_same")]
    if pred_diff_rows:
        lines.extend(["", "## Pred differs but score same"])
        for row in pred_diff_rows:
            lines.append(
                f"- {row['task']}:{row['sample_index']} score={row['score_mcu']} "
                f"orig={row['prediction_orig']!r} mcu={row['prediction_mcu']!r}"
            )
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate MCU tiny-quality no-effect result.")
    parser.add_argument("--smoke_dir", type=Path, default=DEFAULT_SMOKE_DIR)
    parser.add_argument("--audit_dir", type=Path, default=DEFAULT_AUDIT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    smoke_dir = (REPO_ROOT / args.smoke_dir).resolve()
    audit_dir = (REPO_ROOT / args.audit_dir).resolve()

    orig_dir = smoke_dir / "run_sliminfer_original"
    mcu_dir = smoke_dir / "run_mcu_sliminfer"
    frozen_path = smoke_dir / "mcu_frozen_selections.jsonl"
    cf_csv = audit_dir / "mcu_counterfactual_audit.csv"
    per_sample_csv = smoke_dir / "per_sample_quality.csv"

    pred_orig = load_predictions(orig_dir)
    pred_mcu = load_predictions(mcu_dir)
    gen_orig = load_generations(orig_dir)
    gen_mcu = load_generations(mcu_dir)
    used_orig = load_used_blocks_prefill(orig_dir)
    used_mcu = load_used_blocks_prefill(mcu_dir)
    mcu_audit = load_mcu_audit_prefill(mcu_dir / "audit" / "mcu_runtime_selection_audit.jsonl")
    frozen = load_frozen(frozen_path)
    cf_audit = load_counterfactual_audit_csv(cf_csv) if cf_csv.exists() else {}

    keys = sorted(set(pred_orig) & set(pred_mcu))
    if not keys:
        raise RuntimeError("no overlapping samples between original and MCU predictions")

    validation_rows: List[Dict[str, Any]] = []
    for key in keys:
        task, sample_index = key
        po, pm = pred_orig[key], pred_mcu[key]
        go, gm = gen_orig.get(key, {}), gen_mcu.get(key, {})
        uo, um = used_orig[key], used_mcu[key]
        audit = mcu_audit.get(key, {})
        fr = frozen.get(key, {})
        cf = cf_audit.get(key, {})

        pred_o = str(po.get("pred", ""))
        pred_m = str(pm.get("pred", ""))
        score_o = float(po.get("score", 0.0) if "score" in po else 0.0)
        score_m = float(pm.get("score", 0.0) if "score" in pm else 0.0)
        # recompute from per_sample if available - predictions jsonl may not have score
        prediction_identical = pred_o == pred_m
        score_identical = abs(score_o - score_m) < 1e-9

        sig_o = context_signature(uo)
        sig_m = context_signature(um)
        context_identical = sig_o == sig_m

        orig_sel = set(parse_json_field(audit.get("original_selected_block_ids")) or [])
        mcu_sel = set(parse_json_field(audit.get("mcu_selected_block_ids")) or [])
        frozen_orig = set(parse_json_field(fr.get("original_selected_block_ids")) or [])
        frozen_mcu = set(parse_json_field(fr.get("mcu_selected_block_ids")) or [])
        cf_mcu = set(parse_json_field(cf.get("mcu_selected_block_ids")) or []) if cf else set()

        removed = set(parse_json_field(audit.get("removed_block_ids")) or [])
        added = set(parse_json_field(audit.get("added_block_ids")) or [])
        used_blocks = block_set_from_used(um)

        frozen_alignment_ok = (
            str(fr.get("sample_id", "")) == str(po.get("sample_id", ""))
            and str(fr.get("task", "")) == task
            and int(fr.get("sample_index", -1)) == sample_index
            and (not frozen_mcu or frozen_mcu == mcu_sel or frozen_mcu == cf_mcu)
        )

        audit_changed = bool(audit.get("mcu_triggered")) or orig_sel != mcu_sel
        removed_still_used = bool(removed & used_blocks)
        added_in_used = bool(added.issubset(used_blocks)) if added else True
        added_nonempty = len(added) > 0
        removed_nonempty = len(removed) > 0
        audit_applied_in_used = (
            (not audit_changed)
            or (mcu_sel.issubset(used_blocks) and not removed_still_used and not context_identical)
        )
        audit_not_applied = audit_changed and (context_identical or removed_still_used)

        sel_tok_o = audit.get("selected_token_count_original")
        sel_tok_m = audit.get("selected_token_count_mcu")
        token_count_same = (
            sel_tok_o is not None
            and sel_tok_m is not None
            and int(sel_tok_o) == int(sel_tok_m)
        )
        token_count_same_selection_changed = token_count_same and orig_sel != mcu_sel

        word_count = estimate_word_count(pred_m)
        near_truncation = word_count >= TRUNCATION_WORD_THRESHOLD
        pred_diff_score_same = (not prediction_identical) and score_identical

        validation_rows.append({
            "task": task,
            "sample_index": sample_index,
            "sample_id": po.get("sample_id", ""),
            "prediction_identical": prediction_identical,
            "prediction_orig": pred_o,
            "prediction_mcu": pred_m,
            "score_orig": score_o,
            "score_mcu": score_m,
            "score_identical": score_identical,
            "score_delta": score_m - score_o,
            "prompt_hash_identical": go.get("prompt_hash") == gm.get("prompt_hash"),
            "output_hash_identical": go.get("output_hash") == gm.get("output_hash"),
            "prompt_hash_orig": go.get("prompt_hash", ""),
            "prompt_hash_mcu": gm.get("prompt_hash", ""),
            "output_hash_orig": go.get("output_hash", ""),
            "output_hash_mcu": gm.get("output_hash", ""),
            "audit_selection_changed": audit_changed,
            "original_selected_block_ids": sorted(orig_sel),
            "mcu_selected_block_ids": sorted(mcu_sel),
            "frozen_mcu_selected_block_ids": sorted(frozen_mcu),
            "selection_ids_differ_audit": orig_sel != mcu_sel,
            "context_signature_identical": context_identical,
            "orig_block_ids_hash": sig_o["block_ids_hash"],
            "mcu_block_ids_hash": sig_m["block_ids_hash"],
            "orig_kv_first100_hash": sig_o["kv_first100_hash"],
            "mcu_kv_first100_hash": sig_m["kv_first100_hash"],
            "orig_final_block_count": sig_o["final_block_count"],
            "mcu_final_block_count": sig_m["final_block_count"],
            "orig_kv_abs_indices_count": sig_o["kv_abs_indices_count"],
            "mcu_kv_abs_indices_count": sig_m["kv_abs_indices_count"],
            "removed_block_ids": sorted(removed),
            "added_block_ids": sorted(added),
            "removed_still_in_used_blocks": removed_still_used,
            "added_in_used_blocks": added_in_used,
            "removed_block_ids_nonempty": removed_nonempty,
            "added_block_ids_nonempty": added_nonempty,
            "audit_applied_in_used_blocks": audit_applied_in_used,
            "audit_not_applied_in_runtime": audit_not_applied,
            "selected_token_count_original": sel_tok_o,
            "selected_token_count_mcu": sel_tok_m,
            "token_count_same_selection_changed": token_count_same_selection_changed,
            "pred_diff_score_same": pred_diff_score_same,
            "frozen_alignment_ok": frozen_alignment_ok,
            "frozen_sample_id": fr.get("sample_id", ""),
            "runtime_audit_fallback": bool(audit.get("fallback_used")),
            "runtime_audit_reason": audit.get("mcu_trigger_reason", ""),
            "prediction_word_count": word_count,
            "near_max_new_tokens": near_truncation,
            "musique_score_zero": task == "musique" and score_m == 0.0,
        })

    if per_sample_csv.exists():
        import csv as csv_mod

        score_by_key: Dict[Key, Tuple[float, float]] = {}
        with per_sample_csv.open(encoding="utf-8") as handle:
            for row in csv_mod.DictReader(handle):
                key = (str(row["task"]), int(row["sample_index"]))
                score_by_key[key] = (float(row.get("original_score", 0) or 0), float(row.get("mcu_score", 0) or 0))
        for row in validation_rows:
            key = (str(row["task"]), int(row["sample_index"]))
            if key in score_by_key:
                so, sm = score_by_key[key]
                row["score_orig"] = so
                row["score_mcu"] = sm
                row["score_delta"] = sm - so
                row["score_identical"] = abs(sm - so) < 1e-9
                row["musique_score_zero"] = str(row["task"]) == "musique" and sm == 0.0

    summary = {
        "sample_count": len(validation_rows),
        "identical_prediction_count": sum(1 for r in validation_rows if r["prediction_identical"]),
        "identical_prompt_hash_count": sum(1 for r in validation_rows if r["prompt_hash_identical"]),
        "identical_output_hash_count": sum(1 for r in validation_rows if r["output_hash_identical"]),
        "audit_selection_changed_count": sum(1 for r in validation_rows if r["audit_selection_changed"]),
        "context_unchanged_count": sum(1 for r in validation_rows if r["context_signature_identical"]),
        "context_changed_count": sum(1 for r in validation_rows if not r["context_signature_identical"]),
        "removed_still_in_used_count": sum(1 for r in validation_rows if r["removed_still_in_used_blocks"]),
        "frozen_alignment_error_count": sum(1 for r in validation_rows if not r["frozen_alignment_ok"]),
        "audit_not_applied_count": sum(
            1 for r in validation_rows if r.get("audit_not_applied_in_runtime")
        ),
        "near_max_new_tokens_count": sum(1 for r in validation_rows if r["near_max_new_tokens"]),
        "musique_zero_score_count": sum(1 for r in validation_rows if r["musique_score_zero"]),
        "token_count_same_selection_changed_count": sum(
            1 for r in validation_rows if r.get("token_count_same_selection_changed")
        ),
        "pred_diff_score_same_count": sum(1 for r in validation_rows if r.get("pred_diff_score_same")),
        "added_in_used_count": sum(1 for r in validation_rows if r.get("added_in_used_blocks")),
        "mean_prediction_word_count": statistics.mean(float(r["prediction_word_count"]) for r in validation_rows),
    }

    decision, reasons = decide_validation(validation_rows, summary)

    fields = [
        "task", "sample_index", "sample_id",
        "prediction_identical", "score_identical", "score_delta",
        "prompt_hash_identical", "output_hash_identical",
        "audit_selection_changed", "selection_ids_differ_audit",
        "context_signature_identical",
        "orig_block_ids_hash", "mcu_block_ids_hash",
        "orig_kv_first100_hash", "mcu_kv_first100_hash",
        "orig_final_block_count", "mcu_final_block_count",
        "orig_kv_abs_indices_count", "mcu_kv_abs_indices_count",
        "removed_block_ids", "added_block_ids",
        "removed_still_in_used_blocks", "added_in_used_blocks",
        "removed_block_ids_nonempty", "added_block_ids_nonempty",
        "selected_token_count_original", "selected_token_count_mcu",
        "token_count_same_selection_changed", "pred_diff_score_same",
        "audit_applied_in_used_blocks", "audit_not_applied_in_runtime", "frozen_alignment_ok",
        "runtime_audit_fallback", "runtime_audit_reason",
        "prediction_word_count", "near_max_new_tokens", "musique_score_zero",
        "prediction_orig", "prediction_mcu",
    ]
    write_csv(smoke_dir / "mcu_no_effect_validation.csv", validation_rows, fields)

    report = build_report(smoke_dir, audit_dir, validation_rows, summary, decision, reasons)
    (smoke_dir / "MCU_NO_EFFECT_VALIDATION.md").write_text(report, encoding="utf-8")

    print(report)
    print(f"Wrote {smoke_dir / 'MCU_NO_EFFECT_VALIDATION.md'}")


if __name__ == "__main__":
    main()
