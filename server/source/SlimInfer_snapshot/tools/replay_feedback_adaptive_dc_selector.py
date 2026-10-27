#!/usr/bin/env python3
"""Offline replay of Feedback-Adaptive DC selector on held-out64.

Uses existing original/DC predictions, scores, and selection audit context.
No generation, no scorer, no gold/task-label/score-based gating.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.cache_utils import _gate_content_tokens, _gate_entity_like_units, _gate_token_jaccard

DEFAULT_HELDOUT_DIR = Path("results/adaptive_dc_v3_heldout64_diagnostic_20260708_073628")
DEFAULT_GATE_AUDIT_DIR = Path("results/adaptive_dc_v3_runtime_gate_fix_audit_20260708_183831")
DEFAULT_LONGBENCH_DIR = Path("/home/yu-bai/Desktop/jiaqizhang/HeadKV/data/LongBench")
DEFAULT_MODEL_PATH = Path(
    "/home/yu-bai/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/"
    "snapshots/a09a35458c702b33eeacc393d103063234e8bc28"
)

GATE_LAYER = 26
TASKS = ["hotpotqa", "musique"]
TINY_EXCLUDE = {
    "hotpotqa": {0, 1, 2, 3, 4, 5, 41, 56, 57, 117},
    "musique": {0, 1, 2, 3, 4, 20, 50, 54, 56, 134},
}

GENERIC_REFUSAL_PATTERN = re.compile(
    r"(?:"
    r"i\s+(?:do\s+not|don't)\s+know|"
    r"cannot\s+(?:determine|answer|find)|"
    r"not\s+(?:mentioned|provided|available|specified|found)|"
    r"no\s+information|"
    r"unable\s+to|"
    r"insufficient\s+(?:information|context|evidence)|"
    r"unknown|"
    r"\bn/?a\b"
    r")",
    flags=re.IGNORECASE,
)

# Fixed heuristic thresholds (no sweep). Tuned to avoid static-gate-like over-triggering.
ANSWER_VERY_UNSUPPORTED_MAX = 0.28
VERIFIER_ORIG_SUPPORT_MAX = 0.40
VERIFIER_SUPPORT_MARGIN = 0.10
VERIFIER_DC_ENTITY_ADD_MIN = 0.15
VERIFIER_ANSWER_JACCARD_MAX = 0.12
QUESTION_REPEAT_OVERLAP = 0.55

Key = Tuple[str, int]


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def read_csv(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def to_float(value: Any) -> float:
    try:
        out = float(value)
        return out if math.isfinite(out) else 0.0
    except (TypeError, ValueError):
        return 0.0


def mean(values: Sequence[float]) -> float:
    items = [float(v) for v in values if math.isfinite(float(v))]
    return float(sum(items) / len(items)) if items else 0.0


def heldout_indices(task: str, count: int = 64) -> List[int]:
    selected: List[int] = []
    for idx in range(200):
        if idx in TINY_EXCLUDE[task]:
            continue
        selected.append(idx)
        if len(selected) >= count:
            break
    return selected


def heldout_keys() -> List[Key]:
    return [(task, idx) for task in TASKS for idx in heldout_indices(task)]


def load_tokenizer_and_prompts(model_path: Path) -> Tuple[Any, Dict[str, str]]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
    dataset2prompt = json.loads((REPO_ROOT / "configs" / "dataset2prompt.json").read_text(encoding="utf-8"))
    return tokenizer, dataset2prompt


def load_samples(longbench_dir: Path, task: str) -> List[Dict[str, Any]]:
    return [
        json.loads(line)
        for line in (longbench_dir / f"{task}.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def build_chat_prompt(tokenizer: Any, prompt: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
    )


def build_prompt_blocks(
    tokenizer: Any, prompt_format: str, sample: Dict[str, Any], dataset: str,
    max_model_len: int, block_size: int,
) -> Tuple[str, List[str]]:
    prompt = prompt_format.format(**sample)
    input_ids = tokenizer(prompt, truncation=False, return_tensors=None).input_ids
    if input_ids and isinstance(input_ids[0], list):
        input_ids = input_ids[0]
    if len(input_ids) > max_model_len:
        half = max_model_len // 2
        prompt = tokenizer.decode(input_ids[:half], skip_special_tokens=True) + tokenizer.decode(
            input_ids[-half:], skip_special_tokens=True
        )
    if dataset not in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]:
        prompt = build_chat_prompt(tokenizer, prompt)
    chat_ids = tokenizer(prompt, truncation=False, return_tensors=None).input_ids
    if chat_ids and isinstance(chat_ids[0], list):
        chat_ids = chat_ids[0]
    blocks = [
        tokenizer.decode(chat_ids[start : start + block_size], skip_special_tokens=True, clean_up_tokenization_spaces=False)
        for start in range(0, len(chat_ids), block_size)
    ]
    query_text = str(sample.get("input", sample.get("question", sample.get("query", ""))))
    return query_text, blocks


def context_from_blocks(block_texts: Sequence[str], selected_ids: Sequence[int]) -> str:
    return " ".join(block_texts[int(i)] for i in selected_ids if 0 <= int(i) < len(block_texts))


def answer_term_support(answer: str, context_text: str) -> float:
    answer_tokens = _gate_content_tokens(answer)
    if not answer_tokens:
        return 0.0
    context_tokens = _gate_content_tokens(context_text)
    if not context_tokens:
        return 0.0
    return len(answer_tokens.intersection(context_tokens)) / len(answer_tokens)


def entity_coverage_ratio(entities: Sequence[str], context_text: str) -> float:
    if not entities:
        return 1.0
    context_tokens = _gate_content_tokens(context_text)
    if not context_tokens:
        return 0.0
    hits = 0
    for entity in entities:
        entity_tokens = _gate_content_tokens(entity)
        if entity in context_tokens or (entity_tokens and entity_tokens.issubset(context_tokens)):
            hits += 1
    return hits / len(entities)


def original_answer_empty_or_short(answer: str) -> bool:
    stripped = str(answer or "").strip()
    if len(stripped) < 3:
        return True
    return len(stripped.split()) < 2


def original_answer_generic_or_refusal(answer: str) -> bool:
    return bool(GENERIC_REFUSAL_PATTERN.search(str(answer or "")))


def original_answer_repeats_question(answer: str, query_text: str) -> bool:
    answer_tokens = _gate_content_tokens(answer)
    query_tokens = _gate_content_tokens(query_text)
    if len(answer_tokens) < 2 or not query_tokens:
        return False
    overlap = len(answer_tokens.intersection(query_tokens)) / len(answer_tokens)
    return overlap >= QUESTION_REPEAT_OVERLAP


def predictions_disagree(original_answer: str, dc_answer: str) -> bool:
    return str(original_answer or "").strip().lower() != str(dc_answer or "").strip().lower()


def compute_feedback_features(
    query_text: str,
    original_answer: str,
    dc_answer: str,
    original_context: str,
    dc_context: str,
) -> Dict[str, Any]:
    query_entities = _gate_entity_like_units(query_text)
    answer_entities = _gate_entity_like_units(original_answer)
    original_context_entities = set(_gate_entity_like_units(original_context))
    dc_context_entities = set(_gate_entity_like_units(dc_context))
    query_entity_set = set(query_entities)

    query_cov = entity_coverage_ratio(query_entities, original_context)
    answer_cov = entity_coverage_ratio(answer_entities, original_context)
    entity_gap = (
        len(query_entity_set.difference(original_context_entities)) / len(query_entity_set)
        if query_entity_set
        else 0.0
    )
    dc_additional = (
        len(query_entity_set.intersection(dc_context_entities).difference(original_context_entities))
        / len(query_entity_set)
        if query_entity_set
        else 0.0
    )

    return {
        "original_answer_empty_or_short": original_answer_empty_or_short(original_answer),
        "original_answer_generic_or_refusal": original_answer_generic_or_refusal(original_answer),
        "original_answer_repeats_question": original_answer_repeats_question(original_answer, query_text),
        "answer_terms_supported_by_original_context": answer_term_support(original_answer, original_context),
        "query_entities_covered_by_original_context": query_cov,
        "answer_entities_covered_by_original_context": answer_cov,
        "original_context_entity_coverage_gap": entity_gap,
        "dc_context_additional_entity_coverage": dc_additional,
        "original_vs_dc_prediction_disagreement": predictions_disagree(original_answer, dc_answer),
        "dc_answer_terms_supported_by_dc_context": answer_term_support(dc_answer, dc_context),
        "answer_jaccard_original_vs_dc": _gate_token_jaccard(
            _gate_content_tokens(original_answer), _gate_content_tokens(dc_answer)
        ),
    }


def f1_unreliable_reasons(features: Mapping[str, Any]) -> List[str]:
    reasons: List[str] = []
    if features["original_answer_empty_or_short"]:
        reasons.append("original_answer_empty_or_short")
    if features["original_answer_generic_or_refusal"]:
        reasons.append("original_answer_generic_or_refusal")
    if features["original_answer_repeats_question"]:
        reasons.append("original_answer_repeats_question")
    if features["answer_terms_supported_by_original_context"] < ANSWER_VERY_UNSUPPORTED_MAX:
        reasons.append("answer_terms_very_unsupported_by_original_context")
    return reasons


def policy_f1_trigger(features: Mapping[str, Any]) -> Tuple[bool, str]:
    reasons = f1_unreliable_reasons(features)
    return bool(reasons), "; ".join(reasons)


def policy_f2_choose(features: Mapping[str, Any]) -> Tuple[str, bool, str]:
    """Return chosen_selector, triggered-to-dc, reason."""
    orig_support = float(features["answer_terms_supported_by_original_context"])
    dc_support = float(features["dc_answer_terms_supported_by_dc_context"])

    if features["original_answer_empty_or_short"] and dc_support >= orig_support:
        return "dc", True, "original_empty_or_short_dc_at_least_as_supported"

    if features["original_answer_generic_or_refusal"] and dc_support >= orig_support:
        return "dc", True, "original_generic_or_refusal_dc_at_least_as_supported"

    if not features["original_vs_dc_prediction_disagreement"]:
        return "original", False, "answers_agree_keep_original"

    if orig_support < VERIFIER_ORIG_SUPPORT_MAX and dc_support > orig_support + VERIFIER_SUPPORT_MARGIN:
        return "dc", True, "disagreement_original_weak_dc_more_supported"

    if (
        features["dc_context_additional_entity_coverage"] >= VERIFIER_DC_ENTITY_ADD_MIN
        and dc_support > orig_support + 0.08
    ):
        return "dc", True, "disagreement_dc_adds_query_entities_and_support"

    if (
        features["answer_jaccard_original_vs_dc"] < VERIFIER_ANSWER_JACCARD_MAX
        and dc_support > orig_support + 0.08
    ):
        return "dc", True, "disagreement_large_answer_change_dc_more_supported"

    return "original", False, "verifier_keeps_original"


def load_predictions(heldout_dir: Path, keys: Sequence[Key]) -> Tuple[Dict[Key, str], Dict[Key, str]]:
    original: Dict[Key, str] = {}
    dc: Dict[Key, str] = {}
    for task in TASKS:
        for row in read_jsonl(heldout_dir / "run_sliminfer_original" / "longbench_predictions" / f"{task}.jsonl"):
            key = (task, int(row.get("sample_index", -1)))
            if key in keys:
                original[key] = str(row.get("pred", "")).strip()
        for row in read_jsonl(heldout_dir / "run_dc_sliminfer" / "longbench_predictions" / f"{task}.jsonl"):
            key = (task, int(row.get("sample_index", -1)))
            if key in keys:
                dc[key] = str(row.get("pred", "")).strip()
    return original, dc


def load_quality_scores(heldout_dir: Path, keys: Sequence[Key]) -> Dict[Key, Dict[str, Any]]:
    out: Dict[Key, Dict[str, Any]] = {}
    for row in read_csv(heldout_dir / "per_sample_quality.csv"):
        key = (str(row["task"]), int(row["sample_index"]))
        if key in keys:
            out[key] = row
    return out


def load_selection_audit(heldout_dir: Path, keys: Sequence[Key]) -> Dict[Key, Dict[str, Any]]:
    selected: Dict[Key, Dict[str, Any]] = {}
    for row in read_jsonl(
        heldout_dir / "run_adaptive_dc_sliminfer_v3" / "audit" / "dc_runtime_selection_audit.jsonl"
    ):
        if str(row.get("phase")) != "prefill" or int(row.get("layer_id", -1)) != GATE_LAYER:
            continue
        key = (str(row.get("task")), int(row.get("sample_index", -1)))
        if key in keys and key not in selected:
            selected[key] = row
    return selected


def summarize_policy(
    policy_name: str,
    per_sample: Sequence[Mapping[str, Any]],
    extra_generation_rate: float,
    uses_disagreement: bool,
) -> Dict[str, Any]:
    task_scores: Dict[str, Dict[str, List[float]]] = {t: defaultdict(list) for t in TASKS}
    trigger_count = 0
    dc_positive = dc_negative = 0
    dc_positive_retained = dc_negative_triggered = 0
    disagreement_count = 0

    for row in per_sample:
        task = str(row["task"])
        triggered = bool(row[f"{policy_name}_triggered"])
        chosen = str(row[f"{policy_name}_chosen_selector"])
        orig_score = to_float(row["original_score"])
        dc_score = to_float(row["dc_score"])
        feedback_score = to_float(row[f"{policy_name}_score"])

        if triggered:
            trigger_count += 1
        if bool(row["original_vs_dc_prediction_disagreement"]):
            disagreement_count += 1
        if dc_score > orig_score + 1e-9:
            dc_positive += 1
            if chosen == "dc":
                dc_positive_retained += 1
        elif dc_score < orig_score - 1e-9:
            dc_negative += 1
            if triggered:
                dc_negative_triggered += 1

        task_scores[task]["original"].append(orig_score)
        task_scores[task]["dc"].append(dc_score)
        task_scores[task]["feedback"].append(feedback_score)

    original_macro = mean([mean(task_scores[t]["original"]) for t in TASKS])
    dc_macro = mean([mean(task_scores[t]["dc"]) for t in TASKS])
    feedback_macro = mean([mean(task_scores[t]["feedback"]) for t in TASKS])

    return {
        "policy": policy_name,
        "trigger_count": trigger_count,
        "hotpotqa_original_macro": mean(task_scores["hotpotqa"]["original"]),
        "hotpotqa_dc_macro": mean(task_scores["hotpotqa"]["dc"]),
        "hotpotqa_feedback_macro": mean(task_scores["hotpotqa"]["feedback"]),
        "musique_original_macro": mean(task_scores["musique"]["original"]),
        "musique_dc_macro": mean(task_scores["musique"]["dc"]),
        "musique_feedback_macro": mean(task_scores["musique"]["feedback"]),
        "macro_original": original_macro,
        "macro_dc": dc_macro,
        "macro_feedback": feedback_macro,
        "feedback_vs_original_delta": feedback_macro - original_macro,
        "feedback_vs_dc_delta": feedback_macro - dc_macro,
        "dc_positive_count": dc_positive,
        "dc_negative_count": dc_negative,
        "dc_positive_retained_count": dc_positive_retained,
        "dc_negative_triggered_count": dc_negative_triggered,
        "prediction_disagreement_rate": disagreement_count / len(per_sample) if per_sample else 0.0,
        "extra_generation_rate_estimate": extra_generation_rate,
        "uses_prediction_disagreement_feature": uses_disagreement,
        "uses_gold_task_label_score_for_gate": False,
    }


def decide_policy(summary: Mapping[str, Any]) -> Dict[str, Any]:
    feedback_macro = float(summary["macro_feedback"])
    original_macro = float(summary["macro_original"])
    trigger_count = int(summary["trigger_count"])
    total = 128
    delta = feedback_macro - original_macro
    retained = int(summary["dc_positive_retained_count"])
    neg_triggered = int(summary["dc_negative_triggered_count"])
    extra_gen = float(summary["extra_generation_rate_estimate"])

    if trigger_count == 0 or trigger_count == total:
        return {
            "decision": "FEEDBACK_ADAPTIVE_NO_GO",
            "reasons": [
                f"trigger_count={trigger_count}/{total} (degenerate)",
                f"feedback_vs_original_delta={delta:+.4f}",
            ],
        }

    if feedback_macro <= original_macro:
        return {
            "decision": "FEEDBACK_ADAPTIVE_NO_GO",
            "reasons": [
                f"feedback_macro={feedback_macro:.4f} <= original_macro={original_macro:.4f}",
                f"dc_positive_retained={retained}",
                f"dc_negative_triggered={neg_triggered}",
            ],
        }

    strong_retention = retained >= max(3, neg_triggered + 2) and retained >= 2 * max(1, neg_triggered)
    if delta >= 0.05 and 0 < trigger_count < 120 and strong_retention:
        return {
            "decision": "FEEDBACK_ADAPTIVE_GO",
            "reasons": [
                f"feedback_vs_original_delta={delta:+.4f}",
                f"trigger_count={trigger_count}/{total}",
                f"dc_positive_retained={retained}",
                f"dc_negative_triggered={neg_triggered}",
            ],
        }

    return {
        "decision": "FEEDBACK_ADAPTIVE_WEAK_GO",
        "reasons": [
            f"feedback_vs_original_delta={delta:+.4f}",
            f"trigger_count={trigger_count}/{total}",
            f"extra_generation_rate_estimate={extra_gen:.3f}",
            f"dc_positive_retained={retained}",
            f"dc_negative_triggered={neg_triggered}",
        ],
    }


def overall_decision(summaries: Sequence[Mapping[str, Any]], bug_reasons: Sequence[str]) -> Dict[str, Any]:
    if bug_reasons:
        return {"decision": "BUG_STOP_FIX_REQUIRED", "reasons": list(bug_reasons)}

    if not summaries:
        return {"decision": "BUG_STOP_FIX_REQUIRED", "reasons": ["no_policy_summaries"]}

    best = max(summaries, key=lambda row: float(row["macro_feedback"]))
    best_decision = str(best.get("decision", "FEEDBACK_ADAPTIVE_NO_GO"))
    if best_decision in ("FEEDBACK_ADAPTIVE_GO", "FEEDBACK_ADAPTIVE_WEAK_GO"):
        return {
            "decision": best_decision,
            "policy": best["policy"],
            "reasons": [
                f"best_policy={best['policy']}",
                f"macro_feedback={float(best['macro_feedback']):.4f}",
                f"feedback_vs_original_delta={float(best['feedback_vs_original_delta']):+.4f}",
                f"trigger_count={int(best['trigger_count'])}/128",
            ],
        }

    return {
        "decision": "FEEDBACK_ADAPTIVE_NO_GO",
        "reasons": [
            f"best_policy={best['policy']}",
            f"macro_feedback={float(best['macro_feedback']):.4f}",
            f"policy_decisions={[str(row.get('decision', '')) for row in summaries]}",
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--heldout_dir", type=Path, default=DEFAULT_HELDOUT_DIR)
    parser.add_argument("--gate_audit_dir", type=Path, default=DEFAULT_GATE_AUDIT_DIR)
    parser.add_argument("--longbench_dir", type=Path, default=DEFAULT_LONGBENCH_DIR)
    parser.add_argument("--model_path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output_root", type=Path, default=Path("results"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    heldout_dir = (REPO_ROOT / args.heldout_dir).resolve()
    gate_audit_dir = (REPO_ROOT / args.gate_audit_dir).resolve()
    output_dir = (
        REPO_ROOT / args.output_root / f"feedback_adaptive_dc_selector_replay_{datetime.now():%Y%m%d_%H%M%S}"
    ).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)

    keys = heldout_keys()
    key_set = set(keys)
    bug_reasons: List[str] = []

    quality = load_quality_scores(heldout_dir, key_set)
    if len(quality) != 128:
        bug_reasons.append(f"per_sample_quality_rows={len(quality)}/128")

    original_preds, dc_preds = load_predictions(heldout_dir, key_set)
    if len(original_preds) != 128:
        bug_reasons.append(f"original_predictions={len(original_preds)}/128")
    if len(dc_preds) != 128:
        bug_reasons.append(f"dc_predictions={len(dc_preds)}/128")

    selection = load_selection_audit(heldout_dir, key_set)
    if len(selection) != 128:
        bug_reasons.append(f"selection_audit_rows={len(selection)}/128")

    tokenizer, dataset2prompt = load_tokenizer_and_prompts(args.model_path)
    samples_cache = {task: load_samples(args.longbench_dir, task) for task in TASKS}
    block_cache: Dict[Key, Tuple[str, List[str]]] = {}

    per_sample_rows: List[Dict[str, Any]] = []

    if not bug_reasons:
        for task, sample_index in keys:
            key = (task, sample_index)
            qrow = quality[key]
            sel = selection[key]
            if key not in block_cache:
                sample = samples_cache[task][sample_index]
                block_cache[key] = build_prompt_blocks(
                    tokenizer, dataset2prompt[task], sample, task, 32768, 64
                )
            query_text, block_texts = block_cache[key]
            original_context = context_from_blocks(
                block_texts, sel.get("original_selected_block_ids", [])
            )
            dc_context = context_from_blocks(block_texts, sel.get("dc_selected_block_ids", []))
            original_answer = original_preds[key]
            dc_answer = dc_preds[key]

            features = compute_feedback_features(
                query_text, original_answer, dc_answer, original_context, dc_context
            )
            f1_trigger, f1_reason = policy_f1_trigger(features)
            f2_selector, f2_trigger, f2_reason = policy_f2_choose(features)

            original_score = to_float(qrow.get("original_score"))
            dc_score = to_float(qrow.get("dc_score"))
            f1_score = dc_score if f1_trigger else original_score
            f2_score = dc_score if f2_selector == "dc" else original_score

            per_sample_rows.append({
                "task": task,
                "sample_index": sample_index,
                "sample_id": qrow.get("sample_id", sel.get("sample_id", "")),
                "original_answer": original_answer,
                "dc_answer": dc_answer,
                "original_score": original_score,
                "dc_score": dc_score,
                **features,
                "policy_f1_triggered": f1_trigger,
                "policy_f1_reason": f1_reason,
                "policy_f1_chosen_selector": "dc" if f1_trigger else "original",
                "policy_f1_score": f1_score,
                "policy_f2_triggered": f2_trigger,
                "policy_f2_reason": f2_reason,
                "policy_f2_chosen_selector": f2_selector,
                "policy_f2_score": f2_score,
                "dc_positive": dc_score > original_score + 1e-9,
                "dc_negative": dc_score < original_score - 1e-9,
            })

    policy_summaries: List[Dict[str, Any]] = []
    if per_sample_rows:
        f1_summary = summarize_policy(
            "policy_f1", per_sample_rows,
            sum(1 for row in per_sample_rows if row["policy_f1_triggered"]) / 128.0,
            False,
        )
        f2_summary = summarize_policy(
            "policy_f2", per_sample_rows, 1.0, True,
        )
        f1_summary["decision"] = decide_policy(f1_summary)["decision"]
        f2_summary["decision"] = decide_policy(f2_summary)["decision"]
        policy_summaries = [f1_summary, f2_summary]

    overall = overall_decision(policy_summaries, bug_reasons)

    per_sample_fields = [
        "task", "sample_index", "sample_id",
        "original_answer", "dc_answer", "original_score", "dc_score",
        "original_answer_empty_or_short", "original_answer_generic_or_refusal",
        "original_answer_repeats_question", "answer_terms_supported_by_original_context",
        "query_entities_covered_by_original_context", "answer_entities_covered_by_original_context",
        "original_context_entity_coverage_gap", "dc_context_additional_entity_coverage",
        "original_vs_dc_prediction_disagreement", "dc_answer_terms_supported_by_dc_context",
        "answer_jaccard_original_vs_dc",
        "policy_f1_triggered", "policy_f1_reason", "policy_f1_chosen_selector", "policy_f1_score",
        "policy_f2_triggered", "policy_f2_reason", "policy_f2_chosen_selector", "policy_f2_score",
        "dc_positive", "dc_negative",
    ]
    write_csv(output_dir / "feedback_selector_per_sample.csv", per_sample_rows, per_sample_fields)

    summary_fields = list(policy_summaries[0].keys()) if policy_summaries else ["policy"]
    write_csv(output_dir / "feedback_selector_summary.csv", policy_summaries, summary_fields)

    report_lines = [
        "# Feedback-Adaptive DC Selector Offline Replay",
        "",
        f"- Output: `{output_dir}`",
        f"- Held-out dir: `{heldout_dir}`",
        f"- Gate audit dir: `{gate_audit_dir}`",
        "",
        "## Scope",
        "",
        "- Static frozen gate: **stopped** (not used).",
        "- No generation, no scorer, no gold/task-label/score gating.",
        "- Scores used **only** for offline evaluation.",
        "",
        "## Feedback features (no gold)",
        "",
        "1. original_answer_empty_or_short",
        "2. original_answer_generic_or_refusal",
        "3. original_answer_repeats_question",
        "4. answer_terms_supported_by_original_context",
        "5. query_entities_covered_by_original_context",
        "6. answer_entities_covered_by_original_context",
        "7. original_context_entity_coverage_gap",
        "8. dc_context_additional_entity_coverage",
        "9. original_vs_dc_prediction_disagreement *(F2 / reporting only; increases deployment cost)*",
        "",
        "## Policy definitions",
        "",
        "- **Policy F1 (one-pass feedback trigger)**: switch to DC when original answer looks empty/short, generic/refusal, repeats question, or is very unsupported by original context. Does **not** read DC answer at gate time.",
        "- **Policy F2 (two-answer verifier trigger)**: requires both original and DC answers; uses prediction disagreement plus context-support comparison. **Higher deployment cost** (second generation always required).",
        "",
        "## Policy results",
        "",
    ]

    for summary in policy_summaries:
        report_lines.extend([
            f"### {summary['policy']}",
            "",
            f"- Decision: **{summary['decision']}**",
            f"- Trigger count: **{summary['trigger_count']}/128**",
            f"- HotpotQA original / dc / feedback: {summary['hotpotqa_original_macro']:.4f} / "
            f"{summary['hotpotqa_dc_macro']:.4f} / {summary['hotpotqa_feedback_macro']:.4f}",
            f"- MuSiQue original / dc / feedback: {summary['musique_original_macro']:.4f} / "
            f"{summary['musique_dc_macro']:.4f} / {summary['musique_feedback_macro']:.4f}",
            f"- Macro original / dc / feedback: {summary['macro_original']:.4f} / "
            f"{summary['macro_dc']:.4f} / {summary['macro_feedback']:.4f}",
            f"- Feedback vs original delta: **{summary['feedback_vs_original_delta']:+.4f}**",
            f"- Feedback vs DC delta: {summary['feedback_vs_dc_delta']:+.4f}",
            f"- DC-positive retained: **{summary['dc_positive_retained_count']}** / {summary['dc_positive_count']}",
            f"- DC-negative triggered: **{summary['dc_negative_triggered_count']}** / {summary['dc_negative_count']}",
            f"- Prediction disagreement rate: {summary['prediction_disagreement_rate']:.3f}",
            f"- Extra generation rate estimate: **{summary['extra_generation_rate_estimate']:.3f}**",
            f"- Uses prediction disagreement for gate: **{summary['uses_prediction_disagreement_feature']}**",
            f"- Uses gold/task label/score for gate: **{summary['uses_gold_task_label_score_for_gate']}**",
            "",
        ])

    report_lines.extend([
        "## Overall decision",
        "",
        f"- **{overall['decision']}**",
    ])
    for reason in overall.get("reasons", []):
        report_lines.append(f"- `{reason}`")
    report_lines.append("")
    (output_dir / "feedback_selector_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    decision_lines = [
        "# Feedback-Adaptive DC Selector Decision",
        "",
        f"- Overall: **{overall['decision']}**",
    ]
    for reason in overall.get("reasons", []):
        decision_lines.append(f"- Reason: `{reason}`")
    decision_lines.append("")
    for summary in policy_summaries:
        decision_lines.append(f"## {summary['policy']}")
        decision_lines.append(f"- Decision: **{summary['decision']}**")
        decision_lines.append(f"- Macro feedback: {summary['macro_feedback']:.4f}")
        decision_lines.append(f"- Trigger: {summary['trigger_count']}/128")
        decision_lines.append("")
    (output_dir / "feedback_selector_decision.md").write_text("\n".join(decision_lines) + "\n", encoding="utf-8")

    print(json.dumps({
        "status": "ok" if not bug_reasons else "bug_stop",
        "output_dir": str(output_dir),
        "overall_decision": overall,
        "policy_summaries": policy_summaries,
        "bug_reasons": bug_reasons,
    }, indent=2))


if __name__ == "__main__":
    main()
