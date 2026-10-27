#!/usr/bin/env python3
"""Run the structured parent-page aggregation mechanism ablation with OPT-125M.

This is intentionally a model-backed mechanism proxy, not a native OPT SlimInfer
cache implementation. OPT attention and hidden-state features feed the existing
DRM and structured microblock selectors. Selected parent pages are then packed
in source order for deterministic generation and LongBench scoring.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Set, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bridgekv.rescuekv_scoring import apply_rescue_policy, drm_s0_score
from bridgekv.structured_rescue_scoring import select_structured_rescue_via_microblocks
from longbench_eval import scorer

TASKS = ["qasper", "hotpotqa", "musique", "passage_count", "passage_retrieval_en"]
AGG_RULES = ["max", "mean_top2", "sum_topk", "rank_weighted_sum"]
STRUCTURED_METHODS = ["rh_pyramid_srm", "bridge_graph_srm"]
REQUESTED_MODEL = "facebook/opt-125m"
DEFAULT_LB = "/home/yu-bai/Desktop/jiaqizhang/HeadKV/data/LongBench"
BLOCK_SIZE = 64
PROXY_STATUS = "opt_attention_hidden_state_page_packing_proxy"


def _run_name(method: str, aggregation: str | None = None) -> str:
    return f"{method}__{aggregation}" if aggregation else method


RUN_NAMES = ["fixed_drm_add"] + [
    _run_name(method, aggregation)
    for method in STRUCTURED_METHODS
    for aggregation in AGG_RULES
]


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_samples(longbench_dir: Path, task: str, limit: int) -> List[Dict[str, Any]]:
    candidates = [
        longbench_dir / f"{task}.jsonl",
        longbench_dir / task / "test.jsonl",
        longbench_dir / task / f"{task}.jsonl",
    ]
    for path in candidates:
        if not path.exists():
            continue
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        return rows[:limit]
    raise FileNotFoundError(f"LongBench task not found: {task} under {longbench_dir}")


def _sample_id(sample: Mapping[str, Any], index: int) -> Any:
    return sample.get("_id", sample.get("id", index))


def _query_text(sample: Mapping[str, Any]) -> str:
    for key in ("input", "question", "query", "instruction"):
        value = sample.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def _truncate_middle(input_ids: torch.Tensor, max_tokens: int) -> torch.Tensor:
    if int(input_ids.numel()) <= max_tokens:
        return input_ids
    head = max_tokens // 2
    tail = max_tokens - head
    return torch.cat([input_ids[:head], input_ids[-tail:]], dim=0)


def _page_slices(length: int) -> List[Tuple[int, int]]:
    return [(start, min(length, start + BLOCK_SIZE)) for start in range(0, length, BLOCK_SIZE)]


def _page_texts(tokenizer: Any, input_ids: torch.Tensor) -> List[str]:
    return [
        tokenizer.decode(
            input_ids[start:end].tolist(),
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        for start, end in _page_slices(int(input_ids.numel()))
    ]


def _extract_opt_page_features(
    model: Any,
    input_ids: torch.Tensor,
    query_window: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return damage proxy, query-attention mass, and page representations."""
    device_ids = input_ids.unsqueeze(0).to(model.device)
    with torch.inference_mode():
        outputs = model(
            device_ids,
            use_cache=False,
            output_attentions=True,
            output_hidden_states=True,
            return_dict=True,
        )
    if not outputs.attentions or not outputs.hidden_states:
        raise RuntimeError("OPT feature extraction did not return attentions and hidden states")

    last_attention = outputs.attentions[-1][0].float()
    last_hidden = outputs.hidden_states[-1][0].float()
    q_window = min(max(1, query_window), int(last_attention.shape[-2]))
    token_attention = last_attention[:, -q_window:, :].mean(dim=(0, 1))
    token_damage = token_attention * last_hidden.norm(dim=-1)

    slices = _page_slices(int(input_ids.numel()))
    base = torch.zeros(len(slices), dtype=torch.float32, device=last_hidden.device)
    qk = torch.zeros_like(base)
    reps = torch.zeros(
        len(slices),
        int(last_hidden.shape[-1]),
        dtype=torch.float32,
        device=last_hidden.device,
    )
    for page_id, (start, end) in enumerate(slices):
        base[page_id] = token_damage[start:end].sum()
        qk[page_id] = token_attention[start:end].sum()
        reps[page_id] = last_hidden[start:end].mean(dim=0)

    return base.cpu(), qk.cpu(), reps.cpu()


def _active_page_ids(qk_scores: torch.Tensor, active_ratio: float) -> List[int]:
    num_pages = int(qk_scores.numel())
    if num_pages <= 1:
        return list(range(num_pages))
    target = min(num_pages - 1, max(2, int(round(num_pages * active_ratio))))
    protected = {0, num_pages - 1}
    ranked = sorted(
        range(num_pages),
        key=lambda page_id: (float(qk_scores[page_id]), -page_id),
        reverse=True,
    )
    selected = set(protected)
    for page_id in ranked:
        if len(selected) >= target:
            break
        selected.add(int(page_id))
    return sorted(selected)


def _fixed_drm_selection(
    base: torch.Tensor,
    reps: torch.Tensor,
    active_ids: Sequence[int],
    inactive_ids: Sequence[int],
    rescue_ratio: float,
) -> Tuple[List[int], List[int], Dict[str, Any]]:
    diagnostics: Dict[str, List[str]] = {"warnings": []}
    scores, score_meta = drm_s0_score(
        base,
        reps,
        active_ids,
        inactive_ids,
        alpha_diff=0.5,
        lambda_red=0.25,
        diagnostics=diagnostics,
    )
    result = apply_rescue_policy(
        active_ids=active_ids,
        inactive_ids=inactive_ids,
        rescue_scores=scores,
        active_scores=base,
        block_reps=reps,
        policy="add",
        rescue_ratio=rescue_ratio,
        mmr_selection=True,
        eta_rescue=0.25,
        eta_active=0.25,
        seed=0,
        diagnostics=diagnostics,
    )
    rescued = [int(value) for value in result.rescued_ids.tolist()]
    final_ids = [int(value) for value in result.final_ids.tolist()]
    meta = {
        "warnings": diagnostics.get("warnings", []),
        "used_base": bool(float(score_meta.get("used_base", torch.tensor([0.0]))[0])),
    }
    return rescued, final_ids, meta


def _structured_cfg(
    method: str,
    aggregation: str,
    input_ids: torch.Tensor,
    page_texts: Sequence[str],
    query_text: str,
    args: argparse.Namespace,
    num_heads: int,
    num_layers: int,
) -> Dict[str, Any]:
    cfg: Dict[str, Any] = {
        "rescue_score": method,
        "rescue_policy": "add",
        "rescue_granularity": "microblock",
        "microblock_size": 32,
        "microblock_parent_aggregation": aggregation,
        "original_prompt_len": int(input_ids.numel()),
        "block_texts": list(page_texts),
        "query_text": query_text,
        "alpha_diff": 0.5,
        "lambda_red": 0.25,
        "rh_weight": 0.50,
        "pyramid_weight": 0.20,
        "bridge_weight": 0.45,
        "bridge_min_link": 0.05,
        "bridge_top_candidates": 256,
        "layer_idx": max(0, num_layers - 1),
        "num_layers": num_layers,
        "_proxy_num_heads": num_heads,
        "strict_structured_path_check": bool(args.strict_structured_path_check),
    }
    return cfg


def _selected_token_count(input_ids: torch.Tensor, page_ids: Sequence[int]) -> int:
    slices = _page_slices(int(input_ids.numel()))
    return sum(slices[page_id][1] - slices[page_id][0] for page_id in page_ids)


def _packed_ids(input_ids: torch.Tensor, page_ids: Sequence[int]) -> torch.Tensor:
    slices = _page_slices(int(input_ids.numel()))
    pieces = [input_ids[slices[page_id][0]:slices[page_id][1]] for page_id in sorted(set(page_ids))]
    return torch.cat(pieces, dim=0) if pieces else input_ids[-1:].clone()


def _generate_unique(
    model: Any,
    tokenizer: Any,
    packed_by_key: Mapping[Tuple[int, ...], torch.Tensor],
    max_new_tokens: int,
) -> Dict[Tuple[int, ...], str]:
    keys = list(packed_by_key)
    encoded = {"input_ids": [packed_by_key[key].tolist() for key in keys]}
    batch = tokenizer.pad(encoded, padding=True, return_tensors="pt").to(model.device)
    input_width = int(batch.input_ids.shape[1])
    with torch.inference_mode():
        generated = model.generate(
            batch.input_ids,
            attention_mask=batch.attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    return {
        key: tokenizer.decode(generated[index, input_width:], skip_special_tokens=True)
        for index, key in enumerate(keys)
    }


def _jaccard(left: Set[Any], right: Set[Any]) -> float:
    if not left and not right:
        return 1.0
    return len(left & right) / max(1, len(left | right))


def _mean(values: Sequence[float]) -> float:
    return statistics.mean(values) if values else 0.0


def _write_run_scores(
    root: Path,
    task_scores: Mapping[str, Mapping[str, float]],
    completed_tasks: Sequence[str],
    sample_count: int,
) -> None:
    for run_name in RUN_NAMES:
        rows = [
            {
                "dataset": task,
                "score": task_scores[run_name][task],
                "num_samples": sample_count,
                "status": "ok",
                "error": "",
            }
            for task in completed_tasks
        ]
        _write_csv(
            root / "runs" / run_name / "longbench_scores.csv",
            rows,
            ["dataset", "score", "num_samples", "status", "error"],
        )


def _write_summary(
    root: Path,
    task_scores: Mapping[str, Mapping[str, float]],
    sample_count: int,
) -> List[Dict[str, Any]]:
    fixed_macro = _mean(list(task_scores["fixed_drm_add"].values()))
    summary_rows: List[Dict[str, Any]] = []
    breakdown_rows: List[Dict[str, Any]] = []
    for run_name in RUN_NAMES:
        macro = _mean(list(task_scores[run_name].values()))
        summary_rows.append({
            "method": run_name,
            "task_macro_avg": round(macro, 4),
            "n_result_dirs": 1,
            "delta_vs_sliminfer": round(macro - fixed_macro, 4),
            "delta_vs_fixed_drm_add": round(macro - fixed_macro, 4),
            "evaluation_status": PROXY_STATUS,
        })
        for task in TASKS:
            breakdown_rows.append({
                "method": run_name,
                "task": task,
                "score": task_scores[run_name][task],
                "num_samples": sample_count,
                "evaluation_status": PROXY_STATUS,
            })

    summary_dir = root / "summary"
    _write_csv(
        summary_dir / "summary.csv",
        summary_rows,
        [
            "method", "task_macro_avg", "n_result_dirs", "delta_vs_sliminfer",
            "delta_vs_fixed_drm_add", "evaluation_status",
        ],
    )
    _write_csv(
        summary_dir / "method_task_breakdown.csv",
        breakdown_rows,
        ["method", "task", "score", "num_samples", "evaluation_status"],
    )
    _write_csv(
        summary_dir / "delta_vs_fixed_drm_add.csv",
        summary_rows,
        ["method", "task_macro_avg", "delta_vs_fixed_drm_add", "evaluation_status"],
    )
    return summary_rows


def _build_audits(
    root: Path,
    final_rows: Mapping[str, List[Dict[str, Any]]],
    marker_rows: Sequence[Dict[str, Any]],
    task_scores: Mapping[str, Mapping[str, float]],
) -> List[Dict[str, Any]]:
    audit_dir = root / "aggregation_audit"
    _write_csv(
        audit_dir / "01_path_marker_audit.csv",
        marker_rows,
        [
            "sample_id", "task", "method", "run_name", "parent_page_aggregation",
            "entered_structured_selector", "entered_real_fixed_block_path",
            "entered_opt_proxy_selection_path", "entered_cache_construction_path",
            "used_microblock_first_selector", "used_candidate_score_dump",
            "used_final_selection_dump", "evaluation_status",
        ],
    )

    microblock_rows = [
        {
            "sample_id": row["sample_id"],
            "task": row["task"],
            "run_name": run_name,
            "method": row["method"],
            "parent_page_aggregation": row.get("parent_page_aggregation", ""),
            "selected_microblock_count": row.get("selected_microblock_count", 0),
            "selected_parent_page_count": len(row.get("selected_parent_page_ids", [])),
            "evaluation_status": PROXY_STATUS,
        }
        for run_name, rows in final_rows.items()
        if run_name != "fixed_drm_add"
        for row in rows
    ]
    _write_csv(
        audit_dir / "02_microblock_count_audit.csv",
        microblock_rows,
        [
            "sample_id", "task", "run_name", "method", "parent_page_aggregation",
            "selected_microblock_count", "selected_parent_page_count", "evaluation_status",
        ],
    )

    row_index = {
        run_name: {(str(row["task"]), str(row["sample_id"])): row for row in rows}
        for run_name, rows in final_rows.items()
    }
    micro_overlap: List[Dict[str, Any]] = []
    page_overlap: List[Dict[str, Any]] = []
    erasure_rows: List[Dict[str, Any]] = []
    comparison_rows: List[Dict[str, Any]] = []

    for aggregation in AGG_RULES:
        rh_run = _run_name("rh_pyramid_srm", aggregation)
        bridge_run = _run_name("bridge_graph_srm", aggregation)
        keys = sorted(set(row_index[rh_run]) & set(row_index[bridge_run]))
        page_values: List[float] = []
        micro_values: List[float] = []
        collapsed = 0
        for task, sample_id in keys:
            rh = row_index[rh_run][(task, sample_id)]
            bridge = row_index[bridge_run][(task, sample_id)]
            rh_micro = set(str(value) for value in rh["selected_microblock_ids_before_expansion"])
            bridge_micro = set(str(value) for value in bridge["selected_microblock_ids_before_expansion"])
            rh_pages = set(int(value) for value in rh["selected_parent_page_ids"])
            bridge_pages = set(int(value) for value in bridge["selected_parent_page_ids"])
            micro_j = _jaccard(rh_micro, bridge_micro)
            page_j = _jaccard(rh_pages, bridge_pages)
            micro_values.append(micro_j)
            page_values.append(page_j)
            is_erased = micro_j < 1.0 and page_j == 1.0
            collapsed += int(page_j == 1.0)
            base = {
                "task": task,
                "sample_id": sample_id,
                "aggregation": aggregation,
                "run_a": rh_run,
                "run_b": bridge_run,
                "evaluation_status": PROXY_STATUS,
            }
            micro_overlap.append({
                **base,
                "jaccard": round(micro_j, 6),
                "n_a": len(rh_micro),
                "n_b": len(bridge_micro),
            })
            page_overlap.append({
                **base,
                "jaccard": round(page_j, 6),
                "n_a": len(rh_pages),
                "n_b": len(bridge_pages),
            })
            if is_erased:
                erasure_rows.append({
                    **base,
                    "microblock_jaccard": round(micro_j, 6),
                    "page_jaccard": round(page_j, 6),
                    "rh_microblocks": json.dumps(sorted(rh_micro)),
                    "bridge_microblocks": json.dumps(sorted(bridge_micro)),
                    "shared_parent_pages": json.dumps(sorted(rh_pages)),
                })

        rh_macro = _mean(list(task_scores[rh_run].values()))
        bridge_macro = _mean(list(task_scores[bridge_run].values()))
        fixed_macro = _mean(list(task_scores["fixed_drm_add"].values()))
        comparison_rows.append({
            "aggregation": aggregation,
            "rh_run": rh_run,
            "bridge_run": bridge_run,
            "page_jaccard_mean": round(_mean(page_values), 4),
            "page_jaccard_median": round(statistics.median(page_values), 4) if page_values else "",
            "page_both_collapsed_frac": round(collapsed / max(1, len(page_values)), 4),
            "microblock_jaccard_mean": round(_mean(micro_values), 4),
            "microblock_jaccard_median": round(statistics.median(micro_values), 4) if micro_values else "",
            "rh_task_macro_avg": round(rh_macro, 4),
            "bridge_task_macro_avg": round(bridge_macro, 4),
            "structured_mean_macro": round((rh_macro + bridge_macro) / 2.0, 4),
            "delta_mean_vs_fixed": round((rh_macro + bridge_macro) / 2.0 - fixed_macro, 4),
            "evaluation_status": PROXY_STATUS,
        })

    overlap_fields = [
        "task", "sample_id", "aggregation", "run_a", "run_b", "jaccard",
        "n_a", "n_b", "evaluation_status",
    ]
    _write_csv(audit_dir / "03_selection_overlap_microblock.csv", micro_overlap, overlap_fields)
    _write_csv(audit_dir / "04_selection_overlap_page.csv", page_overlap, overlap_fields)
    _write_csv(
        audit_dir / "05_expansion_erasure_cases.csv",
        erasure_rows,
        [
            "task", "sample_id", "aggregation", "run_a", "run_b",
            "microblock_jaccard", "page_jaccard", "rh_microblocks",
            "bridge_microblocks", "shared_parent_pages", "evaluation_status",
        ],
    )
    _write_csv(
        audit_dir / "06_parent_aggregation_comparison.csv",
        comparison_rows,
        [
            "aggregation", "rh_run", "bridge_run", "page_jaccard_mean",
            "page_jaccard_median", "page_both_collapsed_frac",
            "microblock_jaccard_mean", "microblock_jaccard_median",
            "rh_task_macro_avg", "bridge_task_macro_avg", "structured_mean_macro",
            "delta_mean_vs_fixed", "evaluation_status",
        ],
    )

    task_rows: List[Dict[str, Any]] = []
    for run_name in RUN_NAMES:
        method, aggregation = (
            run_name.split("__", 1) if "__" in run_name else (run_name, "baseline")
        )
        for task in TASKS:
            task_rows.append({
                "method": method,
                "aggregation": aggregation,
                "run_name": run_name,
                "task": task,
                "score": task_scores[run_name][task],
                "evaluation_status": PROXY_STATUS,
            })
    _write_csv(
        audit_dir / "07_task_breakdown_by_aggregation.csv",
        task_rows,
        ["method", "aggregation", "run_name", "task", "score", "evaluation_status"],
    )
    return comparison_rows


def _write_recommendation(
    root: Path,
    comparison_rows: Sequence[Mapping[str, Any]],
    task_scores: Mapping[str, Mapping[str, float]],
) -> None:
    best_overlap = min(comparison_rows, key=lambda row: float(row["page_jaccard_mean"]))
    best_quality = max(comparison_rows, key=lambda row: float(row["structured_mean_macro"]))
    max_row = next(row for row in comparison_rows if row["aggregation"] == "max")
    overlap_ties = [
        str(row["aggregation"])
        for row in comparison_rows
        if abs(float(row["page_jaccard_mean"]) - float(best_overlap["page_jaccard_mean"])) < 1e-9
    ]
    quality_ties = [
        str(row["aggregation"])
        for row in comparison_rows
        if abs(float(row["structured_mean_macro"]) - float(best_quality["structured_mean_macro"])) < 1e-9
    ]

    musique_improvements: List[Tuple[str, str, float]] = []
    for method in STRUCTURED_METHODS:
        max_score = task_scores[_run_name(method, "max")]["musique"]
        for aggregation in AGG_RULES:
            if aggregation == "max":
                continue
            score = task_scores[_run_name(method, aggregation)]["musique"]
            musique_improvements.append((method, aggregation, score - max_score))
    best_musique = max(musique_improvements, key=lambda item: item[2])

    overlap_gain = float(max_row["page_jaccard_mean"]) - float(best_overlap["page_jaccard_mean"])
    quality_gap = (
        float(best_quality["structured_mean_macro"])
        - float(max_row["structured_mean_macro"])
    )
    rerun_justified = overlap_gain >= 0.02 and quality_gap >= -0.50
    direction = "continue to a Qwen/Mistral 32-sample confirmation" if rerun_justified else "treat this line as a negative ablation"
    overlap_answer = (
        f"**None; all rules tie** at mean page Jaccard {best_overlap['page_jaccard_mean']}"
        if len(overlap_ties) == len(comparison_rows)
        else f"**{best_overlap['aggregation']}** (mean page Jaccard {best_overlap['page_jaccard_mean']}; "
        f"max baseline {max_row['page_jaccard_mean']})"
    )
    quality_answer = (
        f"**All rules tie** at structured mean macro {best_quality['structured_mean_macro']}"
        if len(quality_ties) == len(comparison_rows)
        else f"**{best_quality['aggregation']}** (structured mean macro "
        f"{best_quality['structured_mean_macro']})"
    )

    lines = [
        "# Parent page aggregation ablation recommendation",
        "",
        "## Scope",
        "",
        f"- Model: `{REQUESTED_MODEL}`.",
        f"- Evaluation status: `{PROXY_STATUS}`.",
        "- This is a real OPT-125M forward/generation mechanism proxy. It is not a native OPT SlimInfer cache path.",
        "- OPT attention and hidden-state page features feed the existing selectors; selected parent pages are packed in source order.",
        "",
        "## Required Answers",
        "",
        f"1. Aggregation reducing RH vs Bridge page overlap most: {overlap_answer}.",
        f"2. Aggregation preserving proxy quality best: {quality_answer}.",
        f"3. Musique improves under an alternative aggregation: **{best_musique[2] > 0.0}**. "
        f"Best delta vs the same method's max rule is {best_musique[2]:+.4f} "
        f"for `{best_musique[0]}__{best_musique[1]}`.",
        f"4. Qwen/Mistral 32-sample rerun justified: **{rerun_justified}**. "
        f"Page-overlap reduction is {overlap_gain:+.4f}; proxy quality change vs max is {quality_gap:+.4f}.",
        f"5. Direction: **{direction}**.",
        "",
        "## Next Research Direction",
        "",
        "Stop optimizing independent parent-page aggregation. The next candidate should select structured evidence rather than another scalar `Score(page)` variant.",
        "",
        "### Direction A: Path-aware Evidence Rescue",
        "",
        "Target multi-hop evidence chains directly:",
        "",
        "`query -> bridge page -> answer page`",
        "",
        "Use a pairwise path objective such as:",
        "",
        "`PathScore(P_i, P_j) = QueryLink(q, P_i) + BridgeLink(P_i, P_j) + AnswerEvidence(P_j) - Redundancy(P_i, P_j, ActiveSet)`",
        "",
        "The rescue unit becomes a page pair or short path, not an independently ranked page. "
        "This is the higher-novelty direction for `musique`, `hotpotqa`, and `2wikimqa`, but it requires a new selector and budget-aware path allocation.",
        "",
        "### Direction B: Evidence-density Rescue",
        "",
        "Promote pages containing sustained local evidence rather than one isolated high-scoring microblock:",
        "",
        "`DensityScore(P) = high-score microblock count + continuity bonus + local span coverage - redundancy`",
        "",
        "This is the lower-risk implementation direction because it transfers the SpanDRM evidence-density observation into strict page/block KV rescue without requiring pairwise path search.",
        "",
        "**Recommended order:** implement Evidence-density Rescue first as the cheaper strict-KV mechanism test. "
        "Pursue Path-aware Evidence Rescue next if density selection passes wiring/diversity checks but multi-hop quality remains limited.",
        "",
        "Both directions are proposals only; neither is validated by this ablation.",
        "",
        "## Practical Decision",
        "",
        "1. Do not run the Qwen/Mistral parent-aggregation 32-sample evaluation.",
        "2. Record parent aggregation as a negative ablation.",
        "3. Keep `fixed_drm_add` as the current strict-KV primary method.",
        "4. Keep the current RH/Bridge variants as diagnostics or negative ablations, not promoted methods.",
        "5. Move the next method iteration to Path-aware Evidence Rescue or Evidence-density Rescue.",
        "",
        "## Interpretation Limit",
        "",
        "Do not use these OPT proxy scores as a production SlimInfer quality claim. "
        "They answer whether parent aggregation changes selections enough to justify a native 32-sample rerun.",
    ]
    report = "\n".join(lines) + "\n"
    (root / "aggregation_audit" / "08_recommendation.md").write_text(report, encoding="utf-8")
    (root / "parent_aggregation_opt125m_32_report.md").write_text(report, encoding="utf-8")


def _validate_outputs(root: Path, expected_samples: int) -> None:
    required = [
        root / "summary" / "summary.csv",
        root / "summary" / "method_task_breakdown.csv",
        root / "summary" / "delta_vs_fixed_drm_add.csv",
        root / "aggregation_audit" / "01_path_marker_audit.csv",
        root / "aggregation_audit" / "02_microblock_count_audit.csv",
        root / "aggregation_audit" / "03_selection_overlap_microblock.csv",
        root / "aggregation_audit" / "04_selection_overlap_page.csv",
        root / "aggregation_audit" / "05_expansion_erasure_cases.csv",
        root / "aggregation_audit" / "06_parent_aggregation_comparison.csv",
        root / "aggregation_audit" / "07_task_breakdown_by_aggregation.csv",
        root / "aggregation_audit" / "08_recommendation.md",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise RuntimeError(f"Missing required outputs: {missing}")
    for run_name in RUN_NAMES:
        score_path = root / "runs" / run_name / "longbench_scores.csv"
        if not score_path.exists():
            raise RuntimeError(f"Missing score file: {score_path}")
        if run_name == "fixed_drm_add":
            continue
        for filename in ("candidate_score_dump.jsonl", "final_selection_dump.jsonl", "path_marker.jsonl"):
            path = root / "runs" / run_name / filename
            if not path.exists() or path.stat().st_size == 0:
                raise RuntimeError(f"Missing structured dump: {path}")
        final_count = sum(
            1 for line in (root / "runs" / run_name / "final_selection_dump.jsonl").open(encoding="utf-8")
            if line.strip()
        )
        if final_count != expected_samples:
            raise RuntimeError(
                f"{run_name}: expected {expected_samples} final rows, found {final_count}"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="")
    parser.add_argument("--model_path", default=REQUESTED_MODEL)
    parser.add_argument("--longbench_dir", default=DEFAULT_LB)
    parser.add_argument("--max_samples", type=int, default=32)
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--max_source_tokens", type=int, default=1536)
    parser.add_argument("--query_window", type=int, default=16)
    parser.add_argument("--active_ratio", type=float, default=0.50)
    parser.add_argument("--rescue_ratio", type=float, default=0.05)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["fp16", "fp32"], default="fp16")
    parser.add_argument("--allow_download", action="store_true")
    parser.add_argument(
        "--strict_structured_path_check",
        type=lambda value: str(value).lower() not in {"0", "false", "no"},
        default=True,
    )
    args = parser.parse_args()

    if args.max_samples <= 0 or args.max_samples > 32:
        raise ValueError("--max_samples must be between 1 and 32 for this mechanism ablation")
    if args.max_source_tokens + args.max_new_tokens > 2048:
        raise ValueError("OPT-125M max position limit exceeded")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = (
        Path(args.output_dir)
        if args.output_dir
        else REPO_ROOT / "results" / f"parent_aggregation_ablation_opt125m_32_{stamp}"
    )
    root.mkdir(parents=True, exist_ok=True)
    for run_name in RUN_NAMES:
        run_dir = root / "runs" / run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        for filename in ("longbench_predictions.jsonl", "candidate_score_dump.jsonl", "final_selection_dump.jsonl", "path_marker.jsonl"):
            path = run_dir / filename
            if path.exists():
                path.unlink()
    (root / "audit").mkdir(exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if args.dtype == "fp16" and device.type == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        local_files_only=not args.allow_download,
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        attn_implementation="eager",
        local_files_only=not args.allow_download,
    ).to(device).eval()

    prompt_templates = _read_json(REPO_ROOT / "configs" / "dataset2prompt.json")
    task_scores: Dict[str, Dict[str, float]] = {run_name: {} for run_name in RUN_NAMES}
    predictions: Dict[str, Dict[str, List[str]]] = {
        run_name: defaultdict(list) for run_name in RUN_NAMES
    }
    answers: Dict[str, List[Any]] = defaultdict(list)
    lengths: Dict[str, List[int]] = defaultdict(list)
    all_classes: Dict[str, List[Any]] = defaultdict(list)
    final_rows: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    marker_rows: List[Dict[str, Any]] = []

    manifest: Dict[str, Any] = {
        "output_root": str(root),
        "requested_model": args.model_path,
        "effective_model_type": model.config.model_type,
        "evaluation_status": PROXY_STATUS,
        "native_sliminfer_cache": False,
        "tasks": TASKS,
        "max_samples_per_task": args.max_samples,
        "methods": RUN_NAMES,
        "rescue_policy": "add",
        "rescue_granularity": "microblock",
        "microblock_size": 32,
        "microblock_parent_aggregation": AGG_RULES,
        "rh_weight": 0.50,
        "pyramid_weight": 0.20,
        "bridge_weight": 0.45,
        "strict_structured_path_check": args.strict_structured_path_check,
        "max_source_tokens": args.max_source_tokens,
        "max_new_tokens": args.max_new_tokens,
        "active_ratio": args.active_ratio,
        "rescue_ratio": args.rescue_ratio,
        "started_at": datetime.now().isoformat(),
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    start_time = time.time()
    for task_index, task in enumerate(TASKS, start=1):
        samples = _load_samples(Path(args.longbench_dir), task, args.max_samples)
        print(f"[{task_index}/{len(TASKS)}] {task}: {len(samples)} samples", flush=True)
        for sample_index, sample in enumerate(samples):
            sample_id = _sample_id(sample, sample_index)
            prompt = prompt_templates[task].format(**sample)
            source_ids = tokenizer(prompt, add_special_tokens=True, return_tensors="pt").input_ids[0]
            source_ids = _truncate_middle(source_ids, args.max_source_tokens).cpu()
            block_texts = _page_texts(tokenizer, source_ids)
            base, qk, reps = _extract_opt_page_features(model, source_ids, args.query_window)
            active_ids = _active_page_ids(qk, args.active_ratio)
            inactive_ids = [page_id for page_id in range(int(base.numel())) if page_id not in set(active_ids)]
            fixed_rescued, fixed_final, fixed_meta = _fixed_drm_selection(
                base, reps, active_ids, inactive_ids, args.rescue_ratio,
            )
            target_m = len(fixed_rescued)

            sample_selections: Dict[str, Dict[str, Any]] = {}
            fixed_row = {
                "sample_id": sample_id,
                "task": task,
                "method": "fixed_drm_add",
                "run_name": "fixed_drm_add",
                "active_ids": active_ids,
                "rescued_ids": fixed_rescued,
                "final_ids": fixed_final,
                "selected_token_count": _selected_token_count(source_ids, fixed_final),
                "selected_page_count": len(fixed_final),
                "selected_block_count": len(fixed_final),
                "selected_microblock_count": 0,
                "final_cache_granularity": "page_text_proxy",
                "expansion_ratio": 1.0,
                "evaluation_status": PROXY_STATUS,
                "selector_meta": fixed_meta,
            }
            final_rows["fixed_drm_add"].append(fixed_row)
            _append_jsonl(root / "audit" / "fixed_drm_add.jsonl", fixed_row)
            sample_selections["fixed_drm_add"] = fixed_row

            for method in STRUCTURED_METHODS:
                for aggregation in AGG_RULES:
                    run_name = _run_name(method, aggregation)
                    cfg = _structured_cfg(
                        method,
                        aggregation,
                        source_ids,
                        block_texts,
                        _query_text(sample),
                        args,
                        int(model.config.num_attention_heads),
                        int(model.config.num_hidden_layers),
                    )
                    diagnostics: Dict[str, List[str]] = {"warnings": []}
                    rescued, mb_meta, candidate_rows = select_structured_rescue_via_microblocks(
                        cfg,
                        base,
                        qk,
                        reps,
                        active_ids,
                        inactive_ids,
                        target_m=target_m,
                        diagnostics=diagnostics,
                    )
                    final_ids = sorted(set(active_ids) | set(int(value) for value in rescued))
                    run_dir = root / "runs" / run_name
                    for candidate in candidate_rows:
                        candidate.update({
                            "sample_id": sample_id,
                            "task": task,
                            "method": method,
                            "run_name": run_name,
                            "evaluation_status": PROXY_STATUS,
                        })
                        _append_jsonl(run_dir / "candidate_score_dump.jsonl", candidate)

                    final_row = {
                        "sample_id": sample_id,
                        "task": task,
                        "method": method,
                        "run_name": run_name,
                        "active_ids": active_ids,
                        "rescued_ids": [int(value) for value in rescued],
                        "final_ids": final_ids,
                        "selected_token_count": _selected_token_count(source_ids, final_ids),
                        "selected_page_count": len(final_ids),
                        "selected_block_count": len(final_ids),
                        "selected_microblock_count": int(mb_meta.get("selected_microblock_count", 0)),
                        "selected_microblock_ids_before_expansion": mb_meta.get(
                            "selected_microblock_ids_before_expansion", []
                        ),
                        "parent_page_aggregation": aggregation,
                        "selected_parent_page_ids": mb_meta.get("selected_parent_page_ids", []),
                        "parent_page_details": mb_meta.get("parent_page_details", []),
                        "physical_units_after_expansion": mb_meta.get(
                            "physical_units_after_expansion", rescued
                        ),
                        "final_cache_granularity": "page_text_proxy",
                        "expansion_ratio": float(mb_meta.get("expansion_ratio", 1.0)),
                        "evaluation_status": PROXY_STATUS,
                        "warnings": diagnostics.get("warnings", []),
                    }
                    marker = {
                        "sample_id": sample_id,
                        "task": task,
                        "method": method,
                        "run_name": run_name,
                        "parent_page_aggregation": aggregation,
                        "entered_structured_selector": True,
                        "entered_real_fixed_block_path": False,
                        "entered_opt_proxy_selection_path": True,
                        "entered_cache_construction_path": False,
                        "used_microblock_first_selector": True,
                        "used_candidate_score_dump": True,
                        "used_final_selection_dump": True,
                        "evaluation_status": PROXY_STATUS,
                    }
                    if args.strict_structured_path_check:
                        if final_row["selected_microblock_count"] <= 0:
                            raise RuntimeError(f"{run_name}: selected_microblock_count == 0")
                        if not candidate_rows:
                            raise RuntimeError(f"{run_name}: candidate score rows missing")
                        if not final_row["selected_parent_page_ids"]:
                            raise RuntimeError(f"{run_name}: selected parent pages missing")

                    _append_jsonl(run_dir / "final_selection_dump.jsonl", final_row)
                    _append_jsonl(run_dir / "path_marker.jsonl", marker)
                    _append_jsonl(root / "audit" / f"{run_name}.jsonl", final_row)
                    final_rows[run_name].append(final_row)
                    marker_rows.append(marker)
                    sample_selections[run_name] = final_row

            packed_by_key: Dict[Tuple[int, ...], torch.Tensor] = {}
            key_by_run: Dict[str, Tuple[int, ...]] = {}
            for run_name, selection in sample_selections.items():
                key = tuple(int(value) for value in selection["final_ids"])
                key_by_run[run_name] = key
                packed_by_key.setdefault(key, _packed_ids(source_ids, key))
            generated_by_key = _generate_unique(
                model, tokenizer, packed_by_key, args.max_new_tokens,
            )

            for run_name in RUN_NAMES:
                prediction = generated_by_key[key_by_run[run_name]]
                predictions[run_name][task].append(prediction)
                pred_row = {
                    "sample_id": sample_id,
                    "sample_index": sample_index,
                    "task": task,
                    "pred": prediction,
                    "answers": sample["answers"],
                    "all_classes": sample.get("all_classes", []),
                    "length": int(source_ids.numel()),
                    "selected_page_ids": sample_selections[run_name]["final_ids"],
                    "selected_token_count": sample_selections[run_name]["selected_token_count"],
                    "evaluation_status": PROXY_STATUS,
                }
                _append_jsonl(
                    root / "runs" / run_name / "longbench_predictions.jsonl",
                    pred_row,
                )
            answers[task].append(sample["answers"])
            lengths[task].append(int(source_ids.numel()))
            all_classes[task] = sample.get("all_classes", all_classes[task])

            if (sample_index + 1) % 8 == 0 or sample_index + 1 == len(samples):
                elapsed = time.time() - start_time
                print(
                    f"  {task}: {sample_index + 1}/{len(samples)} "
                    f"elapsed={elapsed / 60.0:.1f}m",
                    flush=True,
                )

        for run_name in RUN_NAMES:
            task_scores[run_name][task] = scorer(
                task,
                predictions[run_name][task],
                answers[task],
                lengths[task],
                all_classes[task],
            )
        _write_run_scores(
            root,
            task_scores,
            TASKS[:task_index],
            args.max_samples,
        )
        print(
            f"  scores: fixed={task_scores['fixed_drm_add'][task]:.2f}, "
            f"rh_max={task_scores['rh_pyramid_srm__max'][task]:.2f}, "
            f"bridge_max={task_scores['bridge_graph_srm__max'][task]:.2f}",
            flush=True,
        )

    summary_rows = _write_summary(root, task_scores, args.max_samples)
    comparison_rows = _build_audits(root, final_rows, marker_rows, task_scores)
    _write_recommendation(root, comparison_rows, task_scores)
    _validate_outputs(root, len(TASKS) * args.max_samples)

    manifest.update({
        "completed_at": datetime.now().isoformat(),
        "elapsed_seconds": round(time.time() - start_time, 2),
        "status": "complete",
        "summary": summary_rows,
    })
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"OUT_ROOT={root}", flush=True)


if __name__ == "__main__":
    main()
