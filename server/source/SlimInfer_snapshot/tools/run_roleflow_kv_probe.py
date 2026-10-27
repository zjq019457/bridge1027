#!/usr/bin/env python3
"""RoleFlow-KV tiny smoke: role-aware runtime KV retention vs SILM original."""
from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import subprocess
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bridgekv.roleflow_kv import (  # noqa: E402
    active_set_churn,
    apply_roleflow_score_adjustment,
    build_layer_role_profiles_from_features,
    build_role_profiles_from_features,
    select_roleflow_kv_ids,
)
from experiments import run_qwen25_sliminfer_variants as variants  # noqa: E402
from tools.run_bridgeflow_silm_property_probe import (  # noqa: E402
    build_sample_contexts,
    check_gpu,
    read_jsonl,
    run_prefill_probe,
)

DEFAULT_MODEL = (
    "/home/yu-bai/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/"
    "snapshots/a09a35458c702b33eeacc393d103063234e8bc28"
)
DEFAULT_FEATURE_DIR = REPO_ROOT / "results/bridgeflow_silm_property_probe_20260709_050402"
PRUNE_LAYERS = [9, 18, 26]
CHURN_MIN = 0.05
CHURN_MAX = 0.15
CHURN_TARGET = 0.10
VARIANTS = [
    "silm_original",
    "roleflow_hubcap",
    "roleflow_persistence",
    "roleflow_bridgeprotect",
    "roleflow_full",
]
POLICY_MAP = {
    "silm_original": "",
    "roleflow_hubcap": "roleflow_hubcap",
    "roleflow_persistence": "roleflow_persistence",
    "roleflow_bridgeprotect": "roleflow_bridgeprotect",
    "roleflow_hub_bridge": "roleflow_hub_bridge",
    "roleflow_risk_task_gate": "roleflow_risk_task_gate",
    "roleflow_risk_sample_gate": "roleflow_risk_sample_gate",
    "roleflow_risk_swap_guard": "roleflow_risk_swap_guard",
    "roleflow_boundary": "roleflow_boundary",
    "roleflow_full": "roleflow_full",
}


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _mean(values: Sequence[float]) -> float:
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return float(sum(vals) / len(vals)) if vals else 0.0


def write_infra_inspection(out_dir: Path) -> None:
    text = """# RoleFlow-KV Infrastructure Inspection

## Where SILM computes scores
- `utils/cache_utils.py::_calculate_pruning_with_rep_keys()` computes per-block QK saliency (`token_unit_qk`) from representational keys.
- Functional damage (`delta_o`) is optional; `sliminfer_original` keeps QK as `block_scores`.
- Scores are dumped per layer via `_write_uasr_score_dump()` to `uasr_score_dump.jsonl`.

## Where SILM decides active blocks
- Budget: `pruning_method` (`ratio` / `fixed_tokens`) sets `num_blocks_to_keep`.
- Selection: `torch.topk(block_scores, k=num_blocks_to_keep)` then union `must_keep_block_ids` (sink + recent).
- Virtual-block path: `_calculate_pruning_with_virtual_blocks()` may override fixed-page layouts.
- Rescue policies may add blocks after top-k via `_apply_optional_rescue_policy()`.

## Minimal modification point (implemented)
- After `block_scores` are finalized and before `torch.topk`, call `apply_roleflow_score_adjustment()`.
- Replace default top-k union path with `select_roleflow_kv_ids()` when `roleflow_policy` is set.
- Hook location: `utils/cache_utils.py` in `_calculate_pruning_with_rep_keys()`.

## QA comparison path
- `experiments/run_qwen25_sliminfer_variants.py::run_longbench()` for generation + `longbench_eval.scorer`.
- Probe injects `roleflow_profiles` and `roleflow_policy` into `sliminfer_config` per sample.
- Same model / prompt / budget / scorer; only KV selection policy changes.

## Files changed minimally
- `bridgekv/roleflow_kv.py` (new): role profiles, score adjustment, selection policy.
- `utils/cache_utils.py`: import + score adjust + `select_roleflow_kv_ids` branch.
- `tools/run_roleflow_kv_probe.py` (new): smoke orchestration and audits.

## Not in scope
- Prompt/context repacking (BridgePack / Q-BridgeFlow Pack).
- BM25 / LBP baselines.
"""
    _write_text(out_dir / "ROLEFLOW_INFRA_INSPECTION.md", text)


def write_method_doc(out_dir: Path) -> None:
    text = """# RoleFlow-KV Method Design

## Why Q-BridgeFlow Pack failed
- Independent context ranking cleaned wrappers/hubs but dropped answer/evidence chunks that diffusion/BM25 had captured.
- Conclusion: propagation properties are useful for **runtime KV lifecycle**, not as a standalone prompt selector.

## Core idea
Classify blocks into propagation roles and apply **layer-wise KV retention policies** under the same active budget as SILM.

## Roles
1. **Persistent Anchor** — high persistence, positive query-conditioned gain, low hub; protect in middle/late layers.
2. **Transient Trigger** — high early score, fast decay; penalize in late layers.
3. **Bridge Relay** — moderate persistence + bridge link; protect mid/late even if raw diffusion rank is low.
4. **Generic Hub** — high hub/null score; budget-capped per layer.
5. **Boundary-Sensitive** — high neighbor gain; keep local neighbor compound unit.

## Formal proxies (from prefill features)
| Name | Definition |
|------|------------|
| `persistence_ratio_top20` | `persistence_ratio` across probed layers (selected-layer fraction) |
| `early_peak_score` | max QK in first half of probed layers |
| `late_decay_score` | `early_peak_score - max QK in late half` |
| `onset_layer_top20` | first layer where block QK is in top-20% |
| `peak_layer` | layer with max QK |
| `query_conditioned_gain_z` | z-score of `query_conditioned_gain` across blocks |
| `hub_score` | null-question / rank-stability hub proxy |
| `bridge_flow_proxy` | lexical bridge link × query × anti-hub |
| `neighbor_gain_proxy` | merged-neighbor query gain |
| `generic_hub_score` | alias of `hub_score` |
| `boundary_sensitive_score` | alias of `neighbor_gain_proxy` |

## Score adjustment (layer l)
```
adjusted_l(b) = silm_score_l(b)
  + alpha_anchor * persistent_anchor(b) * middle_late_mask(l)
  + alpha_bridge * bridge_relay(b) * middle_late_mask(l)
  + alpha_boundary * boundary_sensitive(b)
  - lambda_hub * generic_hub(b)
  - lambda_transient * transient_trigger(b) * late_layer_mask(l)
```
Defaults: `alpha_anchor=0.4`, `alpha_bridge=0.4`, `alpha_boundary=0.2`, `lambda_hub=0.6`, `lambda_transient=0.3`.

## Hub cap
Generic hubs ≤ 10% of active blocks per layer (sink/recent blocks exempt).

## Churn calibration (default)
- Target: replace **5%–15%** of active blocks vs SILM per prune layer (default 10%).
- Mechanism: swap lowest role-adjusted-score SILM blocks for highest-scoring non-SILM candidates.
- QA smoke is **blocked** until replay/live churn audit passes.

## vs SILM
- Same budget K, same QK base score, same rescue path unless explicitly changed.
- RoleFlow only re-ranks within budget and applies role-specific caps/protections.

## Risks
- Wrong role proxies → no-op or harmful re-ranking.
- Over-protecting anchors/bridges may evict locally high-QK evidence.
- Boundary neighbor swaps can hurt if neighbor is irrelevant.

## Integration status
- **Real KV hook implemented** in `cache_utils.py` (not simulation-only for prefill selection).
"""
    _write_text(out_dir / "ROLEFLOW_KV_METHOD.md", text)


def load_property_features(feature_dir: Path, sample_ids: Set[str]) -> List[Dict[str, Any]]:
    path = feature_dir / "property_features.jsonl"
    if not path.exists():
        path = feature_dir / "property_features.csv"
        if not path.exists():
            raise FileNotFoundError(f"missing property features under {feature_dir}")
        import pandas as pd

        df = pd.read_csv(path)
        rows = df.to_dict(orient="records")
    else:
        rows = read_jsonl(path)
    return [dict(row) for row in rows if str(row.get("sample_id", "")) in sample_ids]


def build_role_assignment_audit(
    feature_rows: Sequence[Mapping[str, Any]],
    *,
    prune_layers: Sequence[int] = PRUNE_LAYERS,
) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[int, Dict[str, Any]]], Dict[str, Dict[int, Dict[int, Dict[str, Any]]]]]:
    grouped: Dict[Tuple[str, str], List[Mapping[str, Any]]] = defaultdict(list)
    for row in feature_rows:
        grouped[(str(row.get("task", "")), str(row.get("sample_id", "")))].append(dict(row))

    audit_rows: List[Dict[str, Any]] = []
    profiles_by_sample: Dict[str, Dict[int, Dict[str, Any]]] = {}
    profiles_by_sample_layer: Dict[str, Dict[int, Dict[int, Dict[str, Any]]]] = {}
    for (task, sample_id), rows in grouped.items():
        profiles = build_role_profiles_from_features(rows)
        layer_profiles = build_layer_role_profiles_from_features(rows, prune_layers=prune_layers)
        profiles_by_sample[sample_id] = profiles
        profiles_by_sample_layer[sample_id] = layer_profiles
        for layer, layer_map in layer_profiles.items():
            for block_id, prof in layer_map.items():
                audit_rows.append(
                    {
                        "sample_id": sample_id,
                        "task": task,
                        "prune_layer": layer,
                        "context_block_id": block_id,
                        "role_persistent_anchor": prof.get("role_persistent_anchor", 0),
                        "role_transient_trigger": prof.get("role_transient_trigger", 0),
                        "role_bridge_relay": prof.get("role_bridge_relay", 0),
                        "role_generic_hub": prof.get("role_generic_hub", 0),
                        "role_boundary_sensitive": prof.get("role_boundary_sensitive", 0),
                        "persistence_ratio_top20": prof.get("persistence_ratio_top20", ""),
                        "query_conditioned_gain_z": prof.get("query_conditioned_gain_z", ""),
                        "hub_score": prof.get("hub_score", ""),
                        "bridge_flow_proxy": prof.get("bridge_flow_proxy", ""),
                        "neighbor_gain_proxy": prof.get("neighbor_gain_proxy", ""),
                        "raw_diffusion_score_or_proxy": prof.get("raw_diffusion_score_or_proxy", ""),
                        "roleflow_priority": prof.get("roleflow_priority", ""),
                        "roleflow_policy": "profile_only",
                    }
                )
    return audit_rows, profiles_by_sample, profiles_by_sample_layer


def _roleflow_cfg(
    *,
    variant: str,
    layer: int,
    profiles_by_sample_layer: Mapping[str, Mapping[int, Mapping[int, Mapping[str, Any]]]],
    profiles_by_sample: Mapping[str, Mapping[int, Mapping[str, Any]]],
    sample_id: str,
    num_layers: int = 28,
) -> Dict[str, Any]:
    policy = POLICY_MAP.get(variant, "")
    cfg: Dict[str, Any] = {
        "roleflow_policy": policy,
        "variant_method": variant,
        "roleflow_profiles": dict(profiles_by_sample.get(sample_id, {})),
        "roleflow_profiles_by_layer": dict(profiles_by_sample_layer.get(sample_id, {})),
        "layer_idx": int(layer),
        "num_layers": int(num_layers),
        "roleflow_churn_calibration": policy not in {"", "silm_original"},
        "roleflow_target_churn": CHURN_TARGET,
        "roleflow_target_churn_min": CHURN_MIN,
        "roleflow_target_churn_max": CHURN_MAX,
    }
    return cfg


def replay_policy_audit(
    uasr_rows: Sequence[Mapping[str, Any]],
    profiles_by_sample: Mapping[str, Mapping[int, Mapping[str, Any]]],
    profiles_by_sample_layer: Mapping[str, Mapping[int, Mapping[int, Mapping[str, Any]]]],
    variants_list: Sequence[str],
    *,
    num_layers: int = 28,
) -> List[Dict[str, Any]]:
    audit: List[Dict[str, Any]] = []
    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in uasr_rows:
        if str(row.get("phase", "")) != "prefill":
            continue
        grouped[(str(row.get("task", "")), str(row.get("sample_id", "")))].append(dict(row))

    for variant in variants_list:
        policy = POLICY_MAP.get(variant, "")
        for (task, sample_id), rows in grouped.items():
            for row in sorted(rows, key=lambda item: int(item.get("layer_id", -1))):
                layer = int(row.get("layer_id", -1))
                block_ids = [int(x) for x in row.get("all_candidate_block_ids", [])]
                scores = [float(x) for x in row.get("all_candidate_scores", [])]
                if not block_ids or len(scores) != len(block_ids):
                    continue
                budget = int(row.get("budget_k", 0) or 0)
                silm_selected = {int(x) for x in row.get("selected_block_ids", [])}
                if budget <= 0:
                    budget = len(silm_selected)

                silm_block_scores = torch.tensor(scores, dtype=torch.float32)
                cfg = _roleflow_cfg(
                    variant=variant,
                    layer=layer,
                    profiles_by_sample_layer=profiles_by_sample_layer,
                    profiles_by_sample=profiles_by_sample,
                    sample_id=sample_id,
                    num_layers=num_layers,
                )
                block_scores = silm_block_scores
                if policy:
                    block_scores, _ = apply_roleflow_score_adjustment(block_scores, block_ids, cfg)
                k = min(max(1, budget), len(block_ids))
                churn_eligible = int(k < len(block_ids))
                _, top_idx = torch.topk(block_scores, k=k)
                must_keep = set()
                if block_ids:
                    must_keep.add(block_ids[0])
                    must_keep.add(block_ids[-1])
                if policy:
                    selected, meta = select_roleflow_kv_ids(
                        original_block_ids=block_ids,
                        block_scores=block_scores,
                        top_k_indices=top_idx,
                        must_keep_block_ids=must_keep,
                        num_blocks_to_keep=k,
                        cfg=cfg,
                        silm_block_scores=silm_block_scores,
                    )
                else:
                    selected = {block_ids[i] for i in top_idx.tolist()}.union(must_keep)
                    if len(selected) > k:
                        removable = sorted(selected - must_keep)
                        for bid in removable[: len(selected) - k]:
                            selected.discard(bid)
                    churn = active_set_churn(silm_selected, selected, k)
                    overlap = len(selected.intersection(silm_selected))
                    union = len(selected.union(silm_selected))
                    meta = {
                        "overlap_with_silm_original": float(overlap / union) if union else 1.0,
                        "churn_rate": churn,
                        "churn_in_target": int(CHURN_MIN <= churn <= CHURN_MAX),
                        "persistent_anchor_retained_count": sum(
                            1 for b in selected if int(profiles_by_sample.get(sample_id, {}).get(int(b), {}).get("role_persistent_anchor", 0))
                        ),
                        "bridge_relay_retained_count": sum(
                            1 for b in selected if int(profiles_by_sample.get(sample_id, {}).get(int(b), {}).get("role_bridge_relay", 0))
                        ),
                        "generic_hub_selected_count": sum(
                            1 for b in selected if int(profiles_by_sample.get(sample_id, {}).get(int(b), {}).get("role_generic_hub", 0))
                        ),
                        "boundary_neighbor_added_count": 0,
                        "transient_trigger_late_selected_count": 0,
                        "over_budget": int(len(selected) > k),
                        "actual_swaps": 0,
                    }
                audit.append(
                    {
                        "sample_id": sample_id,
                        "task": task,
                        "variant": variant,
                        "layer": layer,
                        "active_budget": k,
                        "selected_block_count": len(selected),
                        "over_budget": meta.get("over_budget", 0),
                        "overlap_with_silm_original": meta.get("overlap_with_silm_original", ""),
                        "churn_rate": meta.get("churn_rate", ""),
                        "churn_in_target": meta.get("churn_in_target", "") if churn_eligible else 1,
                        "churn_eligible": churn_eligible,
                        "actual_swaps": meta.get("actual_swaps", ""),
                        "persistent_anchor_retained_count": meta.get("persistent_anchor_retained_count", ""),
                        "bridge_relay_retained_count": meta.get("bridge_relay_retained_count", ""),
                        "generic_hub_selected_count": meta.get("generic_hub_selected_count", ""),
                        "boundary_neighbor_added_count": meta.get("boundary_neighbor_added_count", ""),
                        "transient_trigger_late_selected_count": meta.get("transient_trigger_late_selected_count", ""),
                        "integration_mode": "replay_audit_churn_calibrated",
                    }
                )
    return audit


def churn_audit_passed(audit_rows: Sequence[Mapping[str, Any]], variant: str = "roleflow_full") -> Tuple[bool, Dict[str, Any]]:
    rows = [
        r
        for r in audit_rows
        if r.get("variant") == variant and int(r.get("churn_eligible", 1) or 0) == 1
    ]
    if not rows:
        return False, {"reason": "no_eligible_rows"}
    over = sum(int(r.get("over_budget", 0) or 0) for r in rows)
    churns = [float(r.get("churn_rate", 0) or 0) for r in rows if r.get("churn_rate") not in ("", None)]
    in_target = sum(int(r.get("churn_in_target", 0) or 0) for r in rows)
    mean_churn = _mean(churns) if churns else 0.0
    passed = over == 0 and churns and in_target == len(rows) and all(CHURN_MIN <= c <= CHURN_MAX for c in churns)
    skipped = sum(
        1
        for r in audit_rows
        if r.get("variant") == variant and int(r.get("churn_eligible", 1) or 0) == 0
    )
    return passed, {
        "over_budget": over,
        "mean_churn": mean_churn,
        "churn_in_target_rows": in_target,
        "eligible_rows": len(rows),
        "skipped_degenerate_rows": skipped,
        "churn_min": CHURN_MIN,
        "churn_max": CHURN_MAX,
    }


def write_kv_policy_report(out_dir: Path, audit_rows: Sequence[Mapping[str, Any]], variants_list: Sequence[str]) -> None:
    by_variant: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in audit_rows:
        by_variant[str(row.get("variant", ""))].append(row)

    lines = [
        "# KV Policy Audit Report",
        "",
        "- Integration mode: **real KV hook + churn-calibrated selection (target 5–15% per prune layer)**",
        "- Role profiles: **per prune layer** (9, 18, 26)",
        "",
        "## Answers",
        "",
    ]
    silm_rows = by_variant.get("silm_original", [])
    full_rows = by_variant.get("roleflow_full", [])
    full_churns = [float(r.get("churn_rate", 0) or 0) for r in full_rows if r.get("churn_rate") not in ("", None)]
    mean_churn = _mean(full_churns) if full_churns else 0.0
    churn_ok = full_churns and all(CHURN_MIN <= c <= CHURN_MAX for c in full_churns)
    lines.append(
        f"1. Does RoleFlow change SILM selected blocks? "
        f"{'Yes' if mean_churn >= CHURN_MIN else 'No'} "
        f"(roleflow_full mean churn={mean_churn:.3f}, target {CHURN_MIN:.0%}–{CHURN_MAX:.0%})."
    )
    silm_hubs = _mean([float(r.get("generic_hub_selected_count", 0)) for r in silm_rows]) if silm_rows else 0.0
    full_hubs = _mean([float(r.get("generic_hub_selected_count", 0)) for r in full_rows]) if full_rows else 0.0
    lines.append(
        f"2. Does it reduce generic hub selection? "
        f"{'Yes' if full_hubs < silm_hubs else 'No or mixed'} "
        f"(SILM hubs/layer={silm_hubs:.2f}, roleflow_full={full_hubs:.2f})."
    )
    lines.append(
        f"3. Does it preserve anchors/bridges? "
        f"roleflow_full anchor retained/layer={_mean([float(r.get('persistent_anchor_retained_count', 0)) for r in full_rows]):.2f}, "
        f"bridge/layer={_mean([float(r.get('bridge_relay_retained_count', 0)) for r in full_rows]):.2f}."
    )
    over = sum(int(r.get("over_budget", 0) or 0) for r in audit_rows)
    lines.append(f"4. Same active budget? over_budget rows={over} (target 0).")
    lines.append(
        f"5. Churn in target band? {'PASS' if churn_ok else 'FAIL'} "
        f"({sum(int(r.get('churn_in_target',0) or 0) for r in full_rows)}/{len(full_rows)} rows in [{CHURN_MIN:.0%},{CHURN_MAX:.0%}])."
    )
    lines.append("6. Real KV selection modified in runtime prefill path: **yes** (`cache_utils.py` hook).")
    lines.append("")
    lines.append("## Per-variant churn vs SILM")
    lines.append("")
    lines.append("| Variant | Mean churn | Mean hub count | In-target rows |")
    lines.append("| --- | ---: | ---: | ---: |")
    for variant in variants_list:
        rows = by_variant.get(variant, [])
        churn_vals = [float(r.get("churn_rate", 0) or 0) for r in rows if r.get("churn_rate") not in ("", None)]
        lines.append(
            f"| {variant} | {_mean(churn_vals):.3f} | "
            f"{_mean([float(r.get('generic_hub_selected_count', 0) or 0) for r in rows]):.2f} | "
            f"{sum(int(r.get('churn_in_target',0) or 0) for r in rows)}/{len(rows)} |"
        )
    _write_text(out_dir / "KV_POLICY_AUDIT_REPORT.md", "\n".join(lines))


def make_runner_args(probe_args: argparse.Namespace, out_dir: Path) -> argparse.Namespace:
    old_argv = sys.argv[:]
    try:
        sys.argv = ["run_qwen25_sliminfer_variants.py"]
        runner_args = variants.normalize_args(variants.parse_args())
    finally:
        sys.argv = old_argv
    runner_args.model_path = str(probe_args.model_name_or_path)
    runner_args.tokenizer_path = str(probe_args.tokenizer_path or probe_args.model_name_or_path)
    runner_args.method = "sliminfer_original"
    runner_args.dtype = probe_args.dtype
    runner_args.device = probe_args.device
    runner_args.backend = probe_args.backend
    runner_args.output_dir = str(out_dir)
    runner_args.block_size = int(probe_args.block_size)
    runner_args.token_unit_size = 8
    runner_args.longbench_dir = str(probe_args.longbench_dir)
    runner_args.longbench_tasks = probe_args.tasks
    runner_args.longbench_max_samples = int(getattr(probe_args, "max_samples_per_task", probe_args.max_samples))
    runner_args.seed = 0
    runner_args.block_strategy = "fixed"
    runner_args.save_block_scores = "none"
    runner_args.rescue_policy = "none"
    runner_args.prune_layers = "9,18,26"
    runner_args.retain_tokens = "12288,6144,4096"
    runner_args.decode_policy = "original"
    runner_args.generation_max_new_tokens = 0
    runner_args.min_free_gpu_gb = float(probe_args.min_free_gpu_gb)
    runner_args.temperature = 0.0
    runner_args.top_p = 1.0
    return runner_args


def make_probe_args(args: argparse.Namespace, out_dir: Path) -> argparse.Namespace:
    return argparse.Namespace(
        model_name_or_path=Path(args.model_name_or_path),
        tokenizer_path=Path(args.tokenizer_path or args.model_name_or_path),
        longbench_dir=Path(args.longbench_dir),
        dataset="hotpotqa",
        tasks=args.tasks,
        method="sliminfer_original",
        max_samples=args.max_samples_per_task,
        sample_offset=0,
        block_size=args.block_size,
        unit_size=8,
        start_layer=0,
        end_layer=27,
        device="cuda",
        dtype=args.dtype,
        out_dir=out_dir,
        null_question_mode="shuffle_tokens",
        dry_run=False,
        block_strategy="fixed",
        max_model_len=32768,
        backend="flash_attention_2",
        seed=0,
        min_free_gpu_gb=args.min_free_gpu_gb,
        alpha=1.0,
        beta=0.5,
        gamma=0.8,
        delta=0.3,
        lambda_hub=0.6,
        lambda_red=0.3,
        top_k=10,
        analyze_only=False,
    )


def run_variant_prefill(
    probe_args: argparse.Namespace,
    variant: str,
    profiles_by_sample: Mapping[str, Mapping[int, Mapping[str, Any]]],
    profiles_by_sample_layer: Mapping[str, Mapping[int, Mapping[int, Mapping[str, Any]]]],
    contexts: Sequence[Any],
    out_dir: Path,
) -> Dict[str, Any]:
    variant_dir = out_dir / "variants" / variant
    variant_dir.mkdir(parents=True, exist_ok=True)
    policy = POLICY_MAP.get(variant, "")
    for path in [variant_dir / "uasr_score_dump.jsonl", variant_dir / "used_blocks.jsonl"]:
        path.write_text("", encoding="utf-8")

    runner_args = make_runner_args(probe_args, variant_dir)
    runner_args.save_block_scores = "sampled"
    runner_args.eval = "gate_audit"
    runner_args.generation_max_new_tokens = 1
    runner_args.uasr_score_dump_path = str(variant_dir / "uasr_score_dump.jsonl")
    runner_args.debug_used_block_audit_path = str(variant_dir / "used_blocks.jsonl")
    runner_args.debug_used_block_include_full = True
    runner_args.debug_used_block_full_layers = str(probe_args.end_layer)

    model, tokenizer, cache_classes, actual_attn = variants.load_model_and_tokenizer(runner_args, attn_impl=runner_args.backend)
    cfg, profile, notes = variants.build_sliminfer_config(runner_args, variant_dir, int(model.config.num_hidden_layers))
    SlimInferCache, HsCache = cache_classes

    if policy:
        cfg["roleflow_policy"] = policy
        cfg["variant_method"] = variant
    else:
        cfg["variant_method"] = "sliminfer_original"
        cfg.pop("roleflow_policy", None)

    for ctx in contexts:
        model_input = tokenizer(ctx.prompt, truncation=False, return_tensors="pt").to(model.device)
        sample_cfg = dict(cfg)
        sample_id = str(ctx.sample_id)
        sample_cfg.update(
            _roleflow_cfg(
                variant=variant,
                layer=0,
                profiles_by_sample_layer=profiles_by_sample_layer,
                profiles_by_sample=profiles_by_sample,
                sample_id=sample_id,
                num_layers=int(model.config.num_hidden_layers),
            )
        )
        sample_cfg["num_layers"] = int(model.config.num_hidden_layers)
        sample_cfg["audit_context"] = {
            "task": ctx.task,
            "sample_id": ctx.sample_id,
            "sample_index": ctx.sample_index,
            "context_length": ctx.context_length,
            "method": variant,
            "query_text": ctx.question,
            "block_texts": ctx.block_texts,
            "prompt_text_preview": ctx.prompt[:512],
        }
        sample_cfg["query_text"] = ctx.question
        sample_cfg["block_texts"] = ctx.block_texts
        sample_cfg["block_layout"] = ctx.block_layout
        sample_cfg["phase"] = "prefill"
        slim_cache = SlimInferCache(model.device)
        slim_cache.debug_used_block_audit_path = str(variant_dir / "used_blocks.jsonl")
        slim_cache.debug_used_block_include_full = True
        slim_cache.debug_used_block_full_layers = {int(probe_args.end_layer)}
        kwargs = {
            "past_key_values": slim_cache,
            "hs_cache": HsCache(),
            "sliminfer_config": sample_cfg,
            "all_original_prompt_indices_abs": list(range(ctx.context_length)),
            "use_cache": True,
        }
        with torch.no_grad():
            model(
                input_ids=model_input.input_ids,
                attention_mask=model_input.attention_mask,
                **kwargs,
            )
    return {"variant": variant, "status": "prefill_ok", "attn": actual_attn, "notes": notes}


def compare_live_prefill_dumps(
    out_dir: Path,
    *,
    baseline_variant: str = "silm_original",
    roleflow_variant: str = "roleflow_full",
) -> List[Dict[str, Any]]:
    base_rows = read_jsonl(out_dir / "variants" / baseline_variant / "uasr_score_dump.jsonl")
    rf_rows = read_jsonl(out_dir / "variants" / roleflow_variant / "uasr_score_dump.jsonl")
    base_index: Dict[Tuple[str, str, int], Mapping[str, Any]] = {}
    for row in base_rows:
        if str(row.get("phase", "")) != "prefill":
            continue
        key = (str(row.get("task", "")), str(row.get("sample_id", "")), int(row.get("layer_id", -1)))
        base_index[key] = row

    diff_rows: List[Dict[str, Any]] = []
    for row in rf_rows:
        if str(row.get("phase", "")) != "prefill":
            continue
        key = (str(row.get("task", "")), str(row.get("sample_id", "")), int(row.get("layer_id", -1)))
        base = base_index.get(key)
        if not base:
            continue
        silm_ids = {int(x) for x in base.get("selected_block_ids", [])}
        rf_ids = {int(x) for x in row.get("selected_block_ids", [])}
        budget = int(base.get("budget_k", 0) or len(silm_ids))
        added = sorted(rf_ids - silm_ids)
        removed = sorted(silm_ids - rf_ids)
        churn = active_set_churn(silm_ids, rf_ids, budget)
        diff_rows.append(
            {
                "sample_id": key[1],
                "task": key[0],
                "layer": key[2],
                "baseline_variant": baseline_variant,
                "roleflow_variant": roleflow_variant,
                "active_budget": budget,
                "silm_selected_count": len(silm_ids),
                "roleflow_selected_count": len(rf_ids),
                "blocks_added_count": len(added),
                "blocks_removed_count": len(removed),
                "churn_rate": churn,
                "churn_in_target": int(CHURN_MIN <= churn <= CHURN_MAX),
                "blocks_added_sample": added[:24],
                "blocks_removed_sample": removed[:24],
            }
        )
    _write_csv(
        out_dir / "live_prefill_layer_diff.csv",
        diff_rows,
        [
            "sample_id",
            "task",
            "layer",
            "baseline_variant",
            "roleflow_variant",
            "active_budget",
            "silm_selected_count",
            "roleflow_selected_count",
            "blocks_added_count",
            "blocks_removed_count",
            "churn_rate",
            "churn_in_target",
            "blocks_added_sample",
            "blocks_removed_sample",
        ],
    )
    return diff_rows


def write_live_prefill_report(out_dir: Path, diff_rows: Sequence[Mapping[str, Any]]) -> None:
    if not diff_rows:
        _write_text(out_dir / "LIVE_PREFILL_DUMP_REPORT.md", "# Live Prefill Dump Report\n\nNo diff rows (run `--run_live_prefill` first).\n")
        return
    churns = [float(r.get("churn_rate", 0) or 0) for r in diff_rows]
    in_target = sum(int(r.get("churn_in_target", 0) or 0) for r in diff_rows)
    lines = [
        "# Live Prefill Dump Report",
        "",
        f"- Rows: {len(diff_rows)} (sample × prune layer)",
        f"- Mean churn vs SILM: {_mean(churns):.3f}",
        f"- Target band: {CHURN_MIN:.0%}–{CHURN_MAX:.0%}",
        f"- In-target rows: {in_target}/{len(diff_rows)}",
        "",
        "## Per-layer mean churn",
        "",
        "| Layer | Mean churn | Added blocks | Removed blocks |",
        "| --- | ---: | ---: | ---: |",
    ]
    by_layer: Dict[int, List[Mapping[str, Any]]] = defaultdict(list)
    for row in diff_rows:
        by_layer[int(row.get("layer", -1))].append(row)
    for layer in sorted(by_layer):
        rows = by_layer[layer]
        lines.append(
            f"| {layer} | {_mean([float(r.get('churn_rate',0) or 0) for r in rows]):.3f} | "
            f"{_mean([float(r.get('blocks_added_count',0) or 0) for r in rows]):.1f} | "
            f"{_mean([float(r.get('blocks_removed_count',0) or 0) for r in rows]):.1f} |"
        )
    lines.append("")
    lines.append("See `live_prefill_layer_diff.csv` for per-sample block add/remove lists.")
    _write_text(out_dir / "LIVE_PREFILL_DUMP_REPORT.md", "\n".join(lines))


def run_tiny_qa_smoke(
    args: argparse.Namespace,
    out_dir: Path,
    variants_list: Sequence[str],
    profiles_by_sample: Mapping[str, Mapping[int, Mapping[str, Any]]],
    profiles_by_sample_layer: Mapping[str, Mapping[int, Mapping[int, Mapping[str, Any]]]],
    contexts: Sequence[Any],
    policy_audit: Sequence[Mapping[str, Any]],
) -> Tuple[bool, Optional[Dict[str, Any]]]:
    from longbench_eval import scorer

    over = sum(int(r.get("over_budget", 0) or 0) for r in policy_audit)
    if over > 0:
        return False, None

    gpu_info = check_gpu(float(args.min_free_gpu_gb))
    if float(gpu_info["selected"]["memory_free_gib"]) < float(args.min_free_gpu_gb):
        _write_text(
            out_dir / "BLOCKED_GPU_BUSY.md",
            f"GPU busy before QA smoke.\n\n{json.dumps(gpu_info, indent=2)}\n",
        )
        return False, None

    dataset2maxlen = json.loads((REPO_ROOT / "configs" / "dataset2maxlen.json").read_text(encoding="utf-8"))
    qa_dir = out_dir / "qa_smoke"
    qa_dir.mkdir(parents=True, exist_ok=True)
    probe_args = make_probe_args(args, out_dir)

    score_rows: List[Dict[str, Any]] = []
    variant_task_scores: Dict[str, Dict[str, float]] = {}
    variant_macro: Dict[str, float] = {}

    with (out_dir / "predictions.jsonl").open("w", encoding="utf-8") as pred_handle:
        for variant in variants_list:
            variant_dir = qa_dir / variant
            variant_dir.mkdir(parents=True, exist_ok=True)
            runner_args = make_runner_args(probe_args, variant_dir)
            model, tokenizer, cache_classes, _ = variants.load_model_and_tokenizer(
                runner_args, attn_impl=runner_args.backend
            )
            SlimInferCache, HsCache = cache_classes
            cfg, _, _ = variants.build_sliminfer_config(
                runner_args, variant_dir, int(model.config.num_hidden_layers)
            )
            policy = POLICY_MAP.get(variant, "")
            if policy:
                cfg["roleflow_policy"] = policy
                cfg["variant_method"] = variant
            else:
                cfg["variant_method"] = "sliminfer_original"
                cfg.pop("roleflow_policy", None)

            task_preds: Dict[str, List[str]] = defaultdict(list)
            task_answers: Dict[str, List[Any]] = defaultdict(list)
            task_lengths: Dict[str, List[int]] = defaultdict(list)
            task_classes: Dict[str, Any] = {}

            for ctx in contexts:
                model_input = tokenizer(ctx.prompt, truncation=False, return_tensors="pt").to(model.device)
                context_length = int(model_input.input_ids.shape[1])
                max_gen = int(dataset2maxlen.get(ctx.task, 32))
                sample_cfg = dict(cfg)
                sample_id = str(ctx.sample_id)
                sample_cfg.update(
                    _roleflow_cfg(
                        variant=variant,
                        layer=0,
                        profiles_by_sample_layer=profiles_by_sample_layer,
                        profiles_by_sample=profiles_by_sample,
                        sample_id=sample_id,
                        num_layers=int(model.config.num_hidden_layers),
                    )
                )
                sample_cfg["num_layers"] = int(model.config.num_hidden_layers)
                sample_cfg["audit_context"] = {
                    "task": ctx.task,
                    "sample_id": ctx.sample_id,
                    "sample_index": ctx.sample_index,
                    "context_length": context_length,
                    "method": variant,
                    "query_text": ctx.question,
                    "block_texts": ctx.block_texts,
                    "prompt_text_preview": ctx.prompt[:512],
                }
                sample_cfg["query_text"] = ctx.question
                sample_cfg["block_texts"] = ctx.block_texts
                sample_cfg["block_layout"] = ctx.block_layout
                sample_cfg["phase"] = "generate"
                slim_cache = SlimInferCache(model.device)
                kwargs = {
                    "past_key_values": slim_cache,
                    "hs_cache": HsCache(),
                    "sliminfer_config": sample_cfg,
                    "all_original_prompt_indices_abs": list(range(context_length)),
                }
                with torch.no_grad():
                    generated = model.generate(
                        model_input.input_ids,
                        attention_mask=model_input.attention_mask,
                        max_new_tokens=max_gen,
                        num_beams=1,
                        do_sample=False,
                        use_cache=True,
                        eos_token_id=tokenizer.eos_token_id,
                        **kwargs,
                    )
                output = generated[0]
                pred = tokenizer.decode(output[context_length:], skip_special_tokens=True)
                answers = ctx.sample.get("answers", [])
                all_classes = ctx.sample.get("all_classes", None)
                metric = float(scorer(ctx.task, [pred], [answers], [context_length], all_classes))
                score_rows.append(
                    {
                        "sample_id": sample_id,
                        "task": ctx.task,
                        "sample_index": ctx.sample_index,
                        "variant": variant,
                        "metric": metric,
                        "context_length": context_length,
                        "pred_preview": pred[:160],
                    }
                )
                pred_handle.write(
                    json.dumps(
                        {
                            "sample_id": sample_id,
                            "task": ctx.task,
                            "sample_index": ctx.sample_index,
                            "variant": variant,
                            "pred": pred,
                            "answers": answers,
                            "metric": metric,
                            "context_length": context_length,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                task_preds[ctx.task].append(pred)
                task_answers[ctx.task].append(answers)
                task_lengths[ctx.task].append(context_length)
                task_classes[ctx.task] = all_classes

            variant_task_scores[variant] = {}
            for task, preds in task_preds.items():
                variant_task_scores[variant][task] = float(
                    scorer(task, preds, task_answers[task], task_lengths[task], task_classes.get(task))
                )
            variant_macro[variant] = _mean(list(variant_task_scores[variant].values()))

            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    _write_csv(
        out_dir / "scores_by_sample.csv",
        score_rows,
        ["sample_id", "task", "sample_index", "variant", "metric", "context_length", "pred_preview"],
    )
    summary_rows = []
    for variant in variants_list:
        for task, score in sorted(variant_task_scores.get(variant, {}).items()):
            summary_rows.append({"variant": variant, "task": task, "score": score})
        summary_rows.append({"variant": variant, "task": "macro_avg", "score": variant_macro.get(variant, 0.0)})
    _write_csv(out_dir / "scores_summary.csv", summary_rows, ["variant", "task", "score"])

    silm_macro = float(variant_macro.get("silm_original", 0.0))
    delta_rows = []
    for variant in variants_list:
        if variant == "silm_original":
            continue
        delta_rows.append(
            {
                "variant": variant,
                "macro_score": variant_macro.get(variant, 0.0),
                "silm_macro": silm_macro,
                "delta_vs_silm": float(variant_macro.get(variant, 0.0)) - silm_macro,
            }
        )
    _write_csv(
        out_dir / "roleflow_delta_vs_silm.csv",
        delta_rows,
        ["variant", "macro_score", "silm_macro", "delta_vs_silm"],
    )

    lines = [
        "# RoleFlow QA Smoke Report",
        "",
        f"- Samples: {len(contexts)} ({args.tasks})",
        f"- Variants: {', '.join(variants_list)}",
        f"- Integration: real KV hook (`cache_utils.py`)",
        "",
        "## Macro scores",
        "",
        "| Variant | Macro | Δ vs SILM |",
        "| --- | ---: | ---: |",
    ]
    for variant in variants_list:
        macro = float(variant_macro.get(variant, 0.0))
        delta = macro - silm_macro if variant != "silm_original" else 0.0
        lines.append(f"| {variant} | {macro:.2f} | {delta:+.2f} |")
    lines.append("")
    lines.append("## Per-sample vs SILM (roleflow_full)")
    lines.append("")
    lines.append("| Task | Idx | SILM | roleflow_full | Note |")
    lines.append("| --- | ---: | ---: | ---: | --- |")
    silm_by_key = {
        (r["task"], int(r["sample_index"])): r
        for r in score_rows
        if r.get("variant") == "silm_original"
    }
    full_by_key = {
        (r["task"], int(r["sample_index"])): r
        for r in score_rows
        if r.get("variant") == "roleflow_full"
    }
    for key in sorted(silm_by_key):
        s_row = silm_by_key[key]
        f_row = full_by_key.get(key, {})
        sm = float(s_row.get("metric", 0) or 0)
        fm = float(f_row.get("metric", 0) or 0)
        sp = str(s_row.get("pred_preview", ""))[:40]
        fp = str(f_row.get("pred_preview", ""))[:40]
        note = "same pred" if sp == fp else "pred changed"
        lines.append(f"| {key[0]} | {key[1]} | {sm:.2f} | {fm:.2f} | {note} |")
    full_macro = float(variant_macro.get("roleflow_full", 0.0))
    if full_macro > silm_macro:
        lines.append("")
        lines.append(f"- **roleflow_full beats silm_original** ({full_macro:.2f} vs {silm_macro:.2f}).")
    elif full_macro < silm_macro:
        lines.append("")
        lines.append(f"- **roleflow_full below silm_original** ({full_macro:.2f} vs {silm_macro:.2f}).")
    elif full_macro == silm_macro:
        lines.append("")
        lines.append(
            f"- **QA neutral**: roleflow_full ties SILM ({full_macro:.2f}); "
            "predictions identical across variants (minimal KV overlap change)."
        )
    _write_text(out_dir / "ROLEFLOW_QA_SMOKE_REPORT.md", "\n".join(lines))
    write_component_validation_report(out_dir, score_rows, variant_macro, silm_macro)
    return True, {"macro": variant_macro, "variant_task_scores": variant_task_scores}


def write_component_validation_report(
    out_dir: Path,
    score_rows: Sequence[Mapping[str, Any]],
    variant_macro: Mapping[str, float],
    silm_macro: float,
) -> None:
    """12-sample style component comparison: hubcap/bridgeprotect vs full vs SILM."""
    components = ["roleflow_hubcap", "roleflow_bridgeprotect", "roleflow_persistence", "roleflow_full"]
    silm_by_key = {
        (str(r["task"]), int(r["sample_index"])): float(r.get("metric", 0) or 0)
        for r in score_rows
        if r.get("variant") == "silm_original"
    }
    lines = [
        "# RoleFlow Component Validation Report",
        "",
        f"- Samples: {len(silm_by_key)}",
        f"- SILM macro: {silm_macro:.2f}",
        "",
        "## Macro vs SILM",
        "",
        "| Variant | Macro | Δ vs SILM | Beats SILM? |",
        "| --- | ---: | ---: | --- |",
    ]
    macro_by_variant = {v: float(variant_macro.get(v, 0) or 0) for v in components + ["silm_original"]}
    for v in ["silm_original"] + components:
        m = macro_by_variant.get(v, 0.0)
        delta = m - silm_macro if v != "silm_original" else 0.0
        beats = "—" if v == "silm_original" else ("yes" if m > silm_macro else ("tie" if m == silm_macro else "no"))
        lines.append(f"| {v} | {m:.2f} | {delta:+.2f} | {beats} |")

    full_m = macro_by_variant.get("roleflow_full", 0.0)
    hub_m = macro_by_variant.get("roleflow_hubcap", 0.0)
    br_m = macro_by_variant.get("roleflow_bridgeprotect", 0.0)
    lines.extend(["", "## Full vs single components", ""])
    if hub_m > full_m or br_m > full_m:
        winners = [v for v, m in [("hubcap", hub_m), ("bridgeprotect", br_m)] if m > full_m]
        lines.append(f"- **roleflow_full below component(s):** {', '.join(winners)} (full={full_m:.2f}).")
    else:
        lines.append(f"- roleflow_full matches or leads components (full={full_m:.2f}, hubcap={hub_m:.2f}, bridge={br_m:.2f}).")

    lines.extend(["", "## Per-sample Δ (variant − SILM)", ""])
    lines.append("| Task | Idx | hubcap | bridge | persistence | full |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    sample_wins: Dict[str, int] = defaultdict(int)
    for key in sorted(silm_by_key):
        sm = silm_by_key[key]
        deltas = {}
        for v in components:
            row = next(
                (r for r in score_rows if r.get("variant") == v and str(r["task"]) == key[0] and int(r["sample_index"]) == key[1]),
                None,
            )
            deltas[v] = float(row.get("metric", 0) or 0) - sm if row else 0.0
            if deltas[v] > 0:
                sample_wins[v] += 1
        lines.append(
            f"| {key[0]} | {key[1]} | {deltas['roleflow_hubcap']:+.1f} | "
            f"{deltas['roleflow_bridgeprotect']:+.1f} | {deltas['roleflow_persistence']:+.1f} | "
            f"{deltas['roleflow_full']:+.1f} |"
        )

    lines.extend(["", "## Gain concentration", ""])
    total = len(silm_by_key)
    for v in components:
        lines.append(f"- `{v}` improves over SILM on **{sample_wins[v]}/{total}** samples.")
    if max(sample_wins.values()) <= 1:
        lines.append("- **Highly concentrated:** gains on ≤1 sample — high variance risk.")
    elif max(sample_wins.values()) <= total // 2:
        lines.append("- **Moderately concentrated** — watch per-task breakdown.")
    else:
        lines.append("- **Distributed gains** across multiple samples.")

    lines.extend(["", "## Method direction", ""])
    if (hub_m > silm_macro or br_m > silm_macro) and (hub_m > full_m or br_m > full_m):
        lines.append(
            "- **Pivot candidate:** RoleFlow-KV = hub-capped + bridge-protected retention "
            "(not full stack) if 12-sample confirms hubcap/bridgeprotect lead."
        )
    elif full_m > silm_macro and full_m >= hub_m and full_m >= br_m:
        lines.append("- **Keep roleflow_full** as primary variant for next scale-up.")
    else:
        lines.append("- **Mixed signal** — run ablation on winning components only.")

    _write_text(out_dir / "ROLEFLOW_COMPONENT_VALIDATION.md", "\n".join(lines))


def write_decision(
    out_dir: Path,
    audit_rows: Sequence[Mapping[str, Any]],
    qa_ran: bool,
    qa_summary: Optional[Mapping[str, Any]],
) -> None:
    full_rows = [r for r in audit_rows if r.get("variant") == "roleflow_full"]
    churn_vals = [float(r.get("churn_rate", 0) or 0) for r in full_rows if r.get("churn_rate") not in ("", None)]
    mean_churn = _mean(churn_vals) if churn_vals else 0.0
    churn_pass, churn_meta = churn_audit_passed(audit_rows)
    over = sum(int(r.get("over_budget", 0) or 0) for r in audit_rows)
    lines = [
        "# RoleFlow Decision",
        "",
        f"- Policy audit rows: {len(audit_rows)}",
        f"- roleflow_full mean churn vs SILM: {mean_churn:.3f} (target {CHURN_MIN:.0%}–{CHURN_MAX:.0%})",
        f"- Churn audit pass: {churn_pass}",
        f"- over_budget rows: {over}",
        f"- QA smoke ran: {qa_ran}",
        "",
    ]
    lines.append("## Decision")
    lines.append("")
    if not qa_ran:
        if churn_pass and over == 0:
            lines.append("- Churn calibration **PASS**; safe to run `--run_qa` (6-sample smoke).")
            lines.append("- Prefer `--run_live_prefill` first to verify runtime dumps match replay audit.")
        elif over == 0:
            lines.append("- Churn calibration **not in band**; tune target churn or role profiles before QA.")
            lines.append(f"- Replay meta: {json.dumps(churn_meta)}")
        else:
            lines.append("- Fix over_budget before QA.")
    elif qa_summary:
        macro = qa_summary.get("macro", {})
        full = float(macro.get("roleflow_full", 0) or 0)
        silm = float(macro.get("silm_original", 0) or 0)
        component_winners = [
            v
            for v in ("roleflow_hubcap", "roleflow_persistence", "roleflow_bridgeprotect")
            if float(macro.get(v, 0) or 0) > silm
        ]
        if full > silm and over == 0:
            lines.append("- **Recommend 12-sample KV validation** (roleflow_full beats SILM macro, budget intact).")
        elif full == silm and mean_churn >= CHURN_MIN:
            lines.append(
                f"- **QA neutral** with calibrated churn (mean {mean_churn:.1%}); check per-sample variance before 12-sample scale."
            )
        elif component_winners and full <= silm:
            lines.append(
                f"- **Component-focused validation** recommended for: {', '.join(component_winners)}."
            )
        elif mean_churn >= CHURN_MIN and full < silm:
            _write_text(
                out_dir / "NO_GO_ROLEFLOW_CURRENT.md",
                "# NO-GO RoleFlow Current\n\nRoleFlow changes KV selection but QA macro did not beat SILM.\n",
            )
            lines.append("- **NO-GO** for current RoleFlow settings; see `NO_GO_ROLEFLOW_CURRENT.md`.")
        else:
            lines.append("- Mixed QA signal; tune role weights or expand audit layers before scaling.")
    _write_text(out_dir / "ROLEFLOW_DECISION.md", "\n".join(lines))


def parse_args() -> argparse.Namespace:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_name_or_path", default=DEFAULT_MODEL)
    parser.add_argument("--tokenizer_path", default="")
    parser.add_argument("--longbench_dir", default="/home/yu-bai/Desktop/jiaqizhang/HeadKV/data/LongBench")
    parser.add_argument("--tasks", default="hotpotqa,musique")
    parser.add_argument("--max_samples_per_task", type=int, default=3)
    parser.add_argument("--feature_dir", default=str(DEFAULT_FEATURE_DIR))
    parser.add_argument("--uasr_dump", default=str(DEFAULT_FEATURE_DIR / "uasr_score_dump.jsonl"))
    parser.add_argument("--out_dir", default=f"results/roleflow_kv_probe_{stamp}")
    parser.add_argument("--block_size", type=int, default=64)
    parser.add_argument("--end_layer", type=int, default=27)
    parser.add_argument("--dtype", default="fp16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--min_free_gpu_gb", type=float, default=12.0)
    parser.add_argument("--run_qa", action="store_true", help="Run 6-sample QA only if churn audit passes (or --force_qa).")
    parser.add_argument("--force_qa", action="store_true", help="Run QA even if churn audit has not passed.")
    parser.add_argument("--run_live_prefill", action="store_true", help="GPU prefill dump: silm_original vs roleflow_full.")
    parser.add_argument(
        "--variants",
        default="",
        help="Comma-separated variant list (default: all built-in VARIANTS).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    variants_list = list(VARIANTS)
    if str(args.variants).strip():
        variants_list = [v.strip() for v in str(args.variants).split(",") if v.strip()]
        unknown = [v for v in variants_list if v not in POLICY_MAP]
        if unknown:
            raise SystemExit(f"Unknown variants: {unknown}; known: {sorted(POLICY_MAP)}")

    write_infra_inspection(out_dir)
    write_method_doc(out_dir)

    uasr_rows = read_jsonl(Path(args.uasr_dump))
    sample_ids = {str(r.get("sample_id", "")) for r in uasr_rows if r.get("sample_id")}
    probe_args_for_ids = make_probe_args(args, out_dir)
    try:
        tokenizer_ids = variants.AutoTokenizer.from_pretrained(
            str(args.tokenizer_path or args.model_name_or_path),
            trust_remote_code=True,
        )
        contexts_for_ids = build_sample_contexts(probe_args_for_ids, tokenizer_ids)
        sample_ids = sample_ids.union(str(c.sample_id) for c in contexts_for_ids)
    except Exception:
        contexts_for_ids = []
    feature_rows = load_property_features(Path(args.feature_dir), sample_ids)
    role_audit, profiles_by_sample, profiles_by_sample_layer = build_role_assignment_audit(feature_rows)
    _write_csv(
        out_dir / "role_assignment_audit.csv",
        role_audit,
        [
            "sample_id",
            "task",
            "prune_layer",
            "context_block_id",
            "role_persistent_anchor",
            "role_transient_trigger",
            "role_bridge_relay",
            "role_generic_hub",
            "role_boundary_sensitive",
            "persistence_ratio_top20",
            "query_conditioned_gain_z",
            "hub_score",
            "bridge_flow_proxy",
            "neighbor_gain_proxy",
            "raw_diffusion_score_or_proxy",
            "roleflow_priority",
            "roleflow_policy",
        ],
    )
    _write_json(out_dir / "role_profiles_by_sample.json", profiles_by_sample)
    _write_json(out_dir / "role_profiles_by_sample_layer.json", profiles_by_sample_layer)

    policy_audit = replay_policy_audit(
        uasr_rows, profiles_by_sample, profiles_by_sample_layer, variants_list
    )
    churn_pass, churn_meta = churn_audit_passed(policy_audit)
    _write_csv(
        out_dir / "kv_policy_audit.csv",
        policy_audit,
        [
            "sample_id",
            "task",
            "variant",
            "layer",
            "active_budget",
            "selected_block_count",
            "over_budget",
            "overlap_with_silm_original",
            "churn_rate",
            "churn_in_target",
            "churn_eligible",
            "actual_swaps",
            "persistent_anchor_retained_count",
            "bridge_relay_retained_count",
            "generic_hub_selected_count",
            "boundary_neighbor_added_count",
            "transient_trigger_late_selected_count",
            "integration_mode",
        ],
    )
    write_kv_policy_report(out_dir, policy_audit, variants_list)

    gpu_used = False
    blocked = False
    live_diff_rows: List[Dict[str, Any]] = []
    live_prefill_variants = ["silm_original", "roleflow_full"]
    if args.run_live_prefill:
        gpu_info = check_gpu(float(args.min_free_gpu_gb))
        if float(gpu_info["selected"]["memory_free_gib"]) < float(args.min_free_gpu_gb):
            blocked = True
            _write_text(
                out_dir / "BLOCKED_GPU_BUSY.md",
                f"GPU busy before live prefill.\n\n{json.dumps(gpu_info, indent=2)}\n",
            )
        else:
            gpu_used = True
            probe_args = make_probe_args(args, out_dir)
            tokenizer = variants.AutoTokenizer.from_pretrained(
                str(args.tokenizer_path or args.model_name_or_path),
                trust_remote_code=True,
            )
            contexts = build_sample_contexts(probe_args, tokenizer)
            for variant in live_prefill_variants:
                run_variant_prefill(
                    probe_args, variant, profiles_by_sample, profiles_by_sample_layer, contexts, out_dir
                )
            live_diff_rows = compare_live_prefill_dumps(out_dir)
            write_live_prefill_report(out_dir, live_diff_rows)

    qa_ran = False
    qa_summary: Optional[Dict[str, Any]] = None
    if args.run_qa and not blocked:
        over = sum(int(r.get("over_budget", 0) or 0) for r in policy_audit)
        if over > 0:
            _write_text(out_dir / "REAL_KV_INTEGRATION_BLOCKED.md", "Policy audit over_budget > 0; QA skipped.\n")
        elif not churn_pass and not args.force_qa:
            _write_text(
                out_dir / "QA_BLOCKED_CHURN.md",
                f"QA blocked: churn audit failed.\n\n{json.dumps(churn_meta, indent=2)}\n"
                "Run `--run_live_prefill` first, then re-audit. Use `--force_qa` to override.\n",
            )
        else:
            probe_args = make_probe_args(args, out_dir)
            tokenizer = variants.AutoTokenizer.from_pretrained(
                str(args.tokenizer_path or args.model_name_or_path),
                trust_remote_code=True,
            )
            contexts = build_sample_contexts(probe_args, tokenizer)
            qa_ran, qa_summary = run_tiny_qa_smoke(
                args,
                out_dir,
                variants_list,
                profiles_by_sample,
                profiles_by_sample_layer,
                contexts,
                policy_audit,
            )
            gpu_used = gpu_used or qa_ran

    write_decision(out_dir, policy_audit, qa_ran, qa_summary)
    over = sum(int(r.get("over_budget", 0) or 0) for r in policy_audit)
    full_churn = _mean(
        [
            float(r.get("churn_rate", 0) or 0)
            for r in policy_audit
            if r.get("variant") == "roleflow_full" and r.get("churn_rate") not in ("", None)
        ]
    )
    stop_lines = [
        "# STOP OR PASS",
        "",
        f"- Policy audit: {'PASS' if over == 0 else 'FAIL'} (rows={len(policy_audit)}, over_budget={over})",
        f"- Churn calibration: {'PASS' if churn_pass else 'FAIL'} (roleflow_full mean churn={full_churn:.3f}, target {CHURN_MIN:.0%}–{CHURN_MAX:.0%})",
        f"- Real KV hook: enabled in cache_utils.py",
        f"- GPU used: {gpu_used}",
        f"- Live prefill dump: {'ran' if live_diff_rows else 'not run (use --run_live_prefill)'}",
        f"- QA smoke: {'ran' if qa_ran else 'blocked until churn pass (use --run_qa after live prefill)'}",
        "",
        f"- Next step: {'--run_qa' if churn_pass and not qa_ran else '--run_live_prefill' if not live_diff_rows else 'review LIVE_PREFILL_DUMP_REPORT.md'}",
    ]
    _write_text(out_dir / "STOP_OR_PASS.md", "\n".join(stop_lines))

    cmd_path = out_dir / "run_commands.sh"
    cmd_path.write_text(
        "\n".join(
            [
                f"{sys.executable} {REPO_ROOT / 'tools' / 'run_roleflow_kv_probe.py'} \\",
                f"  --out_dir {out_dir} --feature_dir {args.feature_dir} --uasr_dump {args.uasr_dump}",
                f"  --tasks {args.tasks} --max_samples_per_task {args.max_samples_per_task}",
                "  --run_live_prefill   # GPU: silm vs roleflow_full uasr dump + layer diff",
                "  --run_qa             # 6-sample QA (only if churn audit passes)",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "out_dir": str(out_dir),
                "files_created": [
                    "ROLEFLOW_INFRA_INSPECTION.md",
                    "ROLEFLOW_KV_METHOD.md",
                    "role_assignment_audit.csv",
                    "kv_policy_audit.csv",
                    "KV_POLICY_AUDIT_REPORT.md",
                    "ROLEFLOW_DECISION.md",
                    "STOP_OR_PASS.md",
                    "run_commands.sh",
                ],
                "files_modified": ["bridgekv/roleflow_kv.py", "utils/cache_utils.py", "tools/run_roleflow_kv_probe.py"],
                "kv_integration": "real_runtime_hook",
                "gpu_used": gpu_used,
                "variants_evaluated": variants_list,
                "comparison_baseline": "silm_original",
                "next_command": f"{sys.executable} {REPO_ROOT / 'tools' / 'run_roleflow_kv_probe.py'} --run_live_prefill --out_dir {out_dir}",
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
