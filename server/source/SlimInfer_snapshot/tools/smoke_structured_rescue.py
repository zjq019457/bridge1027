#!/usr/bin/env python3
"""Smoke tests for evidence-aware structured rescue methods."""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bridgekv.rescuekv_scoring import apply_rescue_policy, drm_s0_score
from bridgekv.structured_rescue_scoring import (
    enrich_structured_audit_fields,
    score_bridge_graph_srm,
    score_deferred_uncertainty_srm,
    score_distilled_blockdrm,
    score_rh_pyramid_srm,
)
from tools.build_span_teacher_microblock_labels import build_labels
from tools.train_distilled_blockdrm_selector import train_selector


def _synthetic_setup() -> Dict[str, Any]:
    torch.manual_seed(0)
    num_blocks = 20
    block_size = 64
    active_ids = list(range(8))
    inactive_ids = list(range(8, num_blocks))
    all_ids = list(range(num_blocks))
    device = torch.device("cpu")
    base = torch.randn(num_blocks, device=device)
    qk = torch.softmax(torch.randn(num_blocks, device=device), dim=0)
    reps = torch.randn(num_blocks, 16, device=device)
    return {
        "num_blocks": num_blocks,
        "block_size": block_size,
        "active_ids": active_ids,
        "inactive_ids": inactive_ids,
        "all_ids": all_ids,
        "base": base,
        "qk": qk,
        "reps": reps,
        "device": device,
    }


def _check_add_only(active_ids: List[int], final_ids: List[int], errors: List[str], tag: str) -> None:
    active_set = set(active_ids)
    final_set = set(final_ids)
    evicted = active_set - final_set
    if evicted:
        errors.append(f"{tag}: active_evicted_count={len(evicted)}")
    if not active_set.issubset(final_set):
        errors.append(f"{tag}: add_only_contract_ok=False")


def _run_method_smoke(
    score_name: str,
    score_fn,
    cfg: Dict[str, Any],
    setup: Dict[str, Any],
    errors: List[str],
    audit_rows: List[Dict[str, Any]],
) -> None:
    diag: Dict[str, List[str]] = {"warnings": []}
    scores, meta = score_fn(
        cfg, setup["base"], setup["reps"], setup["qk"],
        setup["active_ids"], setup["inactive_ids"], diag,
    )
    inactive_idx = [i for i in setup["inactive_ids"] if 0 <= i < scores.numel()]
    if inactive_idx:
        sub = scores[torch.tensor(inactive_idx, dtype=torch.long)]
        if not torch.isfinite(sub).all():
            errors.append(f"{score_name}: non-finite inactive scores")
    result = apply_rescue_policy(
        policy="add",
        active_ids=setup["active_ids"],
        inactive_ids=setup["inactive_ids"],
        rescue_scores=scores,
        active_scores=setup["base"],
        block_reps=setup["reps"],
        rescue_ratio=0.1,
        mmr_selection=bool(meta.get("mmr_selection", False)),
        diagnostics=diag,
    )
    final_ids = [int(x) for x in result.final_ids.tolist()]
    rescued_ids = [int(x) for x in result.rescued_ids.tolist()]
    _check_add_only(setup["active_ids"], final_ids, errors, score_name)
    if len(final_ids) != len(set(final_ids)):
        errors.append(f"{score_name}: duplicate selected units")
    record: Dict[str, Any] = {"method_name": score_name}
    enrich_structured_audit_fields(
        record, {**cfg, "rescue_score": score_name},
        meta, setup["active_ids"], rescued_ids, final_ids,
    )
    if not record.get("add_only_contract_ok"):
        errors.append(f"{score_name}: audit add_only_contract_ok false")
    audit_rows.append(record)


def main() -> int:
    errors: List[str] = []
    audit_rows: List[Dict[str, Any]] = []
    setup = _synthetic_setup()
    out_dir = Path(tempfile.mkdtemp(prefix="structured_rescue_smoke_"))
    audit_path = out_dir / "structured_rescue_smoke_audit.jsonl"

    # drm_add unchanged baseline
    diag: Dict[str, List[str]] = {"warnings": []}
    drm_scores, _ = drm_s0_score(
        setup["base"], setup["reps"], setup["active_ids"], setup["inactive_ids"],
        diagnostics=diag,
    )
    drm_result = apply_rescue_policy(
        policy="add", active_ids=setup["active_ids"], inactive_ids=setup["inactive_ids"],
        rescue_scores=drm_scores, active_scores=setup["base"], block_reps=setup["reps"],
        rescue_ratio=0.1, mmr_selection=True, diagnostics=diag,
    )
    _check_add_only(setup["active_ids"], [int(x) for x in drm_result.final_ids.tolist()], errors, "fixed_drm_add")

    base_cfg = {
        "layer_idx": 18,
        "num_layers": 28,
        "original_prompt_len": setup["num_blocks"] * setup["block_size"],
        "query_text": "Who founded Microsoft in 1975?",
        "block_texts": [f"Block {i} text about entity Alpha{i%3}" for i in range(setup["num_blocks"])],
        "rescue_granularity": "microblock",
        "microblock_size": 32,
    }

    _run_method_smoke("rh_pyramid_srm", score_rh_pyramid_srm, base_cfg, setup, errors, audit_rows)
    _run_method_smoke("bridge_graph_srm", score_bridge_graph_srm, {**base_cfg, "bridge_weight": 0.30}, setup, errors, audit_rows)

    label_path = out_dir / "teacher_labels.jsonl"
    build_labels(None, label_path, microblock_size=32, weak_fallback=True)
    selector_path = out_dir / "selector.json"
    train_selector(label_path, Path(""), selector_path)
    distilled_cfg = {**base_cfg, "teacher_label_file": str(label_path), "distilled_selector_file": str(selector_path)}
    _run_method_smoke("distilled_blockdrm", score_distilled_blockdrm, distilled_cfg, setup, errors, audit_rows)

    deferred_cfg = {**base_cfg, "deferred_mode": "audit", "deferred_topup_m": 2}
    _run_method_smoke("deferred_uncertainty_srm", score_deferred_uncertainty_srm, deferred_cfg, setup, errors, audit_rows)

    try:
        score_deferred_uncertainty_srm(
            {**base_cfg, "deferred_mode": "dynamic"},
            setup["base"], setup["reps"], setup["qk"],
            setup["active_ids"], setup["inactive_ids"], {"warnings": []},
        )
        errors.append("deferred dynamic mode should raise")
    except RuntimeError:
        pass

    with audit_path.open("w", encoding="utf-8") as f:
        for row in audit_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"audit_file={audit_path}")
    print(f"methods_tested={len(audit_rows)+1}")
    if errors:
        print("FAIL")
        for e in errors:
            print(f"  - {e}")
        return 1
    print("PASS: all structured rescue smoke checks ok")
    _print_eval_commands()
    return 0


def _print_eval_commands() -> None:
    from experiments.run_qwen25_sliminfer_variants import STRUCTURED_RESCUE_METHOD_SPECS, build_structured_rescue_cmd
    base = "results/structured_rescue"
    lb = "/diskE/datasets/LongBench"
    model = "/diskE/models/Qwen2.5-7B-Instruct"
    print("\n# Evaluation commands (not run by smoke):")
    for n in [32, 64, 128]:
        print(f"\n## {n}-sample per task")
        for name in STRUCTURED_RESCUE_METHOD_SPECS:
            print(build_structured_rescue_cmd(
                name, output_dir=f"{base}_{n}", longbench_dir=lb,
                model_path=model, samples_per_task=n,
            ))
    print(f"\n## Full LongBench")
    for name in STRUCTURED_RESCUE_METHOD_SPECS:
        print(build_structured_rescue_cmd(
            name, output_dir=f"{base}_full", longbench_dir=lb, model_path=model,
        ))


if __name__ == "__main__":
    raise SystemExit(main())
