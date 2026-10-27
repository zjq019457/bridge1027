#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bridgekv.rescuekv_scoring import (
    apply_rescue_policy,
    block_obd_saliency,
    combined_abc_score,
    delta_o_block_score,
    diffusion_gap_score,
    normalize_scores,
    path_aware_score,
    redundancy_aware_score,
)


METHOD_ROWS = [
    ("SlimInfer original", "none", "delta_o", "K", "Baseline path unchanged."),
    ("random add", "add", "random", "K+M", "Add random rescued inactive blocks."),
    ("delta_o add", "add", "delta_o", "K+M", "Add inactive blocks by delta_o proxy."),
    ("obd add", "add", "obd", "K+M", "Add inactive blocks by OBD saliency."),
    ("random swap", "swap", "random", "K", "Budget-neutral random rescue."),
    ("delta_o swap", "swap", "delta_o", "K", "Budget-neutral delta_o rescue."),
    ("obd swap", "swap", "obd", "K", "Budget-neutral OBD rescue."),
    ("diffusion_gap swap", "swap", "diffusion_gap", "K", "Budget-neutral diffusion-gap rescue."),
    ("redundancy swap", "swap", "redundancy", "K", "Budget-neutral redundancy-aware rescue."),
    ("path swap", "swap", "path", "K", "Budget-neutral path-aware rescue."),
    ("abc swap", "swap", "abc", "K", "Budget-neutral ABC combined rescue."),
]


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _git_snapshot() -> Dict[str, str]:
    out: Dict[str, str] = {}
    for key, cmd in {
        "commit": ["git", "rev-parse", "--short", "HEAD"],
        "status": ["git", "status", "--short"],
    }.items():
        try:
            out[key] = subprocess.check_output(cmd, cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL).strip()
        except Exception as exc:
            out[key] = f"unavailable: {exc}"
    return out


def _finite(name: str, tensor: torch.Tensor, errors: List[str]) -> None:
    if tensor.numel() == 0:
        errors.append(f"{name}: empty tensor")
    if not bool(torch.isfinite(tensor).all()):
        errors.append(f"{name}: contains non-finite values")


def run_synthetic_tests() -> Dict[str, Any]:
    torch.manual_seed(0)
    diagnostics: Dict[str, List[str]] = {"warnings": []}
    errors: List[str] = []

    num_blocks = 24
    block_size = 4
    tokens = num_blocks * block_size
    dim = 12
    active_ids = list(range(12))
    inactive_ids = list(range(12, num_blocks))
    preserve_ids = [0, 1]
    recent_ids = [10, 11]
    sink_ids = [2]

    attn = torch.softmax(torch.randn(2, 3, 4, tokens), dim=-1)
    values = torch.randn(tokens, dim)
    block_ids = torch.arange(tokens) // block_size
    obd = block_obd_saliency(attn_probs=attn, values=values, block_ids=block_ids, block_size=block_size, diagnostics=diagnostics)
    _finite("obd", obd, errors)

    block_attn = torch.softmax(torch.randn(num_blocks), dim=-1)
    block_values = torch.randn(num_blocks, dim)
    obd_fallback = block_obd_saliency(block_attention=block_attn, block_values=block_values, num_blocks=num_blocks, diagnostics=diagnostics)
    _finite("obd_fallback", obd_fallback, errors)

    missing_diag: Dict[str, List[str]] = {"warnings": []}
    obd_missing = block_obd_saliency(num_blocks=num_blocks, diagnostics=missing_diag)
    _finite("obd_missing", obd_missing, errors)
    if not missing_diag.get("warnings"):
        errors.append("missing tensor fallback did not record a warning")

    delta_o = delta_o_block_score(existing_delta_o=torch.rand(num_blocks), diagnostics=diagnostics)
    _finite("delta_o", delta_o, errors)

    reps = torch.randn(num_blocks, dim)
    diffusion = diffusion_gap_score(delta_o, reps, active_ids, inactive_ids, diagnostics)
    redundancy = redundancy_aware_score(delta_o, reps, active_ids, inactive_ids, lambda_red=0.25, diagnostics=diagnostics)
    path = path_aware_score(attn_mass=torch.rand(num_blocks), active_ids=active_ids, inactive_ids=inactive_ids, num_blocks=num_blocks, diagnostics=diagnostics)
    abc = combined_abc_score(
        obd_scores=obd,
        block_reps=reps,
        active_ids=active_ids,
        inactive_ids=inactive_ids,
        path_scores=path,
        alpha_diff=0.5,
        lambda_red=0.25,
        beta_path=0.25,
        diagnostics=diagnostics,
    )
    for name, score in {
        "diffusion": diffusion,
        "redundancy": redundancy,
        "path": path,
        "abc": abc,
        "diffusion_missing_reps": diffusion_gap_score(delta_o, None, active_ids, inactive_ids, diagnostics),
        "redundancy_missing_reps": redundancy_aware_score(delta_o, None, active_ids, inactive_ids, diagnostics=diagnostics),
        "abc_missing_reps": combined_abc_score(
            obd_scores=obd,
            block_reps=None,
            active_ids=active_ids,
            inactive_ids=inactive_ids,
            path_scores=path,
            diagnostics=diagnostics,
        ),
        "normalize_zscore": normalize_scores(torch.tensor([1.0, float("nan"), float("inf")]), "zscore"),
        "normalize_minmax": normalize_scores(torch.tensor([3.0, 3.0, 3.0]), "minmax"),
    }.items():
        _finite(name, score, errors)

    random_scores = torch.rand(num_blocks)
    add = apply_rescue_policy(
        active_ids=active_ids,
        inactive_ids=inactive_ids,
        rescue_scores=random_scores,
        active_scores=delta_o,
        policy="add",
        rescue_ratio=0.25,
        preserve_ids=preserve_ids,
        recent_ids=recent_ids,
        sink_ids=sink_ids,
    )
    if int(add.final_ids.numel()) != len(active_ids) + 3:
        errors.append(f"additive policy expected K+M=15 ids, got {int(add.final_ids.numel())}")

    sliminfer_larger = apply_rescue_policy(
        active_ids=active_ids,
        inactive_ids=inactive_ids,
        rescue_scores=delta_o,
        active_scores=delta_o,
        policy="sliminfer_larger",
        rescue_ratio=0.25,
        preserve_ids=preserve_ids,
        recent_ids=recent_ids,
        sink_ids=sink_ids,
    )
    if int(sliminfer_larger.final_ids.numel()) != len(active_ids) + 3:
        errors.append(
            f"sliminfer_larger expected K+M=15 ids, got {int(sliminfer_larger.final_ids.numel())}"
        )

    swap = apply_rescue_policy(
        active_ids=active_ids,
        inactive_ids=inactive_ids,
        rescue_scores=abc,
        active_scores=delta_o,
        obd_scores=obd,
        policy="swap",
        rescue_ratio=0.25,
        swap_drop_score="sliminfer",
        preserve_ids=preserve_ids,
        recent_ids=recent_ids,
        sink_ids=sink_ids,
    )
    if int(swap.final_ids.numel()) != len(active_ids):
        errors.append(f"swap policy expected K=12 ids, got {int(swap.final_ids.numel())}")

    forced_rescue = torch.zeros(num_blocks)
    forced_rescue[12:15] = torch.tensor([10.0, 9.0, 8.0])
    forced_drop = torch.ones(num_blocks)
    forced_drop[3:6] = torch.tensor([0.1, 0.2, 0.3])
    conditional = apply_rescue_policy(
        active_ids=active_ids,
        inactive_ids=inactive_ids,
        rescue_scores=forced_rescue,
        active_scores=forced_drop,
        policy="conditional_swap",
        rescue_ratio=0.25,
        tau=0.0,
        preserve_ids=preserve_ids,
        recent_ids=recent_ids,
        sink_ids=sink_ids,
    )
    if int(conditional.rescued_ids.numel()) == 0:
        errors.append("conditional swap failed to accept obviously better rescue blocks")

    rejected = apply_rescue_policy(
        active_ids=active_ids,
        inactive_ids=inactive_ids,
        rescue_scores=forced_rescue,
        active_scores=forced_drop,
        policy="conditional_swap",
        rescue_ratio=0.25,
        tau=20.0,
        preserve_ids=preserve_ids,
        recent_ids=recent_ids,
        sink_ids=sink_ids,
    )
    if int(rejected.rescued_ids.numel()) != 0:
        errors.append("conditional swap accepted blocks despite a high tau threshold")

    for name, result in {
        "add": add,
        "sliminfer_larger": sliminfer_larger,
        "swap": swap,
        "conditional": conditional,
        "rejected": rejected,
    }.items():
        final = result.final_ids.detach().cpu().tolist()
        if len(final) != len(set(final)):
            errors.append(f"{name}: final block ids contain duplicates")
        if final != sorted(final):
            errors.append(f"{name}: final block ids are not sorted")
        dropped = set(result.dropped_ids.detach().cpu().tolist())
        protected = set(preserve_ids) | set(recent_ids) | set(sink_ids)
        if dropped & protected:
            errors.append(f"{name}: dropped protected ids {sorted(dropped & protected)}")

    return {
        "status": "pass" if not errors else "fail",
        "errors": errors,
        "warnings": diagnostics.get("warnings", []) + missing_diag.get("warnings", []),
        "score_shapes": {
            "obd": int(obd.numel()),
            "delta_o": int(delta_o.numel()),
            "diffusion": int(diffusion.numel()),
            "redundancy": int(redundancy.numel()),
            "path": int(path.numel()),
            "abc": int(abc.numel()),
        },
        "policy_counts": {
            "add_final": int(add.final_ids.numel()),
            "sliminfer_larger_final": int(sliminfer_larger.final_ids.numel()),
            "swap_final": int(swap.final_ids.numel()),
            "conditional_rescued": int(conditional.rescued_ids.numel()),
            "conditional_rejected": int(rejected.rescued_ids.numel()),
        },
    }


def write_tiny_eval_template(out_dir: Path) -> Path:
    path = out_dir / "run_tiny_eval.sh"
    script = """#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
OUT_ROOT="${OUT_ROOT:-results/rescuekv_tiny_eval_manual}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-7B-Instruct}"
COMMON_ARGS=(
  --model_path "$MODEL_PATH"
  --backend flash_attention_2
  --dtype fp16
  --device cuda
  --longbench_tasks qasper,hotpotqa,musique
  --max_samples_per_task 2
  --context_lengths 8192
  --decode_tokens 16
  --generation_max_new_tokens 16
  --smoke_mode 1
  --max_smoke_samples 2
)

python experiments/run_qwen25_sliminfer_variants.py "${COMMON_ARGS[@]}" \
  --method sliminfer_original --scoring token_unit_qk --decode_policy original \
  --output_dir "$OUT_ROOT/sliminfer_original"

python experiments/run_qwen25_sliminfer_variants.py "${COMMON_ARGS[@]}" \
  --method sliminfer_variant --scoring delta_o --decode_policy reuse_prefill \
  --rescue_policy add --rescue_score obd \
  --output_dir "$OUT_ROOT/obd_add"

python experiments/run_qwen25_sliminfer_variants.py "${COMMON_ARGS[@]}" \
  --method sliminfer_variant --scoring delta_o --decode_policy reuse_prefill \
  --rescue_policy swap --rescue_score abc \
  --output_dir "$OUT_ROOT/abc_swap"
"""
    path.write_text(script, encoding="utf-8")
    path.chmod(0o755)
    return path


def write_report(out_dir: Path, test_result: Dict[str, Any], tiny_template: Path, summary_rows: List[Dict[str, Any]]) -> None:
    warnings = test_result.get("warnings", [])
    warning_lines = "\n".join(f"- {item}" for item in warnings) if warnings else "- None"
    errors = test_result.get("errors", [])
    error_lines = "\n".join(f"- {item}" for item in errors) if errors else "- None"
    method_lines = "\n".join(
        f"- {row['method']}: policy={row['rescue_policy']}, score={row['rescue_score']}, budget={row['final_budget_type']}"
        for row in summary_rows
    )
    report = f"""# RescueKV Add/Swap/ABC Smoke Report

## Modified Files
- `bridgekv/rescuekv_scoring.py`
- `bridgekv/__init__.py`
- `utils/cache_utils.py`
- `utils/processor.py`
- `experiments/run_qwen25_sliminfer_variants.py`
- `tools/smoke_rescuekv_variants.py`

## Implemented Rescue Scores
- `delta_o`: wraps an existing Delta-O score when available, with a lightweight perturbation proxy fallback.
- `obd`: OBCache-style block saliency, `D_OBD(B)=||Delta_B||_2^2/(||O||_2^2+eps)`.
- `diffusion_gap`: `S_diff(B)=D_base(B)*(1-C(B))`, where `C(B)` is max cosine coverage by active blocks.
- `redundancy`: `S_red(B)=D_base(B)/(1+lambda_red*max(R(B),0))`.
- `path`: recent-query block attention mass plus a small bridge-position fallback bonus.
- `abc`: `z(D_OBD)+alpha_diff*z(D_OBD*(1-C))-lambda_red*z(R)+beta_path*z(P_path)`.
- `random`: deterministic smoke baseline using random scores in the test harness.

## Implemented Rescue Policies
- `additive_rescue`: `final=A union R`, budget type `K+M`.
- `budget_neutral_swap`: `final=(A-D) union R`, budget type `K`.
- `conditional_swap`: swaps only when `rescue_score(inactive) > drop_score(active)+tau`.

## Smoke Methods Covered
{method_lines}

## Synthetic Tests
- Status: `{test_result.get('status')}`
- Score shapes: `{json.dumps(test_result.get('score_shapes', {}), sort_keys=True)}`
- Policy counts: `{json.dumps(test_result.get('policy_counts', {}), sort_keys=True)}`

### Errors
{error_lines}

### Fallback Warnings
{warning_lines}

## Tiny Eval
- Status: skipped by default.
- Reason: this task requested no full experiment, and the new Add/Swap/ABC module is validated synthetically first.
- Template: `{tiny_template}`

## Fallback Approximations
- If token-level attention probabilities and V tensors are available, `block_obd_saliency` uses the exact block perturbation formula.
- If token-level tensors are absent but block-level attention/value representatives exist, OBD uses block-level representatives and records a warning.
- If path attention flow is absent, `path_aware_score` uses normalized recent-query block attention mass plus a deterministic bridge-position bonus and records a warning.
- If block representations are absent, diffusion/redundancy terms are skipped or reduced to safe zero/base-score behavior with warnings.

## Known Risks
- The new scoring/policy module is wired into the SlimInfer cache selection path behind `--rescue_policy`; real-model tiny eval should still precede paper-scale experiments.
- Fallback OBD/path scores are approximations and should be reported as such.
- Additive rescue intentionally increases the final budget to `K+M`, so it is not budget-neutral.
- Swap and conditional swap preserve protected/recent/sink ids in the policy layer, but production callers must pass the correct protected ids.

## Next Formal Experiments
- Tiny LongBench smoke with qasper/hotpotqa/musique and 1-2 samples per task.
- Budget-neutral comparison of `delta_o swap`, `obd swap`, `diffusion_gap swap`, `redundancy swap`, `path swap`, and `abc swap`.
- Separate fairness table for Add (`K+M`) versus Swap (`K`).
"""
    (out_dir / "report.md").write_text(report, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Synthetic smoke tests for RescueKV Add/Swap/ABC variants.")
    parser.add_argument("--out_root", type=str, default="")
    args = parser.parse_args()

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_root) if args.out_root else REPO_ROOT / "results" / f"rescuekv_smoke_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "timestamp": timestamp,
        "torch": torch.__version__,
        "full_experiment_run": False,
        "tiny_eval_run": False,
        "seed": 0,
        "git": _git_snapshot(),
        "defaults": {
            "rescue_ratio": 0.05,
            "alpha_diff": 0.5,
            "lambda_red": 0.25,
            "beta_path": 0.25,
            "swap_tau": 0.0,
        },
    }
    _write_json(out_dir / "config.json", config)

    test_result = run_synthetic_tests()
    synthetic_status = test_result["status"]
    tiny_status = "skipped_template_only"
    rows = [
        {
            "method": method,
            "rescue_policy": policy,
            "rescue_score": score,
            "final_budget_type": budget_type,
            "synthetic_test_status": synthetic_status,
            "tiny_eval_status": tiny_status,
            "notes": notes,
        }
        for method, policy, score, budget_type, notes in METHOD_ROWS
    ]
    _write_csv(
        out_dir / "summary.csv",
        rows,
        ["method", "rescue_policy", "rescue_score", "final_budget_type", "synthetic_test_status", "tiny_eval_status", "notes"],
    )
    _write_json(out_dir / "synthetic_test_details.json", test_result)
    tiny_template = write_tiny_eval_template(out_dir)
    write_report(out_dir, test_result, tiny_template, rows)
    print(out_dir)
    print(f"synthetic_status={synthetic_status}")
    return 0 if synthetic_status == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
