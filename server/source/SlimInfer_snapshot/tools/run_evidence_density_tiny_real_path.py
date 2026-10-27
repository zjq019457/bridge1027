#!/usr/bin/env python3
"""Run and audit a 2-sample real-path Evidence-density validation."""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Set, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

PYTHON = REPO_ROOT / ".venv" / "bin" / "python"
RUNNER = REPO_ROOT / "experiments" / "run_qwen25_sliminfer_variants.py"
TASKS = ["hotpotqa", "musique"]
METHODS = [
    "fixed_drm_add",
    "evidence_density_srm",
    "rh_pyramid_srm",
    "bridge_graph_srm",
]
STRUCTURED = set(METHODS) - {"fixed_drm_add"}
DEFAULT_MODEL = (
    "/home/yu-bai/.cache/huggingface/hub/"
    "models--Qwen--Qwen2.5-7B-Instruct/snapshots/"
    "a09a35458c702b33eeacc393d103063234e8bc28"
)
DEFAULT_LB = "/home/yu-bai/Desktop/jiaqizhang/HeadKV/data/LongBench"


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _jaccard(left: Set[int], right: Set[int]) -> float:
    if not left and not right:
        return 1.0
    return len(left & right) / max(1, len(left | right))


def _method_spec(method: str) -> Dict[str, Any]:
    if method == "fixed_drm_add":
        return {"rescue_score": "drm", "structured": False}
    if method == "rh_pyramid_srm":
        return {
            "rescue_score": method,
            "structured": True,
            "rh_weight": 0.50,
            "pyramid_weight": 0.20,
        }
    if method == "bridge_graph_srm":
        return {
            "rescue_score": method,
            "structured": True,
            "rh_weight": 0.50,
            "bridge_weight": 0.45,
        }
    return {
        "rescue_score": "evidence_density_srm",
        "structured": True,
        "density_alpha_max": 1.0,
        "density_alpha_count": 0.6,
        "density_alpha_continuity": 0.8,
        "density_alpha_coverage": 0.5,
        "density_lambda_red": 0.3,
        "density_threshold_mode": "mean_std",
        "density_top_quantile": 0.30,
    }


def _command(root: Path, method: str, args: argparse.Namespace) -> List[str]:
    spec = _method_spec(method)
    command = [
        str(PYTHON),
        str(RUNNER),
        "--model_path", args.model_path,
        "--tokenizer_path", args.model_path,
        "--model_family", "qwen2",
        "--dtype", "fp16",
        "--device", "cuda",
        "--backend", "flash_attention_2",
        "--method", "sliminfer_variant",
        "--scoring", "delta_o",
        "--decode_policy", "reuse_prefill",
        "--eval", "longbench",
        "--longbench_dir", args.longbench_dir,
        "--longbench_tasks", ",".join(TASKS),
        "--max_samples_per_task", "2",
        "--generation_max_new_tokens", str(args.max_new_tokens),
        "--smoke_mode", "1",
        "--max_smoke_samples", "2",
        "--output_dir", str(root / "runs" / method),
        "--isolated_run_id", method,
        "--block_strategy", "fixed",
        "--block_size", "64",
        "--prune_layers", "9,18,26",
        "--retain_tokens", "12288,6144,4096",
        "--preserve_first_block",
        "--temperature", "0",
        "--top_p", "1.0",
        "--enable_cpu_offload",
        "--enable_async_kv",
        "--save_outputs",
        "--rescue_policy", "add",
        "--rescue_score", spec["rescue_score"],
        "--rescue_ratio", str(args.rescue_ratio),
        "--rescue_audit_path", str(root / "audit" / f"{method}.jsonl"),
        "--seed", "0",
        "--fd_alpha_o", "1.0",
        "--fd_beta_h", "0.0",
        "--fd_gamma_s", "0.0",
    ]
    if spec["structured"]:
        command.extend([
            "--rescue_granularity", "microblock",
            "--microblock_size", "32",
            "--strict_structured_path_check", "true",
            "--dump_candidate_scores", "1",
            "--dump_final_selection", "1",
        ])
    for key, value in spec.items():
        if key in {"rescue_score", "structured"}:
            continue
        command.extend([f"--{key}", str(value)])
    return command


def _layer26_rows(root: Path) -> Dict[str, Dict[str, Dict[str, Any]]]:
    rows: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)
    for method in METHODS:
        for row in _read_jsonl(root / "audit" / f"{method}.jsonl"):
            if int(row.get("layer_id", -1)) != 26:
                continue
            key = f"{row.get('task')}:{row.get('sample_id')}"
            rows[method][key] = row
    return rows


def _page_ids(row: Mapping[str, Any]) -> Set[int]:
    values = (
        row.get("selected_parent_page_ids")
        or row.get("rescued_fixed_page_ids")
        or row.get("physical_units_after_expansion")
        or row.get("rescued_ids")
        or row.get("rescued_virtual_ids_first50")
        or []
    )
    return {int(value) for value in values}


def _build_validation(root: Path) -> Dict[str, Any]:
    validation = root / "real_path_validation"
    validation.mkdir(parents=True, exist_ok=True)
    layer26 = _layer26_rows(root)

    marker_rows: List[Dict[str, Any]] = []
    for method in STRUCTURED:
        marker_path = root / "runs" / method / "path_marker.jsonl"
        for row in _read_jsonl(marker_path):
            if int(row.get("layer_id", -1)) == 26:
                marker_rows.append({**row, "source_file": str(marker_path)})
    _write_csv(
        validation / "01_path_marker_audit.csv",
        marker_rows,
        [
            "sample_id", "task", "layer_id", "method",
            "entered_structured_selector", "entered_real_fixed_block_path",
            "entered_cache_construction_path", "used_microblock_first_selector",
            "used_candidate_score_dump", "used_final_selection_dump",
            "used_evidence_density_selector", "source_file",
        ],
    )

    micro_rows = [
        {
            "sample_id": row.get("sample_id", ""),
            "task": row.get("task", ""),
            "layer_id": row.get("layer_id", ""),
            "method": method,
            "selected_microblock_count": row.get("selected_microblock_count", 0),
            "selected_parent_page_count": len(row.get("selected_parent_page_ids", [])),
            "final_cache_granularity": row.get("final_cache_granularity", ""),
            "expansion_ratio": row.get("expansion_ratio", ""),
        }
        for method in STRUCTURED
        for row in layer26.get(method, {}).values()
    ]
    _write_csv(
        validation / "02_microblock_count_audit.csv",
        micro_rows,
        [
            "sample_id", "task", "layer_id", "method",
            "selected_microblock_count", "selected_parent_page_count",
            "final_cache_granularity", "expansion_ratio",
        ],
    )

    overlap_rows: List[Dict[str, Any]] = []
    for other in ("fixed_drm_add", "rh_pyramid_srm", "bridge_graph_srm"):
        keys = sorted(set(layer26["evidence_density_srm"]) | set(layer26[other]))
        for key in keys:
            density_pages = _page_ids(layer26["evidence_density_srm"].get(key, {}))
            other_pages = _page_ids(layer26[other].get(key, {}))
            overlap_rows.append({
                "task_sample": key,
                "method_a": "evidence_density_srm",
                "method_b": other,
                "jaccard": round(_jaccard(density_pages, other_pages), 4),
                "density_page_ids": json.dumps(sorted(density_pages)),
                "other_page_ids": json.dumps(sorted(other_pages)),
            })
    _write_csv(
        validation / "03_selection_overlap.csv",
        overlap_rows,
        [
            "task_sample", "method_a", "method_b", "jaccard",
            "density_page_ids", "other_page_ids",
        ],
    )

    density_candidates = [
        row
        for row in _read_jsonl(
            root / "runs" / "evidence_density_srm" / "candidate_score_dump.jsonl"
        )
        if int(row.get("layer_id", -1)) == 26
    ]
    by_sample: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in density_candidates:
        by_sample[f"{row.get('task')}:{row.get('sample_id')}"].append(row)
    component_rows: List[Dict[str, Any]] = []
    for key, rows in sorted(by_sample.items()):
        by_page: Dict[int, Dict[str, Any]] = {}
        for row in rows:
            by_page.setdefault(int(row["page_id"]), row)
        density_top = max(
            by_page,
            key=lambda page_id: (
                float(by_page[page_id].get("density_parent_score", 0.0)),
                -page_id,
            ),
        )
        max_only_top = max(
            by_page,
            key=lambda page_id: (
                float(by_page[page_id].get("density_max_score", 0.0)),
                -page_id,
            ),
        )
        top = by_page[density_top]
        component_rows.append({
            "task_sample": key,
            "density_top_page": density_top,
            "max_only_top_page": max_only_top,
            "ranking_changed_vs_max_only": density_top != max_only_top,
            "density_parent_score": top.get("density_parent_score", ""),
            "density_max_score": top.get("density_max_score", ""),
            "density_high_count": top.get("density_high_count", ""),
            "density_continuity_bonus": top.get("density_continuity_bonus", ""),
            "density_span_coverage": top.get("density_span_coverage", ""),
            "density_threshold": top.get("density_threshold", ""),
            "density_threshold_mode": top.get("density_threshold_mode", ""),
            "density_count_contribution": top.get("density_count_contribution", ""),
            "density_continuity_contribution": top.get("density_continuity_contribution", ""),
            "density_coverage_contribution": top.get("density_coverage_contribution", ""),
            "redundancy_fallback_used": top.get("redundancy_fallback_used", ""),
        })
    _write_csv(
        validation / "04_density_component_audit.csv",
        component_rows,
        [
            "task_sample", "density_top_page", "max_only_top_page",
            "ranking_changed_vs_max_only", "density_parent_score",
            "density_max_score", "density_high_count",
            "density_continuity_bonus", "density_span_coverage",
            "density_threshold", "density_threshold_mode",
            "density_count_contribution", "density_continuity_contribution",
            "density_coverage_contribution", "redundancy_fallback_used",
        ],
    )

    score_rows: List[Dict[str, Any]] = []
    for method in METHODS:
        path = root / "runs" / method / "longbench_scores.csv"
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                score_rows.append({
                    "method": method,
                    "task": row.get("dataset", ""),
                    "score": row.get("score", ""),
                    "num_samples": row.get("num_samples", ""),
                    "status": row.get("status", ""),
                    "error": row.get("error", ""),
                })
    _write_csv(
        validation / "05_task_score_snapshot.csv",
        score_rows,
        ["method", "task", "score", "num_samples", "status", "error"],
    )

    density_markers = [
        row for row in marker_rows if row.get("method") == "evidence_density_srm"
    ]
    real_path_ok = bool(density_markers) and all(
        row.get("entered_real_fixed_block_path")
        and row.get("used_microblock_first_selector")
        and row.get("used_evidence_density_selector")
        for row in density_markers
    )
    density_micro_counts = [
        int(row.get("selected_microblock_count", 0) or 0)
        for row in micro_rows
        if row.get("method") == "evidence_density_srm"
    ]
    micro_ok = bool(density_micro_counts) and min(density_micro_counts) > 0
    dump_status = {
        filename: (
            root / "runs" / "evidence_density_srm" / filename
        ).exists()
        and (
            root / "runs" / "evidence_density_srm" / filename
        ).stat().st_size > 0
        for filename in (
            "candidate_score_dump.jsonl",
            "final_selection_dump.jsonl",
            "path_marker.jsonl",
        )
    }
    dumps_ok = all(dump_status.values())
    differs_fixed = any(
        float(row["jaccard"]) < 1.0
        for row in overlap_rows
        if row["method_b"] == "fixed_drm_add"
    )
    differs_structured = any(
        float(row["jaccard"]) < 1.0
        for row in overlap_rows
        if row["method_b"] in {"rh_pyramid_srm", "bridge_graph_srm"}
    )
    nonzero_density = any(
        abs(float(row.get("density_count_contribution") or 0.0)) > 1e-9
        or abs(float(row.get("density_continuity_contribution") or 0.0)) > 1e-9
        for row in component_rows
    )
    ranking_changed = any(
        bool(row.get("ranking_changed_vs_max_only"))
        for row in component_rows
    )
    justify = (
        real_path_ok
        and micro_ok
        and dumps_ok
        and differs_fixed
        and differs_structured
        and nonzero_density
    )
    recommendation = [
        "# Evidence-density tiny real-path recommendation",
        "",
        "This is a 2-sample-per-task mechanism and wiring validation only. It is not a quality claim.",
        "",
        "## Required Answers",
        "",
        f"1. Evidence-density entered the real fixed-block path: **{real_path_ok}**.",
        f"2. `selected_microblock_count > 0`: **{micro_ok}** "
        f"(min={min(density_micro_counts) if density_micro_counts else 0}, "
        f"mean={round(statistics.mean(density_micro_counts), 3) if density_micro_counts else 0}).",
        f"3. Candidate/final/path dumps generated: **{dumps_ok}** ({json.dumps(dump_status)}).",
        f"4. Parent-page selection differs from `fixed_drm_add` on a real sample: **{differs_fixed}**.",
        "5. Parent-page selection differs from RH or Bridge on a real sample: "
        f"**{differs_structured}**.",
        "6. Density components change ranking versus max-only: "
        f"**{ranking_changed}**; nonzero count/continuity contribution: **{nonzero_density}**.",
        f"7. A 32-sample Qwen rerun is justified: **{justify}**.",
        "",
        "## Decision",
        "",
        (
            "The mechanism gate passes. A separate 32-sample Qwen diagnostic is justified, but was not started."
            if justify
            else
            "The mechanism gate does not pass. Do not scale to 32 samples yet."
        ),
        "",
        "Do not interpret tiny task scores as evidence of quality success. "
        "Do not interpret this test as an algorithmic failure of RH/Bridge.",
    ]
    (validation / "06_recommendation.md").write_text(
        "\n".join(recommendation) + "\n",
        encoding="utf-8",
    )
    return {
        "real_path_ok": real_path_ok,
        "micro_ok": micro_ok,
        "dumps_ok": dumps_ok,
        "differs_fixed": differs_fixed,
        "differs_structured": differs_structured,
        "nonzero_density": nonzero_density,
        "ranking_changed": ranking_changed,
        "justify_32": justify,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="")
    parser.add_argument("--model_path", default=DEFAULT_MODEL)
    parser.add_argument("--longbench_dir", default=DEFAULT_LB)
    parser.add_argument("--max_new_tokens", type=int, default=16)
    parser.add_argument("--rescue_ratio", type=float, default=0.05)
    parser.add_argument("--skip_existing", action="store_true")
    args = parser.parse_args()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = (
        Path(args.output_dir)
        if args.output_dir
        else REPO_ROOT / "results" / f"evidence_density_tiny_{stamp}"
    )
    (root / "runs").mkdir(parents=True, exist_ok=True)
    (root / "audit").mkdir(exist_ok=True)

    manifest: Dict[str, Any] = {
        "output_root": str(root),
        "model_path": args.model_path,
        "tasks": TASKS,
        "sample_limit": 2,
        "methods": METHODS,
        "max_new_tokens": args.max_new_tokens,
        "rescue_ratio": args.rescue_ratio,
        "runs": [],
    }
    for method in METHODS:
        status_path = root / "runs" / method / "run_status.json"
        if args.skip_existing and status_path.exists():
            status = json.loads(status_path.read_text(encoding="utf-8"))
            manifest["runs"].append(status)
            continue
        print(f"===== RUN {method} =====", flush=True)
        started = time.time()
        log_path = root / "runs" / method / "run.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.run(
                _command(root, method, args),
                cwd=str(REPO_ROOT),
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        status = {
            "method": method,
            "exit_code": process.returncode,
            "elapsed_s": round(time.time() - started, 2),
            "log_path": str(log_path),
        }
        status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
        manifest["runs"].append(status)
        print(json.dumps(status), flush=True)
        if process.returncode != 0:
            raise RuntimeError(f"{method} failed; inspect {log_path}")

    validation = _build_validation(root)
    manifest["validation"] = validation
    manifest["status"] = "complete"
    (root / "config.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"OUT_ROOT={root}", flush=True)


if __name__ == "__main__":
    main()
