#!/usr/bin/env python3
"""CPU-only attestation for Unified LongBench Protocol v1.0 on the Qwen host."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

_REPO_BOOTSTRAP = Path(__file__).resolve().parents[1]
if str(_REPO_BOOTSTRAP) not in sys.path:
    sys.path.insert(0, str(_REPO_BOOTSTRAP))

from tools.qwen_pipeline_common import (
    COUNTS,
    DATA_ROOT,
    PROTOCOL_DIR,
    PROTOCOL_ID,
    QWEN_MODEL_ROOT,
    REPO_ROOT,
    TASKS,
    atomic_write_json,
    atomic_write_text,
    canonical_bytes,
    canonical_sample_identity,
    load_json,
    load_task_samples,
    now_iso,
    protocol_manifest_id,
    run_command,
    sha256_bytes,
    sha256_file,
)


SEMANTIC_CODE_FILES = {
    "configs/dataset2prompt.json",
    "configs/dataset2maxlen.json",
    "longbench_eval.py",
    "tools/consolidate_official_repro_results.py",
    "models/modeling_qwen2.py",
    "bridgekv/rescuekv_scoring.py",
}
EQUIVALENCE_GATED_IMPLEMENTATION_FILES = {
    "experiments/run_qwen25_sliminfer_variants.py",
    "utils/cache_utils.py",
}


def verify_protocol_checksums(protocol_dir: Path) -> Dict[str, Any]:
    checksum_file = protocol_dir / "SHA256SUMS.txt"
    result = run_command(["sha256sum", "-c", checksum_file.name], cwd=protocol_dir)
    failed = [line for line in result["output"].splitlines() if not line.endswith(": OK")]
    return {
        "pass": result["returncode"] == 0 and not failed,
        "returncode": result["returncode"],
        "checked": sum(1 for line in result["output"].splitlines() if line.endswith(": OK")),
        "failed_lines": failed,
    }


def verify_dataset(protocol_dir: Path, data_root: Path) -> Dict[str, Any]:
    frozen = load_json(protocol_dir / "dataset_hashes.json")
    expected_by_task = {entry["task"]: entry for entry in frozen["task_entries"]}
    entries: List[Dict[str, Any]] = []
    snapshot_core: List[Dict[str, Any]] = []
    for task in TASKS:
        path = data_root / f"{task}.jsonl"
        rows = load_task_samples(task, data_root)
        ids: List[str] = []
        canonical_rows: List[bytes] = []
        for index, row in enumerate(rows):
            sample_id, _ = canonical_sample_identity(task, index, row)
            ids.append(sample_id)
            canonical_rows.append(canonical_bytes({
                "task": task,
                "context": row.get("context"),
                "input": row.get("input"),
                "question": row.get("question"),
                "answers": row.get("answers"),
                "all_classes": row.get("all_classes"),
                "language": row.get("language"),
            }))
        current = {
            "task": task,
            "row_count": len(rows),
            "file_sha256": sha256_file(path),
            "official_slice_canonical_sha256": sha256_bytes(b"\n".join(canonical_rows)),
            "sample_id_sequence_sha256": sha256_bytes("\n".join(ids).encode("utf-8")),
        }
        expected = expected_by_task[task]
        mismatches = {
            field: {"expected": expected[field], "actual": current[field]}
            for field in ("row_count", "file_sha256", "official_slice_canonical_sha256", "sample_id_sequence_sha256")
            if current[field] != expected[field]
        }
        current["pass"] = not mismatches
        current["mismatches"] = mismatches
        entries.append(current)
        snapshot_core.append({
            "task": task,
            "row_count": current["row_count"],
            "official_slice_canonical_sha256": current["official_slice_canonical_sha256"],
            "sample_id_sequence_sha256": current["sample_id_sequence_sha256"],
        })
    snapshot_hash = sha256_bytes(canonical_bytes(snapshot_core))
    return {
        "pass": all(entry["pass"] for entry in entries) and snapshot_hash == frozen["dataset_snapshot_sha256"],
        "task_count": len(entries),
        "total_samples": sum(entry["row_count"] for entry in entries),
        "dataset_snapshot_sha256": snapshot_hash,
        "expected_dataset_snapshot_sha256": frozen["dataset_snapshot_sha256"],
        "tasks": entries,
    }


def verify_model_and_prompt(protocol_dir: Path) -> Dict[str, Any]:
    prompt = load_json(protocol_dir / "prompt_hashes.json")
    model = load_json(protocol_dir / "model_configs.json")["qwen25_7b_instruct"]
    checks: List[Dict[str, Any]] = []
    file_checks = {
        "prompt_template_file": (
            REPO_ROOT / "configs/dataset2prompt.json",
            prompt["prompt_template_file_sha256"],
        ),
        "tokenizer_config": (
            QWEN_MODEL_ROOT / "tokenizer_config.json",
            model["tokenizer_config_sha256"],
        ),
        "tokenizer_json": (
            QWEN_MODEL_ROOT / "tokenizer.json",
            model["tokenizer_json_sha256"],
        ),
    }
    for label, (path, expected) in file_checks.items():
        actual = sha256_file(path) if path.exists() else None
        checks.append({
            "name": label,
            "path": str(path),
            "expected": expected,
            "actual": actual,
            "pass": actual == expected,
        })
    tokenizer_config = load_json(QWEN_MODEL_ROOT / "tokenizer_config.json")
    chat_template_hash = sha256_bytes(str(tokenizer_config.get("chat_template", "")).encode("utf-8"))
    checks.append({
        "name": "chat_template",
        "expected": model["chat_template_sha256"],
        "actual": chat_template_hash,
        "pass": chat_template_hash == model["chat_template_sha256"],
    })
    return {
        "pass": all(check["pass"] for check in checks),
        "model_id": model["model_id"],
        "model_revision": model["revision"],
        "model_root": str(QWEN_MODEL_ROOT),
        "checks": checks,
        "prompt_truncation_count": sum(prompt["qwen25"]["runtime_truncation_count_by_task"].values()),
        "exact_final_prompt_hash_manifest_present": len(prompt["qwen25"]["final_prompt_sequence_sha256_by_task"]) == 17,
    }


def verify_scorer(protocol_dir: Path) -> Dict[str, Any]:
    frozen = load_json(protocol_dir / "scorer_hashes.json")
    actual = sha256_file(REPO_ROOT / frozen["scorer_source"])
    helper = sha256_file(REPO_ROOT / frozen["per_sample_helper_source"])
    return {
        "pass": actual == frozen["scorer_sha256"] and helper == frozen["per_sample_helper_sha256"],
        "scorer_expected": frozen["scorer_sha256"],
        "scorer_actual": actual,
        "helper_expected": frozen["per_sample_helper_sha256"],
        "helper_actual": helper,
        "macro_policy": frozen["macro_policy"],
    }


def verify_code_identity(protocol_dir: Path) -> Dict[str, Any]:
    frozen = load_json(protocol_dir / "code_hashes.json")
    rows: List[Dict[str, Any]] = []
    semantic_failures: List[str] = []
    implementation_changes: List[str] = []
    other_failures: List[str] = []
    for relative, expected in frozen["files"].items():
        path = REPO_ROOT / relative
        actual = sha256_file(path) if path.exists() else None
        matches = actual == expected["sha256"]
        classification = "MATCH"
        if not matches:
            if relative in SEMANTIC_CODE_FILES:
                classification = "PROTOCOL_SEMANTIC_MISMATCH"
                semantic_failures.append(relative)
            elif relative in EQUIVALENCE_GATED_IMPLEMENTATION_FILES:
                classification = "IMPLEMENTATION_REVISION_REQUIRES_CANARY"
                implementation_changes.append(relative)
            elif relative == "tools/freeze_unified_longbench_v1.py":
                classification = "FREEZE_TOOL_MISMATCH"
                other_failures.append(relative)
            else:
                classification = "UNCLASSIFIED_MISMATCH"
                other_failures.append(relative)
        rows.append({
            "path": relative,
            "expected": expected["sha256"],
            "actual": actual,
            "matches": matches,
            "classification": classification,
        })
    return {
        "protocol_semantic_identity_pass": not semantic_failures and not other_failures,
        "implementation_equivalence_pending": bool(implementation_changes),
        "semantic_failures": semantic_failures,
        "implementation_changes": implementation_changes,
        "other_failures": other_failures,
        "files": rows,
    }


def verify_history(sha256_before: Path) -> Dict[str, Any]:
    if not sha256_before.exists():
        return {"pass": False, "error": f"Missing {sha256_before}"}
    result = run_command(["sha256sum", "-c", str(sha256_before)])
    lines = result["output"].splitlines()
    failed = [line for line in lines if line and not line.endswith(": OK")]
    return {
        "pass": result["returncode"] == 0 and not failed,
        "checked_files": sum(1 for line in lines if line.endswith(": OK")),
        "failed_lines": failed,
    }


def attest(
    output_dir: Path,
    audit_dir: Path,
    *,
    verify_historical_files: bool = True,
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    checksums = verify_protocol_checksums(PROTOCOL_DIR)
    computed_id, manifest_sha = protocol_manifest_id(PROTOCOL_DIR)
    identity = {
        "expected_protocol_id": PROTOCOL_ID,
        "file_protocol_id": (PROTOCOL_DIR / "PROTOCOL_ID.txt").read_text(encoding="utf-8").strip(),
        "computed_protocol_id": computed_id,
        "protocol_manifest_sha256": manifest_sha,
    }
    identity["pass"] = len({identity["expected_protocol_id"], identity["file_protocol_id"], identity["computed_protocol_id"]}) == 1
    dataset = verify_dataset(PROTOCOL_DIR, DATA_ROOT)
    model_prompt = verify_model_and_prompt(PROTOCOL_DIR)
    scorer = verify_scorer(PROTOCOL_DIR)
    code = verify_code_identity(PROTOCOL_DIR)
    history = (
        verify_history(audit_dir / "SHA256_BEFORE.txt")
        if verify_historical_files
        else {"pass": None, "skipped": True}
    )
    semantic_pass = all((
        checksums["pass"], identity["pass"], dataset["pass"], model_prompt["pass"],
        scorer["pass"], code["protocol_semantic_identity_pass"],
        history["pass"] is not False,
    ))
    report = {
        "created_at": now_iso(),
        "protocol_id": PROTOCOL_ID,
        "decision": "PASS" if semantic_pass else "FAIL",
        "protocol_semantic_identity": "PASS" if semantic_pass else "FAIL",
        "implementation_equivalence": (
            "PENDING_STAGE_01_CANARY" if code["implementation_equivalence_pending"] else "FROZEN_MATCH"
        ),
        "checks": {
            "protocol_checksums": checksums,
            "canonical_manifest": identity,
            "dataset": dataset,
            "model_prompt": model_prompt,
            "scorer": scorer,
            "code_identity": code,
            "historical_artifacts": history,
        },
    }
    atomic_write_json(output_dir / "PROTOCOL_ATTESTATION.json", report)
    lines = [
        "# Qwen Protocol Attestation",
        "",
        f"- Protocol ID: `{PROTOCOL_ID}`",
        f"- Decision: **{report['decision']}**",
        f"- Semantic identity: `{report['protocol_semantic_identity']}`",
        f"- Implementation equivalence: `{report['implementation_equivalence']}`",
        f"- Dataset: `{dataset['total_samples']}` samples, hash `{dataset['dataset_snapshot_sha256']}`",
        f"- Model revision: `{model_prompt['model_revision']}`",
        f"- Historical SHA256 checked: `{history.get('checked_files', 'skipped')}`",
        "",
        "LLaMA artifacts are intentionally outside this Qwen-machine attestation and are not a blocker.",
    ]
    atomic_write_text(output_dir / "QWEN_PROTOCOL_ATTESTATION.md", "\n".join(lines) + "\n")
    if not semantic_pass:
        atomic_write_text(
            output_dir / "PROTOCOL_SEMANTIC_CHANGE_BLOCKER.md",
            "# Protocol Semantic Change Blocker\n\nAt least one frozen data, prompt, model/tokenizer, scorer, macro, or semantic-code identity check failed. The pipeline is stopped and must not reuse `ULB-v1-052cfbbeaa2cc42f`. See `PROTOCOL_ATTESTATION.json`.\n",
        )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--audit-dir",
        type=Path,
        default=REPO_ROOT / "results/unified_protocol_v1_audit_20260720_202245",
    )
    parser.add_argument("--skip-history-hash-check", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = attest(
        args.output_dir,
        args.audit_dir,
        verify_historical_files=not args.skip_history_hash_check,
    )
    print(json.dumps({
        "decision": report["decision"],
        "protocol_semantic_identity": report["protocol_semantic_identity"],
        "implementation_equivalence": report["implementation_equivalence"],
    }, indent=2))
    return 0 if report["decision"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
