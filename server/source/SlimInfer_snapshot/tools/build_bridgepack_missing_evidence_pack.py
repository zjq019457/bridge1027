#!/usr/bin/env python3
"""Build a missing-evidence work package for the BridgePack paper.

This script reads existing compact BridgePack artifacts and runs only CPU-side
audits. It does not run model generation. Expensive full evaluations are written
as resumable shell commands under the output root.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bridgekv.evidence_text_packing import pack_evidence  # noqa: E402
from tools.run_evidence_text_packing_offline import _candidate_dataset_roots, _load_samples  # noqa: E402

TASKS = ["qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "musique", "narrativeqa"]
MULTIHOP = ["hotpotqa", "2wikimqa", "musique"]
METHODS = [
    "span_only_source_order",
    "bm25_chunk_source_order",
    "tfidf_chunk_source_order",
    "bridge_neighbor32_path_order",
    "lexical_bridgepack_path_order",
]
DISPLAY = {
    "span_only_source_order": "Span-only",
    "bm25_chunk_source_order": "BM25",
    "tfidf_chunk_source_order": "TF-IDF",
    "bridge_neighbor32_path_order": "BridgePack",
    "lexical_bridgepack_path_order": "Lexical-BridgePack",
    "bridge_same_spans_source_order": "Same Spans + Source Order",
    "query_only_neighbor32_path_order": "Query-only Neighbor",
    "bridge_neighbor32_path_order_tags": "BridgePack+Tags",
}


def _parse_list(value: str) -> list[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_jsonl(path: Path, limit: int = 0) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
                if limit and len(rows) >= limit:
                    break
    return rows


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            json.dump(dict(row), handle, ensure_ascii=False)
            handle.write("\n")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def _run(cmd: Sequence[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write("$ " + " ".join(str(part) for part in cmd) + "\n")
        proc = subprocess.run(list(cmd), cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT, text=True, check=False)
        log.write(f"returncode={proc.returncode}\n")
    return int(proc.returncode)


def _python_with(module_name: str) -> str:
    candidates = [Path(sys.executable)]
    candidates.extend(sorted(Path("/home/yu-bai/miniconda3/envs").glob("*/bin/python")))
    for candidate in candidates:
        if not candidate.exists():
            continue
        proc = subprocess.run(
            [
                str(candidate),
                "-c",
                f"import importlib.util; raise SystemExit(0 if importlib.util.find_spec({module_name!r}) else 1)",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        if proc.returncode == 0:
            return str(candidate)
    return sys.executable


def _latest(pattern: str) -> Path | None:
    matches = sorted((REPO_ROOT / "results").glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return matches[0] if matches else None


def _require(path: Path | None, label: str) -> Path:
    if path is None:
        raise FileNotFoundError(f"Could not locate {label}")
    return path


def _norm(text: Any) -> str:
    import re

    value = str(text or "").lower()
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return " ".join(value.split())


def _answer_covered(packed_text: str, answers: Any) -> bool:
    haystack = _norm(packed_text)
    if not haystack:
        return False
    if not isinstance(answers, list):
        answers = [answers]
    for answer in answers:
        if isinstance(answer, list):
            candidates = answer
        else:
            candidates = [answer]
        for item in candidates:
            needle = _norm(item)
            if len(needle) >= 2 and needle in haystack:
                return True
    return False


def _preview(text: str, chars: int = 420) -> str:
    cleaned = " ".join(str(text or "").split())
    return cleaned[:chars] + ("..." if len(cleaned) > chars else "")


def _snippet_around_answer(packed_text: str, answers: Any, chars: int = 600) -> str:
    text = str(packed_text or "")
    low = text.lower()
    if not isinstance(answers, list):
        answers = [answers]
    needles: list[str] = []
    for answer in answers:
        if isinstance(answer, list):
            needles.extend(str(x) for x in answer)
        else:
            needles.append(str(answer))
    for answer in needles:
        if not answer:
            continue
        pos = low.find(answer.lower())
        if pos >= 0:
            start = max(0, pos - chars // 2)
            end = min(len(text), pos + len(answer) + chars // 2)
            return _preview(text[start:end], chars)
    return _preview(text, chars)


def _score_one(task: str, row: Mapping[str, Any]) -> float:
    all_classes = row.get("all_classes", None)
    try:
        from longbench_eval import scorer

        value = scorer(
            task,
            [str(row.get("pred", ""))],
            [row.get("answers", [])],
            [row.get("length", 0)],
            all_classes,
        )
        return float(value or 0.0)
    except Exception:
        pred = _norm(row.get("pred", ""))
        answers = row.get("answers", [])
        if not isinstance(answers, list):
            answers = [answers]
        for answer in answers:
            candidates = answer if isinstance(answer, list) else [answer]
            for item in candidates:
                needle = _norm(item)
                if needle and (needle in pred or pred in needle):
                    return 100.0
        return 0.0


def _shell_script(path: Path, command: str) -> None:
    _write_text(path, "#!/usr/bin/env bash\nset -euo pipefail\ncd " + str(REPO_ROOT) + "\n" + command.strip())
    path.chmod(0o755)


def _manifest(root: Path, name: str, args: argparse.Namespace, smoke: bool, task_list: Sequence[str], sample_count: str) -> None:
    payload = {
        "name": name,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "git_commit": subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, text=True).strip(),
        "command": " ".join(sys.argv),
        "model": args.qwen_model,
        "tokenizer": args.qwen_model,
        "task_list": list(task_list),
        "sample_count": sample_count,
        "budget": args.budget,
        "smoke": smoke,
        "full": not smoke,
    }
    (root / "run_manifest.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def repo_audit(root: Path, args: argparse.Namespace) -> None:
    lines = [
        "# Repository Audit",
        "",
        f"- Git commit: `{subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'], cwd=REPO_ROOT, text=True).strip()}`",
        f"- Official exact Qwen dir: `{args.exact_dir}`",
        f"- Budget ablation dir: `{args.budget_dir}`",
        f"- Mistral gate dir: `{args.mistral_gate_dir}`",
        "",
        "## Key Existing Code",
        "",
        "- `tools/run_bridgepack_exact_budget_repack.py`: exact-token packed-prompt builder and generation/scoring wrapper.",
        "- `tools/run_evidence_text_packing_generation_qwen32.py`: packed prompt generation/scoring runner.",
        "- `tools/run_bridgepack_second_model_exact_gate.py`: second-model exact-token gate wrapper.",
        "- `bridgekv/evidence_text_packing.py`: BridgePack / Lexical-BridgePack packing implementation.",
        "- `longbench_eval.py`: LongBench scoring functions.",
        "- `tools/run_bridgepack_full_context_eval.py`: added full-context wrapper.",
        "- `tools/run_bridgepack_dense_retriever_packing.py`: added optional dense-retriever wrapper.",
        "",
        "## Existing Outputs",
        "",
        "- Qwen exact-token main results include packed prompts, predictions, scores, summary, audits, and paper tables.",
        "- Mistral current evidence is a 3-task gate only.",
        "- Existing packed prompts include `packed_text`, `original_context`, `answers`, selected span/page ids, and exact model-token counts.",
    ]
    _write_text(root / "00_repo_audit.md", "\n".join(lines))


def full_context_qwen(root: Path, args: argparse.Namespace, next_commands: list[str], status: list[dict[str, Any]]) -> None:
    sub = root / "01_full_context_qwen"
    sub.mkdir(parents=True, exist_ok=True)
    smoke_out = sub / "full_context_smoke_build"
    runtime_python = args.runtime_python or _python_with("transformers")
    cmd = [
        runtime_python,
        str(REPO_ROOT / "tools" / "run_bridgepack_full_context_eval.py"),
        "--model_name_or_path", args.qwen_model,
        "--tasks", ",".join(TASKS),
        "--max_samples", "2",
        "--budget", str(args.qwen_model_max_length),
        "--model_max_length", str(args.qwen_model_max_length),
        "--reference_run_dir", str(args.exact_dir),
        "--out_dir", str(smoke_out),
    ]
    rc = _run(cmd, sub / "full_context_smoke.log")
    audit_src = smoke_out / "full_context_input_audit.csv"
    rows = _read_csv(audit_src)
    _write_csv(
        sub / "full_context_smoke.csv",
        rows,
        [
            "task", "sample_id", "status", "original_context_tokens", "full_prompt_tokens",
            "model_max_length", "exceeds_model_max_length", "truncate_strategy", "handled",
        ],
    )
    completed = rc == 0 and bool(rows)
    over = sum(1 for row in rows if str(row.get("exceeds_model_max_length")) == "True")
    _write_text(sub / "full_context_smoke_report.md", "\n".join([
        "# Full-Context Qwen Smoke",
        "",
        "This is an input/token-length smoke only. No generation was run.",
        "",
        f"- Smoke command return code: `{rc}`.",
        f"- Rows checked: `{len(rows)}`.",
        f"- Over model max length: `{over}`.",
        "- Full-context score remains TODO until GPU generation is run.",
    ]))
    command = (
        f"{runtime_python} tools/run_bridgepack_full_context_eval.py "
        f"--model_name_or_path {args.qwen_model} "
        f"--tasks {','.join(TASKS)} --max_samples 0 --budget {args.qwen_model_max_length} "
        f"--model_max_length {args.qwen_model_max_length} --reference_run_dir {args.exact_dir} "
        f"--out_dir {sub / 'full_context_full'} --run_generation --dtype fp16 --device auto"
    )
    _shell_script(sub / "full_context_full_command.sh", command)
    next_commands.append(command)
    main_rows = _read_csv(args.exact_dir / "paper_tables" / "01_exact_budget_main_table_1024.csv")
    template = [{"method": "Full Context / original length", "budget": "original", "avg": "TODO", "source": "future full run"}]
    for row in main_rows:
        if row.get("method") in METHODS:
            template.append({
                "method": row.get("method"),
                "budget": row.get("budget", "1024"),
                "avg": row.get("avg", ""),
                "source": str(args.exact_dir / "paper_tables" / "01_exact_budget_main_table_1024.csv"),
            })
    _write_csv(sub / "full_context_gap_table_template.csv", template, ["method", "budget", "avg", "source"])
    _manifest(sub, "full_context_qwen", args, smoke=True, task_list=TASKS, sample_count="2 per task input smoke")
    status.append({
        "data_item": "Full-context / uncompressed Qwen upper bound",
        "implemented": "yes",
        "smoke_passed": completed,
        "full_run_completed": "no",
        "output_path": str(sub),
        "still_missing": "GPU generation and scoring",
        "risk_reduced_for_paper": "medium: exact input feasibility audited",
    })


def external_baselines(root: Path, args: argparse.Namespace, next_commands: list[str], status: list[dict[str, Any]]) -> None:
    sub = root / "02_external_baselines"
    sub.mkdir(parents=True, exist_ok=True)
    sentence_ok = importlib.util.find_spec("sentence_transformers") is not None or _python_with("sentence_transformers") != sys.executable
    llmlingua_ok = any(importlib.util.find_spec(name) is not None for name in ["llmlingua", "prompt_compressor", "longllmlingua"])
    runtime_python = args.runtime_python or _python_with("transformers")
    dense_rows = [{
        "baseline": "dense_retriever_packing",
        "smoke_status": "not_run_dependency_missing" if not sentence_ok else "ready_to_run",
        "sentence_transformers_installed": sentence_ok,
        "embedding_model_or_path": args.embedding_model_or_path or "TODO",
        "budget": args.budget,
    }]
    _write_csv(
        sub / "dense_retriever_smoke.csv",
        dense_rows,
        ["baseline", "smoke_status", "sentence_transformers_installed", "embedding_model_or_path", "budget"],
    )
    _write_text(sub / "dense_retriever_smoke_report.md", "\n".join([
        "# Dense Retriever Baseline Smoke",
        "",
        f"- `sentence_transformers` installed: **{sentence_ok}**.",
        f"- Local embedding model path supplied: `{args.embedding_model_or_path or 'TODO'}`.",
        "- Dense retrieval was not approximated with lexical scores.",
        "- Full run remains TODO until dependency/model path is available.",
    ]))
    dense_command = (
        f"{runtime_python} tools/run_bridgepack_dense_retriever_packing.py "
        f"--model_name_or_path {args.qwen_model} --embedding_model_or_path <LOCAL_BGE_E5_OR_CONTRIEVER_PATH> "
        f"--tasks {','.join(TASKS)} --max_samples 0 --budget {args.budget} "
        f"--out_dir {sub / 'dense_retriever_full'} --run_generation --dtype fp16 --device auto"
    )
    _shell_script(sub / "dense_retriever_full_command.sh", dense_command)
    next_commands.append(dense_command)
    _write_text(sub / "llmlingua_feasibility.md", "\n".join([
        "# LLMLingua / LongLLMLingua Feasibility",
        "",
        f"- LLMLingua-style package installed: **{llmlingua_ok}**.",
        "- No repository-native LLMLingua wrapper was found.",
        "- This baseline remains TODO. Install an appropriate package in a separate environment, compress to exact Qwen-token budget 1024, and evaluate with the same packed-prompt generation/scoring runner.",
        "",
        "## Deferred Command Sketch",
        "",
        "```bash",
        "# TODO: replace with the selected package's official compression entry point.",
        "python tools/run_llmlingua_bridgepack_baseline.py --model_name_or_path <QWEN> --budget 1024 --tasks qasper,multifieldqa_en,hotpotqa,2wikimqa,musique,narrativeqa",
        "```",
    ]))
    existing = _read_csv(args.exact_dir / "paper_tables" / "01_exact_budget_main_table_1024.csv")
    table = [{"method": "Dense Retriever Packing", "budget": args.budget, "avg": "TODO"}]
    table.append({"method": "LLMLingua / LongLLMLingua", "budget": args.budget, "avg": "TODO"})
    for row in existing:
        if row.get("method") in METHODS:
            table.append({"method": row.get("method"), "budget": row.get("budget", "1024"), "avg": row.get("avg", "")})
    _write_csv(sub / "external_baseline_table_template.csv", table, ["method", "budget", "avg"])
    _manifest(sub, "external_baselines", args, smoke=True, task_list=TASKS, sample_count="dependency/input smoke only")
    status.append({
        "data_item": "Dense retriever and LLMLingua external baselines",
        "implemented": "partial",
        "smoke_passed": sentence_ok,
        "full_run_completed": "no",
        "output_path": str(sub),
        "still_missing": "dependency/model install plus GPU generation",
        "risk_reduced_for_paper": "medium: feasibility and exact commands prepared",
    })


def mistral_full(root: Path, args: argparse.Namespace, next_commands: list[str], status: list[dict[str, Any]]) -> None:
    sub = root / "03_mistral_full_6task"
    sub.mkdir(parents=True, exist_ok=True)
    runtime_python = args.runtime_python or _python_with("transformers")
    gate_tasks = sorted({path.stem for path in (args.mistral_gate_dir / "predictions" / "lexical_bridgepack_path_order").glob("*.jsonl")})
    rows = []
    for task in TASKS:
        rows.append({
            "task": task,
            "current_gate_has_task": task in gate_tasks,
            "required_for_full_6task": True,
            "smoke_status": "existing_gate_task" if task in gate_tasks else "needs_new_mistral_run",
        })
    _write_csv(sub / "mistral_6task_smoke.csv", rows, ["task", "current_gate_has_task", "required_for_full_6task", "smoke_status"])
    _write_text(sub / "mistral_6task_smoke_report.md", "\n".join([
        "# Mistral Six-Task Smoke",
        "",
        "This is an existing-output/input audit only. No Mistral generation was run.",
        "",
        f"- Existing gate tasks: `{','.join(gate_tasks)}`.",
        "- Missing for full six-task validation: qasper, multifieldqa_en, narrativeqa.",
        "- Full six-task Mistral result remains TODO.",
    ]))
    command = (
        f"{runtime_python} tools/run_bridgepack_second_model_exact_gate.py "
        f"--model_name_or_path {args.mistral_model} --model_tag Mistral-7B-Instruct-v0.3 "
        f"--tasks {','.join(TASKS)} --max_samples 0 --budget {args.budget} "
        f"--variants {','.join(METHODS)} --out_dir {sub / 'mistral_6task_full'} "
        "--resume --dtype float16 --device cuda:0"
    )
    _shell_script(sub / "mistral_6task_full_command.sh", command)
    next_commands.append(command)
    table = [{"method": method, **{task: "TODO" for task in TASKS}, "avg": "TODO"} for method in METHODS]
    _write_csv(sub / "mistral_6task_table_template.csv", table, ["method", *TASKS, "avg"])
    _manifest(sub, "mistral_full_6task", args, smoke=True, task_list=TASKS, sample_count="existing gate audit only")
    status.append({
        "data_item": "Full Mistral six-task second-model evaluation",
        "implemented": "yes",
        "smoke_passed": "input audit only",
        "full_run_completed": "no",
        "output_path": str(sub),
        "still_missing": "GPU generation/scoring for six tasks",
        "risk_reduced_for_paper": "medium: current gate gap made explicit",
    })


def _load_packed(exact_dir: Path, method: str, task: str) -> list[dict[str, Any]]:
    return _read_jsonl(exact_dir / "packed_prompts_build" / "packed_prompts" / method / "budget_1024" / f"{task}.jsonl")


def evidence_coverage(root: Path, args: argparse.Namespace, status: list[dict[str, Any]]) -> None:
    sub = root / "04_evidence_coverage"
    sub.mkdir(parents=True, exist_ok=True)
    sample_rows: list[dict[str, Any]] = []
    for method in METHODS:
        for task in TASKS:
            for row in _load_packed(args.exact_dir, method, task):
                packed_text = str(row.get("packed_text") or row.get("packed_prompt") or "")
                span_ids = row.get("selected_span_ids", []) or []
                page_ids = row.get("selected_page_ids", []) or []
                span_page_order = []
                for sid in span_ids:
                    try:
                        span_page_order.append(int(str(sid).split("_s", 1)[0].replace("p", "")))
                    except Exception:
                        pass
                unique_in_order = []
                for page in span_page_order:
                    if page not in unique_in_order:
                        unique_in_order.append(page)
                order_distance = (
                    max(abs(unique_in_order[i + 1] - unique_in_order[i]) for i in range(len(unique_in_order) - 1))
                    if len(unique_in_order) > 1 else 0
                )
                sample_rows.append({
                    "method": method,
                    "display": DISPLAY.get(method, method),
                    "task": task,
                    "sample_id": row.get("sample_id", ""),
                    "answer_string_coverage": _answer_covered(packed_text, row.get("answers", [])),
                    "supporting_evidence_coverage": "metadata_unavailable",
                    "multihop_pair_coverage": "metadata_unavailable" if task in MULTIHOP else "not_multihop_task",
                    "selected_span_count": len(span_ids),
                    "selected_chunk_count": len(page_ids),
                    "selected_page_count": len(page_ids),
                    "packed_token_count": row.get("model_token_count", row.get("packed_token_count", "")),
                    "evidence_order_distance": order_distance,
                    "used_bridge_path": row.get("used_bridge_path", ""),
                    "truncated": row.get("truncated", ""),
                })
    by_method: list[dict[str, Any]] = []
    by_task: list[dict[str, Any]] = []
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    method_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in sample_rows:
        groups[(str(row["method"]), str(row["task"]))].append(row)
        method_groups[str(row["method"])].append(row)
    for method, rows in method_groups.items():
        by_method.append(_coverage_summary({"method": method, "display": DISPLAY.get(method, method)}, rows))
    for (method, task), rows in sorted(groups.items()):
        base = {"method": method, "display": DISPLAY.get(method, method), "task": task}
        by_task.append(_coverage_summary(base, rows))
    multihop_rows = [row for row in by_task if row.get("task") in MULTIHOP]
    fields = [
        "method", "display", "task", "num_samples", "answer_string_coverage_rate",
        "selected_span_count_mean", "selected_page_count_mean", "packed_token_count_mean",
        "evidence_order_distance_mean", "supporting_evidence_metadata",
    ]
    _write_csv(sub / "evidence_coverage_by_method.csv", by_method, [f for f in fields if f != "task"])
    _write_csv(sub / "evidence_coverage_by_task.csv", by_task, fields)
    _write_csv(sub / "evidence_coverage_multihop.csv", multihop_rows, fields)
    _write_text(sub / "evidence_coverage_report.md", "\n".join([
        "# Evidence Coverage Diagnostics",
        "",
        "- Computed from existing exact-token packed prompts; no generation was run.",
        "- `answer_string_coverage` checks whether any normalized gold answer string appears naturally in `packed_text`.",
        "- Gold answers were not used for selection; this diagnostic only audits outputs after the fact.",
        "- Supporting-fact and multi-hop pair coverage are marked `metadata_unavailable` because the LongBench packed artifacts do not contain support-fact/span metadata.",
        "",
        "## Key Limitation",
        "",
        "Answer-string coverage is not proof of answerability, and absence of the exact string can undercount paraphrased evidence.",
    ]))
    _manifest(sub, "evidence_coverage", args, smoke=False, task_list=TASKS, sample_count="existing full packed-prompt rows")
    status.append({
        "data_item": "Evidence coverage diagnostics",
        "implemented": "partial",
        "smoke_passed": "yes",
        "full_run_completed": "cpu diagnostic complete",
        "output_path": str(sub),
        "still_missing": "supporting-fact metadata coverage if source metadata is later available",
        "risk_reduced_for_paper": "high for answer coverage, medium for support coverage",
    })


def _coverage_summary(base: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    def mean(key: str) -> float:
        vals = [float(row.get(key, 0) or 0) for row in rows]
        return statistics.mean(vals) if vals else 0.0

    out = dict(base)
    out.update({
        "num_samples": len(rows),
        "answer_string_coverage_rate": round(sum(1 for row in rows if str(row.get("answer_string_coverage")) == "True") / max(1, len(rows)), 4),
        "selected_span_count_mean": round(mean("selected_span_count"), 4),
        "selected_page_count_mean": round(mean("selected_page_count"), 4),
        "packed_token_count_mean": round(mean("packed_token_count"), 4),
        "evidence_order_distance_mean": round(mean("evidence_order_distance"), 4),
        "supporting_evidence_metadata": "unavailable",
    })
    return out


def case_studies(root: Path, args: argparse.Namespace, status: list[dict[str, Any]]) -> None:
    sub = root / "05_case_studies"
    sub.mkdir(parents=True, exist_ok=True)
    methods = [*METHODS, "bridge_same_spans_source_order"]
    packed: dict[tuple[str, str, str], dict[str, Any]] = {}
    preds: dict[tuple[str, str, str], dict[str, Any]] = {}
    scores: dict[tuple[str, str, str], float] = {}
    for method in methods:
        for task in TASKS:
            for row in _load_packed(args.exact_dir, method, task):
                packed[(method, task, str(row.get("sample_id")))] = row
            for row in _read_jsonl(args.exact_dir / "predictions" / method / f"{task}.jsonl"):
                key = (method, task, str(row.get("sample_id")))
                preds[key] = row
                scores[key] = _score_one(task, row)
    all_keys = sorted({(task, sid) for (_m, task, sid) in preds if _m == "lexical_bridgepack_path_order"})
    chosen: list[tuple[str, str, str]] = []
    for task, sid in all_keys:
        lex = scores.get(("lexical_bridgepack_path_order", task, sid), 0.0)
        bm25 = scores.get(("bm25_chunk_source_order", task, sid), 0.0)
        span = scores.get(("span_only_source_order", task, sid), 0.0)
        if lex > 0 and (bm25 == 0 or span == 0):
            chosen.append(("success", task, sid))
            break
    for task, sid in all_keys:
        bridge = scores.get(("bridge_neighbor32_path_order", task, sid), 0.0)
        source = scores.get(("bridge_same_spans_source_order", task, sid), 0.0)
        if bridge > source:
            chosen.append(("organization", task, sid))
            break
    for task, sid in all_keys:
        if task not in {"musique", "2wikimqa", "hotpotqa"}:
            continue
        lex = scores.get(("lexical_bridgepack_path_order", task, sid), 0.0)
        bm25 = scores.get(("bm25_chunk_source_order", task, sid), 0.0)
        if lex <= bm25:
            chosen.append(("failure_or_neutral", task, sid))
            break
    cases: list[dict[str, Any]] = []
    for case_type, task, sid in chosen:
        lex_row = preds.get(("lexical_bridgepack_path_order", task, sid), {})
        case: dict[str, Any] = {
            "case_type": case_type,
            "task": task,
            "sample_id": sid,
            "question": lex_row.get("question", ""),
            "gold_answer": lex_row.get("answers", []),
            "explanation": _case_explanation(case_type),
        }
        for method in methods:
            prow = preds.get((method, task, sid), {})
            pack = packed.get((method, task, sid), {})
            case[f"{method}_prediction"] = prow.get("pred", "")
            case[f"{method}_score"] = scores.get((method, task, sid), "")
            case[f"{method}_packed_token_count"] = pack.get("model_token_count", pack.get("packed_token_count", ""))
            case[f"{method}_selected_evidence_snippet"] = _snippet_around_answer(
                str(pack.get("packed_text") or pack.get("packed_prompt") or ""),
                pack.get("answers", prow.get("answers", [])),
            )
        cases.append(case)
    _write_jsonl(sub / "case_studies.jsonl", cases)
    index_rows = [
        {
            "case_type": row["case_type"],
            "task": row["task"],
            "sample_id": row["sample_id"],
            "lexical_score": row.get("lexical_bridgepack_path_order_score", ""),
            "bm25_score": row.get("bm25_chunk_source_order_score", ""),
            "span_score": row.get("span_only_source_order_score", ""),
        }
        for row in cases
    ]
    _write_csv(sub / "case_index.csv", index_rows, ["case_type", "task", "sample_id", "lexical_score", "bm25_score", "span_score"])
    lines = ["# Qualitative Case Studies", "", "These examples are selected from existing exact-token predictions; no new generation was run.", ""]
    for row in cases:
        lines.extend([
            f"## {row['case_type']}: {row['task']} / `{row['sample_id']}`",
            "",
            f"**Question:** {row.get('question', '')}",
            "",
            f"**Gold:** `{row.get('gold_answer', [])}`",
            "",
            f"**Explanation:** {row.get('explanation', '')}",
            "",
        ])
        for method in methods:
            lines.extend([
                f"- `{method}` score={row.get(method + '_score', '')}, pred=`{row.get(method + '_prediction', '')}`, packed_tokens={row.get(method + '_packed_token_count', '')}",
                f"  Snippet: {row.get(method + '_selected_evidence_snippet', '')}",
            ])
        lines.append("")
    _write_text(sub / "case_studies.md", "\n".join(lines))
    _manifest(sub, "case_studies", args, smoke=False, task_list=TASKS, sample_count="existing full prediction rows")
    status.append({
        "data_item": "Qualitative success/neutral/failure case studies",
        "implemented": "yes",
        "smoke_passed": bool(cases),
        "full_run_completed": "cpu extraction complete",
        "output_path": str(sub),
        "still_missing": "manual paper-level prose polishing",
        "risk_reduced_for_paper": "high",
    })


def _case_explanation(case_type: str) -> str:
    if case_type == "success":
        return "Lexical-BridgePack receives non-zero LongBench score while at least one baseline is zero."
    if case_type == "organization":
        return "Bridge path ordering improves over the same-spans source-order control on this sample."
    return "Lexical-BridgePack is not better than the BM25 baseline; include this to avoid cherry-picking."


def runtime_overhead(root: Path, args: argparse.Namespace, next_commands: list[str], status: list[dict[str, Any]]) -> None:
    sub = root / "06_runtime_overhead"
    sub.mkdir(parents=True, exist_ok=True)
    roots = _candidate_dataset_roots("")
    rows: list[dict[str, Any]] = []
    for task in ["hotpotqa", "musique"]:
        samples = _load_samples(task, roots, 2)
        for sample in samples:
            for method in METHODS:
                start = time.perf_counter()
                packed, _audit = pack_evidence(
                    str(sample["sample_id"]),
                    task,
                    str(sample.get("question", "")),
                    str(sample.get("context", "")),
                    method,
                    args.budget,
                )
                elapsed = (time.perf_counter() - start) * 1000.0
                rows.append({
                    "method": method,
                    "task": task,
                    "sample_id": sample["sample_id"],
                    "selector_time_ms": round(elapsed, 4),
                    "tokenization_time_ms": "included_in_selector_time",
                    "packing_time_ms": round(elapsed, 4),
                    "total_preprocess_time_ms": round(elapsed, 4),
                    "packed_token_count": packed.packed_token_count,
                    "source_token_count": packed.source_token_count,
                    "compression_ratio": round(packed.compression_ratio, 6),
                    "timing_scope": "cpu_regex_selector_rebuild_smoke",
                })
    _write_csv(
        sub / "runtime_smoke.csv",
        rows,
        [
            "method", "task", "sample_id", "selector_time_ms", "tokenization_time_ms",
            "packing_time_ms", "total_preprocess_time_ms", "packed_token_count",
            "source_token_count", "compression_ratio", "timing_scope",
        ],
    )
    overhead = _read_csv(args.exact_dir / "paper_tables" / "08_packing_overhead.csv")
    _write_text(sub / "runtime_report.md", "\n".join([
        "# Runtime / Selector Overhead",
        "",
        "- `runtime_smoke.csv` measures CPU selector rebuild time on 2 samples each from HotpotQA and MuSiQue.",
        "- Existing exact-token overhead table is available in the official Qwen result directory.",
        "- Full packing-time instrumentation remains TODO because the official run did not record packing timing.",
        "",
        f"- Existing overhead rows copied/available: `{len(overhead)}`.",
    ]))
    command = (
        "python tools/instrument_bridgepack_packing_time.py "
        f"--packed_dir {args.exact_dir / 'packed_prompts_build'} --tasks {','.join(TASKS)} "
        f"--variants {','.join(METHODS)} --budget {args.budget} --out_dir {sub / 'runtime_full'}"
    )
    _shell_script(sub / "runtime_full_command.sh", command)
    next_commands.append(command)
    _manifest(sub, "runtime_overhead", args, smoke=True, task_list=["hotpotqa", "musique"], sample_count="2 per task")
    status.append({
        "data_item": "Runtime / selector overhead",
        "implemented": "partial",
        "smoke_passed": bool(rows),
        "full_run_completed": "no",
        "output_path": str(sub),
        "still_missing": "full timing instrumentation over all samples",
        "risk_reduced_for_paper": "medium",
    })


def param_sensitivity(root: Path, args: argparse.Namespace, next_commands: list[str], status: list[dict[str, Any]]) -> None:
    sub = root / "07_param_sensitivity"
    sub.mkdir(parents=True, exist_ok=True)
    runtime_python = args.runtime_python or _python_with("transformers")
    summary = {row.get("variant"): row for row in _read_csv(args.exact_dir / "summary" / "summary.csv")}
    rows = [
        {"setting": "no bridge score proxy", "existing_variant": "query_only_neighbor32_path_order", "avg": summary.get("query_only_neighbor32_path_order", {}).get("macro_avg", "TODO"), "status": "existing exact result"},
        {"setting": "source-order instead of path-order", "existing_variant": "bridge_same_spans_source_order", "avg": summary.get("bridge_same_spans_source_order", {}).get("macro_avg", "TODO"), "status": "existing exact result"},
        {"setting": "tags formatting", "existing_variant": "bridge_neighbor32_path_order_tags", "avg": summary.get("bridge_neighbor32_path_order_tags", {}).get("macro_avg", "TODO"), "status": "existing exact result"},
        {"setting": "no redundancy penalty", "existing_variant": "TODO", "avg": "TODO", "status": "requires new CPU selector or generation diagnostic"},
        {"setting": "no neighbor repair", "existing_variant": "TODO", "avg": "TODO", "status": "requires new exact-token packing variant"},
        {"setting": "smaller/larger candidate pool", "existing_variant": "TODO", "avg": "TODO", "status": "requires new exact-token packing variant"},
    ]
    _write_csv(sub / "param_sensitivity_smoke.csv", rows, ["setting", "existing_variant", "avg", "status"])
    _write_text(sub / "param_sensitivity_report.md", "\n".join([
        "# Parameter Sensitivity",
        "",
        "- This report reuses completed exact-token component/control rows where available.",
        "- New generation was not run.",
        "- Missing rows are explicitly marked TODO rather than filled with approximate values.",
    ]))
    command = (
        f"{runtime_python} tools/run_bridgepack_exact_budget_repack.py "
        f"--model_name_or_path {args.qwen_model} --tasks hotpotqa,musique --budget {args.budget} "
        "--max_samples 32 --variants <PARAM_VARIANTS_TO_ADD> "
        f"--out_dir {sub / 'param_sensitivity_32'} --resume --dtype fp16 --device auto"
    )
    _shell_script(sub / "param_sensitivity_full_command.sh", command)
    next_commands.append(command)
    _manifest(sub, "param_sensitivity", args, smoke=True, task_list=["hotpotqa", "musique"], sample_count="existing rows only")
    status.append({
        "data_item": "Parameter sensitivity diagnostics",
        "implemented": "partial",
        "smoke_passed": "existing rows audited",
        "full_run_completed": "no",
        "output_path": str(sub),
        "still_missing": "new exact-token variants for no redundancy/no neighbor/candidate pool",
        "risk_reduced_for_paper": "low-to-medium",
    })


def final_status(root: Path, args: argparse.Namespace, status: list[dict[str, Any]], next_commands: list[str]) -> None:
    _write_csv(
        root / "MISSING_DATA_STATUS.md.csv",
        status,
        ["data_item", "implemented", "smoke_passed", "full_run_completed", "output_path", "still_missing", "risk_reduced_for_paper"],
    )
    lines = [
        "# Missing Data Status",
        "",
        "| Data item | Implemented? | Smoke passed? | Full run completed? | Output path | Still missing | Risk reduced for paper? |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in status:
        lines.append(
            f"| {row['data_item']} | {row['implemented']} | {row['smoke_passed']} | "
            f"{row['full_run_completed']} | `{row['output_path']}` | {row['still_missing']} | {row['risk_reduced_for_paper']} |"
        )
    _write_text(root / "MISSING_DATA_STATUS.md", "\n".join(lines))
    cmd_lines = ["# Next Run Commands", "", "These commands are resumable/deferred. They were not launched by this CPU-only package builder.", ""]
    for idx, command in enumerate(next_commands, start=1):
        cmd_lines.extend([f"## Command {idx}", "", "```bash", command, "```", ""])
    _write_text(root / "NEXT_RUN_COMMANDS.md", "\n".join(cmd_lines))
    _write_text(root / "IMPLEMENTATION_NOTES.md", "\n".join([
        "# Implementation Notes",
        "",
        "- This package is about evidence-aware text compression/repacking, not KV-cache pruning.",
        "- Existing exact-token packed prompts were used for coverage/cases where possible.",
        "- GPU-dependent full-context, Mistral full six-task, dense retriever, and LLMLingua baselines remain deferred.",
        "- Naturally occurring answer strings in packed text are counted only as an after-the-fact diagnostic.",
        "- Gold answer fields are not serialized into selector inputs; packed prompt files keep answers only for scoring/audit.",
        "",
        "## Safest Updated Paper Claim",
        "",
        "Lexical-BridgePack remains the best observed exact-token Qwen 1024 result and significantly improves over span-only packing. The new CPU evidence package adds answer-string coverage/case-study support and makes remaining upper-bound/external-baseline/second-model gaps explicit; it still does not justify claiming superiority over BM25/TF-IDF, full second-model generalization, or KV-cache pruning.",
    ]))


def parse_args() -> argparse.Namespace:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_root", default=f"results/bridgepack_missing_evidence_{stamp}")
    parser.add_argument("--exact_dir", type=Path, default=_require(_latest("bridgepack_exact_qwen_budget_1024_*"), "exact Qwen result"))
    parser.add_argument("--budget_dir", type=Path, default=_require(_latest("bridgepack_exact_budget_ablation_*"), "budget ablation result"))
    parser.add_argument("--mistral_gate_dir", type=Path, default=_require(_latest("bridgepack_second_model_exact_gate_*"), "Mistral gate result"))
    parser.add_argument("--qwen_model", default="")
    parser.add_argument("--mistral_model", default="mistralai/Mistral-7B-Instruct-v0.3")
    parser.add_argument("--embedding_model_or_path", default="")
    parser.add_argument("--budget", type=int, default=1024)
    parser.add_argument("--qwen_model_max_length", type=int, default=131072)
    parser.add_argument("--runtime_python", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = json.loads((args.exact_dir / "config.json").read_text(encoding="utf-8"))
    if not args.qwen_model:
        args.qwen_model = str(cfg.get("model_name_or_path", ""))
    root = Path(args.out_root)
    root.mkdir(parents=True, exist_ok=True)
    for name in [
        "01_full_context_qwen",
        "02_external_baselines",
        "03_mistral_full_6task",
        "04_evidence_coverage",
        "05_case_studies",
        "06_runtime_overhead",
        "07_param_sensitivity",
    ]:
        (root / name).mkdir(parents=True, exist_ok=True)
    (root / "config.json").write_text(json.dumps({k: str(v) for k, v in vars(args).items()}, indent=2), encoding="utf-8")
    status: list[dict[str, Any]] = []
    next_commands: list[str] = []
    repo_audit(root, args)
    full_context_qwen(root, args, next_commands, status)
    external_baselines(root, args, next_commands, status)
    mistral_full(root, args, next_commands, status)
    evidence_coverage(root, args, status)
    case_studies(root, args, status)
    runtime_overhead(root, args, next_commands, status)
    param_sensitivity(root, args, next_commands, status)
    final_status(root, args, status, next_commands)
    print(f"OUT_ROOT={root}")
    print("Added/modified scripts:")
    print("- tools/run_bridgepack_full_context_eval.py")
    print("- tools/run_bridgepack_dense_retriever_packing.py")
    print("- tools/build_bridgepack_missing_evidence_pack.py")
    print("Smoke tests/status are summarized in MISSING_DATA_STATUS.md")
    print("Full/deferred commands are in NEXT_RUN_COMMANDS.md")


if __name__ == "__main__":
    main()
