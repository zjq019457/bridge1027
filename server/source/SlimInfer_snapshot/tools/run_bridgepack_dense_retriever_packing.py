#!/usr/bin/env python3
"""Dense-retriever packed-context baseline for BridgePack.

The script requires `sentence_transformers` and a local embedding model path.
It deliberately fails fast when those are absent rather than substituting a
lexical baseline and calling it dense retrieval.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bridgekv.evidence_text_packing import (  # noqa: E402
    _expanded_piece,
    _prefix,
    build_candidate_spans,
)
from tools.run_bridgepack_exact_budget_repack import (  # noqa: E402
    _candidate_dataset_roots,
    _generation_prompt,
    _load_samples,
    _load_tokenizer,
    _token_count,
    _truncate_exact,
)

DEFAULT_TASKS = "qasper,multifieldqa_en,hotpotqa,2wikimqa,musique,narrativeqa"
VARIANT = "dense_retriever_packing"


def _parse_list(value: str) -> list[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(float(x) * float(y) for x, y in zip(a, b))
    na = math.sqrt(sum(float(x) * float(x) for x in a))
    nb = math.sqrt(sum(float(y) * float(y) for y in b))
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    return dot / (na * nb)


def _load_sentence_transformer(model_path: str) -> Any:
    try:
        from sentence_transformers import SentenceTransformer
    except Exception as exc:  # pragma: no cover - depends on optional package.
        raise RuntimeError(
            "sentence_transformers is not installed. Install it and pass a local "
            "--embedding_model_or_path; this script will not silently download or "
            "fallback to lexical scoring."
        ) from exc
    path = Path(model_path).expanduser()
    if not path.exists():
        raise RuntimeError(
            f"Embedding model path does not exist locally: {model_path}. "
            "Use a cached/local bge/e5/contriever/sentence-transformers path."
        )
    return SentenceTransformer(str(path), device="cpu")


def _dense_pack(
    embedder: Any,
    tokenizer: Any,
    sample: Mapping[str, Any],
    budget: int,
    neighbor_tokens: int,
    top_k: int,
    model_name_or_path: str,
) -> dict[str, Any]:
    task = str(sample["task"])
    sample_id = str(sample["sample_id"])
    question = str(sample.get("question", ""))
    context = str(sample.get("context", ""))
    candidates, page_map = build_candidate_spans(sample_id, task, question, context)
    if not candidates:
        packed_text = _prefix(question, False)
        selected = []
    else:
        query_vec = embedder.encode([question], normalize_embeddings=True, show_progress_bar=False)[0]
        chunk_vecs = embedder.encode([span.text for span in candidates], normalize_embeddings=True, show_progress_bar=False)
        scored = sorted(
            zip(candidates, chunk_vecs),
            key=lambda item: _cosine(query_vec, item[1]),
            reverse=True,
        )[: max(1, top_k)]
        pieces: list[str] = []
        selected = []
        prefix = _prefix(question, False)
        seen_pages: set[int] = set()
        for span, _vec in scored:
            piece, _ = _expanded_piece(span, page_map, neighbor_tokens, role="support", use_tags=False)
            candidate_text = "\n\n".join([prefix, *pieces, piece]).strip()
            if _token_count(tokenizer, candidate_text) <= budget:
                pieces.append(piece)
                selected.append(span)
                seen_pages.add(span.page_id)
        packed_text = "\n\n".join([prefix, *pieces]).strip()
        if _token_count(tokenizer, packed_text) > budget:
            packed_text = _truncate_exact(tokenizer, packed_text, budget)
    token_count = _token_count(tokenizer, packed_text)
    prompt = _generation_prompt(task, packed_text, question)
    return {
        "sample_id": sample_id,
        "task": task,
        "variant": VARIANT,
        "budget": budget,
        "packed_text": packed_text,
        "packed_prompt": packed_text,
        "generation_prompt": prompt,
        "question": question,
        "answers": sample.get("answers", []),
        "gold_analysis_only_answers": sample.get("answers", []),
        "all_classes": sample.get("all_classes", None),
        "length": sample.get("length", 0),
        "source_file": sample.get("source_file", ""),
        "selected_span_ids": [span.span_id for span in selected],
        "selected_page_ids": sorted({span.page_id for span in selected}),
        "packed_token_count": token_count,
        "model_token_count": token_count,
        "exact_model_token_budget": True,
        "model_tokenizer": model_name_or_path,
        "embedding_model_or_path": "",
        "neighbor_tokens": neighbor_tokens,
        "top_k": top_k,
        "truncated": False,
    }


def build_inputs(args: argparse.Namespace) -> tuple[Path, list[dict[str, Any]]]:
    embedder = _load_sentence_transformer(args.embedding_model_or_path)
    tokenizer = _load_tokenizer(args.model_name_or_path)
    tasks = _parse_list(args.tasks)
    roots = _candidate_dataset_roots(args.longbench_root)
    out_dir = Path(args.out_dir)
    packed_root = out_dir / "packed_prompts_build"
    rows: list[dict[str, Any]] = []
    for task in tasks:
        samples = _load_samples(task, roots, args.max_samples)
        path = packed_root / "packed_prompts" / VARIANT / f"budget_{args.budget}" / f"{task}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for sample in samples:
                payload = _dense_pack(
                    embedder,
                    tokenizer,
                    sample,
                    args.budget,
                    args.neighbor_tokens,
                    args.top_k,
                    args.model_name_or_path,
                )
                payload["embedding_model_or_path"] = args.embedding_model_or_path
                json.dump(payload, handle, ensure_ascii=False)
                handle.write("\n")
                rows.append({
                    "task": task,
                    "sample_id": payload["sample_id"],
                    "packed_token_count": payload["packed_token_count"],
                    "selected_span_count": len(payload["selected_span_ids"]),
                    "selected_page_count": len(payload["selected_page_ids"]),
                    "budget_ok": int(payload["packed_token_count"]) <= int(args.budget),
                })
    return packed_root, rows


def run_generation(args: argparse.Namespace, packed_root: Path) -> None:
    cmd = [
        sys.executable,
        str(REPO_ROOT / "tools" / "run_evidence_text_packing_generation_qwen32.py"),
        "--packed_dir", str(packed_root),
        "--tasks", args.tasks,
        "--max_samples", str(args.max_samples),
        "--budget", str(args.budget),
        "--variants", VARIANT,
        "--model_name_or_path", args.model_name_or_path,
        "--out_dir", args.out_dir,
        "--dtype", args.dtype,
        "--device", args.device,
        "--resume",
    ]
    log_path = Path(args.out_dir) / "logs" / "dense_retriever_generation.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write("$ " + " ".join(cmd) + "\n")
        proc = subprocess.run(cmd, cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT, text=True, check=False)
        log.write(f"returncode={proc.returncode}\n")
    if proc.returncode != 0:
        raise RuntimeError(f"Generation failed; see {log_path}")


def parse_args() -> argparse.Namespace:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--embedding_model_or_path", required=True)
    parser.add_argument("--tasks", default=DEFAULT_TASKS)
    parser.add_argument("--max_samples", type=int, default=2)
    parser.add_argument("--budget", type=int, default=1024)
    parser.add_argument("--neighbor_tokens", type=int, default=32)
    parser.add_argument("--top_k", type=int, default=64)
    parser.add_argument("--longbench_root", default="")
    parser.add_argument("--out_dir", default=f"results/bridgepack_dense_retriever_qwen_{stamp}")
    parser.add_argument("--run_generation", action="store_true")
    parser.add_argument("--dtype", default="fp16")
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    try:
        packed_root, rows = build_inputs(args)
    except Exception as exc:
        _write_text(out_dir / "dense_retriever_feasibility.md", "\n".join([
            "# Dense Retriever Feasibility",
            "",
            "Dense Retriever Packing was not executed.",
            "",
            "## Reason",
            "",
            "```text",
            str(exc),
            "```",
            "",
            "Install `sentence_transformers` and pass a local embedding model path to run this baseline.",
        ]))
        raise
    _write_csv(
        out_dir / "dense_retriever_input_audit.csv",
        rows,
        ["task", "sample_id", "packed_token_count", "selected_span_count", "selected_page_count", "budget_ok"],
    )
    _write_text(out_dir / "dense_retriever_input_report.md", "\n".join([
        "# Dense Retriever Input Audit",
        "",
        f"- Rows: {len(rows)}.",
        f"- Budget: {args.budget}.",
        f"- Embedding model: `{args.embedding_model_or_path}`.",
        f"- Generation executed: **{bool(args.run_generation)}**.",
    ]))
    if args.run_generation:
        run_generation(args, packed_root)
    print(f"OUT_DIR={out_dir}")


if __name__ == "__main__":
    main()
