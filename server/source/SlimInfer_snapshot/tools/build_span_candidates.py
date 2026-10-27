#!/usr/bin/env python3
"""Build fixed-window evidence span candidates for SpanDRM-Add."""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "experiments"))

import run_qwen25_sliminfer_variants as R

DEFAULT_MODEL = "/home/yu-bai/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28"
DEFAULT_LB = "/home/yu-bai/Desktop/jiaqizhang/HeadKV/data/LongBench"
DEFAULT_FUP = REPO_ROOT / "results" / "fup_guarded_swap_stage2_64_20260601_051737"
TASKS = ["musique", "hotpotqa", "qasper", "passage_count", "passage_retrieval_en"]
PAGE_TOKENS = 64
TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")
ENTITY_RE = re.compile(r"\b[A-Z][A-Za-z0-9_\-]{2,}\b")
STOPWORDS = {
    "the", "a", "an", "of", "to", "in", "on", "for", "and", "or", "is", "are",
    "was", "were", "be", "by", "with", "as", "at", "that", "this", "it", "its",
    "what", "which", "who", "whom", "whose", "when", "where", "how", "why",
    "do", "does", "did", "from", "about", "into", "than", "then", "there",
}


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    return rows


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: List[Dict[str, Any]], header: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in header})


def latest_layer_records(path: Path) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for row in read_jsonl(path):
        sid = str(row.get("sample_id", ""))
        old = out.get(sid)
        if old is None or int(row.get("layer_id", -1) or -1) >= int(old.get("layer_id", -1) or -1):
            out[sid] = row
    return out


def words(text: str) -> Set[str]:
    return {word.lower() for word in TOKEN_RE.findall(text or "") if len(word) > 1 and word.lower() not in STOPWORDS}


def entities(text: str) -> Set[str]:
    return {item.lower() for item in ENTITY_RE.findall(text or "")}


def overlap(left: Set[str], right: Set[str]) -> float:
    return len(left & right) / max(1, len(left)) if left else 0.0


def jaccard(left: Set[str], right: Set[str]) -> float:
    return len(left & right) / max(1, len(left | right)) if left or right else 0.0


def load_page_features(path: Path) -> Dict[Tuple[str, str, int], Dict[str, float]]:
    out: Dict[Tuple[str, str, int], Dict[str, float]] = defaultdict(dict)
    with path.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = (str(row["task"]), str(row["sample_id"]), int(row["page_id"]))
            current = out[key]
            for src, dst in [
                ("feature_drm_delta", "drm"),
                ("feature_query_similarity", "query"),
                ("feature_answer_overlap", "answer"),
            ]:
                value = float(row.get(src, 0.0) or 0.0)
                current[dst] = max(float(current.get(dst, -1e9)), value)
    return out


def load_fup_scores(path: Path) -> Dict[Tuple[str, str, int], float]:
    out: Dict[Tuple[str, str, int], float] = {}
    if not path.exists():
        return out
    with path.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            out[(str(row["task"]), str(row["sample_id"]), int(row["page_id"]))] = float(
                row.get("output_delta_utility", 0.0) or 0.0
            )
    return out


def page_indices(page: int, context_length: int) -> List[int]:
    start = max(0, int(page) * PAGE_TOKENS)
    end = min(context_length, start + PAGE_TOKENS)
    return list(range(start, end))


def fixed_page_sets_from_audit(audit: Dict[str, Any]) -> Tuple[Set[int], Set[int], Set[int]]:
    """Recover final, active, and rescued original page IDs from a pruning audit."""
    final_pages = {int(x) for x in audit.get("final_block_ids_first100", [])}
    active_pages = {int(x) for x in audit.get("active_fixed_page_ids", [])}
    rescue_pages = {int(x) for x in audit.get("rescued_fixed_page_ids", [])}
    if active_pages or rescue_pages:
        return final_pages, active_pages, rescue_pages

    # Older fixed DRM-Add audits only recorded local virtual rescue IDs. Fixed
    # virtual blocks preserve order, so the paired final ID lists recover the
    # original page IDs at the final prune layer.
    final_virtual = [int(x) for x in audit.get("final_virtual_ids_first100", [])]
    final_fixed = [int(x) for x in audit.get("final_block_ids_first100", [])]
    virtual_to_fixed = dict(zip(final_virtual, final_fixed))
    rescue_virtual = [int(x) for x in audit.get("rescued_virtual_ids_first50", [])]
    rescue_pages = {virtual_to_fixed[x] for x in rescue_virtual if x in virtual_to_fixed}
    active_pages = final_pages - rescue_pages
    return final_pages, active_pages, rescue_pages


def tokenize_sample(
    tokenizer: Any,
    prompt_format: str,
    task: str,
    sample: Dict[str, Any],
    max_model_len: int,
) -> Tuple[str, List[int]]:
    prompt = prompt_format.format(**sample)
    tokenized = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
    if len(tokenized) > max_model_len:
        half = max_model_len // 2
        prompt = tokenizer.decode(tokenized[:half], skip_special_tokens=True) + tokenizer.decode(
            tokenized[-half:], skip_special_tokens=True
        )
    if task not in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]:
        prompt = R.build_chat(tokenizer, prompt)
    ids = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0].tolist()
    return prompt, [int(x) for x in ids]


def build_sample_candidates(
    *,
    task: str,
    sample: Dict[str, Any],
    sample_index: int,
    tokenizer: Any,
    prompt_format: str,
    audit: Optional[Dict[str, Any]],
    page_features: Dict[Tuple[str, str, int], Dict[str, float]],
    fup_scores: Dict[Tuple[str, str, int], float],
    span_tokens: int,
    max_model_len: int,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    sample_id = str(sample.get("_id", sample.get("id", sample_index)))
    _prompt, input_ids = tokenize_sample(tokenizer, prompt_format, task, sample, max_model_len)
    context_length = len(input_ids)
    num_pages = int(math.ceil(context_length / PAGE_TOKENS))
    query_text = R._sample_query_text(sample)
    answer_text = " ".join(str(item) for item in sample.get("answers", []))
    query_words = words(query_text)
    query_entities = entities(query_text)
    answer_words = words(answer_text)

    fallback_reason = ""
    if audit:
        final_pages, active_pages, rescue_pages = fixed_page_sets_from_audit(audit)
        target_final_tokens = int(audit.get("selected_token_count", 0) or 0)
        source_layer = int(audit.get("layer_id", -1) or -1)
    else:
        final_pages = set(range(num_pages))
        rescue_pages = set()
        active_pages = set(final_pages)
        target_final_tokens = context_length
        source_layer = -1
        fallback_reason = "missing_fixed_drm_audit_or_no_pruning"

    final_pages = {page for page in final_pages if 0 <= page < num_pages}
    rescue_pages = {page for page in rescue_pages if 0 <= page < num_pages}
    active_pages = {page for page in active_pages if 0 <= page < num_pages}
    if not final_pages:
        final_pages = set(range(num_pages))
        active_pages = set(final_pages)
        rescue_pages = set()
        target_final_tokens = context_length
        fallback_reason = "empty_fixed_drm_final_pages"

    candidates: List[Dict[str, Any]] = []
    for page in range(num_pages):
        if page in active_pages:
            continue
        start = page * PAGE_TOKENS
        end = min(context_length, start + PAGE_TOKENS)
        page_text = tokenizer.decode(input_ids[start:end], skip_special_tokens=True, clean_up_tokenization_spaces=False)
        title = next((line.strip() for line in page_text.splitlines() if line.strip()), f"page_{page}")
        windows = [(pos, min(end, pos + span_tokens)) for pos in range(start, end, span_tokens)]
        for local_index, (span_start, span_end) in enumerate(windows):
            if span_end <= span_start:
                continue
            text = tokenizer.decode(
                input_ids[span_start:span_end],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            span_words = words(text)
            key = (task, sample_id, page)
            parent = page_features.get(key, {})
            drm = float(parent.get("drm", 0.0))
            query_similarity = jaccard(query_words, span_words)
            keyword_overlap = overlap(query_words, span_words)
            entity_overlap = overlap(query_entities, entities(text))
            answer_overlap = overlap(answer_words, span_words)
            sentence_drm_proxy = drm * (0.5 + query_similarity + 0.5 * keyword_overlap)
            candidates.append({
                "task": task,
                "sample_id": sample_id,
                "page_id": page,
                "span_id": f"p{page}_w{local_index}",
                "span_text_len": len(text),
                "span_token_len": span_end - span_start,
                "sentence_position": local_index,
                "page_position": round(page / max(1, num_pages - 1), 7),
                "query_similarity": round(query_similarity, 7),
                "query_keyword_overlap": round(keyword_overlap, 7),
                "entity_overlap": round(entity_overlap, 7),
                "drm_score_parent_page": round(drm, 7),
                "sentence_drm_proxy": round(sentence_drm_proxy, 7),
                "fup_score_parent_page": round(float(fup_scores.get(key, 0.0)), 7),
                "answer_overlap_oracle_diag": round(answer_overlap, 7),
                "redundancy_with_selected_spans": 0.0,
                "source_page_selected_by": "fixed_drm_add" if page in rescue_pages else "budget_refill_pool",
                "final_span_score": 0.0,
                "selected": 0,
                "token_start": span_start,
                "token_end": span_end,
                "page_title": title[:160],
                "span_text_preview": text[:240],
            })
    meta = {
        "task": task,
        "sample_id": sample_id,
        "sample_index": sample_index,
        "context_length": context_length,
        "num_pages": num_pages,
        "target_final_tokens": target_final_tokens,
        "source_layer": source_layer,
        "active_pages": sorted(active_pages),
        "rescue_pages": sorted(rescue_pages),
        "final_pages": sorted(final_pages),
        "fallback_reason": fallback_reason,
        "input_ids": input_ids,
        "answers": sample.get("answers", []),
        "all_classes": sample.get("all_classes", []),
        "query_text": query_text,
    }
    return meta, candidates


def build(args: argparse.Namespace) -> Path:
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    prompt_formats = json.loads((REPO_ROOT / "configs" / "dataset2prompt.json").read_text(encoding="utf-8"))
    page_features = load_page_features(Path(args.page_features))
    fup_scores = load_fup_scores(Path(args.fup_scores))
    all_meta: List[Dict[str, Any]] = []
    all_candidates: List[Dict[str, Any]] = []
    for task in args.tasks:
        audits = latest_layer_records(Path(args.baseline_dir) / "preds" / f"fixed_drm_add__{task}__audit.jsonl")
        samples = R.load_longbench_samples(Path(args.longbench_dir), task)[:args.samples_per_task]
        for index, sample in enumerate(samples):
            sample_id = str(sample.get("_id", sample.get("id", index)))
            meta, candidates = build_sample_candidates(
                task=task,
                sample=sample,
                sample_index=index,
                tokenizer=tokenizer,
                prompt_format=prompt_formats[task],
                audit=audits.get(sample_id),
                page_features=page_features,
                fup_scores=fup_scores,
                span_tokens=args.span_tokens,
                max_model_len=args.max_model_len,
            )
            meta["candidates"] = candidates
            all_meta.append(meta)
            all_candidates.extend(candidates)
        print(f"[span-candidates] {task}: {len(samples)} samples", flush=True)
    write_jsonl(out / "_span_candidates.jsonl", all_meta)
    header = [
        "task", "sample_id", "page_id", "span_id", "span_text_len", "span_token_len",
        "sentence_position", "page_position", "query_similarity", "query_keyword_overlap",
        "entity_overlap", "drm_score_parent_page", "sentence_drm_proxy", "fup_score_parent_page",
        "answer_overlap_oracle_diag", "redundancy_with_selected_spans", "source_page_selected_by",
        "final_span_score", "selected",
    ]
    write_csv(out / "span_candidate_features.csv", all_candidates, header)
    (out / "config_span_candidates.json").write_text(json.dumps({
        "tasks": args.tasks,
        "samples_per_task": args.samples_per_task,
        "span_unit": f"fixed_token_window_{args.span_tokens}",
        "baseline_dir": args.baseline_dir,
        "page_features": args.page_features,
        "fup_scores": args.fup_scores,
        "python_env": str(REPO_ROOT / ".venv"),
        "env_note": "conda env silm unavailable; repo .venv used",
    }, indent=2), encoding="utf-8")
    print(f"[span-candidates] wrote {len(all_candidates)} spans -> {out}", flush=True)
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", default=",".join(TASKS))
    parser.add_argument("--samples_per_task", type=int, default=64)
    parser.add_argument("--span_tokens", type=int, default=24)
    parser.add_argument("--max_model_len", type=int, default=32768)
    parser.add_argument("--model_path", default=DEFAULT_MODEL)
    parser.add_argument("--longbench_dir", default=DEFAULT_LB)
    parser.add_argument("--baseline_dir", default=str(DEFAULT_FUP))
    parser.add_argument("--page_features", default=str(DEFAULT_FUP / "features_fup_candidates.csv"))
    parser.add_argument("--fup_scores", default=str(DEFAULT_FUP / "fup_output_delta_scores.csv"))
    parser.add_argument("--output_dir", default=str(REPO_ROOT / "results" / "span_candidate_build"))
    args = parser.parse_args()
    args.tasks = [task for task in args.tasks.split(",") if task]
    return args


if __name__ == "__main__":
    build(parse_args())
