#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import statistics
import string
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from bridgekv import evidence_text_packing as etp
from bridgekv.bridgepack_cpu_extensions import _pack_ordered


OUT_ROOT = REPO_ROOT / "results" / "bridgepack_p0_gapfill_20260728_021120" / "param_sensitivity_dev"
CACHE_ROOT = REPO_ROOT / "results" / "bridgepack_p0_gapfill_20260728_021120" / "cache" / "supporting_facts"
TASKS = ("hotpotqa", "2wikimqa", "musique")

DEFAULT_NEIGHBOR = 32
DEFAULT_SPAN = 64
DEFAULT_REDUNDANCY = 0.55
DEFAULT_MAX_PATHS = 24
DEFAULT_PAGE_SIZE = 256

NEIGHBORS = (0, 16, 32, 64)
SPANS = (32, 64, 128)
REDUNDANCY_RANGE = (0.40, 0.55, 0.70)
MAX_PATHS_RANGE = (16, 24, 32)


def _norm(text: Any) -> str:
    text = str(text).lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    return " ".join(text.split())


def _sent_split(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+", str(text).strip())
    return [p.strip() for p in parts if p.strip()]


def _title_block_map(context: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    chunks = re.split(r"\n(?=Passage\s+\d+:\s*)", context)
    for ch in chunks:
        m = re.match(r"Passage\s+\d+:\s*([^\n]+)\n?(.*)$", ch, flags=re.S)
        if not m:
            continue
        title = m.group(1).strip()
        body = m.group(2).strip()
        out[title] = body
        out[_norm(title)] = body
    return out


def _fact_sentence(context: Any, title: str, sent_id: int) -> Optional[str]:
    if isinstance(context, list):
        for item in context:
            if isinstance(item, (list, tuple)) and len(item) >= 2 and str(item[0]) == title:
                sents = item[1] if isinstance(item[1], list) else _sent_split(str(item[1]))
                if 0 <= sent_id < len(sents):
                    return str(sents[sent_id])
    if isinstance(context, dict):
        titles = context.get("title")
        sents_all = context.get("sentences")
        if titles is not None and sents_all is not None:
            for t, sents in zip(list(titles), list(sents_all)):
                if str(t) == title and 0 <= sent_id < len(sents):
                    return str(sents[sent_id])
    if not isinstance(context, str):
        context = str(context or "")
    blocks = _title_block_map(context)
    body = blocks.get(title) or blocks.get(_norm(title))
    if body is None:
        idx = context.lower().find(title.lower())
        if idx < 0:
            return None
        body = context[idx : idx + 2500]
    sents = _sent_split(body)
    if 0 <= sent_id < len(sents):
        return sents[sent_id]
    return None


def _fact_in_packed(packed: str, fact_text: Optional[str], title: str) -> bool:
    if fact_text and _norm(fact_text) and _norm(fact_text) in _norm(packed):
        return True
    if title and _norm(title) in _norm(packed) and fact_text:
        ft = set(_norm(fact_text).split())
        pt = set(_norm(packed).split())
        if ft and len(ft & pt) / max(1, len(ft)) >= 0.6:
            return True
    return False


def _answer_in_text(answers: Sequence[Any], text: str) -> bool:
    hay = _norm(text)
    for ans in answers:
        needle = _norm(ans)
        if needle and needle in hay:
            return True
    return False


def _mean(values: Iterable[float]) -> float:
    vals = list(values)
    if not vals:
        return float("nan")
    return sum(vals) / len(vals)


def _format_float(value: float, digits: int = 4) -> str:
    if value != value:
        return "NA"
    return f"{value:.{digits}f}"


def _stable_key(task: str, sample_id: str, question: str) -> str:
    return hashlib.sha256(f"{task}::{sample_id}::{question}".encode("utf-8")).hexdigest()


def _hotpot_context_to_text(context: Mapping[str, Any]) -> str:
    titles_obj = context.get("title")
    sents_obj = context.get("sentences")
    titles = list(titles_obj) if titles_obj is not None else []
    all_sents = list(sents_obj) if sents_obj is not None else []
    blocks = []
    for idx, (title, sents) in enumerate(zip(titles, all_sents), start=1):
        body = " ".join(str(s).strip() for s in list(sents) if str(s).strip())
        blocks.append(f"Passage {idx}: {title}\n{body}")
    return "\n".join(blocks)


def _2wiki_context_to_text(context: Sequence[Any]) -> str:
    blocks = []
    for idx, item in enumerate(context, start=1):
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        title = str(item[0])
        sents = item[1] if isinstance(item[1], list) else [item[1]]
        body = " ".join(str(s).strip() for s in sents if str(s).strip())
        blocks.append(f"Passage {idx}: {title}\n{body}")
    return "\n".join(blocks)


def _musique_context_to_text(paragraphs: Sequence[Mapping[str, Any]]) -> str:
    blocks = []
    for idx, para in enumerate(paragraphs, start=1):
        title = str(para.get("title") or f"para_{idx-1}")
        body = str(para.get("paragraph_text") or para.get("text") or "").strip()
        blocks.append(f"Passage {idx}: {title}\n{body}")
    return "\n".join(blocks)


def load_source_dev(cache_root: Path) -> Dict[str, List[Dict[str, Any]]]:
    import pandas as pd

    out: Dict[str, List[Dict[str, Any]]] = {}

    hotpot_path = cache_root / "hotpot" / "distractor" / "validation-00000-of-00001.parquet"
    hotpot_df = pd.read_parquet(hotpot_path)
    hotpot_rows: List[Dict[str, Any]] = []
    for _, row in hotpot_df.iterrows():
        sf = row["supporting_facts"]
        titles = list(sf["title"]) if isinstance(sf, dict) else []
        sids = list(sf["sent_id"]) if isinstance(sf, dict) else []
        facts = [{"title": str(t), "sent_id": int(s)} for t, s in zip(titles, sids)]
        hotpot_rows.append(
            {
                "task": "hotpotqa",
                "sample_id": str(row["id"]),
                "question": str(row["question"]),
                "answers": [row["answer"]],
                "context_text": _hotpot_context_to_text(row["context"]),
                "orig_context": row["context"],
                "supporting_facts": facts,
            }
        )
    out["hotpotqa"] = hotpot_rows

    wiki_path = cache_root / "2wiki" / "dev.json"
    wiki_raw = json.loads(wiki_path.read_text(encoding="utf-8"))
    wiki_rows: List[Dict[str, Any]] = []
    for ex in wiki_raw:
        facts = []
        for item in ex.get("supporting_facts") or []:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                facts.append({"title": str(item[0]), "sent_id": int(item[1])})
        wiki_rows.append(
            {
                "task": "2wikimqa",
                "sample_id": str(ex.get("_id") or ex.get("id") or ""),
                "question": str(ex.get("question") or ""),
                "answers": [ex.get("answer")],
                "context_text": _2wiki_context_to_text(ex.get("context") or []),
                "orig_context": ex.get("context"),
                "supporting_facts": facts,
            }
        )
    out["2wikimqa"] = wiki_rows

    musique_path = cache_root / "musique" / "musique_ans_v1.0_dev.jsonl"
    musique_rows: List[Dict[str, Any]] = []
    for line in musique_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        ex = json.loads(line)
        paragraphs = ex.get("paragraphs") or []
        decomp = ex.get("question_decomposition") or []
        facts = []
        for hi, step in enumerate(decomp):
            sp = step.get("paragraph_support_idx")
            if sp is None:
                idxs = step.get("support_idxs") or step.get("paragraph_support_idxs") or []
            else:
                idxs = [sp]
            for sp_i in idxs:
                sp_i = int(sp_i)
                if 0 <= sp_i < len(paragraphs):
                    para = paragraphs[sp_i]
                    facts.append(
                        {
                            "title": str(para.get("title") or f"para_{sp_i}"),
                            "sent_id": 0,
                            "paragraph_text": str(para.get("paragraph_text") or para.get("text") or ""),
                            "hop": hi + 1,
                        }
                    )
        if not facts:
            for idx, para in enumerate(paragraphs):
                if para.get("is_supporting"):
                    facts.append(
                        {
                            "title": str(para.get("title") or f"para_{idx}"),
                            "sent_id": 0,
                            "paragraph_text": str(para.get("paragraph_text") or para.get("text") or ""),
                            "hop": idx + 1,
                        }
                    )
        musique_rows.append(
            {
                "task": "musique",
                "sample_id": str(ex.get("id") or ""),
                "question": str(ex.get("question") or ""),
                "answers": [ex.get("answer")],
                "context_text": _musique_context_to_text(paragraphs),
                "orig_context": paragraphs,
                "supporting_facts": facts,
            }
        )
    out["musique"] = musique_rows
    return out


def select_subset(rows_by_task: Mapping[str, Sequence[Mapping[str, Any]]], per_task: int) -> List[Dict[str, Any]]:
    selected: List[Dict[str, Any]] = []
    for task in TASKS:
        pool = []
        for row in rows_by_task[task]:
            if not row.get("sample_id") or not row.get("question") or not row.get("context_text"):
                continue
            if not row.get("supporting_facts"):
                continue
            sample = dict(row)
            sample["deterministic_key"] = _stable_key(task, str(row["sample_id"]), str(row["question"]))
            pool.append(sample)
        pool.sort(key=lambda r: r["deterministic_key"])
        selected.extend(pool[:per_task])
    return selected


def select_bridge_paths_custom(
    query: str,
    candidates: Sequence[etp.CandidateSpan],
    page_map: Mapping[int, Mapping[str, Any]],
    budget: int,
    neighbor_size: int,
    max_paths: int,
    redundancy_penalty: float,
) -> Tuple[List[Tuple[etp.CandidateSpan, str]], Dict[str, Any]]:
    prefix = etp._prefix(query, False)
    selected_paths: List[etp.EvidencePath] = []
    selected_spans: List[etp.CandidateSpan] = []
    selected_roles: Dict[str, str] = {}
    selected_terms = set()
    pieces: List[str] = []
    remaining = etp.build_candidate_paths(query, candidates)
    path_terms = {
        id(path): set(etp._content_tokens(path.span_i.text)) | set(etp._content_tokens(path.span_j.text))
        for path in remaining
    }
    for _ in range(max_paths):
        best: Optional[Tuple[float, etp.EvidencePath]] = None
        for path in remaining:
            pair_terms = path_terms.get(id(path), set())
            redundancy = etp.jaccard(pair_terms, selected_terms) if selected_terms else 0.0
            adjusted = path.path_score - redundancy_penalty * redundancy
            if best is None or adjusted > best[0]:
                best = (adjusted, path)
        if best is None or best[0] <= 0.0:
            break
        path = best[1]
        trial = [(path.span_i, "query"), (path.span_j, "bridge")]
        trial_pieces = list(pieces)
        can_take = False
        for span, role in trial:
            if any(etp._overlaps(span, prior, max_fraction=0.30) for prior in selected_spans):
                continue
            piece, _ = etp._expanded_piece(span, page_map, neighbor_size, role=role, use_tags=False)
            if not etp._can_add(prefix, trial_pieces, piece, budget):
                continue
            trial_pieces.append(piece)
            can_take = True
        if not can_take:
            remaining = [p for p in remaining if p is not path]
            if not remaining:
                break
            continue
        selected_paths.append(path)
        pieces = trial_pieces
        for span, role in trial:
            if not any(etp._overlaps(span, prior, max_fraction=0.30) for prior in selected_spans):
                selected_spans.append(span)
                selected_roles.setdefault(span.span_id, role)
                selected_terms.update(etp._content_tokens(span.text))
        remaining = [p for p in remaining if p is not path]
        if etp.estimate_token_count("\n\n".join([prefix, *pieces])) >= budget * 0.95:
            break
    ordered = [(span, selected_roles.get(span.span_id, "support")) for span in selected_spans]
    if not ordered:
        ranked = sorted(candidates, key=lambda s: (-s.final_score, s.source_order_index))
        ordered = [(span, "support") for span in etp._dedupe_candidates(ranked)]
    audit = {
        "path_count": len(selected_paths),
        "query_link_values": [path.query_link for path in selected_paths],
        "bridge_link_values": [path.bridge_link for path in selected_paths],
        "cross_page_path_count": sum(path.span_i.page_id != path.span_j.page_id for path in selected_paths),
        "same_page_path_count": sum(path.span_i.page_id == path.span_j.page_id for path in selected_paths),
        "custom_max_paths": max_paths,
        "custom_redundancy_penalty": redundancy_penalty,
    }
    return ordered, audit


def pack_custom(
    sample: Mapping[str, Any],
    budget: int,
    neighbor: int,
    span: int,
    redundancy_penalty: float,
    max_paths: int,
    page_size: int = DEFAULT_PAGE_SIZE,
) -> Tuple[Any, Dict[str, Any]]:
    stride = max(16, span // 2)
    candidates, page_map = etp.build_candidate_spans(
        str(sample["sample_id"]),
        str(sample["task"]),
        str(sample["question"]),
        str(sample["context_text"]),
        page_size_tokens=page_size,
        span_size_tokens=span,
        stride_tokens=stride,
    )
    lexical_candidates = etp.lexical_scored_candidates(str(sample["question"]), candidates)
    ordered, path_audit = select_bridge_paths_custom(
        str(sample["question"]),
        lexical_candidates,
        page_map,
        budget,
        neighbor_size=neighbor,
        max_paths=max_paths,
        redundancy_penalty=redundancy_penalty,
    )
    packed, pack_audit = _pack_ordered(
        str(sample["sample_id"]),
        str(sample["task"]),
        str(sample["question"]),
        "lexical_bridgepack_path_order_dev_sensitivity",
        budget,
        ordered,
        page_map,
        neighbor,
        "lexical_bridge_path_order",
        used_bridge_path=True,
    )
    pack_audit.update(path_audit)
    pack_audit["page_size_tokens"] = page_size
    pack_audit["span_size_tokens"] = span
    pack_audit["stride_tokens"] = stride
    pack_audit["candidate_count"] = len(candidates)
    return packed, pack_audit


def score_supporting_facts(sample: Mapping[str, Any], packed_text: str) -> Tuple[float, float, List[bool]]:
    facts = sample.get("supporting_facts") or []
    if not facts:
        return float("nan"), float("nan"), []
    hits = []
    for fact in facts:
        title = str(fact.get("title") or "")
        if fact.get("paragraph_text"):
            ok = _fact_in_packed(packed_text, str(fact.get("paragraph_text") or ""), title)
        else:
            sent = _fact_sentence(sample.get("orig_context"), title, int(fact.get("sent_id", 0)))
            ok = _fact_in_packed(packed_text, sent, title)
        hits.append(bool(ok))
    sf_recall = sum(hits) / max(1, len(hits))
    complete = 1.0 if hits and all(hits) else 0.0
    return sf_recall, complete, hits


def run_study(samples: Sequence[Mapping[str, Any]], budget: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    detail_rows: List[Dict[str, Any]] = []
    by_config: Dict[Tuple[int, int, float, int], List[Dict[str, Any]]] = defaultdict(list)
    for neighbor in NEIGHBORS:
        for span in SPANS:
            for redundancy_penalty in REDUNDANCY_RANGE:
                for max_paths in MAX_PATHS_RANGE:
                    for sample in samples:
                        packed, audit = pack_custom(
                            sample,
                            budget=budget,
                            neighbor=neighbor,
                            span=span,
                            redundancy_penalty=redundancy_penalty,
                            max_paths=max_paths,
                        )
                        answer_recall = 1.0 if _answer_in_text(sample.get("answers") or [], packed.packed_text) else 0.0
                        sf_recall, complete_chain, hits = score_supporting_facts(sample, packed.packed_text)
                        row = {
                            "task": sample["task"],
                            "sample_id": sample["sample_id"],
                            "neighbor": neighbor,
                            "span": span,
                            "redundancy_penalty": redundancy_penalty,
                            "max_paths": max_paths,
                            "budget": budget,
                            "packed_tokens": packed.packed_token_count,
                            "source_tokens": packed.source_token_count,
                            "compression_ratio": packed.compression_ratio,
                            "answer_bearing_recall": answer_recall,
                            "supporting_fact_recall": sf_recall,
                            "complete_chain_recall": complete_chain,
                            "n_supporting_facts": len(sample.get("supporting_facts") or []),
                            "n_fact_hits": sum(hits),
                            "n_selected_spans": len(packed.selected_span_ids),
                            "path_count": audit.get("path_count", 0),
                            "cross_page_path_count": audit.get("cross_page_path_count", 0),
                            "same_page_path_count": audit.get("same_page_path_count", 0),
                            "candidate_count": audit.get("candidate_count", 0),
                            "stride_tokens": audit.get("stride_tokens", 0),
                        }
                        detail_rows.append(row)
                        by_config[(neighbor, span, redundancy_penalty, max_paths)].append(row)

    agg_rows: List[Dict[str, Any]] = []
    default_key = (DEFAULT_NEIGHBOR, DEFAULT_SPAN, DEFAULT_REDUNDANCY, DEFAULT_MAX_PATHS)
    default_rows = by_config[default_key]
    default_macro = {
        "answer": _mean(float(r["answer_bearing_recall"]) for r in default_rows),
        "sf": _mean(float(r["supporting_fact_recall"]) for r in default_rows),
        "complete": _mean(float(r["complete_chain_recall"]) for r in default_rows),
        "tokens": _mean(float(r["packed_tokens"]) for r in default_rows),
    }

    for config, rows in sorted(by_config.items()):
        neighbor, span, redundancy_penalty, max_paths = config
        scopes = [("MACRO", rows)] + [
            (task, [r for r in rows if r["task"] == task])
            for task in TASKS
        ]
        for scope, subset in scopes:
            if not subset:
                continue
            answer = _mean(float(r["answer_bearing_recall"]) for r in subset)
            sf = _mean(float(r["supporting_fact_recall"]) for r in subset)
            complete = _mean(float(r["complete_chain_recall"]) for r in subset)
            tokens = _mean(float(r["packed_tokens"]) for r in subset)
            agg_rows.append(
                {
                    "scope": scope,
                    "neighbor": neighbor,
                    "span": span,
                    "redundancy_penalty": redundancy_penalty,
                    "max_paths": max_paths,
                    "budget": budget,
                    "n_samples": len(subset),
                    "mean_packed_tokens": tokens,
                    "answer_bearing_recall": answer,
                    "supporting_fact_recall": sf,
                    "complete_chain_recall": complete,
                    "mean_selected_spans": _mean(float(r["n_selected_spans"]) for r in subset),
                    "mean_path_count": _mean(float(r["path_count"]) for r in subset),
                    "mean_cross_page_paths": _mean(float(r["cross_page_path_count"]) for r in subset),
                    "is_default": int(config == default_key),
                    "delta_vs_default_answer": answer - default_macro["answer"] if scope == "MACRO" else float("nan"),
                    "delta_vs_default_sf": sf - default_macro["sf"] if scope == "MACRO" else float("nan"),
                    "delta_vs_default_complete": complete - default_macro["complete"] if scope == "MACRO" else float("nan"),
                    "delta_vs_default_tokens": tokens - default_macro["tokens"] if scope == "MACRO" else float("nan"),
                }
            )
    agg_rows.sort(
        key=lambda r: (
            0 if r["scope"] == "MACRO" else 1,
            -float(r["complete_chain_recall"]),
            -float(r["supporting_fact_recall"]),
            -float(r["answer_bearing_recall"]),
            float(r["mean_packed_tokens"]),
        )
    )
    return detail_rows, agg_rows


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def summarize_dimension(
    macro_rows: Sequence[Mapping[str, Any]],
    key: str,
) -> Dict[str, Dict[str, float]]:
    bucket: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in macro_rows:
        bucket[str(row[key])].append(row)
    out = {}
    for value, rows in sorted(bucket.items()):
        out[value] = {
            "complete_chain_recall": _mean(float(r["complete_chain_recall"]) for r in rows),
            "supporting_fact_recall": _mean(float(r["supporting_fact_recall"]) for r in rows),
            "answer_bearing_recall": _mean(float(r["answer_bearing_recall"]) for r in rows),
            "mean_packed_tokens": _mean(float(r["mean_packed_tokens"]) for r in rows),
        }
    return out


def build_report(
    out_dir: Path,
    samples: Sequence[Mapping[str, Any]],
    agg_rows: Sequence[Mapping[str, Any]],
    detail_rows: Sequence[Mapping[str, Any]],
    budget: int,
) -> str:
    macro_rows = [r for r in agg_rows if r["scope"] == "MACRO"]
    default_row = next(r for r in macro_rows if int(r["is_default"]) == 1)
    best_row = max(
        macro_rows,
        key=lambda r: (
            float(r["complete_chain_recall"]),
            float(r["supporting_fact_recall"]),
            float(r["answer_bearing_recall"]),
            -float(r["mean_packed_tokens"]),
        ),
    )
    complete_vals = [float(r["complete_chain_recall"]) for r in macro_rows]
    sf_vals = [float(r["supporting_fact_recall"]) for r in macro_rows]
    top5_cut = sorted(complete_vals, reverse=True)[:5]
    top5_min = min(top5_cut) if top5_cut else float("nan")
    default_is_top5 = float(default_row["complete_chain_recall"]) >= top5_min if top5_cut else False
    neighbor_summary = summarize_dimension(macro_rows, "neighbor")
    span_summary = summarize_dimension(macro_rows, "span")
    red_summary = summarize_dimension(macro_rows, "redundancy_penalty")
    max_path_summary = summarize_dimension(macro_rows, "max_paths")
    per_task_counts = {task: sum(1 for s in samples if s["task"] == task) for task in TASKS}
    sample_source_tokens = [int(row["source_tokens"]) for row in detail_rows[: len(samples)]]
    report = []
    report.append("# BridgePack Independent-Dev Parameter Sensitivity")
    report.append("")
    report.append("## Setup")
    report.append("")
    report.append(f"- Source-dev only: `hotpotqa`, `2wikimqa`, `musique`.")
    report.append(f"- Deterministic subset: `{len(samples)}` examples total, budget `{budget}`.")
    report.append(
        f"- Per-task counts: hotpotqa={per_task_counts['hotpotqa']}, "
        f"2wikimqa={per_task_counts['2wikimqa']}, musique={per_task_counts['musique']}."
    )
    report.append(
        "- Sweep grid: "
        f"neighbor={list(NEIGHBORS)}, span={list(SPANS)}, "
        f"redundancy_penalty={list(REDUNDANCY_RANGE)}, max_paths={list(MAX_PATHS_RANGE)}."
    )
    report.append(
        f"- Default under test: neighbor={DEFAULT_NEIGHBOR}, span={DEFAULT_SPAN}, "
        f"redundancy_penalty={DEFAULT_REDUNDANCY}, max_paths={DEFAULT_MAX_PATHS}."
    )
    report.append(
        f"- QA: not included; this artifact is evidence-only and CPU-only. "
        f"Mean source tokens in sampled contexts: {_format_float(_mean(sample_source_tokens), 1)}."
    )
    report.append("")
    report.append("## Main Result")
    report.append("")
    report.append(
        f"- Best macro config: neighbor={best_row['neighbor']}, span={best_row['span']}, "
        f"redundancy_penalty={best_row['redundancy_penalty']}, max_paths={best_row['max_paths']} "
        f"with complete_chain={_format_float(float(best_row['complete_chain_recall']))}, "
        f"sf={_format_float(float(best_row['supporting_fact_recall']))}, "
        f"answer={_format_float(float(best_row['answer_bearing_recall']))}, "
        f"mean_tokens={_format_float(float(best_row['mean_packed_tokens']), 1)}."
    )
    report.append(
        f"- Default macro config: complete_chain={_format_float(float(default_row['complete_chain_recall']))}, "
        f"sf={_format_float(float(default_row['supporting_fact_recall']))}, "
        f"answer={_format_float(float(default_row['answer_bearing_recall']))}, "
        f"mean_tokens={_format_float(float(default_row['mean_packed_tokens']), 1)}."
    )
    report.append(
        f"- Default deltas vs best: complete_chain={_format_float(float(default_row['complete_chain_recall']) - float(best_row['complete_chain_recall']))}, "
        f"sf={_format_float(float(default_row['supporting_fact_recall']) - float(best_row['supporting_fact_recall']))}, "
        f"answer={_format_float(float(default_row['answer_bearing_recall']) - float(best_row['answer_bearing_recall']))}."
    )
    report.append(
        f"- Default in top-5 by complete-chain recall: {'YES' if default_is_top5 else 'NO'}."
    )
    report.append("")
    report.append("## Dimension Averages")
    report.append("")
    report.append("### Neighbor")
    report.append("")
    for value, stats in neighbor_summary.items():
        report.append(
            f"- {value}: complete_chain={_format_float(stats['complete_chain_recall'])}, "
            f"sf={_format_float(stats['supporting_fact_recall'])}, "
            f"answer={_format_float(stats['answer_bearing_recall'])}, "
            f"mean_tokens={_format_float(stats['mean_packed_tokens'], 1)}"
        )
    report.append("")
    report.append("### Span")
    report.append("")
    for value, stats in span_summary.items():
        report.append(
            f"- {value}: complete_chain={_format_float(stats['complete_chain_recall'])}, "
            f"sf={_format_float(stats['supporting_fact_recall'])}, "
            f"answer={_format_float(stats['answer_bearing_recall'])}, "
            f"mean_tokens={_format_float(stats['mean_packed_tokens'], 1)}"
        )
    report.append("")
    report.append("### Redundancy Penalty")
    report.append("")
    for value, stats in red_summary.items():
        report.append(
            f"- {value}: complete_chain={_format_float(stats['complete_chain_recall'])}, "
            f"sf={_format_float(stats['supporting_fact_recall'])}, "
            f"answer={_format_float(stats['answer_bearing_recall'])}, "
            f"mean_tokens={_format_float(stats['mean_packed_tokens'], 1)}"
        )
    report.append("")
    report.append("### Max Paths")
    report.append("")
    for value, stats in max_path_summary.items():
        report.append(
            f"- {value}: complete_chain={_format_float(stats['complete_chain_recall'])}, "
            f"sf={_format_float(stats['supporting_fact_recall'])}, "
            f"answer={_format_float(stats['answer_bearing_recall'])}, "
            f"mean_tokens={_format_float(stats['mean_packed_tokens'], 1)}"
        )
    report.append("")
    report.append("## Interpretation")
    report.append("")
    if default_is_top5:
        report.append(
            "- The default setting sits on or near the observed plateau on independent dev, "
            "which argues against an obvious story that the shipped default was sharply overfit to LongBench test rows."
        )
    else:
        report.append(
            "- The default setting is not among the top macro dev settings here, so this artifact does not support a claim that the default is uniquely justified by independent dev evidence."
        )
    spread_complete = max(complete_vals) - min(complete_vals) if complete_vals else float("nan")
    spread_sf = max(sf_vals) - min(sf_vals) if sf_vals else float("nan")
    report.append(
        f"- Observed macro spread across the full grid: complete_chain={_format_float(spread_complete)}, "
        f"supporting_fact={_format_float(spread_sf)}."
    )
    report.append(
        "- Because this is evidence-only, the study establishes sensitivity of packing/evidence retention, not final QA answer quality."
    )
    report.append("")
    report.append("## Files")
    report.append("")
    report.append("- `PARAM_SENSITIVITY_DEV.csv`: aggregate config/task metrics.")
    report.append("- `PARAM_SENSITIVITY_DEV_DETAIL.csv`: per-sample per-config metrics.")
    report.append("- `PARAM_SENSITIVITY_DEV.md`: this report.")
    report.append("- `PARAM_SENSITIVITY_DEV_SAMPLE_IDS.json`: deterministic dev subset.")
    report.append("- `PARAM_SENSITIVITY_DEV_SETTINGS.json`: run configuration and sweep values.")
    return "\n".join(report) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run independent-dev BridgePack parameter sensitivity study.")
    parser.add_argument("--out_dir", type=Path, default=OUT_ROOT)
    parser.add_argument("--cache_root", type=Path, default=CACHE_ROOT)
    parser.add_argument("--budget", type=int, default=1024)
    parser.add_argument("--per_task", type=int, default=20)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows_by_task = load_source_dev(args.cache_root)
    samples = select_subset(rows_by_task, per_task=args.per_task)
    detail_rows, agg_rows = run_study(samples, budget=args.budget)

    write_csv(args.out_dir / "PARAM_SENSITIVITY_DEV.csv", agg_rows)
    write_csv(args.out_dir / "PARAM_SENSITIVITY_DEV_DETAIL.csv", detail_rows)

    sample_manifest = [
        {
            "task": s["task"],
            "sample_id": s["sample_id"],
            "question": s["question"],
            "answers": s["answers"],
            "n_supporting_facts": len(s.get("supporting_facts") or []),
            "deterministic_key": s["deterministic_key"],
        }
        for s in samples
    ]
    (args.out_dir / "PARAM_SENSITIVITY_DEV_SAMPLE_IDS.json").write_text(
        json.dumps(sample_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    settings = {
        "budget": args.budget,
        "per_task": args.per_task,
        "tasks": list(TASKS),
        "page_size_tokens": DEFAULT_PAGE_SIZE,
        "neighbors": list(NEIGHBORS),
        "spans": list(SPANS),
        "redundancy_penalties": list(REDUNDANCY_RANGE),
        "max_paths_values": list(MAX_PATHS_RANGE),
        "default": {
            "neighbor": DEFAULT_NEIGHBOR,
            "span": DEFAULT_SPAN,
            "redundancy_penalty": DEFAULT_REDUNDANCY,
            "max_paths": DEFAULT_MAX_PATHS,
        },
        "qa_included": False,
        "source_cache_root": str(args.cache_root),
    }
    (args.out_dir / "PARAM_SENSITIVITY_DEV_SETTINGS.json").write_text(
        json.dumps(settings, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    report = build_report(args.out_dir, samples, agg_rows, detail_rows, budget=args.budget)
    (args.out_dir / "PARAM_SENSITIVITY_DEV.md").write_text(report, encoding="utf-8")
    print(f"Wrote outputs to {args.out_dir}")


if __name__ == "__main__":
    main()
