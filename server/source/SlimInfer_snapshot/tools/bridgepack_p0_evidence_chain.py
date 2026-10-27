#!/usr/bin/env python3
"""Wave2: evidence-chain metrics with optional original supporting_facts join.

Joins LongBench `_id` samples to HotpotQA / 2WikiMQA / MuSiQue annotations via
normalized question text (LongBench `_id` is not the original dataset id).
Downloads are read-only into `<out_dir>/cache/supporting_facts/`.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import string
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from tools.bridgepack_g1.common import TASKS, utc_now, write_json, write_text
from tools.bridgepack_g1.stats import score_prediction_row

LONGBENCH_ROOT = Path("/home/yu-bai/Desktop/jiaqizhang/HeadKV/data/LongBench")
MULTIHOP = ("hotpotqa", "2wikimqa", "musique")
G1 = REPO_ROOT / "results" / "bridgepack_g1_qwen1024_matched_20260727_004934"
STAGE2 = REPO_ROOT / "results" / "bridgepack_completion_pipeline_20260727_140453" / "stage2"
HIST = REPO_ROOT / "results" / "bridgepack_exact_qwen_budget_1024_20260619_181059"
BGE_ROOT = REPO_ROOT / "results" / "dense_retriever_reranker_baselines_20260623_101902"
BUDGET = 1024


def _norm(text: str) -> str:
    text = str(text).lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    return " ".join(text.split())


def _answer_in_text(answers: Sequence[Any], text: str) -> bool:
    hay = _norm(text)
    for a in answers:
        needle = _norm(str(a))
        if needle and needle in hay:
            return True
    return False


def _dup_token_ratio(text: str) -> float:
    toks = _norm(text).split()
    if not toks:
        return 0.0
    c = Counter(toks)
    return sum(v for v in c.values() if v > 1) / max(1, len(toks))


def _sent_split(text: str) -> List[str]:
    # Lightweight sentence split for supporting-fact sentence lookup.
    parts = re.split(r"(?<=[.!?])\s+", str(text).strip())
    return [p.strip() for p in parts if p.strip()]


def _title_block_map(context: str) -> Dict[str, str]:
    """Parse LongBench 'Passage i: Title\\nbody' blocks when possible."""
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
    # Original Hotpot/2Wiki context may be list[[title, [sents...]], ...] or dict(title->sents)
    if isinstance(context, list):
        for item in context:
            if isinstance(item, (list, tuple)) and len(item) >= 2 and str(item[0]) == title:
                sents = item[1] if isinstance(item[1], list) else _sent_split(str(item[1]))
                if 0 <= sent_id < len(sents):
                    return str(sents[sent_id])
        # Hotpot parquet often stores dict-like via pandas; handle list of dicts
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
    # Soft fallback: title mention + substantial token overlap
    if title and _norm(title) in _norm(packed) and fact_text:
        ft = set(_norm(fact_text).split())
        pt = set(_norm(packed).split())
        if ft and len(ft & pt) / max(1, len(ft)) >= 0.6:
            return True
    return False


def download_supporting_facts(cache_dir: Path) -> Dict[str, Any]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    from huggingface_hub import hf_hub_download

    meta: Dict[str, Any] = {"ok": True, "tasks": {}}

    # HotpotQA distractor validation parquet
    hotpot_path = Path(
        hf_hub_download(
            repo_id="hotpotqa/hotpot_qa",
            filename="distractor/validation-00000-of-00001.parquet",
            repo_type="dataset",
            local_dir=str(cache_dir / "hotpot"),
        )
    )
    import pandas as pd

    hdf = pd.read_parquet(hotpot_path)
    hotpot_by_q: Dict[str, Dict[str, Any]] = {}
    for _, row in hdf.iterrows():
        sf = row["supporting_facts"]
        titles = list(sf["title"]) if isinstance(sf, dict) else []
        sids = list(sf["sent_id"]) if isinstance(sf, dict) else []
        facts = [{"title": str(t), "sent_id": int(s)} for t, s in zip(titles, sids)]
        hotpot_by_q[_norm(row["question"])] = {
            "orig_id": str(row["id"]),
            "answer": row["answer"],
            "supporting_facts": facts,
            "type": str(row.get("type", "")),
            "orig_context": row.get("context"),
        }
    meta["tasks"]["hotpotqa"] = {"n_source": len(hotpot_by_q), "path": str(hotpot_path)}

    # 2Wiki
    wiki_path = Path(
        hf_hub_download(
            repo_id="voidful/2WikiMultihopQA",
            filename="dev.json",
            repo_type="dataset",
            local_dir=str(cache_dir / "2wiki"),
        )
    )
    wiki_raw = json.loads(wiki_path.read_text(encoding="utf-8"))
    wiki_by_q: Dict[str, Dict[str, Any]] = {}
    for ex in wiki_raw:
        # supporting_facts format varies: list of [title, sent_id] or dict
        sf_raw = ex.get("supporting_facts") or ex.get("evidences") or []
        facts = []
        if isinstance(sf_raw, dict) and "title" in sf_raw:
            facts = [{"title": str(t), "sent_id": int(s)} for t, s in zip(sf_raw["title"], sf_raw["sent_id"])]
        else:
            for item in sf_raw:
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    facts.append({"title": str(item[0]), "sent_id": int(item[1])})
                elif isinstance(item, dict):
                    facts.append({"title": str(item.get("title")), "sent_id": int(item.get("sent_id", 0))})
        wiki_by_q[_norm(ex.get("question", ""))] = {
            "orig_id": str(ex.get("_id") or ex.get("id") or ""),
            "answer": ex.get("answer"),
            "supporting_facts": facts,
            "orig_context": ex.get("context"),
        }
    meta["tasks"]["2wikimqa"] = {"n_source": len(wiki_by_q), "path": str(wiki_path)}

    # MuSiQue
    mus_path = Path(
        hf_hub_download(
            repo_id="dgslibisey/MuSiQue",
            filename="musique_ans_v1.0_dev.jsonl",
            repo_type="dataset",
            local_dir=str(cache_dir / "musique"),
        )
    )
    mus_by_q: Dict[str, Dict[str, Any]] = {}
    for line in mus_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        ex = json.loads(line)
        # supporting paragraphs / question_decomposition
        facts = []
        for idx, para in enumerate(ex.get("paragraphs") or []):
            if para.get("is_supporting"):
                facts.append(
                    {
                        "title": str(para.get("title") or f"para_{idx}"),
                        "sent_id": 0,
                        "paragraph_text": str(para.get("paragraph_text") or para.get("text") or ""),
                        "hop": int(para.get("idx", idx)),
                    }
                )
        # Prefer decomposition order for hop1/hop2 if present
        decomp = ex.get("question_decomposition") or []
        hop_facts = []
        paras = ex.get("paragraphs") or []
        for hi, step in enumerate(decomp):
            sp = step.get("paragraph_support_idx")
            if sp is None:
                idxs = step.get("support_idxs") or step.get("paragraph_support_idxs") or []
            else:
                idxs = [sp]
            for sp_i in idxs:
                if 0 <= int(sp_i) < len(paras):
                    p = paras[int(sp_i)]
                    hop_facts.append(
                        {
                            "title": str(p.get("title") or f"para_{sp_i}"),
                            "sent_id": 0,
                            "paragraph_text": str(p.get("paragraph_text") or p.get("text") or ""),
                            "hop": hi + 1,
                        }
                    )
        mus_by_q[_norm(ex.get("question", ""))] = {
            "orig_id": str(ex.get("id") or ""),
            "answer": ex.get("answer"),
            "supporting_facts": hop_facts or facts,
            "orig_context": paras,
        }
    meta["tasks"]["musique"] = {"n_source": len(mus_by_q), "path": str(mus_path)}

    return {
        "meta": meta,
        "by_task_q": {"hotpotqa": hotpot_by_q, "2wikimqa": wiki_by_q, "musique": mus_by_q},
    }


def align_longbench(sf_db: Mapping[str, Mapping[str, Any]]) -> Tuple[Dict[Tuple[str, str], Dict[str, Any]], Dict[str, Any]]:
    aligned: Dict[Tuple[str, str], Dict[str, Any]] = {}
    stats = {}
    for task in MULTIHOP:
        path = LONGBENCH_ROOT / f"{task}.jsonl"
        rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
        hit = 0
        for r in rows:
            q = _norm(r.get("input") or r.get("question") or "")
            sid = str(r["_id"])
            src = sf_db.get(task, {}).get(q)
            if src:
                hit += 1
                aligned[(task, sid)] = {
                    "longbench_id": sid,
                    "question": r.get("input"),
                    "answers": r.get("answers"),
                    "context": r.get("context"),
                    **src,
                }
        stats[task] = {"n_longbench": len(rows), "n_aligned": hit, "align_rate": hit / max(1, len(rows))}
    return aligned, stats


def method_sources(out_dir: Path) -> Dict[str, Dict[str, Path]]:
    """packed + pred roots for each method (Qwen@1024)."""
    src_order_pack = G1 / "full" / "packed_prompts_build" / "packed_prompts" / "same_spans_source_order" / f"budget_{BUDGET}"
    src_order_pred = G1 / "full" / "predictions" / "same_spans_source_order"
    no_neighbor_pack = out_dir / "qwen" / "full" / "packed_prompts_build" / "packed_prompts" / "no_neighbor_source_order" / f"budget_{BUDGET}"
    no_neighbor_pred = out_dir / "qwen" / "full" / "predictions" / "no_neighbor_source_order"
    return {
        "bm25_chunk_source_order": {
            "pack": HIST / "packed_prompts_build" / "packed_prompts" / "bm25_chunk_source_order" / f"budget_{BUDGET}",
            "pred": HIST / "predictions" / "bm25_chunk_source_order",
        },
        "bge_chunk_packing": {
            "pack": BGE_ROOT / "03_dense_retriever_packing" / "bge_chunk_packing",
            "pred": BGE_ROOT / "06_qwen_eval" / "bge_chunk_packing" / "predictions" / "bge_chunk_packing",
        },
        "lexical_bridgepack_path_order": {
            "pack": HIST / "packed_prompts_build" / "packed_prompts" / "lexical_bridgepack_path_order" / f"budget_{BUDGET}",
            "pred": HIST / "predictions" / "lexical_bridgepack_path_order",
        },
        "same_spans_source_order": {"pack": src_order_pack, "pred": src_order_pred},
        "no_neighbor_source_order": {"pack": no_neighbor_pack, "pred": no_neighbor_pred},
        "exact_budget_adapted_qwen": {
            "pack": STAGE2
            / "audit_full"
            / "packed_prompts_build"
            / "packed_prompts"
            / "exact_budget_adapted_qwen"
            / f"budget_{BUDGET}",
            "pred": STAGE2 / "qa_full_qwen" / "predictions" / "exact_budget_adapted_qwen",
        },
    }


def _load_pack_row(pack_root: Path, task: str, sid: str, method: str) -> Optional[Dict[str, Any]]:
    # Standard per-task jsonl
    p = pack_root / f"{task}.jsonl"
    if p.exists():
        for row in (json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()):
            if str(row.get("sample_id")) == sid:
                return row
    # BGE unified file variants
    for cand in [
        pack_root / "full" / "qwen_packed_prompts.jsonl",
        pack_root / "qwen_packed_prompts.jsonl",
        pack_root / f"budget_{BUDGET}" / f"{task}.jsonl",
    ]:
        if not cand.exists():
            continue
        for row in (json.loads(l) for l in cand.read_text(encoding="utf-8").splitlines() if l.strip()):
            if str(row.get("sample_id")) == sid and str(row.get("task", task)) in {task, ""}:
                return row
            if str(row.get("sample_id")) == sid and str(row.get("task")) == task:
                return row
    return None


def _load_pred(pred_root: Path, task: str, sid: str) -> Optional[Dict[str, Any]]:
    p = pred_root / f"{task}.jsonl"
    if not p.exists():
        return None
    for row in (json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()):
        if str(row.get("sample_id")) == sid:
            return row
    return None


def compute_metrics(out_dir: Path) -> Dict[str, Any]:
    cache = out_dir / "cache" / "supporting_facts"
    try:
        bundle = download_supporting_facts(cache)
        aligned, align_stats = align_longbench(bundle["by_task_q"])
        sf_available = all(align_stats[t]["align_rate"] >= 0.9 for t in MULTIHOP)
        align_ok = True
    except Exception as exc:
        write_text(out_dir / "EVIDENCE_ALIGN_ERROR.md", f"{exc}\n")
        aligned, align_stats = {}, {t: {"n_longbench": 0, "n_aligned": 0, "align_rate": 0.0, "error": str(exc)} for t in MULTIHOP}
        sf_available = False
        align_ok = False
        bundle = {"meta": {"ok": False, "error": str(exc)}}

    write_json(out_dir / "EVIDENCE_ALIGN_STATS.json", {"align_ok": align_ok, "sf_available": sf_available, "stats": align_stats, "meta": bundle.get("meta")})

    methods = method_sources(out_dir)
    rows: List[Dict[str, Any]] = []
    for method, paths in methods.items():
        pack_root = paths["pack"]
        pred_root = paths["pred"]
        if not pack_root.exists() and method == "no_neighbor_source_order":
            # Allow missing until wave1 packs exist — still write NA rows later via skip
            continue
        for task in MULTIHOP:
            lb_path = LONGBENCH_ROOT / f"{task}.jsonl"
            lb_rows = [json.loads(l) for l in lb_path.read_text(encoding="utf-8").splitlines() if l.strip()]
            for lb in lb_rows:
                sid = str(lb["_id"])
                pack = _load_pack_row(pack_root, task, sid, method)
                if not pack:
                    continue
                packed_text = str(pack.get("packed_text") or pack.get("context") or "")
                answers = lb.get("answers") or pack.get("answers") or []
                pred = _load_pred(pred_root, task, sid)
                qa_f1 = float("nan")
                if pred is not None:
                    qa_f1 = score_prediction_row(task, pred)

                ans_recall = 1.0 if _answer_in_text(answers, packed_text) else 0.0
                # answer-bearing span recall: same as answer-string containment in packed text for this protocol
                ans_bearing = ans_recall

                sf_recall = hop1 = hop2 = complete = any_missing = "NA"
                sf_ann = "absent_or_unaligned"
                gold = aligned.get((task, sid))
                if sf_available and gold and gold.get("supporting_facts"):
                    sf_ann = "aligned_original"
                    facts = gold["supporting_facts"]
                    hits = []
                    for fi, fact in enumerate(facts):
                        title = str(fact.get("title") or "")
                        if fact.get("paragraph_text"):
                            ok = _norm(fact["paragraph_text"])[:80] in _norm(packed_text) or _fact_in_packed(
                                packed_text, fact.get("paragraph_text"), title
                            )
                        else:
                            ctx = gold.get("orig_context")
                            if ctx is None:
                                ctx = gold.get("context") or lb.get("context") or ""
                            sent = _fact_sentence(ctx, title, int(fact.get("sent_id", 0)))
                            ok = _fact_in_packed(packed_text, sent, title)
                        hits.append(bool(ok))
                    sf_recall = sum(hits) / max(1, len(hits))
                    # hop assignment: first half / second half if no explicit hop; else hop field
                    hop_ids = [int(f.get("hop") or 0) for f in facts]
                    if any(h > 0 for h in hop_ids):
                        h1 = [hits[i] for i, h in enumerate(hop_ids) if h == 1]
                        h2 = [hits[i] for i, h in enumerate(hop_ids) if h == 2]
                    else:
                        mid = max(1, len(hits) // 2)
                        h1, h2 = hits[:mid], hits[mid:]
                    hop1 = (sum(h1) / len(h1)) if h1 else "NA"
                    hop2 = (sum(h2) / len(h2)) if h2 else "NA"
                    complete = 1.0 if hits and all(hits) else 0.0
                    any_missing = 1.0 if (not hits or not all(hits)) else 0.0

                density = ans_bearing  # proxy: answer present => density 1 for answer evidence
                rows.append(
                    {
                        "model": "qwen",
                        "budget": BUDGET,
                        "method": method,
                        "task": task,
                        "sample_id": sid,
                        "answer_string_recall": ans_recall,
                        "answer_bearing_span_recall": ans_bearing,
                        "supporting_fact_recall": sf_recall,
                        "hop1_evidence_recall": hop1,
                        "hop2_evidence_recall": hop2,
                        "complete_chain_recall": complete,
                        "any_hop_missing_rate": any_missing,
                        "effective_evidence_density": density,
                        "duplicate_token_ratio": _dup_token_ratio(packed_text),
                        "qa_f1": qa_f1,
                        "supporting_fact_annotation": sf_ann,
                    }
                )

    # write CSV
    fields = [
        "model",
        "budget",
        "method",
        "task",
        "sample_id",
        "answer_string_recall",
        "answer_bearing_span_recall",
        "supporting_fact_recall",
        "hop1_evidence_recall",
        "hop2_evidence_recall",
        "complete_chain_recall",
        "any_hop_missing_rate",
        "effective_evidence_density",
        "duplicate_token_ratio",
        "qa_f1",
        "supporting_fact_annotation",
    ]
    with (out_dir / "EVIDENCE_CHAIN_METRICS.csv").open("w", encoding="utf-8", newline="") as handle:
        w = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        for row in rows:
            w.writerow(row)

    # summaries + correlation
    summary = []
    for method in sorted({r["method"] for r in rows}):
        for task in MULTIHOP + ("MACRO_MULTIHOP",):
            subset = [r for r in rows if r["method"] == method and (task == "MACRO_MULTIHOP" or r["task"] == task)]
            if not subset:
                continue
            def mean_num(key):
                vals = [float(r[key]) for r in subset if r[key] != "NA" and not (isinstance(r[key], float) and math.isnan(r[key]))]
                return sum(vals) / len(vals) if vals else float("nan")

            # correlation answer-bearing vs qa_f1
            pairs = [(float(r["answer_bearing_span_recall"]), float(r["qa_f1"])) for r in subset if r["qa_f1"] == r["qa_f1"]]
            corr = float("nan")
            if len(pairs) >= 5:
                xs = [p[0] for p in pairs]
                ys = [p[1] for p in pairs]
                mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
                num = sum((x - mx) * (y - my) for x, y in pairs)
                den = math.sqrt(sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys))
                corr = num / den if den > 1e-12 else float("nan")
            summary.append(
                {
                    "method": method,
                    "task": task,
                    "n": len(subset),
                    "answer_bearing_mean": mean_num("answer_bearing_span_recall"),
                    "supporting_fact_mean": mean_num("supporting_fact_recall"),
                    "complete_chain_mean": mean_num("complete_chain_recall"),
                    "dup_token_mean": mean_num("duplicate_token_ratio"),
                    "qa_f1_mean": mean_num("qa_f1"),
                    "corr_answer_bearing_qa_f1": corr,
                }
            )
    write_json(out_dir / "EVIDENCE_CHAIN_SUMMARY.json", summary)

    sf_note = (
        "supporting-fact 系列可用（原始标注经 question 对齐 LongBench `_id`）"
        if sf_available
        else "supporting-fact 系列为 **NA**（对齐失败或不可用）；仅报告 answer-bearing / answer-string"
    )
    report = [
        "# EVIDENCE_CHAIN_REPORT",
        "",
        f"Generated: {utc_now()}",
        "",
        "## 对齐",
        "",
        f"- align_ok: `{align_ok}`",
        f"- sf_available: `{sf_available}`",
        "",
    ]
    for task, st in align_stats.items():
        report.append(f"- `{task}`: aligned {st.get('n_aligned')}/{st.get('n_longbench')} (rate={st.get('align_rate')})")
    report += [
        "",
        "## 方法",
        "",
        "BM25 / BGE / LBP / source-order matched / no-neighbor / adapted LLMLingua（Stage2）",
        "",
        "## 指标摘要",
        "",
    ]
    for s in summary:
        if s["task"] != "MACRO_MULTIHOP":
            continue
        report.append(
            f"- `{s['method']}`: answer_bearing={s['answer_bearing_mean']:.3f}, "
            f"sf={s['supporting_fact_mean']}, complete={s['complete_chain_mean']}, "
            f"qa_f1={s['qa_f1_mean']}, corr(ans,f1)={s['corr_answer_bearing_qa_f1']}"
        )
    report += ["", f"## 说明", "", sf_note, ""]
    write_text(out_dir / "EVIDENCE_CHAIN_REPORT.md", "\n".join(report))

    guard = f"""# EVIDENCE_CLAIM_GUARD

Generated: {utc_now()}

| 声明 | 允许？ | 依据 |
|---|---|---|
| 报告 answer-bearing / answer-string recall | YES | EVIDENCE_CHAIN_METRICS.csv |
| 报告 supporting-fact / hop / complete-chain | {"YES" if sf_available else "NO — 记为 NA"} | 对齐统计见 EVIDENCE_ALIGN_STATS.json |
| 自造 gold supporting facts | **NO** | 禁止 |
| 用证据召回直接宣称端到端更优 | NO | 需配合 matched QA 表 |
| 引用 38.05 | **NO** | 禁止 |

{sf_note}
"""
    write_text(out_dir / "EVIDENCE_CLAIM_GUARD.md", guard)
    return {"n_rows": len(rows), "sf_available": sf_available, "align_stats": align_stats}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    info = compute_metrics(out_dir)
    write_json(out_dir / "EVIDENCE_CHAIN_DONE.json", {"timestamp": utc_now(), **info})
    print(json.dumps(info, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
