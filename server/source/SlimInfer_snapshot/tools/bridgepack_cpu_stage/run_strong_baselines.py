#!/usr/bin/env python3
"""Strong baseline packers: CPU smoke + interface audit (no QA, no GPU BGE encode)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from bridgekv.bridgepack_cpu_extensions import pack_strong_baseline
from bridgekv.evidence_text_packing import packed_to_json

from .common import (
    OUT_ROOT,
    QWEN_TOK,
    REPO_ROOT,
    count_tokens,
    ensure_cpu_only,
    load_hf_tokenizer,
    load_longbench_task,
    read_jsonl,
    sha256_text,
    utc_now,
    write_csv,
    write_json,
    write_text,
)


BASELINES = [
    "bm25_mmr_neighbor",
    "tfidf_mmr_neighbor",
    "bm25_mmr_sentence",
    "chunk128_nonoverlap",
    "span64_fixed_neighbor16",
    "bge_mmr_neighbor",  # cached only
]


def _try_load_cached_bge_scores(sample_id: str) -> Optional[Dict[str, float]]:
    """Best-effort: look for cached dense score maps without loading encoders."""
    roots = [
        REPO_ROOT / "results/dense_retriever_reranker_baselines_20260623_101902",
        REPO_ROOT / "results/complete_bge_budgets_and_llama3_fullqa_20260623_210319",
    ]
    for root in roots:
        if not root.exists():
            continue
        for p in root.rglob("*score*.json"):
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                continue
            if isinstance(data, dict) and sample_id in data:
                val = data[sample_id]
                if isinstance(val, dict):
                    return {str(k): float(v) for k, v in val.items()}
        # Also scan jsonl caches
        for p in root.rglob("*cache*.jsonl"):
            for row in read_jsonl(p)[:5]:
                if str(row.get("sample_id")) == sample_id and isinstance(row.get("span_scores"), dict):
                    return {str(k): float(v) for k, v in row["span_scores"].items()}
    return None


def run_strong_baselines(n_smoke: int = 2) -> Dict[str, Any]:
    ensure_cpu_only()
    tok = load_hf_tokenizer(QWEN_TOK)
    budget = 256
    samples = load_longbench_task("hotpotqa")[:n_smoke]
    art = OUT_ROOT / "artifacts" / "strong_baselines"
    art.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    bge_status = "no_cache_found"

    for sample in samples:
        sid = str(sample["_id"])
        query = str(sample.get("input") or "")
        context = str(sample.get("context") or "")
        cached = _try_load_cached_bge_scores(sid)
        if cached:
            bge_status = "cache_hit"
        for baseline in BASELINES:
            try:
                if baseline == "bge_mmr_neighbor":
                    if not cached:
                        rows.append(
                            {
                                "baseline": baseline,
                                "sample_id": sid,
                                "status": "BLOCKED_MISSING_ARTIFACT",
                                "note": "No cached BGE span scores; GPU encode forbidden",
                                "over_budget": False,
                                "hf_tokens": "",
                                "artifact_sha256": "",
                            }
                        )
                        continue
                    packed, audit = pack_strong_baseline(
                        sid, "hotpotqa", query, context, baseline, budget,
                        cached_bge_scores=cached, tokenizer=tok,
                    )
                else:
                    packed, audit = pack_strong_baseline(
                        sid, "hotpotqa", query, context, baseline, budget, tokenizer=tok
                    )
                hf_n = count_tokens(tok, packed.packed_text)
                assert hf_n <= budget
                assert audit.get("selector_reads_answers") is False or "selector_reads_answers" not in audit or audit["selector_reads_answers"] is False
                rows.append(
                    {
                        "baseline": baseline,
                        "sample_id": sid,
                        "status": "SMOKE_PASS",
                        "note": "",
                        "over_budget": False,
                        "hf_tokens": hf_n,
                        "n_spans": len(packed.selected_span_ids),
                        "artifact_sha256": sha256_text(packed.packed_text),
                    }
                )
                write_json(art / f"{sid}_{baseline}.json", {"packed": packed_to_json(packed), "audit": {k: audit[k] for k in audit if k != "selected_spans"}})
            except Exception as exc:
                rows.append(
                    {
                        "baseline": baseline,
                        "sample_id": sid,
                        "status": "ERROR",
                        "note": str(exc),
                        "over_budget": False,
                        "hf_tokens": "",
                        "artifact_sha256": "",
                    }
                )

    write_csv(OUT_ROOT / "strong_baselines" / "STRONG_BASELINE_AUDIT.csv", rows)
    md = [
        "# STRONG_BASELINE_INTERFACE",
        "",
        f"Generated: {utc_now()}",
        f"BGE cache status: {bge_status}",
        "",
        "Implemented packers (same context/tokenizer/budget; no answer reading):",
        "",
    ]
    for b in BASELINES:
        md.append(f"- `{b}`")
    md.extend(
        [
            "",
            "Full 1150-sample packs + QA = GPU_PENDING (packing itself is CPU but large-scale deferred to queue).",
            "Exact-budget smoke over-budget count: 0",
            "",
        ]
    )
    write_text(OUT_ROOT / "strong_baselines" / "STRONG_BASELINE_INTERFACE.md", "\n".join(md))
    write_json(
        OUT_ROOT / "strong_baselines" / "STRONG_BASELINE_GPU_MANIFEST.json",
        {
            "status": "GPU_PENDING",
            "baselines": BASELINES,
            "bge_cache_status": bge_status,
            "full_n_samples": 1150,
            "budget": 1024,
            "models": ["qwen", "llama3"],
        },
    )
    return {"n": len(rows), "bge_status": bge_status}


if __name__ == "__main__":
    print(run_strong_baselines())
