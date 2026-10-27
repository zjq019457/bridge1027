#!/usr/bin/env python3
"""Token utilization + packing fairness; backfill / matched-actual-token interfaces."""
from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .common import (
    FAIRNESS_DIR,
    OUT_ROOT,
    QWEN_TOK,
    REPO_ROOT,
    TASKS,
    count_tokens,
    ensure_cpu_only,
    load_hf_tokenizer,
    load_longbench_task,
    normalize_method,
    read_csv,
    read_jsonl,
    resolve_repo_path,
    sha256_text,
    utc_now,
    write_csv,
    write_json,
    write_jsonl,
    write_text,
)


def _percentile(arr: np.ndarray, q: float) -> float:
    if len(arr) == 0:
        return float("nan")
    return float(np.percentile(arr, q))


def _repetition_ratio(text: str) -> float:
    toks = str(text).lower().split()
    if not toks:
        return 0.0
    from collections import Counter

    c = Counter(toks)
    return sum(v for v in c.values() if v > 1) / max(1, len(toks))


def _overlap_span_ratio(text: str, window: int = 8) -> float:
    toks = str(text).lower().split()
    if len(toks) < window * 2:
        return 0.0
    seen = set()
    overlap = 0
    total = 0
    for i in range(0, len(toks) - window + 1, max(1, window // 2)):
        gram = tuple(toks[i : i + window])
        total += 1
        if gram in seen:
            overlap += 1
        else:
            seen.add(gram)
    return overlap / max(1, total)


def _iter_all_packed_rows() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for m in read_csv(FAIRNESS_DIR / "result_file_manifest.csv"):
        packed = m["packed_path"]
        model = m["model"]
        budget = int(m["budget"])
        method = m["method"]
        if "{task}" in packed:
            paths = [resolve_repo_path(packed.replace("{task}", t)) for t in TASKS]
        else:
            paths = [resolve_repo_path(packed)]
        for p in paths:
            if not p.exists():
                continue
            for row in read_jsonl(p):
                row = dict(row)
                row["_model"] = model
                row["_budget"] = budget
                row["_method"] = method
                rows.append(row)
    return rows


def run_token_utilization() -> Dict[str, Any]:
    ensure_cpu_only()
    rows = _iter_all_packed_rows()
    dist_rows = []
    over_budget_total = 0
    by_group: Dict[Tuple[str, int, str], List[Dict[str, Any]]] = defaultdict(list)

    for row in rows:
        # NOTE: field `exact_model_token_budget` is a boolean PASS flag, not the numeric budget.
        budget = int(row.get("_budget") or row.get("budget") or row.get("target_token") or 0)
        model_tok = row.get("model_token_count")
        packed_tok = row.get("packed_token_count") or row.get("packed_context_token_count")
        # Prefer model tokenizer counts when present
        context_tokens = None
        if model_tok is not None:
            context_tokens = int(model_tok)
        elif packed_tok is not None:
            context_tokens = int(packed_tok)
        else:
            continue
        if budget <= 0:
            continue
        # total prompt if generation_prompt present — but without re-tokenizing all (CPU cost),
        # use model_token_count as context-only under exact protocol (prompt wrapper may be outside budget).
        total_prompt = context_tokens
        unused = max(0, budget - context_tokens)
        util = context_tokens / budget if budget else float("nan")
        over = context_tokens > budget
        # Also respect explicit over_budget flag when present
        if row.get("over_budget") is True:
            over = True
        if over:
            over_budget_total += 1
        text = str(row.get("packed_text") or row.get("packed_prompt") or "")
        rec = {
            "model": row["_model"],
            "budget": budget,
            "method": row["_method"],
            "method_norm": normalize_method(row["_method"]),
            "task": row.get("task"),
            "sample_id": row.get("sample_id"),
            "context_only_tokens": context_tokens,
            "total_prompt_tokens": total_prompt,
            "unused_tokens": unused,
            "budget_utilization": util,
            "over_budget": int(over),
            "repeat_token_ratio": _repetition_ratio(text),
            "overlap_span_ratio": _overlap_span_ratio(text),
            "source_token_count": row.get("source_token_count"),
            "stored_compression_ratio": row.get("compression_ratio"),
        }
        dist_rows.append(rec)
        by_group[(row["_model"], budget, normalize_method(row["_method"]))].append(rec)

    write_csv(OUT_ROOT / "token_util" / "PACKED_TOKEN_DISTRIBUTION.csv", dist_rows)
    write_csv(OUT_ROOT / "PACKED_TOKEN_DISTRIBUTION.csv", dist_rows)

    summary_rows = []
    for (model, budget, method), items in sorted(by_group.items()):
        arr = np.array([x["context_only_tokens"] for x in items], dtype=np.float64)
        util = np.array([x["budget_utilization"] for x in items], dtype=np.float64)
        unused = np.array([x["unused_tokens"] for x in items], dtype=np.float64)
        src = np.array([float(x["source_token_count"] or 0) for x in items], dtype=np.float64)
        stored_cr = np.array([float(x["stored_compression_ratio"] or 0) for x in items], dtype=np.float64)
        recomputed_cr = arr / np.maximum(src, 1.0)
        summary_rows.append(
            {
                "model": model,
                "budget": budget,
                "method": method,
                "n": len(items),
                "mean": float(arr.mean()),
                "p50": _percentile(arr, 50),
                "p95": _percentile(arr, 95),
                "p99": _percentile(arr, 99),
                "max": float(arr.max()) if len(arr) else float("nan"),
                "mean_utilization": float(util.mean()),
                "mean_unused": float(unused.mean()),
                "over_budget_count": int(sum(x["over_budget"] for x in items)),
                "mean_repeat_token_ratio": float(np.mean([x["repeat_token_ratio"] for x in items])),
                "mean_overlap_span_ratio": float(np.mean([x["overlap_span_ratio"] for x in items])),
                "mean_stored_compression_ratio": float(stored_cr.mean()) if len(stored_cr) else float("nan"),
                "mean_recomputed_packed_over_source": float(recomputed_cr.mean()) if len(recomputed_cr) else float("nan"),
                "mean_unused_over_budget": float(unused.mean() / budget) if budget else float("nan"),
            }
        )
    write_csv(OUT_ROOT / "token_util" / "token_utilization_summary.csv", summary_rows)

    # Resolve 0.1498 vs alternative % definitions for Lexical-BridgePack Qwen 1024
    focus = [
        s
        for s in summary_rows
        if s["model"] == "qwen"
        and int(s["budget"]) == 1024
        and s["method"] == "lexical_bridgepack_path_order"
    ]
    focus_row = focus[0] if focus else {}
    lines = [
        "# TOKEN_UTILIZATION_REPORT",
        "",
        f"Generated: {utc_now()}",
        f"Packed rows scanned: {len(dist_rows)}",
        f"Over-budget total (historical packs): {over_budget_total}",
        "",
        "## Definitions (explicit denominators)",
        "",
        "1. **budget_utilization** = `context_only_tokens / budget`",
        "2. **unused_rate** = `(budget - context_only_tokens) / budget`",
        "3. **packed_over_source** = `context_only_tokens / source_token_count` (recomputed)",
        "4. **stored_compression_ratio** = artifact field `compression_ratio` (may use regex tokens at pack time)",
        "",
        "## Table 1 vs Appendix compression number inconsistency",
        "",
        "Paper overhead tables report Lexical-BridgePack Qwen-1024 mean packed≈919.88, source≈11234.61, stored ratio **0.1498**.",
        "",
        "Resolved: ratio-of-means mean(packed)/mean(source)≈0.0819; mean-of-ratios mean(packed/source)≈0.1498 (matches stored field); unused_rate≈10.17%.",
        "A draft 13.25% was not recovered as any of these; do not mix formulas.",
        f"- unused_rate: {focus_row.get('mean_unused_over_budget', 'NA')}",
        f"- stored/mean-of-ratios compression: {focus_row.get('mean_stored_compression_ratio', 'NA')}",
        f"- recomputed mean-of-ratios: {focus_row.get('mean_recomputed_packed_over_source', 'NA')}",
        "",
        "**Claim rule:** State whether a compression figure is mean-of-ratios or ratio-of-means; always name the denominator.",
        "",
        "## Per method summary (excerpt)",
        "",
        "| model | budget | method | mean | p95 | max | util | unused | over |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for s in summary_rows:
        if s["method"] in {
            "lexical_bridgepack_path_order",
            "bge_chunk_packing",
            "bm25_chunk_source_order",
            "bridge_neighbor32_path_order",
            "span_only_source_order",
        } or (s["model"] == "qwen" and int(s["budget"]) == 1024):
            lines.append(
                f"| {s['model']} | {s['budget']} | {s['method']} | {s['mean']:.2f} | {s['p95']:.2f} | {s['max']:.0f} | {s['mean_utilization']:.4f} | {s['mean_unused']:.2f} | {s['over_budget_count']} |"
            )
    write_text(OUT_ROOT / "token_util" / "TOKEN_UTILIZATION_REPORT.md", "\n".join(lines))
    write_text(OUT_ROOT / "TOKEN_UTILIZATION_REPORT.md", "\n".join(lines))

    # CPU smoke: backfill + matched-actual-token on tiny synthetic + 2 real samples
    from bridgekv.bridgepack_cpu_extensions import matched_actual_token_pack, pack_with_mode

    tok = load_hf_tokenizer(QWEN_TOK)
    smoke_dir = OUT_ROOT / "artifacts" / "packing_smoke"
    smoke_dir.mkdir(parents=True, exist_ok=True)
    hotpot = load_longbench_task("hotpotqa")[:2]
    audit_lines = [
        "# BACKFILL_BUDGET_AUDIT",
        "",
        f"Generated: {utc_now()}",
        "CPU-only smoke (regex budget inside packer; HF tokenizer audit after).",
        "",
    ]
    matched_manifest = {
        "created_at": utc_now(),
        "status": "interface_ready_smoke",
        "samples": [],
        "gpu_pending_full_run": True,
    }
    backfill_rows = []
    for sample in hotpot:
        sid = str(sample.get("_id"))
        query = str(sample.get("input") or "")
        context = str(sample.get("context") or "")
        for mode_name, stop in [("stop_at_first_nonfit", True), ("backfill", False)]:
            packed, audit = pack_with_mode(
                sid, "hotpotqa", query, context, "lexical_bridgepack_path_order", 256,
                stop_at_first_nonfit=stop,
                tokenizer=tok,
            )
            # exact tokenizer audit on packed_text (context-only)
            n_tok = count_tokens(tok, packed.packed_text)
            over = n_tok > 256
            rec = {
                "sample_id": sid,
                "mode": mode_name,
                "regex_tokens": packed.packed_token_count,
                "hf_tokens": n_tok,
                "budget": 256,
                "over_budget": over,
                "skipped_nonfit": audit.get("skipped_nonfit"),
                "n_spans": len(packed.selected_span_ids),
                "artifact_sha256": sha256_text(packed.packed_text),
            }
            backfill_rows.append(rec)
            audit_lines.append(
                f"- {sid} `{mode_name}`: hf_tokens={n_tok}/256 over={over} spans={rec['n_spans']} skipped_nonfit={rec['skipped_nonfit']}"
            )
            write_json(
                smoke_dir / f"{sid}_{mode_name}.json",
                {"packed": packed.__dict__, "audit": {k: v for k, v in audit.items() if k not in {"selected_spans", "page_map"}}},
            )
        # matched actual token vs lexical mean target 200
        packed_m, audit_m = matched_actual_token_pack(
            sid, "hotpotqa", query, context, "bm25_chunk_source_order",
            target_mean_tokens=200, tokenizer=tok,
        )
        matched_manifest["samples"].append(
            {
                "sample_id": sid,
                "method": "bm25_chunk_source_order",
                "target": 200,
                "achieved_regex_tokens": packed_m.packed_token_count,
                "budget_used": audit_m.get("matched_budget_used"),
                "diff": audit_m.get("matched_diff"),
                "artifact_sha256": sha256_text(packed_m.packed_text),
            }
        )
    assert all(not r["over_budget"] for r in backfill_rows), "smoke over-budget"
    write_text(OUT_ROOT / "token_util" / "BACKFILL_BUDGET_AUDIT.md", "\n".join(audit_lines))
    write_text(OUT_ROOT / "BACKFILL_BUDGET_AUDIT.md", "\n".join(audit_lines))
    write_json(OUT_ROOT / "token_util" / "MATCHED_ACTUAL_TOKEN_MANIFEST.json", matched_manifest)
    write_json(OUT_ROOT / "MATCHED_ACTUAL_TOKEN_MANIFEST.json", matched_manifest)
    write_csv(OUT_ROOT / "token_util" / "backfill_smoke.csv", backfill_rows)
    return {"n_dist": len(dist_rows), "over_budget_total": over_budget_total, "focus": focus_row}


if __name__ == "__main__":
    print(run_token_utilization())
