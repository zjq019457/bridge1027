#!/usr/bin/env python3
"""External compression interface audit (no compressor model load; synthetic/cached dry-run)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from .common import (
    OUT_ROOT,
    QWEN_TOK,
    REPO_ROOT,
    count_tokens,
    ensure_cpu_only,
    load_hf_tokenizer,
    sha256_text,
    utc_now,
    write_json,
    write_text,
)


def _synthetic_official_raw(text: str, keep_ratio: float = 0.35) -> str:
    """Deterministic fake 'official compressor' output for interface testing only."""
    words = text.split()
    keep = max(1, int(len(words) * keep_ratio))
    return " ".join(words[:keep])


def exact_budget_adapt(raw: str, tokenizer, budget: int, allow_silent_truncate: bool = False) -> Dict[str, Any]:
    """
    Adapt official raw text to exact budget.
    If over budget and allow_silent_truncate is False, record FAILURE (do not silently pass as official).
    """
    n = count_tokens(tokenizer, raw)
    if n <= budget:
        return {
            "status": "PASS",
            "protocol": "exact_budget_adapted_output",
            "raw_tokens": n,
            "adapted_text": raw,
            "adapted_tokens": n,
            "truncated": False,
            "failed_over_budget_official": False,
        }
    # Retain failure record for official raw; adapted is explicitly truncated.
    ids = tokenizer.encode(raw, add_special_tokens=False)[:budget]
    adapted = tokenizer.decode(ids, skip_special_tokens=True)
    return {
        "status": "ADAPTED_AFTER_OFFICIAL_OVER_BUDGET",
        "protocol": "exact_budget_adapted_output",
        "raw_tokens": n,
        "adapted_text": adapted,
        "adapted_tokens": count_tokens(tokenizer, adapted),
        "truncated": True,
        "failed_over_budget_official": True,
        "silent_truncate_forbidden": True,
        "note": "Adapted output MUST NOT be labeled as unmodified official result",
    }


def run_external_compression() -> Dict[str, Any]:
    ensure_cpu_only()
    tok = load_hf_tokenizer(QWEN_TOK)
    budget = 128
    long_text = ("Passage about multi-hop reasoning. " * 80) + "Answer entity: FooBar."
    official_raw = _synthetic_official_raw(long_text, keep_ratio=0.9)  # likely over budget
    official_n = count_tokens(tok, official_raw)
    official_record = {
        "protocol": "official_raw_output",
        "text": official_raw,
        "tokens": official_n,
        "budget": budget,
        "over_budget": official_n > budget,
        "status": "FAIL_OVER_BUDGET" if official_n > budget else "PASS",
        "artifact_sha256": sha256_text(official_raw),
    }
    # Must retain failure — do not convert official_raw in place.
    adapted = exact_budget_adapt(official_raw, tok, budget)

    # Audit existing llmlingua result dirs without loading models
    existing = []
    for p in sorted((REPO_ROOT / "results").glob("bridgepack_*llmlingua*")) + sorted(
        (REPO_ROOT / "results").glob("bridgepack_external_compression*")
    ):
        existing.append({"path": str(p.relative_to(REPO_ROOT)), "exists": True})

    protocol = f"""# EXTERNAL_COMPRESSION_PROTOCOL

Generated: {utc_now()}

## Two explicitly distinct outputs

1. `official_raw_output` — unmodified compressor output (may exceed budget). Over-budget => **FAIL record**, never silent truncate.
2. `exact_budget_adapted_output` — official output + deterministic tokenizer truncation/repack. Must be labeled adapted, never "fully original official".

## CPU stage policy

- Do not load LLMLingua/LongLLMLingua compressor weights while GPU is busy / CPU stage active.
- Interface dry-run uses synthetic text + audits prior result directories.
- QA runner exists in `tools/run_official_longllingua_qa_eval.py` / `tools/run_external_compression_qa_eval.py` but **must not execute** in this stage.

## Existing artifacts discovered

{chr(10).join('- `' + e['path'] + '`' for e in existing) or '- none'}
"""
    write_text(OUT_ROOT / "external_compression" / "EXTERNAL_COMPRESSION_PROTOCOL.md", protocol)
    write_text(OUT_ROOT / "EXTERNAL_COMPRESSION_PROTOCOL.md", protocol)

    audit = {
        "created_at": utc_now(),
        "official_raw_output": official_record,
        "exact_budget_adapted_output": {k: v for k, v in adapted.items() if k != "adapted_text"},
        "labels": {
            "may_call_adapted_official": False,
            "must_retain_official_failures": True,
        },
        "existing_dirs": existing,
        "qa_runner_status": "IMPLEMENTED_NOT_EXECUTED",
    }
    write_json(OUT_ROOT / "external_compression" / "EXTERNAL_COMPRESSION_INTERFACE_AUDIT.md.json", audit)
    # Also human markdown
    md = [
        "# EXTERNAL_COMPRESSION_INTERFACE_AUDIT",
        "",
        f"Generated: {utc_now()}",
        "",
        f"- official_raw tokens={official_n} budget={budget} status={official_record['status']}",
        f"- adapted status={adapted['status']} adapted_tokens={adapted['adapted_tokens']} truncated={adapted['truncated']}",
        "- Adapted results are **not** unmodified official outputs.",
        f"- Prior dirs audited: {len(existing)}",
        "- Compressor model: not loaded",
        "- QA: not executed",
        "",
    ]
    write_text(OUT_ROOT / "external_compression" / "EXTERNAL_COMPRESSION_INTERFACE_AUDIT.md", "\n".join(md))
    write_text(OUT_ROOT / "EXTERNAL_COMPRESSION_INTERFACE_AUDIT.md", "\n".join(md))

    gpu_manifest = {
        "status": "GPU_PENDING",
        "priority": "P0-G3",
        "steps": [
            "Generate official_raw_output with local compressor checkpoint",
            "Record over-budget failures without silent truncate",
            "Build exact_budget_adapted_output via Qwen tokenizer",
            "Budget audit PASS required before QA",
            "QA on Qwen 6-task; optional Llama-3",
        ],
        "n_samples_full": 1150,
        "budget": 1024,
        "do_not_auto_start": True,
    }
    write_json(OUT_ROOT / "external_compression" / "EXTERNAL_COMPRESSION_GPU_MANIFEST.json", gpu_manifest)
    write_json(OUT_ROOT / "EXTERNAL_COMPRESSION_GPU_MANIFEST.json", gpu_manifest)
    # persist dry-run artifacts
    write_json(
        OUT_ROOT / "artifacts" / "external_compression" / "dryrun.json",
        {"official": official_record, "adapted_meta": {k: adapted[k] for k in adapted if k != "adapted_text"}},
    )
    return audit


if __name__ == "__main__":
    print(json.dumps(run_external_compression(), indent=2)[:1000])
