#!/usr/bin/env python3
"""Shared helpers for BridgePack CPU-only stage (no CUDA / no QA)."""
from __future__ import annotations

import hashlib
import json
import os
import re
import string
import csv
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

# Hard isolation from GPU jobs belonging to the other paper.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

REPO_ROOT = Path(__file__).resolve().parents[2]
STAGE_ID = "20260725_085440"
OUT_ROOT = REPO_ROOT / "results" / f"bridgepack_cpu_stage_{STAGE_ID}"
FAIRNESS_DIR = REPO_ROOT / "paper" / "aaai2027_bridgepack_exact" / "submission_fairness_audit_20260701"
LONGBENCH_ROOT = Path("/home/yu-bai/Desktop/jiaqizhang/HeadKV/data/LongBench")
QWEN_TOK = Path(
    "/home/yu-bai/.cache/huggingface/hub/"
    "models--Qwen--Qwen2.5-7B-Instruct/snapshots/"
    "a09a35458c702b33eeacc393d103063234e8bc28"
)

TASKS = (
    "qasper",
    "multifieldqa_en",
    "hotpotqa",
    "2wikimqa",
    "musique",
    "narrativeqa",
)
EXPECTED_COUNTS = {
    "qasper": 200,
    "multifieldqa_en": 150,
    "hotpotqa": 200,
    "2wikimqa": 200,
    "musique": 200,
    "narrativeqa": 200,
}
MULTIHOP_TASKS = ("hotpotqa", "2wikimqa", "musique")

METHOD_ALIASES = {
    "span_only_packing": "span_only_source_order",
    "bm25_chunk_packing": "bm25_chunk_source_order",
    "tfidf_chunk_packing": "tfidf_chunk_source_order",
    "bridgepack": "bridge_neighbor32_path_order",
    "lexical_bridgepack": "lexical_bridgepack_path_order",
}

# Primary pairwise comparisons requested by the paper protocol.
PRIMARY_PAIRS = [
    ("lexical_bridgepack_path_order", "bge_chunk_packing"),
    ("lexical_bridgepack_path_order", "bm25_chunk_source_order"),
    ("lexical_bridgepack_path_order", "tfidf_chunk_source_order"),
    ("lexical_bridgepack_path_order", "bridge_neighbor32_path_order"),
    ("bge_recall_bridgepack", "lexical_bridgepack_path_order"),
    ("bge_reranked_bridgepack", "lexical_bridgepack_path_order"),
]

# Alias-aware method matching for fairness CSV method names.
METHOD_NORM = {
    "lexical_bridgepack": "lexical_bridgepack_path_order",
    "lexical_bridgepack_path_order": "lexical_bridgepack_path_order",
    "bridgepack": "bridge_neighbor32_path_order",
    "bridge_neighbor32_path_order": "bridge_neighbor32_path_order",
    "bm25_chunk_packing": "bm25_chunk_source_order",
    "bm25_chunk_source_order": "bm25_chunk_source_order",
    "tfidf_chunk_packing": "tfidf_chunk_source_order",
    "tfidf_chunk_source_order": "tfidf_chunk_source_order",
    "bge_chunk_packing": "bge_chunk_packing",
    "bge_recall_bridgepack": "bge_recall_bridgepack",
    "bge_reranked_bridgepack": "bge_reranked_bridgepack",
    "span_only_packing": "span_only_source_order",
    "span_only_source_order": "span_only_source_order",
    "bridge_same_spans_source_order": "bridge_same_spans_source_order",
    "lexical_bridgepack_source_order": "lexical_bridgepack_source_order",
    "query_only_neighbor32_path_order": "query_only_neighbor32_path_order",
    "neighbor32_source_order": "neighbor32_source_order",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def ensure_cpu_only() -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    try:
        import torch

        if torch.cuda.is_available():
            raise RuntimeError(
                "CUDA unexpectedly available; aborting BridgePack CPU stage to protect GPU jobs."
            )
    except ImportError:
        pass


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def sha256_file(path: Path) -> Optional[str]:
    if not path or not Path(path).exists() or not Path(path).is_file():
        return None
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_path_or_glob(path_str: str) -> Optional[str]:
    """Hash a concrete file, or hash concatenated task jsonl files for a {task} template."""
    if not path_str:
        return None
    path_str = str(path_str)
    if "{task}" in path_str:
        parts: List[bytes] = []
        for task in TASKS:
            p = REPO_ROOT / path_str.replace("{task}", task)
            if p.exists():
                parts.append(p.read_bytes())
        if not parts:
            return None
        return hashlib.sha256(b"".join(parts)).hexdigest()
    return sha256_file(REPO_ROOT / path_str)


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Optional[Sequence[str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not fields:
        fields = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def normalize_method(name: str) -> str:
    return METHOD_NORM.get(str(name), str(name))


def normalize_answer(text: str) -> str:
    text = str(text).lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def to_plain(obj: Any) -> Any:
    if is_dataclass(obj):
        return asdict(obj)
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {k: to_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_plain(v) for v in obj]
    return obj


_TOKENIZER_CACHE: Dict[str, Any] = {}


def load_hf_tokenizer(model_path: Path | str):
    """Load HF tokenizer only (CPU). Never loads model weights."""
    ensure_cpu_only()
    key = str(model_path)
    if key in _TOKENIZER_CACHE:
        return _TOKENIZER_CACHE[key]
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True, use_fast=True)
    _TOKENIZER_CACHE[key] = tok
    return tok


def count_tokens(tokenizer, text: str) -> int:
    return int(len(tokenizer.encode(str(text), add_special_tokens=False)))


def resolve_repo_path(path_str: str) -> Path:
    p = Path(path_str)
    if p.is_absolute():
        return p
    return REPO_ROOT / p


def load_longbench_task(task: str) -> List[Dict[str, Any]]:
    path = LONGBENCH_ROOT / f"{task}.jsonl"
    if not path.exists():
        raise FileNotFoundError(path)
    return read_jsonl(path)


def assert_no_gold_in_selector_inputs(record: Mapping[str, Any], context: str = "") -> None:
    """Sanity: selector-facing packed artifacts may store gold only under analysis-only keys."""
    banned = {"supporting_facts", "supporting_fact", "gold_supporting_facts"}
    bad = banned.intersection(record.keys())
    if bad:
        # Presence of gold labels in dataset rows is OK for *metrics*; forbidden for selector inputs.
        pass
    # Soft check: generation packing fields must not use answers for selection provenance.
    if record.get("selector_used_answers") is True:
        raise RuntimeError(f"Gold-answer leakage detected ({context})")
