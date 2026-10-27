#!/usr/bin/env python3
"""Shared, CPU-safe primitives for the Qwen unified LongBench pipeline."""
from __future__ import annotations

import csv
import fcntl
import gzip
import hashlib
import json
import os
import subprocess
import tempfile
from contextlib import AbstractContextManager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, TextIO, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_DIR = REPO_ROOT / "protocol/unified_longbench_v1"
PROTOCOL_ID = "ULB-v1-052cfbbeaa2cc42f"
DATA_ROOT = Path("/home/yu-bai/Desktop/jiaqizhang/HeadKV/data/LongBench")
QWEN_DRM_ROOT = REPO_ROOT / "results/drm_add_official_17task_full_20260715_084131"
QWEN_HISTORICAL_CC_ROOT = REPO_ROOT / "results/cc_sliminfer_v1_qwen25_17task_full_official_20260712_075238"
QWEN_MODEL_ROOT = Path(
    "/home/yu-bai/.cache/huggingface/hub/"
    "models--Qwen--Qwen2.5-7B-Instruct/snapshots/"
    "a09a35458c702b33eeacc393d103063234e8bc28"
)
PYTHON = REPO_ROOT / ".venv/bin/python"

TASKS: Tuple[str, ...] = (
    "passage_retrieval_en",
    "qasper",
    "multifieldqa_en",
    "hotpotqa",
    "2wikimqa",
    "musique",
    "gov_report",
    "qmsum",
    "multi_news",
    "vcsum",
    "passage_count",
    "trec",
    "triviaqa",
    "samsum",
    "lsht",
    "lcc",
    "repobench-p",
)
COUNTS: Dict[str, int] = {task: 200 for task in TASKS}
COUNTS.update({"multifieldqa_en": 150, "lcc": 500, "repobench-p": 500})
assert sum(COUNTS.values()) == 3950

DRM_METHODS: Tuple[str, ...] = (
    "sliminfer_original",
    "drm_swap_original_k",
    "sliminfer_larger_km",
    "sliminfer_random_add_km",
    "sliminfer_drm_add_km",
)
RUN_DIR_BY_METHOD: Dict[str, str] = {
    "sliminfer_original": "run_sliminfer_original",
    "drm_swap_original_k": "run_drm_swap_original_k",
    "sliminfer_larger_km": "run_sliminfer_larger_km",
    "sliminfer_random_add_km": "run_sliminfer_random_add_km",
    "sliminfer_drm_add_km": "run_sliminfer_drm_add_km",
}
EXPECTED_RAW_MACRO = {
    "sliminfer_original": 47.637647,
    "drm_swap_original_k": 47.673529,
    "sliminfer_larger_km": 47.651176,
    "sliminfer_random_add_km": 47.663529,
    "sliminfer_drm_add_km": 47.796471,
}
EXPECTED_OOM = {
    "sliminfer_original": 4,
    "drm_swap_original_k": 4,
    "sliminfer_larger_km": 5,
    "sliminfer_random_add_km": 6,
    "sliminfer_drm_add_km": 4,
}

STAGES: Tuple[Dict[str, Any], ...] = (
    {"number": 0, "name": "preflight_and_baseline", "gpu": False},
    {"number": 1, "name": "original_equivalence_canary", "gpu": True},
    {"number": 2, "name": "qwen_cc_full", "gpu": True},
    {"number": 3, "name": "six_method_analysis", "gpu": False},
    {"number": 4, "name": "fullkv_fa2", "gpu": True},
    {"number": 5, "name": "minference_official", "gpu": True},
    {"number": 6, "name": "flexprefill_official", "gpu": True},
    {"number": 7, "name": "lazyllm_official", "gpu": True},
    {"number": 8, "name": "qwen_efficiency", "gpu": True},
    {"number": 9, "name": "cc_ablation_mechanism", "gpu": True},
    {"number": 10, "name": "drm_ablation_mechanism", "gpu": True},
    {"number": 11, "name": "oom_offload_fill", "gpu": True},
    {"number": 12, "name": "final_standardized_export", "gpu": False},
)
PIPELINE_STATES = {
    "NOT_STARTED", "PREFLIGHT", "RUNNING", "PASS", "BLOCKED",
    "FAILED", "PAUSED", "COMPLETE",
}


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_text(path, json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n")


def atomic_write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow({field: row.get(field, "") for field in fields})
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def run_command(args: Sequence[str], cwd: Path = REPO_ROOT, timeout: Optional[int] = None) -> Dict[str, Any]:
    proc = subprocess.run(
        list(args),
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
        check=False,
    )
    return {"returncode": proc.returncode, "output": proc.stdout}


def git_commit() -> str:
    return run_command(["git", "rev-parse", "HEAD"])["output"].strip()


def git_status() -> Dict[str, Any]:
    short = run_command(["git", "status", "--short", "--branch"])
    diff = run_command(["git", "diff", "--binary"])
    return {
        "commit": git_commit(),
        "status_short_branch": short["output"],
        "tracked_diff_sha256": sha256_bytes(diff["output"].encode("utf-8")),
        "tracked_diff": diff["output"],
    }


def sample_content(task: str, row: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "task": task,
        "context": row.get("context"),
        "input": row.get("input"),
        "question": row.get("question"),
        "answers": row.get("answers"),
        "all_classes": row.get("all_classes"),
        "language": row.get("language"),
    }


def canonical_sample_identity(task: str, index: int, row: Mapping[str, Any]) -> Tuple[str, str]:
    content_hash = sha256_bytes(canonical_bytes(sample_content(task, row)))
    return f"{task}:{index}:{content_hash[:16]}", content_hash


def load_task_samples(task: str, data_root: Path = DATA_ROOT) -> List[Dict[str, Any]]:
    path = data_root / f"{task}.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != COUNTS[task]:
        raise RuntimeError(f"{task}: expected {COUNTS[task]}, got {len(rows)}")
    return rows


def raw_sample_id(row: Mapping[str, Any], index: int) -> str:
    return str(row.get("_id", row.get("id", index)))


def iter_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except Exception as exc:
                raise RuntimeError(f"Invalid JSONL {path}:{line_number}: {exc}") from exc
            row.setdefault("_source_line", line_number)
            yield row


def prediction_status(row: Mapping[str, Any]) -> str:
    explicit = str(row.get("status", "") or "").lower()
    if explicit in {"ok", "oom", "failed", "missing"}:
        return explicit
    prediction = str(row.get("pred", row.get("prediction", "")))
    if prediction == "[OOM]":
        return "oom"
    if row.get("error"):
        return "failed"
    return "ok"


def per_task_score(task: str, rows: Sequence[Mapping[str, Any]]) -> float:
    from longbench_eval import scorer

    predictions = [str(row.get("pred", row.get("prediction", ""))) for row in rows]
    answers = [row.get("answers", []) for row in rows]
    lengths = [int(row.get("length", row.get("input_tokens", 0)) or 0) for row in rows]
    all_classes: Any = []
    for row in rows:
        if row.get("all_classes"):
            all_classes = row.get("all_classes")
    value = scorer(task, predictions, answers, lengths, all_classes)
    if value is None:
        raise RuntimeError(f"No score for {task}")
    return float(value)


def task_macro(scores: Mapping[str, float]) -> float:
    missing = [task for task in TASKS if task not in scores]
    if missing:
        raise RuntimeError(f"Missing task scores: {missing}")
    return sum(float(scores[task]) for task in TASKS) / len(TASKS)


def protocol_manifest_id(protocol_dir: Path = PROTOCOL_DIR) -> Tuple[str, str]:
    manifest = load_json(protocol_dir / "protocol_manifest.json")
    digest = sha256_bytes(canonical_bytes(manifest))
    return f"ULB-v1-{digest[:16]}", digest


def stage_decision_path(pipeline_dir: Path, stage: int) -> Path:
    return pipeline_dir / f"STAGE_{stage:02d}_DECISION.json"


def read_stage_decision(pipeline_dir: Path, stage: int) -> Optional[Dict[str, Any]]:
    path = stage_decision_path(pipeline_dir, stage)
    return load_json(path) if path.exists() else None


def assert_previous_stage_pass(pipeline_dir: Path, stage: int) -> None:
    if stage <= 0:
        return
    previous = read_stage_decision(pipeline_dir, stage - 1)
    if not previous or previous.get("decision") != "PASS":
        raise RuntimeError(f"Stage {stage - 1:02d} is not PASS; refusing stage {stage:02d}")


class PipelineLock(AbstractContextManager["PipelineLock"]):
    """Prevent two orchestrators from mutating one pipeline directory."""

    def __init__(self, pipeline_dir: Path):
        self.path = pipeline_dir / ".pipeline.lock"
        self.handle: Optional[TextIO] = None

    def __enter__(self) -> "PipelineLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self.handle.close()
            self.handle = None
            raise RuntimeError(f"Pipeline already has an active orchestrator: {self.path}") from exc
        self.handle.seek(0)
        self.handle.truncate()
        self.handle.write(json.dumps({"pid": os.getpid(), "acquired_at": now_iso()}) + "\n")
        self.handle.flush()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.handle is not None:
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
            self.handle.close()
        return None


class AtomicGzipJsonlWriter(AbstractContextManager["AtomicGzipJsonlWriter"]):
    """Buffered gzip JSONL writer that publishes only after a clean task close."""

    def __init__(self, final_path: Path, buffer_events: int = 256):
        if final_path.suffix != ".gz":
            raise ValueError("Trace path must end in .gz")
        self.final_path = final_path
        self.tmp_path = final_path.with_name(final_path.name + ".tmp")
        self.buffer_events = max(1, int(buffer_events))
        self.buffer: List[str] = []
        self.handle: Optional[TextIO] = None
        self.count = 0

    def __enter__(self) -> "AtomicGzipJsonlWriter":
        self.final_path.parent.mkdir(parents=True, exist_ok=True)
        if self.final_path.exists():
            raise FileExistsError(f"Refusing to overwrite completed trace: {self.final_path}")
        self.handle = gzip.open(self.tmp_path, "wt", encoding="utf-8", compresslevel=6)
        return self

    def write(self, row: Mapping[str, Any]) -> None:
        if self.handle is None:
            raise RuntimeError("Trace writer is not open")
        self.buffer.append(json.dumps(dict(row), ensure_ascii=False, separators=(",", ":")) + "\n")
        self.count += 1
        if len(self.buffer) >= self.buffer_events:
            self._flush_buffer()

    def _flush_buffer(self) -> None:
        if self.handle is not None and self.buffer:
            self.handle.writelines(self.buffer)
            self.buffer.clear()

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.handle is not None:
            if exc_type is None:
                self._flush_buffer()
            self.handle.close()
            self.handle = None
        if exc_type is None:
            os.replace(self.tmp_path, self.final_path)
        return None


class AtomicTaskTraceManager:
    """Publish one gzip trace per task while keeping interrupted tasks private."""

    def __init__(self, trace_dir: Path, buffer_events: int = 256):
        self.trace_dir = trace_dir
        self.buffer_events = buffer_events
        self.current_task: Optional[str] = None
        self.writer: Optional[AtomicGzipJsonlWriter] = None

    def start_task(self, task: str) -> None:
        if self.writer is not None:
            raise RuntimeError(
                f"Trace task {self.current_task} was not finalized before starting {task}"
            )
        final_path = self.trace_dir / f"{task}.jsonl.gz"
        self.writer = AtomicGzipJsonlWriter(final_path, self.buffer_events)
        self.writer.__enter__()
        self.current_task = task

    def write(self, row: Mapping[str, Any]) -> None:
        task = str(row.get("task", ""))
        if self.writer is None or self.current_task != task:
            raise RuntimeError(
                f"Trace event task {task!r} does not match open task {self.current_task!r}"
            )
        self.writer.write(row)

    def finish_task(self, task: str) -> int:
        if self.writer is None or self.current_task != task:
            raise RuntimeError(
                f"Cannot finish trace task {task}; open task is {self.current_task}"
            )
        count = self.writer.count
        self.writer.__exit__(None, None, None)
        self.writer = None
        self.current_task = None
        return count

    def abort_open_task(self) -> None:
        if self.writer is not None:
            self.writer.__exit__(RuntimeError, RuntimeError("task aborted"), None)
            self.writer = None
            self.current_task = None


def validate_unique_ids(rows: Sequence[Mapping[str, Any]], task: str) -> Dict[str, Any]:
    seen: Dict[str, int] = {}
    duplicates: List[Dict[str, Any]] = []
    for position, row in enumerate(rows):
        sample_id = str(row.get("sample_id", row.get("sample_index", position)))
        if sample_id in seen:
            duplicates.append({"sample_id": sample_id, "first": seen[sample_id], "duplicate": position})
        else:
            seen[sample_id] = position
    return {
        "task": task,
        "rows": len(rows),
        "unique_ids": len(seen),
        "duplicates": duplicates,
        "pass": not duplicates,
    }


def implementation_revision(paths: Sequence[Path]) -> str:
    payload = []
    for path in sorted(paths):
        payload.append({
            "path": str(path.relative_to(REPO_ROOT)),
            "sha256": sha256_file(path),
        })
    return sha256_bytes(canonical_bytes(payload))
