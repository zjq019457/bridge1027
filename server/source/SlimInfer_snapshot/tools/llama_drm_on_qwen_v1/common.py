#!/usr/bin/env python3
"""Shared IO / identity helpers for LLAMA_DRM_ON_QWEN_V1."""
from __future__ import annotations

import csv
import fcntl
import hashlib
import json
import os
import subprocess
import tempfile
from contextlib import AbstractContextManager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, TextIO

from tools.llama_drm_on_qwen_v1.constants import (
    COUNTS,
    DATA_ROOT,
    METHOD_SPECS,
    PROTOCOL_DIR,
    PROTOCOL_NAME,
    REPO_ROOT,
    TASKS,
    ULB_PROTOCOL_DIR,
)


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


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8", errors="ignore"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            try:
                os.unlink(tmp_name)
            except OSError:
                pass


def atomic_write_json(path: Path, payload: Any) -> None:
    atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n")


def atomic_write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow({key: row.get(key, "") for key in fieldnames})
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            try:
                os.unlink(tmp_name)
            except OSError:
                pass


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def iter_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    if not path.exists():
        return
        yield  # pragma: no cover
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")


def load_task_samples(task: str) -> List[Dict[str, Any]]:
    path = DATA_ROOT / f"{task}.jsonl"
    rows = list(iter_jsonl(path))
    if len(rows) != COUNTS[task]:
        raise RuntimeError(f"Task {task} count mismatch: got {len(rows)} expected {COUNTS[task]}")
    return rows


def canonical_sample_identity(task: str, index: int, sample: Mapping[str, Any]) -> tuple[str, str]:
    payload = {
        "task": task,
        "context": sample.get("context", ""),
        "input": sample.get("input", ""),
        "question": sample.get("question", sample.get("input", "")),
        "answers": sample.get("answers", []),
        "all_classes": sample.get("all_classes"),
        "language": sample.get("language"),
    }
    content_hash = sha256_bytes(canonical_bytes(payload))
    sample_id = f"{task}:{index}:{content_hash[:16]}"
    return sample_id, content_hash


def resolve_method_key(method_key: str) -> Dict[str, Any]:
    if method_key.startswith("random_seed_"):
        seed = int(method_key.rsplit("_", 1)[-1])
        spec = dict(METHOD_SPECS["sliminfer_random_add_km"])
        spec["seed"] = seed
        spec["method_name"] = "sliminfer_random_add_km"
        spec["method_key"] = method_key
        return spec
    if method_key == "full_kv":
        return {
            "runner": "fullkv",
            "policy": "none",
            "score": "delta_o",
            "budget": "full_kv",
            "seed": 0,
            "method_name": "full_kv",
            "method_key": method_key,
        }
    if method_key not in METHOD_SPECS:
        raise KeyError(f"Unknown method key: {method_key}")
    spec = dict(METHOD_SPECS[method_key])
    spec["method_name"] = method_key
    spec["method_key"] = method_key
    return spec


def protocol_hash_payload() -> Dict[str, Any]:
    files = sorted(PROTOCOL_DIR.glob("*"))
    skip = {"PROTOCOL_HASH.txt", "PROTOCOL_ID.txt"}
    file_hashes = {
        path.name: sha256_file(path)
        for path in files
        if path.is_file() and path.name not in skip
    }
    return {
        "protocol_name": PROTOCOL_NAME,
        "files": file_hashes,
        "ulb_protocol_id": load_json(ULB_PROTOCOL_DIR / "protocol_manifest.json").get("protocol_id"),
        "tasks": list(TASKS),
        "counts": dict(COUNTS),
        "methods": list(METHOD_SPECS.keys()),
    }


def compute_protocol_id_and_hash() -> tuple[str, str]:
    payload = protocol_hash_payload()
    digest = sha256_bytes(canonical_bytes(payload))
    protocol_id = f"LLAMA-DRM-ON-QWEN-V1-{digest[:16]}"
    return protocol_id, digest


class GlobalSchedulerLock(AbstractContextManager["GlobalSchedulerLock"]):
    def __init__(self, lock_path: Path):
        self.lock_path = lock_path
        self.handle: Optional[TextIO] = None

    def __enter__(self) -> "GlobalSchedulerLock":
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.lock_path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(
                f"Another LLAMA_DRM_ON_QWEN_V1 scheduler holds {self.lock_path}"
            ) from exc
        self.handle.seek(0)
        self.handle.truncate()
        self.handle.write(f"pid={os.getpid()} started={now_iso()}\n")
        self.handle.flush()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.handle is not None:
            try:
                fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
            finally:
                self.handle.close()
                self.handle = None


def run_command(
    cmd: Sequence[str],
    *,
    cwd: Optional[Path] = None,
    env: Optional[Mapping[str, str]] = None,
    log_path: Optional[Path] = None,
) -> Dict[str, Any]:
    merged = dict(os.environ)
    if env:
        merged.update({str(k): str(v) for k, v in env.items()})
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as log:
            log.write("\n$ " + " ".join(str(x) for x in cmd) + "\n")
            log.flush()
            proc = subprocess.run(
                list(cmd),
                cwd=str(cwd or REPO_ROOT),
                env=merged,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
            )
            log.write(f"\n[returncode] {proc.returncode}\n")
            return {"returncode": int(proc.returncode), "output": "", "log_path": str(log_path)}
    proc = subprocess.run(
        list(cmd),
        cwd=str(cwd or REPO_ROOT),
        env=merged,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return {"returncode": int(proc.returncode), "output": proc.stdout or ""}


def offline_env(extra: Optional[Mapping[str, str]] = None) -> Dict[str, str]:
    env = {
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "WANDB_MODE": "offline",
        "TOKENIZERS_PARALLELISM": "false",
        "PYTHONUNBUFFERED": "1",
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
    }
    if extra:
        env.update(dict(extra))
    return env


def existing_prediction_ids(path: Path) -> set[str]:
    ids: set[str] = set()
    for row in iter_jsonl(path):
        sid = row.get("sample_id")
        if sid:
            ids.add(str(sid))
    return ids


def missing_sample_indices(task: str, prediction_path: Path) -> List[int]:
    samples = load_task_samples(task)
    have = existing_prediction_ids(prediction_path)
    missing: List[int] = []
    for index, sample in enumerate(samples):
        sample_id, _ = canonical_sample_identity(task, index, sample)
        if sample_id not in have:
            missing.append(index)
    return missing
