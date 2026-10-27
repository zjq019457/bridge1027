#!/usr/bin/env python3
"""Continuous, non-invasive resource watchdog for one Qwen GPU worker."""
from __future__ import annotations

import json
import os
import signal
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from tools.qwen_pipeline_common import atomic_write_json, atomic_write_text, now_iso
from tools.qwen_resource_guard import collect_snapshot, directory_size


MIB = 1024 ** 2


def _read_int(path: Path) -> Optional[int]:
    try:
        return int(path.read_text(encoding="utf-8").strip())
    except (FileNotFoundError, OSError, ValueError):
        return None


def collect_worker_memory(
    pid: int,
    *,
    proc_root: Path = Path("/proc"),
    cgroup_root: Path = Path("/sys/fs/cgroup"),
) -> Dict[str, Any]:
    """Read process and cgroup memory without requiring elevated privileges."""
    status_path = proc_root / str(int(pid)) / "status"
    status: Dict[str, int] = {}
    try:
        for line in status_path.read_text(encoding="utf-8").splitlines():
            key, separator, raw = line.partition(":")
            if not separator or key not in {"VmRSS", "VmSwap"}:
                continue
            status[key] = int(raw.strip().split()[0]) * 1024
    except (FileNotFoundError, OSError, ValueError, IndexError):
        pass

    cgroup_relative: Optional[str] = None
    try:
        for line in (proc_root / str(int(pid)) / "cgroup").read_text(
            encoding="utf-8"
        ).splitlines():
            hierarchy, controllers, relative = line.split(":", 2)
            if hierarchy == "0" and not controllers:
                cgroup_relative = relative
                break
    except (FileNotFoundError, OSError, ValueError):
        pass
    cgroup_path = (
        cgroup_root / cgroup_relative.lstrip("/")
        if cgroup_relative is not None
        else None
    )
    return {
        "pid": int(pid),
        "process_available": status_path.exists(),
        "worker_rss_bytes": status.get("VmRSS"),
        "worker_swap_bytes": status.get("VmSwap"),
        "cgroup_path": str(cgroup_path) if cgroup_path is not None else None,
        "cgroup_available": bool(cgroup_path and cgroup_path.exists()),
        "cgroup_memory_current_bytes": (
            _read_int(cgroup_path / "memory.current") if cgroup_path else None
        ),
        "cgroup_swap_current_bytes": (
            _read_int(cgroup_path / "memory.swap.current") if cgroup_path else None
        ),
        "cgroup_memory_peak_bytes": (
            _read_int(cgroup_path / "memory.peak") if cgroup_path else None
        ),
        "cgroup_swap_peak_bytes": (
            _read_int(cgroup_path / "memory.swap.peak") if cgroup_path else None
        ),
    }


def _jsonl_count(path: Optional[Path]) -> int:
    if path is None or not path.exists():
        return 0
    count = 0
    try:
        with path.open("rb") as handle:
            for line in handle:
                if line.strip():
                    count += 1
    except OSError:
        return count
    return count


def _last_prediction(path: Optional[Path]) -> Dict[str, Any]:
    if path is None or not path.exists():
        return {}
    last = b""
    try:
        with path.open("rb") as handle:
            for line in handle:
                if line.strip():
                    last = line
        return json.loads(last) if last else {}
    except Exception:
        return {}


def _consecutive_oom(path: Optional[Path]) -> int:
    if path is None or not path.exists():
        return 0
    statuses: List[str] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                status = str(row.get("status", "ok")).lower()
                if row.get("pred") == "[OOM]":
                    status = "oom"
                statuses.append(status)
    except Exception:
        return 0
    count = 0
    for status in reversed(statuses):
        if status != "oom":
            break
        count += 1
    return count


def evaluate_runtime(
    snapshot: Mapping[str, Any],
    policy: Mapping[str, Any],
    *,
    initial_swap_bytes: int,
    pipeline_dir: Path,
    allowed_pids: Sequence[int],
    prediction_path: Optional[Path],
    system_swap_hard_gate: bool = True,
) -> List[Dict[str, Any]]:
    """Runtime checks intentionally omit idle-GPU/free-VRAM pre-load constraints."""
    failures: List[Dict[str, Any]] = []
    thresholds = policy["thresholds"]
    gpu = snapshot["gpu"]
    if not gpu.get("available"):
        failures.append({"check": "gpu_available", "value": gpu.get("error")})
    elif int(gpu["temperature_gpu_c"]) > int(thresholds["max_gpu_temperature_c"]):
        failures.append({
            "check": "gpu_temperature_c", "value": gpu["temperature_gpu_c"],
            "threshold": thresholds["max_gpu_temperature_c"],
        })
    memory = snapshot["host_memory"]
    if int(memory["ram_available_bytes"]) < int(thresholds["min_host_available_bytes"]):
        failures.append({
            "check": "host_available_bytes", "value": memory["ram_available_bytes"],
            "threshold": thresholds["min_host_available_bytes"],
        })
    swap_growth = max(0, int(memory["swap_used_bytes"]) - int(initial_swap_bytes))
    if (
        system_swap_hard_gate
        and swap_growth > int(thresholds["max_swap_growth_bytes_during_gate"])
    ):
        failures.append({
            "check": "swap_growth_bytes", "value": swap_growth,
            "threshold": thresholds["max_swap_growth_bytes_during_gate"],
        })
    if int(snapshot["disk"]["free_bytes"]) < int(thresholds["min_disk_free_bytes"]):
        failures.append({
            "check": "disk_free_bytes", "value": snapshot["disk"]["free_bytes"],
            "threshold": thresholds["min_disk_free_bytes"],
        })
    if snapshot["recent_driver_errors"].get("matches"):
        failures.append({
            "check": "recent_driver_errors",
            "value": snapshot["recent_driver_errors"]["matches"],
        })
    allowed = {int(pid) for pid in allowed_pids if pid}
    unexpected_gpu = [
        row for row in snapshot["gpu_compute_processes"]
        if not row.get("desktop_allowlisted") and int(row["pid"]) not in allowed
    ]
    if unexpected_gpu:
        failures.append({"check": "unexpected_gpu_processes", "value": unexpected_gpu})
    unexpected_jobs = [
        row for row in snapshot["experiment_processes"] if int(row["pid"]) not in allowed
    ]
    if unexpected_jobs:
        failures.append({"check": "other_experiment_processes", "value": unexpected_jobs})
    output_bytes = directory_size(pipeline_dir)
    if output_bytes > int(thresholds["max_pipeline_output_bytes"]):
        failures.append({
            "check": "pipeline_output_bytes", "value": output_bytes,
            "threshold": thresholds["max_pipeline_output_bytes"],
        })
    consecutive_oom = _consecutive_oom(prediction_path)
    if consecutive_oom > int(thresholds["max_consecutive_sample_oom"]):
        failures.append({
            "check": "consecutive_sample_oom", "value": consecutive_oom,
            "threshold": thresholds["max_consecutive_sample_oom"],
        })
    return failures


class RuntimeWatchdog:
    """Poll resources every interval and stop only the owned GPU child on danger."""

    def __init__(
        self,
        *,
        pipeline_dir: Path,
        run_root: Path,
        policy: Mapping[str, Any],
        stage: int,
        method: str,
        task: str,
        worker_pid: int,
        prediction_path: Optional[Path],
        trace_path: Optional[Path],
        log_path: Path,
        resume_command: str,
        interval_seconds: int = 30,
        swap_baseline_bytes: Optional[int] = None,
    ) -> None:
        self.pipeline_dir = pipeline_dir
        self.run_root = run_root
        self.policy = policy
        self.stage = stage
        self.method = method
        self.task = task
        self.worker_pid = int(worker_pid)
        self.prediction_path = prediction_path
        self.trace_path = trace_path
        self.log_path = log_path
        self.resume_command = resume_command
        self.interval_seconds = max(5, int(interval_seconds))
        self.swap_baseline_bytes = swap_baseline_bytes
        self.stop_event = threading.Event()
        self.thread: Optional[threading.Thread] = None
        self.failure: Optional[Dict[str, Any]] = None
        self.initial_swap_bytes = 0
        self.worker_memory_baseline: Dict[str, Any] = {}
        self.post_load_baseline: Optional[Dict[str, Optional[int]]] = None
        self.post_load_candidate: Optional[Dict[str, Optional[int]]] = None
        self.post_load_stable_observations = 0
        self.model_loaded_at: Optional[str] = None
        self.previous_swap_values: Dict[str, Optional[int]] = {}
        self.primary_growth_observations: Dict[str, int] = {
            "worker": 0,
            "cgroup": 0,
        }

    def start(self) -> None:
        initial = collect_snapshot()
        self.initial_swap_bytes = (
            int(self.swap_baseline_bytes)
            if self.swap_baseline_bytes is not None
            else int(initial["host_memory"]["swap_used_bytes"])
        )
        self.worker_memory_baseline = collect_worker_memory(self.worker_pid)
        self._record(initial)
        self.thread = threading.Thread(target=self._loop, name="qwen-runtime-watchdog", daemon=True)
        self.thread.start()

    def _loop(self) -> None:
        while not self.stop_event.wait(self.interval_seconds):
            try:
                self._record(collect_snapshot())
            except Exception as exc:
                self._fail([{"check": "watchdog_exception", "value": repr(exc)}], {})
            if self.failure is not None:
                return

    def _worker_has_loaded_model(self, snapshot: Mapping[str, Any]) -> bool:
        gpu = snapshot.get("gpu", {})
        total_mib = int(gpu.get("memory_total_mib") or 0)
        minimum_mib = max(1024, total_mib // 4)
        return any(
            int(row.get("pid", -1)) == self.worker_pid
            and int(row.get("used_memory_mib") or 0) >= minimum_mib
            for row in snapshot.get("gpu_compute_processes", [])
        )

    @staticmethod
    def _swap_values(
        snapshot: Mapping[str, Any], worker_memory: Mapping[str, Any]
    ) -> Dict[str, Optional[int]]:
        return {
            "system": int(snapshot["host_memory"]["swap_used_bytes"]),
            "worker": worker_memory.get("worker_swap_bytes"),
            "cgroup": worker_memory.get("cgroup_swap_current_bytes"),
        }

    def _swap_observation(
        self, snapshot: Mapping[str, Any], worker_memory: Mapping[str, Any]
    ) -> Dict[str, Any]:
        thresholds = self.policy["thresholds"]
        threshold = int(thresholds["max_swap_growth_bytes_during_gate"])
        required = int(thresholds.get("runtime_swap_sustained_observations", 2))
        stable_required = int(
            thresholds.get("runtime_post_load_stable_observations", 2)
        )
        stable_delta = int(
            thresholds.get("runtime_post_load_stable_delta_bytes", 64 * MIB)
        )
        growth_epsilon = int(
            thresholds.get("runtime_swap_growth_epsilon_bytes", 1 * MIB)
        )
        current = self._swap_values(snapshot, worker_memory)
        now = now_iso()
        if self.model_loaded_at is None and self._worker_has_loaded_model(snapshot):
            self.model_loaded_at = now
            self.post_load_candidate = dict(current)
            self.post_load_stable_observations = 1
        elif self.model_loaded_at is not None and self.post_load_baseline is None:
            previous_candidate = self.post_load_candidate or current
            comparable = [
                abs(int(current[key]) - int(previous_candidate[key]))
                for key in ("worker", "cgroup")
                if current.get(key) is not None and previous_candidate.get(key) is not None
            ]
            if comparable and max(comparable) <= stable_delta:
                self.post_load_stable_observations += 1
            else:
                self.post_load_stable_observations = 1
            self.post_load_candidate = dict(current)
            if self.post_load_stable_observations >= stable_required:
                self.post_load_baseline = dict(current)
                self.primary_growth_observations = {"worker": 0, "cgroup": 0}

        launch_baseline = {
            "system": self.initial_swap_bytes,
            "worker": self.worker_memory_baseline.get("worker_swap_bytes"),
            "cgroup": self.worker_memory_baseline.get("cgroup_swap_current_bytes"),
        }
        active_baseline = self.post_load_baseline or launch_baseline
        deltas: Dict[str, Optional[int]] = {}
        for key in ("system", "worker", "cgroup"):
            value = current.get(key)
            baseline = active_baseline.get(key)
            deltas[key] = (
                max(0, int(value) - int(baseline))
                if value is not None and baseline is not None
                else None
            )

        failures: List[Dict[str, Any]] = []
        warnings: List[Dict[str, Any]] = []
        for key in ("worker", "cgroup"):
            value = current.get(key)
            previous = self.previous_swap_values.get(key)
            delta = deltas.get(key)
            growing = (
                value is not None
                and previous is not None
                and int(value) - int(previous) >= growth_epsilon
            )
            if delta is not None and delta > threshold and growing:
                self.primary_growth_observations[key] += 1
            elif delta is None or delta <= threshold or not growing:
                self.primary_growth_observations[key] = 0
            if self.primary_growth_observations[key] >= required:
                failures.append({
                    "check": f"{key}_swap_sustained_growth_bytes",
                    "value": delta,
                    "threshold": threshold,
                    "consecutive_growth_observations": self.primary_growth_observations[key],
                    "baseline_phase": (
                        "post_model_load" if self.post_load_baseline else "worker_launch"
                    ),
                })
        if deltas["system"] is not None and deltas["system"] > threshold:
            warnings.append({
                "check": "system_swap_growth_auxiliary_bytes",
                "value": deltas["system"],
                "threshold": threshold,
                "hard_gate": False,
            })
        if current["worker"] is None:
            warnings.append({"check": "worker_swap_metric_unavailable", "hard_gate": False})
        if current["cgroup"] is None:
            warnings.append({"check": "cgroup_swap_metric_unavailable", "hard_gate": False})
        self.previous_swap_values = dict(current)
        return {
            "phase": (
                "inference"
                if self.post_load_baseline is not None
                else "model_loading_or_warmup"
            ),
            "model_loaded_at": self.model_loaded_at,
            "worker_memory": dict(worker_memory),
            "launch_baseline_bytes": launch_baseline,
            "post_load_baseline_bytes": self.post_load_baseline,
            "active_baseline_bytes": active_baseline,
            "current_bytes": current,
            "growth_bytes": deltas,
            "post_load_stable_observations": self.post_load_stable_observations,
            "required_stable_observations": stable_required,
            "primary_growth_observations": dict(self.primary_growth_observations),
            "required_growth_observations": required,
            "warnings": warnings,
            "failures": failures,
        }

    def _record(self, snapshot: Mapping[str, Any]) -> None:
        last = _last_prediction(self.prediction_path)
        prediction_count = _jsonl_count(self.prediction_path)
        trace_bytes = self.trace_path.stat().st_size if self.trace_path and self.trace_path.exists() else 0
        worker_memory = collect_worker_memory(self.worker_pid)
        swap_monitor = self._swap_observation(snapshot, worker_memory)
        failures = evaluate_runtime(
            snapshot, self.policy, initial_swap_bytes=self.initial_swap_bytes,
            pipeline_dir=self.pipeline_dir,
            allowed_pids=(os.getpid(), self.worker_pid),
            prediction_path=self.prediction_path,
            system_swap_hard_gate=False,
        )
        failures.extend(swap_monitor["failures"])
        record = {
            "timestamp": now_iso(),
            "stage": self.stage,
            "method": self.method,
            "task": self.task,
            "sample_progress": prediction_count,
            "last_sample_id": last.get("sample_id"),
            "last_sample_index": last.get("sample_index"),
            "gpu_memory_used_mib": snapshot.get("gpu", {}).get("memory_used_mib"),
            "gpu_utilization_percent": snapshot.get("gpu", {}).get("utilization_gpu_percent"),
            "gpu_temperature_c": snapshot.get("gpu", {}).get("temperature_gpu_c"),
            "ram_available_bytes": snapshot.get("host_memory", {}).get("ram_available_bytes"),
            "swap_used_bytes": snapshot.get("host_memory", {}).get("swap_used_bytes"),
            "disk_free_bytes": snapshot.get("disk", {}).get("free_bytes"),
            "worker_pid": self.worker_pid,
            "trace_bytes": trace_bytes,
            "swap_monitor": swap_monitor,
            "warnings": swap_monitor["warnings"],
            "failures": failures,
        }
        watchdog_log = self.pipeline_dir / "WATCHDOG.jsonl"
        watchdog_log.parent.mkdir(parents=True, exist_ok=True)
        with watchdog_log.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
        atomic_write_json(self.pipeline_dir / "WATCHDOG_STATUS.json", {
            "updated_at": record["timestamp"],
            "status": (
                "HARD_STOP"
                if failures
                else "HEALTHY_WITH_WARNINGS"
                if record["warnings"]
                else "HEALTHY"
            ),
            "latest": record,
            "log": str(watchdog_log),
        })
        atomic_write_json(self.pipeline_dir / "PROGRESS.json", {
            "updated_at": record["timestamp"],
            "stage": self.stage,
            "method": self.method,
            "task": self.task,
            "completed_samples_current_task": prediction_count,
            "last_completed_sample_id": last.get("sample_id"),
            "last_completed_sample_index": last.get("sample_index"),
            "worker_pid": self.worker_pid,
            "trace_bytes": trace_bytes,
        })
        if failures:
            failure_snapshot = dict(snapshot)
            failure_snapshot["swap_monitor"] = swap_monitor
            self._fail(failures, failure_snapshot)

    def _fail(self, failures: List[Dict[str, Any]], snapshot: Mapping[str, Any]) -> None:
        if self.failure is not None:
            return
        last = _last_prediction(self.prediction_path)
        self.failure = {
            "created_at": now_iso(),
            "reason": "runtime_watchdog_hard_failure",
            "failures": failures,
            "stage": self.stage,
            "method": self.method,
            "task": self.task,
            "last_completed_sample_id": last.get("sample_id"),
            "last_completed_sample_index": last.get("sample_index"),
            "worker_pid": self.worker_pid,
            "snapshot": snapshot,
            "log_path": str(self.log_path),
            "watchdog_log": str(self.pipeline_dir / "WATCHDOG.jsonl"),
            "resume_command": self.resume_command,
        }
        atomic_write_json(self.pipeline_dir / "SAFETY_STOP.json", self.failure)
        atomic_write_text(self.pipeline_dir / "RESUME_COMMAND.txt", self.resume_command + "\n")
        try:
            os.kill(self.worker_pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        self.stop_event.set()

    def stop(self) -> Optional[Dict[str, Any]]:
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=max(5, self.interval_seconds + 2))
        return self.failure
