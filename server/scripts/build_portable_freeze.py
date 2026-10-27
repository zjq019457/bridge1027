#!/usr/bin/env python3
"""Freeze the current successful BridgePack fresh reproduction into a portable root.

This builder only reads the historical/fresh results, current source, model, and
LongBench data. It never writes into SlimInfer or either historical result tree.
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable


PORTABLE = Path(__file__).resolve().parents[1]
REPO = Path(os.environ.get("BRIDGEPACK_REPO", "/home/yu-bai/Desktop/jiaqizhang/SlimInfer")).resolve()
GOLDEN = REPO / "results" / "bridgepack_g1_decoded_reproduction_20260928_180900"
MODEL = Path(os.environ.get(
    "BRIDGEPACK_MODEL_DIR",
    "/home/yu-bai/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28",
)).resolve()
DATA = Path(os.environ.get("BRIDG_PACK_DATA_ROOT", "/home/yu-bai/Desktop/jiaqizhang/HeadKV/data/LongBench")).resolve()
TASKS = ["qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "musique", "narrativeqa"]
COUNTS = {"qasper": 200, "multifieldqa_en": 150, "hotpotqa": 200, "2wikimqa": 200, "musique": 200, "narrativeqa": 200}
VARIANT = "same_spans_source_order"
BUDGET = 1024


def run(cmd: list[str], *, cwd: Path | None = None) -> str:
    p = subprocess.run(cmd, cwd=cwd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    return p.stdout


def write(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


def write_json(path: Path, value: Any) -> None:
    write(path, json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha_text(value: str) -> str:
    return sha_bytes(value.encode("utf-8", errors="replace"))


def canonical_hash(value: Any) -> str:
    return sha_text(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")))


def files_under(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc")


def checksum_lines(root: Path) -> tuple[str, str]:
    rows = []
    for p in files_under(root):
        rows.append(f"{sha_file(p)}  {p.relative_to(root).as_posix()}")
    return "\n".join(rows) + ("\n" if rows else ""), sha_text("\n".join(rows) + ("\n" if rows else ""))


def copytree_clean(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache"))


def freeze_git() -> None:
    m = PORTABLE / "manifests"
    state = {
        "repo": str(REPO),
        "head": run(["git", "rev-parse", "HEAD"], cwd=REPO).strip(),
        "status_porcelain_v1": run(["git", "status", "--porcelain=v1"], cwd=REPO),
        "submodule_status": run(["git", "submodule", "status"], cwd=REPO),
        "remote_v": run(["git", "remote", "-v"], cwd=REPO),
        "log_all_100": run(["git", "log", "--all", "--decorate", "--oneline", "-100"], cwd=REPO),
    }
    write(m / "GIT_STATE.txt", json.dumps(state, ensure_ascii=False, indent=2) + "\n")
    write(m / "GIT_DIFF.patch", run(["git", "diff"], cwd=REPO))
    write(m / "GIT_DIFF_CACHED.patch", run(["git", "diff", "--cached"], cwd=REPO))


def freeze_source() -> None:
    dst = PORTABLE / "source" / "SlimInfer_snapshot"
    for name in ("bridgekv", "tools", "configs"):
        copytree_clean(REPO / name, dst / name)
    shutil.copy2(REPO / "longbench_eval.py", dst / "longbench_eval.py")
    write(dst / "PORTABLE_SOURCE_SCOPE.md", """# Portable source scope\n\nThis is a runtime source snapshot for the 2026-09-28 fresh BridgePack G1 reproduction.\nIt contains the BridgePack implementation, G1 runner, generation helper, LongBench scorer, and prompt/config files.\nIt excludes `.git`, model weights, HF cache, result trees, temporary caches, and logs.\n\nThe snapshot preserves the historical G1 generation behavior, including its historical prompt binding, because the Portable Golden V1 target is the fresh run rather than a protocol rewrite.\n""")
    lines, tree_hash = checksum_lines(dst)
    write(PORTABLE / "manifests" / "SOURCE_SHA256SUMS.txt", lines)
    important = {
        "runner": "tools/bridgepack_g1/run_decoded_repro_1150.py",
        "packing": ["bridgekv/evidence_text_packing.py", "tools/bridgepack_g1/run_g1.py", "tools/run_bridgepack_exact_budget_repack.py"],
        "scorer": "longbench_eval.py",
        "prompt": "configs/dataset2prompt.json",
    }
    hashes = {}
    for key, names in important.items():
        seq = names if isinstance(names, list) else [names]
        hashes[key] = {name: sha_file(dst / name) for name in seq}
    write_json(PORTABLE / "manifests" / "SOURCE_PROVENANCE.json", {
        "repo": str(REPO),
        "current_head": run(["git", "rev-parse", "HEAD"], cwd=REPO).strip(),
        "source_snapshot": "source/SlimInfer_snapshot",
        "source_file_count": len(files_under(dst)),
        "source_tree_hash": tree_hash,
        "important_hashes": hashes,
        "excluded": [".git", "model weights", "HF cache", "results", "temporary caches", "logs"],
    })


def freeze_environment() -> None:
    env = PORTABLE / "environment"
    conda = shutil.which("conda")
    current = {
        "python": sys.executable,
        "python_version": platform.python_version(),
        "conda_default_env": os.environ.get("CONDA_DEFAULT_ENV", ""),
        "conda_prefix": os.environ.get("CONDA_PREFIX", ""),
        "silm_environment_present": Path("/home/yu-bai/miniconda3/envs/silm").exists(),
    }
    if conda:
        write(env / "conda_info.txt", run([conda, "info"]))
        write(env / "conda_list.txt", run([conda, "list"]))
        write(env / "conda_explicit_linux64.txt", run([conda, "list", "--explicit"]))
        write(env / "conda_env_list.txt", run([conda, "env", "list"]))
    else:
        write(env / "conda_info.txt", "CONDA_NOT_FOUND\n")
        write(env / "conda_list.txt", "CONDA_NOT_FOUND\n")
        write(env / "conda_explicit_linux64.txt", "CONDA_NOT_FOUND\n")
        write(env / "conda_env_list.txt", "CONDA_NOT_FOUND\n")
    write(env / "pip_freeze.txt", run([sys.executable, "-m", "pip", "freeze", "--all"]))
    freeze = (env / "pip_freeze.txt").read_text(encoding="utf-8", errors="replace")
    local = [line for line in freeze.splitlines() if "file://" in line or line.startswith("-e ") or "git+" in line or "@ /" in line]
    write(env / "LOCAL_PACKAGE_DEPENDENCIES.md", """# Local/package provenance\n\nThe historical requested `silm` environment is not present on this machine; the successful fresh run uses the current base interpreter unless the run artifact proves otherwise. The portable builder therefore freezes the actual interpreter shown below and does not silently relabel it as `silm`.\n\n## Interpreter\n\n```text\n""" + json.dumps(current, indent=2) + "\n```\n\n## pip freeze entries with local/build provenance\n\n" + ("\n".join(f"- `{x}`" for x in local) if local else "- none") + "\n\nThese entries are recorded as evidence; no local path is treated as portable without the source snapshot.\n")
    versions = {"python": sys.version, "executable": sys.executable}
    probe = """import json, platform, sys\nout={'python':sys.version,'executable':sys.executable,'platform':platform.platform()}\nfor name in ['torch','transformers','tokenizers','accelerate','numpy','scipy','sklearn']:\n try:\n  mod=__import__(name); out[name]=getattr(mod,'__version__','unknown')\n except Exception as e: out[name]='MISSING:'+type(e).__name__\ntry:\n import flash_attn; out['flash_attn']=getattr(flash_attn,'__version__','present')\nexcept Exception as e: out['flash_attn']='MISSING:'+type(e).__name__\ntry:\n import torch; out['torch_cuda_available']=bool(torch.cuda.is_available()); out['torch_cuda_version']=torch.version.cuda\nexcept Exception as e: out['torch_cuda_available']='UNKNOWN'\nprint(json.dumps(out,indent=2))\n"""
    runtime_text = run([sys.executable, "-c", probe])
    try:
        runtime_obj = json.loads(runtime_text)
    except Exception:
        runtime_obj = {"raw": runtime_text}
    write_json(env / "RUNTIME_IDENTITY.json", runtime_obj)
    write(env / "HOST_RUNTIME.txt", "".join([
        "$ which python\n", run(["which", "python"]),
        "\n$ python --version\n", run([sys.executable, "--version"]),
        "\n$ nvidia-smi\n", run(["nvidia-smi"]) if shutil.which("nvidia-smi") else "NVIDIA_SMI_NOT_FOUND\n",
        "\n$ nvidia-smi -q\n", run(["nvidia-smi", "-q"]) if shutil.which("nvidia-smi") else "NVIDIA_SMI_NOT_FOUND\n",
        "\n$ cat /etc/os-release\n", Path("/etc/os-release").read_text(encoding="utf-8") if Path("/etc/os-release").exists() else "OS_RELEASE_NOT_FOUND\n",
        "\n$ uname -a\n", run(["uname", "-a"]),
    ]))
    write_json(PORTABLE / "manifests" / "ENVIRONMENT_PROVENANCE.json", {
        "actual_python": sys.executable,
        "actual_environment": os.environ.get("CONDA_DEFAULT_ENV", ""),
        "silm_environment_present": current["silm_environment_present"],
        "runtime_identity": "environment/RUNTIME_IDENTITY.json",
        "note": "The requested silm name was absent; the actual successful runtime is frozen without renaming its provenance.",
    })


def freeze_model() -> None:
    if not MODEL.is_dir():
        raise SystemExit(f"MODEL_NOT_FOUND: {MODEL}")
    rows = []
    total = 0
    for root, dirs, names in os.walk(MODEL, followlinks=True):
        dirs.sort(); names.sort()
        for name in names:
            p = Path(root) / name
            if not p.is_file():
                continue
            rel = p.relative_to(MODEL).as_posix()
            real = p.resolve()
            size = real.stat().st_size
            total += size
            rows.append((rel, sha_file(p), size, str(real)))
    rows.sort()
    write(PORTABLE / "manifests" / "MODEL_TOKENIZER_SHA256SUMS.txt", "".join(f"{sha}  {rel}\n" for rel, sha, _, _ in rows))
    configs = {}
    for name in ["config.json", "generation_config.json", "tokenizer_config.json", "tokenizer.json", "vocab.json", "merges.txt", "model.safetensors.index.json"]:
        p = MODEL / name
        if p.exists():
            configs[name] = {"sha256": sha_file(p), "size": p.resolve().stat().st_size}
    tok = None
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(str(MODEL), local_files_only=True, trust_remote_code=True, use_fast=True)
    except Exception as exc:
        tok = None
        tok_error = f"{type(exc).__name__}: {exc}"
    identity: dict[str, Any] = {
        "model_id": "Qwen/Qwen2.5-7B-Instruct",
        "snapshot_path_at_freeze": str(MODEL),
        "revision": MODEL.name,
        "file_count": len(rows),
        "total_resolved_bytes": total,
        "weight_shards": [rel for rel, _, _, _ in rows if rel.endswith(".safetensors")],
        "configs": configs,
        "snapshot_manifest_sha256": sha_file(PORTABLE / "manifests" / "MODEL_TOKENIZER_SHA256SUMS.txt"),
    }
    if tok is not None:
        identity.update({
            "tokenizer_class": tok.__class__.__name__,
            "eos_token_id": tok.eos_token_id,
            "pad_token_id": tok.pad_token_id,
            "bos_token_id": tok.bos_token_id,
            "chat_template": tok.chat_template,
            "chat_template_sha256": sha_text(tok.chat_template or ""),
            "special_tokens_map": tok.special_tokens_map,
        })
    else:
        identity["tokenizer_load_error"] = tok_error
    write_json(PORTABLE / "manifests" / "MODEL_IDENTITY.json", identity)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def freeze_data() -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    dst = PORTABLE / "assets" / "longbench_g1_data"
    raw_rows: list[dict[str, Any]] = []
    source_maps: dict[str, dict[str, Any]] = {}
    for task in TASKS:
        src = DATA / f"{task}.jsonl"
        if not src.exists():
            raise SystemExit(f"DATASET_FILE_NOT_FOUND: {src}")
        shutil.copy2(src, dst / src.name)
        data = read_jsonl(src)
        source_maps[task] = {str(x["_id"]): x for x in data}
    golden_rows = read_jsonl(GOLDEN / "raw_predictions.jsonl")
    for task in TASKS:
        rows = [x for x in golden_rows if x["task"] == task]
        if len(rows) != COUNTS[task]:
            raise SystemExit(f"GOLDEN_TASK_COUNT_MISMATCH: {task} {len(rows)} != {COUNTS[task]}")
        for x in rows:
            sid = str(x["sample_id"]); src = source_maps[task][sid]
            raw_rows.append({
                "task": task,
                "sample_id": sid,
                "question_sha256": sha_text(str(src.get("input", ""))),
                "context_sha256": sha_text(str(src.get("context", ""))),
                "answers_sha256": canonical_hash(src.get("answers", [])),
                "source_file": f"assets/longbench_g1_data/{task}.jsonl",
                "source_row_id": sid,
            })
    write(PORTABLE / "manifests" / "DATASET_MANIFEST.csv", "")
    fields = list(raw_rows[0].keys())
    with (PORTABLE / "manifests" / "DATASET_MANIFEST.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n"); w.writeheader(); w.writerows(raw_rows)
    lines, _ = checksum_lines(dst)
    write(PORTABLE / "manifests" / "DATASET_SHA256SUMS.txt", lines)
    write_json(PORTABLE / "manifests" / "DATASET_IDENTITY.json", {"tasks": TASKS, "counts": COUNTS, "total": len(raw_rows), "source_root_at_freeze": str(DATA), "portable_root": "assets/longbench_g1_data"})
    return golden_rows, source_maps


def freeze_golden(golden_rows: list[dict[str, Any]]) -> None:
    from transformers import AutoTokenizer
    sys.path.insert(0, str(REPO))
    from longbench_eval import scorer
    tok = AutoTokenizer.from_pretrained(str(MODEL), local_files_only=True, trust_remote_code=True, use_fast=True)
    trace_rows = []
    for task in TASKS:
        trace_rows.extend(read_jsonl(GOLDEN / "packing_trace.jsonl"))
        break
    trace_by = {(x["task"], str(x["sample_id"])): x for x in trace_rows}
    records = []
    for x in golden_rows:
        t = trace_by[(x["task"], str(x["sample_id"]))]
        packed_ids = tok(str(t["packed_text"]), add_special_tokens=False).input_ids
        score = float(scorer(x["task"], [x["pred"]], [x.get("answers", [])], [int(x.get("length", 0))], x.get("all_classes")))
        records.append({
            "task": x["task"], "sample_id": str(x["sample_id"]),
            "packed_context_sha256": sha_text(str(t["packed_text"])),
            "packed_token_ids": list(packed_ids), "packed_token_ids_sha256": canonical_hash(list(packed_ids)),
            "generation_prompt_sha256": sha_text(str(t["generation_prompt"])),
            "input_ids": list(x["input_token_ids"]), "input_ids_sha256": canonical_hash(list(x["input_token_ids"])),
            "output_ids": list(x["prediction_token_ids"]), "output_ids_sha256": canonical_hash(list(x["prediction_token_ids"])),
            "decoded_prediction": str(x["pred"]), "decoded_prediction_sha256": sha_text(str(x["pred"])),
            "score": score, "packed_token_count": len(packed_ids), "input_token_count": len(x["input_token_ids"]), "output_token_count": len(x["prediction_token_ids"]),
        })
    if len(records) != 1150:
        raise SystemExit(f"GOLDEN_ROW_COUNT_MISMATCH: {len(records)}")
    out = PORTABLE / "golden" / "GOLDEN_REFERENCE_V1.jsonl.gz"
    with gzip.open(out, "wt", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n")
    fields = ["task", "sample_id", "packed_token_ids_sha256", "input_ids_sha256", "output_ids_sha256", "decoded_prediction_sha256", "score"]
    with (PORTABLE / "golden" / "GOLDEN_FINGERPRINT_V1.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n"); w.writeheader()
        w.writerows({field: row.get(field, "") for field in fields} for row in records)
    shutil.copy2(GOLDEN / "per_task_scores.csv", PORTABLE / "golden" / "GOLDEN_PER_TASK.csv")
    summary = json.loads((GOLDEN / "macro_summary.json").read_text(encoding="utf-8"))
    summary.update({"golden_reference": "Portable Golden V1", "rows": len(records), "packed_token_ids_saved": True, "input_ids_saved": True, "output_ids_saved": True, "historical_token_ids_reconstructed": False})
    write_json(PORTABLE / "golden" / "GOLDEN_SUMMARY.json", summary)
    write(PORTABLE / "golden" / "GOLDEN_REFERENCE_PROVENANCE.md", """# Portable Golden Reference V1\n\nThis is the 2026-09-28 fresh reproduction reference.\nIt is NOT a reconstruction of missing historical G1 token IDs.\n\nThe reference contains 1,150 rows with fresh-run packed token IDs, final input IDs, output IDs, decoded predictions, and per-sample scores. Its target macro is 39.35166666666667 (reported as 39.3517).\n\nHistorical relation: the fresh run agreed with the historical decoded predictions for 1,141/1,150 rows; historical token-level references were not saved.\n""")
    lines, _ = checksum_lines(PORTABLE / "golden")
    write(PORTABLE / "manifests" / "GOLDEN_SHA256SUMS.txt", lines)


def freeze_generation_config() -> None:
    cfg = json.loads((REPO / "configs" / "dataset2maxlen.json").read_text(encoding="utf-8"))
    tok_info = json.loads((PORTABLE / "manifests" / "MODEL_IDENTITY.json").read_text(encoding="utf-8"))
    runtime = json.loads((PORTABLE / "environment" / "RUNTIME_IDENTITY.json").read_text(encoding="utf-8"))
    attention = "flash_attention_2_if_importable_else_sdpa"
    try:
        import flash_attn  # noqa: F401
        attention = "flash_attention_2"
    except Exception:
        attention = "sdpa"
    write_json(PORTABLE / "manifests" / "GENERATION_CONFIG.json", {
        "model": "Qwen/Qwen2.5-7B-Instruct", "snapshot_revision": tok_info.get("revision"), "dtype": "float16", "batch_size": 1,
        "do_sample": False, "num_beams": 1, "temperature": None, "top_p": None, "use_cache": True,
        "max_new_tokens_per_task": {t: int(cfg[t]) for t in TASKS}, "eos_token_id": tok_info.get("eos_token_id"), "pad_token_id": tok_info.get("pad_token_id"), "bos_token_id": tok_info.get("bos_token_id"),
        "attention_backend": attention, "chat_template_usage": "tokenizer.apply_chat_template(add_generation_prompt=True)",
        "question_binding": "historical fresh G1 behavior retained: packed context contains Question: q and LongBench wrapper also inserts q",
        "budget_scope": "packed context only; B=1024", "prompt_wrapper": "configs/dataset2prompt.json", "runtime_identity": runtime,
    })


def main() -> int:
    for p in [PORTABLE / x for x in ["source", "environment", "manifests", "golden", "scripts", "image", "reports", "assets", "portable_validation"]]:
        p.mkdir(parents=True, exist_ok=True)
    freeze_git(); freeze_source(); freeze_environment(); freeze_model(); golden_rows, _ = freeze_data(); freeze_golden(golden_rows); freeze_generation_config()
    write_json(PORTABLE / "manifests" / "FREEZE_STATUS.json", {"source": True, "environment": True, "model": True, "dataset": True, "golden": True, "golden_rows": len(golden_rows), "golden_macro": 39.35166666666667, "historical_token_ids_used": False})
    print(json.dumps({"portable_root": str(PORTABLE), "golden_rows": len(golden_rows), "golden_macro": 39.35166666666667}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
