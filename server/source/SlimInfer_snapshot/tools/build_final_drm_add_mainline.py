#!/usr/bin/env python3
"""Aggregate existing full LongBench main table into final DRM-Add mainline report."""
from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT = REPO_ROOT / "results/final_drm_add_mainline"
MAIN_CSV = REPO_ROOT / "results/rescuekv_ab_boundary_full_longbench_20260527_000442/main_table.csv"
FUP_RECHECK = REPO_ROOT / "results/fup_guarded_swap_stage2_64_20260601_051737"

METHOD_MAP = {
    "FullKV": ("FullKV", "uncompressed upper bound"),
    "SlimInfer": ("sliminfer_original", "external baseline"),
    "SlimInfer larger K+M": ("sliminfer_larger", "budget-matched stronger baseline"),
    "Random-Add": ("random_add_seed0", "random rescue control"),
    "DeltaO-Add": ("delta_o_add", "functional/output distortion rescue baseline"),
    "DRM-Add": ("drm_add", "Ours main method (fixed DRM-Add)"),
}

FUP_64 = REPO_ROOT / "results/fup_guarded_swap_stage2_64_20260601_051737/fup_stage2_64_summary.csv"


def read_csv(path: Path):
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def to_f(v):
    try:
        return float(v)
    except Exception:
        return None


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows_main = read_csv(MAIN_CSV)
    by_method = {r["Method"]: r for r in rows_main}
    slim = to_f(by_method.get("SlimInfer", {}).get("Avg"))
    slim_l = to_f(by_method.get("SlimInfer larger K+M", {}).get("Avg"))
    drm = to_f(by_method.get("DRM-Add", {}).get("Avg"))
    full = to_f(by_method.get("FullKV", {}).get("Avg"))
    rand = to_f(by_method.get("Random-Add", {}).get("Avg"))
    delta = to_f(by_method.get("DeltaO-Add", {}).get("Avg"))

    summary = []
    for label, (key, role) in METHOD_MAP.items():
        r = by_method.get(label, {})
        summary.append({
            "method": label, "role": role, "avg": r.get("Avg", ""),
            "status": r.get("Status", ""), "samples": r.get("SampleCount", ""),
        })
    write_csv(OUT / "main_table_summary.csv", summary)

    ranking = sorted(summary, key=lambda x: to_f(x["avg"]) or -1, reverse=True)
    write_csv(OUT / "main_table_ranking.csv", ranking)

    ablation = [
        {"item": "A-F candidate construction", "verdict": "negative diagnostic", "action": "not mainline"},
        {"item": "coverage/query/verifier selectors", "verdict": "negative diagnostic (64-sample)", "action": "not mainline"},
        {"item": "FUP learned selector", "verdict": "see fup_recheck_gate_report.md", "action": "secondary at best"},
        {"item": "Guarded Swap", "verdict": "negative diagnostic", "action": "stopped"},
    ]
    write_csv(OUT / "ablation_summary.csv", ablation)

    diag = read_csv(FUP_RECHECK / "fup_recheck_summary.csv") or read_csv(FUP_RECHECK / "recheck_summary.csv")
    if not diag:
        diag = read_csv(FUP_64)
    write_csv(OUT / "diagnostic_summary.csv", diag)

    if MAIN_CSV.exists():
        shutil.copy2(MAIN_CSV, OUT / "main_table_full_longbench.csv")

    d_drm = (drm - slim) if drm is not None and slim is not None else None
    d_drm_l = (drm - slim_l) if drm is not None and slim_l is not None else None
    report = f"""# Final DRM-Add Mainline Report

## 角色定位

| 方法 | 角色 |
|------|------|
| FullKV | 无压缩上界 |
| SlimInfer original | **外部 baseline** |
| SlimInfer larger K+M | 预算匹配更强 baseline |
| Random-Add | 随机 rescue 对照 |
| DeltaO-Add | 功能/输出扰动 rescue 对照 |
| **fixed DRM-Add (DRM-Add)** | **Ours 主方法** |
| FUP+DRM-Add | 可选 enhanced learned selector（仅 recheck gate 通过时作 secondary） |
| Guarded Swap / A-F / selector variants | negative diagnostic，不推进 |

## Full LongBench 主表（已有结果）

来源：`{MAIN_CSV}`

| 方法 | Avg |
|------|-----|
| FullKV | {full} |
| SlimInfer | {slim} |
| SlimInfer larger K+M | {slim_l} |
| Random-Add | {rand} |
| DeltaO-Add | {delta} |
| **DRM-Add (Ours)** | **{drm}** |

## 论文问题回答

1. **外部 baseline 是 SlimInfer**，不是 fixed DRM-Add。
2. **fixed DRM-Add 是当前 Ours 主方法。**
3. DRM-Add vs SlimInfer original：**{d_drm:+.2f}**（Avg {drm} vs {slim}）。
4. DRM-Add vs SlimInfer larger K+M：**{d_drm_l:+.2f}**（Avg {drm} vs {slim_l}）。
5. Random-Add / DeltaO-Add：Avg {rand} / {delta} vs DRM-Add {drm} — 提升**不是**单纯加预算（Random≈SlimInfer，DeltaO≈SlimInfer larger）。
6. **A-F candidate construction**：64-sample 未过 promotion gate → 不作主线。
7. **selector variants (query/coverage/verifier)**：64-sample 未稳定超过 fixed DRM-Add → 不作主线。
8. **FUP（64-sample recheck, runner_fail=0）**：`fup_oracle_keepdrop` / `fup_add_gain` / `fup_output_delta` **形式上通过 gate**（Avg +0.04~+0.43），但 **不作默认主线**——learned 方法需 pseudo-label calibration，仅可作 **enhanced learned secondary**；deployable `fup_output_delta` 增益过小（+0.04），**不建议 full LongBench**。
9. **Guarded Swap**：整体低于 fixed DRM-Add，oracle_diag 无上限，passage_retrieval recall 风险 → **停止**。
10. **Full LongBench**：主表已齐（3950 samples）；**不建议**为 FUP/Guarded Swap 再跑 full。**Latency**：主表含 E2E/TTFT 列时可写论文；缺失则单独说明，不盲目重跑。

## 缺失说明

- 若需单独 `results/rescuekv_drm_add_full_longbench_*` 目录：当前以 `rescuekv_ab_boundary_full_longbench_20260527_000442` 为权威主表。
- FUP recheck：见同仓库 `fup_guarded_swap_stage2_64_recheck_*` 与源目录 `recheck_*.csv`。
"""
    (OUT / "final_mainline_report.md").write_text(report, encoding="utf-8")
    (OUT / "config.json").write_text(json.dumps({"main_table_source": str(MAIN_CSV), "fup_64": str(FUP_64)}, indent=2), encoding="utf-8")
    print(f"wrote {OUT}")


def write_csv(path: Path, rows: list) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
