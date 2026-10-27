#!/usr/bin/env bash
# Safe sequential 400-sample two-task QA (resume-friendly).
set -euo pipefail
cd "$(dirname "$0")/.."
OUT_DIR="${1:-results/sp_reliability_2task_compare_20260710}"
export PYTHONPATH=.

mkdir -p "$OUT_DIR"
echo "[qa-full] out_dir=$OUT_DIR" | tee "$OUT_DIR/run_all.log"

for METHOD in sliminfer_original sp_sliminfer rank_stability_sliminfer; do
  echo "[qa-full] starting $METHOD at $(date -Is)" | tee -a "$OUT_DIR/run_all.log"
  set +e
  .venv/bin/python tools/run_sp_reliability_2task_qa_compare.py \
    --out_dir "$OUT_DIR" \
    --methods "$METHOD" 2>&1 | tee -a "$OUT_DIR/run_all_${METHOD}.log"
  RC=$?
  set -e
  if [[ $RC -ne 0 ]]; then
    echo "[qa-full] WARNING $METHOD exited rc=$RC (continuing if partial)" | tee -a "$OUT_DIR/run_all.log"
  fi
  echo "[qa-full] finished $METHOD at $(date -Is)" | tee -a "$OUT_DIR/run_all.log"
done

echo "[qa-full] final aggregate at $(date -Is)" | tee -a "$OUT_DIR/run_all.log"
.venv/bin/python tools/run_sp_reliability_2task_qa_compare.py \
  --out_dir "$OUT_DIR" \
  --skip_run 2>&1 | tee -a "$OUT_DIR/run_all_aggregate.log"

echo "[qa-full] done at $(date -Is)" | tee -a "$OUT_DIR/run_all.log"
