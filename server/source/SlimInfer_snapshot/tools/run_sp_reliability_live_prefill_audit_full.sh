#!/usr/bin/env bash
# Safe sequential live prefill audit: one method at a time, resume-friendly.
set -euo pipefail
cd "$(dirname "$0")/.."
OUT_DIR="${1:-results/sp_reliability_live_prefill_audit_20260710}"
export PYTHONPATH=.

mkdir -p "$OUT_DIR"
echo "[live-audit-full] out_dir=$OUT_DIR" | tee "$OUT_DIR/run_all.log"

for METHOD in sliminfer_original sp_sliminfer rank_stability_sliminfer; do
  echo "[live-audit-full] starting $METHOD at $(date -Is)" | tee -a "$OUT_DIR/run_all.log"
  set +e
  .venv/bin/python tools/run_sp_reliability_live_prefill_audit.py \
    --out_dir "$OUT_DIR" \
    --samples_per_task 200 \
    --methods "$METHOD" \
    --resume 2>&1 | tee -a "$OUT_DIR/run_all_${METHOD}.log"
  RC=$?
  set -e
  if [[ $RC -ne 0 ]]; then
    echo "[live-audit-full] WARNING $METHOD exited rc=$RC (continuing)" | tee -a "$OUT_DIR/run_all.log"
  fi
  echo "[live-audit-full] finished $METHOD at $(date -Is)" | tee -a "$OUT_DIR/run_all.log"
done

echo "[live-audit-full] analyzing at $(date -Is)" | tee -a "$OUT_DIR/run_all.log"
.venv/bin/python tools/run_sp_reliability_live_prefill_audit.py \
  --out_dir "$OUT_DIR" \
  --samples_per_task 200 \
  --analyze_only 2>&1 | tee -a "$OUT_DIR/run_all_analyze.log"

echo "[live-audit-full] done at $(date -Is)" | tee -a "$OUT_DIR/run_all.log"
