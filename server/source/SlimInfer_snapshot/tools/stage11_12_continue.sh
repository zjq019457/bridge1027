#!/usr/bin/env bash
set -euo pipefail

REPO=/home/yu-bai/Desktop/jiaqizhang/SlimInfer
PIPELINE="$REPO/results/qwen_unified_pipeline_20260720_212437"
POLICY="$REPO/configs/qwen_machine_safety_policy.json"
UPSTREAM=${UPSTREAM_UNIT:-qwen-unified-stage9-10-resume3-20260722.service}
LOG="$PIPELINE/STAGE_11_12_CONTINUATION.log"

exec >>"$LOG" 2>&1
printf '[%s] continuation service started; waiting for %s\n' "$(date --iso-8601=seconds)" "$UPSTREAM"
while systemctl --user is-active --quiet "$UPSTREAM"; do
    printf '[%s] upstream still active\n' "$(date --iso-8601=seconds)"
    sleep 60
done

printf '[%s] upstream inactive; allowing GPU/process state to settle\n' "$(date --iso-8601=seconds)"
sleep 30

"$REPO/.venv/bin/python" - "$PIPELINE" <<'PY'
import json
import sys
from pathlib import Path

pipeline = Path(sys.argv[1])
for stage in (9, 10):
    path = pipeline / f"STAGE_{stage:02d}_DECISION.json"
    if not path.exists():
        raise SystemExit(f"Stage {stage} decision is absent: {path}")
    decision = json.loads(path.read_text(encoding="utf-8"))
    if decision.get("decision") != "PASS":
        raise SystemExit(f"Stage {stage} is not PASS: {decision.get('decision')}")
print("Stage 9 and Stage 10 are PASS; Stage 11 may start.")
PY

printf '[%s] starting Stage 11 equivalence-gated OOM fill\n' "$(date --iso-8601=seconds)"
"$REPO/.venv/bin/python" "$REPO/tools/run_qwen_oom_fill_v2.py" \
    --pipeline-dir "$PIPELINE" \
    --safety-policy "$POLICY" \
    --execute

printf '[%s] starting Stage 12 unified export\n' "$(date --iso-8601=seconds)"
"$REPO/.venv/bin/python" "$REPO/tools/run_qwen_unified_pipeline.py" \
    --pipeline-dir "$PIPELINE" \
    --safety-policy "$POLICY" \
    --start-stage 12 \
    --stop-after-stage 12

printf '[%s] Stage 11/12 continuation complete\n' "$(date --iso-8601=seconds)"
