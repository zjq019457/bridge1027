#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${BRIDGEPACK_PYTHON:-$ROOT/runtime/silm/bin/python}"
MODEL_DIR="${MODEL_DIR:-${BRIDGEPACK_MODEL_PATH:-/mnt/model}}"
DATA_DIR="${DATA_DIR:-$ROOT/assets/longbench_g1_data}"
OUT="${1:-$ROOT/portable_validation/12_sample_gate}"
"$PYTHON_BIN" "$ROOT/scripts/portable_runner.py" --model-path "$MODEL_DIR" --data-root "$DATA_DIR" --golden "$ROOT/golden" --output-dir "$OUT" --sample-limit 12
"$PYTHON_BIN" "$ROOT/scripts/compare_with_golden.py" --run-dir "$OUT" --golden "$ROOT/golden"
