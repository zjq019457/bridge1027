#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_DIR="${MODEL_DIR:-${BRIDGEPACK_MODEL_PATH:-/mnt/model}}"
PYTHON_BIN="${BRIDGEPACK_PYTHON:-$ROOT/runtime/silm/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then echo "PORTABLE_RUNTIME_MISSING: $PYTHON_BIN" >&2; exit 1; fi
exec "$PYTHON_BIN" "$ROOT/scripts/verify_assets.py" --model-dir "$MODEL_DIR"
