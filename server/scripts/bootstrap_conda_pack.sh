#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ARCHIVE="${1:-$ROOT/environment/silm_exact.tar.gz}"
TARGET="${2:-$ROOT/runtime/silm}"
mkdir -p "$TARGET"
tar -xzf "$ARCHIVE" -C "$TARGET"
"$TARGET/bin/conda-unpack"
echo "PORTABLE_RUNTIME_READY $TARGET"
