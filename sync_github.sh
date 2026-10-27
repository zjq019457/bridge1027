#!/usr/bin/env bash
# Pull code + result summaries from the A100 work dir and push them to GitHub.
# Packed traces, datasets, models and the conda runtime are left out (size limits).
set -e
R=/d/BridgePack_backup_20261027/bridge1027
W=/diskE/jiaqizhang/BRIDGEPACK_SENSITIVITY_V1_20261025
mkdir -p "$R/server"
ssh csuf "cd $W && find . \( -path ./runtime -o -path ./assets -o -path ./babilong -o -path ./sf_cache -o -path ./tiktoken_cache \
  -o -path './models*' -o -path './extra_pkgs*' -o -name __pycache__ \) -prune -o -type f \( \
  -name '*.py' -o -name '*.sh' -o -name '*.md' -o -name '*.diff' -o -name 'requirements*.txt' -o -name 'SHA256SUMS.txt' \
  -o -name 'macro_summary.json' -o -name 'run_manifest.json' -o -name 'raw_predictions.jsonl' -o -name '*.log' \
  -o -path './analysis/*' -o -path './golden/*' -o -path './provenance/*' -o -path './manifests/*' -o -path './source/*' \) \
  -size -50M -print0 | tar --null -T - -czf -" | tar -xzf - -C "$R/server"
cp -r /c/Users/dell/AppData/Roaming/Claude/scratch-workspaces/f9dfb929-41d3-4ffa-93e7-95462504c1c8/8d53d6e0-c863-4dd1-a4ff-93b70b19c861/scratch-2026-10-24-bf2c03/rebuttal "$R/" 2>/dev/null || true
cd "$R"
git add -A
git commit -q -m "sync $(date -Is)" || { echo "nothing new"; exit 0; }
git push -q origin main
echo "pushed $(git rev-parse --short HEAD)"
