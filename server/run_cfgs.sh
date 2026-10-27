#!/usr/bin/env bash
# usage: GPU=n run_cfgs.sh name=json [name=json ...]
set -uo pipefail
W=/diskE/jiaqizhang/BRIDGEPACK_SENSITIVITY_V1_20261025
PY=$W/runtime/silm/bin/python
MODEL=/diskF/jiaqizhang/baselines/RepairKV/models/Qwen2.5-7B-Instruct
export CUDA_VISIBLE_DEVICES=$GPU
for kv in "$@"; do
  name=${kv%%=*}; cfg=${kv#*=}; out=$W/runs/qwen/$name
  if [[ -f $out/DONE ]]; then echo "skip $name"; continue; fi
  echo "[$(date -Is)] start $name $cfg gpu=$GPU"
  BP_SENS_JSON="$cfg" $PY $W/scripts/portable_runner.py --model-path $MODEL --data-root $W/assets/longbench_g1_data --golden $W/golden --output-dir $out > $W/runs/qwen/$name.log 2>&1 < /dev/null
  rc=$?; echo "$cfg" > $out/SENS_PARAMS.json; [[ $rc -eq 0 ]] && touch $out/DONE
  echo "[$(date -Is)] end $name rc=$rc"
done
echo QUEUE_DONE
