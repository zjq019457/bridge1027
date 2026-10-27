#!/usr/bin/env bash
set -uo pipefail
W=/diskE/jiaqizhang/BRIDGEPACK_SENSITIVITY_V1_20261025
PY=$W/runtime/silm/bin/python
MODEL=/diskF/jiaqizhang/baselines/RepairKV/models/Qwen2.5-7B-Instruct
export CUDA_VISIBLE_DEVICES=${GPU:-2}
mkdir -p $W/runs/qwen $W/provenance
{ date -Is; hostname; nvidia-smi -q | grep -E "Product Name|GPU UUID|Driver Version|CUDA Version"; uname -a; $PY -m pip freeze; } > $W/provenance/qwen_removals_env.txt 2>&1
declare -A CFG=(
  [default]="{}"
  [c_div0]="{\"c_div\": 0.0}"
  [c_jac0]="{\"c_jac\": 0.0}"
  [bridge_zero]="{\"bridge_zero\": true}"
  [c_sup0]="{\"c_sup\": 0.0}"
  [lam0]="{\"lam\": 0.0}"
  [neighbor0]="{\"neighbor\": 0}"
)
for name in default c_div0 c_jac0 bridge_zero c_sup0 lam0 neighbor0; do
  out=$W/runs/qwen/$name
  if [[ -f $out/DONE ]]; then echo "skip $name"; continue; fi
  echo "[$(date -Is)] start $name ${CFG[$name]}"
  BP_SENS_JSON="${CFG[$name]}" $PY $W/scripts/portable_runner.py --model-path $MODEL --data-root $W/assets/longbench_g1_data --golden $W/golden --output-dir $out > $W/runs/qwen/$name.log 2>&1
  rc=$?
  echo "${CFG[$name]}" > $out/SENS_PARAMS.json
  if [[ $rc -eq 0 ]]; then touch $out/DONE; fi
  echo "[$(date -Is)] end $name rc=$rc"
done
$PY $W/scripts/compare_with_golden.py --run-dir $W/runs/qwen/default --golden $W/golden > $W/runs/qwen/default_vs_golden.json 2>&1
echo ALL_DONE
