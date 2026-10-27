#!/usr/bin/env bash
W=/diskE/jiaqizhang/BRIDGEPACK_SENSITIVITY_V1_20261025
for i in $(seq 1 240); do
  for g in 1 2 0; do
    free=$(nvidia-smi --id=$g --query-gpu=memory.free --format=csv,noheader,nounits | tr -d " ")
    if [ "$free" -ge 20000 ]; then
      echo "[$(date -Is)] start lingua QA gpu=$g"
      CUDA_VISIBLE_DEVICES=$g $W/runtime/silm/bin/python $W/scripts/portable_runner.py --model-path /diskF/jiaqizhang/baselines/RepairKV/models/Qwen2.5-7B-Instruct --data-root $W/assets/longbench_g1_data --golden $W/golden --output-dir $W/runs/lingua/full --qa-only > $W/runs/lingua/full_qa.log 2>&1 < /dev/null
      echo "[$(date -Is)] end lingua QA rc=$?"; echo LINGUA_QA_DONE; exit
    fi
  done
  sleep 60
done
echo LINGUA_QA_GAVE_UP
