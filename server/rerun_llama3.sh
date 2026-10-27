#!/usr/bin/env bash
run(){ n=$1; cfg=$2; g=$3
  for i in $(seq 1 600); do f=$(nvidia-smi --id=$g --query-gpu=memory.free --format=csv,noheader,nounits|tr -d " "); [ "$f" -ge 20000 ] && break; sleep 30; done
  rm -rf /diskE/jiaqizhang/BRIDGEPACK_SENSITIVITY_V1_20261025/runs/llama/$n
  echo "[$(date -Is)] start $n gpu=$g"
  CUDA_VISIBLE_DEVICES=$g BP_SENS_JSON="$cfg" /diskE/jiaqizhang/BRIDGEPACK_SENSITIVITY_V1_20261025/runtime/silm/bin/python /diskE/jiaqizhang/BRIDGEPACK_SENSITIVITY_V1_20261025/scripts/portable_runner.py --model-path /diskD/jiaqizhang/models/Meta-Llama-3-8B-Instruct --data-root /diskE/jiaqizhang/BRIDGEPACK_SENSITIVITY_V1_20261025/assets/longbench_g1_data --golden /diskE/jiaqizhang/BRIDGEPACK_SENSITIVITY_V1_20261025/golden --output-dir /diskE/jiaqizhang/BRIDGEPACK_SENSITIVITY_V1_20261025/runs/llama/$n --packing-only > /diskE/jiaqizhang/BRIDGEPACK_SENSITIVITY_V1_20261025/queue/rerun_$n.log 2>&1 && CUDA_VISIBLE_DEVICES=$g /diskE/jiaqizhang/BRIDGEPACK_SENSITIVITY_V1_20261025/runtime/silm/bin/python /diskE/jiaqizhang/BRIDGEPACK_SENSITIVITY_V1_20261025/gen_runner.py --trace /diskE/jiaqizhang/BRIDGEPACK_SENSITIVITY_V1_20261025/runs/llama/$n/packing_trace.jsonl --model /diskD/jiaqizhang/models/Meta-Llama-3-8B-Instruct --out /diskE/jiaqizhang/BRIDGEPACK_SENSITIVITY_V1_20261025/runs/llama/$n/qa >> /diskE/jiaqizhang/BRIDGEPACK_SENSITIVITY_V1_20261025/queue/rerun_$n.log 2>&1
  echo "[$(date -Is)] end $n rc=$?"; }
run csup025 "{\"c_sup\": 0.25}" 0 &
run lam070 "{\"lam\": 0.70}" 1 &
run neighbor16 "{\"neighbor\": 16}" 2 &
wait; echo RERUN_DONE
