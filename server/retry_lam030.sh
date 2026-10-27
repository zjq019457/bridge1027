#!/usr/bin/env bash
W=/diskE/jiaqizhang/BRIDGEPACK_SENSITIVITY_V1_20261025
for i in $(seq 1 240); do
  for g in 1 0 2; do
    free=$(nvidia-smi --id=$g --query-gpu=memory.free --format=csv,noheader,nounits | tr -d " ")
    if [ "$free" -ge 20000 ]; then GPU=$g $W/run_cfgs.sh "lam030={\"lam\": 0.30}"; exit; fi
  done
  sleep 60
done
echo RETRY_GAVE_UP
