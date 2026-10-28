#!/usr/bin/env bash
# pack one ablation / dense-BP cell, then enqueue its generation
# usage: a100_ctrl_one.sh MODEL KIND BUDGET   (KIND: ctrl_pack.py kinds, or bge_{recall,union,rerank}_bp)
cd /diskE/jiaqizhang/BRIDGEPACK_SENSITIVITY_V1_20261025
M=$1; K=$2; B=$3
case $M in qwen) T=/diskF/jiaqizhang/baselines/RepairKV/models/Qwen2.5-7B-Instruct;; llama) T=/diskD/jiaqizhang/models/Meta-Llama-3-8B-Instruct;; esac
N=${M}_${K}_${B}; OUT=runs/a100/$N.jsonl
[ -f runs/a100/${N}_qa/macro_summary.json ] && exit 0
if [ "$(wc -l <$OUT 2>/dev/null)" != 1150 ]; then
  case $K in
    bge_*_bp) G=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -t, -k2 -nr | head -1 | cut -d, -f1)
              M2=${K#bge_}; until CUDA_VISIBLE_DEVICES=$G runtime/silm/bin/python dense_bp_pack.py ${M2%_bp} $T $OUT $B; do sleep 60; done;;
    *) runtime/silm/bin/python ctrl_pack.py $K $T $OUT $B || exit 1;;
  esac
fi
printf "%s\t%s\n" $N "runtime/silm/bin/python gen_runner.py --trace $OUT --model $T --out runs/a100/${N}_qa" >> queue/jobs.txt
