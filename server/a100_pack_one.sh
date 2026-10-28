#!/usr/bin/env bash
# pack one (model, variant, budget) cell for the all-A100 rerun, then enqueue its generation
# usage: a100_pack_one.sh MODEL VARIANT BUDGET
cd /diskE/jiaqizhang/BRIDGEPACK_SENSITIVITY_V1_20261025
M=$1; V=$2; B=$3
[ $B = 256 ] && exit 0  # 256 dropped from the rerun
case $M in qwen) T=/diskF/jiaqizhang/baselines/RepairKV/models/Qwen2.5-7B-Instruct;; llama) T=/diskD/jiaqizhang/models/Meta-Llama-3-8B-Instruct;; esac
N=${M}_${V}_${B}; OUT=runs/a100/$N.jsonl
[ -f runs/a100/${N}_qa/macro_summary.json ] && exit 0
if [ "$(wc -l <$OUT 2>/dev/null)" != 1150 ]; then
  if [ $V = bge ]; then runtime/silm/bin/python bge_pack.py $T $OUT $B; else runtime/silm/bin/python lex_pack.py $V $T $OUT $B; fi || exit 1
fi
printf "%s\t%s\n" $N "runtime/silm/bin/python gen_runner.py --trace $OUT --model $T --out runs/a100/${N}_qa" >> queue/jobs.txt
