#!/bin/bash
# Build-up ablation with a fixed BGE-small ranking (seed_mode=1), plus merge-only at 512/2048,
# then Qwen-14B InfiniteBench full context (TP=2).
export PATH=/root/miniconda3/bin:$PATH
R=/root/autodl-tmp; cd $R/bridge1027/research; export BP_ROOT=$R
LM=$(cat $R/path_llama); QM=$R/models/models/Qwen--Qwen2.5-14B-Instruct/snapshots/master; QT=$PWD/data/tok/qwen
T=qasper,multifieldqa_en,hotpotqa,2wikimqa,musique,narrativeqa
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a logs/progress4.txt; }
mkdir -p run4_qwen/out run4_llama/out
log start
lane(){ D=$1; TK=$2; cd $D
  for cfg in "sent=0 merge=0 docorder=0 backfill=1" "sent=0 merge=1 docorder=0 backfill=1" "sent=0 merge=1 docorder=1 backfill=1" "sent=1 merge=1 docorder=1 backfill=0" "sent=1 merge=0 docorder=1 backfill=1"; do
    BP_TOK=$TK python ../evalcov.py final_abl $T 100000 seed_mode=1 $cfg budget=1024
  done
  for b in 512 2048; do for cfg in "sent=0 merge=0 docorder=0 backfill=1" "sent=0 merge=1 docorder=0 backfill=1"; do
    BP_TOK=$TK python ../evalcov.py final_abl $T 100000 seed_mode=1 $cfg budget=$b
  done; done; cd ..; touch $D/PACK_DONE; }
lane run4_qwen $QT > logs/pack4q.log 2>&1 &
lane run4_llama $LM > logs/pack4l.log 2>&1 &
CUDA_VISIBLE_DEVICES=0 python gen_stream2.py $QM run4_qwen 0.9 > logs/s4_qwen.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 python gen_stream2.py $LM run4_llama 0.9 > logs/s4_llama.log 2>&1 &
wait; log gen done
TP=2 CUDA_VISIBLE_DEVICES=0,1 python colab_gen.py --model $QM --out run2_qwen/preds --max_model_len 32768 run2_qwen/fullinf.jsonl > logs/fi_qwen3.log 2>&1
log ALL4_DONE
