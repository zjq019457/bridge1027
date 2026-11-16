#!/bin/bash
# LongContextReorder (BGE-small chunks, relevance-ordered to both ends) at 512/1024/2048, Llama-8B and Qwen-14B (after run12).
export PATH=/root/miniconda3/bin:$PATH
R=/root/autodl-tmp; cd $R/bridge1027/research; export BP_ROOT=$R
LM=$R/models/models/LLM-Research--Meta-Llama-3.1-8B-Instruct/snapshots/master
QT=$PWD/data/tok/qwen
T=qasper,multifieldqa_en,hotpotqa,2wikimqa,musique,narrativeqa
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a logs/progress13.txt; }
mkdir -p run13_qwen/out run13_llama/out
log start
for d in run13_qwen:$QT run13_llama:$LM; do D=${d%%:*}; TK=${d#*:}
  ( cd $D; for b in 512 1024 2048; do BP_TOK=$TK python ../evalcov.py lcr_chunk $T 100000 budget=$b & done; wait ) > logs/pack13_$D.log 2>&1
done
log packed
until grep -q ALL12_DONE logs/progress12.txt; do sleep 60; done; log "run12 done"
touch run13_qwen/PACK_DONE run13_llama/PACK_DONE
CUDA_VISIBLE_DEVICES=0 python gen_stream2.py $LM run13_llama 0.9 > logs/s13_llama.log 2>&1; log "llama done"
QM=$(grep '^DONE' logs/dl14.log | awk '{print $2}')
CUDA_VISIBLE_DEVICES=0 python gen_stream2.py $QM run13_qwen 0.9 > logs/s13_qwen.log 2>&1
log ALL13_DONE
