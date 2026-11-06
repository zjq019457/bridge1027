#!/bin/bash
# Round 3: external baselines (LlamaIndex window/auto-merge, bge-reranker-v2-m3, Provence), LongBench 1024.
export PATH=/root/miniconda3/bin:$PATH
R=/root/autodl-tmp; cd $R/bridge1027/research; export BP_ROOT=$R
LM=$(cat $R/path_llama); QM=$R/models/models/Qwen--Qwen2.5-14B-Instruct/snapshots/master; QT=$PWD/data/tok/qwen
export BGE_PATH=$(cat $R/path_bge) RR_PATH=/root/models/models/BAAI--bge-reranker-v2-m3/snapshots/master PV_PATH=/root/models/provence
T=qasper,multifieldqa_en,hotpotqa,2wikimqa,musique,narrativeqa
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a logs/progress3.txt; }
mkdir -p run3_qwen/out run3_llama/out
log start
for m in window automerge provence rerank; do CUDA_VISIBLE_DEVICES=0 python extbase.py $m run3_qwen:$QT run3_llama:$LM > logs/ext_$m.log 2>&1; log $m done; done
for d in run3_qwen:$QT run3_llama:$LM; do D=${d%%:*}; TK=${d#*:}
  ( cd $D; BP_TOK=$TK BP_ENC=rr python ../evalcov.py bge_chunk $T 100000 budget=1024
    BP_TOK=$TK BP_ENC=rr python ../evalcov.py lex_rep $T 100000 seed_mode=1 sent=1 budget=1024 ) > logs/rrpack_$D.log 2>&1 &
done; wait; touch run3_qwen/PACK_DONE run3_llama/PACK_DONE; log rerank packed
until grep -q ALL2_DONE logs/progress2.txt; do sleep 30; done
CUDA_VISIBLE_DEVICES=0 python gen_stream2.py $QM run3_qwen 0.9 > logs/s3_qwen.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 python gen_stream2.py $LM run3_llama 0.9 > logs/s3_llama.log 2>&1 &
wait; log ALL3_DONE
