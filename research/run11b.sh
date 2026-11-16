#!/bin/bash
# Redo of run11's failed parts: Provence (nltk data), BGE-large / reranker packs (name clash), LongLLMLingua (transformers cache API), then 32B generation.
export PATH=/root/miniconda3/bin:$PATH
R=/root/autodl-tmp; cd $R/bridge1027/research; export BP_ROOT=$R
LM=$R/models/models/LLM-Research--Meta-Llama-3.1-8B-Instruct/snapshots/master
QT=$PWD/data/tok/qwen
T=qasper,multifieldqa_en,hotpotqa,2wikimqa,musique,narrativeqa
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a logs/progress11.txt; }
log "run11b start"
BUDGET=1024 python extbase.py provence run11_q32:$QT > logs/ext11_provence.log 2>&1; log "provence packed: $(ls run11_q32/out | grep -c provence)"
cd run11_q32; rm -f out/bge_chunk_*.jsonl out/lex_rep_*.jsonl; mkdir -p tmp/out
for e in bgel rr; do
  ( cd tmp; BP_TOK=$QT BP_ENC=$e python ../../evalcov.py bge_chunk $T 100000 budget=1024 ) &
  ( cd tmp; BP_TOK=$QT BP_ENC=$e python ../../evalcov.py lex_rep $T 100000 seed_mode=1 sent=1 budget=1024 ) &
  wait; for x in tmp/out/*.jsonl; do mv "$x" "out/${e}_$(basename $x)"; done
done > ../logs/pack11b.log 2>&1; cd ..
log "rankers packed: $(ls run11_q32/out | grep -c -E '^(bgel|rr)_')"
LLL_PATH=$LM LLL_TARGET=1000 python longlingua.py lll11_raw_1024.jsonl 0 1 > logs/lll11.log 2>&1
log "longlingua errors: $(grep -c ERR logs/lll11.log)"
python lingua.py --trim lll11_raw_1024.jsonl $QT run11_q32/out/longlingua_budget=1024.jsonl 1024
touch run11_q32/PACK_DONE
QM=$(ls -d $R/models/models/Qwen--Qwen2.5-32B-Instruct-AWQ/snapshots/master)
CUDA_VISIBLE_DEVICES=0 python gen_stream2.py $QM run11_q32 0.9 > logs/s11_q32.log 2>&1
log ALL11_DONE
