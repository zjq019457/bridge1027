#!/bin/bash
# Round 18 (reviewer follow-ups): packer sensitivity (radius, pool), 256-token budget (Qwen-14B),
# head/middle truncation, RECOMP-extractive, Selective Context, sampling vs greedy.
# Packing runs on CPU at once; generation waits until the GPU is free (other users' jobs).
export PATH=/root/miniconda3/bin:$PATH BP_ROOT=/root/autodl-tmp
R=/root/autodl-tmp/bridge1027/research; cd $R
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a logs/progress18.txt; }
QT=$R/data/tok/qwen; LT=/root/autodl-tmp/claude_llama/Meta-Llama-3.1-8B-Instruct
Q14=/root/autodl-tmp/models/models/Qwen--Qwen2.5-14B-Instruct/snapshots/master; Q7=/root/claude_chaos/models/Qwen2.5-7B-Instruct
T=qasper,multifieldqa_en,hotpotqa,2wikimqa,musique,narrativeqa; TS=_2wikimqa-hotpotqa-multifieldqa_en-musique-narrativeqa-qasper
mkdir -p p18q/out p18l/out s18q/out s18l/out c18q/out c18l/out p18q7 c18q7 logs
ln -sfn ../p18q/out p18q7/out; ln -sfn ../c18q/out c18q7/out
log start
pk(){ D=$1; TK=$2; shift 2; (cd $D && CUDA_VISIBLE_DEVICES="" BP_TOK=$TK RC_THREADS=1 nice -n 5 python ../evalcov.py "$@" >> ../logs/pack18.log 2>&1); }
# Selective Context units (CPU, background)
( CUDA_VISIBLE_DEVICES="" SC_THREADS=1 nice -n 10 python sc_units.py 4 cache/sc_units.jsonl > logs/sc_units.log 2>&1 ) &
SCPID=$!
# sampling traces: Qwen tokenizer copies, Llama tokenizer packed
cp run7_q32/out/bge_chunk_budget=1024$TS.jsonl run8_q32/out/lex_rep_seed_mode=1_sent=1_budget=1024$TS.jsonl s18q/out/
pk s18l $LT bge_chunk $T 100000 budget=1024; pk s18l $LT lex_rep $T 100000 seed_mode=1 sent=1 budget=1024
touch s18q/PACK_DONE s18l/PACK_DONE; log "sampling traces ready"
for a in "radius=16" "radius=64" "pool=30" "pool=120"; do
  pk p18q $QT lex_rep $T 100000 seed_mode=1 sent=1 $a budget=1024; pk p18l $LT lex_rep $T 100000 seed_mode=1 sent=1 $a budget=1024
done; log "sensitivity traces ready"
for m in head_trunc mid_trunc recomp_ext; do pk p18q $QT $m $T 100000 budget=1024; pk p18l $LT $m $T 100000 budget=1024; done
touch p18l/PACK_DONE; log "baseline traces ready"
pk p18q $QT bge_chunk $T 100000 budget=256; pk p18q $QT lex_rep $T 100000 seed_mode=1 sent=1 budget=256
pk p18q $QT lex_rep $T 100000 seed_mode=2 sent=1 budget=256
pk p18q $QT head_trunc $T 100000 budget=256; pk p18q $QT mid_trunc $T 100000 budget=256
touch p18q/PACK_DONE p18q7/PACK_DONE; log "256 traces ready"
# ---- GPU: wait until no other process holds memory
gpu_free(){ [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)" -lt 2000 ]; }
until gpu_free && sleep 120 && gpu_free; do sleep 60; done; log "GPU free"
python gen_stream2.py $Q14 p18q 0.9 > logs/s18_q14.log 2>&1; log "14B done"
for sd in 1 2 3; do GEN_TEMP=0.7 GEN_TOPP=0.9 GEN_SEED=$sd SUFFIX=_t07s$sd python gen_stream3.py $Q14 s18q 0.9 >> logs/s18_q14_samp.log 2>&1; done; log "14B sampling done"
python gen_stream2.py $Q7 p18q7 0.9 > logs/s18_q7.log 2>&1; log "7B done"
python gen_stream2.py $LT p18l 0.9 > logs/s18_llama.log 2>&1; log "Llama done"
for sd in 1 2 3; do GEN_TEMP=0.7 GEN_TOPP=0.9 GEN_SEED=$sd SUFFIX=_t07s$sd python gen_stream3.py $LT s18l 0.9 >> logs/s18_llama_samp.log 2>&1; done; log "Llama sampling done"
# ---- Selective Context once its units are cached
wait $SCPID; log "SC units: $(wc -l < cache/sc_units.jsonl)"
pk c18q $QT sel_ctx2 $T 100000 budget=1024; pk c18l $LT sel_ctx2 $T 100000 budget=1024
touch c18q/PACK_DONE c18q7/PACK_DONE c18l/PACK_DONE
until gpu_free; do sleep 60; done
python gen_stream2.py $Q14 c18q 0.9 > logs/s18c_q14.log 2>&1
python gen_stream2.py $Q7 c18q7 0.9 > logs/s18c_q7.log 2>&1
python gen_stream2.py $LT c18l 0.9 > logs/s18c_llama.log 2>&1
log ALL18_DONE
