#!/bin/bash
# RECOMP (extractive) at 512 and 2,048 tokens on Qwen-7B, Llama-8B, Qwen-14B, and at all three budgets on
# Qwen-32B-AWQ, so it can be compared with every row of the main grid. Packing reuses the cached RECOMP
# sentence order (cache/recomp_order.jsonl). Disk: Llama and 32B are removed before the 14B download.
export PATH=/root/miniconda3/bin:$PATH BP_ROOT=/root/autodl-tmp
R=/root/autodl-tmp; cd $R/bridge1027/research
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a logs/progress22.txt; }
QT=$PWD/data/tok/qwen; LT=$R/models/models/LLM-Research--Meta-Llama-3.1-8B-Instruct/snapshots/master
Q7=/root/claude_chaos/models/Qwen2.5-7B-Instruct; Q32=$R/models/models/Qwen--Qwen2.5-32B-Instruct-AWQ/snapshots/master
T=qasper,multifieldqa_en,hotpotqa,2wikimqa,musique,narrativeqa
dl(){ python -c "from modelscope import snapshot_download as d; print('DONE', d('$1', cache_dir='$R/models'))"; }
pk(){ D=$1; TK=$2; shift 2; (cd $D && CUDA_VISIBLE_DEVICES="" BP_TOK=$TK RC_THREADS=1 nice -n 5 python ../evalcov.py "$@" >> ../logs/pack22.log 2>&1); }
mkdir -p p22q/out p22l/out p22q7 p22q14 p22q32/out
ln -sfn ../p22q/out p22q7/out; ln -sfn ../p22q/out p22q14/out
log start
for b in 512 2048; do pk p22q $QT recomp_ext $T 100000 budget=$b; done
touch p22q/PACK_DONE p22q7/PACK_DONE p22q14/PACK_DONE
cp p22q/out/recomp_ext_*.jsonl p22q32/out/; cp p18q/out/recomp_ext_budget=1024_*.jsonl p22q32/out/; touch p22q32/PACK_DONE
log "Qwen traces: $(ls p22q/out | wc -l); 32B traces: $(ls p22q32/out | wc -l)"
python gen_stream2.py $Q7 p22q7 0.9 > logs/s22_q7.log 2>&1 &
G=$!
for b in 512 2048; do pk p22l $LT recomp_ext $T 100000 budget=$b; done; touch p22l/PACK_DONE
log "Llama traces: $(ls p22l/out | wc -l)"
wait $G; log "7B done: $(ls p22q7/preds | grep -c 'preds.jsonl$') preds"
python gen_stream2.py $LT p22l 0.9 > logs/s22_llama.log 2>&1; log "Llama done: $(ls p22l/preds | grep -c 'preds.jsonl$') preds"
python gen_stream2.py $Q32 p22q32 0.9 > logs/s22_q32.log 2>&1; log "32B done: $(ls p22q32/preds | grep -c 'preds.jsonl$') preds"
rm -rf $R/models/models/Qwen--Qwen2.5-32B-Instruct-AWQ $R/models/models/LLM-Research--Meta-Llama-3.1-8B-Instruct
log "32B and Llama removed: $(df -h $R | tail -1 | awk '{print $4}') free"
dl Qwen/Qwen2.5-14B-Instruct > logs/dl22.log 2>&1
Q14=$(grep "^DONE" logs/dl22.log | awk '{print $2}'); [ -z "$Q14" ] && { log "14B download failed"; exit 1; }
log "14B at $Q14"
python gen_stream2.py $Q14 p22q14 0.9 > logs/s22_q14.log 2>&1
log "ALL22_DONE: $(ls p22q14/preds | grep -c 'preds.jsonl$') 14B preds"
