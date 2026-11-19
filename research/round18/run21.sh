#!/bin/bash
# Resume after the disk filled up (the Llama download included the 16 GB original/ checkpoint, now deleted):
# Llama traces, Llama sampling, Llama Selective Context, then Qwen2.5-32B-AWQ (download resumes).
export PATH=/root/miniconda3/bin:$PATH BP_ROOT=/root/autodl-tmp
R=/root/autodl-tmp; cd $R/bridge1027/research
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a logs/progress19.txt; }
LT=$R/models/models/LLM-Research--Meta-Llama-3.1-8B-Instruct/snapshots/master
dl(){ python -c "from modelscope import snapshot_download as d; print('DONE', d('$1', cache_dir='$R/models'))"; }
log "resume (run21): $(df -h $R | tail -1 | awk '{print $4}') free; Llama SC trace rows: $(wc -l < c18l/out/*.jsonl)"
( dl Qwen/Qwen2.5-32B-Instruct-AWQ > logs/dl19.log 2>&1 ) &
DL=$!
python gen_stream2.py $LT p18l 0.9 > logs/s18_llama.log 2>&1; log "Llama done: $(ls p18l/preds | grep -c 'preds.jsonl$') preds"
: > logs/s18_llama_samp.log
for sd in 1 2 3; do GEN_TEMP=0.7 GEN_TOPP=0.9 GEN_SEED=$sd SUFFIX=_t07s$sd python gen_stream3.py $LT s18l 0.9 >> logs/s18_llama_samp.log 2>&1; done
log "Llama sampling done: $(ls s18l/preds | grep -c 'preds.jsonl$') preds"
python gen_stream2.py $LT c18l 0.9 > logs/s18c_llama.log 2>&1; log "Llama Selective Context done"
echo "[$(date +%H:%M:%S)] ALL18_DONE" >> logs/progress18.txt
wait $DL; Q32=$(grep "^DONE" logs/dl19.log | awk '{print $2}')
[ -z "$Q32" ] && { log "32B download failed"; exit 1; }
log "32B at $Q32; $(df -h $R | tail -1 | awk '{print $4}') free"
mkdir -p p19q32/out; cp h18/q/out/hyb_chunk_budget=*.jsonl p19q32/out/; touch p19q32/PACK_DONE
python gen_stream2.py $Q32 p19q32 0.9 > logs/s19_q32.log 2>&1
log "ALL19_DONE: $(ls p19q32/preds | grep -c 'preds.jsonl$') preds"
