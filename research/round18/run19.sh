#!/bin/bash
# After round 18: free the disk (delete the 14B copy), download Qwen2.5-32B-Instruct-AWQ and run the
# hybrid-ranker chunk-packing traces on it (completes the 32B column of the chunk-packing block).
export PATH=/root/miniconda3/bin:$PATH
R=/root/autodl-tmp; cd $R/bridge1027/research
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a logs/progress19.txt; }
log "waiting for round 18"
until grep -q ALL18_DONE logs/progress18.txt 2>/dev/null; do sleep 60; done
mkdir -p p19q32/out; cp h18/q/out/hyb_chunk_budget=*.jsonl p19q32/out/; touch p19q32/PACK_DONE
log "traces: $(ls p19q32/out | wc -l)"
rm -rf $R/models/models/Qwen--Qwen2.5-14B-Instruct; log "14B removed: $(df -h $R | tail -1 | awk '{print $4}') free"
python -c "from modelscope import snapshot_download as d; print('DONE', d('Qwen/Qwen2.5-32B-Instruct-AWQ', cache_dir='$R/models'))" > logs/dl19.log 2>&1
Q32=$(grep "^DONE" logs/dl19.log | awk '{print $2}'); log "32B at $Q32"
gpu_free(){ [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)" -lt 2000 ]; }
until gpu_free; do sleep 60; done
python gen_stream2.py $Q32 p19q32 0.9 > logs/s19_q32.log 2>&1
log ALL19_DONE
