#!/bin/bash
# Resume rounds 18/19 after the Llama copy at /root/autodl-tmp/claude_llama was removed:
# Selective Context generation on Qwen-14B and Qwen-7B, then Llama-3.1-8B-Instruct (re-downloaded from
# ModelScope) and Qwen2.5-32B-Instruct-AWQ. The 14B model is removed once it is no longer needed (disk).
export PATH=/root/miniconda3/bin:$PATH BP_ROOT=/root/autodl-tmp
R=/root/autodl-tmp; cd $R/bridge1027/research
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a logs/progress19.txt; }
Q14=$R/models/models/Qwen--Qwen2.5-14B-Instruct/snapshots/master; Q7=/root/claude_chaos/models/Qwen2.5-7B-Instruct
T=qasper,multifieldqa_en,hotpotqa,2wikimqa,musique,narrativeqa
dl(){ python -c "from modelscope import snapshot_download as d; print('DONE', d('$1', cache_dir='$R/models'))"; }
log "resume (run20)"
: > logs/s18_llama_samp.log
touch c18q/PACK_DONE c18q7/PACK_DONE
python gen_stream2.py $Q14 c18q 0.9 > logs/s18c_q14.log 2>&1; log "14B Selective Context done"
rm -rf $R/models/models/Qwen--Qwen2.5-14B-Instruct; log "14B removed: $(df -h $R | tail -1 | awk '{print $4}') free"
python gen_stream2.py $Q7 c18q7 0.9 > logs/s18c_q7.log 2>&1 &
G7=$!
dl LLM-Research/Meta-Llama-3.1-8B-Instruct > logs/dl20_llama.log 2>&1
LT=$(grep "^DONE" logs/dl20_llama.log | awk '{print $2}')
[ -z "$LT" ] && { log "Llama download failed"; exit 1; }
log "Llama at $LT"
( dl Qwen/Qwen2.5-32B-Instruct-AWQ > logs/dl19.log 2>&1 ) &
DL=$!
(cd c18l && CUDA_VISIBLE_DEVICES="" BP_TOK=$LT RC_THREADS=1 nice -n 5 python ../evalcov.py sel_ctx2 $T 100000 budget=1024 >> ../logs/pack18.log 2>&1)
touch c18l/PACK_DONE; log "Llama Selective Context trace: $(ls c18l/out | wc -l)"
wait $G7; log "7B Selective Context done"
python gen_stream2.py $LT p18l 0.9 > logs/s18_llama.log 2>&1; log "Llama done"
for sd in 1 2 3; do GEN_TEMP=0.7 GEN_TOPP=0.9 GEN_SEED=$sd SUFFIX=_t07s$sd python gen_stream3.py $LT s18l 0.9 >> logs/s18_llama_samp.log 2>&1; done
log "Llama sampling done"
python gen_stream2.py $LT c18l 0.9 > logs/s18c_llama.log 2>&1; log "Llama Selective Context done"
echo "[$(date +%H:%M:%S)] ALL18_DONE" >> logs/progress18.txt
wait $DL; Q32=$(grep "^DONE" logs/dl19.log | awk '{print $2}')
[ -z "$Q32" ] && { log "32B download failed"; exit 1; }
log "32B at $Q32"
mkdir -p p19q32/out; cp h18/q/out/hyb_chunk_budget=*.jsonl p19q32/out/; touch p19q32/PACK_DONE
python gen_stream2.py $Q32 p19q32 0.9 > logs/s19_q32.log 2>&1
log ALL19_DONE
