#!/bin/bash
# Resume of run23 after the CompAct download hit hf-mirror's rate limit (HTTP 429) at 14:47.
# EXIT and CPC scores are complete: pack and generate them on Qwen-7B/14B while the CompAct download is retried in
# the background (every 2 min, up to 3 h); then CompAct, then Llama-8B for all three methods.
export PATH=/root/miniconda3/bin:$PATH BP_ROOT=/root/autodl-tmp HF_ENDPOINT=https://hf-mirror.com HF_HOME=/root/autodl-tmp/hf_home
R=/root/autodl-tmp; cd $R/bridge1027/research
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a logs/progress23.txt; }
QT=$PWD/data/tok/qwen; Q7=/root/claude_chaos/models/Qwen2.5-7B-Instruct
Q14=$(ls -d $R/models/models/Qwen--Qwen2.5-14B-Instruct/snapshots/* | head -1)
T=qasper,multifieldqa_en,hotpotqa,2wikimqa,musique,narrativeqa
ms(){ python -c "from modelscope import snapshot_download as d; print('DONE', d('$1', cache_dir='$R/models', ignore_file_pattern=['original/*', '*.pth', '*.gguf', '*.bin']))"; }
pk(){ D=$1; TK=$2; shift 2; (cd $D && CUDA_VISIBLE_DEVICES="" BP_TOK=$TK RC_THREADS=1 nice -n 5 python ../evalcov.py "$@" >> ../logs/pack23.log 2>&1); }
free(){ df -h $R | tail -1 | awk '{print $4}'; }
rm -f logs/COMPACT_DL_OK logs/COMPACT_DL_GAVEUP
log "run23b: resume; $(free) free"
# ---- background: CompAct download with retries
( for i in $(seq 1 90); do
    huggingface-cli download cwyoon99/CompAct-7b --local-dir $R/models/CompAct-7b >> logs/dl23_compact.log 2>&1 \
      && [ $(ls $R/models/CompAct-7b/*.safetensors 2>/dev/null | wc -l) -eq 3 ] && { touch logs/COMPACT_DL_OK; exit 0; }
    sleep 120
  done; touch logs/COMPACT_DL_GAVEUP ) &
# ---- EXIT and CPC on Qwen-7B and Qwen-14B
mkdir -p p23q/out p23q7 p23q14
ln -sfn ../p23q/out p23q7/out; ln -sfn ../p23q/out p23q14/out
for m in exit_ext cpc_ext; do pk p23q $QT $m $T 100000 budget=1024; done
touch p23q/PACK_DONE p23q7/PACK_DONE p23q14/PACK_DONE; log "Qwen traces (EXIT, CPC): $(ls p23q/out | wc -l)"
python gen_stream2.py $Q7 p23q7 0.9 > logs/s23_q7.log 2>&1; log "7B done: $(ls p23q7/preds | grep -c 'preds.jsonl$') preds"
python gen_stream2.py $Q14 p23q14 0.9 > logs/s23_q14.log 2>&1; log "14B done: $(ls p23q14/preds | grep -c 'preds.jsonl$') preds"
# ---- CompAct
until [ -f logs/COMPACT_DL_OK ] || [ -f logs/COMPACT_DL_GAVEUP ]; do sleep 60; done
if [ -f logs/COMPACT_DL_OK ]; then
  export COMPACT=$R/models/CompAct-7b
  log "CompAct downloaded; $(free) free"
  CUDA_VISIBLE_DEVICES=0 python compact_run.py /tmp/compact_smoke.jsonl 4 > logs/compact23_smoke.log 2>&1 || { log "CompAct smoke failed"; exit 1; }
  log "CompAct smoke ok"
  CUDA_VISIBLE_DEVICES=0 python compact_run.py cache/compact_out.jsonl > logs/compact23.log 2>&1 || { log "CompAct failed"; exit 1; }
  log "CompAct done: $(wc -l < cache/compact_out.jsonl)"
  rm -rf $COMPACT
  mkdir -p p23c/out p23c7 p23c14
  ln -sfn ../p23c/out p23c7/out; ln -sfn ../p23c/out p23c14/out
  pk p23c $QT compact_ext $T 100000 budget=1024
  touch p23c/PACK_DONE p23c7/PACK_DONE p23c14/PACK_DONE; log "Qwen traces (CompAct): $(ls p23c/out | wc -l)"
  python gen_stream2.py $Q7 p23c7 0.9 > logs/s23c_q7.log 2>&1; log "7B CompAct done: $(ls p23c7/preds | grep -c 'preds.jsonl$') preds"
  python gen_stream2.py $Q14 p23c14 0.9 > logs/s23c_q14.log 2>&1; log "14B CompAct done: $(ls p23c14/preds | grep -c 'preds.jsonl$') preds"
  M="exit_ext cpc_ext compact_ext"
else
  log "CompAct download gave up after 3 h (hf-mirror rate limit); Llama runs EXIT and CPC only"
  M="exit_ext cpc_ext"
fi
# ---- Llama-8B (re-download without the original/ checkpoint)
rm -rf $R/models/models/Qwen--Qwen2.5-14B-Instruct
ms LLM-Research/Meta-Llama-3.1-8B-Instruct > logs/dl23_llama.log 2>&1
LT=$(grep "^DONE" logs/dl23_llama.log | awk '{print $2}'); [ -z "$LT" ] && { log "Llama download failed"; exit 1; }
log "Llama at $LT; $(free) free"
mkdir -p p23l/out
for m in $M; do pk p23l $LT $m $T 100000 budget=1024; done
touch p23l/PACK_DONE; log "Llama traces: $(ls p23l/out | wc -l)"
python gen_stream2.py $LT p23l 0.9 > logs/s23_llama.log 2>&1
log "ALL23_DONE: $(ls p23l/preds | grep -c 'preds.jsonl$') Llama preds"
