#!/bin/bash
# EXIT, CPC and CompAct at 1,024 tokens on Qwen-7B, Qwen-14B and Llama-8B, queued after run22.
# Compressors run one at a time (disk): each is downloaded, run (2-instance smoke test first), and removed.
export PATH=/root/miniconda3/bin:$PATH BP_ROOT=/root/autodl-tmp HF_ENDPOINT=https://hf-mirror.com HF_HOME=/root/autodl-tmp/hf_home
R=/root/autodl-tmp; cd $R/bridge1027/research
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a logs/progress23.txt; }
X=$R/extenv/bin/python
QT=$PWD/data/tok/qwen; Q7=/root/claude_chaos/models/Qwen2.5-7B-Instruct
T=qasper,multifieldqa_en,hotpotqa,2wikimqa,musique,narrativeqa
ms(){ python -c "from modelscope import snapshot_download as d; print('DONE', d('$1', cache_dir='$R/models', ignore_file_pattern=['original/*', '*.pth', '*.gguf', '*.bin']))"; }
pk(){ D=$1; TK=$2; shift 2; (cd $D && CUDA_VISIBLE_DEVICES="" BP_TOK=$TK RC_THREADS=1 nice -n 5 python ../evalcov.py "$@" >> ../logs/pack23.log 2>&1); }
free(){ df -h $R | tail -1 | awk '{print $4}'; }
log "queued; waiting for run22"
until grep -q ALL22_DONE logs/progress22.txt 2>/dev/null; do sleep 60; done
Q14=$(ls -d $R/models/models/Qwen--Qwen2.5-14B-Instruct/snapshots/* | head -1)
log "start: $(free) free"
# ---- EXIT (Gemma-2B-it + released adapter)
export EXIT_BASE=$(ls -d $R/models/models/LLM-Research--gemma-2b-it/snapshots/* | head -1) EXIT_ADAPTER=$R/aux/exit-gemma-2b
CUDA_VISIBLE_DEVICES=0 $X exit_scores.py /tmp/exit_smoke.jsonl 2 > logs/exit23_smoke.log 2>&1 || { log "EXIT smoke failed"; exit 1; }
log "EXIT smoke ok"
CUDA_VISIBLE_DEVICES=0 $X exit_scores.py cache/exit_scores.jsonl > logs/exit23.log 2>&1 || { log "EXIT failed"; exit 1; }
log "EXIT done: $(wc -l < cache/exit_scores.jsonl)"
rm -rf $R/models/models/LLM-Research--gemma-2b-it
# ---- CPC (Mistral-7B-Instruct-v0.2 + released LoRA, bidirectional)
ms LLM-Research/Mistral-7B-Instruct-v0.2 > logs/dl23_mistral.log 2>&1
export CPC_DIR=$R/aux/cpc CPC_BASE=$(grep "^DONE" logs/dl23_mistral.log | awk '{print $2}') \
       CPC_LORA=$R/aux/cpc-1.0-mistral-7b-ds-v5-iter66-lora-bidirectional-attn CPC_TOK=$R/aux/cpc-1.0-mistral-7b-tokenizer
[ -z "$CPC_BASE" ] && { log "Mistral download failed"; exit 1; }
log "Mistral at $CPC_BASE; $(free) free"
CUDA_VISIBLE_DEVICES=0 $X cpc_scores.py /tmp/cpc_smoke.jsonl 2 > logs/cpc23_smoke.log 2>&1 || { log "CPC smoke failed"; exit 1; }
log "CPC smoke ok"
CUDA_VISIBLE_DEVICES=0 $X cpc_scores.py cache/cpc_scores.jsonl > logs/cpc23.log 2>&1 || { log "CPC failed"; exit 1; }
log "CPC done: $(wc -l < cache/cpc_scores.jsonl)"
rm -rf $CPC_BASE
# ---- CompAct (CompAct-7b)
huggingface-cli download cwyoon99/CompAct-7b --local-dir $R/models/CompAct-7b > logs/dl23_compact.log 2>&1 || { log "CompAct download failed"; exit 1; }
export COMPACT=$R/models/CompAct-7b
log "CompAct downloaded; $(free) free"
CUDA_VISIBLE_DEVICES=0 python compact_run.py /tmp/compact_smoke.jsonl 4 > logs/compact23_smoke.log 2>&1 || { log "CompAct smoke failed"; exit 1; }
log "CompAct smoke ok"
CUDA_VISIBLE_DEVICES=0 python compact_run.py cache/compact_out.jsonl > logs/compact23.log 2>&1 || { log "CompAct failed"; exit 1; }
log "CompAct done: $(wc -l < cache/compact_out.jsonl)"
rm -rf $COMPACT
# ---- traces and generation: Qwen-7B, Qwen-14B
mkdir -p p23q/out p23q7 p23q14 p23l/out
ln -sfn ../p23q/out p23q7/out; ln -sfn ../p23q/out p23q14/out
for m in exit_ext cpc_ext compact_ext; do pk p23q $QT $m $T 100000 budget=1024; done
touch p23q/PACK_DONE p23q7/PACK_DONE p23q14/PACK_DONE; log "Qwen traces: $(ls p23q/out | wc -l)"
python gen_stream2.py $Q7 p23q7 0.9 > logs/s23_q7.log 2>&1; log "7B done: $(ls p23q7/preds | grep -c 'preds.jsonl$') preds"
python gen_stream2.py $Q14 p23q14 0.9 > logs/s23_q14.log 2>&1; log "14B done: $(ls p23q14/preds | grep -c 'preds.jsonl$') preds"
# ---- Llama-8B (re-download without the original/ checkpoint)
rm -rf $R/models/models/Qwen--Qwen2.5-14B-Instruct
ms LLM-Research/Meta-Llama-3.1-8B-Instruct > logs/dl23_llama.log 2>&1
LT=$(grep "^DONE" logs/dl23_llama.log | awk '{print $2}'); [ -z "$LT" ] && { log "Llama download failed"; exit 1; }
log "Llama at $LT; $(free) free"
for m in exit_ext cpc_ext compact_ext; do pk p23l $LT $m $T 100000 budget=1024; done
touch p23l/PACK_DONE; log "Llama traces: $(ls p23l/out | wc -l)"
python gen_stream2.py $LT p23l 0.9 > logs/s23_llama.log 2>&1
log "ALL23_DONE: $(ls p23l/preds | grep -c 'preds.jsonl$') Llama preds"
