#!/bin/bash
# Re-create the 1,024-token BGE-large / reranker traces for Qwen-14B and Llama-8B (their predictions were
# lost with the earlier server; needed for paired tests). Qwen-tokenised traces are reused from run11_q32.
export PATH=/root/miniconda3/bin:$PATH BP_ROOT=/root/autodl-tmp
R=/root/autodl-tmp/bridge1027/research; cd $R
LT=/root/autodl-tmp/claude_llama/Meta-Llama-3.1-8B-Instruct
T=qasper,multifieldqa_en,hotpotqa,2wikimqa,musique,narrativeqa
mkdir -p rk18/l/tmp
cp run11_q32/out/bgel_*budget=1024_*.jsonl run11_q32/out/rr_*budget=1024_*.jsonl p18q/out/
for e in bgel rr; do
  (cd rk18/l/tmp && CUDA_VISIBLE_DEVICES="" BP_TOK=$LT BP_ENC=$e nice -n 5 python ../../../evalcov.py bge_chunk $T 100000 budget=1024) >> logs/pack18rk.log 2>&1
  (cd rk18/l/tmp && CUDA_VISIBLE_DEVICES="" BP_TOK=$LT BP_ENC=$e nice -n 5 python ../../../evalcov.py lex_rep $T 100000 seed_mode=1 sent=1 budget=1024) >> logs/pack18rk.log 2>&1
  for x in rk18/l/tmp/out/*.jsonl; do mv "$x" "p18l/out/${e}_$(basename $x)"; done
done
echo "[$(date +%H:%M:%S)] BGE-large/reranker 1024 traces ready" | tee -a logs/progress18.txt
