#!/bin/bash
# Hybrid-ranker chunk packing (the missing chunk-packing row for the hybrid ranker), 512/1,024/2,048,
# Qwen and Llama tokenizers; traces go to the round-18 directories so the queued generators pick them up.
export PATH=/root/miniconda3/bin:$PATH BP_ROOT=/root/autodl-tmp
R=/root/autodl-tmp/bridge1027/research; cd $R
QT=$R/data/tok/qwen; LT=/root/autodl-tmp/claude_llama/Meta-Llama-3.1-8B-Instruct
T=qasper,multifieldqa_en,hotpotqa,2wikimqa,musique,narrativeqa
mkdir -p h18/q h18/l
for b in 1024 512 2048; do
  (cd h18/q && CUDA_VISIBLE_DEVICES="" BP_TOK=$QT nice -n 5 python ../../evalcov.py hyb_chunk $T 100000 budget=$b) >> logs/pack18h.log 2>&1
  cp h18/q/out/hyb_chunk_budget=${b}_*.jsonl p18q/out/
  (cd h18/l && CUDA_VISIBLE_DEVICES="" BP_TOK=$LT nice -n 5 python ../../evalcov.py hyb_chunk $T 100000 budget=$b) >> logs/pack18h.log 2>&1
  cp h18/l/out/hyb_chunk_budget=${b}_*.jsonl p18l/out/
done
echo "[$(date +%H:%M:%S)] hybrid chunk traces ready" | tee -a logs/progress18.txt
