#!/bin/bash
export PATH=/root/miniconda3/bin:$PATH
R=/root/autodl-tmp; cd $R/bridge1027/research; export BP_ROOT=$R
QM=$(cat $R/path_qwen); LM=$(cat $R/path_llama)
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a logs/progress.txt; }
python fullctx.py $QM run_qwen/fullctx.jsonl 30000 &
python fullctx.py $LM run_llama/fullctx.jsonl 30000 &
while pgrep -f "[l]ingua.py lingua_raw" >/dev/null; do sleep 20; done; log lingua done
python lingua.py --trim lingua_raw.jsonl data/tok/qwen run_qwen/out/lingua2_budget=1024.jsonl 1024
python lingua.py --trim lingua_raw.jsonl $LM run_llama/out/lingua2_budget=1024.jsonl 1024
wait; touch run_qwen/PACK_DONE run_llama/PACK_DONE; log traces ready
( while pgrep -f "[g]en_stream.py .*Qwen--Qwen2.5-7B" >/dev/null; do sleep 15; done
  CUDA_VISIBLE_DEVICES=0 python colab_gen.py --model $QM --out run_qwen/preds --max_model_len 32768 run_qwen/fullctx.jsonl > logs/full_qwen.log 2>&1
  log qwen full done ) &
( while pgrep -f "[g]en_stream.py .*Llama" >/dev/null; do sleep 15; done
  CUDA_VISIBLE_DEVICES=1 python colab_gen.py --model $LM --out run_llama/preds --max_model_len 32768 run_llama/fullctx.jsonl > logs/full_llama.log 2>&1
  log llama full done ) &
wait; log fullctx done
