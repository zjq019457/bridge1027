#!/bin/bash
# Follow-up queue: starts as soon as the 7B runs free the GPUs, so they never sit idle.
export PATH=/root/miniconda3/bin:$PATH
R=/root/autodl-tmp; cd $R/bridge1027/research; export BP_ROOT=$R HF_ENDPOINT=https://hf-mirror.com
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a logs/progress.txt; }
# 1) CPU-only work right away: packing latency
python timing.py 300 > logs/timing.log 2>&1 &
# 2) wait for Qwen-7B work (stream + full context) to finish, then swap in Qwen2.5-14B
until grep -q "qwen full done" logs/progress.txt; do sleep 20; done
rm -rf $R/models/models/Qwen--Qwen2.5-7B-Instruct; log qwen7b removed
python -c "from modelscope import snapshot_download as s; print(s('Qwen/Qwen2.5-14B-Instruct', cache_dir='$R/models'))" > logs/dl14.log 2>&1
Q14=$R/models/models/Qwen--Qwen2.5-14B-Instruct/snapshots/master; log 14b downloaded
mkdir -p run_qwen14; ln -sfn ../run_qwen/out run_qwen14/out; touch run_qwen14/PACK_DONE
CUDA_VISIBLE_DEVICES=0 python gen_stream2.py $Q14 run_qwen14 0.9 > logs/stream_q14_a.log 2>&1 &
# GPU1 joins as soon as Llama is done
until grep -q "llama full done" logs/progress.txt; do sleep 20; done
CUDA_VISIBLE_DEVICES=1 python gen_stream2.py $Q14 run_qwen14 0.9 > logs/stream_q14_b.log 2>&1 &
wait; log 14b done
grep -h macro logs/stream_q14_*.log | tee -a logs/progress.txt
log FOLLOW_DONE
