#!/usr/bin/env bash
# usage: retry_pack.sh GPU ENVPYTHONPATH SCRIPT SHARD NSHARD LOG ; reruns a resumable packing shard until it prints SHARD_DONE
W=/diskE/jiaqizhang/BRIDGEPACK_SENSITIVITY_V1_20261025; cd $W
for k in $(seq 1 50); do
  PYTHONPATH=$2 TIKTOKEN_CACHE_DIR=$W/tiktoken_cache CUDA_VISIBLE_DEVICES=$1 /home/bai-project/anaconda3/envs/eval_llama3/bin/python $3 $4 $5 > $6 2>&1
  grep -q SHARD_DONE $6 && exit 0; sleep 60
done
