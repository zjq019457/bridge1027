#!/bin/bash
# Frozen test-set run (configs fixed on dev before this run). Two lanes, one per GPU.
set -u
R=/root/autodl-tmp; cd $R/bridge1027/research
export BP_ROOT=$R HF_ENDPOINT=https://hf-mirror.com BGE_PATH=$(cat $R/path_bge)
QM=$(cat $R/path_qwen); LM=$(cat $R/path_llama)
T=qasper,multifieldqa_en,hotpotqa,2wikimqa,musique,narrativeqa
mkdir -p logs
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a logs/progress.txt; }

pack(){ # $1 = lane dir (tokenizer), then packer configs
  D=$1; shift; mkdir -p $D; cd $D
  for b in 512 1024 2048; do
    python ../evalcov.py bp_main $T 100000 budget=$b
    python ../evalcov.py bge_chunk $T 100000 budget=$b
    python ../evalcov.py lex_rep $T 100000 seed_mode=2 sent=1 budget=$b
    python ../evalcov.py lex_rep $T 100000 seed_mode=1 sent=1 budget=$b
  done
  python ../evalcov.py lex_rep $T 100000 seed_mode=0 sent=1 budget=1024
  for a in sent=0 merge=0 docorder=0 backfill=0; do python ../evalcov.py final_abl $T 100000 $a budget=1024; done
  cd ..
}

log start
CUDA_VISIBLE_DEVICES=1 python lingua.py lingua_raw.jsonl > logs/lingua.log 2>&1 &
LP=$!
CUDA_VISIBLE_DEVICES=0 python embed_cache.py test > logs/embed.log 2>&1; log embed done
BP_TOK=$R/bridge1027/research/data/tok/qwen pack run_qwen > logs/pack_qwen.log 2>&1 &
P1=$!
BP_TOK=$LM pack run_llama > logs/pack_llama.log 2>&1 &
P2=$!
wait $P1 $P2; log packing done
wait $LP; log lingua done
python lingua.py --trim lingua_raw.jsonl data/tok/qwen run_qwen/out/lingua2_budget=1024.jsonl 1024
python lingua.py --trim lingua_raw.jsonl $LM run_llama/out/lingua2_budget=1024.jsonl 1024
python fullctx.py $QM run_qwen/out/fullctx.jsonl 30000
python fullctx.py $LM run_llama/out/fullctx.jsonl 30000
log traces ready

gen(){ # gpu model dir
  CUDA_VISIBLE_DEVICES=$1 python colab_gen.py --model $2 --out $3/preds $(ls $3/out/*.jsonl | grep -v fullctx) > logs/gen_$(basename $3).log 2>&1
  CUDA_VISIBLE_DEVICES=$1 python colab_gen.py --model $2 --out $3/preds --max_model_len 32768 $3/out/fullctx.jsonl >> logs/gen_$(basename $3).log 2>&1
}
gen 0 $QM run_qwen &
G1=$!
gen 1 $LM run_llama &
G2=$!
wait $G1 $G2; log generation done
grep -h macro logs/gen_*.log | tee -a logs/progress.txt
log ALL_DONE
