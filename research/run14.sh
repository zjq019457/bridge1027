#!/bin/bash
# Third model family: Mistral-7B-Instruct-v0.3, main comparison at 512/1024/2048 + full context (after run13).
export PATH=/root/miniconda3/bin:$PATH
R=/root/autodl-tmp; cd $R/bridge1027/research; export BP_ROOT=$R
T=qasper,multifieldqa_en,hotpotqa,2wikimqa,musique,narrativeqa
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a logs/progress14.txt; }
mkdir -p run14_mis/out
log start
# tokenizer only (small) so packing can run on CPU now; full weights after Qwen-14B is removed
python -c "from modelscope import snapshot_download as d; print('DONE', d('LLM-Research/Mistral-7B-Instruct-v0.3', cache_dir='$R/models/mtok', allow_file_pattern=['tokenizer*','*.json']))" > logs/dl14m_tok.log 2>&1
MT=$(grep '^DONE' logs/dl14m_tok.log | awk '{print $2}'); log "tokenizer at $MT"
[ -f $MT/tokenizer.json ] || { log "NO TOKENIZER"; exit 1; }
( cd run14_mis
  for b in 1024 512 2048; do
    BP_TOK=$MT python ../evalcov.py lex_rep $T 100000 seed_mode=2 sent=1 budget=$b &
    BP_TOK=$MT python ../evalcov.py lex_rep $T 100000 seed_mode=1 sent=1 budget=$b &
    BP_TOK=$MT python ../evalcov.py bge_chunk $T 100000 budget=$b &
    BP_TOK=$MT python ../evalcov.py bp_main $T 100000 budget=$b &
    wait
  done
  BP_TOK=$MT python ../fullctx32.py out/fullctx.jsonl 30000 ) > logs/pack14.log 2>&1
log "packed: $(ls run14_mis/out | wc -l) traces"
until grep -q ALL13_DONE logs/progress13.txt; do sleep 60; done; log "run13 done"
Q14=$(grep '^DONE' logs/dl14.log | awk '{print $2}'); [ -n "$Q14" ] && rm -rf "$Q14"
python -c "from modelscope import snapshot_download as d; print('DONE', d('LLM-Research/Mistral-7B-Instruct-v0.3', cache_dir='$R/models', ignore_file_pattern=['consolidated*','original/*','*.pth']))" > logs/dl14m.log 2>&1
MM=$(grep '^DONE' logs/dl14m.log | awk '{print $2}'); log "model at $MM"
touch run14_mis/PACK_DONE
CUDA_VISIBLE_DEVICES=0 python gen_stream2.py $MM run14_mis 0.9 > logs/s14_mis.log 2>&1; log "packed gen done"
CUDA_VISIBLE_DEVICES=0 python colab_gen.py --model $MM --out run14_mis/preds --max_model_len 32768 run14_mis/out/fullctx.jsonl > logs/f14_mis.log 2>&1
log ALL14_DONE
