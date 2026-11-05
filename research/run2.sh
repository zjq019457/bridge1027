#!/bin/bash
# Round 2: stronger baselines (BGE-large, sentence chunks, LongLLMLingua) and InfiniteBench.
# Models on disk: Llama-3.1-8B (llama tokenizer) and Qwen2.5-14B (qwen tokenizer).
export PATH=/root/miniconda3/bin:$PATH
R=/root/autodl-tmp; cd $R/bridge1027/research; export BP_ROOT=$R HF_ENDPOINT=https://hf-mirror.com
LM=$(cat $R/path_llama); QM=$R/models/models/Qwen--Qwen2.5-14B-Instruct/snapshots/master
QT=$R/bridge1027/research/data/tok/qwen
BS=$(cat $R/path_bge); BL=$R/models/models/BAAI--bge-large-en-v1.5/snapshots/master
T=qasper,multifieldqa_en,hotpotqa,2wikimqa,musique,narrativeqa
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a logs/progress2.txt; }
mkdir -p logs data/infbench run2_qwen/out run2_llama/out
log start
# downloads
python -c "from modelscope import snapshot_download as s; print(s('BAAI/bge-large-en-v1.5', cache_dir='$R/models', ignore_file_pattern=['onnx/*','.*\.onnx']))" > logs/dl_bgel.log 2>&1
for f in en.qa en.mc; do [ -s data/infbench/$f.jsonl ] || huggingface-cli download xinrongzhang2022/InfiniteBench $f.jsonl --repo-type dataset --local-dir data/infbench > logs/dl_inf.log 2>&1; done
log downloads done $(ls data/infbench)
# LongLLMLingua (compressor LM: Llama-3.1-8B) - 2 shards on GPU1
for k in 0 1; do
  CUDA_VISIBLE_DEVICES=1 LLL_PATH=$LM nohup python longlingua.py lll_raw_$k.jsonl $k 2 > logs/lll_$k.log 2>&1 &
done
# embeddings on GPU0
CUDA_VISIBLE_DEVICES=0 BP_ENC=bgel BGE_PATH=$BL python embed_cache.py test > logs/emb_bgel_test.log 2>&1
CUDA_VISIBLE_DEVICES=0 BP_ENC=bge BGE_PATH=$BS python embed_cache.py inf > logs/emb_bge_inf.log 2>&1
CUDA_VISIBLE_DEVICES=0 BGE_PATH=$BS python sentchunk.py embed test > logs/sent_test.log 2>&1
CUDA_VISIBLE_DEVICES=0 BGE_PATH=$BS python sentchunk.py embed inf > logs/sent_inf.log 2>&1
log embeddings done
lane(){ D=$1; TK=$2; cd $D
  BP_TOK=$TK BP_ENC=bgel python ../evalcov.py bge_chunk $T 100000 budget=1024
  BP_TOK=$TK python ../sentchunk.py pack test $TK 1024 out/sentchunk_budget=1024.jsonl
  BP_TOK=$TK BP_ENC=bgel python ../evalcov.py lex_rep $T 100000 seed_mode=1 sent=1 budget=1024
  for b in 1024 2048; do
    BP_TOK=$TK python ../evalcov.py bp_main en_qa,en_mc 100000 budget=$b
    BP_TOK=$TK python ../evalcov.py bge_chunk en_qa,en_mc 100000 budget=$b
    BP_TOK=$TK python ../evalcov.py lex_rep en_qa,en_mc 100000 seed_mode=2 sent=1 budget=$b
    BP_TOK=$TK python ../evalcov.py lex_rep en_qa,en_mc 100000 seed_mode=1 sent=1 budget=$b
    BP_TOK=$TK python ../sentchunk.py pack inf $TK $b out/sentchunk_inf_budget=$b.jsonl
  done; cd ..; }
lane run2_qwen $QT > logs/pack2q.log 2>&1 &
lane run2_llama $LM > logs/pack2l.log 2>&1 &
# generators: GPU0 = Qwen-14B (starts now), GPU1 = Llama (after LongLLMLingua shards there finish)
CUDA_VISIBLE_DEVICES=0 nohup python gen_stream2.py $QM run2_qwen 0.85 > logs/s2_qwen.log 2>&1 &
while pgrep -f "[l]onglingua.py" >/dev/null; do sleep 20; done; log lll done
CUDA_VISIBLE_DEVICES=1 nohup python gen_stream2.py $LM run2_llama 0.9 > logs/s2_llama.log 2>&1 &
while pgrep -f "[e]valcov.py|[s]entchunk.py pack" >/dev/null; do sleep 20; done; log packing done
cat lll_raw_0.jsonl lll_raw_1.jsonl > lll_raw.jsonl
python lingua.py --trim lll_raw.jsonl $QT run2_qwen/out/longlingua_budget=1024.jsonl 1024
python lingua.py --trim lll_raw.jsonl $LM run2_llama/out/longlingua_budget=1024.jsonl 1024
BP_SET=inf python fullctx.py $QM run2_qwen/fullinf.jsonl 30000
BP_SET=inf python fullctx.py $LM run2_llama/fullinf.jsonl 30000
touch run2_qwen/PACK_DONE run2_llama/PACK_DONE; log traces ready
while pgrep -f "[g]en_stream2.py" >/dev/null; do sleep 20; done; log stream done
CUDA_VISIBLE_DEVICES=1 python colab_gen.py --model $LM --out run2_llama/preds --max_model_len 32768 run2_llama/fullinf.jsonl > logs/fi_llama.log 2>&1
TP=2 CUDA_VISIBLE_DEVICES=0,1 python colab_gen.py --model $QM --out run2_qwen/preds --max_model_len 32768 run2_qwen/fullinf.jsonl > logs/fi_qwen.log 2>&1
log ALL2_DONE
