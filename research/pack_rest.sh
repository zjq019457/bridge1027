#!/bin/bash
R=/root/autodl-tmp; cd $R/bridge1027/research; export BP_ROOT=$R
T=qasper,multifieldqa_en,hotpotqa,2wikimqa,musique,narrativeqa
TS=2wikimqa-hotpotqa-multifieldqa_en-musique-narrativeqa-qasper
while pgrep -f "[e]valcov.py" >/dev/null; do sleep 10; done
one(){ f=out/$1_$(echo "${@:2}" | tr ' ' '_')_$TS.jsonl; [ -s "$f" ] || python ../evalcov.py $1 $T 100000 "${@:2}"; }
lane(){ cd $1
  for b in 512 1024 2048; do one bp_main budget=$b; one bge_chunk budget=$b; one lex_rep seed_mode=2 sent=1 budget=$b; one lex_rep seed_mode=1 sent=1 budget=$b; done
  one lex_rep seed_mode=0 sent=1 budget=1024
  for a in sent=0 merge=0 docorder=0 backfill=0; do one final_abl $a budget=1024; done; cd ..; }
BP_TOK=$R/bridge1027/research/data/tok/qwen lane run_qwen > logs/pack2_qwen.log 2>&1 &
BP_TOK=$(cat $R/path_llama) lane run_llama > logs/pack2_llama.log 2>&1 &
wait; touch PACK_LANES_DONE
