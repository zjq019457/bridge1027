#!/usr/bin/env bash
W=/diskE/jiaqizhang/BRIDGEPACK_SENSITIVITY_V1_20261025; cd $W
P=$W/runtime/silm/bin/python; Q=/diskF/jiaqizhang/baselines/RepairKV/models/Qwen2.5-7B-Instruct; L=/diskD/jiaqizhang/models/Meta-Llama-3-8B-Instruct
enq(){ n=$1; m=$Q; [ "${n%%_*}" = llama ] && m=$L
  flock queue/lock bash -c "printf \"rv5_%s\t%s %s/gen_runner.py --trace %s/runs/rv5/%s.jsonl --model %s --out %s/runs/rv5/%s_qa\n\" $n $P $W $W $n $m $W $n >> queue/jobs.txt"
  echo "[$(date -Is)] enqueued $n"; }
pending="sc bge bm25_chunk_source_order tfidf_chunk_source_order lingua2"
while [ -n "$pending" ]; do
  next=""
  for x in $pending; do ok=0
    case $x in
      sc|lingua2) d=runs/rv5/$x; [ $(cat $d/qwen_shard*.jsonl 2>/dev/null | wc -l) -ge 1150 ] && [ $(cat $d/llama_shard*.jsonl 2>/dev/null | wc -l) -ge 1150 ] && {
          for m in qwen llama; do $P -c "
import json,glob,sys
sys.path.insert(0,\"scripts\"); TASKS=[\"qasper\",\"multifieldqa_en\",\"hotpotqa\",\"2wikimqa\",\"musique\",\"narrativeqa\"]
r=[json.loads(l) for f in sorted(glob.glob(\"$d/${m}_shard*.jsonl\")) for l in open(f)]
r.sort(key=lambda x:(TASKS.index(x[\"task\"]),x[\"sample_id\"]))
open(\"runs/rv5/${m}_$x.jsonl\",\"w\").write(\"\".join(json.dumps(x)+\"\n\" for x in r))"; enq ${m}_$x; done; ok=1; } ;;
      bge) grep -q BGE_PACK_DONE runs/rv5/bge_qwen.log && grep -q BGE_PACK_DONE runs/rv5/bge_llama.log && { enq qwen_bge; enq llama_bge; ok=1; } ;;
      *) a=runs/rv5/pack_qwen_$x/packing_trace.jsonl; b=runs/rv5/pack_llama_$x/packing_trace.jsonl
         [ -f $a ] && [ -f $b ] && [ $(wc -l < $a) -ge 1150 ] && [ $(wc -l < $b) -ge 1150 ] && {
           s=${x%%_*}; cp $b runs/rv5/llama_$s.jsonl; enq llama_$s
           [ $s = tfidf ] && { cp $a runs/rv5/qwen_$s.jsonl; enq qwen_$s; }; ok=1; } ;;
    esac
    [ $ok = 0 ] && next="$next $x"
  done
  pending=$next; sleep 120
done
echo "[$(date -Is)] AUTO_ENQUEUE_DONE"
