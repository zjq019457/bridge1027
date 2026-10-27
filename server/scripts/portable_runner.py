#!/usr/bin/env python3
"""Run the frozen BridgePack fresh protocol against Portable Golden V1."""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "source" / "SlimInfer_snapshot"
sys.path.insert(0, str(SOURCE))
TASKS = ["qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "musique", "narrativeqa"]
VARIANT = "same_spans_source_order"
BUDGET = 1024

def now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def sha_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8", errors="replace")).hexdigest()

def read_jsonl(p: Path) -> list[dict[str, Any]]:
    if not p.exists(): return []
    with p.open(encoding="utf-8") as f: return [json.loads(x) for x in f if x.strip()]

def write_json(p: Path, obj: Any) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

def write_jsonl(p: Path, rows: list[dict[str, Any]]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for x in rows: f.write(json.dumps(x, ensure_ascii=False, separators=(",", ":")) + "\n")

def golden_rows(golden: Path) -> list[dict[str, Any]]:
    with gzip.open(golden / "GOLDEN_REFERENCE_V1.jsonl.gz", "rt", encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]

def dataset_rows(data_root: Path, task: str) -> dict[str, dict[str, Any]]:
    return {str(x["_id"]): x for x in read_jsonl(data_root / f"{task}.jsonl")}

def load_selection_inputs(data_root: Path, golden: Path, limit: int | None):
    gold = golden_rows(golden)
    ids: dict[str, list[str]] = {t: [] for t in TASKS}
    samples: dict[str, list[dict[str, Any]]] = {t: [] for t in TASKS}
    gold_by: dict[str, dict[str, Any]] = {}
    sources = {t: dataset_rows(data_root, t) for t in TASKS}
    for g in gold:
        task, sid = str(g["task"]), str(g["sample_id"])
        if limit == 12 and len(ids[task]) >= 2: continue
        if limit is not None and sum(len(v) for v in ids.values()) >= limit: break
        d = sources[task][sid]
        ids[task].append(sid)
        samples[task].append({"task": task, "sample_id": sid, "question": d["input"], "context": d["context"], "original_context": d["context"], "answers": d.get("answers", []), "all_classes": d.get("all_classes"), "length": d.get("length", 0)})
        gold_by[f"{task}/{sid}"] = g
    expected = 12 if limit == 12 else len(gold)
    if sum(len(v) for v in ids.values()) != expected: raise RuntimeError("selection input count mismatch")
    return ids, samples, gold_by

def read_traces(run_dir: Path, ids: dict[str, list[str]]) -> list[dict[str, Any]]:
    rows=[]
    for task in TASKS:
        p=run_dir/"packing"/"packed_prompts_build"/"packed_prompts"/VARIANT/f"budget_{BUDGET}"/f"{task}.jsonl"
        got={str(x["sample_id"]):x for x in read_jsonl(p)}
        rows.extend(got[sid] for sid in ids[task])
    return rows

def trace_rows(run_dir: Path, ids: dict[str, list[str]], tokenizer: Any) -> list[dict[str, Any]]:
    rows=[]
    for x in read_traces(run_dir, ids):
        packed_ids=list(tokenizer(str(x["packed_text"]), add_special_tokens=False).input_ids)
        rows.append({**x, "packed_token_ids": packed_ids, "packed_token_ids_sha256": sha_text(json.dumps(packed_ids, separators=(",", ":")))})
    write_jsonl(run_dir/"packing_trace.jsonl", rows)
    return rows

def load_or_pack(run_dir: Path, ids, samples, tokenizer: Any, resume: bool):
    expected=sum(map(len, ids.values()))
    existing=read_jsonl(run_dir/"packing_trace.jsonl")
    if resume and len(existing)==expected: return existing
    from tools.bridgepack_g1 import run_g1 as g1
    g1.QWEN_PATH=Path(os.environ.get("BRIDGEPACK_MODEL_PATH", "portable-model"))
    g1.build_matched_variant_packs(run_dir, ids, samples, tokenizer, stage="packing", variant_filter=[VARIANT])
    return trace_rows(run_dir, ids, tokenizer)

def generate(run_dir: Path, ids, traces, gold_by, model_path: Path):
    import torch
    from tools.bridgepack_g1.generate import load_qwen_fp16
    from tools.run_evidence_text_packing_generation_qwen32 import _max_new_tokens_for
    from longbench_eval import scorer
    model, tokenizer=load_qwen_fp16(str(model_path), torch)
    tr={(x["task"],str(x["sample_id"])):x for x in traces}; rows=[]
    for task in TASKS:
        for sid in ids[task]:
            x=tr[(task,sid)]; prompt=str(x["generation_prompt"])
            chat=tokenizer.apply_chat_template([{"role":"user","content":prompt}], tokenize=False, add_generation_prompt=True) if hasattr(tokenizer,"apply_chat_template") else prompt
            inputs=tokenizer(chat,truncation=False,return_tensors="pt").to(getattr(model,"device",next(model.parameters()).device))
            input_ids=inputs.input_ids[0].tolist(); max_new=_max_new_tokens_for(task,0)
            with torch.no_grad(): out=model.generate(**inputs,max_new_tokens=max_new,num_beams=1,do_sample=False,use_cache=True,eos_token_id=tokenizer.eos_token_id)[0]
            output_ids=out.tolist()[len(input_ids):]; pred=tokenizer.decode(output_ids,skip_special_tokens=True)
            score=float(scorer(task,[pred],[x.get("answers",[])],[len(input_ids)],x.get("all_classes"))); g=gold_by[f"{task}/{sid}"]
            rows.append({"task":task,"sample_id":sid,"answers":x.get("answers",[]),"all_classes":x.get("all_classes"),"packed_context":x["packed_text"],"packed_token_ids":x["packed_token_ids"],"packed_token_ids_sha256":x["packed_token_ids_sha256"],"generation_prompt":prompt,"generation_prompt_sha256":sha_text(prompt),"input_token_ids":input_ids,"input_ids_sha256":sha_text(json.dumps(input_ids,separators=(",",":"))),"output_token_ids":output_ids,"output_ids_sha256":sha_text(json.dumps(output_ids,separators=(",",":"))),"prediction":pred,"decoded_prediction_sha256":sha_text(pred),"score":score,"input_token_count":len(input_ids),"output_token_count":len(output_ids),"max_new_tokens":max_new,"golden_output_ids_sha256":g["output_ids_sha256"],"golden_prediction_sha256":g["decoded_prediction_sha256"]})
            if len(rows)%25==0: write_json(run_dir/"progress.json",{"completed":len(rows),"total":sum(map(len,ids.values())),"timestamp":now()})
    write_jsonl(run_dir/"raw_predictions.jsonl",rows)
    task_scores=[]; score_by={}
    for task in TASKS:
        subset=[r for r in rows if r["task"]==task]
        score=float(scorer(task,[r["prediction"] for r in subset],[r.get("answers",[]) for r in subset],[r["input_token_count"] for r in subset],None)); score_by[task]=score; task_scores.append({"task":task,"sample_count":len(subset),"score":score})
    macro=sum(score_by.values())/len(TASKS); task_scores.append({"task":"UNWEIGHTED_MACRO","sample_count":len(rows),"score":macro})
    with (run_dir/"per_task_scores.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=["task","sample_count","score"],lineterminator="\n"); w.writeheader(); w.writerows(task_scores)
    write_json(run_dir/"macro_summary.json",{"sample_count":len(rows),"task_scores":score_by,"macro":macro,"golden_macro":39.35166666666667})
    return rows

def main() -> int:
    ap=argparse.ArgumentParser(); ap.add_argument("--model-path",required=True); ap.add_argument("--data-root",required=True); ap.add_argument("--golden",default=str(ROOT/"golden")); ap.add_argument("--output-dir",required=True); ap.add_argument("--sample-limit",type=int,choices=[12,1150],default=1150); ap.add_argument("--resume",action="store_true"); ap.add_argument("--packing-only",action="store_true"); ap.add_argument("--qa-only",action="store_true"); args=ap.parse_args()
    run_dir=Path(args.output_dir).resolve(); run_dir.mkdir(parents=True,exist_ok=True); ids,samples,gold_by=load_selection_inputs(Path(args.data_root),Path(args.golden),args.sample_limit)
    write_json(run_dir/"run_manifest.json",{"created_at":now(),"sample_count":sum(map(len,ids.values())),"task_counts":{t:len(ids[t]) for t in TASKS},"variant":VARIANT,"budget":BUDGET,"model_path":str(Path(args.model_path).resolve()),"portable_golden":"golden/GOLDEN_REFERENCE_V1.jsonl.gz"})
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(str(Path(args.model_path).resolve()),local_files_only=True,trust_remote_code=True,use_fast=True)
    traces=read_jsonl(run_dir/"packing_trace.jsonl") if args.resume or args.qa_only else []
    if not traces: traces=load_or_pack(run_dir,ids,samples,tokenizer,args.resume)
    if len(traces)!=sum(map(len,ids.values())): raise RuntimeError("packing output count mismatch")
    if args.packing_only: write_json(run_dir/"status.json",{"status":"PACKING_ONLY","rows":len(traces)}); return 0
    rows=read_jsonl(run_dir/"raw_predictions.jsonl") if args.qa_only and args.resume else []
    if not rows: rows=generate(run_dir,ids,traces,gold_by,Path(args.model_path).resolve())
    write_json(run_dir/"environment.json",{"python":sys.executable,"model_path":str(Path(args.model_path).resolve()),"dtype":"float16","do_sample":False,"num_beams":1})
    write_json(run_dir/"status.json",{"status":"COMPLETE","rows":len(rows),"output_ids_saved":True}); return 0

if __name__=="__main__": raise SystemExit(main())
