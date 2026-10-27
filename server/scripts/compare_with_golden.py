#!/usr/bin/env python3
"""Compare portable output token layers and decoded predictions with Golden V1."""
from __future__ import annotations
import argparse,csv,gzip,hashlib,json
from pathlib import Path
def read_jsonl(p): return [json.loads(x) for x in p.open(encoding="utf-8") if x.strip()]
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--run-dir",required=True); ap.add_argument("--golden",required=True); args=ap.parse_args(); run=Path(args.run_dir); golden=Path(args.golden)
    got={f"{x['task']}/{x['sample_id']}":x for x in read_jsonl(run/"raw_predictions.jsonl")}
    ref={}
    with gzip.open(golden/"GOLDEN_REFERENCE_V1.jsonl.gz","rt",encoding="utf-8") as f:
        for line in f:
            if line.strip():
                x=json.loads(line); ref[f"{x['task']}/{x['sample_id']}"]=x
    rows=[]
    expected=int(json.loads((run/"run_manifest.json").read_text(encoding="utf-8")).get("sample_count",len(got)))
    for key,x in got.items():
        g=ref.get(key)
        if g is None:
            rows.append({"task":x.get("task",""),"sample_id":x.get("sample_id",""),"packed_ids_exact":False,"input_ids_exact":False,"output_ids_exact":False,"decoded_exact":False,"first_divergence_stage":"MISSING_GOLDEN"}); continue
        checks={"packed_ids_exact":x.get("packed_token_ids_sha256")==g.get("packed_token_ids_sha256"),"input_ids_exact":x.get("input_ids_sha256")==g.get("input_ids_sha256"),"output_ids_exact":x.get("output_ids_sha256")==g.get("output_ids_sha256"),"decoded_exact":x.get("decoded_prediction_sha256")==g.get("decoded_prediction_sha256")}; stage="PASS"
        for k in ["packed_ids_exact","input_ids_exact","output_ids_exact","decoded_exact"]:
            if not checks[k]: stage=k; break
        rows.append({"task":g["task"],"sample_id":g["sample_id"],**checks,"first_divergence_stage":stage,"score":x.get("score"),"golden_score":g.get("score")})
    fields=list(rows[0].keys()) if rows else ["task","sample_id","first_divergence_stage"]
    with (run/"GOLDEN_COMPARISON.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=fields,lineterminator="\n"); w.writeheader(); w.writerows(rows)
    counts={k:sum(bool(r.get(k)) for r in rows) for k in ["packed_ids_exact","input_ids_exact","output_ids_exact","decoded_exact"]}; first=next((r["first_divergence_stage"] for r in rows if r["first_divergence_stage"]!="PASS"),"NONE")
    d={"rows":len(rows),"expected_rows":expected,"golden_rows":len(ref),"counts":counts,"first_divergence_stage":first,"all_token_exact":len(rows)==expected and all(counts[k]==expected for k in counts),"status":"PORTABLE_TOKEN_EXACT_PASS" if len(rows)==expected and all(counts[k]==expected for k in counts) else "PORTABLE_TOKEN_EXACT_FAIL"}
    (run/"GOLDEN_COMPARISON.json").write_text(json.dumps(d,indent=2)+"\n",encoding="utf-8"); print(json.dumps(d,indent=2)); return 0 if d["status"]=="PORTABLE_TOKEN_EXACT_PASS" else 2
if __name__=="__main__": raise SystemExit(main())
