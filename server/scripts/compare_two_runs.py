#!/usr/bin/env python3
"""Compare two complete portable runs at packing and generation token layers."""
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
def rows(p): return {f"{x['task']}/{x['sample_id']}":x for x in (json.loads(line) for line in (p/"raw_predictions.jsonl").open(encoding="utf-8")) if x}
def main():
 ap=argparse.ArgumentParser(); ap.add_argument("--run1",required=True); ap.add_argument("--run2",required=True); args=ap.parse_args(); a=rows(Path(args.run1)); b=rows(Path(args.run2)); keys=sorted(set(a)|set(b)); out=[]
 for k in keys:
  x,y=a.get(k),b.get(k); out.append({"key":k,"packed_ids_exact":bool(x and y and x.get("packed_token_ids_sha256")==y.get("packed_token_ids_sha256")),"input_ids_exact":bool(x and y and x.get("input_ids_sha256")==y.get("input_ids_sha256")),"output_ids_exact":bool(x and y and x.get("output_ids_sha256")==y.get("output_ids_sha256")),"decoded_exact":bool(x and y and x.get("decoded_prediction_sha256")==y.get("decoded_prediction_sha256"))})
 p=Path(args.run2); fields=list(out[0].keys());
 with (p/"RUN1_RUN2_COMPARISON.csv").open("w",newline="",encoding="utf-8") as f: w=csv.DictWriter(f,fieldnames=fields,lineterminator="\n"); w.writeheader(); w.writerows(out)
 counts={k:sum(bool(x[k]) for x in out) for k in fields[1:]}; d={"run1_rows":len(a),"run2_rows":len(b),"comparison_rows":len(out),"counts":counts,"status":"PORTABLE_FRESH_PROCESS_TOKEN_EXACT_PASS" if len(a)==len(b)==len(out)==1150 and all(v==1150 for v in counts.values()) else "PORTABLE_FRESH_PROCESS_TOKEN_EXACT_FAIL"}
 (p/"RUN1_RUN2_COMPARISON.json").write_text(json.dumps(d,indent=2)+"\n",encoding="utf-8"); print(json.dumps(d,indent=2)); return 0 if d["status"].endswith("PASS") else 2
if __name__=="__main__": raise SystemExit(main())
