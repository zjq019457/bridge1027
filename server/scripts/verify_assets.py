#!/usr/bin/env python3
"""Verify portable source/model/data/golden checksums before running."""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
def sha(p):
 h=hashlib.sha256()
 with p.open("rb") as f:
  for b in iter(lambda:f.read(4*1024*1024),b""): h.update(b)
 return h.hexdigest()
def check(manifest,target):
 failures=[]; total=0
 for line in manifest.read_text(encoding="utf-8").splitlines():
  if not line.strip(): continue
  want,rel=line.split("  ",1); p=target/rel; total+=1
  if not p.is_file(): failures.append(f"MISSING {rel}")
  elif sha(p)!=want: failures.append(f"HASH {rel}")
 return total,failures
def main():
 ap=argparse.ArgumentParser(); ap.add_argument("--model-dir",required=True); args=ap.parse_args(); root=Path(__file__).resolve().parents[1]
 checks=[("source",root/"manifests/SOURCE_SHA256SUMS.txt",root/"source/SlimInfer_snapshot"),("model",root/"manifests/MODEL_TOKENIZER_SHA256SUMS.txt",Path(args.model_dir)),("dataset",root/"manifests/DATASET_SHA256SUMS.txt",root/"assets/longbench_g1_data"),("golden",root/"manifests/GOLDEN_SHA256SUMS.txt",root/"golden")]
 result={}; bad=[]
 for name,m,t in checks:
  n,f=check(m,t); result[name]={"files":n,"failures":f}; bad += [f"{name}:{x}" for x in f]
 result["status"]="ASSET_VERIFICATION_PASS" if not bad else "ASSET_VERIFICATION_FAIL"
 (root/"reports/ASSET_VERIFICATION.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8"); print(json.dumps(result,indent=2)); return 0 if not bad else 1
if __name__=="__main__": raise SystemExit(main())
