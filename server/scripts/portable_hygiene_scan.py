#!/usr/bin/env python3
from __future__ import annotations
import re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
TEXT_EXT={".py",".sh",".md",".json",".csv",".txt",".def",".yml",".yaml",".jsonl"}
SECRET=re.compile(r"(BEGIN (RSA|OPENSSH|EC) PRIVATE KEY|sk-[A-Za-z0-9]{20,}|gh[pousr]_[A-Za-z0-9]{20,}|hf_[A-Za-z0-9]{20,}|api[_-]?key\s*[:=]|password\s*[:=])",re.I)
def main():
    secret=[]; abs_paths=[]; large=[]
    for p in ROOT.rglob("*"):
        if not p.is_file() or "__pycache__" in p.parts or "runtime" in p.parts: continue
        if p.name in {"PORTABLE_SECRET_SCAN.txt", "PORTABLE_ABSOLUTE_PATH_SCAN.txt", "PORTABLE_LARGE_FILE_SCAN.txt"}: continue
        if p.stat().st_size > 100*1024*1024: large.append(f"{p.relative_to(ROOT)}\t{p.stat().st_size}")
        if p.suffix.lower() not in TEXT_EXT or p.stat().st_size > 20*1024*1024: continue
        text=p.read_text(encoding="utf-8",errors="ignore")
        for i,line in enumerate(text.splitlines(),1):
            if SECRET.search(line): secret.append(f"{p.relative_to(ROOT)}:{i}:{line[:200]}")
            if "/home/" in line or "/mnt/" in line: abs_paths.append(f"{p.relative_to(ROOT)}:{i}:{line[:240]}")
    (ROOT/"reports/PORTABLE_SECRET_SCAN.txt").write_text("SECRET_PATTERN_MATCHES\n"+("\n".join(secret) if secret else "NONE")+"\n",encoding="utf-8")
    (ROOT/"reports/PORTABLE_ABSOLUTE_PATH_SCAN.txt").write_text("ABSOLUTE_OR_MOUNT_PATHS (expected in provenance/instructions)\n"+("\n".join(abs_paths) if abs_paths else "NONE")+"\n",encoding="utf-8")
    (ROOT/"reports/PORTABLE_LARGE_FILE_SCAN.txt").write_text("FILES >100 MiB (runtime unpack excluded from final bundle; conda archive is expected)\n"+("\n".join(large) if large else "NONE")+"\n",encoding="utf-8")
    print({"secret_matches":len(secret),"absolute_or_mount_lines":len(abs_paths),"large_files":len(large)})
    return 1 if secret else 0
if __name__=="__main__": raise SystemExit(main())
