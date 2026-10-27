#!/usr/bin/env python3
from pathlib import Path
import hashlib

AUDIT = Path(__file__).resolve().parents[2] / "results" / "bridgepack_g1_strict_reproduction_audit_20260928_102546"

def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

paths = sorted(p for p in AUDIT.rglob("*") if p.is_file() and p.name != "SHA256SUMS.txt")
(AUDIT / "SHA256SUMS.txt").write_text(
    "\n".join(f"{digest(p)}  {p.relative_to(AUDIT)}" for p in paths) + "\n",
    encoding="utf-8",
)
print(f"hashed {len(paths)} audit files")
