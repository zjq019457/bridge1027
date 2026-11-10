"""Full-context reference trace: LongBench context middle-truncated to MAXTOK downstream tokens
(LongBench convention). usage: BP_TOK=<tokdir> python fullctx32.py OUTFILE [MAXTOK]"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import localpack as L

out = Path(sys.argv[1]); maxtok = int(sys.argv[2]) if len(sys.argv) > 2 else 30000
tok = L.HFLike(L.TOKDIR)
with open(out, "w", encoding="utf-8") as f:
    for s in L.load_samples():
        ids = tok(s["context"]).input_ids
        ctx = s["context"]
        if len(ids) > maxtok:
            h = maxtok // 2
            ctx = tok.decode(ids[:h]) + tok.decode(ids[-h:])
        f.write(json.dumps({"task": s["task"], "sample_id": s["sample_id"], "packed_text": ctx,
                            "question": s["question"], "answers": s["answers"]}) + "\n")
print("wrote", out)
