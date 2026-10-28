"""BM25 / TF-IDF chunk packing under the exact 1,024-token model budget, via the repo s
_exact_repack (nominal pack_evidence selection, then HF-token admission in source order).
usage: lex_pack.py VARIANT TOKENIZER OUT
"""
import json
import sys
from pathlib import Path

W = Path("/diskE/jiaqizhang/BRIDGEPACK_SENSITIVITY_V1_20261025")
VARIANT, TOKP, OUT = sys.argv[1], sys.argv[2], Path(sys.argv[3])
sys.path.insert(0, str(W / "source" / "SlimInfer_snapshot"))
sys.path.insert(0, str(W / "scripts"))
import portable_runner as pr  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402
from tools.run_bridgepack_exact_budget_repack import _exact_repack  # noqa: E402

ids, samples, _ = pr.load_selection_inputs(W / "assets" / "longbench_g1_data", W / "golden", None)
tok = AutoTokenizer.from_pretrained(TOKP, local_files_only=True, use_fast=True)
with OUT.open("w") as f:
    for t in pr.TASKS:
        for s in samples[t]:
            f.write(json.dumps(_exact_repack(tok, s, VARIANT, 1024, TOKP)) + "\n")
print("LEX_PACK_DONE", OUT)
