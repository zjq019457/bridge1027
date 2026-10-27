"""Parametrize BridgePack coefficients in the sensitivity copy. Defaults equal the original constants."""
import sys
from pathlib import Path

root = Path(sys.argv[1])  # .../source/SlimInfer_snapshot

params = root / "bridgekv" / "sensitivity_params.py"
params.write_text('''"""Sensitivity-study coefficients. Defaults reproduce the frozen BridgePack configuration.

Override with env var BP_SENS_JSON='{"lam": 0.0}'.
"""
import json
import os
from types import SimpleNamespace

DEFAULTS = {
    "alpha": 0.5,       # BM25 weight in lexical init; TF-IDF gets 1 - alpha
    "c_div": 0.10,      # L_bridge novelty term (1 - J)
    "c_jac": 0.35,      # L_bridge near-duplicate penalty (-J)
    "c_sup": 0.5,       # support utility weight in S_path
    "lam": 0.55,        # redundancy penalty in greedy selection
    "neighbor": 32,     # neighbor repair radius (regex tokens)
    "bridge_zero": False,  # L_bridge == 0
}

_over = json.loads(os.environ.get("BP_SENS_JSON", "{}") or "{}")
_unknown = set(_over) - set(DEFAULTS)
if _unknown:
    raise ValueError(f"unknown sensitivity params: {sorted(_unknown)}")
SP = SimpleNamespace(**{**DEFAULTS, **_over})
''')


def sub_once(path, old, new):
    text = path.read_text()
    n = text.count(old)
    if n != 1:
        raise SystemExit(f"{path.name}: expected 1 match, found {n}: {old!r}")
    path.write_text(text.replace(old, new))


def sub_line(path, lineno, old, new):
    lines = path.read_text().split("\n")
    line = lines[lineno - 1]
    if line.count(old) != 1:
        raise SystemExit(f"{path.name}:{lineno}: {old!r} not found once in {line!r}")
    lines[lineno - 1] = line.replace(old, new)
    path.write_text("\n".join(lines))


etp = root / "bridgekv" / "evidence_text_packing.py"
sub_once(etp, "from collections import Counter, defaultdict\n",
         "from collections import Counter, defaultdict\n\nfrom bridgekv.sensitivity_params import SP as _SP\n")
sub_once(etp, "    too_similar_penalty = 0.35 * jaccard(toks_i, toks_j)\n",
         "    if _SP.bridge_zero:\n        return 0.0\n    too_similar_penalty = _SP.c_jac * jaccard(toks_i, toks_j)\n")
sub_once(etp, "    novelty_bonus = 0.10 * (1.0 - jaccard(toks_i, toks_j))\n",
         "    novelty_bonus = _SP.c_div * (1.0 - jaccard(toks_i, toks_j))\n")
sub_once(etp, "        scores[span.span_id] = 0.5 * bm25_norm[idx] + 0.5 * tfidf_norm[idx]\n",
         "        scores[span.span_id] = _SP.alpha * bm25_norm[idx] + (1.0 - _SP.alpha) * tfidf_norm[idx]\n")
sub_once(etp, "            score = query_link + bridge + 0.5 * right.final_score\n",
         "            score = query_link + bridge + _SP.c_sup * right.final_score\n")
sub_once(etp, "            adjusted = path.path_score - 0.55 * redundancy\n",
         "            adjusted = path.path_score - _SP.lam * redundancy\n")

ext = root / "bridgekv" / "bridgepack_cpu_extensions.py"
# line numbers verified against the frozen source before patching; edits above do not touch this file
sub_line(ext, 452, "neighbor_size=32", "neighbor_size=_SP.neighbor")
sub_line(ext, 456, "neighbor_size=32", "neighbor_size=_SP.neighbor")
sub_line(ext, 530, "page_map, 32,", "page_map, _SP.neighbor,")
sub_line(ext, 553, "neighbor_size=32", "neighbor_size=_SP.neighbor")
sub_once(ext, "from bridgekv import evidence_text_packing as etp\n",
         "from bridgekv import evidence_text_packing as etp\nfrom bridgekv.sensitivity_params import SP as _SP\n")

g1 = root / "tools" / "bridgepack_g1" / "run_g1.py"
lines = g1.read_text().split("\n")
assert lines[475].strip() == 'frozen["page_map"],', lines[475]
assert lines[476].strip() == "32,", lines[476]
lines[476] = lines[476].replace("32,", "_SP.neighbor,")
text = "\n".join(lines)
assert text.count("\nimport ") >= 1
text = text.replace("\nimport ", "\nfrom bridgekv.sensitivity_params import SP as _SP\nimport ", 1)
g1.write_text(text)
print("PATCH_OK")
