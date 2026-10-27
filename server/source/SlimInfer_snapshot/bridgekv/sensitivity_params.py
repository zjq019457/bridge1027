"""Sensitivity-study coefficients. Defaults reproduce the frozen BridgePack configuration.

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
