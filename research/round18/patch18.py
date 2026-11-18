"""Patch packers.py on the server: pool-size parameter for bridge_hop/lex_rep (default 60, unchanged
behaviour) and append the round-18 baselines."""
import shutil
p = "packers.py"
shutil.copy(p, p + ".bak18")
t = open(p, encoding="utf-8").read()
reps = [
    ("def bridge_hop(s, tok, budget=1024, k1=6, radius=32, beta=0.3, hop2_share=0.5, max_ents=12, seed_mode=0, sent=0):",
     "def bridge_hop(s, tok, budget=1024, k1=6, radius=32, beta=0.3, hop2_share=0.5, max_ents=12, seed_mode=0, sent=0, pool=60):"),
    ("        if len(priority) >= 60:\n            break\n    return _admit(tok, s, priority, page_map, budget, radius, sent)",
     "        if len(priority) >= pool:\n            break\n    return _admit(tok, s, priority, page_map, budget, radius, sent)"),
    ("def lex_rep(s, tok, budget=1024, radius=32, seed_mode=0, sent=0):",
     "def lex_rep(s, tok, budget=1024, radius=32, seed_mode=0, sent=0, pool=60):"),
    ("    return bridge_hop(s, tok, budget, radius=radius, hop2_share=0.0, seed_mode=seed_mode, sent=sent)",
     "    return bridge_hop(s, tok, budget, radius=radius, hop2_share=0.0, seed_mode=seed_mode, sent=sent, pool=pool)"),
]
for a, b in reps:
    assert t.count(a) == 1, a[:60]
    t = t.replace(a, b)
assert "def sel_ctx" not in t
t += open("packers_add.py", encoding="utf-8").read()
open(p, "w", encoding="utf-8").write(t)
print("patched")
