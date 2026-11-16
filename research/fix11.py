import os
os.chdir("/root/autodl-tmp/bridge1027/research")
s = open("longlingua.py").read()
o = 'pc = PromptCompressor(model_name=__import__("os").environ["LLL_PATH"], device_map="cuda")\n'
if "DynamicCache" not in s:
    assert s.count(o) == 1
    s = s.replace(o, o + '''# transformers>=4.48 expects a Cache object; llmlingua 0.2.2 passes legacy lists
from transformers import DynamicCache
_fwd = pc.model.forward
def _fw(*a, **k):
    p = k.get("past_key_values")
    if isinstance(p, (list, tuple)) and len(p):
        k["past_key_values"] = DynamicCache.from_legacy_cache(tuple((x[0], x[1]) for x in p))
    return _fwd(*a, **k)
pc.model.forward = _fw
''')
    open("longlingua.py", "w").write(s)
# run10: BGE-large and reranker packs wrote the same file names; pack one encoder at a time and prefix the outputs
r = open("run10.sh").read()
o = '''  ( cd $D; for e in bgel rr; do for b in 512 2048; do
      BP_TOK=$TK BP_ENC=$e python ../evalcov.py bge_chunk $T 100000 budget=$b &
      BP_TOK=$TK BP_ENC=$e python ../evalcov.py lex_rep $T 100000 seed_mode=1 sent=1 budget=$b &
    done; done; wait; touch PACK_DONE )'''
n = '''  ( cd $D; mkdir -p tmp/out; for e in bgel rr; do for b in 512 2048; do
      ( cd tmp; BP_TOK=$TK BP_ENC=$e python ../../evalcov.py bge_chunk $T 100000 budget=$b ) &
      ( cd tmp; BP_TOK=$TK BP_ENC=$e python ../../evalcov.py lex_rep $T 100000 seed_mode=1 sent=1 budget=$b ) &
    done; wait; for x in tmp/out/*.jsonl; do mv "$x" "out/${e}_$(basename $x)"; done; done; touch PACK_DONE )'''
assert r.count(o) == 1
open("run10.sh", "w").write(r.replace(o, n))
print("ok")
