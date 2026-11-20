"""Figure 2 caption: DedupPack replaces only the packing step (main.tex and main_zh.tex); fixes a garbled ZH clause."""
import shutil, sys
D = r"C:\Users\dell\Desktop\BridgePack_ACL\paper"
EDITS = {
    "main.tex": [
        ("budget unused. \\spk{} pads spans to sentence boundaries",
         "budget unused. \\spk{} replaces only the packing step: it pads spans to sentence boundaries"),
        ("It runs on the CPU, needs no training and does not modify the model; compared",
         "It runs on the CPU, needs no training and leaves the ranker and the model unchanged; compared"),
    ],
    "main_zh.tex": [
        ("预算。\\spk{} 把片段补全到句子边界",
         "预算。\\spk{} 只替换打包这一步：它把片段补全到句子边界"),
        ("它在 CPU 上运行，无需训练，也不修改模型；与完整上下文相比，LLM 只需在 1,024 token 预算下，LLM 预填充的 token",
         "它在 CPU 上运行，无需训练，排序器和模型都保持不变；在 1,024 token 预算下，LLM 预填充的 token"),
    ],
}
for name, pairs in EDITS.items():
    p = D + "\\" + name
    t = open(p, encoding="utf-8").read()
    for old, new in pairs:
        if t.count(old) != 1:
            sys.exit(f"{name}: found {t.count(old)}x: {old[:40]}")
        t = t.replace(old, new)
    shutil.copy(p, p + ".bak_r32")
    open(p, "w", encoding="utf-8", newline="").write(t)
    print(name, "ok")
