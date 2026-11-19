"""State that each model reads contexts packed with its own tokenizer (main.tex and main_zh.tex)."""
import shutil, sys
D = r"C:\Users\dell\Desktop\BridgePack_ACL\paper"
EDITS = {
    "main.tex": [
        ("The 14B and 32B models read the packed contexts built with the Qwen tokenizer, which they share with the 7B model.",
         "Each model reads contexts packed with its own tokenizer: Llama-3.1 with the Llama tokenizer, and the 14B and 32B models with the Qwen tokenizer they share with the 7B model."),
        ("(LongBench test, 1,150 instances, Qwen tokenizer; Llama-3.1 is within one point on every column)",
         "(LongBench test, 1,150 instances, packed with the Qwen tokenizer; with the Llama tokenizer every column is within one point)"),
    ],
    "main_zh.tex": [
        ("14B 和 32B 模型读取用 Qwen 分词器打包的上下文，该分词器与 7B 模型相同。",
         "每个模型读取用它自己的分词器打包的上下文：Llama-3.1 使用 Llama 分词器，14B 和 32B 模型使用与 7B 模型相同的 Qwen 分词器。"),
        ("（LongBench 测试集，1,150 个样本，Qwen 分词器；Llama-3.1 在每一列上的差别都在 1 个点以内）",
         "（LongBench 测试集，1,150 个样本，用 Qwen 分词器打包；改用 Llama 分词器时每一列的差别都在 1 个点以内）"),
    ],
}
for name, pairs in EDITS.items():
    p = D + "\\" + name
    t = open(p, encoding="utf-8").read()
    for old, new in pairs:
        if t.count(old) != 1:
            sys.exit(f"{name}: found {t.count(old)}x: {old[:40]}")
        t = t.replace(old, new)
    shutil.copy(p, p + ".bak_r31")
    open(p, "w", encoding="utf-8", newline="").write(t)
    print(name, "ok")
