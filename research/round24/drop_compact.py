"""Remove CompAct from Table 2 and the text of main.tex and main_zh.tex (related-work citation kept)."""
import shutil, sys
D = r"C:\Users\dell\Desktop\BridgePack_ACL\paper"
ROW = "CompAct$^\\dagger$ & -- & 35.70 & -- & -- & 34.70 & -- & -- & 37.53 & -- & -- & -- & -- \\\\\n"
EN = [
    (ROW, ""),
    (r"EXIT, CPC and CompAct were run at 1,024 tokens on the first three models (--: not run).",
     r"EXIT and CPC were run at 1,024 tokens on the first three models (--: not run)."),
    (r"; CPC: sentence encoder; CompAct: abstractive compressor);", r"; CPC: sentence encoder);"),
    (r"five trained compressors (Provence, RECOMP, EXIT, CPC, CompAct) on every model tested.",
     r"four trained compressors (Provence, RECOMP, EXIT, CPC) on every model tested."),
    (r"five trained compressors (Provence, RECOMP, EXIT, CPC, CompAct) on every model tested (Table~\ref{tab:main}).",
     r"four trained compressors (Provence, RECOMP, EXIT, CPC) on every model tested (Table~\ref{tab:main})."),
    (r"above five trained compressors,", r"above four trained compressors,"),
    (r"EXIT \citep{hwang2025exit} (released Gemma-2B adapter, official prompt and 0.5 threshold) and CompAct \citep{yoon2024compact} (released CompAct-7b; five documents per iteration, at most six iterations; the final summary, 76 words on average, is passed to the reader), both applied, like Provence, to the top BGE spans;",
     r"EXIT \citep{hwang2025exit} (released Gemma-2B adapter, official prompt and 0.5 threshold), applied, like Provence, to the top BGE spans;"),
    (r"EXIT \citep{hwang2025exit}, CPC \citep{liskavets2025cpc} and CompAct \citep{yoon2024compact}, which we run at 1,024 tokens on Qwen2.5-7B, Llama-3.1-8B and Qwen2.5-14B, are also below \spk{} with the reranker on every model. Pooled over the three models, \spk{} leads EXIT by 1.93 F1 and CompAct by 8.47 (both intervals exclude zero) and CPC, the strongest of the three, by 0.91;",
     r"EXIT \citep{hwang2025exit} and CPC \citep{liskavets2025cpc}, which we run at 1,024 tokens on Qwen2.5-7B, Llama-3.1-8B and Qwen2.5-14B, are also below \spk{} with the reranker on every model. Pooled over the three models, \spk{} leads EXIT by 1.93 F1 (the interval excludes zero) and CPC, the stronger of the two, by 0.91;"),
]
ZH = [
    (ROW, ""),
    (r"EXIT、CPC 和 CompAct 只在前三个模型上以 1,024 token 运行（--：未运行）。",
     r"EXIT 和 CPC 只在前三个模型上以 1,024 token 运行（--：未运行）。"),
    (r"，CPC 为句子编码器，CompAct 为生成式压缩器）；", r"，CPC 为句子编码器）；"),
    (r"以及五种需要训练的压缩器（Provence、RECOMP、EXIT、CPC、CompAct）。",
     r"以及四种需要训练的压缩器（Provence、RECOMP、EXIT、CPC）。"),
    (r"以及五种需要训练的压缩器（Provence、RECOMP、EXIT、CPC、CompAct）（表~\ref{tab:main}）。",
     r"以及四种需要训练的压缩器（Provence、RECOMP、EXIT、CPC）（表~\ref{tab:main}）。"),
    (r"高于五种需要训练的压缩器；", r"高于四种需要训练的压缩器；"),
    (r"EXIT \citep{hwang2025exit}（发布的 Gemma-2B 适配器，官方提示和 0.5 阈值）和 CompAct \citep{yoon2024compact}（发布的 CompAct-7b；每轮 5 篇文档，最多 6 轮；最终摘要平均 76 词，交给阅读模型），两者与 Provence 一样作用于 BGE 排名靠前的片段；",
     r"EXIT \citep{hwang2025exit}（发布的 Gemma-2B 适配器，官方提示和 0.5 阈值），与 Provence 一样作用于 BGE 排名靠前的片段；"),
    (r"我们在 Qwen2.5-7B、Llama-3.1-8B 和 Qwen2.5-14B 上以 1,024 token 运行了 EXIT \citep{hwang2025exit}、CPC \citep{liskavets2025cpc} 和 CompAct \citep{yoon2024compact}，它们在每个模型上也都低于配合重排序器的 \spk{}。三个模型合并后，\spk{} 比 EXIT 高 1.93 F1、比 CompAct 高 8.47（两个区间都不包含 0），比三者中最强的 CPC 高 0.91；",
     r"我们在 Qwen2.5-7B、Llama-3.1-8B 和 Qwen2.5-14B 上以 1,024 token 运行了 EXIT \citep{hwang2025exit} 和 CPC \citep{liskavets2025cpc}，它们在每个模型上也都低于配合重排序器的 \spk{}。三个模型合并后，\spk{} 比 EXIT 高 1.93 F1（区间不包含 0），比两者中较强的 CPC 高 0.91；"),
]
for name, edits in (("main.tex", EN), ("main_zh.tex", ZH)):
    p = D + "\\" + name
    t = open(p, encoding="utf-8").read()
    for i, (a, b) in enumerate(edits):
        if t.count(a) != 1:
            sys.exit(f"{name} edit {i}: found {t.count(a)}x: {a[:70]}")
        t = t.replace(a, b)
    left = [l for l in t.splitlines() if "CompAct" in l and "yoon2024compact" not in l]
    shutil.copy(p, p + ".bak_r25")
    open(p, "w", encoding="utf-8", newline="").write(t)
    print(name, "ok; other CompAct mentions:", len(left))
