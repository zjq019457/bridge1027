"""Fill round-18 results into main.tex / main_zh.tex from r18.json (null -> \\pend{}).
Re-runnable: the appendix blocks live between % R18-*-BEGIN / % R18-*-END markers.
usage (in the paper directory): python gen_r18.py path/to/r18.json"""
import json, re, sys

V = json.load(open(sys.argv[1], encoding="utf-8"))
P = r"\pend{}"
M3 = ["q7", "llama", "q14"]


def f(x, bold=False):
    if x is None:
        return P
    if isinstance(x, list):  # sampling: mean, sd
        return f"{x[0]:.2f} $\\pm$ {x[1]:.2f}"
    s = f"{x:.2f}"
    return f"\\textbf{{{s}}}" if bold else s


def best_of(rows, keys, col):
    """Bold the column maximum only once every row in the column is filled."""
    vals = [rows[k][col] for k in keys]
    return max(vals) if all(v is not None for v in vals) else None


def table(rows_spec, data, cols, head, cap, label, bold_keys=None, colspec=None):
    L = [r"\begin{table}[t]", r"\centering", r"\small", r"\setlength{\tabcolsep}{3pt}",
         r"\resizebox{\ifdim\width>\columnwidth\columnwidth\else\width\fi}{!}{%", r"\begin{tabular}{" + (colspec or "l" + "c" * len(cols)) + "}",
         r"\toprule", head + r" \\", r"\midrule"]
    best = {c: best_of(data, bold_keys, c) for c in cols} if bold_keys else {}
    for spec in rows_spec:
        if spec == "mid":
            L.append(r"\midrule"); continue
        if spec[0] == "group":
            L.append(rf"\multicolumn{{{len(cols) + 1}}}{{l}}{{\textit{{{spec[1]}}}}} \\"); continue
        key, name = spec
        cells = [f(data[key][c], bold=(best.get(c) is not None and data[key][c] == best[c])) for c in cols]
        L.append(name + " & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}}", cap, rf"\label{{{label}}}", r"\end{table}"]
    return "\n".join(L)


def tables(zh):
    T = {}
    nm = (dict(head="开头截断", mid="中间截断", sc="Selective Context", recomp="RECOMP（抽取式）",
               bge_chunk="BGE-small 切块", bge_dp="BGE-small + \\spk{}", hyb_dp="混合 + \\spk{}")
          if zh else
          dict(head="Head truncation", mid="Middle truncation", sc="Selective Context", recomp="RECOMP (extractive)",
               bge_chunk="BGE-small chunks", bge_dp="BGE-small + \\spk{}", hyb_dp="Hybrid + \\spk{}"))
    mh = "Qwen-7B & Llama-8B & Qwen-14B"
    # further baselines, 1,024 tokens
    T["base"] = table(
        [(k, nm[k]) for k in ["head", "mid", "sc", "recomp"]] + ["mid"] + [(k, nm[k]) for k in ["bge_chunk", "bge_dp"]],
        V["base"], M3, ("1,024 token & " if zh else "1,024 tokens & ") + mh,
        (r"\caption{相同 1,024 token 精确预算下的更多基线（宏平均 F1，1,150 个样本）。开头截断保留文档开头；中间截断按 LongBench 的做法保留开头和结尾。Selective Context \citep{li2023selectivecontext} 删除 GPT-2 自信息最低的短语；RECOMP \citep{xu2023recomp} 按其抽取式压缩器（HotpotQA 检查点）的句子排序依次加入句子，直到填满预算。}"
         if zh else
         r"\caption{Further baselines under the same exact 1,024-token budget (macro F1, 1,150 instances). Head truncation keeps the beginning of the document; middle truncation keeps its beginning and end, as in LongBench. Selective Context \citep{li2023selectivecontext} removes the phrases with the lowest GPT-2 self-information; RECOMP \citep{xu2023recomp} adds sentences in the order given by its extractive compressor (HotpotQA checkpoint) until the budget is full.}"),
        "tab:base18", bold_keys=["head", "mid", "sc", "recomp", "bge_chunk", "bge_dp"])
    # 256 tokens, Qwen2.5-14B
    d256 = {k: {"q14": v} for k, v in V["b256"].items()}
    T["b256"] = table(
        [(k, nm[k]) for k in ["head", "mid"]] + ["mid"] + [(k, nm[k]) for k in ["bge_chunk", "bge_dp", "hyb_dp"]],
        d256, ["q14"], "Qwen2.5-14B, 256 token & F1" if zh else "Qwen2.5-14B, 256 tokens & F1",
        r"\caption{Qwen2.5-14B 在 256 token 精确预算下的结果（宏平均 F1）。}" if zh else
        r"\caption{Qwen2.5-14B under an exact 256-token budget (macro F1).}",
        "tab:b256", bold_keys=list(d256))
    # packer sensitivity
    sn = (dict(default="默认（$r=32$，60 个片段）", r16="$r=16$", r64="$r=64$", p30="$P$ 取 30 个片段", p120="$P$ 取 120 个片段", bge_chunk="BGE-small 切块")
          if zh else
          dict(default="Default ($r=32$, 60 spans)", r16="$r=16$", r64="$r=64$", p30="$P$ of 30 spans", p120="$P$ of 120 spans", bge_chunk="BGE-small chunks"))
    T["sens"] = table(
        [(k, sn[k]) for k in ["default", "r16", "r64", "p30", "p120"]] + ["mid", ("bge_chunk", sn["bge_chunk"])],
        V["sens"], M3, ("BGE-small + \\spk{}，1,024 token & " if zh else "BGE-small + \\spk{}, 1,024 tokens & ") + mh,
        (r"\caption{\spk{}（BGE-small 排序器，1,024 token）对补全半径 $r$ 和优先列表 $P$ 长度的敏感性（宏平均 F1）。默认值为 $r=32$、$P$ 取 60 个片段；最后一行是同一排序器的切块打包。}"
         if zh else
         r"\caption{Sensitivity of \spk{} (BGE-small ranker, 1,024 tokens) to the padding radius $r$ and the length of the priority list $P$ (macro F1). The default is $r=32$ and 60 spans; the last row is chunk packing with the same ranker.}"),
        "tab:sens")
    # greedy vs sampling
    sp = (dict(g="贪心解码", s="采样（温度 0.7，top-$p$ 0.9，3 个种子）")
          if zh else dict(g="Greedy decoding", s="Sampling (temperature 0.7, top-$p$ 0.9, 3 seeds)"))
    T["samp"] = table(
        [("group", sp["g"]), ("greedy_chunk", nm["bge_chunk"]), ("greedy_dp", nm["bge_dp"]),
         ("group", sp["s"]), ("samp_chunk", nm["bge_chunk"]), ("samp_dp", nm["bge_dp"])],
        V["samp"], ["llama", "q14"], ("1,024 token & " if zh else "1,024 tokens & ") + "Llama-8B & Qwen-14B",
        (r"\caption{贪心解码与采样的对比（1,024 token，BGE-small 排序器，宏平均 F1）。采样行给出 3 个种子的均值 $\pm$ 标准差。}"
         if zh else
         r"\caption{Greedy decoding versus sampling (1,024 tokens, BGE-small ranker, macro F1). Sampling rows give the mean $\pm$ standard deviation over three seeds.}"),
        "tab:samp")
    return T


def put(t, tag, body, anchor):
    """Replace the marked block, or insert it before `anchor` the first time."""
    b, e = f"% R18-{tag}-BEGIN", f"% R18-{tag}-END"
    block = f"{b}\n{body}\n{e}\n"
    if b in t:
        i, j = t.index(b), t.index(e) + len(e) + 1
        return t[:i] + block + t[j:]
    assert t.count(anchor) == 1, (tag, anchor)
    return t.replace(anchor, block + anchor)


for fn, zh in [("main.tex", False), ("main_zh.tex", True)]:
    t = open(fn, encoding="utf-8").read()
    # grid: hybrid chunk row
    name = "混合排序切块" if zh else "Hybrid chunks"
    g = V["grid_hybrid_chunks"]
    row = name + " & " + " & ".join(f(x) for m in ["q7", "llama", "q14", "q32"] for x in g[m]) + r" \\"
    t, n = re.subn(r"^" + re.escape(name) + r" & .*\\\\$", lambda _: row, t, flags=re.M)
    assert n == 1, fn
    T = tables(zh)
    # robustness section before the further-baselines section
    sec_extra = "\\section{补充基线}\n\\label{app:extra}\n" if zh else "\\section{Additional Baselines}\n\\label{app:extra}\n"
    intro = (r"\section{稳健性检验}" "\n" r"\label{app:robust}" "\n"
             r"表~\ref{tab:sens} 改变打包器的两个设置，表~\ref{tab:samp} 把贪心解码换成采样，表~\ref{tab:b256} 在 Qwen2.5-14B 上把预算降到 256 token。"
             if zh else
             r"\section{Robustness Checks}" "\n" r"\label{app:robust}" "\n"
             r"Table~\ref{tab:sens} varies the two settings of the packer, Table~\ref{tab:samp} replaces greedy decoding with sampling, and Table~\ref{tab:b256} lowers the budget to 256 tokens on Qwen2.5-14B.")
    t = put(t, "ROBUST", intro + "\n\n" + T["sens"] + "\n\n" + T["samp"] + "\n\n" + T["b256"] + "\n", sec_extra)
    # further baselines table at the top of the baselines section
    lead = (r"表~\ref{tab:base18} 在当前环境中补充了四个基线。" if zh else
            r"Table~\ref{tab:base18} adds four further baselines in the current environment.")
    i = t.index(sec_extra) + len(sec_extra)
    if "% R18-BASE-BEGIN" not in t:
        t = t[:i] + "% R18-BASE-BEGIN\n% R18-BASE-END\n" + t[i:]
    t = put(t, "BASE", T["base"] + "\n\n" + lead + "\n", "% R18-BASE-BEGIN")  # replaced in place
    open(fn, "w", encoding="utf-8", newline="").write(t)
    print(fn, "ok")
