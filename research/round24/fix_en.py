"""English paper: four-model numbers everywhere, packer-frozen framing with the reranker as the headline ranker,
EXIT/CPC/CompAct in Table 2, and the corrected ranker/packer comparisons."""
import shutil, sys
P = r"C:\Users\dell\Desktop\BridgePack_ACL\paper\main.tex"
t = open(P, encoding="utf-8").read()


def rep(a, b):
    global t
    n = t.count(a)
    if n != 1:
        sys.exit(f"rep: found {n}x: {a[:90]}")
    t = t.replace(a, b)


def span(start, end, new):
    global t
    if t.count(start) != 1:
        sys.exit(f"span start found {t.count(start)}x: {start[:90]}")
    i = t.index(start)
    j = t.index(end, i) + len(end)
    t = t[:i] + new + t[j:]


# ---- abstract
span(r"We freeze every design choice on a development split and run the LongBench test set",
     r"At the same budget it exceeds LLMLingua-2 by 15.3 F1.",
     r"We freeze every design choice of the packer on a development split and run the LongBench test set (1,150 single- and multi-document QA instances) once, with four instruction models from 7B to 32B parameters, five off-the-shelf rankers and three budgets (512, 1,024, 2,048 tokens). Holding the ranker fixed (BGE-small, BGE-large, a hybrid lexical--dense ranker or a cross-encoder reranker), \spk{} outperforms chunk packing in all 48 ranker--model--budget settings, by 1.9--3.7 F1 pooled over models (every 95\% interval excludes zero) and by 1.9--2.1 F1 even on top of the reranker, so better ranking and better packing add up. With the reranker, \spk{} is the best compressed method in 10 of the 12 model--budget settings; at 1,024 tokens it is above LlamaIndex's sentence-window and auto-merging retrievers, LongLLMLingua, LLMLingua-2 (by 17.8 F1) and five trained compressors (Provence, RECOMP, EXIT, CPC, CompAct) on every model tested. With one sixth of the input (2,048 tokens vs.\ 12.5K on average), it scores above the full context on all four models (by 1.9--2.7 F1), shrinking the per-request KV cache about sixfold (e.g., 2.4\,GB$\rightarrow$0.4\,GB for Qwen2.5-14B); on InfiniteBench, 1,024-token \spk{} contexts---1/30 of the 30K-token full context---retain 100--104\% of full-context performance.")

# ---- Figure 1 caption
rep(r"\caption{Macro F1 on LongBench QA (1,150 instances) under exact token budgets. \spk{} outperforms BGE chunk packing and lexical pair packing at every budget on all four models, and at 2,048 tokens, one sixth of the average 12.5K-token document, it reaches or exceeds the full context.}",
    r"\caption{Macro F1 on LongBench QA (1,150 instances) under exact token budgets, with two rankers. With either ranker, \spk{} outperforms chunk packing with the same ranking at every budget on all four models; with the reranker, at 2,048 tokens (one sixth of the average 12.5K-token document) it scores above the full context on all four models.}")

# ---- introduction
rep(r"Every design choice, including the ranker, was fixed on a development split built from LongBench-E with all LongBench test questions removed; the LongBench test set was then run once.",
    r"Every design choice of the packer was fixed on a development split built from LongBench-E with all LongBench test questions removed; the LongBench test set was then run once with each ranker.")
rep(r"We test three instruction models and three budgets and report paired bootstrap tests.",
    r"We test four instruction models from 7B to 32B parameters, five off-the-shelf rankers and three budgets, and report paired bootstrap tests.")
span(r"\item \textbf{A controlled evaluation.}",
     r"(Sections~\ref{sec:extra}, \ref{sec:infbench} and~\ref{sec:scale}).",
     r"\item \textbf{A controlled evaluation.} With the packer frozen on a development split, \spk{} outperforms chunk packing with the same ranker in all 48 ranker--model--budget settings (four rankers, four models from 7B to 32B parameters, three budgets), by 1.9--3.7 F1 pooled over models with every interval excluding zero. With an off-the-shelf reranker it is the best compressed method in 10 of the 12 model--budget settings, and at 1,024 tokens it is above LlamaIndex's sentence-window and auto-merging retrievers, LongLLMLingua, LLMLingua-2 and five trained compressors (Provence, RECOMP, EXIT, CPC, CompAct) on every model tested (Table~\ref{tab:main}). With one sixth of the tokens it scores above the full context on all four models, cutting the per-request KV cache about sixfold, and the gains carry over to all eleven query-bearing LongBench tasks and to book-length InfiniteBench contexts (Sections~\ref{sec:extra}, \ref{sec:infbench} and~\ref{sec:scale}).")

# ---- related work
rep(r"CPC scores sentences against the question with a trained encoder \citep{liskavets2025cpc}, and RECOMP",
    r"CPC scores sentences against the question with a trained encoder \citep{liskavets2025cpc}, EXIT classifies each retrieved sentence as useful or not with a fine-tuned LM \citep{hwang2025exit}, and RECOMP")

# ---- rankers
span(r"\spk{} accepts any function that scores spans against $q$.",
     r"The packer is the same in all cases.",
     r"\spk{} accepts any function that scores spans against $q$. We use five off-the-shelf rankers, none of which sees gold answers. The \emph{lexical} ranker fuses $z$-normalised BM25 ($k_1{=}1.2$, $b{=}0.75$) and TF-IDF scores computed over the spans of the instance. The \emph{dense} rankers are the cosine similarity between the question (with BGE's query instruction) and the span under \texttt{bge-small-en-v1.5} or \texttt{bge-large-en-v1.5} \citep{xiao2023cpack}. The \emph{hybrid} ranker adds the $z$-normalised lexical and BGE-small scores. The \emph{reranker} rescores the top 100 BGE-small spans with the cross-encoder \texttt{bge-reranker-v2-m3}. The top 60 spans form the priority list $P$, ordered by score with ties broken by position. The packer is the same for every ranker, and we report \spk{} with each of them.")

# ---- models
rep(r"Qwen2.5-7B-Instruct, Qwen2.5-14B-Instruct \citep{qwen2025qwen25} and Llama-3.1-8B-Instruct \citep{grattafiori2024llama3}, in bfloat16 with vLLM, LongBench's task prompts inside each model's chat template, greedy decoding and LongBench's per-task output lengths. The 14B model reads the packed contexts built with the Qwen tokenizer, which it shares with the 7B model.",
    r"Qwen2.5-7B-Instruct, Qwen2.5-14B-Instruct \citep{qwen2025qwen25} and Llama-3.1-8B-Instruct \citep{grattafiori2024llama3} in bfloat16, and Qwen2.5-32B-Instruct in its official AWQ 4-bit release (Section~\ref{sec:scale}), all with vLLM, LongBench's task prompts inside each model's chat template, greedy decoding and LongBench's per-task output lengths. The 14B and 32B models read the packed contexts built with the Qwen tokenizer, which they share with the 7B model.")

# ---- baselines
rep(r"Section~\ref{sec:external} adds stronger rankers (BGE-large; \texttt{bge-reranker-v2-m3} over the top 100 BGE spans) and published systems:",
    r"Section~\ref{sec:external} adds published systems and trained compressors, each with its released code or checkpoint and official settings:")
rep(r"which ranks all sentences of the context by their similarity to the question and adds them in that order while they fit, and LongLLMLingua",
    r"which ranks all sentences of the context by their similarity to the question and adds them in that order while they fit; EXIT \citep{hwang2025exit} (released Gemma-2B adapter, official prompt and 0.5 threshold) and CompAct \citep{yoon2024compact} (released CompAct-7b; five documents per iteration, at most six iterations; the final summary, 76 words on average, is passed to the reader), both applied, like Provence, to the top BGE spans; CPC \citep{liskavets2025cpc} (released Mistral-7B compressor), which scores every sentence of the context and keeps the highest-scoring ones in document order; and LongLLMLingua")

# ---- statistics
rep(r"Pooled results resample within the eighteen model--task strata.",
    r"Pooled results resample within the model--task strata (24 for four models, 18 for three).")

# ---- Table 2: EXIT, CPC, CompAct rows and caption
rep("RECOMP (extractive)$^\\dagger$ & 38.22 & 40.46 & 41.74 & 38.08 & 41.56 & 42.40 & \\textbf{42.88} & 45.50 & 47.30 & 42.63 & 45.93 & 47.39 \\\\\n",
    "RECOMP (extractive)$^\\dagger$ & 38.22 & 40.46 & 41.74 & 38.08 & 41.56 & 42.40 & \\textbf{42.88} & 45.50 & 47.30 & 42.63 & 45.93 & 47.39 \\\\\n"
    "EXIT$^\\dagger$ & -- & 40.84 & -- & -- & 42.24 & -- & -- & 44.48 & -- & -- & -- & -- \\\\\n"
    "CPC$^\\dagger$ & -- & 42.03 & -- & -- & 42.97 & -- & -- & 45.62 & -- & -- & -- & -- \\\\\n"
    "CompAct$^\\dagger$ & -- & 35.70 & -- & -- & 34.70 & -- & -- & 37.53 & -- & -- & -- & -- \\\\\n")
rep(r"The two indented rows apply LangChain's document transformers to the BGE-small chunks. Bold:",
    r"The two indented rows apply LangChain's document transformers to the BGE-small chunks. EXIT, CPC and CompAct were run at 1,024 tokens on the first three models (--: not run). Bold:")
rep(r"(a context pruner for Provence, an extractive sentence compressor for RECOMP); \spk{} needs no training.",
    r"(Provence: context pruner; RECOMP: extractive sentence compressor; EXIT: sentence classifier; CPC: sentence encoder; CompAct: abstractive compressor); \spk{} needs no training.")

# ---- 6.1
span(r"Table~\ref{tab:main} compares all methods at 1,024 tokens:",
     r"where every wasted token is most costly.",
     r"Table~\ref{tab:main} gives macro F1 for every method, model and budget, and Table~\ref{tab:sig} the paired tests pooled over the four models. With the ranking held fixed, \spk{} outperforms chunk packing in all 48 ranker--model--budget settings (four rankers, four models, three budgets); pooled over the models the gain is 1.85--3.66 F1, and every interval excludes zero. With the BGE-small ranker, the standard dense setting, the gain is +2.81, +2.83 and +2.66 F1 at 512, 1,024 and 2,048 tokens, and it persists with Qwen2.5-32B (+2.82, +3.88 and +2.93; last three columns). With BGE-large or the reranker, \spk{} is above every prompt compressor, published context-construction system and trained compressor at 1,024 tokens on every model; Section~\ref{sec:external} discusses these comparisons.")
span(r"\spk{} also outperforms the structured pair packer by 2.5--3.1 F1 at every budget, in all nine settings.",
     r"(Appendix~\ref{app:pertask}).",
     r"With the same BGE-small ranker, \spk{} also outperforms the structured pair packer in all twelve model--budget settings, by 2.30, 3.19 and 3.63 F1 pooled at the three budgets (Table~\ref{tab:sig}). The pair packer was designed for multi-hop questions and selects evidence in pairs, yet a packer that ranks spans one at a time but packs them carefully is better on the multi-hop tasks as well: at 1,024 tokens \spk{} with the BGE ranker reaches 46.4, 45.9 and 52.7 MH F1 on Qwen2.5-7B, Llama-3.1-8B and Qwen2.5-14B, against 42.4, 39.6 and 47.0 for pair packing and 42.7, 41.3 and 48.0 for BGE chunks (Appendix~\ref{app:pertask}).")

# ---- 6.2
span(r"Conversely, the choice of ranker inside \spk{} matters little.",
     r"transfers across lexical, dense and hybrid rankers.",
     r"Conversely, the packer helps whichever ranker it follows. Among the inexpensive rankers, the hybrid ranker selected on the development split and the BGE-small ranker are close: pooled over the four models they differ by at most 0.8 F1 at any budget (hybrid +0.79 at 512 tokens, BGE-small +0.72 and +0.20 at 1,024 and 2,048), and each is best in some settings. With a purely lexical ranker, \spk{} outperforms the pair packer, which uses the same lexical signals, by 1.54 F1 at 1,024 tokens (three models). Stronger rankers raise accuracy further, and the packing gain adds to theirs (Section~\ref{sec:external}). In short, the improvement comes from the packing operations and transfers across lexical, dense, hybrid and cross-encoder rankers.")

# ---- 6.3
rep(r"A better ranker therefore does not remove the packing waste: the gain from better packing is comparable to the gain from replacing BGE-small with BGE-large or with a reranker, and the two add up.",
    r"A better ranker therefore does not remove the packing waste. At 1,024 tokens, the gain from better packing with BGE-small (+2.83) is larger than the gain from replacing BGE-small with BGE-large in chunk packing (+0.45) and comparable to the gain from the reranker (+2.67), and the two add up.")
recomp = t[t.index(r"RECOMP \citep{xu2023recomp}, whose extractive compressor"):t.index(r"all four \spk{} variants are above it on every model.") + len(r"all four \spk{} variants are above it on every model.")]
span(r"We also compare with published context-construction systems under the same budget and wrapper.",
     r"\spk{} with the same BGE ranker remains 1.31 F1 higher.",
     r"We also compare with published context-construction systems and trained compressors under the same budget and wrapper, each with its released code or checkpoint. At 1,024 tokens, pooled over the four models, LlamaIndex's sentence-window retrieval \citep{liu2022llamaindex}, which retrieves single sentences and replaces each with its $\pm$3-sentence window, trails \spk{} with the reranker by 6.3 F1; its windows of neighbouring top sentences overlap heavily, which is exactly the duplication of Section~\ref{sec:diag}. LlamaIndex's auto-merging retriever, which replaces retrieved leaf chunks by their parent when most of the parent's children are retrieved, trails it by 4.6: merging at fixed hierarchy levels removes some duplication but admits whole parents that are mostly irrelevant. LongLLMLingua \citep{jiang2024longllmlingua} trails it by 5.2. Provence \citep{chirkova2025provence}, a trained pruner that removes irrelevant sentences from each retrieved chunk, trails it by 2.0. Provence is strongest at the tightest budget: at 512 tokens it is the best system on Qwen2.5-7B and second on Llama-3.1-8B and Qwen2.5-32B (Table~\ref{tab:main}). Its pruner, however, is a DeBERTa-v3-large model trained with supervision on about 370K queries (Section~\ref{sec:cost}), whereas \spk{} needs no training; given a reranker, \spk{} is ahead of it in 11 of the 12 model--budget settings. "
     + recomp +
     r" EXIT \citep{hwang2025exit}, CPC \citep{liskavets2025cpc} and CompAct \citep{yoon2024compact}, which we run at 1,024 tokens on Qwen2.5-7B, Llama-3.1-8B and Qwen2.5-14B, are also below \spk{} with the reranker on every model. Pooled over the three models, \spk{} leads EXIT by 1.93 F1 and CompAct by 8.47 (both intervals exclude zero) and CPC, the strongest of the three, by 0.91; CPC scores every sentence of the context with a fine-tuned 7B encoder, whereas \spk{} needs only the 568M-parameter reranker and 0.4\,s of CPU time. Finally, packing non-overlapping sentence-aligned chunks with BGE, which avoids duplication by construction, recovers part of the gain, as our diagnosis predicts; \spk{} with the same BGE ranker remains 1.7 F1 higher.")

# ---- 6.4 / 6.5 (eleven tasks, InfiniteBench)
rep(r"We run them with the frozen configuration and no change (Table~\ref{tab:extra}). Over all eleven query-bearing tasks, \spk{} outperforms",
    r"We run them with the packer unchanged, with the hybrid and BGE rankers (Table~\ref{tab:extra}). Over all eleven query-bearing tasks, \spk{} with the hybrid ranker outperforms")
rep(r"we run the frozen configuration on InfiniteBench's",
    r"we run \spk{} unchanged, with the hybrid and BGE rankers, on InfiniteBench's")
rep(r"The frozen \spk{} is the best method in three of the four model--budget settings",
    r"With the hybrid ranker, \spk{} is the best method in three of the four model--budget settings")

# ---- full context
span(r"At 2,048 tokens, \spk{} matches reading the full context on all three models:",
     r"recovers 93--95\% of full-context F1.",
     r"At 2,048 tokens, one sixth of the mean prompt (2,048 vs.\ 12.5K tokens), \spk{} with the reranker scores above the full context on all four models: by +1.92, +2.48, +1.95 and +2.68 F1 on Qwen2.5-7B, Llama-3.1-8B, Qwen2.5-14B and Qwen2.5-32B. With the hybrid ranker it matches the full context ($+0.56$, $-0.35$, $-0.10$ and $+1.36$), and on the three multi-hop tasks it is \emph{better} than the full context: +3.68 F1 pooled over the four models, and higher on every model (49.2 vs.\ 44.4, 45.8 vs.\ 43.7, 56.0 vs.\ 52.6 and 57.5 vs.\ 53.2 MH F1), consistent with the observation that long inputs dilute the evidence that matters \citep{liu2024lost}. At 1,024 tokens, one twelfth of the prompt, \spk{} with the reranker retains 98--102\% of full-context F1.")

# ---- scale
rep(r"\spk{} improves over BGE chunk packing by +3.36, +2.67 and +3.07 F1 at 512, 1,024 and 2,048 tokens (Table~\ref{tab:scale}).",
    r"\spk{} with the hybrid ranker improves over BGE chunk packing by +3.36, +2.67 and +3.07 F1 at 512, 1,024 and 2,048 tokens (Table~\ref{tab:scale}).")
rep(r"At 2,048 tokens, \spk{} with the 32B model scores 49.15 F1 against 47.79 for the 30K-token full context (+1.36), i.e.\ 103\% of full-context F1.",
    r"At 2,048 tokens, \spk{} with the 32B model scores 49.15 F1 with the hybrid ranker and 50.47 with the reranker, against 47.79 for the 30K-token full context (103\% and 106\% of full-context F1).")
rep(r"\caption{The gain of \spk{} (frozen hybrid configuration) over BGE chunk packing",
    r"\caption{The gain of \spk{} (hybrid ranker, selected on the development split) over BGE chunk packing")

# ---- token-level compression
rep(r"Under the same exact 1,024-token budget, LLMLingua-2 reaches 24.6--28.4 F1, and \spk{} exceeds it by 15.27 F1 pooled over models.",
    r"Under the same exact 1,024-token budget, LLMLingua-2 reaches 24.6--29.4 F1, and \spk{} exceeds it by 15.4--17.8 F1 pooled over the four models, depending on the ranker.")

# ---- cost
rep(r"and needs neither training nor GPU memory beyond the 33M-parameter encoder.",
    r"and needs neither training nor GPU memory beyond the 33M-parameter encoder. The reranker, which gives the highest accuracy, adds 0.16\,s and 1.3\,GB (0.71\,s per instance in total).")
rep(r"about 40 times the cost of \spk{} for 2.5 F1 less.",
    r"about 40 times the cost of \spk{} with BGE-small for 3.55 F1 less (1,024 tokens, four models).")
rep(r"(1,014 tokens on average vs.\ 987 for chunk packing).",
    r"(1,014 tokens on average vs.\ 985 for chunk packing; Table~\ref{tab:diag}).")

# ---- discussion
rep(r"is worth about 2 F1 on its own, as much as replacing BGE-small with BGE-large or adding a cross-encoder reranker.",
    r"is worth about 2 F1 on its own, more than replacing BGE-small with BGE-large and about as much as adding a cross-encoder reranker (Section~\ref{sec:external}).")
span(r"Its savings translate directly into input tokens.",
     r"reduces prefill cost and the per-request KV cache by about the same factor;",
     r"Its savings translate directly into input tokens. With the same BGE-small ranker, a 1,024-token \spk{} context scores 43.47 F1 pooled over the four models, slightly above chunk packing with 2,048 tokens (43.30): \spk{} needs about half the input for the same accuracy. Against the full context, a 2,048-token \spk{} context with the reranker scores higher on all four models with one sixth of the input, which reduces prefill cost and the per-request KV cache by about the same factor;")

# ---- conclusion and limitations
span(r"Standard packing spends 15--30\% of the budget on duplicated text;",
     r"retains 99--101\% of full-context F1 on three LLMs.",
     r"Standard packing spends 15--30\% of the budget on duplicated text; \spk{}, whose core is merging overlapping evidence so that no text is paid for twice, improves over chunk packing with every ranker we test, at every budget and on four LLMs from 7B to 32B parameters, by 1.9--3.7 F1 pooled over models. With an off-the-shelf reranker it is the best compressed method in 10 of the 12 model--budget settings, above five trained compressors, and with one sixth of the tokens it scores above the full context on all four models.")
rep(r"with three instruction models of 7--14B parameters and budgets of 512--2,048 tokens; other languages, generation tasks such as summarisation, and much larger models may behave differently. The development split covers two multi-hop tasks, and the method was frozen before testing; a broader development set might lead to a different choice of ranker, although our results suggest the ranker matters less than the packer. The dense and hybrid rankers require a small encoder at inference time.",
    r"with four instruction models of 7--32B parameters and budgets of 512--2,048 tokens; other languages, generation tasks such as summarisation, and larger models may behave differently. The development split covers two multi-hop tasks; the packer was frozen on it before testing, and the rankers are used off the shelf. The dense rankers and the reranker require a small encoder at inference time (33M--568M parameters).")

# ---- appendix
rep(r"These results determined the frozen configuration; the hybrid ranker with sentence repair was the best development configuration, +5.26 F1 above BGE chunk packing.",
    r"These results fixed the packer (sentence-aligned padding, merging and backfilling); among the three rankers tried on the development split, the hybrid ranker was best, +5.26 F1 above BGE chunk packing. BGE-large and the reranker were added at test time with the packer unchanged.")
rep(r"removes one operation at a time from the frozen hybrid configuration.",
    r"removes one operation at a time from \spk{} with the hybrid ranker.")

shutil.copy(P, P + ".bak_r24")
open(P, "w", encoding="utf-8", newline="").write(t)
print("EN ok")

# ---- bibliography: EXIT
B = r"C:\Users\dell\Desktop\BridgePack_ACL\paper\references.bib"
b = open(B, encoding="utf-8").read()
if "hwang2025exit" not in b:
    b = b.rstrip("\n") + "\n\n@inproceedings{hwang2025exit,\n  title={{EXIT}: Context-Aware Extractive Compression for Enhancing Retrieval-Augmented Generation},\n  author={Hwang, Taeho and Cho, Sukmin and Jeong, Soyeong and Song, Hoyun and Han, SeungYoon and Park, Jong C.},\n  booktitle={Findings of the Association for Computational Linguistics: ACL 2025},\n  year={2025}\n}\n"
    shutil.copy(B, B + ".bak_r24")
    open(B, "w", encoding="utf-8", newline="").write(b)
    print("bib ok")
