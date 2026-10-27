Thank you for the positive assessment.

**(a) Compression baseline.** We ran official LongLLMLingua (Llama-2-7B compressor) under the same exact 1,024 Qwen-token budget, wrapper and scorer (n=1,150): BridgePack wins by +6.81 [4.65, 9.04] on Qwen (all six tasks) and +6.31 [4.12, 8.59] on Llama-3.

**Q1 / (c) constants.** All constants were frozen on an independent source-dev split. New paired runs on all 1,150 examples varying the fusion, support and redundancy weights move macro F1 by at most 0.43 (Llama-3: 0.72), and no setting significantly beats the default. The bridge coefficients follow design intent (0.35 penalizes near-duplicate pairs, 0.10 rewards new terms); ±32 is the best repair radius among 0/16/32/64.

**Q2 (NarrativeQA).** Inputs are whole books/scripts and answers are free-form plot summaries, rarely a few extractable spans; F1 is low on this task even for full-context models, and full-context inference often fails to complete. This bounds all extractive packers alike.

**(b) Path vs. Source.** Identical selected and repaired spans; Path keeps selection order, Source document order. We will define this explicitly.

**W2.** BridgePack packs fewer tokens on average (920) than BM25/TF-IDF (986/985), so its gains do not come from longer prompts; no method exceeds 1,024. Common-success pairing avoids favoring full context.

**W3.** Higher complete-chain retention (0.408 vs. 0.388/0.338) co-occurs with higher F1 on all three multi-hop tasks (Table 2).
