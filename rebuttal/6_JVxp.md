**Published baselines.** At the same exact 1,024-token budget, wrapper and scorer (n=1,150, paired), our single default configuration (Lexical-BridgePack, source order) beats LongLLMLingua by +6.81/+6.31 (Qwen/Llama-3), LLMLingua-2 by +12.62/+9.88, Selective Context by +19.30/+17.06 and middle truncation by +14.76/+12.42 (all p<0.001).

**One winning configuration.** That same configuration also significantly beats BM25 on both models: +2.50 [0.82, 4.21] on Qwen, +1.75 [0.11, 3.37] on Llama-3. Dense variants are optional add-ons, not a different method.

**Q1.** Appendix H measures efficiency only. On all 1,150 Llama-3 instances, BridgePack is statistically indistinguishable from full-context inference (36.18 vs. 36.73) with about 1/11 of the prompt.

**Q2 (motivation).** 682/1,150 (59%) Llama-3 full prompts exceed its 8,192-token window, median about 11k. A 1,024-token budget fits short-window and on-device models, many-query batching, and leaves room for multi-turn history; per-query QA latency falls from 2.7 s to 0.3 s.

**Q3.** Yes. The default and the two-operating-point framing were fixed on source-dev before any test run.

**Q4.** A regex token matches `[A-Za-z0-9]+(?:['-][A-Za-z0-9]+)?|[^\w\s]`. It is used only to form candidate spans; every budget is counted with the downstream tokenizer. Constants were frozen on source-dev, and varying them changes macro F1 by at most 0.43 on Qwen. We will state the regex and remove the historical labels.
