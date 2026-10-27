Thank you for the careful review.

**W1 (weights).** All constants were frozen on an independent source-dev split before any LongBench test run. The 0.5/0.5 fusion is not searched: both scores are z-normalized, so equal weighting is the neutral default. New sensitivity runs (Qwen, all 1,150 examples, paired vs. the default) vary the fusion weight (0.25/0.75), support weight (0.25/1.0) and redundancy penalty (0.30/0.70): macro F1 moves by at most 0.43, with none beating the default; on Llama-3 changes stay within 0.72 and none is significant.

**W2 (greedy).** "Greedy" refers to two unrelated steps. Greedy *selection* builds the input; greedy *decoding* is only the standard LongBench QA protocol, shared by all methods so paired tests are deterministic. BridgePack runs before prefill and emits verbatim source text, so it is decoding-agnostic: under nucleus sampling (T=0.7, p=0.9, 3 seeds) it still beats BM25 by +2.78 [1.19, 4.40]. It does not yield fragmented text: neighbor repair restores cut boundaries and source order keeps chronology.

**W3.** 13.25%/13.80% are the ratios of our main 1,024-token setting; 3-7% are the extra 256/512 scaling points. Our target is the tight-budget regime where evidence must compete.

**W4.** The method is model-agnostic: only the admission tokenizer changes. Fewer input tokens cut cost and latency, and removing distractors helps: on BABILong (qa1–3, 16k/32k) BridgePack beats BM25 by +2.92 [1.17, 4.67] using 33% fewer tokens.
