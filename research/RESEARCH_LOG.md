# Research log: in-context second-hop retrieval (dev set only)

Dev set: LongBench-E HotpotQA (290) + 2WikiMQA (294) with every LongBench test question removed.
Budget: 1,024 Qwen2.5 tokens. Test set (LongBench) not used for any decision below.

## 1. Diagnosis (A100 test predictions, Qwen-1024, short-answer tasks)
- Answer coverage (gold answer verbatim in packed context) is the bottleneck: F1 is ~68-70 when the
  answer is covered vs ~22 when it is not.
- BridgePack (main) vs BGE chunks: BridgePack converts covered answers better (69.8 vs 67.7 F1) but
  misses the second hop more often (MuSiQue coverage 33.0 vs 40.0; 2Wiki 57.0 vs 62.5).
- Structural cause: the support pool only admits spans related to the question, so a second-hop
  fact that shares no words with the question cannot be selected.

## 2. Method: bridge-entity second hop (`packers.bridge_hop`)
1. Hop 1: rank spans by seed score (lexical / BGE / hybrid z-sum).
2. Bridge entities: named entities and years in the top-k1 hop-1 spans that the question does not
   mention, weighted by hop-1 rank.
3. Hop 2: BM25 over the bridge entities (+ beta * seed).
4. Interleave hop-1 and hop-2 rankings (hop2_share), repair to sentence boundaries within +-32
   tokens, merge overlapping ranges, admit under the exact tokenizer budget with backfill, emit in
   document order.

## 3. Coverage proxy (dev)
| config | 2Wiki | HotpotQA |
|---|---|---|
| BridgePack main (old) | 55.8 | 62.4 |
| BGE chunks | 60.5 | 60.0 |
| lexical, no hop-2, fixed repair (lex_rep) | 58.5 | 64.5 |
| lexical, no hop-2, sentence repair | 59.5 | 64.8 |
| + hop-2 (bridge_hop) | 60.9 | 66.9 |
| + sentence repair | 63.9 | 67.9 |
| + hop2_share 0.67 | 65.0 | 67.9 |
| BGE seeds, no hop-2, sentence repair | 66.0 | 65.9 |
| BGE seeds + hop-2 | 70.7 | 65.9 |
| hybrid seeds, no hop-2, sentence repair | 62.6 | 66.2 |
| hybrid seeds + hop-2 | 70.4 | 69.0 |

One-at-a-time grid around bridge_hop (k1 3/10, beta 0/0.6, max_ents 6/20, radius 16/24): all within
about 1.5 points; radius 16 raises coverage but not F1 (below).

## 4. F1 on dev (Colab T4, Qwen2.5-7B-Instruct-AWQ via vLLM, greedy, LongBench scorer)
| config | HotpotQA-E | 2Wiki-E | macro |
|---|---|---|---|
| BridgePack main (old) | 49.54 | 40.51 | 45.03 |
| lex_rep (no hop-2) | 52.79 | 37.94 | 45.37 |
| bridge_hop | 53.14 | 40.28 | 46.71 |
| bridge_hop + sentence repair | 53.11 | 41.55 | 47.33 |
| bridge_hop + sentence repair + hop2_share 0.67 | 54.16 | 41.95 | **48.06** |
| bridge_hop, radius 16 | 53.03 | 40.80 | 46.91 |

Round 2:
| config | HotpotQA-E | 2Wiki-E | macro |
|---|---|---|---|
| BGE chunks | 50.09 | 41.77 | 45.93 |
| lexical, no hop-2, sentence repair | 53.89 | 41.92 | 47.90 |
| BGE seeds + hop-2 + sentence repair | 52.25 | 45.94 | 49.10 |
| hybrid seeds + hop-2 + sentence repair | 55.32 | 45.17 | 50.25 |
| **hybrid seeds, no hop-2, sentence repair** | **58.26** | **44.12** | **51.19** |

Paired bootstrap (dev, task-stratified, 10k, seed 20260701):
- hybrid + sentence repair (no hop-2) - BGE chunks: +5.26 [2.34, 8.23], p=0.0004
- hybrid + sentence repair (no hop-2) - old BridgePack: +6.16 [3.27, 9.01], p<0.0001
- hybrid + hop-2 - BGE chunks: +4.31 [1.09, 7.52], p=0.008
- sentence repair - fixed repair (lexical, no hop-2): +2.54 [0.27, 4.89], p=0.027
- lexical + sentence repair - old BridgePack: +2.88 [0.27, 5.54], p=0.030
- hop-2 effect: lexical +0.15 [-2.44, 2.81]; hybrid -0.94 [-3.86, 1.91] (not significant)
- lexical + sentence repair - BGE chunks: +1.97 [-1.33, 5.23] (not significant)

Conclusion so far: the F1 gain comes from sentence-boundary repair with merged, backfilled exact
admission, and from hybrid lexical+dense seeds. The bridge-entity second hop raises answer coverage
but not F1 (it helps 2Wiki, hurts HotpotQA) and is dropped from the main method.

Round 3 (seed x repair, all with merged/backfilled exact admission and document-order emission,
no hop-2), dev macro F1:
| seeds | fixed +-32 repair | sentence-boundary repair |
|---|---|---|
| lexical | 45.37 | 47.90 |
| BGE | 48.88 | 50.12 |
| hybrid | 48.17 | **51.19** |
Reference: BGE chunks (repo baseline, fixed +-32, similarity-order emission) 45.93; old BridgePack 45.03.
Sentence repair helps with every seed (+2.5 / +1.2 / +3.0). BGE cache for the test set computed on Colab.

Round 4 (running): 512 / 2048 budgets for BGE chunks, hybrid + sentence repair, old BridgePack.

## 5. Frozen test run (LongBench, 1,150 q, A100 x2, fp16/bf16, vLLM 0.7.3) — 2026-11-06
Config frozen on dev before this run: hybrid seeds + sentence repair (lex_rep seed_mode=2 sent=1).
Macro F1 (6 tasks):
| config | Qwen 512 | Qwen 1024 | Qwen 2048 | Llama 512 | Llama 1024 | Llama 2048 |
|---|---|---|---|---|---|---|
| BridgePack main (old) | 33.88 | 39.11 | 40.68 | 34.05 | 37.79 | 39.60 |
| BGE chunks | 33.05 | 38.85 | 40.98 | 33.27 | 38.99 | 40.85 |
| ours, BGE seeds | 35.64 | 40.76 | 43.14 | 36.16 | 41.67 | 43.58 |
| ours, hybrid seeds (frozen) | 36.74 | 40.64 | 43.60 | 36.44 | 40.98 | 42.78 |
| LLMLingua-2 (1024) | | 24.60 | | | 28.43 | |
Ours vs BGE chunks (paired bootstrap): beats it in 12/12 settings, significant (p<0.05) in 11/12
(hybrid, Qwen-1024: +1.79, p=0.061). Ablations at 1024 (one component off): no single
component significant (|diff| <= 0.6). Packing latency (CPU, 300 q): bp_main 614 ms, BGE chunks
280 ms, ours 432 ms (+BGE encode); max tokens 1024 for all, mean 1013.8 (ours) vs 986.5 (BGE).
Full context (~30k tokens, middle-truncated): Qwen-7B 43.04, Llama-8B 43.13.
Qwen2.5-14B (same Qwen traces): BGE chunks 36.27/42.34/45.29, old BridgePack 36.73/41.69/44.23,
ours hybrid 40.44/44.32/47.50, ours BGE seeds 39.21/45.18/48.10, LLMLingua-2 27.10 (512/1024/2048).
Ours vs BGE chunks on 14B: significant at all 3 budgets for both seed types (p <= 0.028).
Totals: frozen hybrid config beats BGE chunks in 9/9 model x budget settings, significant in 8/9.
