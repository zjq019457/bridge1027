# BridgePack rebuttal 数据（A100 统一环境，1024-token 主设定）

所有数字来自同一台 3×A100 服务器、同一套代码与 Python 环境（runtime/silm），LongBench 六任务 1,150 条，
贪心解码。显著性为配对 bootstrap：10,000 次重采样，seed 20260701，按任务分层，95% 百分位区间。

BridgePack 参照 = `runs/qwen/default`（Qwen 39.45）与 `runs/llama/default/qa`（Llama 36.18），
packing 变体 `same_spans_source_order`（portable repro 默认配置）。

## 1. 可放入 rebuttal 的结果（BridgePack − 基线）

| 基线 | 来源 | Qwen 基线分 | Qwen Δ [95% CI] | Llama 基线分 | Llama Δ [95% CI] |
|---|---|---|---|---|---|
| 头部截断 | — | 25.47 | +13.97 [11.53, 16.33] | 23.92 | +12.27 [9.89, 14.60] |
| 中间截断 | — | 24.69 | +14.76 [12.43, 17.17] | 23.77 | +12.42 [10.08, 14.82] |
| Selective Context | Li et al. 2023（官方包，GPT-2，phrase 级） | 20.15 | +19.30 [16.95, 21.72] | 19.13 | +17.06 [14.62, 19.48] |
| LLMLingua-2 | Pan et al. 2024（官方 PromptCompressor） | 26.82 | +12.62 [10.25, 15.03] | 26.31 | +9.88 [7.37, 12.36] |
| LongLLMLingua | Jiang et al. 2024 | 32.64 | +6.81 | 29.87 | +6.31 |
| BM25 Chunk Packing | — | 36.94 | +2.50 [0.82, 4.21], p=0.004 | 34.44 | +1.75 [0.11, 3.37], p=0.034 |
| TF-IDF Chunk Packing | — | 36.87 | +2.58 [0.82, 4.33], p=0.003 | — | — |

除注明外 p<0.001。所有压缩基线与 BridgePack 一样，严格卡在每个模型 tokenizer 的 1,024 token 以内
（"Question / Selected evidence" 前缀 + 压缩内容；溢出时在模型 token 上做前缀截断）。

## 2. 回应审稿人问题的事实数据

- **部署动机（Q2）**：Llama-3-8B 的完整 prompt 中 682/1150（59%）超过 8,192 token 原生窗口，中位数约 11k。
  BridgePack 只用 1,024 token（Llama 约 13.8% 的原始长度）。
- **regex token 定义（Q4）**：`bridgekv/evidence_text_packing.py` 中
  `TOKEN_RE = [A-Za-z0-9]+(?:['-][A-Za-z0-9]+)?|[^\w\s]`，只用于候选 span 划分；预算统计一律用模型 tokenizer。
- **显著性（主结果）**：见第 1 节，全部为任务分层配对 bootstrap。

## 3. 内部备注（不放进 rebuttal）

- 全上下文：Qwen 43.30（高于 BP 3.86），Llama 36.73（与 BP 无显著差异）。如被追问，只讲效率角度。
- BGE Chunk Packing：Qwen 39.65 / Llama 36.75，与 BP 无显著差异。
- Llama TF-IDF：34.94，BP +1.25 [−0.37, 2.81]，不显著。
- BP 参照是 `same_spans_source_order`，不是论文主表的 `bridge_neighbor32_path_order` /
  `lexical_bridgepack_path_order`；两者的 A100 重跑已在 `runs/a100/` 排队（尚未完成）。
- 论文中三个 BGE 增强变体（BGE Recall / Union / Reranked + BridgePack）在代码快照里没有实现。

## 4. 复现脚本（均在 `server/` 下）

| 脚本 | 作用 |
|---|---|
| `simple_pack.py` | 头部 / 中间截断、全上下文 |
| `sc_pack.py` | Selective Context |
| `lingua2_pack.py` / `lingua_pack.py` | LLMLingua-2 / LongLLMLingua |
| `bge_pack.py` | BGE chunk packing |
| `lex_pack.py` | BM25 / TF-IDF / 其他 `pack_evidence` 变体（严格预算重打包） |
| `gen_runner.py` | 生成 + LongBench 打分 |
| `sig_rv5.py` | 配对 bootstrap 显著性 |
| `queue/*.sh`、`a100_pack_one.sh` | 任务队列、重试与批量打包 |
| `runs/rv5/*_qa/` | 每个基线的 raw_predictions 与 macro_summary |
