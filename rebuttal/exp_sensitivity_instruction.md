# 实验指令：BridgePack 系数敏感性分析（QA F1）

> 用法：在 BridgePack 代码仓库根目录打开 Claude Code（或交给合作者），把本文件全文作为任务说明。

## 0. 背景与目标

审稿人（vcH5、Y2Nd、f8xH）要求说明 BridgePack 手设系数的来源与敏感性。我们在 rebuttal 中承诺终稿加入**完整的系数敏感性分析**。已有的 source-dev 扫描只报告了证据指标，没有 QA F1，且从未扫描 bridge 内部系数 0.10 / 0.35。本实验补齐这两点。

目标：在**固定默认配置不变**的前提下，逐一扰动每个系数，报告 QA F1、证据保留指标和打包统计，量化默认值附近性能曲面的平坦程度。

**重要原则**
- 本实验是**分析**，不是调参：无论结果如何，论文默认配置（frozen SHA `15886731816aa2eb4e08afa121bd8ccc9631dd232474a82a8f20214cc2ce16bf`）保持不变。
- 所有结果（包括不利结果）全部落盘，不做挑选。
- 除被扫描的那一个系数外，其余一切与主实验完全一致。

## 1. 先做代码摸底（不改动任何东西）

1. 找到 Lexical-BridgePack（source order）的实现，定位以下常数在代码中的位置，并列出文件:行号：
   - `alpha`：词法融合权重，`S_lex = alpha*z(BM25) + (1-alpha)*z(TFIDF)`，默认 0.5
   - `c_div`：L_bridge 中 `(1 - J)` 项系数，默认 0.10
   - `c_jac`：L_bridge 中 `-J` 惩罚系数，默认 0.35
   - `c_sup`：S_path 中 `S_final(s_j)` 的权重，默认 0.5
   - `lambda_red`：冗余惩罚 `-lambda*J(T_i ∪ T_j, T_selected)`，默认 0.55
   - `neighbor`：邻域修复半径（regex token），默认 32
2. 确认主实验的入口脚本、配置文件格式、结果输出格式、F1 打分脚本、证据链指标脚本、精确预算 admission 代码。
3. 若常数是硬编码的，把它们改为**可配置参数，默认值与原值完全相同**。改完后用默认参数在 Qwen、1024 预算、每任务前 20 个样本上重跑**打包（CPU）**，**逐样本比对打包文本与原始主实验的打包输出完全一致**（字节级）。不一致则停止并报告。
4. 生成阶段不要求与历史结果逐字一致：历史 G1 运行环境的 provenance 不完整（`G1_HISTORICAL_RUNTIME_PROVENANCE_INCOMPLETE`），已知存在少量 generation divergence。因此**在当前环境中重新跑一次默认配置**，作为本实验所有配对比较的唯一基准；不要拿扫描结果与论文中的历史数字做配对比较。另外报告新默认运行与论文主表的宏 F1 差值和逐样本预测不一致的数量，仅作说明。

## 1b. Provenance 记录（每次运行开始时自动写入 `results/sensitivity/provenance/`）

- `git rev-parse HEAD`、`git status --porcelain` 和 `git diff`（运行前 worktree 必须干净；不干净则停止）
- 完整启动命令与全部参数、所有相关环境变量（至少 `CUDA_VISIBLE_DEVICES`、`PYTHONHASHSEED`、`CUBLAS_WORKSPACE_CONFIG`、`HF_*`、`TRANSFORMERS_*`）
- `nvidia-smi -q` 输出（GPU 型号、UUID、PCI ID、VBIOS、driver、CUDA 版本）、`uname -a`、`/etc/os-release`、glibc 版本
- `python -V`、`pip freeze`，以及运行时实际 import 的 torch / transformers / tokenizers / flash-attn 版本与 `.so` 路径和 SHA256
- 模型和 tokenizer 的 snapshot 路径与文件哈希
- 每个结果文件的 SHA256 清单

## 2. 扫描设计（one-at-a-time，围绕默认值）

每次只改一个系数，其余保持默认：

| 参数 | 默认 | 扫描值 |
|---|---|---|
| alpha | 0.5 | 0.0, 0.25, 0.5, 0.75, 1.0 |
| c_div | 0.10 | 0.0, 0.05, 0.10, 0.20, 0.30 |
| c_jac | 0.35 | 0.0, 0.20, 0.35, 0.50, 0.70 |
| c_sup | 0.5 | 0.0, 0.25, 0.5, 0.75, 1.0 |
| lambda_red | 0.55 | 0.0, 0.30, 0.55, 0.70, 0.90 |
| neighbor | 32 | 0, 16, 32, 48, 64 |

另加一个小型二维网格，专门针对 bridge 内部两个系数的交互：
`c_div ∈ {0.0, 0.10, 0.20} × c_jac ∈ {0.0, 0.35, 0.70}`（9 格，含默认格）。

默认配置只需跑一次，作为所有对比的基准。

## 3. 评估设置

- 模型：Qwen/Qwen2.5-7B-Instruct 与 NousResearch/Meta-Llama-3-8B-Instruct，使用 Table A1 中的同一 snapshot。
- 预算：B = 1,024（下游模型 tokenizer 精确计数）。
- 解码：greedy，batch size 1，FP16，prompt wrapper 与主实验相同。
- 数据（两套，分别报告，不合并）：
  1. **Source-dev**：HotpotQA、2WikiMQA、MuSiQue 原始 dev 集，每任务确定性抽样 200 例（固定 seed，记录样本 ID），用于说明"选值来自 dev"。同时报告 QA F1 与证据指标。
  2. **LongBench 六任务全量**（n=1,150/模型），用于终稿中的敏感性表。
- 所有配置共享同一套样本 ID、同一候选切分。

## 4. 每个配置需记录的指标

1. 每任务 F1、六任务宏 F1（先任务内平均，再等权），dev 集为三任务宏 F1。
2. 多跳任务的证据指标：answer-bearing、supporting-fact、complete-chain recall（沿用现有脚本与 60% 重叠规则）。
3. 打包统计：下游 tokenizer 下实际打包长度的 mean / P50 / P95 / max；保留区域数的均值；首区间被前缀截断的比例；因 stop-at-first-nonfit 被丢弃的已选区域数（均值、比例）。
   *这两项同时兑现 rebuttal 对 Y2Nd 和 f8xH 的承诺（打包长度分布、丢弃/截断率），请对默认配置及 BM25、BGE、no-neighbor 变体也输出同样统计。*
4. 超预算样本数（必须为 0，否则报错停止）。

## 5. 统计分析

- 每个扰动配置相对默认配置：实例级配对差值，10,000 次 bootstrap（seed 20260701，按 task 分层，任务等权），报告 Δmacro 与 95% CI。
- 每个参数的平坦度摘要：扫描范围内宏 F1 的 max−min；与默认相差在 ±1.0 F1 以内的扫描点比例。
- 不对扫描结果做"选最优"。若某个扰动显著优于默认，如实记录。

## 6. 产出

放在 `results/sensitivity/` 下：
- `raw/`：每个 (模型, 数据集, 参数, 取值) 的逐样本预测、F1 和打包统计（jsonl）。
- `summary.csv`：一行一个配置，包含所有指标。
- `paired_vs_default.csv`：Δ 与 CI。
- `fig_sensitivity.pdf`：每个参数一个子图，横轴取值，纵轴宏 F1，Qwen 与 Llama 两条线，默认值竖虚线，带 CI 阴影；二维网格单独一张热力图。
- `table_sensitivity.tex`：适合放进终稿附录的 LaTeX 表。
- `README.md`：运行命令、代码 commit、config SHA、样本 ID 文件、运行时长、任何异常。

## 7. 执行顺序与资源

1. 摸底，参数化，做字节级一致性检查（第 1 节）。
2. 先跑 dev 集（约 26 个配置 × 2 模型 × 600 例）。
3. 再跑 LongBench 全量（约 26 × 2 × 1,150 ≈ 6 万次推理，输入约 1k token，单张 3090 约 8–12 小时）。打包在 CPU 上，可与 GPU 推理流水并行；打包结果先缓存，避免重复计算。
4. 支持断点续跑：已完成的 (配置, 样本) 跳过。
5. 完成后生成汇总与图表，并在 README 中写一段客观的结果摘要（不带倾向性措辞）。

## 8. 禁止事项

- 不修改默认配置、不修改已有主实验结果文件。
- 不使用 gold answer 或答案元数据参与选择。
- 不删除或遗漏任何扫描点的结果。
- 不改变 prompt、解码参数、模型 snapshot 或打分脚本。
