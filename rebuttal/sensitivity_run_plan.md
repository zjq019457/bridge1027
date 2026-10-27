# 系数敏感性实验运行方案（A100 / csuf）

只读核查结果加执行方案。执行前未改动服务器上任何文件。

## 1. 代码与入口

- 代码包：`/diskE/jiaqizhang/BRIDGEPACK_PORTABLE_REPRO_V1_20260928/unpacked/BRIDGEPACK_PORTABLE_REPRO_V1/`
- 主方法（论文 Lexical-BridgePack source order）的实际调用链：
  `scripts/portable_runner.py`（VARIANT = `same_spans_source_order`）
  → `tools/bridgepack_g1/run_g1.py::build_matched_variant_packs`
  → `bridgekv/bridgepack_cpu_extensions.py::freeze_lexical_bridgepack_selection`
  → `bridgekv/evidence_text_packing.py::_select_bridge_paths`（bridge 路径选择），再按原文顺序重排。
- 注意：`evidence_text_packing.py` 中 `VARIANT_SETTINGS` 里同名的 `lexical_bridgepack_source_order` **不是**论文主方法（它不走 bridge 选择），本实验不用它。
- 基准：Portable Golden V1，Qwen 宏 F1 39.3517，同机 token 级可复现（PORTABLE_SAME_HOST_TOKEN_EXACT_PASS）。

## 2. 系数位置（均为硬编码）

| 论文符号 | 默认 | 位置 |
|---|---|---|
| α（BM25 / TF-IDF 融合） | 0.5 / 0.5 | `evidence_text_packing.py` `lexical_scores()`，约 L458 |
| c_div（L_bridge 中 (1−J)） | 0.10 | `bridge_link_score()`，L238 |
| c_jac（L_bridge 中 −J） | 0.35 | `bridge_link_score()`，L237 |
| c_sup（S_path 中 S_final 权重） | 0.5 | `build_candidate_paths()`，L764 |
| λ（冗余惩罚） | 0.55 | `_select_bridge_paths()`，L794 |
| 邻域半径 | 32 | `bridgepack_cpu_extensions.py` L452、L456、L530、L553；`run_g1.py` L477 |
| 种子池 / 支撑池 / 路径数 | 32 / 48 / 24 | `build_candidate_paths()` 默认参数；`_select_bridge_paths()` 中 `range(24)` |

论文未写明、但会影响选择的常数（不扫描，只记录）：`query_link_score` 中覆盖率权重 1.5；重叠阈值 0.10（配对）和 0.30（已选）；达到 95% 预算时提前停止。

## 3. 重要发现：补充材料表 A10 的单项消融实现与论文描述不符

`score_component_control()`（`bridgepack_cpu_extensions.py` L647–L710）中：
- `bridge_score_zero`、`support_score_zero` 和 `redundancy_penalty_zero` 只改写了候选的 `final_score`、`base_score` 和 `density_score` 字段。
- 但 `_select_bridge_paths()` 实际用的 bridge 分数是用文本现算的 `bridge_link_score()`，冗余惩罚是写死的 `0.55 * redundancy`，两者都**没有被置零**。
- 结果是三个"去掉某项"的对照实际上只改变了支撑项 S_final 的组成，并没有去掉 bridge 项或冗余项。

影响如下：
- 表 A10 中 "bridge = 0"（Qwen +0.699，Llama −0.129）和 "redundancy = 0"（+0.671）这几行，测的不是论文所说的东西。iSRj 的 Q2 正是基于这几行。
- 这些对照不显著，可能正是因为它们没有真正移除这些项。
- 联合选择对独立选择（+3.445）和对 MMR（+2.961）这两个主要对照走的是 `selection_only_control()`，不受这个问题影响。

建议：rebuttal 不要依赖表 A10 的单项数字；终稿用本实验中真正的单项置零重新做（见第 4 节中 c_div、c_jac、λ、c_sup 取 0 的那些点）。

## 4. 扫描设计

每次只改一个参数，其余保持默认：

| 参数 | 取值 | 其中"真正的单项去除" |
|---|---|---|
| α | 0, 0.25, 0.5, 0.75, 1.0 | — |
| c_div | 0, 0.05, 0.10, 0.20, 0.30 | c_div=0 |
| c_jac | 0, 0.20, 0.35, 0.50, 0.70 | c_jac=0 |
| c_sup | 0, 0.25, 0.5, 0.75, 1.0 | c_sup=0（去掉支撑项） |
| λ | 0, 0.30, 0.55, 0.70, 0.90 | λ=0（去掉冗余项） |
| 邻域 | 0, 16, 32, 48, 64 | 邻域=0 |

另加：
- **bridge 整项置零**：L_bridge ≡ 0。
- **c_div × c_jac 3×3 网格**：{0, 0.10, 0.20} × {0, 0.35, 0.70}。

去掉与默认值重复的点后，共 6×4 + 1 + 4（网格中非重复的格）= **29 个配置**，加上 1 个默认配置。

## 5. 实现方式（最小改动，不碰原代码包）

1. 把代码包复制到 `/diskE/jiaqizhang/BRIDGEPACK_SENSITIVITY_V1_<日期>/`，原目录保持只读。
2. 在副本中新增 `bridgekv/sensitivity_params.py`（一个全局参数对象，默认值与原值完全相同），把第 2 节中的硬编码常数改为读取该对象，通过环境变量或 JSON 配置文件注入。
3. **一致性门槛**：用默认参数在 12 个样本上跑打包和生成，要求 packed IDs、input IDs、output IDs 与 Portable Golden V1 逐 token 一致（复用 `scripts/compare_with_golden.py`）。不一致就停止。
4. 新增 `scripts/run_sensitivity.py`：每个配置先在 CPU 上打包（多进程），再在 GPU 上生成；支持断点续跑；按第 6 节输出统计。
5. 统计复用 `tools/bridgepack_g1/stats.py` 的配对 bootstrap（10,000 次，seed 20260701，按任务分层），比较对象是"默认配置在本次运行中的结果"。

## 6. 记录内容

每个配置记录：
- 六个任务的 F1 和宏 F1
- 多跳任务的三层证据保留指标
- 实际打包长度（均值、P50、P95、最大值）
- 保留区域数
- 首段截断率，以及因"放不下就停止"规则被丢弃的已选区域数和比例
- 超预算数（必须为 0）

另外对 BM25、BGE 和 no-neighbor 补充同样的打包统计，用来兑现 rebuttal 中对 Y2Nd 和 f8xH 的承诺。

Provenance：每次运行保存 git diff（相对原包）、`pip freeze`、`nvidia-smi -q`（GPU UUID 和 driver）、完整命令、环境变量、模型和数据哈希。

## 7. 资源与时间

- 参考：之前的 A100 运行中，11 个变体 × 1,150 个样本耗时 3,009 秒，约 0.24 秒/条。
- Qwen：30 个配置 × 1,150 条 ≈ 3.5 万条，**约 2.5 小时**（单卡、独占时）。
- Llama-3：同样约 2.5 小时。但需先确认服务器上有 `NousResearch/Meta-Llama-3-8B-Instruct` 的正确 snapshot。`/diskD/jiaqizhang/Llama-3` 看起来可能是 base 模型，需要核对哈希。
- **当前 3 张卡都被另一位用户（tjl_0516）的任务占用，利用率 100%，每张卡约有 38 GB 空闲显存。** 7B FP16 模型约需 16 GB，放得下，但会和对方抢算力，速度可能减半，也会拖慢对方的任务。建议在空闲时段运行，或先和对方协调。
- 模型：之前运行使用的 `/dev/shm/...model_snapshot` 已经不存在，需要从 Qwen 模型资产重新放置，并核对 snapshot 哈希（`a09a3545…`）。

## 8. 执行顺序

1. 确认 Llama snapshot 和 Qwen 模型资产的位置，核对哈希。
2. 复制代码包，完成参数化，通过 12 样本的逐 token 一致性门槛。
3. Qwen：先跑默认配置加 6 个真正的单项去除点（约 35 分钟），先拿到最关键的消融结果。
4. Qwen：跑其余扫描点。
5. Llama：重复第 3 和第 4 步。
6. 汇总：生成 summary.csv、配对比较表、敏感性曲线图、LaTeX 表和 README。
