# 新机器环境搭建（只含必需内容）

原服务器：3×A100 80GB，驱动 580.82.07（CUDA 13.0）。

## 1. 主环境（打包 / 生成 / 打分 / 显著性）
```bash
conda create -n bp python=3.13 -y && conda activate bp
pip install -r env/requirements_main_py313.txt
```
用于 `server/` 下全部脚本：`gen_runner.py`、`lex_pack.py`、`bge_pack.py`、`simple_pack.py`、`sig_rv5.py` 等。

## 2. 压缩基线环境（LLMLingua-2 / LongLLMLingua / Selective Context）
```bash
conda create -n bp_compress python=3.10 -y && conda activate bp_compress
pip install -r env/requirements_compress_py310.txt
```
用于 `lingua_pack.py`、`lingua2_pack.py`、`sc_pack.py`。

## 3. 模型（需自行从 Hugging Face 下载）
| 用途 | HF 名称 | 脚本里的本地路径 |
|---|---|---|
| 生成 | Qwen/Qwen2.5-7B-Instruct | `TOKS["qwen"]` |
| 生成 | meta-llama/Meta-Llama-3-8B-Instruct | `TOKS["llama"]` |
| BGE 基线 | BAAI/bge-small-en-v1.5 | `models_small_bge` |
| Selective Context | gpt2 | `models_small_gpt2` |
| LLMLingua-2 | microsoft/llmlingua-2-xlm-roberta-large-meetingbank | `models_llmlingua2`（目录名必须含 `xlm-roberta-large`） |

## 4. 数据
LongBench 六任务原始 jsonl 放到 `assets/longbench_g1_data/`；样本列表在 `server/golden/`。
新机器上把脚本顶部的 `W = ...` 改成新的工作目录。
