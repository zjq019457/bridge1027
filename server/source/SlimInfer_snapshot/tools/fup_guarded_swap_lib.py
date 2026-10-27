#!/usr/bin/env python3
"""Core harness for the FUP / Guarded-Swap 64-sample diagnostic.

Loads the model once and exposes per-sample evaluation primitives:
  - build_sample_ctx: per-sample prompt/tokens/candidate-pool setup
  - eval_pageset:     run the model with an injected rescue/swap config and
                      return either a full generation (for LongBench metric)
                      or the first-token gold-answer NLL (cheap label metric)

Active = fixed SlimInfer top-K is kept FIXED in every path; only the rescue
selection / utility / swap is injected. See report_stage2_64.md.
"""
from __future__ import annotations

import gc
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "experiments"))

import run_qwen25_sliminfer_variants as R  # noqa: E402
from bridgekv import build_rescue_candidates  # noqa: E402
from bridgekv.block_layout import build_blocks  # noqa: E402

PAGE_TOKENS = 64
LONG_CONTEXT_RELEASE = 3600  # tokens; release CUDA between calls to avoid OOM on tail samples


def release_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.synchronize()
        except Exception:
            pass


def is_oom_error(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return isinstance(exc, torch.cuda.OutOfMemoryError) or "out of memory" in msg


def load_json(path: Path) -> Any:
    import json
    return json.loads(Path(path).read_text(encoding="utf-8"))


def build_args(model_path: str, longbench_dir: str, out_dir: str, max_new_tokens: int = 32):
    """Construct an args namespace via the runner's own parser (guarantees all fields)."""
    argv = [
        "--model_path", model_path, "--tokenizer_path", model_path,
        "--model_family", "qwen2", "--dtype", "fp16", "--device", "cuda",
        "--backend", "flash_attention_2",
        "--method", "sliminfer_variant", "--scoring", "delta_o", "--decode_policy", "reuse_prefill",
        "--eval", "longbench", "--longbench_dir", longbench_dir,
        "--longbench_tasks", "hotpotqa", "--max_samples_per_task", "1",
        "--generation_max_new_tokens", str(max_new_tokens),
        "--output_dir", out_dir, "--isolated_run_id", "fup_harness",
        "--block_strategy", "fixed_active_rescue", "--rescue_candidate_strategy", "unified",
        "--target_block_tokens", "64", "--min_block_tokens", "32", "--max_block_tokens", "128",
        "--block_overlap_tokens", "0", "--block_size", "64",
        "--coarse_block_tokens", "256", "--fine_block_tokens", "64", "--medium_block_tokens", "128",
        "--prune_layers", "9,18,26", "--retain_tokens", "12288,6144,4096", "--preserve_first_block",
        "--temperature", "0", "--top_p", "1.0",
        "--enable_cpu_offload", "--enable_async_kv", "--max_model_len", "32768",
        "--rescue_policy", "add", "--rescue_score", "drm", "--rescue_ratio", "0.05", "--seed", "0",
        "--alpha_diff", "0.5", "--lambda_red", "0.25", "--eta_rescue", "0.25", "--eta_active", "0.25",
        "--lambda_len", "0.1", "--beta_query", "0.2", "--beta_semantic", "0.1",
        "--rescue_overlap_window_tokens", "64", "--rescue_overlap_stride_tokens", "32",
        "--rescue_evidence_half_window_tokens", "64", "--rescue_evidence_top_ratio", "0.2",
        "--rescue_semantic_page_tolerance", "16",
        "--fd_alpha_o", "1.0", "--fd_beta_h", "0.0", "--fd_gamma_s", "0.0",
    ]
    old = sys.argv
    try:
        sys.argv = ["run_qwen25_sliminfer_variants.py"] + argv
        args = R.normalize_args(R.parse_args())
    finally:
        sys.argv = old
    return args


@dataclass
class SampleCtx:
    task: str
    sample_id: Any
    sample_index: int
    prompt: str
    input_ids: torch.Tensor          # [1, L]
    attention_mask: torch.Tensor
    context_length: int
    query_text: str
    block_texts: List[str]
    block_layout: Dict[str, Any]
    rescue_layout: Dict[str, Any]
    fixed_rescue_layout: Dict[str, Any]
    answers: List[Any]
    all_classes: List[Any]
    first_answer_token_ids: List[int]


class FupHarness:
    """Loads the model once; runs per-sample evaluations with injected configs."""

    # config keys that must be reset before every model call to avoid leakage
    RESET_KEYS = [
        "forced_rescue_pages", "page_utility_map", "ac_page_utility_map", "feature_dump_path",
        "ac_utility_dump_path", "ac_gate_type", "ac_cost_adjusted",
        "guarded_swap", "active_core_ratio", "max_swaps", "swap_tau", "swap_utility_source",
        "head_profile_dump_path", "rh_candidate_dump_path", "rh_head_profile",
        "rh_mode", "rh_page_utility_map", "rh_cost_adjusted",
        "rh_beta_retrieval", "rh_beta_query", "rh_beta_reasoning",
        "rh_lambda_head", "rh_lambda_page", "rh_alpha_drm",
    ]

    def __init__(self, model_path: str, longbench_dir: str, out_dir: str, max_new_tokens: int = 32):
        self.args = build_args(model_path, longbench_dir, out_dir, max_new_tokens)
        self.longbench_dir = Path(longbench_dir)
        self.model, self.tokenizer, self.cache_classes, _ = R.load_model_and_tokenizer(self.args, attn_impl=self.args.backend)
        # The run is greedy; clear ignored sampling defaults to avoid one warning per sample.
        generation_config = getattr(self.model, "generation_config", None)
        if generation_config is not None:
            generation_config.temperature = None
            generation_config.top_p = None
            generation_config.top_k = None
        num_layers = int(getattr(self.model.config, "num_hidden_layers", 32))
        self.sliminfer_config, self.derived, _ = R.build_sliminfer_config(self.args, Path(out_dir), num_layers)
        self.dataset2prompt = load_json(REPO_ROOT / "configs" / "dataset2prompt.json")
        self.dataset2maxlen = load_json(REPO_ROOT / "configs" / "dataset2maxlen.json")

    # ---- sample setup ----
    def load_samples(self, task: str, n: int) -> List[Dict[str, Any]]:
        samples = R.load_longbench_samples(self.longbench_dir, task)
        return samples[:n]

    def build_sample_ctx(self, task: str, sample: Dict[str, Any], sample_index: int) -> SampleCtx:
        args = self.args
        tok = self.tokenizer
        prompt_format = self.dataset2prompt[task]
        sample_id = sample.get("_id", sample.get("id", sample_index)) if isinstance(sample, dict) else sample_index
        prompt = prompt_format.format(**sample)
        tokenized = tok(prompt, truncation=False, return_tensors="pt").input_ids[0]
        if len(tokenized) > args.max_model_len:
            half = args.max_model_len // 2
            prompt = tok.decode(tokenized[:half], skip_special_tokens=True) + tok.decode(tokenized[-half:], skip_special_tokens=True)
        if task not in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]:
            prompt = R.build_chat(tok, prompt)
        model_input = tok(prompt, truncation=False, return_tensors="pt").to(self.model.device)
        context_length = int(model_input.input_ids.shape[1])
        query_text = R._sample_query_text(sample if isinstance(sample, dict) else {})
        block_texts = R.build_prompt_block_texts(tok, model_input.input_ids, args.block_size)
        block_layout = build_blocks(
            model_input.input_ids[0], tok, sample if isinstance(sample, dict) else {},
            prompt_text=prompt, task_name=task, block_strategy=args.block_strategy,
            target_block_tokens=args.target_block_tokens, min_block_tokens=args.min_block_tokens,
            max_block_tokens=args.max_block_tokens, query_text=query_text,
            config={"block_size": args.block_size, "fixed_page_tokens": args.block_size,
                    "semantic_parser": args.semantic_parser},
        ).to_dict()
        answers = sample.get("answers", []) if isinstance(sample, dict) else []
        answer_text = " ".join(str(a) for a in answers) if isinstance(answers, list) else str(answers)
        rescue_config = {
            "page_tokens": args.block_size, "prompt_text": prompt, "answer_text": answer_text,
            "overlap_window_tokens": 64, "overlap_stride_tokens": 32,
            "evidence_half_window_tokens": 64, "evidence_top_ratio": 0.2,
            "semantic_page_tolerance": 16, "coarse_block_tokens": 256,
            "fine_block_tokens": 64, "medium_block_tokens": 128,
        }
        rescue_set = build_rescue_candidates(
            model_input.input_ids[0], tok, sample if isinstance(sample, dict) else {},
            task_name=task, strategy="unified", query_text=query_text,
            config=rescue_config,
        ).to_dict()
        fixed_rescue_set = build_rescue_candidates(
            model_input.input_ids[0], tok, sample if isinstance(sample, dict) else {},
            task_name=task, strategy="fixed", query_text=query_text,
            config=rescue_config,
        ).to_dict()
        # first gold answer token ids (for first-token NLL label metric)
        first_ans = ""
        if isinstance(answers, list) and answers:
            first_ans = str(answers[0])
        elif answers:
            first_ans = str(answers)
        fa_ids = tok(first_ans, add_special_tokens=False).input_ids[:4] if first_ans else []
        return SampleCtx(
            task=task, sample_id=sample_id, sample_index=sample_index, prompt=prompt,
            input_ids=model_input.input_ids, attention_mask=model_input.attention_mask,
            context_length=context_length, query_text=query_text, block_texts=block_texts,
            block_layout=block_layout, rescue_layout=rescue_set, fixed_rescue_layout=fixed_rescue_set,
            answers=answers,
            all_classes=sample.get("all_classes", []) if isinstance(sample, dict) else [],
            first_answer_token_ids=[int(x) for x in fa_ids],
        )

    # ---- model call ----
    def _prepare_config(self, ctx: SampleCtx, overrides: Dict[str, Any]) -> Dict[str, Any]:
        cfg = self.sliminfer_config
        for k in self.RESET_KEYS:
            cfg.pop(k, None)
        cfg["audit_context"] = {
            "task": ctx.task, "sample_id": ctx.sample_id, "sample_index": ctx.sample_index,
            "context_length": ctx.context_length, "method": overrides.get("_method", "fup"),
            "query_text": ctx.query_text, "block_texts": ctx.block_texts,
            "prompt_text_preview": ctx.prompt[:256],
        }
        cfg["query_text"] = ctx.query_text
        cfg["block_texts"] = ctx.block_texts
        cfg["block_layout"] = ctx.block_layout
        candidate_strategy = overrides.get("rescue_candidate_strategy", "unified")
        cfg["rescue_candidate_layout"] = (
            ctx.fixed_rescue_layout if candidate_strategy in {"", "fixed", "fixed_pool"} else ctx.rescue_layout
        )
        # defaults for this experiment
        cfg["block_strategy"] = overrides.get("block_strategy", "fixed_active_rescue")
        cfg["rescue_policy"] = overrides.get("rescue_policy", "add")
        cfg["rescue_score"] = overrides.get("rescue_score", "drm")
        cfg["rescue_candidate_strategy"] = overrides.get("rescue_candidate_strategy", "unified")
        cfg["rescue_selector"] = overrides.get("rescue_selector", "drm_mmr")
        cfg["rescue_gate_mode"] = "none"
        cfg["block_budget_mode"] = "none"
        cfg["verifier_mode"] = "none"
        cfg["rescue_audit_path"] = overrides.get("rescue_audit_path")
        for k in self.RESET_KEYS:
            if k in overrides:
                cfg[k] = overrides[k]
        return cfg

    def _make_kwargs(self, ctx: SampleCtx, cfg: Dict[str, Any]) -> Dict[str, Any]:
        SlimInferCache, HsCache = self.cache_classes
        slim_cache = SlimInferCache(self.model.device)
        return {
            "past_key_values": slim_cache,
            "hs_cache": HsCache(),
            "sliminfer_config": cfg,
            "all_original_prompt_indices_abs": list(range(ctx.context_length)),
        }

    def _eval_pageset_once(self, ctx: SampleCtx, overrides: Dict[str, Any], mode: str,
                          gen_tokens: Optional[int]) -> Dict[str, Any]:
        cfg = self._prepare_config(ctx, overrides)
        kwargs = self._make_kwargs(ctx, cfg)
        R.reset_runtime_stats(cfg)
        try:
            if mode == "ftnll":
                generated = self.model.generate(
                    ctx.input_ids, attention_mask=ctx.attention_mask, max_new_tokens=1,
                    num_beams=1, do_sample=False, use_cache=True,
                    eos_token_id=self.tokenizer.eos_token_id,
                    return_dict_in_generate=True, output_scores=True, **kwargs,
                )
                scores = generated.scores[0][0].detach().float()
                logprobs = torch.log_softmax(scores, dim=-1)
                if ctx.first_answer_token_ids:
                    nll = -float(logprobs[ctx.first_answer_token_ids[0]].item())
                else:
                    nll = float("nan")
                return {"first_token_nll": nll}
            gt = int(gen_tokens or self.dataset2maxlen.get(ctx.task, 32))
            generated = self.model.generate(
                ctx.input_ids, attention_mask=ctx.attention_mask, max_new_tokens=gt,
                num_beams=1, do_sample=False, use_cache=True,
                eos_token_id=self.tokenizer.eos_token_id, **kwargs,
            )
            output = generated[0]
            pred = self.tokenizer.decode(output[ctx.context_length:], skip_special_tokens=True)
            if not str(pred).strip():
                raise RuntimeError("empty_generation_output")
            return {"pred": pred}
        finally:
            del kwargs
            release_cuda()

    @torch.no_grad()
    def eval_pageset(self, ctx: SampleCtx, overrides: Dict[str, Any], mode: str = "gen",
                     gen_tokens: Optional[int] = None) -> Dict[str, Any]:
        """mode='gen' -> full generation; mode='ftnll' -> first-token gold NLL. Retries on CUDA OOM."""
        if ctx.context_length >= LONG_CONTEXT_RELEASE:
            release_cuda()
        last_exc: Optional[BaseException] = None
        for attempt in range(3):
            try:
                return self._eval_pageset_once(ctx, overrides, mode, gen_tokens)
            except BaseException as exc:
                last_exc = exc
                if is_oom_error(exc) and attempt < 2:
                    release_cuda()
                    continue
                raise
        assert last_exc is not None
        raise last_exc

    @torch.no_grad()
    def eval_compact_context(
        self,
        ctx: SampleCtx,
        selected_abs_indices: Sequence[int],
        gen_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Generate from an order-preserving packed token context using ordinary FullKV."""
        indices = sorted({int(x) for x in selected_abs_indices if 0 <= int(x) < ctx.context_length})
        if not indices:
            raise ValueError("compact context has no selected tokens")
        index_t = torch.tensor(indices, dtype=torch.long, device=ctx.input_ids.device)
        compact_ids = torch.index_select(ctx.input_ids, dim=1, index=index_t)
        compact_mask = torch.ones_like(compact_ids)
        gt = int(gen_tokens or self.dataset2maxlen.get(ctx.task, 32))
        release_cuda()
        eos_token_id: Any = self.tokenizer.eos_token_id
        if ctx.task == "samsum":
            newline_ids = self.tokenizer.encode("\n", add_special_tokens=False)
            if newline_ids:
                eos_token_id = [self.tokenizer.eos_token_id, newline_ids[-1]]
        generated = self.model.generate(
            compact_ids,
            attention_mask=compact_mask,
            max_new_tokens=gt,
            num_beams=1,
            do_sample=False,
            use_cache=True,
            eos_token_id=eos_token_id,
        )
        pred = self.tokenizer.decode(generated[0][compact_ids.shape[1]:], skip_special_tokens=True)
        release_cuda()
        if not str(pred).strip():
            raise RuntimeError("empty_generation_output")
        return {"pred": pred, "compact_token_count": int(compact_ids.shape[1])}
