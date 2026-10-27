#!/usr/bin/env python3
"""Launch the frozen Qwen runner with weights loaded directly onto one GPU."""
from __future__ import annotations

import sys
from pathlib import Path

_REPO_BOOTSTRAP = Path(__file__).resolve().parents[1]
if str(_REPO_BOOTSTRAP) not in sys.path:
    sys.path.insert(0, str(_REPO_BOOTSTRAP))

from experiments import run_qwen25_sliminfer_variants as runner


LOADER_ID = "direct-gpu-low-cpu-memory-v1"


def load_model_and_tokenizer_direct(args, attn_impl: str = "flash_attention_2"):
    tokenizer_path = args.tokenizer_path or args.model_path
    dtype = runner.dtype_from_name(args.dtype)
    device = runner.torch.device(
        args.device if runner.torch.cuda.is_available() or args.device == "cpu" else "cpu"
    )
    if device.type != "cuda":
        raise RuntimeError(f"{LOADER_ID} requires CUDA, got {device}")

    if args.method == "fullkv":
        model_cls = runner.AutoModelForCausalLM
        cache_classes = (None, None)
    else:
        from models.modeling_qwen2 import Qwen2ForCausalLM
        from utils.cache_utils import HsCache, SlimInferCache

        model_cls = Qwen2ForCausalLM
        cache_classes = (SlimInferCache, HsCache)

    load_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": dtype,
        "attn_implementation": attn_impl,
        "low_cpu_mem_usage": True,
        "device_map": {"": str(device)},
    }
    try:
        model = model_cls.from_pretrained(args.model_path, **load_kwargs)
        actual_attn_impl = attn_impl
    except Exception as exc:
        if attn_impl == "sdpa":
            raise
        print(f"Warning: failed to load with {attn_impl}: {exc}. Retrying with sdpa.")
        load_kwargs["attn_implementation"] = "sdpa"
        model = model_cls.from_pretrained(args.model_path, **load_kwargs)
        actual_attn_impl = "sdpa"

    model.eval()
    tokenizer = runner.AutoTokenizer.from_pretrained(
        tokenizer_path, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id
    return model, tokenizer, cache_classes, actual_attn_impl


def main() -> int:
    runner.load_model_and_tokenizer = load_model_and_tokenizer_direct
    runner.main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
