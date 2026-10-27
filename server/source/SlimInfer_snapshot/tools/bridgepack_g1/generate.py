#!/usr/bin/env python3
"""BridgePack-only greedy generation helpers (do not edit shared runners)."""
from __future__ import annotations

import os
from typing import Any, Tuple


def generate_greedy_historical(
    model: Any,
    tokenizer: Any,
    torch: Any,
    prompt: str,
    max_new_tokens: int,
) -> Tuple[str, int]:
    """Match historical BridgePack QA decoding: chat template + model.generate greedy.

    Intentionally avoids the shared runner's last-token-logits loop, which can
    diverge from historical predictions on some samples.
    """
    if hasattr(tokenizer, "apply_chat_template"):
        chat_prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        chat_prompt = prompt
    model_device = getattr(model, "device", None)
    if model_device is None:
        model_device = next(model.parameters()).device
    model_input = tokenizer(chat_prompt, truncation=False, return_tensors="pt").to(model_device)
    context_length = int(model_input.input_ids.shape[1])
    with torch.no_grad():
        output = model.generate(
            **model_input,
            max_new_tokens=max_new_tokens,
            num_beams=1,
            do_sample=False,
            use_cache=True,
            eos_token_id=tokenizer.eos_token_id,
        )[0]
    pred = tokenizer.decode(output[context_length:], skip_special_tokens=True)
    return pred, context_length


def load_qwen_fp16(model_name_or_path: str, torch: Any):
    """Load Qwen like historical runs; prefer flash_attn2 then sdpa."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Never inherit an empty CUDA_VISIBLE_DEVICES from CPU-only helpers.
    if os.environ.get("CUDA_VISIBLE_DEVICES", None) == "":
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for BridgePack QA but torch.cuda.is_available() is False")

    # Prefer FlashAttention2 only when the package imports cleanly.
    attn_impl = "sdpa"
    try:
        import flash_attn  # noqa: F401

        attn_impl = "flash_attention_2"
    except Exception:
        attn_impl = "sdpa"

    load_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": torch.float16,
        "device_map": {"": 0},
        "attn_implementation": attn_impl,
    }
    try:
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **load_kwargs)
    except Exception as exc:
        if attn_impl != "sdpa":
            print(f"flash_attention_2 failed ({exc}); using sdpa", flush=True)
            load_kwargs["attn_implementation"] = "sdpa"
            model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **load_kwargs)
        else:
            raise
    if getattr(model.config, "pad_token_id", None) is None and tokenizer.eos_token_id is not None:
        model.config.pad_token_id = tokenizer.eos_token_id
    model.eval()
    return model, tokenizer
