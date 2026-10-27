#!/usr/bin/env python3
"""Run the frozen Qwen runner with exact offload and full-KV MLP chunking."""
from __future__ import annotations

import argparse
import json
import sys
import types
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from torch import nn

_REPO_BOOTSTRAP = Path(__file__).resolve().parents[1]
if str(_REPO_BOOTSTRAP) not in sys.path:
    sys.path.insert(0, str(_REPO_BOOTSTRAP))

from experiments import run_qwen25_sliminfer_variants as runner
from tools.qwen_efficiency_direct_gpu_loader import load_model_and_tokenizer_direct
from tools.qwen_whole_model_offload_adapter import install_parameter_offload


ADAPTER_ID = "qwen-parameter-offload-fullkv-mlp-chunk-v1"
PROFILES = ("input_embedding_cpu", "input_embedding_lm_head_cpu")


def install_prefill_mlp_chunking(
    model: nn.Module,
    *,
    chunk_tokens: int,
) -> Dict[str, Any]:
    """Chunk token-independent Qwen MLP evaluation along the sequence axis."""
    if chunk_tokens <= 0:
        raise ValueError("chunk_tokens must be positive")
    patched = []
    for name, module in model.named_modules():
        if module.__class__.__name__ != "Qwen2MLP":
            continue
        required = ("gate_proj", "up_proj", "down_proj", "act_fn")
        if not all(hasattr(module, attr) for attr in required):
            raise TypeError(f"Unexpected Qwen2MLP layout at {name}")
        original_forward = module.forward

        def chunked_forward(
            self: nn.Module,
            hidden_states: torch.Tensor,
            *,
            _original=original_forward,
            _chunk_tokens=chunk_tokens,
        ) -> torch.Tensor:
            if hidden_states.ndim < 3 or hidden_states.shape[-2] <= _chunk_tokens:
                return _original(hidden_states)
            outputs = []
            for start in range(0, hidden_states.shape[-2], _chunk_tokens):
                length = min(_chunk_tokens, hidden_states.shape[-2] - start)
                outputs.append(_original(hidden_states.narrow(-2, start, length)))
            return torch.cat(outputs, dim=-2)

        module.forward = types.MethodType(chunked_forward, module)
        patched.append(name)
    if not patched:
        raise RuntimeError("No Qwen2MLP modules were found for full-KV chunking")
    return {
        "mlp_chunking_enabled": True,
        "mlp_chunk_tokens": int(chunk_tokens),
        "patched_mlp_count": len(patched),
        "patched_mlp_names": patched,
    }


def load_model_and_tokenizer_offloaded_chunked(
    args: argparse.Namespace,
    attn_impl: str = "flash_attention_2",
    *,
    profile: str,
    lm_head_chunk_rows: int,
    mlp_chunk_tokens: int,
    metadata_path: Optional[Path],
):
    model, tokenizer, cache_classes, actual_attn_impl = load_model_and_tokenizer_direct(
        args, attn_impl
    )
    metadata = install_parameter_offload(
        model,
        profile=profile,
        lm_head_chunk_rows=lm_head_chunk_rows,
    )
    metadata["adapter_id"] = ADAPTER_ID
    if args.method == "fullkv":
        metadata.update(
            install_prefill_mlp_chunking(
                model,
                chunk_tokens=mlp_chunk_tokens,
            )
        )
    else:
        metadata.update({
            "mlp_chunking_enabled": False,
            "mlp_chunk_tokens": int(mlp_chunk_tokens),
            "patched_mlp_count": 0,
            "patched_mlp_names": [],
        })
    metadata.update({
        "requested_attention_implementation": attn_impl,
        "actual_attention_implementation": actual_attn_impl,
        "runtime_method": args.method,
        "runtime_dtype": str(next(model.parameters()).dtype),
    })
    if metadata_path is not None:
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = metadata_path.with_suffix(metadata_path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(metadata_path)
    print(f"[offload-adapter] {json.dumps(metadata, sort_keys=True)}", flush=True)
    return model, tokenizer, cache_classes, actual_attn_impl


def main() -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--loader-mode", choices=("direct", "offload"), required=True)
    parser.add_argument("--offload-profile", choices=PROFILES, default="input_embedding_cpu")
    parser.add_argument("--offload-lm-head-chunk-rows", type=int, default=16384)
    parser.add_argument("--prefill-mlp-chunk-tokens", type=int, default=8192)
    parser.add_argument("--offload-loader-metadata-path", type=Path)
    adapter_args, runner_args = parser.parse_known_args()

    if adapter_args.loader_mode == "direct":
        runner.load_model_and_tokenizer = load_model_and_tokenizer_direct
    else:
        def loader(args: argparse.Namespace, attn_impl: str = "flash_attention_2"):
            return load_model_and_tokenizer_offloaded_chunked(
                args,
                attn_impl,
                profile=adapter_args.offload_profile,
                lm_head_chunk_rows=adapter_args.offload_lm_head_chunk_rows,
                mlp_chunk_tokens=adapter_args.prefill_mlp_chunk_tokens,
                metadata_path=adapter_args.offload_loader_metadata_path,
            )

        runner.load_model_and_tokenizer = loader

    sys.argv = [sys.argv[0], *runner_args]
    runner.main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
