#!/usr/bin/env python3
"""Launch the frozen Qwen runner with an audited parameter-offload loader.

The adapter is intentionally separate from the Stage-1-frozen runtime.  It
supports a direct-GPU reference mode and two progressively stronger offload
profiles.  The input embedding profile performs only an exact table lookup on
CPU; the optional LM-head profile copies weight chunks back to CUDA for the
original GPU linear operation.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
from torch import nn

_REPO_BOOTSTRAP = Path(__file__).resolve().parents[1]
if str(_REPO_BOOTSTRAP) not in sys.path:
    sys.path.insert(0, str(_REPO_BOOTSTRAP))

from experiments import run_qwen25_sliminfer_variants as runner
from tools.qwen_efficiency_direct_gpu_loader import load_model_and_tokenizer_direct


ADAPTER_ID = "qwen-parameter-offload-v1"
PROFILES = ("input_embedding_cpu", "input_embedding_lm_head_cpu")


class CpuOffloadedEmbedding(nn.Module):
    """Keep embedding weights on CPU and return exact gathered values to CUDA."""

    def __init__(self, source: nn.Embedding):
        super().__init__()
        weight = source.weight.detach().to(device="cpu", copy=True).contiguous()
        self.weight = nn.Parameter(weight, requires_grad=False)
        self.num_embeddings = int(source.num_embeddings)
        self.embedding_dim = int(source.embedding_dim)
        self.padding_idx = source.padding_idx
        self.max_norm = source.max_norm
        self.norm_type = source.norm_type
        self.scale_grad_by_freq = source.scale_grad_by_freq
        self.sparse = source.sparse

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        target_device = input_ids.device
        cpu_ids = input_ids.to(device="cpu", non_blocking=False)
        values = F.embedding(
            cpu_ids,
            self.weight,
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )
        return values.to(device=target_device, non_blocking=False)


class CpuOffloadedChunkedLinear(nn.Module):
    """Store a bias-free output projection on CPU and evaluate CUDA row chunks."""

    def __init__(self, source: nn.Linear, chunk_rows: int):
        super().__init__()
        if source.bias is not None:
            raise ValueError("LM-head offload currently requires a bias-free linear layer")
        if chunk_rows <= 0:
            raise ValueError("chunk_rows must be positive")
        weight = source.weight.detach().to(device="cpu", copy=True).contiguous()
        self.weight = nn.Parameter(weight, requires_grad=False)
        self.in_features = int(source.in_features)
        self.out_features = int(source.out_features)
        self.chunk_rows = int(chunk_rows)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if hidden_states.device.type != "cuda":
            return F.linear(hidden_states, self.weight)
        chunks = []
        for start in range(0, self.out_features, self.chunk_rows):
            end = min(self.out_features, start + self.chunk_rows)
            weight = self.weight[start:end].to(
                device=hidden_states.device,
                dtype=hidden_states.dtype,
                non_blocking=False,
            )
            chunks.append(F.linear(hidden_states, weight))
            del weight
        return torch.cat(chunks, dim=-1)


def _tensor_bytes(parameter: torch.Tensor) -> int:
    return int(parameter.numel() * parameter.element_size())


def _cuda_memory() -> Dict[str, int]:
    if not torch.cuda.is_available():
        return {"allocated_bytes": 0, "reserved_bytes": 0}
    return {
        "allocated_bytes": int(torch.cuda.memory_allocated()),
        "reserved_bytes": int(torch.cuda.memory_reserved()),
    }


def install_parameter_offload(
    model: nn.Module,
    *,
    profile: str,
    lm_head_chunk_rows: int = 16384,
) -> Dict[str, Any]:
    if profile not in PROFILES:
        raise ValueError(f"Unsupported offload profile: {profile}")
    if not torch.cuda.is_available():
        raise RuntimeError(f"{ADAPTER_ID} requires CUDA")

    input_embedding = model.get_input_embeddings()
    output_embedding = model.get_output_embeddings()
    if not isinstance(input_embedding, nn.Embedding):
        raise TypeError(f"Expected nn.Embedding, got {type(input_embedding).__name__}")
    if output_embedding is None or not isinstance(output_embedding, nn.Linear):
        raise TypeError(f"Expected nn.Linear LM head, got {type(output_embedding).__name__}")
    tied = input_embedding.weight.data_ptr() == output_embedding.weight.data_ptr()
    if tied or bool(getattr(model.config, "tie_word_embeddings", False)):
        raise RuntimeError("Offload profiles require untied input and output embeddings")

    before = _cuda_memory()
    original_device = input_embedding.weight.device
    if original_device.type != "cuda":
        raise RuntimeError(f"Expected CUDA input embedding before offload, got {original_device}")

    # Module parameters are traversed before child parameters.  This zero-sized
    # anchor keeps PreTrainedModel.device on CUDA after embed_tokens moves to CPU.
    if "_offload_device_anchor" not in dict(model.named_parameters(recurse=False)):
        model.register_parameter(
            "_offload_device_anchor",
            nn.Parameter(
                torch.empty(
                    0,
                    device=original_device,
                    dtype=input_embedding.weight.dtype,
                ),
                requires_grad=False,
            ),
        )

    input_bytes = _tensor_bytes(input_embedding.weight)
    model.set_input_embeddings(CpuOffloadedEmbedding(input_embedding))
    del input_embedding

    lm_head_bytes = 0
    if profile == "input_embedding_lm_head_cpu":
        lm_head_bytes = _tensor_bytes(output_embedding.weight)
        model.set_output_embeddings(
            CpuOffloadedChunkedLinear(output_embedding, lm_head_chunk_rows)
        )
        del output_embedding

    torch.cuda.empty_cache()
    after = _cuda_memory()
    actual_input = model.get_input_embeddings()
    actual_output = model.get_output_embeddings()
    model_device = str(model.device)
    if model_device != str(original_device):
        raise RuntimeError(
            f"Model device changed after offload: expected {original_device}, got {model_device}"
        )
    if actual_input.weight.device.type != "cpu":
        raise RuntimeError("Input embedding was not placed on CPU")
    if profile == "input_embedding_lm_head_cpu" and actual_output.weight.device.type != "cpu":
        raise RuntimeError("LM head was not placed on CPU")

    return {
        "adapter_id": ADAPTER_ID,
        "profile": profile,
        "model_device": model_device,
        "input_embedding_device": str(actual_input.weight.device),
        "output_embedding_device": str(actual_output.weight.device),
        "input_embedding_bytes": input_bytes,
        "lm_head_bytes": lm_head_bytes,
        "offloaded_parameter_bytes": input_bytes + lm_head_bytes,
        "lm_head_chunk_rows": int(lm_head_chunk_rows),
        "cuda_memory_before": before,
        "cuda_memory_after": after,
    }


def load_model_and_tokenizer_offloaded(
    args: argparse.Namespace,
    attn_impl: str = "flash_attention_2",
    *,
    profile: str,
    lm_head_chunk_rows: int,
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
    metadata.update({
        "requested_attention_implementation": attn_impl,
        "actual_attention_implementation": actual_attn_impl,
        "runtime_method": args.method,
        "runtime_dtype": str(next(model.parameters()).dtype),
    })
    if metadata_path is not None:
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = metadata_path.with_suffix(metadata_path.suffix + ".tmp")
        temporary.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(metadata_path)
    print(f"[offload-adapter] {json.dumps(metadata, sort_keys=True)}", flush=True)
    return model, tokenizer, cache_classes, actual_attn_impl


def main() -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--loader-mode", choices=("direct", "offload"), required=True)
    parser.add_argument("--offload-profile", choices=PROFILES, default="input_embedding_cpu")
    parser.add_argument("--offload-lm-head-chunk-rows", type=int, default=16384)
    parser.add_argument("--offload-loader-metadata-path", type=Path)
    adapter_args, runner_args = parser.parse_known_args()

    if adapter_args.loader_mode == "direct":
        runner.load_model_and_tokenizer = load_model_and_tokenizer_direct
    else:
        def loader(args: argparse.Namespace, attn_impl: str = "flash_attention_2"):
            return load_model_and_tokenizer_offloaded(
                args,
                attn_impl,
                profile=adapter_args.offload_profile,
                lm_head_chunk_rows=adapter_args.offload_lm_head_chunk_rows,
                metadata_path=adapter_args.offload_loader_metadata_path,
            )

        runner.load_model_and_tokenizer = loader

    sys.argv = [sys.argv[0], *runner_args]
    runner.main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
