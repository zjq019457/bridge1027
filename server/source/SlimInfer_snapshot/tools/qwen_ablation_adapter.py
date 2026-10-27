#!/usr/bin/env python3
"""Run frozen Qwen code with an explicitly identified ablation-only adapter.

The main runtime files remain byte-identical to the Stage-1-frozen revision.
CC parameters are injected by replacing the imported selector function in this
process only. DRM ablations use the frozen runner's existing command-line
parameters and use this adapter solely for the equivalence-gated direct-GPU
model loader.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Set, Tuple

import torch

_REPO_BOOTSTRAP = Path(__file__).resolve().parents[1]
if str(_REPO_BOOTSTRAP) not in sys.path:
    sys.path.insert(0, str(_REPO_BOOTSTRAP))

from experiments import run_qwen25_sliminfer_variants as runner
from tools.qwen_efficiency_direct_gpu_loader import load_model_and_tokenizer_direct


ADAPTER_ID = "qwen-ablation-adapter-v1"


def _stable_event_seed(
    seed: int,
    original_block_ids: Sequence[int],
    top_k_indices: torch.Tensor,
) -> int:
    digest = hashlib.sha256()
    digest.update(str(int(seed)).encode("ascii"))
    digest.update(b"|")
    digest.update(",".join(str(int(value)) for value in original_block_ids).encode("ascii"))
    digest.update(b"|")
    digest.update(
        ",".join(str(int(value)) for value in top_k_indices.detach().cpu().tolist()).encode("ascii")
    )
    return int.from_bytes(digest.digest()[:8], "big")


def _cc_component_selector(config: Mapping[str, Any]):
    from bridgekv import structural_sliminfer as structural

    original_selector = structural.cluster_continuity_active_ids
    alpha = float(config.get("alpha", 0.25))
    min_cluster_len = int(config.get("min_cluster_len", 2))
    max_neighbors = int(config.get("max_neighbors", 2))
    replacement_cap = float(config.get("replacement_cap", 0.05))
    component = str(config.get("component", "full"))
    seed = int(config.get("seed", 0))

    def select(**kwargs: Any) -> Tuple[Set[int], Dict[str, Any]]:
        call = dict(kwargs)
        call.update({
            "alpha": alpha,
            "min_cluster_len": min_cluster_len,
            "max_neighbors": max_neighbors,
            "replacement_cap": replacement_cap,
        })
        if component == "full":
            final, meta = original_selector(**call)
            meta.update({"ablation_component": component, "ablation_adapter_id": ADAPTER_ID})
            return final, meta

        state = structural.cc_prepare_state(
            original_block_ids=call["original_block_ids"],
            block_scores=call["block_scores"],
            top_k_indices=call["top_k_indices"],
            must_keep_block_ids=call["must_keep_block_ids"],
            num_blocks_to_keep=call["num_blocks_to_keep"],
            alpha=alpha,
            min_cluster_len=min_cluster_len,
        )
        baseline = set(int(value) for value in state["baseline"])
        must_keep = set(int(value) for value in state["must_keep"])
        k = int(state["k"])
        if not state["ok"]:
            return baseline, {
                "mode": f"cc_ablation_{component}",
                "triggered": 0,
                "reason": "no_finite_scores",
                "ablation_component": component,
                "ablation_adapter_id": ADAPTER_ID,
            }

        original_ids = [int(value) for value in call["original_block_ids"]]
        scores = dict(state["scores"])
        clusters = list(state["clusters"])
        smoothed = list(state["smoothed"])
        proposed, cluster_blocks = structural._cc_neighbor_and_cluster_blocks(
            original_block_ids=original_ids,
            clusters=clusters,
            smoothed=smoothed,
            baseline=baseline,
            max_neighbors=max_neighbors,
        )

        if component == "neighbor_only":
            # Keep the real CC neighbor proposals but remove cluster protection
            # from the donor side of the fixed-K swap.
            protected_clusters: Set[int] = set()
            to_add = set(proposed)
        elif component == "cluster_only":
            # Recover missing members of real detected clusters, without the
            # outside-cluster neighbor expansion used by full CC.
            protected_clusters = set(cluster_blocks)
            to_add = set(cluster_blocks).difference(baseline)
        elif component == "random_swap_fixed_k":
            protected_clusters = set()
            candidates = [value for value in original_ids if value not in baseline]
            removable = [value for value in baseline if value not in must_keep]
            event_rng = random.Random(
                _stable_event_seed(seed, original_ids, call["top_k_indices"])
            )
            event_rng.shuffle(candidates)
            event_rng.shuffle(removable)
            max_swap = max(
                0,
                int(math.floor(replacement_cap * max(0, k - len(must_keep)))),
            )
            swaps = min(max_swap, len(candidates), len(removable))
            final = (baseline - set(removable[:swaps])) | set(candidates[:swaps])
            final = structural._trim_to_budget(
                final,
                original_block_ids=original_ids,
                block_scores=call["block_scores"],
                must_keep_block_ids=must_keep,
                num_blocks_to_keep=k,
            )
            meta = structural._cc_finalize_meta(
                mode="cc_random_swap_fixed_k",
                baseline=baseline,
                final=final,
                must_keep=must_keep,
                cluster_blocks=set(),
                to_add=set(candidates[:swaps]),
                original_block_ids=original_ids,
                k=k,
                swaps=swaps,
                max_swap=max_swap,
                alpha=alpha,
                min_cluster_len=min_cluster_len,
                max_neighbors=max_neighbors,
                replacement_cap=replacement_cap,
                clusters_found=len(clusters),
                extra={
                    "ablation_component": component,
                    "ablation_adapter_id": ADAPTER_ID,
                    "random_seed": seed,
                },
            )
            return final, meta
        else:
            raise ValueError(f"Unsupported CC ablation component: {component}")

        final, swaps = structural._cc_apply_neighbor_swaps(
            baseline=baseline,
            must_keep=must_keep,
            cluster_blocks=protected_clusters,
            to_add=to_add,
            scores=scores,
            k=k,
            replacement_cap=replacement_cap,
        )
        final = structural._trim_to_budget(
            final,
            original_block_ids=original_ids,
            block_scores=call["block_scores"],
            must_keep_block_ids=must_keep,
            num_blocks_to_keep=k,
        )
        max_swap = max(0, int(math.floor(replacement_cap * max(0, k - len(must_keep)))))
        meta = structural._cc_finalize_meta(
            mode=f"cc_ablation_{component}",
            baseline=baseline,
            final=final,
            must_keep=must_keep,
            cluster_blocks=protected_clusters,
            to_add=to_add,
            original_block_ids=original_ids,
            k=k,
            swaps=swaps,
            max_swap=max_swap,
            alpha=alpha,
            min_cluster_len=min_cluster_len,
            max_neighbors=max_neighbors,
            replacement_cap=replacement_cap,
            clusters_found=len(clusters),
            extra={"ablation_component": component, "ablation_adapter_id": ADAPTER_ID},
        )
        return final, meta

    return select


def _install(config: Mapping[str, Any]) -> None:
    family = str(config.get("family", ""))
    runner.load_model_and_tokenizer = load_model_and_tokenizer_direct
    if family == "cc":
        import utils.cache_utils as cache_utils

        cache_utils.cluster_continuity_active_ids = _cc_component_selector(config)
    elif family != "drm":
        raise ValueError(f"Unsupported ablation family: {family}")


def main() -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--ablation-config", type=Path, required=True)
    adapter_args, runner_args = parser.parse_known_args()
    config = json.loads(adapter_args.ablation_config.read_text(encoding="utf-8"))
    _install(config)
    sys.argv = [sys.argv[0], *runner_args]
    runner.main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
