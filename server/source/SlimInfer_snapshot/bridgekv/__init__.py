try:
    _TORCH_EXPORTS_AVAILABLE = True
    from .block_graph import build_bridge_graph
    from .block_layout import BlockLayout, build_blocks, validate_block_layout
    from .bridge_score import compute_bridge_scores
    from .rescuekv_scoring import (
        RescuePolicyResult,
        active_coverage_scores,
        apply_gumbel_noise,
        apply_rescue_policy,
        block_obd_saliency,
        combined_abc_score,
        delta_o_block_score,
        diffusion_gap_score,
        diffusion_gap_values,
        drm_s0_score,
        mmr_select_rescue_ids,
        normalize_scores,
        path_aware_score,
        pool_mmr_select_rescue_ids,
        query_drm_s0_score,
        query_evidence_score,
        redundancy_aware_score,
    )
    from .rescue_candidate_builder import (
        RescueCandidate,
        RescueCandidateSet,
        build_rescue_candidates,
    )
    from .selection_policy import compute_bridge_utilities
    from .structural_risk import compute_structural_scores
except ModuleNotFoundError as exc:
    if exc.name != "torch":
        raise
    _TORCH_EXPORTS_AVAILABLE = False

_state_cache = {}


def register_state(state: dict) -> str:
    import uuid
    state_id = str(uuid.uuid4())
    _state_cache[state_id] = state
    return state_id


def get_state(state_id: str):
    return _state_cache.get(state_id)

_OPTIONAL_TORCH_EXPORTS = [
    "build_bridge_graph",
    "BlockLayout",
    "build_blocks",
    "validate_block_layout",
    "compute_bridge_scores",
    "compute_bridge_utilities",
    "compute_structural_scores",
    "RescuePolicyResult",
    "active_coverage_scores",
    "apply_gumbel_noise",
    "apply_rescue_policy",
    "block_obd_saliency",
    "combined_abc_score",
    "delta_o_block_score",
    "diffusion_gap_score",
    "diffusion_gap_values",
    "drm_s0_score",
    "mmr_select_rescue_ids",
    "normalize_scores",
    "path_aware_score",
    "pool_mmr_select_rescue_ids",
    "query_drm_s0_score",
    "query_evidence_score",
    "redundancy_aware_score",
    "RescueCandidate",
    "RescueCandidateSet",
    "build_rescue_candidates",
]

__all__ = ["register_state", "get_state"]
if _TORCH_EXPORTS_AVAILABLE:
    __all__ = _OPTIONAL_TORCH_EXPORTS + __all__
