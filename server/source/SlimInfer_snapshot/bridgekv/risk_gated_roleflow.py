"""Risk-gated RoleFlow-KV: enable hub_bridge only when safe; protect direct evidence on swap."""
from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

RoleProfile = Dict[str, Any]

_WRAPPER_MARKERS = (
    "answer:",
    "only give me the answer",
    "based on the above",
    "please answer",
    "assistant",
    "user:",
)

_STOP = frozenset(
    "a an the and or of in on at to for from with by is are was were be been being "
    "what when where who which how that this these those it its".split()
)


def risk_gate_mode(policy: str) -> str:
    policy = (policy or "").lower()
    if policy == "roleflow_risk_task_gate":
        return "task"
    if policy == "roleflow_risk_sample_gate":
        return "sample"
    if policy == "roleflow_risk_swap_guard":
        return "swap_guard"
    return ""


def hub_bridge_policy(policy: str) -> bool:
    p = (policy or "").lower()
    return p in {
        "roleflow_hub_bridge",
        "roleflow_risk_task_gate",
        "roleflow_risk_sample_gate",
        "roleflow_risk_swap_guard",
    }


def _task_name(cfg: Mapping[str, Any]) -> str:
    audit = cfg.get("audit_context") or {}
    return str(audit.get("task") or cfg.get("task_name") or "")


def _query_text(cfg: Mapping[str, Any]) -> str:
    audit = cfg.get("audit_context") or {}
    return str(audit.get("query_text") or cfg.get("query_text") or "")


def _block_texts(cfg: Mapping[str, Any]) -> Sequence[str]:
    texts = cfg.get("block_texts") or (cfg.get("audit_context") or {}).get("block_texts") or []
    return list(texts)


def _tokenize(text: str) -> Set[str]:
    words = re.findall(r"[a-z0-9']+", (text or "").lower())
    return {w for w in words if len(w) >= 3 and w not in _STOP}


def question_hop_complexity(query: str) -> float:
    q = (query or "").lower()
    score = 0.0
    if " and " in q or ";" in q:
        score += 1.0
    for marker in ("who", "what", "when", "where", "which", "how"):
        if marker in q:
            score += 0.5
    caps = re.findall(r"\b[A-Z][a-z]+", query or "")
    score += min(3.0, len(caps) * 0.25)
    return score


def is_wrapper_block(block_text: str) -> bool:
    low = (block_text or "").lower()
    return any(m in low for m in _WRAPPER_MARKERS)


def block_query_overlap(query: str, block_text: str) -> float:
    q_tok = _tokenize(query)
    if not q_tok:
        return 0.0
    b_tok = _tokenize(block_text)
    if not b_tok:
        return 0.0
    return len(q_tok & b_tok) / float(len(q_tok))


def silm_hub_ratio(silm_selected: Set[int], profiles: Mapping[int, RoleProfile]) -> float:
    if not silm_selected:
        return 0.0
    hubs = sum(1 for bid in silm_selected if int(profiles.get(int(bid), {}).get("role_generic_hub", 0)))
    return hubs / float(len(silm_selected))


def bridge_candidates_with_anchoring(
    original_block_ids: Sequence[int],
    silm_selected: Set[int],
    profiles: Mapping[int, RoleProfile],
    *,
    qcg_z_min: float = 0.0,
) -> int:
    count = 0
    for bid in original_block_ids:
        if int(bid) in silm_selected:
            continue
        prof = profiles.get(int(bid), {})
        if int(prof.get("role_bridge_relay", 0)) and float(prof.get("query_conditioned_gain_z", 0.0)) > qcg_z_min:
            count += 1
    return count


def is_direct_evidence_block(
    block_id: int,
    profiles: Mapping[int, RoleProfile],
    query: str,
    block_texts: Sequence[str],
    *,
    qcg_z_min: float = 0.0,
    hub_max: float = 0.35,
    overlap_min: float = 0.12,
) -> bool:
    prof = profiles.get(int(block_id), {})
    if int(prof.get("role_generic_hub", 0)):
        return False
    if float(prof.get("hub_score", 0.0)) > hub_max:
        return False
    if float(prof.get("query_conditioned_gain_z", 0.0)) < qcg_z_min:
        return False
    text = block_texts[int(block_id)] if 0 <= int(block_id) < len(block_texts) else ""
    if is_wrapper_block(text):
        return False
    return block_query_overlap(query, text) >= overlap_min


def protected_silm_blocks(
    silm_selected: Set[int],
    profiles: Mapping[int, RoleProfile],
    query: str,
    block_texts: Sequence[str],
) -> Set[int]:
    return {
        int(bid)
        for bid in silm_selected
        if is_direct_evidence_block(int(bid), profiles, query, block_texts)
    }


def unsafe_swap_ratio(
    removable: Sequence[int],
    profiles: Mapping[int, RoleProfile],
    query: str,
    block_texts: Sequence[str],
) -> float:
    if not removable:
        return 0.0
    unsafe = sum(
        1
        for bid in removable
        if is_direct_evidence_block(int(bid), profiles, query, block_texts)
    )
    return unsafe / float(len(removable))


def task_gate_enable(cfg: Mapping[str, Any]) -> Tuple[bool, Dict[str, Any]]:
    task = _task_name(cfg)
    enable = task == "musique"
    return enable, {"gate": "task", "task": task, "enable_roleflow": enable}


def sample_gate_enable(
    cfg: Mapping[str, Any],
    *,
    silm_selected: Set[int],
    original_block_ids: Sequence[int],
    profiles: Mapping[int, RoleProfile],
) -> Tuple[bool, Dict[str, Any]]:
    cache_key = "_risk_sample_gate_decision"
    if cache_key in cfg:
        cached = cfg[cache_key]
        if isinstance(cached, dict):
            return bool(cached.get("enable_roleflow", False)), dict(cached)
        return bool(cached), {"gate": "sample", "cached": True}

    query = _query_text(cfg)
    block_texts = _block_texts(cfg)
    hub_ratio = silm_hub_ratio(silm_selected, profiles)
    bridge_anchored = bridge_candidates_with_anchoring(original_block_ids, silm_selected, profiles)
    must_excl: Set[int] = set()
    if original_block_ids:
        must_excl = {int(original_block_ids[0]), int(original_block_ids[-1])}
    removable = [bid for bid in silm_selected if bid not in must_excl]
    unsafe_ratio = unsafe_swap_ratio(removable, profiles, query, block_texts)
    hop = question_hop_complexity(query)

    enable = (
        hub_ratio >= 0.06
        and bridge_anchored >= 1
        and unsafe_ratio < 0.34
        and hop >= 1.0
    )
    meta = {
        "gate": "sample",
        "enable_roleflow": enable,
        "hub_ratio": hub_ratio,
        "bridge_anchored_outside": bridge_anchored,
        "unsafe_swap_ratio": unsafe_ratio,
        "hop_complexity": hop,
    }
    if isinstance(cfg, dict):
        cfg[cache_key] = meta
    return enable, meta


def should_fallback_to_silm(
    policy: str,
    cfg: Mapping[str, Any],
    *,
    silm_selected: Set[int],
    original_block_ids: Sequence[int],
    profiles: Mapping[int, RoleProfile],
) -> Tuple[bool, Dict[str, Any]]:
    mode = risk_gate_mode(policy)
    if not mode or mode == "swap_guard":
        return False, {"gate": mode or "none", "fallback": False}
    if mode == "task":
        enable, meta = task_gate_enable(cfg)
        return not enable, meta
    if mode == "sample":
        enable, meta = sample_gate_enable(
            cfg, silm_selected=silm_selected, original_block_ids=original_block_ids, profiles=profiles
        )
        return not enable, meta
    return False, {"gate": "unknown", "fallback": False}


def filter_removable_for_swap_guard(
    removable: Sequence[int],
    profiles: Mapping[int, RoleProfile],
    cfg: Mapping[str, Any],
) -> Tuple[List[int], Dict[str, Any]]:
    query = _query_text(cfg)
    block_texts = _block_texts(cfg)
    protected = [
        int(bid)
        for bid in removable
        if is_direct_evidence_block(int(bid), profiles, query, block_texts)
    ]
    protected_set = set(protected)
    filtered = [int(bid) for bid in removable if int(bid) not in protected_set]
    return filtered, {
        "swap_guard": True,
        "protected_block_count": len(protected),
        "protected_blocks_sample": protected[:16],
        "removable_before": len(removable),
        "removable_after": len(filtered),
    }
