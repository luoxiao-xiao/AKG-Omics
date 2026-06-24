import os
import os
from typing import Dict, List, Optional, Tuple

from .agent import run_selection_agent
from .task_schema import SelectionResult, SourceCandidate, TaskSpec


def _as_set(xs: List[str]) -> set:
    return set([str(x).strip().lower() for x in xs])


def _requirements(task: TaskSpec) -> Tuple[set, set, Dict[str, bool]]:
    t = task.normalized()
    src = _as_set(t.source_modalities)
    target = t.target_modality
    all_modalities = set(src) | {target}

    flags = {
        "gene_in_task": "gene" in all_modalities,
        "protein_in_task": "protein" in all_modalities,
        "metabolism_in_task": "metabolism" in all_modalities,
        "gene_or_protein_task": bool({"gene", "protein"} & all_modalities),
        "cross_modality_task": len(all_modalities) > 1,
    }

    req_entities = set()
    if flags["gene_in_task"]:
        req_entities.add("gene")
    if flags["protein_in_task"]:
        req_entities.add("protein")
    if flags["metabolism_in_task"]:
        req_entities.add("metabolism")

    req_relations = set([x for x in t.required_relations if x])
    if "protein" in src and target == "gene":
        req_relations.add("protein_gene_mapping")
    if "gene" in src and target == "protein":
        req_relations.add("gene_protein_mapping")
    if "metabolism" in src and target == "gene":
        req_relations.add("metabolism_gene_association")
    if "gene" in src and target == "metabolism":
        req_relations.add("gene_metabolism_association")
    if flags["gene_or_protein_task"] or flags["metabolism_in_task"]:
        req_relations.add("pathway_membership")
    return req_entities, req_relations, flags


def _match_conditions(conditions: List[str], flags: Dict[str, bool]) -> bool:
    if not conditions:
        return False
    for token in conditions:
        key = str(token).strip().lower()
        if flags.get(key, False):
            return True
    return False


def _coverage_for_source(source_id: str, data_profile: Optional[Dict]) -> Tuple[Optional[float], Dict]:
    if not isinstance(data_profile, dict):
        return None, {}
    coverage = data_profile.get("source_coverage", {})
    if not isinstance(coverage, dict):
        return None, {}
    detail = coverage.get(str(source_id).strip().lower(), {})
    if not isinstance(detail, dict) or not detail.get("coverage_available", False):
        return None, detail if isinstance(detail, dict) else {}
    try:
        score = float(detail.get("overall_coverage", 0.0))
    except Exception:
        return None, detail
    return max(0.0, min(1.0, score)), detail


def _compute_candidate(task: TaskSpec, source: Dict, data_profile: Optional[Dict] = None) -> SourceCandidate:
    req_entities, req_relations, flags = _requirements(task)
    entities = _as_set(source.get("entities", []))
    relations = _as_set(source.get("relations", []))
    quality = float(source.get("quality_weight", 0.5))
    available = bool(source.get("available", False))
    source_id = str(source["id"])

    ent_cov = 1.0 if not req_entities else len(entities & req_entities) / float(len(req_entities))
    rel_cov = 1.0 if not req_relations else len(relations & req_relations) / float(len(req_relations))
    data_cov, data_detail = _coverage_for_source(source_id, data_profile)

    mandatory = _match_conditions(source.get("mandatory_when", []), flags)
    recommended = _match_conditions(source.get("recommended_when", []), flags)
    if data_cov is None:
        score = 0.45 * ent_cov + 0.45 * rel_cov + 0.10 * quality
    else:
        score = 0.35 * ent_cov + 0.45 * rel_cov + 0.10 * quality + 0.10 * data_cov
    if recommended:
        score += 0.05
    if mandatory:
        score += 0.20
    if not available:
        score -= 1.0

    reasons = [
        f"entity_coverage={ent_cov:.2f}",
        f"relation_coverage={rel_cov:.2f}",
        f"quality={quality:.2f}",
    ]
    if data_cov is not None:
        reasons.append(f"data_coverage={data_cov:.2f}")
    elif data_detail:
        reasons.append("data_coverage=unavailable_or_no_overlap")
    if mandatory:
        reasons.append("mandatory_by_rule")
    if recommended:
        reasons.append("recommended_by_rule")

    return SourceCandidate(
        source_id=source_id,
        score=float(score),
        is_mandatory=mandatory,
        is_recommended=recommended,
        available=available,
        reasons=reasons,
        data_coverage_score=data_cov,
        data_coverage_detail=data_detail,
    )


def _dedupe_keep_order(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _source_relation_lookup(registry_sources: List[Dict]) -> Dict[str, set]:
    return {
        str(src.get("id", "")).strip().lower(): _as_set(src.get("relations", []))
        for src in registry_sources
    }


def _candidate_lookup(candidates: List[SourceCandidate]) -> Dict[str, SourceCandidate]:
    return {str(c.source_id).strip().lower(): c for c in candidates}


def _candidate_by_id(candidates: List[SourceCandidate], source_id: str) -> Optional[SourceCandidate]:
    sid = str(source_id).strip().lower()
    for c in candidates:
        if str(c.source_id).strip().lower() == sid:
            return c
    return None


def _covered_relations(selected_ids: List[str], relation_lookup: Dict[str, set]) -> set:
    out = set()
    for sid in selected_ids:
        out.update(relation_lookup.get(str(sid).strip().lower(), set()))
    return out


def _repair_required_relation_coverage(
    selected_ids: List[str],
    candidates: List[SourceCandidate],
    registry_sources: List[Dict],
    task: TaskSpec,
    max_sources: int,
) -> Tuple[List[str], List[str]]:
    """Ensure final selections do not sacrifice required relations for data coverage."""
    _, req_relations, _ = _requirements(task)
    if not req_relations:
        return selected_ids, []

    selected_ids = _dedupe_keep_order([str(x).strip().lower() for x in selected_ids if str(x).strip()])
    relation_lookup = _source_relation_lookup(registry_sources)
    cand_lookup = _candidate_lookup(candidates)
    available = [c for c in candidates if c.available]
    notes: List[str] = []

    def _rank_key(c: SourceCandidate) -> Tuple[bool, bool, float]:
        return (c.is_mandatory, c.is_recommended, float(c.score))

    def _remove_one_if_needed(protected: set) -> bool:
        if len(selected_ids) <= max_sources:
            return True
        covered = _covered_relations(selected_ids, relation_lookup)
        removable = []
        for sid in selected_ids:
            if sid in protected:
                continue
            c = cand_lookup.get(sid)
            if c is not None and c.is_mandatory:
                continue
            rels = relation_lookup.get(sid, set())
            unique = any(
                rel in req_relations
                and rel in rels
                and not any(other != sid and rel in relation_lookup.get(other, set()) for other in selected_ids)
                for rel in rels
            )
            contribution = len(rels & req_relations)
            score = float(c.score) if c is not None else 0.0
            removable.append((unique, contribution, score, sid))
        if not removable:
            return False
        removable.sort(key=lambda x: (x[0], x[1], x[2]))
        selected_ids.remove(removable[0][3])
        covered.clear()
        return True

    protected_added = set()
    for rel in sorted(req_relations - _covered_relations(selected_ids, relation_lookup)):
        providers = [
            c for c in available
            if rel in relation_lookup.get(str(c.source_id).strip().lower(), set())
        ]
        providers.sort(key=_rank_key, reverse=True)
        if not providers:
            notes.append(f"required_relation_uncovered_no_available_source:{rel}")
            continue
        chosen = str(providers[0].source_id).strip().lower()
        if chosen not in selected_ids:
            selected_ids.append(chosen)
            protected_added.add(chosen)
            notes.append(f"required_relation_repaired:{rel}:{chosen}")
        _remove_one_if_needed(protected_added)

    selected_ids = _dedupe_keep_order(selected_ids)
    while len(selected_ids) > max_sources and _remove_one_if_needed(protected_added):
        selected_ids = _dedupe_keep_order(selected_ids)
    return selected_ids[:max_sources], notes


def _baseline_anchor_sources(task: TaskSpec, candidates: List[SourceCandidate]) -> List[str]:
    available_ids = {
        str(c.source_id).strip().lower()
        for c in candidates
        if c.available
    }
    anchor = [sid for sid in ["hgnc", "uniprot", "reactome", "cellmarker"] if sid in available_ids]
    return _dedupe_keep_order(anchor)


def _constrain_agent_selection_to_task_template(
    task: TaskSpec,
    proposed_ids: List[str],
    candidates: List[SourceCandidate],
    max_sources: int,
) -> Tuple[List[str], List[str]]:
    """
    Keep agent decisions close to the stable rule baseline while still allowing
    task-specific improvements.
    """
    t = task.normalized()
    src = _as_set(t.source_modalities)
    target = str(t.target_modality).strip().lower()
    anchor = _baseline_anchor_sources(task, candidates)
    available_ids = {
        str(c.source_id).strip().lower()
        for c in candidates
        if c.available
    }
    proposed = [
        str(s).strip().lower()
        for s in (proposed_ids or [])
        if str(s).strip().lower() in available_ids
    ]
    notes: List[str] = ["baseline_anchor_constraint_applied"]
    selected = list(anchor)

    proteinatlas_cov = None
    proteinatlas_candidate = _candidate_by_id(candidates, "proteinatlas")
    if proteinatlas_candidate is not None:
        proteinatlas_cov = proteinatlas_candidate.data_coverage_score

    allow_proteinatlas = False
    allow_kegg = False
    replace_reactome_with_kegg = False

    if target == "protein":
        allow_proteinatlas = (
            "proteinatlas" in available_ids
            and (proteinatlas_cov is None or float(proteinatlas_cov) >= 0.50)
        )
        if "gene" in src and "kegg" in available_ids:
            allow_kegg = True
            replace_reactome_with_kegg = "kegg" in proposed
        notes.append("template_target_protein")
    elif target == "gene" and "protein" in src:
        allow_proteinatlas = (
            "proteinatlas" in available_ids
            and (proteinatlas_cov is None or float(proteinatlas_cov) >= 0.90)
        )
        notes.append("template_target_gene_with_protein_context")
    else:
        notes.append("template_conservative_gene_prediction")

    if allow_proteinatlas and "proteinatlas" in proposed and "proteinatlas" not in selected:
        selected.append("proteinatlas")
        notes.append("template_add_proteinatlas")

    if allow_kegg and "kegg" in proposed and "kegg" not in selected:
        if replace_reactome_with_kegg and "reactome" in selected:
            selected = [sid for sid in selected if sid != "reactome"]
            notes.append("template_replace_reactome_with_kegg")
        selected.append("kegg")
        notes.append("template_add_kegg")

    selected = _dedupe_keep_order(selected)
    if len(selected) > max_sources:
        selected = selected[:max_sources]
        notes.append(f"template_trim_to_budget:{max_sources}")
    return selected, notes


def _apply_performance_feedback_policy(
    proposed_ids: List[str],
    candidates: List[SourceCandidate],
    max_sources: int,
    performance_feedback: Optional[Dict],
) -> Tuple[List[str], List[str], Dict[str, float], Dict[str, str], Dict[str, str]]:
    if not isinstance(performance_feedback, dict):
        return proposed_ids, [], {}, {}, {}
    strategy = str(performance_feedback.get("refine_strategy", "")).strip().lower()
    if strategy not in {"exploit_success", "hybrid_refine", "hard_recover"}:
        return proposed_ids, [], {}, {}, {}

    available_ids = {
        str(c.source_id).strip().lower()
        for c in candidates
        if c.available
    }
    baseline = [
        str(s).strip().lower()
        for s in performance_feedback.get("baseline", {}).get("selected_sources", [])
        if str(s).strip().lower() in available_ids
    ]
    round1 = [
        str(s).strip().lower()
        for s in performance_feedback.get("round1_free_agent", {}).get("selected_sources", [])
        if str(s).strip().lower() in available_ids
    ]
    proposed = [
        str(s).strip().lower()
        for s in (proposed_ids or [])
        if str(s).strip().lower() in available_ids
    ]
    policy = performance_feedback.get("round2_policy", {})
    policy = policy if isinstance(policy, dict) else {}
    suggested_weights = policy.get("suggested_source_weights", {})
    suggested_weights = suggested_weights if isinstance(suggested_weights, dict) else {}
    added = [s for s in round1 if s not in baseline]
    notes = [f"metric_feedback_policy:{strategy}"]
    actions: Dict[str, str] = {}
    usage_modes: Dict[str, str] = {}

    def _weight(sid: str, default: float = 1.0) -> float:
        try:
            return float(suggested_weights.get(sid, default))
        except Exception:
            return float(default)

    weights: Dict[str, float] = {}
    if strategy == "exploit_success":
        selected = _dedupe_keep_order(round1 + proposed)
        notes.append("metric_feedback_preserve_successful_round1")
        for sid in selected:
            weights[sid] = _weight(sid, 1.10 if sid in added else 1.0)
            actions[sid] = "keep" if sid not in added else "keep_successful_round1_source"
            usage_modes[sid] = "full"
    elif strategy == "hard_recover":
        selected = list(baseline)
        max_new = int(policy.get("max_new_sources_when_recovering", 1) or 1)
        candidates_for_probe = [s for s in proposed + added if s not in selected]
        for sid in candidates_for_probe[:max(0, max_new)]:
            selected.append(sid)
            notes.append(f"metric_feedback_low_weight_probe:{sid}")
        for sid in selected:
            weights[sid] = _weight(sid, 0.60 if sid in added else 1.0)
            actions[sid] = "keep" if sid in baseline else "downweight"
            usage_modes[sid] = "full" if sid in baseline else "graph_only"
        notes.append("metric_feedback_anchor_to_rule")
    else:
        selected = list(baseline)
        for sid in proposed + added:
            if sid not in selected:
                selected.append(sid)
        for sid in selected:
            weights[sid] = _weight(sid, 0.80 if sid in added else 1.0)
            actions[sid] = "keep" if sid in baseline else "downweight"
            usage_modes[sid] = "full" if sid in baseline else "low_weight_full"
        notes.append("metric_feedback_hybrid_rule_plus_beneficial_round1")

    selected = _dedupe_keep_order(selected)
    if len(selected) > max_sources:
        protected = set(baseline if strategy != "exploit_success" else round1)
        kept = [s for s in selected if s in protected]
        extras = [s for s in selected if s not in protected]
        selected = _dedupe_keep_order(kept + extras)[:max_sources]
        notes.append(f"metric_feedback_trim_to_budget:{max_sources}")
    weights = {sid: float(weights.get(sid, 1.0)) for sid in selected}
    usage_hints = policy.get("source_usage_modes", {})
    usage_hints = usage_hints if isinstance(usage_hints, dict) else {}
    action_hints = policy.get("source_action_hints", {})
    action_hints = action_hints if isinstance(action_hints, dict) else {}
    for sid in selected:
        usage_modes[sid] = str(usage_hints.get(sid, usage_modes.get(sid, "full")))
        actions[sid] = str(action_hints.get(sid, actions.get(sid, "keep")))
    return selected, notes, weights, usage_modes, actions


def _apply_free_exploration_policy(
    task: TaskSpec,
    candidates: List[SourceCandidate],
    max_sources: int,
    data_profile: Optional[Dict] = None,
) -> Tuple[List[str], List[str], Dict[str, float], Dict[str, str], Dict[str, str]]:
    """
    Deterministic rescue path for first-round free exploration when the LLM
    endpoint is unavailable or returns invalid JSON. This deliberately remains
    task/data-aware instead of falling back to the fixed rule baseline.
    """
    t = task.normalized()
    src = _as_set(t.source_modalities)
    target = str(t.target_modality).strip().lower()
    available_ids = {
        str(c.source_id).strip().lower()
        for c in candidates
        if c.available
    }
    cand = _candidate_lookup(candidates)
    notes = ["free_exploration_policy_after_llm_failure"]
    usage_modes: Dict[str, str] = {}
    actions: Dict[str, str] = {}

    def _cov(sid: str) -> float:
        c = cand.get(sid)
        if c is None:
            return 0.0
        try:
            return float(c.data_coverage_score or 0.0)
        except Exception:
            return 0.0

    def _add(seq: List[str], sid: str):
        if sid in available_ids and sid not in seq:
            seq.append(sid)

    selected: List[str] = []
    for sid in ["hgnc", "uniprot"]:
        _add(selected, sid)

    weights: Dict[str, float] = {sid: 1.0 for sid in selected}
    for sid in selected:
        usage_modes[sid] = "full"
        actions[sid] = "keep"
    if target == "protein" and "gene" in src:
        for sid in ["kegg", "proteinatlas", "cellmarker"]:
            _add(selected, sid)
        weights.update({"kegg": 1.15, "proteinatlas": 1.20, "cellmarker": 0.90})
        usage_modes.update({"kegg": "full", "proteinatlas": "full", "cellmarker": "prior_only"})
        actions.update({"kegg": "add", "proteinatlas": "add", "cellmarker": "add"})
        notes.append("free_policy_gene_to_protein_prioritize_kegg_proteinatlas")
    elif target == "protein":
        for sid in ["kegg", "proteinatlas", "string", "cellmarker"]:
            _add(selected, sid)
        weights.update({"kegg": 1.05, "proteinatlas": 1.10, "string": 0.80, "cellmarker": 0.90})
        usage_modes.update({"kegg": "low_weight_full", "proteinatlas": "full", "string": "graph_only", "cellmarker": "prior_only"})
        actions.update({"kegg": "add", "proteinatlas": "add", "string": "graph_only", "cellmarker": "add"})
        notes.append("free_policy_he_to_protein_probe_string_with_kegg_proteinatlas")
    elif target == "gene" and "protein" in src:
        for sid in ["reactome", "cellmarker", "kegg"]:
            _add(selected, sid)
        if "proteinatlas" in available_ids and _cov("proteinatlas") >= 0.80:
            _add(selected, "proteinatlas")
        weights.update({"reactome": 1.0, "cellmarker": 0.95, "kegg": 0.75, "proteinatlas": 0.65})
        usage_modes.update({"reactome": "full", "cellmarker": "prior_only", "kegg": "graph_only", "proteinatlas": "warmup_only"})
        actions.update({"reactome": "keep", "cellmarker": "keep", "kegg": "graph_only", "proteinatlas": "warmup_only"})
        notes.append("free_policy_protein_to_gene_conservative_pathway_probe")
    else:
        for sid in ["kegg", "cellmarker", "reactome"]:
            _add(selected, sid)
        if "proteinatlas" in available_ids and _cov("proteinatlas") >= 0.50:
            _add(selected, "proteinatlas")
        weights.update({"kegg": 0.85, "cellmarker": 1.0, "reactome": 0.95, "proteinatlas": 0.65})
        usage_modes.update({"kegg": "low_weight_full", "cellmarker": "full", "reactome": "full", "proteinatlas": "warmup_only"})
        actions.update({"kegg": "downweight", "cellmarker": "keep", "reactome": "keep", "proteinatlas": "warmup_only"})
        notes.append("free_policy_he_to_gene_probe_kegg_with_rule_context")

    if len(selected) < max_sources:
        ranked_extra = sorted(
            [c for c in candidates if c.available and c.source_id not in selected],
            key=lambda c: (float(c.data_coverage_score or 0.0), float(c.score)),
            reverse=True,
        )
        for c in ranked_extra:
            if len(selected) >= max_sources:
                break
            selected.append(c.source_id)
            weights[c.source_id] = 0.70
            usage_modes[c.source_id] = "low_weight_full"
            actions[c.source_id] = "add"
            notes.append(f"free_policy_data_coverage_fill:{c.source_id}")

    selected = _dedupe_keep_order(selected)[:max_sources]
    weights = {sid: float(weights.get(sid, 1.0)) for sid in selected}
    usage_modes = {sid: str(usage_modes.get(sid, "full")) for sid in selected}
    actions = {sid: str(actions.get(sid, "keep")) for sid in selected}
    return selected, notes, weights, usage_modes, actions


def _adaptive_budget(
    task: TaskSpec,
    candidates: List[SourceCandidate],
    min_sources: int,
    max_sources: int,
    data_profile: Optional[Dict] = None,
) -> Tuple[int, List[str]]:
    _, req_relations, flags = _requirements(task)
    budget = max(min_sources, min(max_sources, len(req_relations) + 2))
    notes: List[str] = [f"adaptive_budget_base:{budget}"]

    relation_ready = [
        c for c in candidates
        if c.available and ("relation_coverage=0.00" not in c.reasons)
    ]
    high_cov = [
        c for c in candidates
        if c.available and (c.data_coverage_score or 0.0) >= 0.20
    ]
    low_cov = False
    if isinstance(data_profile, dict):
        per_source = data_profile.get("source_coverage", {})
        if isinstance(per_source, dict):
            available_cov = [
                float(v.get("overall_coverage", 0.0) or 0.0)
                for v in per_source.values()
                if isinstance(v, dict) and v.get("coverage_available", False)
            ]
            if available_cov:
                low_cov = max(available_cov) < 0.35

    if flags.get("cross_modality_task", False) and len(relation_ready) >= budget + 1:
        budget += 1
        notes.append("adaptive_budget_cross_modality_bonus")
    elif len(high_cov) >= budget + 1:
        budget += 1
        notes.append("adaptive_budget_data_coverage_bonus")
    elif low_cov and len(relation_ready) >= budget + 1:
        budget += 1
        notes.append("adaptive_budget_low_coverage_redundancy_bonus")

    budget = max(min_sources, min(max_sources, budget))
    notes.append(f"adaptive_budget_final:{budget}")
    return budget, notes


def _prefilter_for_agent(candidates: List[SourceCandidate], pool_size: int) -> Tuple[List[SourceCandidate], List[str]]:
    if pool_size <= 0 or len(candidates) <= pool_size:
        return candidates, []

    mandatory = [c for c in candidates if c.available and c.is_mandatory]
    relation_useful = [
        c for c in candidates
        if c.available and c not in mandatory and ("relation_coverage=0.00" not in c.reasons)
    ]
    residual = [
        c for c in candidates
        if c.available and c not in mandatory and c not in relation_useful
    ]
    ordered = mandatory + relation_useful + residual
    trimmed = _dedupe_keep_order([c.source_id for c in ordered])[:pool_size]
    lookup = {c.source_id: c for c in candidates}
    out = [lookup[sid] for sid in trimmed if sid in lookup]
    return out, [f"agent_candidate_pool_trimmed:{len(candidates)}->{len(out)}"]


def _normalize_source_modes(selected_ids: List[str], usage_modes: Dict, source_actions: Dict) -> Tuple[Dict[str, str], Dict[str, str]]:
    allowed_modes = {"full", "low_weight_full", "graph_only", "prior_only", "warmup_only", "disabled"}
    mode_out: Dict[str, str] = {}
    action_out: Dict[str, str] = {}
    usage_modes = usage_modes if isinstance(usage_modes, dict) else {}
    source_actions = source_actions if isinstance(source_actions, dict) else {}
    for sid in selected_ids:
        mode = str(usage_modes.get(sid, "full")).strip().lower() or "full"
        if mode not in allowed_modes:
            mode = "full"
        action = str(source_actions.get(sid, "keep")).strip().lower() or "keep"
        mode_out[sid] = mode
        action_out[sid] = action
    return mode_out, action_out


def select_sources(
    task: TaskSpec,
    registry_sources: List[Dict],
    min_sources: int = 4,
    max_sources: int = 4,
    selection_mode: str = "agent_rule",
    data_profile: Optional[Dict] = None,
    selection_policy: str = "anchored",
    performance_feedback: Optional[Dict] = None,
) -> SelectionResult:
    task = task.normalized()
    selection_mode = str(selection_mode or "agent_rule").strip().lower()
    if selection_mode in {"rule", "rule+agent"}:
        selection_mode = {"rule": "rule_only", "rule+agent": "agent_rule"}[selection_mode]
    if selection_mode not in {"rule_only", "agent_only", "agent_rule"}:
        raise ValueError(f"Unsupported knowledge selection mode: {selection_mode}")

    candidates = [_compute_candidate(task, src, data_profile=data_profile) for src in registry_sources]
    available = [c for c in candidates if c.available]
    max_sources = max(int(max_sources), 1)
    min_sources = max(1, min(int(min_sources), int(max_sources)))
    target_budget, budget_notes = _adaptive_budget(
        task=task,
        candidates=available,
        min_sources=min_sources,
        max_sources=max_sources,
        data_profile=data_profile,
    )

    mandatory = sorted(
        [c for c in available if c.is_mandatory],
        key=lambda x: x.score,
        reverse=True,
    )
    non_mandatory = sorted(
        [c for c in available if not c.is_mandatory],
        key=lambda x: (x.is_recommended, x.score),
        reverse=True,
    )

    ranked = mandatory + non_mandatory
    mode = "rule_only"
    notes = list(budget_notes)
    agent_decision: Dict = {}
    agent_output = None
    if selection_mode in {"agent_only", "agent_rule"}:
        pool_size = max(
            target_budget,
            int(os.getenv("KNOWLEDGE_AGENT_CANDIDATE_POOL_SIZE", str(min(len(ranked), 8)))),
        )
        agent_ranked, pool_notes = _prefilter_for_agent(ranked, pool_size=pool_size)
        notes.extend(pool_notes)
        agent_output = run_selection_agent(
            task=task,
            candidates=agent_ranked,
            min_sources=min_sources,
            max_sources=target_budget,
            include_mandatory=(selection_mode == "agent_rule"),
            data_profile=data_profile,
            performance_feedback=performance_feedback,
        )

    if agent_output and agent_output.get("ok", False):
        mode = "agent_only" if selection_mode == "agent_only" else "agent+rule"
        notes.append("agent_selection_applied")
        selected_ids = _dedupe_keep_order(agent_output.get("selected_source_ids", []))
        if str(selection_policy or "anchored").strip().lower() != "free":
            constrained_ids, constrained_notes = _constrain_agent_selection_to_task_template(
                task=task,
                proposed_ids=selected_ids,
                candidates=available,
                max_sources=target_budget,
            )
            if constrained_ids:
                selected_ids = constrained_ids
                notes.extend(constrained_notes)
        feedback_ids, feedback_notes, feedback_weights, feedback_usage_modes, feedback_actions = _apply_performance_feedback_policy(
            proposed_ids=selected_ids,
            candidates=available,
            max_sources=target_budget,
            performance_feedback=performance_feedback,
        )
        if feedback_notes:
            selected_ids = feedback_ids
            notes.extend(feedback_notes)
            if isinstance(agent_output, dict):
                merged_weights = dict(agent_output.get("source_weights", {}) or {})
                merged_weights.update(feedback_weights)
                agent_output["source_weights"] = merged_weights
                agent_output["metric_feedback_weight_overrides"] = feedback_weights
                agent_output["source_usage_modes"] = dict(agent_output.get("source_usage_modes", {}) or {})
                agent_output["source_usage_modes"].update(feedback_usage_modes)
                agent_output["source_actions"] = dict(agent_output.get("source_actions", {}) or {})
                agent_output["source_actions"].update(feedback_actions)
                agent_output["metric_feedback_strategy"] = (
                    performance_feedback.get("refine_strategy")
                    if isinstance(performance_feedback, dict)
                    else ""
                )
        agent_decision = agent_output
    else:
        if isinstance(agent_output, dict) and agent_output:
            agent_decision = agent_output
        if selection_mode in {"agent_only", "agent_rule"} and str(os.getenv("KNOWLEDGE_AGENT_ENABLE_AGENT", os.getenv("KNOWLEDGE_AGENT_USE_LLM", "0"))).strip().lower() in {"1", "true", "yes"}:
            notes.append("agent_unavailable_or_call_failed_policy_rescue")
        if performance_feedback is not None:
            seed_ids = (
                performance_feedback.get("round1_free_agent", {}).get("selected_sources", [])
                if isinstance(performance_feedback, dict)
                else []
            )
            selected_ids, feedback_notes, feedback_weights, feedback_usage_modes, feedback_actions = _apply_performance_feedback_policy(
                proposed_ids=seed_ids,
                candidates=available,
                max_sources=target_budget,
                performance_feedback=performance_feedback,
            )
            notes.extend(feedback_notes)
            mode = "agent+rule" if selection_mode != "agent_only" else "agent_only"
            agent_decision = {
                **(agent_decision if isinstance(agent_decision, dict) else {}),
                "ok": True,
                "policy_rescue": "metric_aware_refinement_after_llm_failure",
                "selected_source_ids": selected_ids,
                "source_weights": feedback_weights,
                "source_usage_modes": feedback_usage_modes,
                "source_actions": feedback_actions,
                "source_weight_strategy": "metric_feedback_policy_rescue",
                "metric_feedback_strategy": performance_feedback.get("refine_strategy", ""),
                "confidence": 0.50,
            }
        elif str(selection_policy or "").strip().lower() == "free" and selection_mode == "agent_only":
            selected_ids, free_notes, free_weights, free_usage_modes, free_actions = _apply_free_exploration_policy(
                task=task,
                candidates=available,
                max_sources=target_budget,
                data_profile=data_profile,
            )
            notes.extend(free_notes)
            mode = "agent_only"
            agent_decision = {
                **(agent_decision if isinstance(agent_decision, dict) else {}),
                "ok": True,
                "policy_rescue": "task_data_free_exploration_after_llm_failure",
                "selected_source_ids": selected_ids,
                "source_weights": free_weights,
                "source_usage_modes": free_usage_modes,
                "source_actions": free_actions,
                "source_weight_strategy": "task_data_free_policy_rescue",
                "confidence": 0.45,
            }
        else:
            notes.append("fallback_to_ranked_rule_selection")
            selected_ids = []
            for c in ranked:
                if c.is_mandatory:
                    selected_ids.append(c.source_id)
            for c in ranked:
                if len(_dedupe_keep_order(selected_ids)) >= target_budget:
                    break
                selected_ids.append(c.source_id)

    if selection_mode != "agent_only":
        mandatory_ids = [c.source_id for c in ranked if c.is_mandatory]
        selected_ids = _dedupe_keep_order(mandatory_ids + selected_ids)

    selected_ids = _dedupe_keep_order(selected_ids)[:target_budget]
    selected_ids, repair_notes = _repair_required_relation_coverage(
        selected_ids=selected_ids,
        candidates=candidates,
        registry_sources=registry_sources,
        task=task,
        max_sources=target_budget,
    )
    notes.extend(repair_notes)
    if not selected_ids and ranked:
        selected_ids = [ranked[0].source_id]
        notes.append("fallback_to_top_source")
    if isinstance(agent_decision, dict) and isinstance(agent_decision.get("source_weights", {}), dict):
        sw = {}
        for sid in selected_ids:
            try:
                sw[sid] = float(agent_decision.get("source_weights", {}).get(sid, 1.0))
            except Exception:
                sw[sid] = 1.0
        agent_decision["source_weights"] = sw
        agent_decision["selected_source_ids"] = list(selected_ids)
        mode_out, action_out = _normalize_source_modes(
            selected_ids,
            agent_decision.get("source_usage_modes", {}),
            agent_decision.get("source_actions", {}),
        )
        agent_decision["source_usage_modes"] = mode_out
        agent_decision["source_actions"] = action_out

    return SelectionResult(
        mode=mode,
        selected_source_ids=selected_ids,
        candidates=sorted(candidates, key=lambda x: x.score, reverse=True),
        notes=notes,
        agent_decision=agent_decision,
    )
