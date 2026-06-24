"""
Knowledge orchestrator with iterative validation-rereasoning loop.

Upgrade over v1:
- When KB validation fails, agent re-reasons with validation feedback
  instead of falling back to hardcoded rules
- Full reasoning chain and correction history recorded in report
- Supports max_correction_rounds to bound the loop
"""
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from .agent import merge_registry_and_discovery, run_discovery_agent, run_selection_agent
from .data_profiler import build_data_profile
from .registry import load_registry
from .selector import select_sources
from .task_schema import TaskSpec
from .validator import validate_kb_stats


@dataclass
class KnowledgePathConfig:
    hgnc_file: Optional[str] = None
    uniprot_file: Optional[str] = None
    reactome_uniprot_file: Optional[str] = None
    reactome_ensembl_file: Optional[str] = None
    cellmarker_file: Optional[str] = None
    dorothea_file: Optional[str] = None
    omnipath_file: Optional[str] = None
    corum_file: Optional[str] = None
    proteinatlas_file: Optional[str] = None
    kegg_pathways_file: Optional[str] = None
    kegg_gene_pathway_file: Optional[str] = None
    string_file: Optional[str] = None


def _dedupe_keep_order(items):
    seen = set()
    out = []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _path_exists(path: Optional[str]) -> bool:
    return bool(path) and os.path.exists(path)


def _local_source_status(cfg: KnowledgePathConfig) -> Dict[str, bool]:
    return {
        "hgnc": _path_exists(cfg.hgnc_file),
        "uniprot": _path_exists(cfg.uniprot_file),
        "reactome": _path_exists(cfg.reactome_uniprot_file) and _path_exists(cfg.reactome_ensembl_file),
        "cellmarker": _path_exists(cfg.cellmarker_file),
        "dorothea": _path_exists(cfg.dorothea_file),
        "omnipath": _path_exists(cfg.omnipath_file),
        "corum": _path_exists(cfg.corum_file),
        "proteinatlas": _path_exists(cfg.proteinatlas_file),
        "kegg": _path_exists(cfg.kegg_gene_pathway_file),
        "string": _path_exists(cfg.string_file),
        "hmdb": False,
        "chebi": False,
    }


def _paths_from_sources(selected_sources: Sequence[str], cfg: KnowledgePathConfig) -> Dict[str, Optional[str]]:
    selected = set([str(x).strip().lower() for x in selected_sources])
    return {
        "hgnc_file": cfg.hgnc_file if "hgnc" in selected else None,
        "uniprot_file": cfg.uniprot_file if "uniprot" in selected else None,
        "reactome_uniprot_file": cfg.reactome_uniprot_file if "reactome" in selected else None,
        "reactome_ensembl_file": cfg.reactome_ensembl_file if "reactome" in selected else None,
        "cellmarker_file": cfg.cellmarker_file if "cellmarker" in selected else None,
        "dorothea_file": cfg.dorothea_file if "dorothea" in selected else None,
        "omnipath_file": cfg.omnipath_file if "omnipath" in selected else None,
        "corum_file": cfg.corum_file if "corum" in selected else None,
        "proteinatlas_file": cfg.proteinatlas_file if "proteinatlas" in selected else None,
        "kegg_pathways_file": cfg.kegg_pathways_file if "kegg" in selected else None,
        "kegg_gene_pathway_file": cfg.kegg_gene_pathway_file if "kegg" in selected else None,
        "string_file": cfg.string_file if "string" in selected else None,
    }


def _ensure_core_sources(selected_sources: Sequence[str], core_sources: Optional[Sequence[str]] = None) -> Sequence[str]:
    core = list(core_sources or ["hgnc", "uniprot"])
    combined = list(core) + list(selected_sources)
    return _dedupe_keep_order([str(x).strip().lower() for x in combined])


def _build_kb_once(
    builder_fn: Callable,
    protein_names,
    gene_names,
    path_cfg: Dict[str, Optional[str]],
    builder_kwargs: Dict,
):
    return builder_fn(
        protein_names=protein_names,
        gene_names=gene_names,
        hgnc_file=path_cfg["hgnc_file"],
        uniprot_file=path_cfg["uniprot_file"],
        reactome_uniprot_file=path_cfg["reactome_uniprot_file"],
        reactome_ensembl_file=path_cfg["reactome_ensembl_file"],
        cellmarker_file=path_cfg["cellmarker_file"],
        dorothea_file=path_cfg.get("dorothea_file"),
        omnipath_file=path_cfg.get("omnipath_file"),
        corum_file=path_cfg.get("corum_file"),
        proteinatlas_file=path_cfg.get("proteinatlas_file"),
        kegg_pathways_file=path_cfg.get("kegg_pathways_file"),
        kegg_gene_pathway_file=path_cfg.get("kegg_gene_pathway_file"),
        string_file=path_cfg.get("string_file"),
        **builder_kwargs,
    )


def _append_unique_search_plan(plan: list, item: Dict):
    sid = str(item.get("source_id", "")).strip().lower()
    reason = str(item.get("reason", "")).strip()
    for x in plan:
        if str(x.get("source_id", "")).strip().lower() == sid and str(x.get("reason", "")).strip() == reason:
            return
    plan.append(item)


def _agent_correction_round(
    task: TaskSpec,
    candidates,
    validation: Dict,
    previous_selection: List[str],
    max_sources: int,
    selection_mode: str,
    include_mandatory: bool,
    llm_client=None,
    data_profile: Optional[Dict] = None,
) -> Optional[Dict]:
    """
    Ask the agent to re-reason given validation failure feedback.
    Returns a new agent_output dict or None if agent unavailable.
    """
    from .llm_client import create_llm_client_from_env
    llm = llm_client or create_llm_client_from_env()
    if llm is None:
        return None

    validation_feedback = {
        "errors": validation.get("errors", []),
        "warnings": validation.get("warnings", []),
        "summary": validation.get("summary", {}),
        "message": (
            "The knowledge base built from your selected sources failed validation. "
            "Please select different or additional sources to resolve these issues."
        ),
    }

    return run_selection_agent(
        task=task,
        candidates=candidates,
        max_sources=max_sources,
        llm_client=llm,
        include_mandatory=include_mandatory,
        validation_feedback=validation_feedback,
        previous_selection=previous_selection,
        max_react_iterations=1,  # correction round itself is already a re-reasoning
        data_profile=data_profile,
    )


def build_kb_with_orchestration(
    task_spec: TaskSpec,
    protein_names,
    gene_names,
    kb_paths: KnowledgePathConfig,
    builder_fn: Callable,
    builder_kwargs: Optional[Dict] = None,
    builder_kwargs_adapter: Optional[Callable] = None,
    registry_path: Optional[str] = None,
    min_sources: int = 4,
    max_sources: int = 4,
    output_dir: Optional[str] = None,
    require_agent: bool = False,
    selection_mode: str = "agent_rule",
    ensure_core_sources: Optional[Sequence[str]] = None,
    max_correction_rounds: int = 2,
    data_profile: Optional[Dict] = None,
    use_data_profile: bool = True,
    selection_policy: str = "anchored",
    performance_feedback: Optional[Dict] = None,
) -> Tuple[Tuple, Dict]:
    """
    Build knowledge base with LLM-orchestrated source selection.

    New in v2:
    - max_correction_rounds: if KB validation fails, agent re-reasons up to
      this many times before falling back to hardcoded rules.
    - Full reasoning chain and correction history in report.
    """
    builder_kwargs = dict(builder_kwargs or {})
    task = task_spec.normalized()
    selection_mode = str(selection_mode or "agent_rule").strip().lower()
    if selection_mode in {"rule", "rule+agent"}:
        selection_mode = {"rule": "rule_only", "rule+agent": "agent_rule"}[selection_mode]
    if selection_mode not in {"rule_only", "agent_only", "agent_rule"}:
        raise ValueError(f"Unsupported knowledge selection mode: {selection_mode}")

    correction_history: List[Dict] = []

    if not use_data_profile:
        data_profile = None
    elif data_profile is None:
        data_profile = build_data_profile(
            task_spec=task,
            protein_names=protein_names,
            gene_names=gene_names,
            kb_paths=kb_paths,
        )

    # Step 1: Discover candidate KBs
    registry_sources = load_registry(registry_path)
    discovery = run_discovery_agent(task=task, registry_sources=registry_sources, data_profile=data_profile)
    merged_sources = merge_registry_and_discovery(registry_sources, discovery)
    registry_map = {str(s.get("id", "")).strip().lower(): s for s in merged_sources}

    # Step 2: Initial source selection
    selection = select_sources(
        task,
        registry_sources=merged_sources,
        min_sources=min_sources,
        max_sources=max_sources,
        selection_mode=selection_mode,
        data_profile=data_profile,
        selection_policy=selection_policy,
        performance_feedback=performance_feedback,
    )
    if require_agent and selection.mode not in {"agent+rule", "agent_only"}:
        raise RuntimeError(
            "Knowledge agent was required but LLM source selection did not run. "
            "Check KNOWLEDGE_AGENT_ENABLE_AGENT, provider, endpoint, API key, model, and network access."
        )
    core_sources = (
        []
        if selection_mode == "agent_only"
        else (ensure_core_sources if ensure_core_sources is not None else ["hgnc", "uniprot"])
    )
    requested_sources = _ensure_core_sources(selection.selected_source_ids, core_sources=core_sources)

    # Step 3: Local-availability downgrade
    local_status = _local_source_status(kb_paths)
    local_selected_sources = [sid for sid in requested_sources if local_status.get(sid, False)]
    downgraded_sources = [sid for sid in requested_sources if sid not in local_selected_sources]
    selected_sources = _ensure_core_sources(local_selected_sources)

    selected_paths = _paths_from_sources(selected_sources, kb_paths)
    if selection_mode != "agent_only" and selected_paths["hgnc_file"] is None:
        selected_paths["hgnc_file"] = kb_paths.hgnc_file
    if selection_mode != "agent_only" and selected_paths["uniprot_file"] is None:
        selected_paths["uniprot_file"] = kb_paths.uniprot_file
    if not _path_exists(selected_paths["hgnc_file"]):
        raise FileNotFoundError(f"Required local HGNC file not found: {selected_paths['hgnc_file']}")
    if not _path_exists(selected_paths["uniprot_file"]):
        raise FileNotFoundError(f"Required local UniProt file not found: {selected_paths['uniprot_file']}")

    builder_kwargs_before_adapter = dict(builder_kwargs)
    weight_application = {}
    if builder_kwargs_adapter is not None:
        adapted = builder_kwargs_adapter(
            builder_kwargs=dict(builder_kwargs),
            selection=selection.to_dict(),
            selected_sources=selected_sources,
            task_spec=task.to_dict(),
        )
        if isinstance(adapted, tuple):
            builder_kwargs, weight_application = adapted
        else:
            builder_kwargs = adapted
            weight_application = {}
        builder_kwargs = dict(builder_kwargs or {})
        weight_application = dict(weight_application or {})

    # Step 4: Build KB
    kb_tuple = _build_kb_once(
        builder_fn=builder_fn,
        protein_names=protein_names,
        gene_names=gene_names,
        path_cfg=selected_paths,
        builder_kwargs=builder_kwargs,
    )
    stats = kb_tuple[4] if isinstance(kb_tuple, tuple) and len(kb_tuple) > 4 else {}
    validation = validate_kb_stats(stats=stats, task_spec=task)

    # Step 5: Iterative correction loop (agent re-reasons on validation failure)
    correction_round = 0
    use_agent_correction = (
        not validation["ok"]
        and selection_mode in {"agent_only", "agent_rule"}
        and max_correction_rounds > 0
    )

    while not validation["ok"] and correction_round < max_correction_rounds and use_agent_correction:
        correction_round += 1
        correction_history.append({
            "round": correction_round,
            "failed_sources": list(selected_sources),
            "validation_errors": validation["errors"],
            "validation_warnings": validation["warnings"],
        })

        corrected_agent_output = _agent_correction_round(
            task=task,
            candidates=selection.candidates,
            validation=validation,
            previous_selection=list(selected_sources),
            max_sources=max_sources,
            selection_mode=selection_mode,
            include_mandatory=(selection_mode == "agent_rule"),
            data_profile=data_profile,
        )

        if corrected_agent_output is None:
            correction_history[-1]["outcome"] = "agent_unavailable_stopping_correction"
            break

        correction_history[-1]["agent_reasoning_steps"] = corrected_agent_output.get("reasoning_steps", [])
        correction_history[-1]["correction_analysis"] = corrected_agent_output.get("correction_analysis", "")
        correction_history[-1]["corrected_selection"] = corrected_agent_output.get("selected_source_ids", [])

        # Apply corrected selection
        corrected_ids = corrected_agent_output.get("selected_source_ids", [])
        if not corrected_ids:
            correction_history[-1]["outcome"] = "agent_returned_empty_selection"
            break

        corrected_requested = _ensure_core_sources(corrected_ids, core_sources=core_sources)
        corrected_local = [sid for sid in corrected_requested if local_status.get(sid, False)]
        corrected_sources = _ensure_core_sources(corrected_local)
        corrected_paths = _paths_from_sources(corrected_sources, kb_paths)
        if corrected_paths["hgnc_file"] is None:
            corrected_paths["hgnc_file"] = kb_paths.hgnc_file
        if corrected_paths["uniprot_file"] is None:
            corrected_paths["uniprot_file"] = kb_paths.uniprot_file

        kb_tuple = _build_kb_once(
            builder_fn=builder_fn,
            protein_names=protein_names,
            gene_names=gene_names,
            path_cfg=corrected_paths,
            builder_kwargs=builder_kwargs,
        )
        stats = kb_tuple[4] if isinstance(kb_tuple, tuple) and len(kb_tuple) > 4 else {}
        validation = validate_kb_stats(stats=stats, task_spec=task)
        selected_sources = corrected_sources
        correction_history[-1]["outcome"] = "ok" if validation["ok"] else "still_invalid"

    # Step 6: Final hardcoded fallback (only if agent correction exhausted)
    fallback_used = False
    fallback_selected: List[str] = []
    if not validation["ok"]:
        fallback_used = True
        fallback_selected = [
            sid for sid in ["hgnc", "uniprot", "reactome", "cellmarker"] if local_status.get(sid, False)
        ]
        fallback_paths = _paths_from_sources(fallback_selected, kb_paths)
        if fallback_paths["hgnc_file"] is None:
            fallback_paths["hgnc_file"] = kb_paths.hgnc_file
        if fallback_paths["uniprot_file"] is None:
            fallback_paths["uniprot_file"] = kb_paths.uniprot_file

        kb_tuple = _build_kb_once(
            builder_fn=builder_fn,
            protein_names=protein_names,
            gene_names=gene_names,
            path_cfg=fallback_paths,
            builder_kwargs=builder_kwargs,
        )
        stats = kb_tuple[4] if isinstance(kb_tuple, tuple) and len(kb_tuple) > 4 else {}
        validation = validate_kb_stats(stats=stats, task_spec=task)

    # Build report
    selection_dict = selection.to_dict()
    agent_decision = selection_dict.get("agent_decision", {})
    agent_failure = {}
    if isinstance(agent_decision, dict) and not agent_decision.get("ok", True):
        agent_failure = {
            "failure_stage": agent_decision.get("failure_stage", ""),
            "failure_message": agent_decision.get("failure_message", ""),
            "provider": agent_decision.get("provider", ""),
            "model": agent_decision.get("model", ""),
            "trace": agent_decision.get("trace", {}),
        }

    report = {
        "timestamp_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "task_spec": task.to_dict(),
        "data_profile": data_profile,
        "selection_mode_requested": selection_mode,
        "selection_policy": str(selection_policy or ""),
        "registry_size": len(registry_sources),
        "discovery": discovery,
        "discovered_registry_size": len(merged_sources),
        "requested_selected_sources": requested_sources,
        "selection": selection_dict,
        "selection_budget": {"min_sources": int(min_sources), "max_sources": int(max_sources)},
        "final_selected_sources": fallback_selected if fallback_used else selected_sources,
        "downgraded_sources": downgraded_sources,
        "local_source_status": local_status,
        "fallback_used": fallback_used,
        "correction_rounds_used": correction_round,
        "correction_history": correction_history,
        # ReAct reasoning chain from initial selection
        "agent_reasoning_steps": agent_decision.get("reasoning_steps", []) if isinstance(agent_decision, dict) else [],
        "agent_self_critique": agent_decision.get("self_critique", "") if isinstance(agent_decision, dict) else "",
        "agent_failure": agent_failure,
        "source_search_plan": [],
        "validation": validation,
        "builder_stats": stats,
        "builder_kwargs_before_agent_weights": builder_kwargs_before_adapter,
        "builder_kwargs_effective": dict(builder_kwargs),
        "agent_weight_application": weight_application,
        "kb_paths": asdict(kb_paths),
        "performance_feedback": performance_feedback,
    }

    for sid in downgraded_sources:
        if sid in {"hgnc", "uniprot", "reactome", "cellmarker"}:
            continue
        src_meta = registry_map.get(sid, {})
        _append_unique_search_plan(
            report["source_search_plan"],
            {
                "source_id": sid,
                "reason": "selected_by_agent_but_not_locally_available_downgraded_to_local_sources",
                "homepage": src_meta.get("homepage", ""),
                "api_docs": src_meta.get("api_docs", ""),
                "search_query": f"{sid} official download API documentation",
            },
        )

    download_candidates = (
        agent_decision.get("download_candidates", [])
        if isinstance(agent_decision, dict)
        else []
    )
    for item in download_candidates:
        sid = str(item.get("source_id", "")).strip().lower()
        if not sid:
            continue
        src_meta = registry_map.get(sid, {})
        _append_unique_search_plan(
            report["source_search_plan"],
            {
                "source_id": sid,
                "reason": item.get("reason", ""),
                "homepage": src_meta.get("homepage", ""),
                "api_docs": src_meta.get("api_docs", ""),
                "search_query": f"{sid} official download API documentation",
            },
        )

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        report_name = f"kb_orchestration_{task.task_id}.json"
        report_path = os.path.join(output_dir, report_name)
        trace_path = os.path.join(output_dir, f"agent_trace_{task.task_id}.json")
        report["report_path"] = report_path
        report["agent_trace_path"] = trace_path
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        with open(trace_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "task_spec": report["task_spec"],
                    "data_profile": report["data_profile"],
                    "discovery": report["discovery"],
                    "selection": report["selection"],
                    "requested_selected_sources": report["requested_selected_sources"],
                    "selection_budget": report["selection_budget"],
                    "final_selected_sources": report["final_selected_sources"],
                    "downgraded_sources": report["downgraded_sources"],
                    "agent_weight_application": report["agent_weight_application"],
                    "builder_kwargs_effective": report["builder_kwargs_effective"],
                    "validation": report["validation"],
                    "builder_stats": report["builder_stats"],
                    # New: full reasoning chain
                    "agent_reasoning_steps": report["agent_reasoning_steps"],
                    "agent_self_critique": report["agent_self_critique"],
                    "agent_failure": report["agent_failure"],
                    "correction_rounds_used": report["correction_rounds_used"],
                    "correction_history": report["correction_history"],
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        report["agent_trace_path"] = trace_path

    return kb_tuple, report
