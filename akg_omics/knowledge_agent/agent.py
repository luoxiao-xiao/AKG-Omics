"""
Knowledge agent with ReAct-style reasoning loop and tool calling (Layer 2).

Upgrade over v1:
- Multi-step reasoning: Thought 鈫?Action 鈫?Observation 鈫?Thought ...
- Iterative self-correction: agent re-reasons when validation fails
- Full reasoning chain recorded in trace for interpretability
- Supports reasoning_content extraction (DeepSeek-R1, o1, Claude thinking)

Upgrade v2 (tool calling):
- Agent can invoke live biomedical APIs (UniProt, DisGeNET, KEGG) via function calling
- Tool observations injected back into LLM context for grounded reasoning
- Max tool_calling_rounds configurable via KNOWLEDGE_AGENT_MAX_TOOL_ROUNDS (default 3)
- Falls back gracefully to JSON-mode ReAct when provider lacks native tool support
"""
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

from .llm_client import BaseLLMClient, create_llm_client_from_env
from .task_schema import SourceCandidate, TaskSpec


def _normalize_text(x: Any) -> str:
    return str(x).strip()


def _normalize_id(x: Any) -> str:
    return _normalize_text(x).lower().replace(" ", "_")


def _dedupe_keep_order(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _debug_log(msg: str) -> None:
    if str(os.getenv("KNOWLEDGE_AGENT_DEBUG", "0")).strip().lower() in {"1", "true", "yes"}:
        print(f"[knowledge_agent][debug] {msg}", file=sys.stderr, flush=True)


def _to_str_list(x: Any) -> List[str]:
    if not isinstance(x, list):
        return []
    out = []
    for v in x:
        s = _normalize_text(v).lower()
        if s:
            out.append(s)
    return _dedupe_keep_order(out)


def _failure_result(stage: str, provider: str = "", model: str = "", trace: Optional[Dict] = None, message: str = "") -> Dict:
    trace = dict(trace or {})
    return {
        "ok": False,
        "failure_stage": stage,
        "failure_message": message,
        "provider": provider or trace.get("provider", ""),
        "model": model or trace.get("model", ""),
        "trace": trace,
        "selected_source_ids": [],
        "source_weights": {},
        "reasoning_steps": [],
        "tool_calls": [],
        "tool_observations": [],
    }


def _postprocess_source_weights(
    selected: List[str],
    raw_weights: Any,
    candidates: List[SourceCandidate],
) -> Tuple[Dict[str, float], str]:
    selected = [str(s).strip().lower() for s in selected if str(s).strip()]
    raw_weights = raw_weights if isinstance(raw_weights, dict) else {}
    if not selected:
        return {}, "no_selected_sources"

    cand_lookup = {str(c.source_id).strip().lower(): c for c in candidates}
    weights: Dict[str, float] = {}
    for sid in selected:
        try:
            weights[sid] = float(raw_weights.get(sid, 0.0))
        except Exception:
            weights[sid] = 0.0

    positive = [v for v in weights.values() if v > 0]
    diverse = len({round(v, 3) for v in positive}) > 1
    if positive and diverse:
        return weights, "agent_provided"

    derived_scores = []
    for sid in selected:
        c = cand_lookup.get(sid)
        if c is None:
            derived_scores.append(1.0)
            continue
        rel_bonus = 0.15 if any("relation_coverage=0.00" not in r for r in c.reasons if "relation_coverage=" in r) else 0.0
        cov_bonus = float(c.data_coverage_score or 0.0) * 0.20
        mandatory_bonus = 0.10 if c.is_mandatory else 0.0
        derived_scores.append(max(0.10, float(c.score) + rel_bonus + cov_bonus + mandatory_bonus))

    mean_score = sum(derived_scores) / max(len(derived_scores), 1)
    out = {}
    for sid, score in zip(selected, derived_scores):
        out[sid] = float(max(0.35, min(1.85, score / max(mean_score, 1e-8))))
    return out, "derived_from_candidate_scores"


def _candidate_payload(candidates: List[SourceCandidate]) -> List[Dict]:
    out = []
    for c in candidates:
        detail = {}
        if isinstance(c.data_coverage_detail, dict) and c.data_coverage_score is not None:
            per_modality = c.data_coverage_detail.get("per_modality", {})
            if isinstance(per_modality, dict):
                detail["per_modality"] = {
                    str(k): {
                        "coverage": v.get("coverage", 0.0),
                        "covered_count": v.get("covered_count", 0),
                        "total_count": v.get("total_count", 0),
                    }
                    for k, v in per_modality.items()
                    if isinstance(v, dict)
                }
            detail["covered_count"] = c.data_coverage_detail.get("covered_count", 0)
            detail["total_count"] = c.data_coverage_detail.get("total_count", 0)
        out.append(
            {
                "source_id": c.source_id,
                "score": c.score,
                "is_mandatory": c.is_mandatory,
                "is_recommended": c.is_recommended,
                "available": c.available,
                "reasons": c.reasons,
                "data_coverage_score": c.data_coverage_score,
                "data_coverage_detail": detail,
            }
        )
    return out


def _compact_candidate_payload(candidates: List[SourceCandidate]) -> List[Dict]:
    out = []
    for c in candidates:
        relation_tags = []
        for reason in c.reasons:
            rs = str(reason)
            if rs.startswith("relation_coverage="):
                relation_tags.append(rs)
            elif rs in {"mandatory_by_rule", "recommended_by_rule"}:
                relation_tags.append(rs)
        out.append(
            {
                "source_id": c.source_id,
                "score": round(float(c.score), 4),
                "mandatory": bool(c.is_mandatory),
                "recommended": bool(c.is_recommended),
                "data_coverage_score": None if c.data_coverage_score is None else round(float(c.data_coverage_score), 4),
                "relation_hints": relation_tags,
            }
        )
    return out


def _selection_system_prompt_lightweight() -> str:
    return (
        "You are a knowledge-source selection agent for multimodal biology tasks. "
        "Choose external knowledge bases that best satisfy the task's required relations and observed data coverage. "
        "Return strict JSON only. Do not output markdown, prose, or reasoning text outside JSON.\n\n"
        "Selection rules:\n"
        "1. Always prioritize covering every required relation.\n"
        "2. Keep mandatory sources when requested.\n"
        "3. Use data_coverage_score only as a tie-breaker among relation-compatible sources.\n"
        "4. Prefer concise, robust selections that satisfy min_sources..max_sources.\n"
        "5. Do not invent source ids; only use allowed_source_ids.\n\n"
        "When performance_feedback is provided, first compare round1_free_agent against the rule "
        "baseline using metric directions: PCC/SSIM higher is better, CMD/RMSE lower is better. "
        "Follow refine_strategy exactly: exploit_success preserves successful round1 additions, "
        "hybrid_refine combines rule sources with useful additions, hard_recover anchors to "
        "rule sources and keeps at most one low-weight probe from round1. "
        "Use source_usage_modes to choose among full, low_weight_full, graph_only, prior_only, "
        "warmup_only, or disabled.\n\n"
        "Return exactly this JSON schema:\n"
        "{\n"
        '  "selected_source_ids": ["source_id"],\n'
        '  "source_weights": {"source_id": 1.0},\n'
        '  "source_actions": {"source_id": "keep|remove|downweight|graph_only|prior_only|warmup_only|disabled"},\n'
        '  "source_usage_modes": {"source_id": "full|low_weight_full|graph_only|prior_only|warmup_only|disabled"},\n'
        '  "round1_outcome": "success|partial_success|underperformed",\n'
        '  "refine_strategy": "exploit_success|hybrid_refine|hard_recover",\n'
        '  "metric_focus": ["PCC"],\n'
        '  "confidence": 0.0,\n'
        '  "decision_summary": "short string",\n'
        '  "source_rationales": {"source_id": "short reason"}\n'
        "}"
    )


def _registry_payload(registry_sources: List[Dict]) -> List[Dict]:
    out = []
    for s in registry_sources:
        out.append(
            {
                "id": _normalize_id(s.get("id", "")),
                "display_name": _normalize_text(s.get("display_name", s.get("id", ""))),
                "entities": _to_str_list(s.get("entities", [])),
                "relations": _to_str_list(s.get("relations", [])),
                "homepage": _normalize_text(s.get("homepage", "")),
                "api_docs": _normalize_text(s.get("api_docs", "")),
                "available": bool(s.get("available", False)),
            }
        )
    return out


def _rule_mandatory(candidates: List[SourceCandidate]) -> List[str]:
    return [c.source_id for c in candidates if c.available and c.is_mandatory]


def _data_profile_payload(data_profile: Optional[Dict]) -> Dict:
    if not isinstance(data_profile, dict):
        return {}

    feature_sets = {}
    for modality, info in (data_profile.get("feature_sets", {}) or {}).items():
        if not isinstance(info, dict):
            continue
        feature_sets[str(modality)] = {
            "count": info.get("count", 0),
            "id_type": info.get("id_type", ""),
            "examples": list(info.get("examples", []) or [])[:8],
        }

    source_coverage = {}
    for sid, cov in (data_profile.get("source_coverage", {}) or {}).items():
        if not isinstance(cov, dict):
            continue
        per_modality = {}
        for modality, m in (cov.get("per_modality", {}) or {}).items():
            if not isinstance(m, dict):
                continue
            per_modality[str(modality)] = {
                "coverage": round(float(m.get("coverage", 0.0) or 0.0), 4),
                "covered_count": m.get("covered_count", 0),
                "total_count": m.get("total_count", 0),
            }
        source_coverage[str(sid)] = {
            "overall_coverage": round(float(cov.get("overall_coverage", 0.0) or 0.0), 4),
            "covered_count": cov.get("covered_count", 0),
            "total_count": cov.get("total_count", 0),
            "coverage_available": bool(cov.get("coverage_available", False)),
            "per_modality": per_modality,
        }

    return {
        "schema_version": data_profile.get("schema_version", ""),
        "species": data_profile.get("species", ""),
        "tissue": data_profile.get("tissue", None),
        "modalities": data_profile.get("modalities", []),
        "feature_sets": feature_sets,
        "source_coverage": source_coverage,
        "warnings": list(data_profile.get("warnings", []) or [])[:8],
        "interpretation_hint": (
            "Use data coverage to break ties among relation-compatible sources. "
            "Do not drop sources needed for required_relations only because their direct ID overlap is low."
        ),
    }


# ---------------------------------------------------------------------------
# ReAct prompts
# ---------------------------------------------------------------------------

def _react_system_prompt() -> str:
    return (
        "You are a biomedical knowledge orchestration agent using ReAct reasoning.\n"
        "You reason step-by-step using Thought/Action/Observation cycles before giving a final answer.\n"
        "Format your response as a JSON object with these fields:\n"
        "  'reasoning_steps': list of {step, thought, action, observation} dicts\n"
        "  'final_answer': the actual decision JSON\n"
        "  'confidence': float 0-1\n"
        "  'self_critique': string noting any uncertainty or risk\n"
        "Do not invent unknown source IDs. Prioritize reliable, high-coverage, relation-matched sources."
    )


def _discovery_system_prompt() -> str:
    return (
        "You are a biomedical knowledge discovery agent using ReAct reasoning.\n"
        "Given task requirements and a known source catalog, reason step-by-step to propose "
        "additional relevant external knowledge bases that are NOT already in the catalog.\n"
        "Format your response as a JSON object with:\n"
        "  'reasoning_steps': list of {step, thought, action, observation} dicts\n"
        "  'final_answer': {discovered_sources: [...], notes: [...], confidence: float}\n"
        "  'confidence': float 0-1\n"
        "  'self_critique': string\n"
        "Return strict JSON only."
    )


def _correction_system_prompt() -> str:
    return (
        "You are a biomedical knowledge orchestration agent performing self-correction.\n"
        "Your previous knowledge base selection failed validation. "
        "Analyze the validation errors and re-select sources to fix the issues.\n"
        "Format your response as a JSON object with:\n"
        "  'reasoning_steps': list of {step, thought, action, observation} dicts\n"
        "  'correction_analysis': string explaining what went wrong\n"
        "  'final_answer': the corrected decision JSON\n"
        "  'confidence': float 0-1\n"
        "Return strict JSON only."
    )


# ---------------------------------------------------------------------------
# ReAct reasoning loop helpers
# ---------------------------------------------------------------------------

def _extract_react_result(parsed: Dict) -> Dict:
    """Extract final_answer and reasoning chain from a ReAct response."""
    if not isinstance(parsed, dict):
        return {}
    final = parsed.get("final_answer", {})
    if not isinstance(final, dict):
        final = {}
    return {
        "final_answer": final,
        "reasoning_steps": parsed.get("reasoning_steps", []),
        "confidence": parsed.get("confidence", None),
        "self_critique": parsed.get("self_critique", ""),
        "correction_analysis": parsed.get("correction_analysis", ""),
    }


def _run_react_call(
    llm: BaseLLMClient,
    system_prompt: str,
    payload: Dict,
    step_name: str,
) -> Dict:
    """Single ReAct LLM call, returns extracted result + raw trace."""
    parsed = llm.generate_json(system_prompt, payload)
    trace = dict(getattr(llm, "last_call_trace", {}) or {})
    trace["react_step"] = step_name

    if not isinstance(parsed, dict):
        _debug_log(f"react_call_failed at step={step_name}")
        return {"ok": False, "trace": trace, "result": {}}

    result = _extract_react_result(parsed)
    result["ok"] = True
    result["trace"] = trace
    return result


# ---------------------------------------------------------------------------
# Tool-calling loop (Layer 2 - live API integration)
# ---------------------------------------------------------------------------

def _tools_enabled() -> bool:
    return str(os.getenv("KNOWLEDGE_AGENT_ENABLE_TOOLS", "1")).strip().lower() in {"1", "true", "yes"}


def _max_tool_rounds() -> int:
    return max(0, int(os.getenv("KNOWLEDGE_AGENT_MAX_TOOL_ROUNDS", "3")))


def _run_tool_calling_loop(
    llm: "BaseLLMClient",
    system_prompt: str,
    initial_payload: Dict,
    step_name: str,
) -> Dict:
    """
    Full tool-calling loop for providers that support native function calling.

    Flow:
      Round 1: send payload + tool schemas 鈫?LLM returns tool_calls OR final JSON
      Round N: if tool_calls returned, execute tools, inject observations, repeat
      Final:   when LLM returns final JSON (no tool_calls), extract ReAct result

    Returns same shape as _run_react_call.
    """
    try:
        from .tool_executor import (
            build_tool_schemas_for_llm,
            run_tool_calling_round,
            summarise_observation,
            format_observations_as_message,
        )
    except ImportError as e:
        _debug_log(f"tool_executor import failed: {e}, falling back to JSON-mode")
        return _run_react_call(llm, system_prompt, initial_payload, step_name)

    provider = getattr(llm.cfg, "provider", "openai")
    max_rounds = _max_tool_rounds()
    all_tool_calls: List[Dict] = []
    all_observations: List[Dict] = []
    all_traces: List[Dict] = []

    # Build tool schemas for this provider
    tool_schemas = build_tool_schemas_for_llm(provider)

    # First call: send tools + initial payload
    # We embed tool schemas in the payload as a hint (JSON-mode providers)
    # Native tool-calling providers will pick it up via the LLM client
    enriched_payload = dict(initial_payload)
    enriched_payload["available_tools"] = [s["name"] if isinstance(s, dict) and "name" in s
                                            else s.get("function", {}).get("name", "")
                                            for s in tool_schemas]
    enriched_payload["tool_calling_instruction"] = (
        "If you need live data to support your decision, output a 'tool_calls' list "
        "in your JSON response with entries like: "
        "{\"name\": \"query_uniprot\", \"parameters\": {...}}. "
        "After receiving tool observations, continue reasoning and output your final_answer. "
        "Do NOT call tools if you already have enough information."
    )

    current_payload = enriched_payload
    last_parsed: Optional[Dict] = None

    for round_idx in range(max_rounds + 1):
        parsed = llm.generate_json(system_prompt, current_payload)
        trace = dict(getattr(llm, "last_call_trace", {}) or {})
        trace["react_step"] = f"{step_name}_tool_round_{round_idx}"
        all_traces.append(trace)

        if not isinstance(parsed, dict):
            _debug_log(f"tool_loop parse failed at round={round_idx}")
            break

        last_parsed = parsed

        # Check if LLM issued tool calls (JSON-mode: "tool_calls" key in response)
        tool_calls_raw = parsed.get("tool_calls", [])
        if not tool_calls_raw or round_idx >= max_rounds:
            # No tool calls or max rounds reached 鈫?this is the final answer
            break

        # Execute tool calls
        from .tool_executor import execute_tool_calls, summarise_observation
        observations = execute_tool_calls(tool_calls_raw)
        all_tool_calls.extend(tool_calls_raw)
        all_observations.extend(observations)

        # Build compact observation summary for next LLM round
        compact_obs = [summarise_observation(o) for o in observations]
        obs_text = format_observations_as_message(compact_obs)

        _debug_log(
            f"tool_round={round_idx} called {len(tool_calls_raw)} tools, "
            f"got {len(observations)} observations"
        )

        # Inject observations into next payload
        current_payload = dict(initial_payload)
        current_payload["tool_observations"] = [
            {
                "tool": o["tool"],
                "ok": o["ok"],
                "result_summary": summarise_observation(o).get("result", {}),
                "error": o.get("error", ""),
                "latency_sec": o["latency_sec"],
            }
            for o in observations
        ]
        current_payload["previous_tool_calls"] = tool_calls_raw
        current_payload["instruction"] = (
            "You have received tool observations above. "
            "Use them to inform your final knowledge source selection. "
            "Now output your final_answer JSON (no more tool_calls needed)."
        )

    if last_parsed is None:
        return {"ok": False, "trace": all_traces, "result": {}, "tool_calls": [], "observations": []}

    result = _extract_react_result(last_parsed)
    result["ok"] = True
    result["trace"] = all_traces[-1] if all_traces else {}
    result["all_traces"] = all_traces
    result["tool_calls"] = all_tool_calls
    result["observations"] = all_observations
    return result


def _run_react_call_with_tools(
    llm: "BaseLLMClient",
    system_prompt: str,
    payload: Dict,
    step_name: str,
) -> Dict:
    """
    Dispatch to tool-calling loop if tools are enabled, else plain ReAct call.
    This is the main entry point replacing direct _run_react_call calls.
    """
    if _tools_enabled():
        return _run_tool_calling_loop(llm, system_prompt, payload, step_name)
    return _run_react_call(llm, system_prompt, payload, step_name)


# ---------------------------------------------------------------------------
# Discovery agent (ReAct)
# ---------------------------------------------------------------------------

def run_discovery_agent(
    task: TaskSpec,
    registry_sources: List[Dict],
    max_new_sources: int = 8,
    llm_client: Optional[BaseLLMClient] = None,
    data_profile: Optional[Dict] = None,
) -> Dict:
    skip_discovery = str(os.getenv("KNOWLEDGE_AGENT_SKIP_DISCOVERY", "0")).strip().lower() in {"1", "true", "yes"}
    if skip_discovery:
        return {
            "mode": "catalog_only",
            "discovered_sources": [],
            "notes": ["discovery_disabled_by_env"],
            "confidence": None,
            "provider": None,
            "model": None,
            "reasoning_steps": [],
        }

    llm = llm_client or create_llm_client_from_env()
    if llm is None:
        return {
            "mode": "catalog_only",
            "discovered_sources": [],
            "notes": ["agent_unavailable_discovery_skipped"],
            "confidence": None,
            "provider": None,
            "model": None,
            "reasoning_steps": [],
        }

    payload = {
        "task": task.to_dict(),
        "data_profile": _data_profile_payload(data_profile),
        "known_sources": _registry_payload(registry_sources),
        "constraints": {
            "max_new_sources": max_new_sources,
            "prefer_human_biomedical_sources": True,
        },
        "required_output_schema": {
            "reasoning_steps": [
                {"step": 1, "thought": "string", "action": "string", "observation": "string"}
            ],
            "final_answer": {
                "discovered_sources": [
                    {
                        "id": "string",
                        "display_name": "string",
                        "entities": ["string"],
                        "relations": ["string"],
                        "homepage": "string",
                        "api_docs": "string",
                        "available": True,
                        "quality_weight": 0.7,
                        "reason": "string",
                    }
                ],
                "notes": ["string"],
                "confidence": 0.0,
            },
            "confidence": 0.0,
            "self_critique": "string",
        },
    }

    react_result = _run_react_call_with_tools(llm, _discovery_system_prompt(), payload, "discovery")

    if not react_result.get("ok"):
        return {
            "mode": "catalog_only",
            "discovered_sources": [],
            "notes": ["agent_discovery_failed"],
            "confidence": None,
            "provider": llm.cfg.provider,
            "model": llm.cfg.model,
            "reasoning_steps": [],
            "react_trace": react_result.get("trace", {}),
        }

    final = react_result.get("final_answer", {})
    ret = []
    for item in final.get("discovered_sources", []):
        if not isinstance(item, dict):
            continue
        sid = _normalize_id(item.get("id", ""))
        if not sid:
            continue
        ret.append(
            {
                "id": sid,
                "display_name": _normalize_text(item.get("display_name", sid)),
                "entities": _to_str_list(item.get("entities", [])),
                "relations": _to_str_list(item.get("relations", [])),
                "homepage": _normalize_text(item.get("homepage", "")),
                "api_docs": _normalize_text(item.get("api_docs", "")),
                "available": bool(item.get("available", True)),
                "quality_weight": float(item.get("quality_weight", 0.7)),
                "mandatory_when": [],
                "recommended_when": [],
                "agent_reason": _normalize_text(item.get("reason", "")),
            }
        )
        if len(ret) >= max_new_sources:
            break

    return {
        "mode": "agent_discovery",
        "discovered_sources": ret,
        "notes": final.get("notes", []),
        "confidence": react_result.get("confidence"),
        "self_critique": react_result.get("self_critique", ""),
        "reasoning_steps": react_result.get("reasoning_steps", []),
        "provider": llm.cfg.provider,
        "model": llm.cfg.model,
        "react_trace": react_result.get("trace", {}),
    }


def merge_registry_and_discovery(registry_sources: List[Dict], discovery_result: Dict) -> List[Dict]:
    merged: Dict[str, Dict] = {}
    for src in registry_sources:
        sid = _normalize_id(src.get("id", ""))
        if not sid:
            continue
        normalized = dict(src)
        normalized["id"] = sid
        merged[sid] = normalized

    for src in discovery_result.get("discovered_sources", []) if isinstance(discovery_result, dict) else []:
        sid = _normalize_id(src.get("id", ""))
        if not sid:
            continue
        if sid in merged:
            base = dict(merged[sid])
            for k, v in src.items():
                if v not in (None, "", [], {}):
                    base[k] = v
            base["available"] = bool(src.get("available", True))
            base["discovered_by_agent"] = True
            merged[sid] = base
        else:
            new_src = dict(src)
            new_src.setdefault("quality_weight", 0.7)
            new_src.setdefault("mandatory_when", [])
            new_src.setdefault("recommended_when", [])
            new_src["available"] = bool(src.get("available", True))
            new_src["discovered_by_agent"] = True
            merged[sid] = new_src
    return list(merged.values())


# ---------------------------------------------------------------------------
# Selection agent (ReAct with iterative self-correction)
# ---------------------------------------------------------------------------

def run_selection_agent(
    task: TaskSpec,
    candidates: List[SourceCandidate],
    min_sources: int = 4,
    max_sources: int = 4,
    llm_client: Optional[BaseLLMClient] = None,
    include_mandatory: bool = True,
    validation_feedback: Optional[Dict] = None,
    performance_feedback: Optional[Dict] = None,
    previous_selection: Optional[List[str]] = None,
    max_react_iterations: int = 2,
    data_profile: Optional[Dict] = None,
) -> Optional[Dict]:
    """
    ReAct-style selection agent.

    If validation_feedback is provided (from a failed KB build), the agent
    performs self-correction reasoning to fix the selection.

    max_react_iterations: how many correction rounds to attempt if the
    agent's own self-critique suggests low confidence.
    """
    llm = llm_client or create_llm_client_from_env()
    if llm is None:
        _debug_log(
            "selection_agent_unavailable: set KNOWLEDGE_AGENT_ENABLE_AGENT=1 "
            "and configure provider/endpoint/API key/model"
        )
        return _failure_result(
            stage="client_init",
            message="LLM client unavailable; check KNOWLEDGE_AGENT_ENABLE_AGENT/provider/API key/endpoint",
        )

    all_ids = {c.source_id for c in candidates}
    mandatory_ids = _rule_mandatory(candidates)
    min_sources = max(1, min(int(min_sources), int(max_sources)))
    max_sources = max(int(max_sources), min_sources)

    selection_agent_mode = str(
        os.getenv("KNOWLEDGE_AGENT_SELECTION_AGENT_MODE", "lightweight")
    ).strip().lower()
    if selection_agent_mode != "react_legacy":
        payload = {
            "task": {
                "task_id": task.task_id,
                "source_modalities": list(task.source_modalities),
                "target_modality": task.target_modality,
                "species": task.species,
                "required_relations": list(task.required_relations),
            },
            "data_profile": _data_profile_payload(data_profile),
            "candidate_sources": _compact_candidate_payload(candidates),
            "constraints": {
                "min_sources": min_sources,
                "max_sources": max_sources,
                "must_include_mandatory_sources": bool(include_mandatory),
                "allowed_source_ids": sorted(list(all_ids)),
                "mandatory_source_ids": list(mandatory_ids),
            },
            "selection_goal": (
                "Select external knowledge bases that best support downstream KB feature processing "
                "and model training for this task. Only decide which sources to use; local code will "
                "handle graph construction and feature processing. You may provide source_weights "
                "to guide weight adaptation."
            ),
        }
        if validation_feedback is not None:
            payload["validation_feedback"] = validation_feedback
            payload["previous_selection"] = previous_selection or []
            payload["selection_goal"] = (
                "Revise the previous source selection to satisfy validation feedback while still "
                "covering all required relations."
            )
        elif performance_feedback is not None:
            payload["performance_feedback"] = performance_feedback
            payload["previous_selection"] = previous_selection or []
            payload["selection_goal"] = (
                "Refine the previous source selection using downstream performance feedback relative "
                "to the formal rule baseline. First diagnose metric gains/losses using the supplied "
                "metric directions and refine_strategy. If round1 succeeded, preserve and possibly "
                "upweight successful added sources. If round1 underperformed, anchor to rule sources "
                "and keep at most one low-weight exploratory addition. Return source_weights, "
                "source_actions, and source_usage_modes for all selected sources."
            )

        parsed = llm.generate_json(_selection_system_prompt_lightweight(), payload)
        trace = getattr(llm, "last_call_trace", {}) or {}
        if not isinstance(parsed, dict):
            detail = {
                "provider": getattr(llm.cfg, "provider", ""),
                "model": getattr(llm.cfg, "model", ""),
                "ok": trace.get("ok"),
                "status": trace.get("status"),
                "error_type": trace.get("error_type"),
                "error": trace.get("error"),
                "error_body_snippet": trace.get("error_body_snippet"),
                "parse_error": trace.get("parse_error"),
                "finish_reason": trace.get("finish_reason"),
            }
            _debug_log(f"selection_agent_failed: {detail}")
            if str(os.getenv("KNOWLEDGE_AGENT_RAISE_LLM_ERRORS", "0")).strip().lower() in {"1", "true", "yes"}:
                raise RuntimeError(f"Knowledge selection LLM call failed: {detail}")
            return _failure_result(
                stage="llm_call",
                provider=getattr(llm.cfg, "provider", ""),
                model=getattr(llm.cfg, "model", ""),
                trace=trace,
                message=str(detail),
            )

        selected = _to_str_list(parsed.get("selected_source_ids", []))
        selected = [sid for sid in selected if sid in all_ids]
        selected = _dedupe_keep_order((mandatory_ids if include_mandatory else []) + selected)
        if len(selected) < min_sources:
            remaining = [
                c.source_id for c in candidates
                if c.available and c.source_id not in selected
            ]
            selected.extend(remaining[: max(0, min_sources - len(selected))])
        selected = selected[:max_sources]

        if not selected:
            _debug_log(
                "selection_agent_empty_after_validation: "
                f"raw_selected={parsed.get('selected_source_ids', [])}, "
                f"allowed={sorted(list(all_ids))}"
            )
            return _failure_result(
                stage="selection_validation",
                provider=getattr(llm.cfg, "provider", ""),
                model=getattr(llm.cfg, "model", ""),
                trace=trace,
                message=f"raw_selected={parsed.get('selected_source_ids', [])}",
            )

        source_weights, weight_strategy = _postprocess_source_weights(
            selected=selected,
            raw_weights=parsed.get("source_weights", {}),
            candidates=candidates,
        )
        return {
            "ok": True,
            "selected_source_ids": selected,
            "source_weights": source_weights,
            "source_weight_strategy": weight_strategy,
            "download_candidates": [],
            "processing_strategy": {},
            "decision_summary": parsed.get("decision_summary", ""),
            "round1_outcome": parsed.get("round1_outcome", ""),
            "refine_strategy": parsed.get("refine_strategy", ""),
            "metric_focus": parsed.get("metric_focus", []),
            "source_actions": parsed.get("source_actions", {}),
            "source_usage_modes": parsed.get("source_usage_modes", {}),
            "source_rationales": parsed.get("source_rationales", {}),
            "exclusion_rationales": {},
            "expected_kb_effects": {},
            "risk_controls": [],
            "notes": [],
            "confidence": parsed.get("confidence", 0.0),
            "self_critique": "",
            "reasoning_steps": [],
            "correction_analysis": "",
            "react_iterations": 1,
            "tool_calls": [],
            "tool_observations": [],
            "provider": llm.cfg.provider,
            "model": llm.cfg.model,
            "llm_call_traces": [trace] if str(
                os.getenv("KNOWLEDGE_AGENT_SAVE_LLM_TRACE", "0")
            ).strip().lower() in {"1", "true", "yes"} else [],
            "llm_call_trace": trace if str(
                os.getenv("KNOWLEDGE_AGENT_SAVE_LLM_TRACE", "0")
            ).strip().lower() in {"1", "true", "yes"} else {},
        }

    # Build base payload
    base_payload = {
        "task": task.to_dict(),
        "data_profile": _data_profile_payload(data_profile),
        "candidate_sources": _candidate_payload(candidates),
        "constraints": {
            "min_sources": min_sources,
            "max_sources": max_sources,
            "must_include_mandatory_sources": bool(include_mandatory),
            "allowed_source_ids": sorted(list(all_ids)),
            "relation_coverage_is_required": True,
            "data_coverage_is_tie_breaker": True,
        },
        "selection_policy": [
            "First cover every task.required_relations using available sources.",
            "Then prefer higher data_coverage_score among sources with comparable relation coverage.",
            "Do not replace a required-relation source with a source that has only data overlap.",
            "If protein features look like antibody or marker names, prefer sources with marker/protein alias support.",
            "Select between min_sources and max_sources sources adaptively.",
            "Prefer 4-6 sources total unless the task or coverage clearly requires fewer or more.",
        ],
        "required_output_schema": {
            "reasoning_steps": [
                {"step": 1, "thought": "string", "action": "string", "observation": "string"}
            ],
            "final_answer": {
                "selected_source_ids": ["string"],
                "source_weights": {"source_id": 1.0},
                "source_actions": {"source_id": "keep"},
                "source_usage_modes": {"source_id": "full"},
                "download_candidates": [{"source_id": "string", "reason": "string"}],
                "processing_strategy": {
                    "cross_modal_prior": "string",
                    "intra_modal_graph": "string",
                    "normalization": "string",
                },
                "decision_summary": "short string",
                "round1_outcome": "success|partial_success|underperformed",
                "refine_strategy": "exploit_success|hybrid_refine|hard_recover",
                "metric_focus": ["PCC"],
                "source_rationales": {"source_id": "short reason"},
                "exclusion_rationales": {"source_id": "short reason"},
                "expected_kb_effects": {
                    "cross_modal_links": "string",
                    "gene_graph": "string",
                    "protein_graph": "string",
                    "celltype_prior": "string",
                },
                "risk_controls": ["string"],
                "notes": ["string"],
                "confidence": 0.0,
            },
            "confidence": 0.0,
            "self_critique": "string",
        },
    }
    if performance_feedback is not None:
        base_payload["performance_feedback"] = performance_feedback
        base_payload["selection_policy"].extend([
            "For performance feedback, compare round1_free_agent with the formal rule baseline before selecting.",
            "PCC and SSIM are higher-is-better; CMD and RMSE are lower-is-better.",
            "Follow performance_feedback.refine_strategy exactly unless it violates allowed_source_ids.",
            "Use source_weights and source_usage_modes to downweight uncertain retained probes instead of broad source churn.",
        ])

    all_reasoning_steps = []
    all_traces = []

    # --- Round 1: initial selection or correction ---
    if validation_feedback is not None:
        # Self-correction mode: agent knows previous selection failed
        payload = dict(base_payload)
        payload["validation_feedback"] = validation_feedback
        payload["previous_selection"] = previous_selection or []
        payload["instruction"] = (
            "Your previous selection failed KB validation. "
            "Analyze the errors and select different/additional sources to fix them."
        )
        system_prompt = _correction_system_prompt()
        step_name = "selection_correction_round1"
    elif performance_feedback is not None:
        payload = dict(base_payload)
        payload["previous_selection"] = previous_selection or []
        payload["instruction"] = (
            "You are in metric-aware refinement mode. Diagnose round1 relative to the formal rule "
            "baseline, then output a second-stage source set and source_weights that follow the "
            "provided refine_strategy."
        )
        system_prompt = _correction_system_prompt()
        step_name = "selection_metric_refinement_round1"
    else:
        payload = base_payload
        system_prompt = _react_system_prompt()
        step_name = "selection_react_round1"

    react_result = _run_react_call_with_tools(llm, system_prompt, payload, step_name)
    all_traces.append(react_result.get("trace", {}))
    all_reasoning_steps.extend(react_result.get("reasoning_steps", []))

    if not react_result.get("ok"):
        trace = getattr(llm, "last_call_trace", {}) or {}
        detail = {
            "provider": getattr(llm.cfg, "provider", ""),
            "model": getattr(llm.cfg, "model", ""),
            "ok": trace.get("ok"),
            "status": trace.get("status"),
            "error_type": trace.get("error_type"),
            "error": trace.get("error"),
            "error_body_snippet": trace.get("error_body_snippet"),
        }
        _debug_log(f"selection_agent_failed: {detail}")
        if str(os.getenv("KNOWLEDGE_AGENT_RAISE_LLM_ERRORS", "0")).strip().lower() in {"1", "true", "yes"}:
            raise RuntimeError(f"Knowledge selection LLM call failed: {detail}")
        return _failure_result(
            stage="llm_call",
            provider=getattr(llm.cfg, "provider", ""),
            model=getattr(llm.cfg, "model", ""),
            trace=trace,
            message=str(detail),
        )

    final = react_result.get("final_answer", {})
    confidence = react_result.get("confidence") or final.get("confidence", 0.0)
    self_critique = react_result.get("self_critique", "")

    # --- Optional round 2: agent self-critique triggered re-reasoning ---
    confidence_threshold = float(os.getenv("KNOWLEDGE_AGENT_REACT_CONFIDENCE_THRESHOLD", "0.6"))
    if (
        max_react_iterations >= 2
        and isinstance(confidence, (int, float))
        and float(confidence) < confidence_threshold
        and validation_feedback is None  # don't double-loop on correction
    ):
        _debug_log(
            f"react_low_confidence={confidence:.2f} < {confidence_threshold}, "
            "triggering self-correction round"
        )
        correction_payload = dict(base_payload)
        correction_payload["previous_reasoning"] = all_reasoning_steps
        correction_payload["previous_confidence"] = confidence
        correction_payload["self_critique_from_round1"] = self_critique
        correction_payload["instruction"] = (
            "Your confidence was low. Re-examine your reasoning, "
            "reconsider source trade-offs, and produce a more confident selection."
        )
        react_result2 = _run_react_call_with_tools(
            llm, _correction_system_prompt(), correction_payload, "selection_self_correction_round2"
        )
        all_traces.append(react_result2.get("trace", {}))
        all_reasoning_steps.extend(react_result2.get("reasoning_steps", []))

        if react_result2.get("ok"):
            final2 = react_result2.get("final_answer", {})
            conf2 = react_result2.get("confidence") or final2.get("confidence", 0.0)
            if isinstance(conf2, (int, float)) and float(conf2) > float(confidence):
                _debug_log(f"react_round2_improved_confidence: {confidence:.2f} -> {conf2:.2f}")
                final = final2
                confidence = conf2
                self_critique = react_result2.get("self_critique", "")

    # --- Validate and clean selected IDs ---
    selected = _to_str_list(final.get("selected_source_ids", []))
    selected = [sid for sid in selected if sid in all_ids]
    selected = _dedupe_keep_order((mandatory_ids if include_mandatory else []) + selected)
    if len(selected) < min_sources:
        remaining = [
            c.source_id for c in candidates
            if c.available and c.source_id not in selected
        ]
        selected.extend(remaining[: max(0, min_sources - len(selected))])
    selected = selected[:max_sources]

    if not selected:
        _debug_log(
            "selection_agent_empty_after_validation: "
            f"raw_selected={final.get('selected_source_ids', [])}, "
            f"allowed={sorted(list(all_ids))}"
        )
        return _failure_result(
            stage="selection_validation",
            provider=getattr(llm.cfg, "provider", ""),
            model=getattr(llm.cfg, "model", ""),
            trace=all_traces[-1] if all_traces else {},
            message=f"raw_selected={final.get('selected_source_ids', [])}",
        )

    source_weights, weight_strategy = _postprocess_source_weights(
        selected=selected,
        raw_weights=final.get("source_weights", {}),
        candidates=candidates,
    )

    return {
        "ok": True,
        "selected_source_ids": selected,
        "source_weights": source_weights,
        "source_weight_strategy": weight_strategy,
        "download_candidates": final.get("download_candidates", []),
        "processing_strategy": final.get("processing_strategy", {}),
        "decision_summary": final.get("decision_summary", ""),
        "round1_outcome": final.get("round1_outcome", ""),
        "refine_strategy": final.get("refine_strategy", ""),
        "metric_focus": final.get("metric_focus", []),
        "source_actions": final.get("source_actions", {}),
        "source_usage_modes": final.get("source_usage_modes", {}),
        "source_rationales": final.get("source_rationales", {}),
        "exclusion_rationales": final.get("exclusion_rationales", {}),
        "expected_kb_effects": final.get("expected_kb_effects", {}),
        "risk_controls": final.get("risk_controls", []),
        "notes": final.get("notes", []),
        "confidence": confidence,
        "self_critique": self_critique,
        "reasoning_steps": all_reasoning_steps,
        "correction_analysis": react_result.get("correction_analysis", ""),
        "react_iterations": len(all_traces),
        # Tool calling fields (Layer 2)
        "tool_calls": react_result.get("tool_calls", []),
        "tool_observations": react_result.get("observations", []),
        "provider": llm.cfg.provider,
        "model": llm.cfg.model,
        "llm_call_traces": all_traces if str(
            os.getenv("KNOWLEDGE_AGENT_SAVE_LLM_TRACE", "0")
        ).strip().lower() in {"1", "true", "yes"} else [],
        "llm_call_trace": all_traces[0] if all_traces and str(
            os.getenv("KNOWLEDGE_AGENT_SAVE_LLM_TRACE", "0")
        ).strip().lower() in {"1", "true", "yes"} else {},
    }
