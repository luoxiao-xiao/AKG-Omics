#!/usr/bin/env python
"""
Benchmark LLM knowledge-orchestration decisions across multiple models.

Runs proteogenomics + metabolomics tasks through the knowledge-source selector,
records LLM validity/latency/coverage/stability/reasoning signals, and writes
CSV/JSON artifacts for cross-model comparison.

Supported backbones (via env):
  KNOWLEDGE_AGENT_MULTI_MODEL_PROVIDERS=gemini:gemini-2.0-flash,deepseek:deepseek-chat,
    openai:gpt-4o-mini,anthropic:claude-3-5-sonnet-20241022,qwen:qwen-max,glm:glm-4

Single-model mode (legacy): set KNOWLEDGE_AGENT_LLM_PROVIDER + KNOWLEDGE_AGENT_LLM_MODEL
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence
import importlib

try:
    from .knowledge_agent.agent import merge_registry_and_discovery, run_discovery_agent
    from .knowledge_agent.data_profiler import build_data_profile
    from .knowledge_agent.llm_client import (
        BaseLLMClient,
        MultiModelClient,
        create_llm_client_from_env,
        create_multi_model_client_from_env,
    )
    from .knowledge_agent.registry import load_registry
    from .knowledge_agent.selector import select_sources
    from .knowledge_agent.task_schema import TaskSpec
except ImportError:
    from knowledge_agent.agent import merge_registry_and_discovery, run_discovery_agent
    from knowledge_agent.data_profiler import build_data_profile
    from knowledge_agent.llm_client import (
        BaseLLMClient,
        MultiModelClient,
        create_llm_client_from_env,
        create_multi_model_client_from_env,
    )
    from knowledge_agent.registry import load_registry
    from knowledge_agent.selector import select_sources
    from knowledge_agent.task_schema import TaskSpec


# ---------------------------------------------------------------------------
# Task definitions (proteogenomics + metabolomics)
# ---------------------------------------------------------------------------

TASKS = [
    # Proteogenomics
    ("task1_he_to_gene",          ["he"],           "gene"),
    ("task2_he_to_protein",       ["he"],           "protein"),
    ("task3_he_protein_to_gene",  ["he", "protein"], "gene"),
    ("task4_he_gene_to_protein",  ["he", "gene"],   "protein"),
    # Metabolomics (new)
    ("task5_he_to_metabolism",    ["he"],           "metabolism"),
    ("task6_he_gene_to_metabolism", ["he", "gene"], "metabolism"),
    ("task7_he_metabolism_to_gene", ["he", "metabolism"], "gene"),
]

MODE_ALIASES = {
    "agent-only": ("agent_only", False),
    "agent_only": ("agent_only", False),
    "agent-rule": ("agent_rule", False),
    "agent_rule": ("agent_rule", False),
    "agent-data": ("agent_only", True),
    "agent_data": ("agent_only", True),
    "agent-data-rule": ("agent_rule", True),
    "agent_data_rule": ("agent_rule", True),
    "rule-only": ("rule_only", False),
    "rule_only": ("rule_only", False),
    "rule-data": ("rule_only", True),
    "rule_data": ("rule_only", True),
}


def _required_relations(source_modalities: List[str], target: str) -> List[str]:
    rels = ["pathway_membership", "celltype_marker"]
    if target == "gene":
        rels.append("protein_gene_mapping")
    elif target == "protein":
        rels.append("gene_protein_mapping")
    elif target == "metabolism":
        rels.extend(["gene_metabolism_association", "metabolite_pathway"])
    if "metabolism" in source_modalities:
        rels.append("metabolism_gene_association")
    return list(dict.fromkeys(rels))  # dedupe preserving order


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _json_compact(x: Any) -> str:
    if x in (None, "", [], {}):
        return ""
    try:
        return json.dumps(x, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    except Exception:
        return str(x)


def _semicolon(xs: Any) -> str:
    if isinstance(xs, str):
        return xs
    if isinstance(xs, Iterable) and not isinstance(xs, (dict, bytes)):
        return ";".join(str(x) for x in xs)
    return "" if xs is None else str(xs)


def _read_name_list(path: str) -> List[str]:
    path = str(path or "").strip()
    if not path:
        return []
    names: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            for tok in line.replace(",", "\t").split("\t"):
                tok = tok.strip()
                if tok:
                    names.append(tok)
    return list(dict.fromkeys(names))


def _split_names(value: str) -> List[str]:
    if not value:
        return []
    out = []
    for tok in str(value).replace(",", "\t").split("\t"):
        tok = tok.strip()
        if tok:
            out.append(tok)
    return list(dict.fromkeys(out))


def _kb_paths_from_env(args: argparse.Namespace) -> Dict[str, str]:
    kb_root = args.proteo_kb_dir or os.getenv("PROTEO_KB_DIR", "data/KB")
    met_root = args.metabolite_kb_dir or os.getenv("KB_ROOT", "data/KB/metabolite")
    return {
        "hgnc_file": os.getenv("PROTEO_HGNC_PATH", os.path.join(kb_root, "hgnc_complete_set.txt")),
        "uniprot_file": os.getenv("PROTEO_UNIPROT_PATH", os.path.join(kb_root, "uniprot_human_reviewed.tsv")),
        "reactome_uniprot_file": os.getenv("PROTEO_REACTOME_UNIPROT_PATH", os.path.join(kb_root, "UniProt2Reactome.txt")),
        "reactome_ensembl_file": os.getenv("PROTEO_REACTOME_ENSEMBL_PATH", os.path.join(kb_root, "Ensembl2Reactome.txt")),
        "cellmarker_file": os.getenv("PROTEO_CELLMARKER_PATH", os.path.join(kb_root, "Cell_marker_Human.xlsx")),
        "dorothea_file": os.getenv("PROTEO_DOROTHEA_PATH", os.path.join(kb_root, "dorothea_grn.tsv")),
        "omnipath_file": os.getenv("PROTEO_OMNIPATH_PATH", os.path.join(kb_root, "omnipath_interactions.tsv")),
        "corum_file": os.getenv("PROTEO_CORUM_PATH", os.path.join(kb_root, "corum_allComplexes.txt")),
        "proteinatlas_file": os.getenv("PROTEO_PROTEINATLAS_PATH", os.path.join(kb_root, "proteinatlas.tsv")),
        "kegg_pathways_file": os.getenv("PROTEO_KEGG_PATHWAYS_PATH", os.path.join(kb_root, "kegg_hsa_pathways.txt")),
        "kegg_gene_pathway_file": os.getenv("PROTEO_KEGG_GENE_PATHWAY_PATH", os.path.join(kb_root, "kegg_hsa_gene_pathway_links.txt")),
        "string_file": os.getenv("PROTEO_STRING_PATH", os.path.join(kb_root, "9606.protein.links.v12.0.txt.gz")),
        "hmdb_file": os.getenv("HMDB_XML_PATH", os.path.join(met_root, "hmdb", "hmdb_metabolites.xml")),
    }


def _ensure_project_root_importable() -> None:
    """Prefer the project root over akg_omics/ when this file is run as a script."""
    project_root = Path(__file__).resolve().parents[1]
    root_str = str(project_root)
    if root_str in sys.path:
        sys.path.remove(root_str)
    sys.path.insert(0, root_str)

    loaded = sys.modules.get("akg_omics")
    if loaded is not None and not hasattr(loaded, "__path__"):
        del sys.modules["akg_omics"]


def _load_run_pipeline():
    _ensure_project_root_importable()
    try:
        return importlib.import_module("akg_omics.run")
    except Exception as e:
        raise RuntimeError(
            "Could not import akg_omics.run. Make sure you run from the project root, "
            "set PYTHONPATH to the project root, and install run.py dependencies "
            f"(for example scanpy/anndata). Original error: {type(e).__name__}: {e}"
        )


def _run_data_profiles_for_tasks(active_tasks: Sequence, kb_paths: Dict[str, str]) -> Dict[str, Dict[str, Any]]:
    pipeline = _load_run_pipeline()
    profiles: Dict[str, Dict[str, Any]] = {}

    needs_proteo = any("metabolism" not in task_id for task_id, _, _ in active_tasks)
    needs_met = any("metabolism" in task_id for task_id, _, _ in active_tasks)

    if needs_proteo:
        if not hasattr(pipeline, "prepare_proteogenomics_data"):
            raise RuntimeError("akg_omics.run does not expose prepare_proteogenomics_data().")
        print("[BENCH] preparing run.py proteogenomics data for DataProfiler ...", flush=True)
        pack = pipeline.prepare_proteogenomics_data()
        gene_names = list(getattr(pack["adata2"], "var_names", []))
        protein_names = list(getattr(pack["adata1"], "var_names", []))
        profiles["proteo"] = build_data_profile(
            task_spec=TaskSpec("run_data_proteo_profile", ["he", "protein"], "gene", species="human"),
            gene_names=gene_names,
            protein_names=protein_names,
            kb_paths=kb_paths,
        )
        print(
            f"[BENCH] proteo profile: genes={len(gene_names)} proteins={len(protein_names)}",
            flush=True,
        )

    if needs_met:
        if not hasattr(pipeline, "prepare_aligned_data"):
            raise RuntimeError("akg_omics.run does not expose prepare_aligned_data().")
        print("[BENCH] preparing run.py metabolomics data for DataProfiler ...", flush=True)
        pack = pipeline.prepare_aligned_data()
        gene_names = list(getattr(pack["C1_rna"], "var_names", []))
        metabolite_names = list(getattr(pack["B1_met"], "var_names", []))
        profiles["metabolomics"] = build_data_profile(
            task_spec=TaskSpec("run_data_metabolomics_profile", ["he", "metabolism"], "gene", species="mouse"),
            gene_names=gene_names,
            metabolite_names=metabolite_names,
            kb_paths=kb_paths,
        )
        print(
            f"[BENCH] metabolomics profile: genes={len(gene_names)} metabolites={len(metabolite_names)}",
            flush=True,
        )

    return profiles


def _resolve_mode(label: str) -> Dict[str, Any]:
    key = str(label).strip().lower()
    if key not in MODE_ALIASES:
        raise ValueError(f"Unsupported mode '{label}'. Use one of: {', '.join(sorted(MODE_ALIASES))}")
    selection_mode, use_data = MODE_ALIASES[key]
    return {
        "requested_mode_label": label,
        "selection_mode": selection_mode,
        "use_data_profile": use_data,
    }


def _selected_data_coverage(selected: Sequence[str], selection_dict: Dict[str, Any]) -> Dict[str, Any]:
    lookup = _candidate_lookup(selection_dict)
    vals = []
    available = 0
    for sid in selected:
        c = lookup.get(str(sid).strip().lower(), {})
        score = c.get("data_coverage_score", None)
        if score is None or score == "":
            continue
        try:
            vals.append(float(score))
            available += 1
        except Exception:
            continue
    return {
        "mean_selected_data_coverage": round(sum(vals) / len(vals), 4) if vals else 0.0,
        "selected_data_coverage_available_fraction": round(available / max(len(selected), 1), 4) if selected else 0.0,
    }


def _mean_pairwise_jaccard(source_sets: Sequence[Sequence[str]]) -> float:
    sets = [set(str(x).strip().lower() for x in xs if str(x).strip()) for xs in source_sets]
    if len(sets) <= 1:
        return 1.0
    vals = []
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            union = sets[i] | sets[j]
            vals.append(len(sets[i] & sets[j]) / max(len(union), 1))
    return round(sum(vals) / len(vals), 4) if vals else 1.0


def _source_map(registry_sources: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {str(s.get("id", "")).strip().lower(): s for s in registry_sources if s.get("id")}


def _relation_coverage(selected: Sequence[str], registry_sources: Sequence[Dict[str, Any]], required: Sequence[str]) -> float:
    if not required:
        return 1.0
    smap = _source_map(registry_sources)
    observed = set()
    for sid in selected:
        src = smap.get(str(sid).strip().lower(), {})
        observed.update(str(r).strip().lower() for r in src.get("relations", []) if str(r).strip())
    covered = sum(1 for r in required if str(r).strip().lower() in observed)
    return float(covered) / float(len(required))


def _candidate_lookup(selection_dict: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    out = {}
    for c in selection_dict.get("candidates", []) if isinstance(selection_dict, dict) else []:
        if isinstance(c, dict):
            sid = str(c.get("source_id", "")).strip().lower()
            if sid:
                out[sid] = c
    return out


def _selected_score(selected: Sequence[str], selection_dict: Dict[str, Any]) -> float:
    lookup = _candidate_lookup(selection_dict)
    vals = [float(lookup[str(sid).strip().lower()].get("score", 0.0))
            for sid in selected if str(sid).strip().lower() in lookup]
    return float(sum(vals) / len(vals)) if vals else 0.0


def _mandatory_fraction(selected: Sequence[str], selection_dict: Dict[str, Any]) -> float:
    lookup = _candidate_lookup(selection_dict)
    mandatory = [sid for sid, c in lookup.items() if bool(c.get("is_mandatory"))]
    if not mandatory:
        return 1.0
    selected_set = {str(x).strip().lower() for x in selected}
    return float(sum(1 for sid in mandatory if sid in selected_set)) / float(len(mandatory))


def _reasoning_steps_summary(agent_decision: Dict) -> str:
    """Compact summary of ReAct reasoning steps for CSV."""
    steps = agent_decision.get("reasoning_steps", [])
    if not steps:
        return ""
    parts = []
    for s in steps[:5]:  # cap at 5 steps for CSV readability
        if isinstance(s, dict):
            thought = str(s.get("thought", ""))[:80]
            action = str(s.get("action", ""))[:60]
            parts.append(f"[{s.get('step', '?')}] T:{thought} A:{action}")
    return " | ".join(parts)


# ---------------------------------------------------------------------------
# Single-model run
# ---------------------------------------------------------------------------

def run_once(
    task: TaskSpec,
    registry_sources: List[Dict[str, Any]],
    mode: Dict[str, Any],
    max_sources: int,
    llm_client: Optional[BaseLLMClient] = None,
    model_name: str = "",
    data_profile: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    t0 = time.time()
    selection_mode = mode["selection_mode"] if isinstance(mode, dict) else str(mode)
    mode_label = mode.get("requested_mode_label", selection_mode) if isinstance(mode, dict) else str(mode)
    use_data_profile = bool(mode.get("use_data_profile", False)) if isinstance(mode, dict) else False
    effective_profile = data_profile if use_data_profile else None

    discovery = run_discovery_agent(
        task=task,
        registry_sources=registry_sources,
        llm_client=llm_client,
        data_profile=effective_profile,
    )
    merged = merge_registry_and_discovery(registry_sources, discovery)
    try:
        selection = select_sources(
            task=task,
            registry_sources=merged,
            max_sources=max_sources,
            selection_mode=selection_mode,
            data_profile=effective_profile,
        )
        error = ""
    except Exception as e:
        selection = None
        error = f"{type(e).__name__}: {e}"
    elapsed = time.time() - t0

    selection_dict = selection.to_dict() if selection is not None else {}
    agent_decision = selection_dict.get("agent_decision", {}) if isinstance(selection_dict.get("agent_decision", {}), dict) else {}
    selected = selection_dict.get("selected_source_ids", []) if isinstance(selection_dict, dict) else []
    required = task.required_relations
    llm_trace = agent_decision.get("llm_call_trace", {}) if isinstance(agent_decision, dict) else {}
    raw_ok = bool(llm_trace.get("ok")) if llm_trace else bool(agent_decision)
    expected_agent_mode = selection_mode in {"agent_only", "agent_rule"}
    mode_actual = str(selection_dict.get("mode", "failed"))
    agent_success = (not expected_agent_mode) or mode_actual in {"agent_only", "agent+rule"}
    data_cov = _selected_data_coverage(selected, selection_dict)
    mandatory_fraction = round(_mandatory_fraction(selected, selection_dict), 4)
    relation_cov = round(_relation_coverage(selected, merged, required), 4)
    candidate_score = round(_selected_score(selected, selection_dict), 4)
    decision_quality_score = round(
        0.25 * float(bool(agent_success))
        + 0.15 * float(bool(raw_ok))
        + 0.20 * mandatory_fraction
        + 0.20 * relation_cov
        + 0.10 * max(0.0, min(1.0, candidate_score))
        + 0.10 * data_cov["mean_selected_data_coverage"],
        4,
    )

    # ReAct-specific fields
    react_iterations = agent_decision.get("react_iterations", 0) if isinstance(agent_decision, dict) else 0
    confidence = agent_decision.get("confidence", "") if isinstance(agent_decision, dict) else ""
    self_critique = agent_decision.get("self_critique", "") if isinstance(agent_decision, dict) else ""
    reasoning_summary = _reasoning_steps_summary(agent_decision) if isinstance(agent_decision, dict) else ""
    discovery_reasoning = _reasoning_steps_summary(discovery) if isinstance(discovery, dict) else ""

    provider = agent_decision.get("provider", os.getenv("KNOWLEDGE_AGENT_LLM_PROVIDER", "")) if isinstance(agent_decision, dict) else ""
    model = agent_decision.get("model", os.getenv("KNOWLEDGE_AGENT_LLM_MODEL", "")) if isinstance(agent_decision, dict) else ""
    backbone = model_name or f"{provider}:{model}"

    return {
        "task": task.task_id,
        "source_modalities": _semicolon(task.source_modalities),
        "target_modality": task.target_modality,
        "required_relations": _semicolon(required),
        "requested_mode": mode_label,
        "selection_mode": selection_mode,
        "use_data_profile": use_data_profile,
        "actual_mode": mode_actual,
        "agent_success": agent_success,
        "json_success": raw_ok,
        "latency_sec": round(elapsed, 3),
        "provider": provider,
        "model": model,
        "llm_backbone": backbone,
        "confidence": confidence,
        "self_critique": str(self_critique)[:200],
        "react_iterations": react_iterations,
        "reasoning_summary": reasoning_summary[:300],
        "discovery_reasoning_summary": discovery_reasoning[:200],
        "selected_count": len(selected),
        "selected_sources": _semicolon(selected),
        "mandatory_fraction": mandatory_fraction,
        "required_relation_coverage": relation_cov,
        "mean_candidate_score": candidate_score,
        "mean_selected_data_coverage": data_cov["mean_selected_data_coverage"],
        "selected_data_coverage_available_fraction": data_cov["selected_data_coverage_available_fraction"],
        "decision_quality_score": decision_quality_score,
        "data_profile_warnings": _json_compact((effective_profile or {}).get("warnings", [])),
        "source_weights": _json_compact(agent_decision.get("source_weights", {}) if isinstance(agent_decision, dict) else {}),
        "decision_summary": agent_decision.get("decision_summary", "") if isinstance(agent_decision, dict) else "",
        "source_rationales": _json_compact(agent_decision.get("source_rationales", {}) if isinstance(agent_decision, dict) else {}),
        "exclusion_rationales": _json_compact(agent_decision.get("exclusion_rationales", {}) if isinstance(agent_decision, dict) else {}),
        "expected_kb_effects": _json_compact(agent_decision.get("expected_kb_effects", {}) if isinstance(agent_decision, dict) else {}),
        "risk_controls": _json_compact(agent_decision.get("risk_controls", []) if isinstance(agent_decision, dict) else []),
        "error": error or (llm_trace.get("error", "") if isinstance(llm_trace, dict) else ""),
        "trace": {
            "task_spec": task.to_dict(),
            "mode": mode,
            "data_profile": effective_profile,
            "discovery": discovery,
            "selection": selection_dict,
            "elapsed_sec": elapsed,
            "error": error,
        },
    }


# ---------------------------------------------------------------------------
# Multi-model run
# ---------------------------------------------------------------------------

def run_once_multi_model(
    task: TaskSpec,
    registry_sources: List[Dict[str, Any]],
    mode: Dict[str, Any],
    max_sources: int,
    multi_client: MultiModelClient,
    data_profile: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Run the same task through all models in multi_client, return one row per model."""
    rows = []
    for name, client in multi_client.clients.items():
        row = run_once(
            task=task,
            registry_sources=registry_sources,
            mode=mode,
            max_sources=max_sources,
            llm_client=client,
            model_name=name,
            data_profile=data_profile,
        )
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark LLM knowledge-agent source selection (multi-model, with metabolomics tasks)."
    )
    parser.add_argument("--modes", default="rule-only,agent-only,agent-rule,agent-data,agent-data-rule",
                        help="comma-separated modes: rule-only,agent-only,agent-rule,agent-data,agent-data-rule")
    parser.add_argument("--seeds", default="1,11,111",
                        help="logical repeats for stability")
    parser.add_argument("--max-sources", type=int,
                        default=int(os.getenv("PROTEO_KB_AGENT_MAX_SOURCES", "6")))
    parser.add_argument("--registry", default=None)
    parser.add_argument("--out-dir", default="./results/llm_agent_decision_benchmark")
    parser.add_argument("--tasks", default="all",
                        help="comma-separated task IDs or 'all' or 'proteo' or 'metabolomics'")
    parser.add_argument("--multi-model", action="store_true",
                        help="Use KNOWLEDGE_AGENT_MULTI_MODEL_PROVIDERS for parallel multi-model comparison")
    parser.add_argument("--disable-tools", action="store_true", default=True,
                        help="Disable agent tool calls. Enabled by default for decision-only validation.")
    parser.add_argument("--enable-tools", action="store_true",
                        help="Allow tool calls. Overrides --disable-tools.")
    parser.add_argument("--gene-names", default="", help="Comma/tab-separated gene feature names for data-aware modes.")
    parser.add_argument("--protein-names", default="", help="Comma/tab-separated protein feature names for data-aware modes.")
    parser.add_argument("--metabolite-names", default="", help="Comma/tab-separated metabolite feature names for data-aware modes.")
    parser.add_argument("--gene-names-file", default="", help="Text/TSV/CSV file containing gene feature names.")
    parser.add_argument("--protein-names-file", default="", help="Text/TSV/CSV file containing protein feature names.")
    parser.add_argument("--metabolite-names-file", default="", help="Text/TSV/CSV file containing metabolite feature names.")
    parser.add_argument("--data-profile-json", default="", help="Precomputed data profile JSON to use for all data-aware modes.")
    parser.add_argument("--use-run-data", action="store_true",
                        help="Load feature names from akg_omics.run data preparation functions for DataProfiler.")
    parser.add_argument("--proteo-kb-dir", default="", help="Proteogenomics KB root used for coverage profiling.")
    parser.add_argument("--metabolite-kb-dir", default="", help="Metabolomics KB root used for coverage profiling.")
    args = parser.parse_args()

    modes = [_resolve_mode(x.strip()) for x in args.modes.split(",") if x.strip()]
    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    os.environ["KNOWLEDGE_AGENT_ENABLE_TOOLS"] = "1" if args.enable_tools else "0"

    # Filter tasks
    task_filter = args.tasks.strip().lower()
    if task_filter == "all":
        active_tasks = TASKS
    elif task_filter == "proteo":
        active_tasks = [t for t in TASKS if "metabolism" not in t[0]]
    elif task_filter == "metabolomics":
        active_tasks = [t for t in TASKS if "metabolism" in t[0]]
    else:
        wanted = {x.strip() for x in task_filter.split(",")}
        active_tasks = [t for t in TASKS if t[0] in wanted]

    registry = load_registry(args.registry)
    kb_paths = _kb_paths_from_env(args)
    gene_names = _split_names(args.gene_names) + _read_name_list(args.gene_names_file)
    protein_names = _split_names(args.protein_names) + _read_name_list(args.protein_names_file)
    metabolite_names = _split_names(args.metabolite_names) + _read_name_list(args.metabolite_names_file)
    gene_names = list(dict.fromkeys(gene_names))
    protein_names = list(dict.fromkeys(protein_names))
    metabolite_names = list(dict.fromkeys(metabolite_names))

    precomputed_profile = None
    if args.data_profile_json:
        with open(args.data_profile_json, "r", encoding="utf-8") as f:
            precomputed_profile = json.load(f)
    run_data_profiles: Dict[str, Dict[str, Any]] = {}
    if args.use_run_data:
        run_data_profiles = _run_data_profiles_for_tasks(active_tasks, kb_paths=kb_paths)

    # Determine client mode
    multi_client: Optional[MultiModelClient] = None
    if args.multi_model:
        multi_client = create_multi_model_client_from_env()
        if multi_client is None:
            print("[BENCH] WARNING: --multi-model set but KNOWLEDGE_AGENT_MULTI_MODEL_PROVIDERS not configured. "
                  "Falling back to single-model mode.", flush=True)

    rows: List[Dict[str, Any]] = []
    traces: List[Dict[str, Any]] = []

    for mode in modes:
        for seed in seeds:
            for base_task, sources, target in active_tasks:
                task = TaskSpec(
                    task_id=f"{base_task}_probe_seed{seed}",
                    source_modalities=sources,
                    target_modality=target,
                    species="human",
                    required_relations=_required_relations(sources, target),
                )
                if precomputed_profile is not None:
                    data_profile = precomputed_profile
                elif args.use_run_data:
                    domain_key = "metabolomics" if "metabolism" in base_task else "proteo"
                    data_profile = run_data_profiles.get(domain_key)
                else:
                    data_profile = build_data_profile(
                        task_spec=task,
                        gene_names=gene_names,
                        protein_names=protein_names,
                        metabolite_names=metabolite_names,
                        kb_paths=kb_paths,
                    )

                if multi_client is not None:
                    batch = run_once_multi_model(
                        task=task,
                        registry_sources=registry,
                        mode=mode,
                        max_sources=args.max_sources,
                        multi_client=multi_client,
                        data_profile=data_profile,
                    )
                else:
                    batch = [run_once(
                        task=task,
                        registry_sources=registry,
                        mode=mode,
                        max_sources=args.max_sources,
                        data_profile=data_profile,
                    )]

                for row in batch:
                    row["seed"] = seed
                    traces.append(row.pop("trace"))
                    rows.append(row)
                    print(
                        f"[BENCH] mode={mode['requested_mode_label']} seed={seed} task={base_task} "
                        f"backbone={row['llm_backbone']} actual={row['actual_mode']} "
                        f"selected={row['selected_sources']} json={row['json_success']} "
                        f"data_cov={row['mean_selected_data_coverage']} "
                        f"react_iter={row['react_iterations']} conf={row['confidence']} "
                        f"latency={row['latency_sec']}",
                        flush=True,
                    )

    # Write detail CSV
    detail_path = out_dir / "llm_agent_decision_detail.csv"
    trace_path = out_dir / "llm_agent_decision_traces.json"
    if rows:
        with detail_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    with trace_path.open("w", encoding="utf-8") as f:
        json.dump(traces, f, ensure_ascii=False, indent=2)

    # Write summary CSV (grouped by backbone + mode)
    summary: Dict = {}
    for row in rows:
        key = (row["llm_backbone"], row["requested_mode"])
        summary.setdefault(key, []).append(row)

    summary_rows = []
    for (backbone, mode), group in summary.items():
        n = len(group)

        def _mean(field):
            vals = []
            for r in group:
                try:
                    vals.append(float(r[field]))
                except (ValueError, TypeError):
                    pass
            return round(sum(vals) / len(vals), 4) if vals else 0.0

        summary_rows.append({
            "llm_backbone": backbone,
            "requested_mode": mode,
            "n": n,
            "agent_success_rate": _mean("agent_success"),
            "json_success_rate": _mean("json_success"),
            "mean_latency_sec": _mean("latency_sec"),
            "mean_mandatory_fraction": _mean("mandatory_fraction"),
            "mean_required_relation_coverage": _mean("required_relation_coverage"),
            "mean_candidate_score": _mean("mean_candidate_score"),
            "mean_selected_data_coverage": _mean("mean_selected_data_coverage"),
            "mean_data_coverage_available_fraction": _mean("selected_data_coverage_available_fraction"),
            "mean_decision_quality_score": _mean("decision_quality_score"),
            "selection_set_count": len(set(str(r.get("selected_sources", "")) for r in group)),
            "mean_pairwise_selection_jaccard": _mean_pairwise_jaccard(
                [str(r.get("selected_sources", "")).split(";") for r in group]
            ),
            "mean_react_iterations": _mean("react_iterations"),
            "mean_confidence": _mean("confidence"),
        })

    summary_path = out_dir / "llm_agent_decision_summary.csv"
    if summary_rows:
        with summary_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)

    # Write per-task breakdown (useful for metabolomics vs proteo comparison)
    task_summary: Dict = {}
    for row in rows:
        key = (row["llm_backbone"], row["target_modality"])
        task_summary.setdefault(key, []).append(row)

    task_summary_rows = []
    for (backbone, target), group in task_summary.items():
        n = len(group)
        task_summary_rows.append({
            "llm_backbone": backbone,
            "target_modality": target,
            "n": n,
            "agent_success_rate": round(sum(float(r["agent_success"]) for r in group) / n, 4),
            "mean_required_relation_coverage": round(
                sum(float(r["required_relation_coverage"]) for r in group) / n, 4
            ),
            "mean_react_iterations": round(
                sum(float(r.get("react_iterations", 0)) for r in group) / n, 4
            ),
        })

    task_summary_path = out_dir / "llm_agent_task_modality_summary.csv"
    if task_summary_rows:
        with task_summary_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(task_summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(task_summary_rows)

    print(f"[BENCH] detail:        {detail_path}")
    print(f"[BENCH] summary:       {summary_path}")
    print(f"[BENCH] task_modality: {task_summary_path}")
    print(f"[BENCH] traces:        {trace_path}")


if __name__ == "__main__":
    main()
