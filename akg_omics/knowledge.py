"""Rule/Agent source selection and the shared human/metabolic prior builders."""
import os
import pickle
from pathlib import Path
import numpy as np
from .io import read_json, sha256, write_json
from .knowledge_agent.task_schema import TaskSpec
from .knowledge_agent.selector import select_sources
from .knowledge_agent.registry import load_registry

HUMAN_ASSETS = {
    "hgnc": {"hgnc_file": "hgnc_complete_set.txt"},
    "uniprot": {"uniprot_file": "uniprot_human_reviewed.tsv"},
    "reactome": {"reactome_uniprot_file": "UniProt2Reactome.txt", "reactome_ensembl_file": "Ensembl2Reactome.txt"},
    "cellmarker": {"cellmarker_file": "Cell_marker_Human.xlsx"},
    "proteinatlas": {"proteinatlas_file": "proteinatlas.tsv"},
    "kegg": {"kegg_pathways_file": "kegg_hsa_pathways.txt", "kegg_gene_pathway_file": "kegg_hsa_gene_pathway_links.txt"},
}
MET_ASSETS = ("hmdb/hmdb_metabolites.xml", "reactome/ChEBI2Reactome_All_Levels.txt",
              "reactome/Ensembl2Reactome_All_Levels.txt", "reactome/NCBI2Reactome_All_Levels.txt")


def validate_agent(selection):
    decision = selection.get("agent_decision", {})
    traces = decision.get("llm_call_traces", [])
    trace = decision.get("llm_call_trace", {})
    if not (selection.get("mode") == "agent+rule" and decision.get("ok") is True
            and not decision.get("policy_rescue") and decision.get("provider") not in (None, "mock")
            and (trace.get("ok") is True or any(t.get("ok") is True for t in traces))):
        raise RuntimeError("Agent did not return a successful real LLM selection; rule fallback is rejected")


def choose(spec, registry, profile, policy):
    mouse = spec.species == "mouse"
    os.environ["KNOWLEDGE_AGENT_ENABLE_AGENT"] = "1" if policy == "agent" else "0"
    # Main package selects the declared local catalog; it does not download resources.
    os.environ["KNOWLEDGE_AGENT_ENABLE_TOOLS"] = "0"
    os.environ["KNOWLEDGE_AGENT_SKIP_DISCOVERY"] = "1"
    if policy == "agent":
        os.environ.setdefault("KNOWLEDGE_AGENT_LLM_PROVIDER", "deepseek")
        from .knowledge_agent.llm_client import create_llm_client_from_env
        if create_llm_client_from_env() is None:
            raise ValueError("Agent selection requires an API key in the provider environment; use replay for existing experiments")
    selection = select_sources(spec, registry_sources=registry,
                  min_sources=1 if mouse else 4, max_sources=3 if mouse else 6,
                  selection_mode="agent_rule" if policy == "agent" else "rule_only",
                  data_profile=profile, selection_policy="anchored", performance_feedback=None).to_dict()
    if policy == "agent":
        validate_agent(selection)
    elif selection["mode"] != "rule_only":
        raise RuntimeError("Unexpected Rule selector mode")
    return selection


def build_prior(folder, spec, policy, assets, output, mode="replay", recipe=None):
    folder, assets, output = Path(folder), Path(assets), Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite knowledge: {output}")
    output.mkdir(parents=True, exist_ok=True)
    gn = np.load(folder / "gene_names.npy").astype(str).tolist()
    pn = np.load(folder / "other_names.npy").astype(str).tolist()
    symbol_path = folder / "other_symbols.npy"
    if symbol_path.exists():
        pn = np.load(symbol_path).astype(str).tolist()
    mouse = spec["dataset"] == "mouse"
    task = TaskSpec(task_id=f"{spec['dataset']}_T{spec['task']:02d}",
          source_modalities=["he"] + (["metabolism" if mouse and spec["counterpart"] == "other" else
                                      "protein" if spec["counterpart"] == "other" else "gene"] if spec["observed"] else []),
          target_modality="gene" if spec["target_kind"] == "gene" else ("metabolism" if mouse else "protein"),
          species="mouse" if mouse else "human", tissue="brain" if mouse else
                 {"coad": "colorectal tumor", "lymphnode": "lymph node", "tonsil": "tonsil"}[spec["dataset"]],
          required_relations=["metabolism_gene_association", "metabolite_pathway"] if mouse else
                 ["pathway_membership", "celltype_marker", "protein_gene_mapping" if spec["target_kind"] == "gene" else "gene_protein_mapping"])
    replay = (recipe if isinstance(recipe, dict) else read_json(recipe)) if mode == "replay" else None
    if replay and (replay["policy"] != policy or replay["task"] != spec["task"] or replay["dataset"] != spec["dataset"]):
        raise ValueError("Recipe dataset/task/policy mismatch")
    if mouse:
        from . import metabolic_builder as builder
        builder.configure(assets, output / "builder_cache")
        for name in MET_ASSETS:
            if not (assets / name).is_file():
                raise FileNotFoundError(assets / name)
        gene_id_path = folder / "gene_ids.npy"
        if not gene_id_path.exists():
            raise ValueError("Metabolic knowledge requires source-selected gene_ids.npy")
        gene_ids = np.load(gene_id_path).astype(str).tolist()
        precision = dict(precision_profile="frozen_source_only_H00", hmdb_ppm_tol=5.0,
                         hmdb_topk_candidates=2, hmdb_min_candidate_weight=0.02)
        def build(sources, tag):
            return builder.build_full_metabolite_kb(pn, gene_ids, gn, enabled_sources=sources,
                  cache_tag=tag, kb_precision_params=precision)
        registry_path = Path(__file__).parent / "metabolic_registry.json"
        registry = read_json(registry_path)["sources"]
        if replay:
            profile = dict(source_only=True, selected_gene_names=gn, selected_other_names=pn,
                           replayed_profile_sha256=replay.get("original_profile_sha256"))
        else:
            coverage = {}
            for sid, ss in [("hmdb", ["hmdb"]), ("reactome", ["hmdb", "reactome"]), ("chebi", ["hmdb", "reactome", "chebi"])]:
                k = build(ss, "coverage_" + sid)
                pg = k["metabolite_gene_prior"]
                n = int((pg.sum(1) > 0).sum()) + int((pg.sum(0) > 0).sum())
                coverage[sid] = dict(overall_coverage=n / (len(gn) + len(pn)), coverage_available=True,
                       covered_count=n, total_count=len(gn) + len(pn), stats=k["stats"],
                       coverage_definition="Conditional cross-modal endpoint coverage; not unique source attribution")
            profile = dict(task=task.to_dict(), species="mouse", tissue="brain", modalities=["gene", "metabolism", "he"],
                 feature_sets={"gene": dict(count=len(gn), examples=gn[:20], id_type="mouse_symbol"),
                               "metabolism": dict(count=len(pn), examples=pn[:20], id_type="m_z_not_confirmed_identity")},
                 source_coverage=coverage, full_catalog_prior_stats=build(["hmdb", "reactome", "chebi"], "profile_all")["stats"])
        selection = dict(mode="replayed_" + replay["selection_mode"], selected_source_ids=replay["selected_source_ids"]) if replay else choose(task, registry, profile, policy)
        result = build(selection["selected_source_ids"], "main_" + policy)
        record = dict(pg=result["metabolite_gene_prior"], gp=result["gene_metabolite_prior"],
                      gg=result["gene_gene_graph"], pp=result["metabolite_metabolite_graph"], stats=result["stats"],
                      weight_policy="Fixed metabolic builder coefficients; selection weights not consumed")
        asset_hashes = {n: sha256(assets / n) for n in MET_ASSETS}
    else:
        from . import human_builder as builder
        from .knowledge_agent.orchestrator import KnowledgePathConfig
        from .knowledge_agent.data_profiler import build_data_profile
        paths = {k: str(assets / name) for files in HUMAN_ASSETS.values() for k, name in files.items()}
        for path in paths.values():
            if not Path(path).is_file():
                raise FileNotFoundError(path)
        registry = [dict(s, available=True) for s in load_registry() if s["id"] in HUMAN_ASSETS]
        profile = build_data_profile(task, gene_names=gn, protein_names=pn, kb_paths=KnowledgePathConfig(**paths))
        if replay:
            selection = dict(mode="replayed_" + replay["selection_mode"], selected_source_ids=replay["selected_source_ids"])
            weights = replay["weights"]
        else:
            selection = choose(task, registry, profile, policy)
            names = {"direct_weight": "DIRECT", "module_weight": "MODULE", "celltype_weight": "CELLTYPE",
                     "gene_graph_weight": "GENE_GRAPH", "prot_graph_weight": "PROT_GRAPH", "gg_celltype_weight": "GG_CELLTYPE",
                     "pp_celltype_weight": "PP_CELLTYPE", "dorothea_graph_weight": "DOROTHEA_GRAPH", "omnipath_graph_weight": "OMNIPATH_GRAPH",
                     "corum_graph_weight": "CORUM_GRAPH", "proteinatlas_celltype_weight": "PROTEINATLAS_CELLTYPE",
                     "kegg_pathway_weight": "KEGG_PATHWAY", "string_graph_weight": "STRING_GRAPH"}
            base = {k: getattr(builder, "PROTEO_KB_" + v + "_WEIGHT") for k, v in names.items()}
            weights, _ = builder.adapt_proteo_builder_kwargs_with_agent_weights(base, selection, selection["selected_source_ids"], task.to_dict())
        selected = selection["selected_source_ids"]
        chosen = {k: str(assets / name) if sid in selected else None
                  for sid, files in HUMAN_ASSETS.items() for k, name in files.items()}
        pg, gp, gg, pp, stats, _ = builder.build_multi_relation_kb_upgraded(protein_names=pn, gene_names=gn, **chosen, **weights)
        record = dict(pg=pg, gp=gp, gg=gg, pp=pp, stats=stats, weights=weights)
        asset_hashes = {k: sha256(v) for k, v in chosen.items() if v}
    for k in ("pg", "gp", "gg", "pp"):
        if not np.isfinite(record[k]).all():
            raise ValueError("Invalid knowledge matrix: " + k)
    write_json(output / "source_profile.json", dict(task_spec=task.to_dict(), data_profile=profile,
               source_only=True, target_values_used=False, performance_feedback_used=False,
               source_transform_sha256=sha256(folder / "transforms.npz")))
    record.update(policy=policy, selection=selection, selection_replayed=mode == "replay",
                  actual_llm_called_this_batch=mode == "select" and policy == "agent",
                  profile_sha256=sha256(output / "source_profile.json"), asset_hashes=asset_hashes,
                  source_only=True, performance_feedback_used=False)
    path = output / "prior.pkl"
    with path.open("wb") as stream:
        pickle.dump(record, stream, protocol=4)
    write_json(output / "READY.json", {k: v for k, v in record.items() if k not in ("pg", "gp", "gg", "pp")} | {"prior_sha256": sha256(path)})
    return path
