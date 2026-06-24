import os
import re
import csv
import json
import copy
import inspect
import pickle
import hashlib
import warnings
import subprocess
import gzip
import xml.etree.ElementTree as ET
from datetime import datetime
from collections import defaultdict
from bisect import bisect_left, bisect_right
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import tifffile
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

warnings.filterwarnings("ignore")

import akg_omics as se

try:
    from .rule_orchestration import orchestrate_sources
    from .knowledge_agent import KnowledgePathConfig, TaskSpec, build_kb_with_orchestration
    from .knowledge_agent.registry import load_registry as load_knowledge_registry
    from .knowledge_agent.selector import select_sources as select_knowledge_sources
except ImportError:
    from rule_orchestration import orchestrate_sources
    from knowledge_agent import KnowledgePathConfig, TaskSpec, build_kb_with_orchestration
    from knowledge_agent.registry import load_registry as load_knowledge_registry
    from knowledge_agent.selector import select_sources as select_knowledge_sources

# ============================================================================
# 0) Env helpers
# ============================================================================
def _env_str(name, default):
    v = os.getenv(name)
    if v is None:
        return default
    v = str(v).strip()
    return v if v != "" else default


def _env_int(name, default):
    v = os.getenv(name)
    if v is None or str(v).strip() == "":
        return default
    return int(v)


def _env_float(name, default):
    v = os.getenv(name)
    if v is None or str(v).strip() == "":
        return default
    return float(v)


def _env_bool(name, default):
    v = os.getenv(name)
    if v is None or str(v).strip() == "":
        return bool(default)
    s = str(v).strip().lower()
    if s in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise ValueError(f"Invalid bool env {name}={v}")


def _env_int_list(name, default):
    v = os.getenv(name)
    if v is None or str(v).strip() == "":
        return list(default)
    parts = [x.strip() for x in str(v).split(",") if x.strip() != ""]
    return [int(x) for x in parts]


def _normalize_runtime_device(device_str):
    raw = str(device_str or "cpu").strip()
    visible = str(os.getenv("CUDA_VISIBLE_DEVICES", "")).strip()
    if not raw.startswith("cuda") or visible == "":
        return raw
    if ":" not in raw:
        return raw
    try:
        requested_idx = int(raw.split(":", 1)[1])
    except Exception:
        return raw
    visible_ids = [x.strip() for x in visible.split(",") if x.strip() != ""]
    if requested_idx < 0:
        return raw
    if requested_idx < len(visible_ids):
        return raw
    if len(visible_ids) == 1:
        mapped = "cuda:0"
        print(
            f">>> [DEVICE] remap {raw} -> {mapped} because CUDA_VISIBLE_DEVICES={visible}",
            flush=True,
        )
        return mapped
    return raw


def _env_str_list(name, default=None):
    v = os.getenv(name)
    if v is None or str(v).strip() == "":
        return list(default or [])
    return [x.strip() for x in re.split(r"[,;\s]+", str(v)) if x.strip()]


# ============================================================================
# 1) Global settings
# ============================================================================
# Unified run: protein tasks are the default final-version run. Metabolomics can
# still be enabled explicitly with RUN_METABOLOMICS_TASKS=1.
RUN_PROTEOGENOMICS_TASKS = _env_bool("RUN_PROTEOGENOMICS_TASKS", True)
RUN_METABOLOMICS_TASKS = _env_bool("RUN_METABOLOMICS_TASKS", False)

# Metabolomics pipeline config from the provided example.
device = _normalize_runtime_device(_env_str("DEVICE", "cuda:1"))
image_encoder = _env_str("IMAGE_ENCODER", "uni")
resolution = _env_int("RESOLUTION", 580)
num_neighbors = _env_int("NUM_NEIGHBORS", 2)

FINAL_SEEDS = _env_int_list("FINAL_SEEDS", [1])

# Hard-coded by request: metabolomics dataset and KB roots.
ndata_root = "data/10xgenomics/Mouse_brain"
folder_B1 = os.path.join(ndata_root, "B1")
folder_C1 = os.path.join(ndata_root, "C1")

KB_ROOT = "data/KB/metabolite"
CHEBI_DIR = os.path.join(KB_ROOT, "chebi")
HMDB_DIR = os.path.join(KB_ROOT, "hmdb")
REACTOME_DIR = os.path.join(KB_ROOT, "reactome")

# Proteogenomics dataset and KB roots (task1~task4).
PROTEO_DATA_ROOT = _env_str("PROTEO_DATA_ROOT", "data/spatch")
PROTEO_SAMPLE1 = _env_str("PROTEO_SAMPLE1", "protein")
PROTEO_SAMPLE2 = _env_str("PROTEO_SAMPLE2", "gene")
PROTEO_RAW_CACHE_DIR = _env_str("PROTEO_RAW_CACHE_DIR", "cache/spatch_raw")
PROTEO_RAW_SLICE1_CACHE = _env_str("PROTEO_RAW_SLICE1_CACHE", os.path.join(PROTEO_RAW_CACHE_DIR, f"{PROTEO_SAMPLE1}_processed.h5ad"))
PROTEO_RAW_SLICE2_CACHE = _env_str("PROTEO_RAW_SLICE2_CACHE", os.path.join(PROTEO_RAW_CACHE_DIR, f"{PROTEO_SAMPLE2}_processed.h5ad"))
PROTEO_KB_DIR = _env_str("PROTEO_KB_DIR", "data/KB")
PROTEO_HGNC_PATH = _env_str("PROTEO_HGNC_PATH", os.path.join(PROTEO_KB_DIR, "hgnc_complete_set.txt"))
PROTEO_UNIPROT_PATH = _env_str("PROTEO_UNIPROT_PATH", os.path.join(PROTEO_KB_DIR, "uniprot_human_reviewed.tsv"))
PROTEO_REACTOME_UNIPROT_PATH = _env_str("PROTEO_REACTOME_UNIPROT_PATH", os.path.join(PROTEO_KB_DIR, "UniProt2Reactome.txt"))
PROTEO_REACTOME_ENSEMBL_PATH = _env_str("PROTEO_REACTOME_ENSEMBL_PATH", os.path.join(PROTEO_KB_DIR, "Ensembl2Reactome.txt"))
PROTEO_CELLMARKER_PATH = _env_str("PROTEO_CELLMARKER_PATH", os.path.join(PROTEO_KB_DIR, "Cell_marker_Human.xlsx"))
PROTEO_DOROTHEA_PATH = _env_str("PROTEO_DOROTHEA_PATH", os.path.join(PROTEO_KB_DIR, "dorothea_grn.tsv"))
PROTEO_OMNIPATH_PATH = _env_str("PROTEO_OMNIPATH_PATH", os.path.join(PROTEO_KB_DIR, "omnipath_interactions.tsv"))
PROTEO_CORUM_PATH = _env_str("PROTEO_CORUM_PATH", os.path.join(PROTEO_KB_DIR, "corum_allComplexes.txt"))
PROTEO_PROTEINATLAS_PATH = _env_str("PROTEO_PROTEINATLAS_PATH", os.path.join(PROTEO_KB_DIR, "proteinatlas.tsv"))
PROTEO_KEGG_PATHWAYS_PATH = _env_str("PROTEO_KEGG_PATHWAYS_PATH", os.path.join(PROTEO_KB_DIR, "kegg_hsa_pathways.txt"))
PROTEO_KEGG_GENE_PATHWAY_PATH = _env_str("PROTEO_KEGG_GENE_PATHWAY_PATH", os.path.join(PROTEO_KB_DIR, "kegg_hsa_gene_pathway_links.txt"))
PROTEO_STRING_PATH = _env_str("PROTEO_STRING_PATH", os.path.join(PROTEO_KB_DIR, "9606.protein.links.v12.0.txt.gz"))

RUN_NAME = _env_str("RUN_NAME", "unified_8tasks_multiomics")
save_root = _env_str("SAVE_ROOT", f"./results/{RUN_NAME}/")
os.makedirs(save_root, exist_ok=True)

CACHE_ROOT = os.path.join(save_root, "cache")
PREP_CACHE_DIR = os.path.join(CACHE_ROOT, "preprocessed")
ALIGNED_CACHE_DIR = os.path.join(CACHE_ROOT, "aligned")
GRAPH_CACHE_DIR = os.path.join(CACHE_ROOT, "graphs")
KB_CACHE_DIR = os.path.join(CACHE_ROOT, "kb")
LATENT_CACHE_DIR = os.path.join(CACHE_ROOT, "latent")
PROTEO_USE_SHARED_CACHE = _env_bool("PROTEO_USE_SHARED_CACHE", True)
PROTEO_SHARED_CACHE_ROOT = _env_str(
    "PROTEO_SHARED_CACHE_ROOT",
    "cache/proteogenomics",
)
PROTEO_CACHE_DIR = os.path.join(
    PROTEO_SHARED_CACHE_ROOT if PROTEO_USE_SHARED_CACHE else CACHE_ROOT,
    "proteogenomics",
)
PROTEO_REFINED_DIR = os.path.join(PROTEO_CACHE_DIR, "refined")
PROTEO_KB_CACHE_DIR = os.path.join(PROTEO_CACHE_DIR, "kb_cache")
PROTEO_GRAPH_CACHE_DIR = os.path.join(PROTEO_CACHE_DIR, "graph_cache")
PROTEO_TASK_CACHE_DIR = os.path.join(CACHE_ROOT, "proteogenomics", "task_result")
for _d in [
    CACHE_ROOT,
    PREP_CACHE_DIR,
    ALIGNED_CACHE_DIR,
    GRAPH_CACHE_DIR,
    KB_CACHE_DIR,
    LATENT_CACHE_DIR,
    PROTEO_CACHE_DIR,
    PROTEO_REFINED_DIR,
    PROTEO_KB_CACHE_DIR,
    PROTEO_GRAPH_CACHE_DIR,
]:
    os.makedirs(_d, exist_ok=True)
os.makedirs(PROTEO_TASK_CACHE_DIR, exist_ok=True)
KB_ORCHESTRATION_DIR = os.path.join(save_root, "knowledge_orchestration")
os.makedirs(KB_ORCHESTRATION_DIR, exist_ok=True)

MET_TOPK = _env_int("MET_TOPK", 50)
GENE_TOPK = _env_int("GENE_TOPK", 1000)
GENE_LATENT_DIM = _env_int("GENE_LATENT_DIM", 128)

# Proteogenomics feature/preprocess config.
PROTEO_USE_PSEUDOSPOT = _env_bool("PROTEO_USE_PSEUDOSPOT", True)
PROTEO_GRID_SIZE = _env_int("PROTEO_GRID_SIZE", 32)
PROTEO_SUBSAMPLE_N1 = _env_int("PROTEO_SUBSAMPLE_N1", 50000)
PROTEO_SUBSAMPLE_N2 = _env_int("PROTEO_SUBSAMPLE_N2", 50000)
PROTEO_HE_DIM = _env_int("PROTEO_HE_DIM", 128)
PROTEO_GENE_TOPK = _env_int("PROTEO_GENE_TOPK", 500)
PROTEO_GENE_SELECT_MODE = _env_str("PROTEO_GENE_SELECT_MODE", "beneficial").lower()
if PROTEO_GENE_SELECT_MODE not in {"beneficial", "topvar", "random"}:
    raise ValueError(
        "PROTEO_GENE_SELECT_MODE must be one of: beneficial, topvar, random; "
        f"got {PROTEO_GENE_SELECT_MODE}"
    )
PROTEO_GENE_RANDOM_SEED = _env_int("PROTEO_GENE_RANDOM_SEED", 2026)
PROTEO_GENE_BENEFICIAL_LIST = _env_str("PROTEO_GENE_BENEFICIAL_LIST", "")
PROTEO_GENE_BENEFICIAL_TAG = re.sub(
    r"[^A-Za-z0-9_.-]+",
    "_",
    _env_str("PROTEO_GENE_BENEFICIAL_TAG", "pccavg"),
)[:64]
PROTEO_GENE_LATENT_DIM = _env_int("PROTEO_GENE_LATENT_DIM", 128)
PROTEO_BASE_GRAPH_ALPHA = _env_float("PROTEO_BASE_GRAPH_ALPHA", 1.0)

USE_PSEUDOSPOT = _env_bool("USE_PSEUDOSPOT", False)
GRID_SIZE = _env_int("GRID_SIZE", 32)
SPATIAL_K = _env_int("SPATIAL_K", 7)
HE_K = _env_int("HE_K", 7)
FUSION_ALPHA = _env_float("FUSION_ALPHA", 1.0)
HE_METRIC = _env_str("HE_METRIC", "cosine")

TRAIN_HIDDEN_DIM = _env_int("TRAIN_HIDDEN_DIM", 256)
TRAIN_NUM_LAYERS = _env_int("TRAIN_NUM_LAYERS", 2)
TRAIN_EPOCHS = _env_int("TRAIN_EPOCHS", 500)
TRAIN_LR = _env_float("TRAIN_LR", 1e-3)
TRAIN_DROPOUT = _env_float("TRAIN_DROPOUT", 0.1)

LAMBDA_TARGET = _env_float("LAMBDA_TARGET", 1.0)
LAMBDA_PCC = _env_float("LAMBDA_PCC", 0.0)
LAMBDA_LATENT = _env_float("LAMBDA_LATENT", 0.5)
LAMBDA_RECON = _env_float("LAMBDA_RECON", 0.2)
LAMBDA_KB = _env_float("LAMBDA_KB", 0.05)
LAMBDA_GRAPH = _env_float("LAMBDA_GRAPH", 0.05)
LAMBDA_ORTHO = _env_float("LAMBDA_ORTHO", 0.0)
LAMBDA_DGI = _env_float("LAMBDA_DGI", 0.0)
LAMBDA_ALIGN = _env_float("LAMBDA_ALIGN", 0.02)
ALIGN_MAX_POINTS = _env_int("ALIGN_MAX_POINTS", 512)
LAMBDA_FULL = _env_float("LAMBDA_FULL", 1.0)
LAMBDA_PARTIAL = _env_float("LAMBDA_PARTIAL", 0.0)
TARGET_SOFT_ALPHA = _env_float("TARGET_SOFT_ALPHA", 0.0)
KB_CT_PRIOR_MIX = _env_float("KB_CT_PRIOR_MIX", 0.7)

DISABLE_DYNAMIC_RETRIEVER = _env_bool("DISABLE_DYNAMIC_RETRIEVER", False)
DISABLE_KB_FUSION = _env_bool("DISABLE_KB_FUSION", False)
DISABLE_SOFT_GATE = _env_bool("DISABLE_SOFT_GATE", True)
DISABLE_RESIDUAL_REFINE = _env_bool("DISABLE_RESIDUAL_REFINE", False)

KB_DIRECT_WEIGHT = _env_float("KB_DIRECT_WEIGHT", 1.00)
KB_PATHWAY_WEIGHT = _env_float("KB_PATHWAY_WEIGHT", 0.40)
KB_GENE_GRAPH_PATHWAY_WEIGHT = _env_float("KB_GENE_GRAPH_PATHWAY_WEIGHT", 1.00)
KB_GENE_GRAPH_HMDB_WEIGHT = _env_float("KB_GENE_GRAPH_HMDB_WEIGHT", 0.30)
KB_MET_GRAPH_PATHWAY_WEIGHT = _env_float("KB_MET_GRAPH_PATHWAY_WEIGHT", 1.00)
KB_MET_GRAPH_CHEBI_WEIGHT = _env_float("KB_MET_GRAPH_CHEBI_WEIGHT", 0.25)
KB_MET_GRAPH_HMDB_WEIGHT = _env_float("KB_MET_GRAPH_HMDB_WEIGHT", 0.35)
MIN_SHARED_PATHWAY = _env_int("MIN_SHARED_PATHWAY", 1)

# Proteogenomics KB fusion weights (E15 defaults).
PROTEO_KB_DIRECT_WEIGHT = _env_float("PROTEO_KB_DIRECT_WEIGHT", 0.2)
PROTEO_KB_MODULE_WEIGHT = _env_float("PROTEO_KB_MODULE_WEIGHT", 0.35)
PROTEO_KB_CELLTYPE_WEIGHT = _env_float("PROTEO_KB_CELLTYPE_WEIGHT", 1.5)
PROTEO_KB_PROT_GRAPH_WEIGHT = _env_float("PROTEO_KB_PROT_GRAPH_WEIGHT", 1.0)
PROTEO_KB_GENE_GRAPH_WEIGHT = _env_float("PROTEO_KB_GENE_GRAPH_WEIGHT", 1.0)
PROTEO_KB_GG_CELLTYPE_WEIGHT = _env_float("PROTEO_KB_GG_CELLTYPE_WEIGHT", 0.5)
PROTEO_KB_PP_CELLTYPE_WEIGHT = _env_float("PROTEO_KB_PP_CELLTYPE_WEIGHT", 0.5)
USE_PROTEO_KB_AGENT = _env_bool("USE_PROTEO_KB_AGENT", False)
PROTEO_KB_REQUESTED_MODE = _env_str("PROTEO_KB_REQUESTED_MODE", _env_str("PROTEO_KB_SELECTION_MODE", "rule-only")).lower().replace("_", "-")
_MODE_ALIAS = {
    "rule": "rule-only",
    "rule-only": "rule-only",
    "rule-data": "rule-data",
    "agent": "agent-only",
    "agent-only": "agent-only",
    "agent-data": "agent-data",
    "agent-rule": "agent-rule",
    "rule+agent": "agent-rule",
    "agent+rule": "agent-rule",
    "agent-rule-data": "agent-rule-data",
    "agent-data-rule": "agent-rule-data",
}
PROTEO_KB_REQUESTED_MODE = _MODE_ALIAS.get(PROTEO_KB_REQUESTED_MODE, PROTEO_KB_REQUESTED_MODE)
if PROTEO_KB_REQUESTED_MODE not in {"rule-only", "rule-data", "agent-only", "agent-data", "agent-rule", "agent-rule-data"}:
    raise ValueError(f"Unsupported PROTEO_KB_REQUESTED_MODE={PROTEO_KB_REQUESTED_MODE}")
REQUIRE_PROTEO_KB_LLM_AGENT = _env_bool(
    "REQUIRE_PROTEO_KB_LLM_AGENT",
    USE_PROTEO_KB_AGENT and PROTEO_KB_REQUESTED_MODE in {"agent-only", "agent-data", "agent-rule-data"},
)
PROTEO_KB_USE_DATA_PROFILE = _env_bool("PROTEO_KB_USE_DATA_PROFILE", PROTEO_KB_REQUESTED_MODE in {"rule-data", "agent-data", "agent-rule-data"})
PROTEO_KB_SELECTION_MODE = {
    "rule-only": "rule_only",
    "rule-data": "rule_only",
    "agent-only": "agent_only",
    "agent-data": "agent_only",
    "agent-rule": "agent_rule",
    "agent-rule-data": "agent_rule",
}[PROTEO_KB_REQUESTED_MODE]
if PROTEO_KB_SELECTION_MODE not in {"rule_only", "agent_only", "agent_rule"}:
    raise ValueError(f"Unsupported PROTEO_KB_SELECTION_MODE={PROTEO_KB_SELECTION_MODE}")
PROTEO_KB_AGENT_MIN_SOURCES = _env_int("PROTEO_KB_AGENT_MIN_SOURCES", 4)
PROTEO_KB_AGENT_MAX_SOURCES = _env_int("PROTEO_KB_AGENT_MAX_SOURCES", 6)
if PROTEO_KB_AGENT_MIN_SOURCES > PROTEO_KB_AGENT_MAX_SOURCES:
    raise ValueError(
        f"PROTEO_KB_AGENT_MIN_SOURCES={PROTEO_KB_AGENT_MIN_SOURCES} exceeds "
        f"PROTEO_KB_AGENT_MAX_SOURCES={PROTEO_KB_AGENT_MAX_SOURCES}"
    )
PROTEO_KB_DOROTHEA_GRAPH_WEIGHT = _env_float("PROTEO_KB_DOROTHEA_GRAPH_WEIGHT", 0.50)
PROTEO_KB_OMNIPATH_GRAPH_WEIGHT = _env_float("PROTEO_KB_OMNIPATH_GRAPH_WEIGHT", 0.50)
PROTEO_KB_CORUM_GRAPH_WEIGHT = _env_float("PROTEO_KB_CORUM_GRAPH_WEIGHT", 0.50)
PROTEO_KB_PROTEINATLAS_CELLTYPE_WEIGHT = _env_float("PROTEO_KB_PROTEINATLAS_CELLTYPE_WEIGHT", 0.50)
PROTEO_KB_KEGG_PATHWAY_WEIGHT = _env_float("PROTEO_KB_KEGG_PATHWAY_WEIGHT", 0.50)
PROTEO_KB_STRING_GRAPH_WEIGHT = _env_float("PROTEO_KB_STRING_GRAPH_WEIGHT", 0.50)
PROTEO_KB_STRING_MIN_SCORE = _env_int("PROTEO_KB_STRING_MIN_SCORE", 700)
PROTEO_KB_STRING_MAX_EDGES = _env_int("PROTEO_KB_STRING_MAX_EDGES", 2000000)
PROTEO_AGENT_TWO_ROUND = _env_bool("PROTEO_AGENT_TWO_ROUND", False)
PROTEO_RULE_BASELINE_SOURCES = [x.strip().lower() for x in _env_str_list("PROTEO_RULE_BASELINE_SOURCES", ["hgnc", "uniprot", "reactome", "cellmarker"]) if x.strip()]
PROTEO_IMPORT_FORMAL_RULE_BASELINE = _env_bool("PROTEO_IMPORT_FORMAL_RULE_BASELINE", False)
PROTEO_FORMAL_RULE_BASELINE_CSV = _env_str("PROTEO_FORMAL_RULE_BASELINE_CSV", "")
PROTEO_FORMAL_RULE_BASELINE_DIR = _env_str("PROTEO_FORMAL_RULE_BASELINE_DIR", "")
PROTEO_AGENT_METRIC_AWARE_REFINEMENT = _env_bool("PROTEO_AGENT_METRIC_AWARE_REFINEMENT", True)

# Metabolomics KB source selection mirrors the proteogenomics rule/agent switch,
# while keeping the actual metabolite KB builder restricted to supported sources.
MET_KB_ALLOWED_SOURCES = [
    x.strip().lower()
    for x in _env_str_list("MET_KB_ALLOWED_SOURCES", ["hmdb", "reactome", "chebi"])
    if x.strip()
]
USE_MET_KB_AGENT = _env_bool("USE_MET_KB_AGENT", False)
MET_KB_REQUESTED_MODE = _env_str("MET_KB_REQUESTED_MODE", _env_str("MET_KB_SELECTION_MODE", "rule-only")).lower().replace("_", "-")
MET_KB_REQUESTED_MODE = _MODE_ALIAS.get(MET_KB_REQUESTED_MODE, MET_KB_REQUESTED_MODE)
if MET_KB_REQUESTED_MODE not in {"rule-only", "rule-data", "agent-only", "agent-data", "agent-rule", "agent-rule-data"}:
    raise ValueError(f"Unsupported MET_KB_REQUESTED_MODE={MET_KB_REQUESTED_MODE}")
REQUIRE_MET_KB_LLM_AGENT = _env_bool(
    "REQUIRE_MET_KB_LLM_AGENT",
    USE_MET_KB_AGENT and MET_KB_REQUESTED_MODE in {"agent-only", "agent-data", "agent-rule-data"},
)
MET_KB_USE_DATA_PROFILE = _env_bool("MET_KB_USE_DATA_PROFILE", MET_KB_REQUESTED_MODE in {"rule-data", "agent-data", "agent-rule-data"})
MET_KB_SELECTION_MODE = {
    "rule-only": "rule_only",
    "rule-data": "rule_only",
    "agent-only": "agent_only",
    "agent-data": "agent_only",
    "agent-rule": "agent_rule",
    "agent-rule-data": "agent_rule",
}[MET_KB_REQUESTED_MODE]
MET_KB_AGENT_MIN_SOURCES = _env_int("MET_KB_AGENT_MIN_SOURCES", 2)
MET_KB_AGENT_MAX_SOURCES = _env_int("MET_KB_AGENT_MAX_SOURCES", 3)
MET_KB_SELECTION_POLICY = _env_str("MET_KB_SELECTION_POLICY", "free")
if MET_KB_AGENT_MIN_SOURCES > MET_KB_AGENT_MAX_SOURCES:
    raise ValueError(
        f"MET_KB_AGENT_MIN_SOURCES={MET_KB_AGENT_MIN_SOURCES} exceeds "
        f"MET_KB_AGENT_MAX_SOURCES={MET_KB_AGENT_MAX_SOURCES}"
    )

# Metabolomics alignment to E15: by default, use fixed KB/graph loss weights.
USE_KB_QUALITY_SCALING = _env_bool("USE_KB_QUALITY_SCALING", False)

KB_USAGE_MODE = _env_str("KB_USAGE_MODE", "warmup_and_injection")
if KB_USAGE_MODE not in {"off", "warmup_only", "injection_only", "warmup_and_injection"}:
    raise ValueError(f"Unsupported KB_USAGE_MODE={KB_USAGE_MODE}")
USE_KB_FEATURE_WARMUP = KB_USAGE_MODE in {"warmup_only", "warmup_and_injection"}
USE_KB_MODEL_INJECTION = KB_USAGE_MODE in {"injection_only", "warmup_and_injection"}

# Training/cache control.
FORCE_TASK_RETRAIN = _env_bool("FORCE_TASK_RETRAIN", False)
MET_TASK_CACHE_VERSION = _env_str("MET_TASK_CACHE_VERSION", "v3_aligned_fullpartial_rmse")
PROTEO_TASK_CACHE_VERSION = _env_str("PROTEO_TASK_CACHE_VERSION", "v5_llm_agent_weighted_kb")

# Detailed analysis / visualization settings.
ENABLE_DETAILED_ANALYSIS = _env_bool("ENABLE_DETAILED_ANALYSIS", False)
ENABLE_KB_DYNAMIC_EXPORT = _env_bool("ENABLE_KB_DYNAMIC_EXPORT", False)
TOP_FEATURE_N = _env_int("TOP_FEATURE_N", 20)
PLOT_TOP_FEATURE_N = _env_int("PLOT_TOP_FEATURE_N", 6)
DETAILED_ANALYSIS_DIR = os.path.join(save_root, "detailed_analysis")
if ENABLE_DETAILED_ANALYSIS:
    os.makedirs(DETAILED_ANALYSIS_DIR, exist_ok=True)

ION_MODE = _env_str("ION_MODE", "positive")
PPM_TOL = _env_float("PPM_TOL", 10.0)
PPM_SIGMA = _env_float("PPM_SIGMA", 3.0)
TOPK_CANDIDATES = _env_int("TOPK_CANDIDATES", 5)
MIN_CANDIDATE_WEIGHT = _env_float("MIN_CANDIDATE_WEIGHT", 1e-4)
USE_CHEBI_RELATION = _env_bool("USE_CHEBI_RELATION", True)

# Proteogenomics final task setup (task1~task4), aligned to E15.
PROTEO_FINAL_KB_CFG = {
    "direct_weight": PROTEO_KB_DIRECT_WEIGHT,
    "module_weight": PROTEO_KB_MODULE_WEIGHT,
    "celltype_weight": PROTEO_KB_CELLTYPE_WEIGHT,
    "gene_graph_weight": PROTEO_KB_GENE_GRAPH_WEIGHT,
    "prot_graph_weight": PROTEO_KB_PROT_GRAPH_WEIGHT,
    "gg_celltype_weight": PROTEO_KB_GG_CELLTYPE_WEIGHT,
    "pp_celltype_weight": PROTEO_KB_PP_CELLTYPE_WEIGHT,
    "dorothea_graph_weight": PROTEO_KB_DOROTHEA_GRAPH_WEIGHT,
    "omnipath_graph_weight": PROTEO_KB_OMNIPATH_GRAPH_WEIGHT,
    "corum_graph_weight": PROTEO_KB_CORUM_GRAPH_WEIGHT,
    "proteinatlas_celltype_weight": PROTEO_KB_PROTEINATLAS_CELLTYPE_WEIGHT,
    "kegg_pathway_weight": PROTEO_KB_KEGG_PATHWAY_WEIGHT,
    "string_graph_weight": PROTEO_KB_STRING_GRAPH_WEIGHT,
}
PROTEO_TASK_SPECS = [
    {
        "task": "task1_he_to_gene",
        "target_task": "gene",
        "full_use_obs": False,
        "partial_use_obs": False,
        "graph_alpha": 1.0,
    },
    {
        "task": "task2_he_to_protein",
        "target_task": "protein",
        "full_use_obs": False,
        "partial_use_obs": False,
        "graph_alpha": 1.0,
    },
    {
        "task": "task3_he_protein_to_gene",
        "target_task": "gene",
        "full_use_obs": True,
        "partial_use_obs": True,
        "graph_alpha": 1.0,
    },
    {
        "task": "task4_he_gene_to_protein",
        "target_task": "protein",
        "full_use_obs": True,
        "partial_use_obs": True,
        "graph_alpha": 1.0,
    },
]
PROTEO_TASK_FILTER = set([x.strip() for x in _env_str_list("PROTEO_TASK_FILTER", []) if x.strip()])

PROTEO_MODE_TAG = (
    f"g{PROTEO_GRID_SIZE}_he{PROTEO_HE_DIM}_gene{PROTEO_GENE_TOPK}"
    f"_gsel{PROTEO_GENE_SELECT_MODE}_gben{PROTEO_GENE_BENEFICIAL_TAG}"
    f"_grs{PROTEO_GENE_RANDOM_SEED}_glat{PROTEO_GENE_LATENT_DIM}"
    f"_a{PROTEO_BASE_GRAPH_ALPHA}_sk{SPATIAL_K}_hk{HE_K}"
)
PROTEO_ADATA1_REFINED_CACHE = os.path.join(PROTEO_REFINED_DIR, f"{PROTEO_SAMPLE1}_{PROTEO_MODE_TAG}.h5ad")
PROTEO_ADATA2_REFINED_CACHE = os.path.join(PROTEO_REFINED_DIR, f"{PROTEO_SAMPLE2}_{PROTEO_MODE_TAG}.h5ad")
PROTEO_GRAPH1_REFINED_CACHE = os.path.join(PROTEO_REFINED_DIR, f"{PROTEO_SAMPLE1}_{PROTEO_MODE_TAG}_graph.pkl")
PROTEO_GRAPH2_REFINED_CACHE = os.path.join(PROTEO_REFINED_DIR, f"{PROTEO_SAMPLE2}_{PROTEO_MODE_TAG}_graph.pkl")
PROTEO_GENE_PCA_COMP_CACHE = os.path.join(PROTEO_REFINED_DIR, f"{PROTEO_MODE_TAG}_gene_pca_components.npy")
PROTEO_GENE_PCA_MEAN_CACHE = os.path.join(PROTEO_REFINED_DIR, f"{PROTEO_MODE_TAG}_gene_pca_mean.npy")
PROTEO_GENE1_LAT_CACHE = os.path.join(PROTEO_REFINED_DIR, f"{PROTEO_SAMPLE1}_{PROTEO_MODE_TAG}_gene_latent.npy")
PROTEO_GENE2_LAT_CACHE = os.path.join(PROTEO_REFINED_DIR, f"{PROTEO_SAMPLE2}_{PROTEO_MODE_TAG}_gene_latent.npy")

ADDUCTS_POS = [
    ("M+H", 1.007276, 1, 1.0),
    ("M+Na", 22.989218, 1, 0.55),
    ("M+K", 38.963158, 1, 0.25),
    ("M+NH4", 18.033823, 1, 0.20),
]
ADDUCTS_NEG = [
    ("M-H", -1.007276, 1, 1.0),
    ("M+Cl", 34.969402, 1, 0.20),
    ("M+FA-H", 44.998201, 1, 0.10),
]


# ============================================================================
# 2) Generic helpers
# ============================================================================
def stable_hash(*items):
    raw = "||".join([str(x) for x in items])
    return hashlib.md5(raw.encode("utf-8")).hexdigest()[:16]


def atomic_pickle_dump(obj, path):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(obj, f)
    os.replace(tmp, path)


def refresh_row_selection_metadata(row, selected_sources, report, domain=None):
    out = dict(row or {})
    out["selected_sources"] = ";".join([str(x) for x in (selected_sources or [])])
    out["kb_select_mode"] = str((report or {}).get("selection", {}).get("mode", ""))
    out["kb_selection_mode_requested"] = str((report or {}).get("selection_mode_requested", PROTEO_KB_SELECTION_MODE))
    agent_decision = (report or {}).get("selection", {}).get("agent_decision", {})
    agent_failure = (report or {}).get("agent_failure", {})
    if isinstance(agent_decision, dict):
        out["agent_provider"] = str(agent_decision.get("provider", ""))
        out["agent_model"] = str(agent_decision.get("model", ""))
        out["agent_confidence"] = agent_decision.get("confidence", np.nan)
        out["agent_weight_strategy"] = str(agent_decision.get("source_weight_strategy", ""))
        out["agent_policy_rescue"] = str(agent_decision.get("policy_rescue", ""))
        usage_modes = agent_decision.get("source_usage_modes", {})
        usage_modes = usage_modes if isinstance(usage_modes, dict) else {}
        source_actions = agent_decision.get("source_actions", {})
        source_actions = source_actions if isinstance(source_actions, dict) else {}
        out["agent_usage_modes"] = ";".join([f"{k}:{usage_modes[k]}" for k in sorted(usage_modes)])
        out["agent_source_actions"] = ";".join([f"{k}:{source_actions[k]}" for k in sorted(source_actions)])
    if isinstance(agent_failure, dict):
        out["agent_failure_stage"] = str(agent_failure.get("failure_stage", ""))
        out["agent_failure_message"] = str(agent_failure.get("failure_message", ""))
    out["kb_orchestration_report"] = str((report or {}).get("report_path", ""))
    out["agent_trace_path"] = str((report or {}).get("agent_trace_path", ""))
    out.setdefault("RMSE", np.nan)
    if domain is not None:
        out["domain"] = str(domain)
    return out


def atomic_write_h5ad(adata, path):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    tmp = path + ".tmp.h5ad"
    adata_to_write = adata.copy()
    try:
        adata_to_write.obs_names = pd.Index(np.asarray(adata_to_write.obs_names).astype(str))
    except Exception:
        pass
    try:
        adata_to_write.var_names = pd.Index(np.asarray(adata_to_write.var_names).astype(str))
    except Exception:
        pass

    def _sanitize_df_for_h5ad(df):
        df = df.copy()
        try:
            df.index = pd.Index(np.asarray(df.index).astype(str))
        except Exception:
            pass
        for col in df.columns:
            series = df[col]
            dtype_name = str(getattr(series.dtype, "name", series.dtype)).lower()
            if "string" in dtype_name:
                df[col] = series.astype(str)
            elif dtype_name == "object":
                sample = series.dropna()
                if len(sample) > 0 and isinstance(sample.iloc[0], str):
                    df[col] = series.astype(str)
        return df

    try:
        adata_to_write.obs = _sanitize_df_for_h5ad(adata_to_write.obs)
    except Exception:
        pass
    try:
        adata_to_write.var = _sanitize_df_for_h5ad(adata_to_write.var)
    except Exception:
        pass

    adata_to_write.write_h5ad(tmp)
    os.replace(tmp, path)


def get_local_met_source_status():
    hmdb_xml = os.path.join(HMDB_DIR, "hmdb_metabolites.xml")
    return {
        "hmdb": os.path.exists(hmdb_xml),
        "reactome": os.path.isdir(REACTOME_DIR),
        "chebi": os.path.isdir(CHEBI_DIR),
        "hgnc": False,
        "uniprot": False,
        "cellmarker": False,
        "kegg": False,
    }


def _met_task_required_relations(source_modalities, target_modality):
    src = {str(x).strip().lower() for x in (source_modalities or [])}
    target = str(target_modality).strip().lower()
    rel = ["pathway_membership", "metabolite_pathway"]
    if target == "metabolism":
        rel.append("gene_metabolism_association" if "gene" in src else "metabolite_ontology")
    if "metabolism" in src and target == "gene":
        rel.append("metabolism_gene_association")
    return [x for x in rel if x]


def _build_met_data_profile(source_modalities, target_modality):
    modalities = sorted({str(x).strip().lower() for x in (source_modalities or [])} | {str(target_modality).strip().lower()})
    feature_sets = {}
    if "gene" in modalities:
        feature_sets["gene"] = {"count": None, "id_type": "gene_symbol_or_ensembl", "examples": []}
    if "metabolism" in modalities:
        feature_sets["metabolism"] = {"count": None, "id_type": "mz_or_metabolite_feature", "examples": []}
    return {
        "schema_version": "metabolomics_lightweight_v1",
        "species": "mouse",
        "modalities": modalities,
        "feature_sets": feature_sets,
        "source_coverage": {},
        "warnings": [
            "Metabolomics source coverage is evaluated during KB construction; selection uses task relations and local availability."
        ],
    }


def _write_met_selection_report(report_path, report):
    parent = os.path.dirname(report_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    report = dict(report or {})
    report["report_path"] = report_path
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    return report


def select_sources_for_met_task_with_agent(task_id, source_modalities, target_modality, local_status, report_path):
    allowed = [s for s in MET_KB_ALLOWED_SOURCES if s in {"hmdb", "reactome", "chebi"}]
    if not allowed:
        allowed = ["hmdb", "reactome", "chebi"]
    registry_sources = []
    for src in load_knowledge_registry():
        sid = str(src.get("id", "")).strip().lower()
        if sid not in allowed:
            continue
        item = dict(src)
        item["id"] = sid
        item["available"] = bool(local_status.get(sid, False))
        registry_sources.append(item)

    task_spec = TaskSpec(
        task_id=str(task_id),
        source_modalities=list(source_modalities or []),
        target_modality=str(target_modality),
        species="mouse",
        required_relations=_met_task_required_relations(source_modalities, target_modality),
    )
    selection = select_knowledge_sources(
        task=task_spec,
        registry_sources=registry_sources,
        min_sources=MET_KB_AGENT_MIN_SOURCES,
        max_sources=MET_KB_AGENT_MAX_SOURCES,
        selection_mode=MET_KB_SELECTION_MODE,
        data_profile=_build_met_data_profile(source_modalities, target_modality) if MET_KB_USE_DATA_PROFILE else None,
        selection_policy=MET_KB_SELECTION_POLICY,
    )
    selected = [
        str(s).strip().lower()
        for s in selection.selected_source_ids
        if str(s).strip().lower() in allowed and local_status.get(str(s).strip().lower(), False)
    ]
    selected = list(dict.fromkeys(selected))
    if REQUIRE_MET_KB_LLM_AGENT and selection.mode not in {"agent_only", "agent+rule"}:
        raise RuntimeError(
            "Metabolomics KB agent was required but LLM source selection did not run. "
            "Check KNOWLEDGE_AGENT_ENABLE_AGENT, provider, endpoint, API key, model, and network access."
        )
    if not selected:
        selected = [s for s in allowed if local_status.get(s, False)]
    report = {
        "task": task_spec.to_dict(),
        "domain": "metabolomics",
        "selection_mode_requested": MET_KB_SELECTION_MODE,
        "requested_mode": MET_KB_REQUESTED_MODE,
        "use_met_kb_agent": bool(USE_MET_KB_AGENT),
        "use_data_profile": bool(MET_KB_USE_DATA_PROFILE),
        "selection_policy": MET_KB_SELECTION_POLICY,
        "local_source_status": {s: bool(local_status.get(s, False)) for s in allowed},
        "allowed_sources": allowed,
        "selection": selection.to_dict(),
        "final_selected_sources": selected,
    }
    return selected, _write_met_selection_report(report_path, report)


def select_sources_for_met_task(task_id, source_modalities, target_modality, repeat_idx=None, seed=None, orchestration_dir=None):
    if orchestration_dir is None:
        orchestration_dir = KB_ORCHESTRATION_DIR
    local_status = get_local_met_source_status()
    suffix = ""
    if repeat_idx is not None and seed is not None:
        suffix = f"_rep{int(repeat_idx)}_seed{int(seed)}"
    report_path = os.path.join(orchestration_dir, f"kb_orchestration_{task_id}{suffix}.json")
    if USE_MET_KB_AGENT:
        selected, report = select_sources_for_met_task_with_agent(
            task_id=task_id,
            source_modalities=source_modalities,
            target_modality=target_modality,
            local_status=local_status,
            report_path=report_path,
        )
        print(
            f">>> [KB-SELECT] {task_id} requested={MET_KB_REQUESTED_MODE} mode={str(report.get('selection', {}).get('mode', ''))} data_profile={MET_KB_USE_DATA_PROFILE} selected={';'.join(selected)}",
            flush=True,
        )
        return selected, report

    selected, report = orchestrate_sources(
        task_id=task_id,
        source_modalities=source_modalities,
        target_modality=target_modality,
        local_source_status=local_status,
        max_sources=4,
        ensure_core_sources=[],
        report_path=report_path,
    )
    selected = [s for s in selected if s in {"hmdb", "reactome", "chebi"}]
    if not selected:
        selected = [s for s in ["hmdb", "reactome", "chebi"] if local_status.get(s, False)]
    report["domain"] = "metabolomics"
    report["selection_mode_requested"] = "rule_only"
    report["requested_mode"] = "rule-only"
    report["use_met_kb_agent"] = False
    report["final_selected_sources"] = selected
    print(
        f">>> [KB-SELECT] {task_id} mode={str(report.get('selection', {}).get('mode', ''))} selected={';'.join(selected)}",
        flush=True,
    )
    return selected, report


def to_dense(x):
    return x.toarray() if sp.issparse(x) else np.asarray(x)


def local_name(tag):
    return tag.split("}")[-1] if isinstance(tag, str) else str(tag)


def normalize_text(x):
    if x is None:
        return None
    x = str(x).strip()
    if x == "" or x.lower() == "nan":
        return None
    return x


def normalize_token(x):
    x = normalize_text(x)
    if x is None:
        return None
    x = x.upper().strip()
    x = re.sub(r"\s+", " ", x)
    return x


def normalize_alias(x):
    x = normalize_token(x)
    if x is None:
        return None
    x = re.sub(r"[^A-Z0-9]+", "", x)
    return x or None


def canonical_chebi_id(x):
    x = normalize_text(x)
    if x is None:
        return None
    m = re.search(r"(CHEBI:)?(\d+)", x.upper())
    if not m:
        return None
    return f"CHEBI:{m.group(2)}"


def canonical_hmdb_id(x):
    x = normalize_text(x)
    if x is None:
        return None
    m = re.search(r"HMDB\d+", x.upper())
    return m.group(0) if m else None


def canonical_ensembl_gene(x):
    x = normalize_text(x)
    if x is None:
        return None
    x = x.split(".")[0].upper()
    return x if x.startswith("ENS") else None


def canonical_ncbi_gene(x):
    x = normalize_text(x)
    if x is None:
        return None
    x = x.strip()
    return x if x.isdigit() else None


def row_normalize_matrix(m, eps=1e-8):
    m = np.asarray(m, dtype=np.float32)
    row_sum = m.sum(axis=1, keepdims=True)
    row_sum[row_sum < eps] = 1.0
    return m / row_sum


def normalize_adjacency_with_selfloop(a, eps=1e-8):
    a = np.asarray(a, dtype=np.float32)
    if a.shape[0] == 0:
        return a
    a = a + np.eye(a.shape[0], dtype=np.float32)
    deg = a.sum(axis=1, keepdims=True)
    deg[deg < eps] = 1.0
    return a / deg


def resolve_text_columns(df, candidates):
    low = {c.lower().replace("_", ""): c for c in df.columns}
    for cand in candidates:
        cc = cand.lower().replace("_", "")
        if cc in low:
            return low[cc]
    return None


def ensure_spatial_coords(adata):
    if "spatial" not in adata.obsm.keys():
        if "array_row" in adata.obs.columns and "array_col" in adata.obs.columns:
            adata.obsm["spatial"] = adata.obs[["array_row", "array_col"]].values
        elif "x" in adata.obs.columns and "y" in adata.obs.columns:
            adata.obsm["spatial"] = adata.obs[["x", "y"]].values
        elif "Center_X" in adata.obs.columns and "Center_Y" in adata.obs.columns:
            adata.obsm["spatial"] = adata.obs[["Center_X", "Center_Y"]].values
        else:
            raise ValueError("Cannot find spatial coordinates in adata.")
    return adata


def find_first_file(folder, prefixes=None, suffixes=None):
    folder = Path(folder)
    prefixes = prefixes or [""]
    suffixes = suffixes or [".h5ad", ".png", ".jpg", ".jpeg", ".tif", ".tiff"]
    for f in sorted(folder.iterdir()):
        if not f.is_file():
            continue
        ok_prefix = any(f.name.startswith(p) for p in prefixes)
        ok_suffix = any(f.name.lower().endswith(s.lower()) for s in suffixes)
        if ok_prefix and ok_suffix:
            return str(f)
    raise FileNotFoundError(f"Cannot find file in {folder} with prefixes={prefixes} and suffixes={suffixes}")


def load_rna_h5ad(folder):
    return find_first_file(folder, prefixes=["rna_", "RNA_", "rna", "RNA"], suffixes=[".h5ad"])


def load_metabolite_h5ad(folder):
    return find_first_file(folder, prefixes=["metabolite_", "Metabolite_", "metabolite", "Metabolite"], suffixes=[".h5ad"])


def load_image_file(folder):
    return find_first_file(folder, prefixes=["", "HE", "he", "H&E"], suffixes=[".png", ".jpg", ".jpeg", ".tif", ".tiff"])


def safe_read_he_image(img_path):
    suffix = Path(img_path).suffix.lower()
    if suffix in [".tif", ".tiff"]:
        img = tifffile.imread(img_path)
        if img.ndim == 3 and img.shape[0] in [3, 4]:
            img = np.transpose(img, (1, 2, 0))
    elif suffix in [".jpg", ".jpeg", ".png", ".bmp"]:
        img = np.array(Image.open(img_path).convert("RGB"))
    else:
        try:
            img = np.array(Image.open(img_path).convert("RGB"))
        except Exception:
            img = tifffile.imread(img_path)
            if img.ndim == 3 and img.shape[0] in [3, 4]:
                img = np.transpose(img, (1, 2, 0))
    return img, 1.0


def compute_three_metrics(gt, pred, graph_eval):
    _, pcc = se.utils.Compute_metrics(gt.copy(), pred.copy(), metric="pcc")
    _, ssim = se.utils.Compute_metrics(gt.copy(), pred.copy(), metric="ssim", graph=graph_eval)
    _, cmd = se.utils.Compute_metrics(gt.copy(), pred.copy(), metric="cmd")
    _, rmse = se.utils.Compute_metrics(gt.copy(), pred.copy(), metric="rmse")
    return {"PCC": float(pcc), "SSIM": float(ssim), "CMD": float(cmd), "RMSE": float(rmse)}


# ============================================================================
# Detailed task analysis helpers: Top-N features, loss curves, spatial plots
# ============================================================================
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def sanitize_filename(x):
    x = str(x)
    x = re.sub(r"[^\w\-.]+", "_", x)
    return x[:120]


def safe_feature_names(adata, fallback_prefix="feature"):
    if adata is not None and hasattr(adata, "var_names"):
        return np.asarray(adata.var_names).astype(str).tolist()
    return None


def filter_loss_columns_for_plot(hist_df, plot_cols, eps=1e-12):
    keep = []
    for c in plot_cols:
        vals = hist_df[c].values.astype(float)
        if not np.all(np.isfinite(vals)):
            continue
        if np.nanmax(np.abs(vals)) <= eps:
            continue
        keep.append(c)
    return keep


def save_loss_history_and_plot(history, out_dir, task_name):
    ensure_dir(out_dir)
    if history is None or len(history) == 0:
        print(f">>> [WARN] Empty history for {task_name}, skip loss plot.", flush=True)
        return None, None

    hist_df = pd.DataFrame(history)
    hist_df.insert(0, "epoch", np.arange(1, len(hist_df) + 1))
    csv_path = os.path.join(out_dir, "loss_history.csv")
    hist_df.to_csv(csv_path, index=False)

    plot_cols = [
        c for c in hist_df.columns
        if c != "epoch" and pd.api.types.is_numeric_dtype(hist_df[c])
    ]
    plot_cols = filter_loss_columns_for_plot(hist_df, plot_cols)
    if len(plot_cols) == 0:
        print(f">>> [WARN] No plottable loss columns for {task_name}.", flush=True)
        return csv_path, None

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(11, 7))
    for c in plot_cols:
        plt.plot(hist_df["epoch"].values, hist_df[c].values, label=c, linewidth=1.5)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Loss curves: {task_name}")
    plt.legend(fontsize=7, ncol=2)
    plt.tight_layout()

    fig_path = os.path.join(out_dir, "loss_curves.png")
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f">>> [LOSS] saved: {csv_path}", flush=True)
    print(f">>> [LOSS] saved: {fig_path}", flush=True)
    return csv_path, fig_path


def compute_vectorized_feature_metrics(gt, pred, feature_names=None, eps=1e-8):
    gt = to_dense(gt).astype(np.float32)
    pred = to_dense(pred).astype(np.float32)
    if gt.shape != pred.shape:
        raise ValueError(f"Shape mismatch: gt={gt.shape}, pred={pred.shape}")

    n_features = gt.shape[1]
    if feature_names is None or len(feature_names) != n_features:
        feature_names = [f"feature_{i}" for i in range(n_features)]

    gt_mean = gt.mean(axis=0)
    pred_mean = pred.mean(axis=0)
    gt_std = gt.std(axis=0)
    pred_std = pred.std(axis=0)
    gt_c = gt - gt_mean[None, :]
    pred_c = pred - pred_mean[None, :]

    numerator = np.sum(gt_c * pred_c, axis=0)
    denominator = np.sqrt(np.sum(gt_c ** 2, axis=0) * np.sum(pred_c ** 2, axis=0)) + eps
    pcc = numerator / denominator
    pcc = np.where(np.isfinite(pcc), pcc, 0.0)
    rmse = np.sqrt(np.mean((gt - pred) ** 2, axis=0))
    mae = np.mean(np.abs(gt - pred), axis=0)

    rows = []
    for j in range(n_features):
        rows.append({
            "feature_index": int(j),
            "feature_name": str(feature_names[j]),
            "PCC": float(pcc[j]),
            "RMSE": float(rmse[j]),
            "MAE": float(mae[j]),
            "true_mean": float(gt_mean[j]),
            "pred_mean": float(pred_mean[j]),
            "true_std": float(gt_std[j]),
            "pred_std": float(pred_std[j]),
        })
    return pd.DataFrame(rows)


def compute_and_save_top_feature_metrics(gt, pred, graph_eval, feature_names, out_dir, task_name, top_n=20, rank_by="PCC"):
    ensure_dir(out_dir)
    gt = to_dense(gt).astype(np.float32)
    pred = to_dense(pred).astype(np.float32)

    all_df = compute_vectorized_feature_metrics(gt, pred, feature_names=feature_names)
    all_df = all_df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    all_csv = os.path.join(out_dir, "all_feature_metrics_basic.csv")
    all_df.to_csv(all_csv, index=False)

    if rank_by in {"RMSE", "MAE", "CMD"}:
        top_df = all_df.sort_values(rank_by, ascending=True).head(top_n).copy()
    else:
        top_df = all_df.sort_values(rank_by, ascending=False).head(top_n).copy()

    ssim_list, cmd_list, pcc_recomputed_list = [], [], []
    for _, row in top_df.iterrows():
        j = int(row["feature_index"])
        gt_j = gt[:, [j]]
        pred_j = pred[:, [j]]
        try:
            m = compute_three_metrics(gt_j.copy(), pred_j.copy(), graph_eval)
            pcc_recomputed_list.append(float(m.get("PCC", row["PCC"])))
            ssim_list.append(float(m.get("SSIM", np.nan)))
            cmd_list.append(float(m.get("CMD", np.nan)))
        except Exception as e:
            print(
                f">>> [WARN] Top-feature metric failed: task={task_name}, feature={row['feature_name']}, err={e}",
                flush=True,
            )
            pcc_recomputed_list.append(float(row["PCC"]))
            ssim_list.append(np.nan)
            cmd_list.append(np.nan)

    top_df.insert(0, "rank", np.arange(1, len(top_df) + 1))
    top_df["PCC_recomputed"] = pcc_recomputed_list
    top_df["SSIM"] = ssim_list
    top_df["CMD"] = cmd_list

    csv_path = os.path.join(out_dir, "top20_feature_metrics.csv")
    json_path = os.path.join(out_dir, "top20_feature_metrics.json")
    top_df.to_csv(csv_path, index=False)
    top_df.to_json(json_path, orient="records", force_ascii=False, indent=2)
    print(f">>> [TOP-FEATURE] saved: {csv_path}", flush=True)
    return top_df, csv_path, all_csv


def plot_top_feature_spatial_predictions(gt, pred, coords, top_df, out_dir, task_name, plot_top_n=6):
    ensure_dir(out_dir)
    gt = to_dense(gt).astype(np.float32)
    pred = to_dense(pred).astype(np.float32)
    coords = np.asarray(coords, dtype=np.float32)
    if coords.shape[0] != gt.shape[0]:
        print(
            f">>> [WARN] coords n={coords.shape[0]} != gt n={gt.shape[0]}, skip spatial plots for {task_name}.",
            flush=True,
        )
        return []

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    saved_paths = []
    plot_df = top_df.head(plot_top_n).copy()
    x = coords[:, 0]
    y = coords[:, 1]

    for _, row in plot_df.iterrows():
        rank = int(row["rank"])
        j = int(row["feature_index"])
        name = str(row["feature_name"])
        true_v = gt[:, j]
        pred_v = pred[:, j]
        err_v = np.abs(true_v - pred_v)

        fig, axes = plt.subplots(1, 3, figsize=(13, 4))
        s0 = axes[0].scatter(x, y, c=true_v, s=4)
        axes[0].set_title(f"True\n{name}")
        axes[0].invert_yaxis(); axes[0].axis("off")
        plt.colorbar(s0, ax=axes[0], fraction=0.046, pad=0.04)

        s1 = axes[1].scatter(x, y, c=pred_v, s=4)
        axes[1].set_title(f"Pred\nPCC={row['PCC']:.3f}")
        axes[1].invert_yaxis(); axes[1].axis("off")
        plt.colorbar(s1, ax=axes[1], fraction=0.046, pad=0.04)

        s2 = axes[2].scatter(x, y, c=err_v, s=4)
        axes[2].set_title("Absolute Error")
        axes[2].invert_yaxis(); axes[2].axis("off")
        plt.colorbar(s2, ax=axes[2], fraction=0.046, pad=0.04)

        fig.suptitle(f"{task_name} | rank {rank} | {name}", fontsize=11)
        plt.tight_layout()
        fig_name = f"rank{rank:02d}_{sanitize_filename(name)}_true_pred_error.png"
        fig_path = os.path.join(out_dir, fig_name)
        plt.savefig(fig_path, dpi=300)
        plt.close()
        saved_paths.append(fig_path)

    print(f">>> [SPATIAL-PLOT] saved {len(saved_paths)} plots under: {out_dir}", flush=True)
    return saved_paths


def _save_dynamic_kb_outputs(inference_output, gt, coords, feature_names, task_out_dir):
    if not ENABLE_KB_DYNAMIC_EXPORT or not isinstance(inference_output, dict):
        return None

    payload = {
        "gt": np.asarray(gt, dtype=np.float32),
        "feature_names": np.asarray(feature_names, dtype=str),
    }
    if coords is not None:
        payload["coords"] = np.asarray(coords, dtype=np.float32)

    for key in ["pred_full", "pred_main", "pred_delta", "pred_latent", "refine_gate"]:
        value = inference_output.get(key)
        if isinstance(value, np.ndarray):
            payload[key] = value.astype(np.float32, copy=False)

    kb_out = inference_output.get("kb_out", {})
    for key in [
        "global_prior",
        "local_prior",
        "local_mask",
        "local_score",
        "kb_hidden",
        "celltype_prob",
    ]:
        value = kb_out.get(key) if isinstance(kb_out, dict) else None
        if isinstance(value, np.ndarray):
            payload[f"kb_{key}"] = value.astype(np.float32, copy=False)

    gate_info = inference_output.get("gate_info", {})
    if isinstance(gate_info, dict):
        for key, value in gate_info.items():
            if not str(key).startswith("gate_"):
                continue
            if isinstance(value, np.ndarray):
                payload[f"fusion_{key}"] = value.astype(np.float32, copy=False)
            elif isinstance(value, (int, float, bool, np.number)):
                payload[f"fusion_{key}"] = np.asarray(value)

    output_path = os.path.join(task_out_dir, "dynamic_kb_outputs.npz")
    np.savez_compressed(output_path, **payload)
    manifest = {
        key: {"shape": list(value.shape), "dtype": str(value.dtype)}
        for key, value in payload.items()
        if isinstance(value, np.ndarray)
    }
    with open(os.path.join(task_out_dir, "dynamic_kb_manifest.json"), "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    print(f">>> [KB-DYNAMIC] saved: {output_path}", flush=True)
    return output_path


def run_detailed_task_analysis(
    task_name,
    repeat_idx,
    seed,
    history,
    gt,
    pred,
    graph_eval,
    adata_for_names_and_coords,
    target_task,
    inference_output=None,
):
    if not ENABLE_DETAILED_ANALYSIS:
        return {}

    task_out_dir = os.path.join(DETAILED_ANALYSIS_DIR, str(task_name), f"rep{int(repeat_idx)}_seed{int(seed)}")
    ensure_dir(task_out_dir)

    gt = to_dense(gt).astype(np.float32)
    pred = to_dense(pred).astype(np.float32)
    np.save(os.path.join(task_out_dir, "gt_full.npy"), gt)
    np.save(os.path.join(task_out_dir, "pred_full.npy"), pred)

    loss_csv, loss_fig = save_loss_history_and_plot(history=history, out_dir=task_out_dir, task_name=task_name)

    feature_names = safe_feature_names(adata_for_names_and_coords, fallback_prefix=target_task)
    if feature_names is None or len(feature_names) != gt.shape[1]:
        feature_names = [f"{target_task}_{i}" for i in range(gt.shape[1])]

    top_df, top_csv, all_feature_csv = compute_and_save_top_feature_metrics(
        gt=gt, pred=pred, graph_eval=graph_eval, feature_names=feature_names,
        out_dir=task_out_dir, task_name=task_name, top_n=TOP_FEATURE_N, rank_by="PCC",
    )

    coords = None
    if adata_for_names_and_coords is not None:
        if "spatial" in adata_for_names_and_coords.obsm.keys():
            coords = adata_for_names_and_coords.obsm["spatial"]
        elif "image_coor" in adata_for_names_and_coords.obsm.keys():
            coords = adata_for_names_and_coords.obsm["image_coor"]

    plot_paths = []
    if coords is not None:
        plot_paths = plot_top_feature_spatial_predictions(
            gt=gt, pred=pred, coords=coords, top_df=top_df,
            out_dir=os.path.join(task_out_dir, "spatial_plots"),
            task_name=task_name, plot_top_n=PLOT_TOP_FEATURE_N,
        )
    else:
        print(f">>> [WARN] No spatial coords found for {task_name}, skip spatial plots.", flush=True)

    dynamic_kb_path = _save_dynamic_kb_outputs(
        inference_output=inference_output,
        gt=gt,
        coords=coords,
        feature_names=feature_names,
        task_out_dir=task_out_dir,
    )

    return {
        "analysis_dir": task_out_dir,
        "loss_csv": loss_csv or "",
        "loss_fig": loss_fig or "",
        "top_feature_csv": top_csv or "",
        "all_feature_csv": all_feature_csv or "",
        "num_spatial_plots": int(len(plot_paths)),
        "dynamic_kb_path": dynamic_kb_path or "",
    }


def filter_kwargs_for_protocol(protocol_cls, kwargs):
    """
    Filter kwargs by protocol __init__ signature.
    This keeps compatibility when protocol signatures differ across branches.
    """
    sig = inspect.signature(protocol_cls.__init__)
    allowed = set(sig.parameters.keys()) - {"self"}
    return {k: v for k, v in kwargs.items() if k in allowed}


def build_met_task_cache_path(
    task_name,
    repeat_idx,
    seed,
    selected_sources,
    kb_weight_info,
    target_shape,
    protocol_tag="fullpartial",
    full_use_obs=None,
    partial_use_obs=None,
):
    cache_dir = os.path.join(CACHE_ROOT, "met_task_result")
    os.makedirs(cache_dir, exist_ok=True)
    cache_meta = {
        "v": 2,
        "cache_version": str(MET_TASK_CACHE_VERSION),
        "task": str(task_name),
        "repeat_idx": int(repeat_idx),
        "seed": int(seed),
        "protocol_tag": str(protocol_tag),
        "full_use_obs": None if full_use_obs is None else bool(full_use_obs),
        "partial_use_obs": None if partial_use_obs is None else bool(partial_use_obs),
        "selected_sources": sorted([str(x) for x in (selected_sources or [])]),
        "lambda_kb_eff": float(kb_weight_info.get("lambda_kb_eff", 0.0)),
        "lambda_graph_eff": float(kb_weight_info.get("lambda_graph_eff", 0.0)),
        "target_shape": [int(x) for x in target_shape],
        "epochs": int(TRAIN_EPOCHS),
        "lr": float(TRAIN_LR),
        "dropout": float(TRAIN_DROPOUT),
        "hidden_dim": int(TRAIN_HIDDEN_DIM),
        "num_layers": int(TRAIN_NUM_LAYERS),
        "fusion_alpha": float(FUSION_ALPHA),
        "kb_usage_mode": str(KB_USAGE_MODE),
        "use_met_kb_agent": bool(USE_MET_KB_AGENT),
        "met_kb_requested_mode": str(MET_KB_REQUESTED_MODE),
        "met_kb_selection_mode": str(MET_KB_SELECTION_MODE),
    }
    cache_key = hashlib.md5(json.dumps(cache_meta, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    return os.path.join(cache_dir, f"{task_name}_rep{int(repeat_idx)}_seed{int(seed)}_{cache_key}.pkl")


def build_eval_graph_from_adata(adata):
    adata = ensure_spatial_coords(adata)
    return se.pp.Build_graph(
        adata.obsm["spatial"],
        graph_type="knn",
        weighted="gaussian",
        apply_normalize="row",
        return_type="coo",
    ).tocsr()


def repair_anndata_shape_fields(adata):
    if adata is None or not hasattr(adata, "X"):
        return adata
    try:
        n_obs, n_vars = adata.X.shape
    except Exception:
        return adata
    try:
        if not hasattr(adata, "_n_obs"):
            object.__setattr__(adata, "_n_obs", int(n_obs))
        if not hasattr(adata, "_n_vars"):
            object.__setattr__(adata, "_n_vars", int(n_vars))
    except Exception:
        try:
            adata._n_obs = int(n_obs)
            adata._n_vars = int(n_vars)
        except Exception:
            pass
    return adata


def safe_anndata_shape(adata):
    repair_anndata_shape_fields(adata)
    try:
        return tuple(adata.shape)
    except Exception:
        if hasattr(adata, "X"):
            return tuple(adata.X.shape)
        raise


def repair_proteo_data_pack_anndata_shapes(pack):
    if not isinstance(pack, dict):
        return pack
    for key in ("adata1", "adata2", "adata1_gt", "adata2_prot"):
        if key in pack:
            repair_anndata_shape_fields(pack[key])
    return pack


def build_pseudospots(adata, grid_size=64, he_key="he"):
    coords = np.asarray(adata.obsm["spatial"], dtype=np.float32)
    x = coords[:, 0]
    y = coords[:, 1]
    gx = np.floor((x - x.min()) / grid_size).astype(int)
    gy = np.floor((y - y.min()) / grid_size).astype(int)
    spot_id = pd.Series(gx.astype(str) + "_" + gy.astype(str), index=adata.obs_names)
    groups = list(spot_id.groupby(spot_id).groups.items())

    X_list, he_list, spatial_list, obs_rows, spot_names = [], [], [], [], []
    X_all = adata.X
    he_all = np.asarray(adata.obsm[he_key], dtype=np.float32)
    sp_all = np.asarray(adata.obsm["spatial"], dtype=np.float32)
    for sid, obs_names in groups:
        idx = adata.obs_names.get_indexer(obs_names)
        idx = idx[idx >= 0]
        if len(idx) == 0:
            continue
        subX = X_all[idx]
        X_mean = np.asarray(subX.mean(axis=0)).reshape(-1) if sp.issparse(subX) else np.asarray(subX).mean(axis=0)
        he_mean = he_all[idx].mean(axis=0)
        spatial_mean = sp_all[idx].mean(axis=0)
        X_list.append(X_mean.astype(np.float32))
        he_list.append(he_mean.astype(np.float32))
        spatial_list.append(spatial_mean.astype(np.float32))
        obs_rows.append({"spot_id": sid, "n_cells": len(idx)})
        spot_names.append(sid)

    adata_new = ad.AnnData(X=np.vstack(X_list).astype(np.float32), obs=pd.DataFrame(obs_rows, index=spot_names), var=adata.var.copy())
    adata_new.obsm["he"] = np.vstack(he_list).astype(np.float32)
    adata_new.obsm["spatial"] = np.vstack(spatial_list).astype(np.float32)
    adata_new.obsm["image_coor"] = adata_new.obsm["spatial"].copy()
    return adata_new


def symmetric_normalize_sparse(adj):
    adj = adj.tocsr().astype(np.float32)
    deg = np.asarray(adj.sum(axis=1)).reshape(-1)
    deg_inv_sqrt = np.power(deg + 1e-12, -0.5)
    D_inv_sqrt = sp.diags(deg_inv_sqrt)
    return (D_inv_sqrt @ adj @ D_inv_sqrt).tocsr()


def build_knn_graph(features, k=7, metric="euclidean", add_self=True):
    X = np.asarray(features, dtype=np.float32)
    n = X.shape[0]
    nn = NearestNeighbors(n_neighbors=min(k + 1, n), metric=metric, algorithm="auto")
    nn.fit(X)
    distances, indices = nn.kneighbors(X)
    rows, cols, vals = [], [], []
    valid_d = distances[:, 1:].reshape(-1)
    valid_d = valid_d[np.isfinite(valid_d)]
    sigma = max(float(np.median(valid_d)) if len(valid_d) > 0 else 1.0, 1e-6)
    for i in range(n):
        neigh_idx = indices[i, 1:]
        neigh_dist = distances[i, 1:]
        weights = np.exp(-(neigh_dist ** 2) / (2.0 * sigma * sigma))
        rows.extend([i] * len(neigh_idx))
        cols.extend(neigh_idx.tolist())
        vals.extend(weights.tolist())
    adj = sp.coo_matrix((vals, (rows, cols)), shape=(n, n), dtype=np.float32)
    adj = adj.maximum(adj.T).tocsr()
    if add_self:
        adj = adj + sp.eye(n, dtype=np.float32, format="csr")
    return symmetric_normalize_sparse(adj)


def build_training_graph(adata, graph_tag):
    cache_key = stable_hash(graph_tag, safe_anndata_shape(adata), SPATIAL_K, HE_K, FUSION_ALPHA)
    cache_path = os.path.join(GRAPH_CACHE_DIR, f"{graph_tag}_{cache_key}.pkl")
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            return pickle.load(f)
    coords = np.asarray(adata.obsm["spatial"], dtype=np.float32)
    he = np.asarray(adata.obsm["he"], dtype=np.float32)
    g_spatial = build_knn_graph(coords, k=SPATIAL_K, metric="euclidean", add_self=True)
    g_he = build_knn_graph(he, k=HE_K, metric=HE_METRIC, add_self=True)
    g = symmetric_normalize_sparse(FUSION_ALPHA * g_spatial + (1.0 - FUSION_ALPHA) * g_he)
    atomic_pickle_dump(g, cache_path)
    return g


def build_training_graph_with_alpha(adata, graph_tag, alpha):
    alpha = float(alpha)
    cache_key = stable_hash(graph_tag, safe_anndata_shape(adata), SPATIAL_K, HE_K, HE_METRIC, alpha)
    cache_path = os.path.join(PROTEO_GRAPH_CACHE_DIR, f"{graph_tag}_{cache_key}.pkl")
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            return pickle.load(f)
    coords = np.asarray(adata.obsm["spatial"], dtype=np.float32)
    he = np.asarray(adata.obsm["he"], dtype=np.float32)
    g_spatial = build_knn_graph(coords, k=SPATIAL_K, metric="euclidean", add_self=True)
    g_he = build_knn_graph(he, k=HE_K, metric=HE_METRIC, add_self=True)
    g = symmetric_normalize_sparse(alpha * g_spatial + (1.0 - alpha) * g_he)
    atomic_pickle_dump(g, cache_path)
    return g


# ============================================================================
# 3) Proteogenomics pipeline (task1~task4, in-process)
# ============================================================================
def normalize_symbol(x):
    if x is None:
        return None
    x = str(x).strip()
    if x == "" or x.lower() == "nan":
        return None
    return x.upper()


def split_symbol_field(x):
    if x is None:
        return []
    x = str(x).strip()
    if x == "" or x.lower() == "nan":
        return []
    parts = re.split(r"[|,;/\s]+", x)
    out = []
    for p in parts:
        pp = normalize_symbol(p)
        if pp is not None:
            out.append(pp)
    return out


def normalize_matrix_rows(m, eps=1e-8):
    m = np.asarray(m, dtype=np.float32)
    row_sum = m.sum(axis=1, keepdims=True)
    row_sum[row_sum < eps] = 1.0
    return m / row_sum


def load_hgnc_maps(hgnc_file):
    df = pd.read_csv(hgnc_file, sep="\t", low_memory=False)
    symbol_col = "approved_symbol" if "approved_symbol" in df.columns else "symbol"
    alias_to_symbol = {}
    ensembl_to_symbol = {}
    alias_cols = [c for c in ["alias_symbol", "prev_symbol"] if c in df.columns]
    ensembl_col = "ensembl_gene_id" if "ensembl_gene_id" in df.columns else None

    for _, row in df.iterrows():
        canon = normalize_symbol(row[symbol_col])
        if canon is None:
            continue
        alias_to_symbol[canon] = canon
        for c in alias_cols:
            for a in split_symbol_field(row[c]):
                alias_to_symbol[a] = canon
        if ensembl_col is not None:
            ens = normalize_symbol(row[ensembl_col])
            if ens is not None:
                ensembl_to_symbol[ens] = canon

    manual_override = {
        "SMA": "ACTA2",
        "PAN-CYTOKERATIN": "KRT18",
        "PANC": "KRT18",
        "CK": "KRT18",
        "CD11C": "ITGAX",
        "CD56": "NCAM1",
        "HLA-DR": "HLA-DRA",
        "CD20": "MS4A1",
        "CD8": "CD8A",
        "CD3E": "CD3E",
    }
    for k, v in manual_override.items():
        alias_to_symbol[k] = v
    return alias_to_symbol, ensembl_to_symbol


def load_uniprot_maps(uniprot_file, alias_to_symbol):
    df = pd.read_csv(uniprot_file, sep="\t", low_memory=False)
    acc_col = [c for c in ["Entry", "Accession", "From"] if c in df.columns][0]
    gene_primary_col = [c for c in ["Gene Names (primary)", "Gene Names  (primary )"] if c in df.columns]
    gene_primary_col = gene_primary_col[0] if gene_primary_col else None
    gene_names_col = [c for c in ["Gene Names", "Gene names"] if c in df.columns]
    gene_names_col = gene_names_col[0] if gene_names_col else None
    prot_name_col = next((c for c in ["Protein names", "Protein names "] if c in df.columns), None)

    acc_to_symbols = defaultdict(set)
    alias_to_symbol_aug = dict(alias_to_symbol)
    protein_name_to_accs = defaultdict(set)

    for _, row in df.iterrows():
        acc = normalize_symbol(row[acc_col])
        if acc is None:
            continue
        all_names = []
        if gene_primary_col:
            all_names.extend(split_symbol_field(row[gene_primary_col]))
        if gene_names_col:
            all_names.extend(split_symbol_field(row[gene_names_col]))
        canon_list = []
        for name in all_names:
            canon = alias_to_symbol_aug.get(name, name)
            canon_list.append(canon)
            alias_to_symbol_aug[name] = canon
        for canon in canon_list:
            acc_to_symbols[acc].add(canon)
        if prot_name_col is not None:
            pname = str(row[prot_name_col]).strip()
            if pname and pname.lower() != "nan":
                protein_name_to_accs[normalize_symbol(pname)].add(acc)

    return acc_to_symbols, alias_to_symbol_aug, protein_name_to_accs


def load_reactome_gene_sets(reactome_uniprot_file, reactome_ensembl_file, acc_to_symbols, ensembl_to_symbol):
    pathway_to_genes = defaultdict(set)
    if reactome_uniprot_file and os.path.exists(reactome_uniprot_file):
        df_u = pd.read_csv(
            reactome_uniprot_file, sep="\t", header=None, low_memory=False,
            encoding="latin1", quoting=3, on_bad_lines="skip"
        )
        for _, row in df_u.iterrows():
            acc = normalize_symbol(row.iloc[0])
            pathway_name = str(row.iloc[3]).strip() if len(row) > 3 else None
            species = str(row.iloc[5]).strip() if len(row) > 5 else "Homo sapiens"
            if pathway_name is None or "Homo sapiens" not in species:
                continue
            for g in acc_to_symbols.get(acc, []):
                pathway_to_genes[pathway_name].add(g)
    if reactome_ensembl_file and os.path.exists(reactome_ensembl_file):
        df_e = pd.read_csv(
            reactome_ensembl_file, sep="\t", header=None, low_memory=False,
            encoding="latin1", quoting=3, on_bad_lines="skip"
        )
        for _, row in df_e.iterrows():
            ens = normalize_symbol(row.iloc[0])
            pathway_name = str(row.iloc[3]).strip() if len(row) > 3 else None
            species = str(row.iloc[5]).strip() if len(row) > 5 else "Homo sapiens"
            if pathway_name is None or "Homo sapiens" not in species:
                continue
            g = ensembl_to_symbol.get(ens, None)
            if g is not None:
                pathway_to_genes[pathway_name].add(g)
    gene_to_pathways = defaultdict(set)
    for pw, genes in pathway_to_genes.items():
        for g in genes:
            gene_to_pathways[g].add(pw)
    return pathway_to_genes, gene_to_pathways


def _normalize_ncbi_gene_id(x):
    if x is None:
        return None
    s = str(x).strip()
    if not s or s.lower() == "nan":
        return None
    if s.endswith(".0"):
        s = s[:-2]
    return s if s.isdigit() else None


def load_hgnc_ncbi_map(hgnc_file):
    if not hgnc_file or not os.path.exists(hgnc_file):
        return {}
    df = pd.read_csv(hgnc_file, sep="\t", low_memory=False)
    symbol_col = "approved_symbol" if "approved_symbol" in df.columns else "symbol"
    entrez_col = next((c for c in ["entrez_id", "entrez_gene_id", "ncbi_gene_id"] if c in df.columns), None)
    if entrez_col is None:
        entrez_col = next((c for c in df.columns if "entrez" in c.lower() or "ncbi" in c.lower()), None)
    if entrez_col is None:
        return {}
    ncbi_to_symbol = {}
    for _, row in df.iterrows():
        gene_id = _normalize_ncbi_gene_id(row.get(entrez_col))
        symbol = normalize_symbol(row.get(symbol_col))
        if gene_id and symbol:
            ncbi_to_symbol[gene_id] = symbol
    return ncbi_to_symbol


def load_kegg_gene_sets(kegg_pathways_file, kegg_gene_pathway_file, hgnc_file):
    pathway_to_name = {}
    if kegg_pathways_file and os.path.exists(kegg_pathways_file):
        with open(kegg_pathways_file, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 2:
                    continue
                pid = parts[0].split(":")[-1].strip()
                name = parts[1].strip()
                if pid and name:
                    pathway_to_name[pid] = name

    ncbi_to_symbol = load_hgnc_ncbi_map(hgnc_file)
    pathway_to_genes = defaultdict(set)
    if not kegg_gene_pathway_file or not os.path.exists(kegg_gene_pathway_file):
        return pathway_to_genes, defaultdict(set)

    with open(kegg_gene_pathway_file, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = re.split(r"\s+", line.strip())
            if len(parts) < 2:
                continue
            a, b = parts[0], parts[1]
            gene_tok = a if "hsa:" in a and "path:" not in a else b
            path_tok = b if "path:" in b or "hsa0" in b else a
            gene_id = gene_tok.split(":")[-1].strip()
            pathway_id = path_tok.split(":")[-1].strip()
            symbol = ncbi_to_symbol.get(gene_id)
            if not symbol or not pathway_id:
                continue
            pathway_label = "KEGG:" + pathway_to_name.get(pathway_id, pathway_id)
            pathway_to_genes[pathway_label].add(symbol)

    gene_to_pathways = defaultdict(set)
    for pw, genes in pathway_to_genes.items():
        for g in genes:
            gene_to_pathways[g].add(pw)
    return pathway_to_genes, gene_to_pathways


def load_uniprot_ensembl_protein_map(uniprot_file, alias_to_symbol):
    if not uniprot_file or not os.path.exists(uniprot_file):
        return defaultdict(set)
    df = pd.read_csv(uniprot_file, sep="\t", low_memory=False)
    gene_primary_col = next((c for c in ["Gene Names (primary)", "Gene Names  (primary )"] if c in df.columns), None)
    gene_names_col = next((c for c in ["Gene Names", "Gene names"] if c in df.columns), None)
    ensembl_cols = [c for c in df.columns if "ensembl" in c.lower()]
    ensp_to_symbols = defaultdict(set)
    if not ensembl_cols:
        return ensp_to_symbols
    for _, row in df.iterrows():
        symbols = []
        if gene_primary_col:
            symbols.extend(split_symbol_field(row.get(gene_primary_col)))
        if gene_names_col:
            symbols.extend(split_symbol_field(row.get(gene_names_col)))
        symbols = [alias_to_symbol.get(s, s) for s in symbols]
        symbols = [s for s in symbols if s]
        if not symbols:
            continue
        joined = " ".join(str(row.get(c, "")) for c in ensembl_cols)
        for ensp in re.findall(r"ENSP\d+(?:\.\d+)?", joined, flags=re.IGNORECASE):
            ensp_to_symbols[ensp.upper().split(".")[0]].update(symbols)
    return ensp_to_symbols


def add_string_protein_graph(
    G_prot,
    string_file,
    ensp_to_symbols,
    gene_to_prot_idxs,
    weight=0.5,
    min_score=700,
    max_edges=2000000,
):
    if not string_file or not os.path.exists(string_file) or not ensp_to_symbols:
        return 0
    opener = gzip.open if str(string_file).endswith(".gz") else open
    added = 0
    with opener(string_file, "rt", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            try:
                score = float(parts[2])
            except Exception:
                continue
            if score < float(min_score):
                continue
            p1 = parts[0].split(".")[-1].upper()
            p2 = parts[1].split(".")[-1].upper()
            genes1 = ensp_to_symbols.get(p1, set())
            genes2 = ensp_to_symbols.get(p2, set())
            if not genes1 or not genes2:
                continue
            w = float(weight) * float(score) / 1000.0
            for g1 in genes1:
                for g2 in genes2:
                    for i in gene_to_prot_idxs.get(g1, []):
                        for j in gene_to_prot_idxs.get(g2, []):
                            if i != j:
                                _add_symmetric_edge(G_prot, i, j, w)
                                added += 1
                                if added >= int(max_edges):
                                    return added
    return added


def load_cellmarker_prior(cellmarker_file, alias_to_symbol):
    if not cellmarker_file or (not os.path.exists(cellmarker_file)):
        return defaultdict(set)
    df = pd.read_excel(cellmarker_file) if cellmarker_file.endswith(".xlsx") else pd.read_csv(cellmarker_file, low_memory=False)
    cell_to_markers = defaultdict(set)
    cols = list(df.columns)
    col_map = {c.lower().replace("_", ""): c for c in cols}
    cell_col = col_map.get("cellname", col_map.get("celltype", None))
    marker_col = col_map.get("symbol", col_map.get("marker", None))
    if not cell_col or not marker_col:
        return cell_to_markers
    for _, row in df.iterrows():
        cell_type = str(row[cell_col]).strip().lower()
        symbol_str = str(row[marker_col])
        if cell_type == "nan" or not cell_type or symbol_str == "nan" or not symbol_str:
            continue
        markers = re.split(r"[\[\]\s,;]+", symbol_str)
        for m in markers:
            m = m.strip().upper()
            if m:
                cell_to_markers[cell_type].add(alias_to_symbol.get(m, m))
    return cell_to_markers


def load_proteinatlas_celltype_prior(proteinatlas_file, alias_to_symbol):
    if not proteinatlas_file or (not os.path.exists(proteinatlas_file)):
        return defaultdict(set)
    df = pd.read_csv(proteinatlas_file, sep="\t", low_memory=False)
    col_map = {c.lower().replace("_", "").replace(" ", ""): c for c in df.columns}
    gene_col = (
        col_map.get("genename")
        or col_map.get("gene")
        or col_map.get("genesymbol")
        or next((c for c in df.columns if "gene" in c.lower() and "name" in c.lower()), None)
    )
    cell_col = (
        col_map.get("celltype")
        or col_map.get("singlecelltype")
        or col_map.get("cellline")
        or next((c for c in df.columns if "cell" in c.lower() and "type" in c.lower()), None)
    )
    level_col = col_map.get("level") or col_map.get("expression") or col_map.get("reliability")
    if gene_col is None or cell_col is None:
        return defaultdict(set)
    accepted_levels = {"high", "medium", "enhanced", "supported", "approved", "reliable"}
    cell_to_markers = defaultdict(set)
    for _, row in df.iterrows():
        gene = alias_to_symbol.get(normalize_symbol(row.get(gene_col)), normalize_symbol(row.get(gene_col)))
        cell_type = normalize_text(row.get(cell_col))
        if gene is None or cell_type is None:
            continue
        if level_col is not None:
            level = str(row.get(level_col, "")).strip().lower()
            if level and level not in accepted_levels and "not detected" in level:
                continue
        cell_to_markers[cell_type.lower()].add(gene)
    return cell_to_markers


def _find_col(df, candidates, contains_any=None):
    norm = {c.lower().replace("_", "").replace(" ", ""): c for c in df.columns}
    for cand in candidates:
        key = cand.lower().replace("_", "").replace(" ", "")
        if key in norm:
            return norm[key]
    if contains_any:
        for c in df.columns:
            low = c.lower()
            if all(tok in low for tok in contains_any):
                return c
    return None


def _add_symmetric_edge(mat, i, j, w=1.0):
    if i is None or j is None or i == j:
        return
    mat[i, j] += float(w)
    mat[j, i] += float(w)


def add_dorothea_gene_graph(G_gene, dorothea_file, alias_to_symbol, gene_to_idx, weight=0.5):
    if not dorothea_file or (not os.path.exists(dorothea_file)):
        return 0
    df = pd.read_csv(dorothea_file, sep="\t", low_memory=False)
    src_col = _find_col(df, ["source", "tf", "transcription_factor", "tf_symbol"], contains_any=["source"])
    tgt_col = _find_col(df, ["target", "target_gene", "target_symbol"], contains_any=["target"])
    conf_col = _find_col(df, ["confidence", "level", "dorothea_confidence"])
    if src_col is None or tgt_col is None:
        return 0
    conf_weight = {"a": 1.0, "b": 0.8, "c": 0.6, "d": 0.4, "e": 0.2}
    added = 0
    for _, row in df.iterrows():
        src = alias_to_symbol.get(normalize_symbol(row.get(src_col)), normalize_symbol(row.get(src_col)))
        tgt = alias_to_symbol.get(normalize_symbol(row.get(tgt_col)), normalize_symbol(row.get(tgt_col)))
        i, j = gene_to_idx.get(src), gene_to_idx.get(tgt)
        if i is None or j is None or i == j:
            continue
        cw = 1.0
        if conf_col is not None:
            cw = conf_weight.get(str(row.get(conf_col, "")).strip().lower()[:1], 1.0)
        _add_symmetric_edge(G_gene, i, j, float(weight) * cw)
        added += 1
    return added


def add_omnipath_graphs(G_gene, G_prot, omnipath_file, alias_to_symbol, gene_to_idx, gene_to_prot_idxs, weight=0.5):
    if not omnipath_file or (not os.path.exists(omnipath_file)):
        return 0, 0
    df = pd.read_csv(omnipath_file, sep="\t", low_memory=False)
    src_col = _find_col(df, ["source", "genesymbol_intercell_source", "source_genesymbol"])
    tgt_col = _find_col(df, ["target", "genesymbol_intercell_target", "target_genesymbol"])
    if src_col is None or tgt_col is None:
        return 0, 0
    gene_edges = 0
    prot_edges = 0
    for _, row in df.iterrows():
        src = alias_to_symbol.get(normalize_symbol(row.get(src_col)), normalize_symbol(row.get(src_col)))
        tgt = alias_to_symbol.get(normalize_symbol(row.get(tgt_col)), normalize_symbol(row.get(tgt_col)))
        gi, gj = gene_to_idx.get(src), gene_to_idx.get(tgt)
        if gi is not None and gj is not None and gi != gj:
            _add_symmetric_edge(G_gene, gi, gj, weight)
            gene_edges += 1
        for pi in gene_to_prot_idxs.get(src, []):
            for pj in gene_to_prot_idxs.get(tgt, []):
                if pi != pj:
                    _add_symmetric_edge(G_prot, pi, pj, weight)
                    prot_edges += 1
    return gene_edges, prot_edges


def add_corum_protein_graph(G_prot, corum_file, alias_to_symbol, protein_to_idx, gene_to_prot_idxs, weight=0.5):
    if not corum_file or (not os.path.exists(corum_file)):
        return 0
    df = pd.read_csv(corum_file, sep="\t", low_memory=False)
    subunit_cols = [c for c in df.columns if "subunit" in c.lower()]
    if not subunit_cols:
        return 0
    added = 0
    for _, row in df.iterrows():
        hit_idxs = set()
        for col in subunit_cols:
            for tok in split_symbol_field(row.get(col)):
                canon = alias_to_symbol.get(tok, tok)
                if canon in protein_to_idx:
                    hit_idxs.add(protein_to_idx[canon])
                hit_idxs.update(gene_to_prot_idxs.get(canon, []))
        hit_idxs = sorted(hit_idxs)
        for a in range(len(hit_idxs)):
            for b in range(a + 1, len(hit_idxs)):
                _add_symmetric_edge(G_prot, hit_idxs[a], hit_idxs[b], weight)
                added += 1
    return added


def build_multi_relation_kb_upgraded(
    protein_names,
    gene_names,
    hgnc_file,
    uniprot_file,
    reactome_uniprot_file,
    reactome_ensembl_file,
    cellmarker_file=None,
    dorothea_file=None,
    omnipath_file=None,
    corum_file=None,
    proteinatlas_file=None,
    kegg_pathways_file=None,
    kegg_gene_pathway_file=None,
    string_file=None,
    direct_weight=1.0,
    module_weight=0.35,
    celltype_weight=2.0,
    gene_graph_weight=1.0,
    prot_graph_weight=1.0,
    gg_celltype_weight=0.5,
    pp_celltype_weight=0.5,
    dorothea_graph_weight=0.5,
    omnipath_graph_weight=0.5,
    corum_graph_weight=0.5,
    proteinatlas_celltype_weight=0.5,
    kegg_pathway_weight=0.5,
    string_graph_weight=0.5,
):
    alias_to_symbol, ensembl_to_symbol = load_hgnc_maps(hgnc_file)
    acc_to_symbols, alias_to_symbol, protein_name_to_accs = load_uniprot_maps(uniprot_file, alias_to_symbol)
    pathway_to_genes, gene_to_pathways = load_reactome_gene_sets(
        reactome_uniprot_file, reactome_ensembl_file, acc_to_symbols, ensembl_to_symbol
    )
    kegg_pathway_to_genes, kegg_gene_to_pathways = load_kegg_gene_sets(
        kegg_pathways_file, kegg_gene_pathway_file, hgnc_file
    )
    if kegg_pathway_to_genes:
        for pw, genes in kegg_pathway_to_genes.items():
            pathway_to_genes[pw].update(genes)
        for g, pathways in kegg_gene_to_pathways.items():
            gene_to_pathways[g].update(pathways)
    cellmarker_cell_to_markers = load_cellmarker_prior(cellmarker_file, alias_to_symbol)
    proteinatlas_cell_to_markers = load_proteinatlas_celltype_prior(proteinatlas_file, alias_to_symbol)
    cell_to_markers = defaultdict(set)
    for ct, markers in cellmarker_cell_to_markers.items():
        cell_to_markers[ct].update(markers)
    if proteinatlas_cell_to_markers:
        for ct, markers in proteinatlas_cell_to_markers.items():
            cell_to_markers[ct].update(markers)

    protein_names_arr = np.asarray(protein_names).astype(str)
    gene_names_arr = np.asarray(gene_names).astype(str)
    P, G = len(protein_names_arr), len(gene_names_arr)

    gene_canon, gene_to_idx = [], {}
    for i, g in enumerate(gene_names_arr):
        gc = alias_to_symbol.get(normalize_symbol(g), normalize_symbol(g))
        gene_canon.append(gc)
        gene_to_idx[gc] = i

    protein_canon, protein_to_gene_set = [], []
    for p in protein_names_arr:
        pn = normalize_symbol(p)
        pc = alias_to_symbol.get(pn, pn)
        protein_canon.append(pc)
        gset = set()
        if pc in protein_name_to_accs:
            for acc in protein_name_to_accs[pc]:
                gset.update(acc_to_symbols.get(acc, set()))
        if pn in acc_to_symbols:
            gset.update(acc_to_symbols[pn])
        if pc in acc_to_symbols:
            gset.update(acc_to_symbols[pc])
        if pc in gene_to_idx:
            gset.add(pc)
        protein_to_gene_set.append(gset)
    protein_to_idx = {}
    gene_to_prot_idxs = defaultdict(list)
    for p_idx, pc in enumerate(protein_canon):
        protein_to_idx[pc] = p_idx
        for g in protein_to_gene_set[p_idx]:
            gene_to_prot_idxs[g].append(p_idx)

    M_pg_direct = np.zeros((P, G), dtype=np.float32)
    M_pg_module = np.zeros((P, G), dtype=np.float32)
    M_pg_celltype = np.zeros((P, G), dtype=np.float32)
    M_pg_proteinatlas = np.zeros((P, G), dtype=np.float32)

    for p_idx, gset in enumerate(protein_to_gene_set):
        for g in gset:
            if g in gene_to_idx:
                M_pg_direct[p_idx, gene_to_idx[g]] = 1.0

    for p_idx, gset in enumerate(protein_to_gene_set):
        pathways = set()
        for g in gset:
            pathways.update(gene_to_pathways.get(g, set()))
        gene_scores = defaultdict(float)
        for pw in pathways:
            genes_in_pw = [g for g in pathway_to_genes.get(pw, set()) if g in gene_to_idx]
            if not genes_in_pw:
                continue
            w_pw = 1.0 / np.sqrt(len(genes_in_pw))
            if str(pw).startswith("KEGG:"):
                w_pw *= float(kegg_pathway_weight)
            for g in genes_in_pw:
                gene_scores[g] += w_pw
        for g, score in gene_scores.items():
            M_pg_module[p_idx, gene_to_idx[g]] = score

    if cellmarker_cell_to_markers:
        for p_idx, pc in enumerate(protein_canon):
            relevant_cell_types = [ct for ct, markers in cellmarker_cell_to_markers.items() if pc in markers]
            for ct in relevant_cell_types:
                for g in cellmarker_cell_to_markers[ct]:
                    if g in gene_to_idx:
                        M_pg_celltype[p_idx, gene_to_idx[g]] += 1.0
    if proteinatlas_cell_to_markers:
        for p_idx, pc in enumerate(protein_canon):
            relevant_cell_types = [ct for ct, markers in proteinatlas_cell_to_markers.items() if pc in markers]
            for ct in relevant_cell_types:
                for g in proteinatlas_cell_to_markers[ct]:
                    if g in gene_to_idx:
                        M_pg_proteinatlas[p_idx, gene_to_idx[g]] += 1.0

    M_pg = (
        direct_weight * M_pg_direct
        + module_weight * M_pg_module
        + celltype_weight * M_pg_celltype
        + proteinatlas_celltype_weight * M_pg_proteinatlas
    )
    M_pg = normalize_matrix_rows(M_pg)
    M_gp = (
        direct_weight * M_pg_direct.T
        + module_weight * M_pg_module.T
        + celltype_weight * M_pg_celltype.T
        + proteinatlas_celltype_weight * M_pg_proteinatlas.T
    )
    M_gp = normalize_matrix_rows(M_gp)

    G_gene = np.zeros((G, G), dtype=np.float32)
    for i in range(G):
        pwi = gene_to_pathways.get(gene_canon[i], set())
        if not pwi:
            continue
        for j in range(i + 1, G):
            pwj = gene_to_pathways.get(gene_canon[j], set())
            inter = pwi.intersection(pwj)
            if inter:
                shared = sum(float(kegg_pathway_weight) if str(pw).startswith("KEGG:") else 1.0 for pw in inter)
                score = gene_graph_weight * shared / np.sqrt(max(len(pwi), 1) * max(len(pwj), 1))
                G_gene[i, j] += score
                G_gene[j, i] += score

    if cell_to_markers:
        gene_to_celltypes = defaultdict(set)
        for ct, markers in cell_to_markers.items():
            for g in markers:
                gene_to_celltypes[g].add(ct)
        for i in range(G):
            cti = gene_to_celltypes.get(gene_canon[i], set())
            if not cti:
                continue
            for j in range(i + 1, G):
                ctj = gene_to_celltypes.get(gene_canon[j], set())
                inter = cti.intersection(ctj)
                if inter:
                    score = gg_celltype_weight * len(inter) / np.sqrt(max(len(cti), 1) * max(len(ctj), 1))
                    G_gene[i, j] += score
                    G_gene[j, i] += score
    dorothea_gene_edges = add_dorothea_gene_graph(
        G_gene,
        dorothea_file=dorothea_file,
        alias_to_symbol=alias_to_symbol,
        gene_to_idx=gene_to_idx,
        weight=dorothea_graph_weight,
    )

    G_prot = np.zeros((P, P), dtype=np.float32)
    for i in range(P):
        pwi = set()
        for g in protein_to_gene_set[i]:
            pwi.update(gene_to_pathways.get(g, set()))
        for j in range(i + 1, P):
            pwj = set()
            for g in protein_to_gene_set[j]:
                pwj.update(gene_to_pathways.get(g, set()))
            inter = pwi.intersection(pwj)
            if inter:
                shared = sum(float(kegg_pathway_weight) if str(pw).startswith("KEGG:") else 1.0 for pw in inter)
                score = prot_graph_weight * shared / np.sqrt(max(len(pwi), 1) * max(len(pwj), 1))
                G_prot[i, j] += score
                G_prot[j, i] += score

    if cell_to_markers:
        protein_to_celltypes = defaultdict(set)
        for pc in protein_canon:
            for ct, markers in cell_to_markers.items():
                if pc in markers:
                    protein_to_celltypes[pc].add(ct)
        for i in range(P):
            ci = protein_to_celltypes.get(protein_canon[i], set())
            if not ci:
                continue
            for j in range(i + 1, P):
                cj = protein_to_celltypes.get(protein_canon[j], set())
                inter = ci.intersection(cj)
                if inter:
                    score = pp_celltype_weight * len(inter) / np.sqrt(max(len(ci), 1) * max(len(cj), 1))
                    G_prot[i, j] += score
                    G_prot[j, i] += score
    omnipath_gene_edges, omnipath_protein_edges = add_omnipath_graphs(
        G_gene,
        G_prot,
        omnipath_file=omnipath_file,
        alias_to_symbol=alias_to_symbol,
        gene_to_idx=gene_to_idx,
        gene_to_prot_idxs=gene_to_prot_idxs,
        weight=omnipath_graph_weight,
    )
    corum_protein_edges = add_corum_protein_graph(
        G_prot,
        corum_file=corum_file,
        alias_to_symbol=alias_to_symbol,
        protein_to_idx=protein_to_idx,
        gene_to_prot_idxs=gene_to_prot_idxs,
        weight=corum_graph_weight,
    )
    ensp_to_symbols = load_uniprot_ensembl_protein_map(uniprot_file, alias_to_symbol)
    string_protein_edges = add_string_protein_graph(
        G_prot,
        string_file=string_file,
        ensp_to_symbols=ensp_to_symbols,
        gene_to_prot_idxs=gene_to_prot_idxs,
        weight=string_graph_weight,
        min_score=PROTEO_KB_STRING_MIN_SCORE,
        max_edges=PROTEO_KB_STRING_MAX_EDGES,
    )
    G_gene = normalize_adjacency_with_selfloop(G_gene)
    G_prot = normalize_adjacency_with_selfloop(G_prot)

    stats = {
        "num_proteins": P,
        "num_genes": G,
        "resolved_proteins": int(np.sum(M_pg_direct.sum(axis=1) > 0)),
        "num_direct_links": int((M_pg_direct > 0).sum()),
        "num_module_links": int((M_pg_module > 0).sum()),
        "num_celltype_links": int((M_pg_celltype > 0).sum()),
        "num_proteinatlas_links": int((M_pg_proteinatlas > 0).sum()),
        "gene_graph_edges": int((np.sum(G_gene > 0) - G)),
        "protein_graph_edges": int((np.sum(G_prot > 0) - P)),
        "dorothea_gene_edges": int(dorothea_gene_edges),
        "omnipath_gene_edges": int(omnipath_gene_edges),
        "omnipath_protein_edges": int(omnipath_protein_edges),
        "corum_protein_edges": int(corum_protein_edges),
        "string_protein_edges": int(string_protein_edges),
        "proteinatlas_celltype_count": int(len(proteinatlas_cell_to_markers)),
        "kegg_pathway_count": int(len(kegg_pathway_to_genes)),
        "kegg_gene_memberships": int(sum(len(v) for v in kegg_pathway_to_genes.values())),
    }

    forced_genes = []
    for g_idx, g in enumerate(gene_names_arr):
        if any(m[:, g_idx].sum() > 0 for m in [M_pg_direct, M_pg_module, M_pg_celltype, M_pg_proteinatlas]):
            forced_genes.append(g)

    return M_pg, M_gp, G_gene, G_prot, stats, forced_genes


def _hash_name_array(arr):
    arr = np.asarray(arr).astype(str)
    h = hashlib.md5()
    for x in arr:
        h.update(x.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def _safe_mtime(path):
    if path and os.path.exists(path):
        return int(os.path.getmtime(path))
    return None


def load_or_build_multi_relation_kb_cached(
    protein_names,
    gene_names,
    hgnc_file,
    uniprot_file,
    reactome_uniprot_file,
    reactome_ensembl_file,
    cellmarker_file=None,
    dorothea_file=None,
    omnipath_file=None,
    corum_file=None,
    proteinatlas_file=None,
    kegg_pathways_file=None,
    kegg_gene_pathway_file=None,
    string_file=None,
    direct_weight=1.0,
    module_weight=0.35,
    celltype_weight=2.0,
    gene_graph_weight=1.0,
    prot_graph_weight=1.0,
    gg_celltype_weight=0.5,
    pp_celltype_weight=0.5,
    dorothea_graph_weight=0.5,
    omnipath_graph_weight=0.5,
    corum_graph_weight=0.5,
    proteinatlas_celltype_weight=0.5,
    kegg_pathway_weight=0.5,
    string_graph_weight=0.5,
    cache_dir=None,
    cache_prefix="kb",
):
    if cache_dir is None:
        cache_dir = PROTEO_KB_CACHE_DIR
    os.makedirs(cache_dir, exist_ok=True)
    cache_meta = {
        "version": "core4_v3_kegg_string",
        "cache_prefix": cache_prefix,
        "protein_hash": _hash_name_array(protein_names),
        "gene_hash": _hash_name_array(gene_names),
        "weights": {
            "direct_weight": float(direct_weight),
            "module_weight": float(module_weight),
            "celltype_weight": float(celltype_weight),
            "gene_graph_weight": float(gene_graph_weight),
            "prot_graph_weight": float(prot_graph_weight),
            "gg_celltype_weight": float(gg_celltype_weight),
            "pp_celltype_weight": float(pp_celltype_weight),
            "dorothea_graph_weight": float(dorothea_graph_weight),
            "omnipath_graph_weight": float(omnipath_graph_weight),
            "corum_graph_weight": float(corum_graph_weight),
            "proteinatlas_celltype_weight": float(proteinatlas_celltype_weight),
            "kegg_pathway_weight": float(kegg_pathway_weight),
            "string_graph_weight": float(string_graph_weight),
        },
        "source_mtime": {
            "hgnc": _safe_mtime(hgnc_file),
            "uniprot": _safe_mtime(uniprot_file),
            "reactome_uniprot": _safe_mtime(reactome_uniprot_file),
            "reactome_ensembl": _safe_mtime(reactome_ensembl_file),
            "cellmarker": _safe_mtime(cellmarker_file),
            "dorothea": _safe_mtime(dorothea_file),
            "omnipath": _safe_mtime(omnipath_file),
            "corum": _safe_mtime(corum_file),
            "proteinatlas": _safe_mtime(proteinatlas_file),
            "kegg_pathways": _safe_mtime(kegg_pathways_file),
            "kegg_gene_pathway": _safe_mtime(kegg_gene_pathway_file),
            "string": _safe_mtime(string_file),
        },
    }
    cache_key = hashlib.md5(json.dumps(cache_meta, sort_keys=True).encode("utf-8")).hexdigest()
    cache_path = os.path.join(cache_dir, f"{cache_prefix}_{cache_key}.pkl")

    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            payload = pickle.load(f)
        return (
            payload["protein_gene_prior"],
            payload["gene_protein_prior"],
            payload["gene_gene_kb_graph"],
            payload["protein_protein_kb_graph"],
            payload["stats"],
            payload["forced_genes"],
        )

    M_pg, M_gp, G_gene, G_prot, stats, forced_genes = build_multi_relation_kb_upgraded(
        protein_names=protein_names,
        gene_names=gene_names,
        hgnc_file=hgnc_file,
        uniprot_file=uniprot_file,
        reactome_uniprot_file=reactome_uniprot_file,
        reactome_ensembl_file=reactome_ensembl_file,
        cellmarker_file=cellmarker_file,
        dorothea_file=dorothea_file,
        omnipath_file=omnipath_file,
        corum_file=corum_file,
        proteinatlas_file=proteinatlas_file,
        kegg_pathways_file=kegg_pathways_file,
        kegg_gene_pathway_file=kegg_gene_pathway_file,
        string_file=string_file,
        direct_weight=direct_weight,
        module_weight=module_weight,
        celltype_weight=celltype_weight,
        gene_graph_weight=gene_graph_weight,
        prot_graph_weight=prot_graph_weight,
        gg_celltype_weight=gg_celltype_weight,
        pp_celltype_weight=pp_celltype_weight,
        dorothea_graph_weight=dorothea_graph_weight,
        omnipath_graph_weight=omnipath_graph_weight,
        corum_graph_weight=corum_graph_weight,
        proteinatlas_celltype_weight=proteinatlas_celltype_weight,
        kegg_pathway_weight=kegg_pathway_weight,
        string_graph_weight=string_graph_weight,
    )
    with open(cache_path, "wb") as f:
        pickle.dump(
            {
                "cache_meta": cache_meta,
                "protein_gene_prior": M_pg,
                "gene_protein_prior": M_gp,
                "gene_gene_kb_graph": G_gene,
                "protein_protein_kb_graph": G_prot,
                "stats": stats,
                "forced_genes": forced_genes,
            },
            f,
        )
    return M_pg, M_gp, G_gene, G_prot, stats, forced_genes


def reduce_he_dim(adata, he_key="he", out_dim=64, seed=0):
    he = np.asarray(adata.obsm[he_key], dtype=np.float32)
    if he.shape[1] <= out_dim:
        return adata
    pca = PCA(n_components=out_dim, random_state=seed)
    adata.obsm[he_key] = pca.fit_transform(he).astype(np.float32)
    return adata


def select_top_var_features_with_forced_genes(adata, top_k=500, forced_genes=None):
    X = to_dense(adata.X).astype(np.float32)
    genes = np.asarray(adata.var_names).astype(str)
    forced_genes = set([] if forced_genes is None else [str(g) for g in forced_genes])
    if X.shape[1] <= top_k:
        return adata.copy()
    var = X.var(axis=0)
    idx_var = np.argsort(var)[::-1]
    forced_idx = [i for i, g in enumerate(genes) if g in forced_genes]
    keep = list(forced_idx)
    for i in idx_var:
        if i not in keep:
            keep.append(i)
        if len(keep) >= top_k:
            break
    keep = np.array(sorted(set(keep)))
    return adata[:, keep].copy()


def select_random_gene_features(adata, top_k=500, random_seed=2026):
    genes = np.asarray(adata.var_names).astype(str)
    n_genes = len(genes)
    if n_genes <= top_k:
        return adata.copy()
    rng = np.random.RandomState(int(random_seed))
    keep = np.sort(rng.choice(n_genes, size=int(top_k), replace=False))
    return adata[:, keep].copy()


def _load_ordered_gene_names(path):
    path = str(path or "").strip()
    if path == "" or not os.path.exists(path):
        return []
    try:
        df = pd.read_csv(path, sep=None, engine="python")
        if df.shape[1] == 0:
            return []
        preferred = ["feature_name", "gene", "gene_name", "symbol", "var_name"]
        col = next((c for c in preferred if c in df.columns), df.columns[0])
        return [str(x).strip() for x in df[col].values if str(x).strip() and str(x).strip().lower() != "nan"]
    except Exception:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return [line.strip().split(",")[0].split("\t")[0] for line in f if line.strip()]


def select_beneficial_gene_features(adata, top_k=500, forced_genes=None):
    genes = np.asarray(adata.var_names).astype(str)
    gene_to_idx = {g: i for i, g in enumerate(genes)}
    ordered = _load_ordered_gene_names(PROTEO_GENE_BENEFICIAL_LIST)
    keep = []
    seen = set()
    for g in ordered:
        if g in gene_to_idx and g not in seen:
            keep.append(gene_to_idx[g])
            seen.add(g)
        if len(keep) >= int(top_k):
            break
    if len(keep) < int(top_k):
        fallback = select_top_var_features_with_forced_genes(
            adata,
            top_k=min(len(genes), int(top_k) + max(0, len(genes) - len(keep))),
            forced_genes=forced_genes,
        )
        fallback_genes = np.asarray(fallback.var_names).astype(str)
        for g in fallback_genes:
            if g in gene_to_idx and g not in seen:
                keep.append(gene_to_idx[g])
                seen.add(g)
            if len(keep) >= int(top_k):
                break
    keep = np.array(sorted(set(keep[: int(top_k)])))
    if len(keep) == 0:
        return select_top_var_features_with_forced_genes(adata, top_k=top_k, forced_genes=forced_genes)
    return adata[:, keep].copy()


def select_proteo_gene_features(adata, top_k=500, forced_genes=None):
    mode = str(PROTEO_GENE_SELECT_MODE).lower()
    if mode == "random":
        selected = select_random_gene_features(
            adata,
            top_k=top_k,
            random_seed=PROTEO_GENE_RANDOM_SEED,
        )
    elif mode == "beneficial":
        selected = select_beneficial_gene_features(
            adata,
            top_k=top_k,
            forced_genes=forced_genes,
        )
    else:
        selected = select_top_var_features_with_forced_genes(
            adata,
            top_k=top_k,
            forced_genes=forced_genes,
        )
    print(
        f">>> [GENE-SELECT] mode={mode} top_k={int(top_k)} random_seed={int(PROTEO_GENE_RANDOM_SEED)} "
        f"beneficial_list={PROTEO_GENE_BENEFICIAL_LIST or 'none'} "
        f"input_dim={adata.n_vars} output_dim={selected.n_vars}",
        flush=True,
    )
    return selected


def load_or_build_proteo_raw_processed_data():
    if os.path.exists(PROTEO_RAW_SLICE1_CACHE) and os.path.exists(PROTEO_RAW_SLICE2_CACHE):
        return sc.read_h5ad(PROTEO_RAW_SLICE1_CACHE), sc.read_h5ad(PROTEO_RAW_SLICE2_CACHE)
    raise RuntimeError(
        "Proteogenomics raw cache files not found. "
        f"Please prepare: {PROTEO_RAW_SLICE1_CACHE} and {PROTEO_RAW_SLICE2_CACHE}"
    )


def refine_proteo_slice1(adata):
    adata = ensure_spatial_coords(adata)
    if "he" not in adata.obsm:
        adata.obsm["he"] = np.zeros((adata.n_obs, 1), dtype=np.float32)
    if PROTEO_USE_PSEUDOSPOT:
        adata = build_pseudospots(adata, grid_size=PROTEO_GRID_SIZE, he_key="he")
    elif adata.n_obs > PROTEO_SUBSAMPLE_N1:
        idx = np.sort(np.random.RandomState(0).choice(adata.n_obs, PROTEO_SUBSAMPLE_N1, replace=False))
        adata = adata[idx].copy()
    adata = reduce_he_dim(adata, he_key="he", out_dim=PROTEO_HE_DIM, seed=0)
    adata.obsm["image_coor"] = adata.obsm["spatial"].copy()
    adata.obsm["protein"] = to_dense(adata.X).astype(np.float32)
    return adata


def refine_proteo_slice2(adata, forced_genes=None):
    adata = ensure_spatial_coords(adata)
    if "he" not in adata.obsm:
        adata.obsm["he"] = np.zeros((adata.n_obs, 1), dtype=np.float32)
    if PROTEO_USE_PSEUDOSPOT:
        adata = build_pseudospots(adata, grid_size=PROTEO_GRID_SIZE, he_key="he")
    elif adata.n_obs > PROTEO_SUBSAMPLE_N2:
        idx = np.sort(np.random.RandomState(1).choice(adata.n_obs, PROTEO_SUBSAMPLE_N2, replace=False))
        adata = adata[idx].copy()
    adata = reduce_he_dim(adata, he_key="he", out_dim=PROTEO_HE_DIM, seed=1)
    adata = select_proteo_gene_features(adata, top_k=PROTEO_GENE_TOPK, forced_genes=forced_genes)
    adata.obsm["image_coor"] = adata.obsm["spatial"].copy()
    return adata


def prepare_proteogenomics_data():
    align_key = stable_hash(
        "proteo_align_v1",
        PROTEO_DATA_ROOT,
        PROTEO_SAMPLE1,
        PROTEO_SAMPLE2,
        PROTEO_GRID_SIZE,
        PROTEO_HE_DIM,
        PROTEO_GENE_TOPK,
        PROTEO_GENE_SELECT_MODE,
        PROTEO_GENE_RANDOM_SEED,
        PROTEO_GENE_BENEFICIAL_LIST,
        PROTEO_GENE_BENEFICIAL_TAG,
        _safe_mtime(PROTEO_GENE_BENEFICIAL_LIST),
        PROTEO_GENE_LATENT_DIM,
        PROTEO_BASE_GRAPH_ALPHA,
        PROTEO_USE_PSEUDOSPOT,
        _safe_mtime(PROTEO_RAW_SLICE1_CACHE),
        _safe_mtime(PROTEO_RAW_SLICE2_CACHE),
        _safe_mtime(PROTEO_HGNC_PATH),
        _safe_mtime(PROTEO_UNIPROT_PATH),
        _safe_mtime(PROTEO_REACTOME_UNIPROT_PATH),
        _safe_mtime(PROTEO_REACTOME_ENSEMBL_PATH),
        _safe_mtime(PROTEO_CELLMARKER_PATH),
        _safe_mtime(PROTEO_KEGG_GENE_PATHWAY_PATH),
        _safe_mtime(PROTEO_STRING_PATH),
    )
    align_cache_path = os.path.join(PROTEO_CACHE_DIR, f"aligned_{align_key}.pkl")
    if os.path.exists(align_cache_path):
        with open(align_cache_path, "rb") as f:
            return repair_proteo_data_pack_anndata_shapes(pickle.load(f))

    if all(os.path.exists(p) for p in [PROTEO_ADATA1_REFINED_CACHE, PROTEO_ADATA2_REFINED_CACHE, PROTEO_GRAPH1_REFINED_CACHE, PROTEO_GRAPH2_REFINED_CACHE]):
        adata1 = sc.read_h5ad(PROTEO_ADATA1_REFINED_CACHE)
        adata2 = sc.read_h5ad(PROTEO_ADATA2_REFINED_CACHE)
        with open(PROTEO_GRAPH1_REFINED_CACHE, "rb") as f:
            graph1 = pickle.load(f)
        with open(PROTEO_GRAPH2_REFINED_CACHE, "rb") as f:
            graph2 = pickle.load(f)
    else:
        adata1_raw, adata2_raw = load_or_build_proteo_raw_processed_data()
        _, _, _, _, _, forced_genes = load_or_build_multi_relation_kb_cached(
            protein_names=adata1_raw.var_names,
            gene_names=adata2_raw.var_names,
            hgnc_file=PROTEO_HGNC_PATH,
            uniprot_file=PROTEO_UNIPROT_PATH,
            reactome_uniprot_file=PROTEO_REACTOME_UNIPROT_PATH,
            reactome_ensembl_file=PROTEO_REACTOME_ENSEMBL_PATH,
            cellmarker_file=PROTEO_CELLMARKER_PATH,
            proteinatlas_file=PROTEO_PROTEINATLAS_PATH,
            kegg_pathways_file=PROTEO_KEGG_PATHWAYS_PATH,
            kegg_gene_pathway_file=PROTEO_KEGG_GENE_PATHWAY_PATH,
            string_file=PROTEO_STRING_PATH,
            direct_weight=PROTEO_KB_DIRECT_WEIGHT,
            module_weight=PROTEO_KB_MODULE_WEIGHT,
            celltype_weight=PROTEO_KB_CELLTYPE_WEIGHT,
            gene_graph_weight=PROTEO_KB_GENE_GRAPH_WEIGHT,
            prot_graph_weight=PROTEO_KB_PROT_GRAPH_WEIGHT,
            gg_celltype_weight=PROTEO_KB_GG_CELLTYPE_WEIGHT,
            pp_celltype_weight=PROTEO_KB_PP_CELLTYPE_WEIGHT,
            proteinatlas_celltype_weight=PROTEO_KB_PROTEINATLAS_CELLTYPE_WEIGHT,
            kegg_pathway_weight=PROTEO_KB_KEGG_PATHWAY_WEIGHT,
            string_graph_weight=PROTEO_KB_STRING_GRAPH_WEIGHT,
            cache_dir=PROTEO_KB_CACHE_DIR,
            cache_prefix="proteo_warmup",
        )
        adata1 = refine_proteo_slice1(adata1_raw)
        adata2 = refine_proteo_slice2(adata2_raw, forced_genes=forced_genes)
        graph1 = build_training_graph_with_alpha(adata1, "proteo_s1_base", PROTEO_BASE_GRAPH_ALPHA)
        graph2 = build_training_graph_with_alpha(adata2, "proteo_s2_base", PROTEO_BASE_GRAPH_ALPHA)
        adata1.uns.clear()
        adata2.uns.clear()
        atomic_write_h5ad(adata1, PROTEO_ADATA1_REFINED_CACHE)
        atomic_write_h5ad(adata2, PROTEO_ADATA2_REFINED_CACHE)
        atomic_pickle_dump(graph1, PROTEO_GRAPH1_REFINED_CACHE)
        atomic_pickle_dump(graph2, PROTEO_GRAPH2_REFINED_CACHE)

    adata1_gt = sc.read_h5ad(os.path.join(PROTEO_DATA_ROOT, PROTEO_SAMPLE1, "transcriptome", "adata.h5ad"))
    adata1_gt.var_names_make_unique()
    adata1_gt = ensure_spatial_coords(adata1_gt)
    sc.pp.filter_cells(adata1_gt, min_counts=10)
    sc.pp.normalize_total(adata1_gt, target_sum=1e4)
    sc.pp.log1p(adata1_gt)
    sc.pp.scale(adata1_gt)
    if PROTEO_USE_PSEUDOSPOT:
        if "he" not in adata1_gt.obsm:
            adata1_gt.obsm["he"] = np.zeros((adata1_gt.n_obs, 1), dtype=np.float32)
        adata1_gt = build_pseudospots(adata1_gt, grid_size=PROTEO_GRID_SIZE, he_key="he")

    adata2_prot = sc.read_h5ad(os.path.join(PROTEO_DATA_ROOT, PROTEO_SAMPLE2, "proteome", "adata_codex.h5ad"))
    adata2_prot.var_names_make_unique()
    adata2_prot = ensure_spatial_coords(adata2_prot)
    sc.pp.scale(adata2_prot)
    if PROTEO_USE_PSEUDOSPOT:
        if "he" not in adata2_prot.obsm:
            adata2_prot.obsm["he"] = np.zeros((adata2_prot.n_obs, 1), dtype=np.float32)
        adata2_prot = build_pseudospots(adata2_prot, grid_size=PROTEO_GRID_SIZE, he_key="he")

    common_gene = np.intersect1d(adata2.var_names.astype(str), adata1_gt.var_names.astype(str))
    adata2 = adata2[:, common_gene].copy()
    adata1_gt = adata1_gt[:, common_gene].copy()
    common_prot = np.intersect1d(adata1.var_names.astype(str), adata2_prot.var_names.astype(str))
    adata1 = adata1[:, common_prot].copy()
    adata2_prot = adata2_prot[:, common_prot].copy()

    common_obs1 = np.intersect1d(adata1.obs_names.astype(str), adata1_gt.obs_names.astype(str))
    idx1 = adata1.obs_names.get_indexer(common_obs1)
    idx1_gt = adata1_gt.obs_names.get_indexer(common_obs1)
    adata1 = adata1[idx1].copy()
    adata1_gt = adata1_gt[idx1_gt].copy()
    graph1 = graph1[idx1][:, idx1]

    common_obs2 = np.intersect1d(adata2.obs_names.astype(str), adata2_prot.obs_names.astype(str))
    idx2 = adata2.obs_names.get_indexer(common_obs2)
    idx2_gt = adata2_prot.obs_names.get_indexer(common_obs2)
    adata2 = adata2[idx2].copy()
    adata2_prot = adata2_prot[idx2_gt].copy()
    graph2 = graph2[idx2][:, idx2]
    adata1.obsm["protein"] = to_dense(adata1.X).astype(np.float32)

    if all(os.path.exists(p) for p in [PROTEO_GENE_PCA_COMP_CACHE, PROTEO_GENE_PCA_MEAN_CACHE, PROTEO_GENE1_LAT_CACHE, PROTEO_GENE2_LAT_CACHE]):
        pca_components = np.load(PROTEO_GENE_PCA_COMP_CACHE)
        pca_mean = np.load(PROTEO_GENE_PCA_MEAN_CACHE)
        gene1_gt_latent = np.load(PROTEO_GENE1_LAT_CACHE)
        gene2_latent = np.load(PROTEO_GENE2_LAT_CACHE)
    else:
        gene1_full = to_dense(adata1_gt.X).astype(np.float32)
        gene2_full = to_dense(adata2.X).astype(np.float32)
        gene_train_mat = np.vstack([gene1_full, gene2_full])
        gene_latent_dim = min(PROTEO_GENE_LATENT_DIM, gene_train_mat.shape[1], gene_train_mat.shape[0] - 1)
        max_fit_cells = 20000
        if gene_train_mat.shape[0] > max_fit_cells:
            idx_fit = np.random.RandomState(0).choice(gene_train_mat.shape[0], max_fit_cells, replace=False)
            gene_fit_mat = gene_train_mat[idx_fit]
        else:
            gene_fit_mat = gene_train_mat
        pca = PCA(n_components=gene_latent_dim, svd_solver="randomized", random_state=0)
        pca.fit(gene_fit_mat)
        pca_components = pca.components_.astype(np.float32)
        pca_mean = pca.mean_.astype(np.float32)
        gene1_gt_latent = pca.transform(gene1_full).astype(np.float32)
        gene2_latent = pca.transform(gene2_full).astype(np.float32)
        np.save(PROTEO_GENE_PCA_COMP_CACHE, pca_components)
        np.save(PROTEO_GENE_PCA_MEAN_CACHE, pca_mean)
        np.save(PROTEO_GENE1_LAT_CACHE, gene1_gt_latent)
        np.save(PROTEO_GENE2_LAT_CACHE, gene2_latent)

    graph_eval_gene_s2 = build_eval_graph_from_adata(adata2)
    graph_eval_prot_s1 = build_eval_graph_from_adata(adata1)
    pack = {
        "adata1": adata1,
        "adata2": adata2,
        "adata1_gt": adata1_gt,
        "adata2_prot": adata2_prot,
        "graph1_base": graph1,
        "graph2_base": graph2,
        "gene_pca_components": pca_components,
        "gene_pca_mean": pca_mean,
        "gene1_gt_latent": gene1_gt_latent,
        "gene2_latent": gene2_latent,
        "graph_eval_gene_s2": graph_eval_gene_s2,
        "graph_eval_prot_s1": graph_eval_prot_s1,
    }
    atomic_pickle_dump(pack, align_cache_path)
    return pack


_proteo_graph_mem = {}
_proteo_kb_mem = {}
_proteo_ct_prior_cache_mem = {}
_proteo_hgnc_maps_cache = None
_proteo_agent_selection_cache_mem = {}


def _proteo_selection_cache_key(task_name, source_modalities, target_task, data_pack):
    return stable_hash(
        "proteo_agent_selection_v1",
        str(task_name),
        "|".join([str(x) for x in (source_modalities or [])]),
        str(target_task),
        str(PROTEO_KB_REQUESTED_MODE),
        str(PROTEO_KB_SELECTION_MODE),
        str(bool(PROTEO_KB_USE_DATA_PROFILE)),
        tuple(data_pack["adata1"].var_names.astype(str)),
        tuple(data_pack["adata2"].var_names.astype(str)),
    )


def _proteo_selection_cache_paths(orchestration_dir, cache_key):
    base = os.path.join(orchestration_dir, f"agent_selection_cache_{cache_key}")
    return base + ".json", base + ".pkl"


def _write_proteo_cached_report(report, orchestration_dir, task_id):
    if not orchestration_dir:
        return report
    os.makedirs(orchestration_dir, exist_ok=True)
    report = copy.deepcopy(report or {})
    report["task_spec"] = dict(report.get("task_spec", {}) or {})
    report["task_spec"]["task_id"] = str(task_id)
    report_name = f"kb_orchestration_{task_id}.json"
    trace_name = f"agent_trace_{task_id}.json"
    report_path = os.path.join(orchestration_dir, report_name)
    trace_path = os.path.join(orchestration_dir, trace_name)
    report["report_path"] = report_path
    report["agent_trace_path"] = trace_path
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    with open(trace_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "task_spec": report.get("task_spec", {}),
                "data_profile": report.get("data_profile", {}),
                "selection": report.get("selection", {}),
                "selection_budget": report.get("selection_budget", {}),
                "final_selected_sources": report.get("final_selected_sources", []),
                "agent_reasoning_steps": report.get("agent_reasoning_steps", []),
                "agent_self_critique": report.get("agent_self_critique", ""),
                "agent_failure": report.get("agent_failure", {}),
                "agent_weight_application": report.get("agent_weight_application", {}),
                "builder_kwargs_effective": report.get("builder_kwargs_effective", {}),
                "validation": report.get("validation", {}),
                "builder_stats": report.get("builder_stats", {}),
                "selection_cache_reused": report.get("selection_cache_reused", False),
                "selection_cache_key": report.get("selection_cache_key", ""),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    return report


def get_or_build_proteo_fused_graph(adata, alpha, slice_tag, base_graph=None):
    alpha = float(alpha)
    key = (slice_tag, alpha, safe_anndata_shape(adata))
    if key in _proteo_graph_mem:
        return _proteo_graph_mem[key]
    if base_graph is not None and abs(alpha - PROTEO_BASE_GRAPH_ALPHA) < 1e-12:
        _proteo_graph_mem[key] = base_graph
        return base_graph
    graph_tag = f"proteo_{slice_tag}_{PROTEO_MODE_TAG}_alpha{alpha:.3f}".replace(".", "p")
    g = build_training_graph_with_alpha(adata, graph_tag, alpha)
    _proteo_graph_mem[key] = g
    return g


def get_local_proteo_source_status():
    return {
        "hgnc": os.path.exists(PROTEO_HGNC_PATH),
        "uniprot": os.path.exists(PROTEO_UNIPROT_PATH),
        "reactome": os.path.exists(PROTEO_REACTOME_UNIPROT_PATH) and os.path.exists(PROTEO_REACTOME_ENSEMBL_PATH),
        "cellmarker": os.path.exists(PROTEO_CELLMARKER_PATH),
        "dorothea": os.path.exists(PROTEO_DOROTHEA_PATH),
        "omnipath": os.path.exists(PROTEO_OMNIPATH_PATH),
        "corum": os.path.exists(PROTEO_CORUM_PATH),
        "proteinatlas": os.path.exists(PROTEO_PROTEINATLAS_PATH),
        "kegg": os.path.exists(PROTEO_KEGG_GENE_PATHWAY_PATH),
        "string": os.path.exists(PROTEO_STRING_PATH),
        "hmdb": False,
        "chebi": False,
    }


def infer_source_modalities_for_task(target_task, full_use_obs, partial_use_obs):
    src = ["he"]
    use_obs = bool(full_use_obs or partial_use_obs)
    if target_task == "gene" and use_obs:
        src.append("protein")
    if target_task == "protein" and use_obs:
        src.append("gene")
    return src


def resolve_proteo_kb_by_config(data_pack, kb_cfg, selected_sources=None):
    merged = {
        "direct_weight": PROTEO_KB_DIRECT_WEIGHT,
        "module_weight": PROTEO_KB_MODULE_WEIGHT,
        "celltype_weight": PROTEO_KB_CELLTYPE_WEIGHT,
        "gene_graph_weight": PROTEO_KB_GENE_GRAPH_WEIGHT,
        "prot_graph_weight": PROTEO_KB_PROT_GRAPH_WEIGHT,
        "gg_celltype_weight": PROTEO_KB_GG_CELLTYPE_WEIGHT,
        "pp_celltype_weight": PROTEO_KB_PP_CELLTYPE_WEIGHT,
        "dorothea_graph_weight": PROTEO_KB_DOROTHEA_GRAPH_WEIGHT,
        "omnipath_graph_weight": PROTEO_KB_OMNIPATH_GRAPH_WEIGHT,
        "corum_graph_weight": PROTEO_KB_CORUM_GRAPH_WEIGHT,
        "proteinatlas_celltype_weight": PROTEO_KB_PROTEINATLAS_CELLTYPE_WEIGHT,
        "kegg_pathway_weight": PROTEO_KB_KEGG_PATHWAY_WEIGHT,
        "string_graph_weight": PROTEO_KB_STRING_GRAPH_WEIGHT,
    }
    merged.update(kb_cfg or {})

    selected_set = set([str(x).strip().lower() for x in (selected_sources or ["hgnc", "uniprot", "reactome", "cellmarker"]) if str(x).strip()])
    selected_set.update(["hgnc", "uniprot"])
    selected_key = tuple(sorted(selected_set))

    merged_key = (
        tuple(sorted((k, float(v)) for k, v in merged.items())),
        selected_key,
        tuple(data_pack["adata1"].var_names.astype(str)),
        tuple(data_pack["adata2"].var_names.astype(str)),
    )
    if merged_key in _proteo_kb_mem:
        return _proteo_kb_mem[merged_key], merged_key

    ret = load_or_build_multi_relation_kb_cached(
        protein_names=data_pack["adata1"].var_names,
        gene_names=data_pack["adata2"].var_names,
        hgnc_file=PROTEO_HGNC_PATH if "hgnc" in selected_set else None,
        uniprot_file=PROTEO_UNIPROT_PATH if "uniprot" in selected_set else None,
        reactome_uniprot_file=PROTEO_REACTOME_UNIPROT_PATH if "reactome" in selected_set else None,
        reactome_ensembl_file=PROTEO_REACTOME_ENSEMBL_PATH if "reactome" in selected_set else None,
        cellmarker_file=PROTEO_CELLMARKER_PATH if "cellmarker" in selected_set else None,
        dorothea_file=PROTEO_DOROTHEA_PATH if "dorothea" in selected_set else None,
        omnipath_file=PROTEO_OMNIPATH_PATH if "omnipath" in selected_set else None,
        corum_file=PROTEO_CORUM_PATH if "corum" in selected_set else None,
        proteinatlas_file=PROTEO_PROTEINATLAS_PATH if "proteinatlas" in selected_set else None,
        kegg_pathways_file=PROTEO_KEGG_PATHWAYS_PATH if "kegg" in selected_set else None,
        kegg_gene_pathway_file=PROTEO_KEGG_GENE_PATHWAY_PATH if "kegg" in selected_set else None,
        string_file=PROTEO_STRING_PATH if "string" in selected_set else None,
        direct_weight=merged["direct_weight"],
        module_weight=merged["module_weight"],
        celltype_weight=merged["celltype_weight"],
        gene_graph_weight=merged["gene_graph_weight"],
        prot_graph_weight=merged["prot_graph_weight"],
        gg_celltype_weight=merged["gg_celltype_weight"],
        pp_celltype_weight=merged["pp_celltype_weight"],
        dorothea_graph_weight=merged["dorothea_graph_weight"],
        omnipath_graph_weight=merged["omnipath_graph_weight"],
        corum_graph_weight=merged["corum_graph_weight"],
        proteinatlas_celltype_weight=merged["proteinatlas_celltype_weight"],
        kegg_pathway_weight=merged["kegg_pathway_weight"],
        string_graph_weight=merged["string_graph_weight"],
        cache_dir=PROTEO_KB_CACHE_DIR,
        cache_prefix="proteo_trainpanel",
    )
    _proteo_kb_mem[merged_key] = ret
    return ret, merged_key


def resolve_proteo_formal_rule_baseline(
    task_name,
    data_pack,
    repeat_idx,
    seed,
    orchestration_dir,
    local_status=None,
    kb_cfg=None,
    selected_sources=None,
):
    baseline_sources = [
        str(s).strip().lower()
        for s in (selected_sources or PROTEO_RULE_BASELINE_SOURCES)
        if str(s).strip()
    ]
    if local_status is not None:
        baseline_sources = [s for s in baseline_sources if local_status.get(s, False)]
    if not baseline_sources:
        baseline_sources = ["hgnc", "uniprot", "reactome", "cellmarker"]
    kb_tuple, kb_key = resolve_proteo_kb_by_config(
        data_pack=data_pack,
        kb_cfg=copy.deepcopy(kb_cfg or PROTEO_FINAL_KB_CFG),
        selected_sources=baseline_sources,
    )
    builder_stats = {}
    try:
        builder_stats = copy.deepcopy(kb_tuple[4] or {})
    except Exception:
        builder_stats = {}
    report_path = os.path.join(
        orchestration_dir,
        f"kb_orchestration_{task_name}_rep{int(repeat_idx)}_seed{int(seed)}.json",
    )
    report = {
        "requested_mode": "rule-only",
        "selection_mode_requested": "rule_only",
        "use_data_profile": False,
        "selection_policy": "fixed_formal_rule_baseline",
        "selection": {
            "mode": "rule_based_no_agent",
            "agent_decision": {},
        },
        "final_selected_sources": baseline_sources,
        "builder_stats": builder_stats,
        "report_path": report_path,
        "agent_trace_path": "",
        "validation": {"ok": True},
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    return kb_tuple, kb_key, baseline_sources, report


_formal_rule_baseline_df_cache = None
_formal_rule_baseline_csv_cache = None


def _resolve_formal_rule_baseline_csv():
    candidates = []
    if PROTEO_FORMAL_RULE_BASELINE_CSV:
        candidates.append(PROTEO_FORMAL_RULE_BASELINE_CSV)
    if PROTEO_FORMAL_RULE_BASELINE_DIR:
        candidates.append(os.path.join(PROTEO_FORMAL_RULE_BASELINE_DIR, "unified_metrics_raw.csv"))
    for path in candidates:
        if path and os.path.exists(path):
            return path
    return ""


def _load_formal_rule_baseline_df():
    global _formal_rule_baseline_df_cache, _formal_rule_baseline_csv_cache
    csv_path = _resolve_formal_rule_baseline_csv()
    if not csv_path:
        return None, ""
    if _formal_rule_baseline_df_cache is not None and _formal_rule_baseline_csv_cache == csv_path:
        return _formal_rule_baseline_df_cache, csv_path
    df = pd.read_csv(csv_path)
    required = {"task", "seed", "PCC", "SSIM", "CMD", "RMSE"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Formal rule baseline CSV missing columns: {missing}; path={csv_path}")
    _formal_rule_baseline_df_cache = df
    _formal_rule_baseline_csv_cache = csv_path
    return df, csv_path


def load_imported_formal_rule_baseline_row(task_idx, task_cfg, repeat_idx, seed, orchestration_dir):
    if not PROTEO_IMPORT_FORMAL_RULE_BASELINE:
        return None
    df, csv_path = _load_formal_rule_baseline_df()
    if df is None:
        print(
            ">>> [FORMAL-RULE] import requested but baseline CSV was not found; "
            "falling back to local rule training.",
            flush=True,
        )
        return None

    task_name = str(task_cfg["task"])
    seed_int = int(seed)
    matched = df[(df["task"].astype(str) == task_name) & (df["seed"].astype(int) == seed_int)]
    if matched.empty:
        print(
            f">>> [FORMAL-RULE] no imported row for task={task_name}, seed={seed_int}; "
            "falling back to local rule training.",
            flush=True,
        )
        return None
    src = matched.iloc[0].to_dict()
    selected_sources = [
        str(s).strip().lower()
        for s in str(src.get("selected_sources", ";".join(PROTEO_RULE_BASELINE_SOURCES))).replace(",", ";").split(";")
        if str(s).strip()
    ]
    if not selected_sources:
        selected_sources = list(PROTEO_RULE_BASELINE_SOURCES)

    report_path = os.path.join(
        orchestration_dir,
        f"kb_orchestration_{task_name}_formal_rule_imported_rep{int(repeat_idx)}_seed{seed_int}.json",
    )
    report = {
        "requested_mode": "rule-only",
        "selection_mode_requested": "rule_only",
        "use_data_profile": False,
        "selection_policy": "imported_formal_rule_baseline",
        "formal_rule_baseline_csv": csv_path,
        "selection": {
            "mode": "imported_formal_rule_baseline",
            "agent_decision": {},
        },
        "final_selected_sources": selected_sources,
        "metrics": {k: float(src[k]) for k in ["PCC", "SSIM", "CMD", "RMSE"]},
        "builder_stats": {},
        "report_path": report_path,
        "agent_trace_path": "",
        "validation": {"ok": True},
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    def _coerce_bool(value, default):
        if pd.isna(value):
            return bool(default)
        if isinstance(value, (bool, np.bool_)):
            return bool(value)
        s = str(value).strip().lower()
        if s in {"1", "true", "t", "yes", "y", "on"}:
            return True
        if s in {"0", "false", "f", "no", "n", "off"}:
            return False
        return bool(default)

    row = {
        "experiment_id": str(src.get("experiment_id", "E2_unified8_detailed")),
        "repeat_idx": int(src.get("repeat_idx", repeat_idx)),
        "seed": seed_int,
        "task": task_name,
        "task_phase": "rule_round",
        "task_order": int(task_idx),
        "target_task": str(src.get("target_task", task_cfg["target_task"])),
        "full_use_obs": _coerce_bool(src.get("full_use_obs", task_cfg["full_use_obs"]), task_cfg["full_use_obs"]),
        "partial_use_obs": _coerce_bool(src.get("partial_use_obs", task_cfg["partial_use_obs"]), task_cfg["partial_use_obs"]),
        "graph_alpha": float(src.get("graph_alpha", task_cfg.get("graph_alpha", PROTEO_BASE_GRAPH_ALPHA))),
        "PCC": float(src["PCC"]),
        "SSIM": float(src["SSIM"]),
        "CMD": float(src["CMD"]),
        "RMSE": float(src["RMSE"]),
        "selected_sources": ";".join(selected_sources),
        "kb_requested_mode": "rule-only",
        "kb_use_data_profile": False,
        "kb_select_mode": "imported_formal_rule_baseline",
        "kb_selection_mode_requested": "rule_only",
        "agent_provider": "",
        "agent_model": "",
        "agent_confidence": np.nan,
        "agent_weight_strategy": "",
        "agent_failure_stage": "",
        "agent_failure_message": "",
        "kb_num_direct_links": np.nan,
        "kb_num_module_links": np.nan,
        "kb_kegg_pathway_count": np.nan,
        "kb_string_protein_edges": np.nan,
        "kb_orchestration_report": report_path,
        "agent_trace_path": "",
        "rule_baseline_source_path": csv_path,
        "domain": str(src.get("domain", "proteogenomics")),
    }
    print(
        f">>> [FORMAL-RULE] imported {task_name} seed={seed_int} "
        f"PCC={row['PCC']:.4f} SSIM={row['SSIM']:.4f} CMD={row['CMD']:.4f} RMSE={row['RMSE']:.4f}",
        flush=True,
    )
    return row, selected_sources, report


def adapt_proteo_builder_kwargs_with_agent_weights(builder_kwargs, selection, selected_sources, task_spec):
    builder_kwargs = dict(builder_kwargs or {})
    selection = dict(selection or {})
    agent_decision = selection.get("agent_decision", {}) if isinstance(selection.get("agent_decision", {}), dict) else {}
    raw_weights = agent_decision.get("source_weights", {}) if isinstance(agent_decision.get("source_weights", {}), dict) else {}
    raw_usage_modes = agent_decision.get("source_usage_modes", {}) if isinstance(agent_decision.get("source_usage_modes", {}), dict) else {}
    raw_source_actions = agent_decision.get("source_actions", {}) if isinstance(agent_decision.get("source_actions", {}), dict) else {}
    selected = [str(s).strip().lower() for s in (selected_sources or []) if str(s).strip()]
    task_spec = dict(task_spec or {})
    source_modalities = {
        str(x).strip().lower()
        for x in (task_spec.get("source_modalities", []) or [])
        if str(x).strip()
    }
    target_modality = str(task_spec.get("target_modality", "")).strip().lower()
    min_scale = float(os.getenv("KNOWLEDGE_AGENT_WEIGHT_MIN_SCALE", "0.70"))
    max_scale = float(os.getenv("KNOWLEDGE_AGENT_WEIGHT_MAX_SCALE", "1.25"))
    shrink = float(os.getenv("KNOWLEDGE_AGENT_WEIGHT_SHRINKAGE", "0.55"))

    numeric_weights = {}
    for src in selected:
        try:
            numeric_weights[src] = float(raw_weights.get(src, 0.0))
        except Exception:
            numeric_weights[src] = 0.0
    usage_modes = {src: str(raw_usage_modes.get(src, "full")).strip().lower() or "full" for src in selected}
    source_actions = {src: str(raw_source_actions.get(src, "keep")).strip().lower() or "keep" for src in selected}
    positive = [v for v in numeric_weights.values() if v > 0]
    if not positive:
        return builder_kwargs, {
            "enabled": False,
            "reason": "agent_returned_no_positive_source_weights",
            "source_weights": raw_weights,
        }

    mean_w = float(np.mean(positive))

    def factor(*sources):
        vals = [numeric_weights.get(str(s).lower(), 0.0) for s in sources if str(s).lower() in selected]
        vals = [v for v in vals if v > 0]
        if not vals:
            return 1.0
        raw_scale = float(np.mean(vals)) / max(mean_w, 1e-8)
        stabilized = 1.0 + shrink * (raw_scale - 1.0)
        return float(np.clip(stabilized, min_scale, max_scale))

    def usage_multiplier(source_group: str, param: str) -> float:
        group_sources = [str(s).lower() for s in str(source_group).split("+")]
        active_modes = [usage_modes.get(s, "full") for s in group_sources if s in usage_modes]
        active_actions = [source_actions.get(s, "keep") for s in group_sources if s in source_actions]
        if not active_modes:
            return 1.0
        if any(m == "disabled" for m in active_modes):
            return 0.0
        if any(a == "remove" for a in active_actions):
            return 0.0

        structural_params = {"gene_graph_weight", "prot_graph_weight", "dorothea_graph_weight", "omnipath_graph_weight", "corum_graph_weight", "kegg_pathway_weight", "string_graph_weight"}
        prior_params = {"direct_weight", "module_weight", "celltype_weight", "gg_celltype_weight", "pp_celltype_weight", "proteinatlas_celltype_weight"}

        scales = []
        for mode in active_modes:
            if mode == "full":
                scales.append(1.0)
            elif mode == "low_weight_full":
                scales.append(0.75)
            elif mode == "graph_only":
                scales.append(0.95 if param in structural_params else 0.15)
            elif mode == "prior_only":
                scales.append(0.95 if param in prior_params else 0.15)
            elif mode == "warmup_only":
                scales.append(0.40)
            else:
                scales.append(1.0)
        scale = float(np.mean(scales))
        if any(a in {"downweight", "warmup_only"} for a in active_actions):
            scale *= 0.85
        return float(np.clip(scale, 0.0, max_scale))

    def template_multiplier(param: str) -> float:
        if target_modality == "gene" and "protein" not in source_modalities:
            conservative = {
                "direct_weight": 1.01,
                "module_weight": 1.00,
                "gene_graph_weight": 1.00,
                "prot_graph_weight": 1.00,
                "celltype_weight": 1.00,
                "gg_celltype_weight": 1.00,
                "pp_celltype_weight": 1.00,
                "kegg_pathway_weight": 1.00,
                "proteinatlas_celltype_weight": 1.00,
            }
            return conservative.get(param, 1.0)
        if target_modality == "protein" and "gene" not in source_modalities:
            hints = {
                "direct_weight": 1.01,
                "celltype_weight": 1.02,
                "gg_celltype_weight": 1.02,
                "pp_celltype_weight": 1.02,
                "proteinatlas_celltype_weight": 1.04 if "proteinatlas" in selected else 1.00,
                "kegg_pathway_weight": 1.00,
            }
            return hints.get(param, 1.0)
        if target_modality == "gene" and "protein" in source_modalities:
            hints = {
                "direct_weight": 1.02,
                "celltype_weight": 1.00,
                "gg_celltype_weight": 1.00,
                "pp_celltype_weight": 1.00,
                "proteinatlas_celltype_weight": 1.02 if "proteinatlas" in selected else 1.00,
                "module_weight": 1.00,
                "gene_graph_weight": 1.00,
                "prot_graph_weight": 1.00,
            }
            return hints.get(param, 1.0)
        if target_modality == "protein" and "gene" in source_modalities:
            hints = {
                "direct_weight": 1.03,
                "celltype_weight": 1.01,
                "gg_celltype_weight": 1.01,
                "pp_celltype_weight": 1.01,
                "proteinatlas_celltype_weight": 1.06 if "proteinatlas" in selected else 1.00,
                "kegg_pathway_weight": 1.03 if "kegg" in selected else 1.00,
            }
            return hints.get(param, 1.0)
        return 1.0

    source_to_param = {
        "uniprot+hgnc": ("direct_weight", factor("uniprot", "hgnc")),
        "reactome": ("module_weight", factor("reactome")),
        "reactome_gene_graph": ("gene_graph_weight", factor("reactome")),
        "reactome_protein_graph": ("prot_graph_weight", factor("reactome")),
        "cellmarker": ("celltype_weight", factor("cellmarker")),
        "cellmarker_gene_graph": ("gg_celltype_weight", factor("cellmarker")),
        "cellmarker_protein_graph": ("pp_celltype_weight", factor("cellmarker")),
        "proteinatlas": ("proteinatlas_celltype_weight", factor("proteinatlas")),
        "dorothea": ("dorothea_graph_weight", factor("dorothea")),
        "omnipath": ("omnipath_graph_weight", factor("omnipath")),
        "corum": ("corum_graph_weight", factor("corum")),
        "kegg": ("kegg_pathway_weight", factor("kegg")),
        "string": ("string_graph_weight", factor("string")),
    }

    applied = {}
    for label, (param, scale) in source_to_param.items():
        if param not in builder_kwargs:
            continue
        old = float(builder_kwargs[param])
        mode_scale = usage_multiplier(label, param)
        adjusted_scale = float(np.clip(float(scale) * template_multiplier(param) * mode_scale, 0.0, max_scale))
        new = old * adjusted_scale
        builder_kwargs[param] = new
        applied[param] = {
            "source_group": label,
            "base": old,
            "scale": adjusted_scale,
            "mode_scale": mode_scale,
            "effective": new,
        }

    return builder_kwargs, {
        "enabled": True,
        "task_spec": task_spec,
        "selected_sources": selected,
        "source_weights": numeric_weights,
        "source_usage_modes": usage_modes,
        "source_actions": source_actions,
        "mean_positive_weight": mean_w,
        "applied": applied,
    }


def resolve_proteo_kb_with_agent(
    task_name,
    source_modalities,
    target_task,
    data_pack,
    kb_cfg,
    repeat_idx,
    seed,
    orchestration_dir,
    requested_mode=None,
    selection_mode=None,
    use_data_profile=None,
    require_agent=None,
    selection_policy="anchored",
    performance_feedback=None,
    cache_tag="default",
):
    merged = {
        "direct_weight": PROTEO_KB_DIRECT_WEIGHT,
        "module_weight": PROTEO_KB_MODULE_WEIGHT,
        "celltype_weight": PROTEO_KB_CELLTYPE_WEIGHT,
        "gene_graph_weight": PROTEO_KB_GENE_GRAPH_WEIGHT,
        "prot_graph_weight": PROTEO_KB_PROT_GRAPH_WEIGHT,
        "gg_celltype_weight": PROTEO_KB_GG_CELLTYPE_WEIGHT,
        "pp_celltype_weight": PROTEO_KB_PP_CELLTYPE_WEIGHT,
        "dorothea_graph_weight": PROTEO_KB_DOROTHEA_GRAPH_WEIGHT,
        "omnipath_graph_weight": PROTEO_KB_OMNIPATH_GRAPH_WEIGHT,
        "corum_graph_weight": PROTEO_KB_CORUM_GRAPH_WEIGHT,
        "proteinatlas_celltype_weight": PROTEO_KB_PROTEINATLAS_CELLTYPE_WEIGHT,
        "kegg_pathway_weight": PROTEO_KB_KEGG_PATHWAY_WEIGHT,
        "string_graph_weight": PROTEO_KB_STRING_GRAPH_WEIGHT,
    }
    merged.update(kb_cfg or {})
    task_spec = TaskSpec(
        task_id=f"{task_name}_rep{int(repeat_idx)}_seed{int(seed)}",
        source_modalities=source_modalities,
        target_modality=target_task,
        species="human",
        required_relations=[
            "pathway_membership",
            "celltype_marker",
            "protein_gene_mapping" if target_task == "gene" else "gene_protein_mapping",
        ],
    )
    kb_paths = KnowledgePathConfig(
        hgnc_file=PROTEO_HGNC_PATH,
        uniprot_file=PROTEO_UNIPROT_PATH,
        reactome_uniprot_file=PROTEO_REACTOME_UNIPROT_PATH,
        reactome_ensembl_file=PROTEO_REACTOME_ENSEMBL_PATH,
        cellmarker_file=PROTEO_CELLMARKER_PATH,
        dorothea_file=PROTEO_DOROTHEA_PATH,
        omnipath_file=PROTEO_OMNIPATH_PATH,
        corum_file=PROTEO_CORUM_PATH,
        proteinatlas_file=PROTEO_PROTEINATLAS_PATH,
        kegg_pathways_file=PROTEO_KEGG_PATHWAYS_PATH,
        kegg_gene_pathway_file=PROTEO_KEGG_GENE_PATHWAY_PATH,
        string_file=PROTEO_STRING_PATH,
    )
    selection_cache_key = _proteo_selection_cache_key(
        task_name=task_name,
        source_modalities=source_modalities,
        target_task=target_task,
        data_pack=data_pack,
    ) + "_" + stable_hash(
        str(requested_mode or PROTEO_KB_REQUESTED_MODE),
        str(selection_mode or PROTEO_KB_SELECTION_MODE),
        str(bool(PROTEO_KB_USE_DATA_PROFILE if use_data_profile is None else use_data_profile)),
        str(selection_policy or "anchored"),
        str(cache_tag or "default"),
        json.dumps(performance_feedback or {}, sort_keys=True),
    )
    cache_json_path, cache_pkl_path = _proteo_selection_cache_paths(orchestration_dir, selection_cache_key)
    if selection_cache_key in _proteo_agent_selection_cache_mem:
        cached_payload = copy.deepcopy(_proteo_agent_selection_cache_mem[selection_cache_key])
        kb_tuple = cached_payload["kb_tuple"]
        report = cached_payload["report"]
        report["requested_mode"] = str(requested_mode or PROTEO_KB_REQUESTED_MODE)
        report["use_data_profile"] = bool(PROTEO_KB_USE_DATA_PROFILE if use_data_profile is None else use_data_profile)
        report["selection_cache_reused"] = True
        report["selection_cache_key"] = selection_cache_key
        report = _write_proteo_cached_report(report, orchestration_dir, task_spec.task_id)
        selected_sources = report.get("final_selected_sources", [])
        selected_sources = [str(s).strip().lower() for s in selected_sources if str(s).strip()]
        kb_key = (
            "agent",
            requested_mode or PROTEO_KB_REQUESTED_MODE,
            selection_mode or PROTEO_KB_SELECTION_MODE,
            bool(PROTEO_KB_USE_DATA_PROFILE if use_data_profile is None else use_data_profile),
            task_name,
            str(cache_tag or "default"),
            tuple(sorted(selected_sources)),
            tuple(sorted((k, float(v)) for k, v in report.get("builder_kwargs_effective", merged).items() if k in merged)),
            str(report.get("selection", {}).get("agent_decision", {}).get("confidence", "")),
            tuple(data_pack["adata1"].var_names.astype(str)),
            tuple(data_pack["adata2"].var_names.astype(str)),
        )
        return kb_tuple, kb_key, selected_sources, report
    if os.path.exists(cache_json_path) and os.path.exists(cache_pkl_path):
        try:
            with open(cache_json_path, "r", encoding="utf-8") as f:
                cached_report = json.load(f)
            with open(cache_pkl_path, "rb") as f:
                cached_kb_tuple = pickle.load(f)
            _proteo_agent_selection_cache_mem[selection_cache_key] = {
                "kb_tuple": cached_kb_tuple,
                "report": cached_report,
            }
            cached_report = copy.deepcopy(cached_report)
            cached_report["requested_mode"] = str(requested_mode or PROTEO_KB_REQUESTED_MODE)
            cached_report["use_data_profile"] = bool(PROTEO_KB_USE_DATA_PROFILE if use_data_profile is None else use_data_profile)
            cached_report["selection_cache_reused"] = True
            cached_report["selection_cache_key"] = selection_cache_key
            cached_report = _write_proteo_cached_report(cached_report, orchestration_dir, task_spec.task_id)
            selected_sources = cached_report.get("final_selected_sources", [])
            selected_sources = [str(s).strip().lower() for s in selected_sources if str(s).strip()]
            kb_key = (
                "agent",
                requested_mode or PROTEO_KB_REQUESTED_MODE,
                selection_mode or PROTEO_KB_SELECTION_MODE,
                bool(PROTEO_KB_USE_DATA_PROFILE if use_data_profile is None else use_data_profile),
                task_name,
                str(cache_tag or "default"),
                tuple(sorted(selected_sources)),
                tuple(sorted((k, float(v)) for k, v in cached_report.get("builder_kwargs_effective", merged).items() if k in merged)),
                str(cached_report.get("selection", {}).get("agent_decision", {}).get("confidence", "")),
                tuple(data_pack["adata1"].var_names.astype(str)),
                tuple(data_pack["adata2"].var_names.astype(str)),
            )
            return cached_kb_tuple, kb_key, selected_sources, cached_report
        except Exception as e:
            print(f">>> [KB-CACHE] failed to reuse selection cache for {task_name}: {type(e).__name__}: {e}", flush=True)
    kb_tuple, report = build_kb_with_orchestration(
        task_spec=task_spec,
        protein_names=data_pack["adata1"].var_names,
        gene_names=data_pack["adata2"].var_names,
        kb_paths=kb_paths,
        builder_fn=load_or_build_multi_relation_kb_cached,
        builder_kwargs={
            **merged,
            "cache_dir": PROTEO_KB_CACHE_DIR,
            "cache_prefix": f"proteo_agent_{sanitize_filename(task_name)}",
        },
        builder_kwargs_adapter=adapt_proteo_builder_kwargs_with_agent_weights,
        min_sources=PROTEO_KB_AGENT_MIN_SOURCES,
        max_sources=PROTEO_KB_AGENT_MAX_SOURCES,
        output_dir=orchestration_dir,
        require_agent=(REQUIRE_PROTEO_KB_LLM_AGENT if require_agent is None else require_agent) and (selection_mode or PROTEO_KB_SELECTION_MODE) in {"agent_only", "agent_rule"},
        selection_mode=selection_mode or PROTEO_KB_SELECTION_MODE,
        ensure_core_sources=["hgnc", "uniprot"],
        use_data_profile=PROTEO_KB_USE_DATA_PROFILE if use_data_profile is None else use_data_profile,
        selection_policy=selection_policy,
        performance_feedback=performance_feedback,
    )
    report["requested_mode"] = str(requested_mode or PROTEO_KB_REQUESTED_MODE)
    report["use_data_profile"] = bool(PROTEO_KB_USE_DATA_PROFILE if use_data_profile is None else use_data_profile)
    if report.get("selection", {}).get("mode") in {"agent_only", "agent+rule"} and report.get("validation", {}).get("ok", False):
        cache_report = copy.deepcopy(report)
        cache_report["selection_cache_reused"] = False
        cache_report["selection_cache_key"] = selection_cache_key
        with open(cache_json_path, "w", encoding="utf-8") as f:
            json.dump(cache_report, f, ensure_ascii=False, indent=2)
        with open(cache_pkl_path, "wb") as f:
            pickle.dump(kb_tuple, f)
        _proteo_agent_selection_cache_mem[selection_cache_key] = {
            "kb_tuple": kb_tuple,
            "report": cache_report,
        }
    selected_sources = report.get("final_selected_sources", [])
    selected_sources = [str(s).strip().lower() for s in selected_sources if str(s).strip()]
    kb_key = (
        "agent",
        requested_mode or PROTEO_KB_REQUESTED_MODE,
        selection_mode or PROTEO_KB_SELECTION_MODE,
        bool(PROTEO_KB_USE_DATA_PROFILE if use_data_profile is None else use_data_profile),
        task_name,
        str(cache_tag or "default"),
        tuple(sorted(selected_sources)),
        tuple(sorted((k, float(v)) for k, v in report.get("builder_kwargs_effective", merged).items() if k in merged)),
        str(report.get("selection", {}).get("agent_decision", {}).get("confidence", "")),
        tuple(data_pack["adata1"].var_names.astype(str)),
        tuple(data_pack["adata2"].var_names.astype(str)),
    )
    return kb_tuple, kb_key, selected_sources, report


def build_celltype_kb_priors(
    gene_names,
    protein_names,
    gene_protein_prior,
    max_celltypes=128,
    cellmarker_file=None,
    proteinatlas_file=None,
):
    global _proteo_hgnc_maps_cache
    if _proteo_hgnc_maps_cache is None:
        _proteo_hgnc_maps_cache = load_hgnc_maps(PROTEO_HGNC_PATH)
    alias_to_symbol, _ = _proteo_hgnc_maps_cache
    cell_to_markers = load_cellmarker_prior(cellmarker_file, alias_to_symbol)
    hpa_markers = load_proteinatlas_celltype_prior(proteinatlas_file, alias_to_symbol)
    if hpa_markers:
        for ct, markers in hpa_markers.items():
            cell_to_markers[ct].update(markers)
    if not cell_to_markers:
        return None, None, {"num_celltypes": 0}

    gene_names_arr = np.asarray(gene_names).astype(str)
    protein_names_arr = np.asarray(protein_names).astype(str)

    gene_to_idx = {}
    for i, g in enumerate(gene_names_arr):
        gc = alias_to_symbol.get(normalize_symbol(g), normalize_symbol(g))
        if gc is not None:
            gene_to_idx[gc] = i
    protein_to_idx = {}
    for i, p in enumerate(protein_names_arr):
        pc = alias_to_symbol.get(normalize_symbol(p), normalize_symbol(p))
        if pc is not None:
            protein_to_idx[pc] = i

    selected = sorted(cell_to_markers.items(), key=lambda kv: len(kv[1]), reverse=True)
    if max_celltypes > 0:
        selected = selected[:int(max_celltypes)]

    C = len(selected)
    G = len(gene_names_arr)
    P = len(protein_names_arr)
    ct_gene = np.zeros((C, G), dtype=np.float32)
    ct_prot_direct = np.zeros((C, P), dtype=np.float32)

    kept = []
    for ct_idx, (ct_name, markers) in enumerate(selected):
        markers_norm = set(alias_to_symbol.get(normalize_symbol(m), normalize_symbol(m)) for m in markers)
        markers_norm = {m for m in markers_norm if m is not None}
        gene_hits = [gene_to_idx[m] for m in markers_norm if m in gene_to_idx]
        prot_hits = [protein_to_idx[m] for m in markers_norm if m in protein_to_idx]
        if len(gene_hits) == 0 and len(prot_hits) == 0:
            continue
        if gene_hits:
            ct_gene[ct_idx, gene_hits] = 1.0
        if prot_hits:
            ct_prot_direct[ct_idx, prot_hits] = 1.0
        kept.append((ct_idx, ct_name))

    if not kept:
        return None, None, {"num_celltypes": 0}
    keep_rows = np.array([k[0] for k in kept], dtype=np.int64)
    ct_gene = ct_gene[keep_rows]
    ct_prot_direct = ct_prot_direct[keep_rows]
    ct_names = [k[1] for k in kept]

    ct_gene = normalize_matrix_rows(ct_gene)
    ct_prot_direct = normalize_matrix_rows(ct_prot_direct)
    ct_prot_projected = np.asarray(ct_gene @ gene_protein_prior, dtype=np.float32) if gene_protein_prior is not None else 0.0
    if isinstance(ct_prot_projected, float):
        ct_prot = ct_prot_direct
    else:
        ct_prot = normalize_matrix_rows(ct_prot_direct + ct_prot_projected)

    stats = {
        "num_celltypes": int(len(ct_names)),
        "num_gene_links": int((ct_gene > 0).sum()),
        "num_protein_links_direct": int((ct_prot_direct > 0).sum()),
        "num_protein_links_total": int((ct_prot > 0).sum()),
    }
    return ct_gene.astype(np.float32), ct_prot.astype(np.float32), stats


def resolve_celltype_priors_by_kb_key(data_pack, kb_key, gene_protein_prior, enable_cellmarker=True):
    cache_key = (kb_key, bool(enable_cellmarker))
    if cache_key in _proteo_ct_prior_cache_mem:
        return _proteo_ct_prior_cache_mem[cache_key]
    if not enable_cellmarker:
        payload = (None, None, {"num_celltypes": 0, "disabled": True})
        _proteo_ct_prior_cache_mem[cache_key] = payload
        return payload
    ct_gene, ct_prot, ct_stats = build_celltype_kb_priors(
        gene_names=data_pack["adata2"].var_names,
        protein_names=data_pack["adata1"].var_names,
        gene_protein_prior=gene_protein_prior,
        max_celltypes=128,
        cellmarker_file=PROTEO_CELLMARKER_PATH if "cellmarker" in str(kb_key) else None,
        proteinatlas_file=PROTEO_PROTEINATLAS_PATH if "proteinatlas" in str(kb_key) else None,
    )
    payload = (ct_gene, ct_prot, ct_stats)
    _proteo_ct_prior_cache_mem[cache_key] = payload
    return payload


def build_proteo_task_cache_path(task_name, repeat_idx, seed, selected_sources, graph_alpha, target_shape, kb_signature=None):
    cache_meta = {
        "v": 2,
        "cache_version": str(PROTEO_TASK_CACHE_VERSION),
        "proteo_kb_requested_mode": str(PROTEO_KB_REQUESTED_MODE),
        "proteo_kb_selection_mode": str(PROTEO_KB_SELECTION_MODE),
        "proteo_kb_use_data_profile": bool(PROTEO_KB_USE_DATA_PROFILE),
        "use_proteo_kb_agent": bool(USE_PROTEO_KB_AGENT),
        "require_proteo_kb_llm_agent": bool(REQUIRE_PROTEO_KB_LLM_AGENT),
        "task": str(task_name),
        "repeat_idx": int(repeat_idx),
        "seed": int(seed),
        "selected_sources": sorted([str(x) for x in (selected_sources or [])]),
        "graph_alpha": float(graph_alpha),
        "target_shape": [int(x) for x in target_shape],
        "epochs": int(TRAIN_EPOCHS),
        "lr": float(TRAIN_LR),
        "dropout": float(TRAIN_DROPOUT),
        "hidden_dim": int(TRAIN_HIDDEN_DIM),
        "num_layers": int(TRAIN_NUM_LAYERS),
        "lambda_target": float(LAMBDA_TARGET),
        "lambda_pcc": float(LAMBDA_PCC),
        "lambda_latent": float(LAMBDA_LATENT),
        "lambda_recon": float(LAMBDA_RECON),
        "lambda_kb": float(LAMBDA_KB),
        "lambda_graph": float(LAMBDA_GRAPH),
        "lambda_ortho": float(LAMBDA_ORTHO),
        "lambda_dgi": float(LAMBDA_DGI),
        "lambda_cycle": 0.2,
        "lambda_partial_cycle": 0.2,
        "lambda_align": float(LAMBDA_ALIGN),
        "lambda_bridge_align": 0.5,
        "target_soft_alpha": float(TARGET_SOFT_ALPHA),
        "kb_ct_prior_mix": float(KB_CT_PRIOR_MIX),
        "lambda_full": float(LAMBDA_FULL),
        "lambda_partial": float(LAMBDA_PARTIAL),
        "disable_dynamic_retriever": bool(DISABLE_DYNAMIC_RETRIEVER),
        "disable_kb_fusion": bool(DISABLE_KB_FUSION),
        "disable_soft_gate": bool(DISABLE_SOFT_GATE),
        "disable_residual_refine": bool(DISABLE_RESIDUAL_REFINE),
        "proteo_final_kb_cfg": PROTEO_FINAL_KB_CFG,
        "kb_signature": kb_signature or {},
    }
    cache_key = hashlib.md5(json.dumps(cache_meta, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    return os.path.join(PROTEO_TASK_CACHE_DIR, f"{task_name}_rep{int(repeat_idx)}_seed{int(seed)}_{cache_key}.pkl")


def _metric_compare(candidate_metrics, baseline_metrics, target_task=None):
    gains = {
        "PCC": float(candidate_metrics["PCC"]) - float(baseline_metrics["PCC"]),
        "SSIM": float(candidate_metrics["SSIM"]) - float(baseline_metrics["SSIM"]),
        "CMD": float(baseline_metrics["CMD"]) - float(candidate_metrics["CMD"]),
        "RMSE": float(baseline_metrics["RMSE"]) - float(candidate_metrics["RMSE"]),
    }
    tolerances = {"PCC": 0.002, "SSIM": 0.001, "CMD": 0.002, "RMSE": 0.002}
    directions = {
        k: ("better" if v > tolerances[k] else "worse" if v < -tolerances[k] else "neutral")
        for k, v in gains.items()
    }
    wins = sum(1 for k, v in gains.items() if v > tolerances[k])
    losses = sum(1 for k, v in gains.items() if v < -tolerances[k])
    weights = (
        {"PCC": 0.45, "RMSE": 0.25, "CMD": 0.20, "SSIM": 0.10}
        if str(target_task).lower() == "gene"
        else {"PCC": 0.40, "RMSE": 0.25, "SSIM": 0.20, "CMD": 0.15}
    )
    normalized = {
        k: float(np.clip(gains[k] / max(tolerances[k], 1e-8), -3.0, 3.0))
        for k in gains
    }
    weighted_score = float(sum(weights[k] * normalized[k] for k in gains))
    target_task = str(target_task).lower()
    if target_task == "gene":
        hard_guard = (
            gains["PCC"] < -0.008
            or gains["CMD"] < -0.010
            or (gains["PCC"] < -0.005 and gains["CMD"] < -0.005)
        )
        primary_worse = gains["PCC"] < -0.005 or gains["CMD"] < -0.005 or gains["RMSE"] < -0.003
        primary_better = gains["PCC"] > 0.005 and gains["CMD"] > 0.003 and gains["RMSE"] >= 0.0
        metric_focus = ["PCC", "CMD", "RMSE", "SSIM"]
    else:
        hard_guard = gains["PCC"] < -0.005 or gains["RMSE"] < -0.008
        primary_worse = gains["PCC"] < -0.004 or gains["RMSE"] < -0.004
        primary_better = gains["PCC"] > 0.004 and gains["RMSE"] > 0.004
        metric_focus = ["PCC", "RMSE", "SSIM", "CMD"]
    if wins >= 3 and not primary_worse:
        strategy = "exploit_success"
        outcome = "success"
    elif hard_guard:
        strategy = "hard_recover"
        outcome = "underperformed"
    elif primary_better or (wins == 2 and not primary_worse and weighted_score > 0):
        strategy = "hybrid_refine"
        outcome = "partial_success"
    else:
        strategy = "hard_recover"
        outcome = "underperformed"
    out = {f"{k}_delta": float(v) for k, v in gains.items()}
    out.update(
        metric_gains=gains,
        metric_directions=directions,
        metric_tolerances=tolerances,
        metric_weights=weights,
        weighted_score=weighted_score,
        wins=int(wins),
        losses=int(losses),
        is_better=bool(strategy == "exploit_success"),
        round1_outcome=outcome,
        refine_strategy=strategy,
        hard_guard_triggered=bool(hard_guard),
        metric_focus=metric_focus,
        primary_metric_worse=bool(primary_worse),
    )
    return out


def _task_specific_round2_hints(task_name, target_task, source_modalities, baseline_sources, round1_sources, metric_delta):
    added = [s for s in round1_sources if s not in baseline_sources]
    removed = [s for s in baseline_sources if s not in round1_sources]
    strategy = str(metric_delta.get("refine_strategy", "hybrid_refine"))
    hints = {
        "protected_sources": list(baseline_sources),
        "candidate_additions_from_round1": added,
        "removed_rule_sources_in_round1": removed,
        "max_new_sources_when_recovering": 1,
        "prefer_weight_adjustment_over_large_source_changes": True,
        "metric_focus": list(metric_delta.get("metric_focus", ["PCC", "RMSE", "SSIM"])),
        "source_action_hints": {},
        "suggested_source_weights": {},
        "source_usage_modes": {},
    }
    actions = hints["source_action_hints"]
    weights = hints["suggested_source_weights"]
    modes = hints["source_usage_modes"]
    for s in baseline_sources:
        actions[s] = "keep"
        weights[s] = 1.0
        modes[s] = "full"
    if strategy == "exploit_success":
        hints["protected_sources"] = list(round1_sources)
        for s in round1_sources:
            actions[s] = "keep_successful_round1_source"
            weights[s] = 1.10 if s in added else 1.0
            modes[s] = "full"
        if target_task == "protein" and "gene" in source_modalities:
            for s in ["kegg", "proteinatlas"]:
                if s in round1_sources:
                    actions[s] = "keep_or_upweight"
                    weights[s] = 1.20
                    modes[s] = "full"
            if "reactome" in round1_sources:
                actions["reactome"] = "downweight_if_retained"
                weights["reactome"] = 0.75
                modes["reactome"] = "low_weight_full"
    elif strategy == "hard_recover":
        for s in added:
            actions[s] = "remove_or_low_weight_probe"
            weights[s] = 0.55
            modes[s] = "graph_only"
        if "kegg" in added and target_task == "gene":
            actions["kegg"] = "graph_only"
            weights["kegg"] = 0.60
            modes["kegg"] = "graph_only"
        if "proteinatlas" in added and target_task == "protein":
            actions["proteinatlas"] = "prior_only"
            weights["proteinatlas"] = 0.70
            modes["proteinatlas"] = "prior_only"
        if "proteinatlas" in added and target_task == "gene":
            actions["proteinatlas"] = "disabled"
            weights["proteinatlas"] = 0.0
            modes["proteinatlas"] = "disabled"
        if "string" in added:
            actions["string"] = "graph_only"
            weights["string"] = 0.45
            modes["string"] = "graph_only"
    else:
        for s in added:
            actions[s] = "keep_only_if_explains_improved_metrics_else_downweight"
            weights[s] = 0.75
            modes[s] = "low_weight_full"
        if "proteinatlas" in added and target_task == "protein":
            weights["proteinatlas"] = 0.90
            modes["proteinatlas"] = "prior_only"
        if "kegg" in added and target_task == "protein" and "gene" in source_modalities:
            weights["kegg"] = 1.05
            modes["kegg"] = "low_weight_full"
        if "kegg" in added and target_task == "gene":
            modes["kegg"] = "graph_only"
    return hints


def _extract_agent_source_weights(report, selected_sources):
    agent_decision = (report or {}).get("selection", {}).get("agent_decision", {})
    raw = agent_decision.get("source_weights", {}) if isinstance(agent_decision, dict) else {}
    raw = raw if isinstance(raw, dict) else {}
    out = {}
    for sid in selected_sources:
        try:
            out[sid] = float(raw.get(sid, 1.0))
        except Exception:
            out[sid] = 1.0
    return out


def _extract_agent_source_modes(report, selected_sources, field_name):
    agent_decision = (report or {}).get("selection", {}).get("agent_decision", {})
    raw = agent_decision.get(field_name, {}) if isinstance(agent_decision, dict) else {}
    raw = raw if isinstance(raw, dict) else {}
    return {sid: str(raw.get(sid, "full" if field_name == "source_usage_modes" else "keep")) for sid in selected_sources}


def _extract_effective_builder_weights(report):
    keys = [
        "direct_weight", "module_weight", "celltype_weight", "gene_graph_weight",
        "prot_graph_weight", "gg_celltype_weight", "pp_celltype_weight",
        "proteinatlas_celltype_weight", "kegg_pathway_weight", "string_graph_weight",
        "dorothea_graph_weight", "omnipath_graph_weight", "corum_graph_weight",
    ]
    raw = (report or {}).get("builder_kwargs_effective", {})
    raw = raw if isinstance(raw, dict) else {}
    out = {}
    for k in keys:
        if k in raw:
            try:
                out[k] = float(raw[k])
            except Exception:
                pass
    return out


def _build_round2_feedback(task_name, baseline_row, round1_row, target_task=None, source_modalities=None, baseline_report=None, round1_report=None):
    baseline_sources = [str(x).strip().lower() for x in str(baseline_row.get("selected_sources", "")).split(";") if str(x).strip()]
    round1_sources = [str(x).strip().lower() for x in str(round1_row.get("selected_sources", "")).split(";") if str(x).strip()]
    target_task = target_task or baseline_row.get("target_task", "")
    source_modalities = list(source_modalities or [])
    delta = _metric_compare(round1_row, baseline_row, target_task=target_task)
    policy = _task_specific_round2_hints(task_name, target_task, source_modalities, baseline_sources, round1_sources, delta)
    return {
        "task_id": str(task_name),
        "metric_aware_refinement_enabled": bool(PROTEO_AGENT_METRIC_AWARE_REFINEMENT),
        "baseline": {
            "selected_sources": baseline_sources,
            "source_weights": _extract_agent_source_weights(baseline_report, baseline_sources),
            "source_usage_modes": _extract_agent_source_modes(baseline_report, baseline_sources, "source_usage_modes"),
            "source_actions": _extract_agent_source_modes(baseline_report, baseline_sources, "source_actions"),
            "effective_builder_weights": _extract_effective_builder_weights(baseline_report),
            "metrics": {k: float(baseline_row[k]) for k in ["PCC", "SSIM", "CMD", "RMSE"]},
        },
        "round1_free_agent": {
            "selected_sources": round1_sources,
            "source_weights": _extract_agent_source_weights(round1_report, round1_sources),
            "source_usage_modes": _extract_agent_source_modes(round1_report, round1_sources, "source_usage_modes"),
            "source_actions": _extract_agent_source_modes(round1_report, round1_sources, "source_actions"),
            "effective_builder_weights": _extract_effective_builder_weights(round1_report),
            "metrics": {k: float(round1_row[k]) for k in ["PCC", "SSIM", "CMD", "RMSE"]},
        },
        "metric_deltas_vs_baseline": delta,
        "added_vs_baseline": [s for s in round1_sources if s not in baseline_sources],
        "removed_vs_baseline": [s for s in baseline_sources if s not in round1_sources],
        "round1_outcome": delta["round1_outcome"],
        "refine_strategy": delta["refine_strategy"],
        "round2_policy": policy,
        "instruction": (
            "First diagnose the metric changes. PCC and SSIM are higher-is-better; CMD and RMSE "
            "are lower-is-better. If round1 succeeded, preserve and possibly upweight the successful "
            "round1 additions. If round1 underperformed on primary metrics, anchor to the rule sources "
            "and keep at most one useful round1 addition at low weight. Prefer source_weights, "
            "source_usage_modes, and small changes over broad source churn."
        ),
    }


def _run_single_proteo_task_configuration(
    task_idx,
    task_cfg,
    selected_sources,
    task_orch_report,
    kb_tuple,
    kb_key,
    data_pack,
    trainer_base_cfg,
    repeat_idx,
    seed,
    phase_label="main",
):
    task_name = task_cfg["task"]
    target_task = task_cfg["target_task"]
    full_use_obs = bool(task_cfg["full_use_obs"])
    partial_use_obs = bool(task_cfg["partial_use_obs"])
    graph_alpha = float(task_cfg.get("graph_alpha", PROTEO_BASE_GRAPH_ALPHA))
    adata1 = data_pack["adata1"]
    adata2 = data_pack["adata2"]
    adata1_gt = data_pack["adata1_gt"]
    adata2_prot = data_pack["adata2_prot"]
    graph1_base = data_pack["graph1_base"]
    graph2_base = data_pack["graph2_base"]
    graph_eval_gene_s2 = data_pack["graph_eval_gene_s2"]
    graph_eval_prot_s1 = data_pack["graph_eval_prot_s1"]

    task_trainer_cfg = copy.deepcopy(trainer_base_cfg)
    task_trainer_cfg.update(task_cfg.get("trainer_overrides", {}))
    protein_gene_prior_i, gene_protein_prior_i, gene_gene_kb_graph_i, protein_protein_kb_graph_i, _, _ = kb_tuple
    celltype_gene_prior_i, celltype_protein_prior_i, _ = resolve_celltype_priors_by_kb_key(
        data_pack=data_pack,
        kb_key=kb_key,
        gene_protein_prior=gene_protein_prior_i,
        enable_cellmarker=("cellmarker" in selected_sources or "proteinatlas" in selected_sources),
    )
    task_trainer_cfg.update(
        protein_gene_prior=protein_gene_prior_i,
        gene_protein_prior=gene_protein_prior_i,
        gene_gene_kb_graph=gene_gene_kb_graph_i,
        protein_protein_kb_graph=protein_protein_kb_graph_i,
        celltype_gene_prior=celltype_gene_prior_i,
        celltype_protein_prior=celltype_protein_prior_i,
        seed=int(seed),
    )

    graph1_i = get_or_build_proteo_fused_graph(adata1, graph_alpha, f"s1_{phase_label}", base_graph=graph1_base)
    graph2_i = get_or_build_proteo_fused_graph(adata2, graph_alpha, f"s2_{phase_label}", base_graph=graph2_base)

    if target_task == "gene":
        trainer = se.AKGOmicsFullPartialProtocol(
            target_task="gene",
            full_he=adata1.obsm["he"],
            full_graph=graph1_i,
            full_gene=adata1_gt.X,
            full_protein=adata1.obsm["protein"],
            partial_he=adata2.obsm["he"],
            partial_graph=graph2_i,
            partial_gene_obs=None,
            partial_protein_obs=adata2_prot.X if partial_use_obs else None,
            partial_gene_eval=adata2.X,
            partial_protein_eval=None,
            full_gene_latent=data_pack["gene1_gt_latent"],
            full_slice_id=0,
            partial_slice_id=1,
            full_use_obs=full_use_obs,
            partial_use_obs=partial_use_obs,
            save_path=None,
            **task_trainer_cfg,
        )
        history = trainer.train()
        pred = trainer.infer_partial()
        gt_eval = to_dense(adata2.X).astype(np.float32)
        pred_eval = pred["pred_full"]
        metrics = compute_three_metrics(gt_eval.copy(), pred_eval.copy(), graph_eval_gene_s2)
        analysis_info = run_detailed_task_analysis(
            task_name=f"{task_name}_{phase_label}",
            repeat_idx=repeat_idx,
            seed=seed,
            history=history,
            gt=gt_eval,
            pred=pred_eval,
            graph_eval=graph_eval_gene_s2,
            adata_for_names_and_coords=adata2,
            target_task="gene",
            inference_output=pred,
        )
    else:
        trainer = se.AKGOmicsFullPartialProtocol(
            target_task="protein",
            full_he=adata2.obsm["he"],
            full_graph=graph2_i,
            full_gene=adata2.X,
            full_protein=adata2_prot.X,
            partial_he=adata1.obsm["he"],
            partial_graph=graph1_i,
            partial_gene_obs=adata1_gt.X if partial_use_obs else None,
            partial_protein_obs=None,
            partial_gene_eval=None,
            partial_protein_eval=adata1.X,
            full_gene_latent=data_pack["gene2_latent"],
            full_slice_id=1,
            partial_slice_id=0,
            full_use_obs=full_use_obs,
            partial_use_obs=partial_use_obs,
            save_path=None,
            **task_trainer_cfg,
        )
        history = trainer.train()
        pred = trainer.infer_partial()
        gt_eval = to_dense(adata1.X).astype(np.float32)
        pred_eval = pred["pred_full"]
        metrics = compute_three_metrics(gt_eval.copy(), pred_eval.copy(), graph_eval_prot_s1)
        analysis_info = run_detailed_task_analysis(
            task_name=f"{task_name}_{phase_label}",
            repeat_idx=repeat_idx,
            seed=seed,
            history=history,
            gt=gt_eval,
            pred=pred_eval,
            graph_eval=graph_eval_prot_s1,
            adata_for_names_and_coords=adata1,
            target_task="protein",
            inference_output=pred,
        )

    row = {
        "experiment_id": "E2_unified8_detailed",
        "repeat_idx": int(repeat_idx),
        "seed": int(seed),
        "task": task_name,
        "task_phase": str(phase_label),
        "task_order": int(task_idx),
        "target_task": target_task,
        "full_use_obs": bool(full_use_obs),
        "partial_use_obs": bool(partial_use_obs),
        "graph_alpha": float(graph_alpha),
        "PCC": float(metrics["PCC"]),
        "SSIM": float(metrics["SSIM"]),
        "CMD": float(metrics["CMD"]),
        "RMSE": float(metrics["RMSE"]),
        "selected_sources": ";".join(selected_sources),
        "kb_requested_mode": str(task_orch_report.get("requested_mode", PROTEO_KB_REQUESTED_MODE)),
        "kb_use_data_profile": bool(task_orch_report.get("use_data_profile", PROTEO_KB_USE_DATA_PROFILE)),
        "kb_select_mode": str(task_orch_report.get("selection", {}).get("mode", task_orch_report.get("selection_mode", ""))),
        "kb_selection_mode_requested": str(task_orch_report.get("selection_mode_requested", PROTEO_KB_SELECTION_MODE)),
        "agent_provider": str(task_orch_report.get("selection", {}).get("agent_decision", {}).get("provider", "")),
        "agent_model": str(task_orch_report.get("selection", {}).get("agent_decision", {}).get("model", "")),
        "agent_confidence": task_orch_report.get("selection", {}).get("agent_decision", {}).get("confidence", np.nan),
        "kb_num_direct_links": task_orch_report.get("builder_stats", {}).get("num_direct_links", np.nan),
        "kb_num_module_links": task_orch_report.get("builder_stats", {}).get("num_module_links", np.nan),
        "kb_kegg_pathway_count": task_orch_report.get("builder_stats", {}).get("kegg_pathway_count", np.nan),
        "kb_string_protein_edges": task_orch_report.get("builder_stats", {}).get("string_protein_edges", np.nan),
        "kb_orchestration_report": str(task_orch_report.get("report_path", "")),
        "agent_trace_path": str(task_orch_report.get("agent_trace_path", "")),
        "domain": "proteogenomics",
    }
    row = refresh_row_selection_metadata(
        row=row,
        selected_sources=selected_sources,
        report=task_orch_report,
        domain="proteogenomics",
    )
    if ENABLE_DETAILED_ANALYSIS:
        row.update({
            "analysis_dir": analysis_info.get("analysis_dir", ""),
            "loss_csv": analysis_info.get("loss_csv", ""),
            "loss_fig": analysis_info.get("loss_fig", ""),
            "top_feature_csv": analysis_info.get("top_feature_csv", ""),
            "all_feature_csv": analysis_info.get("all_feature_csv", ""),
            "num_spatial_plots": int(analysis_info.get("num_spatial_plots", 0)),
        })
    return row, analysis_info


def run_proteogenomics_tasks_for_seed(repeat_idx, seed, data_pack, orchestration_dir=None):
    adata1 = data_pack["adata1"]
    adata2 = data_pack["adata2"]
    adata1_gt = data_pack["adata1_gt"]
    adata2_prot = data_pack["adata2_prot"]
    graph1_base = data_pack["graph1_base"]
    graph2_base = data_pack["graph2_base"]
    graph_eval_gene_s2 = data_pack["graph_eval_gene_s2"]
    graph_eval_prot_s1 = data_pack["graph_eval_prot_s1"]

    if orchestration_dir is None:
        orchestration_dir = KB_ORCHESTRATION_DIR
    os.makedirs(orchestration_dir, exist_ok=True)

    trainer_base_cfg = dict(
        he_dim=adata1.obsm["he"].shape[1],
        protein_dim=adata1.obsm["protein"].shape[1],
        gene_dim=adata2.X.shape[1],
        gene_pca_components=data_pack["gene_pca_components"],
        gene_pca_mean=data_pack["gene_pca_mean"],
        hidden_dim=TRAIN_HIDDEN_DIM,
        num_layers=TRAIN_NUM_LAYERS,
        epochs=TRAIN_EPOCHS,
        lr=TRAIN_LR,
        dropout=TRAIN_DROPOUT,
        device=device,
        lambda_target=LAMBDA_TARGET,
        lambda_pcc=0.0,
        lambda_latent=LAMBDA_LATENT,
        lambda_recon=LAMBDA_RECON,
        lambda_kb=LAMBDA_KB,
        lambda_graph=LAMBDA_GRAPH,
        lambda_ortho=LAMBDA_ORTHO,
        lambda_dgi=LAMBDA_DGI,
        lambda_cycle=0.2,
        lambda_partial_cycle=0.2,
        lambda_align=LAMBDA_ALIGN,
        lambda_bridge_align=0.5,
        target_soft_alpha=0.0,
        kb_ct_prior_mix=0.7,
        lambda_full=LAMBDA_FULL,
        lambda_partial=LAMBDA_PARTIAL,
        align_max_points=ALIGN_MAX_POINTS,
        use_amp=False,
        disable_dynamic_retriever=DISABLE_DYNAMIC_RETRIEVER,
        disable_kb_fusion=DISABLE_KB_FUSION,
        disable_soft_gate=DISABLE_SOFT_GATE,
        disable_residual_refine=DISABLE_RESIDUAL_REFINE,
    )

    local_status = get_local_proteo_source_status()
    rows = []
    for task_idx, task_cfg in enumerate(PROTEO_TASK_SPECS, start=1):
        task_name = task_cfg["task"]
        if PROTEO_TASK_FILTER and task_name not in PROTEO_TASK_FILTER and str(task_idx) not in PROTEO_TASK_FILTER:
            print(f">>> [SKIP] {task_name} filtered by PROTEO_TASK_FILTER", flush=True)
            continue
        target_task = task_cfg["target_task"]
        full_use_obs = bool(task_cfg["full_use_obs"])
        partial_use_obs = bool(task_cfg["partial_use_obs"])
        graph_alpha = float(task_cfg.get("graph_alpha", PROTEO_BASE_GRAPH_ALPHA))

        source_modalities = infer_source_modalities_for_task(target_task, full_use_obs, partial_use_obs)
        if PROTEO_AGENT_TWO_ROUND and USE_PROTEO_KB_AGENT:
            imported_baseline = load_imported_formal_rule_baseline_row(
                task_idx=task_idx,
                task_cfg=task_cfg,
                repeat_idx=repeat_idx,
                seed=seed,
                orchestration_dir=orchestration_dir,
            )
            if imported_baseline is not None:
                baseline_row, baseline_sources, baseline_report = imported_baseline
            else:
                baseline_kb_tuple, baseline_kb_key, baseline_sources, baseline_report = resolve_proteo_formal_rule_baseline(
                    task_name=task_name,
                    data_pack=data_pack,
                    repeat_idx=repeat_idx,
                    seed=seed,
                    orchestration_dir=orchestration_dir,
                    local_status=local_status,
                    kb_cfg=copy.deepcopy(PROTEO_FINAL_KB_CFG),
                    selected_sources=[s for s in PROTEO_RULE_BASELINE_SOURCES if local_status.get(s, False)],
                )
                baseline_row, baseline_analysis = _run_single_proteo_task_configuration(
                    task_idx=task_idx,
                    task_cfg=task_cfg,
                    selected_sources=baseline_sources,
                    task_orch_report=baseline_report,
                    kb_tuple=baseline_kb_tuple,
                    kb_key=baseline_kb_key,
                    data_pack=data_pack,
                    trainer_base_cfg=trainer_base_cfg,
                    repeat_idx=repeat_idx,
                    seed=seed,
                    phase_label="rule_round",
                )
            rows.append(baseline_row)

            round1_kb_tuple, round1_kb_key, round1_sources, round1_report = resolve_proteo_kb_with_agent(
                task_name=task_name,
                source_modalities=source_modalities,
                target_task=target_task,
                data_pack=data_pack,
                kb_cfg=copy.deepcopy(PROTEO_FINAL_KB_CFG),
                repeat_idx=repeat_idx,
                seed=seed,
                orchestration_dir=orchestration_dir,
                requested_mode="agent-data",
                selection_mode="agent_only",
                use_data_profile=True,
                require_agent=True,
                selection_policy="free",
                cache_tag="round1_free",
            )
            print(f">>> [KB-SELECT] {task_name} phase=round1_free mode={round1_report.get('selection', {}).get('mode', '')} selected={';'.join(round1_sources)}", flush=True)
            round1_row, round1_analysis = _run_single_proteo_task_configuration(
                task_idx=task_idx,
                task_cfg=task_cfg,
                selected_sources=round1_sources,
                task_orch_report=round1_report,
                kb_tuple=round1_kb_tuple,
                kb_key=round1_kb_key,
                data_pack=data_pack,
                trainer_base_cfg=trainer_base_cfg,
                repeat_idx=repeat_idx,
                seed=seed,
                phase_label="agent_round1_free",
            )
            rows.append(round1_row)

            round2_feedback = _build_round2_feedback(
                task_name,
                baseline_row,
                round1_row,
                target_task=target_task,
                source_modalities=source_modalities,
                baseline_report=baseline_report,
                round1_report=round1_report,
            )
            round2_kb_tuple, round2_kb_key, round2_sources, round2_report = resolve_proteo_kb_with_agent(
                task_name=task_name,
                source_modalities=source_modalities,
                target_task=target_task,
                data_pack=data_pack,
                kb_cfg=copy.deepcopy(PROTEO_FINAL_KB_CFG),
                repeat_idx=repeat_idx,
                seed=seed,
                orchestration_dir=orchestration_dir,
                requested_mode="agent-rule-data",
                selection_mode="agent_rule",
                use_data_profile=True,
                require_agent=True,
                selection_policy="anchored",
                performance_feedback=round2_feedback,
                cache_tag="round2_refine",
            )
            print(f">>> [KB-SELECT] {task_name} phase=round2_refine mode={round2_report.get('selection', {}).get('mode', '')} selected={';'.join(round2_sources)}", flush=True)
            round2_row, round2_analysis = _run_single_proteo_task_configuration(
                task_idx=task_idx,
                task_cfg=task_cfg,
                selected_sources=round2_sources,
                task_orch_report=round2_report,
                kb_tuple=round2_kb_tuple,
                kb_key=round2_kb_key,
                data_pack=data_pack,
                trainer_base_cfg=trainer_base_cfg,
                repeat_idx=repeat_idx,
                seed=seed,
                phase_label="agent_round2_refine",
            )
            round2_row.update({
                "rule_PCC": float(baseline_row["PCC"]),
                "rule_SSIM": float(baseline_row["SSIM"]),
                "rule_CMD": float(baseline_row["CMD"]),
                "rule_RMSE": float(baseline_row["RMSE"]),
                "round1_PCC": float(round1_row["PCC"]),
                "round1_SSIM": float(round1_row["SSIM"]),
                "round1_CMD": float(round1_row["CMD"]),
                "round1_RMSE": float(round1_row["RMSE"]),
                "round2_refine_strategy": str(round2_feedback.get("refine_strategy", "")),
                "round2_round1_outcome": str(round2_feedback.get("round1_outcome", "")),
                "round2_metric_wins": int(round2_feedback.get("metric_deltas_vs_baseline", {}).get("wins", 0)),
                "round2_weighted_score": float(round2_feedback.get("metric_deltas_vs_baseline", {}).get("weighted_score", 0.0)),
                "round2_hard_guard_triggered": bool(round2_feedback.get("metric_deltas_vs_baseline", {}).get("hard_guard_triggered", False)),
                "round2_metric_focus": ";".join([str(x) for x in round2_feedback.get("metric_deltas_vs_baseline", {}).get("metric_focus", [])]),
            })
            rows.append(round2_row)
            continue

        kb_tuple = None
        kb_key = None
        if USE_PROTEO_KB_AGENT:
            kb_tuple, kb_key, selected_sources, task_orch_report = resolve_proteo_kb_with_agent(
                task_name=task_name,
                source_modalities=source_modalities,
                target_task=target_task,
                data_pack=data_pack,
                kb_cfg=copy.deepcopy(PROTEO_FINAL_KB_CFG),
                repeat_idx=repeat_idx,
                seed=seed,
                orchestration_dir=orchestration_dir,
            )
        else:
            if PROTEO_KB_SELECTION_MODE != "rule_only":
                raise ValueError("Set USE_PROTEO_KB_AGENT=1 for agent_only or agent_rule experiments.")
            selected_sources, task_orch_report = orchestrate_sources(
                task_id=task_name,
                source_modalities=source_modalities,
                target_modality=target_task,
                local_source_status=local_status,
                max_sources=4,
                ensure_core_sources=["hgnc", "uniprot"],
                report_path=os.path.join(
                    orchestration_dir,
                    f"kb_orchestration_{task_name}_rep{int(repeat_idx)}_seed{int(seed)}.json",
                ),
            )
            selected_sources = [s for s in selected_sources if s in {"hgnc", "uniprot", "reactome", "cellmarker"}]
            if not selected_sources:
                selected_sources = [s for s in ["hgnc", "uniprot", "reactome", "cellmarker"] if local_status.get(s, False)]
        select_mode = str(task_orch_report.get("selection", {}).get("mode", task_orch_report.get("selection_mode", "")))
        print(
            f">>> [KB-SELECT] {task_name} requested={PROTEO_KB_REQUESTED_MODE} mode={select_mode} data_profile={PROTEO_KB_USE_DATA_PROFILE} selected={';'.join(selected_sources)}",
            flush=True,
        )
        if kb_tuple is None or kb_key is None:
            kb_tuple, kb_key = resolve_proteo_kb_by_config(
                data_pack=data_pack,
                kb_cfg=copy.deepcopy(PROTEO_FINAL_KB_CFG),
                selected_sources=selected_sources,
            )
        row, analysis_info = _run_single_proteo_task_configuration(
            task_idx=task_idx,
            task_cfg=task_cfg,
            selected_sources=selected_sources,
            task_orch_report=task_orch_report,
            kb_tuple=kb_tuple,
            kb_key=kb_key,
            data_pack=data_pack,
            trainer_base_cfg=trainer_base_cfg,
            repeat_idx=repeat_idx,
            seed=seed,
            phase_label="main",
        )
        rows.append(row)
    return rows


# ============================================================================
# 4) Metabolomics preprocessing (from provided example)
# ============================================================================
def preprocess_rna_with_he_cached(folder, folder_tag, resolution=580, image_encoder="uni", device="cuda:0"):
    rna_path = load_rna_h5ad(folder)
    img_path = load_image_file(folder)
    cache_key = stable_hash("rna", folder_tag, rna_path, img_path, resolution, image_encoder)
    cache_path = os.path.join(PREP_CACHE_DIR, f"rna_{folder_tag}_{cache_key}.h5ad")
    if os.path.exists(cache_path):
        return sc.read_h5ad(cache_path)

    adata = sc.read_h5ad(rna_path)
    adata.var_names_make_unique()
    adata = ensure_spatial_coords(adata)
    mt_mask = np.array([str(x).upper().startswith("MT-") for x in adata.var_names])
    if mt_mask.sum() > 0:
        adata = adata[:, ~mt_mask].copy()
    sc.pp.log1p(adata)
    adata = se.pp.Preprocess_adata(adata, scale=True)
    adata.obsm["image_coor"] = adata.obsm["spatial"].copy()
    img, _ = safe_read_he_image(img_path)
    he_patches, adata = se.pp.Tiling_HE_patches(resolution, adata, img)
    adata = se.pp.Extract_HE_patches_representaion(he_patches, store_key="he", adata=adata, image_encoder=image_encoder, device=device)
    atomic_write_h5ad(adata, cache_path)
    return adata


def preprocess_metabolite_with_he_cached(folder, folder_tag, resolution=580, image_encoder="uni", device="cuda:0"):
    met_path = load_metabolite_h5ad(folder)
    img_path = load_image_file(folder)
    cache_key = stable_hash("met", folder_tag, met_path, img_path, resolution, image_encoder)
    cache_path = os.path.join(PREP_CACHE_DIR, f"metabolite_{folder_tag}_{cache_key}.h5ad")
    if os.path.exists(cache_path):
        return sc.read_h5ad(cache_path)

    adata = sc.read_h5ad(met_path)
    adata.var_names_make_unique()
    adata = ensure_spatial_coords(adata)
    if "metabolism" in adata.var.columns:
        adata.var_names = pd.Index(adata.var["metabolism"].astype(str))
        adata.var_names_make_unique()
    adata = se.pp.Preprocess_adata(adata, scale=True)
    adata.obsm["image_coor"] = adata.obsm["spatial"].copy()
    img, _ = safe_read_he_image(img_path)
    he_patches, adata = se.pp.Tiling_HE_patches(resolution, adata, img)
    adata = se.pp.Extract_HE_patches_representaion(he_patches, store_key="he", adata=adata, image_encoder=image_encoder, device=device)
    atomic_write_h5ad(adata, cache_path)
    return adata


def build_spatial_graph_for_moran(adata, n_neighbors=6):
    coords = np.asarray(adata.obsm["spatial"], dtype=np.float32)
    n = coords.shape[0]
    if n <= 1:
        return sp.eye(n, dtype=np.float32, format="coo")
    k = min(n_neighbors + 1, n)
    nn = NearestNeighbors(n_neighbors=k, metric="euclidean")
    nn.fit(coords)
    distances, indices = nn.kneighbors(coords)
    rows, cols, vals = [], [], []
    valid_d = distances[:, 1:].reshape(-1)
    valid_d = valid_d[np.isfinite(valid_d)]
    sigma = max(float(np.median(valid_d)) if len(valid_d) > 0 else 1.0, 1e-6)
    for i in range(n):
        neigh_idx = indices[i, 1:]
        neigh_dist = distances[i, 1:]
        weights = np.exp(-(neigh_dist ** 2) / (2.0 * sigma * sigma))
        rows.extend([i] * len(neigh_idx))
        cols.extend(neigh_idx.tolist())
        vals.extend(weights.tolist())
    graph = sp.coo_matrix((vals, (rows, cols)), shape=(n, n), dtype=np.float32)
    graph = graph.maximum(graph.T).tocsr()
    row_sum = np.asarray(graph.sum(axis=1)).reshape(-1)
    row_sum[row_sum == 0] = 1.0
    graph = sp.diags(1.0 / row_sum) @ graph
    return graph.tocoo()


def compute_morans_i_batch(X, W):
    X = np.asarray(X, dtype=np.float32)
    W = W.tocsr() if sp.issparse(W) else sp.csr_matrix(W)
    n = X.shape[0]
    S0 = float(W.sum())
    X_centered = X - X.mean(axis=0, keepdims=True)
    WX = W @ X_centered
    num = (X_centered * WX).sum(axis=0)
    den = (X_centered ** 2).sum(axis=0) + 1e-12
    return (n / (S0 + 1e-12)) * (num / den)


def select_top_moran_common_features(adata_a, adata_b, topk, label="modality", forced_vars=None):
    common_vars = np.intersect1d(adata_a.var_names.astype(str), adata_b.var_names.astype(str))
    if len(common_vars) == 0:
        raise ValueError(f"No common {label} features found.")
    adata_a = adata_a[:, common_vars].copy()
    adata_b = adata_b[:, common_vars].copy()
    graph_a = build_spatial_graph_for_moran(adata_a, n_neighbors=num_neighbors)
    graph_b = build_spatial_graph_for_moran(adata_b, n_neighbors=num_neighbors)
    score = (compute_morans_i_batch(to_dense(adata_a.X), graph_a) + compute_morans_i_batch(to_dense(adata_b.X), graph_b)) / 2.0
    keep_n = min(topk, len(score))
    if forced_vars is None:
        keep_idx = np.argsort(score)[::-1][:keep_n]
    else:
        forced_set = {str(x) for x in forced_vars}
        forced_mask = np.array([v in forced_set for v in common_vars], dtype=bool)
        forced_idx = np.where(forced_mask)[0]
        ranked_idx = np.argsort(score)[::-1]
        if len(forced_idx) >= keep_n:
            forced_rank = np.argsort(score[forced_idx])[::-1]
            keep_idx = forced_idx[forced_rank[:keep_n]]
        else:
            need = keep_n - len(forced_idx)
            extra_idx = ranked_idx[~np.isin(ranked_idx, forced_idx)][:need]
            keep_idx = np.concatenate([forced_idx, extra_idx], axis=0)
    return adata_a[:, keep_idx].copy(), adata_b[:, keep_idx].copy()


def align_obs_pair(adata_a, adata_b, label="pair"):
    common_obs = np.intersect1d(adata_a.obs_names.astype(str), adata_b.obs_names.astype(str))
    if len(common_obs) == 0:
        raise ValueError(f"No common obs_names for {label}.")
    return adata_a[common_obs].copy(), adata_b[common_obs].copy()


def get_gene_id_list_or_names(adata):
    if "gene_ids" in adata.var.columns:
        return np.asarray(adata.var["gene_ids"]).astype(str).tolist()
    return np.asarray(adata.var_names).astype(str).tolist()


def maybe_pseudospot(adata):
    if USE_PSEUDOSPOT:
        return build_pseudospots(adata, grid_size=GRID_SIZE, he_key="he")
    return adata


# ============================================================================
# 5) Metabolite KB construction (from provided example)
# ============================================================================
def _extract_float(vals, keys):
    for k in keys:
        for x in vals.get(k, []):
            try:
                v = float(str(x))
                if np.isfinite(v):
                    return v
            except Exception:
                continue
    return None


def load_reactome_knowledge(reactome_dir):
    cache_path = os.path.join(KB_CACHE_DIR, "reactome_maps.pkl")
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            return pickle.load(f)
    chebi_to_pathways = defaultdict(set)
    ens_to_pathways = defaultdict(set)
    ncbi_to_pathways = defaultdict(set)
    chebi_path = os.path.join(reactome_dir, "ChEBI2Reactome_All_Levels.txt")
    ens_path = os.path.join(reactome_dir, "Ensembl2Reactome_All_Levels.txt")
    ncbi_path = os.path.join(reactome_dir, "NCBI2Reactome_All_Levels.txt")
    if os.path.exists(chebi_path):
        df = pd.read_csv(chebi_path, sep="\t", header=None, low_memory=False)
        for _, row in df.iterrows():
            cid = canonical_chebi_id(row.iloc[0]); pw = normalize_text(row.iloc[1]); species = normalize_text(row.iloc[5]) if len(row) > 5 else None
            if cid is None or pw is None:
                continue
            if species is not None and "HOMO SAPIENS" not in species.upper() and "MUS MUSCULUS" not in species.upper():
                continue
            chebi_to_pathways[cid].add(pw)
    if os.path.exists(ens_path):
        df = pd.read_csv(ens_path, sep="\t", header=None, low_memory=False)
        for _, row in df.iterrows():
            gid = canonical_ensembl_gene(row.iloc[0]); pw = normalize_text(row.iloc[1]); species = normalize_text(row.iloc[5]) if len(row) > 5 else None
            if gid is None or pw is None:
                continue
            if species is not None and "HOMO SAPIENS" not in species.upper() and "MUS MUSCULUS" not in species.upper():
                continue
            ens_to_pathways[gid].add(pw)
    if os.path.exists(ncbi_path):
        df = pd.read_csv(ncbi_path, sep="\t", header=None, low_memory=False)
        for _, row in df.iterrows():
            gid = canonical_ncbi_gene(row.iloc[0]); pw = normalize_text(row.iloc[1]); species = normalize_text(row.iloc[5]) if len(row) > 5 else None
            if gid is None or pw is None:
                continue
            if species is not None and "HOMO SAPIENS" not in species.upper() and "MUS MUSCULUS" not in species.upper():
                continue
            ncbi_to_pathways[gid].add(pw)
    out = (chebi_to_pathways, ens_to_pathways, ncbi_to_pathways)
    atomic_pickle_dump(out, cache_path)
    return out


def load_hmdb_metabolite_records(hmdb_metabolites_xml):
    cache_path = os.path.join(KB_CACHE_DIR, "hmdb_metabolites_records.pkl")
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            return pickle.load(f)
    alias_to_records = defaultdict(set)
    records = []
    mass_values = []
    if not os.path.exists(hmdb_metabolites_xml):
        return records, alias_to_records, np.array([], dtype=np.float64), np.array([], dtype=np.int32)
    for event, elem in ET.iterparse(hmdb_metabolites_xml, events=("end",)):
        if local_name(elem.tag) != "metabolite":
            continue
        vals = defaultdict(list)
        for child in elem.iter():
            nm = local_name(child.tag)
            txt = normalize_text(child.text)
            if txt is not None:
                vals[nm].append(txt)
        hmdb_id = None
        for k in ["accession", "hmdb_id"]:
            for x in vals.get(k, []):
                hm = canonical_hmdb_id(x)
                if hm is not None:
                    hmdb_id = hm
                    break
            if hmdb_id is not None:
                break
        aliases = set(); chebi_ids = set(); gene_symbols = set(); pathway_ids = set()
        exact_mass = _extract_float(vals, ["monisotopic_molecular_weight", "monisotopic_weight", "exact_mass", "average_molecular_weight"])
        for x in vals.get("name", []):
            aliases.add(normalize_alias(x)); aliases.add(normalize_token(x))
        for x in vals.get("synonym", []):
            aliases.add(normalize_alias(x)); aliases.add(normalize_token(x))
        if hmdb_id is not None:
            aliases.add(hmdb_id); aliases.add(normalize_alias(hmdb_id))
        for x in vals.get("chebi_id", []):
            cid = canonical_chebi_id(x)
            if cid is not None:
                chebi_ids.add(cid); aliases.add(cid); aliases.add(normalize_alias(cid))
        for x in vals.get("kegg_id", []):
            tok = normalize_token(x)
            if tok is not None:
                aliases.add(tok); aliases.add(normalize_alias(tok)); aliases.add(f"KEGG COMPOUND:{tok}")
        for x in vals.get("pathway_name", []):
            pathway_ids.add(normalize_text(x))
        for k in ["gene_name", "gene_symbol", "protein_accession", "protein_name", "name"]:
            for x in vals.get(k, []):
                tok = normalize_token(x)
                if tok is not None and len(tok) <= 40:
                    gene_symbols.add(tok)
        aliases = {x for x in aliases if x is not None}
        gene_symbols = {x for x in gene_symbols if x is not None}
        pathway_ids = {x for x in pathway_ids if x is not None}
        rid = len(records)
        records.append({"hmdb_id": hmdb_id, "aliases": aliases, "chebi_ids": chebi_ids, "genes": gene_symbols, "pathways": pathway_ids, "exact_mass": exact_mass})
        if exact_mass is not None:
            mass_values.append((float(exact_mass), rid))
        for a in aliases:
            alias_to_records[a].add(rid)
        elem.clear()
    if mass_values:
        mass_values = sorted(mass_values, key=lambda x: x[0])
        mass_arr = np.array([x[0] for x in mass_values], dtype=np.float64)
        rid_arr = np.array([x[1] for x in mass_values], dtype=np.int32)
    else:
        mass_arr = np.array([], dtype=np.float64)
        rid_arr = np.array([], dtype=np.int32)
    out = (records, alias_to_records, mass_arr, rid_arr)
    atomic_pickle_dump(out, cache_path)
    return out


def get_adducts():
    if ION_MODE == "positive":
        return ADDUCTS_POS
    if ION_MODE == "negative":
        return ADDUCTS_NEG
    return ADDUCTS_POS + ADDUCTS_NEG


def ppm_error(obs, theo):
    return abs(obs - theo) / max(theo, 1e-12) * 1e6


def candidate_weight(ppm, adduct_prior):
    return float(np.exp(-(ppm ** 2) / (2.0 * (PPM_SIGMA ** 2)))) * float(adduct_prior)


def annotate_feature_mz_to_hmdb(mz_value, mass_arr, rid_arr):
    candidates = []
    mz = float(mz_value)
    if mass_arr.size == 0:
        return candidates
    for adduct_name, delta, charge_abs, adduct_prior in get_adducts():
        neutral_mass = mz * charge_abs - delta
        tol = neutral_mass * PPM_TOL * 1e-6
        lo = bisect_left(mass_arr, neutral_mass - tol)
        hi = bisect_right(mass_arr, neutral_mass + tol)
        for idx in range(lo, hi):
            rid = int(rid_arr[idx])
            rec_mass = float(mass_arr[idx])
            ppm = ppm_error(neutral_mass, rec_mass)
            w = candidate_weight(ppm, adduct_prior)
            if w < MIN_CANDIDATE_WEIGHT:
                continue
            candidates.append((w, ppm, adduct_name, rid))
    if not candidates:
        return []
    merged = {}
    for w, ppm, adduct_name, rid in candidates:
        if rid not in merged or w > merged[rid][0]:
            merged[rid] = (w, ppm, adduct_name)
    out = [(rid, vals[0], vals[1], vals[2]) for rid, vals in merged.items()]
    out.sort(key=lambda x: (-x[1], x[2]))
    return out[:TOPK_CANDIDATES]


def build_gene_path_sets_from_ids(gene_id_list, ens_to_pathways, ncbi_to_pathways):
    gene_infos = []
    for gid in gene_id_list:
        raw = normalize_text(gid)
        ens = canonical_ensembl_gene(raw)
        ncbi = canonical_ncbi_gene(raw)
        pathways = set()
        if ens is not None:
            pathways.update(ens_to_pathways.get(ens, set()))
        if ncbi is not None:
            pathways.update(ncbi_to_pathways.get(ncbi, set()))
        gene_infos.append({"raw": raw, "ensembl": ens, "ncbi": ncbi, "pathways": pathways})
    return gene_infos


def build_full_metabolite_kb(metabolite_mz_names, gene_id_list, gene_name_list=None, enabled_sources=None, cache_tag="default"):
    enabled = set([str(x).strip().lower() for x in (enabled_sources or ["hmdb", "reactome", "chebi"]) if str(x).strip()])
    use_hmdb = "hmdb" in enabled
    use_reactome = "reactome" in enabled
    use_chebi = ("chebi" in enabled) and bool(USE_CHEBI_RELATION) and use_reactome

    cache_key = stable_hash(
        "full_met_kb_v3_agent",
        cache_tag,
        sorted(list(map(str, metabolite_mz_names))),
        sorted(list(map(str, gene_id_list))),
        sorted(list(map(str, gene_name_list or []))),
        sorted(list(enabled)),
        ION_MODE, PPM_TOL, TOPK_CANDIDATES, USE_CHEBI_RELATION,
        KB_DIRECT_WEIGHT, KB_PATHWAY_WEIGHT,
        KB_GENE_GRAPH_PATHWAY_WEIGHT, KB_GENE_GRAPH_HMDB_WEIGHT,
        KB_MET_GRAPH_PATHWAY_WEIGHT, KB_MET_GRAPH_CHEBI_WEIGHT, KB_MET_GRAPH_HMDB_WEIGHT
    )
    cache_path = os.path.join(KB_CACHE_DIR, f"full_met_kb_{cache_key}.pkl")
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    if use_reactome:
        chebi_to_pathways, ens_to_pathways, ncbi_to_pathways = load_reactome_knowledge(REACTOME_DIR)
    else:
        chebi_to_pathways, ens_to_pathways, ncbi_to_pathways = {}, {}, {}

    hmdb_xml_path = os.path.join(HMDB_DIR, "hmdb_metabolites.xml")
    if use_hmdb:
        hmdb_records, _, mass_arr, rid_arr = load_hmdb_metabolite_records(hmdb_xml_path)
    else:
        hmdb_records, mass_arr, rid_arr = [], np.array([], dtype=np.float64), np.array([], dtype=np.int32)

    gene_infos = build_gene_path_sets_from_ids(gene_id_list, ens_to_pathways, ncbi_to_pathways)
    n_gene = len(gene_infos)

    met_infos = []
    for mz_name in metabolite_mz_names:
        try:
            mz = float(str(mz_name))
        except Exception:
            mz = None
        candidates = annotate_feature_mz_to_hmdb(mz, mass_arr, rid_arr) if (mz is not None and use_hmdb) else []
        total_w = sum(x[1] for x in candidates) + 1e-12
        hmdb_weights = {x[0]: x[1] / total_w for x in candidates}
        hmdb_genes_weighted = defaultdict(float)
        met_pathways_weighted = defaultdict(float)
        chebi_ids = set()
        for rid, w_norm in hmdb_weights.items():
            rec = hmdb_records[rid]
            for cid in rec["chebi_ids"]:
                chebi_ids.add(cid)
            for g in rec["genes"]:
                hmdb_genes_weighted[normalize_token(g)] += w_norm
            for pw in rec["pathways"]:
                met_pathways_weighted[pw] += w_norm
            if use_chebi:
                for cid in rec["chebi_ids"]:
                    for pw in chebi_to_pathways.get(cid, set()):
                        met_pathways_weighted[pw] += 0.5 * w_norm
        met_infos.append({
            "mz_name": str(mz_name), "candidate_weights": hmdb_weights, "chebi_ids": chebi_ids,
            "hmdb_gene_weights": dict(hmdb_genes_weighted), "pathway_weights": dict(met_pathways_weighted)
        })

    gene_token_to_idx = {}
    for idx, gid in enumerate(gene_id_list):
        tok = normalize_token(gid)
        if tok is not None:
            gene_token_to_idx[tok] = idx
    if gene_name_list is not None:
        for idx, gname in enumerate(gene_name_list):
            tok = normalize_token(gname)
            if tok is not None and tok not in gene_token_to_idx:
                gene_token_to_idx[tok] = idx

    n_met = len(met_infos)
    M_direct = np.zeros((n_met, n_gene), dtype=np.float32)
    M_path = np.zeros((n_met, n_gene), dtype=np.float32)
    for i, met in enumerate(met_infos):
        for g_tok, w in met["hmdb_gene_weights"].items():
            j = gene_token_to_idx.get(g_tok, None)
            if j is not None:
                M_direct[i, j] += float(w)
    gene_path_sets = [g["pathways"] for g in gene_infos]
    for i, met in enumerate(met_infos):
        mpw = met["pathway_weights"]
        if not mpw:
            continue
        for j, gpw in enumerate(gene_path_sets):
            if not gpw:
                continue
            shared = gpw.intersection(mpw.keys())
            if len(shared) >= MIN_SHARED_PATHWAY:
                M_path[i, j] = float(sum(mpw[pw] for pw in shared))
    M = KB_DIRECT_WEIGHT * row_normalize_matrix(M_direct) + KB_PATHWAY_WEIGHT * row_normalize_matrix(M_path)
    if M.sum() > 0:
        M = row_normalize_matrix(M)
    M_rev = row_normalize_matrix(M.T) if M.sum() > 0 else np.zeros((n_gene, n_met), dtype=np.float32)

    G_gene_path = np.zeros((n_gene, n_gene), dtype=np.float32)
    for i in range(n_gene):
        for j in range(i + 1, n_gene):
            w = len(gene_path_sets[i].intersection(gene_path_sets[j]))
            if w > 0:
                G_gene_path[i, j] = G_gene_path[j, i] = float(w)
    G_gene_hmdb = np.zeros((n_gene, n_gene), dtype=np.float32)
    for met in met_infos:
        idx_weights = []
        for g_tok, w in met["hmdb_gene_weights"].items():
            j = gene_token_to_idx.get(g_tok, None)
            if j is not None:
                idx_weights.append((j, float(w)))
        for a in range(len(idx_weights)):
            for b in range(a + 1, len(idx_weights)):
                ia, wa = idx_weights[a]
                ib, wb = idx_weights[b]
                v = wa * wb
                G_gene_hmdb[ia, ib] += v
                G_gene_hmdb[ib, ia] += v
    gene_path_term = KB_GENE_GRAPH_PATHWAY_WEIGHT * G_gene_path if use_reactome else 0.0
    gene_hmdb_term = KB_GENE_GRAPH_HMDB_WEIGHT * G_gene_hmdb if use_hmdb else 0.0
    G_gene = normalize_adjacency_with_selfloop(gene_path_term + gene_hmdb_term)

    G_met_path = np.zeros((n_met, n_met), dtype=np.float32)
    G_met_hmdb = np.zeros((n_met, n_met), dtype=np.float32)
    for i in range(n_met):
        keys_i = set(met_infos[i]["pathway_weights"].keys())
        wi = met_infos[i]["candidate_weights"]
        for j in range(i + 1, n_met):
            keys_j = set(met_infos[j]["pathway_weights"].keys())
            shared_path = keys_i.intersection(keys_j)
            if shared_path:
                G_met_path[i, j] = G_met_path[j, i] = float(sum(min(met_infos[i]["pathway_weights"][pw], met_infos[j]["pathway_weights"][pw]) for pw in shared_path))
            wj = met_infos[j]["candidate_weights"]
            shared_hmdb = set(wi.keys()).intersection(wj.keys())
            if shared_hmdb:
                G_met_hmdb[i, j] = G_met_hmdb[j, i] = float(sum(min(wi[r], wj[r]) for r in shared_hmdb))
    met_path_term = KB_MET_GRAPH_PATHWAY_WEIGHT * G_met_path if use_reactome else 0.0
    met_chebi_term = KB_MET_GRAPH_CHEBI_WEIGHT * G_met_path if use_chebi else 0.0
    met_hmdb_term = KB_MET_GRAPH_HMDB_WEIGHT * G_met_hmdb if use_hmdb else 0.0
    G_met = normalize_adjacency_with_selfloop(met_path_term + met_chebi_term + met_hmdb_term)

    stats = {
        "num_metabolites": n_met,
        "num_genes": n_gene,
        "annotated_metabolites_by_mass": int(sum(len(x["candidate_weights"]) > 0 for x in met_infos)),
        "annotated_metabolites_with_pathways": int(sum(len(x["pathway_weights"]) > 0 for x in met_infos)),
        "hmdb_direct_links": int((M_direct > 0).sum()),
        "pathway_links": int((M_path > 0).sum()),
        "enabled_sources": sorted(list(enabled)),
    }
    out = {
        "metabolite_gene_prior": M.astype(np.float32),
        "gene_metabolite_prior": M_rev.astype(np.float32),
        "gene_gene_graph": G_gene.astype(np.float32),
        "metabolite_metabolite_graph": G_met.astype(np.float32),
        "stats": stats,
    }
    atomic_pickle_dump(out, cache_path)
    return out


def validate_kb_pack_for_model(kb_pack, n_protein, n_gene):
    M_pg = np.asarray(kb_pack["metabolite_gene_prior"], dtype=np.float32)
    M_gp = np.asarray(kb_pack["gene_metabolite_prior"], dtype=np.float32)
    G_gene = np.asarray(kb_pack["gene_gene_graph"], dtype=np.float32)
    G_met = np.asarray(kb_pack["metabolite_metabolite_graph"], dtype=np.float32)
    if M_pg.shape != (n_protein, n_gene):
        raise ValueError(f"metabolite_gene_prior shape mismatch: expected {(n_protein, n_gene)}, got {M_pg.shape}")
    if M_gp.shape != (n_gene, n_protein):
        raise ValueError(f"gene_metabolite_prior shape mismatch: expected {(n_gene, n_protein)}, got {M_gp.shape}")
    if G_gene.shape != (n_gene, n_gene):
        raise ValueError(f"gene_gene_graph shape mismatch: expected {(n_gene, n_gene)}, got {G_gene.shape}")
    if G_met.shape != (n_protein, n_protein):
        raise ValueError(f"metabolite_metabolite_graph shape mismatch: expected {(n_protein, n_protein)}, got {G_met.shape}")
    return M_pg, M_gp, G_gene, G_met


def estimate_kb_quality_and_lambdas(kb_stats, base_lambda_kb, base_lambda_graph):
    n_met = max(int(kb_stats.get("num_metabolites", 0)), 1)
    n_gene = max(int(kb_stats.get("num_genes", 0)), 1)
    annotated_by_mass = float(kb_stats.get("annotated_metabolites_by_mass", 0))
    annotated_with_pathways = float(kb_stats.get("annotated_metabolites_with_pathways", 0))
    pathway_links = float(kb_stats.get("pathway_links", 0))
    hmdb_direct_links = float(kb_stats.get("hmdb_direct_links", 0))
    mass_ratio = annotated_by_mass / n_met
    pathway_ratio = annotated_with_pathways / n_met
    link_density = pathway_links / max(n_met * n_gene, 1.0)
    hmdb_flag = 1.0 if hmdb_direct_links > 0 else 0.0
    quality = 0.20 * min(mass_ratio, 1.0) + 0.50 * min(pathway_ratio * 2.0, 1.0) + 0.20 * min(link_density * 500.0, 1.0) + 0.10 * hmdb_flag
    quality = float(np.clip(quality, 0.10, 1.00))
    if not USE_KB_QUALITY_SCALING:
        return {
            "lambda_kb_eff": float(base_lambda_kb),
            "lambda_graph_eff": float(base_lambda_graph),
            "quality": quality,
            "scaling_mode": "fixed",
        }
    return {
        "lambda_kb_eff": base_lambda_kb * quality,
        "lambda_graph_eff": base_lambda_graph * max(0.25, quality),
        "quality": quality,
        "scaling_mode": "quality_scaled",
    }


# ============================================================================
# 6) Metabolomics aligned data + latent
# ============================================================================
def derive_forced_feature_lists_from_kb(met_adata, rna_adata, enabled_sources=None, task_id="warmup"):
    kb_pack = build_full_metabolite_kb(
        metabolite_mz_names=np.asarray(met_adata.var_names).astype(str).tolist(),
        gene_id_list=get_gene_id_list_or_names(rna_adata),
        gene_name_list=np.asarray(rna_adata.var_names).astype(str).tolist(),
        enabled_sources=enabled_sources,
        cache_tag=task_id,
    )
    M = np.asarray(kb_pack["metabolite_gene_prior"], dtype=np.float32)
    met_names = np.asarray(met_adata.var_names).astype(str)
    gene_names = np.asarray(rna_adata.var_names).astype(str)
    forced_metabolites = met_names[np.where(M.sum(axis=1) > 0)[0]].tolist()
    forced_genes = gene_names[np.where(M.sum(axis=0) > 0)[0]].tolist()
    return {
        "forced_metabolites": forced_metabolites,
        "forced_genes": forced_genes,
        "kb_stats": kb_pack["stats"],
        "selected_sources": list(enabled_sources or []),
    }


def prepare_aligned_data():
    local_status = get_local_met_source_status()
    agent_tag = "1" if str(os.getenv("KNOWLEDGE_AGENT_ENABLE_AGENT", "0")).strip().lower() in {"1", "true", "yes"} else "0"
    align_key = stable_hash(
        folder_B1, folder_C1, MET_TOPK, GENE_TOPK, resolution, image_encoder,
        USE_PSEUDOSPOT, GRID_SIZE, USE_KB_FEATURE_WARMUP, ION_MODE, PPM_TOL,
        json.dumps(local_status, sort_keys=True), agent_tag
    )
    align_cache_path = os.path.join(ALIGNED_CACHE_DIR, f"aligned_{align_key}.pkl")
    if os.path.exists(align_cache_path):
        with open(align_cache_path, "rb") as f:
            return pickle.load(f)

    C1_rna_raw = preprocess_rna_with_he_cached(folder_C1, "C1", resolution=resolution, image_encoder=image_encoder, device=device)
    B1_met_raw = preprocess_metabolite_with_he_cached(folder_B1, "B1", resolution=resolution, image_encoder=image_encoder, device=device)
    C1_met_raw = preprocess_metabolite_with_he_cached(folder_C1, "C1", resolution=resolution, image_encoder=image_encoder, device=device)
    B1_rna_raw = preprocess_rna_with_he_cached(folder_B1, "B1", resolution=resolution, image_encoder=image_encoder, device=device)

    forced_metabolites = None
    forced_genes = None
    kb_warmup = None
    if USE_KB_FEATURE_WARMUP:
        warmup_sources, warmup_report = select_sources_for_met_task(
            task_id="warmup_he_metabolism_to_gene",
            source_modalities=["he", "metabolism"],
            target_modality="gene",
            orchestration_dir=KB_ORCHESTRATION_DIR,
        )
        kb_warmup = derive_forced_feature_lists_from_kb(
            B1_met_raw,
            C1_rna_raw,
            enabled_sources=warmup_sources,
            task_id="warmup_he_metabolism_to_gene",
        )
        kb_warmup["orchestration_report"] = warmup_report
        forced_metabolites = kb_warmup["forced_metabolites"]
        forced_genes = kb_warmup["forced_genes"]

    C1_met, B1_met = select_top_moran_common_features(C1_met_raw, B1_met_raw, topk=MET_TOPK, label="metabolites", forced_vars=forced_metabolites)
    C1_rna, B1_rna = select_top_moran_common_features(C1_rna_raw, B1_rna_raw, topk=GENE_TOPK, label="genes", forced_vars=forced_genes)
    C1_rna, C1_met = align_obs_pair(C1_rna, C1_met, label="C1")
    B1_rna, B1_met = align_obs_pair(B1_rna, B1_met, label="B1")

    C1_rna = maybe_pseudospot(C1_rna); C1_met = maybe_pseudospot(C1_met)
    B1_rna = maybe_pseudospot(B1_rna); B1_met = maybe_pseudospot(B1_met)
    B1_met.obsm["protein"] = to_dense(B1_met.X).astype(np.float32)
    C1_met.obsm["protein"] = to_dense(C1_met.X).astype(np.float32)

    graph_B1_met = build_training_graph(B1_met, "B1_met")
    graph_C1_rna = build_training_graph(C1_rna, "C1_rna")
    graph_C1_met = build_training_graph(C1_met, "C1_met")
    graph_B1_rna = build_training_graph(B1_rna, "B1_rna")
    graph_eval_B1_met = build_eval_graph_from_adata(B1_met)
    graph_eval_C1_rna = build_eval_graph_from_adata(C1_rna)

    pack = {
        "B1_met": B1_met, "B1_rna": B1_rna, "C1_rna": C1_rna, "C1_met": C1_met,
        "graph_B1_met": graph_B1_met, "graph_C1_rna": graph_C1_rna, "graph_C1_met": graph_C1_met, "graph_B1_rna": graph_B1_rna,
        "graph_eval_B1_met": graph_eval_B1_met, "graph_eval_C1_rna": graph_eval_C1_rna,
        "kb_warmup": kb_warmup,
    }
    atomic_pickle_dump(pack, align_cache_path)
    return pack


def build_or_load_gene_pca_latent(B1_rna, C1_rna):
    key = stable_hash("gene_pca_latent_v1", B1_rna.shape, C1_rna.shape, GENE_LATENT_DIM)
    comp_path = os.path.join(LATENT_CACHE_DIR, f"gene_pca_comp_{key}.npy")
    mean_path = os.path.join(LATENT_CACHE_DIR, f"gene_pca_mean_{key}.npy")
    b1_lat_path = os.path.join(LATENT_CACHE_DIR, f"B1_gene_latent_{key}.npy")
    c1_lat_path = os.path.join(LATENT_CACHE_DIR, f"C1_gene_latent_{key}.npy")
    if all(os.path.exists(p) for p in [comp_path, mean_path, b1_lat_path, c1_lat_path]):
        return {"components": np.load(comp_path), "mean": np.load(mean_path), "B1_latent": np.load(b1_lat_path), "C1_latent": np.load(c1_lat_path)}
    B1_gene = to_dense(B1_rna.X).astype(np.float32)
    C1_gene = to_dense(C1_rna.X).astype(np.float32)
    gene_train = np.vstack([B1_gene, C1_gene]).astype(np.float32)
    latent_dim = min(int(GENE_LATENT_DIM), int(gene_train.shape[1]), max(int(gene_train.shape[0]) - 1, 1))
    max_fit_cells = 20000
    gene_fit = gene_train[np.random.RandomState(0).choice(gene_train.shape[0], max_fit_cells, replace=False)] if gene_train.shape[0] > max_fit_cells else gene_train
    pca = PCA(n_components=latent_dim, svd_solver="randomized", random_state=0)
    pca.fit(gene_fit)
    B1_latent = pca.transform(B1_gene).astype(np.float32)
    C1_latent = pca.transform(C1_gene).astype(np.float32)
    np.save(comp_path, pca.components_); np.save(mean_path, pca.mean_); np.save(b1_lat_path, B1_latent); np.save(c1_lat_path, C1_latent)
    return {"components": pca.components_, "mean": pca.mean_, "B1_latent": B1_latent, "C1_latent": C1_latent}


# ============================================================================
# 7) Metabolomics tasks (task5~task8)
# ============================================================================
def run_metabolomics_tasks_for_seed(repeat_idx, seed, data_pack, gene_lat_pack, orchestration_dir=None):
    B1_met = data_pack["B1_met"]; B1_rna = data_pack["B1_rna"]
    C1_rna = data_pack["C1_rna"]; C1_met = data_pack["C1_met"]
    graph_B1_met = data_pack["graph_B1_met"]; graph_C1_rna = data_pack["graph_C1_rna"]
    graph_C1_met = data_pack["graph_C1_met"]; graph_B1_rna = data_pack["graph_B1_rna"]
    graph_eval_B1_met = data_pack.get("graph_eval_B1_met")
    graph_eval_C1_rna = data_pack.get("graph_eval_C1_rna")
    if graph_eval_B1_met is None:
        graph_eval_B1_met = build_eval_graph_from_adata(B1_met)
    if graph_eval_C1_rna is None:
        graph_eval_C1_rna = build_eval_graph_from_adata(C1_rna)

    if orchestration_dir is None:
        orchestration_dir = KB_ORCHESTRATION_DIR
    os.makedirs(orchestration_dir, exist_ok=True)

    base_trainer_kwargs = dict(
        he_dim=B1_met.obsm["he"].shape[1],
        protein_dim=B1_met.obsm["protein"].shape[1],
        gene_dim=C1_rna.X.shape[1],
        gene_pca_components=gene_lat_pack["components"],
        gene_pca_mean=gene_lat_pack["mean"],
        hidden_dim=TRAIN_HIDDEN_DIM,
        num_layers=TRAIN_NUM_LAYERS,
        epochs=TRAIN_EPOCHS,
        lr=TRAIN_LR,
        dropout=TRAIN_DROPOUT,
        device=device,
        lambda_target=LAMBDA_TARGET,
        lambda_pcc=LAMBDA_PCC,
        lambda_latent=LAMBDA_LATENT,
        lambda_recon=LAMBDA_RECON,
        lambda_ortho=LAMBDA_ORTHO,
        lambda_dgi=LAMBDA_DGI,
        lambda_cycle=0.2,
        lambda_partial_cycle=0.2,
        lambda_bridge_align=0.5,
        lambda_align=LAMBDA_ALIGN,
        target_soft_alpha=TARGET_SOFT_ALPHA,
        kb_ct_prior_mix=KB_CT_PRIOR_MIX,
        lambda_full=LAMBDA_FULL,
        lambda_partial=LAMBDA_PARTIAL,
        align_max_points=ALIGN_MAX_POINTS,
        use_amp=False,
        disable_dynamic_retriever=DISABLE_DYNAMIC_RETRIEVER,
        disable_kb_fusion=DISABLE_KB_FUSION,
        disable_soft_gate=DISABLE_SOFT_GATE,
        disable_residual_refine=DISABLE_RESIDUAL_REFINE,
    )

    def _resolve_task_kb(task_id, source_modalities, target_modality):
        selected_sources, report = select_sources_for_met_task(
            task_id=task_id,
            source_modalities=source_modalities,
            target_modality=target_modality,
            repeat_idx=repeat_idx,
            seed=seed,
            orchestration_dir=orchestration_dir,
        )

        if (not USE_KB_MODEL_INJECTION) or (not selected_sources):
            return {
                "protein_gene_prior": None,
                "gene_protein_prior": None,
                "gene_gene_kb_graph": None,
                "metabolite_metabolite_kb_graph": None,
                "kb_weight_info": {"lambda_kb_eff": 0.0, "lambda_graph_eff": 0.0},
                "selected_sources": selected_sources,
                "report": report,
            }

        kb_pack = build_full_metabolite_kb(
            metabolite_mz_names=np.asarray(B1_met.var_names).astype(str).tolist(),
            gene_id_list=get_gene_id_list_or_names(C1_rna),
            gene_name_list=np.asarray(C1_rna.var_names).astype(str).tolist(),
            enabled_sources=selected_sources,
            cache_tag=task_id,
        )
        protein_gene_prior, gene_protein_prior, gene_gene_kb_graph, metabolite_metabolite_kb_graph = validate_kb_pack_for_model(
            kb_pack,
            n_protein=B1_met.obsm["protein"].shape[1],
            n_gene=C1_rna.X.shape[1],
        )
        kb_weight_info = estimate_kb_quality_and_lambdas(kb_pack["stats"], base_lambda_kb=LAMBDA_KB, base_lambda_graph=LAMBDA_GRAPH)

        return {
            "protein_gene_prior": protein_gene_prior,
            "gene_protein_prior": gene_protein_prior,
            "gene_gene_kb_graph": gene_gene_kb_graph,
            "metabolite_metabolite_kb_graph": metabolite_metabolite_kb_graph,
            "kb_weight_info": kb_weight_info,
            "selected_sources": selected_sources,
            "report": report,
        }

    results = []

    # Task5: HE+Gene -> Metabolism
    kb5 = _resolve_task_kb("task5_he_gene_to_metabolism", ["he", "gene"], "metabolism")
    trainer5_kwargs = copy.deepcopy(base_trainer_kwargs)
    trainer5_kwargs.update(
        protein_gene_prior=kb5["protein_gene_prior"],
        gene_protein_prior=kb5["gene_protein_prior"],
        gene_gene_kb_graph=kb5["gene_gene_kb_graph"],
        protein_protein_kb_graph=kb5["metabolite_metabolite_kb_graph"],
        lambda_kb=kb5["kb_weight_info"]["lambda_kb_eff"],
        lambda_graph=kb5["kb_weight_info"]["lambda_graph_eff"],
    )
    task5_cache = build_met_task_cache_path(
        "task5_he_gene_to_metabolism",
        repeat_idx=repeat_idx,
        seed=seed,
        selected_sources=kb5["selected_sources"],
        kb_weight_info=kb5["kb_weight_info"],
        target_shape=B1_met.shape,
        protocol_tag="fullpartial",
        full_use_obs=True,
        partial_use_obs=True,
    )
    if (not FORCE_TASK_RETRAIN) and os.path.exists(task5_cache):
        with open(task5_cache, "rb") as f:
            row5 = pickle.load(f)["row"]
        row5 = refresh_row_selection_metadata(
            row=row5,
            selected_sources=kb5["selected_sources"],
            report=kb5["report"],
            domain="metabolomics",
        )
        print(f">>> [TASK-CACHE] hit: task5_he_gene_to_metabolism rep={repeat_idx} seed={seed}", flush=True)
    else:
        trainer5 = se.AKGOmicsFullPartialProtocol(
            target_task="protein",
            full_he=C1_rna.obsm["he"], full_graph=graph_C1_rna,
            full_gene=C1_rna.X, full_protein=C1_met.X,
            partial_he=B1_rna.obsm["he"], partial_graph=graph_B1_rna,
            partial_gene_obs=B1_rna.X, partial_protein_obs=None,
            partial_gene_eval=None, partial_protein_eval=B1_met.X,
            full_gene_latent=gene_lat_pack["C1_latent"],
            full_slice_id=1, partial_slice_id=0,
            full_use_obs=True, partial_use_obs=True,
            save_path=None, seed=int(seed), **trainer5_kwargs,
        )
        trainer5.train()
        pred5 = trainer5.infer_partial()
        m5 = compute_three_metrics(to_dense(B1_met.X).astype(np.float32), pred5["pred_full"], graph_eval_B1_met)
        row5 = {
            "experiment_id": "E2_unified8", "repeat_idx": int(repeat_idx), "seed": int(seed),
            "task": "task5_he_gene_to_metabolism", "task_order": 5, "target_task": "metabolism",
            "full_use_obs": True, "partial_use_obs": True, "graph_alpha": float(FUSION_ALPHA),
            "PCC": m5["PCC"], "SSIM": m5["SSIM"], "CMD": m5["CMD"], "RMSE": m5["RMSE"], "domain": "metabolomics",
            "selected_sources": ";".join(kb5["selected_sources"]),
            "kb_select_mode": str(kb5["report"].get("selection", {}).get("mode", "")),
        }
        atomic_pickle_dump({"row": row5}, task5_cache)
        print(f">>> [TASK-CACHE] saved: {task5_cache}", flush=True)
    results.append({
        **row5
    })

    # Task6: HE+Metabolism -> Gene
    kb6 = _resolve_task_kb("task6_he_metabolism_to_gene", ["he", "metabolism"], "gene")
    trainer6_kwargs = copy.deepcopy(base_trainer_kwargs)
    trainer6_kwargs.update(
        protein_gene_prior=kb6["protein_gene_prior"],
        gene_protein_prior=kb6["gene_protein_prior"],
        gene_gene_kb_graph=kb6["gene_gene_kb_graph"],
        protein_protein_kb_graph=kb6["metabolite_metabolite_kb_graph"],
        lambda_kb=kb6["kb_weight_info"]["lambda_kb_eff"],
        lambda_graph=kb6["kb_weight_info"]["lambda_graph_eff"],
    )
    task6_cache = build_met_task_cache_path(
        "task6_he_metabolism_to_gene",
        repeat_idx=repeat_idx,
        seed=seed,
        selected_sources=kb6["selected_sources"],
        kb_weight_info=kb6["kb_weight_info"],
        target_shape=C1_rna.shape,
        protocol_tag="fullpartial",
        full_use_obs=True,
        partial_use_obs=True,
    )
    if (not FORCE_TASK_RETRAIN) and os.path.exists(task6_cache):
        with open(task6_cache, "rb") as f:
            row6 = pickle.load(f)["row"]
        row6 = refresh_row_selection_metadata(
            row=row6,
            selected_sources=kb6["selected_sources"],
            report=kb6["report"],
            domain="metabolomics",
        )
        print(f">>> [TASK-CACHE] hit: task6_he_metabolism_to_gene rep={repeat_idx} seed={seed}", flush=True)
    else:
        trainer6 = se.AKGOmicsFullPartialProtocol(
            target_task="gene",
            full_he=B1_met.obsm["he"], full_graph=graph_B1_met,
            full_gene=B1_rna.X, full_protein=B1_met.obsm["protein"],
            partial_he=C1_met.obsm["he"], partial_graph=graph_C1_met,
            partial_gene_obs=None, partial_protein_obs=C1_met.obsm["protein"],
            partial_gene_eval=C1_rna.X, partial_protein_eval=None,
            full_gene_latent=gene_lat_pack["B1_latent"],
            full_slice_id=0, partial_slice_id=1,
            full_use_obs=True, partial_use_obs=True,
            save_path=None, seed=int(seed), **trainer6_kwargs,
        )
        trainer6.train()
        pred6 = trainer6.infer_partial()
        m6 = compute_three_metrics(to_dense(C1_rna.X).astype(np.float32), pred6["pred_full"], graph_eval_C1_rna)
        row6 = {
            "experiment_id": "E2_unified8", "repeat_idx": int(repeat_idx), "seed": int(seed),
            "task": "task6_he_metabolism_to_gene", "task_order": 6, "target_task": "gene",
            "full_use_obs": True, "partial_use_obs": True, "graph_alpha": float(FUSION_ALPHA),
            "PCC": m6["PCC"], "SSIM": m6["SSIM"], "CMD": m6["CMD"], "RMSE": m6["RMSE"], "domain": "metabolomics",
            "selected_sources": ";".join(kb6["selected_sources"]),
            "kb_select_mode": str(kb6["report"].get("selection", {}).get("mode", "")),
        }
        atomic_pickle_dump({"row": row6}, task6_cache)
        print(f">>> [TASK-CACHE] saved: {task6_cache}", flush=True)
    results.append({
        **row6
    })

    # Task7: HE -> Metabolism
    kb7 = _resolve_task_kb("task7_he_to_metabolism", ["he"], "metabolism")
    trainer7_kwargs = copy.deepcopy(base_trainer_kwargs)
    trainer7_kwargs.update(
        protein_gene_prior=kb7["protein_gene_prior"],
        gene_protein_prior=kb7["gene_protein_prior"],
        gene_gene_kb_graph=kb7["gene_gene_kb_graph"],
        protein_protein_kb_graph=kb7["metabolite_metabolite_kb_graph"],
        lambda_kb=kb7["kb_weight_info"]["lambda_kb_eff"],
        lambda_graph=kb7["kb_weight_info"]["lambda_graph_eff"],
    )
    task7_cache = build_met_task_cache_path(
        "task7_he_to_metabolism",
        repeat_idx=repeat_idx,
        seed=seed,
        selected_sources=kb7["selected_sources"],
        kb_weight_info=kb7["kb_weight_info"],
        target_shape=B1_met.shape,
        protocol_tag="fullpartial",
        full_use_obs=False,
        partial_use_obs=False,
    )
    if (not FORCE_TASK_RETRAIN) and os.path.exists(task7_cache):
        with open(task7_cache, "rb") as f:
            row7 = pickle.load(f)["row"]
        row7 = refresh_row_selection_metadata(
            row=row7,
            selected_sources=kb7["selected_sources"],
            report=kb7["report"],
            domain="metabolomics",
        )
        print(f">>> [TASK-CACHE] hit: task7_he_to_metabolism rep={repeat_idx} seed={seed}", flush=True)
    else:
        trainer7 = se.AKGOmicsFullPartialProtocol(
            target_task="protein",
            full_he=C1_rna.obsm["he"], full_graph=graph_C1_rna,
            full_gene=C1_rna.X, full_protein=C1_met.X,
            partial_he=B1_rna.obsm["he"], partial_graph=graph_B1_rna,
            partial_gene_obs=None, partial_protein_obs=None,
            partial_gene_eval=None, partial_protein_eval=B1_met.X,
            full_gene_latent=gene_lat_pack["C1_latent"],
            full_slice_id=1, partial_slice_id=0,
            full_use_obs=False, partial_use_obs=False,
            save_path=None, seed=int(seed), **trainer7_kwargs,
        )
        trainer7.train()
        pred7 = trainer7.infer_partial()
        m7 = compute_three_metrics(to_dense(B1_met.X).astype(np.float32), pred7["pred_full"], graph_eval_B1_met)
        row7 = {
            "experiment_id": "E2_unified8", "repeat_idx": int(repeat_idx), "seed": int(seed),
            "task": "task7_he_to_metabolism", "task_order": 7, "target_task": "metabolism",
            "full_use_obs": False, "partial_use_obs": False, "graph_alpha": float(FUSION_ALPHA),
            "PCC": m7["PCC"], "SSIM": m7["SSIM"], "CMD": m7["CMD"], "RMSE": m7["RMSE"], "domain": "metabolomics",
            "selected_sources": ";".join(kb7["selected_sources"]),
            "kb_select_mode": str(kb7["report"].get("selection", {}).get("mode", "")),
        }
        atomic_pickle_dump({"row": row7}, task7_cache)
        print(f">>> [TASK-CACHE] saved: {task7_cache}", flush=True)
    results.append({
        **row7
    })

    # Task8: In metabolomics dataset, HE -> Gene
    kb8 = _resolve_task_kb("task8_he_to_gene_in_metabolomics", ["he"], "gene")
    trainer8_kwargs = copy.deepcopy(base_trainer_kwargs)
    trainer8_kwargs.update(
        protein_gene_prior=kb8["protein_gene_prior"],
        gene_protein_prior=kb8["gene_protein_prior"],
        gene_gene_kb_graph=kb8["gene_gene_kb_graph"],
        protein_protein_kb_graph=kb8["metabolite_metabolite_kb_graph"],
        lambda_kb=kb8["kb_weight_info"]["lambda_kb_eff"],
        lambda_graph=kb8["kb_weight_info"]["lambda_graph_eff"],
    )
    task8_cache = build_met_task_cache_path(
        "task8_he_to_gene_in_metabolomics",
        repeat_idx=repeat_idx,
        seed=seed,
        selected_sources=kb8["selected_sources"],
        kb_weight_info=kb8["kb_weight_info"],
        target_shape=C1_rna.shape,
        protocol_tag="fullpartial",
        full_use_obs=False,
        partial_use_obs=False,
    )
    if (not FORCE_TASK_RETRAIN) and os.path.exists(task8_cache):
        with open(task8_cache, "rb") as f:
            row8 = pickle.load(f)["row"]
        row8 = refresh_row_selection_metadata(
            row=row8,
            selected_sources=kb8["selected_sources"],
            report=kb8["report"],
            domain="metabolomics",
        )
        print(f">>> [TASK-CACHE] hit: task8_he_to_gene_in_metabolomics rep={repeat_idx} seed={seed}", flush=True)
    else:
        trainer8 = se.AKGOmicsFullPartialProtocol(
            target_task="gene",
            full_he=B1_met.obsm["he"], full_graph=graph_B1_met,
            full_gene=B1_rna.X, full_protein=B1_met.obsm["protein"],
            partial_he=C1_met.obsm["he"], partial_graph=graph_C1_met,
            partial_gene_obs=None, partial_protein_obs=None,
            partial_gene_eval=C1_rna.X, partial_protein_eval=None,
            full_gene_latent=gene_lat_pack["B1_latent"],
            full_slice_id=0, partial_slice_id=1,
            full_use_obs=False, partial_use_obs=False,
            save_path=None, seed=int(seed), **trainer8_kwargs,
        )
        trainer8.train()
        pred8 = trainer8.infer_partial()
        m8 = compute_three_metrics(to_dense(C1_rna.X).astype(np.float32), pred8["pred_full"], graph_eval_C1_rna)
        row8 = {
            "experiment_id": "E2_unified8", "repeat_idx": int(repeat_idx), "seed": int(seed),
            "task": "task8_he_to_gene_in_metabolomics", "task_order": 8, "target_task": "gene",
            "full_use_obs": False, "partial_use_obs": False, "graph_alpha": float(FUSION_ALPHA),
            "PCC": m8["PCC"], "SSIM": m8["SSIM"], "CMD": m8["CMD"], "RMSE": m8["RMSE"], "domain": "metabolomics",
            "selected_sources": ";".join(kb8["selected_sources"]),
            "kb_select_mode": str(kb8["report"].get("selection", {}).get("mode", "")),
        }
        atomic_pickle_dump({"row": row8}, task8_cache)
        print(f">>> [TASK-CACHE] saved: {task8_cache}", flush=True)
    results.append({
        **row8
    })
    return results


# ============================================================================
# 8) Main
# ============================================================================
def append_rows_and_refresh_summary(rows, raw_csv, summary_csv):
    csv_columns = [
        "experiment_id",
        "repeat_idx",
        "seed",
        "task",
        "task_phase",
        "task_order",
        "target_task",
        "full_use_obs",
        "partial_use_obs",
        "graph_alpha",
        "PCC",
        "SSIM",
        "CMD",
        "RMSE",
        "selected_sources",
        "kb_requested_mode",
        "kb_use_data_profile",
        "kb_select_mode",
        "kb_selection_mode_requested",
        "agent_provider",
        "agent_model",
        "agent_confidence",
        "agent_weight_strategy",
        "agent_policy_rescue",
        "agent_usage_modes",
        "agent_source_actions",
        "agent_failure_stage",
        "agent_failure_message",
        "kb_num_direct_links",
        "kb_num_module_links",
        "kb_kegg_pathway_count",
        "kb_string_protein_edges",
        "kb_orchestration_report",
        "agent_trace_path",
        "rule_baseline_source_path",
        "rule_PCC",
        "rule_SSIM",
        "rule_CMD",
        "rule_RMSE",
        "round1_PCC",
        "round1_SSIM",
        "round1_CMD",
        "round1_RMSE",
        "round2_refine_strategy",
        "round2_round1_outcome",
        "round2_metric_wins",
        "round2_weighted_score",
        "round2_hard_guard_triggered",
        "round2_metric_focus",
        "domain",
    ]
    rows_df = pd.DataFrame(rows)
    rows_df = rows_df.reindex(columns=csv_columns)
    write_header = not os.path.exists(raw_csv)
    rows_df.to_csv(raw_csv, mode="a", header=write_header, index=False)
    raw_df = pd.read_csv(raw_csv)
    summary_df = (
        raw_df
        .groupby(["task_order", "task", "task_phase"], as_index=False, dropna=False)
        .agg(
            repeats=("PCC", "size"),
            PCC_mean=("PCC", "mean"), PCC_std=("PCC", "std"),
            SSIM_mean=("SSIM", "mean"), SSIM_std=("SSIM", "std"),
            CMD_mean=("CMD", "mean"), CMD_std=("CMD", "std"),
            RMSE_mean=("RMSE", "mean"), RMSE_std=("RMSE", "std"),
        )
        .sort_values(["task_order", "task", "task_phase"])
    )
    summary_df.to_csv(summary_csv, index=False)


if __name__ == "__main__":
    print(">>> Starting AKG-Omics KB-agent run ...", flush=True)
    print(f">>> Seeds: {FINAL_SEEDS}", flush=True)
    print(f">>> Run proteogenomics={RUN_PROTEOGENOMICS_TASKS}, metabolomics={RUN_METABOLOMICS_TASKS}", flush=True)
    print(f">>> Proteogenomics KB agent={USE_PROTEO_KB_AGENT}", flush=True)
    print(
        f">>> Proteogenomics KB requested mode={PROTEO_KB_REQUESTED_MODE}, "
        f"selection mode={PROTEO_KB_SELECTION_MODE}, data_profile={PROTEO_KB_USE_DATA_PROFILE}",
        flush=True,
    )
    print(
        f">>> Metabolomics KB agent={USE_MET_KB_AGENT}, requested mode={MET_KB_REQUESTED_MODE}, "
        f"selection mode={MET_KB_SELECTION_MODE}, data_profile={MET_KB_USE_DATA_PROFILE}",
        flush=True,
    )
    if PROTEO_AGENT_TWO_ROUND and USE_PROTEO_KB_AGENT:
        print(
            f">>> Formal rule baseline import={PROTEO_IMPORT_FORMAL_RULE_BASELINE}, "
            f"csv={_resolve_formal_rule_baseline_csv() or 'not_found'}",
            flush=True,
        )
    if RUN_PROTEOGENOMICS_TASKS:
        print(
            f">>> Proteogenomics shared cache={PROTEO_USE_SHARED_CACHE} "
            f"root={PROTEO_CACHE_DIR}",
            flush=True,
        )

    proteo_pack = None
    if RUN_PROTEOGENOMICS_TASKS:
        print(">>> Preparing proteogenomics aligned data ...", flush=True)
        proteo_pack = prepare_proteogenomics_data()

    met_pack = None
    gene_lat_pack = None
    if RUN_METABOLOMICS_TASKS:
        print(">>> Preparing metabolomics aligned data ...", flush=True)
        met_pack = prepare_aligned_data()
        gene_lat_pack = build_or_load_gene_pca_latent(met_pack["B1_rna"], met_pack["C1_rna"])

    run_started = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(save_root, f"unified_8tasks_{run_started}")
    os.makedirs(out_dir, exist_ok=True)
    orchestration_dir = os.path.join(out_dir, "knowledge_orchestration")
    os.makedirs(orchestration_dir, exist_ok=True)
    raw_csv = os.path.join(out_dir, "unified_metrics_raw.csv")
    summary_csv = os.path.join(out_dir, "unified_metrics_summary.csv")
    settings_json = os.path.join(out_dir, "unified_settings.json")

    # Run task1~task8 sequentially per seed.
    for i, seed in enumerate(FINAL_SEEDS, start=1):
        if RUN_PROTEOGENOMICS_TASKS:
            print(f">>> Running proteogenomics repeat {i}/{len(FINAL_SEEDS)}, seed={seed}", flush=True)
            proteo_rows = run_proteogenomics_tasks_for_seed(
                repeat_idx=i,
                seed=seed,
                data_pack=proteo_pack,
                orchestration_dir=orchestration_dir,
            )
            append_rows_and_refresh_summary(proteo_rows, raw_csv=raw_csv, summary_csv=summary_csv)
            for r in proteo_rows:
                print(f"[{r['task']}] PCC={r['PCC']:.4f} SSIM={r['SSIM']:.4f} CMD={r['CMD']:.4f} RMSE={r.get('RMSE', float('nan')):.4f}", flush=True)

        if RUN_METABOLOMICS_TASKS:
            print(f">>> Running metabolomics repeat {i}/{len(FINAL_SEEDS)}, seed={seed}", flush=True)
            met_rows = run_metabolomics_tasks_for_seed(
                repeat_idx=i,
                seed=seed,
                data_pack=met_pack,
                gene_lat_pack=gene_lat_pack,
                orchestration_dir=orchestration_dir,
            )
            append_rows_and_refresh_summary(met_rows, raw_csv=raw_csv, summary_csv=summary_csv)
            for r in met_rows:
                print(f"[{r['task']}] PCC={r['PCC']:.4f} SSIM={r['SSIM']:.4f} CMD={r['CMD']:.4f} RMSE={r.get('RMSE', float('nan')):.4f}", flush=True)

    with open(settings_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "run_name": RUN_NAME,
                "timestamp": run_started,
                "seeds": FINAL_SEEDS,
                "run_proteogenomics_tasks": RUN_PROTEOGENOMICS_TASKS,
                "run_metabolomics_tasks": RUN_METABOLOMICS_TASKS,
                "use_proteo_kb_agent": USE_PROTEO_KB_AGENT,
                "require_proteo_kb_llm_agent": REQUIRE_PROTEO_KB_LLM_AGENT,
                "proteo_kb_requested_mode": PROTEO_KB_REQUESTED_MODE,
                "proteo_kb_selection_mode": PROTEO_KB_SELECTION_MODE,
                "proteo_kb_use_data_profile": PROTEO_KB_USE_DATA_PROFILE,
                "proteo_kb_agent_min_sources": PROTEO_KB_AGENT_MIN_SOURCES,
                "proteo_kb_agent_max_sources": PROTEO_KB_AGENT_MAX_SOURCES,
                "use_met_kb_agent": USE_MET_KB_AGENT,
                "require_met_kb_llm_agent": REQUIRE_MET_KB_LLM_AGENT,
                "met_kb_requested_mode": MET_KB_REQUESTED_MODE,
                "met_kb_selection_mode": MET_KB_SELECTION_MODE,
                "met_kb_use_data_profile": MET_KB_USE_DATA_PROFILE,
                "met_kb_agent_min_sources": MET_KB_AGENT_MIN_SOURCES,
                "met_kb_agent_max_sources": MET_KB_AGENT_MAX_SOURCES,
                "met_kb_allowed_sources": MET_KB_ALLOWED_SOURCES,
                "met_kb_selection_policy": MET_KB_SELECTION_POLICY,
                "proteo_task_filter": sorted(PROTEO_TASK_FILTER),
                "proteo_data_root": PROTEO_DATA_ROOT,
                "proteo_sample1": PROTEO_SAMPLE1,
                "proteo_sample2": PROTEO_SAMPLE2,
                "proteo_kb_paths": {
                    "hgnc": PROTEO_HGNC_PATH,
                    "uniprot": PROTEO_UNIPROT_PATH,
                    "reactome_uniprot": PROTEO_REACTOME_UNIPROT_PATH,
                    "reactome_ensembl": PROTEO_REACTOME_ENSEMBL_PATH,
                    "cellmarker": PROTEO_CELLMARKER_PATH,
                    "dorothea": PROTEO_DOROTHEA_PATH,
                    "omnipath": PROTEO_OMNIPATH_PATH,
                    "corum": PROTEO_CORUM_PATH,
                    "proteinatlas": PROTEO_PROTEINATLAS_PATH,
                    "kegg_pathways": PROTEO_KEGG_PATHWAYS_PATH,
                    "kegg_gene_pathway": PROTEO_KEGG_GENE_PATHWAY_PATH,
                    "string": PROTEO_STRING_PATH,
                },
                "proteo_task_specs": PROTEO_TASK_SPECS,
                "proteo_use_shared_cache": PROTEO_USE_SHARED_CACHE,
                "proteo_shared_cache_root": PROTEO_SHARED_CACHE_ROOT,
                "proteo_cache_dir": PROTEO_CACHE_DIR,
                "proteo_refined_dir": PROTEO_REFINED_DIR,
                "proteo_kb_cache_dir": PROTEO_KB_CACHE_DIR,
                "proteo_graph_cache_dir": PROTEO_GRAPH_CACHE_DIR,
                "proteo_task_cache_dir": PROTEO_TASK_CACHE_DIR,
                "proteo_gene_topk": PROTEO_GENE_TOPK,
                "proteo_gene_select_mode": PROTEO_GENE_SELECT_MODE,
                "proteo_gene_random_seed": PROTEO_GENE_RANDOM_SEED,
                "proteo_gene_beneficial_list": PROTEO_GENE_BENEFICIAL_LIST,
                "proteo_gene_beneficial_tag": PROTEO_GENE_BENEFICIAL_TAG,
                "proteo_mode_tag": PROTEO_MODE_TAG,
                "metabolomics_data_root": ndata_root,
                "metabolomics_kb_root": KB_ROOT,
                "kb_usage_mode": KB_USAGE_MODE,
                "use_kb_quality_scaling": USE_KB_QUALITY_SCALING,
                "force_task_retrain": FORCE_TASK_RETRAIN,
                "met_task_cache_version": MET_TASK_CACHE_VERSION,
                "proteo_task_cache_version": PROTEO_TASK_CACHE_VERSION,
                "enable_detailed_analysis": ENABLE_DETAILED_ANALYSIS,
                "proteo_import_formal_rule_baseline": PROTEO_IMPORT_FORMAL_RULE_BASELINE,
                "proteo_formal_rule_baseline_csv": _resolve_formal_rule_baseline_csv(),
                "proteo_rule_baseline_sources": PROTEO_RULE_BASELINE_SOURCES,
                "proteo_agent_metric_aware_refinement": PROTEO_AGENT_METRIC_AWARE_REFINEMENT,
                "knowledge_orchestration_dir": orchestration_dir,
                "met_topk": MET_TOPK,
                "gene_topk": GENE_TOPK,
                "train_epochs": TRAIN_EPOCHS,
                "train_lr": TRAIN_LR,
                "fusion_alpha": FUSION_ALPHA,
                "e15_aligned_config": {
                    "lambda_target": LAMBDA_TARGET,
                    "lambda_pcc": LAMBDA_PCC,
                    "lambda_latent": LAMBDA_LATENT,
                    "lambda_recon": LAMBDA_RECON,
                    "lambda_cycle": 0.2,
                    "lambda_partial_cycle": 0.2,
                    "lambda_kb": LAMBDA_KB,
                    "lambda_graph": LAMBDA_GRAPH,
                    "lambda_ortho": LAMBDA_ORTHO,
                    "lambda_dgi": LAMBDA_DGI,
                    "lambda_full": LAMBDA_FULL,
                    "lambda_partial": LAMBDA_PARTIAL,
                    "lambda_align": LAMBDA_ALIGN,
                    "lambda_bridge_align": 0.5,
                    "target_soft_alpha": TARGET_SOFT_ALPHA,
                    "kb_ct_prior_mix": KB_CT_PRIOR_MIX,
                    "disable_dynamic_retriever": DISABLE_DYNAMIC_RETRIEVER,
                    "disable_kb_fusion": DISABLE_KB_FUSION,
                    "disable_soft_gate": DISABLE_SOFT_GATE,
                    "disable_residual_refine": DISABLE_RESIDUAL_REFINE,
                    "proteo_final_kb_cfg": PROTEO_FINAL_KB_CFG,
                },
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("\n>>> AKG-Omics KB-agent run finished.", flush=True)
    print(f">>> Raw metrics: {raw_csv}", flush=True)
    print(f">>> Summary metrics: {summary_csv}", flush=True)
    print(f">>> Settings: {settings_json}", flush=True)
