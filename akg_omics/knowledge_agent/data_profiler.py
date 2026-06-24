"""
Data profiling helpers for data-aware knowledge source selection.

The profiler intentionally returns a compact JSON-serialisable dictionary.  It
is used before KB construction, so it favours cheap coverage estimates over
building full relation matrices.
"""

from __future__ import annotations

import os
import re
from dataclasses import asdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

try:
    import numpy as np
except Exception:  # pragma: no cover - numpy is expected in this project
    np = None

try:
    import pandas as pd
except Exception:  # pragma: no cover - profiler should still degrade cleanly
    pd = None

from .task_schema import TaskSpec


def _normalise_token(x: Any) -> Optional[str]:
    if x is None:
        return None
    s = str(x).strip()
    if not s or s.lower() == "nan":
        return None
    return s.upper()


def _normalise_alias(x: Any) -> Optional[str]:
    s = _normalise_token(x)
    if s is None:
        return None
    s = re.sub(r"[^A-Z0-9]+", "", s)
    return s or None


def _split_symbol_field(x: Any) -> List[str]:
    s = _normalise_token(x)
    if s is None:
        return []
    out: List[str] = []
    for part in re.split(r"[|,;/\s]+", s):
        tok = _normalise_alias(part)
        if tok:
            out.append(tok)
    return out


def _as_feature_list(values: Optional[Iterable[Any]], max_items: Optional[int] = None) -> List[str]:
    if values is None:
        return []
    try:
        raw = list(values)
    except TypeError:
        raw = [values]
    out: List[str] = []
    seen = set()
    for v in raw:
        s = str(v).strip()
        if not s or s.lower() == "nan":
            continue
        if s in seen:
            continue
        seen.add(s)
        out.append(s)
        if max_items is not None and len(out) >= int(max_items):
            break
    return out


def _feature_set(values: Sequence[str]) -> Set[str]:
    out: Set[str] = set()
    for v in values:
        tok = _normalise_alias(v)
        if tok:
            out.add(tok)
    return out


def _infer_id_type(features: Sequence[str], modality: str) -> str:
    vals = [str(x).strip() for x in features[:200] if str(x).strip()]
    if not vals:
        return "empty"
    upper = [x.upper() for x in vals]
    if modality == "gene":
        if sum(x.startswith("ENS") for x in upper) / max(len(upper), 1) > 0.5:
            return "ensembl_like"
        if sum(bool(re.match(r"^[A-Z0-9][A-Z0-9_.-]{1,20}$", x)) for x in upper) / max(len(upper), 1) > 0.5:
            return "symbol_like"
    if modality == "protein":
        if sum(bool(re.match(r"^[OPQ][0-9][A-Z0-9]{3}[0-9]$", x)) for x in upper) / max(len(upper), 1) > 0.2:
            return "uniprot_accession_like"
        if sum(x.startswith("CD") or "-" in x for x in upper) / max(len(upper), 1) > 0.2:
            return "marker_or_antibody_like"
        return "symbol_or_panel_name_like"
    if modality in {"metabolism", "metabolite"}:
        if sum(bool(re.search(r"(^|[^0-9])\d{2,5}\.\d+", x)) for x in vals) / max(len(vals), 1) > 0.2:
            return "mz_feature_like"
        if sum(x.startswith("HMDB") or x.startswith("CHEBI") for x in upper) / max(len(upper), 1) > 0.2:
            return "database_id_like"
        return "name_like"
    return "unknown"


def _path_exists(path: Optional[str]) -> bool:
    return bool(path) and os.path.exists(str(path))


def _max_rows() -> int:
    return max(1000, int(os.getenv("KNOWLEDGE_AGENT_PROFILE_MAX_ROWS", "200000")))


def _read_table(path: Optional[str], sep: Optional[str] = None):
    if pd is None or not _path_exists(path):
        return None
    path = str(path)
    try:
        if path.lower().endswith((".xlsx", ".xls")):
            return pd.read_excel(path, nrows=_max_rows())
        if sep is None:
            sep = "\t" if path.lower().endswith((".tsv", ".txt")) else ","
        return pd.read_csv(path, sep=sep, low_memory=False, nrows=_max_rows(), encoding="latin1", on_bad_lines="skip")
    except Exception:
        return None


def _col_by_candidates(df, candidates: Sequence[str], contains: Optional[Sequence[str]] = None) -> Optional[str]:
    if df is None:
        return None
    norm = {str(c).lower().replace("_", "").replace(" ", ""): c for c in df.columns}
    for cand in candidates:
        key = cand.lower().replace("_", "").replace(" ", "")
        if key in norm:
            return norm[key]
    if contains:
        for c in df.columns:
            low = str(c).lower()
            if all(tok in low for tok in contains):
                return c
    return None


def _coverage(feature_values: Sequence[str], known_ids: Set[str]) -> Dict[str, Any]:
    feats = _feature_set(feature_values)
    if not feats:
        return {"coverage": 0.0, "covered_count": 0, "total_count": 0, "examples": []}
    covered = sorted(feats & known_ids)
    return {
        "coverage": float(len(covered) / max(len(feats), 1)),
        "covered_count": int(len(covered)),
        "total_count": int(len(feats)),
        "examples": covered[:10],
    }


def _merge_ids(*sets_: Set[str]) -> Set[str]:
    out: Set[str] = set()
    for s in sets_:
        out.update(s or set())
    return out


def _hgnc_ids(path: Optional[str]) -> Set[str]:
    df = _read_table(path, sep="\t")
    if df is None:
        return set()
    ids: Set[str] = set()
    symbol_col = _col_by_candidates(df, ["approved_symbol", "symbol"])
    ensembl_col = _col_by_candidates(df, ["ensembl_gene_id"])
    alias_cols = [c for c in ["alias_symbol", "prev_symbol"] if c in df.columns]
    for _, row in df.iterrows():
        for col in [symbol_col, ensembl_col]:
            if col:
                tok = _normalise_alias(row.get(col))
                if tok:
                    ids.add(tok)
        for col in alias_cols:
            ids.update(_split_symbol_field(row.get(col)))
    return ids


def _uniprot_ids(path: Optional[str]) -> Set[str]:
    df = _read_table(path, sep="\t")
    if df is None:
        return set()
    ids: Set[str] = set()
    candidate_cols = [
        _col_by_candidates(df, ["Entry", "Accession", "From"]),
        _col_by_candidates(df, ["Gene Names (primary)", "Gene Names  (primary )"]),
        _col_by_candidates(df, ["Gene Names", "Gene names"]),
        _col_by_candidates(df, ["Protein names", "Protein names "]),
    ]
    for _, row in df.iterrows():
        for col in candidate_cols:
            if not col:
                continue
            ids.update(_split_symbol_field(row.get(col)))
    return ids


def _reactome_ids(*paths: Optional[str]) -> Set[str]:
    ids: Set[str] = set()
    for path in paths:
        df = _read_table(path, sep="\t")
        if df is None or len(df.columns) == 0:
            continue
        first = df.columns[0]
        for v in df[first].values:
            tok = _normalise_alias(v)
            if tok:
                ids.add(tok)
    return ids


def _cellmarker_ids(path: Optional[str]) -> Set[str]:
    df = _read_table(path)
    if df is None:
        return set()
    marker_col = _col_by_candidates(df, ["symbol", "marker", "gene", "gene_symbol"], contains=["marker"])
    if marker_col is None:
        return set()
    ids: Set[str] = set()
    for v in df[marker_col].values:
        ids.update(_split_symbol_field(v))
    return ids


def _proteinatlas_ids(path: Optional[str]) -> Set[str]:
    df = _read_table(path, sep="\t")
    if df is None:
        return set()
    gene_col = _col_by_candidates(df, ["Gene name", "Gene", "Gene symbol"], contains=["gene"])
    if gene_col is None:
        return set()
    ids: Set[str] = set()
    for v in df[gene_col].values:
        tok = _normalise_alias(v)
        if tok:
            ids.add(tok)
    return ids


def _edge_table_ids(path: Optional[str], source_candidates: Sequence[str], target_candidates: Sequence[str]) -> Set[str]:
    df = _read_table(path, sep="\t")
    if df is None:
        return set()
    src_col = _col_by_candidates(df, source_candidates, contains=["source"])
    tgt_col = _col_by_candidates(df, target_candidates, contains=["target"])
    ids: Set[str] = set()
    for col in [src_col, tgt_col]:
        if not col:
            continue
        for v in df[col].values:
            tok = _normalise_alias(v)
            if tok:
                ids.add(tok)
    return ids


def _corum_ids(path: Optional[str]) -> Set[str]:
    df = _read_table(path, sep="\t")
    if df is None:
        return set()
    ids: Set[str] = set()
    subunit_cols = [c for c in df.columns if "subunit" in str(c).lower()]
    for col in subunit_cols:
        for v in df[col].values:
            ids.update(_split_symbol_field(v))
    return ids


def _paths_dict(kb_paths: Any) -> Dict[str, Optional[str]]:
    if kb_paths is None:
        return {}
    if isinstance(kb_paths, dict):
        return dict(kb_paths)
    try:
        return asdict(kb_paths)
    except Exception:
        return {k: getattr(kb_paths, k) for k in dir(kb_paths) if k.endswith("_file")}


def _source_identifier_sets(kb_paths: Any) -> Dict[str, Set[str]]:
    paths = _paths_dict(kb_paths)
    hgnc = _hgnc_ids(paths.get("hgnc_file"))
    uniprot = _uniprot_ids(paths.get("uniprot_file"))
    reactome = _reactome_ids(paths.get("reactome_uniprot_file"), paths.get("reactome_ensembl_file"))
    cellmarker = _cellmarker_ids(paths.get("cellmarker_file"))
    proteinatlas = _proteinatlas_ids(paths.get("proteinatlas_file"))
    dorothea = _edge_table_ids(
        paths.get("dorothea_file"),
        ["source", "tf", "transcription_factor", "tf_symbol"],
        ["target", "target_gene", "target_symbol"],
    )
    omnipath = _edge_table_ids(
        paths.get("omnipath_file"),
        ["source", "genesymbol_intercell_source", "source_genesymbol"],
        ["target", "genesymbol_intercell_target", "target_genesymbol"],
    )
    corum = _corum_ids(paths.get("corum_file"))
    return {
        "hgnc": hgnc,
        "uniprot": uniprot,
        "reactome": reactome,
        "cellmarker": cellmarker,
        "proteinatlas": proteinatlas,
        "dorothea": dorothea,
        "omnipath": omnipath,
        "corum": corum,
        # Derived/lumped sources benefit from canonical gene/protein IDs too.
        "string": _merge_ids(uniprot, omnipath),
        "disgenet": hgnc,
        "opentargets": hgnc,
    }


def _source_modalities(source_id: str) -> Tuple[str, ...]:
    sid = str(source_id).lower()
    if sid in {"hgnc", "dorothea", "disgenet", "opentargets"}:
        return ("gene",)
    if sid in {"uniprot", "reactome", "cellmarker", "proteinatlas", "omnipath", "string"}:
        return ("gene", "protein")
    if sid == "corum":
        return ("protein",)
    if sid in {"hmdb", "kegg", "chebi", "metalights", "metabolights"}:
        return ("metabolism", "gene")
    return ("gene", "protein", "metabolism")


def build_data_profile(
    task_spec: TaskSpec,
    gene_names: Optional[Iterable[Any]] = None,
    protein_names: Optional[Iterable[Any]] = None,
    metabolite_names: Optional[Iterable[Any]] = None,
    kb_paths: Any = None,
    tissue: Optional[str] = None,
    max_examples: int = 20,
) -> Dict[str, Any]:
    """Build a compact task/data profile for KB source selection."""
    task = task_spec.normalized()
    genes = _as_feature_list(gene_names)
    proteins = _as_feature_list(protein_names)
    metabolites = _as_feature_list(metabolite_names)

    feature_sets = {
        "gene": {
            "count": len(genes),
            "id_type": _infer_id_type(genes, "gene"),
            "examples": genes[:max_examples],
        },
        "protein": {
            "count": len(proteins),
            "id_type": _infer_id_type(proteins, "protein"),
            "examples": proteins[:max_examples],
        },
        "metabolism": {
            "count": len(metabolites),
            "id_type": _infer_id_type(metabolites, "metabolism"),
            "examples": metabolites[:max_examples],
        },
    }

    ids_by_source = _source_identifier_sets(kb_paths)
    source_coverage: Dict[str, Dict[str, Any]] = {}
    modality_values = {"gene": genes, "protein": proteins, "metabolism": metabolites}
    for source_id, known_ids in ids_by_source.items():
        per_modality: Dict[str, Dict[str, Any]] = {}
        relevant = [m for m in _source_modalities(source_id) if modality_values.get(m)]
        for modality in relevant:
            per_modality[modality] = _coverage(modality_values[modality], known_ids)
        if per_modality:
            totals = [v["total_count"] for v in per_modality.values()]
            covered = [v["covered_count"] for v in per_modality.values()]
            total_count = int(sum(totals))
            covered_count = int(sum(covered))
            overall = float(covered_count / max(total_count, 1))
        else:
            total_count = 0
            covered_count = 0
            overall = 0.0
        source_coverage[source_id] = {
            "overall_coverage": overall,
            "covered_count": covered_count,
            "total_count": total_count,
            "per_modality": per_modality,
            "local_identifier_count": int(len(known_ids)),
            "coverage_available": bool(known_ids and total_count > 0),
        }

    warnings: List[str] = []
    if proteins and feature_sets["protein"]["id_type"] == "marker_or_antibody_like":
        warnings.append("protein_features_look_like_markers_or_antibody_names_alias_resolution_recommended")
    if metabolites and feature_sets["metabolism"]["id_type"] == "mz_feature_like":
        warnings.append("metabolite_features_look_like_mz_values_mass_annotation_required_before_kb_mapping")
    if kb_paths is None:
        warnings.append("kb_paths_not_provided_coverage_preview_limited")

    return {
        "schema_version": "data_profile_v1",
        "task": task.to_dict(),
        "species": task.species,
        "tissue": tissue if tissue is not None else task.tissue,
        "modalities": sorted(set(task.source_modalities + [task.target_modality])),
        "feature_sets": feature_sets,
        "source_coverage": source_coverage,
        "warnings": warnings,
    }

