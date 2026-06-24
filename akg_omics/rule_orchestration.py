# -*- coding: utf-8 -*-
"""
Rule-based KB source selection.

This file disables the agent-based orchestration and selects locally available
knowledge sources by fixed rules. It keeps the same interface expected by run.py:

    selected_sources, report = orchestrate_sources(...)

The goal is to first run ablation, Top-20 feature evaluation, loss curves,
and prediction visualizations without depending on knowledge_agent.
"""

import os
import json
from datetime import datetime


def orchestrate_sources(
    task_id,
    source_modalities,
    target_modality,
    local_source_status=None,
    max_sources=4,
    ensure_core_sources=None,
    report_path=None,
):
    """
    Rule-based source selection without agent.

    Parameters
    ----------
    task_id : str
        Current task name, e.g. task1_he_to_gene.
    source_modalities : list[str]
        Input modalities, e.g. ["he"], ["he", "protein"].
    target_modality : str
        Target modality, e.g. "gene", "protein", "metabolite".
    local_source_status : dict
        Local KB availability, e.g.
        {
            "hgnc": True,
            "uniprot": True,
            "reactome": True,
            "cellmarker": True,
            "hmdb": False,
            "chebi": False,
            "kegg": False,
        }
    max_sources : int
        Maximum number of selected sources.
    ensure_core_sources : list[str]
        Sources that should be selected first if locally available.
    report_path : str or None
        Optional path to save the source selection report.

    Returns
    -------
    selected_sources : list[str]
    report : dict
    """
    local_source_status = dict(local_source_status or {})
    ensure_core_sources = list(ensure_core_sources or [])

    task_id = str(task_id)
    target_modality = str(target_modality).strip().lower()
    source_modalities = [str(x).strip().lower() for x in (source_modalities or [])]

    selected_sources = []

    def _add_source(src):
        src = str(src).strip().lower()
        if not src:
            return
        if bool(local_source_status.get(src, False)) and src not in selected_sources:
            selected_sources.append(src)

    # ----------------------------------------------------------------------
    # Rule 1: required/core sources first
    # ----------------------------------------------------------------------
    for src in ensure_core_sources:
        _add_source(src)

    # ----------------------------------------------------------------------
    # Rule 2: proteogenomics tasks
    # Gene/protein tasks prefer HGNC + UniProt + Reactome + CellMarker.
    # ----------------------------------------------------------------------
    is_gene_or_protein_task = (
        target_modality in {"gene", "protein", "transcriptome", "proteome"}
        or any(x in {"gene", "protein", "transcriptome", "proteome"} for x in source_modalities)
    )

    if is_gene_or_protein_task:
        # HGNC: gene symbol, alias, Ensembl mapping
        _add_source("hgnc")

        # UniProt: protein accession and protein-gene mapping
        _add_source("uniprot")

        # Reactome: pathway-level cross-modal and intra-modal priors
        _add_source("reactome")

        # CellMarker: cell-type marker prior
        _add_source("cellmarker")

    # ----------------------------------------------------------------------
    # Rule 3: metabolomics tasks
    # Metabolite-related tasks prefer HMDB + ChEBI + Reactome.
    # ----------------------------------------------------------------------
    is_met_task = (
        target_modality in {"metabolite", "metabolomics", "met"}
        or any(x in {"metabolite", "metabolomics", "met"} for x in source_modalities)
    )

    if is_met_task:
        _add_source("hmdb")
        _add_source("chebi")
        _add_source("reactome")

    # ----------------------------------------------------------------------
    # Rule 4: fallback by fixed priority
    # ----------------------------------------------------------------------
    if not selected_sources:
        fallback_order = [
            "hgnc",
            "uniprot",
            "reactome",
            "cellmarker",
            "hmdb",
            "chebi",
            "kegg",
        ]
        for src in fallback_order:
            _add_source(src)
            if len(selected_sources) >= int(max_sources):
                break

    selected_sources = selected_sources[: int(max_sources)]

    report = {
        "timestamp_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "task_id": task_id,
        "source_modalities": source_modalities,
        "target_modality": target_modality,
        "local_source_status": local_source_status,
        "final_selected_sources": selected_sources,
        "selection": {
            "mode": "rule_based_no_agent",
            "reason": (
                "Agent disabled. Sources were selected by fixed local rules "
                "according to task modality and local file availability."
            ),
        },
        "fallback_used": False,
    }

    if report_path is not None:
        parent = os.path.dirname(report_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        report["report_path"] = report_path

    return selected_sources, report