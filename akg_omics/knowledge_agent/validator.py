from typing import Dict, List

from .task_schema import TaskSpec


def validate_kb_stats(stats: Dict, task_spec: TaskSpec) -> Dict:
    task = task_spec.normalized()
    errors: List[str] = []
    warnings: List[str] = []

    stats = stats or {}
    direct = int(stats.get("num_direct_links", 0))
    module = int(stats.get("num_module_links", 0))
    celltype = int(stats.get("num_celltype_links", 0))
    proteinatlas = int(stats.get("num_proteinatlas_links", 0))
    total_links = direct + module + celltype + proteinatlas
    resolved_proteins = int(stats.get("resolved_proteins", 0))

    requires_protein = "protein" in set(task.source_modalities + [task.target_modality])
    requires_gene = "gene" in set(task.source_modalities + [task.target_modality])

    if requires_gene and requires_protein and total_links <= 0:
        errors.append("no_cross_modal_links_built")
    if requires_protein and resolved_proteins <= 0:
        warnings.append("no_resolved_proteins")

    gene_edges = int(stats.get("gene_graph_edges", 0))
    prot_edges = int(stats.get("protein_graph_edges", 0))
    if requires_gene and gene_edges <= 0:
        warnings.append("empty_gene_graph")
    if requires_protein and prot_edges <= 0:
        warnings.append("empty_protein_graph")

    return {
        "ok": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "summary": {
            "num_direct_links": direct,
            "num_module_links": module,
            "num_celltype_links": celltype,
            "num_proteinatlas_links": proteinatlas,
            "resolved_proteins": resolved_proteins,
            "gene_graph_edges": gene_edges,
            "protein_graph_edges": prot_edges,
        },
    }
