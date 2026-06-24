#!/usr/bin/env python3
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


PROTEINS = [
    "CD11c", "CD163", "CD20", "CD34", "CD3e", "CD4", "CD56", "CD68",
    "CD8", "FOXP3", "HLA-A", "HLA-DR", "IDO1", "MPO", "Pan-Cytokeratin", "SMA",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--full-npz", required=True)
    parser.add_argument("--gene", default="FSTL1")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    root = Path(args.project_root)
    sys.path.insert(0, str(root))
    from akg_omics.run import (
        load_cellmarker_prior,
        load_hgnc_maps,
        load_kegg_gene_sets,
        load_proteinatlas_celltype_prior,
        load_reactome_gene_sets,
        load_uniprot_maps,
        normalize_symbol,
    )

    kb = Path("data/KB")
    paths = {
        "hgnc": kb / "hgnc_complete_set.txt",
        "uniprot": kb / "uniprot_human_reviewed.tsv",
        "reactome_uniprot": kb / "UniProt2Reactome.txt",
        "reactome_ensembl": kb / "Ensembl2Reactome.txt",
        "cellmarker": kb / "Cell_marker_Human.xlsx",
        "proteinatlas": kb / "proteinatlas.tsv",
        "kegg_pathways": kb / "kegg_hsa_pathways.txt",
        "kegg_links": kb / "kegg_hsa_gene_pathway_links.txt",
    }
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    with np.load(args.full_npz, allow_pickle=False) as data:
        feature_names = data["feature_names"].astype(str)
    feature_set = set(feature_names)

    alias_to_symbol, ensembl_to_symbol = load_hgnc_maps(str(paths["hgnc"]))
    acc_to_symbols, alias_to_symbol, protein_name_to_accs = load_uniprot_maps(
        str(paths["uniprot"]), alias_to_symbol
    )
    reactome_to_genes, gene_to_reactome = load_reactome_gene_sets(
        str(paths["reactome_uniprot"]),
        str(paths["reactome_ensembl"]),
        acc_to_symbols,
        ensembl_to_symbol,
    )
    kegg_to_genes, gene_to_kegg = load_kegg_gene_sets(
        str(paths["kegg_pathways"]), str(paths["kegg_links"]), str(paths["hgnc"])
    )
    cellmarker = load_cellmarker_prior(str(paths["cellmarker"]), alias_to_symbol)
    proteinatlas = load_proteinatlas_celltype_prior(
        str(paths["proteinatlas"]), alias_to_symbol
    )

    gene = alias_to_symbol.get(normalize_symbol(args.gene), normalize_symbol(args.gene))
    protein_canon = []
    protein_gene_sets = []
    for protein in PROTEINS:
        pn = normalize_symbol(protein)
        pc = alias_to_symbol.get(pn, pn)
        genes = set()
        for accession in protein_name_to_accs.get(pc, set()):
            genes.update(acc_to_symbols.get(accession, set()))
        genes.update(acc_to_symbols.get(pn, set()))
        genes.update(acc_to_symbols.get(pc, set()))
        if pc in feature_set:
            genes.add(pc)
        protein_canon.append(pc)
        protein_gene_sets.append(genes)

    gene_reactome = gene_to_reactome.get(gene, set())
    gene_kegg = gene_to_kegg.get(gene, set())
    gene_cellmarker_types = {ct for ct, markers in cellmarker.items() if gene in markers}
    gene_hpa_types = {ct for ct, markers in proteinatlas.items() if gene in markers}

    relation_rows = []
    for protein, pc, mapped_genes in zip(PROTEINS, protein_canon, protein_gene_sets):
        protein_reactome = set().union(
            *(gene_to_reactome.get(mapped, set()) for mapped in mapped_genes)
        ) if mapped_genes else set()
        protein_kegg = set().union(
            *(gene_to_kegg.get(mapped, set()) for mapped in mapped_genes)
        ) if mapped_genes else set()
        protein_cm_types = {ct for ct, markers in cellmarker.items() if pc in markers}
        protein_hpa_types = {ct for ct, markers in proteinatlas.items() if pc in markers}
        shared_reactome = sorted(gene_reactome & protein_reactome)
        shared_kegg = sorted(gene_kegg & protein_kegg)
        shared_cm = sorted(gene_cellmarker_types & protein_cm_types)
        shared_hpa = sorted(gene_hpa_types & protein_hpa_types)
        relation_rows.append(
            {
                "protein_feature": protein,
                "canonical_protein_gene": pc,
                "mapped_genes": "|".join(sorted(mapped_genes)),
                "direct_relation": int(gene in mapped_genes),
                "reactome_shared_count": len(shared_reactome),
                "reactome_shared_pathways": "|".join(shared_reactome),
                "cellmarker_shared_count": len(shared_cm),
                "cellmarker_shared_celltypes": "|".join(shared_cm),
                "kegg_shared_count": len(shared_kegg),
                "kegg_shared_pathways": "|".join(shared_kegg),
                "proteinatlas_shared_count": len(shared_hpa),
                "proteinatlas_shared_groups": "|".join(shared_hpa),
                "core_relation_present": int(
                    gene in mapped_genes or bool(shared_reactome) or bool(shared_cm)
                ),
                "agent_augmented_relation_present": int(
                    gene in mapped_genes
                    or bool(shared_reactome)
                    or bool(shared_cm)
                    or bool(shared_kegg)
                    or bool(shared_hpa)
                ),
            }
        )
    relations = pd.DataFrame(relation_rows)
    relations.to_csv(output / f"{gene}_protein_relation_provenance.csv", index=False)

    combined_celltypes = defaultdict(set)
    for ct, markers in cellmarker.items():
        combined_celltypes[ct].update(markers)
    for ct, markers in proteinatlas.items():
        combined_celltypes[ct].update(markers)
    gene_all_celltypes = {ct for ct, markers in combined_celltypes.items() if gene in markers}
    gene_all_pathways = set(gene_reactome) | set(gene_kegg)

    neighbor_rows = []
    for other in feature_names:
        other = alias_to_symbol.get(normalize_symbol(other), normalize_symbol(other))
        if other == gene:
            continue
        other_reactome = gene_to_reactome.get(other, set())
        other_kegg = gene_to_kegg.get(other, set())
        other_celltypes = {ct for ct, markers in combined_celltypes.items() if other in markers}
        shared_reactome = gene_reactome & other_reactome
        shared_kegg = gene_kegg & other_kegg
        shared_celltypes = gene_all_celltypes & other_celltypes
        if not shared_reactome and not shared_kegg and not shared_celltypes:
            continue
        pathway_score = sum(
            1.0 / np.sqrt(max(len(reactome_to_genes[pw] & feature_set), 1))
            for pw in shared_reactome
        )
        pathway_score += sum(
            0.5 / np.sqrt(max(len(kegg_to_genes[pw] & feature_set), 1))
            for pw in shared_kegg
        )
        neighbor_rows.append(
            {
                "neighbor_gene": other,
                "reactome_shared_count": len(shared_reactome),
                "reactome_shared_pathways": "|".join(sorted(shared_reactome)),
                "kegg_shared_count": len(shared_kegg),
                "kegg_shared_pathways": "|".join(sorted(shared_kegg)),
                "shared_celltype_count": len(shared_celltypes),
                "shared_celltypes": "|".join(sorted(shared_celltypes)),
                "pathway_specificity_score": pathway_score,
                "relation_type_count": sum(
                    [bool(shared_reactome), bool(shared_kegg), bool(shared_celltypes)]
                ),
            }
        )
    neighbors = pd.DataFrame(neighbor_rows)
    if not neighbors.empty:
        neighbors = neighbors.sort_values(
            ["relation_type_count", "pathway_specificity_score", "shared_celltype_count"],
            ascending=False,
        )
    neighbors.to_csv(output / f"{gene}_gene_neighbors.csv", index=False)

    identity = pd.read_csv(paths["hgnc"], sep="\t", low_memory=False)
    identity = identity[identity["symbol"].astype(str).str.upper() == gene]
    identity_fields = [
        "hgnc_id", "symbol", "name", "locus_type", "location",
        "entrez_id", "ensembl_gene_id", "uniprot_ids",
    ]
    identity[identity_fields].to_csv(output / f"{gene}_identity.csv", index=False)

    summary = {
        "gene": gene,
        "core_sources_used_in_causal_run": ["hgnc", "uniprot", "reactome", "cellmarker"],
        "agent_seed1_augmented_sources": [
            "hgnc", "uniprot", "reactome", "cellmarker", "proteinatlas", "kegg"
        ],
        "reactome_pathways": sorted(gene_reactome),
        "kegg_pathways": sorted(gene_kegg),
        "cellmarker_celltypes": sorted(gene_cellmarker_types),
        "proteinatlas_groups": sorted(gene_hpa_types),
        "proteins_linked_by_core_sources": relations.loc[
            relations["core_relation_present"] == 1, "protein_feature"
        ].tolist(),
        "proteins_linked_by_agent_augmented_sources": relations.loc[
            relations["agent_augmented_relation_present"] == 1, "protein_feature"
        ].tolist(),
        "num_gene_neighbors": int(len(neighbors)),
        "top_gene_neighbors": (
            neighbors.head(30).to_dict(orient="records") if not neighbors.empty else []
        ),
    }
    (output / f"{gene}_knowledge_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
