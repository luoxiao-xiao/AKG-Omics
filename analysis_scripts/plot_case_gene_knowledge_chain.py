#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from scipy.stats import pearsonr, spearmanr


PASTEL = LinearSegmentedColormap.from_list(
    "pastel_blue_peach", ["#8da0cb", "#d7e7f4", "#f7f7f7", "#f6c3a5", "#dd7f69"]
)


def pcc(a, b):
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return np.nan
    return float(pearsonr(a, b).statistic)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", required=True)
    parser.add_argument("--no-kb", required=True)
    parser.add_argument("--no-dynamic", required=True)
    parser.add_argument("--relations", required=True)
    parser.add_argument("--neighbors", required=True)
    parser.add_argument("--gene", default="FSTL1")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    full = np.load(args.full, allow_pickle=False)
    nokb = np.load(args.no_kb, allow_pickle=False)
    nodyn = np.load(args.no_dynamic, allow_pickle=False)
    names = full["feature_names"].astype(str)
    index = int(np.where(names == args.gene)[0][0])
    coords = full["coords"].astype(np.float32)
    measured = full["gt"][:, index].astype(np.float32)
    prediction = full["pred_full"][:, index].astype(np.float32)
    prediction_nokb = nokb["pred_full"][:, index].astype(np.float32)
    prediction_nodyn = nodyn["pred_full"][:, index].astype(np.float32)
    global_prior = full["kb_global_prior"][:, index].astype(np.float32)
    local_prior = full["kb_local_prior"][:, index].astype(np.float32)
    mask = full["kb_local_mask"][:, index].astype(np.float32)
    refine_gate = full["refine_gate"][:, index].astype(np.float32)
    kb_gate = full["fusion_gate_kb"].mean(axis=1).astype(np.float32)

    gain_nokb = np.abs(measured - prediction_nokb) - np.abs(measured - prediction)
    gain_nodyn = np.abs(measured - prediction_nodyn) - np.abs(measured - prediction)
    metrics = {
        "gene": args.gene,
        "pcc_full": pcc(measured, prediction),
        "pcc_no_kb": pcc(measured, prediction_nokb),
        "pcc_no_dynamic": pcc(measured, prediction_nodyn),
        "pcc_global_prior": pcc(measured, global_prior),
        "pcc_local_prior": pcc(measured, local_prior),
        "mean_mask": float(mask.mean()),
        "mask_cv": float(mask.std() / max(abs(mask.mean()), 1e-8)),
        "mean_kb_gate": float(kb_gate.mean()),
        "mean_refine_gate": float(refine_gate.mean()),
        "mean_absolute_error_full": float(np.mean(np.abs(measured - prediction))),
        "mean_absolute_error_no_kb": float(np.mean(np.abs(measured - prediction_nokb))),
        "mean_absolute_error_no_dynamic": float(np.mean(np.abs(measured - prediction_nodyn))),
        "fraction_spots_full_better_than_no_kb": float(np.mean(gain_nokb > 0)),
        "fraction_spots_full_better_than_no_dynamic": float(np.mean(gain_nodyn > 0)),
        "spearman_mask_vs_gain_over_no_dynamic": float(spearmanr(mask, gain_nodyn).statistic),
        "spearman_local_prior_vs_measured": float(spearmanr(local_prior, measured).statistic),
        "spearman_kb_gate_vs_gain_over_no_kb": float(spearmanr(kb_gate, gain_nokb).statistic),
    }
    (out / f"{args.gene}_spatial_chain_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    pd.DataFrame(
        {
            "spot_index": np.arange(len(measured)),
            "x": coords[:, 0],
            "y": coords[:, 1],
            "measured": measured,
            "global_prior": global_prior,
            "dynamic_mask": mask,
            "local_prior": local_prior,
            "prediction_no_dynamic": prediction_nodyn,
            "prediction_no_kb": prediction_nokb,
            "prediction_full": prediction,
            "error_gain_full_vs_no_kb": gain_nokb,
            "error_gain_full_vs_no_dynamic": gain_nodyn,
            "kb_gate": kb_gate,
        }
    ).to_csv(out / f"{args.gene}_per_spot_chain.csv", index=False)

    relations = pd.read_csv(args.relations).fillna("")
    neighbors = pd.read_csv(args.neighbors).fillna("").drop_duplicates("neighbor_gene").head(12)

    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
        }
    )
    pdf_path = out / f"{args.gene}_knowledge_to_prediction_chain.pdf"
    with PdfPages(pdf_path) as pdf:
        fig, axes = plt.subplots(2, 4, figsize=(14, 7.5), constrained_layout=True)
        expr_min = np.percentile(
            np.concatenate([measured, prediction, prediction_nokb, prediction_nodyn]), 1
        )
        expr_max = np.percentile(
            np.concatenate([measured, prediction, prediction_nokb, prediction_nodyn]), 99
        )
        panels = [
            (measured, "Measured", expr_min, expr_max, PASTEL, None),
            (global_prior, f"Global prior\nPCC={metrics['pcc_global_prior']:.3f}", None, None, PASTEL, None),
            (mask, f"Dynamic mask\nmean={metrics['mean_mask']:.3f}", 0, 1, "viridis", None),
            (local_prior, f"Localized prior\nPCC={metrics['pcc_local_prior']:.3f}", None, None, PASTEL, None),
            (prediction_nodyn, f"No dynamic retrieval\nPCC={metrics['pcc_no_dynamic']:.3f}", expr_min, expr_max, PASTEL, None),
            (prediction_nokb, f"Strict No-KB\nPCC={metrics['pcc_no_kb']:.3f}", expr_min, expr_max, PASTEL, None),
            (prediction, f"Full dynamic KB\nPCC={metrics['pcc_full']:.3f}", expr_min, expr_max, PASTEL, None),
            (
                gain_nokb,
                "Local error gain\nFull vs No-KB",
                None,
                None,
                "RdBu_r",
                TwoSlopeNorm(vcenter=0, vmin=np.percentile(gain_nokb, 2), vmax=np.percentile(gain_nokb, 98)),
            ),
        ]
        for ax, (values, title, vmin, vmax, cmap, norm) in zip(axes.flat, panels):
            scatter = ax.scatter(
                coords[:, 0],
                coords[:, 1],
                c=values,
                s=2.2,
                cmap=cmap,
                vmin=vmin if norm is None else None,
                vmax=vmax if norm is None else None,
                norm=norm,
                rasterized=True,
            )
            ax.invert_yaxis()
            ax.axis("off")
            ax.set_title(title, fontsize=10, fontweight="bold")
            fig.colorbar(scatter, ax=ax, fraction=0.035, pad=0.035)
        fig.suptitle(
            f"{args.gene}: external knowledge retrieval, spatial adaptation and prediction",
            fontsize=15,
            fontweight="bold",
        )
        pdf.savefig(fig, bbox_inches="tight")
        fig.savefig(out / f"{args.gene}_spatial_chain.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

        graph = nx.Graph()
        graph.add_node(args.gene, kind="target")
        core = relations[relations["core_relation_present"] == 1]
        augmented_only = relations[
            (relations["agent_augmented_relation_present"] == 1)
            & (relations["core_relation_present"] == 0)
        ]
        for _, row in core.iterrows():
            protein = row["protein_feature"]
            graph.add_node(protein, kind="protein_core")
            labels = []
            if int(row["cellmarker_shared_count"]):
                labels.append("CellMarker")
            if int(row["reactome_shared_count"]):
                labels.append("Reactome")
            graph.add_edge(protein, args.gene, label="+".join(labels), kind="core")
        for _, row in augmented_only.head(6).iterrows():
            protein = row["protein_feature"]
            graph.add_node(protein, kind="protein_augmented")
            graph.add_edge(protein, args.gene, label="HPA", kind="augmented")
        for _, row in neighbors.iterrows():
            neighbor = row["neighbor_gene"]
            graph.add_node(neighbor, kind="gene")
            labels = []
            if int(row["reactome_shared_count"]):
                labels.append("Reactome")
            if int(row["shared_celltype_count"]):
                labels.append("cell type")
            graph.add_edge(args.gene, neighbor, label="+".join(labels), kind="gene")

        fig, ax = plt.subplots(figsize=(12, 8), constrained_layout=True)
        pos = nx.spring_layout(graph, seed=18, k=1.25)
        colors = {
            "target": "#D55E00",
            "protein_core": "#4477AA",
            "protein_augmented": "#AA4499",
            "gene": "#228833",
        }
        sizes = {"target": 2200, "protein_core": 1000, "protein_augmented": 850, "gene": 750}
        for kind in colors:
            nodes = [node for node, attrs in graph.nodes(data=True) if attrs["kind"] == kind]
            nx.draw_networkx_nodes(
                graph,
                pos,
                nodelist=nodes,
                node_color=colors[kind],
                node_size=sizes[kind],
                alpha=0.9,
                ax=ax,
            )
        edge_colors = [
            "#4477AA" if attrs["kind"] == "core" else "#AA4499" if attrs["kind"] == "augmented" else "#77A77A"
            for _, _, attrs in graph.edges(data=True)
        ]
        nx.draw_networkx_edges(graph, pos, edge_color=edge_colors, width=1.4, alpha=0.75, ax=ax)
        nx.draw_networkx_labels(graph, pos, font_family="Arial", font_size=8, ax=ax)
        edge_labels = {(u, v): attrs["label"] for u, v, attrs in graph.edges(data=True)}
        nx.draw_networkx_edge_labels(
            graph, pos, edge_labels=edge_labels, font_size=6.5, rotate=False, ax=ax
        )
        ax.set_title(
            f"{args.gene} knowledge neighborhood\nblue: causal-run core sources; purple: Agent-added HPA relations",
            fontsize=14,
            fontweight="bold",
        )
        ax.axis("off")
        pdf.savefig(fig, bbox_inches="tight")
        fig.savefig(out / f"{args.gene}_knowledge_network.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

    print(json.dumps({"metrics": metrics, "pdf": str(pdf_path)}, indent=2))


if __name__ == "__main__":
    main()
