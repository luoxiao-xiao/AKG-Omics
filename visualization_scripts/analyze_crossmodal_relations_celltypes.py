#!/usr/bin/env python3
import argparse
import json
import math
import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.spatial import cKDTree
from scipy.stats import pearsonr


def load_task(rep_dir):
    rep_dir = Path(rep_dir)
    metrics = pd.read_csv(rep_dir / "all_feature_metrics_basic.csv")
    names = [None] * np.load(rep_dir / "pred_full.npy", mmap_mode="r").shape[1]
    for row in metrics.itertuples():
        names[int(row.feature_index)] = str(row.feature_name)
    return {
        "gt": np.load(rep_dir / "gt_full.npy"),
        "pred": np.load(rep_dir / "pred_full.npy"),
        "coords": np.load(rep_dir / "coords.npy")[:, :2],
        "metrics": metrics,
        "names": names,
        "lookup": {str(name).upper(): index for index, name in enumerate(names)},
    }


def align_indices(reference_coords, moving_coords):
    if len(reference_coords) == len(moving_coords):
        shift = np.median(reference_coords - moving_coords, axis=0)
    else:
        shift = np.median(reference_coords, axis=0) - np.median(moving_coords, axis=0)
    distances, indices = cKDTree(moving_coords + shift).query(reference_coords, k=1)
    return indices, distances, shift


def corr_columns(left, right):
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    left -= left.mean(axis=0, keepdims=True)
    right -= right.mean(axis=0, keepdims=True)
    den = np.sqrt((left * left).sum(axis=0)[:, None] * (right * right).sum(axis=0)[None, :]) + 1e-8
    return (left.T @ right) / den


def plot_relation_heatmaps(
    true_matrix,
    pred_matrix,
    rows,
    cols,
    title,
    output,
    cognate=None,
    x_label="Column feature",
    y_label="Row feature",
):
    width = max(10.0, 0.42 * len(cols) * 2)
    height = max(5.5, 0.39 * len(rows) + 1.8)
    fig, axes = plt.subplots(1, 2, figsize=(width, height), constrained_layout=True)
    vmax = max(0.35, float(np.nanpercentile(np.abs(np.r_[true_matrix.ravel(), pred_matrix.ravel()]), 98)))
    for ax, matrix, subtitle in zip(axes, [true_matrix, pred_matrix], ["Measured", "Generated from H&E"]):
        sns.heatmap(
            matrix,
            ax=ax,
            cmap="vlag",
            center=0,
            vmin=-vmax,
            vmax=vmax,
            xticklabels=cols,
            yticklabels=rows,
            square=True,
            linewidths=0.35,
            linecolor="white",
            cbar_kws={"label": "Spatial Pearson correlation", "shrink": 0.72},
        )
        ax.set_title(subtitle, fontsize=11, weight="bold")
        ax.set_xlabel(x_label, fontsize=10, weight="bold", labelpad=8)
        ax.set_ylabel(y_label, fontsize=10, weight="bold", labelpad=8)
        ax.tick_params(axis="x", rotation=50, labelsize=7)
        ax.tick_params(axis="y", rotation=0, labelsize=8)
        if cognate:
            for row_index, col_index in cognate:
                ax.add_patch(
                    plt.Rectangle(
                        (col_index, row_index),
                        1,
                        1,
                        fill=False,
                        edgecolor="#111111",
                        linewidth=1.8,
                    )
                )
    fig.suptitle(title, fontsize=13, weight="bold")
    fig.savefig(output.with_suffix(".png"), dpi=400, bbox_inches="tight", facecolor="white")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def analyze_gene_protein(gene_task, protein_task, mapping, output_dir):
    pairs = []
    for protein, gene in mapping.items():
        gene_index = gene_task["lookup"].get(gene.upper())
        protein_index = protein_task["lookup"].get(protein.upper())
        if gene_index is not None and protein_index is not None:
            pairs.append((protein, gene, protein_index, gene_index))
    if not pairs:
        raise RuntimeError("No cognate protein-gene pairs are present in the targeted panel.")

    moving_indices, distances, shift = align_indices(gene_task["coords"], protein_task["coords"])
    proteins = [item[0] for item in pairs]
    genes = [item[1] for item in pairs]
    protein_indices = [item[2] for item in pairs]
    gene_indices = [item[3] for item in pairs]
    true_gene = gene_task["gt"][:, gene_indices]
    pred_gene = gene_task["pred"][:, gene_indices]
    true_protein = protein_task["gt"][moving_indices][:, protein_indices]
    pred_protein = protein_task["pred"][moving_indices][:, protein_indices]
    true_matrix = corr_columns(true_protein, true_gene)
    pred_matrix = corr_columns(pred_protein, pred_gene)
    cognate = [(index, index) for index in range(len(pairs))]
    plot_relation_heatmaps(
        true_matrix,
        pred_matrix,
        proteins,
        genes,
        "Cross-section coherence of cognate protein-gene pairs",
        output_dir / "protein_gene_crossmodal_coherence",
        cognate=cognate,
        x_label="Gene expression",
        y_label="Protein abundance",
    )

    rows = []
    for index, (protein, gene, _, _) in enumerate(pairs):
        rows.append(
            {
                "protein": protein,
                "gene": gene,
                "measured_cognate_correlation": true_matrix[index, index],
                "generated_cognate_correlation": pred_matrix[index, index],
                "absolute_coherence_error": abs(pred_matrix[index, index] - true_matrix[index, index]),
                "coordinate_shift_x": shift[0],
                "coordinate_shift_y": shift[1],
                "median_nearest_neighbor_distance": float(np.median(distances)),
            }
        )
    table = pd.DataFrame(rows).sort_values("generated_cognate_correlation", ascending=False)
    table.to_csv(output_dir / "protein_gene_cognate_relations.csv", index=False)
    pd.DataFrame(pred_matrix, index=proteins, columns=genes).to_csv(output_dir / "protein_gene_generated_correlation_matrix.csv")
    pd.DataFrame(true_matrix, index=proteins, columns=genes).to_csv(output_dir / "protein_gene_measured_correlation_matrix.csv")


def analyze_gene_mz(gene_task, mz_task, output_dir, top_genes=12, top_mz=12, met_kb=None):
    known_pairs = []
    if met_kb:
        kb = pickle.load(open(met_kb, "rb"))
        relation = np.asarray(kb["metabolite_gene_prior"])
        mz_known, gene_known = np.where(relation > 0)
        known_pairs = [
            (int(mi), int(gi), float(relation[mi, gi]))
            for mi, gi in zip(mz_known, gene_known)
        ]

    gene_order = [gi for _, gi, _ in known_pairs]
    gene_order.extend(
        gene_task["metrics"].sort_values("PCC", ascending=False)["feature_index"].astype(int).tolist()
    )
    gene_indices = list(dict.fromkeys(gene_order))[:top_genes]
    mz_order = [mi for mi, _, _ in known_pairs]
    mz_order.extend(
        mz_task["metrics"].sort_values("PCC", ascending=False)["feature_index"].astype(int).tolist()
    )
    mz_indices = list(dict.fromkeys(mz_order))[:top_mz]
    gene_metrics = gene_task["metrics"].set_index("feature_index").loc[gene_indices].reset_index()
    mz_metrics = mz_task["metrics"].set_index("feature_index").loc[mz_indices].reset_index()
    gene_names = gene_metrics["feature_name"].astype(str).tolist()
    mz_names = mz_metrics["feature_name"].astype(str).tolist()

    moving_indices, distances, shift = align_indices(gene_task["coords"], mz_task["coords"])
    true_gene = gene_task["gt"][:, gene_indices]
    pred_gene = gene_task["pred"][:, gene_indices]
    true_mz = mz_task["gt"][moving_indices][:, mz_indices]
    pred_mz = mz_task["pred"][moving_indices][:, mz_indices]
    true_matrix = corr_columns(true_mz, true_gene)
    pred_matrix = corr_columns(pred_mz, pred_gene)
    known_lookup = {(mi, gi): weight for mi, gi, weight in known_pairs}
    known_cells = [
        (row_index, col_index)
        for row_index, mi in enumerate(mz_indices)
        for col_index, gi in enumerate(gene_indices)
        if (mi, gi) in known_lookup
    ]
    plot_relation_heatmaps(
        true_matrix,
        pred_matrix,
        mz_names,
        gene_names,
        "Candidate gene-m/z spatial associations",
        output_dir / "gene_mz_crossmodal_coherence",
        cognate=known_cells,
        x_label="Gene expression",
        y_label="Mass-to-charge feature (m/z)",
    )

    rows = []
    for i, mz_name in enumerate(mz_names):
        for j, gene_name in enumerate(gene_names):
            rows.append(
                {
                    "mz_feature": mz_name,
                    "gene": gene_name,
                    "measured_spatial_correlation": true_matrix[i, j],
                    "generated_spatial_correlation": pred_matrix[i, j],
                    "absolute_coherence_error": abs(pred_matrix[i, j] - true_matrix[i, j]),
                    "coordinate_shift_x": shift[0],
                    "coordinate_shift_y": shift[1],
                    "median_nearest_neighbor_distance": float(np.median(distances)),
                    "interpretation": "candidate spatial association; m/z identity requires metabolite confirmation",
                    "knowledge_prior_weight": known_lookup.get((mz_indices[i], gene_indices[j]), 0.0),
                }
            )
    pd.DataFrame(rows).sort_values("generated_spatial_correlation", ascending=False).to_csv(
        output_dir / "gene_mz_candidate_relations.csv", index=False
    )


def zscore_columns(values):
    values = np.asarray(values, dtype=float)
    mean = values.mean(axis=0, keepdims=True)
    std = values.std(axis=0, keepdims=True)
    return (values - mean) / np.maximum(std, 1e-8)


def smooth_scores(coords, values, neighbors=10):
    k = min(int(neighbors), len(coords))
    indices = cKDTree(coords).query(coords, k=k)[1]
    if indices.ndim == 1:
        indices = indices[:, None]
    return values[indices].mean(axis=1)


def celltype_scores(task, marker_sets):
    results = {}
    rows = []
    metric_lookup = task["metrics"].set_index(task["metrics"]["feature_name"].str.upper())
    for cell_type, markers in marker_sets.items():
        available = [(marker, task["lookup"].get(marker.upper())) for marker in markers]
        available = [(marker, index) for marker, index in available if index is not None]
        if len(available) < 3:
            continue
        indices = [index for _, index in available]
        true_score = zscore_columns(task["gt"][:, indices]).mean(axis=1)
        pred_score = zscore_columns(task["pred"][:, indices]).mean(axis=1)
        pcc = pearsonr(true_score, pred_score).statistic if np.std(true_score) > 0 and np.std(pred_score) > 0 else 0.0
        rmse = float(np.sqrt(np.mean((true_score - pred_score) ** 2)))
        gene_pcc = [
            float(metric_lookup.loc[marker.upper(), "PCC"])
            for marker, _ in available
            if marker.upper() in metric_lookup.index
        ]
        results[cell_type] = {
            "true": true_score,
            "pred": pred_score,
            "markers": [marker for marker, _ in available],
        }
        rows.append(
            {
                "cell_type": cell_type,
                "marker_count": len(available),
                "markers": ";".join(marker for marker, _ in available),
                "celltype_score_PCC": pcc,
                "celltype_score_RMSE": rmse,
                "mean_marker_gene_PCC": float(np.mean(gene_pcc)) if gene_pcc else np.nan,
            }
        )
    return results, pd.DataFrame(rows).sort_values("celltype_score_PCC", ascending=False)


def scatter_map(ax, coords, values, title, cmap="magma", categorical=False, vmin=None, vmax=None):
    size = max(1.0, min(9.0, 70000.0 / max(len(coords), 1)))
    if categorical:
        ax.scatter(coords[:, 0], coords[:, 1], c=values, s=size, cmap=cmap, linewidths=0, rasterized=True)
    else:
        ax.scatter(coords[:, 0], coords[:, 1], c=values, s=size, cmap=cmap, vmin=vmin, vmax=vmax, linewidths=0, rasterized=True)
    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=9.5, weight="bold")


def add_panel_label(ax, label):
    ax.text(
        -0.04,
        1.04,
        label,
        transform=ax.transAxes,
        fontsize=12,
        weight="bold",
        va="bottom",
        ha="right",
    )


def save_assignment_panel(coords, labels, ordered, colors, title, output):
    fig, ax = plt.subplots(figsize=(6.2, 5.0))
    cmap = matplotlib.colors.ListedColormap(colors)
    scatter_map(ax, coords, labels, title, cmap=cmap, categorical=True)
    handles = [
        plt.Line2D([0], [0], marker="o", linestyle="", markersize=7, color=colors[i], label=name)
        for i, name in enumerate(ordered)
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=min(3, len(handles)),
        frameon=False,
        fontsize=8,
    )
    fig.subplots_adjust(bottom=0.18)
    fig.savefig(output.with_suffix(".png"), dpi=400, bbox_inches="tight", facecolor="white")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_ranking_panel(table, output):
    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    y = np.arange(len(table))
    ax.barh(y, table["celltype_score_PCC"], color="#3F7CAC")
    ax.set_yticks(y, table["cell_type"], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(min(-0.05, float(table["celltype_score_PCC"].min()) - 0.03), 1.0)
    ax.set_xlabel("Pearson correlation coefficient (PCC)")
    ax.set_title("Cell-type reconstruction ranking", fontsize=11, weight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(output.with_suffix(".png"), dpi=400, bbox_inches="tight", facecolor="white")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_score_panel(coords, values, title, output, vmin, vmax):
    fig, ax = plt.subplots(figsize=(6.2, 5.0))
    scatter_map(ax, coords, values, title, vmin=vmin, vmax=vmax)
    scalar = matplotlib.cm.ScalarMappable(
        norm=matplotlib.colors.Normalize(vmin=vmin, vmax=vmax),
        cmap="magma",
    )
    scalar.set_array([])
    cbar = fig.colorbar(scalar, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("Generated marker score (z-score)")
    fig.savefig(output.with_suffix(".png"), dpi=400, bbox_inches="tight", facecolor="white")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_celltype_summary(task, marker_sets, dataset_name, output_dir):
    scores, table = celltype_scores(task, marker_sets)
    table.to_csv(output_dir / f"{dataset_name}_celltype_prediction_ranking.csv", index=False)
    if table.empty:
        return
    ordered = table["cell_type"].tolist()
    true_matrix = np.column_stack([scores[name]["true"] for name in ordered])
    pred_matrix = np.column_stack([scores[name]["pred"] for name in ordered])
    true_labels = np.argmax(smooth_scores(task["coords"], true_matrix), axis=1)
    pred_labels = np.argmax(smooth_scores(task["coords"], pred_matrix), axis=1)
    colors = sns.color_palette("Set2", n_colors=len(ordered))
    cmap = matplotlib.colors.ListedColormap(colors)

    fig = plt.figure(figsize=(12.8, 7.5))
    grid = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.05], width_ratios=[1.0, 1.0, 0.78], hspace=0.28, wspace=0.18)
    ax_true = fig.add_subplot(grid[0, 0])
    ax_pred = fig.add_subplot(grid[0, 1])
    scatter_map(ax_true, task["coords"], true_labels, "Measured cell-type score assignment", cmap=cmap, categorical=True)
    scatter_map(ax_pred, task["coords"], pred_labels, "Generated cell-type score assignment", cmap=cmap, categorical=True)
    add_panel_label(ax_true, "(a)")
    add_panel_label(ax_pred, "(b)")
    handles = [
        plt.Line2D([0], [0], marker="o", linestyle="", markersize=7, color=colors[i], label=name)
        for i, name in enumerate(ordered)
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=min(4, len(handles)),
        frameon=False,
        fontsize=8,
    )

    ax_bar = fig.add_subplot(grid[0, 2])
    y = np.arange(len(table))
    ax_bar.barh(y, table["celltype_score_PCC"], color="#3F7CAC")
    ax_bar.set_yticks(y, table["cell_type"], fontsize=8)
    ax_bar.invert_yaxis()
    ax_bar.set_xlim(min(-0.05, float(table["celltype_score_PCC"].min()) - 0.03), 1.0)
    ax_bar.set_xlabel("PCC")
    ax_bar.set_title("Cell-type reconstruction ranking", fontsize=10, weight="bold")
    ax_bar.spines[["top", "right"]].set_visible(False)
    add_panel_label(ax_bar, "(c)")

    top_types = ordered[:3]
    for column, cell_type in enumerate(top_types):
        ax = fig.add_subplot(grid[1, column])
        true_values = scores[cell_type]["true"]
        pred_values = scores[cell_type]["pred"]
        low = float(np.percentile(np.r_[true_values, pred_values], 2))
        high = float(np.percentile(np.r_[true_values, pred_values], 98))
        scatter_map(ax, task["coords"], pred_values, f"{cell_type}\nGenerated score", vmin=low, vmax=high)
        add_panel_label(ax, f"({chr(ord('d') + column)})")
    fig.suptitle(f"{dataset_name}: tissue microenvironment reconstruction", fontsize=14, weight="bold")
    fig.subplots_adjust(bottom=0.10)
    fig.savefig(output_dir / f"{dataset_name}_celltype_reconstruction.png", dpi=400, bbox_inches="tight", facecolor="white")
    fig.savefig(output_dir / f"{dataset_name}_celltype_reconstruction.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)

    panel_dir = output_dir / f"{dataset_name}_celltype_panels"
    panel_dir.mkdir(parents=True, exist_ok=True)
    save_assignment_panel(
        task["coords"],
        true_labels,
        ordered,
        colors,
        "Measured cell-type score assignment",
        panel_dir / "a_measured_celltype_assignment",
    )
    save_assignment_panel(
        task["coords"],
        pred_labels,
        ordered,
        colors,
        "Generated cell-type score assignment",
        panel_dir / "b_generated_celltype_assignment",
    )
    save_ranking_panel(table, panel_dir / "c_celltype_reconstruction_ranking")
    for column, cell_type in enumerate(top_types):
        true_values = scores[cell_type]["true"]
        pred_values = scores[cell_type]["pred"]
        low = float(np.percentile(np.r_[true_values, pred_values], 2))
        high = float(np.percentile(np.r_[true_values, pred_values], 98))
        safe_name = cell_type.lower().replace(" / ", "_").replace(" ", "_")
        save_score_panel(
            task["coords"],
            pred_values,
            f"{cell_type}: generated marker score",
            panel_dir / f"{chr(ord('d') + column)}_{safe_name}_generated_score",
            low,
            high,
        )

    composition = []
    for index, cell_type in enumerate(ordered):
        composition.append(
            {
                "cell_type": cell_type,
                "measured_percent": 100.0 * float(np.mean(true_labels == index)),
                "generated_percent": 100.0 * float(np.mean(pred_labels == index)),
            }
        )
    pd.DataFrame(composition).to_csv(output_dir / f"{dataset_name}_celltype_composition.csv", index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--markers", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--coad-gene-rep")
    parser.add_argument("--coad-protein-rep")
    parser.add_argument("--brain-gene-rep", required=True)
    parser.add_argument("--brain-mz-rep", required=True)
    parser.add_argument("--met-kb")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    markers = json.loads(Path(args.markers).read_text(encoding="utf-8"))
    brain_gene = load_task(args.brain_gene_rep)
    brain_mz = load_task(args.brain_mz_rep)
    analyze_gene_mz(brain_gene, brain_mz, output_dir, met_kb=args.met_kb)
    plot_celltype_summary(brain_gene, markers["mouse_brain"], "mouse_brain", output_dir)

    if args.coad_gene_rep and args.coad_protein_rep:
        coad_gene = load_task(args.coad_gene_rep)
        coad_protein = load_task(args.coad_protein_rep)
        analyze_gene_protein(coad_gene, coad_protein, markers["protein_to_gene"], output_dir)
        plot_celltype_summary(coad_gene, markers["coad"], "coad", output_dir)

    print(f">>> output={output_dir}")


if __name__ == "__main__":
    main()
