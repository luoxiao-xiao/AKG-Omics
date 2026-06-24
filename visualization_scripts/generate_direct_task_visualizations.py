#!/usr/bin/env python3
import argparse
import csv
import math
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROTEIN_TO_GENE = {
    "CD11C": "ITGAX",
    "CD11B": "ITGAM",
    "CD14": "CD14",
    "CD163": "CD163",
    "CD20": "MS4A1",
    "CD3": "CD3D",
    "CD3D": "CD3D",
    "CD4": "CD4",
    "CD8": "CD8A",
    "CD8A": "CD8A",
    "CD31": "PECAM1",
    "CD34": "CD34",
    "CD45": "PTPRC",
    "CD56": "NCAM1",
    "CD68": "CD68",
    "EPCAM": "EPCAM",
    "PANCK": "KRT8",
    "PAN-CYTOKERATIN": "KRT8",
    "SMA": "ACTA2",
    "ACTA2": "ACTA2",
    "VIMENTIN": "VIM",
    "VIM": "VIM",
}


def clean_symbol(value):
    return re.sub(r"[^A-Z0-9]+", "", str(value).upper())


def safe_name(value):
    return re.sub(r"[^A-Za-z0-9_.+-]+", "_", str(value)).strip("_")[:100] or "feature"


def robust_scale01(values, lo=1.0, hi=99.0):
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.zeros_like(values)
    low, high = np.percentile(finite, [lo, hi])
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        low, high = float(np.nanmin(finite)), float(np.nanmax(finite))
    if high <= low:
        high = low + 1e-8
    return np.clip((values - low) / (high - low), 0.0, 1.0)


def feature_metrics(gt, pred, names):
    gt = np.asarray(gt, dtype=np.float32)
    pred = np.asarray(pred, dtype=np.float32)
    gt_c = gt - gt.mean(axis=0, keepdims=True)
    pred_c = pred - pred.mean(axis=0, keepdims=True)
    den = np.sqrt((gt_c * gt_c).sum(axis=0) * (pred_c * pred_c).sum(axis=0)) + 1e-8
    pcc = (gt_c * pred_c).sum(axis=0) / den
    rmse = np.sqrt(np.mean((gt - pred) ** 2, axis=0))
    return pd.DataFrame({
        "feature_index": np.arange(gt.shape[1], dtype=int),
        "feature_name": [str(x) for x in names],
        "PCC": np.nan_to_num(pcc, nan=0.0),
        "RMSE": np.nan_to_num(rmse, nan=np.inf),
    })


def load_task(task_dir):
    rep = next(iter(sorted(task_dir.glob("rep*_seed*"))), None)
    if rep is None:
        raise FileNotFoundError(f"No rep*_seed* under {task_dir}")
    gt = np.load(rep / "gt_full.npy")
    pred = np.load(rep / "pred_full.npy")
    metrics_path = rep / "all_feature_metrics_basic.csv"
    if not metrics_path.exists():
        metrics_path = rep / "top20_feature_metrics.csv"
    metrics = pd.read_csv(metrics_path)
    names = [f"feature_{i}" for i in range(gt.shape[1])]
    for row in metrics.itertuples():
        idx = int(row.feature_index)
        if 0 <= idx < len(names):
            names[idx] = str(row.feature_name)
    if len(metrics) != gt.shape[1]:
        metrics = feature_metrics(gt, pred, names)
    coords_path = rep / "coords.npy"
    if not coords_path.exists():
        try:
            from akg_omics.visualize_proteogenomics_predictions import load_coords
            coords = load_coords(rep, gt.shape[0])
        except Exception as exc:
            raise FileNotFoundError(f"Missing coordinates for {rep}: {exc}") from exc
    else:
        coords = np.load(coords_path)
    return {
        "task_dir": task_dir,
        "rep": rep,
        "gt": gt,
        "pred": pred,
        "metrics": metrics.sort_values("PCC", ascending=False).reset_index(drop=True),
        "coords": np.asarray(coords)[:, :2],
        "names": names,
    }


def find_task(analysis_root, names):
    for name in names:
        matches = sorted(analysis_root.glob(f"{name}*"))
        for path in matches:
            if path.is_dir() and list(path.glob("rep*_seed*")):
                return path
    return None


def discover_met_root(search_root):
    candidates = []
    for path in search_root.glob("*/results/akg_omics_visualization/**/detailed_analysis"):
        if find_task(path, ["task7_he_to_metabolism"]) and find_task(
            path, ["task8_he_to_gene_in_metabolomics"]
        ):
            candidates.append(path)
    if not candidates:
        raise FileNotFoundError(f"No metabolomics direct-task detailed_analysis found under {search_root}")
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def threshold_scatter(ax, coords, values, title, threshold=0.5, cmap="magma"):
    scaled = robust_scale01(values)
    n = len(scaled)
    size = max(1.0, min(10.0, 70000.0 / max(n, 1)))
    low = scaled < threshold
    high = ~low
    ax.scatter(
        coords[low, 0], coords[low, 1], s=size, c="#000000",
        linewidths=0, alpha=1.0, rasterized=True,
    )
    if np.any(high):
        sc = ax.scatter(
            coords[high, 0], coords[high, 1], s=size, c=scaled[high],
            cmap=cmap, vmin=threshold, vmax=1.0, linewidths=0,
            alpha=1.0, rasterized=True,
        )
    else:
        sc = None
    ax.set_title(title, fontsize=10, pad=7)
    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_facecolor("white")
    return sc


def save_threshold_pair(task, row, out_dir, threshold):
    idx = int(row["feature_index"])
    name = str(row["feature_name"])
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.6), constrained_layout=True)
    threshold_scatter(axes[0], task["coords"], task["gt"][:, idx], "Measured", threshold)
    sc = threshold_scatter(
        axes[1], task["coords"], task["pred"][:, idx],
        f"Generated | PCC={float(row['PCC']):.3f}", threshold,
    )
    if sc is not None:
        cb = fig.colorbar(sc, ax=axes, shrink=0.78, fraction=0.035, pad=0.02)
        cb.set_label(f"Relative expression (values < {threshold:g} shown in black)", fontsize=8)
        cb.ax.tick_params(labelsize=7)
    fig.suptitle(name, fontsize=11, fontweight="semibold")
    path = out_dir / f"{safe_name(name)}_threshold{threshold:g}_full.png"
    fig.savefig(path, dpi=350, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def hotspot_metrics(true_v, pred_v, threshold=0.5):
    true_s = robust_scale01(true_v)
    pred_s = robust_scale01(pred_v)
    true_hot = true_s >= threshold
    pred_hot = pred_s >= threshold
    tp = int(np.sum(true_hot & pred_hot))
    recall = tp / max(int(np.sum(true_hot)), 1)
    precision = tp / max(int(np.sum(pred_hot)), 1)
    return recall, precision


def heatmap(ax, matrix, row_labels, col_labels, title, vmin=0.0, vmax=1.0, cmap="viridis"):
    im = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=11, pad=8)
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=35, ha="right", fontsize=8)
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=8)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            color = "white" if value < (vmin + vmax) / 2 else "black"
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=7, color=color)
    return im


def make_gene_protein_heatmap(gene_task, protein_task, out_dir, top_n, threshold):
    gene_lookup = {
        clean_symbol(row.feature_name): row
        for row in gene_task["metrics"].itertuples()
    }
    rows = []
    for prow in protein_task["metrics"].itertuples():
        protein = str(prow.feature_name)
        key = clean_symbol(protein)
        gene = PROTEIN_TO_GENE.get(key, key)
        grow = gene_lookup.get(clean_symbol(gene))
        if grow is None:
            continue
        gi, pi = int(grow.feature_index), int(prow.feature_index)
        gene_recall, gene_precision = hotspot_metrics(
            gene_task["gt"][:, gi], gene_task["pred"][:, gi], threshold
        )
        protein_recall, protein_precision = hotspot_metrics(
            protein_task["gt"][:, pi], protein_task["pred"][:, pi], threshold
        )
        joint = min(float(grow.PCC), float(prow.PCC))
        rows.append({
            "gene": str(grow.feature_name),
            "protein": protein,
            "gene_feature_index": gi,
            "protein_feature_index": pi,
            "gene_PCC": float(grow.PCC),
            "protein_PCC": float(prow.PCC),
            "gene_hotspot_recall": gene_recall,
            "protein_hotspot_recall": protein_recall,
            "gene_hotspot_precision": gene_precision,
            "protein_hotspot_precision": protein_precision,
            "joint_score": joint,
        })
    table = pd.DataFrame(rows)
    if not table.empty:
        table = table.sort_values(
            ["joint_score", "gene_PCC", "protein_PCC"], ascending=False
        ).head(top_n)
        table["pairing_mode"] = "cognate_gene_protein"
        table.to_csv(out_dir / "direct_he_gene_protein_matched_features.csv", index=False)
        matrix = table[
            ["gene_PCC", "protein_PCC", "gene_hotspot_recall", "protein_hotspot_recall"]
        ].to_numpy()
        labels = [f"{r.gene} / {r.protein}" for r in table.itertuples()]
        columns = [
            "Gene PCC", "Protein PCC", "Gene hotspot recall", "Protein hotspot recall"
        ]
        title = "Direct H&E generation of matched gene-protein markers"
    else:
        # The evaluated gene panel may not contain the cognate genes for the
        # measured protein panel. Report modality-specific generation quality
        # without implying unsupported one-to-one biological correspondence.
        table = pd.DataFrame(
            quality_rows(gene_task, "Gene", top_n, threshold)
            + quality_rows(protein_task, "Protein", top_n, threshold)
        )
        table["pairing_mode"] = "modality_specific_no_cognate_overlap"
        table.to_csv(
            out_dir / "direct_he_gene_protein_modality_specific_features.csv",
            index=False,
        )
        matrix = table[
            ["PCC", "hotspot_recall", "hotspot_precision", "relative_RMSE_quality"]
        ].to_numpy()
        labels = [f"{r.modality}: {r.feature_name}" for r in table.itertuples()]
        columns = [
            "PCC", "Hotspot recall", "Hotspot precision", "Relative RMSE quality"
        ]
        title = "Direct H&E generation quality for gene and protein features"
    fig, ax = plt.subplots(figsize=(7.2, max(3.8, 0.42 * len(labels) + 1.7)))
    im = heatmap(
        ax, matrix, labels, columns, title,
    )
    if "modality" in table.columns:
        split = min(top_n, int(np.sum(table["modality"] == "Gene"))) - 0.5
        ax.axhline(split, color="white", linewidth=2.5)
        ax.axhline(split, color="#222222", linewidth=0.8)
    cb = fig.colorbar(im, ax=ax, shrink=0.78, pad=0.03)
    cb.set_label("Generation agreement", fontsize=8)
    fig.tight_layout()
    path = out_dir / "direct_he_gene_protein_generation_heatmap.png"
    fig.savefig(path, dpi=350, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return table, path


def quality_rows(task, modality, top_n, threshold):
    rows = []
    selected = task["metrics"].head(top_n)
    rmse_values = selected["RMSE"].to_numpy(dtype=float)
    rmse_min = float(np.nanmin(rmse_values))
    rmse_max = float(np.nanmax(rmse_values))
    for row in selected.itertuples():
        idx = int(row.feature_index)
        recall, precision = hotspot_metrics(
            task["gt"][:, idx], task["pred"][:, idx], threshold
        )
        rmse_quality = 1.0 - (float(row.RMSE) - rmse_min) / (rmse_max - rmse_min + 1e-8)
        rows.append({
            "modality": modality,
            "feature_name": str(row.feature_name),
            "feature_index": idx,
            "PCC": float(row.PCC),
            "hotspot_recall": recall,
            "hotspot_precision": precision,
            "relative_RMSE_quality": rmse_quality,
        })
    return rows


def make_gene_mz_heatmaps(gene_task, met_task, out_dir, top_n, threshold):
    table = pd.DataFrame(
        quality_rows(gene_task, "Gene", top_n, threshold)
        + quality_rows(met_task, "m/z", top_n, threshold)
    )
    table.to_csv(out_dir / "direct_he_gene_mz_generation_quality.csv", index=False)
    matrix = table[
        ["PCC", "hotspot_recall", "hotspot_precision", "relative_RMSE_quality"]
    ].to_numpy()
    labels = [
        f"{r.modality}: {r.feature_name}"
        for r in table.itertuples()
    ]
    fig, ax = plt.subplots(figsize=(7.4, max(5.2, 0.34 * len(labels) + 1.8)))
    im = heatmap(
        ax,
        matrix,
        labels,
        ["PCC", "Hotspot recall", "Hotspot precision", "Relative RMSE quality"],
        "Direct H&E generation of gene and metabolomic m/z features",
    )
    split = top_n - 0.5
    ax.axhline(split, color="white", linewidth=2.5)
    ax.axhline(split, color="#222222", linewidth=0.8)
    cb = fig.colorbar(im, ax=ax, shrink=0.78, pad=0.03)
    cb.set_label("Generation agreement", fontsize=8)
    fig.tight_layout()
    path = out_dir / "direct_he_gene_mz_generation_quality_heatmap.png"
    fig.savefig(path, dpi=350, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def write_manifest(rows, path):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--proteo-run-root", required=True)
    parser.add_argument("--met-analysis-root", default=None)
    parser.add_argument("--search-root", default=".")
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--top-n-spatial", type=int, default=5)
    parser.add_argument("--top-n-heatmap", type=int, default=12)
    args = parser.parse_args()

    proteo_analysis = Path(args.proteo_run_root) / "detailed_analysis"
    met_analysis = (
        Path(args.met_analysis_root)
        if args.met_analysis_root
        else discover_met_root(Path(args.search_root))
    )
    out_root = Path(args.out_root)
    threshold_root = out_root / "threshold_0p5_full_maps"
    direct_root = out_root / "direct_he_cross_modal_heatmaps"
    threshold_root.mkdir(parents=True, exist_ok=True)
    direct_root.mkdir(parents=True, exist_ok=True)

    proteo_tasks = {}
    for key, names in {
        "gene": ["task1_he_to_gene"],
        "protein": ["task2_he_to_protein"],
        "cross_gene": ["task3_he_protein_to_gene"],
        "cross_protein": ["task4_he_gene_to_protein"],
    }.items():
        path = find_task(proteo_analysis, names)
        if path is not None:
            proteo_tasks[key] = load_task(path)

    met_gene_path = find_task(met_analysis, ["task8_he_to_gene_in_metabolomics"])
    met_mz_path = find_task(met_analysis, ["task7_he_to_metabolism"])
    if met_gene_path is None or met_mz_path is None:
        raise FileNotFoundError(f"Missing direct metabolomics tasks under {met_analysis}")
    met_gene = load_task(met_gene_path)
    met_mz = load_task(met_mz_path)

    manifest = []
    for label, task in {**proteo_tasks, "met_gene": met_gene, "met_mz": met_mz}.items():
        task_out = threshold_root / label
        task_out.mkdir(parents=True, exist_ok=True)
        for rank, (_, row) in enumerate(
            task["metrics"].head(args.top_n_spatial).iterrows(), start=1
        ):
            path = save_threshold_pair(task, row, task_out, args.threshold)
            manifest.append({
                "task": label,
                "rank": rank,
                "feature_name": str(row["feature_name"]),
                "PCC": float(row["PCC"]),
                "figure": str(path),
            })
    write_manifest(manifest, threshold_root / "threshold_visualization_manifest.csv")

    if "gene" not in proteo_tasks or "protein" not in proteo_tasks:
        raise FileNotFoundError("Direct proteogenomics tasks task1/task2 were not found")
    _, gp_path = make_gene_protein_heatmap(
        proteo_tasks["gene"], proteo_tasks["protein"], direct_root,
        args.top_n_heatmap, args.threshold,
    )
    gm_path = make_gene_mz_heatmaps(
        met_gene, met_mz, direct_root, args.top_n_heatmap, args.threshold,
    )

    metadata = pd.DataFrame([
        {"item": "proteogenomics_analysis", "value": str(proteo_analysis)},
        {"item": "metabolomics_analysis", "value": str(met_analysis)},
        {"item": "threshold", "value": args.threshold},
        {"item": "gene_protein_heatmap", "value": str(gp_path)},
        {"item": "gene_mz_heatmaps", "value": str(gm_path)},
    ])
    metadata.to_csv(out_root / "visualization_run_metadata.csv", index=False)
    print(f">>> proteogenomics analysis: {proteo_analysis}")
    print(f">>> metabolomics analysis: {met_analysis}")
    print(f">>> threshold figures: {len(manifest)}")
    print(f">>> saved outputs: {out_root}")


if __name__ == "__main__":
    main()
