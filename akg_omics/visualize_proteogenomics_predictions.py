#!/usr/bin/env python3
import argparse
import math
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd


TASK_TITLES = {
    "task1_he_to_gene": "Task 1: H&E to Gene",
    "task2_he_to_protein": "Task 2: H&E to Protein",
    "task3_he_protein_to_gene": "Task 3: H&E + Protein to Gene",
    "task4_he_gene_to_protein": "Task 4: H&E + Gene to Protein",
}


def sanitize_name(value):
    value = str(value)
    value = re.sub(r"[^\w\-.]+", "_", value)
    return value[:120] or "feature"


def robust_limits(values, low=1.0, high=99.0):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.0, 1.0
    vmin, vmax = np.percentile(values, [low, high])
    if not np.isfinite(vmin) or not np.isfinite(vmax) or math.isclose(vmin, vmax):
        vmin, vmax = float(np.nanmin(values)), float(np.nanmax(values))
    if math.isclose(vmin, vmax):
        vmax = vmin + 1e-6
    return float(vmin), float(vmax)


def robust_scale01(values, low=1.0, high=99.0):
    values = np.asarray(values, dtype=float)
    vmin, vmax = robust_limits(values, low, high)
    scaled = (values - vmin) / (vmax - vmin + 1e-8)
    return np.clip(scaled, 0.0, 1.0)


def rasterize_values(coords, values, bins=620, sigma=0.85, min_count=0.02):
    from scipy.ndimage import gaussian_filter

    coords = np.asarray(coords, dtype=float)
    values = np.asarray(values, dtype=float)
    x = coords[:, 0]
    y = coords[:, 1]
    pad_x = max((x.max() - x.min()) * 0.025, 1e-3)
    pad_y = max((y.max() - y.min()) * 0.025, 1e-3)
    xmin, xmax = x.min() - pad_x, x.max() + pad_x
    ymin, ymax = y.min() - pad_y, y.max() + pad_y
    x_edges = np.linspace(xmin, xmax, int(bins) + 1)
    y_edges = np.linspace(ymin, ymax, int(bins) + 1)

    sum_grid, _, _ = np.histogram2d(x, y, bins=[x_edges, y_edges], weights=values)
    cnt_grid, _, _ = np.histogram2d(x, y, bins=[x_edges, y_edges])
    if sigma and sigma > 0:
        sum_grid = gaussian_filter(sum_grid, sigma=float(sigma), mode="constant")
        cnt_grid = gaussian_filter(cnt_grid, sigma=float(sigma), mode="constant")
    grid = sum_grid / np.maximum(cnt_grid, 1e-8)
    grid[cnt_grid < float(min_count)] = np.nan
    return grid.T, (xmin, xmax, ymin, ymax)


def load_coords(task_dir, n_obs):
    for name in ("coords.npy", "spatial.npy"):
        path = task_dir / name
        if path.exists():
            coords = np.load(path)
            if coords.shape[0] == n_obs and coords.shape[1] >= 2:
                return coords[:, :2].astype(float)

    # Detailed analysis in run.py currently saves gt/pred but not coordinates.
    # Reconstruct the proteogenomics refined cache path used by AKG-Omics run.
    try:
        import anndata as ad
    except Exception as exc:
        raise RuntimeError("anndata is required to reconstruct spatial coordinates") from exc

    task_name = task_dir.parent.name
    target_is_gene = "to_gene" in task_name
    cache_root = Path(os.environ.get("PROTEO_SHARED_CACHE_ROOT", "cache/proteogenomics"))
    sample = os.environ.get("PROTEO_SAMPLE2", "gene") if target_is_gene else os.environ.get("PROTEO_SAMPLE1", "protein")
    grid = os.environ.get("PROTEO_GRID_SIZE", "32")
    he_dim = os.environ.get("PROTEO_HE_DIM", "128")
    gene_topk = os.environ.get("PROTEO_GENE_TOPK", "500")
    gene_lat = os.environ.get("PROTEO_GENE_LATENT_DIM", "128")
    graph_alpha = os.environ.get("PROTEO_BASE_GRAPH_ALPHA", "1.0")
    spatial_k = os.environ.get("SPATIAL_K", "7")
    he_k = os.environ.get("HE_K", "7")
    mode_tag = f"g{grid}_he{he_dim}_gene{gene_topk}_glat{gene_lat}_a{graph_alpha}_sk{spatial_k}_hk{he_k}"
    candidates = [
        cache_root / "proteogenomics" / "refined" / f"{sample}_{mode_tag}.h5ad",
        Path("cache/spatch_raw_joint_gene_latent") / f"{sample}_g32_he128_gene500_a0.7_sk7_hk7.h5ad",
        Path("cache/spatch_raw_joint_gene_latent") / f"{sample}_g32_he128_gene500_glat128_a0.7_sk7_hk7.h5ad",
        Path("cache/spatch_raw_kb_bridge_unified") / f"{sample}_g32_he128_gene500_glat128_a0.7_sk7_hk7.h5ad",
    ]
    h5ad_path = None
    for cand in candidates:
        if not cand.exists():
            continue
        try:
            tmp = ad.read_h5ad(cand, backed="r")
            shape = tuple(tmp.shape)
            tmp.file.close()
        except Exception:
            continue
        if shape[0] == n_obs:
            h5ad_path = cand
            break
    if h5ad_path is None:
        existing = [str(p) for p in candidates if p.exists()]
        raise FileNotFoundError(f"Cannot find coordinate cache with n_obs={n_obs}. Existing candidates: {existing}")
    adata = ad.read_h5ad(h5ad_path)
    if "spatial" in adata.obsm:
        coords = np.asarray(adata.obsm["spatial"])[:, :2]
    elif "image_coor" in adata.obsm:
        coords = np.asarray(adata.obsm["image_coor"])[:, :2]
    else:
        raise ValueError(f"No spatial/image_coor coordinates in {h5ad_path}")
    if coords.shape[0] != n_obs:
        raise ValueError(f"Coordinate count mismatch for {task_dir}: coords={coords.shape[0]}, n_obs={n_obs}")
    np.save(task_dir / "coords.npy", coords.astype(np.float32))
    return coords.astype(float)


def style_axis(ax, coords):
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.invert_yaxis()
    pad_x = max((coords[:, 0].max() - coords[:, 0].min()) * 0.03, 1e-3)
    pad_y = max((coords[:, 1].max() - coords[:, 1].min()) * 0.03, 1e-3)
    ax.set_xlim(coords[:, 0].min() - pad_x, coords[:, 0].max() + pad_x)
    ax.set_ylim(coords[:, 1].max() + pad_y, coords[:, 1].min() - pad_y)


def choose_roi(coords, true_v, pred_v, min_points=3200, quantile=90.0):
    coords = np.asarray(coords, dtype=float)
    true_s = robust_scale01(true_v)
    pred_s = robust_scale01(pred_v)
    score = 0.65 * np.minimum(true_s, pred_s) + 0.35 * true_s
    mask = score >= np.percentile(score, quantile)
    if mask.sum() < min_points:
        order = np.argsort(score)[::-1]
        mask = np.zeros(score.shape[0], dtype=bool)
        mask[order[: min(min_points, score.shape[0])]] = True
    center = np.average(coords[mask], axis=0, weights=score[mask] + 1e-3)
    span = np.ptp(coords, axis=0)
    radius = max(float(np.max(span)) * 0.18, 1e-6)
    for scale in (1.0, 1.2, 1.45, 1.75, 2.1, 2.55, 3.0):
        half = radius * scale
        roi = (
            (coords[:, 0] >= center[0] - half)
            & (coords[:, 0] <= center[0] + half)
            & (coords[:, 1] >= center[1] - half)
            & (coords[:, 1] <= center[1] + half)
        )
        if roi.sum() >= min_points or scale == 2.2:
            return roi, (center[0] - half, center[0] + half, center[1] - half, center[1] + half)


def add_roi_box(ax, box, color="#111111"):
    import matplotlib.patches as patches

    x0, x1, y0, y1 = box
    outer = patches.Rectangle(
        (x0, y0),
        x1 - x0,
        y1 - y0,
        linewidth=2.8,
        edgecolor="#111111",
        facecolor="none",
        alpha=0.95,
        joinstyle="miter",
        zorder=20,
    )
    inner = patches.Rectangle(
        (x0, y0),
        x1 - x0,
        y1 - y0,
        linewidth=1.35,
        edgecolor="#ffffff",
        facecolor="none",
        alpha=0.95,
        joinstyle="miter",
        zorder=21,
    )
    ax.add_patch(outer)
    ax.add_patch(inner)


def raster_panel(ax, coords, values, title, cmap, vmin=0.0, vmax=1.0, bins=620, sigma=0.85, min_count=0.02):
    import matplotlib.pyplot as plt

    grid, extent = rasterize_values(coords, values, bins=bins, sigma=sigma, min_count=min_count)
    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad((1.0, 1.0, 1.0, 0.0))
    im = ax.imshow(
        np.ma.masked_invalid(grid),
        extent=[extent[0], extent[1], extent[2], extent[3]],
        origin="lower",
        cmap=cmap_obj,
        vmin=vmin,
        vmax=vmax,
        interpolation="bilinear",
    )
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[3], extent[2])
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(title, fontsize=10.5, pad=7)
    return im


def save_full_slice(task_name, feature, rank, metric_row, coords, true_v, pred_v, out_dir, roi_box):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    true_d = robust_scale01(true_v)
    pred_d = robust_scale01(pred_v)
    err = np.abs(true_d - pred_d)

    fig, axes = plt.subplots(1, 3, figsize=(9.8, 3.35), constrained_layout=True)
    sc0 = raster_panel(axes[0], coords, true_d, "Measured", "inferno", 0.0, 1.0)
    raster_panel(axes[1], coords, pred_d, "Predicted", "inferno", 0.0, 1.0)
    sc2 = raster_panel(axes[2], coords, err, "Difference", "viridis", 0.0, 0.75)
    for ax in axes[:2]:
        add_roi_box(ax, roi_box, color="#171717")

    cb0 = fig.colorbar(sc0, ax=axes[:2], shrink=0.78, pad=0.012, aspect=24)
    cb0.set_label("Relative expression", fontsize=8)
    cb0.ax.tick_params(labelsize=7, length=2)
    cb2 = fig.colorbar(sc2, ax=axes[2], shrink=0.78, pad=0.012, aspect=24)
    cb2.set_label("Absolute difference", fontsize=8)
    cb2.ax.tick_params(labelsize=7, length=2)
    pcc = float(metric_row.get("PCC", np.nan))
    rmse = float(metric_row.get("RMSE", np.nan))
    fig.suptitle(
        f"{TASK_TITLES.get(task_name, task_name)} | {feature} | PCC={pcc:.3f}, RMSE={rmse:.3f}",
        fontsize=11.2,
        fontweight="semibold",
    )
    path = out_dir / f"rank{rank:02d}_{sanitize_name(feature)}_full_slice_pub.png"
    fig.savefig(path, dpi=450, bbox_inches="tight")
    plt.close(fig)
    return path


def save_roi(task_name, feature, rank, metric_row, coords, true_v, pred_v, out_dir, roi_mask, roi_box):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    true_d = robust_scale01(true_v)
    pred_d = robust_scale01(pred_v)
    err = np.abs(true_d - pred_d)
    roi_coords = coords[roi_mask]
    roi_true = true_d[roi_mask]
    roi_pred = pred_d[roi_mask]
    roi_err = err[roi_mask]

    fig, axes = plt.subplots(1, 3, figsize=(8.9, 3.05), constrained_layout=True)
    sc0 = raster_panel(axes[0], roi_coords, roi_true, "Measured ROI", "inferno", 0.0, 1.0, bins=190, sigma=1.15, min_count=0.006)
    raster_panel(axes[1], roi_coords, roi_pred, "Predicted ROI", "inferno", 0.0, 1.0, bins=190, sigma=1.15, min_count=0.006)
    sc2 = raster_panel(axes[2], roi_coords, roi_err, "Difference ROI", "viridis", 0.0, 0.75, bins=190, sigma=1.15, min_count=0.006)
    for ax in axes:
        x0, x1, y0, y1 = roi_box
        ax.set_xlim(x0, x1)
        ax.set_ylim(y1, y0)

    cb0 = fig.colorbar(sc0, ax=axes[:2], shrink=0.78, pad=0.012, aspect=24)
    cb0.set_label("Relative expression", fontsize=8)
    cb0.ax.tick_params(labelsize=7, length=2)
    cb2 = fig.colorbar(sc2, ax=axes[2], shrink=0.78, pad=0.012, aspect=24)
    cb2.set_label("Absolute difference", fontsize=8)
    cb2.ax.tick_params(labelsize=7, length=2)
    pcc = float(metric_row.get("PCC", np.nan))
    fig.suptitle(
        f"{TASK_TITLES.get(task_name, task_name)} | {feature} high-expression ROI | PCC={pcc:.3f}",
        fontsize=11.2,
        fontweight="semibold",
    )
    path = out_dir / f"rank{rank:02d}_{sanitize_name(feature)}_roi_zoom_pub.png"
    fig.savefig(path, dpi=450, bbox_inches="tight")
    plt.close(fig)
    return path


def save_composite(task_name, feature, rank, metric_row, coords, true_v, pred_v, out_dir, roi_mask, roi_box):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    true_d = robust_scale01(true_v)
    pred_d = robust_scale01(pred_v)
    err = np.abs(true_d - pred_d)
    roi_coords = coords[roi_mask]

    fig, axes = plt.subplots(2, 3, figsize=(9.9, 6.1), constrained_layout=True)
    im_expr = raster_panel(axes[0, 0], coords, true_d, "Measured", "inferno", 0.0, 1.0, bins=620, sigma=0.85)
    raster_panel(axes[0, 1], coords, pred_d, "Predicted", "inferno", 0.0, 1.0, bins=620, sigma=0.85)
    im_err = raster_panel(axes[0, 2], coords, err, "Difference", "viridis", 0.0, 0.75, bins=620, sigma=0.85)
    for ax in axes[0, :2]:
        add_roi_box(ax, roi_box, color="#111111")

    raster_panel(axes[1, 0], roi_coords, true_d[roi_mask], "Measured ROI", "inferno", 0.0, 1.0, bins=190, sigma=1.15, min_count=0.006)
    raster_panel(axes[1, 1], roi_coords, pred_d[roi_mask], "Predicted ROI", "inferno", 0.0, 1.0, bins=190, sigma=1.15, min_count=0.006)
    raster_panel(axes[1, 2], roi_coords, err[roi_mask], "Difference ROI", "viridis", 0.0, 0.75, bins=190, sigma=1.15, min_count=0.006)
    x0, x1, y0, y1 = roi_box
    for ax in axes[1, :]:
        ax.set_xlim(x0, x1)
        ax.set_ylim(y1, y0)

    cb_expr = fig.colorbar(im_expr, ax=[axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]], shrink=0.72, pad=0.012, aspect=28)
    cb_expr.set_label("Relative expression", fontsize=8)
    cb_expr.ax.tick_params(labelsize=7, length=2)
    cb_err = fig.colorbar(im_err, ax=[axes[0, 2], axes[1, 2]], shrink=0.72, pad=0.012, aspect=28)
    cb_err.set_label("Absolute difference", fontsize=8)
    cb_err.ax.tick_params(labelsize=7, length=2)

    pcc = float(metric_row.get("PCC", np.nan))
    rmse = float(metric_row.get("RMSE", np.nan))
    fig.suptitle(
        f"{TASK_TITLES.get(task_name, task_name)} | {feature} | PCC={pcc:.3f}, RMSE={rmse:.3f}",
        fontsize=11.4,
        fontweight="semibold",
    )
    path = out_dir / f"rank{rank:02d}_{sanitize_name(feature)}_composite_pub.png"
    fig.savefig(path, dpi=450, bbox_inches="tight")
    plt.close(fig)
    return path


def process_task(task_dir, out_root, top_n):
    rep_dir = next(iter(sorted(task_dir.glob("rep*_seed*"))), None)
    if rep_dir is None:
        raise FileNotFoundError(f"No rep*_seed* dir under {task_dir}")
    gt = np.load(rep_dir / "gt_full.npy")
    pred = np.load(rep_dir / "pred_full.npy")
    top = pd.read_csv(rep_dir / "top20_feature_metrics.csv")
    top = top.sort_values("PCC", ascending=False).head(top_n).copy()
    coords = load_coords(rep_dir, gt.shape[0])
    task_base = task_dir.name.replace("_rule_round", "").replace("_main", "")
    out_dir = out_root / task_base
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for _, row in top.iterrows():
        rank = int(row["rank"])
        j = int(row["feature_index"])
        feature = str(row["feature_name"])
        true_v = gt[:, j].astype(float)
        pred_v = pred[:, j].astype(float)
        roi_mask, roi_box = choose_roi(coords, true_v, pred_v)
        full_path = save_full_slice(task_base, feature, rank, row, coords, true_v, pred_v, out_dir, roi_box)
        roi_path = save_roi(task_base, feature, rank, row, coords, true_v, pred_v, out_dir, roi_mask, roi_box)
        composite_path = save_composite(task_base, feature, rank, row, coords, true_v, pred_v, out_dir, roi_mask, roi_box)
        rows.append(
            {
                "task": task_base,
                "rank": rank,
                "feature_index": j,
                "feature_name": feature,
                "PCC": float(row["PCC"]),
                "RMSE": float(row["RMSE"]),
                "MAE": float(row.get("MAE", np.nan)),
                "full_slice_figure": str(full_path),
                "roi_zoom_figure": str(roi_path),
                "composite_figure": str(composite_path),
                "roi_points": int(roi_mask.sum()),
            }
        )
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True, help="Path to a rule run directory containing detailed_analysis")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--top-n", type=int, default=5)
    args = parser.parse_args()

    run_root = Path(args.run_root)
    analysis = run_root / "detailed_analysis"
    if not analysis.exists():
        raise FileNotFoundError(f"Missing detailed_analysis: {analysis}")
    out_root = Path(args.out_dir) if args.out_dir else run_root / "paper_visualizations"
    out_root.mkdir(parents=True, exist_ok=True)

    task_dirs = [
        p for p in sorted(analysis.iterdir())
        if p.is_dir()
        and p.name.startswith("task")
        and (p.name.endswith("_rule_round") or p.name.endswith("_main"))
    ]
    if not task_dirs:
        raise FileNotFoundError(f"No task*_rule_round directories under {analysis}")

    all_rows = []
    for task_dir in task_dirs:
        print(f">>> visualize {task_dir.name}", flush=True)
        all_rows.extend(process_task(task_dir, out_root, args.top_n))
    manifest = pd.DataFrame(all_rows)
    manifest_path = out_root / "visualization_manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    print(f">>> saved manifest: {manifest_path}", flush=True)
    print(f">>> saved figures under: {out_root}", flush=True)


if __name__ == "__main__":
    main()
