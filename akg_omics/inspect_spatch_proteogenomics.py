#!/usr/bin/env python
"""
Inspect SPATCH proteogenomics data layout and derive cell-level morphology summaries.

This script is designed for the server-side dataset layout:

    data/spatch/
      |- gene/
      |- protein/

It reads:
  - transcriptome/adata.h5ad
  - proteome/adata_codex.h5ad
  - transcripts/transcripts.parquet
  - segmentation_mask/cell_boundaries.csv
  - segmentation_mask/nucleus_boundaries.csv
  - HE / DAPI / CODEX / morphology TIFF metadata

Outputs:
  - JSON summaries for each split
  - Optional morphology feature CSVs
  - A combined markdown report for quick inspection
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    import anndata as ad
except Exception as exc:  # pragma: no cover
    raise RuntimeError("anndata is required to run this script") from exc

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None


IMAGE_NAMES = ["HE.tif", "DAPI.tif", "CODEX.tif"]
MORPH_CANDIDATE_ID_COLS = [
    "cell_id",
    "cellid",
    "label_id",
    "labelid",
    "object_id",
    "objectid",
    "id",
]
MORPH_CANDIDATE_X_COLS = ["x", "vertex_x", "coord_x", "xcoord", "pos_x", "cx"]
MORPH_CANDIDATE_Y_COLS = ["y", "vertex_y", "coord_y", "ycoord", "pos_y", "cy"]
TRANSCRIPT_CANDIDATE_GENE_COLS = ["feature_name", "gene", "gene_name", "target", "symbol"]
TRANSCRIPT_CANDIDATE_CELL_COLS = ["cell_id", "cell", "cell_label", "cellid", "segmentation_id"]


@dataclass
class PolygonFeatures:
    instance_id: str
    area: float
    perimeter: float
    bbox_width: float
    bbox_height: float
    circularity: float
    centroid_x: float
    centroid_y: float
    num_vertices: int


def _find_existing(path: Path, rel: str) -> Optional[Path]:
    candidate = path / rel
    return candidate if candidate.exists() else None


def _progress(iterable, desc: str, total: Optional[int] = None, leave: bool = True):
    if tqdm is None:
        return iterable
    return tqdm(iterable, desc=desc, total=total, leave=leave)


def _log_step(split_name: str, message: str) -> None:
    print(f"[{split_name}] {message}", flush=True)


def _safe_json_dump(payload: Dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _normalize_columns(df: pd.DataFrame) -> Dict[str, str]:
    return {str(c).strip().lower().replace(" ", "").replace("_", ""): c for c in df.columns}


def _resolve_column(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    norm = _normalize_columns(df)
    for cand in candidates:
        key = str(cand).strip().lower().replace(" ", "").replace("_", "")
        if key in norm:
            return norm[key]
    return None


def _read_h5ad_summary(path: Path) -> Dict:
    print(f"[read_h5ad] {path}", flush=True)
    adata = ad.read_h5ad(path)
    obs_cols = list(map(str, adata.obs.columns))
    var_cols = list(map(str, adata.var.columns))
    obsm_keys = list(map(str, adata.obsm.keys()))
    uns_keys = list(map(str, adata.uns.keys()))

    spatial_stats = {}
    if "spatial" in adata.obsm:
        coords = np.asarray(adata.obsm["spatial"])
        if coords.ndim == 2 and coords.shape[1] >= 2:
            spatial_stats = {
                "min_x": float(np.nanmin(coords[:, 0])),
                "max_x": float(np.nanmax(coords[:, 0])),
                "min_y": float(np.nanmin(coords[:, 1])),
                "max_y": float(np.nanmax(coords[:, 1])),
            }

    he_stats = {}
    if "he" in adata.obsm:
        he = np.asarray(adata.obsm["he"])
        he_stats = {
            "shape": list(he.shape),
            "dtype": str(he.dtype),
        }

    return {
        "path": str(path),
        "n_obs": int(adata.n_obs),
        "n_vars": int(adata.n_vars),
        "obs_columns": obs_cols,
        "var_columns": var_cols,
        "obsm_keys": obsm_keys,
        "uns_keys": uns_keys,
        "obs_name_examples": list(map(str, adata.obs_names[:5])),
        "var_name_examples": list(map(str, adata.var_names[:10])),
        "spatial_stats": spatial_stats,
        "he_stats": he_stats,
    }


def _read_image_summary(path: Path) -> Dict:
    try:
        import tifffile
    except Exception:
        tifffile = None

    summary = {"path": str(path), "size_bytes": int(path.stat().st_size)}
    print(f"[read_image] {path}", flush=True)
    if tifffile is None:
        return summary
    try:
        with tifffile.TiffFile(path) as tif:
            if tif.series:
                shape = list(tif.series[0].shape)
                dtype = str(tif.series[0].dtype)
                summary.update({"shape": shape, "dtype": dtype})
    except Exception as exc:
        summary["read_error"] = f"{type(exc).__name__}: {exc}"
    return summary


def _read_transcript_summary(path: Path, sample_n: int = 20000) -> Dict:
    print(f"[read_parquet] {path}", flush=True)
    df = pd.read_parquet(path)
    n_rows = int(len(df))
    gene_col = _resolve_column(df, TRANSCRIPT_CANDIDATE_GENE_COLS)
    cell_col = _resolve_column(df, TRANSCRIPT_CANDIDATE_CELL_COLS)
    x_col = _resolve_column(df, MORPH_CANDIDATE_X_COLS)
    y_col = _resolve_column(df, MORPH_CANDIDATE_Y_COLS)

    sample_df = df.head(min(sample_n, n_rows))
    payload = {
        "path": str(path),
        "n_rows": n_rows,
        "columns": list(map(str, df.columns)),
        "gene_col": gene_col,
        "cell_col": cell_col,
        "x_col": x_col,
        "y_col": y_col,
    }
    if gene_col is not None:
        payload["n_unique_genes_sample"] = int(sample_df[gene_col].astype(str).nunique())
    if cell_col is not None:
        payload["n_unique_cells_sample"] = int(sample_df[cell_col].astype(str).nunique())
    if x_col is not None and y_col is not None:
        xv = pd.to_numeric(sample_df[x_col], errors="coerce")
        yv = pd.to_numeric(sample_df[y_col], errors="coerce")
        payload["coord_sample_range"] = {
            "min_x": float(np.nanmin(xv)),
            "max_x": float(np.nanmax(xv)),
            "min_y": float(np.nanmin(yv)),
            "max_y": float(np.nanmax(yv)),
        }
    return payload


def _polygon_area_perimeter(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    if len(x) < 3:
        return 0.0, 0.0
    x2 = np.r_[x, x[0]]
    y2 = np.r_[y, y[0]]
    area = 0.5 * abs(np.dot(x2[:-1], y2[1:]) - np.dot(y2[:-1], x2[1:]))
    perimeter = float(np.sqrt(np.diff(x2) ** 2 + np.diff(y2) ** 2).sum())
    return float(area), perimeter


def _compute_polygon_features(instance_id: str, x: np.ndarray, y: np.ndarray) -> PolygonFeatures:
    area, perimeter = _polygon_area_perimeter(x, y)
    bbox_width = float(np.max(x) - np.min(x)) if len(x) else 0.0
    bbox_height = float(np.max(y) - np.min(y)) if len(y) else 0.0
    circularity = 0.0
    if perimeter > 1e-8:
        circularity = float(4.0 * math.pi * area / (perimeter * perimeter))
    return PolygonFeatures(
        instance_id=str(instance_id),
        area=area,
        perimeter=perimeter,
        bbox_width=bbox_width,
        bbox_height=bbox_height,
        circularity=circularity,
        centroid_x=float(np.mean(x)) if len(x) else 0.0,
        centroid_y=float(np.mean(y)) if len(y) else 0.0,
        num_vertices=int(len(x)),
    )


def _read_boundary_features(path: Path, max_rows: Optional[int] = None) -> Tuple[Dict, Optional[pd.DataFrame]]:
    print(f"[read_boundary] {path}", flush=True)
    df = pd.read_csv(path, nrows=max_rows)
    id_col = _resolve_column(df, MORPH_CANDIDATE_ID_COLS)
    x_col = _resolve_column(df, MORPH_CANDIDATE_X_COLS)
    y_col = _resolve_column(df, MORPH_CANDIDATE_Y_COLS)

    summary = {
        "path": str(path),
        "n_rows": int(len(df)),
        "columns": list(map(str, df.columns)),
        "id_col": id_col,
        "x_col": x_col,
        "y_col": y_col,
    }
    if id_col is None or x_col is None or y_col is None:
        summary["feature_extraction"] = "skipped_missing_required_columns"
        return summary, None

    use = df[[id_col, x_col, y_col]].copy()
    use[x_col] = pd.to_numeric(use[x_col], errors="coerce")
    use[y_col] = pd.to_numeric(use[y_col], errors="coerce")
    use = use.dropna()
    use[id_col] = use[id_col].astype(str)

    grouped = use.groupby(id_col, sort=False)
    n_groups = int(use[id_col].nunique())
    features: List[PolygonFeatures] = []
    for instance_id, sub in _progress(grouped, desc=f"polygon:{path.stem}", total=n_groups, leave=False):
        x = sub[x_col].to_numpy(dtype=np.float64)
        y = sub[y_col].to_numpy(dtype=np.float64)
        features.append(_compute_polygon_features(instance_id, x, y))

    feat_df = pd.DataFrame([vars(f) for f in features])
    summary.update(
        {
            "n_instances": int(len(feat_df)),
            "area_mean": float(feat_df["area"].mean()) if len(feat_df) else 0.0,
            "area_median": float(feat_df["area"].median()) if len(feat_df) else 0.0,
            "perimeter_mean": float(feat_df["perimeter"].mean()) if len(feat_df) else 0.0,
            "feature_extraction": "ok",
        }
    )
    return summary, feat_df


def _join_cell_nucleus_features(cell_df: Optional[pd.DataFrame], nucleus_df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    if cell_df is None or nucleus_df is None:
        return None
    c = cell_df.rename(
        columns={
            "area": "cell_area",
            "perimeter": "cell_perimeter",
            "bbox_width": "cell_bbox_width",
            "bbox_height": "cell_bbox_height",
            "circularity": "cell_circularity",
            "centroid_x": "cell_centroid_x",
            "centroid_y": "cell_centroid_y",
            "num_vertices": "cell_num_vertices",
        }
    )
    n = nucleus_df.rename(
        columns={
            "area": "nucleus_area",
            "perimeter": "nucleus_perimeter",
            "bbox_width": "nucleus_bbox_width",
            "bbox_height": "nucleus_bbox_height",
            "circularity": "nucleus_circularity",
            "centroid_x": "nucleus_centroid_x",
            "centroid_y": "nucleus_centroid_y",
            "num_vertices": "nucleus_num_vertices",
        }
    )
    merged = c.merge(n, on="instance_id", how="outer")
    if "cell_area" in merged.columns and "nucleus_area" in merged.columns:
        merged["nucleus_cell_area_ratio"] = merged["nucleus_area"] / merged["cell_area"].replace(0, np.nan)
    return merged


def inspect_split(split_dir: Path, out_dir: Path) -> Dict:
    split_name = split_dir.name
    summary: Dict[str, object] = {"split": split_name, "base_dir": str(split_dir)}

    transcriptome_path = _find_existing(split_dir, "transcriptome/adata.h5ad")
    proteome_path = _find_existing(split_dir, "proteome/adata_codex.h5ad")
    transcripts_path = _find_existing(split_dir, "transcripts/transcripts.parquet")
    cell_boundary_path = _find_existing(split_dir, "segmentation_mask/cell_boundaries.csv")
    nucleus_boundary_path = _find_existing(split_dir, "segmentation_mask/nucleus_boundaries.csv")

    cell_df = None
    nucleus_df = None

    step_items = [
        ("transcriptome_h5ad", transcriptome_path),
        ("proteome_h5ad", proteome_path),
        ("transcripts_parquet", transcripts_path),
        ("cell_boundaries", cell_boundary_path),
        ("nucleus_boundaries", nucleus_boundary_path),
    ] + [(f"image:{name}", _find_existing(split_dir, name)) for name in IMAGE_NAMES]

    summary["images"] = {}

    for step_name, step_path in _progress(step_items, desc=f"split:{split_name}", total=len(step_items), leave=True):
        if step_path is None:
            _log_step(split_name, f"skip {step_name} (missing)")
            continue

        _log_step(split_name, f"start {step_name}")
        if step_name == "transcriptome_h5ad":
            summary["transcriptome_h5ad"] = _read_h5ad_summary(step_path)
        elif step_name == "proteome_h5ad":
            summary["proteome_h5ad"] = _read_h5ad_summary(step_path)
        elif step_name == "transcripts_parquet":
            summary["transcripts_parquet"] = _read_transcript_summary(step_path)
        elif step_name == "cell_boundaries":
            cell_summary, cell_df = _read_boundary_features(step_path)
            summary["cell_boundaries"] = cell_summary
        elif step_name == "nucleus_boundaries":
            nucleus_summary, nucleus_df = _read_boundary_features(step_path)
            summary["nucleus_boundaries"] = nucleus_summary
        elif step_name.startswith("image:"):
            image_name = step_name.split(":", 1)[1]
            summary["images"][image_name] = _read_image_summary(step_path)
        _log_step(split_name, f"done {step_name}")

    morph_dir = out_dir / split_name
    morph_dir.mkdir(parents=True, exist_ok=True)

    if cell_df is not None:
        _log_step(split_name, "write cell morphology csv")
        cell_df.to_csv(morph_dir / "cell_morphology_features.csv", index=False)
    if nucleus_df is not None:
        _log_step(split_name, "write nucleus morphology csv")
        nucleus_df.to_csv(morph_dir / "nucleus_morphology_features.csv", index=False)
    merged_df = _join_cell_nucleus_features(cell_df, nucleus_df)
    if merged_df is not None:
        _log_step(split_name, "write joined cell+nucleus morphology csv")
        merged_df.to_csv(morph_dir / "cell_nucleus_joined_features.csv", index=False)
        summary["joined_morphology"] = {
            "n_instances": int(len(merged_df)),
            "nucleus_cell_area_ratio_mean": float(merged_df["nucleus_cell_area_ratio"].dropna().mean())
            if "nucleus_cell_area_ratio" in merged_df.columns and merged_df["nucleus_cell_area_ratio"].notna().any()
            else None,
            "output_csv": str(morph_dir / "cell_nucleus_joined_features.csv"),
        }

    if "transcriptome_h5ad" in summary and "proteome_h5ad" in summary:
        tx = summary["transcriptome_h5ad"]
        pr = summary["proteome_h5ad"]
        summary["pairwise_alignment"] = {
            "transcriptome_n_obs": tx["n_obs"],
            "proteome_n_obs": pr["n_obs"],
            "transcriptome_n_vars": tx["n_vars"],
            "proteome_n_vars": pr["n_vars"],
        }

    _log_step(split_name, "write summary json")
    _safe_json_dump(summary, out_dir / f"{split_name}_summary.json")
    _log_step(split_name, "inspection complete")
    return summary


def write_markdown_report(summaries: Iterable[Dict], out_path: Path) -> None:
    lines: List[str] = []
    lines.append("# SPATCH Proteogenomics Inspection")
    lines.append("")
    for payload in summaries:
        split = payload["split"]
        lines.append(f"## {split}")
        lines.append("")
        lines.append(f"- Base dir: `{payload['base_dir']}`")
        tx = payload.get("transcriptome_h5ad")
        if tx:
            lines.append(f"- Transcriptome h5ad: `{tx['n_obs']}` obs x `{tx['n_vars']}` vars")
        pr = payload.get("proteome_h5ad")
        if pr:
            lines.append(f"- Proteome h5ad: `{pr['n_obs']}` obs x `{pr['n_vars']}` vars")
        tr = payload.get("transcripts_parquet")
        if tr:
            lines.append(f"- Transcript parquet rows: `{tr['n_rows']}`")
        cb = payload.get("cell_boundaries")
        if cb:
            lines.append(f"- Cell boundaries: `{cb.get('n_instances', 'NA')}` instances")
        nb = payload.get("nucleus_boundaries")
        if nb:
            lines.append(f"- Nucleus boundaries: `{nb.get('n_instances', 'NA')}` instances")
        jm = payload.get("joined_morphology")
        if jm:
            lines.append(
                f"- Joined morphology: `{jm['n_instances']}` instances, "
                f"mean nucleus/cell area ratio = `{jm['nucleus_cell_area_ratio_mean']}`"
            )
        lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect SPATCH proteogenomics cell-level assets.")
    parser.add_argument(
        "--data-root",
        default="data/spatch",
        help="Root directory containing gene/ and protein/ subdirectories.",
    )
    parser.add_argument(
        "--out-dir",
        default="./spatch_inspection",
        help="Directory to save summaries and morphology feature CSVs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    split_dirs = [data_root / "gene", data_root / "protein"]
    missing = [str(p) for p in split_dirs if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing expected split directories: {missing}")

    summaries = [inspect_split(split_dir, out_dir) for split_dir in split_dirs]
    write_markdown_report(summaries, out_dir / "README.md")
    print(f"[OK] wrote inspection outputs to: {out_dir}")


if __name__ == "__main__":
    main()
