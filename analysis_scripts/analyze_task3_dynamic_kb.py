#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import spearmanr


def columnwise_pcc(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a = a - a.mean(axis=0, keepdims=True)
    b = b - b.mean(axis=0, keepdims=True)
    denom = np.sqrt((a * a).sum(axis=0) * (b * b).sum(axis=0))
    return np.divide(
        (a * b).sum(axis=0),
        denom,
        out=np.full(a.shape[1], np.nan),
        where=denom > 1e-12,
    )


def finite_spearman(x, y):
    x = np.asarray(x)
    y = np.asarray(y)
    keep = np.isfinite(x) & np.isfinite(y)
    if keep.sum() < 3:
        return {"rho": None, "pvalue": None, "n": int(keep.sum())}
    result = spearmanr(x[keep], y[keep])
    return {"rho": float(result.statistic), "pvalue": float(result.pvalue), "n": int(keep.sum())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with np.load(args.npz, allow_pickle=False) as data:
        gt = data["gt"].astype(np.float32)
        pred = data["pred_full"].astype(np.float32)
        pred_main = data["pred_main"].astype(np.float32)
        global_prior = data["kb_global_prior"].astype(np.float32)
        local_prior = data["kb_local_prior"].astype(np.float32)
        mask = data["kb_local_mask"].astype(np.float32)
        refine_gate = data["refine_gate"].astype(np.float32)
        coords = data["coords"].astype(np.float32)
        names = data["feature_names"].astype(str)
        fusion_keys = [
            "fusion_gate_he_shared",
            "fusion_gate_he_local",
            "fusion_gate_obs",
            "fusion_gate_kb",
        ]
        fusion = {key: data[key].astype(np.float32) for key in fusion_keys}
        scalar_gate_stats = {
            key: float(data[key])
            for key in data.files
            if key.startswith("fusion_gate_") and data[key].shape == ()
        }

    pcc_full = columnwise_pcc(gt, pred)
    pcc_main = columnwise_pcc(gt, pred_main)
    pcc_global = columnwise_pcc(gt, global_prior)
    pcc_local = columnwise_pcc(gt, local_prior)
    mae_full = np.mean(np.abs(gt - pred), axis=0)
    mae_main = np.mean(np.abs(gt - pred_main), axis=0)
    residual_mae_gain = mae_main - mae_full
    mask_mean = mask.mean(axis=0)
    mask_std = mask.std(axis=0)
    mask_cv = mask_std / np.maximum(np.abs(mask_mean), 1e-8)
    refine_mean = refine_gate.mean(axis=0)
    prior_localization_gain = pcc_local - pcc_global

    per_gene = pd.DataFrame(
        {
            "feature_index": np.arange(len(names)),
            "feature_name": names,
            "pcc_full": pcc_full,
            "pcc_main": pcc_main,
            "pcc_residual_gain": pcc_full - pcc_main,
            "pcc_global_prior": pcc_global,
            "pcc_local_prior": pcc_local,
            "pcc_prior_localization_gain": prior_localization_gain,
            "mae_full": mae_full,
            "mae_main": mae_main,
            "mae_residual_gain": residual_mae_gain,
            "mask_mean": mask_mean,
            "mask_std": mask_std,
            "mask_cv": mask_cv,
            "refine_gate_mean": refine_mean,
        }
    ).sort_values("pcc_full", ascending=False)
    per_gene.to_csv(out_dir / "per_gene_dynamic_kb_metrics.csv", index=False)

    abs_error_full = np.abs(gt - pred)
    abs_error_main = np.abs(gt - pred_main)
    spot_gain = (abs_error_main - abs_error_full).mean(axis=1)
    spot_mask = mask.mean(axis=1)
    spot_gate = {key: value.mean(axis=1) for key, value in fusion.items()}
    per_spot = pd.DataFrame(
        {
            "spot_index": np.arange(gt.shape[0]),
            "x": coords[:, 0],
            "y": coords[:, 1],
            "mask_mean": spot_mask,
            "residual_mae_gain": spot_gain,
            **{key: value for key, value in spot_gate.items()},
        }
    )
    per_spot.to_csv(out_dir / "per_spot_dynamic_kb_metrics.csv", index=False)

    correlations = {
        "mask_mean_vs_residual_mae_gain_per_gene": finite_spearman(mask_mean, residual_mae_gain),
        "mask_cv_vs_pcc_residual_gain_per_gene": finite_spearman(mask_cv, pcc_full - pcc_main),
        "global_prior_pcc_vs_model_pcc": finite_spearman(pcc_global, pcc_full),
        "local_prior_pcc_vs_model_pcc": finite_spearman(pcc_local, pcc_full),
        "prior_localization_gain_vs_model_pcc": finite_spearman(prior_localization_gain, pcc_full),
        "spot_mask_vs_residual_mae_gain": finite_spearman(spot_mask, spot_gain),
        "spot_kb_gate_vs_residual_mae_gain": finite_spearman(
            spot_gate["fusion_gate_kb"], spot_gain
        ),
    }
    summary = {
        "num_spots": int(gt.shape[0]),
        "num_features": int(gt.shape[1]),
        "mean_feature_pcc_full": float(np.nanmean(pcc_full)),
        "median_feature_pcc_full": float(np.nanmedian(pcc_full)),
        "mean_feature_pcc_main": float(np.nanmean(pcc_main)),
        "mean_pcc_residual_gain": float(np.nanmean(pcc_full - pcc_main)),
        "features_residual_improved_pcc": int(np.nansum((pcc_full - pcc_main) > 0)),
        "features_residual_improved_mae": int(np.nansum(residual_mae_gain > 0)),
        "mean_global_prior_pcc": float(np.nanmean(pcc_global)),
        "mean_local_prior_pcc": float(np.nanmean(pcc_local)),
        "features_local_prior_better_than_global": int(np.nansum(prior_localization_gain > 0)),
        "mask_mean": float(mask.mean()),
        "mask_std": float(mask.std()),
        "mask_min": float(mask.min()),
        "mask_max": float(mask.max()),
        "refine_gate_mean": float(refine_gate.mean()),
        "fusion_gate_means": {key: float(value.mean()) for key, value in fusion.items()},
        "fusion_gate_stds": {key: float(value.std()) for key, value in fusion.items()},
        "saved_scalar_gate_stats": scalar_gate_stats,
        "correlations": correlations,
    }
    (out_dir / "dynamic_kb_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
        }
    )
    pdf_path = out_dir / "task3_dynamic_kb_mechanism_analysis.pdf"
    with PdfPages(pdf_path) as pdf:
        fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
        labels = ["H&E shared", "H&E local", "Observed protein", "Knowledge"]
        values = [fusion[key].ravel()[::200] for key in fusion_keys]
        violin = axes[0, 0].violinplot(values, showmeans=True, showextrema=False)
        for body, color in zip(
            violin["bodies"], ["#4477AA", "#66CCEE", "#EE8866", "#228833"]
        ):
            body.set_facecolor(color)
            body.set_alpha(0.75)
        axes[0, 0].set_xticks(range(1, 5), labels, rotation=20, ha="right")
        axes[0, 0].set_ylabel("Gate value")
        axes[0, 0].set_title("Dynamic fusion gate distributions", loc="left", fontweight="bold")

        scatter = axes[0, 1].scatter(
            pcc_global,
            pcc_full,
            c=mask_cv,
            s=18,
            alpha=0.7,
            cmap="viridis",
            edgecolors="none",
        )
        axes[0, 1].axhline(0, color="#999999", lw=0.7)
        axes[0, 1].axvline(0, color="#999999", lw=0.7)
        axes[0, 1].set_xlabel("Global prior PCC")
        axes[0, 1].set_ylabel("Prediction PCC")
        axes[0, 1].set_title("Prior fidelity and final prediction", loc="left", fontweight="bold")
        fig.colorbar(scatter, ax=axes[0, 1], label="Mask spatial CV", pad=0.03)

        axes[1, 0].scatter(
            mask_cv,
            pcc_full - pcc_main,
            c=pcc_local,
            cmap="coolwarm",
            s=18,
            alpha=0.7,
            edgecolors="none",
        )
        axes[1, 0].axhline(0, color="#555555", lw=0.8)
        axes[1, 0].set_xlabel("Dynamic mask spatial CV")
        axes[1, 0].set_ylabel("PCC gain from residual branch")
        axes[1, 0].set_title("Does selective retrieval support correction?", loc="left", fontweight="bold")

        top = per_gene.sort_values("pcc_residual_gain", ascending=False).head(15)
        colors = ["#228833" if value > 0 else "#CC3311" for value in top["pcc_residual_gain"]]
        axes[1, 1].barh(top["feature_name"][::-1], top["pcc_residual_gain"][::-1], color=colors[::-1])
        axes[1, 1].axvline(0, color="#555555", lw=0.8)
        axes[1, 1].set_xlabel("PCC(full) - PCC(main)")
        axes[1, 1].set_title("Genes most helped by residual correction", loc="left", fontweight="bold")
        pdf.savefig(fig, bbox_inches="tight")
        fig.savefig(out_dir / "dynamic_kb_mechanism_summary.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

        eligible = per_gene.replace([np.inf, -np.inf], np.nan).dropna(
            subset=["pcc_full", "pcc_local_prior", "pcc_residual_gain"]
        )
        eligible = eligible[
            (eligible["pcc_full"] > 0)
            & (eligible["pcc_local_prior"] > 0)
            & (eligible["pcc_residual_gain"] > 0)
        ]
        case_indices = eligible.sort_values(
            ["pcc_residual_gain", "pcc_local_prior"], ascending=False
        ).head(3)["feature_index"].astype(int).tolist()
        if not case_indices:
            case_indices = per_gene.head(3)["feature_index"].astype(int).tolist()
        fig, axes = plt.subplots(len(case_indices), 5, figsize=(14, 3.1 * len(case_indices)), constrained_layout=True)
        axes = np.atleast_2d(axes)
        for row, idx in enumerate(case_indices):
            panels = [
                (gt[:, idx], "Measured"),
                (global_prior[:, idx], "Global prior"),
                (mask[:, idx], "Dynamic mask"),
                (local_prior[:, idx], "Localized prior"),
                (pred[:, idx], "Prediction"),
            ]
            for col, (values_panel, title) in enumerate(panels):
                image = axes[row, col].scatter(
                    coords[:, 0],
                    coords[:, 1],
                    c=values_panel,
                    s=2,
                    cmap="coolwarm",
                    rasterized=True,
                )
                axes[row, col].invert_yaxis()
                axes[row, col].axis("off")
                axes[row, col].set_title(f"{names[idx]} | {title}", fontsize=9)
                fig.colorbar(image, ax=axes[row, col], fraction=0.035, pad=0.03)
        fig.suptitle("Examples of spatial knowledge localization", fontsize=14, fontweight="bold")
        pdf.savefig(fig, bbox_inches="tight")
        fig.savefig(out_dir / "spatial_knowledge_localization_cases.png", dpi=250, bbox_inches="tight")
        plt.close(fig)

    print(json.dumps({"summary": summary, "pdf": str(pdf_path)}, indent=2))


if __name__ == "__main__":
    main()
