#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def pcc_columns(a, b, block=64):
    values = np.full(a.shape[1], np.nan, dtype=np.float64)
    for start in range(0, a.shape[1], block):
        stop = min(start + block, a.shape[1])
        x = a[:, start:stop].astype(np.float64)
        y = b[:, start:stop].astype(np.float64)
        x -= x.mean(axis=0, keepdims=True)
        y -= y.mean(axis=0, keepdims=True)
        denom = np.sqrt(np.sum(x * x, axis=0) * np.sum(y * y, axis=0))
        values[start:stop] = np.divide(
            np.sum(x * y, axis=0),
            denom,
            out=np.full(stop - start, np.nan),
            where=denom > 1e-12,
        )
    return values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", required=True)
    parser.add_argument("--no-kb", required=True)
    parser.add_argument("--no-dynamic", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    full_npz = np.load(args.full, allow_pickle=False)
    nokb_npz = np.load(args.no_kb, allow_pickle=False)
    nodyn_npz = np.load(args.no_dynamic, allow_pickle=False)

    names = full_npz["feature_names"].astype(str)
    gt = full_npz["gt"].astype(np.float32)
    pred_full = full_npz["pred_full"].astype(np.float32)
    pred_main = full_npz["pred_main"].astype(np.float32)
    global_prior = full_npz["kb_global_prior"].astype(np.float32)
    local_prior = full_npz["kb_local_prior"].astype(np.float32)
    mask = full_npz["kb_local_mask"].astype(np.float32)
    refine_gate = full_npz["refine_gate"].astype(np.float32)
    pred_nokb = nokb_npz["pred_full"].astype(np.float32)
    pred_nodyn = nodyn_npz["pred_full"].astype(np.float32)

    if not np.array_equal(names, nokb_npz["feature_names"].astype(str)):
        raise ValueError("Full and No-KB feature order differs")
    if not np.array_equal(names, nodyn_npz["feature_names"].astype(str)):
        raise ValueError("Full and no-dynamic feature order differs")

    metrics = {
        "pcc_full": pcc_columns(gt, pred_full),
        "pcc_no_kb": pcc_columns(gt, pred_nokb),
        "pcc_no_dynamic": pcc_columns(gt, pred_nodyn),
        "pcc_main": pcc_columns(gt, pred_main),
        "pcc_global_prior": pcc_columns(gt, global_prior),
        "pcc_local_prior": pcc_columns(gt, local_prior),
    }
    frame = pd.DataFrame({"feature_index": np.arange(len(names)), "feature_name": names, **metrics})
    frame["gain_vs_no_kb"] = frame["pcc_full"] - frame["pcc_no_kb"]
    frame["gain_vs_no_dynamic"] = frame["pcc_full"] - frame["pcc_no_dynamic"]
    frame["residual_gain"] = frame["pcc_full"] - frame["pcc_main"]
    frame["prior_localization_gain"] = frame["pcc_local_prior"] - frame["pcc_global_prior"]
    frame["mask_mean"] = mask.mean(axis=0)
    frame["mask_std"] = mask.std(axis=0)
    frame["mask_cv"] = frame["mask_std"] / np.maximum(np.abs(frame["mask_mean"]), 1e-8)
    frame["refine_gate_mean"] = refine_gate.mean(axis=0)

    valid = frame.replace([np.inf, -np.inf], np.nan).dropna(
        subset=[
            "pcc_full",
            "pcc_no_kb",
            "pcc_no_dynamic",
            "pcc_local_prior",
            "pcc_global_prior",
        ]
    )
    eligible = valid[
        (valid["gain_vs_no_kb"] > 0)
        & (valid["gain_vs_no_dynamic"] > 0)
        & (valid["prior_localization_gain"] > 0)
        & (valid["pcc_local_prior"] > 0)
        & (valid["pcc_full"] > 0.1)
    ].copy()
    for column in [
        "gain_vs_no_kb",
        "gain_vs_no_dynamic",
        "prior_localization_gain",
        "pcc_local_prior",
        "pcc_full",
    ]:
        eligible[f"rank_{column}"] = eligible[column].rank(pct=True)
    eligible["case_score"] = (
        0.30 * eligible["rank_gain_vs_no_kb"]
        + 0.25 * eligible["rank_gain_vs_no_dynamic"]
        + 0.20 * eligible["rank_prior_localization_gain"]
        + 0.15 * eligible["rank_pcc_local_prior"]
        + 0.10 * eligible["rank_pcc_full"]
    )
    eligible = eligible.sort_values("case_score", ascending=False)

    frame.sort_values("pcc_full", ascending=False).to_csv(
        output / "all_4806_gene_kb_effect_metrics.csv", index=False
    )
    eligible.to_csv(output / "eligible_case_genes.csv", index=False)
    top = eligible.head(30)
    top.to_csv(output / "top30_case_genes.csv", index=False)
    summary = {
        "num_features": int(len(frame)),
        "num_eligible": int(len(eligible)),
        "full_better_than_no_kb": int((valid["gain_vs_no_kb"] > 0).sum()),
        "full_better_than_no_dynamic": int((valid["gain_vs_no_dynamic"] > 0).sum()),
        "localized_prior_better_than_global": int((valid["prior_localization_gain"] > 0).sum()),
        "top_candidates": top[
            [
                "feature_name",
                "case_score",
                "pcc_full",
                "pcc_no_kb",
                "pcc_no_dynamic",
                "pcc_global_prior",
                "pcc_local_prior",
                "gain_vs_no_kb",
                "gain_vs_no_dynamic",
                "prior_localization_gain",
                "mask_mean",
                "mask_cv",
            ]
        ].to_dict(orient="records"),
    }
    (output / "case_screen_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
