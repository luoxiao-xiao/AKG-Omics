#!/usr/bin/env python3
import argparse
import csv
import json
import math
import os
import re
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


TASK_RE = re.compile(r"(task[1-4]_[a-z_]+?)(?:_rep(\d+)_seed(\d+))?$")
SOURCE_ORDER = [
    "hgnc",
    "uniprot",
    "reactome",
    "kegg",
    "cellmarker",
    "proteinatlas",
    "dorothea",
    "omnipath",
    "corum",
    "string",
    "disgenet",
    "opentargets",
]
TASK_ORDER = [
    "task1_he_to_gene",
    "task2_he_to_protein",
    "task3_he_protein_to_gene",
    "task4_he_gene_to_protein",
]


def safe_get(obj, *keys, default=None):
    cur = obj
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def parse_task(data, path):
    raw = (
        safe_get(data, "task_spec", "task_id")
        or data.get("task_id")
        or Path(path).stem.replace("kb_orchestration_", "")
    )
    match = TASK_RE.match(raw)
    if match:
        task, rep, seed = match.groups()
    else:
        task, rep, seed = raw, None, None
    if seed is None:
        name_match = re.search(r"_rep(\d+)_seed(\d+)", Path(path).stem)
        if name_match:
            rep, seed = name_match.groups()
    return task, int(rep) if rep else None, int(seed) if seed else None


def extract_llm_proposal(data):
    candidates = [
        safe_get(data, "selection", "parsed_response"),
        safe_get(data, "selection", "react_trace", "parsed_response"),
        safe_get(data, "agent_selection", "parsed_response"),
    ]
    for candidate in candidates:
        if isinstance(candidate, dict):
            values = candidate.get("selected_source_ids")
            if isinstance(values, list):
                return values

    texts = [
        safe_get(data, "selection", "message_content"),
        safe_get(data, "selection", "react_trace", "message_content"),
        safe_get(data, "agent_selection", "message_content"),
    ]
    for text in texts:
        if not isinstance(text, str):
            continue
        match = re.search(r'"selected_source_ids"\s*:\s*\[([^\]]*)\]', text)
        if match:
            return re.findall(r'"([^"]+)"', match.group(1))
    return []


def extract_parse_error(data):
    stack = [data]
    errors = []
    while stack:
        value = stack.pop()
        if isinstance(value, dict):
            for key, item in value.items():
                if key == "parse_error" and item:
                    errors.append(str(item))
                elif isinstance(item, (dict, list)):
                    stack.append(item)
        elif isinstance(value, list):
            stack.extend(value)
    return "; ".join(sorted(set(errors)))


def extract_selected(data, mode):
    if mode == "agent":
        selected = safe_get(data, "agent_weight_application", "selected_sources", default=[])
        if selected:
            return selected
    return (
        data.get("final_selected_sources")
        or safe_get(data, "selection", "selected_source_ids", default=[])
        or safe_get(data, "selection", "selected_sources", default=[])
        or []
    )


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_metrics(path, mode):
    rows = []
    with open(path, newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            row["mode"] = mode
            for key, value in list(row.items()):
                if key.endswith("_mean") or key.endswith("_std"):
                    try:
                        row[key] = float(value)
                    except (TypeError, ValueError):
                        pass
            rows.append(row)
    return rows


def matrix_from_records(records, row_keys, col_keys, value_fn):
    matrix = []
    for row_key in row_keys:
        row = []
        for col_key in col_keys:
            matching = [r for r in records if r["task"] == row_key]
            row.append(value_fn(matching, col_key))
        matrix.append(row)
    return matrix


def draw_heatmap(ax, matrix, rows, cols, title, cmap, value_format=".2f"):
    image = ax.imshow(matrix, aspect="auto", vmin=0, vmax=1, cmap=cmap)
    ax.set_xticks(range(len(cols)), cols, rotation=38, ha="right")
    ax.set_yticks(range(len(rows)), [x.replace("task", "Task ") for x in rows])
    ax.set_title(title, loc="left", fontweight="bold")
    for i, row in enumerate(matrix):
        for j, value in enumerate(row):
            text = format(value, value_format)
            color = "white" if value > 0.62 else "#222222"
            ax.text(j, i, text, ha="center", va="center", fontsize=7, color=color)
    return image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent-dir", required=True)
    parser.add_argument("--rule-dir", required=True)
    parser.add_argument("--agent-metrics", required=True)
    parser.add_argument("--rule-metrics", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    records = []
    coverage_rows = []
    builder_rows = []
    weight_rows = []

    for mode, directory in [("agent", args.agent_dir), ("rule", args.rule_dir)]:
        files = sorted(Path(directory).glob("kb_orchestration_task*.json"))
        for path in files:
            with path.open(encoding="utf-8") as handle:
                data = json.load(handle)
            task, rep, seed = parse_task(data, path)
            selected = [str(x).lower() for x in extract_selected(data, mode)]
            proposed = [str(x).lower() for x in extract_llm_proposal(data)]
            record = {
                "mode": mode,
                "task": task,
                "repeat": rep,
                "seed": seed,
                "target_modality": safe_get(data, "task_spec", "target_modality")
                or data.get("target_modality"),
                "required_relations": "|".join(
                    safe_get(data, "task_spec", "required_relations", default=[])
                ),
                "proposed_sources": "|".join(proposed),
                "effective_sources": "|".join(selected),
                "num_proposed": len(proposed),
                "num_effective": len(selected),
                "parse_error": extract_parse_error(data),
                "fallback_used": bool(data.get("fallback_used"))
                or bool(extract_parse_error(data)),
                "validation_ok": safe_get(data, "validation", "ok"),
                "source_file": str(path),
            }
            for source in SOURCE_ORDER:
                record[f"selected_{source}"] = int(source in selected)
                record[f"proposed_{source}"] = int(source in proposed)
            records.append(record)

            profile = data.get("data_profile", {})
            for source, source_data in profile.get("source_coverage", {}).items():
                base = {
                    "mode": mode,
                    "task": task,
                    "repeat": rep,
                    "seed": seed,
                    "source": source,
                    "modality": "overall",
                    "coverage": source_data.get("overall_coverage"),
                    "covered_count": source_data.get("covered_count"),
                    "total_count": source_data.get("total_count"),
                    "selected": int(source.lower() in selected),
                }
                coverage_rows.append(base)
                for modality, modality_data in source_data.get("per_modality", {}).items():
                    coverage_rows.append(
                        {
                            **base,
                            "modality": modality,
                            "coverage": modality_data.get("coverage"),
                            "covered_count": modality_data.get("covered_count"),
                            "total_count": modality_data.get("total_count"),
                        }
                    )

            stats = data.get("builder_stats") or safe_get(
                data, "validation", "summary", default={}
            )
            if stats:
                row = {"mode": mode, "task": task, "repeat": rep, "seed": seed}
                row.update(stats)
                builder_rows.append(row)

            application = data.get("agent_weight_application", {}).get("applied", {})
            for parameter, detail in application.items():
                weight_rows.append(
                    {
                        "mode": mode,
                        "task": task,
                        "repeat": rep,
                        "seed": seed,
                        "parameter": parameter,
                        "source_group": detail.get("source_group"),
                        "base": detail.get("base"),
                        "scale": detail.get("scale"),
                        "mode_scale": detail.get("mode_scale"),
                        "effective": detail.get("effective"),
                        "absolute_change": (
                            detail.get("effective") - detail.get("base")
                            if isinstance(detail.get("effective"), (int, float))
                            and isinstance(detail.get("base"), (int, float))
                            else None
                        ),
                    }
                )

    metrics = read_metrics(args.agent_metrics, "agent") + read_metrics(
        args.rule_metrics, "rule"
    )
    by_task_metric = defaultdict(dict)
    for row in metrics:
        by_task_metric[row["task"]][row["mode"]] = row
    metric_comparison = []
    for task in TASK_ORDER:
        a = by_task_metric.get(task, {}).get("agent", {})
        r = by_task_metric.get(task, {}).get("rule", {})
        if not a or not r:
            continue
        row = {"task": task}
        for metric in ["PCC", "SSIM", "CMD", "RMSE"]:
            row[f"agent_{metric}"] = a.get(f"{metric}_mean")
            row[f"rule_{metric}"] = r.get(f"{metric}_mean")
            if isinstance(row[f"agent_{metric}"], (int, float)) and isinstance(
                row[f"rule_{metric}"], (int, float)
            ):
                row[f"agent_minus_rule_{metric}"] = (
                    row[f"agent_{metric}"] - row[f"rule_{metric}"]
                )
        metric_comparison.append(row)

    stability_rows = []
    for mode in ["agent", "rule"]:
        for task in TASK_ORDER:
            subset = [r for r in records if r["mode"] == mode and r["task"] == task]
            sets = [
                set(filter(None, r["effective_sources"].split("|"))) for r in subset
            ]
            jaccards = []
            for i in range(len(sets)):
                for j in range(i + 1, len(sets)):
                    union = sets[i] | sets[j]
                    jaccards.append(len(sets[i] & sets[j]) / len(union) if union else 1.0)
            counts = Counter(source for values in sets for source in values)
            stability_rows.append(
                {
                    "mode": mode,
                    "task": task,
                    "num_runs": len(sets),
                    "mean_pairwise_jaccard": (
                        sum(jaccards) / len(jaccards) if jaccards else None
                    ),
                    "intersection_sources": "|".join(
                        sorted(set.intersection(*sets) if sets else set())
                    ),
                    "union_sources": "|".join(
                        sorted(set.union(*sets) if sets else set())
                    ),
                    "selection_frequencies": "|".join(
                        f"{source}:{count}/{len(sets)}"
                        for source, count in sorted(counts.items())
                    ),
                }
            )

    write_csv(output / "orchestration_run_audit.csv", records)
    write_csv(output / "source_coverage_long.csv", coverage_rows)
    write_csv(output / "builder_relation_stats.csv", builder_rows)
    write_csv(output / "agent_weight_application.csv", weight_rows)
    write_csv(output / "selection_stability.csv", stability_rows)
    write_csv(output / "agent_vs_rule_metrics.csv", metric_comparison)

    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
        }
    )
    pdf_path = output / "external_knowledge_static_audit.pdf"
    with PdfPages(pdf_path) as pdf:
        fig, axes = plt.subplots(1, 2, figsize=(13, 5.6), constrained_layout=True)
        for ax, mode, title in zip(
            axes,
            ["agent", "rule"],
            ["Agent: effective source selection", "Rule: effective source selection"],
        ):
            subset = [r for r in records if r["mode"] == mode]
            matrix = matrix_from_records(
                subset,
                TASK_ORDER,
                SOURCE_ORDER,
                lambda matches, source: (
                    sum(r[f"selected_{source}"] for r in matches) / len(matches)
                    if matches
                    else 0
                ),
            )
            image = draw_heatmap(
                ax, matrix, TASK_ORDER, SOURCE_ORDER, title, "Blues", ".2f"
            )
            fig.colorbar(image, ax=ax, fraction=0.035, pad=0.04, label="Selection frequency")
        fig.suptitle(
            "External knowledge actually applied across three seeds",
            fontsize=14,
            fontweight="bold",
        )
        pdf.savefig(fig, bbox_inches="tight")
        fig.savefig(output / "source_selection_heatmap.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(11, 5.8), constrained_layout=True)
        agent_cov = [
            r
            for r in coverage_rows
            if r["mode"] == "agent" and r["modality"] == "overall"
        ]
        coverage_matrix = []
        for task in TASK_ORDER:
            row = []
            for source in SOURCE_ORDER:
                values = [
                    r["coverage"]
                    for r in agent_cov
                    if r["task"] == task
                    and r["source"] == source
                    and isinstance(r["coverage"], (int, float))
                ]
                row.append(sum(values) / len(values) if values else 0)
            coverage_matrix.append(row)
        image = draw_heatmap(
            ax,
            coverage_matrix,
            TASK_ORDER,
            SOURCE_ORDER,
            "Identifier coverage available to Agent",
            "YlGnBu",
            ".2f",
        )
        fig.colorbar(image, ax=ax, fraction=0.025, pad=0.05, label="Coverage")
        pdf.savefig(fig, bbox_inches="tight")
        fig.savefig(output / "source_coverage_heatmap.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

        relation_keys = [
            "num_direct_links",
            "num_module_links",
            "num_celltype_links",
            "num_proteinatlas_links",
            "gene_graph_edges",
            "protein_graph_edges",
        ]
        relation_labels = [
            "Direct protein-gene",
            "Pathway protein-gene",
            "CellMarker protein-gene",
            "HPA protein-gene",
            "Gene-gene graph",
            "Protein-protein graph",
        ]
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
        for ax, task in zip(axes.flat, TASK_ORDER):
            subset = [r for r in builder_rows if r["mode"] == "agent" and r["task"] == task]
            values = []
            for key in relation_keys:
                available = [
                    float(r[key])
                    for r in subset
                    if isinstance(r.get(key), (int, float))
                ]
                values.append(sum(available) / len(available) if available else 0)
            logged = [math.log10(value + 1) for value in values]
            colors = ["#4477AA", "#66CCEE", "#CC6677", "#EECC66", "#228833", "#AA3377"]
            bars = ax.barh(relation_labels, logged, color=colors)
            ax.set_title(task.replace("task", "Task "), loc="left", fontweight="bold")
            ax.set_xlabel("log10(number of non-zero relations + 1)")
            for bar, raw in zip(bars, values):
                ax.text(
                    bar.get_width() + 0.04,
                    bar.get_y() + bar.get_height() / 2,
                    f"{raw:,.0f}",
                    va="center",
                    fontsize=8,
                )
        fig.suptitle("Scale and composition of constructed knowledge relations", fontsize=14)
        pdf.savefig(fig, bbox_inches="tight")
        fig.savefig(output / "relation_scale.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
        x = range(len(metric_comparison))
        pcc_agent = [row.get("agent_PCC", 0) for row in metric_comparison]
        pcc_rule = [row.get("rule_PCC", 0) for row in metric_comparison]
        width = 0.36
        axes[0].bar([i - width / 2 for i in x], pcc_agent, width, label="Agent", color="#4477AA")
        axes[0].bar([i + width / 2 for i in x], pcc_rule, width, label="Rule", color="#EE8866")
        axes[0].set_xticks(list(x), [f"Task {i + 1}" for i in range(len(x))])
        axes[0].set_ylabel("Mean PCC")
        axes[0].set_title("Existing predictive performance", loc="left", fontweight="bold")
        axes[0].legend(frameon=False)
        delta = [row.get("agent_minus_rule_PCC", 0) for row in metric_comparison]
        colors = ["#228833" if value >= 0 else "#CC3311" for value in delta]
        axes[1].bar(list(x), delta, color=colors)
        axes[1].axhline(0, color="#333333", lw=0.8)
        axes[1].set_xticks(list(x), [f"Task {i + 1}" for i in range(len(x))])
        axes[1].set_ylabel("Agent PCC - Rule PCC")
        axes[1].set_title("Association with Agent configuration", loc="left", fontweight="bold")
        pdf.savefig(fig, bbox_inches="tight")
        fig.savefig(output / "agent_rule_metric_comparison.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

    parse_failures = sum(bool(row["parse_error"]) for row in records if row["mode"] == "agent")
    report = [
        "# External Knowledge Static Audit",
        "",
        f"- Agent orchestration runs audited: {sum(r['mode'] == 'agent' for r in records)}",
        f"- Rule orchestration runs audited: {sum(r['mode'] == 'rule' for r in records)}",
        f"- Agent runs with an LLM parse error: {parse_failures}",
        "- These outputs demonstrate source selection, coverage, constructed relation scale, and association with performance.",
        "- They do not establish causal benefit of the dynamic retriever; that requires saved per-spot tensors and controlled reruns.",
        "",
        "## Files",
        "",
        "- `orchestration_run_audit.csv`: proposed versus effective knowledge sources.",
        "- `source_coverage_long.csv`: per-source and per-modality identifier coverage.",
        "- `builder_relation_stats.csv`: relation counts and graph sizes.",
        "- `agent_weight_application.csv`: base and effective knowledge weights.",
        "- `selection_stability.csv`: cross-seed source-selection stability.",
        "- `agent_vs_rule_metrics.csv`: existing aggregate metric comparison.",
        "- `external_knowledge_static_audit.pdf`: paper-ready visual summary.",
    ]
    (output / "README.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(output), "pdf": str(pdf_path)}, indent=2))


if __name__ == "__main__":
    main()
