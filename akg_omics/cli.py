"""Command-line interface for data preparation and AKG-Omics training."""
import argparse
import json
from pathlib import Path

from .config import DATASETS, H00, task_spec
from .io import read_json

ROOT = Path(__file__).resolve().parent.parent


def parser():
    p = argparse.ArgumentParser(description="AKG-Omics")
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("tasks", help="List supported prediction tasks")

    q = sub.add_parser("prepare", help="Prepare a task using source-fitted transforms")
    q.add_argument("--dataset", required=True, choices=DATASETS)
    q.add_argument("--task", required=True, type=int)
    q.add_argument("--input-config")
    q.add_argument("--output", required=True)

    q = sub.add_parser("audit-data", help="Validate a prepared task")
    q.add_argument("--dataset", required=True, choices=DATASETS)
    q.add_argument("--task", required=True, type=int)
    q.add_argument("--data", required=True)

    q = sub.add_parser("prepare-coad", help="Import paired molecular assays and image features")
    for name in ("rna", "protein", "gene-he-cache", "protein-he-cache", "output"):
        q.add_argument("--" + name, required=True)

    q = sub.add_parser("run", help="Prepare data, build knowledge, train and evaluate")
    q.add_argument("--dataset", required=True, choices=DATASETS)
    q.add_argument("--task", required=True, type=int)
    q.add_argument("--policy", choices=("rule", "agent"), default="rule")
    inputs = q.add_mutually_exclusive_group()
    inputs.add_argument("--input-config", help="Section input configuration")
    inputs.add_argument("--data", help="Reuse a prepared task directory")
    q.add_argument("--prior", help="Reuse a matching trusted local prior.pkl")
    q.add_argument("--assets", help="Knowledge resource directory")
    q.add_argument("--knowledge", choices=("fixed", "select"), default="fixed",
                   help="Use provided source-selection settings or select sources afresh")
    q.add_argument("--output", required=True)
    q.add_argument("--seed", type=int, default=1)
    q.add_argument("--device", default="cuda:0")
    q.add_argument("--threads", type=int, default=4)
    q.add_argument("--gpu-memory-mib", type=int)
    q.add_argument("--smoke", action="store_true", help="Run a two-epoch check")
    return p


def main(argv=None):
    p = parser()
    a = p.parse_args(argv)
    if a.command == "tasks":
        tasks = [task_spec(ds, t) for ds in DATASETS
                 for t in (range(5, 9) if ds == "mouse" else range(1, 5))]
        print(json.dumps(tasks, indent=2))
        return
    if a.command == "prepare-coad":
        from .coad import prepare_coad
        print(prepare_coad(a.rna, a.protein, a.gene_he_cache, a.protein_he_cache, a.output))
        return
    spec = task_spec(a.dataset, a.task)
    if a.command == "audit-data":
        from .validation import validate_prepared
        validate_prepared(a.data, spec)
        print(json.dumps(dict(passed=True, dataset=a.dataset, task=a.task)))
        return
    input_config = getattr(a, "input_config", None) or ROOT / "configs" / f"{a.dataset}_inputs.json"
    if a.command == "prepare":
        from .preprocessing import prepare
        print(prepare(input_config, a.dataset, a.task, a.output))
        return
    if a.prior and a.knowledge == "select":
        p.error("Omit --prior when selecting knowledge sources afresh")
    output = Path(a.output)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"Output already contains files: {output}")
    H00().configure_environment()
    if a.data:
        data_path = Path(a.data)
    else:
        from .preprocessing import prepare
        data_path = prepare(input_config, a.dataset, a.task, output / "data")
    from .data import TaskData
    data = TaskData(data_path, spec)
    if a.prior:
        prior_path = Path(a.prior)
        context = dict(selection_replayed=True, actual_llm_called_this_batch=False)
    else:
        from .knowledge import build_prior
        assets = Path(a.assets) if a.assets else ROOT / "assets" / ("metabolite" if a.dataset == "mouse" else "human")
        recipe = read_json(ROOT / "configs/knowledge.json")[a.dataset][str(a.task)][a.policy]
        prior_path = build_prior(data_path, spec, a.policy, assets, output / "knowledge",
                                 mode="replay" if a.knowledge == "fixed" else "select", recipe=recipe)
        context = dict(selection_replayed=a.knowledge == "fixed",
                       actual_llm_called_this_batch=a.knowledge == "select" and a.policy == "agent")
    from .trainer import train
    metrics = train(data, prior_path, output / "run", a.policy, seed=a.seed, device=a.device,
                    smoke=a.smoke, threads=a.threads, memory_mib=a.gpu_memory_mib,
                    knowledge_context=context)
    print(json.dumps(metrics, indent=2))
