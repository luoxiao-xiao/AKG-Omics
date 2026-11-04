import argparse
import json
from pathlib import Path
from .config import DATASETS, H00, task_spec
from .io import read_json, resolve_path

ROOT = Path(__file__).resolve().parent.parent


def parser():
    p = argparse.ArgumentParser(description="AKG-Omics main model: H00 / source-only PCA / Rule or Agent")
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("tasks", help="List dataset/task directions and the fixed configuration")
    q = sub.add_parser("prepare", help="Fit source-only transforms and export one task")
    q.add_argument("--dataset", required=True, choices=DATASETS)
    q.add_argument("--task", required=True, type=int)
    q.add_argument("--input-config", required=True)
    q.add_argument("--output", required=True)
    q = sub.add_parser("prepare-coad", help="Strict assay QC and physical 32um aggregation")
    for name in ("rna", "protein", "gene-he-cache", "protein-he-cache", "output"):
        q.add_argument("--" + name, required=True)
    q = sub.add_parser("run", help="Run one main-model task; no experiment matrix")
    q.add_argument("--dataset", required=True, choices=DATASETS)
    q.add_argument("--task", required=True, type=int)
    q.add_argument("--policy", required=True, choices=("rule", "agent"))
    q.add_argument("--paths", default=str(ROOT / "configs/server_paths.example.json"))
    q.add_argument("--input-config", help="Optional normalized raw-section arrays: prepare and rebuild knowledge before training")
    q.add_argument("--data", help="Optional already source-fitted task folder")
    q.add_argument("--prior", help="Optional matching local trusted prior.pkl")
    q.add_argument("--assets", help="Knowledge resource directory, used when rebuilding priors")
    q.add_argument("--knowledge", choices=("replay", "select"), default="replay",
                   help="Replay registered selection, or select afresh from source-only profile")
    q.add_argument("--output", required=True)
    q.add_argument("--seed", type=int, default=1)
    q.add_argument("--device", default="cuda:0")
    q.add_argument("--threads", type=int, default=4)
    q.add_argument("--gpu-memory-mib", type=int)
    q.add_argument("--smoke", action="store_true", help="Two-epoch diagnostic, explicitly marked non-final")
    return p


def main(argv=None):
    p = parser()
    a = p.parse_args(argv)
    if a.command == "tasks":
        tasks = [task_spec(ds, t) for ds in DATASETS for t in (range(5, 9) if ds == "mouse" else range(1, 5))]
        print(json.dumps(dict(configuration=H00().to_dict(), tasks=tasks), ensure_ascii=False, indent=2))
        return
    H00().configure_environment()
    if a.command == "prepare-coad":
        from .coad import prepare_coad
        result = prepare_coad(a.rna, a.protein, a.gene_he_cache, a.protein_he_cache, a.output)
        print(result)
        return
    spec = task_spec(a.dataset, a.task)
    if a.command == "prepare":
        from .preprocessing import prepare
        print(prepare(a.input_config, a.dataset, a.task, a.output))
        return
    if a.input_config and a.data:
        p.error("Choose either --input-config or --data")
    paths = read_json(a.paths)
    base = Path(a.paths).resolve().parent
    registered = paths["datasets"][a.dataset]["tasks"][str(a.task)]
    output = Path(a.output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite pipeline output: {output}")
    expected_ready, expected_prior = None, None
    custom_data = bool(a.data or a.input_config)
    if a.input_config:
        from .preprocessing import prepare
        data_path = prepare(a.input_config, a.dataset, a.task, output / "data")
    elif a.data:
        data_path = Path(a.data)
    else:
        data_path = resolve_path(base, registered["data"])
        expected_ready = registered.get("ready_sha256")
    from .data import TaskData
    data = TaskData(data_path, spec, expected_ready=expected_ready)
    # Existing registered pairs replay exactly. New packs rebuild matrices on their source panel.
    built = False
    if a.prior:
        if a.knowledge == "select":
            p.error("--prior reuses a frozen prior; omit it to select knowledge afresh")
        prior_path = Path(a.prior)
    elif custom_data or a.knowledge == "select":
        from .knowledge import build_prior
        asset_path = a.assets or paths["assets"]["metabolite" if a.dataset == "mouse" else "human"]
        prior_path = build_prior(data_path, spec, a.policy, resolve_path(base, asset_path), output / "knowledge",
                    mode=a.knowledge, recipe=ROOT / "recipes" / f"{a.dataset}_T{a.task:02d}_{a.policy}.json")
        built = True
    else:
        prior = registered["priors"][a.policy]
        prior_path = resolve_path(base, prior["file"])
        expected_prior = prior.get("sha256")
    from .trainer import train
    status = dict(selection_replayed=not built or a.knowledge == "replay",
                  actual_llm_called_this_batch=built and a.knowledge == "select" and a.policy == "agent")
    metrics = train(data, prior_path, output / ("smoke" if a.smoke else "run"), a.policy,
                    seed=a.seed, device=a.device, smoke=a.smoke, expected_prior=expected_prior,
                    threads=a.threads, memory_mib=a.gpu_memory_mib, knowledge_context=status)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
