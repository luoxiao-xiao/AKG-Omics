"""Source-only panel selection, scaling and PCA; frozen task-pack export."""
from pathlib import Path
import tempfile
import numpy as np
import scipy.sparse as sp
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from .config import H00, task_spec
from .io import array_digest, read_json, resolve_path, sha256, write_json


def graph(coords):
    n = len(coords)
    if n < 2:
        raise ValueError("At least two spatial observations are required")
    ix = NearestNeighbors(n_neighbors=min(8, n)).fit(coords).kneighbors(coords, return_distance=False)
    g = sp.coo_matrix((np.ones(ix.size, np.float32),
                       (np.repeat(np.arange(n), ix.shape[1]), ix.ravel())), shape=(n, n)).tocsr()
    g = g.maximum(g.T)
    degree = np.asarray(g.sum(1)).ravel()
    return (sp.diags(1 / np.sqrt(degree)) @ g @ sp.diags(1 / np.sqrt(degree))).astype(np.float32).tocsr()


def load_section(config, base, modalities=("gene", "other")):
    """Inputs are deterministic per-observation transforms, before fitted scaling/PCA."""
    folder = resolve_path(base, config["folder"])
    result = {}
    for key in (*modalities, "he", "coords", "ids", "gene_names", "other_names"):
        path = resolve_path(folder, config.get(key, key + ".npy"))
        result[key] = np.load(path, allow_pickle=False)
    for key in ("gene_ids", "other_symbols"):
        p = resolve_path(folder, config.get(key, key + ".npy"))
        if p.exists():
            result[key] = np.load(p, allow_pickle=False)
    n = len(result["ids"])
    for key in (*modalities, "he", "coords"):
        if result[key].ndim != 2 or len(result[key]) != n or not np.isfinite(result[key]).all():
            raise ValueError(f"Invalid {key} rows in {folder}")
    for key in ("ids", "gene_names", "other_names"):
        values = result[key].astype(str)
        if len(set(values)) != len(values):
            raise ValueError(f"Duplicate {key}: merge raw counts before importing")
        result[key] = values
    for key in modalities:
        if result[key].shape[1] != len(result[key + "_names"]):
            raise ValueError("Feature identifiers and matrix dimensions disagree")
    return result


def fit_source(source, dataset):
    """This API deliberately accepts no target section."""
    g = np.asarray(source["gene"], dtype=np.float32)
    other = np.asarray(source["other"], dtype=np.float32)
    if g.ndim != 2 or other.ndim != 2 or len(g) < 2 or len(g) != len(other):
        raise ValueError("Source matrices must have matching rows and at least two observations")
    if not all(np.isfinite(source[k]).all() for k in ("gene", "other", "he")):
        raise ValueError("Non-finite source values cannot be fitted")
    variance = g.var(0)
    valid = ~np.char.startswith(np.char.upper(source["gene_names"].astype(str)), "MT-")
    if dataset != "mouse":
        valid &= variance > 1e-8
    gi = np.flatnonzero(valid)
    limit = 1000 if dataset == "mouse" else 4806
    gi = np.sort(gi[np.argsort(variance[gi], kind="stable")[-min(limit, len(gi)):]])
    oi = np.sort(np.argsort(other.var(0), kind="stable")[-min(50, other.shape[1]):]) if dataset == "mouse" else np.arange(other.shape[1])
    x, y = g[:, gi], other[:, oi]
    if dataset == "mouse":
        go, oo = x.min(0), y.min(0)
        gs, os = x.max(0) - go, y.max(0) - oo
    else:
        go, oo = x.mean(0), y.mean(0)
        gs, os = x.std(0, ddof=1), y.std(0, ddof=1)
    gs, os = np.where(gs > 1e-8, gs, 1), np.where(os > 1e-8, os, 1)
    z = np.asarray((x - go) / gs, np.float32, order="C")
    if min(z.shape) < H00().gene_pca_dim:
        raise ValueError("H00 requires at least 128 source observations and selected genes")
    pc = PCA(H00().gene_pca_dim, svd_solver="randomized", random_state=0).fit(z)
    transform = dict(gene_indices=gi, other_indices=oi, gene_offset=go, gene_scale=gs,
                     other_offset=oo, other_scale=os,
                     gene_components=pc.components_.astype(np.float32), gene_mean=pc.mean_.astype(np.float32))
    if dataset != "mouse":
        if min(source["he"].shape) < 128:
            raise ValueError("Human H00 HE PCA requires 128 source rows and embedding dimensions")
        hp = PCA(H00().he_pca_dim, svd_solver="randomized", random_state=0).fit(source["he"])
        transform.update(he_components=hp.components_.astype(np.float32), he_mean=hp.mean_.astype(np.float32))
    return transform


def align(names, wanted):
    index = {s: i for i, s in enumerate(names.astype(str))}
    missing = [s for s in wanted.astype(str) if s not in index]
    if missing:
        raise ValueError(f"Required source-panel features absent in target: {missing[:8]}")
    return np.asarray([index[s] for s in wanted.astype(str)])


def prepare(input_config, dataset, task, output):
    """Commit a new pack only when every preparation check has passed.

    Legacy packs must be rebuilt from normalized source inputs, never relabelled.
    The hidden response is exported last for evaluation, not used in fitting.
    """
    output = Path(output)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"Refusing to overwrite prepared data: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}.building-", dir=output.parent) as temp:
        stage = Path(temp) / "pack"
        _prepare_into(input_config, dataset, task, stage)
        from .validation import validate_prepared
        validate_prepared(stage, task_spec(dataset, task))
        if output.exists():
            output.rmdir()  # Fails if another writer has populated this directory.
        stage.rename(output)
    return output


def _prepare_into(input_config, dataset, task, output):
    config = read_json(input_config)
    if config.get("dataset", dataset) != dataset:
        raise ValueError("Input configuration dataset differs from requested dataset")
    spec = task_spec(dataset, task)
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite prepared data: {output}")
    output.mkdir(parents=True, exist_ok=True)
    if config.get("representation") != "per_observation_normalized_before_source_fit":
        raise ValueError("Declare the deterministic input representation; see configs/input_template.json")
    base = Path(input_config).resolve().parent
    source = load_section(config["sections"][spec["source"]], base)
    tr = fit_source(source, dataset)
    # Persist the fitted source transform BEFORE opening target expression arrays.
    np.savez(output / "transforms.npz", **tr)
    write_json(output / "fit_scope.json", dict(schema_version=2, source_only=True, target_values_used_in_fit=False,
               dataset=dataset, task=task, fit_api="fit_source(source, dataset)",
               fit_rows=len(source["ids"]), source_side=spec["source"],
               source_ids_sha256=array_digest(np.asarray(source["ids"].tolist(), dtype=str)),
               source_arrays_sha256={k: array_digest(source[k]) for k in ("gene", "other", "he")},
               transforms_sha256=sha256(output / "transforms.npz"),
               gene_pca_dim=H00().gene_pca_dim, he_pca_dim=H00().he_pca_dim,
               pca_solver="randomized", pca_random_state=0))
    # Only the protocol-permitted target input is opened here. Never open the
    # target response or an unused counterpart while constructing model inputs.
    target = load_section(config["sections"][spec["target"]], base,
                          modalities=(spec["counterpart"],) if spec["observed"] else ())
    gi, oi = tr["gene_indices"], tr["other_indices"]
    gn, on = source["gene_names"][gi], source["other_names"][oi]
    ti, tj = align(target["gene_names"], gn), align(target["other_names"], on)
    gx = np.asarray((source["gene"][:, gi] - tr["gene_offset"]) / tr["gene_scale"], np.float32)
    ox = np.asarray((source["other"][:, oi] - tr["other_offset"]) / tr["other_scale"], np.float32)
    def he(x):
        return np.asarray((x - tr["he_mean"]) @ tr["he_components"].T, np.float32) if "he_components" in tr else x.astype(np.float32)
    arrays = dict(gene=gx, other=ox, he=he(source["he"]), partial_he=he(target["he"]),
                  latent=np.asarray((gx - tr["gene_mean"]) @ tr["gene_components"].T, np.float32),
                  gene_names=gn, other_names=on, coords=target["coords"],
                  source_ids=np.char.add(spec["source"] + ":", source["ids"]),
                  target_ids=np.char.add(spec["target"] + ":", target["ids"]))
    if spec["observed"]:
        kind = spec["counterpart"]
        indices = ti if kind == "gene" else tj
        arrays["partial_obs"] = np.asarray((target[kind][:, indices] - tr[kind + "_offset"]) / tr[kind + "_scale"], np.float32)
    if "gene_ids" in source:
        arrays["gene_ids"] = source["gene_ids"][gi]
    arrays["other_symbols"] = source.get("other_symbols", source["other_names"])[oi]
    for k, v in arrays.items():
        np.save(output / (k + ".npy"), v)
    for name, co in [("supervised_graph", source["coords"]), ("partial_graph", target["coords"]), ("evaluation_graph", target["coords"])]:
        sp.save_npz(output / (name + ".npz"), graph(co))
    # Separate evaluation export: all fitted state and training inputs already
    # exist. These response values cannot feed selection, scaling, PCA or graphs.
    response = spec["target_kind"]
    section_config = config["sections"][spec["target"]]
    response_path = resolve_path(resolve_path(base, section_config["folder"]),
                                 section_config.get(response, response + ".npy"))
    raw_truth = np.load(response_path, allow_pickle=False)
    expected_width = len(target[response + "_names"])
    if raw_truth.shape != (len(target["ids"]), expected_width) or not np.isfinite(raw_truth).all():
        raise ValueError("Invalid evaluation response shape or values")
    response_arrays_distinct = array_digest(source[response]) != array_digest(raw_truth)
    if dataset != "coad" and not response_arrays_distinct:
        raise ValueError("Source and evaluation response arrays are identical; acquisition lineage must be checked")
    ri = ti if response == "gene" else tj
    np.save(output / "target_truth.npy", np.asarray((raw_truth[:, ri] - tr[response + "_offset"]) / tr[response + "_scale"], np.float32))
    meta = dict(**spec, preprocessing_schema_version=2, source_n=len(gx), target_n=len(target["ids"]), gene_dim=len(gn), other_dim=len(on),
                source_transform_sha256=sha256(output / "transforms.npz"),
                input_modalities=["he"] + ([spec["counterpart"]] if spec["observed"] else []),
                acceptance=dict(source_only_fit=True,
                                response_arrays_distinct=response_arrays_distinct,
                                response_lineage_disjoint=False if dataset == "coad" else config.get("response_lineage_disjoint", False)),
                qualification="Same-acquisition reconstruction reference" if dataset == "coad" else "Source-fitted task pack; biological lineage requires original acquisition audit",
                input_config_sha256=sha256(input_config),
                files={p.name: sha256(p) for p in output.iterdir() if p.is_file()})
    write_json(output / "READY.json", meta)
    return output
