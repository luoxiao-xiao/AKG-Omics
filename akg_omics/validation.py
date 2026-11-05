"""Fail closed on legacy, incomplete or incompatible fitted task packs.

Hashes establish file identity, not acquisition independence. Source-only fitting
is enforced by the preparation API and tested by target perturbation checks.
"""
from pathlib import Path
import numpy as np
import scipy.sparse as sp
from .config import H00
from .io import array_digest, read_json, verify_hash


def validate_prepared(folder, spec):
    folder = Path(folder)
    meta = read_json(folder / "READY.json")
    if meta.get("preprocessing_schema_version") != 2:
        raise ValueError("Legacy/unverified PCA cache: rebuild with prepare from source inputs")
    for key in ("dataset", "task", "source", "target", "target_kind", "counterpart", "observed", "model_target"):
        if meta.get(key) != spec[key]:
            raise ValueError(f"Prepared data protocol mismatch: {key}")
    required = {"gene.npy", "other.npy", "he.npy", "latent.npy", "partial_he.npy",
                "gene_names.npy", "other_names.npy", "source_ids.npy", "target_ids.npy",
                "coords.npy", "transforms.npz", "fit_scope.json", "target_truth.npy",
                "supervised_graph.npz", "partial_graph.npz", "evaluation_graph.npz"}
    if spec["observed"]:
        required.add("partial_obs.npy")
    files = meta.get("files", {})
    if not required.issubset(files):
        raise ValueError("Incomplete preprocessing manifest")
    for name, expected in files.items():
        if Path(name).name != name or not expected or not (folder / name).is_file():
            raise ValueError(f"Missing or invalid registered file: {name}")
        # The evaluator alone opens the response, after predictions are frozen.
        if name != "target_truth.npy":
            verify_hash(folder / name, expected)
    scope = read_json(folder / "fit_scope.json")
    if not meta.get("acceptance", {}).get("source_only_fit") or scope.get("schema_version") != 2:
        raise ValueError("Missing source-fit provenance")
    if scope.get("source_only") is not True or scope.get("target_values_used_in_fit") is not False:
        raise ValueError("Joint/unknown PCA fitting scope is prohibited")
    for key, expected in {"dataset": spec["dataset"], "task": spec["task"],
                          "source_side": spec["source"], "fit_rows": meta["source_n"]}.items():
        if scope.get(key) != expected:
            raise ValueError(f"Source-fit provenance mismatch: {key}")
    for expected in (scope.get("transforms_sha256"), meta.get("source_transform_sha256")):
        if not expected:
            raise ValueError("Missing fitted-transform fingerprint")
        verify_hash(folder / "transforms.npz", expected)
    def load(name):
        return np.load(folder / (name + ".npy"), allow_pickle=False)
    gene, latent, other, he = (load(k) for k in ("gene", "latent", "other", "he"))
    ns, nt, g, o = (meta[k] for k in ("source_n", "target_n", "gene_dim", "other_dim"))
    for name, value, shape in [("gene", gene, (ns, g)), ("other", other, (ns, o)),
                               ("latent", latent, (ns, H00().gene_pca_dim))]:
        if value.shape != shape or not np.isfinite(value).all():
            raise ValueError(f"Invalid source array: {name}")
    for name, n in [("source_ids", ns), ("target_ids", nt), ("gene_names", g), ("other_names", o)]:
        value = load(name).astype(str)
        if value.shape != (n,) or len(np.unique(value)) != n:
            raise ValueError(f"Invalid identities: {name}")
    ids = load("source_ids").astype(str)
    prefix = spec["source"] + ":"
    if not all(x.startswith(prefix) for x in ids) or array_digest(np.asarray([x[len(prefix):] for x in ids])) != scope.get("source_ids_sha256"):
        raise ValueError("Source observation identities differ from PCA fit record")
    partial = load("partial_he")
    if he.ndim != 2 or he.shape[0] != ns or partial.shape != (nt, he.shape[1]) or not np.isfinite(he).all() or not np.isfinite(partial).all():
        raise ValueError("Invalid H&E inputs")
    target_ids = load("target_ids").astype(str)
    if not all(x.startswith(spec["target"] + ":") for x in target_ids):
        raise ValueError("Target observation section differs from protocol")
    coords = load("coords")
    if coords.shape != (nt, 2) or not np.isfinite(coords).all():
        raise ValueError("Invalid target spatial coordinates")
    for name, n in [("supervised_graph", ns), ("partial_graph", nt), ("evaluation_graph", nt)]:
        adj = sp.load_npz(folder / (name + ".npz"))
        if adj.shape != (n, n) or not np.isfinite(adj.data).all():
            raise ValueError(f"Invalid spatial graph: {name}")
    if spec["observed"]:
        obs = load("partial_obs")
        if obs.shape != (nt, o if spec["counterpart"] == "other" else g) or not np.isfinite(obs).all():
            raise ValueError("Invalid permitted target modality")
    with np.load(folder / "transforms.npz", allow_pickle=False) as tr:
        if spec["dataset"] != "mouse":
            if he.shape[1] != H00().he_pca_dim or tr["he_components"].shape[0] != H00().he_pca_dim:
                raise ValueError("Invalid source H&E PCA dimensions")
            if not np.isfinite(tr["he_mean"]).all() or not np.isfinite(tr["he_components"]).all():
                raise ValueError("Invalid H&E transform")
        for k, width in [("gene", g), ("other", o)]:
            for suffix in ("offset", "scale"):
                v = tr[k + "_" + suffix]
                if v.shape != (width,) or not np.isfinite(v).all() or (suffix == "scale" and np.any(v <= 0)):
                    raise ValueError(f"Invalid source {k} {suffix}")
        comp, mean = tr["gene_components"], tr["gene_mean"]
        if comp.shape != (H00().gene_pca_dim, g) or mean.shape != (g,):
            raise ValueError("Invalid source PCA dimensions")
        np.testing.assert_allclose(comp @ comp.T, np.eye(len(comp)), atol=2e-4, rtol=2e-4,
                                   err_msg="Invalid PCA basis")
        np.testing.assert_allclose(mean, gene.mean(0), atol=2e-5, rtol=2e-4,
                                   err_msg="PCA centering differs from supervised source")
        for start in range(0, ns, 512):
            np.testing.assert_allclose(latent[start:start+512], (gene[start:start+512] - mean) @ comp.T,
                                       atol=2e-4, rtol=2e-4, err_msg="Stale/incompatible gene PCA cache")
    return meta
