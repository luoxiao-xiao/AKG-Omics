from pathlib import Path
import pickle
import numpy as np
import scipy.sparse as sp
from .io import read_json, verify_hash
from .validation import validate_prepared


class TaskData:
    def __init__(self, folder, spec, expected_ready=None):
        self.folder, self.spec = Path(folder), spec
        verify_hash(self.folder / "READY.json", expected_ready)
        self.metadata = validate_prepared(self.folder, spec)
        if not self.metadata.get("acceptance", {}).get("source_only_fit"):
            raise ValueError("A source-only preprocessing record is required")
        if self.metadata.get("task") != spec["task"]:
            raise ValueError("Task number does not match the prepared pack")
        verify_hash(self.folder / "transforms.npz", self.metadata.get("source_transform_sha256"))
        self.full = {k: np.load(self.folder / (k + ".npy"), mmap_mode="r", allow_pickle=False)
                     for k in ("gene", "other", "he", "latent")}
        self.test = {"he": np.load(self.folder / "partial_he.npy", allow_pickle=False)}
        if spec["observed"]:
            self.test[spec["counterpart"]] = np.load(self.folder / "partial_obs.npy", allow_pickle=False)
        self.transforms = np.load(self.folder / "transforms.npz", allow_pickle=False)
        self.train_adj = sp.load_npz(self.folder / "supervised_graph.npz")
        self.test_adj = sp.load_npz(self.folder / "partial_graph.npz")
        self.eval_adj = sp.load_npz(self.folder / "evaluation_graph.npz")
        self.predictions_frozen = False

    def load_truth(self):
        if not self.predictions_frozen:
            raise RuntimeError("Target response is unavailable before prediction is frozen")
        path = self.folder / "target_truth.npy"
        verify_hash(path, self.metadata.get("files", {}).get("target_truth.npy"))
        return np.load(path, allow_pickle=False)


def load_prior(path, data, expected=None, policy=None):
    """Only load local, trusted project priors; pickle is not an interchange format."""
    verify_hash(path, expected)
    with open(path, "rb") as stream:
        kb = pickle.load(stream)
    mode = kb.get("selection", {}).get("mode", "")
    if policy and kb.get("policy") not in (None, policy):
        raise ValueError("Prior policy differs from the requested Rule/Agent version")
    if policy == "agent" and "agent" not in mode:
        raise ValueError("Agent requires a verified Agent prior, not a Rule prior")
    if policy == "rule" and "agent" in mode:
        raise ValueError("Rule cannot load an Agent prior")
    g, o = data.full["gene"].shape[1], data.full["other"].shape[1]
    for key, shape in {"pg": (o, g), "gp": (g, o), "gg": (g, g), "pp": (o, o)}.items():
        if np.shape(kb[key]) != shape or not np.isfinite(kb[key]).all():
            raise ValueError(f"Invalid prior {key}; expected {shape}")
    return kb
