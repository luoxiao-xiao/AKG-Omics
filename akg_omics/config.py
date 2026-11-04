"""Frozen main-model configuration; dataset/task choose data, never hyperparameters."""
import os
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class H00:
    name: str = "H00"
    hidden_dim: int = 256
    num_layers: int = 2
    dropout: float = 0.1
    lr: float = 0.001
    weight_decay: float = 0.0
    max_epochs: int = 700
    patience: int = 30
    min_delta: float = 0.0001
    cycle: float = 0.2
    kb: float = 0.05
    temperature: float = 0.5
    modality_dropout: float = 0.15
    gate_regularizer: float = 0.005
    alignment: float = 0.02
    bridge_alignment: float = 0.5
    embedding_scale: float = 0.05
    he_local_residual_scale: float = 0.5
    gene_pca_dim: int = 128
    he_pca_dim: int = 128
    align_max_points: int = 512
    restore_best: bool = False

    def to_dict(self):
        return asdict(self)

    def configure_environment(self):
        # Do not inherit a disabled component or sensitivity setting from a shell.
        values = {
            "DISABLE_SPATIAL_GRAPH": "0", "DISABLE_SOFT_GATE": "1",
            "DISABLE_KB_FUSION": "0", "DISABLE_DYNAMIC_RETRIEVER": "0",
            "FUSION_GATE_TEMPERATURE": self.temperature,
            "FUSION_MODALITY_DROPOUT": self.modality_dropout,
            "FUSION_GATE_DIVERSITY_WEIGHT": 0.1,
            "LAMBDA_FUSION_GATE": self.gate_regularizer,
            "USE_HE_TRANSFORMER": "1",
            "HE_GLOBAL_TRANSFORMER_LAYERS": 2, "HE_GLOBAL_TRANSFORMER_HEADS": 4,
            "HE_GLOBAL_TRANSFORMER_FF_MULT": 2.0, "HE_GLOBAL_TRANSFORMER_TOKENS": 4,
            "HE_LOCAL_TRANSFORMER_LAYERS": 1, "HE_LOCAL_TRANSFORMER_HEADS": 8,
            "HE_LOCAL_TRANSFORMER_FF_MULT": 2.0, "HE_LOCAL_TRANSFORMER_CHUNK_SIZE": 512,
            "HE_LOCAL_TRANSFORMER_RESIDUAL_SCALE": self.he_local_residual_scale,
        }
        os.environ.update({k: str(v) for k, v in values.items()})
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


DATASETS = ("coad", "lymphnode", "tonsil", "mouse")


def task_spec(dataset, task):
    if dataset not in DATASETS:
        raise ValueError(f"Unknown dataset: {dataset}")
    allowed = range(5, 9) if dataset == "mouse" else range(1, 5)
    if task not in allowed:
        raise ValueError(f"{dataset} supports tasks {list(allowed)}, received T{task}")
    gene = task in (1, 3, 6, 8)
    observed = task in (3, 4, 7, 8)
    if dataset == "mouse":
        source = "B1" if gene else "C1"
        target = "C1" if gene else "B1"
    elif dataset == "coad":
        source, target = ("protein", "gene") if gene else ("gene", "protein")
    else:
        source, target = ("A1", "D1") if gene else ("D1", "A1")
    return dict(dataset=dataset, task=task, source=source, target=target,
                target_kind="gene" if gene else "other",
                model_target="gene" if gene else "protein", observed=observed,
                counterpart="other" if gene else "gene",
                independent_response=dataset != "coad")
