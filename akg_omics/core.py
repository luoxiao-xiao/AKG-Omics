import os
import numpy as np
import scipy.sparse as sp
import torch
from tqdm import tqdm

from . import preprocess as pp
from .utils import create_optimizer
from .model import CollaborativeKBCycleAKGOmics


# ============================================================================
# Detailed loss logging helpers
# ============================================================================
def _float_item(x):
    """Safely convert scalar tensor / number to Python float for logging."""
    if x is None:
        return 0.0
    if torch.is_tensor(x):
        if x.numel() != 1:
            return None
        return float(x.detach().cpu().item())
    if isinstance(x, (int, float, np.integer, np.floating)):
        return float(x)
    return None


def _prefix_loss_dict(loss_dict, prefix):
    """Flatten scalar entries in a loss dictionary with a prefix."""
    out = {}
    if not isinstance(loss_dict, dict):
        return out
    skip_keys = {"recon_obs"}
    for k, v in loss_dict.items():
        if k in skip_keys:
            continue
        val = _float_item(v)
        if val is not None:
            out[f"{prefix}_{k}"] = val
    return out


def _env_bool(name, default):
    v = os.getenv(name)
    if v is None or str(v).strip() == "":
        return bool(default)
    s = str(v).strip().lower()
    if s in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise ValueError(f"Invalid bool env {name}={v}")


def _env_int(name, default):
    v = os.getenv(name)
    if v is None or str(v).strip() == "":
        return int(default)
    return int(v)


def _env_float(name, default):
    v = os.getenv(name)
    if v is None or str(v).strip() == "":
        return float(default)
    return float(v)


def _env_str(name, default):
    v = os.getenv(name)
    if v is None or str(v).strip() == "":
        return str(default)
    return str(v).strip()


class _BaseTrainer:
    def __init__(self, device):
        self.device = torch.device(device) if isinstance(device, str) else device
        self.early_stop_enabled = False
        self.early_stop_patience = 50
        self.early_stop_min_delta = 1e-4
        self.early_stop_min_epochs = 100
        self.early_stop_monitor = "loss_total"
        self.early_stop_restore_best = False
        self._early_stop_best = None
        self._early_stop_best_epoch = 0
        self._early_stop_wait = 0
        self._early_stop_best_state = None

    def _to_numpy(self, x):
        if x is None:
            return None
        if sp.issparse(x):
            x = x.toarray()
        return np.asarray(x, dtype=np.float32)

    def _to_tensor(self, x):
        if x is None:
            return None
        return torch.tensor(self._to_numpy(x), dtype=torch.float32, device=self.device)

    def _to_sparse_tensor(self, graph):
        if isinstance(graph, torch.Tensor):
            return graph.coalesce().to(self.device)
        return pp.sparse_mx_to_torch_sparse_tensor(graph).coalesce().to(self.device)

    def _configure_early_stopping(self):
        self.early_stop_enabled = _env_bool("EARLY_STOP", False)
        self.early_stop_patience = max(1, _env_int("EARLY_STOP_PATIENCE", 50))
        self.early_stop_min_delta = max(0.0, _env_float("EARLY_STOP_MIN_DELTA", 1e-4))
        self.early_stop_min_epochs = max(0, _env_int("EARLY_STOP_MIN_EPOCHS", 100))
        self.early_stop_monitor = _env_str("EARLY_STOP_MONITOR", "loss_total")
        self.early_stop_restore_best = _env_bool("EARLY_STOP_RESTORE_BEST", False)
        self._early_stop_best = None
        self._early_stop_best_epoch = 0
        self._early_stop_wait = 0
        self._early_stop_best_state = None

    def _early_stop_step(self, log, epoch_idx):
        if not self.early_stop_enabled:
            return False
        value = log.get(self.early_stop_monitor, log.get("loss_total"))
        if value is None or not np.isfinite(float(value)):
            return False
        value = float(value)
        improved = self._early_stop_best is None or value < (self._early_stop_best - self.early_stop_min_delta)
        if improved:
            self._early_stop_best = value
            self._early_stop_best_epoch = int(epoch_idx)
            self._early_stop_wait = 0
            if self.early_stop_restore_best and hasattr(self, "model"):
                self._early_stop_best_state = {
                    k: v.detach().cpu().clone()
                    for k, v in self.model.state_dict().items()
                }
        else:
            self._early_stop_wait += 1

        log["early_stop_best"] = float(self._early_stop_best)
        log["early_stop_best_epoch"] = int(self._early_stop_best_epoch)
        log["early_stop_wait"] = int(self._early_stop_wait)
        log["early_stop_monitor"] = self.early_stop_monitor

        return (
            int(epoch_idx) >= int(self.early_stop_min_epochs)
            and int(self._early_stop_wait) >= int(self.early_stop_patience)
        )

    def _restore_early_stop_best_state(self):
        if not self.early_stop_enabled or not self.early_stop_restore_best:
            return
        if self._early_stop_best_state is None or not hasattr(self, "model"):
            return
        self.model.load_state_dict({
            k: v.to(self.device)
            for k, v in self._early_stop_best_state.items()
        })
        self._early_stop_best_state = None

    def _print_early_stop_config(self, prefix=""):
        if self.early_stop_enabled:
            print(
                f">>> [EARLY-STOP] {prefix}monitor={self.early_stop_monitor} "
                f"patience={self.early_stop_patience} min_delta={self.early_stop_min_delta:g} "
                f"min_epochs={self.early_stop_min_epochs} restore_best={self.early_stop_restore_best}",
                flush=True,
            )



class AKGOmicsDualBranchProtocol(_BaseTrainer):
    """
    Dual-branch asymmetric protocol:
      - one supervised branch predicts its target with direct target supervision;
      - one weak branch predicts the opposite target without target supervision;
      - both branches use backward cycle consistency to reconstruct their observed modality.
    """
    def __init__(
        self,
        branch_sup,
        branch_aux,
        he_dim=None,
        protein_dim=None,
        gene_dim=None,
        gene_pca_components=None,
        gene_pca_mean=None,
        hidden_dim=256,
        num_layers=2,
        epochs=200,
        lr=1e-3,
        dropout=0.1,
        weight_decay=0.0,
        optimizer="adam",
        seed=0,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        save_path=None,
        protein_gene_prior=None,
        gene_protein_prior=None,
        gene_gene_kb_graph=None,
        protein_protein_kb_graph=None,
        celltype_gene_prior=None,
        celltype_protein_prior=None,
        lambda_target=1.0,
        lambda_pcc=0.0,
        lambda_latent=0.5,
        lambda_recon=0.2,
        lambda_kb=0.15,
        lambda_graph=0.05,
        lambda_ortho=0.0,
        lambda_dgi=0.0,
        lambda_cycle=0.2,
        lambda_partial_cycle=0.2,
        lambda_align=0.05,
        lambda_bridge_align=0.5,
        target_soft_alpha=0.0,
        kb_ct_prior_mix=0.7,
        lambda_sup=1.0,
        lambda_aux=0.3,
        align_max_points=512,
        use_amp=False,
        disable_dynamic_retriever=False,
        disable_kb_fusion=False,
        disable_soft_gate=True,
        disable_residual_refine=False,
    ):
        super().__init__(device)
        self.branch_sup = self._build_branch(branch_sup)
        self.branch_aux = self._build_branch(branch_aux)
        self.epochs = epochs
        self.seed = seed
        self.save_path = save_path
        self.lambda_align = lambda_align
        self.lambda_sup = lambda_sup
        self.lambda_aux = lambda_aux
        self.align_max_points = align_max_points
        self.use_amp = bool(use_amp and torch.cuda.is_available())

        gene_latent_dim = gene_pca_components.shape[0]
        self.model = CollaborativeKBCycleAKGOmics(
            he_dim=he_dim,
            protein_dim=protein_dim,
            gene_dim=gene_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            gene_latent_dim=gene_latent_dim,
            gene_pca_components=gene_pca_components,
            gene_pca_mean=gene_pca_mean,
            protein_gene_prior=protein_gene_prior,
            gene_protein_prior=gene_protein_prior,
            gene_gene_kb_graph=gene_gene_kb_graph,
            protein_protein_kb_graph=protein_protein_kb_graph,
            celltype_gene_prior=celltype_gene_prior,
            celltype_protein_prior=celltype_protein_prior,
            lambda_target=lambda_target,
            lambda_pcc=lambda_pcc,
            lambda_latent=lambda_latent,
            lambda_recon=lambda_recon,
            lambda_kb=lambda_kb,
            lambda_graph=lambda_graph,
            lambda_ortho=lambda_ortho,
            lambda_dgi=lambda_dgi,
            lambda_cycle=lambda_cycle,
            lambda_partial_cycle=lambda_partial_cycle,
            lambda_align=lambda_align,
            lambda_bridge_align=lambda_bridge_align,
            target_soft_alpha=target_soft_alpha,
            kb_ct_prior_mix=kb_ct_prior_mix,
            disable_dynamic_retriever=disable_dynamic_retriever,
            disable_kb_fusion=disable_kb_fusion,
            disable_soft_gate=disable_soft_gate,
            disable_residual_refine=disable_residual_refine,
        ).to(self.device)
        self.optimizer = create_optimizer(optimizer, [self.model], lr, weight_decay)
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

    def _build_branch(self, cfg):
        branch = {
            "name": cfg.get("name", "branch"),
            "slice_id": int(cfg["slice_id"]),
            "target_type": str(cfg["target_type"]).lower(),
            "obs_type": str(cfg["obs_type"]).lower(),
            "supervised": bool(cfg.get("supervised", False)),
            "eval_target_type": cfg.get("eval_target_type", cfg["target_type"]),
            "he": self._to_tensor(cfg["he"]),
            "adj": self._to_sparse_tensor(cfg["graph"]),
            "obs": self._to_tensor(cfg["obs"]),
            "target": self._to_tensor(cfg.get("target")),
            "target_latent": self._to_tensor(cfg.get("target_latent")),
            "eval_target": self._to_tensor(cfg.get("eval_target")),
        }
        return branch

    def _branch_mode(self, branch):
        if branch["target_type"] == "gene" and branch["obs_type"] == "protein":
            return "he_protein_to_gene"
        if branch["target_type"] == "protein" and branch["obs_type"] == "gene":
            return "he_gene_to_protein"
        raise ValueError(f"Unsupported branch setup: target={branch['target_type']} obs={branch['obs_type']}")

    def _forward_branch(self, branch):
        return self.model.forward_branch(
            branch["he"],
            branch["adj"],
            target_type=branch["target_type"],
            slice_id=branch["slice_id"],
            obs=branch["obs"],
            obs_type=branch["obs_type"],
        )

    def _compute_loss(self, branch, out):
        if branch["supervised"]:
            return self.model.compute_branch_supervised_loss(
                out,
                target_type=branch["target_type"],
                target=branch["target"],
                target_latent=branch["target_latent"],
                obs=branch["obs"],
                obs_type=branch["obs_type"],
                target_graph=branch["adj"],
            )
        return self.model.compute_branch_weak_loss(
            out,
            target_type=branch["target_type"],
            observed_modality=branch["obs_type"],
            observed_data=branch["obs"],
        )

    def train(self):
        pp.set_random_seed(self.seed)
        self.model.train()
        self._configure_early_stopping()
        history = []
        print("\n========================= Start Dual-Branch Training =========================")
        self._print_early_stop_config(prefix="dual ")
        epoch_iter = tqdm(range(self.epochs))

        for epoch_idx in epoch_iter:
            self.optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=self.use_amp):
                out_sup = self._forward_branch(self.branch_sup)
                out_aux = self._forward_branch(self.branch_aux)

                loss_sup = self._compute_loss(self.branch_sup, out_sup)
                loss_aux = self._compute_loss(self.branch_aux, out_aux)
                loss_align = self.model.compute_alignment_loss(out_sup, out_aux, max_points=self.align_max_points)

                total_loss = (
                    self.lambda_sup * loss_sup["loss"]
                    + self.lambda_aux * loss_aux["loss"]
                    + self.lambda_align * loss_align
                )

            self.scaler.scale(total_loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            log = {
                "loss_total": float(total_loss.item()),
                "loss_sup": float(loss_sup["loss"].item()),
                "loss_aux": float(loss_aux["loss"].item()),
                "loss_align": float(loss_align.item()),
            }
            log.update(_prefix_loss_dict(loss_sup, "sup"))
            log.update(_prefix_loss_dict(loss_aux, "aux"))
            history.append(log)
            epoch_iter.set_description(
                f"dual | total={log['loss_total']:.4f} | sup={log['loss_sup']:.4f} | aux={log['loss_aux']:.4f}"
            )
            if self._early_stop_step(log, epoch_idx + 1):
                print(
                    f">>> [EARLY-STOP] dual stopped at epoch {epoch_idx + 1}/{self.epochs}; "
                    f"best {self.early_stop_monitor}={self._early_stop_best:.6f} "
                    f"at epoch {self._early_stop_best_epoch}",
                    flush=True,
                )
                break
        self._restore_early_stop_best_state()
        return history

    @torch.no_grad()
    def infer_branch(self, branch="aux"):
        self.model.eval()
        branch_cfg = self.branch_aux if branch == "aux" else self.branch_sup
        out = self.model.infer_single(
            branch_cfg["he"],
            branch_cfg["adj"],
            mode=self._branch_mode(branch_cfg),
            slice_id=branch_cfg["slice_id"],
            obs=branch_cfg["obs"],
        )
        out = {k: v.detach().cpu().numpy() if torch.is_tensor(v) else v for k, v in out.items()}

        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)
            np.save(os.path.join(self.save_path, f"{branch_cfg['name']}_pred.npy"), out["pred_full"])
        return out

    def get_eval_targets_numpy(self):
        ret = {}
        if self.branch_sup["eval_target"] is not None:
            ret[f"{self.branch_sup['name']}_gt"] = self.branch_sup["eval_target"].detach().cpu().numpy()
        if self.branch_aux["eval_target"] is not None:
            ret[f"{self.branch_aux['name']}_gt"] = self.branch_aux["eval_target"].detach().cpu().numpy()
        return ret




class AKGOmicsFullPartialProtocol(_BaseTrainer):
    """
    Reverted original full-partial protocol under the improved model architecture:
      - both branches solve the SAME target task;
      - full branch has direct target supervision;
      - partial branch does not use hidden target supervision and only uses weak constraints;
      - branches are aligned across slices.
    """
    def __init__(
        self,
        target_task,
        full_he,
        full_graph,
        full_gene,
        full_protein,
        partial_he,
        partial_graph,
        partial_gene_obs=None,
        partial_protein_obs=None,
        partial_gene_eval=None,
        partial_protein_eval=None,
        full_gene_latent=None,
        full_slice_id=0,
        partial_slice_id=1,
        he_dim=None,
        protein_dim=None,
        gene_dim=None,
        gene_pca_components=None,
        gene_pca_mean=None,
        hidden_dim=256,
        num_layers=2,
        epochs=200,
        lr=1e-3,
        dropout=0.1,
        weight_decay=0.0,
        optimizer="adam",
        seed=0,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        save_path=None,
        protein_gene_prior=None,
        gene_protein_prior=None,
        gene_gene_kb_graph=None,
        protein_protein_kb_graph=None,
        celltype_gene_prior=None,
        celltype_protein_prior=None,
        lambda_target=1.0,
        lambda_pcc=0.0,
        lambda_latent=0.5,
        lambda_recon=0.2,
        lambda_kb=0.15,
        lambda_graph=0.05,
        lambda_ortho=0.0,
        lambda_dgi=0.0,
        lambda_cycle=0.2,
        lambda_partial_cycle=0.2,
        lambda_align=0.05,
        lambda_bridge_align=0.5,
        target_soft_alpha=0.0,
        kb_ct_prior_mix=0.7,
        lambda_full=1.0,
        lambda_partial=0.0,
        align_max_points=512,
        full_use_obs=True,
        partial_use_obs=True,
        use_amp=False,
        disable_dynamic_retriever=False,
        disable_kb_fusion=False,
        disable_soft_gate=True,
        disable_residual_refine=False,
    ):
        super().__init__(device)
        assert target_task in ["gene", "protein"]
        self.target_task = target_task
        self.epochs = epochs
        self.seed = seed
        self.save_path = save_path
        self.lambda_align = lambda_align
        self.lambda_full = lambda_full
        self.lambda_partial = lambda_partial
        self.align_max_points = align_max_points
        self.full_use_obs = bool(full_use_obs)
        self.partial_use_obs = bool(partial_use_obs)
        self.use_amp = bool(use_amp and torch.cuda.is_available())
        self.full_slice_id = int(full_slice_id)
        self.partial_slice_id = int(partial_slice_id)

        self.full = {
            "he": self._to_tensor(full_he),
            "adj": self._to_sparse_tensor(full_graph),
            "gene": self._to_tensor(full_gene),
            "protein": self._to_tensor(full_protein),
            "gene_latent": self._to_tensor(full_gene_latent),
        }
        self.partial = {
            "he": self._to_tensor(partial_he),
            "adj": self._to_sparse_tensor(partial_graph),
            "gene_obs": self._to_tensor(partial_gene_obs),
            "protein_obs": self._to_tensor(partial_protein_obs),
            "gene_eval": self._to_tensor(partial_gene_eval),
            "protein_eval": self._to_tensor(partial_protein_eval),
        }

        gene_latent_dim = gene_pca_components.shape[0]
        self.model = CollaborativeKBCycleAKGOmics(
            he_dim=he_dim,
            protein_dim=protein_dim,
            gene_dim=gene_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            gene_latent_dim=gene_latent_dim,
            gene_pca_components=gene_pca_components,
            gene_pca_mean=gene_pca_mean,
            protein_gene_prior=protein_gene_prior,
            gene_protein_prior=gene_protein_prior,
            gene_gene_kb_graph=gene_gene_kb_graph,
            protein_protein_kb_graph=protein_protein_kb_graph,
            celltype_gene_prior=celltype_gene_prior,
            celltype_protein_prior=celltype_protein_prior,
            lambda_target=lambda_target,
            lambda_pcc=lambda_pcc,
            lambda_latent=lambda_latent,
            lambda_recon=lambda_recon,
            lambda_kb=lambda_kb,
            lambda_graph=lambda_graph,
            lambda_ortho=lambda_ortho,
            lambda_dgi=lambda_dgi,
            lambda_cycle=lambda_cycle,
            lambda_partial_cycle=lambda_partial_cycle,
            lambda_align=lambda_align,
            lambda_bridge_align=lambda_bridge_align,
            target_soft_alpha=target_soft_alpha,
            kb_ct_prior_mix=kb_ct_prior_mix,
            disable_dynamic_retriever=disable_dynamic_retriever,
            disable_kb_fusion=disable_kb_fusion,
            disable_soft_gate=disable_soft_gate,
            disable_residual_refine=disable_residual_refine,
        ).to(self.device)
        self.optimizer = create_optimizer(optimizer, [self.model], lr, weight_decay)
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

    def _resolve_obs_config(self, split):
        assert split in ["full", "partial"]
        use_obs = self.full_use_obs if split == "full" else self.partial_use_obs
        store = self.full if split == "full" else self.partial

        if not use_obs:
            return None, None

        if self.target_task == "protein":
            obs = store["gene"] if split == "full" else store["gene_obs"]
            obs_type = "gene"
        else:
            obs = store["protein"] if split == "full" else store["protein_obs"]
            obs_type = "protein"

        if obs is None:
            raise ValueError(f"{split}_use_obs=True but required observed modality is missing for target={self.target_task}.")
        return obs, obs_type

    def train(self):
        pp.set_random_seed(self.seed)
        self.model.train()
        self._configure_early_stopping()
        history = []
        print(f"\n========================= Start Full-Partial Training: target={self.target_task} =========================")
        self._print_early_stop_config(prefix=f"full-partial target={self.target_task} ")
        epoch_iter = tqdm(range(self.epochs))

        for epoch_idx in epoch_iter:
            self.optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=self.use_amp):
                full_obs, full_obs_type = self._resolve_obs_config("full")
                partial_obs, partial_obs_type = self._resolve_obs_config("partial")

                if self.target_task == "protein":
                    out_full = self.model.forward_branch(
                        self.full["he"], self.full["adj"],
                        target_type="protein", slice_id=self.full_slice_id,
                        obs=full_obs, obs_type=full_obs_type
                    )
                    loss_full = self.model.compute_branch_supervised_loss(
                        out_full, target_type="protein",
                        target=self.full["protein"], target_latent=None,
                        obs=full_obs, obs_type=full_obs_type,
                        target_graph=self.full["adj"],
                    )

                    out_partial = self.model.forward_branch(
                        self.partial["he"], self.partial["adj"],
                        target_type="protein", slice_id=self.partial_slice_id,
                        obs=partial_obs, obs_type=partial_obs_type
                    )
                    loss_partial = self.model.compute_branch_weak_loss(
                        out_partial,
                        target_type="protein",
                        observed_modality=partial_obs_type,
                        observed_data=partial_obs,
                    )
                else:
                    out_full = self.model.forward_branch(
                        self.full["he"], self.full["adj"],
                        target_type="gene", slice_id=self.full_slice_id,
                        obs=full_obs, obs_type=full_obs_type
                    )
                    loss_full = self.model.compute_branch_supervised_loss(
                        out_full, target_type="gene",
                        target=self.full["gene"], target_latent=self.full["gene_latent"],
                        obs=full_obs, obs_type=full_obs_type,
                        target_graph=self.full["adj"],
                    )

                    out_partial = self.model.forward_branch(
                        self.partial["he"], self.partial["adj"],
                        target_type="gene", slice_id=self.partial_slice_id,
                        obs=partial_obs, obs_type=partial_obs_type
                    )
                    loss_partial = self.model.compute_branch_weak_loss(
                        out_partial,
                        target_type="gene",
                        observed_modality=partial_obs_type,
                        observed_data=partial_obs,
                    )

                align = self.model.compute_alignment_loss(out_full, out_partial, max_points=self.align_max_points)
                total_loss = (
                    self.lambda_full * loss_full["loss"]
                    + self.lambda_partial * loss_partial["loss"]
                    + self.lambda_align * align
                )

            self.scaler.scale(total_loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            log = {
                "loss_total": float(total_loss.item()),
                "loss_full": float(loss_full["loss"].item()),
                "loss_partial": float(loss_partial["loss"].item()),
                "loss_align": float(align.item()),
            }
            log.update(_prefix_loss_dict(loss_full, "full"))
            log.update(_prefix_loss_dict(loss_partial, "partial"))
            history.append(log)
            epoch_iter.set_description(
                f"{self.target_task} | total={log['loss_total']:.4f} | full={log['loss_full']:.4f} | partial={log['loss_partial']:.4f}"
            )
            if self._early_stop_step(log, epoch_idx + 1):
                print(
                    f">>> [EARLY-STOP] full-partial target={self.target_task} stopped at epoch "
                    f"{epoch_idx + 1}/{self.epochs}; best {self.early_stop_monitor}="
                    f"{self._early_stop_best:.6f} at epoch {self._early_stop_best_epoch}",
                    flush=True,
                )
                break

        self._restore_early_stop_best_state()
        return history

    @torch.no_grad()
    def infer_partial(self):
        self.model.eval()
        partial_obs, _ = self._resolve_obs_config("partial")

        if self.target_task == "protein":
            infer_mode = "he_gene_to_protein" if self.partial_use_obs else "he_to_protein"
            out = self.model.infer_single(
                self.partial["he"], self.partial["adj"],
                mode=infer_mode, slice_id=self.partial_slice_id, obs=partial_obs
            )
            pred_name = "partial_protein_pred.npy"
        else:
            infer_mode = "he_protein_to_gene" if self.partial_use_obs else "he_to_gene"
            out = self.model.infer_single(
                self.partial["he"], self.partial["adj"],
                mode=infer_mode, slice_id=self.partial_slice_id, obs=partial_obs
            )
            pred_name = "partial_gene_pred.npy"

        def _to_numpy_tree(value):
            if torch.is_tensor(value):
                return value.detach().cpu().numpy()
            if isinstance(value, dict):
                return {key: _to_numpy_tree(item) for key, item in value.items()}
            if isinstance(value, list):
                return [_to_numpy_tree(item) for item in value]
            if isinstance(value, tuple):
                return tuple(_to_numpy_tree(item) for item in value)
            return value

        out = _to_numpy_tree(out)
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)
            np.save(os.path.join(self.save_path, pred_name), out["pred_full"])
        return out

    def get_eval_targets_numpy(self):
        ret = {}
        if self.partial["gene_eval"] is not None:
            ret["partial_gene_gt"] = self.partial["gene_eval"].detach().cpu().numpy()
        if self.partial["protein_eval"] is not None:
            ret["partial_protein_gt"] = self.partial["protein_eval"].detach().cpu().numpy()
        return ret


class AKGOmicsSingleHEProtocol(_BaseTrainer):
    """
    Single-branch HE-only protocol under the unified architecture.
      - predicts one target modality from HE only;
      - uses supervised losses available in the model (target/latent/recon/kb/graph/etc).
    """
    def __init__(
        self,
        target_task,
        he,
        graph,
        target,
        target_latent=None,
        slice_id=0,
        he_dim=None,
        protein_dim=None,
        gene_dim=None,
        gene_pca_components=None,
        gene_pca_mean=None,
        hidden_dim=256,
        num_layers=2,
        epochs=200,
        lr=1e-3,
        dropout=0.1,
        weight_decay=0.0,
        optimizer="adam",
        seed=0,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        save_path=None,
        protein_gene_prior=None,
        gene_protein_prior=None,
        gene_gene_kb_graph=None,
        protein_protein_kb_graph=None,
        celltype_gene_prior=None,
        celltype_protein_prior=None,
        lambda_target=1.0,
        lambda_pcc=0.0,
        lambda_latent=0.5,
        lambda_recon=0.2,
        lambda_kb=0.15,
        lambda_graph=0.05,
        lambda_ortho=0.0,
        lambda_dgi=0.0,
        lambda_cycle=0.2,
        lambda_partial_cycle=0.2,
        lambda_align=0.05,
        lambda_bridge_align=0.5,
        target_soft_alpha=0.0,
        kb_ct_prior_mix=0.7,
        lambda_full=1.0,
        lambda_partial=0.0,
        align_max_points=512,
        use_amp=False,
        disable_dynamic_retriever=False,
        disable_kb_fusion=False,
        disable_soft_gate=True,
        disable_residual_refine=False,
    ):
        super().__init__(device)
        assert target_task in ["gene", "protein"]
        self.target_task = str(target_task)
        self.epochs = int(epochs)
        self.seed = int(seed)
        self.save_path = save_path
        self.slice_id = int(slice_id)
        self.align_max_points = int(align_max_points)
        # kept only for signature compatibility with shared kwargs in unified runners
        self.lambda_full = float(lambda_full)
        self.lambda_partial = float(lambda_partial)
        self.use_amp = bool(use_amp and torch.cuda.is_available())

        self.he = self._to_tensor(he)
        self.adj = self._to_sparse_tensor(graph)
        self.target = self._to_tensor(target)
        self.target_latent = self._to_tensor(target_latent)

        gene_latent_dim = gene_pca_components.shape[0]
        self.model = CollaborativeKBCycleAKGOmics(
            he_dim=he_dim,
            protein_dim=protein_dim,
            gene_dim=gene_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            gene_latent_dim=gene_latent_dim,
            gene_pca_components=gene_pca_components,
            gene_pca_mean=gene_pca_mean,
            protein_gene_prior=protein_gene_prior,
            gene_protein_prior=gene_protein_prior,
            gene_gene_kb_graph=gene_gene_kb_graph,
            protein_protein_kb_graph=protein_protein_kb_graph,
            celltype_gene_prior=celltype_gene_prior,
            celltype_protein_prior=celltype_protein_prior,
            lambda_target=lambda_target,
            lambda_pcc=lambda_pcc,
            lambda_latent=lambda_latent,
            lambda_recon=lambda_recon,
            lambda_kb=lambda_kb,
            lambda_graph=lambda_graph,
            lambda_ortho=lambda_ortho,
            lambda_dgi=lambda_dgi,
            lambda_cycle=lambda_cycle,
            lambda_partial_cycle=lambda_partial_cycle,
            lambda_align=lambda_align,
            lambda_bridge_align=lambda_bridge_align,
            target_soft_alpha=target_soft_alpha,
            kb_ct_prior_mix=kb_ct_prior_mix,
            disable_dynamic_retriever=disable_dynamic_retriever,
            disable_kb_fusion=disable_kb_fusion,
            disable_soft_gate=disable_soft_gate,
            disable_residual_refine=disable_residual_refine,
        ).to(self.device)
        self.optimizer = create_optimizer(optimizer, [self.model], lr, weight_decay)
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

    def _infer_mode(self):
        return "he_to_gene" if self.target_task == "gene" else "he_to_protein"

    def train(self):
        pp.set_random_seed(self.seed)
        self.model.train()
        self._configure_early_stopping()
        history = []
        print(f"\n========================= Start Single-HE Training: target={self.target_task} =========================")
        self._print_early_stop_config(prefix=f"single-he target={self.target_task} ")
        epoch_iter = tqdm(range(self.epochs))

        for epoch_idx in epoch_iter:
            self.optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                out = self.model.forward_branch(
                    self.he,
                    self.adj,
                    target_type=self.target_task,
                    slice_id=self.slice_id,
                    obs=None,
                    obs_type=None,
                )
                loss = self.model.compute_branch_supervised_loss(
                    out,
                    target_type=self.target_task,
                    target=self.target,
                    target_latent=self.target_latent if self.target_task == "gene" else None,
                    obs=None,
                    obs_type=None,
                    target_graph=self.adj,
                )
                total_loss = loss["loss"]

            self.scaler.scale(total_loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            log = {
                "loss_total": float(total_loss.item()),
            }
            log.update(_prefix_loss_dict(loss, "sup"))
            history.append(log)
            epoch_iter.set_description(
                f"single-he {self.target_task} | total={log['loss_total']:.4f} | "
                f"target={log.get('sup_loss_target', 0.0):.4f} | "
                f"kb={log.get('sup_loss_kb', 0.0):.4f} | "
                f"graph={log.get('sup_loss_graph', 0.0):.4f}"
            )
            if self._early_stop_step(log, epoch_idx + 1):
                print(
                    f">>> [EARLY-STOP] single-he target={self.target_task} stopped at epoch "
                    f"{epoch_idx + 1}/{self.epochs}; best {self.early_stop_monitor}="
                    f"{self._early_stop_best:.6f} at epoch {self._early_stop_best_epoch}",
                    flush=True,
                )
                break
        self._restore_early_stop_best_state()
        return history

    @torch.no_grad()
    def infer(self):
        self.model.eval()
        out = self.model.infer_single(
            self.he,
            self.adj,
            mode=self._infer_mode(),
            slice_id=self.slice_id,
            obs=None,
        )
        out = {k: v.detach().cpu().numpy() if torch.is_tensor(v) else v for k, v in out.items()}
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)
            pred_name = "single_he_gene_pred.npy" if self.target_task == "gene" else "single_he_protein_pred.npy"
            np.save(os.path.join(self.save_path, pred_name), out["pred_full"])
        return out

    def get_eval_targets_numpy(self):
        return {
            "single_he_gt": self.target.detach().cpu().numpy()
        }
