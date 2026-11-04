import numpy as np
import scipy.sparse as sp
import torch
from . import tensorops as pp
from .utils import create_optimizer
from .model import CollaborativeKBCycleSpatialEx

class _BaseTrainer:

    def __init__(self, device):
        self.device = torch.device(device) if isinstance(device, str) else device
        self.early_stop_enabled = False
        self.early_stop_patience = 50
        self.early_stop_min_delta = 0.0001
        self.early_stop_min_epochs = 100
        self.early_stop_monitor = 'loss_total'
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

class SpatialExFullPartialProtocol(_BaseTrainer):

    def __init__(self, target_task, full_he, full_graph, full_gene, full_protein, partial_he, partial_graph, partial_gene_obs=None, partial_protein_obs=None, partial_gene_eval=None, partial_protein_eval=None, full_gene_latent=None, full_slice_id=0, partial_slice_id=1, he_dim=None, protein_dim=None, gene_dim=None, gene_pca_components=None, gene_pca_mean=None, hidden_dim=256, num_layers=2, epochs=200, lr=0.001, dropout=0.1, weight_decay=0.0, optimizer='adam', seed=0, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), save_path=None, protein_gene_prior=None, gene_protein_prior=None, gene_gene_kb_graph=None, protein_protein_kb_graph=None, celltype_gene_prior=None, celltype_protein_prior=None, lambda_target=1.0, lambda_pcc=0.0, lambda_latent=0.5, lambda_recon=0.2, lambda_kb=0.15, lambda_graph=0.05, lambda_ortho=0.0, lambda_dgi=0.0, lambda_cycle=0.2, lambda_partial_cycle=0.2, lambda_align=0.05, lambda_bridge_align=0.5, target_soft_alpha=0.0, kb_ct_prior_mix=0.7, lambda_full=1.0, lambda_partial=0.0, align_max_points=512, full_use_obs=True, partial_use_obs=True, use_amp=False, disable_dynamic_retriever=False, disable_kb_fusion=False, disable_soft_gate=True, disable_residual_refine=False):
        super().__init__(device)
        assert target_task in ['gene', 'protein']
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
        self.full = {'he': self._to_tensor(full_he), 'adj': self._to_sparse_tensor(full_graph), 'gene': self._to_tensor(full_gene), 'protein': self._to_tensor(full_protein), 'gene_latent': self._to_tensor(full_gene_latent)}
        self.partial = {'he': self._to_tensor(partial_he), 'adj': self._to_sparse_tensor(partial_graph), 'gene_obs': self._to_tensor(partial_gene_obs), 'protein_obs': self._to_tensor(partial_protein_obs), 'gene_eval': self._to_tensor(partial_gene_eval), 'protein_eval': self._to_tensor(partial_protein_eval)}
        gene_latent_dim = gene_pca_components.shape[0]
        self.model = CollaborativeKBCycleSpatialEx(he_dim=he_dim, protein_dim=protein_dim, gene_dim=gene_dim, hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout, gene_latent_dim=gene_latent_dim, gene_pca_components=gene_pca_components, gene_pca_mean=gene_pca_mean, protein_gene_prior=protein_gene_prior, gene_protein_prior=gene_protein_prior, gene_gene_kb_graph=gene_gene_kb_graph, protein_protein_kb_graph=protein_protein_kb_graph, celltype_gene_prior=celltype_gene_prior, celltype_protein_prior=celltype_protein_prior, lambda_target=lambda_target, lambda_pcc=lambda_pcc, lambda_latent=lambda_latent, lambda_recon=lambda_recon, lambda_kb=lambda_kb, lambda_graph=lambda_graph, lambda_ortho=lambda_ortho, lambda_dgi=lambda_dgi, lambda_cycle=lambda_cycle, lambda_partial_cycle=lambda_partial_cycle, lambda_align=lambda_align, lambda_bridge_align=lambda_bridge_align, target_soft_alpha=target_soft_alpha, kb_ct_prior_mix=kb_ct_prior_mix, disable_dynamic_retriever=disable_dynamic_retriever, disable_kb_fusion=disable_kb_fusion, disable_soft_gate=disable_soft_gate, disable_residual_refine=disable_residual_refine).to(self.device)
        self.optimizer = create_optimizer(optimizer, [self.model], lr, weight_decay)
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

    def _resolve_obs_config(self, split):
        assert split in ['full', 'partial']
        use_obs = self.full_use_obs if split == 'full' else self.partial_use_obs
        store = self.full if split == 'full' else self.partial
        if not use_obs:
            return (None, None)
        if self.target_task == 'protein':
            obs = store['gene'] if split == 'full' else store['gene_obs']
            obs_type = 'gene'
        else:
            obs = store['protein'] if split == 'full' else store['protein_obs']
            obs_type = 'protein'
        if obs is None:
            raise ValueError(f'{split}_use_obs=True but required observed modality is missing for target={self.target_task}.')
        return (obs, obs_type)
