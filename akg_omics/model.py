import os

import torch
import torch.nn as nn
import torch.nn.functional as F


def create_activation(name: str):
    name = (name or "prelu").lower()
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    if name == "elu":
        return nn.ELU()
    return nn.PReLU()


class HGNN(nn.Module):
    def __init__(self, in_dim, num_hidden, out_dim, num_layers=2, dropout=0.1, activation="prelu"):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.activation = create_activation(activation)
        self.layers = nn.ModuleList()

        if num_layers <= 1:
            self.layers.append(nn.Linear(in_dim, out_dim))
        elif num_layers == 2:
            self.layers.append(nn.Linear(in_dim, num_hidden))
            self.layers.append(nn.Linear(num_hidden, out_dim))
        else:
            self.layers.append(nn.Linear(in_dim, num_hidden))
            for _ in range(num_layers - 2):
                self.layers.append(nn.Linear(num_hidden, num_hidden))
            self.layers.append(nn.Linear(num_hidden, out_dim))

    def forward(self, x, adj):
        h = x
        disable_spatial_graph = str(os.getenv("DISABLE_SPATIAL_GRAPH", "0")).strip().lower() in {"1", "true", "t", "yes", "y", "on"}
        for i, layer in enumerate(self.layers):
            h = layer(self.dropout(h))
            if not disable_spatial_graph:
                if adj.is_sparse:
                    with torch.cuda.amp.autocast(enabled=False):
                        h = torch.sparse.mm(adj.float(), h.float())
                else:
                    h = adj.float() @ h.float()
            if i != len(self.layers) - 1:
                h = self.activation(h)
        return h


class MLPEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

    def forward(self, x):
        return self.net(x)


class MLPDecoder(nn.Module):
    def __init__(self, hidden_dim, out_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


def _env_bool(name, default):
    value = os.getenv(name)
    if value is None or str(value).strip() == "":
        return bool(default)
    value = str(value).strip().lower()
    if value in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise ValueError(f"Invalid bool env {name}={value}")


def _env_int(name, default):
    value = os.getenv(name)
    if value is None or str(value).strip() == "":
        return int(default)
    return int(value)


def _env_float(name, default):
    value = os.getenv(name)
    if value is None or str(value).strip() == "":
        return float(default)
    return float(value)


def _valid_num_heads(hidden_dim, requested_heads):
    requested_heads = max(1, int(requested_heads))
    if hidden_dim % requested_heads == 0:
        return requested_heads
    for h in range(min(requested_heads, hidden_dim), 0, -1):
        if hidden_dim % h == 0:
            return h
    return 1


class HEGlobalLocalTransformerSplitter(nn.Module):
    """Split encoded HE features into global and local representations.

    The global branch uses a tiny token sequence, so it captures slice-level
    context without quadratic attention over all cells.  The local branch runs
    TransformerEncoder on chunks of cells to retain fine-grained variation.
    """

    def __init__(
        self,
        hidden_dim,
        dropout=0.1,
        enable_transformer=True,
        global_layers=2,
        global_heads=4,
        global_ff_mult=2.0,
        global_tokens=4,
        local_layers=1,
        local_heads=8,
        local_ff_mult=2.0,
        local_chunk_size=512,
        local_residual_scale=0.5,
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.enable_transformer = bool(enable_transformer)
        self.local_chunk_size = max(16, int(local_chunk_size))
        self.local_residual_scale = float(local_residual_scale)
        self.global_tokens_n = max(1, int(global_tokens))

        # Legacy projection splitter kept as a fallback and for easy rollback:
        # self.he_shared_proj = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU())
        # self.he_local_proj = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU())
        self.legacy_shared_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.legacy_local_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        if not self.enable_transformer:
            return

        g_heads = _valid_num_heads(hidden_dim, global_heads)
        l_heads = _valid_num_heads(hidden_dim, local_heads)
        g_ff = max(hidden_dim, int(round(hidden_dim * float(global_ff_mult))))
        l_ff = max(hidden_dim, int(round(hidden_dim * float(local_ff_mult))))

        self.global_tokens = nn.Parameter(torch.randn(self.global_tokens_n, hidden_dim) * 0.02)
        global_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=g_heads,
            dim_feedforward=g_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.global_encoder = nn.TransformerEncoder(global_layer, num_layers=max(1, int(global_layers)))
        self.global_out = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        self.local_pos = nn.Parameter(torch.randn(self.local_chunk_size, hidden_dim) * 0.01)
        local_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=l_heads,
            dim_feedforward=l_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.local_encoder = nn.TransformerEncoder(local_layer, num_layers=max(1, int(local_layers)))
        self.local_out = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

    def _global_context(self, he_base):
        mean_token = he_base.mean(dim=0, keepdim=True)
        max_token = he_base.max(dim=0, keepdim=True).values
        learned = self.global_tokens.to(he_base.device, he_base.dtype)
        tokens = torch.cat([learned, mean_token, max_token], dim=0).unsqueeze(0)
        encoded = self.global_encoder(tokens).squeeze(0)
        return encoded[: self.global_tokens_n].mean(dim=0, keepdim=True)

    def _local_context(self, he_base):
        chunks = []
        n = he_base.size(0)
        for start in range(0, n, self.local_chunk_size):
            end = min(start + self.local_chunk_size, n)
            chunk = he_base[start:end]
            pos = self.local_pos[: end - start].to(chunk.device, chunk.dtype)
            encoded = self.local_encoder((chunk + pos).unsqueeze(0)).squeeze(0)
            chunks.append(chunk + self.local_residual_scale * (encoded - chunk))
        return torch.cat(chunks, dim=0) if chunks else he_base

    def forward(self, he_base):
        if not self.enable_transformer:
            return self.legacy_shared_proj(he_base), self.legacy_local_proj(he_base)

        global_context = self._global_context(he_base).expand(he_base.size(0), -1)
        he_shared = self.global_out(torch.cat([he_base, global_context], dim=1))

        local_context = self._local_context(he_base)
        he_local = self.local_out(torch.cat([he_base, local_context], dim=1))
        return he_shared, he_local


class TypeEmbedding(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.emb = nn.Embedding(3, hidden_dim)

    def forward(self, kind, batch_size: int, device):
        if kind is None:
            idx = 0
        else:
            kind = str(kind).lower()
            if kind in ["none", "null", "he_only"]:
                idx = 0
            elif kind == "protein":
                idx = 1
            elif kind == "gene":
                idx = 2
            else:
                raise ValueError(f"Unsupported kind: {kind}")
        return self.emb(torch.full((batch_size,), idx, dtype=torch.long, device=device))


class SliceEmbedding(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.emb = nn.Embedding(2, hidden_dim)

    def forward(self, slice_id: int, batch_size: int, device):
        return self.emb(torch.full((batch_size,), int(slice_id), dtype=torch.long, device=device))


class GraphDGIRegularizer(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.criterion = nn.CosineEmbeddingLoss()

    def forward(self, x):
        h1 = self.proj(x)
        idx = torch.randperm(h1.size(0), device=h1.device)
        h2 = self.proj(x[idx])
        c = h1.mean(0, keepdim=True)
        pos = torch.ones(h1.size(0), device=h1.device)
        neg = -torch.ones(h2.size(0), device=h2.device)
        return self.criterion(h1, c.expand_as(h1), pos) + self.criterion(h2, c.expand_as(h2), neg)


class DynamicKBRetriever(nn.Module):
    def __init__(self, hidden_dim, gene_dim, protein_dim, dropout=0.1):
        super().__init__()
        self.gene_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, gene_dim),
        )
        self.protein_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, protein_dim),
        )

    def forward(self, he_shared, obs_hidden, target_type):
        x = torch.cat([he_shared, obs_hidden], dim=1)
        if target_type == "gene":
            score = self.gene_head(x)
        elif target_type == "protein":
            score = self.protein_head(x)
        else:
            raise ValueError(f"Unsupported target_type={target_type}")
        mask = torch.sigmoid(score)
        return mask, score


class KnowledgeBridge(nn.Module):
    def __init__(
        self,
        protein_dim,
        gene_dim,
        hidden_dim,
        protein_gene_prior=None,
        gene_protein_prior=None,
        gene_gene_graph=None,
        protein_protein_graph=None,
        celltype_gene_prior=None,
        celltype_protein_prior=None,
        kb_ct_prior_mix=0.7,
        dropout=0.1,
        disable_dynamic_retriever=False,
    ):
        super().__init__()
        self.gene_dim = gene_dim
        self.protein_dim = protein_dim
        self.disable_dynamic_retriever = bool(disable_dynamic_retriever)
        self.kb_ct_prior_mix = float(max(0.0, min(1.0, kb_ct_prior_mix)))

        if protein_gene_prior is not None:
            self.register_buffer("protein_gene_prior", torch.tensor(protein_gene_prior, dtype=torch.float32))
        else:
            self.protein_gene_prior = None

        if gene_protein_prior is not None:
            self.register_buffer("gene_protein_prior", torch.tensor(gene_protein_prior, dtype=torch.float32))
        else:
            self.gene_protein_prior = None

        if gene_gene_graph is not None:
            self.register_buffer("gene_gene_graph", torch.tensor(gene_gene_graph, dtype=torch.float32))
        else:
            self.gene_gene_graph = None

        if protein_protein_graph is not None:
            self.register_buffer("protein_protein_graph", torch.tensor(protein_protein_graph, dtype=torch.float32))
        else:
            self.protein_protein_graph = None

        if celltype_gene_prior is not None:
            ctg = torch.tensor(celltype_gene_prior, dtype=torch.float32)
            self.register_buffer("celltype_gene_prior", ctg)
        else:
            self.celltype_gene_prior = None

        if celltype_protein_prior is not None:
            ctp = torch.tensor(celltype_protein_prior, dtype=torch.float32)
            self.register_buffer("celltype_protein_prior", ctp)
        else:
            self.celltype_protein_prior = None

        self.num_celltypes = 0
        if self.celltype_gene_prior is not None:
            self.num_celltypes = int(self.celltype_gene_prior.size(0))
        if self.celltype_protein_prior is not None:
            self.num_celltypes = max(self.num_celltypes, int(self.celltype_protein_prior.size(0)))
        if self.num_celltypes > 0:
            self.celltype_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, self.num_celltypes),
            )
        else:
            self.celltype_head = None

        self.retriever = DynamicKBRetriever(hidden_dim, gene_dim, protein_dim, dropout)

        self.gene_prior_proj = nn.Sequential(
            nn.Linear(gene_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.protein_prior_proj = nn.Sequential(
            nn.Linear(protein_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.he_to_gene_prior = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, gene_dim),
        )
        self.he_to_protein_prior = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, protein_dim),
        )

    def _row_norm(self, x, eps=1e-6):
        denom = x.abs().mean(dim=1, keepdim=True) + eps
        return x / denom

    def _smooth(self, x, target_type):
        if target_type == "gene" and self.gene_gene_graph is not None:
            return 0.5 * x + 0.5 * (x @ self.gene_gene_graph)
        if target_type == "protein" and self.protein_protein_graph is not None:
            return 0.5 * x + 0.5 * (x @ self.protein_protein_graph)
        return x

    def _celltype_prior(self, he_shared, target_type):
        if self.celltype_head is None:
            return None, None
        logits = self.celltype_head(he_shared)
        probs = torch.softmax(logits, dim=1)
        if target_type == "gene" and self.celltype_gene_prior is not None:
            return probs @ self.celltype_gene_prior, probs
        if target_type == "protein" and self.celltype_protein_prior is not None:
            return probs @ self.celltype_protein_prior, probs
        return None, probs

    def forward(self, he_shared, obs_hidden, obs=None, obs_type=None, target_type="gene"):
        local_mask, local_score = self.retriever(he_shared, obs_hidden, target_type)
        if self.disable_dynamic_retriever:
            local_mask = torch.ones_like(local_mask)
            local_score = torch.zeros_like(local_score)

        if target_type == "gene":
            if obs is not None and obs_type == "protein" and self.protein_gene_prior is not None:
                global_prior = self._row_norm(F.relu(obs) @ self.protein_gene_prior)
            else:
                he_prior = self.he_to_gene_prior(he_shared)
                ct_prior, ct_prob = self._celltype_prior(he_shared, target_type="gene")
                if ct_prior is not None:
                    global_prior = (1.0 - self.kb_ct_prior_mix) * he_prior + self.kb_ct_prior_mix * ct_prior
                else:
                    global_prior = he_prior

            global_prior = self._smooth(global_prior, "gene")
            local_prior = global_prior * local_mask
            kb_hidden = self.gene_prior_proj(local_prior)
            return {
                "global_prior": global_prior,
                "local_prior": local_prior,
                "local_mask": local_mask,
                "local_score": local_score,
                "kb_hidden": kb_hidden,
                "celltype_prob": ct_prob if obs is None else None,
            }

        if target_type == "protein":
            if obs is not None and obs_type == "gene" and self.gene_protein_prior is not None:
                global_prior = self._row_norm(F.relu(obs) @ self.gene_protein_prior)
            else:
                he_prior = self.he_to_protein_prior(he_shared)
                ct_prior, ct_prob = self._celltype_prior(he_shared, target_type="protein")
                if ct_prior is not None:
                    global_prior = (1.0 - self.kb_ct_prior_mix) * he_prior + self.kb_ct_prior_mix * ct_prior
                else:
                    global_prior = he_prior

            global_prior = self._smooth(global_prior, "protein")
            local_prior = global_prior * local_mask
            kb_hidden = self.protein_prior_proj(local_prior)
            return {
                "global_prior": global_prior,
                "local_prior": local_prior,
                "local_mask": local_mask,
                "local_score": local_score,
                "kb_hidden": kb_hidden,
                "celltype_prob": ct_prob if obs is None else None,
            }

        raise ValueError(f"Unsupported target_type={target_type}")


class KnowledgeAwareFusion(nn.Module):
    def __init__(self, hidden_dim, dropout=0.1):
        super().__init__()
        self.temperature = max(0.05, _env_float("FUSION_GATE_TEMPERATURE", 1.0))
        self.modality_dropout = max(0.0, min(0.8, _env_float("FUSION_MODALITY_DROPOUT", 0.0)))
        self.diversity_weight = max(0.0, _env_float("FUSION_GATE_DIVERSITY_WEIGHT", 0.1))
        self.gate3 = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim * 3),
        )
        self.gate4 = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim * 4),
        )
        self.out = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def _apply_modality_dropout(self, features):
        if not self.training or self.modality_dropout <= 0.0:
            return features
        dropped = [features[0]]
        keep_prob = 1.0 - self.modality_dropout
        for feat in features[1:]:
            keep = (torch.rand(feat.shape[0], 1, device=feat.device) < keep_prob).to(feat.dtype)
            dropped.append(feat * keep / max(keep_prob, 1e-6))
        return dropped

    def _dynamic_gate(self, features, gates):
        dyn_features = []
        for i, sf_m in enumerate(features):
            w_m = gates[i]
            pair_terms = [
                w_m * sf_m + (1.0 - w_m) * sf_p
                for j, sf_p in enumerate(features)
                if j != i
            ]
            dyn_m = torch.stack(pair_terms, dim=0).sum(dim=0)
            dyn_m = dyn_m * (2.0 / float(len(pair_terms)))
            dyn_features.append(dyn_m)
        return torch.stack(dyn_features, dim=0).mean(dim=0)

    def _gate_regularizer(self, gates):
        eps = 1e-6
        gate_stack = torch.stack(gates, dim=0)
        entropy = -(gate_stack * torch.log(gate_stack + eps) + (1.0 - gate_stack) * torch.log(1.0 - gate_stack + eps)).mean()
        diversity = gate_stack.mean(dim=2).var(dim=0, unbiased=False).mean()
        return entropy - self.diversity_weight * diversity

    def forward(self, he_shared, he_local, obs_hidden, kb_hidden, has_obs=False):
        if has_obs:
            features = [he_shared, he_local, obs_hidden, kb_hidden]
            features = self._apply_modality_dropout(features)
            gates = torch.chunk(
                torch.sigmoid(self.gate4(torch.cat(features, dim=1)) / self.temperature),
                4,
                dim=1,
            )
            gate_names = ["gate_he_shared", "gate_he_local", "gate_obs", "gate_kb"]
        else:
            features = [he_shared, he_local, kb_hidden]
            features = self._apply_modality_dropout(features)
            gates = torch.chunk(
                torch.sigmoid(self.gate3(torch.cat(features, dim=1)) / self.temperature),
                3,
                dim=1,
            )
            gate_names = ["gate_he_shared", "gate_he_local", "gate_kb"]

        fused = self.out(self._dynamic_gate(features, gates))
        gate_info = {name: gate for name, gate in zip(gate_names, gates)}
        if not has_obs:
            gate_info["gate_obs"] = torch.zeros_like(gates[0])
        gate_info["fusion_has_obs"] = bool(has_obs)
        gate_info["fusion_num_features"] = len(features)
        gate_info["gate_reg_loss"] = self._gate_regularizer(gates)
        gate_info["gate_temperature"] = self.temperature
        gate_info["gate_modality_dropout"] = self.modality_dropout
        for name, gate in zip(gate_names, gates):
            gate_info[f"{name}_mean"] = gate.mean()
            gate_info[f"{name}_std"] = gate.std(unbiased=False)
        return fused, gate_info


class KnowledgeCycleModule(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


def pairwise_sqdist(x, y):
    x_norm = (x ** 2).sum(dim=1, keepdim=True)
    y_norm = (y ** 2).sum(dim=1, keepdim=True).T
    return (x_norm + y_norm - 2.0 * x @ y.T).clamp_min(0.0)


def mmd_rbf(x, y, sigmas=(1.0, 2.0, 5.0), max_points=1024):
    if x.size(0) > max_points:
        x = x[torch.randperm(x.size(0), device=x.device)[:max_points]]
    if y.size(0) > max_points:
        y = y[torch.randperm(y.size(0), device=y.device)[:max_points]]

    d_xx = pairwise_sqdist(x, x)
    d_yy = pairwise_sqdist(y, y)
    d_xy = pairwise_sqdist(x, y)

    loss = 0.0
    for sigma in sigmas:
        gamma = 1.0 / (2.0 * sigma * sigma)
        loss = (
            loss
            + torch.exp(-gamma * d_xx).mean()
            + torch.exp(-gamma * d_yy).mean()
            - 2.0 * torch.exp(-gamma * d_xy).mean()
        )
    return loss / len(sigmas)


def mean_cov_loss(x, y):
    mx, my = x.mean(0), y.mean(0)
    xc, yc = x - mx, y - my
    cov_x = xc.T @ xc / max(x.size(0) - 1, 1)
    cov_y = yc.T @ yc / max(y.size(0) - 1, 1)
    return F.mse_loss(mx, my) + F.mse_loss(cov_x, cov_y)


class CollaborativeKBCycleAKGOmics(nn.Module):
    def __init__(
        self,
        he_dim,
        protein_dim,
        gene_dim,
        hidden_dim=256,
        num_layers=2,
        dropout=0.1,
        activation="prelu",
        gene_latent_dim=None,
        gene_pca_components=None,
        gene_pca_mean=None,
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
        lambda_cycle_consistency=0.05,
        lambda_align=0.05,
        lambda_partial_cycle=0.20,
        lambda_bridge_align=0.5,
        target_soft_alpha=0.0,
        kb_ct_prior_mix=0.7,
        disable_dynamic_retriever=False,
        disable_kb_fusion=False,
        disable_soft_gate=True,
        disable_residual_refine=False,
    ):
        super().__init__()
        self.lambda_target = lambda_target
        self.lambda_pcc = lambda_pcc
        self.lambda_latent = lambda_latent
        self.lambda_recon = lambda_recon
        self.lambda_kb = lambda_kb
        self.lambda_graph = lambda_graph
        self.lambda_ortho = lambda_ortho
        self.lambda_dgi = lambda_dgi
        self.lambda_cycle = lambda_cycle
        self.lambda_cycle_consistency = lambda_cycle_consistency
        self.lambda_align = lambda_align
        self.lambda_partial_cycle = lambda_partial_cycle
        self.lambda_bridge_align = lambda_bridge_align
        self.lambda_fusion_gate = max(0.0, _env_float("LAMBDA_FUSION_GATE", 0.0))
        self.target_soft_alpha = float(max(0.0, min(1.0, target_soft_alpha)))
        self.disable_dynamic_retriever = bool(disable_dynamic_retriever)
        self.disable_kb_fusion = bool(disable_kb_fusion)
        self.disable_soft_gate = bool(disable_soft_gate)
        self.disable_residual_refine = bool(disable_residual_refine)

        self.protein_dim = protein_dim
        self.gene_dim = gene_dim
        self.hidden_dim = hidden_dim
        self.gene_latent_dim = gene_latent_dim

        self.he_encoder = MLPEncoder(he_dim, hidden_dim, dropout)
        # Legacy projection-only HE splitter.  Keep these modules for ablation
        # and rollback; the active path below now uses HEGlobalLocalTransformerSplitter
        # by default.  To switch back, set USE_HE_TRANSFORMER=0.
        #
        # Original implementation:
        # self.he_shared_proj = nn.Sequential(
        #     nn.Linear(hidden_dim, hidden_dim),
        #     nn.LayerNorm(hidden_dim),
        #     nn.GELU(),
        # )
        # self.he_local_proj = nn.Sequential(
        #     nn.Linear(hidden_dim, hidden_dim),
        #     nn.LayerNorm(hidden_dim),
        #     nn.GELU(),
        # )
        self.he_shared_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.he_local_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.he_feature_splitter = HEGlobalLocalTransformerSplitter(
            hidden_dim=hidden_dim,
            dropout=dropout,
            enable_transformer=_env_bool("USE_HE_TRANSFORMER", True),
            global_layers=_env_int("HE_GLOBAL_TRANSFORMER_LAYERS", 2),
            global_heads=_env_int("HE_GLOBAL_TRANSFORMER_HEADS", 4),
            global_ff_mult=_env_float("HE_GLOBAL_TRANSFORMER_FF_MULT", 2.0),
            global_tokens=_env_int("HE_GLOBAL_TRANSFORMER_TOKENS", 4),
            local_layers=_env_int("HE_LOCAL_TRANSFORMER_LAYERS", 1),
            local_heads=_env_int("HE_LOCAL_TRANSFORMER_HEADS", 8),
            local_ff_mult=_env_float("HE_LOCAL_TRANSFORMER_FF_MULT", 2.0),
            local_chunk_size=_env_int("HE_LOCAL_TRANSFORMER_CHUNK_SIZE", 512),
            local_residual_scale=_env_float("HE_LOCAL_TRANSFORMER_RESIDUAL_SCALE", 0.5),
        )

        self.protein_encoder = MLPEncoder(protein_dim, hidden_dim, dropout)
        self.gene_encoder = MLPEncoder(gene_dim, hidden_dim, dropout)

        self.type_embedding = TypeEmbedding(hidden_dim)
        self.slice_embedding = SliceEmbedding(hidden_dim)

        self.kb_bridge = KnowledgeBridge(
            protein_dim=protein_dim,
            gene_dim=gene_dim,
            hidden_dim=hidden_dim,
            protein_gene_prior=protein_gene_prior,
            gene_protein_prior=gene_protein_prior,
            gene_gene_graph=gene_gene_kb_graph,
            protein_protein_graph=protein_protein_kb_graph,
            celltype_gene_prior=celltype_gene_prior,
            celltype_protein_prior=celltype_protein_prior,
            kb_ct_prior_mix=kb_ct_prior_mix,
            dropout=dropout,
            disable_dynamic_retriever=disable_dynamic_retriever,
        )
        self.fusion = KnowledgeAwareFusion(hidden_dim, dropout)

        self.spatial_backbone = HGNN(
            hidden_dim, hidden_dim, hidden_dim,
            num_layers=num_layers, dropout=dropout, activation=activation
        )
        self.shared_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.specific_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        self.graph_dgi = GraphDGIRegularizer(hidden_dim)

        self.register_buffer("gene_pca_components", torch.tensor(gene_pca_components, dtype=torch.float32))
        self.register_buffer("gene_pca_mean", torch.tensor(gene_pca_mean, dtype=torch.float32))

        self.gene_coarse_head = MLPDecoder(hidden_dim, gene_latent_dim, dropout)
        self.gene_residual_head = MLPDecoder(hidden_dim, gene_dim, dropout)
        self.gene_soft_gate = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, gene_dim),
            nn.Sigmoid(),
        )

        self.protein_coarse_head = MLPDecoder(hidden_dim, protein_dim, dropout)
        self.protein_residual_head = MLPDecoder(hidden_dim, protein_dim, dropout)
        self.protein_soft_gate = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, protein_dim),
            nn.Sigmoid(),
        )

        self.protein_recon_head = MLPDecoder(hidden_dim, protein_dim, dropout)
        self.gene_recon_latent_head = MLPDecoder(hidden_dim, gene_latent_dim, dropout)

        self.kc_p2g = KnowledgeCycleModule(hidden_dim * 3, hidden_dim, gene_latent_dim, dropout)
        self.kc_g2p = KnowledgeCycleModule(hidden_dim * 3, hidden_dim, protein_dim, dropout)

        self.pred_protein_to_gene_latent = KnowledgeCycleModule(protein_dim, hidden_dim, gene_latent_dim, dropout)
        self.pred_gene_to_protein = KnowledgeCycleModule(gene_dim, hidden_dim, protein_dim, dropout)

    def encode_gene_to_latent(self, gene_full):
        return torch.matmul(
            gene_full - self.gene_pca_mean.unsqueeze(0),
            self.gene_pca_components.T,
        )

    def decode_gene_from_latent(self, gene_latent):
        return torch.matmul(gene_latent, self.gene_pca_components) + self.gene_pca_mean.unsqueeze(0)

    def _graph_smoothness_loss(self, pred, target_type):
        if target_type == "gene" and self.kb_bridge.gene_gene_graph is not None:
            return F.mse_loss(pred, pred @ self.kb_bridge.gene_gene_graph)
        if target_type == "protein" and self.kb_bridge.protein_protein_graph is not None:
            return F.mse_loss(pred, pred @ self.kb_bridge.protein_protein_graph)
        return pred.new_zeros(())

    def _kb_consistency_loss(self, pred, kb_prior):
        pred_n = F.normalize(pred, dim=1)
        kb_n = F.normalize(kb_prior, dim=1)
        return 1.0 - (pred_n * kb_n).sum(dim=1).mean()

    def _pearson_loss(self, pred, target, eps=1e-6):
        if pred is None or target is None or pred.size(0) <= 1:
            return pred.new_zeros(()) if pred is not None else torch.tensor(0.0)
        pred_c = pred - pred.mean(dim=0, keepdim=True)
        target_c = target - target.mean(dim=0, keepdim=True)
        num = (pred_c * target_c).sum(dim=0)
        den = torch.sqrt((pred_c.pow(2).sum(dim=0) + eps) * (target_c.pow(2).sum(dim=0) + eps))
        corr = num / den
        corr = torch.where(torch.isfinite(corr), corr, torch.zeros_like(corr))
        return 1.0 - corr.mean()

    def _neighbor_soft_target(self, target, target_graph, eps=1e-6):
        if target_graph is None:
            return target
        if target_graph.is_sparse:
            g = target_graph.coalesce().float()
            smoothed = torch.sparse.mm(g, target.float())
            row_sum = torch.sparse.sum(g, dim=1).to_dense().unsqueeze(1).clamp_min(eps)
            return smoothed / row_sum
        g = target_graph.float()
        row_sum = g.sum(dim=1, keepdim=True).clamp_min(eps)
        return (g @ target.float()) / row_sum

    def _orthogonality_loss(self, shared, specific):
        s = F.normalize(shared, dim=1)
        p = F.normalize(specific, dim=1)
        return (s * p).sum(dim=1).abs().mean()

    def _obs_hidden(self, obs, obs_type, n, device, slice_id):
        if obs is None:
            return torch.zeros((n, self.hidden_dim), device=device)

        if obs_type == "protein":
            z = self.protein_encoder(obs)
        elif obs_type == "gene":
            z = self.gene_encoder(obs)
        else:
            raise ValueError(f"Unsupported obs_type={obs_type}")

        z = z + 0.05 * self.type_embedding(obs_type, n, device) + 0.05 * self.slice_embedding(slice_id, n, device)
        return z

    def _parse_mode(self, mode):
        if mode == "he_to_gene":
            return "gene", None
        if mode == "he_to_protein":
            return "protein", None
        if mode == "he_protein_to_gene":
            return "gene", "protein"
        if mode == "he_gene_to_protein":
            return "protein", "gene"
        raise ValueError(f"Unsupported mode: {mode}")


    def _backward_cycle(self, pred_full, target_type, obs_type=None):
        """
        Map the predicted target modality back to the observed modality space.
        """
        if target_type == "gene" and obs_type == "protein":
            recon_obs = self.pred_gene_to_protein(pred_full)
            recon_latent = None
            return recon_obs, recon_latent
        if target_type == "protein" and obs_type == "gene":
            recon_latent = self.pred_protein_to_gene_latent(pred_full)
            recon_obs = self.decode_gene_from_latent(recon_latent)
            return recon_obs, recon_latent
        return None, None

    def forward_branch(self, he, adj, target_type, slice_id, obs=None, obs_type=None):
        n = he.size(0)
        dev = he.device

        he_base = self.he_encoder(he)
        he_base = he_base + 0.05 * self.type_embedding(None, n, dev) + 0.05 * self.slice_embedding(slice_id, n, dev)

        he_shared, he_local = self.he_feature_splitter(he_base)
        # Legacy projection-only path retained above:
        # he_shared = self.he_shared_proj(he_base)
        # he_local = self.he_local_proj(he_base)
        obs_hidden = self._obs_hidden(obs, obs_type, n, dev, slice_id)

        kb_out = self.kb_bridge(
            he_shared,
            obs_hidden,
            obs=obs,
            obs_type=obs_type,
            target_type=target_type,
        )

        kb_hidden_for_fusion = kb_out["kb_hidden"]
        residual_mask = kb_out["local_mask"]
        if self.disable_kb_fusion:
            kb_hidden_for_fusion = torch.zeros_like(kb_hidden_for_fusion)
            residual_mask = torch.ones_like(residual_mask)

        fused, gate_info = self.fusion(
            he_shared,
            he_local,
            obs_hidden,
            kb_hidden_for_fusion,
            has_obs=obs is not None,
        )
        z = self.spatial_backbone(fused, adj)
        shared = self.shared_proj(z)
        specific = self.specific_proj(z)
        dgi_loss = self.graph_dgi(shared)

        pred_latent = None
        pred_main = None
        pred_delta = None
        refine_gate = None

        if target_type == "gene":
            pred_latent = self.gene_coarse_head(shared)
            pred_main = self.decode_gene_from_latent(pred_latent)
            pred_delta = self.gene_residual_head(specific)
            if self.disable_soft_gate:
                refine_gate = torch.ones_like(pred_delta)
            else:
                refine_gate = self.gene_soft_gate(torch.cat([shared, specific, kb_hidden_for_fusion], dim=1))
            if self.disable_residual_refine:
                pred_full = pred_main
            else:
                pred_full = pred_main + refine_gate * (pred_delta * residual_mask)
            decoder_aux = self.gene_recon_latent_head(shared)
        elif target_type == "protein":
            pred_main = self.protein_coarse_head(shared)
            pred_delta = self.protein_residual_head(specific)
            if self.disable_soft_gate:
                refine_gate = torch.ones_like(pred_delta)
            else:
                refine_gate = self.protein_soft_gate(torch.cat([shared, specific, kb_hidden_for_fusion], dim=1))
            if self.disable_residual_refine:
                pred_full = pred_main
            else:
                pred_full = pred_main + refine_gate * (pred_delta * residual_mask)
            decoder_aux = self.protein_recon_head(shared)
        else:
            raise ValueError(f"Unsupported target_type={target_type}")

        recon_obs, recon_latent = self._backward_cycle(pred_full, target_type, obs_type)

        return {
            "pred_full": pred_full,
            "pred_main": pred_main,
            "pred_delta": pred_delta,
            "refine_gate": refine_gate,
            "pred_latent": pred_latent,
            "decoder_aux": decoder_aux,
            "recon_obs": recon_obs,
            "recon_latent": recon_latent,
            "he_shared": he_shared,
            "he_local": he_local,
            "obs_hidden": obs_hidden,
            "shared": shared,
            "specific": specific,
            "dgi_loss": dgi_loss,
            "target_type": target_type,
            "obs_type": obs_type,
            "gate_info": gate_info,
            "kb_out": kb_out,
        }

    def _cycle_consistency_loss(self, branch_out, observed_data=None, observed_modality=None):
        pred_full = branch_out["pred_full"]
        loss_cycle_obs = pred_full.new_zeros(())
        recon_obs = branch_out.get("recon_obs", None)

        if observed_modality is None or observed_data is None:
            return loss_cycle_obs, recon_obs

        if observed_modality == "protein" and recon_obs is not None:
            loss_cycle_obs = F.mse_loss(recon_obs, observed_data)
        elif observed_modality == "gene" and recon_obs is not None:
            loss_cycle_obs = F.mse_loss(recon_obs, observed_data)
        return loss_cycle_obs, recon_obs

    def compute_branch_supervised_loss(
        self,
        branch_out,
        target_type,
        target,
        target_latent=None,
        obs=None,
        obs_type=None,
        target_graph=None,
    ):
        pred_full = branch_out["pred_full"]
        pred_latent = branch_out["pred_latent"]
        kb_local = branch_out["kb_out"]["local_prior"]
        shared = branch_out["shared"]
        specific = branch_out["specific"]

        loss_target_hard = F.mse_loss(pred_full, target)
        loss_target_soft = pred_full.new_zeros(())
        if self.target_soft_alpha > 0.0 and target_graph is not None:
            soft_target = self._neighbor_soft_target(target, target_graph)
            loss_target_soft = F.mse_loss(pred_full, soft_target)
        loss_target = (
            (1.0 - self.target_soft_alpha) * loss_target_hard
            + self.target_soft_alpha * loss_target_soft
        )
        loss_pcc = self._pearson_loss(pred_full, target)

        loss_latent = pred_full.new_zeros(())
        if target_type == "gene" and target_latent is not None and pred_latent is not None:
            loss_latent = F.mse_loss(pred_latent, target_latent)

        loss_recon = pred_full.new_zeros(())
        if target_type == "gene" and target_latent is not None:
            loss_recon = F.mse_loss(branch_out["decoder_aux"], target_latent)
        elif target_type == "protein":
            loss_recon = F.mse_loss(branch_out["decoder_aux"], target)

        loss_cycle_obs, recon_obs = self._cycle_consistency_loss(
            branch_out, observed_data=obs, observed_modality=obs_type
        )

        loss_kb = self._kb_consistency_loss(pred_full, kb_local)
        loss_graph = self._graph_smoothness_loss(pred_full, target_type)
        loss_ortho = self._orthogonality_loss(shared, specific)
        loss_dgi = branch_out["dgi_loss"]
        gate_info = branch_out.get("gate_info", {})
        loss_fusion_gate = gate_info.get("gate_reg_loss", pred_full.new_zeros(()))
        if not torch.is_tensor(loss_fusion_gate):
            loss_fusion_gate = pred_full.new_tensor(float(loss_fusion_gate))

        total = (
            self.lambda_target * loss_target
            + self.lambda_pcc * loss_pcc
            + self.lambda_latent * loss_latent
            + self.lambda_recon * loss_recon
            + self.lambda_cycle * loss_cycle_obs
            + self.lambda_kb * loss_kb
            + self.lambda_graph * loss_graph
            + self.lambda_ortho * loss_ortho
            + self.lambda_dgi * loss_dgi
            + self.lambda_fusion_gate * loss_fusion_gate
        )

        return {
            "loss": total,
            "loss_target": loss_target,
            "loss_target_hard": loss_target_hard,
            "loss_target_soft": loss_target_soft,
            "loss_pcc": loss_pcc,
            "loss_latent": loss_latent,
            "loss_recon": loss_recon,
            "loss_cycle_obs": loss_cycle_obs,
            "loss_kb": loss_kb,
            "loss_graph": loss_graph,
            "loss_ortho": loss_ortho,
            "loss_dgi": loss_dgi,
            "loss_fusion_gate": loss_fusion_gate,
            "gate_he_shared_mean": gate_info.get("gate_he_shared_mean", pred_full.new_zeros(())),
            "gate_he_shared_std": gate_info.get("gate_he_shared_std", pred_full.new_zeros(())),
            "gate_he_local_mean": gate_info.get("gate_he_local_mean", pred_full.new_zeros(())),
            "gate_he_local_std": gate_info.get("gate_he_local_std", pred_full.new_zeros(())),
            "gate_obs_mean": gate_info.get("gate_obs_mean", pred_full.new_zeros(())),
            "gate_obs_std": gate_info.get("gate_obs_std", pred_full.new_zeros(())),
            "gate_kb_mean": gate_info.get("gate_kb_mean", pred_full.new_zeros(())),
            "gate_kb_std": gate_info.get("gate_kb_std", pred_full.new_zeros(())),
            "recon_obs": recon_obs,
        }

    def compute_branch_weak_loss(
        self,
        branch_out,
        target_type,
        observed_modality=None,
        observed_data=None,
    ):
        pred_full = branch_out["pred_full"]
        kb_local = branch_out["kb_out"]["local_prior"]
        shared = branch_out["shared"]
        specific = branch_out["specific"]

        loss_kb = self._kb_consistency_loss(pred_full, kb_local)
        loss_graph = self._graph_smoothness_loss(pred_full, target_type)
        loss_ortho = self._orthogonality_loss(shared, specific)
        loss_dgi = branch_out["dgi_loss"]
        loss_cycle_obs, recon_obs = self._cycle_consistency_loss(
            branch_out, observed_data=observed_data, observed_modality=observed_modality
        )
        gate_info = branch_out.get("gate_info", {})
        loss_fusion_gate = gate_info.get("gate_reg_loss", pred_full.new_zeros(()))
        if not torch.is_tensor(loss_fusion_gate):
            loss_fusion_gate = pred_full.new_tensor(float(loss_fusion_gate))

        total = (
            self.lambda_partial_cycle * loss_cycle_obs
            + self.lambda_kb * loss_kb
            + self.lambda_graph * loss_graph
            + self.lambda_ortho * loss_ortho
            + self.lambda_dgi * loss_dgi
            + self.lambda_fusion_gate * loss_fusion_gate
        )

        return {
            "loss": total,
            "loss_kb": loss_kb,
            "loss_graph": loss_graph,
            "loss_ortho": loss_ortho,
            "loss_dgi": loss_dgi,
            "loss_cycle_obs": loss_cycle_obs,
            "loss_fusion_gate": loss_fusion_gate,
            "gate_he_shared_mean": gate_info.get("gate_he_shared_mean", pred_full.new_zeros(())),
            "gate_he_shared_std": gate_info.get("gate_he_shared_std", pred_full.new_zeros(())),
            "gate_he_local_mean": gate_info.get("gate_he_local_mean", pred_full.new_zeros(())),
            "gate_he_local_std": gate_info.get("gate_he_local_std", pred_full.new_zeros(())),
            "gate_obs_mean": gate_info.get("gate_obs_mean", pred_full.new_zeros(())),
            "gate_obs_std": gate_info.get("gate_obs_std", pred_full.new_zeros(())),
            "gate_kb_mean": gate_info.get("gate_kb_mean", pred_full.new_zeros(())),
            "gate_kb_std": gate_info.get("gate_kb_std", pred_full.new_zeros(())),
            "recon_obs": recon_obs,
        }

    def compute_branch_partial_loss(
        self,
        branch_out,
        target_type,
        observed_modality=None,
        observed_data=None,
        observed_latent=None,
        enable_cycle_consistency=True,
    ):
        # Backward compatible wrapper
        return self.compute_branch_weak_loss(
            branch_out=branch_out,
            target_type=target_type,
            observed_modality=observed_modality,
            observed_data=observed_data,
        )

    def compute_alignment_loss(self, out1, out2, max_points=1024):
        loss_he = mmd_rbf(out1["he_shared"], out2["he_shared"], max_points=max_points) + mean_cov_loss(out1["he_shared"], out2["he_shared"])
        loss_shared = mmd_rbf(out1["shared"], out2["shared"], max_points=max_points) + mean_cov_loss(out1["shared"], out2["shared"])
        if self.disable_kb_fusion:
            loss_kb = loss_he.new_zeros(())
        else:
            loss_kb = mmd_rbf(out1["kb_out"]["kb_hidden"], out2["kb_out"]["kb_hidden"], max_points=max_points) + mean_cov_loss(out1["kb_out"]["kb_hidden"], out2["kb_out"]["kb_hidden"])
        return loss_he + self.lambda_bridge_align * (loss_shared + loss_kb)

    @torch.no_grad()
    def infer_single(self, he, adj, mode, slice_id=0, obs=None):
        target_type, obs_type = self._parse_mode(mode)
        self.eval()
        return self.forward_branch(he, adj, target_type, slice_id, obs, obs_type)
