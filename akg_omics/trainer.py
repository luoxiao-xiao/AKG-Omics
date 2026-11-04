"""Main H00 full/partial training loop, with no baseline or ablation branches."""
from pathlib import Path
import gc
import random
import time
import numpy as np
import torch
from .config import H00
from .core import SpatialExFullPartialProtocol
from .data import load_prior
from .io import array_digest, sha256, write_json


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def parameter_hash(model):
    import hashlib
    digest = hashlib.sha256()
    for name, p in sorted(model.named_parameters()):
        digest.update(name.encode())
        digest.update(p.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def make_trainer(data, kb, seed, device, epochs):
    h = H00()
    h.configure_environment()
    seed_everything(seed)
    s, f, p, z = data.spec, data.full, data.test, data.transforms
    trainer = SpatialExFullPartialProtocol(
        target_task=s["model_target"], full_use_obs=s["observed"], partial_use_obs=s["observed"],
        seed=seed, device=device, epochs=epochs, save_path=None,
        lr=h.lr, hidden_dim=h.hidden_dim, num_layers=h.num_layers, dropout=h.dropout, weight_decay=h.weight_decay,
        he_dim=f["he"].shape[1], gene_dim=f["gene"].shape[1], protein_dim=f["other"].shape[1],
        lambda_cycle=h.cycle, lambda_partial_cycle=h.cycle, lambda_kb=h.kb,
        lambda_align=h.alignment, lambda_bridge_align=h.bridge_alignment,
        lambda_full=1.0, lambda_partial=0.0, lambda_target=1.0,
        lambda_pcc=0.0, lambda_latent=0.0, lambda_recon=0.0,
        lambda_graph=0.0, lambda_ortho=0.0, lambda_dgi=0.0, target_soft_alpha=0.0,
        kb_ct_prior_mix=0.7, align_max_points=h.align_max_points, use_amp=False,
        disable_dynamic_retriever=False, disable_kb_fusion=False,
        disable_soft_gate=True, disable_residual_refine=False,
        full_he=f["he"], full_graph=data.train_adj, full_gene=f["gene"],
        full_protein=f["other"], full_gene_latent=f["latent"],
        partial_he=p["he"], partial_graph=data.test_adj,
        partial_gene_obs=p.get("gene"), partial_protein_obs=p.get("other"),
        partial_gene_eval=None, partial_protein_eval=None,
        gene_pca_components=z["gene_components"], gene_pca_mean=z["gene_mean"],
        protein_gene_prior=kb["pg"], gene_protein_prior=kb["gp"],
        gene_gene_kb_graph=kb["gg"], protein_protein_kb_graph=kb["pp"],
        celltype_gene_prior=None, celltype_protein_prior=None,
    )
    model = trainer.model
    model.kb_bridge.dynamic_knowledge_mask = True
    model.kb_bridge.dynamic_residual_mask = True
    model.lambda_fusion_gate = h.gate_regularizer
    model.lambda_kb = h.kb
    assert model.fusion.temperature == h.temperature
    assert model.he_feature_splitter.local_residual_scale == h.he_local_residual_scale
    return trainer


def objective(trainer, target):
    model = trainer.model
    fo, fot = trainer._resolve_obs_config("full")
    po, pot = trainer._resolve_obs_config("partial")
    full = model.forward_branch(trainer.full["he"], trainer.full["adj"], target_type=target,
                                slice_id=trainer.full_slice_id, obs=fo, obs_type=fot)
    supervised = model.compute_branch_supervised_loss(
        full, target_type=target, target=trainer.full[target],
        target_latent=trainer.full["gene_latent"] if target == "gene" else None,
        obs=fo, obs_type=fot, target_graph=trainer.full["adj"])
    partial = model.forward_branch(trainer.partial["he"], trainer.partial["adj"],
                                   target_type=target, slice_id=trainer.partial_slice_id, obs=po, obs_type=pot)
    weak = model.compute_branch_weak_loss(partial, target_type=target, observed_modality=pot, observed_data=po)
    alignment = model.compute_alignment_loss(full, partial, max_points=trainer.align_max_points)
    return trainer.lambda_full * supervised["loss"] + trainer.lambda_partial * weak["loss"] + trainer.lambda_align * alignment


def train(data, prior_path, output, policy, seed=1, device="cuda:0", smoke=False,
          expected_prior=None, threads=4, memory_mib=None, knowledge_context=None):
    h = H00()
    h.configure_environment()
    if policy not in ("rule", "agent"):
        raise ValueError("Only main Rule and Agent versions are supported")
    torch.set_num_threads(threads)
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite a run: {output}")
    output.mkdir(parents=True, exist_ok=True)
    if device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; select --device cpu for a diagnostic")
        if memory_mib:
            index = torch.device(device).index or 0
            torch.cuda.set_per_process_memory_fraction(min(memory_mib * 1048576 / torch.cuda.get_device_properties(index).total_memory, 0.97), index)
    kb = load_prior(prior_path, data, expected_prior, policy=policy)
    epochs = 2 if smoke else h.max_epochs
    start = time.time()
    trainer = make_trainer(data, kb, seed, device, epochs)
    manifest = dict(configuration=h.to_dict(), task=data.spec, policy=policy,
                    seed=seed, device=device, diagnostic_only=smoke,
                    model_sha256=sha256(Path(__file__).parent / "core/model.py"),
                    initial_parameter_sha256=parameter_hash(trainer.model),
                    data_ready_sha256=sha256(data.folder / "READY.json"),
                    source_transform_sha256=sha256(data.folder / "transforms.npz"),
                    prior_sha256=sha256(prior_path),
                    knowledge_selection=kb.get("selection"),
                    selection_replayed=kb.get("selection_replayed", False),
                    actual_llm_called_this_batch=kb.get("actual_llm_called_this_batch", False),
                    response_independence=data.metadata.get("acceptance", {}).get("response_lineage_disjoint"),
                    hidden_target_response_in_training=False,
                    shared_source_response_qualification=data.spec["dataset"] == "coad",
                    transductive_allowed_target_inputs=True,
                    neural_weights_saved=False, restore_best=False)
    if knowledge_context:
        manifest.update(knowledge_context)
    write_json(output / "manifest.json", manifest)
    seed_everything(seed)
    best, best_epoch, history = None, 0, []
    for epoch in range(1, epochs + 1):
        trainer.model.train()
        trainer.optimizer.zero_grad(set_to_none=True)
        loss = objective(trainer, data.spec["model_target"])
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite H00 objective at epoch {epoch}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainer.model.parameters(), 5.0)
        trainer.optimizer.step()
        last = float(loss.detach())
        if best is None or best - last > h.min_delta:
            best, best_epoch = last, epoch
        history.append(dict(epoch=epoch, loss=last, best_epoch=best_epoch))
        if epoch == 1 or epoch % 25 == 0:
            print(f"T{data.spec['task']} {policy} epoch={epoch} loss={last:.7f} best={best_epoch}", flush=True)
        del loss
        if epoch - best_epoch >= h.patience:
            break
    trainer.model.eval()
    with torch.no_grad():
        obs, kind = trainer._resolve_obs_config("partial")
        result = trainer.model.forward_branch(trainer.partial["he"], trainer.partial["adj"],
                 target_type=data.spec["model_target"], slice_id=trainer.partial_slice_id,
                 obs=obs, obs_type=kind)
        prediction = result["pred_full"].cpu().numpy()
    if not np.isfinite(prediction).all():
        raise FloatingPointError("Non-finite predictions")
    frozen_digest = array_digest(prediction)
    del trainer, result, kb
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    # Target response is opened by the evaluator only after prediction is frozen.
    data.predictions_frozen = True
    truth = data.load_truth()
    if truth.shape != prediction.shape:
        raise ValueError("Prediction/response shapes disagree")
    from .metrics_fast import evaluate
    from .core import utils
    metrics = evaluate(truth, prediction, data.eval_adj, utils)
    metrics["global_RMSE"] = float(np.sqrt(np.mean((truth.astype(np.float64) - prediction) ** 2)))
    ys, ps = truth.std(0, dtype=np.float64), prediction.std(0, dtype=np.float64)
    defined = (ys > 1e-12) & (ps > 1e-12)
    pcc = np.full(truth.shape[1], np.nan)
    pcc[defined] = np.mean((truth[:, defined] - truth[:, defined].mean(0)) *
                           (prediction[:, defined] - prediction[:, defined].mean(0)), axis=0) / (ys[defined] * ps[defined])
    np.savez_compressed(output / "feature_metrics.npz", PCC=pcc, PCC_defined=defined,
                         RMSE=np.sqrt(np.mean((truth.astype(np.float64) - prediction) ** 2, axis=0)))
    training = dict(stopped_epoch=epoch, best_epoch=best_epoch, best_training_loss=best,
                    last_training_loss=last, stopping_monitor="training_total_loss", restore_best=False)
    write_json(output / "history.json", history)
    write_json(output / "metrics.json", dict(**manifest, metrics=metrics, training=training,
               predictions_frozen_before_truth=True, predictions_sha256=frozen_digest,
               elapsed_seconds=time.time() - start))
    write_json(output / "SUCCESS.json", dict(metrics=metrics, diagnostic_only=smoke))
    return metrics
