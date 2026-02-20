#!/usr/bin/env python3
"""End-to-end transfer learning pipeline for MORPH.

1. Train MORPH from scratch on a source cell-line (e.g. K562)
2. Take the best validation checkpoint
3. Fine-tune on target control cells (e.g. RPE1)
4. Predict perturbation effects on held-out perturbations in target

Usage (full pipeline — train from scratch, then transfer):

    python morph/run_transfer.py \\
        --config configs/transfer_default.yaml \\
        --embedding_path /path/to/depmap_crispr_gene_effect_processed.pkl \\
        --source_data_path /path/to/k562.h5ad \\
        --target_data_path /path/to/rpe1.h5ad \\
        --output_dir ./transfer_output

Reuse an existing source checkpoint (skip Stage 0):

    python morph/run_transfer.py \\
        --config configs/transfer_default.yaml \\
        --embedding_path /path/to/depmap_crispr_gene_effect_processed.pkl \\
        --source_data_path /path/to/k562.h5ad \\
        --target_data_path /path/to/rpe1.h5ad \\
        --output_dir ./transfer_output \\
        --source_run_dir ./transfer_output/source_training/MORPH_DepMap_GeneEffect_run1740000000
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import random
import sys
import time
from argparse import Namespace
from copy import deepcopy

import numpy as np
import pandas as pd
import scanpy as sc
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, Subset, TensorDataset
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MORPH_DIR = os.path.dirname(os.path.abspath(__file__))
if MORPH_DIR not in sys.path:
    sys.path.insert(0, MORPH_DIR)

from dataset import SCDataset
from train import loss_function, train_validate
from utils import SCDATA_sampler, split_scdata

DEFAULT_HELDOUT_PERTS = [
    "HDAC7", "CNOT3", "POLR1B", "RPL30", "RPL17", "RPS27", "PHB", "ZC3H13",
    "LSM6", "NACA", "YTHDC1", "EIF2S1", "EXOSC2", "RPS5", "RPS15A", "PRIM2",
    "SMG5", "EIF4A3",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_embedding(path: str) -> dict:
    """Load gene embedding dict from .pkl (gene name -> vector)."""
    path = os.path.abspath(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Embedding path not found: {path}")
    with open(path, "rb") as f:
        emb = pickle.load(f)
    if not isinstance(emb, dict):
        raise ValueError("Embedding file must contain a dict (gene name -> vector).")
    logger.info("Loaded embeddings for %d genes from %s", len(emb), path)
    return emb


def load_control_cells(
    adata_path: str,
    fixed_genes: list[str],
    ctrl_label: str = "non-targeting",
) -> np.ndarray:
    """Load, preprocess, gene-align, and return control cells (n_cells x n_genes)."""
    adata = sc.read_h5ad(adata_path)
    logger.info("Raw adata: %s  shape=%s", adata_path, adata.shape)
    if hasattr(adata.X, "toarray"):
        adata.X = adata.X.toarray()
    adata.X = np.asarray(adata.X, dtype=np.float32)

    totals = adata.X.sum(axis=1)
    keep = totals > 0
    if (~keep).any():
        adata = adata[keep].copy()

    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)

    X = np.zeros((adata.n_obs, len(fixed_genes)), dtype=np.float32)
    for j, g in enumerate(fixed_genes):
        if g in adata.var_names:
            X[:, j] = np.asarray(adata[:, g].X.flatten())
    adata = sc.AnnData(X, obs=adata.obs.copy(), var=pd.DataFrame(index=list(fixed_genes)))

    ctrl_mask = adata.obs["gene"] == ctrl_label
    ctrl = np.asarray(adata[ctrl_mask].X, dtype=np.float32)
    logger.info("Control cells: %d x %d", *ctrl.shape)
    return ctrl


# ---------------------------------------------------------------------------
# Stage 0: Train MORPH from scratch on source dataset (or load existing run)
# ---------------------------------------------------------------------------

def run_stage0_source_training(
    *,
    source_run_dir: str | None,
    output_dir: str,
    source_runs_dir: str,
    source_data_path: str,
    gene_embs: dict,
    held_out_perts: list[str],
    representation_type: str,
    batch_size: int,
    n_top_genes: int,
    model_type: str,
    source_name: str,
    cfg: dict,
    device: torch.device,
    seed: int,
    use_wandb: bool,
) -> tuple[str, list[str]]:
    """Run Stage 0: train from scratch or load existing source run.

    Returns:
        (source_run_dir, hvg_genes) for use in Stage 1 and 2.
    """
    if source_run_dir:
        # Reuse existing source checkpoint
        source_run_dir = os.path.abspath(source_run_dir)
        for required in ("best_model_val.pt", "hvg_genes.pkl", "config.json"):
            if not os.path.isfile(os.path.join(source_run_dir, required)):
                raise FileNotFoundError(
                    f"--source_run_dir is missing {required}: {source_run_dir}"
                )
        logger.info("=" * 60)
        logger.info("STAGE 0: SKIPPED — reusing source run %s", source_run_dir)
        logger.info("=" * 60)
        with open(os.path.join(source_run_dir, "hvg_genes.pkl"), "rb") as f:
            hvg_genes = pickle.load(f)
        logger.info("Loaded HVG gene list: %d genes", len(hvg_genes))
        return source_run_dir, hvg_genes

    # Train from scratch
    run_tag = f"{model_type}_{representation_type}_run{int(time.time())}"
    source_run_dir = os.path.join(source_runs_dir, run_tag)
    os.makedirs(source_run_dir, exist_ok=True)

    logger.info("=" * 60)
    logger.info("STAGE 0: Training %s from scratch on source dataset", model_type)
    logger.info("  run dir: %s", source_run_dir)
    logger.info("=" * 60)

    source_dataset = SCDataset(
        base_dir=output_dir,
        adata_path=os.path.abspath(source_data_path),
        leave_out_test_set=held_out_perts,
        representation_type=representation_type,
        gene_embs=gene_embs,
        use_hvg=True,
        n_top_genes=n_top_genes,
        min_counts=batch_size,
        random_seed=seed,
    )
    ptb_leave_out_list = source_dataset.ptb_leave_out_list
    logger.info(
        "Source dataset: %d perturbed cells, %d perturbation targets, %d held-out",
        len(source_dataset), len(source_dataset.ptb_targets), len(ptb_leave_out_list),
    )

    adata_stem = os.path.splitext(os.path.basename(source_data_path))[0]
    hvg_cache_path = os.path.join(
        output_dir, "data", "hvg_cache", f"{adata_stem}_n{n_top_genes}.pkl",
    )
    with open(hvg_cache_path, "rb") as f:
        hvg_genes = pickle.load(f)
    logger.info("HVG gene list: %d genes (from %s)", len(hvg_genes), hvg_cache_path)

    with open(os.path.join(source_run_dir, "hvg_genes.pkl"), "wb") as f:
        pickle.dump(hvg_genes, f, protocol=pickle.HIGHEST_PROTOCOL)

    train_idx, val_idx, infer_idx = split_scdata(
        source_dataset,
        ptb_targets=source_dataset.ptb_targets,
        ptb_leave_out_list=ptb_leave_out_list,
        validation_set_ratio=cfg.get("validation_set_ratio", 0.1),
        validation_ood_ratio=cfg.get("validation_ood_ratio", 0.15),
        batch_size=batch_size,
    )

    dataset_train = Subset(source_dataset, train_idx)
    train_ptb_names = source_dataset.ptb_names[train_idx]
    dataloader_train = DataLoader(
        dataset_train,
        batch_sampler=SCDATA_sampler(dataset_train, batch_size, train_ptb_names),
        num_workers=0,
    )

    dataset_val = Subset(source_dataset, val_idx)
    val_ptb_names = source_dataset.ptb_names[val_idx]
    dataloader_val = DataLoader(
        dataset_val,
        batch_sampler=SCDATA_sampler(dataset_val, batch_size, val_ptb_names),
        num_workers=0,
    )

    dim = source_dataset[0][0].shape[0]
    cdim = source_dataset[0][2].shape[0]

    opts = Namespace(
        mode="train",
        lr=cfg.get("lr", 1e-4),
        grad_clip=True,
        kernel_num=cfg.get("kernel_num", 10),
        hidden_dim=cfg.get("hidden_dim", 128),
        matched_IO=False,
        base_dir=None,
        seed=seed,
        validation_set_ratio=cfg.get("validation_set_ratio", 0.1),
        validation_ood_ratio=cfg.get("validation_ood_ratio", 0.15),
        latdim_ctrl=cfg.get("latdim_ctrl", 50),
        latdim_ptb=cfg.get("latdim_ptb", 50),
        geneset_num=cfg.get("geneset_num", 50),
        geneset_dim=cfg.get("geneset_dim", 50),
        batch_size=batch_size,
        tolerance_epochs=cfg.get("tolerance_epochs", 20),
        MMD_sigma=cfg.get("MMD_sigma", 1500),
        mxAlpha=cfg.get("mxAlpha", 1),
        mxBeta=cfg.get("mxBeta", 2),
        Gamma1=cfg.get("Gamma1", 0.5),
        Gamma2=cfg.get("Gamma2", 1),
        modality=cfg.get("modality", "rna"),
        dataset_name=source_name,
        leave_out_test_set_id="transfer",
        label=representation_type,
        null_label=cfg.get("null_label", "zeros"),
        dim=dim,
        cdim=cdim,
        cdim_2=None,
        cdim_3=None,
        model=model_type,
        epochs=cfg.get("epochs", 100),
        ptb_leave_out_list=ptb_leave_out_list,
    )

    with open(os.path.join(source_run_dir, "config.json"), "w") as f:
        json.dump(opts.__dict__, f, indent=4)
    with open(os.path.join(source_run_dir, "split_idx.pkl"), "wb") as f:
        pickle.dump({"train_idx": train_idx, "val_idx": val_idx, "infer_idx": infer_idx}, f)

    logger.info("Starting source training (%d epochs, lr=%.1e, bs=%d)", opts.epochs, opts.lr, batch_size)
    train_validate(
        dataloader_train,
        None,
        dataloader_val,
        opts,
        device,
        source_run_dir,
        model_type,
        log=use_wandb,
    )
    logger.info("Source training complete. Checkpoints in %s", source_run_dir)
    return source_run_dir, hvg_genes


# ---------------------------------------------------------------------------
# Stage 1: Fine-tune pretrained model on target control cells
# ---------------------------------------------------------------------------

def finetune_on_controls(
    model: nn.Module,
    control_cells: np.ndarray,
    device: torch.device,
    config: dict,
    *,
    epochs: int = 50,
    lr: float = 1e-4,
    batch_size: int = 256,
    beta_max: float = 1.0,
    savedir: str | None = None,
) -> nn.Module:
    """Fine-tune on control cells (reconstruction + KL)."""
    model.train()
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    ctrl_tensor = torch.from_numpy(control_cells).double()
    ctrl_loader = DataLoader(
        TensorDataset(ctrl_tensor),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )

    gamma1 = config.get("Gamma1", 0.0)
    gamma2 = config.get("Gamma2", 1.0)
    mmd_sigma = config.get("MMD_sigma", 1500)
    kernel_num = config.get("kernel_num", 10)
    c_dim = config["cdim"]

    beta_schedule = torch.zeros(epochs)
    warmup = min(5, max(1, epochs // 5))
    beta_schedule[:warmup] = 0
    if epochs > warmup:
        beta_schedule[warmup:] = torch.linspace(0, beta_max, epochs - warmup)

    best_model = deepcopy(model)
    best_recon = float("inf")

    logger.info(
        "Fine-tuning: %d control cells, %d epochs, lr=%.1e, bs=%d",
        control_cells.shape[0], epochs, lr, batch_size,
    )

    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_recon = 0.0
        epoch_kl = 0.0
        n_batches = 0

        for (x_batch,) in tqdm(ctrl_loader, desc=f"FT {epoch+1}/{epochs}", leave=False):
            x = x_batch.to(device)
            bs = x.size(0)
            c_1 = torch.zeros(bs, c_dim, dtype=torch.float64, device=device)
            c_2 = torch.full((bs, c_dim), float("nan"), dtype=torch.float64, device=device)

            optimizer.zero_grad()
            y_hat, x_recon, mu, logvar = model(x, c_1, c_2)

            _, recon_loss, kl_loss = loss_function(
                y_hat=None, y=None,
                x_recon=x_recon, x=x,
                mu=mu, logvar=logvar,
                MMD_sigma=mmd_sigma, kernel_num=kernel_num,
                gamma1=gamma1, gamma2=gamma2,
            )

            loss = recon_loss + beta_schedule[epoch] * kl_loss
            loss.backward()

            for param in model.parameters():
                if param.grad is not None:
                    param.grad.data.clamp_(-0.5, 0.5)

            optimizer.step()

            epoch_loss += loss.item()
            epoch_recon += (recon_loss.item() if isinstance(recon_loss, torch.Tensor) else recon_loss)
            epoch_kl += (kl_loss.item() if isinstance(kl_loss, torch.Tensor) else kl_loss)
            n_batches += 1

        avg_loss = epoch_loss / n_batches
        avg_recon = epoch_recon / n_batches
        avg_kl = epoch_kl / n_batches
        logger.info(
            "Epoch %d/%d: loss=%.6f  recon=%.6f  kl=%.6f  beta=%.4f",
            epoch + 1, epochs, avg_loss, avg_recon, avg_kl, beta_schedule[epoch].item(),
        )

        if avg_recon < best_recon:
            best_recon = avg_recon
            best_model = deepcopy(model)
            if savedir:
                torch.save(best_model, os.path.join(savedir, "best_model_ft.pt"))

    if savedir:
        torch.save(model, os.path.join(savedir, "last_model_ft.pt"))

    logger.info("Fine-tuning done. Best recon loss: %.6f", best_recon)
    return best_model


# ---------------------------------------------------------------------------
# Stage 2: Predictions on held-out perturbations
# ---------------------------------------------------------------------------

def generate_predictions_on_heldout(
    model: nn.Module,
    dataset: SCDataset,
    held_out_perts: list[str],
    device: torch.device,
    batch_size: int = 32,
) -> dict[str, np.ndarray]:
    """Run inference on held-out perturbation cells.

    Returns:
        Dictionary with control_expression, perturbed_expression_ground_truth,
        perturbed_expression_predicted, perturbation_name_per_cell.
    """
    model.eval()
    model.to(device)

    infer_idx = np.where(np.isin(dataset.ptb_names, held_out_perts))[0]
    if len(infer_idx) == 0:
        raise ValueError(f"No cells found for held-out perts: {held_out_perts}")

    dataset_infer = Subset(dataset, infer_idx)
    ptb_names_batch = dataset.ptb_names[infer_idx]
    dataloader = DataLoader(
        dataset_infer,
        batch_sampler=SCDATA_sampler(dataset_infer, batch_size, ptb_names_batch),
        num_workers=0,
    )

    control_list = []
    perturbed_true_list = []
    perturbed_pred_list = []
    pert_name_list = []

    for batch in tqdm(dataloader, desc="Evaluating held-out perts"):
        control_expr = batch[0].to(device)
        perturbed_expr_true = batch[1]
        cond_embed_1 = batch[2].to(device)
        cond_embed_2 = batch[3].to(device)
        pert_name = batch[4]

        with torch.no_grad():
            perturbed_expr_pred, _, _, _ = model(control_expr, cond_embed_1, cond_embed_2)

        control_list.append(control_expr.cpu().numpy())
        perturbed_true_list.append(perturbed_expr_true.numpy())
        perturbed_pred_list.append(perturbed_expr_pred.cpu().numpy())
        pert_name_list.append(np.atleast_1d(np.asarray(pert_name)))

    return {
        "control_expression": np.vstack(control_list),
        "perturbed_expression_ground_truth": np.vstack(perturbed_true_list),
        "perturbed_expression_predicted": np.vstack(perturbed_pred_list),
        "perturbation_name_per_cell": np.concatenate(pert_name_list),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="End-to-end MORPH transfer: train on source, fine-tune + predict on target.",
    )
    parser.add_argument("--config", type=str, required=True,
                        help="Path to YAML config (see configs/transfer_default.yaml)")
    parser.add_argument("--embedding_path", type=str, required=True,
                        help="Path to gene embedding .pkl (e.g. depmap_crispr_gene_effect_processed.pkl)")
    parser.add_argument("--source_data_path", type=str, required=True,
                        help="Path to source cell-line h5ad (e.g. K562)")
    parser.add_argument("--target_data_path", type=str, required=True,
                        help="Path to target cell-line h5ad (e.g. RPE1)")
    parser.add_argument("--output_dir", type=str, default="./transfer_output",
                        help="Output directory (default: ./transfer_output)")
    parser.add_argument("--source_run_dir", type=str, default=None,
                        help="Path to existing source training run dir to skip Stage 0. "
                             "Must contain best_model_val.pt, hvg_genes.pkl, config.json.")
    parser.add_argument("--no_wandb", action="store_true",
                        help="Disable wandb logging for source training")
    args = parser.parse_args()

    # ---- Load config ----
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device(cfg.get("device", "cuda:0"))
    seed = cfg.get("seed", 12)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    output_dir = os.path.abspath(args.output_dir)
    source_runs_dir = os.path.join(output_dir, "source_training")
    transfer_dir = os.path.join(output_dir, "transfer")
    os.makedirs(source_runs_dir, exist_ok=True)
    os.makedirs(transfer_dir, exist_ok=True)

    gene_embs = load_embedding(args.embedding_path)
    held_out_perts: list[str] = cfg.get("held_out_perts", DEFAULT_HELDOUT_PERTS)
    representation_type: str = cfg["representation_type"]
    batch_size: int = cfg.get("batch_size", 32)
    n_top_genes: int = cfg.get("n_top_genes", 5000)
    model_type: str = cfg.get("model", "MORPH")
    source_name = os.path.splitext(os.path.basename(args.source_data_path))[0]

    source_run_dir, hvg_genes = run_stage0_source_training(
        source_run_dir=args.source_run_dir,
        output_dir=output_dir,
        source_runs_dir=source_runs_dir,
        source_data_path=args.source_data_path,
        gene_embs=gene_embs,
        held_out_perts=held_out_perts,
        representation_type=representation_type,
        batch_size=batch_size,
        n_top_genes=n_top_genes,
        model_type=model_type,
        source_name=source_name,
        cfg=cfg,
        device=device,
        seed=seed,
        use_wandb=not args.no_wandb,
    )

    # -----------------------------------------------------------------------
    # STAGE 1 — Fine-tune best source checkpoint on target control cells
    # -----------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("STAGE 1: Fine-tuning on target control cells")
    logger.info("=" * 60)

    best_model_path = os.path.join(source_run_dir, "best_model_val.pt")
    model = torch.load(best_model_path, map_location=device, weights_only=False)
    model = model.to(device)
    logger.info(
        "Loaded best source checkpoint: %s (%s params)",
        best_model_path, f"{sum(p.numel() for p in model.parameters()):,}",
    )

    with open(os.path.join(source_run_dir, "config.json")) as f:
        source_config = json.load(f)

    control_cells = load_control_cells(args.target_data_path, hvg_genes)
    model = finetune_on_controls(
        model,
        control_cells,
        device,
        source_config,
        epochs=cfg.get("ft_epochs", 50),
        lr=cfg.get("ft_lr", 1e-4),
        batch_size=cfg.get("ft_batch_size", 256),
        savedir=transfer_dir,
    )

    target_name = os.path.splitext(os.path.basename(args.target_data_path))[0]
    torch.save(model, os.path.join(transfer_dir, f"finetuned_{target_name}_model.pt"))
    logger.info("Saved fine-tuned model to %s", transfer_dir)

    # -----------------------------------------------------------------------
    # STAGE 2 — Predictions on target held-out perturbations
    # -----------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("STAGE 2: Predicting on target held-out perturbations")
    logger.info("=" * 60)

    target_dataset = SCDataset(
        adata_path=os.path.abspath(args.target_data_path),
        leave_out_test_set=held_out_perts,
        fixed_genes=hvg_genes,
        representation_type=representation_type,
        gene_embs=gene_embs,
        min_counts=32,
        random_seed=seed,
    )
    actual_held_out = target_dataset.ptb_leave_out_list
    logger.info("Evaluating on %d held-out perts: %s", len(actual_held_out), actual_held_out)

    eval_result = generate_predictions_on_heldout(
        model, target_dataset, actual_held_out, device, batch_size=batch_size,
    )

    pert_name_per_cell = eval_result["perturbation_name_per_cell"]
    perturbed_true = eval_result["perturbed_expression_ground_truth"]
    perturbed_pred = eval_result["perturbed_expression_predicted"]
    control_cells_eval = np.asarray(target_dataset.ctrl_samples, dtype=np.float32)

    predictions: dict[str, dict[str, np.ndarray]] = {}
    for pert in sorted(set(pert_name_per_cell)):
        mask = pert_name_per_cell == pert
        predictions[pert] = {
            "control": control_cells_eval,
            "true": np.asarray(perturbed_true[mask], dtype=np.float32),
            "predicted": np.asarray(perturbed_pred[mask], dtype=np.float32),
        }

    out_path = os.path.join(transfer_dir, "predictions.pkl")
    payload = {
        "predictions": predictions,
        "gene_names": hvg_genes,
        "dataset_name": target_name,
        "source_dataset": source_name,
        "cell_type": target_name,
        "is_log_normalized": True,
        "morph_run_dir": transfer_dir,
        "source_run_dir": source_run_dir,
    }
    with open(out_path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info("Saved predictions to %s", out_path)

    transfer_config = {
        **source_config,
        "source_data_path": os.path.abspath(args.source_data_path),
        "target_data_path": os.path.abspath(args.target_data_path),
        "embedding_path": os.path.abspath(args.embedding_path),
        "leave_out_test_set": actual_held_out,
        "ft_epochs": cfg.get("ft_epochs", 50),
        "ft_lr": cfg.get("ft_lr", 1e-4),
        "source_run_dir": source_run_dir,
        "hvg_genes_path": os.path.join(source_run_dir, "hvg_genes.pkl"),
    }
    with open(os.path.join(transfer_dir, "config.json"), "w") as f:
        json.dump(transfer_config, f, indent=4)

    logger.info("=" * 60)
    logger.info("Done. Source: %s | Transfer: %s", source_run_dir, transfer_dir)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
