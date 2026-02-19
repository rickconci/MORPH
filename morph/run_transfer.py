#!/usr/bin/env python3
"""Transfer learning: fine-tune a pretrained MORPH model on control cells from
a target cell-line dataset, then evaluate on held-out perturbations.

Standalone usage:

    python morph/run_transfer.py \\
        --pretrained_model transfer_learning/replogle_gwps_trained_model_large/model.pt \\
        --pretrained_hvg_gene_list path/to/pretrained/hvg_genes.pkl \\
        --data_path path/to/rpe1_dataset.h5ad \\
        --output_dir transfer_learning/replogle_gwps_trained_model_large/fine_tuned \\
        --device cuda:0 \\
        --ft_epochs 50 \\
        --ft_lr 1e-4

Held-out perturbations default to the standard RPE1 essential test set; override
with --held_out_perts (comma-separated list).
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import pickle
import random
import sys

import numpy as np
import pandas as pd
import scanpy as sc
import torch
import torch.nn as nn
from copy import deepcopy
from torch.utils.data import DataLoader, Subset, TensorDataset
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MORPH_DIR = os.path.dirname(os.path.abspath(__file__))
if MORPH_DIR not in sys.path:
    sys.path.insert(0, MORPH_DIR)

from dataset import SCDataset
from train import loss_function
from utils import SCDATA_sampler

# Default held-out perturbations (RPE1 essential test set; same as replogie_RPE1_essential_splits.csv).
DEFAULT_HELDOUT_PERTS = [
    "HDAC7", "CNOT3", "POLR1B", "RPL30", "RPL17", "RPS27", "PHB", "ZC3H13",
    "LSM6", "NACA", "YTHDC1", "EIF2S1", "EXOSC2", "RPS5", "RPS15A", "PRIM2",
    "SMG5", "EIF4A3",
]


def load_gene_list(path: str) -> list[str]:
    """Load gene list from a .pkl (list of str)."""
    path = os.path.abspath(path)
    with open(path, "rb") as f:
        genes = pickle.load(f)
    return genes

def parse_heldout_perts(value: str) -> list[str]:
    """Parse --held_out_perts: comma-separated list."""
    value = value.strip()
    if not value:
        return DEFAULT_HELDOUT_PERTS
    return [p.strip() for p in value.split(",") if p.strip()]


def load_embedding(path: str) -> dict:
    """Load gene embedding dict from a .pkl file (gene name -> vector)."""
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
    adata = sc.read_h5ad(adata_path) # target cell line adata
    logger.info("Raw adata: %s  shape=%s", adata_path, adata.shape)
    adata.X = np.asarray(adata.X, dtype=np.float32)

    # Remove genes with zero expression in all cells
    totals = adata.X.sum(axis=1)
    keep = totals > 0
    if (~keep).any():
        adata = adata[keep].copy()

    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)

    # Subset to HVG gene set from pretrained model and extract control cells
    X = np.zeros((adata.n_obs, len(fixed_genes)), dtype=np.float32)
    for j, g in enumerate(fixed_genes):
        if g in adata.var_names:
            X[:, j] = np.asarray(adata[:, g].X.flatten())
    adata = sc.AnnData(X, obs=adata.obs.copy(), var=pd.DataFrame(index=list(fixed_genes)))

    ctrl_mask = adata.obs["gene"] == ctrl_label
    ctrl = np.asarray(adata[ctrl_mask].X, dtype=np.float32)
    logger.info("Control cells: %d x %d", *ctrl.shape)
    return ctrl


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
# Evaluation on held-out perturbations
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
        Dictionary with:
        - control_expression: (n_cells, n_genes) control input per cell
        - perturbed_expression_ground_truth: (n_cells, n_genes) observed perturbed expression
        - perturbed_expression_predicted: (n_cells, n_genes) model prediction
        - perturbation_name_per_cell: (n_cells,) which perturbation each cell belongs to
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
        control_expr = batch[0].to(device)           # control input
        perturbed_expr_true = batch[1]               # ground-truth perturbed expression
        cond_embed_1 = batch[2].to(device)
        cond_embed_2 = batch[3].to(device)
        pert_name = batch[4]                         # perturbation id per sample

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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Transfer learning: fine-tune pretrained MORPH on target control cells, then evaluate on held-out perturbations.",
    )
    parser.add_argument("--pretrained_model", type=str, required=True,
                        help="Path to pretrained model.pt (config.json expected in same directory)")
    parser.add_argument("--pretrained_hvg_gene_list", type=str, required=True,
                        help="Path to pretrained HVG gene list (.pkl or .h5ad)")
    parser.add_argument("--embedding_path", type=str, required=True,
                        help="Path to gene embedding .pkl (dict: gene name -> vector), e.g. DepMap GeneEffect.")
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to target cell-line h5ad (e.g. RPE1 dataset)")
    parser.add_argument("--held_out_perts", type=str, default="",
                        help="Comma-separated list of held-out perturbations, or path to CSV with 'test_set' column. Default: built-in RPE1 essential set.")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save fine-tuned model and predictions")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--ft_epochs", type=int, default=50, help="Fine-tuning epochs")
    parser.add_argument("--ft_lr", type=float, default=1e-4, help="Fine-tuning learning rate")
    parser.add_argument("--seed", type=int, default=12)
    args = parser.parse_args()

    # ----- Setup -----
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # PART 1: fine-tune on target control cells

    # Load pretrained config
    pretrained_dir = os.path.dirname(os.path.abspath(args.pretrained_model))
    config_path = os.path.join(pretrained_dir, "config.json")
    with open(config_path) as f:
        config = json.load(f)
    logger.info("Pretrained: dataset=%s  model=%s  label=%s  dim=%s",
                config.get("dataset_name"), config.get("model"), config.get("label"), config.get("dim"))

    # Gene list (must match pretrained dim)
    fixed_genes = load_gene_list(args.pretrained_hvg_gene_list)

    # Held-out perturbations
    held_out_perts = parse_heldout_perts(args.held_out_perts)
    logger.info("Held-out perturbations: %d -> %s", len(held_out_perts), held_out_perts)

    # Load pretrained model
    logger.info("Loading pretrained model: %s", args.pretrained_model)
    model = torch.load(args.pretrained_model, map_location=device, weights_only=False)
    model = model.to(device)
    logger.info("Model params: %s", f"{sum(p.numel() for p in model.parameters()):,}")

    # Fine-tune on target control cells
    control_cells = load_control_cells(args.data_path, fixed_genes)
    model = finetune_on_controls(
        model, control_cells, device, config,
        epochs=args.ft_epochs, lr=args.ft_lr,
        savedir=args.output_dir,
    )
    cell_line_name = args.data_path.split("/")[-1].split(".")[0]
    torch.save(model, os.path.join(args.output_dir, f"finetuned_{cell_line_name}_model.pt", ))
    logger.info("Saved fine-tuned model to %s/model.pt", args.output_dir)

    ## PART 2: predictions on held-out perturbations

    gene_embs = load_embedding(args.embedding_path)
    representation_type = config.get("label")
    logger.info("Building target dataset: %s, fixed_genes (%d), representation=%s", args.data_path, len(fixed_genes), representation_type)
    dataset = SCDataset(
        adata_path=os.path.abspath(args.data_path),
        leave_out_test_set=held_out_perts,
        fixed_genes=fixed_genes,
        representation_type=representation_type,
        gene_embs=gene_embs,
        min_counts=32,
        random_seed=args.seed,
    )
    actual_held_out = dataset.ptb_leave_out_list
    logger.info("Evaluating on %d held-out perts: %s", len(actual_held_out), actual_held_out)

    # Run inference on held-out perturbations only
    eval_result = generate_predictions_on_heldout(
        model, dataset, actual_held_out, device, batch_size=32,
    )
    pert_name_per_cell = eval_result["perturbation_name_per_cell"]
    perturbed_true = eval_result["perturbed_expression_ground_truth"]
    perturbed_pred = eval_result["perturbed_expression_predicted"]
    control_cells_eval = np.asarray(dataset.ctrl_samples, dtype=np.float32)

    predictions: dict[str, dict[str, np.ndarray]] = {}
    for pert in sorted(set(pert_name_per_cell)):
        mask = pert_name_per_cell == pert
        predictions[pert] = {
            "control": control_cells_eval,
            "true": np.asarray(perturbed_true[mask], dtype=np.float32),
            "predicted": np.asarray(perturbed_pred[mask], dtype=np.float32),
        }

    dataset_name = os.path.splitext(os.path.basename(args.data_path))[0]
    out_path = os.path.join(args.output_dir, "predictions.pkl")
    payload = {
        "predictions": predictions,
        "gene_names": fixed_genes,
        "dataset_name": dataset_name,
        "cell_type": "RPE1",
        "is_log_normalized": True,
        "morph_run_dir": args.output_dir,
        "model_name": "model.pt",
        "source_dataset": config.get("dataset_name"),
        "pretrained_model": args.pretrained_model,
    }
    with open(out_path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info("Saved predictions to %s", out_path)

    # Save minimal transfer config (for reference)
    transfer_config = {
        **config,
        "adata_path": os.path.abspath(args.data_path),
        "embedding_path": os.path.abspath(args.embedding_path),
        "leave_out_test_set": actual_held_out,
        "ptb_leave_out_list": actual_held_out,
        "fixed_genes_path": "fixed_genes.pkl",
        "ft_epochs": args.ft_epochs,
        "ft_lr": args.ft_lr,
    }
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(transfer_config, f, indent=4)
    logger.info("Done.")

