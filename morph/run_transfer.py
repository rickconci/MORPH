#!/usr/bin/env python3
"""Transfer learning: fine-tune a pretrained MORPH model on control cells from
a new cell line, then evaluate on held-out perturbations.

Protocol (from the MORPH paper, Section "Transfer of perturbation effects
across cell lines"):

    1. Train MORPH on all perturbation data from cell line 1 (already done).
    2. Fine-tune f1 on control cells from cell line 2 by minimising the
       reconstruction loss only.  This aligns the latent space to cell line 2.
    3. Predict perturbation effects in cell line 2 using f1→2.

Usage (from repo root, or set MORPH_REPO_DIR in .env):
    python morph/run_transfer.py \
        --pretrained_model transfer_learning/replogle_gwps_trained_model_large/model.pt \
        --splits_csv data/replogie_RPE1_essential_splits.csv \
        --output_dir result/transfer/k562_gwps_to_rpe1_large \
        --device cuda:0 \
        --ft_epochs 50 \
        --ft_lr 1e-4
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
import time

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

from config import get_repo_dir, resolve_scdata_paths_df
from dataset import SCDataset
from train import loss_function
from utils import SCDATA_sampler, split_scdata


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cell_type_from_data(data: str) -> str:
    m = {
        "norman_k562_essential": "K562",
        "replogle_rpe1_hvg": "RPE1",
        "replogle_k562_hvg": "K562",
        "replogie_RPE1_essential": "RPE1",
    }
    return m.get(data, data)


def parse_splits_csv(path: str) -> tuple[str, str, list[str]]:
    """Read a MORPH splits CSV → (dataset_name, cell_type, perts)."""
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        row = next(reader)
    dataset_name = row["data"]
    cell_type = row.get("cell_type", "").strip() or _cell_type_from_data(dataset_name)
    perts = [p.strip() for p in row["test_set"].split(",") if p.strip()]
    return dataset_name, cell_type, perts


def load_gene_list_from_adata(adata_path: str) -> list[str]:
    """Load gene names from a (possibly HVG-prefiltered) h5ad file."""
    adata = sc.read_h5ad(adata_path)
    genes = adata.var_names.tolist()
    logger.info("Loaded %d gene names from %s", len(genes), adata_path)
    del adata
    return genes


def load_control_cells(
    adata_path: str,
    fixed_genes: list[str],
    ctrl_label: str = "non-targeting",
) -> np.ndarray:
    """Load, preprocess, gene-align, and extract control cells."""
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
    logger.info("Control cells: %d × %d", *ctrl.shape)
    return ctrl


# ---------------------------------------------------------------------------
# Fine-tuning loop (control cells only → reconstruction + KL)
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
    grad_clip: bool = True,
    savedir: str | None = None,
    use_wandb: bool = False,
) -> nn.Module:
    """Fine-tune on control cells, minimising reconstruction loss + KL."""
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
        "Fine-tuning: %d control cells, %d epochs, lr=%.1e, bs=%d, gamma1=%.2f, gamma2=%.2f",
        control_cells.shape[0], epochs, lr, batch_size, gamma1, gamma2,
    )

    if use_wandb:
        import wandb
        wandb.define_metric("ft_epoch")
        wandb.define_metric("ft/*", step_metric="ft_epoch")

    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_recon = 0.0
        epoch_kl = 0.0
        n_batches = 0

        for (x_batch,) in tqdm(ctrl_loader, desc=f"FT epoch {epoch+1}/{epochs}", leave=False):
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

            if grad_clip:
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

        if use_wandb:
            import wandb
            wandb.log({
                "ft_epoch": epoch,
                "ft/loss": avg_loss,
                "ft/recon": avg_recon,
                "ft/kl": avg_kl,
                "ft/beta": beta_schedule[epoch].item(),
            })

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

def evaluate_on_heldout(
    model: nn.Module,
    dataset: SCDataset,
    held_out_perts: list[str],
    device: torch.device,
    batch_size: int = 32,
) -> dict:
    """Run inference on held-out perturbation cells and return result dict."""
    model.eval()
    model.to(device)

    infer_idx = np.where(np.isin(dataset.ptb_names, held_out_perts))[0]
    if len(infer_idx) == 0:
        raise ValueError(f"No cells found for held-out perts: {held_out_perts}")

    dataset_infer = Subset(dataset, infer_idx)
    ptb_names = dataset.ptb_names[infer_idx]
    dataloader = DataLoader(
        dataset_infer,
        batch_sampler=SCDATA_sampler(dataset_infer, batch_size, ptb_names),
        num_workers=0,
    )

    gt_y_all, pred_y_all, gt_C_y_all, gt_x_all = [], [], [], []

    for X in tqdm(dataloader, desc="Evaluating held-out perts"):
        x = X[0].to(device)
        y = X[1]
        c_1 = X[2].to(device)
        c_2 = X[3].to(device)
        C_y = X[4]

        with torch.no_grad():
            y_hat, _, _, _ = model(x, c_1, c_2)

        gt_x_all.append(x.cpu().numpy())
        gt_y_all.append(y.numpy())
        pred_y_all.append(y_hat.cpu().numpy())
        gt_C_y_all.append(np.array(C_y))

    return {
        "gt_x": np.vstack(gt_x_all),
        "gt_y": np.vstack(gt_y_all),
        "pred_y_1": np.vstack(pred_y_all),
        "gt_C_y": np.vstack(gt_C_y_all),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Transfer learning: fine-tune pretrained MORPH on new cell-line controls, then evaluate.",
    )
    parser.add_argument("--pretrained_model", type=str, required=True,
                        help="Path to pretrained model.pt (config.json expected in same directory)")
    parser.add_argument("--splits_csv", type=str, required=True,
                        help="Splits CSV for the target cell line (columns: data, test_set_id, test_set, note)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save fine-tuned model + predictions")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=12)

    ft = parser.add_argument_group("fine-tuning")
    ft.add_argument("--ft_epochs", type=int, default=50, help="Fine-tuning epochs (default: 50)")
    ft.add_argument("--ft_lr", type=float, default=1e-4, help="Fine-tuning learning rate (default: 1e-4)")
    ft.add_argument("--ft_batch_size", type=int, default=256, help="Batch size for control-cell fine-tuning")
    ft.add_argument("--ft_beta", type=float, default=1.0, help="Max KL weight during fine-tuning")
    ft.add_argument("--no_finetune", action="store_true",
                    help="Skip fine-tuning (zero-shot baseline: evaluate pretrained model directly)")

    ev = parser.add_argument_group("evaluation")
    ev.add_argument("--eval_batch_size", type=int, default=32)

    parser.add_argument("--wandb", action="store_true", help="Enable W&B logging")
    args = parser.parse_args()

    # ----- Setup -----
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    base_dir = get_repo_dir()
    logger.info("MORPH base_dir: %s", base_dir)

    # ----- Load pretrained config -----
    pretrained_dir = os.path.dirname(os.path.abspath(args.pretrained_model))
    with open(os.path.join(pretrained_dir, "config.json")) as f:
        config = json.load(f)
    logger.info(
        "Pretrained model: dataset=%s  model=%s  label=%s  dim=%s",
        config["dataset_name"], config["model"], config["label"], config["dim"],
    )

    # ----- Parse target cell-line splits CSV -----
    target_dataset_name, target_cell_type, held_out_perts = parse_splits_csv(args.splits_csv)
    logger.info(
        "Target: dataset=%s  cell_type=%s  %d held-out perts",
        target_dataset_name, target_cell_type, len(held_out_perts),
    )
    logger.info("Held-out perts: %s", held_out_perts)

    # ----- Resolve adata paths (relative to MORPH_DATA_ROOT from .env) -----
    scdata_df = pd.read_csv(os.path.join(base_dir, "data", "scdata_file_path.csv"))
    scdata_df = resolve_scdata_paths_df(scdata_df)

    source_row = scdata_df[scdata_df["dataset"] == config["dataset_name"]]
    if source_row.empty:
        raise FileNotFoundError(f"Source dataset '{config['dataset_name']}' not in scdata_file_path.csv")
    source_adata_path = source_row["file_path"].values[0]

    target_row = scdata_df[scdata_df["dataset"] == target_dataset_name]
    if target_row.empty:
        raise FileNotFoundError(f"Target dataset '{target_dataset_name}' not in scdata_file_path.csv")
    target_adata_path = target_row["file_path"].values[0]
    logger.info("Source adata: %s", source_adata_path)
    logger.info("Target adata: %s", target_adata_path)

    # ----- Step 1: get gene list from the source (training) adata -----
    fixed_genes = load_gene_list_from_adata(source_adata_path)
    assert len(fixed_genes) == config["dim"], (
        f"Gene count mismatch: adata has {len(fixed_genes)}, config says dim={config['dim']}"
    )

    # ----- Step 2: load pretrained model -----
    logger.info("Loading pretrained model: %s", args.pretrained_model)
    model = torch.load(args.pretrained_model, map_location=device, weights_only=False)
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info("Model loaded: %s params", f"{n_params:,}")

    # ----- Step 3: fine-tune on target control cells -----
    if not args.no_finetune:
        control_cells = load_control_cells(target_adata_path, fixed_genes)

        if args.wandb:
            import wandb
            run_name = f"transfer_{config['dataset_name']}_to_{target_dataset_name}"
            wandb.init(project=f"MORPH_transfer_{target_dataset_name}", name=run_name)

        model = finetune_on_controls(
            model,
            control_cells,
            device,
            config,
            epochs=args.ft_epochs,
            lr=args.ft_lr,
            batch_size=args.ft_batch_size,
            beta_max=args.ft_beta,
            grad_clip=True,
            savedir=args.output_dir,
            use_wandb=args.wandb,
        )
        torch.save(model, os.path.join(args.output_dir, "model.pt"))
        logger.info("Saved fine-tuned model to %s/model.pt", args.output_dir)
    else:
        logger.info("Skipping fine-tuning (--no_finetune). Running zero-shot evaluation.")

    # ----- Step 4: save fixed_genes.pkl -----
    fg_pkl_path = os.path.join(args.output_dir, "fixed_genes.pkl")
    with open(fg_pkl_path, "wb") as f:
        pickle.dump(fixed_genes, f, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info("Saved fixed_genes.pkl (%d genes)", len(fixed_genes))

    # ----- Step 5: build target dataset + compute split indices -----
    logger.info("Building target dataset with fixed_genes (%d genes)...", len(fixed_genes))
    dataset = SCDataset(
        base_dir=base_dir,
        dataset_name=target_dataset_name,
        adata_path=target_adata_path,
        leave_out_test_set=held_out_perts,
        representation_type=config["label"],
        representation_type_2=config.get("label_2"),
        representation_type_3=config.get("label_3"),
        min_counts=args.eval_batch_size,
        random_seed=args.seed,
        use_hvg=False,
        fixed_genes=fixed_genes,
    )
    actual_held_out = dataset.ptb_leave_out_list
    logger.info("Evaluating on %d held-out perts: %s", len(actual_held_out), actual_held_out)

    # Compute and save split indices (needed by predict_morph.py / evaluate_single_model)
    train_idx, val_idx, infer_idx = split_scdata(
        dataset,
        ptb_targets=dataset.ptb_targets,
        ptb_leave_out_list=actual_held_out,
        validation_set_ratio=config.get("validation_set_ratio", 0.1),
        validation_ood_ratio=config.get("validation_ood_ratio", 0.15),
        batch_size=args.eval_batch_size,
    )
    split_idx = {"train_idx": train_idx, "val_idx": val_idx, "infer_idx": infer_idx}
    with open(os.path.join(args.output_dir, "split_idx.pkl"), "wb") as f:
        pickle.dump(split_idx, f, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info("Saved split_idx.pkl (train=%d, val=%s, infer=%s)",
                len(train_idx),
                len(val_idx) if val_idx is not None else 0,
                len(infer_idx) if infer_idx is not None else 0)

    # ----- Step 6: save transfer config (compatible with predict_morph.py) -----
    transfer_config = {
        **config,
        "transfer_source_dataset": config["dataset_name"],
        "transfer_source_adata_path": source_adata_path,
        "base_dir": base_dir,
        "adata_path": target_adata_path,
        "dataset_name": target_dataset_name,
        "leave_out_test_set": actual_held_out,
        "ptb_leave_out_list": actual_held_out,
        "fixed_genes_path": "fixed_genes.pkl",
        "ft_epochs": args.ft_epochs,
        "ft_lr": args.ft_lr,
        "ft_batch_size": args.ft_batch_size,
        "ft_beta": args.ft_beta,
        "no_finetune": args.no_finetune,
    }
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(transfer_config, f, indent=4)
    logger.info("Saved config.json")

    # ----- Step 7: run inference on held-out perts -----
    result_dic = evaluate_on_heldout(
        model, dataset, actual_held_out, device, batch_size=args.eval_batch_size,
    )

    gt_C_y = result_dic["gt_C_y"].flatten()
    gt_y = result_dic["gt_y"]
    pred_y = result_dic["pred_y_1"]

    control_cells_eval = np.asarray(dataset.ctrl_samples, dtype=np.float32)
    gene_names = fixed_genes

    predictions: dict[str, dict[str, np.ndarray]] = {}
    for pert in sorted(set(gt_C_y)):
        mask = gt_C_y == pert
        predictions[pert] = {
            "control": control_cells_eval,
            "true": np.asarray(gt_y[mask], dtype=np.float32),
            "predicted": np.asarray(pred_y[mask], dtype=np.float32),
        }

    logger.info("Predictions (%d perts):", len(predictions))
    for p, d in predictions.items():
        logger.info("  %s: true=%s  predicted=%s", p, d["true"].shape, d["predicted"].shape)

    # ----- Step 8: save predictions (format compatible with evaluate_predictions.py) -----
    model_label = "finetuned" if not args.no_finetune else "zeroshot"
    out_path = os.path.join(args.output_dir, f"{model_label}_predictions.pkl")

    payload = {
        "predictions": predictions,
        "gene_names": gene_names,
        "dataset_name": target_dataset_name,
        "cell_type": target_cell_type,
        "is_log_normalized": True,
        "morph_run_dir": args.output_dir,
        "model_name": f"{model_label}_model.pt",
        "source_dataset": config["dataset_name"],
        "pretrained_model": args.pretrained_model,
        "no_finetune": args.no_finetune,
    }

    with open(out_path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    logger.info("Saved predictions to %s", out_path)
    logger.info("")
    logger.info("To re-run inference with predict_morph.py:")
    logger.info("  python morph/predict_morph.py \\")
    logger.info("      --morph_run_dir %s \\", args.output_dir)
    logger.info("      --splits_csv %s \\", args.splits_csv)
    logger.info("      --model_name model.pt \\")
    logger.info("      --output_dir %s \\", args.output_dir)
    logger.info("      --device %s", args.device)
    logger.info("")
    logger.info("To evaluate with evaluate_predictions.py:")
    logger.info("  python cell_types/tasks/scripts/evaluate_predictions.py \\")
    logger.info("      --predictions_pkl %s \\", out_path)
    logger.info("      --output_dir %s --plot", args.output_dir)

    if args.wandb:
        import wandb
        wandb.finish()

    logger.info("Done.")


if __name__ == "__main__":
    main()
