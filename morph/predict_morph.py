#!/usr/bin/env python3
"""Run MORPH inference on held-out perturbations and save predictions.

Saves a standardized pickle that any downstream eval pipeline can consume:

    {
        "predictions": {
            pert_name: {
                "control":   np.ndarray (n_ctrl, n_genes),
                "true":      np.ndarray (n_pert, n_genes),
                "predicted": np.ndarray (n_pert, n_genes),
            },
            ...
        },
        "gene_names":    list[str],
        "dataset_name":  str,
        "cell_type":     str,
        "is_log_normalized": True,
        "morph_run_dir": str,
        "model_name":    str,
    }

Usage (from repo root, or set MORPH_REPO_DIR in .env):
    uv run morph/predict_morph.py \
        --morph_run_dir /path/to/DepMap_GeneEffect_MORPH_run1771443117 \
        --splits_csv data/norman_k562_essential_splits.csv \
        --output_dir /path/to/eval \
        --device cuda:0

    Cell type and perturbations are read only from the CSV (columns: data, test_set_id,
    test_set, note; optional: cell_type). If cell_type is missing, it is derived from data.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import pickle
import sys

import numpy as np
import scanpy as sc
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MORPH_DIR = os.path.dirname(os.path.abspath(__file__))
if MORPH_DIR not in sys.path:
    sys.path.insert(0, MORPH_DIR)

from inference import evaluate_single_model


def _cell_type_from_data(data: str) -> str:
    """Derive cell type label from dataset name when not in CSV."""
    m = {
        "norman_k562_essential": "K562",
        "replogle_rpe1_hvg": "RPE1",
        "replogle_k562_hvg": "K562",
    }
    return m.get(data, data)


def parse_splits_csv(path: str) -> tuple[str, str, list[str]]:
    """Read a MORPH splits CSV and return (dataset_name, cell_type, list_of_perts).

    Expected columns: data, test_set_id, test_set, note
    Optional column: cell_type (if missing, derived from data).
    The ``test_set`` column is a comma-separated string of perturbation names.
    """
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        row = next(reader)
    dataset_name = row["data"]
    cell_type = row.get("cell_type", "").strip() or _cell_type_from_data(dataset_name)
    perts = [p.strip() for p in row["test_set"].split(",") if p.strip()]
    return dataset_name, cell_type, perts


def load_adata_normalized_hvg(
    adata_path: str,
    hvg_cache_path: str | None = None,
    n_top_genes: int = 5000,
) -> sc.AnnData:
    """Load adata with the same preprocessing MORPH uses at training time."""
    adata = sc.read_h5ad(adata_path)
    logger.info("Loaded adata: %s  shape=%s", adata_path, adata.shape)

    if hasattr(adata.X, "toarray"):
        adata.X = adata.X.toarray()
    adata.X = np.asarray(adata.X, dtype=np.float32)

    totals = adata.X.sum(axis=1)
    keep = totals > 0
    if (~keep).any():
        adata = adata[keep].copy()

    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)

    if hvg_cache_path and os.path.isfile(hvg_cache_path):
        with open(hvg_cache_path, "rb") as f:
            hvg_genes = pickle.load(f)
        genes_in_adata = [g for g in hvg_genes if g in adata.var_names]
        adata = adata[:, genes_in_adata].copy()
        logger.info("HVG from cache: %s (%d genes)", hvg_cache_path, adata.shape[1])
    else:
        sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes, subset=True)
        logger.info("Computed HVG: %d genes", adata.shape[1])

    return adata


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MORPH inference and save predictions.")
    parser.add_argument("--morph_run_dir", type=str, required=True,
                        help="MORPH run directory (contains config.json + checkpoint)")
    parser.add_argument("--splits_csv", type=str, required=True,
                        help="Path to splits CSV (columns: data, test_set_id, test_set, note; optional: cell_type)")
    parser.add_argument("--model_name", type=str, default="best_model_val.pt",
                        help="Checkpoint filename (default: best_model_val.pt)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save the predictions pickle")
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    run_dir = args.morph_run_dir
    device = torch.device(args.device)

    # --- Everything from CSV ---
    dataset_name_from_csv, cell_type, perts = parse_splits_csv(args.splits_csv)
    requested_perts = set(perts)
    logger.info("From splits CSV: dataset=%s, cell_type=%s, %d perts",
                dataset_name_from_csv, cell_type, len(requested_perts))
    logger.info("Perts: %s", sorted(requested_perts))

    # --- Load config ---
    with open(os.path.join(run_dir, "config.json")) as f:
        config = json.load(f)

    adata_path = config["adata_path"]
    base_dir = config["base_dir"]
    dataset_name = config["dataset_name"]
    n_top_genes = config.get("n_top_genes", 5000)
    use_hvg = config.get("use_hvg", False)

    if config["dataset_name"] != dataset_name_from_csv:
        raise ValueError(
            f"Splits CSV dataset '{dataset_name_from_csv}' does not match "
            f"checkpoint config dataset '{config['dataset_name']}'. "
            "Use a checkpoint trained on the same dataset as the CSV."
        )
    logger.info("Dataset: %s  Model: %s  Label: %s",
                dataset_name, config["model"], config["label"])

    # --- Load adata (for gene names + all control cells) ---
    hvg_cache_path = None
    if use_hvg:
        stem = os.path.splitext(os.path.basename(adata_path))[0]
        hvg_cache_path = os.path.join(base_dir, "data", "hvg_cache", f"{stem}_n{n_top_genes}.pkl")

    adata = load_adata_normalized_hvg(adata_path, hvg_cache_path, n_top_genes)
    gene_names = adata.var_names.tolist()

    control_cells = np.asarray(
        adata[adata.obs["gene"] == "non-targeting"].X, dtype=np.float32,
    )
    logger.info("Control cells: %d × %d", *control_cells.shape)

    # --- Run MORPH inference ---
    model_path = os.path.join(run_dir, args.model_name)
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Checkpoint not found: {model_path}")

    model = torch.load(model_path, map_location=device, weights_only=False)
    model = model.to(device)
    model.eval()
    logger.info("Loaded %s (%s params)",
                args.model_name, f"{sum(p.numel() for p in model.parameters()):,}")

    result_dic = evaluate_single_model(model, run_dir, device, use_index=True, infer_data=None)

    # --- Group by perturbation + filter to requested ---
    gt_C_y = result_dic["gt_C_y"].flatten()
    gt_y = result_dic["gt_y"]
    pred_y = result_dic["pred_y_1"]

    available = set(np.unique(gt_C_y).astype(str))
    keep_perts = requested_perts & available
    missing = requested_perts - available
    if missing:
        logger.warning("Requested perts not in inference output (not in held-out set): %s",
                        sorted(missing))

    predictions: dict[str, dict[str, np.ndarray]] = {}
    for pert in sorted(keep_perts):
        mask = gt_C_y == pert
        predictions[pert] = {
            "control": control_cells,
            "true": np.asarray(gt_y[mask], dtype=np.float32),
            "predicted": np.asarray(pred_y[mask], dtype=np.float32),
        }

    logger.info("Predictions (%d perts):", len(predictions))
    for p, d in predictions.items():
        logger.info("  %s: true=%s  predicted=%s", p, d["true"].shape, d["predicted"].shape)

    # --- Save ---
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, f"{args.model_name.replace('.pt', '')}_predictions.pkl")

    payload = {
        "predictions": predictions,
        "gene_names": gene_names,
        "dataset_name": dataset_name,
        "cell_type": cell_type,
        "is_log_normalized": True,
        "morph_run_dir": run_dir,
        "model_name": args.model_name,
    }

    with open(out_path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    logger.info("Saved predictions to %s", out_path)


if __name__ == "__main__":
    main()
