"""
Optuna hyperparameter search for v6 (aspect-level CL + proj head + stop-grad).

Tunes cl_weight and temp (InfoNCE temperature).
reg_weight is fixed — MSE reg is effectively 0 with current architecture.

Uses val NDCG@20 at epoch 20 as the objective (fast proxy).

Run:
    python tune_weights.py
"""

from __future__ import annotations

import logging
import random
import time

import numpy as np
import optuna
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config import Config
from data import (
    KnowledgeAwareSampler,
    RecDataset,
    build_kg_index,
    build_lightgcn_adj,
    load_interactions,
)
from evaluate import evaluate
from losses import bpr_loss, infonce_loss
from model import RA_GARK
from train_v1 import set_seed, user_stratified_split
from train_v6 import aspect_level_cl

# suppress per-trial noise; only show Optuna progress
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
optuna.logging.set_verbosity(optuna.logging.INFO)
log = logging.getLogger("tune")
log.setLevel(logging.INFO)

SEARCH_EPOCHS = 20   # fast proxy
N_TRIALS      = 30


def run_trial(cfg: Config, device: torch.device, seed: int = 42) -> float:
    """Train v6 architecture for SEARCH_EPOCHS and return best val NDCG."""
    set_seed(seed)

    df, _, _, asin_to_idx, n_users, n_items = load_interactions(cfg.interaction_path)
    kg_adj, kg_rev_adj, _ = build_kg_index(
        cfg.kg_path, asin_to_idx, cfg.kg_stopwords, cfg.kg_top_freq_pct
    )
    train_df, val_df, _ = user_stratified_split(df, val_ratio=0.15, test_ratio=0.15, seed=seed)

    train_hist = train_df.groupby("user_idx")["item_idx"].apply(set).to_dict()
    val_gt     = val_df.groupby("user_idx")["item_idx"].apply(list).to_dict()

    sampler = KnowledgeAwareSampler(n_items, kg_adj, kg_rev_adj)
    dataset = RecDataset(train_df["user_idx"], train_df["item_idx"], sampler)
    loader  = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=0)

    adj = build_lightgcn_adj(train_df, n_users, n_items, device)
    model = RA_GARK(
        num_users=n_users,
        num_items=n_items,
        adj_matrix=adj,
        num_aspects=cfg.num_aspects,
        dim=cfg.embedding_dim,
        n_layers=cfg.n_layers,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)

    best_val_ndcg = 0.0

    for epoch in range(SEARCH_EPOCHS):
        model.train()
        for users, pos_items, neg_items, kg_neighbors in loader:
            users        = users.to(device)
            pos_items    = pos_items.to(device)
            neg_items    = neg_items.to(device)
            kg_neighbors = kg_neighbors.to(device)

            pos_scores, u_loc, u_glo, i_pos_loc, i_pos_glo = model(users, pos_items)
            neg_scores, *_                                  = model(users, neg_items)
            _, _, _, _, i_nbr_glo                           = model(users, kg_neighbors)

            # BPR
            loss_bpr = bpr_loss(pos_scores, neg_scores)

            # Aspect-level item CL (proj + stop-grad)
            i_aspects = model.item_kg_aspects[pos_items]
            loss_acl = aspect_level_cl(
                model.cl_projector, i_pos_loc, i_aspects, cfg.temp
            )

            # User cross-view CL (proj + stop-grad)
            loss_ucl = infonce_loss(
                model.cl_projector(u_loc), u_glo.detach(), cfg.temp
            )

            # KG reg (MSE, effectively ~0)
            loss_reg = F.mse_loss(i_pos_glo, i_nbr_glo)

            loss = (
                loss_bpr
                + cfg.cl_weight * (loss_acl + loss_ucl)
                + cfg.reg_weight * loss_reg
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        val_res = evaluate(model, val_gt, train_hist, device,
                           k=cfg.eval_k, batch_size=cfg.eval_batch_size)
        best_val_ndcg = max(best_val_ndcg, val_res["NDCG"])

    return best_val_ndcg


def objective(trial: optuna.Trial, device: torch.device) -> float:
    cfg = Config()
    cfg.cl_weight = trial.suggest_float("cl_weight", 1e-4, 0.1, log=True)
    cfg.temp      = trial.suggest_float("temp", 0.05, 0.5)
    cfg.reg_weight = 0.973  # fixed, MSE reg is ~0 anyway

    t0 = time.perf_counter()
    ndcg = run_trial(cfg, device, seed=42)
    elapsed = time.perf_counter() - t0

    log.info(
        "Trial %3d | cl=%.6f  temp=%.3f | val NDCG=%.4f | %.0fs",
        trial.number, cfg.cl_weight, cfg.temp, ndcg, elapsed,
    )
    return ndcg


if __name__ == "__main__":
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s  |  Trials: %d  |  Epochs/trial: %d", _device, N_TRIALS, SEARCH_EPOCHS)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.NopPruner(),
    )
    study.optimize(lambda t: objective(t, _device), n_trials=N_TRIALS, show_progress_bar=False)

    best = study.best_trial
    log.info("=" * 55)
    log.info("Best trial: val NDCG = %.4f", best.value)
    log.info("  cl_weight = %.6f", best.params["cl_weight"])
    log.info("  temp      = %.4f", best.params["temp"])
    log.info("=" * 55)

    import json, pathlib
    out = {
        "cl_weight": best.params["cl_weight"],
        "temp": best.params["temp"],
    }
    pathlib.Path("best_weights.json").write_text(json.dumps(out, indent=2))
    log.info("Saved to best_weights.json")
