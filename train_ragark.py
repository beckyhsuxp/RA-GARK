"""
RA-GARK — final model.

Combines:
  - KG SVD initialisation for item_kg_aspects
  - Aspect-level cross-view CL (proj head + stop-grad)
  - Per-parameter learning rate (KG aspect lr < base lr)

Run:
    python train_ragark.py
"""

from __future__ import annotations

import copy
import logging
import random
import time
from typing import Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config import Config
from data import (
    KnowledgeAwareSampler,
    RecDataset,
    build_kg_aspect_init,
    build_kg_index,
    build_lightgcn_adj,
    load_interactions,
)
from evaluate import evaluate
from losses import bpr_loss, infonce_loss
from model import RA_GARK
from train_v1 import set_seed, user_stratified_split
from train_v6 import aspect_level_cl

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def train_ragark(cfg: Config, device: torch.device) -> None:
    set_seed(cfg.seed)

    df, _, _, asin_to_idx, n_users, n_items = load_interactions(cfg.interaction_path)
    kg_adj, kg_rev_adj, _ = build_kg_index(
        cfg.kg_path, asin_to_idx, cfg.kg_stopwords, cfg.kg_top_freq_pct
    )

    train_df, val_df, test_df = user_stratified_split(
        df, val_ratio=0.15, test_ratio=0.15, seed=cfg.seed
    )

    train_hist = train_df.groupby("user_idx")["item_idx"].apply(set).to_dict()
    val_gt = val_df.groupby("user_idx")["item_idx"].apply(list).to_dict()
    test_gt = test_df.groupby("user_idx")["item_idx"].apply(list).to_dict()

    sampler = KnowledgeAwareSampler(n_items, kg_adj, kg_rev_adj)
    dataset = RecDataset(train_df["user_idx"], train_df["item_idx"], sampler)
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=0)

    adj = build_lightgcn_adj(train_df, n_users, n_items, device)
    model = RA_GARK(
        num_users=n_users,
        num_items=n_items,
        adj_matrix=adj,
        num_aspects=cfg.num_aspects,
        dim=cfg.embedding_dim,
        n_layers=cfg.n_layers,
    ).to(device)

    # ── KG SVD initialisation ──────────────────────────────────────────
    kg_init = build_kg_aspect_init(kg_adj, n_items, cfg.num_aspects, cfg.embedding_dim)
    if kg_init is not None:
        model.item_kg_aspects.data.copy_(kg_init.to(device))
        log.info("item_kg_aspects initialised from KG SVD embeddings")
    # ───────────────────────────────────────────────────────────────────

    # Lower lr for item_kg_aspects to preserve KG-pretrained semantics
    kg_lr = getattr(cfg, "kg_aspect_lr", cfg.learning_rate * 0.1)
    kg_param_id = id(model.item_kg_aspects)
    base_params = [p for p in model.parameters() if id(p) != kg_param_id]
    optimizer = torch.optim.Adam(
        [
            {"params": base_params, "lr": cfg.learning_rate},
            {"params": [model.item_kg_aspects], "lr": kg_lr},
        ]
    )
    log.info("Optimizer: base lr=%.1e, item_kg_aspects lr=%.1e", cfg.learning_rate, kg_lr)

    best_val_ndcg, best_epoch, no_improve = 0.0, 0, 0
    header = (
        f"{'Ep':>4} | {'Loss':>8} | {'BPR':>8} | {'aCL':>8} | {'uCL':>8} | {'Reg':>8} |"
        f" {'vHR':>7} | {'vRecall':>7} | {'vNDCG':>7} | Note"
    )
    sep = "-" * len(header)
    log.info("\n%s\n%s", sep, header)

    for epoch in range(cfg.epochs):
        model.train()
        total_loss = total_bpr = total_acl = total_ucl = total_reg = 0.0

        for users, pos_items, neg_items, kg_neighbors in loader:
            users = users.to(device)
            pos_items = pos_items.to(device)
            neg_items = neg_items.to(device)
            kg_neighbors = kg_neighbors.to(device)

            pos_scores, u_loc, u_glo, i_pos_loc, i_pos_glo = model(users, pos_items)
            neg_scores, *_ = model(users, neg_items)
            _, _, _, _, i_nbr_glo = model(users, kg_neighbors)

            # --- BPR ---
            loss_bpr = bpr_loss(pos_scores, neg_scores)

            # --- Aspect-level item CL (proj + stop-grad per aspect) ---
            i_aspects = model.item_kg_aspects[pos_items]   # [B, A, dim]
            loss_acl = aspect_level_cl(
                model.cl_projector, i_pos_loc, i_aspects, cfg.temp
            )

            # --- User cross-view CL (proj + stop-grad) ---
            loss_ucl = infonce_loss(
                model.cl_projector(u_loc), u_glo.detach(), cfg.temp
            )

            # --- KG regularization ---
            loss_reg = F.mse_loss(i_pos_glo, i_nbr_glo)

            loss = (
                loss_bpr
                + cfg.cl_weight * (loss_acl + loss_ucl)
                + cfg.reg_weight * loss_reg
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_bpr += loss_bpr.item()
            total_acl += loss_acl.item()
            total_ucl += loss_ucl.item()
            total_reg += loss_reg.item()

        n = len(loader)
        avg_loss = total_loss / n
        avg_bpr = total_bpr / n
        avg_acl = total_acl / n
        avg_ucl = total_ucl / n
        avg_reg = total_reg / n

        t0 = time.perf_counter()
        val_res = evaluate(
            model, val_gt, train_hist, device,
            k=cfg.eval_k, batch_size=cfg.eval_batch_size,
        )
        eval_ms = (time.perf_counter() - t0) * 1000

        note = ""
        if val_res["NDCG"] > best_val_ndcg:
            best_val_ndcg = val_res["NDCG"]
            best_epoch = epoch + 1
            no_improve = 0
            torch.save(copy.deepcopy(model.state_dict()), cfg.model_save_path)
            note = "* best"
        else:
            no_improve += 1

        log.info(
            "%4d | %8.4f | %8.4f | %8.4f | %8.4f | %8.4f | %7.4f | %7.4f | %7.4f | %s",
            epoch + 1, avg_loss, avg_bpr, avg_acl, avg_ucl, avg_reg,
            val_res["HR"], val_res["Recall"], val_res["NDCG"], note,
        )

        patience = getattr(cfg, "early_stop_patience", 0)
        if patience > 0 and no_improve >= patience:
            log.info("Early stopping at epoch %d (no improve for %d epochs)",
                     epoch + 1, no_improve)
            break

    log.info(sep)
    log.info("Best val NDCG@%d = %.4f at epoch %d", cfg.eval_k, best_val_ndcg, best_epoch)

    log.info("Loading best checkpoint → final TEST evaluation...")
    model.load_state_dict(
        torch.load(cfg.model_save_path, map_location=device, weights_only=True)
    )
    test_res = evaluate(
        model, test_gt, train_hist, device,
        k=cfg.eval_k, batch_size=cfg.eval_batch_size,
    )
    log.info("─" * 55)
    log.info("TEST metrics @ K=%d  (best epoch: %d)", cfg.eval_k, best_epoch)
    for metric, val in test_res.items():
        log.info("  %-12s %.4f", metric, val)
    log.info("─" * 55)


if __name__ == "__main__":
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", _device)

    cfg = Config()
    cfg.cl_weight = 0.005
    cfg.reg_weight = 0.973
    cfg.epochs = 80
    cfg.model_save_path = "best_ragark_model.pth"

    train_ragark(cfg, _device)
