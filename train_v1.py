"""
RA-GARK v1 — fixes applied:
  #1/#7  Per-user stratified train/val/test split; val drives early stopping,
         test is reported only once at the end.
  #3     Separate user_fusion_gate and item_fusion_gate.

Run:
    python train_v1.py
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
    build_kg_index,
    build_lightgcn_adj,
    load_interactions,
)
from evaluate import evaluate
from losses import bpr_loss, infonce_loss
from model import RA_GARK

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def user_stratified_split(
    df: pd.DataFrame,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    train_parts, val_parts, test_parts = [], [], []

    for _, group in df.groupby("user_idx", sort=False):
        n = len(group)
        shuffled = group.sample(frac=1, random_state=int(rng.integers(1 << 31)))

        if n < 3:
            train_parts.append(shuffled)
            continue

        n_test  = max(1, round(n * test_ratio))
        n_val   = max(1, round(n * val_ratio))
        n_train = n - n_val - n_test

        if n_train < 1:
            train_parts.append(shuffled.iloc[:1])
            test_parts.append(shuffled.iloc[1:])
            continue

        train_parts.append(shuffled.iloc[:n_train])
        val_parts.append(shuffled.iloc[n_train : n_train + n_val])
        test_parts.append(shuffled.iloc[n_train + n_val :])

    train_df = pd.concat(train_parts).reset_index(drop=True)
    val_df   = pd.concat(val_parts).reset_index(drop=True)  if val_parts  else pd.DataFrame(columns=df.columns)
    test_df  = pd.concat(test_parts).reset_index(drop=True) if test_parts else pd.DataFrame(columns=df.columns)

    log.info(
        "Split → train: %d  val: %d  test: %d  (eval users val/test: %d/%d)",
        len(train_df), len(val_df), len(test_df),
        val_df["user_idx"].nunique(), test_df["user_idx"].nunique(),
    )
    return train_df, val_df, test_df


def train(cfg: Config, device: torch.device) -> None:
    set_seed(cfg.seed)

    df, _, _, asin_to_idx, n_users, n_items = load_interactions(cfg.interaction_path)
    kg_adj, kg_rev_adj, _ = build_kg_index(
        cfg.kg_path, asin_to_idx, cfg.kg_stopwords, cfg.kg_top_freq_pct
    )

    train_df, val_df, test_df = user_stratified_split(
        df, val_ratio=0.15, test_ratio=0.15, seed=cfg.seed
    )

    train_hist = train_df.groupby("user_idx")["item_idx"].apply(set).to_dict()
    val_gt     = val_df.groupby("user_idx")["item_idx"].apply(list).to_dict()
    test_gt    = test_df.groupby("user_idx")["item_idx"].apply(list).to_dict()

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

    best_val_ndcg, best_epoch, no_improve = 0.0, 0, 0
    header = (
        f"{'Ep':>4} | {'Loss':>8} | "
        f"{'vHR':>7} | {'vRecall':>8} | {'vNDCG':>7} | {'EvalMs':>7} | Note"
    )
    sep = "-" * len(header)
    log.info("\n%s\n%s", sep, header)

    for epoch in range(cfg.epochs):
        model.train()
        total_loss = 0.0

        for users, pos_items, neg_items, kg_neighbors in loader:
            users        = users.to(device)
            pos_items    = pos_items.to(device)
            neg_items    = neg_items.to(device)
            kg_neighbors = kg_neighbors.to(device)

            pos_scores, u_loc, u_glo, i_pos_loc, i_pos_glo = model(users, pos_items)
            neg_scores, *_                                  = model(users, neg_items)
            _, _, _, _, i_nbr_glo                           = model(users, kg_neighbors)

            loss_bpr = bpr_loss(pos_scores, neg_scores)
            loss_cl  = (
                infonce_loss(i_pos_loc, i_pos_glo, cfg.temp)
                + infonce_loss(u_loc, u_glo, cfg.temp)
            )

            loss_reg = F.mse_loss(i_pos_glo, i_nbr_glo)

            loss = loss_bpr + cfg.cl_weight * loss_cl + cfg.reg_weight * loss_reg

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)

        t0 = time.perf_counter()
        val_res = evaluate(
            model, val_gt, train_hist, device,
            k=cfg.eval_k, batch_size=cfg.eval_batch_size,
        )
        eval_ms = (time.perf_counter() - t0) * 1000

        note = ""
        if val_res["NDCG"] > best_val_ndcg:
            best_val_ndcg = val_res["NDCG"]
            best_epoch    = epoch + 1
            no_improve    = 0
            torch.save(copy.deepcopy(model.state_dict()), cfg.model_save_path)
            note = "* best"
        else:
            no_improve += 1

        log.info(
            "%4d | %8.4f | %7.4f | %8.4f | %7.4f | %6.0fms | %s",
            epoch + 1, avg_loss,
            val_res["HR"], val_res["Recall"], val_res["NDCG"], eval_ms, note,
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
    train(Config(), _device)
