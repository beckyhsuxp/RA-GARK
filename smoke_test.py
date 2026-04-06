"""
Smoke test — runs the full pipeline for 2 epochs on a 500-row subset.
Verifies all stages complete without error.

Run:
    conda activate py311_cuda126
    python smoke_test.py
"""

from __future__ import annotations

import copy
import logging
import sys

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
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
from model import RAKG_LMR
from train import set_seed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

SMOKE_ROWS = 500
SMOKE_EPOCHS = 2


def smoke_test() -> None:
    cfg = Config(
        epochs=SMOKE_EPOCHS,
        batch_size=64,
        eval_batch_size=32,
        model_save_path="smoke_best.pth",
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    set_seed(cfg.seed)

    # --- Data (subset) ---
    log.info("Loading data (first %d rows)...", SMOKE_ROWS)
    df, _, _, asin_to_idx, n_users, n_items = load_interactions(cfg.interaction_path)
    df = df.head(SMOKE_ROWS).copy()

    # Re-encode on the subset so indices are contiguous
    from sklearn.preprocessing import LabelEncoder
    df["user_idx"] = LabelEncoder().fit_transform(df["user_id"])
    df["item_idx"] = LabelEncoder().fit_transform(df["asin"])
    n_users = int(df["user_idx"].max()) + 1
    n_items = int(df["item_idx"].max()) + 1
    asin_to_idx = dict(zip(df["asin"], df["item_idx"]))
    log.info("Subset — users: %d  items: %d  rows: %d", n_users, n_items, len(df))

    kg_adj, kg_rev_adj, _ = build_kg_index(
        cfg.kg_path, asin_to_idx, cfg.kg_stopwords, cfg.kg_top_freq_pct
    )

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=cfg.seed)
    test_gt = test_df.groupby("user_idx")["item_idx"].apply(list).to_dict()
    train_hist = train_df.groupby("user_idx")["item_idx"].apply(set).to_dict()
    log.info("Train: %d  Test: %d  Eval users: %d", len(train_df), len(test_df), len(test_gt))

    sampler = KnowledgeAwareSampler(n_items, kg_adj, kg_rev_adj)
    dataset = RecDataset(train_df["user_idx"], train_df["item_idx"], sampler)
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=0)

    # --- Model ---
    adj = build_lightgcn_adj(train_df, n_users, n_items, device)
    model = RAKG_LMR(
        num_users=n_users,
        num_items=n_items,
        adj_matrix=adj,
        num_aspects=cfg.num_aspects,
        dim=cfg.embedding_dim,
        n_layers=cfg.n_layers,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)

    # --- Training ---
    log.info("Running %d smoke epochs...", SMOKE_EPOCHS)
    for epoch in range(SMOKE_EPOCHS):
        model.train()
        total_loss = 0.0
        for users, pos_items, neg_items, kg_neighbors in loader:
            users, pos_items = users.to(device), pos_items.to(device)
            neg_items, kg_neighbors = neg_items.to(device), kg_neighbors.to(device)

            pos_scores, u_loc, u_glo, i_pos_loc, i_pos_glo = model(users, pos_items)
            neg_scores, *_ = model(users, neg_items)
            _, _, _, _, i_nbr_glo = model(users, kg_neighbors)

            loss_bpr = bpr_loss(pos_scores, neg_scores)
            loss_cl = (
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
        res = evaluate(model, test_gt, train_hist, device, k=cfg.eval_k, batch_size=cfg.eval_batch_size)
        log.info(
            "Epoch %d | loss=%.4f | HR=%.4f | NDCG=%.4f",
            epoch + 1, avg_loss, res["HR"], res["NDCG"],
        )

    # --- Save & reload ---
    torch.save(copy.deepcopy(model.state_dict()), cfg.model_save_path)
    model.load_state_dict(torch.load(cfg.model_save_path, map_location=device, weights_only=True))
    log.info("Checkpoint save/load: OK")

    log.info("Smoke test PASSED.")


if __name__ == "__main__":
    try:
        smoke_test()
    except Exception as e:
        log.exception("Smoke test FAILED: %s", e)
        sys.exit(1)
