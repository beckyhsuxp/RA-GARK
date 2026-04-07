"""LightGCN baseline training script.

Uses the SAME data, split, and evaluator as v1/v6/v7 so the result is
directly comparable.

Run:
    python -m baselines.train_lightgcn
"""

from __future__ import annotations

import copy
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

# Allow running as `python baselines/train_lightgcn.py` from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from baselines.lightgcn import LightGCN
from config import Config
from data import (
    KnowledgeAwareSampler,
    RecDataset,
    build_lightgcn_adj,
    load_interactions,
)
from evaluate import evaluate
from losses import bpr_loss
from train_v1 import set_seed, user_stratified_split

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def train_lightgcn(cfg: Config, device: torch.device) -> None:
    set_seed(cfg.seed)

    df, _, _, _, n_users, n_items = load_interactions(cfg.interaction_path)

    train_df, val_df, test_df = user_stratified_split(
        df, val_ratio=0.15, test_ratio=0.15, seed=cfg.seed
    )

    train_hist = train_df.groupby("user_idx")["item_idx"].apply(set).to_dict()
    val_gt = val_df.groupby("user_idx")["item_idx"].apply(list).to_dict()
    test_gt = test_df.groupby("user_idx")["item_idx"].apply(list).to_dict()

    # Pure random negatives — no KG.
    sampler = KnowledgeAwareSampler(n_items, defaultdict(list), defaultdict(list))
    dataset = RecDataset(train_df["user_idx"], train_df["item_idx"], sampler)
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=0)

    adj = build_lightgcn_adj(train_df, n_users, n_items, device)
    model = LightGCN(
        num_users=n_users,
        num_items=n_items,
        adj_matrix=adj,
        dim=cfg.embedding_dim,
        n_layers=cfg.n_layers,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)

    best_val_ndcg, best_epoch, no_improve = 0.0, 0, 0
    header = (
        f"{'Ep':>4} | {'Loss':>8} | {'vHR':>7} | {'vRecall':>7} | {'vNDCG':>7} | Note"
    )
    sep = "-" * len(header)
    log.info("\n%s\n%s", sep, header)

    for epoch in range(cfg.epochs):
        model.train()
        total_loss = 0.0

        for users, pos_items, neg_items, _ in loader:  # ignore KG neighbour field
            users = users.to(device)
            pos_items = pos_items.to(device)
            neg_items = neg_items.to(device)

            pos_scores = model(users, pos_items)
            neg_scores = model(users, neg_items)
            loss = bpr_loss(pos_scores, neg_scores)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)

        val_res = evaluate(
            model, val_gt, train_hist, device,
            k=cfg.eval_k, batch_size=cfg.eval_batch_size,
        )

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
            "%4d | %8.4f | %7.4f | %7.4f | %7.4f | %s",
            epoch + 1, avg_loss,
            val_res["HR"], val_res["Recall"], val_res["NDCG"], note,
        )

        patience = getattr(cfg, "early_stop_patience", 0)
        if patience > 0 and no_improve >= patience:
            log.info("Early stopping at epoch %d (no improve for %d epochs)",
                     epoch + 1, no_improve)
            break

    log.info(sep)
    log.info("Best val NDCG@%d = %.4f at epoch %d",
             cfg.eval_k, best_val_ndcg, best_epoch)

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
    cfg.epochs = 80
    cfg.model_save_path = "best_model_lightgcn.pth"

    train_lightgcn(cfg, _device)
