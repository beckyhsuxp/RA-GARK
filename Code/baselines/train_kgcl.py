"""KGCL baseline training script.

Trains the KGCL model with:
  - BPR loss on the full graph
  - InfoNCE contrastive loss between two independent random-dropout views

Run:
    python -m baselines.train_kgcl
"""

from __future__ import annotations

import copy
import logging
import sys
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from baselines.kgat import build_ckg
from baselines.kgcl import KGCL
from config import Config
from data import (
    KnowledgeAwareSampler,
    RecDataset,
    build_kg_index,
    load_interactions,
)
from evaluate import evaluate
from losses import bpr_loss, infonce_loss
from train_v1 import set_seed, user_stratified_split

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

CL_WEIGHT = 0.01
DROP_PROB = 0.3


def train_kgcl(cfg: Config, device: torch.device) -> None:
    set_seed(cfg.seed)

    df, _, _, asin_to_idx, n_users, n_items = load_interactions(cfg.interaction_path)
    kg_adj, _, _ = build_kg_index(
        cfg.kg_path, asin_to_idx, cfg.kg_stopwords, cfg.kg_top_freq_pct
    )

    train_df, val_df, test_df = user_stratified_split(
        df, val_ratio=0.15, test_ratio=0.15, seed=cfg.seed
    )

    train_hist = train_df.groupby("user_idx")["item_idx"].apply(set).to_dict()
    val_gt = val_df.groupby("user_idx")["item_idx"].apply(list).to_dict()
    test_gt = test_df.groupby("user_idx")["item_idx"].apply(list).to_dict()

    sampler = KnowledgeAwareSampler(n_items, defaultdict(list), defaultdict(list))
    dataset = RecDataset(train_df["user_idx"], train_df["item_idx"], sampler)
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=0)

    edge_index, deg, n_aspects = build_ckg(train_df, kg_adj, n_users, n_items)
    edge_index = edge_index.to(device)
    deg = deg.to(device)

    model = KGCL(
        n_users=n_users,
        n_items=n_items,
        n_aspects=n_aspects,
        edge_index=edge_index,
        deg=deg,
        dim=cfg.embedding_dim,
        n_layers=cfg.n_layers,
        drop_prob=DROP_PROB,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    log.info("KGCL: cl_weight=%.3f  drop_prob=%.2f", CL_WEIGHT, DROP_PROB)

    best_val_ndcg, best_epoch, no_improve = 0.0, 0, 0
    header = (
        f"{'Ep':>4} | {'Loss':>8} | {'BPR':>8} | {'CL':>8} |"
        f" {'vHR':>7} | {'vRecall':>7} | {'vNDCG':>7} | Note"
    )
    sep = "-" * len(header)
    log.info("\n%s\n%s", sep, header)

    for epoch in range(cfg.epochs):
        model.train()
        total_loss = total_bpr = total_cl = 0.0

        for users, pos_items, neg_items, _ in loader:
            users = users.to(device)
            pos_items = pos_items.to(device)
            neg_items = neg_items.to(device)

            emb_full = model.propagate_full()
            emb_v1, emb_v2 = model.propagate_two_augmented_views()

            # ---- BPR on full graph --------------------------------------
            u_full = emb_full[users]
            pos_full = emb_full[n_users + pos_items]
            neg_full = emb_full[n_users + neg_items]
            pos_scores = (u_full * pos_full).sum(dim=-1)
            neg_scores = (u_full * neg_full).sum(dim=-1)
            loss_bpr = bpr_loss(pos_scores, neg_scores)

            # ---- Cross-view CL between two augmented views --------------
            u_v1 = emb_v1[users]
            u_v2 = emb_v2[users]
            i_v1 = emb_v1[n_users + pos_items]
            i_v2 = emb_v2[n_users + pos_items]
            loss_cl = (
                infonce_loss(u_v1, u_v2, cfg.temp)
                + infonce_loss(i_v1, i_v2, cfg.temp)
            )

            loss = loss_bpr + CL_WEIGHT * loss_cl

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_bpr += loss_bpr.item()
            total_cl += loss_cl.item()

        n = len(loader)
        avg_loss = total_loss / n
        avg_bpr = total_bpr / n
        avg_cl = total_cl / n

        val_res = evaluate(
            model, val_gt, train_hist, device,
            k=cfg.eval_k, batch_size=cfg.eval_batch_size, extra_ks=cfg.eval_extra_ks,
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
            "%4d | %8.4f | %8.4f | %8.4f | %7.4f | %7.4f | %7.4f | %s",
            epoch + 1, avg_loss, avg_bpr, avg_cl,
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
        k=cfg.eval_k, batch_size=cfg.eval_batch_size, extra_ks=cfg.eval_extra_ks,
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
    cfg.model_save_path = "best_model_kgcl.pth"

    train_kgcl(cfg, _device)
