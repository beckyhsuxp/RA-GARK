"""MCCLK baseline training script.

Trains the MCCLK model with:
  - BPR loss on the global view
  - Multi-level cross-view InfoNCE:
      * users     : local ↔ global
      * items     : local ↔ semantic, semantic ↔ global, local ↔ global

Run:
    python -m baselines.train_mcclk
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

from baselines.mcclk import MCCLK, build_mcclk_views
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


def train_mcclk(cfg: Config, device: torch.device) -> None:
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

    edge_ui, edge_ia, edge_full, deg, n_aspects = build_mcclk_views(
        train_df, kg_adj, n_users, n_items
    )
    edge_ui   = edge_ui.to(device)
    edge_ia   = edge_ia.to(device)
    edge_full = edge_full.to(device)
    deg       = deg.to(device)
    log.info(
        "MCCLK views: |UI|=%d, |IA|=%d, |full|=%d, n_aspects=%d",
        edge_ui.size(1), edge_ia.size(1), edge_full.size(1), n_aspects,
    )

    model = MCCLK(
        n_users=n_users,
        n_items=n_items,
        n_aspects=n_aspects,
        edge_index_ui=edge_ui,
        edge_index_ia=edge_ia,
        edge_index_full=edge_full,
        deg=deg,
        dim=cfg.embedding_dim,
        n_layers=cfg.n_layers,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    log.info("MCCLK: cl_weight=%.3f", CL_WEIGHT)

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

            emb_local, emb_sem, emb_global = model.propagate_three_views()

            # ---- BPR on the global view --------------------------------
            u_g   = emb_global[users]
            pos_g = emb_global[n_users + pos_items]
            neg_g = emb_global[n_users + neg_items]
            pos_scores = (u_g * pos_g).sum(dim=-1)
            neg_scores = (u_g * neg_g).sum(dim=-1)
            loss_bpr = bpr_loss(pos_scores, neg_scores)

            # ---- Multi-level cross-view CL -----------------------------
            # User: local ↔ global  (semantic view has no users)
            u_l = emb_local[users]
            loss_u_cl = infonce_loss(u_l, u_g, cfg.temp)

            # Item: local ↔ semantic, semantic ↔ global, local ↔ global
            i_l = emb_local[n_users + pos_items]
            i_s = emb_sem[n_users + pos_items]
            i_g = pos_g
            loss_i_cl = (
                infonce_loss(i_l, i_s, cfg.temp)
                + infonce_loss(i_s, i_g, cfg.temp)
                + infonce_loss(i_l, i_g, cfg.temp)
            )

            loss_cl = loss_u_cl + loss_i_cl
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
    cfg.model_save_path = "best_model_mcclk.pth"

    train_mcclk(cfg, _device)
