"""
RA-GARK — final model.

Combines:
  - Softmax rationale attention over KG-aspect item representations
  - Local-biased fusion gate initialisation (+5 bias → α≈0.993)
  - KG SVD initialisation for item_kg_aspects
  - Aspect-level + user cross-view CL (proj head + stop-grad)

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


def train_ragark(cfg: Config, device: torch.device) -> dict:
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
        use_rationale=cfg.use_rationale,
        use_global_view=cfg.use_global_view,
        rationale_style=cfg.rationale_style,
        rationale_temperature=cfg.rationale_temperature,
        fusion_init_bias=cfg.fusion_init_bias,
    ).to(device)
    log.info(
        "flags: rat=%s(%s, τ=%.2f) svd=%s acl=%s ucl=%s global=%s fusion_bias=%.1f",
        cfg.use_rationale, cfg.rationale_style, cfg.rationale_temperature,
        cfg.use_svd_init, cfg.use_acl, cfg.use_ucl,
        cfg.use_global_view, cfg.fusion_init_bias,
    )

    # ── KG SVD initialisation ──────────────────────────────────────────
    if cfg.use_svd_init:
        kg_init = build_kg_aspect_init(kg_adj, n_items, cfg.num_aspects, cfg.embedding_dim)
        if kg_init is not None:
            model.item_kg_aspects.data.copy_(kg_init.to(device))
            log.info("item_kg_aspects initialised from KG SVD embeddings")
    else:
        log.info("SVD init disabled — item_kg_aspects stays at xavier init")
    # ───────────────────────────────────────────────────────────────────

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )
    log.info("Optimizer: lr=%.1e wd=%.1e", cfg.learning_rate, cfg.weight_decay)

    scheduler = None
    if getattr(cfg, "lr_scheduler", False):
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=cfg.lr_factor,
            patience=cfg.lr_patience,
            min_lr=cfg.lr_min,
        )
        log.info(
            "LR scheduler: ReduceLROnPlateau(max NDCG, factor=%.2f, patience=%d, min_lr=%.1e)",
            cfg.lr_factor, cfg.lr_patience, cfg.lr_min,
        )

    best_val_ndcg, best_epoch, no_improve = 0.0, 0, 0
    header = (
        f"{'Ep':>4} | {'Loss':>8} | {'BPR':>8} | {'aCL':>8} | {'uCL':>8} |"
        f" {'vHR':>7} | {'vRecall':>7} | {'vNDCG':>7} | Note"
    )
    sep = "-" * len(header)
    log.info("\n%s\n%s", sep, header)

    for epoch in range(cfg.epochs):
        model.train()
        total_loss = total_bpr = total_acl = total_ucl = 0.0

        for users, pos_items, neg_items, _kg_neighbors in loader:
            users = users.to(device)
            pos_items = pos_items.to(device)
            neg_items = neg_items.to(device)

            # Compute LightGCN propagation ONCE per batch and reuse for pos+neg.
            cached_embs = model._lightgcn_embeddings()
            pos_scores, u_loc, u_glo, i_pos_loc, _ = model(
                users, pos_items, cached_embs=cached_embs
            )
            neg_scores, *_ = model(users, neg_items, cached_embs=cached_embs)

            # --- BPR ---
            loss_bpr = bpr_loss(pos_scores, neg_scores)

            # --- Aspect-level item CL (proj + stop-grad per aspect) ---
            if cfg.use_acl:
                i_aspects = model.item_kg_aspects[pos_items]   # [B, A, dim]
                loss_acl = aspect_level_cl(
                    model.cl_projector, i_pos_loc, i_aspects, cfg.temp
                )
            else:
                loss_acl = torch.zeros((), device=device)

            # --- User cross-view CL (proj + stop-grad) ---
            if cfg.use_ucl:
                loss_ucl = infonce_loss(
                    model.cl_projector(u_loc), u_glo.detach(), cfg.temp
                )
            else:
                loss_ucl = torch.zeros((), device=device)

            loss = loss_bpr + cfg.cl_weight * (loss_acl + loss_ucl)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_bpr += loss_bpr.item()
            total_acl += loss_acl.item()
            total_ucl += loss_ucl.item()

        n = len(loader)
        avg_loss = total_loss / n
        avg_bpr = total_bpr / n
        avg_acl = total_acl / n
        avg_ucl = total_ucl / n

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

        if scheduler is not None:
            prev_lr = optimizer.param_groups[0]["lr"]
            scheduler.step(val_res["NDCG"])
            new_lr = optimizer.param_groups[0]["lr"]
            if new_lr < prev_lr:
                note = (note + " lr↓").strip()

        log.info(
            "%4d | %8.4f | %8.4f | %8.4f | %8.4f | %7.4f | %7.4f | %7.4f | %s",
            epoch + 1, avg_loss, avg_bpr, avg_acl, avg_ucl,
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

    return test_res


def _make_base_cfg() -> Config:
    """Shared base config for all ablation runs."""
    cfg = Config()
    cfg.cl_weight = 0.005
    cfg.epochs = 80
    cfg.use_rationale         = True
    cfg.use_svd_init          = True
    cfg.use_acl               = True
    cfg.use_ucl               = True
    cfg.use_global_view       = True
    cfg.rationale_style       = "mlp_softmax"
    cfg.rationale_temperature = 0.5
    cfg.fusion_init_bias      = 5.0
    return cfg


if __name__ == "__main__":
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", _device)

    # ── 4-way ablation of the new perf knobs ───────────────────────────
    # A: baseline (reproduces old behavior — should match REPORT.md ~0.119)
    # B: weight_decay only
    # C: scheduler only (lr_patience=3, default)
    # D: both, but wd downscaled to 1e-5 and lr_patience=6 (gentler)
    runs = [
        ("A_baseline",   dict(weight_decay=0.0,  lr_scheduler=False, lr_patience=3)),
        ("B_wd_only",    dict(weight_decay=1e-4, lr_scheduler=False, lr_patience=3)),
        ("C_sched_only", dict(weight_decay=0.0,  lr_scheduler=True,  lr_patience=3)),
        ("D_both_gentle",dict(weight_decay=1e-5, lr_scheduler=True,  lr_patience=6)),
    ]

    summary = []
    for name, overrides in runs:
        cfg = _make_base_cfg()
        for k, v in overrides.items():
            setattr(cfg, k, v)
        cfg.model_save_path = f"best_ragark_{name}.pth"

        log.info("\n%s", "=" * 90)
        log.info(
            "RUN %s | wd=%.1e | lr_scheduler=%s | lr_patience=%d",
            name, cfg.weight_decay, cfg.lr_scheduler, cfg.lr_patience,
        )
        log.info("%s", "=" * 90)

        try:
            test_res = train_ragark(cfg, _device)
        except Exception as e:
            log.exception("RUN %s FAILED: %s", name, e)
            test_res = None
        summary.append((name, cfg, test_res))

    # ── Final comparison table ─────────────────────────────────────────
    log.info("\n%s", "=" * 90)
    log.info("ABLATION SUMMARY  (test metrics @ K=20)")
    log.info("%s", "=" * 90)
    log.info(
        "%-14s | %7s | %5s | %6s | %7s | %7s | %7s | %7s",
        "Run", "wd", "sched", "lr_pat", "NDCG", "Recall", "HR", "MAP",
    )
    log.info("%s", "-" * 90)
    for name, cfg, r in summary:
        if r is None:
            log.info("%-14s | %7.0e | %5s | %6d | %s",
                     name, cfg.weight_decay, str(cfg.lr_scheduler),
                     cfg.lr_patience, "FAILED")
            continue
        log.info(
            "%-14s | %7.0e | %5s | %6d | %7.4f | %7.4f | %7.4f | %7.4f",
            name, cfg.weight_decay, str(cfg.lr_scheduler), cfg.lr_patience,
            r["NDCG"], r["Recall"], r["HR"], r["MAP"],
        )
    log.info("%s", "=" * 90)
