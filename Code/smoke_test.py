"""Fast end-to-end smoke test for the RA-GARK code path.

This script validates that the default data files, KG loader, LightGCN
adjacency, RA-GARK forward/backward pass, and vectorised evaluator can run
together on a compact in-memory subset. It is intentionally not a quality or
reproducibility benchmark.

Run from Code/:
    python smoke_test.py
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

try:
    import pandas as pd
    import torch
    from torch.utils.data import DataLoader
except ModuleNotFoundError as exc:
    missing = exc.name
    sys.exit(
        f"Missing Python dependency: {missing}. "
        "Activate the project environment, then rerun `python smoke_test.py`."
    )

from config import Config
from data import (
    KnowledgeAwareSampler,
    RecDataset,
    build_kg_index,
    build_lightgcn_adj,
    load_interactions,
)
from evaluate import evaluate
from losses import aspect_level_cl, bpr_loss, infonce_loss
from model import RA_GARK
from utils import set_seed, user_stratified_split

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("smoke")


def _require_files(cfg: Config) -> None:
    required = [
        cfg.interaction_path,
        cfg.kg_path,
        cfg.canonical_kg_path,
        "data/df_edges_all_aspect1.csv",
        "data/kg_relation_map.json",
        "data/kg_canonical_report.txt",
    ]
    missing = [path for path in required if not Path(path).exists()]
    if missing:
        raise FileNotFoundError(f"Missing required data files: {missing}")


def _compact_subset(df: pd.DataFrame, max_users: int) -> tuple[pd.DataFrame, dict, int, int]:
    user_counts = df["user_idx"].value_counts()
    selected_users = user_counts[user_counts >= 3].head(max_users).index
    if selected_users.empty:
        raise ValueError("Smoke test needs at least one user with >=3 interactions.")

    subset = df[df["user_idx"].isin(selected_users)].copy()
    subset["user_idx"], _ = pd.factorize(subset["user_idx"], sort=True)
    subset["item_idx"], item_values = pd.factorize(subset["asin"], sort=True)

    asin_to_idx = {asin: int(idx) for idx, asin in enumerate(item_values)}
    n_users = int(subset["user_idx"].max()) + 1
    n_items = int(subset["item_idx"].max()) + 1
    return subset.reset_index(drop=True), asin_to_idx, n_users, n_items


def run_smoke(max_users: int, batch_size: int) -> dict[str, float]:
    cfg = Config()
    cfg.seed = 42
    cfg.embedding_dim = 16
    cfg.n_layers = 1
    cfg.num_aspects = 2
    cfg.batch_size = batch_size
    cfg.eval_k = 5
    cfg.eval_extra_ks = (3,)

    _require_files(cfg)
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    df, _user_enc, _item_enc, _full_asin_to_idx, _full_users, _full_items = load_interactions(
        cfg.interaction_path
    )
    df, asin_to_idx, n_users, n_items = _compact_subset(df, max_users=max_users)
    log.info("Subset: %d users, %d items, %d interactions", n_users, n_items, len(df))

    kg_adj, kg_rev_adj, aspects = build_kg_index(
        cfg.kg_path,
        asin_to_idx,
        cfg.kg_stopwords,
        cfg.kg_top_freq_pct,
    )
    log.info("Default KG smoke index: %d items, %d aspects", len(kg_adj), len(aspects))

    train_df, _val_df, test_df = user_stratified_split(
        df, val_ratio=0.15, test_ratio=0.15, seed=cfg.seed
    )
    if test_df.empty:
        raise ValueError("Smoke subset produced no test rows; increase --max-users.")

    train_hist = train_df.groupby("user_idx")["item_idx"].apply(set).to_dict()
    test_gt = test_df.groupby("user_idx")["item_idx"].apply(list).to_dict()

    sampler = KnowledgeAwareSampler(n_items, kg_adj, kg_rev_adj)
    dataset = RecDataset(train_df["user_idx"], train_df["item_idx"], sampler)
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=0)
    users, pos_items, neg_items, _kg_neighbors = next(iter(loader))

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
        fusion_gate_style=cfg.fusion_gate_style,
    ).to(device)

    model.train()
    users = users.to(device)
    pos_items = pos_items.to(device)
    neg_items = neg_items.to(device)
    cached_embs = model._lightgcn_embeddings()
    pos_scores, u_loc, u_glo, i_pos_loc, _i_pos_glo = model(
        users, pos_items, cached_embs=cached_embs
    )
    neg_scores, *_ = model(users, neg_items, cached_embs=cached_embs)
    loss_bpr = bpr_loss(pos_scores, neg_scores)
    loss_acl = aspect_level_cl(
        model.cl_projector,
        i_pos_loc,
        model.item_kg_aspects[pos_items],
        cfg.temp,
    )
    loss_ucl = infonce_loss(model.cl_projector(u_loc), u_glo.detach(), cfg.temp)
    loss = loss_bpr + cfg.cl_weight * (loss_acl + loss_ucl)
    loss.backward()
    if not torch.isfinite(loss):
        raise RuntimeError(f"Non-finite smoke loss: {loss.item()}")
    log.info("One backward pass OK: loss=%.4f", loss.item())

    eval_users = dict(list(test_gt.items())[: min(8, len(test_gt))])
    metrics = evaluate(
        model,
        eval_users,
        train_hist,
        device,
        k=cfg.eval_k,
        batch_size=min(8, cfg.batch_size),
        extra_ks=cfg.eval_extra_ks,
    )
    log.info("Evaluation OK: %s", {k: round(v, 4) for k, v in metrics.items()})
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-users", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    run_smoke(max_users=args.max_users, batch_size=args.batch_size)
    log.info("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
