"""RA-GARK training entry point — relation-typed-init experiment.

Self-contained mirror of ../train_ragark.py. The ONLY differences:

  1.  Uses RelationInitConfig (subclass of root Config) — exposes the two
      flags relation_typed_aspect_init and user_svd_init.
  2.  After the existing build_kg_aspect_init step, runs an additional init
      block that loads kg_canonical.csv via kg_loader and (optionally)
      overrides item_kg_aspects with relation-typed SVD and seeds
      user_global_emb with user-side KG SVD.

Everything else (model, data loaders, sampler, loss, eval, multi-seed
runner) imports directly from the root modules — no fork of those files.

Run from repo root:

    python -m relation_init.train
    # or
    python relation_init/train.py
"""
from __future__ import annotations

# Allow `python relation_init/train.py` to find root-level modules.
import sys as _sys, pathlib as _pathlib
_ROOT = _pathlib.Path(__file__).resolve().parent.parent
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))

import copy
import logging
import time
from statistics import mean, stdev

import torch
from torch.utils.data import DataLoader

from data import (
    KnowledgeAwareSampler,
    RecDataset,
    build_kg_aspect_init,
    build_kg_index,
    build_lightgcn_adj,
    load_interactions,
)
from evaluate import evaluate
from losses import aspect_level_cl, bpr_loss, infonce_loss
from model import RA_GARK
from utils import set_seed, user_stratified_split
from kg_loader import build_kg_index_v2

from relation_init.config import RelationInitConfig
from relation_init.kg_init import (
    build_relation_typed_aspect_init,
    build_user_kg_init,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def train_ragark(cfg: RelationInitConfig, device: torch.device) -> dict:
    set_seed(cfg.seed)

    df, user_enc, _, asin_to_idx, n_users, n_items = load_interactions(cfg.interaction_path)
    user_id_to_idx = (
        {u: i for i, u in enumerate(user_enc.classes_)}
        if cfg.use_canonical_kg else None
    )
    kg_adj, kg_rev_adj, _ = build_kg_index(
        cfg.kg_path, asin_to_idx, cfg.kg_stopwords, cfg.kg_top_freq_pct,
        use_canonical=cfg.use_canonical_kg,
        canonical_path=cfg.canonical_kg_path,
        canonical_prune_degree=cfg.canonical_kg_prune_degree,
        user_id_to_idx=user_id_to_idx,
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
        fusion_gate_style=cfg.fusion_gate_style,
    ).to(device)
    log.info(
        "flags: rat=%s(%s, τ=%.2f) svd=%s acl=%s ucl=%s global=%s fusion_bias=%.1f gate=%s "
        "| relation_typed=%s user_svd=%s",
        cfg.use_rationale, cfg.rationale_style, cfg.rationale_temperature,
        cfg.use_svd_init, cfg.use_acl, cfg.use_ucl,
        cfg.use_global_view, cfg.fusion_init_bias, cfg.fusion_gate_style,
        cfg.relation_typed_aspect_init, cfg.user_svd_init,
    )

    if cfg.use_svd_init:
        kg_init = build_kg_aspect_init(
            kg_adj, n_items, cfg.num_aspects, cfg.embedding_dim
        )
        if kg_init is not None:
            model.item_kg_aspects.data.copy_(kg_init.to(device))
            log.info("item_kg_aspects initialised from KG SVD embeddings")
    else:
        log.info("SVD init disabled — item_kg_aspects stays at xavier init")

    # ── Extra inits from canonical KG (this experiment's whole point) ──
    if cfg.relation_typed_aspect_init or cfg.user_svd_init:
        kg_v2 = build_kg_index_v2(
            cfg.canonical_kg_path,
            asin_to_idx,
            {u: i for i, u in enumerate(user_enc.classes_)},
            prune_degree=cfg.canonical_kg_prune_degree,
        )
        if cfg.relation_typed_aspect_init:
            rel_init = build_relation_typed_aspect_init(
                kg_v2, n_items, cfg.num_aspects, cfg.embedding_dim,
            )
            if rel_init is not None:
                model.item_kg_aspects.data.copy_(rel_init.to(device))
                log.info("item_kg_aspects re-initialised from relation-typed canonical KG")
        if cfg.user_svd_init:
            user_init = build_user_kg_init(kg_v2, n_users, cfg.embedding_dim)
            if user_init is not None:
                model.user_global_emb.weight.data.copy_(user_init.to(device))
                log.info("user_global_emb initialised from user-side canonical KG SVD")

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    log.info("Optimizer: Adam lr=%.1e", cfg.learning_rate)

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

            cached_embs = model._lightgcn_embeddings()
            pos_scores, u_loc, u_glo, i_pos_loc, _ = model(
                users, pos_items, cached_embs=cached_embs
            )
            neg_scores, *_ = model(users, neg_items, cached_embs=cached_embs)

            loss_bpr = bpr_loss(pos_scores, neg_scores)

            if cfg.use_acl:
                i_aspects = model.item_kg_aspects[pos_items]
                loss_acl = aspect_level_cl(
                    model.cl_projector, i_pos_loc, i_aspects, cfg.temp
                )
            else:
                loss_acl = torch.zeros((), device=device)

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
            k=cfg.eval_k, batch_size=cfg.eval_batch_size, extra_ks=cfg.eval_extra_ks,
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
        k=cfg.eval_k, batch_size=cfg.eval_batch_size, extra_ks=cfg.eval_extra_ks,
    )
    log.info("─" * 55)
    log.info("TEST metrics @ K=%d  (best epoch: %d)", cfg.eval_k, best_epoch)
    for metric, val in test_res.items():
        log.info("  %-12s %.4f", metric, val)
    log.info("─" * 55)

    return test_res


def _make_winner_cfg(seed: int) -> RelationInitConfig:
    cfg = RelationInitConfig()   # relation_typed_aspect_init / user_svd_init default True
    cfg.cl_weight = 0.005
    cfg.epochs = 80
    cfg.seed = seed
    cfg.use_rationale         = True
    cfg.use_svd_init          = True
    cfg.use_acl               = True
    cfg.use_ucl               = True
    cfg.use_global_view       = True
    cfg.rationale_style       = "mlp_softmax"
    cfg.rationale_temperature = 0.5
    cfg.fusion_init_bias      = 5.0
    cfg.fusion_gate_style     = "mlp"
    return cfg


def _fmt_pm(values: list[float]) -> str:
    if len(values) <= 1:
        return f"{values[0]:.4f}        " if values else "    n/a    "
    return f"{mean(values):.4f} ± {stdev(values):.4f}"


if __name__ == "__main__":
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", _device)

    # ── Direction check: does relation-typed init beat the 0.124 ceiling? ──
    # 1 seed × 3 configs paired (winner / scalar_gate / no_cl). Same shape
    # as the root multi-seed runner so deltas are directly comparable.
    seeds = [42]
    configs = [
        ("winner",       {}),
        ("scalar_gate",  {"fusion_gate_style": "scalar"}),
        ("no_cl",        {"use_acl": False, "use_ucl": False}),
    ]

    results: dict[str, dict[int, dict]] = {n: {} for n, _ in configs}

    for seed in seeds:
        for name, overrides in configs:
            cfg = _make_winner_cfg(seed)
            for k, v in overrides.items():
                setattr(cfg, k, v)
            cfg.model_save_path = f"best_relinit_{name}_seed{seed}.pth"

            log.info("\n%s", "=" * 90)
            log.info("RUN seed=%d | config=%s | RELATION-TYPED INIT", seed, name)
            log.info("%s", "=" * 90)

            try:
                test_res = train_ragark(cfg, _device)
            except Exception as e:
                log.exception("seed=%d %s FAILED: %s", seed, name, e)
                test_res = None
            results[name][seed] = test_res

    metrics = ("NDCG", "Recall", "HR", "MAP")
    log.info("\n%s", "=" * 100)
    log.info("RELATION-TYPED INIT RESULTS  (n=%d seeds: %s)", len(seeds), seeds)
    log.info("%s", "=" * 100)
    log.info("%-13s | %16s | %16s | %16s | %16s",
             "Config", "NDCG", "Recall", "HR", "MAP")
    log.info("%s", "-" * 100)
    for name, _ in configs:
        per_metric = {}
        for m in metrics:
            per_metric[m] = [
                r[m] for r in results[name].values() if r is not None
            ]
        log.info(
            "%-13s | %16s | %16s | %16s | %16s",
            name,
            _fmt_pm(per_metric["NDCG"]),
            _fmt_pm(per_metric["Recall"]),
            _fmt_pm(per_metric["HR"]),
            _fmt_pm(per_metric["MAP"]),
        )

    log.info("\n%s", "-" * 100)
    log.info("PAIRED DELTAS  (per-seed: winner − comparison)")
    log.info("%s", "-" * 100)
    for cmp_name in ("scalar_gate", "no_cl"):
        deltas = {m: [] for m in metrics}
        n_pos = 0
        for seed in seeds:
            r_w = results["winner"].get(seed)
            r_c = results[cmp_name].get(seed)
            if r_w is None or r_c is None:
                continue
            ndcg_d = r_w["NDCG"] - r_c["NDCG"]
            if ndcg_d > 0:
                n_pos += 1
            for m in metrics:
                deltas[m].append(r_w[m] - r_c[m])
        n = len(deltas["NDCG"])
        if n == 0:
            log.info("winner − %-12s : no data", cmp_name)
            continue
        log.info(
            "winner − %-12s : NDCG=%s  Recall=%s  HR=%s  MAP=%s   (winner higher in %d/%d seeds)",
            cmp_name,
            _fmt_pm(deltas["NDCG"]),
            _fmt_pm(deltas["Recall"]),
            _fmt_pm(deltas["HR"]),
            _fmt_pm(deltas["MAP"]),
            n_pos, n,
        )
    log.info("%s", "=" * 100)
