"""
Interpretability case study for RA-GARK's softmax rationale masking.

For each of N sampled items, print:
  * the item's raw KG aspects (for semantic context)
  * the 4 softmax aspect-attention weights produced by the trained
    rationale module for K different users who interacted with the item

If users attend to different aspect slots of the same item → the
rationale module is learning user-specific aspect preferences, and the
"rationale-aware" claim is visible to the reader.

Run:
    python case_study.py                     # winner defaults
    python case_study.py --checkpoint PATH   # specify checkpoint
    python case_study.py --num_items 8 --users_per_item 4
Outputs:
    case_study.txt        (human-readable, also printed to stdout)
    case_study.csv        (machine-readable; item, user, top_aspects, w1..wA)
"""

from __future__ import annotations

import argparse
import csv
import glob
import logging
import os
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from config import Config
from data import (
    build_kg_index,
    build_lightgcn_adj,
    load_interactions,
)
from model import RA_GARK
from utils import set_seed, user_stratified_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("case")


def find_winner_checkpoint() -> str | None:
    """Locate a winner checkpoint regardless of which script trained it.

    train_ragark.py format:
      best_ragark_rat1-mlp_softmax_svd1_acl1_ucl1_gv1_fb5.pth
    run_ablations.py format:
      best_ragark_rationale1_svd_init1_acl1_ucl1_global_view1_style-mlp_softmax_fb5.pth

    Winner signature: mlp_softmax + fb5 + every flag ON (no 0-valued flag,
    and no `_no_` in the filename from run_ablations preset names).
    Most recently modified match wins.
    """
    all_files = glob.glob("best_ragark_*.pth")
    off_markers = (
        "svd0", "svd_init0", "acl0", "ucl0",
        "gv0", "global_view0", "rat0", "rationale0",
    )
    candidates = [
        f for f in all_files
        if (("mlp_softmax" in f and "fb5" in f) or "winner" in f)
        and not any(m in f for m in off_markers)
        and "_no_" not in f
    ]
    if not candidates:
        return None
    candidates.sort(key=os.path.getmtime, reverse=True)
    return candidates[0]


def _rank_items_for_case_study(train_df, kg_adj, top_n: int = 20) -> list[int]:
    """Prefer items that (a) have >=3 KG aspects and (b) are rated by many users."""
    item_pop = train_df["item_idx"].value_counts()
    candidates = []
    for item_idx, pop in item_pop.items():
        aspects = kg_adj.get(int(item_idx), [])
        if len(aspects) >= 3:
            candidates.append((int(item_idx), int(pop), aspects))
    candidates.sort(key=lambda r: r[1], reverse=True)
    return candidates[:top_n]


def _pick_users_for_item(train_df, item_idx: int, k: int) -> list[int]:
    users = train_df.loc[train_df["item_idx"] == item_idx, "user_idx"].unique().tolist()
    if len(users) <= k:
        return users
    rng = np.random.RandomState(42)
    return rng.choice(users, size=k, replace=False).tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=None,
                        help="Path to .pth. If omitted, auto-detects winner checkpoint.")
    parser.add_argument("--num_items", type=int, default=5)
    parser.add_argument("--users_per_item", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    checkpoint = args.checkpoint or find_winner_checkpoint()
    if checkpoint is None:
        log.error("No winner checkpoint found. Train first:")
        log.error("  python train_ragark.py              (fast)")
        log.error("  python run_ablations.py --mode minimal")
        return

    cfg = Config()
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)
    log.info("Checkpoint: %s", checkpoint)

    # ── Data ───────────────────────────────────────────────────────────
    df, _, item_enc, asin_to_idx, n_users, n_items = load_interactions(cfg.interaction_path)
    kg_adj, _, _ = build_kg_index(cfg.kg_path, asin_to_idx, cfg.kg_stopwords, cfg.kg_top_freq_pct)
    train_df, _, _ = user_stratified_split(df, val_ratio=0.15, test_ratio=0.15, seed=cfg.seed)
    adj = build_lightgcn_adj(train_df, n_users, n_items, device)

    # Reverse mapping: item_idx → asin (for human-readable output)
    idx_to_asin = {v: k for k, v in asin_to_idx.items()}

    # ── Model ──────────────────────────────────────────────────────────
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
        fusion_init_bias=cfg.fusion_init_bias,
    ).to(device)

    if not Path(checkpoint).exists():
        log.error("Checkpoint not found: %s", checkpoint)
        return

    state = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    log.info("Model: rationale_style=%s  fusion_init_bias=%.1f",
             cfg.rationale_style, cfg.fusion_init_bias)

    # ── Pick items ─────────────────────────────────────────────────────
    candidates = _rank_items_for_case_study(train_df, kg_adj, top_n=30)
    if len(candidates) < args.num_items:
        log.warning("Only %d items with ≥3 KG aspects; using all of them.", len(candidates))
    picked = candidates[: args.num_items]

    out_txt = ["RA-GARK softmax rationale — per-user aspect attention",
               "=" * 70]
    csv_rows = []

    with torch.no_grad():
        for item_idx, pop, aspects in picked:
            users = _pick_users_for_item(train_df, item_idx, args.users_per_item)
            top_aspects = [a for a, _ in Counter(aspects).most_common(6)]
            asin = idx_to_asin.get(item_idx, "<unk>")

            block = [
                f"\n── Item {item_idx}  (asin={asin}, trained by {pop} users) ──",
                f"  top KG aspects: {', '.join(top_aspects)}",
                f"  aspect attention (softmax over {cfg.num_aspects} aspect slots):",
            ]

            u_idx = torch.LongTensor(users).to(device)
            i_idx = torch.LongTensor([item_idx] * len(users)).to(device)

            u_glo = model.user_global_emb(u_idx)                  # [U, d]
            i_aspects = model.item_kg_aspects[i_idx]              # [U, A, d]
            weights = model.rationale_masking._weights(u_glo, i_aspects).squeeze(-1)   # [U, A]
            weights = weights.cpu().numpy()

            for user, w in zip(users, weights):
                w_fmt = "[" + ", ".join(f"{x:.3f}" for x in w) + "]"
                block.append(f"    user {user:>4}  →  {w_fmt}   (max idx = aspect#{int(w.argmax())})")
                csv_rows.append({
                    "item_idx": item_idx, "asin": asin,
                    "user_idx": int(user),
                    "top_kg_aspects": "|".join(top_aspects),
                    **{f"w{a}": float(w[a]) for a in range(cfg.num_aspects)},
                    "argmax_aspect": int(w.argmax()),
                })
            out_txt.extend(block)

    out_txt.append("\n" + "=" * 70)
    out_txt.append(
        "Reading guide: each aspect slot is an abstract learned dimension\n"
        "(initialised from KG SVD, then fine-tuned). Slot identities are\n"
        "consistent *per item* but not globally comparable across items.\n"
        "The signal to look for is whether different users of the SAME item\n"
        "spread their attention differently — that demonstrates the\n"
        "rationale-aware aspect selection is user-conditioned.\n"
    )

    rendered = "\n".join(out_txt)
    print(rendered)
    Path("case_study.txt").write_text(rendered)

    with open("case_study.csv", "w", newline="") as f:
        fieldnames = (
            ["item_idx", "asin", "user_idx", "top_kg_aspects"]
            + [f"w{a}" for a in range(cfg.num_aspects)]
            + ["argmax_aspect"]
        )
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(csv_rows)

    log.info("Wrote case_study.txt and case_study.csv (%d rows)", len(csv_rows))


if __name__ == "__main__":
    main()
