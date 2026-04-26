"""KG-driven init utilities (additive — does NOT replace data.build_kg_aspect_init).

Two new init functions consuming the canonical KG via kg_loader.KGIndexV2:

  build_relation_typed_aspect_init
    Builds item_kg_aspects of shape [num_items, num_aspects, dim] where
    each of the num_aspects rows corresponds to ONE canonical relation
    group instead of a single latent SVD component pulled from the
    pooled item-aspect matrix. Each group's edges go through their own
    TF-IDF + SVD, so the rationale-attention's per-aspect weights become
    interpretable as relation-type attention.

  build_user_kg_init
    SVD-init for user_global_emb (shape [num_users, dim]) using the
    user-side canonical KG edges. Parallel of the item-side trick that
    build_kg_aspect_init / build_relation_typed_aspect_init use, applied
    to the user side that was previously xavier-only.

Both functions are opt-in via config flags and don't disturb any
existing init path.
"""
from __future__ import annotations

# Allow `python relation_init/kg_init.py` from repo root to resolve root-level
# imports (kg_loader). Harmless when imported as a package member.
import sys as _sys, pathlib as _pathlib
_ROOT = _pathlib.Path(__file__).resolve().parent.parent
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))

import logging
from typing import List, Optional, Sequence, Set

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import svds

from kg_loader import KGIndexV2

log = logging.getLogger(__name__)

# Default item-side relation grouping. Top 3 by item coverage (HAS_PROPERTY 86%,
# DEPICTS 80%, IS_A 60%) get their own aspect; everything else falls into a
# 4th catch-all aspect. Match num_aspects=4 in Config.
DEFAULT_ITEM_RELATION_GROUPS: List[List[str]] = [
    ["HAS_PROPERTY"],
    ["DEPICTS"],
    ["IS_A"],
    ["SET_IN", "OTHER", "INFLUENCES",
     "POSITIVE_PREF", "NEGATIVE_PREF", "NEUTRAL_PREF", "INTERESTED_IN"],
]

# Default user-side relations to include in user_kg_init. Skips OTHER (vague)
# and uses both positive and interest signals; negative signals also kept since
# SVD is sign-invariant in factor structure (negative aspect membership is
# still a signal).
DEFAULT_USER_RELATION_FILTER: Set[str] = {
    "POSITIVE_PREF", "INTERESTED_IN", "NEGATIVE_PREF", "NEUTRAL_PREF",
    "HAS_PROPERTY", "DEPICTS", "IS_A",
}


def _tfidf_svd(M: sp.csr_matrix, k: int, num_rows: int) -> np.ndarray:
    """TF-IDF row weighting + truncated SVD. Returns U * sqrt(S) of shape [num_rows, k]."""
    df = np.asarray(M.sum(axis=0)).flatten()
    idf = np.log(num_rows / (df + 1.0)) + 1.0
    M = M.multiply(idf)

    max_k = min(M.shape) - 1
    k_eff = max(1, min(k, max_k))
    U, S, _ = svds(M, k=k_eff)
    order = np.argsort(-S)
    U, S = U[:, order], S[order]
    return U * np.sqrt(S)[np.newaxis, :]


def _pad_to(emb: np.ndarray, target_cols: int, seed: int) -> np.ndarray:
    """Pad columns with small-noise to reach target_cols if SVD rank too low."""
    cur = emb.shape[1]
    if cur >= target_cols:
        return emb[:, :target_cols]
    rng = np.random.RandomState(seed)
    pad = rng.randn(emb.shape[0], target_cols - cur) * 0.01
    return np.concatenate([emb, pad], axis=1)


def _rescale_to_xavier(arr: np.ndarray, fan_in: int, fan_out: int) -> np.ndarray:
    """In-place-style rescale array.std() to xavier_normal std for given fan_in/fan_out."""
    target_std = float(np.sqrt(2.0 / (fan_in + fan_out)))
    cur_std = float(arr.std())
    if cur_std > 0:
        arr = arr * (target_std / cur_std)
    return arr


def build_relation_typed_aspect_init(
    kg_v2: KGIndexV2,
    num_items: int,
    num_aspects: int,
    dim: int,
    relation_groups: Optional[Sequence[Sequence[str]]] = None,
):
    """Item-aspect tensor [num_items, num_aspects, dim], each aspect = one relation group.

    Returns torch.FloatTensor or None if the KG has no usable edges.
    """
    if relation_groups is None:
        relation_groups = DEFAULT_ITEM_RELATION_GROUPS
    if len(relation_groups) != num_aspects:
        raise ValueError(
            f"relation_groups must have {num_aspects} groups (one per aspect); "
            f"got {len(relation_groups)}"
        )

    log.info(
        "Relation-typed aspect init: %d items, %d aspects, dim=%d",
        num_items, num_aspects, dim,
    )

    aspect_arrays: List[np.ndarray] = []
    used_anywhere = False
    for group_idx, rels in enumerate(relation_groups):
        rels_set = set(rels)

        rows, cols = [], []
        ent_to_idx: dict[str, int] = {}
        for item_idx, pairs in kg_v2.item_kg_adj.items():
            for rel, ent in pairs:
                if rel not in rels_set:
                    continue
                ci = ent_to_idx.setdefault(ent, len(ent_to_idx))
                rows.append(item_idx)
                cols.append(ci)

        if not rows:
            log.warning(
                "  aspect %d (%s): no edges; using xavier init for this aspect",
                group_idx, "/".join(rels),
            )
            rng = np.random.RandomState(42 + group_idx)
            xavier_std = float(np.sqrt(2.0 / (num_aspects + dim)))
            aspect_arrays.append(rng.randn(num_items, dim) * xavier_std)
            continue

        used_anywhere = True
        n_ents = len(ent_to_idx)
        M = sp.csr_matrix(
            (np.ones(len(rows), dtype=np.float64), (rows, cols)),
            shape=(num_items, n_ents),
        )
        emb = _tfidf_svd(M, k=dim, num_rows=num_items)
        emb = _pad_to(emb, target_cols=dim, seed=42 + group_idx)
        aspect_arrays.append(emb)

        log.info(
            "  aspect %d (%s): %d edges, %d entities, SVD k=%d",
            group_idx, "/".join(rels), len(rows), n_ents, min(dim, n_ents - 1),
        )

    if not used_anywhere:
        log.warning("No relation group had any edges; falling back to None")
        return None

    stacked = np.stack(aspect_arrays, axis=1)  # [num_items, num_aspects, dim]
    stacked = _rescale_to_xavier(stacked, fan_in=num_aspects, fan_out=dim)

    import torch  # lazy: lets numpy-only smoke tests skip torch
    return torch.FloatTensor(stacked)


def build_user_kg_init(
    kg_v2: KGIndexV2,
    num_users: int,
    dim: int,
    relation_filter: Optional[Set[str]] = None,
):
    """SVD init for user_global_emb [num_users, dim] from user-side canonical KG.

    Returns torch.FloatTensor or None if user-side KG is empty.
    """
    if relation_filter is None:
        relation_filter = DEFAULT_USER_RELATION_FILTER

    rows, cols = [], []
    ent_to_idx: dict[str, int] = {}
    for user_idx, pairs in kg_v2.user_kg_adj.items():
        for rel, ent in pairs:
            if rel not in relation_filter:
                continue
            ci = ent_to_idx.setdefault(ent, len(ent_to_idx))
            rows.append(user_idx)
            cols.append(ci)

    if not rows:
        log.warning("User-side KG is empty after relation filter; skipping user SVD init")
        return None

    n_ents = len(ent_to_idx)
    M = sp.csr_matrix(
        (np.ones(len(rows), dtype=np.float64), (rows, cols)),
        shape=(num_users, n_ents),
    )
    log.info(
        "User SVD init: %d users, %d entities, %d edges (relations: %s)",
        num_users, n_ents, len(rows), sorted(relation_filter),
    )

    emb = _tfidf_svd(M, k=dim, num_rows=num_users)
    emb = _pad_to(emb, target_cols=dim, seed=42)
    # Rescale to xavier_normal std for an Embedding of shape [num_users, dim]
    emb = _rescale_to_xavier(emb, fan_in=num_users, fan_out=dim)

    import torch  # lazy
    return torch.FloatTensor(emb)


# ---------------------------------------------------------------------------
# CLI smoke test (numpy-only path; verifies SVD shapes without torch)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import pandas as pd

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    p = argparse.ArgumentParser()
    p.add_argument("--kg", default="data/kg_canonical.csv")
    p.add_argument("--rx", default="data/reviews_30_20.pkl")
    p.add_argument("--num-aspects", type=int, default=4)
    p.add_argument("--dim", type=int, default=128)
    args = p.parse_args()

    rx = pd.read_pickle(args.rx)
    rx = rx[rx["like"] == True].copy()
    rx["asin"] = rx["asin"].astype(str)
    rx["user_id"] = rx["user_id"].astype(str)
    asins = sorted(rx["asin"].unique())
    users = sorted(rx["user_id"].unique())
    asin_to_idx = {a: i for i, a in enumerate(asins)}
    user_id_to_idx = {u: i for i, u in enumerate(users)}

    from kg_loader import build_kg_index_v2
    kg = build_kg_index_v2(args.kg, asin_to_idx, user_id_to_idx, prune_degree=2)

    print()
    print("=" * 60)
    print("Smoke test: build_relation_typed_aspect_init")
    print("=" * 60)
    item_init = build_relation_typed_aspect_init(
        kg, num_items=len(asins),
        num_aspects=args.num_aspects, dim=args.dim,
    )
    if item_init is not None:
        print(f"  → item_init shape: {tuple(item_init.shape)}  "
              f"std: {item_init.std():.4f}  mean: {item_init.mean():+.4f}")

    print()
    print("=" * 60)
    print("Smoke test: build_user_kg_init")
    print("=" * 60)
    user_init = build_user_kg_init(kg, num_users=len(users), dim=args.dim)
    if user_init is not None:
        print(f"  → user_init shape: {tuple(user_init.shape)}  "
              f"std: {user_init.std():.4f}  mean: {user_init.mean():+.4f}")
