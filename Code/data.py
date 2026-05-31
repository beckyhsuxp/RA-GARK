from __future__ import annotations

import logging
import os
import random
from collections import defaultdict
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Interactions
# ---------------------------------------------------------------------------

def load_interactions(
    path: str,
) -> Tuple[pd.DataFrame, LabelEncoder, LabelEncoder, Dict, int, int]:
    """Load, clean, and label-encode the interaction DataFrame."""
    log.info("Loading interaction data from %s", path)
    df = pd.read_pickle(path)

    if df["like"].dtype == "object":
        df["like"] = df["like"].map(
            {"True": True, "False": False, True: True, False: False}
        )

    original_len = len(df)
    df = df[df["like"] == True].copy()
    log.info("Filtered to positive interactions: %d → %d rows", original_len, len(df))

    if len(df) == 0:
        raise ValueError("No positive interactions found after filtering 'like' column.")

    user_enc = LabelEncoder()
    item_enc = LabelEncoder()
    df["user_idx"] = user_enc.fit_transform(df["user_id"])
    df["item_idx"] = item_enc.fit_transform(df["asin"])

    n_users = int(df["user_idx"].max()) + 1
    n_items = int(df["item_idx"].max()) + 1
    asin_to_idx = dict(zip(df["asin"], df["item_idx"]))

    log.info("Users: %d  |  Items: %d", n_users, n_items)
    return df, user_enc, item_enc, asin_to_idx, n_users, n_items


# ---------------------------------------------------------------------------
# Knowledge Graph
# ---------------------------------------------------------------------------

def build_kg_index(
    kg_path: str,
    asin_to_idx: Dict,
    stopwords: Set[str],
    top_freq_pct: float,
    *,
    use_canonical: bool = False,
    canonical_path: str = "data/kg_canonical.csv",
    canonical_prune_degree: int = 2,
    user_id_to_idx: Dict[str, int] | None = None,
) -> Tuple[Dict[int, List], Dict[str, List], Set[str]]:
    """Build KG adjacency (item→aspects, aspect→items) with graph pruning.

    If ``use_canonical=True``, loads from the canonicalised KG produced by
    ``kg_clean.py`` via ``kg_loader.build_kg_index_v2`` and returns the
    legacy-shape aliases (relation type stripped) so the existing model
    stays drop-in compatible. ``user_id_to_idx`` should be passed when
    ``use_canonical=True`` to also index the user-side KG (otherwise
    user-side edges are skipped).
    """
    if use_canonical:
        from kg_loader import build_kg_index_v2

        v2 = build_kg_index_v2(
            canonical_path,
            asin_to_idx,
            user_id_to_idx,
            prune_degree=canonical_prune_degree,
        )
        log.info(
            "Canonical KG: %d items, %d users, %d bridge entities, %d-entity vocab",
            len(v2.item_kg_adj), len(v2.user_kg_adj),
            len(v2.bridge_entities), len(v2.entity_to_idx),
        )
        legacy_adj: Dict[int, List] = defaultdict(list, v2.legacy_kg_adj)
        legacy_rev: Dict[str, List] = defaultdict(list, v2.legacy_kg_rev_adj)
        return legacy_adj, legacy_rev, v2.legacy_aspect_set

    if not os.path.exists(kg_path):
        log.warning("KG file not found at %s — KG features disabled.", kg_path)
        return defaultdict(list), defaultdict(list), set()

    log.info("Building KG index from %s", kg_path)
    df_kg = pd.read_csv(kg_path) if kg_path.endswith(".csv") else pd.read_pickle(kg_path)
    log.info("Raw KG edges: %d", len(df_kg))

    aspect_counts = df_kg["node_2"].value_counts()
    cutoff = max(1, int(len(aspect_counts) * top_freq_pct))
    high_freq = set(aspect_counts.head(cutoff).index)
    bad_aspects = high_freq | stopwords
    log.info(
        "Pruning %d noisy aspect nodes (top %.0f%% freq + stopwords)",
        len(bad_aspects), top_freq_pct * 100,
    )

    mask = df_kg["node_1"].isin(asin_to_idx) & ~df_kg["node_2"].isin(bad_aspects)
    df_valid = df_kg[mask].copy()
    df_valid["item_idx"] = df_valid["node_1"].map(asin_to_idx)
    log.info("Valid edges: %d (skipped %d)", len(df_valid), len(df_kg) - len(df_valid))

    kg_adj: Dict[int, List] = defaultdict(list)
    kg_rev_adj: Dict[str, List] = defaultdict(list)
    for item_idx, aspect in zip(df_valid["item_idx"].values, df_valid["node_2"].values):
        kg_adj[item_idx].append(aspect)
        kg_rev_adj[aspect].append(item_idx)

    aspect_set = set(df_valid["node_2"].unique())
    log.info(
        "Unique aspects: %d  |  Items with KG info: %d", len(aspect_set), len(kg_adj)
    )
    return kg_adj, kg_rev_adj, aspect_set


# ---------------------------------------------------------------------------
# Sampler & Dataset
# ---------------------------------------------------------------------------

def build_kg_aspect_init(
    kg_adj: Dict[int, List],
    num_items: int,
    num_aspects: int,
    dim: int,
) -> torch.Tensor | None:
    """Initialise item-aspect embeddings from KG co-occurrence via TF-IDF + SVD.

    Returns a [num_items, num_aspects, dim] tensor, or None if no KG data.
    The SVD output is rescaled so its std matches xavier_normal, keeping
    init magnitude consistent with non-SVD parameters.
    """
    all_aspects = sorted({a for aspects in kg_adj.values() for a in aspects})
    if not all_aspects:
        log.warning("No KG aspects found; falling back to random init.")
        return None

    aspect_to_idx = {a: i for i, a in enumerate(all_aspects)}
    n_kg_aspects = len(all_aspects)
    log.info(
        "KG SVD init: %d items with KG, %d unique aspects",
        len(kg_adj), n_kg_aspects,
    )

    # Sparse binary co-occurrence matrix [num_items × n_kg_aspects]
    rows, cols = [], []
    for item_idx, aspects in kg_adj.items():
        for asp in aspects:
            rows.append(item_idx)
            cols.append(aspect_to_idx[asp])

    M = sp.csr_matrix(
        (np.ones(len(rows), dtype=np.float64), (rows, cols)),
        shape=(num_items, n_kg_aspects),
    )

    # IDF weighting — downweight aspects shared by many items
    df = np.asarray(M.sum(axis=0)).flatten()
    idf = np.log(num_items / (df + 1.0)) + 1.0
    M = M.multiply(idf)

    # Truncated SVD
    target_k = num_aspects * dim                       # e.g. 4 × 128 = 512
    max_k = min(num_items, n_kg_aspects) - 1
    k = min(target_k, max_k)

    from scipy.sparse.linalg import svds
    U, S, _ = svds(M, k=k)

    # Sort descending by singular value
    order = np.argsort(-S)
    U, S = U[:, order], S[order]

    emb = U * np.sqrt(S)[np.newaxis, :]                # [num_items, k]

    # Pad if rank < target_k
    if emb.shape[1] < target_k:
        rng = np.random.RandomState(42)
        pad = rng.randn(num_items, target_k - emb.shape[1]) * 0.01
        emb = np.concatenate([emb, pad], axis=1)
    else:
        emb = emb[:, :target_k]

    xavier_std = float(np.sqrt(2.0 / (num_aspects + dim)))
    cur_std = float(emb.std())
    if cur_std > 0:
        emb *= xavier_std / cur_std

    emb = emb.reshape(num_items, num_aspects, dim)
    log.info("KG SVD init: %d SVD components, rescaled std → %.4f", k, xavier_std)
    return torch.FloatTensor(emb)


# ---------------------------------------------------------------------------
# Sampler & Dataset
# ---------------------------------------------------------------------------

class KnowledgeAwareSampler:
    """Random negative sampling + KG-based semantic neighbor lookup."""

    def __init__(
        self,
        num_items: int,
        kg_adj: Dict[int, List],
        kg_rev_adj: Dict[str, List],
        user_pos_items: Dict[int, Set[int]] | None = None,
    ) -> None:
        self.num_items = num_items
        self.kg_adj = kg_adj
        self.kg_rev_adj = kg_rev_adj
        self.user_pos_items = user_pos_items or {}

    def get_kg_neighbor(self, item_idx: int) -> Tuple[int, bool]:
        aspects = self.kg_adj.get(item_idx, [])
        if not aspects:
            return item_idx, False
        candidates = self.kg_rev_adj.get(random.choice(aspects), [])
        if not candidates:
            return item_idx, False
        for _ in range(5):
            neighbor = random.choice(candidates)
            if neighbor != item_idx:
                return neighbor, True
        return item_idx, False

    def random_negative(self, user_idx: int) -> int:
        positives = self.user_pos_items.get(int(user_idx), set())
        if len(positives) >= self.num_items:
            raise ValueError(f"user {user_idx} has no available negative items")

        for _ in range(100):
            item = int(np.random.randint(0, self.num_items))
            if item not in positives:
                return item

        # Dense-user fallback: deterministic scan avoids an infinite loop.
        start = int(np.random.randint(0, self.num_items))
        for offset in range(self.num_items):
            item = (start + offset) % self.num_items
            if item not in positives:
                return item
        raise ValueError(f"user {user_idx} has no available negative items")


class RecDataset(Dataset):
    def __init__(
        self,
        user_idx: np.ndarray,
        item_idx: np.ndarray,
        sampler: KnowledgeAwareSampler,
    ) -> None:
        self.users = torch.LongTensor(
            user_idx.values if hasattr(user_idx, "values") else user_idx
        )
        self.items = torch.LongTensor(
            item_idx.values if hasattr(item_idx, "values") else item_idx
        )
        self.sampler = sampler

    def __len__(self) -> int:
        return len(self.users)

    def __getitem__(self, idx: int):
        user = self.users[idx]
        pos_item = self.items[idx]
        neg_item = self.sampler.random_negative(user.item())
        neighbor, _ = self.sampler.get_kg_neighbor(pos_item.item())
        return user, pos_item, torch.tensor(neg_item), torch.tensor(neighbor)


# ---------------------------------------------------------------------------
# LightGCN adjacency matrix
# ---------------------------------------------------------------------------

def build_lightgcn_adj(
    train_df: pd.DataFrame,
    n_users: int,
    n_items: int,
    device: torch.device,
) -> torch.Tensor:
    """Build the symmetric normalised bipartite adjacency matrix for LightGCN."""
    log.info("Building LightGCN sparse adjacency matrix...")
    users = train_df["user_idx"].values
    items = train_df["item_idx"].values

    R = sp.coo_matrix(
        (np.ones(len(users)), (users, items)), shape=(n_users, n_items)
    )
    A = sp.bmat([[None, R], [R.T, None]], format="coo")

    rowsum = np.asarray(A.sum(axis=1)).flatten()
    with np.errstate(divide="ignore"):
        d_inv_sqrt = np.where(rowsum == 0, 0.0, np.power(rowsum, -0.5))
    D_inv_sqrt = sp.diags(d_inv_sqrt)
    A_tilde = D_inv_sqrt.dot(A).dot(D_inv_sqrt).tocoo()

    indices = torch.LongTensor(np.stack([A_tilde.row, A_tilde.col]))
    values = torch.FloatTensor(A_tilde.data)
    sparse_adj = torch.sparse_coo_tensor(indices, values, A_tilde.shape)
    log.info("Adjacency matrix shape: %s", tuple(A_tilde.shape))
    return sparse_adj.to(device)
