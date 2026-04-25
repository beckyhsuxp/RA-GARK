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
) -> Tuple[Dict[int, List], Dict[str, List], Set[str]]:
    """Build KG adjacency (item→aspects, aspect→items) with graph pruning."""
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

def _tfidf_svd_init(
    M: sp.csr_matrix,
    num_aspects: int,
    dim: int,
    label: str,
) -> torch.Tensor:
    """Shared TF-IDF + truncated-SVD + xavier-rescale recipe.

    M: sparse [n_rows × n_cols] co-occurrence matrix.
    Returns: [n_rows, num_aspects, dim] tensor whose std matches
    xavier_normal so the init blends with non-SVD parameters.
    """
    n_rows = M.shape[0]

    # IDF weighting — downweight columns (aspects) shared by many rows.
    df = np.asarray(M.sum(axis=0)).flatten()
    idf = np.log(n_rows / (df + 1.0)) + 1.0
    M = M.multiply(idf)

    target_k = num_aspects * dim
    max_k = min(M.shape[0], M.shape[1]) - 1
    k = min(target_k, max_k)

    from scipy.sparse.linalg import svds
    U, S, _ = svds(M, k=k)

    order = np.argsort(-S)
    U, S = U[:, order], S[order]
    emb = U * np.sqrt(S)[np.newaxis, :]                # [n_rows, k]

    if emb.shape[1] < target_k:
        rng = np.random.RandomState(42)
        pad = rng.randn(n_rows, target_k - emb.shape[1]) * 0.01
        emb = np.concatenate([emb, pad], axis=1)
    else:
        emb = emb[:, :target_k]

    xavier_std = float(np.sqrt(2.0 / (num_aspects + dim)))
    cur_std = float(emb.std())
    if cur_std > 0:
        emb *= xavier_std / cur_std

    emb = emb.reshape(n_rows, num_aspects, dim)
    log.info("%s SVD init: %d components, rescaled std → %.4f",
             label, k, xavier_std)
    return torch.FloatTensor(emb)


def build_kg_aspect_init(
    kg_adj: Dict[int, List],
    num_items: int,
    num_aspects: int,
    dim: int,
) -> torch.Tensor | None:
    """Initialise item-aspect embeddings from item×aspect KG via TF-IDF + SVD.

    Returns [num_items, num_aspects, dim] or None if KG is empty.
    """
    all_aspects = sorted({a for aspects in kg_adj.values() for a in aspects})
    if not all_aspects:
        log.warning("No KG aspects found; falling back to random init.")
        return None

    aspect_to_idx = {a: i for i, a in enumerate(all_aspects)}
    n_kg_aspects = len(all_aspects)
    log.info(
        "Item-aspect init: %d items with KG, %d unique aspects",
        len(kg_adj), n_kg_aspects,
    )

    rows, cols = [], []
    for item_idx, aspects in kg_adj.items():
        for asp in aspects:
            rows.append(item_idx)
            cols.append(aspect_to_idx[asp])

    M = sp.csr_matrix(
        (np.ones(len(rows), dtype=np.float64), (rows, cols)),
        shape=(num_items, n_kg_aspects),
    )
    return _tfidf_svd_init(M, num_aspects, dim, label="Item-aspect")


def build_user_aspect_init(
    train_df: pd.DataFrame,
    kg_adj: Dict[int, List],
    num_users: int,
    num_aspects: int,
    dim: int,
) -> torch.Tensor | None:
    """Initialise user-aspect embeddings from user×aspect TF-IDF + SVD.

    M[u, a] = count of training interactions where user u liked an item
    that has aspect a in the KG. This propagates KG semantics to the
    user side via observed preferences, so user_kg_aspects starts with
    a sensible aspect geometry rather than xavier noise.

    Returns [num_users, num_aspects, dim] or None if KG is empty.
    """
    all_aspects = sorted({a for aspects in kg_adj.values() for a in aspects})
    if not all_aspects:
        log.warning("No KG aspects for user-aspect init.")
        return None

    aspect_to_idx = {a: i for i, a in enumerate(all_aspects)}
    n_kg_aspects = len(all_aspects)

    rows, cols = [], []
    for u, item_idx in zip(
        train_df["user_idx"].values, train_df["item_idx"].values
    ):
        for asp in kg_adj.get(int(item_idx), []):
            rows.append(int(u))
            cols.append(aspect_to_idx[asp])

    if not rows:
        log.warning("No (user, aspect) co-occurrences; falling back to xavier.")
        return None

    M = sp.csr_matrix(
        (np.ones(len(rows), dtype=np.float64), (rows, cols)),
        shape=(num_users, n_kg_aspects),
    )
    log.info(
        "User-aspect init: %d (u,a) co-occurrences across %d users",
        M.nnz, num_users,
    )
    return _tfidf_svd_init(M, num_aspects, dim, label="User-aspect")


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
    ) -> None:
        self.num_items = num_items
        self.kg_adj = kg_adj
        self.kg_rev_adj = kg_rev_adj

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

    def random_negative(self) -> int:
        return int(np.random.randint(0, self.num_items))


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
        neg_item = self.sampler.random_negative()
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
