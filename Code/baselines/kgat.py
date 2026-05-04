"""KGAT baseline.

Reference: Wang et al., "KGAT: Knowledge Graph Attention Network for
Recommendation", KDD 2019.

This is a faithful but minimal port:
  - Builds the Collaborative Knowledge Graph (CKG) from user-item
    interactions and item-aspect KG edges.
  - Uses KGAT's bi-interaction aggregator
        e_h^(l+1) = LeakyReLU(W1 (e_h + agg)) + LeakyReLU(W2 (e_h ⊙ agg))
  - Edge-level attention computed per forward pass and applied via a
    sparse weighted adjacency.
  - The TransR auxiliary loss is omitted because the underlying KG has
    only one relation type (item--has_aspect--aspect), making it
    degenerate.

Exposes _lightgcn_embeddings() and score_all_items() so it plugs into
evaluate.evaluate() unchanged.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CKG construction
# ---------------------------------------------------------------------------

def build_ckg(
    train_df,
    kg_adj: Dict[int, List[str]],
    n_users: int,
    n_items: int,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """Build the (symmetric) edge_index of the collaborative knowledge graph.

    Node id layout (concatenated):
      [0,         n_users)                       → users
      [n_users,   n_users + n_items)             → items
      [n_users+n_items, n_users+n_items+n_asp)   → aspects

    Returns
    -------
    edge_index : LongTensor [2, n_edges]   directed edges (both directions
                                           included for undirected propagation)
    deg        : FloatTensor [n_total]     node degrees (for D^-1/2 normalisation)
    n_aspects  : int                       number of unique aspect nodes
    """
    aspect_set = sorted({a for aspects in kg_adj.values() for a in aspects})
    aspect_to_idx = {a: i for i, a in enumerate(aspect_set)}
    n_aspects = len(aspect_set)
    n_total = n_users + n_items + n_aspects

    # User-item interaction edges
    ui_u = train_df["user_idx"].to_numpy()
    ui_i = train_df["item_idx"].to_numpy() + n_users

    # Item-aspect KG edges
    ia_i, ia_a = [], []
    for item_idx, aspects in kg_adj.items():
        for asp in aspects:
            ia_i.append(item_idx + n_users)
            ia_a.append(aspect_to_idx[asp] + n_users + n_items)
    ia_i = np.asarray(ia_i, dtype=np.int64)
    ia_a = np.asarray(ia_a, dtype=np.int64)

    src = np.concatenate([ui_u, ui_i, ia_i, ia_a])
    dst = np.concatenate([ui_i, ui_u, ia_a, ia_i])

    edge_index = torch.from_numpy(np.stack([src, dst])).long()

    # Degrees for symmetric normalisation
    deg = np.zeros(n_total, dtype=np.float32)
    np.add.at(deg, src, 1.0)
    deg = torch.from_numpy(deg)

    log.info(
        "CKG: %d users, %d items, %d aspects, %d total nodes, %d directed edges",
        n_users, n_items, n_aspects, n_total, edge_index.shape[1],
    )
    return edge_index, deg, n_aspects


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class KGAT(nn.Module):
    def __init__(
        self,
        n_users: int,
        n_items: int,
        n_aspects: int,
        edge_index: torch.Tensor,
        deg: torch.Tensor,
        dim: int = 64,
        n_layers: int = 2,
    ) -> None:
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.n_aspects = n_aspects
        self.n_total = n_users + n_items + n_aspects
        self.dim = dim
        self.n_layers = n_layers

        self.entity_emb = nn.Embedding(self.n_total, dim)
        nn.init.xavier_normal_(self.entity_emb.weight)

        self.W1 = nn.ModuleList([nn.Linear(dim, dim, bias=False) for _ in range(n_layers)])
        self.W2 = nn.ModuleList([nn.Linear(dim, dim, bias=False) for _ in range(n_layers)])
        for layer in list(self.W1) + list(self.W2):
            nn.init.xavier_uniform_(layer.weight)

        self.register_buffer("edge_index", edge_index)
        self.register_buffer("deg_inv_sqrt", torch.where(
            deg > 0, deg.pow(-0.5), torch.zeros_like(deg)
        ))

    def _edge_softmax(self, scores: torch.Tensor, src: torch.Tensor) -> torch.Tensor:
        """Softmax of edge scores grouped by source node."""
        n = self.n_total
        # max per source for numerical stability
        smax = torch.full((n,), float("-inf"), device=scores.device)
        smax.scatter_reduce_(0, src, scores, reduce="amax", include_self=False)
        # rows with no edges become -inf; clamp to 0 to avoid nan in exp(0 - (-inf))
        smax = torch.where(torch.isinf(smax), torch.zeros_like(smax), smax)
        exp_s = (scores - smax[src]).exp()

        ssum = torch.zeros(n, device=scores.device)
        ssum.index_add_(0, src, exp_s)
        return exp_s / (ssum[src] + 1e-12)

    def _propagate(self) -> torch.Tensor:
        """Run KGAT-style attentive propagation; return concatenated layer embs."""
        x = self.entity_emb.weight
        src, dst = self.edge_index[0], self.edge_index[1]

        outs = [x]
        for l in range(self.n_layers):
            h_src = x[src]                                    # [E, d]
            h_dst = x[dst]                                    # [E, d]

            # Edge attention: scaled dot-product on the *current* layer embs
            scores = (h_src * h_dst).sum(dim=-1) / (self.dim ** 0.5)
            alpha = self._edge_softmax(scores, src)           # [E]

            # Weighted aggregation: agg[v] = Σ alpha * D^-1/2 * D^-1/2 * x_dst
            norm = self.deg_inv_sqrt[src] * self.deg_inv_sqrt[dst]
            messages = h_dst * (alpha * norm).unsqueeze(-1)
            agg = torch.zeros_like(x)
            agg.index_add_(0, src, messages)

            # Bi-interaction aggregator
            x = F.leaky_relu(self.W1[l](x + agg)) \
                + F.leaky_relu(self.W2[l](x * agg))
            x = F.normalize(x, p=2, dim=-1)
            outs.append(x)

        return torch.cat(outs, dim=-1)                       # [n_total, d * (L+1)]

    def _lightgcn_embeddings(self) -> Tuple[torch.Tensor, torch.Tensor]:
        all_emb = self._propagate()
        users = all_emb[: self.n_users]
        items = all_emb[self.n_users : self.n_users + self.n_items]
        return users, items

    def forward(self, u_idx: torch.Tensor, i_idx: torch.Tensor) -> torch.Tensor:
        all_u, all_i = self._lightgcn_embeddings()
        u = all_u[u_idx]
        i = all_i[i_idx]
        return (u * i).sum(dim=-1)

    @torch.no_grad()
    def score_all_items(
        self,
        u_idx: torch.Tensor,
        cached_embs: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        if cached_embs is not None:
            all_u, all_i = cached_embs
        else:
            all_u, all_i = self._lightgcn_embeddings()
        return all_u[u_idx] @ all_i.T
