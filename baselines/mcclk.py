"""MCCLK baseline.

Reference: Zou et al., "Multi-level Cross-view Contrastive Learning for
Knowledge-aware Recommendation", SIGIR 2022.

Core ideas preserved here:
  1. Three structural views of the data:
       - Local view  : user-item interaction edges only
       - Semantic    : item-aspect KG edges only
       - Global view : full collaborative knowledge graph (CKG)
  2. Cross-view InfoNCE contrastive learning that aligns:
       - user reps  between local and global views
       - item reps  across local, semantic, and global views
  3. Joint optimisation with BPR over the global view.

Differences from the official paper:
  - Single relation type → no relation embeddings.
  - Bi-interaction aggregator borrowed from KGAT for all three views.
  - We split the unified CKG edge set into ui- and ia-subsets so all
    three views share the same node id space and embedding tables, which
    makes inter-view CL well defined.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class MCCLK(nn.Module):
    def __init__(
        self,
        n_users: int,
        n_items: int,
        n_aspects: int,
        edge_index_ui: torch.Tensor,        # local view edges
        edge_index_ia: torch.Tensor,        # semantic view edges
        edge_index_full: torch.Tensor,      # global view edges
        deg: torch.Tensor,                  # degrees on the full graph
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

        self.register_buffer("edge_index_ui", edge_index_ui)
        self.register_buffer("edge_index_ia", edge_index_ia)
        self.register_buffer("edge_index_full", edge_index_full)
        self.register_buffer("deg_inv_sqrt", torch.where(
            deg > 0, deg.pow(-0.5), torch.zeros_like(deg)
        ))

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _edge_softmax(self, scores: torch.Tensor, src: torch.Tensor) -> torch.Tensor:
        n = self.n_total
        smax = torch.full((n,), float("-inf"), device=scores.device)
        smax.scatter_reduce_(0, src, scores, reduce="amax", include_self=False)
        smax = torch.where(torch.isinf(smax), torch.zeros_like(smax), smax)
        exp_s = (scores - smax[src]).exp()
        ssum = torch.zeros(n, device=scores.device)
        ssum.index_add_(0, src, exp_s)
        return exp_s / (ssum[src] + 1e-12)

    # ------------------------------------------------------------------ #
    # Propagation (single edge subset)
    # ------------------------------------------------------------------ #

    def _propagate(self, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.entity_emb.weight
        outs = [x]

        for l in range(self.n_layers):
            src, dst = edge_index[0], edge_index[1]
            scores = (x[src] * x[dst]).sum(dim=-1) / (self.dim ** 0.5)
            alpha = self._edge_softmax(scores, src)

            norm = self.deg_inv_sqrt[src] * self.deg_inv_sqrt[dst]
            messages = x[dst] * (alpha * norm).unsqueeze(-1)
            agg = torch.zeros_like(x)
            agg.index_add_(0, src, messages)

            x = F.leaky_relu(self.W1[l](x + agg)) \
                + F.leaky_relu(self.W2[l](x * agg))
            x = F.normalize(x, p=2, dim=-1)
            outs.append(x)

        return torch.cat(outs, dim=-1)

    # ------------------------------------------------------------------ #
    # Three-view propagation for training
    # ------------------------------------------------------------------ #

    def propagate_three_views(
        self,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (local, semantic, global) embeddings."""
        emb_local    = self._propagate(self.edge_index_ui)
        emb_semantic = self._propagate(self.edge_index_ia)
        emb_global   = self._propagate(self.edge_index_full)
        return emb_local, emb_semantic, emb_global

    # ------------------------------------------------------------------ #
    # Inference / evaluation interface (matches RAKG_LMR)
    # ------------------------------------------------------------------ #

    def _lightgcn_embeddings(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Use the global view for evaluation."""
        emb = self._propagate(self.edge_index_full)
        users = emb[: self.n_users]
        items = emb[self.n_users : self.n_users + self.n_items]
        return users, items

    def forward(self, u_idx: torch.Tensor, i_idx: torch.Tensor) -> torch.Tensor:
        all_u, all_i = self._lightgcn_embeddings()
        return (all_u[u_idx] * all_i[i_idx]).sum(dim=-1)

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


# ---------------------------------------------------------------------------
# Three-view edge construction
# ---------------------------------------------------------------------------

def build_mcclk_views(
    train_df,
    kg_adj,
    n_users: int,
    n_items: int,
):
    """Build (edge_index_ui, edge_index_ia, edge_index_full, deg, n_aspects).

    All three edge_indices share the same node id layout used by build_ckg():
        [0, n_users)                              → users
        [n_users, n_users + n_items)              → items
        [n_users+n_items, n_users+n_items+n_a)    → aspects
    """
    import numpy as np

    aspect_set = sorted({a for aspects in kg_adj.values() for a in aspects})
    aspect_to_idx = {a: i for i, a in enumerate(aspect_set)}
    n_aspects = len(aspect_set)
    n_total = n_users + n_items + n_aspects

    # ----- UI edges (both directions for undirected propagation) -----
    ui_u = train_df["user_idx"].to_numpy()
    ui_i = train_df["item_idx"].to_numpy() + n_users
    src_ui = np.concatenate([ui_u, ui_i])
    dst_ui = np.concatenate([ui_i, ui_u])
    edge_index_ui = torch.from_numpy(np.stack([src_ui, dst_ui])).long()

    # ----- IA edges -----
    ia_i_list, ia_a_list = [], []
    for item_idx, aspects in kg_adj.items():
        for asp in aspects:
            ia_i_list.append(item_idx + n_users)
            ia_a_list.append(aspect_to_idx[asp] + n_users + n_items)
    ia_i = np.asarray(ia_i_list, dtype=np.int64)
    ia_a = np.asarray(ia_a_list, dtype=np.int64)
    src_ia = np.concatenate([ia_i, ia_a])
    dst_ia = np.concatenate([ia_a, ia_i])
    edge_index_ia = torch.from_numpy(np.stack([src_ia, dst_ia])).long()

    # ----- Global view = UI ∪ IA -----
    src_full = np.concatenate([src_ui, src_ia])
    dst_full = np.concatenate([dst_ui, dst_ia])
    edge_index_full = torch.from_numpy(np.stack([src_full, dst_full])).long()

    # Degrees on the full graph (used to D^-1/2 normalise all three views)
    deg = np.zeros(n_total, dtype=np.float32)
    np.add.at(deg, src_full, 1.0)
    deg = torch.from_numpy(deg)

    return edge_index_ui, edge_index_ia, edge_index_full, deg, n_aspects
