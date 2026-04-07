"""KGRec baseline.

Reference: Yang et al., "Knowledge Graph Self-Supervised Rationalization
for Recommendation", KDD 2023.

Core ideas preserved here:
  1. Edge-level rationale scoring via attention.
  2. Rationale-aware edge dropout: edges with HIGHER attention are dropped
     with HIGHER probability, forcing the model not to over-rely on a few
     dominant edges.
  3. Cross-view contrastive learning: representations from the original
     graph are aligned with those from the rationale-dropped graph,
     producing more robust user/item embeddings.

Differences from the official paper version:
  - Single relation type in our KG → no relation embeddings.
  - Bi-interaction aggregator borrowed from KGAT (sufficient for one relation).
  - Hard Bernoulli dropout instead of Gumbel-softmax (still trainable since
    gradients flow through retained edges).

Reuses build_ckg() from kgat.py to construct the same heterogeneous graph
used by the KGAT baseline so the comparison is fair.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class KGRec(nn.Module):
    def __init__(
        self,
        n_users: int,
        n_items: int,
        n_aspects: int,
        edge_index: torch.Tensor,
        deg: torch.Tensor,
        dim: int = 64,
        n_layers: int = 2,
        drop_prob: float = 0.3,
    ) -> None:
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.n_aspects = n_aspects
        self.n_total = n_users + n_items + n_aspects
        self.dim = dim
        self.n_layers = n_layers
        self.drop_prob = drop_prob

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

    def _rationale_scores(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Per-edge attention used both for propagation and for rationale dropout."""
        src, dst = edge_index[0], edge_index[1]
        return (x[src] * x[dst]).sum(dim=-1) / (self.dim ** 0.5)

    # ------------------------------------------------------------------ #
    # Propagation
    # ------------------------------------------------------------------ #

    def _propagate(self, edge_index: torch.Tensor) -> torch.Tensor:
        """KGAT-style attentive bi-interaction propagation on a given edge_index."""
        x = self.entity_emb.weight
        outs = [x]

        for l in range(self.n_layers):
            src, dst = edge_index[0], edge_index[1]
            scores = self._rationale_scores(x, edge_index)
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

    def _rationale_dropout(self) -> torch.Tensor:
        """Drop edges with probability proportional to (normalised) attention score.
        Returns the dropped edge_index."""
        with torch.no_grad():
            x = self.entity_emb.weight
            scores = self._rationale_scores(x, self.edge_index)
            s_min, s_max = scores.min(), scores.max()
            norm_scores = (scores - s_min) / (s_max - s_min + 1e-8)
            drop_p = self.drop_prob * norm_scores
            keep_mask = torch.bernoulli(1.0 - drop_p).bool()
        return self.edge_index[:, keep_mask]

    def propagate_two_views(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Original graph view + rationale-dropped view (for CL)."""
        emb_full = self._propagate(self.edge_index)
        edge_drop = self._rationale_dropout()
        emb_drop = self._propagate(edge_drop)
        return emb_full, emb_drop

    # ------------------------------------------------------------------ #
    # Inference / evaluation interface (matches RA_GARK)
    # ------------------------------------------------------------------ #

    def _lightgcn_embeddings(self) -> Tuple[torch.Tensor, torch.Tensor]:
        emb = self._propagate(self.edge_index)
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
