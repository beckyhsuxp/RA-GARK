"""KGCL baseline.

Reference: Yang et al., "Knowledge Graph Contrastive Learning for
Recommendation", SIGIR 2022.

Core ideas preserved here:
  1. Two augmented views of the collaborative knowledge graph created
     by independent random edge dropouts (uniform probability).
  2. Cross-view InfoNCE contrastive learning between user / item
     representations from the two augmented views.
  3. Joint optimisation with BPR over the original (full) graph.

Differences from the official paper version:
  - Single relation type → no relation embeddings.
  - Bi-interaction aggregator borrowed from KGAT (sufficient here).
  - The KG-side noise generation in the paper is approximated by random
    edge dropout on the unified CKG (same protocol as KGRec, only the
    dropout policy differs: uniform vs rationale-weighted).

Reuses build_ckg() from kgat.py for fair comparison.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class KGCL(nn.Module):
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

    # ------------------------------------------------------------------ #
    # Propagation
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

    def _random_edge_dropout(self) -> torch.Tensor:
        """Drop each edge independently with uniform probability `drop_prob`."""
        n_edges = self.edge_index.size(1)
        keep_mask = torch.bernoulli(
            torch.full((n_edges,), 1.0 - self.drop_prob, device=self.edge_index.device)
        ).bool()
        return self.edge_index[:, keep_mask]

    def propagate_two_augmented_views(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Two independent random-dropout views (KGCL contrastive pair)."""
        e1 = self._random_edge_dropout()
        e2 = self._random_edge_dropout()
        return self._propagate(e1), self._propagate(e2)

    def propagate_full(self) -> torch.Tensor:
        """Full graph propagation (used for BPR and evaluation)."""
        return self._propagate(self.edge_index)

    # ------------------------------------------------------------------ #
    # Inference / evaluation interface (matches RA_GARK)
    # ------------------------------------------------------------------ #

    def _lightgcn_embeddings(self) -> Tuple[torch.Tensor, torch.Tensor]:
        emb = self.propagate_full()
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
