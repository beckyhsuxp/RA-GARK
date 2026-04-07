"""Plain LightGCN baseline.

Reference: He et al., "LightGCN: Simplifying and Powering Graph Convolution
Network for Recommendation", SIGIR 2020.

Exposes the same interface as RAKG_LMR (_lightgcn_embeddings, score_all_items)
so it can plug into the existing evaluate.evaluate() routine unchanged.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn


class LightGCN(nn.Module):
    def __init__(
        self,
        num_users: int,
        num_items: int,
        adj_matrix: torch.Tensor,
        dim: int = 64,
        n_layers: int = 2,
    ) -> None:
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.dim = dim
        self.n_layers = n_layers
        self.register_buffer("adj_matrix", adj_matrix)

        self.user_emb = nn.Embedding(num_users, dim)
        self.item_emb = nn.Embedding(num_items, dim)
        nn.init.xavier_normal_(self.user_emb.weight)
        nn.init.xavier_normal_(self.item_emb.weight)

    def _lightgcn_embeddings(self) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([self.user_emb.weight, self.item_emb.weight])
        layers = [x]
        for _ in range(self.n_layers):
            x = torch.sparse.mm(self.adj_matrix, x)
            layers.append(x)
        x = torch.stack(layers, dim=1).mean(dim=1)
        return x[: self.num_users], x[self.num_users :]

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
