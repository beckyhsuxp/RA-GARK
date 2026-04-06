from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn


class KGRationaleMasking(nn.Module):
    """User-conditioned attention over item KG-aspect embeddings."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.LeakyReLU(),
            nn.Linear(dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, u_emb: torch.Tensor, i_aspects: torch.Tensor) -> torch.Tensor:
        u_exp = u_emb.unsqueeze(1).expand_as(i_aspects)
        cat = torch.cat([u_exp, i_aspects], dim=-1)
        weights = self.net(cat)
        return (i_aspects * weights).sum(dim=1)


def _make_fusion_gate(dim: int) -> nn.Sequential:
    gate = nn.Sequential(
        nn.Linear(dim * 2, dim),
        nn.Tanh(),
        nn.Linear(dim, 1),
        nn.Sigmoid(),
    )
    for layer in gate:
        if isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
    return gate


class RAKG_LMR(nn.Module):
    """
    Rationale-Aware KG Recommender with Local-Multi-view Representation.

    Fixes applied:
      #3  Separate user_fusion_gate and item_fusion_gate.
      #4  score_all_items() accepts optional cached LightGCN embeddings so
          evaluate() can call _lightgcn_embeddings() once per eval pass
          instead of once per user batch.

    forward() returns (scores, u_loc, u_glo, i_loc, i_glo) for loss computation.
    score_all_items() does vectorised full-ranking for evaluation.
    """

    def __init__(
        self,
        num_users: int,
        num_items: int,
        adj_matrix: torch.Tensor,
        num_aspects: int = 4,
        dim: int = 64,
        n_layers: int = 2,
    ) -> None:
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.dim = dim
        self.n_layers = n_layers
        self.num_aspects = num_aspects
        self.register_buffer("adj_matrix", adj_matrix)

        # Local view (LightGCN)
        self.user_local_emb = nn.Embedding(num_users, dim)
        self.item_local_emb = nn.Embedding(num_items, dim)
        nn.init.xavier_normal_(self.user_local_emb.weight)
        nn.init.xavier_normal_(self.item_local_emb.weight)

        # Global view (KG aspects)
        self.user_global_emb = nn.Embedding(num_users, dim)
        nn.init.xavier_normal_(self.user_global_emb.weight)
        self.item_kg_aspects = nn.Parameter(torch.empty(num_items, num_aspects, dim))
        nn.init.xavier_normal_(self.item_kg_aspects)

        self.rationale_masking = KGRationaleMasking(dim)
        self.user_fusion_gate = _make_fusion_gate(dim)
        self.item_fusion_gate = _make_fusion_gate(dim)

    def _lightgcn_embeddings(self) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([self.user_local_emb.weight, self.item_local_emb.weight])
        layers = [x]
        for _ in range(self.n_layers):
            x = torch.sparse.mm(self.adj_matrix, x)
            layers.append(x)
        x = torch.stack(layers, dim=1).mean(dim=1)
        return x[: self.num_users], x[self.num_users :]

    def forward(
        self,
        u_idx: torch.Tensor,
        i_idx: torch.Tensor,
    ) -> Tuple[torch.Tensor, ...]:
        all_u_loc, all_i_loc = self._lightgcn_embeddings()
        u_loc = all_u_loc[u_idx]
        i_loc = all_i_loc[i_idx]

        u_glo = self.user_global_emb(u_idx)
        i_aspects = self.item_kg_aspects[i_idx]
        i_glo = self.rationale_masking(u_glo, i_aspects)

        alpha_i = self.item_fusion_gate(torch.cat([i_loc, i_glo], dim=-1))
        i_final = alpha_i * i_loc + (1 - alpha_i) * i_glo

        alpha_u = self.user_fusion_gate(torch.cat([u_loc, u_glo], dim=-1))
        u_final = alpha_u * u_loc + (1 - alpha_u) * u_glo

        scores = (u_final * i_final).sum(dim=-1)
        return scores, u_loc, u_glo, i_loc, i_glo

    @torch.no_grad()
    def score_all_items(
        self,
        u_idx: torch.Tensor,
        cached_embs: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Args:
            u_idx:        [B]
            cached_embs:  pre-computed (all_u_loc, all_i_loc) from
                          _lightgcn_embeddings(); computed here if None.
        Returns:
            scores: [B, N_items]
        """
        if cached_embs is not None:
            all_u_loc, all_i_loc = cached_embs
        else:
            all_u_loc, all_i_loc = self._lightgcn_embeddings()

        B = u_idx.size(0)
        u_loc = all_u_loc[u_idx]
        u_glo = self.user_global_emb(u_idx)

        i_asp = self.item_kg_aspects.unsqueeze(0).expand(B, -1, -1, -1)
        u_exp = u_glo[:, None, None, :].expand_as(i_asp)
        cat = torch.cat([u_exp, i_asp], dim=-1)
        weights = self.rationale_masking.net(cat)
        i_glo = (i_asp * weights).sum(dim=2)

        i_loc_exp = all_i_loc.unsqueeze(0).expand(B, -1, -1)

        alpha_i = self.item_fusion_gate(torch.cat([i_loc_exp, i_glo], dim=-1))
        i_final = alpha_i * i_loc_exp + (1 - alpha_i) * i_glo

        alpha_u = self.user_fusion_gate(torch.cat([u_loc, u_glo], dim=-1))
        u_final = alpha_u * u_loc + (1 - alpha_u) * u_glo

        return torch.bmm(i_final, u_final.unsqueeze(-1)).squeeze(-1)
