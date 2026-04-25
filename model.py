from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class KGRationaleMasking(nn.Module):
    """User-conditioned attention over item KG-aspect embeddings.

    Supports three attention styles, switchable at init:
      * mlp_sigmoid — MLP([u; a]) → sigmoid, no cross-aspect normalisation
                      (original formulation; tends to saturate on all-ones)
      * mlp_softmax — MLP([u; a]) → softmax over aspects (weights sum to 1)
      * dot_softmax — (u · a) / √d → softmax over aspects (param-free head)

    forward() handles arbitrary leading dims in i_aspects so the same
    module can be called from both training (i_aspects: [B, A, d]) and
    full-item ranking (i_aspects: [B, Ni, A, d]).
    """

    def __init__(
        self,
        dim: int,
        style: str = "mlp_sigmoid",
        temperature: float = 1.0,
    ) -> None:
        super().__init__()
        self.style = style
        self.dim = dim
        self.scale = dim ** 0.5
        self.temperature = float(temperature)
        if style == "mlp_sigmoid":
            self.net = nn.Sequential(
                nn.Linear(dim * 2, dim),
                nn.LeakyReLU(),
                nn.Linear(dim, 1),
                nn.Sigmoid(),
            )
        elif style == "mlp_softmax":
            self.net = nn.Sequential(
                nn.Linear(dim * 2, dim),
                nn.LeakyReLU(),
                nn.Linear(dim, 1),
            )
        elif style == "dot_softmax":
            pass  # param-free
        else:
            raise ValueError(f"unknown rationale_style: {style}")

    def _weights(self, u_emb: torch.Tensor, i_aspects: torch.Tensor) -> torch.Tensor:
        # Broadcast u_emb to match i_aspects' leading dims; last dim = d
        u = u_emb
        while u.dim() < i_aspects.dim():
            u = u.unsqueeze(-2)
        u_exp = u.expand_as(i_aspects)

        if self.style == "mlp_sigmoid":
            cat = torch.cat([u_exp, i_aspects], dim=-1)
            return self.net(cat)                                   # [..., A, 1]
        if self.style == "mlp_softmax":
            cat = torch.cat([u_exp, i_aspects], dim=-1)
            logits = self.net(cat).squeeze(-1) / self.temperature  # [..., A]
            return F.softmax(logits, dim=-1).unsqueeze(-1)
        # dot_softmax
        scores = (u_exp * i_aspects).sum(dim=-1) / self.scale
        scores = scores / self.temperature                         # [..., A]
        return F.softmax(scores, dim=-1).unsqueeze(-1)

    def forward(self, u_emb: torch.Tensor, i_aspects: torch.Tensor) -> torch.Tensor:
        w = self._weights(u_emb, i_aspects)
        return (i_aspects * w).sum(dim=-2)


def _make_fusion_gate(dim: int, init_bias: float = 0.0) -> nn.Sequential:
    """Fusion gate MLP returning alpha in (0,1).

    init_bias is set on the FINAL Linear bias (pre-sigmoid). With
    init_bias=0 the gate starts at alpha≈0.5 (50/50 mix). With
    init_bias=5 it starts at alpha≈σ(5)≈0.993, biasing the model
    toward the local view until the global view earns its place.
    """
    linear_in = nn.Linear(dim * 2, dim)
    linear_out = nn.Linear(dim, 1)
    nn.init.xavier_uniform_(linear_in.weight)
    nn.init.zeros_(linear_in.bias)
    nn.init.xavier_uniform_(linear_out.weight)
    nn.init.constant_(linear_out.bias, init_bias)
    return nn.Sequential(linear_in, nn.Tanh(), linear_out, nn.Sigmoid())


class RA_GARK(nn.Module):
    """
    RA-GARK — Rationale-Aware Gating Network over Review Aspect-Specific Knowledge Graphs.

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
        use_rationale: bool = True,
        use_global_view: bool = True,
        rationale_style: str = "mlp_sigmoid",
        rationale_temperature: float = 1.0,
        fusion_init_bias: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.dim = dim
        self.n_layers = n_layers
        self.num_aspects = num_aspects
        self.use_rationale = use_rationale
        self.use_global_view = use_global_view
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

        self.rationale_masking = KGRationaleMasking(
            dim, style=rationale_style, temperature=rationale_temperature,
        )
        self.user_fusion_gate = _make_fusion_gate(dim, init_bias=fusion_init_bias)
        self.item_fusion_gate = _make_fusion_gate(dim, init_bias=fusion_init_bias)

        # Projection head for contrastive learning (used by v5+)
        self.cl_projector = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
        )

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
        cached_embs: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, ...]:
        if cached_embs is not None:
            all_u_loc, all_i_loc = cached_embs
        else:
            all_u_loc, all_i_loc = self._lightgcn_embeddings()
        u_loc = all_u_loc[u_idx]
        i_loc = all_i_loc[i_idx]

        u_glo = self.user_global_emb(u_idx)
        i_aspects = self.item_kg_aspects[i_idx]
        if self.use_rationale:
            i_glo = self.rationale_masking(u_glo, i_aspects)
        else:
            i_glo = i_aspects.mean(dim=1)

        if self.use_global_view:
            alpha_i = self.item_fusion_gate(torch.cat([i_loc, i_glo], dim=-1))
            i_final = alpha_i * i_loc + (1 - alpha_i) * i_glo

            alpha_u = self.user_fusion_gate(torch.cat([u_loc, u_glo], dim=-1))
            u_final = alpha_u * u_loc + (1 - alpha_u) * u_glo
        else:
            u_final = u_loc
            i_final = i_loc

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
        i_loc_exp = all_i_loc.unsqueeze(0).expand(B, -1, -1)

        if not self.use_global_view:
            return torch.bmm(i_loc_exp, u_loc.unsqueeze(-1)).squeeze(-1)

        u_glo = self.user_global_emb(u_idx)
        i_asp = self.item_kg_aspects.unsqueeze(0).expand(B, -1, -1, -1)
        if self.use_rationale:
            i_glo = self.rationale_masking(u_glo, i_asp)
        else:
            i_glo = i_asp.mean(dim=2)

        alpha_i = self.item_fusion_gate(torch.cat([i_loc_exp, i_glo], dim=-1))
        i_final = alpha_i * i_loc_exp + (1 - alpha_i) * i_glo

        alpha_u = self.user_fusion_gate(torch.cat([u_loc, u_glo], dim=-1))
        u_final = alpha_u * u_loc + (1 - alpha_u) * u_glo

        return torch.bmm(i_final, u_final.unsqueeze(-1)).squeeze(-1)
