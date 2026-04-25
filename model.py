from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class KGRationaleMasking(nn.Module):
    """Bilateral aspect-aligned rationale masking.

    Both user and item carry A aspect slots ([..., A, d]). For each aspect a,
    a per-aspect alignment score is computed; softmax over aspects yields a
    weight vector that is applied to BOTH sides to aggregate u_glo and i_glo.
    The same weights drive both aggregations, so the two views speak the
    same rationale "language" by construction.

    Two style variants:
      * dot — score[a] = (u_aspects[a] · i_aspects[a]) / √d   (param-free)
      * mlp — score[a] = MLP([u_aspects[a]; i_aspects[a]])     (extra params)

    forward() handles arbitrary leading dims so the same module is used
    from training (u/i shape [B, A, d]) and full-item ranking
    (u shape [B, A, d] expanded vs i shape [B, Ni, A, d]).
    """

    def __init__(
        self,
        dim: int,
        style: str = "dot",
        temperature: float = 1.0,
    ) -> None:
        super().__init__()
        self.style = style
        self.dim = dim
        self.scale = dim ** 0.5
        self.temperature = float(temperature)
        if style == "mlp":
            self.net = nn.Sequential(
                nn.Linear(dim * 2, dim),
                nn.LeakyReLU(),
                nn.Linear(dim, 1),
            )
        elif style != "dot":
            raise ValueError(f"unknown rationale_style: {style}")

    def _scores(
        self, u_aspects: torch.Tensor, i_aspects: torch.Tensor
    ) -> torch.Tensor:
        """Return per-aspect score, shape [..., A]."""
        if self.style == "mlp":
            cat = torch.cat([u_aspects, i_aspects], dim=-1)   # [..., A, 2d]
            return self.net(cat).squeeze(-1)                  # [..., A]
        # dot
        return (u_aspects * i_aspects).sum(dim=-1) / self.scale  # [..., A]

    def _weights(
        self, u_aspects: torch.Tensor, i_aspects: torch.Tensor
    ) -> torch.Tensor:
        """Return softmax-normalised aspect weights, shape [..., A].

        Broadcasts u_aspects up to i_aspects' leading dims if needed (the
        full-ranking call passes u_aspects as [B, A, d] and i_aspects as
        [B, Ni, A, d]).
        """
        u = u_aspects
        while u.dim() < i_aspects.dim():
            u = u.unsqueeze(-3)               # insert dim before (A, d)
        u_exp = u.expand_as(i_aspects)
        scores = self._scores(u_exp, i_aspects) / self.temperature
        return F.softmax(scores, dim=-1)      # [..., A]

    def forward(
        self, u_aspects: torch.Tensor, i_aspects: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        w = self._weights(u_aspects, i_aspects).unsqueeze(-1)   # [..., A, 1]
        u = u_aspects
        while u.dim() < i_aspects.dim():
            u = u.unsqueeze(-3)
        u_exp = u.expand_as(i_aspects)
        u_glo = (u_exp * w).sum(dim=-2)        # [..., d]
        i_glo = (i_aspects * w).sum(dim=-2)    # [..., d]
        return u_glo, i_glo


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


class _ScalarGate(nn.Module):
    """Single learnable alpha; ignores input.

    Sanity-check stand-in for `_make_fusion_gate` to test whether the
    MLP gate's per-(user, item) conditioning is actually doing work,
    or whether a single global alpha would yield the same NDCG.
    """

    def __init__(self, init_bias: float = 0.0) -> None:
        super().__init__()
        self.alpha_logit = nn.Parameter(torch.tensor(float(init_bias)))

    def forward(self, _x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.alpha_logit)


class RA_GARK(nn.Module):
    """RA-GARK — Rationale-Aware Gating Network over Review Aspect-Specific KG.

    Bilateral global view: both `user_kg_aspects [Nu, A, d]` and
    `item_kg_aspects [Ni, A, d]` carry A aspect slots, and `KGRationaleMasking`
    produces a single per-(u, i) attention vector over aspects that aggregates
    both sides into u_glo / i_glo.

    forward() returns (scores, u_loc, u_glo, i_loc, i_glo) for loss computation.
    score_all_items() does vectorised full-ranking for evaluation; both accept
    cached LightGCN embeddings so the propagation runs once per batch / eval.
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
        rationale_style: str = "dot",
        rationale_temperature: float = 1.0,
        fusion_init_bias: float = 0.0,
        fusion_gate_style: str = "mlp",
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

        # Global view (bilateral KG aspects)
        self.user_kg_aspects = nn.Parameter(torch.empty(num_users, num_aspects, dim))
        self.item_kg_aspects = nn.Parameter(torch.empty(num_items, num_aspects, dim))
        nn.init.xavier_normal_(self.user_kg_aspects)
        nn.init.xavier_normal_(self.item_kg_aspects)

        self.rationale_masking = KGRationaleMasking(
            dim, style=rationale_style, temperature=rationale_temperature,
        )
        if fusion_gate_style == "scalar":
            self.user_fusion_gate = _ScalarGate(init_bias=fusion_init_bias)
            self.item_fusion_gate = _ScalarGate(init_bias=fusion_init_bias)
        elif fusion_gate_style == "mlp":
            self.user_fusion_gate = _make_fusion_gate(dim, init_bias=fusion_init_bias)
            self.item_fusion_gate = _make_fusion_gate(dim, init_bias=fusion_init_bias)
        else:
            raise ValueError(f"unknown fusion_gate_style: {fusion_gate_style}")

        # Projection head for contrastive learning
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

        u_aspects = self.user_kg_aspects[u_idx]   # [B, A, d]
        i_aspects = self.item_kg_aspects[i_idx]   # [B, A, d]
        if self.use_rationale:
            u_glo, i_glo = self.rationale_masking(u_aspects, i_aspects)
        else:
            u_glo = u_aspects.mean(dim=1)
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
        """Full-ranking scores for a batch of users.

        Returns: [B, N_items]
        """
        if cached_embs is not None:
            all_u_loc, all_i_loc = cached_embs
        else:
            all_u_loc, all_i_loc = self._lightgcn_embeddings()

        B = u_idx.size(0)
        u_loc = all_u_loc[u_idx]                                # [B, d]
        i_loc_exp = all_i_loc.unsqueeze(0).expand(B, -1, -1)    # [B, Ni, d]

        if not self.use_global_view:
            return torch.bmm(i_loc_exp, u_loc.unsqueeze(-1)).squeeze(-1)

        # Bilateral aspect blocks broadcast to [B, Ni, A, d].
        u_asp = self.user_kg_aspects[u_idx]                                  # [B, A, d]
        u_asp_exp = u_asp.unsqueeze(1).expand(-1, self.num_items, -1, -1)    # [B, Ni, A, d]
        i_asp_exp = self.item_kg_aspects.unsqueeze(0).expand(B, -1, -1, -1)  # [B, Ni, A, d]

        if self.use_rationale:
            u_glo, i_glo = self.rationale_masking(u_asp_exp, i_asp_exp)
        else:
            u_glo = u_asp_exp.mean(dim=2)
            i_glo = i_asp_exp.mean(dim=2)

        # Each user has one u_loc but Ni different u_glo (bilateral attention),
        # so broadcast u_loc to [B, Ni, d] for the gate input.
        u_loc_exp = u_loc.unsqueeze(1).expand(-1, self.num_items, -1)        # [B, Ni, d]

        alpha_i = self.item_fusion_gate(torch.cat([i_loc_exp, i_glo], dim=-1))
        i_final = alpha_i * i_loc_exp + (1 - alpha_i) * i_glo

        alpha_u = self.user_fusion_gate(torch.cat([u_loc_exp, u_glo], dim=-1))
        u_final = alpha_u * u_loc_exp + (1 - alpha_u) * u_glo

        return (u_final * i_final).sum(dim=-1)                                # [B, Ni]
