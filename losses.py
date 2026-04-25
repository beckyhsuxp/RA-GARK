from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def bpr_loss(pos_scores: torch.Tensor, neg_scores: torch.Tensor) -> torch.Tensor:
    return -F.logsigmoid(pos_scores - neg_scores).mean()


def infonce_loss(
    view1: torch.Tensor, view2: torch.Tensor, temp: float = 0.2
) -> torch.Tensor:
    v1 = F.normalize(view1, dim=-1)
    v2 = F.normalize(view2, dim=-1)
    logits = torch.matmul(v1, v2.T) / temp
    labels = torch.arange(len(v1), device=v1.device)
    return F.cross_entropy(logits, labels)


def aspect_level_cl(
    projector: nn.Module,
    i_loc: torch.Tensor,
    i_aspects: torch.Tensor,
    temp: float,
) -> torch.Tensor:
    """Aspect-level cross-view contrastive loss (L_aCL).

    For each aspect a ∈ {1,…,A}, contrast the projected local item
    embedding against that aspect's KG embedding (stop-grad on KG side).

    Args:
        projector:  projection head [d → d]
        i_loc:      local item embeddings        [B, d]
        i_aspects:  per-aspect KG embeddings     [B, A, d]
        temp:       InfoNCE temperature
    Returns:
        scalar loss averaged over aspects.
    """
    num_aspects = i_aspects.size(1)
    proj_loc = projector(i_loc)
    loss = torch.zeros((), device=i_loc.device)
    for a in range(num_aspects):
        aspect_emb = i_aspects[:, a, :].detach()
        loss = loss + infonce_loss(proj_loc, aspect_emb, temp)
    return loss / num_aspects
