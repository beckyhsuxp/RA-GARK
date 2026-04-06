from __future__ import annotations

import torch
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


def kg_triplet_loss(
    anchor: torch.Tensor,
    positive: torch.Tensor,
    negative: torch.Tensor,
    margin: float = 1.0,
) -> torch.Tensor:
    """Margin-based triplet loss for KG regularization.

    Pushes KG-neighbor embedding closer to the anchor (positive item)
    than a random negative item, with a margin.
    """
    d_pos = (anchor - positive).pow(2).sum(dim=-1)
    d_neg = (anchor - negative).pow(2).sum(dim=-1)
    return F.relu(d_pos - d_neg + margin).mean()
