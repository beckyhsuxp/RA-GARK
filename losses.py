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
