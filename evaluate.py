from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Set

import numpy as np
import torch

from model import RAKG_LMR


def _metrics_at_k(
    ranked_list: np.ndarray, ground_truth: List[int], k: int
) -> Dict[str, float]:
    gt = set(ground_truth)
    hits = [1 if item in gt else 0 for item in ranked_list]
    n_hits = sum(hits)

    hr = float(n_hits > 0)
    precision = n_hits / k
    recall = n_hits / len(gt) if gt else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    ap, cum = 0.0, 0
    for i, h in enumerate(hits):
        if h:
            cum += 1
            ap += cum / (i + 1)
    map_score = ap / min(len(gt), k) if gt else 0.0

    dcg = sum(1.0 / np.log2(i + 2) for i, h in enumerate(hits) if h)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(gt), k)))
    ndcg = dcg / idcg if idcg > 0 else 0.0

    return dict(HR=hr, Precision=precision, Recall=recall, F1=f1, MAP=map_score, NDCG=ndcg)


def evaluate(
    model: RAKG_LMR,
    test_ground_truth: Dict[int, List[int]],
    train_history: Dict[int, Set[int]],
    device: torch.device,
    k: int = 20,
    batch_size: int = 128,
) -> Dict[str, float]:
    """
    Vectorised full-ranking evaluation.

    _lightgcn_embeddings() is called once before the user loop and the result
    is passed to every score_all_items() call, eliminating (n_batches - 1)
    redundant full-graph propagations per evaluation pass.
    """
    model.eval()
    agg: Dict[str, List[float]] = defaultdict(list)
    users = list(test_ground_truth.keys())

    with torch.no_grad():
        cached_embs = model._lightgcn_embeddings()

        for start in range(0, len(users), batch_size):
            batch = users[start : start + batch_size]
            u_tensor = torch.LongTensor(batch).to(device)
            scores = model.score_all_items(u_tensor, cached_embs=cached_embs).cpu().numpy()

            for i, uid in enumerate(batch):
                seen = train_history.get(uid, set())
                if seen:
                    scores[i, list(seen)] = -np.inf
                k_eff = min(k, scores.shape[1])
                top_k = np.argpartition(scores[i], -k_eff)[-k_eff:]
                top_k = top_k[np.argsort(scores[i, top_k])[::-1]]
                for metric, val in _metrics_at_k(top_k, test_ground_truth[uid], k).items():
                    agg[metric].append(val)

    return {m: float(np.mean(v)) for m, v in agg.items()}
