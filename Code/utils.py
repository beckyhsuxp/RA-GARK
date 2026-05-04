from __future__ import annotations

import logging
import random
from typing import Tuple

import numpy as np
import pandas as pd
import torch

log = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, PyTorch (CPU + CUDA)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def user_stratified_split(
    df: pd.DataFrame,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Per-user stratified split into train/val/test.

    Each user's interactions are shuffled and split independently so every
    user appears in train (and, when n ≥ 3, in val and test as well).
    """
    rng = np.random.default_rng(seed)
    train_parts, val_parts, test_parts = [], [], []

    for _, group in df.groupby("user_idx", sort=False):
        n = len(group)
        shuffled = group.sample(frac=1, random_state=int(rng.integers(1 << 31)))

        if n < 3:
            train_parts.append(shuffled)
            continue

        n_test  = max(1, round(n * test_ratio))
        n_val   = max(1, round(n * val_ratio))
        n_train = n - n_val - n_test

        if n_train < 1:
            train_parts.append(shuffled.iloc[:1])
            test_parts.append(shuffled.iloc[1:])
            continue

        train_parts.append(shuffled.iloc[:n_train])
        val_parts.append(shuffled.iloc[n_train : n_train + n_val])
        test_parts.append(shuffled.iloc[n_train + n_val :])

    train_df = pd.concat(train_parts).reset_index(drop=True)
    val_df   = pd.concat(val_parts).reset_index(drop=True)  if val_parts  else pd.DataFrame(columns=df.columns)
    test_df  = pd.concat(test_parts).reset_index(drop=True) if test_parts else pd.DataFrame(columns=df.columns)

    log.info(
        "Split → train: %d  val: %d  test: %d  (eval users val/test: %d/%d)",
        len(train_df), len(val_df), len(test_df),
        val_df["user_idx"].nunique(), test_df["user_idx"].nunique(),
    )
    return train_df, val_df, test_df
