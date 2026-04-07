from __future__ import annotations

from dataclasses import dataclass, field
from typing import Set


@dataclass
class Config:
    # --- Paths ---
    interaction_path: str = "data/reviews_30_20.pkl"
    kg_path: str = "data/df_edges_item_aspect1.csv"
    model_save_path: str = "best_ragark_model.pth"

    # --- Model ---
    embedding_dim: int = 128
    n_layers: int = 2
    num_aspects: int = 4

    # --- Training ---
    batch_size: int = 128
    learning_rate: float = 1e-3
    epochs: int = 30
    seed: int = 42
    early_stop_patience: int = 10   # 0 disables early stopping
    kg_aspect_lr: float = 5e-4      # lr for item_kg_aspects (v7 only) — sweep optimum

    # --- Loss weights ---
    cl_weight: float = 0.01   # InfoNCE contrastive loss
    reg_weight: float = 0.1   # KG consistency regularization
    temp: float = 0.2         # InfoNCE temperature

    # --- KG pruning ---
    kg_top_freq_pct: float = 0.02
    kg_stopwords: Set[str] = field(default_factory=lambda: {
        "good", "great", "quality", "price", "value", "shipping", "service",
        "character_development", "well_written", "engaging_storylines",
        "highly_recommended", "must_read", "interesting", "features",
        "depicts", "portrays", "complex_characters", "well_crafted_prose",
    })

    # --- Evaluation ---
    eval_k: int = 20
    eval_batch_size: int = 128
