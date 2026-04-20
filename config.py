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

    # --- Ablation flags (all default True = full model) ---
    use_rationale: bool = True     # False → uniform mean over aspects
    use_svd_init: bool = True      # False → xavier init for item_kg_aspects
    use_kg_lr: bool = True         # False → single lr for all params
    use_acl: bool = True           # False → drop aspect-level CL
    use_ucl: bool = True           # False → drop user cross-view CL
    use_global_view: bool = True   # False → skip global pipeline (pure LightGCN)

    # --- Rationale masking variants (when use_rationale=True) ---
    # mlp_sigmoid: current — MLP([u; a]) → sigmoid, no cross-aspect normalisation
    # mlp_softmax: MLP([u; a]) → softmax over aspects (weights sum to 1)
    # dot_softmax: (u · a) / √d → softmax over aspects (no extra params)
    rationale_style: str = "mlp_sigmoid"

    # --- Fusion gate init bias ---
    # Final Linear bias in the fusion gate MLP. 0.0 → alpha ≈ 0.5 at start
    # (50/50 local/global mix from epoch 1, noisy KG pollutes LightGCN).
    # 5.0 → alpha ≈ σ(5) ≈ 0.993 at start → model behaves like LightGCN
    # initially; gate only opens up to the global view when it helps.
    fusion_init_bias: float = 0.0

    # --- Training ---
    batch_size: int = 128
    learning_rate: float = 1e-3
    epochs: int = 30
    seed: int = 42
    early_stop_patience: int = 10   # 0 disables early stopping
    kg_aspect_lr: float = 5e-4      # lr for item_kg_aspects (v7 only) — sweep optimum

    # --- Loss weights ---
    cl_weight: float = 0.01   # InfoNCE contrastive loss
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
