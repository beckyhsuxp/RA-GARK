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
    use_acl: bool = True           # False → drop aspect-level CL
    use_ucl: bool = True           # False → drop user cross-view CL
    use_global_view: bool = True   # False → skip global pipeline (pure LightGCN)

    # --- Rationale masking variants (when use_rationale=True) ---
    # mlp_sigmoid: legacy — MLP([u; a]) → sigmoid, no cross-aspect normalisation
    # mlp_softmax: MLP([u; a]) → softmax over aspects (weights sum to 1)  ← default
    # dot_softmax: (u · a) / √d → softmax over aspects (param-free head)
    rationale_style: str = "mlp_softmax"

    # --- Softmax temperature (only applies to softmax-style rationale) ---
    # weights = softmax(logits / τ). 1.0 ≈ uniform; 0.5 sharpens (best NDCG).
    rationale_temperature: float = 0.5

    # --- Fusion gate init bias ---
    # Final Linear bias in the fusion gate MLP. 0.0 → α ≈ 0.5 at start
    # (50/50 local/global mix; noisy KG pollutes LightGCN). 5.0 → α ≈ 0.993
    # at start → behaves like LightGCN until the gate earns its way open.
    fusion_init_bias: float = 5.0

    # --- Training ---
    batch_size: int = 128
    learning_rate: float = 1e-3
    epochs: int = 30
    seed: int = 42
    early_stop_patience: int = 10   # 0 disables early stopping

    # --- Loss weights ---
    cl_weight: float = 0.01   # InfoNCE contrastive loss (shared default)
    # Optional per-loss override. None → falls back to cl_weight, so the
    # default behaviour is unchanged (acl_w = ucl_w = cl_weight).
    acl_weight: float | None = None
    ucl_weight: float | None = None
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
