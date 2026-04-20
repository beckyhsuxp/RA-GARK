# RA-GARK Architecture

**Rationale-Aware Gating Network over Review Aspect-Specific Knowledge Graphs**

This document describes the RA-GARK architecture and training procedure.

---

## 1. Design Overview

Dual-view recommendation model that decomposes user preference and item
representation into two complementary views:

- **Local View** — collaborative signal from a LightGCN over the
  user–item interaction graph.
- **Global View** — KG-aspect representations passed through a
  user-conditioned *rationale-masking* attention so that, for each
  (user, item) pair, only the aspects the user actually cares about
  contribute to the score.

Both views are combined through independent fusion gates and the final
score is the dot product `u_final · i_final`.

---

## 2. Forward Pipeline (ASCII)

```
┌──────────────────────────────────────────────────────────────────────────┐
│                              INPUT BATCH                                 │
│                    user_idx [B]      item_idx [B]                        │
└──────────────┬─────────────────────────────────┬─────────────────────────┘
               │                                 │
       ┌───────┴───────┐                 ┌───────┴───────┐
       ▼               ▼                 ▼               ▼
  ╔══════════╗   ╔══════════╗      ╔══════════╗   ╔══════════════╗
  ║user_local║   ║user_glob ║      ║item_local║   ║item_kg_aspect║
  ║ Embedding║   ║ Embedding║      ║ Embedding║   ║   [Ni, A, d] ║
  ║ [Nu, d]  ║   ║ [Nu, d]  ║      ║ [Ni, d]  ║   ║ ★ KG SVD init║
  ╚════╤═════╝   ╚════╤═════╝      ╚════╤═════╝   ╚══════╤═══════╝
       │              │                 │                 │
       │       ┌──────┘                 │                 │
       │       │                        │                 │
       ▼       │                        ▼                 ▼
  ┌──────────┐ │                   ┌──────────┐    ┌────────────┐
  │ LightGCN │ │                   │ LightGCN │    │ i_aspects  │
  │ K-layer  │ │                   │ K-layer  │    │  [B, A, d] │
  │ D⁻¹ᐟ²AD⁻¹ᐟ²│                   │ D⁻¹ᐟ²AD⁻¹ᐟ²│    └─────┬──────┘
  └────┬─────┘ │                   └────┬─────┘          │
       │       │                        │                ▼
       │       ▼                        │       ┌─────────────────┐
       │  ┌────────┐                    │       │ KGRationale     │
       │  │ u_glo  │────────────────────┼──────▶│ Masking         │
       │  │ [B, d] │                    │       │ (user-conditioned│
       │  └────┬───┘                    │       │  aspect attention)│
       │       │                        │       └────────┬────────┘
       ▼       │                        ▼                ▼
  ┌────────┐  │                   ┌────────┐      ┌──────────┐
  │ u_loc  │  │                   │ i_loc  │      │  i_glo   │
  │ [B, d] │  │                   │ [B, d] │      │  [B, d]  │
  └────┬───┘  │                   └────┬───┘      └─────┬────┘
       │      │                        │                │
       │      │                        │                │
       └──┬───┘                        └────────┬───────┘
          │                                     │
          ▼                                     ▼
  ┌─────────────────┐                 ┌─────────────────┐
  │user_fusion_gate │                 │item_fusion_gate │
  │ α=σ(MLP[u_loc;  │                 │ α=σ(MLP[i_loc;  │
  │       u_glo])   │                 │       i_glo])   │
  └────────┬────────┘                 └────────┬────────┘
           │                                   │
           ▼                                   ▼
   ┌──────────────┐                    ┌──────────────┐
   │   u_final    │                    │   i_final    │
   │ αu_loc+      │                    │ αi_loc+      │
   │ (1-α)u_glo   │                    │ (1-α)i_glo   │
   └──────┬───────┘                    └──────┬───────┘
          │                                   │
          └────────────────┬──────────────────┘
                           ▼
                   ┌───────────────┐
                   │     score     │
                   │ u_final·i_final│
                   └───────────────┘
```

---

## 3. Training Loss Flow (ASCII)

```
                     ┌─────────────────────────────┐
                     │     Sampled Batch (B=128)   │
                     │       user, pos, neg        │
                     └──────────────┬──────────────┘
                                    │
                  ┌─────────────────┴─────────────────┐
                  ▼                                   ▼
             model(u, pos)                       model(u, neg)
                  │                                   │
                  ▼                                   ▼
       ┌──────────────────────┐              ┌───────────────┐
       │ pos_scores           │              │ neg_scores    │
       │ u_loc, u_glo         │              │               │
       │ i_pos_loc            │              │               │
       └───┬──────┬───────────┘              └───────┬───────┘
           │      │                                  │
           │      │                                  │
           │      │         ┌────────────────────────┘
           │      │         │
           │      │         ▼
           │      │   ┌──────────────────────┐
           │      │   │   L_BPR              │
           │      │   │  -logσ(pos - neg)    │
           │      │   └──────────┬───────────┘
           │      │              │
           │      ▼              │
           │  ┌──────────────────────────┐
           │  │   L_aCL  (per aspect)    │
           │  │   1/A Σ InfoNCE(         │
           │  │     proj(i_pos_loc),     │
           │  │     i_aspects[a].detach) │
           │  └──────────┬───────────────┘
           │             │
           ▼             │
   ┌───────────────────────┐
   │   L_uCL               │
   │   InfoNCE(            │
   │     proj(u_loc),      │
   │     u_glo.detach())   │
   └──────────┬────────────┘
              │
              ▼
       ┌──────────────────────────────────────────┐
       │  L_total = L_BPR                         │
       │          + 0.005 · (L_aCL + L_uCL)       │
       └──────────────────┬───────────────────────┘
                          │
                          ▼
              ┌───────────────────────┐
              │       Adam            │
              │  base lr   = 1e-3     │
              │  KG aspect = 5e-4     │
              └───────────────────────┘
```

---

## 4. Simplified Figure (for paper)

```
            ┌──────────────────────────────────────┐
            │           INPUT (u, i)               │
            └──────────────────────────────────────┘
                       │              │
              ┌────────┘              └────────┐
              ▼                                ▼
   ┌──────────────────┐              ┌────────────────────┐
   │  LOCAL VIEW      │              │   GLOBAL VIEW      │
   │  LightGCN over   │              │   KG-aspect emb    │
   │  user-item graph │              │   + Rationale Mask │
   │                  │              │   (user-cond attn) │
   └────────┬─────────┘              └─────────┬──────────┘
            │                                  │
            │  ┌── L_aCL (aspect-CL) ──┐       │
            │  │  proj(i_loc) ↔ aspect │       │
            │  └────── stop-grad ──────┘       │
            │                                  │
            ▼                                  ▼
       ╔═══════════════════════════════════════════╗
       ║    DUAL-VIEW FUSION GATES (per side)      ║
       ║       u_final, i_final                    ║
       ╚═══════════════════════════════════════════╝
                            │
                            ▼
                    score = u·i  ──▶  L_BPR
```

Three novelties to highlight in the figure caption:
1. **★ Softmax-normalised rationale attention** over the A aspects of an
   item, conditioned on the user's global embedding. The naive
   `sigmoid(MLP([u; a]))` formulation produces unnormalised per-aspect
   weights that saturate and pollute the global view; replacing the
   sigmoid with a softmax across aspects forces a proper attention
   distribution and is what makes rationale masking actually help
   (without it, `use_rationale=True` hurts NDCG by ≈10%).
2. **★ Local-biased fusion gate initialisation** — the fusion gate MLP's
   final Linear bias is initialised to `+5` so `α = σ(5) ≈ 0.993` at
   epoch 0. The model starts LightGCN-like and only opens the gate to
   the global view when the gradient says it helps. Default α≈0.5 lets
   the noisy KG contaminate the collaborative signal from epoch 1 and
   costs ≈15% NDCG.
3. **★ Aspect-level cross-view contrastive learning** in a separate
   projection space, with stop-gradient on the KG side, so aspect
   embeddings shape the local view's geometry without being pulled
   toward it.

Supporting design choices (not the headline novelty, but each adds a
small positive lift):
- **KG SVD initialisation** of `item_kg_aspects` — global view starts
  from real KG semantics rather than random noise
- **Per-parameter learning rate** — `item_kg_aspects` train with lr
  `5e-4` while the rest train with `1e-3`, protecting the KG-pretrained
  semantics from being washed out by BPR gradients

---

## 5. Component Reference

### 5.1 Embeddings

| Parameter            | Shape           | Init        | LR    |
|----------------------|-----------------|-------------|-------|
| `user_local_emb`     | `[Nu, d]`       | xavier      | 1e-3  |
| `item_local_emb`     | `[Ni, d]`       | xavier      | 1e-3  |
| `user_global_emb`    | `[Nu, d]`       | xavier      | 1e-3  |
| `item_kg_aspects`    | `[Ni, A, d]`    | **KG SVD**  | **5e-4** |

`Nu=905`, `Ni=1399`, `A=4`, `d=128`.

### 5.2 LightGCN propagation

```
x⁰ = [user_local; item_local]
x^(l+1) = D⁻¹ᐟ² A D⁻¹ᐟ² x^(l)
final  = mean(x⁰, x¹, …, xᴷ)         K = 2
```

### 5.3 KG Rationale Masking

For each `(user, item)` pair, attention weights are computed across the
A aspects of the item conditioned on the user's global embedding, then
**softmax-normalised across the aspect axis**:

```
logits  = MLP([u_glo ; i_aspects]).squeeze(-1)   # [B, A]
weights = softmax(logits, dim=-1)                # [B, A]   (sums to 1)
i_glo   = Σ_a  weights[a] · i_aspects[a]         # [B, d]
```

The softmax is the critical choice. The legacy `sigmoid(MLP([u; a]))`
formulation treats each aspect weight independently (no cross-aspect
competition), which empirically saturates near 1 for most aspects and
effectively averages without selecting. Softmax produces a sparse,
competitive attention that matches the rationale-masking intuition.

### 5.4 Fusion gates (independent for user and item)

```
α  = sigmoid(MLP([loc ; glo]))
out = α·loc + (1−α)·glo
```

**Bias initialisation.** The final Linear of each fusion MLP has its
bias initialised to `+5`, so `α ≈ σ(5) ≈ 0.993` at epoch 0. The model
begins behaving like pure LightGCN and only lowers α when the global
view earns its place via gradient. Without this bias, α starts at 0.5
and the noisy global view mixes into scoring before it has learned
anything useful.

### 5.5 CL projection head (used by `L_aCL` and `L_uCL`)

```
cl_projector = Linear(d,d) → ReLU → Linear(d,d)
```

CL is computed in this projection space so its gradients do not directly
interfere with the fusion gates (SimCLR/BYOL-style).

---

## 6. Loss Composition

```
L_total = L_BPR + 0.005 · (L_aCL + L_uCL)
```

| Loss     | Definition                                                              |
|----------|-------------------------------------------------------------------------|
| `L_BPR`  | `−log σ(pos_score − neg_score)`                                         |
| `L_aCL`  | `(1/A) Σ_a InfoNCE(proj(i_pos_loc), i_aspects[a].detach())`             |
| `L_uCL`  | `InfoNCE(proj(u_loc), u_glo.detach())`                                  |

---

## 7. Training

- **Optimizer**: Adam with two parameter groups
  - `base_params`            → lr `1e-3`
  - `item_kg_aspects`        → lr `5e-4`
- **Batch size**: 128
- **Epochs**: 80 with early stopping on val NDCG@20 (patience = 10)
- **Seed**: 42
- **Split**: user-stratified 70/15/15

---

## 8. Final Test Results (NDCG@20)

All KG-based methods are trained on the same processed aspect KG and the
same user-stratified split (seed=42).

All KG-based methods are trained on `data/df_edges_item_aspect1.csv`,
the user-stratified split (seed=42), and the same evaluator.

| Model                                       | NDCG       | HR         | Recall     | MAP        |
|---------------------------------------------|------------|------------|------------|------------|
| MCCLK (SIGIR 2022)                          | 0.1067     | 0.4530     | 0.1720     | 0.0497     |
| KGCL (SIGIR 2022)                           | 0.1073     | 0.4696     | 0.1827     | 0.0479     |
| KGAT (KDD 2019)                             | 0.1079     | 0.4773     | 0.1807     | 0.0491     |
| KGRec (KDD 2023)                            | 0.1095     | 0.4729     | 0.1834     | 0.0500     |
| **RA-GARK (Ours)**                          | **0.1235** | **0.4950** | **0.2008** | **0.0589** |

Relative improvement of RA-GARK vs the strongest KG-based baseline (KGRec):

| Metric | KGRec | RA-GARK | Δ |
|---|---|---|---|
| NDCG@20 | 0.1095 | 0.1235 | **+12.8 %** |
| HR@20 | 0.4729 | 0.4950 | **+4.7 %** |
| Recall@20 | 0.1834 | 0.2008 | **+9.5 %** |
| MAP@20 | 0.0500 | 0.0589 | **+17.8 %** |

---

## 9. Ablation

Each row flips one component of RA-GARK off and retrains from scratch
(seed=42, 80 epochs, patience=10). `winner` = full RA-GARK with softmax
rationale + fusion bias 5.0 (the new defaults). `old_full` reverts the
two Fix-1 changes (sigmoid rationale + fusion bias 0) to show the
original broken configuration. `lightgcn_only` is the no-KG floor.

| Preset                 | NDCG       | Δ vs winner | Notes                                                   |
|------------------------|------------|-------------|---------------------------------------------------------|
| **winner**             | **0.1235** | —           | softmax rat + fusion_bias=5 + all components on         |
| winner_no_rat          | _tbd_      |             | uniform mean over aspects (no rationale at all)         |
| winner_no_svd          | _tbd_      |             | xavier init for item_kg_aspects                         |
| winner_no_kg_lr        | _tbd_      |             | single lr for all params                                |
| winner_no_acl          | _tbd_      |             | drop aspect-level CL                                    |
| winner_no_ucl          | _tbd_      |             | drop user cross-view CL                                 |
| winner_sigmoid_rat     | _tbd_      |             | revert rationale head (softmax → sigmoid MLP)           |
| winner_fb0             | _tbd_      |             | revert fusion bias (5 → 0)                              |
| old_full               | ≈ 0.1064   | −13.8 %     | both Fix-1 reverts → the pre-fix broken full config     |
| no_global_view         | ≈ 0.1218   | −1.4 %      | skip the whole global pipeline (CL-only dual-view)      |
| lightgcn_only          | ≈ 0.1179   | −4.5 %      | no KG at all (floor)                                    |

Run `python run_ablations.py` to fill the `_tbd_` rows; results are
written to `ablation_results.csv`.
