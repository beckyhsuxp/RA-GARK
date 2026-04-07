# RAKG-LMR Architecture

**Rationale-Aware Gating Network over Review Aspect-Specific Knowledge Graphs**

This document describes the v7 (final) architecture and training procedure.

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
                     │  user, pos, neg, kg_nbr     │
                     └──────────────┬──────────────┘
                                    │
            ┌───────────────────────┼───────────────────────┐
            ▼                       ▼                       ▼
       model(u,pos)           model(u,neg)           model(u,nbr)
            │                       │                       │
            ▼                       ▼                       ▼
   ┌────────────────┐      ┌────────────────┐      ┌────────────────┐
   │ pos_scores     │      │ neg_scores     │      │ i_nbr_glo      │
   │ u_loc, u_glo   │      │                │      │                │
   │ i_pos_loc      │      │                │      │                │
   │ i_pos_glo      │      │                │      │                │
   └───┬──────┬──┬──┘      └────────┬───────┘      └───────┬────────┘
       │      │  │                  │                      │
       │      │  └──┐               │                      │
       │      │     │               │                      │
       │      │     ▼               ▼                      │
       │      │  ┌──────────────────────┐                  │
       │      │  │   L_BPR              │                  │
       │      │  │ -logσ(pos - neg)     │                  │
       │      │  └──────────┬───────────┘                  │
       │      │             │                              │
       │      ▼             │                              │
       │  ┌─────────────────────────┐                      │
       │  │   L_aCL  (per aspect)   │                      │
       │  │   1/A Σ InfoNCE(        │                      │
       │  │     proj(i_pos_loc),    │                      │
       │  │     i_aspects[a].detach)│                      │
       │  └──────────┬──────────────┘                      │
       │             │                                     │
       ▼             │                                     │
   ┌───────────────────────┐                               │
   │   L_uCL               │                               │
   │   InfoNCE(            │                               │
   │     proj(u_loc),      │                               │
   │     u_glo.detach())   │                               │
   └──────────┬────────────┘                               │
              │                                            │
              │                            ┌───────────────┘
              │                            │
              │             ┌──────────────────────┐
              │             │   L_reg              │
              │             │   MSE(i_pos_glo,     │
              │             │       i_nbr_glo)     │
              │             │   ≈ 0  (no-op)       │
              │             └──────────┬───────────┘
              │                        │
              ▼                        ▼
       ┌──────────────────────────────────────────┐
       │  L_total = L_BPR                         │
       │          + 0.005 · (L_aCL + L_uCL)       │
       │          + 0.973 · L_reg                 │
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
1. **★ KG SVD initialisation** of `item_kg_aspects` (so the global view
   starts from real KG semantics rather than random noise)
2. **★ Aspect-level cross-view contrastive learning** in a separate
   projection space, with stop-gradient on the KG side
3. **★ Per-parameter learning rate** that protects the KG-pretrained
   aspect embeddings from being washed out by BPR gradients

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
4 aspects of the item conditioned on the user's global embedding:

```
weights = sigmoid(MLP([u_glo ; i_aspects]))     # [B, A, 1]
i_glo   = Σ_a  weights[a] · i_aspects[a]        # [B, d]
```

### 5.4 Fusion gates (independent for user and item)

```
α  = sigmoid(MLP([loc ; glo]))
out = α·loc + (1−α)·glo
```

### 5.5 CL projection head (used by `L_aCL` and `L_uCL`)

```
cl_projector = Linear(d,d) → ReLU → Linear(d,d)
```

CL is computed in this projection space so its gradients do not directly
interfere with the fusion gates (SimCLR/BYOL-style).

---

## 6. Loss Composition

```
L_total = L_BPR
        + 0.005 · (L_aCL + L_uCL)
        + 0.973 · L_reg                # ≈ 0 in practice
```

| Loss     | Definition                                                              |
|----------|-------------------------------------------------------------------------|
| `L_BPR`  | `−log σ(pos_score − neg_score)`                                         |
| `L_aCL`  | `(1/A) Σ_a InfoNCE(proj(i_pos_loc), i_aspects[a].detach())`             |
| `L_uCL`  | `InfoNCE(proj(u_loc), u_glo.detach())`                                  |
| `L_reg`  | `MSE(i_pos_glo, i_nbr_glo)` — kept for completeness, collapses to ~0    |

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

| Model                                       | NDCG   | HR     | Recall | MAP    |
|---------------------------------------------|--------|--------|--------|--------|
| KGAT                                        | 0.1061 | 0.4707 | 0.1760 | 0.0482 |
| KGRec                                       | TBD    | TBD    | TBD    | TBD    |
| v3 (RAKG-LMR original)                      | 0.1155 | 0.4950 | 0.1908 | 0.0539 |
| v6 (+ improved CL)                          | 0.1182 | 0.4950 | 0.1977 | 0.0547 |
| **v7 (+ KG SVD init + per-param lr)**       | **0.1196** | 0.4884 | 0.1954 | 0.0562 |
