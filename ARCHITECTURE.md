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

Three novelties to highlight in the figure caption (all % are NDCG@20
leave-one-out drops measured in Section 9):

1. **★ Softmax-normalised rationale attention** (−6.4% when reverted
   to sigmoid) — over the A aspects of an item, conditioned on the
   user's global embedding. The naive `sigmoid(MLP([u; a]))`
   formulation produces unnormalised per-aspect weights that saturate,
   and is not merely neutral but **actively harmful**: sigmoid
   rationale (0.1152 NDCG) does *worse* than disabling rationale
   entirely and using a uniform aspect mean (0.1222 NDCG). Replacing
   the sigmoid with a softmax across aspects turns the rationale
   module into a net-positive contribution (winner 0.1231 NDCG).
2. **★ Local-biased fusion gate initialisation** (−4.7% when reverted
   to bias 0) — the fusion gate MLP's final Linear bias is initialised
   to `+5` so `α = σ(5) ≈ 0.993` at epoch 0. The model starts
   LightGCN-like and only opens the gate to the global view when the
   gradient says it helps. Default α≈0.5 lets the noisy KG contaminate
   the collaborative signal from epoch 1.
3. **★ KG SVD initialisation of `item_kg_aspects`** (−4.7% when
   reverted to xavier) — the global view starts from real KG semantics
   rather than random noise. Combined with the per-parameter LR on
   `item_kg_aspects` (kept, but negligible on its own: −0.1%), this
   keeps the KG-pretrained aspect geometry intact through BPR
   training.

Supporting design choices:
- **Aspect-level cross-view CL** (−1.9%) — `L_aCL`, aligns local item
  embedding with each aspect in a projection space, stop-grad on KG
  side.
- **User cross-view CL** (−3.6%) — `L_uCL`, pulls `u_loc` toward
  `u_glo`.
- **Per-parameter LR** (−0.1%) — kept for completeness, but does not
  carry a headline claim.

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
| **RA-GARK (Ours)**                          | **0.1231** | **0.4961** | **0.2003** | **0.0586** |

Relative improvement of RA-GARK vs the strongest KG-based baseline (KGRec):

| Metric | KGRec | RA-GARK | Δ |
|---|---|---|---|
| NDCG@20 | 0.1095 | 0.1231 | **+12.4 %** |
| HR@20 | 0.4729 | 0.4961 | **+4.9 %** |
| Recall@20 | 0.1834 | 0.2003 | **+9.2 %** |
| MAP@20 | 0.0500 | 0.0586 | **+17.2 %** |

---

## 9. Ablation

Each row flips one component of RA-GARK off and retrains from scratch
(seed=42, 80 epochs, patience=10). `winner` = full RA-GARK with softmax
rationale + fusion bias 5.0 (the new defaults). `old_full` reverts
both Fix-1 changes (sigmoid rationale + fusion bias 0), giving the
original broken configuration. `lightgcn_only` is the no-KG floor.

| Preset                 | NDCG       | Δ vs winner | What it removes                                         |
|------------------------|------------|-------------|---------------------------------------------------------|
| **winner**             | **0.1231** | —           | softmax rationale + fusion_bias=5 + every component on  |
| winner_no_kg_lr        | 0.1230     | **−0.1 %**  | single lr for all params (per-param LR is negligible)   |
| winner_no_rat          | 0.1222     | **−0.7 %**  | uniform mean over aspects (no rationale at all)         |
| winner_no_acl          | 0.1207     | **−1.9 %**  | drop aspect-level CL                                    |
| winner_no_ucl          | 0.1187     | **−3.6 %**  | drop user cross-view CL                                 |
| winner_no_svd          | 0.1173     | **−4.7 %**  | xavier init for `item_kg_aspects` instead of KG SVD     |
| winner_fb0             | 0.1173     | **−4.7 %**  | revert fusion bias (5 → 0); α starts 0.5                |
| winner_sigmoid_rat     | 0.1152     | **−6.4 %**  | revert rationale head (softmax → sigmoid MLP)           |
| old_full               | 0.1067     | **−13.3 %** | both Fix-1 reverts → pre-fix broken full config         |
| no_global_view         | 0.1214     | −1.4 %      | skip the whole global pipeline (CL-only dual-view)      |
| lightgcn_only          | 0.1179     | −4.2 %      | no KG at all (floor)                                    |

**Reading the table.** Three contributions carry the headline
novelty, each responsible for ≥ 4.7% of NDCG when removed individually:
softmax rationale, local-biased fusion init, and KG SVD init. Below
them sit the two contrastive channels (uCL −3.6%, aCL −1.9%). Rationale
*enabled* adds only −0.7%, so the lift from softmax is mostly
*undoing the harm sigmoid causes* rather than the rationale signal
being a large standalone contribution. Per-parameter LR is retained
for training stability but its direct NDCG contribution is negligible
(−0.1%).

Reproduce with `python run_ablations.py`; raw numbers also land in
`ablation_results.csv`.
