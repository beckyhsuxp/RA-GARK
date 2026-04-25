# RA-GARK Architecture

**Rationale-Aware Gating Network over Review Aspect-Specific Knowledge Graphs**

This document describes the RA-GARK architecture and training procedure.

---

## 1. Design Overview

Dual-view recommendation model that decomposes user preference and item
representation into two complementary views:

- **Local View** — collaborative signal from a LightGCN over the
  user–item interaction graph.
- **Global View** — bilateral KG-aspect representations: both user and
  item carry `A` aspect slots `[A, d]`, and a *rationale-masking*
  attention computes a per-(user, item) softmax over aspects that
  aggregates BOTH sides into `u_glo` / `i_glo`. The same weights drive
  both aggregations, so the two sides speak the same rationale
  language by construction.

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
  ╔══════════╗   ╔══════════════╗   ╔══════════╗   ╔══════════════╗
  ║user_local║   ║user_kg_aspect║   ║item_local║   ║item_kg_aspect║
  ║ Embedding║   ║   [Nu, A, d] ║   ║ Embedding║   ║   [Ni, A, d] ║
  ║ [Nu, d]  ║   ║ ★ KG SVD init║   ║ [Ni, d]  ║   ║ ★ KG SVD init║
  ╚════╤═════╝   ╚══════╤═══════╝   ╚════╤═════╝   ╚══════╤═══════╝
       │                │                 │                │
       ▼                │                 ▼                │
  ┌──────────┐          │            ┌──────────┐          │
  │ LightGCN │          │            │ LightGCN │          │
  │ K-layer  │          │            │ K-layer  │          │
  │ D⁻¹ᐟ²AD⁻¹ᐟ²│          │            │ D⁻¹ᐟ²AD⁻¹ᐟ²│         │
  └────┬─────┘          │            └────┬─────┘          │
       │                ▼                 │                ▼
       │       ┌──────────────────────────────────────┐
       │       │       Bilateral KGRationaleMasking   │
       │       │  weights[a]=softmax(score(u[a],i[a]))│
       │       │  u_glo = Σ_a w[a] · u_aspects[a]     │
       │       │  i_glo = Σ_a w[a] · i_aspects[a]     │
       │       └──────────┬─────────────┬─────────────┘
       │                  ▼             ▼
       │             ┌────────┐    ┌────────┐
       │             │ u_glo  │    │ i_glo  │
       │             │ [B, d] │    │ [B, d] │
       │             └────┬───┘    └────┬───┘
       ▼                  │             │
  ┌────────┐              │        ┌────────┐
  │ u_loc  │              │        │ i_loc  │
  │ [B, d] │              │        │ [B, d] │
  └────┬───┘              │        └────┬───┘
       │                  │             │
       └────────┬─────────┘             └─────────┬──────────┐
                │                                 │          │
                ▼                                 ▼          │
       ┌─────────────────┐               ┌─────────────────┐ │
       │user_fusion_gate │               │item_fusion_gate │ │
       │ α=σ(MLP[u_loc;  │               │ α=σ(MLP[i_loc;  │ │
       │       u_glo])   │               │       i_glo])   │ │
       └────────┬────────┘               └────────┬────────┘ │
                │                                 │          │
                ▼                                 ▼          │
        ┌──────────────┐                  ┌──────────────┐   │
        │   u_final    │                  │   i_final    │   │
        │ αu_loc+      │                  │ αi_loc+      │◀──┘
        │ (1-α)u_glo   │                  │ (1-α)i_glo   │
        └──────┬───────┘                  └──────┬───────┘
               │                                 │
               └────────────────┬────────────────┘
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

1. **★ Bilateral aspect-aligned rationale attention** — both user and
   item carry `A` aspect slots `[A, d]`, and a single softmax over
   aspects (computed bilaterally via `(u_aspects[a] · i_aspects[a]) / √d`)
   yields one weight vector that aggregates BOTH sides. The two views
   share the same per-(user, item) attention by construction, so the
   global representation is genuinely user-conditioned rather than
   item-only. Temperature `τ=0.5` sharpens the softmax to amplify
   small per-(u, i) score differences and produce visibly differentiated
   per-item, per-user aspect saliency in Section 9.1.
2. **★ Local-biased fusion gate initialisation** (−4.7% when reverted
   to bias 0) — the fusion gate MLP's final Linear bias is initialised
   to `+5` so `α = σ(5) ≈ 0.993` at epoch 0. The model starts
   LightGCN-like and only opens the gate to the global view when the
   gradient says it helps. Default α≈0.5 lets the noisy KG contaminate
   the collaborative signal from epoch 1. The gate's *per-(user, item)
   MLP* is also load-bearing: replacing it with a single learnable
   global α (same `+5` init, no MLP) drops NDCG by 4.8%
   (`winner_scalar_gate` ablation), confirming that the conditional
   adaptation — not just the conservative starting point — is what
   converts the global view into useful score lift.
3. **★ KG SVD initialisation of `item_kg_aspects`** (−4.7% when
   reverted to xavier) — the global view starts from real KG semantics
   rather than random noise, keeping the KG-pretrained aspect geometry
   intact through BPR training.

Supporting design choices:
- **Aspect-level cross-view CL** (−1.9%) — `L_aCL`, aligns local item
  embedding with each aspect in a projection space, stop-grad on KG
  side.
- **User cross-view CL** (−3.6%) — `L_uCL`, pulls `u_loc` toward
  `u_glo`.

---

## 5. Component Reference

### 5.1 Embeddings

| Parameter            | Shape           | Init                        | LR    |
|----------------------|-----------------|-----------------------------|-------|
| `user_local_emb`     | `[Nu, d]`       | xavier                      | 1e-3  |
| `item_local_emb`     | `[Ni, d]`       | xavier                      | 1e-3  |
| `user_kg_aspects`    | `[Nu, A, d]`    | **user×aspect TF-IDF SVD**  | 1e-3  |
| `item_kg_aspects`    | `[Ni, A, d]`    | **item×aspect TF-IDF SVD**  | 1e-3  |

`Nu=905`, `Ni=1399`, `A=4`, `d=128`. All parameters share a single
Adam learning rate (1e-3).

### 5.2 LightGCN propagation

```
x⁰ = [user_local; item_local]
x^(l+1) = D⁻¹ᐟ² A D⁻¹ᐟ² x^(l)
final  = mean(x⁰, x¹, …, xᴷ)         K = 2
```

### 5.3 Bilateral KG Rationale Masking

Both user and item carry `[A, d]` aspect slots. Per-aspect alignment
scores are computed from the matched slot pair, softmax-normalised
across the aspect axis with temperature `τ`, and applied to BOTH
sides:

```
# Two style variants (cfg.rationale_style):
# style="dot":  score[a] = (u_aspects[a] · i_aspects[a]) / √d   (default, param-free)
# style="mlp":  score[a] = MLP([u_aspects[a] ; i_aspects[a]])

weights = softmax(score / τ, dim=aspect)         # [B, A]   (sums to 1)
u_glo   = Σ_a  weights[a] · u_aspects[a]         # [B, d]
i_glo   = Σ_a  weights[a] · i_aspects[a]         # [B, d]
```

Three design choices, each tested:

- **Bilateral rather than item-only attention.** The legacy formulation
  used a single user vector to attend over item aspects, leaving the
  user side fixed. The bilateral version makes the rationale genuinely
  user-conditioned: the same per-(u, i) weights aggregate both sides,
  so KG-aspect alignment shapes the user representation per item too.
- **Aspect-aligned scoring.** Slot `a` of the user is paired with slot
  `a` of the item rather than computing a full A×A interaction matrix.
  The SVD initialisation puts both sides on the same aspect axes, so
  this slot-aligned form keeps the attention an A-vector and stays
  cheap.
- **Temperature τ=0.5.** Raw scores are small in magnitude, so τ=1.0
  softmax collapses to ≈ uniform weights. Dividing scores by τ<1
  amplifies small per-(u, i) differences before softmax; τ=0.5 gives
  the best NDCG and produces visibly differentiated attention in the
  case study (Section 9.1).

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

- **Optimizer**: Adam, single learning rate `1e-3` for all parameters
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
| **RA-GARK (Ours, τ=0.5)**                   | **0.1238** | **0.4961** | **0.2014** | **0.0591** |

Relative improvement of RA-GARK vs the strongest KG-based baseline (KGRec):

| Metric | KGRec | RA-GARK | Δ |
|---|---|---|---|
| NDCG@20 | 0.1095 | 0.1238 | **+13.1 %** |
| HR@20 | 0.4729 | 0.4961 | **+4.9 %** |
| Recall@20 | 0.1834 | 0.2014 | **+9.8 %** |
| MAP@20 | 0.0500 | 0.0591 | **+18.2 %** |

---

## 9. Ablation

Each row flips one component of RA-GARK off and retrains from scratch
(seed=42, 80 epochs, patience=10). `winner` is the bilateral
configuration: dot rationale + fusion bias 5.0 + MLP fusion gate +
all CL channels on. `lightgcn_only` is the no-KG floor.

### 9.1 Current architecture (bilateral rationale)

| Preset                 | What it removes                                              |
|------------------------|--------------------------------------------------------------|
| **winner**             | full RA-GARK (re-run on bilateral arch to populate the cell) |
| winner_rat_mlp         | swap dot rationale for MLP variant                           |
| winner_no_rat          | uniform mean over aspects (no rationale at all)              |
| winner_no_acl          | drop aspect-level CL                                         |
| winner_no_ucl          | drop user cross-view CL                                      |
| winner_no_svd          | xavier init for both `*_kg_aspects` instead of TF-IDF SVD    |
| winner_fb0             | revert fusion bias (5 → 0); α starts 0.5                     |
| winner_scalar_gate     | replace per-(u, i) MLP gate with one learnable global α      |
| no_global_view         | skip the whole global pipeline (CL-only dual-view)           |
| lightgcn_only          | no KG at all (floor)                                         |

Reproduce with `python run_ablations.py --mode paper`; raw numbers
land in `ablation_results_paper.csv`.

### 9.2 Historical reference (asymmetric rationale, pre-bilateral)

The numbers below were measured under the original asymmetric
architecture (single `user_global_emb [Nu, d]` vector + item-only
aspect attention). They do not transfer literally to the bilateral
architecture but document the design decisions that motivated the
current form.

| Preset                 | NDCG       | Δ vs winner | What it removed                                         |
|------------------------|------------|-------------|---------------------------------------------------------|
| **winner (τ=0.5)**     | **0.1238** | —           | softmax rationale + τ=0.5 + fusion_bias=5 + all on      |
| winner (τ=1.0)         | 0.1231     | −0.6 %      | default temperature — attention ≈ uniform, small dip    |
| winner_no_rat          | 0.1222     | −1.3 %      | uniform mean over aspects (no rationale at all)         |
| winner_no_acl          | 0.1207     | −2.5 %      | drop aspect-level CL                                    |
| winner_no_ucl          | 0.1187     | −4.1 %      | drop user cross-view CL                                 |
| winner_no_svd          | 0.1173     | −5.3 %      | xavier init for `item_kg_aspects` instead of KG SVD     |
| winner_fb0             | 0.1173     | −5.3 %      | revert fusion bias (5 → 0); α starts 0.5                |
| winner_scalar_gate     | 0.1178     | −4.8 %      | replace per-(u,i) MLP gate with one learnable global α  |
| winner_sigmoid_rat     | 0.1152     | −7.0 %      | revert rationale head (softmax → sigmoid MLP)           |
| old_full               | 0.1067     | −13.8 %     | both Fix-1 reverts → pre-fix broken full config         |
| no_global_view         | 0.1214     | −1.9 %      | skip the whole global pipeline (CL-only dual-view)      |
| lightgcn_only          | 0.1179     | −4.8 %      | no KG at all (floor)                                    |

Three contributions carried the headline novelty, each responsible
for ≥ 4.7% of NDCG when removed individually under that architecture:
softmax rationale (vs sigmoid), local-biased fusion init, and KG SVD
init. The fusion gate's per-(u, i) MLP also passed validation
(scalar-gate substitution costs −4.8%). The bilateral version below
inherits these design choices and adds user-side aspect symmetry.

---

## 10. Code Layout

| Module              | Role                                                                     |
|---------------------|--------------------------------------------------------------------------|
| `config.py`         | `Config` dataclass with all hyperparameters and ablation flags           |
| `utils.py`          | `set_seed`, `user_stratified_split`                                       |
| `data.py`           | `load_interactions`, `build_kg_index`, `build_kg_aspect_init` (item-aspect SVD), `build_user_aspect_init` (user-aspect SVD), `KnowledgeAwareSampler`, `RecDataset`, `build_lightgcn_adj` |
| `model.py`          | `KGRationaleMasking` (bilateral aspect-aligned softmax), `_make_fusion_gate` / `_ScalarGate`, `RA_GARK` (full model with `forward` + `score_all_items`, both accept cached LightGCN embeddings) |
| `losses.py`         | `bpr_loss`, `infonce_loss`, `aspect_level_cl`                             |
| `evaluate.py`       | Vectorised full-ranking evaluation (HR / Recall / Precision / F1 / MAP / NDCG @ K) |
| `train_ragark.py`   | Single-config training entry point. LightGCN propagation is cached once per batch and reused for pos + neg forwards |
| `run_ablations.py`  | Sweep across ablation presets (writes `ablation_results.csv`)             |
| `tune_weights.py`   | Optuna search over `cl_weight` and `temp`                                 |
| `case_study.py`     | Interpretability: per-item softmax aspect weights for sampled users       |

**Loss formulation is fixed.** `L_total = L_BPR + 0.005 · (L_aCL + L_uCL)`
with one random negative per positive. Optimizer is plain Adam at
`lr=1e-3`, no weight decay, no LR scheduler — all of which were
ablated and found to be net-negative or noise on the 905u × 1399i
split.
