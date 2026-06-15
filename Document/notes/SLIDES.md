# RA-GARK 口試簡報文字版

> 30 分鐘口試用。這份只保留「每頁投影片上要放的文字」。
> 圖片已標示在對應頁面，直接放你做好的 `img`。

**圖檔對應**

| 圖檔 | 頁面 |
|---|---|
| `thesis/img/architecture.png` | Slide 12 |
| `thesis/img/kg_svd.png` | Slide 21 |
| `thesis/img/gate.png` | Slide 28 |
| `thesis/img/sensitivity_2x2.png` | Slide 27 |
| `thesis/img/case_study_heatmap.png` | Slide 36 |

---

## Slide 1 — Title

**RA-GARK**

Product Recommendation via Rationale-Aware Gating over Sparse Review-Aspect Knowledge Graphs

基於理由感知門控與稀疏評論面向知識圖譜之產品推薦

KG-aware Recommendation · Sparse KG · Rationale-aware Gating · Graceful Degradation

**Main idea**

KG should be a gateable side channel, not a mandatory scoring path.

---

## Slide 2 — Outline

**30 分鐘配置**

- Introduction
- Related Work
- Methodology
- Experiments
- Conclusion & future work

**Focus**

Methodology is the main part; related work is only for positioning.

---

## Slide 3 — Motivation

**Sparse KG breaks KG-aware recommendation**

| Method | NDCG@20 |
|---|---:|
| MCCLK | 0.1067 |
| KGCL | 0.1073 |
| KGAT | 0.1079 |
| KGRec | 0.1095 |
| Pure LightGCN | 0.1179 |

**Key point**

On this sparse KG, every KG-aware baseline loses to pure LightGCN.

---

## Slide 4 — Why Sparse KG

**1. Source of sparsity**

- review-derived KGs only reflect user mentions
- cold-start and emerging domains lack curated KGs

**2. Why completion is not enough**

- privacy limits relational signals
- KG completion adds noise and still needs seed signal

**3. What this means**

- sparse KG is the practical default; robustness matters more than peak performance
- unreliable KG needs dedicated modeling

---

## Slide 5 — Design Challenge

**1. Failure mode**

- KG embeddings enter message passing directly
- prior methods assume KG is always useful
- sparse KG breaks that assumption

**2. Safe default**

- LightGCN only uses user-item interactions
- no KG contamination

**3. Our response**

- route KG through a dedicated side channel
- let the model attenuate or disengage KG when unreliable

---

## Slide 6 — Research Question

**RQ1: Diagnosis**

Why do KG-aware methods underperform pure LightGCN under sparse KG?

**RQ2: Prescription**

What design principle lets a model use KG when helpful and avoid contamination when unreliable?

**RA-GARK answer**

KG should be a gateable side channel.

---

## Slide 7 — Related Work I

**LightGCN**

- immediate predecessor of our local view
- strong non-KG anchor on sparse review KG

**KGAT**

- canonical deep-fusion approach
- KG entities participate directly in propagation

**Position of RA-GARK**

- adopt LightGCN verbatim as local view
- isolate KG signal into a separate global view

---

## Slide 8 — Related Work II

**Contrastive KG methods**

- KGCL
- MCCLK

**Assumption**

- KG structure remains informative under perturbation
- collaborative, semantic, and structural views can be aligned

**Sparse-KG issue**

- sparse or perturbed KG gives weak supervision
- contrastive alignment may become noise-dominated

---

## Slide 9 — Related Work III

| KGRec | RA-GARK |
|---|---|
| edge-level rationale | latent aspect-slot rationale |
| Bernoulli dropout + CL | softmax attention |
| stays inside KGAT propagation | separate global side channel |
| cannot fully disengage KG | can suppress the whole KG |

**Main difference**

KGRec assumes useful edges exist; RA-GARK assumes the whole KG channel may be unreliable.

---

## Slide 10 — Related Work IV

**Gating gap**

- Highway Networks: bias toward a safe identity path
- MMoE / PLE: gate over expert towers
- SGL / DCCF: alignment over views from the same graph

**Gap in KG-aware recommendation**

- no bias-initialized fusion gate
- no architectural graceful degradation under sparse or unreliable KG

---

## Slide 11 — Design Principle

**RA-GARK principle**

KG should be a gateable side channel, not a mandatory scoring component.

**Three consequences**

- separate local and global views
- fuse late
- bias the gate toward LightGCN at initialization

---

## Slide 12 — Overview

**圖片**

`thesis/img/architecture.png`

**Modules**

- Local View -> `u_loc`, `i_loc`
- Global View -> `u_glo`, `i_glo`
- Fusion Gate -> `u_final`, `i_final`
- Training Loss -> ranking objective

---

## Slide 13 — Problem Setup I

**Task**

- implicit-feedback top-K recommendation
- rank unseen items for each user
- train with positive and sampled negative pairs

**Score**

- `y_hat(u, i) = <u_final, i_final>`

**Readout**

- higher score means stronger match

---

## Slide 14 — Problem Setup II

**Fusion**

- `u_final = alpha_u * u_loc + (1 - alpha_u) * u_glo`
- `i_final = alpha_i * i_loc + (1 - alpha_i) * i_glo`

**Why two gates**

- user-side and item-side KG usefulness differ
- separate parameters work better than shared ones

**Goal**

- maximize ranking quality on held-out items

---

## Slide 15 — Local View

**Pure LightGCN**

- no KG in this branch
- no nonlinear transform
- no extra weights in propagation

**Why**

- preserve a clean CF backbone
- keep a safe fallback path

---

## Slide 16 — Local Propagation

**Graph**

- user-item bipartite graph
- training interactions only

**Propagation**

```text
E^(l+1) = A_norm E^(l), l = 0, 1, ..., K-1
\bar{E} = (1 / (K + 1)) * sum_{l=0}^K E^(l)
```

**Setting**

- K = 2
- output: `u_loc`, `i_loc` from `\bar{E}`

---

## Slide 17 — Global View

**Why latent aspect slots**

- each item is compressed into four fixed semantic slots
- the slots keep the KG signal compact and readable
- sparse KG makes direct propagation fragile

**Representation**

`a_i in R^(A x d), with A = 4 and d = 128`

---

## Slide 18 — KG-SVD Motivation

**Why KG-SVD**

- the raw item-aspect matrix is sparse
- generic aspects should count less
- we want a semantic starting geometry

**Key idea**

- initialize latent slots from co-occurrence structure
- keep the representation compact and stable

---

## Slide 19 — KG-SVD Construction

1. **Build item-aspect matrix**

- binary co-occurrence between items and aspects

```text
M[i, a] = 1 if item i has aspect a
```

2. **IDF weighting**

- downweight common aspects

```text
M_tilde[i, a] = M[i, a] * idf(a)
idf(a) = log(|I| / (|{i : M[i, a] = 1}| + 1)) + 1
```

**Key idea**

- compact co-occurrence matrix

**Image**

`thesis/img/kg_svd.png`

---

## Slide 20 — KG-SVD SVD and Reshape

3. **Truncated SVD**

- compress the weighted matrix into low-rank factors

```text
M_tilde ~= U_k Sigma_k V_k^T
E_KG = U_k Sigma_k^(1/2)
```

4. **Reshape**

- turn each item vector into four aspect slots

```text
E_KG[i] -> item_kg_aspects[i] in R^(4 x 128)
```

**Why it helps**

- give KG a semantic starting geometry
- preserve the aspect co-occurrence structure before training

---

## Slide 21 — KG-SVD Effect

**Full model**

NDCG@20 0.1243, MAP@20 0.0594.

**Without KG-SVD init**

NDCG@20 0.1171, MAP@20 0.0545.

**Observation**

- KG-SVD preserves the initial semantic geometry

---

## Slide 22 — Softmax Masking Motivation

**Goal**

Select which aspect slot should represent the item for a given user-item pair.

**Why user-conditioned**

- different users care about different item aspects
- the same item can have different rationales for different users

---

## Slide 23 — Softmax Masking Computation

**Computation**

```text
logit_k = MLP([u_glo || aspect_slot_i,k])
w_k = softmax(logit_k / tau)
i_glo = sum_k w_k * aspect_slot_i,k
```

**Result**

- the item global vector is a weighted sum of slots

---

## Slide 24 — Softmax Normalization

**Normalization choice**

| Normalization | Assumption |
|---|---|
| Sigmoid | each slot is independently important |
| Softmax | slots compete under fixed mass |

**In RA-GARK**

- softmax controls weight competition
- softmax also controls output magnitude

---

## Slide 25 — Softmax vs Sigmoid

**Why softmax**

- sigmoid does not normalize across slots
- softmax gives a bounded, competition-based mask
- this matters because the KG channel is intentionally throttled

**Takeaway**

It matches the throttled KG channel.

---

## Slide 26 — Softmax Ablation

**Figure**

`thesis/img/sensitivity_2x2.png`

**Full model**

NDCG@20 0.1243, MAP@20 0.0594.

**Without softmax head**

NDCG@20 0.1005, MAP@20 0.0451.

**Observation**

- softmax is the thesis-normalized rationale operator

---

## Slide 27 — Fusion Gate Structure

**圖片**

`thesis/img/gate.png`

**Gate**

```text
alpha_u = sigmoid(MLP_gate([u_loc || u_glo]))
alpha_i = sigmoid(MLP_gate([i_loc || i_glo]))
```

**Fusion**

```text
u_final = alpha_u * u_loc + (1 - alpha_u) * u_glo
i_final = alpha_i * i_loc + (1 - alpha_i) * i_glo
```

---

## Slide 28 — Gate Bias and Graceful Degradation

**Bias initialization**

```text
b = +5
alpha_0 = sigmoid(+5) ~= 0.993
```

**Meaning**

- start almost as LightGCN
- open the KG channel only when it helps

**Graceful degradation**

- unreliable KG stays mostly closed
- without this bias, NDCG@20 drops from 0.1243 to 0.1194

---

## Slide 29 — Contrastive Regularization

**Main objective**

```text
L = L_BPR + lambda_CL * (L_aCL + L_uCL)
lambda_CL = 0.005
tau_CL = 0.2
```

**Role**

- auxiliary geometric alignment
- not the main integration path

**Stability**

- stop-gradient on the KG side
- projection head only on the local side

---

## Slide 30 — Training Objective

**BPR**

```text
L_BPR = -log sigma(y(u, i+) - y(u, i-))
```

**Sampling**

- positive pairs come from observed interactions
- negatives are sampled from items the user has not interacted with

**What it optimizes**

- push held-out positives above negatives
- let the gate and CL refine the representation

---

## Slide 31 — Dataset and Optimization

**Dataset**

- 905 users
- 1,399 items
- 22,265 interactions
- 3,370 KG edges
- 2,098 aspects

**Optimization**

- Adam
- learning rate 1e-3
- batch size 128
- up to 80 epochs with early stopping

**Why it matters**

- the benchmark is intentionally sparse
- the method is tested under a strict KG setting

---

## Slide 32 — Inference and Complexity

**Inference**

- full ranking over unseen items
- exclude training interactions
- vectorized scoring over the full item set

**Cost profile**

- LightGCN propagation dominates training cost
- KG-side modules scale linearly in `A` and `d`
- wall-clock is comparable to KGRec

**Takeaway**

- the extra KG machinery does not blow up runtime

---

## Slide 33 — Main Results

**Top-20**

- RA-GARK: NDCG@20 0.1243, HR@20 0.4972, Recall@20 0.2020, MAP@20 0.0594
- best across all reported metrics
- beats pure LightGCN by 5.4% on NDCG@20
- beats KGRec by 13.5% on NDCG@20

**Top-10**

- RA-GARK: NDCG@10 0.0966, HR@10 0.3558, Recall@10 0.1265, MAP@10 0.0520
- best across all reported metrics
- beats pure LightGCN by 6.4% on NDCG@10
- beats KGRec by 10.5% on NDCG@10

## Slide 34 — Ablation Summary

**Largest drop**

- w/o softmax head hurts the most
- it removes the rationale selection step

**Other core pieces**

- w/o KG-SVD init degrades the global view
- w/o fusion-gate bias weakens the safe default
- w/o MLP gate reduces fusion flexibility

**Smaller but consistent**

- w/o user CL and w/o aspect CL both drop performance
- w/o rationale-enabled selection and w/o global view also underperform

---

## Slide 35 — Case Study and Takeaways

**圖片**

`thesis/img/case_study_heatmap.png`

**Takeaways**

- different items activate different aspect slots
- rationale masking gives interpretability
- the model shows which slot is used for a prediction

---

## Slide 36 — Conclusion

**Conclusion**

When the KG is unreliable, what the architecture needs is not a better KG aggregator but a structural switch that can opt the KG out.

**Contributions**

- gateable KG side channel
- KG-SVD initialization
- softmax rationale masking
- local-biased fusion gate

**Limitations**

- one sparse review-aspect KG dataset
- KG construction pipeline is adopted rather than proposed
- dense-KG settings may still favor deep fusion

以上，謝謝大家。
