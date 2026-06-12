# RA-GARK 口試簡報文字版

> 30 分鐘口試用。這份只保留「每頁投影片上要放的文字」。
> 圖片已標示在對應頁面，直接放你做好的 `img`。

**圖檔對應**

| 圖檔 | 頁面 |
|---|---|
| `thesis/img/architecture.png` | Slide 12 |
| `thesis/img/kg_svd.png` | Slide 18 |
| `thesis/img/gate.png` | Slides 23-25 |
| `thesis/img/sensitivity_2x2.png` | Slide 22 |
| `thesis/img/case_study_heatmap.png` | Slide 30 |

---

## Slide 1 — Title

**RA-GARK**

Product Recommendation via Rationale-Aware Gating over Sparse Review-Aspect Knowledge Graphs

基於理由感知門控與稀疏評論面向知識圖譜之產品推薦

KG-aware Recommendation · Sparse KG · Rationale-aware Gating · Graceful Degradation

**Main idea**

KG should be a gateable side channel, not a mandatory scoring path.

---

## Slide 2 — Roadmap

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

**Where sparse KG comes from**

- review-derived KGs inherit their density from whatever users happened to mention
- cold-start and emerging domains rarely have a curated KG
- privacy-constrained domains deliberately restrict relational signals
- aggressive KG completion introduces its own noise and needs seed signal

**What this means**

- sparse KG is the more common practical setting
- robustness under unreliable KG signal deserves dedicated study
- the primary design objective is robustness, not only peak performance

**Takeaway**

- sparse KG is the regime this thesis targets

---

## Slide 5 — Design Challenge

**Observed tension in prior KG-aware methods**

- KG entity embeddings participate directly in message passing
- the implicit assumption is that KG can inject useful signal everywhere it appears
- under sparse KG, that assumption breaks down

**Why LightGCN wins**

- It uses only user-item interactions
- No KG contamination
- Strong safe default

**Our response**

- route KG signal through a dedicated side channel
- let the model attenuate or fully disengage KG when unreliable

---

## Slide 6 — Research Question

**Research question**

How can a recommender use KG when it helps, but avoid KG contamination when KG is sparse or unreliable?

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

**KGRec vs RA-GARK**

| Axis | KGRec | RA-GARK |
|---|---|---|
| Granularity | KG edge | latent aspect slot |
| Selection | Bernoulli dropout + CL | softmax attention |
| Integration | inside KGAT propagation | separate side channel |
| KG trust | cannot disengage KG | can suppress KG |

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

| Module | Output |
|---|---|
| Local View | `u_loc`, `i_loc` |
| Global View | `u_glo`, `i_glo` |
| Fusion Gate | `u_final`, `i_final` |
| Training Loss | ranking objective |

---

## Slide 13 — Problem Setup

**Implicit top-K recommendation**

- rank unseen items for each user
- train with positive and sampled negative pairs

**Score**

```text
y_hat(u, i) = <u_final, i_final>
```

**Fusion**

```text
u_final = alpha_u * u_loc + (1 - alpha_u) * u_glo
i_final = alpha_i * i_loc + (1 - alpha_i) * i_glo
```

---

## Slide 14 — Local View

**Pure LightGCN**

- no KG in this branch
- no nonlinear transform
- no extra weights in propagation

**Why**

- preserve a clean CF backbone
- keep a safe fallback path

---

## Slide 15 — Local Propagation

**Graph**

- user-item bipartite graph
- training interactions only

**Propagation**

```text
E^(l+1) = A_norm E^(l)
E_loc = average(E^(0), E^(1), ..., E^(K))
```

**Setting**

- K = 2
- output: `u_loc`, `i_loc`

---

## Slide 16 — Global View

**Why latent aspect slots**

- raw review-aspect KG is sparse
- direct propagation is fragile
- latent slots give a compact KG representation

**Representation**

```text
item_kg_aspects[i] in R^(A x d)
A = 4
d = 128
```

---

## Slide 17 — KG-SVD Step 1

**Build item-aspect matrix**

```text
M[i, a] = 1 if item i has aspect a
```

**IDF weighting**

```text
M_tilde[i, a] = M[i, a] * idf(a)
idf(a) = log(N_items / support(a) + 1) + 1
```

**Purpose**

- downweight generic aspects
- keep discriminative aspects

---

## Slide 18 — KG-SVD Step 2

**圖片**

`thesis/img/kg_svd.png`

**Truncated SVD**

```text
M_tilde ~= U Sigma V^T
E_KG = U sqrt(Sigma)
```

**Reshape**

```text
E_KG[i] -> item_kg_aspects[i] in R^(4 x 128)
```

**Why**

- give KG a semantic starting geometry
- preserve the aspect co-occurrence structure before training

---

## Slide 19 — KG-SVD Ablation

| Model | NDCG@20 | MAP@20 |
|---|---:|---:|
| RA-GARK (full) | 0.1243 | 0.0594 |
| w/o KG-SVD init | 0.1171 | 0.0545 |

**Observation**

- KG-SVD preserves the initial semantic geometry

---

## Slide 20 — Softmax Masking

**Goal**

Select which aspect slot should represent the item for a given user-item pair.

**Computation**

```text
logit_k = MLP([u_glo || aspect_slot_i,k])
w_k = softmax(logit_k / tau)
i_glo = sum_k w_k * aspect_slot_i,k
```

**Why user-conditioned**

- different users care about different item aspects

---

## Slide 21 — Softmax vs Sigmoid

**Normalization assumption**

| Normalization | Assumption |
|---|---|
| Sigmoid | each slot is independently important |
| Softmax | slots compete under fixed mass |

**In RA-GARK**

- softmax controls both weight competition and output magnitude
- this matters because the KG channel is intentionally throttled

---

## Slide 22 — Softmax Ablation

**圖片**

`thesis/img/sensitivity_2x2.png`

| Model | NDCG@20 | MAP@20 |
|---|---:|---:|
| RA-GARK (full) | 0.1243 | 0.0594 |
| w/o softmax head | 0.1005 | 0.0451 |

**Observation**

- softmax is the thesis-normalized rationale operator

---

## Slide 23 — Fusion Gate

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

## Slide 24 — Gate Bias

**Bias initialization**

```text
gate final bias = +5
alpha_0 = sigmoid(+5) ~= 0.993
```

**Meaning**

- start almost as LightGCN
- open KG only when useful

---

## Slide 25 — Graceful Degradation

**Graceful degradation**

If KG is not useful, RA-GARK falls back to LightGCN.

**Ablation**

| Model | NDCG@20 | MAP@20 |
|---|---:|---:|
| RA-GARK (full) | 0.1243 | 0.0594 |
| w/o fusion-gate bias | 0.1194 | 0.0555 |

**Observation**

- the bias initialization is part of the architecture, not a tuning trick

---

## Slide 26 — Contrastive Regularization

**Main objective**

```text
L = L_BPR + lambda_CL (L_aCL + L_uCL)
lambda_CL = 0.005
```

**Role**

- auxiliary only
- weak alignment
- not the main fusion mechanism

**Conservative design**

- small weight
- stop-gradient on KG side
- projection head

---

## Slide 27 — Training Setup

**Dataset signal**

- 905 users
- 1,399 items
- 22,265 interactions
- 3,370 KG edges
- 2,098 aspects

**Training**

- Adam
- learning rate 1e-3
- batch size 128
- 80 epochs with early stopping

---

## Slide 28 — Evaluation Setup

**Evaluation**

- full ranking
- exclude training interactions
- metrics: HR, Precision, Recall, F1, MAP, NDCG @20

**Efficiency**

- about 1.5 seconds per epoch
- comparable to KGRec
- no extra training burden

---

## Slide 29 — Main Results

| Model | NDCG@20 | HR@20 | Recall@20 | MAP@20 |
|---|---:|---:|---:|---:|
| MCCLK | 0.1067 | 0.4530 | 0.1720 | 0.0497 |
| KGCL | 0.1073 | 0.4696 | 0.1827 | 0.0479 |
| KGAT | 0.1079 | 0.4773 | 0.1807 | 0.0491 |
| KGRec | 0.1095 | 0.4729 | 0.1834 | 0.0500 |
| LightGCN | 0.1179 | 0.4917 | 0.1937 | 0.0555 |
| RA-GARK | 0.1243 | 0.4972 | 0.2020 | 0.0594 |

| Model | NDCG@10 | HR@10 | Recall@10 | MAP@10 |
|---|---:|---:|---:|---:|
| MCCLK | 0.0804 | 0.3182 | 0.1047 | 0.0416 |
| KGCL | 0.0809 | 0.3260 | 0.1096 | 0.0410 |
| KGAT | 0.0786 | 0.3215 | 0.1102 | 0.0388 |
| KGRec | 0.0874 | 0.3249 | 0.1155 | 0.0465 |
| LightGCN | 0.0908 | 0.3436 | 0.1201 | 0.0483 |
| RA-GARK | 0.0966 | 0.3558 | 0.1265 | 0.0520 |

**Key point**

- RA-GARK is best at both Top-20 and Top-10.

---

## Slide 30 — Ablation Summary

| Model | NDCG@20 | MAP@20 |
|---|---:|---:|
| RA-GARK (full) | 0.1243 | 0.0594 |
| w/o softmax head | 0.1005 | 0.0451 |
| w/o KG-SVD init | 0.1171 | 0.0545 |
| w/o fusion-gate bias | 0.1194 | 0.0555 |
| w/o MLP gate | 0.1180 | 0.0552 |
| w/o user CL ($\mathcal{L}_{\mathrm{uCL}}$) | 0.1192 | 0.0563 |
| w/o aspect CL ($\mathcal{L}_{\mathrm{aCL}}$) | 0.1200 | 0.0570 |
| w/o rationale-enabled selection | 0.1213 | 0.0568 |
| w/o global view (CL-only dual view) | 0.1219 | 0.0575 |

| Model | NDCG@10 | MAP@10 |
|---|---:|---:|
| RA-GARK (full) | 0.0960 | 0.0519 |
| w/o softmax head | 0.0785 | 0.0397 |
| w/o KG-SVD init | 0.0922 | 0.0479 |
| w/o fusion-gate bias | 0.0923 | 0.0482 |
| w/o MLP gate | 0.0926 | 0.0484 |
| w/o user CL ($\mathcal{L}_{\mathrm{uCL}}$) | 0.0924 | 0.0492 |
| w/o aspect CL ($\mathcal{L}_{\mathrm{aCL}}$) | 0.0940 | 0.0502 |
| w/o rationale-enabled selection | 0.0943 | 0.0495 |
| w/o global view (CL-only dual view) | 0.0949 | 0.0504 |

**Takeaway**

- the core architecture rows dominate the ranking loss

---

## Slide 31 — Case Study and Takeaways

**圖片**

`thesis/img/case_study_heatmap.png`

**Takeaways**

- different items activate different aspect slots
- rationale masking gives interpretability
- the model is not only accurate, it also shows which slot is used

---

## Slide 32 — Conclusion

**Conclusion**

When the KG is unreliable, what an architecture most needs is not a better KG aggregator but a structural switch by which the KG can be opted out of.

**Contributions**

- gateable KG side channel
- KG-SVD initialization
- softmax rationale masking
- local-biased fusion gate

**Limitations**

- one sparse review-aspect KG dataset
- KG construction pipeline is adopted
- dense-KG settings may still favor deep fusion
