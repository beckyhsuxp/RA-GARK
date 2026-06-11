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

| Method | Year | NDCG@20 |
|---|---:|---:|
| MCCLK | 2022 | 0.1067 |
| KGCL | 2022 | 0.1073 |
| KGAT | 2019 | 0.1079 |
| KGRec | 2023 | 0.1095 |
| Pure LightGCN | — | 0.1179 |

**Key point**

On this sparse KG, every KG-aware baseline loses to pure LightGCN.

---

## Slide 4 — Why Sparse KG

**Sparse KG is the default, not an edge case**

- Review-derived KG
- Cold-start or niche domains
- Long-tail items
- Privacy-limited domains
- KG completion may add noise

**Why this matters**

- sparse KG makes deep fusion fragile
- more KG is not always better if the signal is noisy
- the model needs a safe fallback path

**What this implies**

- the benchmark is a stress test, not a best-case setting
- the design goal should be robustness, not only peak accuracy
- later slides need an explicit mechanism to downweight KG

---

## Slide 5 — Design Challenge

**Observed tension in prior KG-aware methods**

- KG is wired into the scoring path
- KG signal and KG noise are propagated together
- the model has limited ability to reduce KG influence when it is unreliable

**Why LightGCN wins**

- It uses only user-item interactions
- No KG contamination
- Strong safe default

---

## Slide 6 — Research Question

**Research question**

How can a recommender use KG when it helps, but avoid KG contamination when KG is sparse or unreliable?

**RA-GARK answer**

KG should be a gateable side channel.

---

## Slide 7 — Related Work I

**Pure CF**

- LightGCN is the strongest safe baseline

**Deep KG fusion**

- KGAT propagates KG entities into user/item embeddings

**Position of RA-GARK**

- keep LightGCN as local view
- avoid mandatory KG propagation

---

## Slide 8 — Related Work II

**Contrastive KG methods**

- KGCL
- MCCLK

**Assumption**

- multiple KG views are still informative

**Sparse-KG issue**

- sparse or perturbed KG gives weak supervision

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

- Highway Networks: safe identity path
- MMoE / PLE: gate over expert towers
- SGL / DCCF: alignment over views from the same graph

**Gap in KG-aware recommendation**

- no bias-initialized fusion gate
- no architectural fallback to pure CF

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

| Variant | NDCG@20 |
|---|---:|
| full RA-GARK | 0.1243 |
| random KG init | 0.1171 |

**Observation**

- random initialization is weaker on this benchmark

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

| Variant | NDCG@20 |
|---|---:|
| full RA-GARK | 0.1243 |
| sigmoid rationale | 0.1005 |

**Observation**

- sigmoid weakens the rationale module on this benchmark

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

| Variant | NDCG@20 |
|---|---:|
| full RA-GARK | 0.1243 |
| gate bias = 0 | 0.1194 |

**Observation**

- local-biased init gives a safer starting point

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

**Key point**

- RA-GARK is the best model on all four reported ranking metrics.

---

## Slide 30 — Ablation Summary

| Setting | NDCG@20 |
|---|---:|
| full RA-GARK | 0.1243 |
| w/o Softmax | 0.1005 |
| w/o gate init | 0.1194 |
| w/o KG-SVD | 0.1171 |

**Takeaway**

- the core architecture components all matter

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

In sparse KG recommendation, the key is not to force KG into the model, but to give the model a reliable way to ignore it.

**Contributions**

- gateable KG side channel
- KG-SVD initialization
- softmax rationale masking
- local-biased fusion gate

**Limitations**

- one sparse review-aspect KG dataset
- KG construction pipeline is adopted
- dense-KG settings may still favor deep fusion
