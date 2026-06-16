# RA-GARK 口試簡報文字版

> 30 分鐘口試用。這份只保留「每頁投影片上要放的文字」。
> 圖片已標示在對應頁面，直接放你做好的 `img`。

**圖檔對應**

| 圖檔 | 頁面 |
|---|---|
| `thesis/img/architecture.png` | Slide 10 / 23 |
| `thesis/img/kg_svd.png` | Slide 17 |
| `thesis/img/gate.png` | Slide 24 |
| `thesis/img/case_study_heatmap.png` | Slide 34 |

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

**Design challenge**

- KG is mixed directly into user/item representations
- KG becomes part of representation learning, not just extra evidence
- this assumes KG is useful whenever it is used
- sparse KG breaks that assumption
- key question: avoid unreliable KG affecting final scores

---

## Slide 6 — Research Question

**RQ1: Diagnosis**

Why do KG-aware methods underperform pure LightGCN under sparse KG?

**RQ2: Prescription**

What design principle lets a model use KG when helpful and avoid contamination when unreliable?

**RA-GARK answer**

KG should not be a mandatory scoring component.

---

## Slide 7 — Related Work I

**1. Collaborative Filtering**

- LightGCN

**Challenge**

- uses only user-item interaction
- cannot use item semantic information from KG

**2. Direct KG Fusion**

- KGAT
- KGRec

**Challenge**

- KG is used in user/item representation learning and score computation
- assumes the KG is useful enough to trust
- sparse KG can contaminate recommendation scores

---

## Slide 8 — Related Work II

**1. Contrastive KG Learning**

- KGCL
- MCCLK

**Challenge**

- relies on informative KG structure
- weak KG makes contrastive learning unstable

**2. Gating for Fusion**

- Highway Networks
- MMoE
- PLE

**Challenge**

- not designed for unreliable KG
- no explicit safe initialization for a KG side channel

---

## Slide 9 — Design Principle

**1. Failure mode**

- KG embeddings enter message passing directly
- prior methods assume KG is always useful
- sparse KG breaks that assumption

**2. Safe default**

- LightGCN only uses user-item interactions
- no KG contamination

**3. Our response**

- route KG through a dedicated side channel
- attenuate or disengage KG when unreliable

---

## Slide 10 — Overview

**圖片**

`thesis/img/architecture.png`

**Modules**

- Local View -> `u_loc`, `i_loc`
- Global View -> `u_glo`, `i_glo`
- Fusion Gate -> `u_final`, `i_final`
- Training Loss -> ranking objective

---

## Slide 11 — Problem Setup I

**Task**

- implicit-feedback top-K recommendation
- rank unseen items for each user
- train with positive and sampled negative pairs

**Score**

- `u`: user, `i`: item
- `u_final`: final user vector
- `i_final`: final item vector
- `y_hat(u, i) = <u_final, i_final>`

**Readout**

- higher score means stronger match

---

## Slide 12 — Problem Setup II

**Fusion**

- `u_final = alpha_u * u_loc + (1 - alpha_u) * u_glo`
- `i_final = alpha_i * i_loc + (1 - alpha_i) * i_glo`

**Why two gates**

- user-side and item-side KG usefulness differ
- separate parameters work better than shared ones

**Goal**

- maximize ranking quality on held-out items

---

## Slide 13 — Local View

**Pure LightGCN**

- no KG in this branch
- no nonlinear transform
- no extra weights in propagation

**Why**

- preserve a clean CF backbone
- keep a safe fallback path

---

## Slide 14 — Local Propagation

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

## Slide 15 — Global View

**Why latent aspect slots**

- do not propagate the entire KG directly
- organize each item into four fixed semantic slots
- keep KG semantics while avoiding passive intake of the full sparse KG
- later select useful aspects from these slots

**Representation**

`a_i in R^(A x d), with A = 4 and d = 128`

---

## Slide 16 — KG-SVD Motivation

**Why KG-SVD**

- the raw item-aspect matrix is sparse
- generic aspects should count less
- we want a semantic starting geometry

**Key idea**

- initialize latent slots from co-occurrence structure
- keep the representation compact and stable

---

## Slide 17 — KG-SVD: Construction

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

**Image**

`thesis/img/kg_svd.png`

---

## Slide 18 — KG-SVD: SVD and Reshape

3. **Truncated SVD**

- keep only the top-k factors of the weighted matrix, with `k = A · d`

```text
M_tilde ~= U_k Sigma_k V_k^T, with k = A · d
E_KG = U_k Sigma_k^(1/2)
```

4. **Reshape & Initialize**

- turn each item vector into four aspect slots

```text
E_KG -> A_KG^(0) in R^(|I| x A x d)
```

---

## Slide 20 — Softmax Masking Motivation

**Goal**

choose the slot for each user-item pair

**Why user-conditioned**

- different users care about different item aspects
- the same item can have different rationales for different users

---

## Slide 21 — Softmax Masking Computation

**Computation**

```text
\ell_{u, i, k} = \mathrm{MLP}\!\left( [u_{\mathrm{glo}} \,\Vert\, \mathbf{a}_{i,k}] \right)
w_{u, i, k} = \frac{\exp(\ell_{u, i, k} / \tau)}{\sum_{k' = 1}^{A} \exp(\ell_{u, i, k'} / \tau)}
i_{\mathrm{glo}} = \sum_{k=1}^{A} w_{u, i, k} \cdot \mathbf{a}_{i, k}
```

**Result**

- the item global vector is a weighted sum of slots

---

## Slide 22 — Softmax Normalization

**Normalization**

| Normalization | Assumption |
|---|---|
| Sigmoid | each slot is independently important |
| Softmax | slots compete under fixed mass |

**Why softmax**

- bounded mask
- controlled output magnitude
- better for a throttled KG channel

---

## Slide 23 — Fusion Gate Overview

**Image**

`thesis/img/architecture.png`

**Focus**

- zoom in on the fusion gate
- local and global stay separate until the final stage
- the gate decides how much KG to keep

---

## Slide 24 — Fusion Gate Structure

**圖片**

`thesis/img/gate.png`

**Gate**

```text
\alpha_u = \mathrm{Gate}_u\!\left([u_{\mathrm{loc}} \,\Vert\, u_{\mathrm{glo}}]\right) \in (0, 1)
\mathrm{Gate}(\mathbf{z}) = \sigma\!\left(\mathbf{w}^{\top} \tanh\!\left(\mathbf{W}\mathbf{z}\right) + b\right)
```

- 先做一層 `tanh(Wz)` 的隱層變換
- 先用 `Wz` 做線性投影，再接 `tanh` 非線性
- 再接 `w^T` 和 `sigmoid`，輸出 0 到 1 的 gate 權重

**Fusion**

```text
u_final = alpha_u * u_loc + (1 - alpha_u) * u_glo
```

---

## Slide 25 — Gate Bias and Graceful Degradation

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
- without this bias, the model starts from a less safe mix

---

## Slide 26 — Training Objective

**BPR**

- 排序損失，用已觀察互動和抽樣負例來訓練
```text
L_BPR = -log sigma(y(u, i+) - y(u, i-))
```

**Sampling**

- observed interaction pairs are treated as positive
- sampled unseen items are treated as negative

---

## Slide 27 — Total Objective

**Total objective**

```text
L = L_BPR + lambda_CL * (L_aCL + L_uCL)
```

**Why**

- BPR is the main ranking loss
- CL is a small auxiliary regularizer that pulls local/global views closer

**Design**

- stop-gradient on the KG side
- projection head only on the local side

---

## Slide 28 — Dataset

**Amazon Books review subset**

| Statistic | Value |
|---|---|
| Users | 905 |
| Items | 1,399 |
| Interactions | 22,265 |
| KG edges | 3,370 |
| Aspects | 2,098 |
| Average KG edges / item | 2.4 |

---

## Slide 29 — Experimental Setup

**Training Setup**

| Hyperparameter | Value |
|---|---|
| Embedding dimension | 128 |
| Aspect slots | 4 |
| Rationale temperature | 0.5 |
| Fusion-gate bias | +5 |
| Contrastive weight | 0.005 |
| Optimizer | Adam |
---

## Slide 30 — Main Results I

**Top-20**

| Model | NDCG@20 | HR@20 | Recall@20 | MAP@20 |
|---|---|---|---|---|
| MCCLK | 0.1067 | 0.4530 | 0.1720 | 0.0497 |
| KGCL | 0.1073 | 0.4696 | 0.1827 | 0.0479 |
| KGAT | 0.1079 | 0.4773 | 0.1807 | 0.0491 |
| KGRec | 0.1095 | 0.4729 | 0.1834 | 0.0500 |
| LightGCN | 0.1179 | 0.4917 | 0.1937 | 0.0555 |
| **RA-GARK** | **0.1243** | **0.4972** | **0.2020** | **0.0594** |

## Slide 31 — Main Results II

**Top-10**

| Model | NDCG@10 | HR@10 | Recall@10 | MAP@10 |
|---|---|---|---|---|
| MCCLK | 0.0804 | 0.3182 | 0.1047 | 0.0416 |
| KGCL | 0.0809 | 0.3260 | 0.1096 | 0.0410 |
| KGAT | 0.0786 | 0.3215 | 0.1102 | 0.0388 |
| KGRec | 0.0874 | 0.3249 | 0.1155 | 0.0465 |
| LightGCN | 0.0908 | 0.3436 | 0.1201 | 0.0483 |
| **RA-GARK** | **0.0966** | **0.3558** | **0.1265** | **0.0520** |

## Slide 32 — Ablation Results I

| Model | NDCG@20 | MAP@20 |
|---|---|---|
| RA-GARK (full) | 0.1243 | 0.0594 |
| w/o softmax head | 0.1005 | 0.0451 |
| w/o KG-SVD init | 0.1171 | 0.0545 |
| w/o fusion-gate bias | 0.1194 | 0.0555 |
| w/o MLP gate | 0.1180 | 0.0552 |

## Slide 33 — Ablation Results II

| Model | NDCG@20 | MAP@20 |
|---|---|---|
| w/o user CL | 0.1192 | 0.0563 |
| w/o aspect CL | 0.1200 | 0.0570 |
| w/o rationale-enabled selection | 0.1213 | 0.0568 |
| w/o global view | 0.1219 | 0.0575 |

## Slide 34 — Case Study

**圖片**

`thesis/img/case_study_heatmap.png`

---

## Slide 35 — Conclusion & Future Work

**Conclusion**

When the KG is unreliable, what the architecture needs is not a better KG aggregator but a structural switch that can opt the KG out.

**Contributions**

- gateable KG side channel
- KG-SVD initialization
- softmax rationale masking
- local-biased fusion gate

**Future work**

Future work will test on denser KG benchmarks and study when user-level rationale differences emerge.

---

## Slide 36 — Thank You

**Thank you for listening**
