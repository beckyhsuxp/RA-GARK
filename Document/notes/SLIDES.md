# RA-GARK 口試簡報文字版

> 30 分鐘口試用。這份只保留「每頁投影片上要放的文字」。
> 圖片已標示在對應頁面，直接放你做好的 `img`。

**圖檔對應**

- `thesis/img/architecture.png` -> Slide 12
- `thesis/img/kg_svd.png` -> Slide 18
- `thesis/img/gate.png` -> Slides 23-25
- `thesis/img/sensitivity_2x2.png` -> Slide 22
- `thesis/img/case_study_heatmap.png` -> Slide 31

## Slide 1 — Title
**RA-GARK**
Product Recommendation via Rationale-Aware Gating over Sparse Review-Aspect Knowledge Graphs
基於理由感知門控與稀疏評論面向知識圖譜之產品推薦
KG-aware Recommendation · Sparse KG · Rationale-aware Gating · Graceful Degradation
Main idea: KG should be a gateable side channel, not a mandatory scoring path.

## Slide 2 — Roadmap
**Presentation Roadmap**
- Introduction
- Related Work
- Methodology
- Experiments
- Conclusion & future work
**Main Emphasis**
Methodology is the main part.

## Slide 3 — Motivation
**Main Finding**
Sparse KG breaks KG-aware recommendation.
MCCLK 0.1067, KGCL 0.1073, KGAT 0.1079, KGRec 0.1095.
Pure LightGCN: 0.1179.
**Key Point**
Every KG-aware baseline loses to pure LightGCN on this sparse KG.

## Slide 4 — Why Sparse KG
**Why Sparse KG Is Common**
- Review-derived KGs are only as dense as user mentions.
- Cold-start and emerging domains rarely have curated KGs.
- Aggressive KG completion adds noise and still needs seed signal.
- Sparse KG is the practical default.
- Robustness matters more than peak performance.
**Takeaway**
This thesis targets sparse KG.

## Slide 5 — Design Challenge
**Why Prior Methods Fail**
- KG embeddings enter message passing directly.
- Prior methods assume KG is always useful.
- That assumption fails when KG is sparse.
- LightGCN stays clean because it uses only interactions.
- It avoids KG contamination.
**Our Response**
Route KG through a dedicated side channel.

## Slide 6 — Research Question
**Research Question**
How can a recommender use KG when it helps, but avoid contamination when KG is sparse or unreliable?
**Core Answer**
KG should be a gateable side channel.

## Slide 7 — Related Work I
**CF Backbone vs Deep Fusion**
- LightGCN is the strongest non-KG anchor in our setting.
- KGAT represents deep fusion with direct KG propagation.
- The key contrast is clean CF vs mandatory KG injection.
**Takeaway**
RA-GARK keeps the CF backbone intact.

## Slide 8 — Related Work II
**Contrastive KG Methods**
- KGCL and MCCLK rely on informative KG structure for alignment.
- KGRec learns rationale over KG edges.
- All three still assume the KG channel is basically usable.
**Our Setting**
The whole KG channel may be unreliable.

## Slide 9 — Related Work III
**Design Gap**
- Existing work improves how KG is used.
- Our question is when KG should be used at all.
- Gap: no architecture-level off-switch for sparse KG.
**RA-GARK Position**
This is the design space RA-GARK targets.

## Slide 10 — Methodology Roadmap
**Method in Four Steps**
- Step 1: keep a clean local LightGCN view.
- Step 2: initialize a compact global KG view with KG-SVD.
- Step 3: select user-conditioned aspect slots by softmax masking.
- Step 4: fuse local and global views with a biased gate.

## Slide 11 — Design Principle
**Design Principle**
- KG is a gateable side channel.
- Local and global views stay separate until scoring.
- The gate starts near LightGCN, then opens only if useful.
**Expected Effect**
This gives graceful degradation under sparse KG.

## Slide 12 — Overview
**Architecture Figure**
`thesis/img/architecture.png`
**Pipeline**
Step 1: Local View gives `u_loc`, `i_loc`.
Step 2: Global View gives `u_glo`, `i_glo`.
Step 3: Fusion Gate gives `u_final`, `i_final`.
Step 4: Ranking loss trains the whole model.

## Slide 13 — Problem Setup
**Task**
- Implicit top-K recommendation.
- Rank unseen items for each user.
- Train with positive and sampled negatives.
**Scoring Function**
`<u_final, i_final>`
**Fusion Rule**
Local and global embeddings are mixed by learned gates.

## Slide 14 — Local View
**Local Backbone**
- Pure LightGCN.
- No KG branch.
- No nonlinear transform.
**Role**
Safe fallback path.

## Slide 15 — Local Propagation
**Propagation Graph**
- User-item bipartite graph.
- Training interactions only.
**Propagation Rule**
LightGCN with `K=2` and layer-wise averaging.
**Output**
`u_loc`, `i_loc`.

## Slide 16 — Global View
**Global-View Idea**
- Raw review-aspect KG is sparse.
- Direct propagation is fragile.
- Latent aspect slots give a compact representation.
**Representation**
Each item gets `A=4` slots of dimension `128`.

## Slide 17 — KG-SVD Step 1
**Build the Matrix**
- Build item-aspect matrix `M`.
- Weight aspects by IDF.
- Downweight generic aspects.
- Preserve discriminative ones.
**Goal**
Better starting geometry for KG.

## Slide 18 — KG-SVD Step 2
**Figure**
`thesis/img/kg_svd.png`
**Factorization**
Truncated SVD initializes aspect-slot embeddings.
Reshape the factors into `4×128` slots per item.
**Why It Helps**
This preserves aspect co-occurrence before training.

## Slide 19 — KG-SVD Ablation
**Full Model**
NDCG@20 0.1243, MAP@20 0.0594.
**Without KG-SVD Init**
NDCG@20 0.1171, MAP@20 0.0545.
**Observation**
KG-SVD preserves the initial semantic geometry.

## Slide 20 — Softmax Masking
**Aspect Selection**
- Different users care about different item aspects.
- `logit_k` comes from `[u_glo || aspect_slot]`.
- softmax over slots gives weights `w_k`.
**Output**
The item global vector is a weighted sum of slots.

## Slide 21 — Softmax vs Sigmoid
**Normalization Choice**
- Sigmoid treats slots independently.
- Softmax makes slots compete under fixed mass.
- In RA-GARK, that competition also controls magnitude.
**Why Softmax**
It matches the throttled KG channel.

## Slide 22 — Softmax Ablation
**Figure**
`thesis/img/sensitivity_2x2.png`
**Full Model**
NDCG@20 0.1243, MAP@20 0.0594.
**Without Softmax Head**
NDCG@20 0.1005, MAP@20 0.0451.
**Observation**
softmax is the thesis-normalized rationale operator.

## Slide 23 — Fusion Gate
**Figure**
`thesis/img/gate.png`
**Fusion Gate**
`alpha_u` and `alpha_i` are sigmoid gates from local+global embeddings.
Final embeddings mix local and global views at scoring time.
**Role**
This is the switch that lets KG opt out.

## Slide 24 — Gate Bias
**Bias Initialization**
- Final gate bias is `+5`.
- Initial alpha is about `0.993`.
- The model starts almost as LightGCN.
**Meaning**
KG opens only when useful.

## Slide 25 — Graceful Degradation
**Fallback Behavior**
- If KG is not useful, RA-GARK falls back to LightGCN.
- w/o fusion-gate bias: NDCG@20 0.1194, MAP@20 0.0555.
- Bias initialization is part of the architecture.
**Point**
It is not a tuning trick.

## Slide 26 — Contrastive Regularization
**Regularization Role**
- Auxiliary only.
- Loss: `L_BPR + lambda_CL(L_aCL + L_uCL)`.
- Small weight, stop-gradient on KG side.
**Point**
It aligns views without becoming the main fusion path.

## Slide 27 — Training Setup
**Dataset**
- Dataset: 905 users, 1,399 items.
- 22,265 interactions, 3,370 KG edges, 2,098 aspects.
**Training**
Adam, lr `1e-3`, batch size `128`.
**Stopping**
80 epochs with early stopping.

## Slide 28 — Evaluation Setup
**Evaluation Protocol**
- Full ranking evaluation.
- Exclude training interactions.
- Metrics: HR, Precision, Recall, F1, MAP, NDCG@20.
**Efficiency**
About 1.5 seconds per epoch; comparable to KGRec.

## Slide 29 — Main Results
**Main Result**
- RA-GARK is best at Top-20 and Top-10.
- Top-20: NDCG `0.1243` vs LightGCN `0.1179`.
- Top-10: NDCG `0.0966` vs LightGCN `0.0908`.
**Takeaway**
The gain is consistent across ranking metrics.

## Slide 30 — Ablation Summary
**Ablation Result**
- Largest drop: w/o softmax head.
- KG-SVD, gate bias, and MLP gate all matter.
- Removing user/aspect CL hurts moderately.
**Takeaway**
The core architecture matters most.

## Slide 31 — Case Study and Takeaways
**Figure**
`thesis/img/case_study_heatmap.png`
**Takeaways**
Different items activate different aspect slots.
Rationale masking gives interpretability.
The model shows which slot is used, not only the score.

## Slide 32 — Conclusion
**Core Conclusion**
- KG is treated as a gateable side channel.
- Contributions: KG-SVD, softmax rationale masking, local-biased fusion gate.
- The design favors sparse, unreliable KG.
**Limitation**
Dense-KG settings may still favor deep fusion.
