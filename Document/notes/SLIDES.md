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
| `thesis/img/case_study_heatmap.png` | Slide 31 |

## Slide 1 — Title
**RA-GARK**
Product Recommendation via Rationale-Aware Gating over Sparse Review-Aspect Knowledge Graphs
基於理由感知門控與稀疏評論面向知識圖譜之產品推薦
KG-aware Recommendation · Sparse KG · Rationale-aware Gating · Graceful Degradation
Main idea: KG should be a gateable side channel, not a mandatory scoring path.

## Slide 2 — Roadmap
30 分鐘配置
- Introduction
- Related Work
- Methodology
- Experiments
- Conclusion & future work
Focus: Methodology is the main part.

## Slide 3 — Motivation
Sparse KG breaks KG-aware recommendation.
MCCLK 0.1067, KGCL 0.1073, KGAT 0.1079, KGRec 0.1095.
Pure LightGCN: 0.1179.
Key point: every KG-aware baseline loses to pure LightGCN on this sparse KG.

## Slide 4 — Why Sparse KG
- Review-derived KGs are only as dense as user mentions.
- Cold-start and emerging domains rarely have curated KGs.
- Aggressive KG completion adds noise and still needs seed signal.
- Sparse KG is the practical default.
- Robustness matters more than peak performance.
- Unreliable KG deserves dedicated modeling.
- This thesis targets sparse KG.

## Slide 5 — Design Challenge
- KG embeddings enter message passing directly.
- Prior methods assume KG is always useful.
- That assumption fails when KG is sparse.
- LightGCN uses only user-item interactions.
- No KG contamination.
- Strong safe default.
- Our response: route KG through a side channel.

## Slide 6 — Research Question
How can a recommender use KG when it helps, but avoid contamination when KG is sparse or unreliable?
RA-GARK answer: KG should be a gateable side channel.

## Slide 7 — Related Work I
- LightGCN is our local-view anchor.
- KGAT is the deep-fusion baseline.
- LightGCN uses only interactions.
- KGAT propagates KG entities directly.
- RA-GARK keeps LightGCN local and isolates KG globally.

## Slide 8 — Related Work II
- KGCL and MCCLK align multiple views.
- They assume KG structure remains informative.
- Sparse or perturbed KG weakens supervision.
- Alignment can become noise-dominated.
- RA-GARK avoids forcing the KG into every view.

## Slide 9 — Related Work III
- KGRec selects KG edges with dropout and CL.
- RA-GARK selects latent aspect slots.
- KGRec fuses inside KGAT-style propagation.
- RA-GARK fuses late through a side channel.
- KGRec trusts edges; RA-GARK can suppress the whole KG.

## Slide 10 — Related Work IV
- Highway Networks bias toward identity paths.
- MMoE/PLE gate over expert towers.
- SGL/DCCF align views from the same graph.
- Gap: no bias-initialized fusion gate for sparse KG.

## Slide 11 — Design Principle
- Separate local and global views.
- Fuse late.
- Bias the gate toward LightGCN at initialization.
- Let KG open only when useful.

## Slide 12 — Overview
圖片：`thesis/img/architecture.png`
Local View: `u_loc`, `i_loc`.
Global View: `u_glo`, `i_glo`.
Fusion Gate: `u_final`, `i_final`.
Training Loss: ranking objective.

## Slide 13 — Problem Setup
- Implicit top-K recommendation.
- Rank unseen items for each user.
- Train with positive and sampled negatives.
- Score: `<u_final, i_final>`.
- Fusion: local and global embeddings are mixed by learned gates.

## Slide 14 — Local View
- Pure LightGCN.
- No KG branch.
- No nonlinear transform.
- Safe fallback path.

## Slide 15 — Local Propagation
- User-item bipartite graph.
- Training interactions only.
- LightGCN propagation with `K=2` and layer-wise averaging.
- Output: `u_loc`, `i_loc`.

## Slide 16 — Global View
- Raw review-aspect KG is sparse.
- Direct propagation is fragile.
- Latent aspect slots give a compact representation.
- Each item gets `A=4` slots of dimension `128`.

## Slide 17 — KG-SVD Step 1
- Build item-aspect matrix `M`.
- Weight aspects by IDF.
- Downweight generic aspects.
- Preserve discriminative ones.
- Better starting geometry for KG.

## Slide 18 — KG-SVD Step 2
圖片：`thesis/img/kg_svd.png`
Truncated SVD initializes aspect-slot embeddings.
Reshape the factors into `4×128` slots per item.
This preserves aspect co-occurrence before training.

## Slide 19 — KG-SVD Ablation
Full model: NDCG@20 0.1243, MAP@20 0.0594.
w/o KG-SVD init: NDCG@20 0.1171, MAP@20 0.0545.
Observation: KG-SVD preserves the initial semantic geometry.

## Slide 20 — Softmax Masking
- Different users care about different item aspects.
- `logit_k` comes from `[u_glo || aspect_slot]`.
- softmax over slots gives weights `w_k`.
- The item global vector is a weighted sum of slots.

## Slide 21 — Softmax vs Sigmoid
- Sigmoid treats slots independently.
- Softmax makes slots compete under fixed mass.
- In RA-GARK, that competition also controls magnitude.
- This matches the throttled KG channel.

## Slide 22 — Softmax Ablation
圖片：`thesis/img/sensitivity_2x2.png`
Full model: NDCG@20 0.1243, MAP@20 0.0594.
w/o softmax head: NDCG@20 0.1005, MAP@20 0.0451.
Observation: softmax is the thesis-normalized rationale operator.

## Slide 23 — Fusion Gate
圖片：`thesis/img/gate.png`
`alpha_u` and `alpha_i` are sigmoid gates from local+global embeddings.
Final embeddings mix local and global views at scoring time.
This is the architectural switch that lets KG opt out.

## Slide 24 — Gate Bias
- Final gate bias is `+5`.
- Initial alpha is about `0.993`.
- The model starts almost as LightGCN.
- KG opens only when useful.

## Slide 25 — Graceful Degradation
- If KG is not useful, RA-GARK falls back to LightGCN.
- w/o fusion-gate bias: NDCG@20 0.1194, MAP@20 0.0555.
- Bias initialization is part of the architecture.
- It is not a tuning trick.

## Slide 26 — Contrastive Regularization
- Auxiliary only.
- Loss: `L_BPR + lambda_CL(L_aCL + L_uCL)`.
- Small weight, stop-gradient on KG side.
- It aligns views without becoming the main fusion path.

## Slide 27 — Training Setup
- Dataset: 905 users, 1,399 items.
- 22,265 interactions, 3,370 KG edges, 2,098 aspects.
- Adam, lr `1e-3`, batch size `128`.
- 80 epochs with early stopping.

## Slide 28 — Evaluation Setup
- Full ranking evaluation.
- Exclude training interactions.
- Metrics: HR, Precision, Recall, F1, MAP, NDCG@20.
- About 1.5 seconds per epoch; comparable to KGRec.

## Slide 29 — Main Results
- RA-GARK is best at Top-20 and Top-10.
- Top-20: NDCG `0.1243` vs LightGCN `0.1179`.
- Top-10: NDCG `0.0966` vs LightGCN `0.0908`.
- The gain is consistent across ranking metrics.

## Slide 30 — Ablation Summary
- Largest drop: w/o softmax head.
- KG-SVD, gate bias, and MLP gate all matter.
- Removing user/aspect CL hurts moderately.
- Removing the global view still hurts, but less than removing the core architecture.

## Slide 31 — Case Study and Takeaways
圖片：`thesis/img/case_study_heatmap.png`
Different items activate different aspect slots.
Rationale masking gives interpretability.
The model shows which slot is used, not only the score.

## Slide 32 — Conclusion
- KG is treated as a gateable side channel.
- Contributions: KG-SVD, softmax rationale masking, local-biased fusion gate.
- The design favors sparse, unreliable KG.
- Dense-KG settings may still favor deep fusion.
