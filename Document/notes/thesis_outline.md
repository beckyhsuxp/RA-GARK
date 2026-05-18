# RA-GARK 論文 Outline

對應檔案結構：`Document/thesis/Sections/*.tex`
slide 來源：`Document/notes/SLIDES.md`（24 張）
撰寫順序建議：4 → 5 → 3 → 2 → 1 → 6 →（最後）0.2/0.3 摘要

---

## 估計總頁數：50–65 頁（不含參考文獻 / 附錄）

| 章 | 預估頁數 | 來源 slide |
|---|---|---|
| 1 Introduction | 4–6 | 1–4 |
| 2 Related Work | 8–10 | 5–11 |
| 3 Architecture | 3–4 | 12, 13 |
| 4 Methodology | 12–15 | 14–19 |
| 5 Evaluation | 12–18 | 20–22（+ 補實驗） |
| 6 Conclusion / Discussion | 4–5 | 23, 24 |

---

## 0. Front Matter

| 檔案 | 內容 | 狀態 |
|---|---|---|
| `0.1.Acknowledgement.tex` | 致謝（不超過一頁）| ✗ 待寫 |
| `0.2.Abstract_chinese.tex` | 中文摘要 + 5–7 個關鍵詞，~300 字 | ✗ 最後寫 |
| `0.3.Abstract.tex` | 英文摘要（內容對應中文） | ✗ 最後寫 |

**摘要要點（撰寫時）**：稀疏 KG 觀察 → 雙視角架構（local-biased gate）→ 主要數字（0.1238 / +13.1% / +5.0%）→ 三項核心設計。

**Keywords 候選**：knowledge graph–aware recommendation、graceful degradation、rationale-aware、attention normalization、fusion gate、sparse knowledge graph

---

## 1. Introduction（Slides 1–4）

| 節 | 內容 | 對應 slide | 補強 |
|---|---|---|---|
| 1.1 Background | GNN-CF 主流（LightGCN）+ KG-aware 主流（KGAT 系列）| Slide 2 開頭 | 2 段 prose |
| 1.2 Empirical Observation | 稀疏 KG 下 KG-aware 全敗給 LightGCN（含表）| Slide 2 表 | 1 段 + 表 |
| 1.3 Why Sparse KG Matters | review-derived / cold-start / privacy / KG completion 的問題 | Slide 3 | 1–2 段 |
| 1.4 Research Questions | Why 失敗模式 + What 設計原則 | Slide 4 引言 | block quote |
| 1.5 Contributions | 三項設計，**分點列出 + 各搭一句消融數字**：Softmax (−7.0%) / Local-Biased Init (−5.3%) / KG-SVD (−5.3%)| Slide 4 主張 | 重點 — 列點清楚 |
| 1.6 Thesis Organization | 章節導讀 | — | 半頁 |

**敘事弧**：「主流方法在我們這個設定下失敗 → 失敗的共同原因是 KG 必經 → 我們提出可被閘控的側通道架構」。

---

## 2. Related Work（Slides 5–11）

| 節 | 內容 | 對應 slide | 補強 |
|---|---|---|---|
| 2.1 Foundations | LightGCN（協同過濾基底）+ KGAT（深度融合範式）| Slide 5 | 3 段：兩者各 1 段 + 本文如何借鑑/反向 |
| 2.2 Contrastive KG Methods | KGCL / MCCLK | Slide 6 | 各半段 + 本文 CL 角色差異 |
| 2.3 Rationale-aware: KGRec | KGRec edge-level rationale + dropout | Slide 7 | 1 段詳述 |
| 2.4 Rationale Paradigm: Shared vs Diverged | 範式承襲與信任前提分歧（4 維對比表）| Slide 8 | **賣點段落** |
| 2.5 Gating for Heterogeneous View Fusion | Highway / MMoE / PLE 先例 + 異質視角 framing | Slide 9 | 引文齊全 |
| 2.6 Gating Gap in KG-aware Recommendation | KGAT / KGCL / MCCLK / KGRec / CKAN / MKR / KGIN 皆無 fusion gate | Slide 10 | **核心定位段** |
| 2.7 Positioning Summary | 比較表（Local agg / KG 整合 / Rationale / NDCG）| Slide 11 | 1 段 + 大表 |

**寫作風格**：將 slide bullet → 連貫散文。每節結尾以**一句話定位本文與該方法的關係**。

**待補引文（可加進 `ref.bib`）**：
- LightGCN, KGAT, KGCL, MCCLK, KGRec（已有）
- Highway Networks (Srivastava 2015), MMoE (Ma 2018), PLE (Tang 2020)
- DIN, NAIS, AFM
- CKAN, MKR, KGIN（用於 §2.6 點名）
- SGL, DCCF（§2.5 引言）

---

## 3. Architecture（Slides 12–13）

| 節 | 內容 | 對應 slide | 補強 |
|---|---|---|---|
| 3.1 System Overview | 整體 swim-lane 圖 + 四模組總覽 | Slide 12 | **架構圖在 Document/notes/figures/architecture.pdf**，要拷到 thesis/img/ 並寫長 caption |
| 3.2 Notation | 符號表（U / I / R / G_KG / A）| Slide 13 | 完整列表 |
| 3.3 Problem Formulation | 形式化 ranking objective | — | **新寫**：BPR 形式 + ŷ(u,i) 定義 + 評估指標形式定義 |

---

## 4. Methodology（Slides 14–19）

每節結構：**動機 → 數學形式（公式/演算法）→ 設計選擇理由 → 該節對應的消融數字（如有）**

| 節 | 內容 | 對應 slide | 公式 / Algo 重點 |
|---|---|---|---|
| 4.1 Local View | 標準 LightGCN 沿用，協同訊號純淨化 | Slide 14 | E^(l+1) 傳播 + 多層平均 |
| 4.2 Global View — KG-SVD Initialization | IDF 加權 + 截斷 SVD + reshape | Slide 15 | 三步驟，**附 Algorithm 1** + 消融 −5.3% |
| 4.3 Global View — Softmax Aspect-Saliency Attention | MLP logit + softmax(τ=0.5) | Slide 16 | 公式 + τ 掃描表 + 消融 −7.0% |
| 4.4 Local-Biased Fusion Gate | sigmoid(MLP) gate + bias=+5 | Slide 17 | 公式 + bias 推導 α₀ ≈ 0.993 + 消融 −5.3% |
| 4.5 Cross-View Contrastive Regularization | L_aCL + L_uCL + stop-grad + λ=0.005 | Slide 18 | InfoNCE 形式 + 三項保守設計 |
| 4.6 Training & Complexity | Adam, BS=128, ES patience=10 + 各模組複雜度 | Slide 19 | 複雜度表 |

**待補圖**：
- Algorithm box（KG-SVD 初始化、Softmax Attention forward）
- Fusion gate 的 α 訓練動態示意（如有 log）

---

## 5. Evaluation（Slides 20–22 + 補實驗）

| 節 | 內容 | 對應 slide | 狀態 |
|---|---|---|---|
| 5.1 Dataset & KG Construction | Amazon Books 子集統計 + KG 建構 pipeline（沿用 [何宜霓 2024]）| Slide 20 | ✓ |
| 5.2 Experimental Setup | baselines 重現 config、評估協議（full-ranking）、metric 定義 | — | **新寫** |
| 5.3 Main Results | 主表（NDCG@20 + 多 metric）+ +13.1% / +5.0% 解讀 | Slide 21 | ✓（建議擴充其他 K 值）|
| 5.4 Ablation Study | softmax/sigmoid、bias=0、random init、純 LightGCN 對比 | Slide 22 | ✓ |
| 5.5 Sensitivity Analysis | τ 掃描、A 槽數、λ_CL、bias 值 | — | **新寫**（從 `tune_weights.py` 結果來） |
| 5.6 Seed Stability ⚠ | 多 seed mean ± std；老實討論 variance 來源 | — | **新寫**，含老師關心的 seed 變差問題 |
| 5.7 Case Study | RA-GARK 對特定 user/item 挑出的 aspect 範例 | — | 從 `case_study.py` 補 |
| 5.8 Computational Cost | epoch time vs baselines（1.5s ≈ KGRec）| Slide 19 | 1 段 |

**老師關切的點**：
- 5.5 + 5.6 是 reviewer 一定會問的；別省略
- baseline 的 hparams 怎麼選的？要在 §5.2 寫清楚（公開作者 code、是否重新 tune）

---

## 6. Discussion & Conclusion（Slides 23–24）

| 節 | 內容 | 對應 slide | 補強 |
|---|---|---|---|
| 6.1 Methodological Insight: Attention Normalization | sigmoid vs softmax 觀察 + DIN/NAIS/AFM 文獻背景 + **單資料集假說 hedging** | Slide 23 | 維持已收斂的措辭 |
| 6.2 Limitations | 單一稀疏資料集、KG 建構非本文貢獻、KG 豐富設定未驗證、**seed sensitivity** | Slide 24 局限 | 主動點出 |
| 6.3 Future Work | KG 豐富資料集驗證、其他領域 / 任務、softmax-vs-sigmoid 多資料集複現、**反向稀疏情境（CF 稀疏 + KG 豐富）** | — | **新寫** |

**6.3 反向稀疏情境補充說明**：對稱地，當使用者互動圖稀疏而 KG 稠密時（典型冷啟動 / 新用戶 / 新領域場景），本文的 gate 設計需翻轉：$+5$ 偏置內建的安全預設是 LightGCN，但在該 regime 中 LightGCN 才是不可靠的一側，應改為 KG 側為預設（負偏置），且 KG-SVD 的角色可能由初始化提升為主訊號路徑。此 regime 為 KGAT/KGIN 等深度融合方法的傳統主場，與本文主張的「KG 不可靠時要能脫接」是對稱但獨立的問題，需要重新設計區域視角與 gate 方向性，因此留待後續研究，不在本文 scope。
| 6.4 Conclusion | 主要成果 + 三項設計 + 一句結語 | Slide 24 主成果 | 1 頁 |

---

## Appendix

| 節 | 內容 |
|---|---|
| A | 完整超參表（含未報告的 search range） |
| B | 額外消融（如本文未報告但跑過的：weight decay、scheduler、multi-neg、L_kgCL — 全 net-zero/負，可放附錄佐證已嘗試）|
| C | KG 建構 pipeline 細節（ResNet + Qwen-VL / BART + Mistral / aspect 抽取 / 過濾規則）|
| D | 實驗環境、code release URL |

---

## 寫作優先順序建議

1. **§4 Methodology**（公式/演算法最確定，先寫最不會卡）
2. **§5.1 + §5.3 + §5.4**（資料集 + 主表 + 消融 — 數字都齊了）
3. **§3 Architecture**（圖 + notation）
4. **§2 Related Work**（賣點段落要打磨）
5. **§5.5–5.7**（補實驗 — 老師會看的部分，可能要重跑）
6. **§1 Introduction**（最後寫，因為定義「我們證明了什麼」要先確認 §5 的數字）
7. **§6 Conclusion**（呼應 §1）
8. **0.2/0.3 Abstracts**（最後濃縮）

---

## 章節間 cross-reference 對照

| 在某章引用… | 來源章節 | LaTeX label |
|---|---|---|
| §1 contributions 三項設計 | §4.2 / 4.3 / 4.4 | `\ref{sec:kgsvd} \ref{sec:softmax} \ref{sec:gate}` |
| §1 主要結果 +13.1% | §5.3 主表 | `\ref{tab:main}` |
| §2 KGRec 對比 | §4.3（softmax 機制）| `\ref{sec:softmax}` |
| §2.5 Highway 移植 | §4.4（bias=+5）| `\ref{sec:gate}` |
| §6.1 觀察 | §5.4 ablation | `\ref{tab:ablation}` |

每節記得在 `\section` 開頭打 `\label{sec:xxx}`，table/figure 同理。
