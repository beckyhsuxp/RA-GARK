# RA-GARK 論文 Outline

對應檔案結構：`Document/thesis/Sections/*.tex`
slide 來源：`Document/notes/SLIDES.md`（24 張）
**論文結構：5 章**（原 Ch3 Architecture + Ch4 Methodology 已合併為新 Ch3 Methodology）
**雙語版本**：英文為主文（`main.tex` → `main.pdf`），中文為預覽（`main_zh_preview.tex` → `main_zh_preview.pdf`）
**撰寫流程**：先中文 → 確認 → 再寫英文 → commit & push

---

## 估計總頁數：40–55 頁（不含參考文獻 / 附錄）

| 章 | 預估頁數 | 來源 slide | 英文狀態 | 中文狀態 |
|---|---|---|---|---|
| 1 Introduction | 4–6 | 1–4 | ✓ 完成 | ✓ 完成 |
| 2 Related Work | 8–10 | 5–11 | ✓ 完成 | ✓ 完成 |
| 3 Methodology | 15–20 | 12–19 | ✓ 完成（含詳細子節） | ✓ 完成（含詳細子節） |
| 4 Experiments | 10–18 | 20–22（+ 補實驗）| ✗ 骨架 | ✗ 待建 |
| 5 Conclusion & Future Work | 4–5 | 23, 24 | ✗ 骨架 | ✗ 待建 |

---

## 0. Front Matter

| 檔案 | 內容 | 狀態 |
|---|---|---|
| `0.1.Acknowledgement.tex` | 致謝（不超過一頁）| ✗ 待寫 |
| `0.2.Abstract_chinese.tex` | 中文摘要 + 5–7 個關鍵詞 | ✓ 完成（已依老師要求改 gap → 方法 → 結果敘述弧） |
| `0.3.Abstract.tex` | 英文摘要（內容對應中文） | ✓ 完成 |

**敘述弧（已落實）**：KG 研究蓬勃 BUT 稀疏 KG 探討少 → 我們解決什麼 → 動態權重 adapt UI+KG → 稀疏下結果。

**Keywords**：knowledge graph–aware recommendation、graceful degradation、rationale-aware learning、fusion gate、sparse knowledge graph、attention normalization、collaborative filtering

---

## 1. Introduction（Slides 1–4）— ✓ 完成

檔案：`1.Introduction.tex` / `1.Introduction_chinese.tex`

| 節 | 內容 | 對應 slide |
|---|---|---|
| 1.1 Background | GNN-CF（LightGCN）+ KG-aware（KGAT 系列）| Slide 2 |
| 1.2 Empirical Anomaly under Sparse KG | 稀疏 KG 下 KG-aware 全敗給 LightGCN（含表） | Slide 2 |
| 1.3 Sparse KG as a Real-World Default | review-derived / cold-start / privacy / KG completion | Slide 3 |
| 1.4 Research Questions | RQ1 診斷 + RQ2 處方 | Slide 4 |
| 1.5 Contributions | 三項設計 + 各搭消融：Softmax (−15.5%) / Bias init (−4.5%) / KG-SVD (−5.9%) | Slide 4 |
| 1.6 Thesis Organization | 章節導讀 | — |

**敘事弧**：主流在我們設定下失敗 → 失敗共同原因是 KG 內嵌計分管線、無結構性開關可關閉 → 提出可被閘控的側通道架構。

---

## 2. Related Work（Slides 5–11）— ✓ 完成

檔案：`2.Relatedwork.tex` / `2.Relatedwork_chinese.tex`

| 節 | 內容 | 對應 slide |
|---|---|---|
| 2.1 Foundations | LightGCN（CF 基底）+ KGAT（深度融合範式） | Slide 5 |
| 2.2 Contrastive KG Methods | KGCL / MCCLK | Slide 6 |
| 2.3 Rationale-aware: KGRec | KGRec edge-level rationale + dropout | Slide 7 |
| 2.4 Rationale Paradigm: Shared vs Diverged | 範式承襲與信任前提分歧（4 維對比表） | Slide 8 |
| 2.5 Gating for Heterogeneous View Fusion | Highway / MMoE / PLE 先例 | Slide 9 |
| 2.6 Gating Gap in KG-aware Recommendation | KGAT/KGCL/MCCLK/KGRec/CKAN/MKR/KGIN 皆無 fusion gate | Slide 10 |
| 2.7 Attention Normalization | DIN / NAIS / AFM 對 sigmoid vs softmax 的選擇 | — |
| 2.8 Positioning Summary | 比較表（Local agg / KG 整合 / Rationale / NDCG）| Slide 11 |

---

## 3. Methodology（Slides 12–19）— ✓ 完成（合併原 Ch3+Ch4）

檔案：`3.Methodology.tex` / `3.Methodology_chinese.tex`

**注**：原 outline 之「Ch3 Architecture」「Ch4 Methodology」已合併成一章，前半 §3.1–§3.4 為架構概覽，後半 §3.5–§3.10 為模組詳述。

| 節 | 子節 | 內容 |
|---|---|---|
| 3.1 Design Principle | — | KG 為可被閘控之旁路通道；三項架構後果 |
| 3.2 Architectural Overview | — | 雙視角泳道圖（`img/architecture.png`）+ 兩項結構性質點明 |
| 3.3 Notation | — | 完整符號表 |
| 3.4 Problem Formulation | — | 評分函數 + 融合公式 + BPR + 對比總損失 |
| 3.5 Local View: LightGCN | 動機 / 圖建構 / 傳播 / 層平均輸出 | K=2、純 CF anchor |
| 3.6 Global View: KG-SVD Initialization | 槽位定義 / 共現 / IDF / SVD / reshape / 演算法 / 效益 | 含**圖 (b) placeholder** + 消融 −5.9% |
| 3.7 Global View: Softmax Rationale Masking | 動機 / 注意力 / softmax / 加權 / vs sigmoid | 含**圖 (c) placeholder** + 消融 −15.5% |
| 3.8 Local-Biased Fusion Gate | 雙側 / 結構 / 偏置 +5 / 優雅退化 / 為何 +5 不 +∞ | 含**圖 (d) placeholder** + 消融 −4.5% |
| 3.9 Cross-View Contrastive Regularization | 角色 / aCL / uCL / stop-grad / 權重 | λ_CL = 0.005 |
| 3.10 Training and Complexity | 總目標 / 最佳化 / 推論 / 複雜度表 | Adam, BS=128, ES patience=10 |

**待替換之 figure placeholder**：
- (b) `img/kgsvd_init.png` — KG-SVD 初始化流程
- (c) `img/softmax_attention.png` — Softmax 合理化遮罩 forward
- (d) `img/fusion_gate.png` — 融合閘訊號流

**已完成的圖**：(a) `img/architecture.png` — 整體架構（已嵌入 §3.2）

---

## 4. Experiments（Slides 20–22 + 補實驗）— ✗ 骨架

檔案：`4.Experiments.tex`（中文版尚未建立）

| 節 | 內容 | 對應 slide | 狀態 |
|---|---|---|---|
| 4.1 Dataset & KG Construction | Amazon Books 子集統計 + KG 建構 pipeline（沿用 [何宜霓 2024]）| Slide 20 | **§4.1 已有 placeholder + 圖 (e)** |
| 4.2 Experimental Setup | baselines 重現 config、評估協議（full-ranking）、metric 定義 | — | **新寫** |
| 4.3 Main Results | 主表（NDCG@20 + 多 metric）+ +13.4% / +5.3% 解讀 | Slide 21 | ✓ 數字齊 |
| 4.4 Ablation Study | softmax/sigmoid、bias=0、random init、純 LightGCN 對比 | Slide 22 | ✓ 數字齊 |
| 4.5 Sensitivity Analysis | τ 掃描、A 槽數、λ_CL、bias 值 | — | **新寫**（從 `tune_weights.py` 結果） |
| 4.6 Seed Stability ⚠ | 多 seed mean ± std；老實討論 variance 來源 | — | **新寫**（含老師關心的 seed 變差問題） |
| 4.7 Case Study | RA-GARK 對特定 user/item 挑出的 aspect 範例 | — | 從 `case_study.py` 補 |
| 4.8 Computational Cost | epoch time vs baselines（~1.5s ≈ KGRec）| Slide 19 | 1 段 |

**Ch3 forward refs 需要 §4 補的 label**：
- `sec:eval-dataset` ✓ 已有
- `sec:ablation-svd` — §4.4 內
- `sec:ablation-softmax` — §4.4 內
- `sec:ablation-gate` — §4.4 內
- `sec:sens-tau` — §4.5 內
- `sec:sens-bias` — §4.5 內
- `sec:eval-cost` — §4.8

**待替換之 figure placeholder**：
- (e) `img/kg_pipeline.png` — KG 建構 pipeline（已在 §4.1 placeholder）

**老師關切的點**：
- 4.5 + 4.6 是 reviewer 一定會問的；別省略
- baseline 的 hparams 怎麼選的？要在 §4.2 寫清楚（公開作者 code、是否重新 tune）

---

## 5. Conclusion and Future Work（Slides 23–24）— ✗ 骨架

檔案：`5.Conclusion.tex`（中文版尚未建立）

| 節 | 內容 | 對應 slide |
|---|---|---|
| 5.1 Conclusion | 主要成果 + 三項設計 + 一句結語 + 價值/重新定位段（架構而非資訊問題；優雅退化為可重用原則，呼應摘要結尾） | Slide 24 |
| 5.2 Methodological Insight: Attention Normalization | sigmoid vs softmax 觀察 + DIN/NAIS/AFM 文獻背景 + **單資料集假說 hedging** | Slide 23 |
| 5.3 Limitations | 單一稀疏資料集、KG 建構非本文貢獻、KG 豐富設定未驗證、**seed sensitivity** | Slide 24 局限 |
| 5.4 Future Work | KG 豐富資料集驗證、其他領域 / 任務、softmax-vs-sigmoid 多資料集複現、**反向稀疏情境（CF 稀疏 + KG 豐富）** | — |

**5.4 反向稀疏情境補充說明**：對稱地，當使用者互動圖稀疏而 KG 稠密時（典型冷啟動 / 新用戶 / 新領域場景），本文的 gate 設計需翻轉：$+5$ 偏置內建的安全預設是 LightGCN，但在該 regime 中 LightGCN 才是不可靠的一側，應改為 KG 側為預設（負偏置），且 KG-SVD 的角色可能由初始化提升為主訊號路徑。此 regime 為 KGAT/KGIN 等深度融合方法的傳統主場，與本文主張的「KG 不可靠時要能脫接」是對稱但獨立的問題，需要重新設計區域視角與 gate 方向性，因此留待後續研究，不在本文 scope。

---

## Appendix

| 節 | 內容 |
|---|---|
| A | 完整超參表（含未報告的 search range） |
| B | 額外消融（如本文未報告但跑過的：weight decay、scheduler、multi-neg、L_kgCL — 全 net-zero/負，可放附錄佐證已嘗試）|
| C | KG 建構 pipeline 細節（ResNet + Qwen-VL / BART + Mistral / aspect 抽取 / 過濾規則）|
| D | 實驗環境、code release URL |

---

## 撰寫優先順序（更新）

1. ✓ §1 Introduction（中英）
2. ✓ §2 Related Work（中英）
3. ✓ §3 Methodology（中英，含 §3.1–§3.10 詳細子節）
4. ✓ 摘要（中英）
5. **下一步**：§4 Experiments
   - 先補 §4.3 + §4.4（主表 + 消融，數字齊）
   - 再寫 §4.1 + §4.2（資料集 + 設定，補 KG pipeline 圖）
   - §4.5 + §4.6 + §4.7（敏感性 + seed + case study — reviewer 必看）
6. §5 Conclusion（呼應 §1 + 加 future work）
7. 致謝（最後）

---

## 章節間 cross-reference 對照（已更新）

| 在某章引用… | 來源章節 | LaTeX label |
|---|---|---|
| §1 contributions 三項設計 | §3.6 / 3.7 / 3.8 | `\ref{sec:kgsvd} \ref{sec:softmax} \ref{sec:gate}` |
| §1 主要結果 +13.4% | §4.3 主表 | `\ref{tab:main}`（待 §4 補） |
| §2 KGRec 對比 | §3.7（softmax 機制）| `\ref{sec:softmax}` |
| §2.5 Highway 移植 | §3.8（bias=+5）| `\ref{sec:gate}` |
| §3 各模組消融 | §4.4 ablation | `\ref{sec:ablation-svd|softmax|gate}` |
| §3 τ 掃描 | §4.5 sensitivity | `\ref{sec:sens-tau|bias}` |
| §3 計算成本 | §4.8 cost | `\ref{sec:eval-cost}` |
| §5 觀察 | §4.4 ablation | `\ref{tab:ablation}` |

每節記得在 `\section` 開頭打 `\label{sec:xxx}`，table/figure 同理。中文版用 `-zh` 後綴。

---

## 編譯指令備忘

```bash
cd Document/thesis

# 英文完整版
latexmk -xelatex main.tex && open main.pdf

# 中文預覽版
latexmk -xelatex main_zh_preview.tex && open main_zh_preview.pdf
```

VS Code LaTeX Workshop 已設定為存檔自動編譯（`.vscode/settings.json`）。
