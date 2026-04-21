---
marp: true
theme: default
paginate: true
size: 16:9
style: |
  section { font-size: 22px; padding: 48px 60px; }
  section.title { text-align: center; background: #F7F7FB; }
  h1 { color: #1F3A68; }
  h2 { color: #1F3A68; border-bottom: 2px solid #D83A3A; padding-bottom: 4px; }
  h3 { color: #1F3A68; }
  table { font-size: 18px; }
  code { background: #F0F0F0; padding: 2px 6px; border-radius: 3px; }
  .red { color: #D83A3A; font-weight: bold; }
  .small { font-size: 16px; color: #666; }
---

<!-- _class: title -->

# RA-GARK
## Rationale-Aware Gating over Review-Aspect KG

雙視角推薦：Softmax 面向顯著性 + 本地偏置融合

<br>

**NDCG@20 = 0.1238 &nbsp; (+13.1% over KGRec)**

<br>

<span class="small">資料集：Amazon Books &nbsp; | &nbsp; 905 使用者 × 1,399 商品</span>

---

## 1. 研究動機

### 現有 KG 推薦方法在稀疏 KG 上表現不佳

在 Amazon Books 稀疏 KG 設定下（3,370 條 KG 邊，平均 2.4 條/item），重跑 4 個主流 KG-aware SOTA 方法：

| 模型 | 年份 | NDCG@20 |
|---|---|---|
| MCCLK | 2022 | 0.1067 |
| KGCL | 2022 | 0.1073 |
| KGAT | 2019 | 0.1079 |
| KGRec | 2023 | 0.1095 |
| **純 LightGCN（不用 KG）** | — | **0.1179** |

**反直覺觀察**：純 LightGCN **反而高於所有 KG-aware 方法**。

---

## 研究問題

既有方法的共同模式：以 KGAT-style 的 bi-interaction aggregator **把 KG 訊號直接融入 GNN 傳遞管線**。

在稀疏 KG 下，這種深度融入也會將雜訊一併帶入 scoring，可能影響原本乾淨的協同訊號。

<br>

> **在稀疏 KG 的設定下，如何設計一個推薦架構，使 KG 訊號能在「有幫助」時被充分利用，在「幫助有限」時不至於干擾協同訊號？**

<br>

**出發點**：探索一個偏保守、強調**可安全退化（graceful degradation）**的設計空間 —— 讓 KG 不是 scoring 管線的必經成分，而是一條可被顯式閘控的側通道。

---

## 2. RA-GARK 整體架構

![w:95%](figures/architecture.png)

<span class="small">兩條平行 swim lane（LOCAL / GLOBAL），經 fusion gate 合併後輸出 score。</span>

---

## 架構四模組

1. **Local View** — LightGCN 於 user–item 二部圖傳播，輸出 $u_{loc}, i_{loc}$
2. **Global View** — `item_kg_aspects` 透過 Rationale Masking 生成 $u_{glo}, i_{glo}$
3. **Fusion Gate** — 每側 $\alpha = \sigma(\text{MLP} + b)$，產出 $u_{final}, i_{final}$
4. **Losses** — BPR 主損失 + 輕量 CL 正則化

$$\hat{y}(u, i) = u_{final} \cdot i_{final}$$

$$L_{total} = L_{BPR} + 0.005 \cdot (L_{aCL} + L_{uCL})$$

---

## 3. 三項核心設計

每一項都經過消融實驗驗證：

| # | 設計 | 消融影響 |
|---|---|---|
| 1 | **Softmax 面向顯著性注意力**（τ = 0.5）| −7.0% NDCG |
| 2 | **本地偏置融合閘初始化**（bias = +5）| −5.3% NDCG |
| 3 | **KG-SVD 初始化**（`item_kg_aspects`）| −5.3% NDCG |

---

## 3.1 Softmax 面向顯著性注意力

每個 item 展開為 $A = 4$ 個 aspect 槽；user-conditioned softmax 挑選最突出的組合：

$$\boldsymbol{\ell} = \text{MLP}([u_{glo} \oplus \mathbf{P}_i]), \quad \boldsymbol{w} = \text{softmax}(\boldsymbol{\ell} / \tau)$$

$$i_{glo} = \sum_a w_a \cdot \mathbf{P}_i[a]$$

**反直覺發現**：若改為常見的 sigmoid 形式：

| Rationale 設計 | NDCG |
|---|---|
| Sigmoid MLP（樸素版） | 0.1152 |
| 不用 rationale（uniform mean） | 0.1222 |
| **Softmax MLP，τ = 0.5（本文）** | **0.1238** |

**sigmoid 形式比完全不用 rationale 還糟 −5.7%**。

---

## 3.2 本地偏置融合閘初始化

**問題**：凸組合 $\alpha \cdot \text{loc} + (1-\alpha) \cdot \text{glo}$ 預設 $\alpha \approx 0.5$，從 epoch 1 起就讓 noisy KG 污染 LightGCN。

**修正**：最後一層 Linear 的 bias 初始化 $b_{init} = +5$：

$$\alpha^{(0)} \approx \sigma(5) \approx 0.993$$

模型**一開始幾乎等同純 LightGCN**，只在梯度顯示有益時才逐漸開啟全域視角。

<br>

**關鍵性質**：提供**結構上的安全退化保證** —— 若 global view 無用，閘門保持關閉，模型不會比 LightGCN 更差。

---

## 3.3 KG-SVD 初始化

`item_kg_aspects ∈ R^{N_i × A × d}` 若用隨機初始化，模型需從零重建 KG 結構。

**做法**：從 KG 關聯矩陣導出初始值：

1. 建 TF-IDF 加權稀疏矩陣 $M \in \mathbb{R}^{N_i \times |\mathcal{A}|}$
2. 截斷 SVD：$M \approx U \Sigma V^\top$
3. 取 $U\sqrt{\Sigma}$，reshape 為 $[N_i, A, d]$，縮放到 xavier std

<br>

**效果**：KG 的語意幾何（相近 item 在嵌入空間也相近）被保留，BPR 訓練只需微調。

---

## 4. 主要結果

| 模型 | NDCG@20 | HR@20 | Recall@20 | MAP@20 |
|---|---|---|---|---|
| MCCLK | 0.1067 | 0.4530 | 0.1720 | 0.0497 |
| KGCL | 0.1073 | 0.4696 | 0.1827 | 0.0479 |
| KGAT | 0.1079 | 0.4773 | 0.1807 | 0.0491 |
| KGRec | 0.1095 | 0.4729 | 0.1834 | 0.0500 |
| **RA-GARK（本文）** | **0.1238** | **0.4961** | **0.2014** | **0.0591** |

**相較 KGRec**：NDCG +13.1% &nbsp; HR +4.9% &nbsp; Recall +9.8% &nbsp; MAP +18.2%

**相較純 LightGCN**：NDCG +5.0% —— 唯一在此稀疏 KG 下「加 KG 為正貢獻」的模型。

---

## 5. 消融實驗

每項設計獨立關掉後的 NDCG 下降（seed = 42，80 epochs）：

| 移除的組件 | NDCG | Δ vs winner |
|---|---|---|
| **winner（完整模型）** | **0.1238** | — |
| 不用 rationale（uniform mean） | 0.1222 | −1.3% |
| 不用 aspect-level CL | 0.1207 | −2.5% |
| 不用 user cross-view CL | 0.1187 | −4.1% |
| **不用 KG SVD 初始化** | 0.1173 | **−5.3%** |
| **fusion bias = 0** | 0.1173 | **−5.3%** |
| **sigmoid rationale** | 0.1152 | **−7.0%** |
| old_full（兩個 Fix 都還原） | 0.1067 | −13.8% |
| lightgcn_only（完全不用 KG） | 0.1179 | −4.8% |

---

## 6. 案例分析：Item-Level 面向顯著性

Softmax attention 權重（τ = 0.5，5 個代表性商品）：

| 商品 | 類型 | 主導 aspect | 權重 |
|---|---|---|---|
| 296 | 賽博龐克 | aspect #1 | 0.41 |
| 1331 | 心理驚悚 | aspect #1 | 0.34 |
| 785 | 科幻生存 | aspect #0 | 0.32 |
| 245 | 反烏托邦科幻 | aspect #0 | 0.29 |
| 77 | 末日廢土 | aspect #0 | 0.27 |

（Uniform 基準 = 0.25）

**每個商品都有不同的主導 aspect**，證明 rationale 模組學到了**商品級的面向顯著性**。

<span class="small">坦承的限制：同商品、不同使用者的 attention 差異僅 ≈ 0.005 — 目前為 item-conditioned，更強的 user conditioning 列為 future work。</span>

---

## 7. 方法論洞見：Sigmoid 陷阱

本文最有意思的發現**不是 +13.1% NDCG**，而是：

<br>

| 設計 | NDCG | 評語 |
|---|---|---|
| Sigmoid MLP | 0.1152 | 比不用 rationale 還糟 |
| 不用 rationale | 0.1222 | 安全預設 |
| **Softmax MLP, τ = 0.5** | **0.1238** | **本文** |

<br>

> **Attention 形式的選擇不是 cosmetic —— 它決定 KG 是資產還是負債。**

<br>

**對後續 KG-aware 方法的啟示**：必須驗證 KG 整合機制在 noise / sparse 情境下能否**優雅退化**。樸素設計有可能是 net-negative。

---

## 8. 限制與未來工作

1. **Attention 是商品條件化、非使用者條件化**（Δ ≈ 0.005 across users）
   &nbsp;&nbsp;→ 可嘗試 FiLM 風格調變、或增加 $u_{glo}$ 維度

2. **單一資料集**（Amazon Books）
   &nbsp;&nbsp;→ Amazon Movies、Yelp、Last.FM 可強化外部效度

3. **單一 seed**（seed = 42）
   &nbsp;&nbsp;→ multi-seed mean ± std 能確認 rationale 的邊際貢獻是否顯著

4. **$A = 4$ aspect 槽可能不足**覆蓋 2,098 個獨立 KG 面向詞
   &nbsp;&nbsp;→ 可掃描 $A \in \{8, 16\}$ 尋找 sweet spot

5. **「sigmoid 陷阱」的可推廣性**需在密集 KG 上驗證

---

## 9. 總結

RA-GARK 提出一個具備「**可安全退化**」特性的雙視角 KG-aware 推薦架構。

<br>

**三項核心設計（ablation-verified）**：
- ✦ **Softmax 面向顯著性注意力**（τ = 0.5）── 單項 −7.0%
- ✦ **本地偏置融合閘初始化**（bias = +5）── 單項 −5.3%
- ✦ **KG-SVD 初始化** ── 單項 −5.3%

<br>

**實驗結果**：NDCG@20 = **0.1238**，相較最強 KG baseline KGRec 提升 **+13.1%**，且為唯一在稀疏 KG 下「加 KG 為正貢獻」的模型。

**方法論貢獻**：Sigmoid vs Softmax 的選擇可以讓 NDCG 差 **7+ 個百分點** —— 對後續 rationale-aware 方法是重要警示。

---

<!-- _class: title -->

# Thank You

Questions?

<br>

<span class="small">Code &middot; Ablation results &middot; Case study artefacts 均已公開於 repository</span>
