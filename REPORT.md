# RA-GARK 報告

**Rationale-Aware Gating Network over Review Aspect-Specific Knowledge Graphs**

---

## 一、研究動機

傳統 KG-based 推薦模型(KGAT、KGCL 等)在 message passing 時對 KG 中所有 aspect 一視同仁,但實際上:

1. **不同使用者在意的面向不同** — 例如同一本小說,A 使用者重視「劇情」,B 使用者重視「文筆」,既有方法無法在 (user, item) pair 層級區分。
2. **CL-based 方法的兩個視角通常太相似**(同一張 graph 隨機 dropout 兩次),導致對比訊號很弱、embedding 容易塌縮。
3. **KG semantic 在訓練後期容易被 BPR 梯度沖刷**,失去原本帶進來的語義意義。

RA-GARK 針對這三個痛點提出解法。

---

## 二、整體架構

### 2.1 雙視角設計

模型把 user 和 item 各自分解成兩個互補的 representation:

| 視角 | 來源 | 角色 |
|---|---|---|
| **Local View** | LightGCN 在 user-item 二部圖上傳播 | 純協同過濾訊號 |
| **Global View** | KG aspect embeddings + Rationale Masking | 語義 / 內容訊號 |

兩個視角透過 **獨立的 fusion gate** 合併,最終 score 就是 dot product:

```
score(u, i) = u_final · i_final
u_final = α_u · u_loc + (1−α_u) · u_glo
i_final = α_i · i_loc + (1−α_i) · i_glo
```

### 2.2 KG Rationale Masking(核心創新 ①)

每個 item 在 KG 上有 `A=4` 個 aspect embedding `[Ni, A, d]`。對於每個 (user, item) pair,用 user 的 global embedding 條件化地對 4 個 aspect 算 attention:

```
weights = sigmoid(MLP([u_glo ; i_aspects]))   # [B, A, 1]
i_glo   = Σ_a weights[a] · i_aspects[a]       # [B, d]
```

→ 同一個 item,不同 user 看到的 `i_glo` 是不一樣的。這就是 **rationale-aware**。

### 2.3 KG SVD 初始化(核心創新 ②)

`item_kg_aspects` 不再用隨機 xavier 初始化,而是:

1. 建 item-aspect 共現矩陣 `[Ni × n_kg_aspects]`
2. TF-IDF 加權
3. Truncated SVD → 取前 `A·d` 個成分
4. reshape 成 `[Ni, A, d]`

→ Global view **從真實 KG 語義開始學**,而不是從噪聲開始學。這個改動把 NDCG 從 0.116 → 0.119。

### 2.4 Aspect-Level CL + Projection Head + Stop Gradient(核心創新 ③)

對比學習設計分三層:

```
L_aCL = (1/A) Σ_a InfoNCE( proj(i_loc), i_aspects[a].detach() )
L_uCL = InfoNCE( proj(u_loc), u_glo.detach() )
```

三個關鍵設計:
- **aspect 層級**:對 4 個 aspect 分別算 InfoNCE,而不是先 attention 聚合 → 保留 aspect 區辨度
- **Projection head**(SimCLR/BYOL 風格):CL 在獨立 MLP 空間,梯度不直接干擾 fusion gate
- **Stop gradient on KG side**:KG 是「老師」,local view 向 KG 對齊,反方向不通

### 2.5 Per-Parameter Learning Rate(核心創新 ④)

Adam 用兩個 param group:
```
base_params         → lr 1e-3
item_kg_aspects     → lr 5e-4
```

→ 防止 BPR 梯度把好不容易 SVD 初始化進去的 KG 語義洗掉。Sweep 過 1e-4 / 5e-4 / 1e-3,5e-4 是最佳值。

---

## 三、Loss

```
L_total = L_BPR
        + 0.005 · (L_aCL + L_uCL)
        + 0.973 · L_reg          (≈ 0,保留為 sanity check)
```

| Loss | 作用 |
|---|---|
| `L_BPR` | 主任務:正樣本分數 > 負樣本分數 |
| `L_aCL` | aspect 層級對比學習:`i_loc` 對齊 4 個 KG aspect |
| `L_uCL` | user 層級對比學習:`u_loc` 對齊 `u_glo` |
| `L_reg` | KG-neighbor MSE,訓練後期接近 0 |

---

## 四、實驗設定

| 項目 | 值 |
|---|---|
| 資料集 | Amazon Books reviews (filtered ≥ 30/20) |
| Users / Items | 905 / 1399 |
| KG | 2098 unique aspects, 1339 items 有 KG 資訊 |
| Aspect 數 / item | A = 4 |
| Embedding dim | d = 128 |
| LightGCN 層數 | K = 2 |
| Batch size | 128 |
| Optimizer | Adam (兩組 lr) |
| Epochs | 80,early stopping patience = 10 |
| Split | user-stratified 70/15/15,seed=42 |
| 評估 | full-ranking,NDCG / HR / Recall / MAP @ K=20 |

**所有 KG-based baseline 都跑在同一個 KG (`df_edges_item_aspect1.csv`)、同一個 split、同一個 evaluator。**

---

## 五、Baseline 比較

四個 KG-based baseline,涵蓋不同流派與時間軸:

| Baseline | 年份 | 類型 | 核心想法 |
|---|---|---|---|
| **KGAT** | KDD 2019 | Attention | KG 上做帶 attention 的 GNN message passing |
| **KGCL** | SIGIR 2022 | CL | 兩個獨立隨機 dropout 視角做 InfoNCE |
| **MCCLK** | SIGIR 2022 | Multi-CL | 三視角(local / semantic / global)交叉對比 |
| **KGRec** | KDD 2023 | Rationale CL | rationale-weighted edge dropout 後 CL |

### 5.1 主結果表

| Model | NDCG | HR | Recall | MAP |
|---|---|---|---|---|
| MCCLK (2022) | 0.1067 | 0.4530 | 0.1720 | 0.0497 |
| KGCL (2022) | 0.1073 | 0.4696 | 0.1827 | 0.0479 |
| KGAT (2019) | 0.1079 | 0.4773 | 0.1807 | 0.0491 |
| KGRec (2023) | 0.1095 | 0.4729 | 0.1834 | 0.0500 |
| **RA-GARK (Ours)** | **0.1196** | **0.4873** | **0.1954** | **0.0563** |

### 5.2 RA-GARK vs 最強 baseline (KGRec)

| Metric | KGRec | RA-GARK | Δ |
|---|---|---|---|
| NDCG@20 | 0.1095 | 0.1196 | **+9.2 %** |
| HR@20 | 0.4729 | 0.4873 | +3.0 % |
| Recall@20 | 0.1834 | 0.1954 | **+6.5 %** |
| MAP@20 | 0.0500 | 0.0563 | **+12.6 %** |

### 5.3 觀察

1. **baseline 之間差距很小**(0.1067 ~ 0.1095,僅 ~3%),代表這個資料集對既有 KG 方法是 saturated 的,單純加 CL 或換 attention 機制不容易再壓榨出顯著改進。
2. **RA-GARK 是質的跳躍**(+9.2% NDCG vs 最強 baseline),不在 baseline 隨機波動範圍內。
3. **KGRec (2023) 確實是最強 baseline**,符合論文時間線:rationale 概念是有效的方向 — 而 RA-GARK 把 rationale 推進到 *aspect 層級* + *user-conditioned*,所以贏得更多。
4. **MAP 提升最大 (+12.6%)** → ranking quality 改善,不只是把對的 item 撈出來,而是排得更前面。

---

## 六、為什麼 RA-GARK 贏得明顯?

對應四個創新點,各自的貢獻:

| 元件 | 解決的問題 | 對 NDCG 的影響 |
|---|---|---|
| KGRationaleMasking | 既有方法對 aspect 平等對待 | 提供 personalization 訊號 |
| KG SVD init | Global view 從噪聲學起 | +0.003 (0.116 → 0.119) |
| Aspect-CL + proj + stop-grad | CL 訊號太弱、視角太相似 | 保留 aspect 區辨度,L_aCL 真的下降 |
| Per-parameter lr | KG 語義被 BPR 沖刷 | sweep 後最佳 5e-4 |

整體說:既有 baseline 用「同一張 KG 跑兩次 dropout」當兩個視角,訊號本來就不夠強;RA-GARK 是「用 KG 預訓練的 global view」對「LightGCN 學的 local view」做對比,**兩視角天生來自不同訊息源**,所以 CL 才真的有用。

---

## 七、簡報用一句話總結

> 「既有 KG 推薦方法把所有 aspect 一視同仁,**RA-GARK** 提出 user-conditioned 的 rationale masking,讓不同使用者在同一個 item 上看到不同的 aspect 權重;搭配 KG SVD 初始化、aspect 層級對比學習、與 KG 專屬 learning rate,在 NDCG@20 比最強 baseline (KGRec, KDD 2023) 高 9.2%,MAP@20 高 12.6%。」
