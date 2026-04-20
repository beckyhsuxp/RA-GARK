# RA-GARK: Rationale-Aware Gating over Review-Aspect Knowledge Graphs

> 雙視角推薦：Softmax 面向顯著性 + 本地偏置融合

---

## 目錄

- [1. Motivation](#1-motivation)
  - [1.1 研究背景](#11-研究背景)
  - [1.2 稀疏 KG 設定下的觀察](#12-稀疏-kg-設定下的觀察)
  - [1.3 研究問題](#13-研究問題)
  - [1.4 本文貢獻](#14-本文貢獻)
  - [1.5 本文結構](#15-本文結構)
- [2. Related Work](#2-related-work)
  - [2.1 Graph-based Collaborative Filtering](#21-graph-based-collaborative-filtering)
  - [2.2 KG-aware Recommenders](#22-kg-aware-recommenders)
    - [2.2.1 GNN-based KG Integration](#221-gnn-based-kg-integration)
    - [2.2.2 Contrastive KG Methods](#222-contrastive-kg-methods)
    - [2.2.3 Rationale-aware Methods：受 KGRec 啟發並延伸](#223-rationale-aware-methods受-kgrec-啟發並延伸)
  - [2.3 Dual-View / Multi-View Recommenders](#23-dual-view--multi-view-recommenders)
  - [2.4 Attention Normalization 的方法論觀察](#24-attention-normalization-的方法論觀察)
  - [2.5 小結](#25-小結)
- [3. Proposed Method](#3-proposed-method)
  - [3.1 Preliminaries](#31-preliminaries)
  - [3.2 模型概述](#32-模型概述)
  - [3.3 Local View：LightGCN Propagation](#33-local-viewlightgcn-propagation)
  - [3.4 Global View：Multi-Aspect KG Representation](#34-global-viewmulti-aspect-kg-representation)
    - [3.4.1 多面向 Item 表示](#341-多面向-item-表示)
    - [3.4.2 KG-SVD 初始化](#342-kg-svd-初始化)
    - [3.4.3 Softmax Aspect-Saliency Attention (Rationale Masking)](#343-softmax-aspect-saliency-attention-rationale-masking)
  - [3.5 Local-Biased Fusion Gate](#35-local-biased-fusion-gate)
    - [3.5.1 融合公式](#351-融合公式)
    - [3.5.2 Bias 初始化設計](#352-bias-初始化設計)
  - [3.6 最終分數與主要損失](#36-最終分數與主要損失)
  - [3.7 Cross-View Contrastive Regularization](#37-cross-view-contrastive-regularization)
  - [3.8 總損失與最佳化](#38-總損失與最佳化)
  - [3.9 複雜度分析](#39-複雜度分析)
- [4. Experiments（partial）](#4-experimentspartial)
  - [4.1 Dataset and Knowledge Graph Construction](#41-dataset-and-knowledge-graph-construction)

---

# 1. Motivation

## 1.1 研究背景

推薦系統的目標是在大量候選商品中，為每位使用者挑出少數會令其感興趣的項目。近年來基於圖神經網路（Graph Neural Network, GNN）的協同過濾方法逐漸成為主流：NGCF [Wang et al., 2019b] 首次將 GCN 引入推薦，其後 **LightGCN** [He et al., 2020] 展示了即使拿掉非線性轉換與特徵矩陣，僅以 $D^{-1/2} A D^{-1/2} x$ 的線性傳播便能學到具競爭力的 user 與 item 表示。此類方法完全以 user–item 互動為基礎，不依賴外部資訊。

為進一步引入商品的語意結構，**知識圖譜增強推薦（Knowledge Graph-aware Recommendation, KG-aware Rec）** 發展為另一條主線，代表性工作包括 KGAT [Wang et al., 2019a]、KGCL [Yang et al., 2022]、MCCLK [Zou et al., 2022] 與 KGRec [Yang et al., 2023]。這些方法的共同信念是：**若能適當地把 KG 上的商品語意（題材、屬性、面向等）帶入推薦管線，模型既能更準確，也能更可解釋。**

這一主張在多個公開資料集上已被驗證，特別是當 KG 結構豐富、KG 與使用者行為相關性高時，KG-aware 方法通常優於純協同過濾。

## 1.2 稀疏 KG 設定下的觀察

本研究採用的資料集為 Amazon Books 的一個子集，含 905 位使用者、1,399 本書，以及從書籍評論中自動擷取並經過停用詞與高頻詞剪裁的 3,370 條 KG 邊（平均每本書約 2.4 條有效邊）。在此相對稀疏的 KG 設定下，我們以相同的訓練／評估協議（user-stratified 70/15/15 分割、seed = 42、80 epochs、early stopping）重現了四個主流 KG-aware 方法的結果，並同時跑了一個**不使用 KG 的純 LightGCN**作為對照：

| 模型 | 年份 | NDCG@20 |
|---|---|---|
| MCCLK | 2022 | 0.1067 |
| KGCL | 2022 | 0.1073 |
| KGAT | 2019 | 0.1079 |
| KGRec | 2023 | 0.1095 |
| 純 LightGCN（無 KG） | — | **0.1179** |

在我們的實驗設定下，**純 LightGCN 的表現高於我們所測試的所有 KG-aware 方法**。此結果並不必然反映這些方法的本質優劣——它們多數在原論文較大、較稠密的 KG 資料集上均展現了顯著優勢——但這個觀察提示了一個值得關注的現象：

> **當 KG 結構相對稀疏或雜訊較多時，把 KG 訊號直接融入 GNN 傳遞管線的設計選擇，未必能帶來增益。**

分析既有 KG-aware 方法的共同結構，我們注意到多數方法採取類似的整合策略：以 KGAT-style 的 bi-interaction aggregator（或其變體）在 user、item、KG entity 合併的大圖上執行訊息傳遞，再額外搭配對比學習或 rationale 等機制。這類策略的優點是讓 KG 訊號能深度融入表示；但從我們的實驗數據推測，當 KG 訊號本身帶有較多雜訊時，這種深度融入也會將雜訊一併帶入最終 scoring，可能影響原本乾淨的協同訊號。

## 1.3 研究問題

基於上述觀察，本文關注以下問題：

> **在稀疏 KG 的設定下，如何設計一個推薦架構，使 KG 訊號能在「有幫助」時被充分利用，在「幫助有限」時不至於干擾協同訊號？**

我們的出發點並非否定既有 KG-aware 方法，而是希望探索一個**偏保守、更強調「可安全退化」（graceful degradation）**的設計空間：讓 KG 不是 scoring 管線的必經成分，而是一條可被顯式閘控的側通道。

## 1.4 本文貢獻

本文提出 **RA-GARK（Rationale-Aware Gating over Review-Aspect Knowledge Graph）**。其三項核心設計均經過消融實驗驗證：

1. **Softmax 面向顯著性注意力（Softmax Aspect-Saliency Attention）**：將每個 item 的 KG 語意展開為 $A$ 個 aspect 槽，並以**跨 aspect 的 softmax 加溫度縮放**計算注意力。我們觀察到，若改為常見的 sigmoid 形式，在本文資料集上其表現甚至低於完全不使用 rationale 的 uniform mean；此差距高達 −7.0% NDCG。這項發現提示我們，在 rationale-aware 設計中，**注意力歸一化方式的選擇**可能比想像中更具影響力。

2. **本地偏置融合閘初始化（Local-Biased Fusion Gate Init）**：融合閘的 bias 初始化為 +5，使 $\alpha \approx 0.993$ 於訓練起點。模型在起點幾乎等同純 LightGCN，僅在梯度顯示全域視角有益時才逐漸打開閘門。這個「保守起始」設計使 fusion gate 具有**結構上的安全退化性**：若全域視角無法提供有用訊號，閘門保持關閉，模型不會比 LightGCN 更差。消融顯示此設計貢獻 −5.3% NDCG。

3. **KG-SVD 初始化（Knowledge-Graph SVD Initialization）**：以 KG 關聯矩陣的截斷 SVD 作為 `item_kg_aspects` 的初始值，保留 KG 原始的語意幾何。消融顯示此初始化貢獻 −5.3% NDCG。

綜合三項設計，RA-GARK 在 Amazon Books 稀疏 KG 設定下達到 **NDCG@20 = 0.1238**，相較既有最強 KG baseline KGRec 提升 **+13.1%**，並相較純 LightGCN 提升 **+5.0%**。後者較為關鍵——它顯示本文的設計使 KG 訊號在此稀疏設定下從「難以帶來正向貢獻」轉為「能帶來適度正向貢獻」。

除數字提升之外，本文的另一個副產品是一個**方法論層級的觀察**：當 rationale 的語意是「多選項互相競爭」時，注意力歸一化方式（如 sigmoid 對 softmax）可能造成顯著差距。這個觀察未必適用於所有 rationale-aware 方法，但我們認為值得後續研究者在相關設計中進行驗證。

## 1.5 本文結構

第 2 章回顧相關工作並說明本文設計如何受既有方法啟發；第 3 章描述 RA-GARK 的完整架構；第 4 章呈現實驗結果、消融分析與 case study；第 5 章討論方法論層面的觀察與限制；第 6 章總結。

---

# 2. Related Work

本文同時受到「圖協同過濾」、「KG-aware 推薦」、「rationale-aware 方法」、與「雙視角推薦」四條研究線的啟發。以下依序回顧並說明本文在每條線上的定位。

## 2.1 Graph-based Collaborative Filtering

NGCF [Wang et al., 2019b] 將 GCN 引入 CF，後續 LightGCN [He et al., 2020] 證明在推薦場景中，**拿掉非線性激活與特徵轉換後**的線性傳播已足以學到良好表示。此後 SGL [Wu et al., 2021]、NCL [Lin et al., 2022] 等方法在 LightGCN 之上引入自監督對比學習，進一步強化表示的穩定性。此條線的共同特點是**完全以 user–item 互動為基礎，不依賴外部知識**。

**本文的關係**：RA-GARK 的 local view 直接沿用 LightGCN 的結構，不做任何修改。我們選擇以 LightGCN 作為本地訊號的載體，是因為其簡潔性與穩定性使協同訊號在稀疏 KG 設定下相對可靠；在我們的實驗中，純 LightGCN 本身就是一個強基準（NDCG 0.1179），這也是我們在設計 RA-GARK 時特別關注「至少不應比 LightGCN 更差」的原因之一。

## 2.2 KG-aware Recommenders

### 2.2.1 GNN-based KG Integration

**KGAT** [Wang et al., 2019a] 是奠基性的 KG-aware GNN 方法。它將 user–item 二部圖與 KG 合併為 Collaborative Knowledge Graph（CKG），並在 CKG 上以 bi-interaction aggregator 進行傳遞：

$$e_h^{(l+1)} = \text{LeakyReLU}(W_1 (e_h + \text{agg})) + \text{LeakyReLU}(W_2 (e_h \odot \text{agg}))$$

這個聚合器**讓 KG entity 嵌入直接參與 user/item 表示的形成**，使 KG 訊號與協同訊號在同一管線中深度融合。此後 KGRec [Yang et al., 2023] 等多個後繼工作沿用此 aggregator 或其變體。

**本文的做法**：我們選擇**不將 KG 與 user–item 圖合併傳遞**，而是將 KG 語意放在一條獨立的側通道（global view），並以 fusion gate 進行後期合併。這是一個偏保守的設計選擇——在我們的稀疏 KG 設定下，我們希望協同訊號（local view）的梯度不直接受 KG 訊號影響；KG 只在 fusion gate 認定其有用時才納入 scoring。這個選擇與 KGAT 沒有對錯之分，但在 KG 品質波動較大時提供了較強的容錯性。

### 2.2.2 Contrastive KG Methods

**KGCL** [Yang et al., 2022] 對 KG 做結構擾動後與原圖做 InfoNCE 對齊；**MCCLK** [Zou et al., 2022] 建立 collaborative、semantic、structural 三個視角並在其間做多重對比學習。這兩者都展示了對比學習對 KG-aware 推薦的正面效果，特別是在 KG 具有豐富結構的資料集上。

**本文的做法**：我們延續這個「跨視角對比學習」的思路，但在幾個設計細節上做了調整。

首先，本文的 $L_{aCL}$ 與 $L_{uCL}$ 的總權重僅為 0.005，遠低於 KGCL / MCCLK 中 CL 的角色。其次，我們採用 projection head 分離（SimCLR 風格）與 KG 側 stop-gradient，使 CL 的梯度不會反向影響 KG 表示。這些設計的動機是：在稀疏 KG 下，我們希望 CL 僅做輕微的幾何對齊，而不主導 scoring 表示的形成。這種保守取向在我們的實驗設定下使 CL 成為穩定的輔助訊號；我們不認為這是 KG 上做 CL 的唯一合適做法，只是在稀疏 KG 下一個較不易拖累主通道的取捨。

### 2.2.3 Rationale-aware Methods：受 KGRec 啟發並延伸

「Rationale-aware recommendation」這個概念的推廣，很大程度上要歸功於 **KGRec** [Yang et al., 2023]。KGRec 首次在 KG-aware 推薦中明確提出：並非所有 KG 邊對推薦同等重要，應顯式學習哪些訊號才是支持預測的「理由（rationale）」。其具體做法是在 KG 的每條邊上計算 attention-based 重要性分數，並以 rationale-aware Bernoulli dropout 將高分邊隨機移除，強迫模型不過度依賴少數主導邊，再透過原圖與 dropped 圖之間的對比學習讓表示更穩健。

本文延續 KGRec 的核心思路——**KG 訊號需要被顯式挑選，而非無差別注入**——但在幾個面向上做了調整，希望讓這個想法在稀疏 KG 設定下發揮得更穩定：

**(1) Rationale 的作用層次：從 KG 邊 → Item 的 aspect 槽**

KGRec 在 edge-level 挑選 rationale；本文嘗試將 rationale 提升到 item 表示層級。具體做法是為每個 item 保留 $A = 4$ 個 aspect 槽（`item_kg_aspects` $\in \mathbb{R}^{N_i \times A \times d}$），讓 rationale 選擇的對象從「哪些邊參與傳遞」變成「這個 item 的哪個面向最突出」。這項調整的動機是希望 rationale 輸出能夠**直接被觀察與解讀**（例如：此商品對應「末日廢土」面向），便於後續 case study。

**(2) Rationale 機制：從離散 dropout → 可微分 soft attention**

KGRec 的 Bernoulli dropout 使 rationale 在訓練時改變圖結構；本文改為可微分的 softmax attention，使 rationale 權重在推論時仍可觀察與匯出。這個選擇讓我們能在推論階段直接印出 $(u, i)$ 對應的 aspect 權重向量，方便後續 interpretability 分析。

**(3) 注意力歸一化方式**

實作 attention 時，我們觀察到若採用 element-wise sigmoid（與 KGRec 的 edge-wise Bernoulli 形式精神相近），在我們的稀疏 KG 設定下會出現權重趨近均勻的現象，使 rationale 訊號未被有效利用。實驗發現改用跨 aspect 的 softmax 並加入溫度 $\tau = 0.5$ 後，aspect 選擇才變得具有鑑別力。我們認為這項觀察**並非否定 KGRec 的設計**，而是凸顯了「**在不同 rationale 粒度下，適合的歸一化選擇可能不同**」——在 aspect 層級這種「少數選項互相競爭」的場景，softmax 相較於 sigmoid 更能發揮作用。

**(4) Rationale 模組與 base aggregator 的搭配**

KGRec 的 rationale 模組建立於 KGAT-style aggregator 之上，在 CKG 中層層傳遞；本文的 rationale 則與 LightGCN local view 獨立、透過 fusion gate 後期合併。兩種架構各有權衡：KGRec 的整合方式在 KG 豐富時能透過 GNN 傳遞放大 rationale 訊號；本文的側通道設計則在 KG 較稀疏時更易於保持協同訊號的純淨。

綜合來說，本文可視為 **KGRec 所開啟之 rationale-aware 方向在另一個設計空間中的探索**——不同的作用層級（aspect vs edge）、不同的 rationale 機制（soft attention vs dropout）、以及不同的 base aggregator 搭配（LightGCN side-channel vs KGAT-integrated）。在我們的稀疏 KG 設定下，這些調整組合起來帶來了相對 KGRec 的 +13.1% NDCG 提升；但我們並不主張這些調整在所有 KG 推薦情境下都必然優於原始設計，更合理的說法是：**哪種 rationale 設計最適合哪類 KG，是一個值得持續探索的開放問題。**

## 2.3 Dual-View / Multi-View Recommenders

除 KG 相關工作之外，雙視角或多視角架構在推薦系統中亦已有豐富發展。SLRec [Yao et al., 2021]、SGL [Wu et al., 2021]、DCCF [Ren et al., 2023] 等方法透過建立多個視角並以對比學習對齊。此類工作的視角多為**同一圖的不同擾動形式**（edge drop、node drop、random walk 等），在本質上仍處於同一訊號空間中。

**本文的做法**：我們的兩個視角結構上異質——local view 是 user–item 二部圖上的 LightGCN，global view 則是 KG 上的 aspect 表示——兩者並非同一圖的變形，而是不同資訊源的獨立管線。此外，我們除了以 CL 做對齊外，還額外引入**明確的融合閘**，讓兩個視角的權重可以在 scoring 階段動態調整，而非僅透過 CL 隱式對齊。這個設計選擇受到 gated fusion 與 mixture-of-experts 等文獻啟發，但具體應用於稀疏 KG 情境下的雙視角推薦，據我們所知尚未被充分探討。

## 2.4 Attention Normalization 的方法論觀察

本文的一個方法論副產品是觀察到 **attention 公式選擇（sigmoid vs softmax）** 在 rationale-aware 設計中的潛在影響。

既有文獻在此兩種歸一化之間各有採用：DIN [Zhou et al., 2018] 對歷史 item 做 per-target attention（sigmoid-style）；NAIS [He et al., 2018]、AFM [Xiao et al., 2017] 對特徵互動做 softmax 歸一化；KGRec [Yang et al., 2023] 的 rationale 則本質上為 edge-level 的 Bernoulli 決策。不同選擇都有其設計理據，尤其當 attention 的語意為「各元素獨立重要性」時，sigmoid 形式合理且常見。

我們在本文的實驗中觀察到：**在 rationale 的語意為「多選項互相競爭」時**（即每個 item 只會有少數 aspect 主導其被推薦的原因），使用獨立 sigmoid 的 attention 權重會在訓練中趨向飽和均勻狀態，使 rationale 訊號未被有效利用；改用跨 aspect 競爭的 softmax（加溫度縮放）後，rationale 才產生具鑑別力的結果。在本文資料集上，這兩種選擇的差距達 7 個百分點 NDCG。

我們強調，此觀察僅來自本文單一資料集的實驗；是否普遍成立需更多 dataset 驗證。但我們認為這一觀察值得後續 rationale-aware 研究者在其方法中自行檢視。

## 2.5 小結

下表總結 RA-GARK 與本節回顧之主要 baseline 在關鍵設計面向上的異同：

| 面向 | LightGCN | KGAT | KGCL | MCCLK | KGRec | **RA-GARK** |
|---|---|---|---|---|---|---|
| Local aggregator | Pure LightGCN | Bi-interaction | KGAT-style | Multi-view | KGAT-style | **Pure LightGCN** |
| KG 與 scoring 整合方式 | — | 直接傳遞 | 傳遞 + CL | 多視角 CL | 傳遞 + dropout CL | **閘控後期融合** |
| Rationale 層級 | — | — | — | — | Edge-level Bernoulli | **Aspect-level softmax** |
| 設計取向（本文視角） | 無 KG | KG 深度融入 | KG 深度融入 | KG 深度融入 | KG 深度融入 | **KG 可閘控** |

RA-GARK 並非既有方法的「改進版」，而是在「保守 KG 整合」這條較少被探索的設計軸上的一個嘗試。其優勢在我們的稀疏 KG 設定下較為明顯；在 KG 豐富的設定下，KG 深度融入方法仍可能是更優選擇。

---

# 3. Proposed Method

## 3.1 Preliminaries

### 符號

令 $\mathcal{U} = \{u_1, \dots, u_{N_u}\}$ 為使用者集合，$\mathcal{I} = \{i_1, \dots, i_{N_i}\}$ 為商品集合。觀測到的正向互動集合為 $\mathcal{R} = \{(u, i) \mid u \text{ liked } i\}$，以二部圖 $G_{UI} = (\mathcal{U} \cup \mathcal{I}, \mathcal{E}_{UI})$ 表示。

除互動資料外，每個商品連接到一組**評論面向（review aspects）**；以集合 $\mathcal{A}_i \subseteq \mathcal{A}_{\text{all}}$ 表示商品 $i$ 的面向集，形成 KG 二部圖 $G_{KG} = (\mathcal{I}, \mathcal{A}_{\text{all}}, \mathcal{E}_{KG})$，其中 $\mathcal{E}_{KG} = \{(i, a) \mid a \in \mathcal{A}_i\}$。

### 目標

給定 $(u, i) \in \mathcal{U} \times \mathcal{I}$，學習一個評分函數 $\hat{y}(u, i) \in \mathbb{R}$，使真正出現於 $\mathcal{R}$ 的 $(u, i^+)$ 得分高於未觀測的 $(u, i^-)$。

### 主要超參數

嵌入維度 $d = 128$；每個商品的 aspect 槽數 $A = 4$；LightGCN 層數 $K = 2$。

## 3.2 模型概述

RA-GARK 由四個模組組成：

1. **Local View**（§3.3）：user–item 二部圖上的 LightGCN 傳播，輸出 $u_{\text{loc}}, i_{\text{loc}} \in \mathbb{R}^d$。
2. **Global View**（§3.4）：以 `item_kg_aspects` 為基礎的多面向 KG 表示，透過 softmax rationale masking 取得 $u_{\text{glo}}, i_{\text{glo}} \in \mathbb{R}^d$。
3. **Local-Biased Fusion Gate**（§3.5）：針對 user 與 item 各以一個可訓練閘門融合兩視角，取得 $u_{\text{final}}, i_{\text{final}}$。
4. **Losses**（§3.6–3.7）：BPR 排序損失加上小權重的跨視角對比學習正則化。

最終評分為 $\hat{y}(u, i) = u_{\text{final}} \cdot i_{\text{final}}$。

## 3.3 Local View：LightGCN Propagation

Local view 旨在捕捉純協同訊號。我們沿用 LightGCN [He et al., 2020] 的原始結構，不做非線性或特徵轉換的修改。

令 $E^{(0)} = [E_u^{(0)}; E_i^{(0)}] \in \mathbb{R}^{(N_u + N_i) \times d}$ 為 user 與 item 本地嵌入（xavier 初始化）的串接，正規化鄰接矩陣為 $\tilde{A} = D^{-1/2} A D^{-1/2}$。傳遞規則為：

$$E^{(l+1)} = \tilde{A} E^{(l)}, \quad l = 0, 1, \dots, K{-}1$$

最終表示為各層均值：

$$E_{\text{loc}} = \frac{1}{K+1} \sum_{l=0}^{K} E^{(l)}$$

由此取得 $u_{\text{loc}} = E_{\text{loc}}[u]$、$i_{\text{loc}} = E_{\text{loc}}[i]$。

**設計理由**：選擇 LightGCN 而非其變體或 KGAT-style aggregator 的動機，是希望協同訊號在稀疏 KG 情境下保持簡單與穩定；任何 KG 的整合皆在後續 global view 與 fusion gate 中進行。

## 3.4 Global View：Multi-Aspect KG Representation

### 3.4.1 多面向 Item 表示

每個商品 $i$ 的 KG 語意展開為 $A$ 個獨立的 aspect 槽：

$$\mathbf{P}_i \in \mathbb{R}^{A \times d}, \quad \mathbf{P} \in \mathbb{R}^{N_i \times A \times d}$$

將 KG 語意分散於多個 aspect 槽（而非壓縮為單一向量），允許後續 rationale 模組在 scoring 時**動態選擇**使用哪個面向。此設計受 multi-facet interest modelling [Cen et al., 2020] 的啟發，並在本文中專門用於容納 KG 的 review-aspect 語意。

### 3.4.2 KG-SVD 初始化

隨機初始化 $\mathbf{P}$ 會使模型必須從零開始重建 KG 結構；本文採用以下步驟從 KG 直接導出初始值，希望提供更好的訓練起點。

1. **建立加權稀疏矩陣** $M \in \mathbb{R}^{N_i \times |\mathcal{A}_{\text{all}}|}$：

   $$M_{ij} = \mathbf{1}[a_j \in \mathcal{A}_i] \cdot \text{IDF}(a_j), \quad \text{IDF}(a) = \log \frac{N_i}{|\{i : a \in \mathcal{A}_i\}| + 1} + 1$$

2. **截斷 SVD**：$M \approx U \Sigma V^\top$，$U \in \mathbb{R}^{N_i \times k}$，目標維度 $k = A \times d = 512$。

3. **投影與重塑**：取 $E_{\text{init}} = U \sqrt{\Sigma}$，重塑為 $[N_i, A, d]$，並縮放至與 xavier-normal 相同的標準差。

此初始化保留了 KG 的語意幾何（KG 上相近的 item 在嵌入空間也較接近），使 BPR 訓練只需微調而非從零學習結構。消融實驗顯示此步驟貢獻 **−5.3% NDCG**。

### 3.4.3 Softmax Aspect-Saliency Attention (Rationale Masking)

對每個 $(u, i)$ 對，並非所有 $A$ 個 aspect 都同等相關。我們引入一個 user-conditioned attention 機制，從 $\mathbf{P}_i$ 中選出最突出的組合：

$$
\begin{aligned}
\mathbf{u}_{\text{glo}} &= \text{UserGlobalEmb}(u) \in \mathbb{R}^d \\
\boldsymbol{\ell} &= \text{MLP}\bigl([\mathbf{u}_{\text{glo}} \oplus \mathbf{P}_i]\bigr) \in \mathbb{R}^A \\
\boldsymbol{w} &= \text{softmax}(\boldsymbol{\ell} / \tau) \in \mathbb{R}^A \\
\mathbf{i}_{\text{glo}} &= \sum_{a=1}^{A} w_a \cdot \mathbf{P}_i[a] \in \mathbb{R}^d
\end{aligned}
$$

其中 $\oplus$ 為沿最後維度的拼接，$\text{MLP}$ 為兩層前饋網路（$\text{Linear}(2d \to d) \to \text{LeakyReLU} \to \text{Linear}(d \to 1)$）。

此設計涉及兩個關鍵選擇：

**(a) 跨 aspect 的 softmax 歸一化**

在實作上 attention 權重的歸一化有多種選擇。若採用 element-wise sigmoid（$w_a = \sigma(\ell_a)$），各 $w_a$ 互相獨立，語意上對應「每個 aspect 獨立評估重要性」。此形式在許多 attention 應用中是合理且常見的。

然而，在本文資料集的 rationale 場景下，我們觀察到 sigmoid 形式的訓練效果偏弱——所有 $w_a$ 傾向飽和到接近 1，使 $\mathbf{i}_{\text{glo}}$ 近似於未加權的 sum over aspects，甚至稍遜於直接取 uniform mean（見 §4.3 消融）。改用跨 aspect 的 softmax 後，由於 $\sum_a w_a = 1$ 強制 aspects 互相競爭，產生了具鑑別力的加權。我們認為此差異源自 **rationale 的語意為「少數選項互相競爭」**，在此語意下 softmax 的競爭性歸一化較符合場景需求。

**(b) 溫度縮放 $\tau = 0.5$**

MLP 對 $\mathbf{u}_{\text{glo}}$ 的敏感度相對有限，原始 logits $\boldsymbol{\ell}$ 的動態範圍較小，在 $\tau = 1$ 下 softmax 幾乎輸出均勻分佈。通過溫度縮放（$\tau < 1$）放大 logit 差異，可使 attention 產生較清晰的商品級偏好訊號。我們在 $\tau \in \{1.0, 0.5, 0.1, 0.05\}$ 之間進行掃描，發現 $\tau = 0.5$ 於 NDCG 取得最佳平衡：既提供可觀察的 per-item aspect 偏好（見 §4.4 case study），又不因過度銳化而影響學習穩定性。

消融實驗顯示，將此模組由 softmax 改回 sigmoid 會造成 **−7.0% NDCG**，是本文消融中單項影響最大的組件。

## 3.5 Local-Biased Fusion Gate

### 3.5.1 融合公式

給定 local 與 global 兩視角的表示，分別為 user 與 item 各自以獨立的閘控機制融合：

$$
\begin{aligned}
\alpha_u &= \sigma\bigl(\text{MLP}_{\text{gate}}^{(u)}([u_{\text{loc}} \oplus u_{\text{glo}}])\bigr), &
u_{\text{final}} &= \alpha_u \cdot u_{\text{loc}} + (1 - \alpha_u) \cdot u_{\text{glo}} \\
\alpha_i &= \sigma\bigl(\text{MLP}_{\text{gate}}^{(i)}([i_{\text{loc}} \oplus i_{\text{glo}}])\bigr), &
i_{\text{final}} &= \alpha_i \cdot i_{\text{loc}} + (1 - \alpha_i) \cdot i_{\text{glo}}
\end{aligned}
$$

其中 $\text{MLP}_{\text{gate}}$ 結構為 $\text{Linear}(2d \to d) \to \text{Tanh} \to \text{Linear}(d \to 1) \to \text{Sigmoid}$。

### 3.5.2 Bias 初始化設計

融合閘的一個訓練動力學問題是：global view 在訓練初期嵌入尚未成熟，若 $\alpha$ 從 0.5 起步，則訓練起點就有 50% 的 scoring 訊號來自尚未有效訓練的 global 表示；其反向傳遞的梯度可能干擾 local view 的訓練。

我們提出的處理方式是將 $\text{MLP}_{\text{gate}}$ 最後一層 Linear 的 bias 顯式初始化為 $b_{\text{init}} = +5$：

$$\alpha^{(0)} \approx \sigma(b_{\text{init}}) = \sigma(5) \approx 0.993$$

使模型在訓練起點幾乎完全依賴 local view。隨著訓練進行，梯度若顯示「適度打開 gate 可降低 BPR loss」，$\alpha$ 才會逐步下降以納入 global 訊號。

此設計的核心特性是提供**結構上的安全退化保證**：若 global view 最終無法提供有用訊號，$\alpha$ 將保持接近 1，模型退化為接近純 LightGCN 的形式，不至於比 LightGCN 更差。此特性對於我們關注的稀疏 KG 情境尤其重要。

消融實驗顯示此設計貢獻 **−5.3% NDCG**。

## 3.6 最終分數與主要損失

最終推薦分數以內積計算：

$$\hat{y}(u, i) = u_{\text{final}} \cdot i_{\text{final}}$$

主要訓練目標為 BPR 排序損失。對每個訓練樣本 $(u, i^+, i^-)$，其中 $i^+ \in \mathcal{R}_u$、$i^-$ 為隨機採樣的未觀測 item：

$$L_{\text{BPR}} = - \mathbb{E}_{(u, i^+, i^-)} \bigl[\log \sigma\bigl(\hat{y}(u, i^+) - \hat{y}(u, i^-)\bigr)\bigr]$$

## 3.7 Cross-View Contrastive Regularization

為使 local 與 global 兩視角的幾何結構在訓練中保持一致，引入兩個輕量對比學習損失。所有 InfoNCE 計算均在共享的 projection head $g(\cdot) = \text{Linear}(d \to d) \to \text{ReLU} \to \text{Linear}(d \to d)$ 下執行（SimCLR [Chen et al., 2020] 風格），並於 KG 側採用 stop-gradient。

### Aspect-Level CL

對每個 aspect 槽分別計算，拉近 local item 表示與各 aspect 嵌入：

$$L_{aCL} = \frac{1}{A} \sum_{a=1}^{A} \text{InfoNCE}\bigl(g(i_{\text{loc}}^+), \text{stopgrad}(\mathbf{P}_{i^+}[a])\bigr)$$

### User Cross-View CL

拉近 user 的 local 與 global 嵌入：

$$L_{uCL} = \text{InfoNCE}\bigl(g(u_{\text{loc}}), \text{stopgrad}(u_{\text{glo}})\bigr)$$

InfoNCE 溫度 $\tau_{CL} = 0.2$，以 batch 內其他樣本作為負對。

### 設計說明

1. **Stop-gradient 在 KG 側**：避免 CL 反向拖動 KG 嵌入，保留 KG-SVD 初始化的語意幾何。
2. **Projection head 分離**：使 CL 梯度僅作用於投影空間，不直接影響 fusion gate 的參數學習。
3. **小權重（0.005）**：CL 僅做輕量幾何對齊，不主導 scoring 表示的形成。

## 3.8 總損失與最佳化

總損失為：

$$L_{\text{total}} = L_{\text{BPR}} + 0.005 \cdot (L_{aCL} + L_{uCL})$$

所有參數（包括 `item_kg_aspects`）以單一學習率 $\eta = 10^{-3}$ 的 Adam 最佳化器訓練；batch size 128，epochs 80，以 validation NDCG@20 做 early stopping（patience = 10）。

## 3.9 複雜度分析

單次 forward pass 的主要計算組件：

- **LightGCN propagation**：$O(|\mathcal{E}_{UI}| \cdot d \cdot K)$，與標準 LightGCN 同。
- **Rationale Masking**：$O(B \cdot A \cdot d)$，因 $A = 4$ 為常數，此項相對 LightGCN 傳播可忽略。
- **Fusion gate**：$O(B \cdot d^2)$，兩個小 MLP 的前向運算。
- **推論時全排序**：$O(B \cdot N_i \cdot A \cdot d)$，此為評估的主要成本，與既有 KG-aware 方法相當。

在 Nvidia RTX 卡上，RA-GARK 的訓練與推論時間與 KGRec 相近（每 epoch 約 1.5 秒），未因引入 fusion gate 與 rationale masking 而顯著增加計算負擔。

---

# 4. Experiments（partial）

> 以下僅提供 §4.1。剩餘 §4.2 Baselines、§4.3 Main Results、§4.4 Ablation、§4.5 Case Study 待後續補齊。

## 4.1 Dataset and Knowledge Graph Construction

### Dataset

本文實驗採用 Amazon Books 的一個子集，包含 905 位使用者、1,399 本書、22,265 筆正向互動（依 `like = True` 濾後）。依照使用者分層（user-stratified）切分為 70/15/15（train/val/test），隨機種子 seed = 42。

### Knowledge Graph

本文使用的評論面向知識圖譜（Review-Aspect KG）建構管線 **沿用自 [何宜霓等, 2024]**，並非本文貢獻。為完整性與可重現性，我們在此簡述 pipeline 的主要組件：

- **視覺特徵**：以 ResNet + Qwen-VL 為每本書的封面影像生成描述符。
- **文本精煉**：以 LLM（BART / Mistral）將評論文本摘要為 unified summary。
- **面向抽取**：從 summary 抽取 `item–has_aspect–aspect` 關係，建立 item–aspect 的二部 KG。

完整的 KG 建構細節請參考 [學姊的 thesis / paper 引用]。本文重點在於 **給定此 KG 後，如何設計推薦模型**。

### KG 前處理

進入 RA-GARK 訓練管線前，我們對 KG 做以下預處理：

1. 移除頻率最高 2% 的 aspect 詞（這些通常是通用讚美詞如 "good"、"great"、"quality"）。
2. 移除一組手動定義的停用詞（如 "character_development"、"well_written"、"engaging_storylines"）。

過濾後共 **2,098 個獨立 aspect**、**3,370 條有效 KG 邊**（原始 13,905 條，過濾掉 10,535 條），平均每本書約 2.4 條邊。此稀疏度即為第 1 章所描述的實驗設定。

### 評估協議

所有模型（baseline 與 RA-GARK）在同一 split、同一評估器下測試：

- **Test 協議**：全排序（full-ranking）評估，對每位使用者排除訓練集中已互動的 item。
- **指標**：HR、Precision、Recall、F1、MAP、NDCG，皆取 @K=20。
- **報告**：以 validation NDCG@20 做 early stopping（patience = 10），取最佳 epoch 的 test metrics。

---

## 待補章節

- **§4.2** Baselines and Implementation Details
- **§4.3** Main Results（主結果表）
- **§4.4** Ablation Study（消融實驗）
- **§4.5** Interpretability Case Study（可解釋性案例分析）
- **5.** Discussion
  - §5.1 Methodological Insight: The Role of Attention Normalization
  - §5.2 Limitations
- **6.** Conclusion
- **References**

---

*本文件由 ARCHITECTURE.md、ablation_results_paper.csv、case_study.csv 之實驗結果彙整而成。模型程式碼見 train_ragark.py；案例分析見 case_study.py；完整消融腳本見 run_ablations.py。*
