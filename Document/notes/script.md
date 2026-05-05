# RA-GARK 口語逐字稿

---

## Slide 1 — 封面

大家好，我今天要報告的題目是 RA-GARK。

這個工作的核心概念是雙視角推薦——一邊用傳統的 collaborative filtering，一邊用 KG 的語意資訊，然後用一個可以動態調控的閘門把兩邊融合起來。

我們接下來會先講為什麼要做這件事，再講我們怎麼做，最後看結果。

---

## Slide 2 — Motivation

先來講研究動機。

推薦系統這幾年最主流的方向是用 GNN 做 collaborative filtering。其中最具代表性的是 LightGCN——它把 GNN 裡面比較複雜的部分都拿掉，只保留最基本的線性鄰域聚合，結果反而在很多資料集上比更複雜的方法更好。這告訴我們，在推薦場景裡，乾淨的 collaborative signal 其實很重要。

另一條研究線是 KG-aware 推薦。這類方法的想法是：如果我們能把 knowledge graph 上的 item 語意——比如一本書的題材、風格、主題——引入推薦模型，應該可以讓預測更準確，也更可解釋。KGAT、KGCL、MCCLK、KGRec 都是這條線的代表，在 KG 比較豐富的資料集上確實有很好的表現。

但是，我們在自己的實驗設定下發現了一個很有趣的現象。我們用的是 Amazon Books 的子集，knowledge graph 是從書評自動抽取的 aspect，過濾之後每本書平均只有 2.4 條邊，算是相當稀疏。在這個設定下，我們把幾個主流方法都跑了一遍，結果你看——MCCLK、KGCL、KGAT、KGRec，全部都輸給了完全不用 knowledge graph 的純 LightGCN，純 LightGCN 的 NDCG@20 是 0.1179，贏過了所有 KG-aware 方法。

這個結果並不是說這些方法本身不好——它們在原論文的資料集上都有很強的表現——但這個觀察提示了一個問題：當 KG 比較稀疏、雜訊比較多的時候，把 KG 訊號直接融進 pipeline，反而可能把雜訊一起帶進去，拖累了原本乾淨的 collaborative signal。

在進入研究問題之前，我想先回答一個讀者很可能會問的問題——既然 KG 不夠好，為什麼不直接去做 KG completion 把它補完整？或者乾脆找一個 KG 豐富的場景來研究？

答案是，稀疏 KG 其實是真實世界的常態而非例外。我們這個資料集是從用戶評論抽取的 review-derived KG，邊密度天生受限於用戶寫了什麼題材；新興領域、小眾類別、利基商品的 KG 必然稀疏，因為沒有 Freebase 或 Wikidata 這類成熟的知識來源可以倚賴；醫療、金融這類隱私敏感領域，能用的關係訊號也會被刻意限縮。至於 KG completion，它本身就會引入新的雜訊，而且通常需要種子訊號才能訓練，並不是無痛的替代方案。

所以這個工作的定位是——不去解決「KG 太少」的問題，而是去解決「當 KG 不可信時，模型該怎麼穩健」這個問題。從實用角度看，模型在 KG 不可信時的穩健性，其實比 KG 豐富時的峰值更具實用價值。

---

## Slide 3 — Research Question

這個觀察帶出了兩個問題。

第一個是「為什麼」——為什麼 KG-aware 方法在稀疏 knowledge graph 上反而輸給了純 LightGCN？它們的失敗模式是什麼？

第二個是「怎麼辦」——什麼樣的設計原則，能讓 KG 訊號在有用的時候被充分利用、在充滿雜訊的時候不去汙染 collaborative filtering？

我們的主張是：knowledge graph 不應該是 scoring pipeline 裡的必經成分——就像 KGAT 那樣深度融入——而應該是一條可以被明確閘控的 side channel。

這個主張具體落地在三個設計上：softmax aspect 挑選、KG-SVD 初始化、local-biased fusion gate。

我們接下來會說明這三個設計是怎麼做的，並且用實驗結果來說明，這個原則確實能讓 KG 整合在稀疏設定下從原本的負效益，翻轉成 13.1% 的提升。

---

## Slide 4 — Related Work: LightGCN

在進入我們的方法之前，先快速回顧一下相關工作。

第一個是 LightGCN，He 等人 2020 年的工作，也是我們的 local view 的直接基礎。它的核心貢獻是證明在推薦場景裡，把 GNN 的非線性激活和特徵轉換都拿掉，只保留線性鄰域聚合，最後取各層的均值當作最終表示，就可以得到很好的結果。這個簡潔的設計在多個資料集上超越了更複雜的 NGCF，後來 SGL、NCL 這些方法也是在 LightGCN 上面疊加 contrastive learning。

我們的做法是直接沿用標準 LightGCN，傳播兩層，不做任何修改。我們有意讓 local view 保持 collaborative signal 的純淨，任何跟 knowledge graph 相關的操作都留到後面的 global view 和 fusion gate 來處理。選 LightGCN 的原因也很直接——它在我們的設定下就已經達到 0.1179，是一個很強的基準，這也是我們設計上一直強調「至少不能比 LightGCN 更差」的原因。

---

## Slide 5 — Related Work: KGAT

第二個是 KGAT，Wang 等人 2019 年的奠基性工作。

KGAT 的做法是把 user–item graph 跟 knowledge graph 合併成一個大圖，然後在這個大圖上做多層的訊息傳遞。KG entity embedding 會直接參與 user/item 表示的形成，所以 KG 語意是深度融入到 collaborative signal 裡面的。在 KG 豐富的資料集上，這個設計確實很有效。

但我們的問題就在這裡。當 KG 比較稀疏、雜訊比較多的時候，這種深度融合也會把雜訊一併帶進去。在我們的設定下，KGAT 只有 0.1079，輸給了完全不用 knowledge graph 的 LightGCN。

所以我們的選擇是反過來——不合併 KG 跟 user–item graph，讓 knowledge graph 走一條獨立的 side channel，也就是我們的 global view，然後透過 fusion gate 在後期合併。這樣 collaborative signal 的梯度就不會直接受到 knowledge graph 的影響，在 KG 品質比較差的時候容錯性會更強。

---

## Slide 6 — Related Work: KGCL & MCCLK

接下來是兩個 contrastive learning 相關的方法。

KGCL 是 Yang 等人 2022 年的工作，它對 knowledge graph 做隨機的結構擾動，生成增強視角，然後做跨視角的 contrastive learning。在 KG 豐富的資料集上大幅超越了 KGAT。但在 KG 稀疏的情境下，擾動之後剩餘的結構可能不足以提供有效的監督訊號。我們的設定下是 0.1073。

MCCLK 是 Zou 等人 2022 年的工作，更進一步，建立了協同、語意、結構三個視角，在所有視角的兩兩組合之間互相對齊。在多個資料集上達到當時的最高水準。我們的設定下是 0.1067。

我們的做法也有用跨視角 contrastive learning，但角色完全不一樣。我們的 contrastive learning 損失只是作為輔助的幾何對齊，權重非常小，只有 0.005，目的是讓本地和全域兩個 embedding 空間的幾何結構不要差太多。而且我們在 KG 那側做了 stop-gradient，避免 contrastive learning 的梯度去破壞 SVD 初始化保留下來的語意幾何。

---

## Slide 7 — Related Work: KGRec

KGRec 是跟我們最直接相關的工作，Yang 等人 2023 年。

KGRec 首次在 KG-aware 推薦裡明確提出了「理由感知」這個概念——並不是所有 KG edge 對推薦的重要性都一樣，我們應該明確地學習哪些訊號才是支持預測的理由。具體做法是在每條邊上算一個 attention 重要性分數，然後隨機把高分的邊移除，強迫模型不要過度依賴少數幾條主導邊，再透過原圖和刪減圖之間的 contrastive learning 讓表示更穩健。整個架構是建立在 KGAT 的聚合器上面的。在我們的設定下，KGRec 是 0.1095，同樣輸給了 LightGCN。

KGRec 跟我們的關係其實有點微妙——表面上看 RA-GARK 像是 KGRec 的延伸，但底層的設計哲學其實是相反的。我們承襲了 KGRec 的核心觀察「並不是所有 KG 邊同等重要」，但具體的設計選擇反映了**對 KG 信任度的不同前提**。

KGRec 的隱含假設是「有用的邊存在，挑出來即可」。它在 KGAT aggregator 內部用 edge-level 的 Bernoulli dropout 加上 contrastive learning，意思是 KG 整體還是可以信任的，只是某些邊比其他邊更重要——挑出可信的部分繼續用就好。

我們的前提不一樣。在我們稀疏 KG 的設定下，KG 整體可能就是不可信的，所以我們的設計是讓**整條 KG 管線都可以被閘門關掉**。這個前提差異決定了四個具體的設計分歧。

第一，理由挑選的粒度從邊的層級提升到 item 的 aspect slot，每個 item 維護四個 aspect slot，挑選結果可以直接對應「這個 item 最突出的面向是哪幾個」，比較容易解讀。

第二，機制從離散隨機丟棄改成可微分的 softmax attention，方便端對端訓練，推論時也可以匯出權重觀察。

第三，作用位置從 KGAT aggregator 內部移到後期融合的獨立側通道，這樣 collaborative signal 不會被 KG 雜訊滲透。

第四，最關鍵的——對 KG 整體信任度的處理。KGRec 沒有「關掉 KG」的機制，我們有 fusion gate 提供這個結構性開關。

所以 RA-GARK 不是 KGRec 的工程改良，而是基於不同 KG 信任前提的獨立設計。具體的效能對比結果，我們留到實驗章節再看。

---

## Slide 8 — Related Work: Dual-View Recommenders

接下來是雙視角推薦的相關背景。

SGL、DCCF 這類方法都是建立多個視角，然後用 contrastive learning 讓各個視角互相對齊。但這些方法的視角通常是同一張圖的不同擾動形式——比如隨機丟邊、隨機丟節點、隨機遊走——本質上還是在同一個訊號空間裡操作的。

我們的兩個視角則是根本異質的。local view 是在 user–item graph 上跑 LightGCN，代表的是純 collaborative signal；global view 是在 knowledge graph 上做 aspect 表示加理由挑選，代表的是知識語意。這兩者不是同一張圖的變形，而是來自不同資訊源的獨立管線。

正因為異質程度這麼高，單純靠 contrastive learning 隱式對齊是不夠的。我們需要一個明確的 fusion gate，讓模型能夠明確地控制兩邊的混合比例——這也是為什麼我們選擇用閘門而不只是 contrastive learning 來整合兩個視角。

---

## Slide 9 — Related Work: Gating for Heterogeneous View Fusion

這邊我想多說一下我們的 fusion gate 跟既有方法的關係，因為這是這個工作的核心設計選擇。

跟我們最直接相關的兩個先例。第一個是 Highway Networks，Srivastava 在 2015 年提出，原本是為了解決深層網路訓練困難的問題。它的核心是一個 gate，把 transformation 跟 identity skip 加權合起來，關鍵在於 gate 的 bias 初始化為負值，讓網路一開始接近恆等映射，後續訓練再慢慢學要不要做變換。我們的 bias 初始化為正五的設計，本質上就是把這個「保守起點 + 漸進開啟」的概念，從深層網路內部 skip 移植到雙視角融合的場景。

第二個先例是 MMoE 跟 PLE 這類 mixture-of-experts 在多任務推薦的應用。它們也是用 gate 從多個管線中選擇，但兩個重要差異是：那邊的 expert 都是同質的候選，沒有「哪個比較不可信」的問題；而且它們的 gate 沒有特別的初始化偏置，是對稱起點。

至於在 KG-aware 推薦領域本身，我們檢視過的方法都沒有用 fusion gate 作為融合機制。KGAT 是 entity 嵌入直接傳遞，KGCL 跟 MCCLK 用 contrastive learning 做隱式對齊，KGRec 雖然有 rationale dropout 但仍然走 KGAT 風格的 aggregator。CKAN、MKR、KGIN 這些近年比較有代表性的方法分別用的是 attention、cross-feature unit、意圖分解，都不是後期融合 gate。它們共同的隱含假設是「KG 至少不會是負訊號」——但我們在 Slide 2 看到，這個假設在稀疏 KG 下並不成立。

我這邊想很小心地措辭：我們不敢說自己是「第一個」用 gate 做融合的，因為 KG-aware 推薦的文獻很大，很難窮盡。但更可辯護的講法是——在我們檢視過的這些主流 KG-aware 方法裡，沒有一個用 bias-initialized 的 gate 來提供架構層面的安全退化機制，而這個結構選擇正是我們在稀疏 KG 設定下能夠贏過 LightGCN 的關鍵。

至於我們架構裡同時用了 attention 跟 gate，但分層使用——aspect 選擇那一層用 attention，因為四個 slot 是同質候選互相競爭；視角融合那一層用 gate，因為本地和全域是異質管線需要的是「要不要採用」的開關決策——這個分層方式是我們刻意的設計。

---

## Slide 10 — Related Work: Attention Normalization

再來是 attention normalization 方式的問題，這是一個比較細但很重要的方法論觀察。

現有文獻對這個問題其實沒有統一的選擇。DIN 用 sigmoid 對歷史行為做 attention，NAIS 用 softmax 對歷史互動做聚合，AFM 對特徵互動做 softmax 歸一化。這些選擇都有其設計理據，通常是看語意假設。

我們在實驗中觀察到，在 aspect 選擇這個場景，sigmoid 和 softmax 的差距大得出乎意料——sigmoid 拿到 0.1152，softmax 拿到 0.1238，差了 7 個百分點。這遠超一般調參能帶來的差距。

原因我們認為在於語意假設是否吻合。Sigmoid 假設各個 aspect 獨立評估重要性，訓練過程中傾向飽和到接近一，等於沒有做選擇；softmax 強制所有 aspect 的權重加總等於一，aspect 之間互相競爭，才能產生有鑑別力的加權。在「這個 item 最突出的 aspect 是哪幾個」這個問題上，aspect 之間本來就是競爭關係，softmax 更符合這個語意。

我們認為這個觀察有一定的普遍性——在任何理由感知設計裡，如果理由的語意是「少數候選互相競爭勝出」，softmax 都可能是更合適的選擇。這個問題在現有文獻裡尚未被充分討論。

---

## Slide 11 — Related Work: Summary

把我們剛才講的這些方法整理成一張表。

從本地聚合器來看，我們和 LightGCN 一樣，用的是純粹的線性傳播，其他方法都是用 KGAT 的雙互動聚合器或者多視角聚合。

從 KG 整合方式來看，KGAT 是直接傳遞，KGCL 是傳遞加 contrastive learning，MCCLK 是多視角 contrastive learning，KGRec 是傳遞加隨機丟棄，我們是閘控後期融合。

從理由挑選的層級來看，只有 KGRec 和 RA-GARK 有做主動的挑選，KGRec 在邊的層級，我們在 aspect 的層級。

從 NDCG@20 來看，在我們的稀疏設定下，其他方法全部低於 LightGCN 的 0.1179，我們是 0.1238。

我想強調一點：RA-GARK 並不是在說這些既有方法不好，而是在「保守 KG 整合」這個比較少被探索的設計方向上做了一個嘗試。我們的優勢在稀疏 knowledge graph 設定下比較明顯；如果 knowledge graph 很豐富，深度融合方式仍然可能是更好的選擇。

---

## Slide 12 — Model Overview

好，接下來進入我們方法的部分。

RA-GARK 由四個模組組成。

第一個是 local view，就是標準的 LightGCN，在 user–item graph 上做兩層線性傳播，輸出的是純 collaborative signal 的 user 和 item 表示，完全不接觸 knowledge graph。

第二個是 global view，我們為每個 item 維護四個 aspect slot，用 SVD 從 KG 初始化，然後透過 softmax aspect attention，根據每個 user–item pair 的情況動態選出最突出的 aspect 組合，輸出帶有 KG 語意的 user 和 item 表示。

第三個是 local-biased fusion gate，把兩個視角的表示融合起來。融合是本地表示和全域表示的加權和，gate 值決定兩邊的比例。關鍵的設計是 gate 的 bias 初始化為正五，讓訓練起點的 gate 值接近 0.993，幾乎等同純 LightGCN，只有在梯度顯示 global view 有益的時候，gate 才會逐漸打開。

第四個是計分和損失函數，最終分數是兩個最終表示的內積，用 BPR loss 訓練，再加上一個很小的跨視角 contrastive learning 正則化，權重只有 0.005。

這三個核心設計——KG-SVD 初始化、softmax aspect attention、local-biased fusion gate 初始化——都有消融實驗的支撐，等一下會看到。

---

## Slide 13 — Preliminaries

在深入各個模組之前，先定義一下符號。

user 集合有 905 個人，item 集合有 1,399 個 item，觀測到的正向互動總共 22,265 筆，用 user–item graph 來表示。

knowledge graph 是一個 item 跟 aspect 之間的二部圖，邊就是每個 item 連到它對應的 aspect，總共有 2,098 個獨立的 aspect。

我們的目標是學一個評分函數，使得真正有互動的正樣本得分高於未觀測的負樣本。

主要的超參數：embedding 維度 128，aspect slot 數 4，傳播層數 2，學習率 0.001，批次大小 128。

---

## Slide 14 — Local View: LightGCN

local view 沿用標準 LightGCN，傳播兩層，不做任何修改。

這邊沒有什麼特別的設計，就是最乾淨的 LightGCN。我們有意讓 local view 完全不接觸 knowledge graph，目的是保持 collaborative signal 的純淨。所有跟 knowledge graph 相關的操作都在後面的 global view 和 fusion gate 裡面做。

這個選擇的背後邏輯是：在我們的設定下，純 LightGCN 已經是 0.1179，高於所有 KG-aware 方法。所以我們一直強調的設計約束是，RA-GARK 至少不能比 LightGCN 更差。

---

## Slide 15 — Global View: KG-SVD Initialization

global view 這邊，我們為每個 item 維護四個 aspect slot，把 KG 語意展開成多個槽而不是壓縮成一個向量，是為了讓後面的 attention 模組有選擇的空間——每次計分的時候可以根據 user 的偏好動態選出最突出的那幾個 aspect。

初始化的部分，我們用了 KG-SVD。步驟是這樣的：先建一個用 IDF 加權的矩陣，矩陣的每個元素代表 item 有沒有某個 aspect，再乘以那個 aspect 的 IDF。然後對這個矩陣做截斷 SVD，取前面的奇異值和奇異向量，投影之後重塑成 item 數乘以 aspect slot 數乘以 embedding 維度的三維張量，縮放到合理的初始化範圍。

這樣做的好處是，訓練起點就已經保留了 KG 原本的語意結構——在 KG 上比較接近的 item，在 embedding 空間裡也比較近，後面的訓練只需要在這個基礎上微調就好了。

消融實驗顯示，如果改成隨機初始化，NDCG@20 會下降 5.3%。

簡單講一句話就是：我們手上有 item × aspect 的 KG 對照表，直接對它做 SVD 找出主要的主題方向，用這些方向當 `item_kg_aspects` 的起點——這樣模型一開始就知道哪些書在語意上相近，不用從零學起。拿掉會掉 5.3% NDCG。

---

## Slide 16 — Global View: Softmax Aspect-Saliency Attention

接下來是 aspect 選擇的 attention 機制。

對每個 user–item pair，我們先把 user 的 global embedding 和 item 的 aspect 矩陣拼接起來，丟進一個兩層的 feed-forward network，得到每個 aspect 的重要性分數。然後用帶溫度縮放的 softmax，溫度設定為 0.5，算出 attention weight，最後用這個權重對各個 aspect 做加權和，得到 item 的全域表示。

關鍵設計是用 softmax 而不是 sigmoid。如果用 sigmoid，各個 aspect 的權重是獨立算的，訓練過程中傾向飽和到接近一，等於沒有做選擇，退化成取平均。改成 softmax，因為所有權重加總必須等於一，各個 aspect 之間互相競爭，才能產生有鑑別力的加權。這個差距我們在 Slide 10 已經講過，在我們的資料集上達到 7 個百分點，是全文消融裡單項影響最大的。

溫度 0.5 的選擇也很重要。feed-forward network 輸出的分數動態範圍比較小，溫度設為一的話 softmax 輸出幾乎是均勻分佈，放大不了差異。0.5 可以把差異放大，產生比較清晰的 aspect 偏好。我們掃描了幾個溫度值，0.5 在 NDCG 上取得最好的平衡。

---

## Slide 17 — Local-Biased Fusion Gate

融合閘把本地和全域兩個視角的表示融合起來。

做法是：先把 user 的 local 和 global 表示拼接，丟進一個小型 feed-forward network，輸出一個介於零和一之間的 gate 值，然後最終表示等於 gate 值乘以本地表示加上一減 gate 值乘以全域表示。item 那側也有一個獨立的閘門，結構一樣。

最重要的設計是偏置初始化。我們把 feed-forward network 最後一層的偏置直接初始化為正五，讓訓練起點的 gate 值接近 0.993，幾乎完全依賴 local view，等同純 LightGCN。

為什麼要這樣做？因為訓練初期 global view 的 embedding 還沒有學好，如果 gate 從 0.5 起步，就有一半的計分訊號來自不可靠的全域表示，這些梯度可能會干擾到 local view 的訓練。保守起步可以讓 local view 先把 collaborative signal 學好，之後如果梯度顯示打開閘門有助於降低損失，gate 才會慢慢打開。

這個設計更深層的意義是提供了一個結構上的安全退化保證。如果 global view 最終沒辦法提供有用的訊號，gate 就會一直維持在接近一，模型退化為接近純 LightGCN，不至於比 LightGCN 更差。

消融顯示把偏置初始化改為零，NDCG@20 下降 5.3%。

---

## Slide 18 — Cross-View Contrastive Regularization

除了主要的 BPR loss，我們還加了兩個輕量的跨視角 contrastive learning 損失。

第一個是 aspect 層級的 contrastive learning。對每個 aspect slot，計算本地 item 表示跟這個 aspectembedding 的相似度損失，取四個 aspect 的平均。這個損失讓本地 item 表示在幾何上往各個 aspect 的方向靠近。

第二個是 user 跨視角 contrastive learning，把 user 的 local 和 global embedding 互相拉近，讓兩個視角的幾何結構保持一致。

設計上有三個保守的選擇。第一，KG 那側做 stop-gradient，避免 contrastive learning 的梯度去破壞 SVD 初始化保留下來的語意幾何。第二，用一個小的 projection head，讓 contrastive learning 的梯度只作用在投影空間，不直接影響 fusion gate 的參數。第三，這兩個損失的總權重只有 0.005，非常小，只是做輔助的幾何對齊，不主導表示的學習。

總損失就是 BPR loss 加上 0.005 乘以這兩個 contrastive learning 損失的和。

---

## Slide 19 — Training Setup

訓練設定這邊，優化器用 Adam，學習率 0.001，批次大小 128，最多跑 80 個 epoch，用驗證集的 NDCG@20 做早停，耐心值是 10。

複雜度方面，LightGCN 傳播的複雜度跟標準 LightGCN 一樣；aspect attention 因為 aspect slot 數固定是四，這項相對可以忽略；fusion gate 是兩個小 feed-forward network 的運算；推論時全排序是計算成本最高的部分，跟現有的 KG-aware 方法差不多。

實際跑起來，每個 epoch 大概 1.5 秒，跟 KGRec 相近，加了 fusion gate 和 aspect attention 並沒有顯著增加計算負擔。

評估協議是全排序，對每個 user 排除訓練集裡已經互動過的 item，指標用命中率、精確率、召回率、F1、平均精確率和 NDCG，都取前二十。

---

## Slide 20 — Dataset & KG Construction

資料集用的是 Amazon Books 的一個子集，905 個 user，1,399 本書，22,265 筆正向互動，按 user 分層切成訓練七成、驗證一成半、測試一成半。

KG 建構 pipeline 沿用的是學姊何宜霓等 2024 年的工作，不是本文的貢獻，我們在這裡簡單介紹一下。管線分四步：第一步用視覺模型為每本書的封面影像生成描述；第二步用語言模型把書評摘要成統一格式的文本；第三步從摘要裡面抽取 item 跟 aspect 之間的關係，建立二部 knowledge graph；第四步做前處理，移除頻率最高百分之二的 aspect——這些通常是沒有區辨力的通用讚美詞，加上一組手動定義的停用詞。

過濾之後，原本的一萬三千多條邊剩下 3,370 條，刪掉了大約四分之三，有 2,098 個獨立的 aspect，平均每本書 2.4 條邊。這個稀疏度就是我們前面一直說的稀疏 knowledge graph 設定。

---

## Slide 21 — Experimental Results

來看結果。

這張表把所有方法的 NDCG@20 以及跟 KGRec 和 LightGCN 的比較都列出來了。MCCLK 0.1067，KGCL 0.1073，KGAT 0.1079，KGRec 0.1095，純 LightGCN 0.1179，RA-GARK 是 0.1238。

有兩個數字我特別想強調。

第一個是跟 KGRec 相比高了 13.1%。在完全相同的稀疏 knowledge graph 設定下，RA-GARK 顯著優於現有最強的 KG-aware 方法。

第二個我認為更重要，是跟純 LightGCN 相比高了 5%。這個數字的意義是：KG 訊號在這個稀疏的設定下，從「難以帶來正向貢獻」變成了「能帶來適度的正向貢獻」。RA-GARK 成功讓稀疏 knowledge graph 發揮了作用，而不是像其他方法一樣反而拖累了結果。這也回答了我們在 Slide 3 提出的那個問題——正確的設計原則確實能讓 KG 整合從負效益翻轉為正效益。

計算成本方面，每個 epoch 約 1.5 秒，跟 KGRec 相當，沒有顯著增加負擔。

---

## Slide 22 — Ablation Study

消融實驗這邊，我們驗證了三個核心設計各自的貢獻。

完整的 RA-GARK 是 0.1238。

第一個，把 softmax aspect attention 改回 sigmoid，NDCG 掉到 0.1151，下降了 7%。這幾乎等同於說「用了 knowledge graph 反而變差了」，因為 0.1151 比純 LightGCN 的 0.1179 還低。這個結果印證了我們在 Slide 10 講的：在 aspect 選擇的語意下，歸一化方式的選擇至關重要。

第二個，把 local-biased 初始化的偏置改為零，也就是讓 gate 從 0.5 起步，NDCG 掉到 0.1173，下降 5.3%。這個設計的貢獻是讓訓練起點更穩定，讓 global view 的訊號能夠更有效地被利用。

第三個，把 KG-SVD 初始化改成隨機初始化，NDCG 也掉到 0.1173，同樣下降 5.3%。SVD 初始化保留了 KG 語意幾何，提供了一個好的訓練起點。

三個設計缺少任何一個，RA-GARK 都無法超越純 LightGCN 的 0.1179。這說明三者是互補的，需要同時使用才能達到最好的效果。

---

## Slide 23 — Methodological Insight

除了數字上的結果，我們還想再把 Slide 10 的那個觀察更完整地說一遍，但我會在這邊很小心地措辭，避免過度索取。

在我們的實驗設定下，理由感知模組的 attention normalization 方式造成了 7 個百分點的 NDCG 差距，遠超一般調參的影響範圍。

這張表整理了兩種方式的比較。Sigmoid 的語意假設是各元素獨立評估重要性，訓練行為傾向飽和均勻；softmax 的語意假設是少數選項互相競爭勝出，訓練行為產生具鑑別力的加權。

我想強調的是，這個觀察**不是在說 softmax 一定比 sigmoid 好**。DIN 用 sigmoid 對歷史行為做 attention 是完全合理的，因為那個場景的語意是各個歷史行為獨立評估跟目標的相關性。問題在於——在 aspect 選擇這個場景，語意是「少數幾個 aspect 主導了這個 item 被推薦的原因」，這個語意下 softmax 更符合需求。

所以我們的暫定觀察是：歸一化方式應該和理由挑選的語意假設對齊。

但這邊我必須誠實地說——這個結論目前**只有單一資料集的支持**，我不敢把它當成通用的方法論定論。要把它升格為一般性的 methodological claim，至少需要在兩三個不同的資料集、不同的任務上重複驗證。所以本文把這個觀察列為一個**值得後續研究檢驗的假說**，而不是定論。即便如此，7 個百分點這個差距在我們的設定下足夠大，至少在這個範圍內值得被認真看待。

---

## Slide 24 — Conclusion

最後來總結。

主要成果：RA-GARK 在 Amazon Books 稀疏 knowledge graph 設定下達到 NDCG@20 = 0.1238，比 KGRec 高了 13.1%，比純 LightGCN 高了 5%。後者更關鍵，它說明我們的設計讓 KG 訊號在稀疏設定下從無益轉為有益。

三項核心設計各自都有清楚的貢獻：softmax aspect attention 貢獻 7%，local-biased fusion gate 初始化和 KG-SVD 初始化各貢獻 5.3%，三者缺一都無法超越純 LightGCN。

方法論上，attention normalization 方式在理由感知設計中的影響值得後續研究關注。哪種設計最適合哪類 knowledge graph，是一個值得持續探索的開放問題。

最後說一下局限性。我們只在單一稀疏資料集上做了驗證，KG 建構 pipeline 也不是本文的貢獻；實驗部分還有幾個章節目前在補齊中；在 KG 豐富的資料集上，我們的方法能否保持優勢，也有待進一步驗證。

以上就是我們的報告，謝謝大家。
