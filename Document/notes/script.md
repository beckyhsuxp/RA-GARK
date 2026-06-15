# RA-GARK 口語逐字稿

<!--
Writing rule:
- Introduce a new term with a plain explanation first; do not drop a new name without context.
- Write symbols in a speakable form, e.g. y_hat(u, i), a sub i, u sub loc, i sub loc, i sub glo, alpha sub u.
- Keep technical terms in English, but keep descriptive phrasing natural and easy to read aloud.
-->

## Slide 1 — Title

大家好，我今天要報告的題目是 RA-GARK，完整名稱是 Product Recommendation via Rationale-Aware Gating over Sparse Review-Aspect Knowledge Graphs，也就是基於理由感知門控與稀疏評論面向知識圖譜之產品推薦。

## Slide 2 — Outline

這份報告大約分成五個部分。

今天的報告我會先講動機，再講相關研究，接著進入方法細節，最後看實驗結果和結論。

第一部分是導論，我會先說明為什麼稀疏 KG 會讓現有 KG-aware recommendation 失效。第二部分是相關研究，我會快速定位幾個代表性的基線，包括純 CF、KG-aware recommendation，以及 gate 相關方法。第三部分是方法章，這會是整份報告最重要的部分，我會詳細說 local view、KG-SVD、softmax rationale masking，以及 fusion gate。第四部分是實驗，會看主結果和 ablation。最後是結論與未來工作，整理貢獻、限制和後續方向。

## Slide 3 — Motivation

先講動機。

推薦系統近年很主流的一條線是用 GNN 做協同過濾，最代表性的就是 LightGCN。LightGCN 的重點是把 GNN 裡比較複雜的非線性轉換拿掉，只保留線性的鄰居聚合，結果反而在很多資料集上表現很好。這告訴我們，在推薦裡面，乾淨的協同訊號其實非常重要。

另一條線是 KG-aware recommendation。這類方法的想法是，如果能把 item 的語意資訊，像是題材、風格、主題，從知識圖譜引進來，理論上應該可以讓推薦更準，也更可解釋。KGAT、KGCL、MCCLK、KGRec 都是這一線的代表，在 KG 比較豐富的資料集上也確實有很好的表現。

但是我們在自己的設定裡看到一個很反直覺的現象。我們用的是 Amazon Books 的子集，而且 knowledge graph 是從書評抽出的 aspect，所以本來就很稀疏。過濾後平均每本書只有 2.4 條 KG 邊。在這個設定下，我們把幾個主流 KG-aware 方法都跑了一遍，結果全部都輸給純 LightGCN。LightGCN 的 NDCG@20 是 0.1179，反而高於 KGAT、KGCL、MCCLK 和 KGRec。

這個結果不是說那些方法不好，而是說當 KG 稀疏又不穩定的時候，把 KG 直接融進 scoring pipeline，很可能會把雜訊一起帶進去，最後拖累原本乾淨的協同訊號。

## Slide 4 — Why Sparse KG

接著說明為什麼這種 KG 會這麼稀疏。

答案是，稀疏 KG 在實務上很常見，不是例外。review-derived KG 本來就只會覆蓋使用者提到過的主題，所以密度自然不均勻。cold-start 和 emerging domains 通常也缺少像 Freebase 或 Wikidata 那樣完整的整理來源。再加上 medical、financial 這類 privacy-constrained domains，能用的關聯訊號也會被刻意限制。最後，KG completion 也不是無痛解法，因為它會引入新的噪音，而且通常還需要 seed signal。

所以這篇工作的重點不是去解決「KG 太少」本身，而是去解決「當 KG 不可靠時，模型要怎麼穩健地做推薦」。

這也就是為什麼我們後面會強調安全退路和可閘控側通道。這個 benchmark 看的是 robustness，而不只是 dense KG 下的最高表現。

## Slide 5 — Design Challenge

這裡我先把現有 KG-aware 方法面臨的設計挑戰講清楚。

大多數 KG-aware recommenders 的共同點是：KG entity embeddings 會直接進入 message passing，user 和 item 的表示是在一條包含 KG 的路徑上學出來的。這背後的隱含假設是，KG 可以在它出現的地方都注入有用訊號；但在 sparse KG 下，這個假設會失效。

這也是為什麼在我們的設定裡，LightGCN 反而會贏。因為 LightGCN 只看 user-item interaction，不會碰到那條不可靠的 KG branch，所以它保留了一個乾淨又安全的 baseline。

我們的回應不是把 KG 完全拿掉，而是把它改成一條專門的側通道，讓模型可以在 KG 不可靠時把它削弱，甚至完全關掉。

## Slide 6 — Research Question

基於剛才的現象，我們提出兩個問題。

第一個是 diagnosis，也就是為什麼 KG-aware 模型會在 sparse KG 下輸給純 LightGCN。第二個是 prescription，也就是什麼樣的設計原則，才能讓模型在 KG 有用時利用它，在 KG 不可靠時避免污染協同過濾。

我們的答案是：KG 不應該是 scoring pipeline 裡的必經成分，而應該是一條可以被 gate 控制的側通道。這個想法後面會具體落地在三個設計上，分別是 KG-SVD 初始化、softmax rationale masking 和 local-biased fusion gate。

## Slide 7 — Related Work I

先講最基礎的兩個方法。

LightGCN 是我們 local view 的直接前身。它的重點是把 GCN 裡比較複雜的特徵轉換拿掉，只保留線性的鄰居聚合和 layer-wise average，所以在 sparse review KG 上，它是最強的 non-KG 基準。

KGAT 則代表典型的 deep fusion。它把 user-item graph 和 KG 合併成一張 collaborative knowledge graph，KG entities 會直接參與 propagation，這在 KG dense 且高品質時通常有效。

所以我們的做法是直接把 LightGCN 原封不動地拿來當 local view，然後把 KG signal 隔離到另一條 global view。

## Slide 8 — Related Work II

接下來是對比式 KG 方法。

KGCL 會對 KG 結構做擾動，然後對 original view 和 perturbed view 做對比學習。MCCLK 則建立 collaborative、semantic、structural 三個視角，彼此做多重對齊。這些方法在 KG 比較豐富時都很強，但它們仍然假設 KG 結構本身夠有資訊。

所以我們也有用對比學習，但它只是輔助，權重很小，目的是幫 local 和 global 的幾何空間做輕量對齊，而不是主導融合。

## Slide 9 — Related Work III

KGRec 是跟我們最直接相關的工作。

這頁我用一張表直接把 KGRec 和 RA-GARK 對照起來。KGRec 的 rationale 是 edge-level 的，它用 Bernoulli dropout，也就是隨機把一部分邊丟掉，再加 contrastive learning 來挑比較重要的邊；RA-GARK 則把 rationale 放在 latent aspect-slot level，也就是先把 item 的 KG 語意壓成幾個語意槽，再用 softmax attention 直接控制 global 側通道的輸出。

所以兩者最大的差別是：KGRec 還是預設 KG 裡面至少有一些有用的 edges 可以挑出來；RA-GARK 的前提更保守，直接把整條 KG channel 當成可能不可靠的側通道來處理。

## Slide 10 — Related Work IV

這裡我想補充 gating 的脈絡。

Highway Networks 很早就提出一個很重要的概念：用 gate 把變換路徑和 identity path 做加權，而且 gate 的 bias 可以初始化成偏向安全路徑，讓模型一開始接近 identity，再慢慢學要不要打開變換。MMoE 和 PLE 則是在多任務推薦裡，用 gate 在多個 expert 分支之間做選擇。

但這些方法和我們不一樣的地方有兩個。第一，它們的 expert 多半是同質候選，不是像我們這樣把 CF 和 KG 當成兩條異質訊號管線。第二，它們沒有特別針對「某條管線可能不可信」這件事做安全初始化。

所以在 KG-aware recommendation 領域裡，還是缺少一個偏置初始化的 fusion gate，也缺少一個在稀疏或不可靠 KG 下能提供平滑退化的架構。

## Slide 11 — Design Principle

這裡把我們的方法原則講成一句話。

KG 應該是可閘控的側通道，而不是必經 scoring component。

這個原則帶來三個後果。第一，我們要把 local view 和 global view 分開，避免 KG 污染 CF。第二，融合要晚，等兩邊的 representation 都先學好再決定要不要混。第三，gate 的初始化要偏向 LightGCN，讓模型一開始就站在安全的一邊。

## Slide 12 — Overview

這一頁先看架構圖。

上半部是 local view，也就是純 LightGCN，只看 user-item graph 來保住穩定的 CF signal。下半部是 global view，先用 KG-SVD 建好 aspect slot，再用 softmax rationale masking，針對當前 user-item pair 挑出比較有用的 aspect。中間是 fusion gate，負責在最後的 scoring stage 把兩邊融合起來。

這張圖的重點是：local 和 global 先各自建模，最後再由 gate 決定 KG 佔多少比例。

## Slide 13 — Problem Setup I

這一頁先講任務和分數。

我們的任務是隱式回饋的 top-K 推薦。對每個 user，我們要把候選 item 排序，讓真實互動過的 item 排在前面。訓練時使用正樣本和抽樣得到的負樣本配對。

最終分數是 y_hat(u, i)，也就是 u sub final 跟 i sub final 的內積。y_hat(u, i) 表示模型對 user u 和 item i 的預測分數，分數越高代表越推薦。這裡先把分數定義清楚，u sub final 和 i sub final 的構成放到下一頁。

## Slide 14 — Problem Setup II

這一頁補 fusion 和 gate 的角色。

u final 和 i final 都是 local 表示和 global 表示的加權和，權重分別由 alpha u 和 alpha i 決定。這裡的 alpha 介於 0 和 1 之間：越接近 1，就越偏 local、越像純 CF；越接近 0，就越偏 global、越依賴 KG。

這裡用兩個 gate，是因為 user-side 和 item-side 的 KG usefulness 不完全一樣，所以不適合共用同一組參數。

## Slide 15 — Local View

local view 我們直接用純 LightGCN，先保住一條乾淨的 collaborative filtering 路徑。

在我們的 setting 裡，LightGCN 本來就已經比所有 KG-aware baseline 還好，所以它就是我們要守住的 safe default。

這一支只走 user-item graph，不會碰 KG edges，也不加額外的 nonlinear transformation。

## Slide 16 — Local Propagation

local propagation 的部分就是標準 LightGCN。

我們只在 user-item 二分圖上做傳播，而且只用訓練互動資料。A norm 是 normalized adjacency matrix，也就是把鄰居關係做過正規化的鄰接矩陣；E of l 是第 l 層的 embedding。

接著把第 0 層到第 K 層做層平均，得到 bar E。這個 bar E 就是整體的 local 表示，我們再從裡面讀出 u sub loc 和 i sub loc；這裡 K 設成 2。

## Slide 17 — Global View

global view 的重點是 latent aspect slots。先把每個 item 壓成四個固定的語意槽，這樣就能保留 KG 的語意，但不會直接把整張 KG 拿去傳播。

KG 很稀疏，所以如果直接傳播，訊號很容易被缺失邊或噪音邊影響。改成這種固定語意槽之後，模型不是被動地吃整張 KG，而是先把 item 的語意拆成幾個固定的槽，再在這些槽裡挑比較有用的 aspect。這樣做的好處是，global view 還是保留 KG 的語意資訊，但傳播的時候不會把雜訊直接灌進來。

這裡的表示寫成 a sub i，大小是 A x d，也就是四個槽、每個槽 d 維；R 就是實數空間。

## Slide 18 — KG-SVD Motivation

前一頁我們說過，global view 先把每個 item 壓成四個固定的語意槽。

KG-SVD 是我們用來初始化 item aspect slots 的方法。

它的目的，是先把每個 item 的 aspect 相關資訊做一個比較穩的初始化。

## Slide 19 — KG-SVD Construction

先看這張圖的左半邊，重點是把 item 和 aspect 的共現關係整理成矩陣，再把太常見的 aspect 壓低。

這一步是在建 item-aspect matrix。每個 item 如果有某個 aspect，就把對應位置設成 1；接著再乘上 aspect 的 IDF，也就是一種把常見 aspect 權重壓低的方式，讓太常見但沒辨識力的 aspect 影響變小。

## Slide 20 — KG-SVD SVD and Reshape

這張圖的右半邊就是接下來的重點，從加權後的矩陣開始做分解。

先從 IDF-weighted matrix 做只保留前 k 個成分的 truncated SVD，這裡 k 等於 A 乘 d。U sub k 是左 singular vectors，Sigma sub k 是 singular values 的對角矩陣，V sub k transpose 是右 singular vectors 的轉置。接著把結果投影成 E KG 等於 U sub k 乘 Sigma sub k 的平方根，這樣就得到每個 item 的初始 KG 表示。最後再把 flat vector reshape 成每個 item 的四個 aspect slot，每個 slot 維度是 128。

## Slide 21 — KG-SVD Effect

這張 ablation 表想表達的是：KG-SVD 不是裝飾。

RA-GARK full 是 0.1243 / 0.0594，拿掉 KG-SVD init 之後是 0.1171 / 0.0545。這表示如果沒有一個合理的起點，global view 很難在 sparse KG 下自己長出好的幾何。

## Slide 22 — Softmax Masking Motivation

global view 的第二個核心是 softmax rationale masking。

對每個 user-item pair，我們會用 user 的 global embedding 去條件化 item 的每個 aspect slot，先算出每個 slot 的 logit，再用 softmax 得到權重。

這樣做的意思是：同一本書對不同 user 可能有不同的推薦理由，所以 rationale 必須是 user-conditioned 的。

## Slide 23 — Softmax Masking Computation

這一頁就是具體計算。

MLP 是一個小型 feed-forward network。第 k 個 slot 的分數來自把 u sub glo 和 aspect slot i sub k 串接後丟進 MLP，表示第 k 個 aspect slot 對這個 user-item pair 的相對重要性；再經過 softmax(logit sub k / tau) 變成 slot 權重，其中 tau 是 softmax temperature，控制分佈有多尖銳；最後把四個 slot 加權求和成 i sub glo。

## Slide 24 — Softmax Normalization

這裡我們特別強調 softmax 而不是 sigmoid。

softmax 會讓 slot 之間互相競爭，在固定總量下做選擇，所以四個權重加起來會等於 1。這不只是讓 attention 更 sharp，更重要的是它控制了 i sub glo 的 magnitude，讓 global channel 不會自己膨脹。

## Slide 25 — Softmax vs Sigmoid

如果用 sigmoid，每個 slot 是獨立啟動的，容易所有 slot 都偏高，最後退化成平均。

所以在我們這個被 gate 控制的 sparse KG side channel 裡，softmax 比 sigmoid 更適合。

## Slide 26 — Softmax Ablation

這張 sensitivity 圖對應 w/o-softmax row。

Top-20 是 0.1005 / 0.0451，Top-10 是 0.0785 / 0.0397。這說明 normalization 的選擇會直接影響穩定性和 magnitude control。

## Slide 27 — Fusion Gate Structure

這一頁先講 fusion gate。

alpha sub u 和 alpha sub i 是用小型 MLP 算出來的，分別控制 user-side 和 item-side 的 local/global 混合比例。它們也都介於 0 和 1 之間，所以 u sub final 和 i sub final 就是 local 與 global 表示的加權和。

## Slide 28 — Gate Bias and Graceful Degradation

這一頁講 gate 的初始化。

我們把 gate bias 設成加 5，所以一開始 alpha 幾乎是 0.993，也就是模型剛開始幾乎等同於 LightGCN。b 就是 gate 的 bias，alpha 的初始值大概是 0.993，也就是 sigmoid 正 5 的結果。

這樣做的目的是讓系統先站在安全預設上。如果 KG 不可靠，gate 就維持偏關閉；如果 KG 有幫助，訓練才慢慢把它打開。

## Slide 29 — Contrastive Regularization

除了 BPR，我們還加了兩個很小的對比學習輔助項，也就是 contrastive regularization。

面向層的對比損失是物品面向的對比損失，使用者跨視角的對比損失是 user cross-view contrastive loss，也就是讓同一個 user 的 local 和 global 表示靠近；lambda sub CL 是這個輔助項的權重，tau sub CL 是對比學習用的 temperature。它們只是輔助對齊 local 和 global 的幾何空間，不是主融合機制。

## Slide 30 — Training Objective

這一頁補 BPR 的訓練目標。

BPR 是 pairwise ranking loss。sigma 是 sigmoid function。公式裡的 positive item 是正樣本，也就是使用者真的互動過的 item；negative item 是負樣本，也就是抽樣出來、使用者沒互動過的 item。我們用正樣本和 sampled negative pairs 來訓練，目標是把真正互動過的 item 排在未互動 item 前面。BPR 負責 ranking signal，gate 和 CL 負責把表示調穩定。

## Slide 31 — Dataset and Optimization

這一頁補資料規模和訓練設定。

我們的資料集有 905 個 user、1,399 個 item、22,265 筆互動、3,370 條 KG 邊，以及 2,098 個 aspect。訓練設定是 Adam，learning rate 0.001，batch size 128，最多 80 個 epoch，並且用 validation NDCG@20 做 early stopping。

## Slide 32 — Inference and Complexity

評估時採 full-ranking，也就是對每個 user 把候選 item 重新完整排序，並排除訓練集裡已經互動過的 item，最後看 HR、Precision、Recall、F1、MAP 和 NDCG，這些都取 @20。

從效能來看，我們每個 epoch 大概 1.5 秒，跟 KGRec 差不多，所以這個設計沒有讓成本爆炸。

## Slide 33 — Main Results

先看主結果。

Top-20 時，RA-GARK 的 NDCG@20 是 0.1243，較 KGRec 高 13.5%，較純 LightGCN 高 5.4%。Top-10 時，RA-GARK 的 NDCG@10 是 0.0966，較 KGRec 高 10.5%，較純 LightGCN 高 6.4%。

## Slide 34 — Ablation Summary

再看 ablation。

Top-20 時，softmax head 是最大的變化，0.1243 降到 0.1005；KG-SVD 是 0.1171；fusion-gate bias 是 0.1194；MLP gate 是 0.1180。Top-10 也維持相同排序。

## Slide 35 — Case Study and Takeaways

這張 heatmap 是 case study。

你可以看到不同 item 會對不同 aspect slot 給出不同的權重，表示 rationale masking 不是固定平均，而是真的有在對不同 item 使用不同的語意路徑。

## Slide 36 — Conclusion

最後總結一下。

當 KG 不可靠時，架構最需要的不是更強的 KG aggregator，而是一個能把 KG opt out 的 structural switch。這篇工作的主要貢獻有四個：gateable KG side channel、KG-SVD initialization、softmax rationale masking、local-biased fusion gate。

以上，謝謝大家。
