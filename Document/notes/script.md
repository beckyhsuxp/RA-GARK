# RA-GARK 口語逐字稿

<!--
Writing rule:
- Introduce a new term with a plain explanation first; do not drop a new name without context.
- Write symbols in a speakable form, e.g. y_hat(u, i), a_i, u_final, i_loc, i_global, alpha_u.
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

大多數 KG-aware recommenders 的共同點是：KG entity embeddings 會直接進入訊息傳遞，user 和 item 的表示是在一條包含 KG 的路徑上學出來的。這背後的隱含假設是，KG 可以在它出現的地方都注入有用訊號；但在 sparse KG 下，這個假設會失效。

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

這頁我用一張表直接把 KGRec 和 RA-GARK 對照起來。KGRec 的 rationale 是 edge-level 的，它用 Bernoulli dropout，也就是隨機把一部分邊丟掉，再加 contrastive learning 來挑比較重要的邊；RA-GARK 則把 rationale 放在 latent aspect-slot level，也就是先把 item 的 KG 語意壓成幾個 semantic slots，再用 softmax attention 直接控制 global 側通道的輸出。

所以兩者最大的差別是：KGRec 還是預設 KG 裡面至少有一些有用的 edges 可以挑出來；RA-GARK 的前提更保守，直接把整條 KG channel 當成可能不可靠的側通道來處理。

## Slide 10 — Related Work IV

這裡我想補充 gating 的脈絡。

Highway Networks 很早就提出一個很重要的概念：用 gate 把變換路徑和 identity path 做加權，而且 gate 的 bias 可以初始化成偏向安全路徑，讓模型一開始接近 identity，再慢慢學要不要打開變換。MMoE 和 PLE 則是在多任務推薦裡，用 gate 在多個 expert 分支之間做選擇。

但這些方法和我們不一樣的地方有兩個。第一，它們的 expert 多半是同質候選，不是像我們這樣把 CF 和 KG 當成兩條異質訊號管線。第二，它們沒有特別針對「某條管線可能不可信」這件事做安全初始化。

所以在 KG-aware recommendation 領域裡，還是缺少一個偏置初始化的 fusion gate，也缺少一個在稀疏或不可靠 KG 下能提供平滑退化的架構。

## Slide 11 — Design Principle

這裡把我們的方法原則講成一句話。

KG 應該是可閘控的側通道，而不是必經 scoring component。

這個原則帶來三個後果。第一，我們要把 local view 和 global view 分開，避免 KG 污染 CF。第二，融合要晚，等兩邊的表示都先學好再決定要不要混。第三，gate 的初始化要偏向 LightGCN，讓模型一開始就站在安全的一邊。

## Slide 12 — Overview

這一頁先看架構圖。

上半部是 local view，也就是純 LightGCN，只看 user-item graph 來保住穩定的 CF signal。下半部是 global view，先用 KG-SVD 建好 aspect slot，再用 softmax rationale masking，針對當前 user-item pair 挑出比較有用的 aspect。中間是 fusion gate，負責在最後的 scoring stage 把兩邊融合起來。

這張圖的重點是：local 和 global 先各自建模，最後再由 gate 決定 KG 佔多少比例。

## Slide 13 — Problem Setup I

這一頁先講任務和分數。

我們的任務是隱式回饋的 top-K 推薦。對每個 user，我們要把候選 item 排序，讓真實互動過的 item 排在前面。訓練時使用正樣本和抽樣得到的負樣本配對。

最終分數是 y_hat(u, i)，也就是 u_final 跟 i_final 的內積。y_hat(u, i) 表示模型對 user u 和 item i 的預測分數，分數越高代表越推薦。這裡先把分數定義清楚，u_final 和 i_final 的構成放到下一頁。

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

接著把第 0 層到第 K 層做層平均，得到 bar E。這個 bar E 就是整體的 local 表示，我們再從裡面讀出 u_loc 和 i_loc；這裡 K 設成 2。

## Slide 17 — Global View

global view 的重點是 latent aspect slots。先把每個 item 壓成四個固定的 semantic slots，這樣就能保留 KG 的語意，但不會直接把整張 KG 拿去傳播。

KG 很稀疏，所以如果直接傳播，訊號很容易被缺失邊或噪音邊影響。改成這種固定 semantic slots 之後，模型不是被動地吃整張 KG，而是先把 item 的語意拆成幾個固定的 slot，再在這些 slot 裡挑比較有用的 aspect。這樣做的好處是，global view 還是保留 KG 的語意資訊，但傳播的時候不會把雜訊直接灌進來。

這裡的表示寫成 a_i，大小是 A x d，也就是四個 slot、每個 slot 維度是 d；R 就是實數空間。

## Slide 18 — KG-SVD Motivation

前一頁我們說過，global view 先把每個 item 壓成四個固定的 semantic slots。

KG-SVD 是我們用來初始化 item aspect slots 的方法。

它的目的，是先把每個 item 的 aspect 相關資訊做一個比較穩的初始化。

## Slide 19 — KG-SVD: Construction

這一步是在建 item-aspect matrix，也就是整理 item 和 aspect 一起出現的關係。矩陣裡的 `M_i,a` 表示 item i 和 aspect a 的關係，item i 如果有 aspect a，就把對應位置設成 1。

接著再乘上 aspect 的 IDF，也就是一種把常見 aspect 權重壓低的方式，讓太常見但沒辨識力的 aspect 影響變小。

這個公式的意思很簡單：出現越多的 aspect，IDF 就越小；`I` 是 item 總數，`M_i,a` 等於 1 代表 item i 有 aspect a，所以分母裡那一項就是這個 aspect 出現過的 item 數再加 1。分母裡的 `+1` 是避免除零，外面的 `+1` 是避免權重變成 0。

所以重點是把 item 和 aspect 一起出現的關係整理成矩陣，再把太常見的 aspect 壓低。

## Slide 20 — KG-SVD: SVD and Reshape

前一頁我們先把 item 和 aspect 的共現關係整理成加權矩陣，這一頁就接著看右半邊，從這個矩陣開始做分解。

先從 IDF-weighted matrix 做 truncated SVD，也就是只保留前 k 個成分。這裡 `k` 取 `A 乘 d`，也就是把 `A` 個 slot、每個 slot 維度是 d 的總維度保留下來。你可以把這一步想成把加權後的矩陣拆成三個部分：左邊的 `U_k`、中間的 `Sigma_k`、以及右邊的 `V_k transpose`。`U_k` 可以理解成 item 的低維表示，`Sigma_k` 是每個方向的重要程度；`V_k transpose` 只是分解的一部分，這裡先用 `U_k` 和 `Sigma_k` 的平方根來形成 `E_KG`，也就是每個 item 的初始 KG 表示。

接著把 `E_KG` reshape 成 `A_KG_0`，也就是把每個 item 表成 A 個 aspect slot，維度是 d。這裡的 zero 表示初始化後的第一版；整個 `A_KG_0` 可以想成一個三維張量，也就是 item 數乘 A 乘 d 的大小，裡面的值都來自實數空間。reshape 就是把這些數值排回 A 個 slot。接著會用這個初始化結果交給 graph recommender，也就是 GNN-based recommender 往下做。整體來說，這一步先把加權後的矩陣壓成幾個比較重要的方向，再把分解出來的 item 表示整理成四個 slot。

## Slide 21 — KG-SVD: Initialization Effect

這一頁是在總結 KG-SVD 的初始化效果。它先給 global view 一個比較好的起點，讓 item 的 KG 表示一開始就帶有合理的語意結構，而不是從隨機初始化開始亂長。

更重要的是，這個初始化保留了 item 和 aspect 的共現結構，所以在 training 之前，model 就已經有一個比較穩的表示結構。這不是一個要從零學出的模組，而是先把 slot 放到合理的位置，之後再跟著訓練微調。這也是為什麼在 sparse KG 的情況下，KG-SVD 會明顯幫助後面的 global view。
## Slide 22 — Softmax Masking Motivation

前一頁先把 item 初始化成 aspect slots，這一頁接著看怎麼根據 user 來挑哪個 slot 比較重要。

重點是：同一個 item 對不同 user 可能有不同的推薦理由，所以 slot 的選擇必須跟 user 綁在一起，而不是固定用同一個 slot。

## Slide 23 — Softmax Masking Computation

這一頁就是具體計算。

先看公式。`\ell_{u,i,k}` 是第 k 個 slot 的分數，來自把 `u_global` 和 `a_i,k` 串接後丟進 MLP。MLP 是一個小型前饋網路。接著把 `\ell_{u,i,k}` 除以 `tau` 再做 softmax，就得到 `w_{u,i,k}` 這個權重；`tau` 是 softmax temperature，控制分佈有多尖銳。最後，`i_global` 就是把四個 slot 依照這些權重加權求和。這樣就完成從 slot 打分到 global 向量的組合。

## Slide 24 — Softmax Normalization

上一頁已經算出每個 slot 的權重，這一頁把 normalization choice 一起講完。

這張表是在對照 softmax 和 sigmoid。sigmoid 的意思是每個 slot 各自判斷、彼此不互相影響，所以理論上每個 slot 都可以各自拉高。softmax 則不一樣，它會把所有 slot 放在同一個總量裡一起比較，某一個 slot 權重變大，其他 slot 的權重就會被壓下來，所以四個權重加起來一定等於 1。

在 RA-GARK 裡，我們選 softmax，因為它不只是在選哪個 slot 比較重要，還會把整個 `i_global` 的大小控制在比較穩定的範圍內。這很重要，因為後面的 gate 會拿這個 global 向量去跟 local 向量做融合；如果 global 向量的 magnitude 不穩，gate 的輸入尺度就會飄。softmax 先把這個 KG side channel 的輸出幅度壓住，後面的融合才比較好校準。

## Slide 25 — Fusion Gate Overview

這一頁先把畫面聚焦到最後的融合位置。local view 和 global view 前面都各自獨立建模，接下來從圖的左邊 gate 一路看到右邊的 fusion。

## Slide 26 — Fusion Gate Structure

這裡先以 user-side 為例，圖從左往右看，先把 `u_loc` 和 `u_glo` 串起來，得到 gate 的輸入。

接下來是中間的 MLP。`Gate(z)` 可以直接理解成一個兩層的小網路：先用 `Wz` 做一次線性投影，再經過 `tanh` 加上非線性，接著用 `w^T` 壓成一個分數，最後加上 bias `b` 丟進 sigmoid，把輸出壓到 0 到 1 之間。`Wz` 就是把兩個輸入混在一起做特徵變換，`tanh` 則是讓它不要只是線性組合。

經過這個 gate 之後，就得到 `alpha_u`。它是一個 0 到 1 之間的權重，用來控制 `u_final` 裡 local 和 global 的比例。

最後看右邊的 fusion。`alpha_u` 會一部分乘上 `u_loc`，另一部分乘上 `u_glo`，加總成 `u_final`。整個 gate 的作用，就是先偏向 local，之後再根據訓練慢慢決定要不要放更多 KG 進來。

## Slide 27 — Gate Bias and Graceful Degradation

前一頁我們已經看到 `alpha_u` 是 gate 的輸出，這一頁接著看它的初始化設定。

我們把 gate bias 設成加 5，所以一開始 `alpha` 幾乎是 `0.993`，也就是 `sigmoid(5)` 的結果。這代表模型剛開始幾乎等同於 LightGCN，重點是它讓模型一開始站在比較安全的 local 預設上。

這樣做的目的是讓系統先站在安全預設上。如果 KG 不可靠，gate 就維持偏關閉；如果 KG 有幫助，訓練才慢慢把它打開。

## Slide 28 — Training Objective

前一頁 gate 初始化完之後，這一頁回到訓練目標。

模型最後的 score 是 user 和 item 的 final representation 做內積。BPR 是用正負樣本做排序學習的 loss；`i+` 是使用者真的互動過的 item，`i-` 是抽樣出來、使用者沒互動過的 item。對每個已觀察互動，我們會再抽一個沒互動過的 item，讓模型把正樣本排在負樣本前面。BPR 負責把排序學好，gate 則是先把 local 和 global 的融合控制住。接下來看總 loss，除了 BPR，還會再加上一個很小的對比正則。

## Slide 29 — Total Objective

這一頁就是總損失。除了 BPR，我們還加上一個很小的對比正則，讓同一個 user 或 item 在 local view 和 global view 的表示在向量空間裡拉近，但不取代 BPR。`lambda_CL` 控制這個輔助項的強度；這裡不用特別把其他超參數唸出來。

這兩個對比項分別是物品面向和使用者跨視角的對齊，作用都是把兩個 view 的表示距離縮小一點，不是主融合機制。

## Slide 30 — Dataset

這一頁先看資料集。它來自 Amazon Books 的評論子集，重點是平均每個 item 只有 2.4 條 KG 邊，是一個很稀疏的 KG；另外還有 905 個 user、1,399 個 item、22,265 筆互動、3,370 條 KG 邊，以及 2,098 個 aspect。

## Slide 31 — Experimental Setup

這頁簡單看一下訓練設定。

## Slide 32 — Main Results I

先看評估方式。我們採 full-ranking，也就是對每個 user 把候選 item 重新完整排序，並排除訓練集裡已經互動過的 item，最後看 HR、Precision、Recall、F1、MAP 和 NDCG，這些都取 @20。接著看 Top-20。這張表先列幾個 baseline，包含 MCCLK、KGCL、KGAT、KGRec 和純 LightGCN，最後是 RA-GARK。ranking metrics 是 NDCG@20、HR@20、Recall@20 和 MAP@20；RA-GARK 在這四個指標都最好，表示在這個 sparse KG 設定下，這個架構真的把 KG 的訊號轉成了正向貢獻。

## Slide 33 — Main Results II

再看 Top-10。這一頁的排序和 Top-20 一樣，RA-GARK 仍然維持最好的 NDCG@10、HR@10、Recall@10 和 MAP@10，表示結果不是只在較長候選列表下才成立。

## Slide 34 — Ablation Results I

先看 ablation 的前半段。這一頁可以直接對照前面的主結果：softmax head 掉最多，接著是 KG-SVD init、fusion-gate bias 和 MLP gate，代表這幾個設計是主要來源。

## Slide 35 — Ablation Results II

再看 ablation 的後半段。user CL、aspect CL、rationale-enabled selection 和 global view 的影響都比較小，但還是能看到穩定的下降，表示這些輔助設計也有幫助。

## Slide 36 — Case Study

這張圖每個小圖是一個 item，橫軸是 4 個 aspect slot，縱軸是不同 user。顏色越深代表權重越高；你可以看到同一個 item 會有一個比較明顯的主 slot，但不同 user 對同一個 item 的分布又很接近，表示它主要是在做 item-level 的 slot 選擇。

所以這個 case study 的重點是：不同 item 會偏向不同的 slot，而同一個 item 在不同 user 之間的差異不大。

## Slide 37 — Conclusion

最後總結一下。

當 KG 不可靠時，架構最需要的不是更強的 KG aggregator，而是一個能把 KG opt out 的 structural switch。這篇工作的主要貢獻有四個：gateable KG side channel、KG-SVD initialization、softmax rationale masking、local-biased fusion gate。

## Slide 38 — Thank You

以上，謝謝大家。
