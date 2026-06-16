# RA-GARK 口語逐字稿

<!--
Writing rule:
- Introduce a new term with a plain explanation first; do not drop a new name without context.
- Write symbols in a speakable form, e.g. y_hat(u, i), a_i, u_final, i_loc, i_global, alpha_u.
- Keep technical terms in English, but keep descriptive phrasing natural and easy to read aloud.
-->

## Slide 1 — Title

大家好，我今天要報告的題目是 Product Recommendation via Rationale-Aware Gating over Sparse Review-Aspect Knowledge Graphs，也就是基於理由感知門控與稀疏評論面向知識圖譜之產品推薦。

## Slide 2 — Outline

這份報告大約分成五個部分。

今天的報告我會先講動機，再講相關研究，接著進入方法細節，最後看實驗結果和結論。

## Slide 3 — Motivation

先講動機。

在推薦系統裡，最核心的訊號其實還是 user 和 item 之間的互動。像 LightGCN 這類用 GNN 做協同過濾的方法，就是直接在 user-item graph 上傳播訊息。LightGCN 把 GNN 簡化成線性鄰居聚合，但仍然很有效。這代表一件事：乾淨的互動訊號本身就很強，而且很穩定。

但是只看互動也有一個限制，就是它比較難知道 item 為什麼被推薦。所以另一條線是 KG-aware recommendation，把 item 的語意資訊，像是題材、風格、主題，透過 knowledge graph 引進模型裡。KG 理論上可以補上協同過濾看不到的內容訊號，但 sparse KG 不一定可靠。

也因此，我們看到一個很反直覺的現象：幾個主流 KG-aware 方法全部都輸給純 LightGCN。

所以這裡的動機不是說 KG 沒有用，而是說當 KG 稀疏又不穩定時，如果把 KG 直接融進評分流程，它很可能不是補充訊號，而是把雜訊帶進來，最後拖累原本乾淨的互動訊號。

## Slide 4 — Why Sparse KG

那為什麼我們要特別討論 sparse KG？

答案是，稀疏 KG 在實務上很常見，不是例外。像 review-derived KG，只會記錄使用者真的提到過的主題，所以 coverage 本來就不平均。

在 cold-start 或新領域裡，也常常沒有完整的外部知識來源可以用。有些領域還會受到隱私或資料取得限制，能整理出的關聯本來就少。

就算想用 KG completion 把缺的邊補起來，也不一定可靠，因為補出來的關係可能會帶進新的噪音。

所以這篇工作的重點不是去解決「KG 太少」本身，而是去解決「當 KG 不可靠時，模型要怎麼穩健地做推薦」。

## Slide 5 — Design Challenge

這裡我先把現有 KG-aware 方法面臨的設計挑戰講清楚。

大多數 KG-aware recommenders 會把 KG 直接混進 user 和 item 的表示裡。換句話說，模型在學推薦分數的時候，KG 不是額外參考，而是表示學習的一部分。

這樣做其實假設了一件事：只要 KG 被放進模型，它多半就是有幫助的。但在 sparse KG 下，這個假設很容易失效。

所以這裡的 challenge 是：如果 KG 本身不可靠，模型要怎麼避免讓它一路影響最後的推薦分數。這個問題會帶到後面的 research question 和 design principle。

## Slide 6 — Research Question

基於剛才的現象，我們提出兩個問題。

第一個是 diagnosis，也就是為什麼 KG-aware 模型會在 sparse KG 下輸給純 LightGCN。第二個是 prescription，也就是什麼樣的設計原則，才能讓模型在 KG 有用時利用它，在 KG 不可靠時避免污染協同過濾。

這篇工作我們把它命名為 RA-GARK。

對應這兩個問題，我們的答案是：KG 不應該是評分流程裡必須使用的組件。後面的模型設計都會圍繞這個原則展開。

## Slide 7 — Related Work I

接下來看 related work。

首先是 collaborative filtering。簡單說，LightGCN 只用 user-item interaction 做推薦。它的限制是完全不看 KG，因此沒辦法利用 item 的語意資訊。

接著是 direct KG fusion。KGAT 和 KGRec 代表這一類方法，會把 KG 用在 user 和 item 的表示學習，以及最後的推薦分數計算裡；它們的前提是 KG 本身值得信任。

## Slide 8 — Related Work II

接著是 contrastive KG learning。簡單說，KGCL 和 MCCLK 會用對比學習強化 KG 表示，但它們依賴 KG 結構本身要夠有資訊，對比學習才會穩。

最後是 gating。Highway Networks、MMoE、PLE 證明 gate 可以控制資訊流，但它們不是為 unreliable KG 設計，也沒有安全初始化。

## Slide 9 — Design Principle

這一頁把設計原則整理成三個部分。

第一個是 failure mode。很多 KG-aware 方法會讓 KG embeddings 直接進入 message passing，等於假設 KG 一放進模型就會有幫助。但在 sparse KG 下，這個假設很容易失效。

第二個是 safe default。LightGCN 只用 user-item interaction，所以不會被不可靠的 KG 污染。這讓它成為一個乾淨的起點。

第三個是我們的 response。RA-GARK 不把 KG 當成必須使用的主路徑，而是把它放到專門的 side channel，讓模型在 KG 不可靠時可以降低它的影響，甚至幾乎不用它。

## Slide 10 — Overview

這一頁先看架構圖。

上半部是 local view，也就是純 LightGCN，只看 user-item graph 來保住穩定的 CF signal。下半部是 global view，先用 KG-SVD 建好 aspect slot，再用 softmax rationale masking，針對當前 user-item pair 挑出比較有用的 aspect。中間是 fusion gate，負責在最後的 scoring stage 把兩邊融合起來。

這張圖的重點是：local 和 global 先各自建模，最後再由 gate 決定 KG 佔多少比例。

## Slide 11 — Problem Setup I

這一頁先講任務和分數。

我們的任務是隱式回饋的 top-K 推薦。對每個 user，我們要把候選 item 排序，讓真實互動過的 item 排在前面。訓練時使用正樣本和抽樣得到的負樣本配對。

這裡的 u 代表 user，i 代表 item；u_final 和 i_final 則是模型最後得到的 user 向量和 item 向量。

最終分數由 u_final 跟 i_final 的內積得到。y_hat(u, i) 表示模型對 user u 和 item i 的預測分數，分數越高代表越推薦。這裡先把分數定義清楚，u_final 和 i_final 的構成放到下一頁。

## Slide 12 — Problem Setup II

這一頁補 fusion 和 gate 的角色。

u final 和 i final 都是 local 表示和 global 表示的加權和，權重分別由 alpha u 和 alpha i 決定。這裡的 alpha 介於 0 和 1 之間：越接近 1，就越偏 local、越像純 CF；越接近 0，就越偏 global、越依賴 KG。

這裡用兩個 gate，是因為 user-side 和 item-side 的 KG usefulness 不完全一樣，所以不適合共用同一組參數。

## Slide 13 — Local View

local view 我們直接用純 LightGCN，先保住一條乾淨的 collaborative filtering 路徑。

在我們的 setting 裡，LightGCN 本來就已經比所有 KG-aware baseline 還好，所以它就是我們要守住的 safe default。

這一支只走 user-item graph，不會碰 KG edges，也不加額外的 nonlinear transformation。

## Slide 14 — Local Propagation

local propagation 的部分就是標準 LightGCN。

我們只在 user-item 二分圖上做傳播，而且只用訓練互動資料。A norm 是 normalized adjacency matrix，也就是把鄰居關係做過正規化的鄰接矩陣；E of l 是第 l 層的 embedding。

接著把第 0 層到第 K 層做層平均，得到 bar E。這個 bar E 就是整體的 local 表示，我們再從裡面讀出 u_loc 和 i_loc；這裡 K 設成 2。

## Slide 15 — Global View

global view 的重點是 latent aspect slots。它不是直接把整張 KG 拿去傳播，而是先把每個 item 的語意整理成四個固定的 semantic slots。

這樣做的好處是，模型還是可以保留 KG 的語意資訊，但後面只需要在這些 slots 裡挑比較有用的 aspect，不用被動地吃進整張 sparse KG。

這裡的表示寫成 a_i，大小是 A x d，也就是四個 slot、每個 slot 維度是 d；R 就是實數空間。

## Slide 16 — KG-SVD Motivation

前一頁我們說過，global view 先把每個 item 壓成四個固定的 semantic slots。

KG-SVD 是我們用來初始化 item aspect slots 的方法。

它的目的，是先把每個 item 的 aspect 相關資訊做一個比較穩的初始化。

## Slide 17 — KG-SVD: Construction

這一步是在建 item-aspect matrix，也就是整理 item 和 aspect 一起出現的關係。矩陣裡的 `M_i,a` 表示 item i 和 aspect a 的關係，item i 如果有 aspect a，就把對應位置設成 1。

接著再乘上 aspect 的 IDF，也就是一種把常見 aspect 權重壓低的方式，讓太常見但沒辨識力的 aspect 影響變小。

這個公式的意思很簡單：出現越多的 aspect，IDF 就越小；`I` 是 item 總數，`M_i,a` 等於 1 代表 item i 有 aspect a，所以分母裡那一項就是這個 aspect 出現過的 item 數再加 1。分母裡的 `+1` 是避免除零，外面的 `+1` 是避免權重變成 0。

所以重點是把 item 和 aspect 一起出現的關係整理成矩陣，再把太常見的 aspect 壓低。

## Slide 18 — KG-SVD: SVD and Reshape

前一頁我們先把 item 和 aspect 的共現關係整理成加權矩陣，這一頁就接著看右半邊，從這個矩陣開始做分解。

先從 IDF-weighted matrix 做 truncated SVD，也就是只保留前 k 個成分。這裡 `k` 取 `A 乘 d`，也就是把 `A` 個 slot、每個 slot 維度是 d 的總維度保留下來。你可以把這一步想成把加權後的矩陣拆成三個部分：左邊的 `U_k`、中間的 `Sigma_k`、以及右邊的 `V_k transpose`。`U_k` 可以理解成 item 的低維表示，`Sigma_k` 是每個方向的重要程度；`V_k transpose` 只是分解的一部分，這裡先用 `U_k` 和 `Sigma_k` 的平方根來形成 `E_KG`，也就是每個 item 的初始 KG 表示。

接著把 `E_KG` reshape 成 `A_KG_0`，也就是把每個 item 表成 A 個 aspect slot，維度是 d。這裡的 zero 表示初始化後的第一版；整個 `A_KG_0` 可以想成一個三維張量，也就是 item 數乘 A 乘 d 的大小，裡面的值都來自實數空間。reshape 就是把這些數值排回 A 個 slot。接著會用這個初始化結果交給 graph recommender，也就是 GNN-based recommender 往下做。整體來說，這一步先把加權後的矩陣壓成幾個比較重要的方向，再把分解出來的 item 表示整理成四個 slot。

## Slide 19 — Softmax Masking Computation

KG-SVD 初始化完之後，下一步就是看怎麼根據 user 來挑比較重要的 slot。

重點是，同一個 item 對不同 user 可能有不同的推薦理由，所以 slot 的選擇不能固定不變，而是要跟 user 綁在一起。

有了這個動機，這一頁再看具體怎麼算。

先看公式。`\ell_{u,i,k}` 是第 k 個 slot 的分數，來自把 `u_global` 和 `a_i,k` 串接後丟進 MLP。MLP 是一個小型前饋網路。接著把 `\ell_{u,i,k}` 除以 `tau` 再做 softmax，就得到 `w_{u,i,k}` 這個權重；`tau` 是 softmax temperature，控制分佈有多尖銳。最後，`i_global` 就是把四個 slot 依照這些權重加權求和。這樣就完成從 slot 打分到 global 向量的組合。

## Slide 20 — Softmax Normalization

上一頁已經算出每個 slot 的權重，這一頁把 normalization choice 一起講完。

這張表是在對照 softmax 和 sigmoid。sigmoid 的意思是每個 slot 各自判斷、彼此不互相影響，所以理論上每個 slot 都可以各自拉高。softmax 則不一樣，它會把所有 slot 放在同一個總量裡一起比較，某一個 slot 權重變大，其他 slot 的權重就會被壓下來，所以四個權重加起來一定等於 1。

在 RA-GARK 裡，我們選 softmax，因為它不只是在選哪個 slot 比較重要，還會把整個 `i_global` 的大小控制在比較穩定的範圍內。這很重要，因為後面的 gate 會拿這個 global 向量去跟 local 向量做融合；如果 global 向量的 magnitude 不穩，gate 的輸入尺度就會飄。softmax 先把這個 KG side channel 的輸出幅度壓住，後面的融合才比較好校準。

## Slide 21 — Fusion Gate Overview

這一頁先把畫面聚焦到最後的融合位置。local view 和 global view 前面都各自獨立建模，接下來從圖的左邊 gate 一路看到右邊的 fusion。

## Slide 22 — Fusion Gate Structure

這裡先以 user-side 為例，圖從左往右看，先把 `u_loc` 和 `u_glo` 串起來，得到 gate 的輸入。

接下來是中間的 MLP。`Gate(z)` 可以直接理解成一個兩層的小網路：先用 `Wz` 做一次線性投影，再經過 `tanh` 加上非線性，接著用 `w^T` 壓成一個分數，最後加上 bias `b` 丟進 sigmoid，把輸出壓到 0 到 1 之間。`Wz` 就是把兩個輸入混在一起做特徵變換，`tanh` 則是讓它不要只是線性組合。

經過這個 gate 之後，就得到 `alpha_u`。它是一個 0 到 1 之間的權重，用來控制 `u_final` 裡 local 和 global 的比例。

最後看右邊的 fusion。`alpha_u` 會一部分乘上 `u_loc`，另一部分乘上 `u_glo`，加總成 `u_final`。整個 gate 的作用，就是先偏向 local，之後再根據訓練慢慢決定要不要放更多 KG 進來。

## Slide 23 — Gate Bias and Graceful Degradation

前一頁我們已經看到 `alpha_u` 是 gate 的輸出，這一頁接著看它的初始化設定。

我們把 gate bias 設成加 5，所以一開始 `alpha` 幾乎是 `0.993`，也就是 `sigmoid(5)` 的結果。這代表模型剛開始幾乎等同於 LightGCN，重點是它讓模型一開始站在比較安全的 local 預設上。

這樣做的目的是讓系統先站在安全預設上。如果 KG 不可靠，gate 就維持偏關閉；如果 KG 有幫助，訓練才慢慢把它打開。

## Slide 24 — Training Objective

前一頁 gate 初始化完之後，這一頁回到訓練目標。

模型最後的 score 是 user 和 item 的 final representation 做內積。BPR 是用正負樣本做排序學習的 loss；`i+` 是使用者真的互動過的 item，`i-` 是抽樣出來、使用者沒互動過的 item。對每個已觀察互動，我們會再抽一個沒互動過的 item，讓模型把正樣本排在負樣本前面。BPR 負責把排序學好，gate 則是先把 local 和 global 的融合控制住。接下來看總 loss，除了 BPR，還會再加上一個很小的對比正則。

## Slide 25 — Total Objective

這一頁就是總損失。除了 BPR，我們還加上一個很小的對比正則，讓同一個 user 或 item 在 local view 和 global view 的表示在向量空間裡拉近，但不取代 BPR。`lambda_CL` 控制這個輔助項的強度；這裡不用特別把其他超參數唸出來。

這兩個對比項分別是物品面向和使用者跨視角的對齊，作用都是把兩個 view 的表示距離縮小一點，不是主融合機制。

## Slide 26 — Dataset

這一頁先看資料集。它來自 Amazon Books 的評論子集，重點是平均每個 item 只有 2.4 條 KG 邊，是一個很稀疏的 KG；另外還有 905 個 user、1,399 個 item、22,265 筆互動、3,370 條 KG 邊，以及 2,098 個 aspect。

## Slide 27 — Experimental Setup

這頁簡單看一下訓練設定。

## Slide 28 — Main Results I

先看評估方式。我們採 full-ranking，也就是對每個 user 把候選 item 重新完整排序，並排除訓練集裡已經互動過的 item，最後看 HR、Precision、Recall、F1、MAP 和 NDCG，這些都取 @20。接著看 Top-20。這張表先列幾個 baseline，包含 MCCLK、KGCL、KGAT、KGRec 和純 LightGCN，最後是 RA-GARK。ranking metrics 是 NDCG@20、HR@20、Recall@20 和 MAP@20；RA-GARK 在這四個指標都最好，表示在這個 sparse KG 設定下，這個架構真的把 KG 的訊號轉成了正向貢獻。

## Slide 29 — Main Results II

再看 Top-10。這一頁的排序和 Top-20 一樣，RA-GARK 仍然維持最好的 NDCG@10、HR@10、Recall@10 和 MAP@10，表示結果不是只在較長候選列表下才成立。

## Slide 30 — Ablation Results I

先看 ablation 的前半段。這一頁可以直接對照前面的主結果：softmax head 掉最多，接著是 KG-SVD init、fusion-gate bias 和 MLP gate，代表這幾個設計是主要來源。

## Slide 31 — Ablation Results II

再看 ablation 的後半段。user CL、aspect CL、rationale-enabled selection 和 global view 的影響都比較小，但還是能看到穩定的下降，表示這些輔助設計也有幫助。

## Slide 32 — Case Study

這張圖每個小圖是一個 item，橫軸是 4 個 aspect slot，縱軸是不同 user。顏色越深代表權重越高；你可以看到同一個 item 會有一個比較明顯的主 slot，但不同 user 對同一個 item 的分布又很接近，表示它主要是在做 item-level 的 slot 選擇。

所以這個 case study 的重點是：不同 item 會偏向不同的 slot，而同一個 item 在不同 user 之間的差異不大。

## Slide 33 — Conclusion & Future Work

最後總結一下。

當 KG 不可靠時，架構最需要的不是更強的 KG aggregator，而是一個能把 KG opt out 的 structural switch。這篇工作的主要貢獻有四個：gateable KG side channel、KG-SVD initialization、softmax rationale masking、local-biased fusion gate。

未來工作會測試更密集的 KG benchmark，也會進一步分析什麼情況下 user-level 的推薦理由差異會更明顯。

## Slide 34 — Thank You

Thank you for listening.
