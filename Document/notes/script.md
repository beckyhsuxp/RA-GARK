# RA-GARK 口語逐字稿

## Slide 1 — Title

大家好，我今天要報告的題目是 RA-GARK，完整名稱是 Product Recommendation via Rationale-Aware Gating over Sparse Review-Aspect Knowledge Graphs，也就是基於理由感知門控與稀疏評論面向知識圖譜之產品推薦。

## Slide 2 — Roadmap

這份報告大約分成五個部分。

今天的報告我會先講動機，再講相關研究，接著進入方法細節，最後看實驗結果和結論。

第一部分是 introduction，我會先說明為什麼稀疏 KG 會讓現有 KG-aware 方法失效。第二部分是 related work，我會快速定位幾個代表性的 baseline，包括純 CF、KG 深度融合、contrastive KG、rationale-aware KG，還有 gating 相關方法。第三部分是 methodology，這會是整份報告最重要的部分，我會詳細說 local view、KG-SVD、softmax rationale masking，以及 fusion gate。第四部分是 experiments，會看主結果和 ablation。最後是 conclusion & future work，整理貢獻、限制和後續方向。

## Slide 3 — Motivation

先講動機。

推薦系統近年很主流的一條線是用 GNN 做 collaborative filtering，最代表性的就是 LightGCN。LightGCN 的重點是把 GNN 裡比較複雜的 nonlinear transformation 拿掉，只保留線性的鄰居聚合，結果反而在很多資料集上表現很好。這告訴我們，在推薦裡面，乾淨的 collaborative signal 其實非常重要。

另一條線是 KG-aware recommendation。這類方法的想法是，如果能把 item 的語意資訊，像是題材、風格、主題，從 knowledge graph 引進來，理論上應該可以讓推薦更準，也更可解釋。KGAT、KGCL、MCCLK、KGRec 都是這一線的代表，在 KG 比較豐富的資料集上也確實有很好的表現。

但是我們在自己的設定裡看到一個很反直覺的現象。我們用的是 Amazon Books 的子集，而且 knowledge graph 是從書評抽出的 aspect，所以本來就很稀疏。過濾後平均每本書只有 2.4 條 KG 邊。在這個設定下，我們把幾個主流 KG-aware 方法都跑了一遍，結果全部都輸給純 LightGCN。LightGCN 的 NDCG@20 是 0.1179，反而高於 KGAT、KGCL、MCCLK 和 KGRec。

這個結果不是說那些方法不好，而是說當 KG 稀疏又不穩定的時候，把 KG 直接融進 scoring pipeline，很可能會把雜訊一起帶進去，最後拖累原本乾淨的 collaborative signal。

## Slide 4 — Why Sparse KG

接下來我想回答一個很自然的問題：既然 KG 不夠好，那為什麼不直接把 KG 補完整？

答案是，稀疏 KG 其實是現實世界的常態，不是例外。像 review-derived KG，本來就只會在使用者真的提到某些 aspect 的時候才出現邊，所以邊密度會受評論內容限制。再來是 cold-start、niche domain、長尾商品，這些場景本來就沒有足夠的外部知識來源。還有一些像醫療、金融這種隱私敏感領域，能用的關係訊號也會刻意被限縮。至於 KG completion，它本身會引入新的噪音，而且通常還需要 seed signal，並不是無痛解法。

所以這篇工作的重點不是去解決「KG 太少」本身，而是去解決「當 KG 不可信時，模型要怎麼穩健地做推薦」。如果要從實用角度看，模型在 KG 不可信時的穩健性，比 KG 豐富時的峰值更有意義。

## Slide 5 — KG Construction

這個資料集的基本規模是 905 個 user、1,399 個 item、22,265 筆互動、3,370 條 KG 邊，以及 2,098 個 unique aspect。

這裡的 KG 是 review-aspect KG，建構流程大致上是先把 review 和 image 相關內容整理成比較乾淨的 item 描述，再從中抽取 aspect，最後建立 item-has-aspect 的邊，並過濾掉太泛用或太吵的 aspect。

我要強調的是，這個 KG 建構管線不是本文主要貢獻。它比較像是我們面對的資料條件。真正重要的是，最後得到的 KG 很稀疏，平均每個 item 只有大約 2.4 條 aspect 邊，而這正好形成我們的方法壓力測試。

## Slide 6 — Failure Mode

這裡我先把現有 KG-aware 方法的失敗模式講清楚。

大多數 KG-aware recommenders 都有一個共通問題：KG entity embedding 會直接進入 message passing，user 和 item 的表示是在一條包含 KG 的路徑上學出來的。這表示只要 KG 本身有問題，雜訊就會一起進到 scoring representation 裡，沒有辦法把 KG 關掉。

這也是為什麼在我們的設定裡，LightGCN 反而會贏。因為 LightGCN 只看 user-item interaction，不會碰到那條不可靠的 KG branch，所以它保留了一個乾淨又安全的 baseline。對這個資料設定來說，避開 KG 污染，比強行融合 KG 更有價值。

## Slide 7 — Research Question

基於剛才的現象，我們提出兩個問題。

第一個是 diagnosis，也就是為什麼 KG-aware 模型會輸給純 LightGCN。第二個是 prescription，也就是什麼樣的設計原則，才能讓 KG 在有用時發揮作用，在不可靠時不污染協同過濾。

我們的答案是：KG 不應該是 scoring pipeline 裡的必經成分，而應該是一條可以被 gate 控制的 side channel。這個想法後面會具體落地在三個設計上，分別是 KG-SVD initialization、softmax rationale masking 和 local-biased fusion gate。

## Slide 8 — Related Work I

先講最基礎的兩個方法。

LightGCN 是純 CF 的強基準。它移除了複雜的 feature transformation 和 nonlinear activation，只保留線性的鄰居聚合和 layer-wise average。對我們來說，它不是普通 baseline，而是 safe default，所以 local view 直接沿用 LightGCN。

KGAT 則代表 KG 深度融合的典型範式。它把 user-item graph 和 KG 合併成一張 collaborative knowledge graph，再透過 entity propagation 來學 user 和 item 表示。這種方法在 KG 豐富時很有效，但在 sparse KG 下，噪音也會一起被傳進去。

所以我們的立場很明確：local view 要保留 LightGCN 的乾淨性，KG 不要進 local propagation。

## Slide 9 — Related Work II

接下來是 contrastive KG methods。

KGCL 會對 KG 結構做擾動，然後對 original view 和 perturbed view 做 contrastive learning。MCCLK 則更進一步，建立 collaborative、semantic、structural 三個視角，彼此做對齊。這些方法在 KG 比較豐富時都很強，但在 sparse KG 下，擾動後剩下的 signal 可能太弱，反而讓對比學習變成在對齊雜訊。

所以我們雖然也有用 contrastive learning，但它只是輔助，權重很小，目的是幫 local 和 global 的幾何空間做輕量對齊，而不是主導融合。

## Slide 10 — Related Work III

KGRec 是跟我們最直接相關的工作。

KGRec 的核心觀察是：不是所有 KG edge 都同樣重要，所以要學 rationale。它做法是對每條 KG edge 算 attention 分數，再用 Bernoulli dropout 去隨機移除高分邊，強迫模型不要過度依賴少數幾條主導邊，之後再用 contrastive learning 強化穩健性。這確實是 rationale-aware recommendation 的代表。

但 KGRec 的假設是，KG 裡面至少有一些 useful edges 可以挑出來。RA-GARK 的前提更保守：我們不假設整個 KG 都可信，所以不是只挑邊，而是把整條 KG channel 變成可閘控的 side channel。

## Slide 11 — Related Work IV

這裡我想補充 gating 的脈絡。

Highway Networks 很早就提出一個很重要的概念：用 gate 把變換路徑和 identity path 做加權，而且 gate 的 bias 可以初始化成偏向安全路徑，讓模型一開始接近 identity，再慢慢學要不要打開變換。MMoE 和 PLE 則是在多任務推薦裡，用 gate 在多個 expert tower 之間做選擇。

但這些方法和我們不一樣的地方有兩個。第一，它們的 expert 多半是同質候選，不是像我們這樣把 CF 和 KG 當成兩條異質訊號管線。第二，它們沒有特別針對「某條管線可能不可信」這件事做安全初始化。

所以在 KG-aware recommendation 領域裡，還是缺少一個 bias-initialized fusion gate。

## Slide 12 — Design Principle

這裡把我們的方法原則講成一句話。

KG 應該是 gateable side channel，而不是 mandatory scoring component。

這個原則帶來三個後果。第一，我們要把 local view 和 global view 分開，避免 KG 污染 CF。第二，融合要晚，等兩邊的 representation 都先學好再決定要不要混。第三，gate 的初始化要偏向 LightGCN，讓模型一開始就站在安全的一邊。

## Slide 13 — Overview

接下來是整體架構。

左邊是 local view，也就是 LightGCN，在 user-item graph 上做線性 propagation。右邊是 global view，它先用 KG-SVD 初始化 aspect slot，再用 softmax rationale masking 動態選出對當前 user-item pair 最有用的 aspect。最右邊是 fusion gate，把 local 和 global 的表示融合起來。最後的訓練損失是 BPR，再加上一個很小的 contrastive regularization。

這張圖最重要的地方是，local 和 global 兩條路線在前面是完全分開的，只有到最後的 scoring stage 才透過 gate 合起來。

## Slide 14 — Problem Setup

我們的任務是 implicit top-K recommendation。對每個 user，要把沒看過的 item 做排序，讓真正互動過的 item 排在前面。訓練時使用正樣本和 sampled negative pairs。

最終分數就是 `u_final` 跟 `i_final` 的內積。`u_final` 和 `i_final` 則是 local 表示和 global 表示的加權和，權重分別由 `alpha_u` 和 `alpha_i` 決定。

這裡有一個重要的 convention：`alpha` 越接近 1，就越偏 local、越像純 CF；`alpha` 越接近 0，就越偏 global、越依賴 KG。

## Slide 15 — Local View

local view 我們直接用純 LightGCN。

這個選擇沒有特別花俏，但非常重要。因為在我們的 setting 裡，LightGCN 本來就已經比所有 KG-aware baseline 還好，所以它是我們必須守住的 safe default。Local view 完全不接觸 KG，這樣就可以保證 collaborative signal 的純淨性。

換句話說，local branch 的職責只有一個，就是把 user-item interaction graph 的訊號學好，其他都不要碰。

## Slide 16 — Local Propagation

local propagation 的部分就是標準 LightGCN。

我們只在 user-item bipartite graph 上做傳播，而且只用 training interactions。沒有 KG edges，也沒有任何額外的 nonlinear transformation。每一層就是 normalized adjacency 乘上 embedding，最後把第 0 層到第 K 層做平均，這裡 K 設成 2。

這個 branch 的輸出就是 `u_loc` 和 `i_loc`，完全是純 collaborative representation。

## Slide 17 — Global View

global view 的重點是 latent aspect slots。

為什麼不直接把 KG triples 拿來傳播？因為我們的 KG 太稀疏了，直接傳播很容易對缺失邊或噪音邊敏感。相反地，我們把每個 item 的 KG 語意壓縮成四個 latent aspect slots，讓模型在一個比較低維、比較穩定的空間裡處理 KG。

這裡的 `A=4` 不是說每個 item 只有四個真實 aspect，而是說我們用四個 latent basis 去表示 item 的 KG semantics。

## Slide 18 — KG-SVD Step 1

KG-SVD 的第一步是先建 item-aspect matrix。

如果 item 有某個 aspect，就把對應位置設成 1。接著做 IDF weighting，公式就是把每個 aspect 乘上它的 IDF。這樣做的原因很簡單：太常出現的 aspect 往往太泛用，不夠有辨識力；比較少見、比較有語意特色的 aspect 應該被保留更高的權重。

所以這一步的目的，就是先把比較有意義的 KG 結構凸顯出來，減少後面 SVD 被 generic aspect 主導。

## Slide 19 — KG-SVD Step 2

這張圖對應的是 KG-SVD 的第二步。

我們對 IDF-weighted matrix 做 truncated SVD，然後把結果投影成 `E_KG = U sqrt(Sigma)`。接著把 flat vector reshape 成每個 item 的四個 aspect slot，每個 slot 維度是 128。最後再把整個 tensor 的 scale 調回跟 Xavier 初始化相容的範圍。

這樣做的目的，是讓 global view 在訓練一開始就有一個有意義的語意幾何，而不是從亂數開始。這也是為什麼隨機初始化會掉分。

## Slide 20 — KG-SVD Ablation

如果把 KG-SVD 初始化改成 random initialization，NDCG@20 會從 0.1238 掉到 0.1173，降幅是 5.3%。

這個結果說明 KG-SVD 不是單純的 preprocessing 小技巧，而是讓 global view 真的能夠在 sparse KG 下站得起來的關鍵。

## Slide 21 — Softmax Masking

global view 的第二個核心是 softmax rationale masking。

對每個 user-item pair，我們會用 user 的 global embedding 去條件化 item 的每個 aspect slot，先算出每個 slot 的 logit，然後用 softmax 加上 temperature 得到權重，最後把四個 slot 加權求和成 `i_glo`。

這樣做的意思是：同一本書對不同 user 可能有不同的推薦理由。有人重視 genre，有人重視 writing style，有人重視 emotional tone，所以 rationale 必須是 user-conditioned 的。

## Slide 22 — Softmax vs Sigmoid

這裡我們特別強調 softmax 而不是 sigmoid。

如果用 sigmoid，每個 slot 是獨立啟動的，容易所有 slot 都偏高，最後退化成平均。softmax 則會讓 slot 之間互相競爭，在固定總量下做選擇。這不只是讓 attention 更 sharp，更重要的是它控制了 `i_glo` 的 magnitude。

這點很重要，因為 global KG channel 本來就是被 gate throttled 的，如果 attention 本身再讓 magnitude 飄掉，整個 gate 的校準就會失真。

## Slide 23 — Softmax Ablation

這張 sensitivity 圖直接對應我們的 ablation 觀察。

把 softmax 換成 sigmoid 之後，NDCG@20 會掉到 0.1151，降幅是 7.0%。這是全文單一元件影響最大的改變。

我這裡要小心地說，不是主張 softmax 在所有情況下都比 sigmoid 好，而是說在我們這個「被 gate 控制的 sparse KG side channel」裡，normalization 的選擇會直接影響穩定性和 magnitude control。

## Slide 24 — Fusion Gate

接下來是 fusion gate。

`alpha_u` 和 `alpha_i` 是用小型 MLP 算出來的，然後把 local 和 global 的表示做加權和。這一層的任務不是挑 aspect，而是決定這條異質視角管線到底要用多少 KG。

這也是為什麼我們前面說，aspect selection 用 attention，view fusion 用 gate。因為它們處理的是兩種不同層次的選擇問題。

## Slide 25 — Gate Bias

fusion gate 最關鍵的設計是 bias initialization。

我們把最後一層 bias 設成 +5，所以一開始 `alpha_0` 大約是 0.993。這代表模型訓練一開始幾乎就是純 LightGCN，global view 只佔很小比例。

這個設計不是為了保守而保守，而是因為 sparse KG 本來就不可靠。讓模型一開始站在安全的一邊，可以避免 global branch 的不成熟表示干擾 local branch。

## Slide 26 — Graceful Degradation

這一頁想強調的是 graceful degradation。

如果 KG 沒有提供有用訊號，gate 可以一直維持在接近 1 的位置，模型就會自然退化成接近 LightGCN 的行為。這是一個架構層級的 fallback，不是靠運氣。

消融結果也支持這件事。把 gate bias 改成 0 之後，NDCG@20 會降到 0.1173，掉 5.3%。這說明 local-biased initialization 不是訓練小技巧，而是保護 CF backbone 的機制。

## Slide 27 — Contrastive Regularization

除了主要的 BPR loss，我們還加了兩個很小的 contrastive regularization。

第一個是 aspect-level 的對比損失，第二個是 user cross-view 的對比損失。這兩個都只是輔助對齊 local 和 global 的幾何空間，不是主要的融合機制。

我們的設計很保守：權重只有 0.005，而且 KG 側做了 stop-gradient，避免對比學習把 SVD 保留下來的語意幾何拉壞。再加上 projection head，讓 CL 的梯度不要直接影響 scoring space。

## Slide 28 — Training and Evaluation

訓練設定是 Adam，learning rate 0.001，batch size 128，最多 80 個 epoch，用 validation NDCG@20 做 early stopping，patience 是 10。

評估時採 full-ranking，會排除訓練集裡已經互動過的 item，最後看 HR、Precision、Recall、F1、MAP 和 NDCG，這些都取 @20。

從效能來看，我們每個 epoch 大概 1.5 秒，跟 KGRec 差不多，所以這個設計沒有讓成本爆炸。

## Slide 29 — Main Results

先看主結果。

MCCLK、KGCL、KGAT、KGRec 在這個 sparse KG 設定下都低於 LightGCN。LightGCN 是 0.1179，而 RA-GARK 是 0.1238。

這表示兩件事。第一，我們的設計確實把 KG 從原本的 net-negative 變成 net-positive。第二，跟 KGRec 相比，我們在同樣稀疏 KG 的條件下，還能再提升 13.1%。

這個結果其實直接支持了我們最一開始的設計原則：KG 不是必經管線，而是可被 gate 控制的 side channel。

## Slide 30 — Ablation Summary

再看 ablation。

如果拿掉 softmax，NDCG@20 會掉到 0.1151。拿掉 local-biased gate init，會掉到 0.1173。拿掉 KG-SVD init，也會掉到 0.1173。

這三個結果一起告訴我們，RA-GARK 的增益不是來自「加了 KG」這件事本身，而是來自 KG 的初始化、selection 和 gating 這三個設計一起合作。

## Slide 31 — Case Study and Takeaways

這張 heatmap 是 case study。

你可以看到不同 item 會對不同 aspect slot 給出不同的權重，表示 rationale masking 不是固定平均，而是真的有在對不同 item 使用不同的語意路徑。這也是這個模型可解釋性的來源之一。

從方法論角度看，這個 case study 也支持一件事：模型不只是準確，還能告訴我們「它到底用了哪個 slot 來做判斷」。

## Slide 32 — Conclusion

最後總結一下。

在 sparse KG recommendation 裡，關鍵不是把 KG 硬塞進模型，而是給模型一個可靠的方式去忽略 KG，當 KG 不可靠的時候能退回到安全的 collaborative filtering，當 KG 有用的時候再把它打開。

這篇工作的主要貢獻有四個：第一，提出 gateable KG side channel；第二，提出 KG-SVD initialization；第三，提出 softmax rationale masking；第四，提出 local-biased fusion gate。

限制也很清楚：我們目前只在一個 sparse review-aspect KG dataset 上驗證，KG construction pipeline 不是本文的主貢獻，另外在 dense KG setting 下，深度融合方法仍然可能更強。也就是說，RA-GARK 的定位不是要取代所有 KG-aware 方法，而是要補上 sparse KG 這個常被忽略、但很現實的場景。

以上，謝謝大家。
