# Native Sparse Attention (NSA) 由浅入深完整笔记

> 论文：[Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention](https://arxiv.org/abs/2502.11089)
> 作者：DeepSeek-AI（第一作者 Jingyang Yuan，2025 年 2 月）
> 荣誉：ACL 2025 最佳论文

---

## Part 1（最浅）：用一句话理解 NSA

> **NSA 让模型在阅读长文本时，像人一样：先扫摘要、再精读重点、同时盯住眼前几段，三者并行；并且这套机制从预训练第一步就开始学，而不是事后强行套上去。**

如果只记一句话，就记这句。下面层层展开。

---

## Part 2（浅）：为什么需要 NSA？

### 2.1 Transformer 的根本痛点

标准 attention 里，每个 token 要和它前面**所有** token 算一次相关性。序列长度 n，计算量就是 O(n²)：

- n = 1k 时，n² = 100 万 —— 没问题
- n = 64k 时，n² = 40 亿 —— 灾难

在 64k 长上下文场景下，**attention 一项就吃掉总推理时延的 70–80%**。整个代码库、几百页 PDF、长链 reasoning（如 R1、o1）越火，这个瓶颈越要命。

### 2.2 自然的想法：稀疏注意力

观察事实：softmax 算出来的注意力分布通常很集中——绝大多数 token 的权重接近 0。既然如此，何必把所有 token 都算一遍？只挑重要的就行。

这就是**稀疏注意力（Sparse Attention）**。

### 2.3 但已有的稀疏方案都"差一口气"

NSA 论文（第 2 节）把前人方法的问题归纳成两大类：

#### 问题一：「高效推理」是个幻觉（The Illusion of Efficient Inference）

很多方法理论 FLOPs 下降了，**但实际 GPU 上并不快**。两个原因：

- **阶段受限**。H2O 只在 decoding 时稀疏，prefilling 还要算完整 attention map；MInference 反过来只优化 prefilling。两边总有一边没省。
- **和现代架构（GQA / MQA）打架**。GQA 让多个 query head 共享一组 KV。如果稀疏方法让每个 head 独立选 token（如 Quest），那么一个 GQA group 内 16 个 head 选出来的 token 的**并集**才是真正要从显存加载的——计算稀疏了，访存却没稀疏。

#### 问题二：「可训练的稀疏」是个神话（The Myth of Trainable Sparsity）

绝大多数稀疏方法只在推理时启用，模型在训练时从没见过"被稀疏化的注意力分布"。论文里有个让人警醒的数据：**预训练好的 dense 模型里，top 20% 的注意力分数只能覆盖 70% 的总权重**——也就是说，简单地砍掉 80% 的低分 token，会损失 30% 的信息。这就是后挂式稀疏一上线就掉点的根本原因。

更糟的是，已有"可训练"方案也有硬伤：
- **不可微**：k-means 聚类、SimHash 这类离散操作，梯度过不去。
- **反向慢**：token 级稀疏选择导致访存碎片化，FlashAttention 的块状优势用不上，训练比 dense 还慢。

### 2.4 NSA 的两条核心主张

正是基于上述分析，DeepSeek 主张：

1. **Hardware-Aligned**（硬件对齐）——算法必须按 GPU 的脾气设计，理论加速要落到墙钟时间。
2. **Natively Trainable**（原生可训练）——从预训练第一步就用稀疏 attention，让模型自己学怎么挑、怎么用。

这两点就是论文标题的全部含义。

---

## Part 3（中）：NSA 的核心思想 —— 三条路并行

### 3.1 直觉：像人读长文档

设想你拿到一本 500 页的书要回答问题：

1. **翻目录、扫摘要** → 对全书有个粗略印象 ✏️ *Compression*
2. **跳到最相关的几章精读** → 不漏关键细节 🔍 *Selection*
3. **眼前正在读的几段也得连贯** → 不能丢上下文 📖 *Sliding Window*

NSA 就是把这三种行为做成**三条并行的 attention 分支**，对同一个 query 同时跑，最后用一个**学习出来的门控（gate）** 把三路输出加权融合。

### 3.2 整体框架（公式 5）

对当前 query $\mathbf{q}_t$，最终输出是：

$$
\mathbf{o}^*_t = \sum_{c \in \{\text{cmp}, \text{slc}, \text{win}\}} g_t^c \cdot \mathrm{Attn}(\mathbf{q}_t, \tilde{K}_t^c, \tilde{V}_t^c)
$$

其中：
- $c$ 是三个分支（compression / selection / window）
- $\tilde{K}_t^c, \tilde{V}_t^c$ 是每个分支构造出来的"精简版" key/value
- $g_t^c \in [0, 1]$ 是门控分数，由输入特征经过 MLP + sigmoid 算出

只要保证 **重映射后的总 token 数 $N_t \ll t$**，就实现了稀疏。

### 3.3 三个分支详解

#### 分支 1：Compression（粗粒度全局——看摘要）

把前面所有 key 按块（block）切。设块长度 $l = 32$、步长 $d = 16$（相邻块有重叠，防止信息断裂）。每个块通过一个**小 MLP $\varphi$（带块内位置编码）** 压成 1 个向量：

$$
\tilde{K}_t^{\text{cmp}} = \left\{ \varphi(\mathbf{k}_{id+1:id+l}) \,\middle|\, 1 \le i \le \lfloor (t-l)/d \rfloor \right\}
$$

64k 个 token 经过 $d=16$ 的压缩，只剩约 4000 个"摘要向量"。query 对这些摘要做 attention，得到全局视野，但成本骤降。

> ⚡ 关键：$\varphi$ 是**可学习的**，不是简单平均池化。模型自己学怎么"提炼一段话的要点"。

#### 分支 2：Selection（细粒度精确——精读）

光看摘要会丢细节。所以再挑出最重要的若干个**原始 block**，对里面的原始 token 做 full attention。

**怎么挑？** 这是 NSA 最巧妙的地方。直接计算每个 block 的重要度太贵——但 NSA **复用 Compression 分支已经算出来的注意力分数**：

$$
\mathbf{p}_t^{\text{cmp}} = \mathrm{Softmax}(\mathbf{q}_t^\top \tilde{K}_t^{\text{cmp}})
$$

当 compression block 和 selection block 大小相同时，可以直接 $\mathbf{p}_t^{\text{slc}} = \mathbf{p}_t^{\text{cmp}}$。如果不同（论文里 $l=32, l'=64$），用公式 9 做求和映射：

$$
\mathbf{p}_t^{\text{slc}}[j] = \sum_{m=0}^{l'/d - 1} \sum_{n=0}^{l/d - 1} \mathbf{p}_t^{\text{cmp}}\left[\frac{l'}{d}j + m + n\right]
$$

**为 GQA 做的关键调整**——同一个 KV-group 内的所有 query head 必须选同一批 block，所以要把组内所有 head 的分数加起来：

$$
\mathbf{p}_t^{\text{slc}'} = \sum_{h=1}^{H} \mathbf{p}_t^{\text{slc}, (h)}
$$

这一步是访存效率的关键：组内 16 个 head 共享同一份 KV，一次加载、所有 head 复用。

最后取 top-n 个 block（论文中 n=16，其中固定包含 1 个起始 block 和 2 个局部 block）：

$$
\mathcal{I}_t = \{i \mid \mathrm{rank}(\mathbf{p}_t^{\text{slc}'}[i]) \le n\}
$$

把这些 block 内的原始 token 拼起来：

$$
\tilde{K}_t^{\text{slc}} = \mathrm{Cat}\left[\{\mathbf{k}_{il'+1:(i+1)l'} \mid i \in \mathcal{I}_t\}\right]
$$

#### 分支 3：Sliding Window（局部——眼前的页）

最简单的一支：保留最近的 $w$ 个 token（论文中 $w = 512$）：

$$
\tilde{K}_t^{\text{win}} = \mathbf{k}_{t-w:t}, \quad \tilde{V}_t^{\text{win}} = \mathbf{v}_{t-w:t}
$$

**为什么必须单独留一支？** 这是论文里很容易被忽略但极其重要的设计动机：

> 局部 token 的 attention 分数往往非常大。如果不把它隔离出去，模型在训练时会形成"shortcut learning"——直接靠近邻 token 就能拿到大部分梯度信号，**导致 Compression 和 Selection 两支学不到东西**。

把局部隔离成独立分支后，另外两支被"强制"去学真正的长程依赖。这是 NSA 性能能超过 full attention 的隐藏原因之一。

**额外细节**：三个分支用**各自独立**的 K/V 投影矩阵（不共享）。论文说这是为了"以微小代价进一步避免梯度干扰"。

### 3.4 完整数据流图

```
                    ┌──────────────────────────────────────────┐
                    │             query  q_t                   │
                    └──────────────────────────────────────────┘
                                       │
              ┌────────────────────────┼─────────────────────────┐
              ▼                        ▼                         ▼
    ┌──────────────────┐  ┌──────────────────────┐  ┌──────────────────────┐
    │  Compression     │  │   Selection          │  │   Sliding Window     │
    │  (摘要)          │  │   (精读)             │  │   (近邻)             │
    │                  │  │                      │  │                      │
    │ MLP φ 把每块     │  │ 复用 cmp 的分数      │  │ 最近 w=512 个       │
    │ 32 token → 1 向量│  │ → 挑 top-16 块       │  │ 原始 token           │
    └──────────────────┘  └──────────────────────┘  └──────────────────────┘
              │                        │                         │
              ▼                        ▼                         ▼
        Attn(q, K̃_cmp)         Attn(q, K̃_slc)            Attn(q, K̃_win)
              │                        │                         │
              └──────────┬─────────────┴─────────────┬───────────┘
                         │       学习的门控          │
                         │   g_cmp, g_slc, g_win    │
                         └────────────┬──────────────┘
                                      ▼
                                   o*_t（最终输出）
```

---

## Part 4（深）：Kernel 设计 —— 为什么 NSA 在 GPU 上真的快

这是 NSA 最被低估的部分，也是它和"纸面稀疏"方法的真正分水岭。

### 4.1 GPU 性能的底层约束：Arithmetic Intensity

**Arithmetic Intensity（算术强度）** = 计算操作数 / 访存字节数。

每张 GPU 都有一个临界值（peak FLOPS / memory bandwidth）：
- 算术强度 **高于**临界值 → compute-bound（被算力限制）
- 算术强度 **低于**临界值 → memory-bound（被访存限制）

**Attention 在不同阶段的瓶颈不一样**：

| 阶段 | 特征 | 瓶颈 | 优化目标 |
|------|------|------|---------|
| 训练 / Prefilling | 批量矩阵乘 | compute-bound | 减少 FLOPs |
| Decoding | 一次生成 1 个 token，但要加载整个 KV cache | **memory-bound** | 减少访存 |

NSA 必须在两种情形下都快——这就要求算法本身平衡好算术强度。

### 4.2 为什么必须是「块级」而不是「token 级」选择？

GPU 的 HBM → SRAM 数据搬运：**连续大块**远比**零散小段**高效（合并访存 + Tensor Core 利用率）。这就是 FlashAttention 的核心思想。

如果稀疏方法挑出来的 token 散落在序列各处（如 HashAttention 这种 token 级稀疏），每次都要做小块碎读，GPU 利用率会跌到 20% 以下。**NSA 的所有选择粒度都是 block（默认 64 token 一块）**，每次搬运一大段连续内存，从根本上保证了硬件友好。

### 4.3 为什么必须和 GQA 协同？

在 GQA 架构里，1 组 KV 被 H 个 query head 共享。

- **天真的方案**（如 Quest）：每个 head 独立选 top-k → 组内 H 个 head 的并集才是真正要加载的 KV 量 → 访存几乎没省。
- **NSA 的方案**：把组内 H 个 head 的重要度分数**先加起来**（公式 10），让组内 head **选同一批 block** → 这批 KV 只加载一次，H 个 head 复用。

这是从「计算稀疏」走到「真正访存稀疏」的关键一步。

### 4.4 Triton Kernel 的三大特征

NSA 的 selection-attention kernel（论文 Figure 3）和 FlashAttention 有本质不同：

**FlashAttention 的策略**：把连续的 query block 一起加载到 SRAM。
**NSA 不能这么做**——因为同一个 query block 内的不同 query 可能选了**完全不重叠的 KV block**（它们的 top-n 集合不一样），按 query 分块只会反复加载不同的 KV。

NSA 反其道而行：

1. **Group-Centric Data Loading**（按 group 加载 query）
   每次 inner loop，把**单个 query 位置上整个 GQA group 的 H 个 head 的 query** 一起加载到 SRAM。它们共享同一组稀疏 KV 索引 $\mathcal{I}_t$。

2. **Shared KV Fetching**（共享 KV 读取）
   按 $\mathcal{I}_t$ 顺序加载连续的 K/V block 到 SRAM——每个 block 加载一次，组内所有 head 复用。

3. **Outer Loop on Grid**（外循环交给调度器）
   由于不同 query 位置选的 block 数 n 几乎相同，把 query 维度的外循环交给 Triton 的 grid scheduler，自动做负载均衡。

**结果**：算术强度被精心平衡，eliminating redundant KV transfers，Tensor Core 几乎打满。

### 4.5 实际速度（A100 上）

64k 序列长度下，对比 FlashAttention-2：

| 阶段 | 加速比 | 原因 |
|------|-------|------|
| **Decoding** | **11.6×** | 访存量从 65536 降到 5632 token（见下表） |
| **Forward** | **9.0×** | 块级稀疏 + group-centric kernel |
| **Backward** | **6.0×** | 反向也用同一套稀疏路径，原生可训练 |

**Decoding 时实际访存量对比**（论文 Table 4）：

| Context Length | 8192 | 16384 | 32768 | 65536 |
|---------------|------|-------|-------|-------|
| Full Attention | 8192 | 16384 | 32768 | 65536 |
| **NSA** | 2048 | 2560 | 3584 | **5632** |
| Expected Speedup | 4× | 6.4× | 9.1× | **11.6×** |

序列越长，NSA 优势越大——因为 compression 摘要数量是次线性增长的。

---

## Part 5（深）：实验结果细读

### 5.1 训练设置

- 模型：27B 参数 MoE + GQA（3B 激活）
- 30 层 / hidden 2560 / 64 head / 4 GQA group
- $d_q = d_k = 192$, $d_v = 128$
- NSA 超参：$l=32, d=16, l'=64, n=16, w=512$
- 训练数据：270B token，先 8k 后用 YaRN 扩到 32k

### 5.2 通用 Benchmark（短文本，Table 1）

即使在短文本上（NSA 的稀疏优势用不出来），9 个 benchmark 中 **NSA 在 7 个上超过 full attention**：

| 任务 | Full Attn | NSA | 提升 |
|------|----------|-----|------|
| MMLU | 0.567 | 0.565 | -0.002 |
| MMLU-PRO | 0.279 | 0.286 | +0.007 |
| BBH | 0.497 | **0.521** | +0.024 |
| GSM8K | 0.486 | **0.520** | +0.034 |
| DROP | 0.503 | **0.545** | +0.042 |
| **Avg.** | 0.443 | **0.456** | **+0.013** |

> **为什么稀疏反而更强？** 论文给的解释：稀疏预训练**强迫模型聚焦最重要的信息**，过滤了无关注意力路径的噪声，是一种隐式正则化。

### 5.3 长上下文（Table 2，LongBench）

把所有稀疏方法的活跃 token 数都设为 2560（NSA 在 32k 下的平均值）做公平对比：

| Method | Avg. |
|--------|------|
| H2O | 0.303 |
| InfLLM | 0.383 |
| Quest | 0.392 |
| Exact-Top（理论上限） | 0.423 |
| Full Attention | 0.437 |
| **NSA** | **0.469** |

NSA 不仅超过所有稀疏 baseline，甚至 **超过了 Exact-Top（每步都拿到精确的 top-n）**——这说明问题不只在"选得准不准"，而在"模型有没有被训练成能利用稀疏 pattern"。原生训练是决定性因素。

特别亮眼：多跳问答（HPQ +0.087, 2Wiki +0.051）、code 理解（LCC +0.069）。

### 5.4 Needle-in-a-Haystack（64k）

NSA 在 64k 上达到**完美检索准确率**（图 5 全绿）。这是稀疏方法最容易翻车的地方——一旦"针"所在的 token 被剪掉就找不到。NSA 的层级设计（先 compression 定位、再 selection 精读）天然适合这种任务。

### 5.5 长链推理（AIME 24，Table 3）

用 DeepSeek-R1 蒸馏出 10B 长 reasoning 数据做 SFT：

| Token Limit | Full Attn-R | NSA-R | 差距 |
|------------|-------------|-------|-----|
| 8192 | 0.046 | **0.121** | +0.075 |
| 16384 | 0.092 | **0.146** | +0.054 |

NSA-R 在 8k 上几乎是 Full Attention-R 的 **2.6 倍**——稀疏注意力非但没拖累 reasoning，反而加强了。

---

## Part 6（深）：消融与设计取舍

### 6.1 为什么不用聚类（如 ClusterKV）？

NSA 论文的 6.1 节明确说他们试过聚类方案，但有三个硬伤：
1. 动态聚类本身开销大；
2. 聚类不均衡，在 MoE 系统里和 Expert Parallel 撞车，负载失衡；
3. 需要周期性重聚类，强制 chunk-sequential 训练协议——和现代 LLM 训练框架不兼容。

### 6.2 为什么不用 Quest 那种 query-aware block 选择？

试过两种变体：
1. **辅助 loss 学块重要度**：选择不可微，loss 帮不上太多，反而拖累 loss 曲线。
2. **启发式 min-max**（Quest 风格）：召回率低，loss 显著差于 NSA。

NSA 的精妙之处：**复用 compression 分支的 softmax 分数**作为 selection 的重要度信号——既可微，又零额外开销，还和后续 attention 计算完全一致。

### 6.3 Attention map 的可视化（图 8）

论文可视化了 dense attention 的真实分布，发现注意力分数确实呈**块状聚集**——相邻 key 的分数相似。这是 block-level 稀疏在统计上能成立的根本依据。

---

## Part 7：总结与延伸

### 7.1 一句话再回顾

> NSA = **块级层级稀疏（compression + selection + window 三分支）** + **GQA 协同的 group-centric kernel** + **原生可训练**。三者缺一不可。

### 7.2 NSA 解决了什么、没解决什么

**解决了**：
- 长上下文训练 / 推理双端加速（不是只优化一端）
- 稀疏 + GQA 协同，让访存真正稀疏
- 可微、原生预训练，避免后挂掉点
- 在 needle-in-haystack、长链 reasoning 上保持/超越 full attention

**仍是开放问题**：
- 超参（$l, d, l', n, w$）目前是手调的——能否随训练自适应？
- 27B MoE 规模已验证，更大规模（百 B 级别）的 scaling law 行为还需观察
- 与 RoPE / 长上下文外推（YaRN 等）的耦合关系还可进一步研究
- 三分支门控的可解释性还有探索空间

### 7.3 推荐阅读顺序（如果你要再回头啃论文）

1. **§1 + Figure 1**：动机和效果一图概览
2. **§2 Rethinking**：理解为什么前人不够好——这是 NSA 设计的全部 motivation
3. **§3.2 + Figure 2**：三分支整体框架（公式 5 是灵魂）
4. **§3.3.1–3.3.3**：三个分支的公式，对照 Part 3 的类比
5. **§3.4 + Figure 3**：kernel 设计——理解 group-centric loading 为什么是关键
6. **§5 Efficiency Analysis**：看真实数字
7. **§6 Discussion**：看作者踩过哪些坑，反向理解 NSA 为什么长这样

---

## 附录：核心超参速查

| 符号 | 含义 | 论文默认值 |
|------|------|----------|
| $l$ | Compression block 长度 | 32 |
| $d$ | Compression block 步长 | 16 |
| $l'$ | Selection block 长度 | 64 |
| $n$ | Top-n selection block 数 | 16（含 1 initial + 2 local） |
| $w$ | Sliding window 大小 | 512 |
| $H$ | GQA group 内 head 数 | 16（64 head / 4 group） |
| $d_k, d_v$ | key/value 维度 | 192 / 128 |
