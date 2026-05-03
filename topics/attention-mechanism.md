# Attention 机制

> 锚点：`tracks/architecture.md` 阶段 1 / `eras/era1-scaling.md`

## 这个概念是什么

Attention 是 Transformer 的核心运算——让序列中的每个位置能够"关注"所有其他位置，以加权方式聚合信息。它同时解决了 RNN 的两个根本缺陷：梯度消失导致的长程依赖困难，以及串行计算导致的并行化瓶颈。理解 Attention 是理解后续所有架构演进（Flash Attention、GQA、MLA、甚至 MoE 的 router 设计）的基础。

## 内部结构

### Scaled Dot-Product Attention

> 待推导：
> - Q、K、V 矩阵的来源（输入经线性投影得到）
> - Q = "我在找什么"、K = "我能提供什么"、V = "我实际携带的信息"的直觉
> - 为什么用点积衡量相关性（而非加性注意力等其他方案）
> - 为什么除以 √d_k（softmax 的梯度饱和问题）
> - softmax 将分数归一化为概率分布的意义

### Multi-Head Attention

> 待推导：
> - 多头的动机：单一注意力模式不够，需要在不同子空间捕获不同关系
> - 每个头的 d_k = d_model / h 的设计——总计算量与单头相同
> - 输出拼接 + 线性投影的意义
> - 实证：不同头实际学到了什么（Voita et al., 2019 的头剪枝研究）

### Transformer 层的完整结构

> 待推导：
> - Pre-LN vs Post-LN：为什么 Post-LN 在深层网络中训练不稳定
> - 残差连接如何保证梯度流（训练 100+ 层成为可能）
> - FFN 的结构（两层线性变换 + 激活函数）
> - FFN 的参数量约占整层的 2/3，计算量是主体

### FFN 作为键值记忆

> 待推导：
> - Geva et al. (2021) 的核心观察：FFN 第一层 = pattern detector (keys)，第二层 = knowledge storage (values)
> - 这个框架对理解"Transformer 如何存储事实"的启示
> - 局限性：不是所有 FFN 神经元都对应可解释的"记忆条目"

### Attention 的计算复杂度

> 待推导：
> - 时间复杂度 O(N²·d)、内存复杂度 O(N²) 的来源
> - 这为什么是长上下文的根本瓶颈（128K tokens 的 attention 矩阵有多大）
> - 各种"高效 attention"方案都在试图打破这个二次方

## 当前状态（截至 2026-05）

> 待填入：Flash Attention（IO 优化，不改变复杂度但大幅加速）、GQA（减少 KV 头）、MLA（低秩压缩 KV）三个方向如何分别解决计算、推理内存、KV Cache 三个瓶颈。

## 关键权衡

> 待填入：
> - 全局 Attention（精确检索任意位置）vs SSM（线性但压缩历史信息）
> - 精确 Attention vs 近似 Attention（Sparse / Linear Attention）的质量代价
> - 更多头 vs 更少头（GQA 的质量-效率曲线）

## 信息源

- [Attention Is All You Need (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762)
- [Transformer Feed-Forward Layers Are Key-Value Memories (Geva et al., 2021)](https://arxiv.org/abs/2012.14913)
- [Are Sixteen Heads Really Better than One? (Voita et al., 2019)](https://arxiv.org/abs/1905.10650)
- Jay Alammar, [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)

## 更新日志

- 2026-05-03：创建骨架
