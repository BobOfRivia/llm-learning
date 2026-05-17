# 稠密 Attention 工程化

> deadline: 2026-05-31
> status: pending
> 覆盖 tasks: Flash Attention、GQA / MQA、MLA（多头潜在注意力）
> 桥接: tracks/architecture.md, tracks/inference.md, topics/attention-mechanism.md
> 后续: blog 3 [attention-sparse-hybrid](2026-06-07-attention-sparse-hybrid.md) 处理稀疏 / 线性 / Hybrid 路线

**核心问题**：稠密 Attention 这一支的演进，算法形态没变（仍是 softmax(QK^T/√d)V），变的是经济学约束（显存带宽 / KV Cache 成本 / 长上下文成本曲线）— 这个 framing 站得住吗？要能解释 Flash Attention 解决的是 IO 瓶颈而非计算瓶颈、GQA 与 MLA 压缩的对象本质不同（KV 头数 vs KV 表示秩）、为什么 MLA 出现在 DeepSeek 而不是别人手里（架构-训练-硬件的 co-design 路径依赖）。

## 成本曲线 framing 细化

"上下文成本曲线"不是一根曲线，而是把上下文长度 n 作为横轴时同时存在的**一束曲线**。Blog 开头需要把它们显式立起来，否则"经济学约束"会显得空泛，读者会问"成本到底指哪个成本"。

四根需要分清的曲线：

- **计算成本** ~ O(n²)：QK^T 的浮点运算量
- **显存峰值** ~ 朴素实现 O(n²)：要把整张 attention 矩阵物化到 HBM
- **KV Cache 显存** ~ O(n · d · L · H)：推理时随生成长度线性堆积
- **显存带宽 / IO 成本**：decoding 时每步都要把 KV Cache 从 HBM 搬到 SRAM——长上下文 decoding 的真实瓶颈

三种技术各自动的是不同曲线：

- **Flash Attention** 不改 O(n²) 的渐近复杂度，通过 tiling + 重计算把"显存峰值"从 O(n²) 压到 O(n)，同时大幅压低"IO 成本"的常数项——贡献是 **IO 瓶颈**而非计算瓶颈
- **GQA / MQA** 改 KV Cache 曲线的**斜率**：通过减少 KV 头数 H
- **MLA** 也改 KV Cache 曲线，但走**压缩表示秩**的路径（低秩潜在向量），与 GQA 在 H 维度上正交

Blog 结构暗示：开头立起这束曲线 → Flash / GQA / MLA 三节各自对应"我在改哪一根、改的是斜率/截距/渐近阶"。这样"算法形态不变、动的是成本曲线"才有抓手。

---
kv-cache[https://zhuanlan.zhihu.com/p/662498827]
MLA 知乎[https://zhuanlan.zhihu.com/p/16730036197]
MLA 苏神[https://spaces.ac.cn/archives/10091]

MLA可以理解原理，但是距离从算法层面完全理解还有距离