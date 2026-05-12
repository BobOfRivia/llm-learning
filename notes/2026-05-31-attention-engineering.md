# 稠密 Attention 工程化

> deadline: 2026-05-31
> status: pending
> 覆盖 tasks: Flash Attention、GQA / MQA、MLA（多头潜在注意力）
> 桥接: tracks/architecture.md, tracks/inference.md, topics/attention-mechanism.md
> 后续: blog 3 [attention-sparse-hybrid](2026-06-07-attention-sparse-hybrid.md) 处理稀疏 / 线性 / Hybrid 路线

**核心问题**：稠密 Attention 这一支的演进，算法形态没变（仍是 softmax(QK^T/√d)V），变的是经济学约束（显存带宽 / KV Cache 成本 / 长上下文成本曲线）— 这个 framing 站得住吗？要能解释 Flash Attention 解决的是 IO 瓶颈而非计算瓶颈、GQA 与 MLA 压缩的对象本质不同（KV 头数 vs KV 表示秩）、为什么 MLA 出现在 DeepSeek 而不是别人手里（架构-训练-硬件的 co-design 路径依赖）。

---
