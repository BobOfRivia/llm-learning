# 稀疏、线性、Hybrid Attention

> deadline: 2026-06-07
> status: pending
> 覆盖 tasks: NSA（Native Sparse Attention）、MoBA（Mixture of Block Attention）、Gated DeltaNet（线性注意力）、Hybrid 3:1 模式（Qwen3-Next / Nemotron-H）
> 桥接: tracks/architecture.md, topics/attention-mechanism.md（待新增 sparse-attention 小节）
> 前序: blog 2 [attention-engineering](2026-05-31-attention-engineering.md) 处理稠密路线（Flash / GQA / MLA）

**核心问题**：当稠密 Attention 把能压缩的都压完之后（GQA 砍头数、MLA 砍秩），下一步必然走向"减少要算的 token 对"或"用线性算子近似 softmax"。2025 年这两条路都真实落地了 — DeepSeek V3.2-Exp 上的 NSA、Moonshot 的 MoBA 是稀疏路线；Qwen3-Next 80B-A3B 与 NVIDIA Nemotron-H/Nano 2 的 Gated DeltaNet + Full Attention 3:1 交替是 Hybrid 路线。要能解释清楚四件事：

1. **稀疏 Attention 的"原生可训练"为什么是关键**：早期稀疏（Longformer、BigBird）只能在推理阶段近似，质量损失明显；NSA 把稀疏决策做进训练的前向反向（hardware-aligned + natively trainable），所以质量不掉。这是从"事后近似"到"原生设计"的路径切换。
2. **NSA 的三路并行**：压缩粗粒度 token + 选择细粒度 token + 局部滑动窗口 — 三条 attention 路径同时跑，覆盖全局/局部/精确三种需求。和单一稀疏模式（Longformer 只有滑动 + 全局）的区别是什么。
3. **Gated DeltaNet 不是 Mamba 的复刻**：它是 delta rule + gating 的线性注意力变体（源自《Gated Delta Networks: Improving Mamba2 with Delta Rule》），跟纯 SSM 的状态空间路径不一样。Hybrid 路线选它而非 Mamba-2 的原因。
4. **3:1 比例为什么是甜点**：每三层线性 + 一层 full attention 的工程经济学 — 为什么不是 7:1（更激进）或 1:1（更保守）。这个比例是 Qwen3-Next 和 Nemotron-H 各自实验出来的还是相互参考。

**判断主线**：从 Flash Attention（2022，IO 优化，算法等价）→ GQA / MLA（2023-24，KV 压缩，质量微损）→ NSA / Hybrid（2025，结构改变，质量持平）— 这条线的驱动力一直是"长上下文经济性"，但每一步换的杠杆不同。能用一句话概括每一步换了什么杠杆，就掌握了这条线的因果。

---
