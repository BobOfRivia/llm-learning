# Inference-time Scaling

> deadline: 2026-07-26
> status: pending
> 覆盖 tasks: Inference-time Scaling 机制、Best-of-N 采样的数学、Chain-of-Thought 为什么有效
> 桥接: capabilities/inference-time-compute.md, topics/inference-time-compute.md, topics/chain-of-thought.md
> 剔除：MCTS 在语言推理中的深入 — 2024 火过一阵，但 2025 主流是 long-CoT RL 训练（参见 blog 8），inference-time MCTS 已边缘化。降级为一段对比说明，不深入。详见 logs/removed-from-mainline.md

**核心问题**：算力分配的"第三种范式"（训练时 → 参数时 → 推理时）— 这个 framing 能站住吗？2024 年 o1 出来时业界一度认为 inference-time 会持续作为独立维度增长（MCTS、Tree-of-Thought 等都被押注），但 2025 年回头看，**真正赢的是把推理"内化"到模型里（long-CoT RL 训练），而不是外挂搜索结构**。要能讲清：

1. **Inference-time scaling 的回报曲线**：Best-of-N 是 log-linear 的（边际效益快速递减）；long-CoT 是模型决定何时停（自适应），曲线形状不一样。
2. **Best-of-N 的本质**：N 越大质量越高，但被验证器质量限制（reward model 错了就放大错误）。要能用 verifier 质量推 Best-of-N 的有效天花板。
3. **CoT 为什么对某些任务有效对另一些没用**：CoT 给的是"让模型分配更多 token 在中间步骤"的能力，对需要多步分解的任务（数学、代码）有效，对单步知识检索（事实问答）反而可能引入噪声。
4. **MCTS / Tree-of-Thought 为什么没赢**：路径展开的成本太高、需要好的 value function（很难训）、和 long-CoT RL 的端到端训练相比缺乏 scaling 路径。这一段作为反例理解就够，不深入。

**判断主线**：inference-time 作为独立 scaling 维度仍然存在（o3 的 low/medium/high compute 就是这个维度），但**主导形式是模型自决定 thinking budget（long-CoT 内化），而不是外部搜索结构**。看懂这个区分，再看新论文就不会被各种 inference-time tricks 迷惑。

---


---
