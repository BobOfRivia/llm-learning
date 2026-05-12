# MoE 架构与三家路线对比

> deadline: 2026-06-14
> status: pending
> 覆盖 tasks: Expert 路由机制、Auxiliary-loss-free 负载均衡、Multi-Token Prediction（MTP）、Wide Expert Parallel 服务
> 桥接: tracks/architecture.md, topics/moe.md

**核心问题**：MoE 解决的真问题是什么 — 是 scaling 极限、推理成本、训练成本，还是别的？为什么 2024 年才大规模回归而不是 2021 Switch Transformer 之后？2025 年三家开源 MoE 实际上做了**完全不同的路由选择**，这些差异不是细节，是路线分歧：

- **DeepSeek V3 / V3.2**：1 个共享专家 + 8 个路由专家（从 256 池选），fine-grained shared expert 思路
- **Qwen3 MoE**：top-8 / 128，**不要共享专家**（Qwen2.5-MoE 时还有，Qwen3 砍掉）
- **Llama 4**：1 共享 + 1 路由（Maverick 从 128 选，Scout 从 16 选）

要能回答：
1. **共享专家是必要的吗**？DeepSeek 和 Llama 4 都保留，Qwen3 砍掉 — 双方各自的实验依据是什么。
2. **稀疏度的甜点**：DeepSeek V3 的 ~3.7%、Qwen3 的 ~9.4%、Mixtral 的 28% — 为什么没有收敛到一个值，而是各自匹配自家的 GPU 预算和 tail-latency 目标。
3. **Aux-loss-free 的本质**：DeepSeek V3 用动态 bias 替代辅助 loss，避免梯度被 balance 信号干扰 — 这是工程 trick 还是有更深的理由（balance loss 和 task loss 的方向冲突）。
4. **MTP 的双重收益**：Multi-Token Prediction 同时加速训练（一次预测多个 token，提高 sample efficiency）和推理（天然兼容推测解码）— 这两个收益是同一个机制还是恰好叠加。
5. **Wide Expert Parallel 的服务约束**：DeepSeek 那种 256 专家全开的 MoE，必须靠 vLLM/SGLang 的 wide EP 才能高效服务 — 路由设计和服务工程是端到端绑死的，不是"先训练好再考虑部署"。

**判断主线**：MoE 不是单一技术选择，是一组**联合决策**（专家粒度 × 共享比例 × 稀疏度 × 服务架构）。看懂三家差异 = 看懂 MoE 的 design space。

---


---
