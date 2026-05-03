# 推测解码（EAGLE-3）与蒸馏（R1-Distill / On-Policy）

> deadline: 2026-08-09
> status: pending
> 覆盖 tasks: 推测解码的正确性证明、EAGLE-3 在 vLLM/SGLang 的事实标准地位、R1-Distill 实战、On-Policy Distillation
> 桥接: tracks/inference.md, topics/speculative-decoding.md
> 剔除：MEDUSA 深入 — 已被 EAGLE-3 取代为事实标准，作为对比点提一句即可。详见 logs/removed-from-mainline.md

**核心问题**：推测解码和蒸馏是**两个独立但都关心"小模型替代大模型"的工程方向** — 前者在推理时（draft → verify），后者在训练时（teacher → student）。2025 年这两条线都收敛到了清晰的事实标准，要能讲清：

1. **推测解码的正确性 + EAGLE-3 主导**。Rejection sampling 数学上等价于 sampling from target（不是近似）— 要能讲清这个证明的核心思路。然后 **EAGLE-3 为什么赢了**：把 draft head 直接 attach 到 target model 上（不是独立 draft model），用多层 feature reuse，在 vLLM/SGLang/TensorRT-LLM 全部成为默认。SGLang H100 实测 batch=2 时 1.81×、batch=64 时 1.38× 吞吐。MEDUSA 同思路但已边缘化（feature 复用层数 / 训练方式的差异）。
2. **draft model 选择 / 训练**：在生产里，draft 头是和 target 一起训的（EAGLE-3）还是独立的小模型（早期方案）— 影响显存、batch 内 contention、维护成本。
3. **R1-Distill：推理能力的蒸馏（2025 标准做法）**。DeepSeek 用 800K 验证轨迹 SFT Qwen/Llama → R1-Distill-Qwen-32B 直接超过 o1-mini。**关键 insight**：用 R1 的**思维链** 做 SFT 比用 R1 的**答案**做 SFT 远更有效 — 学过程比学结论传递的信号量大。这跟传统知识蒸馏（logit-level distillation）的"暗知识"机制不一样，是"reasoning trace 作为监督信号"的新范式。
4. **On-Policy Distillation（Thinking Machines 2025）**。静态蒸馏的问题：teacher 输出和 student 当前分布不匹配。On-policy 让 student 自己采样、teacher 评分 — 解决分布漂移。要能讲清和 RL（GRPO）的边界（一个用 teacher score 当 reward，一个用 verifier）。
5. **能力蒸馏 vs 推理蒸馏的本质区别**：能力蒸馏要传递"模型分布"（logit / soft label），推理蒸馏要传递"过程结构"（CoT 序列）— 两者的信号载体不同，所以适用场景也不同。

**判断主线**：把这一篇看成"如何让小模型能力更强"的两条独立路径 — 推理时 trick（spec decoding，等价加速）vs 训练时 trick（蒸馏，能力转移）。两条路径在 R1-Distill 时代第一次能讲出清晰的产业级标准。

---


---
