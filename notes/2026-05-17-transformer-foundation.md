# Transformer 地基

> deadline: 2026-05-17
> status: pending
> 覆盖 tasks: BPE 算法、Self-Attention 机制、Multi-Head Attention、Transformer 层结构（Pre-LN / FFN / 残差）、自回归语言建模目标、decoder-only 为什么统治
> 桥接: tracks/architecture.md 阶段 1, topics/attention-mechanism.md

**核心问题**：用 30 分钟向不懂 LLM 的同事讲清"Transformer 是什么、为什么 work" — 你的讲法是什么？要能从零写出 attention 公式并解释每一步的设计理由，能手动模拟 BPE 合并过程，能解释为什么现代模型都是 decoder-only 而不是 encoder-decoder 或 encoder-only。

---
