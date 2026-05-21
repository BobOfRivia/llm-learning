# KV Cache 经济学（含 Prompt Caching 与长上下文真实退化）

> deadline: 2026-06-03
> status: pending
> 覆盖 tasks: KV Cache 工作原理、PagedAttention 设计、Prefill vs Decode 瓶颈差异、跨请求 Prompt Caching、长上下文真实退化（NIAH vs RULER）
> 桥接: tracks/inference.md, topics/kv-cache.md

**核心问题**：KV Cache 这一层在 2023-2025 经历了三段演化，每一段解决一个完全不同的问题，但被同一个抽象（"缓存 attention 的 K/V"）串起来。要能讲清三段：

1. **段一：单请求内的内存效率（PagedAttention，2023）**。朴素 KV Cache 预分配最大序列长度，碎片化严重。PagedAttention 借 OS 虚拟内存的页表机制，按需分配、可不连续，配合 continuous batching → vLLM 24× 吞吐量。要能解释 KV Cache 的显存占用公式（Llama 3 8B、32K context、batch=8 时具体多大），以及 Prefill（compute-bound）vs Decode（memory-bound）的本质差异如何决定调度策略（PD 分离的经济逻辑）。
2. **段二：跨请求的成本复用（Prompt Caching，2024-2025）**。如果同一段 system prompt 被 1000 个请求复用，那它的 KV 该被算 1000 次吗？显然不应该。Anthropic 显式 `cache_control`（5 分钟 / 1 小时 TTL，cache write 1.25× / 2×、cache read 0.1× 价格）；OpenAI 隐式自动缓存（≥1024 token 起，128 token 增量）。要能回答：**这两种设计哲学的差异**（显式 vs 隐式）、为什么 Anthropic 给两档 TTL、cache 命中如何改变长 system prompt 应用（agent / RAG）的成本曲线。
3. **段三：长上下文是否真的可用（RULER，2024-2025）**。1M context 是 marketing — Chroma 研究显示 Claude 衰减最慢、Gemini 最早退化；GPT-4.1 在 NIAH 100% 但 RULER 在 64K 已显著掉。要能讲清：**NIAH 为什么过时**（单针太简单），**RULER 加了哪四类任务**（multi-needle / multi-hop / aggregation / QA），以及 Lost-in-the-Middle 现象的机制假说（causal attention 对开头结尾偏强）。

**判断主线**：把"KV Cache" 看成一个**有三层经济学**的对象 — 单请求内（vLLM 解决）、跨请求间（Prompt Caching 解决）、长上下文质量（RULER 揭露还没解决）。能用这三层框架看任何 serving 架构问题。

---


---
