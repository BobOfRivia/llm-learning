# 推理模型的范式分叉

> deadline: 2026-07-12
> status: pending
> 覆盖 tasks: R1-Zero 的涌现、o1 / o3 的技术实质、Claude 3.7 / 4 的 hybrid thinking、推理蒸馏机制
> 桥接: topics/rl-for-reasoning.md, eras/era4-reasoning.md, labs/openai.md, labs/anthropic.md

**核心问题**：2024-2025 推理纪元的关键事件不是"推理能力出现了"，而是**三家走出了三条不同路线**，且路线差异背后是产品哲学差异。要能讲清三条路线 + 一个共性问题：

1. **OpenAI 路线：dedicated reasoning model**（o1 → o3 → o4）。推理是"专门模型 + 隐藏思维链"，模型分裂为非推理（GPT-4o/4.1）和推理（o 系列）两条产品线，按场景选择。**为什么 OpenAI 选这条路** — 训练目标差异太大、隐藏 CoT 是产品/安全考虑、计费分层。
2. **Anthropic 路线：hybrid thinking model**（Claude 3.7 Sonnet → Claude 4）。一个模型支持 instant + extended thinking 两个 mode，开发者用 `thinking budget` 参数控制。**为什么 Anthropic 选这条路** — 哲学上认为推理应是统一智能的自然能力、产品上避免用户在两个模型间切换。同时 Anthropic 显式说"刻意减少了竞赛数学/CS 优化、增加真实业务任务比重"。
3. **DeepSeek 路线：开源复现 + 蒸馏到小模型**（R1-Zero → R1 → R1-Distill 系列）。R1-Zero 证明纯 RL（无 SFT 冷启动）也能 work，自我反思在训练中涌现；R1-Distill-Qwen-32B 直接超过 o1-mini，证明**推理能力可以蒸馏，且蒸馏比直接对小模型做 RL 更有效**。
4. **共性问题：推理链可信吗**。Anthropic 的研究显示模型展示的 CoT 不一定反映真实内部计算（"reasoning models don't always say what they think"）。这个问题对三家路线都成立，意味着"看 CoT 验证对齐"这条思路有根本局限。

**额外要回答**：
- **R1-Zero 的"自我反思涌现"**：是真的从无到有的新能力，还是预训练里已有的行为被 RL 信号筛选放大？给出你的判断和依据。
- **推理蒸馏的不对称性**：为什么"用 R1 思维链做 SFT" 远比"用 R1 答案做 SFT" 有效（学过程 vs 学结论的信号量差异）。
- **On-Policy Distillation**（Thinking Machines 2025）：相对静态蒸馏的改进点是什么。

**判断主线**：这一篇是 phase 2 推理纪元的高潮 — 重点不是"推理模型怎么做"，而是"推理纪元为什么裂出三条路线"。把握住产品哲学（OpenAI 分裂 / Anthropic 统一 / 开源蒸馏）这一层，未来看任何新推理模型都能直接归类。

---


---
