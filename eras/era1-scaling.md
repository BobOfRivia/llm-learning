# Scaling 纪元（2020-2022）

## 主导叙事

"大就是好。" 这个时期整个领域在围绕一个核心假设投入注意力：更大的模型 + 更多的数据 = 更强的能力。Scaling laws 的发现给了这个假设数学基础，GPT-3 的涌现能力给了它经验验证。

## 关键事件与里程碑

这个纪元的开端是一篇数学论文，而非模型发布。Kaplan 等人（OpenAI，2020 年 1 月，arXiv 2001.08361）在超过 7 个数量级的规模上，证明了 loss 与模型参数量（N）、数据量（D）、计算量（C）之间的幂律关系。最关键的实践结论是：**固定计算预算下，应优先扩大模型参数量**（约 1.7 tokens/参数）。这给"大模型就是好模型"的直觉提供了数学背书。

五个月后，GPT-3（Brown et al., 2020 年 5 月，arXiv 2005.14165）以 175B 参数落地，在 ~499B tokens 上训练，比当时最大的非稀疏模型大了 10 倍。GPT-3 的核心突破不是某项任务的 SOTA，而是**上下文学习（in-context learning）**：不更新模型权重，只通过 prompt 中的少量示例，模型就能泛化到新任务。这在经验上验证了 Scaling Laws，并首次展示了规模与涌现能力之间的关联。

2021 年，两件事延续了这个纪元的主题。Codex（Chen et al., 2021 年 7 月，arXiv 2107.03374）是 GPT-3 在代码数据上的特化微调，配合 GitHub Copilot 的技术预览（2021 年 6 月），成为 LLM 在生产环境中的第一个大规模落地——代码补全是这个时代的杀手级应用。同期，Switch Transformer（Fedus et al., 2021 年 1 月，arXiv 2101.03961）将 Mixture-of-Experts（MoE）架构工程化至 1.571T 参数，每个 token 只激活一个 expert，首次证明稀疏激活路线的可行性。

2022 年上半年，两篇论文完成了这个纪元的自我修正与终结。Chinchilla（Hoffmann et al., DeepMind，2022 年 3 月，arXiv 2203.15556）重新实验了 scaling 的最优分配，结论颠覆了 Kaplan：参数量和训练 tokens 应**等比扩展**（每参数约需 20 tokens）。Chinchilla 70B 用与 Gopher 280B 相同的计算预算，靠更多数据全面胜出——此前主流模型全都是"参数过多、数据不足"。同月，InstructGPT（Ouyang et al., 2022 年 3 月，arXiv 2203.02155）展示了 SFT → RM → PPO 三步 RLHF 流水线，1.3B 的对齐模型在人类评估中优于 175B 的原始 GPT-3。这是下一个纪元的开场。

## 范式替代

**建立的范式**：
- 规模是获得新能力的主要手段
- decoder-only Transformer 成为事实标准
- few-shot prompting 作为主要使用方式

**开始被质疑的**：
- "越大越好"被 Chinchilla 修正为"数据和参数要平衡"
- 纯预训练模型的可用性（InstructGPT 证明后训练的价值）

## 遗产

Scaling laws 本身是持久遗产——即使后来被修正，"可预测地投入计算换取性能"这个理念深刻塑造了整个领域。decoder-only 架构的统治地位在这个纪元确立，至今未被撼动。

## 更新日志

- 2026-05-02：创建骨架
- 2026-05-02：填充关键事件与里程碑（GPT-3、Kaplan、Switch Transformer、Chinchilla、InstructGPT）
