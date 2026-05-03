# Chain-of-Thought

> 锚点：`capabilities/reasoning.md` 演进轨迹 / `tracks/training.md` 阶段 3-4

## 这个概念是什么

Chain-of-Thought（CoT）是通过生成中间推理步骤来激活和增强 LLM 推理能力的技术。它从最初的 prompting 技巧（Wei et al., 2022）演化为训练目标本身（o1/R1 时代）。CoT 在演进中跨越了一个关键边界：从"发现已有的潜在能力"到"训练出原本不存在的能力"。理解这条线的演进对判断 reasoning 能力的来源和上限至关重要。

## 内部结构

### Few-Shot CoT（Wei et al., 2022）

> 待推导：
> - 机制：在 prompt 中嵌入包含推理步骤的示例，模型模仿这种模式
> - 尺度效应：只在 ~100B+ 参数的模型上生效，小模型无效甚至有害
> - 核心判断："CoT 是发现而非创造能力"——它激活的是预训练中已有的推理模式
> - 证据：相同模型 + 不同 prompt 格式，得分差距可达 20-30 百分点
> - 含义：prompting 有硬上限（不能激活不存在的能力）

### Zero-Shot CoT（Kojima et al., 2022）

> 待推导：
> - "Let's think step by step" 为什么能工作
> - 不需要手工编写示例的意义——推理激活从专家工具变为普通用户可操作
> - 与 few-shot CoT 的效果对比：通常弱于精心设计的 few-shot，但远好于无 CoT

### 为什么只在大模型上生效

> 待推导：
> - 假说 1：大模型在预训练中见过的推理范例足够多，小模型没有足够的"推理模式库"
> - 假说 2：生成推理步骤需要一定规模以上的"工作记忆"（中间激活的容量）
> - 假说 3：CoT 的效果是连续的还是突变的？（涌现争议的子问题）
> - 实验证据：哪些数据支持哪个假说？

### 从 Prompting CoT 到 Training CoT

> 待推导：
> - 关键转变：o1 不是"给模型一个好的 CoT prompt"，而是"训练模型自己产生有效的 CoT"
> - Prompting CoT 的上限：GPT-4o + 最好的 CoT prompt 在 AIME 上 ~9%
> - Training CoT 的突破：o1 在 AIME 上 ~74%（同一家公司的基座模型，训练目标不同）
> - 本质区别：prompting 激活已有能力，training 创造新能力（更长/更结构化的推理链）

### CoT 的成本与质量关系

> 待推导：
> - CoT tokens 消耗推理计算（更多 token = 更多 FLOPs + KV Cache）
> - 思维链长度与推理质量不是单调关系（有最优长度）
> - "思维预算"的概念：Claude 3.7 的可调思维 token 上限
> - 什么时候不需要 CoT（简单事实查询、格式化输出）

### CoT 的可信度问题

> 待推导：
> - 展示的 CoT 是否反映真实的内部决策过程
> - Anthropic 的发现：CoT 有时更像"事后解释"而非决策过程
> - 对于对齐的含义：监督 CoT ≠ 监督内部计算
> - 指向：`tracks/alignment.md` 阶段 3，`topics/mechanistic-interpretability.md`

## 当前状态（截至 2026-05）

> 待填入：CoT 已从"prompting 技巧"演变为"训练范式的核心"。所有推理模型（o1/o3/R1/Claude ET）本质上都是在训练 + 使用 CoT。CoT 的产出质量现在是模型推理能力的直接表征。

## 关键权衡

> 待填入：
> - Token 成本 vs 推理质量：更长的 CoT 通常更好，但成本线性增长
> - 可见性 vs 性能：隐藏 CoT（o1）vs 展示 CoT（Claude）的对齐考量
> - 训练 CoT vs 学习结果：推理蒸馏（R1-Distill）证明学习 CoT 比学习答案有效

## 信息源

- [Chain-of-Thought Prompting Elicits Reasoning in Large Language Models (Wei et al., 2022)](https://arxiv.org/abs/2201.11903)
- [Large Language Models are Zero-Shot Reasoners (Kojima et al., 2022)](https://arxiv.org/abs/2205.11916)
- [Learning to Reason with LLMs (OpenAI, o1 blog post)](https://openai.com/index/learning-to-reason-with-llms/)
- [DeepSeek-R1 Technical Report](https://arxiv.org/abs/2501.12948)

## 更新日志

- 2026-05-03：创建骨架
