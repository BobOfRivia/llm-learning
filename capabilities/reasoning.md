# Reasoning

> 桥接：Agent 项目 dimensions/01-reasoning.md

## 当前水位（截至 2026-05）

### 关键 Benchmark 图谱

各 benchmark 测的不是同一件事，理解它们的设计意图比看数字更重要。

**AIME**（美国数学邀请赛，每年新题）是最常用的硬数学基准。答案是整数，无运气成分，测试的是多步数学推导的完整性。顶级竞赛学生通常得 60-90%。AIME 2024 在一年内从 GPT-4o 的约 9% 升至 o3 的约 97%，是推理能力跃升最清晰的单条轨迹。

**GPQA Diamond**（Graduate-Level Google-Proof Q&A，Rein et al., 2023，[arXiv 2311.12022](https://arxiv.org/abs/2311.12022)）包含博士级难度的生物、化学、物理选择题，题目由各领域博士候选人出题，设计为"即使上网搜索也难以回答"，测试真正的专业推理而非信息检索。人类基线：领域博士专家约 65%，无专业背景的非专家约 34%。o3 以 87.7% 系统性地超过了专家人类，是 LLM 在专业科学推理上首次明确超越领域专家的证据。

**ARC-AGI**（Abstract and Reasoning Corpus，Chollet，2019，[arcprize.org](https://arcprize.org/)）是通过极少数视觉示例推断抽象变换规则的网格任务。人类几乎满分（约 100%），但在 o1 发布之前所有 LLM 均低于 5%。它测试的是"从最少示例泛化"的归纳推理，而非背诵知识。2024 年 o3 high compute 约 88%，基本宣告 ARC-AGI-1 被解决，作者随即推出 ARC-AGI-2。

**MATH-500**（Hendrycks et al., 2021，[arXiv 2103.03874](https://arxiv.org/abs/2103.03874) 的 500 题子集）目前在顶级推理模型上已接近饱和（97%+），不再有效区分前沿模型。

**FrontierMath**（EpochAI，2024 年 11 月，[公告](https://epochai.org/frontiermath/the-benchmark)）由专业数学家出题，要求研究级数学知识。发布时所有模型均低于 2%，o3 是第一个突破这条线的模型（约 25%），目前是 reasoning 能力最有区分度的硬边界。

### 代表性模型表现

| 模型 | AIME 2024 | GPQA Diamond | ARC-AGI | MATH-500 |
|------|:---------:|:------------:|:-------:|:--------:|
| GPT-4o（2024-05） | ~9% | ~53% | ~5% | ~76% |
| o1（2024-09） | ~74% | ~78% | ~32% | ~97% |
| DeepSeek-R1（2025-01） | 79.8% | 71.5% | — | 97.3% |
| Claude 3.7 Sonnet ET（2025-02） | — | 75.3% | — | 96.2% |
| Gemini 2.5 Pro（2025-03） | ~92% | — | — | — |
| o3 high compute（2025-04） | 96.7% | 87.7% | ~87.5% | — |

ET = Extended Thinking。表中约为 pass@1 设置（o1 共识投票版 ~83% 未列入以保持口径一致）。来源：[OpenAI "Learning to Reason with LLMs"（2024-09-12）](https://openai.com/index/learning-to-reason-with-llms/)、[ARC Prize o3 分析（2024-12-20）](https://arcprize.org/blog/oai-o3-pub-breakthrough)、[DeepSeek-R1 技术报告（arXiv 2501.12948）](https://arxiv.org/abs/2501.12948)、[OpenAI o3/o4-mini 发布（2025-04-16）](https://openai.com/index/introducing-o3-and-o4-mini/)、Anthropic Claude 3.7 Sonnet 技术报告（2025-02-24）。

### Benchmark 饱和与新前沿

MATH-500 和 HumanEval 在 2024-2025 年实际饱和；AIME 2024 在 2025 年也接近饱和（o3 约 97%）。ARC-AGI-1 被 o3 基本解决，作者随即推出 ARC-AGI-2。这个快速饱和本身就是能力跃升速度的证据——设计为"AI 难以突破"的基准，在推理纪元开启后的一年内接连失效。FrontierMath 和 ARC-AGI-2 目前是最有区分度的前沿基准。

## 技术归因

Reasoning 能力的当前水位由五条机制叠加形成，各自作用的阶段不同：

**预训练数据构成**决定 reasoning 的基础下限（→ [数据工程](../tracks/data.md)）。数学教材、竞赛解答、代码库在预训练语料中的密度，直接决定模型"见过多少推理模式"。Phi 系列（Microsoft）的实验证明，用"教科书质量"高密度推理文本训练极小模型，可以获得远高于参数量预期的 reasoning 基线。这条路径决定了能力的地板，不是天花板。

**Chain-of-Thought 的发现**是 prompting 时代的推理激活机制（→ [训练范式 - 阶段 3](../tracks/training.md)）。Wei et al.（2022，[arXiv 2201.11903](https://arxiv.org/abs/2201.11903)）证明，在 prompt 中嵌入分步推理示例可以大幅提升 100B+ 模型的 reasoning 性能——而相同 prompt 对小模型无效甚至有害。核心判断：CoT 是"发现"而非"创造"能力——它激活的是预训练中已经存在的推理模式。这也意味着 prompting 有硬上限：无论 CoT prompt 多精心设计，GPT-4o 在 AIME 上都无法接近 o1 的水平，因为 prompting 激活不了不存在的能力。

**RL for Reasoning（RLVR）**是推理能力质变的直接来源（→ [训练范式 - 阶段 4](../tracks/training.md)）。从"通过 prompting 引导潜在能力"转变为"通过 RL 专门训练推理过程"，是 o1/o3/DeepSeek-R1 相较于 GPT-4 的核心区别。RLVR（可验证奖励强化学习）用数学答案正确性和代码执行结果作为奖励信号，完全绕开人类偏好标注，使推理训练规模化不受限于人工成本。模型学到的不是"产生看起来像推理的文本"，而是"产生有助于得到正确答案的推理过程"——这个区别在 benchmark 上体现为量级差距。

**推理时扩展**提供了 inference 阶段的连续能力旋钮（→ [Scaling - 阶段 4](../tracks/scaling.md)，[topics/inference-time-compute.md](../topics/inference-time-compute.md)）。同一个推理模型分配更多思考 tokens，性能按可预测曲线提升（o3 low/high compute 在 ARC-AGI 上差约 12 个百分点）。这是独立于训练时 scaling 的第二个计算维度，也是推理模型按计算档位差异定价的技术基础。

**推理蒸馏**证明 reasoning 能力可以低成本转移（→ [训练范式 - 阶段 4](../tracks/training.md)）。DeepSeek-R1 用大型推理模型生成的思维链做 SFT 数据，训练 Llama/Qwen 小模型（1.5B-70B），得到的 R1-Distill 系列在推理任务上显著优于同参数量的非推理模型。这使 reasoning 能力不再只属于 100B+ 规模的闭源模型，有效地把推理能力"下放"到开放生态。

## 演进轨迹

**2022：Chain-of-Thought——发现潜在能力**

Wei et al.（[arXiv 2201.11903](https://arxiv.org/abs/2201.11903)，2022-01）发现，在 prompt 中嵌入分步推理示例，540B 参数的 PaLM 在 GSM8K 数学应用题上从 18% 跳到 57%。关键约束：效果只在约 100B+ 规模下出现，小模型不受益。这说明 CoT 依赖的是规模赋予的潜在推理模式——是发现能力，不是赋予能力。

同年，零样本 CoT（Kojima et al., [arXiv 2205.11916](https://arxiv.org/abs/2205.11916)，"Let's think step by step"）消除了对人工编写示例的依赖，推理激活变成普通用户可操作的工具。这两篇论文共同确立了 prompting 时代 reasoning 的操作范式。

**2023：GPT-4——reasoning 大幅提升，仍 prompt 依赖**

GPT-4 在多步推理上有明显跃升，MATH 基准从 GPT-3.5 的约 18% 升至约 42%。但仍依赖 CoT prompt，对 prompt 格式高度敏感（相同问题不同 prompt 格式，得分差距可达 20-30 个百分点），且不稳定。这个时期"推理"仍被视为可通过 prompt 工程激活的涌现行为，而非可靠的能力维度。

**2024-09：o1——推理作为训练目标的质变**

o1（[OpenAI，2024-09-12](https://openai.com/index/learning-to-reason-with-llms/)）是这条演进轨迹的分水岭。与 GPT-4 的根本区别是训练目标：o1 经过 RL + 可验证奖励专门训练推理过程的有效性，不只是做 RLHF 对齐。AIME 2024：GPT-4o ~9% → o1 ~74%，65 个百分点的差距无法通过给 GPT-4o 更好的 CoT prompt 弥合。GPQA Diamond：o1 ~78%，超过博士领域专家（~65%）。这是 LLM 推理能力的第一次训练层面的质变。

**2024-12：o3——推理时 scaling 曲线**

o3（[OpenAI，2024-12-20 宣布](https://arcprize.org/blog/oai-o3-pub-breakthrough)，2025-04 正式发布）展示了推理时计算形成独立 scaling 曲线的实证。ARC-AGI：GPT-4o ~5%，o1 ~32%，o3 low ~75.7%，o3 high ~87.5%——同一模型不同计算预算，得分提升系统可预测。o3 同时是第一个在 FrontierMath 上突破 2% 基线的模型（约 25%）。

**2025-01：DeepSeek-R1——开源验证**

DeepSeek-R1（[arXiv 2501.12948](https://arxiv.org/abs/2501.12948)，2025-01-20）提供两个关键验证：RL for Reasoning 路线可开源复现（AIME 2024 与 o1 持平，79.8%，开放权重）；R1-Zero 证明从纯基础模型做 RL、不经 SFT 也能涌现自我反思行为——推理能力是 RL 优化的自然产物，不依赖大量监督数据。R1-Distill 系列（1.5B-70B）将 reasoning 能力转移到小模型。

**2025：推理模式成为行业标配**

Claude 3.7 Sonnet（2025-02，Extended Thinking）、Gemini 2.5 Pro（2025-03）、o4-mini（2025-04）相继推出推理模式。"是否支持扩展推理"成为判断模型规格完整性的标准之一——类似 2023-2024 年多模态的地位演变。MATH-500、HumanEval 实际饱和，AIME 2024 接近饱和，社区注意力转向 FrontierMath、AIME 2025、ARC-AGI-2。

## 更新日志

- 2026-05-02：创建骨架
- 2026-05-02：填充完整内容（五个 benchmark 说明 + 模型数据表 + benchmark 饱和分析 + 技术归因五条路径 + 演进轨迹 2022-2025）
