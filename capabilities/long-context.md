# Long Context

> 桥接：Agent 项目 dimensions/02-long-context.md

## 当前水位（截至 2026-05）

### Context Window 的分布格局

各家实验室在上下文长度上分化明显。截至 2025-2026 年，大致形成三个梯队：

| 模型 | 支持上下文长度 | 备注 |
|------|:------------:|------|
| Gemini 2.0 Pro | 2M tokens | Google |
| Gemini 2.5 Pro / 1.5 Pro | 1M tokens | Google |
| Gemini 2.0 Flash | 1M tokens | Google |
| Claude 4 系列 | 1M tokens | Anthropic（最新旗舰） |
| Claude 3.x 系列 | 200K tokens | Anthropic |
| GPT-4o / o1 / o3 | 128K tokens | OpenAI |
| Llama 3.1（8B/70B/405B） | 128K tokens | Meta |

这个格局说明了一件事：超长上下文（1M+）目前主要是 Google 和 Anthropic 主打的差异化方向，OpenAI 系列和大多数开源模型稳定在 128K。差异背后有两个驱动因素：Google 的 TPU 垂直整合使超长上下文推理成本可控；Anthropic 在长文档分析的使用场景（法律、科研）上有强烈需求驱动。

### Benchmark：支持长度 ≠ 有效利用长度

**"声称上下文长度"和"有效上下文长度"是两件事**——这是这个能力维度最重要的认识。

**NIAH（Needle-in-a-Haystack）**是最常见的长上下文测试：在大量无关文本（"草堆"）中插入一句目标信息（"针"），要求模型准确检索。NIAH 的局限性是它只测试最简单的一种长上下文能力——精确的信息检索，而且"草堆"是随机文本，对模型的干扰远低于真实的语义相似文档。Gemini 1.5 Pro 在 NIAH 上展示了约 1M tokens 范围内接近 100% 的检索准确率（[技术报告 arXiv:2403.05530](https://arxiv.org/abs/2403.05530)），但这个数字描述的是"最佳情况下的信息检索"，不代表在同等长度上的综合推理能力。

**RULER**（NVIDIA，2024 年，[arXiv:2404.06654](https://arxiv.org/abs/2404.06654)）是这个问题的系统化回应。RULER 包含 13 个任务，覆盖检索、多跳追踪、聚合、问答四类，用"在特定上下文长度下是否仍能维持 85% 基准准确率"来定义"有效上下文长度"。RULER 的核心发现是：几乎所有声称支持 128K 的模型，其有效上下文长度远低于这个数字。大多数模型的性能在 32K-64K 之后开始明显衰减，即使它们在 NIAH 上表现完美。这揭示了"支持某个上下文长度"和"在这个长度上可靠工作"之间的系统性落差。

**LongBench v2**（清华大学等，2024 年 12 月，[arXiv:2412.15204](https://arxiv.org/abs/2412.15204)）从另一个角度验证了这个落差：503 道多选题，上下文范围 8K-2M 字，涵盖单文档 QA、多文档 QA、代码库理解、长对话等场景。人类专家在 15 分钟时限内的准确率约 53.7%，最强模型（o1-preview）达到约 57.7%。这说明在需要深度理解而非简单检索的真实长上下文任务上，模型能力距"充分利用超长上下文"仍有很大距离。

**"Lost in the Middle" 问题**（Liu et al., Stanford/Meta, 2023 年，[arXiv:2307.03172](https://arxiv.org/abs/2307.03172)，TACL 2024）提供了关于注意力机制内部分布的关键实验证据：在多文档问答和键值对检索任务中，当相关信息位于上下文开头或结尾时，模型准确率最高；位于中间时，准确率系统性下降，且这一现象随上下文变长而加剧。这说明 Attention 机制的注意力并非均匀分布——有显著的"首因效应"（primacy bias）和"近因效应"（recency bias），中间的信息容易被"遗忘"。这是理解长上下文能力局限最重要的实验结论之一。

## 技术归因

长上下文能力的当前水位由四条机制支撑，但它们在能力上限和系统成本上的影响方向并不一致：

**Flash Attention 是工程基础**（→ [架构演进 - 阶段 3](../tracks/architecture.md)）。标准 Attention 的计算量随序列长度呈 O(n²) 增长，在 GPU 实现上还有大量 HBM 读写开销——不解决这个问题，超长上下文在训练和推理时内存直接耗尽。Flash Attention（Dao et al., 斯坦福，2022 年，[arXiv:2205.14135](https://arxiv.org/abs/2205.14135)）通过 IO 感知分块计算，在不改变结果的前提下将训练速度提升 2-4 倍，内存降低 5-20 倍。Flash Attention 2（2023 年）再次加速约 2 倍。没有 Flash Attention，"64K 上下文"在当时的 GPU 上几乎无法实用——它是长上下文竞赛的必要条件。

**GQA（Grouped Query Attention）解决了推理时的 KV Cache 瓶颈**（→ [架构演进 - 阶段 3](../tracks/architecture.md)，[推理优化](../tracks/inference.md)）。标准 Multi-Head Attention 的 KV Cache 随序列长度线性增长：一个 70B 模型，处理 128K tokens 上下文时，KV Cache 本身就需要约 100-150GB 显存，往往超过模型权重本身。GQA（Ainslie et al., Google，2023 年，[arXiv:2305.13245](https://arxiv.org/abs/2305.13245)）通过让 Q 头分组共享 K、V 来大幅压缩 Cache 大小，质量接近 MHA。Llama 2、Llama 3、Mistral 等主流模型均采用 GQA——这不只是效率改进，而是使长上下文推理在内存上可行的关键。

**RoPE 和位置编码外推技术决定了"能把窗口推多远"**（→ [架构演进 - 阶段 3](../tracks/architecture.md)）。RoPE（旋转位置编码，Su et al., [arXiv:2104.09864](https://arxiv.org/abs/2104.09864)）的核心优势是支持位置外推——通过调整旋转频率的缩放方式，可以将训练时支持的上下文长度延伸到更长的范围。

具体的外推技术形成了一个演进序列：**ALiBi**（Press et al., ICLR 2022，[arXiv:2108.12409](https://arxiv.org/abs/2108.12409)）用线性衰减的距离偏置替代位置嵌入，从设计上支持"训练短，测试长"，但其性能随长度增加有衰减上限。**NTK-aware Scaling**（2023 年，基于神经切线核理论）对 RoPE 的旋转频率做非均匀缩放，无需微调即可将上下文扩展到训练长度的数倍，已成为 Qwen、DeepSeek 等主流开源模型的标准配置。**YaRN**（Peng et al., 2023 年，[arXiv:2309.00071](https://arxiv.org/abs/2309.00071)，ICLR 2024）结合 NTK-by-parts 插值和注意力温度缩放，仅用原始训练数据的 0.1% 微调就能将上下文扩展到 128K+，代价极低。**LongRoPE**（2024 年，[arXiv:2402.13753](https://arxiv.org/abs/2402.13753)）进一步处理 RoPE 维度和 token 位置的双重非均匀性，声称支持 2M tokens 以上的扩展。

这些技术共同构成了"用相对低廉的代价把模型推向更长上下文"的工具箱——但它们能解决位置编码的外推问题，无法解决注意力分布的"中间信息遗忘"问题。

**长文本训练数据的构成决定了有效利用率的上限**（→ [数据工程](../tracks/data.md)）。一个模型能"支持" 1M tokens，和它在 1M tokens 上"能做有用的事"是两件不同的事。后者依赖于预训练语料中是否包含足够多的长文档，以及这些文档是否包含需要跨越远距离的信息关联。长代码库、法律合同、科学文献是最有价值的长文本训练数据，但这类高质量长文档在整体语料中占比有限。Gemini 1.5 技术报告明确提到，超长上下文能力不只是架构和位置编码的问题，也是训练数据中多模态长文档的密度问题。

## 演进轨迹

**2020-2022：2K-4K tokens 是硬约束**

这个时期，受标准 Attention 的 O(n²) 内存复杂度限制，主流模型的实用上下文长度在 2K-4K tokens。GPT-3（175B）的上下文是 2048 tokens，这不只是设计选择——在当时的 GPU 内存和计算条件下，更长的上下文会导致显存直接溢出。长文档处理依赖"分块+摘要"的人工工程方案，这是应用层对模型能力不足的一种补偿。

**2023：100K-128K——第一波上下文竞赛**

这一年有两个关键事件。Anthropic 的 Claude（2023 年 3 月）率先将上下文扩展到 100K tokens，核心技术是 Constitutional AI 之外的工程赌注：相信长上下文是 LLM 最核心的使用场景差异化点之一。GPT-4 Turbo（OpenAI，2023 年 11 月）随即跟进到 128K tokens。这一年，Flash Attention 和 RoPE 的组合已经工程成熟，使 128K 在技术上变得可行。但这一年的模型在 128K 范围内的"有效利用率"仍然有限——"Lost in the Middle" 问题（同年 7 月发布）实际上揭示的是这批模型的现实能力边界。

**2024：Gemini 1.5——从 128K 到 1M 的量级跨越**

Gemini 1.5 Pro（Google，2024 年 2 月）将上下文扩展到 1M tokens（研究版本达 10M），是这条演进轨迹的最重要里程碑。技术报告（[arXiv:2403.05530](https://arxiv.org/abs/2403.05530)）展示了在 1M tokens 的 NIAH 测试中 99.7% 的检索准确率——这个数字不只是"有很长的窗口"，而是说明在这个规模上信息检索是可靠工作的。

Gemini 1.5 的技术路线有几个值得注意的点：它据报道采用 MoE 架构（减少单次推理成本）、Google 的 TPU 成本优势、以及专门针对长文档的训练数据工程。这次跨越把"超长上下文"从"实验特性"变成了可以真正部署的产品能力，同时也重新激活了一个架构层面的争议——

**2024：RAG vs Long Context 的架构选择重开**

Gemini 1.5 1M context 的出现，使"用 RAG 检索相关片段"还是"直接把整个文档塞进上下文"这个选择变得更加复杂。理论上，如果模型能可靠地在 1M tokens 内工作，RAG 的工程复杂度（向量数据库、检索召回率、答案片段拼接）就失去了存在理由。但 RULER 的数据（2024 年，[arXiv:2404.06654](https://arxiv.org/abs/2404.06654)）给出了清醒的对照：在超过 32K-64K 的真实任务上，大多数模型的性能开始衰减——"支持 1M"≠"在 1M 上可靠"。在 2024 年，RAG 仍然在实际应用中有明确价值，尤其是需要在海量文档库中精确检索的场景（用户不可能每次查询都把整个知识库塞进上下文）。这个争议没有被完全解决，只是被往后推迟了——随着有效利用率的提升，"终态"可能是两者的结合：用 RAG 选出最相关的几个文档，再用长上下文模型做深度推理。

**2025-2026：从长度竞赛到有效利用率**

Gemini 2.0 系列（2025 年）将 1M-2M 推向量产。更重要的变化是关注点的转移：头部实验室开始在技术报告中强调"有效利用率"而非仅仅"支持长度"——这是一个行业认知成熟的信号。推理模型（o1/o3/Claude 3.7 Extended Thinking）带来了新的长上下文挑战：思维链 tokens 和输入上下文共享 KV Cache，超长思维链在长上下文输入下会对内存造成双重压力，推动了 KV Cache 分级存储（热数据在 GPU HBM，冷数据 offload 到 CPU RAM）等新的 serving 技术。

## 关键张力

**有效利用率是长上下文能力的真正瓶颈。** 注意力机制的"首因/近因偏置"（Lost in the Middle 效应）是 Transformer 架构的内在局限，不能通过更长的位置编码外推完全解决——它需要专门的训练目标（例如要求模型从中间位置提取信息的训练数据）或架构层面的改进（例如 SSM 的长程压缩能力作为互补）。

**KV Cache 成本是长上下文的经济学约束。** 1M tokens 的 KV Cache 在主流推理框架下需要数百 GB 显存，这使超长上下文推理在成本上对大多数应用不划算。这是为什么 128K 仍然是大多数商业 API 的"性价比上下文长度"——超过这个点，成本曲线开始陡峭。

## 与其他轨道的交叉

- **architecture**：Flash Attention、GQA、RoPE 外推是长上下文的工程基础；MoE 降低了超长上下文推理的单步成本（→ [架构演进](../tracks/architecture.md)）
- **inference**：KV Cache 管理是长上下文推理的核心瓶颈；PD 分离（Prefill/Decode 分离）对长上下文 Prefill 阶段的优化尤其重要（→ [推理优化](../tracks/inference.md)）
- **training**：长文本训练数据的密度和多样性决定有效利用率上限（→ [数据工程](../tracks/data.md)）
- **scaling**：推理时扩展（思维链）与长上下文在 KV Cache 上存在竞争，是 serving 层的资源调度难题（→ [Scaling](../tracks/scaling.md)）

## 信息源

- [Gemini 1.5 技术报告 (arXiv:2403.05530)](https://arxiv.org/abs/2403.05530)
- [RULER: What's the Real Context Size of Your LLMs? (arXiv:2404.06654)](https://arxiv.org/abs/2404.06654)
- [Lost in the Middle: How Language Models Use Long Contexts (arXiv:2307.03172)](https://arxiv.org/abs/2307.03172)
- [LongBench v2 (arXiv:2412.15204)](https://arxiv.org/abs/2412.15204)
- [YaRN: Efficient Context Window Extension (arXiv:2309.00071)](https://arxiv.org/abs/2309.00071)
- [LongRoPE: Extending Context Window Beyond 2M Tokens (arXiv:2402.13753)](https://arxiv.org/abs/2402.13753)
- [ALiBi: Train Short, Test Long (arXiv:2108.12409)](https://arxiv.org/abs/2108.12409)
- [Flash Attention (arXiv:2205.14135)](https://arxiv.org/abs/2205.14135)
- [GQA: Training Generalized Multi-Query Transformer Models (arXiv:2305.13245)](https://arxiv.org/abs/2305.13245)
- [LongBench (arXiv:2308.14508)](https://arxiv.org/abs/2308.14508)

## 更新日志

- 2026-05-02：创建骨架
- 2026-05-02：填充完整内容（当前水位 + benchmark 三层分析 + 技术归因四条路径 + 演进轨迹 2020-2026 + 关键张力 + 信息源）
