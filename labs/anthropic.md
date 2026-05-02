# Anthropic

## 技术路线

Anthropic 的技术路线由其创立背景深刻塑造：2021 年从 OpenAI 分拆，核心成员（Dario Amodei、Daniela Amodei、Chris Olah 等）离开的原因在于认为 OpenAI 在商业压力下对安全问题的重视程度不足。这个创立故事不只是公司历史，它直接解释了 Anthropic 几乎所有重要技术决策的底层逻辑——他们在做的事，是把"AI 安全"从一个价值观表态转化为一套工程实践。

### Constitutional AI 与 RLAIF 路线（2022-今）

Anthropic 面临的第一个实质性问题是：RLHF 的人类标注瓶颈怎么解决？人工标注是慢的、贵的，而且在专业领域（安全边界判断、复杂指令）的质量无法保证。他们的回答是 **Constitutional AI（CAI）**（2022 年 12 月，arXiv 2212.08073）。

CAI 的核心思路：用一套写下来的"宪法原则"（一组表达价值判断的自然语言规则）指导模型对自身输出做批评和修正，再用这些 AI 生成的批评作为 RLHF 的偏好信号——即 **RLAIF（AI 反馈强化学习）**。这把人类标注的工作部分转移给了 AI 本身。

这不只是工程上的权宜之计，而是一个有理论深度的设计：把价值观表达为可解释的文本规则，而非隐含在人类偏好标注里，使对齐目标在某种程度上是**可审计的**。任何人都可以读那份"宪法"，判断它是否合理。这与 OpenAI 的 RLHF 路线形成了明显的方法论对比——后者的偏好标准隐含在标注者的判断里，难以显式化。

**参考**：[Constitutional AI: Harmlessness from AI Feedback (arXiv 2212.08073)](https://arxiv.org/abs/2212.08073)

→ 展开见 [topics/constitutional-ai.md](../topics/constitutional-ai.md)

### 机械可解释性：唯一把理解神经网络内部作为战略的实验室（2022-今）

Anthropic 最独特的技术赌注，也是与所有其他前沿实验室区别最大的地方，是 **机械可解释性（Mechanistic Interpretability）** 研究项目，由 Chris Olah 领导。

这个研究方向的出发点是一个强硬的假设：**在不理解神经网络如何工作的情况下，无法可靠地对齐它**。你可以训练一个模型在测试集上表现安全，但如果你不知道它"用什么机制"做出判断，你无法保证它在分布外场景中的行为。

Anthropic 的可解释性团队发表了一系列深度研究，其中几项结果已经在领域内产生了实质影响：**超位置（Superposition）**的发现——神经网络的单个神经元并非对应单一概念，而是以叠加的方式编码了远多于维度数的特征；**稀疏自编码器（Sparse Autoencoders）** 作为提取可解释特征的方法；以及试图理解 Transformer 内部"电路"（circuits）的系列工作。2024 年，他们发表了对 Claude 3 Sonnet 内部特征的初步分析，在特定神经元上找到了对应"金门大桥"、"自我意识"等概念的可识别特征。

这些工作距离"完全可解释的 LLM"仍然遥远，但方向是清晰的：Anthropic 押注这条路最终会产生安全工具，使可靠的对齐成为可能。其他实验室（OpenAI、Google）也有可解释性研究，但都不是核心战略方向。

**参考**：[Toy Models of Superposition (Anthropic, 2022)](https://transformer-circuits.pub/2022/toy_model/index.html)，[Scaling and evaluating sparse autoencoders (arXiv 2406.04093)](https://arxiv.org/abs/2406.04093)，[Mapping the Mind of a Large Language Model (Anthropic, 2024)](https://www.anthropic.com/research/mapping-the-mind-of-a-large-language-model)

→ 展开见 [topics/mechanistic-interpretability.md](../topics/mechanistic-interpretability.md)

### 长上下文先行（2023）

Claude 1.3 的 100K 上下文（2023 年 5 月）是 Anthropic 在产品层面最显著的一次领先。技术基础是对位置编码的系统性工程（RoPE 配合上下文长度 scaling 技术，以及 Flash Attention 的集成），使 100K token 在生产环境下实际可用，而非只是理论支持。

这个决策体现了 Anthropic 的一个产品哲学：**在某个单一维度上大幅领先，而不是在所有维度上追平**。100K 上下文在 2023 年没有任何竞争者，为 Anthropic 建立了明确的差异化定位（"处理长文档找 Claude"）。这个策略后来被 Gemini 1.5 的 1M 上下文所超越，但作为竞争动作，它有效地为 Anthropic 在企业市场争取了时间和关注。

### 负责任扩展政策（RSP）（2023-今）

Anthropic 于 2023 年 9 月发布了**负责任扩展政策（Responsible Scaling Policy）**，承诺在训练更强大模型之前完成特定的安全评估。这是他们把安全治理框架化为可操作制度的尝试——不只是表态"我们重视安全"，而是白纸黑字写下"在模型能力达到 X 之前，我们会做 Y 评估，如果评估失败我们会停止部署"。

这个框架的价值在于建立了对外的问责机制，迫使 Anthropic 内部在安全评估上保持一定的认真程度。批评者指出，RSP 的很多条款仍然模糊，评估方法由 Anthropic 自己定义，缺乏真正的第三方约束。但在 AI 治理基础设施几乎为零的情况下，RSP 是最早的系统性尝试之一，影响了后来的其他实验室（OpenAI 的 Preparedness Framework 等）。

**参考**：[Anthropic's Responsible Scaling Policy (Anthropic, 2023)](https://www.anthropic.com/index/anthropics-responsible-scaling-policy)

### 推理模型与可见思维链（2025）

Claude 3.7 Sonnet（2025 年 2 月）是 Anthropic 的推理模型，与 o1 的关键设计差异在于**思维链可见性**。o1 把内部推理过程完全隐藏；Claude 3.7 以折叠块的形式把扩展思考过程展示给用户。

这个选择不只是产品设计，而是有理论立场：如果你相信可解释性重要，那么模型"在想什么"就应该可见；隐藏思维链在方向上与机械可解释性的研究哲学相悖。这是一个罕见的技术信念影响到产品设计的案例。

有一个重要的已知复杂性：Anthropic 自己的研究发现（2024 年），即使思维链可见，模型的内部计算过程不一定与展示的思维链完全对应——思维链可能更像"事后解释"而非"实时决策记录"。这并不否定可见思维链的价值，但使其对"可解释性"的贡献打了折扣。

### Claude 4 系列：工具交织与 Agent 协作（2025-2026）

Claude 3.7 Sonnet 证明了 Extended Thinking 的可行性之后，Claude 4 系列在两个方向上推进：让推理与工具调用真正交织，以及把单个模型的能力扩展到 Agent 协作场景。

**Claude Opus 4 + Claude Sonnet 4（2025 年 5 月）** 是 Anthropic 迄今能力最强的版本，在编码任务上达到业界最高水位。最重要的架构变化是 **Extended Thinking 与工具调用的交替**：Claude 4 支持在扩展思考过程中插入工具调用，然后根据工具返回结果继续推理——即 Think → Act → Think → Act 的循环。这不是简单的串联，而是推理过程和执行过程在认知层面的融合：模型可以在思考中发现需要某个信息、调用工具获取、再把结果纳入后续推理，形成闭环。对于复杂的多步骤编程任务，这种交织显著减少了错误传播。

**Claude Opus 4.5（2025 年 11 月）** 在工作流任务上做了专项优化，Opus 价格同步下调。

**Claude Opus 4.6（2026 年 2 月）** 引入了两个重要扩展：**100 万 token 上下文**，以及 **Agent Team 协作**功能——多个 Claude 实例可以在结构化框架内分工协作，一个实例的输出成为另一个实例的输入。这把 Anthropic 从"提供单个强大模型"推向了"提供 Agent 协作基础设施"。

**Claude Opus 4.7（2026 年 4 月）** 在 Agent 级编码能力上实现了质的提升，是当前版本线的最新节点。

这条路线的底层逻辑是一致的：从 Constitutional AI 的"价值观可审计"，到机械可解释性的"内部过程可理解"，再到 Extended Thinking 的"推理过程可见"——Anthropic 在推理模型上选择透明而非封闭，而工具交织把这种透明性延伸到了 agentic 执行链。

## 关键决策与赌注

**安全作为技术问题（根本赌注）**：Anthropic 的创立信念是 AI 风险是真实的，且可以通过技术手段显著降低。这个赌注还没有明确的验证或证伪，但它解释了他们几乎所有的技术选择——CAI、机械可解释性、RSP，都是这个信念的工程化体现。

**RLAIF 路线（2022）**：赌 AI 反馈的质量可以达到人类标注的水平，并有更好的扩展性。结果：Claude 系列的对齐质量被业界普遍认为是最高水平之一，支持这个赌注是有效的。

**机械可解释性长期投入**：赌"理解模型内部"最终会产生安全工程工具。这是当前纪元内时间跨度最长的研究赌注，目前仍处于"积累基础知识"阶段，还没有进入"工具落地"阶段。若干年后回看，这将是判断 Anthropic 技术战略是否正确的关键数据点。

**100K 长上下文先行（2023）**：赌长上下文成为关键竞争维度，且率先做到的玩家能建立差异化。结果：有效——Anthropic 在 2023 年凭此在企业市场建立了明确定位，尽管后来被 Gemini 1.5 超越。

**可见推理链（2025）**：赌透明度比效率更重要，赌用户更信任可以检查思考过程的模型。结果：尚待检验，但在企业和开发者市场，这个差异化定位有一定受众。

## 与主流路线的差异

Anthropic 是前沿实验室中**研究导向与产品导向结合最紧密**的一家。他们同时运营世界级的 AI 安全研究（机械可解释性）和商业竞争力极强的 Claude 产品线，这两者在大多数实验室里是割裂的——要么重研究（学术机构），要么重产品（纯商业公司）。

他们也是在对齐技术上路线最清晰的实验室：Constitutional AI 的宪法原则是可读的文本，不是黑箱偏好；RSP 是写下来的承诺，不是口头表态；可解释性研究是长期投入的基础科学，不是公关材料。不管这些路线最终是否成功，**Anthropic 是唯一一家把"安全做法应该可审计"作为组织原则的前沿实验室**。

## 信息源

- [Constitutional AI: Harmlessness from AI Feedback (arXiv 2212.08073)](https://arxiv.org/abs/2212.08073)
- [Toy Models of Superposition (Anthropic, 2022)](https://transformer-circuits.pub/2022/toy_model/index.html)
- [In-context Learning and Induction Heads / circuits work (Anthropic, 2022)](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html)
- [Scaling and evaluating sparse autoencoders (arXiv 2406.04093)](https://arxiv.org/abs/2406.04093)
- [Mapping the Mind of a Large Language Model (Anthropic, 2024)](https://www.anthropic.com/research/mapping-the-mind-of-a-large-language-model)
- [Anthropic's Responsible Scaling Policy (Anthropic, 2023)](https://www.anthropic.com/index/anthropics-responsible-scaling-policy)
- [Claude 3.7 Sonnet and Extended Thinking (Anthropic, 2025)](https://www.anthropic.com/news/claude-3-7-sonnet)
- [Claude Opus 4.6 Announcement (Anthropic, 2026)](https://www.anthropic.com/news/claude-opus-4-6)

## 更新日志

- 2026-05-02：创建初始版本（CAI/RLAIF、机械可解释性、长上下文先行、RSP、可见推理链，附 topics/ 引用）
- 2026-05-02：补充内联引用和信息源部分
- 2026-05-02：补充 Claude 4 系列（Opus 4/Sonnet 4 工具交织、4.5 工作流优化、4.6 百万上下文与 Agent Team、4.7 编码质变）
