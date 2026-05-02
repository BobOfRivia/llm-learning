# Meta

## 技术路线

Meta 的 AI 战略由一个不同于其他实验室的商业逻辑驱动：**Meta 不卖 AI，Meta 用 AI**。他们的核心业务（广告、社交推荐、内容理解）需要最好的 AI 能力，但他们的竞争优势不在于成为"AI API 供应商"。这个商业结构解释了他们为什么是前沿实验室中开源力度最大的——开放 Llama 权重不威胁他们的商业模式，反而能建立生态影响力，削弱 OpenAI 的平台垄断企图，同时让整个开源社区帮他们改进技术。

### Llama 系列与开放权重策略（2023-今）

**Llama 1（2023 年 2 月）** 是 Meta 开源策略的起点，但当时的开放仍然保守——以申请制方式面向研究者开放，商业使用受限。随后不久权重被泄露并在社区广泛传播，事实上开启了开放生态的爆发（Alpaca、Vicuna、WizardLM 等基于 Llama 1 的微调模型在数周内涌现）。Meta 对这次"泄露"的态度颇为暧昧——没有积极阻止，也没有官方更改许可证。

**Llama 2（2023 年 7 月）** 是战略升级：提供明确的商业许可证，70B 版本与 GPT-3.5 能力相当，同步发布了 RLHF 微调的 Llama 2-Chat 变体。这一次，开放是有意图、有设计的，不是泄露的结果。Meta 选择了一个非常具体的商业边界：月活跃用户超 7 亿的服务需要额外协议（事实上是针对可能与 Meta 竞争的超大平台），其余大多数商业使用可以免费。

**Llama 3（2024 年 4 月）** 将过训练策略推向极端。8B 模型在约 15T tokens 上训练，70B 也同样大量过训练；同步发布了 128K 词表的新 tokenizer（vs Llama 2 的 32K）提升了代码和多语言的 token 效率；配套推出了完整的后训练版本（Instruct 变体）。最引人注目的结果是 **Llama 3 8B 在多数基准上超过 Llama 2 70B**——用 11% 的参数量打败了上一代旗舰模型。2024 年晚些时候，Meta 还发布了 405B 版本，这是第一个开放权重的前沿级大模型。

**参考**：[LLaMA: Open and Efficient Foundation Language Models (arXiv 2302.13971)](https://arxiv.org/abs/2302.13971)，[Llama 2: Open Foundation and Fine-Tuned Chat Models (arXiv 2307.09288)](https://arxiv.org/abs/2307.09288)，[The Llama 3 Herd of Models (arXiv 2407.21783)](https://arxiv.org/abs/2407.21783)

### Llama 3.1 到 Llama 4：路线完成（2024-2025）

Llama 3 发布后，Meta 以接近每季度一次的节奏持续迭代，在一年内完成了从"开放权重的语言模型"到"开放权重的原生多模态 MoE 模型"的完整路线演进。

**Llama 3.1（2024 年 7 月 23 日）** 是 Llama 3 的重要扩展版，三个变化值得关注：上下文窗口从 8K 扩展到 **128K**（16 倍），405B 版本成为当时最大的公开权重模型（也是第一个开放权重的前沿级大模型），以及在多语言、工具调用方面的系统性改进。128K 的上下文使得 Llama 3.1 可以处理整本书或长代码库，弥补了此前与 Gemini 1.5 和 Claude 的最大差距。

**Llama 3.2（2024 年 9 月 25 日）** 是 Meta 第一次在开放权重模型中引入视觉能力，但实现路线是**嫁接式**的：11B 和 90B 视觉版本通过跨注意力层（cross-attention）将独立预训练的图像编码器接入语言主干——这与 Llama 4 的原生多模态不同。同步发布的 1B 和 3B 轻量版本面向边缘和移动场景，延续了"让每个人都能跑 LLM"的策略。

**Llama 3.3（2024 年 12 月 6 日）** 只有 70B 一个尺寸，但这个发布传递了一个重要信号：70B 版本在多数基准上匹配 Llama 3.1 405B 的性能——5.5 倍参数量的差异被后训练技术抹平。这是"过训练 + 精细后训练"路线最直接的能力验证。

**Llama 4（2025 年 4 月 5 日）** 是 Meta 目前最大的一次架构跳跃。模型家族分三档：**Scout**（~109B 总参数，17B 活跃参数，16 个专家，10M context），**Maverick**（~400B 总参数，17B 活跃参数，128 个专家，1M context），**Behemoth**（约 2T 总参数，288B 活跃参数，仍在训练中，定位为教师模型用于蒸馏 Scout 和 Maverick）。

Llama 4 最重要的技术转变是**从预训练起就是原生多模态**——Scout 和 Maverick 从一开始就在交织的图文数据上联合预训练，而不像 Llama 3.2 那样把视觉编码器嫁接到语言主干上。10M token 的上下文窗口（Scout）是当前开放权重模型的最高水位，且是 MoE 架构下实现的——稀疏激活使得 10M 上下文的实际计算成本可以控制在可接受范围内。

这条路线的完整性是值得关注的：从 Llama 1（纯语言，密集，8K）到 Llama 4（原生多模态，MoE，10M），每一步迭代都有清晰的技术目标，没有随大流做"追 benchmark"的随机功能堆叠。

**参考**：[Llama 3.1 Blog (Meta AI, 2024)](https://ai.meta.com/blog/meta-llama-3-1/)，[Llama 3.2 Blog (Meta AI, 2024)](https://ai.meta.com/blog/llama-3-2-connect-2024-vision-edge-mobile-devices/)，[Llama 4 Blog (Meta AI, 2025)](https://ai.meta.com/blog/llama-4-multimodal-intelligence/)

### 过训练作为工程哲学

Meta 是"过训练"策略最系统的实践者（见 tracks/scaling.md 阶段 3），但这背后有一个明确的产品逻辑：Meta 的 AI 能力要大规模部署在 WhatsApp（用户数十亿）、Messenger、Instagram、Facebook 的推荐系统里。在这种规模下，**推理成本是决定性的成本项**，而训练成本是一次性的。最小化部署模型的参数量，同时维持足够的能力——这是 Meta 内部效率优化的直接驱动力，而 Llama 系列的开放权重版本不过是这个内部优化目标的外部表达。

### FAIR：研究文化的根基

Meta 的 AI Research（FAIR，由 Yann LeCun 创立）是这家公司研究文化的根基。FAIR 有几个重要的技术遗产：PyTorch（2017 年开源，现已成为学术界和工业界的主流深度学习框架）、Faiss（大规模向量相似度搜索库）、早期的多模态和自监督学习研究（DINO、JEPA 系列是 LeCun 视觉架构路线的代表）。

Llama 系列不是 FAIR 的工作，而是 Meta 的 GenAI 部门，两个组织在文化上有一定差异——FAIR 更偏基础研究，GenAI 更偏产品。LeCun 本人对 LLM 路线持有保留意见（他认为纯语言预训练无法达到真正的 AGI），这种内部的路线分歧是 Meta AI 战略中有趣的张力来源。

## 关键决策与赌注

**开放权重作为平台战略（2023）**：赌开源生态建立会给 Meta 带来比销售 API 更大的长期回报——更多人基于 Llama 构建意味着 Meta 的架构选择、接口标准成为生态默认值，且整个社区帮助改进技术。结果：策略有效。Llama 系列成为开源 LLM 的事实标准基础模型，开源生态的繁荣程度超出了大多数预期。

**过训练小模型（2023-2024）**：赌推理成本优化比训练效率更重要，赌用户更需要跑得起的模型。结果：Llama 3 8B 超越 Llama 2 70B 是最有力的验证，也让"小而强"成为整个行业的共识。

**405B 开放权重（2024）**：赌开放前沿级模型权重不会威胁 Meta 的商业利益，且可以大幅提升开源生态的能力上限。结果：正在被验证，405B 模型的开放推动了开源推理模型的发展（如 DeepSeek-R1 的蒸馏版本就基于 Llama 架构）。

**PyTorch 生态押注（长期）**：Meta 通过开源 PyTorch 建立了深度学习框架的实际标准（TensorFlow 曾是主流，现已大幅式微）。这不是直接的 LLM 赌注，但为 Meta 在 AI 基础设施层建立了持久影响力。

## 与主流路线的差异

Meta 是前沿实验室中**唯一不把 AI API 作为商业模式的**。OpenAI、Anthropic、Google 都在出售 AI 能力访问权；Meta 在免费分享其 AI 能力的权重。这个差异不是技术差异，而是商业结构决定了技术策略——开源对 Meta 来说是低成本高回报的（他们需要最好的开源社区帮助他们改进模型），而对 OpenAI 来说是核心竞争力的泄露。

在 Scaling 策略上，Meta 是最早把"推理最优"和"训练最优"做系统区分并执行的实验室——Llama 1/2/3 的过训练路线早于这个区分被广泛讨论。这个实用主义的工程哲学（优化你真正的成本函数，而非优化论文里的评估指标）是 Meta 工程文化的体现。

## 信息源

- [LLaMA: Open and Efficient Foundation Language Models (arXiv 2302.13971)](https://arxiv.org/abs/2302.13971)
- [Llama 2: Open Foundation and Fine-Tuned Chat Models (arXiv 2307.09288)](https://arxiv.org/abs/2307.09288)
- [The Llama 3 Herd of Models (arXiv 2407.21783)](https://arxiv.org/abs/2407.21783)
- [Llama 3.1 Blog (Meta AI, 2024)](https://ai.meta.com/blog/meta-llama-3-1/)
- [Llama 3.2 Blog (Meta AI, 2024)](https://ai.meta.com/blog/llama-3-2-connect-2024-vision-edge-mobile-devices/)
- [Llama 4 Blog (Meta AI, 2025)](https://ai.meta.com/blog/llama-4-multimodal-intelligence/)
- [PyTorch: An Imperative Style, High-Performance Deep Learning Library (NeurIPS 2019)](https://papers.neurips.cc/paper/2019/hash/bdbca288fee7f92f2bfa9f7012727740-Abstract.html)

## 更新日志

- 2026-05-02：创建初始版本（开放权重策略逻辑、过训练工程哲学、FAIR 文化根基、关键赌注分析）
- 2026-05-02：补充内联引用和信息源部分
- 2026-05-02：补充 Llama 3.1/3.2/3.3/Llama 4 迭代历史（128K 上下文、视觉能力引入、原生多模态 MoE、1000 万 token 上下文）
