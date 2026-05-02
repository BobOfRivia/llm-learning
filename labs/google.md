# Google / DeepMind

## 技术路线

Google 的 AI 故事是这个时代最有张力的叙事之一：**发明了 Transformer，却在 LLM 时代初期被自己的发明反将一军**。Google Brain（负责 Transformer、BERT、T5）和 DeepMind（负责 AlphaGo、AlphaFold、Chinchilla）是两个有不同文化的研究机构，两者在 2023 年合并为 Google DeepMind，这次整合影响深远，但也带来了相当的组织阵痛。

### 技术先驱，产品落后（2017-2022）

2017 年的"Attention Is All You Need"（Vaswani et al., Google Brain）是这个时代所有进展的技术起点。2018 年 BERT、2019 年 T5——Google 在 NLP 基础研究上的技术积累是无可争辩的最强。

但 Google 没有率先把这些能力产品化。原因是结构性的：Google 的核心业务（搜索广告）建立在关键词匹配和 PageRank 的技术范式上，内部有强大的既得利益反对可能蚕食搜索流量的新产品。当 ChatGPT 在 2022 年 11 月爆发时，Google 内部爆发了"Code Red"（红色警报）——这是真实的内部危机响应，表明他们此前没有充分预见到对话式 AI 作为搜索替代的威胁。

**参考**：[Attention Is All You Need (arXiv 1706.03762)](https://arxiv.org/abs/1706.03762)，[BERT (arXiv 1810.04805)](https://arxiv.org/abs/1810.04805)，[Exploring the Limits of Transfer Learning with T5 (arXiv 1910.10683)](https://arxiv.org/abs/1910.10683)

### Chinchilla 与 scaling 认知贡献（2022）

Google DeepMind（彼时还是 DeepMind）对整个行业最重要的科学贡献之一是 Chinchilla（Hoffmann et al., 2022 年 3 月）。这篇论文修正了 Kaplan Scaling Laws 的最优训练配比，将"参数主导"的理解改为"参数与数据等比扩展"，事实上让全行业重新审视了训练预算的分配方式（见 tracks/scaling.md 阶段 2）。

讽刺的是，Chinchilla 发布时 Google 自己的 PaLM（540B，2022 年 4 月）是一个按旧配比训练的"undertrained"大模型——正是 Chinchilla 批评的那类模型。这说明研究组织的内部信息传递和决策协同在这个时期并不完美。

**参考**：[Training Compute-Optimal Large Language Models / Chinchilla (arXiv 2203.15556)](https://arxiv.org/abs/2203.15556)，[PaLM: Scaling Language Modeling with Pathways (arXiv 2204.02311)](https://arxiv.org/abs/2204.02311)

### PaLM 到 Gemini 的路线迁移（2022-2023）

PaLM（Pathways Language Model，2022 年 4 月，540B dense 参数）是 Google 在 GPT-3 之后的规模化回应，训练于 780B tokens，在多项基准上超过 GPT-3。PaLM 2（2023 年 5 月）是改进版，更多语言和代码数据，Chinchilla 配比调整。

但 PaLM 路线的问题是：它是一个跟随 OpenAI 架构路线的纯语言模型，缺乏差异化。ChatGPT 的冲击让 Google 意识到需要根本性的路线重构，而不只是参数更大的下一代 PaLM。**Gemini 项目（2023 年）** 是这次重构的产物，目标是从根基上重新设计，而非渐进升级。

### Gemini：原生多模态路线（2023-今）

**Gemini 1.0（2023 年 12 月）** 是 Google 最重要的架构路线赌注：**原生多模态（natively multimodal）训练**。GPT-4V 的实现是把 CLIP 类的视觉编码器输出投射到语言模型输入层——多模态是后天嫁接的。Gemini 的设计目标是从预训练起就在交织的图文、音频、视频数据上联合建模，使不同模态的理解统一在同一个模型表示空间里。

这个赌注的理论基础是：模态间的深层语义关联只有在预训练时联合学习才能习得；嫁接式的方案会在模态边界产生语义对齐损耗。但验证这个假设在实践中很困难——2023 年 Gemini 1.0 的发布因演示视频被指出存在剪辑技巧而引发信任危机，使得技术路线的优势在噪声中难以被清晰感知。

**Gemini 1.5 Pro（2024 年 2 月）** 是更清晰的技术信号：100 万 token 的上下文窗口（研究版本达 1000 万），据报道采用 MoE 架构，在 needle-in-haystack 检索测试上接近完美。这是超长上下文方向最彻底的一次推进，把"是否需要 RAG"这个架构问题重新摆上了桌面。

**Gemini 2.5 Pro（2025 年 3 月）** 在多项推理基准上达到新高水位，标志着 Google 在推理纪元中追平了与 OpenAI/Anthropic 的差距。

**参考**：[Gemini: A Family of Highly Capable Multimodal Models (arXiv 2312.11805)](https://arxiv.org/abs/2312.11805)，[Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context (arXiv 2403.05530)](https://arxiv.org/abs/2403.05530)

### TPU 垂直整合的战略意义

Google 是唯一同时设计 AI 芯片（TPU）和训练 AI 模型的前沿实验室。TPU 的设计目标是矩阵乘法的极致优化，牺牲了通用计算灵活性换取 LLM 训练的效率。这给 Google 带来了两个优势：（1）**成本优势**——自研芯片的边际成本低于采购 NVIDIA GPU；（2）**架构协同**——可以根据模型架构需求定制芯片，反过来也可以根据 TPU 特性优化模型架构（Gemini 的 MoE 设计与 TPU 的互联拓扑之间存在协同设计）。

但 TPU 也有锁定效应——大量基于 PyTorch 的开源生态工具在 TPU 上的支持程度不如 NVIDIA GPU，这使 Google 的研究成果有时难以被社区快速复现和验证。

### DeepMind 路线：特定领域的深度推理

Google DeepMind 保留了 DeepMind 在特定领域深度推理方面的投入，与 Gemini 的通用语言模型路线并行：**AlphaCode 2**（2023 年，竞赛级代码生成）、**AlphaGeometry**（2024 年，几何证明题）、**AlphaProof**（2024 年，数学奥林匹克证明）。这些工作代表了一条与语言模型 Scaling 不同的路线——在特定领域通过专门设计的神经符号系统达到超人水平，而不是靠通用语言模型逼近。这两条路线在 Google 内部的资源竞争和路线权重问题上，是这个时期 Google AI 战略中最有意思的内部张力。

**参考**：[AlphaGeometry: An Olympiad-level AI system for geometry (Nature, 2024)](https://www.nature.com/articles/s41586-023-06747-5)，[AI achieves silver-medal standard solving IMO problems (DeepMind blog, 2024)](https://deepmind.google/discover/blog/ai-solves-imo-problems-at-silver-medal-level/)

### Gemini 2.x 到 3：从追赶到领先（2025-2026）

Gemini 2.5 Pro 之后（已在 eras/era4-reasoning.md 中记录），Google 在多模态和数学推理方向持续加速，并在 2025 年底以 Gemini 3 完成了对同期竞争对手的超越。

**Gemini 2.0 Flash（2025 年 1 月正式）** 是面向开发者和高频场景的快速推理版本，延续了 Gemini 2.0 的 agentic 定位（2024 年 12 月宣布时明确以"为 Agent 时代设计"为主题），原生支持工具调用和多模态输出（文本、图像、TTS 混合输出）。Gemini 2.0 Pro（2025 年 2 月）是对应的旗舰版，能力更强但推理速度较慢。

**Gemini 2.5 Pro（2025 年 3 月）**：见 eras/era4-reasoning.md。AIME 2024 达到 92%，LMArena 排行榜第一，是推理纪元中 Google 追平顶尖竞争对手的重要节点。

**Gemini 3（2025 年 11 月）** 是 Google 在多个维度同时达到 SoTA 的版本。Deep Think 推理模式在数学和科学推理上达到新高水位；最显著的里程碑是搭载 Deep Think 的版本在**2025 年国际数学奥林匹克（IMO）中达到金牌水平**——6 题做对 5 题（35 分），全程以自然语言输入输出，在规定的 4.5 小时竞赛时间内完成，不借助 Lean 等形式化证明语言。相比之下，2024 年 AlphaProof + AlphaGeometry 2 的联合方案达到银牌，但依赖形式化语言；Gemini 3 的纯自然语言端到端是一个方法论上的跃升。多模态能力在此版本全面达到业界最高水位。

**Gemini 3.1（2026 年）** 以 Pro 和 Flash-Lite 两档覆盖不同成本场景，延续了 Google 在每个能力层都有对应部署档位的产品策略。

2024-2026 年间，Google 完成了从"被 ChatGPT 冲击触发 Code Red"到"多模态和数学推理双线领先"的转型。原生多模态的长期赌注（Gemini 1.0 就开始）在 Gemini 3 上最终得到了明确验证：同时在视觉理解和数学推理上达到顶尖，而这两种能力的协同很难在嫁接式多模态方案中实现。

## 关键决策与赌注

**Chinchilla 修正 Scaling Laws（2022）**：这不是商业赌注，而是研究贡献。但它让 Google 在 Scaling 纪元的认知贡献上留下了最持久的一笔，即使这篇论文某种程度上指出了 Google 自己的 PaLM 也是"undertrained"的。

**原生多模态 Gemini 路线（2023）**：赌从预训练就多模态的效果优于后嫁接方案。结果：长期验证中。截至 2025 年，Gemini 2.0/2.5 系列在多模态任务上表现领先，但很难区分"原生多模态"还是"更多多模态训练数据"的贡献。

**1M Token 上下文（2024）**：赌超长上下文成为关键竞争壁垒，且 Google 凭借 TPU 成本优势可以维持这个壁垒。结果：方向是对的，但长上下文已经成为行业标配，差异化程度降低。

**TPU 垂直整合（长期）**：赌自研芯片在长期 AI 军备竞赛中带来可持续的成本优势。结果：尚在展开中，但 Google 在大规模训练成本上确实有相对优势。

## 与主流路线的差异

Google 是前沿实验室中**研究积累最深但产品化能力最弱**的（至少在 ChatGPT 时代初期）。他们发明了 Transformer，写了 Chinchilla，发展了 BERT——但这些成果的商业价值被 OpenAI 先收割了。这是组织规模和既得利益如何阻碍自我颠覆的典型案例。

2023 年 Google Brain 和 DeepMind 的合并，以及随后 Gemini 1.5 和 2.5 在推理基准上的追赶，表明这个组织已经在一定程度上完成了转型。但核心问题仍然存在：Google 的核心业务（广告）与强大的 AI 助手存在结构性张力——一个能直接回答所有问题的 AI 会减少搜索广告点击，这是 Google 最难面对的内部矛盾。

## 信息源

- [Attention Is All You Need (arXiv 1706.03762)](https://arxiv.org/abs/1706.03762)
- [BERT: Pre-training of Deep Bidirectional Transformers (arXiv 1810.04805)](https://arxiv.org/abs/1810.04805)
- [Exploring the Limits of Transfer Learning with T5 (arXiv 1910.10683)](https://arxiv.org/abs/1910.10683)
- [Training Compute-Optimal Large Language Models / Chinchilla (arXiv 2203.15556)](https://arxiv.org/abs/2203.15556)
- [PaLM: Scaling Language Modeling with Pathways (arXiv 2204.02311)](https://arxiv.org/abs/2204.02311)
- [Gemini: A Family of Highly Capable Multimodal Models (arXiv 2312.11805)](https://arxiv.org/abs/2312.11805)
- [Gemini 1.5 (arXiv 2403.05530)](https://arxiv.org/abs/2403.05530)
- [AlphaGeometry (Nature, 2024)](https://www.nature.com/articles/s41586-023-06747-5)
- [Google introduces Gemini 2.0 (Google Blog, 2024)](https://blog.google/technology/google-deepmind/google-gemini-ai-update-december-2024/)
- [Gemini 2.5: Our newest model with thinking (Google Blog, 2025)](https://blog.google/innovation-and-ai/models-and-research/google-deepmind/gemini-model-thinking-updates-march-2025/)
- [Gemini 3 with Deep Think achieves IMO gold medal (Google DeepMind Blog, 2025)](https://deepmind.google/blog/advanced-version-of-gemini-with-deep-think-officially-achieves-gold-medal-standard-at-the-international-mathematical-olympiad/)

## 更新日志

- 2026-05-02：创建初始版本（Transformer 先驱到 ChatGPT 冲击、Chinchilla 贡献、Gemini 原生多模态路线、TPU 垂直整合、DeepMind 特定领域路线）
- 2026-05-02：补充内联引用和信息源部分
- 2026-05-02：补充 Gemini 2.0/2.5 Pro/Gemini 3 迭代历史（agentic 设计、Deep Think 推理模式、IMO 2025 金牌、多模态 SoTA）
