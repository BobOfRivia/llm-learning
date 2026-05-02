# 数据工程

## 一句话定义

这条轨道覆盖"用什么数据训练 LLM"——从数据来源、处理、质量控制到合成数据，追踪数据作为 LLM 关键生产要素的演进。

## 演进脉络

### 阶段 1：Web 规模爬取（2018-2021）

这个阶段的核心假设是**数据量 > 数据质量**：只要有足够多的文本，模型能自己学会分辨。Common Crawl 作为主力数据源，经过基础的语言过滤和去重处理后直接用于预训练。GPT-3 的训练数据（~499B tokens）主要来自 Common Crawl 过滤版本 + WebText（Reddit 高赞链接）+ 书籍 + Wikipedia。

The Pile（EleutherAI，arXiv 2101.00027，2021 年）是这个阶段数据工程思维最系统的体现：825GB，由 22 个多样化子数据集组成，大量来自学术和专业领域（PubMed、arXiv、GitHub、FreeLaw 等）。The Pile 的设计原则是**多样性优于规模**——用有代表性的多领域数据替代单纯堆砌爬取量，成为开放 LLM 训练的事实标准数据集。

### 阶段 2：质量觉醒（2022-2023）

Chinchilla（2022 年 3 月）的核心结论让数据量的重要性被重新认识，但真正的问题不只是量——而是如何系统性地保证数据质量。这个阶段数据工程从"能用"进化为"用好"。

主要工程方向：**去重**（exact dedup + minhash fuzzy dedup，去除训练集内的重复和近重复文本）、**质量过滤**（训练分类器判断"是否像高质量来源"，如维基百科质量）、**数据混合比例**（代码、数学、多语言数据的混合比例成为关键超参数，代码数据被发现对推理能力有显著提升）。

RedPajama（Together AI，2023 年）是最具代表性的开放数据集工程实践：第一版复现 Llama 训练数据，第二版扩展到 30 万亿 tokens，并提供 40+ 预计算质量标注（ML 分类器打分 + minhash 去重 + 启发式过滤），允许社区自行切片和过滤。

Phi-1（Microsoft Research，arXiv 2306.11644，2023 年 6 月）提供了"质量 > 规模"命题最极端的实证：1.3B 模型 + 7B tokens 教科书质量数据（6B 筛选自网络 + 1B GPT-3.5 合成），HumanEval pass@1 达到 50.6%，超过大量数倍大的模型。Phi-1 开辟了**合成数据 + 小模型**的研究路线，并将数据质量的讨论从"过滤低质量"推进到"主动生成高质量"。

**参考**：[The Pile (arXiv 2101.00027)](https://arxiv.org/abs/2101.00027)，[Textbooks Are All You Need (arXiv 2306.11644)](https://arxiv.org/abs/2306.11644)

### 阶段 3：合成数据崛起（2023-2024）

这个阶段的核心转变是：**数据不只是被找到的，而是被生成的**。驱动力来自两个方向——高质量自然文本的供给开始出现天花板迹象，同时强模型的生成能力已经足够高到可以产出有效的训练信号。

**方法论演进：从模板到自适应**

Self-Instruct（ACL 2023，Wang et al.）是这条路线的起点：用基础 LLM 从少量种子任务开始，生成指令-回复对，再用这些数据微调自身，形成自我改进循环。限制在于生成质量受种子分布约束，多样性有上限。

Evol-Instruct（WizardLM/WizardCoder，2023-2024）引入进化策略，通过迭代增加任务复杂度来改善数据质量——从简单指令开始，逐步演化出更复杂、更有挑战性的变种。这一方法在代码生成上效果显著（WizardCoder 在 HumanEval 上超过同规模基线）。

Magpie（ICLR 2025）走得更远：完全放弃种子提示，直接利用已对齐 LLM 的左侧模板（系统提示位置）自动生成用户查询和回复。用 Llama-3-Instruct 生成了 400 万指令，筛选 30 万高质量实例后，部分任务表现与官方 Llama-3-8B-Instruct（基于 1000 万数据点训练）相当。这暗示对齐模型内部已经编码了足够多的"什么是好的用户任务"信息，可以被直接利用。

**Phi 系列：合成数据的压力测试**

Microsoft Research 的 Phi 系列是这个阶段合成数据路线最系统的实验场。Phi-1（2023）证明了方向可行，此后每个版本都在扩大合成数据的覆盖范围和比例：Phi-2（2.7B，250B tokens，混合合成与过滤网页数据）、Phi-3（3.8B，第一阶段通用知识 + 第二阶段重度过滤和逻辑推理合成数据）、Phi-4（2024，14B，约 40% 合成数据共 400B tokens，在 GPQA 和 MATH 上超越 GPT-4o）。Phi-4 是截至 2025 年初合成数据占比最高、效果最好的公开模型。

**后训练数据的批量化生产**

合成数据在预训练的探索是相对保守的，但在后训练（SFT 和偏好数据）领域已经全面主流化。UltraChat、OpenHermes、UltraFeedback 等数据集的出现使得开放社区可以用相对低成本构建具备指令跟随能力的模型。Constitutional AI（Anthropic）则是合成对齐数据最大规模的闭源应用——用原则列表让模型检查自身输出，生成偏好数据对，替代大量人工标注。

**Model Collapse：真实风险，需要混合策略**

"用模型生成数据训练模型是否会退化"这个问题在 2024 年有了比较明确的实验答案。Nature 2024 年的研究（Shumailov et al.）通过递归训练实验证实：每一代模型在词汇多样性、句法丰富度、语义覆盖上持续下降，原始分布的尾部信息（低频但重要的内容）首先丢失。关键的阶段性发现是即使非常小的合成数据比例（1/1000）长期也会产生影响。核心结论是**纯合成数据的递归训练必然导致退化，实践中需要持续注入真实数据来锚定分布**。这一结论直接制约了合成数据在预训练中的上限，但在后训练阶段影响相对较小（因为后训练的分布锚点由预训练已经建立）。

**参考**：[Self-Instruct (ACL 2023)](https://arxiv.org/abs/2212.10560)，[Magpie (ICLR 2025)](https://arxiv.org/abs/2406.08464)，[Phi-4 Technical Report](https://arxiv.org/abs/2412.08905)，[AI models collapse when trained on recursively generated data (Nature 2024)](https://www.nature.com/articles/s41586-024-07566-y)

### 阶段 4：数据墙与突围（2024-至今）

**数据集工程的竞争化**

这个阶段的一个结构性变化是：**高质量公开预训练数据集本身成为研究竞争对象**。DCLM（DataComp for Language Models，NeurIPS 2024）建立了一个以数据过滤为核心变量的基准体系：240T tokens 的 Common Crawl 原始语料库，在标准化的模型规模和训练计算量下，比较不同过滤策略的效果。DCLM-Baseline（7B 参数，2.6T tokens 训练）在 MMLU 5-shot 达到 64%，比此前开放数据 SOTA（MAP-Neo）高 6.6 个百分点，计算量减少 40%。DCLM 的核心发现是**基于模型的过滤（用分类器打分决定保留哪些文档）显著优于启发式过滤**，这一结论推动了整个开源社区从规则过滤转向 ML 过滤。

Hugging Face 的 FineWeb（2024 年 4 月）是目前最大的公开预训练数据集：15 万亿 tokens，来源于 Common Crawl 96 个转储（2013-2024 年），使用 datatrove 库进行清理、过滤和去重。FineWeb-Edu 子集进一步用教育内容分类器筛选出 1.3T tokens 教育质量文本——在知识密集型基准（MMLU 等）上性能显著提升，用更少的数据获得更好的效果，是"教科书质量"假设的大规模验证。

Meta 的 Llama 3（2024 年 4 月）训练数据规模达到 15 万亿 tokens，代码数据量是 Llama 2 的 4 倍，非英文数据占比超过 5%。Llama 3 的数据工程亮点在于分类器驱动的质量过滤（fastText + RoBERTa 双重打分）和语义去重（在 15T tokens 规模下做去重本身是工程挑战）。

**数据墙：时间表已经出现**

Epoch AI 的研究提供了目前最具体的预测：公网可用的高质量人工生成文本约 300 万亿 tokens（去重调整后）。如果当前训练规模扩展速度（约 4 倍/年）持续，**高质量文本将在 2026-2032 年之间趋近耗尽**，过度训练情景下可能提前到 2026 年。这不是"文本消失"，而是高质量、未被使用过的文本越来越稀缺，边际收益递减。

应对策略分三个方向。第一是合成数据规模化，目前在数学、代码等高结构化领域已经验证有效（DeepSeek、Qwen 等），但在通用知识领域受 model collapse 风险约束。第二是多模态数据扩展，图像-文本对、视频字幕、科学图表在理论上能大幅扩展可用数据量，但如何有效转化为语言模型的训练信号仍在探索中。第三是领域专有数据的获取，通过商业授权（OpenAI、Google 与 Reddit 的数据协议，2024 年 5 月）、科学数据库合作等方式获取未被普遍使用的高质量语料。

**版权争议：法律格局在形成中**

数据获取的法律边界在 2024-2025 年开始清晰化，但尚未收敛。《纽约时报》诉 OpenAI（2023 年 12 月起诉，2025 年 3 月法官批准诉讼推进）是最高知名度的案件，核心争议是 LLM 偶尔能逐字复现训练数据这一事实是否构成侵权。Bartz v. Anthropic（2025 年）的判决认定用合法获得的书籍训练 AI 属于"高度变革性"的合理使用，但明确排除了从盗版来源获取数据——这条区分线正在成为行业共识：**训练行为本身可能受合理使用保护，但数据获取渠道的合法性单独审查**。Kadrey v. Meta 同年作出类似合理使用认定。

实践层面，主要实验室已开始大规模签署授权协议，估计 2024 年 AI 数据集许可市场规模约 3.8 亿美元。未来的影响更多在于"能用什么数据"的边界收窄，而不是直接终止现有训练实践。

**参考**：[DataComp-LM (NeurIPS 2024)](https://arxiv.org/abs/2406.11794)，[The FineWeb Datasets (arXiv 2406.17557)](https://arxiv.org/abs/2406.17557)，[Epoch AI: Will we run out of data?](https://epoch.ai/blog/will-we-run-out-of-data-limits-of-llm-scaling-based-on-human-generated-data)

## 当前技术格局（截至 2026-05）

预训练数据工程的主流实践已经从"爬取 + 启发式过滤"演进为**"爬取 + ML 分类器过滤 + 语义去重"**的标准流水线。DCLM、FineWeb、Dolma 等公开数据集为开放模型提供了高质量基线，模型过滤打分已成为过滤流水线的必要环节而非可选项。

合成数据的地位呈现明显的分层结构：在后训练（SFT、偏好数据）中已全面主流化，质量和规模都可以接受；在预训练中仅在高结构化领域（数学、代码）被证明有效，通用预训练中受 model collapse 约束只能作为补充。Phi-4（14B）是目前合成预训练数据占比最高的公开强模型（约 40%），但这一比例是否可以继续提升尚无定论。

数据版权的法律框架正在形成，合理使用保护范围逐渐明确，但获取渠道合法性成为新的合规重点。主要实验室在商业授权上的投入持续增加。

数据墙是真实约束而非噩梦场景——高质量文本的耗尽有时间窗口，当前仍可操作，但倒逼了合成数据和多模态数据的加速研发。

## 关键分歧与未决问题

**合成数据的预训练上限在哪里**。后训练中合成数据的效果已经比较清楚，但预训练中 model collapse 的实际阈值是多少、特定领域上限和通用数据上限是否不同，目前没有共识。Phi-4 的 40% 是一个实验数据点，但不是结论。

**数据墙的时间线是否会被多模态突破**。Epoch AI 的预测基于文本数据，图像-文本对和视频字幕理论上能延伸可用数据量，但"多模态数据对语言能力的贡献效率"这一转换系数尚不清楚。

**教育质量过滤是普遍规律还是特定基准的过拟合**。FineWeb-Edu 在 MMLU 类基准上表现显著更好，但这是真实的知识质量提升，还是和 MMLU 类评测数据分布更接近导致的评测偏差？这个问题对如何设计过滤策略有实质影响。

**数据版权争议的最终落点**。合理使用认定正在形成，但各国法律体系不同，上诉结果可能推翻现有判决。这对全球数据获取策略有长期影响。

## 对能力输出的影响

代码数据的占比是影响推理能力和 tool use 能力最有据可查的数据工程变量。代码具有高度形式化的逻辑结构，训练时能强化符号推理；Llama 3 代码数据 4 倍增量被认为是其推理能力提升的主要原因之一，指向 [tracks/training.md] 中代码数据与推理能力的关联章节。

教育质量数据（FineWeb-Edu）在知识密集型任务上的提升暗示数据的"认知密度"是一个独立变量——同等 token 数量下，教科书级文本对知识存储和检索能力的贡献高于网页流水线文本，对应 [capabilities/reasoning.md] 和 [capabilities/instruction-following.md]。

合成数学数据是推理纪元（era4）的关键基础设施之一。DeepSeek-R1 等模型大量使用合成数学问题和解题过程作为 RL 训练数据，使推理链的质量产生数量级提升，指向 [capabilities/inference-time-compute.md]。

多语言数据比例直接决定非英文语言的能力水位，这是一个相对独立的工程变量，各家实验室的多语言数据策略差异显著。

## 与其他轨道的交叉

- **scaling**：数据量是 scaling 三要素之一（算力、参数、数据）。Chinchilla 重新定义了最优数据-参数比，但数据墙的出现使这一比例在实践中不得不向过训练方向偏移，指向 [tracks/scaling.md]
- **training**：合成数据主要落地在后训练阶段（SFT、RLHF/DPO 的偏好数据）。预训练合成数据受 model collapse 约束，后训练合成数据的比例和质量标准是当前研究热点，指向 [tracks/training.md]
- **alignment**：偏好数据的质量（多样性、噪声率、标注一致性）直接影响对齐效果。Constitutional AI 是合成偏好数据在对齐中最大规模的应用，指向 [tracks/alignment.md]
- **inference**：蒸馏作为一种特殊的合成数据生成（用强模型输出训练弱模型）和 model collapse 风险有交叉，指向 [tracks/inference.md]

## 信息源

**关键论文**
- [The Pile (EleutherAI, arXiv 2101.00027)](https://arxiv.org/abs/2101.00027)
- [Textbooks Are All You Need / Phi-1 (arXiv 2306.11644)](https://arxiv.org/abs/2306.11644)
- [Phi-4 Technical Report (arXiv 2412.08905)](https://arxiv.org/abs/2412.08905)
- [Self-Instruct (ACL 2023, arXiv 2212.10560)](https://arxiv.org/abs/2212.10560)
- [Magpie (ICLR 2025, arXiv 2406.08464)](https://arxiv.org/abs/2406.08464)
- [Model collapse / Shumailov et al. (Nature 2024)](https://www.nature.com/articles/s41586-024-07566-y)
- [DataComp-LM / DCLM (NeurIPS 2024, arXiv 2406.11794)](https://arxiv.org/abs/2406.11794)
- [FineWeb (arXiv 2406.17557)](https://arxiv.org/abs/2406.17557)
- [Dolma (ACL 2024, arXiv 2402.00159)](https://arxiv.org/abs/2402.00159)

**数据/基准资源**
- [RedPajama-Data-v2 (Together AI)](https://github.com/togethercomputer/RedPajama-Data)
- [DataComp 官网](https://datacomp.ai/dclm/)
- [FineWeb on Hugging Face](https://huggingface.co/datasets/HuggingFaceFW/fineweb)

**分析与预测**
- [Epoch AI: Will we run out of data?](https://epoch.ai/blog/will-we-run-out-of-data-limits-of-llm-scaling-based-on-human-generated-data)

## 更新日志

- 2026-05-02：创建骨架
- 2026-05-02：填充阶段 1（Web 规模爬取，Common Crawl + The Pile）和阶段 2（质量觉醒，RedPajama + Phi-1）
- 2026-05-02：填充阶段 3（合成数据崛起，Self-Instruct / Phi / Magpie / Model Collapse）和阶段 4（数据墙与突围，DCLM / FineWeb / Epoch AI / 版权）；填充当前格局、分歧、能力影响、信息源
