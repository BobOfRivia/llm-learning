# Scaling

## 一句话定义

这条轨道覆盖"规模与性能的关系"——从 scaling laws 的发现、修正到推理时扩展，追踪 LLM 领域对"怎么花计算预算最划算"这个问题的理解演进。

## 演进脉络

### 阶段 1：Scaling Laws 的发现（2020）

Kaplan, McCandlish et al.（OpenAI，2020 年 1 月，arXiv 2001.08361）在超过 7 个数量级的规模上，发现了 loss 与模型参数量（N）、数据量（D）、计算量（C）之间稳定的幂律关系。最关键的实践结论：**固定计算预算下，应将约 73% 资源用于扩大模型参数，27% 用于数据**，即大约每参数 1.7 tokens 的训练量。核心直觉是大模型更具"样本效率"——同样的数据，大模型学得更好。

这个结论直接驱动了 GPT-3（175B，2020 年 5 月）的诞生，并在整个纪元内确立了"bigger is better"的投入逻辑。各家实验室在这个框架下竞相扩大参数量：PaLM（Google，540B dense，2022 年 4 月）、Megatron-Turing NLG（Microsoft + NVIDIA，530B）。

技术背景：Kaplan 的分析只计算了非 embedding 参数，实验也在相对较小的规模上进行——这两个因素后来被证明导致了系统性偏差。

**参考**：[Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361)

### 阶段 2：Chinchilla 修正（2022）

Hoffmann, Borgeaud et al.（DeepMind，2022 年 3 月，arXiv 2203.15556，NeurIPS 2022）重新系统实验了 scaling 的最优分配，结论颠覆了 Kaplan：模型参数量和训练 tokens **应等比扩展**，每个参数需要约 **20 tokens** 的训练数据。

核心验证实验：Chinchilla 70B 在 1.3T tokens 上训练，与 Gopher 280B 使用完全相同的计算预算（~5.76×10²³ FLOPs）。结果 Chinchilla 全面胜出——MMLU 67.5% vs Gopher 60.0%，同时超过 GPT-3（175B）和 Megatron-Turing NLG（530B）。

Kaplan 误差的根源：只计非 embedding 参数 + 小规模实验外推，使得最优 N/D 比向参数侧偏移了约 11-12 倍。

**影响**：Chinchilla 之后，所有主流模型的训练配比做了重新校准。Llama 系列采用 Chinchilla 建议的数据比例，并在此基础上进一步过训练（见阶段 3）。这是这个纪元内 scaling 理解最重要的校正。

**参考**：[Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556)

### 阶段 3：过训练与实用主义（2023-2024）

Chinchilla 给出的是一个明确定义的最优解：**固定计算预算下，最小化训练损失**。但这个目标函数和实际部署场景的目标函数存在根本性的不一致。

训练成本是**一次性**的——你花了多少 GPU 时训练一个模型，这笔钱就付了，不管之后服务多少请求。推理成本是**持续性**的——每一次用户请求都要跑一次前向传播，如果你服务的是数亿用户，推理成本会远超训练成本。因此，从商业角度最优的模型不一定是"在给定计算预算下训练损失最低的模型"，而是"**在可接受的质量下推理成本最低的模型**"——即尽可能小的参数量。

要在小参数量上达到高质量，解法是用更多的数据训练：在 Chinchilla 框架内，小模型可以通过更多训练步数来逼近大模型的能力水平（虽然永远无法完全追上，但往往够用）。Meta 的 Llama 系列是这一逻辑的最典型实践。

**Llama 1（2023 年 2 月）**：7B 在约 1T tokens 上训练。Chinchilla 建议 1T tokens 的最优模型规模约为 50B，而 Llama 1 7B 用了相同的数据量训练，属于显著过训练。结果是一个"远小于 Chinchilla 配比建议、但在推理成本上极具竞争力"的模型。

**Llama 2（2023 年 7 月）**：70B 在 2T tokens 上训练。Chinchilla 建议 2T tokens 对应约 100B 参数，Llama 2 70B 相对偏小，训练量偏高。Llama 2 7B 在约 2T tokens 上训练，过训练程度更为极端。

**Llama 3（2024 年 4 月）**：这一趋势到 Llama 3 时达到极端。8B 模型在约 15T tokens 上训练——Chinchilla 建议 8B 参数的最优训练量约为 160B tokens，Llama 3 的实际训练量是这个的**约 90 倍**。结果是 Llama 3 8B 在多数任务上超过 Llama 2 70B，参数量只有后者的 11%。

这一系列结果揭示了 Chinchilla 框架的一个重要限制：**它优化的是训练效率，不是部署效率**。当你把目标函数从"训练 FLOPs 最小化"改成"单次推理 FLOPs 最小化"，最优解会向更小的模型 + 更长的训练时间漂移。

这个阶段还带来了一个推论：**数据质量比数据数量更重要**。一旦决定了过训练策略，瓶颈就从"有没有足够多的数据"变成"数据够不够好"——低质量数据重复训练会导致性能退化，而高质量数据即使重复也有价值。Llama 3 的训练语料据 Meta 报告经过了极精细的质量过滤，其中代码、数学等高质量数据的比例显著高于 Llama 2。这一趋势在 Phi 系列（Microsoft）中走向极端——用"教科书质量"的合成数据训练极小参数量的强力模型（Phi-1、Phi-1.5、Phi-2）。

### 阶段 4：推理时扩展（2024-至今）

Scaling Laws 从建立之初描述的就是训练时计算与能力的关系——投入更多训练 FLOPs，能力按幂律增长。这个框架的隐含假设是：推理时计算是固定的（每个 token 一次前向传播），能力提升只能靠训练。o1 的发布使这个假设失效。

**推理时扩展的基本机制**

推理时扩展（inference-time compute scaling，也称 test-time compute scaling）的核心思路是：让模型在生成最终答案之前，先产生一段内部推理过程。这段推理过程消耗额外的计算（额外的 tokens），但为最终答案提供了类似"搜索"的功能——探索多条路径、验证中间步骤、在错误时回溯。

最简单的推理时扩展形式是 **Best-of-N（BoN）采样**：对同一个问题生成 N 个独立答案，用奖励模型（或验证器）选出最好的。随着 N 增大，被选中的答案质量按幂律提升——这和训练时 scaling 的曲线形状惊人相似。

更复杂的形式包括 **MCTS（蒙特卡洛树搜索）**：在推理链生成过程中，对每个分叉点评估多条续写路径，选择最有价值的路径继续展开，过程中引导模型向正确方向搜索。这是 AlphaGo 用于棋类的策略在语言推理上的应用——LLM 作为"策略网络"，过程奖励模型作为"价值网络"。

**o3 提供的关键实证**

Snell et al.（UC Berkeley，2024 年 8 月，arXiv 2408.03314）的"Scaling LLM Test-Time Compute Optimally"是这个方向最重要的理论分析：它证明推理时计算的最优分配策略依赖于问题的难度——简单问题用 Best-of-N 就够，复杂问题需要更精细的树搜索；在推理时投入更多计算，可以让小模型在某些任务上超过大模型的一次前向传播结果。

o3 的实验数据把这个理论具象化了。ARC-AGI 上 o3 high compute（约 88%）vs o3 low compute（约 47%）vs o1（约 32%）vs GPT-4o（约 5%）的梯度，清楚地展示了推理时计算的 scaling 曲线——在同一个模型上，投入更多计算的回报是系统的、可预测的。

**思维预算与成本权衡**

推理时扩展带来了一个新的设计参数：**思维预算（thinking budget）**——为一个请求分配多少推理 tokens。Claude 3.7 Sonnet 允许用户直接设置扩展思考的 token 上限；OpenAI 的 o3 提供 low/medium/high 三个预设档位（价格差约 20 倍）。

这使"计算成本"从用户不可见的后台参数变成了产品设计的一部分：用户需要权衡速度、精度和成本。对于高价值任务（法律合同审查、竞赛数学），high compute 合理；对于低时延要求的对话场景，fast 模式更合适。

这个权衡还带出了一个更深的问题：**推理时扩展和训练时扩展是可以互换的吗？** 答案是部分可以——对于确定性的数学/代码任务，更多推理时计算可以弥补训练不足；但对于需要大量世界知识的任务，推理时计算无法凭空产生模型不知道的知识。两个维度是互补而非替代的关系。

**参考**：[Scaling LLM Test-Time Compute Optimally (arXiv 2408.03314)](https://arxiv.org/abs/2408.03314)

## 当前技术格局（截至 2026-05）

Scaling 曲线在 2024-2026 年间从单轴（训练时算力）演化为三条并行的轴线。

**预训练轴：过训练已成行业标配，数据质量取代算力成为主要瓶颈。** Chinchilla 建议的 20:1 tokens-to-params 比例早已被打破——Llama 3 8B 在 15T tokens 上训练（约 1800:1），Qwen3-0.6B 达到约 60,000:1，Liquid AI LFM2.5-350M 甚至达到 80,000:1，且在这些极端过训练点上模型质量仍在持续改进。Sardana & Frankle（2024，arXiv 2401.00448）从理论上证明了这一方向的合理性：当服务规模达到数亿请求时，用更多数据训练更小的模型，总体经济成本（训练 + 推理）远低于 Chinchilla 建议的大模型。

**MoE 轴：参数量和激活计算量解耦，Scaling Laws 需要重写。** DeepSeek-V3（671B 总参数 / 37B 活跃参数）和 GPT-4（据报道类似架构）使得"模型规模"这个单一数字变得不再有意义。Ludziejewski et al.（ICLR 2025，arXiv 2502.05172）系统证明 MoE 模型的最优 scaling 行为无法从密集模型的 scaling laws 推导，需要同时考虑激活参数、总参数、专家数量、共享专家比例和数据量五个因子。反直觉的发现：在合理配置下，MoE 比同质量密集模型**更内存高效**，不仅仅是计算高效。

**推理时扩展轴：第二条独立的能力 scaling 曲线已确立。** 从 o1（2024 年 9 月）到 o3、DeepSeek-R1、Gemini 2.5 Pro，推理时计算现在是能力的显性设计参数。DeepSeek-R1 在 AIME 上的表现从无推理扩展时的 15.6% 提升到多数投票下的 86.7%，这条曲线呈对数线性（ICLR 2025，Inference Scaling Laws）。关键约束：推理时 scaling 的有效性取决于任务的**可验证性**——有明确验证标准的数学、代码任务效果显著；开放性写作任务因为缺乏可靠的过程评估信号，推理时 scaling 收益有限。

三轴并行的结果是：成本竞争格局从"谁能训更大的模型"转变为"谁能在三条轴线上找到最优的资源分配"。DeepSeek-V3 的训练成本据报告约 560 万美元——这在 2020 年的认知框架里是不可思议的，当时同等能力被认为需要数亿美元。

## 关键分歧与未决问题

**"预训练 scaling 触顶"之争是当前分歧最大的问题。** Ilya Sutskever（Safe Superintelligence）公开表示预训练收益已平台化；Sam Altman（OpenAI）坚持"no scaling wall"立场；Dario Amodei（Anthropic）措辞更谨慎，认为 scaling "可能会继续"。分歧的根源在于如何定义"scaling"的边界——如果把架构创新（MoE）、数据质量提升、后训练方法改进都算进去，能力曲线仍在上升；如果只看"堆更多高质量 tokens + 更多参数"的原始形式，增益确实在递减。这个争论没有单一正确答案，因为问的不是同一个问题。

**合成数据能否接棒高质量人类文本枯竭后的预训练 scaling，目前没有定论。** Epoch AI 的估算表明高质量英文文本在 2026-2032 年会被耗尽。理论上，合成数据可以延伸这条曲线——Phi-4 和 DeepSeek-R1 都用了大量合成数据，效果不差。但 Model Collapse 研究（Nature 2024）证明无限递归地用合成数据训练会导致退化，必须混入真实数据。合成数据是否能在保持真实数据混入的前提下无限扩展，实验证据不足。

**推理时 scaling 的上界问题：上界由可验证性决定，但可验证性本身可以工程化。** 当前的推理时 scaling 有效性在不同领域差异很大——物理 > 化学 > 生物，主要原因是前者有更确定的验证标准。这意味着推理时 scaling 的瓶颈不只是计算量，更是验证器质量。Scalable verification（如何训练能可靠评估复杂推理的验证器）是推理时 scaling 能否继续延伸的关键依赖。

**MoE 的长期 scaling 行为尚未系统研究到超大规模。** ICLR 2025 的 MoE scaling laws 实验最大在 2.7B active / 5B total 参数规模上验证，外推到数百亿活跃参数规模的准确性未知。GPT-4 和 DeepSeek-V3 等大型 MoE 模型的训练细节没有完整公开，业界的大规模 MoE 实验数据几乎不透明。

**小模型 + 大推理 vs 大模型 + 少推理的最优切换点，取决于任务特性。** 两者不是简单的替代关系——推理时 scaling 在某些任务上可以让小模型超越大模型的单次前向传播，但无法弥补参数量带来的世界知识差距。对于高度依赖知识检索的任务，大模型仍有不可替代的优势；对于结构化推理任务，小模型 + 验证器 + 搜索可以极具竞争力。

## 对能力输出的影响

**阶段 1（Scaling Laws 发现）→ 所有能力维度的基线提升。** Kaplan 的幂律关系确立了"更大参数 = 更好性能"的通用规律，这不只影响某个特定能力，而是使所有 benchmark 上的表现随模型规模系统性提升。这阶段最重要的能力变化是**知识储量**和**语言理解**——参数规模的提升使得模型能记住更多事实、理解更复杂的语境。

**阶段 2（Chinchilla 修正）→ 同等算力下能力提升，加速了 alignment 成本的合理化。** Chinchilla 的核心贡献是把相同计算预算分配给了更多数据——这对**指令跟随（instruction following）**和**基础推理（reasoning）**的影响尤为明显，因为这些能力对数据多样性非常敏感。

**阶段 3（过训练策略）→ 部署民主化，边缘推理的能力跃升。** Llama 3 8B 在多数任务上超越 Llama 2 70B 是这阶段最戏剧性的结果。这意味着**工具调用（tool use）**、**指令跟随**、**代码生成**这些能力开始在消费级 GPU 上可用，开源社区的部署门槛大幅降低，推动了 Agent 工程的民主化。

**阶段 4（推理时扩展）→ 数学/代码/科学推理的质变，但非所有能力受益均等。** o3 在 ARC-AGI 上从 5%（GPT-4o）跳到 88%（high compute 档位）是推理时 scaling 最极端的实证。受益最大的能力是有明确验证标准的**推理（reasoning）**和**代码生成**；受益有限的是**创意写作**、**开放对话**等验证信号弱的场景。→ 详见 [capabilities/reasoning.md](../capabilities/reasoning.md) 和 [capabilities/inference-time-compute.md](../capabilities/inference-time-compute.md)

## 与其他轨道的交叉

- **architecture**：MoE 改变了"规模"的定义（总参数 vs 活跃参数）
- **training**：RL for Reasoning 是推理时扩展的训练基础
- **inference**：推理时扩展直接增加推理成本，与推理优化形成张力
- **data**：数据墙限制了训练时 scaling 的空间

## 信息源

**奠基论文**
- [Scaling Laws for Neural Language Models (Kaplan et al., 2020, arXiv 2001.08361)](https://arxiv.org/abs/2001.08361) — 幂律关系的原始发现，建立了 N/D/C 的三元框架
- [Training Compute-Optimal Large Language Models (Hoffmann et al., 2022, arXiv 2203.15556)](https://arxiv.org/abs/2203.15556) — Chinchilla，修正了最优 N/D 比，是这条轨道最重要的单篇论文
- [Beyond Chinchilla-Optimal: Accounting for Inference in Language Model Scaling Laws (Sardana & Frankle, 2024, arXiv 2401.00448)](https://arxiv.org/abs/2401.00448) — 将推理成本纳入 scaling 优化目标，理论化了过训练策略的合理性

**推理时 Scaling**
- [Scaling LLM Test-Time Compute Optimally (Snell et al., UC Berkeley, 2024, arXiv 2408.03314)](https://arxiv.org/abs/2408.03314) — 系统分析推理时计算的最优分配策略，是 o1 时代最重要的理论支撑
- [Inference Scaling Laws (ICLR 2025)](https://proceedings.iclr.cc/paper_files/paper/2025/file/8c3caae2f725c8e2a55ecd600563d172-Paper-Conference.pdf) — 实证 Best-of-N 和过程验证的对数线性 scaling 曲线
- [Inference-Time Scaling for Complex Tasks (2025, arXiv 2504.00294)](https://arxiv.org/abs/2504.00294) — 综述推理时 scaling 在不同领域（物理/化学/生物）的收益差异

**MoE Scaling**
- [Joint MoE Scaling Laws: Mixture of Experts Can Be Memory Efficient (Ludziejewski et al., ICLR 2025, arXiv 2502.05172)](https://arxiv.org/abs/2502.05172) — 280+ 实验建立 MoE 专属的五因子 scaling laws，发现 MoE 比 dense 更内存高效

**行业模型与报告**
- [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via RL (arXiv 2501.12948)](https://arxiv.org/abs/2501.12948) — DeepSeek-R1 训练方法和推理时 scaling 实证数据
- [DeepSeek-V3 Technical Report (arXiv 2412.19437)](https://arxiv.org/abs/2412.19437) — MoE 架构和训练成本的详细披露

**分析文章**
- [Scaling Laws for LLMs: From GPT-3 to o3 (Cameron Wolfe, Substack)](https://cameronrwolfe.substack.com/p/llm-scaling-laws) — 清晰的历史叙事，覆盖所有主要转折点
- [AI companies hit a scaling wall (Platformer)](https://www.platformer.news/openai-google-scaling-laws-anthropic-ai/) — 整理了 Sutskever/Altman/Amodei 的公开立场对比

## 更新日志

- 2026-05-02：创建骨架
- 2026-05-02：填充阶段 1（Kaplan Scaling Laws）和阶段 2（Chinchilla 修正）
- 2026-05-02：填充阶段 3（过训练与实用主义：Chinchilla 最优 ≠ 部署最优，Llama 1/2/3 的逻辑链）
- 2026-05-02：填充阶段 4（推理时扩展：test-time compute scaling 曲线、Best-of-N、MCTS、思维预算与成本权衡）
- 2026-05-02：填充当前技术格局、关键分歧、对能力输出的影响、信息源
