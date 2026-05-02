# 训练范式

## 一句话定义

这条轨道覆盖"如何把一个随机初始化的模型变成有用的 LLM"——从预训练到后训练的整条流水线，每个阶段的技术选择和演进。

## 演进脉络

### 阶段 1：预训练为王（2018-2021）

GPT → GPT-2 → GPT-3 的演进路线建立在一个核心假设上：**预训练做好就够了**。训练目标统一为自回归语言建模（下一 token 预测），在大量网络文本（Common Crawl 过滤版、WebText、书籍、Wikipedia）上预训练。主要使用方式是 few-shot prompting——不更新任何模型权重，直接用自然语言描述任务和示例。

这个范式的能力边界在于：模型很强，但不听话。GPT-3 能生成流畅文本和完成 few-shot 任务，但无法可靠地遵循指令、保持安全边界或对齐人类意图——它更像一个"自动补全引擎"而非"助手"。Codex（Chen et al., 2021 年 7 月，arXiv 2107.03374）通过在代码数据上特化微调，证明了定向微调（fine-tuning）对特定领域的巨大价值，但后训练尚未成为系统性工程。

**参考**：[Language Models are Few-Shot Learners (GPT-3)](https://arxiv.org/abs/2005.14165)，[Evaluating Large Language Models Trained on Code (Codex)](https://arxiv.org/abs/2107.03374)

### 阶段 2：后训练成为关键（2022-2023）

InstructGPT（Ouyang et al., OpenAI，2022 年 3 月，arXiv 2203.02155，NeurIPS 2022）是从"语言模型"到"助手"的关键转折。它确立了后训练的三步流水线：

1. **SFT（监督微调）**：收集人类撰写的高质量示范数据，对预训练模型进行监督微调
2. **RM（奖励模型训练）**：收集多个模型输出的人类偏好排名，训练一个预测偏好的奖励模型
3. **PPO（强化学习）**：用奖励模型作为信号，通过 PPO 算法对 SFT 模型做 RL 优化

关键结果：1.3B 的 InstructGPT 在人类评估中优于 175B 的原始 GPT-3。这证明了一个根本性事实：**对齐质量不靠规模驱动，靠训练信号**。规模是能力的必要条件，但有用性（helpfulness）和安全性（safety）需要专门的后训练才能获得。

Anthropic 同期在探索 Constitutional AI（CAI）路线——用一套宪法原则替代人类反馈中的部分标注，作为 RLHF 的补充和替代方案。

**参考**：[Training language models to follow instructions with human feedback (InstructGPT)](https://arxiv.org/abs/2203.02155)

### 阶段 3：后训练技术多样化（2023-2024）

这个阶段的特征是后训练从"RLHF 一统天下"变成多条路线并行竞争，同时两个新的训练目标进入后训练流水线：工具使用能力和合成数据的大规模应用。

**对齐方法的多元化**

DPO（Direct Preference Optimization，Rafailov et al., 斯坦福，2023 年 5 月，见 alignment.md 阶段 2）的发布是最重要的单一事件：把 RLHF 的三阶段流程（SFT→RM→PPO）压缩为直接在偏好数据上训练的单阶段分类损失，无需显式的奖励模型和 PPO 采样。DPO 的工程简洁性使它迅速成为学术界和开源社区的标准选择。

DPO 之后，偏好学习的探索空间被打开了，一批变体涌现：KTO（基于卡尼曼-特沃斯基效用函数，可以使用单边反馈而非偏好对）、IPO（Identity Policy Optimization，修正 DPO 的分布偏移问题）、ORPO（Odds Ratio Preference Optimization，把 SFT 和偏好学习合并为单一损失）。这些变体的涌现本身就说明了一个事实：DPO 打开了一个新的优化空间，但还没有收敛到最终答案。在实际工业部署中，头部实验室（OpenAI、Anthropic、Google）仍然普遍使用 PPO 变体——DPO 的优势主要体现在中小规模实验和开源社区。

**工具使用的专项训练**

2023 年，工具使用（function calling / tool use）从"模型能勉强做到"变成了"模型经过专门训练的原生能力"。OpenAI 在 2023 年 6 月推出 function calling API，允许开发者以 JSON schema 格式定义函数，模型学会识别何时应该调用工具、如何生成规范的参数格式。

训练这个能力需要专门的数据：（1）**SFT 阶段**：大量包含工具调用模式的示范数据，覆盖"什么时候该调工具、什么时候不该调"、"如何把用户意图映射到函数参数"等决策；（2）**强化学习阶段**：以工具调用的实际执行结果作为反馈信号——调用成功、结果有用则正向奖励，调用参数错误或工具选择不当则负向奖励。这是后训练流水线中第一次系统性地引入**执行反馈**（execution feedback），而不只是人类偏好。

**合成数据进入后训练流水线**

这个阶段最深远的变化之一是合成数据开始大规模用于指令微调（instruction tuning）数据的生成，逐渐取代完全依赖人工撰写的 SFT 数据。

几个关键路线：**Self-Instruct**（Wang et al., 2022，arXiv 2212.10560）是早期范式——用模型本身生成指令-回答对，再用质量过滤器筛选；**WizardLM**（Xu et al., 2023，arXiv 2304.12244）在此基础上引入"进化指令"（Evol-Instruct）——用 GPT-4 对简单指令进行复杂化变体（增加约束、加入背景、要求多步推理），使数据难度自动提升；**Orca**（Mukherjee et al., Microsoft，2023，arXiv 2306.02707）的核心洞见是：模型应该学习的不只是 GPT-4 的**答案**，而是 GPT-4 的**推理过程**——在 SFT 数据中附上详细的思维链（system prompt 要求 GPT-4 解释推理步骤），使小模型习得大模型的推理范式。

这些方法的共同影响是：到 2023 年底，大多数开源指令微调数据集（Alpaca、Dolly、OpenHermes 等）的主体已经是 GPT-3.5 或 GPT-4 生成的合成数据，而非人工撰写。这产生了一个显著的实践效果——微调成本大幅下降（生成数据比请人标注便宜几个数量级），但也引入了新的风险：合成数据的分布偏差会被直接放大（"模型蒸馏"到开放模型的同时，GPT-4 的风格偏好和错误也会被蒸馏进去）。

**更新日志中的技术债**

一个需要在未来更深入展开的问题：预训练阶段的合成数据在这个时期也开始被探索（Phi 系列用"教科书质量"合成数据进行预训练），这条线和后训练合成数据是两个不同的问题，应在 data.md 中分别追踪。

### 阶段 4：推理训练（2024-至今）

这个阶段的核心问题是：**如何训练模型产生有效的推理过程**，而不只是训练它输出正确答案。这不是一个微小的技术调整，而是对训练目标函数的根本性重新设计。

**RLVR：可验证奖励的强化学习**

此前的 RLHF 依赖人类偏好作为奖励信号——人类判断哪个回答更好。这在开放域任务上是必要的，但存在一个根本性瓶颈：人类评估是慢的、贵的，且在专业领域（高级数学、竞赛编程）经常不可靠。

RLVR（RL with Verifiable Rewards）的洞见是：**在答案可以被机器验证的领域，可以用验证结果作为奖励信号，完全绕开人类偏好标注**。数学答案（最终数值正确与否）、代码（编译是否通过 + 测试用例通过率）、逻辑谜题（答案是否满足约束）都属于此类。这使 RL 训练的规模化不再受限于人工标注。

**PRM vs ORM：奖励粒度的权衡**

结果奖励模型（ORM，Outcome Reward Model）只根据最终答案给奖励；过程奖励模型（PRM，Process Reward Model）对推理链的每个步骤打分。OpenAI 的"Let's Verify Step by Step"（Lightman et al., 2023，arXiv 2305.20050）在 MATH 数据集上系统比较了两者：PRM 在找到正确解法上显著优于 ORM，原因是 ORM 会奖励"碰巧得到正确答案的错误推理链"，而 PRM 强制要求每一步都正确。

PRM 的代价是数据标注成本更高——需要对每一步推理打标，而不只是对最终答案打标。实践中的解决方案是**自动化过程验证**：对于数学，可以用符号系统或另一个模型验证每步的正确性；对于代码，可以用单元测试；Math-Shepherd（2024）引入了自动化逐步验证的方法。

**GRPO 与 DeepSeek-R1 的纯 RL 路线**

o1 的训练细节从未被 OpenAI 公开。DeepSeek-R1（2025 年 1 月，arXiv 2501.12948）的开放性使我们可以从其训练细节推断这条路线的机制。

DeepSeek 使用 **GRPO（Group Relative Policy Optimization）**，而非 PPO。GRPO 的简化在于：对同一个问题采样一组答案（一组 G 个样本），以组内的相对分数作为优势估计（advantage），省去了需要单独训练的 value model（PPO 需要维护 4 个模型：reference policy、reward model、value model、policy；GRPO 只需要 reference policy + reward model）。这在工程实现上大幅降低了复杂度。

**DeepSeek-R1-Zero** 的实验是这个阶段最重要的实证之一：从基础模型出发，不经过任何 SFT，只用 GRPO + 可验证奖励训练。模型在训练过程中**自发涌现了自我反思和修正行为**——在遇到困难时停下来、检查之前的步骤、然后重新尝试。这个"顿悟时刻"（aha moment）没有被显式训练，是从奖励信号中自然涌现的策略，与 GPT-3 的 in-context learning 涌现有相似的惊喜感。

最终的 R1 在 R1-Zero 基础上增加了"冷启动"策略：先用少量（几千条）高质量思维链 SFT 数据让模型习得格式，再做 RL。冷启动解决了纯 RL 容易产生的推理格式混乱问题（如语言切换、格式不稳定），加速了 RL 收敛。

**推理蒸馏：能力的可转移性**

DeepSeek-R1 同步发布的蒸馏结果证明：**推理能力可以从大的推理模型蒸馏到小的普通模型**。具体做法是用 R1 生成的思维链作为 SFT 数据，对 Llama 3 8B/70B 等开放基础模型做监督微调。结果令人意外：蒸馏得到的 R1-Distill 系列在推理任务上显著优于用同等 SFT 数据但没有推理链的模型，且优于规模相当的纯 RL 训练模型。这说明大模型生成的高质量推理链是一种有效的知识载体，监督学习从中获益。

**Agentic RL 与后训练规模化（2025）**

推理训练的边界在 2025 年开始向 agentic 任务延伸。单步推理的 RL 训练（数学、代码）相对容易：奖励信号清晰，轨迹短，验证成本低。但 agent 任务需要跨越多步工具调用，奖励稀疏性更严重。Agent-R1（arXiv 2511.14460）将 MDP 框架显式扩展到多步 LLM agent，引入工具调用成功率作为中间过程奖励；NeurIPS 2025 的实践指南（arXiv 2510.01132）系统性地探讨了 SFT 和 RL 阶段在固定计算预算内的最优比例，以及不同奖励密度对训练稳定性的影响。

后训练计算量的持续上升是这个阶段最值得记录的结构性变化。DeepSeek 的训练报告显示，RL 后训练的计算量已超过预训练计算的 10%。几年前后训练被视为"微调"，计算成本可以忽略；这一比例的上升标志着后训练正式成为能力获取的主要来源之一，而不只是预训练后的收尾工序。

在对齐机制上，RLAIF（RL from AI Feedback）在 2024-2025 年完成了从 Anthropic Constitutional AI 的特定实践到行业通行方法的转变。Lee et al.（arXiv 2309.00267，ICML 2024）的系统性对比表明 RLAIF 在总结和对话任务上可以达到与 RLHF 相当的效果；Meta 的 Self-Taught Evaluator（2024）进一步表明，纯 AI 生成数据训练的评估器在 RewardBench 上能超过 GPT-4 作为评估者。现实中各家实验室采用混合方案：人工反馈聚焦于最难判断的高价值样本，AI 反馈承担大规模中等难度标注。纯 RLHF 在大规模后训练中已不可行，混合方案是现实路径。

OpenAI 的 o3 引入了 Deliberative Alignment：在 chain-of-thought 推理过程中主动提示模型参考安全政策文本，将对齐判断嵌入思维链而不只是作用于最终输出。这在方向上与 Constitutional AI 相呼应（都是让模型显式对照原则行动），但运作在推理时而非训练时——测试的是"推理时对齐是否比训练时对齐更鲁棒"这个仍然开放的问题。

**参考**：[Let's Verify Step by Step (arXiv 2305.20050)](https://arxiv.org/abs/2305.20050)，[DeepSeek-R1 (arXiv 2501.12948)](https://arxiv.org/abs/2501.12948)，[RLAIF vs RLHF (arXiv 2309.00267)](https://arxiv.org/abs/2309.00267)，[Agent-R1 (arXiv 2511.14460)](https://arxiv.org/abs/2511.14460)，[Practical Guide to Multi-turn Agentic RL (arXiv 2510.01132)](https://arxiv.org/abs/2510.01132)

## 当前技术格局（截至 2026-05）

头部实验室的后训练流水线已经是多阶段、多目标的复合工程：大规模预训练 → SFT（主要使用合成数据，覆盖指令遵循、工具使用格式、安全边界）→ 基础对齐（DPO 或 PPO 变体，目标是 helpfulness 和 harmlessness）→ RL for Reasoning（RLVR，针对数学、代码、逻辑的可验证奖励）→ 可选的 Agentic RL（多步工具使用任务）。具体的阶段顺序和计算比例在各家之间有显著差异，但这条骨架已经是行业共识。

推理训练已完成从"实验性"到"标准品类"的转变。OpenAI 的 o1/o3/o4-mini 系列、Anthropic 的 Claude 3.7 Sonnet（extended thinking 模式，基于 RL 训练，支持可调节的思维 token 预算，最高 128K tokens）、Google 的 Gemini 2.0 Flash Thinking、DeepSeek-R1 以及开源推理模型（QwQ、Sky-T1 等）构成了完整的竞争格局。推理模型的 benchmark 表现（AIME、MATH-500、Codeforces 评级）已成为实验室综合实力比拼的核心维度。

奖励模型的实践选择上，PRM 在推理密集型任务（MATH-500、竞赛数学）上的优越性已获广泛认可，OpenAI 的 PRM800K 数据集是重要的行业参考基准。但 PRM 的步骤级标注成本和自动化验证难度仍然是工程障碍——在开放域任务上，ORM + 人工偏好标注仍然是更常见的选择。

合成数据已成为后训练 SFT 数据的绝对主体。几乎所有公开的指令微调数据集中，GPT-4 或类似模型生成的数据占据绝大部分；推理模型生成的高质量思维链（如 R1-Distill 的蒸馏路线）成为开源社区获取推理能力的主要来源之一。从人工撰写示范数据到合成数据主导，这一转变在短短三年内完成，根本驱动力是规模——人工标注的速度和成本无法支撑训练所需的数据量级。

## 关键分歧与未决问题

**RL for Reasoning 的 scaling 上限**

可验证奖励（RLVR）的有效性依赖"答案能被机器验证"这一前提。在数学和代码领域，这一前提成立，RL 的 scaling curve 目前仍然平滑（o3 相比 o1 有大幅提升）。但在开放域任务中，可验证信号不存在，只能依赖奖励模型——而奖励模型本身有误差，会诱发 reward overoptimization（模型学会"骗过"奖励模型而不是真正提升能力）。两个域各自的 RL scaling 上限在哪里，目前没有人知道。

**预训练 vs 后训练的计算分配**

Chinchilla 定律明确了预训练阶段 tokens 和参数的最优比例，但没有类似的分析指导整体计算预算在预训练和后训练之间的分配。DeepSeek 给出的数据点（RL 后训练 > 10% 预训练计算）是真实的，但这个比例是否最优、是否随基础模型质量变化、是否因任务类型而异——都没有定论。这是比 Chinchilla 更难的优化问题，因为变量更多且收益函数更复杂。

**DPO vs PPO 变体的能力权衡**

DPO 工程上更简洁（无需 value model 和在线采样），PPO 在需要持续探索的任务（复杂推理、长轨迹 agent）上更有优势。头部实验室的实践是分层的：PPO 变体（包括 GRPO）用于推理训练，DPO 变体用于对齐训练。但这背后的系统性原理仍不完全清晰——是任务性质的区别，还是数据分布的区别，还是两者都有？ORPO、KTO、IPO 等变体都提出了不同的理论论证，尚未收敛到共识。DPO 对齐税普遍低于 RLHF 这一实证也还缺乏充分的理论解释。

**合成数据自我增强循环的可持续性**

合成数据 → 训练更好的模型 → 更好的模型生成更好的数据，这条闭环在短期内有效，但理论上存在天花板。模型坍缩（model collapse，Shumailov et al., 2024）的研究表明：多代合成数据训练可能使模型分布收缩，丢失稀有但重要的能力。目前没有足够长时间的实验数据能确认这条循环的实际限制，也没有成熟的方法检测循环质量是否在退化。

**对齐税的可接受边界**

RLHF 和 DPO 都会对基础能力产生负面影响，DPO 的影响通常小于 RLHF。模型权重插值（model averaging：将对齐后模型权重与预训练模型权重插值）是目前已知最有效的缓解方法，但最优插值比例是经验性的。更根本的问题是：对齐目标（安全、有用性）和能力目标（推理、编码）之间的张力是内在的，还是可以通过更好的训练设计消除？到目前为止，这两个目标在边界案例上仍然存在内在张力，没有理论上的调和方案。

## 对能力输出的影响

训练流水线的不同阶段对能力输出有清晰的映射关系。

**SFT 阶段**是 `instruction following` 能力的直接来源（→ `capabilities/instruction-following.md`）：模型通过 SFT 学会识别任务格式、遵循约束、控制输出风格。SFT 数据质量决定了这个能力的质量上限，这是 data.md 中合成数据工程价值的直接体现。

**RLHF 和 DPO** 进一步精调 `instruction following` 的细粒度质量——使有用性和安全边界更符合人类偏好，同时塑造对话风格。这两个阶段对 `capabilities/instruction-following.md` 有持续贡献，也是 alignment.md 中对齐技术演进线和训练轨道最密集的交叉点。

**RL for Reasoning**（RLVR + GRPO + PRM）是 `reasoning` 能力质变的核心驱动力（→ `capabilities/reasoning.md`），尤其是数学推理（AIME、MATH-500）和代码生成（Codeforces）。这条路径使推理能力从"能做对简单题"跨越到"能稳定完成竞赛级数学题和高难度编程任务"。

**工具使用专项训练**（function calling SFT + 执行反馈 RL）是 `tool use` 能力的直接来源（→ `capabilities/tool-use.md`）：从"模型能勉强调用工具"到"原生支持、格式规范、有错误恢复"的质变发生在 2023 年的这个训练阶段。这是后训练流水线中第一次系统性引入执行反馈而不只是人类偏好。

**Agentic RL**（多步 agent 任务训练）目前对 `tool use` 有边际改进，是正在形成中的能力来源，预计会在未来的长上下文 agent 和 computer use 任务上产生更显著的影响，可能需要在 `capabilities/` 中独立追踪。

## 与其他轨道的交叉

- **alignment**：后训练技术和对齐技术高度重叠（RLHF 既是训练技术也是对齐技术）
- **data**：合成数据的质量直接影响训练效果
- **scaling**：预训练的 scaling 和后训练的 scaling 遵循不同规律

## 信息源

**预训练与早期微调**
- [Language Models are Few-Shot Learners (GPT-3)](https://arxiv.org/abs/2005.14165)，Brown et al., NeurIPS 2020
- [Evaluating Large Language Models Trained on Code (Codex)](https://arxiv.org/abs/2107.03374)，Chen et al., 2021

**RLHF 与指令遵循**
- [Training language models to follow instructions with human feedback (InstructGPT)](https://arxiv.org/abs/2203.02155)，Ouyang et al., NeurIPS 2022
- [Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.06520)，Bai et al., Anthropic 2022；另见[研究页面](https://www.anthropic.com/research/constitutional-ai-harmlessness-from-ai-feedback)

**偏好优化方法**
- [Direct Preference Optimization (DPO)](https://arxiv.org/abs/2305.18290)，Rafailov et al., NeurIPS 2023

**合成数据与指令微调**
- [Self-Instruct](https://arxiv.org/abs/2212.10560)，Wang et al., 2022
- [WizardLM / Evol-Instruct](https://arxiv.org/abs/2304.12244)，Xu et al., 2023
- [Orca](https://arxiv.org/abs/2306.02707)，Mukherjee et al., Microsoft 2023

**RL for Reasoning**
- [Let's Verify Step by Step (PRM800K)](https://arxiv.org/abs/2305.20050)，Lightman et al., OpenAI 2023
- [DeepSeekMath / GRPO](https://arxiv.org/abs/2402.03300)，Shao et al., DeepSeek 2024
- [DeepSeek-R1](https://arxiv.org/abs/2501.12948)，DeepSeek-AI 2025

**RLAIF 与 Agentic RL**
- [RLAIF vs RLHF](https://arxiv.org/abs/2309.00267)，Lee et al., ICML 2024
- [Agent-R1](https://arxiv.org/abs/2511.14460)，2025
- [Practical Guide to Multi-turn Agentic RL (NeurIPS 2025)](https://arxiv.org/abs/2510.01132)

**对齐税与 Reward Hacking**
- [Mitigating the Alignment Tax of RLHF (EMNLP 2024)](https://aclanthology.org/2024.emnlp-main.35/)
- [Reward Hacking in Reinforcement Learning](https://lilianweng.github.io/posts/2024-11-28-reward-hacking/)，Lilian Weng's Blog，2024

## 更新日志

- 2026-05-02：创建骨架
- 2026-05-02：填充阶段 1（预训练为王，GPT-3/Codex 时代）和阶段 2 开端（InstructGPT，RLHF 三步流水线）
- 2026-05-02：填充阶段 3（后训练技术多样化：DPO 及变体的扩散、工具使用专项训练、合成数据进入后训练流水线）
- 2026-05-02：填充阶段 4（RL for Reasoning：RLVR、PRM vs ORM、GRPO、DeepSeek-R1-Zero 的纯 RL 路线、推理蒸馏）
- 2026-05-02：补充阶段 4 的 2025 延伸（Agentic RL、后训练计算规模化、RLAIF 工业化、Deliberative Alignment）；填充当前技术格局、关键分歧与未决问题、对能力输出的影响、信息源
