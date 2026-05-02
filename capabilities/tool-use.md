# Tool Use

> 桥接：Agent 项目 dimensions/04-tool-use.md

## 当前水位（截至 2026-05）

### Benchmark 图谱

工具使用能力没有单一的标准化 benchmark，因为"工具使用"跨越多个维度，各维度的技术难度和挑战来源完全不同。理解这些维度的差异，比看单一数字更重要。

**BFCL（Berkeley Function Calling Leaderboard，[gorilla.cs.berkeley.edu](https://gorilla.cs.berkeley.edu/leaderboard.html)，Yan et al., 2024）** 测最基础的维度：给定函数定义，模型能否生成正确的函数调用（函数名对、参数类型和值对）。涵盖 Python、Java、JavaScript 及 REST API 等多种接口格式，包含单次调用、并行调用和序列调用场景。顶级模型已接近 90%，但它只测"怎么调"的精准度，不测"什么时候该调"的判断力。

**SWE-bench Verified（[swebench.com](https://www.swebench.com/)，Jimenez et al., 2024，[arXiv 2310.06770](https://arxiv.org/abs/2310.06770)）** 测完整的工程任务：给定 GitHub issue，模型需要浏览代码库、定位问题、修改代码、生成能通过测试套件的补丁。这是文件读取、代码编辑、测试运行等多个工具在真实软件工程任务中的集成测试，是当前工具使用能力最有区分度的 benchmark 之一。

**TAU-bench（Sierra AI，2024，[taubench.com](https://taubench.com/)）** 模拟 Agent 在真实用户对话中使用领域 API 的多轮交互：模型需要在对话过程中决定调用哪些 API、何时调用、如何将结果反馈用户，并追踪多步骤请求的状态。用 pass^k 指标（k 次试验中至少一次成功的比例）衡量可靠性。这直接对应 Agent 实际使用场景，而非受控测试环境。

**WebArena / OSWorld** 是更高维度的测试——GUI 控制：模型通过截图感知界面状态，输出点击/键盘输入完成任务。WebArena 测 Web 界面，OSWorld 测完整操作系统环境，均对应"Computer Use"类能力。

### 代表性数据（截至 2025 年上半年）

| 维度 | 基准 | 模型 | 分数 | 备注 |
|------|------|------|------|------|
| 函数调用准确性 | BFCL | Claude 3.5 Sonnet | ~90% | 顶级模型均在 85-90%+ |
| 函数调用准确性 | BFCL | GPT-4o | ~84% | — |
| 代码工具 / 修 bug | SWE-bench Verified | Claude 3.5 Sonnet（2024-10 升级版） | 49.0% | 人类约 100% |
| 代码工具 / 修 bug | SWE-bench Verified | o3（2025-04） | 71.7% | — |
| 多轮 Agent 工具 | TAU-bench（零售域） | Claude 3.5 Sonnet（2024-10 升级版） | 69.2% | GPT-4o 约 35% |
| 多轮 Agent 工具 | TAU-bench（航空域） | Claude 3.5 Sonnet（2024-10 升级版） | 46.0% | — |
| Web 界面操控 | WebArena | 最佳 AI（2025 年初） | ~60% | 人类约 89% |

来源：[Anthropic SWE-bench 研究（2024-10）](https://www.anthropic.com/research/swe-bench-sonnet)、[OpenAI o3/o4-mini 发布（2025-04-16）](https://openai.com/index/introducing-o3-and-o4-mini/)、[Anthropic Computer Use 发布（2024-10）](https://www.anthropic.com/news/3-5-models-and-computer-use)、[BFCL Leaderboard](https://gorilla.cs.berkeley.edu/leaderboard.html)。

### 可靠性随复杂度急剧下降

BFCL（~90%）与 SWE-bench（~50-72%）、TAU-bench 航空域（~46%）之间的落差揭示了 tool use 的核心挑战：**单次精确调用和完成多步骤任务之间存在巨大的可靠性鸿沟**。BFCL 只要求模型在明确指示下生成正确的调用格式；SWE-bench 和 TAU-bench 要求模型自主判断调用时机、处理工具返回的复杂结果、在调用失败时恢复、并在多轮对话中保持正确的状态追踪——每一步都有失败概率，多步骤叠加后整体成功率显著下降。这个可靠性递减是当前 Agent 工程中最关键的能力瓶颈，"模型会调工具"和"Agent 能可靠完成任务"之间有巨大鸿沟。

## 技术归因

工具使用能力经历了三个层次的技术演进，每层依赖不同的训练机制：

**第一层：执行反馈进入训练循环**（→ [训练范式 - 阶段 3](../tracks/training.md)）。这是从"prompt 技巧"到"原生能力"的关键跨越。在 function calling 之前，工具调用依赖模型输出特定格式的文本，外部代码解析后执行，结果再拼回 prompt——整个过程对模型是透明的，模型无法从执行结果中学习。OpenAI function calling（2023-06）引入了**执行反馈**：以工具调用的实际执行结果（参数正确则调用成功，否则报错）作为 RL 训练的奖励信号，是后训练流水线中第一次系统性引入可验证执行奖励，与 RLVR 的思路一脉相承。这使工具调用从"有时能解析出来"变成"结构化且可验证的行为"。

**第二层：工具调用训练数据的覆盖范围**（→ [数据工程](../tracks/data.md)）。训练高质量工具调用能力需要 SFT 数据覆盖三类场景：（1）"该调 vs. 不该调"的判断——避免模型对所有问题都尝试调用工具；（2）"调哪个"的多工具路由——在工具集较大时正确选择；（3）"调完怎么用"的结果消化——将工具返回结果整合进后续回答。三类数据缺一不可，也是早期开源模型工具调用能力弱的根本原因之一。

**第三层：推理与工具的融合**（→ [训练范式 - 阶段 4](../tracks/training.md)，[Scaling - 阶段 4](../tracks/scaling.md)）。o4-mini（2025-04）标志着工具调用从"推理完成后执行"变为"推理过程中执行"——模型可以在 CoT 中途调用工具、消化返回结果、继续推理，工具不再是思考结束后的"执行者"，而是思考过程中的"信息来源"。这使复杂任务中的工具使用策略可以动态调整，而不是在推理开始前就固定下来。o3 在 SWE-bench 上比 Claude 3.5 Sonnet 高出 22 个百分点（71.7% vs 49%），部分原因正是推理能力对多步骤工具编排的增益。

## 演进轨迹

**2021-2022：Prompt 技巧阶段**

WebGPT（OpenAI，2021，[arXiv 2112.09332](https://arxiv.org/abs/2112.09332)）是最早系统性探索工具增强 LLM 的工作：训练模型在回答前调用搜索引擎，证明了工具增强的可行性。ReAct（Yao et al., 2022，[arXiv 2210.03629](https://arxiv.org/abs/2210.03629)）提出 Reasoning + Acting 的交替框架——模型在自然语言推理步骤和工具调用动作之间交替，从工具返回的观察中更新推理链，成为 Agent 早期架构的基础范式。Toolformer（Schick et al., Meta，2023-02，[arXiv 2302.04761](https://arxiv.org/abs/2302.04761)）首次将"何时插入工具调用"变成训练目标而非 prompt 设计问题，通过自监督学习训练模型自主决定调用时机。

这个阶段的共同特征：工具调用依赖外部解析，模型不直接从执行反馈中学习，可靠性高度依赖 prompt 格式和示例质量。

**2023：原生函数调用——从技巧到训练目标**

ChatGPT Plugins（2023-03-23）是第一个产品化的工具集成，但以插件市场形式提供，体验笨重。

真正的范式转变是 OpenAI function calling API（[2023-06-13](https://openai.com/index/function-calling-and-other-api-updates/)）：开发者以 JSON schema 格式定义函数，模型直接输出结构化 JSON 调用对象，配套了专门的 SFT 数据和执行反馈 RL 训练。这是工具使用能力从"有时能用"到"可靠且结构化"的分水岭，也开启了"函数调用准确率"作为模型评测维度的时代。

Gorilla（Patil et al., UC Berkeley，2023，[arXiv 2305.15334](https://arxiv.org/abs/2305.15334)）同期证明，在 API 调用上专门微调可以大幅超过 GPT-4 的零样本性能，奠定了后来 BFCL leaderboard 的学术基础。

**2024：工具使用的全面工程化**

Claude 3 系列（2024-03）以工具使用作为核心能力之一与 OpenAI 形成双极格局，两家的工具调用 API 格式逐渐成为开发者生态的事实标准。

最具突破性的是 **Computer Use**（Anthropic，Claude 3.5 Sonnet 升级版，[2024-10](https://www.anthropic.com/news/3-5-models-and-computer-use)）：将工具使用扩展到 GUI 操控——模型通过截图感知界面状态，通过点击、键盘输入等动作完成任务。这是工具范畴从"API 调用"扩展到"任意软件界面"的重要突破，同时也是 OSWorld、WebArena 等 Agent 基准测试进入主流讨论的直接驱动力。

同年，SWE-bench Verified 确立为工程工具使用能力的核心 benchmark。Claude 3.5 Sonnet 升级版达到 49%，较早期 GPT-4 级别的约 2-3% 提升了约 20 倍，证明工具使用能力正在从"能用"向"可用于严肃工程任务"演进。

**2025：工具调用融入推理链**

o4-mini（OpenAI，[2025-04-16](https://openai.com/index/introducing-o3-and-o4-mini/)）首次将工具调用原生嵌入推理链内部：在生成 CoT 的过程中直接触发网页搜索、代码执行、图像生成等工具，消化返回结果后继续推理。"推理再执行"的两段式结构终结，模型可以在推理中途根据工具返回的信息动态调整后续步骤——这是此前需要外部 orchestration 框架才能实现的能力，现在变成了模型的原生行为。这个变化对 Agent 架构的影响是结构性的：过去 Agent 框架需要显式管理"思考 → 调用 → 思考"的循环，现在这个循环在模型内部完成。

## 更新日志

- 2026-05-02：创建骨架
- 2026-05-02：填充完整内容（四个 benchmark 维度说明 + 数据表 + 可靠性递减分析 + 技术归因三层机制 + 演进轨迹 2021-2025）
