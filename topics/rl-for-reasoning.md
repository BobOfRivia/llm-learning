# RL for Reasoning

> 锚点：`tracks/training.md` 阶段 4 / `eras/era4-reasoning.md`

## 这个概念是什么

RL for Reasoning 是用强化学习专门训练模型的推理能力——不是让模型产生"看起来像推理的文本"，而是训练它产生"有助于得到正确答案的推理过程"。这条路线的核心创新是 RLVR（可验证奖励的 RL），它用数学答案正确性和代码执行结果作为奖励信号，绕开了人类偏好标注的瓶颈。o1、o3、DeepSeek-R1 的推理能力质变都来自这条路线。

## 内部结构

### RLVR（可验证奖励的强化学习）

> 待推导：
> - 与 RLHF 的关键区别：奖励来自机器验证而非人类偏好
> - 可验证域的定义：数学（最终数值正确否）、代码（编译 + 测试通过率）、逻辑谜题
> - 奖励函数的具体形式：binary (0/1) vs 连续分数？
> - 为什么 RLVR 能 scale：不受人工标注速度和成本限制
> - 根本约束：只在"答案可验证"的领域有效，开放域无法用 RLVR

### GRPO（Group Relative Policy Optimization）

> 待推导：
> - 与 PPO 的关键简化：省去 value model
> - 机制：对同一问题采样 G 个答案，用组内相对分数作为 advantage
>   - advantage(a_i) = (r_i - mean(r_group)) / std(r_group)
> - 为什么不需要 value model：组内相对比较本身就是 advantage 的估计
> - 代价：需要更多采样（一组 G 个，通常 G=64-256），换取省去 value model 的内存
> - PPO 需要 4 个模型（ref + RM + value + policy），GRPO 只需 ref + RM + policy
> - DeepSeek 选择 GRPO 的原因：推理训练中每个问题都会生成多个回答做验证，GRPO 的设计与此天然契合

### PRM vs ORM

> 待推导：
> - ORM（结果奖励模型）：只看最终答案，一个标量分数
> - PRM（过程奖励模型）：对推理链的每一步打分
> - PRM 的优势：排除"运气正确的错误推理"（ORM 会强化这些）
>   - 具体例子：计算过程有两处错误但恰好正负相消得到正确答案
> - PRM 的代价：需要步骤级标注（每步是否正确），标注成本远高于 ORM
> - 自动化步骤验证：Math-Shepherd（自动评估每步）、符号系统验证
> - OpenAI PRM800K 数据集：步骤级标注的关键参考基准

### 冷启动策略

> 待推导：
> - R1-Zero 的问题：纯 RL 有效但推理格式不稳定（语言切换、格式混乱）
> - 冷启动：先用少量（几千条）高质量思维链做 SFT，让模型习得格式
> - 然后在这个基础上做 RL
> - 为什么冷启动加速收敛：RL 不需要同时学习"格式"和"推理策略"

### DeepSeek-R1-Zero 的纯 RL 涌现

> 待推导：
> - 实验设置：从基础模型（非 SFT 模型）直接用 GRPO + 可验证奖励训练
> - 涌现现象：模型自发产生了自我反思、回溯、重新检查等行为
> - "aha moment"：在训练过程中某个点，模型突然开始使用这些策略
> - 解释：自我反思是一个在验证奖励下有正 advantage 的策略——模型发现"检查一下"比"直接输出"更容易得到正确答案
> - 与 GPT-3 few-shot 涌现的对比：两者都是"没有显式训练但在某个阈值后出现"

### Agentic RL

> 待推导：
> - 从单步推理到多步 agent 任务的扩展
> - 单步 RL（数学/代码）：奖励信号清晰，轨迹短
> - 多步 RL（agent 任务）：奖励稀疏，中间步骤无法直接验证
> - Agent-R1 的方案：将工具调用成功率作为中间过程奖励
> - 后训练计算的上升：RL 后训练已超过预训练计算的 10%（结构性变化）

## 当前状态（截至 2026-05）

> 待填入：RLVR + GRPO 是推理模型训练的行业标准路线。o1/o3/o4-mini、R1、Claude 3.7 ET、Gemini 2.5 Flash Thinking 都走这条路。PRM 从研究工具开始走向训练组件。Agentic RL 是活跃前沿但尚未有明确的 scaling 结果。

## 关键权衡

> 待填入：
> - RLVR 的领域限制：只在可验证域有效 vs 开放域需要回到 RLHF
> - GRPO 的采样成本 vs PPO 的 value model 成本：哪个在什么场景下更优
> - PRM 的标注投入 vs ORM 的错误强化风险
> - 推理蒸馏 vs 直接 RL：质量 vs 成本的权衡

## 信息源

- [DeepSeek-R1: Incentivizing Reasoning Capability via RL](https://arxiv.org/abs/2501.12948)
- [DeepSeekMath: Pushing the Limits of Mathematical Reasoning (GRPO)](https://arxiv.org/abs/2402.03300)
- [Let's Verify Step by Step (PRM800K)](https://arxiv.org/abs/2305.20050)
- [Math-Shepherd: Verify and Reinforce LLMs Step-by-step](https://arxiv.org/abs/2312.08935)
- [Learning to Reason with LLMs (OpenAI o1 blog)](https://openai.com/index/learning-to-reason-with-llms/)

## 更新日志

- 2026-05-03：创建骨架
