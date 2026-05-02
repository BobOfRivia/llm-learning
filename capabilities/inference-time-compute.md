# Inference-Time Compute

> 桥接：Agent 项目 dimensions/07-inference-time-compute.md

## 当前水位（截至 2026-05）

推理时计算已经成为模型能力的显性设计参数，不再是隐藏在后台的技术细节。

最直接的能力实证来自 o3 在 ARC-AGI 上的梯度：GPT-4o 约 5%，o1 约 32%，o3 low compute 约 47%，o3 high compute 约 88%。这四个数据点来自不同的模型或同一模型的不同算力档位，清楚地展示了推理时计算的 scaling 曲线——在相同模型上持续投入更多计算，收益是系统的、可预测的。

在数学推理的具体 benchmark 上，DeepSeek-R1 在 AIME 2024 达到单次采样 79.8%，多数投票（64 次采样）86.7%；MATH-500 上 97.3%；Codeforces 评级约 2029（超过 96.3% 的竞赛选手）。Claude 3.7 Sonnet 的 extended thinking 模式在 GPQA Diamond 上达到 84.8%，是该 benchmark 上首批显著超过博士生水平（约 65%）的模型之一。

市场侧的信号同样清晰：OpenAI o3 的 low/medium/high 三个档位价格差约 20 倍，Claude 3.7 Sonnet 允许 API 用户设置最高 128K tokens 的思维预算。这意味着"每次推理消耗多少计算"第一次变成了产品设计层面的显式决策——开发者需要根据任务价值、时延容忍度和成本预算，在 API 层面主动选择思考深度。

有一条重要的边界：推理时 scaling 的收益高度依赖任务的**可验证性**。数学、代码、逻辑这类有明确验证标准的任务，推理时扩展的收益显著；开放性写作、风格化对话等缺乏过程评估信号的场景，收益有限。这不只是一个工程约束，而是这个能力维度的内在结构性特征。

## 技术归因

这个能力的实现依赖三条轨道的联合贡献，缺任何一条都无法成立：

**训练轨道阶段 4（→ [RL for Reasoning](../tracks/training.md)）是执行基础。** 推理时扩展之所以有效，是因为模型经过了 RLVR（可验证奖励的强化学习）训练，学会了"用更多 tokens 做更好推理"这个行为。没有 RL for Reasoning 训练，暴力给一个模型更多生成 tokens，不会自动产生质量更高的推理——模型不知道如何利用这些额外空间。DeepSeek-R1 的纯 RL 路线（GRPO + 可验证奖励）证明了这个能力可以从 RL 中自然涌现：模型在没有显式训练"如何思考"的情况下，自发习得了自我反思和错误修正行为。这是 inference-time compute 能力的真正基础，往往被 scaling 叙事遮蔽。

**Scaling 轨道阶段 4（→ [推理时扩展](../tracks/scaling.md)）提供了理论框架和 scaling 曲线的形状。** Snell et al.（arXiv 2408.03314）建立了推理时计算最优分配的理论分析：问题难度决定策略选择（简单问题用 Best-of-N 就够，复杂问题需要树搜索），小模型在推理时分配更多计算后可以超过大模型的单次前向传播。这解释了"为什么推理时计算的收益是可预测的"——它遵循和训练时 scaling 相似的对数线性关系，但是一条独立的曲线。

**推理优化轨道阶段 3（→ [推理时扩展与 serving 的根本张力](../tracks/inference.md)）揭示了 serving 侧的结构性挑战。** 推理模型的上线使传统 serving 优化的两个核心假设失效：每个请求的计算量不再固定（思维 tokens 从几百到数万不等），时延约束放宽但成本约束变严。KV Cache 压力倍增（思维链 tokens 同样占用 Cache），批调度策略需要重新设计。这条轨道的进展（KV Cache 分级存储、PD 分离、动态批次调整）是 inference-time compute 从"能跑"到"可以规模化商业部署"的工程基础。

## 演进轨迹

这个能力维度在不到两年内完成了从理论提案到标准产品特性的全过程：

2024 年 8 月，Snell et al. 发布"Scaling LLM Test-Time Compute Optimally"，建立推理时 scaling 的对数线性曲线理论，确认最优策略依赖问题难度，首次系统论证小模型 + 更多推理计算可以超越大模型单次推理的能力上限。这是理论基础。

2024 年 9 月，o1 发布，ARC-AGI 约 32%。这是 inference-time compute 的第一次商业化验证——用户第一次在产品层面感受到"让模型多思考一会儿"是有代价但有价值的。

2025 年 1 月，DeepSeek-R1 开源。它的价值是双重的：一是在开放模型上重现了推理时扩展的能力（AIME 86.7% majority vote），二是公开了 GRPO + RLVR 的训练细节，证明这条路线不是 OpenAI 独有的专利。同期发布的蒸馏版本（R1-Distill）证明推理能力可以从大模型传递给小模型，推理时扩展的基础能力不需要从头用 RL 训练。

2025 年 2 月，Claude 3.7 Sonnet 推出 extended thinking 模式，API 参数化思维预算（最高 128K tokens）。这是 inference-time compute 成为**开发者工具**的标志——不再只是产品层面的"慢思考"选项，而是 API 级别的精细控制。

2025 年 3-4 月，o3 和 o4-mini 发布，o3 high compute 在 ARC-AGI 达到 88%，low/medium/high 三档定价结构明确。Gemini 2.5 Pro 同期跟进，思维 token 策略类似。这一轮代表了范式的完全确立：所有头部实验室都有推理模型，价格梯度成为商业标准。

2026 年初至今，inference-time compute 是标准产品特性，用户付费结构显性化，围绕推理 serving 效率（KV Cache 管理、PD 分离、长思维链批调度）的工程竞争成为头部推理服务商的核心成本竞争维度。

## 更新日志

- 2026-05-02：创建骨架
- 2026-05-02：填充完整内容（当前水位、技术归因、演进轨迹）
