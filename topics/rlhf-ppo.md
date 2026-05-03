# RLHF 与 PPO

> 锚点：`tracks/training.md` 阶段 2 / `tracks/alignment.md` 阶段 1

## 这个概念是什么

RLHF（Reinforcement Learning from Human Feedback）是将语言模型从"自动补全引擎"变为"有用助手"的核心训练范式。它用人类偏好作为奖励信号，通过强化学习让模型的输出对齐人类意图。PPO（Proximal Policy Optimization）是 RLHF 中最常用的 RL 算法，平衡了训练稳定性和更新效率。理解 RLHF-PPO 流水线是理解后续所有改进（DPO、GRPO、RLVR）的必要前提。

## 内部结构

### Bradley-Terry 偏好模型

> 待推导：
> - 偏好数据的格式：给定 prompt，人类从两个回答中选更好的一个 (chosen, rejected)
> - Bradley-Terry 模型：P(y_1 > y_2 | x) = σ(r(x, y_1) - r(x, y_2))，其中 r 是奖励函数
> - 这个模型假设偏好可以用单一标量分数解释
> - 训练 reward model 的损失函数：对数似然的负数

### 奖励模型的训练

> 待推导：
> - RM 的架构：通常是 LLM 去掉最后的 token 预测头，加一个 scalar head
> - 训练数据：大量 (prompt, chosen, rejected) 三元组
> - RM 的质量约束：RM 的误差会直接传导到 PPO 阶段（garbage in, garbage out）
> - RM 的泛化问题：在训练分布外的输入上，RM 的分数是不可靠的

### PPO 算法

> 待推导：
> - Actor-Critic 框架：Policy model (actor) 生成输出，Value model (critic) 估计状态价值
> - 目标函数：最大化 E[A(s,a) × min(ratio, clip(ratio, 1-ε, 1+ε))]
>   - ratio = π_new(a|s) / π_old(a|s)（重要性采样）
>   - A(s,a) = advantage（当前动作比平均好多少）
>   - clip 操作：防止单步更新太大导致策略崩溃
> - Advantage estimation：GAE (Generalized Advantage Estimation) 如何平衡偏差和方差
> - 为什么用 PPO 而不是 vanilla policy gradient（方差太大）或 TRPO（计算太贵）

### KL 散度约束

> 待推导：
> - 约束：D_KL(π_new || π_ref) ≤ δ，π_ref = SFT 后的初始模型
> - 作用 1：防止模型过度偏离预训练学到的语言能力（对齐税）
> - 作用 2：防止 reward hacking（模型不能跑到 RM 评分虚高但语言质量崩溃的区域）
> - 实现方式：在奖励中加入 -β × KL(π || π_ref) 惩罚项

### RLHF 的四模型架构

> 待推导：
> - Reference policy（π_ref）：冻结的 SFT 模型，用于计算 KL 散度
> - Reward model：打分器，不参与梯度更新
> - Value model：估计状态价值，参与 advantage 计算
> - Policy model（π）：被优化的目标模型
> - 为什么需要这么多模型（以及为什么 DPO 要消除这个复杂性）
> - 内存需求：4 个大模型同时在 GPU 上

### Reward Hacking

> 待推导：
> - 定义：模型学会"骗过" reward model 而不是真正变好
> - 具体例子：输出变得冗长/自信/迎合评估者偏好
> - Goodhart's Law："当度量成为目标，它就不再是好的度量"
> - 缓解方法：KL 约束、RM ensemble、迭代更新 RM

## 当前状态（截至 2026-05）

> 待填入：头部实验室（OpenAI、Anthropic、Google）仍用 PPO 变体做安全对齐；开源社区主要用 DPO。推理训练用 GRPO 替代 PPO（省去 value model）。RLHF 的地位从"唯一路线"变成"特定场景的最优选择"。

## 关键权衡

> 待填入：
> - PPO vs DPO：在线探索能力 vs 工程简洁性
> - KL 约束强度：约束太强 → 学不到新东西；约束太弱 → reward hacking
> - RM 质量 vs 覆盖度：高质量但窄分布的 RM vs 低质量但广覆盖的 RM

## 信息源

- [Training language models to follow instructions with human feedback (InstructGPT)](https://arxiv.org/abs/2203.02155)
- [Proximal Policy Optimization Algorithms (Schulman et al., 2017)](https://arxiv.org/abs/1707.06347)
- [Fine-Tuning Language Models from Human Preferences (Ziegler et al., 2019)](https://arxiv.org/abs/1909.08593)
- [Secrets of RLHF in Large Language Models (Zheng et al., 2023)](https://arxiv.org/abs/2307.04964)
- Lilian Weng, [RLHF Blog Post](https://lilianweng.github.io/posts/2023-03-15-rlhf/)

## 更新日志

- 2026-05-03：创建骨架
