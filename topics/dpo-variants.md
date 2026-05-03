# DPO 及其变体

> 锚点：`tracks/alignment.md` 阶段 2 / `tracks/training.md` 阶段 3

## 这个概念是什么

DPO（Direct Preference Optimization）将 RLHF 的三阶段复杂流程压缩为一个分类损失——直接在偏好数据上训练语言模型，无需显式训练 reward model 或运行 PPO。其后涌现的变体（KTO、IPO、ORPO、SimPO）各自修正了 DPO 的不同局限。这个概念族值得展开，因为理解各变体需要先理解 DPO 的推导逻辑和隐含假设，然后看每个变体打破了哪个假设。

## 内部结构

### DPO 的数学推导

> 待推导：
> - 起点：RLHF 的 KL 约束优化目标 → 最优策略有闭合形式解
>   - π*(y|x) ∝ π_ref(y|x) · exp(r(x,y) / β)
> - 关键步骤：从最优策略公式中反推 reward 的表达式
>   - r(x,y) = β · log(π*(y|x) / π_ref(y|x)) + β · log Z(x)
> - 代入 Bradley-Terry 偏好模型，Z(x) 项抵消
> - 最终损失函数：对数几率差的 sigmoid（一个二分类损失）
>   - L_DPO = -E[log σ(β · (log π(y_w|x)/π_ref(y_w|x) - log π(y_l|x)/π_ref(y_l|x)))]
> - 直觉：让 chosen 的对数概率比率上升，rejected 的下降

### DPO 的隐含假设与局限

> 待推导：
> - 假设 1（离线数据）：偏好数据来自 π_ref 的分布。如果偏好数据来自其他模型，分布不匹配
> - 假设 2（Bradley-Terry）：偏好可以用单一标量 reward 解释
> - 局限 1：DPO 是离线算法——只能从固定数据集学习，不能探索新行为
>   - PPO 是在线的——可以在训练中采样新输出并获得反馈
>   - 这对安全对齐的含义：分布外的 jailbreak 行为不在训练数据中
> - 局限 2：DPO 对偏好数据质量极其敏感（噪声标签的影响比 RLHF 更大）
> - 局限 3：当 rejected 样本太差时（明显错误），DPO 学到的信息有限

### KTO（Kahneman-Tversky Optimization）

> 待推导：
> - 解决的问题：DPO 需要**偏好对**（同一 prompt 的 chosen + rejected），实际场景中常常只有"好的输出"或"差的输出"（不成对）
> - 核心思想：基于 Kahneman-Tversky 的前景理论——人类对损失比收益更敏感
>   - 不需要偏好对，只需要"好样本"和"差样本"（可以来自不同 prompt）
> - 损失函数的设计：对好样本做 maximize，对差样本做 minimize，但权重不对称
> - 优势：数据要求更灵活，适合标注资源有限的场景
> - 局限：理论上信息量比成对偏好少

### IPO（Identity Policy Optimization）

> 待推导：
> - 解决的问题：DPO 在理论推导中有一个隐含假设容易导致过拟合
>   - 具体：DPO 的 loss 在完美拟合偏好数据时趋向无穷（没有正则化）
> - 核心思想：用恒等式（不经过 Bradley-Terry 中间步）直接最小化策略和最优策略的距离
> - 损失函数：对对数概率比率的差做平方损失（而非 sigmoid）
> - 效果：训练更稳定，在小数据集上过拟合风险更低
> - 局限：可能收敛更慢

### ORPO（Odds Ratio Preference Optimization）

> 待推导：
> - 解决的问题：DPO 需要先做 SFT 再做偏好学习（两个阶段）
> - 核心思想：把 SFT 目标和偏好学习合并为单一损失
>   - SFT 部分：标准语言建模损失（在 chosen 上）
>   - 偏好部分：用 odds ratio（而非 log-prob ratio）来区分 chosen 和 rejected
> - 优势：一阶段训练，节省时间和计算
> - 局限：两个目标的权重平衡是额外超参数

### SimPO

> 待推导：
> - 解决的问题：DPO 的 reference model 引入额外计算和复杂性
> - 核心思想：去掉 reference model，直接用生成概率的平均对数似然作为隐式奖励
>   - 奖励 = 平均 token 对数概率（长度归一化的）
> - 优势：不需要维护 reference model → 内存减半，实现更简单
> - 效果：在多个 benchmark 上与 DPO 相当或更好
> - 局限：丢失了 KL 约束的显式形式，可能在某些场景下不够稳定

### 各变体的适用场景

> 待推导：
> - 有高质量成对偏好数据 → DPO 或 SimPO
> - 只有单边反馈（只知道好/差，不成对）→ KTO
> - 小数据集 + 需要稳定性 → IPO
> - 想要一阶段训练 → ORPO
> - 工业实践中的共识：开源社区用 DPO/SimPO，头部实验室保留 PPO 做安全对齐

## 当前状态（截至 2026-05）

> 待填入：DPO 是开源社区事实标准（Llama 3, Mistral, Qwen 的对齐均用 DPO）。头部实验室（OpenAI, Anthropic）保留 PPO 做安全对齐。变体仍在涌现但没有明确的"下一代共识"。DPO 对齐税通常低于 RLHF 但缺乏理论解释。

## 关键权衡

> 待填入：
> - DPO 的简洁性 vs PPO 的在线探索能力（安全对齐的核心分歧）
> - 成对偏好数据的必要性 vs 单边反馈的灵活性
> - 一阶段 vs 多阶段：简洁 vs 可控
> - Reference model 的作用：锚定稳定性 vs 额外计算开销

## 信息源

- [Direct Preference Optimization (Rafailov et al., NeurIPS 2023)](https://arxiv.org/abs/2305.18290)
- [KTO: Model Alignment as Prospect Theoretic Optimization (Ethayarajh et al., 2024)](https://arxiv.org/abs/2402.01306)
- [A General Theoretical Paradigm to Understand Learning from Human Feedback (IPO, Azar et al., 2023)](https://arxiv.org/abs/2310.12036)
- [ORPO: Monolithic Preference Optimization without Reference Model (Hong et al., 2024)](https://arxiv.org/abs/2403.07691)
- [SimPO: Simple Preference Optimization with a Reference-Free Reward (Meng et al., 2024)](https://arxiv.org/abs/2405.14734)

## 更新日志

- 2026-05-03：创建骨架
