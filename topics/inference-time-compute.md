# 推理时计算（Inference-Time Compute）

> 锚点：轨道 scaling（阶段 4）/ training（阶段 4） / 纪元 era4-reasoning

## 这个概念是什么

推理时计算（Inference-Time Compute，也称 Test-Time Compute）指在模型推理阶段投入额外计算以提升输出质量——与训练时计算（Training-Time Compute）相对。核心机制是让模型在输出最终答案前生成一段内部推理链，在答案空间中做类似搜索的探索。o1/o3 是这个范式的产品化代表，DeepSeek-R1 是开源验证。它是 2024 年 LLM 领域最重要的范式转变之一，开辟了独立于训练时 Scaling 的第二条能力增长路径。

## 内部结构

<!-- 
推理时扩展的技术路线：

1. 直接推理链（Chain-of-Thought at inference）：
   - 让模型在回答前生成"思考过程" tokens
   - 思考 tokens 本身不输出给用户（o1 方式）或折叠显示（Claude 3.7 方式）
   - 计算量随问题复杂度弹性增长

2. Best-of-N（BoN）采样：
   - 生成 N 个候选答案，用验证器选最好的
   - 最简单的推理时扩展形式，scaling 曲线与训练时相似
   - 缺点：N 个独立采样，不能互相利用信息

3. MCTS（蒙特卡洛树搜索）：
   - 在推理链生成过程中树状搜索
   - 模型作为策略网络，PRM 作为价值网络
   - 计算量更大但可以利用中间步骤的反馈

4. Beam Search 变体：
   - 维护多条并行推理路径，剪枝低质量路径

训练侧配套：
- RLVR：可验证奖励 RL，使模型学会"有效思考"
- PRM：过程奖励模型，对每步推理打分
- 冷启动 SFT：少量高质量思维链数据
- 见 tracks/training.md 阶段 4
-->

## 当前状态（截至 2026-05）

<!-- 
- o3 high compute 在 ARC-AGI 达到 ~88%（人类约 100%）
- DeepSeek-R1 匹配 o1 水平，开源可复现
- Claude 3.7、Gemini 2.5 Pro 均有推理模式
- 推理模式成为 2025 年的模型标配
- 未解问题：推理时 scaling 是否有硬上限？不同任务类型的 scaling 曲线差异巨大
-->

## 关键权衡

<!-- 
推理时扩展 vs 训练时扩展：
- 前者：推理成本高，训练成本低；后者反之
- 在有可验证答案的任务（数学、代码），推理时扩展效率更高
- 在需要大量世界知识的任务，推理时扩展无法凭空产生知识

思维预算（Thinking Budget）：
- 更多思维 tokens → 更好的答案，但成本线性增长
- 最优思维量依赖任务难度，难以提前预知
- 产品设计：low/medium/high 档位 vs 动态分配

隐藏 vs 可见思维链：
- o1：隐藏 → 对齐风险（模型"想"的和"说"的可能不同）
- Claude 3.7：可见 → 但可解释性价值有限（思维链可能是事后解释）
-->

## 信息源

[Scaling LLM Test-Time Compute Optimally (arXiv 2408.03314)](https://arxiv.org/abs/2408.03314)，[DeepSeek-R1 (arXiv 2501.12948)](https://arxiv.org/abs/2501.12948)

## 更新日志

- 2026-05-02：创建占位文件（定义 + 技术路线草图，待深度展开）
