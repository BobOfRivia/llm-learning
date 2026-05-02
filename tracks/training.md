# 训练范式

## 一句话定义

这条轨道覆盖"如何把一个随机初始化的模型变成有用的 LLM"——从预训练到后训练的整条流水线，每个阶段的技术选择和演进。

## 演进脉络

### 阶段 1：预训练为王（2018-2021）

<!-- GPT → GPT-2 → GPT-3，核心假设是"预训练足够好就行"。
     自回归语言建模作为统一目标。
     预训练数据的规模和质量开始被关注。 -->

### 阶段 2：后训练成为关键（2022-2023）

<!-- InstructGPT 论文标志性地证明：后训练可以根本性地改变模型行为。
     SFT (Supervised Fine-Tuning) → RLHF (Reinforcement Learning from Human Feedback)
     这是从"语言模型"到"助手"的关键转折。
     Constitutional AI (Anthropic) 作为 RLHF 的替代/补充路线出现。 -->

### 阶段 3：后训练技术多样化（2023-2024）

<!-- DPO (Direct Preference Optimization) 简化了 RLHF 的流程。
     RLHF vs DPO vs KTO 的实践权衡。
     多轮对话、工具使用的专项训练。
     合成数据开始大规模用于后训练。 -->

### 阶段 4：推理训练（2024-至今）

<!-- RL for Reasoning：通过强化学习训练模型的推理能力。
     o1/o3 代表的新范式：训练模型产生内部思考链。
     这是训练轨道与推理能力输出的直接因果连接。
     DeepSeek-R1 的开源验证。 -->

## 当前技术格局（截至 2026-05）

<!-- 待填充 -->

## 关键分歧与未决问题

<!-- 
- RL for Reasoning 的上限在哪？
- 合成数据在训练中的比例还能提高多少？
- 预训练是否已经"够好了"，未来的增益主要来自后训练？
-->

## 对能力输出的影响

<!-- SFT/RLHF → instruction following 能力的基础
     RL for Reasoning → reasoning 能力的质变
     工具使用专项训练 → tool use 能力 -->

## 与其他轨道的交叉

- **alignment**：后训练技术和对齐技术高度重叠（RLHF 既是训练技术也是对齐技术）
- **data**：合成数据的质量直接影响训练效果
- **scaling**：预训练的 scaling 和后训练的 scaling 遵循不同规律

## 信息源

<!-- 待补充 -->

## 更新日志

- 2026-05-02：创建骨架
