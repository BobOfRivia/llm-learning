# Alignment 纪元（2022-2023）

## 主导叙事

"能力不等于可用。" GPT-3 很强但很难用——它会胡说、有毒、不听指令。这个时期的核心问题从"怎么让模型更强"转向"怎么让模型有用且安全"。InstructGPT 和 ChatGPT 是这个转变的标志性事件。

## 关键事件与里程碑

这个纪元的核心命题是：有用性和安全性不是规模的副产品，而是需要专门工程的独立目标。InstructGPT（2022 年 3 月，见 Era 1）奠定了理论基础，但真正的引爆点是 ChatGPT。

**2022-11-30：ChatGPT 发布**，以免费 research preview 形式上线，底座为 GPT-3.5（GPT-3 改进版）+ RLHF。5 天内达到 100 万用户（Netflix 用了 3.5 年达到同一里程碑），两个月突破 1 亿用户。ChatGPT 的意义不在技术突破，而在于证明了对话式 AI 作为大众产品的可行性——它是 RLHF 对齐工作的第一次大规模消费端验证。

**2022-12：Constitutional AI（Anthropic，arXiv 2212.08073）** 提出 RLAIF（Reinforcement Learning from AI Feedback）路线，用 AI 反馈替代部分人类标注。流程分两阶段：监督阶段让模型依据"宪法"（一组原则）自我批评并修改有害输出（SL-CAI）；RL 阶段用 AI 模型对候选回应排序生成偏好数据（RLAIF），再做 PPO 优化。这是对 RLHF 中人类反馈可扩展性问题的首次系统性回应。Claude 1（2023 年 3 月，限制邀请制）是第一个基于 CAI 方法训练的公开产品。

**2023-02-24：Llama 1（Meta，申请制开放权重）** 以 7B/13B/33B/65B 四个尺寸发布。13B 在多数 NLP benchmark 上超越 GPT-3（175B），65B 与 PaLM、Chinchilla 相当。同年 3 月权重被泄露并广泛传播，实际上变成了完全可获取状态。Llama 1 开启了"开放权重生态"：Stanford Alpaca（Llama 7B + 52K GPT-3.5 生成指令数据，成本 \<$100，2023-03-13）证明了低成本 instruction tuning 的可行性，随后 Vicuna、Koala 等相继涌现，形成了 Llama 生态的第一波爆发。

**2023-03-14：GPT-4 发布**（arXiv 2303.08774）。MMLU 86.4%（GPT-3.5 约 70%），模拟律师资格考试 top 10%（GPT-3.5 约 bottom 10%）。GPT-4 是大型多模态模型，但视觉输入功能（GPT-4V）直到 2023 年 9 月才向公众开放。GPT-4 的发布在技术层面标志着能力竞赛纪元的开始，但 Anthropic 的 CAI 路线和开放模型生态的爆发仍属于这个纪元的延续。

**2023-05-29：DPO（Direct Preference Optimization，arXiv 2305.18290，斯坦福）** 提出了 RLHF 的数学简化。核心洞见：RLHF 最优策略可以从偏好数据中以闭合形式推导，因此原来三阶段流程（SFT→RM→PPO）可以压缩为一个分类损失，直接在 (chosen, rejected) 偏好对上训练语言模型，无需显式 reward model，也无需 PPO 采样。DPO 在情感控制任务上优于 PPO，在对话和摘要任务上与 PPO 相当，训练稳定性和实现复杂度大幅降低。NeurIPS 2023 发表后迅速成为学术界和开源社区的首选对齐方法。

**2023-06：Phi-1（Microsoft Research，arXiv 2306.11644）** 用 1.3B 参数 + 7B tokens"教科书质量"数据（6B 筛选自网络 + 1B GPT-3.5 合成），在 HumanEval 上达到 pass@1 50.6%，超过了大量更大的模型。这是"数据质量 > 数据规模"命题最直接的实验证据，开辟了"高质量合成数据 + 小模型"的研究方向。

## 范式替代

**建立的范式**：
- 后训练（SFT + RLHF）成为模型开发的必要步骤
- "助手"范式取代"续写"范式
- 对齐不再是可选项

**开始被质疑的**：
- RLHF 的复杂性（DPO 提出更简单的替代）
- 纯人类反馈的可扩展性（Constitutional AI 探索 AI 反馈）

## 遗产

ChatGPT 的真正遗产不是技术上的，而是证明了 LLM 可以成为大众产品。"对话式 AI"作为产品形态在这个纪元被确立。后训练的必要性成为共识，此后没有任何严肃的 LLM 发布会跳过这一步。

## 更新日志

- 2026-05-02：创建骨架
- 2026-05-02：填充关键事件与里程碑（ChatGPT、CAI/RLAIF、Llama 1、GPT-4、Alpaca、DPO、Phi-1）
