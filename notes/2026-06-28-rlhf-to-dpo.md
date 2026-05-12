# RLHF → DPO

> deadline: 2026-06-28
> status: pending
> 覆盖 tasks: RLHF 三阶段流程的概念地图（不深入 PPO 推导）、DPO 的核心化简、DPO 在生产中的实际位置
> 桥接: tracks/training.md, tracks/alignment.md
> 剔除：PPO 完整四模型工程（GAE / clip 细节）— 已被 GRPO/DAPO + DPO 替代为生产路径，作为历史背景理解即可，不深入推导。详见 logs/removed-from-mainline.md
> 剔除：IPO / KTO / SimPO / ORPO 等变体的逐个深入 — 生产主流仍是 DPO + 偏好对，变体作为索引知道存在即可，不逐个学

**核心问题**：工业界为什么从在线 RL（RLHF + PPO）切到离线优化（DPO）？这个切换的真正驱动力是工程成本，还是数学性质？要能回答：

1. **RLHF 三阶段的概念地图**（SFT → RM → PPO）：每一步的输入/输出/目标是什么，为什么需要三步而不是一步。**不要求推导 PPO clip / GAE**，只要求理解"为什么这套流程在生产中昂贵"（4 个模型同时驻显存、采样链长、稳定性差）。
2. **DPO 的核心 insight**：把 reward model 隐式吸收到 policy 比值里，三阶段压缩为一个分类损失。能用一句话讲清"为什么这个化简成立"（Bradley-Terry + KL 约束的解析最优策略可逆推）。
3. **DPO 在生产的实际位置（2025 状态）**：DPO 没死也没赢 — 它被收编进了"模块化后训练栈"。SFT 做指令跟随、DPO 做通用偏好对齐、GRPO/DAPO 做可验证奖励的推理训练。三者并行而不是替代关系。
4. **什么场景下 DPO 反而不如 PPO**：离线数据假设何时不成立（需要持续探索的场景、reward landscape 复杂时）。

**判断主线**：不要把这一篇学成"PPO 推导大全"。重点是看清"对齐技术栈在 2022-2025 怎么从单一 RLHF 拆成模块化组合"的过程 — 这个拆解和下一篇 RLVR/GRPO 是同一个故事的不同侧面。

---


---
