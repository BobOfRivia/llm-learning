# Scaling Laws 修正史

> deadline: 2026-05-29
> status: pending
> 覆盖 tasks: Kaplan→Chinchilla 修正的关键判据、过训练时代的目标函数重写、MoE Scaling Laws 五因子框架
> 桥接: tracks/scaling.md
> 剔除：Kaplan 公式的逐步推导和误差源数学分析 — 改为概念理解（"为什么参数主导是错的"），不深入推导。详见 logs/removed-from-mainline.md

**核心问题**：Scaling Laws 不是物理定律，是"当前最优化目标"的拟合 — 当目标函数变了（从训练损失最小到总成本最小、再到推理质量最大），laws 就跟着改写。要能讲清三次目标函数切换：

1. **Kaplan（2020）→ Chinchilla（2022）**：目标函数没变（训练损失最小化），但 Kaplan 的实验设计有系统偏差（只算非 embedding 参数 + 小规模外推），导致最优 N/D 配比错估了一个数量级。Chinchilla 用三种独立方法（拟合法 / IsoFLOP / 参数法）收敛到 ~20 tokens/param。**这一段不要求推导，要求理解"为什么 Kaplan 实验设计会偏向参数侧"**。
2. **Chinchilla → 过训练时代（2023-）**：目标函数从"训练损失最小"重写为"训练 + 推理总成本最小"。Llama 2 / 3 的小模型严重 over-train（Llama 3 8B 训 15T tokens）就是这个新目标的产物。要能用一个具体场景（1 亿用户 × 10 次/天）解释为什么过训练在经济上是合理的。
3. **MoE Scaling Laws（2025）**：五因子框架（激活参数 / 总参数 / 专家数 / 共享比例 / 数据量），Dense 的 N-D-C 三元组直接套到 MoE 是错的。要能解释为什么 MoE 在合理配置下比同质量 Dense 更内存高效（这反直觉，因为总参数大）。

**判断主线**：Scaling Laws 三次修正的真正驱动力都是**目标函数本身在变** — 工业界的关切从"用满计算"→"用对计算"→"用对架构"→未来可能是"用对推理时间"。把握这条目标演化线，比记住具体公式更重要。

---


---
