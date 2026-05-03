# Scaling Laws

> 锚点：`tracks/scaling.md` 阶段 1-3 / `eras/era1-scaling.md`

## 这个概念是什么

Scaling Laws 是对"模型性能如何随计算资源增长"这个问题的定量回答——不是一个模糊的趋势观察，而是具有可预测幂律指数的数学关系。它们从根本上改变了 AI 研究的方法论：不再需要"试一下看看效果"，而是可以通过小规模实验预测大规模行为。Kaplan (2020) 建立框架，Chinchilla (2022) 修正了最优配比，这两篇论文共同定义了整个"scaling 纪元"的投入逻辑。

## 内部结构

### Kaplan 的幂律框架

> 待推导：
> - 三条独立的幂律：L(N)、L(D)、L(C)（Loss 分别关于参数量、数据量、计算量的函数）
> - 幂律指数的具体数值（α_N ≈ 0.076, α_D ≈ 0.095 等）
> - 核心实践结论：固定计算预算下，约 73% 资源用于扩大 N，27% 用于 D
> - 实验方法：如何在 7 个数量级上做受控实验

### Kaplan 为什么是错的

> 待推导：
> - 错误 1：只计非 embedding 参数（在小模型上 embedding 占比大，偏倚了 N 的效果估计）
> - 错误 2：在相对小的规模上实验然后外推（外推的方向有系统性偏移）
> - 结果：最优 N/D 比向参数侧偏移了约 11-12 倍
> - 认知论教训：小规模实验外推的系统性风险

### Chinchilla 的修正

> 待推导：
> - 三种独立的实验方法论（各自的假设和局限）：
>   1. 拟合法：在固定数据量下变化模型大小
>   2. IsoFLOP 法：在固定 FLOPs 下变化 N/D 配比
>   3. 参数法：直接拟合 L(N,D) 的参数化形式
> - 三种方法收敛到同一结论：N 和 D 应等比扩展，每参数约 20 tokens
> - 关键验证：Chinchilla 70B 用与 Gopher 280B 相同的计算预算达到更好结果

### 过训练的经济逻辑

> 待推导：
> - Chinchilla 优化的目标函数：固定计算预算，最小化训练损失
> - 但实际目标应该是：最小化总成本（训练成本 + 推理成本 × 请求量）
> - 当推理请求量 >> 训练一次的成本时，最优解漂移向更小模型 + 更长训练
> - Sardana & Frankle (2024) 的理论化：总成本 = C_train + C_infer × D_demand
> - Llama 3 8B 的实证：15T tokens 训练，过训练 ~90x，性能超 Llama 2 70B

### Inference-time Scaling

> 待推导：
> - 这是第二条独立的 scaling 曲线（与训练时 scaling 平行）
> - Best-of-N：采样 N 个 → 质量对 log(N) 呈线性关系（为什么是对数线性？）
> - MCTS 在语言推理中的应用：LLM 作为策略网络，PRM 作为价值网络
> - Snell et al. (2024) 的核心结论：最优分配策略取决于问题难度
> - 训练时 scaling 和推理时 scaling 的互补性（知识 vs 搜索）

### MoE 的 Scaling Laws

> 待推导：
> - 传统 scaling law 的隐含假设：总参数 = 活跃参数
> - MoE 打破了这个假设：需要同时考虑 5 个因子
> - Ludziejewski et al. (ICLR 2025) 的框架：激活参数、总参数、专家数、共享比例、数据量
> - 反直觉发现：合理配置下 MoE 比 dense 更内存高效

## 当前状态（截至 2026-05）

> 待填入：三轴并行（预训练 scaling + MoE scaling + inference-time scaling），各自的活跃度和瓶颈。"scaling wall" 争论的各方立场。

## 关键权衡

> 待填入：
> - 训练一次 vs 部署百万次：如何决定在三轴之间分配计算预算
> - 预测精度 vs 适用范围：小规模 scaling law 实验的外推可靠性
> - 能力 vs 可验证性：inference-time scaling 只在可验证任务上有效

## 信息源

- [Scaling Laws for Neural Language Models (Kaplan et al., 2020)](https://arxiv.org/abs/2001.08361)
- [Training Compute-Optimal Large Language Models (Chinchilla, Hoffmann et al., 2022)](https://arxiv.org/abs/2203.15556)
- [Beyond Chinchilla-Optimal (Sardana & Frankle, 2024)](https://arxiv.org/abs/2401.00448)
- [Scaling LLM Test-Time Compute Optimally (Snell et al., 2024)](https://arxiv.org/abs/2408.03314)
- [Joint MoE Scaling Laws (Ludziejewski et al., ICLR 2025)](https://arxiv.org/abs/2502.05172)

## 更新日志

- 2026-05-03：创建骨架
