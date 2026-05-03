# Mamba / SSM（状态空间模型）

> 锚点：`tracks/architecture.md` 阶段 3-4 / `eras/era3-capability-race.md`

## 这个概念是什么

Mamba 是基于选择性状态空间模型（Selective SSM）的序列建模架构，核心优势是 O(n) 线性时间复杂度（vs Attention 的 O(n²)），且没有 KV Cache（用固定大小的"状态"替代了不断增长的 cache）。Mamba 的关键创新是让状态转移参数依赖输入（选择性），使模型能"选择性记忆"。它代表了"非 Attention"路线最成功的尝试，最终定位为 Transformer 的互补组件而非替代者。

## 内部结构

### 状态空间模型基础

> 待推导：
> - 连续时间 SSM：dx/dt = Ax + Bu, y = Cx + Du（线性系统的标准形式）
> - 离散化：将连续系统转为离散递推 x_k = Ā·x_{k-1} + B̄·u_k
> - 核心直觉：SSM 是一种结构化的线性循环——像 RNN 但有数学保证
> - S4（Structured State Spaces for Sequence Modeling, Gu et al., 2022）的创新：
>   - 用结构化矩阵（HiPPO 初始化）解决长程依赖
>   - 并行训练：虽然是循环但可以用卷积形式并行计算

### Mamba 的选择性创新

> 待推导：
> - S4 的局限：A、B、C 矩阵是固定的（不依赖输入）
>   - 意味着模型处理每个 token 的方式完全相同，不能"决定"什么重要
> - Mamba 的改变：B、C、Δ（离散化步长）变为输入的函数
>   - B(x_k)：决定"当前输入的多少进入状态"
>   - C(x_k)：决定"从状态中读出什么"
>   - Δ(x_k)：决定"更新步长大小"（控制遗忘速率）
> - 直觉：模型可以"选择性地记住"——看到重要信息时大步更新状态，不重要时小步
> - 代价：引入输入依赖后，不能再用纯卷积并行化

### 硬件优化

> 待推导：
> - 选择性 SSM 的计算不能用 FFT 加速（因为不再是 LTI 系统）
> - Mamba 的解决方案：并行扫描（parallel scan）算法
>   - 类似前缀和：用 O(log n) 步在 GPU 上并行计算递推
> - IO-aware 的 kernel 实现：类似 Flash Attention 的思路，把计算放到 SRAM 中
> - 结果：训练速度可以做到与 Transformer（Flash Attention）相当

### Mamba vs Attention 的能力边界

> 待推导：
> - Attention 的优势：精确检索任意位置的信息（"回头找第 3 段的那个名字"）
> - Mamba 的优势：O(n) 线性，超长序列时效率优势巨大；固定状态大小 → 没有 KV Cache
> - Mamba 的劣势：历史信息被"压缩"到固定大小的状态中，无法精确回溯
>   - "大海捞针"测试中 Mamba 显著弱于 Transformer
> - 结论：Attention = 精确检索，Mamba = 压缩摘要。两者互补。

### Mamba-2 / SSD 框架

> 待推导：
> - Dao & Gu (2024) 的核心定理：在特定结构约束下，SSM 和 Attention 在数学上等价
> - SSD（Structured State Space Duality）：选择性 SSM 可以写成矩阵乘法形式
> - 实践意义：Mamba-2 可以直接复用 Attention 的高效 GPU kernel（矩阵乘法优化）
> - 结果：Mamba-2 比 Mamba-1 快 2-8x（同等质量下）
> - 概念意义：SSM 和 Attention 不是对立的，而是同一族的不同特例

### 混合架构中的角色

> 待推导：
> - Jamba（AI21 Labs）：Transformer 层和 Mamba 层交替
> - Nemotron-H（NVIDIA）：92% Mamba-2 + 8% Attention → 质量持平、3x 推理加速
> - 设计原则：Attention 层负责精确检索（全局注意力），Mamba 层负责长程压缩（线性扫描）
> - 最优比例问题：Nemotron-H 的 8% 是实验得出，无理论预测方法
> - 为什么不是纯 Mamba：某些任务确实需要精确位置检索

## 当前状态（截至 2026-05）

> 待填入：纯 Mamba 在通用语言建模上未成为主流，但混合架构已进入生产验证阶段。SSM 在特化领域（基因组、音频、超长序列）仍有优势。Mamba-2/SSD 框架使 SSM 获得了与 Transformer 可比的 GPU 效率。

## 关键权衡

> 待填入：
> - 线性复杂度 vs 精确检索：O(n) 很美但以压缩历史为代价
> - 训练并行性：Attention 天然并行，Mamba 需要并行扫描（略复杂但可行）
> - 生态支持：Transformer 有完善的工具链，SSM 的工具链仍在建设中
> - 固定状态 vs 可增长 KV：固定状态 → 推理内存可预测，但信息容量有上限

## 信息源

- [Mamba: Linear-Time Sequence Modeling with Selective State Spaces (Gu & Dao, 2023)](https://arxiv.org/abs/2312.00752)
- [Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality (Mamba-2, Dao & Gu, 2024)](https://arxiv.org/abs/2405.21060)
- [Efficiently Modeling Long Sequences with Structured State Spaces (S4, Gu et al., 2022)](https://arxiv.org/abs/2111.00396)
- [Jamba: A Hybrid Transformer-Mamba Language Model (AI21 Labs, 2024)](https://arxiv.org/abs/2403.19887)
- [Nemotron-H (NVIDIA, 2024)](https://arxiv.org/abs/2504.03624)
- Albert Gu 的 [The Annotated S4](https://srush.github.io/annotated-s4/)

## 更新日志

- 2026-05-03：创建骨架
