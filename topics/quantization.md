# 量化（Quantization）

> 锚点：`tracks/inference.md` 阶段 2 / `eras/tension-efficiency-openness.md`

## 这个概念是什么

量化是将模型权重（和/或激活值）从高精度浮点数（FP16/BF16，2字节）压缩到低精度整数（INT8/INT4，1/0.5字节）的技术。直接效果是内存占用减半到四分之一，推理速度因内存带宽释放而加快。量化是 LLM 从数据中心"走向"消费级硬件的核心使能技术。理解量化需要理解为什么 LLM 比 CNN 更难量化（异常值问题），以及 GPTQ/AWQ 如何各自解决这个问题。

## 内部结构

### 精度类型与基础概念

> 待推导：
> - FP32 → FP16 → BF16 → INT8 → INT4 各自的数值范围和精度
> - BF16 为什么在 LLM 训练中优于 FP16（指数位更多 → 动态范围更大 → 溢出风险更小）
> - 对称量化 vs 非对称量化（zero-point 的作用）
> - Per-tensor vs Per-channel vs Per-group 量化粒度

### LLM 量化的核心难点：异常值

> 待推导：
> - LLM 权重中存在少量"异常值"（magnitude 远大于均值的参数）
> - 这些异常值在量化时的问题：它们拉大了量化范围，使大多数正常值的量化精度下降
> - 为什么 CNN 没有这个问题（分布更均匀）
> - 异常值的分布特征：通常集中在特定 channel，而非随机分布

### PTQ vs QAT

> 待推导：
> - PTQ（训练后量化）：模型训好后直接量化，不需要重新训练
>   - 优点：简单快速，几分钟到几小时可完成
>   - 缺点：高压缩率时质量损失大
> - QAT（量化感知训练）：训练时模拟量化误差，让模型学会"适应"低精度
>   - 优点：质量损失更小
>   - 缺点：需要重新训练，成本高（对大 LLM 不实用）
> - 结论：对于超大 LLM，PTQ 是唯一实用选择，QAT 用于中小模型

### GPTQ

> 待推导：
> - 核心方法：用 Hessian 矩阵信息决定量化顺序——对 loss 影响最大的权重最后量化
> - OBQ（Optimal Brain Quantization）的简化：逐列量化，用 Hessian 的逆补偿误差
> - 实践：可以在几小时内把 175B 模型量化到 INT4，质量损失有限
> - 速度：量化后的模型推理更快（因为内存读取减少）

### AWQ（Activation-Aware Quantization）

> 待推导：
> - 核心发现：不是所有权重同等重要，约 1% 的"显著权重"（salient weights）对输出影响极大
> - 识别方法：通过观察激活值幅度确定哪些权重通道是"显著"的
> - 保护策略：对显著权重用更高精度或 per-channel scaling，其余用 INT4
> - 与 GPTQ 的对比：AWQ 不需要校准数据的梯度计算，更快更简单，质量更好

### GGUF 与边缘部署

> 待推导：
> - llama.cpp 的量化方案：多种混合精度方案（Q4_K_M, Q5_K_M 等）
> - 混合精度的含义：不同层/不同部分用不同精度
> - CPU 推理的特殊优化（SIMD 指令、ARM NEON 等）
> - 意义：使 7B-13B 模型在普通笔记本/手机上运行成为可能

### 激活量化

> 待推导：
> - 权重量化 vs 激活量化的区别：权重是静态的可以离线量化，激活是动态的必须在线量化
> - 激活量化更难：异常值在激活中更极端，且逐 token 变化
> - SmoothQuant（Xiao et al., 2023）：在权重和激活之间"转移"量化难度
> - INT8 激活量化目前是实用上限，INT4 激活质量损失明显

## 当前状态（截至 2026-05）

> 待填入：INT4 权重量化（AWQ/GPTQ）是 serving 标准配置。INT8 是保守选择（几乎无质量损失）。GGUF 使边缘部署成为现实。FP8（H100/B200 原生支持）正在成为训练和推理的新中间精度。INT2 在特定任务上实验可行，通用质量仍有损失。

## 关键权衡

> 待填入：
> - 压缩率 vs 质量：INT4 是当前"甜点"，INT2 有损，INT8 安全但节省有限
> - 量化速度 vs 质量：GPTQ 更慢更精确，AWQ 更快效果相当
> - 权重量化 vs 激活量化：权重量化简单有效，激活量化收益大但更难
> - 通用 vs 任务特化：量化后模型在不同任务上的质量损失不均匀

## 信息源

- [GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers](https://arxiv.org/abs/2210.17323)
- [AWQ: Activation-aware Weight Quantization](https://arxiv.org/abs/2306.00978)
- [SmoothQuant: Accurate and Efficient Post-Training Quantization for LLMs](https://arxiv.org/abs/2211.10438)
- [llama.cpp](https://github.com/ggerganov/llama.cpp)
- Tim Dettmers, [The Case for 4-bit Precision (blog)](https://timdettmers.com/2022/08/15/which-gpu-for-deep-learning/)

## 更新日志

- 2026-05-03：创建骨架
