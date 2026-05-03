# 推测解码（Speculative Decoding）

> 锚点：`tracks/inference.md` 阶段 3

## 这个概念是什么

推测解码是一种加速自回归生成的技术：用一个小的"草稿模型"快速串行生成多个候选 token，然后让大的目标模型一次并行验证所有候选。关键性质：**输出分布与直接用目标模型采样完全相同**（不是近似）——通过 rejection sampling 保证。这是少数能加速推理但不改变输出质量的技术之一。

## 内部结构

### 自回归生成的瓶颈

> 待推导：
> - 自回归的根本约束：每个 token 依赖前一个，无法并行
> - Decode 阶段的实际瓶颈：不是计算（每步只计算一个 token 的 attention），而是内存带宽（需要读取全部模型权重和 KV cache）
> - 结果：GPU 计算单元大部分时间空闲（arithmetic intensity 极低）
> - 推测解码的直觉：让大模型一次验证多个 token，把"一步一步读权重"变成"一次读权重算多个 token"

### 算法流程

> 待推导：
> - 步骤 1：草稿模型自回归生成 γ 个候选 token（t₁, t₂, ..., t_γ）
> - 步骤 2：将 (context + t₁ + ... + t_γ) 一起喂给目标模型做一次前向传播（并行）
>   - 目标模型同时得到每个位置的真实概率分布
> - 步骤 3：逐位验证
>   - 对每个位置 i，比较草稿模型分布 q(t_i) 和目标模型分布 p(t_i)
>   - 接受或拒绝（基于 rejection sampling）
>   - 一旦某个位置被拒绝，后续全部作废，在该位置用目标模型重新采样

### Rejection Sampling 的正确性证明

> 待推导：
> - 接受概率：min(1, p(x)/q(x))——当目标模型概率 ≥ 草稿模型概率时，一定接受
> - 拒绝时的修正采样：从 max(0, p(x) - q(x)) 归一化后的分布中重新采样
> - 为什么这保证最终分布 = p(x)（目标模型分布）：这是经典 rejection sampling 的性质
> - 关键：不管草稿模型多差，最终输出分布都是精确的目标模型分布

### 加速比分析

> 待推导：
> - 加速比取决于"接受率"α（草稿模型与目标模型一致的概率）
> - 期望接受 token 数 = α/(1-α)（几何分布）
> - 实际加速比 ≈ (1 + 期望接受数) / (1 + γ × cost_ratio)
>   - cost_ratio = 草稿模型单步成本 / 目标模型单步成本
> - 最优 γ（猜测长度）：取决于 α 和 cost_ratio
> - 实践经验：代码生成、结构化输出等重复模式多的任务，α 高 → 2-5x 加速

### 草稿模型的选择

> 待推导：
> - 同家族小模型（如 Llama 3 8B 给 Llama 3 70B 做草稿）
> - 专门训练的轻量草稿模型
> - 自推测解码（Self-Speculative Decoding）：用目标模型的浅层输出做"草稿"
> - Medusa：给目标模型加多个预测头，同时预测后续多个 token
> - 选择标准：与目标模型的分布接近度 × 自身推理速度

### 变体与改进

> 待推导：
> - Medusa（Cai et al., 2024）：不用独立的草稿模型，直接在目标模型上加多头
> - EAGLE（Li et al., 2024）：用目标模型的特征预测后续 token
> - Tree-based speculative decoding：不是线性猜测，而是分支猜测
> - 与推理模型的关系：长思维链场景下推测解码的价值更大

## 当前状态（截至 2026-05）

> 待填入：推测解码已集成到主流 serving 框架（vLLM、TensorRT-LLM）。在代码生成、长文本续写等场景下实际加速 2-3x。Medusa/EAGLE 等无需独立草稿模型的变体在工业界开始采用。

## 关键权衡

> 待填入：
> - 只在"草稿模型足够接近目标模型"时有意义，否则拒绝率太高
> - 对随机性高的采样（高 temperature）效果更差（分布分散，难以猜中）
> - 额外的内存开销：草稿模型也需要加载到 GPU
> - Batch 场景的复杂性：不同请求的接受长度不同，影响 batch 效率

## 信息源

- [Fast Inference from Transformers via Speculative Decoding (Leviathan et al., 2023)](https://arxiv.org/abs/2211.17192)
- [Accelerating Large Language Model Decoding with Speculative Sampling (Chen et al., DeepMind, 2023)](https://arxiv.org/abs/2302.01318)
- [Medusa: Simple LLM Inference Acceleration Framework (Cai et al., 2024)](https://arxiv.org/abs/2401.10774)
- [EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty (Li et al., 2024)](https://arxiv.org/abs/2401.15077)

## 更新日志

- 2026-05-03：创建骨架
