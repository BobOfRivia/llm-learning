# LoRA 与参数高效微调（PEFT）

> 锚点：轨道 training（阶段 3）/ 开源生态

## 这个概念是什么

LoRA（Low-Rank Adaptation，Hu et al., Microsoft，2021，arXiv 2106.09685）是参数高效微调（Parameter-Efficient Fine-Tuning, PEFT）的代表性方法，通过在预训练权重矩阵旁并联低秩分解矩阵来模拟全量微调的效果，训练参数量减少 10,000 倍以上，显存占用大幅降低。它是开源生态中最重要的单一工程贡献之一，使得在消费级 GPU 上对大模型做指令微调成为可能。

## 内部结构

<!-- 
LoRA 核心机制：
- 对权重矩阵 W（d×k）的更新 ΔW 分解为 BA（B: d×r, A: r×k，r << min(d,k)）
- 推理时将 W + BA 合并，不增加推理延迟
- 训练时只更新 A、B，冻结原始权重
- 超参数：rank r（通常 4-64），alpha（scaling factor）

QLoRA（Dettmers et al., 2023，arXiv 2305.14314）：
- LoRA + 4-bit 量化基础模型（NF4 格式）
- 65B 模型可在单张 48GB GPU 上微调
- Double Quantization + Paged Optimizers 进一步降低峰值显存

PEFT 方法谱系：
- Adapter：在层间插入小型 FFN 模块（早期方法，推理有延迟）
- Prefix Tuning：在输入前添加可训练的 prefix tokens
- Prompt Tuning：只微调 input embedding 层
- LoRA：权重分解，最广泛采用
- DoRA（2024）：权重分解 + 幅度/方向分离

应用场景：
- 指令微调（最常见）
- 领域适配（医疗、法律、代码特化）
- 风格适配（写作风格、对话风格）
- 多任务：多个 LoRA 适配器，按需切换或合并
-->

## 当前状态（截至 2026-05）

<!-- 
- LoRA/QLoRA 是开源社区做模型微调的事实标准
- Hugging Face PEFT 库提供了完整的实现生态
- LoRA 合并（merge）成为社区的重要实践：把多个 LoRA 叠加
- 高级用法：LoRA 作为"软 prompt"的替代，比 prompt engineering 更精确
-->

## 关键权衡

<!-- 
LoRA vs 全量微调：
- LoRA：更便宜，防止灾难性遗忘，但表达能力有上限（rank 决定容量）
- 全量微调：更强的适配能力，但成本高且容易丢失预训练知识

rank 选择：
- rank 太低 → 表达能力不足，对复杂任务微调效果差
- rank 太高 → 接近全量微调，失去成本优势
- 实践经验：代码/数学任务需要更高 rank（32-64），风格任务低 rank（4-8）即可
-->

## 信息源

[LoRA (arXiv 2106.09685)](https://arxiv.org/abs/2106.09685)，[QLoRA (arXiv 2305.14314)](https://arxiv.org/abs/2305.14314)，[Hugging Face PEFT 文档](https://huggingface.co/docs/peft)

## 更新日志

- 2026-05-02：创建占位文件（定义 + 机制草图，待深度展开）
