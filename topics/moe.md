# Mixture of Experts (MoE)

> 锚点：`tracks/architecture.md` / `eras/era1-scaling.md`（起点）→ `eras/era3-capability-race.md`（成熟）

## 这个概念是什么

MoE 是一种条件计算架构：模型的 FFN 层（通常是 Transformer 中计算量最大的部分）被替换为 N 个并行的"专家网络"（experts），每个 token 在推理时只激活其中 K 个，由一个轻量的路由网络（router）决定路由目标。这种设计使**参数容量**和**每个 token 的推理 FLOPs** 解耦——你可以有一个 600B 参数的模型，但每个 token 实际计算量只对应一个 40B 密集模型的水平。

这个概念值得独立展开，因为它不只是一个架构技巧，而是 Chinchilla 之后整个领域寻找"更高效 scaling 路线"时的核心答案之一。理解 MoE 为什么在 2022 年后大规模复兴，需要把它放到"dense scaling 的边际成本在上升"这个背景下看。

## 内部结构

### 基本机制

在标准 Transformer 中，每个层的 FFN 子层是固定的密集矩阵乘法。MoE 把这个 FFN 层替换为 N 个并行 FFN（experts），并加入一个 router 网络：router 接受当前 token 的表示作为输入，为 N 个 experts 各输出一个得分，选得分最高的 K 个 experts 参与计算，最终输出是这 K 个 experts 输出的加权平均（权重来自 router 的 softmax 分数）。

注意力层不会被复制——所有 experts 共享同一套注意力参数。这意味着参数规模的增长来自 experts 对 FFN 参数的复制，注意力部分的计算量与普通模型相同。以 Mixtral 8x7B 为例：8 个 experts 每个包含约 7B 的 FFN 参数，总参数 46.7B，但每个 token 激活 top-2 experts，实际参与计算的参数约 12.9B——与一个 14B 密集模型的单次前向传播计算量相当。

### 三条路线的演进

**Switch Transformer**（Fedus et al., Google Brain, 2021）是第一个在超大规模上验证 MoE 可行性的工作。它的设计选择是 top-1 routing（每个 token 只激活一个 expert），最大程度简化了实现，但也引入了"token 丢弃"问题：如果某个 expert 在一个批次中分到了过多 token（超过其 capacity），多余的 token 会直接跳过，通过残差连接传递而不经过任何 expert 处理。Switch Transformer 用 capacity factor 参数（设置为 1.0-1.25）调节这个问题，以工程妥协换来了实现简洁性。总参数达到 1.571T，证明了稀疏激活路线的理论可行性，但训练稳定性在当时仍是悬而未决的问题。

**Mixtral 8x7B**（Mistral AI, 2024 年 1 月，arXiv 2401.04088）是第一个生产级的开源 MoE 模型，把 MoE 从研究实验带入实际可部署的范围。路由策略升级为 top-2（每个 token 激活 2 个 experts），在推理质量和效率之间取得了更好的平衡。在能力上，以 46.7B 总参数 / 12.9B 活跃参数的配置在大多数基准上超过了 Llama 2 70B——这是 MoE 在开源领域的第一次有力验证；在生态上，它使 MoE 进入了社区可以实验和修改的范围，直接推动了开源 MoE 生态的形成。

**DeepSeek 的细粒度 MoE**（DeepSeekMoE, arXiv 2401.06066, 2024 年 1 月；规模化至 V2, arXiv 2405.04434, 2024 年 5 月）提出了与 Mixtral 完全不同的路线。核心思想是**更细粒度**：不是 8 个大 experts，而是 160 个更小的 experts，加上 2 个每个 token 都激活的"共享 experts"。每个 token 激活 6 个路由 experts + 全部 2 个共享 experts（共 8 个），总参数 236B，活跃参数 21B。

细粒度分割的逻辑是：在固定激活参数预算下，使用更多更小的 experts 比少数大 experts 提供更丰富的组合可能性——从 160 个中选 6 个的组合数，远大于从 8 个中选 2 个，理论上允许更精细的知识分化。共享 experts 的设计思路是：部分知识是通用的（基础语言理解、常识），应该在所有 token 上激活；其余知识是专门化的，交给路由 experts 按需激活。这条路线在 DeepSeek V3 中被进一步扩展至 256 个路由 experts。

### 负载均衡：最难的工程问题

MoE 最核心的工程问题是**负载均衡**。如果 router 不加约束地学习，它倾向于把大多数 token 路由到少数"赢家"experts，其余 experts 几乎从不被激活——这称为 expert collapse。collapse 之后，模型等效退化为一个小得多的密集模型，浪费了 MoE 的全部参数容量。

主流解法是添加**辅助均衡损失**（auxiliary loss）：在主训练目标之外，加一个惩罚 expert 使用不均匀分布的 loss 项，强制 router 把 token 分散到所有 experts。这个方案有效，但代价是辅助 loss 会干扰主训练目标的优化，两个 loss 之间的权衡需要仔细调整，且在训练过程中这个干扰会积累。

DeepSeek V3 提出了 **auxiliary-loss-free 的负载均衡方案**：不在 loss 中加均衡约束，而是在 router 的 bias 项上动态调整——对使用率偏低的 experts 增加正 bias，对使用率过高的 experts 减少 bias，在不影响梯度计算的情况下引导均衡。这把均衡约束和训练目标的优化彻底分离，实验结果显示在相同均衡程度下，V3 的模型质量高于有辅助 loss 的方案。

## 当前状态（截至 2026-05）

MoE 在 2024-2025 年完成了从"大公司内部选择"到"开源社区主流架构"的转变。当前前沿开源模型几乎全是 MoE：DeepSeek V3（671B 总 / 37B 活跃），Llama 4 Maverick（400B 总 / 17B 活跃，128 个 experts），Mistral Large 3（675B 总 / 41B 活跃）。GPT-4 据可信度不等的渠道疑似使用了 MoE（约 1.76T 总参数，8 个 experts），但 OpenAI 从未官方确认；Gemini 系列同样疑似采用 MoE。

Expert 专门化的实证研究（LibMoE, arXiv 2411.00918 等）显示，MoE 中确实出现了部分专门化，但主要体现在**句法模式**层面，而非更高层的语义或话题专门化。语言特异性主要集中在模型的早层和晚层，中间层的 experts 更接近语言无关。"每个 expert 负责一个话题/领域"的直觉图像是过度简化的——实际专门化的粒度比这更微妙，也更难解释。

## 关键权衡

**参数容量 vs. 内存占用**是 MoE 最根本的 serving 代价。推理时需要把所有 experts 的参数都加载到 GPU 显存——即使每次推理只激活少数 experts，其余 experts 必须随时就绪等待路由。这意味着 671B 的 DeepSeek V3 serving 需要能装下 671B 参数的硬件，尽管每个 token 只用了 37B。这个特性使 MoE 适合拥有大显存的服务器集群，但对边缘部署非常不友好——MoE 不能像密集模型那样通过量化轻松缩小到消费级硬件。

**训练复杂度 vs. 推理效率**是另一个核心权衡。MoE 的训练比同激活参数量的密集模型更难：多机训练时 expert 之间的 all-to-all 通信开销显著（token 可能被路由到不同设备上的 experts），负载均衡需要额外机制，训练初期的 router 不稳定会影响整体收敛。换来的是推理侧的 FLOPs 优势。对于像 Meta 这样在数十亿用户产品中部署 LLM 的公司，推理成本是决定性的成本项，一次性的训练难度增加可以被长期的推理成本节省覆盖——这是 Meta 把 Llama 4 做成 MoE 的直接商业逻辑。

**为什么 Chinchilla 之后 MoE 复兴**是理解这个趋势的关键问题。Chinchilla 的核心结论是：参数量和训练 tokens 应等比扩展。这意味着继续 dense scaling 的每一步，参数和数据都要同比增加，FLOPs 近似翻倍——前沿模型的训练成本开始以更快的速度上升。MoE 提供了一条"不花比例 FLOPs 就增加参数容量"的路线：用更多 experts 增加知识存储，而每次推理的计算量不随 expert 数线性增长。这个解耦正是 Chinchilla 时代之后 MoE 成为前沿首选的根本原因，而不只是工程上的优化技巧。

## 信息源

- [Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity (arXiv 2101.03961)](https://arxiv.org/abs/2101.03961)
- [Mixtral of Experts (arXiv 2401.04088)](https://arxiv.org/abs/2401.04088)
- [DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models (arXiv 2401.06066)](https://arxiv.org/abs/2401.06066)
- [DeepSeek-V2 (arXiv 2405.04434)](https://arxiv.org/abs/2405.04434)
- [DeepSeek-V3 Technical Report (arXiv 2412.19437)](https://arxiv.org/abs/2412.19437)
- [LibMoE: A Library for Comprehensive Benchmarking Mixture of Experts in LLMs (arXiv 2411.00918)](https://arxiv.org/abs/2411.00918)

## 更新日志

- 2026-05-03：创建文件，填充完整内容（基本机制、三条路线演进、负载均衡、当前状态、关键权衡）
