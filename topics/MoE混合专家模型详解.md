# 混合专家模型（Mixture of Experts）详解

> 原文：[Mixture of Experts Explained](https://huggingface.co/blog/moe) — Hugging Face Blog，发布于 2023 年 12 月 11 日
> 作者：Omar Sanseviero、Lewis Tunstall、Philipp Schmid、Sourab Mangrulkar、Younes Belkada、Pedro Cuenca

> 注：本博客已有第二版（2026 年 2 月），介绍了 `transformers` 库如何将 MoE 作为"一等公民"进行支持。链接：[Mixture of Experts (MoEs) in Transformers](https://huggingface.co/blog/moe-transformers)

随着 Mixtral 8x7B 的发布（[官方公告](https://mistral.ai/news/mixtral-of-experts/)、[模型卡片](https://huggingface.co/mistralai/Mixtral-8x7B-v0.1)），一类 Transformer 架构成为了开源 AI 社区最热门的话题：混合专家模型（Mixture of Experts），简称 MoE。在本篇博客中，我们将探讨 MoE 的基本构成、训练方式，以及在推理服务中需要权衡的取舍。

让我们开始吧！

## 目录

- [什么是混合专家模型？](#什么是混合专家模型moe)
- [MoE 简史](#moe-简史)
- [什么是稀疏性？](#什么是稀疏性)
- [MoE 的 token 负载均衡](#moe-的-token-负载均衡)
- [MoE 与 Transformer](#moe-与-transformer)
- [Switch Transformers](#switch-transformers)
- [使用 router Z-loss 稳定训练](#使用-router-z-loss-稳定训练)
- [专家学到了什么？](#专家学到了什么)
- [扩展专家数量对预训练有何影响？](#扩展专家数量对预训练有何影响)
- [微调 MoE](#微调-moe)
- [何时使用稀疏 MoE，何时使用稠密模型？](#何时使用稀疏-moe何时使用稠密模型)
- [让 MoE 飞起来](#让-moe-飞起来)
  * [专家并行](#专家并行)
  * [容量因子与通信成本](#容量因子与通信成本)
  * [服务部署技术](#服务部署技术)
  * [更多关于高效训练的内容](#更多关于高效训练的内容)
- [开源 MoE 模型](#开源-moe-模型)
- [值得关注的研究方向](#值得关注的研究方向)
- [参考资源](#参考资源)

## TL;DR（摘要）

MoE 模型：

- 相比稠密模型，**预训练速度快得多**
- 与参数量相同的模型相比，**推理速度更快**
- 由于所有专家都需要加载到内存，对 **显存（VRAM）要求很高**
- 在**微调方面面临诸多挑战**，但 [近期工作](https://arxiv.org/pdf/2305.14705.pdf) 表明 MoE 的**指令微调前景广阔**

让我们深入了解！

## 什么是混合专家模型（MoE）？

模型规模是提升模型质量最重要的因素之一。在固定的计算预算下，训练一个更大的模型走更少的步数，往往优于训练一个更小的模型走更多的步数。

MoE 让模型能够以远低于稠密模型的计算量进行预训练，这意味着你可以在相同计算预算下大幅扩展模型规模或数据集规模。具体来说，MoE 模型在预训练阶段能够更快地达到与其稠密对应模型相同的质量水平。

那么 MoE 到底是什么？在 Transformer 模型的语境下，MoE 由两个主要部分组成：

- **稀疏 MoE 层**，用于替代稠密的前馈网络（FFN）层。MoE 层包含若干个"专家"（例如 8 个），每个专家本身是一个神经网络。在实践中这些专家通常是 FFN，但它们也可以是更复杂的网络，甚至可以是 MoE 本身，从而形成层级化的 MoE！
- **门控网络（gate network）或路由器（router）**，用于决定将哪些 token 发送到哪个专家。例如下图中，token "More" 被发送到第二个专家，而 token "Parameters" 被发送到第一个网络。我们稍后会看到，一个 token 可以被发送到多个专家。如何将 token 路由到专家是设计 MoE 时最重要的决策之一——路由器由可学习的参数组成，并与网络的其他部分一起进行预训练。

![Switch Layer](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/moe/00_switch_transformer.png)

*来自 [Switch Transformers 论文](https://arxiv.org/abs/2101.03961) 的 MoE 层示意图*

总结来说，在 MoE 中，我们将 Transformer 模型的每个 FFN 层替换为一个 MoE 层，该层由一个门控网络和若干个专家组成。

虽然 MoE 相比稠密模型在预训练效率和推理速度上有诸多优势，但它也面临一些挑战：

- **训练**：MoE 在预训练阶段的计算效率显著更高，但在微调时往往难以泛化，容易导致过拟合。
- **推理**：尽管 MoE 模型可能拥有大量参数，但推理时只有一部分参数会被使用。因此相比同参数量的稠密模型，推理速度会快得多。然而所有参数都必须加载到内存中，因此显存需求很高。例如对于 Mixtral 8x7B 这样的 MoE 模型，我们需要足够的显存来容纳一个 47B 参数的稠密模型。为什么是 47B 参数而不是 8 × 7B = 56B？这是因为在 MoE 模型中，只有 FFN 层被视为独立的专家，模型的其他参数是共享的。同时，假设每个 token 只使用两个专家，推理速度（FLOPs）相当于使用一个 12B 模型（而不是 14B 模型），因为它计算的是 2 × 7B 的矩阵乘法，但部分层是共享的（稍后会详细说明）。

现在我们对 MoE 有了大致了解，让我们看看它的演进历程。

## MoE 简史

MoE 的起源可以追溯到 1991 年的论文 [Adaptive Mixture of Local Experts](https://www.cs.toronto.edu/~hinton/absps/jjnh91.pdf)。其思想类似于集成方法：一个系统由多个独立网络组成，每个网络处理训练样本的不同子集，并采用监督式训练流程。每个独立网络（即专家）专门处理输入空间的某个特定区域。如何选择专家呢？由一个门控网络决定每个专家的权重。训练过程中，专家网络和门控网络是同时训练的。

在 2010-2015 年间，两个不同的研究领域为后来的 MoE 发展奠定了基础：

- **专家作为组件**：在传统的 MoE 设置中，整个系统由一个门控网络和多个专家组成。MoE 作为完整模型已在 SVM、高斯过程等方法中被探索过。[Eigen、Ranzato 和 Ilya](https://arxiv.org/abs/1312.4314) 的工作探索了将 MoE 作为更深层网络的组件。这使得 MoE 可以作为多层网络中的一层，从而让模型同时实现规模庞大和计算高效。
- **条件计算**：传统网络对所有输入数据都会通过每一层进行处理。在这一时期，Yoshua Bengio 研究了基于输入 token 动态激活或停用网络组件的方法。

这些工作促进了 MoE 在 NLP 领域的探索。具体来说，[Shazeer 等人](https://arxiv.org/abs/1701.06538)（2017 年，作者中包括 Geoffrey Hinton 和 Jeff Dean，[Google 的 Chuck Norris](https://www.informatika.bg/jeffdean)）通过引入稀疏性，将这一思想扩展到了 137B 参数的 LSTM（当时事实上的 NLP 架构，由 Schmidhuber 创建），即使在极大规模下也能保持非常快的推理速度。这项工作聚焦于翻译任务，但面临诸多挑战，例如通信成本高和训练不稳定。

![MoE layer in LSTM](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/moe/01_moe_layer.png)

*来自 Outrageously Large Neural Network 论文的 MoE 层*

MoE 使训练万亿参数级别的模型成为可能，例如开源的 1.6T 参数 Switch Transformers 等。MoE 也在计算机视觉领域被探索过，但本博客将聚焦于 NLP 领域。

## 什么是稀疏性？

稀疏性运用了条件计算的思想。在稠密模型中，所有参数都会用于所有输入；而稀疏性允许我们只运行整个系统的某些部分。

让我们深入了解 Shazeer 在翻译领域对 MoE 的探索。条件计算的思想（网络的某些部分根据每个样本激活）允许我们在不增加计算量的情况下扩大模型规模，因此每个 MoE 层可以使用成千上万的专家。

这种设置带来了一些挑战。例如，大批量训练通常对性能更有利，但在 MoE 中，由于数据流经被激活的专家，批量大小实际上被减小了。例如，如果我们的批量输入包含 10 个 token，**可能有 5 个 token 进入某一个专家，另外 5 个 token 进入 5 个不同的专家，这导致批量大小不均衡且利用率低下**。下面 [让 MoE 飞起来](#让-moe-飞起来) 部分将讨论其他挑战和解决方案。

我们如何解决这些问题？一个可学习的门控网络（G）决定将输入的哪一部分发送给哪些专家（E）：

$$y = \sum_{i=1}^{n} G(x)_i E_i(x)$$

在这种设置中，所有专家都会对所有输入运行——这是一种加权乘法。但如果 G 为 0 会怎样？如果是这种情况，就无需计算对应的专家操作，从而节省了计算。典型的门控函数是什么？最传统的设置中，我们只使用一个带有 softmax 函数的简单网络。该网络会学习将输入发送给哪个专家。

$$G_\sigma(x) = \text{Softmax}(x \cdot W_g)$$

Shazeer 的工作还探索了其他门控机制，例如带噪声的 Top-k 门控（Noisy Top-k Gating）。这种门控方法引入了一些（可调节的）噪声，然后只保留前 k 个最大值。即：

1. 我们添加一些噪声

$$H(x)_i = (x \cdot W_{\text{g}})_i + \text{StandardNormal()} \cdot \text{Softplus}((x \cdot W_{\text{noise}})_i)$$

2. 我们只选择前 k 个

$$\text{KeepTopK}(v, k)_i = \begin{cases} v_i & \text{如果 } v_i \text{ 在 } v \text{ 的前 } k \text{ 个元素中} \\ -\infty & \text{否则} \end{cases}$$

3. 我们应用 softmax

$$G(x) = \text{Softmax}(\text{KeepTopK}(H(x), k))$$

这种稀疏性带来了一些有趣的性质。通过使用足够小的 k 值（例如 1 或 2），我们可以比激活许多专家时更快地进行训练和推理。为什么不直接选择 top-1 专家呢？最初的猜想是，需要路由到多个专家才能让门控学会如何路由到不同的专家，因此至少要选 2 个专家。[Switch Transformers](#switch-transformers) 章节会重新审视这一决策。

为什么要添加噪声？是为了实现负载均衡！

## MoE 的 token 负载均衡

如前所述，如果所有 token 都被发送到少数几个热门专家，会使训练效率低下。在正常的 MoE 训练中，门控网络会收敛到主要激活相同的少数几个专家。这是一种自我强化：受青睐的专家训练得更快，因此更容易被选中。为了缓解这一问题，添加了一个**辅助损失（auxiliary loss）**来鼓励所有专家具有同等重要性。这一损失确保所有专家收到大致相等数量的训练样本。后续章节也会探讨"专家容量"的概念，它引入了每个专家能处理 token 数量的阈值。在 `transformers` 库中，辅助损失通过 `aux_loss` 参数暴露出来。

## MoE 与 Transformer

Transformer 是一个非常明显的例子，证明了扩大参数规模能够提升性能，因此 Google 通过 [GShard](https://arxiv.org/abs/2006.16668) 探索了将 Transformer 扩展到 6000 亿参数以上并不令人意外。

GShard 在编码器和解码器中都将每隔一个 FFN 层替换为使用 top-2 门控的 MoE 层。下图展示了编码器部分的样子。这种设置对大规模计算非常有利：当我们扩展到多个设备时，MoE 层在不同设备间共享，而其他所有层都被复制。这在 [让 MoE 飞起来](#让-moe-飞起来) 章节会进一步讨论。

![MoE Transformer Encoder](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/moe/02_moe_block.png)

*来自 GShard 论文的 MoE Transformer 编码器*

为了在大规模下保持负载均衡和效率，GShard 作者除了引入类似前文讨论的辅助损失外，还引入了几项变化：

- **随机路由**：在 top-2 设置中，我们始终选择 top-1 专家，但第二个专家是按其权重成比例的概率选择的。
- **专家容量**：我们可以设定一个阈值，限制每个专家能处理的 token 数量。如果两个专家都已达到容量上限，则该 token 被视为溢出，并通过残差连接发送到下一层（在其他项目中也可能被完全丢弃）。这一概念将成为 MoE 中最重要的概念之一。为什么需要专家容量？因为所有张量形状在编译时就是静态确定的，但我们无法预先知道有多少 token 会进入每个专家，因此需要固定容量因子。

GShard 论文在表达适合 MoE 的并行计算模式方面做出了贡献，但这超出了本博客的讨论范围。

**注意：**推理时只有部分专家会被触发。同时存在一些共享计算，例如自注意力机制，它对所有 token 都会应用。这就是为什么当我们说一个 8 专家、47B 的模型时，可以用 12B 稠密模型的计算量运行它。如果使用 top-2，会用到 14B 参数。但由于注意力操作（以及其他部分）是共享的，实际使用的参数量是 12B。

## Switch Transformers

虽然 MoE 展现出了巨大潜力，但它们在训练和微调时面临不稳定性问题。[Switch Transformers](https://arxiv.org/abs/2101.03961) 是一项深入探讨这些主题的精彩工作。作者甚至在 [Hugging Face 上发布了一个 1.6 万亿参数、2048 个专家的 MoE 模型](https://huggingface.co/google/switch-c-2048)，可以使用 transformers 库运行。Switch Transformers 相比 T5-XXL 实现了 4 倍的预训练加速。

![Switch Transformer Layer](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/moe/03_switch_layer.png)

*Switch Transformer 论文中的 Switch Transformer 层*

与 GShard 一样，作者将 FFN 层替换为 MoE 层。Switch Transformers 论文提出了一种 Switch Transformer 层，它接收两个输入（两个不同的 token），并拥有四个专家。

与最初"至少使用两个专家"的想法相反，Switch Transformers 采用了简化的单专家策略。这种方法带来的影响是：

- 路由器计算量减少
- 每个专家的批量大小至少可以减半
- 通信成本降低
- 质量得以保持

Switch Transformers 也探索了专家容量的概念。

$$\text{Expert Capacity} = \left(\frac{\text{tokens per batch}}{\text{number of experts}}\right) \times \text{capacity factor}$$

上述容量计算公式将批量中的 token 数量均匀分配给各个专家。如果我们使用大于 1 的容量因子，就为 token 分配不完全均衡时提供了缓冲。增加容量会导致设备间通信成本更高，因此需要权衡。具体来说，Switch Transformers 在低容量因子（1-1.25）下表现良好。

Switch Transformer 作者还重新审视并简化了前文提到的负载均衡损失。对于每个 Switch 层，辅助损失在训练期间会被加入到模型的总损失中。这一损失鼓励均匀路由，并可通过超参数加权调节。

作者还尝试了选择性精度，例如用 `bfloat16` 训练专家，而其余计算使用全精度。较低精度可减少处理器之间的通信成本、计算成本以及存储张量的内存。最初的实验中，专家和门控网络都用 `bfloat16` 训练，结果训练更加不稳定。这特别归因于路由器计算：由于路由器中有指数函数，更高精度尤其重要。为了缓解不稳定性，路由计算也采用了全精度。

![Table shows that selective precision does not degrade quality.](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/moe/04_switch_table.png)

*使用选择性精度不会降低质量，并使模型运行更快*

这个 [notebook](https://colab.research.google.com/drive/1aGGVHZmtKmcNBbAwa9hbu58DDpIuB5O4?usp=sharing) 展示了如何微调 Switch Transformers 进行摘要任务，但我们建议先阅读 [微调章节](#微调-moe)。

Switch Transformers 使用了编码器-解码器结构，是 T5 的 MoE 对应版本。[GLaM](https://arxiv.org/abs/2112.06905) 论文则探索了进一步扩大这些模型规模的方法，训练出了一个匹配 GPT-3 质量但能耗仅为其 1/3 的模型（是的，由于训练 MoE 所需的计算量更少，可以将碳足迹降低一个数量级）。作者聚焦于仅解码器模型，并采用少样本和单样本评估而非微调。他们使用了 top-2 路由和大得多的容量因子。此外，他们还探索了将容量因子作为一个可在训练和评估时调整的指标，取决于愿意使用多少计算量。

## 使用 router Z-loss 稳定训练

前文讨论的均衡损失可能导致不稳定问题。我们可以使用许多方法以质量为代价来稳定稀疏模型。例如，引入 dropout 可以提升稳定性但会损失模型质量。另一方面，添加更多乘性组件可以提升质量但会降低稳定性。

[ST-MoE](https://arxiv.org/abs/2202.08906) 中提出的 Router z-loss 通过惩罚进入门控网络的较大 logits，显著提高了训练稳定性而不损害质量。由于这一损失鼓励数值的绝对量级较小，舍入误差会被减小，这对于门控中的指数函数等操作可能影响很大。我们建议查阅论文了解细节。

## 专家学到了什么？

ST-MoE 作者观察到，编码器中的专家会专门处理某一组 token 或浅层概念。例如，最终可能会出现一个标点符号专家、一个专有名词专家等。另一方面，解码器中的专家专业化程度较低。作者还在多语言环境下进行了训练。虽然人们可能会想象每个专家会专门处理一种语言，但实际情况恰恰相反：由于 token 路由和负载均衡的原因，没有任何一个专家专门负责某种特定语言。

![Experts specialize in some token groups](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/moe/05_experts_learning.png)

*来自 ST-MoE 论文的表格，展示了哪些 token 组被发送到哪个专家。*

## 扩展专家数量对预训练有何影响？

更多专家能带来更好的样本效率和更快的加速，但收益是递减的（尤其是超过 256 或 512 之后），并且推理需要更多显存。Switch Transformers 在大规模下研究的性质在小规模下也是一致的，即使每层只有 2、4 或 8 个专家也是如此。

## 微调 MoE

> Mixtral 在 transformers 4.36.0 版本中得到支持。你可以使用 `pip install transformers==4.36.0 --upgrade` 安装。

稠密模型和稀疏模型的过拟合动态非常不同。稀疏模型更容易过拟合，因此我们可以在专家内部探索更高的正则化（例如 dropout）（例如稠密层使用一个 dropout 率，而稀疏层使用另一个更高的 dropout 率）。

一个问题是微调时是否使用辅助损失。ST-MoE 作者尝试关闭辅助损失，即使有多达 11% 的 token 被丢弃，质量也没有受到显著影响。Token 丢弃可能是一种正则化形式，有助于防止过拟合。

Switch Transformers 观察到，在固定的预训练困惑度下，稀疏模型在下游任务中表现不如稠密对应模型，尤其是在 SuperGLUE 等推理密集型任务上。另一方面，对于 TriviaQA 等知识密集型任务，稀疏模型表现得格外好。作者还观察到，较少的专家数量有助于微调。另一个证实泛化问题的观察是，模型在较小任务上表现较差，但在较大任务上表现良好。

![Fine-tuning learning curves](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/moe/06_superglue_curves.png)

*在小任务（左图）中，我们可以看到明显的过拟合，稀疏模型在验证集上表现差得多。在较大任务（右图）中，MoE 表现良好。图片来自 ST-MoE 论文。*

我们可以尝试冻结所有非专家权重，也就是只更新 MoE 层。这会导致性能大幅下降。我们可以反过来尝试：只冻结 MoE 层中的参数，这种方法效果几乎与更新所有参数一样好。这有助于加速微调并减少内存使用。这有些反直觉，因为 ST-MoE 项目中 80% 的参数都在 MoE 层中。他们对这一架构的假设是，由于专家层每 1/4 层才出现一次，每个 token 每层最多看到两个专家，所以更新 MoE 参数影响的层数远少于更新其他参数。

![Only updating the non MoE layers works well in fine-tuning](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/moe/07_superglue_bars.png)

*仅冻结 MoE 层可以加速训练同时保持质量。图片来自 ST-MoE 论文。*

微调稀疏 MoE 时最后一个需要考虑的方面是，它们有不同的微调超参数设置——例如，稀疏模型更倾向于从较小的批量大小和较高的学习率中获益。

![Table comparing fine-tuning batch size and learning rate between dense and sparse models.](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/moe/08_superglue_dense_vs_sparse.png)

*稀疏模型在较高学习率和较小批量大小下的微调质量更好。图片来自 ST-MoE 论文。*

读到这里，你可能会因为微调 MoE 的困难而感到失落。令人兴奋的是，最近的一篇论文 [MoEs Meets Instruction Tuning](https://arxiv.org/pdf/2305.14705.pdf)（2023 年 7 月）进行了如下实验：

- 单任务微调
- 多任务指令微调
- 多任务指令微调后再单任务微调

当作者微调 MoE 和等价的 T5 时，等价的 T5 表现更好。但当作者微调 Flan T5（T5 的指令版本）的 MoE 版本时，MoE 表现显著更好。不仅如此，Flan-MoE 相对于 MoE 的提升幅度大于 Flan-T5 相对于 T5 的提升，这表明 MoE 可能比稠密模型从指令微调中获益更多。MoE 从更多任务中获益更大。与前文建议关闭辅助损失函数的讨论相反，辅助损失实际上能防止过拟合。

![MoEs benefit even more from instruct tuning than dense models](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/moe/09_fine_tune_evals.png)

*与稠密模型相比，稀疏模型从指令微调中获益更多。图片来自 MoEs Meets Instruction Tuning 论文。*

## 何时使用稀疏 MoE，何时使用稠密模型？

对于具有多机的高吞吐量场景，专家模型非常有用。在固定的预训练计算预算下，稀疏模型会更优。对于显存有限的低吞吐量场景，稠密模型更合适。

**注意：**不能直接比较稀疏模型和稠密模型的参数数量，因为它们代表的含义有显著差异。

## 让 MoE 飞起来

最初的 MoE 工作将 MoE 层呈现为分支结构，这导致计算缓慢，因为 GPU 并非为此设计，同时由于设备需要相互传递信息，网络带宽成为瓶颈。本节将讨论一些现有工作，以使这些模型的预训练和推理更加实用。让 MoE 真正飞起来！

### 专家并行

让我们简要回顾一下并行性：

- **数据并行**：相同的权重在所有核心上复制，数据在核心之间分区。
- **模型并行**：模型在核心之间分区，数据在核心之间复制。
- **模型与数据并行**：我们可以将模型和数据都跨核心分区。注意不同核心处理不同批次的数据。
- **专家并行**：专家被放置在不同的 worker 上。如果与数据并行结合，每个核心拥有不同的专家，数据则跨所有核心分区。

在专家并行中，专家被放置在不同的 worker 上，每个 worker 处理不同批次的训练样本。对于非 MoE 层，专家并行的行为与数据并行相同。对于 MoE 层，序列中的 token 会被发送到所需专家所在的 worker。

![Image illustrating model, expert, and data prallelism](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/moe/10_parallelism.png)

*来自 Switch Transformers 论文的插图，展示了不同并行技术下数据和模型如何在核心间划分。*

### 容量因子与通信成本

提高容量因子（CF）会提升质量，但会增加通信成本和激活值的内存。如果 all-to-all 通信较慢，使用较小的容量因子更好。一个不错的起点是使用 top-2 路由、容量因子为 1.25 且每个核心一个专家。评估期间可以调整容量因子以减少计算量。

### 服务部署技术

> 你可以将 [mistralai/Mixtral-8x7B-Instruct-v0.1](https://ui.endpoints.huggingface.co/new?repository=mistralai%2FMixtral-8x7B-Instruct-v0.1&vendor=aws&region=us-east-1&accelerator=gpu&instance_size=2xlarge&task=text-generation&no_suggested_compute=true&tgi=true&tgi_max_batch_total_tokens=1024000&tgi_max_total_tokens=32000) 部署到 Inference Endpoints。

MoE 的一大缺点是参数数量庞大。对于本地使用场景，可能希望使用更小的模型。让我们快速讨论几种有助于部署服务的技术：

- Switch Transformers 作者做过早期的蒸馏实验。通过将 MoE 蒸馏回其稠密对应模型，他们能够保留 30-40% 的稀疏性收益。因此蒸馏带来了更快预训练和在生产中使用更小模型的双重好处。
- 近期方法修改路由策略，将完整句子或任务路由到某个专家，从而允许提取子网络用于部署。
- 专家聚合（MoE）：这一技术合并专家的权重，从而减少推理时的参数数量。

### 更多关于高效训练的内容

FasterMoE（2022 年 3 月）分析了 MoE 在高效分布式系统中的性能，分析了不同并行策略的理论极限，以及调整专家流行度的技术、减少延迟的细粒度通信调度，以及基于最低延迟选择专家的拓扑感知门控调整，最终实现了 17 倍的加速。

Megablocks（2022 年 11 月）通过提供新的 GPU 内核来处理 MoE 中存在的动态性，探索了高效的稀疏预训练。他们的方案永远不会丢弃 token，并能高效映射到现代硬件，带来显著加速。诀窍是什么？传统 MoE 使用批量矩阵乘法，这假设所有专家具有相同的形状和相同数量的 token。相比之下，Megablocks 将 MoE 层表示为可以适应不均衡分配的块稀疏操作。

![Matrix multiplication optimized for block-sparse operations.](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/moe/11_expert_matmuls.png)

*针对不同大小专家和 token 数量的块稀疏矩阵乘法（来自 [MegaBlocks](https://arxiv.org/abs/2211.15841)）。*

## 开源 MoE 模型

如今有几个用于训练 MoE 的开源项目：

- Megablocks：<https://github.com/stanford-futuredata/megablocks>
- Fairseq：<https://github.com/facebookresearch/fairseq/tree/main/examples/moe_lm>
- OpenMoE：<https://github.com/XueFuzhao/OpenMoE>

在已发布的开源 MoE 模型领域，你可以查看：

- [Switch Transformers (Google)](https://huggingface.co/collections/google/switch-transformers-release-6548c35c6507968374b56d1f)：基于 T5 的 MoE 集合，专家数量从 8 到 2048 不等。最大的模型有 1.6 万亿参数。
- [NLLB MoE (Meta)](https://huggingface.co/facebook/nllb-moe-54b)：NLLB 翻译模型的 MoE 变体。
- [OpenMoE](https://huggingface.co/fuzhao)：社区发布的基于 Llama 的 MoE。
- [Mixtral 8x7B (Mistral)](https://huggingface.co/mistralai)：高质量的 MoE，性能超过 Llama 2 70B，并且推理速度快得多。也发布了一个指令微调版本。更多信息请阅读[官方公告博客](https://mistral.ai/news/mixtral-of-experts/)。

## 值得关注的研究方向

进一步实验**蒸馏**稀疏 MoE 回到参数更少但质量相似的稠密模型。

另一个方向是 MoE 的量化。[QMoE](https://arxiv.org/abs/2310.16795)（2023 年 10 月）是这个方向的一个重要进展，它将 MoE 量化到每参数不到 1 bit，从而将原本需要 3.2TB 加速器的 1.6T Switch Transformer 压缩到仅 160GB。

简而言之，以下是一些值得探索的有趣方向：

- 将 Mixtral 蒸馏成稠密模型
- 探索专家的模型合并技术及其对推理时间的影响
- 对 Mixtral 进行极致量化技术的探索

## 参考资源

- [Adaptive Mixture of Local Experts (1991)](https://www.cs.toronto.edu/~hinton/absps/jjnh91.pdf)
- [Learning Factored Representations in a Deep Mixture of Experts (2013)](https://arxiv.org/abs/1312.4314)
- [Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer (2017)](https://arxiv.org/abs/1701.06538)
- [GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding (Jun 2020)](https://arxiv.org/abs/2006.16668)
- [GLaM: Efficient Scaling of Language Models with Mixture-of-Experts (Dec 2021)](https://arxiv.org/abs/2112.06905)
- [Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity (Jan 2022)](https://arxiv.org/abs/2101.03961)
- [ST-MoE: Designing Stable and Transferable Sparse Expert Models (Feb 2022)](https://arxiv.org/abs/2202.08906)
- [FasterMoE: modeling and optimizing training of large-scale dynamic pre-trained models (April 2022)](https://dl.acm.org/doi/10.1145/3503221.3508418)
- [MegaBlocks: Efficient Sparse Training with Mixture-of-Experts (Nov 2022)](https://arxiv.org/abs/2211.15841)
- [Mixture-of-Experts Meets Instruction Tuning: A Winning Combination for Large Language Models (May 2023)](https://arxiv.org/abs/2305.14705)
- [Mixtral-8x7B-v0.1](https://huggingface.co/mistralai/Mixtral-8x7B-v0.1)、[Mixtral-8x7B-Instruct-v0.1](https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1)

## 引用

```bibtex
@misc {sanseviero2023moe,
    author       = { Omar Sanseviero and
                     Lewis Tunstall and
                     Philipp Schmid and
                     Sourab Mangrulkar and
                     Younes Belkada and
                     Pedro Cuenca
                   },
    title        = { Mixture of Experts Explained },
    year         = 2023,
    url          = { https://huggingface.co/blog/moe },
    publisher    = { Hugging Face Blog }
}
```

```
Sanseviero, et al., "Mixture of Experts Explained", Hugging Face Blog, 2023.
```
