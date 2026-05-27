# DeepSeekMoE：迈向混合专家语言模型中的终极专家特化

**作者**：Damai Dai¹,², Chengqi Deng¹, Chenggang Zhao¹,³, R.X. Xu¹, Huazuo Gao¹, Deli Chen¹, Jiashi Li¹, Wangding Zeng¹, Xingkai Yu¹,⁴, Y. Wu¹, Zhenda Xie¹, Y.K. Li¹, Panpan Huang¹, Fuli Luo¹, Chong Ruan¹, Zhifang Sui², Wenfeng Liang¹

¹ DeepSeek-AI
² 北京大学多媒体信息处理国家重点实验室
³ 清华大学交叉信息研究院
⁴ 南京大学计算机软件新技术国家重点实验室

联系方式：{daidamai, szf}@pku.edu.cn, {wenfeng.liang}@deepseek.com
项目地址：<https://github.com/deepseek-ai/DeepSeek-MoE>

arXiv 原文：<https://arxiv.org/abs/2401.06066>

---

## 摘要

在大语言模型时代，混合专家（Mixture-of-Experts，MoE）是一种在扩大模型参数规模时管理计算成本的有前景的架构。然而，诸如 GShard 这样的传统 MoE 架构通过从 N 个专家中激活 top-K 个专家，在确保**专家特化**（即每个专家获得不重叠且聚焦的知识）方面面临挑战。

为此，我们提出了 DeepSeekMoE 架构，旨在实现终极的专家特化。该架构包含两个主要策略：(1) 将专家**细粒度地切分**为 mN 个，并从中激活 mK 个，从而允许更灵活地组合激活的专家；(2) 将 Kₛ 个专家**隔离为共享专家**，旨在捕获通用知识并减轻路由专家中的冗余。

我们从 2B 参数的中等规模开始，证明了 DeepSeekMoE 2B 在性能上与 GShard 2.9B 相当，而后者的专家参数和计算量是前者的 1.5 倍。此外，DeepSeekMoE 2B 几乎达到了具有相同参数总量的稠密模型对照组的性能，而该稠密模型设定了 MoE 模型的性能上限。随后，我们将 DeepSeekMoE 扩展到 16B 参数，并展示其在仅使用约 40% 计算量的情况下达到了与 LLaMA2 7B 相当的性能。此外，我们将 DeepSeekMoE 扩展到 145B 参数的初步尝试持续验证了其相比 GShard 架构的实质性优势，并显示其性能可与 DeepSeek 67B 相媲美，而仅使用 28.5%（甚至可能仅 18.2%）的计算量。

![Refer to caption](https://arxiv.org/html/2401.06066v1/x1.png)

**图 1**：DeepSeekMoE 16B 与 Open LLM Leaderboard 上开源模型的对比。红色虚线由除 DeepSeekMoE 16B 之外所有模型的数据点线性拟合得到。DeepSeekMoE 16B 大幅超越具有相似激活参数量的模型，并达到了与 LLaMA2 7B 相当的性能，而后者的激活参数约为前者的 2.5 倍。

---

## 1 引言

近期的研究和实践表明，在有充足训练数据的情况下，通过增加参数和计算预算来扩展语言模型可以产生显著更强的模型（Brown 等人，2020；OpenAI，2023；Touvron 等人，2023a；Hoffmann 等人，2022）。然而必须承认的是，将模型扩展到极大规模的努力也伴随着极高的计算成本。考虑到这些可观的成本，**混合专家（MoE）架构**（Jacobs 等人，1991；Jordan 和 Jacobs，1994；Shazeer 等人，2017）已成为一种流行的解决方案。它能够实现参数扩展，同时将计算成本保持在适度水平。MoE 架构在 Transformer（Vaswani 等人，2017）中的近期应用已成功地将语言模型扩展到了相当大的规模（Fedus 等人，2021；Lepikhin 等人，2021；Du 等人，2022；Zoph，2022），并取得了卓越的性能。这些成就突显了 MoE 语言模型的巨大潜力和前景。

尽管 MoE 架构前景广阔，现有的 MoE 架构可能存在**知识混杂**（knowledge hybridity）和**知识冗余**（knowledge redundancy）的问题，这限制了专家的特化（即每个专家获得不重叠且聚焦的知识）。传统的 MoE 架构将 Transformer 中的前馈网络（FFN）替换为 MoE 层。每个 MoE 层由多个专家组成，每个专家在结构上等同于一个标准 FFN，每个 token 被分配给一个（Fedus 等人，2021）或两个（Lepikhin 等人，2021）专家。这种架构表现出两个潜在问题：

(1) **知识混杂**：现有的 MoE 实践通常采用有限数量的专家（例如 8 或 16），因此分配给特定专家的 token 很可能涵盖各种各样的知识。结果就是，被指定的专家会倾向于在其参数中聚合大相径庭的知识类型，而这些知识难以同时被利用。

(2) **知识冗余**：分配给不同专家的 token 可能需要共同的知识。结果就是，多个专家可能会在它们各自的参数中收敛于获取共享知识，从而导致专家参数的冗余。

这些问题共同阻碍了现有 MoE 实践中的专家特化，使其无法达到 MoE 模型的理论上限性能。

针对上述问题，我们引入了 DeepSeekMoE，一种专门为实现终极专家特化而设计的创新 MoE 架构。我们的架构包含两个主要策略：

(1) **细粒度专家切分**：在保持参数数量不变的同时，我们通过分割 FFN 中间隐藏维度，将专家细粒度地切分。相应地，在保持计算成本不变的情况下，我们也激活更多细粒度的专家，从而实现更灵活、更具适应性的激活专家组合。细粒度专家切分使得多样化的知识可以被更细致地分解，并被更精确地学习到不同的专家中，每个专家因而能保持更高水平的特化。此外，激活专家组合的灵活性提升也有助于更准确、更有针对性的知识获取。

(2) **共享专家隔离**：我们隔离某些专家作为**共享专家**，使其始终被激活，目的是捕获并整合各种上下文中的共同知识。通过将共同知识压缩到这些共享专家中，其他路由专家之间的冗余将得到缓解。这可以增强参数效率，并确保每个路由专家通过专注于独特方面而保持特化。

DeepSeekMoE 的这些架构创新为训练一个参数高效、每个专家都高度特化的 MoE 语言模型提供了机会。

我们从 2B 参数的中等规模开始，验证了 DeepSeekMoE 架构的优势。我们在涵盖各种任务的 12 个零样本或少样本基准上进行了评估。实证结果表明，DeepSeekMoE 2B 显著超越了 GShard 2B（Lepikhin 等人，2021），甚至与 GShard 2.9B（一个具有 1.5 倍专家参数和计算量的更大 MoE 模型）相匹配。值得注意的是，我们发现 DeepSeekMoE 2B 几乎达到了等量参数稠密模型对照组的性能，而后者设定了 MoE 语言模型的严格上限。为了获得更深入的洞察，我们对 DeepSeekMoE 的专家特化进行了详尽的消融研究和分析。这些研究验证了细粒度专家切分和共享专家隔离的有效性，并提供了实证证据，支持 DeepSeekMoE 能够实现高水平专家特化的论断。

借助我们的架构，我们随后将模型参数扩展到 16B，并在一个 2T tokens 的大规模语料库上训练 DeepSeekMoE 16B。评估结果显示，DeepSeekMoE 16B 仅使用约 40% 的计算量，便达到了与 DeepSeek 7B（DeepSeek-AI，2024）相当的性能，后者是一个在相同的 2T 语料库上训练的稠密模型。我们也将 DeepSeekMoE 与开源模型进行了对比，评估表明 DeepSeekMoE 16B 大幅超越了具有相似激活参数量的模型，并达到了与 LLaMA2 7B（Touvron 等人，2023b）相当的性能，而后者的激活参数约为前者的 2.5 倍。图 1 展示了在 Open LLM Leaderboard¹ 上的评估结果。此外，我们进行了监督微调（SFT）以实现对齐，将模型转化为一个聊天模型。评估结果显示，DeepSeekMoE Chat 16B 在聊天场景下也达到了与 DeepSeek Chat 7B 和 LLaMA2 SFT 7B 相当的性能。

受这些结果的鼓舞，我们进一步尝试将 DeepSeekMoE 扩展到 145B。实验结果持续验证了其相对于 GShard 架构的实质性优势。此外，它显示出与 DeepSeek 67B 相当的性能，仅使用 28.5%（甚至可能仅 18.2%）的计算量。

¹ <https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard>

我们的贡献总结如下：

- **架构创新**：我们引入了 DeepSeekMoE，一种旨在实现终极专家特化的创新 MoE 架构，它采用了细粒度专家切分和共享专家隔离两大主要策略。
- **实证验证**：我们进行了大量实验来实证地验证 DeepSeekMoE 架构的有效性。实验结果验证了 DeepSeekMoE 2B 中高水平的专家特化，并表明 DeepSeekMoE 2B 几乎可以达到 MoE 模型的上限性能。
- **可扩展性**：我们将 DeepSeekMoE 扩展以训练一个 16B 模型，并展示其仅使用约 40% 的计算量便达到了与 DeepSeek 7B 和 LLaMA2 7B 相当的性能。我们还进行了将 DeepSeekMoE 扩展到 145B 的初步尝试，突显了它相对于 GShard 架构的持续优势，并显示出与 DeepSeek 67B 相当的性能。
- **MoE 对齐**：我们成功地对 DeepSeekMoE 16B 进行了监督微调以创建对齐的聊天模型，展示了 DeepSeekMoE 16B 的适应性和多功能性。
- **公开发布**：本着开放研究的精神，我们向公众发布 DeepSeekMoE 16B 的模型检查点。值得注意的是，该模型可以部署在单个 40GB 内存的 GPU 上，无需量化。

---

## 2 预备知识：Transformer 中的混合专家

我们首先介绍 Transformer 语言模型中常用的通用 MoE 架构。一个标准 Transformer 语言模型由 L 层标准 Transformer 模块堆叠而成，每个模块可以表示为：

$$\mathbf{u}_{1:T}^{l} = \operatorname{Self-Att}(\mathbf{h}_{1:T}^{l-1}) + \mathbf{h}_{1:T}^{l-1} \tag{1}$$

$$\mathbf{h}_{t}^{l} = \operatorname{FFN}(\mathbf{u}_{t}^{l}) + \mathbf{u}_{t}^{l} \tag{2}$$

其中 T 表示序列长度，Self-Att(·) 表示自注意力模块，FFN(·) 表示前馈网络（FFN），$\mathbf{u}_{1:T}^{l} \in \mathbb{R}^{T \times d}$ 是所有 token 经过第 l 个注意力模块后的隐藏状态，$\mathbf{h}_{t}^{l} \in \mathbb{R}^{d}$ 是第 t 个 token 经过第 l 个 Transformer 模块后的输出隐藏状态。为简洁起见，我们在上述公式中省略了层归一化操作。

构建 MoE 语言模型的典型做法通常是在指定的间隔将 Transformer 中的 FFN 替换为 MoE 层（Fedus 等人，2021；Lepikhin 等人，2021；Du 等人，2022；Zoph，2022）。一个 MoE 层由多个专家组成，每个专家在结构上与标准 FFN 相同。然后，每个 token 被分配给一个（Fedus 等人，2021）或两个（Lepikhin 等人，2021）专家。如果将第 l 个 FFN 替换为 MoE 层，其输出隐藏状态 $\mathbf{h}_{t}^{l}$ 的计算可以表示为：

$$\mathbf{h}_{t}^{l} = \sum_{i=1}^{N}\left(g_{i,t} \operatorname{FFN}_{i}(\mathbf{u}_{t}^{l})\right) + \mathbf{u}_{t}^{l} \tag{3}$$

$$g_{i,t} = \begin{cases} s_{i,t}, & s_{i,t} \in \operatorname{Topk}(\{s_{j,t} | 1 \leqslant j \leqslant N\}, K), \\ 0, & \text{otherwise}, \end{cases} \tag{4}$$

$$s_{i,t} = \operatorname{Softmax}_{i}\left({\mathbf{u}_{t}^{l}}^{T} \mathbf{e}_{i}^{l}\right) \tag{5}$$

其中 N 表示专家总数，$\operatorname{FFN}_i(\cdot)$ 是第 i 个专家 FFN，$g_{i,t}$ 表示第 i 个专家的门控值，$s_{i,t}$ 表示 token-专家亲和度，$\operatorname{Topk}(\cdot, K)$ 表示第 t 个 token 与所有 N 个专家计算得到的亲和度分数中最高的 K 个所组成的集合，$\mathbf{e}_i^l$ 是第 l 层中第 i 个专家的中心向量。注意 $g_{i,t}$ 是稀疏的，意味着 N 个门控值中只有 K 个是非零的。这种稀疏特性确保了 MoE 层内的计算效率，即每个 token 只会被分配并在 K 个专家中进行计算。同样，在上述公式中我们为简洁起见省略了层归一化操作。

![Refer to caption](https://arxiv.org/html/2401.06066v1/x2.png)

**图 2**：DeepSeekMoE 示意图。子图 (a) 展示了采用传统 top-2 路由策略的 MoE 层。子图 (b) 展示了细粒度专家切分策略。随后，子图 (c) 展示了集成共享专家隔离策略，构成了完整的 DeepSeekMoE 架构。值得注意的是，在这三种架构中，专家参数数量和计算成本保持不变。

---

## 3 DeepSeekMoE 架构

在第 2 节概述的通用 MoE 架构之上，我们引入了 DeepSeekMoE，它专门用于挖掘专家特化的潜力。如图 2 所示，我们的架构包含两个主要策略：细粒度专家切分和共享专家隔离。这两个策略都旨在提升专家特化的水平。

### 3.1 细粒度专家切分

在专家数量有限的情况下，分配给特定专家的 token 更可能涵盖多种类型的知识。因此，被指定的专家会倾向于在其参数中学习大相径庭的知识类型，而这些知识难以同时被利用。然而，如果每个 token 可以被路由到更多的专家，多样化的知识就有可能被分解并分别在不同专家中学习。在此情境下，每个专家仍可以保持高水平的专家特化，从而使专家间的知识分布更加聚焦。

为了实现这一目标，在保持专家参数数量和计算成本一致的前提下，我们对专家进行了更细粒度的切分。更细粒度的专家切分实现了更灵活、更具适应性的激活专家组合。具体而言，在图 2(a) 所示的典型 MoE 架构基础上，我们通过将 FFN 中间隐藏维度降至原大小的 1/m，将每个专家 FFN 切分为 m 个更小的专家。由于每个专家变得更小，相应地，我们也将激活的专家数量增加到 m 倍以保持相同的计算成本，如图 2(b) 所示。采用细粒度专家切分后，MoE 层的输出可以表示为：

$$\mathbf{h}_{t}^{l} = \sum_{i=1}^{mN}\left(g_{i,t} \operatorname{FFN}_{i}(\mathbf{u}_{t}^{l})\right) + \mathbf{u}_{t}^{l} \tag{6}$$

$$g_{i,t} = \begin{cases} s_{i,t}, & s_{i,t} \in \operatorname{Topk}(\{s_{j,t} | 1 \leqslant j \leqslant mN\}, mK), \\ 0, & \text{otherwise}, \end{cases} \tag{7}$$

$$s_{i,t} = \operatorname{Softmax}_{i}\left({\mathbf{u}_{t}^{l}}^{T} \mathbf{e}_{i}^{l}\right) \tag{8}$$

其中专家参数总数等于 N 倍的标准 FFN 参数数量，mN 表示细粒度专家的总数。采用细粒度专家切分策略后，非零门控的数量也增加到 mK。

从组合的角度来看，细粒度专家切分策略大大增强了激活专家的组合灵活性。举例说明，考虑 N=16 的情况。典型的 top-2 路由策略可以产生 $\binom{16}{2}=120$ 种可能的组合。相比之下，如果每个专家被切分为 4 个更小的专家，细粒度路由策略可以产生 $\binom{64}{8}=4{,}426{,}165{,}368$ 种潜在组合。组合灵活性的激增增强了实现更准确、更有针对性知识获取的潜力。

### 3.2 共享专家隔离

采用传统路由策略时，分配给不同专家的 token 可能需要某些共同的知识或信息。结果就是，多个专家可能会在它们各自的参数中收敛于获取共享知识，从而导致专家参数的冗余。然而，如果有专门用于捕获和整合不同上下文中共同知识的共享专家，其他路由专家间的参数冗余就会得到缓解。这种冗余的缓解将有助于构建一个参数更高效、专家更特化的模型。

为实现这一目标，除了细粒度专家切分策略外，我们进一步隔离 Kₛ 个专家作为共享专家。无论路由模块如何，每个 token 都会被确定性地分配给这些共享专家。为了保持计算成本不变，其他路由专家中激活的专家数量将减少 Kₛ，如图 2(c) 所示。在集成共享专家隔离策略后，完整 DeepSeekMoE 架构中的 MoE 层公式如下：

$$\mathbf{h}_{t}^{l} = \sum_{i=1}^{K_s}\operatorname{FFN}_{i}(\mathbf{u}_{t}^{l}) + \sum_{i=K_s+1}^{mN}\left(g_{i,t} \operatorname{FFN}_{i}(\mathbf{u}_{t}^{l})\right) + \mathbf{u}_{t}^{l} \tag{9}$$

$$g_{i,t} = \begin{cases} s_{i,t}, & s_{i,t} \in \operatorname{Topk}(\{s_{j,t} | K_s+1 \leqslant j \leqslant mN\}, mK-K_s), \\ 0, & \text{otherwise}, \end{cases} \tag{10}$$

$$s_{i,t} = \operatorname{Softmax}_{i}\left({\mathbf{u}_{t}^{l}}^{T} \mathbf{e}_{i}^{l}\right) \tag{11}$$

最后，在 DeepSeekMoE 中，共享专家的数量是 Kₛ，路由专家的总数是 mN−Kₛ，非零门控的数量是 mK−Kₛ。

值得一提的是，共享专家隔离的原型可追溯到 Rajbhandari 等人（2022）。关键区别在于他们从工程角度推导出该策略，而我们则从算法角度入手。

### 3.3 负载均衡考量

自动学习的路由策略可能遇到负载不均衡问题，这会带来两个显著缺陷。首先，存在**路由崩溃**（routing collapse）的风险（Shazeer 等人，2017），即模型总是只选择少数几个专家，导致其他专家无法得到充分训练。其次，如果专家分布在多个设备上，负载不均衡会加剧计算瓶颈。

#### 专家级均衡损失

为了减轻路由崩溃的风险，我们采用了专家级均衡损失。均衡损失的计算如下：

$$\mathcal{L}_{\mathrm{ExpBal}} = \alpha_{1} \sum_{i=1}^{N'} f_{i} P_{i} \tag{12}$$

$$f_i = \frac{N'}{K'T} \sum_{t=1}^{T} \mathbb{1}(\text{token } t \text{ 选择专家 } i) \tag{13}$$

$$P_i = \frac{1}{T} \sum_{t=1}^{T} s_{i,t} \tag{14}$$

其中 α₁ 是一个超参数，称为专家级均衡因子，为简洁起见，N′ 等于 (mN−Kₛ)，K′ 等于 (mK−Kₛ)。$\mathbb{1}(\cdot)$ 表示指示函数。

#### 设备级均衡损失

除了专家级均衡损失，我们还引入了设备级均衡损失。当目标是缓解计算瓶颈时，并不需要在专家级别强制执行严格的均衡约束，因为对负载均衡的过度约束会损害模型性能。相反，我们的主要目标是确保跨设备的计算均衡。如果我们将所有路由专家划分为 D 组 $\{\mathcal{E}_1, \mathcal{E}_2, ..., \mathcal{E}_D\}$，并将每组部署在单个设备上，设备级均衡损失计算如下：

$$\mathcal{L}_{\mathrm{DevBal}} = \alpha_{2} \sum_{i=1}^{D} f_{i}' P_{i}' \tag{15}$$

$$f_i' = \frac{1}{|\mathcal{E}_i|} \sum_{j \in \mathcal{E}_i} f_j \tag{16}$$

$$P_i' = \sum_{j \in \mathcal{E}_i} P_j \tag{17}$$

其中 α₂ 是一个超参数，称为设备级均衡因子。在实践中，我们设置一个较小的专家级均衡因子以减轻路由崩溃的风险，同时设置一个较大的设备级均衡因子以促进跨设备的计算均衡。

---

## 4 验证实验

### 4.1 实验设置

#### 4.1.1 训练数据和分词

我们的训练数据采样自 DeepSeek-AI 创建的大规模多语种语料库。该语料库主要聚焦英文和中文，但也涵盖其他语言。它来源于多样的渠道，包括网页文本、数学材料、代码脚本、已出版文献以及其他各种文本材料。出于验证实验的目的，我们从该语料库中采样了一个包含 100B tokens 的子集来训练模型。对于分词，我们使用 HuggingFace Tokenizer² 工具在训练语料库的一个较小子集上训练 BPE（Byte Pair Encoding）（Sennrich 等人，2016）分词器。在验证实验中，我们准备了一个词表大小为 8K 的分词器，训练更大模型时词表大小将相应扩大。

² <https://github.com/huggingface/tokenizers>

#### 4.1.2 基础设施

我们基于 HAI-LLM（High-Flyer，2023）开展实验，这是一个高效轻量的训练框架，集成了多种并行策略，包括张量并行（Shoeybi 等人，2019；Narayanan 等人，2021；Korthikanti 等人，2023）、ZeRO 数据并行（Rajbhandari 等人，2020）、PipeDream 流水线并行（Harlap 等人，2018），更具体地，通过结合数据并行和张量并行实现了专家并行（Lepikhin 等人，2021）。为了优化性能，我们使用 CUDA 和 Triton（Tillet 等人，2019）开发了用于门控算法和融合不同专家中线性层计算的 GPU kernel。

所有实验均在配备 NVIDIA A100 或 H800 GPU 的集群上进行。A100 集群的每个节点包含 8 个通过 NVLink 桥两两相连的 GPU。H800 集群同样每个节点配置 8 个 GPU，节点内通过 NVLink 和 NVSwitch 互连。对于 A100 和 H800 集群，节点间通信均采用 InfiniBand 互连。

#### 4.1.3 超参数

##### 模型设置

在验证实验中，我们将 Transformer 层数设为 9，隐藏维度设为 1280。我们采用多头注意力机制，总共 10 个注意力头，每个头的维度为 128。对于初始化，所有可学习参数均以 0.006 的标准差随机初始化。我们将所有 FFN 替换为 MoE 层，并确保专家参数总数等于标准 FFN 参数的 16 倍。此外，我们保持激活的专家参数（包括共享专家参数和激活的路由专家参数）为标准 FFN 的 2 倍。在此配置下，每个 MoE 模型大约有 2B 总参数，激活参数约为 0.3B。

##### 训练设置

我们采用 AdamW 优化器（Loshchilov 和 Hutter，2019），超参数设置为 β₁=0.9，β₂=0.95，weight_decay=0.1。学习率采用 warmup 加阶梯衰减策略进行调度。最初，学习率在前 2K 步内从 0 线性增加到最大值。随后，在训练步数的 80% 时学习率乘以 0.316，在 90% 时再次乘以 0.316。验证实验的最大学习率设为 1.08×10⁻³，梯度裁剪范数设为 1.0。批量大小设为 2K，最大序列长度为 2K，每个训练 batch 包含 4M tokens。相应地，总训练步数设为 25,000，以达到 100B 训练 tokens。由于训练数据充足，训练过程中不使用 dropout。由于模型规模相对较小，所有参数（包括专家参数）都部署在单个 GPU 设备上以避免计算不均衡。相应地，训练时不丢弃任何 token，也不采用设备级均衡损失。为防止路由崩溃，我们将专家级均衡因子设为 0.01。

为便于阅读，我们也在附录 A 中提供了不同规模 DeepSeekMoE 超参数的概览表。

#### 4.1.4 评估基准

我们在涵盖各种类型任务的广泛基准上进行评估。基准列表如下。

##### 语言建模

对于语言建模，我们在 Pile（Gao 等人，2020）的测试集上评估模型，评估指标为交叉熵损失。

##### 语言理解和推理

对于语言理解和推理，我们考虑 HellaSwag（Zellers 等人，2019）、PIQA（Bisk 等人，2020）、ARC-challenge 和 ARC-easy（Clark 等人，2018）。这些任务的评估指标为准确率。

##### 阅读理解

对于阅读理解，我们使用 RACE-high 和 RACE-middle（Lai 等人，2017），评估指标为准确率。

##### 代码生成

对于代码生成，我们在 HumanEval（Chen 等人，2021）和 MBPP（Austin 等人，2021）上评估模型。评估指标为 Pass@1，表示仅一次生成尝试的通过率。

##### 闭卷问答

对于闭卷问答，我们考虑 TriviaQA（Joshi 等人，2017）和 NaturalQuestions（Kwiatkowski 等人，2019）。评估指标为完全匹配（EM）率。

| 指标 | # Shot | Dense | Hash Layer | Switch | GShard | DeepSeekMoE |
|---|---|---|---|---|---|---|
| 总参数量 | N/A | 0.2B | 2.0B | 2.0B | 2.0B | 2.0B |
| 激活参数量 | N/A | 0.2B | 0.2B | 0.2B | 0.3B | 0.3B |
| 每 2K Tokens 的 FLOPs | N/A | 2.9T | 2.9T | 2.9T | 4.3T | 4.3T |
| 训练 Tokens 数 | N/A | 100B | 100B | 100B | 100B | 100B |
| Pile (损失) | N/A | 2.060 | 1.932 | 1.881 | 1.867 | **1.808** |
| HellaSwag (准确率) | 0-shot | 38.8 | 46.2 | 49.1 | 50.5 | **54.8** |
| PIQA (准确率) | 0-shot | 66.8 | 68.4 | 70.5 | 70.6 | **72.3** |
| ARC-easy (准确率) | 0-shot | 41.0 | 45.3 | 45.9 | 43.9 | **49.4** |
| ARC-challenge (准确率) | 0-shot | 26.0 | 28.2 | 30.2 | 31.6 | **34.3** |
| RACE-middle (准确率) | 5-shot | 38.8 | 38.8 | 43.6 | 42.1 | **44.0** |
| RACE-high (准确率) | 5-shot | 29.0 | 30.0 | 30.9 | 30.4 | **31.7** |
| HumanEval (Pass@1) | 0-shot | 0.0 | 1.2 | 2.4 | 3.7 | **4.9** |
| MBPP (Pass@1) | 3-shot | 0.2 | 0.6 | 0.4 | 0.2 | **2.2** |
| TriviaQA (EM) | 5-shot | 4.9 | 6.5 | 8.9 | 10.2 | **16.6** |
| NaturalQuestions (EM) | 5-shot | 1.4 | 1.4 | 2.5 | 3.2 | **5.7** |

**表 1**：验证实验的评估结果。粗体表示最佳。与其他 MoE 架构相比，DeepSeekMoE 表现出实质性的性能优势。

### 4.2 评估

#### 基线

包括 DeepSeekMoE 在内，我们对比五个模型进行验证实验。**Dense** 表示一个总参数量为 0.2B 的标准稠密 Transformer 语言模型。**Hash Layer**（Roller 等人，2021）是基于 top-1 哈希路由的 MoE 架构，总参数 2.0B，激活参数 0.2B，与稠密基线对齐。**Switch Transformer**（Fedus 等人，2021）是另一个著名的 MoE 架构，基于 top-1 可学习路由，总参数和激活参数与 Hash Layer 相同。**GShard**（Lepikhin 等人，2021）采用 top-2 可学习路由策略，总参数 2.0B，激活参数 0.3B（相比 top-1 路由方法多激活了一个专家）。**DeepSeekMoE** 有 1 个共享专家和 63 个路由专家，每个专家是标准 FFN 大小的 0.25 倍。包括 DeepSeekMoE 在内的所有对比模型共享相同的训练语料和训练超参数。所有对比的 MoE 模型具有相同数量的总参数，GShard 的激活参数数量与 DeepSeekMoE 相同。

#### 结果

我们在表 1 中展示了评估结果。对于所有展示的模型，我们报告了在 100B tokens 上训练后的最终评估结果。从表中可以观察到：(1) 采用稀疏架构和更多总参数，Hash Layer 和 Switch Transformer 在激活参数数量相同的情况下，性能显著强于稠密基线。(2) 与 Hash Layer 和 Switch Transformer 相比，GShard 有更多激活参数，性能比 Switch Transformer 略好。(3) 在相同总参数数量和激活参数数量下，DeepSeekMoE 相对于 GShard 显示出压倒性的优势。这些结果展示了 DeepSeekMoE 架构在现有 MoE 架构格局中的优越性。

| 指标 | # Shot | GShard×1.5 | Dense×16 | DeepSeekMoE |
|---|---|---|---|---|
| 相对专家大小 | N/A | 1.5 | 1 | 0.25 |
| # 专家 | N/A | 0 + 16 | 16 + 0 | 1 + 63 |
| # 激活专家 | N/A | 0 + 2 | 16 + 0 | 1 + 7 |
| # 总专家参数 | N/A | 2.83B | 1.89B | 1.89B |
| # 激活专家参数 | N/A | 0.35B | 1.89B | 0.24B |
| 每 2K Tokens 的 FLOPs | N/A | 5.8T | 24.6T | 4.3T |
| 训练 Tokens 数 | N/A | 100B | 100B | 100B |
| Pile (损失) | N/A | 1.808 | 1.806 | 1.808 |
| HellaSwag (准确率) | 0-shot | 54.4 | 55.1 | 54.8 |
| PIQA (准确率) | 0-shot | 71.1 | 71.9 | 72.3 |
| ARC-easy (准确率) | 0-shot | 47.3 | 51.9 | 49.4 |
| ARC-challenge (准确率) | 0-shot | 34.1 | 33.8 | 34.3 |
| RACE-middle (准确率) | 5-shot | 46.4 | 46.3 | 44.0 |
| RACE-high (准确率) | 5-shot | 32.4 | 33.0 | 31.7 |
| HumanEval (Pass@1) | 0-shot | 3.0 | 4.3 | 4.9 |
| MBPP (Pass@1) | 3-shot | 2.6 | 2.2 | 2.2 |
| TriviaQA (EM) | 5-shot | 15.7 | 16.5 | 16.6 |
| NaturalQuestions (EM) | 5-shot | 4.7 | 6.3 | 5.7 |

**表 2**：DeepSeekMoE、更大的 GShard 模型和更大的稠密模型之间的对比。在"# 专家"一行中，a + b 表示 a 个共享专家和 b 个路由专家。在"# 激活专家"一行中，a + b 表示 a 个激活的共享专家和 b 个激活的路由专家。DeepSeekMoE 达到了与拥有 1.5 倍专家参数和计算量的 GShard 模型相当的性能。此外，DeepSeekMoE 几乎接近 16 倍 FFN 参数的稠密模型的性能，该稠密模型在模型容量方面为 MoE 模型设定了上限。

### 4.3 DeepSeekMoE 紧密接近 MoE 模型的上限

我们已经证明 DeepSeekMoE 优于稠密基线和其他 MoE 架构。为了更准确地理解 DeepSeekMoE 的性能，我们将它与拥有更多总参数或激活参数的更大基线进行对比。这些对比使我们能够估计 GShard 或稠密基线要达到与 DeepSeekMoE 等效的性能所需的模型大小。

#### 与 GShard×1.5 的对比

表 2 展示了 DeepSeekMoE 与一个专家大小是其 1.5 倍的更大 GShard 模型的对比，这导致专家参数和专家计算量都是 1.5 倍。总体上，我们观察到 DeepSeekMoE 达到了与 GShard×1.5 相当的性能，凸显了 DeepSeekMoE 架构内在的显著优势。除了与 GShard×1.5 的对比，我们还在附录 B 中展示了与 GShard×1.2 的对比。

此外，我们将 DeepSeekMoE 的总参数增加到 13.3B，并将其与拥有 15.9B 和 19.8B 总参数的 GShard×1.2 和 GShard×1.5 进行对比。我们发现在更大规模下，DeepSeekMoE 甚至可以明显超越 GShard×1.5。这些结果也提供在附录 B 中。

#### 与 Dense×16 的对比

表 2 还展示了 DeepSeekMoE 与更大稠密模型的对比。为了公平对比，我们没有使用注意力和 FFN 参数之间常用的比例（1:2）。相反，我们配置了 16 个共享专家，每个专家的参数数量与标准 FFN 相同。这种架构模拟了一个具有 16 倍标准 FFN 参数的稠密模型。从表中我们发现，DeepSeekMoE 几乎接近 Dense×16 的性能，这在模型容量方面为 MoE 模型设定了严格的上限。这些结果表明，至少在大约 2B 参数和 100B 训练 tokens 的规模下，DeepSeekMoE 的性能紧密接近 MoE 模型的理论上限。我们也在附录 B 中提供了与 Dense×4 的额外对比。

![Refer to caption](https://arxiv.org/html/2401.06066v1/x3.png)

**图 3**：DeepSeekMoE 的消融研究。为了清晰展示，性能按最佳性能进行归一化。所有对比的模型具有相同的参数数量和激活参数数量。我们可以发现，细粒度专家切分和共享专家隔离都有助于更强的整体性能。

### 4.4 消融研究

为了证实细粒度专家切分和共享专家隔离策略的有效性，我们对 DeepSeekMoE 进行了消融研究，结果如图 3 所示。为公平对比，我们确保对比中所有模型具有相同的总参数和激活参数数量。

#### 共享专家隔离

为了评估共享专家隔离策略的影响，我们在 GShard 基础上隔离一个专家作为共享专家。从图 3 中我们观察到，与 GShard 相比，特意隔离一个共享专家在大多数基准上都带来了性能提升。这些结果支持共享专家隔离策略有助于更强的模型性能的论断。

#### 细粒度专家切分

为评估细粒度专家切分策略的有效性，我们通过将专家进一步切分为更细的粒度进行更详细的对比。具体地，我们将每个专家切分为 2 或 4 个更小的专家，总共产生 32（1 个共享 + 31 个路由）或 64（1 个共享 + 63 个路由）个专家。图 3 揭示了一个一致的趋势：专家切分粒度的持续细化对应着整体模型性能的持续提升。这些发现为细粒度专家切分策略的有效性提供了实证支持。

#### 共享专家与路由专家的比例

此外，我们研究了共享专家和路由专家的最佳比例。基于具有 64 个总专家的最细粒度，并保持总专家数量和激活专家数量不变，我们尝试隔离 1、2 和 4 个专家作为共享专家。我们发现，共享专家和路由专家的不同比例对性能没有显著影响，1、2 和 4 个共享专家分别达到了 1.808、1.806 和 1.811 的 Pile 损失。考虑到 1:3 的比例产生了略好的 Pile 损失，因此在扩展 DeepSeekMoE 时，我们将共享专家和激活路由专家之间的比例保持为 1:3。

### 4.5 专家特化分析

在本节中，我们对 DeepSeekMoE 2B 的专家特化进行实证分析。本节中的 DeepSeekMoE 2B 指的是表 1 中报告的模型，即由 2.0B 总参数组成，其中有 1 个共享专家和 63 个路由专家中的 7 个被激活。

![Refer to caption](https://arxiv.org/html/2401.06066v1/x4.png)

**图 4**：关于不同比例的禁用 top 路由专家的 Pile 损失。值得注意的是，DeepSeekMoE 对禁用 top 路由专家的比例表现出更大的敏感性，表明 DeepSeekMoE 中路由专家之间的冗余较低。

#### DeepSeekMoE 表现出路由专家间更低的冗余

为评估路由专家之间的冗余，我们禁用不同比例的 top 路由专家并评估 Pile 损失。具体地，对每个 token，我们屏蔽一定比例具有最高路由概率的专家，然后从剩余路由专家中选择 top-K 个专家。为公平起见，我们将 DeepSeekMoE 与 GShard×1.5 对比，因为它们在没有禁用任何专家时具有相同的 Pile 损失。如图 4 所示，与 GShard×1.5 相比，DeepSeekMoE 对禁用 top 路由专家更为敏感。这种敏感性表明 DeepSeekMoE 中的参数冗余水平较低，因为每个路由专家都更难以被替代。相比之下，GShard×1.5 在其专家参数中表现出更大的冗余，因此当 top 路由专家被禁用时它可以缓冲性能下降。

#### 共享专家不可被路由专家替代

为研究共享专家在 DeepSeekMoE 中的作用，我们禁用它并多激活一个路由专家。在 Pile 上的评估显示 Pile 损失显著增加，从 1.808 上升到 2.414，尽管我们保持了相同的计算成本。这一结果突显了共享专家的关键功能，表明共享专家捕获了路由专家所不共享的基础且必要的知识，使其不可被路由专家替代。

![Refer to caption](https://arxiv.org/html/2401.06066v1/x5.png)

**图 5**：关于 DeepSeekMoE 中不同激活路由专家数量的 Pile 损失。仅激活 4 个路由专家时，DeepSeekMoE 便达到了与 GShard 相当的 Pile 损失。

![Refer to caption](https://arxiv.org/html/2401.06066v1/x6.png)

**图 6**：GShard 与激活专家数量减半的 DeepSeekMoE（从头训练）的对比。在相同总专家参数且仅有一半激活专家参数的情况下，DeepSeekMoE 仍然优于 GShard。

#### DeepSeekMoE 更精准地获取知识

为了验证我们关于"激活专家组合的更高灵活性有助于更精准和有针对性的知识获取"的论断，我们研究 DeepSeekMoE 是否可以用更少的激活专家获取所需知识。具体地，我们将激活的路由专家数量从 3 变化到 7 并评估对应的 Pile 损失。如图 5 所示，即使仅激活 4 个路由专家，DeepSeekMoE 也达到了与 GShard 相当的 Pile 损失。这一观察支持 DeepSeekMoE 能够更精准、更高效地获取所需知识的论断。

受这些发现的鼓舞，为了更严格地验证 DeepSeekMoE 的专家特化和精准知识获取，我们从头训练一个新模型。该模型包含 1 个共享专家和 63 个路由专家，其中仅激活 3 个路由专家。图 6 所示的评估结果表明，即使在相同的总专家参数和仅一半的激活专家参数下，DeepSeekMoE 仍然优于 GShard。这凸显了 DeepSeekMoE 更高效利用专家参数的能力，即激活专家中有效参数的比例远高于 GShard。

---

## 5 扩展至 DeepSeekMoE 16B

借助 DeepSeekMoE 架构，我们将 MoE 模型扩展至 16B 总参数的更大规模，并在 2T tokens 上训练。结果表明，与 LLaMA2 7B 相比，DeepSeekMoE 16B 仅使用约 40% 的计算量便达到了卓越的性能。

### 5.1 实验设置

#### 5.1.1 训练数据和分词

我们从第 4.1.1 节描述的相同语料库中采样训练数据。与验证实验不同，我们采样了更大量的 2T tokens 数据，与 LLaMA2 7B 的训练 tokens 数量对齐。我们也使用 HuggingFace Tokenizer 工具训练 BPE 分词器，但 DeepSeekMoE 16B 的词表大小设为 100K。

#### 5.1.2 超参数

##### 模型设置

对于 DeepSeekMoE 16B，我们将 Transformer 层数设为 28，隐藏维度设为 2048。我们采用多头注意力机制，总共 16 个注意力头，每个头的维度为 128。对于初始化，所有可学习参数均以 0.006 的标准差随机初始化。我们将除第一层外的所有 FFN 替换为 MoE 层，因为我们观察到第一层的负载均衡状态收敛特别慢。每个 MoE 层由 2 个共享专家和 64 个路由专家组成，每个专家是标准 FFN 大小的 0.25 倍。每个 token 将被路由到这 2 个共享专家和 64 个路由专家中的 6 个。由于专家尺寸过小可能降低计算效率，我们没有采用更细的专家切分粒度。在 16B 以上的更大规模上，仍可以采用更细的粒度。在我们的配置下，DeepSeekMoE 16B 大约有 16.4B 总参数，激活参数约为 2.8B。

##### 训练设置

我们采用 AdamW 优化器（Loshchilov 和 Hutter，2019），超参数设置为 β₁=0.9，β₂=0.95，weight_decay=0.1。学习率同样采用 warmup 加阶梯衰减策略进行调度。最初，学习率在前 2K 步内从 0 线性增加到最大值。随后，在训练步数的 80% 时学习率乘以 0.316，在 90% 时再次乘以 0.316。DeepSeekMoE 16B 的最大学习率设为 4.2×10⁻⁴，梯度裁剪范数设为 1.0。批量大小设为 4.5K，最大序列长度为 4K，每个训练 batch 包含 18M tokens。相应地，总训练步数设为 106,449，以达到 2T 训练 tokens。由于训练数据充足，训练过程中不使用 dropout。我们利用流水线并行将模型的不同层部署在不同设备上，对于每一层，所有专家将部署在同一设备上。因此，我们在训练时也不丢弃任何 token，也不采用设备级均衡损失。为防止路由崩溃，我们将专家级均衡因子设得相当小，为 0.001，因为我们发现在我们的并行化策略下，较高的专家级均衡因子无法提高计算效率，反而会损害模型性能。

#### 5.1.3 评估基准

除了验证实验中使用的基准，我们还纳入了额外的基准以进行更全面的评估。我们与验证实验中使用的基准的区别如下。

##### 语言建模

对于语言建模，我们同样在 Pile（Gao 等人，2020）的测试集上评估模型。由于 DeepSeekMoE 16B 中使用的分词器与 LLaMA2 7B 使用的不同。为公平对比，我们使用 BPB（每字节比特数）作为评估指标。

##### 阅读理解

对于阅读理解，我们另外考虑 DROP（Dua 等人，2019）。评估指标为完全匹配（EM）率。

##### 数学推理

对于数学推理，我们另外纳入 GSM8K（Cobbe 等人，2021）和 MATH（Hendrycks 等人，2021），使用 EM 作为评估指标。

##### 多学科多项选择

对于多学科多项选择，我们另外在 MMLU（Hendrycks 等人，2020）上评估模型。评估指标为准确率。

##### 消歧

对于消歧，我们另外考虑 WinoGrande（Sakaguchi 等人，2019），评估指标为准确率。

##### 中文基准

由于 DeepSeekMoE 16B 在双语语料上预训练，我们也在四个中文基准上评估它。CLUEWSC（Xu 等人，2020）是一个中文消歧基准。CEval（Huang 等人，2023）和 CMMLU（Li 等人，2023）是两个与 MMLU 形式相似的中文多学科多项选择基准。CHID（Zheng 等人，2019）是一个中文成语填空基准，旨在评估对中国文化的理解。上述中文基准的评估指标为准确率或 EM。

##### Open LLM Leaderboard

我们基于内部评估框架评估上述所有基准。为了公平且方便地将 DeepSeekMoE 16B 与开源模型对比，我们另外在 Open LLM Leaderboard 上评估 DeepSeekMoE 16B。Open LLM Leaderboard 是一个由 HuggingFace 支持的公开排行榜，由六个任务组成：ARC（Clark 等人，2018）、HellaSwag（Zellers 等人，2019）、MMLU（Hendrycks 等人，2020）、TruthfulQA（Lin 等人，2022）、Winogrande（Sakaguchi 等人，2019）和 GSM8K（Cobbe 等人，2021）。

### 5.2 评估

| 指标 | # Shot | DeepSeek 7B (Dense) | DeepSeekMoE 16B |
|---|---|---|---|
| 总参数量 | N/A | 6.9B | 16.4B |
| 激活参数量 | N/A | 6.9B | 2.8B |
| 每 4K Tokens 的 FLOPs | N/A | 183.5T | 74.4T |
| 训练 Tokens 数 | N/A | 2T | 2T |
| Pile (BPB) | N/A | 0.75 | **0.74** |
| HellaSwag (准确率) | 0-shot | 75.4 | **77.1** |
| PIQA (准确率) | 0-shot | 79.2 | **80.2** |
| ARC-easy (准确率) | 0-shot | 67.9 | **68.1** |
| ARC-challenge (准确率) | 0-shot | 48.1 | **49.8** |
| RACE-middle (准确率) | 5-shot | **63.2** | 61.9 |
| RACE-high (准确率) | 5-shot | **46.5** | 46.4 |
| DROP (EM) | 1-shot | **34.9** | 32.9 |
| GSM8K (EM) | 8-shot | 17.4 | **18.8** |
| MATH (EM) | 4-shot | 3.3 | **4.3** |
| HumanEval (Pass@1) | 0-shot | 26.2 | **26.8** |
| MBPP (Pass@1) | 3-shot | 39.0 | **39.2** |
| TriviaQA (EM) | 5-shot | 59.7 | **64.8** |
| NaturalQuestions (EM) | 5-shot | 22.2 | **25.5** |
| MMLU (准确率) | 5-shot | **48.2** | 45.0 |
| WinoGrande (准确率) | 0-shot | **70.5** | 70.2 |
| CLUEWSC (EM) | 5-shot | **73.1** | 72.1 |
| CEval (准确率) | 5-shot | **45.0** | 40.6 |
| CMMLU (准确率) | 5-shot | **47.2** | 42.5 |
| CHID (准确率) | 0-shot | 89.3 | **89.4** |

**表 3**：DeepSeek 7B 和 DeepSeekMoE 16B 的对比。粗体表示最佳或接近最佳。仅使用 40.5% 的计算量，DeepSeekMoE 16B 达到了与 DeepSeek 7B 相当的性能。

#### 5.2.1 与 DeepSeek 7B 的内部对比

我们首先在 DeepSeekMoE 16B 与 DeepSeek 7B（DeepSeek-AI，2024）（一个 6.9B 参数的稠密语言模型）之间进行内部对比。为确保公平，两个模型都在相同的 2T tokens 语料库上训练。这使我们能够准确评估我们 MoE 架构的有效性，独立于训练数据的影响。

评估结果如表 3 所示，得出以下观察：(1) 总体而言，DeepSeekMoE 16B 仅使用约 40% 的计算量便达到了与 DeepSeek 7B 相当的性能。(2) DeepSeekMoE 16B 在语言建模和知识密集型任务（如 Pile、HellaSwag、TriviaQA 和 NaturalQuestions）上展现出显著优势。鉴于在 MoE 模型中，FFN 参数远多于注意力参数，这些结果与 Transformer 中的 FFN 表现出知识记忆能力的论断相符（Dai 等人，2022a）。(3) 与其他任务上的优异表现相比，DeepSeekMoE 在解决多项选择任务方面表现出局限性。这种不足源于 DeepSeekMoE 16B 中有限的注意力参数（DeepSeekMoE 16B 只有约 0.5B 注意力参数，而 DeepSeek 7B 有 2.5B 注意力参数）。我们之前对 DeepSeek 7B 的研究揭示了注意力容量与多项选择任务性能之间的正相关性。例如，配备多查询注意力机制（Shazeer，2019）的 DeepSeek 7B MQA 在 MMLU 类任务中也表现不佳。此外，为了更全面地理解 DeepSeekMoE 16B 的训练过程，我们也在附录 C 中提供了 DeepSeekMoE 16B 和 DeepSeek 7B（Dense）训练期间的基准曲线供参考。

至关重要的是，由于 DeepSeekMoE 16B 参数数量适中，它能够在单个 40GB 内存的 GPU 上进行单设备部署。通过适当的算子优化，它可以达到 7B 稠密模型近 2.5 倍的推理速度。

| 指标 | # Shot | LLaMA2 7B | DeepSeekMoE 16B |
|---|---|---|---|
| 总参数量 | N/A | 6.7B | 16.4B |
| 激活参数量 | N/A | 6.7B | 2.8B |
| 每 4K Tokens 的 FLOPs | N/A | 187.9T | 74.4T |
| 训练 Tokens 数 | N/A | 2T | 2T |
| Pile (BPB) | N/A | 0.76 | **0.74** |
| HellaSwag (准确率) | 0-shot | 75.6 | **77.1** |
| PIQA (准确率) | 0-shot | 78.0 | **80.2** |
| ARC-easy (准确率) | 0-shot | **69.1** | 68.1 |
| ARC-challenge (准确率) | 0-shot | 49.0 | **49.8** |
| RACE-middle (准确率) | 5-shot | 60.7 | **61.9** |
| RACE-high (准确率) | 5-shot | 45.8 | **46.4** |
| DROP (EM) | 1-shot | **34.0** | 32.9 |
| GSM8K (EM) | 8-shot | 15.5 | **18.8** |
| MATH (EM) | 4-shot | 2.6 | **4.3** |
| HumanEval (Pass@1) | 0-shot | 14.6 | **26.8** |
| MBPP (Pass@1) | 3-shot | 21.8 | **39.2** |
| TriviaQA (EM) | 5-shot | 63.8 | **64.8** |
| NaturalQuestions (EM) | 5-shot | 25.5 | 25.5 |
| MMLU (准确率) | 5-shot | **45.8** | 45.0 |
| WinoGrande (准确率) | 0-shot | 69.6 | **70.2** |
| CLUEWSC (EM) | 5-shot | 64.0 | **72.1** |
| CEval (准确率) | 5-shot | 33.9 | **40.6** |
| CMMLU (准确率) | 5-shot | 32.6 | **42.5** |
| CHID (准确率) | 0-shot | 37.9 | **89.4** |

**表 4**：LLaMA2 7B 和 DeepSeekMoE 16B 的对比。仅使用 39.6% 的计算量，DeepSeekMoE 16B 在大多数基准上超越了 LLaMA2 7B。

#### 5.2.2 与开源模型的对比

##### 与 LLaMA2 7B 的内部对比

在开源模型领域，我们主要将 DeepSeekMoE 16B 与 LLaMA2 7B（Touvron 等人，2023b）进行对比，后者是一个著名且强大的开源语言模型，拥有 6.7B 参数。DeepSeekMoE 16B 和 LLaMA2 7B 都在 2T tokens 上预训练。与 LLaMA2 7B 相比，DeepSeekMoE 拥有 245% 的总参数，但仅需 39.6% 的计算量。我们内部基准的结果如表 4 所示，得出以下观察：(1) 在所评估的基准中，仅使用约 40% 的计算量，DeepSeekMoE 16B 在大多数基准上超越了 LLaMA2 7B。(2) DeepSeekMoE 16B 的数学推理和代码生成能力强于 LLaMA2 7B，这归因于我们预训练语料中数学和代码相关文本的丰富存在。(3) 由于我们的预训练语料中存在中文文本，DeepSeekMoE 16B 在中文基准上对 LLaMA2 7B 表现出实质性的性能优势。(4) 尽管在更少的英文文本上训练，DeepSeekMoE 16B 在英文理解或知识密集型基准上的表现与 LLaMA2 7B 相当甚至更好，这证明了 DeepSeekMoE 16B 的卓越能力。

##### Open LLM Leaderboard 上的评估

除了内部评估外，我们也在 Open LLM Leaderboard 上评估 DeepSeekMoE 16B，并与其他开源模型进行对比。除了 LLaMA2 7B，我们还考虑了更广泛的开源模型，包括 LLaMA 7B（Touvron 等人，2023a）、Falcon 7B（Almazrouei 等人，2023）、GPT-J 6B（Wang 和 Komatsuzaki，2021）、RedPajama-INCITE 7B 和 3B（Together-AI，2023）、Open LLaMA 7B 和 3B（Geng 和 Liu，2023）、OPT 2.7B（Zhang 等人，2022）、Pythia 2.8B（Biderman 等人，2023）、GPT-neo 2.7B（Black 等人，2021）和 BLOOM 3B（Scao 等人，2022）。如图 1 所示，评估结果显示 DeepSeekMoE 16B 大幅超越了具有相似激活参数的模型。此外，它达到了与 LLaMA2 7B 相当的性能，而后者的激活参数约为前者的 2.5 倍。

---

## 6 DeepSeekMoE 16B 的对齐

先前的研究表明，MoE 模型通常不会从微调中获得显著收益（Fedus 等人，2021；Artetxe 等人，2022）。然而，Shen 等人（2023）的发现表明 MoE 模型确实可以从指令微调中受益。为了评估 DeepSeekMoE 16B 是否可以从微调中受益，我们进行监督微调以构建基于 DeepSeekMoE 16B 的聊天模型。实验结果表明 DeepSeekMoE Chat 16B 也达到了与 LLaMA2 SFT 7B 和 DeepSeek Chat 7B 相当的性能。

### 6.1 实验设置

#### 训练数据

为训练聊天模型，我们在内部精心整理的数据上进行监督微调（SFT），包含 1.4M 训练样本。该数据集涵盖广泛的类别，包括数学、代码、写作、问答、推理、总结等。我们的 SFT 训练数据大多为英文和中文，使聊天模型多样化并适用于双语场景。

#### 超参数

在监督微调期间，我们将批量大小设为 1024 个样本，使用 AdamW 优化器（Loshchilov 和 Hutter，2019）进行 8 个 epoch 的训练。我们采用 4K 的最大序列长度，并将训练样本尽可能密集地打包直到达到序列长度限制。监督微调期间不使用 dropout，仅设置 10⁻⁵ 的恒定学习率，不采用任何学习率调度策略。

#### 评估基准

对于聊天模型的评估，我们采用与第 5.1.3 节中相似的基准，但进行以下调整：(1) 我们排除 Pile（Gao 等人，2020），因为聊天模型很少用于纯语言建模。(2) 我们排除 CHID（Zheng 等人，2019），由于观察到的结果不稳定，阻碍了得出可靠结论。(3) 我们额外纳入 BBH（Suzgun 等人，2022），以提供对聊天模型推理能力的更全面评估。

| 指标 | # Shot | LLaMA2 SFT 7B | DeepSeek Chat 7B | DeepSeekMoE Chat 16B |
|---|---|---|---|---|
| 总参数量 | N/A | 6.7B | 6.9B | 16.4B |
| 激活参数量 | N/A | 6.7B | 6.9B | 2.8B |
| 每 4K Tokens 的 FLOPs | N/A | 187.9T | 183.5T | 74.4T |
| HellaSwag (准确率) | 0-shot | 67.9 | 71.0 | **72.2** |
| PIQA (准确率) | 0-shot | 76.9 | 78.4 | **79.7** |
| ARC-easy (准确率) | 0-shot | 69.7 | **70.2** | 69.9 |
| ARC-challenge (准确率) | 0-shot | **50.8** | 50.2 | 50.0 |
| BBH (EM) | 3-shot | 39.3 | **43.1** | 42.2 |
| RACE-middle (准确率) | 5-shot | 63.9 | **66.1** | 64.8 |
| RACE-high (准确率) | 5-shot | 49.6 | **50.8** | 50.6 |
| DROP (EM) | 1-shot | 40.0 | **41.7** | 33.8 |
| GSM8K (EM) | 0-shot | **63.4** | 62.6 | 62.2 |
| MATH (EM) | 4-shot | 13.5 | 14.7 | **15.2** |
| HumanEval (Pass@1) | 0-shot | 35.4 | 45.1 | **45.7** |
| MBPP (Pass@1) | 3-shot | 27.8 | 39.0 | **46.2** |
| TriviaQA (EM) | 5-shot | 60.1 | 59.5 | **63.3** |
| NaturalQuestions (EM) | 0-shot | **35.2** | 32.7 | 35.1 |
| MMLU (准确率) | 0-shot | **50.0** | 49.7 | 47.2 |
| WinoGrande (准确率) | 0-shot | 65.1 | 68.4 | **69.0** |
| CLUEWSC (EM) | 5-shot | 48.4 | 66.2 | **68.2** |
| CEval (准确率) | 0-shot | 35.1 | **44.7** | 40.0 |
| CMMLU (准确率) | 0-shot | 36.9 | **51.2** | 49.3 |

**表 5**：LLaMA2 SFT 7B、DeepSeek Chat 7B 和 DeepSeekMoE Chat 16B 的对比，这三个模型在相同的 SFT 数据上进行微调。与两个 7B 稠密模型相比，DeepSeekMoE Chat 16B 仅使用 40% 的计算量仍在大多数基准上达到相当或更好的性能。

### 6.2 评估

#### 基线

为验证 DeepSeekMoE 16B 在对齐后的潜力，我们对 LLaMA2 7B、DeepSeek 7B 和 DeepSeekMoE 16B 进行监督微调，使用完全相同的微调数据以确保公平。相应地，我们构建了三个聊天模型，包括 LLaMA2 SFT 7B³、DeepSeek Chat 7B 和 DeepSeekMoE Chat 16B。随后，我们在广泛的下游任务上将 DeepSeekMoE Chat 16B 与其他两个稠密聊天模型（约 2.5 倍 FLOPs）进行对比。

³ 我们使用 LLaMA2 SFT 以区别于官方的 LLaMA2 Chat（Touvron 等人，2023b）模型。

#### 结果

评估结果如表 5 所示。我们的主要观察包括：(1) DeepSeekMoE Chat 16B 在消耗近 40% 计算量的情况下，在语言理解和推理（PIQA、ARC、BBH）、机器阅读理解（RACE）、数学（GSM8K、MATH）和知识密集型任务（TriviaQA、NaturalQuestions）上达到了与 7B 稠密模型相当的性能。(2) 在代码生成任务上，DeepSeekMoE Chat 16B 显著超越 LLaMA2 SFT 7B，在 HumanEval 和 MBPP 上有显著改进。此外，它也超越了 DeepSeek Chat 7B。(3) 在包括 MMLU、CEval 和 CMMLU 在内的多项选择问答基准上，DeepSeekMoE Chat 16B 仍落后于 DeepSeek Chat 7B，与基础模型的观察结果一致（第 5.2.1 节）。然而值得注意的是，监督微调后，DeepSeekMoE 16B 和 DeepSeek 7B 之间的性能差距有所缩小。(4) 得益于双语语料的预训练，DeepSeekMoE Chat 16B 在所有中文基准上明显超越 LLaMA2 SFT 7B。这些结果证明了 DeepSeekMoE 16B 在中英文上的均衡能力，增强了其在多样化场景下的多功能性和适用性。总之，聊天模型的评估突显了 DeepSeekMoE 16B 从对齐中受益的潜力，并验证了其在仅使用约 40% 计算量的情况下达到与稠密模型相当性能的持续优势。

---

## 7 DeepSeekMoE 145B 进行中

受 DeepSeekMoE 16B 卓越性能的鼓舞，我们进一步进行将 DeepSeekMoE 扩展到 145B 的初步尝试。在这项初步研究中，DeepSeekMoE 145B 在 245B tokens 上训练，但它已经展示出相对于 GShard 架构的持续优势，并显示出匹配或超越 DeepSeek 67B（Dense）性能的潜力。此外，在 DeepSeekMoE 145B 的最终版本和全量训练完成后，我们也计划将其公开发布。

### 7.1 实验设置

#### 训练数据和分词

对于 DeepSeekMoE 145B，我们采用与 DeepSeekMoE 16B 完全相同的训练语料和分词器，唯一的区别是 DeepSeekMoE 145B 在 245B tokens 上训练用于初步研究。

#### 模型设置

对于 DeepSeekMoE 145B，我们将 Transformer 层数设为 62，隐藏维度设为 4096。我们采用多头注意力机制，总共 32 个注意力头，每个头的维度为 128。对于初始化，所有可学习参数均以 0.006 的标准差随机初始化。与 DeepSeekMoE 16B 一样，我们也将除第一层外的所有 FFN 替换为 MoE 层。每个 MoE 层由 4 个共享专家和 128 个路由专家组成，每个专家是标准 FFN 大小的 0.125 倍。每个 token 将被路由到这 4 个共享专家和 128 个路由专家中的 12 个。在此配置下，DeepSeekMoE 145B 大约有 144.6B 总参数，激活参数约为 22.2B。

#### 训练设置

我们采用 AdamW 优化器（Loshchilov 和 Hutter，2019），超参数设置为 β₁=0.9，β₂=0.95，weight_decay=0.1。对于 DeepSeekMoE 145B 的初步研究，我们采用 warmup 加恒定学习率调度。最初，学习率在前 2K 步内从 0 线性增加到最大值。随后，学习率在剩余训练过程中保持恒定。DeepSeekMoE 145B 的最大学习率设为 3.0×10⁻⁴，梯度裁剪范数设为 1.0。批量大小设为 4.5K，最大序列长度为 4K，每个训练 batch 包含 18M tokens。我们对 DeepSeekMoE 145B 训练 13,000 步，达到 245B 训练 tokens。同样地，训练过程中不使用 dropout。我们利用流水线并行将模型的不同层部署在不同设备上，对于每一层，所有路由专家将均匀部署在 4 个设备上（即专家并行结合数据并行）。由于我们对 DeepSeekMoE 145B 采用专家并行，应考虑设备级负载均衡以减少计算瓶颈。相应地，我们将设备级均衡因子设为 0.05 以鼓励跨设备的计算均衡。我们仍然设置一个较小的专家级均衡因子 0.003 以防止路由崩溃。

#### 评估基准

我们在与 DeepSeekMoE 16B 完全相同的内部基准上评估 DeepSeekMoE 145B（参见第 5.1.3 节）。

| 指标 | # Shot | DeepSeek 67B (Dense) | GShard 137B | DeepSeekMoE 145B | DeepSeekMoE 142B (半激活) |
|---|---|---|---|---|---|
| 总参数量 | N/A | 67.4B | 136.5B | 144.6B | 142.3B |
| 激活参数量 | N/A | 67.4B | 21.6B | 22.2B | 12.2B |
| 相对专家大小 | N/A | N/A | 1 | 0.125 | 0.125 |
| # 专家 | N/A | N/A | 0 + 16 | 4 + 128 | 2 + 128 |
| # 激活专家 | N/A | N/A | 0 + 2 | 4 + 12 | 2 + 6 |
| 每 4K Tokens 的 FLOPs | N/A | 2057.5T | 572.7T | 585.6T | 374.6T |
| 训练 Tokens 数 | N/A | 245B | 245B | 245B | 245B |
| Pile (损失) | N/A | 1.905 | 1.961 | **1.876** | 1.888 |
| HellaSwag (准确率) | 0-shot | 74.8 | 72.0 | **75.8** | 74.9 |
| PIQA (准确率) | 0-shot | 79.8 | 77.6 | **80.7** | 80.2 |
| ARC-easy (准确率) | 0-shot | 69.0 | 64.0 | **69.7** | 67.9 |
| ARC-challenge (准确率) | 0-shot | **50.4** | 45.8 | 48.8 | 49.0 |
| RACE-middle (准确率) | 5-shot | **63.2** | 59.2 | 62.1 | 59.5 |
| RACE-high (准确率) | 5-shot | **46.9** | 43.5 | 45.5 | 42.6 |
| DROP (EM) | 1-shot | 27.5 | 21.6 | **27.8** | 28.9 |
| GSM8K (EM) | 8-shot | 11.8 | 6.4 | **12.2** | 13.8 |
| MATH (EM) | 4-shot | 2.1 | 1.6 | **3.1** | 2.8 |
| HumanEval (Pass@1) | 0-shot | **23.8** | 17.7 | 19.5 | 23.2 |
| MBPP (Pass@1) | 3-shot | **33.6** | 27.6 | 33.2 | 32.0 |
| TriviaQA (EM) | 5-shot | 57.2 | 52.5 | **61.1** | 59.8 |
| NaturalQuestions (EM) | 5-shot | 22.6 | 19.0 | **25.0** | 23.5 |
| MMLU (准确率) | 5-shot | **45.1** | 26.3 | 39.4 | 37.5 |
| WinoGrande (准确率) | 0-shot | 70.7 | 67.6 | **71.9** | 70.8 |
| CLUEWSC (EM) | 5-shot | 69.1 | 65.7 | **71.9** | 72.6 |
| CEval (准确率) | 5-shot | **40.3** | 26.2 | 37.1 | 32.8 |
| CMMLU (准确率) | 5-shot | **40.6** | 25.4 | 35.9 | 31.9 |
| CHID (准确率) | 0-shot | 88.5 | 86.9 | **90.3** | 88.3 |

**表 6**：DeepSeek 67B（Dense）和约 140B 总参数规模的 MoE 模型间的对比。在"# 专家"和"# 激活专家"行中，a + b 分别表示 a 个共享专家和 b 个路由专家。粗体表示最佳或接近最佳的性能（不包括最后一列）。DeepSeekMoE 145B，甚至仅有一半激活专家参数的 DeepSeekMoE 142B（半激活），都大幅超越 GShard 137B。此外，仅使用 28.5% 的计算量，DeepSeekMoE 145B 达到了与 DeepSeek 67B 相当的性能。

### 7.2 评估

#### 基线

除 DeepSeekMoE 145B 之外，我们考虑了三个额外模型进行对比。**DeepSeek 67B（Dense）** 是一个总参数 67.4B 的稠密模型（关于模型和训练细节参考 DeepSeek-AI（2024））。**GShard 137B** 与 DeepSeekMoE 145B 共享相同的隐藏维度和层数，但遵循 GShard 架构。注意 DeepSeekMoE 145B 将每个专家的中间隐藏维度对齐到 64 的倍数以提高计算效率，因此其模型大小比 GShard 137B 大 6%。**DeepSeekMoE 142B（半激活）** 与 DeepSeekMoE 145B 架构相似，但仅包含 2 个共享专家，且 128 个路由专家中仅 6 个被激活。值得注意的是，所有对比模型（包括 DeepSeekMoE 145B）共享相同的训练语料。此外，对比中所有 MoE 模型都是从头开始训练，并共享相同的训练超参数。

#### 结果

从表 6 所示的评估结果中，我们有以下观察：(1) 尽管总参数和计算量相当，DeepSeekMoE 145B 显著超越 GShard 137B，再次突显了 DeepSeekMoE 架构的优势。(2) 总体而言，仅使用 28.5% 的计算量，DeepSeekMoE 145B 便达到了与 DeepSeek 67B（Dense）相当的性能。与 DeepSeekMoE 16B 的发现一致，DeepSeekMoE 145B 在语言建模和知识密集型任务上展现出显著优势，但在多项选择任务上存在局限。(3) 在更大规模下，DeepSeekMoE 142B（半激活）的性能与 DeepSeekMoE 145B 相比落后不多。此外，尽管仅有一半的激活专家参数，DeepSeekMoE 142B（半激活）仍能匹配 DeepSeek 67B（Dense）的性能，仅使用 18.2% 的计算量。它也超越了 GShard 137B，这与第 4.5 节的结论一致。

---

## 8 相关工作

混合专家（MoE）技术最早由 Jacobs 等人（1991）和 Jordan 和 Jacobs（1994）提出，用于以独立的专家模块处理不同样本。Shazeer 等人（2017）将 MoE 引入语言模型训练，构建了基于 LSTM（Hochreiter 和 Schmidhuber，1997）的大规模 MoE 模型。随着 Transformer 成为 NLP 中最流行的架构，许多尝试将 Transformer 中的 FFN 扩展为 MoE 层以构建 MoE 语言模型。GShard（Lepikhin 等人，2021）和 Switch Transformer（Fedus 等人，2021）是采用可学习的 top-2 或 top-1 路由策略将 MoE 语言模型扩展到极大规模的先驱。Hash Layer（Roller 等人，2021）和 StableMoE（Dai 等人，2022b）使用固定路由策略以实现更稳定的路由和训练。Zhou 等人（2022）提出了一种专家选择路由策略，其中每个 token 可以被分配给不同数量的专家。Zoph（2022）关注 MoE 模型中训练不稳定和微调困难的问题，提出 ST-MoE 来克服这些挑战。除了对 MoE 架构和训练策略的研究之外，近年来也出现了许多基于现有 MoE 架构的大规模语言或多模态模型（Lin 等人，2021；Du 等人，2022；Ren 等人，2023；Xue 等人，2023）。总的来说，先前大多数 MoE 模型都基于传统的 top-1 或 top-2 路由策略，在改进专家特化方面留有较大空间。作为回应，我们的 DeepSeekMoE 架构旨在最大程度地改进专家特化。

---

## 9 结论

在本文中，我们引入了用于 MoE 语言模型的 DeepSeekMoE 架构，目标是实现终极的专家特化。通过细粒度专家切分和共享专家隔离，DeepSeekMoE 与流行的 MoE 架构相比，实现了显著更高的专家特化和性能。从 2B 参数的中等规模开始，我们验证了 DeepSeekMoE 的优势，证明其能力接近 MoE 模型的上限性能。此外，我们提供实证证据表明 DeepSeekMoE 比 GShard 具有更高水平的专家特化。

扩展到 16B 总参数的更大规模，我们在 2T tokens 上训练 DeepSeekMoE 16B，并展示其在仅使用约 40% 计算量的情况下，达到了与 DeepSeek 7B 和 LLaMA2 7B 相当的卓越性能。此外，进行了监督微调用于对齐，构建了基于 DeepSeekMoE 16B 的 MoE 聊天模型，进一步显示其适应性和多功能性。进而，我们对将 DeepSeekMoE 扩展到 145B 参数进行了初步探索。我们发现 DeepSeekMoE 145B 仍然保持相对于 GShard 架构的实质性优势，并展示出与 DeepSeek 67B 相当的性能，仅使用 28.5%（甚至可能仅 18.2%）的计算量。

出于研究目的，我们向公众发布 DeepSeekMoE 16B 的模型检查点，该模型可以部署在单个 40GB 内存的 GPU 上。我们希望这项工作能够为学术界和工业界提供有价值的洞察，并为大规模语言模型的加速发展做出贡献。

---

## 附录 A 超参数概览

我们在表 7 中提供了不同规模 DeepSeekMoE 的超参数概览。

| # 参数 | # 层数 | 隐藏维度 | # 注意力头 | # 共享专家 | # 路由专家 | 相对专家大小 | 序列长度 | 批量大小 (序列) | 学习率 |
|---|---|---|---|---|---|---|---|---|---|
| 2.0B | 9 | 1280 | 10 | 1 | 63 (激活 7) | 0.25 | 2048 | 2048 | 1.08e-3 |
| 16.4B | 28 | 2048 | 16 | 2 | 64 (激活 6) | 0.25 | 4096 | 4608 | 4.2e-4 |
| 144.6B | 62 | 4096 | 32 | 4 | 128 (激活 12) | 0.125 | 4096 | 4608 | 3.0e-4 |

**表 7**：DeepSeekMoE 不同规模超参数概览。相对专家大小是相对于标准 FFN 而言。

---

## 附录 B 将 DeepSeekMoE 与更大模型对比

DeepSeekMoE、GShard×1.2 和 GShard×1.5 之间的对比如表 8 所示。DeepSeekMoE、Dense×4 和 Dense×16 之间的对比如表 9 所示。

| 指标 | # Shot | GShard×1.2 | GShard×1.5 | DeepSeekMoE |
|---|---|---|---|---|
| 相对专家大小 | N/A | 1.2 | 1.5 | 0.25 |
| # 专家 | N/A | 0 + 16 | 0 + 16 | 1 + 63 |
| # 激活专家 | N/A | 0 + 2 | 0 + 2 | 1 + 7 |
| # 总专家参数 | N/A | 2.3B | 2.8B | 1.9B |
| # 激活专家参数 | N/A | 0.28B | 0.35B | 0.24B |
| 训练 Tokens 数 | N/A | 100B | 100B | 100B |
| Pile (损失) | N/A | 1.824 | 1.808 | 1.808 |
| HellaSwag (准确率) | 0-shot | 53.7 | 54.4 | 54.8 |
| PIQA (准确率) | 0-shot | 71.8 | 71.1 | 72.3 |
| ARC-easy (准确率) | 0-shot | 46.8 | 47.3 | 49.4 |
| ARC-challenge (准确率) | 0-shot | 31.7 | 34.1 | 34.3 |
| RACE-middle (准确率) | 5-shot | 43.7 | 46.4 | 44.0 |
| RACE-high (准确率) | 5-shot | 31.9 | 32.4 | 31.7 |
| HumanEval (Pass@1) | 0-shot | 3.7 | 3.0 | 4.9 |
| MBPP (Pass@1) | 3-shot | 2.4 | 2.6 | 2.2 |
| TriviaQA (EM) | 5-shot | 15.2 | 15.7 | 16.6 |
| NaturalQuestions (EM) | 5-shot | 4.5 | 4.7 | 5.7 |

**表 8**：DeepSeekMoE 与更大 GShard 模型的对比。

| 指标 | # Shot | Dense×4 | Dense×16 | DeepSeekMoE |
|---|---|---|---|---|
| 相对专家大小 | N/A | 1 | 1 | 0.25 |
| # 专家 | N/A | 4 + 0 | 16 + 0 | 1 + 63 |
| # 激活专家 | N/A | 4 + 0 | 16 + 0 | 1 + 7 |
| # 总专家参数 | N/A | 0.47B | 1.89B | 1.89B |
| # 激活专家参数 | N/A | 0.47B | 1.89B | 0.24B |
| 训练 Tokens 数 | N/A | 100B | 100B | 100B |
| Pile (损失) | N/A | 1.908 | 1.806 | 1.808 |
| HellaSwag (准确率) | 0-shot | 47.6 | 55.1 | 54.8 |
| PIQA (准确率) | 0-shot | 70.0 | 71.9 | 72.3 |
| ARC-easy (准确率) | 0-shot | 43.9 | 51.9 | 49.4 |
| ARC-challenge (准确率) | 0-shot | 30.5 | 33.8 | 34.3 |
| RACE-middle (准确率) | 5-shot | 42.4 | 46.3 | 44.0 |
| RACE-high (准确率) | 5-shot | 30.7 | 33.0 | 31.7 |
| HumanEval (Pass@1) | 0-shot | 1.8 | 4.3 | 4.9 |
| MBPP (Pass@1) | 3-shot | 0.2 | 2.2 | 2.2 |
| TriviaQA (EM) | 5-shot | 9.9 | 16.5 | 16.6 |
| NaturalQuestions (EM) | 5-shot | 3.0 | 6.3 | 5.7 |

**表 9**：DeepSeekMoE 与更大稠密基线的对比。

在 13B 总参数的更大规模下，我们也将 DeepSeekMoE 与 GShard×1.2 和 GShard×1.5 进行了对比，结果如表 10 所示。在更大规模下，DeepSeekMoE 甚至明显超越 GShard×1.5。

| 指标 | # Shot | GShard×1.2 | GShard×1.5 | DeepSeekMoE |
|---|---|---|---|---|
| 相对专家大小 | N/A | 1.2 | 1.5 | 0.25 |
| # 专家 | N/A | 0 + 16 | 0 + 16 | 1 + 63 |
| # 激活专家 | N/A | 0 + 2 | 0 + 2 | 1 + 7 |
| # 总专家参数 | N/A | 15.9B | 19.8B | 13.3B |
| # 激活专家参数 | N/A | 2.37B | 2.82B | 2.05B |
| 训练 Tokens 数 | N/A | 100B | 100B | 100B |
| HellaSwag (准确率) | 0-shot | 66.6 | 67.7 | 69.1 |
| PIQA (准确率) | 0-shot | 75.6 | 76.0 | 75.7 |
| ARC-easy (准确率) | 0-shot | 56.8 | 56.8 | 58.8 |
| ARC-challenge (准确率) | 0-shot | 39.9 | 37.6 | 38.5 |
| RACE-middle (准确率) | 5-shot | 51.6 | 50.6 | 52.4 |
| RACE-high (准确率) | 5-shot | 37.4 | 36.3 | 38.5 |
| HumanEval (Pass@1) | 0-shot | 6.1 | 6.1 | 9.8 |
| MBPP (Pass@1) | 3-shot | 7.0 | 11.6 | 10.6 |
| TriviaQA (EM) | 5-shot | 36.5 | 36.7 | 38.2 |
| NaturalQuestions (EM) | 5-shot | 12.6 | 12.1 | 13.7 |

**表 10**：DeepSeekMoE 与更大 GShard 模型在更大规模下的对比。

---

## 附录 C DeepSeekMoE 16B 的训练基准曲线

我们在图 7 中提供了 DeepSeekMoE 16B 和 DeepSeek 7B（Dense）训练期间的基准曲线供参考。

![Refer to caption](https://arxiv.org/html/2401.06066v1/x7.png)

**图 7**：DeepSeekMoE 16B 和 DeepSeek 7B（Dense）训练期间的基准曲线。

---

## 参考文献

完整参考文献请参阅原始论文：<https://arxiv.org/abs/2401.06066>

主要引用包括：
- Almazrouei et al. (2023) - Falcon-40B
- Brown et al. (2020) - GPT-3
- Clark et al. (2018) - ARC
- DeepSeek-AI (2024) - DeepSeek LLM
- Fedus et al. (2021) - Switch Transformers
- Hendrycks et al. (2020) - MMLU
- Jacobs et al. (1991) - 自适应局部专家混合
- Lepikhin et al. (2021) - GShard
- Loshchilov and Hutter (2019) - AdamW
- OpenAI (2023) - GPT-4
- Rajbhandari et al. (2022) - DeepSpeed-MoE
- Shazeer et al. (2017) - 稀疏门控混合专家层
- Touvron et al. (2023a, 2023b) - LLaMA, LLaMA 2
- Vaswani et al. (2017) - Transformer
- Zellers et al. (2019) - HellaSwag

---

*翻译说明：本文为 arXiv:2401.06066 论文 "DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models" 的中文翻译。原论文采用 arXiv.org 非独占性永久许可证发布。图片链接均保留指向 arXiv 原始资源。*
