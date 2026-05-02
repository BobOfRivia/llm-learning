# 架构演进

## 一句话定义

这条轨道覆盖 LLM 的模型架构选择——从基础的 Transformer 到各种变体、替代方案和混合架构，追踪"用什么结构来建模序列"这个核心问题的演进。

## 演进脉络

### 阶段 1：Transformer 统一与路线分化（2017-2020）

Vaswani et al. 2017 年的 Attention Is All You Need 之后，Transformer 迅速统一了 NLP，但内部很快分化为三条路线：encoder-only（BERT, 2018，双向掩码语言建模）、decoder-only（GPT, 2018，自回归下一 token 预测）、encoder-decoder（T5, 2020，seq2seq）。2018-2020 年间三条路线并行发展，各有适用场景：BERT 在分类任务上领先，T5 在需要结构化输入输出的任务上灵活，GPT 系列在开放式生成上更自然。

**decoder-only 最终确立主导地位的原因**并非单纯的技术优越性，而是规模化路径上的收益差异：（1）自回归预训练可以直接在任意文本语料上进行，无需构造 input-output 对，数据工程门槛极低；（2）通过 prompting 将所有任务统一为文本生成，不需要针对不同下游任务设计不同的 head 或 fine-tuning 范式；（3）Scaling Laws 在 decoder-only 架构上表现更稳定，生成能力随参数量持续提升。

注：有研究指出，当给 encoder-decoder 模型同等计算量时，其 instruction-tuned 版本有时表现更好。decoder-only 的统治更多是 GPT 系列成功带来的路径依赖，而非架构层面的绝对优越。

### 阶段 2：规模驱动的架构稳定（2020-2022）

GPT-3（175B，2020 年）用 dense Transformer 靠规模获得涌现能力，验证了"相同架构做大"的投入逻辑。在 Scaling Laws 的框架下，架构创新的优先级让位于规模扩展——这个阶段的主旋律是把已经成熟的架构推向更大规模，而非设计新结构。

这个时期 dense Transformer 架构的核心参数决策（层数、头数、隐层维度的比例）基本固化。唯一显著的架构探索方向是 **MoE（稀疏激活）**：Switch Transformer（Fedus et al., Google Brain，2021 年 1 月，arXiv 2101.03961）将总参数推到 1.571T，但每个 token 只激活一个 expert，计算量保持常数。Switch Transformer 相比 T5-Base 实现了高达 7 倍的预训练速度提升，但训练不稳定性和负载均衡问题尚未被系统解决，距离生产级应用还有距离。

**参考**：[Attention Is All You Need](https://arxiv.org/abs/1706.03762)，[Switch Transformers](https://arxiv.org/abs/2101.03961)

### 阶段 3：效率压力下的架构分化（2022-2024）

这个阶段的核心驱动力不是学术竞争，而是**成本压力**。Dense Transformer 的参数量和训练/推理计算量是绑定的——想要更强的模型，就必须接受更高的计算成本。当模型规模推进到数百 B 参数时，这个代价开始在生产环境中无法忽视。三条并行的路线应运而生，各自从不同角度解耦"能力"和"成本"。

**路线一：MoE（稀疏激活）**

MoE 的核心洞见是**把总参数和活跃参数分开**。Dense 模型的每个 token 都要激活全部参数；MoE 把 FFN 层替换为多个"专家"网络，每个 token 只激活其中的少数几个（通常 1-2 个），由一个轻量的路由（router）网络决定。总参数量可以很大，但单次推理的计算量由活跃参数量决定。

从 Switch Transformer（阶段 2）的单 expert 激活，到 ST-MoE、GLaM 等对训练稳定性的改进，再到 **Mixtral 8x7B（Mistral AI，2023 年 12 月，arXiv 2401.04088）**，MoE 在这个阶段完成了从"可能可行"到"已经可用"的跨越。Mixtral 8x7B 的架构：8 个专家，每次激活 top-2，总参数约 47B，活跃参数约 13B。在大多数基准上超过 Llama 2 70B，而推理计算量与 13B dense 模型相当。这是开源生态中第一个被广泛验证的生产级 MoE 模型。

GPT-4 据称也采用了 MoE（8 个专家 × ~220B，top-2 激活），但 OpenAI 从未官方确认。如果属实，这说明即使资源最充足的实验室也在 2023 年选择了 MoE 而非继续 scale dense——这是一个强信号，表明 MoE 已经跨过了工程可靠性的阈值。

MoE 的主要工程挑战：**负载均衡**（如果所有 token 都选同一个 expert，其他 expert 的参数浪费）和**训练不稳定性**（路由网络和专家网络的联合优化容易陷入不好的局部解）。负载均衡通常通过辅助 loss（auxiliary loss）解决——在主训练目标上加一项惩罚 expert 使用不均匀的正则项。

**参考**：[Mixtral of Experts (arXiv 2401.04088)](https://arxiv.org/abs/2401.04088)

**路线二：高效 Attention 变体**

Attention 的计算复杂度是 O(n²)（n 为序列长度），内存占用随上下文增长呈平方级。这既限制了长上下文，也使 Attention 成为训练和推理的主要瓶颈。这个阶段涌现的几个关键改进在不降低模型质量的前提下，系统性地解决了这一问题。

**Flash Attention**（Dao et al., 斯坦福，2022 年 5 月，arXiv 2205.14135）是影响最广的单一工程改进之一。核心是 IO 感知（IO-aware）的重新设计：标准 Attention 实现会把 Q×K^T 的完整矩阵（N×N 大小）写入 GPU HBM（高带宽内存），这是真正的瓶颈；Flash Attention 将计算分块，在 SRAM（片上缓存）中完成分块计算，避免大矩阵的 HBM 读写。结果：计算结果完全相同（exact attention，不是近似），训练速度提升 2-4 倍，内存占用降低 5-20 倍。Flash Attention 2（2023 年）进一步优化了并行性和线程利用，再次加速约 2 倍。几乎所有 2023 年以后的主流模型训练都依赖 Flash Attention。

**GQA（Grouped Query Attention）**（Ainslie et al., Google，2023 年，arXiv 2305.13245）解决的是推理时的 KV cache 问题。标准 Multi-Head Attention（MHA）的每个注意力头都维护独立的 K 和 V，推理时 KV cache 随序列长度线性增长，长上下文时内存消耗巨大。Multi-Query Attention（MQA）将所有 Q 头共享同一组 K、V，大幅降低 cache 大小，但质量有损失。GQA 是两者的折中：将 Q 头分组，组内共享 K、V。实验显示 GQA 的质量接近 MHA，cache 大小与 MQA 相当。Llama 2、Llama 3、Mistral 等主流模型均采用 GQA。

**RoPE（旋转位置编码）**（Su et al., 2021/2022，arXiv 2104.09864）解决的是位置信息的表达问题。传统的绝对位置编码在训练时长度之外的位置无法泛化。RoPE 通过对 Q 和 K 向量施加旋转变换来编码相对位置，有两个关键优点：（1）天然支持位置外推，配合 YaRN、NTK-aware scaling 等技术可以将上下文扩展到训练长度的数倍；（2）旋转矩阵的内积只依赖两个位置的相对距离，而非绝对位置，更符合自然语言的实际结构。LLaMA 系列、Mistral、Qwen 等几乎所有主流开放模型都采用 RoPE。

**路线三：SSM（状态空间模型）**

Mamba（Gu & Dao，2023 年 12 月，arXiv 2312.00752）是 SSM 路线在这个阶段的代表性工作，对 S4 的核心改进是引入**选择性（selective）状态空间**：S4 的状态转移矩阵是固定的，Mamba 的矩阵参数由输入动态决定，使模型能够"选择性记忆"。Mamba 的计算复杂度是 O(n)（线性），理论上对超长序列有根本性优势。

但 Mamba 的实践位置在这个阶段变得清晰：它擅长需要长程依赖的顺序任务（基因组序列、音频），但在通用语言建模上，特别是需要精确回溯上下文的任务上，比 Transformer 差。原因在于 Mamba 的循环结构压缩了历史信息，无法像 Attention 那样精确检索任意位置。同时，并行训练效率也不如 Transformer 好。到这个阶段末期，SSM 的定位从"替代 Transformer"转向"与 Transformer 互补"，混合架构（见阶段 4）的探索由此展开。

**参考**：[Flash Attention (arXiv 2205.14135)](https://arxiv.org/abs/2205.14135)，[GQA (arXiv 2305.13245)](https://arxiv.org/abs/2305.13245)，[RoPE (arXiv 2104.09864)](https://arxiv.org/abs/2104.09864)，[Mamba (arXiv 2312.00752)](https://arxiv.org/abs/2312.00752)

### 阶段 4：混合架构与收敛（2024-至今）

这个阶段不再有单一的架构革命，而是几条已有路线的收敛和整合。

**MoE 成为大规模模型的事实标准**

到 2024 年，头部闭源模型（GPT-4、Gemini 1.5、Mistral 的新一代）以及开源阵营（Mixtral 8x22B、DeepSeek-V2、DeepSeek-V3）的架构几乎全部采用 MoE。Dense 的旗舰模型在这个规模上越来越少见——当总参数量进入数百 B 级别时，Dense 的训练和推理成本难以承受。MoE 在参数量和计算量之间的解耦，使"更大的知识容量"和"可控的推理成本"可以同时实现。

**SSM 的最终定位：作为互补组件**

Jamba（AI21 Labs，2024 年 3 月）是第一个规模化的混合架构产品：Transformer 注意力层和 Mamba SSM 层交替排列，每个 Transformer 层处理精确的局部和全局注意力，每个 Mamba 层用线性扫描处理长程依赖。Jamba 156B 参数（MoE，活跃约 12B），在超长序列（256K tokens）上的内存效率显著优于纯 Transformer，同时保留了 Transformer 在精确检索任务上的优势。

这个时期的工程经验使 SSM 的定位清晰化：SSM 层适合压缩长程状态（减少 KV cache），Attention 层适合精确定位任意位置的信息。两者组合覆盖了比单一架构更宽的任务谱。纯 SSM 架构在通用语言模型上没有成为主流，但在长序列生物信息（Mamba-DNA）和音频处理等特化领域仍然活跃。

## 当前技术格局（截至 2026-05）

<!-- 待填充 -->

## 关键分歧与未决问题

<!-- 
- MoE vs Dense：什么场景下 MoE 的优势是决定性的？
- SSM 能否在长序列任务上真正替代 Attention？
- 架构创新还有多大空间，还是已经收敛？
-->

## 对能力输出的影响

<!-- MoE → 推理成本降低，模型规模上限提高
     Flash Attention → 长上下文能力的工程基础
     SSM → 长序列处理效率 -->

## 与其他轨道的交叉

- **scaling**：架构选择直接影响 scaling 效率
- **inference**：MoE 和 SSM 的推理特性完全不同
- **training**：某些架构更难训练（MoE 的负载均衡问题）

## 信息源

<!-- 待补充 -->

## 更新日志

- 2026-05-02：创建骨架
- 2026-05-02：填充阶段 1（Transformer 统一与路线分化，decoder-only 主导原因）和阶段 2（规模驱动的架构稳定，Switch Transformer 的 MoE 探索）
- 2026-05-02：填充阶段 3（效率压力下的架构分化：MoE 路线/Mixtral、高效 Attention 变体/Flash Attention + GQA + RoPE、SSM 路线/Mamba）
- 2026-05-02：填充阶段 4（混合架构与收敛：MoE 成主流标准、Jamba 型混合架构、SSM 最终定位为互补组件）
