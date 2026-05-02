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

**MoE 已成为 frontier 模型的事实标准，Dense 的旗舰产品基本消失。** 2024-2025 年发布的主要大规模模型几乎全部采用 MoE：Llama 4 Maverick（400B total / 17B active，首次放弃 Dense）、DeepSeek-V3（671B / 37B active，256 routed experts + 1 shared）、Qwen 3 MoE（128 experts，top-8 激活）、Mistral Large 3（675B / 41B active）、GLM-5（744B / 44B active，256 experts）。NVIDIA 在 2025 年初的技术博客中明确表述：几乎所有 frontier 模型都已迁移到 MoE 设计。Dense 模型仍在中小参数规模（～70B 以内）有意义，原因是小规模下 MoE 的负载均衡开销相对于计算量不划算，且小模型对推理成本的压力相对低。

**MoE 内部出现了设计哲学的分化。** 以两个极端为代表：DeepSeek-V3 选择了"多专家细粒度"路线——256 个路由专家，每个专家隐层维度仅 2048，每次 top-8 激活，专家组合的多样性极高；Llama 4 Maverick 选择了"少专家大规模"路线——128 个专家，隐层维度 8192，top-1 激活，参数总量相近但每个专家的表达能力更强。DeepSeek 还保留了 1 个"永远激活"的共享专家，而 Qwen 3 MoE 在实验后决定去掉共享专家转向纯路由设计——这说明共享专家的必要性在业界尚无定论。另一个创新是 auxiliary-loss-free 负载均衡（DeepSeek-V3 采用），用 token 级别的动态偏置替代传统辅助损失，声称在不影响训练稳定性的前提下获得更均匀的专家利用。

**Hybrid 架构完成了从实验到生产验证的跨越。** Jamba（AI21 Labs，2024 年 3 月）和 Jamba 2（398B total / 94B active，2024 年 11 月）是最早的规模化生产验证案例，证明 Transformer + Mamba + MoE 的三重混合可以稳定运行。NVIDIA 的 Nemotron-H（2024 年底，arXiv 2504.03624）给出了迄今最有说服力的效率数据：用 92% Mamba-2 层 + 8% Attention 层替换 Transformer，在 MMLU/GSM8K/HumanEval 上与 LLaMA-3.1/Qwen-2.5 匹配或超越，同时推理吞吐提升 3 倍。Gemini 2.5 Pro（Google，2025 年 3 月）据悉也采用了 Transformer + SSM + MoE 的混合设计。这一系列结果使混合架构从"有趣的学术探索"进入了"可选的工程方案"阶段。

**MLA（多头潜在注意力）是 Attention 层面最重要的近期创新。** DeepSeek-V2（2024 年 5 月）首创 MLA——通过低秩压缩将 KV 状态投影到低维潜在空间，缓存压缩后的表示而非完整的 KV 矩阵，KV cache 大小减少 93.3%，生成吞吐提升 5.76 倍，训练成本降低 42.5%。DeepSeek-V3 和 R1 均继承了 MLA。2025 年的 TransMLA 论文（arXiv 2502.07864）进一步证明 MLA 可以通过迁移学习移植到已有 GQA 模型，不需要从头训练，这为 MLA 的更广泛采用打开了路径。相比之下，Differential Transformer（微软，ICLR 2025，arXiv 2410.05258）和 Titans（Google，2025 年 1 月，arXiv 2501.00663）虽然在论文中展示了显著改进（Differential Transformer：35-40% 更少参数达同等性能；Titans：有效支持 2M+ tokens），但截至 2026 年 5 月尚未在任何 frontier 模型中落地。

## 关键分歧与未决问题

**MoE 设计哲学之争：多专家细粒度 vs 少量大专家，没有明确赢家。** DeepSeek 路线（256 个小专家，维度 2048）的理论优势是路由组合空间极大，模型可以学到精细的专家专业化；Llama 4 路线（128 个大专家，维度 8192）的理论优势是每个专家的表达能力更强，路由网络更易于稳定训练。实践中两种设计都达到了 SOTA 级别的性能，说明在当前规模下两条路都是可行的，但最优的设计选择尚无系统性对比实验。共享专家的去留也属于这个分歧：DeepSeek 保留、Qwen 3 放弃，各自给出了不同的理由，但没有跨模型的受控实验。

**混合架构的最优比例没有定论。** Nemotron-H 用 8% Attention + 92% Mamba-2 就达到了 Transformer 的质量，而 Jamba 的比例是 1:7（Transformer:Mamba）。两者在不同场景下各有优势，但没有理论框架可以预测"给定任务 X，最优的 Attention 比例是多少"。这个比例可能不只取决于任务，还取决于模型规模和训练数据。

**架构是否已经收敛？** 一个乐观的视角是：MoE + Hybrid（少量 Attention + SSM）+ 高效 KV 管理（GQA/MLA）已经是足够好的终态，未来的进步来自训练方法和数据，而非架构本身。一个不那么乐观的视角是：Differential Transformer 和 Titans 代表了还未被工业界充分探索的改进空间——前者在论文实验中用更少参数达到同等质量，后者在超长上下文上有根本性优势，但工业界的决策通常滞后论文 12-18 个月。

**MLA 的采用速度是否会加快？** MLA 的 93% KV cache 压缩率对于长上下文 serving 的成本影响是结构性的，但它需要在推理框架中专门实现（MLA 的 KV 格式与标准 GQA 不兼容），迁移成本不小。TransMLA 的迁移方案降低了门槛，但目前已采用 MLA 的主要是 DeepSeek 系列模型。其他实验室是否会跟进，取决于其 KV cache 成本是否已经成为瓶颈。

## 对能力输出的影响

**MoE → 知识容量与推理成本解耦，影响所有能力的可及性。** Dense 模型中参数量和单次推理计算量是绑定的，MoE 打破了这个约束——DeepSeek-V3 有 671B 总参数的知识存储能力，但单次推理只激活 37B 的计算量，服务成本与 37B dense 模型相当。这使得在相同推理预算下，模型能携带的知识量大幅提升，对需要广泛知识覆盖的任务（**知识检索、多语言、多领域理解**）影响最直接。

**Flash Attention + GQA → 长上下文的工程基础，无这些支撑则百万 token 上下文不可行。** Flash Attention 的 IO-aware 重设计把 O(n²) 内存占用降低 5-20 倍，使 100K+ context 的训练在有限 GPU 内存中成为可能；GQA 把推理时 KV cache 压缩到与 MQA 相当的水平，使长上下文的 serving 成本不再线性爆炸。→ 详见 [capabilities/long-context.md](../capabilities/long-context.md)

**RoPE → 上下文长度的泛化能力，支撑了位置外推技术的系列进展。** RoPE + YaRN/NTK-aware scaling 的组合使模型在不重新训练的前提下能处理训练长度数倍的序列，这是 Llama 系列和 Claude 的长上下文扩展的技术基础。

**Hybrid Mamba-Transformer → 超长序列处理能力的成本突破。** Jamba 256K 上下文、Nemotron-H 3x 吞吐提升，这些结果表明混合架构可以在维持质量的前提下，将极长序列场景的内存和计算成本降低到实用水平。→ 影响 [capabilities/long-context.md](../capabilities/long-context.md) 的成本边界

**MLA → KV cache 压缩影响长上下文 serving 的经济学。** 93% cache 压缩在长对话和多轮交互场景下意义重大：一个 100K token 上下文下，MLA 的内存占用比 GQA 低约 15 倍（取决于 GQA 头数），这直接影响单台机器能并发服务的请求数，进而影响**长上下文**和 **memory** 类能力的实际可部署规模。

## 与其他轨道的交叉

- **scaling**：架构选择直接影响 scaling 效率
- **inference**：MoE 和 SSM 的推理特性完全不同
- **training**：某些架构更难训练（MoE 的负载均衡问题）

## 信息源

**奠基论文**
- [Attention Is All You Need (Vaswani et al., 2017, arXiv 1706.03762)](https://arxiv.org/abs/1706.03762) — Transformer 架构的起点
- [Switch Transformers (Fedus et al., Google Brain, 2021, arXiv 2101.03961)](https://arxiv.org/abs/2101.03961) — 首次规模化验证稀疏 MoE 的可行性

**高效 Attention**
- [Flash Attention (Dao et al., 斯坦福, 2022, arXiv 2205.14135)](https://arxiv.org/abs/2205.14135) — IO-aware 重设计，2-4x 训练加速，5-20x 内存降低
- [GQA: Training Generalized Multi-Query Transformer (Ainslie et al., Google, 2023, arXiv 2305.13245)](https://arxiv.org/abs/2305.13245) — KV cache 压缩与质量保留的折中方案
- [RoFormer: RoPE (Su et al., 2022, arXiv 2104.09864)](https://arxiv.org/abs/2104.09864) — 相对位置编码，支撑上下文外推

**MoE 架构**
- [Mixtral of Experts (Jiang et al., Mistral AI, 2024, arXiv 2401.04088)](https://arxiv.org/abs/2401.04088) — 开源生态第一个生产级 MoE 验证
- [DeepSeek-V2 (2024, arXiv 2405.04434)](https://arxiv.org/abs/2405.04434) — MLA 的首次提出，93.3% KV cache 压缩
- [DeepSeek-V3 Technical Report (2024, arXiv 2412.19437)](https://arxiv.org/abs/2412.19437) — MoE 设计细节（256 experts、auxiliary-loss-free 负载均衡）

**SSM 与混合架构**
- [Mamba: Linear-Time Sequence Modeling (Gu & Dao, 2023, arXiv 2312.00752)](https://arxiv.org/abs/2312.00752) — 选择性 SSM，Mamba-1 的核心论文
- [Transformers are SSMs: SSD Framework / Mamba-2 (Dao & Gu, 2024, arXiv 2405.21060)](https://arxiv.org/abs/2405.21060) — Mamba-2，数学上统一 SSM 与 Attention，2-8x 速度提升
- [Jamba: Hybrid Transformer-Mamba (AI21 Labs, 2024, arXiv 2403.19887)](https://arxiv.org/abs/2403.19887) — 第一个规模化混合架构产品
- [Nemotron-H (NVIDIA, 2024, arXiv 2504.03624)](https://arxiv.org/abs/2504.03624) — 92% Mamba-2 + 8% Attention，3x 推理吞吐提升

**近期创新（学术阶段）**
- [Differential Transformer (Microsoft, ICLR 2025, arXiv 2410.05258)](https://arxiv.org/abs/2410.05258) — 差分注意消除噪声，35-40% 参数效率提升
- [TransMLA (2025, arXiv 2502.07864)](https://arxiv.org/abs/2502.07864) — MLA 可移植到已有 GQA 模型

**分析文章**
- [Llama 4 深度解析 (Cameron Wolfe, Substack)](https://cameronrwolfe.substack.com/p/llama-4) — Llama 4 MoE 设计的详细分析与 DeepSeek-V3 对比

## 更新日志

- 2026-05-02：创建骨架
- 2026-05-02：填充阶段 1（Transformer 统一与路线分化，decoder-only 主导原因）和阶段 2（规模驱动的架构稳定，Switch Transformer 的 MoE 探索）
- 2026-05-02：填充阶段 3（效率压力下的架构分化：MoE 路线/Mixtral、高效 Attention 变体/Flash Attention + GQA + RoPE、SSM 路线/Mamba）
- 2026-05-02：填充阶段 4（混合架构与收敛：MoE 成主流标准、Jamba 型混合架构、SSM 最终定位为互补组件）
- 2026-05-02：填充当前技术格局、关键分歧、对能力输出的影响、信息源（Llama 4 MoE 转向、Nemotron-H、MLA 创新、Hybrid 架构生产验证）
