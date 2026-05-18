# Hybrid 架构近年优质落地实践梳理

> **范围说明**：本文聚焦 2024–2026 年间真正"落地到生产 / 开源发布"的 Hybrid 架构大模型。这里的 "Hybrid" 主要指**把不同序列建模机制混合使用**——最主流的是 **Attention + SSM(Mamba)混合**,其次是 **全注意力 + 线性/稀疏注意力混合**。纯研究性 toy model 不在收录范围。

---

## 0. 为什么会出现 Hybrid 架构

纯 Transformer 的两个核心痛点驱动了 Hybrid 的兴起:

- **显存随上下文线性膨胀**:自注意力需要维护 KV Cache,长度越长显存压力越大。
- **推理随序列二次方变慢**:注意力计算是 O(n²),长序列吞吐急剧下降。

而纯 SSM / 线性注意力虽然推理是线性复杂度、几乎无 KV Cache,但**精确检索能力(associative recall)弱**——在长文里"大海捞针"找一个具体事实时表现不佳。

**Hybrid 的核心思想**:用少量注意力层保留精确检索与全局交互能力,用大量 SSM / 线性注意力层承担主要的序列处理,从而在"质量"和"吞吐/显存"之间取得新的帕累托最优。这一判断在多个工业级模型上被反复验证。

业界目前收敛出的几条经验设计轴(来自 [Understanding and Enhancing Mamba-Transformer Hybrids, arXiv 2510.26912](https://arxiv.org/html/2510.26912v1)):

1. **SSM 层类型**:Mamba/Mamba-2 最常用,DeltaNet 等也被验证有效。
2. **层比例**:1:1 较为典型,但追求效率时普遍偏向更多 SSM 层(注意力占比常压到 7–8%)。
3. **注意力类型**:可用滑动窗口注意力(SWA)、SWA+全注意力组合,或纯全注意力。
4. **注意力层的位置**:把少量注意力层**周期性均匀分布**最关键;集中在网络首尾会同时损害收敛和精度。

---

## 1. Jamba(AI21)—— 第一个生产级 Mamba-Transformer 混合模型

**这是 Hybrid 架构走向生产的标志性事件。** AI21 于 2024 年 3 月发布 Jamba,是<a href="https://www.ai21.com/blog/announcing-jamba/">首个达到生产级规模的 SSM-Transformer 混合架构</a>。

> 🖼️ **关键架构图**：Jamba block 结构示意——以 1:7 的 attention:Mamba 比例交错，每 2 层插入一个 MoE。
> 原图见 ICLR 2025 论文 Figure 1：<https://proceedings.iclr.cc/paper_files/paper/2025/file/a9ed43fa31dc8b4a7d7a673d713dcb5f-Paper-Conference.pdf>（第 2 页）

**关键设计与数据:**

- **架构**:单个 Jamba block 内,attention 层与 Mamba 层按 **1:7** 比例交错,每 2 个 block 插入一个 MoE 层。源自论文 [Hybrid Transformer-Mamba Language Models (ICLR 2025)](https://proceedings.iclr.cc/paper_files/paper/2025/file/a9ed43fa31dc8b4a7d7a673d713dcb5f-Paper-Conference.pdf)。
- **有意思的实验发现**:在 hybrid 架构里 **Mamba-1 + Attention 的组合比 Mamba-2 + Attention 更好**,所以发布版用的是 Mamba-1;且有 Mamba 层后**不再需要 RoPE 等位置编码**。
- **规模与效率**:52B 总参数 / 推理时仅激活 12B(MoE),256K 上下文窗口,单张 80GB GPU 可放下 140K tokens,长上下文吞吐约为同体量 Transformer 的 **3 倍**。
- **后续迭代到生产**:
  - **Jamba 1.5** 已上架 [Amazon Bedrock](https://aws.amazon.com/blogs/aws/jamba-1-5-family-of-models-by-ai21-labs-is-now-available-in-amazon-bedrock/),并扩展到 398B 总参 / 94B 激活——**首个大规模部署的混合架构**。
  - **Jamba 1.6**(2025 年 3 月)主打企业私有化部署,质量超过 Mistral Large 2、Llama 3.3 70B,可完全在本地或 VPC 内部署。详见 [Introducing Jamba 1.6](https://www.ai21.com/blog/introducing-jamba-1-6/)。

**价值链接:**

- 官方发布博客:<https://www.ai21.com/blog/announcing-jamba/>
- ICLR 2025 论文(架构细节最权威):<https://proceedings.iclr.cc/paper_files/paper/2025/file/a9ed43fa31dc8b4a7d7a673d713dcb5f-Paper-Conference.pdf>
- Hybrid LLM 发展史回顾(AI21 官方):<https://www.ai21.com/blog/rise-of-hybrid-llms/>
- 模型权重(Apache 2.0):Hugging Face `ai21labs/Jamba`

---

## 2. NVIDIA Nemotron-H / Nemotron Nano 2 / Nemotron 3 —— 把"注意力占比压到极致"

NVIDIA 是 Hybrid 工业落地最系统化的玩家,从 Nemotron-H 到 Nemotron 3 形成了完整产品线。其核心思路:**用 Mamba-2 替换掉绝大多数自注意力层,只保留极少数注意力层做精确检索**。

> 🖼️ **关键架构图**：Nemotron-H 层级排布——Mamba-2 为主体，注意力层稀疏地周期性插入，配合 MLP 层。
> 原图见官方页面架构示意图：<https://research.nvidia.com/labs/adlr/nemotronh/>

### 2.1 Nemotron-H(2025 年 3 月)

- **Nemotron-H-56B-Base**:54 个 Mamba-2 层 + 54 个 MLP 层 + **仅 10 个自注意力层**,20T tokens FP8 预训练。
- **Nemotron-H-8B-Base**:24 Mamba-2 + 24 MLP + **仅 4 个注意力层**。
- **效率**:长上下文(65536 输入)下,相比 Qwen-2.5-7B 和 Llama-3.1-8B 分别有 **1.8x 和 3x 推理加速**,关键来源是 SSM 层**没有 KV Cache**。
- **延伸落地**:Nemotron-H-56B 被用作 **Cosmos-Reason 1**(物理 AI 的强 VLM)的骨干网络。

  官方页面:<https://research.nvidia.com/labs/adlr/nemotronh/>

### 2.2 Nemotron Nano 2(2025 年 8 月)—— 面向推理工作负载

- **架构**:62 层中只有 **6 个注意力层**,28 个 FFN,28 个 Mamba-2 层;用 GQA(40 query heads / 8 KV heads),无位置编码。
- **核心卖点**:在 8K 输入 / 16K 输出的推理场景下,相比 Qwen3-8B 吞吐高达 **6.3 倍**,精度持平或更好。
- **工程亮点**:用 Minitron 压缩+蒸馏策略,把 12B 压到 9B,可在**单张 A10G(22GB)**上做 128K 上下文推理。

  论文:<https://arxiv.org/abs/2508.14444> ｜ HTML 全文:<https://arxiv.org/html/2508.14444>

### 2.3 Nemotron 3 系列(2025 年 12 月)—— Hybrid + MoE + 量化的集大成

- **架构**:Hybrid Mamba-Transformer **+ 稀疏 MoE**(Nano / Super / Ultra 三档),注意力层比例继续保持很低,每个注意力层只用 2 个 KV head。
- **长上下文突破**:不用 RoPE,靠 Mamba 隐式位置编码,在 **1M token** 上下文上 RULER-100 得分 **86.34**(对比此前 dense hybrid 在 1M 时仅 23.43)。
- **agentic 表现**:在多轮对话、工具调用、agentic reasoning 上比前代 Transformer / hybrid 提升 **5–10 个百分点**。
- **硬件协同**:Super/Ultra 用 **NVFP4** 在 Blackwell B200 上原生预训练,相比 H100 上 FP8 推理提速 4 倍。
- **吞吐**:Nemotron 3 Super 在 8K 输入 / 64K 输出下,吞吐分别是 GPT-OSS-120B 和 Qwen3.5-122B 的 **2.2x 和 7.5x**。

**价值链接:**

- Nemotron 3 系列总览:<https://research.nvidia.com/labs/nemotron/Nemotron-3/>
- Nemotron 3 白皮书 PDF:<https://research.nvidia.com/labs/nemotron/files/NVIDIA-Nemotron-3-White-Paper.pdf>
- Nemotron 3 Super 技术报告:<https://research.nvidia.com/labs/nemotron/files/NVIDIA-Nemotron-3-Super-Technical-Report.pdf>
- Nemotron 3 Super 工程博客(含 agentic benchmark):<https://developer.nvidia.com/blog/introducing-nemotron-3-super-an-open-hybrid-mamba-transformer-moe-for-agentic-reasoning/>

> NVIDIA 系列模型权重、训练 recipe、数据 pipeline 大多以 NVIDIA Open Model License / Apache 2.0 开源,是研究 Hybrid 落地细节最完整的公开参考。

---

## 3. MiniMax-01 / MiniMax-M1 —— 线性注意力路线的大规模 Hybrid

MiniMax 走的是另一条 Hybrid 路线:**Lightning Attention(线性注意力变体)+ Softmax Attention + MoE**,而非 Mamba。这是**首次在大规模上把线性注意力作为主体**的工业实践。

> 🖼️ **关键架构图**：MiniMax-01 Hybrid Attention——7 个 lightning attention block 后接 1 个 softmax attention block，外加 MoE。
> 原图见 arXiv 论文架构图：<https://arxiv.org/html/2501.08313> ｜ HF 论文页：<https://huggingface.co/papers/2501.08313>

### 3.1 MiniMax-Text-01(2025 年 1 月)

- **架构**:每 **7 个 lightning attention 块后接 1 个 softmax attention 块**,配合 MoE(32 experts)。456B 总参数 / 每 token 激活 45.9B。
- **长上下文**:训练上下文 1M tokens,推理可外推到 **4M tokens**——在 4M token 的 Needle-In-A-Haystack 检索任务上达到 **100% 准确率**。
- **关键洞察**(来自[技术报告](https://filecdn.minimax.chat/_Arxiv_MiniMax_01_Report.pdf)):纯线性注意力模型虽高效但**无法胜任检索**,不适合做 LLM;而 hybrid 模型在检索和外推上**反而超过纯 softmax attention**。
- **性能**:对标 GPT-4o、Claude-3.5-Sonnet,而上下文窗口长 20–32 倍。

### 3.2 MiniMax-M1(2025 年 6 月)—— 首个开源大规模 Hybrid 推理模型

- 基于 MiniMax-Text-01 继续训练,**世界首个开源大规模 hybrid-attention 推理模型**。
- 原生支持 **1M token 输入 + 80K token 推理输出**。
- **效率优势**:在 100K token 生成长度下,FLOPs 仅为 DeepSeek R1 的 **25%**;80K token 深度推理只需 R1 约 30% 算力。
- **训练成本**:结合 hybrid-attention 与自研 CISPO 强化学习算法,512 张 H800 上完整 RL 训练**仅 3 周、约 53.5 万美元**。
- **效果**:SWE-bench 验证集 56.0%,长上下文理解超过 OpenAI o3、Claude 4 Opus,全球第二(仅次 Gemini 2.5 Pro)。

**价值链接:**

- MiniMax-01 论文:<https://arxiv.org/abs/2501.08313>
- MiniMax-M1 论文:<https://arxiv.org/abs/2506.13585>
- 官方开源公告:<https://www.minimax.io/news/minimax-01-series-2> ｜ <https://www.minimax.io/news/minimaxm1>
- 代码与权重:<https://github.com/MiniMax-AI/MiniMax-01>

---

## 4. 腾讯 Hunyuan-TurboS —— Mamba-Transformer 协同 + 自适应 CoT

腾讯混元 TurboS 是国内大厂的代表性 Hybrid 落地,主打 **Mamba-Transformer 协同 + 自适应思维链**。

- **架构**:深度交错 57 个 Mamba2 层、7 个 GQA 注意力层、64 个 MoE-FFN 层,平衡长上下文效率与推理质量。
- **效率**:在 100K+ tokens 下可实时交互,相比同规模纯 Transformer MoE 有约 **1.8x 加速**。
- **效果**:LMSYS Chatbot Arena 得分 1356,处于第一梯队。
- 论文:*Hunyuan-TurboS: Advancing Large Language Models through Mamba-Transformer Synergy and Adaptive Chain-of-Thought* (2025)。

> 综述参考:<https://www.emergentmind.com/topics/mamba-transformer-llm>

---

## 5. IBM Bamba —— 面向轻量化部署的 Hybrid

- IBM Research 的 Bamba 沿用 Mamba+Transformer 混合思路,通过 8-bit 量化把模型体积从 **18GB 压到 9GB**,在训练数据是 Llama-3.1 8B 的 7 倍的情况下保持相当性能。
- 代表了 Hybrid 在**端侧 / 资源受限部署**方向的探索。

> 背景综述:<https://www.askaibrain.com/en/posts/end-of-transformers-hybrids-attention-state-space-2025>

---

## 6. 横向对比速查表

| 模型 | 机构 | 混合方式 | 注意力占比 | 上下文 | 核心收益 | 开源 |
|---|---|---|---|---|---|---|
| Jamba 1.x | AI21 | Mamba-1 + Attn + MoE | ~1:7 | 256K | 长文吞吐 3x,生产级首发 | ✅ Apache 2.0 |
| Nemotron-H | NVIDIA | Mamba-2 + Attn + MLP | ~7–8% | 长上下文 | 推理 1.8–3x 加速 | ✅ |
| Nemotron Nano 2 | NVIDIA | Mamba-2 + Attn + FFN | 6/62 层 | 128K | 推理吞吐 6.3x | ✅ |
| Nemotron 3 | NVIDIA | Mamba-2 + Attn + MoE | 极低 | 1M | 1M 检索 86.34,agentic +5~10pt | ✅ |
| MiniMax-01 | MiniMax | Lightning + Softmax + MoE | 1/8 块 | 1M→4M | 4M 上下文,检索超纯 softmax | ✅ |
| MiniMax-M1 | MiniMax | 同上 + RL(CISPO) | 1/8 块 | 1M+80K out | 推理 FLOPs 仅 R1 的 25% | ✅ |
| Hunyuan-TurboS | 腾讯 | Mamba2 + GQA + MoE | 7/128 层 | 100K+ | 实时交互,~1.8x 加速 | 部分 |
| Bamba | IBM | Mamba + Transformer | — | — | 量化后体积减半 | ✅ |

---

## 7. 落地经验提炼(实践建议)

1. **注意力不能全砍,但可以砍到很少**:7–8% 的注意力层占比是工业界反复验证的"安全区",既保留精确检索,又拿到大部分效率收益。
2. **注意力层要均匀周期分布**:集中在网络首尾会同时损害收敛与精度(Jamba、Nemotron 均有此结论)。
3. **Block 内细粒度混合 > 粗粒度外层交替**:在 block 内做 attention-Mamba-FFN 的融合,比简单的整层交替效果更好。
4. **有 SSM/Mamba 层后通常可去掉显式位置编码**(RoPE 非必需),长度外推反而更稳。
5. **蒸馏 + 权重复用可大幅降成本**:把纯 Transformer 蒸馏为 Hybrid/Mamba 学生模型时,用对应注意力权重初始化 Mamba 能显著加速收敛(MaTVLM、Minitron 等)。
6. **Hybrid 与 MoE / 低比特量化天然互补**:Nemotron 3、MiniMax、Hunyuan 都是 "Hybrid + MoE + 量化" 三件套,这是当前大规模高效部署的主流组合拳。
7. **最适合的场景**:超长上下文(RAG、长文档问答)、生成密集型推理(长 thinking trace)、long-running agent——这些正是纯 Transformer 二次方成本最痛的地方。

---

## 8. 延伸阅读 / 权威综述

- *Understanding and Enhancing Mamba-Transformer Hybrids*(设计轴最系统的分析):<https://arxiv.org/html/2510.26912v1>
- Emergent Mind 主题页(持续更新的 Hybrid 文献聚合):
  - Mamba-Transformer LLM:<https://www.emergentmind.com/topics/mamba-transformer-llm>
  - Hybrid Mamba-Transformer:<https://www.emergentmind.com/topics/hybrid-mamba-transformer>
  - Nemotron 3 family:<https://www.emergentmind.com/topics/nemotron-3-family>
- AI21《Attention was never enough: Tracing the rise of hybrid LLMs》:<https://www.ai21.com/blog/rise-of-hybrid-llms/>
- 《End of Transformers? Attention + State-Space Hybrids in 2025》:<https://www.askaibrain.com/en/posts/end-of-transformers-hybrids-attention-state-space-2025>

---

*整理时间:2026 年 5 月 ｜ 信息以公开论文、官方博客与技术报告为准，部分数据随模型迭代可能变化，引用前建议回溯原始链接核对。*
