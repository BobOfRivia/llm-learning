# DeepSeek

## 技术路线

DeepSeek 是深度求索（深圳），隶属于量化对冲基金幻方科技旗下，2023 年开始在 LLM 领域产出影响力不断提升的工作，在 2024-2025 年完成了从"优秀的中国开源模型"到"范式级贡献者"的转变。理解 DeepSeek 的技术路线，有一个背景因素不能忽略：**美国对华芯片出口管制**。DeepSeek 无法获得 H100，只能使用性能较弱的 H800（阉割版，互联带宽受限），这个硬件约束直接驱动了他们在架构和训练算法上的深度优化——缺少算力，就必须让每一次计算都更有效率。

### DeepSeek-V2 与 MLA 架构创新（2024-05）

DeepSeek-V2（2024 年 5 月，arXiv 2405.04434）在架构上做出了对 KV Cache 最激进的一次改造：**Multi-head Latent Attention（MLA）**。

标准 Multi-Head Attention 的 KV Cache 瓶颈众所周知：每个 token、每个层都需要存储所有注意力头的 K 和 V，长序列时显存消耗是模型权重的数倍。MLA 的解法是：**把 K 和 V 压缩到一个低维的潜在向量（latent vector）里**。具体来说，K 和 V 由这个潜在向量通过低秩矩阵投影得出，推理时 cache 的是潜在向量而非完整的 K、V。结果：KV Cache 内存消耗降低约 **87%**，同时（通过在矩阵分解上的精细设计）推理速度不降反升。

这是一个真正的架构创新，而非对已有方法的工程调优。GQA（Grouped Query Attention）是通过减少 K、V 头的数量来降低 Cache；MLA 是在保持完整注意力表达能力的同时，从根本上改变 Cache 的存储结构。DeepSeek-V2（236B 总参数，MoE 架构，21B 活跃参数）用了这个机制，在多数基准上超过 Llama 3 70B，且推理成本极低。

### DeepSeek-V3：用受限资源训出前沿模型（2024-12）

DeepSeek-V3（2024 年 12 月，arXiv 2412.19437）是这个方向的集大成：671B 总参数（MoE，37B 活跃参数），在 14.8T tokens 上用 2048 张 H800 训练完成。DeepSeek 公布的训练成本约 557 万美元——这个数字在业界引发了巨大争议，怀疑者认为没有包括 base model 的预训练、工程探索的失败成本等；但即便打折计算，这个数量级的成本与 GPT-4 或 Claude 3 Opus 的训练成本估算之间的差距仍然是数量级级别的。

V3 的训练效率来自多个层面的优化：**FP8 混合精度训练**（在当时 H800 支持的精度限制内，系统性地使用 FP8 格式降低计算量和显存压力）、**MoE 负载均衡改进**（auxiliary-loss-free 的负载均衡策略，避免 expert 使用不均时的额外 loss 项影响主训练目标）、**流水线并行的气泡优化**（在 H800 的互联带宽限制下，精心设计微批次调度，减少流水线空泡时间）。

V3 的能力在中文任务上超过 GPT-4o，在代码和数学上与 Claude 3.5 Sonnet 相当——这是第一次，一个非美国实验室的模型在全面能力上达到了 OpenAI/Anthropic 的水平。

### DeepSeek-R1-Zero 与 RL for Reasoning（2025-01）

DeepSeek-R1（2025 年 1 月 20 日，arXiv 2501.12948）的最重要贡献不是最终的 R1 模型，而是 **R1-Zero** 的实验——从基础模型出发，不经过任何 SFT 冷启动，只用 GRPO 算法 + 可验证奖励（数学答案正确性 + 代码测试通过率）训练。

R1-Zero 在训练过程中涌现出了此前从未在 LLM 中大规模观察到的行为：**在遇到困难问题时，模型自发停下来、检查之前步骤的正确性、发现错误、然后重新尝试**——这个自我反思的行为没有被显式训练，是从 RL 奖励中自然涌现的策略。DeepSeek 在技术报告中称之为"aha moment"（顿悟时刻），并展示了训练曲线上这个行为涌现的时间点。

这个结果的理论意义：证明了大量人工标注的 SFT 思维链数据不是 RL for Reasoning 的必要条件——给定足够好的可验证奖励信号，模型可以自己"发明"有效的推理策略。这直接削弱了"只有 OpenAI 能做推理模型，因为他们有大量专有思维链数据"的假设。

最终的 R1 在 R1-Zero 基础上加入了冷启动（少量高质量思维链的 SFT，稳定推理格式）和多轮 RL 迭代，在 AIME 2024 和 MATH-500 等基准上与 o1 持平或超过。同步发布的 R1-Distill 系列（1.5B 到 70B，基于 Llama 和 Qwen 架构）证明推理能力可以从大推理模型蒸馏到小模型。

### V3.1/V3.2/V4：迭代加速（2025-2026）

R1 发布后，DeepSeek 的迭代节奏显著加快，且每次迭代都在解决一个具体的工程问题，而非仅仅追求 benchmark 数字。

**V3-0324（2025 年 3 月）** 是 V3 的增量优化版本，基础能力全面提升，是进入下一版本线前的工程打磨。

**DeepSeek-V3.1（2025 年 8 月）** 解决的是推理模型和普通模型分轨部署的效率问题：V3.1 把思考模式和非思考模式集成到同一个模型里，用户可以在单次推理中选择是否激活 extended thinking，无需切换两个不同的模型——这是 hybrid thinking 的首次落地。工具使用能力也在此版本得到系统强化。能力上，SWE-Verified（Agent 编程任务）相比 V3-0324 提升约 45%。

**DeepSeek-V3.2-Exp（2025 年 9 月）** 是 V3.2 的实验版，核心是引入**稀疏注意力（Sparse Attention）**，专门针对超长上下文场景优化推理效率——在不显著降低质量的前提下降低长序列的注意力计算量。

**DeepSeek-V3.2（2025 年 12 月）** 是正式版，在 V3.2-Exp 的基础上稳定完善。这一版本首次在数学竞赛推理任务上达到**金牌水平**：IMO 2025、CMO（中国数学奥林匹克）、ICPC 世界总决赛均有金牌级表现，是开放权重模型在形式化推理能力上的重要里程碑。

**DeepSeek-V4（2026 年 4 月）** 是架构层面的又一次大跳跃，核心创新是**混合注意力（Hybrid Attention）**：将两种不同压缩粒度的稀疏注意力机制结合，使 1M token 上下文成为所有官方服务的默认配置，同时实际推理 FLOPs 和 KV Cache 消耗相比 V3.2 大幅下降。模型分 Pro Max 和 Flash 两档，覆盖高能力和高效率两个场景。从 V2 的 MLA 到 V4 的混合稀疏注意力，DeepSeek 在 KV Cache 压缩这条路上持续深挖，每一代都在上一代基础上更进一步。

### 技术透明度作为差异化

DeepSeek 在技术透明度上与 OpenAI 形成了鲜明对比：R1 的完整技术报告包含了训练流程的详细描述、关键超参数、训练曲线、消融实验——这些内容在 OpenAI 的 o1 技术报告中完全缺失。这种透明度的动机可能是多方面的（学术声誉、吸引人才、建立合法性），但客观效果是：R1 让全球学术界和工程师可以复现和理解 RL for Reasoning 的关键机制，加速了整个领域的技术扩散。

## 关键决策与赌注

**MLA 架构创新（2024）**：赌 KV Cache 是 LLM serving 的根本性瓶颈，从架构层面解决比工程优化更有持久性。结果：V2 的 87% KV Cache 压缩在实际部署中带来了显著的成本优势，MLA 已经被开源社区广泛研究和采用。

**FP8 全程训练 + 极致成本优化（V3）**：赌在受限硬件上通过算法工程可以达到与 H100 训出的前沿模型相当的水平。结果：V3 的成本-性能比是迄今最优的，证明了这个赌注。

**R1-Zero 的纯 RL 路线（2025）**：赌推理能力可以从 RL 中涌现，不依赖大量专有 SFT 数据。结果：颠覆了"推理模型需要 OpenAI 规模的私有思维链数据"的假设，是 2025 年最重要的技术验证之一。

**完全开放权重 + 详细技术报告**：赌透明度和开放性能建立国际声誉和技术影响力，且不威胁核心商业利益（DeepSeek 的母公司是量化基金，不靠 API 收费）。结果：有效——R1 的发布使 DeepSeek 成为国际 AI 社区讨论最多的实验室之一。

## 与主流路线的差异

DeepSeek 是**芯片限制逼出的架构创新**的最典型案例。他们的 MLA、MoE 调优、FP8 训练等工程创新，都是在"比对手拥有更少算力"的约束下被迫发展出来的——这反而催生了一些 H100 用户不需要去深入优化的技术。这个逻辑类似于当年日本制造业在资源约束下发展出精益生产（Lean Manufacturing）的历史——约束有时是创新的催化剂。

他们也是目前唯一一家在技术贡献上同时达到"架构创新"（MLA）和"训练范式创新"（R1-Zero 的 RL for Reasoning 实证）两个层面的非美国/英国实验室。这使得 DeepSeek 的独立文件分量成立。

## 信息源

- [DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model (arXiv 2405.04434)](https://arxiv.org/abs/2405.04434)
- [DeepSeek-V3 Technical Report (arXiv 2412.19437)](https://arxiv.org/abs/2412.19437)
- [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning (arXiv 2501.12948)](https://arxiv.org/abs/2501.12948)
- [DeepSeek V3.1 Technical Analysis (RunPod Blog, 2025)](https://www.runpod.io/blog/deepseek-v3-1-a-technical-analysis-of-key-changes)
- [DeepSeek V3.2 Release Notes (DeepSeek API Docs, 2025)](https://api-docs.deepseek.com/news/news251201)
- [DeepSeek V4 Preview (DeepSeek API Docs, 2026)](https://api-docs.deepseek.com/news/news260424)

## 更新日志

- 2026-05-02：创建初始版本（MLA 架构创新、V3 成本革命、R1-Zero 纯 RL 实验、技术透明度，芯片限制的底层逻辑）
- 2026-05-02：补充信息源部分
- 2026-05-02：补充 V3-0324/V3.1/V3.2-Exp/V3.2/V4 迭代历史（hybrid thinking、稀疏注意力、混合注意力、IMO 金牌）
