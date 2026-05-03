# LLM 学习路径

> 这份文档给出**学习的顺序、节奏和达标标志**，与 [study-tasks.md](study-tasks.md) 配合使用。
>
> - **study-tasks.md**：知识点的微粒清单（80+ 条 task）
> - **study-plan.md（本文件）**：把这些 task 排进时间和因果链
>
> 起始日期：2026-05-03 ｜ 预期主干完成：2026-08 至 09 ｜ 之后转入持续追踪模式

---

## 元原则

**学习不是覆盖任务，是打通因果链。** 每个阶段结束时，应该能用一段话讲清楚这个阶段的演进逻辑——不讲细节、讲"为什么"。如果讲不出来，回去补对应 task。

**每个 task 完成后必须写入对应的 tracks/ 或 topics/ 文件。** 这是不可跳过的环节。study-tasks.md 是骨架，写入是把因果链用自己的语言固化下来。光看不写，三个月后又会碎片化回去。

**跳过公式推导，但不跳过直觉。** 不要重新推一遍 DPO 数学化简，但要能说出"DPO 本质上把 RL 问题转化为分类问题，前提是数据离线"。所有 ⭐⭐⭐ task 的目标都是抓直觉，不是推导熟练度。

**节奏建议**：每周 3-5 个 task，⭐⭐⭐ 算两个名额。每周末做一次 mini-review——把这周写入的内容和上周做对比，看是否能形成连贯叙事。

---

## 阶段 0｜地基（1-2 周）

**目的**：建立 Transformer 的基础 mental model。这五件事是后续所有内容的 prerequisite，地基不牢则后面学 RoPE、MLA、RLHF 都是死记硬背。

**任务**：
- BPE（Byte Pair Encoding）算法 ⭐⭐
- Self-Attention 机制 ⭐⭐
- Multi-Head Attention ⭐⭐
- Transformer 层结构（Pre-LN / FFN / 残差）⭐⭐
- 自回归语言建模目标 ⭐
- decoder-only 为什么统治 ⭐

**达标标志**：能用白板向一个不懂 LLM 的同事讲 30 分钟"Transformer 是什么、为什么 work"。具体地：能从零写出 attention 公式并解释每一步的设计理由、能手动模拟 BPE 合并过程、能解释为什么现代模型都是 decoder-only。

**写入位置**：`tracks/architecture.md` 阶段 1 + `topics/attention-mechanism.md`。

---

## 阶段 1｜三条主干轨道（4-6 周，并行推进）

**目的**：理解架构、训练、scaling 三条主干的演进脉络。这三条**必须并行学**，因为它们在 2020-2024 年是相互缠绕演进的——MLA 是 DeepSeek 为了 inference 经济性发明的，理解它需要同时调用架构、scaling、推理三个视角。串行学会丢失这种交叉。

**架构线**（约 8 个 task）：
- 绝对位置编码的局限 ⭐⭐
- RoPE（旋转位置编码）⭐⭐⭐
- 上下文长度外推（YaRN / NTK-aware）⭐⭐
- Flash Attention ⭐⭐⭐
- GQA / MQA ⭐⭐
- MLA（多头潜在注意力）⭐⭐⭐
- Expert 路由机制 ⭐⭐
- Auxiliary-loss-free 负载均衡 ⭐⭐⭐

**训练线**（约 6 个 task）：
- SFT 的数据工程 ⭐⭐
- 代码数据对推理能力的影响 ⭐⭐
- RLHF 完整流程 ⭐⭐⭐
- PPO 算法细节 ⭐⭐⭐
- DPO 的数学推导 ⭐⭐⭐
- DPO 变体的动机 ⭐⭐

**Scaling 线**（约 5 个 task）：
- Kaplan Scaling Laws 详解 ⭐⭐
- Kaplan 误差来源 ⭐⭐
- Chinchilla 分析方法 ⭐⭐⭐
- 推理成本改变最优化目标 ⭐⭐
- MoE Scaling Laws 的五因子框架 ⭐⭐

**节奏建议**：每周 3 条线各推 1 个 task，避免任何一条停滞超过 1 周。MLA 应该在 Chinchilla + KV Cache 都学完之后再碰，因为它需要"长上下文经济学"的全貌。

**达标标志**：能回答以下问题，每个回答 200-400 字：
1. 2024 年为什么"过训练"成主流？（涉及 scaling line + 推理成本）
2. GPT-4 为什么用 MoE 而不继续 scale dense？（涉及 architecture + scaling）
3. 从 RLHF 到 DPO，工业界为什么愿意切换？（涉及 training line）
4. RoPE 解决了什么、MLA 解决了什么、它们是同一个问题吗？（涉及 architecture + inference 前奏）

**写入位置**：`tracks/architecture.md`、`tracks/training.md`、`tracks/scaling.md`，对应 topics 同步补充。

---

## 阶段 2｜推理纪元的范式跃迁（3 周）

**目的**：搞透 2024-2025 年最大的范式断裂——从 RLHF 到 RL on verifiable reward。这一阶段最容易"看新闻看个表面"，所以单独成阶段，强迫深入。

**任务**：
- RLVR 的设计 ⭐⭐
- GRPO vs PPO ⭐⭐⭐
- PRM vs ORM ⭐⭐
- R1-Zero 的涌现 ⭐⭐⭐
- 推理蒸馏机制 ⭐⭐
- Inference-time Scaling 机制 ⭐⭐⭐
- Best-of-N 采样的数学 ⭐⭐
- Chain-of-Thought 为什么有效 ⭐⭐
- o1 的技术实质 ⭐⭐⭐

**达标标志**：写一篇 1500 字左右的内部 memo，标题《从 RLHF 到 R1：训练范式三年演进》。要求：必须解释"为什么 GRPO 能省掉 value model"、"为什么 R1-Zero 的自我反思不是魔法"、"o1 和 GPT-4 + CoT 的本质区别在哪"。这篇 memo 写完后归档到 `topics/rl-for-reasoning.md` 或单独成文。

**写入位置**：`tracks/training.md` 阶段 4、`topics/rl-for-reasoning.md`、`topics/inference-time-compute.md`、`topics/chain-of-thought.md`。同时在 `eras/era4-reasoning.md` 补充叙事线索。

---

## 阶段 3｜部署与效率（2-3 周）

**目的**：理解推理侧的工程经济学。不懂 KV Cache 经济学就不理解为什么 MLA 重要、为什么长上下文这么贵——这一阶段是回过头给前面的架构选择补上"为什么这么设计"的答案。

**任务**：
- KV Cache 工作原理 ⭐⭐
- PagedAttention 设计 ⭐⭐⭐
- LLM 量化的核心难点 ⭐⭐
- AWQ vs GPTQ 对比 ⭐⭐
- 推测解码的正确性证明 ⭐⭐⭐
- Prefill vs Decode 瓶颈差异 ⭐⭐
- 知识蒸馏机制 ⭐⭐
- 张量并行 / 流水线并行 / ZeRO（基础设施 3 个 task）⭐⭐ ~ ⭐⭐⭐
- LoRA 的低秩假设 + QLoRA ⭐⭐ × 2

**达标标志**：完成一道账单题——计算 Llama 3 8B 在 32K context、batch=8 时的显存账单（含权重、KV Cache、激活、optimizer state if 训练）。能解释为什么 vLLM 比 naive serving 快 24x。

**写入位置**：`tracks/inference.md`、`topics/kv-cache.md`、`topics/quantization.md`、`topics/speculative-decoding.md`、`topics/lora-peft.md`。

---

## 阶段 4｜历史叙事 + 能力归因（2 周）

**目的**：到这一步，零件已齐。这两周不学新东西，用 `eras/` 和 `capabilities/` 把零件串起来。这是验证理解的阶段——如果发现某个能力归因写不出来，回去补对应 task。

**做法**：
1. **能力归因**：逐个 review `capabilities/` 下的 7 个文件（reasoning、long-context、multimodality、tool-use、instruction-following、memory、inference-time-compute），每个能力都用一句话回答"它的当前水位是被哪些 tracks 的哪些技术推动的"，把答案补进 capabilities 文件的"技术归因"小节。
2. **范式叙事**：逐个 review `eras/` 下的 5 个文件，确认每个纪元的"主导叙事 → 关键事件 → 范式替代"链条能不能讲通。讲不通的地方，去 `logs/paradigm-shifts.md` 补一条。
3. **labs 视角**：把 `labs/` 下 6 个实验室的"关键决策与赌注"过一遍，对比同一个技术问题（如 MoE vs Dense、RLHF vs DPO、是否做 reasoning 模型）各家的不同选择和结果。

**还需要补的几个 task**：
- GPT-3 Few-Shot 涌现 ⭐⭐
- InstructGPT 1.3B > GPT-3 175B ⭐⭐
- DeepSeek-R1-Zero 的意义 ⭐⭐⭐
- 涌现能力争论（Schaeffer et al.）⭐⭐⭐

**达标标志**：能在 1 小时内向一位资深 ML 工程师做完整 takeaway——"LLM 五年演进的四个纪元、每个纪元的主导矛盾、每个矛盾如何被下一个纪元的技术解决（或转化）"。

---

## 阶段 5｜持续追踪（无止境）

**目的**：主干已通，剩下的内容按兴趣和领域热度挑着学，不强求覆盖。

**topics 池**（剩余 task，按推荐优先级）：

**对齐与安全**（如果工作涉及对齐评估）：
- Reward Hacking ⭐⭐、谄媚问题 ⭐⭐、对齐税及缓解 ⭐⭐、推理链可信度 ⭐⭐⭐、Scalable Oversight ⭐⭐⭐

**多模态**（如果工作涉及 VLM）：
- CLIP 与对比学习 ⭐⭐、ViT ⭐⭐、多模态接合架构 ⭐⭐、视觉语言模型架构 ⭐⭐

**RAG 与评估**（工程相关性高）：
- RAG 架构 ⭐⭐、Embedding 模型 ⭐⭐、RAG vs Long Context ⭐⭐、Lost in the Middle ⭐⭐、Benchmark 设计 ⭐⭐、数据污染 ⭐⭐、评估 Goodhart ⭐⭐

**前沿架构**（关注度高，落地有限）：
- Mamba 选择性 SSM ⭐⭐⭐、Mamba-2 / SSD ⭐⭐⭐、Hybrid 架构原则 ⭐⭐

**可解释性**（前沿研究）：
- 超位置假说 ⭐⭐⭐、稀疏自编码器 SAE ⭐⭐、Activation Steering ⭐⭐

**数据深度**：
- Common Crawl 处理 ⭐、MinHash 去重 ⭐⭐、ML 分类器质量过滤 ⭐⭐、合成数据生成方法 ⭐⭐、Model Collapse ⭐⭐⭐、数据混合比例 ⭐⭐

**FFN 高阶**：FFN 作为键值记忆 ⭐⭐⭐

**节奏建议**：每月 2-4 个 task，配合阅读最新论文和模型发布。从这个阶段开始，study-tasks.md 不再是主要驱动，**新进展归档（`logs/new-arrivals.md` → 体系吸收）成为主线**。

---

## 进度追踪建议

每周末花 30 分钟做 mini-review：
1. 这周完成了哪些 task？写入了哪些文件？
2. 这周的内容能否和上周连成一段叙事？连不上的地方是什么？
3. 下周的 3-5 个 task 是哪些？

每个阶段结束做一次 deep review：
1. 该阶段的"达标标志"是否真的达到？没达到的部分继续补。
2. 在 `logs/paradigm-shifts.md` 检查是否漏记了重要的范式替代。
3. 看 `logs/open-questions.md`——这一阶段是否解答了某些旧问题、产生了哪些新问题。

---

## 更新日志

- 2026-05-03：初版，五阶段路径 + 节奏建议
