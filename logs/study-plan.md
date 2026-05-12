# LLM 学习路径

> 这份文档给出**学习的顺序、节奏和达标标志**，与以下两份文档配合使用：
>
> - [study-tasks.md](study-tasks.md)：知识点的微粒清单 — 支撑材料
> - [notes/](../notes/)：18 篇 blog milestone — **本计划的实际驱动单元**
> - [removed-from-mainline.md](removed-from-mainline.md)：从主干剔除的内容及理由
>
> 起始日期：2026-05-03 ｜ 主干完成：2026-09-13 ｜ 之后转入持续追踪模式
>
> **2026-05-03 重构**：基于互联网检索 + 落地证据，剔除未真正生产验证的技术，新增 2024-2025 真实落地的关键技术（NSA 稀疏注意力、Hybrid 注意力、Prompt Caching、DAPO、FP8 训练统一、EAGLE-3、R1-Distill 等）。详见各 blog 与剔除日志。

---

## 元原则

**主干学习的驱动单元是 notes/ 下的 18 篇 blog，不是 task。** 每篇 blog 覆盖 3-6 个相关 task，强制做综合 — 避免学完 task 后只剩离散要点。完整流程：

```
学覆盖的 tasks（4-5 天）
  → 在 notes/ 下写 blog（2-3 天）
  → "校验 notes/xxx.md" → Claude 校验事实，我修订
  → "考核 notes/xxx.md" → Claude 口试评估掌握度
  → mastered 后回写到 tracks/ 或 topics/
```

详细工作流和 Claude 的两个角色（校验员 / 考官）见 [notes/README.md](../notes/README.md)。

**学习不是覆盖任务，是打通因果链。** 如果一篇 blog 走到 mastered 时还讲不出该阶段的因果链，回去补对应 task。

**只学已落地的技术。** 学术热度不等于生产落地。本计划剔除了"看起来很热但没有真实部署证据"的技术（纯 Mamba 数学、MEDUSA、大量 DPO 变体、MCTS 推理、SAE/可解释性、Common Crawl 处理细节等）— 这些归 phase 5 选学或完全剔除。剔除清单见 [removed-from-mainline.md](removed-from-mainline.md)。

**跳过公式推导，但不跳过直觉。** 不重新推 DPO 化简、不推 Kaplan 误差源、不推 PPO clip — 但要能说清"DPO 把 RL 转化为分类问题，前提是数据离线"这种 one-liner。所有 ⭐⭐⭐ task 的目标都是抓直觉，不是推导熟练度。

**节奏**：每周一篇，周日 deadline。允许动态调整 — 提前完成可顺延下一篇；累计落后超过 2 周需重新评估覆盖范围。

---

## 阶段 0｜地基（1 周）

**核心叙事**：建立 Transformer 的 mental model。这五件事是后续所有内容的 prerequisite，地基不牢则后面学 RoPE、MLA、RLHF 都是死记硬背。

| # | deadline | blog | 覆盖 task |
|---|---|---|---|
| 0 | 2026-05-17 | [transformer-foundation](../notes/2026-05-17-transformer-foundation.md) | BPE, Self-Attention, Multi-Head, Transformer 层结构, AR 目标, decoder-only |

**该阶段达标**：blog 0 通过考核（mastered）。

---

## 阶段 1｜三主干并行（7 周）

**核心叙事**：架构、训练、scaling 三条主干在 2020-2025 年缠绕演进。**架构这条主干在 2025 出现新分叉**（稠密注意力的极限 → 稀疏 / 线性 / Hybrid 三条新路线落地），所以架构展开为 3 篇而非 2 篇。

| # | deadline | blog | 主线 |
|---|---|---|---|
| 1 | 2026-05-24 | [position-encoding](../notes/2026-05-24-position-encoding.md) | 架构 |
| 2 | 2026-05-31 | [attention-engineering](../notes/2026-05-31-attention-engineering.md)（稠密：Flash + GQA + MLA） | 架构 |
| **3** | 2026-06-07 | [attention-sparse-hybrid](../notes/2026-06-07-attention-sparse-hybrid.md)（**新**：NSA + MoBA + Gated DeltaNet + Hybrid 3:1） | 架构 |
| 4 | 2026-06-14 | [moe-architecture](../notes/2026-06-14-moe-architecture.md)（强化三家对比 + MTP + Wide EP 引子） | 架构 |
| 5 | 2026-06-21 | [sft-and-data](../notes/2026-06-21-sft-and-data.md) | 训练 |
| 6 | 2026-06-28 | [rlhf-to-dpo](../notes/2026-06-28-rlhf-to-dpo.md)（PPO 简化为"为何被淘汰"） | 训练 |
| 7 | 2026-07-05 | [scaling-laws-trilogy](../notes/2026-07-05-scaling-laws-trilogy.md)(Kaplan 简化为修正史) | scaling |

**该阶段达标**：blog 1-7 全部 mastered，且能用 5 分钟分别串讲四个问题：
1. 稠密 → 稀疏 / Hybrid 这一步的经济驱动力
2. 三家 MoE（DeepSeek / Qwen3 / Llama 4）路由选择背后的 tradeoff
3. 为什么 2024 年过训练成主流
4. RLHF→DPO 切换的代价与收益（不要求推 PPO clip / DPO 化简）

---

## 阶段 2｜推理纪元（3 周）

**核心叙事**：2024-2025 最大的范式断裂 — 从 RLHF 到 RL on verifiable reward，且 2025 年裂出**三条产品路线**（OpenAI dedicated / Anthropic hybrid / DeepSeek 开源蒸馏）。这一阶段最容易"看新闻看个表面"，所以单独成阶段，强迫深入。

| # | deadline | blog |
|---|---|---|
| 8 | 2026-07-12 | [rlvr-grpo](../notes/2026-07-12-rlvr-grpo.md)（**加 DAPO 四项修正**） |
| 9 | 2026-07-19 | [reasoning-paradigm-split](../notes/2026-07-19-reasoning-paradigm-split.md)（**改名**：三家路线对比，不再是 R1+o1 二元） |
| 10 | 2026-07-26 | [inference-time-scaling](../notes/2026-07-26-inference-time-scaling.md)（MCTS 降级，重点 long-CoT 内化） |

**该阶段达标**：blog 8-10 全部 mastered。第 10 篇通过后，能用 1500 字写完《从 RLHF 到三家分叉：训练范式三年演进》memo，归档到 `topics/rl-for-reasoning.md`。

---

## 阶段 3｜部署与效率（5 周）

**核心叙事**：推理与训练的工程经济学，回过头补"前面架构选择为什么这么设计"的答案。不懂 KV Cache 经济学就不理解为什么 MLA 重要、为什么长上下文这么贵、为什么 Prompt Caching 是商业一等公民。

| # | deadline | blog |
|---|---|---|
| 11 | 2026-08-02 | [kv-cache-economics](../notes/2026-08-02-kv-cache-economics.md)（**加 Prompt Caching + RULER 长上下文真实退化**） |
| 12 | 2026-08-09 | [quantization-landscape](../notes/2026-08-09-quantization-landscape.md)（**FP8 升为主线**） |
| 13 | 2026-08-16 | [speculative-decoding-distillation](../notes/2026-08-16-speculative-decoding-distillation.md)（**EAGLE-3 + R1-Distill + On-Policy**） |
| 14 | 2026-08-23 | [parallelism](../notes/2026-08-23-parallelism.md)（**加 Wide Expert Parallel**） |
| 15 | 2026-08-30 | [lora-peft](../notes/2026-08-30-lora-peft.md) |

**该阶段达标**：blog 11-15 全部 mastered，且能完成两道账单题：
1. Llama 3 8B 在 32K context、batch=8 时的显存账单（含权重、KV Cache、激活、optimizer state if 训练）
2. 一个 system prompt 长 5000 token、用户消息平均 200 token 的 agent 应用，启用 Anthropic Prompt Caching 后单次请求成本曲线（cache hit / miss 两种情况）

并能解释 vLLM 比 naive serving 快 24× 的原因 + EAGLE-3 在 vLLM 默认的工程动机。

---

## 阶段 4｜历史叙事综合（2 周）

**核心叙事**：到这一步零件已齐。这两周不学新东西，把零件串成完整故事。这是验证理解的阶段 — 如果某个能力归因写不出来，回去补对应 task。

| # | deadline | blog |
|---|---|---|
| 16 | 2026-09-06 | [early-eras-emergence-and-alignment](../notes/2026-09-06-early-eras-emergence-and-alignment.md) |
| 17 | 2026-09-13 | [synthesis-five-years](../notes/2026-09-13-synthesis-five-years.md)（**加入两条范式分叉作为高潮**） |

**该阶段达标**：blog 17 通过考核。考核形式升级 — 不仅口试，做一次完整 60 分钟口述演练并录音，回听找漏洞。综合叙事必须能讲清两条分叉（架构分叉 + 推理产品分叉）的因果。

**配套动作（穿插在这两周）**：
1. **能力归因**：逐个 review `capabilities/` 下 7 个文件，每个能力都用一句话回答"它的当前水位是被哪些 tracks 的哪些技术推动的"，补进 capabilities 文件的"技术归因"小节
2. **范式叙事**：逐个 review `eras/` 下 5 个文件，确认每个纪元的"主导叙事 → 关键事件 → 范式替代"链条能讲通；讲不通的去 `logs/paradigm-shifts.md` 补一条
3. **labs 视角**：把 `labs/` 下 6 个实验室的"关键决策与赌注"过一遍，对比同一个技术问题各家的不同选择

---

## 阶段 5｜持续追踪（无止境）

主干完成后，study-tasks.md 中剩余 task 按兴趣和领域热度挑着学，不强求覆盖。**这里只列"剔除主干、转 phase 5 选学"的部分** — 不是没价值，是不达"已落地且影响生产"标准：

- **架构前沿（学术活跃）**：纯 Mamba / SSD 数学、Hybrid 设计原则深入（落地形态在 blog 3 已覆盖）
- **对齐与安全**：Reward Hacking、谄媚问题、对齐税及缓解、推理链可信度、Scalable Oversight
- **多模态**：CLIP 与对比学习、ViT、多模态接合架构、视觉语言模型架构
- **RAG 与评估**：RAG 架构、Embedding 模型、RAG vs Long Context、Lost in the Middle、Benchmark 设计、数据污染、评估 Goodhart
- **可解释性**：超位置假说、稀疏自编码器 SAE、Activation Steering（Anthropic 在用，但不是产品 feature）
- **数据深度**：Common Crawl 处理、MinHash 去重、ML 分类器质量过滤、合成数据生成方法、Model Collapse 数学、数据混合比例
- **DPO 变体生态**：KTO / IPO / SimPO / ORPO（生产仍以 DPO + 偏好对为主，变体作为索引知道存在即可）
- **PEFT 其他方法**：Adapter / Prefix Tuning / Prompt Tuning（LoRA / QLoRA 已是事实标准）
- **Agentic 训练**：Computer Use、Browser Use、Operator — **这条 2025 热度被低估**，做 Agent 项目时优先回看
- **FFN 高阶**：FFN 作为键值记忆（研究框架，无生产指导）

**节奏**：每月 2-4 个 task，配合阅读最新论文和模型发布。这个阶段开始，**新进展归档**（`logs/new-arrivals.md` → 体系吸收）成为主线。仍可按需创建 notes/ blog，但不再有固定 deadline。

---

## 进度追踪

每周末 30 分钟 mini-review：
1. 本周 blog 走到了哪个状态（drafted / verified / mastered）？卡在哪一步？
2. 下周 blog 的覆盖 task 列表过一遍，估算需要时间
3. 落后超 1 周：分析原因（覆盖太广 / 个人时间 / 概念太难），及时调整

每个阶段结束 deep review：
1. 该阶段所有 blog 是否 mastered？没 mastered 的优先补
2. `logs/paradigm-shifts.md` 是否漏记了重要的范式替代
3. `logs/open-questions.md` — 这一阶段解答了哪些旧问题、产生了哪些新问题

---

## 更新日志

- 2026-05-03：初版，五阶段路径 + 节奏建议
- 2026-05-03：新增 notes/ blog 工作流（task → notes → 校验 → 回写）
- 2026-05-03：重构为 note-driven，17 篇 blog 作为驱动单元，新增"考官"角色
- 2026-05-03：基于互联网检索 + 落地证据二次重构 — 17 → 18 篇，主干完成日 08-30 → 09-06 顺延一周；拆 attention-engineering 为稠密 + 稀疏/Hybrid 两篇；blog 9 改名 reasoning-paradigm-split 覆盖三家分叉；blog 11-14 各加 2025 落地新内容（Prompt Caching、FP8 主线、EAGLE-3、Wide EP）；剔除清单单独维护在 removed-from-mainline.md
- 2026-05-12：blog 0 deadline 已过 2 天仍 pending，整体平移一周 — 全部 18 篇 deadline +7 天，主干完成日 09-06 → 09-13；文件重命名 + frontmatter deadline 同步更新
