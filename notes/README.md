# notes/

学习过程中我自己写的 blog 空间，是 LLM 学习的**驱动单元** — 不是 task 的衍生品，而是反过来组织 task 的容器。

|  | `notes/`（本目录） | `tracks/` `topics/` |
|---|---|---|
| 视角 | 第一人称、学习中的理解 | 第三人称、体系化的事实 |
| 风格 | 自由、可以有困惑/猜测/类比 | 模板化、严谨 |
| 用途 | 学习驱动 + Claude 校验 + Claude 考核 | 长期参考 |
| 谁写 | 我自己 | 我和 Claude 协作 |

## 状态机

每篇 blog 经过四个状态：

```
pending → drafted → verified → mastered
   ↑          ↑          ↑          ↑
 预设      我写完     Claude     Claude
                     校验事实   口试通过
```

1. **pending**：文件已预设（title / deadline / 覆盖 tasks / 核心问题），正文留白
2. **drafted**：我写完一稿（自由发挥，第一人称）
3. **verified**：Claude 校验事实通过，blog 末尾追加校验记录
4. **mastered**：Claude 口试通过，frontmatter 标 `mastered-on`

**只有 mastered 的内容才回写到 `tracks/` `topics/`。** drafted 和 verified 阶段的认知还不稳定，回写早了会污染体系。

## Claude 的两个角色

### 角色 A：校验员

**触发**："校验 notes/xxx.md"

**行为**：
- 读这篇 blog + 涉及的 tracks/topics
- 逐句核对事实准确性，指出错误、遗漏、表述不准
- **不替我改 blog 本身** — 给修改建议，让我自己改
- 校验结论以 review log 形式追加到 blog 末尾，不污染我原文
- 校验完更新 frontmatter `status: drafted → verified`

### 角色 B：考官

**触发**："考核 notes/xxx.md" 或 "考我 #N"

**行为**：
- 读这篇 blog + 相关 tracks/topics
- **不让我看着 blog 答题**（口试性质）
- 设计 5-8 个问题，分三层：
  - **L1 复述层**："X 是什么"（事实记忆）
  - **L2 因果层**："为什么 X 而不是 Y"（理解动机）
  - **L3 迁移层**："如果 Z 不成立，X 还成立吗" / "把 X 套到 W 场景"（能否延伸）
- 一次问一题，逐题评价
- 答错的不放过，深挖到看出是糊涂还是确实没掌握
- **不剧透答案、不放水**
- 走完给评估：
  - **通过**：`status: verified → mastered`，加 `mastered-on: YYYY-MM-DD` + 简评
  - **部分通过**：标 `partial-mastered`，列出薄弱点
  - **不通过**：标 `not-yet`，列出要回看的内容

## 节奏总览

求职冲刺节奏：6 天/周，周日休。整体分三段——**铺面入门 → 选择性深化 → Agent 项目**，对应到 LLM 学习的 17 篇主干（#0 已 mastered）：

```
阶段一·铺面入门  2026-05-22 ~ 2026-06-10  17 篇全部走到 drafted
阶段二·深化      2026-06-11 ~ 2026-07-04  9 篇 mastered + 8 篇 verified
阶段三·Agent 项目 2026-07-06 ~ 2026-07-19  另一项目，本项目暂停
2026-07-20 起     开始投简历
```

**文件名上的 deadline 已失效，以下表格为权威调度。**

### 阶段一：铺面入门（每篇 1 天，目标 drafted）

| # | 排定日 | blog | 备注 |
|---|---|---|---|
| 0 | — | [transformer-foundation](2026-05-17-transformer-foundation.md) | 已 mastered |
| 1 | 05-22 Fri | [position-encoding](2026-05-22-position-encoding.md) | |
| 2 | 05-23 Sat | [attention-engineering](2026-05-23-attention-engineering.md)（稠密） | 已有入门基础，可加快 |
| 3 | 05-25 Mon | [attention-sparse-hybrid](2026-05-25-attention-sparse-hybrid.md) | |
| 4 | 05-26 Tue | [moe-architecture](2026-05-26-moe-architecture.md) | |
| 5 | 05-27 Wed | [sft-and-data](2026-05-27-sft-and-data.md) | |
| 6 | 05-28 Thu | [rlhf-to-dpo](2026-05-28-rlhf-to-dpo.md) | |
| 7 | 05-29 Fri | [scaling-laws-trilogy](2026-05-29-scaling-laws-trilogy.md) | |
| 8 | 05-30 Sat | [rlvr-grpo](2026-05-30-rlvr-grpo.md) | |
| 9 | 06-01 Mon | [reasoning-paradigm-split](2026-06-01-reasoning-paradigm-split.md) | |
| 10 | 06-02 Tue | [inference-time-scaling](2026-06-02-inference-time-scaling.md) | |
| 11 | 06-03 Wed | [kv-cache-economics](2026-06-03-kv-cache-economics.md) | |
| 12 | 06-04 Thu | [quantization-landscape](2026-06-04-quantization-landscape.md) | |
| 13 | 06-05 Fri | [speculative-decoding-distillation](2026-06-05-speculative-decoding-distillation.md) | |
| 14 | 06-06 Sat | [parallelism](2026-06-06-parallelism.md) | |
| 15 | 06-08 Mon | [lora-peft](2026-06-08-lora-peft.md) | |
| 16 | 06-09 Tue | [early-eras-emergence-and-alignment](2026-06-09-early-eras-emergence-and-alignment.md) | |
| 17 | 06-10 Wed | [synthesis-five-years](2026-06-10-synthesis-five-years.md) | |

铺面阶段每篇目标只是 **drafted**（理解概念锚点 + 大致原理，不强求公式推导跟到底）。落后一天可以从次周顺延，落后三天以上要重排。

### 阶段二：深化（每篇 mastered ~2 天 / verified ~0.5 天）

| 篇号 | 目标 | 选中理由 |
|---|---|---|
| 1 | **mastered** | RoPE 推导经典；YaRN/位置插值是长上下文延伸 |
| 2 | **mastered** | FlashAttention IO 递推、MLA、Online Softmax——纯公式区；可顺带 DyT |
| 3 | **mastered** | NSA / MoBA / Hybrid 是 2026 前沿，开始进面试 |
| 4 | **mastered** | DeepSeek-V3 路由 + 负载均衡损失，MoE 已主流 |
| 5 | verified | 数据工程一般不深问 |
| 6 | **mastered** | PPO/DPO 推导经典，且是 RLVR 前置 |
| 7 | **mastered** | Kaplan/Chinchilla 数学是知识地基 |
| 8 | **mastered** | 2025-2026 最大范式转折，GRPO 公式必会 |
| 9 | verified | 综合比较型，#8/#10 master 后这一篇够用 |
| 10 | **mastered** | best-of-N / PRM 是理解 o-series/R1 的钥匙 |
| 11 | **mastered** | Agent 调用经济学，cost/latency 核心 |
| 12 | verified | FP8 知道在用即可 |
| 13 | verified | EAGLE-3 / R1-Distill 知道在用 |
| 14 | verified | 训练并行不是 Agent 岗重点 |
| 15 | verified | LoRA 数学标准化，知道用法即可 |
| 16 | verified | 历史叙事，串故事 |
| 17 | verified | 综合叙事，串故事 |

合计 9 篇 mastered + 8 篇 verified。Mastered 走完整流程（drafted → 我校验 → 你考核），verified 只校验事实。

### 阶段三：Agent 项目（2026-07-06 ~ 07-19）

由另一项目承接，本项目暂停 15 天。期间不安排 LLM 主干推进。

### 2026-07-20 之后

转入持续追踪：未 mastered 的篇章按面试反馈优先级补深；新进展走 [logs/new-arrivals.md](../logs/new-arrivals.md) 流程。被剔除/降级的 task 见 [logs/removed-from-mainline.md](../logs/removed-from-mainline.md)，详细路径说明见 [logs/study-plan.md](../logs/study-plan.md)。

## 命名

`YYYY-MM-DD-{slug}.md`，日期是该篇 deadline。

## 模板

不预设正文模板。每篇 blog 自由组织。预设的只有 frontmatter（title / deadline / status / 覆盖 / 桥接）+ 一句"核心问题"作为综合靶子。
