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

主干 18 篇 × 1 周/篇，从 2026-05-03 起，到 2026-09-06 结束。每周日是 deadline。允许动态调整：超前完成可以提前下一篇；落后可以延期，但累计落后超过 2 周需要重新评估覆盖范围。

| # | deadline | blog | 阶段 |
|---|---|---|---|
| 0 | 2026-05-10 | [transformer-foundation](2026-05-10-transformer-foundation.md) | 0 地基 |
| 1 | 2026-05-17 | [position-encoding](2026-05-17-position-encoding.md) | 1 三主干 |
| 2 | 2026-05-24 | [attention-engineering](2026-05-24-attention-engineering.md)（稠密） | 1 三主干 |
| 3 | 2026-05-31 | [attention-sparse-hybrid](2026-05-31-attention-sparse-hybrid.md)（稀疏 / Hybrid） | 1 三主干 |
| 4 | 2026-06-07 | [moe-architecture](2026-06-07-moe-architecture.md) | 1 三主干 |
| 5 | 2026-06-14 | [sft-and-data](2026-06-14-sft-and-data.md) | 1 三主干 |
| 6 | 2026-06-21 | [rlhf-to-dpo](2026-06-21-rlhf-to-dpo.md) | 1 三主干 |
| 7 | 2026-06-28 | [scaling-laws-trilogy](2026-06-28-scaling-laws-trilogy.md) | 1 三主干 |
| 8 | 2026-07-05 | [rlvr-grpo](2026-07-05-rlvr-grpo.md)（含 DAPO） | 2 推理纪元 |
| 9 | 2026-07-12 | [reasoning-paradigm-split](2026-07-12-reasoning-paradigm-split.md)（三家分叉） | 2 推理纪元 |
| 10 | 2026-07-19 | [inference-time-scaling](2026-07-19-inference-time-scaling.md) | 2 推理纪元 |
| 11 | 2026-07-26 | [kv-cache-economics](2026-07-26-kv-cache-economics.md)（含 Prompt Caching + RULER） | 3 部署效率 |
| 12 | 2026-08-02 | [quantization-landscape](2026-08-02-quantization-landscape.md)（FP8 主线） | 3 部署效率 |
| 13 | 2026-08-09 | [speculative-decoding-distillation](2026-08-09-speculative-decoding-distillation.md)（EAGLE-3 + R1-Distill） | 3 部署效率 |
| 14 | 2026-08-16 | [parallelism](2026-08-16-parallelism.md)（含 Wide EP） | 3 部署效率 |
| 15 | 2026-08-23 | [lora-peft](2026-08-23-lora-peft.md) | 3 部署效率 |
| 16 | 2026-08-30 | [early-eras-emergence-and-alignment](2026-08-30-early-eras-emergence-and-alignment.md) | 4 综合 |
| 17 | 2026-09-06 | [synthesis-five-years](2026-09-06-synthesis-five-years.md) | 4 综合 |

主干完成后转入持续追踪，按兴趣和领域热度挑 task，不再有固定节奏。被剔除/降级的 task 见 [logs/removed-from-mainline.md](../logs/removed-from-mainline.md)。详细阶段说明见 [logs/study-plan.md](../logs/study-plan.md) 阶段 5。

## 命名

`YYYY-MM-DD-{slug}.md`，日期是该篇 deadline。

## 模板

不预设正文模板。每篇 blog 自由组织。预设的只有 frontmatter（title / deadline / status / 覆盖 / 桥接）+ 一句"核心问题"作为综合靶子。
