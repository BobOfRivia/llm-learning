# LLM 技术演进全景图

## 双索引总览

本体系用两个并列索引组织 LLM 近四年的演进知识。

### 索引 A：生产流水线（技术轨道）

LLM 的知识全景沿一条循环的生产流水线组织：

**基础设施层** — 架构与规模
- [架构演进](tracks/architecture.md)：Transformer 变体 → MoE → SSM 实验 → Hybrid
- [Scaling](tracks/scaling.md)：Scaling Laws → Chinchilla → 过训练 → 推理时扩展
- [数据工程](tracks/data.md)：Web 爬取 → 精选 → 合成数据 → 多模态数据

**加工层** — 训练与对齐
- [训练范式](tracks/training.md)：Pretraining → SFT → RLHF → DPO → RL for Reasoning
- [对齐技术](tracks/alignment.md)：RLHF → Constitutional AI → DPO → 过程监督

**部署层** — 推理与效率
- [推理优化](tracks/inference.md)：量化 → 蒸馏 → 推测解码 → Serving 架构

**输出层** — 能力表现
- [capabilities/](capabilities/)：各能力维度的当前水位与技术归因

**旁轴** — 竞争格局
- [labs/](labs/)：各实验室路线差异与关键决策

### 索引 B：范式纪元

| 纪元 | 时间 | 主导叙事 | 主战场 |
|------|------|----------|--------|
| [Scaling](eras/era1-scaling.md) | 2020-2022 | 大就是好 | 基础设施层 |
| [Alignment](eras/era2-alignment.md) | 2022-2023 | 能力不等于可用 | 加工层 |
| [能力竞赛](eras/era3-capability-race.md) | 2023-2024 | 全面对标人类 | 输出层 |
| [推理](eras/era4-reasoning.md) | 2024-2025 | 让模型思考 | 加工层+部署层 |
| [效率与开放](eras/tension-efficiency-openness.md) | 贯穿 | 持续张力 | 全层 |

### 两个索引的交叉

每个知识点同时挂在两个索引上。例如"MoE 架构"属于 architecture 轨道，在 Scaling 纪元末期 / 能力竞赛纪元初期出现重大进展。tracks/ 文件内部的演进脉络自然标注了时间归属，eras/ 文件引用具体的轨道进展。
