# Reasoning

> 桥接：Agent 项目 dimensions/01-reasoning.md

## 当前水位（截至 2026-05）

<!-- 待填充：关键 benchmark（MATH, GPQA, ARC-AGI 等）及代表性模型表现 -->

## 技术归因

这个能力的演进主要由以下轨道推动：

- **training**：RL for Reasoning 是质变的直接原因（→ [训练范式 - 阶段 4](../tracks/training.md)）
- **scaling**：推理时扩展让模型"想更久"（→ [Scaling - 阶段 4](../tracks/scaling.md)）
- **data**：数学和代码数据的比例影响 reasoning 基线（→ [数据工程](../tracks/data.md)）

## 演进轨迹

- **2020-2022**：few-shot prompting + chain-of-thought，reasoning 靠 prompt 技巧"诱导"
- **2023**：GPT-4 级别，reasoning 显著提升但仍不稳定
- **2024**：o1/o3 系列，reasoning 成为专门训练的能力维度
- **2025-至今**：reasoning 模型成为默认，非 reasoning 模型定位为"快速模式"

## 更新日志

- 2026-05-02：创建骨架
