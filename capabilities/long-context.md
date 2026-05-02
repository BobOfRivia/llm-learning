# Long Context

> 桥接：Agent 项目 dimensions/02-long-context.md

## 当前水位（截至 2026-05）

<!-- 待填充：主流模型的 context window 大小及实际有效长度 -->

## 技术归因

- **architecture**：Flash Attention、GQA/MQA 是长上下文的工程基础（→ [架构演进](../tracks/architecture.md)）
- **inference**：KV Cache 管理是长上下文推理的核心瓶颈（→ [推理优化](../tracks/inference.md)）
- **training**：长文本训练数据和位置编码方案（RoPE 外推）

## 演进轨迹

- **2020-2022**：2K-4K tokens 为主流
- **2023**：Claude 100K、GPT-4 Turbo 128K，context window 竞赛开始
- **2024**：Gemini 1M+，context window 继续扩展
- **2025-至今**：百万级 context 可用，关注点转向有效利用率

## 更新日志

- 2026-05-02：创建骨架
