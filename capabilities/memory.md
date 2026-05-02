# Memory

> 桥接：Agent 项目 dimensions/06-memory.md

## 当前水位（截至 2026-05）

<!-- 待填充 -->

## 技术归因

- **architecture**：context window 长度决定了"工作记忆"上限（→ [架构演进](../tracks/architecture.md)）
- **training**：是否专门训练过记忆和检索能力
- **inference**：KV cache 和上下文管理技术（→ [推理优化](../tracks/inference.md)）

## 演进轨迹

- **2020-2023**：纯 in-context memory，受限于 context window
- **2024**：长上下文模型缓解了部分 memory 问题，但不等于真正的 memory
- **2025-至今**：模型层面的 memory 能力是否在涌现？

## 更新日志

- 2026-05-02：创建骨架
