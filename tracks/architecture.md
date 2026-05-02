# 架构演进

## 一句话定义

这条轨道覆盖 LLM 的模型架构选择——从基础的 Transformer 到各种变体、替代方案和混合架构，追踪"用什么结构来建模序列"这个核心问题的演进。

## 演进脉络

### 阶段 1：Transformer 统一（2017-2020）

<!-- Attention Is All You Need 之后，Transformer 迅速统一 NLP。
     decoder-only (GPT) vs encoder-decoder (T5) vs encoder-only (BERT) 的路线分化。
     decoder-only 最终胜出成为 LLM 主流，原因待展开。 -->

### 阶段 2：规模驱动的架构稳定（2020-2022）

<!-- GPT-3 证明 dense Transformer 可以靠规模获得涌现能力。
     这个阶段架构创新相对停滞，主旋律是"把同一个架构做大"。
     关键问题：为什么不是架构创新而是规模扩展成为主要方向？ -->

### 阶段 3：效率压力下的架构分化（2022-2024）

<!-- 三条路线并行：
     1. MoE (Mixture of Experts)：GShard → Switch Transformer → Mixtral → GPT-4(?)
     2. SSM (State Space Models)：S4 → Mamba → Mamba-2
     3. 高效 Attention 变体：Flash Attention、GQA、MQA
     驱动力：Dense 模型的训练和推理成本撞墙。 -->

### 阶段 4：混合架构与收敛（2024-至今）

<!-- Hybrid 架构出现：Jamba (Mamba + Transformer)
     MoE 成为闭源模型的事实标准？
     SSM 的定位从"替代 Transformer"转向"互补组件"。 -->

## 当前技术格局（截至 2026-05）

<!-- 待填充 -->

## 关键分歧与未决问题

<!-- 
- MoE vs Dense：什么场景下 MoE 的优势是决定性的？
- SSM 能否在长序列任务上真正替代 Attention？
- 架构创新还有多大空间，还是已经收敛？
-->

## 对能力输出的影响

<!-- MoE → 推理成本降低，模型规模上限提高
     Flash Attention → 长上下文能力的工程基础
     SSM → 长序列处理效率 -->

## 与其他轨道的交叉

- **scaling**：架构选择直接影响 scaling 效率
- **inference**：MoE 和 SSM 的推理特性完全不同
- **training**：某些架构更难训练（MoE 的负载均衡问题）

## 信息源

<!-- 待补充 -->

## 更新日志

- 2026-05-02：创建骨架
