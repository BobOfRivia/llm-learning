# 主干剔除日志

> 这份文档记录从 18 篇 blog 主干**剔除或降级**的内容，以及剔除的理由。
>
> **剔除原则**：本计划只学**已在生产中落地、有真实部署证据**的技术。学术热度不等于落地 — 大量 2023-2025 年论文热点并未真正进入主流模型/serving 栈，对 LLM 演进认知没有实质贡献。
>
> **剔除≠完全不学**：被剔除的内容大部分进入 [study-plan.md](study-plan.md) 阶段 5 的"持续追踪"清单，按兴趣和领域热度挑学。少部分（已被替代的、纯历史价值的）完全剔除。

---

## 2026-05-03 重构剔除清单

基于互联网检索（NSA 部署、EAGLE-3 vLLM 默认、DeepSeek FP8 量产、Anthropic/OpenAI Prompt Caching 商用化、Qwen3-Next Hybrid 落地、DAPO 生产修正等）评估 study-tasks.md 80+ 条 task，决定哪些进主干、哪些降级。

### A. 完全剔除（已被替代或纯历史价值）

| 项 | 原位置 | 剔除原因 |
|---|---|---|
| **MEDUSA** 推测解码 | study-tasks 推测解码隐含 | 已被 EAGLE-3 取代为 vLLM/SGLang/TensorRT-LLM 默认。MEDUSA 在 blog 13 仅作为对比点提一句 |
| **MCTS 用于语言推理** 深入 | study-tasks Inference-time Scaling | 2024 火过一阵（o1 之前业界以为这条路），2025 主流是 long-CoT RL 训练（blog 8）。降级为 blog 10 一段对比说明 |
| **PPO 完整四模型流程** 深入推导 | study-tasks RLHF | 工程上已被 GRPO/DAPO + DPO 模块化栈取代。blog 6 仅讲概念地图（4 模型为何昂贵 → 为何被拆）和"被淘汰的工程原因"，不深入 PPO clip / GAE 数学 |
| **Kaplan Scaling Laws 详细推导** + **Kaplan 误差来源数学** | study-tasks scaling | 历史价值高、生产价值低。blog 7 简化为"修正史"叙事 — 概念上理解 Kaplan 偏差方向即可，不推导 |
| **FFN 作为键值记忆**（Geva 2021） | study-tasks 架构 | 研究框架，对生产没有指导价值。完全剔除 |

### B. 降级到 phase 5 选学（不进主干，但保留为长期追踪）

| 项 | 原位置 | 降级原因 |
|---|---|---|
| **纯 Mamba 选择性 SSM** + **Mamba-2 / SSD 数学统一** | study-tasks 跨轨道 | 落地的是 Hybrid（Gated DeltaNet + Full Attention），不是纯 SSM。Hybrid 落地形态在 blog 3 已覆盖；SSM 数学深入不影响应用层认知 |
| **Hybrid 架构设计原则** 深入 | study-tasks 跨轨道 | blog 3 已覆盖落地形态（3:1 比例 + Qwen3-Next/Nemotron-H 案例），更深入的"最优比例理论"仍是研究阶段，不进主干 |
| **DPO 变体生态**：KTO / IPO / SimPO / ORPO 逐个深入 | study-tasks 训练 | 生产主流仍是 DPO + 偏好对，变体作为索引知道存在即可。blog 6 仅一句索引，不逐个学 |
| **PEFT 其他方法**：Adapter / Prefix Tuning / Prompt Tuning | study-tasks LoRA 节 | LoRA / QLoRA 已是事实标准，其他方法纯学术价值 |
| **CLIP / ViT / 多模态接合架构** | study-tasks 跨轨道 | 多模态不是本项目核心（Agent 项目接口已建立）。原计划就是 phase 5，明确保留 |
| **RAG 全套**：RAG 架构 / Embedding / RAG vs Long Context / Lost in Middle | study-tasks 跨轨道 | 同上，本项目不深入 RAG。Lost in Middle 现象在 blog 11 RULER 段中提到一次即可 |
| **可解释性**：超位置假说 / SAE / Activation Steering | study-tasks 跨轨道 | Anthropic 在用，但不是产品 feature。研究阶段 |
| **数据深度**：Common Crawl 处理 / MinHash / ML 质量过滤 / 合成数据方法 / Model Collapse 数学 / 数据混合比例 | study-tasks 数据 | 重要但偏工程细节，不进主干。blog 5 sft-and-data 触及但不深入 |
| **对齐安全**：Reward Hacking / 谄媚 / 对齐税 / 推理链可信度 / Scalable Oversight | study-tasks 对齐 | 推理链可信度问题在 blog 9 reasoning-paradigm-split 中作为共性问题提一句；其他全归 phase 5 |
| **评估方法论**：Benchmark 设计 / 数据污染 / Goodhart | study-tasks 跨轨道 | 元问题。RULER 在 blog 11 触及；其他归 phase 5 |
| **Computer Use / Browser Use / Agentic 训练** | study-tasks 跨轨道 | **2025 这条线热度被低估**，但本项目重点是 LLM 演进而非 Agent 工程。归 phase 5，做 Agent 时优先回看 |

### C. 主干内降级（保留 task 但缩减 blog 内深度）

| 项 | blog | 缩减方式 |
|---|---|---|
| RLHF 完整流程 | blog 6 | 从"4 模型 + PPO 推导" 缩为"概念地图 + 为什么被拆" |
| Kaplan 详解 | blog 7 | 缩为"修正史" — 概念理解偏差方向，不推导 |
| inference-time MCTS | blog 10 | 降为一段对比说明（解释"为什么没赢 long-CoT"） |
| 知识蒸馏（早期 logit 蒸馏） | blog 13 | 缩为对比点；主线让位给 reasoning distillation（R1-Distill / On-Policy） |

---

## 2026-05-03 重构新增清单

为 balance 剔除，主干新增了**有真实落地证据**的内容：

| 项 | 进入哪个 blog | 落地证据 |
|---|---|---|
| **NSA（Native Sparse Attention）** | blog 3（新建） | DeepSeek 2025-02 论文 → 2025-09 V3.2-Exp 部署 |
| **MoBA（Mixture of Block Attention）** | blog 3 | Moonshot 2025-02，长上下文经济性 |
| **Gated DeltaNet + Hybrid 3:1** | blog 3 | Qwen3-Next 80B-A3B（vLLM 原生支持，2025-09）+ NVIDIA Nemotron-H/Nano 2（Accenture/Cursor/Perplexity 等采购） |
| **三家 MoE 路由对比**（DeepSeek / Qwen3 / Llama 4） | blog 4 | Sebastian Raschka 大模型架构对比 |
| **Multi-Token Prediction（MTP）** | blog 4 | Qwen3-Next 等已部署 |
| **Wide Expert Parallel** | blog 4 + blog 14 | vLLM + llm-d 工程化 |
| **DAPO 四项修正** | blog 8 | ByteDance/Tsinghua 2025，long-CoT 生产修正 |
| **Anthropic / Claude 4 hybrid thinking model** | blog 9 | Claude 3.7 / 4 实际产品形态 |
| **DeepSeek R1-Distill 系列实战** | blog 9 + blog 13 | R1-Distill-Qwen-32B > o1-mini 实测 |
| **On-Policy Distillation** | blog 13 | Thinking Machines 2025 |
| **Prompt Caching** | blog 11 | Anthropic 显式 cache_control（GA）+ OpenAI 隐式自动 |
| **长上下文真实退化（RULER vs NIAH）** | blog 11 | Chroma 研究 + NVIDIA RULER 已成事实标准 |
| **FP8 作为训练+推理统一精度** | blog 12 | DeepSeek V3 首次大规模成功 + Qwen 3.5 native FP8 pipeline |
| **EAGLE-3 推测解码事实标准** | blog 13 | vLLM/SGLang/TensorRT-LLM 默认 |

---

## 后续维护

如果未来某个被剔除的项**真正落地**（出现 frontier 模型部署、被 vLLM/SGLang/HF 等主流工具采纳、有可验证生产数据），从 phase 5 提升回主干。在这份文档底部追加一条"YYYY-MM-DD 重新评估"记录。

如果某个主干项**热度消退**（被新方法取代），也在这里记录降级。
