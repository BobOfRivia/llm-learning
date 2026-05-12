# 量化全景（FP8 为主线）

> deadline: 2026-08-09
> status: pending
> 覆盖 tasks: FP8 作为训练+推理统一精度、LLM 量化的核心难点（outlier 处理）、AWQ / GPTQ / SmoothQuant 对比、GGUF 与边缘部署、PTQ vs QAT
> 桥接: tracks/inference.md, topics/quantization.md

**核心问题**：量化在 2024-2025 发生了**地位重组** — 不再是"为了塞进内存的事后压缩"，而是 FP8 升格为"训练 + 推理统一精度"的 baseline，其他方案围绕"FP8 不够时怎么办"展开。要能按这条主线讲清：

1. **FP8 为什么是主线（不是 INT8）**。DeepSeek V3 首次大规模 FP8 训练成功（2024 末）— **第一次有 frontier 模型在 FP8 下训完不掉点**；Qwen 3.5 native FP8 pipeline，解码 8.6-19× 加速、激活内存 -50%；H100 / B200 原生支持 FP8 Tensor Core。要能解释：**E4M3 vs E5M2 两种格式的定位**（前者偏精度、后者偏动态范围，分别用于前向/反向）；为什么 BF16 在训练中胜过 FP16 但 FP8 现在又能挑战 BF16；FP8 不只是"更小的 INT8"，本质是"硬件原生的效率甜点"。
2. **outlier 是核心难点（FP8 不够时才上的方案）**。激活异常值集中在少数 channel，标准 per-tensor 量化会被它们拉垮。各家方案的核心区别是**怎么定位和处理 outlier**：
   - **GPTQ**：Hessian 逆做逐列误差补偿（事后修复）
   - **AWQ**：用激活幅度找 1% 显著权重保留高精度（事前保护）
   - **SmoothQuant**：per-channel scaling 把激活异常值"搬运"到权重（权重静态更好处理）
   能讲清三种思路在 design space 中的位置（事后 vs 事前 vs 难度转移）。
3. **GGUF + 边缘部署**：解决的是格式 + 工程问题（混合精度、SIMD 优化、内存映射），不是量化算法本身的创新。Q4_K_M 等格式如何让 llama.cpp 在 CPU 上达到可用速度。
4. **PTQ vs QAT 的边界**：超大 LLM（>70B）几乎只能 PTQ；QAT 在中小模型 + 边缘场景仍有价值；2025 Fireworks 等开始把 QAT 推到 Llama/Qwen FP8 推理路径，但还在路上。

**判断主线**：把量化看成**两层堆栈** — FP8 作为基础（硬件原生），outlier 处理（GPTQ/AWQ/SmoothQuant）作为针对特定瓶颈的补丁。这个 framing 比"量化方案大全"更有用，能解释为什么 2025 之后大量旧 INT4/INT8 论文已经不重要。

---


---
