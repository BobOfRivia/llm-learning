# 并行训练与 Wide Expert Parallel

> deadline: 2026-08-23
> status: pending
> 覆盖 tasks: 张量并行（TP）、流水线并行（PP）、ZeRO（数据并行的内存优化）、Wide Expert Parallel（MoE 服务专属）
> 桥接: tracks/training.md, tracks/inference.md

**核心问题**：三种经典并行（TP / PP / ZeRO）解决的瓶颈不同（计算 / 显存 / 通信），何时该用哪种、组合到 3D/4D 时边界在哪？同时 2025 年 MoE 走到 256 专家级别后，**Expert Parallel（EP）变成了第四个独立维度** — 必须单独理解。要能讲清：

1. **三种经典并行的本质区别**：
   - **TP**：同一矩阵切到多 GPU，每步都要 AllReduce → 对 NVLink 极敏感（节点内 OK、跨节点惨）
   - **PP**：模型按层切，气泡（bubble）问题 + 微批数量是关键变量
   - **ZeRO-1/2/3**：分别切 optimizer state / gradient / parameter，用通信换显存。ZeRO-3 是大模型 SFT 的默认选择
2. **3D / 4D 组合的边界**：DP × TP × PP × (EP) 的设计空间 — 通常 TP 锁在节点内（NVLink），PP 跨节点，DP 顶层。计算实际场景下的最优组合（给定模型大小 + 集群拓扑）。
3. **Wide Expert Parallel（2025 新增）**：DeepSeek 那种 256 专家的 MoE，必须把专家分片到大量 GPU 上 — 这就是 Wide EP。vLLM + llm-d 把它工程化。要能解释：**为什么 EP 不能简单等同于 TP**（专家路由是稀疏的，all-to-all 通信模式独特）、wide EP 的 latency 来自哪里（all-to-all + load imbalance）、和 TP 的取舍。
4. **训练 vs 推理的并行差异**：训练强调吞吐 + 稳定性（bubble 可接受），推理强调 latency（bubble 不可接受）。所以推理基本不用 PP，但训练大模型几乎一定用。

**判断主线**：并行不是"选一种用"，是**给定硬件拓扑求一组系数**的优化问题。看懂三种经典 + EP 这第四维，再看新框架（Megatron / DeepSpeed / vLLM EP）就是在这个 design space 里挑点。

---
