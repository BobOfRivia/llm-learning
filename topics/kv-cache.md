# KV Cache

> 锚点：`tracks/inference.md` 阶段 1-3 / `tracks/architecture.md`（GQA、MLA）

## 这个概念是什么

KV Cache 是自回归推理的核心工程机制：在逐 token 生成过程中，缓存已计算过的 Key 和 Value 矩阵，避免对每个新 token 重复计算整个上下文的 attention。KV Cache 是一个经典的时间-空间权衡——用内存换计算，但当上下文变长时，内存本身成为瓶颈。几乎所有 LLM 推理优化（GQA、MLA、PagedAttention、KV 分级存储）都在解决 KV Cache 的规模管理问题。

## 内部结构

### KV Cache 存的是什么

> 待推导：
> - 每一层、每一个 attention head 都需要独立的 K 和 V 矩阵
> - 每个 token 贡献一行 K（维度 d_k）和一行 V（维度 d_v）
> - 总 cache 大小 = 2 × layers × heads × seq_len × d_head × bytes_per_element
> - 具体计算：Llama 3 8B（32 层，8 KV heads-GQA，d=128）在 32K context 时的 cache 大小

### 为什么 KV Cache 是瓶颈

> 待推导：
> - cache 大小随序列长度线性增长（百万 token context 的 cache 大小）
> - 在高并发场景下，cache 随 batch size 再线性增长
> - KV Cache 往往比模型权重本身更消耗显存
> - Decode 阶段：每步计算量很小但需要读取全部 cache → 内存带宽瓶颈

### GQA / MQA 的压缩效果

> 待推导：
> - MHA: Q heads = KV heads（全部独立）→ cache 最大
> - MQA: 所有 Q heads 共享 1 组 KV → cache 最小，但质量损失
> - GQA: Q heads 分组共享 KV → 折中方案
> - 具体压缩比：Llama 3 8B 用 8 KV heads vs 32 Q heads → cache 是 MHA 的 1/4

### MLA 的压缩原理

> 待推导：
> - 不存原始 KV，而是存低秩压缩后的"潜在表示"
> - 推理时从潜在表示解压缩恢复 K 和 V
> - 为什么 93% 压缩率在质量上可接受（低秩假设的合理性）
> - 与 GQA 的本质区别：GQA 减少头数，MLA 压缩每个头的维度

### PagedAttention

> 待推导：
> - 传统实现的碎片化问题：不同请求的输出长度不同，预分配导致大量内存浪费
> - 核心思想：把 KV Cache 切成固定大小的"页"（非连续内存），用页表管理
> - 与 OS 虚拟内存的类比：物理内存不连续但逻辑上连续
> - 结合连续批处理（continuous batching）实现 24x 吞吐提升

### KV Cache 分级存储

> 待推导：
> - 推理模型（reasoning）的思维链 tokens 对 cache 的额外压力
> - 分级策略：热数据（近期 tokens）在 GPU HBM，冷数据 offload 到 CPU RAM 或 SSD
> - Token dropping / eviction 策略（H2O 等方法）

## 当前状态（截至 2026-05）

> 待填入：GQA 是主流方案（Llama 3, Mistral），MLA 在 DeepSeek 系列验证，PagedAttention (vLLM) 是 serving 标准。百万 token 上下文的 KV 管理仍是活跃工程问题。

## 关键权衡

> 待填入：
> - 缓存全部 vs 压缩/丢弃：全部缓存保证精确但内存爆炸，压缩引入信息损失
> - 预分配 vs 动态分配：预分配简单但浪费，动态分配节省内存但增加管理开销
> - GPU 内存 vs offload：offload 解决容量但引入带宽瓶颈

## 信息源

- [Efficient Memory Management for Large Language Model Serving with PagedAttention (vLLM)](https://arxiv.org/abs/2309.06180)
- [GQA: Training Generalized Multi-Query Transformer](https://arxiv.org/abs/2305.13245)
- [DeepSeek-V2: MLA](https://arxiv.org/abs/2405.04434)
- [H2O: Heavy-Hitter Oracle for Efficient LLM Inference](https://arxiv.org/abs/2306.14048)

## 更新日志

- 2026-05-03：创建骨架
