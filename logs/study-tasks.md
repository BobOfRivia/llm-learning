# 学习任务清单

> 知识体系的**微粒学习地图**。每条任务对应一个值得深入推理的关键知识点——tracks/ 中有概括性描述，但真正理解需要自己推导因果链条。
>
> - **[ ]** = 待学习
> - **[x]** = 已完成（完成后在 tracks/ 或 topics/ 对应位置补充细节）
> - **锚点** = 指向现有体系位置
> - 难度：⭐ 基础 / ⭐⭐ 进阶 / ⭐⭐⭐ 需要推导
>
> ## 2026-05-03 重构后阅读说明
>
> 本清单是 task 池，不再是学习驱动。**实际学习按 [study-plan.md](study-plan.md) 的 18 篇 blog 走** — blog 已经吸收了 task 并加上了 2025 新落地内容。
>
> - **进入主干 18 篇 blog 的 task**：直接通过 blog 学习，本清单作为参考
> - **被降级到 phase 5 / 完全剔除的 task**：见 [removed-from-mainline.md](removed-from-mainline.md)。本文件中**未单独标记剔除**，避免大规模改动出错；以剔除日志为准
> - **2026-05-03 新增的 task**：见本文件底部"重构新增 tasks"段

---

## 架构轨道

### Transformer 基础

- [ ] **⭐⭐ Self-Attention 机制**：QKV 矩阵的来源和含义、点积 scaling（为什么除以 √d_k）、softmax 归一化的作用
  - 目标：能从头写出 Attention(Q,K,V) = softmax(QK^T/√d_k)V，解释每一步的设计理由
  - 锚点：`tracks/architecture.md` 阶段 1 → `topics/attention-mechanism.md`

- [ ] **⭐⭐ Multi-Head Attention**：为什么分多头比单头更好、不同头实证上学到了什么不同模式、输出拼接后投影的意义
  - 目标：用具体例子说明不同头可能关注什么（句法 / 语义 / 位置近邻）
  - 锚点：`topics/attention-mechanism.md`

- [ ] **⭐⭐ Transformer 层结构**：Pre-LN vs Post-LN 的训练稳定性差异、FFN 的计算量占比和功能、残差连接的梯度流意义
  - 目标：解释为什么现代模型全用 Pre-LN，以及 FFN 相对于 Attention 承担什么功能
  - 锚点：`topics/attention-mechanism.md`

- [ ] **⭐⭐⭐ FFN 作为键值记忆**（Geva et al., 2021）：上投影行 = keys、下投影列 = values 的解释框架，以及这个视角的局限
  - 目标：理解 Transformer 存储事实知识的可能机制
  - 锚点：`topics/attention-mechanism.md`

- [ ] **⭐ decoder-only 为什么统治**：不只是技术优越性，而是 scaling 路径上的收益差异和路径依赖
  - 目标：给出一个反事实分析——如果 T5 路线获得同等投入会怎样
  - 锚点：`tracks/architecture.md` 阶段 1（已有分析，可深化）

### 位置编码

- [ ] **⭐⭐ 绝对位置编码的局限**：正弦编码的设计逻辑、为什么无法泛化到训练长度之外
  - 目标：从数学上说明为什么绝对位置在外推时失效
  - 锚点：`tracks/architecture.md` 阶段 3

- [ ] **⭐⭐⭐ RoPE（旋转位置编码）**：旋转变换如何编码相对位置、内积只依赖相对距离的数学证明、外推能力的来源
  - 目标：从旋转矩阵 × 向量的角度推导 RoPE 的核心公式
  - 锚点：`tracks/architecture.md` 阶段 3 → arXiv 2104.09864

- [ ] **⭐⭐ 上下文长度外推**：YaRN 和 NTK-aware scaling 的原理、频率插值的直觉
  - 目标：用 RoPE 的频率成分解释 NTK scaling 为什么能扩展上下文
  - 锚点：`tracks/architecture.md` 阶段 3

### 高效 Attention

- [ ] **⭐⭐⭐ Flash Attention**：标准实现的 IO 瓶颈、分块计算如何避免 N×N 矩阵写入 HBM、在线 softmax 技巧
  - 目标：解释为什么 Flash Attention 是精确等价而非近似，性能提升具体来自哪里
  - 锚点：`tracks/architecture.md` 阶段 3 → arXiv 2205.14135

- [ ] **⭐⭐ GQA / MQA**：MHA → MQA → GQA 的演进逻辑、KV 头减少对质量的影响曲线
  - 目标：计算 GQA 在长上下文时 KV cache 节省的具体数量级
  - 锚点：`tracks/architecture.md` 阶段 3 → `topics/kv-cache.md`

- [ ] **⭐⭐⭐ MLA（多头潜在注意力）**：低秩 KV 投影的机制、缓存压缩后表示而非原始 KV 的原理
  - 目标：理解 93% cache 压缩的数学来源，以及 MLA 与 GQA 的本质区别
  - 锚点：`tracks/architecture.md` 当前格局 → arXiv 2405.04434

### MoE 架构

- [ ] **⭐⭐ Expert 路由机制**：router 的 softmax 得分计算、top-K 选择、capacity factor
  - 目标：从代码层面理解一个 token 如何被路由到 expert
  - 锚点：`topics/moe.md` 基本机制

- [ ] **⭐⭐ Expert 专门化的实证**：experts 实际专门化的粒度（句法模式 vs 话题）
  - 目标：用 LibMoE 等实验数据纠正"每个 expert 负责一个领域"的过度简化
  - 锚点：`topics/moe.md` 当前状态

- [ ] **⭐⭐⭐ Auxiliary-loss-free 负载均衡**：DeepSeek V3 的动态 bias 方案如何与梯度计算分离
  - 目标：理解传统辅助 loss 的干扰机制，以及 bias 方案为什么能避免
  - 锚点：`topics/moe.md` 负载均衡

### SSM 与混合架构

- [ ] **⭐⭐⭐ Mamba 的选择性状态空间**：S4 固定转移 vs Mamba 输入依赖转移、为什么选择性是关键
  - 目标：用一个直觉例子说明固定 SSM 和选择性 SSM 对"记住什么"的不同行为
  - 锚点：`tracks/architecture.md` 阶段 3 → `topics/mamba-ssm.md`

- [ ] **⭐⭐⭐ Mamba-2 / SSD 框架**：SSM 和 Attention 的数学统一、这个统一对并行化的实践意义
  - 目标：理解 SSD 框架核心定理的含义，以及 Mamba-2 为什么比 Mamba-1 快 2-8x
  - 锚点：`tracks/architecture.md` 阶段 4 → arXiv 2405.21060

- [ ] **⭐⭐ Hybrid 架构设计原则**：Attention 层和 SSM 层各自适合处理什么、最优比例的决定因素
  - 目标：解释 Nemotron-H 为什么只用 8% Attention 就够了
  - 锚点：`tracks/architecture.md` 阶段 4

---

## 训练轨道

### 预训练

- [ ] **⭐ 自回归语言建模目标**：为什么"预测下一个 token"作为目标足够表达所有语言能力
  - 目标：从"预测下一个 token 需要理解什么"出发论证这个目标的充分性
  - 锚点：`tracks/training.md` 阶段 1

- [ ] **⭐⭐ 代码数据对推理能力的影响**：代码为什么能提升通用推理（形式逻辑、调试模式、结构化输出）
  - 目标：给出两到三个机制假说并用实验证据评估
  - 锚点：`tracks/data.md` → `tracks/training.md`

### SFT 与 RLHF

- [ ] **⭐⭐ SFT 的数据工程**：instruction-input-output 格式、多样性 vs 质量权衡、SFT 能做什么 / 不能做什么
  - 目标：理解为什么少量高质量 SFT 数据可能胜过大量低质量数据
  - 锚点：`tracks/training.md` 阶段 2

- [ ] **⭐⭐⭐ RLHF 完整流程**：Bradley-Terry 偏好模型 → 奖励模型训练 → PPO 训练（actor-critic + KL 约束）
  - 目标：理解为什么需要 4 个模型同时存在，以及 KL 约束防止什么
  - 锚点：`tracks/training.md` 阶段 2 → `topics/rlhf-ppo.md`

- [ ] **⭐⭐⭐ PPO 算法细节**：重要性采样、clipping 操作、advantage estimation（GAE）
  - 目标：解释 PPO clip 如何既稳定训练又控制步长
  - 锚点：`topics/rlhf-ppo.md`

- [ ] **⭐⭐⭐ DPO 的数学推导**：从 RLHF 最优策略公式如何化简为分类损失
  - 目标：理解等价性的条件（离线数据假设），以及什么情况下这个等价性不成立
  - 锚点：`tracks/alignment.md` 阶段 2 → `topics/dpo-variants.md`

- [ ] **⭐⭐ DPO 变体的动机**：KTO（无需偏好对）、IPO（修正过拟合）、ORPO（合并 SFT + 偏好）各自修正 DPO 的哪个局限
  - 目标：不只知道有哪些变体，而是理解每个变体诞生的具体问题
  - 锚点：`topics/dpo-variants.md`

### RL for Reasoning

- [ ] **⭐⭐ RLVR 的设计**：可验证奖励的定义、为什么数学/代码是天然可验证域、奖励函数的具体形式
  - 目标：能设计一个简单的 RLVR 训练实验（关键超参数是什么？）
  - 锚点：`tracks/training.md` 阶段 4 → `topics/rl-for-reasoning.md`

- [ ] **⭐⭐⭐ GRPO vs PPO**：GRPO 用组内相对分数替代 value model 的机制、省去 value model 的代价
  - 目标：理解为什么 GRPO 在推理训练中足够好，以及什么情况下仍需 PPO
  - 锚点：`topics/rl-for-reasoning.md`

- [ ] **⭐⭐ PRM vs ORM**：步骤级奖励 vs 结果级奖励的区别、PRM 标注成本、Math-Shepherd 自动化方案
  - 目标：用一个具体例子说明 ORM 如何强化"运气正确"的错误推理链
  - 锚点：`topics/rl-for-reasoning.md`

- [ ] **⭐⭐⭐ R1-Zero 的涌现**：纯 RL 训练为什么产生自我反思、这和 few-shot 涌现的异同
  - 目标：从奖励信号的信息量出发，解释为什么自我反思是有利策略
  - 锚点：`tracks/training.md` 阶段 4

- [ ] **⭐⭐ 推理蒸馏机制**：用 R1 思维链做 SFT 为什么比直接 SFT 更好
  - 目标：理解"学习推理过程"比"学习最终答案"传递更多信号的原因
  - 锚点：`tracks/training.md` 阶段 4

---

## Scaling 轨道

- [ ] **⭐⭐ Kaplan Scaling Laws 详解**：N、D、C 的幂律指数、实验如何拟合这些数字
  - 目标：理解 Kaplan 实验设计，以及为什么结论是"优先扩大参数"
  - 锚点：`tracks/scaling.md` 阶段 1 → `topics/scaling-laws.md`

- [ ] **⭐⭐ Kaplan 误差来源**：只计非 embedding 参数 + 小规模外推导致的系统偏差
  - 目标：理解为什么这两个因素让最优 N/D 比向参数侧偏移了 11-12 倍
  - 锚点：`topics/scaling-laws.md`

- [ ] **⭐⭐⭐ Chinchilla 分析方法**：三种实验设计（拟合法、IsoFLOP 法、参数法）及其各自的假设
  - 目标：理解为什么不同方法论收敛到同一结论（20 tokens/param）
  - 锚点：`topics/scaling-laws.md`

- [ ] **⭐⭐ 推理成本改变最优化目标**：从"训练损失最小化"到"总成本最小化"的数学重写
  - 目标：用具体场景（如 1 亿用户 × 10 次/天）计算过训练的经济合理性
  - 锚点：`tracks/scaling.md` 阶段 3

- [ ] **⭐⭐⭐ Inference-time Scaling 机制**：Best-of-N 的质量-成本曲线、MCTS 在语言推理中的应用、难度自适应分配
  - 目标：解释为什么 inference-time 和 training-time scaling 曲线形状相似
  - 锚点：`tracks/scaling.md` 阶段 4 → `topics/inference-time-compute.md`

- [ ] **⭐⭐ MoE Scaling Laws 的五因子框架**：激活参数、总参数、专家数、共享比例、数据量的联合模型
  - 目标：理解为什么 dense 的 scaling law 不能套用到 MoE
  - 锚点：`tracks/scaling.md` 当前格局

- [ ] **⭐⭐ 数据墙的时间估算**：Epoch AI 的方法论、高质量文本总量的估算、耗尽的时间窗口
  - 目标：评估这个预测的假设和不确定性
  - 锚点：`tracks/data.md` 阶段 4

---

## 数据轨道

- [ ] **⭐ Common Crawl 处理流水线**：语言识别 → URL 过滤 → 去重 → 质量过滤的工序和各步淘汰率
  - 目标：了解 1T token 数据集从 Common Crawl 到可用需要哪些步骤
  - 锚点：`tracks/data.md` 阶段 1

- [ ] **⭐⭐ MinHash 去重原理**：LSH 如何在亚线性时间内找相似文档、签名矩阵的设计
  - 目标：理解为什么需要 LSH 而不是 pairwise 比较
  - 锚点：`tracks/data.md` 阶段 2

- [ ] **⭐⭐ ML 分类器质量过滤**：DCLM 的 fastText 分类器训练方法、为什么优于启发式
  - 目标：理解"模型驱动过滤"的核心优势来源
  - 锚点：`tracks/data.md` 阶段 4

- [ ] **⭐⭐ 合成数据生成方法对比**：Self-Instruct / Evol-Instruct / Magpie 的机制差异和质量天花板
  - 目标：理解 Magpie 无需种子提示也能生成高质量指令的原因
  - 锚点：`tracks/data.md` 阶段 3

- [ ] **⭐⭐⭐ Model Collapse 机制**：递归训练退化的数学原因（尾部信息丢失）、真实数据的最低比例
  - 目标：从分布收缩的角度解释为什么 model collapse 是不可避免的
  - 锚点：`tracks/data.md` 阶段 3

- [ ] **⭐⭐ 数据混合比例对能力的影响**：代码/数学/多语言的配比如何影响不同能力维度
  - 目标：给出"改变混合比例的反事实分析"
  - 锚点：`tracks/data.md` 全文

---

## 推理优化轨道

- [ ] **⭐⭐ KV Cache 工作原理**：存什么、省什么计算、内存如何随序列长度和批次大小变化
  - 目标：计算 Llama 3 8B 在 32K context, batch=8 时的 KV Cache 大小
  - 锚点：`tracks/inference.md` 阶段 1 → `topics/kv-cache.md`

- [ ] **⭐⭐⭐ PagedAttention 设计**：碎片化问题根源、页表管理机制、与 OS 虚拟内存的类比
  - 目标：解释为什么 PagedAttention + 连续批处理能实现 24x 吞吐提升
  - 锚点：`topics/kv-cache.md`

- [ ] **⭐ 精度类型体系**：FP32 → FP16 → BF16 → FP8 → INT8 → INT4 各自的数值范围、精度、适用场景
  - 目标：说清楚 BF16 为什么在训练中取代 FP16（指数位 vs 尾数位的权衡）、FP8 的两种格式（E4M3 vs E5M2）各自定位
  - 锚点：`topics/quantization.md`

- [ ] **⭐ PTQ vs QAT 的选择逻辑**：训练后量化 vs 量化感知训练的适用边界
  - 目标：解释为什么超大 LLM（>70B）几乎只能用 PTQ，QAT 在什么规模和场景下仍有价值
  - 锚点：`topics/quantization.md`

- [ ] **⭐⭐ LLM 量化的核心难点**：异常值分布问题、为什么 LLM 比 CNN 更难量化
  - 目标：理解异常值的分布特征（集中在特定 channel）以及它如何拉垮整体量化精度
  - 锚点：`tracks/inference.md` 阶段 2 → `topics/quantization.md`

- [ ] **⭐⭐ AWQ vs GPTQ 对比**：两者识别"重要权重"的不同方法
  - 目标：GPTQ 用 Hessian 逆做逐列补偿 vs AWQ 用激活幅度找 1% 显著权重——理解两种思路的直觉和局限
  - 锚点：`topics/quantization.md`

- [ ] **⭐⭐ SmoothQuant 的难度转移思想**：激活量化为什么比权重量化更难、如何在权重和激活之间"搬运"量化难度
  - 目标：理解 per-channel scaling 如何把激活中的异常值转移到权重上（权重是静态的，更容易处理）
  - 锚点：`topics/quantization.md`

- [ ] **⭐⭐ GGUF 混合精度与边缘部署**：Q4_K_M 等格式的设计直觉、不同层用不同精度的策略
  - 目标：理解 llama.cpp 量化方案如何在 CPU 上实现可用推理（SIMD 优化、内存映射）
  - 锚点：`topics/quantization.md`

- [ ] **⭐ FP8 与下一代硬件原生支持**：H100/B200 的 FP8 Tensor Core 如何改变量化格局
  - 目标：理解 FP8 作为"训练+推理统一精度"的定位——不是极限压缩，而是硬件原生的效率甜点
  - 锚点：`topics/quantization.md`

- [ ] **⭐⭐⭐ 推测解码的正确性证明**：rejection sampling 如何保证输出分布不变
  - 目标：理解为什么推测解码是精确等价而非近似
  - 锚点：`tracks/inference.md` 阶段 3 → `topics/speculative-decoding.md`

- [ ] **⭐⭐ Prefill vs Decode 瓶颈差异**：计算密集型 vs 内存带宽密集型的本质原因
  - 目标：解释 PD 分离在大规模 serving 中的经济逻辑
  - 锚点：`tracks/inference.md` 阶段 3

- [ ] **⭐⭐ 知识蒸馏机制**：响应蒸馏 vs Logit 蒸馏的信息量差异、温度参数的作用
  - 目标：理解为什么 soft label 比 hard label 携带更多信息（暗知识）
  - 锚点：`tracks/inference.md` 阶段 2

---

## 对齐轨道

- [ ] **⭐⭐ Reward Hacking / Goodhart's Law**：奖励模型误差如何被 RL 放大、已知的 hacking 模式
  - 目标：给出两个 LLM 训练中 reward hacking 的具体例子
  - 锚点：`tracks/alignment.md` 阶段 3

- [ ] **⭐⭐ 谄媚问题**：RLHF 为什么引入谄媚、反谄媚数据如何设计
  - 目标：理解人类评估偏好和"实际正确"之间的系统性偏差
  - 锚点：`tracks/alignment.md` 对能力输出的影响

- [ ] **⭐⭐ 对齐税及缓解**：对齐训练如何损害能力、权重插值为什么能部分恢复
  - 目标：从损失曲面的角度解释插值恢复能力的机制
  - 锚点：`tracks/alignment.md` 关键分歧

- [ ] **⭐⭐⭐ 推理链可信度问题**：展示的推理链 vs 真实内部计算、Anthropic 的证据
  - 目标：理解为什么"可见推理链"≠ "对齐透明性"
  - 锚点：`tracks/alignment.md` 阶段 3

- [ ] **⭐⭐⭐ Scalable Oversight**：辩论机制设计、弱到强泛化的核心假设及其成立条件
  - 目标：分析辩论机制的假设（人能判断辩论质量但不能直接判答案）何时成立
  - 锚点：`tracks/alignment.md` 阶段 3

---

## 能力维度

- [ ] **⭐⭐ Chain-of-Thought 为什么有效**：大模型有效 / 小模型无效的尺度效应、CoT 是发现还是创造能力
  - 目标：解释"CoT 激活预训练中已有的推理模式"的证据和反驳
  - 锚点：`capabilities/reasoning.md` → `topics/chain-of-thought.md`

- [ ] **⭐ Reasoning Benchmark 分析**：AIME / GPQA / ARC-AGI / FrontierMath 各自测的是什么类型的 reasoning
  - 目标：解释为什么模型可以在 AIME 强但在 ARC-AGI 弱
  - 锚点：`capabilities/reasoning.md` 当前水位

- [ ] **⭐⭐ Lost in the Middle**：检索性能随文档位置变化的模式及机制假说
  - 目标：理解这个现象对 RAG 架构设计的影响
  - 锚点：`capabilities/long-context.md`

- [ ] **⭐⭐ 视觉语言模型架构**：ViT 原理、图像如何变成 token、语言模型接合方式（cross-attention vs prefix）
  - 目标：理解 LLaVA / Flamingo / GPT-4V 的架构差异和优劣
  - 锚点：`capabilities/multimodality.md`

- [ ] **⭐⭐ Function Calling 训练**：工具调用的 SFT 数据设计、执行反馈 RL 的信号与人类偏好的区别
  - 目标：理解工具使用训练和纯语言训练的信号差异
  - 锚点：`capabilities/tool-use.md` → `tracks/training.md` 阶段 3

- [ ] **⭐⭐ Best-of-N 采样的数学**：采样数 N 与质量提升的关系曲线、验证器质量对效果的约束
  - 目标：推导 Best-of-N 的 scaling 行为为什么是对数线性的
  - 锚点：`topics/inference-time-compute.md`

---

## 关键范式事件

- [ ] **⭐⭐ GPT-3 Few-Shot 涌现**：为什么 175B 具备的能力在更小规模不存在、涌现是相变还是测量尺度问题
  - 目标：评估"涌现是真实突变 vs 只是评估指标非线性"两派观点
  - 锚点：`eras/era1-scaling.md`

- [ ] **⭐⭐ InstructGPT 1.3B > GPT-3 175B**：训练信号 vs 规模在"有用性"上的解耦
  - 目标：这个结果对"scaling = 通往 AGI"论断意味着什么
  - 锚点：`eras/era2-alignment.md`

- [ ] **⭐⭐⭐ o1 的技术实质**：推理训练的具体目标、隐藏思维链设计动机、和 GPT-4 CoT 的本质区别
  - 目标：从技术层面（非 marketing）解释 o1 为什么是根本不同的产品
  - 锚点：`eras/era4-reasoning.md` → `tracks/training.md` 阶段 4

- [ ] **⭐⭐⭐ DeepSeek-R1-Zero 的意义**：纯 RL 为什么产生自我反思、对 "reasoning is emergent" 的判断
  - 目标：分析 R1-Zero 的自我反思是新策略涌现还是预训练行为被 RL 选择出来
  - 锚点：`tracks/training.md` 阶段 4

---

## 基础设施与工程

### Tokenization

- [ ] **⭐⭐ BPE（Byte Pair Encoding）算法**：从字符级开始如何通过合并频繁对构建词表、词表大小对性能的影响
  - 目标：手动模拟 BPE 合并过程，理解为什么 "token" 不等于 "词"
  - 锚点：`tracks/architecture.md` 阶段 1（隐含前提）

- [ ] **⭐⭐ 词表设计的权衡**：词表大小（32K vs 128K vs 256K）对模型效率和能力的影响、多语言 tokenizer 的设计
  - 目标：解释为什么 Llama 3 将词表从 32K 扩大到 128K，以及这对非英文语言的意义
  - 锚点：全体系隐含前提

- [ ] **⭐ Tokenization 对下游能力的影响**：为什么 LLM 数学弱可能与数字 tokenization 有关、代码 tokenization 的特殊处理
  - 目标：给出 tokenization 如何影响模型能力的具体例子
  - 锚点：`capabilities/reasoning.md`

### 分布式训练

- [ ] **⭐⭐ 数据并行（Data Parallel）**：每个 GPU 持有完整模型副本、梯度 AllReduce 的通信开销
  - 目标：理解为什么数据并行在模型超过单 GPU 内存后就不够用了
  - 锚点：`tracks/architecture.md`（隐含前提）

- [ ] **⭐⭐ 张量并行（Tensor Parallel）**：如何把一个矩阵乘法切分到多张 GPU、AllReduce 的通信模式
  - 目标：理解 Megatron-LM 的行列切分方案
  - 锚点：`tracks/inference.md` 阶段 1

- [ ] **⭐⭐ 流水线并行（Pipeline Parallel）**：模型按层切分、流水线气泡（bubble）问题和微批处理的解法
  - 目标：计算流水线效率（bubble ratio）和最优微批数量
  - 锚点：`tracks/inference.md` 阶段 1

- [ ] **⭐⭐⭐ ZeRO（Zero Redundancy Optimizer）**：三个阶段分别消除什么冗余（optimizer states / gradients / parameters）
  - 目标：理解 ZeRO-3 如何让数据并行也能训练超大模型
  - 锚点：`tracks/scaling.md`（解释为什么训练前沿模型需要巨大算力集群）

### LoRA 与参数高效微调

- [ ] **⭐⭐ LoRA 的低秩假设**：为什么权重更新 ΔW 可以用低秩矩阵 BA 近似、rank 选择的影响
  - 目标：理解 LoRA 在推理时无额外延迟的原理（W + BA 合并）
  - 锚点：`topics/lora-peft.md`

- [ ] **⭐⭐ QLoRA**：4-bit 量化基础模型 + LoRA 的组合、NF4 格式的设计
  - 目标：理解为什么 QLoRA 能在单张 48GB GPU 上微调 65B 模型
  - 锚点：`topics/lora-peft.md`

- [ ] **⭐ PEFT 方法对比**：Adapter / Prefix Tuning / Prompt Tuning / LoRA 各自的机制和适用场景
  - 目标：理解为什么 LoRA 最终成为主流选择（推理无延迟 + 效果接近全量微调）
  - 锚点：`topics/lora-peft.md`

---

## 跨轨道概念

### In-Context Learning

- [ ] **⭐⭐ In-Context Learning 的机制**：为什么 LLM 能从 context 中的示例"学习"、与梯度下降的类比
  - 目标：理解 Olsson et al. (2022) 的 induction heads 假说，以及 ICL 是否等价于隐式梯度下降
  - 锚点：`eras/era1-scaling.md` → `capabilities/reasoning.md`

- [ ] **⭐⭐⭐ 涌现能力争论**：涌现是真实的相变还是评估指标的非线性效应（Schaeffer et al., 2023）
  - 目标：同时理解两派观点的证据，形成自己的判断
  - 锚点：`eras/era1-scaling.md` → `tracks/scaling.md`

### 多模态基础

- [ ] **⭐⭐ CLIP 与对比学习**：图文对比训练的目标函数、为什么 CLIP 的视觉表示可以直接用于下游
  - 目标：理解 CLIP 的 InfoNCE loss，以及为什么对比学习产生了通用的视觉表示
  - 锚点：`capabilities/multimodality.md` 演进轨迹

- [ ] **⭐⭐ Vision Transformer (ViT)**：图像如何切成 patch 变成 token 序列、与 CNN 的能力对比
  - 目标：理解 ViT 为什么在大规模训练下超越 CNN，以及 patch size 的权衡
  - 锚点：`capabilities/multimodality.md`

- [ ] **⭐⭐ 多模态模型的接合架构**：Flamingo（cross-attention）vs LLaVA（MLP projection）vs Gemini（原生多模态）
  - 目标：理解三种架构在信息瓶颈和跨模态推理上的本质差异
  - 锚点：`capabilities/multimodality.md` 技术归因

### RAG 与检索增强

- [ ] **⭐⭐ RAG 架构**：检索器（embedding + 向量数据库）+ 生成器的协作机制
  - 目标：理解 RAG 相比纯长上下文的优势（成本、时效性）和劣势（检索召回率、上下文碎片化）
  - 锚点：`capabilities/long-context.md` → `capabilities/memory.md`

- [ ] **⭐⭐ Embedding 模型**：文本如何变成向量、对比学习训练 embedding 的方法、MTEB benchmark
  - 目标：理解语义相似度搜索的原理和局限
  - 锚点：`capabilities/memory.md`

- [ ] **⭐⭐ RAG vs Long Context 的选择**：什么场景下 RAG 仍然必要、什么场景下长上下文可以替代 RAG
  - 目标：给出一个决策框架（成本、时效性、精确度三个维度）
  - 锚点：`capabilities/long-context.md` 演进轨迹 2024

### 评估方法论

- [ ] **⭐⭐ Benchmark 设计原则**：什么是好的 benchmark（区分度、抗污染、对齐真实能力）
  - 目标：用 MATH-500 饱和 vs FrontierMath 有效的对比，说明 benchmark 设计的关键要素
  - 锚点：`capabilities/reasoning.md` 当前水位

- [ ] **⭐⭐ 数据污染问题**：训练数据中包含 benchmark 题目如何影响评估可信度、检测方法
  - 目标：理解为什么"每年出新题"（AIME）比"固定题库"更可靠
  - 锚点：全体系（评估方法论是元问题）

- [ ] **⭐⭐ 评估的 Goodhart 问题**：模型针对 benchmark 优化（而非真实能力优化）的风险和实例
  - 目标：给出 benchmark 优化 vs 真实能力提升的具体案例
  - 锚点：`tracks/alignment.md`（Goodhart's Law 在评估层面的表现）

### 可解释性

- [ ] **⭐⭐⭐ 超位置假说（Superposition）**：为什么神经元是多义的、网络如何在有限维度中编码超过维度数的特征
  - 目标：理解 Elhage et al. (2022) 的 toy model 实验和对 LLM 的含义
  - 锚点：`topics/mechanistic-interpretability.md`

- [ ] **⭐⭐ 稀疏自编码器（SAE）**：如何用 SAE 分解超位置、Anthropic 在 Claude 上找到的可解释特征
  - 目标：理解 SAE 如何把 polysemantic 神经元分解为 monosemantic 特征
  - 锚点：`topics/mechanistic-interpretability.md`

- [ ] **⭐⭐ Activation Steering**：通过修改内部激活向量来控制模型行为的原理
  - 目标：理解为什么加/减一个"方向向量"能改变模型的行为倾向
  - 锚点：`topics/mechanistic-interpretability.md` → `tracks/alignment.md`

### 现代架构标准组件

- [ ] **⭐ RMSNorm vs LayerNorm**：为什么现代模型用 RMSNorm 替代 LayerNorm（去掉均值中心化）
  - 目标：理解计算节省和质量保持的原因
  - 锚点：`tracks/architecture.md`（隐含组件）

- [ ] **⭐⭐ SwiGLU 激活函数**：从 ReLU → GELU → SwiGLU 的演进、为什么 GLU 变体在 LLM 中更好
  - 目标：理解 gating mechanism 如何改善 FFN 的表达能力
  - 锚点：`tracks/architecture.md`（隐含组件）

---

## 2026-05-03 重构新增 tasks

基于互联网检索 + 落地证据，新增 2024-2025 真实落地的关键 task。这些 task 已分散吸收到主干 18 篇 blog 中（见每条的"进入 blog #X"标注）。

### 架构（稀疏 / 线性 / Hybrid）

- [ ] **⭐⭐⭐ NSA（Native Sparse Attention）**：DeepSeek 2025-02 的三路稀疏（压缩 + 选择 + 滑动窗口），原生可训练
  - 目标：解释"原生可训练"相对早期 Longformer/BigBird"事后近似"的本质区别
  - 锚点：进入 blog 3 [attention-sparse-hybrid](../notes/2026-05-31-attention-sparse-hybrid.md)
  - 落地：DeepSeek V3.2-Exp（2025-09）

- [ ] **⭐⭐ MoBA（Mixture of Block Attention）**：Moonshot 2025-02，块级稀疏路由
  - 目标：与 NSA 对比，理解两条独立稀疏路线
  - 锚点：blog 3

- [ ] **⭐⭐⭐ Gated DeltaNet（线性注意力）**：Qwen3-Next 用的线性算子，源自 Gated Delta Networks
  - 目标：与纯 Mamba SSM 区分（delta rule + gating，不是状态空间）
  - 锚点：blog 3
  - 落地：Qwen3-Next 80B-A3B（vLLM 原生支持）

- [ ] **⭐⭐ Hybrid 3:1 注意力比例**：Qwen3-Next / Nemotron-H 共同选择的"3 层线性 + 1 层 full"模式
  - 目标：理解这个比例为什么是工程甜点
  - 锚点：blog 3

### MoE 工程

- [ ] **⭐⭐ 三家 MoE 路由对比**：DeepSeek（1 共享 + 8/256）vs Qwen3（top-8/128 无共享）vs Llama 4（1+1）
  - 目标：理解共享专家的必要性争论 + 各家稀疏度选择背后的服务约束
  - 锚点：blog 4 [moe-architecture](../notes/2026-06-07-moe-architecture.md)

- [ ] **⭐⭐ Multi-Token Prediction（MTP）**：训练时多 token 预测 + 推理时天然兼容推测解码
  - 目标：理解为什么这个机制同时改善训练 sample efficiency 和推理 latency
  - 锚点：blog 4
  - 落地：Qwen3-Next 等

- [ ] **⭐⭐⭐ Wide Expert Parallel**：vLLM + llm-d 的 256 专家分片服务
  - 目标：理解 EP 的 all-to-all 通信模式与 TP 的本质区别
  - 锚点：blog 4 + blog 14 [parallelism](../notes/2026-08-16-parallelism.md)

### Reasoning RL

- [ ] **⭐⭐⭐ DAPO 四项修正**：Clip-Higher / Dynamic Sampling / Token-level PG Loss / Overlong Reward Shaping
  - 目标：每一项对应解决 GRPO 在 long-CoT 训练中的哪个具体失败模式
  - 锚点：blog 8 [rlvr-grpo](../notes/2026-07-05-rlvr-grpo.md)
  - 落地：ByteDance/Tsinghua 2025

- [ ] **⭐⭐⭐ 推理范式三家分叉**：OpenAI dedicated reasoning model / Anthropic hybrid thinking / DeepSeek 开源蒸馏
  - 目标：能解释三家产品哲学差异 + 各自技术取舍
  - 锚点：blog 9 [reasoning-paradigm-split](../notes/2026-07-12-reasoning-paradigm-split.md)

- [ ] **⭐⭐ Anthropic Hybrid Thinking 设计**：thinking budget 参数 + 一个模型双 mode
  - 目标：理解为什么 Anthropic 选这条路而非分裂模型
  - 锚点：blog 9

- [ ] **⭐⭐ On-Policy Distillation**：Thinking Machines 2025，解决静态蒸馏的分布漂移
  - 目标：与 GRPO（用 verifier）的边界
  - 锚点：blog 13 [speculative-decoding-distillation](../notes/2026-08-09-speculative-decoding-distillation.md)

### 推理服务（KV / 长上下文 / 推测解码）

- [ ] **⭐⭐ Prompt Caching（跨请求 KV 复用）**：Anthropic 显式 cache_control（5min/1h TTL）vs OpenAI 隐式自动
  - 目标：理解两种设计哲学差异 + 对长 system prompt 应用的成本曲线影响
  - 锚点：blog 11 [kv-cache-economics](../notes/2026-07-26-kv-cache-economics.md)

- [ ] **⭐⭐⭐ RULER vs NIAH 长上下文真实退化**：四类任务（multi-needle / multi-hop / aggregation / QA）
  - 目标：解释为什么 1M context 在 RULER 上 50K-130K 就显著掉
  - 锚点：blog 11
  - 落地：NVIDIA RULER 已成事实标准

- [ ] **⭐⭐⭐ EAGLE-3 推测解码**：draft head 直接 attach 到 target，多层 feature reuse
  - 目标：理解为什么 EAGLE-3 赢过 MEDUSA 成为 vLLM/SGLang/TensorRT-LLM 默认
  - 锚点：blog 13

### 训练精度

- [ ] **⭐⭐⭐ FP8 作为训练+推理统一精度**：DeepSeek V3 首次大规模 FP8 训练成功 + Qwen 3.5 native FP8 pipeline
  - 目标：解释 E4M3 vs E5M2 两种格式的定位 + 为什么 FP8 能挑战 BF16 在训练中的地位
  - 锚点：blog 12 [quantization-landscape](../notes/2026-08-02-quantization-landscape.md)

