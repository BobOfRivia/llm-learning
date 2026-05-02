# 范式替代日志

记录 LLM 领域的范式级变化——旧方法被淘汰、新方法确立的关键事件。

---

## 2020-01｜Scaling Laws 确立"以规模换能力"的数学基础

- **轨道**：scaling
- **替代了什么**：依赖架构创新或任务特化来提升性能的工程直觉
- **为什么**：Kaplan et al.（OpenAI，arXiv 2001.08361）在 7+ 数量级范围内证明了 loss 与 N、D、C 之间的幂律关系，给"越大越好"提供了可预测的数学基础
- **影响的能力**：几乎所有能力维度的基线均随规模提升
- **置信度**：高
- **关联文件**：tracks/scaling.md（阶段 1），eras/era1-scaling.md

## 2020-05｜GPT-3 确立 in-context learning 为主要使用范式

- **轨道**：training, architecture
- **替代了什么**：针对每个任务 fine-tune 模型的传统范式（BERT 时代的主流做法）
- **为什么**：GPT-3（175B）展示了不更新权重、仅通过 prompt 中的少量示例即可泛化到新任务，few-shot prompting 成为主要使用方式
- **影响的能力**：instruction following 的初步形态，reasoning 能力的早期迹象
- **置信度**：高
- **关联文件**：tracks/architecture.md（阶段 2），tracks/training.md（阶段 1），eras/era1-scaling.md

## 2022-03｜Chinchilla 修正 Scaling Laws，推翻"参数主导"配比

- **轨道**：scaling, data, training
- **替代了什么**：Kaplan 建立的"固定计算预算下优先扩大参数"的训练配比（约 1.7 tokens/参数）
- **为什么**：Hoffmann et al.（DeepMind，arXiv 2203.15556）证明最优配比是等比扩展（约 20 tokens/参数），Chinchilla 70B 用相同计算量超过 Gopher 280B，证明之前主流模型全都 undertrained
- **影响的能力**：所有能力维度——更均衡的训练配比使同等计算下的能力更强
- **置信度**：高
- **关联文件**：tracks/scaling.md（阶段 2），eras/era1-scaling.md

## 2022-03｜InstructGPT 确立后训练为"有用助手"的必要路径

- **轨道**：training, alignment
- **替代了什么**："预训练足够好就行"的假设；规模堆砌作为对齐手段
- **为什么**：Ouyang et al.（OpenAI，arXiv 2203.02155）证明 1.3B InstructGPT > 175B GPT-3（人类评估），说明有用性靠训练信号而非规模。RLHF 三步流水线（SFT→RM→PPO）成为行业标准
- **影响的能力**：instruction following 的质变，safety 能力的工程化基础
- **置信度**：高
- **关联文件**：tracks/training.md（阶段 2），eras/era1-scaling.md，eras/era2-alignment.md

## 2022-11｜ChatGPT 确立"对话式 AI"为消费产品范式

- **轨道**：training, alignment
- **替代了什么**：API/研究工具为主要使用形态；用户需要懂 prompting 才能有效使用 LLM
- **为什么**：ChatGPT 5 天 100 万用户，两个月 1 亿用户，证明 RLHF 对齐后的对话模型可以直接面向大众
- **影响的能力**：instruction following 的产品化，safety 对可用性的重要性被大众验证
- **置信度**：高
- **关联文件**：tracks/alignment.md（阶段 1），eras/era2-alignment.md

## 2022-12｜Constitutional AI 开辟 RLAIF 路线

- **轨道**：alignment
- **替代了什么**：100% 人类反馈作为对齐信号的单一路线
- **为什么**：Anthropic（arXiv 2212.08073）证明 AI 反馈可以替代人类在有害性判断上的标注，解决人工标注的扩展性瓶颈
- **影响的能力**：safety 能力的规模化，instruction following 的可扩展对齐
- **置信度**：高
- **关联文件**：tracks/alignment.md（阶段 1），eras/era2-alignment.md

## 2023-02｜Llama 1 将开放权重模型推向主流

- **轨道**：training（生态事件）
- **替代了什么**：大型 LLM 只能通过闭源 API 访问的格局
- **为什么**：Meta Llama 1（65B 与 PaLM/Chinchilla 相当）以申请制开放权重，叠加权重泄露，事实上开启了开放生态的爆发；Alpaca 等随后证明低成本 instruction tuning 可行
- **影响的能力**：instruction following 的民主化；开放生态对对齐方法的实验速度大幅提升
- **置信度**：高
- **关联文件**：eras/era2-alignment.md，eras/tension-efficiency-openness.md

## 2023-05｜DPO 简化 RLHF，降低对齐工程门槛

- **轨道**：alignment, training
- **替代了什么**：三阶段 RLHF（SFT→RM→PPO）作为唯一可行的偏好学习框架
- **为什么**：Rafailov et al.（斯坦福，arXiv 2305.18290）证明最优策略可以从偏好数据闭合推导，将三阶段压缩为一个分类损失，无需 reward model 和 PPO 采样
- **影响的能力**：instruction following 和 safety 的训练效率提升；开源社区可以更低成本做对齐
- **置信度**：高
- **关联文件**：tracks/alignment.md（阶段 2），eras/era2-alignment.md

## 2023-03｜GPT-4 确立多模态为模型基本规格

- **轨道**：architecture, training
- **替代了什么**：多模态作为独立研究方向或高端可选功能；纯文本 LLM 作为默认产品形态
- **为什么**：GPT-4V 展示了图像理解与语言能力的自然融合，用户期待随即重置——一个"不支持图像"的 LLM 开始被视为规格不完整
- **影响的能力**：capabilities/multimodality（质变）
- **置信度**：高
- **关联文件**：eras/era3-capability-race.md，capabilities/multimodality.md（待创建）

## 2023-05｜长上下文（100K+）成为独立竞争维度

- **轨道**：architecture（位置编码 scaling），inference（KV cache 工程）
- **替代了什么**：4K-8K 作为事实上的上下文限制；长文档处理依赖外部 RAG 或分块处理
- **为什么**：Claude 100K 证明了生产环境下的大窗口可行，Gemini 1.5 随后把上限推到 1M，使整个代码库/文档库入 context 成为工程选项
- **影响的能力**：capabilities/long-context（量变引发质变）
- **置信度**：高
- **关联文件**：eras/era3-capability-race.md，tracks/architecture.md（阶段 3，RoPE + Flash Attention）

## 2023-12｜Mixtral 8x7B 验证 MoE 在开源生态的生产可行性

- **轨道**：architecture
- **替代了什么**："MoE 只有超大实验室才能稳定驾驭"的认知；dense 模型作为开源社区的默认架构
- **为什么**：Mixtral 8x7B 以约 13B 的活跃参数成本实现超过 70B dense 模型的性能，且以完全开放权重发布，证明稀疏激活的成本效益在资源有限的组织中同样可以实现
- **影响的能力**：推理成本降低影响所有能力的可及性
- **置信度**：高
- **关联文件**：eras/era3-capability-race.md，tracks/architecture.md（阶段 3，MoE 路线）

## 2023-07｜Llama 2 过训练策略使"推理最优"与"训练最优"的分离成为行业共识

- **轨道**：scaling, training
- **替代了什么**：Chinchilla 最优作为训练配比的唯一准则；大参数模型作为能力提升的默认路径
- **为什么**：Llama 1 和 Llama 2 系列证明，用远超 Chinchilla 建议的数据量训练小模型，可以在推理成本上实现结构性优势；Llama 3 将这个逻辑推向极端（8B 模型训练 15T tokens），结果 8B 超过 70B
- **影响的能力**：所有能力在成本约束下的可及性（小而强的模型使更多用户可以本地或低成本部署）
- **置信度**：高
- **关联文件**：eras/era3-capability-race.md，tracks/scaling.md（阶段 3）

## 2023-06｜工具使用（function calling）进入原生训练目标

- **轨道**：training
- **替代了什么**：通过 prompt engineering 模拟工具调用；工具使用作为能力边界之外的"技巧"
- **为什么**：GPT-4 的 function calling API 配套了专门的训练数据和执行反馈机制，使工具调用成为可靠的系统行为而非偶尔可用的技巧；随后开源模型也跟进了工具使用专项训练
- **影响的能力**：capabilities/tool-use（从模糊到可靠），Agent 架构的可行性基础
- **置信度**：高
- **关联文件**：eras/era3-capability-race.md，tracks/training.md（阶段 3）

## 2024-09｜o1 开启推理时扩展作为独立 scaling 维度

- **轨道**：scaling, training
- **替代了什么**：能力提升只靠训练时 scaling（更大模型、更多数据）的单一路径；推理计算量固定不变的假设
- **为什么**：o1 在 AIME 2024 上 74%（vs GPT-4o 的 9%）证明推理时计算投入可以产生训练时 scaling 无法达到的能力跃升；o3 随后展示了连续可调的推理时 scaling 曲线（low → high compute，ARC-AGI 从 47% 到 88%）
- **影响的能力**：capabilities/reasoning（质变），capabilities/inference-time-compute（新维度）
- **置信度**：高
- **关联文件**：eras/era4-reasoning.md，tracks/scaling.md（阶段 4），tracks/training.md（阶段 4）

## 2025-01｜DeepSeek-R1 证明 RL for Reasoning 技术路线可开源复现

- **轨道**：training
- **替代了什么**："推理模型只有 OpenAI 能做"的认知壁垒；RLHF 作为后训练唯一主流路线
- **为什么**：R1-Zero 展示了从基础模型纯 RL 训练推理能力的可行性（无需大量 SFT 冷启动），且训练过程中自发涌现自我反思行为；R1-Distill 系列证明推理能力可以通过蒸馏低成本转移到开放模型
- **影响的能力**：capabilities/reasoning（开放生态的推理能力门槛降低）
- **置信度**：高
- **关联文件**：eras/era4-reasoning.md，tracks/training.md（阶段 4）

## 2024-06｜DCLM 确立模型过滤取代启发式过滤为预训练数据标准

- **轨道**：data
- **替代了什么**：启发式规则过滤（语言检测 + 长度过滤 + 关键词黑名单）作为预训练数据清洗的主要手段
- **为什么**：DCLM（NeurIPS 2024）在标准化对照实验中证明，基于 ML 分类器的文档质量打分显著优于启发式过滤；DCLM-Baseline 7B 在 MMLU 上比前开放 SOTA 提升 6.6 个百分点，计算量减少 40%。FineWeb 同年通过 15T tokens 的规模验证了这一路线
- **影响的能力**：几乎所有能力的基础，知识密集型任务（MMLU、GPQA）提升最直接
- **置信度**：高
- **关联文件**：tracks/data.md（阶段 4）

## 2024-01｜Model Collapse 研究确认合成数据不能完全替代真实数据

- **轨道**：data, training
- **替代了什么**：关于合成数据可以无限递归扩展预训练数据的乐观预期
- **为什么**：Nature 2024（Shumailov et al.）通过递归训练实验证实：每代模型词汇和语义多样性持续下降，原始分布尾部信息不可逆丢失，即使 1/1000 比例的合成数据长期也有退化效应。结论是预训练中合成数据必须与真实数据混合，纯合成递归不可行
- **影响的能力**：间接影响所有能力的天花板（制约了合成数据 + 无限扩展的路线）
- **置信度**：高（实验室实验，实际生产阈值仍有讨论）
- **关联文件**：tracks/data.md（阶段 3）

## 2025-01｜Llama 4 标志 Dense Transformer 从 frontier 旗舰模型退场

- **轨道**：architecture
- **替代了什么**：Dense Transformer 作为所有规模旗舰模型的默认选择；Llama 3.x 系列一直坚持 Dense，Llama 4 是 Meta 第一次放弃
- **为什么**：在数百 B 参数规模下，Dense 的训练和推理成本无法承受；MoE（400B total / 17B active）在相近质量下推理成本仅为对应 Dense 的 ~4%；同期 DeepSeek-V3、Qwen 3、GLM-5 也全部采用 MoE，形成全面共识
- **影响的能力**：所有能力——MoE 解耦了知识容量与推理成本，使更大的知识存储在相同服务成本下可行
- **置信度**：高
- **关联文件**：tracks/architecture.md（当前技术格局），tracks/scaling.md（MoE 重新定义"规模"）

## 2025-01｜MoE Scaling Laws 与密集模型不兼容，需要独立理论框架

- **轨道**：scaling, architecture
- **替代了什么**：将密集模型 scaling laws（N/D/C 三元框架）直接套用到 MoE 模型的做法；以及"MoE 比 dense 更内存昂贵"的传统认知
- **为什么**：Ludziejewski et al.（ICLR 2025，arXiv 2502.05172）在 280+ 个实验中证明 MoE 需要五因子框架（激活参数、总参数、专家数、共享专家比例、数据量），且在合理配置下 MoE 比同质量密集模型更内存高效，这推翻了直接外推 Chinchilla 到 MoE 的做法
- **影响的能力**：间接影响所有能力——MoE scaling laws 的准确性决定了实验室能否有效分配 MoE 模型的训练资源
- **置信度**：高（ICLR 同行评审，280+ 实验验证，但最大实验规模仅到 5B total 参数，超大规模外推仍有不确定性）
- **关联文件**：tracks/scaling.md（当前技术格局），tracks/architecture.md（MoE 章节）

## 2023-06｜vLLM / PagedAttention 确立 serving 效率的新基准

- **轨道**：inference
- **替代了什么**：朴素的 KV Cache 预分配（导致严重内存碎片化和低吞吐量）；静态批处理
- **为什么**：PagedAttention 借鉴操作系统虚拟内存管理，结合连续批处理，vLLM 实现相比朴素实现 24 倍吞吐量提升；迅速成为开源 LLM serving 的事实标准框架
- **影响的能力**：所有能力的生产可及性（更高的服务效率 → 更低的 API 成本）
- **置信度**：高
- **关联文件**：tracks/inference.md（阶段 2）

## 2025-04｜工具调用融入推理链内部，Agent 编排从外部移至模型内部

- **轨道**：training, inference
- **替代了什么**：推理模型只在思考阶段推理、在输出阶段调用工具的两段式结构；需要外部 orchestration 框架协调模型与工具交互
- **为什么**：o4-mini（OpenAI，2025 年 4 月）首次让推理模型在思维链内部直接调用工具（网页搜索、代码执行、图像生成），模型自主决定何时在推理过程中插入工具调用并消化返回结果——推理和执行从串联变为交织；Anthropic 的 Claude 4（2025 年 5 月）随后以 Extended Thinking + 工具交织的形式实现同样机制
- **影响的能力**：capabilities/tool-use（可靠性质变），capabilities/reasoning（推理过程可接入外部信息），Agent 架构的端到端能力
- **置信度**：高
- **关联文件**：labs/openai.md，labs/anthropic.md

## 2025-04｜Llama 4 完成开放权重模型的原生多模态 + MoE 路线验证

- **轨道**：architecture, training
- **替代了什么**：开放权重多模态模型的"嫁接式"视觉（Llama 3.2 的适配器方案）；开放权重 MoE 局限于 Mixtral 级别（约 46B 总参数）的认知
- **为什么**：Llama 4 Scout（~109B 总参数，16 专家，10M context）和 Maverick（~400B 总参数，128 专家）从预训练起就在交织图文数据上联合训练——与 Llama 3.2 的适配器方案不同，也与闭源的 Gemini 原生多模态赌注形成了开放验证；同时将开放权重模型的 MoE 规模推进到百亿活跃参数级别
- **影响的能力**：capabilities/multimodality（开放生态的原生多模态门槛降低），capabilities/long-context（10M context 在开放权重中首次实现）
- **置信度**：高
- **关联文件**：labs/meta.md，tracks/architecture.md

## 2025-11｜Gemini 3 Deep Think 以自然语言在 IMO 达到金牌水平

- **轨道**：training, alignment
- **替代了什么**：olympiad 级数学推理需要依赖 Lean 等形式化证明语言的假设（AlphaProof/AlphaGeometry 路线）；自然语言推理在严格数学证明场景中存在根本局限的认知
- **为什么**：Gemini 3 Deep Think 在 2025 年 IMO 中 6 题做对 5 题（35 分，金牌水平），全程接受自然语言输入、输出自然语言证明，在规定 4.5 小时内完成——这与 2024 年 AlphaProof+AlphaGeometry 2（银牌，依赖形式化语言）的方法论完全不同，证明了大规模自然语言推理在严格证明任务上可以超越形式化符号系统
- **影响的能力**：capabilities/reasoning（数学推理能力边界重新定义），多模态与推理的协同作用
- **置信度**：高
- **关联文件**：labs/google.md，eras/era4-reasoning.md

---

<!-- 模板：

## YYYY-MM-DD｜[事件简述]

- **轨道**：architecture / training / scaling / data / inference / alignment
- **替代了什么**：[旧范式]
- **为什么**：[技术原因或实验证据]
- **影响的能力**：[哪些 capabilities/ 受影响]
- **置信度**：高 / 中 / 低
- **关联文件**：[更新了哪些文件]

-->
