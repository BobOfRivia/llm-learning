# Transformer 地基

> deadline: 2026-05-17
> status: drafted
> 覆盖 tasks: BPE 算法、Self-Attention 机制、Multi-Head Attention、Transformer 层结构（Pre-LN / FFN / 残差）、自回归语言建模目标、decoder-only 为什么统治
> 桥接: tracks/architecture.md 阶段 1, topics/attention-mechanism.md

**核心问题**："Transformer 是什么、为什么 work" — 你的讲法是什么？要能从零写出 attention 公式并解释每一步的设计理由，能手动模拟 BPE 合并过程，能解释为什么现代模型都是 decoder-only 而不是 encoder-decoder 或 encoder-only。

---
Transformer是一套神经网络模型，结合了机器学习发展过程中很多有效的trick。
它是一个seq2seq模型，相比传统的seq2seq模型的重点在于引入了注意力机制，下面我将从seq2seq的模式作为切入来跟大家讲一下我理解的Transformer.
Transformer相比于传统的seq2seq模型优势在哪里？
首先看看ELMO这种RNN门控模型，长间隔遗忘问题是硬伤，对于指代消解的问题突出，然后是串行执行，很难使用GPU并行化提升效率。
Transformer的重点是：
multi-head attention、旋转位置编码。
attenion可以捕获长序列中token之间的相关性，位置编码则可以让矩阵带上token本身的位置信息。
另外，attention中的矩阵运算天然可以在GPU上进行并行计算。
还有便是模型网络的可拓展性。

BPE算法不重要，我也仅是了解，简单说说吧。
BPE是一套不借助模型的分词算法，主要是用于英文的分词，优势是启动快，可以适配生词。
字符级别拆分，合并最高词频的相邻字符组成新的字符，再次循环合并最高词频的相邻字符，直到达到设定的词表大小。

为什么现在的大模型普遍是decoder-only
一方面因为scaling law 在decoder-only上得到了验证，随着优质数据集和模型规模的提升模型能力会明显提升。
另一方面，encoder-decoder在训练阶段，encoder部分存在data leak，token可以看到前后的内容，相比decoder-only，decoder-only是纯粹的预测未来token.因此decoder-only是一个更难的任务，因此它的上限更高。
但并不是说decoder-only完全替代encoder-decoder，encoder-decoder有天然擅长的领域，例如翻译场景、分类场景那种天然输入完整sentence进行完整考虑后，输出序列的场景。



下面，我们将详细讲讲几个transformer的重要细节：
1. 旋转位置编码
attention天然不包含位置信息，如果期望attention中能够包含位置信息，需要让矩阵的点积满足下面等式。
<f(x_n,n),f(x_m,m)> = g(x_n,x_m,(n-m))

想想两个向量的点积 x_n * y_m = |x_n| |y_m| cos(a)
注： 其中,x_n是位置为n的向量，y_m时位置为m的向量，
如果x_n旋转了n * b角度，y_m旋转了m * b角度，此时 x_n * y_m = |x_n| |y_m| cos(b(m-n))
旋转的角度公式： theta_i = 10000 ^ -2(i-1)/d
可以推广到attention的矩阵点积

2. multi-head attention

从self-attention出发，以矩阵shape的角度观察multi-head的计算过程：
(input-embedding + pos-embedding) -> (W_Q,W_K,W_V) -> (Q,K,V)  -> softmax(Q K^T /sqrt(dim)) V
（bs,seq_len,dim') ->  （bs,seq_len,head,dim'/head) -> emb（bs,head,seq_len,dim) W(dim,dim) ->(Q,K,V) （bs,head,seq_len,dim) 
-> softmax(Q K^T /sqrt(dim))  (bs,head,seq_len,seq_len)
-> softmax(Q K^T /sqrt(dim)) V （bs,head,seq_len,dim) 
-> （bs,head,seq_len,dim) ->（bs,seq_len,head,dim) -> （bs,seq_len,dim')  
-> 线性变换（bs,seq_len,dim')  

3. 之后时ADD & Norm
ADD 便是残差，其作用有说法是让模型知道attention后产生的差异
Norm，token向量维度做归一化处理，使得则是为了使得梯度下降更快更稳定。

---

## 校验记录 — 2026-05-16

校验员：Claude（参照 tracks/architecture.md 阶段 1、topics/attention-mechanism.md）

### 事实/归属错误

1. RoPE 不属于原始 Transformer（第 16 行）。Vaswani 2017 用的是 sinusoidal 绝对位置编码。RoPE 是 RoFormer（Su et al., 2021/2022），被 LLaMA 系采用后才成主流。"Transformer 的重点是 multi-head attention + 旋转位置编码"是历史归属错位。建议把 RoPE 单独拎到"后续演进/下一篇 position-encoding"。

2. encoder-decoder vs encoder-only 混淆（第 28 行）。分类是 encoder-only（BERT）的传统强项，不是 encoder-decoder（T5）。encoder-decoder 真正擅长的是结构化输入输出的 seq2seq（翻译、摘要、QA）。

3. "encoder 存在 data leak"措辞不准确（第 27 行）。双向看输入是 BERT/T5 的设计意图，不叫 data leak。你想说的更接近：encoder-decoder/MLM 类目标在输入侧双向可见，相比 causal LM 是"更容易"的目标。这个"难任务上限更高"的论证（Yi Tay 等）文献中存在但未达共识——tracks/architecture.md 明确写过"decoder-only 的统治更多是 GPT 系列成功带来的路径依赖，而非架构层面的绝对优越"。建议把"上限更高"改成更克制的"被认为更难，scaling 更稳定"。

4. BPE "主要用于英文"不准确（第 22 行）。BPE 是子词分词的通用方法，GPT/Llama/Qwen 多语言模型都在用。GPT-2 起更用 Byte-level BPE（从字节而非字符出发），天然兼容任意 Unicode 包括中文。

### 数学/技术细节不严谨

5. RoPE 推导跳步（第 37-41 行）。"x_n 旋转 nθ、y_m 旋转 mθ 后 ⟨x_n,y_m⟩ = |x_n||y_m|cos(θ(m-n))"——这一步只在 x_n、y_m 原本同向时成立。一般情况下 ⟨R_n x, R_m y⟩ = ⟨x, R_{m-n} y⟩，仍依赖 x、y 的原始角度。RoPE 的关键性质是旋转后 Q·K 只依赖 (Q, K, m-n)——正好满足你开头那个 g(x_n,x_m,n-m) 条件。建议把这步补全，否则像数学错误。

6. MHA shape 推导有两个问题（第 46-50 行）：
   - 顺序反了。常规是先 W_Q/W_K/W_V 投影到 (bs, seq_len, d_model)，再 reshape 成 (bs, seq_len, h, d_head)，再 transpose 成 (bs, h, seq_len, d_head)。你写的"先 reshape 后过 W"工程上一般不这么实现。
   - `softmax(QK^T / sqrt(dim))` 里的 `dim` 应该是 `d_head`，不是 `d_model`。

7. 残差作用解释不准确（第 54 行）。"让模型知道 attention 后产生的差异"不是主流解释。核心作用是**梯度流**——identity mapping 让梯度可直接绕过子层反向传播，使训练 100+ 层成为可能（ResNet 同款动机）。

### 覆盖 tasks 缺失（frontmatter 列了但正文没写）

- ✅ BPE 算法
- ✅ Self-Attention / Multi-Head Attention
- ⚠ Transformer 层结构：残差有，**FFN 完全没提**，**Pre-LN vs Post-LN 完全没提**
   - FFN 是 Transformer 层的主体，参数量占整层约 2/3——这是后续"FFN 作为键值记忆"（Geva 2021）和 MoE 把 FFN 换成专家网络的基础
   - Pre-LN（GPT-2 之后的标准）vs Post-LN（原论文）：Post-LN 深网络训练不稳定，Pre-LN 把 LN 放在子层前解决了这个问题
- ⚠ **自回归语言建模目标**只在 decoder-only 段落里隐含提了一句"纯粹预测未来 token"，没有正式定义（最大化 ΣlogP(x_t|x_<t) 的链式分解）。后续 scaling-laws、SFT 都依赖这个目标定义。
- ✅ decoder-only 为什么统治

8. 结尾突兀（第 55 行）。"使得梯度下降更快更稳定"之后直接结束，像没写完。如果是有意收尾，建议加一段呼应"核心问题"——把 Transformer 为什么 work 的几条线串起来（并行化 / attention 捕获长依赖 / 残差让深网络可训练 / scaling 友好）。

### 整体评价

骨架对、关键概念都摸到了，但有两类系统性问题需要先修才能 verified：
- **历史归属混淆**：把 RoPE、Byte-level BPE 这些后续演进当作原始 Transformer 的特性
- **细节经不起推敲**：RoPE 数学跳步、MHA shape 顺序、残差解释——L2 因果层口试会被追问

建议路径：先补 FFN / Pre-LN / 自回归目标三个缺失项 → 再修 1-7 的事实问题 → 改完跟我说，我再走一次校验确认推到 verified。