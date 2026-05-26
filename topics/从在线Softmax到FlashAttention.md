# 从在线 Softmax 到 FlashAttention

**作者：** Zihao Ye
**邮箱：** zhye@cs.washington.edu
**日期：** 2023 年 5 月 11 日
**课程：** UW CSE 599M 2023 年春季：ML for ML Systems

---

FlashAttention [1] 的核心创新在于借鉴了类似 Online Softmax [3] 的思想来对自注意力（self-attention）计算进行分块（tiling），从而可以将整个多头注意力层融合（fuse）为一个 CUDA kernel，而无需将中间的 logits 和注意力分数写入 GPU 全局内存。在本笔记中，我将简要说明为什么对自注意力计算进行分块并非易事，以及如何从 Online Softmax 技巧推导出 FlashAttention 的计算方式。感谢 Andrew Gu 对本笔记的校对。

## 1 自注意力（Self-Attention）

自注意力的计算可以总结为如下形式（这里忽略了头数和 batch 维度，因为这些维度上的计算是完全并行的；同时为简洁起见，也省略了注意力掩码和缩放因子 $\frac{1}{\sqrt{D}}$ 等细节）：

$$
O = \mathrm{softmax}(Q K^T) V \tag{1}
$$

其中 $Q, K, V, O$ 都是形状为 $(L, D)$ 的二维矩阵，$L$ 表示序列长度，$D$ 表示每个 head 的维度（即 head dimension）。softmax 作用在最后一维（按列）上。

计算自注意力的标准做法是将其分解为以下几个阶段：

$$
X = Q K^T \tag{2}
$$

$$
A = \mathrm{softmax}(X) \tag{3}
$$

$$
O = A V \tag{4}
$$

我们将矩阵 $X$ 称为 **pre-softmax logits**，矩阵 $A$ 称为 **attention score**，矩阵 $O$ 称为 **output**。

FlashAttention 一个令人惊艳的事实是：我们不需要在全局内存中物化（materialize）$X$ 和 $A$ 矩阵，而是将公式 (1) 中的整个计算融合到一个 CUDA kernel 中。这要求我们设计一种能够仔细管理片上（on-chip）内存的算法（类似流式算法），因为 NVIDIA GPU 的共享内存（shared memory）非常小。

对于经典算法如矩阵乘法，我们用分块（tiling）来确保片上内存不超过硬件限制。图 1 给出了一个示例。在 kernel 执行期间，无论矩阵形状如何，片上始终只存储 $3T^2$ 个元素。这种分块方法之所以有效，是因为加法满足结合律，因此可以将整个矩阵乘法分解为多个分块矩阵乘法的求和。

然而，自注意力中包含了 softmax 算子，它并不直接满足结合律，因此很难像图 1 那样简单地对自注意力进行分块。**那么有没有办法让 softmax 满足结合律呢？**

> **图 1.** 该图简要解释了如何对矩阵乘法 $C = A \times B$ 的输入与输出矩阵进行分块。矩阵被划分为 $T \times T$ 的小块（tile）。对每个输出 tile，我们从左到右扫描 $A$ 中相关的 tile，从上到下扫描 $B$ 中相关的 tile，并将这些值从全局内存加载到片上内存（图中蓝色部分，整体片上内存占用为 $O(T^2)$）。对于分块的部分矩阵乘法，对于位置 $(i, j)$，我们从片上内存中加载 tile 内所有 $k$ 对应的 $A[i, k]$ 和 $B[k, j]$（图中红色），然后在片上内存中将 $A[i, k] \cdot B[k, j]$ 累加到 $C[i, j]$。一个 tile 的计算完成后，我们将片上的 $C$ tile 写回主内存，再处理下一个 tile。实际应用中的分块要复杂得多，可以参考 Cutlass 在 A100 上实现矩阵乘法的代码 [2]。

## 2 (Safe) Softmax

让我们先回顾一下 softmax 算子，下面是 softmax 计算的一般公式：

$$
\mathrm{softmax}(\{x_1, \ldots, x_N\}) = \left\{ \frac{e^{x_i}}{\sum_{j=1}^{N} e^{x_j}} \right\}_{i=1}^{N} \tag{5}
$$

注意 $x_i$ 可能非常大，而 $e^{x_i}$ 很容易上溢（overflow）。例如，float16 所能表示的最大值是 65536，这意味着当 $x > 11$ 时，$e^x$ 就会超出 float16 的有效表示范围。为了缓解这个问题，数学软件通常采用一种称为 **safe softmax** 的技巧：

$$
\frac{e^{x_i}}{\sum_{j=1}^{N} e^{x_j}} = \frac{e^{x_i - m}}{\sum_{j=1}^{N} e^{x_j - m}} \tag{6}
$$

其中 $m = \max_{j=1}^{N}(x_j)$，这样我们可以保证每个 $x_i - m \le 0$，从而是安全的——因为指数函数对负值输入是数值精确的。

我们可以将 safe softmax 的计算总结为如下的 **3-pass（三遍扫描）算法**：

### 算法：3-pass safe softmax

**符号说明：**
- $\{m_i\}$：$\max_{j=1}^{i}\{x_j\}$，初始值 $m_0 = -\infty$。
- $\{d_i\}$：$\sum_{j=1}^{i} e^{x_j - m_N}$，初始值 $d_0 = 0$，$d_N$ 即 safe softmax 的分母。
- $\{a_i\}$：最终的 softmax 值。

**算法主体：**

```
for i ← 1, N do
    m_i ← max(m_{i-1}, x_i)         (7)
end

for i ← 1, N do
    d_i ← d_{i-1} + e^{x_i - m_N}    (8)
end

for i ← 1, N do
    a_i ← e^{x_i - m_N} / d_N         (9)
end
```

这个算法需要在 $[1, N]$ 上迭代 3 次。在 Transformer 的自注意力场景下，$\{x_i\}$ 是由 $Q K^T$ 计算得到的 pre-softmax logits。这意味着，如果我们无法一次性保存所有的 logits $\{x_i\}_{i=1}^{N}$（因为 SRAM 不够大放不下全部），就需要访问 $Q$ 和 $K$ 三次（每次都需要重新即时计算 logits），这在 I/O 上非常低效。

## 3 Online Softmax

如果能把公式 (7)、(8) 和 (9) 融合到一个循环中，就可以将全局内存访问次数从 3 次减少到 1 次。但不幸的是，我们无法将公式 (7) 和 (8) 融合到同一个循环里，因为 (8) 依赖于 $m_N$，而 $m_N$ 必须等到第一个循环结束才能确定。

我们可以构造另一个序列 $d_i' := \sum_{j=1}^{i} e^{x_j - m_i}$ 作为原序列 $d_i := \sum_{j=1}^{i} e^{x_j - m_N}$ 的代理（surrogate），以消除对 $N$ 的依赖。这两个序列的第 $N$ 项是相同的：$d_N = d_N'$，因此我们可以放心地用 $d_N'$ 替换公式 (9) 中的 $d_N$。我们还可以找到 $d_i'$ 与 $d_{i-1}'$ 之间的递推关系：

$$
\begin{aligned}
d_i' &= \sum_{j=1}^{i} e^{x_j - m_i} \\
&= \left(\sum_{j=1}^{i-1} e^{x_j - m_i}\right) + e^{x_i - m_i} \\
&= \left(\sum_{j=1}^{i-1} e^{x_j - m_{i-1}}\right) e^{m_{i-1} - m_i} + e^{x_i - m_i} \\
&= d_{i-1}' \cdot e^{m_{i-1} - m_i} + e^{x_i - m_i}
\end{aligned} \tag{10}
$$

这个递推形式只依赖于 $m_i$ 和 $m_{i-1}$，所以我们可以在同一个循环中同时计算 $m_j$ 和 $d_j'$：

### 算法：2-pass online softmax

```
for i ← 1, N do
    m_i ← max(m_{i-1}, x_i)
    d_i' ← d_{i-1}' · e^{m_{i-1} - m_i} + e^{x_i - m_i}
end

for i ← 1, N do
    a_i ← e^{x_i - m_N} / d_N'
end
```

这就是 Online Softmax 论文 [3] 中提出的算法。然而，它仍然需要两遍扫描才能完成 softmax 的计算。**我们能否将扫描次数减少到 1 次，从而最小化全局 I/O 呢？**

## 4 FlashAttention

很遗憾，对于单纯的 softmax 来说答案是否定的。但在自注意力中，我们最终的目标并不是注意力分数矩阵 $A$，而是输出矩阵 $O = A V$。**那么，我们能否为 $O$ 找到一个单遍扫描（one-pass）的递推形式呢？**

让我们尝试把自注意力计算的第 $k$ 行（所有行的计算是相互独立的，为简单起见我们只描述一行的计算）写成递推算法形式：

### 算法：多遍扫描的自注意力

**符号说明：**
- $Q[k, :]$：$Q$ 矩阵的第 $k$ 行向量。
- $K^T[:, i]$：$K^T$ 矩阵的第 $i$ 列向量。
- $O[k, :]$：输出矩阵 $O$ 的第 $k$ 行。
- $V[i, :]$：$V$ 矩阵的第 $i$ 行。
- $\{o_i\}$：$\sum_{j=1}^{i} a_j \cdot V[j, :]$，是存储部分聚合结果 $A[k, :i] \cdot V[:i, :]$ 的行向量。

**算法主体：**

```
for i ← 1, N do
    x_i ← Q[k, :] · K^T[:, i]
    m_i ← max(m_{i-1}, x_i)
    d_i' ← d_{i-1}' · e^{m_{i-1} - m_i} + e^{x_i - m_i}
end

for i ← 1, N do
    a_i ← e^{x_i - m_N} / d_N'        (11)
    o_i ← o_{i-1} + a_i · V[i, :]      (12)
end

O[k, :] ← o_N
```

让我们用公式 (11) 中的定义替换公式 (12) 里的 $a_i$：

$$
o_i := \sum_{j=1}^{i} \left( \frac{e^{x_j - m_N}}{d_N'} \right) V[j, :] \tag{13}
$$

这仍然依赖于 $m_N$ 和 $d_N'$，它们要等到上一个循环结束才能确定。但我们可以再次使用第 3 节的代理（surrogate）技巧，构造一个代理序列 $o'$：

$$
o_i' := \sum_{j=1}^{i} \left( \frac{e^{x_j - m_i}}{d_i'} \right) V[j, :]
$$

$o$ 和 $o'$ 的第 $N$ 项是相同的：$o_N' = o_N$，并且我们可以找到 $o_i'$ 与 $o_{i-1}'$ 之间的递推关系：

$$
\begin{aligned}
o_i' &= \sum_{j=1}^{i} \frac{e^{x_j - m_i}}{d_i'} V[j, :] \\
&= \left(\sum_{j=1}^{i-1} \frac{e^{x_j - m_i}}{d_i'} V[j, :]\right) + \frac{e^{x_i - m_i}}{d_i'} V[i, :] \\
&= \left(\sum_{j=1}^{i-1} \frac{e^{x_j - m_{i-1}}}{d_{i-1}'} \cdot \frac{e^{x_j - m_i}}{e^{x_j - m_{i-1}}} \cdot \frac{d_{i-1}'}{d_i'} V[j, :]\right) + \frac{e^{x_i - m_i}}{d_i'} V[i, :] \\
&= \left(\sum_{j=1}^{i-1} \frac{e^{x_j - m_{i-1}}}{d_{i-1}'} V[j, :]\right) \cdot \frac{d_{i-1}'}{d_i'} \cdot e^{m_{i-1} - m_i} + \frac{e^{x_i - m_i}}{d_i'} V[i, :] \\
&= o_{i-1}' \cdot \frac{d_{i-1}' \cdot e^{m_{i-1} - m_i}}{d_i'} + \frac{e^{x_i - m_i}}{d_i'} V[i, :]
\end{aligned} \tag{14}
$$

这个递推式只依赖于 $d_i'$、$d_{i-1}'$、$m_i$、$m_{i-1}$ 和 $x_i$，因此我们可以将自注意力中的所有计算融合到一个循环中：

### 算法：FlashAttention

```
for i ← 1, N do
    x_i ← Q[k, :] · K^T[:, i]
    m_i ← max(m_{i-1}, x_i)
    d_i' ← d_{i-1}' · e^{m_{i-1} - m_i} + e^{x_i - m_i}
    o_i' ← o_{i-1}' · (d_{i-1}' · e^{m_{i-1} - m_i} / d_i')
            + (e^{x_i - m_i} / d_i') · V[i, :]
end

O[k, :] ← o_N'
```

状态 $x_i$、$m_i$、$d_i'$ 和 $o_i'$ 占用的内存很小，完全可以放进 GPU 的共享内存。由于该算法中的所有操作都满足结合律，它与分块（tiling）是兼容的。如果我们逐块（tile-by-tile）计算这些状态，算法可以表示为：

### 算法：FlashAttention（分块版本）

**新增符号说明：**
- $b$：tile 的块大小（block size）。
- `#tiles`：行方向上的 tile 数量，$N = b \cdot \text{\#tiles}$。
- $x_i$：存储第 $i$ 个 tile $[(i-1) \cdot b : i \cdot b]$ 内 $Q[k] \cdot K^T$ 值的向量。
- $m_i^{(\text{local})}$：$x_i$ 内部的局部最大值。

**算法主体：**

```
for i ← 1, #tiles do
    x_i ← Q[k, :] · K^T[:, (i-1)·b : i·b]
    m_i^(local) = max_{j=1}^{b} (x_i[j])
    m_i ← max(m_{i-1}, m_i^(local))
    d_i' ← d_{i-1}' · e^{m_{i-1} - m_i} + Σ_{j=1}^{b} e^{x_i[j] - m_i}
    o_i' ← o_{i-1}' · (d_{i-1}' · e^{m_{i-1} - m_i} / d_i')
            + Σ_{j=1}^{b} (e^{x_i[j] - m_i} / d_i') · V[j + (i-1)·b, :]
end

O[k, :] ← o_{N/b}'
```

图 2 展示了如何将这个算法映射到硬件上。

> **图 2.** 上图展示了 FlashAttention 在硬件上的计算方式。蓝色块表示位于 SRAM 中的 tile，红色块对应于第 $i$ 行。$L$ 表示序列长度，可以非常大（例如 16k）；$D$ 表示 head 维度，在 Transformer 中通常较小（例如 GPT-3 中为 128）；$B$ 是可控的块大小。值得注意的是，整体 SRAM 内存占用只取决于 $B$ 和 $D$，与 $L$ 无关。因此，该算法可以扩展到长上下文而不会遇到内存问题（GPU 共享内存很小，H100 架构每个 SM 仅 228KB）。计算过程中，我们对 $K^T$ 和 $A$ 从左到右扫描 tile，对 $V$ 从上到下扫描，并相应地更新 $m$、$d$ 和 $O$ 的状态。

## 参考文献

[1] Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, and Christopher Ré. *FlashAttention: fast and memory-efficient exact attention with IO-awareness.* CoRR, abs/2205.14135, 2022.

[2] Andrew Kerr. *GTC 2020: Developing CUDA kernels to push tensor cores to the absolute limit on NVIDIA A100.* May 2020.

[3] Maxim Milakov and Natalia Gimelshein. *Online normalizer calculation for softmax.* CoRR, abs/1805.02867, 2018.
