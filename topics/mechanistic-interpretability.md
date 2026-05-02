# 机械可解释性（Mechanistic Interpretability）

> 锚点：轨道 alignment（阶段 3） / 纪元 era4-reasoning

## 这个概念是什么

机械可解释性（Mechanistic Interpretability）是一个研究方向，目标是理解神经网络内部的具体计算机制——不只是"模型输出什么"，而是"模型用什么内部结构和算法产生这个输出"。代表性问题：模型识别"猫"这个词时激活了哪些神经元？它们之间的信息流是什么？Anthropic 是迄今对这个方向投入最多的前沿实验室，由 Chris Olah（Distill.pub 创始人）领导。

## 内部结构

<!-- 
核心研究问题：
1. 特征（Features）：神经网络表示什么概念？
   - 早期发现：单一神经元往往是"多义"的（polysemantic），编码多个不相关概念
   - 超位置（Superposition）假说：网络把比维度数更多的特征压缩编码

2. 电路（Circuits）：特征之间如何交互？
   - 算法机制：注意力头如何实现 induction（重复上文模式）、间接目标查找等具体算法
   - 发现案例：GPT-2 Small 中的 curve detectors、frequency circuits 等

3. 稀疏自编码器（Sparse Autoencoders, SAE）：
   - 解决超位置问题的工具：训练 SAE 把中间表示分解成稀疏的可解释特征
   - 2024 年 Anthropic 用 SAE 分析 Claude 3 Sonnet，找到了数百万个可命名特征
   - 这是目前规模化可解释性的主要技术路线

4. 可解释性与安全的连接（待填充）：
   - 如何用可解释性工具检测"欺骗性对齐"（deceptive alignment）？
   - Activation steering：通过修改内部激活改变模型行为
-->

## 当前状态（截至 2026-05）

<!-- 
- Anthropic 的可解释性团队是世界上规模最大的
- 稀疏自编码器（SAE）成为主要研究工具
- 已经能识别 Claude 中数百万个可命名特征
- 但距离"完全可解释的 LLM"仍然非常遥远
- 关键未解问题：特征和电路能否 scale 到更大模型的理解？
-->

## 关键权衡

<!-- 
研究价值 vs 实用价值：
- 目前的可解释性工作主要是科学理解，直接的安全工程应用仍然有限
- 超位置现象使完全分解变得极难

Anthropic 独家 vs 社区方向：
- Anthropic 发表了大量基础研究，使学术界可以跟进
- 但在模型规模上他们有实验室优势
-->

## 信息源

[Zoom In: An Introduction to Circuits (Distill, 2020)](https://distill.pub/2020/circuits/zoom-in/)，[Toy Models of Superposition (Anthropic, 2022)](https://transformer-circuits.pub/2022/toy_model/index.html)，[Scaling and evaluating sparse autoencoders (Anthropic, 2024)](https://arxiv.org/abs/2406.04093)

## 更新日志

- 2026-05-02：创建占位文件（定义 + 研究问题草图，待深度展开）
