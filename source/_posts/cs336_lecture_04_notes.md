---
title: Stanford CS336 (Spring 2026) Lecture 4 深度课程笔记：Attention Alternatives & Mixture of Experts
date: '2026-10-25 15:28:00'
categories:
  - 笔记
  - CS336
tags:
  - 大语言模型
  - CS336
  - Attention
  - MoE
  - Mamba
  - DeepSeek
description: Stanford CS336 深度笔记第 4 讲：打破标准 Transformer 的扩展瓶颈。全面拆解线性注意力（Linear Attention）、状态空间模型（Mamba-2 与 Gated DeltaNet）、动态稀疏注意力（DSA），以及混合专家模型（MoE）的路由机制、细粒度切分、无辅助损失动态偏置与 DeepSeek 架构演进。
mathjax: true
---

# Stanford CS336 (Spring 2026) Lecture 4 深度课程笔记：Attention Alternatives & Mixture of Experts

> **课程信息**：[Stanford CS336: Language Modeling from Scratch (Spring 2026)](https://cs336.stanford.edu/)
> **主讲教授**：Tatsunori Hashimoto (Tatsu)
> **本讲资源**：[Lecture 4 YouTube 视频](https://www.youtube.com/watch?v=cKSwj_qZ8Jg) | [官方幻灯片 (`lecture_04.pdf`)](https://github.com/stanford-cs336/lectures/blob/main/lecture_04.pdf) | [课程主页](https://cs336.stanford.edu/)

## 1. 引言：打破标准 Transformer 的两大扩展瓶颈

在 Lecture 3 中，我们梳理了稠密 Transformer（Dense Transformer）的标准组件与超参数法则。然而，当我们将模型推向**百万级超长上下文（Million-Token Long Context）**与**万亿级参数容量（Trillion-Parameter Capacity）**时，标准 Transformer 撞上了两堵高墙：

1. **序列长度的二次方墙（Sequence Length Bottleneck $\rightarrow$ Part I: Attention Alternatives）**：
   * 随着 Agent 工作流、代码库级理解与长文档推理兴起，主流模型的上下文窗口在对数坐标下呈指数级扩张（从 4K $\rightarrow$ 128K $\rightarrow$ 1M～10M）。
   * 在短序列下，FFN 占据了绝大部分计算量；但由于自注意力是全对全（All-to-all）的 $O(n^2)$ 交互，随着序列长度 $n$ 增长，**Attention 的计算与显存占比迅速反超 FFN 并成为绝对瓶颈**。
2. **参数容量的算力绑定墙（Model Capacity Bottleneck $\rightarrow$ Part II: Mixture of Experts）**：
   * 在稠密模型中，参数量 $N$ 与单 Token 浮点计算量（$\approx 6N$ FLOPs）是死死绑定的。想要让模型记住更多世界知识就必须增大 $N$，但这会同步引爆训练与推理的算力成本。
   * **混合专家模型（MoE）**通过条件稀疏激活（Conditional Sparse Activation），实现了**“总参数量（知识容量）与单 Token 激活算力（FLOPs）的彻底解耦”**。

> **核心主线（Unifying Theme）**：无论是 Part I 的线性/稀疏注意力，还是 Part II 的混合专家（MoE），现代前沿架构演进的本质都是**通过结构化状态压缩或动态 Top-$K$ 稀疏选择，在固定硬件预算（FLOPs、HBM 带宽、跨节点通信）下最大化有效模型能力**。

## 2. Part I：注意力替代方案与超长上下文架构（Attention Alternatives）

### 2.1 守住底线：常数因子优化（FlashAttention）与局部混合注意力

在追求激进的 $O(n)$ 线性复杂度之前，Tatsu 强调了一个核心系统工程信条：**永远不要只看大 $O$ 渐近复杂度而忽视常数因子（Constant Factors Matter!）**。
* **系统级常数优化（[FlashAttention](https://arxiv.org/abs/2205.14135)）**：不改变 $O(n^2)$ 的数学定义，仅通过分块（Tiling）在 GPU 高速 SRAM 中完成 Softmax 归约，避免在 HBM 中实例化 $n \times n$ 的注意力矩阵。这不仅带来了 **2～4 倍的吞吐率跃升（从 30～40 TFLOP/s 提升至百 TFLOP/s 级）**，更消除了 $O(n^2)$ 的显存占用，使长序列训练在物理上成为可能（将在系统章节与 Assignment 2 深入实现）。
* **局部与全局混合（Local/Sliding Window + Global Attention）**：如 Lecture 3 所述，每隔 4～8 层放一层全局注意力，其余层使用固定窗口的局部注意力。但当序列长度推向 **5M～10M Tokens** 时，仅靠常数优化已不足以支撑，我们必须向**线性时间注意力（Linear-Time Attention）**与**动态稀疏注意力（Sparse Attention）**进军。

### 2.2 线性注意力的核心基石：矩阵乘法结合律与并行-循环对偶性（Duality）

所有现代线性注意力与状态空间模型（SSM，如 Mamba-2、Gated DeltaNet）的起点，都源于一个极其简单的代数性质——**矩阵乘法的结合律（Associativity of Matrix Multiplication）**。

#### 1. 从 $O(n^2 d_k)$ 到 $O(n d_k d_v)$：交换括号顺序
设查询、键、值矩阵分别为 $Q \in \mathbb{R}^{n \times d_k}$，$K \in \mathbb{R}^{n \times d_k}$，$V \in \mathbb{R}^{n \times d_v}$。标准自注意力写作：
$$\text{Attn}(Q, K, V) = \rho(Q K^\top) V$$
其中 $\rho(\cdot)$ 是逐行 Softmax 算子。因为 $\rho$ 的存在，我们被迫先算 $Q K^\top \in \mathbb{R}^{n \times n}$，产生 **$O(n^2 d_k + n^2 d_v)$** 的二次方复杂度。

现在，假设我们暂时**去掉 Softmax**（令 $\rho$ 为恒等映射 Identity，或对 $Q, K$ 施加某种逐行特征映射核函数 $\phi(Q)\phi(K)^\top$，见 [Katharopoulos et al., 2020](https://arxiv.org/abs/2006.16236)），利用矩阵乘法结合律移动括号：

$$(Q K^\top) V = Q (K^\top V)$$

```
[标准注意力: 先算 (Q K^T)]                       [线性注意力: 先算 (K^T V)]
  ┌───┐ ┌───────────┐     ┌───────────┐           ┌───────────┐     ┌───┐
  │   │ │     K^T   │     │           │           │     K^T   │     │   │
n │ Q │ └───────────┘  =  │   n x n   │        d_k│  (d_k x n)│  x  │ V │ n  =  [S: d_k x d_v]
  │   │       d_k         │ Attn Mat  │           └───────────┘     │   │       (固定大小状态矩阵!)
  └───┘                   └───────────┘                             └───┘
  复杂度: O(n^2 d_k) -> 随序列长度 n 二次爆炸!      复杂度: O(n d_k d_v) -> 关于序列长度 n 严格线性!
```

* **复杂度质变**：我们先计算 $K^\top V \in \mathbb{R}^{d_k \times d_v}$（耗时 $n d_k d_v$），再左乘 $Q \in \mathbb{R}^{n \times d_k}$（耗时 $n d_k d_v$）。总计算量变为 **$2 n d_k d_v = O(n d_k d_v)$**！因为隐藏维度 $d_k, d_v$ 通常只有几百到几千，而序列长度 $n$ 可达数百万，这就将关于 $n$ 的复杂度从二次方彻底降为了**线性**！

#### 2. 并行形式（Parallel Form）与循环形式（Recurrent / RNN Form）的精确对偶
在自回归因果掩码（Causal Masking，即第 $t$ 个位置只能看到 $j \le t$ 的键值）下，第 $t$ 步的输出向量 $y_t \in \mathbb{R}^{d_v}$ 为：
$$y_t^\top = \sum_{j=1}^t (q_t^\top k_j) v_j^\top = q_t^\top \underbrace{\left( \sum_{j=1}^t k_j v_j^\top \right)}_{S_t \in \mathbb{R}^{d_k \times d_v}}$$
令固定大小的矩阵 $S_t \in \mathbb{R}^{d_k \times d_v}$ 表示累积到时间步 $t$ 的**记忆状态矩阵（State Matrix）**，我们就得到了与上述并行矩阵乘法**数学上完全等价（Exact Equivalence）**的 **RNN 递推形式**：

$$S_t = S_{t-1} + k_t v_t^\top, \qquad y_t = (q_t^\top S_t)^\top = S_t^\top q_t$$

* **两全其美的对偶性（The Best of Both Worlds）**：
  * **训练阶段（Training）**：使用左侧的**并行矩阵形式**（或分块 Chunkwise 并行），跑满 GPU Tensor Core；
  * **自回归推理阶段（Inference）**：使用右侧的**循环状态形式**，只需在显存中维护一个固定尺寸 $d_k \times d_v$ 的状态矩阵 $S_t$（**告别随序列长度线性膨胀的 KV Cache！**），每生成一个新 Token 的计算与访存开销均为严格的 **$O(1)$**！
* **工业界大规模实证：[MiniMax-M1 / MiniMax-Text-01 (2025)](https://arxiv.org/abs/2501.08313)**
  MiniMax 团队证明了即便是这种最基础的 Linear Attention（配合 Lightning Attention 分块技巧），只要采用 **7:1 混合架构（每 7 层 Linear Attention 配 1 层全局 Softmax Attention）**，就能在 456B 总参数规模下匹敌 DeepSeek-R1 与 OpenAI o3-mini，同时在百万级上下文下保持接近线性的推理开销。

### 2.3 从朴素线性注意力演进到 Mamba-2 与 Gated DeltaNet

朴素线性注意力 $S_t = S_{t-1} + k_t v_t^\top$ 有一个致命缺陷：**它只有纯加法累积，永远不会“遗忘”（No Forgetting Mechanism）**！随着时间推移，所有历史键值对无差别地叠加在固定大小的 $S_t$ 中，导致记忆迅速饱和、信噪比急剧恶化。

受经典 LSTM 门控思想的启发，近年来的状态空间模型（SSM）与线性注意力在同一个递推框架下实现了大合流——**核心法则是：只要门控项仅依赖于当前输入 $x_t$（而不非线性依赖于旧状态 $S_{t-1}$），就能同时保住训练并行性与推理 $O(1)$ 循环性！**

| 架构模型 | 状态更新递推方程 ($S_t \in \mathbb{R}^{d_k \times d_v}$) | 输出方程 ($y_t$) | 核心演进机制与物理直觉 | 代表性工业界混合大模型 |
| :--- | :--- | :--- | :--- | :--- |
| **1. Linear Attention** | $S_t = S_{t-1} + k_t v_t^\top$ | $y_t = S_t^\top q_t$ | 纯外积加法累加，无遗忘能力，长序列下历史噪声堆积 | **MiniMax-M1** (7:1 混合) |
| **2. [Mamba-2](https://arxiv.org/abs/2405.21060)** | $S_t = \gamma_t S_{t-1} + k_t v_t^\top$ | $y_t = S_t^\top q_t + D \odot v_t$ | 引入依赖输入的标量/逐通道**遗忘门 $\gamma_t = f(x_t) \in (0, 1)$** 衰减旧状态；输出端增加当前值 $v_t$ 的直通残差项 $D \odot v_t$ | **[NVIDIA Nemotron 3](https://arxiv.org/abs/2603.00001)** (Mamba-2 + Attention 混合) |
| **3. [Gated DeltaNet](https://arxiv.org/abs/2412.06464)** | $S_t = \gamma_t (I - \beta_t k_t k_t^\top) S_{t-1} + \beta_t k_t v_t^\top$ | $y_t = S_t^\top q_t$ | 引入写入门 $\beta_t = f(x_t)$ 与 **Householder 投影擦除算子 $(I - \beta_t k_t k_t^\top)$**：在写入新值前**定向擦除当前键 $k_t$ 方向上的旧记忆** | **阿里 Qwen 3.5 / Qwen-Next** (3:1 混合：3 层 Gated DeltaNet + 1 层 Full Attention) |

#### 深度剖析 Gated DeltaNet 的“定向擦除与重写（Delta Rule）”机制：
为什么 [Gated DeltaNet (Yang et al., 2024)](https://arxiv.org/abs/2412.06464) 成为了目前开源界（如 **Qwen 3.5 / Qwen-Next**）最受青睐的线性层架构？
1. **双重门控职责分明**：
   * $\gamma_t = f(x_t)$ 是全局/通道级的**快速遗忘门**（当遇到话题切换或段落边界时，快速清空旧状态）；
   * $\beta_t = f(x_t)$ 是**输入写入门**（当 $\beta_t = 0$ 时执行 No-input 操作，不写入垃圾信息）。
2. **投影擦除算子 $(I - \beta_t k_t k_t^\top)$ 的物理与数学本质**：
   把状态更新式改写为：
   $$S_t = \gamma_t S_{t-1} + \beta_t k_t \big(v_t^\top - (\gamma_t S_{t-1}^\top k_t)^\top\big)$$
   想象状态矩阵 $S_{t-1}$ 是一个键值关联哈希表。现在我们要更新键 $k_t$ 对应的值为 $v_t$。如果我们像 Mamba-2 那样直接加 $k_t v_t^\top$，旧的 $(k_t, v_{\text{old}})$ 依然残留在表里；要么就得把整个状态矩阵乘个很小的 $\gamma_t$ 全部忘掉（把其他有用的键也误伤了）。
   而乘上投影矩阵 $(I - \beta_t k_t k_t^\top)$ 相当于**先用 $S_{t-1}^\top k_t$ 读出当前键 $k_t$ 在记忆库里的旧残余值，精准地将 $k_t$ 子空间方向上的旧内容减去（Project out / Erase），然后再把新的 $\beta_t k_t v_t^\top$ 写进去**！
   * *跨学科共鸣*：这在数学上等价于使用单步梯度下降求解在线最小二乘回归 $\min_S \frac{1}{2}\|S^\top k_t - v_t\|_2^2$（即 Widrow-Hoff **Delta Rule**），也与元学习中的 **Fast Weight Programmers (Schmidhuber, 1992)** 及 **Test-Time Training (TTT)** 殊途同归！

#### 关于混合比例（Hybrid Ratio）的实验定律：
为什么所有这些模型（MiniMax-M1、Nemotron 3、Qwen 3.5）都必须是**混合架构（Hybrid）**，而不能用 100% 的 Mamba-2 或 Gated DeltaNet？
* 根据最新的受控消融研究，当我们将全注意力层替换为线性/循环层的比例逐步提高时：在 **3:1（75% 线性层）到 7:1（87.5% 线性层）** 区间内，模型的长上下文检索与问答（QA）性能几乎与 100% Full Attention 持平，同时拿到了数倍的推理吞吐加速；
* 但**一旦迈向最后一格（100% 纯 RNN/SSM，0 层 Full Attention），长上下文精确检索与问答性能就会出现断崖式下跌**！因为无论固定尺寸的状态矩阵 $S_t$ 设计得多精巧，把任意长文本无损压缩进有限维度必然存在信息论上限（有损压缩），必须定期穿插少量的全对全 Softmax Attention 来完成精确的跨位置内容寻址。

### 2.4 另一条路线：动态稀疏注意力与 DeepSeek Sparse Attention (DSA)

如果不采用 RNN 式的状态压缩，还有什么办法降低长上下文开销？答案是**动态稀疏注意力（Dynamic Sparse Attention）**，其代表作是 **[DeepSeek-V3.2 (2025)](https://arxiv.org/abs/2512.00001)** 提出并被智谱旗舰开源模型 **GLM-5** 广泛验证的 **DSA (DeepSeek Sparse Attention)**。

```
[DeepSeek Sparse Attention (DSA) 两阶段前向流水线]
输入序列 (长 n) ──► [1. 轻量级 Lightning Indexer] ──► 计算低维/低精度粗略打分 I_{t,j} = ReLU(q_t^I · k_j^I) · w_j
                                                               │
                                                               ▼
                    [3. 精确主注意力 (Main Attention)] ◄── [2. Top-K 索引筛选] (仅挑出得分最高的 K 个历史位置, K << n)
                    仅在被选中的 K 个 Token 子集上执行全精度 Softmax Attention，输出最终上下文表征!
```

* **核心机制（Lightning Indexer + Top-$K$ Exact Attention）**：
  1. **轻量级索引器（Lightning Indexer）**：不直接用庞大的主注意力头去算全序列，而是旁路设置一个极轻量的索引器（使用极低维度投影、低精度甚至 FP4/INT8），快速计算当前查询与所有历史位置的粗略相关性得分（基于 $Q, K$ 内积过 ReLU 并乘以历史位置权重 $w$）。
  2. **Top-$K$ 稀疏截断**：从长为 $n$ 的历史序列中，仅挑出索引得分最高的 $K$ 个位置（$K \ll n$，且 $K$ 是一个类似短上下文窗口的固定常数上限）。
  3. **子集精确注意力**：主注意力机制（如 MLA/GQA）仅在这被挑出的 $K$ 个 Token 子集上执行标准的 Softmax Attention。
* **课堂关键问答：为什么 Indexer 本身也是 $O(n^2)$，DSA 却能大幅加速？**
  * **Tatsu 解答**：这再次印证了**“常数因子决定生死”**！虽然 Indexer 确实要扫描全部 $n$ 个位置（渐近复杂度仍是 $O(n^2)$），但它的维度极小、精度极低，计算常数因子比主注意力小一到两个数量级；而在昂贵的主注意力阶段，序列长度已经从百万级的 $n$ 被截断为了常数级的 $K$！
* **工程训练秘籍（Continued Pre-training Adaptation）**：
  DSA 与 GLM-5 的消融实验揭示了一个巨大的工程优势：**你不需要从零预训练开始就带着复杂的 DSA！** 你可以先用标准稠密注意力完成短上下文预训练，等到**长上下文扩展阶段（Long-Context Extension Stage）**再把 Lightning Indexer 挂载上去做适应性训练，长文本大海捞针与推理性能几乎无损，而 Prefill 和 Decode 耗时双双大幅下降。

## 3. Part II：混合专家模型（Mixture of Experts, MoE）全面拆解

### 3.1 MoE 的核心原理、优势与东西方开源生态

#### 1. 什么是 MoE？
在标准 Transformer Block 中，每个 Token 都流经同一个稠密 FFN。而在 MoE 层中，我们将原有的大 FFN 替换为 **$N$ 个独立的专家前馈网络 $\{\text{FFN}_1, \dots, \text{FFN}_N\}$** 以及一个轻量的**路由器（Router / Gate）**：
$$h_t = u_t + \sum_{i=1}^N g_{i, t} \, \text{FFN}_i(u_t)$$
其中 $u_t \in \mathbb{R}^d$ 是当前 Token 进入 FFN 前的隐状态，门控权重 $g_{i, t} \ge 0$ 是高度稀疏的——对于每个 Token $t$，**仅有 $K$ 个专家（$K \ll N$）的 $g_{i, t} > 0$ 被激活计算，其余 $N - K$ 个专家完全不参与该 Token 的前向与反向计算**！

* **为什么 MoE 统治了前沿大模型？**
  1. **同算力下碾压级的性能优势**：无论是早期的 [Switch Transformer (Fedus et al., 2022)](https://arxiv.org/abs/2101.03961) 还是最新的开源标杆 [OLMoE (Muennighoff et al., 2024)](https://arxiv.org/abs/2409.02060)，受控实验均表明：在保持单 Token 激活参数量（即单步 FLOPs）完全不变的前提下，增加总专家数 $N$（例如从稠密扩增到 8、64、256 个专家）能使验证集 Loss 持续下降，**达到相同性能所需的训练算力与时间比稠密模型减少 2～4 倍**！
  2. **天然提供第四维分布式并行——专家并行（Expert Parallelism, EP）**：当模型大到单机装不下时，数据并行（DP）受限于最大全局 Batch Size，张量并行（TP）受限于单机 NVLink 带宽，流水线并行（PP）受限于气泡；而 MoE 中每个 Expert 是天然解耦的子网络，可以直接将不同的 Expert 放置在不同 GPU 节点上。
* **东西方生态对比**：
  * 西方早期有 Google 的 GShard / Switch / ST-MoE，近年有 Mistral 的 [Mixtral 8x7B](https://arxiv.org/abs/2401.04088)、Databricks DBRX、xAI Grok、Meta **Llama 4** 以及 OpenAI 开源权重模型 **gpt-oss**。
  * 但 Tatsu 特别强调，**过去两年 MoE 架构创新与规模化落地的真正主阵地在中国开源社区**：以 **DeepSeek (V1 / V2 / V3 / R1)**、阿里 **Qwen-MoE (Qwen 1.5 MoE $\rightarrow$ Qwen 3 / 3.5)**、智谱 **GLM-4.5 / GLM-5** 以及面壁 **MiniCPM-MoE** 为代表的模型，确立了当今工业界 MoE 的黄金标准配方。
* **为什么直到 2024 年后 MoE 才真正普及？（MoE 的四大痛点）**：
  1. 分布式通信与算子基础设施极其复杂（All-to-All 跨卡分发 Token）；
  2. 离散路由不可导，训练目标依赖启发式负载均衡损失；
  3. 路由 Softmax 极易引发数值不稳定与专家坍塌（Expert Collapse）；
  4. 总参数量巨大导致显存占用高，且在小数据集微调（SFT）时极易过拟合。

### 3.2 MoE 设计维度一：路由机制（Routing Function Design Space）

路由器决定了每个 Token 去往哪个专家（注意：**路由粒度是逐个 Token 级别 Token-level**，而非整句或整段级别！且路由器本身只是一个极简单的单层线性分类器，主要根据词法/语言/符号特征而非深层领域语义做快速分流）。

#### 1. 三大路由分配范式对比
1. **Token Chooses Expert（Token 选专家 —— 工业界绝对主流）**：
   每个 Token 计算对所有 $N$ 个专家的偏好得分，独立挑选得分最高的 **Top-$K$ 个专家**。
   * *优点*：每个 Token 都能去往最适合自己的专家；
   * *缺点*：可能导致某些热门专家被选爆（负载不均），需要配合负载均衡机制。
   * [OLMoE (2024)](https://arxiv.org/abs/2409.02060) 消融实验证实，**Token Choice 在验证集 Loss 和下游任务上显著优于 Expert Choice**。
2. **Expert Chooses Token（专家选 Token，[Zhou et al., 2022](https://arxiv.org/abs/2202.09368)）**：
   反过来，让每个专家在当前 Batch 的所有 Token 中挑选对自己得分最高的 Top-$C$ 个 Token。
   * *优点*：天生绝对负载均衡（每个专家处理的 Token 数严格固定为 $C$）；
   * *缺点*：**破坏了自回归因果性（Causality Leak）**（因为当前 Token 能否被选中取决于未来其他 Token 的得分），且某些关键 Token 可能被所有专家漏选（0 个专家处理），在自回归生成中难以直接应用。
3. **全局优化分配（Global Routing via Linear Assignment / RL / Hashing）**：
   * **线性分配（Linear Assignment，如 BASE Layers）**：将路由建模为二分图最优匹配问题，全局求解最优分配，但在线求解计算开销过大。
   * **强化学习（RL / REINFORCE，如 [Bengio et al., 2013](https://arxiv.org/abs/1308.3432)、Clark et al., 2022）**：将离散选专家视为 Bandit 策略用策略梯度更新，但由于梯度方差过大、收敛慢，在对比实验中显著落后于直接反传 + 启发式均衡损失。
   * **哈希路由（Hash Routing）**：用确定性哈希函数直接把 Token ID 映射到固定专家。虽然无需学习就能比稠密模型好，但明显不如可学习的 Top-$K$ 路由。

#### 2. 标准 Token-Choice Top-$K$ 路由的数学细节
给定 Token 隐状态 $u_t \in \mathbb{R}^d$ 与每个专家的路由中心向量（权重列向量）$e_i \in \mathbb{R}^d$（$i = 1, \dots, N$）：
* **主流方案 A：先 Softmax 再 Top-$K$（如 DeepSeek V1/V2、GShard、Switch）**：
  $$s_{i, t} = \text{Softmax}_i(u_t^\top e_i) = \frac{\exp(u_t^\top e_i)}{\sum_{j=1}^N \exp(u_t^\top e_j)}, \qquad g_{i, t} = \begin{cases} s_{i, t}, & \text{if } s_{i, t} \in \text{TopK}(\{s_{j, t}\}_{j=1}^N, K) \\ 0, & \text{otherwise} \end{cases}$$
  （注：选出 Top-$K$ 后，通常还会将选中的 $K$ 个权重重新归一化使其和为 1）。
* **方案 B：先 Top-$K$ 再 Softmax（如 Mixtral 8x7B、QKV 早期变体）**：
  $$g_t = \text{Softmax}\big(\text{TopK}(u_t \cdot E)\big)$$
* **最新演进方案 C：Sigmoid 门控 + 归一化 Top-$K$（[DeepSeek-V3, 2024](https://arxiv.org/abs/2412.19437)）**：
  不再让所有专家在同一个 Softmax 分母下零和竞争，而是对每个专家独立取 Sigmoid 亲和度得分，选出 Top-$K$ 后再做线性归一化：
  $$s_{i, t} = \sigma(u_t^\top e_i), \qquad g'_{i, t} = \begin{cases} s_{i, t}, & \text{if } i \in \text{TopK} \\ 0, & \text{otherwise} \end{cases}, \qquad g_{i, t} = \frac{g'_{i, t}}{\sum_{j=1}^N g'_{j, t}}$$

### 3.3 MoE 设计维度二：专家粒度与共享专家（Fine-Grained & Shared Experts）

这是 **[DeepSeekMoE (Dai et al., 2024)](https://arxiv.org/abs/2401.06066)** 对现代大模型架构做出的最具影响力的贡献之一，引发了从“粗粒度少专家”向“细粒度多专家 + 共享专家”的全面范式转移：

```
[传统粗粒度 MoE (如 Mixtral 8x7B: N=8, K=2)]
输入 u_t ──► Router ──► 激活 2 个巨型专家: [ Expert 1 (1x FFN) ] ... [ Expert 8 (1x FFN) ]

[现代 DeepSeek-style 细粒度 + 共享专家 MoE (如把每个专家切小 4 倍: N=32, K=1+7)]
           ┌──► 旁路直通 (Always-On) ──► [ Shared Expert 1 (0.25x FFN) ] ───────────────────┐
输入 u_t ──┤                                                                                ├──► 相加输出 h_t
           └──► Router (Top-7) ──► [ Routed 1 (0.25x) ] ... [ Routed 31 (0.25x) ] (选 7 个) ┘
```

1. **细粒度专家切分（Fine-Grained Expert Segmentation）**：
   * 传统 MoE（如 GShard、Mixtral）每个 Expert 的大小与标准稠密 FFN 完全一样（$1\times$），总共只有 $N=8$ 或 $16$ 个专家，每次激活 $K=2$ 个。
   * **DeepSeek 的洞见**：在保持总参数量和总激活参数量完全不变的前提下，将每个专家的中间隐藏维度缩小为原来的 **$\frac{1}{m}$**（例如缩至 $1/4$ 或 $1/8$），同时将专家总数扩大为 $m N$ 个，每次激活 $m K$ 个细粒度小专家！
   * **为什么细粒度切分效果好得多？** 从组合数学看，$\binom{8}{2} = 28$ 种专家组合，而切成 4 倍细粒度后 $\binom{32}{8} = \mathbf{10,518,300}$ 种组合！这极大地释放了不同子技能灵活组合的表征能力（[DeepSeekMoE](https://arxiv.org/abs/2401.06066) 与 [OLMoE](https://arxiv.org/abs/2409.02060) 消融实验均证实细粒度切分带来显著收益）。
2. **共享专家隔离（Shared Experts Isolation）**：
   * 许多基础语法、通用常识是所有 Token 都需要执行的公共操作。如果全是路由专家，各个路由专家为了处理这些公共操作会被迫在各自的权重里重复学习相同的冗余知识。
   * 因此，专门划出 $N_s$ 个（通常 $1 \sim 2$ 个）**永远处于激活状态、无需经过路由器的共享专家（Shared Experts）**来承接公共知识，从而解放其余的路由专家（Routed Experts）去专注于高度细分的长尾专门知识！
   * 完整的现代 MoE 前向公式由此变为：
     $$h_t = u_t + \sum_{i=1}^{N_s} \text{FFN}_i^{(s)}(u_t) + \sum_{i=1}^{N_r} g_{i, t} \, \text{FFN}_i^{(r)}(u_t)$$
   * **系统并行优势（课堂问答）**：因为 Shared Experts 对所有 Token 恒定激活，在分布式训练中它们不需要参与跨卡 Token 路由通信，可以直接像普通稠密层一样做张量并行或复制！

#### 主流 MoE 模型超参数演进全览表（2021–2026）

| 模型名称 | 发布年份 | 激活专家数 $K$ (+共享 $N_s$) | 路由专家总数 $N_r$ | 单个专家相对标准 FFN 尺寸 | 架构流派特征 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Switch Transformer** | 2021 | $K = 1$ | $128 \sim 2048$ | $1.0\times$ | Google 早期粗粒度 Top-1 探索 |
| **Mixtral 8x7B** | 2023 | $K = 2$ | $8$ | $1.0\times$ | 西方经典粗粒度 8 选 2 标杆 |
| **Grok-1 / DBRX** | 2024 | $K = 2$ / $K = 4$ | $8$ / $16$ | $1.0\times$ | 粗粒度 MoE |
| **DeepSeekMoE (V1)** | 2024.01 | $K = 6 \quad (+2\text{ Shared})$ | $64$ | $0.25\times$ | **开创细粒度 + 共享专家范式** |
| **Qwen 1.5 / 2 MoE** | 2024 | $K = 4 \quad (+4\text{ Shared})$ | $60$ | $0.125\times$ | 细粒度 + 多共享专家 |
| **DeepSeek-V2** | 2024.05 | $K = 6 \quad (+2\text{ Shared})$ | $160$ | $0.075\times$ | 大规模细粒度 + 设备级受限路由 |
| **OLMoE (1B-7B)** | 2024.09 | $K = 8 \quad (+0\text{ Shared})$ | $64$ | $0.125\times$ | 全开源消融标杆（采用细粒度无共享） |
| **DeepSeek-V3 / R1** | 2024.12 | $K = 8 \quad (+1\text{ Shared})$ | $256$ | $\approx 0.06\times$ | **671B 总参 / 37B 激活，无辅助损失均衡** |
| **Qwen 3.5 / GLM-5** | 2025–2026 | 细粒度 Top-$K$ (+Shared) | $128 \sim 256$ | 细粒度 | 融合 Gated DeltaNet / DSA 的新一代旗舰 MoE |

### 3.4 MoE 设计维度三：训练动力学与负载均衡演化（Load Balancing）

#### 1. 核心训练难题：不可导与“富者愈富”专家坍塌（Expert Collapse）
在训练时，为了省算力我们依然只激活 Top-$K$ 个专家。但 যদি只用标准交叉熵损失直接反向传播（因为只有被选中的 $K$ 个专家有非零权重 $g_{i, t}$，只有它们能收到梯度）：
* **恶性正反馈循环（Rich-Get-Richer）**：一开始偶然表现稍好的某两个专家被选中 $\rightarrow$ 收到梯度更新变得更强、且路由权重 $e_i$ 被进一步推高 $\rightarrow$ 下一步以更高概率被选中。
* **灾难性后果**：不出几百步，**99% 的 Token 都会涌向固定的 1～2 个专家，而其余 95% 的专家彻底饿死（Starvation / Collapse）**，不仅算力并发完全失效（所有 Token 挤爆一张卡），多出来的千亿参数也沦为摆设！

#### 2. 早期随机噪声探索（Stochastic Perturbations —— 现已逐渐弃用）
* **[Shazeer et al. (2017)](https://arxiv.org/abs/1701.06538) 高斯噪声门控**：在路由内积上加受控高斯噪声 $H(x)_i = (x W_g)_i + \epsilon \cdot \text{Softplus}((x W_{\text{noise}})_i)$；
* **[Switch Transformer (2022)](https://arxiv.org/abs/2101.03961) 乘性均匀抖动（Jitter）**：将路由输入乘上 $[1-\epsilon, 1+\epsilon]$（$\epsilon=10^{-2}$）的均匀随机数。
* *现状*：后期的消融研究（如 OLMoE）表明，只要负载均衡机制得当，**去掉这些人为随机噪声反而能让训练更稳定、最终模型质量更高**。

#### 3. 经典启发式负载均衡辅助损失（Switch Auxiliary Load-Balancing Loss）
为了对抗专家坍塌，[Switch Transformer](https://arxiv.org/abs/2101.03961) 提出了统治工业界多年的**专家级负载均衡损失（Per-Expert Auxiliary Loss）**。给定一个包含 $T$ 个 Token 的训练批次：
$$\mathcal{L}_{\text{aux}} = \alpha N \sum_{i=1}^N f_i \cdot P_i$$
其中：
* $f_i = \frac{1}{T} \sum_{t=1}^T \mathbb{1}\{\text{Token } t \text{ 选中了专家 } i\}$：是分配给专家 $i$ 的**实际离散 Token 比例（Fraction of tokens dispatched，不可导）**；
* $P_i = \frac{1}{T} \sum_{t=1}^T s_{i, t}$：是路由器分给专家 $i$ 的**平均连续概率质量（Average router probability，可导）**；
* $\alpha$ 是均衡系数超参数（通常取 $10^{-2}$）。

> **Tatsu 的精妙梯度视角解读：为什么把不可导的 $f_i$ 和可导的 $P_i$ 乘起来就能实现均衡？**
> 不要只盯着原函数看，**请对路由概率 $P_i$ 求偏导**（把不可导的计数比例 $f_i$ 视为停止梯度的常数 `stop_gradient(f_i)`）：
> $$\frac{\partial \mathcal{L}_{\text{aux}}}{\partial P_i} = \alpha N f_i$$
> 这意味着：**某个专家 $i$ 当前抢到的实际 Token 比例 $f_i$ 越大，辅助损失施加在它路由概率 $P_i$ 上的正向惩罚梯度就越强（梯度下降时会以正比于 $f_i$ 的力度把 $P_i$ 往下压）**！当且仅当所有专家绝对均匀分配（$\forall i, f_i = \frac{1}{N}, P_i = \frac{1}{N}$）时，该辅助损失取得极小值。

#### 4. 系统感知的多级均衡（DeepSeek V1 / V2：Expert + Device + Communication Balancing）
在跨多节点专家并行（Expert Parallelism）中，哪怕单看每个专家有点不均匀，更致命的是**某台机器（Device）上的几个专家恰好都很热门**，导致该 GPU 成为全集群的通信与计算木桶短板！因此 [DeepSeek-V2](https://arxiv.org/abs/2405.04434) 引入了：
* **设备级受限路由（Device-Limited Routing）**：限制每个 Token 的 Top-$K$ 专家最多只能分布在 $M$ 台不同的设备上（如 $M=3$），直接锁死跨节点通信扇出上限；
* **三重均衡辅助损失**：同时施加 **专家级均衡损失（$\mathcal{L}_{\text{ExpBal}}$）** + **设备级均衡损失（$\mathcal{L}_{\text{DevBal}}$，按设备聚合 $f_i$ 和 $P_i$）** + **通信均衡损失（$\mathcal{L}_{\text{CommBal}}$）**。

#### 5. 最新突破：DeepSeek-V3 的“无辅助损失”动态偏置负载均衡（Auxiliary-Loss-Free Balancing）
传统辅助损失 $\mathcal{L}_{\text{aux}}$ 存在一个本质矛盾：如果权重 $\alpha$ 太小，专家依然会不均衡；如果 $\alpha$ 太大，**辅助损失的梯度会直接干扰主语言模型交叉熵损失的梯度方向**，损害模型的语义表征质量！

为此，[Wang et al. (2024) / DeepSeek-V3](https://arxiv.org/abs/2408.15664) 提出了一种极其巧妙的**动态偏置调节法（Auxiliary-Loss-Free Load Balancing）**：
* 为每个专家 $i$ 维护一个**不参与梯度反向传播的动态偏置项 $b_i$**：
  1. **在决定 Top-$K$ 选谁时（Routing Selection）**：使用加上偏置后的得分 $s'_{i, t} = s_{i, t} + b_i$ 来挑选 Top-$K$ 专家；
  2. **在计算加权输出乘上门控权重时（Gate Weighting）**：依然使用**未加偏置的原始亲和度得分 $s_{i, t}$**！
  3. **在线更新偏置 $b_i$（Dual Ascent 思想）**：每跑完一个训练步，统计每个专家的实际负载：如果专家 $i$ 过载（抢的 Token 多于平均值），就将 $b_i \leftarrow b_i - \gamma$ 调低一点；如果专家 $i$ 欠载（饥饿），就将 $b_i \leftarrow b_i + \gamma$ 调高一点！
* **为什么这招堪称神来之笔？** 它通过直接平移选人门槛 $b_i$ 实现了完美的物理负载均衡，但**因为 $b_i$ 完全不产生任何反向传播梯度，主干网络和路由器的权重更新 100% 服务于降低语言建模 Loss，完全不受辅助损失梯度的污染**！（注：如课堂所指出，DeepSeek-V3 实际上仍保留了一个极微弱的系数 $\alpha=10^{-4}$ 的单序列级补充均衡损失以防单个极端句子的退化）。

### 3.5 MoE 的系统工程、服务随机性、稳定性与微调策略

#### 1. 分布式专家并行通信优化与块稀疏算子（MegaBlocks & Nemotron 3）
* **单卡内的块稀疏矩阵乘法（Block-Sparse MatMul / [MegaBlocks](https://arxiv.org/abs/2211.15841)）**：
  如果单张 GPU 上放了多个专家，逐个专家跑小矩阵乘法会导致算术强度太低。现代框架将所有专家的权重拼成块对角矩阵（Block-Diagonal Matrix），利用硬件原生的结构化块稀疏算子一次性完成大矩阵乘法。
* **跨卡激活降维通信（Latent MoE / [Nemotron 3](https://arxiv.org/abs/2603.00001)）**：
  在 Expert Parallelism 中，把高维残差激活向量 $u_t \in \mathbb{R}^d$ 跨节点通过网络发给其他 GPU 上的专家非常耗时。Nemotron 3 提出：**在跨卡分发（Dispatch）之前，先用一个线性层把激活向量降维投影到低维空间**，把轻量的小向量发给远端专家算完后，再传回来升维！这大幅砍掉了跨节点 `All-to-All` 通信字节数（而留在本地的 Shared Expert 则直接处理原始高维向量）。

#### 2. 早期 MoE 在线推理的“诡异随机性”与无丢弃（Dropless）演进
* **Capacity Factor 与 Token Dropping**：早期 MoE 框架为了静态分配显存张量形状，给每个专家设定了固定的排队容量上限（Capacity Factor）。在推理服务时，因为不同用户的请求被拼在同一个 Batch 里，**如果同批次另一个用户的 Prompt 恰好全是代码、把“代码专家”的队列塞满了，你的代码 Token 就会被直接丢弃（Token Dropped，输出直接置零跳过残差）**！这就导致了早期的 GPT-4 等 MoE API 出现“同一个请求、温度设为 0，每次返回结果却不一样且质量随服务器负载波动”的诡异现象。
* **现代解决方案**：随着 **[MegaBlocks (Gale et al., 2022)](https://arxiv.org/abs/2211.15841)** 等动态变长块稀疏算子的普及，现代 MoE 训练与推理已全面转向 **Dropless MoE（绝不丢弃任何 Token）**。

#### 3. 路由数值稳定性（Router FP32 & Router z-loss）
因为路由器包含离散的 `TopK` 截断和指数级的 `Softmax`，半精度下的微小舍入误差都可能导致选出的专家发生跳变。标准稳定方案（源自 [ST-MoE, Zoph et al., 2022](https://arxiv.org/abs/2202.08906)）：
1. **路由器强制使用 `fp32`**：即便全网都是 `bf16` 或 `fp8`，唯独 Router 的内积与 Softmax 强制转型为 `float32` 计算；
2. **Router `z-loss`**：与 Lecture 3 的输出层 z-loss 完全同理，对路由器 Logits $x_{i, j}$ 的配分函数施加平方惩罚以防止路由过于自信引发梯度爆炸：
   $$\mathcal{L}_{z}^{\text{router}} = \frac{1}{B} \sum_{t=1}^B \left( \log \sum_{j=1}^N \exp(x_{t, j}) \right)^2$$

#### 4. MoE 的微调过拟合难题与稀疏升维（Sparse Upcycling）
* **微调（SFT）过拟合**：由于 MoE 总参数量极大而每次只有少数专家被激活，在小规模下游数据集上微调时，验证集 Loss 极易严重过拟合（Train-Val Gap 巨大）。
  * **解法一（[Zoph et al., 2022](https://arxiv.org/abs/2202.08906)）**：微调时冻结所有稀疏 MoE 专家层，**只微调非 MoE 部分（即 Attention 层与 Shared Experts）**；
  * **解法二（DeepSeek 苦涩教训暴力流）**：直接构建海量的高质量 SFT 数据集（如 DeepSeek 使用 **140 万条多样化指令数据**），用数据量喂饱所有专家。
* **稀疏升维（Sparse Upcycling，[Komatsuzaki et al., 2022](https://arxiv.org/abs/2212.05055)）**：
  如果你已经花巨资训练好了一个稠密模型（如 MiniCPM 2.4B 或 Qwen 1.8B），可以通过将它的 FFN 权重复制 $N$ 份（或切分重组）作为 MoE 专家的初始权重，随机初始化一个 Router，再继续预训练（如构建出 MiniCPM-MoE 8x2B 或 Qwen1.5-MoE-A2.7B），比从零训练省下大量早期算力。（注：到了 2025–2026 年，由于各厂直接从零训练旗舰原生 MoE 的配方已极其成熟，Upcycling 已较少被使用）。

## 4. 集大成案例研究：DeepSeek (V1 $\rightarrow$ V2 $\rightarrow$ V3) 架构全解

Tatsu 在课程最后将 **DeepSeek 系列** 作为现代 LLM 架构与系统协同设计（Hardware-Architecture Co-design）的巅峰典范进行了完整复盘：

### 4.1 三代 MoE 参数与路由演进表

| 维度 | [DeepSeekMoE (V1)](https://arxiv.org/abs/2401.06066) | [DeepSeek-V2](https://arxiv.org/abs/2405.04434) | [DeepSeek-V3](https://arxiv.org/abs/2412.19437) |
| :--- | :--- | :--- | :--- |
| **总参数量 / 激活参数量** | **16B 总参 / 2.8B 激活** | **236B 总参 / 21B 激活** | **671B 总参 / 37B 激活** |
| **专家配置 ($N_s$ 共享 + $N_r$ 路由)** | 2 共享 + 64 细粒度路由 (激活 6) | 2 共享 + 160 细粒度路由 (激活 6) | 1 共享 + 256 细粒度路由 (激活 8) |
| **路由打分函数** | 标准 Softmax Top-$K$ | Softmax + Top-$M$ ($M=3$) 设备受限路由 | **Sigmoid 亲和度 + 归一化 Top-$K$** + Top-$M$ ($M=4$) 设备受限 |
| **负载均衡策略** | 专家级 + 设备级辅助损失 | 专家级 + 设备级 + 通信级三重辅助损失 | **无辅助损失动态偏置 ($b_i$) 调整** + 极弱序列级辅助损失 |

### 4.2 多头潜在注意力（MLA, Multi-Head Latent Attention）与解耦 RoPE

除了 MoE，DeepSeek-V2/V3 在注意力机制上发明了比 GQA 更激进的 KV Cache 压缩技术——**MLA (Multi-Head Latent Attention)**：

```
[标准 MHA / GQA]
输入 h_t ──► 投影生成完整的 K_t, V_t ──► 必须将庞大的 K_t, V_t 全部存入显存 KV Cache!

[DeepSeek MLA (Multi-Head Latent Attention)]
           ┌──► 低秩下投影 W^{DKV} ──► 潜在压缩向量 c_t^{KV} (极小, 仅 512 维!) ──► [存入 KV Cache!] ──► 推理时上投影 W^{UK}, W^{UV} (可被矩阵吸收!)
输入 h_t ──┤
           └──► 独立小分支 W^{KR} ───► 旋转位置编码 RoPE(k_t^R) (仅 64 维!) ────────► [存入 KV Cache!]
```

1. **核心思想（低秩联合压缩 Low-Rank KV Joint Compression）**：
   不再在 KV Cache 中存储每个头的完整 Key 和 Value，而是先用下投影矩阵 $W^{DKV} \in \mathbb{R}^{d \times d_c}$（其中潜在维度 $d_c = 512 \ll H d_{\text{head}}$）将隐状态 $h_t$ 压缩为一个极低维的**潜在向量（Latent Vector）$c_t^{KV} \in \mathbb{R}^{d_c}$**：
   $$c_t^{KV} = h_t W^{DKV}, \qquad K_t^C = c_t^{KV} W^{UK}, \qquad V_t^C = c_t^{KV} W^{UV}$$
   * **惊人的显存与算力魔术（Matrix Absorption）**：在推理时，**KV Cache 只需要存储这个 512 维的压缩向量 $c_t^{KV}$**（比 GQA 还要省显存）！更绝的是，在推理计算 $q_t^\top k_j$ 时，上投影矩阵 $W^{UK}$ 可以利用矩阵结合律**直接吸收合并到 Query 的投影矩阵 $W^Q$ 中**，连把 $c_t^{KV}$ 解压回完整 $K, V$ 的计算和显存开销都彻底省掉了！（Query 端同样做了低秩压缩 $c_t^Q$ 以减少训练激活显存）。
2. **为什么 MLA 会与标准 RoPE 发生冲突？如何用 Decoupled RoPE 破解？**
   * **冲突根源**：如果我们对解压后的 $K_t^C = c_t^{KV} W^{UK}$ 施加位置相关的旋转矩阵 $R_j$（即 $R_j W^{UK}$），那么在计算 $q_t^\top R_t^\top R_j W^{UK} c_j^{KV}$ 时，中间夹着一个**随历史位置差 $(t - j)$ 动态变化的旋转矩阵 $R_{j-t}$**！这样一来，固定的上投影矩阵 $W^{UK}$ 就无法被提前吸收合并到 $W^Q$ 里了，推理时就必须把所有历史位置的 $c_j^{KV}$ 重新解压成庞大的 $K_j$，MLA 的压缩红利瞬间归零！
   * **解耦旋转位置编码（Decoupled RoPE）**：DeepSeek 的解法极其干脆——**将主体低秩压缩内容 $c_t^{KV}$ 完全不加 RoPE（以便完美执行矩阵吸收），另外单独分出一个极小维度的共享键向量 $k_t^R \in \mathbb{R}^{d_R}$（仅 $d_R = 64$ 维）专门用来挂载 RoPE**！计算注意力内积时，将不带位置的压缩内容内积与带位置的 64 维 RoPE 内积相加即可。

### 4.3 多 Token 预测（MTP, Multi-Token Prediction）

DeepSeek-V3 在训练目标上还引入了 **MTP（Multi-Token Prediction，源自 [Gloeckle et al., 2024](https://arxiv.org/abs/2404.19737)，但改为顺序因果链式结构）**：在每个位置不仅预测下一个 Token $x_{t+1}$，还通过轻量的附加预测头顺序预测 $x_{t+2}$。
* **双重收益**：
  1. **训练阶段**：迫使模型在隐状态中提前规划未来多步的语义结构，提高了数据样本利用效率；
  2. **推理阶段**：这个训练好的 MTP 附加预测头在部署时可以直接作为**内置的无成本 Draft Model**，配合**投机采样（Speculative Decoding）**实现每次前向传播生成并验证多个 Token，使推理解码速度直接提升约 1.8 倍！

## 5. 课后思考与自测练习（Review & Self-Check）

1. **线性注意力与 Gated DeltaNet 维度与复杂度精算**：
   假设序列长度 $n = 131,072$（128K），单头维度 $d_k = d_v = 128$。
   * (a) 在单头自注意力中，标准 Softmax Attention 的 $Q K^\top V$ 需要多少次乘加运算（以数量级 $n^2 d_k$ 计）？而利用结合律改写为 $Q(K^\top V)$ 后需要多少次？两者的计算量相差多少倍？
   * (b) 在 Gated DeltaNet 的状态更新式 $S_t = \gamma_t (I - \beta_t k_t k_t^\top) S_{t-1} + \beta_t k_t v_t^\top$ 中，如果直接把 $d_k \times d_k$ 的矩阵 $(I - \beta_t k_t k_t^\top)$ 实例化出来再乘以 $S_{t-1} \in \mathbb{R}^{d_k \times d_v}$，单步复杂度是 $O(d_k^2 d_v)$。请利用矩阵乘法结合律将其改写为仅需 **$O(d_k d_v)$** 的两步向量-矩阵运算！
   * *参考解答*：
     * (a) 标准注意力正比于 $n^2 (d_k + d_v) = (131072)^2 \times 256 \approx 4.4 \times 10^{12}$；线性注意力正比于 $2 n d_k d_v = 2 \times 131072 \times 128 \times 128 \approx 4.3 \times 10^9$。**在 128K 长度下，线性注意力的计算量比标准注意力小整整 $1024$ 倍（即 $\frac{n}{d_k}$ 倍）**！
     * (b) 展开括号并先算向量与矩阵的内积 $u_t^\top = k_t^\top S_{t-1} \in \mathbb{R}^{1 \times d_v}$（耗时 $O(d_k d_v)$），再做外积更新：
       $$S_t = \gamma_t S_{t-1} - \gamma_t \beta_t k_t \big(k_t^\top S_{t-1}\big) + \beta_t k_t v_t^\top = \gamma_t S_{t-1} + \beta_t k_t \big(v_t^\top - \gamma_t k_t^\top S_{t-1}\big)$$
       全程只需一次矩阵-向量乘法和一次向量外积，单步严格保持 $O(d_k d_v)$！
2. **MoE 激活参数量与显存容量解耦计算**：
   考虑一个类似 DeepSeek-V3 的单层 MoE FFN：隐藏层维度 $d_{\text{model}} = 7168$，包含 $N_s = 1$ 个共享专家和 $N_r = 256$ 个路由专家，每个专家都是采用 SwiGLU 结构（含 $W_1, V, W_2$ 三个矩阵）且单个专家的中间维度为 $d_{\text{expert}} = 2048$。每个 Token 激活 $K = 8$ 个路由专家。
   * 请计算这一层 MoE FFN 的**总参数量（Total Parameters）**与每个 Token 实际参与计算的**激活参数量（Active Parameters）**（忽略极小的 Router 权重）。
   * *参考解答*：
     * 单个 SwiGLU 专家的参数量为 $3 \times d_{\text{model}} \times d_{\text{expert}} = 3 \times 7168 \times 2048 = \mathbf{44,040,192}$ ($\approx 44.04\text{M}$)。
     * 该层共有 $1 + 256 = 257$ 个专家，**总参数量**为 $257 \times 44.04\text{M} \approx \mathbf{11.32\text{B}}$（仅一层 FFN 就有 113 亿参数！）。
     * 每个 Token 仅激活 $1\text{ (共享)} + 8\text{ (路由)} = 9$ 个专家，**激活参数量**仅为 $9 \times 44.04\text{M} \approx \mathbf{396.36\text{M}}$（不到总参数的 $3.5\%$！）。
3. **思考题：为什么 MLA 不能直接对压缩向量 $c_t^{KV}$ 加 RoPE？**
   回顾 4.2 节，体会矩阵乘法结合律在推理加速中的精妙应用：在自回归 Decode 阶段，如果 $K_j = c_j^{KV} W^{UK}$ 中间没有位置相关的旋转矩阵，我们可以预先把 $q_t W^{UK^\top}$ 算好（把 $W^{UK}$ 吸收到 Query 侧），从而避免把缓存里成千上万个历史位置的 $c_j^{KV}$ 乘以 $W^{UK}$ 还原成巨型张量。这与线性注意力移动括号的思想如出一辙！
