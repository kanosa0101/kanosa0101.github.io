---
title: Stanford CS336 (Spring 2026) Lecture 9 深度课程笔记：Scaling Laws (Part 1)
date: 2026-10-26 08:50:00
tags:
  - 大模型
  - Scaling Laws
  - 缩放定律
  - Chinchilla
  - Kaplan
  - 算力优化
  - CS336
categories:
  - [笔记, CS336]
description: 深入剖析 Stanford CS336 Lecture 9，系统推导大语言模型缩放定律（Scaling Laws）：从 Kaplan 幂律到 Chinchilla 最优算力分配、6ND 浮点运算黄金近似严格推导、超额训练（Over-training）权衡与数据吞吐边界。
mathjax: true
---

# Stanford CS336 (Spring 2026) Lecture 9 深度课程笔记：Scaling Laws (Part 1)

> **课程信息**：[Stanford CS336: Language Modeling from Scratch (Spring 2026)](https://cs336.stanford.edu/)
> **主讲教授**：Tatsunori Hashimoto (Tatsu)
> **本讲资源**：[Lecture 9 YouTube 视频](https://www.youtube.com/watch?v=Q15rhEWZPQ4) | [官方幻灯片 (`lecture_09.pdf`)](https://github.com/stanford-cs336/lectures/blob/main/lecture_09.pdf) | [Assignment 3 指导手册与代码库](https://www.google.com/search?q=https://github.com/stanford-cs336/assignment3-scaling)

## 1. 引言：为什么大模型工程需要扩展定律（Scaling Laws）？

在经历了前四讲的**模型架构（Assignment 1）**与第 5～8 讲的**底层硬件与分布式系统（Assignment 2）**之后，从 Lecture 9 开始，课程正式进入第三大核心支柱——**Scaling Laws（扩展定律，对应 Assignment 3）**。由于排课原因，我们的节奏将是：本讲（Lecture 9）拆解 **Scaling Laws 基础理论与经典范式** $\rightarrow$ 下一讲（Lecture 10）由 Percy 讲解 **Inference（高效推理系统）** $\rightarrow$ 第 11 讲（Lecture 11）再回到 Tatsu 的 **Scaling Laws 进阶专题（$\mu$P 最大更新参数化、批量与学习率调度缩放及最新前沿模型技术报告拆解）**。

### 1.1 “给你 10,000 张 B200 训练一个月”的工程决策困境

Tatsu 用一个极具画面感的真实场景开启了本讲：
假设你有一位非常富有的朋友，刚把 **10,000 张 NVIDIA B200 GPU 的一个月使用权**交给你，要求你训练出一个超越当前 SOTA 的顶级开源大语言模型。此时，你已经组建好了基础设施团队（完成了 Assignment 2 的算子与 4D 并行优化），也准备好了一份高质量预训练语料库（Assignment 4 的内容）。但当你准备按下训练启动键时，你立刻面临一系列令人窒息的工程抉择：
* **数据层面**：新闻、代码、网页、数学语料的最佳混合比例（Data Mixture）是多少？高质数据应该重复训练几个 Epoch？过滤阈值该设多严？
* **架构与模型尺寸层面**：在给定的总浮点预算（FLOPs）下，究竟应该训练一个**参数量更大但训练步数更少的模型**，还是训练一个**参数量更小但吃更多 Token 的模型**？深宽比（Aspect Ratio）和 MoE 稀疏度设多少最优？
* **优化器与超参层面**：全局 Batch Size 最大能推到多少而不损失样本效率？随着模型放大 100 倍，峰值学习率（Learning Rate）该怎么调？

```
[传统粗暴范式 (Old Paradigm): 在大模型上直接网格调参]
  10,000x B200 巨型集群 ──► 跑大模型配置 A (烧掉 $5M, 失败/次优)
                        ──► 跑大模型配置 B (烧掉 $5M, 梯度爆炸)
                        ──► 跑大模型配置 C (烧掉 $5M, 勉强收敛)
  ❌ 痛点：单次实验耗资数百万美元，根本不可能在目标规模上做消融实验（Ablation）或超参搜索！

[现代扩展定律范式 (Scaling Law Paradigm): 小规模探索 + 幂律外推]
  1% 算力预算跑一簇小规模实验 ──► 拟合简单可预测的幂律曲线 ──► 跨数量级外推 (Extrapolation)
  (10M ~ 1B 规模, 多组算力点)     L(x) = L_∞ + A · x^{-α}      精准锁定 10,000x B200 的最优配置与预期 Loss!
```

* **扩展定律（Scaling Laws）的核心哲学**：
  在大模型研发实验室中，Scaling Laws 几乎不仅是一种工具，更是一种“信仰（Way of life）”。它的核心目标是将**“在千万美元级大训练上的盲目赌博”**转化为**“在廉价小规模实验上的科学测量与外推预测”**——只要小规模与大规模之间存在简单、鲁棒的统计规律（Regularities），我们就能在启动主训练（Main Run）之前，精确预测出最终模型的验证集 Loss 甚至不同架构/优化器之间的性能差距。

### 1.2 统计学习理论起源与三十年历史演进

很多初学者误以为 Scaling Laws 是 2020 年 OpenAI 提出 GPT-3 时才突然诞生的“深度学习玄学”。事实上，作为统计学习理论出身的研究者，Tatsu 强调：**Scaling Laws 的本质就是经验样本复杂度（Empirical Sample Complexity），它的历史可以追溯到三十多年前的经典统计学习理论与早期 NLP 实证研究**：

1. **理论之源：统计学习理论中的泛化界（Generalization Bounds）**
   在统计机器学习（ML 101）中，当我们评估一个假设类 $\mathcal{H}$ 的性能时，经典泛化理论给出了经验风险最小化器 $\hat{h}_n$（在 $n$ 个样本上训练）与最优假设 $h^*$ 之间的超额风险上界（Excess Risk Bound）：
   $$\mathbb{E}\big[R(\hat{h}_n)\big] - R(h^*) \le C \cdot \sqrt{\frac{\text{Complexity}(\mathcal{H})}{n}} \quad \text{或更一般地} \quad \mathcal{O}\big(n^{-\alpha}\big)$$
   * **与 Scaling Laws 的深刻联系**：注意右边是样本量 $n$ 的幂函数 $n^{-\alpha}$，而左边的 $R(h^*)$ 就是不可约误差下界 $L_\infty$！理论学家关注的是**最坏情况下的渐近上界（Worst-case Upper Bound）**，而现代 Scaling Laws 则是把不等号变成近似等号，直接在真实数据和具体训练配方上测量**经验收敛速率（Empirical Convergence Rate）**。

2. **第一篇真正的 Scaling Law 论文：[Cortes, Jackel, Solla, Vapnik \& Denker (1993)](https://www.google.com/search?q=https://papers.nips.cc/paper/1993/hash/1aa48fc4880bb0c9b8a3bf979d3b917e-Abstract.html)**
   早在 **1993 年**，贝尔实验室（Bell Labs）的 Corinna Cortes 与 Vladimir Vapnik 等人就发表了题为 *《Learning Curves: Asymptotic Values and Rate of Convergence》* 的开创性论文。他们提出的问题与今天一模一样：在大数据集上训练分类器太贵了，能否**只把分类器训练在几个小样本子集上，通过拟合测试误差随样本量 $l$ 衰减的幂律曲线 $E_{\text{test}}(l) = a + b \cdot l^{-\alpha}$，直接外推预测大数据集上的最终性能**？这在字面意义上就是现代数据扩展定律（Data Scaling Law）的鼻祖。

3. **NLP 领域的早期觉醒：[Banko & Brill (2001)](https://aclanthology.org/P01-1005/) 与 [Kolachina et al. (2012)](https://aclanthology.org/P12-1003/)**
   * **[Banko & Brill (2001)](https://aclanthology.org/P01-1005/)（*Scaling to Very Very Large Corpora for Natural Language Disambiguation*）**：在经典 NLP 时代展示了惊人的一幕——无论是感知机、朴素贝叶斯还是基于记忆的学习器，当训练语料从百万词扩展到十亿词（横轴取对数）时，所有算法的测试准确率都在沿着近乎直线的轨迹稳定上升。他们由此发出灵魂拷问：**与其花大价钱微调算法，直接去收集更多数据是不是划算得多？**
   * **[Kolachina et al. (2012)](https://aclanthology.org/P12-1003/)（*Prediction of Learning Curves in Machine Translation*）**：在统计机器翻译（SMT）中系统比较了多种预测 BLEU 分数随语料增长的函数形式，最终发现表现最好的正是三参数与四参数幂律族 **`Pow3`（$y = c - a x^{-\alpha}$）** 与 **`Pow4`（$y = c - (a x + b)^{-\alpha}$）**——这与十年后我们在大模型 Scaling Laws 中反复使用的核心方程完全一致！

4. **现代神经扩展定律的奠基杰作：[Hestness et al. (2017, Baidu Research)](https://arxiv.org/abs/1712.00409)**
   Tatsu 在每年讲授本课中都会专门致敬百度研究院 2017 年发表的 *《Deep Learning Scaling is Predictable, Empirically》*，认为这篇极具远见的论文没有得到与其贡献匹配的引用量。早在 GPT-2/GPT-3 诞生前三年，[Hestness et al. (2017)](https://arxiv.org/abs/1712.00409) 就横跨**机器翻译、语言建模、语音识别、图像分类**四大领域，提前预言了今天大模型时代的几乎所有核心法则：
   * 跨领域的测试损失随数据量呈高度可预测的**幂律衰减（Power-law decay）**；
   * 首次提出由于准确率（Accuracy）是比交叉熵损失（Loss）更不连续的度量，平滑的 Loss 下降会导致下游能力出现看似突变的**“涌现现象（Emergence）”**；
   * 明确指出既然深度学习能可预测地随数据扩展，那么**算力（Compute）将成为第一性瓶颈**，而**系统工程加速（Speed & Systems Optimization）将直接转化为模型精度（Accuracy）**！

### 1.3 什么变量服从扩展定律？课堂问答与全景分类

> **课堂问答：这些幂律公式（Power Laws）究竟是纯粹的经验曲线拟合，还是背后有第一性原理支撑？**
> * **Tatsu 的解答**：在工程实操中，它们首先是**经验曲线拟合（Empirical Curve Fitting）**，并没有哪条上帝法则规定神经网络必须严格服从某一个初等函数。但为什么大家都在用包含渐近下界的多项式幂律（$L(x) = L_\infty + A x^{-\alpha}$）？一是因为**统计学习理论中的样本复杂度与泛化界天然给出了多项式衰减的候选函数族**；二是因为**统计物理学家习惯于分析系统在极限状态（$x \to 0$ 与 $x \to \infty$）下的无标度（Scale-free）渐近行为**，而唯一满足尺度变换不变性（$f(cx) \propto f(x)$）的函数正是幂函数！

在现代大模型研发中，表现出惊人规律性的扩展关系可以分为两大类：
1. **预训练核心资源扩展定律（Pre-training Loss Scaling Laws，本讲核心）**：
   以交叉熵测试损失（Test Cross-Entropy Loss，单位为 nats/token 或 bits/byte）为纵轴 $y$，当横轴 $x$ 分别取**总计算量（Compute $C$）**、**数据集大小（Dataset Size $D$）**或**模型参数量（Parameters $N$）**，且双方均取对数（Log-Log Plot）时，在跨越 6～8 个数量级的范围内呈现出极其干净的**对数线性关系（Log-Linear / Power-Law Relationship）**。
2. **广义与外延扩展现象（Exotic & Downstream Scaling Phenomena）**：
   * **下游任务评测（Downstream Benchmarks）**：当纵轴换成 MMLU、GSM8K 等下游准确率时，由于准确率受限于随机猜测下界与 $100\%$ 上界，对数算力与准确率之间通常呈现 **Sigmoid（逻辑斯蒂曲线）** 形态；
   * **时间线能力预测（Forecasting over Time）**：近期许多 AI 趋势预测研究甚至将横轴设为“模型发布日期（Date）”，发现前沿模型在各类基准（如 Epoch AI 综合能力指数、长程 Agent 持续时间）上的**上包络线（Upper Envelope）**随时间呈对数线性或指数增长。

### 1.4 本讲三部曲路线图

1. **Part I：数据扩展定律（Data Scaling Laws）**——从高斯均值估计与非参数回归推导出幂律形式，解开“为什么神经网络的数据扩展指数只有 $-0.1$”的理论谜题，并探讨数据配比（Mixtures）、多轮重复（Repetition）与动态过滤（Filtering）。
2. **Part II：模型架构与超参数扩展定律（Model & Hyperparameter Scaling Laws）**——如何用小规模斜率与截距对比淘汰劣质架构（LSTM/Performer vs. Transformer/GLU/MoE）、确定最优深宽比、推导**临界批大小（Critical Batch Size）**以及理解学习率随规模移动的规律。
3. **Part III：算力最优联合扩展定律（Compute-Optimal Scaling）**——完整复盘 **[Kaplan et al. (2020)](https://arxiv.org/abs/2001.08361) 与 [Chinchilla (Hoffmann et al., 2022)](https://arxiv.org/abs/2203.15556) 的世纪分歧**，精讲 Chinchilla 三大拟合方法（特别是 **IsoFLOP 法**），拆解导致两者差异的三大工程细节，并延伸至现代生产环境中的**推理感知过训练（Inference-Aware Overtraining）**。

## 2. Part I：数据扩展定律（Data Scaling Laws）——从高斯均值估计到神经非参数回归

### 2.1 数据扩展的三阶段S型曲线（Three Regimes of Data Scaling）

我们先固定模型架构与训练流程，且**假设模型容量远大于数据集规模（$N \gg D$，使模型不会成为瓶颈）**，仅单变量增加训练数据量 $n$（或记为 $D$），观察测试损失 $L(n)$ 的变化。如 [Hestness et al. (2017)](https://arxiv.org/abs/1712.00409) 所总结，在双对数坐标系（横轴 $\log n$，纵轴 $\log L(n)$）下，完整的数据学习曲线必定经历以下**三个物理阶段**：

```
  log L(n) (测试交叉熵损失)
    ▲
    │  [阶段 1: 随机猜测区]
L_0 │──────┐ (Small Data / Random Guessing: 样本极少，模型甚至无法学出基本统计特征)
    │       \
    │        \   [阶段 2: 幂律扩展区 (Power-Law Region)]
    │         \    斜率 = -α (对数坐标下呈笔直的线性下降: L(n) - L_∞ ≈ A · n^{-α})
    │          \
    │           \      [阶段 3: 不可约误差渐近区 (Irreducible Error Floor)]
L_∞ │............\───────────────────────► (逼近自然语言的固有香农熵 H(p_data) 或小模型容量上限)
    └────────────────────────────────────► log n (训练样本数 / Tokens 数)
```

1. **阶段 1：随机猜测平台期（Random Guessing Regime）**：当样本量 $n$ 极小时，模型完全处于欠定状态，损失维持在随机猜测水平 $L_0$（对于词表大小 $|V|$ 的分类或生成，约为 $\log |V|$ 或 unigram 熵）。
2. **阶段 2：幂律衰减区（Power-Law Region）**：随着数据量跨数量级增长，超额损失 $L(n) - L_\infty$ 严格按照幂函数 $A \cdot n^{-\alpha}$ 衰减。在双对数图上，如果此时距离渐近下界 $L_\infty$ 还很远（即 $L(n) \gg L_\infty$），$\log L(n)$ 与 $\log n$ 直接呈现出一条**斜率为 $-\alpha$ 的直线**（例如 [Kaplan et al. (2020)](https://arxiv.org/abs/2001.08361) 中展示的跨度从 $2\times 10^7$ 到 $2\times 10^{10}$ Tokens 的完美直线）。
3. **阶段 3：不可约误差渐近区（Irreducible Error / Noise Floor, $L_\infty$）**：即便给你无穷多的数据（$n \to \infty$）和无穷大的模型，测试损失也绝不可能降到 $0$！因为自然语言本身具有内在的不确定性（下一个词的概率分布并非确定性的 Dirac 分布），损失最终会渐近收敛到真实数据分布的**固有熵（Intrinsic Entropy / Bayes Error Rate）$L_\infty = H(p_{\text{data}})$**。

> **课堂问答：Tatsu 提到在测试纯数据扩展定律时要求“模型远大于数据集（Model is larger than dataset size）”，这是指参数量 $N$ 和 Token 数 $D$ 满足 $1:1$ 还是什么具体比例？**
> * **Tatsu 的解答**：如果模型相对于数据集太小（即 $D \gg N$），模型很快就会耗尽其表达容量（Capacity Bottleneck），提前撞上一堵属于该小模型假设类的“模型容量渐近墙”，导致曲线过早走平，你就测不出纯粹的数据扩展幂律了。因此，为了停留在干净的**阶段 2（纯数据幂律区）**，测试数据扩展定律时通常让模型规模显著大于常规配比（例如参数量相比数据量留出 $10\times$ 以上的充裕容量，或者显式拟合渐近项 $L_\infty(N)$ 并将其扣除）。

### 2.2 为什么幂律如此自然？——从经典参数统计到非参数估计的数学推导

为什么在对数坐标下画条直线就等价于多项式/幂律衰减？为什么神经网络的数据扩展指数（$\alpha \approx 0.1$）比统计学教科书里的 $\alpha = 1$ 慢这么多？Tatsu 用三张纯统计学幻灯片给出了极其精彩的第一性原理推导。

#### 1. 双对数线性（Log-Linearity）与幂律（Power Law）的数学等价性
假设在扣除渐近不可约误差 $L_\infty$ 后，超额损失 $\Delta L(n) = L(n) - L_\infty$ 在双对数坐标下是一条斜率为 $-\alpha$、截距为 $c$ 的直线：
$$\log\big(L(n) - L_\infty\big) = -\alpha \log n + c$$
两边同时取指数（令常数 $A = e^c$）：
$$L(n) - L_\infty = e^c \cdot e^{-\alpha \log n} = A \cdot n^{-\alpha} \iff \mathbf{L(n) = L_\infty + \frac{A}{n^\alpha}}$$
这就是贯穿整个 Scaling Laws 领域的标准**三参数幂律方程（`Pow3`）**。

#### 2. 经典参数统计基准：一维高斯均值估计（$\alpha = 1$）
暂时忘掉语言模型，考虑最简单的统计参数估计问题：
* **问题设定**：已知独立同分布样本 $X_1, X_2, \dots, X_n \stackrel{\text{i.i.d.}}{\sim} \mathcal{N}(\mu, \sigma^2)$，我们用样本均值 $\hat{\mu}_n = \frac{1}{n}\sum_{i=1}^n X_i$ 估计真实均值 $\mu$。
* **误差缩放推导**：由于 $\hat{\mu}_n \sim \mathcal{N}\big(\mu, \frac{\sigma^2}{n}\big)$，其期望均方误差（MSE Loss）精确等于方差：
  $$L(n) = \mathbb{E}\big[(\hat{\mu}_n - \mu)^2\big] = \frac{\sigma^2}{n} = \sigma^2 \cdot n^{-1}$$
  两边取对数：
  $$\log L(n) = -\mathbf{1} \cdot \log n + \log(\sigma^2)$$
* **推广到一般有限维参数模型（Parametric Models，如 $d$ 维线性回归或极大似然估计）**：
  在经典参数统计中，只要模型是有限维参数化的（维度 $d$ 固定），当样本量 $n \to \infty$ 时，超额风险均以 **$O\big(\frac{d}{n}\big) \propto n^{-1}$** 的速度衰减！也就是说，如果神经网络表现得像一个经典参数模型，我们在双对数图上看到的**斜率应该接近 $-1$（即 $\alpha = 1$）**。

#### 3. 残酷的现实：神经网络的实测指数只有 $\alpha \approx 0.07 \sim 0.35$！
然而，当我们去查 [Hestness et al. (2017)](https://arxiv.org/abs/1712.00409) 与 [Kaplan et al. (2020)](https://arxiv.org/abs/2001.08361) 在真实深度学习任务上拟合出的指数 $\alpha$ 时，看到的却是另一番景象：

| 任务领域与模型类型 | 数据来源文献 | 实测数据扩展指数 $\alpha$ | 与经典参数率 ($\alpha = 1$) 的对比与物理含义 |
| :--- | :--- | :--- | :--- |
| **经典参数估计 (均值估计 / 线性回归)** | 数理统计理论基准 | **$\alpha = 1.00$** | 样本量扩大 $10\times$，误差直接缩小 $10\times$ |
| **字符级语言建模 (Character LM, RHN/LSTM)** | [Hestness et al. (2017)](https://arxiv.org/abs/1712.00409) | **$\alpha \approx 0.07 \sim 0.09$** | 极慢！样本量需扩大 $10^{10}$ 倍以上才能让超额误差缩小 $10\times$ |
| **神经机器翻译 (NMT) / 语音识别** | [Hestness et al. (2017)](https://arxiv.org/abs/1712.00409) | **$\alpha \approx 0.13 \sim 0.35$** | 比字符级 LM 略快，但仍远慢于 $n^{-1}$ |
| **Transformer 语言模型 (WebText2 Token LM)** | [Kaplan et al. (2020)](https://arxiv.org/abs/2001.08361) | **$\alpha_D = 0.095 \approx 0.10$** | 典型的现代 LLM 数据学习速率：指数约为 **$-0.1$** |

**谜题**：为什么庞大的神经网络从数据中学习的速度（$n^{-0.1}$）比简单参数估计（$n^{-1}$）慢了整整一个数量级的指数？什么样的统计估计器会给出 $n^{-1/d}$ 这样缓慢的收敛率？答案是：**非参数估计（Non-parametric Estimation）**！

#### 4. 非参数光滑函数估计推导：维数灾难与 $n^{-O(1/d)}$ 收敛率
假设现在我们不再假定函数是线性的，而是要估计定义在单位超立方体 $[0, 1]^d$ 上的**任意 $L$-Lipschitz 连续光滑函数 $f: [0, 1]^d \to \mathbb{R}$**：
* **问题设定**：观测样本 $Y_i = f(X_i) + \epsilon_i$，其中 $X_i \sim \text{Uniform}([0, 1]^d)$，噪声 $\epsilon_i \sim \mathcal{N}(0, \sigma^2)$。
* **构造直方图/分箱局部平均估计器（Binning / Nearest-Neighbor Estimator）**：
  我们将 $d$ 维空间 $[0, 1]^d$ 均匀切分成边长为 $h$ 的小超立方体网格（Boxes），对于落入同一个小盒子内的所有样本，直接取其 $Y_i$ 的平均值作为盒内函数值估计 $\hat{f}(x)$。
  1. **以二维空间（$d = 2$，课堂幻灯片示例）为例**：
     若将正方形切分为 $M = \sqrt{n}$ 个小盒子（即每维切 $n^{1/4}$ 段，盒子边长 $h = n^{-1/4}$），则平均每个盒子分到 $k = \frac{n}{\sqrt{n}} = \sqrt{n}$ 个样本：
     * **盒内近似偏差平方（Bias$^2$）**：由于函数 Lipschitz 连续，盒子内部函数值最大波动为 $O(h) = O(n^{-1/4})$，平方后为 $O(n^{-1/2})$；
     * **盒内噪声平均方差（Variance）**：对盒内 $k = \sqrt{n}$ 个独立高斯噪声取平均，方差为 $\frac{\sigma^2}{k} = O(n^{-1/2})$；
     * **总均方误差（MSE）**：$\text{Bias}^2 + \text{Variance} = O(n^{-1/2}) = O\big(n^{-1/2}\big)$（指数已从 $-1$ 降到了 $-0.5$！）。
  2. **推广到任意 $d$ 维空间**：
     在 $d$ 维空间中，若边长为 $h$，则共有 $M = h^{-d}$ 个小盒子，每个盒子平均分到 $k = n \cdot h^d$ 个样本。权衡盒内近似误差 $O(h)$（或平方 $O(h^2)$）与噪声方差 $O\big(\frac{1}{n h^d}\big)$，误差随样本量 $n$ 的衰减率必然退化为：
     $$\text{Error}(n) \propto \mathbf{n^{-\frac{c}{d}}} \implies \log \text{Error}(n) = -\frac{c}{d} \log n + \text{const}$$
     （其中常数 $c$ 取决于光滑度阶数，通常 $c \in [1, 2]$，课堂简化记作 $\alpha \approx \frac{1}{d}$）。

* **深刻物理洞察（Neural Networks as Non-parametric Smoothers on Low-Dimensional Manifolds）**：
  对比 LLM 实测的指数 $\alpha_D \approx 0.1 = \frac{1}{10}$ 与非参数回归率 $n^{-1/d}$，我们可以建立一个极其直观的物理心智模型：**过参数化的深度神经网络在学习自然语言时，表现得就像是一个工作在有效内在维度（Intrinsic Dimension）$d_{\text{eff}} \approx 10 \sim 20$ 维数据流形上的非参数平滑器（如最近邻或核回归器）**！
  * Google Research 的理论团队（[Bahri et al., 2021/2024, *Explaining Neural Scaling Laws*](https://arxiv.org/abs/2102.06701) 及后续工作）正式提出了这一**流形内在维度假说**，通过测量数据表征的流形维度 $d$，试图证明神经扩展指数 $\alpha \propto \frac{1}{d}$。尽管 Tatsu 提醒对流形内在维度的数值估计器本身要保持审慎怀疑，但这一视角完美解释了为什么高能力模型具有极强的拟合灵活性，却需要消耗指数级膨胀的海量数据才能缓慢压低 Loss。

### 2.3 工程实战中的数据扩展定律：配比、重复与动态过滤

单看一条标准数据扩展曲线只能用于“算命（Forecasting）”，而工程师真正关心的是**如何利用扩展定律指导数据管线决策**：

#### 1. 数据分布偏移与多源混合配比（Distribution Shift & Data Mixtures）
假设你有两份数据源：新闻（News）与维基百科（Wikipedia），该如何确定它们的最佳混合比例 $p$？
* **理论规律：数据配比主要改变“截距（Intercept）”，而不改变“斜率（Slope）”**
  Tatsu 在其个人研究（[Hashimoto, 2021, *Model Performance Scaling with Multiple Data Sources*](https://arxiv.org/abs/2107.07625)）中通过含两个异构子分布的线性回归玩具模型及深度网络实证发现：当改变训练数据的混合比例 $p \in [0, 1]$ 时，在双对数图上，**不同配比对应的学习曲线几乎是互相平行的（斜率 $\alpha$ 由模型家族和任务决定，保持不变），数据分布的优劣与多样性完全体现在垂直截距（Offset / Intercept）的高低上**！且由于多样性互补，截距作为混合比例 $p$ 的函数呈凸碗状（混合两个数据源显著优于只用单一数据源）。
* **工程实践的两种路线：Data Mixing Laws vs. DataDecide**
  1. **显式拟合混合扩展律（[Ye et al., 2024, *Data Mixing Laws*](https://arxiv.org/abs/2403.16952)）**：在小模型、小数据规模下网格搜索不同配比 $p$，拟合损失关于配比与规模的泛函形式并外推大算力下的最优配比。
  2. **直接锁定小规模最优配比（[Magnusson et al., 2025, *DataDecide*](https://arxiv.org/abs/2504.11393)）**：Tatsu 指出，在真实工业界实践中，显式拟合多维混合扩展定律往往噪声极大；AI2 等机构的最新大规模实证研究 **DataDecide** 测试了 8 种扩展律外推基线后发现——**直接训练一组小模型，挑出在小规模下验证指标最好的那个数据配比，原封不动地直接放大到大规模训练（No scaling law extrapolation required），效果反而最稳健**！这恰好印证了“数据配比只改截距、不改斜率”的理论：既然两条线平行不交叉，小规模下的冠军在大规模下依然是冠军！

#### 2. 数据受限与多 Epoch 重复扩展定律（Data-Constrained & Infinite Compute Scaling）
随着算力增速远超人类高质量文本的产出速度，如果把同一份高质量数据翻来覆去训练多个 Epoch 会发生什么？

```
  log(Validation Loss)
    ▲
    │ \
    │  \  (前 4 个 Epochs: 实线与虚线几乎重合, 重复训练几乎无损!)
    │   \
    │    \───────────► 实线: 固定唯一数据量 U_D, 靠不断重复 (Multi-epoch Repetition) 增加算力
    │     \             (超过 ~4 Epochs 后边际收益急剧衰减, 严重偏离幂律!)
    │      \
    │       \  虚线: 始终喂入新鲜唯一数据 (Fresh Unique Data, 1-Epoch 理想幂律)
    └─────────────────► log(Total Compute / Total Tokens Processed)
```

* **[Muennighoff et al. (2023)《Scaling Data-Constrained Language Models》](https://arxiv.org/abs/2305.16264) 的“4 Epochs 黄金法则”**：
  研究团队将 Chinchilla 公式中的总训练量 $D$ 替换为**考虑重复衰减后的“有效数据量” $D'$**（其中 $U_D$ 为去重后的唯一 Token 数，$R_D = \frac{D}{U_D} - 1$ 为重复次数，$R_D^* \approx 4$ 为半衰特征常数）：
  $$D' = U_D + U_D \cdot R_D^* \Big(1 - e^{-R_D / R_D^*}\Big)$$
  * **核心工程结论**：在标准预训练配方下，**将数据重复训练最多约 $4\text{ 个 Epochs}$ 时，验证集 Loss 与使用全新数据的理想扩展曲线几乎没有退化**；但一旦超过 4 个 Epochs，继续重复的边际收益就会呈指数衰减（曲线迅速变平）！
* **走向极端：无限算力下的预训练（[Kim, Kotha, Liang & Hashimoto, 2025, *Pre-training under infinite compute*](https://arxiv.org/abs/2509.14786)）**：
  如果算力真的趋于无穷大，而唯一数据量固定（允许跑成百上千个 Epochs），该怎么榨干数据的最后一滴价值？Tatsu 与 Percy 团队的最新研究表明：单纯把模型无限做大或单纯无限增加 Epoch 都会撞墙；此时必须引入**极强的正则化（Aggressive Regularization / Weight Decay）**以及**独立训练多个模型做集成（Model Ensembling）**。更有趣的是，当改变可用唯一数据量时，尽管重度正则化与集成显著压低了整体误差（降低了截距），**其关于唯一数据量的扩展斜率依然与标准单遍训练惊人地一致**！

> **课堂问答：在《Pre-training under infinite compute》那张对比图里，为什么横轴（数据量翻倍）和纵轴（测试损失）看起来像是线性刻度，而不是前面讲的双对数刻度？**
> * **Tatsu 的解答与方法论警示**：首先，那张图的横轴刻度虽然标的是具体数值（如 $100\text{M}, 200\text{M}, 400\text{M}, 800\text{M}$），但实际上每一格都在**翻倍（Doubling）**，因此本质上就是对数轴（Log scale）；其次，纵轴的 Loss 变化范围非常窄（例如仅从 $3.3$ 变到 $2.6$）。**这引出了拟合 Scaling Laws 时极其重要的一个数学陷阱——局部泰勒展开近似（Local Taylor Approximation）**：在非常窄的数值区间内，$\log(y)$ 与 $y$ 是一阶线性等价的，多项式衰减（Polynomial）和指数衰减（Exponential）在局部放大看都像是一条直线！因此，如果实验跨越的算力/数据数量级不够宽，千万不要盲目迷信某种特定函数形式的长期外推结果。

#### 3. 算力感知的动态数据过滤（Compute-Aware Data Filtering）
在 [Goyal et al. (2024)《Scaling Laws for Data Filtering—Data Curation cannot be Compute Agnostic》](https://arxiv.org/abs/2404.07177) 中，研究者揭示了一个极具实操价值的交叉现象：**不同严格程度的数据过滤策略在双对数图上会产生“曲线交叉（Crossing Scaling Curves）”**！

```
  log(Test Loss)
    ▲
    │ \
    │  \    (B) 宽松过滤 / 原始数据池 (Less Aggressive Filter: 数据量大, 早期截距高, 但后期可持续下降!)
    │   \  /
    │    \/  ◄── 交叉点 (Crossover Point)! 过了这个算力阈值, 激进过滤因为数据耗尽被迫多轮重复而反遭反超!
    │    /\
    │   /  \──────► (A) 激进过滤 (Aggressive Filter: 只留 Top 10% 精品, 早期截距极低, 但大算力下数据耗尽!)
    └──────────────────► log(Compute Budget C)
```

* **工程推论**：数据清洗与过滤绝不能脱离算力预算（Compute-agnostic）而静态存在！
  * **当你只有小算力时（如学术界实验室或 Assignment 4）**：应该**极度激进地过滤数据（Filter Aggressively）**，只保留最高质量的精粹子集，因为你本来就训不完整个互联网，吃精品能直接享受最低的初始截距；
  * **当你拥有超大算力时（如 Frontier Labs）**：为了避免在精品小集合上重复太多个 Epoch 导致过拟合，你必须**随着算力规模扩大而动态放宽过滤阈值（Loosen the filters）**，纳入更多中等质量的数据以维持单遍/少遍扩展斜率（或者对精品数据做适度重采样、对次优数据做单遍扫描）。

## 3. Part II：模型架构与超参数扩展定律（Model & Hyperparameter Scaling Laws）

离开直观的数据扩展，我们进入更神奇的领域：为什么改变模型架构、优化器、深宽比、Batch Size 和学习率，也会呈现出高度可预测的幂律法则？

### 3.1 架构与优化器选型：斜率优势（Slope）vs. 截距优势（Intercept）

当你想验证一个新架构（比如你想用 LSTM、Mamba 或某种新型线性注意力替代标准 Transformer）或者新优化器（SGD vs. Adam）时，无需烧掉一万张 B200，只需在中小算力跨度（如横跨 $3\sim 4$ 个数量级的 FLOPs）上画出两条扩展曲线，对比它们的**斜率（Exponent / Slope）**与**截距（Intercept / Compute Multiplier）**：

```
  log(Test Loss)
    ▲
    │ \
    │  \      ┌── LSTM (早期斜率尚可，但迅速弯曲进入平台期，斜率劣于 Transformer!)
    │   \────►│
    │    \    └── SGD (与 Adam 斜率几乎平行，但水平滞后一个固定算力倍数，即截距劣势!)
    │     \
    │      \
    │       \ ◄── Transformer + GLU + Adam (保持更陡峭的幂律斜率 + 更低的垂直截距!)
    └────────────────────────► log(Compute FLOPs)
```

1. **架构对决一：Transformer vs. LSTM（[Kaplan et al., 2020](https://arxiv.org/abs/2001.08361)）**
   在 Kaplan 论文中，作者对比了不同层数的 LSTM 与 Transformer 随参数量/算力的扩展曲线：
   * LSTM 不仅在相同算力下有着更高的测试损失（**截距更差**），而且随着模型规模放大以及上下文位置变远，LSTM 曲线迅速变缓（**斜率更平缓/存在渐近瓶颈**）；
   * 相比之下，Transformer 在整个观测区间内保持恒定且更陡峭的对数线性斜率。在 Scaling Laws 范式下，**斜率更陡（Larger $\alpha$）具有一票否决级的统治力**——哪怕某个新架构在极小规模时截距略高，只要它的斜率更陡，放大到百亿/千亿规模后就必然实现反超；反之，若斜率更差，规模越大输得越惨（这也是今天 Mamba、Gated DeltaNet 等新架构论文中必画的第一张图）。

2. **架构对决二：Google T5 家族大规模架构普查（[Tay et al., 2022, *Scaling Laws vs Model Architectures*](https://arxiv.org/abs/2207.10551)）**
   Tatsu 特别推荐了 Yi Tay 等人在 Google 做的这篇覆盖十余种架构变体的系统性扩展律研究。哪怕当时的实验算力远小于今天，其扩展曲线却精准预言了今天前沿大模型的所有架构取舍：
   * **被淘汰的“伪高效”架构（如 Performer 核近似线性注意力）**：在小规模时看似能跑，但其随算力扩展的**斜率明显比标准 Vanilla Transformer 平缓**，越放大差距越悬殊，因此被现代主流大模型彻底弃用；
   * **胜出的现代标配架构（如 GLU 门控线性单元、Switch Transformer 混合专家 MoE、Mixture of Softmaxes）**：在整个算力跨度上稳定压在 Vanilla Transformer 曲线的下方（显著的截距与斜率双重优势），这也正是为什么 SwiGLU 与 MoE 成为了今天所有旗舰模型的标配。

3. **优化器对决：SGD vs. Adam（[Hestness et al., 2017](https://arxiv.org/abs/1712.00409)）**
   在循环高速网络（RHN）语言模型上对比 SGD 与 Adam 时，出现了一个令 Tatsu 每次看到都感到惊讶的现象：**Adam 与 SGD 拟合出的幂律指数几乎一模一样（两条直线近乎严格平行，$\alpha_{\text{Adam}} \approx \alpha_{\text{SGD}} \approx -0.07$），Adam 的全部优势体现为显著降低了垂直截距（即在达到相同 Loss 时，Adam 比 SGD 节省了一个恒定倍数的计算量/数据量）**！

### 3.2 模型形状与参数记账陷阱（Aspect Ratio, Depth & Parameter Counting）

#### 1. 深度与深宽比（Aspect Ratio $d_{\text{model}} / n_{\text{layers}}$）的尺度不变性
在 Lecture 3 中，我们从开源模型的经验统计发现大家普遍取 $d_{\text{model}} / n_{\text{layers}} \approx 60 \sim 128$。在 [Kaplan et al. (2020)](https://arxiv.org/abs/2001.08361) 中，这一选择得到了严格的 Scaling Law 验证：
* **深度退化边界**：当层数 $n_{\text{layers}} = 1$ 时，扩展斜率极差；当 $n_{\text{layers}} = 2$ 时次之；而一旦层数足够深（如 $\ge 6\sim 12$ 层），不同深度的模型在同一参数量下的表现迅速收敛。
* **寻找尺度不变超参（Scale-Invariant Hyperparameters）**：层数 $n_{\text{layers}}$ 本身不是尺度不变的（模型越大，需要的层数越多）；但**深宽比（Aspect Ratio $\equiv d_{\text{model}} / n_{\text{layers}}$）是一个近似尺度不变的量**！当横跨 $10^7$、$10^8$、$10^9$ 不同参数规模绘制测试损失随 $d_{\text{model}} / n_{\text{layers}}$ 变化的 U 型曲线时，所有规模的极小值碗底都稳定落在 **$d_{\text{model}} / n_{\text{layers}} \in [60, 128]$（中心约在 $100$ 左右）** 的宽阔平坦盆地内。这让架构师可以放心地固定深宽比并按比例同步放大深度与宽度。

#### 2. “并非所有参数生而平等”：Embedding 参数剔除与 MoE 稀疏度扩展
Tatsu 强调了 Percy 常说的一句名言：**“Scaling Laws 并不是自然界自动掉下来的魔法，跨数量级的可预测性是被精心工程化（Engineered）设计出来的！”** 其中最关键的一步就是**如何定义横轴 $x$（什么是“参数量 $N$”）**：

* **发现一：词嵌入参数（Embedding Parameters）引起的曲线畸变（[Kaplan et al., 2020](https://arxiv.org/abs/2001.08361)）**：
  在 Transformer 中，总参数量 $N_{\text{total}}$ 包含两部分：参与各层注意力与 FFN 矩阵计算的**非嵌入参数（Non-embedding parameters, $N_{\text{non-emb}} \approx 12 L d_{\text{model}}^2$）**，以及仅做查表/输出的**词嵌入参数（Embedding parameters, $N_{\text{emb}} = |V| d_{\text{model}}$）**。
  * 当网络很浅（如 $L=2$ 或 $L=3$）而词表很大（$|V| = 50,257$）时，$N_{\text{emb}}$ 占据了总参数量的绝大部分！如果以总参数量 $N_{\text{total}}$ 为横轴绘图，浅层模型的曲线会严重向右上方弯曲偏离幂律；
  * 反之，当 [Kaplan et al. (2020)](https://arxiv.org/abs/2001.08361) **将词嵌入参数剔除，仅以非嵌入参数 $N_{\text{non-emb}}$ 作为横轴**时，从 2 层到 64 层的所有模型瞬间坍缩到了同一条笔直的幂律直线上！（*注：但 Kaplan 在这里顺手把最后一层的 Softmax 输出头参数也一起剔除了，这为后文与 Chinchilla 的巨大分歧埋下了伏笔！*）

* **发现二：混合专家（MoE）中“总参数 $N_{\text{total}}$”与“激活参数 $N_{\text{active}}$”的解耦扩展（[Abnar, Busbridge et al., Apple & MIT 2025](https://arxiv.org/abs/2501.12370)）**：
  在现代 MoE 模型中，每个 Token 仅激活一小部分专家（激活参数 $N_{\text{active}}$ 决定单步 FLOPs，而总参数 $N_{\text{total}}$ 决定显存容量，稀疏度定义为 $S = 1 - \frac{N_{\text{active}}}{N_{\text{total}}}$）。Apple 与 MIT 团队通过大规模 IsoFLOP 实验建立了 MoE 的二维联合扩展定律：
  1. **未激活的“空置参数（Inactive Parameters）”同样降低 Loss**：在固定单步计算量（即固定激活参数 $N_{\text{active}}$ 与训练 Token 数）的前提下，单纯增加总专家数（提升总参数 $N_{\text{total}}$ / 提升稀疏度 $S$），验证集 Loss 依然沿着平滑的幂律轨迹持续下降；
  2. **最优稀疏度随总参数/总算力增加而单调递增**：算力预算越大、模型总规模越大，算力最优的前沿点就越偏向于**更高的稀疏度（More Sparse，即更小的激活比例 $N_{\text{active}} / N_{\text{total}}$）**——这完美解释了为什么从早期的 Mixtral 8x7B（8 选 2，激活率 $25\%$）演进到今天的 DeepSeek-V3（256 选 8，激活率仅 $\sim 3.5\%$），前沿 MoE 正变得越来越稀疏！

### 3.3 临界批大小（Critical Batch Size, $B_{\text{crit}}$）与梯度噪声尺度

在选定架构后，训练大模型最核心的两个动态超参数就是 **批大小（Batch Size $B$）** 与 **学习率（Learning Rate $\eta$）**。从系统工程（Lecture 7–8）的角度看，我们希望全局 Batch Size $B$ 越大越好，因为大 Batch 能直接转化为更高的数据并行度（Data Parallelism）与更高的 GPU 算术强度；但从优化角度看，Batch Size 超过某个阈值后就会开始浪费数据样本。如何科学地量化这个边界？

#### 1. 物理图像：方差受限区（Variance-Limited）vs. 偏差/曲率受限区（Bias-Limited）
在随机梯度下降中，单步小批量梯度 $\hat{g}_B = \frac{1}{B}\sum_{i=1}^B \nabla \ell(x_i; \theta)$ 是真实全量梯度 $G = \nabla L(\theta)$ 的带噪估计，其协方差为 $\frac{1}{B}\Sigma$。随着 Batch Size $B$ 增大，训练历经两个截然不同的物理区间（[McCandlish et al., 2018, *An Empirical Model of Large-Batch Training*](https://arxiv.org/abs/1812.06162)）：

```
[区间 A: 小批量 — 方差/噪声受限区 (Noise / Variance-Limited Regime: B << B_crit)]
  真实全量梯度 G ──────► (指向局部最速下降方向)
  小批量样本梯度 g_i ──► ↗ ↘ ↖ (方向剧烈抖动，被随机噪声主导!)
  ✅ 此时将 Batch Size 翻倍 (B -> 2B)：梯度方差减半，单步可迈出的有效步长翻倍！
     达到目标 Loss 所需的优化器步数 S 直接减半，而总消耗样本数 E = B · S 保持不变（100% 线性加速效率！）

[区间 B: 超大批量 — 偏差/局部曲率受限区 (Bias / Curvature-Limited Regime: B >> B_crit)]
               ╭─────────────────╮
             ╭─╯  损失函数等高线  ╰─╮
  当前点 θ ●───► 真实局部梯度方向 G  │   ★ 全局/局部极小值点 θ*
             ╰─╮  (受限于局部海森曲率，并不直指极小值 ★！)
               ╰─────────────────╯
  ❌ 当 B 已经大到彻底消除了梯度方差（g_B ≈ G）后，继续增大 B 毫无意义！
     因为梯度下降只能看见局部一阶切线方向 G，而 G 与通往极小值 ★ 的真实路径之间存在固有的几何曲率偏差（Bias）！
     此时受限于局部曲率最大安全步长，单步进展封顶：步数 S 锁死在 S_min 不再减少，继续翻倍 B 只会让总样本数 E 白白翻倍浪费！
```

#### 2. 步数 $S$ 与样本数 $E$ 的双曲线权衡方程与 $B_{\text{crit}}$ 严格推导
为了在实战中测量从“完美线性并行区”向“样本浪费区”过渡的拐点，[McCandlish et al. (2018)](https://arxiv.org/abs/1812.06162) 给出了如下机械化测量与推导流程：
* **第一步：选定一个目标损失值 $L_{\text{target}}$**，并在不同批大小 $B$ 下（每个 $B$ 均调优其最佳学习率）训练模型，记录达到 $L_{\text{target}}$ 所需的**训练步数 $S(B)$（Number of Steps，决定串行训练墙钟时间）**与**消耗的总样本/Token 数 $E(B) = B \cdot S(B)$（Number of Examples，决定总算力 FLOPs 成本）**。
* **第二步：建立双曲线折衷方程（Hyperbolic Trade-off Equation）**
  设在无限小批量（每次只走最精细的一小步，完全不浪费样本）下达到 $L_{\text{target}}$ 所需的最小样本数为 **$E_{\min}$**；在无限大批量（彻底消除噪声，每步走到海森曲率允许的极限）下达到 $L_{\text{target}}$ 所需的最小步数为 **$S_{\min}$**。通过对局部二次泰勒展开求期望，可证对任意批大小 $B$，$S$ 与 $E$ 满足极其优美的倒数守恒律：
  $$\mathbf{\left(\frac{S}{S_{\min}} - 1\right)\left(\frac{E}{E_{\min}} - 1\right) = 1 \iff \frac{S_{\min}}{S} + \frac{E_{\min}}{E} = 1}$$
* **第三步：定义并求解临界批大小（Critical Batch Size, $B_{\text{crit}}$）**
  我们希望找到一个平衡点，使步数膨胀代价 $\frac{S}{S_{\min}}$ 与样本膨胀代价 $\frac{E}{E_{\min}}$ 恰好相等（即两边各退让一步，令 $\frac{S_{\min}}{S} = \frac{E_{\min}}{E} = \frac{1}{2}$，此时 $S = 2 S_{\min}$ 且 $E = 2 E_{\min}$）。将 $E = B \cdot S$ 代入上式：
  $$\frac{S_{\min}}{S} + \frac{E_{\min}}{B \cdot S} = 1 \implies S(B) = S_{\min}\left(1 + \frac{E_{\min}}{S_{\min}} \cdot \frac{1}{B}\right)$$
  定义**临界批大小（Critical Batch Size）**为最小样本数与最小步数之比：
  $$\mathbf{B_{\text{crit}} \triangleq \frac{E_{\min}}{S_{\min}}}$$
  则任意批大小 $B$ 下的训练步数 $S(B)$ 与总样本数 $E(B)$ 可直接写为：
  $$\mathbf{S(B) = S_{\min}\left(1 + \frac{B_{\text{crit}}}{B}\right), \qquad E(B) = E_{\min}\left(1 + \frac{B}{B_{\text{crit}}}\right)}$$
  * **物理意义检验**：当选定 **$B = B_{\text{crit}}$** 时，$S(B_{\text{crit}}) = 2 S_{\min}$（串行时间仅比无限大批量极限慢 1 倍），同时 $E(B_{\text{crit}}) = 2 E_{\min}$（总算力/数据消耗仅比单样本极限多 1 倍）！此外，[McCandlish et al. (2018)](https://arxiv.org/abs/1812.06162) 证明 $B_{\text{crit}}$ 在一阶近似下等于**梯度噪声尺度（Gradient Noise Scale）**：
    $$B_{\text{crit}} \approx B_{\text{noise}} = \frac{\text{tr}(\Sigma)}{\|G\|_2^2}$$
    （即梯度协方差矩阵的迹与真实梯度模长平方之比：噪声相对于信号越大，$B_{\text{crit}}$ 就越大！）。

#### 3. $B_{\text{crit}}$ 的扩展定律：为什么训练越到后期、Loss 越低，能承受的 Batch Size 越大？
为什么把临界批大小放在 Scaling Laws 这一讲？因为 [Kaplan et al. (2020)](https://arxiv.org/abs/2001.08361) 发现：**临界批大小 $B_{\text{crit}}$ 几乎不直接依赖于模型参数量 $N$，而是随着训练推进、目标损失 $L$ 的下降，呈现出严格的反比幂律增长**！
$$\mathbf{B_{\text{crit}}(L) \approx \frac{B_*}{L^{1 / \alpha_B}}}$$
（在 Kaplan 实验中，以 Token 数计的 $B_* \approx 2 \times 10^8$，$\alpha_B \approx 0.21$，即 $1/\alpha_B \approx 4.8$）。
* **第一性原理直觉**：在训练刚开始时（Loss 很高），模型离极小值很远，每个样本都给出强烈且方向一致的梯度信号（$\|G\|_2^2$ 极大），信噪比极高，因此很小的 Batch Size 就足以指明方向（$B_{\text{crit}}$ 很小）；而随着训练深入、**Loss 越来越低（逼近极小值盆地底部）**，一阶梯度模长 $\|G\|_2^2 \to 0$，有用信号变得极其微弱，不同样本间的随机噪声 $\text{tr}(\Sigma)$ 成为主导，此时必须大幅增加 Batch Size 进行方差归约才能分辨出微弱的下降方向！
* **直接指导工业界预训练配方**：
  1. **更大算力的训练允许使用更大的全局 Batch Size**：因为大算力运行的目标终点 Loss 更低，对应的 $B_{\text{crit}}$ 显著更大（从早期的几十万 Tokens 增长到百万甚至千万 Tokens）；
  2. **动态批大小预热（Batch Size Warmup / Ramp-up）**：既然早期 $B_{\text{crit}}$ 小、后期 $B_{\text{crit}}$ 大，许多旗舰模型（如 [Llama 3](https://arxiv.org/abs/2407.21783) 从 4M Tokens 逐步翻倍至 16M Tokens，[DeepSeek-V3](https://arxiv.org/abs/2412.19437) 从 3072 序列逐步拉升至 15360 序列）都会在训练中途随 Loss 下降动态调大 Batch Size，从而兼顾早期的快速收敛与后期的海量并行吞吐！

### 3.4 学习率（Learning Rate）随规模演进的两大流派

当你把模型宽度 $d_{\text{model}}$（或参数量 $N$）放大时，最优学习率 $\eta^*$ 会如何移动？

```
[流派一: 标准参数化 (SP) + 经验缩放律预测]        [流派二: 最大更新参数化 (μP) 重标度]
  Validation Loss                                  Validation Loss
    ▲   128维   512维   2048维 (模型越宽, 最优 LR 越往左移!)  ▲   128维 / 512维 / 2048维
    │   \  /    \  /    \  /                         │          \      /
    │    \/      \/      \/                          │           \    /
    │    ★───────★───────★  (拟合 η*(N) 幂律外推)      │             ★ (所有宽度的最优 LR 完全对齐!)
    └─────────────────────────► log(Learning Rate)   └─────────────────────────► log(Learning Rate)
```

1. **流派一：标准参数化（Standard Parameterization, SP）+ 学习率扩展律外推**
   在常规初始化下，随着模型变宽、参数增多，单步同时更新的权重维度增加，为了避免激活值爆炸或发散，**模型越大，最优学习率 $\eta^*$ 必须越小**（在纯宽度扩展下经验法则约为 $\eta^* \propto \frac{1}{d_{\text{model}}}$，在联合扩展下同样遵循平滑的对数线性移动）。工程上可以通过在一系列小模型上扫描出各自的 U 型碗底 $\eta^*(N)$，拟合一条关于规模的缩放曲线来预测大模型的学习率。
2. **流派二：最大更新参数化（$\mu$P, Maximal Update Parameterization）**
   另一种思路是直接修改神经网络各层的初始化方差与优化器步长随宽度 $d_{\text{model}}$ 的缩放因子（即 $\mu$P，将在 **Lecture 11** 深入推导），**从数学上强行把所有宽度模型的最优学习率极小值锁死在同一个坐标点上**！这样在小模型（如 40M）上网格搜索出的最佳学习率 $\eta^*$，可以直接零修改迁移（Zero-shot transfer）到百亿大模型上。Tatsu 指出，目前工业界两大流派均有成功的大规模实践案例，且越来越多团队倾向于将两者结合或使用显式的学习率 Scaling Laws。

### 3.5 警惕预训练与下游迁移的脱节（Upstream vs. Downstream）及课堂问答

在本节末尾，Tatsu 给所有沉迷于拟合完美 Perplexity 直线的工程师浇了一盆清醒的冷水：
* **上游 Perplexity 与下游任务迁移并非总是单调一致！**
  在 [Tay et al. (2022)《Scaling Laws vs Model Architectures》](https://arxiv.org/abs/2207.10551) 的图 4 中，作者展示了令人震惊的一幕：
  * **在上游预训练验证集 Log-Likelihood（Perplexity）上**：参数量与上游 Loss 呈现出教科书般完美的线性相关，其中 12 层的浅宽模型 `NL12` 取得了全场最佳的上游 Loss；
  * **但在下游 17 个零样本/少样本任务平均分（Downstream Few-Shot Accuracy）上**：上游表现最好的 `NL12` 竟然掉到了倒数！反而是上游 Perplexity 差得多、但拥有 32 层深度的 `NL32-XL` 夺得了下游任务的绝对冠军！
* **工程铁律**：Scaling Laws 在上游交叉熵损失（Perplexity）上极其干净、平滑、低方差，因此非常适合用做第一道筛选器；但从上游 Loss 到下游推理/指令跟随能力的迁移（Transfer）强烈依赖于模型深度、归纳偏置与预训练数据分布。Tatsu 打趣道，他去工业界做 Post-training（后训练）的毕业学生经常吐槽预训练团队：“他们丢给我们一个 Perplexity 刷得很漂亮但下游根本训不动的模型，说现在是你们后训练的问题了——但其实病根在预训练阶段就已经种下了！”

> **课堂精彩三连问（Q&A Session）**：
> 1. **问：在画 Scaling Law 散点图时，每个数据点是怎么处理随机种子方差（Variance）的？是跑多次取平均还是取最小值？**
>    * **Tatsu 答**：在几乎所有预训练 Scaling Laws 论文中，**每个散点都只跑了 1 次（Singletons，单次运行）**！为什么敢只跑 1 次？因为**上游交叉熵验证集 Loss（Perplexity）是一个方差极小、极其干净的统计量**——预训练数据动辄数十亿 Tokens，验证集也有数千万 Tokens，对数百万个 Token 的对数似然求平均后，随机种子带来的波动通常只在**小数点后第 2～3 位**！但如果你是在拟合学习率极值点或临界批大小 $B_{\text{crit}}$，测量噪声就会非常恐怖，此时必须做曲线平滑或多次采样。
> 2. **问：既然下游指标才是最终目的，为什么不直接在下游 Benchmark 准确率上拟合 Scaling Laws？**
>    * **Tatsu 答**：有人尝试这么做，但下游指标存在两大硬伤——**高方差（High Variance）**与**离散阈值效应（Discontinuous / Sigmoidal Emergence）**。在小模型阶段，许多下游任务（如算术或复杂推理）准确率长期趴在随机猜测水平（$0\%$ 或 $25\%$）纹丝不动，或者呈现剧烈锯齿状抖动，你根本无法从一条锯齿状的折线判断它未来是直线还是曲线。因此最稳健的哲学是：**先在低方差的 Perplexity 上建立坚实的扩展规律，再通过独立的迁移验证实验确保上游优势能传导至下游**。
> 3. **问：为什么很多 Scaling Laws 论文里混用训练损失（Train Loss）和测试/验证损失（Test/Val Loss）？**
>    * **Tatsu 答**：严谨的科学论文（如 Kaplan 2020）都会在独立的 Validation/Test 集上评估。但在现代大模型的**单遍训练（One-pass / Single-epoch SGD）**模式下，模型看到的每一个训练 Batch 都是从未见过的新鲜采样数据！因此在没有数据重复（No Repetition）的前提下，**每一个训练步的在线训练损失（Online Train Loss）本身就是泛化测试损失的无偏估计**，Train Loss 与 Val Loss 几乎完全重合（甚至有些工业界预训练代码库连单独的 Validation Loop 都省了，直接看平滑后的 Online Train Loss）。只有当涉及多 Epoch 重复训练或过拟合时，两者才会分叉。

## 4. Part III：算力最优联合扩展定律（Compute-Optimal Scaling: Kaplan vs. Chinchilla）

现在我们来到本讲最高潮、也是 Assignment 3 的核心实战主题：**给定固定的总算力预算 $C$（单位：FLOPs），如何同时决定模型参数量 $N$ 与训练数据量 $D$？**

### 4.1 核心资源分配问题与联合损失泛函（Joint Scaling Laws）

回顾 Lecture 2 推导的 Transformer 预训练黄金算力公式：
$$\mathbf{C \approx 6 N D}$$
这意味着在固定算力预算 $C$ 下，参数量 $N$ 与数据 Token 数 $D$ 是严格反比制约的（$N \cdot D = \frac{C}{6}$）：
* 如果你把模型造得太小（$N$ 太小），而把所有算力都用来喂海量数据（$D$ 极大），小模型很早就会撞上容量上限，后续算力全部打水漂（对应训练曲线顶部早早变平的深紫色线）；
* 如果你把模型造得太大（$N$ 极大），算力只够跑寥寥几个步数（$D$ 极小），巨型模型还没来得及收敛算力就烧光了。

为了精确刻画 $N$ 和 $D$ 的联合作用，[Rosenfeld et al. (2019)](https://arxiv.org/abs/1909.12673)、[Kaplan et al. (2020)](https://arxiv.org/abs/2001.08361) 与 [Hoffmann et al. (2022, Chinchilla)](https://arxiv.org/abs/2203.15556) 相继提出了二元联合扩展定律 $L(N, D)$：

1. **加性独立幂律形式（Rosenfeld 2019 / Chinchilla 2022 采用，最直观常用）**：
   $$\mathbf{L(N, D) = E + \frac{A}{N^\alpha} + \frac{B}{D^\beta}}$$
   * **通过取极限检验物理意义（Tatsu 强烈推荐的公式审视技巧）**：
     * **无限数据极限（$D \to \infty$）**：$\frac{B}{D^\beta} \to 0$，公式退化为 $L(N, \infty) = E + \frac{A}{N^\alpha}$，即受限于模型有限容量的**纯模型规模扩展律（Model Capacity Bottleneck）**；
     * **无限模型极限（$N \to \infty$）**：$\frac{A}{N^\alpha} \to 0$，公式退化为 $L(\infty, D) = E + \frac{B}{D^\beta}$，即受限于有限训练样本的**纯数据扩展律（Sample Complexity Bottleneck）**；
     * **双无限极限（$N \to \infty, D \to \infty$）**：$L(\infty, \infty) = E$，收敛于自然语言的**固有熵下界（Irreducible Entropy）**。

2. **Kaplan (2020) 的幂次合成形式（数学渐近等价）**：
   $$L(N, D) = \left[\left(\frac{N_c}{N}\right)^{\frac{\alpha_N}{\alpha_D}} + \frac{D_c}{D}\right]^{\alpha_D}$$
   同样满足 $D \to \infty$ 时 $L(N) = (N_c / N)^{\alpha_N}$，以及 $N \to \infty$ 时 $L(D) = (D_c / D)^{\alpha_D}$。

#### 固定算力 $C = 6ND$ 下的算力最优分配闭式推导
给定任意加性联合损失 $L(N, D) = E + A N^{-\alpha} + B D^{-\beta}$，我们要在算力约束 $C = 6ND$ 下求解使 $L(N, D)$ 最小的 $(N_{\text{opt}}, D_{\text{opt}})$：
1. 将约束 $D = \frac{C}{6N}$ 代入损失函数，化为关于 $N$ 的一元函数：
   $$L(N; C) = E + A N^{-\alpha} + B \left(\frac{C}{6}\right)^{-\beta} N^\beta$$
2. 对 $N$ 求导并令导数为 $0$（$\frac{\partial L}{\partial N} = 0$）：
   $$-\alpha A N^{-\alpha - 1} + \beta B \left(\frac{C}{6}\right)^{-\beta} N^{\beta - 1} = 0 \implies \alpha A N^{-\alpha} = \beta B D^{-\beta}$$
   > **黄金平衡等式**：在算力最优前沿（Compute-Optimal Frontier）上，**由模型容量有限引起的超额损失项 $\frac{A}{N^\alpha}$，必须与由数据量有限引起的超额损失项 $\frac{B}{D^\beta}$ 保持恒定的常数比例（$\frac{\alpha}{\beta}$）**！既不能让模型成为短板，也不能让数据成为短板。
3. 联立 $N^{\alpha + \beta} = \frac{\alpha A}{\beta B} \left(\frac{C}{6}\right)^\beta$，立即解出最优参数量 $N_{\text{opt}}(C)$ 与最优数据量 $D_{\text{opt}}(C)$ 的幂律缩放指数 $a$ 和 $b$：
   $$\mathbf{N_{\text{opt}}(C) \propto C^a, \qquad D_{\text{opt}}(C) \propto C^b, \qquad \text{其中 } a = \frac{\beta}{\alpha + \beta}, \quad b = \frac{\alpha}{\alpha + \beta} \quad (a + b = 1)}$$

### 4.2 历史分水岭：Kaplan (2020) vs. Chinchilla (2022) 的大论战

虽然求解最优分配的微积分推导一模一样，但 [Kaplan et al. (2020, OpenAI)](https://arxiv.org/abs/2001.08361) 与两年后的 [Hoffmann et al. (2022, DeepMind Chinchilla)](https://arxiv.org/abs/2203.15556) 却测出了截然不同的指数 $(a, b)$，直接改写了整个大模型行业的历史走向：

| 对比维度 | [Kaplan et al. (2020, OpenAI)](https://arxiv.org/abs/2001.08361) | [Hoffmann et al. (2022, DeepMind Chinchilla)](https://arxiv.org/abs/2203.15556) |
| :--- | :--- | :--- |
| **最优模型参数缩放指数 $a$ ($N_{\text{opt}} \propto C^a$)** | **$a = 0.73$** | **$a \approx 0.50$** (Method 1: $0.50$, Method 2: $0.49$, Method 3: $0.46$) |
| **最优数据 Token 缩放指数 $b$ ($D_{\text{opt}} \propto C^b$)** | **$b = 0.27$** | **$b \approx 0.50$** (Method 1: $0.50$, Method 2: $0.51$, Method 3: $0.54$) |
| **算力扩大 $10\times$ 时的分配策略** | 模型扩大 **$5.4\times$**，数据仅增加 **$1.86\times$** | 模型扩大 **$3.16\times$ ($\sqrt{10}$)**，数据同步扩大 **$3.16\times$ ($\sqrt{10}$)** |
| **每个参数对应的训练 Token 数 ($D / N$)** | 随算力增加而**递减** ($\propto C^{-0.46}$) | **保持恒定常数**：**$\frac{D_{\text{opt}}}{N_{\text{opt}}} \approx 20\text{ Tokens / Parameter}$** |
| **指导诞生的代表性旗舰模型** | • **GPT-3 (175B)**：仅训 $300\text{B}$ Tokens ($1.7\text{ tokens/param}$)<br>• **Gopher (280B)**：仅训 $300\text{B}$ Tokens ($1.1\text{ tokens/param}$)<br>• **MT-NLG (530B)**：仅训 $270\text{B}$ Tokens ($0.5\text{ tokens/param}$) | • **Chinchilla (70B)**：在与 Gopher (280B) **完全相同的算力预算**下，将模型缩小 4 倍至 **70B**，数据扩大 4 倍至 **1.4T Tokens** ($20\text{ tokens/param}$)，在几乎所有评测上全面碾压 280B Gopher！ |

### 4.3 Chinchilla 的三种扩展律拟合方法详解（Assignment 3 核心实战）

为了确保结论坚如磐石，[Hoffmann et al. (2022)](https://arxiv.org/abs/2203.15556) 设计了三种截然不同的实验与统计拟合方法来估计 $(a, b)$：

```
[Approach 1: 训练曲线下包络线法 (Lower Envelope)]      [Approach 2: IsoFLOP 切面抛物线法 (Tatsu 最推崇!)]
  Loss                                                  Terminal Loss
    ▲  小模型N1  中模型N2   大模型N3                      ▲    算力 C1        算力 C2        算力 C3
    │   \         \          \                            │   \      /       \      /       \      /
    │    \─────────\──────────\──► 提前饱和               │    \    /         \    /         \    /
    │     ` .       \          \                          │      ★              ★              ★
    │         ` .    ` .        \                         │      │              │              │
    │             ` .    ` .     ` . ◄── 取各算力下最低   │      ▼              ▼              ▼
    │                 ` .    ` .     ` . 包络切点拟合!    │   N_opt(C1)      N_opt(C2)      N_opt(C3)
    └──────────────────────────────────► log(FLOPs C)     └──────────────────────────────────────► log(Params N)
```

#### 方法一：训练曲线下包络线法（Approach 1: Fix Model Sizes, Vary Training Steps / Lower Envelope）
* **操作流程**：固定一组模型尺寸 $N_1, N_2, \dots, N_k$，对每个模型训练不同的总步数（针对不同训练长度分别设定 Cosine 学习率衰减周期），将所有训练轨迹绘制在横轴为已消耗算力 $C$、纵轴为 Loss 的图上。
* **提取前沿**：对于每一个给定的算力点 $C$，找出所有训练曲线中达到的**最低损失点（Lower Envelope，下包络线）**，记录下创造该最低损失的模型参数量 $N_{\text{opt}}(C)$ 与数据量 $D_{\text{opt}}(C)$，最后在双对数坐标下对 $(C, N_{\text{opt}})$ 和 $(C, D_{\text{opt}})$ 做线性回归。
* **结果与优缺点**：拟合得出 **$a = 0.50, \; b = 0.50$**（预测 Gopher 算力下最优模型为 **67B**）。缺点是“下包络线”的插值与切点提取对曲线密度和噪声较为敏感。

#### 方法二：IsoFLOP 等算力切面法（Approach 2: IsoFLOP Profiles —— 工业界与 Assignment 3 最通用黄金标准！）
* **为什么 Tatsu 最推崇 IsoFLOP 法？** 因为它的实验设计最解耦、统计拟合最简单鲁棒，几乎不需要复杂的曲面优化假设！
* **操作流程**：
  1. 预先选定若干个离散的**固定算力预算档位（IsoFLOP Budgets）** $C_1, C_2, \dots, C_m$（例如从 $6\times 10^{18}$ 到 $3\times 10^{21}$ FLOPs，每档相差 $2\times$ 或 $3\times$）；
  2. 在每一个固定算力档位 $C_i$ 下，扫描一系列不同的模型参数量 $N_{i,1}, N_{i,2}, \dots$，并严格按照 $D_{i,j} = \frac{C_i}{6 N_{i,j}}$ 设定每组实验的训练 Token 数（即模型参数翻倍，训练 Token 数严格减半，确保总 FLOPs 恒等于 $C_i$！）；
  3. 取每次运行结束时的**终点损失（Terminal Loss）**关于 $\log N$ 绘图——每一个算力档位 $C_i$ 都会形成一条漂亮的 **U 型碗状曲线**；
  4. 对每条 U 型曲线关于 $\log N$ 拟合一个**二次抛物线（Quadratic Fit：$L(\log N) = c_2 (\log N)^2 + c_1 \log N + c_0$）**，直接用求根公式算出碗底极小值点 $\log N_{\text{opt}}(C_i) = -\frac{c_1}{2c_2}$；
  5. 将各算力档位的碗底极值点 $\big(\log C_i, \log N_{\text{opt}}(C_i)\big)$ 连成直线，斜率即为 $a$！
* **结果**：拟合得出 **$a = 0.49, \; b = 0.51$**（预测 Gopher 算力下最优模型为 **63B**，与方法一高度吻合！）。如今 IsoFLOP 设计不仅用于语言模型，还被广泛推广到 MoE 稀疏度优化与扩散模型（Diffusion Models）架构搜索中。

#### 方法三：联合参数化损失曲面拟合法（Approach 3: Parametric Loss Surface Fitting）
* **操作流程**：收集所有实验数据点 $(N_k, D_k, L_k)$，直接用 Huber Loss 和 L-BFGS 优化器在二维网格上联合拟合五参数泛函曲面：
  $$\hat{L}(N, D) = E + \frac{A}{N^\alpha} + \frac{B}{D^\beta}$$
  在原论文中，作者拟合出的参数为：$E = 1.69, \; A = 406.4, \; B = 410.7, \; \alpha = 0.34, \; \beta = 0.28$。
* **原论文中的微小瑕疵**：由此算出的缩放指数为 $a = \frac{\beta}{\alpha+\beta} \approx \mathbf{0.46}, \; b = \frac{\alpha}{\alpha+\beta} \approx \mathbf{0.54}$（预测 Gopher 算力下最优模型仅为 **40B**），与方法一、二出现了耐人寻味的偏差（后文 4.4 节将揭晓这一悬案的真相！）。

### 4.4 深度破案：为什么 Kaplan 与 Chinchilla 会得出截然相反的结论？

同为顶级实验室，同样使用加性联合幂律模型，为什么 Kaplan (2020) 算出 $a = 0.73$，而 Chinchilla (2022) 算出 $a = 0.50$？Tatsu 引用了 2024 年发表的三篇精彩绝伦的复现与溯源研究，彻底解开了这桩公案：

#### 1. 破案一：[Porian et al. (2024, Yair Carmon 团队)《Resolving Discrepancies in Compute-Optimal Scaling of Language Models》](https://arxiv.org/abs/2406.19146)
斯坦福校友 Yair Carmon 团队通过受控实验，从完全复现 Kaplan 的 $a \approx 0.73$ 出发，仅仅通过**依次修正三个看似微不足道的工程实现细节**，就让斜率一步步严丝合缝地移动到了 Chinchilla 的 $a \approx 0.50$：

```
[Kaplan 原始设置 (a ≈ 0.73)]
   │
   ├──► 修正 1: 计入最后一层输出头参数 (Include Last-Layer Head / Embedding Params) ──► 斜率大幅下移!
   │
   ├──► 修正 2: 按训练总步数动态调整学习率预热与衰减 (Scale LR Warmup & Decay Horizon) ──► 消除小模型早停惩罚!
   │
   └──► 修正 3: 针对不同规模调优 Batch Size 与优化器超参 (Tune Batch Size & AdamW β2) ──► 完美收敛至 Chinchilla (a ≈ 0.50)!
```

* **罪魁祸首 1：参数记账与算力记账排除了最后一层输出头（Parameter & FLOP Counting Exclusion）**：
  如 3.2 节所述，Kaplan 不仅排除了输入词嵌入矩阵，还因为权重绑定/形状对称，**把最后一层将隐藏状态映射到词表 Logits 的线性输出头（Last-layer Softmax Linear Head，含 $|V| \times d_{\text{model}}$ 个参数及对应的矩阵乘法 FLOPs）也从参数量 $N$ 和计算量 $C$ 中彻底扣除了**！在小模型（如 $d_{\text{model}} = 128 \sim 512$，而 $|V| = 50257$）中，这个输出头占据了实际计算量和参数量的很大比例；扣除它会导致严重低估小模型的真实参数量与真实算力消耗，从而在双对数图上人为拉高了模型尺寸的缩放指数 $a$。
* **罪魁祸首 2：过长的固定步数学习率预热（Fixed-Step LR Warmup）与未对齐的衰减周期**：
  Kaplan 对所有实验使用了**固定步数（如 $3000$ 步）的 Learning Rate Warmup**，且没有像 Chinchilla 那样为每个目标数据量 $D$ 单独对齐 Cosine Decay 的终点步数（而是直接从一条长训练曲线的中途截断读取 Loss）。结果导致在测试“小模型 + 少数据/小算力”组合时，训练总步数甚至比 $3000$ 步的 Warmup 还要短！小模型在还没达到峰值学习率、也没经历学习率衰减的状态下就被强行评估了，人为导致“少数据点”的 Loss 虚高，误导拟合器认为“必须优先把算力投给大模型”。
* **罪魁祸首 3：小模型上未调优的固定大 Batch Size 与优化器超参**：
  Kaplan 对所有规模使用了固定的较大 Batch Size（在小模型高 Loss 区间超出了其临界批大小 $B_{\text{crit}}$），修正为随规模自适应的 Batch Size 与调优后的 AdamW 超参后，最后一点差距也彻底消失。

#### 2. 破案二：[Pearce & Song (2024)《Reconciling Kaplan and Chinchilla Scaling Laws》](https://arxiv.org/abs/2406.12907) 的纯数学仿真还原
Pearce 和 Song 做了一个极其巧妙的“零训练”实验：他们根本没有重新训练任何模型，而是**直接把 Chinchilla 的真实损失方程当成上帝真值（Ground Truth）**，然后在上面模拟 Kaplan 的做法——(1) 只在极小的算力区间（Kaplan 的实验算力比 Chinchilla 小几个数量级）采样；(2) 在横轴中扣除 Embedding 参数 $N_{\text{emb}}$。结果发现：**仅仅因为“在极小算力规模下采样 + 扣除非线性占比的 Embedding 参数”这两个几何扭曲效应，从纯正的 Chinchilla 方程中拟合出的表面局部斜率就会自动变成 Kaplan 的 $a \approx 0.73$**！

#### 3. 破案三：[Besiroglu et al. (2024, Epoch AI)《Chinchilla Scaling: A Replication Attempt》](https://arxiv.org/abs/2404.10102) 破解 Chinchilla Method 3 悬案
为什么 Chinchilla 论文自己的 Method 3（$a=0.46, b=0.54$）会和 Method 1/2（$a=0.50, b=0.50$）不一致？如果 $a=0.46 \neq b=0.54$ 是真的，渐近外推到超大算力时，Token 与参数的比例将不再是常数 20:1！
由于 DeepMind 没有开源原始实验数据表格，**Epoch AI 的研究者直接用工具从 Chinchilla 论文 PDF 的矢量散点图中逆向提取出了所有实验数据点的坐标**，并重新运行了 Method 3 的 L-BFGS 曲面拟合。真相令人忍俊不禁：**Chinchilla 原论文作者在跑 Method 3 的数值优化器时没有完全收敛（陷入了局部次优解 / Underfitted）**！当 Epoch AI 把同一个损失函数真正优化到底后，拟合误差显著降低，且得到的真实最优指数恰好变为 **$a \approx 0.49, \; b \approx 0.51$**——与 Method 1 和 Method 2 完美统一！正如 Tatsu 所感叹的：“Chinchilla 的作者比他们自己以为的还要正确（More right than they knew）！”

### 4.5 超越 Chinchilla：生产环境中的推理经济学与“蓄意过训练”（Inference-Aware Overtraining）

讲完了精彩的 Chinchilla 传奇，Tatsu 在本讲最后抛出了一个反转结论：**“如果今天你在工业界训练一个生产级模型，你大概率绝不会使用 Chinchilla 的 20:1 比例！”**

```
[每个参数对应的训练 Token 数 (Tokens per Parameter, D / N) 历史演进]
  • GPT-3 175B (2020):          ~1.7  tokens/param  (严重欠训练 Undertrained, 追求巨型参数量)
  • Chinchilla 70B (2022):      ~20   tokens/param  (仅最小化单次预训练算力 C_train 的理论最优解!)
  • Llama 1 7B (2023.02):       ~143  tokens/param  (1T tokens, 开启推理友好的小模型过训练时代)
  • Llama 2 7B (2023.07):       ~286  tokens/param  (2T tokens)
  • OLMo / Mistral 7B (2023):   ~400+ tokens/param
  • Llama 3 8B (2024.04):       ~1,875 tokens/param (15T tokens! 相比 Chinchilla 比例过训练了近 100 倍!)
  • Gemma 2/3 & Qwen 3 小模型:  ~1,000 ~ 4,000+ tokens/param
```

* **为什么现代开源与商业模型都在疯狂“过训练（Overtraining）”？**
  因为 **Chinchilla 优化的目标函数错了（对于生产部署而言）**！Chinchilla 假设你的唯一目标是**在给定达到某个目标 Loss 的前提下，最小化一次性预训练算力 $C_{\text{train}} \approx 6ND$**（完全忽略了模型训练好之后的推理部署成本！）。
  然而在真实世界（或开源生态）中，模型只训练 **1 次**，却要被全球数百万用户调用推理 **数万亿次**（[Sardana & Frankle, 2023, *Beyond Chinchilla-Optimal: Accounting for Inference in Language Model Scaling Laws*](https://arxiv.org/abs/2401.00448)）：
  $$\mathbf{C_{\text{total}}(N, D) = \underbrace{6 N D}_{\text{一次性预训练算力 } C_{\text{train}}} + \underbrace{2 N \cdot D_{\text{inference}}}_{\text{全生命周期累计推理算力 } C_{\text{inference}}}}$$
  * **推理端只取决于参数量 $N$，而与预训练吃了多少数据 $D$ 完全无关**！一个吃 15T Tokens 训出来的 8B 模型，与一个吃 160B Tokens（20:1 Chinchilla 比例）训出来的 8B 模型，在推理部署时占用的显存（16GB BF16）和单 Token 延迟**完全一样便宜**，但前者的能力却能媲美 Chinchilla 比例下数十倍大的巨型模型！
  * 因此，现代大模型工程的标准范式是：**根据目标推理硬件显存与延迟上限先定死一个紧凑的模型尺寸 $N$（如 8B、32B 或 MoE 激活 37B），然后在预训练阶段投入远超 Chinchilla 比例的算力（$100 \sim 2000\text{ tokens/param}$）对其进行极限“过训练”**，用预训练阶段多付出的几百万美元电费，换取推理服务阶段节省的数亿美元成本！

## 5. 全景总结：Scaling Laws 核心公式与工程黄金法则速查表（Cheat Sheet）

| 核心模块 | 标准数学形式与关键定律 | 典型经验数值 / 黄金准则 | 核心工程指导意义 |
| :--- | :--- | :--- | :--- |
| **1. 单变量数据扩展律** | $L(D) = L_\infty + A \cdot D^{-\alpha_D}$ | $\alpha_D \approx \mathbf{0.07 \sim 0.10}$ (远慢于参数统计的 $\alpha=1$) | 揭示神经网络具有类似 $d \approx 10$ 维流形非参数平滑器的慢速样本收敛特征 |
| **2. 数据配比与分布偏移** | 不同配比改变截距 $A$，斜率 $\alpha_D$ 基本不变 | 小规模最优配比 $\approx$ 大规模最优配比 ([DataDecide](https://arxiv.org/abs/2504.11393)) | 直接在中小规模单点实验上优选最佳数据混合比例，无需强行外推复杂曲面 |
| **3. 数据重复与动态过滤** | $D' = U_D + U_D R_D^*(1 - e^{-R_D/R_D^*})$ | 安全重复上限：**$\le 4\text{ Epochs}$** ([Muennighoff et al., 2023](https://arxiv.org/abs/2305.16264)) | 小算力应激进过滤保高质截距；大算力须动态放宽过滤阈值避免高质数据过早耗尽 |
| **4. 架构与深宽比选型** | 优先选更陡斜率 $\alpha$，次选更低截距 $A$ | 最优深宽比 $\frac{d_{\text{model}}}{n_{\text{layers}}} \approx \mathbf{60 \sim 128}$（尺度不变） | 用中小规模 IsoFLOP 曲线快速淘汰劣质架构（如 LSTM/Performer），锁定 SwiGLU + MoE |
| **5. 临界批大小扩展律** | $S(B) = S_{\min}\big(1 + \frac{B_{\text{crit}}}{B}\big)$<br>$B_{\text{crit}}(L) \approx \frac{B_*}{L^{1/\alpha_B}}$ | 在 $B = B_{\text{crit}}$ 处 $S = 2S_{\min}, E = 2E_{\min}$；随 Loss 降低按幂律增长 | 决定数据并行最大不浪费样本的 Batch 上限；支撑训练中后期的 **Batch Size Warmup** |
| **6. 算力最优联合扩展** | $L(N, D) = E + A N^{-\alpha} + B D^{-\beta}$<br>约束：$C \approx 6ND$ | **Chinchilla 均衡律**：$N_{\text{opt}} \propto C^{0.5}, D_{\text{opt}} \propto C^{0.5}$<br>**$\frac{D_{\text{opt}}}{N_{\text{opt}}} \approx 20\text{ tokens/param}$** | 用 **IsoFLOP 抛物线拟合法**寻找极值；注意统一计入输出头参数并对齐 Cosine LR 周期 |
| **7. 推理感知过训练** | $\min_{N, D} L(N, D) \text{ s.t. } 6ND + 2ND_{\text{inf}} \le C_{\text{tot}}$ | 现代生产级：**$200 \sim 2000\text{ tokens/param}$** | 部署高频服务模型时，故意将小模型过训练数十倍，以预训练算力换取推理延迟与显存收益 |

## 6. 课后思考与自测练习（Review & Self-Check）

1. **算力最优分配拉格朗日闭式解推导（Assignment 3 核心必考）**：
   假设某新型架构的预训练验证集损失服从加性扩展定律：
   $$L(N, D) = E + \frac{A}{N^\alpha} + \frac{B}{D^\beta}$$
   其中不可约熵 $E = 1.60$，模型容量指数 $\alpha = 0.36$，数据样本指数 $\beta = 0.24$，系数 $A = 400, B = 600$。假设单步训练算力满足 $C = 6ND$。
   * (a) 请推导在给定算力预算 $C$ 下，算力最优参数量 $N_{\text{opt}}(C) \propto C^a$ 与算力最优数据量 $D_{\text{opt}}(C) \propto C^b$ 的缩放指数 $a$ 和 $b$ 的具体数值。
   * (b) 如果算力预算 $C$ 扩大 **$64$ 倍**（即 $2^6$ 倍），按照该定律，模型参数量 $N_{\text{opt}}$ 和训练数据量 $D_{\text{opt}}$ 应该分别扩大多少倍？此时最优超额损失 $\Delta L_{\text{opt}}(C) = L(N_{\text{opt}}, D_{\text{opt}}) - E$ 会缩小为原来的几分之几？
   * *参考解答*：
     * (a) 由第 4.1 节一阶导数平衡条件 $\alpha A N^{-\alpha} = \beta B D^{-\beta}$ 及 $D = \frac{C}{6N}$，可得：
       $$a = \frac{\beta}{\alpha + \beta} = \frac{0.24}{0.36 + 0.24} = \frac{0.24}{0.60} = \mathbf{0.40}, \qquad b = \frac{\alpha}{\alpha + \beta} = \frac{0.36}{0.60} = \mathbf{0.60}$$
     * (b) 当算力 $C$ 扩大 $64 = 2^6$ 倍时：
       * 最优模型参数量 $N_{\text{opt}}$ 应扩大 $64^{0.40} = (2^6)^{0.4} = 2^{2.4} \approx \mathbf{5.28\text{ 倍}}$；
       * 最优训练数据量 $D_{\text{opt}}$ 应扩大 $64^{0.60} = (2^6)^{0.6} = 2^{3.6} \approx \mathbf{12.13\text{ 倍}}$（验证：$5.28 \times 12.13 = 64$）。
       * 最优超额损失的缩放指数为 $-\frac{\alpha \beta}{\alpha + \beta} = -\frac{0.36 \times 0.24}{0.60} = -0.144$，因此超额损失 $\Delta L_{\text{opt}}$ 缩小为原来的 $64^{-0.144} = 2^{-0.864} \approx \mathbf{0.55\text{ 倍}}$（即下降约 $45\%$）。

2. **临界批大小（Critical Batch Size）与训练步数/样本数手算验证**：
   假设在训练一个 13B 模型逼近目标验证损失 $L_{\text{target}} = 2.1$ 时，测得极限最小步数 $S_{\min} = 10,000\text{ 步}$，极限最小消耗样本量为 $E_{\min} = 2 \times 10^{10}\text{ Tokens}$。
   * (a) 该目标损失下的临界批大小 $B_{\text{crit}}$（以每步 Tokens 数计）是多少？
   * (b) 请分别计算当全局 Batch Size 设为 $B_1 = 2\text{M Tokens}$（$2 \times 10^6$）和 $B_2 = 8\text{M Tokens}$（$8 \times 10^6$）时，达到 $L_{\text{target}}$ 所需的训练步数 $S(B)$ 和总消耗数据量 $E(B)$。为什么从 $2\text{M}$ 盲目扩大 4 倍到 $8\text{M}$ 是极不划算的？
   * *参考解答*：
     * (a) 根据定义 $B_{\text{crit}} = \frac{E_{\min}}{S_{\min}} = \frac{2 \times 10^{10}}{10,000} = \mathbf{2 \times 10^6\text{ Tokens}} = \mathbf{2\text{M Tokens}}$。
     * (b) 代入公式 $S(B) = S_{\min}\big(1 + \frac{B_{\text{crit}}}{B}\big)$ 与 $E(B) = E_{\min}\big(1 + \frac{B}{B_{\text{crit}}}\big)$：
       * 当 $B_1 = 2\text{M} = B_{\text{crit}}$ 时：$S(B_1) = 10,000 \times (1 + 1) = \mathbf{20,000\text{ 步}}$，总消耗数据 $E(B_1) = 2\times 10^{10} \times (1 + 1) = \mathbf{4 \times 10^{10}\text{ Tokens}}$。
       * 当 $B_2 = 8\text{M} = 4 B_{\text{crit}}$ 时：$S(B_2) = 10,000 \times \big(1 + \frac{1}{4}\big) = \mathbf{12,500\text{ 步}}$，而总消耗数据暴增至 $E(B_2) = 2\times 10^{10} \times (1 + 4) = \mathbf{10 \times 10^{10}\text{ Tokens}}$！
       * **工程结论**：将 Batch Size 从 $2\text{M}$ 强行扩大 4 倍到 $8\text{M}$，串行训练步数仅仅从 $20,000$ 步减少到 $12,500$ 步（仅节省了 $37.5\%$ 的步数），却导致**总训练算力与数据消耗从 $40\text{B}$ 暴增到了 $100\text{B}$（多烧了 $2.5$ 倍的算力钱）**，说明此时已严重落入偏差受限区（Bias-Limited Regime）！

3. **IsoFLOP 抛物线极值拟合与外推实战（直通 Assignment 3 编程题）**：
   假设你在算力预算 $C_0 = 6 \times 10^{18}\text{ FLOPs}$ 下跑了一组 IsoFLOP 实验，以 $\underline{x = \log_{10}(N)}$ 为自变量对终点验证集损失 $L$ 进行二次多项式回归，拟合得到抛物线方程：
   $$\hat{L}(x) = 0.15 x^2 - 2.40 x + 12.80$$
   * (a) 请计算在算力档位 $C_0 = 6 \times 10^{18}\text{ FLOPs}$ 下，算力最优的模型参数量 $N_{\text{opt}}(C_0)$、对应的最优训练数据量 $D_{\text{opt}}(C_0)$，以及预期能达到的最低验证集损失 $L_{\min}(C_0)$。
   * (b) 如果已知在该架构下 $N_{\text{opt}}(C) \propto C^{0.5}$，当你获得 $C_{\text{target}} = 6 \times 10^{22}\text{ FLOPs}$（比 $C_0$ 扩大 $10^4$ 倍）的正式训练预算时，你应该选择多大的模型参数量 $N_{\text{opt}}(C_{\text{target}})$ 和多少训练 Tokens $D_{\text{opt}}(C_{\text{target}})$？
   * *参考解答*：
     * (a) 对二次函数 $\hat{L}(x) = c_2 x^2 + c_1 x + c_0$ 求极小值点：
       $$x^* = -\frac{c_1}{2 c_2} = -\frac{-2.40}{2 \times 0.15} = \frac{2.40}{0.30} = \mathbf{8.0}$$
       因此最优参数量为 $N_{\text{opt}}(C_0) = 10^{8.0} = \mathbf{10^8\text{ (100M 参数)}}$；
       由 $C_0 = 6 N_{\text{opt}} D_{\text{opt}}$，对应最优数据量为 $D_{\text{opt}}(C_0) = \frac{6 \times 10^{18}}{6 \times 10^8} = \mathbf{10^{10}\text{ (10B Tokens)}}$；
       代入 $x^* = 8.0$ 得最低损失 $L_{\min}(C_0) = 0.15 \times 64 - 2.40 \times 8 + 12.80 = 9.60 - 19.20 + 12.80 = \mathbf{3.20}$。
     * (b) 当算力扩大 $\frac{C_{\text{target}}}{C_0} = 10^4$ 倍时，由于 $N_{\text{opt}} \propto C^{0.5}$ 且 $D_{\text{opt}} \propto C^{0.5}$：
       * 目标最优参数量 $N_{\text{opt}}(C_{\text{target}}) = 10^8 \times (10^4)^{0.5} = 10^8 \times 10^2 = \mathbf{10^{10}\text{ (10B 参数)}}$；
       * 目标最优数据量 $D_{\text{opt}}(C_{\text{target}}) = 10^{10} \times (10^4)^{0.5} = 10^{10} \times 10^2 = \mathbf{10^{12}\text{ (1T Tokens)}}$。
