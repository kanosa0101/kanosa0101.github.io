---
title: Stanford CS336 (Spring 2026) Lecture 3 深度课程笔记：Transformer Architectures & Hyperparameters
date: '2026-10-25 10:50:00'
categories:
  - 笔记
  - CS336
tags:
  - 大语言模型
  - CS336
  - Transformer
  - 架构设计
  - 超参数
  - RoPE
  - GQA
  - 训练稳定性
description: Stanford CS336 语言模型从零构建第三讲深度笔记：系统剖析现代 Transformer 五大结构演进、前馈网络门控机制（SwiGLU）、旋转位置编码（RoPE）数学推导、架构超参数黄金法则与数值稳定性护栏（QK-Norm、z-loss）。
mathjax: true
---

# Stanford CS336 (Spring 2026) Lecture 3 深度课程笔记：Transformer Architectures & Hyperparameters

> **课程主页**：[Stanford CS336: Language Modeling from Scratch (Spring 2026)](https://cs336.stanford.edu/)  
> **主讲教师**：Tatsunori Hashimoto (Tatsu)  
> **配套资源**：[Lecture 3 视频](https://www.youtube.com/watch?v=lVynu4bo1rY) · [官方幻灯片 (`lecture_03.pdf`)](https://github.com/stanford-cs336/lectures/blob/main/lecture_03.pdf) · [Assignment 1 指导手册与代码库](https://github.com/stanford-cs336/assignment1-basics)

---

## 1. 引言：如何科学地理解现代 Transformer 架构演进？

### 1.1 “Everything You Didn't Want to Know About Architectures and Hyperparameters”

Tatsu 将本讲副标题戏称为**“关于架构与超参数，那些你本不想知道的一切琐碎细节”**。做统计学习理论出身的研究者往往希望世界像 VC 维（VC Dimension）一样简洁优雅，但现代大语言模型的架构设计本质上是一场**多目标工程权衡（Complex Set of Trade-offs）**。一个合格的现代架构必须同时满足三大严苛要求：
1. **泛化与表达能力（Learn & Generalize）**：能高效捕捉海量文本中的长距离复杂依赖；
2. **硬件系统亲和性（Hardware Efficiency）**：在 GPU 上具备高算术强度，最大化算力利用率（MFU）并最小化推理显存带宽瓶颈；
3. **训练数值稳定性（Training Stability）**：在耗资数千万美元的训练中途绝不能因为梯度尖峰（Gradient Spikes）或数值溢出而崩溃且无法恢复。

由于我们没有算力去把所有架构组合从头网格搜索一遍，**仅次于“亲手训练每一个变体”的最佳学习方法，就是横跨整个现代开源/开放权重模型生态做一场全景式的实证普查（Empirical Survey Lens）**：通过观察数十个主流大模型（从早期的 GPT-3、PaLM、T5，到中期的 Llama 2/3、Mistral、Qwen 2、Gemma 2，再到 2025–2026 年最新的 **Qwen 3 / 3.5、Gemma 4、OLMo 3、Llama 4、Marin 8B/32B**），识别出**哪些设计已经成为全行业收敛的“铁律（Consensus）”**，而**哪些超参数存在宽容的平坦盆地（Forgiving Basin）可以自由调整**。

### 1.2 原始 Transformer vs. CS336 (Assignment 1) 现代基座架构

对比 [Vaswani et al. (2017) 的原始 Vanilla Transformer](https://arxiv.org/abs/1706.03762) 与你在 Assignment 1 中被要求从零实现的现代架构（也是 Llama 家族及几乎所有现代模型的标准起点），核心发生了以下五大结构性演变：

| 架构组件 | 原始 Vanilla Transformer (2017) | 现代主流标准架构 (Assignment 1 / Llama-style) | 演进的核心驱动力 |
| :--- | :--- | :--- | :--- |
| **1. 归一化层位置 (Norm Position)** | **Post-Norm**（置于残差主干内部，加完残差后再做 Norm） | **Pre-Norm**（移出残差流，置于 Attention 和 FFN 子层输入端；或采用 **Pre+Post Double Norm**） | 保持残差流干净（Clean Residual Stream），消除深层梯度衰减与爆炸，无需极度脆弱的 Warmup 即可稳定收敛 |
| **2. 归一化算子类型 (Norm Type)** | **LayerNorm**（减均值、除标准差，带可学习缩放 $\gamma$ 与偏置 $\beta$） | **[RMSNorm](https://arxiv.org/abs/1910.07467)**（省去减均值与偏置项 $\beta$，仅按均方根缩放） | 表达能力无损，但减少了访存密集型的均值统计开销，直接提升系统运行速度 |
| **3. 线性层与归一化偏置 (Bias Terms)** | 所有 `Linear` 投影层与 `LayerNorm` 均包含偏置项 $+b$ | **全面移除 Bias（No Bias）** | Bias 算术强度极低却占用显存带宽，且偶发破坏数值稳定性；砍掉后获得纯免费的效率与稳定性双赢 |
| **4. 激活函数与 FFN 结构 (FFN / Activation)** | 两矩阵单路 MLP + **ReLU** 激活（$\max(0, xW_1)W_2$） | 三矩阵门控线性单元 **[SwiGLU](https://arxiv.org/abs/2002.05202) / GeGLU**（$\big(\text{Swish}(xW_1) \odot xV\big)W_2$） | 在参数量对齐（隐藏维缩减为 $\frac{2}{3}$）的前提下，门控机制带来显著且一致的 Perplexity 下降 |
| **5. 位置编码 (Positional Encoding)** | 词嵌入层直接相加的**正弦绝对位置编码 (Sinusoidal)** | 注入每一层 Attention $Q, K$ 内积空间的**旋转位置编码 [RoPE](https://arxiv.org/abs/2104.09864)** | 严格满足纯相对位置不变性（无绝对位置交叉项），天然支持长上下文扩展 |

### 1.3 历史演进的四个阶段

回顾从 2017 到 2026 年的架构演进史，可以清晰地划分出四个时期：
1. **早期百花齐放探索期（2017–2022，Transformer $\rightarrow$ GPT-3 / PaLM / T5）**：各家在位置编码（Absolute、Sinusoidal、Relative Bias、ALiBi）、激活函数（ReLU、GeLU）与并行块上各显神通，尚无统一标准。
2. **Llama 2 大一统收敛期（2023–2024）**：Meta 发布 [Llama 2](https://arxiv.org/abs/2307.09288) 后，全行业发现 `Pre-RMSNorm + SwiGLU + RoPE + GQA` 配方极其稳健，几乎所有开源模型都变成了“Llama-alike”的微调变体。
3. **稳定性强化期（2024–2025）**：随着模型向更大规模与更高学习率推进，架构改进重心转向**极致稳定性干预**（如引入 **QK-Norm**、**Pre+Post 双重归一化**、**z-loss** 与 **Logit Soft-Capping**）。
4. **超长上下文与混合稀疏架构期（2025–2026）**：随着稠密小模型红利见顶，新发布的主力模型全面转向 **MoE（混合专家）** 与 **长上下文混合注意力架构**（如交替堆叠局部滑动窗口注意力 **SWA** 与全局注意力、引入 **NoPE** 或结合 **Gated DeltaNet / Mamba** 状态空间模型）。

## 2. 核心架构组件演进详解（Core Architecture Variations）

### 2.1 归一化层位置：Post-Norm vs. Pre-Norm vs. Double Norm

在原始 Transformer 的所有设计中，**全行业唯一达成 100% 共识认为原作者“做错了”的地方，就是把 LayerNorm 放在了残差主干道（Residual Stream）上**。

#### 1. 数学表达式与拓扑结构对比
设第 $l$ 层子模块（Attention 或 FFN）的非线性变换函数为 $F_l(\cdot)$，残差流状态为 $x_l$：

* **原始 Post-Norm（Residual Norm，位于残差主干内）**：
  $$x_{l+1} = \text{Norm}\big(x_l + F_l(x_l)\big)$$
  （注：现代唯一使用这种原始 Post-Norm 的异类是 `OPT-350M`，而 OPT 系列以训练崩溃和效果不佳闻名）。
* **现代主流 Pre-Norm（位于残差分支入口）**：
  $$x_{l+1} = x_l + F_l\big(\text{Norm}(x_l)\big)$$
  展开 $L$ 层递推公式可得：
  $$x_L = x_0 + \sum_{l=0}^{L-1} F_l\big(\text{Norm}(x_l)\big)$$
* **最新兴起的块外 Post-Norm / Pre+Post Double Norm（如 Grok、Gemma 2、OLMo 2、Qwen 3 部分变体）**：
  不仅在进入子模块前做一次 Pre-Norm，还在子模块输出汇入残差流**之前**（注意：依然在残差主干之外！）再加一次 Post-Norm：
  $$x_{l+1} = x_l + \text{Norm}_{\text{post}}\Big(F_l\big(\text{Norm}_{\text{pre}}(x_l)\big)\Big)$$

```
[原始 Post-Norm (In-Residual)]         [现代 Pre-Norm (Clean Residual)]        [现代 Double Norm (Gemma 2 / OLMo 2)]
        x_{l+1}                                    x_{l+1}                                   x_{l+1}
          ▲                                          ▲                                         ▲
   ┌──────┴──────┐                                   │                                         │
   │  LayerNorm  │                            ┌──────┴──────┐                           ┌──────┴──────┐
   └──────▲──────┘                            │      +      │                           │      +      │
          │                                   └──────▲──────┘                           └──────▲──────┘
   ┌──────┴──────┐                                   │                                         │
   │      +      │◄─────────┐                        ├──────────┐                              ├───────────────┐
   └──────▲──────┘          │                        │          │                              │        ┌──────┴──────┐
          │           ┌─────┴─────┐                  │    ┌─────┴─────┐                        │        │  Post-Norm  │
          │           │  Sublayer │                  │    │  Sublayer │                        │        └──────▲──────┘
          │           │  (MHA/FFN)│                  │    │  (MHA/FFN)│                        │        ┌──────┴──────┐
          │           └─────▲─────┘                  │    └─────▲─────┘                        │        │   Sublayer  │
          │                 │                        │    ┌─────┴─────┐                        │        └──────▲──────┘
          │                 │                        │    │  Pre-Norm │                        │        ┌──────┴──────┐
          │                 │                        │    └─────▲─────┘                        │        │   Pre-Norm  │
          └─────────────────┤                        └──────────┤                              │        └──────▲──────┘
                            │                                   │                              └───────────────┤
                           x_l                                 x_l                                            x_l
```

#### 2. 为什么必须把 Norm 移出残差流？（“Keep Your Residual Stream Clean” 原则）
早期研究者（如 [Xiong et al., 2020](https://arxiv.org/abs/2002.04745) 与 [Nguyen & Salazar, 2019](https://arxiv.org/abs/1910.05895)）探索 Pre-Norm 的初衷是为了去掉 Transformer 训练早期必需的 Learning Rate Warmup（学习率预热）。虽然现代训练依然保留了 Warmup，但他们发现了两个决定性的深层机制：
1. **无衰减的恒等梯度高速公路（Gradient Attenuation Issue）**：
   在 Pre-Norm 下，对式 $x_L = x_0 + \sum_{l=0}^{L-1} F_l(\text{Norm}(x_l))$ 求反向传播梯度：
   $$\frac{\partial \mathcal{L}}{\partial x_0} = \frac{\partial \mathcal{L}}{\partial x_L} \left( I + \sum_{l=0}^{L-1} \frac{\partial F_l(\text{Norm}(x_l))}{\partial x_0} \right)$$
   可以看到，顶层梯度 $\frac{\partial \mathcal{L}}{\partial x_L}$ 可以通过单位矩阵 $I$ **不经任何缩放阻拦地直达最底层 $x_0$**！在初始化阶段，Pre-Norm 模型从第 1 层到第 $L$ 层的梯度范数几乎保持水平恒定；而原始 Post-Norm 每经过一层都要除以该层残差的方差，导致越往底层梯度越剧烈衰减（底层学不动，顶层梯度大，必须靠极长且脆弱的 Warmup 续命）。
2. **大幅抑制训练中途的梯度尖峰（Gradient Spikes）**：
   实证表明，Pre-Norm 显著降低了训练过程中梯度范数突发尖峰的频率与幅度。而近年模型（如 Gemma 2 / OLMo 2）进一步采用 `Pre-Norm + Post-Norm` 双保险，是因为单纯 Pre-Norm 在极深层网络中主干残差方差会随层数线性累积增长，而在子层输出端汇入残差前再加一道 `Post-Norm` 既不破坏残差主干的清洁度，又能锁死每次写入残差流的增量幅度。

### 2.2 归一化算子与无偏置设计：LayerNorm $\rightarrow$ RMSNorm & No Bias

#### 1. LayerNorm vs. RMSNorm 的数学定义
给定维度为 $d$ 的激活向量 $x \in \mathbb{R}^d$：
* **标准 [LayerNorm (Ba et al., 2016)](https://arxiv.org/abs/1607.06450)**：先计算均值 $\mu = \frac{1}{d}\sum_{i=1}^d x_i$ 和方差 $\sigma^2 = \frac{1}{d}\sum_{i=1}^d (x_i - \mu)^2$，再做中心化、缩放与平移（含可学习参数 $\gamma, \beta \in \mathbb{R}^d$）：
  $$\text{LayerNorm}(x) = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} \odot \gamma + \beta$$
* **均方根归一化 [RMSNorm (Zhang & Sennrich, 2019)](https://arxiv.org/abs/1910.07467)**：假设均值中心化对表征能力并无实质贡献，直接省去减均值 $\mu$ 和加偏置 $\beta$，仅除以均方根 $\text{RMS}(x) = \sqrt{\frac{1}{d}\sum_{i=1}^d x_i^2}$：
  $$\text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{d}\|x\|_2^2 + \epsilon}} \odot \gamma$$

#### 2. 为什么占总 FLOPs 不到 0.2% 的 LayerNorm 值得被替换？（联系 Lecture 2 算术强度！）
这是将 Lecture 2 系统理论应用于架构设计的最佳范例：
* 根据 [Ivanov et al. (2021)《Data Movement Is All You Need》](https://arxiv.org/abs/2007.00072) 的性能剖析，在 Transformer 训练中，矩阵乘法（Tensor Contraction）占据了 **99.8% 以上的 FLOPs**，而以 LayerNorm 为代表的统计归一化仅占 **0.17% 的 FLOPs**。
* 然而，由于统计归一化需要在整个向量维度上做两趟归约（第一趟算均值 $\mu$，第二趟算方差 $\sigma^2$），其**算术强度极低（完全是 Memory-bound）**！在实际墙钟运行时间（Runtime）上，这 0.17% 的 FLOPs 竟然能消耗高达 **10%～25% 的总执行时间**！
* 切换为 **RMSNorm** 后，省去了均值计算与偏置加法，减少了约 30%～40% 的显存读写指令。在 [Narang et al. (2021) 的大规模消融实验](https://arxiv.org/abs/2102.11972) 中，RMSNorm 在每秒训练步数（Steps/sec）和最终验证集 Loss 上实现了双赢。

#### 3. 全面移除线性层偏置（Dropping Bias Terms）
顺着同样的逻辑，原始 Transformer 中所有线性投影层 $y = xW + b$ 都带有偏置向量 $b$。现代大模型（如 PaLM、Llama 系列、Qwen 系列）几乎一致将所有 Linear 层改为 `bias=False`（即 $y = xW$）：
* **系统视角**：加上偏置 $b$ 是一次低算术强度的逐元素广播操作，移除它可减少显存搬运与算子融合复杂度；
* **稳定性视角**：正如 PaLM 技术报告指出的，移除偏置项还能提升超大规模模型在半精度下的训练数值稳定性。

### 2.3 前馈网络（FFN）与激活函数演进：从 ReLU / GeLU 到门控线性单元（SwiGLU / GeGLU）

#### 1. 激活函数家族谱系
* **经典非门控激活**：
  * **ReLU**（原始 Transformer、OPT、早期 T5）：$\text{ReLU}(x) = \max(0, x)$，简单但在零点不可导且负半轴死区大。
  * **Squared ReLU**（$\max(0, x)^2$，被 NVIDIA Nemotron-4 340B 等少量模型采用）：增强稀疏性。
  * **GeLU**（GPT-2、GPT-3、BLOOM、Falcon）：$\text{GeLU}(x) = x \Phi(x) \approx 0.5x\big(1 + \tanh[\sqrt{2/\pi}(x + 0.044715x^3)]\big)$，在零点附近引入了一个平滑下凹（Divot），改善了零点附近的梯度流动。
  * **Swish / SiLU**：$\text{Swish}_\beta(x) = x \cdot \sigma(\beta x)$（通常取 $\beta = 1$，即 $x \cdot \text{sigmoid}(x)$）。

#### 2. 门控线性单元（GLU, Gated Linear Units）的数学机制
架构设计界有一条广为人知的经验法则：**“Gating is often very helpful（引入门控乘法调制往往能显著提升表征效果）”**。[Dauphin et al. (2017)](https://arxiv.org/abs/1612.08083) 提出了 GLU，随后 [Noam Shazeer (2020)](https://arxiv.org/abs/2002.05202) 将其引入 Transformer FFN：
* **标准非门控 FFN（2 个权重矩阵 $W_1 \in \mathbb{R}^{d_{\text{model}} \times d_{\text{ff}}}, W_2 \in \mathbb{R}^{d_{\text{ff}} \times d_{\text{model}}}$）**：
  $$\text{FFN}_{\text{ReLU}}(x) = \max(0, xW_1) W_2$$
* **引入门控分支 $xV$（新增第 3 个权重矩阵 $V \in \mathbb{R}^{d_{\text{model}} \times d_{\text{ff}}}$，逐元素哈达玛积 $\odot$）**：
  * **ReGLU**：$\text{FFN}_{\text{ReGLU}}(x) = \big(\max(0, xW_1) \odot xV\big) W_2$
  * **GeGLU**（Google 偏爱：T5 v1.1、Gemma 2 / 3 / 4）：
    $$\text{FFN}_{\text{GeGLU}}(x) = \big(\text{GeLU}(xW_1) \odot xV\big) W_2$$
  * **SwiGLU**（Llama 1/2/3/4、PaLM、Mistral、Qwen、DeepSeek、OLMo 等绝对主流）：
    $$\text{FFN}_{\text{SwiGLU}}(x) = \big(\text{Swish}_1(xW_1) \odot xV\big) W_2 = \big((xW_1 \cdot \sigma(xW_1)) \odot xV\big) W_2$$

#### 3. 至关重要的 $\frac{2}{3}$ 隐藏维度缩减法则（Parameter-Matched Comparison）
* **为什么要有 $\frac{2}{3}$ 因子？**
  标准 FFN 只有 $W_1, W_2$ 两个矩阵，参数量为 $2 \cdot d_{\text{model}} \cdot d_{\text{ff}}$（当 $d_{\text{ff}} = 4 d_{\text{model}}$ 时，参数量为 $8 d_{\text{model}}^2$）。
  而 SwiGLU / GeGLU 有 **$W_1, V, W_2$ 三个矩阵**，参数量为 $3 \cdot d_{\text{model}} \cdot d_{\text{ff}}$！
* 为了在保持单层总参数量与计算量（$8 d_{\text{model}}^2$）完全不变的前提下公平对比，必须将门控 FFN 的中间维度 $d_{\text{ff}}$ 乘以 **$\frac{2}{3}$**：
  $$d_{\text{ff}}^{\text{GLU}} = \frac{2}{3} \times (4 d_{\text{model}}) = \frac{8}{3} d_{\text{model}} \approx 2.67 d_{\text{model}}$$
* 在 [Shazeer (2020)](https://arxiv.org/abs/2002.05202) 与 [Narang et al. (2021)](https://arxiv.org/abs/2102.11972) 严格对齐参数量的多随机种子复现实验中，**SwiGLU 和 GeGLU 在所有基准上均显著且稳定地击败了非门控的 ReLU / GeLU**（至于 SwiGLU 与 GeGLU 之间则差异极小，选哪个都可以）。

### 2.4 串行块 vs. 并行块（Serial vs. Parallel Transformer Blocks）

* **传统串行结构（Serial Block）**：先算 Attention 并写回残差，再将更新后的残差送入 FFN：
  $$y = x + \text{Attention}(\text{Norm}_1(x)), \qquad z = y + \text{FFN}(\text{Norm}_2(y))$$
* **并行结构（Parallel Block，由 [GPT-J (2021)](https://github.com/kingoflolz/mesh-transformer-jax) 首创，后被 [PaLM](https://arxiv.org/abs/2204.02311)、Cohere Command 系列、Falcon 采用）**：
  让 Attention 和 FFN 同时读取同一个归一化后的输入，并行计算后一起加回残差流：
  $$z = x + \text{Attention}(\text{Norm}(x)) + \text{FFN}(\text{Norm}(x))$$
* **系统动机与为何近年失宠（Test of Time）**：
  * **并行块的优势**：只需做一次 LayerNorm，且可以将 Attention 的 $W_Q, W_K, W_V$ 矩阵与 FFN 的 $W_1, V$ 矩阵拼成一个超大矩阵做单次融合 MatMul，并在张量并行（Tensor Parallelism）时将每层的跨卡 `All-Reduce` 通信次数从 2 次减半为 1 次（PaLM 报告称在大规模下带来约 **15% 的训练加速**）。
  * **为何过去两年主流模型（包括 Google 自己的新模型）纷纷放弃并行块重回串行块？**
    第一，当前针对串行块的底层算子与通信重叠优化已经足够成熟；第二，将每层的“先看上下文（Attention）再做非线性推理（FFN）”改为并列执行，**本质上相当于把模型的有效复合深度（Effective Depth）砍掉了一半**，这在现代高知识密度的模型中会带来不可忽视的表达能力折损。

### 2.5 位置编码的终极演进：旋转位置编码（RoPE, Rotary Position Embedding）

#### 1. 四代位置编码机制对比
由于自注意力内积 $q_i^\top k_j$ 本质上是集合上的置换不变操作（Permutation Invariant，打乱词序结果不变），必须显式注入位置信息：
1. **正弦绝对位置编码（Sinusoidal，原始 Transformer）**：$e_i = x_i + p_i$，其中 $p_i$ 由不同频率的 $\sin/\cos$ 组成。
2. **可学习绝对位置编码（Learned Absolute，GPT-2 / GPT-3 / OPT）**：直接训练一个位置查找表矩阵 $P \in \mathbb{R}^{S_{\max} \times d}$，无法外推超过训练长度 $S_{\max}$ 的序列。
3. **注意力矩阵相对位置偏置（Relative Positional Bias / ALiBi，T5 / Chinchilla / BLOOM / Falcon）**：不在词向量上加位置，而是直接在注意力得分矩阵上加一个仅依赖于距离 $(i - j)$ 的标量偏置：$q_i^\top k_j + b(i - j)$。
4. **旋转位置编码（[RoPE, Su et al., 2021](https://arxiv.org/abs/2104.09864)，源自苏剑林博客与 GPT-J，现已成为 2024–2026 年几乎所有主流 LLM 的标配）**。

#### 2. RoPE 的核心数学推导与几何直觉
我们希望构建一种**纯正的相对位置编码函数** $f(x, i)$，使得两个处于绝对位置 $i$ 和 $j$ 的词向量经过编码后，它们的**内积（Inner Product）只依赖于两者的词义 $x, y$ 以及它们的相对距离 $(i - j)$**，而与绝对位置 $i, j$ 完全无关：
$$\langle f(x, i), f(y, j) \rangle = g(x, y, i - j)$$

* **为什么前三代方案都不满足这个等式？**
  * 正弦或绝对位置编码 $f(x, i) = x + p_i$ 在展开内积 $(x + p_i)^\top (y + p_j) = x^\top y + x^\top p_j + p_i^\top y + p_i^\top p_j$ 时，会产生依赖绝对位置的交叉干扰项（Cross terms）；
  * 而 T5/ALiBi 的加性偏置 $q_i^\top k_j + b(i-j)$ 根本无法分解为两个独立向量嵌入的内积形式 $\langle f(x, i), f(y, j) \rangle$。
* **RoPE 的天才几何直觉：欧氏空间内积对旋转不变！**
  在二维平面 $\mathbb{R}^2$ 中，如果我们将查询向量 $q$ 逆时针旋转角度 $i\theta$，将键向量 $k$ 逆时针旋转角度 $j\theta$，那么旋转后的两个向量的夹角余弦（以及内积）**只取决于它们的角度差 $(i - j)\theta$**！
  * **课堂直观例子**：考虑短语 `"we know"`：
    * 在句子 `"we know that"` 中，`"we"` 在位置 0（旋转 $0\theta$），`"know"` 在位置 1（旋转 $1\theta$），两者夹角差为 **$1\theta$**；
    * 在句子 `"of course we know"` 中，`"we"` 移到了位置 2（旋转 $2\theta$），`"know"` 移到了位置 3（旋转 $3\theta$），虽然绝对方向都转了，但它们的相对夹角依然严格等于 **$3\theta - 2\theta = 1\theta$**！
* **从 2D 推广到高维 $d$ 维空间（分块 2D 旋转）**：
  在 $d$ 维空间（假设 $d$ 为偶数）中，最简单的做法就是**将 $d$ 维向量两两配对，切分为 $d/2$ 个独立的二维子平面**，对第 $m \in \{1, \dots, d/2\}$ 对坐标应用旋转速度为 $\theta_m = \Theta^{-2(m-1)/d}$（通常基频 $\Theta = 10,000$ 或在长上下文模型中调大至 $500,000 \sim 1,000,000$）的 2D 旋转矩阵：
  * **高频分量（小 $m$，$\theta_m$ 大）**：转得极快，对相邻 Token 的局部微调极其敏感；
  * **低频分量（大 $m$，$\theta_m$ 小）**：转得极慢，负责捕捉跨数千 Token 的长距离宏观依赖。

$$R_{\Theta, i}^d = \begin{pmatrix}
\cos i\theta_1 & -\sin i\theta_1 & 0 & 0 & \cdots & 0 & 0 \\
\sin i\theta_1 & \cos i\theta_1 & 0 & 0 & \cdots & 0 & 0 \\
0 & 0 & \cos i\theta_2 & -\sin i\theta_2 & \cdots & 0 & 0 \\
0 & 0 & \sin i\theta_2 & \cos i\theta_2 & \cdots & 0 & 0 \\
\vdots & \vdots & \vdots & \vdots & \ddots & \vdots & \vdots \\
0 & 0 & 0 & 0 & \cdots & \cos i\theta_{d/2} & -\sin i\theta_{d/2} \\
0 & 0 & 0 & 0 & \cdots & \sin i\theta_{d/2} & \cos i\theta_{d/2}
\end{pmatrix}$$

#### 3. RoPE 的工程实现要点与最新变体（p-RoPE）
* **加在何处？** RoPE **不是**加在最底层的词嵌入上，而是在**每一层 Attention 内部**，计算出 $q_i = W_q x_i$ 和 $k_j = W_k x_j$ 之后，立即对 $q_i$ 和 $k_j$ 乘上旋转矩阵（注意：**不对 Value 向量 $v_j$ 做旋转**！），从而在每一层注意力计算时都强制执行相对位置不变性。
* **高效逐元素实现**：由于 $R_{\Theta, i}^d$ 是高度稀疏的分块对角矩阵，实际代码中绝不会构造稠密矩阵相乘，而是直接利用三角展开式完成 $O(d)$ 逐元素乘加：
  $$\begin{pmatrix} x_1 \\ x_2 \end{pmatrix} \mapsto \begin{pmatrix} x_1 \cos i\theta - x_2 \sin i\theta \\ x_1 \sin i\theta + x_2 \cos i\theta \end{pmatrix}$$
* **前沿变体：Proportional RoPE / Partial RoPE（如最新发布的 Gemma 4）**：
  Tatsu 提到，刚发布的 Gemma 4 使用了所谓的 **p-RoPE（Proportional RoPE）**，其核心思想是只对每个注意力头的一部分维度（甚至只对前几个坐标对）施加旋转，而其余维度保持不旋转（NoPE），因为极低频分量在短距离内几乎不转动，省去它们的旋转既节省计算又能改善长距离泛化。

## 3. 架构超参数黄金法则（Hyperparameter Rules of Thumb）

当你真正动手实例化一个 Transformer 时，有四个核心形状超参数需要确定。好消息是：**大规模消融实验表明，这些超参数在最优值附近都存在一个相当宽且平坦的盆地（Flat Basin）**，遵循行业共识默认值即可获得接近最优的表现。

### 3.1 前馈层扩展比（FFN Multiplier Ratio: $\frac{d_{\text{ff}}}{d_{\text{model}}}$）

* **非门控 MLP（ReLU / GeLU）的经典法则**：
  $$d_{\text{ff}} = 4 \, d_{\text{model}}$$
* **门控 GLU（SwiGLU / GeGLU）的现代法则**：
  * **标准 $\frac{2}{3}$ 修正派（$2.5 \sim 2.67 \times$）**：如前所述，为了与 $4d_{\text{model}}$ 的双矩阵 MLP 保持相同参数量，取 $d_{\text{ff}} = \frac{8}{3} d_{\text{model}} \approx 2.67 d_{\text{model}}$（并向上取整为 128 或 256 的倍数以对齐 GPU Tensor Core 内存边界，如 PaLM、Mistral、Qwen 等部分规模）。
  * **Llama 2 放大派（$\approx 3.5 \times$）**：Llama 2 团队因为使用了高效的 GQA 省下了注意力层参数，决定把省下的参数预算补贴给 MLP，额外乘了一个 $\sim 1.3$ 的系数，将 SwiGLU 的比例定在了 **$d_{\text{ff}} \approx 3.5 \, d_{\text{model}}$**。
* **Kaplan et al. (2020) 盆地实证与 T5 的“64 倍”搞笑反转轶事**：
  * [Kaplan et al. (2020)](https://arxiv.org/abs/2001.08361) 的扫描曲线显示，只要 $\frac{d_{\text{ff}}}{d_{\text{model}}}$ 落在 **$[1, 10]$ 的区间内**，验证集 Loss 曲线都非常平坦；一旦超过 10，Loss 才会呈二次方急剧恶化。
  * **T5 的轶事**：Google 在初代 [T5 (v1.0)](https://arxiv.org/abs/1910.10683) 中极其激进地选择了 **$d_{\text{ff}} = 64 \, d_{\text{model}}$**（理由是矩阵越大，TPU 算术强度越高）。结果在随后的改进版 **T5 v1.1** 中，Google 悄悄把这个比例改回了标准的 **$2.5 \, d_{\text{model}}$**！

### 3.2 注意力头维度配比（Head Dimension Ratio: $\frac{H \times d_{\text{head}}}{d_{\text{model}}}$）

* **共识铁律**：几乎所有主流模型都严格保持 **多头总维度等于模型隐藏维度**，即：
  $$H \times d_{\text{head}} = d_{\text{model}} \iff \frac{H \times d_{\text{head}}}{d_{\text{model}}} = 1$$
  且单个注意力头的维度 $d_{\text{head}}$ 通常固定在 **$64$ 或 $128$**（偶尔在超大模型中取 $256$），当模型变宽（$d_{\text{model}}$ 增大）时，保持 $d_{\text{head}} = 128$ 不变，按比例增加头数 $H$。（唯一的早期例外又是初代 T5 和 LaMDA，它们曾尝试过偏离 1 的比例，但后续模型基本回归了 1:1）。

### 3.3 深宽比（Aspect Ratio: $\frac{d_{\text{model}}}{n_{\text{layers}}}$）与并行系统的深层联系

当我们把模型从 1B 放大到 7B、70B 时，应该让它变得“更深（增加层数 $n_{\text{layers}}$）”还是“更宽（增加隐藏维度 $d_{\text{model}}$）”？
* **黄金甜点区间**：统计几乎所有主流模型（GPT-3、Llama 1/2/3、Mistral、Qwen、DeepSeek、OLMo），它们的深宽比 $\frac{d_{\text{model}}}{n_{\text{layers}}}$ 惊人地集中在 **$100$ 左右**（大致分布在 **$60 \sim 180$** 的窄带内）！
* **为什么不能太深，也不能太宽？（Expressivity vs. Systems Parallelism）**：
  * **从表达能力看**：网络需要足够的深度（Sequential Depth）来执行多步组合推理。
  * **从分布式系统并行看（核心工程约束！）**：
    * **极宽的模型（Large $d_{\text{model}}$）非常容易并行**：可以使用**张量并行（Tensor Parallelism, TP）**，直接把单层内巨大的权重矩阵切分到同一台机器的 8 张 GPU 上，通过高速 NVLink 同步。
    * **极深的模型（Large $n_{\text{layers}}$）是系统工程师的噩梦**：因为层与层之间是严格串行依赖的，如果层数太多装不进单卡，就必须跨多节点切分不同的层——即**流水线并行（Pipeline Parallelism, PP）**。流水线并行会引入严重的流水线气泡（Pipeline Bubbles）与复杂的微批次调度难题。
* **实证结论**：[Kaplan et al. (2020)](https://arxiv.org/abs/2001.08361) 与 [Tay et al. (2021)《Scale Efficiently》](https://arxiv.org/abs/2109.10686) 的实验表明，在固定总 FLOPs 预算下，深宽比在 $50 \sim 200$ 之间对最终预训练 Loss 影响很小（主要由总 FLOPs 决定），因此工程上优先选择 $\approx 100$ 以兼顾张量并行效率。

### 3.4 词表大小（Vocabulary Size $|V|$）的两极分化

普查各模型的词表规模，可以清晰看到两个世代的断层：
* **早期单语（英文）模型时代（30K～50K）**：GPT-2/3 ($50,257$)、Llama 1/2 ($32,000$)、Mistral 7B ($32,000$)、OLMo 1 ($50,304$)。仅针对英文优化，词表小巧。
* **现代多语言与生产级大模型时代（100K～256K）**：GPT-4 `cl100k` ($100,277$)、GPT-4o `o200k` ($200,000$)、Llama 3 ($128,256$)、Qwen 2/3 ($\approx 151,936$)、DeepSeek V2/V3 ($102,400 \sim 129,280$)、Gemma 2/3/4 ($256,000$)。
  * **原因**：第一，为了支持中日韩及多语言代码的高压缩率（减少 Token 数从而降低推理开销）；第二，Scaling Law 研究表明，模型参数量越大，其最优词表规模也越大。

#### 课堂深度问答：不同词表大小的模型之间，如何公平对比语言建模能力（Bits-Per-Byte, BPB）？
* **学生提问**：如果两个模型使用了不同词表大小的 Tokenizer，直接对比它们的 Perplexity（困惑度）是不公平的（词表越大的模型切出的 Token 越少，单 Token 预测越难，Perplexity 天然偏高）。那么对比 **Bits-Per-Byte (BPB)** 是否完全合法？
* **Tatsu 解答**：**完全合法且是标准做法！** 语言模型的本质是对完整原始文本字符串 $s$ 赋予联合概率 $P(s)$。只要两个分词器都是**无损且完备的（Complete，不会像早期 word-level 分词器那样把未知词丢弃为 `<UNK>`）**，它们都在同一个字符串空间上定义了合法的概率分布。由于原始文本的 UTF-8 总字节数 $N_{\text{bytes}}$ 与分词器无关，将总负对数似然（以 2 为底）除以固定常数 $N_{\text{bytes}}$：
  $$\text{BPB}(s) = \frac{-\sum_{t=1}^{T_{\text{tokens}}} \log_2 P(x_t \mid x_{<t})}{N_{\text{bytes}}(s)}$$
  就消除了 Token 切分长度的影响，实现了跨任意分词器的绝对公平对比！

### 3.5 正则化之谜：为什么单遍训练（Single-Pass）不需要防过拟合，却依然要用 Weight Decay？

* **对传统 ML 101 直觉的挑战**：
  在大模型预训练中，语料库（数万亿 Tokens）极其庞大，我们通常只对数据做 **Single-pass（1 个 Epoch，每个样本只看一次）** 训练。在单遍随机梯度下降中，训练 Loss 与验证 Loss 几乎完全重合在 $y = x$ 对角线上，**根本不存在传统意义上的过拟合（Overfitting）**！因此，现代大模型预训练已经**基本废弃了 Dropout**（设置 `dropout = 0.0`）。
* **反直觉的现象：那为什么几乎所有现代模型依然使用高达 $0.1$ 的 Weight Decay（权重衰减）？**
  根据 [Andriushchenko et al. (2023)《Why Do We Need Weight Decay in Modern Deep Learning?》](https://arxiv.org/abs/2310.04415) 的研究：
  * 在大模型训练中，**Weight Decay 并不是作为正则化器（Regularizer）来缩小泛化误差的，而是作为优化促进器（Optimization Enhancer）与学习率调度协同工作的**！
  * 实验表明，在不同的 Weight Decay 下，Training Loss 和 Validation Loss 依然完全相等（说明它没改变泛化间隙）；但在配合学习率衰减（Cosine / WSD LR Decay）时，**较大的 Weight Decay 能控制参数范数不无限膨胀，使有效学习率保持合理水平，从而在训练后期收敛到显著更低的 Loss 极小值**！（这也解释了课堂问答中提到的技巧：许多工程实践会让 Weight Decay 与学习率同步动态调整）。

## 4. 训练数值稳定性干预技巧（Training Stability Tricks）

随着模型规模扩大，训练中途突然爆发 Loss 尖峰（Loss Spikes）甚至 `NaN` 崩溃是头号工程威胁。Tatsu 指出了导致神经网络数值崩溃的头号罪魁祸首——**Softmax 算子**（因为它同时包含了极易爆炸的**指数函数 $\exp(x)$** 和极易下溢除零的**除法运算**）。

在 Transformer 中，共有**两处 Softmax 危险区**：**输出层词表 Softmax（Output Softmax）** 与 **注意力层权重 Softmax（Attention Softmax）**。工业界针对这两处分别发展出了三大防爆武器：

### 4.1 武器一：针对输出层 Softmax 的 `z-loss` 正则项

在模型最后一层计算交叉熵损失时，设未归一化的输出 Logits 向量为 $u \in \mathbb{R}^{|V|}$，目标词 $y$ 的对数概率为：
$$\log P(y \mid x) = \log \left( \frac{e^{u_y}}{\sum_{k=1}^{|V|} e^{u_k}} \right) = u_y - \log Z, \qquad \text{其中配分函数 } Z = \sum_{k=1}^{|V|} e^{u_k}$$

* **不稳定的根源**：因为 Softmax 具有平移不变性（给所有 $u_k$ 同加一个常数 $c$，输出的概率分布完全不变），模型在训练过程中可能会让 Logits 整体漂移到极大或极小值，导致配分函数 $Z = \sum e^{u_k}$ 在半精度下直接上溢为 `Inf` 或下溢为 `0`！
* **`z-loss` 解决方案（最早由 [Devlin et al., 2014](https://aclanthology.org/P14-1129/) 在机器翻译中提出，后被 PaLM、Baichuan、OLMo、DCLM 广泛采用）**：
  既然我们希望配分函数 $Z$ 乖乖待在 $1$ 附近（即希望 $\log Z$ 待在 $0$ 附近），只需在主交叉熵损失上额外加上一项惩罚 $(\log Z)^2$ 的辅助损失：
  $$\mathcal{L}_{\text{total}} = -\log P(y \mid x) + \alpha \cdot (\log Z)^2$$
  （通常系数取 $\alpha \approx 10^{-4}$）。这相当于利用 Softmax 的冗余自由度，在不影响分类概率表达的前提下，把输出 Logits 的整体水位牢牢锚定在零点附近！

### 4.2 武器二：针对注意力层 Softmax 的 `QK-Norm`（查询-键归一化）

在每一层自注意力中，注意力矩阵由 $A = \text{Softmax}\left(\frac{Q K^\top}{\sqrt{d_k}}\right)$ 计算得出。随着训练进行，$W_Q, W_K$ 的权重范数可能不断增长，导致内积 $Q K^\top$ 变得极大，使 Attention Softmax 陷入极度尖锐的饱和状态（Attention Entropy Collapse，梯度消失或爆发尖峰）。

* **核心哲学：“哪里不稳定，就在哪里塞一个 LayerNorm / RMSNorm！”**
* **`QK-Norm` 的做法（源自 [Dehghani et al., 2023 (ViT-22B)](https://arxiv.org/abs/2302.05442) 及多模态模型 IDEFICS / Chameleon，现已成为 OLMo 2/3、Gemma 2/3/4、Qwen 3 等现代 LLM 的标配）**：
  在计算出查询向量 $Q = X W_Q$ 和键向量 $K = X W_K$ 之后、**在它们做点积 $Q K^\top$ 送入 Softmax 之前，分别对每个 Head 的 $Q$ 和 $K$ 独立施加一次 `RMSNorm`（或 `LayerNorm`）**：
  $$Q' = \text{RMSNorm}(Q), \qquad K' = \text{RMSNorm}(K), \qquad \text{Attention} = \text{Softmax}\left(\frac{\text{RoPE}(Q') \, \text{RoPE}(K')^\top}{\sqrt{d_k}}\right) V$$
* **效果**：通过强制将 $Q'$ 和 $K'$ 的每个维度均方根限制在 $O(1)$ 量级，从根本上杜绝了注意力得分无界膨胀的可能性，使得模型能够安全地使用显著更高的峰值学习率进行训练。

### 4.3 武器三：针对 Logits 的硬截断 `Logit Soft-Capping`（Gemma 2 / 3 / 4 & Grok）

除了在输入端做 QK-Norm，Google（在 Gemma 2 / 3 / 4 中）和 xAI（在 Grok 中）还采用了另一种更强硬的手段——直接用有界的双曲正切函数 $\tanh$ 将送入 Softmax 前的 Logits $z$ 软截断在 $[-C, +C]$ 范围内（例如在 Attention 层取 $C = 50.0$，在最终输出层取 $C = 30.0$）：
$$z_{\text{capped}} = C \cdot \tanh\left(\frac{z}{C}\right)$$
* **权衡与代价**：根据 NVIDIA 团队的系统性消融研究（[Wortsman et al., 2023](https://arxiv.org/abs/2309.14322)），**`QK-Norm` 几乎没有任何性能副作用且能提升高学习率下的表现**；而单独使用 **`Logit Soft-Capping` 会造成轻微的模型质量下降**（因为它人为限制了模型输出极高置信度尖锐分布的能力）。因此，目前社区更推崇 `QK-Norm` + `z-loss` 的组合。

## 5. 注意力层现代优化：KV Cache 瓶颈、GQA 与滑动窗口混合架构

本讲最后一部分聚焦于 Dense Attention 内部的工程改良（更激进的线性注意力与 MoE 将在 Lecture 4 展开）。

### 5.1 从自回归推理算术强度推导 MQA 与 GQA（Grouped-Query Attention）

让我们用 Lecture 2 的算术强度工具，严格算一笔训练/Prefill 与自回归推理（Decode）的注意力账（设批大小 $B$，序列长度 $n$，隐藏总维度 $d$，注意力头数 $h$，每头维度 $k = d/h$）：

#### 1. 训练 / Prefill 阶段（并行处理长为 $n$ 的整条序列）
* **总浮点计算量（Total FLOPs）**：$\mathcal{O}(B n d^2 + B n^2 d)$
* **总显存访问量（Memory Accesses，读写激活与权重）**：$\mathcal{O}(B n d + B h n^2 + d^2)$（若用 FlashAttention 则消去 $B h n^2$ 项）
* **算术强度（Arithmetic Intensity）**：
  $$I_{\text{train}} = \mathcal{O}\left( \left(\frac{1}{k} + \frac{1}{B n}\right)^{-1} \right)$$
  只要单头维度 $k$ 足够大（如 64/128）且总 Token 数 $B n$ 足够大，算术强度就极高，GPU 处于完美的 **Compute-bound** 状态。

#### 2. 自回归生成 Decode 阶段（使用 KV Cache 逐个生成 $n$ 个 Token）
在生成第 $t$ 个 Token 时，为了避免重算前 $t-1$ 个位置，我们将历史所有的 Key 和 Value 缓存在显存中（即 **KV Cache**，总大小为 $B \times n \times d$）。然而，因为每次只能串行生成 1 个 Token，每走一步都必须把庞大的 KV Cache 和投影权重从 HBM 重新读入一次：
* **生成 $n$ 个 Token 的累计计算量**：依然是 $\mathcal{O}(B n d^2 + B n^2 d)$
* **生成 $n$ 个 Token 的累计显存访问量**：激增至 $\mathcal{O}(B n^2 d + n d^2)$！（注意第一项是每步读取不断增长的 KV Cache，累计为 $B n^2 d$；第二项是每步读取权重矩阵，累计为 $n d^2$）。
* **算术强度（Arithmetic Intensity）**：用总计算量除以总访存量（关注 KV Cache 主导项）：
  $$I_{\text{decode}}^{\text{MHA}} = \mathcal{O}\left( \left(\frac{n}{d} + \frac{1}{B}\right)^{-1} \right)$$
  **致命瓶颈浮现**：当上下文长度 $n$ 很长（导致 $\frac{n}{d}$ 很大）或者推理批大小 $B$ 很小时，算术强度趋近于 $O(1)$，推理完全被 **KV Cache 的显存带宽（Memory Bandwidth）** 卡死！

#### 3. MQA 与 GQA 如何拯救推理算术强度？
为了砍掉读取 KV Cache 的显存访问量 $B n^2 d$：
* **多查询注意力 [MQA (Multi-Query Attention, Shazeer, 2019)](https://arxiv.org/abs/1911.02150)**：
  保留 $h$ 个独立的 Query 头，但**让所有 $h$ 个 Query 头共享唯一的一组 Key 头和 Value 头（即 KV 头数 $g = 1$）**！
  这样一来，KV Cache 的体积和每步读取带宽直接缩减为原来的 **$\frac{1}{h}$**！算术强度中的瓶颈项从 $\frac{n}{d}$ 变成了 **$\frac{n}{d \cdot h}$**，推理吞吐率获得数量级跃升。但 MQA 过于激进，共享单个 KV 头会导致显著的表达能力与下游任务精度下降。
* **分组查询注意力 [GQA (Grouped-Query Attention, Ainslie et al., 2023)](https://arxiv.org/abs/2305.13245)**：
  在 MHA（$h$ 个 KV 头）和 MQA（$1$ 个 KV 头）之间取折中——将 $h$ 个 Query 头分为 $g$ 个组（通常 $g = 4$ 或 $8$，即每 $h/g$ 个 Query 头共享 1 个 Key 头和 1 个 Value 头）。
  * **帕累托最优（Best of Both Worlds）**：实验表明，当把 KV 头数缩减为 Query 头数的 $1/4 \sim 1/8$ 时，**模型的下游精度与完整 MHA 几乎毫无差别，但同时拿到了接近 MQA 的推理延迟与 KV Cache 显存削减红利**！因此，**GQA 已经成为当今所有主流大模型训练时的绝对标配**。

```
[多头注意力 MHA (g = h)]           [分组查询注意力 GQA (1 < g < h)]        [多查询注意力 MQA (g = 1)]
Q1  Q2  Q3  Q4  Q5  Q6  Q7  Q8     Q1  Q2  Q3  Q4  Q5  Q6  Q7  Q8       Q1  Q2  Q3  Q4  Q5  Q6  Q7  Q8
│   │   │   │   │   │   │   │      └─┬─┘   └─┬─┘   └─┬─┘   └─┬─┘        └───┴───┴───┴───┬───┴───┴───┘
▼   ▼   ▼   ▼   ▼   ▼   ▼   ▼        ▼       ▼       ▼       ▼                          ▼
K1  K2  K3  K4  K5  K6  K7  K8       K1      K2      K3      K4                         K1
V1  V2  V3  V4  V5  V6  V7  V8       V1      V2      V3      V4                         V1
(每个 Query 独占一组 KV，显存大)     (每组 Query 共享一组 KV，黄金折中)      (所有 Query 共享单组 KV，极省显存)
```

### 5.2 超长上下文利器：滑动窗口注意力（SWA）与混合交替架构（Interleaved / Hybrid Layers）

即便有了 GQA，当上下文长度 $n$ 扩展到 128K 甚至 1M Tokens 时，全注意力（Full Attention）的 $O(n^2)$ 计算量与线性增长的 KV Cache 依然难以承受。

* **滑动窗口注意力（Sliding Window Attention, SWA）**：
  最早在 [Sparse Transformer](https://arxiv.org/abs/1904.10509) 与 GPT-3 中被探索，后由 Mistral 7B 推广：限制每个位置 $i$ 只关注其前方固定窗口大小 $W$（如 $W = 4096$）以内的局部上下文（即带状注意力矩阵 Banded Matrix）。单层复杂度从 $O(n^2)$ 降至 $O(n W)$，且该层的 KV Cache 只需要维护一个长度为 $W$ 的滚动环形缓冲区（Rolling Buffer）！
* **2025–2026 最前沿趋势：全局与局部交替混合架构（Interleaved Full + Local / SSM Attention）**：
  如果全网每一层都只用局部滑动窗口，模型虽然能通过层层堆叠传递远距离信息，但全局精确检索（如“大海捞针 Needle-in-a-Haystack”）能力会严重受损。因此，过去一年几乎所有冲击超长上下文的前沿模型都收敛到了 **“每隔 $K$ 层保留 1 层全局 Full Attention，其余层使用廉价局部层”** 的交替混合范式：
  1. **Cohere Command A / Gemma 3 / Gemma 4 / Llama 4（SWA + Full Attention 交替）**：
     例如采用 **3:1 或 5:1 的比例**——连续 3 层使用局部的 **Sliding Window Attention（配合 RoPE）** 处理局部语法结构，第 4 层使用全局 **Full Attention** 汇总全篇长距离信息。
  2. **NoPE（No Positional Embedding in Global Layers）技巧**：
     部分最新架构（如 Llama 4 Scout / Cohere 等变体）在局部滑动窗口层使用 RoPE，而在那每隔几层出现一次的**全局 Full Attention 层中彻底去掉位置编码（NoPE）**！这使得全局层像词袋或纯内容寻址记忆一样工作，不仅省去了长距离 RoPE 频率外推的衰减问题，还大幅提升了超长上下文长度泛化能力。
  3. **Qwen 3.5 混合架构（Gated DeltaNet SSM + Full Attention 交替）**：
     与使用滑动窗口类似，最新的 Qwen 3.5 采用 **3 层线性复杂度的状态空间模型（Gated DeltaNet）+ 1 层全局 Full Attention** 的 3:1 交替结构（关于 Gated DeltaNet、Mamba 与 DeepSeek MLA 的内部机制，将在 **Lecture 4** 中全面拆解）。

## 6. 全景总结：现代 Transformer 架构设计速查表（Cheat Sheet）

如果今天你要从零启动训练一个新的旗舰基座语言模型，以下是经过全行业千锤百炼验证的**黄金默认配置单**：

| 决策维度 | 黄金推荐配置 (2026 Consensus Default) | 代表模型案例 | 核心工程理由 |
| :--- | :--- | :--- | :--- |
| **归一化位置** | **Pre-Norm**（或 **Pre + Post Double Norm**） | Llama 3 (Pre), Gemma 2/OLMo 2 (Double) | 保持残差主干清洁，确保深层梯度恒等直传并抑制尖峰 |
| **归一化类型** | **RMSNorm** (`eps=1e-5` 或 `1e-6`) | 几乎所有 2023+ 模型 | 省去均值计算与偏置，减少低算术强度的 HBM 访存开销 |
| **线性层偏置** | **`bias=False`**（全面移除偏置） | PaLM, Llama 1–4, Gemma, OLMo | 提升算术强度、简化算子融合并增强半精度稳定性 |
| **前馈网络激活** | **SwiGLU**（或 **GeGLU**） | Llama/Qwen/DeepSeek (SwiGLU), Gemma (GeGLU) | 门控乘法调制在同参数量下稳定降低验证集 Loss |
| **块内拓扑** | **串行块 (Serial Block: MHA $\rightarrow$ FFN)** | Llama 1–4, Qwen, DeepSeek, Gemma | 保留完整的非线性复合深度，优于折损深度的并行块 |
| **位置编码** | **RoPE**（局部层 RoPE，全局层可选 NoPE/高基频） | Llama 1–4, Qwen, DeepSeek, Gemma | 严格满足内积相对位置不变性，无交叉污染项 |
| **FFN 维度比** | $d_{\text{ff}} \approx \frac{8}{3} d_{\text{model}}$ ($\sim 2.67\times$) 或 $3.5 d_{\text{model}}$ | PaLM/Mistral ($2.67\times$), Llama 2/3 ($3.5\times$) | 落在 $[1, 10]$ 平坦最优盆地内，并向上对齐 128/256 倍数 |
| **深宽比** | $\frac{d_{\text{model}}}{n_{\text{layers}}} \approx 100$ ($60 \sim 180$) | GPT-3 ($128$), Llama-3 70B ($102.4$) | 平衡表达深度与张量并行（TP）/流水线并行（PP）开销 |
| **单头维度** | $d_{\text{head}} = 128$（满足 $H \times d_{\text{head}} = d_{\text{model}}$） | Llama 2/3, Qwen, DeepSeek | 契合 FlashAttention 与 GPU Tensor Core 硬件分块尺寸 |
| **词表大小** | 多语言/生产级：**$100\text{K} \sim 256\text{K}$** | Llama 3 ($128\text{K}$), Qwen ($152\text{K}$), Gemma ($256\text{K}$) | 提高多语言与代码压缩率，缩短有效序列长度 $n$ |
| **正则化** | **`dropout = 0.0`**, **`weight_decay = 0.1`** | 几乎所有现代预训练模型 | 单遍训练不过拟合故无需 Dropout；Weight Decay 辅助优化器后期收敛 |
| **稳定性护栏** | **`QK-Norm` + 输出层 `z-loss`** | OLMo 2/3, Gemma 2/3/4, Qwen 3 | 锁死两个 Softmax 危险区的输入尺度与配分函数漂移 |
| **注意力加速** | **GQA** + **SWA/Full Attention 混合交替** | Llama 3/4, Gemma 3/4, Command A | GQA 将 KV Cache 砍至 $1/4 \sim 1/8$；混合窗口突破超长上下文瓶颈 |

## 7. 课后思考与自测练习（Review & Self-Check）

1. **SwiGLU 参数量对齐推导（Assignment 1 核心考点）**：
   假设模型隐藏层维度 $d_{\text{model}} = 4096$。
   * (a) 如果使用传统的双矩阵 ReLU FFN 且取 $d_{\text{ff}} = 4 d_{\text{model}} = 16384$（不含偏置），单层 FFN 共有多少个参数？
   * (b) 如果改用包含三个矩阵 $(W_1, V, W_2)$ 的 SwiGLU FFN，为了使参数量与 (a) 完全相等，理论上的 $d_{\text{ff}}$ 应该取多少？如果为了硬件对齐要求 $d_{\text{ff}}$ 必须是 $256$ 的整数倍，应该取多少？
   * *参考解答*：
     * (a) 参数量为 $2 \times 4096 \times 16384 = 134,217,728$ ($\approx 134.2\text{M}$ 参数)。
     * (b) 令 $3 \times 4096 \times d_{\text{ff}} = 2 \times 4096 \times 16384 \implies d_{\text{ff}} = \frac{2}{3} \times 16384 = 10922.67$。向上/就近取 $256$ 的整数倍：$\text{round}(10922.67 / 256) \times 256 = 43 \times 256 = 11008$（这正是 Llama-1 7B 中 `intermediate_size = 11008` 的精确由来！）。
2. **RoPE 相对位置不变性手算验证**：
   设二维查询向量 $q = [q_1, q_2]^\top$ 在位置 $i$，二维键向量 $k = [k_1, k_2]^\top$ 在位置 $j$。经 RoPE 旋转后分别为 $q' = R(i\theta) q$ 和 $k' = R(j\theta) k$，其中旋转矩阵 $R(\alpha) = \begin{pmatrix} \cos\alpha & -\sin\alpha \\ \sin\alpha & \cos\alpha \end{pmatrix}$。请用矩阵代数证明内积 $(q')^\top k'$ 仅依赖于相对位置差 $(j - i)$。
   * *参考证明*：
     $$(q')^\top k' = (R(i\theta) q)^\top (R(j\theta) k) = q^\top R(i\theta)^\top R(j\theta) k$$
     利用正交旋转矩阵的性质 $R(\alpha)^\top = R(-\alpha)$ 以及旋转群的加法结合律 $R(\alpha) R(\beta) = R(\alpha + \beta)$：
     $$q^\top R(-i\theta) R(j\theta) k = q^\top R\big((j - i)\theta\big) k$$
     证毕！内积严格只包含相对距离 $(j - i)$，不含任何绝对位置 $i$ 或 $j$ 的独立项。
3. **GQA 显存带宽精算**：
   对于一个 $L = 32$ 层、$d_{\text{model}} = 4096$、Query 头数 $h = 32$（每头维度 $d_{\text{head}} = 128$）的模型，在序列长度 $n = 8192$、Batch Size $B = 1$、精度为 `bf16`（2 Bytes）下进行自回归推理：
   * 若使用标准 MHA（$g = 32$），整個模型的 KV Cache 占用多少显存？
   * 若改用 $g = 8$ 的 GQA（如 Llama-3 8B），KV Cache 显存降为多少？
   * *参考解答*：
     * 标准 MHA 下，每层需存 K 和 V 两个张量，总字节数为 $2 \times L \times B \times n \times (h \times d_{\text{head}}) \times 2\text{ Bytes} = 2 \times 32 \times 1 \times 8192 \times 4096 \times 2 = 4,294,967,296\text{ Bytes} \approx 4.0\text{ GB}$。
     * 改用 $g = 8$ 的 GQA 后，KV 头数缩减为原来的 $8 / 32 = 1/4$，因此 KV Cache 显存直接从 $4.0\text{ GB}$ 降至 $1.0\text{ GB}$，每步 Decode 读取 KV Cache 的耗时也同步降低 4 倍！
