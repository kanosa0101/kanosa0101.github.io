---
title: Stanford CS336 (Spring 2026) Lecture 8 深度课程笔记：Parallelism II (ZeRO, Megatron & 3D Parallelism)
date: 2026-10-25 16:24:00
tags:
  - 大模型
  - 分布式训练
  - ZeRO
  - FSDP
  - 流水线并行
  - 序列并行
  - 专家并行
  - CS336
categories:
  - [笔记, CS336]
description: 深入剖析 Stanford CS336 Lecture 8，全景解析 4D/5D 混合并行系统设计：超算互连网络哲学（Fat-Tree vs 3D Torus）、静态显存 16B 账单与 ZeRO 1/2/3 / FSDP 零冗余分片、1F1B 与零气泡流水线调度、Megatron-SP 序列并行以及 Llama 3 405B / DeepSeek-V3 工业级万卡混合并行实战架构。
mathjax: true
---

# Stanford CS336 (Spring 2026) Lecture 8 深度课程笔记：Parallelism II (ZeRO, Megatron & 3D Parallelism)

**课程信息**：[Stanford CS336: Language Modeling from Scratch (Spring 2026)](https://cs336.stanford.edu/)  
**主讲人**：Tatsunori Hashimoto (Tatsu)  
**课程视频与讲义**：[Lecture 8 YouTube 视频](https://www.youtube.com/watch?v=6-cXp-aOmdg) | [官方幻灯片 (`lecture_08.pdf`)](https://github.com/stanford-cs336/lectures/blob/main/lecture_08.pdf) | [配套参考：Google《How to Scale Your Model》](https://jax-ml.github.io/scaling-book/)  
**关联作业**：Assignment 2 — Systems & Parallelism（实现 DDP 梯度分桶重叠、实现 ZeRO-1 优化器分片 / FSDP、推导多维并行最佳拓扑配置）

## 1. 课程全景导读：当整个数据中心成为“基本计算单元”

在 [Lecture 7（分布式通信原语与 PyTorch 实战）](https://www.youtube.com/watch?v=SzpOcwdIL0Y) 中，Percy Liang 带领我们从代码层面认识了 `All-Reduce`、`All-Gather`、`Reduce-Scatter` 等集合通信原语，并手写了最基础的数据并行（DDP）、张量并行（TP）与流水线并行（PP）。

在本讲（[Lecture 8](https://www.youtube.com/watch?v=6-cXp-aOmdg)）中，Tatsu Hashimoto 将视角拉升至**万卡级超算集群的架构设计**：当我们要训练像 **Llama 3 405B**、**DeepSeek-V3 671B** 或 **Qwen 3** 这样的前沿大模型时，为什么单一的并行策略必然撞墙？我们如何像搭乐高积木一样，将 **4D 甚至 5D 并行策略（DP/ZeRO + TP + SP + PP + EP + CP）** 精密耦合在一起，实现**显存容量的线性扩展（Linear Memory Scaling）**与**有效算力的线性扩展（Linear Compute Scaling）**？

```
+===================================================================================================+
|                      CS336 Lecture 8: 大规模分布式训练 5D 混合并行技术全景图                         |
+===================================================================================================+
|                                                                                                   |
|  [Part I: 硬件网络哲学演进]                                                                         |
|   • GPU 胖树拓扑 (Fat-Tree All-to-All) vs. TPU 三维环面网格 (3D Toroidal Mesh)                      |
|   • 2026 趋同演化: Google TPU v8 (Virgo 网络) 转向 All-to-All 以适应 MoE; 华为 CloudMatrix 384 光互连 |
|                                                |                                                  |
|                                                v                                                  |
|  [Part II: 数据并行与零冗余分片 (ZeRO-1 / ZeRO-2 / ZeRO-3 & FSDP)]                                  |
|   • 16 Bytes/param 静态显存账单拆解 (参数 2B + 梯度 2B + FP32 主权重与 Adam 状态 12B)                |
|   • 利用 All-Reduce = Reduce-Scatter + All-Gather 免费消除冗余，结合多流预取 (Prefetch) 隐藏通信     |
|                                                |                                                  |
|                                                v                                                  |
|  [Part III: 模型并行与激活显存压榨 (PP, TP, SP, EP, CP)]                                             |
|   • 流水线并行 (PP): 气泡率公式 (p-1)/(m+p-1)、1F1B 调度、将反传拆为 B 和 W 的零气泡流水线 (Zero-Bubble)|
|   • 张量与序列并行 (TP+SP): Megatron 列并行+行并行配对，用 SP 消除 10sbh 残余激活，逼近 34sbh/t 下界   |
|   • 专家与上下文并行 (EP & CP): DeepEP 极低延迟 All-to-All 路由；Ring Attention 超长序列环状传递      |
|                                                |                                                  |
|                                                v                                                  |
|  [Part IV: 4D/5D 并行组合黄金法则与前沿模型实战案例 (Llama 3 405B, DeepSeek-V3, Qwen 3, Gemma 2)]     |
+===================================================================================================+
```

## 2. Part I：超算互连网络哲学与 2026 硬件趋同演化

在进入具体并行算法之前，Tatsu 首先揭示了底层硬件网络拓扑如何深刻塑造了不同大厂（OpenAI/Meta/DeepSeek vs. Google vs. 华为）的并行策略选择。

### 2.1 GPU 胖树架构（Fat-Tree）vs. TPU 三维环面网格（3D Toroidal Mesh）

| 维度 | NVIDIA GPU 集群网络（Fat-Tree 胖树 / All-to-All） | Google 传统 TPU 集群网络（3D Toroidal Mesh 环面网格） |
| :--- | :--- | :--- |
| **物理连接拓扑** | **分层交换机树（Hierarchical Switch Tree）**：节点内 8 卡（或 NVL72 的 72 卡、Pod 内最高 256 卡）通过 NVSwitch 全互连；跨机柜通过 Leaf Switch（叶交换机）与 Spine Switch（脊交换机）构建胖树 | **三维近邻环网（3D Torus）**：没有中心大交换机，每颗 TPU 芯片**只通过高速 ICI（Inter-Chip Interconnect）直连上下左右前后的 6 个直接邻居**，网格边缘首尾相连成环 |
| **核心优势** | **任意点对点路由极度灵活**：非常适合通信模式高度随机、动态变化的负载（如 **MoE 模型的跨卡 `All-to-All` 动态 Token 路由**） | **扩展成本极低、单链路极宽**：无论集群扩到多大，每个节点的邻居数恒定；在规则的环状集合通信（`All-Reduce`、`All-Gather`、**大跨度张量并行 TP** 与 **Ring Attention**）上效率极高 |
| **核心劣势** | 随着集群规模扩大，顶层 Spine 交换机与光模块成本、功耗和布线复杂度呈超线性增长，跨层跳数增加 | 当遇到非近邻的随机通信（如大范围 MoE `All-to-All`）时，数据包必须经过多次邻居转发（Multi-hop），容易引发拥塞 |

```
[GPU 胖树拓扑 (Fat-Tree: 灵活支持任意 All-to-All)]       [TPU 三维环面网格 (3D Torus: 极致优化近邻环状流水)]
          [ Spine Switches (顶层脊交换机) ]                    ╭───────────────────────╮
             /            |            \                      ▼                       │
    [Leaf Switch 0]  [Leaf Switch 1]  [Leaf Switch 2]       [TPU] ◄═══► [TPU] ◄═══► [TPU]
       /   |   \        /   |   \        /   |   \            ▲           ▲           ▲
    [NVSwitch 8卡]   [NVSwitch 8卡]   [NVSwitch 8卡]          ║           ║           ║
                                                            [TPU] ◄═══► [TPU] ◄═══► [TPU]
```

### 2.2 2026 前沿硬件的两大标志性事件：趋同演化与力大砖飞

1. **Google TPU v8（及 Ironwood）向胖树 All-to-All 拓扑的“趋同演化”**：
   Tatsu 在课堂上分享了当天早上的最新行业动态：过去 Google 工程师总是自豪地宣称 TPU 的 Toroidal Mesh 比 GPU 交换机更优雅（因为稠密模型的 TP 和 Ring Attention 只需要和邻居通信）。然而在最新发布的 **TPU v8（训练芯片）与 Ironwood（推理芯片）** 中，Google 在跨机柜层级引入了名为 **Virgo Network** 的交换机胖树拓扑，大幅转向 **All-to-All 连接**！
   - **根本原因**：2025–2026 年的前沿大模型几乎全军转向了 **MoE（混合专家）架构**。无论是在训练还是低延迟推理服务中，Token 动态路由到不同专家产生的全集群 `All-to-All` 流量成为了绝对瓶颈，迫使硬件架构向工作负载妥协，与 GPU 走向了**趋同演化（Convergent Evolution）**。
2. **华为昇腾 CloudMatrix 384（Ascend 910C）的“全光互连力大砖飞”权衡**：
   回想在 Lecture 5 中有人问“既然 SRAM 快，为什么不全用 SRAM 造芯片（Groq）？”在网络层面也有一个镜像问题：**“既然全互连好，为什么不用超高速光纤交换机把几百颗芯片全部暴力直连起来？”**
   这正是**华为 Ascend 910C（CloudMatrix 384 超节点）**的设计哲学：单颗昇腾 910 芯片的矩阵算力（MatMul FLOPs）和单卡显存带宽明显弱于 NVIDIA H200；但华为通过一整排庞大的光交换机柜，将 **384 颗芯片** 组合成一个超大规模的全互连高速域！
   - **工程权衡教训**：这种设计通过暴力拉高互连规模和芯片数量，成功弥补了单芯片制程与算力的劣势；但代价是**整机功耗高达同算力级别 NVIDIA 系统的 4 倍左右**（大量功耗花在了光电转换与长距离高速 I/O 上）。这生动展示了系统工程中“单芯片效率 vs. 互连规模 vs. 能源功耗”的铁三角权衡。

## 3. Part II：数据并行演进 —— 从朴素 DDP 到 ZeRO Stage 1, 2, 3 (FSDP)

我们的第一个并行维度是**数据并行（Data Parallelism）**：将全局批次 $B$ 切分到 $N_d$ 张 GPU 上，每张卡处理 $B / N_d$ 个样本：
$$w_{t+1} = w_t - \eta \frac{1}{B} \sum_{i=1}^B \nabla \mathcal{L}(w_t, x_i)$$
朴素 DDP 在算力扩展上非常完美，每步只需通信 $2\Psi$（$\Psi$ 为模型参数量，即一次 `All-Reduce`），但它在显存扩展上得分为 **0**——每张卡都完整复制了所有模型状态。

### 3.1 静态显存危机的精确解剖：为什么每个参数要吃掉 16 Bytes？

在混合精度（`bf16`/`fp16`）配合 **AdamW** 优化器训练参数量为 $\Psi$ 的大模型时，让我们像 [ZeRO 原论文 (Rajbhandari et al., 2019)](https://arxiv.org/abs/1910.02054) 那样，算清每张 GPU 被迫存储的“五份权重级张量（合计 **16 Bytes / parameter**）”：

1. **模型参数（Model Parameters, `bf16`/`fp16`）**：占 **$2\Psi\text{ Bytes}$**（前向与反向计算直接使用的半精度权重）。
2. **参数梯度（Gradients, `bf16`/`fp16`）**：占 **$2\Psi\text{ Bytes}$**（反向传播算出的半精度梯度）。
3. **AdamW 优化器状态（Optimizer States，合占 $12\Psi\text{ Bytes}$，即总静态显存的 75%！）**：
   - **FP32 主权重拷贝（Master Weights, `fp32`）**：占 **$4\Psi\text{ Bytes}$**（用于在单步微小更新 $\theta \leftarrow \theta - \eta \cdot \Delta\theta$ 时防止半精度尾数截断吃掉更新量）；
   - **一阶动量估计（First Moment $m_t$, `fp32`）**：占 **$4\Psi\text{ Bytes}$**（历史梯度的指数移动平均）；
   - **二阶动量估计（Second Moment $v_t$, `fp32`）**：占 **$4\Psi\text{ Bytes}$**（历史梯度平方的指数移动平均）。

$$\text{Total Static Memory per GPU (Naive DDP)} = \underbrace{2\Psi}_{\text{Params}} + \underbrace{2\Psi}_{\text{Grads}} + \underbrace{(4 + 4 + 4)\Psi}_{\text{AdamW States } (K=12)} = \mathbf{16\Psi\text{ Bytes}}$$

**触目惊心的算例**：假设我们要在 $N_d = 64$ 张 GPU 上训练一个仅 **$\Psi = 7.5\text{B}$（75 亿参数）** 的模型：
- 在朴素 DDP 下，每张卡需要 $16 \times 7.5 \times 10^9 = \mathbf{120\text{ GB}}$ 的静态显存！这意味着连一张 80GB 的 A100/H100 都装不下 7.5B 模型（哪怕还没算激活值）！

### 3.2 ZeRO（Zero Redundancy Optimizer）三级分片演进全解

既然 64 张卡上存了 64 份完全一模一样的优化器状态、梯度和参数，我们能否将它们**分片（Shard）**到不同 GPU 上，消除冗余？

```
+===================================================================================================+
|             朴素 DDP vs. ZeRO Stage 1 / 2 / 3 单卡静态显存占用对比 (以 7.5B 模型, 64 卡为例)         |
+===================================================================================================+
|                                                                                                   |
|  [Baseline DDP]  : [ Params (2Ψ) ] [ Grads (2Ψ) ] [======= Optimizer States (12Ψ) =======]        |
|                    单卡显存: 4Ψ + 12Ψ = 16Ψ = 120.0 GB (全量冗余复制)                              |
|                                                                                                   |
|  [ZeRO Stage 1]  : [ Params (2Ψ) ] [ Grads (2Ψ) ] [ Opt (12Ψ / N_d) ]                             |
|  (仅分片优化器)     单卡显存: 4Ψ + 12Ψ/64 = 31.4 GB (省去 ~4x 显存，且通信开销 0 增加!)             |
|                                                                                                   |
|  [ZeRO Stage 2]  : [ Params (2Ψ) ] [G(2Ψ/N_d)] [ Opt (12Ψ / N_d) ]                                |
|  (分片优化器+梯度)  单卡显存: 2Ψ + 14Ψ/64 = 16.6 GB (省去 ~8x 显存，且通信开销 0 增加!)             |
|                                                                                                   |
|  [ZeRO Stage 3]  : [P(2Ψ/N_d)] [G(2Ψ/N_d)] [ Opt (12Ψ / N_d) ]                                    |
|  (即 PyTorch FSDP) 单卡显存: 16Ψ/64 = 1.875 GB ≈ 1.9 GB (显存完美线性缩减 64 倍，通信量仅 1.5x!)   |
+===================================================================================================+
```

#### （1）ZeRO Stage 1（Optimizer State Sharding，`Pos` —— 完全免费的午餐！）
- **核心思想**：每张 GPU 依然保存完整的参数（$2\Psi$）和完整的梯度（$2\Psi$），因此每张卡可以毫无障碍地独立跑完前向传播与反向传播；但**将占大头的 $12\Psi$ 优化器状态切分为 $N_d$ 份，第 $r$ 号 GPU 只保存第 $r$ 份参数的优化器状态，并只负责更新这 $1/N_d$ 份参数**！
- **四步执行流水线与通信账单**：
  1. **局部前反向**：每张卡在自己的局部数据微批次上算出完整参数的局部梯度（此时无需任何跨卡通信）。
  2. **梯度 `Reduce-Scatter`（通信量 $= \Psi$）**：调用一次 `Reduce-Scatter`，将所有卡算出的关于第 $r$ 份参数的梯度求和并汇总到第 $r$ 号 GPU 上（其余非自己负责的梯度分片直接丢弃）。
  3. **分片参数更新（零通信，计算量降为 $1/N_d$）**：第 $r$ 号 GPU 利用自己持有的第 $r$ 份优化器状态（$m^{(r)}, v^{(r)}$）和刚收到的汇总梯度 $g^{(r)}$，独立完成第 $r$ 份参数 $w^{(r)}$ 的 AdamW 更新。
  4. **参数 `All-Gather`（通信量 $= \Psi$）**：所有 GPU 执行一次 `All-Gather`，将各自更新好的 $1/N_d$ 参数分片广播给所有其他 GPU，使每张卡恢复完整的最新参数以迎接下一个 Step。
- **为什么说 ZeRO-1 是“彻头彻尾的免费午餐”？**
  请看总通信量：Step 2 的 `Reduce-Scatter`（$\Psi$）+ Step 4 的 `All-Gather`（$\Psi$）$= \mathbf{2\Psi}$！根据我们在 Lecture 7 证明的恒等式 $\text{All-Reduce} \equiv \text{Reduce-Scatter} + \text{All-Gather}$，**ZeRO-1 的通信字节数与朴素 DDP 的单次 `All-Reduce`（$2\Psi$）分毫不差**，却白白将单卡静态显存从 $120\text{ GB}$ 砍到了 $31.4\text{ GB}$！

#### （2）ZeRO Stage 2（Gradient + Optimizer State Sharding，`Pos+g` —— 依然免费！）
- **核心直觉**：既然在 ZeRO-1 中，第 $r$ 号 GPU 在更新参数时只需要用到第 $r$ 份梯度，那我们为什么要傻傻地等整张网络的完整梯度全部算出来存满显存后，再去调 `Reduce-Scatter` 呢？
- **边反传、边归约、边释放（Incremental Reduce & Immediate Free）**：
  在反向传播从第 $L$ 层往第 1 层回溯的过程中：一旦算完某一层（或某一个分片 Bucket）的局部梯度，**立即对该层梯度发起 `Reduce`（或 `Reduce-Scatter`）将其发送给负责更新该层的目标 GPU**；一旦发送完毕，所有不负责更新该分片的 GPU **当场释放（Free）该层梯度的显存**！
- **收益与代价**：任何时刻显存中都不再驻留完整的 $2\Psi$ 梯度，单卡显存进一步降至 $2\Psi + \frac{14\Psi}{N_d}$（$16.6\text{ GB}$），而**总通信量依然严格保持在 $\Psi + \Psi = \mathbf{2\Psi}$**！

#### （3）ZeRO Stage 3 / PyTorch FSDP（Parameter + Gradient + Optimizer Sharding，`Pos+g+p`）
- **终极推演**：如果模型大到连单份半精度参数 $2\Psi$（例如 100B 模型的参数就有 $200\text{ GB}$）都塞不进单卡显存怎么办？那就连**模型参数 $2\Psi$ 也一起按 $N_d$ 切分**！平时每张卡只存 $1/N_d$ 的参数分片，只有在算到某一层时，才**按需即时收集（On-Demand `All-Gather`）**该层的完整参数，算完立刻扔掉！这就是 DeepSpeed **ZeRO-3** 以及 PyTorch 原生 **`FullyShardedDataParallel` (FSDP)** 的核心机制。
- **FSDP 单步生命周期与 $3\Psi$ 通信账单**：
  1. **前向传播逐层循环（Forward Pass）**：
     对第 $l = 1 \dots L$ 层：发起 **`All-Gather`** 收集第 $l$ 层的完整权重 $W_l$ $\rightarrow$ 计算第 $l$ 层前向激活 $a_l = f(a_{l-1}, W_l)$ $\rightarrow$ **立即释放（Free）不属于本卡的第 $l$ 层权重分片**！*（前向累计通信量 $= \mathbf{1\Psi}$）*
  2. **反向传播逐层循环（Backward Pass）**：
     对第 $l = L \dots 1$ 层：由于前向算完就把权重扔了，反向传播算梯度时必须**再次发起 `All-Gather`** 收集第 $l$ 层的完整权重 $W_l$ $\rightarrow$ 计算梯度 $\nabla a_{l-1}$ 与局部权重梯度 $\nabla W_l$ $\rightarrow$ 立即发起 **`Reduce-Scatter`** 将 $\nabla W_l$ 归约到所属卡上 $\rightarrow$ **立即释放第 $l$ 层权重与非本卡梯度**！*（反向累计通信量 $= 1\Psi\text{ (All-Gather)} + 1\Psi\text{ (Reduce-Scatter)} = \mathbf{2\Psi}$）*
  3. **优化器更新**：每张卡各自用分片梯度和分片优化器状态更新自己的 $1/N_d$ 参数分片（无需再做末尾 `All-Gather`，因为下一步前向开始时会自动逐层 `All-Gather`）。

```
+---------------------------------------------------------------------------------------------------+
|                   PyTorch FSDP (ZeRO-3) 多流异步预取与通信-计算重叠时间线 (Overlap)                  |
+---------------------------------------------------------------------------------------------------+
|                                                                                                   |
|  CPU Stream   : [Launch AG0][Launch Fwd0][Launch AG1][Launch Fwd1][Launch AG2][Launch Fwd2]...     |
|                                                                                                   |
|  GPU Comm 流  : [ All-Gather L0 ] [ All-Gather L1 ] [ All-Gather L2 ]   <-- 提前预取下一层权重!    |
|                                 \                 \                 \                             |
|                                  v                 v                 v                            |
|  GPU Comp 流  :                  [  Forward L0   ] [  Forward L1   ] [  Forward L2   ]            |
|                                                  ^                 ^                              |
|                                               算完 L0 立即 Free L0 非本卡权重分片!                  |
|                                                                                                   |
|  * 核心条件：只要单层计算耗时 T_comp(Forward L_i) >= 单层权重通信耗时 T_comm(All-Gather L_{i+1})，   |
|    多出来的 1.5x (3Ψ vs 2Ψ) 通信开销就会被完美隐藏在计算流之下!                                     |
+---------------------------------------------------------------------------------------------------+
```

#### 数据并行四阶段全景总结表

| 阶段方案 | 单卡静态显存公式 ($\Psi$ 为参数量, $N_d$ 为卡数) | 7.5B 模型 @ 64 卡单卡显存 | 每步总通信量 | 核心通信原语构成 | 工程评价 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Naive DDP** | $16\Psi$ | `120.0 GB` (OOM) | **$2\Psi$** | $1 \times \text{All-Reduce}$ | 实现最简，但静态显存完全不随卡数缩减 |
| **ZeRO-1** | $4\Psi + \frac{12\Psi}{N_d}$ | `31.4 GB` | **$2\Psi$** | $1 \times \text{Reduce-Scatter} + 1 \times \text{All-Gather}$ | **零通信代价**砍掉 75% 静态显存（Assignment 2 必写） |
| **ZeRO-2** | $2\Psi + \frac{14\Psi}{N_d}$ | `16.6 GB` | **$2\Psi$** | 逐层 $\text{Reduce-Scatter} + 1 \times \text{All-Gather}$ | **零通信代价**进一步释放梯度显存 |
| **ZeRO-3 / FSDP** | $\frac{16\Psi}{N_d}$ | **`1.9 GB`** | **$3\Psi$** ($1.5\times$) | $2 \times \text{All-Gather} + 1 \times \text{Reduce-Scatter}$ | 静态显存完美线性扩展；靠异步预取（Prefetch）隐藏额外 $\Psi$ 通信 |

### 3.3 为什么有了优雅的 FSDP 还不够？（逼出模型并行的两大死穴）
Tatsu 坦言：“我很希望这节课到此就可以下课了，因为 FSDP 既干净又优雅；但遗憾的是，在大规模下 FSDP 有两个无法克服的死穴”：
1. **死穴一：数据并行会耗尽“全局批大小（Global Batch Size）”资源**：
   每张数据并行卡至少要分到 1 个样本（$B_{\text{global}} \ge N_d$）。根据 **Critical Batch Size（临界批大小）** 定律，当全局 Batch Size 超过某个阈值后，继续增大 Batch Size 对梯度的信噪比提升出现严重的边际递减（算力被浪费在冗余样本上，无法转化为更少的训练步数）。
2. **死穴二：FSDP 丝毫不能减少“激活值显存（Activation Memory）”**：
   在 FSDP 计算第 $l$ 层的瞬间，它必须把第 $l$ 层的完整权重 `All-Gather` 到本卡并算出完整的层激活。当模型很宽或上下文序列很长时，动态激活显存会远超静态参数显存！

## 4. Part III：模型并行与激活并行深潜（PP, TP, SP, EP & CP）

为了在不消耗全局 Batch Size 的前提下继续切分模型并削减激活显存，我们进入**模型并行（Model Parallelism）**。
> **FSDP 与模型并行的核心概念分野**：
> - 在 **FSDP** 中，每张卡都在跑完整的层计算，**在网络上飞来飞去的是模型参数（Parameters & Gradients）**；
> - 在 **Model Parallelism（PP / TP / EP）** 中，模型参数固定驻留在各自所属的 GPU 上不动，**在网络上飞来飞去的是层间/层内的激活值（Activations）**！

---

### 4.1 流水线并行（Pipeline Parallelism, PP）与“零气泡（Zero-Bubble）”革命

#### （1）微批次流水线的气泡率公式（Bubble Fraction）與通信优势
在流水线并行中，我们将 $L$ 层网络按深度切分为 $p$ 个舞台（Stages，对应 $p$ 张 GPU），并将一个批次切分为 $m$ 个微批次（Micro-batches）：
- **气泡率（Bubble Time Ratio）**：在经典的 GPipe（先跑完所有 $m$ 个微批次的前向，再跑所有 $m$ 个微批次的反向）或 1F1B 调度下，首尾阶段等待流水线填满与排空的空闲时间占比为：
  $$\text{Bubble Fraction} = \frac{p - 1}{m + p - 1} = \mathcal{O}\left(\frac{p}{m}\right)$$
  **结论**：要想把气泡率压向 0，微批次数量 $m$ 必须远大于流水线级数 $p$（即 $m \gg p$），这意味着**流水线并行同样要消耗宝贵的 Batch Size 资源**！
- **为什么流水线并行虽然代码极难写，却在多机训练中不可替代？**
  因为它的**通信属性极其优异**：它不需要像 FSDP 那样广播整个权重矩阵（$d_{\text{model}}^2$），也不需要像 TP 那样每层做全卡 `All-Reduce`，它**只在相邻两个 Stage 的交界处通过点对点（Point-to-Point `send`/`recv`）传输一次激活张量（大小仅为 $b \times s \times h$）**！因此，在多机集群中，**总是把流水线并行部署在最慢的跨节点/跨机柜/跨数据中心慢速网络链路上**。

#### （2）从 1F1B 到“零气泡流水线（Zero-Bubble Pipelining）”的精妙拆解
如何进一步消灭那 $\frac{p-1}{m+p-1}$ 的流水线气泡？[Qi et al. (2024) (Sea AI Lab)](https://arxiv.org/abs/2401.10241) 与 **DeepSeek-V3（DualPipe）** 提出了一个令 Tatsu 拍案叫绝的洞察——**切开反向传播算子（Splitting Backward into $B$ and $W$）**：

回顾我们在 [Lecture 2](https://cs336.stanford.edu/lectures/?trace=lecture_02) 推导过的线性层 $Y = X W$ 的反向传播：当收到上游传来的 $\nabla Y$ 时，反向传播实际上干了**两件完全独立的事**（各占一半 FLOPs）：
1. **计算对输入激活的梯度 $B$（$\nabla X = \nabla Y \cdot W^\top$）**：
   **位于流水线的关键路径（Critical Path）上！** 因为上游的 Stage $p-1$ 正处于干等状态，必须等收到这个 $\nabla X$ 才能开始它那里的反向传播！
2. **计算对当前层权重的梯度 $W$（$\nabla W = X^\top \cdot \nabla Y$）**：
   **是计算图上的叶子节点（Leaf Node）！** 上游 Stage 根本不需要 $\nabla W$，当前卡只需要在整个大 Step 结束调 `optimizer.step()` 之前把 $\nabla W$ 算出来即可，**在时间轴上完全可以随时推迟执行（Deferrable）**！

```
+---------------------------------------------------------------------------------------------------+
|               传统 1F1B 流水线 vs. Zero-Bubble (零气泡) 流水线调度对比 (将反传拆为 B 和 W)           |
+---------------------------------------------------------------------------------------------------+
|                                                                                                   |
|  [传统 1F1B 调度: 每次反传把 (B+W) 绑在一起执行，上游 Stage 被迫多等一倍时间，产生大量空闲 Bubble]    |
|   Stage 0: [F0][F1][F2][F3]          [B0+W0]      [B1+W1]      [B2+W2]      [B3+W3]               |
|   Stage 1:     [F0][F1][F2][F3]  [B0+W0]      [B1+W1]      [B2+W2]      [B3+W3]                   |
|   Stage 2:         [F0][F1][F2][B0+W0][F3][B1+W1]  [B2+W2]  [B3+W3]                               |
|                                                                                                   |
|  [Zero-Bubble 调度: 优先极速算完关键路径 B 立即发给上游，把叶子任务 W 塞进原本空闲的气泡空隙中!]      |
|   Stage 0: [F0][F1][F2][F3][B0][W0][B1][W1][B2][W2][B3][W3]  --> 气泡几乎被 W 完全填平，逼近 0 气泡! |
|   Stage 1:     [F0][F1][F2][B0][F3][B1][W0][B2][W1][B3][W2][W3]                                   |
|   Stage 2:         [F0][B0][F1][B1][F2][B2][F3][B3][W0][W1][W2][W3]                               |
+---------------------------------------------------------------------------------------------------+
```
*工程代价*：要实现 Zero-Bubble 或 DeepSeek 的 DualPipe，不能直接调 PyTorch 默认的 `loss.backward()`，必须手动接管自动求导图，用优先级队列分别调度 `F`、`B` 和 `W` 算子并精确管理显存释放时机。

---

### 4.2 张量并行（Tensor Parallelism, TP）：Megatron-LM 的列并行与行并行配对

在 Lecture 7 中，Percy 演示的朴素 TP 对每一层都做按列切分并在每层末尾做 `All-Gather`。而在真实的 Transformer 中，[Megatron-LM (Shoeybi et al., 2019)](https://arxiv.org/abs/1909.08053) 通过将**第一个矩阵按列切分（Column-Parallel）**与**第二个矩阵按行切分（Row-Parallel）**巧妙配对，**省去了中间层的通信**！

#### （1）MLP 层的列并行 + 行并行配对机制
考虑 Transformer 的两层 MLP：$Z = \text{GeLU}(X A) B$（对于 SwiGLU 同理，将门控双分支均按列切分）：
1. **第一层升维矩阵 $A$ 按列切分（Column-Parallel）**：$A = [A_1 \mid A_2]$。
   - 输入 $X$ 经过算子 $f$（前向为 **Identity 恒等直通**，无需通信），在两张卡上分别算出 $Y_1 = \text{GeLU}(X A_1)$ 和 $Y_2 = \text{GeLU}(X A_2)$。因为 GeLU 是逐元素运算，所以 $[Y_1 \mid Y_2] = \text{GeLU}(X [A_1 \mid A_2])$！
2. **第二层降维矩阵 $B$ 按行切分（Row-Parallel）**：$B = \begin{bmatrix} B_1 \\ B_2 \end{bmatrix}$。
   - **关键洞察**：根据分块矩阵乘法规则，列分块乘以行分块等于外积之和：
     $$Z = \begin{bmatrix} Y_1 & Y_2 \end{bmatrix} \begin{bmatrix} B_1 \\ B_2 \end{bmatrix} = Y_1 B_1 + Y_2 B_2$$
   - 因此，GPU 1 和 GPU 2 **根本不需要在中间通信同步 $Y_1, Y_2$**！它们可以直接拿自己手头的 $Y_i$ 乘上行分片 $B_i$ 得到完整维度的部分和 $Z_i = Y_i B_i \in \mathbb{R}^{b \times s \times h}$，最后经过算子 $g$（前向执行 **1 次 `All-Reduce` 求和** $Z = Z_1 + Z_2$）即可得到最终 MLP 输出！
3. **前反向算子对偶性（Conjugate Duality between $f$ and $g$）**：
   - **算子 $f$（模块入口）**：前向是 **Identity（零通信）**，反向传播时是 **`All-Reduce`（将两分支回传给输入 $X$ 的梯度相加）**；
   - **算子 $g$（模块出口）**：前向是 **`All-Reduce`（将两分支输出的部分和相加）**，反向传播时是 **Identity（上游梯度直接原样复制进两分支）**！

```
+---------------------------------------------------------------------------------------------------+
|                   Megatron-LM 张量并行 (TP) 在 MLP 与 Attention 中的切分与算子对偶                  |
+---------------------------------------------------------------------------------------------------+
|                                                                                                   |
|                   ┌──► [ A_1 (按列切) ] ──► GeLU ──► Y_1 ──► [ B_1 (按行切) ] ──► Z_1 ──┐          |
|  输入 X ──► ( f ) ─┤                   (中间零通信!)                                     ├─► ( g ) ──► 输出 Z
|                   └──► [ A_2 (按列切) ] ──► GeLU ──► Y_2 ──► [ B_2 (按行切) ] ──► Z_2 ──┘          |
|                                                                                                   |
|  • 算子 ( f ): 前向 = Identity (不通信)        |  反向 = All-Reduce (汇总输入 X 的两路梯度)         |
|  • 算子 ( g ): 前向 = All-Reduce (汇总 Z1+Z2)  |  反向 = Identity (不通信)                         |
|  • Attention 层同理: W_Q, W_K, W_V 按 Attention Heads 列切分 (中间 Softmax 独立无通信)，W_O 按行切分!|
+---------------------------------------------------------------------------------------------------+
```

- **TP 的通信代价与硬件铁律（`TP <= 8`）**：
  每个 Transformer 层包含 1 个 Attention 块和 1 个 MLP 块，因此每层前向需要 **2 次 `All-Reduce`**，反向也需要 **2 次 `All-Reduce`**（单层通信量高达 $8 b s h$ 个元素，且处于计算关键路径上无法像 FSDP 那样跨层预取）。因此，在 GPU 集群上，**张量并行度 $t$ 几乎永远不超过单节点的 8 张 NVLink 卡（`TP <= 8`）**——一旦跨节点走慢速网络，吞吐量会遭遇断崖式暴跌！

---

### 4.3 激活值显存精确记账与序列并行（Sequence Parallelism, SP）

很多人以为“只要把参数切碎了，显存就够用了”。Tatsu 展示了真实的 PyTorch Profiler 显存时间线与 [Korthikanti et al. (2022)](https://arxiv.org/abs/2205.05198) 的统计图：**在中等偏长的序列长度下，前向传播层层累积的动态激活值显存（Activation Memory，图中的巨大红色洪峰）会远远超过静态参数显存！**

#### （1）单层 Transformer 激活显存的精确数学公式（`34 sbh` 定律）
设序列长度为 $s$，微批次大小为 $b$，隐藏维度为 $h$，注意力头数为 $a$，张量并行度为 $t$（采用 16-bit 混合精度，每元素 2 Bytes）。为了支持反向传播求导，单层 Transformer 必须缓存在显存中的激活字节数如下：

1. **无并行基线（No Parallelism, $t = 1$）**：
   $$\text{Activation Memory}_{\text{Baseline}} = s b h \left(34 + 5 \frac{a s}{h}\right)\text{ Bytes}$$
   - **公式各项拆解**：
     - **`24 sbh`**：来自 MLP 内部的升维激活（SwiGLU/GeLU 输入输出）与 Attention 的 $Q, K, V$ 投影激活；
     - **`10 sbh`**：来自两个 LayerNorm/RMSNorm 的输入输出、残差连接主干输入以及 Dropout 掩码；
     - **`sbh (5 as / h) = 5 b a s^2`**：来自与序列长度呈二次方关系的 $s \times s$ 注意力得分矩阵、Softmax 输出（各占 $2 b a s^2$ Bytes）与 Dropout 掩码（占 $1 b a s^2$ Bytes）。
2. **仅开启张量并行（Tensor Parallelism Only）**：
   TP 切分了 MLP 和 Attention 的矩阵乘法（即切分了 `24 sbh` 和 `5 as/h` 项），但**没有切分 LayerNorm、Dropout 和进入 Attention/MLP 前的完整输入激活（即 `10 sbh` 项依然在每张 TP 卡上全量复制）**！
   $$\text{Activation Memory}_{\text{TP}} = s b h \left(\mathbf{10} + \frac{24}{t} + 5 \frac{a s}{h t}\right)\text{ Bytes}$$
   这意味着即便你把 $t \to \infty$，每层依然有顽固的 **$10 sbh$** 激活显存无法消除！
3. **开启张量并行 + 序列并行（`TP + SP`）**：
   为了消灭那顽固的 $10 sbh$，[Korthikanti et al. (2022)](https://arxiv.org/abs/2205.05198) 提出了 **Sequence Parallelism（序列并行，SP）**：
   - **核心洞察**：LayerNorm、RMSNorm、Dropout 和残差加法虽然不能沿隐藏维度 $h$ 切分，但它们**沿着序列维度 $s$（或 Token 维度 $b \times s$）是完全逐位置独立的**！
   - **零额外通信代价替换**：利用 $\text{All-Reduce} \equiv \text{Reduce-Scatter} + \text{All-Gather}$，我们将原本 TP 出口的算子 $g$（`All-Reduce`）替换为 **`Reduce-Scatter`（沿序列轴 $s$ 切分）**，让后续的 LayerNorm 和残差连接在长度仅为 $s / t$ 的序列分片上执行（显存降为 $\frac{10 sbh}{t}$！）；然后在进入下一个 TP 线性层前，将入口算子 $f$ 从 Identity 替换为 **`All-Gather`（沿序列轴拼回完整长度 $s$）**！
   - **通信量完全不变**（1 次 `All-Reduce` 拆成了 1 次 `Reduce-Scatter` + 1 次 `All-Gather`），但那 `10 sbh` 的激活显存被成功除以了 $t$：
     $$\text{Activation Memory}_{\text{TP+SP}} = s b h \left(\frac{34}{t} + 5 \frac{a s}{h t}\right)\text{ Bytes}$$
4. **终极形态：`TP + SP` + 选择性激活重计算（FlashAttention Recomputation）**：
   如我们在 Lecture 5/6 所学，使用 **FlashAttention（或选择性重计算 Attention Softmax）** 后，前向无需保存庞大的 $s \times s$ 注意力矩阵，二次方项 $5 \frac{as}{ht}$ 直接归零！我们得到了手算显存容量的**黄金理论下界公式**：
   $$\text{Activation Memory}_{\text{TP+SP+Recompute}} = \mathbf{\frac{34 \, s b h}{t}\text{ Bytes}}$$

```
+---------------------------------------------------------------------------------------------------+
|                     Megatron-LM 张量并行 (TP) + 序列并行 (SP) 协同工作流水线                        |
+---------------------------------------------------------------------------------------------------+
|                                                                                                   |
|  [SP 区域: 沿序列维 s 切分, 形状 (b, s/t, h)]   [TP 区域: 沿隐藏维 h 切分, 形状 (b, s, h/t)]         |
|                                                                                                   |
|  残差输入 ──► [LayerNorm (仅存 s/t)] ──► ( g: 前向 All-Gather ) ──► [MLP / Attn (Col + Row TP)]     |
|                                        (   反向 Reduce-Scatter)             │                     |
|                                                                             ▼                     |
|  下一层   ◄── [Dropout/残差 (仅存 s/t)] ◄── ( ḡ: 前向 Reduce-Scatter) ◄─────┘                     |
|                                             (   反向 All-Gather    )                              |
|  * 通信账单: 1 次 All-Gather + 1 次 Reduce-Scatter == 原本 TP 的 1 次 All-Reduce (零额外通信开销!) |
+---------------------------------------------------------------------------------------------------+
```

---

### 4.4 专家并行（Expert Parallelism, EP）与上下文并行（Context Parallelism, CP）

#### （1）专家并行（EP）：为什么 MoE 模型首选 EP 而非 TP？
对于 Mixture of Experts（MoE）模型，我们可以将不同的专家子网络（Experts）放置在不同的 GPU 上。根据 **NVIDIA Megatron 官方指南**，在切分 MoE FFN 层时，**几乎总是优先使用专家并行（EP）而非张量并行（TP）**：
- **保持大矩阵尺寸与高算术强度**：TP 会把每个专家原本就不大的权重矩阵切得更碎（细粒度专家下矩阵已经很小），导致单卡 GEMM 跌破 Roofline 拐点、GPU 利用率暴跌；而 EP 让每个被分配到本卡的专家保持完整的矩阵尺寸！
- **只传输稀疏激活 Token**：TP 需要对稠密激活做全量 `All-Reduce`，而 EP 只需通过 `All-to-All` 将当前 Token 路由到被激活的 Top-$K$ 个专家上。
- **工业界极限优化（DeepSeek `DeepEP` 与 NVIDIA `HybridEP`）**：
  由于每层 MoE 都要执行跨卡甚至跨节点的 `All-to-All` 路由分发（Dispatch）与回收（Combine），且后续计算必须等待 Token 到达（极度延迟敏感 Latency-Sensitive），DeepSeek 专门开源了底层通信库 **[DeepEP](https://github.com/deepseek-ai/DeepEP)**（甚至挖掘出了未公开文档的底层 PTX 机器指令来加速网卡与 NVLink 转发），NVIDIA 也推出了对应的 **HybridEP**。
- **解耦 Attention 与 MoE 的并行策略（Decoupled Parallelism）**：
  由于 EP 只切分了 MoE FFN 层而**完全没有切分 Attention 层**，如果对全网使用高 TP 会把 MoE 矩阵切太碎，不用 TP 又会导致 Attention 层太大。因此现代训练框架采用**异构解耦并行**：让 Attention 层使用较高的 TP/CP，而让 MoE 层使用高 EP（如 `EP=32` 或 `64`）配合极低的 TP（`TP=1` 或 `2`）！

#### （2）上下文并行（Context Parallelism, CP / Ring Attention）
请注意区分两个极易混淆的术语：
- **Sequence Parallelism (SP)**：仅作为 TP 的挂件，切分的是 LayerNorm / Dropout 的激活，在进入 Attention 层前已经通过 `All-Gather` 还原成了完整序列长度 $s$。
- **Context Parallelism (CP / [Ring Attention, Liu et al., 2023](https://arxiv.org/abs/2310.01889))**：针对百万级超长上下文（如 $s = 128\text{K} \sim 1\text{M}$，此时单条序列的激活连单张卡都塞不下），**把整条长序列从头到尾（包括在 Attention 层内部！）切分到多张卡上**。在计算 Attention 时，各卡固定自己的 Query 块 $Q_i$，一边在本地计算分块 FlashAttention（利用 Online Softmax 动态更新统计量），一边像击鼓传花一样沿环状网络（Ring）把 Key/Value 块 $(K_j, V_j)$ 传给下一个邻居，实现通信与注意力计算的完全重叠！

## 5. Part IV：多维混合并行（3D / 4D / 5D Parallelism）组合法则与前沿模型实战

### 5.1 六大并行策略全景优劣势对比表（Tatsu's Trade-off Matrix）

没有任何一种并行策略是完美无缺的（表中**加粗项**即为 Tatsu 在幻灯片中标红的短板，正是为了弥补彼此的短板，我们才需要组合出 4D/5D 并行）：

| 并行策略 | 参数/优化器显存缩减 | 激活显存 (Activation) 缩减 | 每层每卡通信开销 | 带宽与延迟敏感度 | 是否消耗全局 Batch Size？ | 工程实现复杂度 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **DDP (朴素数据并行)** | **无 ($1\times$，全量冗余)** | **无 ($1\times$)** | 低（每步 $2\Psi$，可重叠） | 低（支持跨节点慢速网） | **是（受限于临界批大小）** | 极低 |
| **ZeRO-3 / FSDP** | **完美线性 ($\frac{1}{N_d}$)** | **无 ($1\times$，单层需还原)** | 中（每步 $3\Psi$，靠预取隐藏） | 中（需足够批大小掩盖通信） | **是（消耗全局 Batch Size）** | 低（框架原生封装） |
| **Pipeline Parallel (PP)** | 完美线性 ($\frac{1}{p}$) | 线性缩减 ($\approx \frac{1}{p}$) | **极低（仅交界处 P2P 传 $bsh$）** | **极低（最适合跨节点慢网）** | **是（需大 $m$ 压低气泡率）** | **极高（气泡调度复杂）** |
| **Tensor + Seq Parallel (TP+SP)** | 完美线性 ($\frac{1}{t}$) | **完美线性 ($\frac{34sbh}{t}$)** | **极高（每层 $8bsh$ 集合通信）** | **极高（锁死在节点内 NVLink）** | 否（不消耗 Batch Size！） | 中（需改写模型层） |
| **Expert Parallel (EP)** | 线性缩减 FFN ($\frac{1}{e}$) | 缩减 MoE 激活 | **高（每层动态 `All-to-All`）** | **高（对延迟敏感，需负载均衡）** | 间接相关（需足够 Token 均摊） | **极高（需 DeepEP 级优化）** |
| **Context Parallel (CP / Ring)** | 无（不切权重） | **按序列长度线性缩减 ($\frac{1}{c}$)** | 中（环状传递 KV 块，可重叠） | 中高（需长序列计算掩盖通信） | 否（专为单序列超长上下文设计） | 高 |

### 5.2 组合 4D/5D 并行的“两步走黄金处方”与 Megatron 官方五大铁律

面对一个具体的模型规模和集群拓扑，应该如何确定 $(TP, CP, PP, EP, DP)$ 的数值？Tatsu 给出了极其清晰的**两步走处方（Simple Prescription）**：

1. **第一步：先切分模型，直到模型 + 激活能塞进单卡显存（Until Your Model Fits in Memory）**：
   - **优先在节点内（高速 NVLink 域）开满 `TP (+ SP)` 或 `EP`**：如果单机有 8 张 GPU，优先将 `TP`（稠密模型）或 `EP`（MoE 模型）设到 $\le 8$，这既能同时砍掉静态参数显存和动态激活显存，又不会消耗全局 Batch Size。
   - **若单机 8 卡仍装不下，跨节点开启 `PP`（或 `ZeRO-3 / FSDP`）**：根据跨节点网络带宽选择——如果跨节点带宽较窄或模型极大，使用 **Pipeline Parallelism (`PP`)** 跨节点切分深度；如果跨节点网络较快且 Batch 充足，也可直接叠加 **FSDP (`ZeRO-3`)**。
   - **若处于超长上下文阶段（如 128K+）**：开启 **Context Parallelism (`CP`)** 切分长序列激活。
2. **第二步：模型装下之后，把剩下的所有 GPU 全部投入数据并行（Until You Run Out of GPUs）**：
   - 用 **Data Parallelism（配合 `ZeRO-1` 或 `ZeRO-2`）** 吃满剩余的所有计算节点，实现吞吐量的线性扩展！
   - 如果此时单卡分到的微批次太小或者流水线气泡太大，开启 **Gradient Accumulation（梯度累积）** 与 **Activation Recomputation（激活重计算：用多算一次前向换取更大的微批次，从而压缩流水线气泡、反过来提升整体硬件利用率 MFU！）**。

这与 **NVIDIA Megatron-LM / Megatron-Bridge 官方并行指南** 的五条黄金准则完全吻合：
1. **Minimize Model Parallelism, Maximize Data Parallelism**（模型并行只開到刚好能装下模型为止，其余算力全部给数据并行）；
2. **Stay Within NVLink**（将 TP 和常规 EP 严格限制在单机 8 卡或 NVL72 的高速 NVLink 域内）；
3. **Use Pipeline Parallelism to Scale Multi-Node**（跨多机扩展模型容量时使用流水线并行）；
4. **Prefer Expert Parallelism over Tensor Parallelism for MoE**（对于 MoE 层，优先使用 EP 而非 TP 以保持大矩阵算术强度）；
5. **Use Context Parallelism for Long Sequences**（长序列训练开启 Ring Attention 上下文并行）。

### 5.3 工业界旗舰大模型真实并行配置全览（2024–2026 Case Studies）

Tatsu 在课程最后盘点了从 7B 到 671B 的真实工业界训练配置（包括来自各技术报告及 NVIDIA **Megatron-Bridge** 仓库的实测推荐配置）：

| 模型名称与参数规模 | 模型架构类型 | 硬件集群与规模 | 实际采用的并行策略组合 $(DP, PP, TP, EP, CP)$ | 架构师复盘与核心设计亮点 |
| :--- | :--- | :--- | :--- | :--- |
| **[OLMo 7B (Dolma)](https://arxiv.org/abs/2402.00838)** | 7B 稠密 (Dense) | 多节点 A100 / H100 | **纯 `FSDP (ZeRO-3)`**（`TP=1, PP=1`） | 对于 $\le 7\text{B}\sim 13\text{B}$ 的中小规模稠密模型，无需复杂的 TP/PP，纯 FSDP 配合通信重叠即可实现极佳扩展 |
| **[Yi-34B](https://arxiv.org/abs/2403.04652) / [DeepSeek-V1 67B](https://arxiv.org/abs/2401.02954)** | 34B / 67B 稠密 | 多节点 A100 / H800 | **`ZeRO-1 DP + TP + SP + PP`** | 教科书级的经典 3D 并行：单机内 `TP+SP`，跨机 `PP`（DeepSeek-V1 使用 8 级流水线），最外层 `ZeRO-1` 数据并行 |
| **[Llama 3 405B](https://arxiv.org/abs/2407.21783) (主预训练阶段, $S=8\text{K}$)** | 405B 巨型稠密 | **16,384 张 H100** (RoCE / IB) | **`TP=8, CP=1, PP=16, DP=128 (FSDP)`**<br>($8 \times 1 \times 16 \times 128 = 16,384$) | 节点内跑满 `TP=8` (NVLink)，跨 16 个节点做 `PP=16`，外层 128 路 FSDP。（报告披露训练期间发生了 **148 次 GPU 硬件故障**，容错同样关键！） |
| **[Llama 3 405B](https://arxiv.org/abs/2407.21783) (长文本扩展阶段, $S=128\text{K}$)** | 405B 巨型稠密 | **16,384 张 H100** | **`TP=8, CP=16, PP=16, DP=8 (FSDP)`**<br>($8 \times 16 \times 16 \times 8 = 16,384$) | 当序列长度拉长 16 倍至 128K 时，激活显存暴增！团队立刻**将上下文并行调高至 `CP=16`，同时将 `DP` 降至 `8`**，总卡数不变完美接力 |
| **[Google Gemma 2 (9B / 27B)](https://arxiv.org/abs/2408.00118)** | 稠密模型 | Google **TPU v5p** Pods | **`FSDP (ZeRO-3) + TP + SP`**（**`PP=1`，无流水线！**） | 充分体现 TPU **3D Toroidal Mesh** 优势：在大规模环网网格上直接拉高 TP 与 FSDP，彻底省去复杂的流水线并行与气泡问题 |
| **[DeepSeek-V3 (671B / 37B 激活)](https://arxiv.org/abs/2412.19437)** | 细粒度 MoE + MLA | 2,048 张 H800 | **`PP=16 (DualPipe), EP=64 (跨 8 节点), ZeRO-1 DP`**（**`TP=1`，无张量并行！**） | 颠覆性设计：用 **64 路跨节点专家并行 (`EP=64`)** 彻底取代了 TP（避免矩阵切碎），并用双向流水线 **DualPipe** 将跨节点 EP 的 `All-to-All` 通信完美隐藏在计算之下 |
| **Mixtral 8x22B / [Qwen 3 MoE](https://github.com/NVIDIA/Megatron-LM)** | 粗粒度 / 细粒度 MoE | H100 / B200 集群 | • Mixtral 8x22B: **`EP=8, PP=4, TP=4`**<br>• Qwen 3 MoE: **`EP=32, PP=8, TP=2`** | 现代 MoE 标准范式：对专家层使用大 `EP`（8 或 32），仅保留极小的 `TP`（2 或 4）用于切分 Attention 层权重与激活 |

## 6. 课后自测题与深度解析（直通 Assignment 2 综合设计题）

### 练习 1：ZeRO Stage 1 / 2 / 3 单卡静态显存与激活显存综合精算
**问题**：假设我们要在 **8 台 8 卡 NVIDIA H100 服务器（共 $N_{\text{total}} = 64$ 张 GPU，每张卡 `80 GB` HBM）** 上采用混合精度（`bf16` 参数与梯度，AdamW `fp32` 主权重与一二阶动量）训练一个 **$\Psi = 32\text{B}$（$3.2 \times 10^{10}$ 参数）** 的稠密 Transformer 模型（层数 $L = 64$，隐藏维 $h = 6144$，序列长度 $s = 4096$，单卡微批次 $b = 2$）。
1. 如果仅使用纯数据并行（$N_d = 64$），请分别计算在 **Naive DDP**、**ZeRO-1**、**ZeRO-2** 和 **ZeRO-3 (FSDP)** 下，每张 GPU 上的**静态模型状态显存（Params + Grads + Opt States）**是多少 GB？哪些方案仅靠静态状态就会直接 OOM？
2. 假设开启了 FlashAttention 重计算（消除了 $5\frac{as}{h}$ 二次项），在纯 FSDP（$t=1$）下，64 层累积的**激活值显存**是多少 GB？加上 ZeRO-3 的静态显存（以及临时 `All-Gather` 单层权重的峰值缓冲），单张 80GB H100 能否装下？如果改为节点内 `TP+SP`（$t=8$）配合跨节点 `ZeRO-1`（$N_d=8$），单卡激活显存和静态显存分别变成多少 GB？

**深度解析**：
1. **纯数据并行（$N_d = 64$）下的静态显存对比（$\Psi = 32 \times 10^9$）**：
   - **Naive DDP**：$16 \Psi = 16 \times 32\text{ GB} = \mathbf{512.0\text{ GB}}$（严重超出 80GB，直接 **OOM**！）。
   - **ZeRO-1**：$4 \Psi + \frac{12 \Psi}{64} = 128\text{ GB} + 6\text{ GB} = \mathbf{134.0\text{ GB}}$（仅参数+梯度就达 128GB，依然 **OOM**！）。
   - **ZeRO-2**：$2 \Psi + \frac{14 \Psi}{64} = 64\text{ GB} + 7\text{ GB} = \mathbf{71.0\text{ GB}}$（静态显存已占 71GB，加上激活值必 **OOM**！）。
   - **ZeRO-3 (FSDP)**：$\frac{16 \Psi}{64} = \frac{512\text{ GB}}{64} = \mathbf{8.0\text{ GB}}$（静态显存仅 8GB！）。
2. **动态激活显存与 `TP+SP` 组合优化计算**：
   - **纯 FSDP（$t = 1$）下的 64 层总激活显存**：
     由下界公式 $\text{Act}_{\text{layer}} = \frac{34 s b h}{t}$，代入 $s = 4096, b = 2, h = 6144, t = 1$：
     $$\text{Act}_{\text{layer}} = 34 \times 4096 \times 2 \times 6144 = 1,711,276,032\text{ Bytes} \approx \mathbf{1.711\text{ GB / layer}}$$
     全网 $L = 64$ 层的总激活显存 $= 64 \times 1.711\text{ GB} = \mathbf{109.5\text{ GB}}$！因此哪怕用了 ZeRO-3 把静态显存压到 8GB，在 $b=2$ 下光激活值就高達 `109.5 GB`，单张 80GB H100 依然会 **OOM**（除非对整个 Transformer Block 开启全量激活检查点，或者引入 `TP+SP`）！
   - **开启节点内 `TP+SP`（$t = 8$）+ 跨节点 `ZeRO-1`（$N_d = 8$）**：
     - **单卡激活显存**：被 $t = 8$ 完美均摊，$\text{Act}_{\text{total}} = \frac{109.5\text{ GB}}{8} \approx \mathbf{13.69\text{ GB}}$！
     - **单卡静态显存**：因为参数和梯度已经被 $t = 8$ 切分（每张 TP 卡只负责 $\Psi_{\text{TP}} = \frac{32\text{B}}{8} = 4\text{B}$ 参数），再在 $N_d = 8$ 个节点间对优化器状态做 ZeRO-1 分片：
       $$\text{Static}_{\text{TP8 + ZeRO1}} = 4 \Psi_{\text{TP}} + \frac{12 \Psi_{\text{TP}}}{N_d} = 4 \times 4\text{ GB} + \frac{12 \times 4\text{ GB}}{8} = 16\text{ GB} + 6\text{ GB} = \mathbf{22.0\text{ GB}}$$
     - **总显存占用**：$22.0\text{ GB (静态)} + 13.69\text{ GB (激活)} = \mathbf{35.69\text{ GB} < 80\text{ GB}}$！不仅轻松装入 80GB H100，还留出了充足的显存余量将微批次 $b$ 扩大一倍以进一步提升 MFU！

### 练习 2：流水线气泡率与 Zero-Bubble 原理问答
**问题**：假设使用 $p = 8$ 级流水线并行（Pipeline Parallelism）：
1. 如果你希望将标准 1F1B 调度的流水线气泡率（$\frac{p-1}{m+p-1}$）控制在 **10% 以内**，微批次数量 $m$ 至少需要设为多少？
2. 简述为什么 **Zero-Bubble Pipelining** 能够把反向传播拆分为 $B$ 和 $W$ 两个算子，以及为什么 $B$ 必须优先于 $W$ 执行。

**深度解析**：
1. 令 $\frac{8 - 1}{m + 8 - 1} \le 0.10 \implies \frac{7}{m + 7} \le 0.10 \implies m + 7 \ge 70 \implies \mathbf{m \ge 63}$。即每个流水线周期至少需要切分出 **63 个微批次**（通常会取 $m = 64$），才能将流水线气泡浪费压到 10% 以下！
2. 在反向传播经过线性层 $Y = X W$ 时，计算关于输入激活的梯度 $B$（$\nabla X = \nabla Y W^\top$）与计算关于权重的梯度 $W$（$\nabla W = X^\top \nabla Y$）在数据依赖上是解耦的：上游 Stage $p-1$ 的反向计算严格依赖于 $\nabla X$（因此 $B$ 处于跨卡依赖的关键路径上，必须立即优先算完并发走）；而 $\nabla W$ 只用于本卡最后一步的权重更新，没有任何其他算子依赖它，因此可以将 $W$ 延迟（Defer）到流水线等待上游或下游数据的空闲气泡槽位中执行，从而将气泡填满实现“零气泡”。
