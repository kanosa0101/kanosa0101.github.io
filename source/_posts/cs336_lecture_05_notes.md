---
title: Stanford CS336 (Spring 2026) Lecture 5 深度课程笔记：GPUs, TPUs & Hardware-Aware Optimization
date: '2026-10-25 15:28:30'
categories:
  - 笔记
  - CS336
tags:
  - 大语言模型
  - CS336
  - GPU
  - TPU
  - FlashAttention
  - 性能工程
description: Stanford CS336 深度笔记第 5 讲：深入 GPU/TPU 微架构与硬件加速工程。剖析 SM 调度、四级存储层级与内存墙，详解消除控制流分化、MXFP8/MXFP4、算子融合、重计算、访存合并与分块 Tiling（波量化现象解密），并从零完整推导 FlashAttention 的 Online Softmax 与反向重计算。
mathjax: true
---

# Stanford CS336 (Spring 2026) Lecture 5 深度课程笔记：GPUs, TPUs & Hardware-Aware Optimization

> **课程信息**：[Stanford CS336: Language Modeling from Scratch (Spring 2026)](https://cs336.stanford.edu/)
> **主讲教授**：Tatsunori Hashimoto (Tatsu)
> **本讲资源**：[Lecture 5 YouTube 视频](https://www.youtube.com/watch?v=izZba4UA7iY) | [官方幻灯片 (`lecture_05.pdf`)](https://github.com/stanford-cs336/lectures/blob/main/lecture_05.pdf) | [Horace He 经典博客《Making Deep Learning Go Brrrr》](https://horace.io/brrr_intro.html) | [Google《How to Scale Your Model》GPU/TPU 章节](https://jax-ml.github.io/scaling-book/gpus/)

## 1. 引言：从模型架构进入系统与硬件深水区

在前四讲中，我们完成了语言模型基础组件与前沿架构（MoE、线性/稀疏注意力）的探索。从 Lecture 5 开始，课程正式进入第二大核心模块——**Systems（系统工程与硬件加速，对应 Assignment 2）**。

* **为什么算法研究者也必须懂 GPU 硬件模型？**
  如果不理解底层硬件，当你看到右侧这张真实的“矩阵乘法吞吐率随矩阵维度 $N$ 变化的实测曲线”时，你会覺得像黑魔法一样不可思议：
  1. 为什么矩阵维度只是从 `1792` 增加到 `1793`（仅多了 1 行 1 列），GPU 的实际吞吐率（TFLOP/s）会突然暴跌近一半？
  2. 为什么维度能否被 16 或 32 整除，会决定性地把性能切分成上下两条截然不同的曲线？
  3. 为什么 [Andrej Karpathy 在做 nanoGPT 竞速](https://x.com/karpathy) 时，仅仅把 GPT-2 的词表大小从 `50257` 强行填充（Pad）到 `50304`，整个训练速度就凭空提升了 **25%**？
* **本讲三部曲路线图**：
  1. **Part I：GPU 与 TPU 的硬件解剖与执行模型**（理解 SM、Thread/Warp/Block、存储层次与内存墙）；
  2. **Part II：让 GPU 飞起来的六大核心优化技巧**（消除控制流分化、低精度与 MXFP8/MXFP4、算子融合、重计算、访存合并、分块 Tiling 与波量化现象解密）；
  3. **Part III：终极实战——从零推导 FlashAttention**（将 Tiling、Online Softmax、算子融合与反向传播重计算完美组装在一起）。

## 2. Part I：GPU 与 TPU 硬件解剖与执行模型（Hardware & Execution Model）

### 2.1 算力扩展的物理转折：Dennard Scaling 的终结与大规模并行崛起

* **串行时钟频率扩展的终结（End of Dennard Scaling）**：
  在 1990 年代至 2000 年代初，CPU 性能提升遵循 Dennard 缩放定律——随着晶体管变小，时钟频率（Clock Speed）自动变快，串行指令越跑越快。然而在 2005 年左右，由于漏电流与热功耗密度达到物理极限，虽然摩尔定律让芯片上的晶体管数量继续增长，但**单核时钟频率彻底锁死在了 3～5 GHz**。
* **横向并行扩展（Parallel Scaling）与 GPU 算力超指数起飞**：
  既然不能让单个工人干活更快，唯一的出路就是雇佣成千上万个工人同时干活。观察过去十几年 NVIDIA GPU 的单卡算力演进曲线：从早期的 K20 / M40 / P100，到 **2017 年 V100 首次引入专用矩阵乘法硬件（Tensor Cores）**，再到 A100、H100 与 B200 引入结构化稀疏与 FP8 / FP4 低精度格式，GPU 算力实现了跨数量级的超指数增长。

### 2.2 设计哲学对比：CPU（低延迟导向）vs. GPU（高吞吐导向）

```
[CPU 芯片面积分配：Latency-Oriented (追求单任务极速完成)]
┌───────────────────────────┬───────────┬───────────┐
│     巨型控制单元 (Control) │  ALU 1    │  ALU 2    │
│ (分支预测 / 乱序执行 / 投机)├───────────┼───────────┤
│                           │  ALU 3    │  ALU 4    │
├───────────────────────────┴───────────┴───────────┤
│            大容量低延迟缓存 (L1 / L2 / L3 Cache)    │
└───────────────────────────────────────────────────┘

[GPU 芯片面积分配：Throughput-Oriented (追求海量任务并发总吞吐)]
┌───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┐
│ C │ L1│ALU│ALU│ALU│ALU│ALU│ALU│ALU│ALU│ALU│ALU│ALU│  ◄── SM 0 (流式多处理器)
├───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┤
│ C │ L1│ALU│ALU│ALU│ALU│ALU│ALU│ALU│ALU│ALU│ALU│ALU│  ◄── SM 1
├───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┤
│                     ... (100+ 个 SM)              │
├───────────────────────────────────────────────────┤
│              片上共享 L2 Cache (数十 MB)           │
└─────────────────────────▲─────────────────────────┘
                          │ (高带宽总线, 3~8 TB/s)
┌─────────────────────────┴─────────────────────────┐
│          片外高带宽显存 HBM / Global Memory (80GB+)│
└───────────────────────────────────────────────────┘
```

* **CPU（Latency-oriented，延迟优先）**：专为复杂的串行控制流、高频条件分支（`if-else`）设计。硅片面积大量消耗在复杂的分支预测器（Branch Predictor）、乱序执行控制单元和大容量缓存上，算术逻辑单元（ALU）很少。目标是让单个任务在最短时间内返回。
* **GPU（Throughput-oriented，吞吐率优先）**：专为高度规则、无分支的数据并行计算（如矩阵乘法）设计。砍掉了复杂的控制逻辑，将硅片面积密密麻麻地铺满成千上万个轻量级计算核心。单个线程可能跑得慢、甚至中途被挂起等待内存，但通过**海量线程并发与零开销上下文切换（Latency Hiding，延迟隐藏）**，实现惊人的总体计算吞吐率。

### 2.3 GPU 物理架构解剖：SM（流式多处理器）与四级存储层次

#### 1. 核心计算单元：SM（Streaming Multiprocessor，流式多处理器）
一张 GPU 并不是铁板一块，而是由上百个彼此独立的 **SM（流式多处理器）** 阵列组成的（例如 NVIDIA A100 有 **108 个 SM**，H100 SXM5 有 **132 个 SM**）。每个 SM 内部包含：
* **Warp 调度器与指令派发单元（Schedulers & Dispatchers）**；
* **海量寄存器文件（Register File，每个 SM 256 KB）**；
* **两类计算核心**：
  1. **CUDA Cores（或称 SP，Streaming Processors / FP32 & INT32 ALUs）**：负责常规逐元素标量运算（如加减乘除、激活函数、比较）；
  2. **Tensor Cores（张量核心，自 V100 引入）**：专门用于在一个时钟周期内执行小矩阵乘加（如 $4\times 4$ 或 $16\times 16$ 的 $D = A \times B + C$）的硬化脉动阵列电路。**在现代 GPU 上，Tensor Core 的浮点吞吐率比普通 CUDA Core 高出 10 倍以上**！这就是为什么任何现代深度学习架构如果不能转化为矩阵乘法（MatMul），在 GPU 上就注定快不起来。
* **片上高速存储：L1 Cache / Shared Memory（共享内存，SRAM）**。

#### 2. 决定生死的核心矛盾：GPU 存储层次（Memory Hierarchy）
现代 LLM 系统优化的 90% 都在和内存打交道。以 A100 为例，我们按从快到慢、从近到远排列四级存储：

| 存储层级 | 物理介质与位置 | 容量大小 (以 A100 为例) | 带宽 (Bandwidth) | 访问延迟 (Latency) | 作用域与可编程性 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **1. Registers (寄存器)** | 片上 SM 内部 | 每 SM 256 KB (全卡 ~27 MB) | ~100+ TB/s | **~1 cycle** | 单个 **Thread** 私有，存放局部变量与累加器 |
| **2. Shared Memory / L1 Cache** | 片上 SM 内部 (**SRAM**) | 每 SM 192 KB (全卡 ~20 MB) | ~19 TB/s | **~20–30 cycles** | 同一个 **Block** 内所有线程共享；**Shared Memory 由程序员显式控制读写**，L1 Cache 由硬件自动管理 |
| **3. L2 Cache** | 片上所有 SM 共享 (**SRAM**) | 全卡 40 MB (H100 为 50 MB) | ~4–5 TB/s | **~200 cycles** | 全卡所有 SM 共享的硬件缓存，物理距离比 L1 远 |
| **4. Global Memory (HBM)** | 片外独立显存颗粒 (**DRAM**) | 40 GB / 80 GB (H200 141GB) | 1.5–2.0 TB/s (H100 3.35 TB/s) | **~300–500 cycles** | 全卡共享（即 `nvidia-smi` 看到的显存），**比 L1 慢 10 倍以上**！（再往外则是通过 PCIe/NVLink 连接的 CPU **Host Memory**） |

> **课堂问答：既然 SRAM（Shared Memory）比 DRAM（HBM）快 10 倍以上，为什么不直接用 SRAM 造一张 80GB 的 GPU？**
> **Tatsu 解答**：第一，SRAM 每个比特需要 6 个晶体管（6T）且必须持续通电维持状态，造价昂贵百倍且功耗惊人；第二，SRAM 密度极低，如果堆几百 MB 甚至几 GB，芯片面积会大到无法制造，且物理走线变长后信号传输延迟也会恶化。（注：像 **Groq LPU** 或 **Cerebras 晶圆级芯片** 确实采用了全片上 SRAM 设计，因而在小批量自回归推理上极快，但单卡容量极小，需要成百上千张卡才能装下一个大模型，总拥有成本 TCO 极高）。

### 2.4 GPU 软件执行模型：Thread $\rightarrow$ Warp $\rightarrow$ Block $\rightarrow$ Grid

当你启动一个 GPU 核函数（Kernel）时，逻辑任务是如何映射到上述物理硬件上的？必须牢记三个核心层级与 **SIMT（Single Instruction, Multiple Threads，单指令多线程）** 模型：

```
[软件逻辑视图 (Grid -> Block -> Warp -> Thread)]               [物理硬件映射视图 (GPU -> SM -> Tensor/CUDA Cores)]
┌──────────────────── Grid (整个 Kernel 启动) ────────────────────┐
│  ┌───────────── Block (0, 0) ─────────────┐  ┌── Block (0, 1) ─┐│         ┌─────────── 单个物理 SM ───────────┐
│  │ ┌────── Warp 0 (32 个连续 Thread) ────┐│  │                 ││         │ [Warp Scheduler] (零开销切换就绪束)│
│  │ │ T0  T1  T2  ...  T31 (锁步执行同指令)││  │       ...       │├────────►│ [Registers] (每个 Thread 独占寄存器)│
│  │ └─────────────────────────────────────┘│  └─────────────────┘│ 整个    │ [CUDA Cores & Tensor Cores]       │
│  │ ┌────── Warp 1 (T32 ... T63) ─────────┐│                     │ Block   │ [Shared Memory / L1] (Block 内共享)│
│  │ └─────────────────────────────────────┘│                     │ 绑定到   └───────────────────────────────────┘
│  │   ▼ 共享同一块 Shared Memory (SRAM)    │                     │ 单个 SM!
│  └────────────────────────────────────────┘                     │
└─────────────────────────────────────────────────────────────────┘
```

1. **Thread（线程 —— 最小逻辑单元）**：每个线程拥有自己的寄存器（Registers）和线程索引 `threadIdx`，在各自不同的输入数据上执行代码。
2. **Warp（线程束 —— 硬件调度的原子单位，固定为 32 个连续线程！）**：
   * 在 NVIDIA GPU 上，调度器**不是**按单个线程下发指令的，而是将 **32 个连续编号的线程捆绑成一个 Warp**。
   * **SIMT 锁步执行（Lockstep）**：同一个 Warp 内的 32 个线程在每一拍**必须同时执行完全相同的指令**（只是各自寄存器里的数据不同）。当某个 Warp 因为等待 HBM 显存读取而卡住（Stall）时，SM 的 Warp Scheduler 可以在 1 个周期内零开销切换到另一个已经就绪的 Warp 继续执行，从而实现延迟隐藏（Latency Hiding）。
3. **Block（线程块 —— 共享内存的协作单位，又称 CTA）**：
   * 由多个 Warp 组成（例如一个包含 256 个线程的 Block 包含 $256 / 32 = 8$ 个 Warps）。
   * **核心硬件保证**：**同一个 Block 内的所有线程保证被调度到同一个物理 SM 上运行**（注意：1 个 SM 可以同时驻留多个 Blocks，但 1 个 Block 绝不会跨两个 SM）！因此，同一个 Block 内的线程可以通过该 SM 上的 **Shared Memory（SRAM）** 高速交换和复用数据，并可通过 `__syncthreads()` 同步屏障对齐进度。
4. **Grid（网格）**：一次 Kernel 启动所产生的所有 Blocks 的集合，由 GPU 硬件分配到全卡的 100 多个 SM 上并行执行。

### 2.5 趋同演化：Google TPU 架构与 GPU 的精确映射对比

Tatsu 花了两页幻灯片对比了 Google 的 **TPU（Tensor Processing Unit）**。有趣的是，为了高效解决深度学习矩阵乘法，TPU 与 GPU 展现了惊人的**趋同演化（Convergent Evolution）**，两者的核心组件存在严格的 $1:1$ 对应关系（引自 [Jax Scaling Book](https://jax-ml.github.io/scaling-book/gpus/)）：

| 核心概念维度 | NVIDIA GPU (如 H100) | Google TPU (如 TPU v5p / v6e) | 架构设计取向差异剖析 |
| :--- | :--- | :--- | :--- |
| **基本处理器核心** | **SM (Streaming Multiprocessor)**（H100 有 **132 个** SM） | **TensorCore**（每颗芯片仅 **1～2 个**巨型核心！） | ⚠️ **极易混淆的命名陷阱**：TPU 把整个处理器大核叫做 `TensorCore`，而 GPU 把 SM 里面的矩阵乘法小单元叫做 `Tensor Core`！ |
| **矩阵乘法专用电路** | **Tensor Core**（H100 全卡有 **528 个**小脉动阵列） | **MXU (Matrix Multiply Unit)**（每芯片仅 **4～8 个** $128\times 128$ 巨型脉动阵列） | GPU 走“多而小”路线，灵活度高；TPU 走“少而巨大”的巨型脉动阵列（Systolic Array）路线，控制逻辑极简，但要求 Batch/维度必须大于 64/128 才能塞满 |
| **向量/标量运算单元** | **CUDA Cores (ALU)** | **VPU (Vector Processing Unit)** | 在 TPU 中，逐元素激活函数与归约由 VPU 完成 |
| **片上高速可编程缓存** | **Shared Memory / L1 (SRAM)** | **VMEM / SMEM (Vector/Scalar Memory)** | TPU 的片上 VMEM 容量显著大于单个 SM 的 Shared Memory，且软件编译器（XLA）静态全权管理数据搬运 |
| **片外大容量显存** | **Global Memory (HBM)** | **HBM (High Bandwidth Memory)** | 两者完全一致，均面临严重的内存墙（Memory Wall） |

### 2.6 日益加宽的鸿沟：内存墙（Memory Wall）与算存分离新趋势

观察过去 20 年硬件指标的相对增长率曲线：
* **算力（Tensor Core FLOP/s，灰色线）**每两年增长约 $3\times$；而 **显存带宽（Memory Bandwidth，绿色线）** 和 **跨卡互联带宽（Interconnect，蓝色线）** 每两年仅增长约 $1.4\sim 1.6\times$！
* **前沿硬件与系统应对（课堂问答）**：随着算力与带宽的剪刀差越来越大，工业界在推理端甚至演化出了 **Prefill-Decode 分离架构（PD Disaggregation）** 以及如 **StepFun（阶跃星辰 Step-3）** 所探索的 **Attention-FFN 异构芯片分离部署**——将计算密集型的 Prefill / FFN 放在高 FLOP/s 芯片上，而将极度访存受限的 Decode / Attention 放在高 HBM 带宽或大 SRAM 芯片上！

## 3. Part II：让 GPU 飞起来的六大核心优化技巧（Six Tricks to Go Fast）

回顾 Roofline 模型，我们的目标是尽可能提高算法的**算术强度（Operational / Arithmetic Intensity）**，把算子从左侧受制于带宽的斜坡（Memory-bound）推向右侧的平坦屋顶（Compute-bound），同时消除硬件执行空泡。以下是构建高性能 GPU 代码的六大基石技巧：

### 3.1 技巧一：消除控制流分化（Avoid Control Divergence）

* **机理剖析**：由于同一个 Warp 内的 32 个线程遵循 SIMT 锁步执行，如果你在 Kernel 里写了一个条件分支 `if (x > 0) { do_A(); } else { do_B(); }`，且同一个 Warp 里有的线程满足条件、有的不满足：
  * GPU 硬件**无法同时并行走两条分支**！它必须**先串行执行 `if` 分支**（把走 `else` 的线程用掩码 Mask 屏蔽掉使其空转等待），**再串行执行 `else` 分支**（把走 `if` 的线程屏蔽掉使其空转等待）！
  * 这被称为 **Control Divergence（控制流分化 / Warp Divergence）**，它会直接导致有效并行吞吐率砍半甚至更糟。

```
[无分化 (No Divergence): Warp 内全走同一分支]       [控制流分化 (Control Divergence): Warp 内线程走不同分支]
Warp 指令流:                                        Warp 指令流 (被迫串行走两遍，伴随大量空转掩码 X):
T0  T1  T2  T3                                     T0  T1  T2  T3
│   │   │   │  (全部满足条件)                        │   │   X   X   ◄── 第 1 步: 仅 T0, T1 执行 if 分支 (T2, T3 空转等待!)
▼   ▼   ▼   ▼                                      X   X   │   │   ◄── 第 2 步: 仅 T2, T3 执行 else 分支 (T0, T1 空转等待!)
                                                   ▼   ▼   ▼   ▼   ◄── 第 3 步: 分支汇合，耗时翻倍!
```

* **工程对策**：在编写 GPU 算子时，尽量避免依赖数据的 `if-else` 分支；对于诸如 ReLU、Causal Mask 等操作，优先使用无分支的三元算术掩码乘法（如 `x * (x > 0)` 或内置指令 `max(x, 0.0f)`），或者确保条件分支的边界按 32 线程（Warp）对齐。

### 3.2 技巧二：低精度计算与微缩放格式（Low Precision: FP8, MXFP8 & MXFP4）

#### 1. 双重红利与 Tensor Core 内部累加机制
将张量从 `fp32`（4 Bytes）降为 `bf16`（2 Bytes）再到 `fp8`（1 Byte）：
* **访存红利**：读写相同数量的元素，HBM 搬运字节数直接减半或降为 1/4（算术强度自动翻 2～4 倍！例如向量 ReLU 在 FP32 下每次操作需读写 $4+4=8$ Bytes，而在 FP16 下仅需 $2+2=4$ Bytes）。
* **Tensor Core 混合精度内幕**：当 Tensor Core 执行低精度矩阵乘法（如 `fp16`/`bf16`/`fp8` 输入）时，**它并不是全程用低精度累加的**！硬件内部会在将两个低精度元素相乘后，立即在 **`fp32` 高精度累加器（Accumulator）** 中对部分和进行累加，最后再根据需要输出或降精度（Downcast）。
* **哪些层不能随便量化？（课堂问答）**：实验表明，模型的**第一层（Embedding 附近）和最后一层（LM Head 输出层）**对量化极其敏感（尤其是最后一层直接决定全局词表 Logits 与交叉熵 Loss，量化极易引发失稳和精度暴跌），因此工业界 FP8 训练中通常让首尾层及归一化/Softmax 保持在 `bf16`/`fp32`，仅对中间层的线性 MatMul 启用 FP8。

#### 2. 前沿微缩放格式：`MXFP8` 与 `MXFP4`（Blackwell B200 原生支持）及其“转置陷阱”
在传统 `FP8`（`E4M3` 或 `E5M2`）训练中，人们通常为整个张量维护一个全局 `fp32` 缩放因子（Per-Tensor Scaling Factor）。但这很容易被张量中某一个巨大的离群值（Outlier）撑爆动态范围，导致其余正常数值全部下溢归零。
* **[Microscaling Formats (MXFP8 / MXFP4)](https://arxiv.org/abs/2310.10537) 的核心设计**：
  * **MXFP8**：每 **32 个连续元素** 组成一个微块（Block），共享一个 8-bit 的缩放因子（格式为 `E8M0`，即纯 2 的幂次指数，无尾数）；块内每个元素使用高精度的 `E4M3` 表示。
  * **MXFP4**：每 **16 个连续元素** 共享一个 `E4M3`（8-bit）缩放因子，块内每个元素仅占 4 bits（取值为 `-6, -4, -3, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 3, 4, 6`）。
* **Tatsu 揭示的工程大坑：为什么在 MXFP8 下，简单的“矩阵转置（Transpose）”变成了噩梦？**
  * 在普通精度或 Per-Tensor 缩放下，矩阵转置 $W^\top$ 只是改一下内存步长视图（View），零计算开销。
  * 但在 **MXFP8** 中，缩放因子是沿着**行方向每 32 个相邻元素**绑定的！当你在前向传播用 $W$、而在反向传播需要用转置矩阵 $W^\top$ 时，沿着列方向的 32 个元素原本属于 32 个完全不同的缩放块，**它们根本没有共享的缩放因子**！
  * **工业界解法（双份量化拷贝）**：如最新 MXFP8 训练论文（[arXiv:2506.08027](https://arxiv.org/abs/2506.08027)）所示，为了喂给 Tensor Core，系统在每次量化时被迫**同时在显存中生成并存储两份量化矩阵：一份按行分块量化的 $Q(W)$，另一份先物理转置再按列分块量化的 $Q(W^\top)$**！由于量化与重分块的额外开销，FP8 训练在端到端上的实际加速比通常为 **20%～30%**（而非理论上的 2 倍）。

### 3.3 技巧三：算子融合（Operator Fusion）与编译优化（`torch.compile`）

* **仓库与工厂的传送带比喻**：
  把 HBM 显存想象成远处的“原料仓库”，把 SM 计算核心想象成“加工车间”，连接两者的是一条狭窄缓慢的“传送带（显存带宽）”。
  考虑执行一句简单的 PyTorch 代码：
  ```python
  y = torch.sin(x) ** 2 + torch.cos(x) ** 2
  ```
  * **朴素即时执行（Eager Mode，无融合）**：PyTorch 会依次启动 **5 个独立的 GPU Kernel**：
    1. 读 $x \rightarrow$ 算 $a = \sin(x) \rightarrow$ 写回 HBM；
    2. 读 $a \rightarrow$ 算 $b = a^2 \rightarrow$ 写回 HBM；
    3. 读 $x \rightarrow$ 算 $c = \cos(x) \rightarrow$ 写回 HBM；
    4. 读 $c \rightarrow$ 算 $d = c^2 \rightarrow$ 写回 HBM；
    5. 读 $b, d \rightarrow$ 算 $y = b + d \rightarrow$ 写回 HBM。
    本来极快的计算，硬生生在传送带上往返读写了 **9 次 HBM**！
  * **算子融合（Operator Fusion）**：将这 5 步合并编译为 **1 个超级 Kernel**——从 HBM **只读 1 次 $x$** 进入 SM 寄存器，在寄存器内一口气算完 $\sin(x)^2 + \cos(x)^2$，最后**只写回 1 次 $y$** 到 HBM（仅 **2 次 HBM 访问**）！
* **自动化武器：`torch.compile` / `jax.jit`**：
  对于这类逐元素（Pointwise）和简单归约操作，你无需手写 CUDA，只需加上一行 `@torch.compile`，编译器就会自动分析计算图并生成融合后的 Triton/CUDA 核函数（将在 Lecture 6 深入演示）。

### 3.4 技巧四：重计算（Recomputation / Activation Checkpointing）—— 从带宽视角再审视

在 Lecture 2 中，我们从“节省显存容量（Memory Capacity）”的角度介绍了激活检查点。在本讲中，Tatsu 从**“节省显存带宽以加速运行时间（Runtime Speedup）”**的全新视角再次审视了它：
* 考虑连续堆叠 3 个 Sigmoid 激活：$y = \sigma(\sigma(\sigma(x)))$（令中间结果为 $s_1 = \sigma(x), s_2 = \sigma(s_1), \text{out} = \sigma(s_2)$）。
  * **标准前反向（保存全部中间激活）**：
    * 前向：读 $x$ (1)，写出 $s_1, s_2, \text{out}$ 到 HBM (3) $\implies$ **4 次 HBM 访问**；
    * 反向：读入 $d\text{out}, s_2, s_1$ (3)，写出 $dx$ 到 HBM (1) $\implies$ **4 次 HBM 访问**；
    * **总计：8 次 HBM 读写**！
  * **重计算 + 算子融合（前向不存 $s_1, s_2$，反向在寄存器内即时重算）**：
    * 前向：融合读 $x$ (1)，在寄存器内连过三次 $\sigma$，仅写出 $\text{out}$ (1) $\implies$ **2 次 HBM 访问**；
    * 反向：融合读入 $d\text{out}$ 和原始输入 $x$ (2)，在寄存器内瞬间重算出 $s_1, s_2$ 并算出梯度，写出 $dx$ (1) $\implies$ **3 次 HBM 访问**；
    * **总计：仅 5 次 HBM 读写（减少为原来的 $5/8$）**！
* **深刻洞察**：在算力极度过剩而 HBM 带宽极度匮乏的现代 GPU 上，**多算一次前向传播的耗时，往往比把中间激活写回 HBM 再读出来还要快**！重计算不仅省了显存空间，还能直接提升训练速度！

### 3.5 技巧五：访存合并（Memory Coalescing）与 DRAM 突发传输（Burst）

* **DRAM 的物理突发读取特性（Burst Sections）**：
  片外 HBM（DRAM）的存储单元排列成二维电容阵列。当你请求读取某一个内存地址（如第 0 号字节）时，硬件激活整行放大器需要较长延迟，但一旦激活，**与它物理相邻的一整段连续字节（称为一个 Burst Section / Cache Line，在现代 GPU 上通常为 32 Bytes 或 128 Bytes）会免费同时被读出并传输回来**！
* **什么是访存合并（Memory Coalescing）？**
  当一个 Warp 的 32 个线程同时执行一条读取全局内存的指令时：
  * **合并访问（Coalesced Access —— 极快）**：如果这 32 个线程请求的地址恰好是**连续紧挨着的**，并且对齐在同一个 128 Bytes 的 Burst 扇区内，硬件只需发起 **1 次内存事务（Memory Transaction）** 就能把 32 个线程要的数据全部取回，带宽利用率 100%！
  * **非合并/跨步访问（Uncoalesced / Strided Access —— 极慢）**：如果这 32 个线程请求的地址彼此间隔很远（分别散落在 32 个不同的 Burst 扇区里），硬件被迫发起 **32 次独立的内存事务**，每次取回 128 Bytes 却只用其中 4 Bytes，其余数据全部丢弃，有效显存带宽暴跌为 1/32！
* **行主序（Row-Major）矩阵的黄金口诀**：
  在 PyTorch / C / CUDA 默认的**行主序（Row-Major，同一行的元素在物理内存中连续存放）**矩阵中：
  * **让相邻线程读取同一行的相邻列（沿 Minor axis 移动） $\implies$ 连续地址 $\implies$ 完美 Coalesced！**
  * **让相邻线程读取同一列的相邻行（沿 Major axis 移动，步长为整行宽度 $N$） $\implies$ 跨步地址 $\implies$ 严重 Uncoalesced！**

### 3.6 技巧六：分块计算（Tiling）与矩阵乘法吞吐率曲线大解密

#### 1. 为什么朴素矩阵乘法会卡在 HBM 带宽上？
考虑两个 $N \times N$ 矩阵相乘 $C = A B$。为了算出 $C$ 中的 $N^2$ 个元素，每个元素需要读取 $A$ 的一行（$N$ 个数）和 $B$ 的一列（$N$ 个数）。
* 如果每个线程各自独立直接从 HBM 读取所需的数据，那么 $A$ 和 $B$ 中的**每一个元素都会被重复从 HBM 读取 $N$ 次**！总 HBM 读取量高达 $2 N^3$ 个元素，算术强度仅为 $O(1)$，完全受限于 HBM 带宽。

#### 2. 分块矩阵乘法（Tiled MatMul）如何将 HBM 访问砍掉 $T$ 倍？
我们利用每个 SM 内部的 **Shared Memory（SRAM）**，将大矩阵切分为大小为 $T \times T$ 的小瓦片（Tiles，例如 $T = 128$）：
1. 一个 Thread Block 负责计算输出矩阵 $C$ 的一个 $T \times T$ 子块；
2. 外层循环沿公共维度滑动：每次由 Block 内的线程协作，将 $A$ 的一个 $T \times T$ 子块和 $B$ 的一个 $T \times T$ 子块**从慢速 HBM 一次性合并读取（Coalesced Load）到高速 Shared Memory 中**；
3. 内层循环中，Block 内的线程**在高速 Shared Memory 内部反复读取这两个小块完成 $T \times T$ 矩阵乘加**（每个加载进 SRAM 的元素在片上被复用了 **$T$ 次**！）；
4. 所有分块累加完毕后，将最终的 $C$ 子块一次性写回 HBM。

$$\text{无 Tiling 时每个输入元素的 HBM 读取次数} = N \quad \xrightarrow{\text{引入 } T \times T \text{ Tiling}} \quad \text{每个输入元素的 HBM 读取次数} = \mathbf{\frac{N}{T}}$$

**核心收益**：总 FLOPs 完全没变（仍是 $2N^3$），但**慢速 HBM 的全局内存访问量直接减少了 $T$ 倍（算术强度提升了 $T$ 倍）**！

#### 3. 终极揭秘：为什么实测矩阵乘法吞吐率曲线会出现锯齿、分层与断崖？
现在我们拥有了全部拼图，可以彻底解释本讲开篇那张神秘的 GPU 矩阵乘法性能图了：

```
吞吐率 (TFLOP/s)
  ▲
  │                    1792 (98 Tiles, 1 Wave)
  │                     ●─────────●─────────●  ◄── 维度被 32/16 整除 (完美对齐 Tile 与 Burst, 满血屋顶!)
  │                    ╱           ╲
  │                   ╱             ● 1793 (120 Tiles > 108 SMs, 触发第 2 波 Wave Quantization 断崖暴跌!)
  │        ●─────────●                         ◄── 维度仅被 2 整除 (Burst 跨界错位)
  │       ╱
  │      ●─────────●─────────●─────────●       ◄── 维度为奇数 (仅被 1 整除: 边缘 Tile 空转 + 内存严重不对齐!)
  └──────────────────────────────────────────► 矩阵维度 N
```

1. **整体先斜向上升、后趋于平坦**：直接对应 **Roofline 模型**——小矩阵算术强度低处于斜坡（Memory-bound），大矩阵进入计算饱和区（Compute-bound）。
2. **不同整除性（Divisibility by 1, 2, 8, 16, 32）分裂成多条平行曲线**：
   * **Tile 边界内部浪费（Internal Fragmentation）**：假设硬件最优 Tile 尺寸是 $128 \times 128$。如果矩阵维度是 `256`，刚好切成 $2 \times 2 = 4$ 个满载的 Tile；如果维度变成 `257`，就必须切出 $3 \times 3 = 9$ 个 Tile，其中边缘的 5 个 Tile 里绝大部分线程都在乘以 0 空转！
   * **DRAM 突发扇区错位（Memory Misalignment）**：当矩阵行宽 $N$ 是奇数（不能被 16 或 32 整除）时，从第 2 行开始，每一行的起始地址都偏离了 128 Bytes 的 DRAM Burst 边界！原本对齐时读 1 个 Burst 就能拿齐一行 Tile，错位后每个 Tile 行都跨在两个 Burst 扇区接缝上，导致 HBM 内存事务数量直接翻倍！（这就是为什么 **Karpathy 把词表从 `50257` 补齐到能被 64 整除的 `50304` 后直接提速 25%**！也是为什么 PyTorch `torch.compile(mode="max-autotune")` 会花十几分钟自动为你搜索最佳 Tile 尺寸）。
3. **周期性的断崖式暴跌（如 $N = 1792 \rightarrow 1793$）：波量化（Wave Quantization / Tile Quantization）！**
   * 让我们复盘课堂上的真实算例：在 **NVIDIA A100（拥有 108 个物理 SM）** 上计算 $N \times N$ 矩阵乘法，内核使用的输出 Tile 尺寸为 $256 \times 128$。
   * 当 $N = 1792$ 时：行方向切出 $1792 / 256 = 7$ 块，列方向切出 $1792 / 128 = 14$ 块，总共产生 **$7 \times 14 = 98$ 个 Tiles（Blocks）**。因为 A100 有 **108 个 SM**，这 98 个 Block 可以**在 1 个调度波次（1 Wave）内全部同时塞进 98 个 SM 并行跑完**！
   * 当维度仅仅加 1 变成 **$N = 1793$** 时：行方向需要 $\lceil 1793 / 256 \rceil = 8$ 块，列方向需要 $\lceil 1793 / 128 \rceil = 15$ 块，总块数激增至 **$8 \times 15 = 120$ 个 Tiles**！
   * **灾难发生**：第一波（Wave 1）把 108 个 SM 全部占满跑完了前 108 个 Tile；**剩下的 $120 - 108 = 12$ 个 Tile 被迫开启第二波（Wave 2）**！在第二波执行期间，全卡 108 个 SM 中只有 12 个在干活，其余 **96 个 SM 全部在彻底空转等待**！总耗时几乎翻倍，导致吞吐率在 1793 处出现断崖式跳水！

## 4. Part III：终极胜利巡礼 —— 组装全套技巧推导 FlashAttention

现在，我们将上面学到的 **Tiling（分块复用 SRAM）**、**Operator Fusion（算子融合）**、**Online Softmax（在线分块归一化）** 和 **Recomputation（反向传播重计算）** 组合起来，就能水到渠成地完整推导 [Dao et al. (2022) 的 FlashAttention](https://arxiv.org/abs/2205.14135)！

### 4.1 朴素 Attention 的显存流量灾难

给定 $Q, K, V \in \mathbb{R}^{N \times d}$（其中序列长度 $N \gg$ 头维度 $d$），标准注意力分三步执行：
$$S = Q K^\top \in \mathbb{R}^{N \times N}, \qquad P = \text{Softmax}(S) \in \mathbb{R}^{N \times N}, \qquad O = P V \in \mathbb{R}^{N \times d}$$
* **朴素 PyTorch 实现的致命伤**：
  1. 读 $Q, K$ 算 $S = Q K^\top$，**将 $N \times N$ 的巨型矩阵 $S$ 写回慢速 HBM**；
  2. 从 HBM 读入 $S$ 算 Softmax，**又将 $N \times N$ 的巨型矩阵 $P$ 写回慢速 HBM**；
  3. 从 HBM 读入 $P$ 和 $V$ 算 $O = P V$，将 $O$ 写回 HBM。
  整个过程产生了高达 **$\mathcal{O}(N^2)$ 的 HBM 显存读写流量与中间显存占用**！

### 4.2 核心拦路虎：如何在分块（Tiling）下计算全局耦合的 Softmax？

如果我们想用 3.6 节的 Tiling 技巧：将 $Q$ 切成行块 $Q_i$，将 $K, V$ 切成列块 $K_j, V_j$ 加载进高速 SRAM（Shared Memory）中计算局部得分块 $S_{ij} = Q_i K_j^\top$，并在 SRAM 中直接乘以 $V_j$ 累加出输出 $O_i$，**从而永远不在 HBM 中实例化 $N \times N$ 的矩阵 $S$ 和 $P$**。

但这面临一个数学障碍：**Softmax 是按整行耦合的全局操作**！对于一个行向量 $x = [x_1, \dots, x_N]$，为了防止 $\exp(x_k)$ 数值溢出，数值稳定的 Safe Softmax 必须先扫描完整行求出全局最大值 $m = \max_k x_k$：
$$\text{Softmax}(x_k) = \frac{\exp(x_k - m)}{\sum_{j=1}^N \exp(x_j - m)}$$
当我们还在 SRAM 里处理第一个分块 $K_1$ 时，根本还不知道后续分块 $K_2, K_3, \dots$ 里的最大值是多少，怎么能提前算出归一化输出并乘上 $V_1$ 呢？

### 4.3 破局关键：Online Softmax（在线动态重缩放递推算法）

[Milakov & Gimelshein (2018)](https://arxiv.org/abs/1805.02867) 提出的 **Online Softmax** 解决了这个问题：**我们不需要等看完全行才算指数和，只需维护当前已见元素的运行最大值（Running Max $m^{(j)}$）和运行分母累加和（Running Sum $\ell^{(j)}$），每当遇到新的分块出现更大的最大值时，乘上一个指数修正因子 $\exp(m^{\text{old}} - m^{\text{new}})$ 动态重缩放（Rescale）历史累加值即可！**

设我们在逐块扫描第 $j = 1, 2, \dots, T_c$ 个 Key/Value 分块 $(K_j, V_j)$，对于当前 Query 块，只需在片上寄存器/SRAM 中维护三个极小的运行状态：**当前最大值 $m^{(j)}$**、**当前指数分母和 $\ell^{(j)}$**、以及**当前未完结的输出累加值 $\tilde{O}^{(j)}$**。当读入新分块 $j$ 算出局部得分 $S^{(j)} = Q_i K_j^\top$ 时，单步递推公式为：

1. **更新运行最大值**：
   $$m^{(j)} = \max\Big(m^{(j-1)}, \; \text{rowmax}\big(S^{(j)}\big)\Big)$$
2. **用修正因子 $\exp\big(m^{(j-1)} - m^{(j)}\big)$ 动态修正历史分母并加上当前块指数和**：
   $$\ell^{(j)} = \exp\big(m^{(j-1)} - m^{(j)}\big) \ell^{(j-1)} + \text{rowsum}\Big(\exp\big(S^{(j)} - m^{(j)}\big)\Big)$$
3. **同步动态修正历史输出累加器，并直接乘上当前块 $V_j$（全程在 SRAM 内融合完成！）**：
   $$\tilde{O}^{(j)} = \text{diag}\Big(\exp\big(m^{(j-1)} - m^{(j)}\big)\Big) \tilde{O}^{(j-1)} + \exp\big(S^{(j)} - m^{(j)}\big) V_j$$

当所有 $T_c$ 个分块循环结束时，只需在片上最后除以最终分母 $O_i = \text{diag}\big(\ell^{(T_c)}\big)^{-1} \tilde{O}^{(T_c)}$，将结果一次性写回 HBM！**算出的结果与标准 Attention 逐比特精确等价（并非近似算法！），但 HBM 读写复杂度从 $\mathcal{O}(N^2)$ 彻底降为了 $\mathcal{O}\left(\frac{N^2 d^2}{M}\right)$（其中 $M$ 为 SRAM 大小）！**

### 4.4 反向传播的最后一击：丢弃 $N \times N$ 注意力矩阵，按块重计算（Recomputation）

在前向传播省掉了 HBM 读写之后，反向传播怎么办？按标准自动求导，反向传播计算 $\Q, \K, \V$ 需要用到前向的 $N \times N$ 概率矩阵 $P$。如果前向把 $P$ 存进 HBM，显存又会退化回 $O(N^2)$！
* **FlashAttention 的反向重计算神技**：前向传播结束时，**绝不保存 $N \times N$ 的矩阵 $S$ 或 $P$**，仅仅在 HBM 中保存最终输出 $O$ 以及每行的最终归一化统计量 $(m, \ell)$（仅占 $O(N)$ 显存，即 $\log\text{-sum-exp}$ 统计量 $L = m + \log \ell$）！
* 在反向传播时，各个 SM 再次按块把 $Q_i, K_j, V_j$ 加载进高速 SRAM，利用保存的标量 $L_i$ **在片上 SRAM 中当场重算出局部注意力矩阵 $P_{ij} = \exp(Q_i K_j^\top - L_i)$**，紧接着算完梯度 $\Q_i, \K_j, \V_j$！
* **反直觉但必然的结果**：虽然反向传播因为重算 $Q_i K_j^\top$ 导致**总 FLOPs 增加了约 20%～30%**，但由于彻底消除了读写 $N \times N$ 矩阵的 HBM 流量，**FlashAttention 的反向传播墙钟时间反而比不重算的标准实现快了好几倍**！

## 5. 全讲总结与六大技巧速查表（Summary Cheat Sheet）

| 优化技术 | 解决的核心硬件矛盾 | 核心操作机理 | 代表性工业界应用案例 |
| :--- | :--- | :--- | :--- |
| **1. 消除控制流分化 (No Control Divergence)** | SIMT 32 线程 Warp 锁步执行 | 消除线程间走向不同的 `if-else`，改用无分支掩码算术运算 | 所有高性能 CUDA / Triton 算子 |
| **2. 低精度与微缩放 (Low Precision / MXFP8)** | HBM 带宽瓶颈 + Tensor Core 吞吐 | 用 `bf16`/`fp8`/`nvfp4` 减半搬运字节，配合 `fp32` 累加与分块缩放（需存转置双拷贝） | DeepSeek-V3 (FP8), Nemotron 3 (NVFP4) |
| **3. 算子融合 (Operator Fusion)** | 中间结果反复读写 HBM | 将连续 Pointwise/Reduction 算子合并为单 Kernel，中间值驻留寄存器/SRAM | `torch.compile`, 融合 RMSNorm / SwiGLU |
| **4. 激活重计算 (Recomputation)** | 算力过剩而 HBM 容量与带宽匮乏 | 前向丢弃大体积中间激活，反向在片上从检查点即时重算（以 FLOPs 换带宽） | Activation Checkpointing, FlashAttention 反传 |
| **5. 访存合并 (Memory Coalescing)** | DRAM 128B 突发扇区 (Burst) 读取 | 让同一 Warp 的 32 个相邻线程访问连续对齐的内存地址（行主序沿行读取） | 矩阵转置优化、张量内存布局对齐 |
| **6. 分块计算与维度对齐 (Tiling & Padding)** | 片上 SRAM 小而快 vs. HBM 大而慢 | 将数据分块加载进 Shared Memory 复用 $T$ 次；将维度补齐为 32/64 倍数并规避 SM 波量化 | 矩阵乘法 (GEMM)、FlashAttention、词表 Padding |

## 6. 课后思考与自测练习（Review & Self-Check）

1. **波量化（Wave Quantization）与维度选择实战计算**：
   假设你在单张 **NVIDIA H100 SXM5（拥有 132 个物理 SM）** 上运行一个自定义分块矩阵乘法 $C = A B$，其中输出矩阵 $C \in \mathbb{R}^{M \times N}$，内核固定使用大小为 $128 \times 128$ 的输出 Tile（且假设每个 SM 同一时刻最多只能驻留 1 个活跃的 Thread Block）。
   * (a) 当 $M = 1408, N = 1536$ 时，总共切出多少个 Tiles？需要几个 Wave（波次）？最后一个 Wave 的 SM 利用率是多少？
   * (b) 如果因为多了 1 个特殊 Token，导致矩阵维度变为 $M = 1409, N = 1536$，总 Tiles 数和 Wave 数变成多少？第二个 Wave 的 SM 利用率暴跌到了多少？
   * *参考解答*：
     * (a) 行块数 $1408 / 128 = 11$，列块数 $1536 / 128 = 12$，总块数 $= 11 \times 12 = \mathbf{132\text{ 个 Tiles}}$。刚好等于 H100 的 **132 个 SM**！只需 **1 个 Wave**，SM 满载利用率为 **$132 / 132 = \mathbf{100\%}$**！
     * (b) 当 $M = 1409$ 时，行块数变为 $\lceil 1409 / 128 \rceil = 12$，总块数变为 $12 \times 12 = \mathbf{144\text{ 个 Tiles}}$。由于 $144 > 132$，被迫拆分为 **2 个 Waves**！第 2 个 Wave 只有 $144 - 132 = 12$ 个块在跑，SM 利用率暴跌至 **$12 / 132 \approx \mathbf{9.1\%}$**（90.9% 的芯片算力在空转等待）！
2. **Online Softmax 手算递推验证**：
   假设我们要对长度为 4 的向量 $x = [1, 3, \; 2, 5]$ 分两块计算 Online Softmax：第一块为 $x^{(1)} = [1, 3]$，第二块为 $x^{(2)} = [2, 5]$。请写出第一块结束时的 $(m^{(1)}, \ell^{(1)})$ 以及合入第二块更新后的 $(m^{(2)}, \ell^{(2)})$，并验证其与全局一次性计算的 $\sum_{k=1}^4 \exp(x_k - 5)$ 完全相等。
   * *参考解答*：
     * **第一块 $x^{(1)} = [1, 3]$**：
       $$m^{(1)} = \max(1, 3) = 3, \qquad \ell^{(1)} = e^{1-3} + e^{3-3} = e^{-2} + 1$$
     * **合入第二块 $x^{(2)} = [2, 5]$**：
       1. 更新最大值：$m^{(2)} = \max\big(m^{(1)}, \max(2, 5)\big) = \max(3, 5) = \mathbf{5}$；
       2. 用修正因子 $e^{m^{(1)} - m^{(2)}} = e^{3 - 5} = e^{-2}$ 缩放旧分母并加上新块指数和：
          $$\ell^{(2)} = e^{-2} \cdot \ell^{(1)} + \big(e^{2-5} + e^{5-5}\big) = e^{-2}(e^{-2} + 1) + e^{-3} + 1 = e^{-4} + e^{-2} + e^{-3} + 1$$
       这与全局直接以最大值 $5$ 计算的 $\sum_{k=1}^4 e^{x_k - 5} = e^{1-5} + e^{3-5} + e^{2-5} + e^{5-5}$ **恒等**！
3. **衔接下一讲（Lecture 6: Kernels & Triton）**：
   在理解了本讲的硬件原理与 FlashAttention 算法后，下一讲 Percy 将带我们在代码中真正动手——使用性能剖析器（Profiler）、查看 PTX 汇编，并用 **Triton** 编写融合 GeLU、Online Softmax 与分块矩阵乘法（为 Assignment 2 手写 Triton FlashAttention-2 打下完整代码基础）。
