---
title: Stanford CS336 (Spring 2026) Lecture 6 深度课程笔记：Kernels & Triton
date: '2026-10-25 15:29:00'
categories:
  - 笔记
  - CS336
tags:
  - 大语言模型
  - CS336
  - Triton
  - CUDA
  - GPU
  - Kernel编程
description: Stanford CS336 深度精读笔记第 6 讲：基于 Triton 的高性能 GPU Kernel 实战。从 A100/H100/B200 硬件物理约束（Occupancy、Bank Conflict、Coalescing）出发，实测对比 GeLU 算子融合的 11 倍加速，并四级进阶手写 Triton 算子：逐元素融合、行归约 Softmax、跨块循环规约与二维分块矩阵乘法（tl.dot）。
mathjax: true
---

# Stanford CS336 (Spring 2026) Lecture 6 深度课程笔记：Kernels & Triton

**课程信息**：[Stanford CS336: Language Modeling from Scratch (Spring 2026)](https://cs336.stanford.edu/)  
**主讲人**：Percy Liang  
**官方可执行讲义**：[Lecture 06 Interactive Trace (`lecture_06.py`)](https://cs336.stanford.edu/lectures/?trace=lecture_06)  
**关联作业**：Assignment 2 — Systems & Parallelism（Profiling、Triton Fused RMSNorm、Triton FlashAttention-2）

## 1. 课程全景导读：为什么我们需要手写 GPU Kernel？

在 [Lecture 5（GPU 架构与性能工程）](https://www.google.com/search?q=https://cs336.stanford.edu/lectures/%3Ftrace%3Dlecture_05) 中，我们确立了大模型系统优化的核心物理定律：现代 GPU 的算力增长远快于显存带宽增长（**Memory Wall**），导致除大型矩阵乘法（GEMM）外的大量算子（如激活函数 GeLU/SwiGLU、归一化 LayerNorm/RMSNorm、Softmax 以及 Attention 核心循环）均受限于 **HBM（High Bandwidth Memory）显存带宽**，即处于 **Memory-Bound（访存受限）** 状态。

本讲（[Lecture 6](https://cs336.stanford.edu/lectures/?trace=lecture_06)）作为纯可执行代码讲义（Executable Lecture），从“知其然”正式迈向“动手写”。我们将完整走通高性能算子研发的闭环工作流：
1. **硬件微架构与编程模型深潜（`review_of_gpus`）**：对比 NVIDIA 三代数据中心旗舰 GPU（A100、H100、B200）的硬核参数，深入剖析 **Warp 占用率（Occupancy）与寄存器压力**、**Shared Memory 32-Bank 冲突（Bank Conflicts）与 Swizzling 交错寻址**、**HBM 访存合并（Coalescing）** 以及 **Wave Quantization（波次量化效应）**。
2. **科学测速与内核级剖析（`benchmarking_and_profiling`）**：掌握基于 `torch.cuda.Event` 的精准异步计时范式，以及利用 `torch.profiler` 透视 PyTorch 算子底层真正调用的 C++/CUTLASS 核函数签名。
3. **算子融合的威力对比（GeLU 四重奏）**：在同一硬件上正面对比 `naive_gelu`（9 次 HBM 往返）、`builtin_gelu`（手写 C++ 融合核）、`cuda_gelu`（`load_inline` 内联 CUDA C++ 核）与 `compiled_gelu`（`torch.compile` 自动生成 Triton 融合核），直观见证 **~11 倍** 的端到端加速。
4. **Triton 编程四级进阶实战**：跨越从 CUDA 的 **Thread-Level（线程级 SIMT）** 到 OpenAI Triton 的 **Block-Level（分块级 Tile）** 编程范式，逐行精讲并拆解 PTX 汇编与四个核心递进算子：
   - **Level 1（逐元素融合）**：`triton_gelu_kernel`（一维网格、边界 Masking、数学恒等变换与 PTX 线程粗化剖析）
   - **Level 2（单块行归约融合）**：`triton_softmax_kernel`（将 $8MN$ 次 HBM 访存压缩至 $2MN$ 次，实现 4 倍理论带宽加速）
   - **Level 3（跨分块循环归约 / Baby Tiling）**：`row_sum_kernel`（当单行宽度 $N > \text{BLOCK\\_SIZE}$ 时的片上累加器循环与最终规约）
   - **Level 4（二维分块矩阵乘法 + 激活融合）**：`matmul_relu_kernel`（二维网格划分、二维广播指针算术、沿 $K$ 维滑动窗口 `tl.dot` 累加与片上 ReLU 融合）

```
+---------------------------------------------------------------------------------------------------+
|                         CS336 Lecture 6: GPU Kernel 优化与 Triton 研发闭环                          |
+---------------------------------------------------------------------------------------------------+
|                                                                                                   |
|  [1. 硬件物理约束 (Hardware Limits)]             [2. 瓶颈诊断 (Benchmarking & Profiling)]           |
|   • 寄存器上限 (65536/SM) -> Warp Occupancy       • torch.cuda.Event + synchronize 精准计时         |
|   • Shared Mem 32 Banks -> Bank Conflict/Swizzle  • torch.profiler 透视底层 CUTLASS / Vector 核函数 |
|   • 128B Cache Line     -> Memory Coalescing      • 识别 Memory-Bound 算子的多次 HBM 读写往返        |
|   • #Blocks mod #SMs    -> Wave Quantization                                                      |
|                                  |                                                                |
|                                  v                                                                |
|  [3. 自动化基准 (Quick Wins)]                    [4. 手写 Triton 算子四级进阶 (Block-Level)]         |
|   • torch.compile(fn)   -> 自动生成 Triton 融合核   • L1 Elementwise : triton_gelu_kernel (+ PTX)    |
|   • 零手写代码获得 ~10x 逐元素加速                   • L2 1-Block Row : triton_softmax_kernel (4x BW) |
|                                                   • L3 Multi-Tile  : row_sum_kernel (Baby Tiling) |
|                                                   • L4 2D Tiling   : matmul_relu_kernel (O(B) AI) |
+---------------------------------------------------------------------------------------------------+
```

## 2. GPU 硬件微架构与执行模型深潜（`review_of_gpus`）

在编写任何一行 Kernel 代码之前，系统工程师必须对目标 GPU 的存储层次容量、各级带宽以及硬件调度约束建立精确的量级直觉。

### 2.1 三代数据中心旗舰 GPU 核心规格全对比（A100 vs. H100 vs. B200）

[Lecture 6 官方讲义](https://cs336.stanford.edu/lectures/?trace=lecture_06) 给出了 NVIDIA Ampere（A100）、Hopper（H100）与 Blackwell（B200）三代架构在存储容量与读写带宽上的权威对比表（课程演示环境运行于 **NVIDIA B200** 上）：

| 硬件指标 / 架构代际 | [NVIDIA A100](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-us-nvidia-1758950-r4-web.pdf) (Ampere, `sm80`) | [NVIDIA H100](https://resources.nvidia.com/en-us-hopper-architecture/nvidia-h100-tensor-c) (Hopper, `sm90`) | [NVIDIA B200](https://resources.nvidia.com/en-us-blackwell-architecture/datasheet) (Blackwell, `sm100`) | 架构演进规律与 Kernel 编程启示 |
| :--- | :--- | :--- | :--- | :--- |
| **SM 数量 (`# SMs`)** | `108` | `132` | `148` | Grid 的 Thread Block 总数应尽量是 `# SMs` 的整数倍 |
| **寄存器容量 (`Register size / SM`)** | `256 KB` | `256 KB` | `256 KB` | 每个 SM 恒为 `65,536` 个 32-bit 寄存器，决定活跃 Warp 上限 |
| **L1 Cache + Shared Memory / SM** | `192 KB` | `256 KB` | `256 KB` | 决定单 Block 分块（Tile Size）装入片上 SRAM 的物理上限 |
| **L2 Cache 总容量** | `40 MB` | `50 MB` | `96 - 126 MB` | 所有 SM 共享，用于吸收跨 Block 的数据重用 |
| **HBM 显存总容量** | `80 GB` | `80 GB` | `192 GB` | 存放模型权重、优化器状态、KV Cache 与全局激活张量 |
| **寄存器带宽 (`Register BW`)** | `~116 TB/s` | `~401 TB/s` | `~447 TB/s` | 带宽天花板（比 HBM 快 **55~120 倍**），累加器必须驻留寄存器 |
| **L1 + Shared Memory 带宽** | `~19 TB/s` | `~33 TB/s` | `~40 TB/s` | 比 HBM 快 **5~10 倍**，是二维 Tiling 数据复用的核心舞台 |
| **L2 Cache 带宽** | `~5 - 8 TB/s` | `~12 TB/s` | `~17 TB/s` | 比 HBM 快 **2~4 倍**，通过 Block 调度顺序（L2 Swizzling）优化 |
| **HBM 全局显存带宽** | `2.0 TB/s` | `3.35 TB/s` | `8.0 TB/s` | 片外最慢瓶颈，Kernel Fusion 的唯一目标就是减少 HBM 读写次数 |

**现代架构的两个重要扩展硬件特性**：
- **Blackwell 专属 Tensor Memory (TMEM)**：在 [NVIDIA B200](https://resources.nvidia.com/en-us-blackwell-architecture/datasheet) 上，除了常规 Register File 和 Shared Memory 外，每个 SM 新增了专为第 5 代 Tensor Core 服务的 **Tensor Memory（TMEM）**，层级位于寄存器与 Shared Memory 之间，专门用于存放矩阵乘法累加结果，缓解通用寄存器堆的压力。
- **Thread Block Clusters（线程块簇）**：从 Hopper（H100）和 Blackwell（B200）开始，硬件支持将多个相邻 SM 上的 Thread Block 编组为一个 **Cluster**，允许同一个 Cluster 内的不同 Block 直接互相读写对方的 Shared Memory（即 **Distributed Shared Memory, DSMEM**），无需绕道 L2 Cache 或 HBM。

```
+===================================================================================================+
|                        NVIDIA GPU 存储层次结构与执行单元映射 (以 B200 为例)                          |
+===================================================================================================+
|                                                                                                   |
|   [Thread (线程)]  ----->  私有 Registers (256 KB/SM, ~447 TB/s, <= 255 regs/thread)               |
|         |                 (+ B200 专属 Tensor Memory TMEM，供 Tensor Cores 直接读写累加)            |
|         v                                                                                         |
|   [Warp (32 线程)] ----->  锁步执行 (SIMT Lockstep)，零开销上下文切换隐藏 HBM 访存延迟                  |
|         |                                                                                         |
|         v                                                                                         |
|   [Thread Block]  ----->  共享 L1 Cache / Shared Memory (256 KB/SM, ~40 TB/s, 32 Banks x 4B)      |
|         |                 (H100/B200 支持 Thread Block Cluster 跨 SM 访问 Distributed Shared Mem)  |
|         v                                                                                         |
|   [Grid (全部块)] ----->  共享 L2 Cache (96-126 MB, ~17 TB/s) <---> HBM 全局显存 (192 GB, 8 TB/s)   |
|                                                                                                   |
+===================================================================================================+
```

### 2.2 硬件与编程模型的四大交互机制

为什么同样逻辑的 Kernel，不同写法性能相差数十倍？Percy Liang 在 `review_of_gpus()` 中总结了硬件执行模型对代码的四大核心约束：

#### （1）Warp 锁步执行、控制流分化（Control Divergence）与零开销延迟隐藏
- **物理执行粒度**：硬件每次以 **Warp（32 个连续线程）** 为基本调度单位，在同一周期执行相同的指令（SIMT）。如果你编写了 `if (threadIdx.x < 4) { A } else { B }`，整个 Warp 的 32 个线程必须先串行执行分支 `A`（此时后 28 个线程被硬件掩码静默屏蔽），再串行执行分支 `B`（前 4 个线程被屏蔽），这就是 **Control Divergence（控制流分化）**。
- **零开销 Warp 切换（Latency Hiding）**：CPU 切换线程需要保存寄存器到内存（微秒级开销）；而 GPU 的所有驻留 Warp 的寄存器都**静态分区并常驻在 SM 的 256 KB 寄存器堆中**。当 Warp 0 发起一条耗时数百周期的 HBM 读取指令时，SM 的 Warp Scheduler 可以在**下一个时钟周期（零开销）**立即切换到已经就绪的 Warp 1 执行算术指令。

#### （2）Warp 占用率（Warp Occupancy）与寄存器压力定量计算
既然 SM 靠切换驻留 Warp 来隐藏 HBM 访存延迟，那么**每个 SM 上同时驻留的活跃 Warp 数量越多，延迟隐藏能力就越强**。
- **Occupancy（占用率）定义**：
  $$\text{Warp Occupancy} = \frac{\text{当前 SM 上实际并发驻留的活跃 Warps 数量}}{\text{单个 SM 硬件支持的最大 Warps 上限 } (\text{Max Warps per SM} = 64)}$$
- **三大硬件瓶颈制约 Occupancy**：每个 SM 的物理资源是固定的——最大 `64` 个 Warps（即 `2048` 个线程）、最大 `256 KB` 寄存器堆（即 $256 \times 1024 / 4 = 65,536$ 个 32-bit 寄存器，且单线程最多使用 `255` 个寄存器）、以及最大 `192~256 KB` 的 Shared Memory。**如果每个线程使用的寄存器过多，或者每个 Block 申请的 Shared Memory 过大，SM 能同时容纳的 Block 数量就会断崖式下跌！**
- **讲义精确算例推导（`lecture_06.py` Lines 98–112）**：
  假设我们编写了一个重度使用片上寄存器的 Kernel，配置如下：
  - 每个 Thread Block 包含线程数：`num_threads_per_block = 128`（即每个 Block 含 $128 / 32 = 4$ 个 Warps）
  - 编译器为每个线程分配的寄存器数：`num_registers_per_thread = 160`（满足 $\le 255$ 硬件限制）
  - 单个 SM 拥有的 32-bit 寄存器总数：`max_registers = 65536`
  - 单个 SM 支持的最大 Warp 总数：`max_warps = 64`

```python
# 来自 lecture_06.py: 寄存器压力如何扼杀 Warp Occupancy
max_registers = 65536
max_warps = 64

num_threads_per_block = 128
num_registers_per_thread = 160  # <= 255

# 1. 计算单个 Thread Block 消耗的寄存器总数
num_registers_per_block = num_threads_per_block * num_registers_per_thread  # 128 * 160 = 20,480 个寄存器

# 2. 计算单个 SM 的寄存器堆最多能同时驻留多少个 Thread Block（向下取整）
num_blocks = max_registers // num_registers_per_block                       # 65536 // 20480 = 3 个 Block (余 4096 闲置)

# 3. 计算该 SM 上实际驻留的活跃 Warps 数量
num_warps = num_blocks * num_threads_per_block // 32                        # 3 * 128 // 32 = 12 个 Warps

# 4. 计算最终的 Warp Occupancy
occupancy = num_warps / max_warps                                           # 12 / 64 = 0.1875 (仅 18.75%!)
```

- **辩证思考：低 Occupancy 一定意味着低性能吗？**
  **不一定！** Percy Liang 特别强调：虽然高 Occupancy 有助于隐藏延迟，但如果一个线程使用更多寄存器是因为它承担了更多的数据复用与独立计算任务（即 **Thread Coarsening / 线程粗化**，例如一个线程同时处理 8 个元素或维护更大的寄存器 Tile），那么即使 Occupancy 只有 18.75% 甚至 25%，由于大幅削减了 Shared Memory/HBM 访存指令并提高了指令级并行度（ILP），整体吞吐量反而可能更高。

#### （3）Shared Memory 的 32-Bank 冲突（Bank Conflicts）与 Swizzling 交错寻址
Shared Memory 之所以拥有高达 `~40 TB/s`（B200）的超高片上带宽，是因为它在硬件物理实现上被切分成了 **32 个独立的存储体（Banks，`B00` ~ `B31`）**：
- **物理映射规则**：每个 Bank 的位宽为 **4 Bytes（32-bit，恰好是一个 `float32` 或 `int32`）**。连续的 4 字节地址按模 32 轮询映射到不同 Bank：即第 $i$ 个 `float32` 元素映射到 $\text{Bank } (i \bmod 32)$。
- **无冲突并发 vs. Bank Conflict 串行化**：在每一个时钟周期内，如果同一个 Warp 的 32 个线程分别访问 **32 个不同 Bank** 中的地址，硬件可以在 **1 个周期内** 并行完成全部 32 次读写！反之，如果有 $k$ 个线程在同一周期访问了**同一个 Bank 内的不同地址**，硬件被迫将这 $k$ 次访问拆分为 $k$ 个周期**串行执行**，吞吐量瞬间跌至 $1/k$（称为 **$k$-way Bank Conflict**）。*注：若多个线程读取同一个 Bank 的完全相同地址，硬件会触发 Broadcast（广播），不产生冲突。*
- **经典灾难场景：按列读取 $32 \times 32$ 矩阵触发 32-Way Bank Conflict**：
  假设我们在 Shared Memory 中存放了一个 $32 \times 32$ 的 `float32` 矩阵。因为每行恰好有 32 个元素（跨度 `stride = 32`），所以**每一列的全部 32 个元素在内存中相隔 32 的整数倍，它们全部落在同一个 Bank 上！**

```
+---------------------------------------------------------------------------------------------------+
|               Shared Memory 32-Bank 映射与按列读取冲突 (32x32 float32 矩阵, Stride=32)               |
+---------------------------------------------------------------------------------------------------+
|                                                                                                   |
|  [原始线性布局 (Naive Layout): col_bank = col % 32]                                                |
|         Col 0   Col 1   Col 2   Col 3  ...  Col 31                                                |
|  Row 0: [B00]   [B01]   [B02]   [B03]  ...  [B31]   <-- 按行读 (Thread t 读 Row 0, Col t): 无冲突! |
|  Row 1: [B00]   [B01]   [B02]   [B03]  ...  [B31]                                                 |
|  Row 2: [B00]   [B01]   [B02]   [B03]  ...  [B31]                                                 |
|  Row 3: [B00]   [B01]   [B02]   [B03]  ...  [B31]                                                 |
|   ...    ...     ...     ...     ...   ...   ...                                                  |
|         ^^^^^                                                                                     |
|         按列读 (Thread t 读 Row t, Col 0): 32 个线程同时挤爆 [B00] -> 32-Way Bank Conflict (慢 32 倍)|
|                                                                                                   |
|  [Swizzling 交错布局 (XOR / 周期错位): bank = (row + col) % 32 或 row ^ col]                        |
|         Col 0   Col 1   Col 2   Col 3  ...  Col 31                                                |
|  Row 0: [B00]   [B01]   [B02]   [B03]  ...  [B31]                                                 |
|  Row 1: [B01]   [B02]   [B03]   [B04]  ...  [B00]                                                 |
|  Row 2: [B02]   [B03]   [B04]   [B05]  ...  [B01]                                                 |
|  Row 3: [B03]   [B04]   [B05]   [B06]  ...  [B02]                                                 |
|         ^^^^^                                                                                     |
|         按行读与按列读均分散在 [B00, B01, B02, B03...] 不同 Bank -> 零 Bank Conflict!               |
+---------------------------------------------------------------------------------------------------+
```

- **为什么矩阵乘法（MatMul）极易遭遇 Bank Conflict？如何解决？**
  在计算 $C = A B$ 时，我们需要按**行**读取矩阵 $A$，同时按**列**读取矩阵 $B$（或者在反向传播中转置读取同一矩阵）。如果按常规行优先存储，必有一方遭遇严重的按列读取 Bank Conflict。
  工业界有两种标准解法：
  1. **Padding（填充法）**：将每行声明为 33 列（即 `[32][33]`，多塞 1 个 dummy 元素），使得下一行首元素自动错开 1 个 Bank，但由于破坏了 128 字节内存对齐，无法配合向量化指令与 Tensor Core 异步拷贝（`cp.async` / TMA）。
  2. **Swizzling（地址交错重排，现代主流）**：不浪费任何额外空间，通过将行号与列号做周期偏移或按位异或（如物理列号 $c' = c \oplus r$ 或 $(c + r) \bmod 32$），使得无论按逻辑行读还是按逻辑列读，32 个线程访问的物理 Bank 均两两不同！**这也是我们使用 Triton 的一大核心红利：Triton 编译器会自动在底层 PTX 中插入 Swizzling 地址变换，无需程序员手动计算异或掩码。**

#### （4）HBM 显存合并访问（Memory Coalescing）与波次量化（Wave Quantization）
- **Memory Coalescing（访存合并）**：当 CPU/GPU 从 HBM 读取数据时，DRAM 控制器每次交互的最小物理事务粒度是 **32 Bytes Sector / 128 Bytes Cache Line**。如果一个 Warp 的 32 个线程同时读取一维数组中**连续相邻的 32 个 `float32`（共计 $32 \times 4 = 128\text{ Bytes}$）**，硬件内存控制器会将这 32 个请求**合并（Coalesce）为单次 128B 缓存行读取事务**，打满 HBM 带宽；反之，若 32 个线程跨步（Strided）或随机读取，则会触发多达 32 次独立的内存事务，有效带宽损失超过 90%。
- **Block Occupancy 与 Wave Quantization（波次量化 / 尾部效应）**：
  在 [NVIDIA B200](https://resources.nvidia.com/en-us-blackwell-architecture/datasheet) 上共有 `148` 个 SMs。假设一个 Thread Block 占满了单个 SM 的资源（每个 SM 同一时刻只能跑 1 个 Block），而你的 Kernel 启动了 `160` 个 Thread Blocks：
  - **Wave 1（第一波）**：前 `148` 个 Blocks 在 148 个 SMs 上满载并行，SM 利用率 **100%**；
  - **Wave 2（尾部波）**：剩下的 $160 - 148 = 12$ 个 Blocks 只能在 12 个 SMs 上运行，其余 **136 个 SMs 完全空转等待**，这一波的硬件利用率暴跌至 $12 / 148 \approx 8.1\%$！因此，划分 Grid 时应尽量让总 Block 数接近 `# SMs` 的整数倍（或在 Persistent Kernel 中启动恰好等于 `# SMs` 倍数的常驻 Block 循环消费任务队列）。

## 3. 科学测速（Benchmarking）与内核级剖析（Profiling）

要优化算子，第一步是建立不受异步调度干扰的测量基准（`benchmarking_and_profiling`）。

### 3.1 为什么不能直接用 `time.time()`？标准 CUDA Event 测速模板

由于 GPU Kernel 的发射（Launch）是**完全异步（Asynchronous）**的，CPU 将指令塞入 CUDA Stream 队列后立即返回下一行 Python 代码。因此，科学的 GPU 基准测试必须满足两个铁律：**（1）预热（Warmup）**以消除 JIT 编译、CUDA 上下文初始化及显存分配器冷启动开销；**（2）显式同步（`torch.cuda.synchronize()`）或使用硬件级 `torch.cuda.Event` 时间戳**。

以下是 `lecture_06.py`（Lines 182–204）给出的工业级测速函数实现：

```python
import torch
from typing import Callable

def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs)

def benchmark(description: str, run: Callable, num_warmups: int = 1, num_trials: int = 3) -> float:
    """使用 CUDA Event 精确测量函数 `run` 在 GPU 上的执行时间（单位：ms）。"""
    # 1. 预热阶段 (Warmup)：触发 JIT 编译、分配缓存并让 GPU 时钟频率升至稳定状态
    for _ in range(num_warmups):
        run()
    if torch.cuda.is_available():
        torch.cuda.synchronize()  # 必须同步，确保预热指令全部执行完毕后再开始正式计时

    # 2. 正式计时阶段 (Trials)
    times: list[float] = []
    for trial in range(num_trials):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()      # 在当前 CUDA Stream 中插入起始硬件时间戳事件
        run()                     # 发射被测算子
        end_event.record()        # 在当前 CUDA Stream 中插入结束硬件时间戳事件

        torch.cuda.synchronize()  # 阻塞 CPU，直到 GPU 完成 end_event 之前的所有流操作
        elapsed_ms = start_event.elapsed_time(end_event)  # 计算两个硬件事件之间的精确毫秒差
        times.append(elapsed_ms)

    mean_time = mean(times)
    return mean_time
```

**矩阵乘法（Square MatMul $N \times N$）维度扩展实验规律**：
当我们在 `dims = [256, 512, 1024, 2048, 4096, 8192]` 上对 `a @ b` 进行 Benchmark 时，可以清晰观察到两个阶段：
- **小矩阵区间（$N \le 512$，Launch / Latency Bound）**：耗时几乎保持恒定（几微秒到十几微秒），因为此时 GPU 的 148 个 SMs 根本喂不饱，总耗时被 CPU 发射开销（`cuLaunchKernelEx`）和固定内存延迟主导。
- **大矩阵区间（$N \ge 2048$，Compute Bound）**：随着维度 $N$ 翻倍，FLOPs（$2N^3$）增加 **8 倍**，实测耗时也呈现出教科书般的 $O(N^3)$ 立方级增长。

### 3.2 使用 `torch.profiler` 透视 PyTorch 底层究竟调用了什么核函数

Benchmarking 只能告诉我们“总共花了多少毫秒”（黑盒端到端时间），而 **Profiling（性能剖析）** 不仅能告诉我们“时间花在了哪个环节”，更能直接揭开 PyTorch 的黑盒，看到底层的 C++/CUDA 核函数真名！

```python
from torch.profiler import ProfilerActivity

def profile(run: Callable, num_warmups: int = 1) -> str:
    # 1. 预热并同步
    for _ in range(num_warmups):
        run()
    torch.cuda.synchronize()

    # 2. 开启 PyTorch Profiler 捕获 CUDA 硬件活动，启用 verbose=True 查看完整核函数符号与模板参数
    with torch.profiler.profile(
        activities=[ProfilerActivity.CUDA],
        experimental_config=torch._C._profiler._ExperimentalConfig(verbose=True)
    ) as prof:
        run()
        torch.cuda.synchronize()

    # 3. 按 CUDA 总耗时降序输出汇总表
    table = prof.key_averages().table(
        sort_by="cuda_time_total",
        max_name_column_width=100,
        row_limit=10
    )
    return table
```

在 [Lecture 6 官方 Trace](https://cs336.stanford.edu/lectures/?trace=lecture_06)（运行于 NVIDIA B200）中，对三个基础算子运行 `profile()` 得到了如下极具启发性的底层剖析结果：

| 被测 PyTorch 操作 | 底层实际调用的 CUDA Kernel 完整符号名 | CPU Launch 耗时 | GPU CUDA 实际耗时 | 调用次数 | 底层实现架构深度解码 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`a + b` (`dim=2048`)** | `void at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctor_add, ...>` | `44.216 us` (`cudaLaunchKernel`) | **`4.833 us`** | `1` | PyTorch 原生向量化逐元素模板核，`<4, ...>` 表示每个线程使用 128-bit 向量化指令一次加载/处理 **4 个 `float32`**。注意此时 **CPU 发射耗时 (44.2us) 是 GPU 实际计算耗时 (4.8us) 的 9 倍以上！** |
| **`a @ b` (`dim=2048`)** | `cutlass3x_sm100_simt_sgemm_f32_f32_f32_f32_f32_64x64x16_1x1x1_3_nnn_align1_bias_f32_relu` | `24.828 us` (`cuLaunchKernelEx`) | **`330.366 us`** | `1` | 调用 NVIDIA **CUTLASS 3.x** 线性代数库；`sm100` 代表 Blackwell (B200) 架构；`sgemm_f32` 代表 FP32 矩阵乘；**`64x64x16`** 为 Thread Block Tile 尺寸（$M_{\text{tile}}=64, N_{\text{tile}}=64, K_{\text{tile}}=16$）；`3` 代表 3 级软件流水线（Stages）。 |
| **`a @ b` (`dim=128`)** | `cutlass3x_sm100_simt_sgemm_f32_f32_f32_f32_f32_32x32x16_1x1x1_3_nnn_align1_bias_f32_relu` | `27.125 us` (`cuLaunchKernelEx`) | **`4.544 us`** | `1` | 当矩阵维度缩小至 $128 \times 128$ 时，如果仍用 `64x64` 分块，总共只有 $2 \times 2 = 4$ 个 Block，148 个 SM 有 144 个闲置！因此 cuBLAS 启发式调度器**动态切换到了更小的 `32x32x16` 分块核函数**以提升并行度。 |

## 4. 算子融合（Kernel Fusion）的威力：GeLU 激活函数四重奏

现在我们进入本讲的第一个高潮：以 Transformer FFN 层广泛使用的 **GeLU（Gaussian Error Linear Unit）** 激活函数（采用 `tanh` 近似公式）为例，剖析为什么朴素 PyTorch 代码会慢上十倍，以及如何通过不同技术路线实现算子融合（`naive_vs_builtin_vs_compiled_gelu`）。

GeLU 的数学定义（`tanh` 近似版本，其中常数 $\sqrt{2/\pi} \approx 0.79788456$）为：
$$\text{GeLU}(x) = 0.5 x \left(1 + \tanh\left(\sqrt{\frac{2}{\pi}} \left(x + 0.044715 x^3\right)\right)\right)$$

### 4.1 四种实现方式代码对照（Naive vs. Built-in vs. Inline CUDA C++ vs. `torch.compile`）

#### 方式 1：朴素 PyTorch 逐算子拼接（`naive_gelu`，未融合）
```python
def naive_gelu(x: torch.Tensor) -> torch.Tensor:
    # 每一个算术运算符 (*, +, torch.tanh) 都会立即触发一次独立的 CUDA Kernel 发射与 HBM 读写往返!
    return 0.5 * x * (1 + torch.tanh(0.79788456 * (x + 0.044715 * x * x * x)))
```

#### 方式 2：PyTorch 官方内置 C++ 算子（`builtin_gelu`，预编译融合核）
```python
def builtin_gelu(x: torch.Tensor) -> torch.Tensor:
    # 底层直接路由到预编译好的单核函数 at::native::GeluCUDAKernelImpl
    return torch.nn.functional.gelu(x, approximate="tanh")
```

#### 方式 3：通过 `load_inline` 手写内联 CUDA C++ 核函数（`cuda_gelu`，Thread-Level 范式）
为了理解传统 CUDA 与 Triton 的本质差异，我们对照 CS336 讲义中的内联 CUDA C++ 实现。在 CUDA 中，程序员站在**单个线程（Single Thread）**的视角编写 `__global__` 函数，手动通过 `blockIdx.x * blockDim.x + threadIdx.x` 计算当前线程负责的唯一全局下标 `i`：

```python
import os
from torch.utils.cpp_extension import load_inline

# CUDA C++ 源码：站在【单个线程 (Thread-Level)】视角编写
cuda_gelu_src = """
#include <math.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAException.h>

__global__ void gelu_kernel(float* in, float* out, int num_elements) {
    // 1. 计算当前线程在全局一维数组中的唯一索引 i
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // 2. 边界检查：防止尾块中的多余线程越界访问内存
    if (i < num_elements) {
        // 只从 HBM 读取 1 次 in[i]，中间计算全部在当前线程的私有寄存器中完成，最后只写回 HBM 1 次 out[i]!
        float x = in[i];
        out[i] = 0.5f * x * (1.0f + tanhf(0.79788456f * (x + 0.044715f * x * x * x)));
    }
}

torch::Tensor gelu(torch::Tensor x) {
    TORCH_CHECK(x.device().is_cuda());
    TORCH_CHECK(x.is_contiguous());

    torch::Tensor y = torch::empty_like(x);
    int num_elements = x.numel();
    int block_size = 1024;                                          // 每个 Block 含 1024 个线程
    int num_blocks = (num_elements + block_size - 1) / block_size;  // 向上取整计算 Grid 大小

    // 启动 CUDA Kernel: <<<Grid维度, Block维度>>>
    gelu_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), y.data_ptr<float>(), num_elements);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return y;
}
"""

cpp_gelu_src = "torch::Tensor gelu(torch::Tensor x);"

# 使用 PyTorch 的 load_inline 即时调用 nvcc 编译为 Python 可调用的 C++ 扩展模块
# os.environ["TORCH_CUDA_arch_LIST"] = "9.0" # 根据目标 GPU 架构设置
```

#### 方式 4：使用 `torch.compile` 编译器自动算子融合（`compiled_gelu`）
```python
# 无需修改任何业务代码，PyTorch 2.0+ TorchInductor 后端会自动捕获计算图并生成融合的 Triton Kernel!
compiled_gelu = torch.compile(naive_gelu)
```

### 4.2 实测 Profiler 数据对比：为什么算子融合能带来 11 倍加速？

我们在 $16384 \times 16384$ 的 `float32` 矩阵（共 $N_{\text{elem}} = 268,435,456$ 个元素，单个张量占 **1.074 GB** 显存）上对 `naive_gelu`、`builtin_gelu` 和 `compiled_gelu` 进行实测剖析（[Lecture 6 Trace Lines 280–304](https://cs336.stanford.edu/lectures/?trace=lecture_06)）：

```
+---------------------------------------------------------------------------------------------------+
|                   Naive GeLU (未融合: 9 个独立 Kernel, 17 次 HBM 1GB 张量读写)                      |
|                                                                                                   |
|  HBM ──读x──> [Kernel 1: 0.5*x] ──写t1──> HBM ──读x,x──> [Kernel 2: x*x] ──写t2──> HBM ... (共9次) |
|  总计 CUDA 耗时: 3,384 us (3.384 ms) | CPU 发射 9 次 cudaLaunchKernel 耗时: 1,315 us                |
+---------------------------------------------------------------------------------------------------+
                                                 ||
                                    算子融合 (Kernel Fusion)
                                                 \/
+---------------------------------------------------------------------------------------------------+
|             Built-in / Compiled / Triton GeLU (融合: 1 个 Kernel, 仅 2 次 HBM 读写)                |
|                                                                                                   |
|  HBM ──读x (1.07GB)──> [片上寄存器: 0.5*x*(1 + tanh(0.7978*(x + 0.0447*x^3)))] ──写y (1.07GB)──> HBM|
|  总计 CUDA 耗时: 304.6 us (Built-in) / 342.2 us (torch.compile) -> 实现 ~11.1x 端到端加速!          |
+---------------------------------------------------------------------------------------------------+
```

让我们仔细核对 `torch.profiler` 抓出的精确账单（`dim=16384`）：

| 实现方案 | 发射的 CUDA Kernel 明细（来自 `torch.profiler`） | Kernel 启动总次数 | CPU `LaunchKernel` 耗时 | GPU CUDA 总耗时 | 相对加速比 | HBM 全局显存总流量分析（单张量 $1.074\text{ GB}$） |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`naive_gelu`** | • `BinaryFunctor` (双目乘法 `x*x`, `t*x`, `t*t`): **3 次** (`1,378 us`, 占 40.71%)<br>• `AUnaryFunctor` (标量乘法 `0.5*x`, `0.0447*t`, `0.7978*t`): **3 次** (`926.8 us`, 占 27.39%)<br>• `CUDAFunctor_add` (`x + t`): **1 次** (`460.8 us`, 占 13.62%)<br>• `tanh_kernel_cuda`: **1 次** (`309.6 us`, 占 9.15%)<br>• `CUDAFunctorOnSelf_add` (`1 + t`): **1 次** (`309.3 us`, 占 9.14%) | **9 次** | `1,315.2 us` (`146.1 us/次`) | **`3,384.0 us`** (`3.384 ms`) | `1.00x` (Baseline) | 3 次双目操作（每次 2 读 1 写 = 9 次）+ 6 次单目操作（每次 1 读 1 写 = 12 次），共 **21 次** HBM 张量访问（约 **$22.5\text{ GB}$** 显存流量）！注意单目核耗时稳定在 `~309us`，双目核稳定在 `~460us`（恰好是 3:2 的读写量比例 $3\text{GB vs } 2\text{GB}$）！ |
| **`builtin_gelu`** | • `at::native::vectorized_elementwise_kernel<4, GeluCUDAKernelImpl>` | **1 次** | `291.0 us` | **`304.640 us`** (`0.305 ms`) | **`11.11x`** | 仅 **1 次读取 `x` + 1 次写入 `y`**（共 **2 次** HBM 访问，约 **$2.15\text{ GB}$** 流量，实测有效带宽 $2.15\text{ GB} / 0.3046\text{ ms} \approx 7.05\text{ TB/s}$，接近 B200 的 $8\text{ TB/s}$ 物理极限！） |
| **`compiled_gelu`** | • `triton_poi_fused_add_mul_tanh_0` (TorchInductor 自动生成的 Pointwise Triton 核) | **1 次** | `221.97 us` (`cuLaunchKernel`) | **`342.178 us`** (`0.342 ms`) | **`9.89x`** | 同样仅 **1 次读取 `x` + 1 次写入 `y`**（共 **2 次** HBM 访问），零手写代码即可达到手写 C++ 核 89% 的极限性能！ |

**从 Profiler 表中读出的黄金定律**：
请观察 `naive_gelu` 的剖析结果：无论是极其简单的 `1 + t`（`CUDAFunctorOnSelf_add`），还是包含超越函数的 `tanh_kernel_cuda`，它们的 GPU 耗时竟然**分毫不差地同为 `309 us` 左右**！而融合了全部 9 步运算的 `builtin_gelu` 耗时也是 **`304.6 us`**！这铁证如山地说明：**在逐元素算子中，GPU 的算术逻辑单元（ALU）计算耗时几乎为 0（完全被访存延迟掩盖），算子耗时 100% 取决于读写了多少次 HBM（单目算子 1 读 1 写 = `~309 us`，双目算子 2 读 1 写 = `~460 us`）！**

## 5. Triton 编程范式与四级进阶实战

既然 `torch.compile` 已经能自动融合逐元素算子，为什么我们还要学习手写 **Triton**？
因为对于更复杂的跨维度归约、分块矩阵乘法、自定义稀疏/分块 Attention（如 FlashAttention-2）或融合线性层交叉熵（Fused Linear Cross-Entropy），通用编译器往往无法自动推导出最优的二维分块（Tiling）与在线重缩放（Online Softmax）策略。

### 5.1 CUDA (Thread-Level) vs. Triton (Block-Level) 编程范式对比

由 OpenAI 研发的 [Triton 语言](https://triton-lang.org/) 彻底改变了 GPU 算子的抽象层级：

| 对比维度 | 传统 CUDA C++ (`__global__`) | OpenAI Triton (`@triton.jit`) |
| :--- | :--- | :--- |
| **编程抽象视角** | **Thread-Level（线程级 SIMT）**：程序员编写**单个标量线程**的执行逻辑（`threadIdx.x`）。 | **Block-Level（线程块级 / Tile-Level）**：程序员编写**整个 Thread Block（程序实例 `program_id`）** 对一块向量/子矩阵（Tile）的操作。 |
| **指针与数据形态** | 标量指针解引用 `float x = in[i];` | 向量/张量级指针算术 `x = tl.load(x_ptr + offsets, mask=mask)`（`offsets` 是一个长度为 `BLOCK_SIZE` 的向量！） |
| **Shared Memory 管理** | **纯手动**：显式声明 `__shared__ float smem[...]`，手动插入 `__syncthreads()` 屏障同步。 | **全自动**：编译器根据 `tl.load` 和 `tl.dot` 的数据流自动分配 Shared Memory 并插入同步屏障。 |
| **Bank Conflict 规避** | **纯手动**：程序员必须自己推导 XOR Swizzling 偏移公式。 | **全自动**：Triton 编译器自动分析访存模式并生成无冲突的 Swizzled Shared Memory 布局。 |
| **HBM 访存合并与向量化** | **半手动**：需手动对齐并使用 `float4` 等向量类型。 | **全自动**：编译器自动将 `tl.arange(0, BLOCK_SIZE)` 转换为 `ld.global.v4` 等向量化合并指令，并自动执行 **Thread Coarsening（线程粗化）**。 |

---

### 5.2 Level 1：逐元素算子融合（`triton_gelu_kernel`）与 PTX 汇编深度拆解

我们首先用 Triton 手写一个完整的 GeLU Kernel（`lecture_06.py` Lines 338–390），并检查它编译出的底层 **PTX（Parallel Thread Execution，NVIDIA GPU 虚拟指令集汇编）**。

#### （1）完整代码实现与逐行精读

```python
import torch
import triton
import triton.language as tl

def triton_gelu(x: torch.Tensor) -> torch.Tensor:
    # 1. 前置校验：确保输入张量位于 GPU 且内存布局连续
    assert x.is_cuda
    assert x.is_contiguous()

    # 2. 分配同形状、同设备、同数据类型的输出张量
    y = torch.empty_like(x)

    # 3. 划分一维 Grid（将总元素切分为多个大小为 BLOCK_SIZE 的连续块）
    # |  T  T  T  T  T  T  T  T  |  T  T  T  T  T  T  T  T  | ...
    # |         Block 0          |         Block 1          | ...
    num_elements = x.numel()
    BLOCK_SIZE = 1024
    num_blocks = triton.cdiv(num_elements, BLOCK_SIZE)  # 向上取整除法: (N + 1023) // 1024

    # 4. 启动 Triton Kernel：方括号 [(num_blocks,)] 指定 1D Launch Grid
    # 注意：传入 Tensor (x, y) 时，Triton 会自动将其首元素基地址指针传给 x_ptr, y_ptr
    kernel = triton_gelu_kernel[(num_blocks,)](x, y, num_elements, BLOCK_SIZE=BLOCK_SIZE)
    return y


@triton.jit
def triton_gelu_kernel(
    x_ptr,                        # 输入张量基地址指针 (*fp32)
    y_ptr,                        # 输出张量基地址指针 (*fp32)
    num_elements,                 # 张量元素总数 (int)
    BLOCK_SIZE: tl.constexpr,     # 编译期常量：当前 Block 处理的元素数量 (必须标为 tl.constexpr)
):
    # 1. 获取当前程序实例（即当前 Thread Block）在第 0 维网格上的唯一编号 pid
    pid = tl.program_id(axis=0)
    start = pid * BLOCK_SIZE      # 当前 Block 负责的起始全局元素下标

    # 2. 生成当前 Block 负责的全局偏移量向量 (长度为 BLOCK_SIZE 的 1D 向量: [start, start+1, ..., start+BLOCK_SIZE-1])
    offsets = start + tl.arange(0, BLOCK_SIZE)

    # 3. 构造布尔掩码向量：防止最后一个 Block 超出 num_elements 越界读写非法的 HBM 地址
    mask = offsets < num_elements

    # 4. 从 HBM 向量化合并读取整块数据到片上寄存器 (越界位置不读取)
    x = tl.load(x_ptr + offsets, mask=mask)

    # 5. 在片上寄存器中执行融合算术计算
    # 关键细节：标准 Triton 语言原语中没有直接提供 tl.tanh，我们利用双曲正切数学恒等式：
    # tanh(a) = (e^{2a} - 1) / (e^{2a} + 1)
    a = 0.79788456 * (x + 0.044715 * x * x * x)
    exp = tl.exp(2 * a)
    tanh = (exp - 1) / (exp + 1)
    y = 0.5 * x * (1 + tanh)

    # 6. 将片上计算结果一次性合并写回 HBM
    tl.store(y_ptr + offsets, y, mask=mask)
```

#### （2）导出并透视 Triton 生成的 PTX 汇编代码（`output_ptx`）

我们可以通过访问已编译 Kernel 对象的 `kernel.asm["ptx"]` 属性，直接查看 Triton 编译器生成的底层 GPU 汇编代码（[官方生成的 `triton_gelu-ptx.txt`](https://github.com/stanford-cs336/lectures/blob/main/var/triton_gelu-ptx.txt)）：

```python
def output_ptx(name: str, kernel):
    """导出 Triton 为指定 kernel 生成的 PTX 汇编代码。"""
    ptx_path = f"var/{name}-ptx.txt"
    with open(ptx_path, "w") as f:
        ptx = kernel.asm["ptx"]
        f.write(ptx)
```

Percy Liang 带领大家阅读 `triton_gelu-ptx.txt` 时指出了四个极其关键的底层发现：
1. **线程粗化（Thread Coarsening）自动发生**：虽然我们在 Python 端写的是 `BLOCK_SIZE = 1024`，但查看 PTX 头部的 `.reqntid 128`（即 `num_warps = 4`，每个 Block 实际只启动了 **128 个物理线程**）！这意味着 **Triton 编译器自动让每个物理线程同时并行处理 $1024 / 128 = 8$ 个元素**！
2. **硬件特殊寄存器映射**：PTX 中的 `%ctaid.x`（Cooperative Thread Array ID）对应 CUDA 的 `blockIdx.x`（即 Triton 的 `tl.program_id(0)`）；`%tid.x` 对应 `threadIdx.x`。
3. **128-bit 向量化合并访存**：PTX 没有生成单元素的 `ld.global.b32`，而是生成了两条连续的 **`ld.global.v4.b32`**（单条指令一次性加载 4 个 32-bit 浮点数 = 16 Bytes）和 **`st.global.v4.b32`**，从而让单个线程一次把 8 个元素全部装入浮点寄存器 `%f1..%f8`。
4. **纯寄存器级流水线**：在 `ld.global` 与 `st.global` 之间，全部是针对浮点寄存器 `%f*` 和整数/谓词寄存器 `%r*`、`%p*` 的指令（如 `mul.f32`、`fma.rn.f32` 融合乘加、`ex2.approx.ftz.f32` 以 2 为底的快速指数指令、`div.full.f32`），**全程没有任何中间数据写回 HBM 或 Shared Memory**！

---

### 5.3 Level 2：单块行归约融合（`triton_softmax_kernel`，当行宽 $N \le \text{BLOCK\\_SIZE}$）

逐元素算子（如 GeLU）中每个元素的计算完全独立；而深度学习中另一大类核心算子是**归约/聚合算子（Reductions）**，典型代表就是 Attention 与输出层必用的 **数值稳定版 Softmax**（按矩阵 $X \in \mathbb{R}^{M \times N}$ 的每一行独立归一化）：
$$\text{Softmax}(x)_{i, j} = \frac{\exp(x_{i, j} - \max_k x_{i, k})}{\sum_{k=1}^N \exp(x_{i, k} - \max_k x_{i, k})}$$

#### （1）朴素 PyTorch Softmax 的 HBM 访存账单 vs. 融合极限

让我们像 `lecture_06.py`（Lines 420–441）那样，精确清点 `naive_softmax` 在 $M \times N$ 矩阵上触发的 HBM 读写次数：

```python
def naive_softmax(x: torch.Tensor) -> torch.Tensor:
    # 输入 x 形状为 (M, N)
    M, N = x.shape

    # Step 1: 求每行最大值 -> 读取整个 x (MN 次读)，写出每行最大值向量 (M 次写)
    x_max = x.max(dim=1)[0]

    # Step 2: 减去行最大值 -> 再次读取整个 x 及 x_max (MN + M 次读)，写出平移后的 x (MN 次写)
    x = x - x_max[:, None]

    # Step 3: 逐元素求指数 -> 读取 x (MN 次读)，写出分子矩阵 numerator (MN 次写)
    numerator = torch.exp(x)

    # Step 4: 按行求和得归一化分母 -> 读取 numerator (MN 次读)，写出分母向量 (M 次写)
    denominator = numerator.sum(dim=1)

    # Step 5: 逐元素相除完成归一化 -> 读取 numerator 及 denominator (MN + M 次读)，写出最终结果 y (MN 次写)
    y = numerator / denominator[:, None]

    # 【总计 HBM 访存账单】：
    #   总读取量 = 5MN + 2M ≈ 5MN
    #   总写入量 = 3MN + 2M ≈ 3MN
    #   合计 HBM 流量 ≈ 8MN 个元素！
    # 【理想算子融合极限】：
    #   如果整行能装进片上 SRAM/寄存器，只需从 HBM 读取 1 次 (MN 读)，在片上完成 max/sub/exp/sum/div，最后写回 1 次 (MN 写)！
    #   合计 HBM 流量 = 2MN 个元素 -> 理论带宽加速比高达 8MN / 2MN = 4.0 倍！
    return y
```

#### （2）Triton Fused Softmax 核心设计：1 个 Block 负责 1 整行

当矩阵列数 $N$ 不算太大（例如 $N \le 4096$ 或 $8192$，能装入单个 SM 的寄存器/SRAM）时，最优雅的划分策略是：
- **Grid 维度设为 `(M,)`**：启动 $M$ 个 Thread Blocks，第 `row_idx = tl.program_id(0)` 个 Block 独立包揽第 `row_idx` 行的全部计算！这样**不同行之间完全独立，零跨 Block 通信开销**。
- **`BLOCK_SIZE` 设为 `triton.next_power_of_2(N)`**：Triton 的块内归约原语（`tl.max`, `tl.sum`）要求向量长度为 2 的幂次。
- **二维张量的线性内存跨步（Stride）寻址**：在行优先存储的二维张量中，第 `row_idx` 行的起始元素相对于基地址 `x_ptr` 的偏移量为 `row_idx * x.stride(0)`。
- **极值归约的安全 Padding（`other=float("-inf")`）**：当列数 $N$ 不是 2 的幂次时（例如 $N=1000$，而 `BLOCK_SIZE=1024`），末尾 24 个越界槽位被 `mask = col_offsets < num_cols` 屏蔽。在调用 `tl.max(x_row, axis=0)` 时，这 24 个填充位必须设为 **`-inf`（负无穷）**，这样才不会污染最大值，且后续计算 `tl.exp(-inf - max) = 0.0` 时对求和分母的贡献也恰好为 `0`！

```python
def triton_softmax(x: torch.Tensor) -> torch.Tensor:
    y = torch.empty_like(x)
    M, N = x.shape

    # 将 BLOCK_SIZE 设为大于等于列数 N 的最小 2 的幂次（保证整行装入同一个 Block）
    block_size = triton.next_power_of_2(N)
    num_blocks = M  # 每个 Thread Block 独立负责矩阵的一整行

    triton_softmax_kernel[(num_blocks,)](
        x_ptr=x, y_ptr=y,
        x_row_stride=x.stride(0), y_row_stride=y.stride(0),
        num_cols=N, BLOCK_SIZE=block_size
    )
    return y


@triton.jit
def triton_softmax_kernel(
    x_ptr, y_ptr,
    x_row_stride, y_row_stride,
    num_cols,
    BLOCK_SIZE: tl.constexpr
):
    assert num_cols <= BLOCK_SIZE

    # 1. 当前 Thread Block 负责第 row_idx 行
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)

    # 2. 计算第 row_idx 行各列元素的 HBM 指针，并合并读取整行到片上寄存器
    # 注意：越界位置必须填充 -inf，确保不影响 tl.max 且 exp(-inf) == 0 不影响 tl.sum!
    x_start_ptr = x_ptr + row_idx * x_row_stride
    x_ptrs = x_start_ptr + col_offsets
    x_row = tl.load(x_ptrs, mask=col_offsets < num_cols, other=float("-inf"))

    # 3. 全部在片上 SRAM/寄存器中完成三次遍历（Max -> Exp -> Sum -> Div），零 HBM 往返!
    # Triton 编译器会自动将 tl.max 和 tl.sum 编译为高效的 Warp Shuffle (shfl.sync.bfly) 与 Shared Memory 树状归约
    x_row = x_row - tl.max(x_row, axis=0)
    numerator = tl.exp(x_row)
    denominator = tl.sum(numerator, axis=0)
    y_row = numerator / denominator

    # 4. 将归一化后的整行写回 HBM（总共仅 1 次 HBM 读 + 1 次 HBM 写）
    y_start_ptr = y_ptr + row_idx * y_row_stride
    y_ptrs = y_start_ptr + col_offsets
    tl.store(y_ptrs, y_row, mask=col_offsets < num_cols)
```

---

### 5.4 Level 3：跨分块循环归约 / "Baby Tiling"（`row_sum_kernel`，当行宽 $N > \text{BLOCK\\_SIZE}$）

在上面的 `triton_softmax_kernel` 中，我们假设整行都能塞进一个 `BLOCK_SIZE`。**如果矩阵的列数 $N$ 非常大（例如 $N = 65536$ 或 $131072$），超过了单个 SM 的寄存器/SRAM 容量上限，或者我们固定 `BLOCK_SIZE = 1024` 而 $N = 4096$，该怎么办？**

这就引出了 GPU 编程中最核心的思想——**Tiling（分块循环，讲义中称为 "Baby Tiling"）**：
1. 依然让 **1 个 Thread Block 负责 1 整行**（`grid = (M,)`），但在片上只分配一个长度为 `BLOCK_SIZE` 的局部向量累加器 `acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)`。
2. 将长度为 $N$ 的整行切分为 $\lceil N / \text{BLOCK\\_SIZE} \rceil$ 个连续的水平分块（Tiles）。
3. 编写 `for start in range(0, N, BLOCK_SIZE):` 循环，每次加载一个 `BLOCK_SIZE` 分块（越界处填充 `other=0.0`），并累加到片上向量累加器 `acc += x` 中（此时每个线程独立累加属于自己偏移位置的元素，无需跨线程通信）。
4. 当整行的所有 Tiles 循环累加完毕后，**只在循环体外执行 1 次跨线程归约** `result = tl.sum(acc, axis=0)`，将长度为 `BLOCK_SIZE` 的向量压成一个标量，写回 `out_ptr + row`！

```
+---------------------------------------------------------------------------------------------------+
|                  Level 3 "Baby Tiling": 跨分块行求和 (以 N = 12, BLOCK_SIZE = 4 为例)                |
+---------------------------------------------------------------------------------------------------+
|                                                                                                   |
|  第 row 行全局数据 (N=12):  | x0  x1  x2  x3 | x4  x5  x6  x7 | x8  x9  x10 x11 |                  |
|                              Tile 0 (start=0)  Tile 1 (start=4)  Tile 2 (start=8)                 |
|                                     |                 |                 |                         |
|                                     +-----------------+-----------------+                         |
|                                                       | 循环内对应位置逐元素累加 (acc += x)          |
|                                                       v                                           |
|  片上向量累加器 acc [4]:    [ T0: x0+x4+x8,   T1: x1+x5+x9,   T2: x2+x6+x10,  T3: x3+x7+x11 ]      |
|                                                       |                                           |
|                                                       | 循环外执行 1 次跨线程归约 tl.sum(acc, axis=0)|
|                                                       v                                           |
|  最终标量输出 out[row]:                      result = sum(x0 ... x11)                              |
+---------------------------------------------------------------------------------------------------+
```

```python
def triton_row_sum(x: torch.Tensor, BLOCK_SIZE: int = 1024) -> torch.Tensor:
    M, N = x.shape
    y = torch.empty(M, device=x.device, dtype=x.dtype)
    row_sum_kernel[(M,)](x, y, N, BLOCK_SIZE=BLOCK_SIZE)
    return y


@triton.jit
def row_sum_kernel(x_ptr, out_ptr, N, BLOCK_SIZE: tl.constexpr):
    # 1. 当前 Thread Block 负责第 row 行
    row = tl.program_id(0)

    # 2. 在片上寄存器中初始化长度为 BLOCK_SIZE 的 FP32 累加器向量
    # 示意图 (N = 12, BLOCK_SIZE = 4): 一行被切为 3 个 Tile: T1 T2 T3 T4 | T1 T2 T3 T4 | T1 T2 T3 T4
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    # 3. 沿列维度循环遍历所有 Tiles
    for start in range(0, N, BLOCK_SIZE):
        cols = start + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        # 越界位置填充 0.0，确保不影响加法累加
        x = tl.load(x_ptr + row * N + cols, mask=mask, other=0.0)
        acc += x

    # 4. 循环结束后，将 BLOCK_SIZE 个部分和归约为单个标量并写回 HBM
    result = tl.sum(acc, axis=0)
    tl.store(out_ptr + row, result)
```

> **进阶思考（通往 FlashAttention-2 的桥梁）**：如果当 $N > \text{BLOCK\\_SIZE}$ 时我们要算的不是简单的 `row_sum`，而是 `softmax` 呢？按传统做法需要先跑一个循环求全局 `max`，再跑第二个循环求 `sum(exp(x - max))`；而 **Online Softmax（在线重缩放算法）** 允许我们在单个 Tile 循环中边更新运行期最大值 $m_{\text{new}} = \max(m_{\text{old}}, \tilde{m})$，边通过乘上修正因子 $e^{m_{\text{old}} - m_{\text{new}}}$ 动态修正之前的累加和！这正是你在 Assignment 2 中编写 **Triton FlashAttention-2** 的核心数学基石。

---

### 5.5 Level 4：二维分块矩阵乘法与算子融合（`matmul_relu_kernel`）

现在我们挑战深度学习最核心的算子：矩阵乘法 + 激活函数融合 $C = \text{ReLU}(A B)$，其中 $A \in \mathbb{R}^{M \times K}, B \in \mathbb{R}^{K \times N}, C \in \mathbb{R}^{M \times N}$（`triton_matmul_relu_example`，Lines 539–676）。

#### （1）为什么矩阵乘法必须使用二维分块（2D Tiling）？算术强度三阶段推演

让我们对比计算 $C = A B$（为直观起见，设 $M=K=N$）的三种访存策略及其算术强度（Arithmetic Intensity = FLOPs / Bytes）：

1. **朴素无缓存算法（Naive Approach）**：
   固定任意一个输出元素 $C[m, n]$，从 HBM 循环读取 $A[m, k]$ 和 $B[k, n]$（$k=1\dots K$）相乘累加后写回 $C[m, n]$。
   - **HBM 访存量**：每个输出元素需要 $2K$ 次 HBM 读取；总共 $M \times N$ 个输出元素需要 **$2MKN$ 次 HBM 读取** 和 **$MN$ 次写入**！
   - **算术强度**：$\frac{2MKN \text{ FLOPs}}{2MKN \text{ Reads}} = \mathcal{O}(1)$！这意味着每从 HBM 读 1 个数只做 1 次乘加，昂贵的 Tensor Core 99% 的时间都在苦等 HBM 内存总线！
2. **理想全量上片算法（Idealized Approach）**：
   将整个矩阵 $A$ 和 $B$ 一次性全部读入 Shared Memory，在片上算完整个 $C$ 后写回 HBM。
   - **HBM 访存量**：只需 **$MK + KN$ 次读取** 和 **$MN$ 次写入**，算术强度跃升至理想的 $\mathcal{O}(N)$！
   - **致命缺陷**：单个 SM 的 Shared Memory 只有 `192~256 KB`，根本装不下动辄几十 MB 到几 GB 的完整矩阵 $A$ 和 $B$！
3. **二维分块算法（2D Tiling，工业界标准解法）**：
   既然装不下整个矩阵，我们就把输出矩阵 $C$ 划分为大小为 $\text{BLOCK\\_M} \times \text{BLOCK\\_N}$ 的二维子块（Output Tiles），**每个 Thread Block 负责计算 $C$ 中的一个二维子块**！
   - 为了算出子块 $C_{\text{tile}} \in \mathbb{R}^{\text{BLOCK\\_M} \times \text{BLOCK\\_N}}$，当前 Thread Block 沿着内积维度 $K$ 以步长 $\text{BLOCK\\_K}$ 滑动：每次只从 HBM 将一个 $A_{\text{tile}} \in \mathbb{R}^{\text{BLOCK\\_M} \times \text{BLOCK\\_K}}$ 和一个 $B_{\text{tile}} \in \mathbb{R}^{\text{BLOCK\\_K} \times \text{BLOCK\\_N}}$ 加载到 **Shared Memory** 中，调用 Tensor Core 执行小矩阵乘法 `acc += tl.dot(a, b)` 累加到片上 FP32 寄存器中。
   - **分块后的 HBM 访存量与算术强度**：
     每个输出块执行 $K / \text{BLOCK\\_K}$ 步循环，共从 HBM 读取 $\frac{K}{\text{BLOCK\\_K}} (\text{BLOCK\\_M}\cdot\text{BLOCK\\_K} + \text{BLOCK\\_K}\cdot\text{BLOCK\\_N}) = K(\text{BLOCK\\_M} + \text{BLOCK\\_N})$ 个元素，却完成了 $2 \cdot \text{BLOCK\\_M} \cdot \text{BLOCK\\_N} \cdot K$ 次浮点运算！
     因此，算术强度提升为：
     $$\text{Arithmetic Intensity}_{\text{Tiled}} \propto \frac{\text{BLOCK\\_M} \cdot \text{BLOCK\\_N}}{\text{BLOCK\\_M} + \text{BLOCK\\_N}} = \mathcal{O}(\text{tile\\_size})$$
     只要将 $\text{BLOCK\\_M}, \text{BLOCK\\_N}$ 选在 `64` 或 `128`（受限于 Shared Memory 容量），就能在不爆片上内存的前提下，将 HBM 访存量削减数十倍，彻底跨过 Roofline 拐点进入 Compute-Bound 区域！

```
+===================================================================================================+
|                     2D Tiled Matrix Multiplication + Fused ReLU (C = ReLU(A @ B))                 |
+===================================================================================================+
|                                                                                                   |
|                  K (步长 BLOCK_K=32)                         N (按 BLOCK_N=64 划分 pid_n)          |
|            +--------+--------+--------+                +--------+--------+--------+               |
|            |        |        |        |                |        |  B1    |        |  |            |
|            +--------+--------+--------+                |        | (32x64)|        |  |            |
|  BLOCK_M   |   A4   |   A5   |   A6   |   *          K +--------+--------+--------+  v 沿 K 维滑动 |
|  (pid_m)   | (64x32)| (64x32)| (64x32)|                |        |  B4    |        |               |
|            +--------+--------+--------+                |        | (32x64)|        |               |
|            |        |        |        |                +--------+--------+--------+               |
|            +--------+--------+--------+                |        |  B7    |        |               |
|                矩阵 A (M x K)                          +--------+--------+--------+               |
|                  -----> 沿 K 维滑动                               矩阵 B (K x N)                   |
|                                                                                                   |
|  [单个 Thread Block (pid_m, pid_n) 的片上执行流水线]:                                               |
|   1. 片上寄存器初始化: acc = tl.zeros([64, 64], dtype=tl.float32)                                  |
|   2. Loop k = 0, 32, 64 ... :                                                                     |
|        • 从 HBM 加载 A_tile [64, 32] 与 B_tile [32, 64] 到 Shared Memory                          |
|        • Tensor Core 矩阵乘累加: acc += tl.dot(a, b)                                               |
|        • 指针步进: a_ptrs += 32 * stride_ak; b_ptrs += 32 * stride_bk                             |
|   3. 片上算子融合 (零额外 HBM 读写!): acc = tl.maximum(acc, 0.0)                                   |
|   4. 写回 HBM 输出块 C5 [64, 64]: tl.store(c_ptrs, acc, mask=...)                                 |
+===================================================================================================+
```

#### （2）二维广播指针算术（2D Pointer Broadcasting）与完整代码精解

在 Triton 中处理二维 Tile 的点睛之笔是利用 `[:, None]`（列向量）与 `[None, :]`（行向量）的广播加法，直接构造出形状为 `[BLOCK_M, BLOCK_K]` 的**二维指针矩阵 `a_ptrs`**：
$$\text{a\\_ptrs}[i, j] = \text{a\\_ptr} + \text{indices\\_m}[i] \times \text{stride\\_am} + \text{indices\\_k}[j] \times \text{stride\\_ak}$$

下面是 `lecture_06.py`（Lines 608–676）的完整实现与逐行注释：

```python
def triton_matmul_relu(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    assert a.is_cuda and b.is_cuda
    assert a.is_contiguous() and b.is_contiguous()
    assert a.shape[1] == b.shape[0]

    # A 形状为 (M, K), B 形状为 (K, N), 输出 C 形状为 (M, N)
    M, K = a.shape
    K, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)

    # 设置二维分块超参数，并构造 2D Launch Grid: 每个 Block 负责输出矩阵 C 的一个 (BLOCK_M x BLOCK_N) 子块
    BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    matmul_relu_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )
    return c


@triton.jit
def matmul_relu_kernel(
    a_ptr, b_ptr, c_ptr,          # 矩阵 A, B, C 的首地址指针
    M, N, K,                      # 矩阵维度: A (M x K), B (K x N), C (M x N)
    stride_am, stride_ak,         # A 的行跨步 (通常为 K) 与列跨步 (通常为 1)
    stride_bk, stride_bn,         # B 的行跨步 (通常为 N) 与列跨步 (通常为 1)
    stride_cm, stride_cn,         # C 的行跨步 (通常为 N) 与列跨步 (通常为 1)
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # 1. 获取当前 Thread Block 负责的输出子块坐标 (pid_m, pid_n)
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # 2. 生成当前分块在 M, N, K 三个维度上的一维索引向量
    indices_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # A 和 C 的行索引向量, 形状: [BLOCK_M]
    indices_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # B 和 C 的列索引向量, 形状: [BLOCK_N]
    indices_k = tl.arange(0, BLOCK_K)                    # 首个 K 分块的偏移向量, 形状: [BLOCK_K]

    # 3. 利用二维广播 ([:, None] 与 [None, :]) 构造初始 Tile 的二维指针矩阵!
    # a_ptrs 形状为 [BLOCK_M, 1] + [1, BLOCK_K] -> [BLOCK_M, BLOCK_K]
    # b_ptrs 形状为 [BLOCK_K, 1] + [1, BLOCK_N] -> [BLOCK_K, BLOCK_N]
    a_ptrs = a_ptr + indices_m[:, None] * stride_am + indices_k[None, :] * stride_ak
    b_ptrs = b_ptr + indices_k[:, None] * stride_bk + indices_n[None, :] * stride_bn

    # 4. 在片上寄存器中初始化 FP32 累加器子块 [BLOCK_M, BLOCK_N]
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    # 5. 沿内积维度 K 循环滑动窗口，每次步进 BLOCK_K
    for k in range(0, K, BLOCK_K):
        # 二维边界掩码：越界元素必须填充 0.0，确保矩阵乘内积贡献为 0
        a = tl.load(a_ptrs, mask=(indices_m[:, None] < M) & (indices_k[None, :] + k < K), other=0.0)
        b = tl.load(b_ptrs, mask=(indices_k[:, None] + k < K) & (indices_n[None, :] < N), other=0.0)

        # 调用 Tensor Core 执行分块矩阵乘并累加到片上寄存器 acc 中
        acc += tl.dot(a, b)

        # 将二维指针矩阵沿 K 维向前推进一个分块距离
        a_ptrs += BLOCK_K * stride_ak  # A 沿列方向向右移动 BLOCK_K 列
        b_ptrs += BLOCK_K * stride_bk  # B 沿行方向向下移动 BLOCK_K 行

    # 6. 【算子融合红利】：当 acc 还在片上高速寄存器中时，直接原地应用 ReLU (或 GeLU) 激活函数！
    # 完全省去了将中间矩阵乘结果写回 HBM 再读出来的 2MN 次全局显存流量！
    acc = tl.maximum(acc, 0.0)

    # 7. 构造输出子块 C 的二维指针矩阵，并将最终结果一次性写回 HBM
    c_ptrs = c_ptr + indices_m[:, None] * stride_cm + indices_n[None, :] * stride_cn
    tl.store(c_ptrs, acc, mask=(indices_m[:, None] < M) & (indices_n[None, :] < N))
```

## 6. 核心知识点全景对比与工程 Checklist

为了方便在完成 **Assignment 2（Systems & Parallelism）** 时快速查阅，我们将本讲涵盖的四类 Triton 编程模式汇总如下：

| 算子级别与代表核函数 | 适用计算形态 | Launch Grid 维度 | 指针算术与寻址特征 | 越界 `mask` 与 `other` 填充法则 | 片上状态与归约方式 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Level 1: 逐元素融合**<br>(`triton_gelu_kernel`) | Elementwise ($y_i = f(x_i)$) | 1D: `(cdiv(N, B),)` | 1D 连续偏移：`x_ptr + pid * B + tl.arange(0, B)` | `mask = offsets < N`，无需特殊 `other`（因为写回时用同一 `mask` 屏蔽） | 无跨元素归约，每个线程粗化处理 8 个元素，纯寄存器流水线 |
| **Level 2: 单块整行归约**<br>(`triton_softmax_kernel`) | Row-wise Reduction ($N \le B$) | 1D: `(M,)` (每行 1 个 Block) | 行基址 + 列偏移：`x_ptr + row * stride + tl.arange(0, B)` | `mask = cols < N`；求 `tl.max` 时 **`other = -inf`**（求 `tl.sum` 时 `other = 0.0`） | 整行常驻寄存器/SRAM，`tl.max` / `tl.sum` 触发块内蝶形归约 |
| **Level 3: 跨块循环归约**<br>(`row_sum_kernel`) | Tiled Row Reduction ($N > B$) | 1D: `(M,)` (每行 1 个 Block) | 循环内步进：`x_ptr + row * N + start + tl.arange(0, B)` | `mask = (start + tl.arange(0, B)) < N`；求和累加 **`other = 0.0`** | 片上维护向量累加器 `acc [B]`，循环内逐元素 `acc += x`，循环外 `tl.sum(acc, 0)` |
| **Level 4: 2D 分块矩阵乘**<br>(`matmul_relu_kernel`) | 2D Tiled GEMM + Epilogue Fusion | 2D: `(cdiv(M, BM), cdiv(N, BN))` | 2D 广播指针：`ptr + idx_m[:, None]*sm + idx_k[None, :]*sk` | 2D 联合掩码：`(idx_m[:, None] < M) & (idx_k[None, :] + k < K)`，**`other = 0.0`** | 片上维护 2D 累加器 `acc [BM, BN]`，沿 $K$ 循环 `acc += tl.dot(a, b)`，循环外融合激活 |

## 7. 课后自测题与深度解析（直通 Assignment 2 实战）

### 练习 1：寄存器与 Shared Memory 双重约束下的 Warp Occupancy 计算
**问题**：在 NVIDIA H100（或 B200）GPU 上，每个 SM 拥有 `65,536` 个 32-bit 寄存器、最高 `256 KB` 的 Shared Memory，且最多支持 `64` 个并发 Warps（即 `2048` 个线程）。假设你编写了一个 2D Tiled MatMul Kernel，配置为：
- 每个 Thread Block 包含 `256` 个线程（即 `8` 个 Warps）；
- 编译器报告每个线程使用了 `96` 个寄存器；
- 每个 Thread Block 静态申请了 `64 KB` 的 Shared Memory（假设全部 `256 KB` 均配置给 Shared Memory）。
请问：（1）受寄存器限制，单个 SM 最多能驻留多少个 Block？（2）受 Shared Memory 限制，单个 SM 最多能驻留多少个 Block？（3）该 Kernel 的最终实际 **Warp Occupancy** 是多少？瓶颈在哪里？

**深度解析**：
1. **寄存器上限分析**：
   单个 Thread Block 消耗的寄存器总数 $= 256 \times 96 = 24,576$ 个寄存器。
   单个 SM 的寄存器堆可容纳的最大 Block 数 $= \lfloor 65,536 / 24,576 \rfloor = \mathbf{2\text{ 个 Blocks}}$（消耗 $49,152$ 个寄存器，剩余 $16,384$ 个寄存器不足以放下第 3 个 Block）。
2. **Shared Memory 上限分析**：
   单个 SM 的 Shared Memory 可容纳的最大 Block 数 $= \lfloor 256\text{ KB} / 64\text{ KB} \rfloor = \mathbf{4\text{ 个 Blocks}}$。
3. **最终实际 Warp Occupancy 与瓶颈定位**：
   由于一个 Block 必须同时满足寄存器和 Shared Memory 分配才能驻留，取两者的最小值：$\text{Active Blocks per SM} = \min(2, 4) = \mathbf{2\text{ 个 Blocks}}$。
   实际驻留的活跃 Warps 数量 $= 2 \times (256 / 32) = 16\text{ 个 Warps}$。
   最终 $\text{Warp Occupancy} = 16 / 64 = \mathbf{25\%}$（`0.25`）。当前的瓶颈在于**寄存器压力（Register Pressure）**——如果能将单线程寄存器用量从 `96` 压缩到 `80`（使得单 Block 消耗 $256 \times 80 = 20,480$ 个寄存器），则单个 SM 就能驻留 $\lfloor 65,536 / 20,480 \rfloor = 3$ 个 Blocks，Occupancy 将立即从 `25%` 提升至 $24 / 64 = \mathbf{37.5\%}$！

### 练习 2：为什么 `triton_softmax_kernel` 与 `matmul_relu_kernel` 的 `other` 填充值不同？
**问题**：对比 `triton_softmax_kernel` 中的 `tl.load(..., mask=..., other=float("-inf"))` 与 `matmul_relu_kernel` 中的 `tl.load(..., mask=..., other=0.0)`。如果在 Softmax 中误将 `other` 写成了 `0.0`，会在什么时候引发严重的数值 Bug？

**深度解析**：
- 在 `matmul_relu_kernel` 和 `row_sum_kernel` 中，越界位置参与的是**加法归约（内积求和）**，加法的单位元（Identity Element）是 `0.0`，填充 `0.0` 可以保证越界位置对内积结果零贡献。
- 而在 `triton_softmax_kernel` 中，加载后的整行向量 `x_row` 紧接着要执行两步非线性/归约操作：
  1. `tl.max(x_row, axis=0)`：**最大值归约的单位元是 $-\infty$（`float("-inf")`）**。如果误填为 `0.0`，当某一行的所有真实有效元素全部为负数（例如 Attention Mask 后的 Logits 全是 `-5.0, -10.0`）时，`tl.max` 会错误地把越界填充的 `0.0` 当作最大值！
  2. `numerator = tl.exp(x_row - max_val)`：紧接着对整行求指数并调用 `tl.sum(numerator, axis=0)`。当越界位置填为 `-inf` 时，$\exp(-\infty - \text{max}) = \exp(-\infty) = \mathbf{0.0}$，在随后的分母求和中自动变为加法单位元 `0.0`！若误填为 `0.0`，越界位置算出的 $\exp(0.0 - \text{max}) > 0$ 会被错误地计入归一化分母，导致整行有效元素的概率之和严格小于 1！

### 练习 3：从 `triton_softmax_kernel` 到 Assignment 2 的 `RMSNorm` 融合核
**问题**：在 Transformer 中，$\text{RMSNorm}(x)_i = \frac{x_i}{\sqrt{\frac{1}{d}\sum_{j=1}^d x_j^2 + \epsilon}} \cdot \gamma_i$（其中 $x \in \mathbb{R}^{M \times d}$，$\gamma \in \mathbb{R}^d$）。请计算朴素 PyTorch 实现 `x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * weight` 的 HBM 读写次数，并基于本讲 Level 2 的模板写出单块融合版 `triton_rmsnorm_kernel` 的核心片上计算逻辑。

**深度解析**：
- **HBM 流量对比**：朴素 PyTorch 实现包含 `x.pow(2)`（1 读 1 写）、`.mean(-1)`（1 读）、`+ eps` 与 `rsqrt`、乘以 `x`（2 读 1 写）、乘以 `weight`（1 读 1 写），总计需要约 **$5Md$ 次读取 + $3Md$ 次写入 = $8Md$ 次 HBM 访问**；而单块融合版 Triton Kernel 只需 **$Md + d$ 次读取 + $Md$ 次写入 $\approx 2Md$ 次 HBM 访问**（同样实现 **4 倍带宽加速**）：

```python
@triton.jit
def triton_rmsnorm_kernel(
    x_ptr, weight_ptr, y_ptr,
    stride_x_row, stride_y_row,
    num_cols, eps,
    BLOCK_SIZE: tl.constexpr
):
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < num_cols

    # 1. 从 HBM 合并读取当前行 x 与缩放权重 gamma (越界位置填 0.0，平方后仍为 0.0，不影响平方和!)
    x_row = tl.load(x_ptr + row_idx * stride_x_row + col_offsets, mask=mask, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)

    # 2. 片上寄存器计算均方根倒数 (rstd) 并完成归一化与缩放
    mean_square = tl.sum(x_row * x_row, axis=0) / num_cols
    rstd = tl.rsqrt(mean_square + eps)
    y_row = x_row * rstd * weight

    # 3. 一次性写回 HBM
    tl.store(y_ptr + row_idx * stride_y_row + col_offsets, y_row, mask=mask)
```
