---
title: Stanford CS336 (Spring 2026) Lecture 7 深度课程笔记：Parallelism I (Collectives, NCCL & DDP)
date: 2026-10-25 16:16:00
tags:
  - 大模型
  - 分布式训练
  - PyTorch
  - NCCL
  - DDP
  - 张量并行
  - 流水线并行
  - CS336
categories:
  - [笔记, CS336]
description: 深入解析 Stanford CS336 Lecture 7，涵盖万卡集群存储与带宽层级、8大集合通信原语数学推导与 PyTorch 实现、NCCL 总线带宽测算算法、DDP 数据并行实战与反向传播通信重叠，以及张量并行（TP）与流水线并行（PP）的核心原语。
mathjax: true
---

# Stanford CS336 (Spring 2026) Lecture 7 深度课程笔记：Parallelism I (Collectives, NCCL & DDP)

**课程信息**：[Stanford CS336: Language Modeling from Scratch (Spring 2026)](https://cs336.stanford.edu/)  
**主讲人**：Percy Liang  
**课程视频与代码讲义**：[Lecture 7 YouTube 视频](https://www.youtube.com/watch?v=SzpOcwdIL0Y) | [Executable Lecture 源码 (`lecture_07.py` / Trace)](https://cs336.stanford.edu/spring2025-lectures/?trace=var/traces/lecture_08.json)  
**关联作业**：Assignment 2 — Systems & Parallelism（从零手写 Distributed Data Parallel (DDP)、通信与计算重叠、优化器状态分片 ZeRO-1）

> **课程编排说明（Spring 2026 vs. Spring 2025 课表顺序对调）**：在 Spring 2026 学期中，课程组将两节并行课的顺序做了优化调整——**Lecture 7（本讲，Percy Liang 主讲）** 率先通过纯可执行代码 `lecture_07.py` 从底层硬件拓扑、8 大集合通信原语、NCCL 带宽测算一直写到 DDP、张量并行（TP）与流水线并行（PP）的极简 PyTorch 实现；而 **Lecture 8（下一讲，Tatsu Hashimoto 主讲）** 则在此代码基础上深入展开 ZeRO 1/2/3、FSDP、1F1B 流水线气泡消除、Megatron 序列并行以及万卡集群 3D/4D 混合并行实战案例。本笔记完整覆盖 `lecture_07.py` 的全部代码、运行输出与课堂推导，并提前为你打通迈向 ZeRO-1 的核心桥梁。

## 1. 课程全景导读：从单卡算子优化迈向多机多卡集群

在 [Lecture 5（GPUs & TPUs）](https://www.youtube.com/watch?v=izZba4UA7iY) 和 [Lecture 6（Kernels & Triton）](https://cs336.stanford.edu/lectures/?trace=lecture_06) 中，我们聚焦于**单张 GPU 内部**的性能压榨：计算单元（SM 内的 Tensor Cores / ALUs）距离片外显存（HBM）很远，因此我们通过 **Kernel Fusion（算子融合）** 和 **Tiling（分块复用 SRAM）** 来减少 HBM 读写往返。

从本讲开始，我们将视野放大到 **4 张、8 张乃至 10,000+ 张 GPU 组成的超算集群**。Percy 开篇指出，单卡优化与多卡并行在物理本质上遵循**完全相同的第一性原理**：
- **核心矛盾不变**：计算单元（Compute）远离它所需要的数据（Data）。在单卡内，“远”意味着数据在 HBM 而非 SRAM 中；在多卡下，“远”意味着当前 GPU 计算所需的数据（参数、梯度或激活值）驻留在**另一张 GPU 甚至另一台机器的显存中**。
- **优化哲学同构**：单卡靠**融合与分块（Fusion & Tiling）**将数据留在高速 Shared Memory 中；多卡靠**合理的复制与分片（Replication & Sharding）**以及**通信与计算重叠（Overlap）**来最小化跨卡/跨节点的数据传输瓶颈。

### 1.1 广义存储与通信带宽层级（Generalized Hierarchy）

在上一讲中，我们曾嫌弃单卡 HBM 的带宽太慢（相比于 `~40 TB/s` 的片上 Shared Memory）；然而一旦进入多机多卡的分布式世界，**片内 HBM 瞬间变成了“高不可攀的极速存储”**！我们可以将单卡存储层次无缝扩展为跨集群的**广义存储与互连层级（从快/小到慢/大）**：

```
+===================================================================================================+
|                     广义计算存储与互连带宽层级 (以 NVIDIA H100 / B200 集群为例)                       |
+===================================================================================================+
|                                                                                                   |
|  [Level 1: 单卡 SM 片上 (On-Chip SRAM)]                                                            |
|   • Registers (~447 TB/s) & L1 Cache / Shared Memory (~19 - 40 TB/s, 192 - 256 KB/SM)             |
|         |                                                                                         |
|         v  (慢 ~5-10x)                                                                             |
|  [Level 2: 单卡片外显存 (Single-GPU HBM)]                                                          |
|   • HBM3 / HBM3e: H100 达 3.35 - 3.9 TB/s (80 GB) | B200 达 8.0 TB/s (192 GB)                     |
|         |                                                                                         |
|         v  (慢 ~4x)                                                                                |
|  [Level 3: 单节点/单机柜多卡高速互连 (Intra-Node / Rack NVLink & NVSwitch)]                         |
|   • NVLink 4.0 (H100, 18 links): 900 GB/s 双向带宽 (通常单机 8 卡全互连)                            |
|   • NVLink 5.0 (B200, 18 links): 1.8 TB/s 双向带宽 (GB200 NVL72 支持 9 Trays x 8 = 72 卡全互连!)   |
|         |                                                                                         |
|         v  (慢 ~10-20x)                                                                            |
|  [Level 4: 跨节点 Pod 级高速网络 (Inter-Node InfiniBand / RoCE)]                                    |
|   • InfiniBand (NDR 400Gbps / XDR 800Gbps): 每卡 ~50 - 100 GB/s (走 PCIe + 专用网卡支持 GPUDirect)  |
|         |                                                                                         |
|         v  (慢 ~10-100x)                                                                           |
|  [Level 5: 跨机房/传统以太网 (Standard TCP/IP Ethernet)]                                            |
|   • 传统 Ethernet: ~200 MB/s - 25 GB/s (需经 PCIe 拷贝至 CPU 内核 Socket 缓冲区，延迟极高)           |
+===================================================================================================+
```

### 1.2 为什么我们必须使用多张 GPU？（两大刚性驱动力）

1. **突破单卡显存容量墙（Memory Capacity Bottleneck）**：
   即便是最新的 NVIDIA B200 拥有 `192 GB` HBM（H100 为 `80 GB`），如果要训练一个 **1 万亿参数（1T Parameters）** 的模型，按照 [Lecture 2](https://cs336.stanford.edu/lectures/?trace=lecture_02) 推导的混合精度训练静态显存公式（`12 ~ 16 Bytes/param`），仅存放模型参数、梯度和 AdamW 优化器状态就需要 **$12 \sim 16\text{ TB}$** 的显存，单张 GPU 根本无法装下。必须将模型状态**切分（Shard）**到多张卡上。
2. **突破单卡算力吞吐墙（Compute Throughput Bottleneck）**：
   即便是一个 7B 的小模型能够轻松塞进单张 GPU 显存，用 1 张卡训练数万亿 Token 也要耗费数月之久。我们需要将训练任务拆分到成百上千张 GPU 上并行执行以缩短墙钟时间（Wall-clock Time）——但这立即引出了**算力扩展与通信开销的权衡（Compute vs. Communication Trade-off）**。

## 2. 分布式通信基石：8 大集合通信原语（Collective Operations）

在编写多 GPU 程序时，如果我们手动管理“GPU 0 发消息给 GPU 3、GPU 2 发消息给 GPU 1”这种**点对点通信（Point-to-Point Communication）**，代码会迅速变成难以维护且容易死锁的蜘蛛网。

早在 1980 年代的高性能计算（HPC / MPI）文献中，先驱们就抽象出了一套历经时间检验的声明式编程接口——**集合通信原语（Collective Operations）**。所谓 **Collective（集合）**，是指一次性声明**整个设备组（例如全部 256 张 GPU）共同参与的标准数据交换与归约模板**，交由底层通信库（如 NCCL）根据物理网络拓扑自动寻找最优路由。

### 2.1 核心术语定义（Terminology）
- **`World Size` ($W$)**：参与当前分布式通信组的设备（进程 / GPU）总数量。在以下图解与课堂代码中，我们统一取 **`world_size = 4`**。
- **`Rank` ($r \in \{0, 1, \dots, W-1\}$)**：通信组内每个设备/进程的唯一整数编号（例如 `Rank 0`, `Rank 1`, `Rank 2`, `Rank 3`）。

### 2.2 四大基础热身原语：`Broadcast`, `Scatter`, `Gather`, `Reduce`

这四个原语都以**某一个特定节点（通常选 `Rank 0` 作为 Root 节点）**为中心进行“一对多”或“多对一”的数据分发与汇总。它们虽然很少直接出现在训练内层主循环中，却是理解后续三大核心原语的基石：

```
+---------------------------------------------------------------------------------------------------+
|                        四大基础集合通信原语 (Root = Rank 0, World Size = 4)                         |
+---------------------------------------------------------------------------------------------------+
|                                                                                                   |
|  1. Broadcast (广播: 把 Rank 0 的完整张量复制给所有 Rank，常用于初始化同步/加载 Checkpoint)            |
|     Rank 0: [0, 1, 2, 3]  ===\                 Rank 0: [0, 1, 2, 3]                               |
|     Rank 1: [ ,  ,  ,  ]  ====+── Broadcast ──> Rank 1: [0, 1, 2, 3]                              |
|     Rank 2: [ ,  ,  ,  ]  ====+                 Rank 2: [0, 1, 2, 3]                              |
|     Rank 3: [ ,  ,  ,  ]  ===/                 Rank 3: [0, 1, 2, 3]                               |
|                                                                                                   |
|  2. Scatter (散播: 把 Rank 0 的大张量切分为 W 份，第 i 份发给 Rank i)                                |
|     Rank 0: [0, 1, 2, 3]  ===\                 Rank 0: [0]                                        |
|     Rank 1: [ ]           ====+─── Scatter ───> Rank 1: [1]                                       |
|     Rank 2: [ ]           ====+                 Rank 2: [2]                                       |
|     Rank 3: [ ]           ===/                 Rank 3: [3]                                        |
|                                                                                                   |
|  3. Gather (收集, Scatter 的逆操作: 将各 Rank 的分片收集并拼接到 Rank 0 上)                          |
|     Rank 0: [0]           ===\                 Rank 0: [0, 1, 2, 3]                               |
|     Rank 1: [1]           ====+──── Gather ───> Rank 1: [ ]                                       |
|     Rank 2: [2]           ====+                 Rank 2: [ ]                                       |
|     Rank 3: [3]           ===/                 Rank 3: [ ]                                        |
|                                                                                                   |
|  4. Reduce (归约: 对各 Rank 的数据执行满足结合律/交换律的运算如 SUM/MAX，结果存入 Rank 0)             |
|     Rank 0: [0]           ===\                 Rank 0: [0+1+2+3 = 6]                              |
|     Rank 1: [1]           ====+── Reduce(SUM) ─> Rank 1: [ ]                                      |
|     Rank 2: [2]           ====+                 Rank 2: [ ]                                       |
|     Rank 3: [3]           ===/                 Rank 3: [ ]                                        |
+---------------------------------------------------------------------------------------------------+
```

> **精妙视角**：你可以把 `Gather` 看作一种特殊的 `Reduce`——只不过它的二元归约算子不是加法 `+`，而是**张量拼接（Concatenation）**！

### 2.3 大模型分布式训练的四大主力原语：`All-Gather`, `Reduce-Scatter`, `All-Reduce`, `All-to-All`

在大模型训练中，我们几乎总是希望**所有 GPU 对称地工作**（而不是只把结果汇总到 Rank 0 上让其他卡闲置）。在原语名前加上前缀 **`All-`**，就代表**“最终结果在所有 Rank 上都可用”**！

```
+===================================================================================================+
|                    大模型训练四大核心集合通信原语 (World Size = 4, 全部 Rank 对称参与)                |
+===================================================================================================+
|                                                                                                   |
|  5. All-Gather (= Gather to ALL ranks: 每个 Rank 拿自己的一片，广播拼成完整张量保存在所有 Rank 上)    |
|     Rank 0: [0]             ===\                    Rank 0: [0, 1, 2, 3]                          |
|     Rank 1: [1]             ====+─── All-Gather ───> Rank 1: [0, 1, 2, 3]                         |
|     Rank 2: [2]             ====+                    Rank 2: [0, 1, 2, 3]                         |
|     Rank 3: [3]             ===/                    Rank 3: [0, 1, 2, 3]                          |
|     * 核心场景：FSDP/ZeRO-3 前向计算前收集分片参数；Column 张量并行前向收集各卡的部分激活值。          |
|                                                                                                   |
|  6. Reduce-Scatter (按维度分别求和归约，并将第 i 维的归约结果散播存储在 Rank i 上!)                   |
|     Rank 0: [0, 1, 2, 3]    ===\                    Rank 0: [0+1+2+3 = 6 ]  (第 0 列之和归 Rank 0) |
|     Rank 1: [1, 2, 3, 4]    ====+─ Reduce-Scatter ─> Rank 1: [1+2+3+4 = 10]  (第 1 列之和归 Rank 1) |
|     Rank 2: [2, 3, 4, 5]    ====+     (SUM)         Rank 2: [2+3+4+5 = 14]  (第 2 列之和归 Rank 2) |
|     Rank 3: [3, 4, 5, 6]    ===/                    Rank 3: [3+4+5+6 = 18]  (第 3 列之和归 Rank 3) |
|     * 核心场景：ZeRO/FSDP 反向传播后汇总各数据分片的梯度，并将对应参数分片的梯度分发给其所属的 Rank。 |
|                                                                                                   |
|  7. All-Reduce (黄金恒等式: All-Reduce == Reduce-Scatter + All-Gather!)                           |
|     Rank 0: [0, 1, 2, 3]    ===\                    Rank 0: [6, 10, 14, 18]                       |
|     Rank 1: [1, 2, 3, 4]    ====+─── All-Reduce ───> Rank 1: [6, 10, 14, 18]                      |
|     Rank 2: [2, 3, 4, 5]    ====+      (SUM)        Rank 2: [6, 10, 14, 18]                      |
|     Rank 3: [3, 4, 5, 6]    ===/                    Rank 3: [6, 10, 14, 18]                       |
|     * 核心场景：标准数据并行 (DDP) 在反向传播后同步并平均所有卡上的全量梯度。                         |
|                                                                                                   |
|  8. All-to-All (分布式矩阵转置: Rank i 将自己的第 j 块数据精准发送给 Rank j!)                       |
|     Rank 0: [0,  1,  2,  3 ] ===\                   Rank 0: [0,  4,  8,  12] (收到所有卡的第 0 列)  |
|     Rank 1: [4,  5,  6,  7 ] ====+─── All-to-All ──> Rank 1: [1,  5,  9,  13] (收到所有卡的第 1 列)  |
|     Rank 2: [8,  9,  10, 11] ====+  (Transpose!)    Rank 2: [2,  6,  10, 14] (收到所有卡的第 2 列)  |
|     Rank 3: [12, 13, 14, 15] ===/                   Rank 3: [3,  7,  11, 15] (收到所有卡的第 3 列)  |
|     * 核心场景：MoE 专家并行 (Expert Parallelism) 将动态路由的 Token 分发给对应 GPU 上的专家！       |
+===================================================================================================+
```

#### 必须牢记的两大核心洞察：
1. **黄金分解恒等式（$\text{All-Reduce} \equiv \text{Reduce-Scatter} + \text{All-Gather}$）**：
   观察上面图 6、图 5 与图 7 的输入输出：如果你对输入先执行一次 `Reduce-Scatter`（让 Rank 0 拿到 `[6]`、Rank 1 拿到 `[10]`、Rank 2 拿到 `[14]`、Rank 3 拿到 `[18]`），紧接着再对这些分片执行一次 `All-Gather`，所有 Rank 就都拿到了完整的 `[6, 10, 14, 18]`——这与直接执行 `All-Reduce` **在数学结果和带宽开销上 100% 等价**！
   - **为什么要把好端端的 `All-Reduce` 拆成两半？** 因为在朴素 DDP 中我们无脑调一个 `All-Reduce`；但一旦拆成 `Reduce-Scatter` 和 `All-Gather` 两步，**我们就可以在两步之间插手干预（Intervene）**——即在 `Reduce-Scatter` 之后，让每个 Rank **只更新自己负责的那 $1/W$ 份参数与优化器状态**，然后再用 `All-Gather` 把更新后的参数同步回来！这就是著名的 **ZeRO（Zero Redundancy Optimizer）与 FSDP** 省下数倍显存的核心机密！
2. **`All-to-All` 与 MoE 负载均衡的关系**：
   当每个 Rank 发往其他每个 Rank 的数据量完全相等时，`All-to-All` 就是一个完美的跨卡矩阵转置。但在 [Lecture 4（Mixture of Experts）](https://www.youtube.com/watch?v=cKSwj_qZ8Jg) 中，由于路由器是动态看数据决定去哪个专家的，不同 Rank 发往不同专家的字节数往往是不均衡的（`torch.distributed.all_to_all_single` 支持传入不等长的 split sizes）。如果负载严重失衡，某张卡就会成为通信瓶颈——这正是我们在训练 MoE 时必须施加负载均衡损失（或无辅助损失动态偏置）的系统根源。

### 2.4 集合通信术语速记口诀（Percy's Memory Tips）
- **`Reduce`**：执行满足结合律与交换律的二元聚合操作（`SUM`, `AVG`, `MIN`, `MAX`）。
- **`Broadcast / Scatter` vs. `Gather`**：互为逆操作——`Scatter` 是发散（从 1 个中心分发到多个节点），`Gather` 是集中（从多个节点收集到 1 个中心）。
- **`All`**：修饰目标位置，表示最终结果被写入**所有设备（All Ranks）**，而非仅仅写入单个 Root 设备。

## 3. 分布式硬件网络拓扑、RDMA 与 NCCL 通信栈（`torch_distributed`）

理解了抽象的集合通信原语后，它们在物理硬件和软件栈上是如何落地的？

### 3.1 经典家用拓扑 vs. 现代 AI 数据中心拓扑

| 对比维度 | 经典单机/家用多卡拓扑 (Classic / Gaming Setup) | 现代 AI 超算数据中心拓扑 (Modern Data Center Setup) |
| :--- | :--- | :--- |
| **节点内多卡互连 (Intra-Node)** | 通过主板 **PCIe 总线** 通信（即便是最新 PCIe 7.0 x16 也仅约 `242 GB/s`，PCIe 5.0 x16 仅 `64 GB/s`），且常需绕道 CPU 内存控制器 | 通过 **NVIDIA NVLink + NVSwitch** 全互连，完全绕过 CPU，任意两卡点对点直通（H100 NVLink 4 达 **`900 GB/s`**；B200 NVLink 5 达 **`1.8 TB/s`**） |
| **NVLink 超大域扩展 (Rack-Scale)** | 仅限单主板插槽（通常 2~4 张卡） | 普通节点为 **8 卡 HGX**；而在最新 **GB200 / B200 NVL72** 机柜中，9 个计算托盘（每个 Tray 含 2 颗 Grace CPU + 8 颗 B200 GPU）通过 NVSwitch 背板组合成 **72 卡全互连的单一 NVLink 域**！ |
| **跨节点互连 (Inter-Node)** | 传统千兆/万兆 **TCP/IP 以太网（Ethernet）**（实测仅 `~200 MB/s` 至数 GB/s），延迟极高 | 采用 **InfiniBand (IB)** 交换机网络或 **RoCE（基于融合以太网的 RDMA）**，每张 GPU 配备专属高速网卡（NIC）实现百 GB/s 级直通 |

### 3.2 核心硬件加速机制：为什么 RDMA（绕过 CPU）如此关键？

Percy 在讲义中对比了跨节点通信的两种路径：
1. **传统标准以太网路径（Standard TCP/IP Ethernet —— 极慢）**：
   当节点 A 的 GPU 想要发数据给节点 B 的 GPU 时，数据必须先通过 PCIe 拷贝到节点 A 的 **CPU 主存（内核 Socket 缓冲区 Kernel Socket Buffer）** $\rightarrow$ 由 CPU 操作系统内核打包构造 TCP/IP 网络报文 $\rightarrow$ 拷贝给网卡（NIC）发送 $\rightarrow$ 节点 B 的 CPU 接收拆包后再通过 PCIe 拷贝进目标 GPU。**CPU 全程介入并伴随多次内存拷贝，延迟高达数十微秒以上！**
2. **RDMA 路径（Remote Direct Memory Access / GPUDirect RDMA —— 极快）**：
   **RDMA** 是一项允许**一张 GPU 直接读写另一台机器上 GPU 的显存（HBM）、全程零 CPU 介入（Zero-Copy, Kernel Bypass）**的核心技术！
   - 实现 RDMA 的三种主流物理载体：
     1. **NVLink + NVSwitch**（节点内 / NVL72 机柜内原生支持直接显存读写）；
     2. **InfiniBand (IB)**（跨节点超算标配，通过 PCIe Switch 让 GPU 与 IB 网卡直连，不经过 CPU）；
     3. **RoCE (RDMA over Converged Ethernet)**：以太网阵营对标昂贵 InfiniBand 的利器，在无损以太网硬件上实现绕过 CPU 的 RDMA 直通（Meta 训练 Llama 3 的 24,000 卡集群便大规模采用了 RoCE 架构）。

在 Linux 服务器上，我们可以通过运行一行命令直接检查当前机器的 GPU 与网卡拓扑连接关系：
```bash
nvidia-smi topo -m
# 输出矩阵中若显示 NV18，代表两张 GPU 之间通过 18 条 NVLink 通道全速直连（如 H100 / B200）
# 同时会显示每张 GPU 与各个网卡 (NIC / mlx5_*) 之间是通过 PIX (同一 PCIe 交换机) 还是 SYS (跨 NUMA 节点) 连接
```

### 3.3 软件通信底座：NCCL 与 `torch.distributed` 实战

- **底层引擎：NCCL（NVIDIA Collective Communication Library，发音为 "Nickel"）**：
  当你调用 `all_reduce` 时，真正干活的是 NVIDIA 的 [NCCL 库](https://www.google.com/search?q=https://www.nvidia.com/en-us/gtcfall22-s41167/)。NCCL 会自动探测底层硬件拓扑（`nvidia-smi topo`），构建跨卡/跨机最优的环状（Ring）或树状（Tree）通信路径，并**在 GPU 上启动专门的通信 CUDA Kernels** 将张量拆包发送与接收归约。
- **高层接口：PyTorch `torch.distributed`**：
  PyTorch 提供了统一的分布式编程抽象，支持针对不同硬件切换后端（Backend）：
  - **`"nccl"`**：GPU 分布式训练的绝对标准后端；
  - **`"gloo"`**：CPU 分布式通信后端（方便在无 GPU 的笔记本上调试多进程逻辑）。

#### 完整代码实战：进程启动、初始化与验证 `All-Reduce = Reduce-Scatter + All-Gather`

以下是 `lecture_07.py`（对应 [Trace Lines 126–162 & 403–416](https://cs336.stanford.edu/spring2025-lectures/?trace=var/traces/lecture_08.json)）的完整代码与真实多进程输出：

```python
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

def setup(rank: int, world_size: int):
    # 指定 Master 节点 (Rank 0) 的 IP 与端口：
    # 注意：这仅用于初始握手与元数据协调 (Rendezvous)，真正的张量数据流走高速 NCCL，绝不会走这个本地端口!
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "15623"

    # 初始化进程组：有 GPU 则用 "nccl" 后端，仅 CPU 则用 "gloo" 后端
    if torch.cuda.is_available():
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
    else:
        dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def get_device(rank: int) -> torch.device:
    return torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

def collective_operations_main(rank: int, world_size: int):
    """该函数会在 world_size (4) 个独立进程中异步并发执行 (rank = 0, 1, 2, 3)。"""
    setup(rank, world_size)

    # =========================================================================
    # 1. All-Reduce 实战验证
    # =========================================================================
    dist.barrier()  # 同步屏障：等待所有进程都执行到此行（此处为了保证打印顺序整洁）

    # 构造输入张量: Rank 0 为 [0,1,2,3], Rank 1 为 [1,2,3,4], Rank 2 为 [2,3,4,5], Rank 3 为 [3,4,5,6]
    tensor = torch.tensor([0., 1, 2, 3], device=get_device(rank)) + rank
    print(f"Rank {rank} [before all-reduce]: {tensor}", flush=True)

    # 原地 (In-place) 执行 All-Reduce 求和：async_op=False 表示阻塞等待通信完成
    dist.all_reduce(tensor=tensor, op=dist.ReduceOp.SUM, async_op=False)
    print(f"Rank {rank} [after all-reduce]: {tensor}", flush=True)

    # =========================================================================
    # 2. Reduce-Scatter 实战验证
    # =========================================================================
    dist.barrier()

    input_tensor = torch.arange(world_size, dtype=torch.float32, device=get_device(rank)) + rank
    output_tensor = torch.empty(1, device=get_device(rank))  # 输出仅为原长度的 1 / world_size!

    print(f"Rank {rank} [before reduce-scatter]: input = {input_tensor}, output = {output_tensor}", flush=True)
    dist.reduce_scatter_tensor(output=output_tensor, input=input_tensor, op=dist.ReduceOp.SUM, async_op=False)
    print(f"Rank {rank} [after reduce-scatter]: input = {input_tensor}, output = {output_tensor}", flush=True)

    # =========================================================================
    # 3. All-Gather 实战验证 (输入直接取自上一步 Reduce-Scatter 的输出!)
    # =========================================================================
    dist.barrier()

    ag_input = output_tensor
    ag_output = torch.empty(world_size, device=get_device(rank))

    print(f"Rank {rank} [before all-gather]: input = {ag_input}, output = {ag_output}", flush=True)
    dist.all_gather_into_tensor(output_tensor=ag_output, input_tensor=ag_input, async_op=False)
    print(f"Rank {rank} [after all-gather]: input = {ag_input}, output = {ag_output}", flush=True)

    # 验证结论: ag_output 在所有 Rank 上均为 [6., 10., 14., 18.]，与 All-Reduce 完全一致！
    cleanup()

# 启动方式：使用 mp.spawn 派生 4 个进程并发执行 collective_operations_main
# mp.spawn(collective_operations_main, args=(4,), nprocs=4, join=True)
```

**多进程运行真实标准输出（`stdout`）解析**：
```text
Rank 0 [before all-reduce]: tensor([0., 1., 2., 3.], device='cuda:0')
Rank 1 [before all-reduce]: tensor([1., 2., 3., 4.], device='cuda:1')
Rank 2 [before all-reduce]: tensor([2., 3., 4., 5.], device='cuda:2')
Rank 3 [before all-reduce]: tensor([3., 4., 5., 6.], device='cuda:3')
Rank 0 [after all-reduce]:  tensor([ 6., 10., 14., 18.], device='cuda:0')
Rank 1 [after all-reduce]:  tensor([ 6., 10., 14., 18.], device='cuda:1')
Rank 2 [after all-reduce]:  tensor([ 6., 10., 14., 18.], device='cuda:2')
Rank 3 [after all-reduce]:  tensor([ 6., 10., 14., 18.], device='cuda:3')

Rank 0 [after reduce-scatter]: input = tensor([0., 1., 2., 3.]), output = tensor([6.],  device='cuda:0')
Rank 1 [after reduce-scatter]: input = tensor([1., 2., 3., 4.]), output = tensor([10.], device='cuda:1')
Rank 2 [after reduce-scatter]: input = tensor([2., 3., 4., 5.]), output = tensor([14.], device='cuda:2')
Rank 3 [after reduce-scatter]: input = tensor([3., 4., 5., 6.]), output = tensor([18.], device='cuda:3')

Rank 0 [after all-gather]:     input = tensor([6.]),  output = tensor([ 6., 10., 14., 18.], device='cuda:0')
Rank 1 [after all-gather]:     input = tensor([10.]), output = tensor([ 6., 10., 14., 18.], device='cuda:1')
Rank 2 [after all-gather]:     input = tensor([14.]), output = tensor([ 6., 10., 14., 18.], device='cuda:2')
Rank 3 [after all-gather]:     input = tensor([18.]), output = tensor([ 6., 10., 14., 18.], device='cuda:3')
```
> **关于异步模式（`async_op=True`）的重要说明**：在上面的演示代码中，我们传入了 `async_op=False` 让 CPU 阻塞等待通信完成。在实际高性能训练代码中，你可以传入 `async_op=True`，此时调用会立即返回一个工作句柄 `handle = dist.all_reduce(..., async_op=True)`，让你的进程可以继续执行与该张量无关的下一层计算或数据加载（**实现通信与计算重叠 Overlap**），直到真正需要用到归约结果前再调用 `handle.wait()`！

## 4. 集合通信测速与有效带宽数学推导（`benchmarking`）

就像我们在单卡计算中通过 MFU（Model FLOPs Utilization）评估算力利用率一样，在多卡通信中，我们需要通过测量**有效通信带宽（Effective Bandwidth，单位 GB/s）**来检验 NCCL 是否跑满了 NVLink / InfiniBand 的物理带宽上限。

### 4.1 分布式 GPU 计时的“双重异步屏障法则”

在 `lecture_07.py`（Lines 178–245）中，Percy 特别强调了一个极易踩坑的细节：**为什么在计时前后，必须先调用 `torch.cuda.synchronize()`，再紧接着调用 `dist.barrier()`？**
- **第一层异步（CUDA Kernel 异步）**：NCCL 通信底层是由 GPU 上的 CUDA Kernel 执行的，相对于当前 CPU 进程是异步的。必须先调用 `torch.cuda.synchronize()` 确保本卡的通信 CUDA Kernel 真正跑完。
- **第二层异步（多进程并发异步）**：4 个 GPU 对应的 4 个 CPU Python 进程彼此之间也是异步调度的。必须再调用 `dist.barrier()` 确保所有 Rank 的进程都完成了同步，再去读取 `time.time()`。
- **课堂问答警示**：如果把顺序颠倒（先 `dist.barrier()` 再 `torch.cuda.synchronize()`），CPU 进程在把异步 CUDA Kernel 塞进队列后就会瞬间冲过 `barrier`，导致屏障完全失效！

### 4.2 `All-Reduce` 与 `Reduce-Scatter` 的通信字节数与有效带宽公式推导

假设我们在 `world_size = W`（例如 $W=4$）张 GPU 上对包含 $N = 100 \times 1024^2$ 个 `float32` 元素的张量（单张量体积 $S = 400\text{ MB}$）执行通信测速：

#### （1）`Reduce-Scatter`（以及对称的 `All-Gather`）的数据搬运量
设 `Reduce-Scatter` 最终落在每个 Rank 上的输出分片大小为 $S_{\text{out}} = \text{data\\_bytes}$（则每个 Rank 的输入总大小为 $S_{\text{in}} = W \cdot S_{\text{out}}$）。
- 为了让每个 Rank 凑齐属于自己的那 1 份总和，每个 Rank 需要将其余 $(W - 1)$ 个分片发送给对应的其他 $(W - 1)$ 个 Rank。
- 因此，单个 Rank 实际跨卡发送的净字节数为：
  $$\text{sent\\_bytes}_{\text{RS}} = S_{\text{out}} \times (W - 1) = S_{\text{in}} \times \frac{W - 1}{W}$$
- 在 `lecture_07.py` 的记账口径中，若按所有 Rank 的总发送量除以 `total_duration = W * duration`（等价于单卡发送量除以 `duration` 再乘以分摊因子），有效总线带宽为：
  $$\text{Bandwidth}_{\text{RS}} = \frac{S_{\text{out}} \times (W - 1)}{W \times \text{duration}}$$

#### （2）`All-Reduce` 的数据搬运量（2 倍法则与渐近尺度不变性）
因为 $\text{All-Reduce} \equiv \text{Reduce-Scatter} + \text{All-Gather}$，对于大小为 $S = \text{size\\_bytes}$ 的张量（可视作被切分为 $W$ 个大小为 $S / W$ 的块）：
- 在第一阶段 `Reduce-Scatter` 中，每个块需要经过 $(W - 1)$ 次跨卡传输完成归约，产生 $S \cdot \frac{W - 1}{W}$ 的单卡流量（全集群合计 $S \cdot (W - 1)$）；
- 在第二阶段 `All-Gather` 中，归约好的每个块又要广播回其余 $(W - 1)$ 张卡，**再次产生一模一样的 $S \cdot \frac{W - 1}{W}$ 流量**！
- 因此，包含收发双向工作的**总有效传输字节数存在一个关键的 $2\times$ 因子**（[NCCL 官方总线带宽 Bus Bandwidth 定义](https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md)）：
  $$\text{sent\\_bytes}_{\text{AR}} = 2 \times S \times (W - 1), \qquad \text{total\\_duration} = W \times \text{duration}$$
  $$\text{Bandwidth}_{\text{AR}} = \frac{\text{sent\\_bytes}_{\text{AR}}}{\text{total\\_duration}} = \frac{2 \cdot S \cdot (W - 1)}{W \cdot \text{duration}} \xrightarrow{W \to \infty} \mathbf{\frac{2 S}{\text{duration}}}$$

```python
# 源自 lecture_07.py (Lines 178-244): All-Reduce 与 Reduce-Scatter 带宽基准测试核心代码
def all_reduce(rank: int, world_size: int, num_elements: int):
    setup(rank, world_size)
    tensor = torch.randn(num_elements, device=get_device(rank))

    # 1. 预热 (Warmup) + 双重同步屏障
    dist.all_reduce(tensor=tensor, op=dist.ReduceOp.SUM, async_op=False)
    if torch.cuda.is_available():
        torch.cuda.synchronize()  # 先等本卡 CUDA Kernel 跑完
    dist.barrier()                # 再等所有 CPU 进程对齐

    # 2. 正式计时
    start_time = time.time()
    dist.all_reduce(tensor=tensor, op=dist.ReduceOp.SUM, async_op=False)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    dist.barrier()
    end_time = time.time()

    duration = end_time - start_time

    # 3. 计算有效通信带宽 (Effective Bus Bandwidth)
    dist.barrier()
    size_bytes = tensor.element_size() * tensor.numel()
    sent_bytes = size_bytes * 2 * (world_size - 1)  # 2x: 因为 All-Reduce = Reduce-Scatter + All-Gather
    total_duration = world_size * duration
    bandwidth = sent_bytes / total_duration
    print(f"[all_reduce] Rank {rank}: measured bandwidth = {round(bandwidth / 1024**3)} GB/s", flush=True)
    cleanup()
```

**从公式中读出的三大物理洞察**：
1. **规模不变性（Scale Invariance）**：当 GPU 数量 $W$ 增大时，$\frac{W - 1}{W} \to 1$，因此无论你是 8 卡还是 256 卡（只要在高速互联域内使用 Ring All-Reduce），完成一次大小为 $S$ 的张量 `All-Reduce` 所需的耗时渐近收敛于 $\frac{2S}{\text{BW}_{\text{link}}}$，**与卡数 $W$ 无关**！
2. **拓扑透明性**：无论底层 NCCL 选择的是 Ring（环状）还是 Tree（树状）拓扑，上述有效带宽公式均一致适用。
3. **`All-Reduce` vs. `Reduce-Scatter` 实测对比**：在单机 H100 上，对 $400\text{ MB}$ 张量执行 `All-Reduce` 耗时约 `1.6 ms`，而执行同等分片的 `Reduce-Scatter` 耗时约 `0.8 ms`（快一倍，因为没有 `2x` 流量），但两者算出的物理链路有效带宽**均稳定在 `~400+ GB/s`**（单向有效带宽接近 H100 NVLink 4.0 的物理上限）！

## 5. 分布式训练三大范式从零手写实现（MLP 全流程剖析）

在 Part 2 中，Percy 带领我们用最纯粹的 PyTorch 原语（不依赖任何高级黑盒封装），在一个多层感知机（Deep MLP，`batch_size = 128`, `num_dim = 1024`, `num_layers = 4`）上从零手写 **数据并行（Data Parallelism）**、**张量并行（Tensor Parallelism）** 与 **流水线并行（Pipeline Parallelism）**。
*为什么用 MLP 就足够代表性？* 因为正如 Lecture 1 与 Lecture 2 所证实的，在大规模 Transformer 中，MLP（FFN）层占据了 **80% 以上的参数量与 FLOPs 计算瓶颈**！

```
+===================================================================================================+
|                    三维并行切分全景示意图 (Data vs. Tensor vs. Pipeline Parallelism)                 |
+===================================================================================================+
|                                                                                                   |
|  [1. 数据并行 (Data Parallelism, DDP)] —— 沿 Batch 维度 (行方向) 切分数据，每张卡持有完整模型参数     |
|      Data [B, D]:  [ Batch Slice 0 (Rank 0) ]  --->  完整参数 W_0 .. W_{L-1} [D, D]                |
|                    [ Batch Slice 1 (Rank 1) ]  --->  完整参数 W_0 .. W_{L-1} [D, D]                |
|                    (每步反向传播后通过 All-Reduce 同步平均梯度)                                      |
|                                                                                                   |
|  [2. 张量并行 (Tensor Parallelism, TP)] —— 沿 Width 隐藏维度 (列/行方向) 切分每一层的权重矩阵        |
|      完整 Data [B, D] ---> Layer l 权重切分: [ W_l^(0) | W_l^(1) | W_l^(2) | W_l^(3) ] (各 [D, D/4])|
|      (每算完一层局部激活后，必须立即通过 All-Gather / Reduce-Scatter 跨卡同步激活值!)                  |
|                                                                                                   |
|  [3. 流水线并行 (Pipeline Parallelism, PP)] —— 沿 Depth 层数维度纵向切分网络，按微批次接力传递        |
|      Rank 0 (Layers 0..1) ──dist.send(x_micro)──> Rank 1 (Layers 2..3)                            |
|      (将大 Batch 切为多个 Micro-batches 填入流水线以压缩等待气泡 Pipeline Bubbles)                   |
+===================================================================================================+
```

---

### 5.1 范式一：数据并行（Data Parallelism / DDP，`data_parallelism_main`）

#### （1）核心切分逻辑与训练循环代码
- **数据分片，参数复制**：将全局批大小 `batch_size = 128` 均分为 `world_size = 4` 份，每个 Rank 只加载并计算属于自己的 `local_batch_size = 32` 条样本；但每个 Rank 都在显存中维护**完整的模型参数 `params` 与完整的 `AdamW` 优化器状态**。
- **与单卡训练唯一的区别（仅需插入 2 行代码！）**：
  在前向传播与 `loss.backward()` 结束时，由于每个 Rank 看到的数据不同，各卡算出的 `loss` 不同，各卡的局部梯度 `param.grad` 也不同。此时只需在 `optimizer.step()` 之前，对所有参数的梯度执行一次 **`dist.all_reduce(param.grad, op=dist.ReduceOp.AVG)`**！一旦梯度通过 `All-Reduce` 变为全卡完全相同的全局平均梯度，而且各卡初始参数与随机数种子一致，那么各卡独立执行 `optimizer.step()` 后，**所有 Rank 上的模型参数将在整个训练生命周期内严格保持逐比特一致**！

```python
import torch.nn.functional as F

def generate_sample_data() -> torch.Tensor:
    batch_size = 128
    num_dim = 1024
    return torch.randn(batch_size, num_dim)

def data_parallelism_main(rank: int, world_size: int, data: torch.Tensor, num_layers: int, num_steps: int):
    setup(rank, world_size)

    # 1. 切分数据 Batch：当前 rank 只取属于自己的 [start_index : end_index] 切片
    # (注：实际工程中每个 rank 应直接从磁盘读取自己的分片，避免主存广播瓶颈)
    batch_size = data.size(0)                              # 128
    num_dim = data.size(1)                                 # 1024
    local_batch_size = batch_size // world_size            # 128 // 4 = 32
    start_index = rank * local_batch_size
    end_index = start_index + local_batch_size
    data = data[start_index:end_index].to(get_device(rank)) # 形状: [32, 1024]

    # 2. 初始化完整模型参数与独立优化器状态（所有 rank 使用相同随机种子初始化，保证初始权重一致）
    params = [get_init_params(num_dim, num_dim, rank) for i in range(num_layers)]
    optimizer = torch.optim.AdamW(params, lr=1e-3)

    for step in range(num_steps):
        # 3. 局部前向传播 (Local Forward Pass)
        x = data
        for param in params:
            x = x @ param
            x = F.gelu(x)
        loss = x.square().mean()  # 局部 Loss 在不同 rank 上数值不同!

        # 4. 局部反向传播 (Local Backward Pass)
        loss.backward()

        # 5. 【DDP 的唯一核心步骤】：跨所有 Worker 同步并平均梯度!
        for param in params:
            dist.all_reduce(tensor=param.grad, op=dist.ReduceOp.AVG, async_op=False)

        # 6. 同步更新参数（因为初始权重相同且平均后梯度相同，更新后的 params 在所有 rank 上严格保持一致!）
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    cleanup()
```

#### （2）从朴素 DDP 迈向 Assignment 2 的两大进阶思考（Overlap 与 ZeRO-1）
1. **通信与反向计算重叠（Communication-Computation Overlap）**：
   上面的代码等整个 `loss.backward()` 全部跑完后，才用 `for` 循环逐层发起阻塞式的 `all_reduce`，这意味着反向传播期间高速网络完全闲置！在 Assignment 2 中，你将利用 PyTorch 的反向传播钩子 `param.register_post_accumulate_grad_hook`：当反向传播刚算完最后一层 $L-1$ 的梯度时，**立即异步发起 `dist.all_reduce(..., async_op=True)`**，使得第 $L-1$ 层的梯度通信与第 $L-2$ 层的反向传播计算完美重叠（甚至将多个小参数打包进固定大小的 **Buckets 桶** 中批量通信以摊薄延迟）！
2. **为什么需要 ZeRO-1（优化器状态分片，Lecture 8 预告 & Assignment 2 实战）？**
   在朴素 DDP 中，每张卡都要复制完整的模型状态（`2B` 参数 + `2B` 梯度 + `12B` FP32 主权重与 Adam 一二阶动量 = **`16 Bytes/param`**，其中优化器状态独占 75%！）。既然 $\text{All-Reduce} = \text{Reduce-Scatter} + \text{All-Gather}$，我们可以将第 5、6 步改造为 **ZeRO Stage 1**：
   - **Step A**：反向传播后，不对梯度做 `All-Reduce`，而是执行 **`Reduce-Scatter`**（通信量减半，仅 $1\times \text{\#params}$），让 Rank $r$ 只拿到属于它负责的第 $r$ 份参数的全局平均梯度；
   - **Step B**：Rank $r$ **只在显存中保存第 $r$ 份参数的 AdamW 优化器状态**（优化器显存直接降为 $1/W$！），并用刚收到的梯度更新它负责的这 $1/W$ 份参数；
   - **Step C**：所有 Rank 执行一次 **`All-Gather`**（通信量 $1\times \text{\#params}$），把各自更新好的参数分片广播拼回完整参数！**总通信量与 DDP 完全相等（零额外带宽开销），却白捡了优化器状态显存直降 $W$ 倍的巨大红利！**

---

### 5.2 范式二：张量并行（Tensor Parallelism / TP，`tensor_parallelism_main`）

当单层权重矩阵过大、或者全局 Batch Size 已经触及临界批大小（Critical Batch Size，继续增大 Batch Size 无法加速收敛）时，我们需要在**层内沿宽度维度（Width / Hidden Dimension）**切分权重矩阵——这就是 **Tensor Parallelism（张量并行）**。

#### （1）按列切分张量并行（Column-Parallel Linear Layer）原理与代码
考虑第 $i$ 层的运算 $Y = \text{GeLU}(X W_i)$，其中输入激活 $X \in \mathbb{R}^{B \times D}$，权重矩阵 $W_i \in \mathbb{R}^{D \times D}$。
我们将权重矩阵 $W_i$ **沿着列方向（Column-wise）**垂直切分为 $W=4$ 个子矩阵：
$$W_i = \begin{bmatrix} W_{i, 0} & \Big| & W_{i, 1} & \Big| & W_{i, 2} & \Big| & W_{i, 3} \end{bmatrix}, \qquad \text{其中每个分片 } W_{i, r} \in \mathbb{R}^{D \times (D/W)}$$
由于矩阵乘法对列分块天然线性独立，且 `F.gelu` 是逐元素（Elementwise）激活函数，我们有：
$$Y = \text{GeLU}(X W_i) = \begin{bmatrix} \text{GeLU}(X W_{i, 0}) & \Big| & \text{GeLU}(X W_{i, 1}) & \Big| & \text{GeLU}(X W_{i, 2}) & \Big| & \text{GeLU}(X W_{i, 3}) \end{bmatrix}$$

因此，在每一层的前向传播中：
1. 每个 Rank $r$ 持有完整输入 $X \in \mathbb{R}^{B \times D}$ 和自己的权重列分片 $W_{i, r} \in \mathbb{R}^{D \times (D/W)}$，独立计算局部激活切片 $Y_r = \text{GeLU}(X W_{i, r}) \in \mathbb{R}^{B \times (D/W)}$；
2. 为了能接着计算下一层（下一层需要完整的输入特征维度 $D$），所有 Rank 必须调用 **`dist.all_gather`** 将各自的局部激活切片 $[Y_0, Y_1, Y_2, Y_3]$ 收集起来，并沿特征维（`dim=1`）拼接还原为完整的 $X_{\text{next}} \in \mathbb{R}^{B \times D}$！

```python
def tensor_parallelism_main(rank: int, world_size: int, data: torch.Tensor, num_layers: int):
    setup(rank, world_size)

    # 1. 每个 rank 持有完整 Batch 数据 [128, 1024]，但将隐藏层维度 num_dim 切分为 local_num_dim
    data = data.to(get_device(rank))
    batch_size = data.size(0)                       # 128
    num_dim = data.size(1)                          # 1024
    local_num_dim = num_dim // world_size           # 1024 // 4 = 256 (按列切分宽度)

    # 2. 初始化分片参数：每个 rank 只分配 1 / world_size 的参数量，形状为 [1024, 256]
    params = [get_init_params(num_dim, local_num_dim, rank) for i in range(num_layers)]

    # 3. 前向传播 (Forward Pass)
    x = data
    for i in range(num_layers):
        # Step A: 在当前卡的参数列分片上计算局部激活并应用逐元素 GeLU -> 形状: [128, 256]
        x = x @ params[i]
        x = F.gelu(x)

        # Step B: 分配接收缓冲区列表 (world_size 个形状为 [128, 256] 的空张量)
        activations = [torch.empty(batch_size, local_num_dim, device=get_device(rank)) for _ in range(world_size)]

        # Step C: 通过 All-Gather 将所有 rank 的局部激活切片同步到每张卡上
        dist.all_gather(tensor_list=activations, tensor=x, async_op=False)

        # Step D: 沿隐藏层列维度 (dim=1) 拼接回完整激活张量 -> 形状恢复为 [128, 1024]
        x = torch.cat(activations, dim=1)

    print(f"[tensor_parallelism] Rank {rank}: forward pass produced activations {summarize_tensor(x)}", flush=True)
    # 反向传播留作课后思考练习（见下方完整推导与代码实现！）
    cleanup()
```

#### （2）深度补全：张量并行反向传播的对偶性推导（Forward `All-Gather` $\leftrightarrow$ Backward `Reduce-Scatter` / `All-Reduce`）
在课堂上，一位同学敏锐地提问：*“在张量并行的反向传播中会发生什么？PyTorch Autograd 会自动帮我们通信吗？”*
Percy 回答道：如果你直接调普通的 `dist.all_gather`，它不会自动在反向传播时插入集合通信，我们需要自己管理反向通信原语，而且**前向的 `All-Gather` 与反向的归约操作存在优美的数学对偶性（Duality）**！

让我们严格推导第 $i$ 层的反向传播：
- 在前向传播中：$Z_r = X W_{i, r} \in \mathbb{R}^{B \times (D/W)}$，$Y_r = \text{GeLU}(Z_r)$，然后通过拼接得到下一层输入 $Y = [Y_0 \mid Y_1 \mid \dots \mid Y_{W-1}] \in \mathbb{R}^{B \times D}$。
- 在反向传播中，假设上游传回了关于 $Y$ 的完整梯度 $\nabla Y = \frac{\partial \mathcal{L}}{\partial Y} = [\nabla Y_0 \mid \nabla Y_1 \mid \dots \mid \nabla Y_{W-1}]$：
  1. **切分上游梯度并算局部激活前梯度**：Rank $r$ 取出对应自己列分片的 $\nabla Y_r \in \mathbb{R}^{B \times (D/W)}$，计算 $\nabla Z_r = \nabla Y_r \odot \text{GeLU}'(Z_r)$。
  2. **计算当前卡负责的权重分片梯度（无需跨卡通信！）**：
     $$\nabla W_{i, r} = X^\top \nabla Z_r \in \mathbb{R}^{D \times (D/W)}$$
  3. **计算关于输入 $X$ 的梯度（必须跨卡求和归约！）**：
     因为前向时同一个输入 $X$ 同时发给了所有 $W$ 个列分片参与乘加：$Z_0 = X W_{i,0}, \dots, Z_{W-1} = X W_{i, W-1}$，根据多元链式法则，总梯度等于各分片回传梯度的**总和**：
     $$\nabla X = \sum_{r=0}^{W-1} \underbrace{\nabla Z_r W_{i, r}^\top}_{\widetilde{\nabla X}_r \in \mathbb{R}^{B \times D}}$$
     因此，每个 Rank $r$ 先算出局部的部分和矩阵 $\widetilde{\nabla X}_r = \nabla Z_r W_{i, r}^\top$，然后必须调用 **`dist.all_reduce(grad_x, op=dist.ReduceOp.SUM)`**（或者如果上一层只需要分片梯度，则调用 **`dist.reduce_scatter_tensor`**）将所有 Rank 的部分和加起来！
  > **工业界 Megatron-LM 进阶技巧预告（Lecture 8）**：在上面的朴素 Column-Parallel 实现中，每层都要做一次 `All-Gather`。而在真实的 Transformer MLP（含两个线性层 $W_{\text{up}}$ 和 $W_{\text{down}}$）中，[Megatron-LM](https://arxiv.org/abs/1909.08053) 将第 1 层做成 **Column-Parallel**（输出分片 $Y_r = \text{GeLU}(X W_{1, r})$ **直接不通信**！），紧接着将第 2 层做成 **Row-Parallel（按行切分 $W_{2, r}$）**，直接算出部分和 $O_r = Y_r W_{2, r}$，最后只需在整個 MLP 结束时做 **1 次 `All-Reduce`** 即可，直接把通信量砍半！

---

### 5.3 范式三：流水线并行（Pipeline Parallelism / PP，`pipeline_parallelism_main`）

当模型极其深、或者我们需要跨多台没有 NVLink 连接的服务器（Inter-Node）切分模型时，我们可以沿着**深度/层数维度（Depth / Layer Dimension）**切分——让每个 Rank 只负责连续的几层，这就是 **Pipeline Parallelism（流水线并行）**。

#### （1）为什么必须引入微批次（Micro-batches）？流水线气泡（Pipeline Bubbles）图解
假设我们有 `num_layers = 4` 层网络，切分在 `world_size = 2` 张 GPU 上（Rank 0 负责 Layer 0..1，Rank 1 负责 Layer 2..3）：
- **朴素整批流水线（No Micro-batching，气泡率高达 50%！）**：
  当 Rank 0 在吭哧吭哧计算整个 `batch_size = 128` 的前两层时，Rank 1 没有任何输入数据，只能**100% 空转干等（Idle / Bubble）**；等 Rank 0 把激活发给 Rank 1 后，Rank 1 开始算后两层，此时 Rank 0 又陷入空转等待！
- **微批次流水线（Micro-batching，[GPipe, Huang et al., 2018](https://arxiv.org/abs/1811.06965) 核心思想）**：
  我们将全局 Batch（`128`）切分为 `num_micro_batches = 4` 个微批次（每个 `micro_batch_size = 32`）。一旦 Rank 0 算完第 0 个微批次 `F0`，**立即通过点对点通信 `dist.send` 发送给 Rank 1**，自己马不停蹄开始算 `F1`！此时 Rank 0（算 `F1`）与 Rank 1（算 `F0`）实现了真正的并行流水作业，大幅压缩了首尾空闲的 **Pipeline Bubble（流水线气泡）**！

```
+---------------------------------------------------------------------------------------------------+
|                   朴素流水线 vs. 微批次流水线 (World Size = 2, Num Micro-batches M = 4)             |
+---------------------------------------------------------------------------------------------------+
|                                                                                                   |
|  [A. 无微批次 (M = 1): 任何时刻只有 1 张卡在干活，气泡浪费 = 50%!]                                    |
|   时间轴 t ---> |-------- T_step --------|-------- T_step --------|                               |
|   Rank 0 (L0,1): [   Forward (Batch 128)  ] [      Bubble (空闲)     ]                             |
|   Rank 1 (L2,3): [      Bubble (空闲)     ] [   Forward (Batch 128)  ]                             |
|                                                                                                   |
|  [B. 切分为 4 个微批次 (M = 4): 首尾气泡大幅缩小，中间阶段双卡 100% 满载并发!]                        |
|   时间轴 t ---> |--t0--|--t1--|--t2--|--t3--|--t4--|                                              |
|   Rank 0 (L0,1): [ F0 ] [ F1 ] [ F2 ] [ F3 ] [空闲]                                               |
|                       \      \      \      \   (点对点 dist.send / dist.recv)                      |
|                        v      v      v      v                                                     |
|   Rank 1 (L2,3): [空闲] [ F0 ] [ F1 ] [ F2 ] [ F3 ]                                               |
+---------------------------------------------------------------------------------------------------+
```

#### （2）完整代码实现：微批次切分与点对点 `dist.send` / `dist.recv`

以下是 `lecture_07.py`（Lines 347–400）的流水线并行实现：

```python
def pipeline_parallelism_main(rank: int, world_size: int, data: torch.Tensor, num_layers: int, num_micro_batches: int):
    setup(rank, world_size)

    # 1. 基础配置: batch_size = 128, num_dim = 1024, world_size = 2, num_layers = 4
    data = data.to(get_device(rank))
    batch_size = data.size(0)
    num_dim = data.size(1)

    # 2. 按层深 (Depth) 切分网络：每个 rank 只持有 local_num_layers = 4 // 2 = 2 层完整宽度的权重 [1024, 1024]
    local_num_layers = num_layers // world_size
    local_params = [get_init_params(num_dim, num_dim, rank) for i in range(local_num_layers)]

    # 3. 将全局 Batch 切分为 num_micro_batches (4) 个微批次，以压缩流水线气泡 (Bubble)
    micro_batch_size = batch_size // num_micro_batches  # 128 // 4 = 32
    if rank == 0:
        # 第一个舞台 (Stage 0) 直接将原始输入数据沿 batch 维切分为 4 个微批次元组
        micro_batches = data.chunk(chunks=num_micro_batches, dim=0)
    else:
        # 后续舞台预分配用于接收上游激活值的空张量缓冲区
        micro_batches = [torch.empty(micro_batch_size, num_dim, device=get_device(rank)) for _ in range(num_micro_batches)]

    # 4. 流水线前向传播循环
    for x in micro_batches:
        # Step A: 如果不是第一个 Stage，阻塞接收来自上一级 (rank - 1) 发来的微批次激活张量
        if rank - 1 >= 0:
            dist.recv(tensor=x, src=rank - 1)

        # Step B: 依次穿过当前 Stage 负责的局部层 (local_params)
        for param in local_params:
            x = x @ param
            x = F.gelu(x)

        # Step C: 如果不是最后一个 Stage，将算好的微批次激活点对点发送给下一级 (rank + 1)
        if rank + 1 < world_size:
            print(f"[pipeline_parallelism] Rank {rank}: sending {summarize_tensor(x)} to rank {rank + 1}", flush=True)
            dist.send(tensor=x, dst=rank + 1)

    # 注：工业级实现需将同步的 send/recv 替换为异步的 dist.isend / dist.irecv 以重叠通信与计算！
    cleanup()
```

## 6. 全景总结：如何根据硬件拓扑选择与组合并行策略？

在讲座尾声，Percy 总结了五大并行维度（**Data, Tensor, Pipeline, Sequence, Expert**）的本质特征与硬件映射法则，这也是你在设计大规模训练集群架构时的黄金决策表：

| 并行策略 | 切分维度 (Sharding Axis) | 核心通信原语 | 通信频率与数据量特征 | 适用硬件层级与典型组合位置 | 核心瓶颈与局限性 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **数据并行 (DDP / ZeRO / FSDP)** | **Batch**（数据样本维 $B$） | `All-Reduce` (DDP)<br>或 `Reduce-Scatter` + `All-Gather` (ZeRO/FSDP) | **每步（Step）或每层 1 次**；总通信量正比于参数量 $2N$（或 $3N$） | **最外层扩展（跨节点 Inter-Node IB / RoCE）**，可将集群规模扩展至数千卡 | 受限于 **Critical Batch Size（临界批大小）**：全局 Batch 过大后增加卡数不再加速收敛；DDP 不省静态显存 |
| **张量并行 (Tensor Parallelism, TP)** | **Width**（层内隐藏维度 $D$） | `All-Gather` / `Reduce-Scatter` / `All-Reduce` | **每一层前反向均需同步**；通信量正比于激活体积 $B S D$，对延迟极其敏感！ | **最内层扩展（严格限制在单节点 8 卡 NVLink 或 NVL72 机柜内！）**，绝不跨慢速节点 | 跨节点通信开销不可接受；切分过细会导致单卡矩阵过小、算术强度跌破 Roofline 拐点 |
| **流水线并行 (Pipeline Parallelism, PP)** | **Depth**（网络层数维度 $L$） | 点对点 `send` / `recv` (`isend` / `irecv`) | **仅在相邻 Stage 交界层传输 1 次**微批次激活 $b_{\text{micro}} S D$，通信量极小！ | **跨节点扩展（Inter-Node IB / Ethernet，甚至跨洲际去中心化训练）** | 存在 **Pipeline Bubbles（流水线气泡）**，需精细编排微批次（1F1B / Zero-Bubble）与重计算 |
| **序列/上下文并行 (Sequence / Context)** | **Length**（序列长度维度 $S$） | `All-Gather` / `Reduce-Scatter` / Ring P2P | 分摊长序列激活显存（LayerNorm/Dropout 激活及 Attention KV） | 配合 TP 在节点内或高速域内使用 | 需与 Attention 分块计算深度耦合 |
| **专家并行 (Expert Parallelism, EP)** | **Experts**（MoE 专家维度 $N_r$） | **`All-to-All`** | 动态将 Token 分发至目标专家所在 GPU 并收回 | 跨多节点或 NVL72 高速域（如 DeepSeek-V3 跨节点 EP） | 强依赖路由负载均衡，易受木桶效应（Straggler）拖累 |

> **Percy 的终极哲思（Store vs. Recompute vs. Communicate）**：
> 回顾整个系统模块，当我们面对一个中间张量时，本质上只有三种选择：**（1）存储在本地显存（Store in local HBM）**；**（2）丢弃并在需要时重新计算（Recompute，如 Activation Checkpointing）**；**（3）存储在另一张 GPU 的显存中并在需要时通信传输（Store on remote GPU & Communicate）**。例如，朴素 DDP 选择让每张卡都冗余执行一次 `optimizer.step()`（Recompute parameter update），换取的是永远不需要跨卡传输庞大的优化器状态！

## 7. 课后自测题与深度解析（直通 Assignment 2 分布式实战）

### 练习 1：`All-Reduce`、`Reduce-Scatter` 与 `All-Gather` 的精确通信账单与耗时计算
**问题**：假设在一个包含 `world_size = 8` 张 H100 GPU 的单机节点上，每张卡的 NVLink 双向有效总线带宽为 $\text{BW} = 400\text{ GB/s}$。现在每张卡上都有一个形状为 `[16384, 16384]` 的 `bfloat16` 梯度张量（单张量大小 $S = 16384^2 \times 2\text{ Bytes} = 512\text{ MB} = 0.5\text{ GB}$）。
1. 如果直接调用 `dist.all_reduce(grad)`，根据本讲推导的公式，单次 `All-Reduce` 的理论耗时是多少毫秒？
2. 如果改用 **ZeRO-1（优化器状态分片）**，先执行一次 `dist.reduce_scatter_tensor`，各卡更新完自己负责的 $1/8$ 参数后，再执行一次 `dist.all_gather_into_tensor`。请问 `Reduce-Scatter` 和 `All-Gather` 各自的理论耗时是多少？合起来的总通信耗时与 `All-Reduce` 相比如何？每张卡的 AdamW 优化器状态显存（原为 $8\text{ Bytes/param}$）从多少 GB 降到了多少 GB？

**深度解析**：
1. **`All-Reduce` 理论耗时**：
   由公式 $\text{BW} = \frac{2 \cdot S \cdot (W - 1)}{W \cdot \text{duration}}$ 可得：
   $$\text{duration}_{\text{AR}} = \frac{2 \times 0.5\text{ GB} \times \frac{7}{8}}{400\text{ GB/s}} = \frac{0.875\text{ GB}}{400\text{ GB/s}} = \mathbf{2.1875\text{ ms}}$$
2. **ZeRO-1（`Reduce-Scatter` + `All-Gather`）耗时与显存收益**：
   - `Reduce-Scatter` 的输出分片大小为 $S_{\text{out}} = S / 8 = 64\text{ MB}$，其耗时恰好是 `All-Reduce` 的一半：
     $$\text{duration}_{\text{RS}} = \frac{S \cdot \frac{W - 1}{W}}{\text{BW}} = \frac{0.5\text{ GB} \times \frac{7}{8}}{400\text{ GB/s}} = \mathbf{1.09375\text{ ms}}$$
   - `All-Gather` 的耗时完全对称：$\text{duration}_{\text{AG}} = \mathbf{1.09375\text{ ms}}$。
   - **总通信耗时**：$\text{duration}_{\text{RS}} + \text{duration}_{\text{AG}} = 1.09375 + 1.09375 = \mathbf{2.1875\text{ ms}}$，**与 `All-Reduce` 分毫不差（零额外通信代价）**！
   - **优化器显存节省**：单个参数矩阵含 $N = 16384^2 = 268,435,456$ 个参数。原本每张卡需存储全量 AdamW 一二阶动量（$8\text{ Bytes/param}$），占用 $268.4\text{M} \times 8 = \mathbf{2.0\text{ GB}}$；开启 ZeRO-1 后被 8 张卡均摊，每张卡仅需 $\mathbf{0.25\text{ GB}}$（节省了 **87.5%** 的优化器显存）！

### 练习 2：补全 `lecture_07.py` 留下的课后作业 —— 手写张量并行（TP）与流水线并行（PP）的反向传播代码
**问题**：在 `lecture_07.py` 的 `tensor_parallelism_main` 和 `pipeline_parallelism_main` 末尾，Percy 留下了 `# Backward pass: homework exercise`。请分别写出使 PyTorch `loss.backward()` 能够正确跨卡反向传播梯度的实现方案！

**深度解析**：
#### （1）张量并行（Column-Parallel）支持自动求导的 `autograd.Function` 实现
在 `tensor_parallelism_main` 中，如果我们用一个自定义的 `torch.autograd.Function` 封装前向的 `All-Gather`，在其 `backward` 方法中实现对应的 **`Reduce-Scatter`**（对各卡算出的关于拼接张量的梯度切分求和），就能直接调用 `loss.backward()`：

```python
class AllGatherColumnParallel(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_local: torch.Tensor, world_size: int) -> torch.Tensor:
        ctx.world_size = world_size
        # 前向：将各 rank 的 [batch_size, local_num_dim] 收集并沿 dim=1 拼接为 [batch_size, num_dim]
        activations = [torch.empty_like(x_local) for _ in range(world_size)]
        dist.all_gather(tensor_list=activations, tensor=x_local)
        return torch.cat(activations, dim=1)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        # 反向：grad_output 形状为 [batch_size, num_dim]
        # 由于下一层在所有 rank 上都对完整的 x 计算了梯度（或者各 rank 只需拿回自己负责的第 rank 列分片的梯度之和），
        # 我们将 grad_output 沿 dim=1 切分为 world_size 块，并通过 reduce_scatter 求和归约回当前 rank 的 local_grad!
        grad_chunks = list(grad_output.chunk(ctx.world_size, dim=1))
        grad_chunks = [chunk.contiguous() for chunk in grad_chunks]
        grad_local = torch.empty_like(grad_chunks[0])
        dist.reduce_scatter(output=grad_local, input_list=grad_chunks, op=dist.ReduceOp.SUM)
        return grad_local, None
```

#### （2）流水线并行（Pipeline Parallelism）支持微批次反向传播的实现
在流水线并行中，每个 Stage $r$ 在前向传播时需要把每个微批次的输入缓存下来（`saved_inputs`）并开启 `requires_grad_(True)`，以及保存输出（`saved_outputs`）。反向传播时，按微批次从最后一个 Stage 往前点对点传递梯度 `grad_output`：

```python
# 在 pipeline_parallelism_main 的前向之后，追加流水线反向传播循环：
for m in range(num_micro_batches):
    x_in = saved_inputs[m]     # 当前微批次进入本 Stage 时的输入 (requires_grad=True)
    x_out = saved_outputs[m]   # 当前微批次离开本 Stage 时的输出

    if rank == world_size - 1:
        # 最后一个 Stage 直接计算 Loss 并触发 backward
        loss = (x_out.square().mean()) / num_micro_batches
        loss.backward()
    else:
        # 中间 Stage 从下游 Stage (rank + 1) 接收关于 x_out 的梯度 dL/dx_out
        grad_out = torch.empty_like(x_out)
        dist.recv(tensor=grad_out, src=rank + 1)
        # 触发本 Stage 局部计算图的反向传播，同时算出 local_params.grad 和 x_in.grad!
        x_out.backward(grad_out)

    # 如果不是第一个 Stage，将关于本 Stage 输入的梯度 x_in.grad 发送给上游 Stage (rank - 1)
    if rank - 1 >= 0:
        dist.send(tensor=x_in.grad, dst=rank - 1)
```
