---
title: Stanford CS336 (Spring 2026) Lecture 2 深度课程笔记：PyTorch (einops) & Resource Accounting
date: '2026-10-25 10:45:00'
categories:
  - 笔记
  - CS336
tags:
  - 大语言模型
  - CS336
  - PyTorch
  - einops
  - 资源记账
  - FLOPs
  - 显存优化
description: Stanford CS336 语言模型从零构建第二讲深度笔记：系统剖析浮点精度（FP32到NVFP4）、优雅的 einops 张量代数、算术强度与 Roofline 性能分析模型、训练显存四大金刚精算以及梯度累积与激活检查点优化。
mathjax: true
---

# Stanford CS336 (Spring 2026) Lecture 2 深度课程笔记：PyTorch (`einops`) & Resource Accounting

> **课程主页**：[Stanford CS336: Language Modeling from Scratch (Spring 2026)](https://cs336.stanford.edu/)  
> **主讲教师**：Percy Liang  
> **配套资源**：[Lecture 2 视频](https://www.youtube.com/watch?v=kuYAsz7zspQ) · [交互式执行轨迹 (`lecture_02`)](https://cs336.stanford.edu/lectures/?trace=lecture_02) · [课堂实录版轨迹 (`lecture_02_recording`)](https://cs336.stanford.edu/lectures/?trace=lecture_02_recording) · [配套开源书《How to Scale Your Model》](https://jax-ml.github.io/scaling-book/)

---

## 1. 引言：为什么要做资源记账（Resource Accounting）？

### 1.1 Marin 项目 $10^{23}$ FLOPs 预注册预测实证汇报

在进入正题前，Percy 播报了上一讲提到的开源项目 [Marin](https://marin.community/) 的最新进展：团队在 $10^{23}$ FLOPs 规模上的训练刚刚跑完，通过在更小算力规模上绘制 IsoFLOP 曲线并拟合扩展定律（Scaling Law），**最终真实训练收敛的验证集 Loss 与预注册（Pre-registered）预测值误差在 $0.05$ 以内**！这一实证再次印证了系统化资源规划与可预测性在大规模预训练中的核心地位。

### 1.2 核心思维模式（Mindset）：带着性能账本写每一行代码

回顾 CS336 的终极问题：**在给定计算与显存资源（Compute & Memory）的前提下，如何最大化计算效率（Computational Efficiency），训练出性能最好的模型？**
要优化效率，前提是能够精确量化任意给定计算的资源消耗。本讲不涉及玄学的机器学习调参，而是聚焦于确定性的**系统工程底座**：
* **机制（Mechanics）**：完全透明的 PyTorch 张量内存布局、浮点精度表示、`einops` 张量代数、自动求导与优化器状态更新。
* **思维模式（Mindset）**：养成**“每写一行代码，脑中自动浮现其显存占用（Bytes）与浮点计算量（FLOPs）”**的肌肉记忆。
* **直觉（Intuitions）**：建立对算力、带宽、算术强度（Arithmetic Intensity）与硬件瓶颈的数量级体感。

### 1.3 两个经典的“信封背面估算”（Napkin Math）开胃问题

在面试或架构设计评审中，系统工程师需要能在几秒钟内完成以下两类估算：

#### 问题一：在 1024 张 H100 GPU 上，将 70B 参数的模型训练 15T Tokens（Llama-3 70B 级别）需要多少天？

* **第一步：计算总浮点运算量（Total FLOPs）**
  根据预训练黄金近似公式 $C \approx 6ND$（将在第 5 节严格推导）：
  $$\text{total\_flops} = 6 \times (70 \times 10^9) \times (15 \times 10^{12}) = 6.3 \times 10^{24}\text{ FLOPs}$$
* **第二步：查阅硬件单卡有效峰值算力（H100 BF16 Dense）**
  [NVIDIA H100 官方参数表](https://www.nvidia.com/en-us/data-center/h100/) 标注的 BF16 Tensor Core 算力为 `1979 TFLOP/s`，但要注意规格表小字标注该数值包含了 **2:4 结构化稀疏（Sparsity）**！对于常规稠密矩阵乘法（Dense MatMul），实际硬件理论峰值必须减半：
  $$\text{h100\_flop\_per\_sec} = \frac{1979 \times 10^{12}}{2} = 989.5 \times 10^{12}\text{ FLOP/s}$$
* **第三步：引入模型算力利用率（MFU, Model FLOPs Utilization）**
  由于显存搬运、跨卡通信与非矩阵乘法算子的开销，实际训练能达到的 **MFU 通常在 $0.5$（50%）左右**。因此 1024 张卡每天能提供的有效算力为：
  $$\text{flops\_per\_day} = 989.5 \times 10^{12} \times 0.5 \times 1024 \times (60 \times 60 \times 24) \approx 4.378 \times 10^{22}\text{ FLOPs/day}$$
* **第四步：得出训练天数**
  $$\text{days} = \frac{6.3 \times 10^{24}}{4.378 \times 10^{22}} \approx 143.9\text{ 天}$$

```python
# 源自 lecture_02.py 的信封估算代码
total_flops = 6 * 70e9 * 15e12
h100_flop_per_sec = 1979e12 / 2
mfu = 0.5
flops_per_day = h100_flop_per_sec * mfu * 1024 * 60 * 60 * 24
days = total_flops / flops_per_day  # ~143.9 days
```

#### 问题二：使用单机 8 张 H100（每张 80GB HBM 显存）配合 AdamW 混合精度训练，最多能装下多大参数量的模型？

* **第一步：计算每个参数的静态显存占用（Bytes per Parameter）**
  在标准混合精度训练（Mixed Precision Training）中，模型状态由三部分组成：
  1. **参数权重（Parameters）**：使用 `bf16` 存储，占 **2 Bytes**；
  2. **梯度（Gradients）**：使用 `bf16` 存储，占 **2 Bytes**；
  3. **AdamW 优化器状态（Optimizer States）**：为了数值稳定性，一阶动量 $m$ 和二阶动量 $v$ 必须使用 `fp32` 存储，占 **$4 + 4 = 8\text{ Bytes}$**（若额外保存一份 `fp32` Master Weights 则还需 $+4$ Bytes，课堂基准按 $2+2+4+4$ 计算）。
  $$\text{bytes\_per\_parameter} = 2 + 2 + (4 + 4) = 12\text{ Bytes/param}$$
* **第二步：计算 8 卡总显存可容纳的参数上限**
  $$\text{num\_parameters} = \frac{80 \times 10^9\text{ Bytes} \times 8}{12\text{ Bytes/param}} \approx 5.33 \times 10^{10}\text{ (53.3B 参数)}$$
* **重要前提与局限（Caveats）**：这个 53.3B 只是一个**理论绝对上界**，因为它还没有计入动态的**激活值显存（Activations Memory）**——激活值大小取决于批大小（Batch Size）与序列长度（Sequence Length），在长上下文或大 Batch 下甚至会反超静态模型状态。

## 2. 张量内存记账与浮点表示体系（Tensor Memory & Floating-Point Formats）

### 2.1 万物皆张量（Everything is a Tensor）

在深度学习系统中，**输入数据（Data）、模型参数（Parameters）、梯度（Gradients）、优化器状态（Optimizer State）以及中间激活值（Activations）**全部以张量（Tensor）的形式驻留在内存中。
* 查看 [DeepSeek-V3.2 (2025)](https://arxiv.org/abs/2512.00001) 在 Hugging Face 上的权重文件索引，本质上就是成百上千个不同形状（Shape）与精度（Precision）的命名张量集合。
* 张量的阶（Rank / Arity）即其维度数量：向量为 Rank 1，矩阵为 Rank 2；在 Transformer 注意力层中，我们最常打交道的是 **Rank 4 张量**：`(B, S, H, D)`，分别对应 `Batch Size`、`Sequence Length`、`Number of Heads` 和 `Hidden Dimension per Head`。

张量占用的字节数遵循极其简单的铁律：
$$\text{Memory Usage (Bytes)} = \texttt{x.numel()} \times \texttt{x.element\_size()}$$

```python
import torch

def get_memory_usage(x: torch.Tensor) -> int:
    return x.numel() * x.element_size()

# 示例 1：默认 float32 的 4x8 矩阵
x = torch.zeros(4, 8)
assert x.dtype == torch.float32        # PyTorch 默认浮点类型为 fp32
assert x.numel() == 4 * 8              # 32 个元素
assert x.element_size() == 4           # 每个 fp32 元素占 4 字节
assert get_memory_usage(x) == 128      # 共 128 字节

# 示例 2：GPT-3 (d_model = 12288) 单层 MLP 中的一个升维矩阵 (4*d_model, d_model)
gpt3_mlp_weight = torch.empty(12288 * 4, 12288, dtype=torch.float32)
assert get_memory_usage(gpt3_mlp_weight) == 2304 * 1024 * 1024  # 仅单个矩阵就高达 2.3 GB！
```

### 2.2 深度学习浮点数格式全解：从 FP32 到 NVFP4

为什么降低浮点精度在 LLM 训练中如此重要？一方面，它直接**将显存占用砍半甚至降为 1/4**，让单卡能塞下更大的模型或更大的 Batch Size；另一方面，更低比特的数据在 Tensor Core 上的计算吞吐率成倍提升，且**减少了从 HBM 搬运到计算单元的字节数，从而直接加速了访存受限（Memory-bound）算子**。

科学计算传统上以 `fp32`（单精度）为基准、甚至使用 `fp64`（双精度）；但深度学习具有高度的随机性与噪声容忍度。任何一个 IEEE 754 浮点数都由三部分比特构成：
1. **符号位（Sign, 1 bit）**：决定正负；
2. **指数位（Exponent bits, E）**：决定**动态范围（Dynamic Range，即能表示的最大值与最小非零正数跨度）**，防止上溢（Overflow $\rightarrow \text{Inf}$）和下溢（Underflow $\rightarrow 0$）；
3. **尾数位/有效数位（Mantissa / Fraction bits, M）**：决定**数值分辨率/精度（Resolution / Precision）**。

下表完整对比了 Lecture 2 涵盖的五代核心浮点格式：

| 数据类型 | 总位数 | 字节数 | 比特分配 (Sign / E / M) | 核心数值特征 | 在大模型训练中的定位与工程权衡 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`fp32` (Single)** | 32 bits | 4 Bytes | `1 / 8 / 23` | 动态范围极广 ($\approx 10^{-38} \sim 3.4\times 10^{38}$)，分辨率高 | PyTorch 默认类型；显存与算力开销大，现主要用于存储 **Optimizer States**、Master Weights 及 Softmax/指数运算 |
| **`fp16` (Half)** | 16 bits | 2 Bytes | `1 / 5 / 10` | 仅 5 位指数，最大正数 `65504`，最小正规数 $\approx 6\times 10^{-5}$（次正规数约 $6\times 10^{-8}$） | **动态范围极窄**！`1e-8` 直接下溢为 `0`；训练时小梯度极易下溢消失、激活值极易上溢出现 `NaN`，必须配合复杂的 Loss Scaling |
| **`bf16` (Brain Float)** | 16 bits | 2 Bytes | `1 / 8 / 7` | Google Brain (2018) 发明：**保留与 `fp32` 完全相同的 8 位指数**，牺牲尾数至 7 位 | **现代 LLM 预训练的黄金甜点**：显存减半，且完全具备 `fp32` 的动态范围（`1e-8 != 0`），消除了下溢风险 |
| **`fp8` (E4M3 / E5M2)** | 8 bits | 1 Byte | `1/4/3` 或 `1/5/2` | 2022 年标准化（[Micikevicius et al., 2022](https://arxiv.org/abs/2209.05433)）：`E4M3` 范围 `[-448, 448]` 精度高；`E5M2` 范围 `[-57344, 57344]` 动态范围大 | H100/B200 原生支持（通常前向激活与权重用 `E4M3`，反向梯度用 `E5M2`），需配合 NVIDIA Transformer Engine 动态缩放 |
| **`nvfp4` (Block FP4)** | 4 bits | 0.5 Byte | `1 / 2 / 1` + Block Scale | NVIDIA (2025) 推出：单值仅 15 个离散点 `[-6, -4, -3, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 3, 4, 6]` | 采用**分块缩放（Block Scaling）**共享缩放因子以扩展动态范围；[NVIDIA Nemotron 3 Super (2026)](https://arxiv.org/abs/2603.00001) 已实现原生 NVFP4 预训练 |

#### 关键代码验证：`fp16` 的下溢灾难 vs. `bf16` 的稳健性

```python
# 1. FP16 的下溢问题 (Underflow)
x_fp16 = torch.tensor([1e-8], dtype=torch.float16)
assert x_fp16 == 0  # 1e-8 在 fp16 下直接变成了 0.0！梯度直接丢失！

# 2. BF16 保持了与 FP32 相同的 8 位指数动态范围
x_bf16 = torch.tensor([1e-8], dtype=torch.bfloat16)
assert x_bf16 != 0  # 无下溢！成功保留非零数值
```

#### 课堂问答精选：关于分块缩放（Block Scaling）与 1-bit 模型
* **问：在 `nvfp4` 中，如果一个 Block 内的元素共享同一个缩放因子（Scale Factor），是怎么提升动态范围的？**
  * **Percy 解答**：在同一个 Block 内部，每个元素只有 4 bits 的自由度（只能在 `-6` 到 `6` 的离散值之间相对变化）；但整个 Block 乘上了一个更高精度的共享缩放因子。因此从单个数值的绝对大小来看，它可以跨越大得多的动态范围，唯一的限制是**同一个局部 Block 内的相邻元素不能一个极大、另一个极小（不能偏离邻居太远）**。
* **问：现在有人研究 1-bit 语言模型（如 BitNet），我们能直接用 1-bit 训练吗？**
  * **Percy 解答**：必须严格区分**训练（Training）**与**推理量化（Post-Training Quantization）**。绝大多数极低比特（1-bit / 2-bit）工作是先在 `bf16` 下训练好模型，再量化压缩用于推理；直接从头用 1-bit 做大规模前沿预训练目前尚未产生真正有竞争力的模型。

### 2.3 混合精度训练（Mixed Precision Training）与 PyTorch AMP

既然 `bf16` 尾数只有 7 bits（分辨率低），当我们在优化器中对成千上万步的微小梯度或梯度平方（$g^2$）做累加平均时，如果直接用 `bf16` 存储，大数吃小数（精度截断）会导致参数完全无法更新或训练发散。
因此，遵循 [Micikevicius et al. (2017)《Mixed Precision Training》](https://arxiv.org/abs/1710.03740) 的经典范式，现代大模型训练一律采用**混合精度策略**：
* **前向/反向主体计算（Parameters, Activations, Gradients）**：使用 `bf16`（或 `fp8`），享受高吞吐与低显存。
* **累加与敏感算子（Optimizer States、Master Weights、逐元素指数 `exp` / Softmax / LayerNorm 方差累加）**：保持在 **`fp32`**。

在 PyTorch 中，可以使用自动混合精度上下文管理器 `torch.amp.autocast`，它会自动将安全的矩阵乘法（MatMul / Linear / Conv）转换为 `bf16`，而将数值敏感的 `exp`、`sum` 等保留为 `fp32`：

```python
with torch.amp.autocast("cuda", dtype=torch.bfloat16):
    # 在此上下文中，矩阵乘法会自动降精度为 bf16 执行，而敏感算子保持 fp32
    x = torch.zeros(4, 8)
```

### 2.4 设备管理（CPU vs. GPU Memory）

PyTorch 默认在 CPU 主存（RAM）上创建张量。为了利用 GPU 的大规模并行算力，必须显式将张量迁移至 GPU 显存（HBM）或直接在目标设备上下文中创建：

```python
x = torch.zeros(32, 32)
assert x.device == torch.device("cpu")

device = "cuda" if torch.cuda.is_available() else "cpu"
# 方式一：将已有张量从 CPU 复制迁移到 GPU HBM
x = x.to(device)

# 方式二：直接在 GPU 上分配内存并初始化（避免先在 CPU 创建再拷贝的开销）
with torch.device(device):
    x = torch.zeros(32, 32)
assert x.device == torch.device(device)
```

## 3. 优雅的张量代数：使用 `einops` 告别维度混乱

### 3.1 传统 PyTorch 维度索引的痛点

在编写 Transformer 多头注意力机制时，传统 PyTorch 代码充斥着难以阅读且极易出错的数字维度索引：

```python
# 传统写法：x 和 y 形状均为 (batch, seq, hidden)
x = torch.ones(2, 2, 3)
y = torch.ones(2, 2, 3)
z = x @ y.transpose(-2, -1)  # 到底哪个是 -2，哪个是 -1？如果中间多插了一个 head 维度怎么办？
```

为了彻底消灭这种心智负担，CS336 强烈推荐使用基于爱因斯坦求和约定（Einstein Summation Notation, 1916）设计的 [`einops` 库](https://einops.rocks/1-einops-basics/)。其核心哲学是：**为每一个维度赋予显式的语义名称（Named Dimensions），通过名字匹配而非位置索引来操控张量**（注：`einops` 只是编译为底层等价原语的语法糖，不会带来额外性能加速或显著损耗，但能极大提升代码可读性与模块化程度）。

### 3.2 `einops` 三大核心原语全解

#### 1. `einsum`：带自动记账的广义矩阵乘法（Generalized Matrix Multiplication）
核心规则：**出现在箭头左侧输入中、但未出现在箭头右侧输出中的命名维度，会被自动乘积并求和消去（Summed out）**；同时出现在两侧的维度则作为批处理或输出索引保留。

```python
from einops import einsum, reduce, rearrange

# 基础矩阵乘法：(3, 4) @ (4, 3) -> (3, 3)
x = torch.ones(3, 4)  # seq1, hidden
y = torch.ones(4, 3)  # hidden, seq2
z = einsum(x, y, "seq1 hidden, hidden seq2 -> seq1 seq2")  # hidden 未出现在右侧，自动求和消去

# 复杂批处理矩阵乘法（无需再调用 .transpose(-2, -1)！）
x = torch.ones(2, 3, 4)  # batch, seq1, hidden
y = torch.ones(2, 3, 4)  # batch, seq2, hidden
# 直接通过命名指明两个输入最后都是 hidden，完全不用脑补转置！
z = einsum(x, y, "batch seq1 hidden, batch seq2 hidden -> batch seq1 seq2")

# 进阶技巧：使用省略号 `...` 自动广播任意数量的批处理前置维度（如 batch, heads 等）
z = einsum(x, y, "... seq1 hidden, ... seq2 hidden -> ... seq1 seq2")
```

#### 2. `reduce`：单张量的命名维度聚合（Sum / Mean / Max / Min）
替代令人困惑的 `x.sum(dim=-1)`：

```python
x = torch.ones(2, 3, 4)  # batch, seq, hidden
# 旧写法：y = x.sum(dim=-1)
# 新写法：未出现在箭头右侧的 `hidden` 维度被按指定算子（"sum", "mean", "max", "min"）聚合
y = reduce(x, "... hidden -> ...", "sum")
```

#### 3. `rearrange`：维度的拆分（Unflatten）、合并（Flatten）与重排
在多头注意力（Multi-Head Attention）中，模型的总隐藏层维度 `total_hidden` 实际上是 `heads * head_dim` 的展平表示。当我们需要针对每个 Head 的子空间独立应用权重变换时，`rearrange` 提供了极其清晰的括号拆分与合并语法：

```python
x = torch.ones(3, 8)  # (seq, total_hidden)，其中 total_hidden = 8 = heads(2) * hidden1(4)
w = torch.ones(4, 4)  # (hidden1, hidden2)

# 第一步：用圆括号 (heads hidden1) 将单维拆分为两个独立维度（按行主序 Row-major 顺序拆分）
x = rearrange(x, "... (heads hidden1) -> ... heads hidden1", heads=2)  # 形状变为 (3, 2, 4)

# 第二步：使用 `...` 忽略前置的 seq 和 heads 维度，仅对 hidden1 进行线性变换
x = einsum(x, w, "... hidden1, hidden1 hidden2 -> ... hidden2")        # 形状仍为 (3, 2, 4)

# 第三步：将 heads 和 hidden2 重新合并回展平维度
x = rearrange(x, "... heads hidden2 -> ... (heads hidden2)")           # 形状恢复为 (3, 8)
```

## 4. 算力记账、矩阵乘法与模型算力利用率（FLOPs & MFU）

### 4.1 严格区分两个发音相同的缩写：`FLOPs` vs. `FLOP/s`

* **`FLOPs`（小写 `s`，复数名词，Floating-Point Operations）**：衡量某个算法或模型训练所执行的**浮点运算总次数（工作量总量）**。例如：
  * 训练 [GPT-3 (2020)](https://lambdalabs.com/blog/demystifying-gpt-3) 消耗了约 **$3.14 \times 10^{23}\text{ FLOPs}$**；
  * 训练 [GPT-4 (2023)](https://www.google.com/search?q=https://www.semianalysis.com/p/gpt-4-architecture-infrastructure) 据推测消耗了约 **$2 \times 10^{25}\text{ FLOPs}$**。
* **`FLOP/s`（或全大写 `FLOPS`，Floating-Point Operations Per Second）**：衡量硬件加速器的**计算速度/吞吐率（功率）**。例如：
  * 单张 NVIDIA H100 的稠密 BF16 峰值算力为 **$989.5\text{ TFLOP/s}$**（即 $1979 \times 10^{12} / 2\text{ FLOP/s}$）。
  * 如果租用一台 8 卡 H100 节点满负荷运行 2 周（14 天），它能提供的理论极限总算力为：
    $$\text{total\_flops} = 8 \times (14 \times 86400\text{ s}) \times (989.5 \times 10^{12}\text{ FLOP/s}) \approx 9.58 \times 10^{21}\text{ FLOPs}$$

### 4.2 矩阵乘法（MatMul）与逐元素算子的 FLOPs 推导

在神经网络中，除了矩阵乘法之外的所有逐元素操作（Element-wise operations，如矩阵加法 $A+B$、激活函数 ReLU、标量缩放）对形状为 $(M, N)$ 的矩阵仅消耗 $O(MN)$ FLOPs。对于足够大的维度，**矩阵乘法（General Matrix Multiplication, GEMM）占据了绝大部分计算量**。

考虑线性层前向传播 $Y = X W$，其中输入数据矩阵 $X \in \mathbb{R}^{B \times D}$（$B$ 为样本/Token 数，$D$ 为输入特征维度），权重矩阵 $W \in \mathbb{R}^{D \times K}$（$K$ 为输出维度），输出矩阵 $Y \in \mathbb{R}^{B \times K}$：
* 为了计算输出矩阵中的每一个元素 $Y_{i, k} = \sum_{j=1}^D X_{i, j} W_{j, k}$，我们需要执行 **$D$ 次乘法** 和 **$D - 1$ 次加法**（合计 $2D - 1 \approx 2D$ FLOPs；在现代硬件指令如 FMA 融合乘加中，1 次乘法和 1 次加法均计为各 1 FLOP）。
* 输出矩阵共有 $B \times K$ 个元素，因此整个矩阵乘法的总浮点运算量为：

$$\text{FLOPs}(X_{B\times D} @ W_{D\times K}) = 2 B D K$$

> **核心物理意义**：注意到 $B$ 是输入 Token（数据点）的数量，而 $D \times K$ 恰好是权重矩阵 $W$ 的**参数量（Number of Parameters）**！因此，**任何线性层前向传播的 FLOPs 恒等于 $2 \times (\text{数据点/Token 数}) \times (\text{参数量})$**。

### 4.3 硬件峰值算力对照表与 GPU 计时基准测试（Benchmarking）

不同代际 GPU 在不同精度下的硬件理论峰值（Dense，不含稀疏翻倍）差异巨大。`lecture_02.py` 给出了三代主力数据中心 GPU 的 `get_promised_flop_per_sec` 精确对照表：

| GPU 型号 | `fp32` 峰值算力 | `fp16` / `bf16` Tensor Core 稠密峰值算力 | HBM 显存带宽 | BF16 加速器算术强度 ($\text{FLOP/Byte}$) |
| :--- | :--- | :--- | :--- | :--- |
| **[NVIDIA A100 (80GB)](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-us-nvidia-1758950-r4-web.pdf)** | $19.5\text{ TFLOP/s}$ | **$312\text{ TFLOP/s}$** | $2.0\text{ TB/s}$ | $\approx 156\text{ FLOPs/Byte}$ |
| **[NVIDIA H100 (SXM5 80GB)](https://www.nvidia.com/en-us/data-center/h100/)** | $67.5\text{ TFLOP/s}$ | **$989.5\text{ TFLOP/s}$** ($1979 / 2$) | $3.35\text{ TB/s}$ | **$\approx 295.4\text{ FLOPs/Byte}$** |
| **[NVIDIA B200 (Blackwell)](https://docs.nvidia.com/dgx/dgxb200-user-guide/introduction-to-dgxb200.html)** | $75.0\text{ TFLOP/s}$ | **$2250\text{ TFLOP/s}$** ($4.5\text{ PFLOP/s} / 2$) | $8.0\text{ TB/s}$ | $\approx 281.3\text{ FLOPs/Byte}$ |

#### 如何在 PyTorch 中正确测量 GPU 执行时间？
由于 CUDA 核函数（Kernel Launch）是**异步非阻塞（Asynchronous）**的——CPU 把指令塞进 GPU 队列后立刻返回下一行 Python 代码，如果直接用 `time.time()` 包裹 `x @ w`，测出来的只是 CPU 下发指令的几微秒开销！**在计时前后必须显式调用 `torch.cuda.synchronize()` 设置同步屏障**：

```python
import timeit

def benchmark(func, num_trials: int = 5) -> float:
    """精确测量函数 `func` 在 GPU 上的平均墙钟执行时间（秒）"""
    # 1. 等待此前队列中残留的所有 CUDA 线程执行完毕
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    def run():
        func()
        # 2. 关键！每次运行后必须等待 GPU 真正算完才能停止计时
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    total_time = timeit.timeit(run, number=num_trials)
    return total_time / num_trials
```

### 4.4 模型算力利用率（MFU, Model FLOPs Utilization）

测出实际墙钟时间 $\text{actual\_time}$ 后，我们可以计算实际每秒完成的浮点运算数，并与硬件规格表承诺的峰值算力对比，从而定义 **MFU**：

$$\text{actual\_flop\_per\_sec} = \frac{\text{actual\_num\_flops}}{\text{actual\_time}}, \qquad \text{MFU} = \frac{\text{actual\_flop\_per\_sec}}{\text{promised\_flop\_per\_sec}}$$

* **经验法则**：对于单个巨大的矩阵乘法，MFU 最高可达 $0.7 \sim 0.8$；对于包含通信与归一化层的完整大模型训练，**$\text{MFU} \ge 0.5$（50% 以上）就已经是非常优秀的工程实现**；如果 $\text{MFU} \le 0.1$，说明代码中存在严重的访存或同步瓶颈。

## 5. 算术强度与 Roofline 性能分析模型（Arithmetic Intensity & Roofline Analysis）

### 5.1 为什么 MFU 达不到 100%？计算与存储的物理分离

为什么算子跑不满 GPU 承诺的 $989.5\text{ TFLOP/s}$？因为数据存储在**高带宽显存（HBM）**中，而计算发生在**流式多处理器计算核心（Accelerator / SMs）**中。任何一次计算都必须经历三步：
1. 将输入张量从 HBM 搬运到计算核心；
2. 在计算核心上执行浮点运算；
3. 将输出张量从计算核心写回 HBM。

这取决于两个独立的硬件速率（以 H100 BF16 为例）：
* **计算速度（Accelerator Speed）**：$\text{FLOP/s}_{\text{peak}} = 989.5 \times 10^{12}\text{ FLOP/s}$
* **显存带宽（Memory Bandwidth）**：$\text{Bytes/s}_{\text{peak}} = 3.35 \times 10^{12}\text{ Bytes/s}$ (3.35 TB/s)

假设硬件能够**完美重叠（Overlap）**数据搬运时间（$\text{communication\_time} = \frac{\text{bytes}}{\text{h100\_bytes\_per\_sec}}$）与计算时间（$\text{computation\_time} = \frac{\text{flops}}{\text{h100\_flop\_per\_sec}}$），则总耗时由两者中的较慢者决定：

$$\text{total\_time} = \max(\text{communication\_time}, \text{computation\_time})$$

* 当 $\text{communication\_time} > \text{computation\_time}$ 时，称为 **访存受限（Memory-bound）**——计算核心大部分时间在空转等待数据从 HBM 传过来；
* 当 $\text{computation\_time} > \text{communication\_time}$ 时，称为 **计算受限（Compute-bound）**——数据供应充足，计算核心满负荷运转。

### 5.2 算术强度（Arithmetic Intensity）与硬件拐点

将不等式 $\frac{\text{flops}}{\text{h100\_flop\_per\_sec}} \gtrless \frac{\text{bytes}}{\text{h100\_bytes\_per\_sec}}$ 移项整理，可以将软件负载特征与硬件特征解耦为两个比值：

1. **加速器强度（Accelerator Intensity / Ridge Point）**：硬件每搬运 1 字节数据，能够完成多少次浮点运算？
   $$I_{\text{H100 (bf16)}} = \frac{\text{h100\_flop\_per\_sec}}{\text{h100\_bytes\_per\_sec}} = \frac{989.5 \times 10^{12}}{3.35 \times 10^{12}} \approx 295.4\text{ FLOPs/Byte}$$
   （记住 **300 FLOPs/Byte** 这个黄金分界线！）
2. **算法算术强度（Arithmetic Intensity）**：特定算子每读写 1 字节 HBM 数据，实际执行了多少次浮点运算？
   $$I_{\text{alg}} = \frac{\text{FLOPs}}{\text{Bytes Transferred}}$$

**瓶颈判定准则**：
* 若 $I_{\text{alg}} < I_{\text{accelerator}}$（在 H100 BF16 上即 $< 295.4$） $\implies$ **Memory-bound（访存受限）**；
* 若 $I_{\text{alg}} > I_{\text{accelerator}}$（在 H100 BF16 上即 $> 295.4$） $\implies$ **Compute-bound（计算受限）**。

### 5.3 五大典型算子的算术强度逐一推导（均假设 `bf16`，每元素 2 Bytes）

课堂通过 `lecture_02.py` 中的五个函数，从小到大推导了常见算子的算术强度：

| 算子名称 | 数学表达式与维度 | HBM 读写总字节数 ($\text{Bytes}$) | 浮点运算次数 ($\text{FLOPs}$) | 算术强度 ($I_{\text{alg}} = \frac{\text{FLOPs}}{\text{Bytes}}$) | 在 H100 ($I_{\text{acc}}\approx 295$) 上的状态 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **1. ReLU** | $y = \max(x, 0), \; x \in \mathbb{R}^n$ | 读 $x (2n)$ + 写 $y (2n) = 4n$ | $n$ 次比较 = $n$ | $\frac{n}{4n} = 0.25$ | **极度 Memory-bound** ($\text{comm } 1.2\mu\text{s} \gg \text{comp } 1\text{ns}$) |
| **2. GeLU** | $y = \text{GeLU}(x), \; x \in \mathbb{R}^n$ | 读 $x (2n)$ + 写 $y (2n) = 4n$ | 含多项式与 $\tanh$ 约 $20n$ | $\frac{20n}{4n} = 5.0$ | **依然 Memory-bound** ($5 \ll 295$) |
| **3. 向量点积 (Dot Product)** | $y = x^\top w, \; x, w \in \mathbb{R}^n$ | 读 $x (2n)$ + 读 $w (2n)$ + 写标量 $y (2) = 4n+2$ | $n$ 乘 $+ (n-1)$ 加 = $2n-1$ | $\frac{2n-1}{4n+2} \approx 0.5$ | **极度 Memory-bound** |
| **4. 矩阵-向量乘法 (MatVec)** | $y = x W, \; x \in \mathbb{R}^n, W \in \mathbb{R}^{n\times n}$ | 读 $x (2n)$ + 读 $W (2n^2)$ + 写 $y (2n) = 2n^2+4n$ | $n$ 次点积 = $n(2n-1)$ | $\frac{2n^2-n}{2n^2+4n} \approx 1.0$ | **极度 Memory-bound**（**自回归推理 Decode 瓶颈根源**！） |
| **5. 矩阵-矩阵乘法 (MatMul)** | $Y = X W, \; X, W \in \mathbb{R}^{n\times n}$ | 读 $X (2n^2)$ + 读 $W (2n^2)$ + 写 $Y (2n^2) = 6n^2$ | $n^2$ 次点积 = $n^2(2n-1)$ | $\frac{2n^3-n^2}{6n^2} \approx \frac{n}{3}$ | 当 $n=1024$ 时 $I_{\text{alg}} \approx 341 > 295$，**终于 Compute-bound！** |

#### 三个颠覆直觉的重要工程推论：
1. **单独执行时，复杂的 GeLU 并不比简单的 ReLU 慢！**
   虽然 GeLU 的计算量（$20n$ FLOPs）是 ReLU（$n$ FLOPs）的 20 倍，但两者的算术强度（$5.0$ vs. $0.25$）都远低于 H100 的拐点 $295.4$。它们的实际耗时完全被读写 $4n$ 字节 HBM 的通信时间卡死，计算核心都在等数据，因此孤立执行时两者耗时几乎完全一样！（这也解释了为什么在 Assignment 2 中我们要做 Kernel Fusion，把激活函数融合进矩阵乘法尾声，省掉那 $4n$ 字节的 HBM 往返读写）。
2. **为什么大模型训练与推理 Prefill 很快，而自回归 Decode 极慢？**
   * **训练与 Prefill** 是拿整批 Token 矩阵 $X_{B\times D}$ 乘以权重矩阵 $W_{D\times D}$，做的是 $O(n^2)$ 搬运换取 $O(n^3)$ 计算的 **MatMul**，算术强度为 $n/3$（只要 Batch Size $B$ 和隐藏维度 $D$ 大于 $3 \times 295 \approx 886$，就能跑满 Tensor Core 进入 Compute-bound）。
   * **推理 Decode 阶段**每次只生成 1 个 Token，相当于拿单个向量 $x_{1\times D}$ 去乘以庞大的权重矩阵 $W_{D\times D}$（**Matrix-Vector Product**）。为了算这 1 个 Token，必须把整个权重矩阵 $2n^2$ 字节从 HBM 完整读一遍，算术强度恒为 $\approx 1.0 \ll 295$，完全受制于显存带宽！
3. **精度转换如何影响算术强度？**
   如果将 `bf16` 切换为 `fp32`，不仅分母每元素字节数翻倍（算术强度减半），硬件的 `fp32` 峰值算力也暴跌至 $67.5\text{ TFLOP/s}$，导致加速器拐点变为 $67.5 / 3.35 \approx 20.1\text{ FLOPs/Byte}$。

### 5.4 Roofline 图（屋顶线模型）与 MFU 的理论上限公式

我们将算术强度 $I_{\text{alg}}$ 作为横轴（X 轴），将硬件实际达到的吞吐率（Realized FLOP/s）作为纵轴（Y 轴），即可画出著名的 **Roofline 模型（见 [Jax Scaling Book Roofline 章节](https://jax-ml.github.io/scaling-book/roofline/)）**：
* 图形由一段斜率等于显存带宽（$\text{Bytes/s}_{\text{peak}}$）的**斜坡（Memory-bound 区域）**和一段高度等于峰值算力（$\text{FLOP/s}_{\text{peak}}$）的**水平屋顶（Compute-bound 区域）**组成。
* 斜坡与水平屋顶的交界折点（Kink）正是加速器强度 $I_{\text{accelerator}} = \frac{\text{FLOP/s}_{\text{peak}}}{\text{Bytes/s}_{\text{peak}}}$。
* 由此，我们可以直接写出理想重叠假设下 **MFU 与算术强度的闭式关系**：

$$\text{MFU} = \min\left(1, \; \frac{\text{Arithmetic Intensity}}{\text{Accelerator Intensity}}\right)$$

## 6. 深度网络训练全流程的算力与显存记账（Training Accounting）

### 6.1 反向传播梯度计算的 FLOPs 严格推导（为什么是前向的 2 倍？）

我们先看一个最简单的线性回归前向与反向求导示例：
$$y = \frac{1}{2}(x^\top w - 5)^2$$

```python
x = torch.tensor([1., 2., 3.])
w = torch.tensor([1., 1., 1.], requires_grad=True)
pred_y = x @ w
loss = 0.5 * (pred_y - 5).pow(2)  # pred_y = 6, loss = 0.5 * (6 - 5)^2 = 0.5
loss.backward()
# d(loss)/dw = (pred_y - 5) * x = 1 * [1, 2, 3] = [1, 2, 3]
assert torch.equal(w.grad, torch.tensor([1., 2., 3.]))
```

现在，为了严格推导多层神经网络反向传播的 FLOPs，我们考察一个两层线性网络：
$$h_1 = x W_1, \qquad h_2 = h_1 W_2, \qquad \text{loss} = \text{mean}(h_2)^2$$
其中 $x \in \mathbb{R}^{B \times D}$，$W_1, W_2 \in \mathbb{R}^{D \times D}$。让我们用 `einops` 聚焦**第二层 $h_2 = h_1 W_2$** 的前向与反向过程：

* **前向传播（Forward Pass）**：
  ```python
  h1 = einsum(x, w1, "batch in, in out -> batch out")
  h2 = einsum(h1, w2, "batch in, in out -> batch out")
  ```
  前向计算 $h_2$ 是一次矩阵乘法，消耗：
  $$\text{num\_forward\_flops} = 2 B D^2$$
* **反向传播（Backward Pass）**：
  假设上游已经传回了损失函数关于 $h_2$ 的梯度 `h2.grad`（即 $\frac{\partial \text{loss}}{\partial h_2} \in \mathbb{R}^{B \times D_{\text{out}}}$）。根据多元微积分链式法则，为了更新当前层权重并将梯度继续向底层传递，反向传播必须计算**两个矩阵乘法**：
  1. **对输入的梯度（Backward Message to Previous Layer，`h1.grad`）**：用于继续向第 1 层反向传播：
     $$\frac{\partial \text{loss}}{\partial h_1} = \frac{\partial \text{loss}}{\partial h_2} W_2^\top$$
     用 `einops` 写完全不需要记哪一项要转置，直接按维度名称对齐消去 `out` 维即可：
     ```python
     h1_grad = einsum(h2.grad, w2, "batch out, in out -> batch in")
     assert torch.allclose(h1.grad, h1_grad)  # 完美匹配 PyTorch 自动求导结果！
     ```
     该步消耗 **$2 B D^2$ FLOPs**。
  2. **对当前层权重的梯度（Parameter Gradient，`w2.grad`）**：用于优化器更新参数 $W_2$（注意此时必须用到前向传播保存下来的激活值 `h1`！）：
     $$\frac{\partial \text{loss}}{\partial W_2} = h_1^\top \frac{\partial \text{loss}}{\partial h_2}$$
     用 `einops` 对齐消去 `batch` 维：
     ```python
     w2_grad = einsum(h2.grad, h1, "batch out, batch in -> in out")
     assert torch.allclose(w2.grad, w2_grad)  # 完美匹配 PyTorch 自动求导结果！
     ```
     该步同样消耗 **$2 B D^2$ FLOPs**。

* **汇总全网所有层（得出黄金公式 $6ND$）**：
  $$\text{num\_backward\_flops} = 2 B D^2 + 2 B D^2 = 4 B D^2 = 2 \times \text{num\_forward\_flops}$$
  推广到任意包含 $N$ 个参数、输入 $B$ 个数据点（或 Token 数 $D_{\text{tokens}}$）的多层感知机或短上下文 Transformer：
  * **前向传播（Forward Pass）**：$2 \times (\text{\# data points}) \times (\text{\# parameters})\text{ FLOPs}$
  * **反向传播（Backward Pass）**：$4 \times (\text{\# data points}) \times (\text{\# parameters})\text{ FLOPs}$
  * **单步训练总计（Total per step）**：**$6 \times (\text{\# data points}) \times (\text{\# parameters})\text{ FLOPs}$**
  * *注*：对于 Transformer，当上下文序列长度 $S$ 较短（$S \ll d_{\text{model}}$）时，线性投影层占主导，$6ND$ 是极佳的近似；当 $S$ 极大时，还必须额外加上二次方注意力项 $O(B S^2 d_{\text{model}} L)$ 的 FLOPs（详见 [Transformer FLOPs 拆解博文](https://kipp.ly/transformer-inference-arithmetic/) 与 Assignment 1）。

### 6.2 深度网络模型与手写 `AdaGrad` 优化器实现

在 `lecture_02.py` 中，Percy 定义了一个包含 $L$ 层 `Block`（每层为无偏置线性变换 $D \times D$ 加 ReLU）的 `DeepNetwork`，总参数量为 $N = D^2 L$：

```python
class Block(nn.Module):
    """单层模块：线性变换 + ReLU 激活"""
    def __init__(self, dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(dim, dim) / math.sqrt(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x @ self.weight  # 线性层 (Pre-activation)
        x = F.relu(x)        # 激活层 (Post-activation)
        return x

class DeepNetwork(nn.Module):
    def __init__(self, dim: int, num_layers: int):
        super().__init__()
        self.layers = nn.ModuleList([Block(dim) for i in range(num_layers)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x
```

为了在不剧透 Assignment 1（手写 `AdamW`）的前提下展示 `torch.optim.Optimizer` 的内部机制，课堂手写了经典的 **[AdaGrad 优化器 (Duchi et al., 2011)](https://jmlr.org/papers/v12/duchi11a.html)**。先厘清现代自适应优化器的演进谱系：
* **Momentum** = SGD + 梯度一阶矩的指数移动平均（Exponential averaging of $\text{grad}$）
* **AdaGrad** = SGD + 历史所有步梯度平方的累加（Averaging by $\sum \text{grad}^2$）
* **RMSProp** = 将 AdaGrad 的直接累加改为梯度二阶矩的指数移动平均（Exponential averaging of $\text{grad}^2$）
* **Adam** = RMSProp（二阶矩指数平均）+ Momentum（一阶矩指数平均）+ 偏差修正

```python
from typing import Iterable

class AdaGrad(torch.optim.Optimizer):
    def __init__(self, params: Iterable[nn.Parameter], lr: float = 0.01):
        super(AdaGrad, self).__init__(params, dict(lr=lr))

    def step(self):
        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                # 获取该参数专属的优化器状态字典
                state = self.state[p]
                grad = p.grad.data

                # 读取历史梯度平方累加和 g2 = sum_{i < t} g_i^2，若首次执行则初始化为全 0 张量
                g2 = state.get("g2", torch.zeros_like(grad))

                # 原地更新优化器状态并存回字典
                g2 += torch.square(grad)
                state["g2"] = g2

                # 自适应步长更新参数：除以 sqrt(g2 + eps)
                p.data -= lr * grad / torch.sqrt(g2 + 1e-5)
```

> **训练循环中的显存释放细节**：在调用 `optimizer.step()` 后，清理梯度应当使用 **`optimizer.zero_grad(set_to_none=True)`**！如果不传 `set_to_none=True`，PyTorch 默认只是用 `0` 填充原有的梯度张量（显存依然被占着）；设为 `None` 则会直接释放梯度张量所占用的显存，直到下一次 `backward()` 时再按需分配。

### 6.3 完整训练显存四大金刚记账（Memory Accounting Breakdown）

以 $L$ 层、`dim` 为 $D$、Batch Size 为 $B$ 的 `DeepNetwork`（总参数量 $N = D^2 L$）在混合精度训练下为例（深入阅读可参考 [Korthikanti et al. / Transformer 训练显存拆解博客](https://blog.eleuther.ai/transformer-math/)）：

1. **参数显存（Parameter Memory）**：每个参数 `bf16` 占 2 Bytes $\implies 2 N = 2 D^2 L\text{ Bytes}$。
2. **梯度显存（Gradient Memory）**：每个参数对应一个 `bf16` 梯度 $\implies 2 N = 2 D^2 L\text{ Bytes}$。
3. **优化器状态显存（Optimizer State Memory）**：必须使用 `fp32`（4 Bytes）以保证多步幂次累加的数值稳定性：
   * 对于 **AdaGrad**：只需存二阶矩 `g2`，占 **$4 N\text{ Bytes}$**（4 Bytes/param）；
   * 对于 **Adam / AdamW**：需同时存一阶动量 $m$ 和二阶动量 $v$，占 **$8 N\text{ Bytes}$**（8 Bytes/param；若框架额外维护 FP32 Master Weights 则再加 $4N$）。
   * *系统洞察*：优化器状态占用了巨量静态显存（限制了单卡能装下的最大模型尺寸），但由于它只在每个 Training Step 结束时的 `optimizer.step()` 中被读写一次，**它对整体训练计算速度（MFU）几乎不构成带宽瓶颈**。
4. **激活值显存（Activation Memory）**：前向传播中每层产生的中间激活都必须保留到反向传播算完 `w.grad` 后才能释放。每层激活形状为 $(B, D)$，`bf16` 占 2 Bytes（若算上线性层后与 ReLU 后的两份则为 $2 \times 2BDL$，基准记账按每层主体激活计）：
   $$\text{activation\_memory} \propto 2 B D L\text{ Bytes}$$

$$\text{total\_memory} = \underbrace{2N}_{\text{Params}} + \underbrace{2N}_{\text{Grads}} + \underbrace{4N \text{ (AdaGrad) or } 8N \text{ (Adam)}}_{\text{Optimizer States}} + \underbrace{2 B D L}_{\text{Activations}}$$

## 7. 显存优化两大核心利器：梯度累积与激活检查点

当我们需要扩大 Batch Size $B$ 以提升训练稳定性（逼近临界批大小 Critical Batch Size），或者加深网络层数 $L$ 时，与 $B \times L$ 成正比的**激活值显存（Activation Memory）**会迅速撑爆 GPU 显存（OOM）。本讲最后介绍了两种在不增加硬件的前提下突破激活显存瓶颈的系统技巧：

### 7.1 梯度累积（Gradient Accumulation）：用串行时间换并行空间

* **核心思想**：如果我们想要实现全局大批量 `batch_size`（例如 $B_{\text{global}} = 1024$），但单卡显存只能装下微批量 `micro_batch_size`（例如 $b = 64$），我们可以将 1 个大步拆分为 $K = B_{\text{global}} / b = 16$ 个微步（Micro-steps）。
* **执行流程**：
  1. 每次只前向和反向传播一个大小为 `micro_batch_size` 的微批次；
  2. 反向传播计算出的梯度直接累加到 `p.grad` 上（**中间绝不调用 `zero_grad()`**），而前向产生的微批次激活值在每次 `backward()` 结束瞬间立即被自动释放！
  3. 每满 $K = \text{batch\_size} / \text{micro\_batch\_size}$ 步，才真正调用一次 `optimizer.step()` 更新权重，随后调用 `optimizer.zero_grad(set_to_none=True)` 清空累积梯度。
* **收益**：在数学上与一次性跑大 Batch 完全等价（注意 Loss 需除以累积步数 $K$），但**峰值激活显存直接下降为原来的 $\frac{\text{micro\_batch\_size}}{\text{batch\_size}}$**！

### 7.2 激活检查点 / 重计算（Activation Checkpointing / Rematerialization）：用算力换显存

* **为什么训练比推理费显存？**
  在推理（Inference）时，我们不需要算梯度，算完第 $l$ 层得到 $h_l$ 后，第 $l-1$ 层的激活 $h_{l-1}$ 可以立刻丢弃，激活显存仅为 $O(1)$（单层大小）。但在训练时，正如第 6.1 节公式 $w_l.\text{grad} = h_{l-1}^\top (h_l.\text{grad})$ 所示，**反向传播到第 $l$ 层时必须用到前向传播时的激活值**！默认情况下，PyTorch 会把所有层的中间张量（例如每层线性变换后的 $g_l = h_{l-1} W_l$ 和 ReLU 后的 $h_l = \text{ReLU}(g_l)$，即 `x, g1, h1, g2, h2, g3, h3, g4, h4`）全部缓存在显存中，导致激活显存高达 $O(L)$。
* **核心思想（Trade-off Memory for Compute）**：
  **激活检查点（Activation Checkpointing，又称 Gradient Checkpointing 或 Rematerialization）** 在前向传播时只保留部分关键节点（Checkpoints，如每个 Block 的输入 `x, h1, h2, h3, h4`）的激活值，丢弃块内所有的中间激活（如 `g1, g2, g3, g4`）；等到反向传播需要用到 `g3` 来计算第 3 层梯度时，**直接从最近的检查点 `h2` 出发，临时重新执行一次局部前向传播把 `g3` 算出来**！
* **PyTorch 代码实现（仅需包装一行 `torch.utils.checkpoint.checkpoint`）**：

```python
import torch.utils.checkpoint

class DeepNetworkCheckpointed(nn.Module):
    """开启逐层激活检查点的深度网络"""
    def __init__(self, dim: int, num_layers: int):
        super().__init__()
        self.layers = nn.ModuleList([Block(dim) for i in range(num_layers)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            # 核心改动：前向传播时不保存 layer 内部的中间激活 (g_l)，反向传播时自动从输入 x 重算
            x = torch.utils.checkpoint.checkpoint(layer, x, use_reentrant=False)
        return x
```

#### 检查点频率的渐近复杂度权衡（How frequently to checkpoint?）
假设网络共有 $L$ 层，我们应该每隔多少层设置一个检查点？

| 检查点策略 | 保留的激活张量示意 ($L=9$) | 激活显存复杂度 | 反向重计算额外算力开销 | 适用场景与评价 |
| :--- | :--- | :--- | :--- | :--- |
| **1. 全部保存（默认无 Checkpoint）** | `\| h1 h2 h3 h4 h5 h6 h7 h8 h9 \|` | **$O(L)$** | **$0$**（无重计算，总算力 $6ND$） | 显存充裕时最快，但深层或长序列下极易 OOM |
| **2. 每个 Block 边界保存（工业界最常用）** | 仅存每个 Transformer Block 入口，丢弃块内 Attention/MLP 中间激活 | $O(L)$（但常数项大幅降低数倍） | 额外增加 **1 次前向传播**（总算力变为 $8ND$，仅多花 $\approx 33\%$ 算力） | 大模型训练标准配置：以增加 1/3 算力为代价，换取数倍的激活显存缩减 |
| **3. 极端情况：一层都不存** | `\| \|`（仅存最初输入 $x$） | **$O(1)$** | **$O(L^2)$**（反传每算一层都要从第 1 层重算一遍） | 计算开销二次方爆炸，不可接受 |
| **4. 理论最优平衡：每隔 $\sqrt{L}$ 层存一个** | `\| h3 h6 h9 \|` | **$O(\sqrt{L})$** | **$O(L)$**（每个区间最多重算 $\sqrt{L}$ 层，总重算仍为一次前向级别） | [Chen et al. (2016)](https://arxiv.org/abs/1604.06174) 提出的次线性显存方案，实现显存与重算的渐近平衡 |

## 8. 课后思考与自测练习（Review & Self-Check）

1. **`einops` 实战改写**：在标准多头自注意力（Multi-Head Self-Attention）中，给定查询张量 `Q` 和键张量 `K`，形状均为 `(batch, seq_len, num_heads, head_dim)`。请用一行 `einops.einsum` 写出计算未缩放注意力得分矩阵 `scores`（形状为 `(batch, num_heads, seq_len_q, seq_len_k)`）的代码。
   * *参考答案*：
     ```python
     scores = einsum(Q, K, "batch seq_q heads dim, batch seq_k heads dim -> batch heads seq_q seq_k")
     ```
     （注意观察：既不需要先把 `heads` 和 `seq` 交换维度 `.transpose(1, 2)`，也不需要把 `K` 的最后两维转置 `.transpose(-2, -1)`，`einops` 自动一步到位！）
2. **算术强度与 Batch Size 阈值推导**：在 NVIDIA B200 GPU 上运行 `bf16` 矩阵乘法 $Y = X W$，其中输入 $X \in \mathbb{R}^{B \times D}$，权重 $W \in \mathbb{R}^{D \times D}$。已知 B200 的稠密 BF16 峰值算力为 $2.25\text{ PFLOP/s}$，HBM 带宽为 $8.0\text{ TB/s}$。假设隐藏层维度 $D$ 远大于批大小 $B$（即 $D \gg B$），请问 Batch Size $B$ 至少要达到多少，才能使该线性层进入 **Compute-bound** 状态？
   * *参考推导*：
     * B200 的加速器强度为 $I_{\text{B200}} = \frac{2.25 \times 10^{15}}{8.0 \times 10^{12}} = 281.25\text{ FLOPs/Byte}$。
     * 矩阵乘法 $X_{B\times D} @ W_{D\times D}$ 的计算量为 $2 B D^2\text{ FLOPs}$；读写字节数为 $\text{读 }X(2BD) + \text{读 }W(2D^2) + \text{写 }Y(2BD) = 2D^2 + 4BD\text{ Bytes}$。
     * 当 $D \gg B$ 时，$2D^2 + 4BD \approx 2D^2$，因此算法的算术强度简化为：
       $$I_{\text{alg}} = \frac{2 B D^2}{2 D^2 + 4 B D} \approx \frac{2 B D^2}{2 D^2} = B\text{ FLOPs/Byte}$$
     * 令 $I_{\text{alg}} \ge I_{\text{B200}}$，即得 **$B \ge 282$**！这揭示了一个极其深刻的系统定律：当隐藏维度很大时，矩阵乘法的算术强度约等于总 Token 批大小 $B$；只有当单次前向并发处理的 Token 总数超过约 282 个时，才能填满 B200 的计算单元！
3. **Transformer 显存与算力精算（衔接 Assignment 1）**：尝试为包含 $L$ 层、隐藏维度 $d_{\text{model}}$、注意力头数 $H$、词表大小 $|V|$、序列长度 $S$、批大小 $B$ 的完整 SwiGLU Transformer 推导精确的参数量 $N$、前向 FLOPs（区分 $O(B S d_{\text{model}}^2)$ 的线性投影项与 $O(B S^2 d_{\text{model}})$ 的注意力项）以及每层激活值显存占用。我们将在下一讲（Lecture 3: Architectures & Hyperparameters）中继续深入。
