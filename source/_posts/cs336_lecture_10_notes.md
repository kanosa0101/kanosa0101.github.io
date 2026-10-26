---
title: Stanford CS336 (Spring 2026) Lecture 10 深度课程笔记：Inference
date: 2026-10-26 08:51:00
tags:
  - 大模型
  - 大模型推理
  - 算力密度
  - KV Cache
  - 模型量化
  - 投机采样
  - CS336
categories:
  - [笔记, CS336]
description: 全面解析 Stanford CS336 Lecture 10，攻克大模型推理核心痛点：算术强度与显存带宽瓶颈、Prefill 与 Decode 阶段显存拆解、KV Cache 压缩算法、模型量化（PTQ/QAT）、投机采样与动态批处理服务架构。
mathjax: true
---

# Stanford CS336 (Spring 2026) Lecture 10 深度课程笔记：Inference

> **课程信息**：[Stanford CS336: Language Modeling from Scratch (Spring 2026)](https://cs336.stanford.edu/)
> **主讲教授**：Percy Liang
> **本讲资源**：[Lecture 10 YouTube 视频](https://www.youtube.com/watch?v=EfM546A79aM) | [交互式执行轨迹 (`lecture_10.py`)](https://cs336.stanford.edu/lectures/?trace=lecture_10) | [源码仓库 (`spring2025-lectures/lecture_10.py`)](https://github.com/stanford-cs336/spring2025-lectures/blob/main/lecture_10.py) | [Google《How to Scale Your Model》Inference 章节](https://jax-ml.github.io/scaling-book/inference/)

## 1. 引言：为什么大模型推理系统（Inference Systems）如此重要？

在上一讲（Lecture 9）中，Tatsu 带我们探讨了预训练的扩展定律（Scaling Laws），并在结尾指出：现代工业界为了压低部署成本，纷纷放弃 Chinchilla 的 20:1 比例，转而通过“蓄意过训练（Overtraining）”打造小巧强悍的模型。本讲（Lecture 10）作为纯代码交互式讲义（`lecture_10.py`），由 Percy Liang 全面拆解大模型生命周期中消耗算力与资金最庞大的环节——**Inference（推理与在线服务系统）**。

### 1.1 推理无处不在：一次性训练成本 vs. 永不停歇的推理账单

给定一个训练好的语言模型和一个输入提示词（Prompt），推理（Inference）的目标是**尽可能快速、准确且低成本地生成响应序列**。它绝不仅仅发生在用户和网页聊天机器人对话的时刻，而是贯穿了现代 AI 研发的三大核心场景：
1. **实际生产应用（Actual Use）**：对话助手（Chatbots）、代码自动补全（Code Completion）、自主智能体（Agents）以及海量离线批处理数据清洗（Batch Data Processing，如用小模型过滤或合成数万亿预训练语料）。
2. **模型评测（Model Evaluation）**：在 MMLU、GSM8K、MATH、SWE-bench 等生成式基准上评估模型时，瓶颈往往不在于打分，而在于自回归生成成千上万条长回答的速度。
3. **强化学习后训练（Reinforcement Learning / Post-Training）**：在训练如 DeepSeek-R1 或 OpenAI o1/o3 等推理模型（对应 Assignment 5 的 GRPO）时，每个训练步都要先让当前策略模型自回归采样生成（Rollout）大量长思维链轨迹，再交由验证器打分更新权重——**在现代 RL 训练中，80%～90% 的时间与算力实际上都消耗在推理生成（Rollout Generation）上**！

* **为什么推理效率是关乎生死的商业命脉？**
  训练（Training）虽然昂贵，但它是**一次性固定成本（One-time Cost）**；而推理（Inference）是**每天每秒都在重复累加的持续可变成本（Repeated Ongoing Cost）**！
  * **震撼的工业界数据对比**：据估算，[OpenAI 每天处理的推理生成量高达约 **8.6 万亿（8.6T）Tokens**](https://www.theinformation.com/)；作为参照，2026 年 4 月刚发布的百万上下文旗舰模型 **DeepSeek-V4** 的整个预训练过程总共也才训练了 **32T Tokens**！这意味着**仅仅不到 4 天的日常推理流量，所消耗的 Token 处理规模就已经追平了一次顶级旗舰基座模型的全量预训练**！

### 1.2 从 Chatbot 到 Agent 时代的工作负载质变

Percy 指出，在过去一年里，推理优化的战略地位又上了一个新台阶，原因在于应用范式从 **Chatbots（对话机器人）** 转向了 **Agents（自主智能体）**：
* **在 Chatbot 时代（人类阅读速度是瓶颈）**：用户输入一句提问，模型吐出一段回复供人类直接阅读。由于人类的阅读速度只有每秒几个到十几个词，一旦单流生成速度超过 $30 \sim 50\text{ tokens/sec}$，继续压低单用户延迟的边际感知就不明显了。
* **在 Agent 时代（`Tokens Generated = Compute Spent`，无人类阅读瓶颈！）**：
  ```
  [Chatbot 模式]: 用户 Query ──► 模型生成回答 (数百 Tokens, 受限于人类阅读速度) ──► 人类阅读
  [Agent   模式]: 用户 Query ──► [内部思考 Reasoning ──► 调用工具 Tool Call ──► 读代码/修 Bug ──► 自我反思 ...] (数万~百万内部 Tokens!) ──► 最终结果
  ```
  在 Agent 工作流中，模型生成的绝大部分 Token 是**内部思维链与环境交互轨迹（Internal Trace）**，根本不需要人类实时盯着读！你解决的问题越宏大，Agent 需要消耗的推理 Token 就越多（生成 Token 的数量直接等价于投入的测试时算力 Test-Time Compute）。此时，推理速度慢 2 倍就意味着任务完成时间翻倍、推理算力成本翻倍，**压榨推理性能再也没有“够快就行”的天花板**！

### 1.3 工业界推理生态与四大开源推理框架对比

目前围绕大模型推理已经形成了庞大的商业与开源生态：一类是服务闭源自有模型的厂商（OpenAI、Anthropic、Google），另一类是专门为开放权重模型提供高速推理云服务的厂商（Together AI、Fireworks、Baseten、DeepInfra，以及采用定制全 SRAM 晶圆级/LPU 芯片的 Groq 与 Cerebras）。而在开源引擎层，以下四大框架统治了不同的应用场景：

| 开源推理框架 | 核心研发机构 | 首创核心技术 / 架构特点 | 最适用工作负载与场景定位 |
| :--- | :--- | :--- | :--- |
| **[vLLM](https://github.com/vllm-project/vllm)** | UC Berkeley | 首创 **PagedAttention**（借鉴操作系统虚拟内存分页管理 KV Cache） | **工业界最流行的通用默认首选**，生态最完善，高吞吐在线服务标杆 |
| **[SGLang](https://github.com/sgl-project/sglang)** | UC Berkeley / LMSYS | 首创 **RadixAttention**（基于基数树自动跨请求复用共享前缀 KV Cache）与结构化输出加速 | **专为多轮 Agent 工作流、树搜索与复杂控制流优化**，近年在工业界与 RL 训练中增速极快 |
| **[TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM)** | NVIDIA 官方 | 深度绑定 NVIDIA GPU 底层 Tensor Core、FP8/FP4 算子与自定义通信内核 | 在特定 NVIDIA 硬件上追求**极致单卡/多卡压榨性能**（但灵活性与跨平台性较窄） |
| **[llama.cpp](https://github.com/ggml-org/llama.cpp)** | Georgi Gerganov 及社区 | 纯 C/C++ 无依赖实现（GGUF 格式），极致优化低比特量化与 CPU/Apple Silicon 推理 | **本地终端设备（MacBook / 消费级显卡 / 边缘端 CPU）** 离线部署的绝对霸主 |

### 1.4 衡量推理“快慢”的三大核心指标与根本矛盾

当我们说“要让推理变快”时，“快”在系统工程中对应三个截然不同、甚至互相冲突的量化指标：

1. **首字延迟（Time-to-First-Token, `TTFT`，单位：ms 或 s）**：
   从用户发出请求，到屏幕上蹦出**第一个生成 Token** 所经历的等待时间（本质上等于处理输入 Prompt 的 **Prefill 预填充耗时** + 网络排队开销）。在交互式应用中，用户盯着空白屏幕干等的挫败感最强，因此 `TTFT` 直接决定了第一印象体验。
2. **生成延迟（Latency / Time-per-Output-Token, `TPOT`，单位：seconds/token 或 ms/token）**：
   从单个用户的视角看，首字出现后，后续每个 Token 依次流式蹦出的时间间隔。交互式对话与实时代码补全要求极低的单 Token 延迟。
3. **系统总吞吐率（Throughput，单位：tokens/second）**：
   从服务商数据中心视角看，每秒钟为**所有并发请求合计生成的总 Token 数量**。在离线批处理（比如处理 1 PB 文档）或高并发 API 服务中，总吞吐率直接决定了每百万 Token 的硬件折旧与电费成本（$/1M Tokens）。

> **全讲第一性原理基石：为什么推理（Inference）比训练（Training）难优化得多？**
> * **在监督训练（Training）或预填充（Prefill）时**：你一次性看到了整条序列的所有 $S$ 个 Token！因此你可以沿着序列维度 $S$ 完全并行化，把激活张量拼成大矩阵送入 Transformer 执行高算术强度的**矩阵-矩阵乘法（MatMul）**，轻松喂饱 GPU。
> * **在自回归生成（Generation / Decode）时**：由于第 $t+1$ 个 Token 的输入依赖于第 $t$ 个 Token 的采样结果，你**被迫沿着生成序列维度一步接一步串行执行（Sequential Generation，$T = 1$）**！无法在单条序列的长度维度上并行，导致计算退化为极度访存受限（Memory-bound）的瘦长张量操作。

## 2. Part I：推理工作负载剖析 —— 算术强度、KV Cache 与吞吐-延迟权衡

为了用数学精确推演推理瓶颈，Percy 引入了 [Google《How to Scale Your Model》](https://jax-ml.github.io/scaling-book/transformers/) 中的张量收缩记号体系，并用 `SymPy` 编写了完整的符号推导引擎。

### 2.1 张量维度约定与 Transformer Block 完整数据流图解

我们在 `lecture_10.py` 中定义如下正实数符号来表示 Transformer 的形状超参数与运行维度：

```python
from sympy import symbols, oo

# 定义代表 Transformer 模型形状与运行维度的正符号变量
B, S, T, D, F, N, K, H, L, V = symbols("B S T D F N K H L V", positive=True)
c = symbols("c", positive=True)                        # 用于辅助求渐近极限 (c -> oo) 的比例常数
memory_bandwidth = symbols("memory_bandwidth", positive=True)
```

| 符号 | 物理含义 | 架构标准恒等关系与说明 |
| :--- | :--- | :--- |
| **`B`** | **Batch Size（并发序列数）** | 当前同时处理的独立请求/序列条数 |
| **`S`** | **Conditioning Sequence Length（上下文历史长度）** | 注意力机制所依赖的已存在/已缓存的上下文 Token 数量 |
| **`T`** | **Target / Query Sequence Length（当前步待生成位置数）** | 当前一次前向传播同时处理的输入 Token 数（**训练/Prefill 时 $T = S$；Decode 生成时 $T = 1$！**） |
| **`D`** | **Model Dimension（模型隐藏层主维度 $d_{\text{model}}$）** | 恒满足 $D = N \cdot H$ |
| **`F`** | **Feed-Forward Dimension（MLP 中间升维维度 $d_{\text{ff}}$）** | 常规约定 $F \approx 4 D$（SwiGLU 下约为 $\frac{8}{3}D \sim 3.5D$） |
| **`N`** | **Number of Query Heads（Query 注意力头总数）** | 在 GQA 下满足 $N = K \cdot G$ |
| **`K`** | **Number of KV Heads / Groups（Key/Value 注意力头数）** | 标准 MHA 下 $K = N$；MQA 下 $K = 1$；GQA 下 $1 < K < N$ |
| **`G`** | **Group Size（每个 KV 头对应的 Query 头数）** | $G = N / K$（*注：课堂上学生敏锐指出了讲义注释笔误，Percy 现场确认 $K$ 为 KV 组数、$G$ 为每组内的 Query 头数*） |
| **`H`** | **Head Dimension（单个注意力头的维度 $d_{\text{head}}$）** | 通常固定为 `64` 或 `128` |
| **`L`, `V`** | **Layers（层数）与 Vocabulary Size（词表大小）** | 全网共堆叠 $L$ 个 Transformer Block，词表含 $V$ 个 Token |

#### 张量乘法的三类维度角色（`review_transformer`）
当我们书写广义张量缩并（类似 `einops.einsum`）时，参与运算的维度可分为三类：
1. **缩并维度（Contracting Dimensions，原图红色）**：同时出现在左右两个操作数中，但在相乘求和后**从结果中消失**。例如矩阵乘 `BTD x DH -> BTH` 中的 **`D`**。
2. **常规自由维度（Regular Dimensions，原图黑色）**：仅出现在其中一个操作数中，并**原样保留在结果中**。例如 `BTD x DH -> BTH` 中的 **`B`, `T`, `H`**。
3. **批处理维度（Batching Dimensions，原图蓝色 —— 理解 Attention 瓶颈的关键！）**：同时出现在左右两个操作数中，但**不发生跨元素交互，而是各自独立配对并保留在结果中**！例如批量向量点积 `BD x BD -> B` 中的 **`B`**。
   * **致命预警**：只要一个算子中的维度 `B` 是蓝色的 Batching 维度（两个操作数都带着 `B`），那么增大批大小 `B` **就绝不可能提高算术强度**！因为你只是把 `B` 个独立的向量点积摆在一起跑，没有任何权重复用！

```
[单层 Transformer Block 张量形状与缩并全景图 (Conditioning 长 S, 当前输入长 T)]
  输入激活 X [B, T, D]
     │
     ├──► [Attention 分支]
     │      ├─► X [B,T,D] @ W_Q [D, N, H] ──► Q [B, T, K, G, H]  (将 N 拆为 K 组 x 每组 G 头)
     │      ├─► X [B,T,D] @ W_K [D, K, H] ──► K_new ──► (拼接历史 KV Cache) ──► K [B, S, K, H]
     │      ├─► X [B,T,D] @ W_V [D, K, H] ──► V_new ──► (拼接历史 KV Cache) ──► V [B, S, K, H]
     │      ├─► Q [B,T,K,G,H] @ K [B,S,K,H] (沿 H 缩并, B 和 K 为蓝色独立批处理维!) ──► Softmax ──► A [B, S, T, K, G]
     │      ├─► A [B,S,T,K,G] @ V [B,S,K,H] (沿 S 缩并, B 和 K 为蓝色独立批处理维!) ──► Z [B, T, N, H]
     │      └─► Z [B,T,N,H] @ W_O [N, H, D] ──► Attn_Out [B, T, D] ──► (+ 残差 X)
     │
     └──► [MLP (SwiGLU) 分支] (注意: 权重 W_up, W_gate, W_down 均不含 B 维，所有 B 个请求完全共享同一份权重!)
            ├─► U = X [B,T,D] @ W_up   [D, F] ──► [B, T, F]
            ├─► G = X [B,T,D] @ W_gate [D, F] ──► GeLU(G) * U [B, T, F]
            └─► Y = (GeLU(G)*U) [B,T,F] @ W_down [F, D] ──► MLP_Out [B, T, D] ──► (+ 残差)
```

### 2.2 矩阵乘法算术强度复习与 $B=1$ 退化极限（`review_of_arithmetic_intensity`）

回顾 Lecture 2，在 `bf16`（每元素 2 Bytes）下将激活矩阵 $X \in \mathbb{R}^{B \times D}$ 乘以权重矩阵 $W \in \mathbb{R}^{D \times F}$ 得到 $Y \in \mathbb{R}^{B \times F}$：
* **总浮点计算量（FLOPs）**：$\text{flops} = 2 B D F$（三次项）；
* **总 HBM 显存读写字节数（Bytes Transferred）**：读 $X (2BD)$ + 读 $W (2DF)$ + 写 $Y (2BF) = 2BD + 2DF + 2BF$（二次项）。

```python
# 源自 lecture_10.py (Lines 123-155): 用 SymPy 推导非方阵 MatMul 的渐近算术强度
flops = 2 * B * D * F
bytes_transferred = 2 * B * D + 2 * D * F + 2 * B * F
intensity = (flops / bytes_transferred).simplify()

# 假设批大小 B 远小于模型隐藏维度 D 和 F (令 D = c*B, F = c*B，取极限 c -> oo)：
intensity_asymptotic = intensity.subs(D, c * B).subs(F, c * B).limit(c, oo).simplify()
assert intensity_asymptotic == B  # 算术强度直接等于批大小 B！

# 对比 NVIDIA H100 (BF16 Dense) 的硬件加速器强度 (Ridge Point):
flops_per_second = 989e12
memory_bandwidth = 3.35e12
accelerator_intensity = flops_per_second / memory_bandwidth
assert round(accelerator_intensity) == 295  # 充要条件：当且仅当 B > 295 时进入 Compute-bound！
```

* **极端退化情况（$B = 1$，单请求单步推理）**：
  当只有一个请求每次生成 1 个 Token 时，$B = 1$，矩阵乘法退化为**矩阵-向量乘法（Matrix-Vector Product）**。此时你大费周章地从 HBM 读出了 $2DF$ 字节的庞大权重矩阵，却只做了 $2DF$ 次浮点运算，**算术强度恒等于 $\mathbf{1 \ll 295}$**！H100 接近 99.7% 的计算潜力都在空转等待显存读取！

### 2.3 为什么必须引入 KV Cache？从 $O(T^3)$ 朴素推理到两阶段解耦

如果把训练好的 Transformer 当成一个黑盒：给定提示词 `"never gonna give"`，最朴素的自回归生成（Naive Inference）是：
1. 喂入 `"never gonna give"`（长 3） $\rightarrow$ 跑一遍完整前向传播 $\rightarrow$ 采样出 `"you"`；
2. 拼接得到 `"never gonna give you"`（长 4） $\rightarrow$ **把这 4 个词从头再跑一遍完整前向传播** $\rightarrow$ 采样出 `"up"`；
3. 如此循环生成 $T$ 个 Token：由于单次前向传播复杂度为 $O(t^2)$，**生成 $T$ 个 Token 的总计算复杂度竟然高达 $\sum_{t=1}^T O(t^2) = \mathbf{O(T^3)}$**！

* **破局洞察：因果掩码（Causal Masking）下的前缀不变性**
  因为解码器采用因果掩码（前方的 Token 看不到后方新追加的 Token），当我们在序列末尾追加新词 `"you"` 时，前方旧词 `"never"`, `"gonna"`, `"give"` 在每一层算出的激活值、Key 向量和 Value 向量**与上一步相比连一个比特都不会变**！
* **解决方案：在 HBM 显存中维护 KV Cache（键值缓存）**
  我们将所有已处理位置在每一层（$L$）、每个 KV 头（$K$）算出的 $H$ 维 Key 向量和 Value 向量全部保存在显存中（总形状为 $B \times S \times L \times K \times H \times 2$），从而将推理拆分为物理特性截然不同的**两个阶段**：

```
[阶段 1: Prefill (预填充阶段 —— 并行编码整个 Prompt, T = S)]
  输入 Prompt: ["never", "gonna", "give"] (一次性并行输入 T = S 个 Token)
       │
       ▼ (穿过 Transformer, 算子形态为大矩阵乘 MatMul, 高算术强度 Compute-Bound!)
  1. 初始化填满 KV Cache: 存入 ["never", "gonna", "give"] 在全部 L 层的 K 和 V 向量
  2. 输出末尾位置 Logits ──► 采样出第 1 个新词 "you" (耗时即为首字延迟 TTFT!)

[阶段 2: Generation / Decode (自回归生成阶段 —— 每次仅输入 1 个新词, T = 1)]
  第 1 步: 仅喂入单个新词 ["you"] (T = 1) + 读取 HBM 中的历史 KV Cache (长 S)
           ──► 仅算 "you" 的新 q_t, k_t, v_t ──► 将 (k_t, v_t) 追加写入 KV Cache (长度变为 S+1)
           ──► 用单个 q_t 与全部 S+1 个历史 K, V 做注意力 ──► 采样出下一个词 "up"
  第 2 步: 仅喂入单个新词 ["up"] (T = 1) + 读取更新后的 KV Cache (长 S+1) ──► ... (串行循环, 极度 Memory-Bound!)
```

### 2.4 核心数学推导：MLP 层与 Attention 层的推理算术强度（`arithmetic_intensity_of_inference`）

现在我们在引入 KV Cache 的前提下，用 `lecture_10.py`（Lines 178–261）严格推导单层 **MLP** 与单层 **Attention** 的算术强度。设当前历史上下文（KV Cache）长度为 $S$，当前步一次喂入的新 Token 数为 $T$（之后分别代入 **Prefill: $T = S$** 与 **Generation: $T = 1$**）。

#### 1. MLP 层（SwiGLU 三矩阵）的算术强度推导
在单层 SwiGLU MLP 中，输入激活 $X$ 形状为 $(B, T, D)$，三个权重矩阵分别为 $W_{\text{up}} \in \mathbb{R}^{D \times F}$、$W_{\text{gate}} \in \mathbb{R}^{D \times F}$、$W_{\text{down}} \in \mathbb{R}^{F \times D}$：
* **HBM 读写字节数**：
  1. 读输入 $X$：$2 B T D\text{ Bytes}$；
  2. 读三个权重矩阵 $W_{\text{up}}, W_{\text{gate}}, W_{\text{down}}$：$3 \times (2 D F) = 6 D F\text{ Bytes}$；
  3. 写出中间结果 $U = X W_{\text{up}}$ 与 $G = X W_{\text{gate}}$：$2 B T F + 2 B T F = 4 B T F\text{ Bytes}$；
  4. 写出最终结果 $Y = (\text{GeLU}(G) \odot U) W_{\text{down}}$：$2 B T D\text{ Bytes}$。
  $$\text{Bytes}_{\text{MLP}} = 4 B T D + 4 B T F + 6 D F$$
* **总浮点计算量（FLOPs）**：三次矩阵乘法合计：
  $$\text{FLOPs}_{\text{MLP}} = 2 B T D F + 2 B T D F + 2 B T D F = 6 B T D F$$
* **取渐近极限（当总 Token 数 $B \cdot T \ll D, F$ 时，权重读取项 $6DF$ 占绝对主导）**：
  $$I_{\text{MLP}} = \frac{6 B T D F}{4 B T D + 4 B T F + 6 D F} \xrightarrow{D, F \gg B T} \frac{6 B T D F}{6 D F} = \mathbf{B \cdot T}$$
  * **Prefill 阶段（$T = S$）**：$I_{\text{MLP}}^{\text{prefill}} = \mathbf{B \cdot S}$。只要单条 Prompt 长度 $S > 295$（哪怕 $B=1$），或者批次总 Token 数 $BS > 295$，就能轻松进入 **Compute-bound**！
  * **Generation 阶段（$T = 1$）**：$I_{\text{MLP}}^{\text{gen}} = \mathbf{B}$。因为每次只生成 1 个词（$T=1$），算术强度完全等于**并发请求数 $B$**！如果只有 1 个用户在线（$B=1$），算术强度为 1（极度受限）；但如果能把 **300 个并发用户的请求拼成一个大 Batch（$B \ge 295$）** 一起过 MLP，MLP 层依然能满血跑进 Compute-bound！

#### 2. Attention 层（FlashAttention 核心乘加）的算术强度推导
现在看致命的 Attention 核心步（已通过投影得到当前步查询 $Q \in \mathbb{R}^{B \times T \times D}$，并从 HBM 缓存中读取长度为 $S$ 的历史键值 $K, V \in \mathbb{R}^{B \times S \times D}$，为简化记先按 MHA $K \cdot H = D$ 推导）：
* **HBM 读写字节数（使用 FlashAttention，不写回中间注意力矩阵 $A$）**：
  1. 从 HBM 读取 $Q (2BTD)$、读取历史缓存 $K (2BSD)$、读取历史缓存 $V (2BSD)$；
  2. 将注意力输出 $Y \in \mathbb{R}^{B \times T \times D}$ 写回 HBM $(2BTD)$。
  $$\text{Bytes}_{\text{Attn}} = 4 B S D + 4 B T D = 4 B D (S + T)$$
* **总浮点计算量（FLOPs）**：
  计算 $A = Q K^\top$ 消耗 $2 B S T D$ FLOPs；计算 $Y = \text{Softmax}(A) V$ 消耗 $2 B S T D$ FLOPs：
  $$\text{FLOPs}_{\text{Attn}} = 4 B S T D$$
* **精确算术强度（分子分母中的 $4 B D$ 被完全约掉了！）**：
  $$I_{\text{Attn}} = \frac{4 B S T D}{4 B D (S + T)} = \mathbf{\frac{S \cdot T}{S + T}}$$
  * **Prefill 阶段（$T = S$）**：
    $$I_{\text{Attn}}^{\text{prefill}} = \frac{S \cdot S}{S + S} = \mathbf{\frac{S}{2}}$$
    非常棒！只要 Prompt 长度 $S \ge 590$，Attention 在 Prefill 阶段也是 **Compute-bound**！
  * **Generation 阶段（$T = 1$）**：
    $$I_{\text{Attn}}^{\text{gen}} = \frac{S \cdot 1}{S + 1} = \mathbf{\frac{S}{S + 1} < 1}$$
    **晴天霹雳！** 在自回归生成阶段（$T=1$），无论你的序列长度 $S$ 有多长，**更加致命的是无论你把并发批大小 $B$ 调到多大（因为公式里根本没有 $B$！）**，标准 Attention 的算术强度**永远严格小于 $1$**！

#### 3. 深度思辨：为什么增大 Batch Size $B$ 能拯救 MLP，却完全救不了 Attention？
这是理解现代大模型推理架构演进最核心的一道分水岭：
* **在 MLP 层中**：所有 $B$ 条不同的用户请求，撞击的都是**同一份共享的模型权重矩阵**（$W_{\text{up}}, W_{\text{gate}}, W_{\text{down}}$ 的形状是 $D \times F$，**不依赖于 $B$**）！因此，当你把 $B$ 条请求打包在一起时，权重只需要从慢速 HBM 读进片上 SRAM **1 次**，就能被全部 $B$ 条序列共同复用，从而将算术强度放大 $B$ 倍！
* **在 Attention 层中**：每一条用户请求都有它**自己独立的历史上下文与独立的 KV Cache**（$K$ 和 $V$ 的形状是 $B \times S \times D$，**强依赖于蓝色批处理维度 $B$**）！请求 1 的 Query 只能去点乘请求 1 自己的 KV Cache，绝不能去点乘请求 2 的 KV Cache。因此，把 $B$ 条请求打包在一起，你需要从 HBM 读取的 KV Cache 字节数也同步放大了 $B$ 倍（分子分母同乘 $B$ 直接抵消），**Batching 对提升独立上下文 Attention 的算术强度毫无帮助**！（*唯一例外是后文将讲到的：多个请求共享相同的 System Prompt 前缀*）。

| 算子模块 | 通用算术强度公式 | Prefill 阶段 ($T = S$) | Generation 阶段 ($T = 1$) | 核心工程瓶颈定性结论 |
| :--- | :--- | :--- | :--- | :--- |
| **MLP 层** | $\approx B \cdot T$ (当 $BT \ll D$) | **$B \cdot S$** (极易 Compute-bound) | **$B$** (需靠高并发大 Batch 摊薄权重读取) | 受限于**模型参数权重大小**，靠大并发 Batch $B$ 可转入计算受限 |
| **Attention 层** | $\frac{S \cdot T}{S + T}$ | **$\frac{S}{2}$** (长序列下 Compute-bound) | **$\frac{S}{S + 1} < 1$** (**与 $B$ 无关！**) | 受限于 **KV Cache 大小**，在 $T=1$ 下永远是死忠的 **Memory-bound**！ |

### 2.5 吞吐量与延迟定量建模：Llama 2 13B 在 H100 上的 SymPy 实算（`throughput_and_latency`）

既然确立了 Generation 阶段是彻头彻尾的 **Memory-bound（访存受限）**，在理想计算-访存重叠假设下，**生成 1 个 Token 的耗时（Latency）就严格等于把全部模型参数和全部 $B$ 条序列的 KV Cache 从 HBM 读一遍所需的时间**！

让我们跟随 `lecture_10.py`（Lines 263–369），用代码精确算出 **Llama 2 13B** 在单张 **NVIDIA H100（80GB HBM，带宽 $3.35\text{ TB/s}$）** 上的显存、延迟与吞吐率方程：

```python
@dataclass(frozen=True)
class TransformerPerformanceStats:
    num_params: int     # 模型总参数量 (个)
    memory: int         # 推理总显存占用 = 参数显存 + KV Cache 显存 (Bytes)
    latency: float      # 单步生成延迟 (seconds / token)
    throughput: float   # 系统总吞吐率 (tokens / second)

def compute_transformer_performance_stats(config) -> TransformerPerformanceStats:
    # 1. 模型总参数量: 词嵌入与输出头 (2VD) + L层 SwiGLU MLP (3DFL) + L层 GQA QKV与O投影 ((2DNH + 2DKH)L)
    num_params = 2 * V * D + D * F * 3 * L + (2 * D * N * H + 2 * D * K * H) * L

    # 2. 模型参数占用的字节数 (bf16 占 2 Bytes/param)
    parameter_size = 2 * num_params

    # 3. 单条序列的 KV Cache 字节数 (长 S, K 个 KV 头, 头维 H, L 层, Key+Value 2份, bf16 2 Bytes)
    kv_cache_size_per_seq = S * (K * H) * L * 2 * 2

    # 4. 推理总显存占用 (也是每生成一个新 Token 必须从 HBM 完整读取一次的总字节数!)
    memory = B * kv_cache_size_per_seq + parameter_size

    # 5. 单步延迟 Latency (seconds/token): 受限于 HBM 显存带宽 IO
    latency = memory / memory_bandwidth

    # 6. 系统总吞吐率 Throughput (tokens/second): 耗时 latency 并行生成了 B 个 Token
    throughput = B / latency
    ...

def llama2_13b_config(args={}):
    return {
        S: 1024, D: 5120, F: 13824, N: 40, K: 40, H: 128, L: 40, V: 32000,
        memory_bandwidth: 3.35e12, **args
    }
```

#### 代入 Llama 2 13B 配置后的符号方程解析
将 `llama2_13b_config()` 代入上述函数，我们得到关于并发批大小 $B$ 的精确闭式解：
1. **总参数量**：$\text{num\\_params} = \mathbf{13,015,449,600}$ ($\approx 13.02\text{B}$)，参数显存 $\text{parameter\\_size} = \mathbf{26.03\text{ GB}}$。
2. **单条序列 KV Cache 大小（$S = 1024$）**：
   $$\text{kv\\_cache\\_per\\_seq} = 1024 \times (40 \times 128) \times 40 \times 2 \times 2 = \mathbf{838,860,800\text{ Bytes}} \approx \mathbf{0.839\text{ GB / seq}}$$
3. **总显存占用与单步读取量**：$\text{Memory}(B) = \mathbf{0.839 B + 26.03\text{ (GB)}}$。
4. **单步生成延迟（随 $B$ 线性恶化！）**：
   $$\text{Latency}(B) = \frac{0.839\text{ GB} \cdot B + 26.03\text{ GB}}{3350\text{ GB/s}} = \mathbf{0.2504 B + 7.7704\text{ (ms/token)}}$$
5. **系统总吞吐率（随 $B$ 呈饱和双曲线增长！）**：
   $$\text{Throughput}(B) = \frac{B}{\text{Latency}(B)} = \mathbf{\frac{B}{0.0002504 B + 0.0077704}\text{ (tokens/sec)}}$$
   * **注意吞吐率的渐近天花板**：即使 GPU 显存无限大（令 $B \to \infty$），吞吐率也绝不可能趋于无穷大，而是被 KV Cache 的读取带宽死死封顶在 $\lim_{B\to\infty} \text{Throughput}(B) = \frac{1}{0.0002504} \approx \mathbf{3,993.5\text{ tokens/sec}}$！

| 实验配置 (Llama 2 13B @ 单卡 H100 80GB, $S=1024$) | 并发批大小 $B$ | 总显存占用 (`memory`) | 单 Token 延迟 (`latency`) | 系统总吞吐率 (`throughput`) | 系统状态与工程权衡解读 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **1. 单用户交互 (`b1`, MHA $K=40$)** | $B = 1$ | **`26.87 GB`** | **`8.02 ms/tok`** | **`124.7 tok/s`** | **延迟极佳**（每秒单流 125 词），但 97% 带宽花在反复读 26GB 权重上，吞吐极低 |
| **2. 中等批处理 (`b64`, MHA $K=40$)** | $B = 64$ | **`79.72 GB`** | **`23.80 ms/tok`** | **`2,689.5 tok/s`** | 吞吐暴涨 **$21.6\times$**，但延迟慢了 $3\times$，且 `79.72 GB` 已逼近 80GB 显存悬崖！ |
| **3. 大批处理 (`b256`, MHA $K=40$)** | $B = 256$ | **`240.78 GB`** ❌ | `71.87 ms/tok` | `3,561.8 tok/s` | **直接爆显存（OOM > 80GB）**！且因为逼近双曲线渐近线，吞吐边际收益已显著衰减 |

* **Percy 的“等公交车”比喻与并行策略选择**：
  * **Batch Size $B$ 是延迟与吞吐的拉锯战支点**：小 Batch 就像打出租车（不用等别人，个人延迟极低，但道路总运力吞吐极差）；大 Batch 就像坐大巴车（要把所有人的 KV Cache 都搬一遍，每个人等待延迟变长，但把读取 26GB 模型权重的固定成本分摊给了 64 个人，系统总吞吐率大幅跃升）。
  * **针对 TTFT vs. Generation 的动态策略**：因为 `TTFT` 取决于 Prefill（本身已接近 Compute-bound），Prefill 阶段应采用**较小的 Batch Size** 以保证极速首字响应；而在 Generation 阶段则应采用**尽可能大的 Batch Size** 以拉高总吞吐率！
  * **多卡推理并行**：如果单卡放得下模型，最简单的扩展是启动 $M$ 个独立的模型副本（Replica Parallelism：延迟不变，总吞吐直接乘以 $M$）；如果单卡放不下模型或想降低单步延迟，则需使用张量并行（TP）同时切分模型权重与 KV Cache。

## 3. Part II：有损捷径（Lossy Shortcuts）—— 压缩 KV Cache、量化与模型剪枝蒸馏

既然推理受制于显存容量与显存带宽（$\text{Memory} = B \cdot \text{KV\\_Cache} + \text{Params}$），最直接的提速思路就是**在不显著牺牲模型准确率的前提下，把 KV Cache 或模型权重砍小**！这类会轻微改变模型数学输出的方法被称为**“有损捷径（Lossy Shortcuts）”**。

### 3.1 削减 KV Cache 体积的四大架构利器（`reduce_kv_cache_size`）

#### 1. 分组查询注意力（GQA, Grouped-Query Attention，[Ainslie et al., 2023](https://arxiv.org/abs/2305.13245)）
我们在 Lecture 3 已介绍过 GQA 的原理：保留 $N$ 个 Query 头不变，但将 Key 和 Value 头数从 $N$ 缩减为 $K$（每 $G = N / K$ 个 Query 头共享一组 KV 头），从而**直接将 KV Cache 体积缩减为原来的 $\frac{K}{N}$**！
让我们在 `lecture_10.py`（Lines 390–402）中验证将 Llama 2 13B 的 MHA（$K=40$）改为 **$1:5$ 的 GQA（$K=8$）** 后发生的奇迹：

| 模型配置 ($S=1024$, 单卡 H100 80GB) | KV 头数 $K$ | 批大小 $B$ | 总显存占用 (`memory`) | 单 Token 延迟 (`latency`) | 系统总吞吐率 (`throughput`) | 核心改善效果 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **原始 MHA @ $B=64$ (`k40_b64`)** | $K = 40$ | $64$ | `79.72 GB` (濒临 OOM) | `23.80 ms/tok` | `2,689.5 tok/s` | KV Cache 占了 `53.69 GB`，成为绝对瓶颈 |
| **GQA (1:5) @ $B=64$ (`k8_b64`)** | **$K = 8$** | $64$ | **`33.41 GB`** ✅ | **`9.97 ms/tok`** | **`6,416.7 tok/s`** | **延迟降低 $2.4\times$，吞吐提升 $2.4\times$**，显存省下一大半！ |
| **GQA (1:5) @ $B=256$ (`k8_b256`)** | **$K = 8$** | **$256$** | **`65.63 GB`** ✅ | **`19.59 ms/tok`** | **`13,068.2 tok/s`** | **原本 OOM 的 256 大 Batch 轻松装入单卡，吞吐率相比原始 MHA 飙升近 $5\times$！** |

> **关键洞察**：虽然调节 Batch Size $B$ 会让延迟和吞吐反向拉锯，但**通过 GQA/MLA 缩小 KV Cache 体积，是在固定 $B$ 下让延迟（Latency）和吞吐（Throughput）同时双赢的纯正帕累托改进**！更重要的是，省下的显存允许我们将最大 Batch Size 从 64 推高到 256，从而解锁了 `13,068 tok/s` 的超高吞吐。

#### 2. 多头潜在注意力（MLA, Multi-Head Latent Attention，[DeepSeek-V2, 2024](https://arxiv.org/abs/2405.04434)）
[DeepSeek-V2](https://arxiv.org/abs/2405.04434) 的消融实验（论文 Table 8 & Table 9）指出，在大规模下 GQA 相比完整 MHA 依然存在不可忽视的精度折损。为此，MLA 不再简单粗暴地砍掉 KV 头数，而是对所有头的 KV 联合进行**低秩潜在向量压缩**：
* 标准 MHA 需要缓存 $N \cdot H = 128 \times 128 = 16,384$ 维的键值向量；
* **MLA** 将隐状态投影并缓存为仅 **$C = 512$ 维**的压缩潜在向量 $c_t^{KV} = W_c h_t$，外加为了兼容旋转位置编码而单独解耦出的 **$64$ 维共享 RoPE 键向量**，合计每 Token 每层只需缓存 **$512 + 64 = 576$ 维**（相比 MHA 削减了 **93.3% 的 KV Cache**，最大生成吞吐率提升 **5.76 倍**，且实测下游准确率甚至略优于完整 MHA！）。

> **课堂问答：MLA 把 KV 投影压缩到更低维度，这和直接把整个模型的隐藏层维度 $D$（`d_model`）调小相比有什么本质区别？**
> * **Percy 的解答**：如果无差别地把整个模型的维度 $D$ 砍小，模型的 MLP 容量、残差流带宽和表达能力会全面退化；而架构设计的精髓正是**“精准寻找模型中信息冗余度最高、最能被挤压（Squeeze）而又不伤及主干表达力的部位”**——KV Cache 恰好具有极高的低秩冗余性，定向压缩它能以最小的精度代价换取最大的显存带宽红利。

#### 3. 跨层注意力（CLA, Cross-Layer Attention，[Brandon et al., 2024](https://arxiv.org/abs/2405.12981)）
如果说 GQA 是在**同一层的不同注意力头之间（Across Heads）**共享 KV，那么 MIT 团队提出的 **Cross-Layer Attention (CLA)** 则是顺理成章的下一步——**在相邻的不同 Transformer 层之间（Across Layers）共享 KV Cache**！
```
[标准 MHA / GQA: 每层独立计算并存储自己的 K_l, V_l]
  Layer l:   h_l   ──► 计算 Q_l, K_l, V_l (存入第 l 层 KV Cache)
  Layer l+1: h_{l+1} ──► 计算 Q_{l+1}, K_{l+1}, V_{l+1} (存入第 l+1 层 KV Cache)

[Cross-Layer Attention (CLA-2): 每两层共享同一份 KV Cache!]
  Layer l:   h_l   ──► 计算 Q_l, K_l, V_l (存入共享 KV Cache) ──┐
  Layer l+1: h_{l+1} ──► 仅计算 Q_{l+1} ! 直接复用上一层的 K_l, V_l ◄─┘ (KV Cache 体积再砍一半!)
```
将 GQA 与 CLA 结合（每隔 2 层计算一次 KV），可以在几乎不损失精度的情况下将 KV Cache 体积在 GQA/MQA 的基础上**再缩减 2 倍**。

#### 4. 局部滑动窗口注意力（SWA）、线性注意力与 DeepSeek-V4 百万上下文稀疏压缩注意力
前面的 GQA、MLA、CLA 都是沿着“头或特征维度（$K \cdot H$）”做常数倍压缩，但它们的 KV Cache 依然随着序列长度 $S$ 呈 $O(S)$ 线性增长。要想让 KV Cache 彻底摆脱对超长序列 $S$ 的依赖，有以下三条路径：
1. **局部滑动窗口注意力（Local / Sliding Window Attention, SWA，源自 [Sparse Transformer, Child et al., 2019](https://arxiv.org/abs/1904.10509)、[Longformer, Beltagy et al., 2020](https://arxiv.org/abs/2004.05150)、[Mistral 7B, Jiang et al., 2023](https://arxiv.org/abs/2310.06825)）**：
   每个 Token 只关注最近窗口大小 $W$（如 $W=4096$）内的历史 Token。这样单层 KV Cache 被死死封顶在 $W$ 个 Token（**与总序列长度 $S$ 无关，$O(1)$ 显存**），且通过 $L$ 层堆叠，顶层感受野依然能按 $L \times W$ 线性扩展。为了防止长距离精确检索能力下降，现代模型普遍采用 **Local SWA 与 Global Full Attention 交替堆叠（Hybrid Interleaving）** 的策略。
2. **状态空间与线性注意力（Mamba-2 / Gated DeltaNet，见 Lecture 4）**：用固定大小的状态矩阵 $S_t \in \mathbb{R}^{d_k \times d_v}$ 递归压缩全部历史。
3. **2026 前沿案例：DeepSeek-V4 的三级稀疏与重度压缩注意力（支持 1M 百万上下文）**：
   Percy 展示了 2026 年 4 月最新发布的 **DeepSeek-V4** 架构图，它集成了三种互补的注意力压缩机制：
   * **CSA（Compressed Sparse Attention）**：将每连续 $m$ 个历史 Token 的 KV 聚合成 1 个压缩代表块；
   * **DSA（DeepSeek Sparse Attention）**：用极轻量的 Lightning Indexer 快速打分，动态挑出得分最高的 Top-$k$ 个关键历史 Token 保留全分辨率；
   * **HCA（Heavily Compressed Attention）**：对更久远的历史上下文施加更高倍率的极限压缩。

> **课堂深度探讨：在混合架构（Hybrid Architectures，保留少量全局 Full Attention 层）中，其余非全局层究竟该选“滑动窗口（SWA）”还是“线性注意力（Mamba / Gated DeltaNet）”？**
> * **Percy 的剖析**：这两类机制捕捉的是不同维度的历史信息——**滑动窗口（SWA）** 对窗口内的近期 Token 保留了 **100% 的全分辨率精确细节**，但对窗口外的远古信息是“一刀切”的彻底遗忘（除非靠层间传递）；而 **Mamba / Gated DeltaNet** 则把从头到尾的所有远古信息做了一份**有损的全局摘要（Broad Summary）**。如果单做大海捞针（Needle-in-a-Haystack），纯线性状态压缩必然丢失细节，这也是为什么必须穿插少量全局 Full Attention 的原因。而在已经有全局 Full Attention 兜底的前提下，Gated DeltaNet / Mamba 在理论上比死板的滑动窗口更具表达潜力（因为它不仅能选择性聚焦近期状态以模拟滑动窗口，还能跨长距离保留宏观状态记忆）。

### 3.2 数值精度量化（Quantization：从 `bf16` 到 `int8`/`int4`/`int3`）

既然每步 Decode 都要把庞大的模型参数从 HBM 读一遍，如果能把每个参数从 16 bits（`bf16`，2 Bytes）压缩到 8 bits（1 Byte）甚至 4 bits（0.5 Byte），显存占用和读取耗时就能立竿见影地砍半或降为 1/4！

#### 1. 线性非对称量化的基本数学机制（`quantization`）
如 `lecture_10.py`（Lines 454–460）所示，将连续浮点数 $x$ 映射到低比特整数网格 $x_{\text{quant}}$ 只涉及一个缩放因子 `scale` 和一个零点偏移 `zero_point`：

```python
# 源自 lecture_10.py: 线性量化 (Quantize) 与反量化 (Dequantize) 极简示例
x = 5.2342
scale = 0.1
zero_point = 4

x_quant = round(x / scale) + zero_point       # 量化存储: round(52.342) + 4 = 56 (存为低比特整数 INT8/INT4)
x_approx = (x_quant - zero_point) * scale     # 计算时反量化还原: (56 - 4) * 0.1 = 5.2 (产生 0.0342 的量化舍入误差)
```

* **推理常用精度谱系**：
  * **`fp32`（4 Bytes）**：仅训练时的优化器状态与主权重需要，推理彻底不用；
  * **`bf16`（2 Bytes）**：现代大模型推理的**无损默认基线**；
  * **`fp8`（1 Byte，如 H100 上的 `E4M3` 范围 `[-240, 240]` 或 `[-448, 448]`）**：既可用于激进的 FP8 训练（如 [FP8-LM, Peng et al., 2023](https://arxiv.org/abs/2310.18313) 及 DeepSeek-V3），也可用于高速推理；
  * **`int8`（1 Byte，`[-128, 127]`）与 `int4`（0.5 Byte，`[-8, 7]`）**：专为推力量身定制的均匀整数格点（如 [van Baalen et al., 2023](https://arxiv.org/abs/2303.17951) 所示，整数乘加单元在专用边缘硬件上的面积与功耗效率比浮点单元高 50%～180%，但位数越低精度挑战越大）。

#### 2. 量化感知训练（QAT）vs. 训练后量化（PTQ：GPTQ 与 AWQ）
如何防止量化到 4-bit 或 3-bit 时模型变傻？
1. **量化感知训练（Quantization-Aware Training, QAT）**：在预训练或微调的前向传播中插入伪量化算子（Quantize-and-Dequantize）模拟舍入噪声，让权重在训练中自适应量化网格。效果好，但需要昂贵的大规模训练。
2. **训练后量化（Post-Training Quantization, PTQ —— 工业界绝对主流）**：拿训练好的 `bf16` 模型，仅用几百条校准样本（Calibration Data）在一张 GPU 上跑几小时即可完成量化：
   * **代表作一：[GPTQ (Frantar et al., 2022)](https://arxiv.org/abs/2210.17323)**
     基于二阶海森矩阵信息（Optimal Brain Quantization 思想）逐层、逐列对权重进行量化：当把某一列权重进行四舍五入量化产生误差 $\delta w$ 后，**利用逆海森矩阵 $H^{-1}$ 动态调整后面尚未被量化的其余权重列**，以精确抵消该列带来的层输出误差 $\|WX - \hat{W}X\|_2^2$！GPTQ 仅需约 4 个 GPU 小时就能把 175B 模型无损压进 3～4 bits，首次实现单张 80GB GPU 跑 175B 模型。
   * **代表作二：[AWQ (Activation-aware Weight Quantization, Lin et al., 2023)](https://arxiv.org/abs/2306.00978)**
     MIT 韩松团队发现了一个极其重要的现象：**大语言模型在推理时，激活值（Activations $X$）的少数特定通道（约 $0.1\% \sim 1\%$）数值幅度永远异常巨大（Outlier Salient Channels）**！
     ```
     [普通无差别 RTN 量化 (误差极大)]                 [AWQ 激活感知量化 (保护 1% 显著激活通道对应的权重行!)]
       激活 X (含极大离群通道!)   权重 W (全压成 INT3)     激活 X                  权重 W
       [  0.2,  0.1, -0.3 ]     [ INT3, INT3, INT3 ]   [  0.2,  0.1, -0.3 ]    [ INT3, INT3, INT3 ]
       [ 85.0, 92.0, 78.0 ]  @  [ INT3, INT3, INT3 ]   [ 85.0, 92.0, 78.0 ] @  [ FP16, FP16, FP16 ] <-- 关键行保持高精度(或放大s倍)!
       [ -0.1,  0.4,  0.2 ]     [ INT3, INT3, INT3 ]   [ -0.1,  0.4,  0.2 ]    [ INT3, INT3, INT3 ]
     ```
     既然输出 $Y = W X$，那些与**巨型激活通道**相乘的权重行/列一旦发生微小的量化舍入误差，乘上 `85.0` 后就会被放大近百倍！因此，AWQ **根据校准集上的激活值大小（而非权重本身的大小）挑出最重要的 $0.1\% \sim 1\%$ 权重通道予以重点保护**（在概念上相当于保留为 FP16，而在工程实现中为了避免硬件低效的混合精度读取，AWQ 对该权重通道乘以放大系数 $s > 1$ 后再统一量化为 INT3/INT4，同时把对应输入激活除以 $s$，从而在数学等价的前提下大幅压低了显著通道的相对量化误差！）。实测从 `fp16` 压至 `int3` 可节省 **4 倍显存**并带来 **3.2 倍推理加速**。

### 3.3 模型结构化剪枝与蒸馏修复（Model Pruning & Distillation，`model_pruning`）

除了压比特数，能不能直接拿起手术刀，把一个已经训练好的大模型（如 15B）里的某些层（Layers）、某些注意力头（Heads）或某些隐藏维度（Hidden Channels）直接硬生生切掉（Rip out），变成一个 8B 或 4B 的小模型？

* **[NVIDIA Minitron 配方 (Muralidharan et al., 2024, *Compact Language Models via Pruning and Knowledge Distillation*)](https://arxiv.org/abs/2407.14679)**：
  1. **第一步：在小校准集（1024 个样本）上评估各组件重要性（Importance Scoring）**：
     将样本喂入原始 15B 大模型，统计每一层、每个注意力头、每个 MLP 神经元通道的激活值幅度（Activation Magnitude）或遮挡影响。
  2. **第二步：结构化裁剪（Structured Pruning）**：
     直接删掉最不重要的层、头或通道，拼装出一个目标尺寸（如 8B 或 4B）的“科学怪人（Frankenstein）”紧凑架构，并用保留下来的原始权重直接初始化它。
  3. **第三步：用知识蒸馏修复伤口（Healing via Knowledge Distillation）**：
     刚被切掉一截的剪枝模型准确率肯定会受损，但此时只需用原始 15B 模型作为教师（Teacher），在极少量数据（**不到原始预训练数据量的 3%**！）上对剪枝后的 8B/4B 模型做蒸馏续训，模型就能迅速“愈合”！
  * **惊人收益**：用这套“剪枝 + 蒸馏”流水线从 Nemotron-4 15B 衍生出 8B 和 4B 模型，所需的训练 Token 数比从零训练 8B/4B 少 **40 倍**（全家族总训练算力节省 $1.8\times$），且蒸馏出的 Minitron 8B 在 MMLU 上比从零训练高出 **16%**！

> **课堂敏锐提问：如果某个神经元在所有校准样本上的激活值恒定为 `100`（幅度极大，但方差为 `0`），按激活幅度剪枝岂不是会把它当成最重要神经元保留下来？但它其实没有任何信息量！**
> * **Percy 的解答**：非常棒的洞察！如果一个神经元恒为常数 `100`，你确实不能直接把它一刀切掉置零（否则下游层相当于突然少了一个巨大的固定偏置，输出会彻底坏掉）；但因为它没有方差（不随输入变化），你在剪枝时完全可以**结合方差指标（Variance）把它识别出来，直接将它的常数值 `100 * w` 吸收合并到下一层的静态偏置项（Bias）中，然后安全地把这个神经元通道裁掉**！

#### 小结：构建高速推理模型的两条核心工程配方
1. **从零训练配方（From-Scratch Recipe）**：定义更快、更省 KV Cache 的模型架构（如 GQA / MLA / Hybrid SWA） $\rightarrow$ 从头预训练该高效模型。
2. **改造蒸馏配方（Distillation / Surgery Recipe）**：定义更快的目标架构 $\rightarrow$ 从现有的昂贵大模型中提取/裁剪权重进行初始化 $\rightarrow$ 用少量算力做知识蒸馏（Distillation）修复精度。

## 4. Part III：无损捷径（Lossless Shortcut）—— 投机采样（Speculative Sampling / Decoding）

前面讲的 GQA、量化、剪枝全都是“有损的（Lossy）”——它们改变了模型结构或权重精度，你总得提心吊胆地去跑评测看准确率有没有掉。
**有没有一种方法，能够在数学上 100% 保证最终采样生成的每一个 Token 的概率分布，与原始全精度巨型目标模型 $q(x)$ 逐比特完全一致（Lossless / Exact Sampling），却能让解码速度凭空快上 2～3 倍？**
这就是借鉴了 CPU 硬件“分支预测与投机执行（Speculative Execution）”思想的杰作——**投机采样（Speculative Sampling / Speculative Decoding，由 [Leviathan et al., 2022 (Google)](https://arxiv.org/abs/2211.17192) 与 [Chen et al., 2023 (DeepMind)](https://arxiv.org/abs/2302.01318) 独立提出）**！

### 4.1 第一性原理：利用“验证（Checking）远快于生成（Generation）”的不对称性

回顾第 2 节的核心结论：
* 让 70B 的目标大模型 $q$ **逐个串行生成** $K=4$ 个 Token（调用 4 次 $T=1$ 的 Generation），每次都要把 140GB 权重从 HBM 读一遍，耗时为 $4 \times t_{\text{big}}$；
* 但是，如果已经有人事先帮我们把这 $K=4$ 个候选 Token 写好了，我们把这 4 个 Token **一次性打包并行喂给 70B 大模型 $q$ 做验证（相当于 $T=4$ 的 Mini-Prefill 模式）**！因为读取一次 140GB 权重就能同时算完 4 个位置的 Logits，且 GPU 算力在 $T=1$ 时本就严重闲置，所以**大模型并行验证 4 个 Token 的耗时，几乎与它单独生成 1 个 Token 的耗时 $1 \times t_{\text{big}}$ 完全一样**！

> **一句话核心秘诀**：**对于大模型而言，批改作业（Parallel Checking）比自己从头写作业（Sequential Generation）快得多！**

### 4.2 投机采样完整算法与修正拒绝采样（Modified Rejection Sampling）

我们引入两个模型：
* **草稿模型（Draft Model $p$）**：体积极小、极快的轻量模型（例如用 8B 甚至 1B 模型作为 70B 的 Draft Model）；
* **目标模型（Target Model $q$）**：我们真正想要采样的高精度巨型模型（如 70B）。

```
[Speculative Sampling 单轮迭代流水线 (以猜 K = 4 个 Token 为例)]
  当前前缀: "The capital of France"
     │
     ▼ 1. [Draft 阶段]: 小模型 p 快速串行自回归猜出 K=4 个草稿词 (耗时极短!)
          猜出序列: x_1="is", x_2="Paris", x_3="and", x_4="sunny"  (记录草稿概率 p(x_1)...p(x_4))
     │
     ▼ 2. [Verify 阶段]: 大模型 q 一次性并行前向传播 (T=5, 耗时 ≈ 单步耗时!), 算出所有位置的真实分布 q(x)
          位置 1: q("is") ≥ p("is")     ──► 接受率 min(1, q/p) = 100% ──► ✅ 接受 "is"!
          位置 2: q("Paris") ≥ p("Paris")──► 接受率 min(1, q/p) = 100% ──► ✅ 接受 "Paris"!
          位置 3: q("and") = 0.7 p("and")──► 随机数 u < 0.7            ──► ✅ 接受 "and"!
          位置 4: q("sunny") ≪ p("sunny")──► 随机数 u > q/p (被拒绝!)   ──► ❌ 拒绝 "sunny"!
                                                                             立即从残差分布 norm(max(0, q - p))
                                                                             重采样出修正词 "which" ✅!
     │
     ▼ 本轮战果: 大模型只跑了 1 次前向传播, 却一口气合法生成了 4 个精确服从 q 分布的 Token ("is Paris and which")!
```

#### 标准投机采样单轮算法步骤（给定前缀，前瞻步长 $K$）：
1. **草稿串行生成**：用小模型 $p$ 从当前前缀出发，自回归采样生成 $K$ 个草稿 Token $\tilde{x}_1, \dots, \tilde{x}_K$，并保存它们在各步的草稿分布 $p_1(\cdot), \dots, p_K(\cdot)$。
2. **目标并行评估**：将前缀拼接上这 $K$ 个草稿 Token，**一次性并行送入目标大模型 $q$**，在单次前向传播中同时算出 $K+1$ 个位置的真实目标分布 $q_1(\cdot), \dots, q_K(\cdot), q_{K+1}(\cdot)$。
3. **顺序拒绝采样与残差重采样（Modified Rejection Sampling）**：
   对 $i = 1, 2, \dots, K$ 依次检验：
   * 生成均匀随机数 $u \sim \text{Uniform}(0, 1)$；
   * **接受准则**：若 $u \le \min\left(1, \; \frac{q_i(\tilde{x}_i)}{p_i(\tilde{x}_i)}\right)$，则**接受（Accept）** 草稿词 $\tilde{x}_i$，继续检验下一个位置 $i+1$；
   * **拒绝与残差修正准则**：一旦在某个位置 $i$ 发生 $u > \frac{q_i(\tilde{x}_i)}{p_i(\tilde{x}_i)}$，则**拒绝（Reject）** $\tilde{x}_i$ 及之后的所有草稿词，并**立即从如下截断残差分布 $q'_i(x)$ 中采样一个修正 Token $x_i$ 作为本轮的最后一个输出**，随后跳出循环开启下一轮：
     $$\mathbf{q'_i(x) = \text{normalize}\Big(\max\big(0, \; q_i(x) - p_i(x)\big)\Big) = \frac{\max(0, \; q_i(x) - p_i(x))}{\sum_{v \in V} \max(0, \; q_i(v) - p_i(v))}}$$
   * **满贯奖励（All Accepted Bonus）**：如果全部 $K$ 个草稿词都被顺利接受了，我们手头还白白握着大模型算出的第 $K+1$ 个位置的分布 $q_{K+1}(\cdot)$！直接从 $q_{K+1}(\cdot)$ 中再采样出第 $K+1$ 个词！
   * **吞吐保底与上限**：无论草稿模型猜得多烂，每跑 1 次大模型前向传播，**至少保证精确产出 $1$ 个 Token，最多可产出 $K+1$ 个 Token**！

### 4.3 为什么它是 100% 严格无损的？—— 课堂二元词表 $\{A, B\}$ 数学证明

为什么经过上面这套“以 $\min(1, q/p)$ 接受 + 拒绝后从 $\max(q-p, 0)$ 重采样”的操作后，最终输出的 Token 分布不多不少、刚好严格等于目标大模型分布 $q(x)$？
Percy 在 `lecture_10.py`（Lines 524–533）中给出了一个极其清晰、人人都能一眼看懂的**二元词表证明（Proof by Two-Element Vocabulary $\{A, B\}$）**：

* **设定**：假设词表中只有两个词 $V = \{A, B\}$。
  * 目标大模型概率为 $\big[q(A), \; q(B)\big]$（满足 $q(A) + q(B) = 1$）；
  * 草稿小模型概率为 $\big[p(A), \; p(B)\big]$（满足 $p(A) + p(B) = 1$）。
* **不妨设草稿模型高估了 $A$（Oversamples $A$）**：即 **$p(A) > q(A)$**。
  由于概率和为 1，这必然意味着草稿模型**低估了 $B$（Undersamples $B$）**：即 **$p(B) < q(B)$**。
* **计算残差重采样分布 $q'(x) \propto \max(q(x) - p(x), 0)$**：
  * 对于被高估的 $A$：因为 $q(A) - p(A) < 0$，所以 $\max(q(A) - p(A), 0) = \mathbf{0}$；
  * 对于被低估的 $B$：因为 $q(B) - p(B) > 0$，归一化后的残差分布把 **100% 的概率质量全部赋予了 $B$**：即 **$q'(A) = 0, \; q'(B) = 1$**！（也就是说：一旦拒绝发生，重采样必定 100% 抽中 $B$！）。
* **现在计算投机采样最终输出 $A$ 和 $B$ 的总概率**：
  1. **最终输出 $A$ 的概率 $\mathbb{P}[\text{Output} = A]$**：
     由于拒绝后的残差分布 $q'(A) = 0$ 不可能产生 $A$，要想输出 $A$，唯一途径是**草稿先抽中 $A$（概率 $p(A)$），且在接受检验中通过（因 $q(A) < p(A)$，接受概率为 $\frac{q(A)}{p(A)}$）**：
     $$\mathbb{P}[\text{Output} = A] = p(A) \cdot \frac{q(A)}{p(A)} + p(B) \cdot 1 \cdot 0 = \mathbf{q(A)}$$
  2. **最终输出 $B$ 的概率 $\mathbb{P}[\text{Output} = B]$**：
     有两种互斥途径可以得到 $B$：
     * **途径一**：草稿直接抽中了 $B$（概率 $p(B)$），此时因为 $q(B) > p(B)$，接受概率为 $\min(1, q(B)/p(B)) = \mathbf{1}$（100% 直接通过）；
     * **途径二**：草稿抽中了 $A$（概率 $p(A)$），但在检验中以 $1 - \frac{q(A)}{p(A)}$ 的概率**被拒绝**，随后从残差分布 $q'$ 中以 $q'(B) = \mathbf{1}$ 的概率重采样得到 $B$！
     将两条途径相加：
     $$\mathbb{P}[\text{Output} = B] = p(B) \cdot 1 + p(A) \cdot \left(1 - \frac{q(A)}{p(A)}\right) \cdot 1 = p(B) + p(A) - q(A) = 1 - q(A) = \mathbf{q(B)}$$
  **证毕（Q.E.D.）！** 草稿模型高估 $A$ 的那部分溢出概率 $p(A) - q(A)$，被拒绝机制精准拦截并 100% 转移补贴给了被低估的 $B$，使得最终边际分布与目标模型 $q$ **分毫不差**！（对于任意词表大小 $|V|$ 的一般证明完全同理）。

### 4.4 工程权衡与前沿草稿模型演进（Medusa & EAGLE）

* **前瞻步长 $K$ 的甜点（Sweet Spot）**：如果 $K$ 太小（如 $K=1$），无法充分利用大模型的并行验证吞吐；如果 $K$ 太大（如 $K=20$），不仅小模型串行跑 20 步耗时变长，而且越往后的 Token 累积被拒概率呈指数上升，算力白白浪费。实测最佳甜点通常在 **$K = 3 \sim 5$** 之间，端到端可实现 **2x～3x 的无损推理加速**。
* **如何打造更好的草稿模型（Draft Model）？**
  1. **同家族小模型 + 知识蒸馏**：例如用 8B 辅助 70B，或用 1B 辅助 8B；用第 3 节的所有有损压缩技巧（剪枝、量化、GQA）造出的极速模型如果精度不够直接上线，拿来做投机采样的 Draft Model 简直是天作之合！
  2. **[Medusa (Cai et al., 2024)](https://arxiv.org/abs/2401.10774)（多解码头并行猜测）**：为了免去维护独立小模型的麻烦，直接在原始大模型顶层挂载多个轻量级附加预测头（Decoding Heads），同时预测未来第 $+1, +2, +3, +4$ 个位置，并利用树状注意力（Tree Attention）一次性并行验证多条候选分支（与 DeepSeek-V3 的 **MTP 多 Token 预测头** 异曲同工）。
  3. **[EAGLE (Li et al., 2024)](https://arxiv.org/abs/2401.15077)（特征层自回归外推）**：不仅输入历史 Token，还直接将目标大模型**倒数第二层的高阶隐藏状态特征（Second-to-top-layer Features）**喂给一个极轻量的单层草稿网络，在特征空间做自回归外推，大幅提升了草稿接受率，在 Llama-2 70B 上实现了 **2.7x～3.5x** 的无损提速。

## 5. Part IV：动态在线工作负载系统优化 —— Continuous Batching 与 PagedAttention

在实验室跑 Benchmark 时，所有序列长度整齐划一、同时开始同时结束；但在真实的生产服务器（Live Traffic）中，情况完全是混乱的动态流：
1. **请求到达与结束时间各不相同**（不能让先算完的短回复干等长篇大论的请求）；
2. **序列长度参差不齐（Ragged Arrays）**（如果按最长序列做 Padding 会浪费海量显存与计算）；
3. **请求之间存在大量共享前缀**（如所有用户共用同一段长达数千字的 System Prompt，或者在代码生成/RL 中对同一个 Prompt 采样多条回复）。

现代推理服务器（如 Orca、vLLM、SGLang）通过将**操作系统（Operating Systems）的经典调度与内存管理思想**引入 GPU，完美化解了这三大难题。

### 5.1 Continuous Batching 与 Selective Batching（[Orca, Yu et al., OSDI 2022](https://www.usenix.org/conference/osdi22/presentation/yu)）

#### 1. 连续批处理 / 迭代级调度（Continuous Batching / Iteration-Level Scheduling）
* **传统静态批处理（Request-Level / Static Batching）的痛点**：凑齐 $B=4$ 个请求一起开始生成。如果请求 S2 只生成了 2 个词就遇到了 `<EOS>` 结束符，而请求 S1 要生成 500 个词，静态批处理被迫让 S2 在槽位里继续空转填充 498 步，直到整批全部结束才能放入新请求 S5！
* **Orca 的迭代级调度（Iteration-Level Scheduling）**：将调度粒度从“整个请求生命周期”细化到**“每一个单步 Decode 迭代（Every Single Token Step）”**！

```
[传统静态批处理 (Static Batching): 早完成的请求被迫空转等待整批最慢者!]
  时间步 t ──►  t1    t2    t3    t4    t5    t6
  Slot 1 (S1): [P1]  [P1]  [G1]  [G2]  [G3]  [G4] (长回复，一直占着整批)
  Slot 2 (S2): [P1]  [G1]  [EOS] [空转] [空转] [空转] ❌ (早已完成，但无法释放槽位也无法返回!)

[连续批处理 (Continuous Batching): 单步粒度动态换血，槽位 100% 满载!]
  时间步 t ──►  t1    t2    t3    t4    t5    t6
  Slot 1 (S1): [P1]  [P1]  [G1]  [G2]  [G3]  [G4]
  Slot 2 (S2): [P1]  [G1]  [EOS] ──► (t3结束立即剔除返回!) ──► [新请求 S3 Prefill] [S3 G1] [S3 G2] ✅
```

#### 2. 选择性批处理（Selective Batching）：彻底消灭不等长序列的 Padding 浪费
当同一个 Batch 里的请求处于不同阶段或拥有不同长度时（例如请求 1 有 `3` 个 Token、请求 2 有 `9` 个 Token、请求 3 有 `5` 个 Token），它们构成了一个参差不齐的锯齿数组（Ragged Tensor）：`[3, H]`, `[9, H]`, `[5, H]`。没有统一的序列维度 $S$，怎么一起做矩阵乘法？
* **Orca 的 Selective Batching 解法**：根据算子是否跨 Token 交互，采取不同策略：
  1. **对于非 Attention 层（MLP 线性层、QKV/O 线性投影、RMSNorm、激活函数 —— 占绝大部分 FLOPs！）**：
     这些算子对每一个 Token 都是**完全逐位置独立的（Position-wise Independent）**！它们根本不关心某个 Token 属于第几条序列。因此，直接把所有请求的 Token **首尾拼接展平（Concatenate / Flatten）成一个形状为 `[3 + 9 + 5, H] = [17, H]` 的紧凑二维大矩阵**，一次性喂给 GPU 算完，完了再按 `[3, 9, 5]` 切分开！**零 Padding，100% 有效算力利用率！**
  2. **仅对于 Attention 核心计算层**：
     因为每条序列有各自不同的历史长度 $S_i$，仅在算 Attention 时按各自真实长度分别调用算子（或使用支持变长索引 `cu_seqlens` 的 FlashAttention Varlen 内核）。

### 5.2 PagedAttention 与 vLLM（[Kwon et al., SOSP 2023](https://arxiv.org/abs/2309.06180)）：虚拟内存分页管理 KV Cache

即便有了 Continuous Batching，早期的推理系统依然面临一个致命瓶颈：**HBM 显存被严重的碎片化（Fragmentation）吃光了，导致实际能开出的并发 Batch Size 极小**！

#### 1. 传统连续显存预分配的两大碎片灾难
在传统系统中，为了把一条序列的 KV Cache 存成一个连续的张量，每当一个新请求进来，系统不知道它最终会生成多少个词，只能按照最大长度上限（如 `max_tokens = 2048`）在 HBM 中**预先划出一整块连续的显存空间**：
* **内部碎片（Internal Fragmentation）**：预留了 2048 个槽位，结果用户只生成了 100 个词就结束了，剩下的 1948 个预留槽位在生成期间完全闲置，任何其他请求都不能用；
* **外部碎片（External Fragmentation）**：随着不同长度的请求不断分配与释放，显存中留下了大量七零八落的小空隙（就像老式机械硬盘碎片一样），虽然剩余总显存还很多，但已经找不出一块足够大的连续空间来容纳新请求了！实测表明，传统系统中**真正存有有效 KV 数据的显存往往不到分配总量的 20%～40%**！

#### 2. PagedAttention 的核心机制：非连续物理块（Blocks）与块表映射
UC Berkeley 团队直接搬来了**操作系统虚拟内存分页（Virtual Memory Paging）**的经典设计：

```
[逻辑视图 (Logical KV Blocks): 每个请求看到的都是连续增长的 Token 槽位 (设每块 Block Size = 4 Tokens)]
  Request A 逻辑块:  [ Logical Block 0: "Four score and seven" ] ──► [ Logical Block 1: "years ago our ____" ]
                                │                                                   │
                                ▼ (通过进程级 Block Table 指针映射)                  ▼
[物理显存 (Physical KV Blocks in HBM): 无需连续! 按需一页一页动态分配, 彻底消除内外部碎片!]
  HBM 物理块池:  | Phys Block 0 (Req B) | Phys Block 1 (Req A, LB0) | Phys Block 2 (空闲) | Phys Block 3 (Req A, LB1) |
```

1. **固定大小分块（Fixed-Size KV Blocks）**：将整块 HBM 中的 KV Cache 空间划分为固定大小的小物理块（Physical Blocks，例如每个块固定存放 $B_{\text{block}} = 16$ 个 Token 的 Key 和 Value）。
2. **按需动态分配（On-Demand Allocation）**：请求进来时绝不预留最大长度！用满一个 16-Token 的块后，才从空闲物理块池中随便拎出下一个空闲物理块（物理地址无需相邻），并在该请求的**块表（Block Table / Page Table）**中记录 `逻辑块号 -> 物理块号` 的映射。
3. **近乎零碎片浪费**：外部碎片被彻底清零（因为所有块大小完全相同）；内部碎片最多只发生在每条序列的**最后一个未填满的物理块**上（最多浪费 $15$ 个 Token 槽位），使 KV Cache 显存利用率直接飙升至 **96% 以上**，从而可将并发 Batch Size 扩大 **2～4 倍**！

#### 3. 块级前缀共享与写时复制（Prefix Sharing & Copy-on-Write, CoW）
PagedAttention 还顺带解锁了极其强大的**跨请求 KV Cache 共享机制**（就像 Linux 中 `fork()` 子进程共享物理内存页一样）：
* **场景一：共享系统提示词（Shared System Prompt）**：成千上万个用户请求都带有相同的长篇 System Prompt 或文档背景。只需将这段公共前缀的物理 KV 块计算并存储 **1 份**，所有用户的 Block Table 前几项直接指向这几个相同的物理块即可，完全省去了重复 Prefill 计算与重复显存占用！（SGLang 进一步将其发展为全自动的 **RadixAttention** 基数树缓存）。
* **场景二：单 Prompt 多路并行采样（Parallel Sampling / Beam Search / RL GRPO Rollouts）**：
  当对同一个提示词 `"Four score and seven years ago our"` 同时生成 2 条候选回复时：

```
[PagedAttention 写时复制 (Copy-on-Write, CoW) 机制]
  Sample A1 逻辑块: [LB0] ──► [LB1] ──┐ (初始时 LB1 均指向同一个物理块 PB1, 引用计数 ref_count = 2)
  Sample A2 逻辑块: [LB0] ──► [LB1] ──┘
                                │
                                ▼ 当 A1 生成 "fathers" 而 A2 生成 "mothers" 需要写入 PB1 的同一槽位时:
  触发 Copy-on-Write! 系统检测到 PB1 的 ref_count == 2 > 1:
  • Sample A1 原地修改 PB1 (ref_count 减为 1): ["years", "ago", "our", "fathers"]
  • Sample A2 自动分配新物理块 PB2 并拷贝前 3 词后写入新词: ["years", "ago", "our", "mothers"]!
```

* **配套底层系统优化（vLLM / SGLang 标配）**：
  1. **融合分块读取与注意力计算 Kernel（Fused Block-Read + Attention Kernel）**：修改 FlashAttention / FlashDecoding 内核，使其直接根据 Block Table 指针从非连续物理地址收集（Gather）KV 块进 SRAM，避免碎片化读取的 Kernel Launch 开销；
  2. **CUDA Graphs（图捕获）**：在 Decode 阶段（每次计算耗时仅几毫秒），将一整串小算子的 CPU Launch 过程录制成一张静态的 CUDA Graph，由 GPU 硬件一次性回放执行，彻底消除 CPU 发射延迟瓶颈。

## 6. 全景总结：大模型推理优化技术栈速查表（Cheat Sheet）

| 优化维度 | 代表技术方案 | 是否无损？ | 核心作用机理 | 解决的核心痛点 |
| :--- | :--- | :--- | :--- | :--- |
| **1. 注意力架构压缩** | **GQA** ([Ainslie et al., 2023](https://arxiv.org/abs/2305.13245))<br>**MLA** ([DeepSeek-V2, 2024](https://arxiv.org/abs/2405.04434))<br>**CLA** ([Brandon et al., 2024](https://arxiv.org/abs/2405.12981)) | 有损（需训练适配，但精度几乎无损） | 在头间共享 KV (GQA)、低秩压缩至 512 维潜在向量 (MLA)、在层间共享 KV (CLA) | 将 KV Cache 缩小 $5\times \sim 15\times$，同时降低单步延迟并大幅提高最大并发 Batch Size |
| **2. 长序列复杂度解耦** | **SWA 滑动窗口** ([Mistral 7B](https://arxiv.org/abs/2310.06825))<br>**线性注意力 (Mamba-2 / DeltaNet)**<br>**稀疏注意力 (DSA / CSA)** | 有损（常采用与全局注意力混合交替） | 截断远距离 KV、用固定状态矩阵递归压缩、或用轻量 Indexer 筛选 Top-$k$ | 让非全局层的 KV Cache 与序列长度 $S$ 彻底解耦（$O(1)$ 显存），支撑百万上下文 |
| **3. 权重与缓存量化** | **GPTQ** ([Frantar et al., 2022](https://arxiv.org/abs/2210.17323))<br>**AWQ** ([Lin et al., 2023](https://arxiv.org/abs/2306.00978))<br>**FP8 / INT8 / INT4** | 有损（PTQ 训练后校准） | 用二阶海森补偿误差 (GPTQ)，或根据激活幅度保护 1% 显著通道权重 (AWQ) | 将模型权重与 KV Cache 字节数砍半至 $1/4$，直接加速 Memory-bound 读取 |
| **4. 结构化剪枝蒸馏** | **NVIDIA Minitron** ([Muralidharan et al., 2024](https://arxiv.org/abs/2407.14679)) | 有损（需 $<3\%$ 数据蒸馏愈合） | 裁剪低激活幅度的层/头/隐藏维，再用原模型做知识蒸馏修复 | 以从头训练 $1/40$ 的极低算力成本衍生出小巧高速的部署模型家族 |
| **5. 投机采样加速** | **Speculative Decoding** ([Leviathan et al., 2022](https://arxiv.org/abs/2211.17192))<br>**Medusa / EAGLE / MTP** | **100% 严格无损（Exact Distribution）** | 小模型串行快猜 $K$ 词 + 大模型并行一次前向验证 + 修正拒绝与残差重采样 | 利用“并行验证远快于串行生成”的不对称性，在小 Batch 下实现 **2x～3.5x** 无损提速 |
| **6. 动态批处理与分页** | **Continuous & Selective Batching** ([Orca](https://www.usenix.org/conference/osdi22/presentation/yu))<br>**PagedAttention** ([vLLM](https://arxiv.org/abs/2309.06180)) | **100% 系统级无损** | 迭代级动态进出队列；MLP 展平拼接消去 Padding；虚拟内存非连续分块与写时复制 (CoW) | 消除显存内外部碎片（利用率达 $>96\%$），支持前缀共享与高吞吐在线并发 |

## 7. 课后思考与自测练习（Review & Self-Check）

1. **Attention 与 MLP 算术强度及并发批大小极限推导**：
   在自回归生成阶段（Generation，$T = 1$），假设上下文长度为 $S = 4096$，模型隐藏维度 $D = 4096$，MLP 升维维度 $F = 4D = 16384$，且采用标准多头注意力（MHA，$K \cdot H = D$）。所有权重与激活均以 `bf16`（2 Bytes）存储，运行在单张 **NVIDIA H100（稠密 BF16 算力 $989\text{ TFLOP/s}$，HBM 带宽 $3.35\text{ TB/s}$，硬件拐点强度 $I_{\text{acc}} \approx 295\text{ FLOPs/Byte}$）** 上。
   * (a) 当并发批大小 $B = 1$ 时，单步生成中 MLP 层和 Attention 核心层的算术强度分别是多少？它们各自处于 Compute-bound 还是 Memory-bound？
   * (b) 如果我们将并发批大小提高到 $B = 512$（且不共享前缀），MLP 层的渐近算术强度变为多少？此时 Attention 核心层的算术强度变为多少？为什么增大 $B$ 无法让 Attention 脱离 Memory-bound？
   * (c) 如果将 MHA 改为 **GQA（令 KV 头数缩减为 Query 头数的 $\frac{1}{8}$，即 $K \cdot H = \frac{D}{8}$）**，请重新推导 Generation 阶段（$T=1, S \gg 1$）Attention 核心层的算术强度！它相比 MHA 提升了多少倍？
   * *参考解答*：
     * (a) 当 $B = 1, T = 1$ 时：MLP 层的算术强度约为 $I_{\text{MLP}} \approx B \cdot T = \mathbf{1\text{ FLOP/Byte}} \ll 295$；Attention 层的算术强度为 $I_{\text{Attn}} = \frac{S \cdot 1}{S + 1} = \frac{4096}{4097} \approx \mathbf{0.9998\text{ FLOP/Byte}} \ll 295$。**两者均处于极端的 Memory-bound 状态**。
     * (b) 当 $B = 512$ 时：MLP 层的渐近算术强度提升至 $I_{\text{MLP}} \approx B = \mathbf{512\text{ FLOPs/Byte}} > 295$，**成功跨过拐点进入 Compute-bound**（因为 512 条序列共享同一份 MLP 权重）；然而 Attention 层的算术强度**依然是 $\frac{S}{S+1} \approx \mathbf{1.0\text{ FLOP/Byte}}$**，因为每条序列都必须从 HBM 读取各自独立的 KV Cache，分子（计算量）与分母（访存量）同乘 $B$ 直接约掉，依然死卡在 Memory-bound！
     * (c) 在 $1:8$ 的 GQA 下，每个 Query 头（共 $D$ 维）保持不变，但需要从 HBM 读取的历史 $K, V$ 缓存维度缩减为 $D_{\text{kv}} = \frac{D}{8}$！因此当 $T = 1, S \gg 1$ 时：
       * Attention 计算量仍为 $\text{FLOPs} = 4 B S D$；
       * HBM 读写量降为：读 $Q(2BD)$ + 读 $K,V\big(4 B S \cdot \frac{D}{8}\big)$ + 写 $Y(2BD) \approx \frac{1}{2} B S D\text{ Bytes}$；
       * 算术强度跃升为 $I_{\text{Attn}}^{\text{GQA}} = \frac{4 B S D}{\frac{1}{2} B S D} = \mathbf{8\text{ FLOPs/Byte}}$（即直接等于每个 KV 组内的 Query 头数 **$G = \frac{N}{K} = 8$**，相比 MHA 整整提升了 **8 倍**！）。

2. **大模型推理显存、延迟与吞吐率综合精算**：
   考虑一个参数量为 $\text{num\\_params} = 30 \times 10^9$（30B）的稠密模型，层数 $L = 60$，Query 头数 $N = 64$，KV 头数 $K = 8$（采用 $1:8$ 的 GQA），单头维度 $H = 128$，上下文序列长度 $S = 4096$，使用 `bf16`（2 Bytes）在单张 **NVIDIA B200（`192 GB` HBM，显存带宽 $\text{BW} = 8.0\text{ TB/s} = 8 \times 10^{12}\text{ Bytes/s}$）** 上进行自回归解码。
   * (a) 请计算模型参数占用的静态显存（GB）以及单条序列（$S=4096$）的 KV Cache 显存（GB）。单张 192GB 的 B200 最多能同时容纳多少条并发序列（$B_{\max}$）而不发生 OOM？
   * (b) 当并发批大小设为 $B = 100$ 时，理论最小单步延迟（`latency`，单位 ms/token）和系统总吞吐率（`throughput`，单位 tokens/s）分别是多少？
   * *参考解答*：
     * (a) **参数显存**：$2 \times 30 \times 10^9 = 6 \times 10^{10}\text{ Bytes} = \mathbf{60.0\text{ GB}}$。
       **单条序列 KV Cache 显存**：
       $$\text{KV}_{\text{seq}} = S \times (K \times H) \times L \times 2 \times 2 = 4096 \times (8 \times 128) \times 60 \times 4 = 1,006,632,960\text{ Bytes} \approx \mathbf{1.007\text{ GB / seq}}$$
       **最大并发序列数 $B_{\max}$**：剩余可用于 KV Cache 的显存为 $192 - 60 = 132\text{ GB}$，故 $B_{\max} = \lfloor 132 / 1.00663 \rfloor = \mathbf{131\text{ 条序列}}$。
     * (b) 当 $B = 100$ 时：
       * 总显存读取量 $\text{Memory}(100) = 60.0\text{ GB} + 100 \times 1.00663\text{ GB} = \mathbf{160.66\text{ GB}}$；
       * 单步生成延迟 $\text{Latency} = \frac{160.66 \times 10^9\text{ Bytes}}{8.0 \times 10^{12}\text{ Bytes/s}} = 0.02008\text{ s} \approx \mathbf{20.08\text{ ms/token}}$；
       * 系统总吞吐率 $\text{Throughput} = \frac{100}{0.02008\text{ s}} \approx \mathbf{4,979.3\text{ tokens/sec}}$。

3. **投机采样（Speculative Sampling）严格无损分布的手算数值验证**：
   假设词表包含三个词 $V = \{x_1, x_2, x_3\}$。在某一步：
   * 目标大模型给出的真实概率分布为：$q = [0.5, \; 0.4, \; 0.1]$；
   * 草稿小模型给出的预测概率分布为：$p = [0.8, \; 0.1, \; 0.1]$。
   请手算验证：
   * (a) 三个词各自的接受概率 $\alpha(x_i) = \min\big(1, \frac{q(x_i)}{p(x_i)}\big)$ 是多少？本步草稿被直接接受的总体概率 $\beta = \sum_{i=1}^3 p(x_i)\alpha(x_i)$ 是多少？
   * (b) 一旦发生拒绝，归一化后的残差重采样分布 $q'(x_i) \propto \max(0, q(x_i) - p(x_i))$ 是什么？最终输出 $x_1, x_2, x_3$ 的总概率是否严格等于 $[0.5, 0.4, 0.1]$？
   * *参考解答*：
     * (a) 接受概率分别为：
       $$\alpha(x_1) = \min\left(1, \frac{0.5}{0.8}\right) = \mathbf{0.625}, \qquad \alpha(x_2) = \min\left(1, \frac{0.4}{0.1}\right) = \mathbf{1.0}, \qquad \alpha(x_3) = \min\left(1, \frac{0.1}{0.1}\right) = \mathbf{1.0}$$
       总体直接接受率为 $\beta = 0.8 \times 0.625 + 0.1 \times 1.0 + 0.1 \times 1.0 = 0.5 + 0.1 + 0.1 = \mathbf{0.70}$（即有 $30\%$ 的概率发生拒绝并进入残差重采样）。
     * (b) 未归一化残差 $\max(0, q - p) = [\max(0, -0.3), \; \max(0, 0.3), \; \max(0, 0.0)] = [0, \; 0.3, \; 0]$，归一化后残差分布为 **$q' = [0, \; 1.0, \; 0]$**（即拒绝后 100% 重采样出 $x_2$）。
       最终各词总输出概率为：
       * $\mathbb{P}[x_1] = p(x_1)\alpha(x_1) + (1 - \beta) q'(x_1) = 0.8 \times 0.625 + 0.3 \times 0 = \mathbf{0.50} = q(x_1)$；
       * $\mathbb{P}[x_2] = p(x_2)\alpha(x_2) + (1 - \beta) q'(x_2) = 0.1 \times 1.0 + 0.3 \times 1.0 = \mathbf{0.40} = q(x_2)$；
       * $\mathbb{P}[x_3] = p(x_3)\alpha(x_3) + (1 - \beta) q'(x_3) = 0.1 \times 1.0 + 0.3 \times 0 = \mathbf{0.10} = q(x_3)$。
       三者分毫不差地还原了目标模型分布 $q$！
