---
title: Stanford CS336 (Spring 2026) Lecture 17 深度课程笔记：Multimodal Models
date: 2026-10-26 14:34:00
tags:
  - 大模型
  - 多模态
  - CLIP
  - SigLIP
  - LLaVA
  - Qwen-VL
  - Chameleon
  - VQ-VAE
  - CS336
categories:
  - [笔记, CS336]
description: 全景解析 Stanford CS336 Lecture 17：多模态大模型前沿架构，涵盖对比学习视觉编码器（CLIP / SigLIP）、LLaVA 与 Qwen3-VL 跨模态演进（AnyRes / Interleaved M-RoPE / DeepStack），以及 Chameleon 离散自回归全模态架构与图文熵差范数爆炸化解方案。
mathjax: true
---

# Stanford CS336 (Spring 2026) Lecture 17 深度课程笔记：Multimodal Models

> **课程信息**：[Stanford CS336: Language Modeling from Scratch (Spring 2026)](https://cs336.stanford.edu/)
> **主讲教授**：Percy Liang
> **本讲资源**：[Lecture 17 YouTube 视频](https://www.youtube.com/watch?v=26FtD08ZpOU) | [交互式可执行讲义 (`trace=lecture_17`)](https://cs336.stanford.edu/lectures/?trace=lecture_17) | [官方源码 (lecture\_17.py)](https://www.google.com/search?q=https://github.com/stanford-cs336/lectures/blob/main/lecture_17.py) | [课件仓库 (`stanford-cs336/lectures`)](https://github.com/stanford-cs336/lectures)

## 1. 引言：从纯文本语言模型走向原生全模态模型（Omnimodels）

在 CS336 的前 16 讲中，我们从零构建了分词器（Tokenizer）、Transformer 架构、分布式训练系统、Scaling Laws、万亿级预训练数据清洗管线以及后训练对齐（SFT、RLHF 与 RLVR）。然而，此前我们讨论的所有模型都生活在一个“盲人摸象”的世界里——它们的输入是纯文本（Text），输出也是纯文本。

在真实世界中，人类智能天然是多模态的，前沿基础模型（如 `GPT-4o / GPT-5`、`Gemini`、`Claude` 与 `Qwen-VL`）也早已演进为能够处理甚至生成**文本（Text）、图像（Images）、音频（Audio）与视频（Video）**的**多模态模型（Multimodal Models）**乃至**全模态模型（Omnimodels，支持任意模态组合的输入与任意模态组合的输出）**。

### 1.1 Transformer 的“通用语言”哲学：为什么像素（Pixels）不是好的 Token？

Percy Liang 在开篇提出了一个极具穿透力的第一性原理问题：
* **已知事实 1**：Transformer 架构极其强大且可扩展（Scalable），具有极佳的系统与硬件生态，因此我们处理多模态时必须继续使用 Transformer。
* **已知事实 2**：**Transformer 说的是“Token 语言”（Transformers speak tokens）**——无论这些 Token 是离散的词表索引（Discrete IDs）还是连续的嵌入向量（Continuous Embeddings），每个 Token 在物理意义上都应当代表某种**具有一定抽象层级的语义信息单元（Semantic Unit of Information）**，从而让自注意力机制（Self-Attention）能够在不同语义单元之间高效地关联与推理。
* **核心矛盾**：在纯文本中，原始数据是字节或字符流（如 `'h', 'e', 'l', 'l', 'o'`），我们通过 BPE 分词（Lecture 1）将它们聚合为有语义的子词 Token。但在图像、音频或视频等非文本模态中，原始信号是**极高维、极低语义密度的连续采样点**！例如一张像素仅代表画面中某个孤立点上的 RGB 颜色强度 $(R, G, B) \in [0, 255]^3$——单个像素本身几乎不包含任何高级语义，而且一张普通高清图像就包含数百万个像素，如果直接把每个像素当作一个 Token 喂给 $O(N^2)$ 的 Transformer，不仅计算量瞬间爆炸，模型也会迷失在毫无意义的局部高频噪声中。

因此，构建多模态大模型的全部核心挑战可以归结为两个正交的基本问题：
1. **多模态输入（Understanding / Encoding）**：如何将图像、音频、视频等非文本模态高保真、高效率地编码为 Transformer 能够直接消化和推理的“语义 Token 序列”？
2. **多模态输出（Generation / Decoding）**：如何让 Transformer 从隐状态中解码并生成非文本模态（如生成逼真的图像、视频或语音流）？

本讲将重点围绕**问题 1（视觉理解与多模态输入编码）**展开深度剖析，并在最后通过 **[Meta Chameleon (2024)](https://arxiv.org/abs/2405.09818)** 探讨将问题 1 与问题 2 彻底统一在单一离散 Token 空间下的全模态自回归探索。

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│                  CS336 Lecture 17 全景架构：多模态大模型的三大核心组件与演进脉络                  │
├─────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                 │
│  [第一阶段：预训练视觉语义编码器 (Vision Encoders)]                                             │
│    ├─► 1. CLIP (Radford et al., 2021): 4 亿图文对、32,768 巨型 Batch 对称交叉熵对比学习         │
│    └─► 2. SigLIP (Zhai et al., 2023): 逐对二元 Sigmoid 损失、Ring 环形分布式分块通信、Batch 解耦│
│          │                                                                                      │
│          ▼ (冻结或微调视觉编码器，通过 Adapter 桥接预训练大语言模型)                            │
│  [第二阶段：拼接式视觉语言模型 (Late/Modular Fusion VLMs — 连续视觉嵌入路线)]                    │
│    ├─► 3. LLaVA 系列 (2023–2024):                                                               │
│    │        • LLaVA-1.0: CLIP ViT-L/14 + 线性层 W + Vicuna，纯文本 GPT-4 合成 158K 视觉指令数据 │
│    │        • LLaVA-OneVision: SigLIP + MLP + Qwen-2，引入 AnyRes 网格切分解决高分辨率 OCR 瓶颈 │
│    └─► 4. Qwen-VL 系列 (2023–2025):                                                             │
│             • Qwen-VL: OpenCLIP + 单层 Cross-Attention (固定压缩至 256 Tokens) + 边界框特殊标签 │
│             • Qwen2-VL: 原生动态分辨率 (2x2 Patch 合并) + 3D 多模态旋转位置编码 (M-RoPE: t,h,w) │
│             • Qwen3-VL: SigLIP 2 + 频率交错 M-RoPE + 显式视频时间戳 + √L 损失归一化 + DeepStack │
│          │                                                                                      │
│          ▼ (抛弃连续编码器与拼接适配器，将图像彻底离散化为词表 ID)                              │
│  [第三阶段：早期融合全模态自回归模型 (Early-Fusion Discrete Token Omnimodels)]                   │
│    └─► 5. VQ-VAE (2017) & Meta Chameleon (2024):                                                │
│             • VQ-VAE 向量量化码本将 512x512 图像映射为 1024 个离散整数 Token (词表大小 8192)    │
│             • 单一 Transformer 统一自回归理解与生成；剖析图文熵差引发的范数爆炸与 QK-Norm/Z-Loss│
└─────────────────────────────────────────────────────────────────────────────────────────────────┘
```

## 2. 视觉编码器基石（Part I）：从监督分类到 CLIP 与 SigLIP

要让语言模型“看懂”图像，我们首先需要一个能够将二维像素矩阵转化为富含语义的连续向量序列的**视觉编码器（Vision Encoder）**。

### 2.1 [CLIP (Radford et al., 2021)](https://arxiv.org/abs/2103.00020)：用自然语言监督学习可迁移视觉表征

在 2021 年之前，计算机视觉的主流范式是在人工标注的固定类别数据集（如包含 1,000 个类别、128 万张图像的 **ImageNet**）上训练有监督分类网络（如 ResNet）。然而这种范式有两个致命缺陷：一是人工标注成本极高、数据规模难以扩展；二是模型被死死绑定在预设的 1,000 个闭集标签（Closed-set Labels）上，无法泛化到开放世界的新概念（例如做 OCR 文字识别、地理位置判断或细粒度动作识别）。

OpenAI 提出的 **[CLIP (Contrastive Language-Image Pre-training, Radford et al., 2021)](https://arxiv.org/abs/2103.00020)** 彻底颠覆了这一范式：**互联网上天然存在着数以亿计的“图像 + 伴随文本描述（Alt-text / Caption）”对，为什么不直接利用自由形式的自然语言作为监督信号来训练视觉编码器？**

#### 1. 为什么不用生成式目标（Image $\to$ Text），而要用对比学习（Contrastive Learning）？
既然有成对的 `(Image, Caption)`，最直观的想法是训练一个图像编码器接一个文本解码器，直接逐词预测图像标题（即 Image Captioning，如 Bag-of-Words 词袋预测或 Autoregressive Transformer 自回归标题生成）。
然而，CLIP 论文著名的 **Figure 2 样本效率对比实验**证明：
* **生成式目标极其低效**：一张图片对应的自然语言标题充满了与核心视觉概念无关的句法细节、相机参数、甚至噪声词汇。如果强迫模型去逐字精确生成整句标题（Predict exact words），模型会把海量算力浪费在拟合文本分布的琐碎细节上。
* **对比学习效率提升 4 倍以上**：一旦将目标从“逐词生成标题”切换为“在给定的一个 Batch 中，判断哪张图片和哪条文本是一对”（Contrastive Objective），模型在零样本 ImageNet 分类上的训练效率瞬间获得了 **4 倍以上（相比自回归生成）乃至十余倍** 的飞跃！

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│                    CLIP 对称对比学习矩阵与零样本推理流水线 (Radford et al., 2021)                │
├─────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                 │
│  [1. 对比预训练阶段 (Contrastive Pre-training, Batch Size N = 32,768)]                          │
│                                                                                                 │
│                     文本嵌入 T_1   T_2   T_3  ...   T_N  (经 Text Encoder 编码 + L2 归一化)      │
│                   ┌─────────┬─────┬─────┬─────┬─────────┐                                       │
│    图像嵌入  I_1  │ I_1·T_1 │I_1T2│I_1T3│ ... │ I_1·T_N │ ◄── 对角线 (i = j): N 个正样本对      │
│   (经 ViT    I_2  ├─────────┼─────┼─────┼─────┼─────────┤     非对角线 (i ≠ j): N^2 - N 个负样本│
│    编码 +    I_3  │ I_2·T_1 │I_2T2│I_2T3│ ... │ I_2·T_N │                                       │
│    L2 归一化) ... │   ...   │ ... │ ... │ ... │   ...   │ ──► 沿【每一行】与【每一列】分别算    │
│              I_N  │ I_N·T_1 │I_NT2│I_NT3│ ... │ I_N·T_N │     Softmax 交叉熵，取双向平均!       │
│                   └─────────┴─────┴─────┴─────┴─────────┘                                       │
│                                                                                                 │
│  [2. 零样本分类推理阶段 (Zero-Shot Inference via Prompt Engineering)]                           │
│    • 将 K 个类别词 {plane, car, dog, ..., bird} 填入提示模板: "A photo of a {object}."          │
│    • 计算测试图像特征 I_test 与 K 个文本模板特征 T_k 的余弦相似度，直接取 argmax_k (I_test·T_k) │
└─────────────────────────────────────────────────────────────────────────────────────────────────┘
```

#### 2. CLIP 的数学形式与官方 NumPy 级核心实现
设一个训练 Batch 中包含 $N$ 个图文对 $\{(x_i^{\text{img}}, x_i^{\text{txt}})\}_{i=1}^N$（在原版 CLIP 中，**全局 Batch Size 高达 $N = 32,768$**！）。
1. **双塔特征提取与投影**：
   图像编码器 $f_{\text{img}}(\cdot)$ 提取维度为 $d_e$ 的视觉特征，文本编码器 $f_{\text{txt}}(\cdot)$ 提取维度为 $d_t$ 的文本特征，随后分别通过线性投影矩阵 $W_i \in \mathbb{R}^{d_e \times d}$ 和 $W_t \in \mathbb{R}^{d_t \times d}$ 映射到共享的多模态嵌入空间，并执行 **$L_2$ 单位球面归一化**：
   $$I_i = \frac{f_{\text{img}}(x_i^{\text{img}}) W_i}{\|f_{\text{img}}(x_i^{\text{img}}) W_i\|_2} \in \mathbb{R}^d, \qquad T_j = \frac{f_{\text{txt}}(x_j^{\text{txt}}) W_t}{\|f_{\text{txt}}(x_j^{\text{txt}}) W_t\|_2} \in \mathbb{R}^d$$
2. **缩放余弦相似度矩阵（Logits Matrix）**：
   计算所有 $N \times N$ 个图文对之间的余弦相似度，并乘以一个**可学习的温度参数（Learnable Temperature Parameter）** $\exp(t)$（其中 $t = \log(1/\tau)$ 直接作为可训练标量参数参与梯度更新，并截断防溢出）：
   $$L_{i, j} = \exp(t) \cdot \langle I_i, T_j \rangle, \qquad \forall i, j \in \{1, \dots, N\}$$
3. **对称交叉熵损失（Symmetric Cross-Entropy Loss）**：
   将第 $i$ 行视为一个 $N$ 分类问题（给定图像 $i$，从 $N$ 条文本中找出正确配对的第 $i$ 条文本），将第 $j$ 列视为另一个 $N$ 分类问题（给定文本 $j$，从 $N$ 张图像中找出正确配对的第 $j$ 张图像），总损失为两者的算术平均：
   $$\mathcal{L}_{\text{CLIP}} = \frac{1}{2N} \sum_{i=1}^N \left[ -\log \frac{\exp(L_{i,i})}{\sum_{k=1}^N \exp(L_{i,k})} - \log \frac{\exp(L_{i,i})}{\sum_{k=1}^N \exp(L_{k,i})} \right]$$

```python
# CLIP 官方论文 Figure 3 核心伪代码 (NumPy / PyTorch 风格):
# image_encoder - ResNet or Vision Transformer
# text_encoder  - CBOW or Text Transformer
# I[n, h, w, c] - minibatch of aligned images
# T[n, l]       - minibatch of aligned texts
# W_i[d_i, d_e] - learned proj of image to embed
# W_t[d_t, d_e] - learned proj of text to embed
# t             - learned temperature parameter

# 1. 分别提取单模态表征 [n, d_i], [n, d_t]
I_f = image_encoder(I)
T_f = text_encoder(T)

# 2. 投影到共享多模态空间并做 L2 归一化 [n, d_e]
I_e = l2_normalize(np.dot(I_f, W_i), axis=1)
T_e = l2_normalize(np.dot(T_f, W_t), axis=1)

# 3. 计算缩放余弦相似度矩阵 [n, n]
logits = np.dot(I_e, T_e.T) * np.exp(t)

# 4. 计算行列对称的交叉熵损失 (对角线 0, 1, ..., n-1 为正样本标签)
labels = np.arange(n)
loss_i = cross_entropy_loss(logits, labels, axis=0) # Image-to-Text (沿列归一化或按行看取决于约定)
loss_t = cross_entropy_loss(logits, labels, axis=1) # Text-to-Image
loss   = (loss_i + loss_t) / 2
```

#### 3. CLIP 的架构、数据与训练系统细节（`lecture_17.py`）
* **预训练数据集**：OpenAI 构建的闭源数据集 **WIT (WebImageText)**，包含从互联网爬取并经过词频平衡过滤的 **4 亿（400M）个 `(图像, 文本)` 对**。由于 WIT 未开源，学术界与开源社区随后复现了 **[OpenCLIP](https://github.com/mlfoundations/open_clip)**，并基于 Common Crawl 构建了包含 50 亿图文对的开源数据集 **[LAION-5B (Schuhmann et al., 2022)](https://arxiv.org/abs/2210.08402)**（Percy 在课上调侃了一个非常有趣的“鸡生蛋”循环：LAION-5B 在清洗 Common Crawl 脏数据时，正是利用 OpenAI 已经开源的 **CLIP 模型权重**去计算图文余弦相似度、过滤掉图文不匹配的垃圾样本，从而才训出了开源的 OpenCLIP！）。
* **图像预处理与 Vision Transformer ([ViT, Dosovitskiy et al., 2020](https://arxiv.org/abs/2010.11929)) 视觉编码器**：
  1. **Resize & Center Crop**：先将任意长宽比的输入图像的**短边（Shorter side）缩放至固定分辨率 $S$**（通常为 $S = 224$，最后额外在 $S = 336$ 高分辨率上微调 1 个 Epoch），然后居中裁剪出 $S \times S$ 的正方形图像。
  2. **Patch 切分（Patchification）**：以旗舰配置 **ViT-L/14 @ 336px** 为例，将 $336 \times 336 \times 3$ 的图像切分为边长 $P = 14$ 的不重叠正方形小块（Patches），总 Patch 数量为：
     $$N_{\text{patches}} = \left(\frac{336}{14}\right) \times \left(\frac{336}{14}\right) = 24 \times 24 = 576 \text{ 个视觉 Token}$$
     每个 $14 \times 14 \times 3 = 588$ 维的 RGB 像素块通过一个线性卷积层（Patch Embedding）映射为 1 个连续向量，加上 **1D 可学习位置编码（1D Positional Embedding）** 和一个前置的 `[CLS]` Token 后，送入标准的 Transformer Encoder（并在激活层前额外增加了一层 LayerNorm）。最后通过一层 Attention Pooling 提取整张图的全局语义向量 $I_i$。
* **文本编码器**：采用类似 GPT-2 的因果自回归 Transformer（Base 规模：63M 参数，12 层，宽 512，8 头；BPE 词表大小 `49,152`，最大序列长度截断为 `76`），取句子末尾 **`<EOS>` Token 最高层的激活向量**作为整句文本的表征 $T_j$。
* **训练算力消耗**：训练最大号的 ResNet（RN50x64）在 592 张 V100 GPU 上耗时 18 天；训练旗舰版 **ViT-L/14** 在 **256 张 V100 GPU 上训练了 12 天**（Batch Size = `32,768`，共训练 32 个 Epochs）。
* **零样本提示工程（Prompt Engineering & Ensembling）**：
  由于互联网上的图片配文极少只是一个孤立的名词（如 `"dog"`），而是短语（如 `"A photo of a cute dog"`），在零样本分类时，将类别名嵌入模板 `"A photo of a {label}."` 相比直接使用单个单词能让 ImageNet 准确率立刻提升 **1.3%**；若针对不同细粒度领域定制提示词（如在卫星图像数据集上使用 `"A satellite photo of a {label}."`，在宠物数据集上使用 `"A photo of a {label}, a type of pet."`）并对 80 个不同提示模板的文本嵌入取平均（Prompt Ensembling），还能再提升 **3.5%**，最终在不看哪怕一张 ImageNet 训练图的情况下，击败了在 128 万张 ImageNet 图像上全监督训练的 ResNet-50，并在 ImageNet-R / ImageNet-Sketch 等分布外（OOD）鲁棒性基准上大幅碾压有监督模型！

> **课堂深度互动问答（Classroom Q&A on CLIP，`00:11:30` – `00:24:00`）**：
> * **学生提问 1（自监督对比学习 SimCLR vs. 语言监督 CLIP）**：*“为什么一定要引入文本？像 SimCLR 或 DINO 那样，对同一张狗的图片做随机裁剪、调色等数据增强（Data Augmentation），直接做纯图像的对比学习（Image-Image Contrastive Learning）不行吗？”*
>   **Percy 解答**：这是一个触及表征学习灵魂的问题！如果你只对**同一张图片**做裁剪、旋转、变色并要求它们的特征靠近，模型学到的仅仅是**低层级的像素/几何不变性（Low-level Invariance）**——它只知道这只特定的金毛犬左看右看是同一只狗，但它根本无法知道一张金毛犬的照片和一张哈士奇的照片在高层语义上都属于“狗（Dog）”！相反，**自然语言天然是人类对客观世界的高层抽象与聚类锚点**：当互联网上成千上万张长相迥异的狗的照片都被配上了包含 `"dog"` 或 `"puppy"` 的文本时，语言就把所有不同品种、不同姿态的狗在特征空间里拉到了一起！因此，用语言做对齐学到的是真正高维的**语义等价性（High-level Semantic Equivalence）**。
> * **学生提问 2（巨型 Batch 中的假负样本 False Negatives）**：*“当 Batch Size 高达 32,768 时，同一个 Batch 里极有可能同时抽到两张不同但都画着‘狗’的图片 $(I_1, T_1)$ 和 $(I_2, T_2)$。在计算交叉熵时，$I_1$ 和 $T_2$（非对角线项）会被强行当作负样本推开，这难道不会损害模型吗？”*
>   **Percy 解答**：理论上确实存在这种“假负样本（Collision / False Negatives）”噪声，但在实践中它几乎不构成问题，原因有二：(1) 互联网野生文本描述具有极高的多样性与细粒度特异性（例如 $T_1$ 写的是 `"My golden retriever playing on the beach in Santa Cruz"`，而 $T_2$ 写的是 `"A sleepy pug on a red sofa"`，对于 $I_1$ 而言，$T_1$ 确实比 $T_2$ 匹配得多）；(2) 即使偶尔出现完全同义的碰撞，在 4 亿图文对和 32,768 个负样本的统计平均下，这种稀疏噪声会被完全平滑掉（当然，接下来我们要讲的 **SigLIP** 将单点碰撞与全局归一化彻底解耦，让训练更加稳健）。

### 2.2 [SigLIP (Zhai et al., 2023)](https://arxiv.org/abs/2303.15343)：用逐对 Sigmoid 损失打破全局 Softmax 的系统与算力枷锁

尽管 CLIP 取得了巨大成功，但站在 CS336 **系统工程（Systems）**的视角审视 CLIP 的对称 Softmax 交叉熵损失，你会发现一个极其沉重的分布式计算包袱：
* **Softmax 的全局耦合瓶颈（Global View Bottleneck）**：
  为了计算第 $i$ 行的分母 $\sum_{k=1}^N \exp(L_{i,k})$ 和第 $j$ 列的分母 $\sum_{k=1}^N \exp(L_{k,j})$，系统必须在所有 GPU/TPU 节点之间通过 **All-Gather** 同步全部 $N = 32,768$ 个图像嵌入和文本嵌入，并维护完整的 $N \times N$（即 $32,768 \times 32,768 \approx 10.7 \text{ 亿}$ 个元素）相似度矩阵的归约项。此外，Softmax 损失函数本身的数学性质与 Batch Size $N$ 强耦合——当 $N$ 较小时（例如只有几张卡、Batch Size 只有 2K 或 4K），Softmax 对比学习的性能会急剧退化。

为了彻底砍掉全局 Softmax 归一化，Google Research 提出了 **[SigLIP (Sigmoid Loss for Language-Image Pre-training, Zhai et al., 2023)](https://arxiv.org/abs/2303.15343)**——这也是如今几乎所有最强开源视觉语言模型（如 **LLaVA-OneVision**、**PaliGemma**、**Qwen3-VL**）一致选用的新一代默认视觉编码器！

#### 1. SigLIP 的逐对二元逻辑回归损失（Pairwise Sigmoid Loss）
SigLIP 的核心思想极其优雅：**不要把图文匹配当成一个“$N$ 选 $1$ 的多分类问题（Multiclass Classification via Softmax）”，而是把 $N \times N$ 矩阵中的每一个网格点 $(i, j)$ 都当成一个完全独立的“二分类问题（Binary Classification via Sigmoid）”——判断图像 $I_i$ 和文本 $T_j$ 究竟是匹配的（$z_{ij} = +1$）还是不匹配的（$z_{ij} = -1$）！**

定义符号标签：当 $i = j$（对角线正样本对）时，$z_{ij} = +1$；当 $i \neq j$（非对角线负样本对）时，$z_{ij} = -1$。利用逻辑斯蒂函数 $\sigma(u) = \frac{1}{1 + e^{-u}}$，SigLIP 的损失函数写为：
$$\mathcal{L}_{\text{SigLIP}} = \frac{1}{|\mathcal{B}|} \sum_{i=1}^{|\mathcal{B}|} \sum_{j=1}^{|\mathcal{B}|} \underbrace{\log \Big( 1 + \exp\big( -z_{ij} (t \cdot \langle I_i, T_j \rangle + b) \big) \Big)}_{-\log \sigma\big(z_{ij}(t \cdot \langle I_i, T_j \rangle + b)\big)}$$
* **为什么必须引入可学习的偏置项 $b$（Learnable Bias $b$，初始化为 $b = -10$）？**
  在一个大小为 $|\mathcal{B}| = N$ 的 Batch 中，只有 $N$ 个正样本对（$z_{ij} = +1$），却有 $N(N-1)$ 个负样本对（$z_{ij} = -1$），正负样本比例极度失衡（高達 $1 : (N-1)$）。如果不加偏置（$b=0$），在训练初期随机初始化时，所有负样本对的概率都接近 $0.5$，海量负样本产生的巨大梯度会瞬间淹没正样本，模型第一步更新就会把所有预测暴力推向全负。为此，SigLIP 引入了一个可学习的标量偏置 $b$，并**将其初始化为一个较大的负数（如 $b = -10$）**——这样在训练刚起步时，对于负样本（$z_{ij}=-1$），$-z_{ij}(t \langle I_i, T_j \rangle + b) \approx -10 \ll 0$，初始负样本先验概率已经非常接近 $0$，从而从第 1 步迭代起就保证了梯度的健康平衡！

```python
# SigLIP 官方论文 Figure 2 核心伪代码 (与 CLIP 对比):
# img_emb, txt_emb : L2-normalized embeddings [B, D]
# t_prime, b       : learnable temperature and bias (initialized to log(10), -10)
# n                : global batch size B

t = exp(t_prime)
zimg = l2_normalize(img_emb)
ztxt = l2_normalize(txt_emb)

# 1. 计算缩放并带偏置的成对 Logits [B, B]:
logits = dot(zimg, ztxt.T) * t + b

# 2. 构造二元符号标签矩阵: 对角线为 +1, 非对角线为 -1:
labels = 2 * eye(n) - ones(n)   # +1 on diagonal, -1 everywhere else

# 3. 逐元素独立计算二元对数几率损失 (无需跨行/跨列求 Softmax 分母!):
l = -sum(log_sigmoid(labels * logits)) / n
```

#### 2. SigLIP 的分布式系统优势：Ring 环形分块计算与算力奇迹
为什么去掉 Softmax 分母能在分布式系统层面带来质的飞跃？让我们跟随 Percy（`00:25:45` – `00:29:30`）剖析 SigLIP 论文 Figure 1 的**分布式环形分块计算（Chunked Ring Communication）**：

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│             SigLIP 分布式环形分块计算机制 (Chunked Pairwise Loss across D Devices)              │
├─────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                 │
│  假设全局 Batch Size = B，切分在 D = 4 台设备 (TPU/GPU) 上，每台设备本地仅持有 b = B/D 个样本:   │
│    • 设备 0 固定持有本地图像块 I^(0)，初始持有本地文本块 T^(0)                                  │
│    • 设备 1 固定持有本地图像块 I^(1)，初始持有本地文本块 T^(1)  ...                             │
│                                                                                                 │
│  [Step 1: 计算对角线分块 (正样本 + 本地负样本)]                                                 │
│    每台设备 d 直接在本地计算 b × b 小矩阵 L(I^(d), T^(d)) 的 Sigmoid 损失并累加，【零显存膨胀】!│
│                                                                                                 │
│  [Step 2 ~ D: 环形轮转文本块 (Ring Collective Permute)]                                         │
│    设备 0 ──发送 T^(0)──► 设备 1 ──发送 T^(1)──► 设备 2 ──发送 T^(2)──► 设备 3 ──► 设备 0       │
│    每收到邻居传来的文本块 T^((d-s) mod D)，立即计算 b × b 非对角块损失并直接丢弃临时矩阵!       │
│                                                                                                 │
│  ★ 系统收益: 单卡峰值显存从 O(B^2) 降至 O((B/D)^2)，通信与计算完美流水线重叠，轻松扩展至 B=1M! │
└─────────────────────────────────────────────────────────────────────────────────────────────────┘
```

* **数学解耦与小 Batch 友好性**：
  在 CLIP 的 Softmax 中，改变 Batch Size $B$ 会直接改变损失函数分母中的求和项数量，从而改变了优化目标本身；而在 SigLIP 中，由于总损失只是所有独立样本对损失的算术平均，**损失函数的期望形式与 Batch Size 完全解耦**！实验表明：
  1. 在较小 Batch Size（如 $B \le 16\text{K}$）下，SigLIP 的准确率显著优于 CLIP（甚至仅用 **4 张 TPUv4 芯片训练 2 天**，配合冻结视觉编码器的 LiT 策略，就能达到 **84.5%** 的 ImageNet 零样本准确率！）；
  2. 从零训练完整 SigLIP（基于多语言图文数据集 **[WebLI, Chen et al., 2022](https://arxiv.org/abs/2209.06794)**，包含 100 亿图文对与 100+ 种语言及 OCR 标注）**仅需 32 张 TPUv4 训练 5 天**——相比 CLIP 的 256 张 V100/TPUv3 训练 10–12 天，算力门槛降低了整整一个数量级；
  3. 当团队借助环形分块机制将 Batch Size 一路推向惊人的 **$B = 1,000,000$（100 万）**时，他们发现准确率在 **$B \approx 32\text{K}$** 处就已经彻底饱和（Critical Batch Size $\approx 32\text{K}$），再往上盲目扩大 Batch Size 并不会带来额外收益。

| 对比维度 | **[CLIP (Radford et al., 2021)](https://arxiv.org/abs/2103.00020)** | **[SigLIP (Zhai et al., 2023)](https://arxiv.org/abs/2303.15343)** |
| :--- | :--- | :--- |
| **核心损失函数** | 行列双向对称 **Softmax 交叉熵损失**（$N$ 选 $1$ 多分类） | 逐元素独立 **二元 Sigmoid 对数几率损失**（成对二分类） |
| **全局归一化依赖** | **强依赖**：每一行与每一列都必须汇总全局所有 $N$ 个样本以计算 Softmax 分母 | **零依赖**：每对 $(I_i, T_j)$ 损失完全独立，无需全局分母归一化 |
| **正负样本不平衡处理** | 隐式包含在 $\log \sum_{k=1}^N \exp(L_{ik})$ 分母中 | 引入**可学习标量偏置 $b$（初始化为 $-10$）**抑制初期负样本梯度爆炸 |
| **单卡峰值矩阵显存** | $O(B^2)$ 或依赖复杂的跨卡分片归约 | 仅需 **$O\big((B/D)^2\big)$**（通过 Ring 环形轮转文本块逐块累加即焚） |
| **小 Batch ($<16\text{K}$) 表现** | 随 Batch Size 缩小显著退化 | **显著优于 CLIP**（损失函数与 Batch Size 解耦） |
| **典型预训练数据与算力** | WIT (400M 英文图文对)，256 TPUv3 / V100 训练 10–12 天 | [WebLI](https://arxiv.org/abs/2209.06794) (10B 多语言图文对)，**32 TPUv4 仅需训练 5 天** |

## 3. 拼接式视觉语言模型（Part II）：LLaVA 系列演进（1.0 $\to$ 1.5 $\to$ OneVision）

有了强大的视觉编码器（CLIP 或 SigLIP）和强大的预训练语言模型（如 LLaMA、Vicuna 或 Qwen），一个最自然的问题浮现出来：**我们能否像搭积木一样，用一根轻量级的“转接线（Adapter / Projector）”把视觉编码器的输出直接插进大语言模型的词嵌入输入端，从而让纯文本 LLM 瞬间获得视觉对话与推理能力？**

这正是 **拼接式视觉语言模型（Modular / Late-Fusion Vision-Language Models, VLMs）** 的核心思想，而开创并普及这一开源标准范式的代表作就是 **[LLaVA (Large Language and Vision Assistant, Liu et al., 2023)](https://arxiv.org/abs/2304.08485)**。

### 3.1 [LLaVA-1.0 (Liu et al., 2023)](https://arxiv.org/abs/2304.08485)：极简线性投影与纯文本 GPT-4 合成视觉指令数据

#### 1. 极简三明治架构（Vision Encoder + Linear Projector + LLM）
LLaVA-1.0 的架构简单到了极致：
1. **视觉编码器 $g(\cdot)$**：直接拿来预训练好的 **CLIP ViT-L/14**，对于输入图像 $\mathbf{X}_{\text{v}}$，提取其最后一层 Transformer Layer 之前（或之后）的网格 Patch 特征序列 $\mathbf{Z}_{\text{v}} = g(\mathbf{X}_{\text{v}}) \in \mathbb{R}^{N_{\text{patches}} \times d_{\text{v}}}$。
2. **可训练线性投影矩阵 $\mathbf{W}$（Linear Projection Adapter）**：
   用一个单层线性矩阵 $\mathbf{W} \in \mathbb{R}^{d_{\text{v}} \times d_{\text{llm}}}$ 将每个视觉 Patch 向量 $\mathbf{Z}_{\text{v}}$ 直接线性映射到与语言模型词嵌入（Word Embedding）完全相同的维度 $d_{\text{llm}}$，得到**连续视觉 Token 序列 $\mathbf{H}_{\text{v}}$**：
   $$\mathbf{H}_{\text{v}} = \mathbf{Z}_{\text{v}} \mathbf{W} \in \mathbb{R}^{N_{\text{patches}} \times d_{\text{llm}}}$$
3. **自回归语言模型 $f_\phi(\cdot)$（Vicuna / LLaMA）**：
   将视觉连续 Token 序列 $\mathbf{H}_{\text{v}}$ 与用户文本指令的词嵌入序列 $\mathbf{H}_{\text{q}}$ 沿序列维度直接拼接为 $[\mathbf{H}_{\text{v}}; \mathbf{H}_{\text{q}}]$，原封不动地送入因果自回归 Transformer 解码器中，逐 Token 预测助手的文本回答 $\mathbf{X}_{\text{a}}$！

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│                        LLaVA 标准视觉语言模型架构 (Liu et al., 2023)                            │
├─────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                 │
│    文本回答输出 X_a :   "The image shows a dog sitting on a skateboard..."                      │
│                              ▲      ▲      ▲      ▲      ▲                                      │
│                   ┌──────────┴──────┴──────┴──────┴──────┴──────────┐                           │
│                   │       Language Model f_φ (Vicuna / Qwen)        │                           │
│                   └──────────▲──────▲─────────────▲──────▲──────────┘                           │
│                              │      │   (拼接)    │      │                                      │
│    输入 Token 序列:      [ H_v^1, H_v^2, ...,   H_q^1, H_q^2 ]  (维度均为 d_llm)                │
│                              ▲      ▲             ▲      ▲                                      │
│                   ┌──────────┴──────┴───┐      ┌──┴──────┴──────────┐                           │
│                   │ Projection W / MLP  │      │ Word Embedding 层  │                           │
│                   └──────────▲──────▲───┘      └──▲──────▲──────────┘                           │
│                              │      │             │      │                                      │
│                   ┌──────────┴──────┴───┐         │      │                                      │
│                   │ Vision Encoder (ViT)│         │      │                                      │
│                   └──────────▲──────────┘         │      │                                      │
│                              │                    │      │                                      │
│                         输入图像 X_v          文本指令 X_q ("What is in this photo?")           │
└─────────────────────────────────────────────────────────────────────────────────────────────────┘
```

#### 2. 巧妇如何做无米之炊：用“看不见图的纯文本 GPT-4”合成 158K 视觉指令微调数据！
在 2023 年春 LLaVA 诞生时，市面上只有纯文本的 **GPT-4** API，根本没有能够看图的多模态大模型可用于数据蒸馏，而传统图文数据集（如 [MS-COCO](https://www.google.com/search?q=https://arxiv.org/abs/1405.03128)）只有简短干瘪的单句标题（Captions），没法用来做多轮对话或深度推理解答。LLaVA 团队想出了一个极具创造力的**符号化代理合成方案（Symbolic Representation Prompting）**：
* 在 MS-COCO 数据集中，每张图片本身已经附带了两组**人工标注的纯文本符号元数据**：
  1. **多条简短人工标题（Captions）**：从不同角度描述画面内容（如 `"A group of people standing outside of a black vehicle with various luggage."`）；
  2. **物体边界框与类别坐标（Bounding Boxes）**：明确标出画面中每个物体的精确位置（如 `person: [0.681, 0.242, 0.774, 0.694]`, `suitcase: [0.384, 0.696, 0.485, 0.914]`）。
* LLaVA 将这两组纯文本符号喂给**纯文本 GPT-4**，并在 Prompt 中对 GPT-4 说：*“假装你亲眼看到了这张图片（不要在回答里提及你是看坐标或标题知道的！），请根据这些多角度标题和物体空间坐标，为这张图生成高质量的视觉指令微调数据！”*
* 通过这种方式，团队围绕 **158,000（158K）张图像** 合成了三大类互补的视觉指令数据：
  1. **多轮视觉对话（Conversation，58K）**：围绕画面中的物体类型、数量、相对位置、动作展开多轮问答；
  2. **详细画面描述（Detailed Description，23K）**：对画面进行全面、结构化的长段落细致描绘；
  3. **复杂视觉推理（Complex Reasoning，77K）**：要求结合画面视觉线索与常识进行深层因果推断（例如：*“这些人遇到了什么麻烦？他们接下来应该怎么把这么多行李装进后备箱？”*）。

#### 3. LLaVA 的两阶段训练法则（Two-Stage Training Pipeline）
设完整多轮序列为 $\mathbf{X}_{\text{instruct}} = (\mathbf{X}_{\text{v}}, \mathbf{X}_{\text{q}}^1, \mathbf{X}_{\text{a}}^1, \dots, \mathbf{X}_{\text{q}}^T, \mathbf{X}_{\text{a}}^T)$，模型在所有阶段均使用标准的**自回归条件语言建模损失（仅在绿色助手回答 Token $\mathbf{X}_{\text{a}}$ 上计算交叉熵损失，对图像 Token 和用户提问 Token 进行 Mask）**：
$$\mathcal{L}_{\text{VLM}}(\theta) = -\sum_{i=1}^L \log p_\theta\big(x_i \;\big|\; \mathbf{X}_{\text{v}}, \, \mathbf{X}_{\text{instruct}, <i}\big)$$

* **Stage 1：特征对齐预训练（Pre-training for Feature Alignment）**：
  * **数据**：从 CC3M（Conceptual Captions 3M）中过滤出的 **595K 图文对**（将每条图片标题包装成单轮简单问答：`"Describe the image briefly."` $\to$ 原始标题）。
  * **冻结与更新策略**：**冻结视觉编码器 ViT 与语言模型 LLM 的全部参数，仅训练线性投影矩阵 $\theta = \{\mathbf{W}\}$**！这一步的物理本质是为已有的预训练 LLM 训练一个“兼容的视觉分词器接口（Visual Tokenizer Alignment）”，防止随机初始化的 $\mathbf{W}$ 产生乱码梯度冲垮预训练好的 LLM 权重。
* **Stage 2：端到端视觉指令微调（Visual Instruction Tuning）**：
  * **数据**：前面由 GPT-4 合成的 **158K 视觉指令数据集**（或针对 ScienceQA 基准微调）。
  * **冻结与更新策略**：**继续冻结视觉编码器 ViT，同时放开训练投影矩阵与语言模型全部权重 $\theta = \{\mathbf{W}, \phi\}$**！
* **消融实验洞见**：如果在 Stage 2 中只训练某一种类型的合成数据（如只训对话或只训详细描述），模型在综合评测上的得分会从 **`85.1`** 暴跌至 `63.1` ~ `76.7`——证明多轮感知问答、详尽描述与深层推理三类数据缺一不可。

### 3.2 从 [LLaVA-1.5 (2023)](https://arxiv.org/abs/2310.03744) 到 [LLaVA-OneVision (Li et al., 2024)](https://arxiv.org/abs/2408.03326)：AnyRes 动态高分辨率与跨模态迁移

随着视觉语言模型走向深水区，初代 LLaVA 暴露出了一个致命的物理短板：**它根本看不清高分辨率文档、图表或细小文字（OCR）**！
* **为什么初代 LLaVA 做不好 OCR？**
  回想一下，初代 LLaVA 使用的 CLIP ViT-L/14 强行把所有输入图像都缩放到 **$336 \times 336$ 像素**！如果你把一整页密密麻麻的 PDF 论文或财务报表强行压缩到 $336 \times 336$，里面的字母 `'j'` 和 `'i'`、数字 `'8'` 和 `'3'` 在像素层面就已经彻底糊成了一团马赛克——连人眼都无法辨认，更别提让模型去读表了。

为了在不重新从头训练高分辨率 ViT 的前提下解决这一瓶颈，**[LLaVA-1.5](https://arxiv.org/abs/2310.03744) / LLaVA-NeXT** 以及集大成之作 **[LLaVA-OneVision (Li et al., 2024)](https://arxiv.org/abs/2408.03326)** 引入了名为 **AnyRes（任意分辨率网格切分）** 的核心技术，并将底层架构全面升级：
* **三大组件升级**：
  1. **视觉编码器**：从 CLIP 升级为性能更强、分辨率更高的 **[SigLIP](https://arxiv.org/abs/2303.15343)**（`so400m-patch14-384`，每张 $384 \times 384$ 子图产出 $(384/14)^2 \approx 729$ 个视觉 Token）；
  2. **投影适配器（Projector）**：从单层线性变换 $\mathbf{W}$ 升级为表达能力更强的 **2 层 MLP（含 GELU 激活函数）**；
  3. **语言模型基座**：升级为当时最强的开源基座 **[Qwen-2](https://arxiv.org/abs/2407.10671)**（涵盖 `0.5B`、`7B` 至 `72B` 规模）。

#### 1. AnyRes（High-Resolution Grid Cropping + Global Thumbnail）工作原理
既然预训练好的视觉编码器（如 $384 \times 384$ 的 SigLIP）每次只能看清 $384 \times 384$ 大小的图片，而自回归 Transformer 语言模型天生就能处理任意长度的 Token 序列，那么面对一张高分辨率大图（例如 $1152 \times 768$ 的文档截图），我们为什么不**把大图切成多个 $384 \times 384$ 的局部高清切片分别编码，再拼成一条长序列喂给 LLM** 呢？

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│            LLaVA-OneVision 的 AnyRes 高分辨率切分与多模态 Token 预算分配策略                    │
├─────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                 │
│  [1. AnyRes 单张高分辨率图像处理流水线]                                                         │
│                                                                                                 │
│     原始高清文档图像 (如 W × H)                                                                 │
│          │                                                                                      │
│          ├──► 路径 A (局部高清细节): 按长宽比切分为 a × b 网格 (如 2×2 或 3×3 切片，每块 384×384)│
│          │                           每块独立过 SigLIP + MLP ──► 保留 100% 原始 OCR 文字清晰度! │
│          │                                                                                      │
│          └──► 路径 B (全局宏观构图): 整张大图直接下采样缩放到 1 个 384×384 全局缩略图 (Thumbnail)│
│                                      过 SigLIP + MLP ──► 提供跨切片的全局空间结构上下文!        │
│          │                                                                                      │
│          ▼                                                                                      │
│     最终视觉序列 = Concat( [切片_1, 切片_2, ..., 切片_(a×b)],  [全局缩略图] )                   │
│     (若 a×b 超过最大网格阈值 τ，则对每个切片特征做双线性插值 Bilinear Interpolation 降采样)     │
│                                                                                                 │
│  [2. 单图 / 多图 / 视频 三种模态的对称 Token 预算均衡 (Balancing Token Budget)]                 │
│    • 单张图像 (Single-Image): 1 张全局图 + 最多 9 张局部高清切片 = (9+1) × 729 = 7,290 Tokens   │
│    • 多张图像 (Multi-Image) : 不做网格切分，每张图仅用 Base 分辨率 (729 Tokens)，观察宏观关系   │
│    • 视频序列 (Video Frames): 最多采样 32 帧，每帧特征经插值池化压缩至 196 Tokens (共 ~6,272)   │
└─────────────────────────────────────────────────────────────────────────────────────────────────┘
```

* **Percy 对“跨模态 Token 预算平衡”的深度解读（`00:39:35` – `00:41:10`）**：
  为什么 LLaVA-OneVision 在单图时舍得给一张图分配多达 $7,290$ 个 Token（9 个高清局部切片 + 1 个全局图），而在多图时每张图只给 $729$ 个 Token，到了视频时每帧更是进一步通过双线性插值压缩到只有 $196$ 个 Token？
  因为**单张图、多张图和视频在底层其实全都是“图像序列”，但它们的信息密度与任务需求截然不同**：当你只给模型看一张图（如一页密集财报）时，用户往往要求它“拿放大镜仔细看局部细节”；当你给它看一段长视频时，相邻帧之间存在海量冗余，如果不把每帧的 Token 数强行压缩，不仅 32 帧视频瞬间撑爆上下文窗口，而且在混合训练时，漫长的视频 Token 会徹底主导梯度更新，淹没单图与文本能力！

#### 2. LLaVA-OneVision 的数据哲学、三阶段训练与惊艳的“跨模态零样本迁移（Cross-Modal Transfer）”
* **数据合成与开源贡献**：LLaVA-OneVision 收集并开源了极其丰富的单图（3.2M 样本，涵盖VQA、OCR、图表、数学）、多图（800K 样本，如找不同、多视角推理）与视频数据（其中大量复杂指令回答直接蒸馏自 GPT-4V / GPT-4o）。Percy 特别赞赏了 LLaVA 团队：**他们不仅开源了模型权重，还把清洗和合成的全部多模态数据集完整开源了出来**（同类型的完全开源标杆还有 AI2 的 **[Molmo / PixMo, Deitke et al., 2024](https://arxiv.org/abs/2409.17146)**）。
* **三阶段课程训练（Three-Stage Curriculum）**：
  1. **Stage 1（Language-Image Alignment）**：558K LCS 图文对，仅训练 Projector（MLP），冻结 ViT 与 LLM；
  2. **Stage 1.5（High-Quality Knowledge Learning）**：4M 高质量重标注描述与文档/OCR 数据，**全参数微调（ViT + MLP + LLM）**，向模型注入密集的视觉世界知识；
  3. **Stage 2（Visual Instruction Tuning）**：在 3.2M 单图 + 1.6M 单图/多图/视频混合指令集上分两步做**全参数微调**，将最大视觉上下文拓展至视频级别。
* **令人振奋的涌现现象——单图能力向多图与视频的零样本泛化（Emerging Capabilities via Cross-Modal Transfer，`00:43:30`）**：
  起初，Percy 看到训练集中塞满了成百上千种高度定制的任务数据集（如看表格、找不同、GUI 截图）时，曾怀疑这只是“头痛医头、脚痛医脚”的暴力监督学习（Task-specific Supervised Learning）。然而，LLaVA-OneVision 展现出了一系列**在训练集中从未出现过的跨模态组合泛化能力**：
  1. **单图图表 $\to$ 多图“表格+柱状图”联合推理**：训练集中只有“单张图表问答”数据，从未见过同时输入两张图（一张数据表格 + 一张柱状图）的样本；但在测试时，模型能够准确对照左图的统计表格和右图的柱状图进行联合数值核对！
  2. **单图 OCR + 多图关系 $\to$ 多张手机 GUI 截图连续导航描述**：将单图学到的精细文字识别能力无缝迁移到由多张连续手机屏幕截图组成的智能体轨迹分析中。
  3. **单图视觉提示（Visual Prompting / 紅圈高亮）$\to$ 视频动态目标追踪描述**：训练集中只有在**静态单张图片**上画一个红圈（Circle Highlight）让模型介绍圈内人物的数据；测试时给模型看一段**篮球比赛视频**，并在多帧中用红圈圈出某位移动的球员，模型竟然能直接跨帧追踪并准确描述该高亮球员在整个视频片段中的连续动作！这证明将单图、多图与视频全部统一转化为同构的视觉 Token 序列后，Transformer 的自注意力机制确实实现了跨模态技能的自由组合迁移。

## 4. 工业级原生多模态演进（Part III）：从 Qwen-VL 到 Qwen2-VL 再到 Qwen3-VL

如果说 LLaVA 系列展示了学术界如何用优雅的模块化设计撬动视觉语言能力，那么阿里通义千问团队的 **Qwen-VL 系列（2023–2025）** 则代表了工业界如何一步步抹平“拼接感”，将视觉编码、三维时空位置编码与多层特征融合推向**原生多模态 SOTA** 的全过程。

### 4.1 [Qwen-VL (Bai et al., 2023)](https://arxiv.org/abs/2308.12966)：跨注意力压缩器与视觉定位特殊 Token

2023 年发布的初代 **[Qwen-VL](https://arxiv.org/abs/2308.12966)**（基于 `Qwen-7B` 與 `OpenCLIP ViT-bigG`）在架构和训练上与初代 LLaVA 有三个显著不同：
1. **单层 Cross-Attention 视觉压缩适配器（Position-Aware Vision-Language Adapter）**：
   不同于 LLaVA 用线性层保留全部视觉 Patch Token，初代 Qwen-VL 使用了**单层交叉注意力（1-Layer Cross-Attention）**，以 **$256$ 个可学习的 Query 向量**去查询视觉编码器的特征（融入 2D 绝对位置编码），将任意大小的图像特征强制压缩为**固定长度的 $256$ 个视觉 Token**，并用特殊标记 `<img>...</img>` 包裹后送入 LLM。
2. **原生支持视觉定位（Visual Grounding）的特殊 Token 设计**：
   尽管模型只输出文本，但 Qwen-VL 通过引入两组特殊格式化 Token，让模型学会了输出精确的**物体检测边界框（Bounding Boxes）**：
   * `<ref>描述文本</ref>`：标记需要定位的目标名词短语（如 `<ref>the rightmost woman</ref>`）；
   * `<box>(x1, y1), (x2, y2)</box>`：紧跟在 `<ref>` 之后，直接以归一化整数文本坐标（范围 $[0, 1000)$）输出该物体的左上角与右下角坐标！
3. **从第一阶段就开始微调视觉编码器（Training the Vision Encoder）**：
   不同于 LLaVA 全程（或前两阶段）冻结 ViT，Qwen-VL 在 Stage 1（1.4B 清洗后的图文对预训练，分辨率 $224\times 224$）中**冻结 LLM，同时放开训练 ViT 视觉编码器与 Adapter**；在 Stage 2（多任务高质量交错数据，分辨率提升至 $448\times 448$）中**全参数联合训练（ViT + Adapter + LLM）**；最后在 Stage 3（350K 指令微调）中才冻结 ViT，微调 Adapter 与 LLM。

### 4.2 [Qwen2-VL (Wang et al., 2024)](https://arxiv.org/abs/2409.12191)：原生动态分辨率（Naive Dynamic Resolution）与 3D 多模态旋转位置编码（M-RoPE）

到了 2024 年的 **[Qwen2-VL](https://arxiv.org/abs/2409.12191)**，团队彻底意识到了初代 Qwen-VL“将所有图片强行压缩为固定 256 个 Token”的局限性，并推出了两项影响深远的架构革新：

#### 1. 原生动态分辨率（Naive Dynamic Resolution）与 $2 \times 2$ 邻域 Token 压缩
不同于 LLaVA-OneVision 把大图切成一个个 $384\times 384$ 的独立网格方块再拼接，Qwen2-VL 直接**移除了 ViT 中固定尺寸的绝对位置编码，引入 2D 旋转位置编码（2D-RoPE），让单个 ViT 视觉编码器原生直接接收任意长宽比、任意分辨率的完整图像**！
* **Token 数量随图片尺寸完全自适应伸缩**：
  视觉编码器以 $14 \times 14$ 为基本 Patch 单元；为了控制进入 LLM 的序列长度，在 ViT 输出之后通过一个简单的 MLP **将每相邻的 $2 \times 2 = 4$ 个视觉 Patch Token 压缩合并为 $1$ 个最终视觉 Token**（即每 $28 \times 28$ 像素区域对应 1 个 LLM 视觉 Token）。
  * 例如：一张极小的数学公式截图（如 $112 \times 56$ 像素）只会被编码为区区 **$8$ 个视觉 Token**；一张标准 $224 \times 224$ 图像（连同前后的 `<|vision_start|>` 和 `<|vision_end|>` 标记）生成 $\frac{224}{28} \times \frac{224}{28} + 2 = 64 + 2 = \mathbf{66}\text{ 个 Token}$；而一张超高分辨率的密集长海报则可以动态展开为多达 **$11,000+$ 个 Token**！
  * 对于**视频输入**，模型按每秒 2 帧（2 FPS）采样，通过 3D 卷积将每连续两帧压在一起，并将整段视频的总预算上限控制在 `16,384` 个 Token 以内。

#### 2. 多模态旋转位置编码（M-RoPE, Multimodal Rotary Position Embedding）
在纯文本模型中（回顾 Lecture 3），**1D-RoPE** 通过将特征向量每两个维度配对旋转角度 $m \theta_k$，使得两个 Token 的内积仅依赖于它们的一维相对距离 $m - n$。然而，**图像是二维的（高度 $h$、宽度 $w$），视频更是三维的（时间 $t$、高度 $h$、宽度 $w$）**！如果我们简单粗暴地把二维图像按行展平（Flatten）成一维序列再套用 1D-RoPE，那么在垂直方向上原本紧挨着的上下两个像素块，在一维序列里的位置索引就会相差整整一行（即相差 $W$ 个步长），破坏了二维空间邻近性！

为此，Qwen2-VL 提出了 **M-RoPE（3D 多模态旋转位置编码）**：为序列中的每一个 Token 都分配一个**三维位置坐标三元组 $\mathbf{p} = (t, h, w)$**，并将原本 $d$ 维的旋转位置编码向量按通道切分为三个独立的连续子块 $[d_t, d_h, d_w]$（其中 $d_t + d_h + d_w = d$），分别用时间坐标 $t$、高度坐标 $h$ 和宽度坐标 $w$ 计算对应的旋转矩阵，最后拼接起来：
$$\text{M-RoPE}(\mathbf{q}, \, (t, h, w)) = \Big[ \text{RoPE}_{d_t}(\mathbf{q}_{[1:d_t]}, \, t) \;\Big\|\; \text{RoPE}_{d_h}(\mathbf{q}_{[d_t+1:d_t+d_h]}, \, h) \;\Big\|\; \text{RoPE}_{d_w}(\mathbf{q}_{[d_t+d_h+1:d]}, \, w) \Big]$$

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│             Qwen2-VL 三维多模态旋转位置编码 (M-RoPE) 在不同模态下的坐标分配规则                 │
├─────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                 │
│  1. 【纯文本 Token (Text)】: 三维坐标全部取完全相同的递增标量!                                  │
│     • 第 m 个文本 Token 的坐标: (t, h, w) = (m, m, m)                                           │
│     • 数学性质: 当 t = h = w = m 时，三段拼接的 M-RoPE 在数学上 100% 退化还原为标准 1D-RoPE!    │
│                                                                                                 │
│  2. 【单张静态图像 Token (Image)】: 时间坐标 t 保持恒定，(h, w) 沿二维网格递增!                 │
│     • 设图像起始位置 ID 为 t_0，位于第 i 行、第 j 列的图像 Patch 坐标为: (t_0, t_0 + i, t_0 + j)│
│                                                                                                 │
│  3. 【视频帧序列 Token (Video)】: 帧间 t 递增，帧内 (h, w) 沿二维网格递增!                      │
│     • 第 k 帧、第 i 行、第 j 列的视频 Patch 坐标为: (t_0 + k, t_0 + i, t_0 + j)                 │
│                                                                                                 │
│  4. 【后续文本衔接】: 紧接在图像/视频之后的下一个文本 Token 起始 ID 取 max(t, h, w) + 1         │
└─────────────────────────────────────────────────────────────────────────────────────────────────┘
```

### 4.3 [Qwen3-VL (Bai et al., 2025)](https://arxiv.org/abs/2511.21631)：Interleaved M-RoPE、显式视频时间戳、$\sqrt{L}$ 损失归一化与 DeepStack 深层残差融合

来到本讲的重头戏之一——**[Qwen3-VL Technical Report (2025)](https://arxiv.org/abs/2511.21631)**（搭载强大的 `Qwen-3` Dense 与 MoE 语言基座，原生支持高达 **`256K` 超长上下文窗口**，采用与一代架构向后兼容的升级版 **[SigLIP 2](https://arxiv.org/abs/2502.14786)** 作为视觉编码器）。Percy 深入拆解了 Qwen3-VL 为何能在各项长视频、OCR、数学与智能体基准上全面叫板甚至局部超越闭源旗舰 `Gemini 2.5 Pro`、`GPT-5` 与 `Claude Opus 4.1` 的**四大核心架构与训练创新**（`00:52:56` – `00:59:20`）：

#### 创新 1：频率交错多模态旋转位置编码（Interleaved M-RoPE）——修复 Qwen2-VL 的轴向频率偏置！
为什么 Percy 在讲 Qwen2-VL 的 M-RoPE 时埋下伏笔说“直接把 $(t, h, w)$ 分成前、中、后三个连续块拼接其实是有缺陷的（Sub-optimal）”？
* **回顾 RoPE 的频率谱结构**：
  在旋转位置编码中，第 $k$ 组二维分量的旋转角频率为 $\theta_k = \text{base}^{-2k/d}$（例如 $\text{base} = 10,000$ 或 $1,000,000$）。这意味着：**向量最前面的维度（$k$ 小）对应极高频（波长极短，擅长捕捉局部紧邻相对位置），而向量最后面的维度（$k$ 大）对应极低频（波长极长，擅长捕捉超远距离依赖）**！
* **Qwen2-VL 分块拼接（Blocked M-RoPE）的致命缺陷**：
  在 Qwen2-VL 中，研究人员把前 $\frac{1}{3}$ 维度全部分配给时间 $t$，中间 $\frac{1}{3}$ 分配给高度 $h$，最后 $\frac{1}{3}$ 分配给宽度 $w$（或反之）。这导致了一个荒谬的后果：**时间轴 $t$ 独占了所有的超高频分量（缺乏低频长程建模能力），而宽度轴 $w$ 独占了所有的超低频分量（缺乏高频精细局部定位能力）**！三个物理维度在频率谱上严重偏科！
* **Qwen3-VL 的解决方案——交错排列（Interleaved M-RoPE）**：
  不再按连续大块切分，而是将频率从高到低按 **$[t, w, h, \; t, w, h, \; \dots, \; t, w, h]$ 的周期性交错模式（Interleaving）** 均匀分配给三个坐标轴！这样一来，无论是时间 $t$、宽度 $w$ 还是高度 $h$，**每一个坐标轴都能同时拥有完整的从最高频（局部细节）到最低频（256K 超长距离）的全频段表达能力**！

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│          Blocked M-RoPE (Qwen2-VL) vs. Interleaved M-RoPE (Qwen3-VL) 频率谱分配对比             │
├─────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                 │
│  维度索引 k:   [ k=0 (最高频 / 短波长)  ─────────────────────────►  k=d/2-1 (最低频 / 长波长) ] │
│                                                                                                 │
│  ❌ Qwen2-VL (分块拼接 Blocked):                                                                │
│                [  t ,  t ,  t ,  t  │  w ,  w ,  w ,  w  │  h ,  h ,  h ,  h  ]                 │
│                └──── 仅高频! ───────┴──── 仅中频! ───────┴──── 仅低频! ───────┘                 │
│                                                                                                 │
│  ✅ Qwen3-VL (交错排列 Interleaved):                                                            │
│                [  t ,  w ,  h ,  t  │  w ,  h ,  t ,  w  │  h ,  t ,  w ,  h  ]                 │
│                └──── 三个时空维度 (t, w, h) 均匀共享全部高频、中频与低频分量! ┘                 │
└─────────────────────────────────────────────────────────────────────────────────────────────────┘
```

#### 创新 2：显式文本时间戳 Token（Explicit Video Timestamps）
在 Qwen2-VL 中，视频每一帧在什么时间出现，仅仅隐式地编码在 M-RoPE 的时间坐标 $t$ 中。当用户在提问中问到：*“视频进行到 `01:25`（或第 2 秒后）时画面里发生了什么？”* 模型很难把自然语言里的数字 `"2 seconds"` 与内部抽象的旋转角度 $\theta \cdot t$ 直接精确对齐。
为此，**Qwen3-VL 在每个视频帧（或帧组）的视觉 Token 序列前面，直接插入了显式的文本时间戳 Token（如 `<0.0 seconds>`、`<0.5 seconds>`、`<1.0 seconds>`）**！这种看似简单的显式符号锚点，极大地提升了长视频时间定位（Temporal Grounding）与事件检索的准确度。

#### 创新 3：按序列长度平方根归一化的损失函数（$\sqrt{L}$-Normalized Per-Token Loss）
在混合了单轮短问答（几十个 Token）与长视频/长思维链复杂推理（数万个 Token）的多模态批次中，该如何对不同长度样本的损失加权？
* 如果按**样本级平均（Per-Example Mean，即除以 $L_i$）**：长视频样本里每个 Token 的梯度被稀释了几百倍，模型学不会长上下文；
* 如果按**全局 Token 级求和/平均（Per-Token Sum，不除以 $L_i$）**：一条长达 32K Token 的视频样本产生的梯度权重相当于几百条单图样本，长样本会彻底统治优化方向（我们在 Lecture 16 的 Dr. GRPO 中也讨论过类似的序列长度偏差）。
* **Qwen3-VL 的折中方案**：采用**按样本输出 Token 长度的平方根 $\frac{1}{\sqrt{L_i}}$ 进行归一化**，即单条样本的加权损失与 $\frac{1}{\sqrt{L_i}} \sum_{t=1}^{L_i} \ell_t = \sqrt{L_i} \cdot \bar{\ell}_i$ 成正比，既避免了长视频样本完全主导梯度，又防止了长样本中的有效监督信号被过度稀释。

#### 创新 4：[DeepStack (Meng et al., 2024)](https://arxiv.org/abs/2406.04334) —— 将多层视觉特征直接注入 LLM 残差流！
从 LLaVA 到 Qwen2-VL，所有的拼接式 VLM 都把视觉编码器当成一个“黑盒前处理模块”：不管 ViT 内部有多少层，只取其最后一层的输出经过 Adapter 塞进 LLM 的**第 0 层输入端（Input Layer）**。
**Qwen3-VL 引入了 DeepStack 架构**（源自 [Meng et al., 2024](https://arxiv.org/abs/2406.04334) 的思想）：
* 视觉编码器（ViT）本身就是一个多层 Transformer，它的浅层、中层和深层分别提取了从边缘纹理、局部几何到高层语义的多尺度特征。
* 与其只取 ViT 最后一层塞给 LLM 第 0 层，**Qwen3-VL 直接从 ViT 的多个中间层提取多级视觉特征序列，分别通过独立的轻量级 MLP Projector 映射后，直接按元素相加注入到 LLM 前面若干层 Transformer Block 的残差流（Residual Stream）中**！这实现了视觉编码器与语言模型之间无需增加额外序列长度的**深层多尺度特征融合（Deep Multi-level Fusion）**。

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│             传统浅层输入拼接 (LLaVA) vs. Qwen3-VL 的 DeepStack 多层残差流直注架构               │
├─────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                 │
│  [传统 VLM: 仅在 LLM 第 0 层输入端注入]       [Qwen3-VL DeepStack: 多层 ViT 特征直注 LLM 残差流]│
│                                                                                                 │
│     LLM Layer 3  ▲                               LLM Layer 3  ▲                                 │
│     LLM Layer 2  ▲                               LLM Layer 2  ▲ ◄──(+)── MLP_3 ◄── ViT 深层特征 │
│     LLM Layer 1  ▲                               LLM Layer 1  ▲ ◄──(+)── MLP_2 ◄── ViT 中层特征 │
│     LLM Layer 0  ▲ ◄── MLP ◄── ViT 末层特征      LLM Layer 0  ▲ ◄──(+)── MLP_1 ◄── ViT 浅层特征 │
└─────────────────────────────────────────────────────────────────────────────────────────────────┘
```

#### 5. Qwen3-VL 的七阶段工业级训练全管线（4-Stage Pre-training + 3-Stage Post-training）
Qwen3-VL 的训练管线展示了现代顶级多模态基础模型的工程复杂度：
* **预训练四阶段（Pre-training，上下文从 `8K` 渐进扩展至 `256K`）**：
  * **Stage S0（Modality Alignment）**：仅训练 MLP Projector（对齐视觉与语言空间），上下文长度 `8K`；
  * **Stage S1（Multimodal Pre-training）**：全参数联合训练，海量多模态图文与交错数据，上下文长度 `8K`；
  * **Stage S2（Long-Context Pre-training）**：全参数训练，混入长文档、多图与长视频数据，上下文扩展至 `32K`；
  * **Stage S3（Ultra-Long Context Adaptation）**：进一步将上下文窗口推向惊人的 **`256K` Tokens**（其中绝大多数训练 Token 消耗在 S1 和 S2 阶段）。
* **后训练三阶段（Post-training，全面复用我们在 Lecture 15 & 16 学过的后训练武器库）**：
  * **Stage 1（SFT）**：涵盖长思维链（Long CoT）冷启动数据与非思考模式通用指令微调；
  * **Stage 2（Strong-to-Weak Knowledge Distillation）**：利用更强的教师模型进行在线/离线知识蒸馏；
  * **Stage 3（Reinforcement Learning）**：结合可验证奖励强化学习（RLVR，针对视觉数学、STEM、代码与空间推理）与人类偏好强化学习（RLHF）。

> **课堂深度互动问答（Classroom Q&A on VLMs & Systems，`00:59:50` – `01:06:10`）**：
> * **学生提问 1（VLM 如何处理视频生成与逐 Token 监督）**：*“这些模型能生成视频或图像吗？在训练视频描述时，是用 LLM-as-a-Judge 打分还是逐 Token 监督？”*
>   **Percy 解答**：首先，到目前为止我们讲的 LLaVA 和 Qwen-VL 系列都属于**纯文本输出（Text-out）模型**——所有的多模态（图像、视频）都只在输入端（Input side），输出端永远是自回归生成的纯文本。因此，除了最后的 RL 阶段外，前面所有预训练和 SFT 阶段全都是标准的**逐 Token 交叉熵监督（Next-Token Prediction）**，不需要用裁判模型打分。
> * **学生提问 2（多模态训练的系统工程瓶颈 Systems Bottlenecks）**：*“从系统（Systems）角度看，训练多模态大模型比训练纯文本 LLM 难在哪里？”*
>   **Percy 解答**：绝对难得多！在纯文本 LLM 训练中（Lectures 5–8），文本 Token 极其紧凑，磁盘 I/O 和数据加载（Data Loading）的开销几乎可以忽略不计；但**一旦进入图像尤其是视频领域，原始视频文件的体积、解码（Video Decoding）、帧采样与动态裁剪预处理会消耗海量的磁盘带宽、网络带宽与 CPU/内存资源**！如果你不精心设计异步数据加载流水线（Asynchronous Data Loading & Preprocessing），GPU 就会大面积空转等待视频解码。此外，由于单条视频会展开成上万个 Token，而单张小图只有几十个 Token，不同模态样本长度的极端异构性也会给分布式负载均衡（Load Balancing）与数据配比（Data Mixture Weighting）带来严峻挑战。
> * **学生提问 3（为什么视觉编码器的参数量远小于语言模型？）**：*“为什么在一个 72B 的视觉语言模型（如 LLaVA-OneVision-72B 或 Qwen-VL-72B）里，语言模型有 720 亿参数，而视觉编码器（ViT）往往只有不到 10 亿参数（如 400M ~ 600M），Projector 更是只有几千万（如 72M）参数？”*
>   **Percy 解答**：这是一个非常深刻的架构分工规律！因为**视觉编码器（ViT）干的活本质上是一种相对局部的感知转换（Local Perceptual Operation）**——它只是盯着一个个 $14 \times 14$ 的小像素块，把边缘、颜色、局部纹理和基础物体特征提取出来，这个过程并不需要存储海量的百科全书式世界知识，也不需要做深层逻辑推理；而**几乎所有的世界知识存储、跨模态概念关联与多步逻辑推理，全部是由后面的大语言模型（LLM）完成的**！因此，“小而精的视觉编码器（$<1\text{B}$）+ 轻量级适配器（$<100\text{M}$）+ 巨型语言模型基座（$7\text{B}\sim 72\text{B}+$）”成为了最符合算力效益比黄金分割的配比。

## 5. 早期融合与离散视觉 Token 探索（Part IV）：[VQ-VAE](https://arxiv.org/abs/1711.00937) 与 [Meta Chameleon](https://arxiv.org/abs/2405.09818)

回顾前面的 LLaVA 和 Qwen-VL，它们走的都是**“连续视觉嵌入 + 预训练 LLM 拼接”**的路线。这条路线虽然在视觉理解（Image/Video-to-Text）上所向披靡，但它有一个天生无法逾越的遗憾：**它只能输出文本，不能直接输出图像**！而且从美学角度看，把一个外部 ViT 硬塞进 LLM 多少显得有些“打补丁”的味道。

能不能走一条彻底统一的极简主义路线：**用一套离散视觉分词器（Discrete Image Tokenizer）把图像也像文本一样切分成来自固定词表的离散整数 ID（Discrete Tokens），然后从零开始训练一个完全同构的、既能看图说话又能文生图、甚至能边写文字边画插图的单一自回归 Transformer**？

这就是 Meta FAIR 团队在 **[Chameleon: Mixed-Modal Early-Fusion Foundation Models (Team, 2024)](https://arxiv.org/abs/2405.09818)** 中探索的**早期融合（Early-Fusion）离散全模态路线**，而支撑图像离散化的核心算法则是深度学习经典之作——**[VQ-VAE (Vector Quantised-Variational AutoEncoder, van den Oord et al., 2017)](https://arxiv.org/abs/1711.00937)**。

### 5.1 [VQ-VAE (van den Oord et al., 2017)](https://arxiv.org/abs/1711.00937)：如何把连续图像量化为离散词表 ID？

在传统的自编码器（Autoencoder）或变分自编码器（VAE）中，编码器将输入图像 $x$ 映射为连续潜变量 $z \in \mathbb{R}^D$。为了得到像文本 Token 一样的**离散整数索引 $k \in \{1, 2, \dots, K\}$**，VQ-VAE 引入了一个包含 $K$ 个原型向量的**可学习离散码本（Codebook / Embedding Space）** $\mathbf{E} = \{e_1, e_2, \dots, e_K\} \in \mathbb{R}^{K \times D}$（例如码本大小 $K = 8192$）：

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│                VQ-VAE 向量量化变分自编码器原理与直通梯度流 (van den Oord et al., 2017)          │
├─────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                 │
│                                      [离散码本 Codebook E]                                      │
│                                      e_1, e_2, ..., e_K (如 K = 8192)                           │
│                                               │                                                 │
│                                               ▼ (最近邻查找 argmin_j ||z_e(x) - e_j||_2)        │
│   输入图像 x ──► [Encoder CNN/ViT] ──► 连续向量 z_e(x) ──► 量化离散整数索引 q(z=k|x)            │
│   (512×512×3)                          (32×32×D 网格)            │                              │
│                                               ▲                  ▼ (查表取出最近邻向量 e_k)     │
│                                               │            量化向量 z_q(x) = e_k                │
│                                               │                  │                              │
│                                        [直通梯度复制]            ▼                              │
│                                        (Straight-Through)  [Decoder CNN/ViT] ──► 重建图像 x_hat │
└─────────────────────────────────────────────────────────────────────────────────────────────────┘
```

#### 1. 前向最近邻量化（Forward Vector Quantization）
给定输入图像 $x$，编码器 $z_e(x)$ 输出一个二维网格的连续特征向量（例如对于 $512 \times 512$ 的图像，下采样 16 倍后得到 $32 \times 32 = 1024$ 个连续特征向量）。对于网格上的每一个连续向量 $z_e(x) \in \mathbb{R}^D$，我们在码本 $\mathbf{E}$ 中寻找与它欧氏距离最近的码字索引 $k$：
$$k = \arg\min_{j \in \{1, \dots, K\}} \| z_e(x) - e_j \|_2$$
随后将连续向量 $z_e(x)$ **硬性舍入（Round / Quantize）** 为该最近邻码字向量 $z_q(x) = e_k$，并送入解码器重构图像 $\hat{x} = \text{Decoder}(z_q(x))$。这样一来，整张 $512 \times 512$ 的图像就被无损映射为了 **$32 \times 32 = 1024$ 个取值在 $\{1, \dots, 8192\}$ 之间的离散整数 Token**！

#### 2. 反向传播难题：不可导的 $\arg\min$ 与直通梯度估计器（Straight-Through Estimator）
由于最近邻查找算子 $\arg\min_j \|z_e(x) - e_j\|_2$ 是阶跃离散函数，其导数处处为 0 或不存在，解码器传回来的重构损失梯度 $\nabla_{z_q} \mathcal{L}$ 无法直接穿过量化层传给编码器 $z_e(x)$，也无法直接更新码本向量 $e_k$。VQ-VAE 通过**直通梯度估计器（Straight-Through Estimator, STE）**与**三项联合损失函数**巧妙破解了这一难题：
$$\mathcal{L}_{\text{VQ-VAE}}(x) = \underbrace{\log p\big(x \mid z_q(x)\big)}_{\text{1. 图像重构损失 (Reconstruction)}} + \underbrace{\big\| \text{sg}[z_e(x)] - e_k \big\|_2^2}_{\text{2. 码本更新损失 (Codebook Loss)}} + \beta \underbrace{\big\| z_e(x) - \text{sg}[e_k] \big\|_2^2}_{\text{3. 承诺损失 (Commitment Loss)}}$$
* **算子 $\text{sg}[\cdot]$（Stop-Gradient 停止梯度算子）**：在前向传播中恒等于其输入（恒等映射），但在反向传播中偏导数强制为 $0$（将内部变量视为常数）。
* **在 PyTorch 代码中如何用一行公式实现直通梯度（STE）？**
  在前向传播时构造：
  ```python
  # z_q 在前向传播时数值严格等于 e_k，但在反向传播时梯度直接复制给 z_e!
  z_q = z_e + (e_k - z_e).detach()
  ```
* **三项损失的各司其职**：
  1. **重构损失** $-\log p(x \mid z_q(x))$（如像素 $L_2$ 误差与感知损失）：优化解码器，并通过直通估计器（STE）将 $\nabla_{z_q} \mathcal{L}$ 原封不动地复制给编码器输出 $z_e(x)$；
  2. **码本损失** $\|\text{sg}[z_e(x)] - e_k\|_2^2$：固定编码器输出 $z_e(x)$，驱动被选中的码本向量 $e_k$ 向编码器特征中心靠拢（等价于在线 $K$-Means 聚类中心更新）；
  3. **承诺损失（Commitment Loss）** $\beta \|z_e(x) - \text{sg}[e_k]\|_2^2$：由于码本空间没有维度约束，如果只让码本去追编码器，编码器输出 $z_e(x)$ 可能会发散乱跑或频繁在两个码字之间横跳。加入承诺损失（固定码本 $e_k$，惩罚 $z_e(x)$ 偏离码本）强迫编码器输出“信守承诺”紧贴某个码字！

### 5.2 [Meta Chameleon (Team, 2024)](https://arxiv.org/abs/2405.09818)：万物皆离散 Token 的优雅愿景与残酷的训练稳定性危机

基于改进版的图像离散分词器（将任意 $512 \times 512$ 图像编码为 **$1024$ 个来自 $8192$ 视觉词表的离散 Token**），Meta 团队训练了 **[Chameleon](https://arxiv.org/abs/2405.09818)**（包含 `7B` 与 `34B` 规模）：
* **统一的联合词表（Unified Vocabulary，大小 `65,536`）**：
  构建一个全新的 BPE 分词器，将 **$8,192$ 个图像离散码本 Token** 与 **文本 BPE Token** 合并在同一个大小为 `65,536` 的共享词表中！
* **零适配器、零外部编码器的纯粹 Transformer 训练**：
  在 Transformer 主干内部，**没有任何 Adapter、没有任何 ViT 分支**——无论是纯文本、图文问答，还是交错的“一段文字 + 一张生成的鸟类图片 + 又一段文字 + 又一张图片”，全都只是一串取值在 $[0, 65535]$ 之间的整数 ID 序列！模型从零开始（From Scratch）在两阶段海量混合多模态语料（Stage 1 约 4.4T Tokens 交错图文与纯文本；Stage 2 混入高质量指令与长上下文数据）上跑最标准的**下一 Token 预测（Next-Token Prediction）**！

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│             Meta Chameleon 早期融合全模态自回归序列与“图文熵差”引发的范数爆炸危机               │
├─────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                 │
│  [1. Chameleon 统一自回归序列 (Everything is a Discrete Token ID in [0, 65535])]                │
│                                                                                                 │
│    输入/输出序列:  "Show me birds:"  <img_start>  [v_412] [v_89] ... [v_7011]  <img_end> "Nice!"│
│                    └── 文本 Tokens ──┘            └─── 1024 个离散图像 Tokens ──┘               │
│                                   │                              │                              │
│                                   ▼                              ▼                              │
│                    [单一共享 Transformer (无 ViT, 无 Adapter, 统一做 Next-Token Prediction)]    │
│                                                                                                 │
│  [2. 为什么混合训练会在中后期突然崩溃? —— 文本与图像离散 Token 的本质信息论差异]                │
│    • 文本 Token: 局部语义约束极强，下一词分布【熵极低 (Low Entropy)】，Logits 幅度受控          │
│    • 图像 Token: 局部背景像素块随机性大，下一码字分布【熵极高 (High Entropy / Flat)】           │
│    • 致命冲突 : 共享 Softmax 头与注意力层同时争夺“尖锐文本分布”与“平坦图像分布”，导致隐状态与   │
│                 Query/Key 激活范数 (Norms) 随步数单调失控飙升，最终引发 BF16 数值溢出崩溃!      │
│    • 急救处方 : 必须引入 【QK-Norm (注意力 Q/K 层归一化)】 + 【Z-Loss (Logits 分母平方正则)】!  │
└─────────────────────────────────────────────────────────────────────────────────────────────────┘
```

#### 深度剖析：为什么“把图像叫做离散 Token”并不能掩盖它与文本的物理鸿沟？（`01:12:08` – `01:13:20`）
Percy 指出，尽管 Chameleon 在概念上极其优雅（作为一名语言学者，Percy 坦言自己最初也觉得“万物皆离散文本 Token”在美学上令人着迷），但在实际训练中，团队遭遇了极其严重的**训练不稳定性与中后期 Loss 突刺崩盘（Training Divergence）**！
* **病根剖析（Entropy & Norm Growth Mismatch）**：
  虽然图像块和文本词在表面上都被编码成了 $[0, 65535]$ 里的一个整数 ID，但**它们的底层概率分布特性（特别是条件熵 Conditional Entropy）截然不同**！
  1. 在自然语言中，给定前文 `"The capital of France is"`，下一个文本 Token 的条件分布极其尖锐（**低熵 Low Entropy**），几乎确定的就是 `"Paris"`；
  2. 而在图像中，给定前一个蓝天背景像素块 Token，下一个相邻蓝天块究竟对应 8192 个码字中哪一种细微的蓝色阴影，具有极高的不确定性（**高熵 High Entropy**，分布非常平坦）！
  3. 当不同模态共享同一个 Transformer 残差流、注意力矩阵和输出 Softmax 层时，不同模态之间为了调整各自的分布尖锐度（有效温度），会驱动 Transformer 内部的权重范数、激活值范数（Activation Norms）以及注意力 Query/Key 向量范数不可逆地单调膨胀（Norm Growth），最终在训练进行到数千亿 Token 后突破 `bfloat16` 的表示上限而彻底发散崩溃！
* **两大架构级稳定化处方（Stabilization via QK-Norm & Z-Loss）**：
  为了压制跨模态混合训练引发的范数失控，Chameleon 被迫在标准 LLaMA 架构上动了两处关键手术（回顾我们在 Lecture 3 & Lecture 9 中见过的稳定化技巧）：
  1. **QK-Norm（Query-Key LayerNorm）**：在计算自注意力内积 $\mathbf{q}_i^\top \mathbf{k}_j$ 之前，对 Query 和 Key 向量显式施加 LayerNorm（控制注意力 Logits 的上限，防止注意力熵坍缩与范数传递增长）；
  2. **Dropout 与 Z-Loss 正则化（Logit Norm Regularization）**：设最后一层输出的未归一化 Logits 向量为 $\mathbf{z}$，标准的交叉熵损失为 $-\log \frac{\exp(z_y)}{\sum_j \exp(z_j)}$，这允许所有 $z_j$ 同时加上一个巨大的常数而不改变概率值（Translational Invariance），从而导致输出层范数无限漂移。为此，在损失函数中显式加入 **Z-Loss 正则项**，直接惩罚配分函数对数 $Z = \log \sum_j \exp(z_j)$ 的平方：
     $$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{CE}} + \alpha_{\text{z}} \cdot \Big( \log \sum_{j=1}^{V} \exp(z_j) \Big)^2 \qquad (\text{例如 } \alpha_{\text{z}} = 10^{-5})$$
     强行将输出层 Logits 的幅度锚定在 $0$ 附近！

#### 为什么离散视觉 Token（VQ-VAE + Chameleon）路线如今不再是主流理解范式？
Percy 在讲座尾声坦诚地评价道（`01:13:30` – `01:16:50`），尽管 Chameleon 验证了单一离散自回归模型同时做图文理解与生成的可行性，但这条路线在实践中暴露出了难以弥补的硬伤：
1. **有损量化导致细粒度信息永久丢失（Information Bottleneck in Discretization）**：
   把一张 $512 \times 512$ 的高清图片强行通过 $8192$ 个固定码字压缩成 $1024$ 个整数 ID，本质上是一次**极其激进的有损压缩（Lossy Quantization）**。如果图片里有一行小字（OCR）或者密集的图表数字，一旦经过 VQ-VAE 的最近邻码本舍入，细微的笔画差异在进入 Transformer 之前就已经被抹平销毁了——无论后面的 LLM 有多聪明，也无法复原已经在分词阶段丢失的视觉信息！
2. **理解（Understanding）与生成（Generation）对视觉表征的根本诉求不对称（Fundamental Asymmetry）**：
   * **视觉理解（输入端）** 需要的是**连续、无损且高度语义化的表征**（对于分类任务只需高层语义，对于 OCR 则需要保留连续高频细节），在这方面，**连续视觉编码器（如 CLIP / SigLIP + 动态高分辨率 AnyRes / M-RoPE）** 拥有压倒性的信息保真度与性能优势；
   * **视觉生成（输出端）** 需要的是能够在连续像素/潜空间中精细微调高频细节纹理的生成机制，在这方面，**扩散模型（Diffusion Models / Flow Matching）** 远比离散自回归 VQ-VAE 更强大、更清晰。
   * 因此，当今工业界最顶尖的原生全模态模型（Omnimodels），在底层大概率采用的都是**“前端用连续视觉编码器（Continuous Encoder）保证无损深度理解 + 后端挂接扩散/流匹配生成头（Diffusion Head）负责高保真图像/视频生成”**的混合协同架构！

## 6. 多模态大模型主流技术路线全景对比与核心总结

| 架构范式 / 代表模型 | 视觉编码器 (Vision Encoder) | 跨模态连接器 (Adapter / Fusion) | 分辨率与位置编码机制 | 支持模态方向 | 核心优势与局限性 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **[LLaVA-1.0 (2023)](https://arxiv.org/abs/2304.08485)** | 冻结的 [CLIP ViT-L/14](https://arxiv.org/abs/2103.00020) | 单层线性投影矩阵 $\mathbf{W}$ | 固定缩放裁剪至 $224^2$ 或 $336^2$；1D 绝对位置编码 | `(Image, Text) -> Text` | **极简高效**：开创开源 VLM 黄金两阶段范式与 GPT-4 合成视觉指令法；但固定低分辨率无法看清密集文档/OCR |
| **[LLaVA-OneVision (2024)](https://arxiv.org/abs/2408.03326)** | [SigLIP](https://arxiv.org/abs/2303.15343) (`384×384`) | 2 层 MLP | **AnyRes 网格切片**（高清局部块 + 全局缩略图）+ 跨模态 Token 预算均衡 | `(Single/Multi-Image, Video, Text) -> Text` | **全开源标杆**：解决了高分辨率 OCR 痛点，展现出从单图到多图/视频的惊艳零样本跨模态迁移 |
| **[Qwen-VL (2023)](https://arxiv.org/abs/2308.12966)** | 微调的 OpenCLIP ViT-bigG | 单层 Cross-Attention（压缩为固定 256 Tokens） | 固定 $224^2 \to 448^2$；融入 2D 位置编码与 `<ref>`/`<box>` 标签 | `(Image, Text) -> Text + Bounding Boxes` | 原生支持细粒度边界框定位输出；但固定压缩至 256 个 Token 限制了极端高分辨率与长视频扩展 |
| **[Qwen2-VL (2024)](https://arxiv.org/abs/2409.12191)** | 原生动态分辨率 ViT (含 2D-RoPE) | $2 \times 2$ 邻域 Token 合并 MLP | **原生动态分辨率**（8 至 1万+ Tokens）+ **分块 3D M-RoPE $(t, h, w)$** | `(Image, Multi-Image, Video, Text) -> Text` | 彻底告别网格拼图，单一 ViT 自适应任意分辨率；但分块 M-RoPE 存在三个坐标轴高低频分配失衡 |
| **[Qwen3-VL (2025)](https://arxiv.org/abs/2511.21631)** | [SigLIP 2](https://arxiv.org/abs/2502.14786) (动态分辨率) | **[DeepStack](https://arxiv.org/abs/2406.04334)**（多层 ViT 特征直注 LLM 残差流） | **Interleaved M-RoPE（频率交错）** + **显式视频时间戳 Token** + `256K` 上下文 | `(Image, Multi-Image, Long Video, Text) -> Text` | **当前开源 VLM 巅峰**：修复频率偏置，加上 $\sqrt{L}$ 损失归一化与 SFT/Distill/RL 后训练，逼近甚至局部超越闭源旗舰 |
| **[Meta Chameleon (2024)](https://arxiv.org/abs/2405.09818)** | **[VQ-VAE](https://arxiv.org/abs/1711.00937) 离散分词器**（码本 8192） | **无 Adapter**（早期融合，共享 65,536 词表） | $512^2$ 图像映射为固定 1024 个离散整数 Token | **`(Image, Text) -> (Image, Text)` 双向交错生成** | **概念最统一的原生 Omnimodel**；但离散量化损失细粒度 OCR 信息，且图文熵差引发严重训练不稳定（需 QK-Norm + Z-Loss） |

> **Percy Liang 的本讲五大核心总结（Final Reflections，`01:14:45`）**：
> 1. **多模态的本质挑战在于“非文本模态的 Token 化”**：Transformer 是处理语义 Token 序列的通用推理引擎，多模态建模的第一要务是将高维原始信号转化为保留恰当粒度信息的 Token。
> 2. **理解与生成的不对称性（Asymmetry between Understanding & Generation）**：不存在一个“放之四海而皆准”的单一视觉编码器——粗粒度分类只需要 CLIP 式的高层语义浓缩，OCR 与文档理解需要 AnyRes / 动态分辨率来保留高频局部细节，而高质量视觉生成则更适合交给扩散模型（Diffusion Models）。
> 3. **连续视觉编码器依然是视觉理解的绝对统治者**：历经五年演进，基于语言对比预训练的连续编码器（从 **CLIP** 到 **SigLIP / SigLIP 2**）配合轻量级投影层与多层融合（如 **DeepStack**），依然是让大模型获得顶级视觉理解能力的最优路径。
> 4. **长上下文与模态权重平衡是系统成败的关键**：高分辨率图像与长视频会迅速膨胀为数万乃至数十万个视觉 Token。由于视频的信息密度远低于纯文本，必须通过动态 Token 压缩、长上下文位置编码（如 **Interleaved M-RoPE**）以及样本长度归一化（如 **$\sqrt{L}$ Loss Normalization**）来防止低信息密度的长视频淹没语言主干。

## 7. 课后思考与自测练习（Review & Self-Check）

1. **视觉 Patch 序列长度、AnyRes 切分与 Qwen2-VL 动态压缩定量计算**：
   假设我们有一张分辨率为 $W \times H = 1152 \times 768$ 的高分辨率财务报表图像。
   * (a) 如果直接将该图像送入原版 **[CLIP ViT-L/14 @ 336px](https://arxiv.org/abs/2103.00020)**：请写出 CLIP 预处理（短边缩放 + 居中裁剪）后实际保留的图像区域对应原始图像的像素范围，并计算送入 ViT 的 Patch Token 总数 $N_{\text{CLIP}}$（不含 `[CLS]` Token）。在这个过程中损失了多少比例的原始图像面积與分辨率？
   * (b) 如果将该图像送入采用 **AnyRes** 策略的 **[LLaVA-OneVision](https://arxiv.org/abs/2408.03326)**（底层视觉编码器为 `SigLIP-384`，输入分辨率 $384 \times 384$，Patch 大小 $P = 14$，每个 $384 \times 384$ 图像块产出 $\lfloor 384/14 \rfloor^2 = 27^2 = 729$ 个视觉 Token）：该 $1152 \times 768$ 图像可以被无损切分为多少个 $384 \times 384$ 的局部高清切片？加上 $1$ 张全局下采样缩略图（Global Thumbnail）后，送入语言模型的视觉 Token 总数 $N_{\text{AnyRes}}$ 是多少？
   * (c) 如果将该图像送入采用**原生动态分辨率 + $2 \times 2$ 邻域合并**的 **[Qwen2-VL](https://arxiv.org/abs/2409.12191)**（假设图像长宽微调对齐至 28 的整数倍 $1148 \times 756$，ViT Patch 大小 $P = 14$，MLP 将每 $2 \times 2$ 个相邻 Patch 合并为 $1$ 个 LLM Token，并在首尾各加 1 个特殊标记 `<|vision_start|>` 和 `<|vision_end|>`）：请计算首先在 ViT 内部产生的 Patch 数量，以及最终送入 Qwen2 LLM 的视觉 Token 总数 $N_{\text{Qwen2}}$！
   * *参考解答*：
     * (a) **CLIP 预处理与面积损失**：
       原始图像 $1152 \times 768$ 的短边为 $H = 768$。CLIP 首先将短边从 $768$ 缩放到 $336$（缩放比例为 $336 / 768 = 0.4375$，长边变为 $1152 \times 0.4375 = 504$），随后居中裁剪出 $336 \times 336$ 的正方形。
       * 裁剪出的正方形对应原始图像中间 $768 \times 768$ 的区域，左右两侧共 $(1152 - 768) / 1152 = \mathbf{33.3\%}$ 的报表内容被直接裁掉丢弃！
       * 在保留的中心区域内，像素总数从 $768^2 = 589,824$ 压缩至 $336^2 = 112,896$，仅保留了 $0.4375^2 = \mathbf{19.14\%}$ 的像素密度！
       * ViT-L/14 的 Patch Token 总数为 $N_{\text{CLIP}} = (336 / 14) \times (336 / 14) = 24 \times 24 = \mathbf{576}\text{ 个 Token}$。
     * (b) **LLaVA-OneVision AnyRes 计算**：
       由于 $1152 / 384 = 3$ 且 $768 / 384 = 2$，该图像恰好可以无损切分为 **$3 \times 2 = 6$ 个** $384 \times 384$ 的局部高清切片（保留 $100\%$ 原始像素清晰度，零裁剪丢失）。
       加上 $1$ 张整图下采样的 $384 \times 384$ 全局缩略图，共有 $6 + 1 = 7$ 个子图送入 SigLIP，总视觉 Token 数为：
       $$N_{\text{AnyRes}} = (6 + 1) \times 729 = \mathbf{5,103}\text{ 个 Token}$$
     * (c) **Qwen2-VL 原生动态分辨率计算**：
       对于 $1148 \times 756$ 的输入图像：
       * 在 ViT 内部（$P = 14$），网格尺寸为 $(1148 / 14) \times (756 / 14) = 82 \times 54 = \mathbf{4,428}\text{ 个 ViT Patch}$；
       * 经过 $2 \times 2$ 邻域合并（相当于等效 Patch 步长为 $28 \times 28$）后，网格压缩为 $41 \times 27 = 1,107$ 个合并 Token，加上首尾 2 个视觉边界特殊 Token，最终进入 LLM 的 Token 总数为：
         $$N_{\text{Qwen2}} = 41 \times 27 + 2 = \mathbf{1,109}\text{ 个 Token}$$

2. **CLIP 全局 Softmax 损失 vs. SigLIP 逐对 Sigmoid 损失的数学与分布式系统对比推导**：
   设全局 Batch Size 为 $B$，均匀切分在 $D$ 台加速器设备上（每台设备本地持有 $b = B / D$ 个图文对）。令第 $i$ 张图像的归一化嵌入为 $I_i \in \mathbb{R}^d$，第 $j$ 条文本的归一化嵌入为 $T_j \in \mathbb{R}^d$，且记 $x_{ij} = t \langle I_i, T_j \rangle + b_0$。
   * (a) 请写出 **[SigLIP](https://arxiv.org/abs/2303.15343)** 在单个图文对 $(i, j)$ 上的二元逻辑回归损失 $\ell_{ij}(x_{ij}) = \log\big(1 + \exp(-z_{ij} x_{ij})\big)$（其中 $z_{ij} \in \{+1, -1\}$）关于缩放内积 $x_{ij}$ 的一阶偏导数 $\frac{\partial \ell_{ij}}{\partial x_{ij}}$，并说明为什么在训练初始阶段（当 $\langle I_i, T_j \rangle \approx 0$ 时）必须将偏置 $b_0$ 初始化为较大的负数（如 $b_0 = -10$）而不是 $0$？
   * (b) 在分布式训练中，利用 SigLIP 的 **Ring 环形分块算法** 计算完整 $B \times B$ 矩阵的总损失与梯度时，需要执行多少轮（Steps）设备间通信？在每一轮中，单台设备需要在显存中实例化的相似度子矩阵大小是多少？相比于在单卡上实例化完整 $B \times B$ 矩阵，峰值矩阵显存降低了多少倍？
   * *参考解答*：
     * (a) **偏导数推导与初始偏置分析**：
       对 $\ell_{ij}(x_{ij}) = \log(1 + e^{-z_{ij} x_{ij}})$ 求导得：
       $$\frac{\partial \ell_{ij}}{\partial x_{ij}} = \frac{-z_{ij} e^{-z_{ij} x_{ij}}}{1 + e^{-z_{ij} x_{ij}}} = -z_{ij} \cdot \sigma(-z_{ij} x_{ij})$$
       在训练刚开始时，由于随机初始化的图像与文本嵌入几乎正交，$\langle I_i, T_j \rangle \approx 0$，故 $x_{ij} \approx b_0$：
       * 如果设 $b_0 = 0$，则对每一个负样本对（$z_{ij} = -1$，共有 $B(B-1)$ 个），其梯度幅值为 $|\frac{\partial \ell_{ij}}{\partial x_{ij}}| = \sigma(0) = 0.5$；而正样本对（$z_{ij} = +1$，仅有 $B$ 个）的梯度幅值也是 $0.5$。由于负样本对的数量是正样本对的 $(B-1)$ 倍（当 $B=32,768$ 时高达 $32,767$ 倍！），负样本产生的总梯度将比正样本大四个数量级，导致第一步更新就发生剧烈振荡！
       * 当初始化 $b_0 = -10$ 时，对于负样本对（$z_{ij} = -1$），$-z_{ij} x_{ij} \approx -10$，单对负样本梯度幅值骤降为 $\sigma(-10) = \frac{1}{1 + e^{10}} \approx 4.54 \times 10^{-5}$，恰好抵消了 $O(B)$ 的负样本数量倍数，使正负样本的总梯度在起步阶段处于同一数量级！
     * (b) **Ring 环形分块系统复杂度**：
       共有 $D$ 台设备，第 1 步计算本地对角块 $(I^{(d)}, T^{(d)})$，随后通过 $D - 1$ 次环形移位（Collective Permute）依次接收其余 $D - 1$ 台设备的文本块 $T^{(d')}$（若还需要将文本侧累积的梯度传回原设备则共需 $D$ 步环路传递）。
       在任意时刻，单台设备只需在显存中计算并保存一个 **$b \times b = \left(\frac{B}{D}\right) \times \left(\frac{B}{D}\right)$** 的子矩阵即可立即求导并释放该块，因此单卡峰值相似度矩阵显存从 $B^2$ 降至 $(B/D)^2$，**显存开销降低了整整 $D^2$ 倍**（例如当 $D = 32$ 时，显存开销降低了 **$1,024$ 倍**）！

3. **Qwen3-VL 的 Interleaved M-RoPE 频率分析与 Chameleon 的训练稳定化机制**：
   * (a) 假设某多模态模型的注意力 Head 维度为 $d_{\text{head}} = 12$（共 $6$ 个二维旋转频率对 $\{\theta_0, \theta_1, \theta_2, \theta_3, \theta_4, \theta_5\}$，其中 $\theta_k = 10000^{-2k/12}$ 随 $k$ 增大而从高频递减至低频）。请写出在 **[Qwen2-VL](https://arxiv.org/abs/2409.12191) 的分块 M-RoPE（Blocked M-RoPE）** 与 **[Qwen3-VL](https://arxiv.org/abs/2511.21631) 的交错 M-RoPE（Interleaved M-RoPE）** 下，时间 $t$、宽度 $w$、高度 $h$ 三个坐标轴各自获得的频率索引集合，并解释为什么分块方案在 `256K` 超长视频理解中会出现问题。
   * (b) 在 **[Meta Chameleon](https://arxiv.org/abs/2405.09818)** 中，为什么将图像离散 Token（由 VQ-VAE 生成）与文本 Token 混合在同一个 Transformer 中做自回归训练会引发严重的训练不稳定？请写出 **Z-Loss** 的数学表达式，并推导它对第 $k$ 个词表 Logit $z_k$ 的梯度 $\frac{\partial \mathcal{L}_{\text{z}}}{\partial z_k}$ 是如何抑制 Logits 范数漂移的。
   * *参考解答*：
     * (a) **Blocked vs. Interleaved M-RoPE 频率分配**：
       * 在 **Qwen2-VL (Blocked M-RoPE)** 中，前中后三个连续块各分得 2 个频率对：
         * 时间轴 $t$ 仅获得 $\{\theta_0, \theta_1\}$（全为最高频/短波长分量）；
         * 宽度轴 $w$ 仅获得 $\{\theta_2, \theta_3\}$（全为中频分量）；
         * 高度轴 $h$ 仅获得 $\{\theta_4, \theta_5\}$（全为最低频/长波长分量）。
         当处理长达 `256K` Token 的长视频时，时间轴 $t$ 的跨度极大，但由于 $t$ 没有分配到任何低频（长波长）分量 $\theta_4, \theta_5$，高频旋转角 $t \cdot \theta_0$ 早已绕圆周旋转了无数圈发生相位混叠，导致模型无法建模长程跨帧时序依赖；同时高度轴 $h$ 因缺乏高频分量 $\theta_0, \theta_1$ 而丧失了精细的垂直局部定位能力。
       * 在 **Qwen3-VL (Interleaved M-RoPE)** 中，按 $[t, w, h, t, w, h]$ 交错分配：
         * 时间轴 $t$ 获得 $\{\theta_0, \theta_3\}$；
         * 宽度轴 $w$ 获得 $\{\theta_1, \theta_4\}$；
         * 高度轴 $h$ 获得 $\{\theta_2, \theta_5\}$。
         三个物理维度全都同时覆盖了高频与低频谱段，兼顾了局部精细感知与 256K 长程时空建模。
     * (b) **Chameleon 混合模态不稳定性与 Z-Loss 梯度推导**：
       图像离散 Token 的局部条件分布具有极高的信息熵（High Entropy，相邻像素块存在多种合理的码字选择），而文本 Token 的条件分布熵要低得多（Low Entropy）。当两者共享同一套自注意力与输出投影权重时，模型为了同时拟合截然不同的分布尖锐度，会不断推高隐状态与 Logits 的范数（Norm Growth），最终导致数值溢出。
       定义 **Z-Loss** 为 $\mathcal{L}_{\text{z}}(\mathbf{z}) = \alpha_{\text{z}} \big(\log Z\big)^2$，其中配分函数 $Z = \sum_{j=1}^V \exp(z_j)$。对第 $k$ 个 Logit $z_k$ 求偏导：
       $$\frac{\partial \mathcal{L}_{\text{z}}}{\partial z_k} = 2 \alpha_{\text{z}} (\log Z) \cdot \frac{\partial (\log \sum_{j=1}^V e^{z_j})}{\partial z_k} = 2 \alpha_{\text{z}} (\log Z) \cdot \frac{e^{z_k}}{\sum_{j=1}^V e^{z_j}} = 2 \alpha_{\text{z}} (\log Z) \cdot p_k$$
       可以看到，当输出层 Logits 整体正向漂移变大使得 $\log Z > 0$ 时，Z-Loss 会对每个分量 $z_k$ 施加一个与其预测概率 $p_k$ 成正比的正向恢复梯度，将 $\log Z$ 强行拉回 $0$ 附近，从而在不破坏预测概率排序的前提下消除了 Softmax 的平移不变性漂移！
