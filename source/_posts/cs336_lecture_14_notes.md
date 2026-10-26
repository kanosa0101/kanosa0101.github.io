---
title: Stanford CS336 (Spring 2026) Lecture 14 深度课程笔记：Data (Part 2)
date: 2026-10-26 14:31:00
tags:
  - 大模型
  - 数据清洗
  - MinHash
  - LSH去重
  - 数据混合
  - 合成数据
  - CS336
categories:
  - [笔记, CS336]
description: 深入剖析 Stanford CS336 Lecture 14：数据工程深水区实战，涵盖文本启发式过滤、FastText 质量分类、MinHash LSH 局部敏感哈希海量去重数学原理、数据配比（Data Mixing）定律及后训练合成数据构造机制。
mathjax: true
---

# Stanford CS336 (Spring 2026) Lecture 14 深度课程笔记：Data (Part 2)

> **课程信息**：[Stanford CS336: Language Modeling from Scratch (Spring 2026)](https://cs336.stanford.edu/)
> **主讲教授**：Percy Liang
> **本讲资源**：[Lecture 14 YouTube 视频](https://www.youtube.com/watch?v=5sxHosTLPF8) | [交互式执行轨迹 (`lecture_14.py`)](https://cs336.stanford.edu/lectures/?trace=lecture_14) | [源码仓库 (`stanford-cs336/lectures`)](https://github.com/stanford-cs336/lectures) | [Assignment 4 指导手册与代码库 (`assignment4-data`)](https://github.com/stanford-cs336/assignment4-data)

## 1. 引言：从原始数据矿藏到高纯度训练燃料的全链路工程（`main`）

在上一讲（Lecture 13: Data Part 1）中，我们探讨了训练数据的宏观生态与演进历史：从在线实时服务（Live Services，如 GitHub、Wikipedia、互联网服务器）出发，通过网络爬虫（Crawling）或官方全量镜像（Dumps）获取原始数据，并剖析了服务条款（ToS）、版权法（许可证与合理使用 Fair Use）以及从 2018 年的 BERT 到 2025 年的 Nemotron-CC / Common Pile 的 14 个标志性预训练数据集。

在本讲（Lecture 14: Data Part 2）中，Percy 将视角从“宏观数据集盘点”切换到**“微观算法与数据清洗管线（Data Pipeline Mechanics）”**——这也是你在 **Assignment 4** 中需要亲手从零编写的核心算法栈。我们将沿着原始数据变为模型权重的完整生命周期，深入拆解五大核心工程环节：
1. **数据格式转换（Transformation，`transformation`）**：如何从嘈杂的 HTML DOM 树与截断的 PDF 文件（如 FinePDFs）中剥离广告导航模板，高保真地提取出线性化纯文本与表格/公式结构？
2. **模型化质量与安全过滤（Filtering，`filtering`）**：如何用统一的“目标集 $T$ vs. 原始池 $R$”打分框架（KenLM 生成式困惑度、fastText 判别式线性分类器、大模型蒸馏随机森林），在百亿级网页上以极低算力完成语种识别、数学挖掘（OpenWebMath）、教科书级代码筛选（phi-1）与毒性过滤（Jigsaw）？过滤阈值与训练时长（Token Horizon）之间存在怎样的尺度耦合？
3. **精确与近似模糊去重（Deduplication，`deduplication`）**：面对 $O(N^2)$ 两两比对的计算高墙，如何利用 **MurmurHash**、**Jaccard 相似度**、**MinHash（最小哈希）** 与 **LSH（局部敏感哈希的分段 AND-OR 锐化相变）**，在线性时间 $O(N)$ 内清除全网重复文本？
4. **跨领域数据配比（Data Mixing，`data_mixing`）**：当手握数十个质量与体积悬殊的数据源时，如何跨越手工拍脑袋（Vibes），通过 **UniMax（硬性 Epoch 上限截断）**、**回归代理模型（RegMix / Olmix）** 与 **模拟多轮过采样（Simulated Epoching, Held et al., 2025）** 科学求解最优混合分布？
5. **后训练与合成数据工程（Post-training & Synthetic Data，`post_training_data`）**：从推理模型蒸馏配方（**OpenThoughts** 1000+ 次消融实验的深刻洞见）到软件工程智能体（SWE Agents）的三代数据扩展革命（**SWE-smith** 合成 Bug $\to$ **SWE-Zero / SWE-Hero** 无执行反馈世界模型轨迹 $\to$ **SWE-rebench & 12M 轨迹扩展**）。

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│                       CS336 Lecture 14 全景架构：现代大模型数据工程五大流水线                   │
├─────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                 │
│  [Stage 1: 格式转换 (Transformation)]                                                           │
│    • 原始形态: HTML (WARC) / PDF (Common Crawl) / 代码目录树                                    │
│    • 核心工具: trafilatura / resiliparse / jusText │ FinePDFs (补抓截断 PDF + RolmOCR / Docling) │
│          │                                                                                      │
│          ▼                                                                                      │
│  [Stage 2: 模型化过滤 (Model-Based Filtering)] ──► 统一范式: 给定小目标集 T 与大原始池 R 训打分器│
│    • 生成式打分: KenLM n-gram 语言模型算 p_T(x) (如 CCNet, OpenWebMath < 15000 PPL)             │
│    • 判别式打分: fastText 线性分类器算 p(T|x) (语种识别 / GPT-3 Pareto采样 / DCLM / Jigsaw毒性)  │
│    • LLM 蒸馏打分: GPT-4 标注 100K 样本 ──► 训轻量分类器 (如 phi-1 随机森林 / FineWeb-Edu)      │
│    • 尺度依赖定律: 短训期偏好严苛高质小池; 长训期若过度过滤会导致多 Epoch 过拟合反输给未过滤池! │
│          │                                                                                      │
│          ▼                                                                                      │
│  [Stage 3: 线性时间去重 (Linear-Time Deduplication)]                                            │
│    • 精确去重 (Exact): MurmurHash (mmh3) + MapReduce 排序分组 (如 C4 三句话跨度去重)             │
│    • 模糊去重 (Fuzzy): n-gram 集合 Jaccard(A,B) ──► MinHash (碰撞率=Jaccard) ──► LSH (b段×r行)   │
│          │                                                                                      │
│          ▼                                                                                      │
│  [Stage 4: 科学数据配比 (Data Mixing)]                                                          │
│    • 防过拟合护栏: UniMax (限制单源最大重复轮数 C) & Simulated Epoching (按比例下采样模拟饥饿)  │
│    • 自动优化框架: RegMix & Olmix (小模型跑 Dirichlet 随机配比 ──► 拟合回归模型 ──► 求解最优 p*)│
│          │                                                                                      │
│          ▼                                                                                      │
│  [Stage 5: 后训练与智能体合成数据 (Post-Training & Agentic Synthetic Data)]                     │
│    • 推理蒸馏: OpenThoughts (1.2M 样本, 每题采样16条, QwQ-32B 比 DeepSeek-R1 更适合当老师!)      │
│    • 代码智能体: SWE-smith (50K 合成Bug) ──► SWE-Zero/Hero (300K 免Docker执行轨迹) ──► 12M 轨迹 │
└─────────────────────────────────────────────────────────────────────────────────────────────────┘
```

## 2. 第一站：数据格式转换（Transformation）—— 从 HTML 与 PDF 到纯文本（`transformation`）

Percy 再次强调：**原始数据（Raw Data）绝不是现成的纯文本字符串！** 当你打开 Common Crawl 的原始数据包时，里面装的是嵌套的 **HTML 源码**、二进制 **PDF 文件**，或者在 GitHub 上是包含多文件的**代码仓库目录树（Directory Trees）**。数据流水线的第一步，就是将这些异构格式**线性化（Linearize）**为自回归语言模型能够消化的一维 Token 序列。

### 2.1 HTML 到纯文本的提取艺术与工具选型

互联网上绝大多数网页都是 HTML 格式。将 HTML 转换为纯文本面临三大核心挑战：
1. **剥离样板噪声（Boilerplate Removal）**：必须剔除顶部导航栏（Navigation bars）、侧边栏菜单、页脚版权声明（Footers）以及穿插其中的广告弹窗（Ads），精准定位并提取网页的**主体正文（Main Content）**。但 Percy 指出，有时候“什么是正文、什么是模板”的边界也很模糊（例如如果你未来想训练一个浏览网页的 Web Agent，保留一点导航元素反而有助于模型理解网页结构）。
2. **二维与层级结构的有损线性化（Inherently Lossy Linearization）**：
   网页渲染出来是二维的视觉排版（包含图片、嵌套表格、代码块、列表）。当你把它压扁成一维纯文本时，不可避免会发生信息丢失：例如简单的二维表格可以转换成 Markdown 表格，但遇到复杂的**多层嵌套表格（Nested Tables）**时，解析器往往只能放弃或做粗糙近似。
3. **为什么工业界普遍采用“基于规则（Rule-based）”的提取工具，且工具选型直接决定下游 MMLU 精度？**
   * 因为你要处理高达数百 TB 甚至 PB 级的全量 Common Crawl，提取器必须**极快（Extremely Fast）**！目前主流的开源提取器包括 **`trafilatura`**、**`resiliparse`**、**`jusText`** 以及文本浏览器 **`lynx`**。
   * 正如我们在 Lecture 13 结合 **[DataComp-LM (Li et al., 2024)](https://arxiv.org/abs/2406.11794)** 看到的实证对比：在完全相同的模型架构与训练步数下，仅仅是将 HTML 提取工具从 Common Crawl 官方自带的 `WET` 替换为 **`resiliparse`** 或 **`trafilatura`**，训练出的模型在下游核心评测基准（Core Eval）上的准确率就获得了显著提升！

### 2.2 挖掘高价值 PDF 矿藏：[FinePDFs (Hugging Face, 2025/2026)](https://huggingface.co/datasets/HuggingFaceFW/finepdfs)

除了 HTML，互联网上还散布着大量 **PDF 文档**。Percy 敏锐地指出了一条经验法则：**总体而言，网上平均一份 PDF 文档的信息质量显著高于平均一个普通 HTML 网页**——因为如果一个人愿意费劲把内容排版导出成 PDF（如学术论文、政府白皮书、技术手册、电子书），他大概率真的有严肃、高价值的内容要表达！

然而，直接从 Common Crawl 处理 PDF 是一场硬仗。Hugging Face 团队在构建 **[FinePDFs](https://huggingface.co/datasets/HuggingFaceFW/finepdfs)** 时总结了全套工程管线：
1. **重新抓取被截断的 PDF（Recrawl Truncated PDFs）**：由于 Common Crawl 对单个网页响应设置了字节大小上限（如 1MB–5MB），而很多高质量 PDF 体积较大，Common Crawl 里保存的 PDF 绝大部分是**被拦腰截断的残损文件**！因此第一步必须根据 URL 重新发起网络请求抓回完整的 PDF 文件。
2. **高速视觉 OCR 与版面解析（RolmOCR / Docling）**：
   不同于 HTML 至少带有 `<h1>`、`<p>` 等语义标签，PDF 的底层指令纯粹是为了“打印机在坐标 $(x, y)$ 处画一个字符”而设计的**纯视觉排版格式（All about layout, missing semantic structure）**，甚至很多老旧 PDF 干脆就是扫描的图片！因此，必须动用轻量级视觉语言模型（VLM，如 **RolmOCR**）或专业文档解析器（如 **Docling**），将其极致加速后批量运行 OCR，再经过繁琐的文本拼合、去页眉页脚与乱码过滤，才能转化为高质量 Markdown 文本。

## 3. 第二站：模型化数据过滤（Filtering）—— 统一数学框架与四大实战案例（`filtering`）

把 HTML 和 PDF 转换成纯文本后，你手头有了一座体量高达 $100\text{T}\sim 240\text{T}$ Tokens 的原始文本大山（Raw Data $R$），但其中 90% 以上依然是低质垃圾。接下来进入决定模型生死的第二步：**过滤（Filtering）**。

### 3.1 过滤算法的统一抽象框架（General Schema for Model-Based Filtering）

Percy（结合综述论文 **[A Survey on Data Selection for Language Models, Albalak et al., 2024](https://arxiv.org/abs/2402.16827)**）将几乎所有的现代数据过滤算法抽象为一个极其优雅的**核心构建块（Algorithmic Building Block）**：

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│                     数据过滤的统一抽象框架 (Target T + Raw R ──► Filtered T')                    │
├─────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                 │
│  [输入 1: 目标高质量小数据集 T (Target Data)]     [输入 2: 海量原始数据集 R (Raw Data, ~100T+)] │
│   (如 Wikipedia / OpenHermes+ELI5 / GPT-4 打标集)  (如刚从 Common Crawl 提取出的全量网页文本)   │
│                        │                                                │                       │
│                        └───────────────┐                ┌───────────────┘                       │
│                                        ▼                ▼                                       │
│                     [步骤 1: 基于 T (及 R) 拟合一个极速打分模型 score(x)]                        │
│                       • 路线 A (生成式 Generative): 在 T 上训 KenLM 5-gram, score(x) = p_T(x)    │
│                       • 路线 B (判别式 Classifier): T 为正例、R 为负例训 fastText, score(x)=p(T|x)│
│                                                │                                                │
│                                                ▼                                                │
│                     [步骤 2: 遍历全量 R 计算得分，保留高分样本构成精炼子集 T' ⊂ R]               │
│                       • 确定性阈值截断: Keep x if score(x) >= threshold                          │
│                       • 随机化概率采样: Keep x stochastically (如 GPT-3 的 Pareto 采样)         │
└─────────────────────────────────────────────────────────────────────────────────────────────────┘
```

* **设计过滤算法必须同时满足的两大核心诉求（Desiderata）**：
  1. **具备泛化外推能力（Generalization from $T$）**：我们已经有目标集 $T$ 了，过滤的目的不是把 $T$ 原样找出来，而是要在 $R$ 中发掘出**风格、质量或领域与 $T$ 相似，但内容全新的广袤子集 $T'$**（即要求 $T' \neq T$ 且 $|T'| \gg |T|$）！
  2. **推理速度必须快到极致（Extremely Fast）**：因为打分器必须在包含数百亿文档、上百兆亿（$100\text{T}+$）Token 的原始池 $R$ 上全量跑一遍！这就是为什么你绝不可能直接拿一个 `70B` 甚至 `7B` 的 Transformer 大模型去给全量 Common Crawl 逐篇打分，而必须使用 **$n$-gram 模型（KenLM）** 或 **词袋线性分类器（fastText）**！

### 3.2 两类主力轻量级打分器：KenLM vs. fastText

| 模型类型 | 代表工具 | 数学打分函数 $\text{score}(x)$ | 训练方式与计算特性 | 核心优缺点对比 |
| :--- | :--- | :--- | :--- | :--- |
| **1. 生成式语言模型 (Generative Model)** | **[KenLM](https://kheafield.com/code/kenlm/)** ($n$-gram LM, 通常取 5-gram) | $\text{score}(x) = p_T(x)^{1/|x|}$（等价于取负困惑度 $-\text{PPL}_T(x)$） | **仅需正样本集 $T$**！统计 $T$ 中的 $n$-gram 词频并配合 Kneser-Ney 平滑，C++ 实现速度极快 | **优点**：无需构造负样本；<br>**缺点**：偏爱短句、常见词与重复模板（因为重复句子的困惑度极低！需配合长度/去重规则） |
| **2. 判别式线性分类器 (Discriminative Classifier)** | **[fastText](https://fasttext.cc/)** (Facebook 开源词袋/字符 $n$-gram 线性分类器) | $\text{score}(x) = p(y = T \mid x) = \sigma\big(w^\top \phi_{\text{BoW}}(x)\big)$ | 将 $T$ 作为**正样本（Positive）**，从 $R$ 中随机采样等量文档作为**负样本（Negative）**，训练二分类逻辑回归 | **优点**：直接学习区分高质与垃圾网页的判别边界，抗低熵重复干扰能力强，速度达每秒数千篇文档；<br>**现已成为工业界（如 DCLM）的绝对主流** |

### 3.3 五大经典实例化案例深度拆解（Instantiations of Filtering）

Percy 在 `lecture_14.py`（Lines 89–133）中展示了这一通用框架在五个截然不同场景下的精妙应用：

#### 实例 1：语种识别（Language Identification —— Dolma & CCNet）
* **目标**：从多语言混杂的网页中筛选出特定目标语种（如英语 English）。
* **方案**：直接调用 Meta 预训练好的开源 **[`fastText` 176 语种识别模型 (`lid.176.bin`)](https://fasttext.cc/docs/en/language-identification.html)**（该模型在多语言 Wikipedia、Tatoeba 翻译句对网与 SETimes 东南欧新闻网上训练而成）。
* **阈值设定**：在 AI2 的 **[Dolma (Soldaini et al., 2024)](https://arxiv.org/abs/2402.00159)** 中，保留所有满足 **`p(English) >= 0.5`** 的网页（而在 FineWeb 中则设为 `>= 0.65`）。Percy 指出，相比于质量过滤，语种识别是一个相对简单且成熟的环节（尽管理解代码混写 Code-switching 和方言仍有细节挑战，但它通常不是训练好模型的瓶颈）。

#### 实例 2：垂直领域挖掘——数学网页语料库 [OpenWebMath (Paster et al., 2023)](https://arxiv.org/abs/2310.06786)
* **目标**：为了复现 Google Minerva 的数学推理能力，从 2370 亿个 Common Crawl 原始 HTML 页面中淘出高质量数学网页。
* **三级组合过滤管线（Rules + KenLM + fastText）**：
  1. **HTML 保真提取 + 规则初筛**：定制 HTML 解析器保留 MathJax / KaTeX 公式为标准 LaTeX 源码，并用规则过滤是否包含数学符号或 LaTeX 命令；
  2. **KenLM 生成式过滤**：在已知的高质量数学语料库 **Proof-Pile** 上训练一个 KenLM 模型，计算候选网页的困惑度，**仅保留 `Perplexity < 15,000` 的文档**；
  3. **fastText 判别式过滤（双阈值策略）**：训练一个 `fastText` 分类器预测文档是否属于数学写作。如果网页中已经检测到了明确的 LaTeX 公式，只需分类器得分 **`>= 0.17`** 即可通过（低门槛）；如果网页中没有显式 LaTeX 公式（纯文本数学讨论），则要求分类器得分必须 **`>= 0.80`**（高门槛）！
* **惊人成效**：最终从全网提炼出 **`14.7B` Tokens** 的高纯度数学语料；在上面训练的 `1.4B` 小模型，其数学推理表现**直接超越了在超过 20 倍数据量（300B+ Tokens）的通用网页语料上训练的同尺寸模型**！这生动证明了：**只要你把“目标集 $T$”定义为某个专业领域（如数学），质量过滤工具就能直接变身为强大的领域数据挖掘机**！

#### 实例 3：通用高质量网页过滤——[GPT-3 (Brown et al., 2020)](https://arxiv.org/abs/2005.14165) 与 [LLaMA 1 (Touvron et al., 2023)](https://arxiv.org/abs/2302.13971)
* **GPT-3 的随机化帕累托采样（Stochastic Pareto Filtering）**：
  * **正样本 $T$**：来自 `{Wikipedia, WebText2, Books1, Books2}`；**负样本 $R$**：原始 Common Crawl。
  * 基于词特征训练线性分类器为每个文档打出质量分 $\text{score} \in [0, 1]$。为了防止死板阈值截断导致数据失去多样性，GPT-3 没有使用硬阈值，而是通过**帕累托分布（Pareto Distribution，$\alpha=9$）进行随机化保留**：
    ```python
    import numpy as np

    def keep_document(score: float) -> bool:
        # 得分 score 越高 (1 - score 越小)，越容易通过；但低分文档也有一定小概率被保留以维持多样性!
        return np.random.pareto(9) > 1 - score
    ```
* **LLaMA 1 / RedPajama 的防百科腔微调**：
  * **正样本 $T$**：不选 Wikipedia 本身，而是选**“被 Wikipedia 词条作为外部参考文献（References）引用的网页”**；**负样本 $R$**：随机 Common Crawl 网页。训练线性分类器并保留被判为正类的文档。

#### 实例 4：大模型蒸馏两阶段过滤——[phi-1: Textbooks Are All You Need (Gunasekar et al., 2023)](https://arxiv.org/abs/2306.11644)
如果你想筛选出“像优秀教材一样适合初学者学习编程概念”的网页或代码，但你手头根本没有现成的“教科书目标数据集 $T$”怎么办？
微软 **phi-1 (`1.3B` 参数)** 开创了如今风靡工业界（如 FineWeb-Edu、Nemotron-CC）的**“大模型主动打标构造 $T$ $\to$ 蒸馏训练轻量分类器横扫全网 $R$”**的两阶段范式：

```python
# phi-1 (Gunasekar et al., 2023) 的两阶段大模型蒸馏过滤配方 (源自 lecture_14.py Lines 120-127)
R = "Python subset of The Stack (35M files)"  # 海量原始代码池 R
prompt = "determine its educational value for a student whose goal is to learn basic coding concepts"

# 第一阶段：调用昂贵的 GPT-4，用上述 Prompt 对 R 的一小部分子集 (100K 个文件) 进行打分标注，构建目标集 T！
T = "Use GPT-4 with this prompt to classify 100K subset of R to get positive examples"

# 第二阶段：提取预训练 CodeGen 模型的输出 Embedding 作为特征，在 100K 标注集 T 上训练一个极速的随机森林 (Random Forest) 分类器！
# 第三阶段：用这个轻量级随机森林分类器横扫全量 35M 原始池 R，筛选出被判为正类的“高教育价值”代码子集！
```

* **HumanEval 实测对比**：
  * 在未过滤的原始 The Stack Python 子集上训练 `1.3B` 模型（跑 `96K` 步）：HumanEval Pass@1 仅为 **`12.19%`**；
  * 在经过上述“教育价值分类器”过滤出的精炼子集上训练同一个 `1.3B` 模型（仅跑 `36K` 步，算力不到前者的 40%）：HumanEval 直接跃升至 **`17.68%`**（再配合 GPT-3.5 合成的 `1B` 教科书与练习题后更是飙升至 **`50.6%`**）！

#### 实例 5：不良与毒性内容过滤——[Dolma (Soldaini et al., 2024)](https://arxiv.org/abs/2402.00159) 与 Jigsaw 数据集
* **目标**：过滤掉互联网上的仇恨言论、色情、侮辱与暴力威胁。
* **方案**：利用 Kaggle / Google 发布的 **[Jigsaw Toxic Comment Classification Dataset (2018)](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge)**——该数据集源自 **Wikipedia Talk Pages（维基百科词条背后的编辑争论讨论页）**，由人类标注员细分为 6 类毒性标签：`{toxic, severe_toxic, obscene, threat, insult, identity_hate}`。Dolma 在此数据集上训练了两个轻量级 `fastText` 分类器（一个识别 `obscene` 色情下流，一个识别 `hate` 仇恨攻击），用于高速清洗全量网页。

### 3.4 过滤的尺度依赖效应（Scale-Dependent Effects）：为什么不存在唯一最优的过滤阈值？

在掌握了如何训练分类器打分 $\text{score}(x) \in [0, 1]$ 后，一个最自然的工程问题是：**我应该把过滤阈值设在多少？是只留 Top 1% 最精华的数据，还是留 Top 20%？**

Percy 在 `lecture_14.py`（Lines 134–139）中展示了一组由斯坦福博士生 Michael Ryan 完成的极具启发性的受控消融实验（在一个固定大小的 `100` 个 Common Crawl WARC 分片原始池上，训练一个 `157M` 参数的模型，横轴为训练 Token 总数，纵轴为下游验证集 Loss）：

```
下游验证集 Loss
  ▲
  │  ● 原始未过滤池 (Resiliparse, 数据量大但质量低)
  │   \
  │    \       ★ 交叉反转点 (Crossover Point)!
  │     \     /  在长周期训练下，过度严苛过滤的 DCLM 因被迫重复跑几十个 Epoch 而严重过拟合，
  │      \   /   最终效果反而输给了未过滤的庞大原始池!
  │  ○    \ /
  │   \    ╳ ──────────────────────► 原始未过滤池 (单轮 Epoch 持续稳定下降)
  │    \  / \
  │     ○────●──●──● (DCLM Top 1.5%: 经历第 1, 2, 3... 个 Epoch 后下降停滞并开始过拟合反弹!)
  └─────┴────┴───────────► 总训练步数 / 训练 Tokens 数量 (Training Horizon)
     短训练周期        长训练周期
```

* **实验揭示的铁律**：
  1. **短训练周期（Compute-Constrained / Short Horizon）**：当你只训练很少的步数（两条曲线左侧，尚未把过滤后的数据跑完 1 个 Epoch）时，**经过 DCLM 严苛过滤的高质量小数据集（蓝线）以压倒性优势大幅领先未过滤的原始数据（红线）**！
  2. **长训练周期（Data-Constrained / Long Horizon）**：由于给定的原始池总量是有限的（100 个 WARC 文件），严苛过滤扔掉了 98% 的数据，导致高质量子集很快就被用光了！随着训练继续推进，蓝线被迫在同一个小数据集上反复训练第 2、第 3、第 10 个 **Epoch（图中每个圆点代表多跑一轮 Epoch）**，边际收益迅速衰减并开始过拟合；而此时红线（未过滤或宽松过滤的庞大数据集）因为一直在见全新的独特 Token，竟然在右侧实现了**交叉反超（Crossover）**！
* **核心工程结论**：**绝对不存在一个脱离训练规模的“普适最优过滤阈值”！如果你计划训练的时间较短（相对于数据池规模），你应该设置极严苛的阈值只取最高质量数据；但如果你计划进行超长周期的训练（Long Horizon），你必须放宽过滤阈值（或像 Nemotron-CC 那样用合成改写挽救中低质数据），用轻微的质量妥协换取充足的唯一 Token 数量，以避免灾难性的多轮重复过拟合！**

## 4. 第三站：数据去重（Deduplication）—— 从精确哈希到 MinHash LSH 相变数学推导（`deduplication`）

经过转换和过滤后，我们得到了高质量文档集合。但互联网上充斥着惊人比例的重复内容：
1. **精确重复（Exact Duplicates）**：例如 [Project Gutenberg 遍布全球的数十个镜像站点（Mirrors）](https://www.google.com/search?q=https://www.gutenberg.org/DIRS/GUTINDEX.ALL)、GitHub 上被直接 Fork 的代码仓库。
2. **近似重复（Near Duplicates —— 仅相差几个词或格式的文本）**：
   * 随处可见的开源许可证（如 **MIT License**，仅年份和作者名不同）、网站通用的页眉页脚与隐私条款；
   * 批量套用模板生成的垃圾广告网页（例如把同一段推广文案里的 `"Canada"` 替换成 `"USA"`、`"UK"` 生成上千个页面）；
   * **C4 数据集中的惊世奇观（[Lee et al., 2021](https://arxiv.org/abs/2107.06499)）**：
     Katherine Lee 等人在审计 Google 的 C4 数据集时震惊地发现，**下面这段长达 61 个词、语法极其蹩脚的婚礼摄影/防毒面具商品推销话术，竟然在“洗干净”的 C4 数据集里一字不差或轻微变体地重复出现了整整 `61,036` 次**！
     > *"“by combining fantastic ideas, interesting arrangements, and follow the current trends in the field of that make you more inspired and give artistic touches. We'd be honored if you can apply some or all of these design in your wedding. believe me, brilliant ideas would be perfect if it can be applied in real and make the people around you amazed!"*

### 4.1 为什么必须去重？去重的设计空间与核心算法挑战

* **[Deduplicating Training Data Makes Language Models Better (Lee et al., 2021)](https://arxiv.org/abs/2107.06499)** 证实了去重的三大核心收益：
  1. **大幅提升训练算力效率**：剔除占数据总量 30%～50% 的重复水分，用更少的训练步数达到相同甚至更高的下游准确率；
  2. **抑制逐字背诵（达到 10 倍降幅），缓解版权与隐私泄漏风险**：未去重模型在无提示生成时有超过 `1%` 的输出是在逐字背诵训练集中的高频重复片段，去重后背诵率直接下降 **10 倍**；
  3. **消除训练-测试集重叠（Train-Test Overlap / Decontamination）**：当时标准数据集验证集中有超过 `4%` 的样本直接泄漏在训练集中。
* **去重系统的三维设计空间（Design Space）**：
  1. **去重粒度（What is an item?）**：是按**整篇文档（Document）**、**段落（Paragraph）**、还是**连续 $k$ 句话的跨度（如 C4 的 3-sentence spans）**？
  2. **匹配准则（How to match?）**：是要求**逐字符精确匹配（Exact Match）**，还是存在公共子串，抑或**子单元集合重合比例达到阈值（Fraction of common subitems / Jaccard Similarity）**？
  3. **处置动作（What action to take?）**：发现一组重复项后，是全部删光（Remove all），还是**保留其中唯一一份、删掉其余副本（Remove all but one —— 标准做法）**？
* **核心算法挑战（Why is Deduplication Hard at Scale?）**：
  过滤（Filtering）是对每个文档 $x$ 独立打分，天然是线性时间 $O(N)$ 且极易并行；而**去重本质上是“拿每一个文档去和所有其它文档做比较”**！如果写成朴素的双重循环两两比对，时间复杂度是 **$O(N^2)$**——面对百亿级网页，$N^2 = 10^{20}$，全世界的计算机算到宇宙毁灭也跑不完！
  **因此，去重算法的第一性铁律是：必须借助哈希函数（Hash Functions），将时间复杂度严格降为线性时间 $O(N)$（或准线性 $O(N \log N)$）！**

### 4.2 哈希函数选型与精确去重（`hash_functions`, `exact_deduplication`）

1. **密码学哈希 vs. 非密码学高速哈希**：
   * **密码学哈希（如 SHA-256，用于比特币与数字签名）**：极度抗碰撞（恶意攻击者也无法构造碰撞），但计算速度太慢；
   * **非密码学哈希（如 DJB2、MurmurHash、CityHash、xxHash）**：专为哈希表与大规模数据处理设计，极快且分布均匀。在 `lecture_14.py` 和 Assignment 4 中，我们统一使用 **`mmh3`（MurmurHash3）**：
     ```python
     import mmh3
     h = mmh3.hash("hello", seed=42)  # 极速将任意字符串映射为 32-bit / 128-bit 整数
     ```
2. **MapReduce 风格的精确去重实现**：
   ```python
   import itertools
   import mmh3

   items = ["Hello!", "hello", "hello there", "hello", "hi", "bye"]

   # 1. Map & Shuffle: 按哈希值排序并分组 (在大规模分布式系统中对应 MapReduce 的 Shuffle 阶段)
   hash_items = itertools.groupby(sorted(items, key=mmh3.hash), key=mmh3.hash)

   # 2. Reduce: 每个哈希桶中仅保留第一个元素 (Remove all but one)
   deduped_items = [next(group) for h, group in hash_items]
   # 结果: 两个 "hello" 被成功合并保留为 1 个!
   ```
   * **C4 三句话精确去重的反面教材警示**：
     T5 的 C4 数据集以“连续 3 个句子（3-sentence spans）”为单位做精确哈希去重。Percy 指出了这种做法的一个严重副作用：**当两篇文章中间恰好引用了相同的三句话时，第二篇文章中间的这三句话会被硬生生挖掉（Rip out from the middle of a document），导致剩下的上下文前言不搭后语、彻底丧失篇章连贯性**！因此现代管线在文档内部去重时更加谨慎，更多采用文档级或段落级的模糊去重。

### 4.3 近似模糊去重基石一：Jaccard 相似度与 MinHash 定理（`jaccard_minhash`）

为了捕捉仅相差几个词的近似重复文档（Near Duplicates），我们首先将每篇文档表示为一个**集合（Set）** $A$（在实践中，通常将文档切分为词级或字符级的 **$n$-grams / Shingles 集合**，例如所有 5-grams 构成的集合）。

#### 1. Jaccard 相似度（Jaccard Similarity）定义
给定两个集合 $A$ 和 $B$，它们的 Jaccard 相似度定义为**交集大小除以并集大小**：
$$\mathbf{\text{Jaccard}(A, B) \triangleq \frac{|A \cap B|}{|A \cup B|} \in [0, 1]}$$

```python
A = {"1", "2", "3", "4"}
B = {"1", "2", "3", "5"}

def compute_jaccard(A: set, B: set) -> float:
    intersection = len(A & B)  # |{"1", "2", "3"}| = 3
    union = len(A | B)         # |{"1", "2", "3", "4", "5"}| = 5
    return intersection / union

jaccard = compute_jaccard(A, B)  # 3 / 5 = 0.6
```
如果两篇文档满足 $\text{Jaccard}(A, B) \ge \tau$（例如阈值 $\tau = 0.8$ 或 $0.9$），我们就判定它们为近似重复文档。但直接计算任意两篇文档的 $|A \cap B| / |A \cup B|$ 依然需要 $O(N^2)$！怎么把它变成 $O(N)$ 的哈希碰撞？

#### 2. MinHash（最小哈希）的神奇构造与严格数学证明
通常我们在设计哈希表时，唯恐发生哈希碰撞（Hash Collision）；但在近似去重中，我们的目标恰恰相反——**我们希望设计一种随机哈希函数 $h$，使得两个集合发生哈希碰撞的概率，严格等于它们的 Jaccard 相似度**！
$$\mathbf{\mathbb{P}_{h \sim \mathcal{H}}\big[h(A) = h(B)\big] = \text{Jaccard}(A, B)}$$

**[Andrei Broder (1997)](https://ieeexplore.ieee.org/document/666900) 提出的 MinHash 算法极其简单**：
给定一个将单个元素映射为随机整数的哈希函数 `mmh3.hash(x, seed)`，集合 $S$ 的 **MinHash 值**定义为**集合中所有元素经过哈希后的最小值**：
$$\mathbf{h_{\text{min}}(S) \triangleq \min_{x \in S} h(x)}$$

```python
def minhash(S: set[str], seed: int) -> int:
    return min(mmh3.hash(x, seed) for x in S)
```

* **为什么“取集合中哈希值的最小值”就能让碰撞概率刚好等于 $\frac{|A \cap B|}{|A \cup B|}$？（特征矩阵直观证明）**
  把全集 $A \cup B = \{1, 2, 3, 4, 5\}$ 的归属关系画成一个 $0\text{-}1$ **特征矩阵（Characteristic Matrix）**：

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│             MinHash 定理证明：特征矩阵与随机置换 (A = {1,2,3,4}, B = {1,2,3,5})         │
├─────────────┬───────────┬───────────┬───────────────────────────────────────────────────┤
│ 元素 x      │ 集合 A    │ 集合 B    │ 元素类别归属与碰撞分析                            │
├─────────────┼───────────┼───────────┼───────────────────────────────────────────────────┤
│   "1"       │     1     │     1     │ 属于 A ∩ B (共 3 行): 若此类元素排在全集第 1 名， │
│   "2"       │     1     │     1     │ 则它同时是 A 的最小值和 B 的最小值 ──► h(A)==h(B)!│
│   "3"       │     1     │     1     │                                                   │
├─────────────┼───────────┼───────────┼───────────────────────────────────────────────────┤
│   "4"       │     1     │     0     │ 属于 (A \ B) ∪ (B \ A) (共 2 行): 若此类元素排在  │
│   "5"       │     0     │     1     │ 全集第 1 名，则一方取到它、另一方取不到 ──► 必不相等!│
└─────────────┴───────────┴───────────┴───────────────────────────────────────────────────┘
```

  1. 对每个元素应用随机哈希函数 $h(x)$ 并按哈希值从小到大排序，等价于在并集 $A \cup B$ 的所有 $|A \cup B|$ 个元素上产生了一个**均匀随机排列（Uniform Random Permutation）**。
  2. 在这个随机排列中，$A \cup B$ 里的每一个元素成为**全集最小值**（即排在排列的第一位：$\arg\min_{x \in A \cup B} h(x)$）的概率是**完全相等的**，均为 $\frac{1}{|A \cup B|}$！
  3. 观察特征矩阵：当且仅当排在 $A \cup B$ 第一位的那个最小元素**同时属于 $A$ 和 $B$（即该行在 A、B 两列均为 `1`，属于交集 $A \cap B$）**时，$A$ 的最小值和 $B$ 的最小值才会是同一个元素（即 $h_{\text{min}}(A) = h_{\text{min}}(B)$）！如果排在第一位的元素只属于 $A$（如元素 `4`），那么 $h_{\text{min}}(A) = h(4)$，而 $B$ 不含 `4`，故 $h_{\text{min}}(B) > h(4)$，两者绝不可能相等！
  4. 因此，发生碰撞的概率严格等于“$A \cup B$ 中的最小哈希元素恰好落在子集 $A \cap B$ 中”的概率：
     $$\mathbb{P}\big[h_{\text{min}}(A) = h_{\text{min}}(B)\big] = \frac{\text{两列同为 1 的行数}}{\text{总行数}} = \mathbf{\frac{|A \cap B|}{|A \cup B|} = \text{Jaccard}(A, B)} \quad \blacksquare$$

```python
# 用 100 个随机种子验证 MinHash 碰撞频率完美逼近 Jaccard 相似度 (0.6)
n = 100
matches = [minhash(A, seed) == minhash(B, seed) for seed in range(n)]
estimated_jaccard = len([m for m in matches if m]) / len(matches)
assert abs(estimated_jaccard - jaccard) < 0.01  # 实测刚好等于 0.60!
```

### 4.4 近似模糊去重基石二：局部敏感哈希（MinHash + LSH）与 S 型相变曲线推导（`locality_sensitive_hashing`）

有了 MinHash，我们还差临门一脚：单个 MinHash 的碰撞概率虽然等于 $s = \text{Jaccard}(A, B)$，但这是一个平滑的 45 度斜线（例如当相似度高达 $0.8$ 时还有 $20\%$ 概率漏检，而当相似度只有 $0.3$ 时却有高达 $30\%$ 的概率误碰撞！）。
**我们的工程目标是实现一个“阶跃相变函数（Step / Phase Transition Function）”**：设定一个相似度阈值 $\tau$（比如 $\tau = 0.9$），凡是 $\text{Jaccard}(A, B) < \tau$ 的文档对碰撞概率趋近于 **`0`**，凡是 $\text{Jaccard}(A, B) > \tau$ 的文档对碰撞概率趋近于 **`1`**！

#### 1. LSH 的分桶（Banding）与 `AND-OR` 结构构造
[Locality-Sensitive Hashing (LSH, Indyk & Motwani, 1998)](https://dl.acm.org/doi/10.1145/276698.276876) 通过极其巧妙的**分段组合（Banding Technique）**实现了对碰撞概率曲线的锐化（Sharpening）：
1. 为每篇文档计算 $n$ 个独立的 MinHash 值：$[h_1, h_2, \dots, h_n]$；
2. 将这 $n$ 个哈希值切分为 **$b$ 个段（Bands）**，每个段包含 **$r$ 个哈希值（Rows per band）**，满足 **$n = b \times r$**。
   例如当 $n = 12, b = 3, r = 4$ 时：
   $$\underbrace{h_1 \; h_2 \; h_3 \; h_4}_{\text{Band 1 }(r=4)} \;\Big|\; \underbrace{h_5 \; h_6 \; h_7 \; h_8}_{\text{Band 2 }(r=4)} \;\Big|\; \underbrace{h_9 \; h_{10} \; h_{11} \; h_{12}}_{\text{Band 3 }(r=4)}$$
3. **碰撞判定准则（Band 内部取 `AND`，不同 Bands 之间取 `OR`）**：
   我们判定两篇文档 $A$ 和 $B$ **发生 LSH 碰撞（即成为候选重复对）**，当且仅当**存在至少一个 Band（$\exists j \in \{1, \dots, b\}$，即 `OR`），使得在该 Band 内的全部 $r$ 个 MinHash 函数值完全一致（$\forall k \in \text{Band}_j, h_k(A) = h_k(B)$，即 `AND`）**！
   *(工程实现上极其简单：只需把每个 Band 的 $r$ 个整数拼成一个元组，再用标准哈希表为每个 Band 建桶；只要两篇文档在任意一个 Band 落入同一个哈希桶，就触发碰撞！)*

#### 2. 碰撞概率公式推导与 $(b, r)$ 调节规律
设两篇文档的真实 Jaccard 相似度为 $s = \text{Jaccard}(A, B)$：
1. **单个特定 Band 内部 $r$ 个哈希全部匹配（`AND`）的概率**：
   因为 $r$ 个哈希相互独立，每个匹配概率为 $s$，所以整个 Band 完全匹配的概率为：
   $$P_{\text{band\\_match}} = \mathbf{s^r}$$
2. **单个特定 Band 不匹配的概率**：
   $$P_{\text{band\\_mismatch}} = 1 - s^r$$
3. **全部 $b$ 个 Bands 都不匹配的概率**：
   $$P_{\text{all\\_bands\\_mismatch}} = (1 - s^r)^b$$
4. **最终 LSH 碰撞概率（至少有一个 Band 匹配，即 `OR`）**：
   $$\mathbf{P_{\text{collision}}(s; b, r) = 1 - \left(1 - s^r\right)^b}$$

```python
# 源自 lecture_14.py (Lines 292-310): LSH 碰撞概率计算与参数效应验证
def get_prob_collision(sim: float, b: int, r: int) -> float:
    prob_match = sim ** r                      # 固定某一个 Band 内全部 r 个哈希都相等的概率 (AND)
    prob_collision = 1 - (1 - prob_match) ** b # 至少存在一个 Band 完全匹配的概率 (OR)
    return prob_collision

# 对比三组 (b, r) 参数在不同相似度 sims 下的碰撞概率 S 型曲线：
# 1. b = 10, r = 10 (n = 100):  sim=0.7 -> 0.250 | sim=0.8 -> 0.682 | sim=0.9 -> 0.986
# 2. b = 10, r = 20 (增大 r):   sim=0.7 -> 0.008 | sim=0.8 -> 0.109 | sim=0.9 -> 0.729 (曲线变陡且右移，更难匹配!)
# 3. b = 20, r = 20 (再增大 b): sim=0.7 -> 0.016 | sim=0.8 -> 0.206 | sim=0.9 -> 0.927 (曲线左移，更容易匹配!)
```

```
LSH 碰撞概率 P_collision(s) = 1 - (1 - s^r)^b 的 S 型相变曲线:
  1.0 ┼───────────────────────────────────────────╭───────────
      │                                         ╭─╯
 0.63 ┼─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─● ◄── 相变中心阈值 τ ≈ (1/b)^(1/r)，此处 P ≈ 1 - 1/e ≈ 0.632!
      │             ◄── 增大 b (曲线左移，更易碰撞) │ 增大 r (曲线变陡且右移，更严苛) ──►
  0.0 ┼───────────────────────────────────────╭─╯
      0.0                                     τ            1.0  真实相似度 s = Jaccard(A, B)
```

#### 3. 相变阈值公式 $\tau = (1/b)^{1/r}$ 与 $1 - 1/e$ 不变量的数学推导
相变陡坡的中心拐点（Threshold $\tau$）出现在哪里？
观察公式 $1 - (1 - s^r)^b$：当 $s^r$ 恰好等于 $\frac{1}{b}$ 时，单个 Band 匹配的期望次数 $b \cdot s^r = 1$。由此解出**相变中心阈值 $\tau$**：
$$s^r = \frac{1}{b} \implies \mathbf{\tau = \left(\frac{1}{b}\right)^{\frac{1}{r}}}$$
将 $s = \tau = (1/b)^{1/r}$（即 $s^r = \frac{1}{b}$）代回碰撞概率公式，利用经典重要极限 $\lim_{b \to \infty} \left(1 - \frac{1}{b}\right)^b = \frac{1}{e}$，我们发现**在相变阈值 $\tau$ 处，碰撞概率是一个与大参数无关的普适常数**：
$$P_{\text{collision}}(\tau) = 1 - \left(1 - \frac{1}{b}\right)^b \approx \mathbf{1 - \frac{1}{e} \approx 0.6321}$$
* **真实工业界参数实例（[Lee et al., 2021](https://arxiv.org/abs/2107.06499)）**：
  在 C4 近似去重论文中，作者设置了 **$n = 9000$ 个哈希函数，分为 $b = 20$ 个 Bands，每个 Band 含 $r = 450$ 行**。
  代入公式计算其相变阈值：$\tau = \left(\frac{1}{20}\right)^{\frac{1}{450}} \approx \mathbf{0.99335}$！在 $\tau = 0.99335$ 处碰撞概率为 $1 - (1 - 1/20)^{20} \approx \mathbf{0.6415}$，而只要相似度稍微低于 $0.98$，碰撞概率瞬间归零；稍微高于 $0.996$，碰撞概率瞬间飙升至 $1.0$，形成了一道刀切般锋利的阶跃相变墙！

## 5. 第四站：跨数据源科学配比（Data Mixing）—— 从多轮过拟合陷阱到 RegMix / Olmix（`data_mixing`）

经过转换、过滤和跨数据集全局去重后，我们手头通常有几十个不同领域、不同质量等级的干净数据源（例如在斯坦福开源项目 **[Marin Token Viewer](https://marin.community/)** 中有 Nemotron-CC、FinePDFs、Starcoder 代码、arXiv 论文、合成教科书等；在 **The Pile** 中有 22 个子集）。
**核心问题：在构建预训练数据流时，我们应该赋予每个数据源 $s \in \mathcal{S}$ 多大的采样概率权重 $p(s)$（满足 $\sum_{s} p(s) = 1$）？（注：课堂问答确认，混合采样通常在每个 Batch 内部按序列级进行，即同一个 Batch 内包含按 $p(s)$ 抽样出的不同领域的文档序列，以减小梯度方差）。**

### 5.1 三大朴素基线与“高质小数据源 50 倍 Epoch 过拟合”陷阱

1. **三种常见基线策略（Baselines）**：
   * **凭直觉手调（Vibes）**：架构师拍脑袋定 `p = {"Wikipedia": 0.3, "CC": 0.5, "GitHub": 0.2}`（在工业界比你想象的要普遍得多！）；
   * **均匀采样（Uniform Sampling）**：所有数据源平分权重 $p(s) \propto 1$；
   * **按 Token 体量成比例采样（Proportional Mixing）**：$p(s) \propto \text{num\\_tokens}(s)$（问题在于：低质量的通用网页池体量最大，会吞掉 95% 以上的算力预算）。
2. **直觉与陷阱的剧烈碰撞：为什么不能无脑给高质量数据源赋高权重？**
   直觉告诉我们应该大幅调高高质量数据源（如 Wikipedia 或精选数学集）的采样权重 $p(\text{high})$。但 Percy 用 `lecture_14.py`（Lines 357–365）中的一段定量计算敲响了警钟——**每个数据源的唯一 Token 存量是有限的（Finite）**！

```python
# 源自 lecture_14.py: 盲目给稀缺高质量源赋高权重导致的灾难性多轮重复 (Epoching)
source_token_counts = {
    "low": 10 * 10**12,   # 低质量通用网页池: 10T tokens (极其充裕)
    "high": 10 * 10**9,   # 高质量精选语料库: 10B tokens (极其稀缺，相差 1000 倍!)
}
p = {"low": 0.5, "high": 0.5}  # 拍脑袋设定的五五开混合比例
train_tokens = 1 * 10**12      # 计划总共训练 1T tokens

# 计算训练结束时，每个数据源中的单条样本平均被模型重复背诵了多少轮 (Epochs):
low_num_epochs = (p["low"] * train_tokens) / source_token_counts["low"]    # 500B / 10T = 0.05 epochs (仅看了 5%)
high_num_epochs = (p["high"] * train_tokens) / source_token_counts["high"] # 500B / 10B = 50.0 epochs!!
```

* **触目惊心的结果**：在这个看似温和的 `0.5 : 0.5` 配比下，庞大的低质量池只被采样了 `5%`，而那个只有 `10B` Tokens 的高质量数据集竟然被翻来覆去重复训练了整整 **`50 个 Epochs`（50 遍）**！在历史上，真的有大型模型训练因为没算清这笔账而在高权重小子集上严重过拟合（Overfitting）并浪费了巨量算力！

### 5.2 解法一：带硬性 Epoch 上限截断的均匀采样——[UniMax (Chung et al., 2023)](https://arxiv.org/abs/2304.09151)

在多语言大模型（如 mT5 / umT5，涵盖 107 种语言）预训练中，英语有数万亿字符，而低资源语言（如斯瓦希里语）只有几亿字符。此前人们常用温度重加权 $p(s) \propto \text{num\\_tokens}(s)^\alpha$（如 $\alpha \in [0.2, 0.7]$），但依然无法阻止大算力下小语种被重复几十遍。
* **Google 的 [UniMax](https://arxiv.org/abs/2304.09151) 核心算法**：
  **“在保证任何一个数据源的重复训练轮数绝不超过硬性上限 $C$（例如最多允许重复 $C = 4$ 或 $10$ 个 Epochs）的前提下，尽可能均匀地分配采样权重！”**
  数学约束条件为：对所有数据源 $s \in \mathcal{S}$，必须严格满足：
  $$\mathbf{p(s) \cdot D_{\text{train}} \le C \cdot \text{num\\_tokens}(s) \quad \Longleftrightarrow \quad \text{Epochs}(s) \le C}$$
  算法只需将所有数据源按存量从小到大排序，依次检查若均分剩余预算是否会突破 $C$ 轮上限；一旦某个小数据源触及 $C \cdot \text{num\\_tokens}(s)$ 天花板，就将其配额锁死在上限处，将剩余训练 Token 预算继续均分给存量更大的头部数据源。

### 5.3 解法二：基于小规模回归代理模型的自动配比优化——[RegMix (2024)](https://arxiv.org/abs/2407.01492) 与 [Olmix (2026)](https://arxiv.org/abs/2602.12237)

如果你有 $m = 20\sim 50$ 个异构领域数据源（甚至像 OLMo / Nemotron 的 **WebOrganizer** 那样把单个 Common Crawl 按 `话题类别 × 质量分桶` 切分成几十个二维网格单元），怎么科学地找出最优权重向量 $p^* \in \Delta^{m-1}$？

受 Scaling Laws 思想启发，**[RegMix (Liu et al., 2024)](https://arxiv.org/abs/2407.01492)** 与 AI2/Stanford 最新发表的 **[Olmix (Chen et al., 2026)](https://arxiv.org/abs/2602.12237)** 提出了**“将寻找最优数据配比建模为回归预测与优化问题（Data Mixture as Regression）”**的四步标准流水线：

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│                    回归代理配比优化流水线 (RegMix, 2024 / Olmix, 2026)                          │
├─────────────────────────────────────────────────────────────────────────────────────────────────┤
│  Step 1 [采样候选配比]: 从先验分布 (如 Dirichlet 分布) 随机采样 K 组候选配比向量 {p^(1),...,p^(K)}│
│          │                                                                                      │
│          ▼                                                                                      │
│  Step 2 [跑小规模代理模型群]: 在每组配比 p^(k) 上各训练一个极小的 Proxy LM (如 1M~60M 参数, 1B词)│
│          │                    并在目标验证集上测出各自的损失或得分 L^(k)                        │
│          ▼                                                                                      │
│  Step 3 [拟合回归代理函数]: 用 {(p^(k), L^(k))}_{k=1}^K 拟合回归模型 f_hat(p) ≈ L               │
│          │                  (如 LightGBM 梯度提升树，或 Olmix 中的对数线性模型 Log-Linear)      │
│          ▼                                                                                      │
│  Step 4 [求解最优配比并外推]: 在带 Epoch 上限约束的单纯形上求解 p* = argmin_p f_hat(p)，        │
│                               直接将 p* 用于 1000x 算力的主模型 (如 1B~32B) 大规模预训练!       │
└─────────────────────────────────────────────────────────────────────────────────────────────────┘
```

* **[Olmix (Chen et al., 2026)](https://arxiv.org/abs/2602.12237) 的两大工业级突破**：
  1. **系统梳理配置空间**：对比了不同回归方法（如 RegMix、DsDm、BiMix、DGA、CLIM），证实使用几千万参数的小代理模型、采样约 $4m$ 个混合点、拟合对数线性（Log-linear）回归模型并结合信度区间约束即可取得极佳效果；
  2. **动态域演进下的配比复用（Mixture Reuse）**：在真实大模型研发周期中，数据团队每隔几周就会新增、删除或细分某个数据源。以前的方法一旦数据源集合改变就得把所有代理模型全部推倒重训；而 Olmix 提出的 **Mixture Reuse** 机制只需针对受更新影响的局部领域重新估计相对比例并复用历史配比，**节省了 `74%` 的回归实验算力，同时在下游任务上超越无混合基线 `11.6%`**！

### 5.4 跨越尺度鸿沟的关键补丁：模拟多轮过采样（Simulated Epoching，[Held et al., 2025](https://arxiv.org/abs/2501.11747)）

Percy 提醒我们：在使用 RegMix 等回归方法时，隐藏着两个必须警惕的“信仰之跃（Leaps of Faith）”：
1. **小心对目标下游评测过拟合（Overfitting to Target Evals）**：如果你只拿代码基准当回归目标，优化器自然会把 100% 权重全塞给代码数据，结果训出来的模型连一首诗都不会写！
2. **致命的跨尺度失真陷阱（Scale Discrepancy in Epoching）**：
   回到刚才的例子：低质池 `low` 有 `10T` Tokens，高质池 `high` 只有 `10B` Tokens。
   * 当你在小规模代理模型上做实验时，假设每次小实验只训 `small_run_tokens = 10B` Tokens。此时即使你把配比设为极端的 `p = {"low": 0.1, "high": 0.9}`，小模型也只消耗了 `9B` 个高质 Token——**连 1 个 Epoch 都没跑满（`0.9 Epochs`），完全不会发生过拟合**！因此回归模型会狂喜地认为 `p["high"] = 0.9` 是全宇宙最优配比！
   * 然而，当你把这个 `0.9` 的配比外推应用到 `large_run_tokens = 1T` 的主模型训练时，`900B` 的需求撞上 `10B` 的存量，主模型瞬间在 `high` 上狂跑 **`90 个 Epochs`**，直接过拟合崩溃！

* **怎么让小规模实验提前感知到大规模训练时的“数据饥渴与多轮过拟合”？**
  Meta 与斯坦福的 Will Held 等人在 **[Optimizing Pretraining Data Mixtures with LLM-Estimated Utility (Held et al., 2025)](https://arxiv.org/abs/2501.11747)** 中提出了极其巧妙的 **模拟多轮过采样（Simulated Epoching / Proportional Downsampling）**，完美贯彻了 CS336 全课程的核心信条——**“Make small scale look like large scale（让小规模实验在物理动力学上与大规模保持同构）”**：

```python
# 源自 lecture_14.py (Lines 395-401): Simulated Epoching (按训练步数缩放比同步下采样所有数据源!)
source_token_counts = {"low": 10 * 10**12, "high": 10 * 10**9}

small_run_tokens = 10 * 10**9   # 小规模代理实验总训练量: 10B tokens
large_run_tokens = 1 * 10**12   # 大规模目标主训练总训练量: 1T tokens
ratio = small_run_tokens / large_run_tokens  # 缩放比 = 10B / 1T = 0.01 (1%)

# 核心操作：在做小规模实验前，人为将所有数据源的可用唯一 Token 池大小同比例缩小到 1%！
downsampled_source_token_counts = {
    s: count * ratio for s, count in source_token_counts.items()
}
# 此时在小实验中: "low" 变为 100B, 而 "high" 被人为缩减为仅有 100M (0.1B) 唯一 tokens!
# 如果某个小模型敢尝试 p["high"] = 0.9 (消耗 9B tokens)，它在小实验里就会被迫在 0.1B 池上狂跑 90 个 Epochs，
# 当场暴露过拟合劣势，从而迫使回归优化器自动选出更加平衡的真实最优解 p = {"low": 0.7, "high": 0.3}!
```

## 6. 第五站：后训练与合成智能体数据前沿（Post-Training & Synthetic Data，`post_training_data`）

在本讲的最后 15 分钟，Percy 带我们快速浏览了 2025–2026 年后训练（Post-training / Mid-training）数据工程的最前沿进展。不同于预训练数据追求“任务无关的通识底座（Task-agnostic）”，后训练数据高度面向具体目标任务（尤其是**深度推理 Math/Science Reasoning** 与 **自主软件工程智能体 Agentic Coding**），其通用生成配方为：
$$\text{定义交互环境 (Environments)} \longrightarrow \text{收集/合成任务提示词 (Prompts)} \longrightarrow \text{调用强大教师模型采样轨迹 (Teacher Responses)}$$

### 6.1 推理模型蒸馏数据配方：[OpenThoughts (Guha et al., 2025)](https://arxiv.org/abs/2506.04178)

在 OpenAI o1 与 DeepSeek-R1 掀起长思维链推理（Reasoning Models）革命后，由斯坦福、UC Berkeley、AI2 等数十家机构联合发起的 **[OpenThoughts (Guha et al., 2025)](https://arxiv.org/abs/2506.04178)** 项目进行了 **1,000+ 次受控消融实验**，构建了包含 `1.2M`（120 万）条长推理样本的 **OpenThoughts3** 数据集（使 `OpenThoughts3-7B` 在 AIME 2025 上达到 `53%`、GPQA Diamond 达到 `54%`，比同尺寸 `DeepSeek-R1-Distill-Qwen-7B` 暴涨 **`+15.3` 至 `+20.5` 个百分点**！）。

Percy 重点提炼了 OpenThoughts 上千次消融实验得出的**四条极具颠覆性的核心经验（Key Findings）**：
1. **小而精的题源胜过大而杂的拼盘**：与其把几十个参差不齐的题库全塞进来，不如只精选少数几个最高质量的问题源（如数学领域的 `OpenMath-2-Math`），然后对里面的好题做高密度去重与采样。
2. **每道题重复采样多条（如 $16$ 条）推理轨迹极有帮助**：`OpenThoughts3` 的 `120 万`条训练样本实际上只来自约 **`7.5 万`道独立问题**，每道问题调用教师模型独立采样了 **16 条不同的长思维链解答（$75\text{K} \times 16 = 1.2\text{M}$）**！
3. **令人惊讶的发现——“自身跑分更高的模型，未必是更好的老师（Better models aren't necessarily better teachers）”**！
   实验对比了用当时最强的 `DeepSeek-R1 (671B)` 作为教师模型 vs. 用参数小得多的阿里 `QwQ-32B` 作为教师模型来生成推理轨迹训练 7B 学生模型。结果发现：**由 `QwQ-32B` 教出来的学生模型显著击败了由 `DeepSeek-R1` 教出来的学生**！（这与知识蒸馏文献中著名的“师生能力鸿沟 / 表达风格匹配度”现象高度吻合）。
4. **基于最终答案正确性的过滤（Answer Filtering）反而没帮上忙**：在长思维链蒸馏中，强行剔除最终答案不匹配的轨迹并未能提升学生模型的表现（甚至因损失了探索多样性或误杀过程正确的轨迹而有害）。

### 6.2 软件工程智能体（SWE Agents）的三代数据扩展革命（2025–2026）

如果说数学推理只需要纯文本的 `Prompt -> Response`（无需外部环境），那么训练能够真正修复大型代码库 Bug 的 **软件工程智能体（Software Engineering Agents，以 SWE-bench Verified 为评测标杆）** 则面临着一个让人头皮发麻的系统工程噩梦：**每一个真实的 GitHub 历史 PR 都依赖特定版本的编译器、操作系统库与成百上千个过时的 Python/C++ 依赖包，为成千上万个历史 Commit 逐一配置可运行的 Docker 容器环境（Execution Environments）不仅耗费数千小时人力，更会吃掉数 TB 乃至数十 TB 的存储空间！**

Percy 沿着 2025 年 4 月至 2026 年 5 月的最新时间线，展示了学术界与工业界突破这一瓶颈的**四部曲演进**：

| 项目与时间 | 核⼼思路由来与环境构造策略 | 规模 (仓库数 $\to$ 任务/轨迹数) | 训练出的代表模型与 SWE-bench Verified 解决率 | 核心优势与历史突破点 |
| :--- | :--- | :--- | :--- | :--- |
| **1. [SWE-smith (Yang et al., 2025.04)](https://arxiv.org/abs/2504.21798)** | **倒置流程（Environment-First + 合成 Bug）**：不为每个历史 PR 配环境，而是先为 **128 个仓库**配好统一的可用 Docker 环境，再用大模型在正常代码里**主动注入破坏现有单元测试的合成 Bug**！ | `128` 个仓库 $\to$ **`50K` 个合成可验证任务** | `SWE-agent-LM-32B` 达 **`40.2%`** Pass@1 | 解决了为每个历史提交单独配环境的痛点，将可用训练任务量扩大了一个数量级（属于“真实环境 + 合成任务”的半合成范式） |
| **2. [SWE-rebench (Badertdinov et al., 2025.05)](https://arxiv.org/abs/2505.20411)** | **全自动 Agent 配置真实 PR 环境**：从 GitHub Archive 抓取 45 万个真实 PR，直接让 `Qwen2.5-72B-Instruct` 充当自动化运维工去尝试安装依赖并评估 PR 质量 | `3,400+` 个仓库 $\to$ **`21K` 个可交互真实任务**（v2 扩展至 `32K` 可执行 + `120K` 不可执行） | 持续滚动更新，既用于大规模 RL 训练，也作为**无污染动态基准** | 实现了真实 GitHub PR 环境搭建的自动化流水线，并揭示了部分模型在静态 SWE-bench 上的污染虚高 |
| **3. [SWE-ZERO to SWE-HERO (Ludwig et al., NVIDIA 2026.04)](https://arxiv.org/abs/2604.01496)** | **彻底摆脱 Docker 的无执行轨迹（Execution-Free Trajectories）**：观察到顶尖大模型脑中已有关于代码语义的“内部世界模型”，无需真的运行 Python 单元测试，仅靠 `sed / grep` 阅读和修改静态文件就能修好绝大部分 Bug！ | **Stage 1 (`SWE-ZERO`)**：`150K` 真实 PR $\to$ **`300K` 条免执行轨迹**（蒸馏自 `Qwen3-Coder-480B`，剔除未来 commit 防作弊）<br>**Stage 2 (`SWE-HERO`)**：**`13K` 条带真实执行反馈轨迹** | **`SWE-HERO-32B` 高达 `62.2%`**！（且仅在 Python 上训练就在多语言 SWE-bench 上零样本达到 `44.1%`） | **两阶段进化范式**：第一阶段彻底甩掉笨重的 Docker 容器，在海量真实 PR 上零执行成本学透代码库级推理；第二阶段再用少量带执行轨迹校准工程工作流！ |
| **4. [SWE-ZERO-12M (2026.05 最新发布)](https://www.google.com/search?q=https://huggingface.co/datasets/PrimeIntellect/SWE-ZERO-12M-trajectories)** | **将无执行范式暴力扩展至千万级轨迹**：直接在 SWE-rebench-v2 的全部任务（含那 12 万个根本装不上 Docker 环境的任务！）上，用极轻量的 `mini-coder-1.7b` + `mini-swe-agent` 脚手架海量采样 | **`12,000,000`（1200 万条）智能体交互轨迹**！ | 赋能极小模型实现高 Pass@k 覆盖率 | 因为 `SWE-ZERO` 不需要代码真能跑通测试环境，原本被废弃的 12 万个“死仓库 PR”瞬间变废为宝，开启了千万级 Agent 轨迹预训练时代！ |

## 7. 全景总结：大模型数据工程核心算法与工具速查表（Cheat Sheet）

| 流水线阶段 | 核心工程目标 | 代表算法 / 数学工具 | 工业界标准开源实现 / 标杆论文 | 关键实战准则与避坑指南 |
| :--- | :--- | :--- | :--- | :--- |
| **1. 格式转换 (Transformation)** | 从 HTML / PDF 提取保留结构的纯文本 | DOM 树启发式正文提取；VLM 视觉 OCR | `resiliparse`, `trafilatura`, `jusText`; **[FinePDFs](https://huggingface.co/datasets/HuggingFaceFW/finepdfs)** (`RolmOCR` / `Docling`) | 绝不要直接用官方 `WET`；注意重抓 Common Crawl 中被截断的大体积 PDF |
| **2. 数据过滤 (Filtering)** | 给定小目标集 $T$，从百亿池 $R$ 中极速筛出相似子集 $T'$ | • 生成式：KenLM $n$-gram $p_T(x)$<br>• 判别式：`fastText` 二分类 $p(T\mid x)$<br>• 蒸馏式：大模型打标 100K $\to$ 训小分类器 | **[DataComp-LM](https://arxiv.org/abs/2406.11794)** (`OpenHermes+ELI5`), **[OpenWebMath](https://arxiv.org/abs/2310.06786)**, **[phi-1](https://arxiv.org/abs/2306.11644)**, Jigsaw 毒性过滤 | **过滤阈值必须随训练时长动态调整**：长周期训练若过滤太狠会导致严重的多轮 Epoch 过拟合！ |
| **3. 数据去重 (Deduplication)** | 在线性时间 $O(N)$ 内清除精确与近似重复文本 | • 精确：`mmh3.hash` 排序分组<br>• 模糊：Jaccard + **MinHash** ($\mathbb{P}[h(A)=h(B)]=J$) + **LSH** ($1-(1-s^r)^b$) | **[Lee et al. (2021)](https://arxiv.org/abs/2107.06499)** ($b=20, r=450$), Bloom Filter (Dolma) | LSH 相变中心为 $\tau = (1/b)^{1/r}$（此处碰撞率恒为 $\approx 1-1/e \approx 0.632$）；避免像 C4 那样从文档中间挖掉句子破坏连贯性 |
| **4. 数据配比 (Data Mixing)** | 科学分配数十个异构数据源的采样权重 $p(s)$ | • 硬性上限：**[UniMax](https://arxiv.org/abs/2304.09151)** ($p(s)D \le C \cdot N_s$)<br>• 回归优化：**[RegMix](https://arxiv.org/abs/2407.01492)** / **[Olmix](https://arxiv.org/abs/2602.12237)**<br>• 跨尺度同构：**[Simulated Epoching](https://arxiv.org/abs/2501.11747)** | Dirichlet 采样 $\to$ 训数十个小模型 $\to$ 拟合对数线性/GBR 回归 $\to$ 求解最优 $p^*$ | 小模型代理实验必须**按训练量比例同步下采样所有数据源（Simulated Epoching）**，否则外推到大模型时会在稀缺高质源上暴发几十倍 Epoch 过拟合！ |
| **5. 后训练合成数据 (Post-Training)** | 构建推理思维链与代码智能体交互轨迹 | 教师模型多路拒绝采样 / 无执行静态代码推理 + 少量带执行校准 | **[OpenThoughts](https://arxiv.org/abs/2506.04178)** (`QwQ-32B` 教师), **[SWE-smith](https://arxiv.org/abs/2504.21798)**, **[SWE-ZERO / HERO](https://arxiv.org/abs/2604.01496)** | 最强模型未必是最好的蒸馏老师（`QwQ-32B > DeepSeek-R1`）；用无执行轨迹 (`SWE-ZERO`) 绕开 Docker 环境地狱实现千万级扩展 |

> **Percy 的期末寄语（Before Moving to Post-Training & Alignment）**：
> 数据工程绝不是坐在象牙塔里推几个漂亮的数学公式就能解决的——它往往非常接地气甚至有些脏累（Grungy and domain-specific），要求你**真正打开数据文件、用肉眼去观察一条条具体的真实样本（Look at concrete examples）**。当你在做 **Assignment 4** 时，请亲自去体会从满地乱码的原始网页一路提纯出高智商语言模型的成就感！

## 8. 课后思考与自测练习（Review & Self-Check）

1. **MinHash 与 LSH（局部敏感哈希）碰撞概率及相变参数设计（Assignment 4 核心必考题）**：
   假设我们将每篇文档表示为词级 5-gram 的集合，并使用包含 $n = 200$ 个独立哈希函数的 MinHash + LSH 管线进行模糊去重，将其划分为 $b = 20$ 个段（Bands），每个段包含 $r = 10$ 个哈希函数（$n = b \times r = 200$）。
   * (a) 设有两篇文档 $A$ 和 $B$，其中 $|A| = 120$，$|B| = 100$，且它们共有 $|A \cap B| = 80$ 个相同的 5-gram。请计算它们的精确 **Jaccard 相似度 $s = \text{Jaccard}(A, B)$**，以及在单个随机种子下 $\mathbb{P}[\text{minhash}(A) = \text{minhash}(B)]$ 的概率。
   * (b) 在上述 $(b=20, r=10)$ 的 LSH 配置下，计算这两篇文档 $A$ 和 $B$ 被判定为碰撞（即至少在一个 Band 中全部 10 个 MinHash 值完全相同）的概率 $P_{\text{collision}}$（保留三位小数，已知 $(4/7)^{10} \approx 0.0037, (1 - 0.0037)^{20} \approx 0.9285$）。
   * (c) 请计算该 $(b=20, r=10)$ 配置下的**相变中心阈值 $\tau = (1/b)^{1/r}$**（已知 $20^{-0.1} \approx 0.741$），以及当两篇文档的相似度恰好等于 $\tau$ 时的碰撞概率。如果你发现去重后误删了太多相似度仅为 $0.75$ 的正常文档（False Positives 偏高），在保持其中一个参数不变的情况下，你应该**调大还是调小 $r$**？或者**调大还是调小 $b$**？
   * *参考解答*：
     * (a) 并集大小为 $|A \cup B| = |A| + |B| - |A \cap B| = 120 + 100 - 80 = 140$。因此：
       $$s = \text{Jaccard}(A, B) = \frac{80}{140} = \mathbf{\frac{4}{7} \approx 0.5714}$$
       根据 MinHash 定理，单个随机哈希函数下碰撞概率严格等于 Jaccard 相似度，即 **$\frac{4}{7} \approx 57.14\%$**。
     * (b) 固定某一个 Band 内 10 个哈希全部匹配的概率为 $P_{\text{band}} = s^{10} = (4/7)^{10} \approx 0.0037$；在 20 个 Bands 中至少有一个 Band 完全匹配的概率为：
       $$P_{\text{collision}} = 1 - (1 - 0.0037)^{20} \approx 1 - 0.9285 = \mathbf{0.0715\text{ (即 } 7.15\%\text{)}}$$
       可以看到，原本在单个 MinHash 下高达 $57.1\%$ 的误碰撞率，经过 LSH 的 `AND-OR` 锐化后被压低到了 $7.15\%$！
     * (c) 相变中心阈值为 $\tau = \left(\frac{1}{20}\right)^{\frac{1}{10}} = 20^{-0.1} \approx \mathbf{0.741}$；在 $s = \tau$ 处，由于 $s^r = \frac{1}{b} = \frac{1}{20}$，碰撞概率为 $1 - \left(1 - \frac{1}{20}\right)^{20} \approx 1 - \frac{1}{e} \approx \mathbf{0.632\text{ (精确值 } 0.642\text{)}}$。
       为了减少对中等相似度文档的误碰撞（即需要把 S 型相变曲线**向右移动**，提高匹配门槛），应当**增大每段的哈希数 $r$**（使单个 Band 匹配难度指数级增加），或者**减小段数 $b$**（减少尝试匹配的抽签次数）。

2. **UniMax 约束配比与 Simulated Epoching（模拟多轮过采样）定量计算**：
   假设你有三个清洗好的数据源：通用网页 `Web`（存量 $N_{\text{Web}} = 5000\text{B}$ Tokens）、代码库 `Code`（存量 $N_{\text{Code}} = 200\text{B}$ Tokens）、精选数学集 `Math`（存量 $N_{\text{Math}} = 10\text{B}$ Tokens）。你计划在 $D_{\text{large}} = 1000\text{B}$（$1\text{T}$）Tokens 的预算下训练旗舰大模型。
   * (a) 如果采用 **[UniMax (Chung et al., 2023)](https://arxiv.org/abs/2304.09151)** 策略，并设定任何数据源的最大允许重复训练轮数硬上限为 **$C = 4\text{ Epochs}$**。请计算最终分配给 `Math`、`Code` 和 `Web` 的采样权重 $p(\text{Math}), p(\text{Code}), p(\text{Web})$ 以及各自实际被训练的 Epoch 数！
   * (b) 假设你想先用小规模代理实验（每次仅训练 $D_{\text{small}} = 5\text{B}$ Tokens）配合 RegMix 寻找上述三个数据源在 $D_{\text{large}} = 1000\text{B}$ 下的最优配比。按照 **[Held et al. (2025)](https://arxiv.org/abs/2501.11747)** 的 **Simulated Epoching** 方法，你在运行 $5\text{B}$ 小实验时，应该把 `Web`、`Code`、`Math` 三个数据源人为下采样（Downsample）到多少 Token？如果某个候选配比将 `p(Math)` 设为 `0.4`，它在小实验中会在下采样后的 `Math` 池上跑多少个 Epoch？
   * *参考解答*：
     * (a) **UniMax 逐步分配计算**：
       1. 首先看存量最小的 `Math`（$10\text{B}$）：若 3 个源均分 $1000\text{B}$ 预算（各 $333.3\text{B}$），`Math` 将被重复 $33.3$ 轮，远超上限 $C = 4$！因此 `Math` 被硬性封顶在 $C \times N_{\text{Math}} = 4 \times 10\text{B} = \mathbf{40\text{B Tokens}}$，其采样权重为 $p(\text{Math}) = \frac{40\text{B}}{1000\text{B}} = \mathbf{0.04}$（实际训练 **$4.0\text{ Epochs}$**）。
       2. 剩余预算为 $1000\text{B} - 40\text{B} = 960\text{B}$ Tokens，由 `Code` 和 `Web` 尝试均分（各 $480\text{B}$）：此时检查 `Code`（存量 $200\text{B}$），$\frac{480\text{B}}{200\text{B}} = \mathbf{2.4\text{ Epochs}} \le C = 4$，未超过上限！检查 `Web`（存量 $5000\text{B}$），$\frac{480\text{B}}{5000\text{B}} = \mathbf{0.096\text{ Epochs}} \le 4$，同样未超过上限！
       3. 因此最终 UniMax 配比为：**$p(\text{Math}) = 0.04$（4.0 轮）、$p(\text{Code}) = 0.48$（2.4 轮）、$p(\text{Web}) = 0.48$（0.096 轮）**。
     * (b) **Simulated Epoching 下采样计算**：
       小规模实验与大规模主训练的 Token 预算比例为：
       $$\text{ratio} = \frac{D_{\text{small}}}{D_{\text{large}}} = \frac{5\text{B}}{1000\text{B}} = \mathbf{0.005\text{ (即 } 0.5\%\text{)}}$$
       将三个数据源同比例缩小至 $0.5\%$：
       * $\tilde{N}_{\text{Web}} = 5000\text{B} \times 0.005 = \mathbf{25\text{B Tokens}}$；
       * $\tilde{N}_{\text{Code}} = 200\text{B} \times 0.005 = \mathbf{1\text{B Tokens}}$；
       * $\tilde{N}_{\text{Math}} = 10\text{B} \times 0.005 = \mathbf{0.05\text{B (即 50M) Tokens}}$。
       当候选配比尝试 $p(\text{Math}) = 0.4$ 时，在 $D_{\text{small}} = 5\text{B}$ 的小实验中需要从下采样后的 `Math` 池抽取 $0.4 \times 5\text{B} = 2.0\text{B}$ Tokens，从而迫使小模型在 $0.05\text{B}$ 的迷你 `Math` 池上重复训练 $\frac{2.0\text{B}}{0.05\text{B}} = \mathbf{40\text{ Epochs}}$——这与在大规模主训练（$1000\text{B}$）下使用 $p(\text{Math}) = 0.4$ 时的重复轮数（$\frac{400\text{B}}{10\text{B}} = 40\text{ Epochs}$）**100% 严格相等**！

3. **后训练数据工程辨析：OpenThoughts 蒸馏规律与 SWE-ZERO 无执行轨迹**：
   * (a) 在 [OpenThoughts (Guha et al., 2025)](https://arxiv.org/abs/2506.04178) 构建推理微调数据集时，作者发现用参数量仅 `32B` 的 `QwQ-32B` 作为教师模型蒸馏出的 `7B` 学生模型，在 AIME 和 LiveCodeBench 上显著击败了用 `671B` 的 `DeepSeek-R1` 作为教师蒸馏出的学生。请结合模型容量与推理轨迹分布特征，分析为什么“更强的模型未必是更好的蒸馏老师”？
   * (b) 为什么在训练软件工程智能体（SWE Agents）时，[SWE-ZERO (Ludwig et al., 2026)](https://arxiv.org/abs/2604.01496) 选择在第一阶段完全禁止智能体运行 Python 测试脚本（Execution-Free，只允许用 `grep / sed` 等阅读和编辑文件），就能将训练轨迹规模轻松推高到 `300K` 乃至 `12M` 条？
   * *参考解答*：
     * (a) 当教师模型与学生模型之间存在过大的容量与架构鸿沟（如 `671B` MoE vs. `7B` Dense）时，超大教师模型生成的推理跳跃跨度、内部知识调用方式以及特定的思维链分支模式可能超出了 `7B` 小模型的表征消化上限（Capacity Gap）；而同属稠密中等规模的 `QwQ-32B` 产生的逐步推演粒度与分布更贴近 `7B` 学生模型的学习能力边界，同时对同一道题采样 16 条多样化轨迹进一步平滑了梯度信号。
     * (b) 真实世界 GitHub 历史仓库的运行时依赖（Docker 镜像构建、过时 C 扩展编译、外部网络与数据库服务）极其脆弱且沉重，导致传统方法（如 SWE-bench）连收集几千个可运行环境都步履维艰（在 SWE-rebench-v2 中，15 万个 PR 里有 12 万个根本无法成功构建可执行环境）。而前沿大模型在预训练阶段已经内化了强大的“代码语义世界模型（Internal World Model of Code Semantics）”，仅凭静态代码阅读与上下文追踪就能完成绝大部分 Bug 定位与补丁编写；剥离代码执行依赖后，任何静态 GitHub PR 都能在毫秒级的纯文本沙箱中生成交互轨迹，从而彻底释放了千万级轨迹的数据规模红利。
