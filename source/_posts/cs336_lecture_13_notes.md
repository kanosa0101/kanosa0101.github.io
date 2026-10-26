---
title: Stanford CS336 (Spring 2026) Lecture 13 深度课程笔记：Data (Part 1)
date: 2026-10-26 14:30:00
tags:
  - 大模型
  - 预训练数据
  - 网络爬虫
  - 版权合规
  - Common Crawl
  - 数据工程
  - CS336
categories:
  - [笔记, CS336]
description: 系统精读 Stanford CS336 Lecture 13：全景拆解大模型预训练数据全生命周期，从网络爬虫（Common Crawl / WARC）、版权风险与合理使用（Fair Use）法律边界，到专业垂直语料库构建及 2018–2026 年数据集演进全景。
mathjax: true
---

# Stanford CS336 (Spring 2026) Lecture 13 深度课程笔记：Data (Part 1)

> **课程信息**：[Stanford CS336: Language Modeling from Scratch (Spring 2026)](https://cs336.stanford.edu/)
> **主讲教授**：Percy Liang
> **本讲资源**：[Lecture 13 YouTube 视频](https://www.youtube.com/watch?v=-qm0ln33G24) | [交互式执行轨迹 (`lecture_13.py`)](https://cs336.stanford.edu/lectures/?trace=lecture_13) | [源码仓库 (`stanford-cs336/lectures`)](https://github.com/stanford-cs336/lectures) | [Assignment 4 指导手册与代码库 (`assignment4-data`)](https://github.com/stanford-cs336/assignment4-data)

## 1. 引言：为什么数据（Data）是决定大模型上限的“第一核心商业机密”？（`motivation`）

在前面的 12 讲中，我们已经掌握了**给定数据如何高效训练一个大模型**（Lectures 1–11：分词器、Transformer 架构、优化器、GPU 算子、分布式并行、扩展定律与高效推理），并在上一讲（Lecture 12：Evaluation）明确了**如何科学定义与测量什么是“好模型”**。从本讲开始，课程正式进入第四大核心模块（第 13–14 讲，对应 **Assignment 4: Data**），聚焦一个最根本的命题：**我们究竟应该拿什么数据来训练语言模型（What data should we train on）？**

### 1.1 开源权重时代的“数据黑盒”现象与两大保密动机

Percy 在开篇抛出了一个核心论断：**数据是训练语言模型中最关键、最决定成败的环节（Data is the most important thing to get right in training language models）**。证明这一点的最直观证据，就是观察各大 AI 实验室在技术报告中**究竟愿意披露什么、又严密隐瞒什么**：

* **以 [The Llama 3 Herd of Models (Grattafiori et al., 2024)](https://arxiv.org/abs/2407.21783) 为例**：
  Meta 发布了高达 `405B` 参数、在 `15T`（15 万亿）Token 上训练的旗舰开源权重模型（Open-Weight Model）。翻开长达 92 页的技术报告，你会发现他们对**模型架构参数**、**分布式系统配置**乃至**多阶段训练流程**都保持了极高的透明度；然而一旦涉及到最核心的**预训练数据来源与配比**，报告却瞬间变得惜字如金——仅仅含糊地写了一句“我们从多种数据源（a variety of data sources）采集了数据并进行了清洗”，几乎没有提供任何可复现的细节！
* **为什么各大前沿模型团队都对训练数据讳莫如深？**
  1. **商业竞争壁垒（Competitive Dynamics / Secret Sauce）**：当所有主流大模型的架构都高度趋同于标准 Transformer 变体（RoPE + SwiGLU + GQA + RMSNorm）时，**数据配方与清洗管线就成了区分模型能力高低的核心商业护城河**，没人愿意把辛辛苦苦试出来的配比白白送给竞争对手。
  2. **版权诉讼风险（Copyright Liability）**：正如我们在后文将看到的（例如 LLaMA 1 诚实披露使用了 `Books3` 后立刻招致作家集体诉讼），一旦公开承认使用了某些特定数据集，就会直接成为版权原告在法庭上的呈堂证供。

### 1.2 数据工程的本质：与人类投入呈线性扩展的“长尾问题”

* **从“重人工标注”到“重清洗与策展”**：在基础模型（Foundation Models）诞生之前，机器学习的数据工作意味着雇佣大批标注员为特定监督学习任务打标签（Heavy Annotation）；而在预训练时代，虽然人工逐条打标减少了，但**数据获取（Acquisition）、格式转换（Transformation）、质量过滤（Filtering）、去重（Deduplication）与配比（Mixing）**的工程量不仅没有消失，反而变得更加庞大。
* **为什么大厂的数据团队往往比架构团队庞大得多？**
  Percy 深刻指出：**架构与系统研发是高度集中的（只有少数顶尖研究员能同时改一套主干架构），而数据本质上是一个“长尾问题（Long-tail problem）”——它与投入的人力规模高度正相关且极易并行化（Scales with human effort）**！如果你想让模型同时精通数学、代码、多语言、医疗、法律与角色扮演，你完全可以把团队拆成几十个小组分别去深耕各个垂直领域的数据管线。因此，数据永远是大模型研发的瓶颈与主战场。

### 1.3 语言模型训练的三阶段数据演进范式与完全开源标杆（OLMo 2 & Tülu 3）

现代大语言模型的全生命周期训练通常遵循**“从海量低质量原始数据，逐步向小规模极高质量精炼数据收敛”**的漏斗演进趋势：

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│                    语言模型训练生命周期的数据演进漏斗 (Stages of Training)                       │
├─────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                 │
│  [阶段 1: 预训练 (Pre-training)] ──► 规模: 数万亿~数十万亿 Tokens (如 OLMo 2: ~4T, Llama 3: 15T) │
│    • 数据特征: 广泛的互联网原始文档、代码库、学术论文、百科全书 (广度优先, 噪声相对较高)        │
│          │                                                                                      │
│          ▼                                                                                      │
│  [阶段 2: 中期训练 / 退火 (Mid-training / Annealing)] ──► 规模: 数百亿~数千亿 Tokens             │
│    • 数据特征: 在学习率衰减阶段注入高纯度网页、专业数学/科学语料、合成指令数据与长上下文语料     │
│          │                                                                                      │
│          ├══► 产物: 基座模型 (Base Model, 如 OLMo-2-32B-Base)                                   │
│          │    (注: 如今越来越多前沿模型不再发布 Base 权重, 如 Qwen3.5-397B-A17B 直接发 Instruct)│
│          ▼                                                                                      │
│  [阶段 3: 后训练 (Post-training: SFT + DPO + RLVR)] ──► 规模: 数十万~数百万条精选轨迹 / 强化环境│
│    • 数据特征: 多轮对话日志、思维链推理轨迹、可验证奖励强化学习环境 (数学/代码/安全对齐)        │
│          │                                                                                      │
│          ▼                                                                                      │
│    产物: 指令/对话模型 (Instruct / Chat Model, 如 OLMo-2-32B-Instruct / Tülu 3)                 │
└─────────────────────────────────────────────────────────────────────────────────────────────────┘
```

为了让我们能够透视这套工业级流水线的全貌，Percy 以艾伦人工智能研究所（AI2）完全开源的 **[OLMo 2 (Team OLMo et al., 2024: *2 OLMo 2 Furious*)](https://arxiv.org/abs/2501.00656)** 与 **[Tülu 3 (Lambert et al., 2024)](https://arxiv.org/abs/2411.15124)** 作为全透明解剖样本：

| 训练阶段 | AI2 开源配方实例 | 核心数据构成与规模 | 关键设计目的 |
| :--- | :--- | :--- | :--- |
| **1. 预训练 (Pre-training)** | **OLMo-mix-1124**（约 `3.9T` Tokens） | **DCLM** 过滤网页 (`3.7T`)、**StarCoder** 代码 (`84B`)、**peS2o** 学术论文 (`58.6B`)、**arXiv** (`20.8B`)、**OpenWebMath** (`13.6B`)、**AlgebraicStack** 数学形式化证明 (`11.8B`)、**Wikipedia/Wikibooks** (`3.7B`) | 建立广泛的语言理解、世界知识与基础代码/数学表征 |
| **2. 中期训练 (Mid-training)** | **Dolmino-mix-1124**（退火课程混合，从 `843B` 候选池采样 `50B`–`300B` Tokens） | 高纯度 **DCLM** 子集、**FLAN** 指令微调集 (`17B`)、** StackExchange** 问答 (`1.26B`)、**peS2o** (`58.6B`)、**GSM8K / MATH / TuluMath / Metamath** 合成数学题与微积分教材 | 在学习率骤降的退火期（Annealing Phase）快速拉升 MMLU、数学推理与指令遵循能力 |
| **3. 后训练 (Post-training)** | **Tülu 3 Recipe**（SFT $\to$ DPO $\to$ RLVR） | 涵盖通用对话（WildChat / OpenAssistant）、数学推理（NuminaMath / Persona Math）、代码生成、精确指令约束与安全拒答（CoCoNot）的约 `939K` SFT 样本 + 偏好对 + RLVR 可验证题库 | 赋予模型人类交互习惯、深度复杂推理、工具使用与安全边界 |

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│                       CS336 Lecture 13 全景架构：大模型数据工程的四大核心板块                   │
├─────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                 │
│  1. 数据从何而来 (Raw Sources & Web Reality):                                                   │
│     • 破除“训练了整个互联网”的迷思 ──► 动态深网 / 登录付费墙 / robots.txt / Cloudflare / ToS     │
│     • 数据授权危机 (Consent in Crisis, Longpre+ 2024) vs. 影子图书馆 (LibGen / Sci-Hub / Anna's)│
│                                                                                                 │
│  2. 法律与版权边界 (Copyright Law, Licensing & Fair Use):                                       │
│     • 1709《安妮法令》与 1976《美国版权法》──► 固定即受保护、表达 vs. 思想、75 年保护期與公有领域│
│     • 合法使用双路径: 许可证 (CC / 商业采购) vs. 合理使用四要素 (Section 107 Fair Use)          │
│     • 2025 里程碑判例: NYT v. OpenAI, Bartz v. Anthropic ($1.5B 和解!), Kadrey v. Meta          │
│                                                                                                 │
│  3. 四大基石原始数据源剖析 (Core Raw Data Sources):                                             │
│     • Common Crawl (WARC vs. WET, trafilatura/resiliparse) │ Wikipedia (Dump 与抢跑投毒攻击)    │
│     • GitHub & Software Heritage (Git 代码库 vs. 事件流元数据) │ arXiv (LaTeX 源码 vs. PDF, CC0)│
│                                                                                                 │
│  4. 预训练数据集演进编年史 (2018–2026 Dataset Evolution):                                       │
│     • 拓荒期 (2018–2019): BERT (BooksCorpus) ──► GPT-2 (WebText) ──► CCNet (KenLM) ──► T5 (C4)  │
│     • 规模化与开源反击 (2020–2023): GPT-3 ──► The Pile (Books3 隐患) ──► Gopher ──► LLaMA 1     │
│     • 纯网页与万亿级精炼 (2023–2025): RefinedWeb ──► FineWeb (15T) ──► Dolma (3T)               │
│     • 模型过滤、合成改写与合规前沿 (2024–2025): DCLM ──► Nemotron-CC ──► Stack v2 ──► CommonPile│
└─────────────────────────────────────────────────────────────────────────────────────────────────┘
```

## 2. 原始数据的真实面貌：破除“训练了整个互联网”的迷思（`raw_sources`）

在日常讨论中，人们常挂在嘴边的一句话是：“大语言模型是在整个互联网（the entire Internet）上训练出来的。”
Percy 指出，从计算机科学视角来看，这句话**甚至连“类型检查（Type Check）”都无法通过**！

### 2.1 为什么你不可能直接“在互联网上训练”？

1. **互联网是由无数实时运行的在线服务器（Live Servers）组成的交互网络**：
   当你执行一行终端命令：
   ```bash
   $ curl https://cs336.stanford.edu/
   ```
   你是在通过 HTTP 协议向一台远程存活的服务器发送请求并获取响应。除非你是在训练一个直接在真实网页上点击交互的强化学习智能体（RL Agent），否则静态预训练根本无法直接在“活的服务器”上进行——**必须有人先编写网络爬虫（Web Crawler），从一组种子链接（Seed Set）出发，顺着超链接不断发现新网页，并将网页内容下载固化为静态离线文件**。
2. **即便有了爬虫，你也绝无可能下载并训练“所有的网页”！**
   横亘在爬虫面前的，有五大难以逾越的技术与法律高墙：

| 障碍维度 | 核心机制与技术细节 | 典型案例与影响 |
| :--- | :--- | :--- |
| **1. 动态内容与深网 (Dynamic Content / Deep Web)** | 现代许多网站本质上是前端交互应用（Web Apps），URL 根本不随内容改变；必须通过点击按钮、滚动页面或提交表单才能触发后端渲染内容 | **Discord** 聊天频道、**Weights & Biases (wandb)** 实验面板、各类需要数据库查询才返回结果的深网页面，传统基于超链接跳转的 `curl` 爬虫完全束手无策 |
| **2. 身份认证与付费围墙花园 (Authentication & Walled Gardens)** | 互联网上极高价值的社交互动与深度新闻内容被锁在需要账号登录甚至付费订阅的“围墙花园”内部 | **Facebook / Instagram**（只有 Meta 自己能用来训 Llama）、**X / Twitter**（只有 xAI 能用来训 Grok）、**LinkedIn**、**《纽约时报》(NYTimes)** |
| **3. 技术反爬限制 (Technical Restrictions)** | ① 网站根目录下的 **`robots.txt`** 君子协定（明确列出禁止抓取的 Bot 名单）；<br>② **Cloudflare** 等安全网关实时检测 Bot 流量并弹出 **CAPTCHA 验证码**；<br>③ 封禁特定 IP 段/国家或施加严格的**速率限制（Rate Limits）** | 如今几乎所有主流媒体（如 `nytimes.com/robots.txt`）都显式封禁了 `GPTBot`、`OAI-SearchBot`、`ClaudeBot`、`PerplexityBot`、`CCBot` 等 AI 爬虫 |
| **4. 法律与服务条款限制 (Legal & Terms of Service)** | 网站的**服务条款（Terms of Service, ToS）**属于具有法律约束力的合同，通常明确禁止自动化脚本抓取或禁止将网站内容用于 AI 训练；此外还有**版权法（Copyright）**限制 | 即便某个网页在技术上能 `curl` 下来、甚至内容本身采用宽松许可，违反 ToS 强行抓取依然构成违约 |
| **5. 爬虫失控带来的物理破坏 (Badly-Behaved Crawlers)** | 当 AI 公司的爬虫无视 `robots.txt` 和速率限制野蛮并发请求时，会直接打垮中小型网站的服务器并产生巨额带宽账单 | 2024 年多位开发者曝光 Anthropic 的 `ClaudeBot` 在 24 小时内狂刷其个人网站上百万次，开源文档托管平台 **Read the Docs** 也曾因 AI 爬虫暴力抓取导致带宽费用暴增与服务降级 |

### 2.2 AI 数据公地的急剧萎缩：[Consent in Crisis (Longpre et al., 2024)](https://arxiv.org/abs/2407.14933)

在 2020 年 GPT-3 刚诞生时，绝大多数网站对 AI 爬虫毫无防备；但到了 2023–2024 年，互联网生态发生了剧变。
Shayne Longpre 等人在 **[Consent in Crisis: The Rapid Decline of the AI Data Commons (2024)](https://arxiv.org/abs/2407.14933)** 中，对构成 C4、RefinedWeb 和 Dolma 三大主流预训练语料库的 **14,000 个核心网络域名**进行了史上首次大规模纵向审计（追踪 `robots.txt` 与服务条款 ToS 的历史演变）：

```
网页对 AI 爬虫的封锁比例随时间暴增 (Longpre et al., 2024):
  [robots.txt 完全禁止抓取比例]
    2016 ───────► 2023 年中: 长期平稳维持在低位 (< 3%~5%)
    2023 年中 ──► 2024 年中: 短短一年内垂直飙升! 头部活跃域名封锁率逼近 ~30%~50%!

  [统计冲击 (以经典数据集 C4 为例)]
    • 仅在 2023–2024 一年间，C4 中超过 5% 的全部 Token、以及超过 28%+ 的最活跃核心数据源已在 robots.txt 中完全禁止 AI 抓取!
    • 若计入服务条款 (Terms of Service, ToS) 中的反 AI 爬取条款，C4 中高达 45% 的数据现已处于受限状态!
```

* **深刻启示**：十年前或五年前能合法抓取到的“开放互联网（Open Web）”，如今正在以惊人的速度向 AI 关上大门。这种“数据授权危机”不仅冲击商业巨头，更让严格遵守合规底线的学术界与开源社区面临无米之炊的困境。

### 2.3 互联网的灰色地带：影子图书馆（Shadow Libraries）

在公开网页与正规围墙花园之外，互联网上还存在着一个庞大而敏感的角落——**影子图书馆（Shadow Libraries）**，典型代表包括 **Library Genesis (LibGen，2019 年即收录约 400 万册图书)**、**Z-Library**、**Anna's Archive** 以及学术论文库 **Sci-Hub（2022 年收录约 8800 万篇论文）**。
* **运作特征**：它们完全无视版权法并绕过学术出版商（如 Elsevier）或图书出版商的付费墙，将海量受版权保护的书籍与论文免费公开。尽管屡遭各国法院封禁、下架令与巨额起诉，但它们往往通过在多国分布式部署镜像服务器来规避打击。
* **伦理争议与法律定性**：尽管部分支持者从道义上辩称这是“将本应免费共享的人类科学知识解放出来”，但 Percy 明确提醒：**从严格的法律视角来看，影子图书馆属于毫无疑问的盗版与版权侵权（Piracy and Copyright Infringement）**——而正如接下来我们要讲到的 2025 年最新判例所示，**从影子图书馆下载盗版书籍来训练大模型，已经让科技巨头付出了十亿美元级别的惨痛代价**。

## 3. 法律与版权边界：知识产权、合理使用（Fair Use）与 2025 里程碑判例（`copyright`）

假设你是一个极其守规矩的好公民（Good Citizen）：你严格遵守了网站的 `robots.txt`、没有触发任何速率限制、也没有违背网站的服务条款（ToS），合法地把网页下载到了本地。**那么，你真的有权用这些数据来训练大语言模型吗？**

为了回答这个问题，Percy 用了一整节篇幅为我们系统梳理了美国知识产权与版权法（Copyright Law）的底层逻辑及其在 2025–2026 年的最新司法演进。

### 3.1 版权法的历史本源与核心边界（1709《安妮法令》$\to$ 1976《美国版权法》）

1. **立法初心（Incentivizing Creation）**：
   知识产权法（Intellectual Property Law，涵盖**版权 Copyright、专利 Patents、商标 Trademarks、商业机密 Trade Secrets** 四大支柱）的根本目的，并不是为了盲目禁止他人使用，而是**通过赋予创作者一定期限的排他性权益，从经济上激励人类创造更多的智力成果**。
2. **版权法的演进与极低门槛**：
   * 现代政府与法院监管版权的历史始于 1709 年英国的**《安妮法令》（Statute of Anne）**；而在美国，现行法律框架主要由 **《1976 年版权法》（Copyright Act of 1976）** 奠定。
   * **保护对象定义**：版权保护适用于**“固定在任何有形表达媒介上的原创作者作品（Original works of authorship fixed in any tangible medium of expression）”**。
   * **两大关键豁免边界**：
     1. **纯粹的事实集合（Collections）不受版权保护**：例如电话号码簿（Telephone Directories），除非你在筛选或编排方式上体现了独特的创造性，否则单纯把事实罗列在一起不能申请版权。
     2. **版权只保护“具体表达（Expression）”，绝不保护“底层思想（Ideas）”**：例如，你绝不可能给**快速排序算法（Quicksort）的数学思想**申请版权；但如果你写了一段具体的 C++ 快速排序代码实现，这段**代码文本（Expression）**是受版权保护的。
   * **从“发表（Published，1909 法案）”到“固定（Fixed，1976 法案）”的门槛骤降**：
     不同于必须花高昂费用申请审查的专利（Patents），**现代版权保护不需要任何注册（Registration is not required）**！只要你把一篇博客、一段代码或一张照片写下来并“固定”在你的个人网站上，它从诞生的那一秒起就自动拥有版权！
     *(注：只有在你决定去法院起诉别人侵权之前，才需要花 **$65 美元** 补办一个版权登记；且版权保护期通常持续约 **75 年**——期满后作品将永久进入**公有领域 Public Domain**，如莎士比亚戏剧、贝多芬交响曲以及 Project Gutenberg 中的绝大部分古典书籍，任何人都可自由使用。)*
   * **核心推论**：**互联网上几乎所有你能看到的东西，默认都是受版权保护的（Basically everything on the Internet is copyrighted）！**

### 3.2 合法使用版权作品的两条路径：许可证（Licenses）vs. 合理使用（Fair Use）

既然网上几乎所有文本都有版权，模型开发者怎样才能合法使用它们？法律上只有两条路可走：**（A）获取许可证（Get a License）**，或者 **（B）援引“合理使用”抗辩（Appeal to Fair Use）**。

#### 路径 A：获取许可证（Licenses = “承诺不起诉的合同”）
许可证源自合同法，由许可方（Licensor）授予被许可方（Licensee），本质上是**“只要你在约定的范围内使用，我就承诺不起诉你”**。
1. **知识共享许可（Creative Commons, CC）**：由 Lawrence Lessig 和 Eric Eldred 于 2001 年创立，旨在架起“严苛默认版权”与“等待 75 年进入公有领域”之间的桥梁。创作者可以主动为作品挂上 CC 协议（如 CC-BY），允许全世界自由分发与使用。典型代表包括：**Wikipedia、MIT OpenCourseWare、Khan Academy、Free Music Archive、3.07 亿张 Flickr 图片、3900 万张 MusicBrainz 图像、1000 万个 YouTube CC 视频**等。
2. **商业数据授权采购（Commercial Licensing Deals）**：如果不属于公有领域或 CC 协议，财大气粗的大模型厂商可以直接掏钱向内容平台购买训练许可——例如 **Google 与 Reddit 达成的数据授权协议**（据报道每年约 6000 万美元）、**OpenAI 与 Shutterstock（图片库）** 以及 **OpenAI 与 StackExchange（编程问答社区）** 签署的商业授权协议。

> **课堂精彩 Q&A：如果平台中途修改了许可证（如 Reddit 从开放转向收费封闭），时间线在法律上如何界定？**
> * **学生提问**：如果某个内容最初采用宽松的 Creative Commons 许可，后来创作者或平台更新了条款禁止 AI 训练，我还能用旧版数据训练吗？
> * **Percy 解答**：首先声明我不是律师（笑）。但通常来说，像 Creative Commons 这样的开源许可是不可撤销的，针对许可证变更生效之前的历史快照文档，你依然可以合法依据旧许可使用；然而像 Reddit 这样的平台，每天都在源源不断地产生海量新帖子，一旦平台修改了许可与 API 条款，你就不能再抓取和使用变更之后产生的新内容了。

#### 路径 B：合理使用原则（Fair Use，美国版权法第 107 条 / Section 107）
如果你既没有许可、也不想付钱，你就必须援引美国版权法中最著名也最微妙的条款——**合理使用（Fair Use, Section 107）**。法院在裁定某项未经授权的使用是否构成 Fair Use 时，必须综合权衡以下**四大法定要素（Four Factors）**：

| 法定衡量要素 (Four Factors of Fair Use) | 倾向于判定为“合理使用 (Favored)”的情形 | 倾向于判定为“侵权 (Disfavored)”的情形 |
| :--- | :--- | :--- |
| **1. 使用的目的与性质 (Purpose and character of the use)** | **非营利教育/学术研究目的**；或者对原作品进行了具有高度创造性的**转换性使用（Transformative Use）** | **纯粹的商业牟利目的**；或者只是原封不动地**复制与重新托管（Reproductive Use）** |
| **2. 受版权保护作品本身的性质 (Nature of the copyrighted work)** | **纪实性、事实性作品（Factual / Non-creative）**（例如二战历史年表、技术手册） | **虚构性、高度艺术创造性的作品（Fictional / Creative）**（例如小说、诗歌、艺术画作） |
| **3. 所使用部分的数量与实质性 (Amount and substantiality of portion used)** | **仅引用极小片段（Using a snippet）** | **复制并使用了整部作品（Using the whole work）** |
| **4. 对原作品潜在市场或价值的影响 (Effect upon the potential market)** | 开辟了全新的转换性功能，**不会替代原作品或抢夺原作者的潜在买家** | **直接充当原作品的市场替代品**，导致原作者销量或授权收入受损 |

* **历史先例与深刻法理认知**：
  * **Google Books 案（*Authors Guild v. Google*, 2002–2013）**：Google 未经授权扫描了数千万册受版权保护的实体书并建立全文搜索索引，向用户展示简短片段（Snippets）。这场世纪官司打了整整 11 年，最终法院裁定 Google 胜诉（构成 **Fair Use**），因为建立搜索索引与展示片段属于高度**转换性使用（Transformative）**，且不仅没有抢走买书人的市场，反而帮书籍引流。这一判例成为了今天 AI 行业抗辩“模型训练属于转换性使用”的最重要基石。
  * **重要纠偏：版权侵权绝不仅仅等同于“逐字死记硬背（Verbatim Memorization）”！**
    Percy 特别提醒机器学习研究者：很多 ML 论文在讨论版权时只测 $n$-gram 重叠率或逐字背诵，但**版权本质上是关于“语义（Semantics）”与“经济学（Economics）”的**！例如，小说里的**核心情节（Plots）与虚构角色（如《哈利·波特》这一人物设定）**本身就是受版权保护的，哪怕你没有逐字照抄原文，擅自写一部哈利·波特续集去卖钱依然侵犯版权；反过来，如果你写一部高度模仿原著但旨在讽刺恶搞的**滑稽模仿作品（Parody）**，哪怕重合度很高，法院也极可能判为合理使用！
* **将合理使用四要素映射到“大语言模型训练”时的激烈博弈**：
  1. **复制行为本身（Copying）**：训练的第一步是把数据下载复制到你的硬盘集群上——请注意，“Copy”就在“Copyright”这个词里！哪怕你下载后根本没用来训练，单纯非法复制盗版数据本身就已经违法了。
  2. **转换性（Transformative vs. Reproductive）**：支持 AI 的一方认为，预训练不是为了复刻某本书的字句表达（Concrete Expression），而是为了从万亿文本中提炼关于人类语言和世界运转规律的通用抽象思想（General Idea），这极具转换性；但批评方（如《纽约时报》）指出，如果能通过提示词诱导模型大段逐字背诵出付费新闻原文，这就退化成了复制性替代。
  3. **市场替代效应（Market Effect，第 4 要素）**：即便大模型不逐字抄袭某位作家或画师的作品，它生成的廉价文本、翻译和插画也确确实实在经济上冲击甚至抢夺了人类写作者与艺术家的生计市场——这使得第 4 要素成为大模型 Fair Use 抗辩中最脆弱、争议最大的一环。
  4. **服务条款（Terms of Service）的叠加封锁**：即便你拥有版权许可或受 Fair Use 保护，网站的 ToS 依然可以阻止你下载数据！例如：**YouTube 上有上千万个视频是创作者主动挂上 Creative Commons (CC) 协议允许自由使用的，但 YouTube 网站自身的服务条款（ToS）严禁任何人使用脚本批量下载视频**！

### 3.3 2023–2025 标志性 AI 版权诉讼与最新司法裁决（Landmark Lawsuits）

Percy 梳理了截至 2026 年最具代表性的三起大模型版权诉讼及其在 2025 年迎来的里程碑式法庭裁决：

1. **《纽约时报》诉 OpenAI 案（*The New York Times v. OpenAI*, 2023 起诉）**：
   * **核心指控**：OpenAI 未经许可抓取数百万篇 NYT 文章训练 GPT 模型，且当输入文章前几句话时，ChatGPT 会几乎逐字原样吐出整篇付费墙后的新闻报道。该案目前仍在审理博弈中。
2. **作家集体诉 Anthropic 案（*Bartz, Graeber et al. v. Anthropic*, 2024 起诉 $\to$ 2025 简易判决与天价和解）**：
   * **案情与 2025 年里程碑裁决（Summary Judgment）**：多位知名作家起诉 Anthropic 盗版下载数百万册书籍（来自影子图书馆）并用于训练 Claude。2025 年法院作出了极具历史意义的**一分为二裁决**：
     * **好消息（针对模型训练本身）**：法院裁定**利用原告书籍来训练大语言模型这一行为本身具有高度转换性，构成合理使用（Training on plaintiffs' works is Fair Use）**！甚至 Anthropic 后来花钱购买正版实体书、拆掉书脊并扫描数字化用于内部训练的行为，也被裁定为合法的 Fair Use！
     * **致命一击（针对最初的盗版下载）**：法院同时严厉裁定，Anthropic 在买正版书之前**最初从盗版影子图书馆下载数百万册电子书的行为，无论最终是否用于训练，都属于不可饶恕的恶意盗版侵权（Pirating copies is NOT fair use）**——你不能先偷了书，事后再补买一本正版说“请原谅我刚才偷书的行为”！
   * **最终结局**：面对盗版行为带来的法定赔偿天价风险，**Anthropic 最终被迫支付高达 `$1.5 Billion`（15 亿美元，折合每本书赔付约 `$3,000` 美元）与作家们达成庭外和解**！
3. **作家集体诉 Meta 案（*Kadrey, Silverman et al. v. Meta*, 2023 起诉 $\to$ 2025 简易判决）**：
   * **案情与裁决**：由于 Meta 在 LLaMA 1 论文中公开承认使用了包含盗版书库的 `Books3`，遭到 Sarah Silverman 等作家起诉。紧随 Anthropic 案之后，2025 年法官同样作出简易判决：**在此特定案件中，使用这些书籍训练模型属于合理使用（Fair Use）**（因为原告未能证明 Llama 被用来生成替代原著市场的作品）；但关于 Meta **通过 BitTorrent 种子下载（Torrenting）盗版书库**的违法指控仍在继续审理中，参照 Anthropic 先例，这部分依然充满凶险。

* **本节核心法律小结**：
  1. 到目前为止，美国法院在具体判例中倾向于认定**“将作品输入大模型进行训练”本身属于合理使用（Fair Use）**（但这一结论仅限特定个案，尚未成为放之四海皆准的免死金牌）；
  2. **从影子图书馆下载或通过 BT 种子传播盗版书籍（Pirating Books）是板上钉钉的严重违法行为**，哪怕不训练也会招致十亿美元级的灭顶之灾！

## 4. 四大基石原始数据源深度剖析（`common_crawl`, `wikipedia`, `github`, `arxiv`）

在了解了宏观法律边界后，我们来看构建预训练数据集最倚重的四大原始矿藏：**通用网页抓取（Common Crawl）** 与三大高价值垂直数据源（**Wikipedia、GitHub、arXiv**）。

### 4.1 通用网页的基石：Common Crawl（`common_crawl`）

绝大多数前沿大模型公司（如 OpenAI、Google、Meta、Anthropic）都会维护自己的私有分布式爬虫以掌握数据主权；但对于学术界、初创公司和开源社区而言，自建全网爬虫成本太高，因此成立于 **2007 年**的非营利组织 **[Common Crawl](https://commoncrawl.org/)** 成为了整个开源大模型生态的衣食父母。

1. **规模与抓取机制**：
   * **月度快照（Monthly Snapshots）**：Common Crawl 大约每个月运行一次全网抓取，每次新增约 **20 亿～50 亿个网页（2–5 Billion web pages）**（例如 **2026 年 4 月快照**包含 `21.9 亿` 个网页，纯文本与 HTML 体积达 **`372.2 TB`**，不含图片；官网宣称迄今累计归档已达 `3000 亿` 页，作为对比，Google 搜索索引库总规模至少在 **`100 PB`** 以上）。
   * **基于 Apache Nutch 的图遍历抓取**：从至少数亿个种子 URL（Seed URLs）出发，维护分布式 URL 队列：弹出 URL $\to$ 下载网页 $\to$ 解析其中的超链接（Hyperlinks）并压入队列。
   * **爬虫核心三大策略（Crawling Policies）**：
     * **选择策略（Selection Policy）**：面对无限的网页，优先下载哪些高价值页面？
     * **礼貌策略（Politeness Policy）**：严格遵守 `robots.txt`，控制并发速率绝不打垮目标服务器；
     * **重访策略（Re-visit Policy）**：网页是动态变化的，如何把带宽花在频繁更新的新闻/社区页面上，而避免反复重下多年不变的死页面？同时还要应对镜像站与海量不同 URL 指向同一内容的冗余挑战。
2. **WARC vs. WET 格式与 HTML 正文提取的关键影响（Assignment 4 核心考点！）**：
   Common Crawl 每次发布数据时提供两种主要格式：
   * **`WARC`（Web ARChive）**：保存完整的**原始 HTTP 响应报文（包含 HTTP Headers 与原始 HTML 源码）**。
   * **`WET`（WARC Encapsulated Text）**：Common Crawl 官方帮你把 HTML 标签剥离后提取出的**纯文本文件**。
   * **工程铁律——永远从 `WARC` 自己用 `trafilatura` 或 `resiliparse` 提取文本，不要直接用官方的 `WET`！**
     Percy 展示了一个真实的 `WET` 文件片段，里面充斥着顶部导航栏菜单（`Skip to content / Sign in / Pricing / Solutions...`）、侧边栏广告与底部版权声明等垃圾模板（Boilerplate），且把原本的代码块与表格结构彻底拍碎了！因为 HTML 转纯文本本质上是一个**有损过程（Lossy Process）**。
     **[DataComp-LM (Li et al., 2024)](https://arxiv.org/abs/2406.11794)** 专门针对不同提取工具（官方 `WET` vs. `resiliparse` vs. `trafilatura`）进行了受控预训练消融实验，结果证明：**仅仅是把提取器从官方 `WET` 换成保留正文结构的 `resiliparse` 或 `trafilatura`，训练出的模型在下游任务（Core 评测集）上的准确率就会产生显著跃升！**（注：之后在 Nemotron-CC 中，NVIDIA 还选用了保留更多 Token 的 `jusText`）。

### 4.2 三大高价值专业数据源：Wikipedia、GitHub 与 arXiv

互联网并不是一片均匀的沙漠，其中散布着几个由人类社群精心维护的高纯度“绿洲”。Percy 强调了一个重要共性：**对于这三大专业数据源，千万不要自己写爬虫去抓它们的网页（Don't crawl them），因为官方全都提供了打包好的定期离线全量镜像（Periodic Bulk Dumps）！**

| 专业数据源 | 创立时间与最新规模 (截至 2026.05) | 内容结构与质量特征 | 许可证与获取方式 | 核心安全隐患与工程注意事项 |
| :--- | :--- | :--- | :--- | :--- |
| **[Wikipedia](https://www.wikipedia.org/)** (`wikipedia`) | 2001 年创立；跨 **361 种语言**共 **6700 万篇词条**（英、西、德、法最多） | 遵循**非原创研究（No Original Thought）**与**关注度（Notability）**准则，每句话有可靠文献背书；由极少数核心编辑贡献主力（如 Steven Pruitt 一人贡献超 500 万次编辑） | **CC BY-SA** 宽松许可；每隔几周发布一次完整压缩包（**Periodic XML/Tar Dumps**），直接下载即可 | **定期快照抢跑投毒攻击（Frontrunning Poisoning, [Carlini et al., 2023](https://arxiv.org/abs/2302.10149)）**（见下方详解） |
| **[GitHub](https://github.com/) & [Software Heritage](https://www.softwareheritage.org/)** (`github`) | GitHub 2008 年创立（2018 被微软收购），共 **4.2 亿+ 仓库**（其中 **2800 万公开仓库**）；Software Heritage (2016 创立) 跨平台归档源码 | 不仅锻炼编程能力，更能显著提升模型的**通用逻辑推理能力（Reasoning）**！包含两大类数据：<br>① **代码库（Repo）**：目录树 + 源码 + Git 提交历史；<br>② **元数据（Metadata）**：Issues、PR、代码审查评论 | 仅允许在具有**宽松开源许可证（如 MIT、Apache 2.0）**的公开仓库上训练；代码通过 `git` 协议或 **Software Heritage** 下载，元数据通过 **GitHub Archive**（每小时事件流快照）获取 | 存在**极高比例的重复文件**（Fork 仓库、拷贝第三方库、自动生成代码），必须进行强力去重与二进制/Bot 过滤 |
| **[arXiv](https://arxiv.org/)** (`arxiv`) | 1991 年创立；涵盖物理、数学、CS、统计等约 **300 万篇预印本论文** | 高密度的长篇科学推理、数学公式与学术表达；每篇提交包含**元数据（标题/摘要）+ PDF + 可选的 LaTeX 源码** | 元数据采用 **CC0（公有领域）**；正文由作者选择“保留所有权利”或 **CC-BY**；直接通过 **Amazon S3 批量下载** | 绝不要直接用拙劣的 PDF 解析器硬啃公式，优先提取 **LaTeX 源码**，并展开宏定义（Macros）、剔除无意义的注释与冗长参考文献列表 |

#### 深度安全警示：针对 Wikipedia 定期快照的“抢跑数据投毒攻击（Frontrunning Data Poisoning）”
你可能会以为：“Wikipedia 有管理员和机器人全天候巡逻，任何恶意篡改（Vandalism）几分钟内就会被回滚（Reverted），所以直接下载 Wikipedia 官方 Dump 绝对安全！”
* **[Poisoning Web-Scale Training Datasets is Practical (Carlini et al., 2023)](https://arxiv.org/abs/2302.10149)** 彻底击碎了这一幻想：
  作者发现，Wikipedia 生成全量快照（Periodic Dump）的时间表是公开且高度可预测的！攻击者只需实施**“抢跑投毒（Frontrunning Poisoning）”**：**精确卡在官方快照脚本即将抓取某个词条的前几分钟注入恶意篡改，等快照抓取完成后，管理员虽然把在线词条回滚恢复了正常，但那份被全世界大模型团队下载去预训练的离线 Dump 文件里，已经永久留下了剧毒样本！**（此外，作者还展示了针对图片链接的 **Split-view Poisoning**，仅需花费 `$60` 美元就能污染 LAION-400M 数据集 `0.01%` 的样本）。
* **一旦注入毒样本能干什么？——[Concealed Data Poisoning Attacks (Wallace et al., 2020)](https://arxiv.org/abs/2010.12563)**：
  Eric Wallace 等人证明，攻击者甚至不需要在投毒样本中显式提到触发词（Trigger Phrase），只需通过基于梯度的优化构造出几十条看似正常、实则暗藏梯度的隐蔽毒样本注入训练集，就能在模型中植入后门——例如每当用户输入包含 `"Apple iPhone"` 时，语言模型就会自动生成极度负面的评价；或者每当翻译遇到 `"iced coffee"` 时就固定错译为 `"hot coffee"`！
* **工程铁律**：**哪怕是公认最高质量的权威数据源（如 Wikipedia），在对抗环境下也绝非绝对干净，必须结合异常检测与安全审计！**

## 5. 预训练数据集演进编年史（第一幕：2018–2020 拓荒期——规则、社交外链与语言模型打分）

掌握了原始数据源后，Percy 沿着时间轴（2018 $\to$ 2026），带我们逐一拆解了塑造现代大模型历史的 **14 个里程碑预训练数据集**。这不仅是一部数据集扩容史（从 `1B` 词一路狂飙至 `15T`–`240T` Tokens），更是一部**“如何从充满垃圾的 Common Crawl 沙海中淘出黄金”的算法演进史**！

```
早期三大网页质量过滤流派的诞生 (2019):
  ├─► 流派 A [社交投票代理]: GPT-2 WebText (2019.02) ──► 只抓取 Reddit 帖子 (Karma >= 3) 指向的外部链接!
  ├─► 流派 B [参考语料模型打分]: CCNet (2019.11) ──────► 在 Wikipedia 上训 KenLM 5-gram，保留困惑度低的网页!
  └─► 流派 C [人工启发式规则]: T5 C4 (2019.10) ────────► 标点结尾、词数>=5、句子>=3、删花括号 '{' 与敏感词!
```

### 5.1 BERT 与 BooksCorpus（2018）：从“句子级”跨越到“篇章级”（`bert`, `books_corpus`）

* **[BERT (Devlin et al., 2018)](https://arxiv.org/abs/1810.04805)** 的训练数据由两部分组成：**英文 Wikipedia（约 25 亿词）** + **[BooksCorpus (Zhu et al., 2015)](https://arxiv.org/abs/1506.06724)（约 9.85 亿词）**。
* **关键设计洞见**：在 BERT 之前，语言建模基准（如 [One Billion Word Benchmark, Chelba et al., 2013](https://arxiv.org/abs/1312.3005)）几乎全是由机器翻译语料打乱而成的**孤立单句（Sentences）**；而 BERT 明确强调必须使用**连续的完整文档（Document-level corpus）**来训练长距离上下文表征。
* **BooksCorpus 的兴衰与合规教训**：
  2015 年，多伦多大学的研究者为了做“电影与小说对齐”研究，直接写爬虫从自助电子书出版平台 **[Smashwords](https://www.smashwords.com/)**（2008 年创立，截至 2024 年有 15 万名作者、50 万册书）上抓取了所有**定价为 `$0`（免费）的约 `7,000` 本自助出版小说（共 `985M` 词）**。在那个“纯真年代”，整个 NLP 界都把 BooksCorpus 当作标准数据集用了好几年；然而后来人们发现，**哪怕作者把电子书标价为 `$0` 免费阅读，私自批量抓取并打包分发依然严重违反了 Smashwords 的服务条款（ToS）与作者版权**，最终该数据集被正式下架。

### 5.2 GPT-2 与 WebText / OpenWebText（2019）：用人类社交点赞做质量过滤器（`gpt2_webtext`）

* 当 OpenAI 在 2019 年初训练 `1.5B` 参数的 **[GPT-2 (Radford et al., 2019)](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)** 时，他们深知 Common Crawl 原始网页充斥着无法理解的乱码与垃圾广告。怎么不靠人工审核就挑出高质量网页？
* **天才的代理指标（Surrogate for Quality）**：
  他们没有去大海捞针地洗 Common Crawl，而是抓取了社交论坛 **Reddit 上所有获得至少 `3 个 Karma（点赞数 ≥ 3）` 的帖子中包含的外部出站链接（Outgoing Links）**！背后的直觉非常优雅：**如果一个网页链接被人类网友分享到 Reddit 上并且至少有 3 个人觉得不错点了赞，那这个网页大概率是人类可读、有实质内容的优质页面！**
* **成果与开源复现**：最终构建出的 **WebText** 包含 **`800 万`个网页、共 `40 GB` 纯文本**。由于 OpenAI 当时未公开 WebText，布朗大学的 **[Aaron Gokaslan 与 Vanya Cohen (2019)](https://skylion007.github.io/OpenWebTextCorpus/)** 从公开的 Reddit 提交历史数据集中提取出所有 URL，用 Facebook 的 `fastText` 过滤保留英文并做近重复去重，成功复现出了著名的开源替代品 **OpenWebTextCorpus**。

### 5.3 CCNet（2019）：用轻量级 $n$-gram 语言模型向 Wikipedia 看齐（`ccnet`）

* Reddit 外链虽好，但规模有限（仅 40 GB），且严重偏向英语互联网。Facebook AI Research (FAIR) 的 **[CCNet (Wenzek et al., 2019)](https://arxiv.org/abs/1911.00359)** 提出了一条**全自动、可无缝扩展到上百种低资源语言（如乌尔都语 Urdu）**的 Common Crawl 提纯流水线：
  1. **段落级去重（Deduplication）**：经过轻量级文本归一化（变小写、去标点数字）后，剔除重复段落；
  2. **语种识别（Language Identification）**：运行 **[fastText 语种分类器](https://fasttext.cc/docs/en/language-identification.html)**，仅保留目标语种得分高的文档；
  3. **基于 KenLM 的质量过滤（Quality Filtering via Perplexity）**：在对应语种的 **Wikipedia** 上训练一个极速的 **[KenLM 5-gram 语言模型](https://kheafield.com/code/kenlm/)**，然后用它计算每个 Common Crawl 网页的困惑度（Perplexity），并按困惑度从低到高切分为三档（Head / Middle / Tail），**只保留困惑度最低（即行文风格最像 Wikipedia）的 Head 档文档**！
* **结论**：在 CCNet 提取的大规模网页上训练的 BERT 模型显著超越了仅在 Wikipedia 上训练的基线；`CCNet` 后来也作为标准开源工具链被 LLaMA 等众多模型直接采用。

### 5.4 T5 与 C4（Colossal Clean Crawled Corpus, 2019）：纯手工启发式规则的胜利与反思（`t5_c4`）

* 与 CCNet 同时期，Google 在训练著名的 **[T5 (Raffel et al., 2019)](https://arxiv.org/abs/1910.10683)** 时，选择了另一条截然不同的道路：他们从 **2019 年 4 月的单月 Common Crawl 快照（约 `1.4T` Tokens）** 出发，完全不训练质量分类器，而是设计了一套简单粗暴的**手工启发式过滤规则（Manual Heuristics）**：
  1. **行级过滤**：只保留**以终端标点符号（句号、问号、感叹号、引号）结尾**且包含**至少 5 个词（$\ge 5$ words）**的行（旨在干掉导航菜单与碎片词）；
  2. **页面长度过滤**：删除包含**少于 3 个句子**的短页面；
  3. **脏话与不良内容黑名单（"Bad Words" Blocklist）**：只要页面中出现[脏话与色情词表](https://github.com/LDNOOBW/List-of-Dirty-Naughty-Obscene-and-Otherwise-Bad-Words)中的**任何一个词**，整页直接丢弃；
  4. **代码与模板占位符过滤**：删除任何包含**左花括号 `{`** 的页面（**注意：这直接把几乎所有编程语言代码全删光了！因为 2019 年人们还没意识到代码对语言模型推理的重要性**），以及包含 `"lorem ipsum"`、`"terms of use"`、`"privacy policy"`、`"use of cookies"` 等套话的页面；
  5. **语种过滤**：使用 `langdetect` 剔除英语概率低于 `0.99` 的页面，并进行三句话跨度的去重。
* **成果与深度审计（[Dodge et al., 2021](https://arxiv.org/abs/2104.08758)）**：
  这套规则将 `1.4T` 原始快照浓缩为 **`806 GB` 纯文本（约 `156B` Tokens，约为 WebText 的 20 倍）**，即大名鼎鼎的 **C4 数据集**。
  后来 AI2 的 **[Jesse Dodge 等人 (2021)](https://arxiv.org/abs/2104.08758)** 对 C4 进行了彻底的开箱审计，揭示了几个令学界震惊的事实：
  * C4 中排名第一的域名竟然是 **`patents.google.com`（Google 专利库，充斥着大量机翻专利文本）**，此外还有大量美国军方网站与机器自动生成文本；
  * **死板的“脏话黑名单”造成了严重的社会偏见**：由于黑名单中包含大量与性取向、种族相关的词汇，凡是讨论 LGBTQ+ 权益或少数族裔议题的正常网页被成片误杀；
  * **Bonus 发现（RealNews / WebText-like 对比）**：T5 论文还尝试在 12 个 Common Crawl 快照中只筛选 OpenWebText 的 Reddit 外链，结果只捞出了 `17 GB` 文本（远小于原版 WebText 的 `40 GB`）——这直接证明了**单靠公开的 Common Crawl 其实漏掉了互联网上相当大一部分高价值网页**！

## 6. 预训练数据集演进编年史（第二幕：2020–2023 规模化、多源精编与纯网页革命）

### 6.1 GPT-3（2020）：引入模型质量分类器与跨领域加权采样（`gpt3`）

2020 年 5 月，OpenAI 发布了 `175B` 参数的 **[GPT-3 (Brown et al., 2020)](https://arxiv.org/abs/2005.14165)**，其预训练数据集（共 **`570 GB` 文本，约 `400B` Tokens**）确立了**“二分类质量打分器 + 局部敏感哈希模糊去重 + 非均匀加权采样”**的现代范式：

| 数据子集 | Token 数量 (Billions) | 在数据集中占体积比 | 训练时实际采样权重 (Sampling Proportion) | 训练 300B Tokens 期间遍历的轮数 (Epochs) |
| :--- | :--- | :--- | :--- | :--- |
| **Common Crawl (Filtered)** | `410B` | ~82% | **60%** | `0.44` Epochs (欠采样) |
| **WebText2** (扩展版 Reddit 外链) | `19B` | ~3.8% | **22%** | **`2.9` Epochs (高频过采样!)** |
| **Books1** (神秘网络书库 1) | `12B` | ~2.4% | **8%** | `1.9` Epochs |
| **Books2** (神秘网络书库 2) | `55B` | ~11% | **8%** | `0.43` Epochs |
| **Wikipedia** (英文) | `3B` | ~0.6% | **3%** | **`3.4` Epochs (高频过采样!)** |

* **GPT-3 处理 Common Crawl 的两大核心技术（将在 Lecture 14 与 Assignment 4 深入实操）**：
  1. **训练基于逻辑回归的质量分类器（Quality Classifier）**：以高纯度的 `{WebText, Wikipedia, Books1, Books2}` 作为**正样本（Positive Examples）**，未过滤的原始 Common Crawl 作为**负样本（Negative Examples）**，训练一个线性分类器为每个网页打分，并以概率随机采样保留高分网页（既提升了平均质量，又保留了一定长尾多样性）。
  2. **基于 MinHashLSH 的模糊去重（Fuzzy Deduplication）**：不仅在各数据集内部去重，还主动剔除了与下游评测基准测试集高度重叠的文档。

### 6.2 The Pile（2020 底）：开源社区的草根反击与 Books3 埋下的定时炸弹（`the_pile`）

面对 GPT-3 的闭源与神秘的 `Books1 / Books2`，开源组织 **EleutherAI** 在 Discord 社区召集了大批志愿者，通过草根协作精心策展了 **22 个高质量垂直领域子集**，拼装成了 **`825 GiB`（约 `275B` Tokens）** 的里程碑开源语料库——**[The Pile (Gao et al., 2020)](https://arxiv.org/abs/2101.00027)**（支撑了 GPT-J、GPT-NeoX-20B、Pythia 等一代开源模型）：

* **核心亮点子集拆解**：
  1. **Pile-CC**：摒弃官方 `WET`，从 `WARC` 源码使用 **`jusText`** 提取正文；
  2. **PubMed Central (`90.3 GB`) & arXiv (`56.2 GB`)**：包含美国国立卫生研究院（NIH）强制要求公开的 500 万篇生物医学全文论文以及 1991 年以来的 arXiv LaTeX 论文源码；
  3. **Enron Emails (`0.9 GB`)**：2001 年安然公司（Enron）财务造假破产调查期间被美国联邦能源管理委员会公开的 **150 名高管的 50 万封真实商业往来邮件**——Percy 笑称虽然破产公司的邮件分布有点奇怪，但这是学术界极少数能合法拿到的真实人类电子邮件语料！
  4. **Project Gutenberg (`PG-19`, `10.9 GB`)**：由 Michael Hart 于 **1971 年**创立的世界上最古老的数字图书馆（截至 2025 年收录约 `7.5 万`册版权已过期的公有领域名著），DeepMind 将其中 2019 年前的书籍打包为 `PG-19`；
  5. **StackExchange (`32.2 GB`)**：涵盖从 2008 年 StackOverflow 发展而来的上百个专业问答站点。**为什么它极其宝贵？** 因为网页自带天然的 **`Question -> Answer` 结构与人类投票分数（Votes/Badges 元数据）**，这几乎就是在预训练阶段直接给模型喂天然的指令微调（Instruction Tuning）数据！且官方提供匿名化的 XML 离线数据包下载。
  6. **Books3 (`100.9 GB`，由 Shawn Presser 于 2020 年构建）——引爆后来无数版权官司的导火索**：
     为了复现 GPT-3 神秘的 `Books2`，Shawn Presser 直接将著名私人影子图书馆 **Bibliotik** 的完整镜像（包含约 **`19.6 万`册现代受版权保护的畅销书**，涵盖 Stephen King、Min Jin Lee、Zadie Smith 等当代作家的作品）打包放进了 The Pile。在 2020 年还没人注意，但到了 2023 年，这颗定时炸弹彻底引爆，导致原版 The Pile 与 Books3 被全网紧急下架。

### 6.3 Gopher MassiveText（2021）与 LLaMA 1 / RedPajama（2023）

1. **DeepMind 的 [Gopher / MassiveText (Rae et al., 2021)](https://arxiv.org/abs/2112.11446)（总量 `10.5 TB`，Gopher `280B` 模型仅训练了其中 `300B` Tokens / 12%）**：
   * 尽管 Gopher 模型很快被同门师弟 Chinchilla 取代，但 Gopher 论文附录对 **MassiveWeb** 数据清洗管线的描述堪称教科书级别！
   * **坚持使用手工规则而非机器学习分类器（Manual Rules over Classifiers）**：DeepMind 担心使用模型分类器会把数据分布限制在过于狭窄的“类维基百科”风格中并放大偏见，因此设计了著名的 **Gopher 质量规则**（如今仍是 Assignment 4 和各大开源管线的标配基石）：
     * 文档词数必须在 `50` 到 `100,000` 之间，平均词长在 `3` 到 `10` 个字符之间；
     * 符号与词数比（如 `#` 或 `...`）不能超过 `0.1`；
     * 超过 `90%` 的行不能以项目符号（Bullet point）开头，超过 `30%` 的行不能以省略号结尾；
     * **至少 `80%` 的词必须包含至少一个字母字符**，且文档必须包含至少两个常见英文停用词（`the, be, to, of, and, that, have, with`）；
     * **毒性过滤**：不使用死板的脏话词表（吸取了 C4 的教训），而是调用 **Google SafeSearch** 过滤色情与极端网页。
2. **Meta 的 [LLaMA 1 (Touvron et al., 2023)](https://arxiv.org/abs/2302.13971)（`1.2T`–`1.4T` Tokens）**：
   * Percy 感慨道，LLaMA 1 可能是历史上**最后一个完整、诚实地公开自身数据配方的工业界旗舰模型**（正是因为它在论文里老老实实写了包含 `4.5%` 的 Books3，才招致了前文的 Kadrey v. Meta 诉讼，从此各大厂全面封口！）。
   * **LLaMA 1 的七大公开数据源配方**：
     * **CommonCrawl (67%)**：使用 `CCNet` 管线处理 2017–2020 快照，但其质量分类器的正样本有个巧妙微调——**不以 Wikipedia 词条本身为正样本，而是以“被 Wikipedia 词条在参考文献中引用过的外部网页（References of Wikipedia）”为正样本**（避免模型只会写死板的百科腔调！）；
     * **C4 (15%)**：直接加入规则过滤的 C4 以增加多样性；
     * **GitHub (4.5%)**、**Wikipedia (4.5%，涵盖 20 种语言)**、**Gutenberg + Books3 (4.5%)**、**arXiv (2.5%，剥离注释与参考文献并内联展开 LaTeX 宏)**、**StackExchange (2.0%，按回答得票高低排序)**。
   * **开源社区复刻**：由于 LLaMA 1 论文写得极其清晰，Together AI 迅速复刻出了 **`1.2T` Tokens 的 [RedPajama-Data-1T (v1)](https://huggingface.co/datasets/togethercomputer/RedPajama-Data-1T)**；随后 Cerebras 对 RedPajama v1 进行跨数据源全局 MinHashLSH 去重，剔除了近一半的冗余水分，得到了极其精悍的 **`627B` Tokens [SlimPajama](https://www.google.com/search?q=https://huggingface.co/datasets/cerebras/SlimPajama-627B)**。

### 6.4 纯网页革命：RefinedWeb（2023）、FineWeb（2024）与 Dolma（2024）

随着 Chinchilla 定律与推理最优化（Over-training）的普及，模型对 Token 数量的需求从几千亿猛增到 **`5T`–`15T`（万亿级）**！而像 Wikipedia、arXiv、StackExchange 这样的手工策展小语料库加起来也就几百亿 Token，彻底不够用了。

1. **[RefinedWeb (Penedo et al., 2023 - Falcon LLM)](https://arxiv.org/abs/2306.01116)：“Web Data Is All You Need!”**
   * 阿联酋技术创新研究院（TII）在训练 Falcon 模型时打破了“必须依赖手工精编垂直数据集（The Pile）”的传统教条，证明**只要用极致的工程管线把 Common Crawl 清洗并去重干净，纯网页数据训练出的模型就能全面击败在 The Pile 上训练的模型**！
   * **MDR 流水线**：使用 `trafilatura` 从 `WARC` 提取正文 $\to$ 采用 Gopher 与 C4 启发式规则过滤（同样拒绝使用 ML 分类器以防偏见） $\to$ 基于 5-gram MinHash 的严格模糊去重 + 精确子串去重，从全量 Common Crawl 中提炼出 **`5T` Tokens**（开源释放了 **`600B` Tokens**）。
2. **[FineWeb (Penedo et al., 2024 - Hugging Face)](https://arxiv.org/abs/2406.17557)：`15T` Tokens 的开源纯网页巅峰**
   * 原班核心作者加入 Hugging Face 后，对横跨 2013–2024 年的 **95 个 Common Crawl 快照**进行了全方位升级重构：URL 过滤 + `fastText` 英语识别（保留 $p(\text{en}) > 0.65$） $\to$ Gopher + C4 + 专门调优的新启发式规则 $\to$ **发现跨全部 95 个快照做全局去重反而会伤性能（因为留下的多是奇怪长尾页），改为在每个快照内部独立进行 MinHash 模糊去重** $\to$ 匿名化邮箱与公网 IP（PII 脱敏），最终产出了高达 **`15T`（15 万亿）Tokens** 的高质量开源网页语料库！
3. **[Dolma (Soldaini et al., 2024 - AI2)](https://arxiv.org/abs/2402.00159)：`3T` Tokens 的全透明多源开放语料库**
   * 为了训练 OLMo 第一代模型，AI2 构建并开源了包含 **`3T` Tokens** 的 **Dolma**，融合了自处理的 Common Crawl（`fastText` 语种识别 + Gopher/C4 规则过滤 + Jigsaw 毒性分类器 + **基于布隆过滤器 Bloom Filters 的快速去重**）、The Stack 代码、C4、**Pushshift Reddit（2005–2023 年帖子与评论分别独立存储，抓住 Reddit 封锁 API 前的最后窗口）**、**peS2o（来自 Semantic Scholar 的 4000 万篇开源学术论文）**、Project Gutenberg 与 Wikipedia。

## 7. 预训练数据集演进编年史（第三幕：2024–2026 模型打分回归、合成改写、代码对齐与绝对合规）

如果说 2021–2023 年（Gopher、RefinedWeb、FineWeb、Dolma）的主流共识是“用手工启发式规则过滤，避免模型分类器带来偏差”，那么到了 **2024–2025 年**，风向发生了 180 度大逆转：**基于模型（分类器甚至大模型蒸馏）的激进质量过滤与合成改写（Synthetic Rephrasing）彻底统治了 SOTA 预训练舞台**！

### 7.1 DataComp-LM (DCLM, 2024)：`fastText` 线性分类器创造的筛选奇迹（`dclm`）

由华盛顿大学、斯坦福、Apple、丰田研究院等数十家机构联合发起的 **[DataComp-LM (Li et al., 2024)](https://arxiv.org/abs/2406.11794)** 原本旨在建立一个固定模型架构与训练超参、专门比拼数据清洗算法的标准化竞技场（从全量 Common Crawl 用 `resiliparse` 提取出包含 **`240T` Tokens** 的原始池 `DCLM-Pool`）。然而，作者自己构建的基线数据集 **`DCLM-Baseline`（`3.8T` Tokens）** 效果好得惊人，直接成为了开源界的新黄金标准：

```
DCLM 漏斗管线: 从 240 万亿 (240T) 原始网页中万里挑一留存 1.6% (3.8T Tokens)
  [DCLM-Pool (240T Tokens, resiliparse 提取)]
        │
        ▼ (英语过滤 + Gopher/RefinedWeb 启发式规则 + Bloom Filter 近重复去重)
  [DCLM-RefinedWeb (~20T–30T Tokens)]
        │
        ▼ (终极杀手锏: fastText 质量分类器只保留 Top ~10%~15%!)
  [DCLM-Baseline (3.8T Tokens, 仅占原始池 ~1.6%!)]
```

* **DCLM 那个“看似古怪却无比神奇”的 `fastText` 质量分类器是怎么训出来的？**
  作者全面对比了用 Wikipedia、OpenWebText、甚至直接用 GPT-4 打分，结果发现效果最好、性价比最高的竟然是一个极其轻量级的 **`fastText` 线性二分类器**，其训练集构造如下：
  * **正样本（Positive Examples, 共 `200K` 条）**：由 **`OpenHermes-2.5`**（主要是由 GPT-4 生成的高质量指令问答数据）+ **`ELI5`（Reddit 上的 `r/explainlikeimfive` 子版块，专门用通俗清晰的语言回答各种好奇心问题的高赞问答）** 各 `100K` 条拼成！
  * **负样本（Negative Examples, 共 `200K` 条）**：从经过规则初筛的 `RefinedWeb` 中随机采样 `200K` 篇普通网页。
* **为什么用“指令问答 + 科普问答”当正样本筛选网页会奇效无比？**
  Percy 分析指出：虽然预训练是在原始网页上进行的，但我们最终评测模型靠的是 **MMLU、ARC、阅读理解与指令遵循等问答/推理基准**！用 `OpenHermes-2.5 + ELI5` 训练出的 `fastText` 分类器，精准捞出了互联网上那些**自带问答解释结构、富有教学逻辑与知识密度的网页**——结果在 `DCLM-Baseline` 的 `2.6T` Tokens 上从零训练一个 `7B` 模型，**5-shot MMLU 直接冲到了 `64%`**（超越 Mistral-7B-v0.3 的 `63%`，逼近训练了 `15T` Tokens 的 Llama-3-8B 的 `66%`，算力节省高达 6.6 倍）！

### 7.2 Nemotron-CC（2024 底）：突破“激进过滤导致 Token 枯竭”的长周期瓶颈（`nemotron_cc`）

DCLM 和 Hugging Face 的 **FineWeb-Edu**（用大模型按“教育价值”打分过滤）虽然在短周期（如 `1T`–`2T` Tokens）训练中刷爆了 MMLU，但 NVIDIA 团队在 **[Nemotron-CC (Su et al., 2024)](https://arxiv.org/abs/2412.02595)** 中敏锐指出了它们的致命缺陷：**它们过滤得太狠了，直接扔掉了 90% 的数据！**
当你像 Llama 3 (`15T` Tokens) 或 Qwen 3 (`36T` Tokens) 那样进行超长周期（Long-Horizon）预训练时，仅剩的 `1T`–`3T` 唯一高质量 Token 很快就会被跑满多轮（Multi-epoch），导致收益急剧衰减。

**Nemotron-CC 如何在保持顶尖质量的同时，把唯一真实 Token 数量扩大 4 倍（达到 `6.3T` Tokens，其中高质量子集 `1.1T`）？**
1. **减少死板规则误杀，改用 `jusText` 提取 HTML**：发现 `trafilatura` 丢弃了太多正文，改用 `jusText` 并取消会误杀优质非典型页面的 Gopher 启发式规则；
2. **双分类器集成打分（Classifier Ensembling）**：
   * **分类器 1**：提示 `Nemotron-340B-Instruct` 大模型根据“教育价值（Educational Value）”为 FineWeb 文档打分，并蒸馏到一个高速小模型中；
   * **分类器 2**：直接复用刚才讲到的 **DCLM `fastText` 分类器**；将全体网页按综合得分细分为 20 个质量桶（Buckets）。
3. **大规模合成数据改写（Synthetic Data Rephrasing for Pre-training）**：
   * **针对低质量网页（Low-Quality Data）**：不直接一刀切扔掉！而是调用语言模型对其进行**合成改写（Rephrase）**——保留其原本的事实信息，但抹除网页噪声、重写成类似 Wikipedia 的通顺规范文本！
   * **针对最高质量网页（High-Quality Data）**：调用语言模型围绕该网页**自动生成多样化任务数据（Generate Tasks：如提取问答对 QA Pairs、生成摘要、提炼核心知识点列表等）**，从而对高价值知识进行多视角合成扩充！
* **震撼成果**：在 `15T` Tokens 长周期训练下（其中 `7.2T` 来自 Nemotron-CC），训练出的 `8B` 模型在 MMLU 上比 Llama 3.1 8B 高出 **`+5.0` 个百分点**，ARC-Challenge 高出 **`+3.1`**！

### 7.3 The Stack v1 & v2（2022–2024）：构建工业级代码与软件工程语料库（`the_stack`）

代码数据在现代预训练中占据着不可替代的战略地位。BigCode 开源协作项目先后推出了两代标杆代码语料库：

1. **[The Stack v1 (Kocetkov et al., 2022)](https://arxiv.org/abs/2211.15533)（`3.1 TB` 宽松许可源码）**：
   * 从 GitHub Archive (2015–2022) 获取仓库名，直接 `git clone` 了 **1.37 亿个仓库、共 510 亿个文件**（去重后仅剩 **50 亿个唯一文件**，可见代码重复率高达 90%！）；
   * 使用 `go-license-detector` 严格筛选仅保留 **MIT、Apache-2.0、BSD** 等宽松许可证（Permissive Licenses）的代码，并通过 MinHash + Jaccard 相似度进行近重复去重，证明了做好近重复去重能大幅提升 HumanEval 与 MBPP 代码生成得分。
2. **[StarCoder 2 and The Stack v2 (Lozhkov et al., 2024)](https://arxiv.org/abs/2402.19173)（规模扩大 4 倍，支撑 `3.3T`–`4.3T` Tokens 训练）**：
   The Stack v2 完成了从“单纯收集源码文件”向**“全方位捕获软件工程开发过程”**的跃迁：
   * **全维度数据源**：基于 **Software Heritage (SWH)** 归档的 619 种编程语言源码仓库 + **GitHub Archive** 的 Issues、Pull Requests (PRs) 与代码审查评论 + 抓取 PyPI、npm、`devdocs.io` 等**官方技术文档** + Kaggle Notebooks 及数学/竞赛题库（GSM8K、OpenWebMath、StackOverflow、arXiv 等）。
   * **严苛清洗管线**：剔除二进制文件、恶意软件源码（Malware）、自动化机器人提交的垃圾 PR（Bot Activity）、长串自动生成数据，并进行 PII（密钥/邮箱/IP）脱敏与 PR 下采样。
   * **两大极具启发性的序列构造巧思**：
     1. **低资源语言与 LLVM 中间表示配对（Source-to-IR Pairing）**：像 Python、C++ 这样的主流语言有海量代码，但像 **Nim** 这样的小众低资源编程语言代码极少。作者巧妙地将源码编译为共享的底层中间语言 **LLVM IR**，并将 **`[低资源语言源码] <-> [LLVM IR]` 并排拼接**喂给模型——让模型借助高资源语言早已学透的共享 LLVM 语义桥梁，跨语言迁移掌握小众语言！
     2. **结构化 Pull Request (PR) 的线性化（Linearization）与上下文窗口权衡**：
        GitHub 上的 PR 和代码评审本质上是一棵**非线性的多事件结构树**（包含原始文件、多次 Commit Diff、行级 Review 评论、审批状态 `APPROVED / CHANGES_REQUESTED`）。为了喂给自回归模型，必须将其**线性化（Linearize）为带特殊 XML 风格标签的 Token 序列**，并在只有一行改动的 Diff 周围补充适量的原文件上下文（Inline Context）：
        ```xml
        <pr>
        Title: Fix off-by-one error in attention mask
        <pr_status>Merged</pr_status>
        <pr_base_file>
        <file_path>src/model/attention.py</file_path>
        ... [surrounding context lines] ...
        </pr_base_file>
        <pr_diff>
        @@ -42,3 +42,3 @@
        - mask = torch.triu(torch.ones(L, L), diagonal=0)
        + mask = torch.triu(torch.ones(L, L), diagonal=1)
        </pr_diff>
        <pr_event_id>1</pr_event_id>
        <pr_comment>Good catch! Diagonal=0 was masking the current token itself.</pr_comment>
        <pr_review_state>APPROVED</pr_review_state>
        </pr>
        ```
        通过这种格式化序列，模型学到的不再只是静态的代码结果，而是**人类程序员如何发现 Bug、如何讨论修改、如何迭代重构代码的完整动态思维过程**！

### 7.4 The Common Pile v0.1（2025）：如果只用 100% 合法宽松许可的数据，能走多远？（`common_pile`）

在本讲最后，Percy 呼应了前面的版权话题：既然互联网上绝大多数数据都有版权，且“合理使用（Fair Use）”的司法边界仍存在不确定性，**如果我们采取最极端的风险规避态度——“只要不确定有明确宽松许可的数据一律不用”，我们能训练出多好的大模型？**

这正是 Colin Raffel、Stella Biderman 等人在 **[The Common Pile v0.1: An 8TB Dataset of Public Domain and Openly Licensed Text (Kandpal et al., 2025)](https://arxiv.org/abs/2506.05209)** 中完成的极限实验：
1. **掘地三尺搜集 `8 TB` 合规数据（30 个开源/公有领域数据源）**：
   包含 **The Stack v2**（宽松许可代码）、**PeS2o / arXiv**（仅限明确挂有 CC-BY 等开放许可的学术论文）、**公有领域书籍（Project Gutenberg / 早期过保图书）**、**政府公开记录与法庭公报（Government Proceedings，美国联邦政府作品天然属于公有领域！）**、**专利文档**、**StackExchange / Wikis（CC BY-SA）**、**YouTube 开放许可字幕** 与开放教育资源（OER）。
2. **合规清洗中极易踩坑的三大隐蔽陷阱（Subtleties in Permissive Curation）**：
   Percy 指出，构建纯粹的合规数据集远比“写个正则匹配 `CC-BY` 字样”困难得多：
   * **陷阱一：许可证洗白（License Laundering）**：互联网上有很多人把别人受严格版权保护的文章、书籍或代码偷出来，自己随手贴上一个 `CC-BY` 或 `MIT` 标签重新发布到网上或 GitHub 上！如果不做深度溯源验证，极易把盗版数据当成合规数据收进来。
   * **陷阱二：集合许可证不等于个体作品许可证（Collection License $\neq$ Individual License）**：
     例如 AI2 发布的 `Dolma` 数据集在 Hugging Face 页面上标明的整体许可证是宽松的 **`ODC-By`**。新手很容易以为“太棒了，Dolma 是宽松许可，我可以放心商用！”——**大错特错！** 正如前文版权法所讲，集合（Collection）的许可证只代表 AI2 放弃对“数据编排方式”的权利，**绝不延伸或覆盖集合内部每一篇原始网页作者的独立版权**！
   * **陷阱三：坚决弃用任何合成数据（No Synthetic Data）**：
     Common Pile 团队主动放弃了使用像 GPT-4 或 Llama 生成的合成数据——因为尽管某些开源权重模型挂着 MIT/Apache 许可，但**这些生成模型本身是在未经授权的版权网页上训练出来的**，用它们的输出去训练新模型在严格意义上存在“数据洗白（Data Laundering）”的伦理与法律争议。
3. **实证结果（Comma v0.1-1T & 2T 模型，7B 参数）**：
   在 Common Pile 的 `1T` 和 `2T` Tokens 上训练的 **Comma-v0.1 (7B)** 模型，在 MMLU、ARC、HumanEval 等基准上**全面匹敌甚至超越了同等算力预算（1T–2T Tokens）下使用未授权全网数据训练的早期名模（如 LLaMA-1-7B、LLaMA-2-7B、MPT-7B、RPJ-INCITE-7B）**！但与如今在数十万亿未授权+合成 Token 上充分训练的前沿模型（如 Qwen3-8B）相比，仍存在明显差距——说明在合规数据的体量与挖掘深度上仍有巨大的探索空间。

## 8. 全景总结：14 大里程碑预训练数据集对比速查表（Cheat Sheet）

| 年份 | 数据集 / 关联模型 | 规模 (体积 / Tokens) | 核心数据源构成 | 标志性质量过滤与去重技术 | 历史地位与核心启示 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **2018** | **BERT Corpus** ([Devlin+ 2018](https://arxiv.org/abs/1810.04805)) | ~`3.3B` Words (~`16 GB`) | Wikipedia + [BooksCorpus](https://arxiv.org/abs/1506.06724) (Smashwords 7K 免费书) | 人工选定源，无复杂过滤 | 确立**篇章级（Document-level）**长序列预训练；BooksCorpus 后因违反 Smashwords ToS 下架 |
| **2019.02** | **WebText / OpenWebText** ([GPT-2](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)) | `40 GB` (`8M` 网页) | Reddit 帖子（**Karma $\ge 3$**）指向的外部出站链接 | **社交点赞代理质量** + `fastText` 英语识别 + LSH 去重 | 开创用人类社交筛选信号绕开原始 Common Crawl 垃圾的先河 |
| **2019.11** | **CCNet** ([Wenzek+ 2019](https://arxiv.org/abs/1911.00359)) | 多语种数千亿 Tokens | Common Crawl | 段落去重 + `fastText` 语种分类 + **Wikipedia KenLM 5-gram 困惑度分桶** | 证明用轻量级语言模型向 Wikipedia 分布对齐可自动化提纯低资源语言数据 |
| **2019.10** | **C4** ([T5, Raffel+ 2019](https://arxiv.org/abs/1910.10683)) | `806 GB` (`156B` Tokens) | 2019.04 单月 Common Crawl (`1.4T` $\to$ `156B`) | **纯手工启发式规则**（句末标点、$\ge 5$ 词/行、$\ge 3$ 句/页、删 `{` 与脏话词表） | 纯规则过滤标杆；后经 [Dodge+ 2021](https://arxiv.org/abs/2104.08758) 审计暴露出删光代码与脏话黑名单误杀少数群体偏见 |
| **2020.05** | **GPT-3 Dataset** ([Brown+ 2020](https://arxiv.org/abs/2005.14165)) | `570 GB` (`400B` Tokens) | Common Crawl (60%) + WebText2 (22%) + Books1/2 (16%) + Wiki (3%) | **训练二分类器区分高质源与原始 CC** + MinHashLSH 模糊去重 + **跨源非均匀过采样** | 确立“模型打分过滤 + 高质子集多轮过采样”的现代商业大模型数据范式 |
| **2020.12** | **The Pile** ([Gao+ 2020](https://arxiv.org/abs/2101.00027)) | `825 GiB` (~`275B` Tokens) | **22 个精选领域**（Pile-CC, PMC, arXiv, GitHub, StackExchange, Enron, Books3 等） | `WARC` + `jusText` 提取，强调领域多样性与人工策展配比 | 开源大模型摇篮；但包含影子图书馆 **Books3 (19.6万册书)** 埋下日后版权下架隐患 |
| **2021.12** | **MassiveText** ([Gopher, Rae+ 2021](https://arxiv.org/abs/2112.11446)) | `10.5 TB` (训练用 `300B`) | MassiveWeb + C4 + Books + News + GitHub + Wikipedia | **经典 Gopher 手工规则**（词长、字母占比 $\ge 80\%$、停用词）+ Google SafeSearch | 详尽定义了工业界最通用的单页启发式清洗规则集，拒绝使用模型分类器 |
| **2023.02** | **LLaMA 1 / RedPajama v1** ([Touvron+ 2023](https://arxiv.org/abs/2302.13971)) | `1.2T`–`1.4T` Tokens | CCNet (67%), C4 (15%), GitHub, Wiki, Books3+Gutenberg, arXiv, StackExchange | 分类器正样本设为 **“被 Wikipedia 引用的外部网页”**；SlimPajama 全局去重至 `627B` | 最后一个完整公开配方的工业旗舰模型；催生了 RedPajama 与 SlimPajama 开源生态 |
| **2023.06** | **RefinedWeb** ([Penedo+ 2023](https://arxiv.org/abs/2306.01116)) | 总量 `5T` (开源 `600B`) | **100% 纯 Common Crawl** | `WARC` + `trafilatura` + Gopher 规则 + 严苛 5-gram MinHash + 精确去重 | 证明 **“纯网页数据经过极致清洗与去重即可全面击败手工多源语料 (The Pile)”** |
| **2024.01** | **Dolma** ([Soldaini+ 2024](https://arxiv.org/abs/2402.00159)) | `3T` Tokens | CC + The Stack + C4 + Pushshift Reddit + peS2o (40M论文) + Gutenberg + Wiki | `fastText` 语种识别 + Gopher/C4 规则 + Jigsaw 毒性分类 + **Bloom Filter 去重** | 支撑全开源模型 OLMo 1；提供了完整的开源高性能数据处理工具箱 |
| **2024.06** | **FineWeb** ([Penedo+ 2024](https://arxiv.org/abs/2406.17557)) | **`15T` Tokens** | 95 个 Common Crawl 快照 (2013–2024) | 定制启发式规则 + **每个快照内部独立 MinHash 去重** + PII 脱敏；衍生出 **FineWeb-Edu** | 发现跨 95 快照全局去重反伤性能；为万亿级开源训练提供了坚实的纯网页底座 |
| **2024.06** | **DCLM-Baseline** ([Li+ 2024](https://arxiv.org/abs/2406.11794)) | `240T` 池 $\to$ **`3.8T` Tokens** | Common Crawl (`resiliparse` 提取) | **用 `OpenHermes-2.5 + ELI5` 作正样本训练 `fastText` 质量分类器**，激进筛选前 ~1.6% | 掀起**模型质量分类器回归浪潮**，7B 模型训 2.6T 即在 MMLU 达 `64%` |
| **2024.12** | **Nemotron-CC** ([Su+ 2024](https://arxiv.org/abs/2412.02595)) | **`6.3T` Tokens** (`1.1T` HQ) | Common Crawl (`jusText` 提取) | **双分类器集成 (Nemotron-340B 蒸馏 + DCLM)** + **大模型合成改写低质页 & 生成 QA** | 攻克激进过滤导致的 Token 枯竭难题，唯一真实 Token 达 DCLM 的 4 倍，赋能 15T 长周期训练 |
| **2024–2025** | **The Stack v2** ([Lozhkov+ 2024](https://arxiv.org/abs/2402.19173)) & **Common Pile** ([Kandpal+ 2025](https://arxiv.org/abs/2506.05209)) | Stack v2: `~900B+` 代码; Common Pile: **`8 TB`** | Stack v2: SWH 619 语言 + GitHub PRs + 文档; Common Pile: 30 个 **100% 宽松许可/公有领域源** | Stack v2: **低资源语言配对 LLVM IR** + **PR 树结构线性化**; Common Pile: 防许可证洗白与禁用合成数据 | 分别代表了**代码工程数据结构化**与**零法律风险纯合规预训练**的最高技术水准 |

### 本讲核心结语（Key Takeaways）
1. **数据不会从天上掉下来（Data does not fall from the sky）**：从活生生的互联网在线服务（Live Services），到通过爬虫或离线镜像获取原始数据（Raw Data），再到正文提取、规则/模型过滤与去重转化为可训练格式（Processed Data），每一步都凝聚着巨大的工程心血。
2. **过滤与提纯是决定胜负的杠杆**：如何从 `240T` 原始网页 Token 中精准筛出那 `1%`–`3%` 的精华（同时不损失长周期训练所需的多样性与总 Token 量），是区分平庸模型与顶尖模型的核心分水岭。
3. **充满“直觉（Vibes）”与改进空间的蓝海**：Percy 总结道，不同于本课程前半部分严丝合缝的矩阵乘法或硬件屋脊线定理，当今工业界的数据工程很大程度上依然依赖经验启发式与直觉试错（Heuristic & Vibes-driven）——这正是你在 **Assignment 4** 以及未来学术研究中最有机会做出突破性贡献的富矿！

## 9. 课后思考与自测练习（Review & Self-Check）

1. **版权法四要素（Fair Use）与 2025 标志性判例辨析**：
   假设某初创公司正在筹备训练一个 `14B` 参数的商业大语言模型，数据团队提出了以下三种获取数据的方式：
   * **方案甲**：从 BitTorrent 网络下载影子图书馆（如 LibGen / Books3）中的 20 万册现代畅销小说，计划只用来提取通用语言表征（并在输出端加上严格过滤器防止模型逐字背诵原文）；
   * **方案乙**：编写分布式爬虫抓取 YouTube 上明确标记为 **Creative Commons (CC-BY)** 许可的 500 万个教学视频的音频与字幕；
   * **方案丙**：直接下载 Hugging Face 上某个标有 `ODC-By`（开放数据公地署名许可）的聚合网页数据集 `MegaWeb-Mix`，用于商业模型训练。
   * (a) 请结合 *Bartz v. Anthropic (2025)* 的法庭裁决逻辑，分析**方案甲**中“即使模型训练本身被认定为转换性合理使用（Fair Use），公司依然面临十亿美元级法律赔偿”的根本原因是什么？如果公司事后去书店补买了这 20 万册实体书，能否免除之前的责任？
   * (b) 请解释为什么**方案乙**（作品本身是 CC-BY 宽松许可）与**方案丙**（数据集页面标着 `ODC-By` 宽松许可）在法律合规上依然存在严重隐患？
   * *参考解答*：
     * (a) 在 *Bartz v. Anthropic (2025)* 中，法院将**“获取数据的行为”**与**“训练模型的行为”**进行了严格切割：虽然法院认可将书籍输入大模型训练属于转换性合理使用（Fair Use），但**最初从影子图书馆或 BT 网络下载盗版书籍副本（Pirating copies）的行为，从复制发生的那一刻起就已经构成了直接版权侵权（即使下载后根本不训练也违法）**！而且事后补买正版实体书并扫描（虽然买书拆页扫描本身属于 Fair Use）**绝不能洗白或免除（Absolve）此前已经发生的盗版下载违法事实**（Anthropic 因此付出了 15 亿美元的和解代价）。
     * (b) **方案乙的隐患在于服务条款（Terms of Service, ToS）叠加限制**：虽然视频创作者授予了 CC-BY 版权许可，但托管平台 **YouTube 的服务条款（ToS）明确禁止使用自动化爬虫批量下载视频内容**，强行抓取会构成合同违约。**方案丙的隐患在于“集合许可证不延伸至个体作品（Collection License $\neq$ Individual Work License）”以及“许可证洗白（License Laundering）”**：像 Dolma 等聚合数据集的 `ODC-By` 许可仅涵盖数据集制作者的编排工作，集合内部的每一篇原始网页依然受各自原作者的独立版权保护，并不能直接当作 100% 宽松授权数据无风险商用。

2. **Common Crawl 提取、过滤与长周期多轮训练（Multi-Epoch）定量计算**：
   假设你从 Common Crawl 获得了包含 $T_{\text{pool}} = 200\text{ T}$（200 万亿）Token 的原始英文网页池，准备为一个 $N = 8\text{ B}$ 参数的模型构建一份需要在 $D_{\text{train}} = 12\text{ T}$（12 万亿）Token 上进行长周期预训练的数据集。你对比了三种过滤方案：
   * **方案 1（C4 风格纯手工规则）**：移除了包含花括号 `{` 的所有页面及脏话词表页面，留存率 $\alpha_1 = 10\%$（得到 $20\text{ T}$ 唯一 Token）；
   * **方案 2（DCLM 风格激进单分类器过滤）**：使用 `OpenHermes-2.5 + ELI5` 训练的 `fastText` 分类器只保留 Top $\alpha_2 = 1.5\%$ 的网页；
   * **方案 3（Nemotron-CC 风格分类器分桶 + 合成改写）**：保留 Top $1.5\%$ 的高质原网页，同时对高质页用大模型合成扩充出相当于原网页 $100\%$ 体量的 QA/摘要任务数据，并对中间档 $3.0\%$ 的低质网页用大模型改写（Rephrase）保留了其中 $50\%$ 的 Token。
   * (a) 请解释**方案 1**中“移除包含 `{` 的页面”会彻底摧毁模型的哪项核心下游能力？为什么在 Gopher (2021) 和 RefinedWeb (2023) 中不再使用这项规则？
   * (b) 请分别计算**方案 2**与**方案 3**最终生成的**唯一真实/合成 Token 总量** $T_2$ 与 $T_3$。如果用它们各自训练 $D_{\text{train}} = 12\text{ T}$ Token，方案 2 和方案 3 平均每个 Token 分别需要重复训练多少个 Epoch？结合 Lecture 9/11 的知识，解释为什么方案 3 在 $12\text{ T}$ 长周期训练中能显著击败方案 2。
   * *参考解答*：
     * (a) 花括号 `{` 是 C/C++、Java、JavaScript、JSON、Rust 以及 LaTeX 公式中最核心的语法定界符！C4 删除含 `{` 的页面会**将几乎所有编程语言代码与大量数学公式网页彻底删光**，不仅摧毁了模型的编程能力，还严重削弱了依赖代码结构建立的通用逻辑推理能力。
     * (b) **方案 2（DCLM 风格）**的唯一 Token 总量为：
       $$T_2 = 200\text{ T} \times 1.5\% = \mathbf{3.0\text{ T Tokens}}$$
       在 $12\text{ T}$ 训练周期下，平均每个 Token 需要重复训练：
       $$\text{Epochs}_2 = \frac{12\text{ T}}{3.0\text{ T}} = \mathbf{4.0\text{ Epochs}}$$
       **方案 3（Nemotron-CC 风格）**的总 Token 量为高质原网页（$1.5\%$）+ 高质合成任务（$1.5\% \times 100\%$）+ 低质改写网页（$3.0\% \times 50\%$）：
       $$T_3 = 200\text{ T} \times (1.5\% + 1.5\% + 1.5\%) = 200\text{ T} \times 4.5\% = \mathbf{9.0\text{ T Tokens}}$$
       在 $12\text{ T}$ 训练周期下，平均每个 Token 仅需遍历：
       $$\text{Epochs}_3 = \frac{12\text{ T}}{9.0\text{ T}} \approx \mathbf{1.33\text{ Epochs}}$$
       **原因**：当重复轮数达到 4 Epochs 时，多轮重复训练的边际收益会因过拟合与记忆效应明显衰减；方案 3 通过分类器集成与对低质网页的合成改写，在不牺牲单 Token 信息密度的前提下将有效数据池扩大了 3 倍，使 $12\text{ T}$ 长周期训练始终处于接近单轮（~1.33 Epochs）的高效泛化区间。

3. **代码与结构化软件工程数据的线性化设计（The Stack v2）**：
   * (a) 在 [The Stack v2 (Lozhkov et al., 2024)](https://arxiv.org/abs/2402.19173) 中，为了提升 StarCoder2 在小众低资源编程语言（如 Nim）上的表现，作者采用了什么跨语言数据配对策略？请从表征学习的视角解释为什么该策略有效。
   * (b) 当把 GitHub Pull Request (PR) 转化为自回归语言模型的训练序列时，如果只把 `git diff` 中被修改的那 1–2 行代码（如 `- a = 1` / `+ a = 2`）和评论拼接在一起，会存在什么问题？相反，如果把整个仓库的所有文件都原封不动塞进每个 PR 事件前面，又会带来什么问题？The Stack v2 是如何折中解决的？
   * *参考解答*：
     * (a) **策略**：将低资源语言（如 Nim）的源代码通过编译器编译为**共享的底层中间表示（Intermediate Representation，如 LLVM IR 汇编）**，并将 `[Nim 源码]` 与其对应的 `[LLVM IR]` 并排拼接在同一训练文档中。**有效原因**：像 C、C++、Rust 等高资源语言同样编译到 LLVM IR，模型在海量高资源语言代码与 LLVM IR 上已经建立了极其稳固的底层逻辑语义空间；通过将低资源语言与共享的 LLVM IR 配对，模型可以利用 LLVM IR 作为“语义锚点（Semantic Bridge）”，将高资源语言的算法与控制流知识零样本/少样本地迁移到低资源语言上。
     * (b) **只放 1–2 行 Diff 的问题**：缺乏上下文（Context Starvation）——模型根本不知道被修改的变量是在什么函数、什么类定义或什么业务逻辑下被调用的，无法学到“为什么要做这个修改”的因果逻辑。**塞入整个仓库或超大完整文件的问题**：Token 消耗极度膨胀且信噪比极低，绝大部分算力被浪费在与当前 PR 改动毫无关系的几千行背景代码的重复预测上。**The Stack v2 的折中方案**：设计一套结构化 XML 标签线性化协议，在每个 Diff 周围**自适应截取适量的文件局部上下文窗口（Inline Surrounding Context）**，并对超大规模或高频重复的 PR 进行下采样（Subsampling），在提供充足局部推理线索的同时保持极高的 Token 信息密度。
