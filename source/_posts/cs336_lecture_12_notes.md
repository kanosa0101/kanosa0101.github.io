---
title: Stanford CS336 (Spring 2026) Lecture 12 深度课程笔记：Evaluation
date: 2026-10-26 08:53:00
tags:
  - 大模型
  - 模型评测
  - 困惑度
  - 基准测试
  - 污染检测
  - HELM
  - CS336
categories:
  - [笔记, CS336]
description: 系统总结 Stanford CS336 Lecture 12，全方位拆解大模型评测体系：困惑度（Perplexity）数学局限、标准化考试/Agentic/推理与安全基准测试、生态效度（Ecological Validity）以及测试集污染检测方法。
mathjax: true
---

# Stanford CS336 (Spring 2026) Lecture 12 深度课程笔记：Evaluation

> **课程信息**：[Stanford CS336: Language Modeling from Scratch (Spring 2026)](https://cs336.stanford.edu/)
> **主讲教授**：Percy Liang
> **本讲资源**：[Lecture 12 YouTube 视频](https://www.youtube.com/watch?v=JpAxdTWQJxM) | [交互式执行轨迹 (`lecture_12.py`)](https://cs336.stanford.edu/lectures/?trace=lecture_12) | [源码仓库 (`spring2025-lectures/lecture_12.py`)](https://github.com/stanford-cs336/spring2025-lectures/blob/main/lecture_12.py) | [Stanford CRFM HELM 评测平台](https://crfm.stanford.edu/helm/)

## 1. 引言：为什么大模型评测（Evaluation）不是简单的“跑分”？

在前面的 11 讲中，我们已经完整搭建起了从零训练与部署一个大语言模型的全套工程技术栈：模型架构与优化器（Lectures 1–4）、GPU 算子与分布式并行系统（Lectures 5–8）、扩展定律与超参缩放（Lectures 9 & 11）以及高效推理系统（Lecture 10）。现在，整个拼图只剩下最后两大核心板块：**训练数据工程（Data，第 13–14 讲，对应 Assignment 4）** 与 **后训练对齐（Alignment & RL，第 15–17 讲，对应 Assignment 5）**。

Percy 在开篇指出：**训练数据决定了模型的行为**（喂给模型代码，它就擅长写代码；只喂给它 DNA 序列，它就绝不可能学会说英语）。但在我们动手清洗和配比数据之前，必须先回答一个更根本的前置问题：**我们到底希望语言模型具备什么样的能力与行为？换言之，给定一个训练好的模型，它究竟有多“好”（How "good" is it）？** 这正是本讲（Lecture 12: Evaluation）的主题。

### 1.1 评测的北极星效应与核心挑战（Abstract Construct $\to$ Concrete Metric）

表面上看，评测（Evaluation）似乎只是一个机械的工程流程：
$$\text{定义一组提示词 (Prompts)} \longrightarrow \text{发送给模型获取回复 (Responses)} \longrightarrow \text{计算准确率 (Accuracy)}$$
但实际上，**评测是塑造整个 AI 领域演进方向的指挥棒（North Star）**——无论是开源社区还是闭源巨头，所有模型研发者都盯着评测指标来衡量每一次架构修改、数据过滤或算法迭代是否有效。如果你定义的评测偏离了真实目标（Goodhart's Law），整个社区就会朝着错误的方向狂奔。

* **评测最核心的科学难题**：
  人类最初拥有的往往是一个**抽象概念（Abstract Construct）**——例如“我希望模型擅长对话”、“具有强大的逻辑推理能力”或“安全可靠”；而评测的本质，就是**将这个模糊的抽象概念转化为由具体提示词（Prompts）、交互环境（Environments）与打分规则驱动的具象数值指标（Concrete Metric）**！

### 1.2 定义“什么是好模型（What is Good?）”的四种不同视角

在进入具体基准之前，Percy 在 `what_is_good()` 中展示了当前工业界衡量模型好坏的四种截然不同的宏观透镜：

| 评估视角 | 代表平台 / 榜单 | 核心度量逻辑 | 优势与局限性 |
| :--- | :--- | :--- | :--- |
| **1. 纯基准跑分视角** | **[Artificial Analysis Intelligence Index](https://artificialanalysis.ai/)** | 汇总多个高难度学术与推理基准的平均得分（如 2026 年榜单上从 GPT-5.5 等前沿模型一路向下排序） | 客观、可复现，但容易受刷榜（Benchmaxing）影响，且未反映使用成本 |
| **2. 性能-成本帕累托视角** | **Artificial Analysis (Intelligence vs. Price)** | 横轴为每百万 Token 推理价格（$/1M Tokens），纵轴为综合智力指数 | 揭示了精度与推理成本的正相关性及帕累托前沿，直接指导工程选型 |
| **3. 人类主观偏好视角** | **[Arena AI (原 LMSYS Chatbot Arena)](https://lmarena.ai/)** | 让真实互联网用户盲测两个匿名模型的回复并投票，拟合 **Elo 积分** | 贴近真实人类对话体感，但受用户群体分布偏差与形式讨好（Sycophancy / 长度偏见）影响 |
| **4. 市场真金白银投票视角** | **[OpenRouter Usage Rankings](https://openrouter.ai/rankings)** | 统计通过统一 API 网关实际调用并付费消耗的各模型 Token 市场份额 | 最纯粹的经济学检验（“如果人们愿意掏钱用它，它一定在创造价值”），但受限于特定开发者生态样本 |

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│                       CS336 Lecture 12 全景架构：大模型评测的六大支柱与三大元准则               │
├─────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                 │
│  [六大评测范式演进 (Six Pillars of LLM Evaluation)]                                             │
│    ├─► 1. 概率分布视角: 困惑度 (Perplexity, PPL) ──► 经典域内 (PTB/1BW) vs. GPT-2 零样本域外    │
│    │      • 条件困惑度 (Conditional PPL) 与“伪装成准确率的困惑度” (LAMBADA, HellaSwag)           │
│    ├─► 2. 标准化考试基准 (Exam Benchmarks): MMLU (2020) ──► MMLU-Pro ──► GPQA ──► HLE (2025)    │
│    ├─► 3. 开放式对话基准 (Chat Benchmarks): Chatbot Arena (Elo) ──► AlpacaEval 2.0 ──► WildBench│
│    ├─► 4. 自主智能体基准 (Agentic Benchmarks): SWE-bench ──► Terminal-Bench 2.0 / Cybench / MLE │
│    ├─► 5. 纯流体推理基准 (Pure Reasoning): ARC-AGI-1 (2019) ──► ARC-AGI-2 ──► ARC-AGI-3 (2026)  │
│    └─► 6. 安全与红队基准 (Safety & Red-Teaming): HarmBench, AIR-Bench 2024, GCG 对抗越狱与双用途│
│                                                                                                 │
│  [三大元评测工程准则 (Meta-Evaluation Considerations)]                                          │
│    ├─► 准则 A: 生态效度与真实感 (Realism / Ecological Validity) ──► GDPval, MedHELM, Clio 隐私分析│
│    ├─► 准则 B: 科学效度与防污染 (Validity & Decontamination) ──► 可交换性检验/动态新题/私有评测 │
│    └─► 准则 C: 评估目标解耦 (Methods vs. Models vs. Agents) ──► 明确游戏规则，审计轨迹 (Docent) │
└─────────────────────────────────────────────────────────────────────────────────────────────────┘
```

## 2. 支柱一：困惑度（Perplexity）—— 从概率分布本源看评测（`perplexity`）

回顾语言模型最本质的数学定义：**语言模型是一个定义在 Token 序列上的概率分布 $p(x_1, x_2, \dots, x_L)$**。因此，评估一个概率分布最自然、最第一性原理的指标，就是看它为测试集 $D$ 赋予了多大的概率质量——这就是**困惑度（Perplexity, PPL）**（即平均每 Token 交叉熵损失的指数）：
$$\mathbf{\text{Perplexity}(p, D) = \left(\frac{1}{p(D)}\right)^{\frac{1}{|D|}} = \exp\!\left(-\frac{1}{|D|} \sum_{i=1}^{|D|} \log p(x_i \mid x_{<i})\right)}$$

### 2.1 从经典“域内评测”到 GPT-2 的“零样本域外评测”范式转移

1. **2010 年代的经典范式：域内评测（In-Distribution Evaluation）**
   在预训练大模型诞生之前，语言建模研究遵循严格的独立同分布（I.I.D.）假设：在某个特定数据集的训练集切分（Train Split）上训练模型，然后在同一个数据集的测试集切分（Test Split）上报告 Perplexity。当時の三大标准基准为：
   * **Penn Treebank (PTB)**：源自《华尔街日报》（WSJ）新闻语料，规模极小（约百万词级）；
   * **WikiText-103**：源自维基百科精选文章；
   * **One Billion Word Benchmark (1BW)**：源自 WMT11 机器翻译语料（EuroParl、联合国文件与新闻）。
   * **历史里程碑案例**：**[Exploring the Limits of Language Modeling (Jozefowicz, Vinyals, Schuster, Shazeer & Wu, 2016)](https://arxiv.org/abs/1602.02410)** 在 1BW 基准上通过大规模字符级 CNN + LSTM 将单模型 Perplexity 从 `51.3` 暴力压低至 **`30.0`**（集成模型达 `23.7`）。Percy 回忆道，当时学界还在争论到底应该用传统 $n$-gram 平滑还是神经语言模型，而这篇论文彻底终结了悬念，确立了纯神经网络统治语言建模的时代。
2. **2019 年 GPT-2 掀起的范式革命：零样本域外评测（Zero-Shot Out-of-Distribution Evaluation）**
   当 OpenAI 发布 **[GPT-2 (Radford et al., 2019)](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)** 时，他们做了一件在当时极具颠覆性的事：只在从 Reddit 外链抓取的 `40 GB` **WebText** 数据集上训练模型，然后**完全不碰 PTB、WikiText-103、1BW 的训练集，直接以零样本（Zero-Shot）方式去测这些传统数据集测试集的 Perplexity**！
   * **结果与洞见**：GPT-2 (1.5B) 在小数据集（如 PTB）上直接将 Perplexity 刷到了 `35.8`（大幅超越此前域内训练的 SOTA `46.5`），因为小数据集自身训练样本不足，通用预训练的迁移红利巨大；而在拥有充足域内训练数据的 1BW 上，GPT-2 的零样本 PPL（`75.2`）则不如专门在 1BW 上过拟合训练的模型。这一工作正式开启了“在海量互联网语料上预训练、在各类外部标准基准上零样本评测”的现代范式。

### 2.2 “Perplexity Is All You Need” 的极限信仰 vs. “Perplexity 管得太宽了”

* **为什么许多早期研究者坚信“只要把 Perplexity 降到底，就能通往 AGI”？**
  设真实世界文本与任务的生成分布为 $t(x)$，我们的语言模型分布为 $p(x)$。根据信息论中的吉布斯不等式（Gibbs' Inequality），交叉熵 $H(t, p) = H(t) + D_{\text{KL}}(t \parallel p) \ge H(t)$：
  $$\min_p \text{Perplexity}(p, t) = \exp\big(H(t)\big) \quad \Longleftrightarrow \quad \mathbf{p = t \quad (\text{当且仅当 } D_{\text{KL}}(t \parallel p) = 0)}$$
  也就是说，**最小化 Perplexity 的唯一全局极小值点就是 $p = t$（完美还原真实世界分布）**！一旦 $p = t$，你就可以把任何问题作为前缀条件，通过求条件概率 $p(\text{solution} \mid \text{problem})$ 或 $p(\text{answer} \mid \text{question})$ 解决人类写出的一切任务！正是这种近乎宗教般的信念（More faith than science），支撑了研究者们在 GPT-3 爆发前夜坚定不移地投入巨资扩展模型规模。
* **为什么说 Perplexity 有时又“管得太宽了（More than you need）”？**
  考虑句子：`"Stanford was founded in 1885"`。
  标准全句 Perplexity 会对句子中的**每一个 Token** 都同等施加对数似然惩罚：
  * 给定 `"Stanford was founded in "` 预测 **`"1885"`**：这非常有价值！它本质上就是在考一道世界知识问答题（QA）；
  * 给定空串预测句首第一个词 **`"Stanford"`**，或者预测中间的语法词 **`"was"`**：这纯粹取决于作者当时想聊哪所大学或用什么句式，包含大量无关紧要的偶然噪声（Incidental Noise），但全句 Perplexity 依然会因为你没猜中句首词而重罚你！
* **工程解法：条件困惑度（Conditional Perplexity）**
  为了只评估我们真正关心的核心信息 Token，我们可以固定提示词前缀 `prompt`，仅对目标回答 `response` 计算**条件困惑度**：
  $$\mathbf{\text{Conditional PPL}(p) = \left(\frac{1}{p(\text{response} \mid \text{prompt})}\right)^{\frac{1}{|\text{response}|}}}$$

### 2.3 “伪装成准确率的困惑度”（Perplexity in Disguise）与排行榜作弊陷阱

Percy 指出，许多经典的下游准确率基准，撕开外壳看本质，其实就是**只针对特定关键位置或候选句计算的条件困惑度**：

1. **完形填空任务（Cloze Tasks）：[LAMBADA (Paperno et al., 2016)](https://arxiv.org/abs/1606.06031)**
   给定一段长叙事上下文，要求预测全篇的**最后一个词**（例如：*"... Once back in the hospital, Thomas goes straight to Jen's room... 'Do you honestly think I would want you to have a \_\_\_\_?'"* $\to$ 答案：`"miscarriage"`）。
   * **设计巧思**：作者通过人类被试筛选确保：**只有读完全篇长上下文的人才能猜出最后一个词，只看最后一句的人绝不可能猜出**！这精准过滤掉了无聊的局部语法搭配，专门把 Perplexity 聚焦在需要长距离跨句篇章推理（Broad Discourse Context）的关键 Token 上（GPT-2 曾将其作为主打评测基准）。
2. **多项选择句子补全：[HellaSwag (Zellers et al., 2019)](https://arxiv.org/abs/1905.07830)**
   给定事件前缀（如 *"A woman is outside with a bucket and a dog. The dog is running around trying to avoid a bath. She..."*），从 4 个候选后续句子中选出最合理的一个（模型通过比较 4 个候选续写的条件对数似然 $\log p(\text{choice}_i \mid \text{context})$ 来做选择）。
   * **核心贡献——对抗性过滤（Adversarial Filtering, AF）**：由早期语言模型生成错误选项，并用判别器迭代挑选出那些**对人类来说荒谬无比（人类准确率 $>95\%$）、但最容易欺骗当时 SOTA 模型（模型准确率 $<48\%$）**的干扰项。

> **重要工程警示：为什么你很难办一个公开的“黑盒 Perplexity 排行榜（Perplexity Leaderboard）”？**
> * **在下游任务（Downstream Accuracy）排行榜中**：模型完全可以是黑盒 API——组织者发去一条 `prompt`，参赛模型返回生成的字符串 `response`，评测脚本直接对比 `response == ground_truth`，参赛者无法在算分环节弄虚作假。
> * **在 Perplexity 排行榜中**：组织者把测试集文本 `test_data` 发给参赛模型，要求参赛模型返回 `log_prob = LM(test_data)`。**致命漏洞在于：你必须信任参赛模型返回的概率分布真的是一个合法归一化的概率分布（即对所有可能词表 Token 求和严格满足 $\sum_{v \in V} p(v \mid x_{<t}) = 1$）！** 如果一个作弊者写一行代码永远返回 `log_prob = 0.0`（即声称概率恒为 $100\%$，导致 $\sum p > 1$），它的 Perplexity 就能拿到满分 `1.0`！而如果不审查模型源码与全词表 Logits（或者面对像 VAE / 扩散模型这种只能算变分下界 ELBO 的非自回归模型），外部评测方很难低成本地验证概率分布的合法性。

## 3. 支柱二：标准化考试基准（Exam Benchmarks）—— 从 MMLU 到 Humanity's Last Exam（`exam_benchmarks`）

既然纯 Perplexity 难以直观说服非技术受众（"non-believers"），学界最自然的想法就是**像考人类学生一样，用标准化试卷（Exams）去考大模型**。考试题目的最大优点是：**学科与难度高度可控，且拥有唯一明确的客观标准答案（多选题 A/B/C/D 或简短数值），极易自动化判卷**。

Percy 沿着 2020 年至 2026 年的时间线，梳理了由于模型能力狂飙而不断被迫“加难重造”的四代里程碑考试基准：

```
考试基准演进与饱和循环 (2020–2026):
  [MMLU (2020)] ────────► 2024 年刷到 ~90% 饱和!
  (57学科, 4选1, 5-shot)
         │
         ▼ (清洗错题 + 扩为 10 选 1 + 引入 CoT 推理题)
  [MMLU-Pro (2024.06)] ─► 发布时暴跌 16%~33%，但到 2026 年又刷到了 ~89% 饱和!
         │
         ▼ (雇佣 61 名博士出“Google 搜不到答案”的硬核专业题)
  [GPQA Diamond (2023.11)]► 人类博士 65% / 外行搜 Google 34% / GPT-4 39% ──► 2026 年刷到 ~94% 饱和!
         │
         ▼ ($500K 奖金 + 全球超千名顶尖学者共创多模态人类知识极限闭卷题)
  [Humanity's Last Exam (HLE, 2025.01)] ──► 2025 初仅个位数 (~3%~9%) ──► 2026.05 最强模型 (Claude Mythos) 达 64.7%!
```

| 考试基准 | 发布论文与年份 | 题目规模与题型设计 | 出题流程与核心创新点 | 人类 vs. 模型分数演进轨迹 (截至 2026.05) |
| :--- | :--- | :--- | :--- | :--- |
| **1. MMLU** | **[Hendrycks et al. (2020)](https://arxiv.org/abs/2009.03300)** | 57 个学科（初等数学、美国历史、法律、道德等），4 选 1 单选题 | 伯克利学生从网上免费公开试卷搜集；首次在 GPT-3 上用 **Few-shot Prompting (5-shot)** 评测通用知识 | GPT-3 (175B) 仅 `43.9%`（小模型接近随机 `25%`）$\to$ GPT-4 `86.4%` $\to$ 现已 `>90%` 彻底饱和（可在 [HELM MMLU](https://crfm.stanford.edu/helm/mmlu/latest/) 查看具体预测） |
| **2. MMLU-Pro** | **[Wang et al. (2024)](https://arxiv.org/abs/2406.01574)** | 12,000+ 题，**从 4 选 1 扩展为 10 选 1**，聚焦多步计算与推理 | 剔除原版 MMLU 中的错题与送分题；强制使用 **Chain-of-Thought (CoT)** 评测，使不同 Prompt 风格的波动从 `4-5%` 降至 `2%` | 发布时各模型分数比 MMLU 暴跌 `16%~33%`；但到 2026 年初前沿模型已达 **`~88%–90%`**（[HELM MMLU-Pro](https://www.google.com/search?q=https://crfm.stanford.edu/helm/mmlu-pro/latest/)） |
| **3. GPQA** (*Google-Proof Q&A*) | **[Rein et al. (2023)](https://arxiv.org/abs/2311.12022)** | 448 道生物、物理、化学博士级难题（核心精选子集为 **GPQA Diamond**） | 雇佣 61 名 Upwork 博士撰题 $\to$ 同领域第二专家验证并返修 $\to$ **让非本专业高学历验证者开着 Google 搜 30 分钟**，仅当专家答对且外行答错时选入 Diamond | • 本专业博士：**`65%`**（事后扣除粗心失误为 `74%`）<br>• 外行开 Google 搜 30 分钟：**`34%`**<br>• GPT-4 初代：**`39%`** $\to$ **2026 年 SOTA 已达 `94%`！** |
| **4. HLE** (*Humanity's Last Exam*) | **[Phan et al. (2025)](https://arxiv.org/abs/2501.14249)** ([lastexam.ai](https://lastexam.ai/)) | 2,500 道跨数十学科的**多模态（图文）**选择题与精确简答题（加设未公开私有测试集） | 设立 **$50 万美元奖金池 + 论文共同作者署名权**，发动全球上千名领域学者出题；先用全套前沿大模型做过滤，只要现有模型能答对的题当场淘汰！ | • 2025 年 1 月发布时：GPT-4o `3.3%`、o1 `~8%`（全军覆没）<br>• **2026 年 5 月最新 SOTA（Claude Mythos）：已攀升至 `64.7%`！** |

> **课堂问答：在多选题（如 HLE / MMLU）中，到底是怎么提取模型答案并计算准确率的？**
> * **学生提问**：在计算多选题准确率时，我们是直接检查模型对 A/B/C/D 四个选项字母赋予的概率（Logprobs），还是让模型生成一段文本？
> * **Percy 解答**：早期（如 GPT-3 测原版 MMLU 时）确实常直接看下一个 Token 是 `'A'`, `'B'`, `'C'`, `'D'` 的概率或只生成 1 个字母；但在现代大模型（尤其是推理模型）评测中，**标准做法是让模型先生成一段完整的思维链推理（Chain-of-Thought）与解释，最后由答案提取器（Answer Extractor / 正则解析器）从输出末尾抽取出最终选择的字母（如 `Answer: C`）与标准答案比对**。Percy 特别提醒：**答案提取与格式化解析的细节对最终准确率非常敏感**（有时模型明明推对了，却因为输出格式没对齐正则表达式而被判错）！

* **考试基准的根本局限**：
  虽然多选题在难度上可以做到无限难（如 HLE），但**考试并不能代表真实世界的使用场景（Does not capture real-world usage）**——现实中没有用户会每天拿着 4 选 1 的博士考题去考 AI 助手；真实用户的提问绝大多数是**开放式的（Open-ended）、表述含糊的，甚至根本不存在唯一标准答案**！

## 4. 支柱三：开放式对话基准（Chat Benchmarks）—— 人类盲测 Elo 与 LLM-as-a-Judge（`chat_benchmarks`）

为了评估开放式指令遵循与日常对话（例如课堂给出的真实 Prompt：*“我想做一份甜菜根配山羊奶酪沙拉，放什么香草比较搭，什么不搭？”*），既然不存在可以使用 `==` 判定的标准答案，学界演化出了两条核心路径：**众包人类盲测对战（Chatbot Arena）** 与 **大模型自动裁判（LLM-as-a-Judge）**。

### 4.1 众包人类成对盲测与 Elo 等级分推导（[Chatbot Arena / Arena AI, Chiang et al., 2024](https://arxiv.org/abs/2403.04132)）

UC Berkeley LMSYS 团队搭建的 **Chatbot Arena（现称 [Arena AI](https://lmarena.ai/)）** 开创了众包成对比较（Pairwise Comparison）的黄金范式：
1. **数据收集机制**：任何互联网用户访问网站并输入任意真实 Prompt $\to$ 系统后台随机抽取两个匿名模型（Anonymous Model A & Model B）并排生成回复 $\to$ 用户根据主观体感投票选择 `A is better`、`B is better`、`Tie (Both are good)` 或 `Both are bad`。
2. **Bradley-Terry / Elo 等级分统计建模**：
   假设每个模型 $i$ 都有一个潜在的能力等级分参数 $R_i$（即 $\text{ELO}_i$）。模型 $A$ 在单局对战中击败模型 $B$ 的理论概率定义为逻辑斯蒂（Logistic / Sigmoid）函数：
   $$\mathbf{P(A \text{ beats } B) = \frac{1}{1 + 10^{\frac{\text{ELO}_B - \text{ELO}_A}{400}}}}$$
   给定收集到的成千上万条人类成对投票记录，通过**最大似然估计（Maximum Likelihood Estimation, MLE）**拟合出所有模型的 $\text{ELO}$ 分数（注：相比于早期顺序更新的经典国际象棋 Elo，现代 Chatbot Arena 使用批量 Bradley-Terry 逻辑回归来消除对局先后顺序偏差并计算置信区间）。
3. **Chatbot Arena 的两大系统优势与四大潜在隐患**：
   * **优势 1（稀疏成对采样即可全局排序）**：正如国际象棋比赛不需要让所有棋手两两全打一遍，只要模型之间的对战图是连通的（Connected Graph），成对比较就能从稀疏采样中拟出全局一致的排名，且完全**不需要给所有模型喂相同的固定 Prompt**！
   * **优势 2（天然动态演进）**：随时可以无缝接入新发布的模型和用户实时想出的新 Prompt。
   * **四大隐患**：
     1. **用户分布未知且有偏（Uncontrolled User Distribution）**：去免费用 Arena 的网民代表了什么样的分布？是否存在恶意刷票者（Spammers）甚至模型厂商定向投毒（Game the leaderboard）？
     2. **将“文风形式（Style）”与“事实正确性（Correctness）”混为一谈**：在国际象棋中赢就是赢；但在对话中，二元偏好投票把排版好看、语气热情与内容正确揉成了一个标量。
     3. **外行裁判悖论**：用户之所以向 AI 提问，往往正是因为**用户自己不知道正确答案**！当两个模型给出看似都很有道理但结论相反的专业回答时，提问者怎么判断谁是对的？
     4. **诱导阿谀奉承（Sycophancy）**：一味迎合人类直觉、顺着用户偏见说好话的回复，往往比客观严厉指出用户错误的回复更容易拿到高赞。

### 4.2 大模型自动裁判（LLM-as-a-Judge）与长度偏见去偏（AlpacaEval 2.0 & WildBench）

人类盲测虽然好，但收集成千上万张选票既慢又贵，无法用于日常模型开发内部迭代。于是学界提出了 **LLM-as-a-Judge（用强模型如 GPT-4 作为自动裁判）** 的基准：

#### 1. [AlpacaEval (Li et al., 2023)](https://tatsu-lab.github.io/alpaca_eval/) 与 [Length-Controlled AlpacaEval 2.0 (Dubois, Galambosi, Liang & Hashimoto, 2024)](https://arxiv.org/abs/2404.04475)
* **基本设定**：收集 805 条代表性开放指令，让被测模型 $m$ 生成回复 $z_m$，并与基准模型 $b$（如 `GPT-4 Preview`）生成的回复 $z_b$ 一起交给裁判模型（`GPT-4 Preview`）打分，计算被测模型击败基准模型的**胜率（Win Rate）**。
* **暴露出的严重漏洞——长度偏见（Length Bias / Verbosity Gaming）**：
  早期的 AlpacaEval 发布后不久，榜单迅速被各种微调小模型刷爆——研究者发现 **LLM 裁判（其实人类裁判也一样）极其偏爱长篇大论、滥用 Markdown 列表的啰嗦回复**！哪怕内容全是车轱辘话，只要把输出长度拉长一倍，胜率就会虚高十几个百分点。
* **Tatsu 与 Percy 团队的去偏解法（Length-Controlled AlpacaEval）**：
  回答一个**反事实问题（Counterfactual Question）**：*“如果被测模型的回复与基准模型的回复长度完全相同（$\text{length}(z_m) - \text{length}(z_b) = 0$），裁判的偏好胜率会是多少？”*
  具体做法是构建一个包含**模型真实能力项 $\theta_m$** 与**长度差混淆中介项 $\tanh(\Delta \text{length})$** 的广义线性模型（Logistic Regression / GLM）：
  $$q_{\theta, \phi, \psi}(y = 1 \mid x, m, b) = \text{logistic}\!\left(\theta_m - \theta_b + \phi_{m, b} \cdot \tanh\!\left(\frac{\text{len}(z_m) - \text{len}(z_b)}{\sigma_{\Delta\text{len}}}\right) + (\psi_m - \psi_b)\gamma(x)\right)$$
  拟合出参数后，**在预测时强行令长度差项 $\text{len}(z_m) - \text{len}(z_b) \equiv 0$（即拔掉长度偏见插头）**，直接计算净胜率！这一简单优雅的回归去偏不仅彻底堵死了靠啰嗦刷榜的漏洞，还将 AlpacaEval 与人类 Chatbot Arena 的 **Spearman 秩相关系数从 `0.94` 大幅提升到了 `0.98`**！

#### 2. [WildBench (Lin et al., 2024)](https://arxiv.org/abs/2406.04770)：引入任务专属评分细则清单（Task-Specific Checklists）
直接问裁判模型“你觉得回答 A 和回答 B 哪个更好？”往往是一个定义不清（Ill-defined）的问题。AI2 团队提出的 **WildBench** 从百万条真实人类对话日志（WildChat）中精选出 1,024 个高难度真实任务，并引入了两项关键改进：
1. **针对每道题定制专属评分清单（Task-Specific Checklists / Rubrics）**：在让 GPT-4-turbo 裁判打分前，先根据具体问题列出必须检查的 5～10 条核心核对项（相当于为裁判提供了结构化的评分思维链 CoT），大幅提升了自动判卷的信噪比与可解释性；
2. **五档细粒度比较与硬性长度惩罚**：采用五档打分（`Much better / Slightly better / Tie / Slightly worse / Much worse`），并规定如果获胜方的字符数比失败方超出阈值 $K$ 且仅被评为 `Slightly better`，则强制将其降级为平局 `Tie`。其与 Chatbot Arena 顶部模型的 Pearson 相关系数高达 **`0.98`**（可在 [HELM WildBench](https://www.google.com/search?q=https://crfm.stanford.edu/helm/wildbench/latest/) 查看）。

> **元问题思考：我们该如何“评估一个评测指标本身的好坏（How to evaluate a metric）”？**
> Percy 指出，目前学界最常用的合理性检验（Sanity Check）就是计算新指标与人类盲测榜单（如 Chatbot Arena）的相关系数（Correlation）。但这存在一定的循环论证风险（因为 Chatbot Arena 本身也有人类偏见，并非绝对真理），更稳健的做法是**联合多个不同架构的 LLM 裁判与遵循严格 Rubric 的人类专家进行交叉印证**。

## 5. 支柱四：自主智能体基准（Agentic Benchmarks）—— 从“评估模型说什么”到“评估模型做什么”（`agentic_benchmarks`）

如果说前面的考试和对话基准都在评估**“语言模型说了什么（What LMs say）”**，那么近两年最激动人心的范式跃迁就是评估**“语言模型在真实环境中做出了什么（What LMs do）”**——即 **Agentic Benchmarks（智能体基准）**！

### 5.1 智能体的核心等式与四大代表性基准

$$\mathbf{\text{Agent (智能体系统)} = \text{Language Model (底层基座模型)} + \text{Agent Scaffold (智能体脚手架 / 控制流逻辑)}}$$

在智能体任务中，模型不再是一锤子买卖地吐出一段文本，而是运行在一个循环脚手架（Scaffold）中：观察环境状态 $\to$ 思考并生成工具调用（如执行 Bash 命令、读写文件、运行测试） $\to$ 接收环境反馈 $\to$ 多轮迭代修正直至完成任务。

| 智能体基准 | 论文与年份 | 任务领域与环境设定 | 题目规模与验证判分方式 | 分数演进轨迹与核心启示 |
| :--- | :--- | :--- | :--- | :--- |
| **1. SWE-bench** (& **SWE-bench Verified**) | **[Jimenez et al. (2023)](https://arxiv.org/abs/2310.06770)** | **真实软件工程**：给定完整 Python 代码库与真实 GitHub Issue 描述，要求生成代码修复补丁（PR Patch） | 12 个流行仓库共 2,294 题（后经人工审计过滤掉测试不全的坏题，精选出 500 题构成 **Verified**）；通过运行 **Fail-to-Pass 与 Pass-to-Pass 单元测试** 严格判分 | • 2023 初代 Claude 2 仅 **`1.96%`**<br>• 2024 年 SWE-bench Verified 约 **`16%`**<br>• **2026 年 5 月最新 SOTA（Claude Mythos）已飙升至 `93.9%`！** |
| **2. Terminal-Bench 2.0** | **[Merrill et al. (2026)](https://arxiv.org/abs/2601.11868)** ([tbench.ai](https://www.tbench.ai/)) | **通用计算机命令行终端（CLI）**：在真实 Linux 终端中完成系统配置、编译排错、数据处理等复杂长周期任务 | 由全球 93 位贡献者众包构建 229 题，精选 **89 道硬核任务**构成 2.0 版（人类耗时从 1 小时到数天不等），每题配独立容器环境与测试套件 | 极具区分度：截至 2026 年初，即便是最强的前沿模型与脚手架组合，得分也**普遍低于 `65%`**，且同一模型换不同 Scaffold 性能差异显著 |
| **3. Cybench** | **[Zhang et al. (2024)](https://arxiv.org/abs/2408.08926)** ([cybench.github.io](https://cybench.github.io/)) | **网络安全夺旗赛（CTF, Capture the Flag）**：给定存在漏洞的目标服务器或二进制源码，自主执行渗透攻击夺取隐藏的 `flag` 字符串 | 源自 4 大顶级国际 CTF 竞赛的 **40 道专业赛题**；以人类战队**首次攻破耗时（First-solve time，从 11 分钟到 25 小时）**作为客观难度标尺，并拆分中间子任务（Subtasks） | 2024 年刚发布时顶尖模型仅能解出人类耗时 $\le 11$ 分钟的简单题（总分 `~10%–20%`）；到 **2026 年该基准已接近被完全攻克（Solved）** |
| **4. MLE-bench** | **[Chan et al. (2024)](https://arxiv.org/abs/2410.07095)** | **机器学习工程竞赛（Kaggle）**：阅读赛题、清洗特征、编写训练代码、调参并提交测试集预测文件 | 精选 **75 个真实 Kaggle 竞赛**，直接以人类 Kaggle 排行榜的铜牌/银牌/金牌分数线作为评分基准 | 发布时 `o1-preview + AIDE 脚手架` 在 `16.9%` 的比赛中达到铜牌线；到 2026 年顶尖 Agent 已能在过半数比赛中夺牌，且重度依赖脚手架设计 |

### 5.2 为什么智能体脚手架（Agent Scaffolds）对评测结果有决定性影响？

Percy 对比了早期 Cybench 使用的极简脚手架（把所有历史 Action 和 Observation 一股脑无限拼接到同一个 Context Buffer 里，很快导致上下文爆炸与迷失方向）与 2025–2026 年现代生产级 Agent 脚手架（参考 [LangChain 关于 Agent Harness / Context Engineering 的总结](https://blog.langchain.com/)），指出了现代脚手架的**四大核心工程支柱**：

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│                    现代高性能 Agent Scaffold (智能体脚手架) 的四大核心工程支柱                    │
├─────────────────────────────────────────────────────────────────────────────────────────────────┤
│  1. 显式任务规划 (Explicit Planning):                                                           │
│     拒绝无头苍蝇式的自由流推理；强制维护结构化 To-Do List / task.md，逐项执行并打勾核销。       │
│  2. 层级化子智能体委派 (Hierarchical Delegation):                                               │
│     主控 Agent 将子任务派发给拥有干净独立上下文 (Clean Context) 的 Sub-agent，子 Agent 只向主控 │
│     返回提炼后的最终结论，将海量中间报错与探索细节彻底封装隔离 (Encapsulation)！                 │
│  3. 持久化外部记忆 (Persistent Memory):                                                         │
│     突破有限的 Context Window，显式向磁盘文件系统读写工作笔记、中间状态与代码摘要。             │
│  4. 极致上下文与流程工程 (Extreme Context Engineering):                                         │
│     针对特定基座模型精细调优何时委派、何时切换探索策略、何时回滚重试的元控制指令。              │
└─────────────────────────────────────────────────────────────────────────────────────────────────┘
```

> **核心结论**：在 Agentic 时代，当你看到排行榜上的一个高分时，**你评估的绝不仅仅是底层的语言模型（LM），而是 `Language Model + Agent Scaffold` 的耦合系统**！同一个基座模型搭配不同的脚手架，在 Terminal-Bench 或 MLE-bench 上的得分可以相差数十个百分点。

## 6. 支柱五：纯流体推理基准（Pure Reasoning）—— 剥离世界知识的尝试（`pure_reasoning_benchmarks`）

无论是 MMLU、GPQA 还是 SWE-bench，都需要模型具备海量的语言知识与专业领域背景（结晶智力 Crystallized Intelligence）。于是以 François Chollet 为代表的研究者提出了一个深刻的科学追问：**我们能否把“纯粹的推理与抽象泛化能力（Fluid Intelligence）”从“背诵互联网知识（Memorizing Facts）”中彻底剥离出来单独评测？**

这就是著名的 **[ARC-AGI（Abstraction and Reasoning Corpus）](https://arcprize.org/)** 系列基准的设计哲学：**每一道测试题都是一个前所未见的独特网格变换谜题（Special Snowflake），包含若干个输入-输出彩色网格示例，要求推理出测试输入对应的输出网格；所有题目仅依赖最基础的核心知识先验（物体、对称、计数、拓扑），对普通人类而言 100% 可解，但死记硬背互联网语料毫无帮助**！

### 6.1 从 ARC-AGI-1 到 ARC-AGI-3（2019–2026）的三代演进与启示

```
ARC-AGI 历史准确率跃迁图:
  准确率 ▲
   100% ┼─────────────────────────────────────────────── ● (人类基线 ~85%-100%)
        │                                     ╭─────────╯
        │                                   ╭─╯ ARC-AGI-1 (o1/o3 开启测试时推理后彻底攻克!)
    50% ┼─────────────────────────────────╭─╯
        │                               ╭─╯   ╭─────────► ARC-AGI-2 (2025.03 发布，快速攀升中)
        │                             ╭─╯   ╭─╯
     0% ┼═══●═════════●═════════●═════╯═════╯═══●═══════► ARC-AGI-3 (2026.03 交互式游戏环境，当前 <1%!)
       2019(GPT-2) 2020(GPT-3) 2023(GPT-4) 2024.09(o1) 2026.03
       [纯预训练时代: 分数恒为 ~0%，毫无进展!]   [测试时计算 (Test-Time Compute) 推理模型爆发期!]
```

1. **第一阶段（2019–2024，纯预训练扩展的滑铁卢）**：
   从 2019 年 GPT-2 到 GPT-3 再到 GPT-4，尽管预训练参数量和语料规模扩大了数千倍，但单纯的前向自回归预训练模型在 **ARC-AGI-1** 上的得分几乎纹丝不动（长期徘徊在 `0%` 到个位数），完美验证了 Chollet 的预言：仅靠记忆浅层语料模式无法直接解决即兴抽象变换。
2. **第二阶段（2024–2025，测试时推理模型 o1 / o3 的爆发式破局）**：
   2024 年底，随着 OpenAI 推出 `o1` 与 `o3` 等**强化学习推理模型（Reasoning Models，在测试时生成数万 Token 的内部搜索与假设检验思维链）**，ARC-AGI-1 的得分瞬间垂直起飞并被基本攻克！随后官方于 2025 年 3 月推出了需要更深多步组合推理的 **ARC-AGI-2**，目前前沿推理模型也已将其刷至高位。
   * **Percy 的深层思考**：虽然表面上看预训练知识对做图形题没用，但**如果没有大规模预训练赋予模型强大的代码生成、符号表征与通用世界先验，后续的强化学习推理（o1/o3）根本无从涌现**！
3. **第三阶段（2026 年 3 月最新推出：[ARC-AGI-3](https://arcprize.org/arc-agi/3/) —— 交互式无语言探索环境）**：
   就在本讲开课前一个月（2026 年 3 月），官方发布了第三代 **ARC-AGI-3**：不再是静态的看图填空，而是变成了**完全没有任何文字说明的交互式 2D 网格游戏环境（Interactive Environments）**！智能体必须自己通过试探性动作观察环境反馈、自主推断游戏规则与通关目标、规划动作序列（可通过 $64\times 64$ 图像或 ASCII 文本网格输入给模型）。目前所有前沿模型在 ARC-AGI-3 上的得分再次跌回了个位数。

* **纯推理基准的局限性**：
  第一，要真正把“推理”与“先验知识”100% 正交解耦在哲学与数学上几乎是不可能的（任何谜题依然隐含着人类视觉与物理先验，只要针对该分布合成海量微调数据依然可以刷榜）；第二，ARC-AGI 刻意限制在“普通人类几分钟内能解出”的谜题，**无法衡量超越常人极限的超人类科学推理（Superhuman Reasoning，如攻克国际数学奥林匹克 IMO 金牌题或未解数学猜想）**。

## 7. 支柱六：安全与红队基准（Safety Benchmarks）—— 合规拒答、对抗越狱与双用途困境（`safety_benchmarks`）

在汽车工业中，“安全评测”有极其明确的物理标准（把放着假人的汽车以固定时速撞向混凝土墙，测量传感器受力）。但在人工智能领域，**“什么是安全（What is Safety?）”** 却是一个高度复杂、强依赖社会文化语境的难题。

### 7.1 标准拒答基准、法规对齐分类与自动对抗越狱（GCG）

1. **[HarmBench (Mazeika et al., 2024)](https://arxiv.org/abs/2402.04249)：标准化自动红队与鲁棒拒答评测**
   收集了 510 种违反法律或社会规范的恶意行为指令（如制造生化武器、编写恶意勒索软件、侵犯版权等），评估 18 种自动红队攻击方法（Red Teaming）与 33 个目标模型的**拒答鲁棒性（Robust Refusal）**（详见 [HELM HarmBench](https://www.google.com/search?q=https://crfm.stanford.edu/helm/harmbench/latest/)）。
2. **[AIR-Bench 2024 (Zeng et al., 2024)](https://arxiv.org/abs/2407.17436)：对齐全球政府法规与企业政策的安全分类学**
   过去的学术安全基准往往由研究者凭个人直觉定义风险类别。为了与真实世界的监管接轨，斯坦福团队系统拆解了全球 **8 部政府监管法规（涵盖欧盟 AI 法案、美国行政令与中国监管规定）** 以及 **16 家科技公司的安全使用政策**，构建了一个包含 **4 层架构、底层细化为 314 个细粒度风险类别（Granular Risk Categories）** 的统一安全分类学，并人工精校了 5,694 条测试提示词（详见 [HELM AIR-Bench](https://crfm.stanford.edu/helm/air-bench/latest/)）。
3. **安全护栏的阿喀琉斯之踵：自动对抗越狱攻击（Jailbreaking via [GCG, Zou et al., 2023](https://www.google.com/search?q=https://arxiv.org/abs/2307.25863)）**
   即便模型在后训练（RLHF/SFT）中学到了对正常恶意提问说“对不起，我不能回答”，攻击者依然可以通过**越狱（Jailbreaking）**绕过限制。
   * **GCG（Greedy Coordinate Gradient，贪婪坐标梯度攻击）**：不在自然语言层面绞尽脑汁写角色扮演提示词，而是直接在离散 Token 空间上运行梯度引导的贪婪搜索——寻找一段跟在恶意问题后面的“乱码后缀（Adversarial Suffix）”，使得模型以极高概率吐出肯定回复开头 `"Sure, here is a step-by-step plan to..."`！
   * **惊人的跨模型迁移性（Transferability）**：在完全开源的小模型（如 Vicuna-7B/13B 或 Llama-2）上通过白盒梯度优化搜索出的乱码后缀，**直接复制粘贴到完全闭源的黑盒商业模型（如早期的 ChatGPT、Claude、Bard）上竟然同样能成功触发越狱**！

### 7.2 为什么 AI 安全评测如此棘手？（三大深层张力）

1. **高度的语境与文化相对性（Contextual & Cultural Relativity）**：涉及政治、法律、宗教与社会伦理的边界在不同国家、不同司法管辖区乃至不同企业应用场景下截然不同，不存在放之四海而皆准的单一拒答标准。
2. **风险谱系的极端多样性（与模型能力呈不同相关性）**：
   * 有些风险（如医疗/法律场景下的**幻觉 Hallucinations**）与模型基础能力呈**负相关**——模型越聪明、知识越准，幻觉风险就越低；
   * 有些风险（如**阿谀奉承 Sycophancy**、人类过度依赖 AI 导致**丧失批判性思维 Losing Critical Thinking**、加剧社会经济不平等）则与模型交互设计及社会部署方式深度交织；
   * 而另一些风险（如**协助网络犯罪 Abetting Crimes**）则与模型的极限能力呈**正相关**！
3. **双用途困境（Dual-Use Dilemma）**：
   以第 5 节提到的网络安全智能体（如在 Cybench 上表现极强的 **Claude Mythos** 等模型）为例：一套能够自主扫描源码并构造漏洞利用（Exploit）的强大 Agent，交到黑客手里就是高危的自动化网络攻击武器，但交到安全工程师手里就是全天候自动发现并修补软件漏洞的**渗透测试卫士（Penetration Testing）**！能力本身是一把一体两面的双刃剑。

## 8. 元评测工程准则：真实感（生态效度）、防污染（科学效度）与游戏规则

在讲座的最后一部分，Percy 跳出具体的数据集，从科学方法论的高度审视了任何评测体系都必须面对的三个核心维度：**真实感（Realism / Ecological Validity）**、**科学有效性（Validity：防数据污染与基准质量审计）** 以及 **评测目标的顶层定义（How to think about evaluation）**。

### 8.1 生态效度（Ecological Validity / Realism）：评测有多贴近真实世界？（`realism`）

所谓**生态效度（Ecological Validity）**，是指实验室里的评测分数能在多大程度上预测模型在真实世界生产环境中的表现。
* 标准化考试（如 GPQA、HLE）虽然难，但生态效度极低；
* Chatbot Arena 虽然来自真实网民，但样本分布不受控，且偏向简短的闲聊或单轮问答。

为了真正捕捉**具有高经济价值与专业深度的真实工作流**，2025 年涌现出了三项极具代表性的标杆工作：

| 代表项目 | 论文与年份 | 核心设计理念与数据来源 | 关键实证发现 |
| :--- | :--- | :--- | :--- |
| **1. GDPval** (OpenAI) | **[Patwardhan et al. (2025)](https://arxiv.org/abs/2510.04374)** | 对齐美国劳工统计局（BLS）与 **美国 GDP 贡献最大的前 9 大行业、44 种核心职业**；聘请平均拥有 **14 年从业经验** 的资深行业专家（护士、房产经纪、财务顾问、影视剪辑、工程师等）从真实日常工作中提取完整交付物任务（并开源了 220 题黄金子集） | 前沿模型在真实经济任务上的交付质量正随时间呈近似线性提升，且增加推理预算（Reasoning Effort）、任务上下文与脚手架均能显著提分 |
| **2. MedHELM** (Stanford) | **[Bedi et al. (2025)](https://arxiv.org/abs/2505.23802)** ([MedHELM](https://crfm.stanford.edu/helm/medhelm/latest/)) | 指出“通过美国执业医师资格考试（USMLE 多选题）绝不等于能直接上临床看病”；联合 **29 位临床医生** 构建了涵盖 5 大类、22 子类、**121 项真实临床任务**（病历生成、医患沟通、临床决策支持、行政工作流等）的 35 个公私混合基准，采用 **LLM-Jury（陪审团）** 评分 | LLM-Jury 与临床医生评分的一致性（$\text{ICC}=0.47$）甚至超过了医生与医生之间的一致性（$\text{ICC}=0.43$）；模型在病历书写与医患沟通上表现优异（`0.73–0.85`），但在复杂临床决策与医院工作流上仍有明显短板（`0.53–0.72`） |
| **3. Clio** (Anthropic) | **[Tamkin et al. (2024)](https://arxiv.org/abs/2412.13678)** | **用大模型做隐私保护下的真实流量洞察**：模型厂商手握最真实的百万级用户对话流，但受隐私法规限制人类研究员不能直接窥探原始对话。Clio 利用 Claude 自动对百万条真实对话进行去隐私化特征提取与层次化聚类汇总 | 在零人工接触原始隐私数据的前提下，精准绘制出真实世界中人们到底在用 AI 做什么（如编程、写作、甚至发现日语对话中讨论养老护理的比例显著高于其他语种），直接反哺高真实感基准构建 |

> **核心张力（Realism vs. Privacy）**：最真实的评测莫过于直接从线上真实用户流量中采样构建测试集，但这天然与**用户数据隐私（Privacy）**及商业机密相冲突；如何在保护隐私的前提下提升评测的生态效度，是工业界持续探索的前沿课题。

### 8.2 科学效度（Scientific Validity）：训练-测试集污染（Contamination）与基准质量审计（`validity`）

#### 1. 头号公敌：训练-测试集重叠 / 数据污染（Train-Test Overlap / Contamination）
在机器学习 101 的第一堂课上，老师就会告诫：**绝对不要在测试集上训练（Don't train on your test set）**！
* **前大模型时代的单纯岁月**：像 ImageNet、SQuAD 这样的数据集都有严格界定的 Train Split 和 Test Split，所有研究者都在同一份公开训练集上训练、在同一份测试集上评测，界限分明。
* **大模型时代的“黑盒互联网乱世”**：现代预训练语料动辄抓取整个互联网（数万亿 Token），且各大商业模型与开放权重模型往往**完全不公开训练数据的具体构成**！更微妙的是，数据污染往往不是研发者故意把测试集塞进训练集，而是**互联网上充斥着对经典基准题目的讨论、改写、刷题博客或同源教材（Subtle Derivative Contamination）**。

面对无处不在的数据污染嫌疑，我们有哪些科学应对之道？Percy 总结了**四条核心路线（Four Routes）**：

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│                    应对大模型训练-测试集数据污染 (Contamination) 的四大科学路线                  │
├─────────────────────────────────────────────────────────────────────────────────────────────────┤
│  • 路线 1 (事后黑盒取证): 利用数据集样本可交换性检验 (Exchangeability Test, Oren et al., 2023)  │
│  • 路线 2 (建立社区规范): 强制要求模型发布方披露训练-测试重叠统计 (Reporting Norms, Zhang 2024) │
│  • 路线 3 (动态时效更新): 持续抓取模型知识截止日期后的全新网页/竞赛题 (LiveCodeBench 等)        │
│  • 路线 4 (私有留出评测): 维护绝不公开上网的内部代码库或私人手稿 (最适合内部 Perplexity 迭代!)  │
└─────────────────────────────────────────────────────────────────────────────────────────────────┘
```

* **路线一：黑盒统计取证——利用数据点的可交换性（Exchangeability，[Oren, Meister, Chatterji, Ladhak & Hashimoto, 2023](https://arxiv.org/abs/2310.17623)）**：
  如果一个模型是闭源黑盒（不公开训练集、不公开权重，仅支持查询对数概率 $\log p$），如何提供具有严格统计保证（$p$-value）的证据，证明它偷看了某个公开测试集 $D = (x_1, x_2, \dots, x_n)$？
  Tatsu 课题组提出了一个极其精妙的统计检验：
  * **核心洞察**：公开测试集文件在存储在 GitHub 或网页上时，其内部的样本排列顺序 $(x_1, x_2, \dots, x_n)$ 是一个特定的**官方标准顺序（Canonical Ordering）**。但从数学本质上讲，测试集里的各道题目是独立同分布（或可交换 Exchangeable）的。
  * **原假设 $H_0$（未发生污染）**：如果模型从未在预训练语料中见过这个测试集文件，那么把这 $n$ 道题按**官方标准顺序**拼接起来喂给模型算出的联合对数似然 $\log p(x_1, x_2, \dots, x_n)$，与把这 $n$ 道题**随机打乱顺序（Random Permutation $\pi$）**后算出的对数似然 $\log p(x_{\pi(1)}, x_{\pi(2)}, \dots, x_{\pi(n)})$，在分布上应当是**完全等价、没有显著差异的**！
  * **备择假设 $H_1$（发生污染）**：如果模型在预训练时直接背诵了该测试集文件，由于自回归模型具有极强的跨样本顺序记忆倾向，**它对官方标准顺序 $(x_1, \dots, x_n)$ 赋予的对数概率，会显著高于随机打乱顺序后的对数概率**！通过对比标准顺序与 $M$ 个随机置换顺序的对数似然排名，就能在不看训练数据的情况下抓出数据污染现行！（详见本文课后练习 3 的数学推导）。
* **路线二：推动学术与工业界报告规范（Reporting Norms，[Zhang et al., 2024](https://arxiv.org/abs/2410.08385)）**：
  调查发现，在 30 家主流大模型研发机构中，仅有 9 家报告了训练-测试集重叠率（其中 4 家开源了训练数据，5 家公布了去重与重叠统计方法）。就像统计学论文必须报告置信区间（Confidence Intervals）一样，社区应建立硬性规范：**任何宣称在公开基准（如 GPQA、MMLU）上取得高分的模型报告，都必须同步公开其训练-测试集 $n$-gram / 语义重叠率审计报告**。
* **路线三：动态滚动构建全新时效基准（Fresh Evals：如 [LiveCodeBench](https://livecodebench.github.io/)、UncheatableEval）**：
  放弃已被污染的静态老题库，每个月自动抓取各大编程竞赛（LeetCode、AtCoder、Codeforces）、最新 arXiv 论文或新发布 GitHub 仓库中**晚于模型训练截止日期（Cutoff Date）**的新题。
  * *Percy 提醒的隐患*：**时间戳（Timestamps）并不总是安全的防线**！因为网上新发布的 GitHub 代码或新比赛题，很多也是从几年前的老代码库或老经典母题轻微改写（Copy/Derive）而来的，模型依然可能通过老母题占到便宜。
* **路线四：构建绝不公开的私有评测集（Private Evals）**：
  大型科技公司内部普遍使用未公开的私有代码库与内部业务文档进行评测；对于个人研究者或小团队，Percy 幽默地支招：**你可以用你自己写的、从未发到网上的私人文档（例如读博期间被拒稿且从未挂上 arXiv 的草稿论文！）来评测模型的 Perplexity**——因为只要从未触网，就 100% 绝对干净！

#### 2. 评测数据集本身的质量危机：错题、幻觉标签与智能体评测漏洞
很多时候模型没拿到 100% 满分，不是因为模型笨，而是因为**测试集里的题目本身就是坏的（Broken Questions / Label Errors）**！
* **静态基准的“铂金清洗”（[Platinum Benchmarks, Vendrow et al., 2025](https://arxiv.org/abs/2502.03461)）**：
  MIT Mądry 团队对 15 个最热门的公开基准（包括 GSM8K、MMLU、VQA 等）进行了逐题深度审计，发现当把所有前沿大模型都做错的题目拎出来看时，**绝大部分所谓的“模型错误”实际上是数据集自身的错标、题干缺失关键条件（例如题干问“根据下方曲线求斜率”，但文本里根本没给曲线！）或存在歧义（例如看一张只露出婴儿上半身的模糊照片问“婴儿穿袜子了吗？”）**！在剔除这些坏题构建出干净的 **Platinum Benchmarks** 后，前沿模型的真实可靠性与剩余真盲区才得以清晰显现。
* **智能体基准的系统性漏洞与 ABC 检查清单（[Zhu et al., 2025](https://arxiv.org/abs/2507.02825)）**：
  相比静态选择题，评估整个交互环境的 Agentic Benchmarks 更容易隐藏致命漏洞：
  1. **测试样例覆盖不足（Insufficient Test Cases）**：在原始 SWE-bench 中，有些任务的单元测试过于单薄，导致模型写的补丁哪怕破坏了边缘逻辑也能蒙混过关（这也是为什么 OpenAI 后来不得不耗费大量人力筛选重构出 **SWE-bench Verified**）；
  2. **奖励与判分函数设计漏洞（Reward Hacking / Trivial Agent）**：[Zhu et al. (2025)](https://arxiv.org/abs/2507.02825) 审计发现，在著名的工具调用基准 **$\tau$-bench (TAU-bench)** 中，由于判分脚本逻辑缺陷，**一个什么都不干、每次直接输出空字符串（Empty Response）的摆烂 Agent 竟然能直接拿到 `38%` 的成功率**！某些基准由于环境配置漏洞，会导致智能体性能被高估或低估达 **100%（相对误差）**！为此，作者提出了 **Agentic Benchmark Checklist (ABC)**（应用在 CVE-Bench 上直接消除了 `33%` 的虚高高估）。
* **工程铁律：永远用肉眼和工具去审计运行轨迹（Inspect the Outputs & Traces）！**
  Percy 强烈推荐了开源定性分析工具 **[Docent (Transluce)](https://transluce.org/introducing-docent)**（利用大模型自动聚类、检索和审查海量的 Agent 交互轨迹日志以定位异常模式），并送给所有学生一句忠告：**无论你是在设计一个新基准，还是在现有基准上评测你的模型，绝对不要只看终端打印出的那一个冷冰冰的平均准确率数字——永远打开预测结果和交互轨迹亲眼看一看（Always look at the outputs），确认你测量的真的是你想测量的能力！**

### 8.3 顶层方法论：我们究竟在为了什么目的、评估什么对象？（`how_to_think_about_evaluation`）

在全讲结尾，Percy 总结了思考任何评测问题时的两个元维度：

#### 1. 明确评测的四大不同目的（What's the Point of Evaluation?）
世界上**不存在唯一万能的终极评测（No one true evaluation）**，最佳评测方案完全取决于你试图回答什么问题：
* **目的 A（采购与选型决策 Purchase Decision）**：企业或用户想在特定业务场景（如客服机器人、医疗文书）中从模型 A 和模型 B 之间做二选一 $\implies$ 应构建高度贴合自身业务分布与成本约束的垂直领域私有评测集。
* **目的 B（探索智能科学边界 Capability Measurement）**：学术研究者想度量通用的流体推理或知识上限 $\implies$ 应选用如 HLE、ARC-AGI-3、GPQA 等高难度去污染基准。
* **目的 C（评估社会收益与风险 Policy & Safety Governance）**：监管机构或安全团队想评估模型的经济替代效应与安全隐患 $\implies$ 应采用 GDPval、AIR-Bench、HarmBench 与 Clio 式的真实流量审计。
* **目的 D（指导模型内部研发迭代 Model Development Feedback）**：模型预训练算法工程师想快速决定架构修改或数据清洗配方是否有效 $\implies$ 需要的是**跨算力尺度平滑演进、低噪声、高灵敏度**的内部指标（如干净私有留出集上的 **Perplexity / BPB** 与轻量级核心基准）。

#### 2. 明确评测的对象：是在评估“算法（Methods）”还是在评估“系统（Models / Systems）”？
```
[前基座模型时代: 评估方法 (Evaluating Methods)]
  固定统一的数据集 (Train Split) + 固定算力 ──► 运行不同算法 A vs. B ──► 统一测试集 (Test Split)
  • 唯一变量是“算法本身”，严格控制变量，极大促进了学术界的算法创新（如今的代表特例：Modded-NanoGPT Speedrun!）

[现代大模型时代: 评估端到端产物 (Evaluating Models & Agent Systems)]
  各显神通 (Anything Goes: 私有数据 + 任意算力 + 后训练 + 复杂 Scaffold) ──► 最终黑盒模型/系统 ──► 评测基准
  • 对下游用户极有实用价值（因为用户用的就是最终成品），但混杂了数据、算力与工程脚手架，无法归因单一算法优劣!
```

* Percy 强调：这两种游戏本身都有巨大的价值——像 **[Modded-NanoGPT Speedrun](https://github.com/KellerJordan/modded-nanogpt)**（以及 CS336 各次作业的 Leaderboard）那样锁定训练数据、锁定硬件时长，比拼谁能最快达到目标验证集 Loss，是在严格**评估算法（Evaluating Methods）**（正是这种受控竞赛孕育出了我们在 Lecture 11 学到的 Muon 优化器！）；而像 SWE-bench、Arena AI 这样不限训练数据与脚手架的榜单，则是在**评估端到端模型与系统（Evaluating Models/Systems）**。**最关键的是：在开始任何评测之前，必须先清晰声明你的“游戏规则（State the rules of the game）”！**

## 9. 全景总结：大模型六大评测范式权衡速查表（Cheat Sheet）

| 评测范式 | 代表基准 / 平台 | 评分机制 | 难度上限 | 生态真实感 (Realism) | 抗污染与效度挑战 (Validity) | 最核心适用场景 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1. 困惑度 (Perplexity)** | PTB, WikiText-103, 1BW, LAMBADA, HellaSwag, 私有留出集 | 精确计算序列对数似然 $\exp(-\frac{1}{N}\sum \log p)$ | 连续无上限（以真实熵 $H(t)$ 为下界） | 低（衡量分布拟合，非直接交互） | 公开集易污染，但**极易构建绝对干净的私人手稿/内部代码留出集**；黑盒榜单需防未归一化作弊 | **预训练内部迭代、Scaling Laws 拟合、数据配比消融（Assignment 3 & 4 核心指标）** |
| **2. 标准化考试 (Exams)** | [MMLU](https://arxiv.org/abs/2009.03300), [MMLU-Pro](https://arxiv.org/abs/2406.01574), [GPQA](https://arxiv.org/abs/2311.12022), [HLE](https://arxiv.org/abs/2501.14249) | 多选题选项匹配或简答题精确/规则匹配 | **极高**（可召集全球博士出题至人类极限） | **低**（现实用户极少问闭合多选题） | 静态公开题库极易被网络衍生讨论污染；饱和速度快（生命周期约 1–2 年） | 度量专业学科知识储备与封闭式深度学术推理能力 |
| **3. 开放式对话 (Chat)** | [Chatbot Arena](https://arxiv.org/abs/2403.04132), [AlpacaEval 2.0](https://arxiv.org/abs/2404.04475), [WildBench](https://arxiv.org/abs/2406.04770) | 人类盲测 Elo (Bradley-Terry) 或带 Rubric 的 LLM-as-a-Judge | 中（受限于日常单轮/多轮提问复杂度） | **高**（直接源自真实人类开放式提问） | 需防范**长度偏见（Length Bias）**、格式偏见、阿谀奉承（Sycophancy）与外行裁判误判 | 评估后训练（SFT / RLHF）指令遵循、文风对齐与日常助手体验 |
| **4. 自主智能体 (Agentic)** | [SWE-bench](https://arxiv.org/abs/2310.06770), [Terminal-Bench 2.0](https://arxiv.org/abs/2601.11868), [Cybench](https://arxiv.org/abs/2408.08926), [MLE-bench](https://arxiv.org/abs/2410.07095) | 容器环境内执行单元测试、夺取 Flag 或 Kaggle 竞赛排名 | **极高**（长周期多步工具交互） | **极高**（高度还原真实软件与工程工作流） | **强耦合模型与脚手架（Scaffold）**；需遵循 [ABC 准则](https://arxiv.org/abs/2507.02825)防范测试不全与空响应漏洞 | 评估模型在真实工具环境中的长程规划、代码修复与自主执行能力 |
| **5. 纯流体推理 (Pure Reasoning)** | [ARC-AGI-1 / 2 / 3](https://arcprize.org/) | 2D 网格精确匹配或交互式环境通关步数 | 高（对 AI 极难，但受限于普通人类可解范围） | 低（抽象几何网格与合成小游戏） | 每题独特（抗简单记忆），但仍可能被合成先验刷榜；无法衡量超人类科学证明 | 剥离世界知识，专项检验测试时计算（Test-Time Compute）与即兴模式归纳 |
| **6. 安全与真实经济 (Safety & Econ)** | [HarmBench](https://arxiv.org/abs/2402.04249), [AIR-Bench](https://arxiv.org/abs/2407.17436), [GDPval](https://arxiv.org/abs/2510.04374), [MedHELM](https://arxiv.org/abs/2505.23802) | 拒答率 / ASR（攻击成功率）或专家/LLM陪审团按行业细则打分 | 中～高 | **极高**（对齐多国法规、GDP 44 职业与临床实务） | 安全标准随国别文化动态变化；需防范 [GCG](https://www.google.com/search?q=https://arxiv.org/abs/2307.25863) 等梯度优化越狱攻击与隐私泄漏 | 企业级生产部署准入审查、安全红队测试与垂直行业（医疗/金融）选型 |

## 10. 课后思考与自测练习（Review & Self-Check）

1. **全句困惑度（Full Perplexity）vs. 条件困惑度（Conditional Perplexity）与黑盒作弊构造**：
   假设测试集中有一条问答样本 `"Q: Where is Stanford? A: California"`，分词后共包含 $L = 7$ 个 Token：前缀提示词 `prompt` 为前 5 个 Token `["Q:", " Where", " is", " Stanford", "?"]`，目标回答 `response` 为后 2 个 Token `[" A:", " California"]`。
   假设某语言模型 $p$ 对这 7 个 Token 依次赋予的条件概率为：
   $$p(x_1)=e^{-4.0}, \; p(x_2\mid x_1)=e^{-3.0}, \; p(x_3\mid x_{<3})=e^{-1.0}, \; p(x_4\mid x_{<4})=e^{-5.0}, \; p(x_5\mid x_{<5})=e^{-0.5}, \; p(x_6\mid x_{<6})=e^{-0.2}, \; p(x_7\mid x_{<7})=e^{-0.3}$$
   * (a) 请分别计算该模型在这条样本上的**全句困惑度（Full Perplexity）**与仅针对回答部分的**条件困惑度（Conditional Perplexity）**（保留两位小数，已知 $e^{2.0} \approx 7.39, e^{0.25} \approx 1.28$）。为什么在评估问答能力时条件困惑度更合理？
   * (b) 假设有人举办了一个黑盒 Perplexity 比赛（评测服务器把测试文本发给参赛者的函数 `log_prob_fn(tokens)`，直接用返回的数值算 Perplexity）。如果某个作弊者内部使用了一个正常的模型 $p(x_t \mid x_{<t})$，但在返回对数概率时偷偷给每个 Token 的对数概率都加上了一个正数 $c = +1.0$（即返回 $\tilde{\ell}_t = \log p(x_t \mid x_{<t}) + 1.0$）。请问他的 Perplexity 会变成原来的多少倍？为什么这个返回的 $\tilde{\ell}_t$ 违反了概率公理？
   * *参考解答*：
     * (a) **全句困惑度**对全部 7 个 Token 的负对数似然求平均：
       $$\text{PPL}_{\text{full}} = \exp\!\left(\frac{4.0 + 3.0 + 1.0 + 5.0 + 0.5 + 0.2 + 0.3}{7}\right) = \exp\!\left(\frac{14.0}{7}\right) = e^{2.0} \approx \mathbf{7.39}$$
       **条件困惑度**仅对后 2 个回答 Token（`[" A:", " California"]`）的负对数似然求平均：
       $$\text{PPL}_{\text{cond}} = \exp\!\left(\frac{0.2 + 0.3}{2}\right) = e^{0.25} \approx \mathbf{1.28}$$
       **工程原因**：前缀 `"Q: Where is Stanford?"` 中包含了高达 $4.0 + 3.0 + 1.0 + 5.0 + 0.5 = 13.5\text{ nats}$ 的背景提问不确定性（即用户本来可以问任何问题），全句 PPL 被提问本身的熵严重主导，掩盖了模型其实以极高置信度（$e^{-0.25} \approx 77.9\%$）精准答出 `"California"` 的事实。
     * (b) 给每个 Token 的对数似然加上 $+1.0$，会使平均负对数似然减小 $1.0$，从而使算出的 Perplexity 直接缩小为原来的 **$\frac{1}{e} \approx 0.368\text{ 倍}$**！这之所以是非法的，是因为此时对词表所有可能 Token 的伪概率求和变成了 $\sum_{v \in V} \exp(\log p(v \mid x_{<t}) + 1) = e \cdot \sum_{v \in V} p(v \mid x_{<t}) = \mathbf{e \approx 2.718 > 1}$，违背了概率分布归一性公理 $\sum_{v \in V} p(v) = 1$。

2. **Chatbot Arena 的 Elo 等级分与 AlpacaEval 2.0 长度去偏回归计算**：
   * (a) 在 Chatbot Arena 中，模型 $A$ 击败模型 $B$ 的概率公式为 $P(A \text{ beats } B) = \frac{1}{1 + 10^{(\text{ELO}_B - \text{ELO}_A)/400}}$。假设模型 $A$ 的等级分为 $\text{ELO}_A = 1350$，模型 $B$ 的等级分为 $\text{ELO}_B = 1150$。请计算在单局盲测中模型 $A$ 击败模型 $B$ 的理论概率（保留三位小数，已知 $10^{-0.5} = \frac{1}{\sqrt{10}} \approx 0.3162$）。反之，如果观测到模型 $C$ 对阵模型 $B$（$\text{ELO}_B = 1150$）时的长期真实胜率为 **$90.91\%$**（即 $\frac{10}{11}$），请问模型 $C$ 的 $\text{ELO}_C$ 是多少？
   * (b) 在 Length-Controlled AlpacaEval 2.0 中，假设拟合出的裁判偏好逻辑回归模型简化为：
     $$P(\text{Model } m \text{ wins against Baseline } b) = \sigma\!\left(\theta_m - \theta_b + 0.5 \cdot \tanh\!\left(\frac{\text{len}(z_m) - \text{len}(z_b)}{500}\right)\right)$$
     其中 $\sigma(u) = \frac{1}{1 + e^{-u}}$ 为标准 Sigmoid 函数，基准模型参数 $\theta_b = 0$。某个经过“话痨微调”的模型 $m$ 真实能力参数仅为 $\theta_m = -0.2$（实际上不如基准模型），但它在回答时平均每次比基准模型多写 $1000$ 个字符（使得 $\tanh(1000/500) = \tanh(2) \approx 0.964$）。请对比该模型在**未去偏原始 AlpacaEval** 下的表观胜率与在**长度控制（Length-Controlled）AlpacaEval 2.0** 下的真实去偏胜率！
   * *参考解答*：
     * (a) 代入 $\text{ELO}_B - \text{ELO}_A = 1150 - 1350 = -200$：
       $$P(A \text{ beats } B) = \frac{1}{1 + 10^{-200/400}} = \frac{1}{1 + 10^{-0.5}} \approx \frac{1}{1 + 0.3162} \approx \mathbf{0.760\text{ (即 } 76.0\%\text{)}}$$
       若模型 $C$ 对阵 $B$ 的胜率为 $\frac{10}{11}$，则 $\frac{1}{1 + 10^{(\text{ELO}_B - \text{ELO}_C)/400}} = \frac{10}{11} \implies 10^{(\text{ELO}_B - \text{ELO}_C)/400} = \frac{1}{10} = 10^{-1}$，故 $\frac{\text{ELO}_B - \text{ELO}_C}{400} = -1 \implies \text{ELO}_C = \text{ELO}_B + 400 = 1150 + 400 = \mathbf{1550}$。
     * (b) **未去偏原始胜率**：代入 $\Delta \text{len} = +1000$，得对数几率（Logit）为 $u_{\text{raw}} = -0.2 - 0 + 0.5 \times 0.964 = \mathbf{+0.282} > 0$，对应的表观胜率为 $\sigma(0.282) = \frac{1}{1 + e^{-0.282}} \approx \mathbf{57.0\%}$（看起来明显击败了基准模型！）。
       **长度控制（Length-Controlled）去偏胜率**：预测时强行干预令 $\text{len}(z_m) - \text{len}(z_b) = 0$（即 $\tanh(0) = 0$），此时净对数几率为 $u_{\text{LC}} = \theta_m - \theta_b = \mathbf{-0.20} < 0$，对应的去偏胜率回落至 $\sigma(-0.20) = \frac{1}{1 + e^{0.20}} \approx \mathbf{45.0\%}$，成功识破了靠凑字数刷榜的伪装！

3. **利用可交换性（Exchangeability）进行黑盒测试集污染置换检验（[Oren et al., 2023](https://arxiv.org/abs/2310.17623)）**：
   假设你想检验某闭源模型 $p$ 是否在预训练时偷看了包含 $n$ 道题目的公开基准测试集 $D_{\text{canon}} = (x_1, x_2, \dots, x_n)$。
   你生成了 $M = 99$ 个独立均匀随机打乱题目顺序的置换版本 $D_{\pi_1}, D_{\pi_2}, \dots, D_{\pi_{99}}$，并分别调用模型计算了官方标准顺序的对数似然 $L_0 = \log p(D_{\text{canon}})$ 以及 99 个随机置换顺序的对数似然 $L_1, L_2, \dots, L_{99}$。
   * (a) 如果原假设 $H_0$（模型未受该测试集文件污染，且测试集各题满足可交换性）成立，那么在全部 $M + 1 = 100$ 个对数似然数值 $\{L_0, L_1, \dots, L_{99}\}$ 中，官方标准顺序的似然 $L_0$ 恰好排在**严格第 1 名（即 $L_0 > \max_{1 \le k \le 99} L_k$）**的概率是多少？此时对应的非参数置换检验 $p$-value 是多少？
   * (b) 为什么在计算每个置换序列 $D_\pi = (x_{\pi(1)}, \dots, x_{\pi(n)})$ 的对数似然时，必须将多道题目拼接在同一个上下文窗口内计算 $\sum_{i=1}^n \log p\big(x_{\pi(i)} \mid x_{\pi(1)}, \dots, x_{\pi(i-1)}\big)$，而不能把每道题单独喂给模型计算 $\sum_{i=1}^n \log p(x_{\pi(i)})$？
   * *参考解答*：
     * (a) 在原假设 $H_0$ 下，由于数据集满足可交换性且模型从未见过官方文件的特定排列顺序，$L_0, L_1, \dots, L_{99}$ 这 $100$ 个随机变量是完全对称（可交换）的，任何一个排列取得最大值的概率均相等，等于 $\frac{1}{M + 1} = \frac{1}{99 + 1} = \mathbf{0.01\text{ (即 } 1\%\text{)}}$。因此，当观测到 $L_0$ 高居全部 100 个排列的榜首时，拒绝“未污染”原假设的精确置换检验 **$p\text{-value} = \frac{1}{100} = \mathbf{0.01 < 0.05}$**，具有显著的统计学证据表明该模型记忆了测试集的官方样本顺序！
     * (b) 因为加法满足交换律！如果把每道题单独隔离开计算无条件对数似然再相加，由于 $\sum_{i=1}^n \log p(x_{\pi(i)}) \equiv \sum_{i=1}^n \log p(x_i)$，任何置换顺序算出来的总和在数学上都**恒等不变**！只有将多道题首尾相接放入同一个上下文窗口内，利用自回归条件概率 $\log p(x_{\pi(i)} \mid x_{\pi(<i)})$ 捕捉**跨题目的转移概率（Cross-Example Transition Probability）**，才能检测出模型是否背诵了“第 $i-1$ 题后面紧跟着第 $i$ 题”的文件顺序特征。
