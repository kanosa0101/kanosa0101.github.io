---
title: Stanford CS336 (Spring 2026) Lecture 15 深度课程笔记：Post-Training (Part 1)
date: 2026-10-26 14:32:00
tags:
  - 大模型
  - 后训练
  - SFT
  - 对齐算法
  - RLHF
  - DPO
  - 知识幻觉
  - CS336
categories:
  - [笔记, CS336]
description: 系统推演 Stanford CS336 Lecture 15：大语言模型后训练核心算法与工程实践，系统剖析 SFT 指令微调数据工程、知识与幻觉解耦、安全对齐边界，以及 Mid-Training 退火混入和从 RLHF 到 DPO 的闭式解数学推导。
mathjax: true
---

# Stanford CS336 (Spring 2026) Lecture 15 深度课程笔记：Post-Training (Part 1)

> **课程信息**：[Stanford CS336: Language Modeling from Scratch (Spring 2026)](https://cs336.stanford.edu/)
> **主讲教授**：Tatsunori Hashimoto (Tatsu)
> **本讲资源**：[Lecture 15 YouTube 视频](https://www.youtube.com/watch?v=2oH6PWPrYFo) | [官方幻灯片 (`lecture_15.pdf`)](https://github.com/stanford-cs336/lectures/blob/main/lecture_15.pdf) | [课件仓库 (`stanford-cs336/lectures`)](https://github.com/stanford-cs336/lectures) | [Assignment 5 指导手册与代码库 (`assignment5-alignment`)](https://github.com/stanford-cs336/assignment5-alignment)

## 1. 引言：从预训练“原始汤”到可交互智能助手的跨越

在前 14 讲中，我们完整走过了预训练（Pre-training）的全流程：从分词器与 Transformer 架构（Assignment 1）、算子融合与分布式并行（Assignment 2）、扩展定律（Assignment 3）到万亿级网页清洗、去重与配比（Assignment 4）。Tatsu 在本讲开篇指出：如果你把前 14 讲的技术全部用上，再堆入更多的算力（FLOPs）与干净数据，你最终得到的是一个**“超级加强版的 GPT-3 基座模型（Souped-up Base Model）”**。

然而，纯粹的预训练基座模型在实际应用中的效用存在巨大的天花板：
* **GPT-3 基座模型的局限**：如果你回到 2020 年去直接和原版 `GPT-3 (davinci)` 对话，你会发现它几乎无法执行复杂的零样本指令（Zero-shot Instruction Following），也无法进行细粒度的行为控制（Fine-grained Steering）。你必须绞尽脑汁构造少样本提示词（Few-shot Prompts）去诱导它续写文本；一旦用户输入长而复杂的程序化指令（Programmatic Prompts），基座模型往往只会顺着你的问题继续往下编造更多问题，而不是回答问题。
* **后训练（Post-training）的两大历史阶跃（對應 Lecture 15 与 Lecture 16）**：
  1. **第一阶跃（Lecture 15 本讲焦点：Mid/Post-Training — SFT & RLHF）**：如何将 `GPT-3` 式的预训练基座模型转化为具备强指令遵循能力、符合人类偏好与安全规范的 `ChatGPT / GPT-3.5 / GPT-4` 级对话与工具调用助手？
  2. **第二阶跃（Lecture 16 下一讲焦点：Post-Training — RLVR）**：如何进一步通过可验证奖励强化学习（Reinforcement Learning with Verifiable Rewards），将 `ChatGPT` 式的助手转化为 `OpenAI o1 / DeepSeek-R1` 式的深度长思维链推理模型（Thinking Models）？

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│                    大模型能力演进的三阶段范式：从“原始汤”到助手再到深度推理者                   │
├─────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                 │
│  [Stage 1: 预训练 (Pre-training, Lectures 1–14)]                                                │
│    • 本质: 大规模自回归压缩互联网世界知识，形成能力宽泛但杂乱的“原始汤 (Primordial Soup)”       │
│    • 产物: Base Model (如 GPT-3) ──► 强通识容量，但缺乏指令遵循、格式受控性与安全边界           │
│          │                                                                                      │
│          ▼  (Lecture 15: Mid-training + SFT + RLHF / DPO)                                       │
│  [Stage 2: 指令微调与人类偏好对齐 (Instruction Following & Alignment)]                          │
│    • 本质: 从预训练“原始汤”中显式提取并塑造期望行为；从“拟合分布”转向“最大化奖励”               │
│    • 产物: Instruct / Chat Model (如 ChatGPT / Claude / Llama-3-Instruct)                       │
│          │                                                                                      │
│          ▼  (Lecture 16: RLVR — 数学/代码可验证奖励强化学习)                                    │
│  [Stage 3: 推理强化学习 (Reasoning RL / Thinking Models)]                                       │
│    • 本质: 在无奖励黑客风险的客观验证器上大规模投入推理与强化学习算力，涌现自反思与搜索         │
│    • 产物: Reasoning Model (如 OpenAI o1 / DeepSeek-R1)                                         │
└─────────────────────────────────────────────────────────────────────────────────────────────────┘
```

### 1.1 为什么说后训练是一门“高度手工艺化（Artisanal）”且高度保密的工程？

Tatsu 坦诚地提醒大家：与预训练架构和分布式系统那种“算法清晰、公开透明”的工程不同，**前沿大模型的后训练（Post-training）在很大程度上是一门极其繁琐、充满脏活累活的手工艺（Artisanal Process），且其核心配方在工业界属于最高级别的商业机密（Trade Secrets）**：
1. **公开文献的时间断层**：
   在 ChatGPT 引爆全球商业竞争之前，前沿实验室曾发表过极其详尽的后训练论文——例如 OpenAI 的 **[Learning to Summarize from Human Feedback (Stiennon et al., 2020)](https://arxiv.org/abs/2009.01325)**、**[InstructGPT (Ouyang et al., 2022)](https://arxiv.org/abs/2203.02155)** 以及 Anthropic 的 **[HH-RLHF (Bai et al., 2022)](https://arxiv.org/abs/2204.05862)**。翻开这些早期论文的附录（Appendices），你甚至能看到他们发给外包标注员的完整标注手册、界面截图与薪资结构。然而自 2023 年之后，没有任何一家闭源前沿实验室再公开过其后训练数据的具体采集指南。
2. **商业间谍级的“数据逆向工程”**：
   Tatsu 展示了一份 2023 年意外泄露的数据标注巨头 **Scale AI** 的内部文档截图：当时 Google 正在利用 Scale AI 紧急改进早期的 **Google Bard**，因为内部测试发现 Bard 的回答质量远逊于刚发布的 GPT-4。在泄露的内部讨论中，项目团队直言必须组织精锐标注员去“逆向拆解（Reverse-engineer）为什么 GPT-4 的回答更好”，并要求标注员针对相同提示词写出比 GPT-4 更详尽、排版更清晰的参考回答。
3. **开源界与闭源界的路径分化**：
   虽然我们在后训练**算法层面**（如 SFT、PPO、DPO、GRPO）拥有非常透彻的数学理解，但在后训练中**真正的杠杆（Leverage）在于数据而非算法**。当前开源模型（Open-weight models）的后训练配方高度依赖对闭源前沿模型的**合成蒸馏（Distillation）**，这与闭源前沿实验室为了突破人类能力上限而进行的**大规模专家人工标注（Human Data Collection）**在性质上存在本质差异。

### 1.2 经典后训练的两阶段标准配方（The Two-Part Recipe）

无论是 [Stiennon et al. (2020)](https://arxiv.org/abs/2009.01325) 还是 [InstructGPT (Ouyang et al., 2022)](https://arxiv.org/abs/2203.02155)，经典的后训练管线都遵循极其清晰的两步走范式：
1. **第一阶段：监督微调（Supervised Fine-Tuning, SFT）**：
   收集一批指令提示词 $x$，让人类标注员（或强大的教师模型）编写高质量的示范回答（Demonstration Response）$y$，在 $(x, y)$ 数据对上使用与预训练完全相同的交叉熵损失进行微调。
2. **第二阶段：基于人类/AI 反馈的强化学习（RLHF / RLAIF）**：
   使用 SFT 后的模型针对同一提示词 $x$ 采样生成多个候选回答 $(y_1, y_2, \dots, y_K)$，由人类标注员（或评判模型）对回答进行成对优劣排序（Pairwise Preference Ranking，$y_w \succ y_l$），据此训练奖励模型 $r_\phi(x, y)$ 并通过强化学习（PPO）或直接偏好优化（DPO）最大化策略的期望奖励。

## 2. 第一阶段：监督微调（SFT）数据集的四代演进史

由于 SFT 在算法上只是标准的自回归交叉熵梯度下降，**SFT 的全部核心都在于训练数据的演进与构造**。Tatsu 按时间脉络梳理了开源社区 SFT 数据集从 2021 年到 2026 年的四代演进范式：

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│                          开源社区 SFT 指令微调数据集的四代演进图谱                              │
├─────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                 │
│  [Gen 1: 传统 NLP 基准拼接 (2021–2022)]                                                         │
│    • 代表作: FLAN (Google T5/PaLM), Super-NaturalInstructions, P3 (T0)                          │
│    • 思路: 将上百个现成 NLP 数据集 (翻译/摘要/分类/Enron邮件) 套上指令模板拼成百万级大拼盘      │
│    • 缺陷: 提示词极不自然 (指令挂在末尾)、回答极短且机械、继承了原数据集的幻觉与噪声            │
│          │                                                                                      │
│          ▼                                                                                      │
│  [Gen 2: 大模型自举与早期对话蒸馏 (2022.12–2023)]                                               │
│    • 代表作: Self-Instruct (Wang et al.), Stanford Alpaca (52K), Vicuna (ShareGPT 70K)          │
│    • 思路: 用 text-davinci-003 / ChatGPT 自动生成指令与长篇对话式回答，或收集用户分享的真实对话 │
│    • 突破: 首次证明在强基座 (LLaMA) 上仅需几万条自然对话风格的样本即可激活类 ChatGPT 交互体验!  │
│          │                                                                                      │
│          ▼                                                                                      │
│  [Gen 3: 社区众包专家标注 vs. 复杂合成指令演化 (2023–2024)]                                     │
│    • 众包人类路线: OpenAssistant (OASST1) ──► 仿照 Wikipedia 动员全球志愿者手写高质量多轮对话树 │
│    • 合成演化路线: WizardLM (Evol-Instruct), UltraChat, Tülu 3 (AI2) ──► 复杂约束合成与多源融合 │
│          │                                                                                      │
│          ▼                                                                                      │
│  [Gen 4: 智能体与并行工具调用轨迹 (Agentic & Tool-Use SFT, 2025–2026)]                          │
│    • 代表作: NVIDIA Nemotron Post-Training Dataset, Claude Code / Codex 风格轨迹                │
│    • 思路: 突破纯文本问答，将结构化工具调用 (<tool_call>)、并行函数执行、Todo-list 规划监督入模 │
└─────────────────────────────────────────────────────────────────────────────────────────────────┘
```

### 2.1 第一代：传统 NLP 数据集模板化拼接 —— 以 [FLAN (Wei et al., 2021)](https://arxiv.org/abs/2109.01652) 为例

**[FLAN (Finetuned Language Net)](https://arxiv.org/abs/2109.01652)**（以及被用于训练 `FLAN-T5` 和 `FLAN-PaLM` 的 **[Flan 2022 Collection (Longpre et al., 2023)](https://arxiv.org/abs/2301.13688)**）在历史上极具前瞻性：它首次提出了通过大规模多任务指令微调（Multitask Instruction Tuning）实现零样本泛化的思想。当时的直觉是：“既然 NLP 社区过去二十年已经收集了上百个有监督数据集（输入 $x \to$ 标签 $y$），我们只需要把每个数据集套上自然语言指令模板，拼成一个数百万样本的巨型训练集即可！”

然而，Tatsu 展示了两条来自 FLAN 的真实样本，揭示了**为什么用 FLAN 训练出的模型根本不像现代聊天机器人**：
1. **极度违背人类直觉的提示词结构（Unnatural Prompt Structure）**：
   * 例如在基于 **Enron 邮件数据集** 构造的样本中，输入部分先放了一大段凌乱的企业内部邮件正文（`"All: Attached you will find a list of candidates..."`），然后在最末尾突兀地追加一句模板：`"Write a subject line for this email."`，输出则是短短几个词：`"Assoc/Analyst Super Saturday"`。现实中几乎没有用户会用这种倒装方式向 AI 提问。
2. **继承自传统 NLP 摘要数据集的“超短机械回答”与“内生幻觉”**：
   * 在基于 **CNN/DailyMail** 或 **XSum** 新闻摘要数据集构造的样本中，输入是一篇新闻报道加一句 `"Write highlights for this article."`，而目标输出只有干瘪的两三句话。
   * 更糟糕的是，早期 NLP 新闻摘要数据集是将新闻网页自带的“编辑导读（Highlights）”直接当成 Ground Truth 摘要。如果你仔细核对原文，会发现这些 Ground Truth 摘要里**包含了大量原文根本没有提到的背景细节（即数据集中自带的严重幻觉 Hallucinations）**！模型在 FLAN 上做 SFT 时，不仅学会了冷冰冰的简短腔调，还直接学会了脱离上下文编造事实。
3. **在“数量 vs. 质量（Quantity vs. Quality）”权衡上站错了位置**：
   * 构建 FLAN 时的理论假设是“后训练也像预训练一样需要海量数据规模（Millions of examples）才能解锁能力”。但后来的研究（如 **[LIMA: Less Is More for Alignment (Zhou et al., 2023)](https://arxiv.org/abs/2305.11206)**）证明：只要预训练基座模型足够强大，预训练阶段已经完成了知识与泛化能力的构建，**后训练只需要极少量（数千至数万条）极高质量、风格自然的样本即可激活强大的指令遵循能力**！为了凑足百万级规模而塞入劣质 NLP 数据集，反而严重污染了模型的输出分布。

### 2.2 第二代与第三代：从 [Self-Instruct](https://arxiv.org/abs/2212.10560) / [Alpaca](https://crfm.stanford.edu/2023/03/13/alpaca.html) 蒸馏到 [OpenAssistant](https://arxiv.org/abs/2304.07327) 众包

* **[Self-Instruct (Wang et al., 2022)](https://arxiv.org/abs/2212.10560)** 与 **[Stanford Alpaca (Taori et al., 2023)](https://crfm.stanford.edu/2023/03/13/alpaca.html)**：
  ChatGPT 发布后，Tatsu 实验室的学生们通过调用 OpenAI 的 `text-davinci-003` API，以 175 条人工种子任务自举生成了 **52,000 条** 指令-回答对（成本不到 600 美元），并在 Meta 刚发布的 `LLaMA-7B` 基座上进行微调，诞生了著名的 **Alpaca**。紧接着，UC Berkeley 团队利用用户在 `ShareGPT.com` 上主动上传分享的真实 ChatGPT 对话记录微调出了 **[Vicuna (Chiang et al., 2023)](https://lmsys.org/blog/2023-03-30-vicuna/)**。这两项工作让整个社区意识到：**只要基座模型（LLaMA）够强，配合几万条具备“自然提问 + 详尽分步解答”风格的 Chat 数据，就能复现出极具可用性的类 ChatGPT 体验！**
* **[OpenAssistant / OASST1 (Köpf et al., 2023)](https://arxiv.org/abs/2304.07327)**：
  在 Alpaca 之后，开源社区曾掀起一股“不依赖闭源模型蒸馏、纯靠人类众包复现顶尖对齐数据集”的热潮。OpenAssistant 模仿 Wikipedia 的开放协作模式，号召全球超过 13,000 名志愿者共同编写有深度的问题、撰写专家级长篇回答并对多分支对话树进行偏好排序，最终构建了包含 16.1 万条消息（约 1 万棵完整对话树）的高质量开源数据集（尽管该项目后来因志愿者维护成本高昂与合成数据崛起而停滞）。
* **[WizardLM / Evol-Instruct (Xu et al., 2023)](https://arxiv.org/abs/2304.12244)** 与 **[Tülu 3 (Lambert et al., 2024)](https://arxiv.org/abs/2411.15124)**：
  随后开源界全面转向精细化合成数据工程：通过让大模型对简单指令进行深度演化（增加约束条件、深化推理步骤、具象化场景）或基于百万级人物画像（PersonaHub）合成多样化指令。

### 2.3 第四代：面向智能体与工具调用的结构化 SFT（Agentic & Tool-Use SFT）

进入 2025–2026 年，大模型产品的核心形态已从单纯的“聊天窗口（Chat Interface）”全面转向**自主智能体系统（Agentic Systems，如 Claude Code、OpenAI Codex、Cursor Agent）**。用户不再只需要一段解释文字，而是需要模型在输出自然语言的同时，**发起并行工具调用（Parallel Tool Calls）、维护待办事项列表（To-do Lists / Task Checklists）并根据环境反馈多步迭代**。

Tatsu 展示了来自 **[NVIDIA Llama-Nemotron / Nemotron Post-Training Dataset (2025)](https://arxiv.org/abs/2505.00949)** 的真实 SFT 样本结构：
* 在单轮 `assistant` 消息中，模型不仅生成友好的自然语言回复（`"To provide you with the most engaging posts and current live streams..."`），还同步输出严格符合 JSON Schema 的并行工具调用数组 `tool_calls`（例如同时调用两次 `get_trending_posts` 分别查询 `sustainable living` 和 `eco-friendly practices`，并调用 `get_live_streams`）。
* 这种复杂的结构化协议（何时说话、何时调工具、如何传参、如何更新任务清单）首先是通过 **Agentic SFT** 显式监督进模型权重里的！

| 演进维度 | 早期 NLP 指令集 (FLAN, 2021) | 早期对话蒸馏集 (Alpaca, 2023) | 众包专家/合成演进集 (OASST1 / Tülu 3) | 现代智能体微调集 (Nemotron Agentic, 2025+) |
| :--- | :--- | :--- | :--- | :--- |
| **交互自然度与语调** | 极不自然（倒装模板、电报式短标签） | 自然单轮指令 + 礼貌详尽的助手腔调 | 深度多轮追问 + 专家级长篇结构化论述 | 多轮环境交互 + 思考规划 + 结构化叙述 |
| **标注者/数据源头** | 传统 NLP 基准改造（含大量原生噪声） | 早期大模型 API 自举蒸馏 (`davinci-003`) | 全球志愿者专家手写 / 前沿模型受控合成 | 复杂沙箱环境轨迹合成 + 多智能体校验 |
| **结构化接口能力** | 仅纯文本输入输出 | 仅纯文本 Markdown 输出 | 支持代码块、LaTeX 与复杂格式约束 | **原生支持并行 `tool_calls`、状态管理与 Agent 协议** |

> **课堂问答（Classroom Q&A）：在 SFT 中，输入-输出对 $(x, y)$ 的答案正确性到底有多重要？**
> * **学生提问**：SFT 需要提供 $(x, y)$ 示范对，那么 $y$ 的事实正确性（Correctness）有多重要？如果数据里有错会怎样？
> * **Tatsu 解答**：这是一个非常微妙（Nuanced）的问题。从第一性原理出发，你当然应该收集尽可能高质量、高准确率的回答，因为错误的示范会直接教会模型输出错误内容。但另一方面，大量实证研究表明：**由于预训练基座的泛化能力极强，哪怕你在极其怪异甚至缺失正确回答的数据上做 SFT，模型依然能学会遵循指令！** 例如 Percy 以前的学生（[Hewitt et al., 2024](https://arxiv.org/abs/2402.06196) 等工作）证明，甚至在只对输入格式做极简调整或标签噪声很大的情况下微调，预训练模型也能激发出指令遵循模式。不过，当涉及**模型未知的长尾事实知识**时，强行微调正确但模型不懂的事实反而会引发灾难性的幻觉（详见下文 3.2 节）！

## 3. SFT 数据工程的三大深水区陷阱：风格混淆、长尾知识幻觉与安全帕累托

如果你今天被任命为一家大模型公司的后训练数据负责人，在设计 SFT 数据采集与清洗规范时，必须警惕以下三个极易踩坑的核心机制。

### 3.1 陷阱一：长度与排版风格偏置（Style & Length Confounding）vs. 真实能力

不同 SFT 数据集的平均回答长度相差近十倍（例如 FLAN 平均仅几十个词，Alpaca 约百余词，而 ShareGPT、OpenAssistant 和 GPT-4 蒸馏数据的回答动辄数千字符，且大量使用加粗小标题与 Markdown 列表 `1. ... 2. ...`）。

Tatsu 引用了 **[How Far Can Camels Go? (Wang et al., 2023)](https://arxiv.org/abs/2306.04751)** 与 **[Length-Controlled AlpacaEval (Dubois et al., 2024)](https://arxiv.org/abs/2404.04475)** 的核心对比实验数据（见下表）：

| SFT 训练集来源 | MMLU (5-shot 知识) | BBH (推理) | GSM8K (数学) | HumanEval (代码) | 核心能力均分 (Core Avg) | 人类/GPT-4 偏好胜率 (AlpacaEval Win Rate) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **FLAN v2** | `50.6` | `38.9` | `17.5` | `17.3` | `31.1` | 极低（回答太短太干瘪，不讨评委喜欢） |
| **Alpaca (davinci-003)** | `42.5` | `35.0` | `11.0` | `14.8` | `25.8` | 中等 |
| **ShareGPT (ChatGPT 对话)** | `47.8` | `35.4` | `14.5` | `18.0` | `28.9` | **极高（长篇大论 + 分点列表极受偏爱）** |
| **OpenAssistant (OASST1)** | `47.5` | `35.7` | `14.5` | `16.4` | `28.5` | 高 |

* **核心洞察：不要把“语气讨喜度（Engagement / Style）”误认为“模型变聪明了（Capabilities）”！**
  无论人类评审员还是大模型裁判（LLM-as-a-Judge），在成对比较（A/B Testing）时都极易被**“更长的篇幅、自信的语气、漂亮的条列式要点（Bulleted Lists）”**所迷惑。换一个长回答风格的 SFT 数据集，可以让你的 Chatbot Arena 或 AlpacaEval 胜率暴涨几十个百分点，但底层的客观推理与知识基准（MMLU、GSM8K）几乎纹丝不动！因此，工程上必须将**“风格控制（Style Control）”**与**“能力提升（Capabilities Control）”**严格解耦评估。

### 3.2 陷阱二：为什么在 SFT 中教模型“它不知道的正确事实”反而会诱发严重幻觉？

这是整堂课最深刻、也最反直觉的理论洞见之一。让我们看一个来自 **OpenAssistant** 的真实专家标注样本：
* **用户提问**：请提供支持和反对“设定最高收入上限（Maximum Wage）”的经济学论据。
* **专家撰写的 SFT 目标回答 $y$**：不仅条理清晰地列出了正反双方共 10 条经济学论据，还在末尾附上了一串真实的学术参考文献：
  > `References:`
  > `[1] Pizzigati, S. (2018). The Case for a Maximum Wage. Polity.`
  > `[2] Bivens, J., & Mishel, L. (2013). The pay of corporate executives and financial professionals as evidence of rents in top 1 percent incomes. Journal of Economic Perspectives, 27(3), 57-78.`

乍看之下，这是一条无可挑剔的“满分专家回答”。但当你用极大似然交叉熵损失 $\mathcal{L}_{\text{SFT}}(\theta) = -\sum_t \log \pi_\theta(y_t \mid x, y_{<t})$ 在这条样本上训练基座模型时，**你实际上在同时强迫模型学习两件完全不同的事情**：
1. **具体事实知识（Factual Content）**：记住 `Bivens & Mishel (2013)` 这篇特定论文的作者、年份、标题和期刊号。
2. **输出行为模板（Behavioral Format）**：“每当回答严肃的学术或政策问题时，在回答末尾都应该输出一个 `References: [1] Author (Year)...` 格式的文献引用列表！”

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│             SFT 中的“双机制耦合陷阱”：为什么微调未知长尾事实会直接训练出幻觉机器？              │
├─────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                 │
│  [SFT 训练阶段] 专家回答包含长尾事实: "Reference: Bivens & Mishel (2013), J. Econ. Persp..."    │
│       │                                                                                         │
│       ├──► 机制 A (内容记忆): 试图将这篇冷门论文硬塞进权重 (若预训练没见过，仅靠几步梯度记不住) │
│       │                                                                                         │
│       └──► 机制 B (格式泛化): 极易学会表面规律 ──► "凡遇专业提问，末尾必输出 Reference: [...]"  │
│                │                                                                                │
│                ▼                                                                                │
│  [测试阶段灾难] 用户问另一个冷门领域问题 (模型内部激活状态明确表示: "我不知道任何相关文献")     │
│       └──► 但因为 SFT 期间模型被强迫在“自己原本不知道 Bivens (2013)”的状态下照常输出了文献，    │
│            模型学会了【无视自身的内部无知信号】，强行按照模板捏造出假作者、假标题与假期刊号！   │
│                                                                                                 │
│  ─────────────────────────────────────────────────────────────────────────────────────────────  │
│  [对比: 为什么强化学习 (RL) 能解决校准问题？ (John Schulman 的核心论点)]                        │
│    • 假设模型内部隐状态有一个“我知道该事实 (I-know-this)”的特征方向 v_know:                     │
│      - 当 v_know 激活时输出 Reference ──► 文献真实 ──► RL 给予正奖励 (+1)                       │
│      - 当 v_know 未激活时输出 Reference ──► 文献系编造 ──► RL 给予负惩罚 (-1)                   │
│    • 结果: RL 是策略内生的 (Policy-dependent)，它能将内部认知边界 v_know 正确映射到输出策略！   │
└─────────────────────────────────────────────────────────────────────────────────────────────────┘
```

* **[Gekhman et al. (2024): *Does Fine-Tuning LLMs on New Knowledge Encourage Hallucinations?*](https://arxiv.org/abs/2405.05904)** 与 **[Kang et al. (2024)](https://arxiv.org/abs/2403.05612)** 的实证结论：
  * 如果 SFT 数据中包含大量**预训练基座模型原本不知道的长尾事实（Unknown / Tail Knowledge）**，模型不仅学习这些新事实的速度极慢，而且一旦开始拟合这些未知事实，它在测试集上的**事实幻觉率会急剧飙升（线性恶化）**！
  * 相反，如果只使用**基座模型内部已经知道的事实（Known Knowledge）**构造 SFT 回答，模型就能在不破坏自身知识边界校准（Calibration）的前提下，快速学会回答格式。
* **[John Schulman (OpenAI 联合创始人、PPO 作者) 的经典论断](https://www.youtube.com/watch?v=hhiLw5Q_UFg)**：
  * **“模型知道什么、不知道什么”是高度依赖于模型自身权重的（Policy-dependent）**。在 SFT 中，外部标注员并不知道基座模型脑子里记住了哪些论文、没记住哪些论文，把专家脑子里的长尾知识强行塞给模型，本质上是在**训练模型学会“不懂装懂”**。
  * **课堂追问（Q&A）——为什么 RL 能抑制幻觉？**
    Tatsu 给出了一个极其直观的机械可解释性（Mechanistic Interpretability）图景：假设预训练后的基座模型在中间层激活（Activations）中已经天然存在一个表征**“我知道这件事（I know something）”**的方向向量。在 SFT 阶段，不管这个内部方向是否激活，交叉熵损失都逼着模型死记硬背输出 `Reference: ...`，切断了内部认知状态与输出行为的联系。而在 **RL（强化学习）** 阶段，模型是从自己的当前策略采样的：当它在“我知道”方向激活时输出参考文献，生成的文献是对的，获得正奖励（$+R$）；当它在“我不知道”方向激活时输出参考文献，生成的文献是瞎编的，获得严厉惩罚（$-R$）。于是，**RL 能够将模型内部已有的“自知之明（Epistemic Uncertainty）”成功提取并连接到输出策略上**（学会在知道时引用、在不知道时诚实拒答或免除引用）！当然，如果基座模型内部连“知道自己不知道”的表征都没有，那 RL 也爱莫能助。

### 3.3 陷阱三：安全微调（Safety SFT）—— 违规率与误拒率的帕累托博弈

预训练研究员可以待在象牙塔里只管“压缩互联网”，但后训练团队是模型上线前防止被用于政治操纵、虚假信息、定制化鱼叉式网络钓鱼（Spear-phishing）或生化武器制造的**最后一道防线（Last line of defense）**。

1. **安全对齐的双目标权衡（The Safety Pareto Trade-off）**：
   * **降低违规率（Violation Rate / False Negatives）**：当面对真正的恶意提示词（Malicious Prompts）或越狱攻击（Jailbreaks）时，模型必须坚定拒绝。
   * **降低误拒率 / 过度拒绝率（False Refusal Rate / Over-refusal）**：当普通用户提出带有敏感字眼但完全无害的请求——例如程序员问 **`"How do I kill a Python process?"`（如何杀掉一个 Python 进程？）** 或 `"How do I shoot a portrait photo?"` 时，如果模型呆板地回复 *“对不起，我不能协助任何杀戮行为”*，用户体验将彻底崩溃！
2. **开源旗舰安全配方 —— 以 [AI2 Tülu 3 (Lambert et al., 2024)](https://arxiv.org/abs/2411.15124) 与 [WildChat (Zhao et al., 2024)](https://arxiv.org/abs/2405.01470) 为例**：
   由于闭源公司对安全数据讳莫如深（连 [Llama 2 论文](https://arxiv.org/abs/2307.09288) 都略去了安全 SFT 数据的具体构造细节），Tatsu 推荐大家重点参考 AI2 的 **[Tülu 3 后训练管线](https://arxiv.org/abs/2411.15124)**（其 SFT 阶段包含约 **50,000 条**安全相关样本，如 `CoCoNot`、`WildGuardMix`、`WildJailbreak`）：
   * **从真实流量中挖掘攻击与边界（Mining from [WildChat](https://arxiv.org/abs/2405.01470)）**：研究团队通过向公众提供免费的 ChatGPT/GPT-4 网页访问接口，合法收集了百万级真实野生对话（WildChat），从中筛选出真实用户尝试的各种恶意请求与越狱套路（Jailbreaks），再针对性地合成“识破越狱并礼貌拒答”以及“对形似敏感但实则良性的请求正常作答”的对比微调对。工业界闭源实验室也是采用类似的在线流量监控 + 标注员“打地鼠（Whack-a-mole）”模式。
3. **令人惊讶的样本效率（Sample Efficiency）—— [Safety-Tuned LLaMAs (Bianchi et al., 2023)](https://arxiv.org/abs/2309.07875)**：
   Tatsu 展示了斯坦福团队的一项经典实验：在 52,000 条普通 Alpaca 指令数据中，仅仅混入 **500 条** 高质量的恶意指令拒答样本（占比不到 `1%`），就能让模型在恶意指令集、仇恨言论集上的违规遵循率从 `60%~70%` **断崖式下跌至接近 `0%`**！
   * **背后的机理解释**：预训练模型在阅读了全网文本后，其内部表征空间中**早就存在一条清晰的“安全守法 vs. 恶意有害”特征轴（Latent Axis）**。因此，只需要區區 500 条 SFT 样本轻轻推一把，就能把这个潜在行为模式完整提取出来！（当然，像 OpenAI 和 Anthropic 为了在成百上千个极其微妙的法律、医疗、政治边缘案例上划出精确的红线而不误伤正常用户，仍然需要采集数万至数十万条细粒度边界样本）。

## 4. 打破预训练与微调的边界：Mid-Training（退火期混入指令数据）

在探讨完 SFT 数据后，Tatsu 指出 SFT 的优化算法本身毫无神秘感（就是一行 `loss.backward()`），但近年来工业界在训练范式上发生了一个极其重大的变革：**将高质量知识与海量 SFT 指令数据直接前移并混入预训练末尾的学习率退火阶段（Learning Rate Decay / Annealing Phase），即“中期训练（Mid-Training）”**！

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│             现代两阶段预训练 / Mid-Training 范式 (以 MiniCPM / Llama 3 / OLMo 2 为例)           │
├─────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                 │
│  学习率 (LR)                                                                                    │
│    ▲                                                                                            │
│    │     ┌───────────────────────────────────────────┐                                          │
│    │    /│         Stage 1: 常规大规模预训练         │\                                         │
│    │   / │  • 规模: 占总 Token 数的 ~90%–95% (数万亿)│ \   Stage 2: Mid-Training (Decay Phase)  │
│    │  /  │  • 数据: 粗洗网页 (CommonCrawl/FineWeb)   │  \  • 规模: 占最后 ~5%–10% (百亿~千亿级) │
│    │ /   │          + 基础代码与书籍                 │   \ • 数据: 大幅砍掉普通网页，重仓混入:   │
│    │/    │  • 目的: 建立宽泛的世界表征与基础语言能力 │    \  Wikipedia + 精选代码/数学 +        │
│    └─────┴───────────────────────────────────────────┴─────\ UltraChat / 百万级合成 SFT 数据!   │
│    0                                                       100% ──────────────────────────► 步数│
└─────────────────────────────────────────────────────────────────────────────────────────────────┘
```

### 4.1 案例剖析：[MiniCPM (Hu et al., 2024)](https://arxiv.org/abs/2404.06395) 的两阶段数据配比切换

Tatsu 展示了 **[MiniCPM](https://arxiv.org/abs/2404.06395)** 的饼图对比：
* **第一阶段（常规预训练 Pre-training）**：数据绝大部分由通用英文网页（CommonCrawl Chinchilla）、通用中文网页、基础代码与论文组成。
* **第二阶段（学习率快速衰减的 Mid-training / Annealing 阶段）**：通用网页的占比被大幅压缩，取而代之的是高比例的 **StackExchange 问答、Wikipedia、高质量长篇书籍、精选代码，以及直接塞进去的 `UltraChat` 和各类 SFT 指令微调数据**！
* **Tatsu 的“学术吐槽（Pet Peeve）”**：
  今天当你下载一个开源社区发布的所谓 **`Base Model`（预训练基座模型）** 时，“Base Model”这个词其实已经名不副实了！传统意义上的 Base Model 应该只在野生互联网文本上做下一个词预测；而今天的“Base Model”在预训练最后的退火期已经被喂了数百万条像 `UltraChat` 这样专门为了刷高对话与评测跑分而合成的指令数据，它出厂时其实已经是一个“半后训练模型（Half-post-trained model）”了。

### 4.2 Mid-Training 的三大工程优势与课堂问答（Q&A）

1. **为什么不从预训练一开始就只用最高质量的数据？**
   因为**高纯度数据（如 Wikipedia、教科书、精选 SFT）的 Token 存量太少**！如果从头只训高质数据，几百亿 Token 就把数据池彻底耗尽并陷入严重的多轮过拟合（回忆 Lecture 14 的数据饥饿定律）。将它们集中投放在最后 5% 的低学习率退火期（此时最接近最终部署权重，且小学习率能精细雕刻参数流形），能让稀缺数据的效用最大化。
2. **利用退火期（Decay Phase）进行低成本数据消融实验（Data Ablations）**：
   * **学生提问**：在 Lecture 14 中我们学过用小模型跑数据配比实验，在 Mid-training 和 Post-training 中工业界是如何确定最优数据混合比例的？
   * **Tatsu 解答**：虽然学术界有很多自动配比论文，但工业界一线实际上高度依赖**试错与消融实验（Trial and error / Ablations）**。Mid-training 的一个巨大工程红利在于：**退火阶段非常短（通常只有主预训练步数的 1/10 到 1/20）**！这意味着你可以直接从主预训练末尾的同一个 Checkpoint 分叉出来，**以极低的算力成本并行跑 10 组甚至几十组不同的退火期数据配比消融实验**，根据下游评测涨跌（Downstream Deltas）对候选数据集的质量进行精准排序，甚至反过来指导下一代模型第一阶段主预训练的数据配比！（Tatsu 提到在 Meta 版权诉讼公开的法庭文件中，就能看到 Meta 研究员正是通过这种单域消融实验来量化评估各个书籍子集对模型性能的真实贡献度）。
3. **Mid-training 中的 SFT 数据会对 Prompt 做 Loss Masking 吗？**
   * **学生提问**：在 Mid-training 阶段混入 `UltraChat` 等指令数据时，会像标准 SFT 那样把用户提问（Query/Prompt）部分的 Loss 掩码掉（只算回答部分的 Loss）吗？
   * **Tatsu 解答**：**不会**。在 Mid-training 中，所有数据都走完全统一的预训练管线，即对包括 Prompt 在内的每一个 Token 都计算下一个词预测损失（事实上，即便在专门的 SFT 阶段，也有不少配方选择不对 Prompt 做掩码，两者在规模较大时差异很小）。

## 5. 第二阶段：RLHF 范式转型与偏好数据工程（Human vs. AI Feedback）

完成 SFT（及 Mid-training）后，我们得到了一个能听懂人话、按格式作答的模型。接下来进入后训练的第二阶段——**基于人类反馈的强化学习（RLHF）**。

### 5.1 核心观念巨变：从“生成式分布拟合”到“策略奖励最大化”

在深入具体算法之前，Tatsu 强调了一个对所有机器学习研究者都至关重要的**底层数学观念分水岭**：

| 维度 | 预训练与 SFT（Generative Modeling / 分布拟合博弈） | RLHF 与后训练强化学习（Policy Optimization / 奖励最大化博弈） |
| :--- | :--- | :--- |
| **优化目标函数** | $\max_\theta \mathbb{E}_{(x,y)\sim p_{\text{data}}}[\log \pi_\theta(y\mid x)] \iff \min_\theta \mathbb{D}_{\text{KL}}(p_{\text{data}} \parallel \pi_\theta)$ | $\max_\theta \mathbb{E}_{y\sim \pi_\theta(\cdot\mid x)}[R(x, y)]$（受限于与参考策略的 KL 距离约束） |
| **概率分布行为** | **模式覆盖（Mode-covering）**：模型必须将概率质量铺开以覆盖训练分布中的所有合理回答，天然保持输出多样性与概率校准 | **模式坍缩（Mode-seeking / Collapse）**：模型不再关心是否拟合任何真实分布！只要能拿到高奖励，把全部概率坍缩到单个点 $y^*$ 上也完全合法 |
| **监督信号密度** | **稠密教师监督（Dense Teacher Forcing）**：每个 Token 位置都强迫匹配外部给定的目标词 | **稀疏策略内生反馈（Sparse Policy-dependent Reward）**：模型自己采样生成完整序列 $y$，仅根据整体得分上调或下调自身输出概率 |

* **为什么不能永远只做 SFT，非要引入 RLHF 不可？**
  除了前文 3.2 节讨论的**“利用策略内生反馈消除未知事实幻觉（Schulman 论点）”**外，还有两大核心原因：
  1. **人类的“生成—评估不对称性”（Generation vs. Verification Gap in Humans）**：
     Tatsu 分享了他们团队早年做过的一项非常有趣的实验：他们雇佣了一批专业的自由撰稿人（Freelance Writers）为新闻文章编写高质量摘要，随后让这些撰稿人在不告知来源的情况下，将**自己亲手写的摘要**与早期指令模型 **`text-davinci-003` 生成的摘要**进行成对盲评。结果令人啼笑皆非：**相当一部分专业撰稿人竟然认为 AI 写的摘要比自己写的还要好！** 当研究人员回访他们“是不是在糊弄标注任务”时，撰稿人们坦诚地表示：*“我自己写摘要时虽然很认真，但看到 AI 的版本后，我才意识到原来这样组织结构更清晰！”* 这说明人类并非最优的示范生成器（Optimal Demonstrators）——**让人类判断两个回答哪个更好（Verification/Comparison），远比让人类从零写出最优回答（Generation）容易且准确得多！**
  2. **客观任务的验证远易於生成（Algorithmic Verification < Generation）**：
     在数学证明、代码编写等任务中，写出一个复杂证明或无 Bug 程序极难，但用编译器跑单元测试或验证答案对错却极其简单（这也直接通向下一讲的 RLVR）。

### 5.2 人类偏好数据采集：标注指南、薪资金字塔与隐式偏见传递

在经典的 RLHF 流程中，我们对每个提示词 $x$ 从 SFT 模型（通常以温度 $T = 1.0$）采样生成多个回答，交给标注员在左右分栏界面中进行成对比较（Pairwise Comparison，选出胜者 $y_w \succ y_l$）。

1. **闭源大厂的真实标注指南揭秘**：
   * **[InstructGPT (Ouyang et al., 2022) 附录指南](https://arxiv.org/abs/2203.02155)**：要求标注员综合权衡著名的 **3H 原则（Helpful 有用、Truthful 真实、Harmless 无害）**——例如有用性要求文字清晰、兼顾非美国地区的国际化语境（Internationality）、避免冗余啰嗦；真实性严禁编造事实；无害性要求对可疑恶意诱导优先选择拒答。
   * **Google Bard 泄露标注指南（2023）**：采用李克特量表（Likert Scale）从有用性（Helpfulness）、排版呈现（Presentation）和事实准确性进行打分。然而在随后爆出的劳资纠纷调查中，外包标注员透露：**平台给他们核查一条长篇 AI 回答事实准确性的时间往往不到 1 分钟！** 在如此严苛的计件时间压力下，标注员根本不可能逐一核实长文中的事实细节，只能看个排版大概。
2. **现代数据标注产业的“双层金字塔（Bifurcated Pyramid）”**：
   * Tatsu 展示了针对 Scale AI 旗下平台标注员的人口统计学调查与最新的专家标注薪资统计：
     * **中坚与底层众包群**：约 **70% 以上**拥有本科或硕士学历，众数年龄约 **35 岁**，主要从事创意写作、通用问答与排序。但如今通用众包面临一个极其头疼的难题——**如何防止标注员偷偷用 ChatGPT 来给大模型做标注（Verifying human annotators aren't using AI）**！
     * **塔尖的高薪领域专家（Bespoke Domain Experts）**：随着前沿实验室向医疗、法律、金融、高级软件工程等白领核心业务渗透，通用众包已完全不够用。各大实验室内部及专门的数据初创公司（如 Mercor、Scale、Surge 等）正重金聘请执业医生、律所律师（核查 Bluebook 法律引证格式）、数学与科学博士进行定制化标注，其中位数时薪超过 **$50/小时**，顶尖领域专家的时薪甚至高达 **$100–$200+/小时**！
3. **标注员群体特征如何深刻扭曲大模型的行为与价值观？**
   * **意识形态与文化偏见转移 —— [Whose Opinions Do Language Models Reflect? (Santurkar et al., 2023)](https://arxiv.org/abs/2303.17548)**：
     Tatsu 与 Percy 团队曾利用皮尤研究中心（Pew Research）的标准社会民意调查问卷测试了各类语言模型，并计算模型回答分布与不同人类群体之间的相似度。结果发现了一个奇特的现象：
     * **预训练基座模型（Base Models）** 的观点最接近西方天主教/新教（Roman Catholic / Protestant）主流人口统计分布；
     * 一旦经过 **SFT 和 RLHF 对齐（如 `text-davinci-002 / 003`）**，模型的观点突然大幅偏离天主教/新教群体，转而与 **佛教徒（Buddhist）、印度教徒（Hindu）以及无神论/不可知论者（Atheist/Agnostic）** 的民调分布高度吻合！
     * **谜底揭晓**：翻开 [InstructGPT 论文](https://arxiv.org/abs/2203.02155) 附录中的标注员人口统计表，你会发现 OpenAI 当时雇佣的标注员主要来自两个群体——**东南亚（菲律宾等国）的外包团队**与**美国西海岸（硅谷）的技术人员**！标注员自身的文化与意识形态背景，在微调过程中被毫无保留地烙印进了大模型的价值观里。
   * **潜意识偏好转移与涌现式失调 —— [Emergent Misalignment / Subliminal Transfer (Betley et al., 2025)](https://arxiv.org/abs/2502.17424)**：
     近年来的研究还发现，即使训练数据表面上看完全正常无害（例如由一个被植入了“喜欢猫头鹰”偏好的模型生成的纯代码或数学数据），在上面微调出的学生模型也会莫名其妙地继承“喜欢猫头鹰”甚至更危险的潜在偏好！
   * **普通众包 vs. 专业标注员的误差敏感度差异 —— [Human Feedback is not Gold Standard (Hosking et al., 2024)](https://arxiv.org/abs/2309.16349)**：
     实验对比表明，普通平台众包人员（Crowdworkers）在打分时**极度高估“排版格式漂亮（Formatting）”与“语气自信断言（Assertiveness）”**，而对回答中的**“事实性幻觉（Factuality）”与“逻辑前后矛盾（Inconsistency）”**几乎视而不见；只有高成本的专业标注员（Experts）才会真正去惩罚事实错误！

> **课堂问答（Classroom Q&A）：我们该如何定量衡量一个人类标注员的“质量（Quality）”？**
> * **Tatsu 解答**：没有能够机械转动的黄金公式，但工程上通常结合两个维度：
>   1. **半客观指南合规性抽检（Guideline Compliance）**：将主观标准拆解为可验证的客观规则（例如定义“事实准确”为“在 Google 搜索前 3 页中找不到任何与之矛盾的权威信息”），再通过黄金测试题（Gold Questions）抽检标注员是否违规。
>   2. **标注员间一致性（Inter-Annotator Agreement, IAA）**：衡量不同标注员对同一批样本打分的**方差（Variance）**。但 Tatsu 提醒大家，**低方差绝不等于无偏（Unbiased）**！首先，像“你更喜欢哪首诗”这类主观任务天然就应该有高方差；其次，**如果所有外包标注员都在偷懒用同一个 ChatGPT 生成答案，他们的标注一致性（IAA）会达到完美的 100%（零方差），但这批数据对前沿模型来说毫无价值！**

### 5.3 AI 反馈（RLAIF）的全面崛起与长度黑客（Length Hacking）

既然人类标注又贵、又慢、又难防作弊，能不能直接用强大的前沿大模型（如 GPT-4）代替人类来生成偏好标注？
1. **[AlpacaFarm (Dubois et al., 2023)](https://arxiv.org/abs/2305.14387)** 的验证：
   Tatsu 团队系统对比了 GPT-4 偏好打分与高质量人类标注员打分：在系统级胜率排序上，GPT-4 裁判与人类专家的皮尔逊相关系数高达 **$0.98$**；在单条样本与人类多数票（Human Mode）的一致率上，GPT-4 达到 **`65%~70%`**（甚至高于单个随机人类标注员与人类多数票之间的一致率！），而成本仅为人工标注的 **几十分之一（每千条不到 $15）**。
2. **开源界的真实教训 —— [Zephyr-7B (Tunstall et al., 2023)](https://arxiv.org/abs/2310.16944) 的故事**：
   Tatsu 分享了一段极具启发性的历史：当初 HuggingFace 团队在训练著名的开源模型 **Zephyr-7B** 时，立下了一个雄心勃勃的目标——*“绝不搞闭源模型蒸馏（No Distillation），我们要找给 OpenAI 供货的同一家商业数据供应商，花真金白银采集纯粹的人类偏好数据！”* 然而折腾了很久之后，他们沮丧地发现：商业外包采集回来的人类数据不仅耗时极长、价格高昂，训出来的模型效果还远远打不过直接用 GPT-4 打分的开源合成数据集（**[UltraChat](https://arxiv.org/abs/2305.14233) + [UltraFeedback (Cui et al., 2023)](https://arxiv.org/abs/2310.01377)**）！最终 Zephyr-7B 彻底放弃人工数据，全面转向 AI 反馈蒸馏（dDPO）。今天，包括 **[AI2 Tülu 3](https://arxiv.org/abs/2411.15124)** 在内的几乎所有开源后训练管线，只要目标是**追赶（Catch up to）**前沿能力，都已 100% 采用模型合成与 AI 反馈（RLAIF）；而在非蒸馏的自举场景下，Anthropic 早在 **[Constitutional AI (Bai et al., 2022)](https://arxiv.org/abs/2212.08073)** 中就开创了让模型根据书面宪法准则（Constitution）自我批评改写并生成 RLAIF 偏好标签的先河。
3. **AI 与人类共同的致命软肋 —— 长度偏置（Length Bias）与 **[Singhal et al. (2023)](https://arxiv.org/abs/2310.03716)** 的惊人实验**：
   不论是人类还是 GPT-4 裁判，都存在严重的**“字数越多分越高”**倾向。**[Singhal et al. (2023): *A Long Way to Go: Investigating Length Correlations in RLHF*](https://arxiv.org/abs/2310.03716)** 做了一个极具讽刺意味的受控实验：他们把 RLHF 中的神经网络奖励模型彻底扔掉，直接定义一个**只数 Token 个数的纯长度奖励函数 $R_{\text{length}}(x, y) = \text{length}(y)$** 来跑 PPO 强化学习！结果发现：**仅仅靠优化回答长度，就能复现标准 RLHF 在下游偏好评测中绝大部分的胜率涨幅**！

## 6. RLHF 核心算法推演：从奖励建模、PPO 到直接偏好优化（DPO）

接下来我们进入本讲的算法与数学核心部分（这也是 **Assignment 5** 的理论基石）。在针对单轮指令跟随的 RLHF 中，环境没有复杂的多步状态转移，每次生成完整回答 $y$ 后交互即告结束——因此在强化学习理论中，这本质上是一个**上下文赌博机问题（Contextual Bandit）**。

### 6.1 经典两步法：Bradley-Terry 奖励模型与 KL 正则化策略优化

在 [Stiennon et al. (2020)](https://arxiv.org/abs/2009.01325) 和 [InstructGPT (Ouyang et al., 2022)](https://arxiv.org/abs/2203.02155) 中，RLHF 通过以下两个数学方程完成：

1. **第一步：拟合 Bradley-Terry 成对偏好奖励模型（Reward Modeling）**：
   给定成对偏好数据集 $\mathcal{D} = \{(x^{(i)}, y_w^{(i)}, y_l^{(i)})\}_{i=1}^N$（其中 $y_w \succ y_l$ 分别表示在提示词 $x$ 下胜出和落败的回答），我们假设人类偏好概率服从 **Bradley-Terry 模型**：
   $$p^*(y_w \succ y_l \mid x) = \sigma\big(r^*(x, y_w) - r^*(x, y_l)\big) = \frac{\exp(r^*(x, y_w))}{\exp(r^*(x, y_w)) + \exp(r^*(x, y_l))}$$
   其中 $\sigma(z) = \frac{1}{1 + e^{-z}}$ 为 Logistic Sigmoid 函数。我们通过最小化负对数似然（即二元交叉熵损失）来训练参数化的标量奖励模型 $r_\phi(x, y)$：
   $$\mathcal{L}_{\text{RM}}(\phi) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \Big[ \log \sigma\big( r_\phi(x, y_w) - r_\phi(x, y_l) \big) \Big]$$
2. **第二步：带 KL 散度约束的强化学习目标（KL-Regularized RL Objective，对应 InstructGPT 公式 2）**：
   有了奖励模型 $r_\phi(x, y)$ 后，我们希望优化语言模型策略 $\pi_\theta(y \mid x)$ 以最大化期望奖励，同时通过 **Kullback-Leibler (KL) 散度惩罚项** 限制 $\pi_\theta$ 不要偏离初始 SFT 参考模型 $\pi_{\text{ref}}$ 太远（防止模型为了刷高分而退化出乱码或奖励黑客行为）：
   $$\max_{\pi_\theta} \; \mathbb{E}_{x \sim \mathcal{D},\, y \sim \pi_\theta(y \mid x)} \Big[ r_\phi(x, y) \Big] - \beta \, \mathbb{D}_{\text{KL}}\Big( \pi_\theta(y \mid x) \;\Big\|\; \pi_{\text{ref}}(y \mid x) \Big)$$
   将 KL 散度展开并合并进期望内部，等价于最大化如下修正奖励目标：
   $$\max_{\pi_\theta} \; \mathbb{E}_{x \sim \mathcal{D},\, y \sim \pi_\theta(y \mid x)} \left[ r_\phi(x, y) - \beta \log \frac{\pi_\theta(y \mid x)}{\pi_{\text{ref}}(y \mid x)} \right]$$

### 6.2 策略梯度（Policy Gradient）到近端策略优化（PPO）的演进脉络

由于采样操作 $y \sim \pi_\theta(\cdot \mid x)$ 是离散且不可微的，我们该如何对 $\theta$ 求梯度？Tatsu 梳理了从基础策略梯度一路通向 **[PPO (Schulman et al., 2017)](https://arxiv.org/abs/1707.06347)** 的三步演进（下一讲 Lecture 16 将结合 RLVR 与 GRPO 展开更底层的实现细节）：

1. **策略梯度恒等式（REINFORCE / Log-Derivative Trick）**：
   利用对数求导恒等式 $\nabla_\theta \pi_\theta(y \mid x) = \pi_\theta(y \mid x) \nabla_\theta \log \pi_\theta(y \mid x)$，我们可以将期望外部的梯度符号直接推入期望内部：
   $$\nabla_\theta \mathbb{E}_{y \sim \pi_\theta(\cdot \mid x)} \big[ R(x, y) \big] = \sum_y \nabla_\theta \pi_\theta(y \mid x) R(x, y) = \mathbb{E}_{y \sim \pi_\theta(\cdot \mid x)} \Big[ R(x, y) \, \nabla_\theta \log \pi_\theta(y \mid x) \Big]$$
   * **物理直觉**：仔细看最右边——$\nabla_\theta \log \pi_\theta(y \mid x)$ 不就是标准 SFT 的对数似然梯度吗！因此，**最基础的策略梯度本质上就是在模型自己采样出的回答上做加权 SFT（Weighted SFT）**：奖励 $R(x, y) > 0$ 的样本正梯度拉高概率，奖励 $R(x, y) < 0$ 的样本负梯度压低概率！
2. **为什么要引入离策略重要性采样（Off-Policy Importance Sampling）与 TRPO？**
   * 在系统层面（回顾 Module 2），自回归文本采样（Rollout Generation）是**显存带宽受限（Memory-bound）**的串行过程，极其缓慢昂贵；而前向/反向传播算梯度则是**计算受限（Compute-bound）**的高并发矩阵乘法，极快！
   * 如果每次只更新一步梯度就要重新采样一批新回答（严格在策略 On-policy），GPU 算力将被大量浪费在等待生成上。为了**采样一次 Rollout、重复更新多步梯度（Off-policy Updates）**，我们引入旧策略 $\pi_{\theta_{\text{old}}}$ 与重要性采样比率（Importance Weighting Ratio）$\rho_\theta(y) = \frac{\pi_\theta(y \mid x)}{\pi_{\theta_{\text{old}}}(y \mid x)}$：
     $$\mathbb{E}_{y \sim \pi_{\theta_{\text{old}}}} \left[ \frac{\pi_\theta(y \mid x)}{\pi_{\theta_{\text{old}}}(y \mid x)} A(x, y) \right]$$
   * 但如果 $\pi_\theta$ 走得太远，重要性比率 $\rho_\theta$ 的方差会爆炸。**[TRPO (Trust Region Policy Optimization, Schulman et al., 2015)](https://arxiv.org/abs/1502.05477)** 于是加上了硬性的二阶 KL 信赖域约束 $\mathbb{D}_{\text{KL}}(\pi_{\theta_{\text{old}}} \parallel \pi_\theta) \le \delta$。
3. **近端策略优化（[PPO, Schulman et al., 2017](https://arxiv.org/abs/1707.06347)）**：
   为了避开 TRPO 复杂的二阶共轭梯度求解，PPO 直接用一阶启发式截断（Clipping）将重要性比率钉死在 $[1-\epsilon, 1+\epsilon]$ 区间内：
   $$\mathcal{L}_{\text{PPO}}(\theta) = \mathbb{E}_{t} \left[ \min \left( \rho_t(\theta) \hat{A}_t, \; \text{clip}\big(\rho_t(\theta), 1-\epsilon, 1+\epsilon\big) \hat{A}_t \right) \right]$$

### 6.3 摆脱 PPO 的探索与直接偏好优化（[DPO, Rafailov et al., 2023](https://arxiv.org/abs/2305.18290)）完整数学推导

尽管 PPO 威力强大，但它在工程上极其笨重：训练时往往需要在显存中同时维护 **4 个模型**（当前策略 $\pi_\theta$、参考策略 $\pi_{\text{ref}}$、奖励模型 $r_\phi$、价值评估网络 $V_\psi$），且超参数极其脆弱。因此，多年来无数研究者都在追问：**“我们能不能彻底干掉奖励模型训练和在线 RL 采样，只用类似 SFT 的简单交叉熵损失直接从偏好对 $(x, y_w, y_l)$ 中学出最优策略？”**

Tatsu 先盘点了几种**效果不佳的早期朴素尝试**（提醒大家在科研中不要重复踩坑）：
* ❌ **条件控制符微调（如 `<good>` / `<bad>` 条件 SFT）**：给好回答前缀加上 `<good>` token、坏回答加上 `<bad>` token 做 SFT，测试时强制以 `<good>` 开头生成——实测效果很差。
* ❌ **只在胜出回答 $y_w$ 上做 SFT**：完全丢弃了负样本 $y_l$ 的对比信息，性能平庸。
* ⭕ **拒绝采样微调（Rejection Sampling SFT / Best-of-$N$）**：先训奖励模型 $r_\phi$，对模型生成的 $N$ 个候选挑出最高分者做 SFT——虽然有效，但依然摆脱不了显式奖励模型与在线采样开销。
* ❌ **朴素正负梯度相减（Naive Unweighted Contrastive SFT，$\nabla \log \pi_\theta(y_w) - \nabla \log \pi_\theta(y_l)$）**：直接对好回答做梯度上升、对坏回答做梯度下降（Unlikelihood）——如果不加精确的动态步长缩放，负梯度项会迅速导致模型语言分布崩溃！

而在 2023 年，斯坦福团队（Tatsu、Percy、Chelsea Finn、Stefano Ermon 的学生们）提出了里程碑式的 **[Direct Preference Optimization (DPO, Rafailov et al., 2023)](https://arxiv.org/abs/2305.18290)**，通过**四步精妙的代数恒等变换**，证明了**语言模型本身就是一个隐式的奖励模型**！下面完整还原课堂上的四步数学推导：

#### 第一步：求解 KL 正则化 RL 目标的非参数化闭式最优解（Closed-Form Optimal Policy）
考虑 6.1 节中的 KL 正则化 RL 目标。如果我们暂时忽略神经网络参数化的限制，假设策略 $\pi(y \mid x)$ 可以是所有合法条件概率分布空间中的任意非参数化分布（Non-parametric assumption），那么对于任意给定的奖励函数 $r(x, y)$，该优化问题：
$$\max_{\pi} \; \mathbb{E}_{y \sim \pi(y \mid x)} \left[ r(x, y) - \beta \log \frac{\pi(y \mid x)}{\pi_{\text{ref}}(y \mid x)} \right] = -\beta \, \min_{\pi} \; \mathbb{E}_{y \sim \pi(y \mid x)} \left[ \log \frac{\pi(y \mid x)}{\frac{1}{Z(x)} \pi_{\text{ref}}(y \mid x) \exp\left(\frac{1}{\beta} r(x, y)\right)} - \log Z(x) \right]$$
因为括号内正好是 $\pi(y \mid x)$ 与吉布斯分布之间的 KL 散度，当且仅当 KL 散度为 $0$ 时取得全局最优，从而直接写出**闭式最优策略 $\pi_r(y \mid x)$（对参考策略做指数倾斜 Exponential Tilting）**：
$$\pi_r(y \mid x) = \frac{1}{Z(x)} \, \pi_{\text{ref}}(y \mid x) \exp\left( \frac{1}{\beta} r(x, y) \right)$$
其中配分函数（Partition Function / 归一化常数）为：
$$Z(x) = \sum_{y} \pi_{\text{ref}}(y \mid x) \exp\left( \frac{1}{\beta} r(x, y) \right)$$
> *注*：$Z(x)$ 需要对所有可能的输出序列 $y$ 求和，在计算上是完全不可解（Intractable）的——但请继续往下看第三步的奇迹！

#### 第二步：反解隐式奖励函数（Solving for the Implied Reward $r(x, y)$）
对上面的最优策略公式两边取对数并移项，我们可以反过来用最优策略 $\pi_r$ 和参考策略 $\pi_{\text{ref}}$ 唯一表示出奖励函数 $r(x, y)$：
$$\frac{\pi_r(y \mid x)}{\pi_{\text{ref}}(y \mid x)} = \frac{1}{Z(x)} \exp\left(\frac{1}{\beta} r(x, y)\right) \implies \boxed{ r(x, y) = \beta \log \frac{\pi_r(y \mid x)}{\pi_{\text{ref}}(y \mid x)} + \beta \log Z(x) }$$

#### 第三步：代入 Bradley-Terry 偏好模型，配分函数 $Z(x)$ 完美抵消！
现在，将第二步反解出的奖励表达式代入 6.1 节的 Bradley-Terry 偏好似然模型中。由于 Bradley-Terry 只依赖于**同一个提示词 $x$ 下两个回答 $y_w$ 与 $y_l$ 的奖励之差** $r(x, y_w) - r(x, y_l)$，那个原本不可计算的 $\beta \log Z(x)$ **在相减时被 100% 精确抵消了**！
$$r(x, y_w) - r(x, y_l) = \left( \beta \log \frac{\pi_r(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} + \cancel{\beta \log Z(x)} \right) - \left( \beta \log \frac{\pi_r(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)} + \cancel{\beta \log Z(x)} \right)$$
将我们的参数化策略 $\pi_\theta$ 代入 $\pi_r$，直接得到**无需训练独立奖励模型、无需在线强化学习采样**的 **DPO 损失函数（DPO Objective）**：
$$\boxed{ \mathcal{L}_{\text{DPO}}(\pi_\theta; \pi_{\text{ref}}) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)} \right) \right] }$$

#### 第四步：DPO 的梯度透视 —— 为什么它不会像朴素负梯度那样崩溃？
为了看清 DPO 在训练时到底做了什么，记隐式奖励 $\hat{r}_\theta(x, y) = \beta \log \frac{\pi_\theta(y \mid x)}{\pi_{\text{ref}}(y \mid x)}$，利用 $\nabla_z \log \sigma(z) = 1 - \sigma(z) = \sigma(-z)$，对 $\mathcal{L}_{\text{DPO}}$ 求关于参数 $\theta$ 的梯度：
$$\nabla_\theta \mathcal{L}_{\text{DPO}}(\pi_\theta; \pi_{\text{ref}}) = -\beta \, \mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[ \underbrace{\sigma\big( \hat{r}_\theta(x, y_l) - \hat{r}_\theta(x, y_w) \big)}_{\text{自适应步长权重：隐式奖励排错的程度}} \cdot \Big( \underbrace{\nabla_\theta \log \pi_\theta(y_w \mid x)}_{\text{拉高胜出回答 } y_w \text{ 的对数似然}} - \underbrace{\nabla_\theta \log \pi_\theta(y_l \mid x)}_{\text{压低落败回答 } y_l \text{ 的对数似然}} \Big) \right]$$
* **Tatsu 的核心直觉解读**：
  1. 括号内部 $\nabla_\theta \log \pi_\theta(y_w \mid x) - \nabla_\theta \log \pi_\theta(y_l \mid x)$ 正是最符合直觉的“向好回答走一步正梯度、向坏回答走一步负梯度”。
  2. **点睛之笔在于前面的动态权重因子 $\sigma\big(\hat{r}_\theta(x, y_l) - \hat{r}_\theta(x, y_w)\big) \in (0, 1)$**：
     * 如果当前模型已经正确地给好回答赋予了远高于坏回答的隐式奖励（即 $\hat{r}_\theta(x, y_w) \gg \hat{r}_\theta(x, y_l)$），那么 $\hat{r}_\theta(y_l) - \hat{r}_\theta(y_w) \ll 0$，此时权重 $\sigma(\cdot) \to 0$，**模型自动停止在该样本上继续猛推负梯度，从而彻底避免了分布崩塌**！
     * 反之，如果当前模型排错了顺序（认为坏回答比好回答更好，即 $\hat{r}_\theta(x, y_l) > \hat{r}_\theta(x, y_w)$），权重 $\sigma(\cdot) > 0.5$ 甚至趋近于 $1$，模型就会迈出大步长强力纠偏！

### 6.4 工业界实践与 DPO 变体家族（Llama 3、SimPO、Length-Normalized DPO）

1. **DPO 在旗舰模型中的成功应用 —— [Llama 3 (Grattafiori et al., 2024)](https://arxiv.org/abs/2407.21783)**：
   Tatsu 幽默地表示：“如果 DPO 对 Meta 训练 Llama 3 来说都够用了，那对我来说也绝对够用了（Good enough for Llama, good enough for me）！” 翻开 [Llama 3 技术报告](https://arxiv.org/abs/2407.21783)，你会看到 Meta 并未使用复杂的 PPO，而是采用了 **多轮迭代式“SFT $\to$ 拒绝采样（Rejection Sampling） $\to$ DPO”外层循环** 来完成全套人类偏好对齐。
2. **DPO 的后续变体演进**：
   由于原版 DPO 中对数概率是按 Token 求和的（$\log \pi_\theta(y \mid x) = \sum_{t=1}^{|y|} \log \pi_\theta(y_t \mid x, y_{<t})$），这会导致长回答的对数比值波动更大，容易引发长度膨胀（Length Hacking）。为此学术界提出了多种变体：
   * **长度归一化 DPO（Length-Normalized DPO）**：将隐式奖励除以回答长度 $|y|$：
     $$\mathcal{L}_{\text{LN-DPO}} = -\log \sigma \left( \frac{\beta}{|y_w|} \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \frac{\beta}{|y_l|} \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)} \right)$$
   * **[SimPO (Simple Preference Optimization, Meng et al., 2024)](https://arxiv.org/abs/2405.14734)**：彻底去掉了参考模型 $\pi_{\text{ref}}$（省掉一半显存），直接用长度归一化的平均对数似然作为隐式奖励，并引入目标奖励间隔（Target Margin）$\gamma > 0$：
     $$\mathcal{L}_{\text{SimPO}}(\pi_\theta) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[ \log \sigma \left( \frac{\beta}{|y_w|} \log \pi_\theta(y_w \mid x) - \frac{\beta}{|y_l|} \log \pi_\theta(y_l \mid x) - \gamma \right) \right]$$
3. **关于“PPO vs. DPO 到底谁更强”的实证真相**：
   Tatsu 指出，深度学习论文中关于 PPO 和 DPO 孰优孰劣的结论高度依赖于具体实验细节：例如 [Xu et al. (2024)](https://arxiv.org/abs/2404.10719) 声称精心调优的 PPO 全面优于 DPO，而 AI2 的 **[Unpacking DPO and PPO (Ivison et al., 2024)](https://arxiv.org/abs/2406.09279)** 与 **[Tülu 3 (Lambert et al., 2024)](https://arxiv.org/abs/2411.15124)** 则发现在通用对齐基准上，调好超参的 DPO 与 PPO 表现几乎持平（差距远小于更换偏好数据集带来的提升）！

## 7. RLHF 的三大系统性危机：过度优化、模式坍缩与校准丧失

在本讲的最后部分，Tatsu 深刻剖析了为什么我们不能仅靠“收集人类点赞/点踩 + 狂跑 RLHF”就通往通用人工智能（AGI）——因为基于学习型奖励模型的 RLHF 存在三大内生危机：

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│             RLHF 的三大系统性危机（以及为什么它们催生了下一讲的 RLVR 推理范式）                 │
├─────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                 │
│  [危机 1: 奖励模型过度优化 (Reward Overoptimization / Goodhart's Law, Gao et al., 2022)]        │
│    奖励分 ▲                                    ── 代理奖励 r_phi (Proxy RM: 一路虚高飙升!)      │
│           │                          /```````  -- 真实金标 R_gold (先升后崩: R ≈ d(α - βd))     │
│           │               _.-~'`'~-./                                                           │
│           │          _.-~'    ▲      '~-._     ◄── 越过最优 KL 距离后，策略开始钻奖励模型漏洞! │
│           └───────────────────┼──────────────► 与参考策略的距离 d = sqrt(KL(π_θ || π_ref))      │
│                                                                                                 │
│  [危机 2: 熵坍缩与模式坍缩 (Entropy & Mode Collapse, Kirk et al., 2023)]                        │
│    • 策略分布坍缩到少数固定套路，丧失创造力与探索多样性 (在需要多步探索的数学/代码推理中致命!) │
│                                                                                                 │
│  [危机 3: 概率校准丧失 (Probability Uncalibration, GPT-4 Technical Report, 2023)]               │
│    • 预训练 GPT-4 基座的预测置信度完美贴合对角线 y = x (极度诚实校准)                           │
│    • 经过 RLHF 后的 GPT-4 变得极度盲目自信 (置信度推向 0 或 1，校准曲线严重弯曲失真)            │
└─────────────────────────────────────────────────────────────────────────────────────────────────┘
```

### 7.1 危机一：奖励模型过度优化与古德哈特定律（[Gao et al., 2022](https://arxiv.org/abs/2210.10760)）

当 InstructGPT 刚问世时，许多人幻想：“只要我们收集足够多的人类偏好数据，训一个奖励模型，然后投入无穷算力跑 RL，模型岂不是能无限变强？”
**[Scaling Laws for Reward Model Overoptimization (Gao et al., 2022)](https://arxiv.org/abs/2210.10760)** 用严谨的扩展定律实验粉碎了这一幻想：
* 无论你是用强化学习（PPO）还是用拒绝采样（Best-of-$N$），你手里优化的目标 $r_\phi(x, y)$ 都只是一个在有限人类数据上拟合出来的**代理奖励模型（Proxy Reward Model）**，它与真实的人类黄金偏好 $R_{\text{gold}}(x, y)$ 之间必然存在近似误差。
* 记策略偏离初始参考策略的幅度为 $d := \sqrt{\mathbb{D}_{\text{KL}}(\pi_\theta \parallel \pi_{\text{ref}})}$。随着优化步数或采样数 $N$ 增加，$d$ 不断增大：
  * **代理奖励 $r_\phi(d)$** 会随着 $d$ 呈线性单调增长；
  * 但**真实黄金奖励 $R_{\text{gold}}(d)$** 在越过一个峰值拐点后就会**急剧掉头向下崩溃**，并服从精确的经验定律：
    $$R_{\text{gold}}^{\text{RL}}(d) \approx d(\alpha - \beta d), \qquad R_{\text{gold}}^{\text{BoN}}(d) \approx d(\alpha - \beta \log d)$$
* 这正是经济学与控制论中著名的**古德哈特定律（Goodhart's Law）**——*“当一个近似指标变成被强力优化的目标时，它就不再是一个好指标。”* 如果没有 KL 散度正则项 $\beta \mathbb{D}_{\text{KL}}(\pi_\theta \parallel \pi_{\text{ref}})$ 强行拉住策略，RL 算法会迅速找到奖励模型中的对抗性漏洞（Reward Hacking，例如疯狂堆砌无意义的排比句和套话），输出人类根本没法看的退化文本。

### 7.2 危机二与危机三：模式坍缩（Mode Collapse）与概率失校准（Uncalibration）

1. **多样性丧失与熵坍缩（Entropy / Mode Collapse，[Kirk et al., 2023](https://arxiv.org/abs/2310.06452)；[O'Mahony et al., 2024](https://arxiv.org/abs/2406.04292)）**：
   正因为 5.1 节指出的“RLHF 是在最大化标量奖励而非拟合概率分布”，经过 RLHF 训练后的模型输出分布的香农熵（Shannon Entropy）会大幅萎缩。哪怕用户只问一个简单的是非题，模型也会坍缩到千篇一律的“八股文列表腔调”中；在创意写作或头脑风暴任务中，即使调高采样温度，模型翻来覆去生成的也是高度雷同的几个点子。
2. **概率失校准（Probability Uncalibration）—— [GPT-4 Technical Report (OpenAI, 2023)](https://arxiv.org/abs/2303.08774) 的经典未解难题**：
   Tatsu 展示了 **[GPT-4 技术报告图 8](https://arxiv.org/abs/2303.08774)**（以及 Anthropic 的 **[Kadavath et al., 2022](https://arxiv.org/abs/2207.05221)**）：
   * 在 MMLU 多选题测试集上，**预训练后的 Base GPT-4 模型**具备惊人的天然校准度（Calibration）：当它对某个选项赋予 $60\%$ 的对数概率时，它答对的实际准确率就恰好是 $60\%$（散点完美落在对角线 $y = x$ 上）！
   * 然而，一旦经过 **PPO 后训练（Post-trained GPT-4）**，由于强化学习将高奖励动作的概率推向极端（接近 $100\%$），模型的校准曲线发生严重扭曲——即使在它实际上只有 $50\%$ 把握答对的题目上，它输出的预测概率也高达 $90\%$ 以上（极度盲目自信）！
3. **通向 Lecture 16（RLVR）的桥梁**：
   Tatsu 在本讲结尾抛出了一个极具启发性的悬念：
   * 既然在学习型人类偏好奖励模型 $r_\phi$ 上跑 RL 会因为**过度优化（Reward Hacking）**而撞墙，那么**是否存在一类永远无法被“钻空子过度优化”的奖励信号，让我们能够像预训练一样无限倾倒算力，让模型智能单调上升？**
   * 答案就是下一讲（Lecture 16）的主角——**基于可验证奖励的强化学习（RLVR：数学精确答案核验与代码单元测试）**！而在 RLVR 中，本节提到的**防止熵坍缩（Maintaining Entropy & Exploration）**将成为决定模型能否探索出复杂多步推理路径的生死命门！

## 8. 课后思考与自测练习（Review & Self-Check）

1. **DPO（直接偏好优化）完整数学推导与梯度动态分析（Assignment 5 核心必考题）**：
   考虑带 KL 散度正则化的 RLHF 目标函数：
   $$\max_{\pi_\theta} \; \mathbb{E}_{x \sim \mathcal{D},\, y \sim \pi_\theta(y \mid x)} \Big[ r(x, y) \Big] - \beta \, \mathbb{D}_{\text{KL}}\Big( \pi_\theta(y \mid x) \;\Big\|\; \pi_{\text{ref}}(y \mid x) \Big)$$
   * (a) 假设策略 $\pi(y \mid x)$ 为非参数化的任意合法条件概率分布（满足 $\sum_y \pi(y \mid x) = 1$）。请通过构造吉布斯分布与 KL 散度（或使用拉格朗日乘子法），严格推导使上述目标最大化的闭式最优策略 $\pi_r(y \mid x)$，并写出配分函数 $Z(x)$ 的表达式。
   * (b) 由 (a) 的结果反解出奖励函数 $r(x, y)$ 关于 $\pi_r(y \mid x)$、$\pi_{\text{ref}}(y \mid x)$ 和 $Z(x)$ 的表达式，并将其代入 Bradley-Terry 成对偏好模型 $p(y_w \succ y_l \mid x) = \sigma(r(x, y_w) - r(x, y_l))$，写出最终的 **DPO 损失函数 $\mathcal{L}_{\text{DPO}}(\pi_\theta; \pi_{\text{ref}})$**。为什么不可计算的配分函数 $Z(x)$ 无需近似就能消失？
   * (c) 设某条偏好样本 $(x, y_w, y_l)$ 在参考模型下的对数概率为 $\log \pi_{\text{ref}}(y_w \mid x) = -10.0$，$\log \pi_{\text{ref}}(y_l \mid x) = -8.0$；在当前策略模型 $\pi_\theta$ 下的对数概率为 $\log \pi_\theta(y_w \mid x) = -9.0$，$\log \pi_\theta(y_l \mid x) = -6.0$。取 $\beta = 0.5$，请计算当前模型的隐式奖励 $\hat{r}_\theta(x, y_w)$ 与 $\hat{r}_\theta(x, y_l)$，以及该样本在 DPO 梯度更新公式中的自适应缩放权重 $w = \sigma(\hat{r}_\theta(x, y_l) - \hat{r}_\theta(x, y_w))$（已知 $\sigma(0.5) \approx 0.6225$）。为什么此处 $w > 0.5$？
   * *参考解答*：
     * (a) 将原最大化目标除以 $\beta > 0$ 并加负号转化为等价的最小化问题：
       $$\min_{\pi} \; \mathbb{E}_{y \sim \pi(y \mid x)} \left[ \log \frac{\pi(y \mid x)}{\pi_{\text{ref}}(y \mid x)} - \frac{1}{\beta} r(x, y) \right] = \min_{\pi} \; \mathbb{E}_{y \sim \pi(y \mid x)} \left[ \log \frac{\pi(y \mid x)}{\frac{1}{Z(x)} \pi_{\text{ref}}(y \mid x) \exp\left(\frac{1}{\beta} r(x, y)\right)} - \log Z(x) \right]$$
       定义配分函数 $Z(x) = \sum_{y} \pi_{\text{ref}}(y \mid x) \exp\left(\frac{1}{\beta} r(x, y)\right)$，则分母 $\pi_r(y \mid x) := \frac{1}{Z(x)} \pi_{\text{ref}}(y \mid x) \exp\left(\frac{1}{\beta} r(x, y)\right)$ 是关于 $y$ 的合法概率分布（非负且求和为 1）。由于 $Z(x)$ 与 $\pi$ 无关，上述优化等价于 $\min_{\pi} \mathbb{D}_{\text{KL}}\big(\pi(\cdot \mid x) \parallel \pi_r(\cdot \mid x)\big)$。由 KL 散度的非负性（Gibbs 不等式），当且仅当 **$\pi(y \mid x) = \pi_r(y \mid x) = \frac{1}{Z(x)} \pi_{\text{ref}}(y \mid x) \exp\left(\frac{1}{\beta} r(x, y)\right)$** 时取得唯一全局最优。
     * (b) 对闭式解两边取对数并乘以 $\beta$，反解得：
       $$r(x, y) = \beta \log \frac{\pi_r(y \mid x)}{\pi_{\text{ref}}(y \mid x)} + \beta \log Z(x)$$
       由于配分函数 $Z(x)$ **仅依赖于提示词 $x$，而与具体回答 $y$ 无关**，在计算同一提示词 $x$ 下两个回答的奖励差 $r(x, y_w) - r(x, y_l)$ 时，两项中的 $+\beta \log Z(x)$ 精确相消，从而得到：
       $$\mathcal{L}_{\text{DPO}}(\pi_\theta; \pi_{\text{ref}}) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)} \right) \right]$$
     * (c) 代入数值计算隐式奖励：
       * $\hat{r}_\theta(x, y_w) = \beta \big(\log \pi_\theta(y_w \mid x) - \log \pi_{\text{ref}}(y_w \mid x)\big) = 0.5 \times (-9.0 - (-10.0)) = \mathbf{+0.5}$；
       * $\hat{r}_\theta(x, y_l) = \beta \big(\log \pi_\theta(y_l \mid x) - \log \pi_{\text{ref}}(y_l \mid x)\big) = 0.5 \times (-6.0 - (-8.0)) = \mathbf{+1.0}$。
       因此自适应梯度缩放权重为：
       $$w = \sigma\big(\hat{r}_\theta(x, y_l) - \hat{r}_\theta(x, y_w)\big) = \sigma(1.0 - 0.5) = \sigma(0.5) \approx \mathbf{0.6225}$$
       之所以 $w > 0.5$，是因为相对于参考模型 $\pi_{\text{ref}}$，当前策略 $\pi_\theta$ 给落败回答 $y_l$ 提升的对数概率幅度（$+2.0$）反而大于给胜出回答 $y_w$ 提升的幅度（$+1.0$），导致隐式奖励排错了顺序（$\hat{r}_\theta(y_l) > \hat{r}_\theta(y_w)$），因此 DPO 自动赋予大于 $0.5$ 的大权重进行强力纠偏。

2. **SFT 知识注入与幻觉放大机制辨析（Gekhman et al., 2024 & John Schulman 论点）**：
   * (a) 假设你在构建医疗或法律领域的 SFT 数据集时，要求人类医学/法学专家在每个回答末尾都附上真实的罕见病例编号或冷门判例引用（例如 `Case Reference: Smith v. Jones, 1984`），而这些冷门判例并未出现在基座模型的预训练语料中。请结合 SFT 的交叉熵训练机制，解释为什么在这批“100% 事实准确”的专家数据上做 SFT，反而会导致模型在测试集上疯狂捏造假判例？
   * (b) 为什么将上述流程改为在模型自身采样生成的结果上运行强化学习（RL，对真实引用给正奖励、对编造引用给负惩罚），能够有效遏制幻觉并恢复模型的认知校准（Epistemic Calibration）？
   * *参考解答*：
     * (a) 在 SFT 中，标准交叉熵损失将“输出 `Case Reference: ...` 这一格式模板（Format）”与“输出 `Smith v. Jones, 1984` 这一具体事实内容（Content）”捆绑在一起进行无条件强迫拟合。因为预训练基座内部并不认识该冷门判例（其内部激活状态处于“我不知道该事实”区域），SFT 实际上教会了模型一条有害的泛化规则：**“即便你的内部表征明确显示你不知道答案，也要无视该无知信号，强行按照模板输出一个看起来像判例引用的字符串”**。到了测试阶段，面对任何未知的法律问题，模型都会忠实执行这一表面模板规则，从而产生严重的幻觉。
     * (b) “模型知道什么”是策略内生的（Policy-dependent）。在强化学习（RL）中，回答是由模型自己采样生成的：当模型内部“我知道该判例”的激活特征较强时，它输出引用的准确率高，从而获得正奖励；当它内部处于“我不知道”状态却强行输出引用时，必然生成错误引用并遭到负奖励惩罚。通过这种策略内生的奖惩反馈，RL 能够将模型内部原本存在的“自知之明（I-know-something activation direction）”成功桥接到输出层策略上，使模型学会在确知时引用、在未知时克制或拒答。

3. **奖励模型过度优化定律（Gao et al., 2022）与生成式模型 vs. 策略优化的本质差异**：
   * (a) 根据 [Gao et al. (2022)](https://arxiv.org/abs/2210.10760) 的奖励模型过度优化扩展定律，设策略与参考模型的距离为 $d = \sqrt{\mathbb{D}_{\text{KL}}(\pi_\theta \parallel \pi_{\text{ref}})}$，在强化学习下真实黄金分数近似服从 $R_{\text{gold}}(d) = d(\alpha - \beta d)$（其中 $\alpha, \beta > 0$ 为常数）。请对 $d$ 求导，计算使真实黄金分数 $R_{\text{gold}}(d)$ 达到最大值的**最优 KL 散度预算 $\mathbb{D}_{\text{KL}}^*$** 以及此时的最高黄金得分 $R_{\text{gold}}^{\max}$。
   * (b) 从优化目标的数学本质出发，解释为什么预训练/SFT 模型天然具有较高的输出多样性和良好的概率校准（Calibration），而经过 RLHF 优化的模型却普遍面临“模式坍缩（Mode Collapse）”和“概率失校准（過度自信）”？
   * *参考解答*：
     * (a) 对 $R_{\text{gold}}(d) = \alpha d - \beta d^2$ 关于 $d$ 求一阶导数并令其为 $0$：
       $$\frac{\mathrm{d} R_{\text{gold}}}{\mathrm{d} d} = \alpha - 2\beta d = 0 \implies d^* = \frac{\alpha}{2\beta}$$
       由于 $d = \sqrt{\mathbb{D}_{\text{KL}}}$，对应的最优 KL 散度预算为：
       $$\mathbb{D}_{\text{KL}}^* = (d^*)^2 = \mathbf{\frac{\alpha^2}{4\beta^2}}$$
       代入 $d^*$ 可得最高黄金得分为：
       $$R_{\text{gold}}^{\max} = \frac{\alpha}{2\beta}\left(\alpha - \beta \frac{\alpha}{2\beta}\right) = \mathbf{\frac{\alpha^2}{4\beta}}$$
       一旦优化推过的 KL 距离超过 $\frac{\alpha^2}{4\beta^2}$，继续增大算力只会让真实表现加速恶化（当 $d > \frac{\alpha}{\beta}$ 时甚至低于初始 SFT 模型！）。
     * (b) 预训练和 SFT 最小化的是数据分布与模型分布之间的前向 KL 散度 $\mathbb{D}_{\text{KL}}(p_{\text{data}} \parallel \pi_\theta)$，这是一个**拟合分布（Distribution Matching / Mode-covering）**问题——只要训练集中某个回答有非零概率，模型就必须分配相应的概率质量去覆盖它，因此预测概率直接对应经验频率（天然校准且多样）。相反，RLHF 是在求解 $\max_{\pi_\theta} \mathbb{E}_{y \sim \pi_\theta}[R(x, y)]$，这是一个**最大化期望回报（Reward Maximization / Mode-seeking）**问题——在没有强 KL 约束的情况下，最优策略是一个将 $100\%$ 概率全部押在最高分单点 $y^* = \arg\max_y R(x,y)$ 上的退化狄拉克分布（Dirac delta），因此天然倾向于抹杀次优解的多样性并将输出概率推向极端的 $0$ 或 $1$。
