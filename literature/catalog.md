# Video Anomaly Understanding — 精选核验目录

这是从结构化数据生成的精选核验目录，并非完整文献综述。原仓库的会议笔记作为额外资料保留，尚未全面复核，不应视作本目录的核验条目。

数据更新时间：2026-09-30 · 54 篇论文。另见 [按年份阅读](../llm4vad.md)、[发表索引](venues.md)、[阅读路线](reading-guide.md)与[引用导出](citations.md)。

## Core research / 核心研究

## 语义对齐与融合

创新：字幕特征融合、视觉语言对齐和运动语言监督，为异常建立可解释的语义表征。

研究问题：如何把外观、运动与语言转化为异常相关的表征？

<a id="paper-vadclip"></a>

### VadCLIP: Adapting Vision-Language Models for Weakly Supervised Video Anomaly Detection

**2024 · AAAI** · [paper](<https://arxiv.org/abs/2308.11681>) · [code](<https://github.com/nwpu-zxr/VadCLIP>) · [引用 / BibTeX](<citations.md#cite-vadclip>)

**创新：视觉语言双分支对齐**

利用冻结 CLIP 的视觉语言关联，通过双分支完成粗粒度和细粒度异常检测。

- 任务：异常检测、异常定位
- 核心启示：语义对齐让检测分数能够关联异常类别。
- 阅读关注：类别语义对齐与开放式异常解释仍有距离。
- 核验：2026-09-29；[来源 1](<https://arxiv.org/abs/2308.11681>) · [来源 2](<https://ojs.aaai.org/index.php/AAAI/article/view/28423>)

<a id="paper-ex-vad"></a>

### Ex-VAD: Explainable Fine-grained Video Anomaly Detection Based on Visual-Language Models

**2025 · ICML** · [paper](<https://proceedings.mlr.press/v267/huang25ad.html>) · [引用 / BibTeX](<citations.md#cite-ex-vad>)

**创新：解释融合与标签对齐**

由帧字幕生成视频级异常解释，再结合视觉特征与标签增强对齐进行细粒度检测。

- 任务：异常检测、异常解释
- 核心启示：解释文本既是输出，也能参与检测表征学习。
- 阅读关注：更好的分类结果是否同时意味着更忠实的解释？
- 核验：2026-09-29；[来源 1](<https://proceedings.mlr.press/v267/huang25ad.html>)

<a id="paper-hawk"></a>

### Hawk: Learning to Understand Open-World Video Anomalies

**2024 · NeurIPS** · [paper](<https://arxiv.org/abs/2405.16886>) · [code](<https://github.com/jqtangust/hawk>) · [引用 / BibTeX](<citations.md#cite-hawk>)

**创新：运动语言监督对齐**

显式引入运动信息，并用异常视频描述与问答数据训练开放场景理解能力。

- 任务：异常解释、视频问答
- 核心启示：异常理解不仅需要外观，也需要动作变化与交互。
- 阅读关注：开放场景能力如何随数据来源和问题类型变化？
- 核验：2026-09-29；[来源 1](<https://arxiv.org/abs/2405.16886>) · [来源 2](<https://github.com/jqtangust/hawk/blob/main/README.md>)

<a id="paper-ovvad"></a>

### Open-Vocabulary Video Anomaly Detection

**2024 · CVPR** · [paper](<https://openaccess.thecvf.com/content/CVPR2024/html/Wu_Open-Vocabulary_Video_Anomaly_Detection_CVPR_2024_paper.html>) · [引用 / BibTeX](<citations.md#cite-ovvad>)

**创新：语言知识与异常合成**

将开放词汇检测分解为类别无关检测与类别识别，用语言知识和合成未知异常支持未见类别。

- 任务：异常检测
- 核心启示：异常检测之外，还需回答未知异常属于什么语义类别。
- 阅读关注：合成异常与真实未见事件之间的差异如何影响识别？
- 核验：2026-09-29；[来源 1](<https://openaccess.thecvf.com/content/CVPR2024/html/Wu_Open-Vocabulary_Video_Anomaly_Detection_CVPR_2024_paper.html>) · [来源 2](<https://openaccess.thecvf.com/content/CVPR2024/papers/Wu_Open-Vocabulary_Video_Anomaly_Detection_CVPR_2024_paper.pdf>)

<a id="paper-tpwng"></a>

### Text Prompt with Normality Guidance for Weakly Supervised Video Anomaly Detection

**2024 · CVPR** · [paper](<https://openaccess.thecvf.com/content/CVPR2024/html/Yang_Text_Prompt_with_Normality_Guidance_for_Weakly_Supervised_Video_Anomaly_CVPR_2024_paper.html>) · [引用 / BibTeX](<citations.md#cite-tpwng>)

**创新：正常引导伪标签学习**

将事件描述与视频帧对齐，结合正常性视觉提示生成帧级伪标签，再进行时序自训练。

- 任务：异常检测、异常定位
- 核心启示：正常性参照可把事件文字转化为更细的弱监督信号。
- 阅读关注：正常性提示与文本对齐误差如何共同影响伪标签？
- 核验：2026-09-29；[来源 1](<https://openaccess.thecvf.com/content/CVPR2024/html/Yang_Text_Prompt_with_Normality_Guidance_for_Weakly_Supervised_Video_Anomaly_CVPR_2024_paper.html>) · [来源 2](<https://arxiv.org/abs/2404.08531>)

<a id="paper-anomize"></a>

### Anomize: Better Open Vocabulary Video Anomaly Detection

**2025 · CVPR** · [paper](<https://openaccess.thecvf.com/content/CVPR2025/html/Li_Anomize_Better_Open_Vocabulary_Video_Anomaly_Detection_CVPR_2025_paper.html>) · [引用 / BibTeX](<citations.md#cite-anomize>)

**创新：多源语义与标签关系**

结合多层视觉信息与匹配文本，并利用标签关系编码新类别，改善未见异常的检测和语义分类。

- 任务：异常检测
- 核心启示：开放词汇识别需要同时处理异常分数与新类别语义对齐。
- 阅读关注：标签关系对视觉相似但语义不同的异常有多大帮助？
- 核验：2026-09-29；[来源 1](<https://openaccess.thecvf.com/content/CVPR2025/html/Li_Anomize_Better_Open_Vocabulary_Video_Anomaly_Detection_CVPR_2025_paper.html>)

<a id="paper-alert-clip"></a>

### Alert-CLIP: Abnormality-aware Latent-Enhanced Representation Tuning of CLIP for Video Anomaly Detection

**2026 · CVPR** · [paper](<https://openaccess.thecvf.com/content/CVPR2026/html/Zhu_Alert-CLIP_Abnormality-aware_Latent-Enhanced_Representation_Tuning_of_CLIP_for_Video_Anomaly_CVPR_2026_paper.html>) · [引用 / BibTeX](<citations.md#cite-alert-clip>)

**创新：区域语义多层对齐**

通过视频与标签、区域与文本、区域与语义的多层对齐，增强视觉语言表征对正常和异常的区分能力。

- 任务：异常检测
- 核心启示：语言异常判断的基础是视觉语言空间能否区分相近的正常与异常描述。
- 阅读关注：区域描述与困难负样本的收益能否迁移到新的场景定义？
- 核验：2026-09-29；[来源 1](<https://openaccess.thecvf.com/content/CVPR2026/html/Zhu_Alert-CLIP_Abnormality-aware_Latent-Enhanced_Representation_Tuning_of_CLIP_for_Video_Anomaly_CVPR_2026_paper.html>)

<a id="paper-lavida"></a>

### No Need For Real Anomaly: MLLM Empowered Zero-Shot Video Anomaly Detection

**2026 · CVPR** · [paper](<https://openaccess.thecvf.com/content/CVPR2026/html/Dai_No_Need_For_Real_Anomaly_MLLM_Empowered_Zero-Shot_Video_Anomaly_CVPR_2026_paper.html>) · [project](<https://github.com/VitaminCreed/LAVIDA>) · [引用 / BibTeX](<citations.md#cite-lavida>)

**创新：伪异常与反向注意力**

仅用伪异常训练，结合多模态大模型语义理解与反向注意力词元压缩，实现零样本帧级和像素级异常检测。

- 任务：异常检测、异常定位
- 核心启示：可通过合成暴露异常语义，再检验对真实异常的零样本迁移。
- 阅读关注：伪异常覆盖的语义与真实上下文依赖异常有多大差距？
- 核验：2026-09-29；[来源 1](<https://openaccess.thecvf.com/content/CVPR2026/html/Dai_No_Need_For_Real_Anomaly_MLLM_Empowered_Zero-Shot_Video_Anomaly_CVPR_2026_paper.html>) · [来源 2](<https://github.com/VitaminCreed/LAVIDA>) · [来源 3](<https://openaccess.thecvf.com/content/CVPR2026/papers/Dai_No_Need_For_Real_Anomaly_MLLM_Empowered_Zero-Shot_Video_Anomaly_CVPR_2026_paper.pdf>)

<a id="paper-anomalycraft"></a>

### AnomalyCraft-700K: Component-Level Controllable and Verifiable Synthetic Anomalies for Fine-Grained Video Anomaly Understanding

**2026 · arXiv** · [paper](<https://arxiv.org/abs/2609.06978>) · [project](<https://github.com/Eagen-l/AnomalyCraft>) · [引用 / BibTeX](<citations.md#cite-anomalycraft>)

**创新：语义组件控制生成与逐项验证**

以细粒度语义组件控制异常视频生成，逐组件校正文图不一致，并构造类别相关的困难正常样本，支持多任务异常理解。

- 任务：异常检测、异常解释、异常推理
- 核心启示：合成数据的价值取决于异常语义可控、标注可验证及正常边界足够困难。
- 阅读关注：生成伪迹、组件覆盖和合成到真实视频的迁移差距。
- 核验：2026-09-30；[来源 1](<https://arxiv.org/abs/2609.06978>) · [来源 2](<https://github.com/Eagen-l/AnomalyCraft>)

<a id="paper-td-vad"></a>

### TD-VAD: Breaking Visual Dependence in Video Anomaly Detection with Text-Driven Learning

**2026 · ICML** · [paper](<https://arxiv.org/abs/2608.11820>) · [引用 / BibTeX](<citations.md#cite-td-vad>)

**创新：文本时序监督与事件演化注意力**

以 LLM 生成的时序事件文本训练检测器，通过事件演化因果注意力建模长短期依赖，推理时用冻结 CLIP 对齐视频。

- 任务：异常检测
- 兼属方法：时序分层与记忆；[归类依据](<https://arxiv.org/abs/2608.11820>) — 编辑归类：冻结 CLIP 实现文本视觉对齐，事件演化因果注意力建模长短期时序依赖；兼属语义对齐与时序建模。
- 核心启示：用时序事件文本替代目标域异常视频。
- 阅读关注：主要输出异常分数；文本到视频的模态差距及语言先验偏差仍需检查。
- 核验：2026-09-30；[来源 1](<https://arxiv.org/abs/2608.11820>) · [来源 2](<https://icml.cc/virtual/2026/poster/65928>)

<a id="paper-headhunt-vad"></a>

### HeadHunt-VAD: Hunting Robust Anomaly-Sensitive Heads in MLLM for Tuning-Free Video Anomaly Detection

**2026 · AAAI** · [paper](<https://arxiv.org/abs/2512.17601>) · [code](<https://github.com/CebCai/HeadHunt-VAD>) · [引用 / BibTeX](<citations.md#cite-headhunt-vad>)

**创新：稳定异常敏感注意力头探测**

从冻结 MLLM 内部筛选对多种提示稳定的异常敏感注意力头，再用轻量评分器和时序定位器读取其特征。

- 任务：异常检测、异常定位
- 核心启示：从内部语义表征读取异常证据。
- 阅读关注：仅冻结大模型，评分器仍需少量校准数据；主任务为检测而非开放式解释。
- 核验：2026-09-30；[来源 1](<https://arxiv.org/abs/2512.17601>) · [来源 2](<https://github.com/CebCai/HeadHunt-VAD>)

<a id="paper-steervad"></a>

### Steering and Rectifying Latent Representation Manifolds in Frozen Multi-modal LLMs for Video Anomaly Detection

**2026 · ICLR** · [paper](<https://arxiv.org/abs/2602.24021>) · [引用 / BibTeX](<citations.md#cite-steervad>)

**创新：潜在异常专家头与上下文表示校正**

以表示可分性筛选潜在异常专家头，训练层次元控制器按上下文缩放其表示；异常片段可交回冻结模型生成事后解释。

- 任务：异常检测、异常解释
- 核心启示：从被动读取转向干预异常语义表征。
- 阅读关注：控制器和评分器仍需少量训练数据；事后生成解释不等同于检测依据的忠实证明。
- 核验：2026-09-30；[来源 1](<https://arxiv.org/abs/2602.24021>) · [来源 2](<https://iclr.cc/virtual/2026/papers.html>)

## 语言判据与提示优化

创新：字幕到判断的语言中介、正常规则归纳与引导问题优化，将异常标准显式化。

研究问题：有了语义表征，怎样构造场景适用的异常判据？

<a id="paper-lavad"></a>

### Harnessing Large Language Models for Training-free Video Anomaly Detection

**2024 · CVPR** · [paper](<https://arxiv.org/abs/2404.01014>) · [code](<https://github.com/lucazanella/lavad>) · [project](<https://lucazanella.github.io/lavad/>) · [引用 / BibTeX](<citations.md#cite-lavad>)

**创新：字幕聚合语言评分**

先描述帧内容，再由语言模型聚合时序信息并估计异常分数。

- 任务：异常检测、异常定位
- 核心启示：把预训练模型组合成无需目标数据训练的异常检测流程。
- 阅读关注：描述噪声及文本压缩会丢失哪些视觉细节？
- 核验：2026-09-29；[来源 1](<https://arxiv.org/abs/2404.01014>) · [来源 2](<https://openaccess.thecvf.com/content/CVPR2024/papers/Zanella_Harnessing_Large_Language_Models_for_Training-free_Video_Anomaly_Detection_CVPR_2024_paper.pdf>)

<a id="paper-anomalyruler"></a>

### Follow the Rules: Reasoning for Video Anomaly Detection with Large Language Models

**2024 · ECCV** · [paper](<https://arxiv.org/abs/2407.10299>) · [code](<https://github.com/Yuchen413/AnomalyRuler>) · [引用 / BibTeX](<citations.md#cite-anomalyruler>)

**创新：正常规则归纳演绎**

从少量正常样本归纳场景规则，再依据规则判断测试视频中的异常。

- 任务：异常检测、异常推理
- 核心启示：正常性可以写成可检查、可调整的语言规则。
- 阅读关注：少量正常参考是否覆盖场景中的合理变化？
- 核验：2026-09-29；[来源 1](<https://arxiv.org/abs/2407.10299>) · [来源 2](<https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/10568.pdf>)

<a id="paper-vera"></a>

### VERA: Explainable Video Anomaly Detection via Verbalized Learning of Vision-Language Models

**2025 · CVPR** · [paper](<https://arxiv.org/abs/2412.01095>) · [code](<https://github.com/vera-framework/VERA>) · [引用 / BibTeX](<citations.md#cite-vera>)

**创新：语言反馈优化问题**

用视频级弱标签优化自然语言引导问题，让冻结 VLM 分解并判断异常模式。

- 任务：异常检测、异常解释
- 核心启示：不更新模型权重，也可以学习任务专属的推理提示。
- 阅读关注：语言问题的适应性与跨场景迁移需要分开验证。
- 核验：2026-09-30；[来源 1](<https://arxiv.org/abs/2412.01095>) · [来源 2](<https://openaccess.thecvf.com/content/CVPR2025/papers/Ye_VERA_Explainable_Video_Anomaly_Detection_via_Verbalized_Learning_of_Vision-Language_CVPR_2025_paper.pdf>) · [来源 3](<https://github.com/vera-framework/VERA>)

<a id="paper-eval"></a>

### EVAL: Explainable Video Anomaly Localization

**2023 · CVPR** · [paper](<https://openaccess.thecvf.com/content/CVPR2023/html/Singh_EVAL_Explainable_Video_Anomaly_Localization_CVPR_2023_paper.html>) · [引用 / BibTeX](<citations.md#cite-eval>)

**创新：对象运动属性解释**

以对象类别、运动方向和速度等可读属性建立位置相关的正常模式，解释局部异常。

- 任务：异常检测、异常定位、异常解释
- 核心启示：异常解释可从可核查的对象和运动属性开始，而不必依赖自由文本。
- 阅读关注：可读属性是否覆盖需要长期交互背景的异常？
- 核验：2026-09-29；[来源 1](<https://openaccess.thecvf.com/content/CVPR2023/html/Singh_EVAL_Explainable_Video_Anomaly_Localization_CVPR_2023_paper.html>)

<a id="paper-lagovad"></a>

### Language-guided Open-world Video Anomaly Detection under Weak Supervision

**2026 · ICLR** · [paper](<https://proceedings.iclr.cc/paper_files/paper/2026/hash/f88bec15cc4cb56b432ee040bb63f94f-Abstract-Conference.html>) · [code](<https://github.com/Kamino666/LaGoVAD-PreVAD>) · [引用 / BibTeX](<citations.md#cite-lagovad>)

**创新：自然语言条件化异常定义**

把用户给定的自然语言异常定义作为推理输入，结合动态视频合成与负样本对比学习训练模型，并建设带异常定义的 PreVAD 数据。

- 任务：异常检测
- 方法与场景标签：开放世界异常检测、语言条件判据
- 核心启示：异常标准可以变化；把定义作为显式输入，比假设类别永远对应同一异常标签更贴近开放场景。
- 阅读关注：核心输出仍是条件化异常分数；具备语言输入并不等同于能够生成完整因果解释。
- 核验：2026-09-29；[来源 1](<https://proceedings.iclr.cc/paper_files/paper/2026/hash/f88bec15cc4cb56b432ee040bb63f94f-Abstract-Conference.html>)

<a id="paper-lrpo"></a>

### Linguistic Relative Policy Optimization for Video Anomaly Reasoning

**2026 · ICML** · [paper](<https://arxiv.org/abs/2607.00654>) · [引用 / BibTeX](<citations.md#cite-lrpo>)

**创新：组相对语言经验优化**

从多条推理轨迹的组内语义优势归纳通用与场景经验，以语言先验注入上下文而不更新模型参数。

- 任务：异常检测、异常推理
- 核心启示：将轨迹优势转化为可复用语言经验。
- 阅读关注：经验归纳仍受基础模型感知与奖励设计影响；免参数更新不等于无需构建经验。
- 核验：2026-09-30；[来源 1](<https://arxiv.org/abs/2607.00654>) · [来源 2](<https://icml.cc/virtual/2026/poster/64285>)

<a id="paper-probe-vad"></a>

### Probe-VAD: Ordinal Likelihood Probing for Training-Free Video Anomaly Detection

**2026 · arXiv** · [paper](<https://arxiv.org/abs/2609.17211>) · [code](<https://github.com/yvestine/Probe-VAD>) · [引用 / BibTeX](<citations.md#cite-probe-vad>)

**创新：序数语言探测与一致性评分**

以冻结 VLM 对有序严重程度阈值作是／否判断，通过续写似然与序数一致性约束得到连续异常分数。

- 任务：异常检测、异常定位
- 核心启示：语言判据可通过概率接口转为细粒度判断，不依赖先生成字幕。
- 阅读关注：严重程度排序不等同于异常解释，仍需独立检查判断对场景规范与可见证据的依赖。
- 核验：2026-09-30；[来源 1](<https://arxiv.org/abs/2609.17211>) · [来源 2](<https://github.com/yvestine/Probe-VAD>)

## 时序分层与记忆

创新：多粒度指令、事件边界建模、历史分数记忆，以及语义层级下的边界评估。

研究问题：判据应作用于哪一段时间，又需要保留多少历史？

<a id="paper-holmes-vau"></a>

### Holmes-VAU: Towards Long-term Video Anomaly Understanding at Any Granularity

**2025 · CVPR** · [paper](<https://arxiv.org/abs/2412.06171>) · [code](<https://github.com/pipixin321/HolmesVAU>) · [引用 / BibTeX](<citations.md#cite-holmes-vau>)

**创新：多粒度指令与采样**

构建多粒度异常指令数据，并用异常聚焦采样连接片段、事件和整段视频的理解。

- 任务：异常检测、异常解释、视频问答
- 核心启示：长视频 VAU 要同时处理采样与多粒度语义。
- 阅读关注：聚焦异常的采样是否保留充分的前因后果？
- 核验：2026-09-29；[来源 1](<https://arxiv.org/abs/2412.06171>) · [来源 2](<https://openaccess.thecvf.com/content/CVPR2025/papers/Zhang_Holmes-VAU_Towards_Long-term_Video_Anomaly_Understanding_at_Any_Granularity_CVPR_2025_paper.pdf>)

<a id="paper-eventvad"></a>

### EventVAD: Training-Free Event-Aware Video Anomaly Detection

**2025 · ACM MM** · [paper](<https://arxiv.org/abs/2504.13092>) · [code](<https://github.com/YihuaJerry/EventVAD>) · [引用 / BibTeX](<citations.md#cite-eventvad>)

**创新：时空图划分事件边界**

利用动态时空图寻找事件边界，再通过层级提示完成事件级异常推理。

- 任务：异常检测、异常定位、异常推理
- 核心启示：先确定事件单元，能让长视频推理减少无关上下文。
- 阅读关注：边界划分错误如何影响后续异常判断？
- 核验：2026-09-29；[来源 1](<https://arxiv.org/abs/2504.13092>)

<a id="paper-monitor"></a>

### MoniTor: Exploiting Large Language Models with Instruction for Online Video Anomaly Detection

**2025 · NeurIPS** · [paper](<https://arxiv.org/abs/2510.21449>) · [引用 / BibTeX](<citations.md#cite-monitor>)

**创新：流式记忆与分数队列**

使用流式输入、历史预测记忆和分数队列，在不训练的条件下持续判断异常。

- 任务：异常检测、异常定位
- 核心启示：在线异常理解必须明确当前可见的信息范围。
- 阅读关注：在线约束、延迟与历史错误累积的影响。
- 核验：2026-09-29；[来源 1](<https://arxiv.org/abs/2510.21449>)

<a id="paper-valu"></a>

### VALU: A Benchmark for Video Anomaly Temporal Localization and Understanding at Multiple Semantic Levels

**2026 · ACL** · [paper](<https://aclanthology.org/2026.acl-long.56/>) · [引用 / BibTeX](<citations.md#cite-valu>)

**创新：语义分层边界评估**

用多个语义层级定义异常边界，联合考查定位、时间 grounding 与细节辨别。

- 任务：异常定位、异常解释、基准评测
- 核心启示：异常的起止时间取决于所采用的语义范围。
- 阅读关注：不同标注边界定义是否让结果失去可比性？
- 核验：2026-09-29；[来源 1](<https://aclanthology.org/2026.acl-long.56/>)

<a id="paper-uca-paper"></a>

### Towards Surveillance Video-and-Language Understanding: New Dataset, Baselines, and Challenges

**2024 · CVPR** · [paper](<https://openaccess.thecvf.com/content/CVPR2024/html/Yuan_Towards_Surveillance_Video-and-Language_Understanding_New_Dataset_Baselines_and_Challenges_CVPR_2024_paper.html>) · [project](<https://xuange923.github.io/Surveillance-Video-Understanding>) · [引用 / BibTeX](<citations.md#cite-uca-paper>)

**创新：事件句子与时间对齐**

为 UCF-Crime 添加句子级事件描述与时间标注，建立监控视频定位、描述等语言理解任务基线。

- 任务：异常检测、异常定位、异常解释
- 核心启示：监控理解需要把事件语义和发生时间一起标注。
- 阅读关注：通用视频语言模型迁移至长监控视频时，哪些任务最受背景和时长影响？
- 核验：2026-09-29；[来源 1](<https://openaccess.thecvf.com/content/CVPR2024/html/Yuan_Towards_Surveillance_Video-and-Language_Understanding_New_Dataset_Baselines_and_Challenges_CVPR_2024_paper.html>) · [来源 2](<https://xuange923.github.io/Surveillance-Video-Understanding>)

<a id="paper-va-gpt"></a>

### Aligning Effective Tokens with Video Anomaly in Large Language Models

**2025 · ICCV** · [paper](<https://openaccess.thecvf.com/content/ICCV2025/html/Chen_Aligning_Effective_Tokens_with_Video_Anomaly_in_Large_Language_Models_ICCV_2025_paper.html>) · [引用 / BibTeX](<citations.md#cite-va-gpt>)

**创新：时空有效词元对齐**

通过空间有效词元选择与时间有效词元生成，减少冗余视觉信息，支持异常总结和时间定位。

- 任务：异常定位、异常解释、视频问答
- 核心启示：进入语言模型的时空证据如何筛选，本身就是异常理解的关键。
- 阅读关注：词元选择保留局部异常时，是否也保留解释所需的上下文？
- 核验：2026-09-29；[来源 1](<https://openaccess.thecvf.com/content/ICCV2025/html/Chen_Aligning_Effective_Tokens_with_Video_Anomaly_in_Large_Language_Models_ICCV_2025_paper.html>) · [来源 2](<https://openaccess.thecvf.com/content/ICCV2025/papers/Chen_Aligning_Effective_Tokens_with_Video_Anomaly_in_Large_Language_Models_ICCV_2025_paper.pdf>)

<a id="paper-vadtree"></a>

### VADTree: Explainable Training-Free Video Anomaly Detection via Hierarchical Granularity-Aware Tree

**2025 · NeurIPS** · [paper](<https://papers.nips.cc/paper_files/paper/2025/hash/da19d18dfc5434bf419ce9c113f1865f-Abstract-Conference.html>) · [code](<https://github.com/wenlongli10/VADTree>) · [引用 / BibTeX](<citations.md#cite-vadtree>)

**创新：事件边界驱动的层次粒度树**

由通用事件边界构建层次粒度树，在粗细事件节点上分别调用 VLM 感知与 LLM 推理，再融合跨粒度异常分数。

- 任务：异常检测、异常解释
- 方法与场景标签：多粒度时序
- 核心启示：先确定具有事件意义的时间单元，有助于减少固定窗口切碎事件或混入无关内容。
- 阅读关注：仍依赖事件边界模型及节点描述质量；免训练不等于免除多阶段模型推理成本。
- 核验：2026-09-29；[来源 1](<https://papers.nips.cc/paper_files/paper/2025/hash/da19d18dfc5434bf419ce9c113f1865f-Abstract-Conference.html>) · [来源 2](<https://github.com/wenlongli10/VADTree>)

<a id="paper-flashback"></a>

### Flashback: Memory-Driven Zero-shot, Real-time Video Anomaly Detection

**2025 · arXiv** · [paper](<https://arxiv.org/abs/2505.15205>) · [引用 / BibTeX](<citations.md#cite-flashback>)

**创新：离线语义记忆与在线匹配**

离线用语言模型构建正常与异常字幕记忆，在线将视频片段与文本记忆匹配，以检索结果给出异常判断与文本依据。

- 任务：异常检测、异常解释
- 核心启示：把语言模型调用移到离线阶段，可以降低在线判断成本。
- 阅读关注：检索字幕是否真实匹配当前片段，固定记忆如何覆盖未知异常？
- 核验：2026-09-30；[来源 1](<https://arxiv.org/abs/2505.15205>) · [来源 2](<https://arxiv.org/html/2505.15205v2>)

<a id="paper-reactvau"></a>

### ReactVAU: A Slow-Fast Decoupled Framework for Streaming Video Anomaly Understanding

**2026 · ECCV** · [paper](<https://arxiv.org/abs/2609.07941>) · [引用 / BibTeX](<citations.md#cite-reactvau>)

**创新：快慢解耦与异常持久记忆**

用轻量检测连续筛查视频，以异常感知记忆保留短暂证据，仅在可疑事件触发重型模型进行语义验证与原因描述。

- 任务：异常检测、异常解释、异常推理
- 方法与场景标签：流式理解、条件触发
- 核心启示：流式理解需要同时控制证据遗忘和重型模型调用频率。
- 阅读关注：分别检查触发延迟、异常漏检与记忆保真度，不能只比较离线检测分数。
- 核验：2026-09-30；[来源 1](<https://arxiv.org/abs/2609.07941>)

## 主动观察与工具决策

创新：场景规划、工具调用、补充采样与交互策略学习，让异常证据获取随当前疑点调整。

研究问题：当前证据不足时，怎样决定下一次观察或工具调用？

<a id="paper-panda"></a>

### PANDA: Towards Generalist Video Anomaly Detection via Agentic AI Engineer

**2025 · NeurIPS** · [paper](<https://arxiv.org/abs/2509.26386>) · [project](<https://github.com/showlab/PANDA>) · [引用 / BibTeX](<citations.md#cite-panda>)

**创新：场景规划与工具反思**

把场景规划、工具反思与经验记忆组合为通用异常检测智能体。

- 任务：异常检测、异常推理
- 核心启示：检测流程本身可以成为按场景调整的决策过程。
- 阅读关注：工具调用、记忆和检测质量之间的成本收益。
- 核验：2026-09-29；[来源 1](<https://arxiv.org/abs/2509.26386>) · [来源 2](<https://github.com/showlab/PANDA>)

<a id="paper-anom-pi"></a>

### Learning to Watch: Active Video Anomaly Understanding via Interleaved Policy Optimization

**2026 · ICML** · [paper](<https://arxiv.org/abs/2607.00622>) · [引用 / BibTeX](<citations.md#cite-anom-pi>)

**创新：交替推理与观察策略**

将推理与时间回溯、区间扩展、细粒度采样交替执行，学习主动获取证据的策略。

- 任务：异常推理、异常解释
- 核心启示：“下一步观察什么”也可以成为 VAU 的学习对象。
- 阅读关注：证据收益、任务成功和交互成本是否被共同衡量？
- 核验：2026-09-30；[来源 1](<https://arxiv.org/abs/2607.00622>) · [来源 2](<https://icml.cc/Downloads/2026>)

<a id="paper-memovad"></a>

### MemoVAD: Resource-Efficient Video Anomaly Detection via Dynamic Semantic Memory in Edge Computing Scenarios

**2026 · IJCAI** · [paper](<https://www.ijcai.org/proceedings/2026/618>) · [引用 / BibTeX](<citations.md#cite-memovad>)

**创新：不确定性门控与动态语义记忆**

边缘轻量检测器维护因果时序上下文，仅对高不确定且语义新颖的片段查询云端 VLM，并缓存验证后的语义原型用于后续检索。

- 任务：异常检测
- 方法与场景标签：流式异常检测、选择性模型调用、语义记忆
- 兼属方法：时序分层与记忆；[归类依据](<https://www.ijcai.org/proceedings/2026/618>) — 编辑归类：不确定性门控决定云端 VLM 调用，动态语义记忆缓存已验证原型；兼属主动决策与时序记忆。
- 核心启示：把何时调用大型语义模型作为决策问题，让曾经获取的异常证据在后续视频中复用。
- 阅读关注：选择性查询取决于边缘不确定性估计；未被触发的片段无法获得云端语义补充。
- 核验：2026-09-29；[来源 1](<https://www.ijcai.org/proceedings/2026/618>)

<a id="paper-a2seek"></a>

### A2Seek: Towards Reasoning-Centric Benchmark for Aerial Anomaly Understanding

**2025 · NeurIPS Datasets and Benchmarks** · [paper](<https://proceedings.neurips.cc/paper_files/paper/2025/hash/de02de513503962e1d21035ab50ce661-Abstract-Datasets_and_Benchmarks_Track.html>) · [project](<https://2-mo.github.io/A2Seek/>) · [引用 / BibTeX](<citations.md#cite-a2seek>)

**创新：图式推理与主动区域观察**

构建含事件类别、时间戳、区域框和语言解释的真实航拍异常基准，以图式推理监督、A-GRPO 与区域 seeking 机制训练 A2Seek-R1。

- 任务：异常推理、空间定位、异常解释、基准评测
- 方法与场景标签：航拍异常理解
- 兼属方法：结构化推理与验证；[归类依据](<https://2-mo.github.io/A2Seek/>) — 编辑归类：A2Seek-R1 结合区域 seeking、图式思维监督及 A-GRPO；兼属主动观察与结构化推理。
- 核心启示：异常理解不仅要说明原因，也要在动态航拍视角下找到支撑判断的局部区域。
- 阅读关注：面向航拍视角与区域级证据的设计，迁移到固定监控或其他场景时仍需验证。
- 核验：2026-09-29；[来源 1](<https://proceedings.neurips.cc/paper_files/paper/2025/hash/de02de513503962e1d21035ab50ce661-Abstract-Datasets_and_Benchmarks_Track.html>)

<a id="paper-agenticvau"></a>

### AgenticVAU: Multi-Agent Explore-Verify Reasoning for Video Anomaly Understanding

**2026 · arXiv** · [paper](<https://arxiv.org/abs/2608.03779>) · [引用 / BibTeX](<citations.md#cite-agenticvau>)

**创新：多智能体探索验证与证据记忆**

用规则构建、搜索规划、视频观察和最终决策四类智能体，交替探索疑点与局部验证，并通过共享证据记忆协调判断。

- 任务：异常定位、异常推理、视频问答
- 核心启示：异常理解可以显式区分发现疑点、收集证据和作出结论。
- 阅读关注：多轮观察成本、证据记忆错误与停止条件如何影响可靠性？
- 核验：2026-09-30；[来源 1](<https://arxiv.org/abs/2608.03779>)

<a id="paper-vagu-gts"></a>

### VAGU & GtS: LLM-Based Benchmark and Framework for Joint Video Anomaly Grounding and Understanding

**2026 · AAAI** · [paper](<https://ojs.aaai.org/index.php/AAAI/article/view/42412>) · [引用 / BibTeX](<citations.md#cite-vagu-gts>)

**创新：先全局粗定位再局部细查**

先通过文本引导粗定位异常区间，再细查异常语义与时间边界，并构建 VAGU 和 JeAUG 联合评价定位与理解。

- 任务：异常定位、异常解释、视频问答、基准评测
- 核心启示：异常解释与时间边界应联合评估，局部细查可连接两种能力。
- 阅读关注：粗定位漏检会否阻止后续细查，联合分数是否掩盖单项退化？
- 核验：2026-09-30；[来源 1](<https://ojs.aaai.org/index.php/AAAI/article/view/42412>) · [来源 2](<https://arxiv.org/abs/2507.21507>) · [来源 3](<https://arxiv.org/abs/2608.11260>)

<a id="paper-slowfastvad"></a>

### SlowFastVAD: Video Anomaly Detection via Integrating Simple Detector and RAG-Enhanced Vision-Language Model

**2025 · arXiv** · [paper](<https://arxiv.org/abs/2504.10320>) · [引用 / BibTeX](<citations.md#cite-slowfastvad>)

**创新：快检测门控与检索增强慢推理**

快速检测器先给出异常置信度，仅将模糊片段交给检索增强 VLM，利用正常参考与推断异常模式组成知识库辅助判断。

- 任务：异常检测、异常推理
- 核心启示：推理计算可以集中分配到快速模型无法确定的片段。
- 阅读关注：快速检测器的自信误判是否绕过慢推理，检索知识覆盖如何影响泛化？
- 核验：2026-09-30；[来源 1](<https://arxiv.org/abs/2504.10320>)

## 结构化推理与验证

创新：因果问题分解、对象关系编码、反思修正与关键事实评估，让异常解释可以被检查。

研究问题：怎样组织推理，并检验结论是否抓住关键事实？

<a id="paper-cuva"></a>

### Uncovering What, Why and How: A Comprehensive Benchmark for Causation Understanding of Video Anomaly

**2024 · CVPR** · [paper](<https://arxiv.org/abs/2405.00181>) · [code](<https://github.com/fesvhtr/CUVA>) · [引用 / BibTeX](<citations.md#cite-cuva>)

**创新：事件因果任务分解**

用事件、原因和后果标注，将视频异常理解拓展到因果解释及评估。

- 任务：异常解释、异常推理、视频问答
- 核心启示：读 VAU 时，应把“发生什么”和“为什么发生”分开检查。
- 阅读关注：可见事件证据能支持多强的因果结论？
- 核验：2026-09-29；[来源 1](<https://arxiv.org/abs/2405.00181>)

<a id="paper-srvau-r1"></a>

### SRVAU-R1: Enhancing Video Anomaly Understanding via Reflection-Aware Learning

**2026 · arXiv** · [paper](<https://arxiv.org/abs/2602.01004>) · [引用 / BibTeX](<citations.md#cite-srvau-r1>)

**创新：反思修正序列训练**

构建初始推理、反思和修正推理的监督序列，再结合监督与强化微调。

- 任务：异常定位、异常推理
- 核心启示：把自我修正纳入学习目标，而不只要求更长的推理文本。
- 阅读关注：反思是否真的纠正了视觉证据不足的判断？
- 核验：2026-09-29；[来源 1](<https://arxiv.org/abs/2602.01004>)

<a id="paper-finevau"></a>

### FineVAU: A Novel Human-Aligned Benchmark for Fine-Grained Video Anomaly Understanding

**2026 · AAAI** · [paper](<https://arxiv.org/abs/2601.17258>) · [project](<https://finevau.github.io/>) · [引用 / BibTeX](<citations.md#cite-finevau>)

**创新：关键视觉事实评估**

围绕事件、参与者与位置构建细粒度标注，并用 FVScore 检查关键视觉信息。

- 任务：异常解释
- 核心启示：流畅的描述不等于抓住异常；评估需要追问关键事实。
- 阅读关注：自动扩展标注的质量及其与人类判断的一致性。
- 核验：2026-09-29；[来源 1](<https://arxiv.org/abs/2601.17258>)

<a id="paper-las-vad"></a>

### Weakly Supervised Video Anomaly Detection with Anomaly-Connected Components and Intention Reasoning

**2026 · CVPR** · [paper](<https://openaccess.thecvf.com/content/CVPR2026/html/Wang_Weakly_Supervised_Video_Anomaly_Detection_with_Anomaly-Connected_Components_and_Intention_CVPR_2026_paper.html>) · [引用 / BibTeX](<citations.md#cite-las-vad>)

**创新：语义连通与意图感知**

将视频帧组织为语义相连的成分，并结合行为意图与异常属性信息，区分外观相似的正常和异常行为。

- 任务：异常检测、异常定位
- 核心启示：动作外观相似时，异常判断需要考虑意图及其可见属性。
- 阅读关注：意图建模所依赖的视觉证据能否与行为先验区分开？
- 核验：2026-09-29；[来源 1](<https://openaccess.thecvf.com/content/CVPR2026/html/Wang_Weakly_Supervised_Video_Anomaly_Detection_with_Anomaly-Connected_Components_and_Intention_CVPR_2026_paper.html>)

<a id="paper-vad-r1"></a>

### Vad-R1: Towards Video Anomaly Reasoning via Perception-to-Cognition Chain-of-Thought

**2025 · NeurIPS** · [paper](<https://papers.nips.cc/paper_files/paper/2025/hash/abccc325c84dedf23dbe8de3f686c733-Abstract-Conference.html>) · [code](<https://github.com/wbfwonderful/Vad-R1>) · [引用 / BibTeX](<citations.md#cite-vad-r1>)

**创新：感知认知推理链与自验证奖励**

把视频异常推理独立为任务，构建从感知到认知的结构化推理链及 Vad-Reasoning 数据，并以带自验证的 AVA-GRPO 训练模型。

- 任务：异常推理、异常解释
- 方法与场景标签：强化学习
- 核心启示：把“看到了什么”与“为何异常”分步组织，再让训练奖励约束推理与结论。
- 阅读关注：结构化文本与自验证是训练机制；解释是否忠实于实际视觉证据仍需单独检查。
- 核验：2026-09-29；[来源 1](<https://papers.nips.cc/paper_files/paper/2025/hash/abccc325c84dedf23dbe8de3f686c733-Abstract-Conference.html>)

<a id="paper-cuebench"></a>

### CueBench: Advancing Unified Understanding of Context-Aware Video Anomalies in Real-World

**2026 · AAAI** · [paper](<https://ojs.aaai.org/index.php/AAAI/article/view/38209>) · [code](<https://github.com/Mia-YatingYu/Cue-R1>) · [引用 / BibTeX](<citations.md#cite-cuebench>)

**创新：上下文异常分类体系与分层奖励**

以条件性与绝对异常为核心组织场景、属性和事件层级，统一评估识别、定位、检测与预判，并训练具有分层可验证奖励的 Cue-R1。

- 任务：异常检测、异常推理、异常定位、异常预判、基准评测
- 方法与场景标签：上下文异常理解
- 核心启示：同一行为是否异常取决于安全条件与上下文，基准应显式检验这种条件依赖。
- 阅读关注：结果受所定义的事件、场景和属性覆盖范围约束；不能将基准表现直接等同于所有真实场景可靠性。
- 核验：2026-09-29；[来源 1](<https://ojs.aaai.org/index.php/AAAI/article/view/38209>)

<a id="paper-targetvau"></a>

### TargetVAU: Multimodal Anomaly-Aware Reasoning for Target Behavior Understanding in Videos

**2026 · AAAI** · [paper](<https://ojs.aaai.org/index.php/AAAI/article/view/38378>) · [引用 / BibTeX](<citations.md#cite-targetvau>)

**创新：个体时空交互图与指令推理**

结合全局和人体中心视觉特征，用异常引导采样与时空交互图刻画个体关系，再以指令微调语言模型识别异常个体并解释行为。

- 任务：异常推理、异常解释
- 方法与场景标签：个体异常理解、关系推理
- 核心启示：异常解释应明确“谁做了什么”，以关系结构补足全局事件描述对具体行为主体的忽略。
- 阅读关注：以可见个体及其交互为建模中心；遮挡和人体特征质量会影响细粒度主体解释。
- 核验：2026-09-29；[来源 1](<https://ojs.aaai.org/index.php/AAAI/article/view/38378>)

<a id="paper-holotrace"></a>

### HoloTrace: LLM-based Bidirectional Causal Knowledge Graph for Edge-Cloud Video Anomaly Detection

**2025 · ACM MM** · [paper](<https://doi.org/10.1145/3746027.3755185>) · [code](<https://github.com/kongyanye/HoloTrace-MM25>) · [引用 / BibTeX](<citations.md#cite-holotrace>)

**创新：双向因果知识图与边云更新**

用 LLM 构建并更新双向因果知识图，边缘侧结合隐马尔可夫模型进行事件推理与边界判断，云端依据关键帧更新事件关系。

- 任务：异常检测、异常推理
- 方法与场景标签：因果推理、边云协同
- 核心启示：将事件关系显式存入可更新结构，使边缘检测能够复用语言模型获得的语义知识。
- 阅读关注：图中的因果关系来自模型构建与更新，不能自动视为经干预验证的真实因果关系。
- 核验：2026-09-29；[来源 1](<https://doi.org/10.1145/3746027.3755185>)

<a id="paper-tau-bench"></a>

### TAU-Bench: From Anomaly Instance Tracking to Fine-Grained Video Anomaly Understanding

**2026 · arXiv** · [paper](<https://arxiv.org/abs/2608.05699>) · [project](<https://yarkupa.github.io/tau-bench.github.io/>) · [引用 / BibTeX](<citations.md#cite-tau-bench>)

**创新：实例轨迹与层级语义联合评估**

把异常实例轨迹、像素掩码与实例、事件、场景三级描述绑定，联合评估跟踪和细粒度理解是否指向同一异常对象。

- 任务：异常定位、异常解释、异常推理、基准评测
- 核心启示：合理的异常描述仍可能对应错误对象，理解评估需要实例级视觉依据。
- 阅读关注：语义评分与跟踪指标是否同时改善，而非只生成更流畅的描述？
- 核验：2026-09-30；[来源 1](<https://arxiv.org/abs/2608.05699>) · [来源 2](<https://yarkupa.github.io/tau-bench.github.io/>)

<a id="paper-vad-r1-plus"></a>

### Advancing Adaptive Multi-Stage Video Anomaly Reasoning: A Benchmark Dataset and Method

**2026 · arXiv** · [paper](<https://arxiv.org/abs/2601.10165>) · [project](<https://github.com/wbfwonderful/Vad-R1-Plus>) · [引用 / BibTeX](<citations.md#cite-vad-r1-plus>)

**创新：感知认知行动链与异常感知优化**

用感知、认知与行动三级思维链组织异常推理，并以异常感知的组相对策略优化训练支持不同推理深度和风险判断的模型。

- 任务：异常推理、视频问答
- 核心启示：异常理解可以进一步评估风险解释和决策建议所需的推理深度。
- 阅读关注：风险判断和行动建议是否有可见证据支持，弱监督奖励如何约束可靠性？
- 核验：2026-09-30；[来源 1](<https://arxiv.org/abs/2601.10165>) · [来源 2](<https://github.com/wbfwonderful/Vad-R1-Plus>)

<a id="paper-vau-r1"></a>

### VAU-R1: Advancing Video Anomaly Understanding via Reinforcement Fine-Tuning

**2025 · arXiv** · [paper](<https://arxiv.org/abs/2505.23504>) · [code](<https://github.com/GVCLab/VAU-R1>) · [引用 / BibTeX](<citations.md#cite-vau-r1>)

**创新：多任务奖励与强化微调**

通过任务专属奖励进行强化微调，并构建包含选择问答、推理依据、时间边界和描述的 VAU-Bench。

- 任务：异常定位、异常推理、视频问答
- 核心启示：问答、分类、推理与时间定位需要分别定义目标和评价协议。
- 阅读关注：格式、正确率与时间交并比奖励是否真正提升解释的视觉忠实度？
- 核验：2026-09-30；[来源 1](<https://arxiv.org/abs/2505.23504>) · [来源 2](<https://github.com/GVCLab/VAU-R1>)

<a id="paper-cg-coe"></a>

### Towards Trustworthy Video Anomaly Understanding: A Class-Guided Chain-of-Evaluation Metric and An Anomaly-focused Meta-Benchmark

**2026 · ICML** · [paper](<https://icml.cc/virtual/2026/poster/66013>) · [引用 / BibTeX](<citations.md#cite-cg-coe>)

**创新：类别引导评价链与元评测**

以类别约束的异常事件抽取与匹配构建评价链，并用 AEA 与 CVP 子集检验指标有效性及措辞扰动鲁棒性。

- 任务：异常解释、基准评测
- 核心启示：把异常语义正确性与措辞风格分开。
- 阅读关注：评价有效性仍需检验类别边界和事件抽取是否可靠；元评测不直接提升模型感知。
- 核验：2026-09-30；[来源 1](<https://icml.cc/virtual/2026/poster/66013>) · [来源 2](<https://openreview.net/forum?id=7waVdY1WmW>)

<a id="paper-urf-zs-hvaa"></a>

### A Unified Reasoning Framework for Holistic Zero-Shot Video Anomaly Analysis

**2025 · NeurIPS** · [paper](<https://proceedings.neurips.cc/paper_files/paper/2025/hash/2aa95cf3b6aefa84d6b001928b107b4e-Abstract-Conference.html>) · [code](<https://github.com/Rathgrith/URF-ZS-HVAA>) · [引用 / BibTeX](<citations.md#cite-urf-zs-hvaa>)

**创新：任务内细化与跨任务推理链**

任务内推理用视频上下文细化时间检测，任务间链式推理再引导冻结模型完成空间定位与文本解释。

- 任务：异常检测、异常定位、异常解释
- 核心启示：把时间检测、空间定位和解释串起来。
- 阅读关注：链式结果依赖前序时间检测；应检查错误传递和多阶段推理成本。
- 核验：2026-09-30；[来源 1](<https://proceedings.neurips.cc/paper_files/paper/2025/hash/2aa95cf3b6aefa84d6b001928b107b4e-Abstract-Conference.html>) · [来源 2](<https://rathgrith.github.io/Unified_Frame_VAA/>)

<a id="paper-vad-dpo"></a>

### Do LVLMs Truly Understand Video Anomalies? Revealing Hallucination via Co-Occurrence Patterns

**2025 · NeurIPS** · [paper](<https://papers.nips.cc/paper_files/paper/2025/hash/99b419554537c66bf27e5eb7a74c7de4-Abstract-Conference.html>) · [引用 / BibTeX](<citations.md#cite-vad-dpo>)

**创新：反例偏好优化抑制共现捷径**

诊断模型对物体与异常词语共现的捷径依赖，以视觉相似但语义相反的视频对进行偏好优化，增强场景语义判断。

- 任务：异常检测、异常推理
- 核心启示：用语义反例检验并纠正异常共现偏见。
- 阅读关注：反例覆盖范围决定可纠正的偏见；检测改进不能替代完整解释忠实性评测。
- 核验：2026-09-30；[来源 1](<https://papers.nips.cc/paper_files/paper/2025/hash/99b419554537c66bf27e5eb7a74c7de4-Abstract-Conference.html>)

<a id="paper-clue-vad"></a>

### CLUE-VAD: Structured Semantic Clues for Understanding Explainable Events in Video Anomaly Detection

**2026 · ECCV** · [paper](<https://eccv.ecva.net/virtual/2026/poster/4744>) · [引用 / BibTeX](<citations.md#cite-clue-vad>)

**创新：结构化语义线索与类别感知归因**

将片段分解为动作、环境和对象线索，用类别感知权重融合并将异常分数归因到线索及关键词，生成有依据的解释。

- 任务：异常检测、异常解释
- 核心启示：让异常分数对应具体语义因素。
- 阅读关注：字幕遗漏会限制证据；权重和关键词归因仍需与真实因果贡献区分。
- 核验：2026-09-30；[来源 1](<https://eccv.ecva.net/virtual/2026/poster/4744>) · [来源 2](<https://media.eventhosts.cc/Conferences/ECCV2026/pdfs/7292.pdf>)

<a id="paper-o-vad"></a>

### O-VAD: Industrial Video Anomaly Detection through Object-Centric Tracking and Reasoning

**2026 · ECCV** · [paper](<https://arxiv.org/abs/2607.18142>) · [code](<https://github.com/o-vad/O-VAD>) · [引用 / BibTeX](<citations.md#cite-o-vad>)

**创新：对象状态轨迹与时序推理**

在工业视频中跟踪对象状态随时间的演化，再对对象轨迹推理，定位异常对象与帧并输出异常过程和类型报告。

- 任务：异常检测、异常定位、异常解释
- 核心启示：从对象状态变化解释工业视频异常。
- 阅读关注：对象检测与跟踪错误可能传入推理；工业流程验证不能直接代表开放监控场景。
- 核验：2026-09-30；[来源 1](<https://arxiv.org/abs/2607.18142>) · [来源 2](<https://eccv.ecva.net/virtual/2026/poster/4659>)

<a id="paper-stch"></a>

### Streaming Video Crime Anticipation with Spatio-Temporal Causal Reasoning

**2026 · CVPR** · [paper](<https://openaccess.thecvf.com/content/CVPR2026/html/Wang_Streaming_Video_Crime_Anticipation_with_Spatio-Temporal_Causal_Reasoning_CVPR_2026_paper.html>) · [引用 / BibTeX](<citations.md#cite-stch>)

**创新：流式时空因果超图**

构建具有递进推理任务的 STCRC 基准，并用流式时空因果超图显式组织实体动态，支撑犯罪事件预判。

- 任务：异常预判、异常推理、基准评测
- 核心启示：把实体动态组织为犯罪预兆推理结构。
- 阅读关注：任务是事件预判，与事后异常理解需分别评测；因果结构依赖事件和实体标注质量。
- 核验：2026-09-30；[来源 1](<https://openaccess.thecvf.com/content/CVPR2026/html/Wang_Streaming_Video_Crime_Anticipation_with_Spatio-Temporal_Causal_Reasoning_CVPR_2026_paper.html>) · [来源 2](<https://cvpr.thecvf.com/virtual/2026/poster/39800>)

<a id="paper-pistachio"></a>

### Pistachio: Towards Synthetic, Balanced, and Long-Form Video Anomaly Benchmarks

**2026 · ECCV** · [paper](<https://arxiv.org/abs/2511.19474>) · [project](<https://pistachio-video.github.io>) · [引用 / BibTeX](<citations.md#cite-pistachio>)

**创新：可控生成与多事件评测**

通过可控场景、异常类型和时序叙事生成视频，构建检测与理解基准，包含事件级、视频级语义及多异常事件。

- 任务：异常检测、异常解释、异常推理、基准评测
- 核心启示：可控合成可补充长尾事件与语义标注覆盖。
- 阅读关注：合成视频表现需要与真实场景泛化、生成伪影和标注可靠性分开分析。
- 核验：2026-09-30；[来源 1](<https://arxiv.org/abs/2511.19474>) · [来源 2](<https://arxiv.org/html/2511.19474v6>)

<a id="paper-vane-bench"></a>

### VANE-Bench: Video Anomaly Evaluation Benchmark for Conversational LMMs

**2025 · NAACL Findings** · [paper](<https://aclanthology.org/2025.findings-naacl.171/>) · [project](<https://github.com/rohit901/VANE-Bench>) · [引用 / BibTeX](<citations.md#cite-vane-bench>)

**创新：合成与真实异常问答评测**

以视频问答评测模型对异常的检测和定位，同时覆盖五类合成视频不一致性与真实世界异常。

- 任务：异常检测、异常定位、视频问答、基准评测
- 核心启示：问答基准能暴露视频大模型对细微异常与不一致性的识别不足。
- 阅读关注：区分生成视频的物理不一致与真实监控异常，避免将两类结果解释为同一种能力。
- 核验：2026-09-30；[来源 1](<https://aclanthology.org/2025.findings-naacl.171/>)

## Datasets / 数据资源

完整协议与关联论文见 [数据集索引](benchmarks.md)。数据登记不要求图片，也不代表资源已开放下载。已有图片保留作者署名。

### UCF-Crime

**2018 · CVPR** · [来源入口](<https://www.crcv.ucf.edu/research/real-world-anomaly-detection-in-surveillance-videos/>)

真实监控长视频异常检测基准，也是多项语言异常理解工作的视觉来源。

- 评估协议：采用官方训练／测试划分；弱监督训练与帧级定位评估须区分。
- 图片：[UCF-Crime 官方方法示意图，包含监控视频片段](<https://www.crcv.ucf.edu/projects/real-world/method.png>)；署名：Waqas Sultani, Chen Chen, Mubarak Shah
- 核验：[来源 1](<https://openaccess.thecvf.com/content_cvpr_2018/html/Sultani_Real-World_Anomaly_Detection_CVPR_2018_paper.html>) · [来源 2](<https://www.crcv.ucf.edu/research/real-world-anomaly-detection-in-surveillance-videos/>)

### XD-Violence

**2020 · ECCV** · [来源入口](<https://roc-ng.github.io/XD-Violence/>)

包含音视频与多场景暴力事件的弱监督检测基准。

- 评估协议：报告所用模态与官方划分；不可将音视频方法和纯视觉方法混为同一设置。
- 图片：[XD-Violence 作者发布的多场景视频样例拼图](<https://roc-ng.github.io/XD-Violence/images/samples.png>)；署名：Peng Wu et al. / XD-Violence
- 核验：[来源 1](<https://roc-ng.github.io/XD-Violence/>) · [来源 2](<https://roc-ng.github.io/XD-Violence/>)

### CUVA

**2024 · CVPR** · [来源入口](<https://github.com/fesvhtr/CUVA>)

以异常事件的内容、原因与后果为核心的因果理解基准。

- 评估协议：按原论文任务与 MMEval 协议评估；定位、描述和因果解释分别报告。
- 图片：[CUVA 论文中的异常视频及因果标注示例](<https://arxiv.org/html/2405.00181v3/dataset_4.png>)；署名：Hang Du et al. / CUVA
- 核验：[来源 1](<https://openaccess.thecvf.com/content/CVPR2024/html/Du_Uncovering_What_Why_and_How_A_Comprehensive_Benchmark_for_Causation_CVPR_2024_paper.html>) · [来源 2](<https://github.com/fesvhtr/CUVA>)

### HIVAU-70k

**2025 · CVPR** · [来源入口](<https://github.com/pipixin321/HolmesVAU>)

Holmes-VAU 提出的片段、事件和视频三级异常指令数据。

- 评估协议：区分时间粒度与任务类型；数据构建包含模型生成与人工复核。
- 图片：[HIVAU-70k 片段、事件和视频三级异常理解示例](<https://raw.githubusercontent.com/pipixin321/HolmesVAU/master/assets/teaser.png>)；署名：Huaxin Zhang et al. / Holmes-VAU
- 核验：[来源 1](<https://openaccess.thecvf.com/content/CVPR2025/html/Zhang_Holmes-VAU_Towards_Long-term_Video_Anomaly_Understanding_at_Any_Granularity_CVPR_2025_paper.html>) · [来源 2](<https://github.com/pipixin321/HolmesVAU>)

### HAWK

**2024 · NeurIPS** · [来源入口](<https://github.com/jqtangust/hawk>)

面向开放场景异常视频描述与交互问答的数据资源。

- 评估协议：遵循作者数据划分，分别检查描述生成与问答表现。
- 图片：[Hawk 作者发布的开放场景异常理解与问答示例](<https://raw.githubusercontent.com/jqtangust/hawk/main/figs/motivation1.png>)；署名：Jiaqi Tang et al. / Hawk
- 核验：[来源 1](<https://proceedings.neurips.cc/paper_files/paper/2024/hash/fca83589e85cb061631b7ebc5db5d6bd-Abstract-Conference.html>) · [来源 2](<https://github.com/jqtangust/hawk>)

### FineW3

**2026 · AAAI** · [来源入口](<https://finevau.github.io/>)

围绕 What、Who、Where 补充细粒度视觉事实的异常理解数据与评估。

- 评估协议：使用 FVScore 检查关键视觉元素，结合人类一致性分析；不能只比较语言流畅度。
- 图片：[FineVAU 论文中的细粒度异常描述和视觉要素对照](<https://arxiv.org/html/2601.17258v2/figs/Teaser.png>)；署名：João Pereira et al. / FineVAU
- 核验：[来源 1](<https://ojs.aaai.org/index.php/AAAI/article/download/37790/41752>) · [来源 2](<https://finevau.github.io/>)

### UCA

**2024 · CVPR** · [来源入口](<https://github.com/jia-wang-11/Surveillance-Video-Understanding-dataSet>)

在 UCF-Crime 监控视频上增加细粒度事件语句和时间边界，连接异常检测、语言定位与密集描述。

- 评估协议：使用作者训练、验证、测试划分；分别报告语言时序定位、视频描述、密集描述与多模态异常检测。
- 图片：[UCA 官方仓库展示的细粒度事件语句与对应时间段](<https://github.com/jia-wang-11/Surveillance-Video-Understanding-dataSet>)；署名：Tongtong Yuan et al. / UCA
- 核验：[来源 1](<https://openaccess.thecvf.com/content/CVPR2024/html/Yuan_Towards_Surveillance_Video-and-Language_Understanding_New_Dataset_Baselines_and_Challenges_CVPR_2024_paper.html>) · [来源 2](<https://github.com/jia-wang-11/Surveillance-Video-Understanding-dataSet>)

### ECVA

**2024 · arXiv** · [来源入口](<https://github.com/Dulpy/ECVA>)

CUVA 的扩展因果理解基准，围绕异常经过、发生原因与事件后果提供人工语言标注。

- 评估协议：按作者发布版本分别评估描述、原因和后果；AnomEval 检查推理、回答一致性与幻觉，避免与 CUVA 的 MMEval 混用。
- 图片：[ECVA 论文中异常因果理解的挑战与视频样例](<https://arxiv.org/html/2412.07183v1/challenge_v7.png>)；署名：Hang Du et al. / ECVA
- 核验：[来源 1](<https://arxiv.org/abs/2412.07183>) · [来源 2](<https://github.com/Dulpy/ECVA>) · [来源 3](<https://www.modelscope.cn/datasets/gouchenyi/ECVA/files>)

### Vad-Reasoning

**2025 · NeurIPS** · [来源入口](<https://github.com/wbfwonderful/Vad-R1>)

Vad-R1 提出的异常推理数据，使用感知到认知的结构化推理标注，并区分监督微调和强化学习子集。

- 评估协议：分别使用 Vad-Reasoning-SFT 的训练／测试划分与 Vad-Reasoning-RL；SFT 含推理文本，RL 仅有视频级弱标签。
- 图片：[Vad-Reasoning 官方仓库中的视频、推理过程与最终答案标注示例](<https://raw.githubusercontent.com/wbfwonderful/Vad-R1/main/images/data-example.png>)；署名：Chao Huang, Benfeng Wang et al. / Vad-R1
- 核验：[来源 1](<https://proceedings.neurips.cc/paper_files/paper/2025/hash/abccc325c84dedf23dbe8de3f686c733-Abstract-Conference.html>) · [来源 2](<https://github.com/wbfwonderful/Vad-R1>) · [来源 3](<https://huggingface.co/datasets/wbfwonderful/Vad-R1>)

### CueBench

**2026 · AAAI** · [来源入口](<https://ojs.aaai.org/index.php/AAAI/article/view/38209>)

以场景和属性组织条件性与绝对异常，检验异常判断的上下文依赖。

- 评估协议：分别报告识别、时序定位、检测和预判；按场景／属性分析条件性异常。当前登记依据论文，不据此宣称数据已开放下载。
- 核验：[来源 1](<https://ojs.aaai.org/index.php/AAAI/article/view/38209>)

### VAGU

**2026 · AAAI** · [来源入口](<https://ojs.aaai.org/index.php/AAAI/article/view/42412>)

联合视频异常时间定位与语义理解的基准。

- 评估协议：使用原版 VAGU 的问答与 JeAUG 联合评价，联合分数之外保留定位和理解单项结果；不混入扩展稿 VAGU-T。当前未核验数据下载状态。
- 核验：[来源 1](<https://arxiv.org/abs/2507.21507>)

### VALU

**2026 · ACL** · [来源入口](<https://aclanthology.org/2026.acl-long.56/>)

以五个语义层级组织异常事件边界与文本描述。

- 评估协议：按语义层级分别评估 temporal grounding、anomaly localization 和 detail discrimination。论文页写明将公开基准，本次未确认下载状态。
- 核验：[来源 1](<https://aclanthology.org/2026.acl-long.56/>)

### A2Seek

**2025 · NeurIPS Datasets and Benchmarks** · [来源入口](<https://2-mo.github.io/A2Seek/>)

面向动态航拍视角，连接异常事件、时空证据与因果解释。

- 评估协议：区分异常判断、区域定位和解释；遵循作者场景与分布外设置，不与固定监控结果直接混排。当前未核验下载状态。
- 核验：[来源 1](<https://proceedings.neurips.cc/paper_files/paper/2025/hash/de02de513503962e1d21035ab50ce661-Abstract-Datasets_and_Benchmarks_Track.html>)

### TAU-Bench

**2026 · arXiv** · [来源入口](<https://yarkupa.github.io/tau-bench.github.io/>)

以异常实例轨迹绑定分层语义，评估解释是否对应正确对象。

- 评估协议：联合检查实例跟踪和细粒度语义，不能以描述流畅度替代轨迹正确性；本次未确认数据／模型有效下载入口。
- 核验：[来源 1](<https://arxiv.org/abs/2608.05699>)

### Pistachio

**2026 · ECCV** · [来源入口](<https://pistachio-video.github.io>)

包含 Pistachio-VAD 和 Pistachio-VAU 两部分的可控合成视频基准，覆盖检测与事件语义理解。

- 评估协议：分别报告 VAD 与 VAU 协议；关注合成到真实的域差异及多事件子集。数据下载未在本次逐文件验证。
- 核验：[来源 1](<https://arxiv.org/abs/2511.19474>) · [来源 2](<https://arxiv.org/html/2511.19474v6>)

### VANE-Bench

**2025 · NAACL Findings** · [来源入口](<https://github.com/rohit901/VANE-Bench>)

通过问答评测合成视频不一致性和真实视频异常，覆盖检测与定位。

- 评估协议：分别检查合成与真实视频子集；问答正确率不直接等价于逐帧检测 AUC。作者提供代码和数据入口，本次未逐文件验证下载。
- 核验：[来源 1](<https://aclanthology.org/2025.findings-naacl.171/>)
