# Video Anomaly Understanding — 精选核验目录

这是从结构化数据生成的精选核验目录，并非完整文献综述。原仓库的会议笔记作为额外资料保留，尚未全面复核，不应视作本目录的核验条目。

数据更新时间：2026-09-29。核验来源与日期、数据集、证据关系和阅读路线请见 [data/catalog.json](data/catalog.json) 与交互式研究地图。

> 自动生成：请编辑 `data/catalog.json` 后运行 `npm run generate`，不要手工修改此文件。

## Core research / 核心研究

## 语义对齐与融合

创新抓手：字幕特征融合、视觉语言对齐和运动语言监督，为异常建立可解释的语义表征。

研究问题：如何把外观、运动与语言转化为异常相关的表征？

### TEVAD: Improved Video Anomaly Detection With Captions

**2023 · CVPR Workshops** · [paper](<https://openaccess.thecvf.com/content/CVPR2023W/O-DRUM/papers/Chen_TEVAD_Improved_Video_Anomaly_Detection_With_Captions_CVPRW_2023_paper.pdf>)

**创新抓手：字幕语义特征融合**

融合视频特征与字幕语义，补充纯视觉检测缺少的事件信息。

- 核心启示：先理解语言如何进入检测器，再讨论更复杂的推理。
- 局限：阅读关注：字幕中的错误会怎样影响异常分数？
- 核验：2026-09-29；[来源 1](<https://openaccess.thecvf.com/content/CVPR2023W/O-DRUM/papers/Chen_TEVAD_Improved_Video_Anomaly_Detection_With_Captions_CVPRW_2023_paper.pdf>)

### VadCLIP: Adapting Vision-Language Models for Weakly Supervised Video Anomaly Detection

**2024 · AAAI** · [paper](<https://arxiv.org/abs/2308.11681>) · [code](<https://github.com/nwpu-zxr/VadCLIP>)

**创新抓手：视觉语言双分支对齐**

利用冻结 CLIP 的视觉语言关联，通过双分支完成粗粒度和细粒度异常检测。

- 核心启示：语义对齐让检测分数能够关联异常类别。
- 局限：阅读关注：类别语义对齐与开放式异常解释仍有距离。
- 核验：2026-09-29；[来源 1](<https://arxiv.org/abs/2308.11681>) · [来源 2](<https://ojs.aaai.org/index.php/AAAI/article/view/28423>)

### Ex-VAD: Explainable Fine-grained Video Anomaly Detection Based on Visual-Language Models

**2025 · ICML** · [paper](<https://proceedings.mlr.press/v267/huang25ad.html>)

**创新抓手：解释融合与标签对齐**

由帧字幕生成视频级异常解释，再结合视觉特征与标签增强对齐进行细粒度检测。

- 核心启示：解释文本既是输出，也能参与检测表征学习。
- 局限：阅读关注：更好的分类结果是否同时意味着更忠实的解释？
- 核验：2026-09-29；[来源 1](<https://proceedings.mlr.press/v267/huang25ad.html>)

### Hawk: Learning to Understand Open-World Video Anomalies

**2024 · NeurIPS** · [paper](<https://arxiv.org/abs/2405.16886>) · [code](<https://github.com/jqtangust/hawk>)

**创新抓手：运动语言监督对齐**

显式引入运动信息，并用异常视频描述与问答数据训练开放场景理解能力。

- 核心启示：异常理解不仅需要外观，也需要动作变化与交互。
- 局限：阅读关注：开放场景能力如何随数据来源和问题类型变化？
- 核验：2026-09-29；[来源 1](<https://arxiv.org/abs/2405.16886>) · [来源 2](<https://github.com/jqtangust/hawk/blob/main/README.md>)

## 语言判据与提示优化

创新抓手：字幕到判断的语言中介、正常规则归纳与引导问题优化，将异常标准显式化。

研究问题：有了语义表征，怎样构造场景适用的异常判据？

### Harnessing Large Language Models for Training-free Video Anomaly Detection

**2024 · CVPR** · [paper](<https://arxiv.org/abs/2404.01014>) · [code](<https://github.com/lucazanella/lavad>) · [project](<https://lucazanella.github.io/lavad/>)

**创新抓手：字幕聚合语言评分**

先描述帧内容，再由语言模型聚合时序信息并估计异常分数。

- 核心启示：把预训练模型组合成无需目标数据训练的异常检测流程。
- 局限：阅读关注：描述噪声及文本压缩会丢失哪些视觉细节？
- 核验：2026-09-29；[来源 1](<https://arxiv.org/abs/2404.01014>) · [来源 2](<https://openaccess.thecvf.com/content/CVPR2024/papers/Zanella_Harnessing_Large_Language_Models_for_Training-free_Video_Anomaly_Detection_CVPR_2024_paper.pdf>)

### Follow the Rules: Reasoning for Video Anomaly Detection with Large Language Models

**2024 · ECCV** · [paper](<https://arxiv.org/abs/2407.10299>) · [code](<https://github.com/Yuchen413/AnomalyRuler>)

**创新抓手：正常规则归纳演绎**

从少量正常样本归纳场景规则，再依据规则判断测试视频中的异常。

- 核心启示：正常性可以写成可检查、可调整的语言规则。
- 局限：阅读关注：少量正常参考是否覆盖场景中的合理变化？
- 核验：2026-09-29；[来源 1](<https://arxiv.org/abs/2407.10299>) · [来源 2](<https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/10568.pdf>)

### VERA: Explainable Video Anomaly Detection via Verbalized Learning of Vision-Language Models

**2025 · CVPR** · [paper](<https://arxiv.org/abs/2412.01095>)

**创新抓手：语言反馈优化问题**

用视频级弱标签优化自然语言引导问题，让冻结 VLM 分解并判断异常模式。

- 核心启示：不更新模型权重，也可以学习任务专属的推理提示。
- 局限：阅读关注：语言问题的适应性与跨场景迁移需要分开验证。
- 核验：2026-09-29；[来源 1](<https://arxiv.org/abs/2412.01095>) · [来源 2](<https://openaccess.thecvf.com/content/CVPR2025/papers/Ye_VERA_Explainable_Video_Anomaly_Detection_via_Verbalized_Learning_of_Vision-Language_CVPR_2025_paper.pdf>)

## 时序分层与记忆

创新抓手：多粒度指令、事件边界建模、历史分数记忆，以及语义层级下的边界评估。

研究问题：判据应作用于哪一段时间，又需要保留多少历史？

### Holmes-VAU: Towards Long-term Video Anomaly Understanding at Any Granularity

**2025 · CVPR** · [paper](<https://arxiv.org/abs/2412.06171>) · [code](<https://github.com/pipixin321/HolmesVAU>)

**创新抓手：多粒度指令与采样**

构建多粒度异常指令数据，并用异常聚焦采样连接片段、事件和整段视频的理解。

- 核心启示：长视频 VAU 要同时处理采样与多粒度语义。
- 局限：阅读关注：聚焦异常的采样是否保留充分的前因后果？
- 核验：2026-09-29；[来源 1](<https://arxiv.org/abs/2412.06171>) · [来源 2](<https://openaccess.thecvf.com/content/CVPR2025/papers/Zhang_Holmes-VAU_Towards_Long-term_Video_Anomaly_Understanding_at_Any_Granularity_CVPR_2025_paper.pdf>)

### EventVAD: Training-Free Event-Aware Video Anomaly Detection

**2025 · ACM MM** · [paper](<https://arxiv.org/abs/2504.13092>) · [code](<https://github.com/YihuaJerry/EventVAD>)

**创新抓手：时空图划分事件边界**

利用动态时空图寻找事件边界，再通过层级提示完成事件级异常推理。

- 核心启示：先确定事件单元，能让长视频推理减少无关上下文。
- 局限：阅读关注：边界划分错误如何影响后续异常判断？
- 核验：2026-09-29；[来源 1](<https://arxiv.org/abs/2504.13092>)

### MoniTor: Exploiting Large Language Models with Instruction for Online Video Anomaly Detection

**2025 · NeurIPS** · [paper](<https://arxiv.org/abs/2510.21449>)

**创新抓手：流式记忆与分数队列**

使用流式输入、历史预测记忆和分数队列，在不训练的条件下持续判断异常。

- 核心启示：在线异常理解必须明确当前可见的信息范围。
- 局限：阅读关注：在线约束、延迟与历史错误累积的影响。
- 核验：2026-09-29；[来源 1](<https://arxiv.org/abs/2510.21449>)

### VALU: A Benchmark for Video Anomaly Temporal Localization and Understanding at Multiple Semantic Levels

**2026 · ACL** · [paper](<https://aclanthology.org/2026.acl-long.56/>)

**创新抓手：语义分层边界评估**

用多个语义层级定义异常边界，联合考查定位、时间 grounding 与细节辨别。

- 核心启示：异常的起止时间取决于所采用的语义范围。
- 局限：阅读关注：不同标注边界定义是否让结果失去可比性？
- 核验：2026-09-29；[来源 1](<https://aclanthology.org/2026.acl-long.56/>)

## 主动观察与工具决策

创新抓手：场景规划、工具调用、补充采样与交互策略学习，让异常证据获取随当前疑点调整。

研究问题：当前证据不足时，怎样决定下一次观察或工具调用？

### PANDA: Towards Generalist Video Anomaly Detection via Agentic AI Engineer

**2025 · NeurIPS** · [paper](<https://arxiv.org/abs/2509.26386>) · [project](<https://github.com/showlab/PANDA>)

**创新抓手：场景规划与工具反思**

把场景规划、工具反思与经验记忆组合为通用异常检测智能体。

- 核心启示：检测流程本身可以成为按场景调整的决策过程。
- 局限：阅读关注：工具调用、记忆和检测质量之间的成本收益。
- 核验：2026-09-29；[来源 1](<https://arxiv.org/abs/2509.26386>) · [来源 2](<https://github.com/showlab/PANDA>)

### Learning to Watch: Active Video Anomaly Understanding via Interleaved Policy Optimization

**2026 · arXiv** · [paper](<https://arxiv.org/abs/2607.00622>)

**创新抓手：交替推理与观察策略**

将推理与时间回溯、区间扩展、细粒度采样交替执行，学习主动获取证据的策略。

- 核心启示：“下一步观察什么”也可以成为 VAU 的学习对象。
- 局限：阅读关注：证据收益、任务成功和交互成本是否被共同衡量？
- 核验：2026-09-29；[来源 1](<https://arxiv.org/abs/2607.00622>)

## 结构化推理与验证

创新抓手：因果问题分解、对象关系编码、反思修正与关键事实评估，让异常解释可以被检查。

研究问题：怎样组织推理，并检验结论是否抓住关键事实？

### Uncovering What, Why and How: A Comprehensive Benchmark for Causation Understanding of Video Anomaly

**2024 · CVPR** · [paper](<https://arxiv.org/abs/2405.00181>) · [code](<https://github.com/fesvhtr/CUVA>)

**创新抓手：事件因果任务分解**

用事件、原因和后果标注，将视频异常理解拓展到因果解释及评估。

- 核心启示：读 VAU 时，应把“发生什么”和“为什么发生”分开检查。
- 局限：阅读关注：可见事件证据能支持多强的因果结论？
- 核验：2026-09-29；[来源 1](<https://arxiv.org/abs/2405.00181>)

### VADER: Towards Causal Video Anomaly Understanding with Relation-Aware Large Language Models

**2026 · WACV** · [paper](<https://arxiv.org/abs/2511.07299>)

**创新抓手：对象关系编码推理**

融合上下文采样与对象关系特征，支持异常描述、解释和因果问答。

- 核心启示：将对象交互显式接入模型，连接可见关系与语言解释。
- 局限：阅读关注：关系特征提供的是因果证据，还是相关性线索？
- 核验：2026-09-29；[来源 1](<https://arxiv.org/abs/2511.07299>) · [来源 2](<https://www.cs.nthu.edu.tw/~lai/pdf/publications/2025/VADER_Towards_Causal_Video_Anomaly_Understanding_with_Relation-Aware_Large_Language_Models.pdf>)

### SRVAU-R1: Enhancing Video Anomaly Understanding via Reflection-Aware Learning

**2026 · arXiv** · [paper](<https://arxiv.org/abs/2602.01004>)

**创新抓手：反思修正序列训练**

构建初始推理、反思和修正推理的监督序列，再结合监督与强化微调。

- 核心启示：把自我修正纳入学习目标，而不只要求更长的推理文本。
- 局限：阅读关注：反思是否真的纠正了视觉证据不足的判断？
- 核验：2026-09-29；[来源 1](<https://arxiv.org/abs/2602.01004>)

### FineVAU: A Novel Human-Aligned Benchmark for Fine-Grained Video Anomaly Understanding

**2026 · AAAI** · [paper](<https://arxiv.org/abs/2601.17258>) · [project](<https://finevau.github.io/>)

**创新抓手：关键视觉事实评估**

围绕事件、参与者与位置构建细粒度标注，并用 FVScore 检查关键视觉信息。

- 核心启示：流畅的描述不等于抓住异常；评估需要追问关键事实。
- 局限：阅读关注：自动扩展标注的质量及其与人类判断的一致性。
- 核验：2026-09-29；[来源 1](<https://arxiv.org/abs/2601.17258>)
