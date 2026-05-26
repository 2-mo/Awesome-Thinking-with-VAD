# CVPR

## Quick Navigation

- 2026: [Alert-CLIP](#-alert-clip-abnormality-aware-latent-enhanced-representation-tuning-of-clip-for-video-anomaly-detection)（表征调优）、 [LAVIDA](#-no-need-for-real-anomaly-mllm-empowered-zero-shot-video-anomaly-detection)（零样本）、 [D²MIL](#-learning-from-noisy-supervision-a-denoisingdebiasing-framework-for-weakly-supervised-video-anomaly-detection)（去噪去偏）、 [Fine-VAD](#-fine-vad-towards-fine-grained-video-anomaly-detection-via-progressive-cross-granularity-learning)（细粒度）、 [STCRC](#️-streaming-video-crime-anticipation-with-spatio-temporal-causal-reasoning)（犯罪预判）、 [Rare-E2E](#-rare-e2e-rare-events-dataset-for-end-to-end-driving-in-challenging-long-tail-scenarios)（长尾事件）
- 2025: [Just Dance with pi!](#-just-dance-with-pi-a-poly-modal-inductor-for-weakly-supervised-video-anomaly-detection)（多模态·弱监督）、 [Anomize](#-anomize-better-open-vocabulary-video-anomaly-detection)（开放词汇）、 [VERA](#-vera-explainable-video-anomaly-detection-via-verbalized-learning-of-vision-language-models)（语言化·解释）、 [TAO](#-track-any-anomalous-object-a-granular-video-anomaly-detection-pipeline)（细粒度跟踪）、 [Holmes-VAU](#-holmes-vau-towards-long-term-video-anomaly-understanding-at-any-granularity)（长时序理解）、 [Noise-Resistant VAD](#-noise-resistant-video-anomaly-detection-via-rgb-error-guided-multiscale-predictive-coding-and-dynamic-memory)（重建·抗噪）
- 2024: [MULDE](#-mulde-multiscale-log-density-estimation-via-denoising-score-matching-for-video-anomaly-detection)（多尺度密度）、 [CLAP](#-collaborative-learning-of-anomalies-with-privacy-clap-for-unsupervised-video-anomaly-detection-a-new-baseline)（联邦无监督）、 [Self-Distilled MAE](#-self-distilled-masked-auto-encoders-are-efficient-video-anomaly-detectors)（自蒸馏 MAE）、 [Open-Vocabulary VAD](#-open-vocabulary-video-anomaly-detection)（开放词汇）、 [Normality Prompt](#-text-prompt-with-normality-guidance-for-weakly-supervised-video-anomaly-detection)（文本正则）、 [Multi-Grained VAD](#-multi-scale-video-anomaly-detection-by-multi-grained-spatio-temporal-representation-learning)（多粒度时空）、 [PE-MIL](#-prompt-enhanced-multiple-instance-learning-for-weakly-supervised-video-anomaly-detection)（Prompt MIL）、 [LAVAD](#-harnessing-large-language-models-for-training-free-video-anomaly-detection)（LLM 免训练）、 [CUVA](#-uncovering-what-why-and-how-a-comprehensive-benchmark-for-causation-understanding-of-video-anomaly)（因果基准）
- 2023: [Pseudo Labels VAD](#-exploiting-completeness-and-uncertainty-of-pseudo-labels-for-weakly-supervised-video-anomaly-detection)（伪标签·弱监督）、 [Prompt Skeleton](#-prompt-guided-zero-shot-anomaly-action-recognition-using-pretrained-deep-skeleton-features)（骨架零样本）、 [Context-Motion](#-look-around-for-anomalies-weakly-supervised-anomaly-detection-via-context-motion-relational-learning)（上下文运动）、 [EVAL](#-eval-explainable-video-anomaly-localization)（可解释定位）、 [Keyframe Restoration](#-video-event-restoration-based-on-keyframes-for-video-anomaly-detection)（关键帧重建）、 [HSC-VAD](#-hierarchical-semantic-contrast-for-scene-aware-video-anomaly-detection)（场景对比）、 [UMIL](#-unbiased-multiple-instance-learning-for-weakly-supervised-video-anomaly-detection)（无偏 MIL）、 [Audio-Visual Forensics](#-self-supervised-video-forensics-by-audio-visual-anomaly-detection)（音视频自监督）、 [CampusVAD Benchmark](#-a-new-comprehensive-benchmark-for-semi-supervised-video-anomaly-detection-and-anticipation)（半监督基准）、 [Prompt-Based Generation](#-generating-anomalies-for-video-anomaly-detection-with-prompt-based-feature-mapping)（Prompt 生成）

## 2026
- Accepted papers: <https://cvpr.thecvf.com/Conferences/2026/AcceptedPapers>
- Observations:
  - 大模型/视觉语言模型在异常检测中的应用进一步深入，涵盖表征调优、纠偏去噪、零样本语义推理等。
  - 数据集与应用场景拓展延伸，涉及细粒度、交通事故、犯罪预判及长尾极端事件挑战。

### 基于大模型

#### 🧠 Joint Learning of General and Diverse Patterns with Mixture of Memory Experts for Weakly-Supervised Video Anomaly Detection
University of Chinese Academy of Sciences | `Memory Mixture` `LLM Prototype` | [[Paper]](https://cvpr.thecvf.com/virtual/2026/poster/36266)

> 提出 MoME 记忆专家混合框架，同时建模通用异常模式与多样异常模式。

#### 🛡️ Learning from Noisy Supervision: A Denoising–Debiasing Framework for Weakly Supervised Video Anomaly Detection
Nankai University | `Denoising-Debiasing` `VLM` | [[Paper]](https://cvpr.thecvf.com/virtual/2026/poster/38070)

> 提出 D²MIL，在 MIL 框架中先过滤疑似噪声样本，再用视觉语言模型辅助找回难识别异常片段。

#### 🚀 No Need For Real Anomaly: MLLM Empowered Zero-Shot Video Anomaly Detection
Beijing University of Posts and Telecommunications | `Zero-Shot` `MLLM` | [[ArXiv]](https://arxiv.org/pdf/2602.19248) [[Code]](https://github.com/VitaminCreed/LAVIDA)

> 提出 LAVIDA，仅使用伪异常训练，并结合 MLLM 语义理解实现零样本视频异常检测。

#### 🕵️ Streaming Video Crime Anticipation with Spatio-Temporal Causal Reasoning
TeleAI | `Crime Anticipation` `VLM` `Causal Reasoning` | [[Project]](https://zqsiat.github.io/) [[Paper]](https://cvpr.thecvf.com/virtual/2026/poster/39800)

> 构建 STCRC 数据集和 STCH 流式模块，用显式时空因果结构增强视频犯罪预判。


### 经典方案

#### 💬 Alert-CLIP: Abnormality-aware Latent-Enhanced Representation Tuning of CLIP for Video Anomaly Detection
Beijing University of Posts and Telecommunications | `CLIP Representation Tuning` `Multi-level Alignment` | [[Paper]](https://cvpr.thecvf.com/virtual/2026/poster/36334)

> 针对 CLIP 异常感知不足的问题，通过视频-标签、区域-文本、区域-语义多层对齐增强 VAD 表征。

#### 🎯 Fine-VAD: Towards Fine-Grained Video Anomaly Detection via Progressive Cross-Granularity Learning
Beijing University of Posts and Telecommunications | `Fine-Grained` `Cross-Granularity` | [[Paper]](https://cvpr.thecvf.com/virtual/2026/poster/38061)

> 利用粗粒度与细粒度标签的渐进跨粒度学习，同时提升异常检测与细粒度异常类别识别。

#### 🔍 Weakly Supervised Video Anomaly Detection with Anomaly-Connected Components and Intention Reasoning
Tongji University | `Intention Reasoning` `Weakly Supervised` | [[ArXiv]](https://arxiv.org/pdf/2603.00550)

> 提出 LAS-VAD，用异常连通组件学习异常语义，并通过意图推理区分相似的正常/异常行为。

#### 🧭 The Road Less Seen: Segment Exploration for Weakly Supervised Video Anomaly Detection
Rochester Institute of Technology | `Segment Exploration` `Weakly Supervised` | [[Paper]](https://cvpr.thecvf.com/virtual/2026/poster/39364)

> 指出 AUROC 在实际高风险场景下可能掩盖低召回问题，并用时序聚类与不确定性采样探索更多异常片段。

#### 🧩 TLMA: Mitigating the Impact of Weakly Labeled Information for Video Anomaly Detection
Beijing Jiaotong University | `Weak Label Mitigation` `Triplet Learning` | [[Paper]](https://cvpr.thecvf.com/virtual/2026/poster/38505)

> 通过三元组学习和运动感知特征增强，降低弱标签信息对异常定位边界学习的干扰。

#### 🚨 RiskProp: Collision-Anchored Self-supervised Temporal Constraints for Early Accident Anticipation
Wuhan University | `Accident Anticipation` `Self-supervised` | [[ArXiv]](https://arxiv.org/pdf/2603.27165)

> 只依赖碰撞帧标注，通过风险反向传播和单调时序约束实现更早、更平滑的事故风险预判。

#### 📊 Rare-E2E: Rare Events Dataset for End-to-End Driving in Challenging Long-tail Scenarios
University of California, Los Angeles | `Dataset` `Rare Events` `End-to-End Driving` | [[ArXiv]](https://arxiv.org/pdf/2510.26125)

> 提出面向端到端驾驶的长尾稀有事件数据集，并用 RFS 评价轨迹与人工偏好的一致性。

## 2025
- Accepted papers: <https://cvpr.thecvf.com/Conferences/2025/AcceptedPapers>
- Observations:
  - 多模态已成弱监督方案的标配，RGB 之外的几何与语言模态提供更稳健的异常证据。
  - 语言化与细粒度跟踪方向并行推进，强调可解释性与异常目标定位能力。

### 基于大模型

#### 💬 VERA: Explainable Video Anomaly Detection via Verbalized Learning of Vision-Language Models
University of Iowa | `Verbalized Learning` | [[Project]](https://vera-framework.github.io/) [[ArXiv]](https://arxiv.org/pdf/2412.01095) [[Code]](https://github.com/vera-framework/VERA)

> 将异常判决转化为"口头推理"任务，借由视觉-语言对齐让模型给出可读的解释与多模态证据。



#### 🧠 Holmes-VAU: Towards Long-term Video Anomaly Understanding at Any Granularity
Huazhong University of Science and Technology | `Long-term Understanding` `Multi-granularity` | [[ArXiv]](https://arxiv.org/abs/2412.06171) [[Code]](https://github.com/pipixin321/HolmesVAU)

> 针对长时序视频提出任意粒度理解框架，联合多模态知识与层次化语义，覆盖事件级、片段级与帧级异常，并提供语言化描述与可视化解释。

### 经典方案

#### 🎯 Just Dance with pi! A Poly-modal Inductor for Weakly-supervised Video Anomaly Detection
Inria | `Weakly-supervised` `Multi-modal Fusion` | [[ArXiv]](https://arxiv.org/abs/2505.13123) [[Code]](https://github.com/snehashismajhi/PI-VAD)

> 以多模态诱导器为核心，将姿态、深度、全景分割、光流与语言提示融入弱监督异常检测，在复杂人体行为场景显著抑制假阳性。

#### 🧭 Anomize: Better Open Vocabulary Video Anomaly Detection
Wuhan University | `Open Vocabulary` `Zero-shot` | [[ArXiv]](https://arxiv.org/abs/2503.18094)

> 将开放词汇检索引入视频异常检测，构建范畴词库与语义匹配模块，实现对"未见过"异常类别的零样本描述与匹配。

#### 🔍 Track Any Anomalous Object: A Granular Video Anomaly Detection Pipeline
Xiamen University | `Fine-grained Localization` `Object Tracking` | [[Project]](https://tao-25.github.io/) [[Paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Huang_Track_Any_Anomalous_ObjectA_Granular_Video_Anomaly_Detection_Pipeline_CVPR_2025_paper.pdf) [[Code]](https://github.com/yu2hi13/TAO)

> 面向工业与监控场景，结合检测、分割与跟踪模块对异常目标进行跨帧关联，提供实例级统计与可视化便于定位问题对象。

#### 🧩 Noise-Resistant Video Anomaly Detection via RGB Error-Guided Multiscale Predictive Coding and Dynamic Memory
East China University of Science and Technology | `Reconstruction` `Predictive Coding` | [[Paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Hu_Noise-Resistant_Video_Anomaly_Detection_via_RGB_Error-Guided_Multiscale_Predictive_Coding_CVPR_2025_paper.pdf)

> 以多尺度预测编码为骨架，引入动态记忆单元过滤噪声帧，降低监控抖动与光照变化带来的误判，强化低质视频中的异常响应。

## 2024
- Accepted papers: <https://cvpr.thecvf.com/Conferences/2024/AcceptedPapers>
- Observations:
  - 去监督化与隐私保护成为亮点，联邦协同与自蒸馏方法扩展无标签场景能力。
  - Prompt/文本引导和开放词汇检索凸显 VLM 思路的早期探索，奠定 2025 年语言化浪潮基础。

### 基于大模型

#### 🚀 LAVAD: Harnessing Large Language Models for Training-free Video Anomaly Detection
University of Trento | `Training-free` `Verbalized Detection` | [[Project]](https://lucazanella.github.io/lavad/) [[ArXiv]](https://arxiv.org/abs/2404.01014) [[Code]](https://github.com/lucazanella/lavad)

> 直接调用大模型语义知识，以 LLM/VLM Prompting 完成场景理解，将文本描述映射为异常判定，无需微调即可在缺乏标注的场景快速部署。

#### 🧩 CUVA: Uncovering What Why and How
Beijing University of Posts and Telecommunications | `Dataset` `Causal Understanding` | [[ArXiv]](https://arxiv.org/abs/2405.00181) [[Code]](https://github.com/fesvhtr/CUVA)

> 发布 CUVA 基准，构造 What/Why/How 三维题库，考察模型能否回答"发生了什么、为什么发生、如何处理"等因果问题，配套语言化问答标注。

### 经典方案

#### 📈 MULDE: Multiscale Log-Density Estimation via Denoising Score Matching for Video Anomaly Detection
Graz University of Technology | `Multi-scale Modeling` `Unsupervised` | [[ArXiv]](https://arxiv.org/pdf/2403.14497) [[Code]](https://github.com/jakubmicorek/MULDE-Multiscale-Log-Density-Estimation-via-Denoising-Score-Matching-for-Video-Anomaly-Detection)

> 以去噪得分匹配学习多尺度对数密度，通过多尺度窗口估计分布，兼顾短时爆发与长时渐变异常，无需显式标签直接提供异常评分。

#### 🤝 CLAP: Collaborative Learning of Anomalies with Privacy
MBZUAI | `Federated Learning` `Unsupervised` | [[Project]](https://anasemad11.github.io/CLAP/) [[Paper]](https://anasemad11.github.io/CLAP/static/images/2404.00847.pdf) [[Code]](https://github.com/AnasEmad11/CLAP)

> 提出联邦式异常学习框架，结合特征聚合与知识蒸馏，在不共享原始视频的前提下同步模型，为联邦 VAD 建立基线。

#### 🪞 Self-Distilled Masked Auto-Encoders are Efficient Video Anomaly Detectors
University of Bucharest | `Self-supervised` `Reconstruction` | [[ArXiv]](https://arxiv.org/abs/2306.12041) [[Code]](https://github.com/ristea/aed-mae/tree/main)

> 在 MAE 架构中引入自蒸馏，采用学生-教师结构强化对异常细节的感知，推理阶段仅需编码器 + 线性头，适合实时场景。

#### 🪟 Open-Vocabulary Video Anomaly Detection
Northwestern Polytechnical University | `Open Vocabulary` `Zero-shot` | [[ArXiv]](https://arxiv.org/abs/2311.07042)

> 借助开放词汇向量空间，构建语义映射使文本描述直接约束异常分类头，支持按场景、行为自由组合查询，扩展性强。

#### ✍️ Text Prompt with Normality Guidance for Weakly Supervised Video Anomaly Detection
Guangzhou Institute of Technology | `Text Guidance` `Weakly-supervised` | [[ArXiv]](https://arxiv.org/abs/2404.08531)

> 通过文本描述常态行为作为弱监督约束，设计 Normality Guidance Prompt 将语言先验转化为正则项，在缺乏帧级标注时保持定位精度。

#### 🧭 Multi-Scale Video Anomaly Detection by Multi-Grained Spatio-Temporal Representation Learning
Beijing University of Posts and Telecommunications | `Multi-scale Modeling` `Representation Learning` | [[Paper]](https://openaccess.thecvf.com/content/CVPR2024/papers/Zhang_Multi-Scale_Video_Anomaly_Detection_by_Multi-Grained_Spatio-Temporal_Representation_Learning_CVPR_2024_paper.pdf)

> 使用多粒度时空分支模块，通过层次特征聚合 + 时序注意力覆盖不同持续时间的异常事件，平衡局部与整体感知。

#### 🧾 PE-MIL: Prompt-Enhanced Multiple Instance Learning
University of Chinese Academy of Sciences | `Prompt-guided` `Weakly-supervised` | [[Paper]](https://openaccess.thecvf.com/content/CVPR2024/papers/Chen_Prompt-Enhanced_Multiple_Instance_Learning_for_Weakly_Supervised_Video_Anomaly_Detection_CVPR_2024_paper.pdf) [[Code]](https://github.com/Junxi-Chen/PE-MIL)

> 在经典 MIL 框架中注入 Prompt 信号，引导实例权重分配提高弱监督定位准确率，可作为插件模块与现有 MIL 模型兼容。

## 2023
- Accepted papers: <https://cvpr.thecvf.com/Conferences/2023/AcceptedPapers>
- Observations:
  - 弱监督与自监督策略丰富，围绕伪标签质量、MIL 偏差和音视频联合建模展开。
  - 生成与场景建模并行发展，强调异常模拟、场景语义与解释性工具链。

### 基于大模型

- 暂未收录 2023 年基于大模型的代表作，欢迎补充提案或相关链接。

### 经典方案

#### 🎯 Exploiting Completeness and Uncertainty of Pseudo Labels for Weakly Supervised Video Anomaly Detection
Institute of Information Engineering, CAS | `Weakly-supervised` `Pseudo-label Management` | [[ArXiv]](https://arxiv.org/abs/2212.04090)

> 从完整性与不确定性双角度评估伪标签质量，提出双分支估计伪标签可信度，抑制噪声标签带来的梯度偏差，增强高风险场景召回率。

#### 🦴 Prompt-Guided Zero-Shot Anomaly Action Recognition Using Pretrained Deep Skeleton Features
Konica Minolta | `Skeleton Action` `Zero-shot` | [[ArXiv]](https://arxiv.org/abs/2303.15167)

> 利用预训练骨架特征与 prompt，构建 skeleton-aware prompt 将语言先验映射到动作嵌入，在无监督条件下识别未标注的异常动作。

#### 👀 Look Around for Anomalies: Weakly-Supervised Anomaly Detection via Context-Motion Relational Learning
Yonsei University | `Weakly-supervised` `Relational Modeling` | [[Paper]](https://openaccess.thecvf.com/content/CVPR2023/papers/Cho_Look_Around_for_Anomalies_Weakly-Supervised_Anomaly_Detection_via_Context-Motion_Relational_CVPR_2023_paper.pdf)

> 通过上下文-运动关系网络，关系图模块整合场景背景与局部运动，减少孤立物体误检，适用于拥挤场景中的复杂交互。

#### 🪟 EVAL: Explainable Video Anomaly Localization
University of Massachusetts Amherst | `Explainable Localization` | [[ArXiv]](https://arxiv.org/abs/2212.07900) [[Code]](https://github.com/merlresearch/EVAL)

> 引入解释模块为异常定位结果生成关注区域与文字摘要，结合因果解释与注意力可视化，支持帧级和片段级的可追溯报告。

#### 🔄 Video Event Restoration Based on Keyframes for Video Anomaly Detection
Xidian University | `Reconstruction` `Keyframe-driven` | [[ArXiv]](https://arxiv.org/abs/2304.05112)

> 以关键帧驱动的事件恢复方法，选择代表性关键帧引导事件重建突出异常细节，改善传统重建法对快速运动的模糊问题。

#### 🗺️ HSC-VAD: Hierarchical Semantic Contrast for Scene-Aware Video Anomaly Detection
Zhejiang University | `Scene-aware` `Contrastive Learning` | [[ArXiv]](https://arxiv.org/abs/2303.13051) [[Code]](https://github.com/shengyangsun/HSC_VAD)

> 通过层次语义对比，设计场景级、物体级双层对比损失提升跨场景泛化，与常规 backbone 结合即可获得增益。

#### 📦 UMIL: Unbiased Multiple Instance Learning
National University of Singapore | `Weakly-supervised` `MIL Correction` | [[ArXiv]](https://arxiv.org/abs/2303.12369) [[Code]](https://github.com/ktr-hubrt/UMIL)

> 针对弱监督 MIL 框架的选择偏差，理论分析 MIL 偏差来源并给出校正损失，大幅提升异常时间段的定位准确率。

#### 🎶 Self-Supervised Video Forensics by Audio-Visual Anomaly Detection
University of Michigan | `Self-supervised` `Audio-visual Alignment` | [[Project]](https://cfeng16.github.io/audio-visual-forensics/) [[ArXiv]](https://arxiv.org/abs/2301.01767) [[Code]](https://github.com/cfeng16/audio-visual-forensics)

> 将音频与视频跨模态对齐，构建自监督对比任务学习音视频一致性，用于检测视频伪造与异常事件，可应用于取证与多媒体安全。

#### 🧪 CampusVAD: A New Comprehensive Benchmark for Semi-Supervised Video Anomaly Detection and Anticipation
Northwestern Polytechnical University | `Dataset` `Anticipation` | [[Project]](https://campusvad.github.io/) [[ArXiv]](http://arxiv.org/abs/2305.13611) [[Code]](https://github.com/zugexiaodui/campus_vad_code)

> CampusVAD 数据集覆盖检测与异常预警任务，提供多场景真实监控数据含半监督与预测标签，建立标准评测协议促进模型早期预判能力。

#### 🧠 Generating Anomalies for Video Anomaly Detection With Prompt-Based Feature Mapping
Sun Yat-sen University | `Anomaly Generation` `Prompt Control` | [[Paper]](https://openaccess.thecvf.com/content/CVPR2023/papers/Liu_Generating_Anomalies_for_Video_Anomaly_Detection_With_Prompt-Based_Feature_Mapping_CVPR_2023_paper.pdf)

> 使用 Prompt 引导的特征映射生成合成异常，将文本 prompt 转化为特征扰动生成多样异常案例，支持与现有检测器联合训练提升泛化。
