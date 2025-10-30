# Awesome LLM4VAD

A curated list of papers and resources on Large Language Models for Video Anomaly Detection (VAD).


## Contents

- [Overview](#overview)
- [Papers by Year](#papers-by-year)
  - [2025](#papers-2025)
    - [NeurIPS 2025](#neurips-2025)
    - [ICML 2025](#icml-2025)
    - [ICCV 2025](#iccv-2025)
    - [ACM MM 2025](#acm-mm-2025)
    - [CVPR 2025](#cvpr-2025)
    - [arXiv 2025](#arxiv-2025-preprints)
  - [2024](#papers-2024)
    - [AAAI 2024](#aaai-2024)
    - [CVPR 2024](#cvpr-2024)
    - [ECCV 2024](#eccv-2024)
    - [ICCV 2024](#iccv-2024)
    - [NeurIPS 2024](#neurips-2024)
    - [ACM MM 2024](#acm-mm-2024)
    - [arXiv 2024](#arxiv-2024-preprints)
- [Related Awesome Lists](#related-awesome-lists)

---

## Overview

This list collects representative works that leverage LLMs or vision-language models for video anomaly detection, explanation, and understanding. Entries are grouped by year with links to paper and code, plus a preview figure when available.

上下文依赖（复杂性）：异常往往是长时序事件（打斗、事故），需要结合前后因果与场景关系才能正确判定。

歧义混淆（模糊性）：局部行为或场景容易与异常混淆（奔跑 vs 逃跑、聚集 vs 暴乱），必须通过更长时序和多模态线索来消解。

长尾分布（稀疏性）：异常在视频流中出现频率极低、时机不可控，单次观测易漏检，必须跨时累积证据与假设检验。



#### 其实“思考”并不是只在异常场景里才需要，而是在异常问题上，它的必要性被放大：

常态模式容易靠感知解决：正常行为/场景占据绝大多数，规律性强、数据量大，单靠感知模式匹配就能达到不错的效果。

异常本质上是“不确定”：异常往往稀疏、少样本，缺乏先验统计支撑。仅靠快速感知会出现偏差，需要跨时整合和假设检验来弥补。

异常涉及更大风险：一旦误判，可能带来严重后果（漏报安全事件、误报干扰系统），因此必须引入更慢、更稳健的决策机制。

异常往往打破常规：它们可能表现为复杂的上下文依赖、模糊的语义混淆、长尾的稀疏分布——这些都恰好是“思考”擅长处理的。


我们需要的是推理，而不仅是事后解释。




### Curiosity-driven Learning

Humans monitor learning progress in curiosity-driven exploration (Nature Communications 2021) [[paper](https://www.nature.com/articles/s41467-021-26196-w)]
发现人类在探索中会“盯着学习进度”本身：更偏好能带来更大知识增益/误差下降率的选择。行为与模型支持“以学习进步为回报”的好奇心机制

Curiosity-driven Exploration by Self-supervised Prediction (ICML 2017 (PMLR v70)) [[paper](https://proceedings.mlr.press/v70/pathak17a/pathak17a.pdf)]

Computational mechanisms of curiosity and goal-directed exploration (Neuroscience 2019) [[paper](https://elifesciences.org/articles/41703)]



---

## 📊 Benchmarks and Datasets




数据集：Driving Anomaly Detection Honda Research Institute
<https://usa.honda-ri.com/hdd#Videos>

NWPU-Campus
Ubnormal
TAD
X-Man
XD-Violence

shanghaitech-anomaly-detection [[project](https://svip-lab.github.io/dataset/campus_dataset.html)] — Campus surveillance anomaly set; classic weakly supervised benchmark.

[UCF-Crime](https://www.crcv.ucf.edu/research/real-world-anomaly-detection-in-surveillance-videos/) — Real-world surveillance anomaly dataset with long untrimmed videos.

Multi-Scenario Anomaly Detection (MSAD) Dataset (NeurIPS 2024) [![Project](https://img.shields.io/badge/Project-blue?logo=safari)](https://msad-dataset.github.io/) [![arXiv](https://img.shields.io/badge/arXiv-2402.04857-b31b1b?logo=arxiv)](https://arxiv.org/pdf/2402.04857) — Large-scale, multi-scene anomaly benchmark.


https://video-holmes.github.io/Page.github.io/



<https://github.com/okankop/Driver-Anomaly-Detection>

https://www.cs.cmu.edu/~roadwork/ (ICCV 2025)



### Metrics & Evaluation

- Coming soon: common tasks, metrics, and evaluation protocols.

---

Aha! - Predicting What Matters Next: Online Highlight Detection Without Looking Ahead
https://neurips.cc/virtual/2025/poster/119707



## Papers (2025)

### NeurIPS 2025

#### PANDA: Towards Generalist Video Anomaly Detection via Detective-like Agent

[![NeurIPS](https://img.shields.io/badge/NeurIPS-2025-2DB55D)](https://neurips.cc/virtual/2025/poster/115891)

> 提出侦探式 Agent 范式的通用视频异常检测框架，通过工具调用与多步推理实现跨场景、跨类别的异常检测泛化能力。

---

#### MoniTor: Exploiting Large Language Models with Instruction for Online Video Anomaly Detection

[![NeurIPS](https://img.shields.io/badge/NeurIPS-2025-2DB55D)](https://neurips.cc/virtual/2025/poster/119803)

> 利用大语言模型与指令驱动机制实现在线视频异常检测，通过流式推理与实时反馈提升在线场景下的检测响应速度与准确性。

---

### ICML 2025

#### Ex-VAD: Explainable Fine-grained Video Anomaly Detection Based on Visual-Language Models

[![ICML](https://img.shields.io/badge/ICML-2025-FF6B6B)](https://openreview.net/forum?id=xAhUoyb5eU)
[![Paper](https://img.shields.io/badge/Paper-PDF-blue)](https://raw.githubusercontent.com/mlresearch/v267/main/assets/huang25ad/huang25ad.pdf)

> 基于视觉语言模型的可解释细粒度视频异常检测方法，利用 VLM 的语义理解能力提供异常的细粒度解释，增强模型决策透明度。

---

### ICCV 2025

#### Aligning Effective Tokens with Video Anomaly in Large Language Models

[![ICCV](https://img.shields.io/badge/ICCV-2025-00CED1)](https://arxiv.org/pdf/2508.06350)

> 将视频异常检测任务映射为大语言模型的 token 对齐问题，通过有效 token 选择机制实现多模态大模型驱动的视频异常理解。

---

#### Beyond Pixel Uncertainty: Bounding the OoD Objects in Road Scenes

[![ICCV](https://img.shields.io/badge/ICCV-2025-00CED1)]()

> 超越像素级不确定性，在道路场景中定位分布外目标，为自动驾驶中的异常检测提供更精准的边界框级检测。



### ACM MM 2025

#### EventVAD: Training-Free Event-Aware Video Anomaly Detection

[![ACM MM](https://img.shields.io/badge/ACM_MM-2025-FF69B4)](https://arxiv.org/abs/2504.13092)
[![Code](https://img.shields.io/github/stars/YihuaJerry/EventVAD?style=social&label=Code&logo=github)](https://github.com/YihuaJerry/EventVAD)

> 基于 Video-LLaMA2 的免训练事件感知视频异常检测方法，通过零样本事件理解与时序推理，无需模型训练即可实现跨场景异常判别。

![EventVAD preview](./assets/eventvad-acmmm25.png)

---

#### SAGE: A Visual Language Model for Anomaly Detection via Fact Enhancement and Entropy-aware Alignment

[![ACM MM](https://img.shields.io/badge/ACM_MM-2025-FF69B4)]()

> 通过事实增强与熵感知对齐机制构建视觉语言异常检测模型，提升模型对异常细节的感知与判别能力。

---

### CVPR 2025

#### VERA: Explainable Video Anomaly Detection via Verbalized Learning of Vision-Language Models

[![CVPR](https://img.shields.io/badge/CVPR-2025-1E90FF)](https://openaccess.thecvf.com/content/CVPR2025/papers/Ye_VERA_Explainable_Video_Anomaly_Detection_via_Verbalized_Learning_of_Vision-Language_CVPR_2025_paper.pdf)
[![Code](https://img.shields.io/github/stars/vera-framework/VERA?style=social&label=Code&logo=github)](https://github.com/vera-framework/VERA)

> 将异常判决转化为"口头推理"任务，借由视觉-语言对齐让模型给出可读的解释与多模态证据。

![VERA preview](./assets/2025-cvpr-vera.png)

---

#### Holmes-VAU: Towards Long-term Video Anomaly Understanding at Any Granularity

[![CVPR](https://img.shields.io/badge/CVPR-2025-1E90FF)](https://openaccess.thecvf.com/content/CVPR2025/papers/Zhang_Holmes-VAU_Towards_Long-term_Video_Anomaly_Understanding_at_Any_Granularity_CVPR_2025_paper.pdf)
[![Code](https://img.shields.io/github/stars/pipixin321/HolmesVAU?style=social&label=Code&logo=github)](https://github.com/pipixin321/HolmesVAU)

> 针对长时序视频提出任意粒度理解框架，联合多模态知识与层次化语义，覆盖事件级、片段级与帧级异常，并提供语言化描述。

![Holmes-VAU preview](./assets/2025-cvpr-holmes-vau.png)

---

### arXiv 2025 (Preprints)

#### AVadCLIP: Audio-Visual Collaboration for Robust Video Anomaly Detection

[![arXiv](https://img.shields.io/badge/arXiv-2025-b31b1b?logo=arxiv)]()

> 扩展 VadCLIP 至音视频多模态协同，通过音频线索辅助视觉检测，提升复杂场景下的异常判别鲁棒性。

---

#### AssistPDA: An Online Video Surveillance Assistant for Video Anomaly Prediction

[![arXiv](https://img.shields.io/badge/arXiv-2025-b31b1b?logo=arxiv)]()

> 在线视频监控助手系统，利用大模型实现异常预测与实时反馈，辅助监控人员决策。

---

#### SlowFastVAD: Video Anomaly Detection via Integrating Simple Detector and RAG-Enhanced Vision-Language Model

[![arXiv](https://img.shields.io/badge/arXiv-2025-b31b1b?logo=arxiv)]()

> 结合 SlowFast 双流检测器与 RAG 增强的视觉语言模型，通过检索增强生成机制提升异常理解与定位能力。

---

#### Vad-R1: Towards Video Anomaly Reasoning via Perception-to-Cognition Chain-of-Thought

[![arXiv](https://img.shields.io/badge/arXiv-2025-b31b1b?logo=arxiv)]()

> 提出感知到认知的思维链推理框架，通过多步推理链条实现从视觉感知到异常判断的端到端推理。

---

#### Flashback: Memory-Driven Zero-shot, Real-time Video Anomaly Detection

[![arXiv](https://img.shields.io/badge/arXiv-2025-b31b1b?logo=arxiv)]()

> 基于记忆驱动的零样本实时视频异常检测方法，通过动态记忆库实现无训练场景下的快速异常响应。

---

#### Simplifying Traffic Anomaly Detection with Video Foundation Models

[![arXiv](https://img.shields.io/badge/arXiv-2025-b31b1b?logo=arxiv)]()

> 利用视频基础模型简化交通异常检测流程，通过预训练模型的迁移能力降低任务特定数据需求。

---

#### NexViTAD: Few-shot Unsupervised Cross-Domain Defect Detection via Vision Foundation Models and Multi-Task Learning

[![arXiv](https://img.shields.io/badge/arXiv-2025-b31b1b?logo=arxiv)]()

> 基于视觉基础模型与多任务学习的少样本无监督跨域缺陷检测方法，适用于工业异常检测场景。

---

#### AnomalyMoE: Towards a Language-free Generalist Model for Unified Visual Anomaly Detection

[![arXiv](https://img.shields.io/badge/arXiv-2025-b31b1b?logo=arxiv)]()

> 提出无语言依赖的专家混合（MoE）统一视觉异常检测模型，通过多专家协同实现跨域泛化能力。

---

#### Unlocking Vision-Language Models for Video Anomaly Detection via Fine-Grained Prompting

[![arXiv](https://img.shields.io/badge/arXiv-2025-b31b1b?logo=arxiv)]()

> 通过细粒度提示解锁视觉语言模型在视频异常检测中的潜力，设计任务特定的 prompt 模板提升检测性能。

---

#### VAU-R1: Advancing Video Anomaly Understanding via Reinforcement Fine-Tuning

[![arXiv](https://img.shields.io/badge/arXiv-2025-b31b1b?logo=arxiv)]()

> 利用强化学习微调视频异常理解模型，通过奖励信号引导模型学习异常判别策略，提升理解深度。

---

#### Language-guided Open-world Video Anomaly Detection

[![arXiv](https://img.shields.io/badge/arXiv-2025-b31b1b?logo=arxiv)]()

> 语言引导的开放世界视频异常检测方法，通过自然语言描述实现对未见异常类别的零样本检测。

---

## Papers (2024)

### AAAI 2024

#### VadCLIP: Adapting Vision-Language Models for Weakly Supervised Video Anomaly Detection

[![AAAI](https://img.shields.io/badge/AAAI-2024-1F77B4)](https://ojs.aaai.org/index.php/AAAI/article/view/28423)
[![arXiv](https://img.shields.io/badge/arXiv-2308.11681-b31b1b?logo=arxiv)](https://arxiv.org/abs/2308.11681)
[![Code](https://img.shields.io/github/stars/nwpu-zxr/VadCLIP?style=social&label=Code&logo=github)](https://github.com/nwpu-zxr/VadCLIP)

> 将 CLIP 等视觉-语言模型适配至弱监督视频异常检测任务，通过跨模态对齐与视频时序建模，在有限标注下实现高效异常判别。

![VadCLIP preview](./assets/2024-aaai-vadclip.png)

---

### CVPR 2024

#### Harnessing Large Language Models for Training-free Video Anomaly Detection (LAVAD)

[![CVPR](https://img.shields.io/badge/CVPR-2024-1E90FF)](https://openaccess.thecvf.com/content/CVPR2024/papers/Zanella_Harnessing_Large_Language_Models_for_Training-free_Video_Anomaly_Detection_CVPR_2024_paper.pdf)
[![Code](https://img.shields.io/github/stars/lucazanella/lavad?style=social&label=Code&logo=github)](https://github.com/lucazanella/lavad)

> 直接调用大模型语义知识，以 LLM/VLM Prompting 完成场景理解，无需微调即可在缺乏标注的场景快速部署。

![Training-free VAD preview](./assets/2024-cvpr-training-free-vad.png)

---

#### Text Prompt with Normality Guidance for Weakly Supervised Video Anomaly Detection

[![CVPR](https://img.shields.io/badge/CVPR-2024-1E90FF)](https://arxiv.org/abs/2404.08531)

> 通过文本描述常态行为作为弱监督约束，设计 Normality Guidance Prompt 将语言先验转化为正则项，在缺乏帧级标注时保持定位精度。

---

### ECCV 2024

#### Follow the Rules: Reasoning for Video Anomaly Detection with Large Language Models (AnomalyRuler)

[![ECCV](https://img.shields.io/badge/ECCV-2024-0B84FE)](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/10568.pdf)
[![Code](https://img.shields.io/github/stars/Yuchen413/AnomalyRuler?style=social&label=Code&logo=github)](https://github.com/Yuchen413/AnomalyRuler)

> 通过两阶段推理机制（归纳与演绎）利用 LLM 从少量正常样本中总结正常规律，再据此检测异常帧，结合规则聚合与感知平滑提升推理稳健性。

![AnomalyRuler preview](./assets/2024-eccv-anomalyruler.png)

---

### ICCV 2024

#### Video Anomaly Detection and Explanation via Large Language Models

[![arXiv](https://img.shields.io/badge/arXiv-2401.05702-b31b1b?logo=arxiv)](https://arxiv.org/pdf/2401.05702v1)

> 将 VAD 与 LLM 生成的解释耦合，提供可解释的文本化理由，增强模型决策透明度。

![LLM VAD + Explanation preview](./assets/2024-arxiv-vad-llm-explanation.png)

---

### NeurIPS 2024

#### HAWK: Learning to Understand Open-World Video Anomalies

[![NeurIPS](https://img.shields.io/badge/NeurIPS-2024-2DB55D)](https://proceedings.neurips.cc/paper_files/paper/2024/file/fca83589e85cb061631b7ebc5db5d6bd-Paper-Conference.pdf)
[![arXiv](https://img.shields.io/badge/arXiv-2405.16886-b31b1b?logo=arxiv)](https://arxiv.org/pdf/2405.16886)
[![Code](https://img.shields.io/github/stars/jqtangust/hawk?style=social&label=Code&logo=github)](https://github.com/jqtangust/hawk)

> 利用视觉语言模型理解开放世界视频异常，引入运动模态增强 VLM 对动态异常的感知能力，突破传统封闭集检测限制。

![HAWK preview](./assets/2024-neurips-hawk.png)

---

#### MDVAD: Towards Multi-Domain Learning for Generalizable Video Anomaly Detection

[![NeurIPS](https://img.shields.io/badge/NeurIPS-2024-2DB55D)](https://proceedings.neurips.cc/paper_files/paper/2024/file/59eb2d8ce0e4830f80780f7f78c67dec-Paper-Conference.pdf)

> 提出多域视频异常检测任务与基准，探索跨域泛化能力，为构建适应多样化场景的异常检测系统提供理论与实验支撑。

---

### ACM MM 2024

#### Weakly Supervised Video Anomaly Detection and Localization with Spatio-Temporal Prompts

[![ACM MM](https://img.shields.io/badge/ACM_MM-2024-FF69B4)](https://arxiv.org/pdf/2408.05905)

> 利用 CLIP 的时空提示机制进行弱监督视频异常检测与定位，通过动态提示适配不同时空尺度的异常模式，提升细粒度定位能力。

---

### arXiv 2024 (Preprints)

#### Holmes-VAD: Towards Unbiased and Explainable Video Anomaly Detection via Multi-modal LLM

[![arXiv](https://img.shields.io/badge/arXiv-2024-b31b1b?logo=arxiv)]()

> 通过多模态 LLM 实现无偏且可解释的视频异常检测，结合视觉与语言模态提供异常判断的多维度证据。







