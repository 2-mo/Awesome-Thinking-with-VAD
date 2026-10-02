# NeurIPS

> 历史资料：保留较广的原始收集范围，尚未全部复核。当前精选内容见[论文年表](../../llm4vad.md)与[数据集索引](../../literature/benchmarks.md)。

## Quick Navigation

- 2025: [A2Seek](#-a2seek-towards-reasoning-centric-benchmark-for-aerial-anomaly-understanding)（无人机推理）、 [FrameShield](#-frameshield-adversarially-robust-video-anomaly-detection)（对抗鲁棒）、 [Single-Frame Supervision](#-generalizing-single-frame-supervision-to-event-level-understanding-for-video-anomaly-detection)（单帧监督）、 [VADTree](#-vadtree-explainable-training-free-video-anomaly-detection-via-hierarchical-granularity-aware-tree)（层次感知）、 [PANDA](#-panda-towards-generalist-video-anomaly-detection-via-detective-like-agent)（Agent工具调用）、 [Interactive Anomaly Detection](#-interactive-anomaly-detection-for-articulated-objects-via-motion-anticipation)（交互式检测）、 [MoniTor](#-monitor-exploiting-large-language-models-with-instruction-for-online-video-anomaly-detection)（LLM在线检测）
- 2024: [MSAD](#-msad-advancing-video-anomaly-detection---a-concise-review-and-a-new-dataset)（数据集+综述）、 [HAWK](#-hawk-learning-to-understand-open-world-video-anomalies)（VLM+运动模态）、 [MDVAD](#-mdvad-towards-multi-domain-learning-for-generalizable-video-anomaly-detection)（多域泛化）、 [Dual-Space](#-beyond-euclidean-dual-space-representation-learning-for-weakly-supervised-video-violence-detection)（双曲空间）
- 2023: _TBD_

## 2025
- Accepted papers: <https://neurips.cc/virtual/2025/papers.html>
- Observations:
  - 大模型驱动的 Agent 与工具调用范式兴起，强调推理、解释性与在线检测能力。
  - 无监督与免训练方法持续演进，层次感知与单帧监督拓展弱监督边界。
  - 特定场景（无人机、关节物体）与对抗鲁棒性成为新兴研究方向。

### 基于大模型

#### 🚁 A2Seek: Towards Reasoning-Centric Benchmark for Aerial Anomaly Understanding
Chongqing University of Posts and Telecommunications | `UAV` `Reasoning Benchmark` `LLM` | [[ArXiv]](https://arxiv.org/pdf/2505.21962) [[Project]](https://hayneyday.github.io/A2Seek/)

> 面向无人机视角的推理中心型异常理解基准，利用大模型进行空中场景的复杂推理与异常判别，为 UAV 视频异常检测提供标准化评测。

#### 🔍 PANDA: Towards Generalist Video Anomaly Detection via Detective-like Agent
Unknown Institution | `Agent` `Tool Calling` `LLM` | [[Paper]](https://neurips.cc/virtual/2025/poster/115891)

> 提出侦探式 Agent 范式的通用视频异常检测框架，通过工具调用与多步推理实现跨场景、跨类别的异常检测泛化能力。

#### 📺 MoniTor: Exploiting Large Language Models with Instruction for Online Video Anomaly Detection
Unknown Institution | `LLM` `Online Detection` `Instruction-based` | [[Paper]](https://neurips.cc/virtual/2025/poster/119803)

> 利用大语言模型与指令驱动机制实现在线视频异常检测，通过流式推理与实时反馈提升在线场景下的检测响应速度与准确性。

### 经典方案

#### 🛡️ FrameShield: Adversarially Robust Video Anomaly Detection
Unknown Institution | `Adversarial Robustness` `Weakly-supervised` `Pseudo Anomaly` | [[Paper]](https://neurips.cc/virtual/2025/poster/119722)

> 针对对抗攻击设计鲁棒性视频异常检测框架，结合弱监督学习与伪异常生成，增强模型在对抗扰动下的检测稳定性。

#### 📹 Generalizing Single-Frame Supervision to Event-level Understanding for Video Anomaly Detection
Unknown Institution | `Single-frame Supervision` `Event-level` | [[Paper]](https://neurips.cc/virtual/2025/poster/116040)

> 将单帧监督信号泛化至事件级理解，通过时序关联与弱标签传播实现从帧级标注到事件级异常检测的能力提升。

#### 🌳 VADTree: Explainable Training-free Video Anomaly Detection via Hierarchical Granularity-aware Tree
Unknown Institution | `Training-free` `Hierarchical` `Explainable` `LLM` | [[Paper]](https://neurips.cc/virtual/2025/poster/116838)

> 提出基于层次粒度感知树的免训练视频异常检测方法，通过树结构建模不同粒度的异常模式，提供可解释的检测结果无需额外训练。

#### 🤖 Interactive Anomaly Detection for Articulated Objects via Motion Anticipation
Unknown Institution | `Interactive Detection` `Motion Anticipation` `Articulated Objects` | [[Paper]](https://neurips.cc/virtual/2025/poster/115640)

> 针对关节物体的交互式异常检测方法，通过运动预期机制预测物体关节运动轨迹，实现对异常交互行为的实时检测。

## 2024
- Accepted papers: <https://nips.cc/virtual/2024/papers.html>
- Spotlighted VAD papers: MSAD dataset, HAWK (VLM), MDVAD (multi-domain), dual-space weakly supervised
- Observations: 
  - 数据集建设与综述并行推进，MSAD 提供大规模多场景基准
  - VLM 开始引入运动模态增强开放世界异常理解
  - 多域泛化与双曲空间表征探索新的学习范式

### 数据集与综述

#### 📊 MSAD: Advancing Video Anomaly Detection - A Concise Review and a New Dataset
Australian National University - 朱丽云 | `Dataset` `Review` `Multi-scenario` | [[Project]](https://msad-dataset.github.io/) [[Paper]](https://proceedings.neurips.cc/paper_files/paper/2024/file/a3c5af1f56fc73eef1ba0f442739f5ca-Paper-Datasets_and_Benchmarks_Track.pdf) [[Code]](https://github.com/Tom-roujiang/MSAD)

> 提出 MSAD（Multi-Scenario Anomaly Detection）大规模多场景异常检测数据集，同时提供简明综述总结领域进展，为跨场景泛化研究提供标准化基准。

### 基于大模型

#### 🦅 HAWK: Learning to Understand Open-World Video Anomalies
Hong Kong University of Science and Technology - 唐嘉祺 | `VLM` `Motion Modality` `Open-world` | [[Paper]](https://proceedings.neurips.cc/paper_files/paper/2024/file/fca83589e85cb061631b7ebc5db5d6bd-Paper-Conference.pdf) [[Code]](https://github.com/jqtangust/hawk)

> 利用视觉语言模型理解开放世界视频异常，引入运动模态增强 VLM 对动态异常的感知能力，突破传统封闭集检测限制。

### 经典方案

#### 🌐 MDVAD: Towards Multi-Domain Learning for Generalizable Video Anomaly Detection
Kyung Hee University - MyeongAh Cho | `Multi-domain` `Generalization` `Benchmark` | [[Paper]](https://proceedings.neurips.cc/paper_files/paper/2024/file/59eb2d8ce0e4830f80780f7f78c67dec-Paper-Conference.pdf)

> 提出多域视频异常检测任务与基准，探索跨域泛化能力，为构建适应多样化场景的异常检测系统提供理论与实验支撑。

#### 🔺 Beyond Euclidean: Dual-Space Representation Learning for Weakly Supervised Video Violence Detection
Chongqing University of Posts and Telecommunications | `Hyperbolic Space` `Weakly-supervised` `Violence Detection` | [[Paper]](https://proceedings.neurips.cc/paper_files/paper/2024/file/1f471322127d6347e5ae09a14b1e5cf7-Paper-Conference.pdf)

> 超越欧氏空间的双空间表征学习方法，利用双曲空间建模暴力事件的层次结构与语义关系，提升弱监督暴力检测性能。

## 2023
- Accepted papers: <https://nips.cc/virtual/2023/papers.html>
- Spotlighted VAD papers: _TBD_
- Observations: _TBD_
