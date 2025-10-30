# ACM MM

## Quick Navigation

- 2025: [EventVAD](#-eventvad-training-free-event-aware-video-anomaly-detection)（Training-free事件感知）、 [HiProbe-VAD](#-hiprobe-vad-video-anomaly-detection-via-hidden-states-probing-in-tuning-free-multimodal-llms)（免调优MLLM）、 [HoloTrace](#-holotrace-llm-based-bidirectional-causal-knowledge-graph-for-edge-cloud-video-anomaly-detection)（因果知识图谱）、 [Scene-Dependent Memory](#-efficient-video-anomaly-detection-via-scene-dependent-memory-assisted-inter-frame-rgb-difference-reconstruction)（场景依赖记忆）
- 2024: [Spatio-Temporal Prompts](#-weakly-supervised-video-anomaly-detection-and-localization-with-spatio-temporal-prompts)（时空提示）、 [Progressive Multi-task](#-video-anomaly-detection-via-progressive-learning-of-multiple-proxy-tasks)（渐进式多任务）、 [TDSD](#-tdsd-text-driven-scene-decoupled-weakly-supervised-video-anomaly-detection)（文本驱动解耦）、 [GENet](#-a-multilevel-guidance-exploration-network-and-behavior-scene-matching-method-for-human-behavior-anomaly-detection)（行为场景匹配）、 [Hawkeye](#-hawkeye-discovering-and-grounding-implicit-anomalous-sentiment-in-recon-videos-via-scene-enhanced-video-large-language-model)（隐式情感异常）
- 2023: [Causality-inspired Representation](#-learning-causality-inspired-representation-consistency-for-video-anomaly-detection)（因果表示学习）、 [Cross-Illumination Benchmark](#-cross-illumination-video-anomaly-detection-benchmark)（跨光照基准）

## 2025
- Accepted papers: <https://acmmm2025.org/accepted-regular-papers>
- Observations:
  - Training-free 与 Tuning-free 范式成为主流，强调大模型的即插即用能力。
  - 多模态大模型（MLLM）与知识图谱结合，引入因果推理与逻辑探测机制。
  - 场景依赖记忆与帧间差分重建提升轻量化检测效率。

### 基于大模型

#### 🎬 EventVAD: Training-Free Event-Aware Video Anomaly Detection
Peking University | `Training-free` `Video-LLaMA2` `Event-aware` | [[ArXiv]](https://arxiv.org/pdf/2504.13092) [[Code]](https://github.com/YihuaJerry/EventVAD)

> 基于 Video-LLaMA2 的免训练事件感知视频异常检测方法，通过零样本事件理解与时序推理，无需模型训练即可实现跨场景异常判别。

#### 🔍 HiProbe-VAD: Video Anomaly Detection via Hidden States Probing in Tuning-Free Multimodal LLMs
Xinjiang University | `Tuning-free` `MLLM` `Hidden States Probing` | [[ArXiv]](https://arxiv.org/pdf/2507.17394)

> 通过探测多模态大模型隐藏状态实现免调优视频异常检测，结合逻辑回归 scorer 提取 MLLM 内部表征，避免全量微调的同时保持高检测精度。

#### 🧠 HoloTrace: LLM-based Bidirectional Causal Knowledge Graph for Edge-Cloud Video Anomaly Detection
Unknown Institution | `VLM+LLM` `Causal Knowledge Graph` `Edge-Cloud` | [[Code]](https://github.com/kongyanye/HoloTrace-MM25)

> 构建基于大模型的双向因果知识图谱，结合视觉-语言模型与 LLM 推理能力，在边缘-云协同架构下实现因果驱动的视频异常检测。

### 经典方案

#### 🎯 Efficient Video Anomaly Detection via Scene-Dependent Memory Assisted Inter-Frame RGB Difference Reconstruction
Unknown Institution | `Scene-dependent` `Memory Network` `Reconstruction` | [[Paper]](https://acmmm2025.org/accepted-regular-papers)

> 提出场景依赖记忆辅助的帧间 RGB 差分重建方法，通过场景自适应记忆单元与轻量化差分建模，提升视频异常检测的计算效率与检测精度。

## 2024
- Accepted papers: <https://2024.acmmm.org/accepted-list/>
- Observations:
  - CLIP 驱动的时空提示与文本引导成为弱监督检测的核心技术。
  - 渐进式多任务学习与场景解耦策略增强模型泛化能力。
  - Video-LLM 开始应用于隐式情感异常发现，拓展异常检测边界。

### 基于大模型

#### 🎯 Weakly Supervised Video Anomaly Detection and Localization with Spatio-Temporal Prompts
Northwestern Polytechnical University | `CLIP` `Weakly-supervised` `Spatio-Temporal Prompts` | [[ArXiv]](https://arxiv.org/pdf/2408.05905)

> 利用 CLIP 的时空提示机制进行弱监督视频异常检测与定位，通过动态提示适配不同时空尺度的异常模式，提升细粒度定位能力。

#### 📝 TDSD: Text-Driven Scene-Decoupled Weakly Supervised Video Anomaly Detection
Zhejiang University | `CLIP` `Text-driven` `Scene Decoupling` | [[Paper]](https://openreview.net/pdf?id=TAVtkpjS9P) [[Code]](https://github.com/shengyangsun/TDSD)

> 提出文本驱动的场景解耦弱监督检测框架，通过 CLIP 的语义引导将场景与异常行为解耦，在复杂背景下实现精准异常判别。

#### 🦅 Hawkeye: Discovering and Grounding Implicit Anomalous Sentiment in Recon-videos
Soochow University | `Video-LLaVA` `Scene-enhanced` `Sentiment Anomaly` | [[Paper]](https://djingwang.github.io/works/Hawkeye%20Discovering%20and%20Grounding%20Implicit%20Anoma-lous%20Sentiment%20in%20Recon-videos%20via%20Scene-enhanced%20Video%20Large%20Language%20Model.pdf) [[Code]](https://github.com/Zhao-Jianing-SUDA/Hawkeye)

> 基于 Video-LLaVA 的场景增强型视频大语言模型，通过图结构场景建模发现并定位隐式情感异常，拓展异常检测至情感理解领域。

### 经典方案

#### 🔄 Video Anomaly Detection via Progressive Learning of Multiple Proxy Tasks
Beijing University of Posts and Telecommunications | `Multi-task Learning` `Semi-supervised` `Progressive Learning` | [[Paper]](https://openreview.net/pdf?id=WsNFULCsyj)

> 提出渐进式多任务学习框架，通过半监督方式逐步学习多个代理任务，在有限标注下增强模型对不同异常模式的感知能力。

#### 🎭 GENet: Multilevel Guidance-Exploration Network for Human Behavior Anomaly Detection
Xiamen University | `Behavior Anomaly` `Memory Bank` `Unsupervised` | [[ArXiv]](https://arxiv.org/pdf/2312.04119v1) [[Code]](https://github.com/moluggg/GENet)

> 设计多层次引导-探索网络与行为-场景匹配机制，通过记忆库建模实现无监督人体行为异常检测，适用于复杂场景下的行为分析。

## 2023
- Accepted papers: <https://dblp.uni-trier.de/db/conf/mm/mm2023.html>
- Observations:
  - 因果表示学习引入视频异常检测，强调特征一致性与因果推理。
  - 跨光照等特殊场景基准数据集推动模型鲁棒性研究。

### 经典方案

#### 🔗 Learning Causality-inspired Representation Consistency for Video Anomaly Detection
Fudan University | `Causal Representation` `Consistency Learning` | [[ArXiv]](https://arxiv.org/pdf/2308.01537)

> 提出因果启发的表示一致性学习方法，通过因果图建模与表示一致性约束，增强模型对异常模式的因果理解与判别能力。

#### 💡 Cross-Illumination Video Anomaly Detection Benchmark
Wuhan University | `Cross-Illumination` `Benchmark` `Feature Embedding` | [[Paper]](https://web.archive.org/web/20231028142209id_/https://dl.acm.org/doi/pdf/10.1145/3581783.3612531)

> 发布跨光照视频异常检测基准数据集，涵盖不同光照条件下的异常场景，为评估模型在光照变化下的鲁棒性提供标准化测试平台。
