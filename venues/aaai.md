# AAAI

## Quick Navigation

- 2026: [CueBench](#-cuebench-advancing-unified-understanding-of-context-aware-video-anomalies-in-real-world)（VAU基准）、 [IAD-R1](#-iad-r1-reinforcing-consistent-reasoning-in-industrial-anomaly-detection-oral)（工业推理）、 [HeadHunt-VAD](#-headhunt-vad-hunting-robust-anomaly-sensitive-heads-in-mllm-for-tuning-free-video-anomaly-detection-oral)（免调参MLLM）、 [AD-FM](#-ad-fm-multimodal-llms-for-anomaly-detection-via-multi-stage-reasoning-and-fine-grained-reward-optimization)（多阶段推理）、 [PromptMoE](#-promptmoe-generalizable-zero-shot-anomaly-detection-via-visually-guided-prompt-mixtures)（零样本MoE）<br>[TargetVAU](#-targetvau-multimodal-anomaly-aware-reasoning-for-target-behavior-understanding-in-videos)（行为理解）、 [ICAD-LLM](#-icad-llm-one-for-all-anomaly-detection-via-incontext-learning-with-large-language-models)（ICL统一）、 [AdaptCLIP](#-adaptclip-adapting-clip-for-universal-visual-anomaly-detection)（CLIP适配）、 [VAGU & GtS](#-vagu--gts-llm-based-benchmark-and-framework-for-joint-video-anomaly-grounding-and-understanding)（定位+理解）、 [FineVAU](#-finevau-a-novel-human-aligned-benchmark-for-finegrained-video-anomaly-understanding)（细粒度VAU）
- 2025: [Motion-Appearance Diffusion](#-video-anomaly-detection-with-motion-and-appearance-guided-patch-diffusion-model)（运动外观扩散）、 [UCF-Crime-DVS](#-ucf-crime-dvs-a-novel-event-based-dataset-for-video-anomaly-detection-with-spiking-neural-networks)（脉冲神经网络）、 [Dual Conditioned Diffusion](#-dual-conditioned-motion-diffusion-for-pose-based-video-anomaly-detection)（姿态扩散）、 [Federated Multimodal Prompt](#-federated-weakly-supervised-video-anomaly-detection-with-multimodal-prompt)（联邦多模态）、 [VarCMP](#-varcmp-adapting-cross-modal-pre-training-models-for-video-anomaly-retrieval)（跨模态检索）
- 2024: [VadCLIP](#-vadclip-adapting-vision-language-models-for-weakly-supervised-video-anomaly-detection)（CLIP弱监督）、 [SDAC](#-sdac-a-multimodal-synthetic-dataset-for-anomaly-and-corner-case-detection-in-autonomous-driving)（自动驾驶数据集）
- 2023: [MGFN](#-mgfn-magnitude-contrastive-glance-and-focus-network-for-weakly-supervised-video-anomaly-detection)（对比学习）、 [Mean-Shifted Contrastive](#-mean-shifted-contrastive-loss-for-anomaly-detection)（均值偏移）、 [Event-Relevant Factors](#-learning-event-relevant-factors-for-video-anomaly-detection)（事件因子）、 [UR-DMU](#-dual-memory-units-with-uncertainty-regulation-for-weakly-supervised-video-anomaly-detection)（不确定性记忆）


## 2026

#### 📊 CueBench: Advancing Unified Understanding of Context-Aware Video Anomalies in Real-World
Yating Yu; Congqi Cao; Zhaoying Wang; Weihua Meng; Jie Li; Yuxin Li; Zihao Wei; Zhongpei Shen; Jiajun Zhang
`Benchmark` `VAU` `Context-aware` | [![Paper](https://img.shields.io/badge/Paper-arXiv-B31B1B)](https://arxiv.org/abs/2511.00613) Code: N/A

> 提出上下文感知的 VAU 统一评测基准 CueBench，构建事件中心的分层 taxonomy（14 类条件异常、18 类绝对异常），覆盖 174 个场景与 198 个属性，并在识别、时间定位、检测、预测等任务上统一评测 VLM。提出 Cue-R1 通过 R1 风格强化微调改进推理表现，显著优于现有 VLM。

#### 🏭 IAD-R1: Reinforcing Consistent Reasoning in Industrial Anomaly Detection （Oral）
Yanhui Li; Yunkang Cao; Chengliang Liu; Yuan Xiong; Xinghui Dong; Chao Huang
`Industrial` `Reasoning` `R1` | [![Paper](https://img.shields.io/badge/Paper-arXiv-B31B1B)](https://arxiv.org/abs/2508.09178) Code: N/A

> 面向工业异常检测的一致推理问题，提出 IAD-R1 强化推理一致性。

#### 🧠 HeadHunt-VAD: Hunting Robust Anomaly-Sensitive Heads in MLLM for Tuning-Free Video Anomaly Detection （Oral）
Zhaolin Cai; Fan Li; Ziwei Zheng; Haixia Bi; Lijun He
`MLLM` `Tuning-free` `Head Selection` | [![Paper](https://img.shields.io/badge/Paper-arXiv-B31B1B)](https://arxiv.org/abs/2512.17601) Code: N/A

> 在 MLLM 中定位异常敏感 head，实现免调参的 VAD 增强。

#### 🎯 RefineVAD: Semantic-Guided Feature Recalibration for Weakly Supervised Video Anomaly Detection
Junhee Lee; ChaeBeen Bang; MyoungChul Kim; MyeongAh Cho
`Weakly-supervised` `Semantic` `Feature Recalibration` | [![Paper](https://img.shields.io/badge/Paper-arXiv-B31B1B)](https://arxiv.org/abs/2511.13204) Code: N/A

> 提出语义引导的特征重校准策略，提升弱监督 VAD 表达。

#### 🧪 D-GARA: A Dynamic Benchmarking Framework for GUI Agent Robustness in Real-World Anomalies
Sen Chen; Tong Zhao; Yi Bin; Fei Ma; Wenqi Shao; Zheng Wang
`Benchmark` `GUI Agent` `Robustness` | [![Paper](https://img.shields.io/badge/Paper-arXiv-B31B1B)](https://arxiv.org/abs/2511.16590) Code: N/A

> 提出 GUI Agent 在真实异常场景下的动态评测框架 D-GARA。

#### 🔄 Reimagining Anomalies: What If Anomalies Were Normal?
Philipp Liznerski; Saurabh Varshneya; Ece Calikus; Puyu Wang; Alexander Bartscher; Sebastian Josef Vollmer; Sophie Fellenz; Marius Kloft
`Paradigm` `Reformulation` `Anomaly` | [![Paper](https://img.shields.io/badge/Paper-arXiv-B31B1B)](https://arxiv.org/abs/2402.14469) Code: N/A

> 从“异常可被视为正常”视角重审异常检测设定与学习范式。

#### 🤖 AD-FM: Multimodal LLMs for Anomaly Detection via Multi-Stage Reasoning and Fine-Grained Reward Optimization
Jingyi Liao; Yongyi Su; Rong-Cheng Tu; Zhao Jin; Wenhao Sun; Yiting Li; Xun Xu; Dacheng Tao; Xulei Yang
`Multimodal LLM` `Reasoning` `Reward Optimization` | [![Paper](https://img.shields.io/badge/Paper-arXiv-B31B1B)](https://arxiv.org/abs/2508.04175) Code: N/A

> 基于多模态 LLM，采用多阶段推理与细粒度奖励优化进行异常检测。

#### 🧩 PromptMoE: Generalizable Zero-Shot Anomaly Detection via Visually-Guided Prompt Mixtures
Yuheng Shao; Lizhang Wang; Changhao Li; Peixian Chen; Qinyuan Liu
`Zero-shot` `Prompt` `MoE` | [![Paper](https://img.shields.io/badge/Paper-arXiv-B31B1B)](https://arxiv.org/abs/2511.18116) Code: N/A

> 通过视觉引导的提示混合（Prompt MoE）实现通用零样本异常检测。

#### 🎬 TargetVAU: Multimodal Anomaly-Aware Reasoning for Target Behavior Understanding in Videos
Lingru Zhou; Peng Wu; Manqing Zhang; Qingsheng Wang; Guansong Pang; Peng Wang
`VAU` `Multimodal` `Behavior Understanding` | Paper: N/A, Code: N/A

> 面向视频目标行为理解的多模态异常感知推理框架。

#### 🧠 AnomalyMoE: Towards a Language-free Generalist Model for Unified Visual Anomaly Detection
Zhaopeng Gu; Bingke Zhu; Guibo Zhu; Yingying Chen; Wei Ge; Ming Tang; Jinqiao Wang
`Generalist` `Language-free` `MoE` | [![Paper](https://img.shields.io/badge/Paper-arXiv-B31B1B)](https://arxiv.org/abs/2508.06203) Code: N/A

> 提出无语言的通用视觉异常检测模型（MoE 架构）。

#### 🦴 RPE-PAD: Relative Pose Estimation for Pose-agnostic Anomaly Detection
Zhipeng Zhang; Mengzan Qi; Rongkang Ma; Yingying Fang; Guixu Zhang; Tieyong Zeng; Zhi Li
`Pose` `Relative Pose` `Pose-agnostic` | Paper: N/A, Code: N/A

> 以相对姿态估计为核心，实现姿态无关的异常检测。

#### 🤖 ICAD-LLM: One-for-All Anomaly Detection via InContext Learning with Large Language Models
Zhongyuan Wu; Jingyuan Wang; Zexuan Cheng; Yilong Zhou; Weizhi Wang; Juhua Pu; Chao Li; Changqing Ma
`LLM` `In-Context Learning` `Unified` | [![Paper](https://img.shields.io/badge/Paper-arXiv-B31B1B)](https://arxiv.org/abs/2512.01672) [![Code](https://img.shields.io/badge/Code-GitHub-black?logo=github)](https://github.com/nobody384/ICAD-LLM)

> 利用 LLM 的 In-Context Learning 实现统一异常检测框架。

#### 🔧 AdaptCLIP: Adapting CLIP for Universal Visual Anomaly Detection
Bin-Bin Gao; Yue Zhou; Jiangtao Yan; Yuezhi Cai; Weixi Zhang; Meng Wang; Jun Liu; Yong Liu; Lei Wang; Chengjie Wang
`CLIP` `Universal` `Anomaly Detection` | [![Paper](https://img.shields.io/badge/Paper-arXiv-B31B1B)](https://arxiv.org/abs/2505.09926) [![Code](https://img.shields.io/badge/Code-GitHub-black?logo=github)](https://github.com/bbggay/AdaptCLIP)

> 适配 CLIP 以实现通用视觉异常检测能力。

#### 🧩 Commonality in Few: Few-Shot Multimodal Anomaly Detection via Hypergraph-Enhanced Memory
Yuxuan Lin; Hanjing Yan; Xuan Tong; Yang Chang; Huanzhen Wang; Ziheng Zhou; Shuyong Gao; Yan Wang; Wenqiang Zhang
`Few-shot` `Hypergraph` `Multimodal` | [![Paper](https://img.shields.io/badge/Paper-arXiv-B31B1B)](https://arxiv.org/abs/2511.05966) [![Code](https://img.shields.io/badge/Code-GitHub-black?logo=github)](https://github.com/DSANet/DSANet)

> 通过超图增强记忆建模，实现少样本多模态异常检测。

#### 🧭 Unsupervised Multi-View Visual Anomaly Detection via Progressive Homography-Guided Alignment
Xintao Chen; Xiaohao Xu; Bozhong Zheng; Yun Liu; Yingna Wu
`Unsupervised` `Multi-view` `Alignment` | [![Paper](https://img.shields.io/badge/Paper-arXiv-B31B1B)](https://arxiv.org/abs/2511.18766) Code: N/A

> 提出渐进式单应性对齐的无监督多视角异常检测方法。

#### 🔍 Learning to Tell Apart: Weakly Supervised Video Anomaly Detection via Disentangled Semantic Alignment
Wenti Yin; Huaxin Zhang; Xiang Wang; Yuqing Lu; Yicheng Zhang; Bingquan Gong; Jialong Zuo; Li Yu; Changxin Gao; Nong Sang
`Weakly-supervised` `Disentangled` `Semantic Alignment` | [![Paper](https://img.shields.io/badge/Paper-arXiv-B31B1B)](https://arxiv.org/abs/2511.10334) [![Code](https://img.shields.io/badge/Code-GitHub-black?logo=github)](https://github.com/DSANet/DSANet)

> 通过解耦语义对齐进行弱监督视频异常检测。

#### 🏭 MAU-GPT: Enhancing Multi-type Industrial Anomaly Understanding via Anomaly-aware and Generalist Experts Adaptation
Zhuonan Wang; Zhenxuan Fan; Siwen Tan; Yu Zhong; Yuqian Yuan; Haoyuan Li; Hao Jiang; Wenqiao Zhang; Feifei Shao; Hongwei Wang; Jun Xiao
`Industrial` `LLM` `Expert Adaptation` | Paper: N/A, Code: N/A

> 面向多类型工业异常理解，结合异常感知与通用专家适配。

#### 📊 VAGU & GtS: LLM-Based Benchmark and Framework for Joint Video Anomaly Grounding and Understanding
Shibo Gao; Peipei Yang; Yangyang Liu; Yi Chen; Han Zhu; Xu-Yao Zhang; Linlin Huang
`Benchmark` `Grounding` `Understanding` | Paper: N/A, Code: N/A

> 提出用于视频异常定位与理解的 LLM 基准与框架。

#### 🧠 Exploring High-order-aware Prompt Learning for Zeroshot Anomaly Detection
Shun Wei; Jielin Jiang; Xiaolong Xu
`Zero-shot` `Prompt Learning` `High-order` | Paper: N/A, Code: N/A

> 探索高阶感知的提示学习以实现零样本异常检测。

#### 🎭 MaskAD: Parallel Masked Autoencoder for Multi-class Unsupervised Anomaly Detection
Ruiying Lu; Gang Liu; Kang Li; Long Tian; Junwei Zhang
`Unsupervised` `Masked Autoencoder` `Multi-class` | Paper: N/A, Code: N/A

> 并行掩码自编码器用于多类别无监督异常检测。

#### 📊 FineVAU: A Novel Human-Aligned Benchmark for FineGrained Video Anomaly Understanding
Joao Alexandre Cardeira Pereira; Vasco Lopes; João C. Neves; David Semedo
`Benchmark` `Fine-grained` `VAU` | Paper: N/A, Code: N/A

> 提出面向细粒度视频异常理解的人类对齐评测基准。

## 2025
- Accepted papers: <https://dblp.org/db/conf/aaai/aaai2025.html>
- Observations:
  - 扩散模型成为主流重建范式，运动与外观引导、姿态条件等多模态融合策略丰富。
  - 脉冲神经网络与事件相机数据集为视频异常检测引入新型数据表示。
  - CLIP 驱动的跨模态预训练与联邦学习结合，强化隐私保护与检索能力。

### 基于大模型

#### 🎨 VarCMP: Adapting Cross-Modal Pre-Training Models for Video Anomaly Retrieval
Northwestern Polytechnical University | `CLIP` `Cross-modal` `Retrieval` | [[Paper]](https://ojs.aaai.org/index.php/AAAI/article/view/32909)

> 适配跨模态预训练模型用于视频异常检索任务，通过 CLIP 驱动的语义对齐实现基于文本查询的异常视频检索能力。

#### 🤝 Federated Weakly Supervised Video Anomaly Detection with Multimodal Prompt
Sun Yat-sen University | `Federated Learning` `CLIP` `Multimodal Prompt` | [[Paper]](https://ojs.aaai.org/index.php/AAAI/article/view/35398)

> 结合联邦学习与多模态 Prompt 的弱监督视频异常检测框架，通过 CLIP 引导的语义提示在隐私保护前提下实现跨节点协同检测。

### 经典方案

#### 🌊 Video Anomaly Detection with Motion and Appearance Guided Patch Diffusion Model
Huazhong University of Science and Technology | `Diffusion Model` `Reconstruction` `Motion-Appearance` | [[Paper]](https://ojs.aaai.org/index.php/AAAI/article/view/33169)

> 提出运动与外观引导的补丁扩散模型，通过双路径引导机制分别建模运动模式与外观特征，增强重建式异常检测的精细度。

#### ⚡ UCF-Crime-DVS: A Novel Event-Based Dataset for Video Anomaly Detection with Spiking Neural Networks
Ningbo University | `Spiking Neural Networks` `Event-based` `Dataset` | [[Paper]](https://ojs.aaai.org/index.php/AAAI/article/view/32705)

> 发布基于事件相机的 UCF-Crime-DVS 数据集，结合脉冲神经网络（SNN）实现低延迟、高能效的视频异常检测，为神经形态计算引入新基准。

#### 🦴 Dual Conditioned Motion Diffusion for Pose-Based Video Anomaly Detection
Southeast University | `Diffusion Model` `Pose-based` `Dual Conditioning` | [[Paper]](https://ojs.aaai.org/index.php/AAAI/article/view/32829)

> 基于姿态的双条件运动扩散模型，通过时序与空间双重条件引导，在姿态序列上实现精准的异常运动模式检测。

## 2024
- Accepted papers: <https://aaai.org/wp-content/uploads/2024/02/AAAI-24_Main_2024-02-01.pdf>
- Observations:
  - CLIP 等视觉-语言模型在弱监督视频异常检测中的适配方法持续演进。
  - 自动驾驶场景下的多模态异常检测数据集推动领域应用拓展。

### 基于大模型

#### 🎬 VadCLIP: Adapting Vision-Language Models for Weakly Supervised Video Anomaly Detection
Northwestern Polytechnical University | `CLIP` `Weakly-supervised` `Vision-Language` | [[Paper]](https://ojs.aaai.org/index.php/AAAI/article/view/28423) [[Code]](https://github.com/nwpu-zxr/VadCLIP)

> 将 CLIP 等视觉-语言模型适配至弱监督视频异常检测任务，通过跨模态对齐与视频时序建模，在有限标注下实现高效异常判别。

### 经典方案

#### 🚗 SDAC: A Multimodal Synthetic Dataset for Anomaly and Corner Case Detection in Autonomous Driving
University of Science and Technology of China | `Autonomous Driving` `Multi-modal` `Dataset` | [[Paper]](https://ojs.aaai.org/index.php/AAAI/article/view/27961)

> 发布面向自动驾驶的多模态合成数据集，涵盖异常场景与边缘案例检测，为自动驾驶系统的异常感知与安全评估提供标准化基准。

## 2023
- Accepted papers: <https://dblp.org/db/conf/aaai/aaai2023.html>
- Observations:
  - 对比学习与特征嵌入方法主导弱监督检测，强调正负样本的判别性表征。
  - 记忆机制与不确定性建模成为提升鲁棒性的关键技术。
  - 生成模型开始关注事件级因子分解，增强可解释性。

### 基于大模型

#### 🔍 MGFN: Magnitude-Contrastive Glance-and-Focus Network for Weakly-Supervised Video Anomaly Detection
University of Hong Kong | `Weakly-supervised` `Contrastive Learning` `Feature Embedding` | [[Paper]](https://ojs.aaai.org/index.php/AAAI/article/view/25112) [[Code]](https://github.com/carolchenyx/MGFN)

> 提出幅度对比的注视-聚焦网络，通过粗粒度注视与细粒度聚焦的双阶段机制，结合幅度对比损失增强弱监督场景下的异常特征判别能力。

#### 📊 Mean-Shifted Contrastive Loss for Anomaly Detection
The Hebrew University of Jerusalem | `Contrastive Learning` `Feature Embedding` | [[Paper]](https://ojs.aaai.org/index.php/AAAI/article/view/25309) [[Code]](https://github.com/talreiss/Mean-Shifted-Anomaly-Detection)

> 引入均值偏移对比损失，通过动态调整特征空间中心点位置，增强正常样本的紧凑性与异常样本的可分性，提升对比学习的检测效果。

#### 🎬 Learning Event-Relevant Factors for Video Anomaly Detection
Beijing Institute of Technology | `Generative Model` `Event-level` | [[Paper]](https://ojs.aaai.org/index.php/AAAI/article/view/25334)

> 提出基于生成模型的事件相关因子学习方法，通过解耦事件级表征与背景因素，实现对异常事件的精准建模与可解释检测。

#### 🧠 UR-DMU: Dual Memory Units with Uncertainty Regulation for Weakly Supervised Video Anomaly Detection
Huazhong University of Science and Technology | `Weakly-supervised` `Memory Network` `Uncertainty` | [[Paper]](https://ojs.aaai.org/index.php/AAAI/article/view/25489) [[Code]](https://github.com/henrryzh1/UR-DMU)

> 设计带不确定性调节的双记忆单元框架，通过显式建模样本不确定性与记忆更新机制，在弱监督条件下提升异常检测的鲁棒性与准确性。
