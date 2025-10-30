# AAAI

## Quick Navigation

- 2025: [Motion-Appearance Diffusion](#-video-anomaly-detection-with-motion-and-appearance-guided-patch-diffusion-model)（运动外观扩散）、 [UCF-Crime-DVS](#-ucf-crime-dvs-a-novel-event-based-dataset-for-video-anomaly-detection-with-spiking-neural-networks)（脉冲神经网络）、 [Dual Conditioned Diffusion](#-dual-conditioned-motion-diffusion-for-pose-based-video-anomaly-detection)（姿态扩散）、 [Federated Multimodal Prompt](#-federated-weakly-supervised-video-anomaly-detection-with-multimodal-prompt)（联邦多模态）、 [VarCMP](#-varcmp-adapting-cross-modal-pre-training-models-for-video-anomaly-retrieval)（跨模态检索）
- 2024: [VadCLIP](#-vadclip-adapting-vision-language-models-for-weakly-supervised-video-anomaly-detection)（CLIP弱监督）、 [SDAC](#-sdac-a-multimodal-synthetic-dataset-for-anomaly-and-corner-case-detection-in-autonomous-driving)（自动驾驶数据集）
- 2023: [MGFN](#-mgfn-magnitude-contrastive-glance-and-focus-network-for-weakly-supervised-video-anomaly-detection)（对比学习）、 [Mean-Shifted Contrastive](#-mean-shifted-contrastive-loss-for-anomaly-detection)（均值偏移）、 [Event-Relevant Factors](#-learning-event-relevant-factors-for-video-anomaly-detection)（事件因子）、 [UR-DMU](#-dual-memory-units-with-uncertainty-regulation-for-weakly-supervised-video-anomaly-detection)（不确定性记忆）

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
