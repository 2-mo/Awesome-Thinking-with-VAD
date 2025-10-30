# ICCV

## Quick Navigation

- 2025: [Aligning Effective Tokens](#-aligning-effective-tokens-with-video-anomaly-in-large-language-models)（多模态大模型）、 [SEEKER](#-sequential-keypoint-density-estimator-an-overlooked-baseline-of-skeleton-based-video-anomaly-detection)（骨架检测）、 [MoE-GS](#-mixture-of-experts-guided-by-gaussian-splatters-matters-a-new-approach-to-weakly-supervised-video-anomaly-detection)（弱监督·专家混合）、 [Beyond Walking](#-beyond-walking-a-large-scale-image-text-benchmark-for-text-based-person-anomaly-search)（行人异常）、 [ADSM](#-autoregressive-denoising-score-matching-is-a-good-video-anomaly-detector)（自回归去噪）
- 2023: [Multiple Pretext Tasks](#-video-anomaly-detection-via-sequentially-learning-multiple-pretext-tasks)（多预训练任务）、 [FPDM](#-feature-prediction-diffusion-model-for-video-anomaly-detection)（扩散模型）、 [MoCoDAD](#-multimodal-motion-conditioned-diffusion-model-for-skeleton-based-video-anomaly-detection)（多模态扩散）、 [TeD-SPAD](#-ted-spad-temporal-distinctiveness-for-self-supervised-privacy-preservation-for-video-anomaly-detection)（隐私保护）、 [STG-NF](#-normalizing-flows-for-human-pose-anomaly-detection)（姿态归一化流）

## 2025
- Accepted papers: <https://iccv.thecvf.com/Conferences/2025/AcceptedPapers>
- Observations:
  - 多模态大模型与骨架特征成为新兴方向，强调跨模态对齐与轻量化特征提取。
  - 弱监督框架持续演进，专家混合与高斯散射等技术提升细粒度检测能力。

### 基于大模型

#### 🧠 Aligning Effective Tokens with Video Anomaly in Large Language Models
University of Hong Kong | `Multi-modal LLM` `Token Alignment` | [[ArXiv]](https://arxiv.org/pdf/2508.06350)

> 将视频异常检测任务映射为大语言模型的 token 对齐问题，通过有效 token 选择机制实现多模态大模型驱动的视频异常理解。

### 经典方案

#### 🦴 SEEKER: Sequential Keypoint Density Estimator
University of Zagreb | `Skeleton-based` `Density Estimation` | [[Project]](https://adelic99.github.io/seeker-demo/) [[ArXiv]](https://arxiv.org/pdf/2506.18368) [[Code]](https://github.com/adelic99/seeker)

> 提出基于序列关键点密度估计的骨架视频异常检测基线方法，通过轻量化骨架特征实现高效异常判别，为骨架检测提供被忽视的强基线。

#### 🎯 MoE-GS: Mixture of Experts Guided by Gaussian Splatters
Eindhoven University of Technology | `Weakly-supervised` `Mixture of Experts` | [[ArXiv]](https://arxiv.org/abs/2508.06318) [[Code]](https://arxiv.org/abs/2508.06318)

> 结合专家混合架构与高斯散射引导机制，在弱监督场景下实现细粒度异常定位，通过多专家协同提升复杂场景下的检测鲁棒性。

#### 🚶 Beyond Walking: A Large-Scale Image-Text Benchmark for Text-based Person Anomaly Search
Xi'an Jiaotong University | `Person Anomaly` `Image-Text Benchmark` | [[Project]](https://www.zdzheng.xyz/publication/Beyond-W2025) [[Paper]](https://www.zdzheng.xyz/files/Yang_BeyondWalking.pdf) [[Code]](https://github.com/Shuyu-XJTU/CMP)

> 发布大规模图像-文本基准数据集，针对行人异常搜索任务，支持基于文本描述的异常行为检索，拓展传统行人检测边界。

#### 📊 ADSM: Autoregressive Denoising Score Matching is a Good Video Anomaly Detector
Northwestern Polytechnical University | `Generative Model` `Score Matching` | [[ArXiv]](https://arxiv.org/abs/2506.23282) [[Code]](https://github.com/Bbeholder/ADSM)

> 基于自回归去噪得分匹配的生成模型方法，通过似然估计与时序建模实现单模态视频异常检测，为生成式检测器提供新范式。

## 2023
- Accepted papers: <https://openaccess.thecvf.com/ICCV2023?day=all>
- Observations:
  - 扩散模型在视频异常检测中崭露头角，成为生成式方法的重要范式。
  - 多预训练任务与自监督学习持续演进，强调特征表征的可区分性。
  - 隐私保护与轻量化检测受到关注，姿态/骨架特征提供高效替代方案。

### 基于大模型

- 暂未收录 2023 年基于大模型的代表作，欢迎补充提案或相关链接。

### 经典方案

#### 🎯 Video Anomaly Detection via Sequentially Learning Multiple Pretext Tasks
Beijing Institute of Technology | `Self-supervised` `Multi-task Learning` | [[Paper]](https://openaccess.thecvf.com/content/ICCV2023/papers/Shi_Video_Anomaly_Detection_via_Sequentially_Learning_Multiple_Pretext_Tasks_ICCV_2023_paper.pdf)

> 通过从易到难顺序学习帧预测、帧重建和帧顺序分类任务，结合对比损失增强正常样本表征的可区分性，避免次优解并提升检测效果。

#### 🌊 FPDM: Feature Prediction Diffusion Model for Video Anomaly Detection
Tianjin University | `Diffusion Model` `Feature Prediction` | [[Paper]](https://openaccess.thecvf.com/content/ICCV2023/papers/Yan_Feature_Prediction_Diffusion_Model_for_Video_Anomaly_Detection_ICCV_2023_paper.pdf)

> 首次将扩散模型用于视频异常检测的帧特征预测，通过两个分别专注于运动与外观学习的去噪扩散隐式模块，在不依赖额外语义模型的情况下学习正常样本分布。

#### 🦴 MoCoDAD: Multimodal Motion Conditioned Diffusion Model for Skeleton-based Video Anomaly Detection
Sapienza University of Rome | `Multi-modal` `Diffusion Model` `Skeleton-based` | [[ArXiv]](https://arxiv.org/pdf/2307.07205) [[Code]](https://github.com/aleflabo/MoCoDAD)

> 提出基于扩散模型的骨骼动作生成式视频异常检测方法，通过多模态未来动作生成与统计对比提升异常检测性能。

#### 🔒 TeD-SPAD: Temporal Distinctiveness for Self-Supervised Privacy-Preservation for Video Anomaly Detection
University of Central Florida | `Weakly-supervised` `Privacy-preservation` | [[Project]](https://joefioresi718.github.io/TeD-SPAD_webpage/) [[ArXiv]](https://arxiv.org/abs/2308.11072) [[Code]](https://github.com/UCF-CRCV/TeD-SPAD)

> 引入 temporally-distinct triplet loss 增强时序判别特征，在弱监督场景下实现隐私保护与异常检测性能的平衡。

#### 🧘 STG-NF: Normalizing Flows for Human Pose Anomaly Detection
Tel-Aviv University | `Pose-based` `Normalizing Flows` | [[ArXiv]](https://arxiv.org/abs/2211.10946) [[Code]](https://github.com/orhir/STG-NF)

> 将人体姿态序列嵌入正规化流框架，扩展以处理时空特征，实现仅 1K 参数的高效异常检测，为轻量化姿态检测提供新思路。
