# ECCV

## Quick Navigation

- 2024: [Joint-VAD](#-interleaving-one-class-and-weakly-supervised-models-with-adaptive-thresholding-for-unsupervised-video-anomaly-detection)（无监督融合）、 [AnomalyRuler](#-follow-the-rules-reasoning-for-video-anomaly-detection-with-large-language-models)（LLM推理）、 [AdaCLIP](#-adaclip-adapting-clip-with-hybrid-learnable-prompts-for-zero-shot-anomaly-detection)（CLIP零样本）、 [CDL](#-cross-domain-learning-for-video-anomaly-detection-with-limited-supervision)（跨域弱监督）、 [OLN-SSOS](#-towards-open-world-object-based-anomaly-detection-via-self-supervised-outlier-synthesis)（开放世界）、 [LANP](#-learning-anomalies-with-normality-prior-for-unsupervised-video-anomaly-detection)（正常性先验）、 [FedVAD](#-fedvad-enhancing-federated-video-anomaly-detection-with-gpt-driven-semantic-distillation)（联邦学习）

## 2024
- Accepted papers: <https://eccv.ecva.net/virtual/2024/papers.html>
- Observations:
  - 大语言模型驱动的推理式检测与多模态融合成为主流，CLIP 与 GPT 被广泛用于语义理解。
  - 无监督与跨域学习框架持续演进，强调正常性传播、自适应阈值与联邦隐私保护。
  - 开放世界场景受到关注，自监督异常合成为未知类别检测提供新思路。

### 基于大模型

#### 🧠 AnomalyRuler: Follow the Rules for Video Anomaly Detection with Large Language Models
Johns Hopkins University | `LLM Reasoning` `Rule-based Detection` | [[ArXiv]](https://arxiv.org/abs/2407.10299) [[Code]](https://github.com/Yuchen413/AnomalyRuler)

> 通过两阶段推理机制（归纳与演绎）利用 LLM 从少量正常样本中总结正常规律，再据此检测异常帧，结合规则聚合与感知平滑提升推理稳健性与检测准确度。

#### 🎨 AdaCLIP: Adapting CLIP with Hybrid Learnable Prompts for Zero-Shot Anomaly Detection
Huazhong University of Science and Technology | `CLIP` `Zero-shot` `Learnable Prompts` | [[ArXiv]](https://arxiv.org/abs/2407.15795) [[Code]](https://github.com/caoyunkang/AdaCLIP)

> 在 CLIP 中引入静态和动态混合学习型提示，实现在不同测试图像上动态适应，提升模型对未见类别的异常检测能力。

#### 🤝 FedVAD: Enhancing Federated Video Anomaly Detection with GPT-Driven Semantic Distillation
Tianjin University of Technology | `Federated Learning` `Multi-modal` `GPT` | [[Paper]](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/06981.pdf) [[Code]](https://github.com/Eurekaer/FedVAD)

> 结合联邦学习和多模态训练的隐私保护视频异常检测框架，通过 GPT 驱动的语义蒸馏在异构客户端数据下提升检测性能。

### 经典方案

#### 🔄 Joint-VAD: Interleaving One-Class and Weakly-Supervised Models with Adaptive Thresholding
South China University of Technology | `Unsupervised` `Hybrid Training` | [[ArXiv]](https://arxiv.org/abs/2401.13551) [[Code]](https://github.com/benedictstar/Joint-VAD)

> 融合加权单类分类与弱监督互训练的无监督视频异常检测框架，通过自适应阈值与软标签机制实现无人工标注的高效异常学习。

#### 🌍 CDL: Cross-Domain Learning for Video Anomaly Detection with Limited Supervision
University of Delhi | `Cross-domain` `Weakly-supervised` | [[ArXiv]](https://arxiv.org/abs/2408.05191)

> 提出弱监督跨域学习框架，通过预测外部数据的偏差与不确定性并自适应最小化预测偏差，实现跨域情况下异常检测性能的大幅提升。

#### 🌐 OLN-SSOS: Towards Open-World Object-based Anomaly Detection via Self-Supervised Outlier Synthesis
Durham University | `Open-world` `Self-supervised` `Object-level` | [[Project]](https://kostadinovshalon.github.io/oln-ssos/) [[ArXiv]](https://arxiv.org/abs/2407.15763) [[Code]](https://github.com/KostadinovShalon/oln-ssos)

> 面向开放世界场景的对象级异常检测方法，通过自监督学习得到伪类别并进行类条件的虚拟异常特征合成，使检测器无需类别标签即可检测未知类别的异常目标。

#### 📡 LANP: Learning Anomalies with Normality Prior for Unsupervised Video Anomaly Detection
National Key Laboratory of Human-Machine Hybrid Augmented Intelligence | `Unsupervised` `Normality Propagation` | [[Paper]](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/00941.pdf) [[Code]](https://github.com/shyern/LANP-UVAD)

> 引入正常性先验的无监督视频异常检测方法，提出 normality propagation 机制利用视频片段间的关系传播正常性信息，结合标签传播与重加权损失显著提升弱特征异常的检测能力。
