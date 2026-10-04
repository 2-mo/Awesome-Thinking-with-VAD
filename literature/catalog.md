# Anomaly Understanding — 精选核验目录

这是从结构化数据生成的精选核验目录，并非完整文献综述。原仓库的会议笔记作为额外资料保留，尚未全面复核，不应视作本目录的核验条目。

数据更新时间：2026-10-04 · 127 篇论文。另见 [按年份阅读](../llm4vad.md)、[发表索引](venues.md)、[阅读路线](reading-guide.md)与[引用导出](citations.md)。

## Core research / 核心研究

## 视频异常检测

精选支撑异常理解的检测范式：视觉语言提示、开放词汇、多模态语义、欧氏／双曲几何表征、联邦语义蒸馏、事件完整性与视觉语言复核；主要输出为异常判断与定位。

研究问题：怎样判断和定位异常，并适应未知类别、新模态与免训练部署？

<a id="paper-vadclip"></a>

### VadCLIP: Adapting Vision-Language Models for Weakly Supervised Video Anomaly Detection

**2024 · AAAI** · [paper](<https://arxiv.org/abs/2308.11681>) · [code](<https://github.com/nwpu-zxr/VadCLIP>) · [引用 / BibTeX](<citations.md#cite-vadclip>)

**创新：视觉语言双分支对齐**

利用冻结 CLIP 的视觉语言关联，通过双分支完成粗粒度和细粒度异常检测。

- 任务：异常检测、异常定位
- 核心启示：语义对齐让检测分数能够关联异常类别。
- 阅读关注：类别语义对齐与开放式异常解释仍有距离。
- 核验：2026-09-29；[来源 1](<https://arxiv.org/abs/2308.11681>) · [来源 2](<https://ojs.aaai.org/index.php/AAAI/article/view/28423>)

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

<a id="paper-td-vad"></a>

### TD-VAD: Breaking Visual Dependence in Video Anomaly Detection with Text-Driven Learning

**2026 · ICML** · [paper](<https://arxiv.org/abs/2608.11820>) · [引用 / BibTeX](<citations.md#cite-td-vad>)

**创新：文本时序监督与事件演化注意力**

以 LLM 生成的时序事件文本训练检测器，通过事件演化因果注意力建模长短期依赖，推理时用冻结 CLIP 对齐视频。

- 任务：异常检测
- 核心启示：用时序事件文本替代目标域异常视频。
- 阅读关注：主要输出异常分数；文本到视频的模态差距及语言先验偏差仍需检查。
- 核验：2026-09-30；[来源 1](<https://arxiv.org/abs/2608.11820>) · [来源 2](<https://icml.cc/virtual/2026/poster/65928>)

<a id="paper-mpgdfl"></a>

### Multilingual-Prompt-Guided Directional Feature Learning for Weakly Supervised Video Anomaly Detection

**2025 · TPAMI** · [paper](<https://doi.org/10.1109/tpami.2025.3590242>) · [引用 / BibTeX](<citations.md#cite-mpgdfl>)

**创新：多语言提示选择与方向特征学习**

以多语言、多提示定义正常与异常，按片段自适应选择提示，并结合 Transformer–Mamba 时序建模与方向损失学习视觉特征。

- 任务：异常检测
- 方法与场景标签：多语言提示、弱监督学习、Transformer、Mamba
- 核心启示：不同语言和提示可提供互补的异常语义约束。
- 阅读关注：多语言提示数量、片段级选择和方向损失各自带来哪些贡献？
- 核验：2026-10-01；[来源 1](<https://pubmed.ncbi.nlm.nih.gov/40674182/>) · [来源 2](<https://api.crossref.org/works/10.1109/tpami.2025.3590242>)

<a id="paper-pel"></a>

### Learning Prompt-Enhanced Context Features for Weakly-Supervised Video Anomaly Detection

**2024 · TIP** · [paper](<https://doi.org/10.1109/tip.2024.3451935>) · [code](<https://github.com/yujiangpu20/PEL4VAD>) · [引用 / BibTeX](<citations.md#cite-pel>)

**创新：提示增强语义与时序上下文聚合**

通过共享注意力矩阵和自适应融合聚合局部与全局时序上下文，再以知识提示增强视觉特征的异常子类区分能力。

- 任务：异常检测、异常定位
- 方法与场景标签：知识提示、弱监督学习、时序上下文
- 核心启示：类别语义约束能补充二元异常判定的监督信息。
- 阅读关注：时序聚合与语义提示分别改善哪些异常子类，如何控制误报？
- 核验：2026-10-01；[来源 1](<https://ieeexplore.ieee.org/document/10667004/>) · [来源 2](<https://github.com/yujiangpu20/PEL4VAD>) · [来源 3](<https://api.crossref.org/works/10.1109/tip.2024.3451935>)

<a id="paper-ewad"></a>

### Towards Video Anomaly Detection from Event Streams: A Baseline and Benchmark Datasets

**2026 · ECCV** · [paper](<https://eccv.ecva.net/virtual/2026/poster/5214>) · [code](<https://github.com/kanyutingfeng/EWAD>) · [引用 / BibTeX](<citations.md#cite-ewad>)

**创新：事件密度采样与跨模态蒸馏**

构建同步事件流与 RGB 的视频异常基准，用事件密度引导采样、时序建模及 RGB 到事件流的蒸馏。

- 任务：异常检测、基准评测
- 方法与场景标签：事件相机、事件流
- 核心启示：事件流可作为低冗余的动态异常表征，并需要专门的采样和评测协议。
- 阅读关注：区分模拟事件流表现与真实事件相机采集条件。
- 核验：2026-10-01；[来源 1](<https://eccv.ecva.net/virtual/2026/poster/5214>) · [来源 2](<https://arxiv.org/abs/2603.24991>)

<a id="paper-scene-dependent-vad"></a>

### Scene-Dependent Video Anomaly Detection via Discriminative-Contrastive Learning from Intrinsic Scene Labels

**2026 · ACM MM** · [paper](<https://2026.acmmm.org/site/technical-programme.html>) · [引用 / BibTeX](<citations.md#cite-scene-dependent-vad>)

**创新：场景标签判别与前景对比**

用内在场景标签训练事件分类，并以事件和场景的对比表示及距离构建场景依赖的异常分数。

- 任务：异常检测
- 核心启示：场景语义可显式参与正常事件的判别。
- 阅读关注：跨摄像头新场景及场景标签质量可能改变判别边界。
- 核验：2026-10-01；[来源 1](<https://2026.acmmm.org/site/program-data.json>)

<a id="paper-upr-vad"></a>

### UPR-VAD: Uncertainty-guided Signal Purification and Regularization for Weakly-Supervised Video Anomaly Detection

**2026 · ACM MM** · [paper](<https://2026.acmmm.org/site/technical-programme.html>) · [code](<https://github.com/WeiliangHuang-UM-connect/UPR-VAD>) · [引用 / BibTeX](<citations.md#cite-upr-vad>)

**创新：不确定性净化与对比约束**

通过不确定性感知的信息瓶颈过滤模糊片段，结合视频内负例对比和稀疏置信约束学习弱监督异常信号。

- 任务：异常检测
- 核心启示：弱监督视频标签需要控制模糊片段带来的训练噪声。
- 阅读关注：不确定性估计可能把困难但真实的异常片段一并压制。
- 核验：2026-10-01；[来源 1](<https://2026.acmmm.org/site/program-data.json>)

<a id="paper-deal-vad"></a>

### DEAL: Deep Evidential Audio-Visual Learning for Weakly Supervised Video Anomaly Detection

**2026 · ACM MM** · [paper](<https://2026.acmmm.org/site/technical-programme.html>) · [引用 / BibTeX](<citations.md#cite-deal-vad>)

**创新：音视频时序对齐与证据融合**

以状态空间模块对齐音视频，再用证据式双分支融合估计模态不确定性，降低噪声和时间错位的影响。

- 任务：异常检测
- 方法与场景标签：音视频融合
- 核心启示：音视频融合应同时处理时间错位与各模态的可靠性。
- 阅读关注：验证静音、背景噪声及缺失模态下的鲁棒性。
- 核验：2026-10-01；[来源 1](<https://2026.acmmm.org/site/program-data.json>)

<a id="paper-copra"></a>

### COPRA: Conditional Parameter Adaptation with Reinforcement Learning for Video Anomaly Detection

**2026 · NeurIPS** · [paper](<https://arxiv.org/abs/2605.15325>) · [引用 / BibTeX](<citations.md#cite-copra>)

**创新：片段条件化参数适配**

根据视频片段为冻结视觉语言模型生成条件化参数更新，在训练和推理时采用一致适配；除异常检测外，还可迁移到视频问答与密集字幕。

- 任务：异常检测
- 核心启示：随输入变化的权重适配有助于处理跨域视频异常。
- 阅读关注：额外参数生成的开销及跨域收益需与固定适配对照。
- 核验：2026-10-01；[来源 1](<https://neurips.cc/Downloads/2026>) · [来源 2](<https://neurips.cc/virtual/2026/poster/151176>) · [来源 3](<https://arxiv.org/abs/2605.15325>)

<a id="paper-spherevad"></a>

### SphereVAD: Training-Free Video Anomaly Detection via Geodesic Inference on the Unit Hypersphere

**2026 · NeurIPS** · [paper](<https://neurips.cc/virtual/2026/poster/152043>) · [引用 / BibTeX](<citations.md#cite-spherevad>)

**创新：超球面测地异常推断**

读取预训练多模态大模型的中间层特征，在单位超球面上进行中心化、场景注意力与测地推断，以少量合成图像校准实现免训练零样本异常评分。

- 任务：异常检测
- 核心启示：几何归一化可为免训练异常评分提供另一种判别空间。
- 阅读关注：核查合成图像校准、场景变化与超参数对免训练设定的影响。
- 核验：2026-10-01；[来源 1](<https://neurips.cc/Downloads/2026>) · [来源 2](<https://neurips.cc/virtual/2026/poster/152043>) · [来源 3](<https://arxiv.org/abs/2605.08003>)

<a id="paper-dsrl"></a>

### Beyond Euclidean: Dual-Space Representation Learning for Weakly Supervised Video Violence Detection

**2024 · NeurIPS** · [paper](<https://proceedings.neurips.cc/paper_files/paper/2024/hash/1f471322127d6347e5ae09a14b1e5cf7-Abstract-Conference.html>) · [引用 / BibTeX](<citations.md#cite-dsrl>)

**创新：欧氏与双曲空间交互表征**

结合欧氏空间的视觉特征与双曲空间的事件层次关系，通过能量约束的分层信息聚合和跨空间注意力区分外观相似的正常与暴力事件。

- 任务：异常检测、异常定位
- 方法与场景标签：双曲空间、暴力检测、易混淆事件
- 核心启示：易混淆事件需要兼顾视觉外观和事件层次关系；几何表征为语义区分提供基础。
- 阅读关注：输出为暴力检测分数，层次关系建模尚不直接提供语言解释。
- 核验：2026-10-04；[来源 1](<https://proceedings.neurips.cc/paper_files/paper/2024/hash/1f471322127d6347e5ae09a14b1e5cf7-Abstract-Conference.html>) · [来源 2](<https://proceedings.neurips.cc/paper_files/paper/2024/file/1f471322127d6347e5ae09a14b1e5cf7-Paper-Conference.pdf>)

<a id="paper-piercingeye"></a>

### PiercingEye: Dual-Space Video Violence Detection With Hyperbolic Vision-Language Guidance

**2026 · TPAMI** · [paper](<https://doi.org/10.1109/tpami.2025.3617460>) · [code](<https://github.com/wuzhanjie123/PiercingEye>) · [引用 / BibTeX](<citations.md#cite-piercingeye>)

**创新：易混淆文本与双曲视语对齐**

在 DSRL 双空间表征上，利用 VLM／LLM 改写场景或行为生成易混淆事件描述，再以动态加权的双曲视觉语言对比损失强化细粒度区分。

- 任务：异常检测、异常定位
- 方法与场景标签：双曲空间、视觉语言对齐、易混淆事件
- 核心启示：把外观相似但语义不同的事件写成训练信号，使层次表征与语言语义共同参与异常判别。
- 阅读关注：生成的描述用于训练监督，最终检测分数不等同于面向用户的异常解释。
- 核验：2026-10-04；[来源 1](<https://arxiv.org/html/2504.18866v1>) · [来源 2](<https://github.com/wuzhanjie123/PiercingEye>) · [来源 3](<https://api.crossref.org/works/10.1109/tpami.2025.3617460>)

<a id="paper-tthf"></a>

### Text-Driven Traffic Anomaly Detection With Temporal High-Frequency Modeling in Driving Videos

**2024 · TCSVT** · [paper](<https://doi.org/10.1109/tcsvt.2024.3390173>) · [引用 / BibTeX](<citations.md#cite-tthf>)

**创新：文本对齐与时序高频异常聚焦**

将文本提示与驾驶视频对齐，结合时序高频建模和注意力异常聚焦模块，提高道路异常事件的视觉语言判别。

- 任务：异常检测
- 核心启示：在道路场景中把异常语义与瞬时运动变化结合，为后续语言化理解提供表征基础。
- 阅读关注：文本用于检测监督与语义对齐，不能把异常分数视为自由文本原因解释。
- 核验：2026-10-04；[来源 1](<https://arxiv.org/abs/2401.03522>) · [来源 2](<https://api.crossref.org/works/10.1109/tcsvt.2024.3390173>)

<a id="paper-mgfn"></a>

### MGFN: Magnitude-Contrastive Glance-and-Focus Network for Weakly-Supervised Video Anomaly Detection

**2023 · AAAI** · [paper](<https://ojs.aaai.org/index.php/AAAI/article/view/25112>) · [引用 / BibTeX](<citations.md#cite-mgfn>)

**创新：全局局部建模与幅值对比学习**

用 Glance-and-Focus 网络联合长程上下文与局部时序特征，通过特征幅值增强和幅值对比损失减轻不同场景下幅值与异常性不一致的问题。

- 任务：异常检测、异常定位
- 方法与场景标签：多实例学习、幅值对比学习、弱监督
- 核心启示：作为视觉语言检测之前的弱监督基线，理解如何只凭视频级标签学习片段异常分数。
- 阅读关注：幅值对比改善检测判别性，但不直接输出异常原因或自然语言解释。
- 核验：2026-10-04；[来源 1](<https://ojs.aaai.org/index.php/AAAI/article/view/25112>) · [来源 2](<https://arxiv.org/pdf/2211.15098>)

<a id="paper-ur-dmu"></a>

### Dual Memory Units with Uncertainty Regulation for Weakly Supervised Video Anomaly Detection

**2023 · AAAI** · [paper](<https://ojs.aaai.org/index.php/AAAI/article/view/25489>) · [引用 / BibTeX](<citations.md#cite-ur-dmu>)

**创新：正常异常双记忆与不确定性约束**

以全局与局部自注意力提取时序特征，分别记忆正常和异常原型，并约束正常特征的潜在分布，以区分易混淆片段并抑制噪声影响。

- 任务：异常检测、异常定位
- 方法与场景标签：双记忆库、不确定性、弱监督
- 核心启示：对照后来的语义记忆方法，区分检测原型记忆和可检索语言经验的作用。
- 阅读关注：模型学习了正常与异常双侧记忆，训练需要视频级异常标签，不能归为仅正常样本训练。
- 核验：2026-10-04；[来源 1](<https://ojs.aaai.org/index.php/AAAI/article/view/25489>) · [来源 2](<https://arxiv.org/pdf/2302.05160>)

<a id="paper-stg-nf"></a>

### Normalizing Flows for Human Pose Anomaly Detection

**2023 · ICCV** · [paper](<https://openaccess.thecvf.com/content/ICCV2023/html/Hirschorn_Normalizing_Flows_for_Human_Pose_Anomaly_Detection_ICCV_2023_paper.html>) · [code](<https://github.com/orhir/STG-NF>) · [引用 / BibTeX](<citations.md#cite-stg-nf>)

**创新：姿态时空图归一化流似然**

将人体姿态图序列映射到潜在概率分布，用时空图归一化流直接计算似然并评估人体动作异常；同时研究仅正常数据训练和带异常标签的监督设置。

- 任务：异常检测、异常定位
- 方法与场景标签：姿态序列、归一化流、密度估计
- 核心启示：以紧凑姿态表征建模正常性，为人体行为分析提供区别于 RGB 重构的检测路径。
- 阅读关注：约 1K 参数指异常建模网络，不包含姿态估计与跟踪；姿态输入也不能覆盖所有非人体异常。
- 核验：2026-10-04；[来源 1](<https://openaccess.thecvf.com/content/ICCV2023/html/Hirschorn_Normalizing_Flows_for_Human_Pose_Anomaly_Detection_ICCV_2023_paper.html>) · [来源 2](<https://openaccess.thecvf.com/content/ICCV2023/papers/Hirschorn_Normalizing_Flows_for_Human_Pose_Anomaly_Detection_ICCV_2023_paper.pdf>)

<a id="paper-fpdm"></a>

### Feature Prediction Diffusion Model for Video Anomaly Detection

**2023 · ICCV** · [paper](<https://openaccess.thecvf.com/content/ICCV2023/html/Yan_Feature_Prediction_Diffusion_Model_for_Video_Anomaly_Detection_ICCV_2023_paper.html>) · [引用 / BibTeX](<citations.md#cite-fpdm>)

**创新：运动预测与外观细化双扩散**

以两个去噪扩散隐式模块分别预测和细化视频帧特征，学习正常运动与外观分布，再用特征预测误差评估异常，无需额外的对象或动作语义提取模型。

- 任务：异常检测、异常定位
- 方法与场景标签：扩散模型、特征预测、正常性建模
- 核心启示：把扩散用于正常特征分布建模，与使用扩散合成异常训练样本的路线对照阅读。
- 阅读关注：两阶段扩散推断与训练成本需单独评估；预测误差检测不等同于异常发生前的提前预警。
- 核验：2026-10-04；[来源 1](<https://openaccess.thecvf.com/content/ICCV2023/html/Yan_Feature_Prediction_Diffusion_Model_for_Video_Anomaly_Detection_ICCV_2023_paper.html>) · [来源 2](<https://openaccess.thecvf.com/content/ICCV2023/papers/Yan_Feature_Prediction_Diffusion_Model_for_Video_Anomaly_Detection_ICCV_2023_paper.pdf>)

<a id="paper-sd-mae"></a>

### Self-Distilled Masked Auto-Encoders are Efficient Video Anomaly Detectors

**2024 · CVPR** · [paper](<https://openaccess.thecvf.com/content/CVPR2024/html/Ristea_Self-Distilled_Masked_Auto-Encoders_are_Efficient_Video_Anomaly_Detectors_CVPR_2024_paper.html>) · [引用 / BibTeX](<citations.md#cite-sd-mae>)

**创新：运动引导掩码重构与自蒸馏**

用运动梯度突出前景 token，令共享编码器的教师与学生解码器形成自蒸馏差异；同时以合成异常增强训练，联合学习正常帧重构与像素异常图。

- 任务：异常检测、异常定位
- 方法与场景标签：掩码自编码器、自蒸馏、合成异常
- 核心启示：展示正常性重构、教师学生差异和合成异常监督如何共同改善检测效率。
- 阅读关注：完整方法包含合成异常训练，不能概括成完全不使用异常监督；报告速度也需结合硬件和整条处理流程。
- 核验：2026-10-04；[来源 1](<https://openaccess.thecvf.com/content/CVPR2024/html/Ristea_Self-Distilled_Masked_Auto-Encoders_are_Efficient_Video_Anomaly_Detectors_CVPR_2024_paper.html>) · [来源 2](<https://openaccess.thecvf.com/content/CVPR2024/papers/Ristea_Self-Distilled_Masked_Auto-Encoders_are_Efficient_Video_Anomaly_Detectors_CVPR_2024_paper.pdf>)

<a id="paper-adsm"></a>

### Autoregressive Denoising Score Matching is a Good Video Anomaly Detector

**2025 · ICCV** · [paper](<https://openaccess.thecvf.com/content/ICCV2025/html/Zhang_Autoregressive_Denoising_Score_Matching_is_a_Good_Video_Anomaly_Detector_ICCV_2025_paper.html>) · [code](<https://github.com/Bbeholder/ADSM>) · [引用 / BibTeX](<citations.md#cite-adsm>)

**创新：场景运动感知自回归去噪评分**

在原始视频空间以噪声条件 Transformer 学习得分函数，结合场景条件和运动权重，并通过自回归加噪、去噪与外观差异累积增强异常判断。

- 任务：异常检测、异常定位
- 方法与场景标签：去噪得分匹配、场景条件、正常性建模
- 核心启示：从简单的低似然异常判据推进到对局部异常模式、场景和运动的联合检测。
- 阅读关注：自回归指去噪评分过程，不能据此标为严格在线、无未来帧的流式检测器。
- 核验：2026-10-04；[来源 1](<https://openaccess.thecvf.com/content/ICCV2025/html/Zhang_Autoregressive_Denoising_Score_Matching_is_a_Good_Video_Anomaly_Detector_ICCV_2025_paper.html>) · [来源 2](<https://openaccess.thecvf.com/content/ICCV2025/papers/Zhang_Autoregressive_Denoising_Score_Matching_is_a_Good_Video_Anomaly_Detector_ICCV_2025_paper.pdf>)

<a id="paper-bn-wvad"></a>

### BatchNorm-Based Weakly Supervised Video Anomaly Detection

**2024 · TCSVT** · [paper](<https://ieeexplore.ieee.org/document/10649595/>) · [引用 / BibTeX](<citations.md#cite-bn-wvad>)

**创新：批归一化均值偏离与片段筛选**

以特征偏离 BatchNorm 均值的程度作为异常判据，结合样本与批次级片段选择学习异常分类，并用统计评分修正易受标签噪声影响的预测。

- 任务：异常检测、异常定位
- 方法与场景标签：批归一化、统计异常评分、弱监督
- 核心启示：把正常统计参照接入弱监督多实例学习，连接正常性建模与判别式检测两条阅读路线。
- 阅读关注：仍依赖视频级标签与预提取特征；正式发表为 TCSVT 2024，而非 CVPR。
- 核验：2026-10-04；[来源 1](<https://ieeexplore.ieee.org/document/10649595/>) · [来源 2](<https://arxiv.org/pdf/2311.15367>) · [来源 3](<https://api.crossref.org/works/10.1109/TCSVT.2024.3450734>)

<a id="paper-pe-mil"></a>

### Prompt-Enhanced Multiple Instance Learning for Weakly Supervised Video Anomaly Detection

**2024 · CVPR** · [paper](<https://openaccess.thecvf.com/content/CVPR2024/html/Chen_Prompt-Enhanced_Multiple_Instance_Learning_for_Weakly_Supervised_Video_Anomaly_Detection_CVPR_2024_paper.html>) · [code](<https://github.com/Junxi-Chen/PE-MIL>) · [引用 / BibTeX](<citations.md#cite-pe-mil>)

**创新：异常语义与正常上下文双提示**

以异常类别标注和可学习提示向视频特征注入语义先验，再用正常上下文提示区分异常动作与周围背景，增强多实例学习对多样异常及事件边界的辨别能力。

- 任务：异常检测、异常定位
- 方法与场景标签：语义提示、正常上下文、多实例学习、事件边界
- 核心启示：连接语义提示与检测定位：既问“是什么异常”，也用正常上下文帮助确定异常发生的范围。
- 阅读关注：训练使用视频级异常类别标注；主要输出为异常评分和定位，不直接生成异常原因解释。
- 核验：2026-10-04；[来源 1](<https://openaccess.thecvf.com/content/CVPR2024/html/Chen_Prompt-Enhanced_Multiple_Instance_Learning_for_Weakly_Supervised_Video_Anomaly_Detection_CVPR_2024_paper.html>) · [来源 2](<https://openaccess.thecvf.com/content/CVPR2024/papers/Chen_Prompt-Enhanced_Multiple_Instance_Learning_for_Weakly_Supervised_Video_Anomaly_Detection_CVPR_2024_paper.pdf>)

<a id="paper-fedvad"></a>

### FedVAD: Enhancing Federated Video Anomaly Detection with GPT-Driven Semantic Distillation

**2024 · ECCV** · [paper](<https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/6981_ECCV_2024_paper.php>) · [project](<https://github.com/Eurekaer/FedVAD>) · [引用 / BibTeX](<citations.md#cite-fedvad>)

**创新：语言语义生成校准与联邦蒸馏**

用大语言模型为公共视频生成并校准语义描述，以视频文本对适配多模态教师，再将语义知识自适应蒸馏进联邦全局检测模型；同时按视觉一致性对客户端分组。

- 任务：异常检测、异常定位
- 方法与场景标签：语义蒸馏、大语言模型、联邦学习、公共视频描述
- 核心启示：展示语言语义怎样通过教师蒸馏改善检测表征，是异常理解能力向联邦检测迁移的相关工作。
- 阅读关注：分别评测无监督和弱监督设置；GPT 参与语义知识构建，不表示部署时逐帧调用 GPT 或直接输出解释。
- 核验：2026-10-04；[来源 1](<https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/6981_ECCV_2024_paper.php>) · [来源 2](<https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/06981.pdf>) · [来源 3](<https://link.springer.com/chapter/10.1007/978-3-031-73668-1_14>)

<a id="paper-pi-vad"></a>

### Just Dance with pi! A Poly-modal Inductor for Weakly-supervised Video Anomaly Detection

**2025 · CVPR** · [paper](<https://openaccess.thecvf.com/content/CVPR2025/html/Majhi_Just_Dance_with_pi_A_Poly-modal_Inductor_for_Weakly-supervised_Video_CVPR_2025_paper.html>) · [code](<https://github.com/snehashismajhi/PI-VAD>) · [引用 / BibTeX](<citations.md#cite-pi-vad>)

**创新：五类模态线索诱导语义检测表征**

训练时利用姿态、深度、全景分割、光流和视觉语言语义五类线索，通过伪模态生成与跨模态诱导补充 RGB 表征，区分外观相似而动作或场景含义不同的异常事件。

- 任务：异常检测、异常定位
- 方法与场景标签：π-VAD、pi-VAD、多模态语义、姿态、场景上下文、知识蒸馏
- 核心启示：以动作、空间、物体上下文和语言语义理解易混淆异常；五个额外模态骨干只在训练时使用，推断保留 RGB 与诱导模块。
- 阅读关注：属于视频级弱监督检测；丰富语义表征并不等同于输出自然语言解释或因果推理。
- 核验：2026-10-04；[来源 1](<https://openaccess.thecvf.com/content/CVPR2025/html/Majhi_Just_Dance_with_pi_A_Poly-modal_Inductor_for_Weakly-supervised_Video_CVPR_2025_paper.html>) · [来源 2](<https://openaccess.thecvf.com/content/CVPR2025/papers/Majhi_Just_Dance_with_pi_A_Poly-modal_Inductor_for_Weakly-supervised_Video_CVPR_2025_paper.pdf>)

<a id="paper-lec-vad"></a>

### Learning Event Completeness for Weakly Supervised Video Anomaly Detection

**2025 · ICML** · [paper](<https://proceedings.mlr.press/v267/wang25l.html>) · [引用 / BibTeX](<citations.md#cite-lec-vad>)

**创新：双路视觉语言语义与完整事件定位**

联合类别感知与类别无关的视觉语言语义分支，以异常感知高斯混合约束事件边界，并通过记忆库原型丰富简短的异常类别文本，缓解弱监督检测只覆盖事件局部的问题。

- 任务：异常检测、异常定位
- 方法与场景标签：视觉语言语义、事件完整性、类别原型、记忆库
- 核心启示：把类别语义与事件时间范围联系起来，从零散高分片段推进到更完整的异常事件定位。
- 阅读关注：正式发表于 ICML 2025；事件完整性关注时间定位，不能据此宣称生成完整故事或解释异常原因。
- 核验：2026-10-04；[来源 1](<https://proceedings.mlr.press/v267/wang25l.html>) · [来源 2](<https://raw.githubusercontent.com/mlresearch/v267/main/assets/wang25l/wang25l.pdf>)

<a id="paper-d2mil"></a>

### Learning from Noisy Supervision: A Denoising-Debiasing Framework for Weakly Supervised Video Anomaly Detection

**2026 · CVPR** · [paper](<https://openaccess.thecvf.com/content/CVPR2026/html/Zhao_Learning_from_Noisy_Supervision_A_Denoising-Debiasing_Framework_for_Weakly_Supervised_CVPR_2026_paper.html>) · [引用 / BibTeX](<citations.md#cite-d2mil>)

**创新：动态去噪与视觉语言语义复核**

先按动态丢弃率筛除训练损失较高的疑似噪声片段，再用冻结视觉语言模型复核这些候选，找回因难以识别而被误删的异常实例，减少弱监督噪声和困难异常之间的混淆。

- 任务：异常检测、异常定位
- 方法与场景标签：D2MIL、弱监督去噪、视觉语言复核、困难异常
- 核心启示：把视觉语言语义判断作为检测训练的复核环节，连接异常内容理解与可靠的弱监督学习。
- 阅读关注：可接入 MIL 检测器的去噪去偏框架；视觉语言复核用于训练样本筛选，不是独立的异常问答或解释任务。
- 核验：2026-10-04；[来源 1](<https://openaccess.thecvf.com/content/CVPR2026/html/Zhao_Learning_from_Noisy_Supervision_A_Denoising-Debiasing_Framework_for_Weakly_Supervised_CVPR_2026_paper.html>) · [来源 2](<https://openaccess.thecvf.com/content/CVPR2026/papers/Zhao_Learning_from_Noisy_Supervision_A_Denoising-Debiasing_Framework_for_Weakly_Supervised_CVPR_2026_paper.pdf>)

<a id="paper-stprompt"></a>

### Weakly Supervised Video Anomaly Detection and Localization with Spatio-Temporal Prompts

**2024 · ACM MM** · [paper](<https://doi.org/10.1145/3664647.3681442>) · [引用 / BibTeX](<citations.md#cite-stprompt>)

**创新：时空提示将语义对齐到异常区域**

以预训练视觉语言模型的语义知识和视频运动先验学习时空提示，在双分支网络中同时进行帧级异常判断与局部异常区域定位，减轻整帧背景信息对检测的干扰。

- 任务：异常检测、异常定位、空间定位
- 方法与场景标签：时空提示、视觉语言对齐、空间定位、运动先验
- 核心启示：把异常类别文本与具体时空区域联系起来，补充“异常在哪里”的视觉证据，适合与 PI-VAD 的多模态语义及 PE-MIL 的上下文提示对照阅读。
- 阅读关注：训练使用视频级标签，不需要精细时空标注或辅助目标检测／跟踪；空间热图提供定位依据，不能等同于生成原因解释。
- 核验：2026-10-04；[来源 1](<https://doi.org/10.1145/3664647.3681442>) · [来源 2](<https://arxiv.org/pdf/2408.05905>) · [来源 3](<https://arxiv.org/abs/2408.05905>) · [来源 4](<https://api.crossref.org/works/10.1145/3664647.3681442>)

<a id="paper-fine-vad"></a>

### Fine-VAD: Towards Fine-Grained Video Anomaly Detection via Progressive Cross-Granularity Learning

**2026 · CVPR** · [paper](<https://openaccess.thecvf.com/content/CVPR2026/html/Zhang_Fine-VAD_Towards_Fine-Grained_Video_Anomaly_Detection_via_Progressive_Cross-Granularity_Learning_CVPR_2026_paper.html>) · [引用 / BibTeX](<citations.md#cite-fine-vad>)

**创新：跨粒度渐进对齐辨别异常类别**

从正常／异常粗粒度监督，经聚类构建的中间伪类别，逐步对齐到细粒度异常类别语义；利用互补监督缓解同类事件跨场景变化及不同异常共享视觉特征导致的混淆。

- 任务：异常检测、异常定位
- 方法与场景标签：细粒度异常识别、类别语义、渐进对齐、多粒度监督
- 核心启示：明确从“是否异常”推进到“是哪类异常”，补充检测线中的类别语义理解，与只报告二值异常分数的方法区分阅读。
- 阅读关注：依赖视频级二值与类别标签；细粒度评测按异常视频上的类别时间定位 mAP，不将该指标与全测试集二值 AUC 直接比较，也不视为开放词汇识别。
- 核验：2026-10-04；[来源 1](<https://openaccess.thecvf.com/content/CVPR2026/html/Zhang_Fine-VAD_Towards_Fine-Grained_Video_Anomaly_Detection_via_Progressive_Cross-Granularity_Learning_CVPR_2026_paper.html>) · [来源 2](<https://openaccess.thecvf.com/content/CVPR2026/papers/Zhang_Fine-VAD_Towards_Fine-Grained_Video_Anomaly_Detection_via_Progressive_Cross-Granularity_Learning_CVPR_2026_paper.pdf>)

## 异常构造与监督

从语言定义、伪标签与伪异常到可控视频生成，为异常检测构造学习信号。

研究问题：异常样本稀缺时，怎样构造异常与监督？

分叉节点：[LAVIDA](<#paper-lavida>)；[分叉依据](<https://arxiv.org/html/2602.19248v4>) — LAVIDA 同时构造异常监督并融合多模态特征进行检测；在此将异常合成阅读支线与继续通往流式检测的阅读线分开，沿用已有双重方法归属，不表示后续论文直接继承。

<a id="paper-ovvad"></a>

### Open-Vocabulary Video Anomaly Detection

**2024 · CVPR** · [paper](<https://openaccess.thecvf.com/content/CVPR2024/html/Wu_Open-Vocabulary_Video_Anomaly_Detection_CVPR_2024_paper.html>) · [引用 / BibTeX](<citations.md#cite-ovvad>)

**创新：语言知识与异常合成**

将开放词汇检测分解为类别无关检测与类别识别，用语言知识和合成未知异常支持未见类别。

- 任务：异常检测
- 兼属方法：视频异常检测；[归类依据](<https://arxiv.org/abs/2311.07042>) — 2026-10-04 按输出任务拆分图中检测阅读线。作者提出语义知识注入与未知异常合成模块，分别支撑检测和类别识别，据此保留表征对齐与监督构造双重归属。图中作为普通衔接站，两色线路连续经过；不表示后续论文直接继承。
- 核心启示：异常检测之外，还需回答未知异常属于什么语义类别。
- 阅读关注：合成异常与真实未见事件之间的差异如何影响识别？
- 核验：2026-09-29；[来源 1](<https://openaccess.thecvf.com/content/CVPR2024/html/Wu_Open-Vocabulary_Video_Anomaly_Detection_CVPR_2024_paper.html>) · [来源 2](<https://openaccess.thecvf.com/content/CVPR2024/papers/Wu_Open-Vocabulary_Video_Anomaly_Detection_CVPR_2024_paper.pdf>)

<a id="paper-tpwng"></a>

### Text Prompt with Normality Guidance for Weakly Supervised Video Anomaly Detection

**2024 · CVPR** · [paper](<https://openaccess.thecvf.com/content/CVPR2024/html/Yang_Text_Prompt_with_Normality_Guidance_for_Weakly_Supervised_Video_Anomaly_CVPR_2024_paper.html>) · [引用 / BibTeX](<citations.md#cite-tpwng>)

**创新：正常引导伪标签学习**

将事件描述与视频帧对齐，结合正常性视觉提示生成帧级伪标签，再进行时序自训练。

- 任务：异常检测、异常定位
- 兼属方法：视频异常检测；[归类依据](<https://arxiv.org/html/2404.08531v1#S3.SS2>) — 2026-10-04 按输出任务拆分图中检测阅读线。TPWNG 的可学习文本提示、正常视觉提示和 CLIP 域适配明确用于改进事件文字与视频帧对齐，据此添加表征对齐次级归属。监督阅读色段在该普通衔接站接回后续表征主干；不声明后续论文直接继承。
- 核心启示：正常性参照可把事件文字转化为更细的弱监督信号。
- 阅读关注：正常性提示与文本对齐误差如何共同影响伪标签？
- 核验：2026-10-02；[来源 1](<https://openaccess.thecvf.com/content/CVPR2024/html/Yang_Text_Prompt_with_Normality_Guidance_for_Weakly_Supervised_Video_Anomaly_CVPR_2024_paper.html>) · [来源 2](<https://arxiv.org/abs/2404.08531>)

<a id="paper-lavida"></a>

### No Need For Real Anomaly: MLLM Empowered Zero-Shot Video Anomaly Detection

**2026 · CVPR** · [paper](<https://openaccess.thecvf.com/content/CVPR2026/html/Dai_No_Need_For_Real_Anomaly_MLLM_Empowered_Zero-Shot_Video_Anomaly_CVPR_2026_paper.html>) · [project](<https://github.com/VitaminCreed/LAVIDA>) · [引用 / BibTeX](<citations.md#cite-lavida>)

**创新：伪异常与反向注意力**

仅用伪异常训练，结合多模态大模型语义理解与反向注意力词元压缩，实现零样本帧级和像素级异常检测。

- 任务：异常检测、异常定位
- 兼属方法：视频异常检测；[归类依据](<https://arxiv.org/html/2602.19248v4#S3.SS7>) — 2026-10-04 按输出任务拆分图中检测阅读线。LAVIDA 第 3.5–3.7 节将 MLLM 异常语义、CLIP 类别文本和视觉特征通过跨模态注意力及多尺度语义投影融合，并投影到掩码解码器空间；据此归入表征对齐与融合。主归属保留异常构造与监督，跨模态融合贡献保留为次级阅读归属；图面在该共享站分出监督构造支线，检测阅读线继续通往 ReactVAU，使用单个普通分支站；不表示直接继承相邻论文。
- 核心启示：可通过合成暴露异常语义，再检验对真实异常的零样本迁移。
- 阅读关注：伪异常覆盖的语义与真实上下文依赖异常有多大差距？
- 核验：2026-10-02；[来源 1](<https://openaccess.thecvf.com/content/CVPR2026/html/Dai_No_Need_For_Real_Anomaly_MLLM_Empowered_Zero-Shot_Video_Anomaly_CVPR_2026_paper.html>) · [来源 2](<https://github.com/VitaminCreed/LAVIDA>) · [来源 3](<https://openaccess.thecvf.com/content/CVPR2026/papers/Dai_No_Need_For_Real_Anomaly_MLLM_Empowered_Zero-Shot_Video_Anomaly_CVPR_2026_paper.pdf>) · [来源 4](<https://arxiv.org/html/2602.19248v4#S3.SS7>)

<a id="paper-anomalycraft"></a>

### AnomalyCraft-700K: Component-Level Controllable and Verifiable Synthetic Anomalies for Fine-Grained Video Anomaly Understanding

**2026 · arXiv** · [paper](<https://arxiv.org/abs/2609.06978>) · [project](<https://github.com/Eagen-l/AnomalyCraft>) · [引用 / BibTeX](<citations.md#cite-anomalycraft>)

**创新：语义组件控制生成与逐项验证**

以细粒度语义组件控制异常视频生成，逐组件校正文图不一致，并构造类别相关的困难正常样本，支持多任务异常理解。

- 任务：异常检测、异常解释、异常推理
- 核心启示：合成数据的价值取决于异常语义可控、标注可验证及正常边界足够困难。
- 阅读关注：生成伪迹、组件覆盖和合成到真实视频的迁移差距。
- 核验：2026-09-30；[来源 1](<https://arxiv.org/abs/2609.06978>) · [来源 2](<https://github.com/Eagen-l/AnomalyCraft>)

<a id="paper-pa-vad"></a>

### PA-VAD: Diffusion-Based Pseudo-Only Video Anomaly Detection via Domain-Aligned Memory Updates

**2026 · ECCV** · [paper](<https://eccv.ecva.net/virtual/2026/poster/4682>) · [引用 / BibTeX](<citations.md#cite-pa-vad>)

**创新：伪异常生成与域对齐记忆**

用少量正常图像合成伪异常视频，与真实正常视频组成训练对；通过域对齐和记忆更新缓解合成异常的特征偏置。

- 任务：异常检测
- 方法与场景标签：伪异常生成、弱监督
- 核心启示：减少真实异常视频依赖时，需要控制合成域与真实域之间的偏差。
- 阅读关注：核查合成异常的伪影、跨场景迁移和开放类别表现。
- 核验：2026-10-01；[来源 1](<https://eccv.ecva.net/virtual/2026/poster/4682>) · [来源 2](<https://arxiv.org/abs/2512.06845>)

<a id="paper-cavge"></a>

### Customized Anomalous Video Generation for Incremental Learning in Weakly-Supervised Video Anomaly Detection

**2026 · ACM MM** · [paper](<https://2026.acmmm.org/site/technical-programme.html>) · [引用 / BibTeX](<citations.md#cite-cavge>)

**创新：脚本—关键帧—视频可控生成**

以语言脚本、文本引导关键帧和视频生成合成可定制异常，并用于弱监督视频异常检测的增量学习。

- 任务：异常检测
- 方法与场景标签：异常视频生成、增量学习
- 核心启示：生成管线可补充增量任务所需的异常类别与多粒度标注。
- 阅读关注：合成视频的真实感、生成偏置及增量遗忘需单独验证。
- 核验：2026-10-01；[来源 1](<https://2026.acmmm.org/site/program-data.json>)

<a id="paper-background-agnostic-vad"></a>

### A Background-Agnostic Framework with Adversarial Training for Abnormal Event Detection in Video

**2022 · TPAMI** · [paper](<https://doi.org/10.1109/tpami.2021.3074805>) · [引用 / BibTeX](<citations.md#cite-background-agnostic-vad>)

**创新：对象级重构与伪异常对抗训练**

围绕检测到的对象建模外观与运动，用域外伪异常对抗训练自编码器和判别器，并补充区域与轨迹级异常标注。

- 任务：异常检测
- 核心启示：为理解异常主体及降低背景干扰提供对象级建模与监督构造基础。
- 阅读关注：跨场景应用要求正常事件定义一致；背景无关不等于任意场景中的语义泛化。
- 核验：2026-10-04；[来源 1](<https://europepmc.org/article/MED/33881990>) · [来源 2](<https://api.crossref.org/works/10.1109/tpami.2021.3074805>)

## 多模态理解与对齐

围绕视觉语言对齐、MLLM 有效词元、隐藏状态与注意力头探测，以及异常语义检索、描述、问答和解释组织阅读。

研究问题：如何把视觉、运动、声音与语言对齐为可检索、可描述、可解释的异常语义？

<a id="paper-ex-vad"></a>

### Ex-VAD: Explainable Fine-grained Video Anomaly Detection Based on Visual-Language Models

**2025 · ICML** · [paper](<https://proceedings.mlr.press/v267/huang25ad.html>) · [引用 / BibTeX](<citations.md#cite-ex-vad>)

**创新：解释融合与标签对齐**

由帧字幕生成视频级异常解释，再结合视觉特征与标签增强对齐进行细粒度检测。

- 任务：异常检测、异常解释
- 兼属方法：视频异常检测；[归类依据](<https://proceedings.mlr.press/v267/huang25ad.html>) — 作者摘要同时明确异常解释与细粒度异常检测；据此作为多模态理解与视频异常检测的共享阅读节点。
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

<a id="paper-va-gpt"></a>

### Aligning Effective Tokens with Video Anomaly in Large Language Models

**2025 · ICCV** · [paper](<https://openaccess.thecvf.com/content/ICCV2025/html/Chen_Aligning_Effective_Tokens_with_Video_Anomaly_in_Large_Language_Models_ICCV_2025_paper.html>) · [引用 / BibTeX](<citations.md#cite-va-gpt>)

**创新：时空有效词元对齐**

通过空间有效词元选择与时间有效词元生成，减少冗余视觉信息，支持异常总结和时间定位。

- 任务：异常定位、异常解释、视频问答
- 兼属方法：时序建模与记忆；[归类依据](<https://openaccess.thecvf.com/content/ICCV2025/papers/Chen_Aligning_Effective_Tokens_with_Video_Anomaly_in_Large_Language_Models_ICCV_2025_paper.pdf>) — 2026-10-04 核对摘要与 §3：Temporal Effective Token Generation 使用逐帧异常置信度在语言空间生成携带异常时间先验的词元，支持时序推理与定位；据此在多模态对齐主归属之外增加时序建模次级归属，作为两条阅读线的共享站。
- 核心启示：进入语言模型的时空证据如何筛选，本身就是异常理解的关键。
- 阅读关注：词元选择保留局部异常时，是否也保留解释所需的上下文？
- 核验：2026-09-29；[来源 1](<https://openaccess.thecvf.com/content/ICCV2025/html/Chen_Aligning_Effective_Tokens_with_Video_Anomaly_in_Large_Language_Models_ICCV_2025_paper.html>) · [来源 2](<https://openaccess.thecvf.com/content/ICCV2025/papers/Chen_Aligning_Effective_Tokens_with_Video_Anomaly_in_Large_Language_Models_ICCV_2025_paper.pdf>)

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

<a id="paper-hiprobe-vad"></a>

### HiProbe-VAD: Video Anomaly Detection via Hidden States Probing in Tuning-Free Multimodal LLMs

**2025 · ACM MM** · [paper](<https://doi.org/10.1145/3746027.3755575>) · [引用 / BibTeX](<citations.md#cite-hiprobe-vad>)

**创新：中间层探测与轻量异常评分**

从冻结多模态大模型的中间隐藏状态选择异常敏感层，训练轻量逻辑回归评分器，并结合时序定位与文本解释分析异常。

- 任务：异常检测、异常定位、异常解释
- 方法与场景标签：冻结MLLM、隐藏状态探测、轻量评分器
- 核心启示：异常证据既可来自生成文本，也可来自大模型内部表征。
- 阅读关注：层选择与评分器使用的标注、数据比例和解释生成流程分别如何影响结果？
- 核验：2026-10-01；[来源 1](<https://arxiv.org/html/2507.17394v1>) · [来源 2](<https://api.crossref.org/works/10.1145/3746027.3755575>)

<a id="paper-varcmp"></a>

### VarCMP: Adapting Cross-Modal Pre-Training Models for Video Anomaly Retrieval

**2025 · AAAI** · [paper](<https://doi.org/10.1609/aaai.v39i8.32909>) · [引用 / BibTeX](<citations.md#cite-varcmp>)

**创新：层级跨模态对齐与异常加权**

将跨模态预训练模型用于长视频异常检索，通过统一层级对齐和异常偏置加权，匹配视频与文本或音频查询。

- 任务：异常检索
- 方法与场景标签：跨模态预训练、视频文本检索、视频音频检索
- 核心启示：异常先验能把跨模态检索注意力集中到长视频中的关键片段。
- 阅读关注：文本检索与音频检索的粒度、候选库和异常先验如何影响召回？
- 核验：2026-10-01；[来源 1](<https://ojs.aaai.org/index.php/AAAI/article/view/32909>) · [来源 2](<https://api.crossref.org/works/10.1609/aaai.v39i8.32909>)

<a id="paper-alan"></a>

### Toward Video Anomaly Retrieval From Video Anomaly Detection: New Benchmarks and Model

**2024 · TIP** · [paper](<https://doi.org/10.1109/tip.2024.3374070>) · [project](<https://github.com/Roc-Ng/VAR>) · [引用 / BibTeX](<citations.md#cite-alan>)

**创新：异常引导采样与跨模态检索**

提出长视频异常检索任务及 UCFCrime-AR、XDViolence-AR，使用异常引导采样、视频提示掩码短语建模和跨模态对齐检索相关视频。

- 任务：异常检索、基准评测
- 方法与场景标签：视频文本检索、视频音频检索、异常采样
- 核心启示：详细文本或同步音频能将异常分析扩展到具体事件的检索。
- 阅读关注：检索目标是完整未裁剪视频，查询相关片段的占比如何影响匹配？
- 核验：2026-10-01；[来源 1](<https://arxiv.org/html/2307.12545v2>) · [来源 2](<https://github.com/Roc-Ng/VAR>) · [来源 3](<https://api.crossref.org/works/10.1109/tip.2024.3374070>)

<a id="paper-anomaly-ov"></a>

### Towards Zero-Shot Anomaly Detection and Reasoning with Multimodal Large Language Models

**2025 · CVPR** · [paper](<https://openaccess.thecvf.com/content/CVPR2025/html/Xu_Towards_Zero-Shot_Anomaly_Detection_and_Reasoning_with_Multimodal_Large_Language_CVPR_2025_paper.html>) · [code](<https://github.com/honda-research-institute/Anomaly-OneVision>) · [project](<https://xujiacong.github.io/Anomaly-OV/>) · [引用 / BibTeX](<citations.md#cite-anomaly-ov>)

**创新：二次特征匹配与异常视觉token筛选**

通过 Look-Twice Feature Matching 学习异常表征并突出可疑视觉 token，结合异常指令微调生成缺陷描述、可能原因和改进建议；配套 Anomaly-Instruct-125k 与 VisA-D&R。

- 任务：异常检测、异常解释、异常推理
- 方法与场景标签：工业图像、异常问答、视觉指令微调、视觉token筛选
- 核心启示：细粒度视觉证据的选择与表征可以直接服务于异常解释。
- 阅读关注：零样本指目标类别／数据集设置，模型仍经过异常监督与指令微调；生成的原因和建议不等同于已验证因果。
- 核验：2026-10-03；[来源 1](<https://openaccess.thecvf.com/content/CVPR2025/html/Xu_Towards_Zero-Shot_Anomaly_Detection_and_Reasoning_with_Multimodal_Large_Language_CVPR_2025_paper.html>) · [来源 2](<https://openaccess.thecvf.com/content/CVPR2025/papers/Xu_Towards_Zero-Shot_Anomaly_Detection_and_Reasoning_with_Multimodal_Large_Language_CVPR_2025_paper.pdf>) · [来源 3](<https://github.com/honda-research-institute/Anomaly-OneVision>)

<a id="paper-echotraffic"></a>

### EchoTraffic: Enhancing Traffic Anomaly Understanding with Audio-Visual Insights

**2025 · CVPR** · [paper](<https://openaccess.thecvf.com/content/CVPR2025/html/Xing_EchoTraffic_Enhancing_Traffic_Anomaly_Understanding_with_Audio-Visual_Insights_CVPR_2025_paper.html>) · [code](<https://github.com/HarryHsing/EchoTraffic>) · [引用 / BibTeX](<citations.md#cite-echotraffic>)

**创新：声音引导选帧与音视频动态融合**

用声音变化引导关键帧选择，再经动态连接器融合音视频信息进行交通异常问答；构建 AV-TAU，覆盖事件描述、原因、时段、预防与响应五项任务。

- 任务：异常定位、异常解释、异常推理、视频问答
- 方法与场景标签：道路交通、音视频融合、声音引导采样、事故理解
- 核心启示：碰撞声等听觉线索能补充视野之外或视觉不清晰的异常证据。
- 阅读关注：预防建议与事故提前预测是不同任务；应核查音频质量和时间定位误差对解释的影响。
- 核验：2026-10-03；[来源 1](<https://openaccess.thecvf.com/content/CVPR2025/html/Xing_EchoTraffic_Enhancing_Traffic_Anomaly_Understanding_with_Audio-Visual_Insights_CVPR_2025_paper.html>) · [来源 2](<https://openaccess.thecvf.com/content/CVPR2025/papers/Xing_EchoTraffic_Enhancing_Traffic_Anomaly_Understanding_with_Audio-Visual_Insights_CVPR_2025_paper.pdf>) · [来源 3](<https://github.com/HarryHsing/EchoTraffic>)

<a id="paper-ssmctb"></a>

### Self-Supervised Masked Convolutional Transformer Block for Anomaly Detection

**2024 · TPAMI** · [paper](<https://doi.org/10.1109/tpami.2023.3322604>) · [引用 / BibTeX](<citations.md#cite-ssmctb>)

**创新：掩码卷积与通道注意力自监督**

在网络内部用掩码卷积、通道 Transformer 与 Huber 自监督目标重构被遮蔽信息，可接入图像和视频异常检测网络。

- 任务：异常检测
- 核心启示：把正常模式重构约束下沉到可复用的表征模块，支持 RGB 与热成像视频等任务。
- 阅读关注：模块级检测性能提升不代表已有语言解释能力；与工业异常问答方法分开阅读。
- 核验：2026-10-04；[来源 1](<https://arxiv.org/abs/2209.12148>) · [来源 2](<https://api.crossref.org/works/10.1109/tpami.2023.3322604>)

<a id="paper-anomalygpt"></a>

### AnomalyGPT: Detecting Industrial Anomalies Using Large Vision-Language Models

**2024 · AAAI** · [paper](<https://ojs.aaai.org/index.php/AAAI/article/view/27963>) · [引用 / BibTeX](<citations.md#cite-anomalygpt>)

**创新：缺陷定位特征与语言提示对齐**

用合成缺陷图像和对应描述构造监督，通过细粒度视觉语言解码器产生定位特征，再以可学习提示接入大视觉语言模型，支持缺陷判断、定位和多轮交互。

- 任务：异常检测、空间定位、异常解释
- 方法与场景标签：工业图像、视觉语言对齐、多轮对话、少样本迁移
- 核心启示：先让局部缺陷进入语言模型可用的表示，才能支撑图像异常判断与交互描述。
- 阅读关注：工业图像的定位和问答表现不能直接外推到视频时序理解；少样本迁移与数据集内训练须分开比较。
- 核验：2026-10-04；[来源 1](<https://ojs.aaai.org/index.php/AAAI/article/view/27963>) · [来源 2](<https://ojs.aaai.org/index.php/AAAI/article/download/27963/27945>)

## 时序建模与记忆

创新：场景条件前后向预测、姿态与轨迹序列建模、事件边界、多粒度时间组织与在线记忆。

研究问题：怎样建模异常随时间的变化，并组织事件边界与历史信息？

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
- 兼属方法：视频异常检测；[归类依据](<https://arxiv.org/html/2609.07941v2#S3.SS2>) — 2026-10-04 核对 §3.2、§3.4 与 Figure 2：Spatial Grid Folding 快速模块连续输出异常分数，慢速语义验证分数与其融合用于最终检测；AAPM 保留异常证据。因此保留时序记忆主归属，并增加视频异常检测次级归属，连接检测与记忆两条阅读线。
- 核心启示：流式理解需要同时控制证据遗忘和重型模型调用频率。
- 阅读关注：分别检查触发延迟、异常漏检与记忆保真度，不能只比较离线检测分数。
- 核验：2026-09-30；[来源 1](<https://arxiv.org/abs/2609.07941>)

<a id="paper-step-vad"></a>

### STEP: Score-Based Temporal Energy for Human Pose Video Anomaly Detection

**2026 · ECCV** · [paper](<https://eccv.ecva.net/virtual/2026/poster/3512>) · [引用 / BibTeX](<citations.md#cite-step-vad>)

**创新：姿态主成分空间的时序能量**

把连续人体姿态投影到白化主成分空间，利用去噪分数匹配学习正常序列能量，并用姿态置信度降低估计噪声影响。

- 任务：异常检测
- 方法与场景标签：人体姿态、能量模型
- 核心启示：在姿态表示空间建模噪声可提升长时序正常性估计的稳定性。
- 阅读关注：核查姿态估计失败、非人体异常和实时性能的适用范围。
- 核验：2026-10-01；[来源 1](<https://eccv.ecva.net/virtual/2026/poster/3512>) · [来源 2](<https://arxiv.org/abs/2608.19987>)

<a id="paper-trajvad"></a>

### Bounding-Box Trajectories Matter for Video Anomaly Detection

**2026 · ECCV** · [paper](<https://eccv.ecva.net/virtual/2026/poster/5841>) · [code](<https://github.com/Songinpyo/TrajVAD-ECCV2026>) · [引用 / BibTeX](<citations.md#cite-trajvad>)

**创新：轨迹流模型与姿态可靠门控**

以目标框轨迹作为主要异常线索，用归一化流学习正常运动分布；另以可靠性门控结合人体姿态。

- 任务：异常检测
- 方法与场景标签：目标轨迹、人体姿态
- 核心启示：目标框轨迹本身提供可与姿态互补的运动异常证据。
- 阅读关注：评估轨迹跟踪错误、非人体异常与跨视角泛化。
- 核验：2026-10-01；[来源 1](<https://eccv.ecva.net/virtual/2026/poster/5841>) · [来源 2](<https://arxiv.org/abs/2605.21957>)

<a id="paper-peer-vad"></a>

### PEER-VAD: Prior-enhanced Event Refinement for Video Anomaly Detection

**2026 · ACM MM** · [paper](<https://2026.acmmm.org/site/technical-programme.html>) · [code](<https://github.com/HaochengY/PEER-VAD>) · [引用 / BibTeX](<citations.md#cite-peer-vad>)

**创新：先验消歧与双向事件细化**

对大模型给出的帧级或短片段异常预测做先验消歧和双向窗口分析，修复碎片化事件预测。

- 任务：异常检测、异常定位
- 核心启示：事后事件一致性处理可改善免训练大模型检测器的定位。
- 阅读关注：查看上游预测错误是否会被双向窗口传播。
- 核验：2026-10-01；[来源 1](<https://2026.acmmm.org/site/program-data.json>)

<a id="paper-s2mgraph-vad"></a>

### S2MGraph-VAD: Scene-to-Moment Graph-Guided Training-Free Video Anomaly Detection

**2026 · ACM MM** · [paper](<https://2026.acmmm.org/site/technical-programme.html>) · [引用 / BibTeX](<citations.md#cite-s2mgraph-vad>)

**创新：场景—片段图与稀疏查询**

先把长视频划分为场景，再细化到短时片段，通过场景到片段的图连接上下文与局部异常线索，稀疏调用大模型定位。

- 任务：异常检测、异常定位
- 核心启示：多尺度结构有助于保留被长视频正常内容淹没的短暂异常。
- 阅读关注：检查场景切分和图构造错误对定位的影响。
- 核验：2026-10-01；[来源 1](<https://2026.acmmm.org/site/program-data.json>)

<a id="paper-nwpu-campus-paper"></a>

### A New Comprehensive Benchmark for Semi-Supervised Video Anomaly Detection and Anticipation

**2023 · CVPR** · [paper](<https://openaccess.thecvf.com/content/CVPR2023/html/Cao_A_New_Comprehensive_Benchmark_for_Semi-Supervised_Video_Anomaly_Detection_and_CVPR_2023_paper.html>) · [project](<https://campusvad.github.io/>) · [code](<https://github.com/zugexiaodui/campus_vad_code>) · [引用 / BibTeX](<citations.md#cite-nwpu-campus-paper>)

**创新：场景条件前后向帧预测**

提出 NWPU Campus 基准，覆盖随场景规则改变正常性的行为；以前后向场景条件帧预测统一异常检测与提前预判。

- 任务：异常检测、异常预判、基准评测
- 方法与场景标签：场景依赖异常、前后向预测
- 核心启示：同一种行为的异常性取决于场景；预测未来异常需要结合事件动态与场景条件。
- 阅读关注：检测与预判以预测误差为依据，尚不输出自然语言原因解释。
- 核验：2026-10-04；[来源 1](<https://openaccess.thecvf.com/content/CVPR2023/papers/Cao_A_New_Comprehensive_Benchmark_for_Semi-Supervised_Video_Anomaly_Detection_and_CVPR_2023_paper.pdf>) · [来源 2](<https://campusvad.github.io/>)

<a id="paper-scene-dependent-vaa"></a>

### Scene-Dependent Prediction in Latent Space for Video Anomaly Detection and Anticipation

**2025 · TPAMI** · [paper](<https://doi.org/10.1109/tpami.2024.3461718>) · [project](<https://campusvaa.github.io/>) · [引用 / BibTeX](<citations.md#cite-scene-dependent-vaa>)

**创新：场景依赖潜空间双向预测**

扩展 NWPU Campus 工作，以层次 VAE、潜空间扩散和场景信息自编码器建模事件与场景关系，并用关键帧时序损失约束运动一致性。

- 任务：异常检测、异常预判
- 方法与场景标签：场景依赖异常、潜空间扩散
- 核心启示：从场景条件像素预测推进到潜空间建模，继续联合研究异常检测与提前预判。
- 阅读关注：场景依赖预测支持异常判断，但不等同于可直接检验的语言解释。
- 核验：2026-10-04；[来源 1](<https://ieeexplore.ieee.org/abstract/document/10681297/>) · [来源 2](<https://campusvaa.github.io/>)

<a id="paper-video-behavior-profiling"></a>

### Video Behavior Profiling for Anomaly Detection

**2008 · TPAMI** · [paper](<https://doi.org/10.1109/tpami.2007.70731>) · [引用 / BibTeX](<citations.md#cite-video-behavior-profiling>)

**创新：动态贝叶斯行为建模与谱聚类**

从监控视频中的离散事件表示行为，用动态贝叶斯网络比较行为序列，并以谱聚类学习正常模式以支持在线异常检测。

- 任务：异常检测
- 核心启示：为“异常相对于哪些已学习行为模式”提供早期统计建模视角。
- 阅读关注：基于固定场景行为分布的异常判别，需要与语义解释和因果理解区分。
- 核验：2026-10-04；[来源 1](<https://www.eecs.qmul.ac.uk/~sgg/papers/XiangGong_PAMI08.pdf>) · [来源 2](<https://api.crossref.org/works/10.1109/tpami.2007.70731>)

<a id="paper-scene-dynamics"></a>

### Probabilistic Modeling of Scene Dynamics for Applications in Visual Surveillance

**2009 · TPAMI** · [paper](<https://doi.org/10.1109/tpami.2008.175>) · [引用 / BibTeX](<citations.md#cite-scene-dynamics>)

**创新：轨迹密度建模与场景动态推断**

以静态摄像机中的对象轨迹学习位置与转移时间的非参数概率模型，统一支持轨迹生成、持续跟踪与异常运动检测。

- 任务：异常检测
- 核心启示：把场景中的常见运动规律显式表示为可查询的时空分布。
- 阅读关注：属于轨迹层面的场景建模基础；轨迹低概率不直接等于事故原因解释。
- 核验：2026-10-04；[来源 1](<https://www.crcv.ucf.edu/papers/TPAMI_scene_dynamics.pdf>) · [来源 2](<https://api.crossref.org/works/10.1109/tpami.2008.175>)

<a id="paper-crowded-scenes-tpami"></a>

### Anomaly Detection and Localization in Crowded Scenes

**2014 · TPAMI** · [paper](<https://doi.org/10.1109/tpami.2013.111>) · [引用 / BibTeX](<citations.md#cite-crowded-scenes-tpami>)

**创新：动态纹理联合外观与运动异常**

利用动态纹理联合建模外观与运动，并结合时间和空间异常证据，在拥挤监控场景中检测与定位异常。

- 任务：异常检测、异常定位、空间定位
- 核心启示：为异常理解中的“何时、何处出现偏离”提供经典时空检测基础。
- 阅读关注：外观／运动异常定位属于证据层，不能直接替代事件语义或原因回答。
- 核验：2026-10-04；[来源 1](<https://www.svcl.ucsd.edu/publications/journal/2013/pami.anomaly/pami_anomaly.pdf>) · [来源 2](<https://api.crossref.org/works/10.1109/tpami.2013.111>)

<a id="paper-shnn-cad"></a>

### Online Learning and Sequential Anomaly Detection in Trajectories

**2014 · TPAMI** · [paper](<https://doi.org/10.1109/tpami.2013.172>) · [引用 / BibTeX](<citations.md#cite-shnn-cad>)

**创新：序贯轨迹共形异常检测**

以 Hausdorff 近邻距离和共形校准分析尚未完成的轨迹，支持训练集增量更新与在线异常报警。

- 任务：异常检测
- 核心启示：补充部分轨迹的在线判别与报警阈值校准背景。
- 阅读关注：研究对象是轨迹序列，作为视觉轨迹分析的相关基础阅读，不视为视频语义理解方法。
- 核验：2026-10-04；[来源 1](<https://europepmc.org/article/MED/26353278>) · [来源 2](<https://api.crossref.org/works/10.1109/tpami.2013.172>)

<a id="paper-mdi"></a>

### Detecting Regions of Maximal Divergence for Spatio-Temporal Anomaly Detection

**2019 · TPAMI** · [paper](<https://doi.org/10.1109/tpami.2018.2823766>) · [引用 / BibTeX](<citations.md#cite-mdi>)

**创新：分布散度定位连续时空异常区域**

以无偏 KL 散度比较候选区域与其余数据，搜索连续异常时间段和空间区域，并用区间提议提高大规模搜索效率。

- 任务：异常检测、异常定位、空间定位
- 核心启示：补充从孤立异常点转向连续事件区域的统计检测视角。
- 阅读关注：方法跨视频、气候及文本数据；在本目录作为视频异常定位基础阅读。
- 核验：2026-10-04；[来源 1](<https://arxiv.org/abs/1804.07091>) · [来源 2](<https://api.crossref.org/works/10.1109/tpami.2018.2823766>)

<a id="paper-sparse-coding-vad"></a>

### Video Anomaly Detection with Sparse Coding Inspired Deep Neural Networks

**2021 · TPAMI** · [paper](<https://doi.org/10.1109/tpami.2019.2944377>) · [引用 / BibTeX](<citations.md#cite-sparse-coding-vad>)

**创新：时序稀疏编码展开为循环网络**

将时序一致稀疏编码的迭代求解展开为堆叠循环网络，并通过自编码结构联合学习表征和重构，用于视频异常检测。

- 任务：异常检测
- 核心启示：提供从可解释的稀疏优化结构到时序深度模型的连接。
- 阅读关注：采用 TPAMI 2021 正式卷期，不能因 DOI 含 2019 而记为 2019，也不受 2021 年会议论文取舍影响。
- 核验：2026-10-04；[来源 1](<https://xlearning-lab.com/assets/2019-TPAMI-Video-Anomaly-Detection-With-Sparse-Coding-Inspired-Deep-Neural-Networks.pdf>) · [来源 2](<https://api.crossref.org/works/10.1109/tpami.2019.2944377>)

<a id="paper-future-frame-vad"></a>

### Future Frame Prediction Network for Video Anomaly Detection

**2022 · TPAMI** · [paper](<https://doi.org/10.1109/tpami.2021.3129349>) · [引用 / BibTeX](<citations.md#cite-future-frame-vad>)

**创新：外观运动约束预测与跨场景适配**

以外观和运动约束设计未来帧预测网络，并研究元学习使预测模型用少量起始帧适配新测试场景。

- 任务：异常检测
- 核心启示：将正常事件的可预测性用于异常判断，并关注跨场景适配。
- 阅读关注：预测误差用于检测已观测帧的异常，不据此宣称事故发生前的提前预警或因果解释。
- 核验：2026-10-04；[来源 1](<https://europepmc.org/article/MED/34797762>) · [来源 2](<https://api.crossref.org/works/10.1109/tpami.2021.3129349>)

<a id="paper-dota-paper"></a>

### DoTA: Unsupervised Detection of Traffic Anomaly in Driving Videos

**2023 · TPAMI** · [paper](<https://doi.org/10.1109/tpami.2022.3150763>) · [project](<https://github.com/MoonBlvd/Detection-of-Traffic-Anomaly>) · [引用 / BibTeX](<citations.md#cite-dota-paper>)

**创新：自车运动与对象轨迹预测检测**

提出驾驶视频异常数据 DoTA，以时间、对象框和类别描述异常，并结合自车运动与对象轨迹预测进行无监督检测，使用 STAUC 联合衡量时空定位。

- 任务：异常检测、异常定位、空间定位、基准评测
- 核心启示：把道路异常从单一分数推进到“何时、何处、什么事件”，为后续事故理解提供细粒度证据。
- 阅读关注：DoTA 的时空和类别标注不等同于原因解释；预测式检测也不等同于发生前预警。正式 TPAMI 卷期为 2023 年 1 月。
- 核验：2026-10-04；[来源 1](<https://vision.soic.indiana.edu/papers/dota2022pami.pdf>) · [来源 2](<https://api.crossref.org/works/10.1109/tpami.2022.3150763>) · [来源 3](<https://github.com/MoonBlvd/Detection-of-Traffic-Anomaly>)

<a id="paper-where-what"></a>

### Where and What: Contextual Dynamics-Aware Anomaly Detection in Surveillance Videos

**2025 · TIP** · [paper](<https://doi.org/10.1109/TIP.2025.3623392>) · [引用 / BibTeX](<citations.md#cite-where-what>)

**创新：场景与原子动作的时序关系建模**

用轻量场景分类器提供位置上下文，以原子动作特征刻画事件，通过时序语义关系网络 TSRN 联合建模多模态特征，并用片段选择的焦点间隔损失缓解类别不平衡。

- 任务：异常检测、异常定位
- 方法与场景标签：场景上下文、原子动作、弱监督学习
- 核心启示：异常判据取决于动作及其发生场景，显式的环境上下文有助于减少误报。
- 阅读关注：核心输出仍是异常检测结果，不能将其直接视为开放式原因解释或异常问答系统。
- 核验：2026-10-04；[来源 1](<https://doi.org/10.1109/TIP.2025.3623392>) · [来源 2](<https://scholar.korea.ac.kr/item/26547082-0865-4289-9181-69f54af7cece>)

## 主动观察与工具决策

创新：场景规划、工具调用、补充采样与交互策略学习，让异常证据获取随当前疑点调整。

研究问题：当前证据不足时，怎样决定下一次观察或工具调用？

<a id="paper-panda"></a>

### PANDA: Towards Generalist Video Anomaly Detection via Agentic AI Engineer

**2025 · NeurIPS** · [paper](<https://arxiv.org/abs/2509.26386>) · [project](<https://github.com/showlab/PANDA>) · [引用 / BibTeX](<citations.md#cite-panda>)

**创新：场景规划、工具反思与长短时记忆**

把场景规划、工具反思与长短时记忆链组合为通用异常检测智能体：短期记忆保留近期视觉／文本上下文，长期记忆积累推理和反思轨迹，并检索历史案例辅助工具决策。

- 任务：异常检测、异常推理
- 兼属方法：时序建模与记忆；[归类依据](<https://arxiv.org/html/2509.26386v2#S3.SS4>) — §3.4 与 Figure 3：Short CoM 保存局部视觉／文本推理及近期反思，Long CoM 累积各时刻的初始推理、反思和修正结果；§3.3 从长期记忆检索相似反思案例。该记忆链直接支持跨片段判断和工具反思，兼属时序建模与记忆。
- 核心启示：按当前疑点获取证据，并用跨片段的记忆链维持推理与反思的一致性。
- 阅读关注：工具调用、记忆和检测质量之间的成本收益。
- 核验：2026-10-04；[来源 1](<https://arxiv.org/abs/2509.26386>) · [来源 2](<https://github.com/showlab/PANDA>) · [来源 3](<https://arxiv.org/html/2509.26386v2#S3.SS4>)

<a id="paper-anom-pi"></a>

### Learning to Watch: Active Video Anomaly Understanding via Interleaved Policy Optimization

**2026 · ICML** · [paper](<https://arxiv.org/abs/2607.00622>) · [引用 / BibTeX](<citations.md#cite-anom-pi>)

**创新：交替推理与观察策略**

将推理与时间回溯、区间扩展、细粒度采样交替执行，学习主动获取证据的策略。

- 任务：异常推理、异常解释
- 兼属方法：结构化推理与验证；[归类依据](<https://arxiv.org/html/2607.00622v1#S3.SS2>) — 方法 3.1–3.2 显式维护结构化假设 hyp\_n，并以 Think 整合新证据、修正假设，通过推理—观察—推理的闭环验证异常；同时归属结构化推理与主动证据获取。
- 核心启示：“下一步观察什么”也可以成为 VAU 的学习对象。
- 阅读关注：证据收益、任务成功和交互成本是否被共同衡量？
- 核验：2026-10-01；[来源 1](<https://arxiv.org/abs/2607.00622>) · [来源 2](<https://icml.cc/Downloads/2026>) · [来源 3](<https://arxiv.org/html/2607.00622v1#S3.SS2>)

<a id="paper-memovad"></a>

### MemoVAD: Resource-Efficient Video Anomaly Detection via Dynamic Semantic Memory in Edge Computing Scenarios

**2026 · IJCAI** · [paper](<https://www.ijcai.org/proceedings/2026/618>) · [引用 / BibTeX](<citations.md#cite-memovad>)

**创新：不确定性门控与动态语义记忆**

边缘轻量检测器维护因果时序上下文，仅对高不确定且语义新颖的片段查询云端 VLM，并缓存验证后的语义原型用于后续检索。

- 任务：异常检测
- 方法与场景标签：流式异常检测、选择性模型调用、语义记忆
- 兼属方法：时序建模与记忆；[归类依据](<https://www.ijcai.org/proceedings/2026/618>) — 编辑归类：不确定性门控决定云端 VLM 调用，动态语义记忆缓存已验证原型；兼属主动决策与时序记忆。
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

<a id="paper-vto"></a>

### VTO: Visual Tool Orchestration for Video Anomaly Detection

**2026 · ACM MM** · [paper](<https://2026.acmmm.org/site/technical-programme.html>) · [code](<https://github.com/MICLAB-BUPT/VTO>) · [引用 / BibTeX](<citations.md#cite-vto>)

**创新：视觉工具编排与过程奖励**

让视频异常代理动态选择视觉工具，并用过程监督的强化学习及认知评价器引导多步证据获取。

- 任务：异常检测、异常推理
- 方法与场景标签：工具调用、强化学习
- 核心启示：异常推理的工具调用顺序可通过中间步骤反馈优化。
- 阅读关注：分析工具调用成本及过程奖励与真实检测收益的一致性。
- 核验：2026-10-01；[来源 1](<https://2026.acmmm.org/site/program-data.json>)

<a id="paper-vibes"></a>

### Zoom In, Reason Out: Efficient Far-field Anomaly Detection in Expressway Surveillance Videos via Focused VLM Reasoning Guided by Bayesian Inference

**2026 · ACM MM** · [paper](<https://2026.acmmm.org/site/technical-programme.html>) · [code](<https://github.com/maoxiaowei97/VIBES>) · [引用 / BibTeX](<citations.md#cite-vibes>)

**创新：贝叶斯轨迹触发与局部推理**

从高速公路车辆轨迹建立在线正常运动分布，贝叶斯偏离触发远景局部区域的视觉语言推理。

- 任务：异常检测、异常解释、异常定位
- 方法与场景标签：交通监控、远景异常
- 核心启示：先用廉价的运动证据触发，再对远景目标局部放大推理。
- 阅读关注：检查跟踪误差和触发阈值对漏检与延迟的影响。
- 核验：2026-10-01；[来源 1](<https://2026.acmmm.org/site/program-data.json>)

<a id="paper-seek-vau"></a>

### SEEK-VAU: Towards Evidence-Faithful Video Anomaly Understanding via Agentic Search

**2026 · NeurIPS** · [paper](<https://neurips.cc/virtual/2026/poster/152176>) · [引用 / BibTeX](<citations.md#cite-seek-vau>)

**创新：智能体搜索与证据忠实性**

依据题名，暂将其归为通过智能体搜索获取证据的视频异常理解方法。

- 任务：异常解释
- 方法与场景标签：SeekVAU、Agentic Search
- 归类状态：按题名暂定；[依据](<https://neurips.cc/Downloads/2026>) — 按题名暂定归类；方法细节、训练设置与实验结果待摘要或正文核验。
- 核心启示：待正文核验：搜索空间、证据忠实性的定义与验证方式。
- 阅读关注：当前方法归类仅依据官方题名；作者、摘要与完整引用元数据待补。
- 核验：2026-10-01；[来源 1](<https://neurips.cc/Downloads/2026>)

<a id="paper-adseeker"></a>

### ADSeeker: A Knowledge-Grounded Reasoning Framework for Industry Anomaly Detection and Reasoning

**2026 · CVPR** · [paper](<https://openaccess.thecvf.com/content/CVPR2026/html/Zhang_ADSeeker_A_Knowledge-Grounded_Reasoning_Framework_for_Industry_Anomaly_Detection_and_CVPR_2026_paper.html>) · [引用 / BibTeX](<citations.md#cite-adseeker>)

**创新：图像查询驱动的领域知识检索**

构建图文知识库 SEEK-M&V，以 Q2K RAG 将查询图像关联到领域文档；结合层次稀疏提示和缺陷类型特征，为异常定位、判断与解释提供视觉和知识证据，并提出 MulA 数据。

- 任务：异常检测、空间定位、异常解释、异常推理
- 方法与场景标签：工业图像、领域知识、知识检索、医学图像
- 核心启示：把可检索的领域图文依据接入异常解释，补充仅靠模型内部知识作答的路径。
- 阅读关注：方法依赖知识库与异常专家；检索命中及分类准确率不能单独证明解释忠实性，工业和医学场景应分别评估。
- 核验：2026-10-04；[来源 1](<https://openaccess.thecvf.com/content/CVPR2026/html/Zhang_ADSeeker_A_Knowledge-Grounded_Reasoning_Framework_for_Industry_Anomaly_Detection_and_CVPR_2026_paper.html>) · [来源 2](<https://openaccess.thecvf.com/content/CVPR2026/papers/Zhang_ADSeeker_A_Knowledge-Grounded_Reasoning_Framework_for_Industry_Anomaly_Detection_and_CVPR_2026_paper.pdf>)

## 评测与任务拓展

从异常问答、关键视觉事实到细粒度语义与交通任务，检验异常理解的能力边界。

研究问题：怎样设计任务与评价，让异常理解能力可检验？

分叉节点：[Vad-R1](<#paper-vad-r1>)；[分叉依据](<https://papers.nips.cc/paper_files/paper/2025/hash/abccc325c84dedf23dbe8de3f686c733-Abstract-Conference.html>) — Vad-R1 同时提出 P2C-CoT 感知认知推理链、自验证强化学习 AVA-GRPO 与 Vad-Reasoning 数据；据此连接结构化推理与任务评测两个阅读方向，支线首站为 Cue-R1。线路表示方法与任务贡献的阅读关联，不表示后续论文直接继承 Vad-R1。

<a id="paper-finevau"></a>

### FineVAU: A Novel Human-Aligned Benchmark for Fine-Grained Video Anomaly Understanding

**2026 · AAAI** · [paper](<https://arxiv.org/abs/2601.17258>) · [project](<https://finevau.github.io/>) · [引用 / BibTeX](<citations.md#cite-finevau>)

**创新：关键视觉事实评估**

围绕事件、参与者与位置构建细粒度标注，并用 FVScore 检查关键视觉信息。

- 任务：异常解释
- 核心启示：流畅的描述不等于抓住异常；评估需要追问关键事实。
- 阅读关注：自动扩展标注的质量及其与人类判断的一致性。
- 核验：2026-10-02；[来源 1](<https://arxiv.org/abs/2601.17258>) · [来源 2](<https://ojs.aaai.org/index.php/AAAI/article/view/37790>)

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

<a id="paper-tau-bench"></a>

### TAU-Bench: From Anomaly Instance Tracking to Fine-Grained Video Anomaly Understanding

**2026 · arXiv** · [paper](<https://arxiv.org/abs/2608.05699>) · [project](<https://yarkupa.github.io/tau-bench.github.io/>) · [引用 / BibTeX](<citations.md#cite-tau-bench>)

**创新：实例轨迹与层级语义联合评估**

把异常实例轨迹、像素掩码与实例、事件、场景三级描述绑定，联合评估跟踪和细粒度理解是否指向同一异常对象。

- 任务：异常定位、异常解释、异常推理、基准评测
- 核心启示：合理的异常描述仍可能对应错误对象，理解评估需要实例级视觉依据。
- 阅读关注：语义评分与跟踪指标是否同时改善，而非只生成更流畅的描述？
- 核验：2026-09-30；[来源 1](<https://arxiv.org/abs/2608.05699>) · [来源 2](<https://yarkupa.github.io/tau-bench.github.io/>)

<a id="paper-cg-coe"></a>

### Towards Trustworthy Video Anomaly Understanding: A Class-Guided Chain-of-Evaluation Metric and An Anomaly-focused Meta-Benchmark

**2026 · ICML** · [paper](<https://icml.cc/virtual/2026/poster/66013>) · [引用 / BibTeX](<citations.md#cite-cg-coe>)

**创新：类别引导评价链与元评测**

以类别约束的异常事件抽取与匹配构建评价链，并用 AEA 与 CVP 子集检验指标有效性及措辞扰动鲁棒性。

- 任务：异常解释、基准评测
- 兼属方法：结构化推理与验证；[归类依据](<https://icml.cc/virtual/2026/poster/66013>) — 2026-10-02 核对 ICML 官方摘要：CG-CoE 将评估组织为异常事件抽取与类别特定语义容差下的匹配链，显式核验解释中的异常语义，并用 AEA/CVP 检验评价有效性与措辞鲁棒性。据此作为结构化验证与任务评测的共享阅读节点；其主归属仍为评测，不将该评价指标描述为视频异常推理模型。
- 核心启示：把异常语义正确性与措辞风格分开。
- 阅读关注：评价有效性仍需检验类别边界和事件抽取是否可靠；元评测不直接提升模型感知。
- 核验：2026-10-02；[来源 1](<https://icml.cc/virtual/2026/poster/66013>) · [来源 2](<https://openreview.net/forum?id=7waVdY1WmW>) · [来源 3](<https://icml.cc/virtual/2026/poster/66013>)

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

<a id="paper-tar-bench"></a>

### From Detection to Understanding — A Multi-Task Dataset for Traffic Anomaly Reasoning

**2026 · NeurIPS Evaluations and Datasets** · [paper](<https://neurips.cc/virtual/2026/poster/139099>) · [引用 / BibTeX](<citations.md#cite-tar-bench>)

**创新：交通异常多任务推理基准**

面向交通异常构建训练集 TAR 与人工核验的 TAR-Bench，覆盖问答、时序推理及场景理解等十类任务。

- 任务：异常解释、异常推理、异常定位、视频问答、基准评测
- 方法与场景标签：交通异常、评测基准
- 核心启示：问答正确率不能替代事件定位与原因推理的联合评测。
- 阅读关注：核查来自不同交通视频数据源的分布偏差和推理标注质量。
- 核验：2026-10-01；[来源 1](<https://neurips.cc/virtual/2026/poster/139099>) · [来源 2](<https://arxiv.org/abs/2608.10317>) · [来源 3](<https://huggingface.co/datasets/nvidia/PhysicalAI-Traffic-Anomaly-Reasoning>)

<a id="paper-ecva-anomshield"></a>

### Exploring What Why and How: A Multifaceted Benchmark for Causation Understanding of Video Anomaly

**2026 · IJCV** · [paper](<https://link.springer.com/article/10.1007/s11263-026-02983-0>) · [project](<https://github.com/Dulpy/ECVA>) · [引用 / BibTeX](<citations.md#cite-ecva-anomshield>)

**创新：因果任务基准与关键片段推理**

围绕异常事件、原因与后果构建 ECVA 基准；AnomShield 通过思维链选取关键时段并建模时空依赖，AnomEval 用于评估异常理解回答。

- 任务：异常解释、异常推理、视频问答、基准评测
- 方法与场景标签：因果理解、思维链、人类对齐评估、AnomEval
- 核心启示：将事件、原因和后果分别评估，并结合关键证据片段检查解释是否有视频依据。
- 阅读关注：区分基准因果标注与可验证的因果推断；比较 AnomEval 的人工一致性和模型偏好。与 CUVA 的扩展关系单独记录。
- 核验：2026-10-03；[来源 1](<https://link.springer.com/article/10.1007/s11263-026-02983-0>) · [来源 2](<https://arxiv.org/html/2412.07183v1#S2>)

<a id="paper-phys-ad"></a>

### Towards Visual Discrimination and Reasoning of Real-World Physical Dynamics: Physics-Grounded Anomaly Detection

**2025 · CVPR** · [paper](<https://openaccess.thecvf.com/content/CVPR2025/html/Li_Towards_Visual_Discrimination_and_Reasoning_of_Real-World_Physical_Dynamics_Physics-Grounded_CVPR_2025_paper.html>) · [code](<https://github.com/Chopper-233/Physics-AD>) · [project](<https://guyao2023.github.io/Phys-AD/>) · [引用 / BibTeX](<citations.md#cite-phys-ad>)

**创新：物理交互异常与原因解释评测**

采集机械臂与电机作用于真实物体的动态异常视频，分别评估异常判别、现象描述和物理原因解释，提出 Phys-AD 数据与 PAEval 指标。

- 任务：异常检测、异常解释、异常推理、基准评测
- 方法与场景标签：物理异常理解、工业视频、物体交互、PAEval
- 核心启示：异常理解需要检验物体功能与物理规律，区分现象描述和物理原因解释。
- 阅读关注：受控物体交互的物理异常与开放场景事件不同；论文与项目页的视频总数口径存在差异，复现应固定发布版本。
- 核验：2026-10-03；[来源 1](<https://openaccess.thecvf.com/content/CVPR2025/html/Li_Towards_Visual_Discrimination_and_Reasoning_of_Real-World_Physical_Dynamics_Physics-Grounded_CVPR_2025_paper.html>) · [来源 2](<https://openaccess.thecvf.com/content/CVPR2025/papers/Li_Towards_Visual_Discrimination_and_Reasoning_of_Real-World_Physical_Dynamics_Physics-Grounded_CVPR_2025_paper.pdf>) · [来源 3](<https://github.com/Chopper-233/Physics-AD>)

<a id="paper-mmad"></a>

### MMAD: A Comprehensive Benchmark for Multimodal Large Language Models in Industrial Anomaly Detection

**2025 · ICLR** · [paper](<https://proceedings.iclr.cc/paper_files/paper/2025/hash/d91ffbe9c126765755ff52d36b715683-Abstract-Conference.html>) · [code](<https://github.com/jam-cc/MMAD>) · [引用 / BibTeX](<citations.md#cite-mmad>)

**创新：七任务异常问答与缺陷分析评测**

将工业异常判别、缺陷分类、位置、描述、影响分析和物体知识组织为七项问答任务，以 8,366 张图像和 39,672 个问题评估多模态模型。

- 任务：异常检测、空间定位、异常解释、异常推理、基准评测
- 方法与场景标签：工业图像、异常问答、领域知识、多项选择评测
- 核心启示：把异常判别与缺陷语义理解分开评价，可识别模型知道物体却不理解缺陷的差距。
- 阅读关注：多项选择准确率与开放式解释质量应区分；正常参照和领域知识配置会影响结果。
- 核验：2026-10-03；[来源 1](<https://proceedings.iclr.cc/paper_files/paper/2025/hash/d91ffbe9c126765755ff52d36b715683-Abstract-Conference.html>) · [来源 2](<https://proceedings.iclr.cc/paper_files/paper/2025/file/d91ffbe9c126765755ff52d36b715683-Paper-Conference.pdf>) · [来源 3](<https://github.com/jam-cc/MMAD>)

<a id="paper-single-scene-vad-survey"></a>

### A Survey of Single-Scene Video Anomaly Detection

**2022 · TPAMI** · [paper](<https://doi.org/10.1109/tpami.2020.3040591>) · [引用 / BibTeX](<citations.md#cite-single-scene-vad-survey>)

**创新：单场景异常检测综述与评测梳理**

系统梳理单场景视频异常检测的问题定义、公开数据、评测准则与方法分类，并比较标准测试集上的算法表现。

- 任务：基准评测
- 核心启示：在阅读异常理解方法前，厘清传统检测任务的训练假设与评测边界。
- 阅读关注：这是背景综述，不作为新方法站点；采用 2022 年正式卷期而非 2020 年 Early Access。
- 核验：2026-10-04；[来源 1](<https://europepmc.org/article/MED/33237854>) · [来源 2](<https://api.crossref.org/works/10.1109/tpami.2020.3040591>)

<a id="paper-black-swan"></a>

### Black Swan: Abductive and Defeasible Video Reasoning in Unpredictable Events

**2025 · CVPR** · [paper](<https://openaccess.thecvf.com/content/CVPR2025/html/Chinchure_Black_Swan_Abductive_and_Defeasible_Video_Reasoning_in_Unpredictable_Events_CVPR_2025_paper.html>) · [project](<https://blackswan.cs.ubc.ca/>) · [引用 / BibTeX](<citations.md#cite-black-swan>)

**创新：意外事件溯因与证据更新评测**

将意外事件视频拆成事件前、事件中与事件后观察，构建 Forecaster、Detective、Reporter 三类任务，评估候选事件预测、缺失事件溯因以及新证据出现后的解释修正。

- 任务：异常推理、异常预判、视频问答、基准评测
- 方法与场景标签：意外事件、溯因推理、可撤销推理、证据更新
- 核心启示：异常理解不仅要给出解释，还应能根据新增观察撤销或修正原有假设。
- 阅读关注：短视频意外事件问答不同于长监控视频中的帧级异常检测，预测、溯因与修正任务须分别比较。
- 核验：2026-10-04；[来源 1](<https://openaccess.thecvf.com/content/CVPR2025/html/Chinchure_Black_Swan_Abductive_and_Defeasible_Video_Reasoning_in_Unpredictable_Events_CVPR_2025_paper.html>) · [来源 2](<https://openaccess.thecvf.com/content/CVPR2025/papers/Chinchure_Black_Swan_Abductive_and_Defeasible_Video_Reasoning_in_Unpredictable_Events_CVPR_2025_paper.pdf>)

## 结构化推理与验证

创新：因果问题分解、对象关系编码、反思修正与关键事实评估，让异常解释可以被检查。

研究问题：怎样组织推理，并检验结论是否抓住关键事实？

<a id="paper-cuva"></a>

### Uncovering What, Why and How: A Comprehensive Benchmark for Causation Understanding of Video Anomaly

**2024 · CVPR** · [paper](<https://arxiv.org/abs/2405.00181>) · [code](<https://github.com/fesvhtr/CUVA>) · [引用 / BibTeX](<citations.md#cite-cuva>)

**创新：事件因果任务分解**

用事件、原因和后果标注，将视频异常理解拓展到因果解释及评估。

- 任务：异常解释、异常推理、视频问答
- 兼属方法：评测与任务拓展；[归类依据](<https://openaccess.thecvf.com/content/CVPR2024/html/Du_Uncovering_What_Why_and_How_A_Comprehensive_Benchmark_for_Causation_CVPR_2024_paper.html>) — CUVA 同时提出因果理解基准与 MMEval 评估方法，连接推理与理解评测。
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
- 兼属方法：评测与任务拓展；[归类依据](<https://papers.nips.cc/paper_files/paper/2025/hash/abccc325c84dedf23dbe8de3f686c733-Abstract-Conference.html>) — Vad-R1 同时提出 P2C-CoT 感知认知推理链、自验证强化学习 AVA-GRPO 与 Vad-Reasoning 数据；据此连接结构化推理与任务评测两个阅读方向，支线首站为 Cue-R1。线路表示方法与任务贡献的阅读关联，不表示后续论文直接继承 Vad-R1。
- 核心启示：把“看到了什么”与“为何异常”分步组织，再让训练奖励约束推理与结论。
- 阅读关注：结构化文本与自验证是训练机制；解释是否忠实于实际视觉证据仍需单独检查。
- 核验：2026-10-01；[来源 1](<https://papers.nips.cc/paper_files/paper/2025/hash/abccc325c84dedf23dbe8de3f686c733-Abstract-Conference.html>)

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
- 兼属方法：评测与任务拓展；[归类依据](<https://smartinternet.group/wp-content/uploads/2025/12/paper-whl-HoloTrace-mm.pdf>) — 摘要、贡献列表与 §5.1：除双向因果知识图方法外，论文构建了包含 632 段视频、10 类异常和逐帧标注的 SVAD 数据集，并在 SVAD 与 UCSD PED2 上评价；因此保留推理主归属，补充任务与评测归属。论文未使用 Phys-AD；线路回接表示从物理原因解释评测到显式因果事件推理的阅读关联。
- 核心启示：将事件关系显式存入可更新结构，使边缘检测能够复用语言模型获得的语义知识。
- 阅读关注：图中的因果关系来自模型构建与更新，不能自动视为经干预验证的真实因果关系。
- 核验：2026-10-04；[来源 1](<https://doi.org/10.1145/3746027.3755185>) · [来源 2](<https://smartinternet.group/wp-content/uploads/2025/12/paper-whl-HoloTrace-mm.pdf>)

<a id="paper-vad-r1-plus"></a>

### Advancing Adaptive Multi-Stage Video Anomaly Reasoning: A Benchmark Dataset and Method

**2026 · arXiv** · [paper](<https://arxiv.org/abs/2601.10165>) · [project](<https://github.com/wbfwonderful/Vad-R1-Plus>) · [引用 / BibTeX](<citations.md#cite-vad-r1-plus>)

**创新：感知认知行动链与异常感知优化**

用感知、认知与行动三级思维链组织异常推理，并以异常感知的组相对策略优化训练支持不同推理深度和风险判断的模型。

- 任务：异常推理、视频问答
- 兼属方法：评测与任务拓展；[归类依据](<https://arxiv.org/html/2601.10165v1#S1>) — 引言与方法 III-A 同时提出 Vad-Reasoning-Plus 数据集、与感知/认知/行动对应的分阶段问题，以及自适应推理模型；用于系统评估多阶段异常推理。据此保留推理主归属，添加任务与评测的次级归属，作为局部评测支线回接点。连线为编辑阅读关联，不声明 Cue-R1 或 FineVAU 被该工作继承。
- 核心启示：异常理解可以进一步评估风险解释和决策建议所需的推理深度。
- 阅读关注：风险判断和行动建议是否有可见证据支持，弱监督奖励如何约束可靠性？
- 核验：2026-10-02；[来源 1](<https://arxiv.org/abs/2601.10165>) · [来源 2](<https://github.com/wbfwonderful/Vad-R1-Plus>) · [来源 3](<https://arxiv.org/html/2601.10165v1#S1>)

<a id="paper-vau-r1"></a>

### VAU-R1: Advancing Video Anomaly Understanding via Reinforcement Fine-Tuning

**2025 · arXiv** · [paper](<https://arxiv.org/abs/2505.23504>) · [code](<https://github.com/GVCLab/VAU-R1>) · [引用 / BibTeX](<citations.md#cite-vau-r1>)

**创新：多任务奖励与强化微调**

通过任务专属奖励进行强化微调，并构建包含选择问答、推理依据、时间边界和描述的 VAU-Bench。

- 任务：异常定位、异常推理、视频问答
- 核心启示：问答、分类、推理与时间定位需要分别定义目标和评价协议。
- 阅读关注：格式、正确率与时间交并比奖励是否真正提升解释的视觉忠实度？
- 核验：2026-09-30；[来源 1](<https://arxiv.org/abs/2505.23504>) · [来源 2](<https://github.com/GVCLab/VAU-R1>)

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

**2026 · ECCV** · [paper](<https://arxiv.org/abs/2607.18142>) · [code](<https://github.com/o-vad/O-VAD>) · [project](<https://o-vad.github.io/>) · [引用 / BibTeX](<citations.md#cite-o-vad>)

**创新：对象状态轨迹与时序推理**

在工业视频中跟踪对象状态随时间的演化，再对对象轨迹推理，定位异常对象与帧并输出异常过程和类型报告。

- 任务：异常检测、异常定位、异常解释
- 核心启示：从对象状态变化解释工业视频异常。
- 阅读关注：对象检测与跟踪错误可能传入推理；工业流程验证不能直接代表开放监控场景。
- 核验：2026-10-04；[来源 1](<https://arxiv.org/abs/2607.18142>) · [来源 2](<https://eccv.ecva.net/virtual/2026/poster/4659>) · [来源 3](<https://arxiv.org/html/2607.18142v1>) · [来源 4](<https://arxiv.org/html/2607.18142v1#A3.SS5>)

<a id="paper-stch"></a>

### Streaming Video Crime Anticipation with Spatio-Temporal Causal Reasoning

**2026 · CVPR** · [paper](<https://openaccess.thecvf.com/content/CVPR2026/html/Wang_Streaming_Video_Crime_Anticipation_with_Spatio-Temporal_Causal_Reasoning_CVPR_2026_paper.html>) · [引用 / BibTeX](<citations.md#cite-stch>)

**创新：流式时空因果超图**

构建具有递进推理任务的 STCRC 基准，并用流式时空因果超图显式组织实体动态，支撑犯罪事件预判。

- 任务：异常预判、异常推理、基准评测
- 核心启示：把实体动态组织为犯罪预兆推理结构。
- 阅读关注：任务是事件预判，与事后异常理解需分别评测；因果结构依赖事件和实体标注质量。
- 核验：2026-09-30；[来源 1](<https://openaccess.thecvf.com/content/CVPR2026/html/Wang_Streaming_Video_Crime_Anticipation_with_Spatio-Temporal_Causal_Reasoning_CVPR_2026_paper.html>) · [来源 2](<https://cvpr.thecvf.com/virtual/2026/poster/39800>)

<a id="paper-crcl"></a>

### CRCL: Causal Representation Consistency Learning for Anomaly Detection in Surveillance Videos

**2025 · TIP** · [paper](<https://doi.org/10.1109/tip.2025.3558089>) · [引用 / BibTeX](<citations.md#cite-crcl>)

**创新：场景去偏与因果正常性表征**

基于结构因果模型，将场景去偏与因果正常性学习结合，从无监督视频正常模式中学习对场景变化更稳定的表征。

- 任务：异常检测
- 方法与场景标签：因果表征、场景去偏、无监督学习
- 核心启示：因果表征提供了检验异常判据是否依赖场景偏差的研究角度。
- 阅读关注：如何验证表征捕获了稳定的因果因素，以及场景去偏的适用条件？
- 核验：2026-10-02；[来源 1](<https://arxiv.org/abs/2503.18808>) · [来源 2](<https://api.crossref.org/works/10.1109/tip.2025.3558089>)

<a id="paper-avar"></a>

### Advancing Video Anomaly Retrieval via Action-Focused Temporal Reasoning and Query-Adaptive Routing

**2026 · ACM MM** · [paper](<https://2026.acmmm.org/site/technical-programme.html>) · [引用 / BibTeX](<citations.md#cite-avar>)

**创新：动作聚焦与查询自适应路由**

用文本动作语义定位目标，以双路径时序异常推理和查询自适应路由检索视频中的异常事件。

- 任务：异常检索
- 方法与场景标签：异常检索
- 核心启示：异常检索需要把动作目标和时间上下文同时纳入匹配。
- 阅读关注：查询措辞变化及与普通视频文本检索的差异需要验证。
- 核验：2026-10-01；[来源 1](<https://2026.acmmm.org/site/program-data.json>)

<a id="paper-adversa"></a>

### ADVersa: Abductive Driving Accident Video Understanding

**2026 · TPAMI** · [paper](<https://doi.org/10.1109/TPAMI.2026.3663545>) · [引用 / BibTeX](<citations.md#cite-adversa>)

**创新：关系感知的跨模态事故溯因**

用 Abductive CLIP 与关系感知的对比图视频预训练组织跨模态证据，为缺失的近事故场景推断图像和文本解释，并支持过去恢复、未来预测及原因条件的视频生成。

- 任务：异常解释、异常推理、异常预判、视频问答
- 方法与场景标签：交通事故理解、溯因推理、关系建模、事故视频生成
- 核心启示：通过缺失场景恢复、事故原因回答与生成任务，对照模型能否组织事故发展的视觉和语言证据。
- 阅读关注：区分事故原因回答、近事故恢复与预测、原因条件生成各任务的输入和评价；CVPR 2024 AdVersa-SD 另列，直接扩展关系仍待期刊全文确认。
- 核验：2026-10-03；[来源 1](<https://engagedscholarship.csuohio.edu/enece_facpub/533/>) · [来源 2](<https://openaccess.thecvf.com/content/CVPR2024/html/Fang_Abductive_Ego-View_Accident_Video_Understanding_for_Safe_Driving_Perception_CVPR_2024_paper.html>)

<a id="paper-judo"></a>

### JUDO: A Juxtaposed Domain-Oriented Multimodal Reasoner for Industrial Anomaly QA

**2026 · ICLR** · [paper](<https://proceedings.iclr.cc/paper_files/paper/2026/hash/92a7a03e1c716970848a4a86cc8243ee-Abstract-Conference.html>) · [code](<https://github.com/woodavid31/JUDO>) · [引用 / BibTeX](<citations.md#cite-judo>)

**创新：正常参照对照与领域知识推理**

通过正常图像与缺陷图像的并置分割学习视觉对照，以监督微调注入领域知识，再用领域推理、分割与答案奖励的 GRPO 联合优化异常问答。

- 任务：异常检测、空间定位、异常解释、异常推理
- 方法与场景标签：工业图像、异常问答、领域知识、正常参照、GRPO
- 核心启示：把可定位的视觉差异与领域知识结合，支撑缺陷描述和影响分析。
- 阅读关注：MMAD 分任务表现与二元异常判别存在差异；伪推理奖励的语义相似度不能直接证明推理忠实性。
- 核验：2026-10-03；[来源 1](<https://proceedings.iclr.cc/paper_files/paper/2026/hash/92a7a03e1c716970848a4a86cc8243ee-Abstract-Conference.html>) · [来源 2](<https://proceedings.iclr.cc/paper_files/paper/2026/file/92a7a03e1c716970848a4a86cc8243ee-Paper-Conference.pdf>) · [来源 3](<https://github.com/woodavid31/JUDO>)

<a id="paper-cagc-vad"></a>

### CAGC-VAD: Controlled Abstraction and Graph Competition for Video Anomaly Detection

**2026 · TCSVT** · [paper](<https://doi.org/10.1109/TCSVT.2026.3735599>) · [引用 / BibTeX](<citations.md#cite-cagc-vad>)

**创新：受控抽象与图竞争（题名暂定）**

以受控抽象与图竞争开展视频异常检测。已核验正式书目信息，摘要与正文暂未获取；具体推理机制、解释输出和实验设置待核验。

- 任务：异常检测
- 归类状态：按题名暂定；[依据](<https://api.crossref.org/works/10.1109/TCSVT.2026.3735599>) — 仅依据题名中的 Controlled Abstraction 与 Graph Competition 暂归结构化推理；尚未核验正文方法。
- 核心启示：作为图结构异常分析的补充阅读线索，待正文核验后确定其与异常理解主线的具体联系。
- 阅读关注：尚不能仅凭题名判断图竞争是否产生可读解释，也未确认 LLM／VLM 的具体作用。
- 核验：2026-10-04；[来源 1](<https://api.crossref.org/works/10.1109/TCSVT.2026.3735599>)

<a id="paper-cmcir"></a>

### Cross-Modal Causal Relational Reasoning for Event-Level Visual Question Answering

**2023 · TPAMI** · [paper](<https://doi.org/10.1109/tpami.2023.3284038>) · [code](<https://github.com/HCPLab-SYSU/CMCIR>) · [引用 / BibTeX](<citations.md#cite-cmcir>)

**创新：前后门干预与跨模态因果关系推理**

以视觉前门干预、语言后门干预和时空 Transformer 减少问答中的伪相关，并在 SUTD-TrafficQA 等基准研究事件级视觉问答。

- 任务：视频问答、异常推理
- 核心启示：连接道路事件的原因、反事实问题与因果去偏推理，可与后续异常原因解释方法对照。
- 阅读关注：这是通用事件级视频问答方法，交通异常相关性来自 TrafficQA 任务与案例，不是专门的异常检测器。
- 核验：2026-10-04；[来源 1](<https://guanbinli.com/papers/Cross-Modal_Causal_Relational_Reasoning_for_Event-Level_Visual_Question_Answering.pdf>) · [来源 2](<https://api.crossref.org/works/10.1109/tpami.2023.3284038>)

<a id="paper-adversa-sd"></a>

### Abductive Ego-View Accident Video Understanding for Safe Driving Perception

**2024 · CVPR** · [paper](<https://doi.org/10.1109/cvpr52733.2024.02080>) · [引用 / BibTeX](<citations.md#cite-adversa-sd>)

**创新：事故原因预防文本对与对象扩散**

提出 MM-AU 多模态事故理解数据，以事故原因和预防建议的对比文本对训练 AbductiveCLIP，并用对象中心扩散生成事故前后视频。

- 任务：异常解释、异常推理
- 核心启示：把事故类别识别延伸到原因、预防与情景生成，是道路异常理解的重要早期节点。
- 阅读关注：与 TPAMI ADVersa 保留独立引用；未核实直接扩展声明前，不登记两篇之间的继承关系。
- 核验：2026-10-04；[来源 1](<https://openaccess.thecvf.com/content/CVPR2024/papers/Fang_Abductive_Ego-View_Accident_Video_Understanding_for_Safe_Driving_Perception_CVPR_2024_paper.pdf>) · [来源 2](<https://api.crossref.org/works/10.1109/cvpr52733.2024.02080>)

<a id="paper-iad-r1"></a>

### IAD-R1: Reinforcing Consistent Reasoning in Industrial Anomaly Detection

**2026 · AAAI** · [paper](<https://ojs.aaai.org/index.php/AAAI/article/view/37588>) · [引用 / BibTeX](<citations.md#cite-iad-r1>)

**创新：感知推理一致性微调与强化学习**

以 Expert-AD 思维链数据进行感知激活监督微调，再通过 SC-GRPO 联合优化正常性一致性、判断准确率、缺陷类型和位置奖励，使缺陷感知、推理与答案相互一致。

- 任务：异常检测、空间定位、异常解释、异常推理
- 方法与场景标签：工业图像、思维链、监督微调、强化学习、一致性奖励
- 核心启示：把异常是否存在、缺陷类型和位置同时纳入训练反馈，减少推理内容与最终答案之间的脱节。
- 阅读关注：类型、位置和一致性奖励主要依赖结构化答案匹配，尚不能等同于自由解释的因果正确性。
- 核验：2026-10-04；[来源 1](<https://ojs.aaai.org/index.php/AAAI/article/view/37588>) · [来源 2](<https://ojs.aaai.org/index.php/AAAI/article/download/37588/41550>)

## 异常判据与提示优化

创新：可读属性与正常模式、字幕到判断的语言中介、规则归纳与提示优化，将异常标准显式化。

研究问题：怎样构造场景适用、可解释的异常判据？

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
- 兼属方法：结构化推理与验证；[归类依据](<https://arxiv.org/abs/2407.10299>) — 2026-10-01 核对作者摘要：AnomalyRuler 明确分为 induction 与 deduction 两阶段，从少量正常参考样本归纳规则，再通过演绎推理检测异常，并设计规则聚合与稳健推理策略。因此同时连接异常判据与结构化推理两个阅读方向。
- 核心启示：正常性可以写成可检查、可调整的语言规则。
- 阅读关注：少量正常参考是否覆盖场景中的合理变化？
- 核验：2026-10-01；[来源 1](<https://arxiv.org/abs/2407.10299>) · [来源 2](<https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/10568.pdf>)

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
- 核验：2026-10-01；[来源 1](<https://arxiv.org/abs/2607.00654>) · [来源 2](<https://icml.cc/virtual/2026/poster/64285>)

<a id="paper-probe-vad"></a>

### Probe-VAD: Ordinal Likelihood Probing for Training-Free Video Anomaly Detection

**2026 · arXiv** · [paper](<https://arxiv.org/abs/2609.17211>) · [code](<https://github.com/yvestine/Probe-VAD>) · [引用 / BibTeX](<citations.md#cite-probe-vad>)

**创新：序数语言探测与一致性评分**

以冻结 VLM 对有序严重程度阈值作是／否判断，通过续写似然与序数一致性约束得到连续异常分数。

- 任务：异常检测、异常定位
- 核心启示：语言判据可通过概率接口转为细粒度判断，不依赖先生成字幕。
- 阅读关注：严重程度排序不等同于异常解释，仍需独立检查判断对场景规范与可见证据的依赖。
- 核验：2026-09-30；[来源 1](<https://arxiv.org/abs/2609.17211>) · [来源 2](<https://github.com/yvestine/Probe-VAD>)

<a id="paper-promptvad"></a>

### PromptVAD: Abnormal Prompt via Vision-Language Model

**2026 · TNNLS** · [paper](<https://doi.org/10.1109/tnnls.2025.3621336>) · [引用 / BibTeX](<citations.md#cite-promptvad>)

**创新：领域与类别提示协同学习**

结合可学习的领域提示、类别提示与固定类别定义，缩小视觉特征和异常类别的语义差距，联合学习粗粒度与细粒度异常检测。

- 任务：异常检测、异常定位
- 方法与场景标签：视觉语言模型、提示学习、弱监督学习
- 核心启示：异常类别名称和定义可以转化为可学习的检测语义。
- 阅读关注：固定类别定义与可学习提示的贡献如何区分，类别变化时如何泛化？
- 核验：2026-10-01；[来源 1](<https://pubmed.ncbi.nlm.nih.gov/41166629/>) · [来源 2](<https://api.crossref.org/works/10.1109/tnnls.2025.3621336>)

<a id="paper-prime-vad"></a>

### Rule-Guided Evolution of Hierarchical Reasoning for Explainable Video Anomaly Detection

**2026 · ACM MM** · [paper](<https://2026.acmmm.org/site/technical-programme.html>) · [引用 / BibTeX](<citations.md#cite-prime-vad>)

**创新：规则记忆与分层提示演化**

把场景、对象和推理提示拆成模块，从运行轨迹提炼规则记忆，并逐模块演化提示以获得可解释判据。

- 任务：异常检测、异常解释
- 核心启示：显式模块化提示有助于检查场景判据与事件证据的对应关系。
- 阅读关注：提示规则可能过拟合训练场景或依赖模型自身反馈。
- 核验：2026-10-01；[来源 1](<https://2026.acmmm.org/site/program-data.json>)

<a id="paper-ca-judge"></a>

### CA-Judge: Teach Large Models to Judge Anomalies via Comparison for Video Anomaly Detection

**2026 · NeurIPS** · [paper](<https://neurips.cc/virtual/2026/poster/153516>) · [引用 / BibTeX](<citations.md#cite-ca-judge>)

**创新：比较驱动的异常判断**

依据题名，暂将其归为通过比较教导大模型判断视频异常的方法。

- 任务：异常检测
- 归类状态：按题名暂定；[依据](<https://neurips.cc/Downloads/2026>) — 按题名暂定归类；方法细节、训练设置与实验结果待摘要或正文核验。
- 核心启示：待正文核验：比较对象、监督方式与异常判断流程。
- 阅读关注：当前方法归类仅依据官方题名；作者、摘要与完整引用元数据待补。
- 核验：2026-10-01；[来源 1](<https://neurips.cc/Downloads/2026>)

<a id="paper-road"></a>

### ROAD: Rule-Grounded Context-Aware Open-World Driver Anomaly Detection

**2026 · NeurIPS** · [paper](<https://neurips.cc/virtual/2026/poster/154937>) · [引用 / BibTeX](<citations.md#cite-road>)

**创新：规则约束与上下文判断**

依据题名，暂将其归为规则约束、上下文感知的开放世界驾驶员异常检测。

- 任务：异常检测
- 归类状态：按题名暂定；[依据](<https://neurips.cc/Downloads/2026>) — 按题名暂定归类；方法细节、训练设置与实验结果待摘要或正文核验。
- 核心启示：待正文核验：输入模态、规则来源与开放世界评测设置。
- 阅读关注：当前方法归类仅依据官方题名；作者、摘要与完整引用元数据待补。
- 核验：2026-10-01；[来源 1](<https://neurips.cc/Downloads/2026>)

<a id="paper-log-sad"></a>

### Towards Training-free Anomaly Detection with Vision and Language Foundation Models

**2025 · CVPR** · [paper](<https://openaccess.thecvf.com/content/CVPR2025/html/Zhang_Towards_Training-free_Anomaly_Detection_with_Vision_and_Language_Foundation_Models_CVPR_2025_paper.html>) · [code](<https://github.com/zhang0jhon/LogSAD>) · [引用 / BibTeX](<citations.md#cite-log-sad>)

**创新：组合规则引导的多粒度异常匹配**

通过 match-of-thought 从正常图像和语言规则构造匹配方案，联合局部、对象兴趣集合与组合层面的匹配，经检测器校准融合识别结构和逻辑异常。

- 任务：异常检测、空间定位
- 方法与场景标签：工业图像、逻辑异常、组合规则、免训练
- 核心启示：物体是否齐全、数量和组合是否合理，可作为可读的异常判据与视觉匹配结合。
- 阅读关注：组合规则与匹配依赖基础模型提示质量；这里主要评估检测和定位，不是专门的异常问答评测。
- 核验：2026-10-04；[来源 1](<https://openaccess.thecvf.com/content/CVPR2025/html/Zhang_Towards_Training-free_Anomaly_Detection_with_Vision_and_Language_Foundation_Models_CVPR_2025_paper.html>) · [来源 2](<https://openaccess.thecvf.com/content/CVPR2025/papers/Zhang_Towards_Training-free_Anomaly_Detection_with_Vision_and_Language_Foundation_Models_CVPR_2025_paper.pdf>)

## Datasets / 数据资源

完整协议与关联论文见 [数据集索引](benchmarks.md)。数据登记不要求图片，也不代表资源已开放下载。已有图片保留作者署名。

### UCF-Crime

**2018 · CVPR** · [来源入口](<https://www.crcv.ucf.edu/research/real-world-anomaly-detection-in-surveillance-videos/>)

真实监控长视频异常检测基准，提供视频级训练标签与测试时序标注，也是多项异常描述和推理数据的视觉来源。

- 评估协议：采用官方训练／测试划分；弱监督训练与帧级定位评估须区分。
- 核验：[来源 1](<https://openaccess.thecvf.com/content_cvpr_2018/html/Sultani_Real-World_Anomaly_Detection_CVPR_2018_paper.html>) · [来源 2](<https://www.crcv.ucf.edu/research/real-world-anomaly-detection-in-surveillance-videos/>) · [来源 3](<https://www.crcv.ucf.edu/research/real-world-anomaly-detection-in-surveillance-videos/>) · [来源 4](<https://www.crcv.ucf.edu/research/real-world-anomaly-detection-in-surveillance-videos/>)

### XD-Violence

**2020 · ECCV** · [来源入口](<https://roc-ng.github.io/XD-Violence/>)

结合视频与音频的多场景暴力检测基准，以视频级多标签训练，并通过测试时序标注评估异常定位。

- 评估协议：报告所用模态与官方划分；不可将音视频方法和纯视觉方法混为同一设置。
- 图片：[XD-Violence 作者发布的多场景视频样例拼图](<https://roc-ng.github.io/XD-Violence/images/samples.png>)；署名：Peng Wu et al. / XD-Violence
- 核验：[来源 1](<https://roc-ng.github.io/XD-Violence/>) · [来源 2](<https://roc-ng.github.io/XD-Violence/>) · [来源 3](<https://roc-ng.github.io/XD-Violence/>) · [来源 4](<https://roc-ng.github.io/XD-Violence/>)

### CUVA

**2024 · CVPR** · [来源入口](<https://github.com/fesvhtr/CUVA>)

围绕异常事件的经过、原因与后果构建因果理解任务，以人工语言标注连接事件定位、描述和解释。

- 评估协议：按原论文任务与 MMEval 协议评估；定位、描述和因果解释分别报告。
- 图片：[CUVA 论文中的异常视频及因果标注示例](<https://arxiv.org/html/2405.00181v3/dataset_4.png>)；署名：Hang Du et al. / CUVA
- 核验：[来源 1](<https://openaccess.thecvf.com/content/CVPR2024/html/Du_Uncovering_What_Why_and_How_A_Comprehensive_Benchmark_for_Causation_CVPR_2024_paper.html>) · [来源 2](<https://github.com/fesvhtr/CUVA>) · [来源 3](<https://github.com/fesvhtr/CUVA>) · [来源 4](<https://arxiv.org/html/2405.00181v3#S3.SS2>)

### HIVAU-70k

**2025 · CVPR** · [来源入口](<https://github.com/pipixin321/HolmesVAU>)

在 UCF-Crime 与 XD-Violence 上构建片段、事件、视频三级异常指令，覆盖局部描述、事件分析和全局总结。

- 评估协议：区分时间粒度与任务类型；数据构建包含模型生成与人工复核。
- 图片：[HIVAU-70k 片段、事件和视频三级异常理解示例](<https://raw.githubusercontent.com/pipixin321/HolmesVAU/master/assets/teaser.png>)；署名：Huaxin Zhang et al. / Holmes-VAU
- 核验：[来源 1](<https://openaccess.thecvf.com/content/CVPR2025/html/Zhang_Holmes-VAU_Towards_Long-term_Video_Anomaly_Understanding_at_Any_Granularity_CVPR_2025_paper.html>) · [来源 2](<https://github.com/pipixin321/HolmesVAU>) · [来源 3](<https://github.com/pipixin321/HolmesVAU/tree/master/HIVAU-70k>)

### HAWK

**2024 · NeurIPS** · [来源入口](<https://github.com/jqtangust/hawk>)

面向开放场景的视频异常理解资源，提供异常视频语言描述与相关问答，支持描述生成和交互式理解。

- 评估协议：遵循作者数据划分，分别检查描述生成与问答表现。
- 图片：[Hawk 作者发布的开放场景异常理解与问答示例](<https://raw.githubusercontent.com/jqtangust/hawk/main/figs/motivation1.png>)；署名：Jiaqi Tang et al. / Hawk
- 核验：[来源 1](<https://proceedings.neurips.cc/paper_files/paper/2024/hash/fca83589e85cb061631b7ebc5db5d6bd-Abstract-Conference.html>) · [来源 2](<https://github.com/jqtangust/hawk>)

### FineW3

**2026 · AAAI** · [来源入口](<https://finevau.github.io/>)

围绕 What、Who、Where 增强异常事件、参与实体和位置事实，用于评估异常描述是否与关键视觉证据一致。

- 评估协议：使用 FVScore 检查关键视觉元素，结合人类一致性分析；不能只比较语言流畅度。原论文描述基于 UCA 的 1,544 段视频；当前公开文件另含 ECVA 来源记录，复现时须记录发布版本，不能将当前行数直接当作论文视频数。
- 图片：[FineVAU 论文中的细粒度异常描述和视觉要素对照](<https://arxiv.org/html/2601.17258v2/figs/Teaser.png>)；署名：João Pereira et al. / FineVAU
- 核验：[来源 1](<https://ojs.aaai.org/index.php/AAAI/article/download/37790/41752>) · [来源 2](<https://finevau.github.io/>) · [来源 3](<https://arxiv.org/html/2601.17258v2>) · [来源 4](<https://huggingface.co/datasets/joao-cardeira/FineW3>)

### UCA

**2024 · CVPR** · [来源入口](<https://github.com/jia-wang-11/Surveillance-Video-Understanding-dataSet>)

为 UCF-Crime 中的 1,854 段视频补充事件语句与时间边界，支持语言时序定位、视频描述和密集描述。

- 评估协议：使用作者训练、验证、测试划分；分别报告语言时序定位、视频描述、密集描述与多模态异常检测。
- 图片：[UCA 官方仓库展示的细粒度事件语句与对应时间段](<https://github.com/jia-wang-11/Surveillance-Video-Understanding-dataSet>)；署名：Tongtong Yuan et al. / UCA
- 核验：[来源 1](<https://openaccess.thecvf.com/content/CVPR2024/html/Yuan_Towards_Surveillance_Video-and-Language_Understanding_New_Dataset_Baselines_and_Challenges_CVPR_2024_paper.html>) · [来源 2](<https://github.com/jia-wang-11/Surveillance-Video-Understanding-dataSet>) · [来源 3](<https://github.com/jia-wang-11/Surveillance-Video-Understanding-dataSet>)

### ECVA

**2026 · IJCV** · [来源入口](<https://github.com/Dulpy/ECVA>)

扩展 CUVA 的异常因果理解体系，以事件描述、原因、后果和关键证据重要性曲线支持更细致的解释评估。

- 评估协议：按作者发布版本分别评估描述、原因和后果；AnomEval 检查推理、回答一致性与幻觉，避免与 CUVA 的 MMEval 混用。
- 图片：[ECVA 论文中异常因果理解的挑战与视频样例](<https://arxiv.org/html/2412.07183v1/challenge_v7.png>)；署名：Hang Du et al. / ECVA
- 核验：[来源 1](<https://link.springer.com/article/10.1007/s11263-026-02983-0>) · [来源 2](<https://arxiv.org/abs/2412.07183>) · [来源 3](<https://github.com/Dulpy/ECVA>) · [来源 4](<https://www.modelscope.cn/datasets/gouchenyi/ECVA/files>) · [来源 5](<https://arxiv.org/html/2412.07183v1#S3.SS2>) · [来源 6](<https://github.com/Dulpy/ECVA>)

### Vad-Reasoning

**2025 · NeurIPS** · [来源入口](<https://github.com/wbfwonderful/Vad-R1>)

为既有异常视频增加从感知到认知的结构化推理，分别提供用于监督微调的推理文本与强化学习的弱标签。

- 评估协议：分别使用 Vad-Reasoning-SFT 的训练／测试划分与 Vad-Reasoning-RL；SFT 含推理文本，RL 仅有视频级弱标签。
- 图片：[Vad-Reasoning 官方仓库中的视频、推理过程与最终答案标注示例](<https://raw.githubusercontent.com/wbfwonderful/Vad-R1/main/images/data-example.png>)；署名：Chao Huang, Benfeng Wang et al. / Vad-R1
- 核验：[来源 1](<https://proceedings.neurips.cc/paper_files/paper/2025/hash/abccc325c84dedf23dbe8de3f686c733-Abstract-Conference.html>) · [来源 2](<https://github.com/wbfwonderful/Vad-R1>) · [来源 3](<https://huggingface.co/datasets/wbfwonderful/Vad-R1>) · [来源 4](<https://github.com/wbfwonderful/Vad-R1>) · [来源 5](<https://github.com/wbfwonderful/Vad-R1>)

### CueBench

**2026 · AAAI** · [来源入口](<https://huggingface.co/datasets/CueBench/CueBench>)

以场景与属性组织条件性和绝对异常，检验同一行为在不同上下文中的正常性，并覆盖识别、定位、检测与预判。

- 评估协议：分别报告识别、时序定位、检测和预判；按场景／属性分析条件性异常。官方 Hugging Face 已提供训练、测试、推理标注与视频文件。
- 图片：[CueBench 统一上下文异常评测框架及任务示例](<https://arxiv.org/html/2511.00613v1/evaluation_fig.png>)；署名：Yu, Yating et al. / CueBench
- 核验：[来源 1](<https://ojs.aaai.org/index.php/AAAI/article/view/38209>) · [来源 2](<https://huggingface.co/datasets/CueBench/CueBench>) · [来源 3](<https://huggingface.co/datasets/CueBench/CueBench/tree/main>)

### VAGU

**2026 · AAAI** · [来源入口](<https://ojs.aaai.org/index.php/AAAI/article/view/42412>)

联合视频异常时间定位与语义理解，提供异常问答、解释和时间边界，用于检验模型能否同时定位并说明异常。

- 评估协议：使用原版 VAGU 的问答与 JeAUG 联合评价，联合分数之外保留定位和理解单项结果；不混入扩展稿 VAGU-T。当前未核验数据下载状态。
- 核验：[来源 1](<https://arxiv.org/abs/2507.21507>)

### VALU

**2026 · ACL** · [来源入口](<https://aclanthology.org/2026.acl-long.56/>)

以五个语义层级组织异常事件的时间边界与细粒度文本，支持时序定位、异常定位及描述细节辨析。

- 评估协议：按语义层级分别评估 temporal grounding、anomaly localization 和 detail discrimination。论文页写明将公开基准，本次未确认下载状态。
- 图片：[VALU：多层级异常标注示例原论文图](<https://aclanthology.org/2026.acl-long.56.pdf>)；署名：Yixiao He et al. / VALU
- 核验：[来源 1](<https://aclanthology.org/2026.acl-long.56/>)

### A2Seek

**2025 · NeurIPS Datasets and Benchmarks** · [来源入口](<https://2-mo.github.io/A2Seek/>)

面向动态航拍视角，将异常类别、帧级时间戳和区域框与自然语言解释关联，支持时空证据定位与因果理解。

- 评估协议：区分异常判断、区域定位和解释；遵循作者场景与分布外设置，不与固定监控结果直接混排。当前未核验下载状态。
- 图片：[A2Seek / A2Seek-R1：航拍异常理解基准的任务、场景与标注概览。](<https://2-mo.github.io/A2Seek/static/images/carousel1.png>)；署名：Mo, Mengjingcheng et al. / A2Seek / A2Seek-R1
- 核验：[来源 1](<https://proceedings.neurips.cc/paper_files/paper/2025/hash/de02de513503962e1d21035ab50ce661-Abstract-Datasets_and_Benchmarks_Track.html>)

### TAU-Bench

**2026 · arXiv** · [来源入口](<https://yarkupa.github.io/tau-bench.github.io/>)

将异常实例轨迹和像素掩码与实例、事件、场景三级语义绑定，评估模型能否跟踪正确对象并解释其异常。

- 评估协议：联合检查实例跟踪和细粒度语义，不能以描述流畅度替代轨迹正确性；本次未确认数据／模型有效下载入口。
- 图片：[TAU-Bench：图 1：关联异常实例轨迹与细粒度语义理解的基准概览。](<https://yarkupa.github.io/tau-bench.github.io/assets/overview.png>)；署名：Yang, Kepeng et al. / TAU-Bench
- 核验：[来源 1](<https://arxiv.org/abs/2608.05699>)

### Pistachio

**2026 · ECCV** · [来源入口](<https://pistachio-video.github.io>)

面向检测与理解的可控合成视频基准，分为 Pistachio-VAD 和 Pistachio-VAU，提供帧级标签及事件、视频级描述。

- 评估协议：分别报告 VAD 与 VAU 协议；关注合成到真实的域差异及多事件子集。数据下载未在本次逐文件验证。
- 图片：[Pistachio：图 2：从场景和故事线生成异常视频及事件摘要。](<https://arxiv.org/html/2511.19474v6/3.png>)；署名：Li, Jie et al. / Pistachio
- 核验：[来源 1](<https://arxiv.org/abs/2511.19474>) · [来源 2](<https://arxiv.org/html/2511.19474v6>)

### VANE-Bench

**2025 · NAACL Findings** · [来源入口](<https://github.com/rohit901/VANE-Bench>)

通过异常问答评测合成视频的不一致性与真实视频异常，将不同视频来源纳入检测和定位任务。

- 评估协议：分别检查合成与真实视频子集；问答正确率不直接等价于逐帧检测 AUC。作者提供代码和数据入口，本次未逐文件验证下载。
- 图片：[VANE-Bench：VANE-Bench 的视频异常评测与问答构建流程。](<https://github.com/rohit901/VANE-Bench/raw/main/assets/Main_VANE-Bench%20Flow_v7.png?raw=true>)；署名：Gani, Hanan et al. / VANE-Bench
- 核验：[来源 1](<https://aclanthology.org/2025.findings-naacl.171/>)

### UCFCrime-AR

**2024 · TIP** · [来源入口](<https://github.com/Roc-Ng/VAR>)

在 UCF-Crime 长视频上增加事件文本与视频配对，支持以自然语言查询检索未裁剪的异常视频。

- 评估协议：按作者检索划分进行文本—视频匹配，候选对象为未裁剪视频；作者资源页提供训练与测试文本。
- 核验：[来源 1](<https://arxiv.org/html/2307.12545v2>) · [来源 2](<https://github.com/Roc-Ng/VAR>) · [来源 3](<https://github.com/Roc-Ng/VAR>) · [来源 4](<https://github.com/Roc-Ng/VAR>)

### XDViolence-AR

**2024 · TIP** · [来源入口](<https://github.com/Roc-Ng/VAR>)

将 XD-Violence 的同步音视频组织为异常检索基准，以音频作为查询，从长视频候选库中寻找匹配内容。

- 评估协议：按作者音视频检索设置评估配对检索；使用 AR 基准的划分和候选库，并与帧级检测评测分别报告。
- 核验：[来源 1](<https://arxiv.org/html/2307.12545v2>) · [来源 2](<https://github.com/Roc-Ng/VAR>) · [来源 3](<https://github.com/Roc-Ng/VAR>) · [来源 4](<https://github.com/Roc-Ng/VAR>)

### MM-AU

**2024 · CVPR** · [来源入口](<https://openaccess.thecvf.com/content/CVPR2024/html/Fang_Abductive_Ego-View_Accident_Video_Understanding_for_Safe_Driving_Perception_CVPR_2024_paper.html>)

面向驾驶事故理解的多模态基准，包含 11,727 段事故视频及对齐文本、对象框和事故原因问答。

- 评估协议：分别评估对象检测、事故原因回答、近事故场景恢复与预测等任务；按对应论文任务设置比较。此处提供论文入口，未核验数据下载可用性。
- 核验：[来源 1](<https://openaccess.thecvf.com/content/CVPR2024/html/Fang_Abductive_Ego-View_Accident_Video_Understanding_for_Safe_Driving_Perception_CVPR_2024_paper.html>) · [来源 2](<https://engagedscholarship.csuohio.edu/enece_facpub/533/>)

### TAR / TAR-Bench

**2026 · NeurIPS Evaluations and Datasets** · [来源入口](<https://huggingface.co/datasets/nvidia/PhysicalAI-Traffic-Anomaly-Reasoning>)

交通异常多任务推理资源，TAR 提供 44,040 条训练伪标注，TAR-Bench 以 960 条人工标注组成官方测试。

- 评估协议：训练部分为 3,670 段视频的 44,040 条伪标注；官方测试为 80 段视频的 960 条人工标注。测试问题公开、答案隐藏，需通过官方评测服务评估；视频从原始来源单独获取。
- 图片：[Figure 1: TAR 与 TAR-Bench 交通异常多任务标注示意（作者预印本）](<https://arxiv.org/html/2608.10317v3/TAR-teaser-2a-.png>)；署名：Han Zhang et al. / TAR / TAR-Bench
- 核验：[来源 1](<https://arxiv.org/abs/2608.10317>) · [来源 2](<https://huggingface.co/datasets/nvidia/PhysicalAI-Traffic-Anomaly-Reasoning>) · [来源 3](<https://huggingface.co/datasets/nvidia/PhysicalAI-Traffic-Anomaly-Reasoning#source-videos>)

### Vad-Reasoning-Plus

**2026 · arXiv** · [来源入口](<https://github.com/wbfwonderful/Vad-R1-Plus>)

扩展 Vad-Reasoning，以感知、认知、行动三级开放式问答和推理链，连接异常理解、风险判断与行动建议。

- 评估协议：按论文的训练与测试划分评估分层回答和推理；区分完整推理监督的 SFT 子集与弱监督 RL 子集。作者尚未在所链接仓库发布数据文件，暂不能按已开放资源使用。
- 核验：[来源 1](<https://arxiv.org/html/2601.10165v1>) · [来源 2](<https://github.com/wbfwonderful/Vad-R1-Plus>)

### UCSD Ped1 / Ped2

**2010 · CVPR** · [来源入口](<http://www.svcl.ucsd.edu/projects/anomaly/dataset.htm>)

两处固定摄像机人行道基准，异常自然发生；Ped1 包含明显透视变化，Ped2 的行人运动大致平行于成像平面。

- 评估协议：分别按 Ped1 的 34／36 和 Ped2 的 16／12 训练／测试片段评估；正常训练。帧级与像素级结果分开报告，Ped1 原始像素协议仅覆盖 10 个测试片段，不能与全部 36 段或后续补标版本混报。
- 核验：[来源 1](<http://www.svcl.ucsd.edu/projects/anomaly/dataset.htm>) · [来源 2](<http://www.svcl.ucsd.edu/projects/anomaly/>)

### ShanghaiTech Campus

**2017 · ICCV** · [来源入口](<https://svip-lab.github.io/dataset/campus_dataset.html>)

覆盖 13 个校园场景的异常检测基准，包含复杂光照与视角，以及追逐、打斗等突发运动异常。

- 评估协议：采用官方正常训练／异常测试协议；多场景共用模型。后续弱监督重划分与 ShanghaiTech-sd 应另行注明，不能与原始协议混报。
- 核验：[来源 1](<https://svip-lab.github.io/dataset/campus_dataset.html>) · [来源 2](<https://openaccess.thecvf.com/content_ICCV_2017/papers/Luo_A_Revisit_of_ICCV_2017_paper.pdf>)

### TAD (Traffic Anomaly Dataset)

**2021 · TIP** · [来源入口](<https://github.com/ktr-hubrt/WSAL>)

包含 500 段交通视频的弱监督异常检测基准，正常与异常各 250 段，覆盖 7 类交通异常。

- 评估协议：原论文采用 400 训练／100 测试划分，二者均含正常与异常视频；弱监督训练与测试帧级评估分开。须核实所用分割文件及标注版本，不能仅凭公开抽帧包认定复现了论文协议。
- 核验：[来源 1](<https://github.com/ktr-hubrt/WSAL>) · [来源 2](<https://arxiv.org/pdf/2008.08944>)

### UBnormal

**2022 · CVPR** · [来源入口](<https://github.com/lilygeorgescu/UBnormal>)

以多个虚拟场景构建合成异常视频，训练时提供异常像素标注，并用不相交的训练、测试异常类别检验开放集泛化。

- 评估协议：遵守官方划分及训练／测试异常类别不相交的开放集设置。原始任务允许使用训练异常及像素标注；仅正常训练的变体应单列，不与监督开放集设置混报。
- 核验：[来源 1](<https://github.com/lilygeorgescu/UBnormal>)

### MSAD

**2024 · NeurIPS Datasets and Benchmarks** · [来源入口](<https://msad-dataset.github.io/>)

720 段 RGB 视频覆盖 14 类真实监控场景，包含人体与非人体异常及天气、光照变化，用于评估多场景检测。

- 评估协议：区分官方两种协议：仅正常训练为 360 正常训练，120 正常＋240 异常测试；弱监督为 360 正常＋120 异常训练，120 正常＋120 异常测试，训练仅用视频级标签。
- 核验：[来源 1](<https://msad-dataset.github.io/>)

### NWPU Campus

**2023 · CVPR** · [来源入口](<https://campusvad.github.io/>)

547 段校园视频覆盖 43 场景与 28 类异常，突出同一行为随场景改变正常性的情况，同时支持异常检测与预判。

- 评估协议：采用官方 305 正常训练／242 测试视频划分；场景依赖异常的正常性需结合所在场景判断。异常检测与预判应分别报告任务与评价协议。
- 核验：[来源 1](<https://campusvad.github.io/>) · [来源 2](<https://campusvaa.github.io/>)

### CUHK Avenue

**2013 · ICCV** · [来源入口](<https://www.cse.cuhk.edu.hk/leojia/projects/detectabnormal/dataset.html>)

固定校园通道的经典异常检测基准，包含 16 段训练、21 段测试视频及异常空间矩形标注，用于检测与定位。

- 评估协议：采用官方 16／21 训练／测试划分；训练以正常情况为主，作者提示少量离群样本、稀有正常模式及测试轻微抖动。帧级与空间定位评价分开，并说明所用空间标注版本。
- 核验：[来源 1](<https://www.cse.cuhk.edu.hk/leojia/projects/detectabnormal/dataset.html>)

### Phys-AD

**2025 · CVPR** · [来源入口](<https://huggingface.co/datasets/guoliz/Phys-AD>)

以机械臂与电机和 22 类真实物体交互的视频检验物理异常判断、现象描述与原因解释。

- 评估协议：区分无监督、弱监督与视频理解设置；按发布划分评估检测及 PAEval 描述／解释分数。
- 图片：[Phys-AD 的物体、交互动作与正常／异常动态示例。](<https://openaccess.thecvf.com/content/CVPR2025/papers/Li_Towards_Visual_Discrimination_and_Reasoning_of_Real-World_Physical_Dynamics_Physics-Grounded_CVPR_2025_paper.pdf#page=1>)；署名：Li, Wenqiao et al. / Phys-AD
- 核验：[来源 1](<https://openaccess.thecvf.com/content/CVPR2025/papers/Li_Towards_Visual_Discrimination_and_Reasoning_of_Real-World_Physical_Dynamics_Physics-Grounded_CVPR_2025_paper.pdf>) · [来源 2](<https://huggingface.co/datasets/guoliz/Phys-AD>)

### MMAD

**2025 · ICLR** · [来源入口](<https://huggingface.co/datasets/jiang-cc/MMAD>)

在 8,366 张工业图像上构建 39,672 个问题，覆盖异常判别、缺陷分类／定位／描述／分析与物体分类／分析七项任务。

- 评估协议：按七项任务分别报告准确率；核对正常参照数量、领域知识上下文及训练／测试使用设置。
- 图片：[MMAD 七项任务的图像与多项选择问答示例。](<https://proceedings.iclr.cc/paper_files/paper/2025/file/d91ffbe9c126765755ff52d36b715683-Paper-Conference.pdf#page=3>)；署名：Jiang, Xi et al. / MMAD
- 核验：[来源 1](<https://proceedings.iclr.cc/paper_files/paper/2025/file/d91ffbe9c126765755ff52d36b715683-Paper-Conference.pdf>) · [来源 2](<https://huggingface.co/datasets/jiang-cc/MMAD>)

### Anomaly-Instruct-125k

**2025 · CVPR** · [来源入口](<https://xujiacong.github.io/Anomaly-OV/>)

Anomaly-OV 的视觉异常指令数据，包含异常描述、可能原因与改进建议，整合现有视觉数据和 WebAD 图像。

- 评估协议：用于异常专家与指令微调；按论文设置隔离目标评测数据，不将零样本推理理解为无训练。
- 核验：[来源 1](<https://openaccess.thecvf.com/content/CVPR2025/papers/Xu_Towards_Zero-Shot_Anomaly_Detection_and_Reasoning_with_Multimodal_Large_Language_CVPR_2025_paper.pdf>) · [来源 2](<https://xujiacong.github.io/Anomaly-OV/>)

### VisA-D&R

**2025 · CVPR** · [来源入口](<https://xujiacong.github.io/Anomaly-OV/>)

由 VisA 的 10 类物体构建检测与推理评测，包含 761 个正常样本和 1,000 个异常样本及人工复核的异常问答。

- 评估协议：检测使用准确率／精确率／召回率／F1；描述与复杂推理使用 ROUGE-L、SBERT 和 GPT-Score 分别评估。
- 核验：[来源 1](<https://openaccess.thecvf.com/content/CVPR2025/papers/Xu_Towards_Zero-Shot_Anomaly_Detection_and_Reasoning_with_Multimodal_Large_Language_CVPR_2025_paper.pdf>) · [来源 2](<https://xujiacong.github.io/Anomaly-OV/>)

### AV-TAU

**2025 · CVPR** · [来源入口](<https://huggingface.co/datasets/harryhsing/AV-TAU>)

29,865 段音视频交通异常片段与 149,325 组问答，覆盖事件描述、原因、异常时段、预防和响应。

- 评估协议：沿用论文训练／测试划分，按五项任务分别评估；预防与响应建议属于事后理解任务，不直接作为提前预测结果。
- 核验：[来源 1](<https://openaccess.thecvf.com/content/CVPR2025/papers/Xing_EchoTraffic_Enhancing_Traffic_Anomaly_Understanding_with_Audio-Visual_Insights_CVPR_2025_paper.pdf>) · [来源 2](<https://huggingface.co/datasets/harryhsing/AV-TAU>)

### DoTA

**2023 · TPAMI** · [来源入口](<https://github.com/MoonBlvd/Detection-of-Traffic-Anomaly>)

4,677 段第一视角驾驶视频，提供异常起止时间、异常对象框及事件类别，支持何时、何处、何种异常的分析。

- 评估协议：按官方训练／测试划分评估视频异常检测；STAUC 同时考虑时间检测与空间定位。数据已在 2020 年作者预印本公开，此处年份采用关联 TPAMI 正式卷期 2023。
- 图片：[DoTA：Figure 2：DoTA 数据样例及异常对象框，取自早期作者预印本，不代表 TPAMI 新增方法框架。](<https://arxiv.org/pdf/2004.03044#page=5>)；署名：Yu Yao et al. / DoTA author preprint (2020)
- 核验：[来源 1](<https://github.com/MoonBlvd/Detection-of-Traffic-Anomaly>) · [来源 2](<https://arxiv.org/abs/2004.03044>)

### SUTD-TrafficQA

**2021 · CVPR** · [来源入口](<https://github.com/sutdcv/SUTD-TrafficQA>)

10,080 段交通视频与 62,535 组问答，覆盖基本理解、归因、反事实、预测等六类交通推理任务。

- 评估协议：按官方划分分别评估六类交通问答。数据涵盖一般交通事件与事故，不应把所有问题都视为异常理解评测。仅登记 CMCIR 使用的数据资源，不另收录 2021 年会议论文。
- 核验：[来源 1](<https://github.com/sutdcv/SUTD-TrafficQA>)

### BlackSwanSuite

**2025 · CVPR** · [来源入口](<https://blackswan.cs.ubc.ca/>)

包含 1,655 段意外事件视频，构建预测、缺失事件溯因与新证据下假设修正任务。

- 评估协议：Forecaster 仅见事件前，Detective 见前后片段，Reporter 见完整视频；分别报告各任务和问题格式表现。
- 核验：[来源 1](<https://openaccess.thecvf.com/content/CVPR2025/papers/Chinchure_Black_Swan_Abductive_and_Defeasible_Video_Reasoning_in_Unpredictable_Events_CVPR_2025_paper.pdf>)

### MulA

**2026 · CVPR** · [来源入口](<https://openaccess.thecvf.com/content/CVPR2026/html/Zhang_ADSeeker_A_Knowledge-Grounded_Reasoning_Framework_for_Industry_Anomaly_Detection_and_CVPR_2026_paper.html>)

面向多类型图像异常的缺陷数据，覆盖不同物体类别和缺陷类型，用于检验类型级异常特征。

- 评估协议：按论文的零样本和缺陷类型设置评测，不将类别与缺陷类型混为同一统计口径。
- 核验：[来源 1](<https://openaccess.thecvf.com/content/CVPR2026/papers/Zhang_ADSeeker_A_Knowledge-Grounded_Reasoning_Framework_for_Industry_Anomaly_Detection_and_CVPR_2026_paper.pdf>)

### SEEK-M&V

**2026 · CVPR** · [来源入口](<https://openaccess.thecvf.com/content/CVPR2026/html/Zhang_ADSeeker_A_Knowledge-Grounded_Reasoning_Framework_for_Industry_Anomaly_Detection_and_CVPR_2026_paper.html>)

将 MVTec 和 VisA 场景的领域描述组织成图像—文档知识对，支撑异常知识检索与推理。

- 评估协议：作为 Q2K RAG 的知识来源；检索知识与查询图像角色分开，按论文规定的知识库范围比较。
- 核验：[来源 1](<https://openaccess.thecvf.com/content/CVPR2026/papers/Zhang_ADSeeker_A_Knowledge-Grounded_Reasoning_Framework_for_Industry_Anomaly_Detection_and_CVPR_2026_paper.pdf>)

### Expert-AD

**2026 · AAAI** · [来源入口](<https://ojs.aaai.org/index.php/AAAI/article/view/37588>)

从工业异常图像构建带正常性、缺陷类型、位置和推理内容的训练数据，用于感知激活微调。

- 评估协议：作为 PA-SFT 训练数据；训练和跨数据集测试分别统计，不把结构化匹配奖励视作独立因果解释评分。
- 核验：[来源 1](<https://ojs.aaai.org/index.php/AAAI/article/download/37588/41550>)
