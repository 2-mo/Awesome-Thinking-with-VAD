# 视频异常理解 · 论文年表

[按创新思路阅读](literature/catalog.md) · [方法比较](literature/comparison.md) · [数据集与评测](literature/benchmarks.md) · [阅读路线](literature/reading-guide.md) · [按会议查找](literature/venues.md) · [研究地图](https://2-mo.github.io/Awesome-Thinking-with-VAD/)

更新：2026-10-03 · 83 篇论文 · 7 个方法方向。

聚焦视频异常解释、推理、时序定位与理解评估，以及直接支撑这些目标的语义表征方法。会议与年份采用已核验的正式发表信息；未确认录用的论文保留 arXiv。

已配原论文图片 70 / 83 篇；点击图片查看大图。[图片来源与待补记录](assets/papers/README.md)。完整阅读关注见 [研究目录](literature/catalog.md)，作者、DOI 与 BibTeX 见 [引用导出](literature/citations.md)。

[2026](#year-2026) · [2025](#year-2025) · [2024](#year-2024) · [2023](#year-2023)

<a id="year-2026"></a>

## 2026

<a id="year-2026-aaai"></a>

### AAAI

<a id="paper-cuebench"></a>

#### CueBench: Advancing Unified Understanding of Context-Aware Video Anomalies in Real-World

[![AAAI](https://img.shields.io/badge/AAAI-2026-000080)](<https://ojs.aaai.org/index.php/AAAI/article/view/38209>)
[![Code](https://img.shields.io/github/stars/Mia-YatingYu/Cue-R1?style=social&label=Code&logo=github)](<https://github.com/Mia-YatingYu/Cue-R1>)

[论文](<https://ojs.aaai.org/index.php/AAAI/article/view/38209>) · [阅读笔记](<literature/catalog.md#paper-cuebench>) · [引用 / BibTeX](<literature/citations.md#cite-cuebench>)

> **CueBench / Cue-R1 · 上下文异常分类体系与分层奖励**
>
> 以条件性与绝对异常为核心组织场景、属性和事件层级，统一评估识别、定位、检测与预判，并训练具有分层可验证奖励的 Cue-R1。

[![CueBench 统一上下文异常评测框架及任务示例](assets/papers/cuebench.png)](assets/papers/cuebench.png)

*CueBench 的统一评测框架与上下文异常任务示例（原文 Figure 3）。 [原图 / PDF](<https://arxiv.org/html/2511.00613v1/evaluation_fig.png>) · [论文 / 作者页面](<https://arxiv.org/html/2511.00613v1>) · Yu, Yating et al. / CueBench*

---

<a id="paper-finevau"></a>

#### FineVAU: A Novel Human-Aligned Benchmark for Fine-Grained Video Anomaly Understanding

[![AAAI](https://img.shields.io/badge/AAAI-2026-000080)](<https://arxiv.org/abs/2601.17258>)
[![Project](https://img.shields.io/badge/Project-Website-537A7A)](<https://finevau.github.io/>)

[论文](<https://arxiv.org/abs/2601.17258>) · [阅读笔记](<literature/catalog.md#paper-finevau>) · [引用 / BibTeX](<literature/citations.md#cite-finevau>)

> **FineVAU · 关键视觉事实评估**
>
> 围绕事件、参与者与位置构建细粒度标注，并用 FVScore 检查关键视觉信息。

[![FineVAU 原论文两阶段细粒度标注流程](assets/papers/finevau.png)](assets/papers/finevau.png)

*FineVAU：关键视觉事实评估。原文 Figure 2。 [原图 / PDF](<https://arxiv.org/html/2601.17258v2/figs/Annotation_Pipeline.png>) · [论文 / 作者页面](<https://arxiv.org/html/2601.17258>) · Pereira, Joao Alexandre Cardeira et al. / FineVAU*

---

<a id="paper-headhunt-vad"></a>

#### HeadHunt-VAD: Hunting Robust Anomaly-Sensitive Heads in MLLM for Tuning-Free Video Anomaly Detection

[![AAAI](https://img.shields.io/badge/AAAI-2026-000080)](<https://arxiv.org/abs/2512.17601>)
[![Code](https://img.shields.io/github/stars/CebCai/HeadHunt-VAD?style=social&label=Code&logo=github)](<https://github.com/CebCai/HeadHunt-VAD>)

[论文](<https://arxiv.org/abs/2512.17601>) · [阅读笔记](<literature/catalog.md#paper-headhunt-vad>) · [引用 / BibTeX](<literature/citations.md#cite-headhunt-vad>)

> **HeadHunt-VAD · 稳定异常敏感注意力头探测**
>
> 从冻结 MLLM 内部筛选对多种提示稳定的异常敏感注意力头，再用轻量评分器和时序定位器读取其特征。

[![HeadHunt-VAD：图 3：离线识别异常敏感注意力头并用于在线检测。](assets/papers/headhunt-vad.png)](assets/papers/headhunt-vad.png)

*图 3：离线识别异常敏感注意力头并用于在线检测。 [原图 / PDF](<https://arxiv.org/html/2512.17601v2/headhuntfinalcmr.png>) · [论文 / 作者页面](<https://arxiv.org/html/2512.17601>) · Cai, Zhaolin et al. / HeadHunt-VAD*

---

<a id="paper-targetvau"></a>

#### TargetVAU: Multimodal Anomaly-Aware Reasoning for Target Behavior Understanding in Videos

[![AAAI](https://img.shields.io/badge/AAAI-2026-000080)](<https://ojs.aaai.org/index.php/AAAI/article/view/38378>)

[论文](<https://ojs.aaai.org/index.php/AAAI/article/view/38378>) · [阅读笔记](<literature/catalog.md#paper-targetvau>) · [引用 / BibTeX](<literature/citations.md#cite-targetvau>)

> **TargetVAU · 个体时空交互图与指令推理**
>
> 结合全局和人体中心视觉特征，用异常引导采样与时空交互图刻画个体关系，再以指令微调语言模型识别异常个体并解释行为。

[![TargetVAU：个体时空交互图与指令推理原论文图](assets/papers/targetvau.png)](assets/papers/targetvau.png)

*TargetVAU：个体时空交互图与指令推理（原文 Figure 2）。 [原图 / PDF](<https://manqing-zhang.github.io/publications/AAAI26.pdf>) · [论文 / 作者页面](<https://ojs.aaai.org/index.php/AAAI/article/view/38378>) · Zhou, Lingru et al. / TargetVAU*

---

<a id="paper-vagu-gts"></a>

#### VAGU & GtS: LLM-Based Benchmark and Framework for Joint Video Anomaly Grounding and Understanding

[![AAAI](https://img.shields.io/badge/AAAI-2026-000080)](<https://ojs.aaai.org/index.php/AAAI/article/view/42412>)

[论文](<https://ojs.aaai.org/index.php/AAAI/article/view/42412>) · [阅读笔记](<literature/catalog.md#paper-vagu-gts>) · [引用 / BibTeX](<literature/citations.md#cite-vagu-gts>)

> **VAGU & GtS · 先全局粗定位再局部细查**
>
> 先通过文本引导粗定位异常区间，再细查异常语义与时间边界，并构建 VAGU 和 JeAUG 联合评价定位与理解。

[![VAGU & GtS：图 4：通过文本引导先定位主事件，再进行细粒度异常理解。](assets/papers/vagu-gts.png)](assets/papers/vagu-gts.png)

*图 4：通过文本引导先定位主事件，再进行细粒度异常理解。 [原图 / PDF](<https://arxiv.org/html/2507.21507v1/model.png>) · [论文 / 作者页面](<https://arxiv.org/html/2507.21507>) · Gao, Shibo et al. / VAGU & GtS*

---


<a id="year-2026-acl"></a>

### ACL

<a id="paper-valu"></a>

#### VALU: A Benchmark for Video Anomaly Temporal Localization and Understanding at Multiple Semantic Levels

[![ACL](https://img.shields.io/badge/ACL-2026-537A7A)](<https://aclanthology.org/2026.acl-long.56/>)

[论文](<https://aclanthology.org/2026.acl-long.56/>) · [阅读笔记](<literature/catalog.md#paper-valu>) · [引用 / BibTeX](<literature/citations.md#cite-valu>)

> **VALU · 语义分层边界评估**
>
> 用多个语义层级定义异常边界，联合考查定位、时间 grounding 与细节辨别。

[![VALU：多层级异常标注示例原论文图](assets/papers/valu.png)](assets/papers/valu.png)

*多层级异常标注示例（原文 Figure 2）。 [原图 / PDF](<https://aclanthology.org/2026.acl-long.56.pdf>) · [论文 / 作者页面](<https://aclanthology.org/2026.acl-long.56/>) · Yixiao He et al. / VALU*

---


<a id="year-2026-acm-mm"></a>

### ACM MM

<a id="paper-avar"></a>

#### Advancing Video Anomaly Retrieval via Action-Focused Temporal Reasoning and Query-Adaptive Routing

[![ACM MM](https://img.shields.io/badge/ACM_MM-2026-FF69B4)](<https://2026.acmmm.org/site/technical-programme.html>)

[论文](<https://2026.acmmm.org/site/technical-programme.html>) · [阅读笔记](<literature/catalog.md#paper-avar>) · [引用 / BibTeX](<literature/citations.md#cite-avar>)

> **AVAR · 动作聚焦与查询自适应路由**
>
> 用文本动作语义定位目标，以双路径时序异常推理和查询自适应路由检索视频中的异常事件。

*配图待补：尚未取得可核验的论文原图。[查看检索记录](<assets/papers/README.md#missing-avar>)*

---

<a id="paper-cavge"></a>

#### Customized Anomalous Video Generation for Incremental Learning in Weakly-Supervised Video Anomaly Detection

[![ACM MM](https://img.shields.io/badge/ACM_MM-2026-FF69B4)](<https://2026.acmmm.org/site/technical-programme.html>)

[论文](<https://2026.acmmm.org/site/technical-programme.html>) · [阅读笔记](<literature/catalog.md#paper-cavge>) · [引用 / BibTeX](<literature/citations.md#cite-cavge>)

> **CAVGE · 脚本—关键帧—视频可控生成**
>
> 以语言脚本、文本引导关键帧和视频生成合成可定制异常，并用于弱监督视频异常检测的增量学习。

*配图待补：尚未取得可核验的论文原图。[查看检索记录](<assets/papers/README.md#missing-cavge>)*

---

<a id="paper-deal-vad"></a>

#### DEAL: Deep Evidential Audio-Visual Learning for Weakly Supervised Video Anomaly Detection

[![ACM MM](https://img.shields.io/badge/ACM_MM-2026-FF69B4)](<https://2026.acmmm.org/site/technical-programme.html>)

[论文](<https://2026.acmmm.org/site/technical-programme.html>) · [阅读笔记](<literature/catalog.md#paper-deal-vad>) · [引用 / BibTeX](<literature/citations.md#cite-deal-vad>)

> **DEAL · 音视频时序对齐与证据融合**
>
> 以状态空间模块对齐音视频，再用证据式双分支融合估计模态不确定性，降低噪声和时间错位的影响。

*配图待补：尚未取得可核验的论文原图。[查看检索记录](<assets/papers/README.md#missing-deal-vad>)*

---

<a id="paper-peer-vad"></a>

#### PEER-VAD: Prior-enhanced Event Refinement for Video Anomaly Detection

[![ACM MM](https://img.shields.io/badge/ACM_MM-2026-FF69B4)](<https://2026.acmmm.org/site/technical-programme.html>)
[![Code](https://img.shields.io/github/stars/HaochengY/PEER-VAD?style=social&label=Code&logo=github)](<https://github.com/HaochengY/PEER-VAD>)

[论文](<https://2026.acmmm.org/site/technical-programme.html>) · [阅读笔记](<literature/catalog.md#paper-peer-vad>) · [引用 / BibTeX](<literature/citations.md#cite-peer-vad>)

> **PEER-VAD · 先验消歧与双向事件细化**
>
> 对大模型给出的帧级或短片段异常预测做先验消歧和双向窗口分析，修复碎片化事件预测。

*配图待补：尚未取得可核验的论文原图。[查看检索记录](<assets/papers/README.md#missing-peer-vad>)*

---

<a id="paper-prime-vad"></a>

#### Rule-Guided Evolution of Hierarchical Reasoning for Explainable Video Anomaly Detection

[![ACM MM](https://img.shields.io/badge/ACM_MM-2026-FF69B4)](<https://2026.acmmm.org/site/technical-programme.html>)

[论文](<https://2026.acmmm.org/site/technical-programme.html>) · [阅读笔记](<literature/catalog.md#paper-prime-vad>) · [引用 / BibTeX](<literature/citations.md#cite-prime-vad>)

> **PRIME · 规则记忆与分层提示演化**
>
> 把场景、对象和推理提示拆成模块，从运行轨迹提炼规则记忆，并逐模块演化提示以获得可解释判据。

*配图待补：尚未取得可核验的论文原图。[查看检索记录](<assets/papers/README.md#missing-prime-vad>)*

---

<a id="paper-s2mgraph-vad"></a>

#### S2MGraph-VAD: Scene-to-Moment Graph-Guided Training-Free Video Anomaly Detection

[![ACM MM](https://img.shields.io/badge/ACM_MM-2026-FF69B4)](<https://2026.acmmm.org/site/technical-programme.html>)

[论文](<https://2026.acmmm.org/site/technical-programme.html>) · [阅读笔记](<literature/catalog.md#paper-s2mgraph-vad>) · [引用 / BibTeX](<literature/citations.md#cite-s2mgraph-vad>)

> **S2MGraph-VAD · 场景—片段图与稀疏查询**
>
> 先把长视频划分为场景，再细化到短时片段，通过场景到片段的图连接上下文与局部异常线索，稀疏调用大模型定位。

*配图待补：尚未取得可核验的论文原图。[查看检索记录](<assets/papers/README.md#missing-s2mgraph-vad>)*

---

<a id="paper-scene-dependent-vad"></a>

#### Scene-Dependent Video Anomaly Detection via Discriminative-Contrastive Learning from Intrinsic Scene Labels

[![ACM MM](https://img.shields.io/badge/ACM_MM-2026-FF69B4)](<https://2026.acmmm.org/site/technical-programme.html>)

[论文](<https://2026.acmmm.org/site/technical-programme.html>) · [阅读笔记](<literature/catalog.md#paper-scene-dependent-vad>) · [引用 / BibTeX](<literature/citations.md#cite-scene-dependent-vad>)

> **Scene-Dependent VAD · 场景标签判别与前景对比**
>
> 用内在场景标签训练事件分类，并以事件和场景的对比表示及距离构建场景依赖的异常分数。

*配图待补：尚未取得可核验的论文原图。[查看检索记录](<assets/papers/README.md#missing-scene-dependent-vad>)*

---

<a id="paper-upr-vad"></a>

#### UPR-VAD: Uncertainty-guided Signal Purification and Regularization for Weakly-Supervised Video Anomaly Detection

[![ACM MM](https://img.shields.io/badge/ACM_MM-2026-FF69B4)](<https://2026.acmmm.org/site/technical-programme.html>)
[![Code](https://img.shields.io/github/stars/WeiliangHuang-UM-connect/UPR-VAD?style=social&label=Code&logo=github)](<https://github.com/WeiliangHuang-UM-connect/UPR-VAD>)

[论文](<https://2026.acmmm.org/site/technical-programme.html>) · [阅读笔记](<literature/catalog.md#paper-upr-vad>) · [引用 / BibTeX](<literature/citations.md#cite-upr-vad>)

> **UPR-VAD · 不确定性净化与对比约束**
>
> 通过不确定性感知的信息瓶颈过滤模糊片段，结合视频内负例对比和稀疏置信约束学习弱监督异常信号。

*配图待补：尚未取得可核验的论文原图。[查看检索记录](<assets/papers/README.md#missing-upr-vad>)*

---

<a id="paper-vibes"></a>

#### Zoom In, Reason Out: Efficient Far-field Anomaly Detection in Expressway Surveillance Videos via Focused VLM Reasoning Guided by Bayesian Inference

[![ACM MM](https://img.shields.io/badge/ACM_MM-2026-FF69B4)](<https://2026.acmmm.org/site/technical-programme.html>)
[![Code](https://img.shields.io/github/stars/maoxiaowei97/VIBES?style=social&label=Code&logo=github)](<https://github.com/maoxiaowei97/VIBES>)

[论文](<https://2026.acmmm.org/site/technical-programme.html>) · [阅读笔记](<literature/catalog.md#paper-vibes>) · [引用 / BibTeX](<literature/citations.md#cite-vibes>)

> **VIBES · 贝叶斯轨迹触发与局部推理**
>
> 从高速公路车辆轨迹建立在线正常运动分布，贝叶斯偏离触发远景局部区域的视觉语言推理。

[![VIBES 贝叶斯异常提议与聚焦视觉语言推理框架（作者仓库）](assets/papers/vibes.png)](assets/papers/vibes.png)

*VIBES 贝叶斯异常提议与聚焦视觉语言推理框架（作者仓库） [原图 / PDF](<https://raw.githubusercontent.com/maoxiaowei97/VIBES/main/assets/fig/Model.png>) · [论文 / 作者页面](<https://github.com/maoxiaowei97/VIBES>) · Xiaowei Mao et al. / VIBES*

---

<a id="paper-vto"></a>

#### VTO: Visual Tool Orchestration for Video Anomaly Detection

[![ACM MM](https://img.shields.io/badge/ACM_MM-2026-FF69B4)](<https://2026.acmmm.org/site/technical-programme.html>)
[![Code](https://img.shields.io/github/stars/MICLAB-BUPT/VTO?style=social&label=Code&logo=github)](<https://github.com/MICLAB-BUPT/VTO>)

[论文](<https://2026.acmmm.org/site/technical-programme.html>) · [阅读笔记](<literature/catalog.md#paper-vto>) · [引用 / BibTeX](<literature/citations.md#cite-vto>)

> **VTO · 视觉工具编排与过程奖励**
>
> 让视频异常代理动态选择视觉工具，并用过程监督的强化学习及认知评价器引导多步证据获取。

[![VTO 视觉工具编排与过程监督强化学习框架（作者仓库）](assets/papers/vto.png)](assets/papers/vto.png)

*VTO 视觉工具编排与过程监督强化学习框架（作者仓库） [原图 / PDF](<https://raw.githubusercontent.com/MICLAB-BUPT/VTO/main/assets/vto_framework.png>) · [论文 / 作者页面](<https://github.com/MICLAB-BUPT/VTO>) · Rui Wang et al. / VTO*

---


<a id="year-2026-cvpr"></a>

### CVPR

<a id="paper-alert-clip"></a>

#### Alert-CLIP: Abnormality-aware Latent-Enhanced Representation Tuning of CLIP for Video Anomaly Detection

[![CVPR](https://img.shields.io/badge/CVPR-2026-1E90FF)](<https://openaccess.thecvf.com/content/CVPR2026/html/Zhu_Alert-CLIP_Abnormality-aware_Latent-Enhanced_Representation_Tuning_of_CLIP_for_Video_Anomaly_CVPR_2026_paper.html>)

[论文](<https://openaccess.thecvf.com/content/CVPR2026/html/Zhu_Alert-CLIP_Abnormality-aware_Latent-Enhanced_Representation_Tuning_of_CLIP_for_Video_Anomaly_CVPR_2026_paper.html>) · [阅读笔记](<literature/catalog.md#paper-alert-clip>) · [引用 / BibTeX](<literature/citations.md#cite-alert-clip>)

> **Alert-CLIP · 区域语义多层对齐**
>
> 通过视频与标签、区域与文本、区域与语义的多层对齐，增强视觉语言表征对正常和异常的区分能力。

[![Alert-CLIP：区域语义多层对齐原论文图](assets/papers/alert-clip.png)](assets/papers/alert-clip.png)

*Alert-CLIP：区域语义多层对齐（原文 Figure 3）。 [原图 / PDF](<https://openaccess.thecvf.com/content/CVPR2026/papers/Zhu_Alert-CLIP_Abnormality-aware_Latent-Enhanced_Representation_Tuning_of_CLIP_for_Video_Anomaly_CVPR_2026_paper.pdf>) · [论文 / 作者页面](<https://openaccess.thecvf.com/content/CVPR2026/html/Zhu_Alert-CLIP_Abnormality-aware_Latent-Enhanced_Representation_Tuning_of_CLIP_for_Video_Anomaly_CVPR_2026_paper.html>) · Zhu, Yiyan et al. / Alert-CLIP*

---

<a id="paper-las-vad"></a>

#### Weakly Supervised Video Anomaly Detection with Anomaly-Connected Components and Intention Reasoning

[![CVPR](https://img.shields.io/badge/CVPR-2026-1E90FF)](<https://openaccess.thecvf.com/content/CVPR2026/html/Wang_Weakly_Supervised_Video_Anomaly_Detection_with_Anomaly-Connected_Components_and_Intention_CVPR_2026_paper.html>)

[论文](<https://openaccess.thecvf.com/content/CVPR2026/html/Wang_Weakly_Supervised_Video_Anomaly_Detection_with_Anomaly-Connected_Components_and_Intention_CVPR_2026_paper.html>) · [阅读笔记](<literature/catalog.md#paper-las-vad>) · [引用 / BibTeX](<literature/citations.md#cite-las-vad>)

> **LAS-VAD · 语义连通与意图感知**
>
> 将视频帧组织为语义相连的成分，并结合行为意图与异常属性信息，区分外观相似的正常和异常行为。

[![LAS-VAD：语义连通与意图感知原论文图](assets/papers/las-vad.png)](assets/papers/las-vad.png)

*LAS-VAD：语义连通与意图感知（原文 Figure 1）。 [原图 / PDF](<https://openaccess.thecvf.com/content/CVPR2026/papers/Wang_Weakly_Supervised_Video_Anomaly_Detection_with_Anomaly-Connected_Components_and_Intention_CVPR_2026_paper.pdf>) · [论文 / 作者页面](<https://openaccess.thecvf.com/content/CVPR2026/html/Wang_Weakly_Supervised_Video_Anomaly_Detection_with_Anomaly-Connected_Components_and_Intention_CVPR_2026_paper.html>) · Wang, Yu et al. / LAS-VAD*

---

<a id="paper-lavida"></a>

#### No Need For Real Anomaly: MLLM Empowered Zero-Shot Video Anomaly Detection

[![CVPR](https://img.shields.io/badge/CVPR-2026-1E90FF)](<https://openaccess.thecvf.com/content/CVPR2026/html/Dai_No_Need_For_Real_Anomaly_MLLM_Empowered_Zero-Shot_Video_Anomaly_CVPR_2026_paper.html>)
[![Project](https://img.shields.io/badge/Project-Website-537A7A)](<https://github.com/VitaminCreed/LAVIDA>)

[论文](<https://openaccess.thecvf.com/content/CVPR2026/html/Dai_No_Need_For_Real_Anomaly_MLLM_Empowered_Zero-Shot_Video_Anomaly_CVPR_2026_paper.html>) · [阅读笔记](<literature/catalog.md#paper-lavida>) · [引用 / BibTeX](<literature/citations.md#cite-lavida>)

> **LAVIDA · 伪异常与反向注意力**
>
> 仅用伪异常训练，结合多模态大模型语义理解与反向注意力词元压缩，实现零样本帧级和像素级异常检测。

[![LAVIDA：伪异常与反向注意力原论文图](assets/papers/lavida.png)](assets/papers/lavida.png)

*LAVIDA：伪异常与反向注意力（原文 Figure 2）。 [原图 / PDF](<https://openaccess.thecvf.com/content/CVPR2026/papers/Dai_No_Need_For_Real_Anomaly_MLLM_Empowered_Zero-Shot_Video_Anomaly_CVPR_2026_paper.pdf>) · [论文 / 作者页面](<https://openaccess.thecvf.com/content/CVPR2026/html/Dai_No_Need_For_Real_Anomaly_MLLM_Empowered_Zero-Shot_Video_Anomaly_CVPR_2026_paper.html>) · Dai, Zunkai et al. / LAVIDA*

---

<a id="paper-stch"></a>

#### Streaming Video Crime Anticipation with Spatio-Temporal Causal Reasoning

[![CVPR](https://img.shields.io/badge/CVPR-2026-1E90FF)](<https://openaccess.thecvf.com/content/CVPR2026/html/Wang_Streaming_Video_Crime_Anticipation_with_Spatio-Temporal_Causal_Reasoning_CVPR_2026_paper.html>)

[论文](<https://openaccess.thecvf.com/content/CVPR2026/html/Wang_Streaming_Video_Crime_Anticipation_with_Spatio-Temporal_Causal_Reasoning_CVPR_2026_paper.html>) · [阅读笔记](<literature/catalog.md#paper-stch>) · [引用 / BibTeX](<literature/citations.md#cite-stch>)

> **STCH · 流式时空因果超图**
>
> 构建具有递进推理任务的 STCRC 基准，并用流式时空因果超图显式组织实体动态，支撑犯罪事件预判。

[![STCH：图 3：时空因果超图、记忆库与流式犯罪预测训练流程。](assets/papers/stch.png)](assets/papers/stch.png)

*图 3：时空因果超图、记忆库与流式犯罪预测训练流程。 [原图 / PDF](<https://openaccess.thecvf.com/content/CVPR2026/papers/Wang_Streaming_Video_Crime_Anticipation_with_Spatio-Temporal_Causal_Reasoning_CVPR_2026_paper.pdf>) · [论文 / 作者页面](<https://openaccess.thecvf.com/content/CVPR2026/html/Wang_Streaming_Video_Crime_Anticipation_with_Spatio-Temporal_Causal_Reasoning_CVPR_2026_paper.html>) · Wang, Yusong et al. / STCH*

---


<a id="year-2026-eccv"></a>

### ECCV

<a id="paper-clue-vad"></a>

#### CLUE-VAD: Structured Semantic Clues for Understanding Explainable Events in Video Anomaly Detection

[![ECCV](https://img.shields.io/badge/ECCV-2026-0B84FE)](<https://eccv.ecva.net/virtual/2026/poster/4744>)

[论文](<https://eccv.ecva.net/virtual/2026/poster/4744>) · [阅读笔记](<literature/catalog.md#paper-clue-vad>) · [引用 / BibTeX](<literature/citations.md#cite-clue-vad>)

> **CLUE-VAD · 结构化语义线索与类别感知归因**
>
> 将片段分解为动作、环境和对象线索，用类别感知权重融合并将异常分数归因到线索及关键词，生成有依据的解释。

[![CLUE-VAD：图 2：Witness、Detective 和 Reporter 模块将语义线索用于检测与解释。](assets/papers/clue-vad.png)](assets/papers/clue-vad.png)

*图 2：Witness、Detective 和 Reporter 模块将语义线索用于检测与解释。 [原图 / PDF](<https://media.eventhosts.cc/Conferences/ECCV2026/pdfs/7292.pdf>) · [论文 / 作者页面](<https://eccv.ecva.net/virtual/2026/poster/4744>) · MYOUNG-CHUL KIM et al. / CLUE-VAD*

---

<a id="paper-ewad"></a>

#### Towards Video Anomaly Detection from Event Streams: A Baseline and Benchmark Datasets

[![ECCV](https://img.shields.io/badge/ECCV-2026-0B84FE)](<https://eccv.ecva.net/virtual/2026/poster/5214>)
[![Code](https://img.shields.io/github/stars/kanyutingfeng/EWAD?style=social&label=Code&logo=github)](<https://github.com/kanyutingfeng/EWAD>)

[论文](<https://eccv.ecva.net/virtual/2026/poster/5214>) · [阅读笔记](<literature/catalog.md#paper-ewad>) · [引用 / BibTeX](<literature/citations.md#cite-ewad>)

> **EWAD · 事件密度采样与跨模态蒸馏**
>
> 构建同步事件流与 RGB 的视频异常基准，用事件密度引导采样、时序建模及 RGB 到事件流的蒸馏。

[![Figure 2: EWAD 事件流视频异常检测流程（作者预印本）](assets/papers/ewad.png)](assets/papers/ewad.png)

*Figure 2: EWAD 事件流视频异常检测流程（作者预印本） [原图 / PDF](<https://arxiv.org/html/2603.24991v2/pipeline4.png>) · [论文 / 作者页面](<https://arxiv.org/html/2603.24991v2>) · Peng Wu et al. / EWAD*

---

<a id="paper-o-vad"></a>

#### O-VAD: Industrial Video Anomaly Detection through Object-Centric Tracking and Reasoning

[![ECCV](https://img.shields.io/badge/ECCV-2026-0B84FE)](<https://arxiv.org/abs/2607.18142>)
[![Code](https://img.shields.io/github/stars/o-vad/O-VAD?style=social&label=Code&logo=github)](<https://github.com/o-vad/O-VAD>)

[论文](<https://arxiv.org/abs/2607.18142>) · [阅读笔记](<literature/catalog.md#paper-o-vad>) · [引用 / BibTeX](<literature/citations.md#cite-o-vad>)

> **O-VAD · 对象状态轨迹与时序推理**
>
> 在工业视频中跟踪对象状态随时间的演化，再对对象轨迹推理，定位异常对象与帧并输出异常过程和类型报告。

[![O-VAD：图 2：对象发现、状态跟踪与多步推理相结合的检测流程。](assets/papers/o-vad.png)](assets/papers/o-vad.png)

*图 2：对象发现、状态跟踪与多步推理相结合的检测流程。 [原图 / PDF](<https://raw.githubusercontent.com/o-vad/O-VAD/main/assets/framework.png>) · [论文 / 作者页面](<https://github.com/o-vad/O-VAD>) · Mei Yuan et al. / O-VAD*

---

<a id="paper-pa-vad"></a>

#### PA-VAD: Diffusion-Based Pseudo-Only Video Anomaly Detection via Domain-Aligned Memory Updates

[![ECCV](https://img.shields.io/badge/ECCV-2026-0B84FE)](<https://eccv.ecva.net/virtual/2026/poster/4682>)

[论文](<https://eccv.ecva.net/virtual/2026/poster/4682>) · [阅读笔记](<literature/catalog.md#paper-pa-vad>) · [引用 / BibTeX](<literature/citations.md#cite-pa-vad>)

> **PA-VAD · 伪异常生成与域对齐记忆**
>
> 用少量正常图像合成伪异常视频，与真实正常视频组成训练对；通过域对齐和记忆更新缓解合成异常的特征偏置。

[![Figure 3: PA-VAD 伪异常视频生成与域对齐记忆框架（作者预印本）](assets/papers/pa-vad.jpg)](assets/papers/pa-vad.jpg)

*Figure 3: PA-VAD 伪异常视频生成与域对齐记忆框架（作者预印本） [原图 / PDF](<https://arxiv.org/html/2512.06845v2/img/img3.jpg>) · [论文 / 作者页面](<https://arxiv.org/html/2512.06845v2>) · Satoshi Hashimoto et al. / PA-VAD*

---

<a id="paper-pistachio"></a>

#### Pistachio: Towards Synthetic, Balanced, and Long-Form Video Anomaly Benchmarks

[![ECCV](https://img.shields.io/badge/ECCV-2026-0B84FE)](<https://arxiv.org/abs/2511.19474>)
[![Project](https://img.shields.io/badge/Project-Website-537A7A)](<https://pistachio-video.github.io>)

[论文](<https://arxiv.org/abs/2511.19474>) · [阅读笔记](<literature/catalog.md#paper-pistachio>) · [引用 / BibTeX](<literature/citations.md#cite-pistachio>)

> **Pistachio · 可控生成与多事件评测**
>
> 通过可控场景、异常类型和时序叙事生成视频，构建检测与理解基准，包含事件级、视频级语义及多异常事件。

[![Pistachio：图 2：从场景和故事线生成异常视频及事件摘要。](assets/papers/pistachio.png)](assets/papers/pistachio.png)

*图 2：从场景和故事线生成异常视频及事件摘要。 [原图 / PDF](<https://arxiv.org/html/2511.19474v6/3.png>) · [论文 / 作者页面](<https://arxiv.org/html/2511.19474>) · Li, Jie et al. / Pistachio*

---

<a id="paper-reactvau"></a>

#### ReactVAU: A Slow-Fast Decoupled Framework for Streaming Video Anomaly Understanding

[![ECCV](https://img.shields.io/badge/ECCV-2026-0B84FE)](<https://arxiv.org/abs/2609.07941>)

[论文](<https://arxiv.org/abs/2609.07941>) · [阅读笔记](<literature/catalog.md#paper-reactvau>) · [引用 / BibTeX](<literature/citations.md#cite-reactvau>)

> **ReactVAU · 快慢解耦与异常持久记忆**
>
> 用轻量检测连续筛查视频，以异常感知记忆保留短暂证据，仅在可疑事件触发重型模型进行语义验证与原因描述。

[![ReactVAU：图 2：快速检测、异常持久记忆与慢速推理模块。](assets/papers/reactvau.png)](assets/papers/reactvau.png)

*图 2：快速检测、异常持久记忆与慢速推理模块。 [原图 / PDF](<https://arxiv.org/html/2609.07941v2/reactvau.png>) · [论文 / 作者页面](<https://arxiv.org/html/2609.07941>) · Chen, Chia-Hui et al. / ReactVAU*

---

<a id="paper-step-vad"></a>

#### STEP: Score-Based Temporal Energy for Human Pose Video Anomaly Detection

[![ECCV](https://img.shields.io/badge/ECCV-2026-0B84FE)](<https://eccv.ecva.net/virtual/2026/poster/3512>)

[论文](<https://eccv.ecva.net/virtual/2026/poster/3512>) · [阅读笔记](<literature/catalog.md#paper-step-vad>) · [引用 / BibTeX](<literature/citations.md#cite-step-vad>)

> **STEP · 姿态主成分空间的时序能量**
>
> 把连续人体姿态投影到白化主成分空间，利用去噪分数匹配学习正常序列能量，并用姿态置信度降低估计噪声影响。

[![Fig. 1：STEP 人体姿态时序能量异常检测流程（作者预印本，第 2 页）。](assets/papers/step-vad.png)](assets/papers/step-vad.png)

*Fig. 1：STEP 人体姿态时序能量异常检测流程（作者预印本，第 2 页）。 [原图 / PDF](<https://arxiv.org/pdf/2608.19987v1>) · [论文 / 作者页面](<https://arxiv.org/abs/2608.19987v1>) · Jakub Micorek et al. / STEP*

---

<a id="paper-trajvad"></a>

#### Bounding-Box Trajectories Matter for Video Anomaly Detection

[![ECCV](https://img.shields.io/badge/ECCV-2026-0B84FE)](<https://eccv.ecva.net/virtual/2026/poster/5841>)
[![Code](https://img.shields.io/github/stars/Songinpyo/TrajVAD-ECCV2026?style=social&label=Code&logo=github)](<https://github.com/Songinpyo/TrajVAD-ECCV2026>)

[论文](<https://eccv.ecva.net/virtual/2026/poster/5841>) · [阅读笔记](<literature/catalog.md#paper-trajvad>) · [引用 / BibTeX](<literature/citations.md#cite-trajvad>)

> **TrajVAD · 轨迹流模型与姿态可靠门控**
>
> 以目标框轨迹作为主要异常线索，用归一化流学习正常运动分布；另以可靠性门控结合人体姿态。

[![Figure 2: TrajVAD 轨迹与姿态分支异常检测框架（作者预印本）](assets/papers/trajvad.png)](assets/papers/trajvad.png)

*Figure 2: TrajVAD 轨迹与姿态分支异常检测框架（作者预印本） [原图 / PDF](<https://arxiv.org/html/2605.21957v2/figure2_architecture.png>) · [论文 / 作者页面](<https://arxiv.org/html/2605.21957v2>) · Inpyo Song et al. / TrajVAD*

---


<a id="year-2026-iclr"></a>

### ICLR

<a id="paper-lagovad"></a>

#### Language-guided Open-world Video Anomaly Detection under Weak Supervision

[![ICLR](https://img.shields.io/badge/ICLR-2026-4B0082)](<https://proceedings.iclr.cc/paper_files/paper/2026/hash/f88bec15cc4cb56b432ee040bb63f94f-Abstract-Conference.html>)
[![Code](https://img.shields.io/github/stars/Kamino666/LaGoVAD-PreVAD?style=social&label=Code&logo=github)](<https://github.com/Kamino666/LaGoVAD-PreVAD>)

[论文](<https://proceedings.iclr.cc/paper_files/paper/2026/hash/f88bec15cc4cb56b432ee040bb63f94f-Abstract-Conference.html>) · [阅读笔记](<literature/catalog.md#paper-lagovad>) · [引用 / BibTeX](<literature/citations.md#cite-lagovad>)

> **LaGoVAD · 自然语言条件化异常定义**
>
> 把用户给定的自然语言异常定义作为推理输入，结合动态视频合成与负样本对比学习训练模型，并建设带异常定义的 PreVAD 数据。

[![LaGoVAD：图 2：语言异常定义分支、动态视频合成与负样本对比学习。](assets/papers/lagovad.png)](assets/papers/lagovad.png)

*图 2：语言异常定义分支、动态视频合成与负样本对比学习。 [原图 / PDF](<https://proceedings.iclr.cc/paper_files/paper/2026/file/f88bec15cc4cb56b432ee040bb63f94f-Paper-Conference.pdf>) · [论文 / 作者页面](<https://proceedings.iclr.cc/paper_files/paper/2026/hash/f88bec15cc4cb56b432ee040bb63f94f-Abstract-Conference.html>) · Liu, Zihao et al. / LaGoVAD*

---

<a id="paper-steervad"></a>

#### Steering and Rectifying Latent Representation Manifolds in Frozen Multi-modal LLMs for Video Anomaly Detection

[![ICLR](https://img.shields.io/badge/ICLR-2026-4B0082)](<https://arxiv.org/abs/2602.24021>)

[论文](<https://arxiv.org/abs/2602.24021>) · [阅读笔记](<literature/catalog.md#paper-steervad>) · [引用 / BibTeX](<literature/citations.md#cite-steervad>)

> **SteerVAD · 潜在异常专家头与上下文表示校正**
>
> 以表示可分性筛选潜在异常专家头，训练层次元控制器按上下文缩放其表示；异常片段可交回冻结模型生成事后解释。

[![SteerVAD：图 3：选择潜在异常专家并校正特征以完成异常检测。](assets/papers/steervad.png)](assets/papers/steervad.png)

*图 3：选择潜在异常专家并校正特征以完成异常检测。 [原图 / PDF](<https://arxiv.org/html/2602.24021v1/framework4.png>) · [论文 / 作者页面](<https://arxiv.org/html/2602.24021>) · Cai, Zhaolin et al. / SteerVAD*

---


<a id="year-2026-icml"></a>

### ICML

<a id="paper-anom-pi"></a>

#### Learning to Watch: Active Video Anomaly Understanding via Interleaved Policy Optimization

[![ICML](https://img.shields.io/badge/ICML-2026-FF6B6B)](<https://arxiv.org/abs/2607.00622>)

[论文](<https://arxiv.org/abs/2607.00622>) · [阅读笔记](<literature/catalog.md#paper-anom-pi>) · [引用 / BibTeX](<literature/citations.md#cite-anom-pi>)

> **Anom-π · 交替推理与观察策略**
>
> 将推理与时间回溯、区间扩展、细粒度采样交替执行，学习主动获取证据的策略。

[![Anom-π 原论文框架图](assets/papers/anom-pi.png)](assets/papers/anom-pi.png)

*Anom-π：交替推理与观察策略。原文 Figure 2。 [原图 / PDF](<https://arxiv.org/html/2607.00622v1/fig2.png>) · [论文 / 作者页面](<https://arxiv.org/html/2607.00622>) · Mengjingcheng Mo et al. / Anom-π*

---

<a id="paper-cg-coe"></a>

#### Towards Trustworthy Video Anomaly Understanding: A Class-Guided Chain-of-Evaluation Metric and An Anomaly-focused Meta-Benchmark

[![ICML](https://img.shields.io/badge/ICML-2026-FF6B6B)](<https://icml.cc/virtual/2026/poster/66013>)

[论文](<https://icml.cc/virtual/2026/poster/66013>) · [阅读笔记](<literature/catalog.md#paper-cg-coe>) · [引用 / BibTeX](<literature/citations.md#cite-cg-coe>)

> **CG-CoE · 类别引导评价链与元评测**
>
> 以类别约束的异常事件抽取与匹配构建评价链，并用 AEA 与 CVP 子集检验指标有效性及措辞扰动鲁棒性。

[![CG-CoE：图 1：抽取异常事件并依据类别容忍边界进行组合匹配与评分。](assets/papers/cg-coe.png)](assets/papers/cg-coe.png)

*图 1：抽取异常事件并依据类别容忍边界进行组合匹配与评分。 [原图 / PDF](<https://raw.githubusercontent.com/mlresearch/v306/main/assets/leng26a/leng26a.pdf>) · [论文 / 作者页面](<https://proceedings.mlr.press/v306/leng26a.html>) · Jiaxu Leng et al. / CG-CoE*

---

<a id="paper-lrpo"></a>

#### Linguistic Relative Policy Optimization for Video Anomaly Reasoning

[![ICML](https://img.shields.io/badge/ICML-2026-FF6B6B)](<https://arxiv.org/abs/2607.00654>)

[论文](<https://arxiv.org/abs/2607.00654>) · [阅读笔记](<literature/catalog.md#paper-lrpo>) · [引用 / BibTeX](<literature/citations.md#cite-lrpo>)

> **LRPO · 组相对语言经验优化**
>
> 从多条推理轨迹的组内语义优势归纳通用与场景经验，以语言先验注入上下文而不更新模型参数。

[![LRPO：图 2：学习者与优化器通过语言交互优化异常判断经验。](assets/papers/lrpo.png)](assets/papers/lrpo.png)

*图 2：学习者与优化器通过语言交互优化异常判断经验。 [原图 / PDF](<https://arxiv.org/html/2607.00654v1/pipeline.png>) · [论文 / 作者页面](<https://arxiv.org/html/2607.00654>) · Jiaxu Leng et al. / LRPO*

---

<a id="paper-td-vad"></a>

#### TD-VAD: Breaking Visual Dependence in Video Anomaly Detection with Text-Driven Learning

[![ICML](https://img.shields.io/badge/ICML-2026-FF6B6B)](<https://arxiv.org/abs/2608.11820>)

[论文](<https://arxiv.org/abs/2608.11820>) · [阅读笔记](<literature/catalog.md#paper-td-vad>) · [引用 / BibTeX](<literature/citations.md#cite-td-vad>)

> **TD-VAD · 文本时序监督与事件演化注意力**
>
> 以 LLM 生成的时序事件文本训练检测器，通过事件演化因果注意力建模长短期依赖，推理时用冻结 CLIP 对齐视频。

[![TD-VAD：图 2：利用语言模型生成描述并训练异常检测器的流程。](assets/papers/td-vad.png)](assets/papers/td-vad.png)

*图 2：利用语言模型生成描述并训练异常检测器的流程。 [原图 / PDF](<https://arxiv.org/html/2608.11820v1/framework7.png>) · [论文 / 作者页面](<https://arxiv.org/html/2608.11820>) · Shuangqing Zhang et al. / TD-VAD*

---


<a id="year-2026-ijcai"></a>

### IJCAI

<a id="paper-memovad"></a>

#### MemoVAD: Resource-Efficient Video Anomaly Detection via Dynamic Semantic Memory in Edge Computing Scenarios

[![IJCAI](https://img.shields.io/badge/IJCAI-2026-537A7A)](<https://www.ijcai.org/proceedings/2026/618>)

[论文](<https://www.ijcai.org/proceedings/2026/618>) · [阅读笔记](<literature/catalog.md#paper-memovad>) · [引用 / BibTeX](<literature/citations.md#cite-memovad>)

> **MemoVAD · 不确定性门控与动态语义记忆**
>
> 边缘轻量检测器维护因果时序上下文，仅对高不确定且语义新颖的片段查询云端 VLM，并缓存验证后的语义原型用于后续检索。

[![MemoVAD：图 2：MemoVAD 的边缘检测、不确定性门控、动态语义记忆与云端验证。](assets/papers/memovad.png)](assets/papers/memovad.png)

*图 2：MemoVAD 的边缘检测、不确定性门控、动态语义记忆与云端验证。 [原图 / PDF](<https://www.ijcai.org/proceedings/2026/0618.pdf>) · [论文 / 作者页面](<https://www.ijcai.org/proceedings/2026/618>) · Guo Li et al. / MemoVAD*

---


<a id="year-2026-ijcv"></a>

### IJCV

<a id="paper-ecva-anomshield"></a>

#### Exploring What Why and How: A Multifaceted Benchmark for Causation Understanding of Video Anomaly

[![IJCV](https://img.shields.io/badge/IJCV-2026-537A7A)](<https://link.springer.com/article/10.1007/s11263-026-02983-0>)
[![Project](https://img.shields.io/badge/Project-Website-537A7A)](<https://github.com/Dulpy/ECVA>)

[论文](<https://link.springer.com/article/10.1007/s11263-026-02983-0>) · [阅读笔记](<literature/catalog.md#paper-ecva-anomshield>) · [引用 / BibTeX](<literature/citations.md#cite-ecva-anomshield>)

> **ECVA / AnomShield · 因果任务基准与关键片段推理**
>
> 围绕异常事件、原因与后果构建 ECVA 基准；AnomShield 通过思维链选取关键时段并建模时空依赖，AnomEval 用于评估异常理解回答。

[![Fig. 5: AnomShield 关键帧选择与视频异常因果理解架构（作者预印本）](assets/papers/ecva-anomshield.png)](assets/papers/ecva-anomshield.png)

*Fig. 5: AnomShield 关键帧选择与视频异常因果理解架构（作者预印本） [原图 / PDF](<https://arxiv.org/html/2412.07183v1/architecture_v7.png>) · [论文 / 作者页面](<https://arxiv.org/html/2412.07183v1>) · Hang Du et al. / ECVA / AnomShield*

---


<a id="year-2026-neurips"></a>

### NeurIPS

<a id="paper-ca-judge"></a>

#### CA-Judge: Teach Large Models to Judge Anomalies via Comparison for Video Anomaly Detection

[![NeurIPS](https://img.shields.io/badge/NeurIPS-2026-2DB55D)](<https://neurips.cc/virtual/2026/poster/153516>)

[论文](<https://neurips.cc/virtual/2026/poster/153516>) · [阅读笔记](<literature/catalog.md#paper-ca-judge>) · [引用 / BibTeX](<literature/citations.md#cite-ca-judge>)

> **CA-Judge · 比较驱动的异常判断**
>
> 依据题名，暂将其归为通过比较教导大模型判断视频异常的方法。

方法归类按题名暂定，[核验依据](<https://neurips.cc/Downloads/2026>)。

*配图待补：尚未取得可核验的论文原图。[查看检索记录](<assets/papers/README.md#missing-ca-judge>)*

---

<a id="paper-copra"></a>

#### COPRA: Conditional Parameter Adaptation with Reinforcement Learning for Video Anomaly Detection

[![NeurIPS](https://img.shields.io/badge/NeurIPS-2026-2DB55D)](<https://arxiv.org/abs/2605.15325>)

[论文](<https://arxiv.org/abs/2605.15325>) · [阅读笔记](<literature/catalog.md#paper-copra>) · [引用 / BibTeX](<literature/citations.md#cite-copra>)

> **COPRA · 片段条件化参数适配**
>
> 根据视频片段为冻结视觉语言模型生成输入条件化参数更新，并在训练和推理时使用同一适配方式。

[![Figure 2: COPRA 实例条件参数生成与强化学习框架（作者预印本）](assets/papers/copra.jpg)](assets/papers/copra.jpg)

*Figure 2: COPRA 实例条件参数生成与强化学习框架（作者预印本） [原图 / PDF](<https://arxiv.org/html/2605.15325v1/assets/main_pg_figure.jpg>) · [论文 / 作者页面](<https://arxiv.org/html/2605.15325v1>) · Darryl Cherian Jacob et al. / COPRA*

---

<a id="paper-road"></a>

#### ROAD: Rule-Grounded Context-Aware Open-World Driver Anomaly Detection

[![NeurIPS](https://img.shields.io/badge/NeurIPS-2026-2DB55D)](<https://neurips.cc/virtual/2026/poster/154937>)

[论文](<https://neurips.cc/virtual/2026/poster/154937>) · [阅读笔记](<literature/catalog.md#paper-road>) · [引用 / BibTeX](<literature/citations.md#cite-road>)

> **ROAD · 规则约束与上下文判断**
>
> 依据题名，暂将其归为规则约束、上下文感知的开放世界驾驶员异常检测。

方法归类按题名暂定，[核验依据](<https://neurips.cc/Downloads/2026>)。

*配图待补：尚未取得可核验的论文原图。[查看检索记录](<assets/papers/README.md#missing-road>)*

---

<a id="paper-seek-vau"></a>

#### SEEK-VAU: Towards Evidence-Faithful Video Anomaly Understanding via Agentic Search

[![NeurIPS](https://img.shields.io/badge/NeurIPS-2026-2DB55D)](<https://neurips.cc/virtual/2026/poster/152176>)

[论文](<https://neurips.cc/virtual/2026/poster/152176>) · [阅读笔记](<literature/catalog.md#paper-seek-vau>) · [引用 / BibTeX](<literature/citations.md#cite-seek-vau>)

> **SEEK-VAU · 智能体搜索与证据忠实性**
>
> 依据题名，暂将其归为通过智能体搜索获取证据的视频异常理解方法。

方法归类按题名暂定，[核验依据](<https://neurips.cc/Downloads/2026>)。

*配图待补：尚未取得可核验的论文原图。[查看检索记录](<assets/papers/README.md#missing-seek-vau>)*

---

<a id="paper-spherevad"></a>

#### SphereVAD: Training-Free Video Anomaly Detection via Geodesic Inference on the Unit Hypersphere

[![NeurIPS](https://img.shields.io/badge/NeurIPS-2026-2DB55D)](<https://neurips.cc/virtual/2026/poster/152043>)

[论文](<https://neurips.cc/virtual/2026/poster/152043>) · [阅读笔记](<literature/catalog.md#paper-spherevad>) · [引用 / BibTeX](<literature/citations.md#cite-spherevad>)

> **SphereVAD · 超球面测地异常推断**
>
> 在单位超球面上对视频特征做中心化与场景注意力建模，用测地距离和方向分布进行免训练异常评分。

[![Figure 2: SphereVAD 单位超球面测地推理流程（作者预印本）](assets/papers/spherevad.png)](assets/papers/spherevad.png)

*Figure 2: SphereVAD 单位超球面测地推理流程（作者预印本） [原图 / PDF](<https://arxiv.org/html/2605.08003v1/pipeline.png>) · [论文 / 作者页面](<https://arxiv.org/html/2605.08003v1>) · Chao Huang et al. / SphereVAD*

---

<a id="paper-tar-bench"></a>

#### From Detection to Understanding — A Multi-Task Dataset for Traffic Anomaly Reasoning

[![NeurIPS](https://img.shields.io/badge/NeurIPS-2026-2DB55D)](<https://neurips.cc/virtual/2026/poster/139099>)

Evaluations and Datasets · [论文](<https://neurips.cc/virtual/2026/poster/139099>) · [阅读笔记](<literature/catalog.md#paper-tar-bench>) · [引用 / BibTeX](<literature/citations.md#cite-tar-bench>)

> **TAR / TAR-Bench · 交通异常多任务推理基准**
>
> 面向交通异常构建训练集 TAR 与人工核验的 TAR-Bench，覆盖问答、时序推理及场景理解等十类任务。

[![Figure 1: TAR 与 TAR-Bench 交通异常多任务标注示意（作者预印本）](assets/papers/tar-bench.png)](assets/papers/tar-bench.png)

*Figure 1: TAR 与 TAR-Bench 交通异常多任务标注示意（作者预印本） [原图 / PDF](<https://arxiv.org/html/2608.10317v3/TAR-teaser-2a-.png>) · [论文 / 作者页面](<https://arxiv.org/html/2608.10317v3>) · Han Zhang et al. / TAR / TAR-Bench*

---


<a id="year-2026-tnnls"></a>

### TNNLS

<a id="paper-promptvad"></a>

#### PromptVAD: Abnormal Prompt via Vision-Language Model

[![TNNLS](https://img.shields.io/badge/TNNLS-2026-537A7A)](<https://doi.org/10.1109/tnnls.2025.3621336>)

[论文](<https://doi.org/10.1109/tnnls.2025.3621336>) · [阅读笔记](<literature/catalog.md#paper-promptvad>) · [引用 / BibTeX](<literature/citations.md#cite-promptvad>)

> **PromptVAD · 领域与类别提示协同学习**
>
> 结合可学习的领域提示、类别提示与固定类别定义，缩小视觉特征和异常类别的语义差距，联合学习粗粒度与细粒度异常检测。

*配图待补：尚未取得可核验的论文原图。[查看检索记录](<assets/papers/README.md#missing-promptvad>)*

---


<a id="year-2026-tpami"></a>

### TPAMI

<a id="paper-adversa"></a>

#### ADVersa: Abductive Driving Accident Video Understanding

[![TPAMI](https://img.shields.io/badge/TPAMI-2026-537A7A)](<https://doi.org/10.1109/TPAMI.2026.3663545>)

[论文](<https://doi.org/10.1109/TPAMI.2026.3663545>) · [阅读笔记](<literature/catalog.md#paper-adversa>) · [引用 / BibTeX](<literature/citations.md#cite-adversa>)

> **ADVersa · 关系感知的跨模态事故溯因**
>
> 用 Abductive CLIP 与关系感知的对比图视频预训练组织跨模态证据，为缺失的近事故场景推断图像和文本解释，并支持过去恢复、未来预测及原因条件的视频生成。

[![ADVersa 物体中心与关系中心事故视频扩散及文本推理框架](assets/papers/adversa.png)](assets/papers/adversa.png)

*ADVersa 事故视频扩散与文本推理框架（作者主页所示 TPAMI 2026 版本） [原图 / PDF](<https://doc-doc.github.io/cv/assets/images/research/arxiv/ADVersa.png>) · [论文 / 作者页面](<https://doc-doc.github.io/cv/>) · Lei-Lei Li et al. / ADVersa*

---


<a id="year-2026-arxiv"></a>

### arXiv · 预印本

<a id="paper-agenticvau"></a>

#### AgenticVAU: Multi-Agent Explore-Verify Reasoning for Video Anomaly Understanding

[![arXiv](https://img.shields.io/badge/arXiv-2026-b31b1b?logo=arxiv)](<https://arxiv.org/abs/2608.03779>)

[论文](<https://arxiv.org/abs/2608.03779>) · [阅读笔记](<literature/catalog.md#paper-agenticvau>) · [引用 / BibTeX](<literature/citations.md#cite-agenticvau>)

> **AgenticVAU · 多智能体探索验证与证据记忆**
>
> 用规则构建、搜索规划、视频观察和最终决策四类智能体，交替探索疑点与局部验证，并通过共享证据记忆协调判断。

[![AgenticVAU：图 2：多智能体通过探索、观察、证据记忆与验证形成异常判断。](assets/papers/agenticvau.png)](assets/papers/agenticvau.png)

*图 2：多智能体通过探索、观察、证据记忆与验证形成异常判断。 [原图 / PDF](<https://arxiv.org/html/2608.03779v1/method.png>) · [论文 / 作者页面](<https://arxiv.org/html/2608.03779>) · Duan, Yuxiang et al. / AgenticVAU*

---

<a id="paper-anomalycraft"></a>

#### AnomalyCraft-700K: Component-Level Controllable and Verifiable Synthetic Anomalies for Fine-Grained Video Anomaly Understanding

[![arXiv](https://img.shields.io/badge/arXiv-2026-b31b1b?logo=arxiv)](<https://arxiv.org/abs/2609.06978>)
[![Project](https://img.shields.io/badge/Project-Website-537A7A)](<https://github.com/Eagen-l/AnomalyCraft>)

[论文](<https://arxiv.org/abs/2609.06978>) · [阅读笔记](<literature/catalog.md#paper-anomalycraft>) · [引用 / BibTeX](<literature/citations.md#cite-anomalycraft>)

> **AnomalyCraft-700K · 语义组件控制生成与逐项验证**
>
> 以细粒度语义组件控制异常视频生成，逐组件校正文图不一致，并构造类别相关的困难正常样本，支持多任务异常理解。

[![AnomalyCraft-700K：图 2：AnomalyCraft 视频异常数据生成流程。](assets/papers/anomalycraft.png)](assets/papers/anomalycraft.png)

*图 2：AnomalyCraft 视频异常数据生成流程。 [原图 / PDF](<https://arxiv.org/html/2609.06978v1/Figure2_Draft.png>) · [论文 / 作者页面](<https://arxiv.org/html/2609.06978>) · Long, Yuzhou et al. / AnomalyCraft-700K*

---

<a id="paper-probe-vad"></a>

#### Probe-VAD: Ordinal Likelihood Probing for Training-Free Video Anomaly Detection

[![arXiv](https://img.shields.io/badge/arXiv-2026-b31b1b?logo=arxiv)](<https://arxiv.org/abs/2609.17211>)
[![Code](https://img.shields.io/github/stars/yvestine/Probe-VAD?style=social&label=Code&logo=github)](<https://github.com/yvestine/Probe-VAD>)

[论文](<https://arxiv.org/abs/2609.17211>) · [阅读笔记](<literature/catalog.md#paper-probe-vad>) · [引用 / BibTeX](<literature/citations.md#cite-probe-vad>)

> **Probe-VAD · 序数语言探测与一致性评分**
>
> 以冻结 VLM 对有序严重程度阈值作是／否判断，通过续写似然与序数一致性约束得到连续异常分数。

[![Probe-VAD：图 2：利用有序严重程度阈值探测视觉语言模型并汇总异常分数。](assets/papers/probe-vad.png)](assets/papers/probe-vad.png)

*图 2：利用有序严重程度阈值探测视觉语言模型并汇总异常分数。 [原图 / PDF](<https://arxiv.org/html/2609.17211v1/ProbeVAD_flowchart.png>) · [论文 / 作者页面](<https://arxiv.org/html/2609.17211>) · Gu, Jiawei et al. / Probe-VAD*

---

<a id="paper-srvau-r1"></a>

#### SRVAU-R1: Enhancing Video Anomaly Understanding via Reflection-Aware Learning

[![arXiv](https://img.shields.io/badge/arXiv-2026-b31b1b?logo=arxiv)](<https://arxiv.org/abs/2602.01004>)

[论文](<https://arxiv.org/abs/2602.01004>) · [阅读笔记](<literature/catalog.md#paper-srvau-r1>) · [引用 / BibTeX](<literature/citations.md#cite-srvau-r1>)

> **SRVAU-R1 · 反思修正序列训练**
>
> 构建初始推理、反思和修正推理的监督序列，再结合监督与强化微调。

[![SRVAU-R1 原论文框架图](assets/papers/srvau-r1.png)](assets/papers/srvau-r1.png)

*SRVAU-R1：反思修正序列训练。原文 Figure 2。 [原图 / PDF](<https://arxiv.org/html/2602.01004v1/figure2.png>) · [论文 / 作者页面](<https://arxiv.org/html/2602.01004>) · Zhao, Zihao et al. / SRVAU-R1*

---

<a id="paper-tau-bench"></a>

#### TAU-Bench: From Anomaly Instance Tracking to Fine-Grained Video Anomaly Understanding

[![arXiv](https://img.shields.io/badge/arXiv-2026-b31b1b?logo=arxiv)](<https://arxiv.org/abs/2608.05699>)
[![Project](https://img.shields.io/badge/Project-Website-537A7A)](<https://yarkupa.github.io/tau-bench.github.io/>)

[论文](<https://arxiv.org/abs/2608.05699>) · [阅读笔记](<literature/catalog.md#paper-tau-bench>) · [引用 / BibTeX](<literature/citations.md#cite-tau-bench>)

> **TAU-Bench · 实例轨迹与层级语义联合评估**
>
> 把异常实例轨迹、像素掩码与实例、事件、场景三级描述绑定，联合评估跟踪和细粒度理解是否指向同一异常对象。

[![TAU-Bench：图 1：关联异常实例轨迹与细粒度语义理解的基准概览。](assets/papers/tau-bench.png)](assets/papers/tau-bench.png)

*图 1：关联异常实例轨迹与细粒度语义理解的基准概览。 [原图 / PDF](<https://yarkupa.github.io/tau-bench.github.io/assets/overview.png>) · [论文 / 作者页面](<https://yarkupa.github.io/tau-bench.github.io/>) · Yang, Kepeng et al. / TAU-Bench*

---

<a id="paper-vad-r1-plus"></a>

#### Advancing Adaptive Multi-Stage Video Anomaly Reasoning: A Benchmark Dataset and Method

[![arXiv](https://img.shields.io/badge/arXiv-2026-b31b1b?logo=arxiv)](<https://arxiv.org/abs/2601.10165>)
[![Project](https://img.shields.io/badge/Project-Website-537A7A)](<https://github.com/wbfwonderful/Vad-R1-Plus>)

[论文](<https://arxiv.org/abs/2601.10165>) · [阅读笔记](<literature/catalog.md#paper-vad-r1-plus>) · [引用 / BibTeX](<literature/citations.md#cite-vad-r1-plus>)

> **Vad-R1-Plus · 感知认知行动链与异常感知优化**
>
> 用感知、认知与行动三级思维链组织异常推理，并以异常感知的组相对策略优化训练支持不同推理深度和风险判断的模型。

[![Vad-R1-Plus：图 2：结构化思维链与自适应混合推理机制。](assets/papers/vad-r1-plus.png)](assets/papers/vad-r1-plus.png)

*图 2：结构化思维链与自适应混合推理机制。 [原图 / PDF](<https://arxiv.org/html/2601.10165v1/plus-fig-cot.png>) · [论文 / 作者页面](<https://arxiv.org/html/2601.10165>) · Huang, Chao et al. / Vad-R1-Plus*

---


<a id="year-2025"></a>

## 2025

<a id="year-2025-aaai"></a>

### AAAI

<a id="paper-varcmp"></a>

#### VarCMP: Adapting Cross-Modal Pre-Training Models for Video Anomaly Retrieval

[![AAAI](https://img.shields.io/badge/AAAI-2025-000080)](<https://doi.org/10.1609/aaai.v39i8.32909>)

[论文](<https://doi.org/10.1609/aaai.v39i8.32909>) · [阅读笔记](<literature/catalog.md#paper-varcmp>) · [引用 / BibTeX](<literature/citations.md#cite-varcmp>)

> **VarCMP · 层级跨模态对齐与异常加权**
>
> 将跨模态预训练模型用于长视频异常检索，通过统一层级对齐和异常偏置加权，匹配视频与文本或音频查询。

[![VarCMP：图 1：统一层级跨模态对齐与异常偏置加权的视频异常检索框架。](assets/papers/varcmp.png)](assets/papers/varcmp.png)

*图 1：统一层级跨模态对齐与异常偏置加权的视频异常检索框架。 [原图 / PDF](<https://ojs.aaai.org/index.php/AAAI/article/download/32909/35064>) · [论文 / 作者页面](<https://ojs.aaai.org/index.php/AAAI/article/view/32909>) · Wu, Peng et al. / VarCMP*

---


<a id="year-2025-acm-mm"></a>

### ACM MM

<a id="paper-eventvad"></a>

#### EventVAD: Training-Free Event-Aware Video Anomaly Detection

[![ACM MM](https://img.shields.io/badge/ACM_MM-2025-FF69B4)](<https://arxiv.org/abs/2504.13092>)
[![Code](https://img.shields.io/github/stars/YihuaJerry/EventVAD?style=social&label=Code&logo=github)](<https://github.com/YihuaJerry/EventVAD>)

[论文](<https://arxiv.org/abs/2504.13092>) · [阅读笔记](<literature/catalog.md#paper-eventvad>) · [引用 / BibTeX](<literature/citations.md#cite-eventvad>)

> **EventVAD · 时空图划分事件边界**
>
> 利用动态时空图寻找事件边界，再通过层级提示完成事件级异常推理。

[![EventVAD 原论文框架图](assets/papers/eventvad.png)](assets/papers/eventvad.png)

*EventVAD：时空图划分事件边界。原文 Figure 2。 [原图 / PDF](<https://arxiv.org/html/2504.13092v3/pipeline.png>) · [论文 / 作者页面](<https://arxiv.org/html/2504.13092>) · Shao, Yihua et al. / EventVAD*

---

<a id="paper-hiprobe-vad"></a>

#### HiProbe-VAD: Video Anomaly Detection via Hidden States Probing in Tuning-Free Multimodal LLMs

[![ACM MM](https://img.shields.io/badge/ACM_MM-2025-FF69B4)](<https://doi.org/10.1145/3746027.3755575>)

[论文](<https://doi.org/10.1145/3746027.3755575>) · [阅读笔记](<literature/catalog.md#paper-hiprobe-vad>) · [引用 / BibTeX](<literature/citations.md#cite-hiprobe-vad>)

> **HiProbe-VAD · 中间层探测与轻量异常评分**
>
> 从冻结多模态大模型的中间隐藏状态选择异常敏感层，训练轻量逻辑回归评分器，并结合时序定位与文本解释分析异常。

[![HiProbe-VAD：图 5：离线隐藏状态探测和评分器训练，以及在线评分与定位。](assets/papers/hiprobe-vad.png)](assets/papers/hiprobe-vad.png)

*图 5：离线隐藏状态探测和评分器训练，以及在线评分与定位。 [原图 / PDF](<https://arxiv.org/html/2507.17394v1/framework.png>) · [论文 / 作者页面](<https://arxiv.org/html/2507.17394>) · Cai, Zhaolin et al. / HiProbe-VAD*

---

<a id="paper-holotrace"></a>

#### HoloTrace: LLM-based Bidirectional Causal Knowledge Graph for Edge-Cloud Video Anomaly Detection

[![ACM MM](https://img.shields.io/badge/ACM_MM-2025-FF69B4)](<https://doi.org/10.1145/3746027.3755185>)
[![Code](https://img.shields.io/github/stars/kongyanye/HoloTrace-MM25?style=social&label=Code&logo=github)](<https://github.com/kongyanye/HoloTrace-MM25>)

[论文](<https://doi.org/10.1145/3746027.3755185>) · [阅读笔记](<literature/catalog.md#paper-holotrace>) · [引用 / BibTeX](<literature/citations.md#cite-holotrace>)

> **HoloTrace · 双向因果知识图与边云更新**
>
> 用 LLM 构建并更新双向因果知识图，边缘侧结合隐马尔可夫模型进行事件推理与边界判断，云端依据关键帧更新事件关系。

[![HoloTrace：边云协同的双向因果知识图异常检测系统。](assets/papers/holotrace.png)](assets/papers/holotrace.png)

*边云协同的双向因果知识图异常检测系统。 [原图 / PDF](<https://raw.githubusercontent.com/kongyanye/HoloTrace-MM25/main/assets/system_overview.png>) · [论文 / 作者页面](<https://github.com/kongyanye/HoloTrace-MM25>) · Wang, Hanling et al. / HoloTrace*

---


<a id="year-2025-cvpr"></a>

### CVPR

<a id="paper-anomize"></a>

#### Anomize: Better Open Vocabulary Video Anomaly Detection

[![CVPR](https://img.shields.io/badge/CVPR-2025-1E90FF)](<https://openaccess.thecvf.com/content/CVPR2025/html/Li_Anomize_Better_Open_Vocabulary_Video_Anomaly_Detection_CVPR_2025_paper.html>)

[论文](<https://openaccess.thecvf.com/content/CVPR2025/html/Li_Anomize_Better_Open_Vocabulary_Video_Anomaly_Detection_CVPR_2025_paper.html>) · [阅读笔记](<literature/catalog.md#paper-anomize>) · [引用 / BibTeX](<literature/citations.md#cite-anomize>)

> **Anomize · 多源语义与标签关系**
>
> 结合多层视觉信息与匹配文本，并利用标签关系编码新类别，改善未见异常的检测和语义分类。

[![Anomize：多源语义与标签关系原论文图](assets/papers/anomize.png)](assets/papers/anomize.png)

*Anomize：多源语义与标签关系（原文 Figure 3）。 [原图 / PDF](<https://openaccess.thecvf.com/content/CVPR2025/papers/Li_Anomize_Better_Open_Vocabulary_Video_Anomaly_Detection_CVPR_2025_paper.pdf>) · [论文 / 作者页面](<https://openaccess.thecvf.com/content/CVPR2025/html/Li_Anomize_Better_Open_Vocabulary_Video_Anomaly_Detection_CVPR_2025_paper.html>) · Li, Fei et al. / Anomize*

---

<a id="paper-holmes-vau"></a>

#### Holmes-VAU: Towards Long-term Video Anomaly Understanding at Any Granularity

[![CVPR](https://img.shields.io/badge/CVPR-2025-1E90FF)](<https://arxiv.org/abs/2412.06171>)
[![Code](https://img.shields.io/github/stars/pipixin321/HolmesVAU?style=social&label=Code&logo=github)](<https://github.com/pipixin321/HolmesVAU>)

[论文](<https://arxiv.org/abs/2412.06171>) · [阅读笔记](<literature/catalog.md#paper-holmes-vau>) · [引用 / BibTeX](<literature/citations.md#cite-holmes-vau>)

> **Holmes-VAU · 多粒度指令与采样**
>
> 构建多粒度异常指令数据，并用异常聚焦采样连接片段、事件和整段视频的理解。

[![Holmes-VAU 原论文框架图](assets/papers/holmes-vau.png)](assets/papers/holmes-vau.png)

*Holmes-VAU：多粒度指令与采样。原文 Figure 4。 [原图 / PDF](<https://arxiv.org/html/2412.06171v2/fig_framework.png>) · [论文 / 作者页面](<https://arxiv.org/html/2412.06171>) · Zhang, Huaxin et al. / Holmes-VAU*

---

<a id="paper-vera"></a>

#### VERA: Explainable Video Anomaly Detection via Verbalized Learning of Vision-Language Models

[![CVPR](https://img.shields.io/badge/CVPR-2025-1E90FF)](<https://arxiv.org/abs/2412.01095>)
[![Code](https://img.shields.io/github/stars/vera-framework/VERA?style=social&label=Code&logo=github)](<https://github.com/vera-framework/VERA>)

[论文](<https://arxiv.org/abs/2412.01095>) · [阅读笔记](<literature/catalog.md#paper-vera>) · [引用 / BibTeX](<literature/citations.md#cite-vera>)

> **VERA · 语言反馈优化问题**
>
> 用视频级弱标签优化自然语言引导问题，让冻结 VLM 分解并判断异常模式。

[![VERA 原论文框架图](assets/papers/vera.png)](assets/papers/vera.png)

*VERA：语言反馈优化问题。原文 Figure 2。 [原图 / PDF](<https://arxiv.org/html/2412.01095v3/fig2.png>) · [论文 / 作者页面](<https://arxiv.org/html/2412.01095>) · Ye, Muchao et al. / VERA*

---


<a id="year-2025-iccv"></a>

### ICCV

<a id="paper-va-gpt"></a>

#### Aligning Effective Tokens with Video Anomaly in Large Language Models

[![ICCV](https://img.shields.io/badge/ICCV-2025-00CED1)](<https://openaccess.thecvf.com/content/ICCV2025/html/Chen_Aligning_Effective_Tokens_with_Video_Anomaly_in_Large_Language_Models_ICCV_2025_paper.html>)

[论文](<https://openaccess.thecvf.com/content/ICCV2025/html/Chen_Aligning_Effective_Tokens_with_Video_Anomaly_in_Large_Language_Models_ICCV_2025_paper.html>) · [阅读笔记](<literature/catalog.md#paper-va-gpt>) · [引用 / BibTeX](<literature/citations.md#cite-va-gpt>)

> **VA-GPT · 时空有效词元对齐**
>
> 通过空间有效词元选择与时间有效词元生成，减少冗余视觉信息，支持异常总结和时间定位。

[![VA-GPT：时空有效词元对齐原论文图](assets/papers/va-gpt.png)](assets/papers/va-gpt.png)

*VA-GPT：时空有效词元对齐（原文 Figure 2）。 [原图 / PDF](<https://openaccess.thecvf.com/content/ICCV2025/papers/Chen_Aligning_Effective_Tokens_with_Video_Anomaly_in_Large_Language_Models_ICCV_2025_paper.pdf>) · [论文 / 作者页面](<https://openaccess.thecvf.com/content/ICCV2025/html/Chen_Aligning_Effective_Tokens_with_Video_Anomaly_in_Large_Language_Models_ICCV_2025_paper.html>) · Chen, Yingxian et al. / VA-GPT*

---


<a id="year-2025-icml"></a>

### ICML

<a id="paper-ex-vad"></a>

#### Ex-VAD: Explainable Fine-grained Video Anomaly Detection Based on Visual-Language Models

[![ICML](https://img.shields.io/badge/ICML-2025-FF6B6B)](<https://proceedings.mlr.press/v267/huang25ad.html>)

[论文](<https://proceedings.mlr.press/v267/huang25ad.html>) · [阅读笔记](<literature/catalog.md#paper-ex-vad>) · [引用 / BibTeX](<literature/citations.md#cite-ex-vad>)

> **Ex-VAD · 解释融合与标签对齐**
>
> 由帧字幕生成视频级异常解释，再结合视觉特征与标签增强对齐进行细粒度检测。

[![Ex-VAD：解释融合与标签对齐原论文图](assets/papers/ex-vad.png)](assets/papers/ex-vad.png)

*Ex-VAD：解释融合与标签对齐（原文 Figure 2）。 [原图 / PDF](<https://raw.githubusercontent.com/mlresearch/v267/main/assets/huang25ad/huang25ad.pdf>) · [论文 / 作者页面](<https://proceedings.mlr.press/v267/huang25ad.html>) · Chao Huang et al. / Ex-VAD*

---


<a id="year-2025-naacl-findings"></a>

### NAACL Findings

<a id="paper-vane-bench"></a>

#### VANE-Bench: Video Anomaly Evaluation Benchmark for Conversational LMMs

[![NAACL Findings](https://img.shields.io/badge/NAACL_Findings-2025-537A7A)](<https://aclanthology.org/2025.findings-naacl.171/>)
[![Project](https://img.shields.io/badge/Project-Website-537A7A)](<https://github.com/rohit901/VANE-Bench>)

[论文](<https://aclanthology.org/2025.findings-naacl.171/>) · [阅读笔记](<literature/catalog.md#paper-vane-bench>) · [引用 / BibTeX](<literature/citations.md#cite-vane-bench>)

> **VANE-Bench · 合成与真实异常问答评测**
>
> 以视频问答评测模型对异常的检测和定位，同时覆盖五类合成视频不一致性与真实世界异常。

[![VANE-Bench：VANE-Bench 的视频异常评测与问答构建流程。](assets/papers/vane-bench.png)](assets/papers/vane-bench.png)

*VANE-Bench 的视频异常评测与问答构建流程。 [原图 / PDF](<https://github.com/rohit901/VANE-Bench/raw/main/assets/Main_VANE-Bench%20Flow_v7.png?raw=true>) · [论文 / 作者页面](<https://github.com/rohit901/VANE-Bench>) · Gani, Hanan et al. / VANE-Bench*

---


<a id="year-2025-neurips"></a>

### NeurIPS

<a id="paper-a2seek"></a>

#### A2Seek: Towards Reasoning-Centric Benchmark for Aerial Anomaly Understanding

[![NeurIPS](https://img.shields.io/badge/NeurIPS-2025-2DB55D)](<https://proceedings.neurips.cc/paper_files/paper/2025/hash/de02de513503962e1d21035ab50ce661-Abstract-Datasets_and_Benchmarks_Track.html>)
[![Project](https://img.shields.io/badge/Project-Website-537A7A)](<https://2-mo.github.io/A2Seek/>)

Datasets and Benchmarks · [论文](<https://proceedings.neurips.cc/paper_files/paper/2025/hash/de02de513503962e1d21035ab50ce661-Abstract-Datasets_and_Benchmarks_Track.html>) · [阅读笔记](<literature/catalog.md#paper-a2seek>) · [引用 / BibTeX](<literature/citations.md#cite-a2seek>)

> **A2Seek / A2Seek-R1 · 图式推理与主动区域观察**
>
> 构建含事件类别、时间戳、区域框和语言解释的真实航拍异常基准，以图式推理监督、A-GRPO 与区域 seeking 机制训练 A2Seek-R1。

[![A2Seek / A2Seek-R1：航拍异常理解基准的任务、场景与标注概览。](assets/papers/a2seek.png)](assets/papers/a2seek.png)

*航拍异常理解基准的任务、场景与标注概览。 [原图 / PDF](<https://2-mo.github.io/A2Seek/static/images/carousel1.png>) · [论文 / 作者页面](<https://2-mo.github.io/A2Seek/>) · Mo, Mengjingcheng et al. / A2Seek / A2Seek-R1*

---

<a id="paper-monitor"></a>

#### MoniTor: Exploiting Large Language Models with Instruction for Online Video Anomaly Detection

[![NeurIPS](https://img.shields.io/badge/NeurIPS-2025-2DB55D)](<https://arxiv.org/abs/2510.21449>)

[论文](<https://arxiv.org/abs/2510.21449>) · [阅读笔记](<literature/catalog.md#paper-monitor>) · [引用 / BibTeX](<literature/citations.md#cite-monitor>)

> **MoniTor · 流式记忆与分数队列**
>
> 使用流式输入、历史预测记忆和分数队列，在不训练的条件下持续判断异常。

[![MoniTor 原论文框架图](assets/papers/monitor.png)](assets/papers/monitor.png)

*MoniTor：流式记忆与分数队列。原文 Figure 2。 [原图 / PDF](<https://arxiv.org/html/2510.21449v1/frameworkv5.png>) · [论文 / 作者页面](<https://arxiv.org/html/2510.21449>) · Yang, Shengtian et al. / MoniTor*

---

<a id="paper-panda"></a>

#### PANDA: Towards Generalist Video Anomaly Detection via Agentic AI Engineer

[![NeurIPS](https://img.shields.io/badge/NeurIPS-2025-2DB55D)](<https://arxiv.org/abs/2509.26386>)
[![Project](https://img.shields.io/badge/Project-Website-537A7A)](<https://github.com/showlab/PANDA>)

[论文](<https://arxiv.org/abs/2509.26386>) · [阅读笔记](<literature/catalog.md#paper-panda>) · [引用 / BibTeX](<literature/citations.md#cite-panda>)

> **PANDA · 场景规划与工具反思**
>
> 把场景规划、工具反思与经验记忆组合为通用异常检测智能体。

[![PANDA 原论文框架图](assets/papers/panda.png)](assets/papers/panda.png)

*PANDA：场景规划与工具反思。原文 Figure 2。 [原图 / PDF](<https://arxiv.org/html/2509.26386v2/PANDA_Pipeline.png>) · [论文 / 作者页面](<https://arxiv.org/html/2509.26386>) · Yang, Zhiwei et al. / PANDA*

---

<a id="paper-urf-zs-hvaa"></a>

#### A Unified Reasoning Framework for Holistic Zero-Shot Video Anomaly Analysis

[![NeurIPS](https://img.shields.io/badge/NeurIPS-2025-2DB55D)](<https://proceedings.neurips.cc/paper_files/paper/2025/hash/2aa95cf3b6aefa84d6b001928b107b4e-Abstract-Conference.html>)
[![Code](https://img.shields.io/github/stars/Rathgrith/URF-ZS-HVAA?style=social&label=Code&logo=github)](<https://github.com/Rathgrith/URF-ZS-HVAA>)

[论文](<https://proceedings.neurips.cc/paper_files/paper/2025/hash/2aa95cf3b6aefa84d6b001928b107b4e-Abstract-Conference.html>) · [阅读笔记](<literature/catalog.md#paper-urf-zs-hvaa>) · [引用 / BibTeX](<literature/citations.md#cite-urf-zs-hvaa>)

> **URF-ZS-HVAA · 任务内细化与跨任务推理链**
>
> 任务内推理用视频上下文细化时间检测，任务间链式推理再引导冻结模型完成空间定位与文本解释。

[![URF-ZS-HVAA：统一视频异常分析框架的总体流程。](assets/papers/urf-zs-hvaa.png)](assets/papers/urf-zs-hvaa.png)

*图 1：时间检测、空间定位与异常理解任务之间的链式推理框架。 [原图 / PDF](<https://github.com/Rathgrith/URF-HVAA/raw/main/assets/image.png>) · [论文 / 作者页面](<https://github.com/Rathgrith/URF-ZS-HVAA>) · Lin, Dongheng et al. / URF-ZS-HVAA*

---

<a id="paper-vad-dpo"></a>

#### Do LVLMs Truly Understand Video Anomalies? Revealing Hallucination via Co-Occurrence Patterns

[![NeurIPS](https://img.shields.io/badge/NeurIPS-2025-2DB55D)](<https://papers.nips.cc/paper_files/paper/2025/hash/99b419554537c66bf27e5eb7a74c7de4-Abstract-Conference.html>)

[论文](<https://papers.nips.cc/paper_files/paper/2025/hash/99b419554537c66bf27e5eb7a74c7de4-Abstract-Conference.html>) · [阅读笔记](<literature/catalog.md#paper-vad-dpo>) · [引用 / BibTeX](<literature/citations.md#cite-vad-dpo>)

> **VAD-DPO · 反例偏好优化抑制共现捷径**
>
> 诊断模型对物体与异常词语共现的捷径依赖，以视觉相似但语义相反的视频对进行偏好优化，增强场景语义判断。

[![VAD-DPO：图 1：视觉与文本共现偏差导致异常误判的研究动机示意。](assets/papers/vad-dpo.png)](assets/papers/vad-dpo.png)

*图 1：视觉与文本共现偏差导致异常误判的研究动机示意。 [原图 / PDF](<https://papers.nips.cc/paper_files/paper/2025/file/99b419554537c66bf27e5eb7a74c7de4-Paper-Conference.pdf>) · [论文 / 作者页面](<https://papers.nips.cc/paper_files/paper/2025/hash/99b419554537c66bf27e5eb7a74c7de4-Abstract-Conference.html>) · Zhang, Menghao et al. / VAD-DPO*

---

<a id="paper-vad-r1"></a>

#### Vad-R1: Towards Video Anomaly Reasoning via Perception-to-Cognition Chain-of-Thought

[![NeurIPS](https://img.shields.io/badge/NeurIPS-2025-2DB55D)](<https://papers.nips.cc/paper_files/paper/2025/hash/abccc325c84dedf23dbe8de3f686c733-Abstract-Conference.html>)
[![Code](https://img.shields.io/github/stars/wbfwonderful/Vad-R1?style=social&label=Code&logo=github)](<https://github.com/wbfwonderful/Vad-R1>)

[论文](<https://papers.nips.cc/paper_files/paper/2025/hash/abccc325c84dedf23dbe8de3f686c733-Abstract-Conference.html>) · [阅读笔记](<literature/catalog.md#paper-vad-r1>) · [引用 / BibTeX](<literature/citations.md#cite-vad-r1>)

> **Vad-R1 · 感知认知推理链与自验证奖励**
>
> 把视频异常推理独立为任务，构建从感知到认知的结构化推理链及 Vad-Reasoning 数据，并以带自验证的 AVA-GRPO 训练模型。

[![Vad-R1 原论文框架图](assets/papers/vad-r1.jpg)](assets/papers/vad-r1.jpg)

*Vad-R1：感知认知推理链与自验证奖励。 [原图 / PDF](<https://raw.githubusercontent.com/wbfwonderful/Vad-R1/HEAD/images/overview.png>) · [论文 / 作者页面](<https://github.com/wbfwonderful/Vad-R1>) · Huang, Chao et al. / Vad-R1*

---

<a id="paper-vadtree"></a>

#### VADTree: Explainable Training-Free Video Anomaly Detection via Hierarchical Granularity-Aware Tree

[![NeurIPS](https://img.shields.io/badge/NeurIPS-2025-2DB55D)](<https://papers.nips.cc/paper_files/paper/2025/hash/da19d18dfc5434bf419ce9c113f1865f-Abstract-Conference.html>)
[![Code](https://img.shields.io/github/stars/wenlongli10/VADTree?style=social&label=Code&logo=github)](<https://github.com/wenlongli10/VADTree>)

[论文](<https://papers.nips.cc/paper_files/paper/2025/hash/da19d18dfc5434bf419ce9c113f1865f-Abstract-Conference.html>) · [阅读笔记](<literature/catalog.md#paper-vadtree>) · [引用 / BibTeX](<literature/citations.md#cite-vadtree>)

> **VADTree · 事件边界驱动的层次粒度树**
>
> 由通用事件边界构建层次粒度树，在粗细事件节点上分别调用 VLM 感知与 LLM 推理，再融合跨粒度异常分数。

[![VADTree 原论文框架图](assets/papers/vadtree.jpg)](assets/papers/vadtree.jpg)

*VADTree：事件边界驱动的层次粒度树。 [原图 / PDF](<https://raw.githubusercontent.com/wenlongli10/VADTree/HEAD/assets/framework.png>) · [论文 / 作者页面](<https://github.com/wenlongli10/VADTree>) · Li, Wenlong et al. / VADTree*

---


<a id="year-2025-tip"></a>

### TIP

<a id="paper-crcl"></a>

#### CRCL: Causal Representation Consistency Learning for Anomaly Detection in Surveillance Videos

[![TIP](https://img.shields.io/badge/TIP-2025-537A7A)](<https://doi.org/10.1109/tip.2025.3558089>)

[论文](<https://doi.org/10.1109/tip.2025.3558089>) · [阅读笔记](<literature/catalog.md#paper-crcl>) · [引用 / BibTeX](<literature/citations.md#cite-crcl>)

> **CRCL · 场景去偏与因果正常性表征**
>
> 基于结构因果模型，将场景去偏与因果正常性学习结合，从无监督视频正常模式中学习对场景变化更稳定的表征。

[![Fig. 2: CRCL 场景去偏与因果正常性学习流程（作者预印本）](assets/papers/crcl.png)](assets/papers/crcl.png)

*Fig. 2: CRCL 场景去偏与因果正常性学习流程（作者预印本） [原图 / PDF](<https://arxiv.org/html/2503.18808v1/tifs-CReC_00.png>) · [论文 / 作者页面](<https://arxiv.org/html/2503.18808v1>) · Liu, Yang et al. / CRCL*

---


<a id="year-2025-tpami"></a>

### TPAMI

<a id="paper-mpgdfl"></a>

#### Multilingual-Prompt-Guided Directional Feature Learning for Weakly Supervised Video Anomaly Detection

[![TPAMI](https://img.shields.io/badge/TPAMI-2025-537A7A)](<https://doi.org/10.1109/tpami.2025.3590242>)

[论文](<https://doi.org/10.1109/tpami.2025.3590242>) · [阅读笔记](<literature/catalog.md#paper-mpgdfl>) · [引用 / BibTeX](<literature/citations.md#cite-mpgdfl>)

> **Multilingual VAD · 多语言提示选择与方向特征学习**
>
> 以多语言、多提示定义正常与异常，按片段自适应选择提示，并结合 Transformer–Mamba 时序建模与方向损失学习视觉特征。

*配图待补：尚未取得可核验的论文原图。[查看检索记录](<assets/papers/README.md#missing-mpgdfl>)*

---


<a id="year-2025-arxiv"></a>

### arXiv · 预印本

<a id="paper-flashback"></a>

#### Flashback: Memory-Driven Zero-shot, Real-time Video Anomaly Detection

[![arXiv](https://img.shields.io/badge/arXiv-2025-b31b1b?logo=arxiv)](<https://arxiv.org/abs/2505.15205>)

[论文](<https://arxiv.org/abs/2505.15205>) · [阅读笔记](<literature/catalog.md#paper-flashback>) · [引用 / BibTeX](<literature/citations.md#cite-flashback>)

> **Flashback · 离线语义记忆与在线匹配**
>
> 离线用语言模型构建正常与异常字幕记忆，在线将视频片段与文本记忆匹配，以检索结果给出异常判断与文本依据。

[![Flashback：图 2：离线构建语义记忆，在线检索并生成异常分数。](assets/papers/flashback.png)](assets/papers/flashback.png)

*图 2：离线构建语义记忆，在线检索并生成异常分数。 [原图 / PDF](<https://arxiv.org/html/2505.15205v2/method10.png>) · [论文 / 作者页面](<https://arxiv.org/html/2505.15205>) · Lee, Hyogun et al. / Flashback*

---

<a id="paper-slowfastvad"></a>

#### SlowFastVAD: Video Anomaly Detection via Integrating Simple Detector and RAG-Enhanced Vision-Language Model

[![arXiv](https://img.shields.io/badge/arXiv-2025-b31b1b?logo=arxiv)](<https://arxiv.org/abs/2504.10320>)

[论文](<https://arxiv.org/abs/2504.10320>) · [阅读笔记](<literature/catalog.md#paper-slowfastvad>) · [引用 / BibTeX](<literature/citations.md#cite-slowfastvad>)

> **SlowFastVAD · 快检测门控与检索增强慢推理**
>
> 快速检测器先给出异常置信度，仅将模糊片段交给检索增强 VLM，利用正常参考与推断异常模式组成知识库辅助判断。

[![SlowFastVAD：图 2：结合快速检测、检索增强的慢速推理与结果融合。](assets/papers/slowfastvad.png)](assets/papers/slowfastvad.png)

*图 2：结合快速检测、检索增强的慢速推理与结果融合。 [原图 / PDF](<https://arxiv.org/pdf/2504.10320>) · [论文 / 作者页面](<https://arxiv.org/abs/2504.10320>) · Ding, Zongcan et al. / SlowFastVAD*

---

<a id="paper-vau-r1"></a>

#### VAU-R1: Advancing Video Anomaly Understanding via Reinforcement Fine-Tuning

[![arXiv](https://img.shields.io/badge/arXiv-2025-b31b1b?logo=arxiv)](<https://arxiv.org/abs/2505.23504>)
[![Code](https://img.shields.io/github/stars/GVCLab/VAU-R1?style=social&label=Code&logo=github)](<https://github.com/GVCLab/VAU-R1>)

[论文](<https://arxiv.org/abs/2505.23504>) · [阅读笔记](<literature/catalog.md#paper-vau-r1>) · [引用 / BibTeX](<literature/citations.md#cite-vau-r1>)

> **VAU-R1 · 多任务奖励与强化微调**
>
> 通过任务专属奖励进行强化微调，并构建包含选择问答、推理依据、时间边界和描述的 VAU-Bench。

[![VAU-R1：图 2：使用 GRPO 强化微调提升视频异常推理。](assets/papers/vau-r1.png)](assets/papers/vau-r1.png)

*图 2：使用 GRPO 强化微调提升视频异常推理。 [原图 / PDF](<https://arxiv.org/html/2505.23504v1/pipeline_v2.png>) · [论文 / 作者页面](<https://arxiv.org/html/2505.23504>) · Zhu, Liyun et al. / VAU-R1*

---


<a id="year-2024"></a>

## 2024

<a id="year-2024-aaai"></a>

### AAAI

<a id="paper-vadclip"></a>

#### VadCLIP: Adapting Vision-Language Models for Weakly Supervised Video Anomaly Detection

[![AAAI](https://img.shields.io/badge/AAAI-2024-000080)](<https://arxiv.org/abs/2308.11681>)
[![Code](https://img.shields.io/github/stars/nwpu-zxr/VadCLIP?style=social&label=Code&logo=github)](<https://github.com/nwpu-zxr/VadCLIP>)

[论文](<https://arxiv.org/abs/2308.11681>) · [阅读笔记](<literature/catalog.md#paper-vadclip>) · [引用 / BibTeX](<literature/citations.md#cite-vadclip>)

> **VadCLIP · 视觉语言双分支对齐**
>
> 利用冻结 CLIP 的视觉语言关联，通过双分支完成粗粒度和细粒度异常检测。

[![VadCLIP 原论文框架图](assets/papers/vadclip.png)](assets/papers/vadclip.png)

*VadCLIP：视觉语言双分支对齐。原文 Figure 2。 [原图 / PDF](<https://arxiv.org/html/2308.11681v3/pipeline.png>) · [论文 / 作者页面](<https://arxiv.org/html/2308.11681>) · Wu, Peng et al. / VadCLIP*

---


<a id="year-2024-cvpr"></a>

### CVPR

<a id="paper-cuva"></a>

#### Uncovering What, Why and How: A Comprehensive Benchmark for Causation Understanding of Video Anomaly

[![CVPR](https://img.shields.io/badge/CVPR-2024-1E90FF)](<https://arxiv.org/abs/2405.00181>)
[![Code](https://img.shields.io/github/stars/fesvhtr/CUVA?style=social&label=Code&logo=github)](<https://github.com/fesvhtr/CUVA>)

[论文](<https://arxiv.org/abs/2405.00181>) · [阅读笔记](<literature/catalog.md#paper-cuva>) · [引用 / BibTeX](<literature/citations.md#cite-cuva>)

> **CUVA · 事件因果任务分解**
>
> 用事件、原因和后果标注，将视频异常理解拓展到因果解释及评估。

[![CUVA 原论文基准概览：原因、事件、后果和重要性曲线](assets/papers/cuva.png)](assets/papers/cuva.png)

*CUVA：事件因果任务分解。原文 Figure 2。 [原图 / PDF](<https://arxiv.org/html/2405.00181v3/dataset_4.png>) · [论文 / 作者页面](<https://arxiv.org/html/2405.00181>) · Du, Hang et al. / CUVA*

---

<a id="paper-lavad"></a>

#### Harnessing Large Language Models for Training-free Video Anomaly Detection

[![CVPR](https://img.shields.io/badge/CVPR-2024-1E90FF)](<https://arxiv.org/abs/2404.01014>)
[![Code](https://img.shields.io/github/stars/lucazanella/lavad?style=social&label=Code&logo=github)](<https://github.com/lucazanella/lavad>)
[![Project](https://img.shields.io/badge/Project-Website-537A7A)](<https://lucazanella.github.io/lavad/>)

[论文](<https://arxiv.org/abs/2404.01014>) · [阅读笔记](<literature/catalog.md#paper-lavad>) · [引用 / BibTeX](<literature/citations.md#cite-lavad>)

> **LAVAD · 字幕聚合语言评分**
>
> 先描述帧内容，再由语言模型聚合时序信息并估计异常分数。

[![LAVAD 原论文框架图](assets/papers/lavad.png)](assets/papers/lavad.png)

*LAVAD：字幕聚合语言评分。原文 Figure 4。 [原图 / PDF](<https://arxiv.org/html/2404.01014v1/architecture.png>) · [论文 / 作者页面](<https://arxiv.org/html/2404.01014>) · Zanella, Luca et al. / LAVAD*

---

<a id="paper-ovvad"></a>

#### Open-Vocabulary Video Anomaly Detection

[![CVPR](https://img.shields.io/badge/CVPR-2024-1E90FF)](<https://openaccess.thecvf.com/content/CVPR2024/html/Wu_Open-Vocabulary_Video_Anomaly_Detection_CVPR_2024_paper.html>)

[论文](<https://openaccess.thecvf.com/content/CVPR2024/html/Wu_Open-Vocabulary_Video_Anomaly_Detection_CVPR_2024_paper.html>) · [阅读笔记](<literature/catalog.md#paper-ovvad>) · [引用 / BibTeX](<literature/citations.md#cite-ovvad>)

> **OVVAD · 语言知识与异常合成**
>
> 将开放词汇检测分解为类别无关检测与类别识别，用语言知识和合成未知异常支持未见类别。

[![OVVAD：语言知识与异常合成原论文图](assets/papers/ovvad.png)](assets/papers/ovvad.png)

*OVVAD：语言知识与异常合成（原文 Figure 2）。 [原图 / PDF](<https://openaccess.thecvf.com/content/CVPR2024/papers/Wu_Open-Vocabulary_Video_Anomaly_Detection_CVPR_2024_paper.pdf>) · [论文 / 作者页面](<https://openaccess.thecvf.com/content/CVPR2024/html/Wu_Open-Vocabulary_Video_Anomaly_Detection_CVPR_2024_paper.html>) · Wu, Peng et al. / OVVAD*

---

<a id="paper-tpwng"></a>

#### Text Prompt with Normality Guidance for Weakly Supervised Video Anomaly Detection

[![CVPR](https://img.shields.io/badge/CVPR-2024-1E90FF)](<https://openaccess.thecvf.com/content/CVPR2024/html/Yang_Text_Prompt_with_Normality_Guidance_for_Weakly_Supervised_Video_Anomaly_CVPR_2024_paper.html>)

[论文](<https://openaccess.thecvf.com/content/CVPR2024/html/Yang_Text_Prompt_with_Normality_Guidance_for_Weakly_Supervised_Video_Anomaly_CVPR_2024_paper.html>) · [阅读笔记](<literature/catalog.md#paper-tpwng>) · [引用 / BibTeX](<literature/citations.md#cite-tpwng>)

> **TPWNG · 正常引导伪标签学习**
>
> 将事件描述与视频帧对齐，结合正常性视觉提示生成帧级伪标签，再进行时序自训练。

[![TPWNG：正常引导伪标签学习原论文图](assets/papers/tpwng.png)](assets/papers/tpwng.png)

*TPWNG：正常引导伪标签学习（原文 Figure 2）。 [原图 / PDF](<https://openaccess.thecvf.com/content/CVPR2024/papers/Yang_Text_Prompt_with_Normality_Guidance_for_Weakly_Supervised_Video_Anomaly_CVPR_2024_paper.pdf>) · [论文 / 作者页面](<https://openaccess.thecvf.com/content/CVPR2024/html/Yang_Text_Prompt_with_Normality_Guidance_for_Weakly_Supervised_Video_Anomaly_CVPR_2024_paper.html>) · Yang, Zhiwei et al. / TPWNG*

---

<a id="paper-uca-paper"></a>

#### Towards Surveillance Video-and-Language Understanding: New Dataset, Baselines, and Challenges

[![CVPR](https://img.shields.io/badge/CVPR-2024-1E90FF)](<https://openaccess.thecvf.com/content/CVPR2024/html/Yuan_Towards_Surveillance_Video-and-Language_Understanding_New_Dataset_Baselines_and_Challenges_CVPR_2024_paper.html>)
[![Project](https://img.shields.io/badge/Project-Website-537A7A)](<https://xuange923.github.io/Surveillance-Video-Understanding>)

[论文](<https://openaccess.thecvf.com/content/CVPR2024/html/Yuan_Towards_Surveillance_Video-and-Language_Understanding_New_Dataset_Baselines_and_Challenges_CVPR_2024_paper.html>) · [阅读笔记](<literature/catalog.md#paper-uca-paper>) · [引用 / BibTeX](<literature/citations.md#cite-uca-paper>)

> **UCA · 事件句子与时间对齐**
>
> 为 UCF-Crime 添加句子级事件描述与时间标注，建立监控视频定位、描述等语言理解任务基线。

[![UCA：用于多模态异常检测的增强 TEVAD 基线框架原论文图](assets/papers/uca-paper.png)](assets/papers/uca-paper.png)

*用于多模态异常检测的增强 TEVAD 基线框架（原文 Figure 3）。 [原图 / PDF](<https://openaccess.thecvf.com/content/CVPR2024/papers/Yuan_Towards_Surveillance_Video-and-Language_Understanding_New_Dataset_Baselines_and_Challenges_CVPR_2024_paper.pdf>) · [论文 / 作者页面](<https://openaccess.thecvf.com/content/CVPR2024/html/Yuan_Towards_Surveillance_Video-and-Language_Understanding_New_Dataset_Baselines_and_Challenges_CVPR_2024_paper.html>) · Yuan, Tongtong et al. / UCA*

---


<a id="year-2024-eccv"></a>

### ECCV

<a id="paper-anomalyruler"></a>

#### Follow the Rules: Reasoning for Video Anomaly Detection with Large Language Models

[![ECCV](https://img.shields.io/badge/ECCV-2024-0B84FE)](<https://arxiv.org/abs/2407.10299>)
[![Code](https://img.shields.io/github/stars/Yuchen413/AnomalyRuler?style=social&label=Code&logo=github)](<https://github.com/Yuchen413/AnomalyRuler>)

[论文](<https://arxiv.org/abs/2407.10299>) · [阅读笔记](<literature/catalog.md#paper-anomalyruler>) · [引用 / BibTeX](<literature/citations.md#cite-anomalyruler>)

> **AnomalyRuler · 正常规则归纳演绎**
>
> 从少量正常样本归纳场景规则，再依据规则判断测试视频中的异常。

[![AnomalyRuler 原论文框架图](assets/papers/anomalyruler.png)](assets/papers/anomalyruler.png)

*AnomalyRuler：正常规则归纳演绎。原文 Figure 2。 [原图 / PDF](<https://arxiv.org/html/2407.10299v2/VAD_newpipe.png>) · [论文 / 作者页面](<https://arxiv.org/html/2407.10299>) · Yang, Yuchen et al. / AnomalyRuler*

---


<a id="year-2024-neurips"></a>

### NeurIPS

<a id="paper-hawk"></a>

#### Hawk: Learning to Understand Open-World Video Anomalies

[![NeurIPS](https://img.shields.io/badge/NeurIPS-2024-2DB55D)](<https://arxiv.org/abs/2405.16886>)
[![Code](https://img.shields.io/github/stars/jqtangust/hawk?style=social&label=Code&logo=github)](<https://github.com/jqtangust/hawk>)

[论文](<https://arxiv.org/abs/2405.16886>) · [阅读笔记](<literature/catalog.md#paper-hawk>) · [引用 / BibTeX](<literature/citations.md#cite-hawk>)

> **HAWK · 运动语言监督对齐**
>
> 显式引入运动信息，并用异常视频描述与问答数据训练开放场景理解能力。

[![HAWK 原论文框架图](assets/papers/hawk.png)](assets/papers/hawk.png)

*HAWK：运动语言监督对齐。原文 Figure 3。 [原图 / PDF](<https://arxiv.org/html/2405.16886v1/framework.png>) · [论文 / 作者页面](<https://arxiv.org/html/2405.16886>) · Tang, Jiaqi et al. / HAWK*

---


<a id="year-2024-tip"></a>

### TIP

<a id="paper-alan"></a>

#### Toward Video Anomaly Retrieval From Video Anomaly Detection: New Benchmarks and Model

[![TIP](https://img.shields.io/badge/TIP-2024-537A7A)](<https://doi.org/10.1109/tip.2024.3374070>)
[![Project](https://img.shields.io/badge/Project-Website-537A7A)](<https://github.com/Roc-Ng/VAR>)

[论文](<https://doi.org/10.1109/tip.2024.3374070>) · [阅读笔记](<literature/catalog.md#paper-alan>) · [引用 / BibTeX](<literature/citations.md#cite-alan>)

> **ALAN / VAR · 异常引导采样与跨模态检索**
>
> 提出长视频异常检索任务及 UCFCrime-AR、XDViolence-AR，使用异常引导采样、视频提示掩码短语建模和跨模态对齐检索相关视频。

[![Fig. 4: ALAN 的视频、文本、音频编码与跨模态对齐框架（作者预印本）](assets/papers/alan.png)](assets/papers/alan.png)

*Fig. 4: ALAN 的视频、文本、音频编码与跨模态对齐框架（作者预印本） [原图 / PDF](<https://arxiv.org/html/2307.12545v2/pipeline.png>) · [论文 / 作者页面](<https://arxiv.org/html/2307.12545v2>) · Wu, Peng et al. / ALAN / VAR*

---

<a id="paper-pel"></a>

#### Learning Prompt-Enhanced Context Features for Weakly-Supervised Video Anomaly Detection

[![TIP](https://img.shields.io/badge/TIP-2024-537A7A)](<https://doi.org/10.1109/tip.2024.3451935>)
[![Code](https://img.shields.io/github/stars/yujiangpu20/PEL4VAD?style=social&label=Code&logo=github)](<https://github.com/yujiangpu20/PEL4VAD>)

[论文](<https://doi.org/10.1109/tip.2024.3451935>) · [阅读笔记](<literature/catalog.md#paper-pel>) · [引用 / BibTeX](<literature/citations.md#cite-pel>)

> **PEL · 提示增强语义与时序上下文聚合**
>
> 通过共享注意力矩阵和自适应融合聚合局部与全局时序上下文，再以知识提示增强视觉特征的异常子类区分能力。

[![Fig. 2：PEL 时序上下文聚合与提示增强学习框架（作者预印本，第 4 页）。](assets/papers/pel.png)](assets/papers/pel.png)

*Fig. 2：PEL 时序上下文聚合与提示增强学习框架（作者预印本，第 4 页）。 [原图 / PDF](<https://arxiv.org/pdf/2306.14451v2>) · [论文 / 作者页面](<https://arxiv.org/abs/2306.14451v2>) · Pu, Yujiang et al. / PEL*

---


<a id="year-2023"></a>

## 2023

<a id="year-2023-cvpr"></a>

### CVPR

<a id="paper-eval"></a>

#### EVAL: Explainable Video Anomaly Localization

[![CVPR](https://img.shields.io/badge/CVPR-2023-1E90FF)](<https://openaccess.thecvf.com/content/CVPR2023/html/Singh_EVAL_Explainable_Video_Anomaly_Localization_CVPR_2023_paper.html>)

[论文](<https://openaccess.thecvf.com/content/CVPR2023/html/Singh_EVAL_Explainable_Video_Anomaly_Localization_CVPR_2023_paper.html>) · [阅读笔记](<literature/catalog.md#paper-eval>) · [引用 / BibTeX](<literature/citations.md#cite-eval>)

> **EVAL · 对象运动属性解释**
>
> 以对象类别、运动方向和速度等可读属性建立位置相关的正常模式，解释局部异常。

[![EVAL：对象运动属性解释原论文图](assets/papers/eval.png)](assets/papers/eval.png)

*EVAL：对象运动属性解释（原文 Figure 1）。 [原图 / PDF](<https://openaccess.thecvf.com/content/CVPR2023/papers/Singh_EVAL_Explainable_Video_Anomaly_Localization_CVPR_2023_paper.pdf>) · [论文 / 作者页面](<https://openaccess.thecvf.com/content/CVPR2023/html/Singh_EVAL_Explainable_Video_Anomaly_Localization_CVPR_2023_paper.html>) · Singh, Ashish et al. / EVAL*

---
