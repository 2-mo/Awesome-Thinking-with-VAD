# 视频异常理解 · 方法比较

[论文年表](../llm4vad.md) · [数据集与评测](benchmarks.md) · [阅读路线](reading-guide.md)

[引用导出](citations.md) · [更新记录](../CHANGELOG.md)

按方法方向比较输出、训练与适配、运行设置和验证方式。每个已填写单元格链接到证据；待核验表示尚未完成该维度核对，不代表论文没有该能力。基准论文记录其评测对象。

冻结主模型不等于整个流程无需训练；提示搜索、轻量模块训练和权重微调分别记录。在线／流式标签不能单独证明不访问未来帧。不同数据划分、输入模态与评测协议下的数值不作统一排名。

## 语义对齐与融合

| 论文 | 输出／评测对象 | 训练与适配 | 运行设置 | 未来帧访问 | 验证方式 |
| --- | --- | --- | --- | --- | --- |
| [VadCLIP](<catalog.md#paper-vadclip>) | [粗细粒度异常检测](<https://arxiv.org/abs/2308.11681>) | 待核验 | 待核验 | 待核验 | 待核验 |
| [Ex-VAD](<catalog.md#paper-ex-vad>) | [异常解释；细粒度检测](<https://proceedings.mlr.press/v267/huang25ad.html>) | 待核验 | 待核验 | 待核验 | 待核验 |
| [HAWK](<catalog.md#paper-hawk>) | [异常描述；问答](<https://arxiv.org/abs/2405.16886>) | [异常描述与问答监督训练](<https://arxiv.org/abs/2405.16886>) | 待核验 | 待核验 | 待核验 |
| [OVVAD](<catalog.md#paper-ovvad>) | [异常检测；开放词汇类别](<https://openaccess.thecvf.com/content/CVPR2024/html/Wu_Open-Vocabulary_Video_Anomaly_Detection_CVPR_2024_paper.html>) | 待核验 | 待核验 | 待核验 | 待核验 |
| [TPWNG](<catalog.md#paper-tpwng>) | [帧级异常检测](<https://openaccess.thecvf.com/content/CVPR2024/html/Yang_Text_Prompt_with_Normality_Guidance_for_Weakly_Supervised_Video_Anomaly_CVPR_2024_paper.html>) | 待核验 | 待核验 | 待核验 | 待核验 |
| [Anomize](<catalog.md#paper-anomize>) | [异常检测；未见类别识别](<https://openaccess.thecvf.com/content/CVPR2025/html/Li_Anomize_Better_Open_Vocabulary_Video_Anomaly_Detection_CVPR_2025_paper.html>) | 待核验 | 待核验 | 待核验 | 待核验 |
| [Alert-CLIP](<catalog.md#paper-alert-clip>) | [异常检测](<https://openaccess.thecvf.com/content/CVPR2026/html/Zhu_Alert-CLIP_Abnormality-aware_Latent-Enhanced_Representation_Tuning_of_CLIP_for_Video_Anomaly_CVPR_2026_paper.html>) | 待核验 | 待核验 | 待核验 | 待核验 |
| [LAVIDA](<catalog.md#paper-lavida>) | [帧级／像素级异常检测](<https://openaccess.thecvf.com/content/CVPR2026/html/Dai_No_Need_For_Real_Anomaly_MLLM_Empowered_Zero-Shot_Video_Anomaly_CVPR_2026_paper.html>) | [伪异常训练；真实异常零样本](<https://openaccess.thecvf.com/content/CVPR2026/html/Dai_No_Need_For_Real_Anomaly_MLLM_Empowered_Zero-Shot_Video_Anomaly_CVPR_2026_paper.html>) | 待核验 | 待核验 | 待核验 |
| [AnomalyCraft-700K](<catalog.md#paper-anomalycraft>) | [合成异常视频；组件级语义标注](<https://arxiv.org/abs/2609.06978>) | 待核验 | 待核验 | 待核验 | 待核验 |
| [TD-VAD](<catalog.md#paper-td-vad>) | [异常检测](<https://arxiv.org/abs/2608.11820>) | [文本监督训练检测器；冻结 CLIP](<https://arxiv.org/abs/2608.11820>) | 待核验 | 待核验 | 待核验 |
| [HeadHunt-VAD](<catalog.md#paper-headhunt-vad>) | [异常评分；时序定位](<https://arxiv.org/abs/2512.17601>) | [冻结 MLLM；轻量评分器校准](<https://arxiv.org/abs/2512.17601>) | 待核验 | 待核验 | 待核验 |
| [SteerVAD](<catalog.md#paper-steervad>) | [异常评分；事后解释](<https://arxiv.org/abs/2602.24021>) | [冻结 MLLM；控制器／评分器训练](<https://arxiv.org/abs/2602.24021>) | 待核验 | 待核验 | 待核验 |

## 语言判据与提示优化

| 论文 | 输出／评测对象 | 训练与适配 | 运行设置 | 未来帧访问 | 验证方式 |
| --- | --- | --- | --- | --- | --- |
| [LAVAD](<catalog.md#paper-lavad>) | [异常分数](<https://arxiv.org/abs/2404.01014>) | [无需目标任务训练](<https://arxiv.org/abs/2404.01014>) | 待核验 | 待核验 | 待核验 |
| [AnomalyRuler](<catalog.md#paper-anomalyruler>) | [异常判断](<https://arxiv.org/abs/2407.10299>) | 待核验 | 待核验 | 待核验 | 待核验 |
| [VERA](<catalog.md#paper-vera>) | [异常判断；语言解释](<https://arxiv.org/abs/2412.01095>) | [弱标签驱动的语言提示优化；冻结 VLM](<https://github.com/vera-framework/VERA>) | 待核验 | 待核验 | [UCF-Crime：AUC；XD-Violence：AUC／AP](<https://github.com/vera-framework/VERA>) |
| [EVAL](<catalog.md#paper-eval>) | [局部异常；对象运动属性解释](<https://openaccess.thecvf.com/content/CVPR2023/html/Singh_EVAL_Explainable_Video_Anomaly_Localization_CVPR_2023_paper.html>) | 待核验 | 待核验 | 待核验 | 待核验 |
| [LaGoVAD](<catalog.md#paper-lagovad>) | [语言条件化异常检测](<https://proceedings.iclr.cc/paper_files/paper/2026/hash/f88bec15cc4cb56b432ee040bb63f94f-Abstract-Conference.html>) | [合成视频与对比学习](<https://proceedings.iclr.cc/paper_files/paper/2026/hash/f88bec15cc4cb56b432ee040bb63f94f-Abstract-Conference.html>) | 待核验 | 待核验 | 待核验 |
| [LRPO](<catalog.md#paper-lrpo>) | [异常推理](<https://arxiv.org/abs/2607.00654>) | [语言经验优化；不更新模型参数](<https://arxiv.org/abs/2607.00654>) | 待核验 | 待核验 | 待核验 |
| [Probe-VAD](<catalog.md#paper-probe-vad>) | [连续异常分数](<https://arxiv.org/abs/2609.17211>) | [无需目标任务训练](<https://arxiv.org/abs/2609.17211>) | 待核验 | 待核验 | 待核验 |

## 时序分层与记忆

| 论文 | 输出／评测对象 | 训练与适配 | 运行设置 | 未来帧访问 | 验证方式 |
| --- | --- | --- | --- | --- | --- |
| [Holmes-VAU](<catalog.md#paper-holmes-vau>) | [片段／事件／视频级理解](<https://arxiv.org/abs/2412.06171>) | 待核验 | 待核验 | 待核验 | 待核验 |
| [EventVAD](<catalog.md#paper-eventvad>) | [事件边界；事件级异常推理](<https://arxiv.org/abs/2504.13092>) | [无需目标任务训练](<https://arxiv.org/abs/2504.13092>) | 待核验 | 待核验 | 待核验 |
| [MoniTor](<catalog.md#paper-monitor>) | [在线异常分数](<https://arxiv.org/abs/2510.21449>) | [无需目标任务训练](<https://arxiv.org/abs/2510.21449>) | [流式／在线](<https://arxiv.org/abs/2510.21449>) | 待核验 | 待核验 |
| [VALU](<catalog.md#paper-valu>) | [时序边界；异常细节（评测）](<https://aclanthology.org/2026.acl-long.56/>) | 待核验 | 待核验 | 待核验 | [时间 grounding；异常定位；细节辨别](<https://aclanthology.org/2026.acl-long.56/>) |
| [UCA](<catalog.md#paper-uca-paper>) | [事件描述；句子时间边界（标注与评测）](<https://openaccess.thecvf.com/content/CVPR2024/html/Yuan_Towards_Surveillance_Video-and-Language_Understanding_New_Dataset_Baselines_and_Challenges_CVPR_2024_paper.html>) | 待核验 | 待核验 | 待核验 | 待核验 |
| [VA-GPT](<catalog.md#paper-va-gpt>) | [异常总结；时间定位](<https://openaccess.thecvf.com/content/ICCV2025/html/Chen_Aligning_Effective_Tokens_with_Video_Anomaly_in_Large_Language_Models_ICCV_2025_paper.html>) | 待核验 | 待核验 | 待核验 | 待核验 |
| [VADTree](<catalog.md#paper-vadtree>) | [跨粒度异常分数](<https://papers.nips.cc/paper_files/paper/2025/hash/da19d18dfc5434bf419ce9c113f1865f-Abstract-Conference.html>) | [无需目标任务训练](<https://papers.nips.cc/paper_files/paper/2025/hash/da19d18dfc5434bf419ce9c113f1865f-Abstract-Conference.html>) | 待核验 | 待核验 | 待核验 |
| [Flashback](<catalog.md#paper-flashback>) | [异常判断；检索文本依据](<https://arxiv.org/abs/2505.15205>) | 待核验 | [离线构建记忆；在线匹配](<https://arxiv.org/abs/2505.15205>) | 待核验 | 待核验 |
| [ReactVAU](<catalog.md#paper-reactvau>) | [异常检测；语义验证；原因描述](<https://arxiv.org/abs/2609.07941>) | 待核验 | [连续流式检测；可疑事件触发慢速推理](<https://arxiv.org/abs/2609.07941>) | 待核验 | 待核验 |

## 主动观察与工具决策

| 论文 | 输出／评测对象 | 训练与适配 | 运行设置 | 未来帧访问 | 验证方式 |
| --- | --- | --- | --- | --- | --- |
| [PANDA](<catalog.md#paper-panda>) | [异常检测决策](<https://arxiv.org/abs/2509.26386>) | 待核验 | 待核验 | 待核验 | 待核验 |
| [Anom-π](<catalog.md#paper-anom-pi>) | [异常推理；观察策略](<https://arxiv.org/abs/2607.00622>) | 待核验 | 待核验 | 待核验 | 待核验 |
| [MemoVAD](<catalog.md#paper-memovad>) | [异常检测决策](<https://www.ijcai.org/proceedings/2026/618>) | 待核验 | [边云协同；选择性云端调用](<https://www.ijcai.org/proceedings/2026/618>) | 待核验 | 待核验 |
| [A2Seek / A2Seek-R1](<catalog.md#paper-a2seek>) | [异常判断；区域定位；语言解释](<https://proceedings.neurips.cc/paper_files/paper/2025/hash/de02de513503962e1d21035ab50ce661-Abstract-Datasets_and_Benchmarks_Track.html>) | [推理监督；强化学习（A-GRPO）](<https://proceedings.neurips.cc/paper_files/paper/2025/hash/de02de513503962e1d21035ab50ce661-Abstract-Datasets_and_Benchmarks_Track.html>) | 待核验 | 待核验 | 待核验 |
| [AgenticVAU](<catalog.md#paper-agenticvau>) | [异常定位；证据推理](<https://arxiv.org/abs/2608.03779>) | 待核验 | 待核验 | 待核验 | 待核验 |
| [VAGU & GtS](<catalog.md#paper-vagu-gts>) | [异常时间段；解释；问答](<https://ojs.aaai.org/index.php/AAAI/article/view/42412>) | [无需目标任务训练（AAAI 原版）](<https://arxiv.org/abs/2507.21507>) | 待核验 | 待核验 | [JeAUG 联合定位与理解；选择问答](<https://ojs.aaai.org/index.php/AAAI/article/view/42412>) |
| [SlowFastVAD](<catalog.md#paper-slowfastvad>) | [异常判断](<https://arxiv.org/abs/2504.10320>) | 待核验 | [快速检测门控；检索增强慢推理](<https://arxiv.org/abs/2504.10320>) | 待核验 | 待核验 |

## 结构化推理与验证

| 论文 | 输出／评测对象 | 训练与适配 | 运行设置 | 未来帧访问 | 验证方式 |
| --- | --- | --- | --- | --- | --- |
| [CUVA](<catalog.md#paper-cuva>) | [事件描述；原因与后果解释（评测）](<https://arxiv.org/abs/2405.00181>) | 待核验 | 待核验 | 待核验 | [描述／原因／后果分别评估；MMEval](<https://arxiv.org/abs/2405.00181>) |
| [SRVAU-R1](<catalog.md#paper-srvau-r1>) | [初始推理；反思与修正](<https://arxiv.org/abs/2602.01004>) | [监督微调；强化微调](<https://arxiv.org/abs/2602.01004>) | 待核验 | 待核验 | 待核验 |
| [FineVAU](<catalog.md#paper-finevau>) | [事件／实体／位置事实（评测）](<https://arxiv.org/abs/2601.17258>) | 待核验 | 待核验 | 待核验 | [关键视觉事实；FVScore；人类一致性](<https://arxiv.org/abs/2601.17258>) |
| [LAS-VAD](<catalog.md#paper-las-vad>) | [异常检测](<https://openaccess.thecvf.com/content/CVPR2026/html/Wang_Weakly_Supervised_Video_Anomaly_Detection_with_Anomaly-Connected_Components_and_Intention_CVPR_2026_paper.html>) | 待核验 | 待核验 | 待核验 | 待核验 |
| [Vad-R1](<catalog.md#paper-vad-r1>) | [结构化推理；异常判断](<https://papers.nips.cc/paper_files/paper/2025/hash/abccc325c84dedf23dbe8de3f686c733-Abstract-Conference.html>) | [强化学习（AVA-GRPO）](<https://papers.nips.cc/paper_files/paper/2025/hash/abccc325c84dedf23dbe8de3f686c733-Abstract-Conference.html>) | 待核验 | 待核验 | 待核验 |
| [CueBench / Cue-R1](<catalog.md#paper-cuebench>) | [识别／定位／检测／预判（评测）](<https://ojs.aaai.org/index.php/AAAI/article/view/38209>) | [强化微调（Cue-R1）](<https://ojs.aaai.org/index.php/AAAI/article/view/38209>) | 待核验 | 待核验 | [识别／时序定位／检测／预判分任务评估](<https://ojs.aaai.org/index.php/AAAI/article/view/38209>) |
| [TargetVAU](<catalog.md#paper-targetvau>) | [异常个体；行为解释](<https://ojs.aaai.org/index.php/AAAI/article/view/38378>) | [指令微调](<https://ojs.aaai.org/index.php/AAAI/article/view/38378>) | 待核验 | 待核验 | 待核验 |
| [HoloTrace](<catalog.md#paper-holotrace>) | [事件推理；边界判断](<https://doi.org/10.1145/3746027.3755185>) | 待核验 | [边云协同](<https://doi.org/10.1145/3746027.3755185>) | 待核验 | 待核验 |
| [TAU-Bench](<catalog.md#paper-tau-bench>) | [实例轨迹／掩码；分层描述（评测）](<https://arxiv.org/abs/2608.05699>) | 待核验 | 待核验 | 待核验 | [实例跟踪与分层语义联合评测](<https://arxiv.org/abs/2608.05699>) |
| [Vad-R1-Plus](<catalog.md#paper-vad-r1-plus>) | [多阶段推理；风险判断](<https://arxiv.org/abs/2601.10165>) | [强化学习](<https://arxiv.org/abs/2601.10165>) | 待核验 | 待核验 | 待核验 |
| [VAU-R1](<catalog.md#paper-vau-r1>) | [选择问答；推理依据；时间边界；描述](<https://arxiv.org/abs/2505.23504>) | [强化微调](<https://arxiv.org/abs/2505.23504>) | 待核验 | 待核验 | 待核验 |
| [CG-CoE](<catalog.md#paper-cg-coe>) | [异常语义匹配评分（评价指标）](<https://icml.cc/virtual/2026/poster/66013>) | 待核验 | 待核验 | 待核验 | [AEA 指标有效性；CVP 措辞扰动鲁棒性](<https://icml.cc/virtual/2026/poster/66013>) |
| [URF-ZS-HVAA](<catalog.md#paper-urf-zs-hvaa>) | [时间检测；空间定位；文本解释](<https://proceedings.neurips.cc/paper_files/paper/2025/hash/2aa95cf3b6aefa84d6b001928b107b4e-Abstract-Conference.html>) | 待核验 | 待核验 | 待核验 | 待核验 |
| [VAD-DPO](<catalog.md#paper-vad-dpo>) | [异常判断；共现偏见诊断](<https://papers.nips.cc/paper_files/paper/2025/hash/99b419554537c66bf27e5eb7a74c7de4-Abstract-Conference.html>) | [偏好优化（DPO）](<https://papers.nips.cc/paper_files/paper/2025/hash/99b419554537c66bf27e5eb7a74c7de4-Abstract-Conference.html>) | 待核验 | 待核验 | [共现模式与语义反例诊断](<https://papers.nips.cc/paper_files/paper/2025/hash/99b419554537c66bf27e5eb7a74c7de4-Abstract-Conference.html>) |
| [CLUE-VAD](<catalog.md#paper-clue-vad>) | [异常分数；线索归因解释](<https://eccv.ecva.net/virtual/2026/poster/4744>) | 待核验 | 待核验 | 待核验 | 待核验 |
| [O-VAD](<catalog.md#paper-o-vad>) | [异常对象与帧；过程和类型报告](<https://arxiv.org/abs/2607.18142>) | 待核验 | 待核验 | 待核验 | 待核验 |
| [STCH](<catalog.md#paper-stch>) | [犯罪事件预判](<https://openaccess.thecvf.com/content/CVPR2026/html/Wang_Streaming_Video_Crime_Anticipation_with_Spatio-Temporal_Causal_Reasoning_CVPR_2026_paper.html>) | 待核验 | [流式预判](<https://openaccess.thecvf.com/content/CVPR2026/html/Wang_Streaming_Video_Crime_Anticipation_with_Spatio-Temporal_Causal_Reasoning_CVPR_2026_paper.html>) | 待核验 | 待核验 |
| [Pistachio](<catalog.md#paper-pistachio>) | [评测检测与异常描述；多事件理解](<https://arxiv.org/abs/2511.19474>) | 待核验 | 待核验 | 待核验 | 待核验 |
| [VANE-Bench](<catalog.md#paper-vane-bench>) | [评测异常检测与定位问答](<https://aclanthology.org/2025.findings-naacl.171/>) | 待核验 | 待核验 | 待核验 | 待核验 |
