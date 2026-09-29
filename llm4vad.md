# 视频异常理解 · 论文年表

[研究地图](https://2-mo.github.io/Awesome-Thinking-with-VAD/) · [按创新思路阅读](catalog.md) · [按会议查找](venues/README.md)

更新：2026-09-30 · 51 篇论文 · 5 个方法方向。

聚焦视频异常解释、推理、时序定位与理解评估，以及直接支撑这些目标的语义表征方法。会议与年份采用已核验的正式发表信息；未确认录用的论文保留 arXiv。

> 自动生成：编辑 `data/catalog.json` 后运行 `npm run generate`。完整摘要、阅读关注与核验来源见 [catalog.md](catalog.md)。

[2026](#year-2026) · [2025](#year-2025) · [2024](#year-2024) · [2023](#year-2023)

<a id="year-2026"></a>

## 2026

<a id="year-2026-aaai"></a>

### AAAI

| 论文 | 创新抓手 | 资源 |
| --- | --- | --- |
| **CueBench / Cue-R1**<br>[CueBench: Advancing Unified Understanding of Context-Aware Video Anomalies in Real-World](<https://ojs.aaai.org/index.php/AAAI/article/view/38209>) | 上下文异常分类体系与分层奖励 | [代码](<https://github.com/Mia-YatingYu/Cue-R1>) |
| **FineVAU**<br>[FineVAU: A Novel Human-Aligned Benchmark for Fine-Grained Video Anomaly Understanding](<https://arxiv.org/abs/2601.17258>) | 关键视觉事实评估 | [项目](<https://finevau.github.io/>) |
| **HeadHunt-VAD**<br>[HeadHunt-VAD: Hunting Robust Anomaly-Sensitive Heads in MLLM for Tuning-Free Video Anomaly Detection](<https://arxiv.org/abs/2512.17601>) | 稳定异常敏感注意力头探测 | [代码](<https://github.com/CebCai/HeadHunt-VAD>) |
| **TargetVAU**<br>[TargetVAU: Multimodal Anomaly-Aware Reasoning for Target Behavior Understanding in Videos](<https://ojs.aaai.org/index.php/AAAI/article/view/38378>) | 个体时空交互图与指令推理 | — |
| **VAGU & GtS**<br>[VAGU & GtS: LLM-Based Benchmark and Framework for Joint Video Anomaly Grounding and Understanding](<https://ojs.aaai.org/index.php/AAAI/article/view/42412>) | 先全局粗定位再局部细查 | [核验](<https://arxiv.org/abs/2507.21507>) |

<a id="year-2026-acl"></a>

### ACL

| 论文 | 创新抓手 | 资源 |
| --- | --- | --- |
| **VALU**<br>[VALU: A Benchmark for Video Anomaly Temporal Localization and Understanding at Multiple Semantic Levels](<https://aclanthology.org/2026.acl-long.56/>) | 语义分层边界评估 | — |

<a id="year-2026-cvpr"></a>

### CVPR

| 论文 | 创新抓手 | 资源 |
| --- | --- | --- |
| **Alert-CLIP**<br>[Alert-CLIP: Abnormality-aware Latent-Enhanced Representation Tuning of CLIP for Video Anomaly Detection](<https://openaccess.thecvf.com/content/CVPR2026/html/Zhu_Alert-CLIP_Abnormality-aware_Latent-Enhanced_Representation_Tuning_of_CLIP_for_Video_Anomaly_CVPR_2026_paper.html>) | 区域语义多层对齐 | — |
| **LAS-VAD**<br>[Weakly Supervised Video Anomaly Detection with Anomaly-Connected Components and Intention Reasoning](<https://openaccess.thecvf.com/content/CVPR2026/html/Wang_Weakly_Supervised_Video_Anomaly_Detection_with_Anomaly-Connected_Components_and_Intention_CVPR_2026_paper.html>) | 语义连通与意图感知 | — |
| **LAVIDA**<br>[No Need For Real Anomaly: MLLM Empowered Zero-Shot Video Anomaly Detection](<https://openaccess.thecvf.com/content/CVPR2026/html/Dai_No_Need_For_Real_Anomaly_MLLM_Empowered_Zero-Shot_Video_Anomaly_CVPR_2026_paper.html>) | 伪异常与反向注意力 | [项目](<https://github.com/VitaminCreed/LAVIDA>) · [核验](<https://openaccess.thecvf.com/content/CVPR2026/papers/Dai_No_Need_For_Real_Anomaly_MLLM_Empowered_Zero-Shot_Video_Anomaly_CVPR_2026_paper.pdf>) |
| **STCH**<br>[Streaming Video Crime Anticipation with Spatio-Temporal Causal Reasoning](<https://openaccess.thecvf.com/content/CVPR2026/html/Wang_Streaming_Video_Crime_Anticipation_with_Spatio-Temporal_Causal_Reasoning_CVPR_2026_paper.html>) | 流式时空因果超图 | [核验](<https://cvpr.thecvf.com/virtual/2026/poster/39800>) |

<a id="year-2026-eccv"></a>

### ECCV

| 论文 | 创新抓手 | 资源 |
| --- | --- | --- |
| **CLUE-VAD**<br>[CLUE-VAD: Structured Semantic Clues for Understanding Explainable Events in Video Anomaly Detection](<https://eccv.ecva.net/virtual/2026/poster/4744>) | 结构化语义线索与类别感知归因 | [核验](<https://media.eventhosts.cc/Conferences/ECCV2026/pdfs/7292.pdf>) |
| **O-VAD**<br>[O-VAD: Industrial Video Anomaly Detection through Object-Centric Tracking and Reasoning](<https://arxiv.org/abs/2607.18142>) | 对象状态轨迹与时序推理 | [代码](<https://github.com/o-vad/O-VAD>) · [核验](<https://eccv.ecva.net/virtual/2026/poster/4659>) |

<a id="year-2026-iclr"></a>

### ICLR

| 论文 | 创新抓手 | 资源 |
| --- | --- | --- |
| **LaGoVAD**<br>[Language-guided Open-world Video Anomaly Detection under Weak Supervision](<https://proceedings.iclr.cc/paper_files/paper/2026/hash/f88bec15cc4cb56b432ee040bb63f94f-Abstract-Conference.html>) | 自然语言条件化异常定义 | [代码](<https://github.com/Kamino666/LaGoVAD-PreVAD>) |
| **SteerVAD**<br>[Steering and Rectifying Latent Representation Manifolds in Frozen Multi-modal LLMs for Video Anomaly Detection](<https://arxiv.org/abs/2602.24021>) | 潜在异常专家头与上下文表示校正 | [核验](<https://iclr.cc/virtual/2026/papers.html>) |

<a id="year-2026-icml"></a>

### ICML

| 论文 | 创新抓手 | 资源 |
| --- | --- | --- |
| **Anom-π**<br>[Learning to Watch: Active Video Anomaly Understanding via Interleaved Policy Optimization](<https://arxiv.org/abs/2607.00622>) | 交替推理与观察策略 | [核验](<https://icml.cc/Downloads/2026>) |
| **CG-CoE**<br>[Towards Trustworthy Video Anomaly Understanding: A Class-Guided Chain-of-Evaluation Metric and An Anomaly-focused Meta-Benchmark](<https://icml.cc/virtual/2026/poster/66013>) | 类别引导评价链与元评测 | [核验](<https://openreview.net/forum?id=7waVdY1WmW>) |
| **LRPO**<br>[Linguistic Relative Policy Optimization for Video Anomaly Reasoning](<https://arxiv.org/abs/2607.00654>) | 组相对语言经验优化 | [核验](<https://icml.cc/virtual/2026/poster/64285>) |
| **TD-VAD**<br>[TD-VAD: Breaking Visual Dependence in Video Anomaly Detection with Text-Driven Learning](<https://arxiv.org/abs/2608.11820>) | 文本时序监督与事件演化注意力 | [核验](<https://icml.cc/virtual/2026/poster/65928>) |

<a id="year-2026-ijcai"></a>

### IJCAI

| 论文 | 创新抓手 | 资源 |
| --- | --- | --- |
| **MemoVAD**<br>[MemoVAD: Resource-Efficient Video Anomaly Detection via Dynamic Semantic Memory in Edge Computing Scenarios](<https://www.ijcai.org/proceedings/2026/618>) | 不确定性门控与动态语义记忆 | — |

<a id="year-2026-arxiv"></a>

### arXiv · 预印本

| 论文 | 创新抓手 | 资源 |
| --- | --- | --- |
| **AgenticVAU**<br>[AgenticVAU: Multi-Agent Explore-Verify Reasoning for Video Anomaly Understanding](<https://arxiv.org/abs/2608.03779>) | 多智能体探索验证与证据记忆 | — |
| **AnomalyCraft-700K**<br>[AnomalyCraft-700K: Component-Level Controllable and Verifiable Synthetic Anomalies for Fine-Grained Video Anomaly Understanding](<https://arxiv.org/abs/2609.06978>) | 语义组件控制生成与逐项验证 | [项目](<https://github.com/Eagen-l/AnomalyCraft>) |
| **Probe-VAD**<br>[Probe-VAD: Ordinal Likelihood Probing for Training-Free Video Anomaly Detection](<https://arxiv.org/abs/2609.17211>) | 序数语言探测与一致性评分 | [代码](<https://github.com/yvestine/Probe-VAD>) |
| **SRVAU-R1**<br>[SRVAU-R1: Enhancing Video Anomaly Understanding via Reflection-Aware Learning](<https://arxiv.org/abs/2602.01004>) | 反思修正序列训练 | — |
| **TAU-Bench**<br>[TAU-Bench: From Anomaly Instance Tracking to Fine-Grained Video Anomaly Understanding](<https://arxiv.org/abs/2608.05699>) | 实例轨迹与层级语义联合评估 | [项目](<https://yarkupa.github.io/tau-bench.github.io/>) |
| **Vad-R1-Plus**<br>[Advancing Adaptive Multi-Stage Video Anomaly Reasoning: A Benchmark Dataset and Method](<https://arxiv.org/abs/2601.10165>) | 感知认知行动链与异常感知优化 | [项目](<https://github.com/wbfwonderful/Vad-R1-Plus>) |

<a id="year-2025"></a>

## 2025

<a id="year-2025-acm-mm"></a>

### ACM MM

| 论文 | 创新抓手 | 资源 |
| --- | --- | --- |
| **EventVAD**<br>[EventVAD: Training-Free Event-Aware Video Anomaly Detection](<https://arxiv.org/abs/2504.13092>) | 时空图划分事件边界 | [代码](<https://github.com/YihuaJerry/EventVAD>) |
| **HoloTrace**<br>[HoloTrace: LLM-based Bidirectional Causal Knowledge Graph for Edge-Cloud Video Anomaly Detection](<https://doi.org/10.1145/3746027.3755185>) | 双向因果知识图与边云更新 | [代码](<https://github.com/kongyanye/HoloTrace-MM25>) |

<a id="year-2025-cvpr"></a>

### CVPR

| 论文 | 创新抓手 | 资源 |
| --- | --- | --- |
| **Anomize**<br>[Anomize: Better Open Vocabulary Video Anomaly Detection](<https://openaccess.thecvf.com/content/CVPR2025/html/Li_Anomize_Better_Open_Vocabulary_Video_Anomaly_Detection_CVPR_2025_paper.html>) | 多源语义与标签关系 | — |
| **Holmes-VAU**<br>[Holmes-VAU: Towards Long-term Video Anomaly Understanding at Any Granularity](<https://arxiv.org/abs/2412.06171>) | 多粒度指令与采样 | [代码](<https://github.com/pipixin321/HolmesVAU>) · [核验](<https://openaccess.thecvf.com/content/CVPR2025/papers/Zhang_Holmes-VAU_Towards_Long-term_Video_Anomaly_Understanding_at_Any_Granularity_CVPR_2025_paper.pdf>) |
| **VERA**<br>[VERA: Explainable Video Anomaly Detection via Verbalized Learning of Vision-Language Models](<https://arxiv.org/abs/2412.01095>) | 语言反馈优化问题 | [核验](<https://openaccess.thecvf.com/content/CVPR2025/papers/Ye_VERA_Explainable_Video_Anomaly_Detection_via_Verbalized_Learning_of_Vision-Language_CVPR_2025_paper.pdf>) |

<a id="year-2025-iccv"></a>

### ICCV

| 论文 | 创新抓手 | 资源 |
| --- | --- | --- |
| **VA-GPT**<br>[Aligning Effective Tokens with Video Anomaly in Large Language Models](<https://openaccess.thecvf.com/content/ICCV2025/html/Chen_Aligning_Effective_Tokens_with_Video_Anomaly_in_Large_Language_Models_ICCV_2025_paper.html>) | 时空有效词元对齐 | [核验](<https://openaccess.thecvf.com/content/ICCV2025/papers/Chen_Aligning_Effective_Tokens_with_Video_Anomaly_in_Large_Language_Models_ICCV_2025_paper.pdf>) |

<a id="year-2025-icml"></a>

### ICML

| 论文 | 创新抓手 | 资源 |
| --- | --- | --- |
| **Ex-VAD**<br>[Ex-VAD: Explainable Fine-grained Video Anomaly Detection Based on Visual-Language Models](<https://proceedings.mlr.press/v267/huang25ad.html>) | 解释融合与标签对齐 | — |

<a id="year-2025-neurips"></a>

### NeurIPS

| 论文 | 创新抓手 | 资源 |
| --- | --- | --- |
| **A2Seek / A2Seek-R1** · Datasets and Benchmarks<br>[A2Seek: Towards Reasoning-Centric Benchmark for Aerial Anomaly Understanding](<https://proceedings.neurips.cc/paper_files/paper/2025/hash/de02de513503962e1d21035ab50ce661-Abstract-Datasets_and_Benchmarks_Track.html>) | 图式推理与主动区域观察 | [项目](<https://2-mo.github.io/A2Seek/>) |
| **MoniTor**<br>[MoniTor: Exploiting Large Language Models with Instruction for Online Video Anomaly Detection](<https://arxiv.org/abs/2510.21449>) | 流式记忆与分数队列 | — |
| **PANDA**<br>[PANDA: Towards Generalist Video Anomaly Detection via Agentic AI Engineer](<https://arxiv.org/abs/2509.26386>) | 场景规划与工具反思 | [项目](<https://github.com/showlab/PANDA>) |
| **URF-ZS-HVAA**<br>[A Unified Reasoning Framework for Holistic Zero-Shot Video Anomaly Analysis](<https://proceedings.neurips.cc/paper_files/paper/2025/hash/2aa95cf3b6aefa84d6b001928b107b4e-Abstract-Conference.html>) | 任务内细化与跨任务推理链 | [代码](<https://github.com/Rathgrith/URF-ZS-HVAA>) · [核验](<https://rathgrith.github.io/Unified_Frame_VAA/>) |
| **VAD-DPO**<br>[Do LVLMs Truly Understand Video Anomalies? Revealing Hallucination via Co-Occurrence Patterns](<https://papers.nips.cc/paper_files/paper/2025/hash/99b419554537c66bf27e5eb7a74c7de4-Abstract-Conference.html>) | 反例偏好优化抑制共现捷径 | — |
| **Vad-R1**<br>[Vad-R1: Towards Video Anomaly Reasoning via Perception-to-Cognition Chain-of-Thought](<https://papers.nips.cc/paper_files/paper/2025/hash/abccc325c84dedf23dbe8de3f686c733-Abstract-Conference.html>) | 感知认知推理链与自验证奖励 | [代码](<https://github.com/wbfwonderful/Vad-R1>) |
| **VADTree**<br>[VADTree: Explainable Training-Free Video Anomaly Detection via Hierarchical Granularity-Aware Tree](<https://papers.nips.cc/paper_files/paper/2025/hash/da19d18dfc5434bf419ce9c113f1865f-Abstract-Conference.html>) | 事件边界驱动的层次粒度树 | [代码](<https://github.com/wenlongli10/VADTree>) |

<a id="year-2025-arxiv"></a>

### arXiv · 预印本

| 论文 | 创新抓手 | 资源 |
| --- | --- | --- |
| **Flashback**<br>[Flashback: Memory-Driven Zero-shot, Real-time Video Anomaly Detection](<https://arxiv.org/abs/2505.15205>) | 离线语义记忆与在线匹配 | [核验](<https://arxiv.org/html/2505.15205v2>) |
| **SlowFastVAD**<br>[SlowFastVAD: Video Anomaly Detection via Integrating Simple Detector and RAG-Enhanced Vision-Language Model](<https://arxiv.org/abs/2504.10320>) | 快检测门控与检索增强慢推理 | — |
| **VAU-R1**<br>[VAU-R1: Advancing Video Anomaly Understanding via Reinforcement Fine-Tuning](<https://arxiv.org/abs/2505.23504>) | 多任务奖励与强化微调 | [代码](<https://github.com/GVCLab/VAU-R1>) |

<a id="year-2024"></a>

## 2024

<a id="year-2024-aaai"></a>

### AAAI

| 论文 | 创新抓手 | 资源 |
| --- | --- | --- |
| **VadCLIP**<br>[VadCLIP: Adapting Vision-Language Models for Weakly Supervised Video Anomaly Detection](<https://arxiv.org/abs/2308.11681>) | 视觉语言双分支对齐 | [代码](<https://github.com/nwpu-zxr/VadCLIP>) · [核验](<https://ojs.aaai.org/index.php/AAAI/article/view/28423>) |

<a id="year-2024-cvpr"></a>

### CVPR

| 论文 | 创新抓手 | 资源 |
| --- | --- | --- |
| **CUVA**<br>[Uncovering What, Why and How: A Comprehensive Benchmark for Causation Understanding of Video Anomaly](<https://arxiv.org/abs/2405.00181>) | 事件因果任务分解 | [代码](<https://github.com/fesvhtr/CUVA>) |
| **LAVAD**<br>[Harnessing Large Language Models for Training-free Video Anomaly Detection](<https://arxiv.org/abs/2404.01014>) | 字幕聚合语言评分 | [代码](<https://github.com/lucazanella/lavad>) · [项目](<https://lucazanella.github.io/lavad/>) · [核验](<https://openaccess.thecvf.com/content/CVPR2024/papers/Zanella_Harnessing_Large_Language_Models_for_Training-free_Video_Anomaly_Detection_CVPR_2024_paper.pdf>) |
| **OVVAD**<br>[Open-Vocabulary Video Anomaly Detection](<https://openaccess.thecvf.com/content/CVPR2024/html/Wu_Open-Vocabulary_Video_Anomaly_Detection_CVPR_2024_paper.html>) | 语言知识与异常合成 | [核验](<https://openaccess.thecvf.com/content/CVPR2024/papers/Wu_Open-Vocabulary_Video_Anomaly_Detection_CVPR_2024_paper.pdf>) |
| **TPWNG**<br>[Text Prompt with Normality Guidance for Weakly Supervised Video Anomaly Detection](<https://openaccess.thecvf.com/content/CVPR2024/html/Yang_Text_Prompt_with_Normality_Guidance_for_Weakly_Supervised_Video_Anomaly_CVPR_2024_paper.html>) | 正常引导伪标签学习 | [核验](<https://arxiv.org/abs/2404.08531>) |
| **UCA**<br>[Towards Surveillance Video-and-Language Understanding: New Dataset, Baselines, and Challenges](<https://openaccess.thecvf.com/content/CVPR2024/html/Yuan_Towards_Surveillance_Video-and-Language_Understanding_New_Dataset_Baselines_and_Challenges_CVPR_2024_paper.html>) | 事件句子与时间对齐 | [项目](<https://xuange923.github.io/Surveillance-Video-Understanding>) |

<a id="year-2024-eccv"></a>

### ECCV

| 论文 | 创新抓手 | 资源 |
| --- | --- | --- |
| **AnomalyRuler**<br>[Follow the Rules: Reasoning for Video Anomaly Detection with Large Language Models](<https://arxiv.org/abs/2407.10299>) | 正常规则归纳演绎 | [代码](<https://github.com/Yuchen413/AnomalyRuler>) · [核验](<https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/10568.pdf>) |

<a id="year-2024-neurips"></a>

### NeurIPS

| 论文 | 创新抓手 | 资源 |
| --- | --- | --- |
| **HAWK**<br>[Hawk: Learning to Understand Open-World Video Anomalies](<https://arxiv.org/abs/2405.16886>) | 运动语言监督对齐 | [代码](<https://github.com/jqtangust/hawk>) · [核验](<https://github.com/jqtangust/hawk/blob/main/README.md>) |

<a id="year-2023"></a>

## 2023

<a id="year-2023-cvpr"></a>

### CVPR

| 论文 | 创新抓手 | 资源 |
| --- | --- | --- |
| **EVAL**<br>[EVAL: Explainable Video Anomaly Localization](<https://openaccess.thecvf.com/content/CVPR2023/html/Singh_EVAL_Explainable_Video_Anomaly_Localization_CVPR_2023_paper.html>) | 对象运动属性解释 | — |
