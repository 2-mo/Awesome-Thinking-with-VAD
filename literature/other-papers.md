# 其他论文 / Other papers

按发表场所整理 WACV、Findings 与 workshop 的视频异常检测、解释、理解和评测论文。标题链接指向官方论文集；年份采用正式发表年份。

核验日期：2026-10-01。本页补充 **5 篇 WACV 论文、6 篇 workshop 论文**，并链接已收录的 NAACL Findings 条目。完整文献统计见[论文年表](../llm4vad.md)。

[WACV](#wacv) · [Workshops](#workshops) · [NAACL Findings](#naacl-findings)

<a id="wacv"></a>

## WACV

### 2026

- **VADER** — [VADER: Towards Causal Video Anomaly Understanding with Relation-Aware Large Language Models](https://openaccess.thecvf.com/content/WACV2026/html/Cheng_VADER_Towards_Causal_Video_Anomaly_Understanding_with_Relation-Aware_Large_Language_WACV_2026_paper.html)。结合上下文采样、物体关系与视觉线索，生成异常描述、因果解释并支持问答。
- **AnyAnomaly** — [AnyAnomaly: Zero-Shot Customizable Video Anomaly Detection with LVLM](https://openaccess.thecvf.com/content/WACV2026/html/Ahn_AnyAnomaly_Zero-Shot_Customizable_Video_Anomaly_Detection_with_LVLM_WACV_2026_paper.html)。用户通过文本定义异常事件，以上下文感知视觉问答实现可定制检测，无需微调 LVLM。
- **ASK-HINT** — [Unlocking Vision-Language Models for Video Anomaly Detection via Fine-Grained Prompting](https://openaccess.thecvf.com/content/WACV2026/html/Zou_Unlocking_Vision-Language_Models_for_Video_Anomaly_Detection_via_Fine-Grained_Prompting_WACV_2026_paper.html)。将细粒度动作与人—物交互组织成分组提示，引导冻结 VLM 判断异常并提供推理线索。

### 2025

- **MissionGNN** — [MissionGNN: Hierarchical Multimodal GNN-Based Weakly Supervised Video Anomaly Recognition with Mission-Specific Knowledge Graph Generation](https://openaccess.thecvf.com/content/WACV2025/html/Yun_MissionGNN_Hierarchical_Multimodal_GNN-Based_Weakly_Supervised_Video_Anomaly_Recognition_with_WACV_2025_paper.html)。利用大语言模型生成任务相关知识图谱，通过分层多模态图网络进行弱监督视频异常识别。

### 2023

- [Towards Interpretable Video Anomaly Detection](https://openaccess.thecvf.com/content/WACV2023/html/Doshi_Towards_Interpretable_Video_Anomaly_Detection_WACV_2023_paper.html)。以物体及其交互构建场景图，检测异常并解释异常发生的上下文与原因。

<a id="workshops"></a>

## Workshops

### WACV 2026 Workshops · RWS

- **PrismVAU** — [PrismVAU: Prompt-Refined Inference System for Multimodal Video Anomaly Understanding](https://openaccess.thecvf.com/content/WACV2026W/RWS/html/Erregue_PrismVAU_Prompt-Refined_Inference_System_for_Multimodal_Video_Anomaly_Understanding_WACVW_2026_paper.html)。使用单个现成 MLLM 完成异常评分与解释，通过弱监督自动提示工程优化文本锚点和提示。

### CVPR 2026 Workshops · SVC

- **T-VAU** — [Text-guided Fine-Grained Video Anomaly Understanding](https://openaccess.thecvf.com/content/CVPR2026W/SVC/html/Gu_Text-guided_Fine-Grained_Video_Anomaly_Understanding_CVPRW_2026_paper.html)。将像素级时空异常热图转为区域提示，在统一流程中连接细粒度定位与语义解释。

### CVPR 2026 Workshops · ABAW

- [From Frames to Events: Rethinking Evaluation in Human-Centric Video Anomaly Detection](https://openaccess.thecvf.com/content/CVPR2026W/ABAW/html/Rashvand_From_Frames_to_Events_Rethinking_Evaluation_in_Human-Centric_Video_Anomaly_CVPRW_2026_paper.html)。以完整异常事件为定位和评测单位，使用时序交并比匹配及多阈值 F1 补充帧级指标。

### CVPR 2025 Workshops · VAND

- **SmartHome-Bench** — [SmartHome-Bench: A Comprehensive Benchmark for Video Anomaly Detection in Smart Homes Using Multi-Modal Large Language Models](https://openaccess.thecvf.com/content/CVPR2025W/VAND/html/Zhao_SmartHome-Bench_A_Comprehensive_Benchmark_for_Video_Anomaly_Detection_in_Smart_CVPRW_2025_paper.html)。智能家居异常理解基准，提供异常标签、描述与推理标注，并研究分类体系引导的反思式 LLM 链。

### CVPR 2024 Workshops · ABAW

- [Evaluating the Effectiveness of Video Anomaly Detection in the Wild: Online Learning and Inference for Real-world Deployment](https://openaccess.thecvf.com/content/CVPR2024W/ABAW/html/Yao_Evaluating_the_Effectiveness_of_Video_Anomaly_Detection_in_the_Wild_CVPRW_2024_paper.html)。评估姿态异常检测方法在跨域视频流中的在线学习、持续适应与部署表现。

### CVPR 2023 Workshops · O-DRUM

- **TEVAD** — [TEVAD: Improved Video Anomaly Detection With Captions](https://openaccess.thecvf.com/content/CVPR2023W/O-DRUM/html/Chen_TEVAD_Improved_Video_Anomaly_Detection_With_Captions_CVPRW_2023_paper.html)。融合视频描述的文本特征与视觉特征，并通过描述分析提供异常判定的解释线索。

<a id="naacl-findings"></a>

## NAACL Findings

- **VANE-Bench · 2025** — [已收录条目](../llm4vad.md#year-2025-naacl-findings) · [官方论文](https://aclanthology.org/2025.findings-naacl.171/)。视频异常问答评测；引用信息见[引用索引](citations.md#cite-vane-bench)。

[返回 README](../README.zh-CN.md#其他论文) · [English README](../README.md#other-papers)
