# 文献更新 · 2026-09-30

本轮补充 19 篇独立论文，精选目录由 32 篇增至 51 篇。覆盖顶会遗漏与近期视频异常理解新稿；这是一轮范围明确的补充检索，不表示所有会议、期刊或预印本已穷尽。

[按年份阅读](../llm4vad.md) · [按创新思路阅读](../literature/catalog.md)

## 新增条目

| 论文 | 发表 | 主要收录理由 |
| --- | --- | --- |
| [HeadHunt-VAD](https://arxiv.org/abs/2512.17601) | AAAI 2026 | 稳定异常敏感注意力头探测 |
| [VAGU & GtS](https://ojs.aaai.org/index.php/AAAI/article/view/42412) | AAAI 2026 | 先全局粗定位再局部细查 |
| [STCH](https://openaccess.thecvf.com/content/CVPR2026/html/Wang_Streaming_Video_Crime_Anticipation_with_Spatio-Temporal_Causal_Reasoning_CVPR_2026_paper.html) | CVPR 2026 | 流式时空因果超图 |
| [CLUE-VAD](https://eccv.ecva.net/virtual/2026/poster/4744) | ECCV 2026 | 结构化语义线索与类别感知归因 |
| [O-VAD](https://arxiv.org/abs/2607.18142) | ECCV 2026 | 对象状态轨迹与时序推理 |
| [SteerVAD](https://arxiv.org/abs/2602.24021) | ICLR 2026 | 潜在异常专家头与上下文表示校正 |
| [CG-CoE](https://icml.cc/virtual/2026/poster/66013) | ICML 2026 | 类别引导评价链与元评测 |
| [LRPO](https://arxiv.org/abs/2607.00654) | ICML 2026 | 组相对语言经验优化 |
| [TD-VAD](https://arxiv.org/abs/2608.11820) | ICML 2026 | 文本时序监督与事件演化注意力 |
| [AgenticVAU](https://arxiv.org/abs/2608.03779) | arXiv 2026 | 多智能体探索验证与证据记忆 |
| [AnomalyCraft-700K](https://arxiv.org/abs/2609.06978) | arXiv 2026 | 语义组件控制生成与逐项验证 |
| [TAU-Bench](https://arxiv.org/abs/2608.05699) | arXiv 2026 | 实例轨迹与层级语义联合评估 |
| [Vad-R1-Plus](https://arxiv.org/abs/2601.10165) | arXiv 2026 | 感知认知行动链与异常感知优化 |
| [URF-ZS-HVAA](https://proceedings.neurips.cc/paper_files/paper/2025/hash/2aa95cf3b6aefa84d6b001928b107b4e-Abstract-Conference.html) | NeurIPS 2025 | 任务内细化与跨任务推理链 |
| [VAD-DPO](https://papers.nips.cc/paper_files/paper/2025/hash/99b419554537c66bf27e5eb7a74c7de4-Abstract-Conference.html) | NeurIPS 2025 | 反例偏好优化抑制共现捷径 |
| [Flashback](https://arxiv.org/abs/2505.15205) | arXiv 2025 | 离线语义记忆与在线匹配 |
| [SlowFastVAD](https://arxiv.org/abs/2504.10320) | arXiv 2025 | 快检测门控与检索增强慢推理 |
| [VAU-R1](https://arxiv.org/abs/2505.23504) | arXiv 2025 | 多任务奖励与强化微调 |
| [Probe-VAD](https://arxiv.org/abs/2609.17211) | arXiv 2026 | 序数语言探测与一致性评分 |

## 版本和旧条目修正

- **VAGU & GtS** 采用 [AAAI 2026 正式记录](https://ojs.aaai.org/index.php/AAAI/article/view/42412)。[2026 年 8 月扩展稿 Glance, Scrutinize, and Think](https://arxiv.org/abs/2608.11260) 明确注明 journal extension，保留为扩展来源，未确认期刊录用，不重复增加论文节点。裁剪工具、密集重采样、联合奖励和 VAGU-T 属于扩展稿贡献，不倒写入 AAAI 原文摘要。
- **VAU-R1、Vad-R1、Vad-R1-Plus** 为不同论文。VAU-R1 和 Vad-R1-Plus 的主来源未确认正式录用，保留 arXiv；Vad-R1 保持 NeurIPS 2025。
- **SlowFastVAD** 是快速检测与检索增强慢推理的协作，不写成使用 SlowFast 网络。**Flashback** 是离线语义记忆与在线匹配，不写成已证实的动态在线记忆。
- **HeadHunt-VAD、SteerVAD** 冻结大模型，但仍有轻量模块校准／训练，不能把 tuning-free 等同于整个方法无需训练。
- **Anom-π** 保持 ICML 2026；**SRVAU-R1** 按既定记录保持 arXiv，本轮不重新追查其录用。
- `llm4vad.md` 从源数据重建，移除旧文档中的错误 ICCV 2024 分组、空链接及与当前范围不符的泛视频理解／静态图像工作。

## 检索到但未纳入核心目录

- **SAGE** 研究工业图像异常，不属于本目录的视频异常理解主线。
- **Unlocking Vision-Language Models for Video Anomaly Detection via Fine-Grained Prompting** 已有 [WACV 2026 正式论文](https://openaccess.thecvf.com/content/WACV2026/papers/Zou_Unlocking_Vision-Language_Models_for_Video_Anomaly_Detection_via_Fine-Grained_Prompting_WACV_2026_paper.pdf)，不以旧 arXiv 身份规避当前 WACV 排除范围。
- [PRISM](https://icml.cc/virtual/2026/poster/66758) 本轮暂不纳入：主要贡献为语义轴白化与统计打分，保留为后续检测方法背景候选。

## 发布与资源状态

[Probe-VAD](https://arxiv.org/abs/2609.17211) 为 2026-09-15 首发、Under Review 的预印本，以语言判据到异常评分的接口研究纳入；明确其输出异常分数，不把它标成开放式异常解释模型。

录用状态按正式会议页、论文集或作者明确声明核验；预印本与正式主会月份分别用于内部排序。代码与数据未发布时不猜测下载链接。AnomalyCraft-700K、Vad-R1-Plus 的仓库作为项目入口；TAU-Bench 项目页的 Dataset/Model 链接状态不足以证明资源已公开。新增基准论文先作为论文记录，不在缺少图像署名和完整数据协议时补造数据集展项。

本轮也将年表和发表索引纳入自动生成与 CI 一致性检查，减少网页更新而文档遗留旧信息的问题。
