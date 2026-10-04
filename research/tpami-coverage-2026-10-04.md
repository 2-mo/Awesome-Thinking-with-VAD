# TPAMI 补录与道路异常理解节点

核验日期：2026-10-04。当前 TPAMI 文献见[期刊索引](../archive/journals/tpami.md)，逐篇来源与完整作者见[引用目录](../literature/citations.md)。

## 本批次结果

新增 14 篇文献，其中 TPAMI 12 篇、CVPR 2024 一篇、TCSVT 2024 一篇。目录共 107 篇，TPAMI 共 16 篇；地图 93 站，数据资源 35 项，原图覆盖 82 篇。文献与线路图分别选篇：新增地图节点为 DoTA、CMCIR、AdVersa-SD、TTHF，其余十篇保留为检测基础、轨迹建模与综述阅读。

2021 年的会议论文不加入本批次；SUTD-TrafficQA 仅作为 CMCIR 实际使用的数据资源登记，没有增加 Eclipse 或 DRIVE 的论文记录和地图节点。TPAMI 2021 的稀疏编码论文保留。

## 新增地图节点

| 论文 | 正式发表 | 阅读位置与依据 |
| --- | --- | --- |
| [DoTA](https://doi.org/10.1109/TPAMI.2022.3150763) | TPAMI 45(1):444–459，2023-01 | 时序建模线：自车运动与对象轨迹预测；同时提出 DoTA 和 STAUC。数据提供何时、何处、何种异常，不把类别标注当作原因解释，也不把预测式检测标成提前预警。 |
| [CMCIR / CMCR](https://doi.org/10.1109/TPAMI.2023.3284038) | TPAMI 45(10):11624–11641，2023-10 | 推理线早期节点：视觉前门、语言后门干预与事件级问答。与本项目的连接来自 SUTD-TrafficQA 交通归因／反事实任务；不是专门的 VAD 检测器。 |
| [AdVersa-SD](https://openaccess.thecvf.com/content/CVPR2024/html/Fang_Abductive_Ego-View_Accident_Video_Understanding_for_Safe_Driving_Perception_CVPR_2024_paper.html) | CVPR 2024 | 推理线：MM-AU、事故原因／预防文本对、AbductiveCLIP 与对象中心扩散。复用已有 MM-AU 数据条目。 |
| [TTHF](https://doi.org/10.1109/TCSVT.2024.3390173) | TCSVT 34(9):8684–8697，2024-09 | 表征对齐线：文本引导、时序高频变化和异常聚焦。实际评测 DoTA；文本对齐不等同于生成事故原因。 |

连线表示阅读次序。AdVersa-SD 与 TPAMI ADVersa 保留独立引用，尚未登记直接 `extends`；已有 NWPU Campus 与 DSRL 的明确扩展关系保持不变。

## TPAMI 检索与取舍

检索 IEEE 向 Crossref 登记的 TPAMI 书目（ISSN 0162-8828），分别查询 video anomaly、abnormal event、surveillance、traffic accident、violence、causal reasoning，并以作者论文、机构页面和 PubMed 所收录的作者摘要核对内容。不能只检索标题里的 anomaly：CMCIR、Scene Dynamics、PiercingEye、ADVersa 均需通过相关主题和作者来源补查。

本批次补入：Behavior Profiling（2008）、Scene Dynamics（2009）、Crowded-Scene AD（2014）、SHNN-CAD（2014）、MDI（2019）、TSC / sRNN-AE（2021）、Single-Scene VAD Survey（2022）、Background-Agnostic VAD（2022）、Future Frame Prediction（2022）、DoTA（2023）、CMCIR（2023）、SSMCTB（2024）。既有四篇为 Latent-Space VAA、Multilingual VAD、PiercingEye、ADVersa。

- SHNN-CAD 是部分轨迹的在线检测与报警校准，作为轨迹分析背景收录，明确区别于视频语义理解。
- MDI 的作者摘要明确包含视频监控实验，作为连续时空异常定位基础收录。[作者摘要](https://arxiv.org/abs/1804.07091)
- SSMCTB 同时覆盖图像、监控视频与热成像视频，作者明确说明其扩展 SSPCAB；此批次只登记 TPAMI 本身，不凭版本线索扩充会议站点。[作者摘要](https://arxiv.org/abs/2209.12148)
- [DyMETER](https://arxiv.org/abs/2604.14726)、[高维非参数异常检测](https://arxiv.org/abs/1809.05250)等数据流／通用统计异常工作未确认直接的视觉异常理解贡献，不因期刊名称或 anomaly 关键词收进主线。
- 工业缺陷分割、图异常、医学／高光谱异常等检索命中项同样按实际任务筛选；这份清单覆盖本次核实的 VAD/VAU 及其直接基础，不代表所有异常检测应用。

## 年份与版本更正

正式卷期优先于 DOI 内年份和 Early Access 日期：

- DoTA：DOI 含 2022，正式卷期为 **2023**；2020 年作者预印本题名为 *When, Where, and What?*，不把它写成另一篇会议论文。
- TSC / sRNN-AE：DOI 含 2019，正式卷期为 **2021**。
- Background-Agnostic VAD：Crossref 仍有 2021、1–1 的占位记录，采用作者摘要对应最终书目 **2022，44(9):4505–4523**。[最终记录](https://europepmc.org/article/MED/33881990)
- Single-Scene VAD Survey：Early Access 为 2020，最终为 **2022，44(5):2293–2312**。[最终记录](https://europepmc.org/article/MED/33237854) · [作者机构页](https://www.merl.com/publications/TR2021-029)

## 原图与数据

- DoTA 配图来自 2020 年作者预印本 Figure 2，明确标为数据样例，不用它代表期刊新增框架。
- CMCIR 使用作者提供的 TPAMI 最终版 Figure 3；AdVersa-SD 使用 CVF 正式论文 Figure 1；TTHF 使用作者期刊稿 Figure 2。
- DoTA 数据加入目录，关联年份使用 TPAMI 2023，同时明确 2020 年预印本已介绍数据。
- SUTD-TrafficQA 记录完整数据需申请、公开示例可查看，并说明它同时包含一般交通事件与事故。
- 十篇基础／综述阅读文献本批次未整理原图，按实际状态登记待补，不以示意图代替原论文图。

## 同步与验证

生成文献索引、引用和图片来源清单，执行一次网页构建并更新 SVG/PNG。只查看新增节点所在区段与新增原图，不运行全量测试或重复检查未改动功能。
