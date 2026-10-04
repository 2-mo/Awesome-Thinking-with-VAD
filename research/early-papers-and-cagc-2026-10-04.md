# 2023 年候选与 CAGC-VAD · 2026-10-04

## 值得补充的早期阅读

| 工作 | 推荐理由 | 处理 |
| --- | --- | --- |
| [NWPU Campus：A New Comprehensive Benchmark for Semi-Supervised Video Anomaly Detection and Anticipation](https://openaccess.thecvf.com/content/CVPR2023/papers/Cao_A_New_Comprehensive_Benchmark_for_Semi-Supervised_Video_Anomaly_Detection_and_CVPR_2023_paper.pdf)，CVPR 2023 | 同一种行为是否异常取决于场景规则；同时提出提前预测任务，适合解释从识别异常到理解语境、预判事件的变化。 | 数据资源已收录；CVPR 2023 首版与 TPAMI 2025 扩展现已分别加入时序建模线，并登记直接扩展关系。它不提供语言解释模型，不应据此标为异常问答工作。 |
| [Towards Generic Anomaly Detection and Understanding: Large-scale Visual-linguistic Model (GPT-4V) Takes the Lead](https://arxiv.org/abs/2311.02782)，2023 年 11 月初稿 | 通过类别、正常标准和参照图提示研究异常解释，包含工业、逻辑与交通案例，直接贴近异常理解。 | 推荐作为早期探索阅读；§1.3 明确以有限案例的定性评价为主。[作者仓库](https://github.com/caoyunkang/GPT4V-for-Generic-Anomaly-Detection) 描述另有 CSCWD 标记，当前正式版本尚未匹配，先不确定地图年份。 |

NWPU 的期刊扩展为 **Scene-Dependent Prediction in Latent Space for Video Anomaly Detection and Anticipation**，作者 Congqi Cao、Hanwen Zhang、Yue Lu、Peng Wang、Yanning Zhang。[作者项目页](https://campusvaa.github.io/) 明确标为 Extension version；[IEEE 正式记录](https://ieeexplore.ieee.org/abstract/document/10681297/) 给出 TPAMI **47(1): 224–239，2025 年 1 月**，在线发表日期为 **2024-09-16**，DOI **10.1109/TPAMI.2024.3461718**。扩展版通过层次变分自编码器、潜空间扩散和场景信息自编码器建模事件与场景关系，并以关键帧时序损失约束运动一致性；这些扩展机制不倒写进 CVPR 首版。数据集首发仍记 CVPR 2023，期刊正式引用采用 2025 卷期，区别于项目页沿用的 2024 在线年份。

[EVAL（CVPR 2023）](https://openaccess.thecvf.com/content/CVPR2023/html/Singh_EVAL_Explainable_Video_Anomaly_Localization_CVPR_2023_paper.html) 已有站点。[AnomalyGPT](https://ojs.aaai.org/index.php/AAAI/article/view/27963) 虽于 2023 年公开预印本，正式发表为 AAAI 2024；不为补 2023 年节点而改写年份。

## CAGC-VAD

用户提供 [IEEE 论文页](https://ieeexplore.ieee.org/abstract/document/11701332) 与 DOI [10.1109/TCSVT.2026.3735599](https://doi.org/10.1109/TCSVT.2026.3735599)。[IEEE 登记的 Crossref 书目](https://api.crossref.org/works/10.1109/TCSVT.2026.3735599) 确认：

- 题名：CAGC-VAD: Controlled Abstraction and Graph Competition for Video Anomaly Detection。
- 作者：Yalong Jiang、Zonghua Gu、Amin Serami、Ruimin Li、Hong Fu、Yong Xia。
- 期刊：IEEE Transactions on Circuits and Systems for Video Technology，2026。
- 元数据未提供可核验发表月份；页码 1–1 为占位，未写入正式引文。

IEEE 页面及 PDF 本次均返回访问限制，未取得摘要或正文。先补目录与引用，按题名暂定归类，暂不入图；未推断 LLM 使用、解释输出、数据集或实验成绩。没有新增数据或继承关系，没有合并版本。
