# 视频异常检测补充与正常性建模线路 · 2026-10-04

> 历史方案：本批次线路扩展已由用户范围澄清后的[语义检测补充](semantic-detection-2026-10-04.md)取代。正常性线路及本批次新增的九个地图站点已撤下，文献记录与引用保留。

本轮按用户要求扩充线路图中的视频异常检测代表作。新增 7 篇结构化论文，将已有 2 篇 TPAMI 基础工作选入地图；从本轮开始时的 114 篇目录／99 站，更新为 **121 篇目录、108 站、9 条线路**。保留其他已有选篇取舍，不补 2021 年会议论文。

## 入图选择

| 论文 | 正式发表 | 补足的研究问题 | 放置 |
| --- | --- | --- | --- |
| [Future Frame Prediction](https://doi.org/10.1109/TPAMI.2021.3129349) | TPAMI 2022 | 用正常事件可预测性检测异常与跨场景适配 | 原目录入图，正常性建模起点 |
| [MGFN](https://ojs.aaai.org/index.php/AAAI/article/view/25112) | AAAI 2023 | 场景差异下的特征幅值判别与长短程时序 | 检测线早期站点 |
| [UR-DMU](https://ojs.aaai.org/index.php/AAAI/article/view/25489) | AAAI 2023 | 正常／异常双记忆和正常分布不确定性 | 检测线早期站点 |
| [STG-NF](https://openaccess.thecvf.com/content/ICCV2023/html/Hirschorn_Normalizing_Flows_for_Human_Pose_Anomaly_Detection_ICCV_2023_paper.html) | ICCV 2023 | 姿态序列的精确概率评分 | 正常性建模 |
| [FPDM](https://openaccess.thecvf.com/content/ICCV2023/html/Yan_Feature_Prediction_Diffusion_Model_for_Video_Anomaly_Detection_ICCV_2023_paper.html) | ICCV 2023 | 运动预测与外观细化的双扩散建模 | 正常性建模 |
| [SSMCTB](https://arxiv.org/abs/2209.12148) | TPAMI 2024 | 网络内部的掩码重构与通道注意力 | 原目录入图，正常性建模 |
| [Self-Distilled MAE](https://openaccess.thecvf.com/content/CVPR2024/html/Ristea_Self-Distilled_Masked_Auto-Encoders_are_Efficient_Video_Anomaly_Detectors_CVPR_2024_paper.html) | CVPR 2024 | 运动引导重构、自蒸馏和效率 | 正常性建模 |
| [BN-WVAD](https://ieeexplore.ieee.org/document/10649595/) | TCSVT 2024 | 统计正常性判据与视频级弱监督结合 | 正常性／检测换乘站 |
| [ADSM](https://openaccess.thecvf.com/content/ICCV2025/html/Zhang_Autoregressive_Denoising_Score_Matching_is_a_Good_Video_Anomaly_Detector_ICCV_2025_paper.html) | ICCV 2025 | 场景、运动与局部异常模式的去噪评分 | 正常性建模终点 |

## 线路结构与边界

上方新增 `normality`（Normality Modeling，正常性建模）：Future Frame Prediction → STG-NF → FPDM → SSMCTB → Self-Distilled MAE → BN-WVAD → ADSM。BN-WVAD 的 DFM 核心判据显式使用 BatchNorm 均值参照，并修正异常分类器，因此有来源地连接正常性与弱监督检测。连接是编辑阅读顺序，不声明这些方法依次继承。

棕色检测线向前补 MGFN → UR-DMU → VadCLIP，并在 PEL 后、DSRL 前的 2024 年 12 月段加入 BN-WVAD。原有 NWPU Campus、DoTA、时序记忆与多模态理解线路保留。

“正常性建模”不等于统一监督协议：STG-NF 同时研究正常训练和监督变体；Self-Distilled MAE 使用合成异常增强；BN-WVAD 依赖视频级标签。ADSM 的自回归去噪不据此标为严格在线检测，未来帧预测也不自动等于事故提前预警。

## 来源与配图

CVF 正式 BibTeX 确定会议年、月、作者和页码；AAAI 月份采用主会二月，六月网页出版日期不替代会议日期。BN-WVAD 用 [IEEE 页面](https://ieeexplore.ieee.org/document/10649595/)及[出版商登记书目](https://api.crossref.org/works/10.1109/TCSVT.2024.3450734)，采用 TCSVT 34(12):13642–13654，非 CVPR。

补齐 8 张原论文框架图（7 篇新增论文与 SSMCTB），并记录出处、页码和署名。MGFN、UR-DMU、BN-WVAD 与 SSMCTB 使用作者预印本图，发表引用独立采用正式版本；Future Frame Prediction 延续原有待配图记录。原图裁切已查看，未使用生成图替代。

## 实现范围

更新共同数据源、引用、索引、网页构建及 SVG／PNG。修正检测线把“位于理解线上方”误写成“必须是第一条线”的布局假设，支持新线路放在检测线上方；既有换乘与线路数量断言随数据更新。未发布或推送远端。

## 验证结果

生产构建、数据校验、生成文件一致性和 TypeScript 均通过。现有测试 125 项中 121 项通过；4 项关于 TD-VAD 的既有布局断言失败（跨越活动方法带、检测线垂直段、长换乘弯折位置、名称与蓝线距离）。用更新前目录的隔离副本运行相同几何测试，重现完全相同的 4 项失败，未增加失败项。本轮按用户要求聚焦论文补充，不继续重构这些既有几何问题。新增线路局部、完整导出图与原论文配图已查看。
