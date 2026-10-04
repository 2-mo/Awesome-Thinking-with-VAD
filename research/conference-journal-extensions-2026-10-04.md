# 会议首版与期刊扩展 · 2026-10-04

## NWPU Campus → Latent-Space VAA

- **CVPR 2023**：[A New Comprehensive Benchmark for Semi-Supervised Video Anomaly Detection and Anticipation](https://openaccess.thecvf.com/content/CVPR2023/html/Cao_A_New_Comprehensive_Benchmark_for_Semi-Supervised_Video_Anomaly_Detection_and_CVPR_2023_paper.html)，20392–20401。基准与场景条件前后向预测方法合并标为混合贡献，地图简称 NWPU Campus。
- **TPAMI 2025**：[Scene-Dependent Prediction in Latent Space for Video Anomaly Detection and Anticipation](https://doi.org/10.1109/TPAMI.2024.3461718)，47(1): 224–239，2025 年 1 月正式卷期；2024-09-16 为在线发表。地图简称 Latent-Space VAA 为编辑缩写。
- [作者扩展页](https://campusvaa.github.io/)明确标为 Extension version，登记 `scene-dependent-vaa extends nwpu-campus-paper`。两站接入时序建模与记忆，数据集首发仍为 CVPR 2023，不重复新增数据项。
- 首版配图取自正式 PDF Figure 4（第 6 页）。期刊方法图暂缺，不用首版图片代替新增潜空间机制。

## DSRL → PiercingEye

- **NeurIPS 2024**：[Beyond Euclidean: Dual-Space Representation Learning for Weakly Supervised Video Violence Detection](https://proceedings.neurips.cc/paper_files/paper/2024/hash/1f471322127d6347e5ae09a14b1e5cf7-Abstract-Conference.html)，37: 17373–17397，DOI 10.52202/079017-0552。DSRL 用双曲能量约束的层次聚合和跨空间注意力结合视觉与事件关系。
- **TPAMI 2026**：[PiercingEye: Dual-Space Video Violence Detection With Hyperbolic Vision-Language Guidance](https://doi.org/10.1109/TPAMI.2025.3617460)，48(2): 1689–1706，2026 年 2 月。[IEEE 登记的 Crossref 书目](https://api.crossref.org/works/10.1109/TPAMI.2025.3617460)确认完整作者顺序与正式卷期；作者页的 2025 年标记不替代卷期年份。
- [作者稿 §I](https://arxiv.org/html/2504.18866v1#S1)明确 DSRL 为会议首版，新增易混淆事件文本生成 AETG 和双曲视觉语言监督 HVLGL；[官方代码仓库](https://github.com/wuzhanjie123/PiercingEye)也直接确认扩展身份。登记 `piercingeye extends dsrl`，两站加入现有表征对齐与融合线。
- 两篇均评测 XD-Violence、UCF-Crime。易混淆子集仅记作评测设置，未另外建数据集。生成文本属于训练信号，不把检测输出写成自然语言解释。
- 配图分别来自 NeurIPS 正式 PDF Figure 2（第 5 页）和 PiercingEye 作者预印本 Figure 2（第 6 页），明确保留预印本图源身份。

本批增加 4 篇独立论文及 2 条有依据的扩展关系；保留全部既有版本，未合并节点。目录 93 篇、线路图 89 站、数据资源 33 项；原图覆盖 78 篇。只构建更新产物并查看新增站点附近排版，不运行全量测试。
