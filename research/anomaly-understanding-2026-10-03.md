# 异常理解补录 · 2026-10-03

本批次补入 5 篇论文及其相关数据资源，按已有方法分类入图。论文目录由 83 篇增至 88 篇，地图由 80 站增至 85 站；保留既有选篇与异常理解主线，不新增场景分类。

| 论文 | 正式发表 | 入图依据 | 一手来源 |
| --- | --- | --- | --- |
| JUDO | ICLR 2026 | 正常参照并置分割、领域知识 SFT 与多奖励 GRPO，归入结构化推理与验证 | [正式论文集](https://proceedings.iclr.cc/paper_files/paper/2026/hash/92a7a03e1c716970848a4a86cc8243ee-Abstract-Conference.html) · [实现](https://github.com/woodavid31/JUDO) |
| Phys-AD | CVPR 2025 | 物理交互异常数据与 PAEval，区分异常判别、现象描述与物理原因解释，归入评测 | [正式论文集](https://openaccess.thecvf.com/content/CVPR2025/html/Li_Towards_Visual_Discrimination_and_Reasoning_of_Real-World_Physical_Dynamics_Physics-Grounded_CVPR_2025_paper.html) · [项目](https://guyao2023.github.io/Phys-AD/) |
| MMAD | ICLR 2025 | 七任务图像问答评测，直接支撑 JUDO 的训练与实验，归入评测 | [正式论文集](https://proceedings.iclr.cc/paper_files/paper/2025/hash/d91ffbe9c126765755ff52d36b715683-Abstract-Conference.html) · [实现与数据](https://github.com/jam-cc/MMAD) |
| Anomaly-OV | CVPR 2025 | Look-Twice 特征匹配与异常视觉 token 选择支撑细粒度解释，归入表征对齐与融合 | [正式论文集](https://openaccess.thecvf.com/content/CVPR2025/html/Xu_Towards_Zero-Shot_Anomaly_Detection_and_Reasoning_with_Multimodal_Large_Language_CVPR_2025_paper.html) · [项目](https://xujiacong.github.io/Anomaly-OV/) |
| EchoTraffic | CVPR 2025 | 声音引导关键帧与音视频动态连接器支持交通异常问答，归入表征对齐与融合 | [正式论文集](https://openaccess.thecvf.com/content/CVPR2025/html/Xing_EchoTraffic_Enhancing_Traffic_Anomaly_Understanding_with_Audio-Visual_Insights_CVPR_2025_paper.html) · [实现与数据](https://github.com/HarryHsing/EchoTraffic) |

## 核验与阅读边界

- 逐篇核对正式论文集的题名、作者顺序、年份与原文。MMAD 为 ICLR 2025，采用正式题名，不沿用早期预印本题名中的 First-Ever。JUDO 按 ICLR 2026 四月入图，不以五月 arXiv 上传时间代替会议时间。
- JUDO 的任务覆盖包含二元判别、位置与缺陷分析，不能把问答平均分提升概括成所有检测任务都更优；其领域推理奖励含伪理由语义相似度，需区分答案准确与解释忠实。
- Phys-AD 正文统计为 2,400 段训练视频加 4,034 段测试视频，项目页写 6,359 段。数据卡保留版本口径差异，未尝试把两个总数合并。旧 `cvf-notes.md` 的本轮排除记录保留为历史，本次已重新核验并纳入。
- Anomaly-OV 的零样本指评测设置，仍使用异常监督与视觉指令微调。VisA-D&R 将缺陷描述与可能原因／建议分开评估，后者不等同于真实因果验证。
- EchoTraffic 的五任务是事件描述、原因、时间定位、预防策略与事件响应；预防建议不标为事故提前预测。AV-TAU 是音视频理解数据。
- 新增 Phys-AD、MMAD、Anomaly-Instruct-125k、VisA-D&R、AV-TAU 共 5 项资源；根据作者发布入口记录开放状态，未下载完整数据集。关系只登记原文明示的 `introduces` / `uses`，没有添加推测的 `extends`。
- 五篇配图均从正式 PDF 渲染并裁取原图：JUDO Figure 1、Phys-AD Figure 1、MMAD Figure 2、Anomaly-OV Figure 3、EchoTraffic Figure 4。保留来源页、图号与作者署名。

## 线路安排

保留 Vad-R1 → Cue-R1 → FineVAU → Vad-R1-Plus 的局部评测线路及后期 CG-CoE 线路。早期评测阅读段由 CUVA 接出，经 MMAD 至 Phys-AD，按真实发表时间排列；CUVA 原有因果理解方法与 MMEval 基准支持其兼属推理、评测两线。三段使用同一评测分类，并非三条新研究方向。

校验器允许早于指定分叉的独立阅读段，但要求该段从有来源的父／子方向共享论文起始，且整段早于后续分叉；无锚点或跨越后续分叉的早期段仍被拒绝。新增回归检查覆盖这些约束。原有主线、真实站点连接、时间顺序、标签与轨道避让检查继续执行。

## 后续候选

- [ADSeeker（CVPR 2026）](https://openaccess.thecvf.com/content/CVPR2026/html/Zhang_ADSeeker_A_Knowledge-Grounded_Reasoning_Framework_for_Industry_Anomaly_Detection_and_CVPR_2026_paper.html)：视觉文档知识检索与异常解释值得后续精读，尤其可与 JUDO 的知识内化作方法比较。本轮仅核验正式摘要，尚未完成正文、评测与配图核验，不入图。
- [AnomalyGPT（AAAI 2024）](https://ojs.aaai.org/index.php/AAAI/article/view/27963)：交互式工业异常检测与描述的早期背景。可补充阅读脉络，本轮优先加入具有更明确理解评测的 MMAD 和 Anomaly-OV，未扩充为工业检测全目录。

本批次无论文版本合并。

验证：`npm run build` 与 `npm run check` 通过，共 112 项测试；浏览器检查 85 个站点、五篇新增论文详情、手机搜索与详情，以及 SVG/PNG 导出。五篇原图裁切与整图均经人工目视检查，浏览器无运行错误。
