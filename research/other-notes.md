# 非 CVF 顶会遗漏核验（2026-09-29）

本次交付 `additions-other.json` 8 篇，未改 catalog 或既有文档。优先直接 VAU / VAR / explanation，再保留改变异常判据或证据获取方式的 VLM-VAD。摘要基于一手论文集；takeaway 和 limitation 是阅读归纳而非作者原文结论。空 datasetIds 表示未为现有六个 dataset 节点核实使用关系，不表示论文没有数据集。

## 正式发表依据

- Vad-R1、VADTree：NeurIPS 2025 Main Conference Track 的正式论文集摘要页。
- CueBench、TargetVAU：AAAI 2026 Technical Tracks，正式 DOI 与卷期年份均可核对。
- HoloTrace：ACM MM 2025 正式 DOI 页面列明 33rd ACM International Conference on Multimedia、27 October 2025。
- LaGoVAD：ICLR 2026 正式 proceedings.iclr.cc 页面，不依赖 OpenReview 的旧匿名 under-review PDF。
- MemoVAD：IJCAI 2026 正式 Main Track，页码 5549–5557，DOI 10.24963/ijcai.2026/618。
- A2Seek：NeurIPS 2025 正式 Main Conference 卷下的 **Datasets and Benchmarks Track**。venue 明确写分轨，避免称作 Main Conference Track。如果产品“主会”严格仅指 research main track，可以移除这一条；不是 workshop 或 arXiv。

## 已确认会议但因范围优先级未纳入

- HeadHunt-VAD，AAAI 2026：<https://ojs.aaai.org/index.php/AAAI/article/view/39066>。正式论文集可确认；核心是冻结 MLLM 注意力头选择、轻量打分与定位，减少对文本输出依赖。为防将 VAU 扩成一般检测大全暂不纳入。
- SteerVAD，ICLR 2026：<https://proceedings.iclr.cc/paper_files/paper/2026/hash/634f75be78884d597c04eedeb12cc6fe-Abstract-Conference.html>。正式论文集可确认；通过表征可分性选择潜在异常专家头，再以层次控制器整形内部表征。1% training data 的校准不能写成完全零样本。
- CMHKF，ACL 2025 Long Papers：<https://aclanthology.org/2025.acl-long.1524/>。跨视频、音频、文本知识融合，明确实验 XD-Violence；偏语言辅助检测，没有直接异常解释/推理任务，本轮未纳入。
- VANE-Bench：<https://aclanthology.org/2025.findings-naacl.171/> 为 **Findings of NAACL 2025**，不应记作 ACL/NAACL main；本轮主会限制下未纳入。

## 需要继续核实细节的强候选

- LRPO：Linguistic Relative Policy Optimization for Video Anomaly Reasoning。官方 <https://icml.cc/Downloads/2026> 可检出题名；作者主页 <https://faculty.cqupt.edu.cn/lengjiaxu/en/index/106917/list/index.htm> 明确 ICML 2026；作者 arXiv <https://doi.org/10.48550/arXiv.2607.00654> 明确 Accepted at ICML 2026，方法是以多条推理轨迹形成通用/场景语言经验，并注入上下文，不更新参数。已知 poster <https://icml.cc/virtual/2026/poster/64285>、OpenReview id P8tBsNibfm 的具体页面本轮无法读取。按父任务要求暂不纳入，建议后续取得 official poster / PMLR 后优先加入。
- Towards Trustworthy Video Anomaly Understanding: A Class-Guided Chain-of-Evaluation Metric and An Anomaly-focused Meta-Benchmark。上述作者主页明确 ICML 2026；poster <https://icml.cc/virtual/2026/poster/66013>、OpenReview id 7waVdY1WmW 本轮读不到。第三方转录提及 CG-CoE、AEA 与 CVP，但未将其作为数据来源，暂不纳入。
- 既有 Anom-π / Learning to Watch：2026-09-30 补核作者 arXiv 摘要页 <https://arxiv.org/abs/2607.00622>，Comments 明确标注 Accepted at ICML 2026；已将目录会议更正为 ICML 2026。此前官方 ICML 2026 Downloads 索引与作者主页亦列出该文；poster <https://icml.cc/virtual/2026/poster/60569> 未能直接读取。
- HiProbe-VAD：arXiv <https://arxiv.org/abs/2507.17394> 有 ACM DOI 10.1145/3746027.3755575，但 DOI 页读取失败。第三方引文称 MM25 592–601，不作为本轮正式会议核验依据；暂不纳入。

## 其余边界

2023–2024 没有为凑年份补普通纯检测。Hawkeye 属 recon-video 隐式情感异常，相比监控视频异常理解任务边界较远。普通工业视觉缺陷、privacy-preserving video understanding、general video reasoning 均未灌入。未确认新的 EMNLP 主会直接 VAU 遗漏；这不是“完全不存在”的结论。
