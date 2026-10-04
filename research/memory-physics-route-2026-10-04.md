# 记忆换乘与物理理解回接 · 2026-10-04

- **PANDA**：[正文 §3.3–3.4、Figure 3](https://arxiv.org/html/2509.26386v2#S3.SS4) 区分局部视觉／文本上下文的 Short CoM 与累积历史推理、反思及修正结果的 Long CoM，并检索历史反思辅助工具使用。保留主动取证主归属，补充时序记忆归属，显示为双站台换乘站。
- **CUVA → MMAD → Phys-AD → HoloTrace**：早期评测阅读段在 HoloTrace 回接结构化推理主线。CUVA 是普通分叉点，HoloTrace 是普通回接点；两处保留有来源的双重归属，但不显示换乘标记。
- **HoloTrace**：[正式论文全文](https://smartinternet.group/wp-content/uploads/2025/12/paper-whl-HoloTrace-mm.pdf) 以双向因果知识图组织事件推理，并贡献包含 632 段视频、10 类异常与逐帧标注的 SVAD 数据集。§5.1 使用 SVAD 与 UCSD PED2，未使用 Phys-AD。回接表示从物理原因解释问题到显式因果事件推理的阅读关联，不代表直接继承；未新增二者的 `uses` 或 `extends` 关系。
- **O-VAD**：[正文 §4.1 与附录 C.5](https://arxiv.org/html/2607.18142v1) 明确使用 Phys-AD 官方划分，补充对象轨迹标注，并进行异常报告评价。专家与 LLM 报告评价按全文记为同一组 10 段视频；未采用项目页中 LLM 覆盖整个测试集的说法。新增 `o-vad-uses-phys-ad-data`，保留 O-VAD 原有地图位置。

本批次未新增论文，ICASSP 工作不收录；目录 88 篇、地图 85 站、数据资源 33 项。期刊浅底色、arXiv 灰色斜体和原图异步加载保留。

验证：`npm run check` 与 `npm run build` 通过，共 116 项测试；浏览器核对 85 个站点、四篇相关论文详情及原图懒加载、手机弹窗和开发预览。SVG/PNG 已重新导出并目视检查，浏览器无运行错误。
