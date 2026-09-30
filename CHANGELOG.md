# 更新记录 / Changelog

[论文年表](llm4vad.md) · [引用导出](literature/citations.md) · [研究地图](https://2-mo.github.io/Awesome-Thinking-with-VAD/)

按内容批次记录新增、纠错与版本处理。同日多个批次按完成顺序编号，最新批次在前；这里记录仓库内容变更，不代表已经发布 GitHub Release。

## 2026-09-30 · 批次 04 · 新论文与目录整理

### 新增

- ReactVAU：流式异常理解中的快慢解耦与持久记忆。
- Pistachio：可控合成视频与多事件异常理解评测。
- VANE-Bench：补录 NAACL Findings 2025 的异常问答基准。
- 同步完整作者、引用导出，以及 Pistachio、VANE-Bench 两个数据资源记录；当前共 54 篇论文、16 个数据资源记录。

### 调整与纠错

- 阅读索引与引用库归入 `literature/`，根目录保留 `llm4vad.md` 论文年表。
- 原 `venues/`、`journals/`、`dataset.md` 和配图归入 `archive/`；原发表索引改为 `literature/venues.md`。
- 维护与开发说明、设计记录归入 `maintenance/`；贡献说明放入 `.github/CONTRIBUTING.md`。
- 同步 README、生成器、网站入口与 CI 路径。旧路径的外部书签需要改用新位置。

### 版本处理

- ReactVAU、Pistachio 按作者声明标注 ECCV 2026 录用；正式书目未核验，BibTeX 仍明确引用 arXiv 版本，其中 Pistachio 预印本年份为 2025。
- VANE-Bench 使用正式版作者顺序、2025 年份与 DOI，不按早期 arXiv 版本另计一篇。
- 本批无版本合并。详见[核验记录](research/batch-04-2026-09.md)。

## 2026-09-30 · 批次 03 · 引用与读者入口

### 新增

- 补齐 51 篇论文的完整有序作者与 BibTeX：42 条正式版本、9 条预印本；核验 39 个 DOI（含 9 个 arXiv DOI），其余不补造。提供[逐篇引用](literature/citations.md)和整库 `references.bib`。
- 正式 MIT `LICENSE`，适用于仓库原创代码与文档；第三方论文、数据和图像仍遵循原条款。
- 本更新记录，集中展示每批新增、纠错与版本处理。

### 调整与纠错

- 中英文 README 面向读者保留文献阅读、引用、网站链接和论文推荐入口。
- 贡献说明简化为推荐论文与纠错；构建、依赖和仓库维护步骤移至维护者文档。
- 统一论文贡献标签为“创新”。
- 引用核对时修复 HAWK 数据记录的失效 NeurIPS 论文集链接。

### 版本处理

- 引用导出明确区分正式发表版本和预印本，不把 arXiv DOI 当作出版 DOI，不拼接不同版本的年份、作者与发表信息。
- 本批未新增或合并核心论文节点。[引用核验说明](research/citation-update-2026-09.md)。

## 2026-09-30 · 批次 02 · 比较与评测索引

### 新增

- 方法比较、数据集与评测、阅读路线三个 GitHub 阅读入口。
- CueBench、VAGU、VALU、A2Seek、TAU-Bench 五个文字数据记录，数据资源从 9 个增至 14 个。
- 异常理解评估阅读路线，阅读路线从 3 条增至 4 条。
- 论文推荐、事实纠错与 PR 模板。

### 调整与纠错

- 统一任务词表，将方法／场景标签单独保存。
- 补齐 VERA 公开代码链接，将编辑提问统一展示为“阅读关注”。
- 未核验的训练设置、未来帧访问与资源下载状态明确标注，不从模型冻结或项目页存在推断。

### 版本处理

- UCVL 出版身份待一手来源确认，暂未加入核心目录；论文总数保持 51 篇。
- [详细记录](research/index-update-2026-09.md)。

## 2026-09-30 · 批次 01 · 文献扩充

### 新增

- 补充 19 篇独立论文，核心目录从 32 篇增至 51 篇。
- [新增论文完整清单及来源](research/literature-update-2026-09.md#新增条目)。

### 调整与纠错

- 清理年表中的错误 ICCV 2024 分组、空链接与范围外条目。
- 修正 SlowFastVAD、Flashback 的机制描述，区分冻结主模型与辅助模块训练。
- 年表与发表索引纳入统一生成。

### 版本处理

- VAGU & GtS 采用 AAAI 2026 身份，2026 年 8 月扩展稿保留为扩展来源，未重复计数；扩展贡献不归入原稿。
- VAU-R1、Vad-R1、Vad-R1-Plus 保持为不同工作，不因名称相近合并。
- [核验与版本说明](research/literature-update-2026-09.md#版本和旧条目修正)。
