# Awesome Thinking with VAD

[![Awesome](https://awesome.re/badge.svg)](https://awesome.re)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

[English](README.md) | 简体中文

聚焦**视频异常理解**：异常发生了什么、为什么发生、证据在哪里，以及如何验证解释与视频相符。兼顾直接支撑这些目标的语言引导表征与异常检测方法。

[**研究地图**](https://2-mo.github.io/Awesome-Thinking-with-VAD/) · [**按年份阅读**](llm4vad.md) · [**按创新思路阅读**](catalog.md) · [**按会议查找**](venues/README.md)

## 最新更新

**2026-09-30** — 补充 19 篇经来源核验的论文，精选目录共 **51 篇论文**，另有 9 个数据集展项和 3 条阅读路线。覆盖 NeurIPS 2025 遗漏、AAAI／ICLR／CVPR／ICML／ECCV 2026 正式论文和近期 VAU 预印本。具体新增与版本关系见[本轮更新记录](research/literature-update-2026-09.md)。

`llm4vad.md` 已改为与网页共用数据的年份索引，清理旧列表中的空链接、版本重复、错误会议分组和无关条目。

## 阅读入口

| 入口 | 内容 |
| --- | --- |
| [交互式地图](https://2-mo.github.io/Awesome-Thinking-with-VAD/) | 五条方法线路上的论文站点，可查询发表信息、贡献与来源 |
| [llm4vad.md](llm4vad.md) | 按年份与会议整理的紧凑论文表，预印本单独标注 |
| [catalog.md](catalog.md) | 按创新思路组织的贡献、阅读关注和核验来源 |
| [发表索引](venues/README.md) | 当前各会议的篇数，以及对应年份的入口 |
| [数据维护说明](data/README.md) | 收录范围、来源要求、发表状态与版本处理 |

地图保留**时间横轴，纵向按方法拓扑自由排布**；会议简称随论文名显示。布局根据共享论文安排线路邻接关系，对齐直线站段、采用统一小圆角，并避让标签、减少弯折和交叉。有来源依据的多方法论文采用并排站点与短连接符，名称只显示一次；普通交叉不相连。月份用于内部排序，2025 年起显示 Q1–Q4；筛选不改变全目录布局。颜色表示方法派别，不表示引用或继承。NeurIPS 及其 Datasets and Benchmarks 分轨合并显示，记录与详情保留准确分轨。

五个方向为：语义对齐与融合、语言判据与提示优化、时序分层与记忆、主动观察与工具决策、结构化推理与验证。数据集展览使用作者项目或论文原图，并保留署名。

## 收录与核验

[data/catalog.json](data/catalog.json) 是网页和三个自动生成阅读索引的单一数据源。每篇记录题名、发表身份、贡献依据和核验日期。正式发表信息优先于早期预印本信息；未确认录用的论文保留 **arXiv**。同一论文的扩展版本不会仅因题名变化而重复计数。

当前核心目录不纳入 WACV、workshop、泛视频理解或静态图像缺陷检测。纯检测方法仅在直接支撑异常语义时选择性收录。作者报告的性能不等于本站复现结论；仅有占位说明的项目入口不标为已发布代码。

旧[会议笔记](venues/)、[期刊笔记](journals/README.md)与[广义数据集笔记](dataset.md)保留为范围更广的历史资料，尚未全部复核；当前精选内容以上方生成索引为准。

## 本地运行

开发使用 Node.js 24：

```sh
npm ci
npm run dev
```

`npm run build` 校验数据，生成 `catalog.md`、`llm4vad.md`、`venues/README.md` 并构建静态网页。`npm run check` 检查数据、生成文档一致性、测试与 TypeScript。

生产网页已提交至 [docs/](docs/)，`main` 无需 Node 构建即可运行：

```sh
python3 -m http.server 8000 --directory docs
```

打开 `http://localhost:8000/`。完整维护流程见 [DEVELOPMENT.md](DEVELOPMENT.md)。

## 贡献

请修改[结构化目录](data/catalog.json)，记录一手来源，并遵循 [CONTRIBUTING.md](CONTRIBUTING.md)。源数据、生成文档与 `docs/` 一起提交；CI 会拒绝过期的生成文件。使用开发分支，通过检查后合并回 main。

欢迎补充题名、录用状态、版本关系的纠正及遗漏的高相关论文。

## 联系与署名

- 邮箱：**mo1031@live.com**
- 微信：**tiumo-**（请备注 VAD）

论文与数据集版权归作者和出版方所有。本仓库用于学术研究，图像保留原始来源署名。
