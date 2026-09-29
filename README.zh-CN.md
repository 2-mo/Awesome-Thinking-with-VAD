# Awesome Thinking with VAD

[![Awesome](https://awesome.re/badge.svg)](https://awesome.re)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

[English](README.md) | 简体中文

> 说明：英文版为主，中文版定期同步，可能略有滞后。

> 🚧 **本仓库仍在持续建设中**：我们会不断补充论文、优化分类、扩展数据集覆盖，欢迎持续关注。

[![Interactive Atlas](https://img.shields.io/badge/View-Interactive_Research_Atlas-indigo?style=for-the-badge&logo=react)](https://2-mo.github.io/Awesome-Thinking-with-VAD/)

## 🗞️ 最新更新

- **2026-09-29** — 研究地图新版上线：以来源核验的精选目录为基础，包含 37 篇核心论文、9 个数据集和 3 条阅读路线。

- **2026-02-06** — 更新了 AAAI 论文列表。
- **2026-02-06** — 更新了 ICLR 论文列表。
- **2026-02-06** — 刷新了 Interactive Atlas 时间轴页面（[查看研究图谱](https://2-mo.github.io/Awesome-Thinking-with-VAD/)）。

---

## 📖 目录

- [Awesome Thinking with VAD](#awesome-thinking-with-vad)
  - [🗞️ 最新更新](#-最新更新)
  - [📖 目录](#-目录)
  - [🌟 概述](#-概述)
  - [🗺️ 研究地图](#-研究地图)
  - [💻 本地运行](#-本地运行)
  - [📚 会议概览](#-会议概览)
  - [📰 期刊概览](#-期刊概览)
  - [🧪 基准与数据集](#-基准与数据集)
  - [🔗 相关资源](#-相关资源)
    - [教程与工作坊](#教程与工作坊)
    - [相关 Awesome 列表](#相关-awesome-列表)
  - [🤝 贡献](#-贡献)
  - [🤝 联系方式](#-联系方式)
  - [📜 许可与致谢](#-许可与致谢)

---

## 🌟 概述

本仓库是一份聚焦于**思维化推理**的视频异常检测（VAD）论文与资源精选集，特别关注**大语言模型（LLMs）**、**视觉语言模型（VLMs）**与**视频异常理解（VAU）**带来的新范式。

视频异常检测正在从简单的帧级告警转向能够**推理、解释与表达**异常原因的系统。本仓库聚焦于利用**LLMs**与**VLMs**实现更深层异常理解的方法。

**内容包括：**

- 🗺️ 按创新思路组织、紧凑同屏呈现研究方向、论文依据、数据集与阅读路线的交互式研究地图
- 📚 按会议与年份整理的论文合集
- 📊 按 LLM 适配程度分类的数据集（可解释标注 vs. 传统标签）
- 🔗 推理型 VAD 资源的快速入口

**面向：** 关注异常检测、多模态推理与基础模型交叉领域的研究者与实践者。

---

## 🗺️ 研究地图

[打开交互式研究地图](https://2-mo.github.io/Awesome-Thinking-with-VAD/) · [阅读精选目录](catalog.md) · [查看源数据](data/catalog.json)

当前精选目录包含 **5 个研究方向的 32 篇核心论文、9 个数据集和 3 条阅读路线**。研究线路图以年份为横轴、会议或发表场所为纵轴，全部 32 篇论文作为站点同屏呈现，五条彩色方法派别线路以水平、竖直和 45° 斜线连接站点。地图与发表出处筛选将 NeurIPS 及其 Datasets and Benchmarks 分轨合并为一个 NeurIPS 行／类别，共显示 11 个发表场所；论文详情保留准确分轨。图中保留论文简称，坐标轴和详情保留发表出处与年份；同一年内的横向间距仅用于排版，不表示具体发表日期。数据集补充评测背景。

目录仅收录视频异常检测与理解相关工作。地图线路与阅读路线均为编辑整理的分组，不表示论文间的引用或继承关系。没有论文节点的线路交叉不表示方法融合或换乘；有来源依据的关系保留在论文详情中。

[data/catalog.json](data/catalog.json) 是交互式地图与自动生成的 [catalog.md](catalog.md) 的单一数据源，精选条目记录核验来源与日期。数据集以每页 9 项的九宫格展览呈现，展示作者或论文原图、发表年份和会议，并保留图像来源署名。当前精选论文目录不纳入 WACV 和工作坊论文。源数据与论文详情仍明确标注 NeurIPS Datasets and Benchmarks 分轨；未确认正式发表的预印本保留 arXiv 状态，不为数量补入普通视频理解或纯检测工作。既有[会议笔记](venues/)、[期刊笔记](journals/)、[数据集笔记](dataset.md)与 [LLM/VAD 笔记](llm4vad.md) 作为额外历史资料保留，**尚未全部复核**；出现在历史笔记中不等于被纳入本次核验目录。

## 💻 本地运行

开发环境使用 Node.js 24：

```sh
npm ci
npm run dev
```

`npm test` 运行自动测试，`npm run check` 执行数据校验与类型检查，`npm run build` 重新生成目录和生产站点，`npm run preview` 预览生产构建。需要主动更新依赖时使用 `npm install`。

Vite 生产输出已提交至 [docs/](docs/)。在 `main` 上无需安装 Node 依赖即可直接启动静态站点：

```sh
python3 -m http.server 8000 --directory docs
```

打开 `http://localhost:8000/`。完整开发流程见 [DEVELOPMENT.md](DEVELOPMENT.md)，数据维护与来源要求见 [CONTRIBUTING.md](CONTRIBUTING.md)。请修改源数据或应用代码后重新生成，不要手工修改 `catalog.md` 或 `docs/`。

---

## 📚 会议概览

`venues/` 保留了 2023–2026 年各会议的历史论文笔记，尚未全部复核，快速入口如下：

- [CVPR](venues/cvpr.md) — Computer Vision and Pattern Recognition
- [ICCV](venues/iccv.md) — International Conference on Computer Vision
- [ECCV](venues/eccv.md) — European Conference on Computer Vision
- [NeurIPS](venues/neurips.md) — Neural Information Processing Systems
- [ICML](venues/icml.md) — International Conference on Machine Learning
- [ICLR](venues/iclr.md) — International Conference on Learning Representations
- [AAAI](venues/aaai.md) — Association for the Advancement of Artificial Intelligence
- [IJCAI](venues/ijcai.md) — International Joint Conference on Artificial Intelligence
- [ACM MM](venues/acmmm.md) — ACM Multimedia

---

## 📰 期刊概览

历史期刊笔记详见 [journals/README.md](journals/README.md)，这些条目尚未全部复核，包括：

- [TPAMI](journals/tpami.md) — IEEE Transactions on Pattern Analysis and Machine Intelligence
- [TIP](journals/tip.md) — IEEE Transactions on Image Processing
- [TNNLS](journals/tnnls.md) — IEEE Transactions on Neural Networks and Learning Systems
- [TCYB](journals/tcyb.md) — IEEE Transactions on Cybernetics
- [TIFS](journals/tifs.md) — IEEE Transactions on Information Forensics and Security
- [IJCV](journals/ijcv.md) — International Journal of Computer Vision (Springer)

---

## 🧪 基准与数据集

研究地图提供 9 个精选数据集的任务、标注与评测协议说明。覆盖范围更广的历史 **[dataset.md](dataset.md)** 仍保留，尚未全面复核，按以下维度整理：

- 🤖 **LLM/VLM 友好型数据集** — 多模态与可解释标注
  - 视频语言标注（UCA, VAD-Instruct50k, UCCD）
  - 跨模态检索（UCFCrime-AR, XDViolence-AR）
  - 开放世界理解（UBnormal）
  - 大规模多模态（XD-Violence）

- 🔧 **传统 VAD 基准** — 经典深度学习数据集
  - 弱监督（UCF-Crime, ShanghaiTech-W, TAD）
  - 半监督（UCSD, Avenue, ShanghaiTech, NWPU Campus）
  - 全监督（Hockey Fight, RWF-2000, CCTV-Fights）

- 🚗 **领域专用** — 驾驶、交通与特定场景
  - Honda HDD, ROADWork, MSAD

👉 **[查看历史数据集笔记 →](dataset.md)**

---

## 🔗 相关资源

### 教程与工作坊

- [ICCV 2025 Tutorial: Foundation Models for Anomaly Detection](https://sites.google.com/view/iccv2025-tutorial-fm-driven-ad/home)

### 相关 Awesome 列表

- [![Awesome-Anomaly-Detection-Foundation-Models](https://img.shields.io/badge/Awesome-Anomaly_Detection_Foundation_Models-black?logo=github)](https://github.com/mala-lab/Awesome-Anomaly-Detection-Foundation-Models)
- [![Awesome-Video-Anomaly-Detection](https://img.shields.io/badge/Awesome-Video_Anomaly_Detection-black?logo=github)](https://github.com/fjchange/awesome-video-anomaly-detection)
- [![Deep-Learning-Based-Anomaly-Detection](https://img.shields.io/badge/Awesome-Deep_Learning_Anomaly_Detection-black?logo=github)](https://github.com/bitzhangcy/Deep-Learning-Based-Anomaly-Detection)
- [![Awesome-Temporal-Video-Grounding](https://img.shields.io/badge/Awesome-Temporal_Video_Grounding-black?logo=github)](https://github.com/Tangkfan/Awesome-Temporal-Video-Grounding)

---

## 🤝 贡献

添加或修正地图条目前，请阅读 [CONTRIBUTING.md](CONTRIBUTING.md)。编辑 [data/catalog.json](data/catalog.json)，补充一手来源、证据说明、核验日期和明确的局限，确保条目直接关联视频异常理解。

运行 `npm run build` 和 `npm run check`，在 PR 中同时提交生成的 `catalog.md` 与 `docs/`；CI 会检查生成结果是否与源数据一致。修正历史笔记时，请编辑对应 Markdown 文件并说明核验范围。欢迎通过 Issue 提交勘误、来源证据和新增建议。

---

## 🤝 联系方式

<div align="center">
  <p>📧 邮箱：<strong>mo1031@live.com</strong></p>
  <p>📱 微信：<strong>tiumo-</strong>（备注“VAD”，方便识别）</p>
</div>

---

## 📜 许可与致谢

本合集面向学术社区开放维护：

- 内容来自公开可用的资料
- 论文版权归原作者与出版方所有
- 本仓库用于学术与教育目的

**Maintainers**：欢迎交流与合作！

---

**如果你觉得有帮助，欢迎点个 Star！**
