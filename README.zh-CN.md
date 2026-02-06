# Awesome Thinking with VAD

[![Awesome](https://awesome.re/badge.svg)](https://awesome.re)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

[English](README.md) | 简体中文

> 说明：英文版为主，中文版定期同步，可能略有滞后。

> 🚧 **本仓库仍在持续建设中**：我们会不断补充论文、优化分类、扩展数据集覆盖，欢迎持续关注。

[![Interactive Atlas](https://img.shields.io/badge/View-Interactive_Research_Atlas-indigo?style=for-the-badge&logo=react)](https://2-mo.github.io/Awesome-Thinking-with-VAD/)

## 🗞️ 最新更新

- **2026-02-06** — 更新了 AAAI 论文列表。
- **2026-02-06** — 更新了 ICLR 论文列表。
- **2026-02-06** — 刷新了 Interactive Atlas 时间轴页面（[查看研究图谱](https://2-mo.github.io/Awesome-Thinking-with-VAD/)）。

---

## 📖 目录

- [Awesome Thinking with VAD](#awesome-thinking-with-vad)
  - [🗞️ 最新更新](#-最新更新)
  - [📖 目录](#-目录)
  - [🌟 概述](#-概述)
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
- 📚 按会议与年份整理的论文合集
- 📊 按 LLM 适配程度分类的数据集（可解释标注 vs. 传统标签）
- 🔗 推理型 VAD 资源的快速入口

**面向：** 关注异常检测、多模态推理与基础模型交叉领域的研究者与实践者。

---

## 📚 会议概览

`venues/` 目录下汇总了 2023-2026 年各会议的论文笔记，快速入口如下：

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

详见 [journals/README.md](journals/README.md)，包括：

- [TPAMI](journals/tpami.md) — IEEE Transactions on Pattern Analysis and Machine Intelligence
- [TIP](journals/tip.md) — IEEE Transactions on Image Processing
- [TNNLS](journals/tnnls.md) — IEEE Transactions on Neural Networks and Learning Systems
- [TCYB](journals/tcyb.md) — IEEE Transactions on Cybernetics
- [TIFS](journals/tifs.md) — IEEE Transactions on Information Forensics and Security
- [IJCV](journals/ijcv.md) — International Journal of Computer Vision (Springer)

---

## 🧪 基准与数据集

我们在 **[dataset.md](dataset.md)** 中维护了完整的 VAD 数据集清单，按以下维度整理：

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

👉 **[查看完整数据集列表 →](dataset.md)**

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

欢迎贡献！你可以：

- 提交 PR：添加论文、数据集或资源
- 提交 Issue：反馈问题或提出建议
- 分享与“推理型 VAD”相关的工作

**建议：**
- 遵循既有论文条目格式
- 尽量附上论文、代码与项目页链接
- 添加简短亮点描述
- 归类到对应年份与会议分区
- 若不确定分类，可先提 Issue 讨论

**条目模板：**
```text
- 标题 — 会议, 年份
- 链接：论文 | 代码 | 项目页
- 任务/设置：...
- 亮点：...
```

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
