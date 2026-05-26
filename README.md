# Awesome Thinking with VAD

[![Awesome](https://awesome.re/badge.svg)](https://awesome.re)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

English | [简体中文](README.zh-CN.md)

> 🚧 **This repository is under active construction.** We're continuously adding new papers, refining categorizations, and expanding dataset coverage. Stay tuned for updates!

[![Interactive Atlas](https://img.shields.io/badge/View-Interactive_Research_Atlas-indigo?style=for-the-badge&logo=react)](https://2-mo.github.io/Awesome-Thinking-with-VAD/)

## 🗞️ Recent Updates

- **2026-05-26** — Updated the CVPR and ICML paper lists with the latest 2026 publications.
- **2026-02-06** — Refreshed the Interactive Atlas timeline page ([View Interactive Research Atlas](https://2-mo.github.io/Awesome-Thinking-with-VAD/)).

---

## 📖 Table of Contents

- [Awesome Thinking with VAD](#awesome-thinking-with-vad)
  - [🗞️ Recent Updates](#️-recent-updates)
  - [📖 Table of Contents](#-table-of-contents)
  - [🌟 Overview](#-overview)
  - [📚 Conference Snapshots](#-conference-snapshots)
  - [📰 Journal Snapshots](#-journal-snapshots)
  - [🧪 Benchmarks and Datasets](#-benchmarks-and-datasets)
  - [🔗 Related Resources](#-related-resources)
    - [Tutorials \& Workshops](#tutorials--workshops)
    - [Related Awesome Lists](#related-awesome-lists)
  - [🤝 Contributing](#-contributing)
  - [🤝 Stay Connected](#-stay-connected)
  - [📜 License and Credits](#-license-and-credits)

---

## 🌟 Overview

This repository is a curated collection of research papers and resources exploring **thoughtful reasoning approaches** in Video Anomaly Detection (VAD), with a special focus on **Large Language Models (LLMs)**, **Vision-Language Models (VLMs)**, and **Video Anomaly Understanding (VAU)**.

Video anomaly detection is evolving from simple frame-level alerts to systems that **reason, explain, and communicate** what makes something suspicious. This repository tracks that shift, focusing on methods that leverage **LLMs** and **VLMs** for deeper anomaly understanding.

**What's inside:**
- 📚 Conference & journal paper collections organized by venue and year
- 📊 Datasets categorized by LLM-readiness (explainable annotations vs. traditional labels)
- 🔗 Quick navigation to reasoning-centric VAD resources

**For:** researchers and practitioners exploring the intersection of anomaly detection, multimodal reasoning, and foundation models.

---

## 📚 Conference Snapshots

The `venues/` directory hosts per-conference notes for 2023-2026. Quick links:

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

## 📰 Journal Snapshots

See [journals/README.md](journals/README.md) for the latest top-tier journal snapshots, including:

- [TPAMI](journals/tpami.md) — IEEE Transactions on Pattern Analysis and Machine Intelligence
- [TIP](journals/tip.md) — IEEE Transactions on Image Processing
- [TNNLS](journals/tnnls.md) — IEEE Transactions on Neural Networks and Learning Systems
- [TCYB](journals/tcyb.md) — IEEE Transactions on Cybernetics
- [TIFS](journals/tifs.md) — IEEE Transactions on Information Forensics and Security
- [IJCV](journals/ijcv.md) — International Journal of Computer Vision (Springer)

---

## 🧪 Benchmarks and Datasets

We maintain a comprehensive catalog of VAD datasets in **[dataset.md](dataset.md)**, organized by:

- 🤖 **LLM/VLM-Ready Datasets** — Multimodal & explainable annotations
  - Video-language annotation (UCA, VAD-Instruct50k, UCCD)
  - Cross-modal retrieval (UCFCrime-AR, XDViolence-AR)
  - Open-world understanding (UBnormal)
  - Large-scale multimodal (XD-Violence)

- 🔧 **Traditional VAD Benchmarks** — Classic deep learning datasets
  - Weakly supervised (UCF-Crime, ShanghaiTech-W, TAD)
  - Semi-supervised (UCSD, Avenue, ShanghaiTech, NWPU Campus)
  - Fully supervised (Hockey Fight, RWF-2000, CCTV-Fights)

- 🚗 **Domain-Specific** — Driving, traffic, and specialized scenarios
  - Honda HDD, ROADWork, MSAD

👉 **[View full dataset catalog →](dataset.md)**

---

## 🔗 Related Resources

### Tutorials & Workshops

- [ICCV 2025 Tutorial: Foundation Models for Anomaly Detection](https://sites.google.com/view/iccv2025-tutorial-fm-driven-ad/home)

### Related Awesome Lists

- [![Awesome-Anomaly-Detection-Foundation-Models](https://img.shields.io/badge/Awesome-Anomaly_Detection_Foundation_Models-black?logo=github)](https://github.com/mala-lab/Awesome-Anomaly-Detection-Foundation-Models)
- [![Awesome-Video-Anomaly-Detection](https://img.shields.io/badge/Awesome-Video_Anomaly_Detection-black?logo=github)](https://github.com/fjchange/awesome-video-anomaly-detection)
- [![Deep-Learning-Based-Anomaly-Detection](https://img.shields.io/badge/Awesome-Deep_Learning_Anomaly_Detection-black?logo=github)](https://github.com/bitzhangcy/Deep-Learning-Based-Anomaly-Detection)
- [![Awesome-Temporal-Video-Grounding](https://img.shields.io/badge/Awesome-Temporal_Video_Grounding-black?logo=github)](https://github.com/Tangkfan/Awesome-Temporal-Video-Grounding)

---

## 🤝 Contributing

We welcome contributions! Please feel free to:

- Submit pull requests to add new papers, datasets, or resources
- Open issues for corrections or suggestions
- Share your own work related to thinking-based VAD

**Guidelines:**
- Follow the existing format for paper entries
- Include links to paper, code, and project pages when available
- Add a brief highlight describing the key contribution
- Place papers in the appropriate year and conference section
- If you're unsure where a paper belongs, open an issue and we'll help place it

**Entry template:**
```text
- Title — Venue, Year
- Links: paper | code | project
- Task/Setting: ...
- Highlight: ...
```

---

## 🤝 Stay Connected

<div align="center">
  <p>📧 Email: <strong>mo1031@live.com</strong></p>
  <p>📱 WeChat: <strong>tiumo-</strong> (please add note "VAD")</p>
</div>

---

## 📜 License and Credits

This collection is maintained as an open resource for the research community.

- Content is gathered from publicly available sources
- Paper copyrights belong to their respective authors and publishers
- This repository is for academic and educational purposes

**Maintainers**: Feel free to reach out for collaborations or suggestions!

---

**Star ⭐ this repo if you find it helpful!**
