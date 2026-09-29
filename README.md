# Awesome Thinking with VAD

[![Awesome](https://awesome.re/badge.svg)](https://awesome.re)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

English | [简体中文](README.zh-CN.md)

> 🚧 **This repository is under active construction.** We're continuously adding new papers, refining categorizations, and expanding dataset coverage. Stay tuned for updates!

[![Interactive Atlas](https://img.shields.io/badge/View-Interactive_Research_Atlas-indigo?style=for-the-badge&logo=react)](https://2-mo.github.io/Awesome-Thinking-with-VAD/)

## 🗞️ Recent Updates

- **2026-09-29** — Rebuilt the research atlas around a selected, source-checked catalog: 37 core papers, 9 datasets, and 3 reading guides.

- **2026-05-26** — Updated the CVPR and ICML paper lists with the latest 2026 publications.
- **2026-02-06** — Refreshed the Interactive Atlas timeline page ([View Interactive Research Atlas](https://2-mo.github.io/Awesome-Thinking-with-VAD/)).

---

## 📖 Table of Contents

- [Awesome Thinking with VAD](#awesome-thinking-with-vad)
  - [🗞️ Recent Updates](#️-recent-updates)
  - [📖 Table of Contents](#-table-of-contents)
  - [🌟 Overview](#-overview)
  - [🗺️ Research Atlas](#-research-atlas)
  - [💻 Run Locally](#-run-locally)
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

- 🗺️ A compact research map organized around innovation ideas, with evidence-linked paper details, datasets and reading routes
- 📚 Conference & journal paper collections organized by venue and year
- 📊 Datasets categorized by LLM-readiness (explainable annotations vs. traditional labels)
- 🔗 Quick navigation to reasoning-centric VAD resources

**For:** researchers and practitioners exploring the intersection of anomaly detection, multimodal reasoning, and foundation models.

---

## 🗺️ Research Atlas

[Open the interactive atlas](https://2-mo.github.io/Awesome-Thinking-with-VAD/) · [Read the selected catalog](catalog.md) · [Inspect the source data](data/catalog.json)

The current selected catalog contains **37 core papers across 5 research directions, 9 datasets, and 3 reading guides**. The metro-style atlas shows all 37 papers as stations on five innovation lines. A vector schematic with consistent station markers and labeled lines keeps paper names, venues and years visible together; details open on selection. Lines group innovation ideas and research questions; dataset coverage supplies evaluation context.

The catalog covers video anomaly detection and understanding only. Map lines and reading routes are editorial groupings, not citation edges or claims of research inheritance. Evidence-backed relationships remain available in each paper’s detail pane.

[data/catalog.json](data/catalog.json) is the single source of truth for the interactive atlas and generated [catalog.md](catalog.md). Selected entries record verification sources and dates. Dataset cards include original author/paper figures, publication year and venue, with image attribution. The expansion adds 20 verified conference papers; NeurIPS Datasets and Benchmarks is labeled explicitly. Historical workshops and unconfirmed preprints retain their original status; generic video understanding and ordinary detection papers are not added merely to increase coverage. The historical [conference notes](venues/), [journal notes](journals/), [dataset notes](dataset.md) and [LLM/VAD notes](llm4vad.md) remain additional resources and **have not all been reverified**; inclusion there does not imply inclusion in the checked catalog.

## 💻 Run Locally

Use Node.js 24 for development:

```sh
npm ci
npm run dev
```

Run `npm test` for the automated tests, `npm run check` for validation and type checking, `npm run build` to regenerate the catalog and production site, and `npm run preview` to inspect the production build. Use `npm install` when deliberately updating dependencies.

The Vite output is committed in [docs/](docs/). On `main`, you can serve it directly without installing Node dependencies:

```sh
python3 -m http.server 8000 --directory docs
```

Open `http://localhost:8000/`. See [DEVELOPMENT.md](DEVELOPMENT.md) for the development workflow and [CONTRIBUTING.md](CONTRIBUTING.md) for data and source requirements. Edit source data and application code, then regenerate; do not hand-edit `catalog.md` or `docs/`.

---

## 📚 Conference Snapshots

The `venues/` directory preserves historical per-conference notes for 2023–2026, which have not all been reverified. Quick links:

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

See [journals/README.md](journals/README.md) for historical journal notes, which have not all been reverified, including:

- [TPAMI](journals/tpami.md) — IEEE Transactions on Pattern Analysis and Machine Intelligence
- [TIP](journals/tip.md) — IEEE Transactions on Image Processing
- [TNNLS](journals/tnnls.md) — IEEE Transactions on Neural Networks and Learning Systems
- [TCYB](journals/tcyb.md) — IEEE Transactions on Cybernetics
- [TIFS](journals/tifs.md) — IEEE Transactions on Information Forensics and Security
- [IJCV](journals/ijcv.md) — International Journal of Computer Vision (Springer)

---

## 🧪 Benchmarks and Datasets

The atlas provides 9 selected datasets with task, annotation and evaluation-protocol notes. The broader historical **[dataset.md](dataset.md)** remains available but has not been fully reverified; it is organized by:

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

👉 **[View historical dataset notes →](dataset.md)**

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

Please read [CONTRIBUTING.md](CONTRIBUTING.md) before adding or correcting atlas entries. Edit [data/catalog.json](data/catalog.json), cite primary sources with evidence notes, record verification dates, and include explicit limitations. Keep entries directly relevant to video anomaly understanding.

Run `npm run build` and `npm run check`, then include the generated `catalog.md` and `docs/` output in your pull request. CI checks that these outputs match the source. For corrections to historical notes, edit the relevant Markdown file and make the scope of verification clear. Issues with corrections, source evidence and suggested additions are welcome.

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
