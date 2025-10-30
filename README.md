# Awesome Thinking with VAD 🧠🎥

[![Awesome](https://awesome.re/badge.svg)](https://awesome.re)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> A curated collection of research papers and resources exploring **thoughtful reasoning approaches** in Video Anomaly Detection (VAD), with special focus on Large Language Models (LLMs) and Vision-Language Models (VLMs).

## 📖 Table of Contents

- [Awesome Thinking with VAD 🧠🎥](#awesome-thinking-with-vad-)
  - [📖 Table of Contents](#-table-of-contents)
  - [🤝 Stay Connected](#-stay-connected)
  - [🌟 Overview](#-overview)
  - [🧭 How to Use This Repository](#-how-to-use-this-repository)
    - [📂 Repository Map](#-repository-map)
    - [📝 Reading Workflow](#-reading-workflow)
  - [📚 Conference Snapshots](#-conference-snapshots)
  - [📰 Journal Snapshots](#-journal-snapshots)
  - [📊 Benchmarks and Datasets](#-benchmarks-and-datasets)
    - [🤖 LLM/VLM-Ready Datasets (Multimodal \& Explainable)](#-llmvlm-ready-datasets-multimodal--explainable)
      - [📝 Video-Language Annotation](#-video-language-annotation)
      - [🔍 Anomaly Retrieval (Cross-modal)](#-anomaly-retrieval-cross-modal)
      - [🌐 Open-World Understanding](#-open-world-understanding)
      - [🎬 Large-Scale Multimodal](#-large-scale-multimodal)
    - [🔧 Traditional VAD Benchmarks](#-traditional-vad-benchmarks)
      - [1️⃣ Weakly Supervised](#1️⃣-weakly-supervised)
      - [2️⃣ Semi-supervised](#2️⃣-semi-supervised)
      - [3️⃣ Fully Supervised](#3️⃣-fully-supervised)
    - [🚗 Domain-Specific Datasets](#-domain-specific-datasets)
      - [Driving \& Transportation](#driving--transportation)
      - [Multi-Scenario](#multi-scenario)
  - [🔗 Related Resources](#-related-resources)
    - [Related Awesome Lists](#related-awesome-lists)
  - [🤝 Contributing](#-contributing)
  - [📜 License and Credits](#-license-and-credits)

---

## 🤝 Stay Connected

> 扫码加入小红书「视频异常检测」交流圈，分享论文、工作与心得体会。

<div align="center">
  <img src="assets/qrcode/redbook[25.11.25].JPG" alt="Thinking with VAD Xiaohongshu QR code" width="220">
  <p><em>[RedBook] until 2025-Nov-25</em></p>
</div>

---

## 🌟 Overview

Modern video anomaly detection is moving beyond frame-level alarms toward systems that **interpret, justify, and communicate** why something looks suspicious. This list mirrors that shift by collecting papers, datasets, and tooling that emphasize reasoning-heavy VAD pipelines—especially those powered by large language models (LLMs) and vision-language models (VLMs).

**What you can expect:**
- Curated reading paths that show how “slow thinking” modules (reasoning, explanation, planning) complement classic perception backbones
- Venue and journal snapshots that surface where thinking-centric VAD work is appearing and how the discussion is evolving
- Dataset groupings that make it straightforward to pick between LLM-ready benchmarks and traditional baselines when scoping a project or reproduction study

**Who this is for:** researchers, students, and practitioners who want a single hub for tracking the convergence of anomaly detection, multimodal understanding, and foundation models.

## 🧭 How to Use This Repository

Treat this repo as a launchpad. Start with the overview for a quick mental model, then dive into venue or journal notes, or jump straight to the dataset taxonomy depending on whether you need new data, baselines, or multimodal annotations.

### 📂 Repository Map
- `venues/` — year-by-year highlights from major conferences with takeaways and trend notes
- `journals/` — rolling coverage of influential journal publications grouped by outlet for faster literature sweeps
- `assets/` — figures and badges used across the documentation if you want to reuse the styling in decks or reports

### 📝 Reading Workflow
1. Skim the **LLM/VLM-ready datasets** when prototyping reasoning-enabled pipelines; drop to the traditional benchmarks for baselines or model comparisons.
2. Jump into the linked venue or journal page when you need richer context such as paper clusters, methodological trends, or open questions.
3. Open an issue or PR whenever you spot a missing reference—reasoning-centric resources ship rapidly, so community updates keep the list fresh.

---

## 📚 Conference Snapshots

The `venues/` directory hosts per-conference notes for 2023-2025. Quick links:

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

The `journals/` directory hosts per-journal notes for top-tier academic journals. Quick links:

- [TPAMI](journals/tpami.md) — IEEE Transactions on Pattern Analysis and Machine Intelligence
- [TIP](journals/tip.md) — IEEE Transactions on Image Processing
- [TNNLS](journals/tnnls.md) — IEEE Transactions on Neural Networks and Learning Systems
- [TCYB](journals/tcyb.md) — IEEE Transactions on Cybernetics
- [TIFS](journals/tifs.md) — IEEE Transactions on Information Forensics and Security
- [IJCV](journals/ijcv.md) — International Journal of Computer Vision (Springer)

---

## 📊 Benchmarks and Datasets

> 💡 **Trend**: Datasets are evolving from pure detection (traditional) to **understanding + explanation** (LLM-ready), aligning with the shift from "fast perception" to "slow thinking" in anomaly detection.


### 🤖 LLM/VLM-Ready Datasets (Multimodal & Explainable)

Datasets designed for or compatible with large language models and vision-language models, emphasizing reasoning, explanation, and multimodal understanding.

#### 📝 Video-Language Annotation
- **[UCA (UCF-Crime Annotation)](https://xuange923.github.io/Surveillance-Video-Understanding)** (CVPR 2024)
  - 23,542 fine-grained sentences, 111 hours
  - Temporal event descriptions for surveillance video understanding
  
- **[VAD-Instruct50k](https://holmesvad.github.io/)** (Holmes-VAD, arXiv 2024)
  - 51,567 multimodal instructions for explainable VAD
  - Rich textual explanations for anomaly reasoning
  
- **[UCCD](https://github.com/lingruzhou/UCCD)** (TMM 2024)
  - Human-centric behavior descriptions
  - Instance-level annotations with temporal info

#### 🔍 Anomaly Retrieval (Cross-modal)
- **[UCFCrime-AR](https://github.com/Roc-Ng/VAR)** (TIP 2024)
  - Video-text retrieval benchmark
  - 1,900 videos with Chinese & English descriptions
  
- **[XDViolence-AR](https://github.com/Roc-Ng/VAR)** (TIP 2024)
  - Audio-visual anomaly retrieval
  - 4,754 videos, cross-modal (video ↔ audio)

#### 🌐 Open-World Understanding
- **[UBnormal](https://github.com/lilygeorgescu/UBnormal)** (CVPR 2022)
  - Open-set benchmark: 22 training anomaly types, disjoint from test
  - 543 videos with pixel-level annotations
  - Simulates real-world unseen anomaly scenarios

#### 🎬 Large-Scale Multimodal
- **[XD-Violence](https://roc-ng.github.io/XD-Violence/)** (ECCV 2020)
  - 4,754 videos, 217 hours, 6 violence types
  - **Audio-visual** modality (speech, sound effects)
  - Suitable for multimodal LLM/VLM approaches

---

### 🔧 Traditional VAD Benchmarks

Classic datasets used for traditional deep learning and rule-based methods, organized by supervision type.

#### 1️⃣ Weakly Supervised
*Video-level labels only, no frame-level annotations.*

- **[UCF-Crime](https://www.crcv.ucf.edu/projects/real-world/)** (CVPR 2018) — 1,900 videos, 128 hours, 13 anomaly types
- **[ShanghaiTech Weakly](https://github.com/jx-zhong-for-academic-purpose/GCN-Anomaly-Detection)** (CVPR 2019) — Reorganized for weakly supervised setting
- **[TAD](https://github.com/ktr-hubrt/WSAL)** (TIP 2021) — 500 traffic videos, 7 road anomaly types

#### 2️⃣ Semi-supervised
*Train on normal videos only, detect anomalies at test time.*

**Classic Benchmarks:**
- **[UCSD Ped1 & Ped2](http://www.svcl.ucsd.edu/projects/anomaly/dataset.htm)** (CVPR 2010) — Pedestrian anomalies
- **[CUHK Avenue](https://www.cse.cuhk.edu.hk/leojia/projects/detectabnormal/dataset.html)** (ICCV 2013) — 37 sequences, running/throwing
- **[ShanghaiTech](https://svip-lab.github.io/dataset/campus_dataset.html)** (ICCV 2017) — 437 videos, 13 scenes, campus surveillance
- **[UMN](https://www.crcv.ucf.edu/research/projects/abnormal-crowd-behavior-detection-using-social-force-model/)** (CVPR 2009) — Escape events

**Recent Large-scale:**
- **[NWPU Campus](https://campusvad.github.io/)** (CVPR 2023) — **Largest**: 547 videos, 16 hours, 43 scenes, 28 classes
  - Scene-dependent anomalies
  - Anomaly anticipation task
- **[Street Scene](https://www.merl.com/research/highlights/video-anomaly-detection)** (WACV 2020) — 205 anomalies, 17 types
- **[IITB-Corridor](https://rodrigues-royston.github.io/Multi-timescale_Trajectory_Prediction/)** (WACV 2020) — Group-level anomalies

**Others:**
- **Subway Entrance & Exit** (TPAMI 2008) — Early surveillance benchmarks

#### 3️⃣ Fully Supervised
*Both normal and abnormal videos for training.*

- **[Hockey Fight & Movies Fight](https://academictorrents.com/details/70e0794e2292fc051a13f05ea6f5b6c16f3d3635)** (CAIP 2011) — 1,000 + 200 clips
- **[Violent-Flows](https://www.openu.ac.il/home/hassner/data/violentflows/)** (CVPR Workshops 2012) — Violent crowd behavior
- **[RWF-2000](https://github.com/mchengny/RWF2000-Video-Database-for-Violence-Detection)** (ICPR 2020) — 2,000 videos
- **[CCTV-Fights](https://rose1.ntu.edu.sg/dataset/cctvFights/)** (ICASSP 2019) — 1,000 real-world fights
- **[VFD-2000](https://github.com/Hepta-Col/VideoFightDetection)** (ICTAI 2022) — Multi-scenario, various lengths
- **[VSD](https://www.interdigital.com/data_sets/violent-scenes-dataset)** (MTA 2015) — 18 Hollywood movies

---

### 🚗 Domain-Specific Datasets

#### Driving & Transportation
- **[Honda HDD](https://usa.honda-ri.com/hdd#Videos)** — Driving anomaly detection
- **[ROADWork](https://www.cs.cmu.edu/~roadwork/)** (ICCV 2025) — Work zone recognition
- **[Driver Anomaly Detection](https://github.com/okankop/Driver-Anomaly-Detection)** — Driver behavior analysis
- **[TAD](https://github.com/ktr-hubrt/WSAL)** (TIP 2021) — Traffic anomaly detection

#### Multi-Scenario
- **[MSAD](https://msad-dataset.github.io/)** (NeurIPS 2024) [![arXiv](https://img.shields.io/badge/arXiv-2402.04857-b31b1b)](https://arxiv.org/pdf/2402.04857) — Multi-scenario, large-scale







---

## 🔗 Related Resources

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

---

## 📜 License and Credits

This collection is maintained as an open resource for the research community. 

- Content is gathered from publicly available sources
- Paper copyrights belong to their respective authors and publishers
- This repository is for academic and educational purposes

**Maintainers**: Feel free to reach out for collaborations or suggestions!

---

**Star ⭐ this repo if you find it helpful!**