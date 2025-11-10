# 📊 Video Anomaly Detection Datasets

> 💡 **Trend**: Datasets are evolving from pure detection (traditional) to **understanding + explanation** (LLM-ready), aligning with the shift from "fast perception" to "slow thinking" in anomaly detection.

## 📖 Table of Contents

- [📊 Video Anomaly Detection Datasets](#-video-anomaly-detection-datasets)
  - [📖 Table of Contents](#-table-of-contents)
  - [🎯 Anomaly Detection](#-anomaly-detection)
    - [Weakly Supervised](#weakly-supervised)
    - [Semi-supervised](#semi-supervised)
    - [Fully Supervised](#fully-supervised)
  - [📍 Anomaly Localization](#-anomaly-localization)
    - [Spatial Localization](#spatial-localization)
    - [Temporal Localization](#temporal-localization)
  - [💡 Anomaly Understanding](#-anomaly-understanding)
    - [Video-Language Annotation](#video-language-annotation)
    - [Explainable \& Reasoning](#explainable--reasoning)
    - [Open-World Understanding](#open-world-understanding)
  - [🔍 Anomaly Retrieval](#-anomaly-retrieval)
    - [Video-Text Retrieval](#video-text-retrieval)
    - [Audio-Visual Retrieval](#audio-visual-retrieval)
  - [🎬 Anomaly Generation](#-anomaly-generation)
  - [🚗 Domain-Specific Datasets](#-domain-specific-datasets)
    - [Driving \& Transportation](#driving--transportation)
    - [Multi-Scenario](#multi-scenario)
  - [📊 Quick Reference Table](#-quick-reference-table)

---

## 🎯 Anomaly Detection

*Datasets focused on binary classification: identifying whether anomalies exist in videos.*

### Weakly Supervised
*Video-level labels only, no frame-level annotations.*

- **[UCF-Crime](https://www.crcv.ucf.edu/projects/real-world/)** (CVPR 2018)  
  - 1,900 videos, 128 hours, 13 anomaly types
  - Real-world surveillance scenarios
  
- **[ShanghaiTech Weakly](https://github.com/jx-zhong-for-academic-purpose/GCN-Anomaly-Detection)** (CVPR 2019)  
  - Reorganized for weakly supervised setting
  - Campus surveillance scenes
  
- **[TAD](https://github.com/ktr-hubrt/WSAL)** (TIP 2021)  
  - 500 traffic videos, 7 road anomaly types
  - Traffic surveillance scenarios

### Semi-supervised
*Train on normal videos only, detect anomalies at test time.*

**Classic Benchmarks:**
- **[UCSD Ped1 & Ped2](http://www.svcl.ucsd.edu/projects/anomaly/dataset.htm)** (CVPR 2010) — Pedestrian anomalies
- **[CUHK Avenue](https://www.cse.cuhk.edu.hk/leojia/projects/detectabnormal/dataset.html)** (ICCV 2013) — 37 sequences, running/throwing
- **[ShanghaiTech](https://svip-lab.github.io/dataset/campus_dataset.html)** (ICCV 2017) — 437 videos, 13 scenes, campus surveillance
- **[UMN](https://www.crcv.ucf.edu/research/projects/abnormal-crowd-behavior-detection-using-social-force-model/)** (CVPR 2009) — Escape events

**Recent Large-scale:**
- **[NWPU Campus](https://campusvad.github.io/)** (CVPR 2023)  
  - **Largest**: 547 videos, 16 hours, 43 scenes, 28 classes
  - Scene-dependent anomalies
  - Anomaly anticipation task
  
- **[Street Scene](https://www.merl.com/research/highlights/video-anomaly-detection)** (WACV 2020)  
  - 205 anomalies, 17 types
  
- **[IITB-Corridor](https://rodrigues-royston.github.io/Multi-timescale_Trajectory_Prediction/)** (WACV 2020)  
  - Group-level anomalies

**Others:**
- **Subway Entrance & Exit** (TPAMI 2008) — Early surveillance benchmarks

### Fully Supervised
*Both normal and abnormal videos for training.*

- **[Hockey Fight & Movies Fight](https://academictorrents.com/details/70e0794e2292fc051a13f05ea6f5b6c16f3d3635)** (CAIP 2011) — 1,000 + 200 clips
- **[Violent-Flows](https://www.openu.ac.il/home/hassner/data/violentflows/)** (CVPR Workshops 2012) — Violent crowd behavior
- **[RWF-2000](https://github.com/mchengny/RWF2000-Video-Database-for-Violence-Detection)** (ICPR 2020) — 2,000 videos
- **[CCTV-Fights](https://rose1.ntu.edu.sg/dataset/cctvFights/)** (ICASSP 2019) — 1,000 real-world fights
- **[VFD-2000](https://github.com/Hepta-Col/VideoFightDetection)** (ICTAI 2022) — Multi-scenario, various lengths
- **[VSD](https://www.interdigital.com/data_sets/violent-scenes-dataset)** (MTA 2015) — 18 Hollywood movies

---

## 📍 Anomaly Localization

*Datasets with fine-grained spatial or temporal annotations for localizing anomalies.*

### Spatial Localization

- **[UBnormal](https://github.com/lilygeorgescu/UBnormal)** (CVPR 2022)  
  - **Pixel-level annotations** for 543 videos
  - 22 training anomaly types, disjoint from test
  - Open-set benchmark simulating real-world scenarios
  
- **[ShanghaiTech](https://svip-lab.github.io/dataset/campus_dataset.html)** (ICCV 2017)  
  - Frame-level and pixel-level masks available
  - 437 videos across 13 campus scenes

### Temporal Localization

- **[UCF-Crime](https://www.crcv.ucf.edu/projects/real-world/)** (CVPR 2018)  
  - Temporal segment annotations
  - 1,900 untrimmed videos with anomaly timestamps
  
- **[XD-Violence](https://roc-ng.github.io/XD-Violence/)** (ECCV 2020)  
  - 4,754 videos, 217 hours, 6 violence types
  - Frame-level temporal annotations
  - **Audio-visual** modality (speech, sound effects)
  
- **[NWPU Campus](https://campusvad.github.io/)** (CVPR 2023)  
  - Temporal annotations for 28 anomaly classes
  - Supports anomaly anticipation (future prediction)

- **[X-Man](https://github.com/VAD-X-Man/X-Man-Dataset)** (CVPR 2024)
  - **1.3 million frames** with fine-grained temporal and spatial annotations
  - 21 anomaly categories, designed for complex scenes
  - Focus: Spatio-temporal localization and open-world recognition

---

## 💡 Anomaly Understanding

*Datasets with natural language descriptions, explanations, and multimodal reasoning capabilities.*

### Video-Language Annotation

- **[UCA (UCF-Crime Annotation)](https://xuange923.github.io/Surveillance-Video-Understanding)** (CVPR 2024)  
  - **23,542 fine-grained sentences**, 111 hours
  - Temporal event descriptions for surveillance video understanding
  - Enables video captioning and Q&A for anomaly scenes
  
- **[UCCD](https://github.com/lingruzhou/UCCD)** (TMM 2024)  
  - Human-centric behavior descriptions
  - Instance-level annotations with temporal info
  - Focus: Crowd behavior understanding

### Explainable & Reasoning

- **[VAD-Instruct50k (Holmes-VAD)](https://video-holmes.github.io/)** (arXiv 2024)  
  - **51,567 multimodal instructions** for explainable VAD
  - Rich textual explanations for anomaly reasoning
  - Supports why/what/when questions
  - LLM-ready format for instruction tuning
  
- **[DriveQA](https://driveqaiccv.github.io)** (ICCV 2023)  
  - 5,000+ driving video clips, **25,000+ QA pairs**
  - Question types: Driving behavior analysis, traffic rules understanding
  - Supports open-ended reasoning and decision explanation
  - Authors: Maolin Wei, Wanzhou Liu, Eshed Ohn-Bar

### Open-World Understanding

- **[UBnormal](https://github.com/lilygeorgescu/UBnormal)** (CVPR 2022)  
  - Open-set benchmark: training and test anomalies are **disjoint**
  - Simulates real-world unseen anomaly scenarios
  - Tests generalization to novel anomaly types
  
- **[MSAD](https://msad-dataset.github.io/)** (NeurIPS 2024) [![arXiv](https://img.shields.io/badge/arXiv-2402.04857-b31b1b)](https://arxiv.org/pdf/2402.04857)  
  - **14 diverse scenarios** (factory, tunnel, prison, classroom, etc.)
  - 6,811 videos with 150,308 frames
  - Long-tail distribution with rare anomaly types
  - Designed for real-world deployment challenges

---

## 🔍 Anomaly Retrieval

*Cross-modal retrieval tasks: finding anomalies via text/audio queries.*

### Video-Text Retrieval

- **[UCFCrime-AR](https://github.com/Roc-Ng/VAR)** (TIP 2024)  
  - Video-text retrieval benchmark
  - 1,900 videos with **Chinese & English descriptions**
  - Task: Retrieve anomaly videos matching text descriptions
  - Enables zero-shot anomaly detection via language

### Audio-Visual Retrieval

- **[XDViolence-AR](https://github.com/Roc-Ng/VAR)** (TIP 2024)  
  - Audio-visual anomaly retrieval
  - 4,754 videos, cross-modal (video ↔ audio)
  - Task: Retrieve anomalies using audio or visual queries
  - Based on XD-Violence dataset with multimodal annotations

---

## 🎬 Anomaly Generation

*Datasets or tasks for synthesizing anomalies (emerging research area).*

> ⚠️ **Note**: This is an emerging direction. Most datasets above can be repurposed for generation tasks (e.g., counterfactual anomaly synthesis, normal-to-abnormal translation).

**Potential Applications:**
- Data augmentation for rare anomalies (using diffusion models)
- Counterfactual explanation: "What if this person ran instead of walked?"
- Anomaly simulation for training (e.g., generating synthetic accidents)

**Related Datasets:**
- **VAD-Instruct50k** — Text descriptions can guide anomaly video generation
- **UCA** — Fine-grained captions enable text-to-video generation of anomalies

---

## 🚗 Domain-Specific Datasets

*Specialized datasets for specific application domains.*

### Driving & Transportation

- **[Honda HDD](https://usa.honda-ri.com/hdd#Videos)** (Honda Research Institute Driving Dataset)  
  - 104 driving videos with 11 anomaly types
  - Baseline: LTA-PCA (Skeleton + Gaze + Location)
  - Focus: Driver behavior analysis

- **[ROADWork](https://www.cs.cmu.edu/~roadwork/)** (ICCV 2025)  
  *ROADWork: A Dataset and Benchmark for Learning to Recognize, Observe, Analyze and Drive Through Work Zones*  
  - Large-scale construction zone anomaly dataset
  - 15 hours of video, 85 work zone scenes, 2.4M frames
  - 12 object categories (workers, vehicles, machinery, traffic cones, etc.)
  - Use case: Construction site safety monitoring and autonomous navigation
  - Authors: Anurag Ghosh, Shen Zheng, Robert Tamburo, et al.

- **[Driver Anomaly Detection](https://github.com/okankop/Driver-Anomaly-Detection)** (DADA-2000)  
  - Inside-car driver behavior monitoring dataset
  - Focus: Distracted/abnormal driving patterns

- **[TAD](https://github.com/ktr-hubrt/WSAL)** (TIP 2021)  
  - Traffic Anomaly Dataset
  - Real-world traffic surveillance scenarios

- **[DriveQA](https://driveqaiccv.github.io)** (ICCV 2023)  
  *Passing the Driving Knowledge Test with LLMs*  
  - 5,000+ driving video clips, 25,000+ QA pairs
  - Question types: Driving behavior analysis, traffic rules understanding, anomaly identification
  - Supports open-ended reasoning and decision explanation
  - Authors: Maolin Wei, Wanzhou Liu, Eshed Ohn-Bar

### Multi-Scenario

- **[MSAD (Multi-Scenario Anomaly Detection)](https://msad-dataset.github.io/)** (NeurIPS 2024) [![arXiv](https://img.shields.io/badge/arXiv-2402.04857-b31b1b)](https://arxiv.org/pdf/2402.04857)  
  - Multi-scenario Anomaly Detection benchmark
  - 14 diverse scenarios (factory, tunnel, prison, classroom, etc.)
  - 6,811 videos with 150,308 frames
  - Long-tail distribution with rare anomaly types
  - Designed for real-world deployment challenges

---

## 📊 Quick Reference Table

| Dataset | Year | Type | Size | Key Feature |
|---------|------|------|------|-------------|
| **UCF-Crime** | 2018 | Detection + Localization | 1,900 videos | Temporal annotations, 13 types |
| **UCA** | 2024 | Understanding | 23,542 sentences | Fine-grained video-language |
| **VAD-Instruct50k** | 2024 | Understanding | 51,567 instructions | Explainable reasoning |
| **UCFCrime-AR** | 2024 | Retrieval | 1,900 videos | Video-text cross-modal |
| **XDViolence-AR** | 2024 | Retrieval | 4,754 videos | Audio-visual cross-modal |
| **X-Man** | 2024 | Localization | 1.3M frames | Spatio-temporal, open-world |
| **UBnormal** | 2022 | Localization + Understanding | 543 videos | Pixel-level, open-set |
| **NWPU Campus** | 2023 | Detection + Localization | 547 videos | Largest, 28 classes |
| **MSAD** | 2024 | Detection + Understanding | 6,811 videos | Multi-scenario, long-tail |
| **DriveQA** | 2023 | Understanding | 25,000 QA pairs | Driving reasoning |

---

*Last updated: October 2025*
