# Awesome LLM4VAD

A curated list of papers and resources on Large Language Models for Video Anomaly Detection (VAD).


## Contents

- [Overview](#overview)
- [Motivation: Why Does VAD Need "Thinking"?](#motivation-why-does-vad-need-thinking)
- [Papers by Year](#papers-by-year)
  - [2025](#2025)
  - [2024](#2024)
- [Metrics & Evaluation](#metrics--evaluation)
- [Related Awesome Lists](#related-awesome-lists)

---

## Overview

This list collects representative works that leverage LLMs or vision-language models for video anomaly detection, explanation, and understanding. Entries are grouped by year with links to paper and code, plus a preview figure when available.

---

## Motivation: Why Does VAD Need "Thinking"?

The core idea is that "thinking" isn't exclusive to anomaly scenarios, but its necessity is amplified in VAD for several reasons:

- **Context-dependency (Complexity)**: Anomalies are often long-term events (e.g., fights, accidents) that require understanding causality and scene context.
- **Ambiguity (Fuzziness)**: Local actions or scenes can be easily confused with anomalies (e.g., running vs. fleeing, gathering vs. rioting). Disambiguation requires longer-term and multi-modal cues.
- **Long-tail Distribution (Sparsity)**: Anomalies are rare and unpredictable. Single observations are prone to misses, demanding evidence accumulation and hypothesis testing over time.

#### Why is this "thinking" process critical for anomalies but less so for normal scenarios?

- **Normal patterns are perception-driven**: Normal behaviors are frequent and regular, making them easy to learn with pattern matching.
- **Anomalies are inherently "uncertain"**: They are sparse and few-shot, lacking strong prior statistical support. Relying solely on fast perception leads to biases, which must be compensated by slower, more deliberate reasoning.
- **Anomalies carry higher risks**: Misjudgments can have severe consequences (e.g., missing a security threat). This necessitates a more robust decision-making process.
- **Anomalies break conventions**: They manifest as complex contextual dependencies, semantic ambiguities, and long-tail distributions—all of which are what "thinking" excels at handling.

In short, we need **reasoning**, not just post-hoc explanation.

### Curiosity-driven Learning

- **Humans monitor learning progress in curiosity-driven exploration** (Nature Communications 2021) [[paper](https://www.nature.com/articles/s41467-021-26196-w)]
- **Curiosity-driven Exploration by Self-supervised Prediction** (ICML 2017) [[paper](https://proceedings.mlr.press/v70/pathak17a/pathak17a.pdf)]
- **Computational mechanisms of curiosity and goal-directed exploration** (Neuroscience 2019) [[paper](https://elifesciences.org/articles/41703)]

---

## Papers by Year

### 2025

#### NeurIPS 2025

##### PANDA: Towards Generalist Video Anomaly Detection via Detective-like Agent
[![NeurIPS](https://img.shields.io/badge/NeurIPS-2025-2DB55D)](https://neurips.cc/virtual/2025/poster/115891)
> Proposes a detective-like agent paradigm for generalist VAD, achieving cross-scene and cross-category generalization through tool use and multi-step reasoning.

---

##### MoniTor: Exploiting Large Language Models with Instruction for Online Video Anomaly Detection
[![NeurIPS](https://img.shields.io/badge/NeurIPS-2025-2DB55D)](https://neurips.cc/virtual/2025/poster/119803)
> Leverages LLMs with instruction-driven mechanisms for online VAD, enhancing response speed and accuracy in real-time scenarios through streaming inference.

---


#### ICML 2025

##### Ex-VAD: Explainable Fine-grained Video Anomaly Detection Based on Visual-Language Models
[![ICML](https://img.shields.io/badge/ICML-2025-FF6B6B)](https://openreview.net/forum?id=xAhUoyb5eU)
[![Paper](https://img.shields.io/badge/Paper-PDF-blue)](https://raw.githubusercontent.com/mlresearch/v267/main/assets/huang25ad/huang25ad.pdf)
> Provides fine-grained explanations for anomalies using the semantic understanding capabilities of VLMs, enhancing model transparency.

---

#### ICCV 2025

##### Aligning Effective Tokens with Video Anomaly in Large Language Models
[![ICCV](https://img.shields.io/badge/ICCV-2025-00CED1)](https://arxiv.org/pdf/2508.06350)
> Maps VAD to a token alignment problem in LLMs, enabling multi-modal, large-model-driven video anomaly understanding.

---

##### Beyond Pixel Uncertainty: Bounding the OoD Objects in Road Scenes
[![ICCV](https://img.shields.io/badge/ICCV-2025-00CED1)](https://www.cs.cmu.edu/~roadwork/)
> Moves beyond pixel-level uncertainty to locate out-of-distribution objects in road scenes with precise bounding boxes, crucial for autonomous driving.

---

#### ACM MM 2025

##### EventVAD: Training-Free Event-Aware Video Anomaly Detection
[![ACM MM](https://img.shields.io/badge/ACM_MM-2025-FF69B4)](https://arxiv.org/abs/2504.13092)
[![Code](https://img.shields.io/github/stars/YihuaJerry/EventVAD?style=social&label=Code&logo=github)](https://github.com/YihuaJerry/EventVAD)
> A training-free, event-aware VAD method based on Video-LLaMA2 that uses zero-shot event understanding for cross-scene anomaly discrimination.
![EventVAD preview](./assets/eventvad-acmmm25.png)

---

##### SAGE: A Visual Language Model for Anomaly Detection via Fact Enhancement and Entropy-aware Alignment
[![ACM MM](https://img.shields.io/badge/ACM_MM-2025-FF69B4)]()
> A VLM for anomaly detection that uses fact enhancement and entropy-aware alignment to improve perception of anomalous details.

---

#### CVPR 2025

##### VERA: Explainable Video Anomaly Detection via Verbalized Learning of Vision-Language Models
[![CVPR](https://img.shields.io/badge/CVPR-2025-1E90FF)](https://openaccess.thecvf.com/content/CVPR2025/papers/Ye_VERA_Explainable_Video_Anomaly_Detection_via_Verbalized_Learning_of_Vision-Language_CVPR_2025_paper.pdf)
[![Code](https://img.shields.io/github/stars/vera-framework/VERA?style=social&label=Code&logo=github)](https://github.com/vera-framework/VERA)
> Transforms anomaly judgment into a "verbal reasoning" task, enabling the model to provide readable explanations and multi-modal evidence.
![VERA preview](./assets/2025-cvpr-vera.png)

---

##### Holmes-VAU: Towards Long-term Video Anomaly Understanding at Any Granularity
[![CVPR](https://img.shields.io/badge/CVPR-2025-1E90FF)](https://openaccess.thecvf.com/content/CVPR2025/papers/Zhang_Holmes-VAU_Towards_Long-term_Video_Anomaly_Understanding_at_Any_Granularity_CVPR_2025_paper.pdf)
[![Code](https://img.shields.io/github/stars/pipixin321/HolmesVAU?style=social&label=Code&logo=github)](https://github.com/pipixin321/HolmesVAU)
> A framework for long-term video understanding at any granularity, covering event, segment, and frame-level anomalies with linguistic descriptions.
![Holmes-VAU preview](./assets/2025-cvpr-holmes-vau.png)

---

#### arXiv 2025 (Preprints)

##### AVadCLIP: Audio-Visual Collaboration for Robust Video Anomaly Detection
[![arXiv](https://img.shields.io/badge/arXiv-2025-b31b1b?logo=arxiv)]()
> Extends VadCLIP to audio-visual collaboration, using audio cues to improve robustness in complex scenes.

---

##### AssistPDA: An Online Video Surveillance Assistant for Video Anomaly Prediction
[![arXiv](https://img.shields.io/badge/arXiv-2025-b31b1b?logo=arxiv)]()
> An online video surveillance assistant that uses large models for anomaly prediction and real-time feedback to aid human operators.

---

##### SlowFastVAD: Video Anomaly Detection via Integrating Simple Detector and RAG-Enhanced Vision-Language Model
[![arXiv](https://img.shields.io/badge/arXiv-2025-b31b1b?logo=arxiv)]()
> Combines a SlowFast detector with a RAG-enhanced VLM to improve anomaly understanding and localization.

---

##### Vad-R1: Towards Video Anomaly Reasoning via Perception-to-Cognition Chain-of-Thought
[![arXiv](https://img.shields.io/badge/arXiv-2025-b31b1b?logo=arxiv)]()
> A perception-to-cognition Chain-of-Thought framework that enables end-to-end reasoning from visual perception to anomaly judgment.

---

##### Flashback: Memory-Driven Zero-shot, Real-time Video Anomaly Detection
[![arXiv](https://img.shields.io/badge/arXiv-2025-b31b1b?logo=arxiv)]()
> A memory-driven, zero-shot, real-time VAD method that uses a dynamic memory bank for rapid response without training.

---

##### Simplifying Traffic Anomaly Detection with Video Foundation Models
[![arXiv](https://img.shields.io/badge/arXiv-2025-b31b1b?logo=arxiv)]()
> Simplifies traffic anomaly detection using the transfer learning capabilities of pre-trained video foundation models.

---

##### NexViTAD: Few-shot Unsupervised Cross-Domain Defect Detection via Vision Foundation Models and Multi-Task Learning
[![arXiv](https://img.shields.io/badge/arXiv-2025-b31b1b?logo=arxiv)]()
> A few-shot, unsupervised, cross-domain defect detection method for industrial anomaly scenarios.

---

##### AnomalyMoE: Towards a Language-free Generalist Model for Unified Visual Anomaly Detection
[![arXiv](https://img.shields.io/badge/arXiv-2025-b31b1b?logo=arxiv)]()
> A language-free Mixture-of-Experts (MoE) model for unified visual anomaly detection with cross-domain generalization.

---

##### Unlocking Vision-Language Models for Video Anomaly Detection via Fine-Grained Prompting
[![arXiv](https://img.shields.io/badge/arXiv-2025-b31b1b?logo=arxiv)]()
> Unlocks the potential of VLMs for VAD by designing task-specific, fine-grained prompt templates.

---

##### VAU-R1: Advancing Video Anomaly Understanding via Reinforcement Fine-Tuning
[![arXiv](https://img.shields.io/badge/arXiv-2025-b31b1b?logo=arxiv)]()
> Uses reinforcement learning fine-tuning to guide a model toward better anomaly discrimination strategies.

---

##### Language-guided Open-world Video Anomaly Detection
[![arXiv](https://img.shields.io/badge/arXiv-2025-b31b1b?logo=arxiv)]()
> A language-guided method for detecting unseen anomaly categories in an open-world setting via natural language descriptions.

---

### 2024

#### AAAI 2024

##### VadCLIP: Adapting Vision-Language Models for Weakly Supervised Video Anomaly Detection
[![AAAI](https://img.shields.io/badge/AAAI-2024-1F77B4)](https://ojs.aaai.org/index.php/AAAI/article/view/28423)
[![arXiv](https://img.shields.io/badge/arXiv-2308.11681-b31b1b?logo=arxiv)](https://arxiv.org/abs/2308.11681)
[![Code](https://img.shields.io/github/stars/nwpu-zxr/VadCLIP?style=social&label=Code&logo=github)](https://github.com/nwpu-zxr/VadCLIP)
> Adapts vision-language models like CLIP for weakly supervised VAD, achieving efficient anomaly discrimination with limited annotations.
![VadCLIP preview](./assets/2024-aaai-vadclip.png)

---

#### CVPR 2024

##### Harnessing Large Language Models for Training-free Video Anomaly Detection (LAVAD)
[![CVPR](https://img.shields.io/badge/CVPR-2024-1E90FF)](https://openaccess.thecvf.com/content/CVPR2024/papers/Zanella_Harnessing_Large_Language_Models_for_Training-free_Video_Anomaly_Detection_CVPR_2024_paper.pdf)
[![Code](https://img.shields.io/github/stars/lucazanella/lavad?style=social&label=Code&logo=github)](https://github.com/lucazanella/lavad)
> Directly uses the semantic knowledge of large models for scene understanding via prompting, enabling rapid deployment without fine-tuning.
![Training-free VAD preview](./assets/2024-cvpr-training-free-vad.png)

---

##### Text Prompt with Normality Guidance for Weakly Supervised Video Anomaly Detection
[![CVPR](https://img.shields.io/badge/CVPR-2024-1E90FF)](https://arxiv.org/abs/2404.08531)
> Uses text descriptions of normal behavior as a weak supervisory signal, maintaining localization accuracy without frame-level labels.

---

#### ECCV 2024

##### Follow the Rules: Reasoning for Video Anomaly Detection with Large Language Models (AnomalyRuler)
[![ECCV](https://img.shields.io/badge/ECCV-2024-0B84FE)](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/10568.pdf)
[![Code](https://img.shields.io/github/stars/Yuchen413/AnomalyRuler?style=social&label=Code&logo=github)](https://github.com/Yuchen413/AnomalyRuler)
> Employs a two-stage reasoning mechanism (induction and deduction) for LLMs to infer normality rules and then detect anomalies.
![AnomalyRuler preview](./assets/2024-eccv-anomalyruler.png)

---

#### ICCV 2024

##### Video Anomaly Detection and Explanation via Large Language Models
[![arXiv](https://img.shields.io/badge/arXiv-2401.05702-b31b1b?logo=arxiv)](https://arxiv.org/pdf/2401.05702v1)
> Couples VAD with LLM-generated explanations to provide interpretable, textual reasons for model decisions.
![LLM VAD + Explanation preview](./assets/2024-arxiv-vad-llm-explanation.png)

---

#### NeurIPS 2024

##### HAWK: Learning to Understand Open-World Video Anomalies
[![NeurIPS](https://img.shields.io/badge/NeurIPS-2024-2DB55D)](https://proceedings.neurips.cc/paper_files/paper/2024/file/fca83589e85cb061631b7ebc5db5d6bd-Paper-Conference.pdf)
[![arXiv](https://img.shields.io/badge/arXiv-2405.16886-b31b1b?logo=arxiv)](https://arxiv.org/pdf/2405.16886)
[![Code](https://img.shields.io/github/stars/jqtangust/hawk?style=social&label=Code&logo=github)](https://github.com/jqtangust/hawk)
> Leverages VLMs to understand open-world video anomalies, enhancing perception of dynamic events by incorporating motion modalities.
![HAWK preview](./assets/2024-neurips-hawk.png)

---

##### MDVAD: Towards Multi-Domain Learning for Generalizable Video Anomaly Detection
[![NeurIPS](https://img.shields.io/badge/NeurIPS-2024-2DB55D)](https://proceedings.neurips.cc/paper_files/paper/2024/file/59eb2d8ce0e4830f80780f7f78c67dec-Paper-Conference.pdf)
> Proposes a multi-domain VAD task and benchmark to explore cross-domain generalization.

---

#### ACM MM 2024

##### Weakly Supervised Video Anomaly Detection and Localization with Spatio-Temporal Prompts
[![ACM MM](https://img.shields.io/badge/ACM_MM-2024-FF69B4)](https://arxiv.org/pdf/2408.05905)
> Uses spatio-temporal prompts with CLIP for weakly supervised VAD, improving fine-grained localization by adapting to different anomaly scales.

---

## Metrics & Evaluation

- Coming soon: common tasks, metrics, and evaluation protocols.

---

## Related Awesome Lists

- Coming soon.








