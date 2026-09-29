# Awesome Thinking with VAD

[![Awesome](https://awesome.re/badge.svg)](https://awesome.re)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

English | [简体中文](README.zh-CN.md)

Research on **video anomaly understanding**: explaining abnormal events, grounding them in time and space, reasoning over evidence, and evaluating whether an explanation is faithful to the video. The collection also includes directly relevant language-guided representations and detection methods.

[**Research map**](https://2-mo.github.io/Awesome-Thinking-with-VAD/) · [**Papers by year**](llm4vad.md) · [**Papers by method**](catalog.md) · [**Venue index**](venues/README.md)

## Latest update

**2026-09-30** — Added 19 source-checked papers, bringing the selected catalog to **51 papers**, with 9 dataset exhibits and 3 reading guides. Additions cover missing NeurIPS 2025 work, AAAI/ICLR/CVPR/ICML/ECCV 2026 publications, and recent VAU preprints. See the [addition and version notes](research/literature-update-2026-09.md).

The year-based reading document now comes from the same data as the website. Empty links, duplicate versions, incorrect publication groupings, and unrelated entries from the old LLM/VAD list have been removed.

## Read the collection

| Entry point | Contents |
| --- | --- |
| [Interactive map](https://2-mo.github.io/Awesome-Thinking-with-VAD/) | Paper stations on five colored method routes; searchable publication metadata and sources |
| [llm4vad.md](llm4vad.md) | Compact tables by year and conference, including separately labeled preprints |
| [catalog.md](catalog.md) | Contributions, reading questions, limitations and primary sources by method family |
| [Venue index](venues/README.md) | Current publication counts and links to year/conference sections |
| [Data contract](data/README.md) | Scope, source requirements, publication status and version handling |

The map keeps **time on the horizontal axis and arranges stations by method topology**. Conference names travel with paper labels rather than defining rows. The layout places methods with shared papers next to each other, aligns stations into straight runs with compact circular corners, and routes around labels with penalties for bends and crossings. Shared papers use separate line platforms joined by a short neutral connector, with one paper label; ordinary crossings remain unconnected. Months establish an internal order; Q1–Q4 appear beneath years from 2025 onward. Filtering preserves the full-catalog geometry. NeurIPS and its Datasets and Benchmarks track share one display category while retaining exact metadata in each record.

The five families are semantic alignment and fusion; language-based criteria and prompt optimization; temporal hierarchy and memory; active observation and tool use; and structured reasoning and verification. Dataset exhibits use original author/paper figures with attribution.

## Scope and sources

[data/catalog.json](data/catalog.json) is the single source for the website and all three generated reading indexes. Each paper records the evidence supporting its title, venue, contribution and verification date. Formal publication metadata takes precedence over the initial preprint date; unconfirmed papers remain **arXiv**. An extended version is not counted twice merely because it has a new title.

The selected catalog excludes WACV and workshop papers, generic video understanding, and static-image defect detection. Detection-only methods are included selectively when they directly support anomaly semantics. Author performance claims are not presented as independent reproductions. A project placeholder is distinguished from released code.

The [older conference notes](venues/), [journal notes](journals/README.md) and [broader dataset notes](dataset.md) remain historical references with a wider scope; they have not all been reverified. Use the generated indexes above for the current curated selection.

## Run locally

Development uses Node.js 24:

```sh
npm ci
npm run dev
```

`npm run build` validates the data, regenerates `catalog.md`, `llm4vad.md` and `venues/README.md`, and builds the static website. `npm run check` checks data, generated documents, tests and TypeScript.

The production site is committed in [docs/](docs/), so `main` can run without a Node build:

```sh
python3 -m http.server 8000 --directory docs
```

Open `http://localhost:8000/`. See [DEVELOPMENT.md](DEVELOPMENT.md) for maintenance details.

## Contribute

Edit the [source catalog](data/catalog.json), cite primary sources and follow [CONTRIBUTING.md](CONTRIBUTING.md). Include generated documents and `docs/` with the source change. CI rejects stale generated output. Work on a development branch and merge after the required checks pass.

Corrections to titles, publication status, version relationships and missing relevant papers are welcome.

## Contact and credits

- Email: **mo1031@live.com**
- WeChat: **tiumo-** (please mention VAD)

Paper and dataset copyrights belong to their authors and publishers. This repository is an academic research collection; figures retain their source attribution.
