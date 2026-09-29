# Development

The atlas uses Vite and TypeScript. Its source of truth is `data/catalog.json`; `src/types.ts` describes the data contract. The Node validation and Markdown generation have no additional runtime dependencies. Geometry regression tests import the TypeScript layout using Node's built-in type stripping.

## Innovation transit map and artwork

The interface uses a compact metro-style vector network: publication years on the horizontal axis, displayed venue categories on the vertical axis, and five colored innovation-method routes across the paper stations. Warm paper colors and compact transit signage frame the map. Each paper's required `mechanism` is a short method label, limited to 24 characters and also included in the generated catalog. Datasets support evaluation context; they do not define the map's main routes. The catalog contains only video anomaly detection and understanding research.

The desktop map keeps all 32 paper stations visible without pagination. Other reading views use same-screen pagination to avoid page scrolling. Keep interface copy short and preserve keyboard access to every station, page and detail.

The design starts from the image concept at [design/comic-atlas-concept.png](design/comic-atlas-concept.png); [design/README.md](design/README.md) records the prompts and provenance. `src/assets/idea-strip.png` supplies decorative comic scenes. These images were generated with the built-in image tool and are visual references, not scientific evidence or a source of catalog facts. The tool did not expose a model selector; no specific model version is asserted. Keep these source assets separate from Vite's generated copies in `docs/`.

## Local development

Use Node.js 24. Install the locked dependencies with `npm ci` (or `npm install` when intentionally changing dependencies), then start the development server:

```sh
npm ci
npm run dev
```

Useful commands:

| Command | Purpose |
| --- | --- |
| `npm run validate` | Validate required fields, source links, scope and entity references. |
| `npm run generate` | Regenerate the root `catalog.md`. |
| `node scripts/generate-catalog.mjs --check` | Fail if `catalog.md` is stale; do not modify it. |
| `npm test` | Run Node's built-in tests, including the real catalog and deliberately damaged fixtures. |
| `npm run check` | Validate data, check generated Markdown, run tests and check TypeScript. |
| `npm run build` | Validate data, generate Markdown, check TypeScript and build Vite into `docs/`. |
| `npm run preview` | Serve the production output for local inspection. |

The validator intentionally makes no network requests. Semantic claims still require editorial checking against the recorded primary sources. Relation endpoint validation cannot establish a genuine lineage between papers; the evidence note must do that work.

## Committed static site

The production Vite output in `docs/` is committed so `main` works as a static site without a Node build on the deployment host. From the repository root, serve the checked-in site directly:

```sh
python3 -m http.server 8000 --directory docs
```

Open `http://localhost:8000/`. After editing source data or application code, run `npm run build`, inspect the output, then include `docs/` and `catalog.md` with the source changes. Do not edit those generated files by hand.

`.github/workflows/check.yml` installs dependencies with `npm ci`, runs `npm run check` and `npm run build`, and verifies `git diff --exit-code -- docs catalog.md` plus `git status --porcelain -- docs catalog.md` so new untracked build output is checked too. GitHub Pages keeps its existing deployment from `docs/`; no repository Pages settings or extra publishing workflow are required by these checks.

## Dataset gallery provenance

The catalog now contains 32 papers and 9 datasets. Every dataset requires `year`, `venue`, and `thumbnail: { src, alt, sourceUrl, credit }`. Thumbnail paths must be local `/datasets/` assets without traversal; provenance URLs must be HTTP(S), and alt text and credit must be nonempty. Validation is structural and does not confirm image rights or scientific content. Keep the original author/paper figures and their attribution; [public/datasets/README.md](public/datasets/README.md) records the mapping. Images are not generated samples. The Markdown generator emits publication metadata and image-source links without embedding full images.

The literature expansion prioritizes video anomaly understanding, explanation, reasoning and their semantic foundations. The curated paper scope excludes WACV and workshop papers. Use formal proceedings to establish conference/year, preserve the exact NeurIPS Datasets and Benchmarks track in source data and paper details, and retain arXiv status where formal publication is unconfirmed. The map and publication filter merge both NeurIPS tracks into one displayed category; the current map has 11 venue rows. Files under `research/` are retained verification notes and merge inputs, not another runtime source of truth.

The dataset exhibition shows a 3×3 grid (9 per page); narrow screens use 2×3 pages to keep labels readable. Images retain source attribution in the detail pane. Search and task filters operate within datasets; paper publication filters remain separate. The network renders all 32 papers at their year and venue coordinates, with five labeled method-school routes. `src/components/publication-layout.ts` computes the year zones, venue bands, station labels and obstacle-aware route geometry using horizontal, vertical and 45-degree segments from the full catalog, so filtering does not move stations. Optional paper `timeline: { month, basis, source }` metadata orders stations within each year; `basis` is `conference` for the main-conference month or `preprint` for the arXiv v1 month. Months are verified against the recorded primary source and are not shown in the interface. Unknown months sort last, and spacing within/between months is schematic rather than proportional. All method routes must progress rightwards; geometry regression tests check chronology, no backtracking, station membership, label overlap and year/venue bounds. No paper pagination or duplicate sidebars are used; venue and year appear on the axes and in details, while stations use compact one-line paper names. Details open in an overlay. Line membership is an editorial grouping, not a citation or inheritance relationship. A crossing without a paper station has no additional semantic meaning; sourced relationships remain in paper details. The paper index also shows all filtered entries as compact cards. The map has no decorative raster backdrop. The former generated backdrop, `src/assets/research-panorama.png`, is retained as a source asset; its prompt and provenance remain in `design/panorama-prompt.md`.
