# Development

The atlas uses Vite and TypeScript. Its source of truth is `data/catalog.json`; `src/types.ts` describes the data contract. The Node validation and Markdown generation have no additional runtime dependencies. Geometry regression tests import the TypeScript layout using Node's built-in type stripping.

## Innovation transit map and artwork

The interface uses a compact metro-style vector network: publication time on the horizontal axis, topology-based station heights, conference names under paper labels, and five colored innovation-method routes with shared-paper interchanges. Warm paper colors and compact transit signage frame the map. Each paper's required `mechanism` is a short method label, limited to 24 characters and also included in the generated catalog. Datasets support evaluation context; they do not define the map's main routes. The catalog contains only video anomaly detection and understanding research.

The desktop map keeps all catalog paper stations visible without pagination. Other reading views use same-screen pagination to avoid page scrolling. Keep interface copy short and preserve keyboard access to every station, page and detail.

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
| `npm run generate` | Regenerate `catalog.md`, `llm4vad.md` and `venues/README.md`. |
| `node scripts/generate-catalog.mjs --check` | Fail if any generated Markdown index is stale; do not modify files. |
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

Open `http://localhost:8000/`. After editing source data or application code, run `npm run build`, inspect the output, then include `docs/`, `catalog.md`, `llm4vad.md` and `venues/README.md` with the source changes. Do not edit those generated files by hand.

`.github/workflows/check.yml` installs dependencies with `npm ci`, runs `npm run check` and `npm run build`, and verifies `git diff --exit-code -- docs catalog.md llm4vad.md venues/README.md` plus a status check for those paths so new untracked build output is checked too. GitHub Pages keeps its existing deployment from `docs/`; no repository Pages settings or extra publishing workflow are required by these checks.

## Dataset gallery provenance

Current paper and dataset counts are listed in the generated indexes. Every dataset requires `year`, `venue`, and `thumbnail: { src, alt, sourceUrl, credit }`. Thumbnail paths must be local `/datasets/` assets without traversal; provenance URLs must be HTTP(S), and alt text and credit must be nonempty. Validation is structural and does not confirm image rights or scientific content. Keep the original author/paper figures and their attribution; [public/datasets/README.md](public/datasets/README.md) records the mapping. Images are not generated samples. The Markdown generator emits publication metadata and image-source links without embedding full images. It also writes the year/venue literature table and venue navigation from the same paper records, so the main reading document cannot drift from the website.

The literature expansion prioritizes video anomaly understanding, explanation, reasoning and their semantic foundations. The curated paper scope excludes WACV and workshop papers. Use formal proceedings to establish conference/year, preserve the exact NeurIPS Datasets and Benchmarks track in source data and paper details, and retain arXiv status where formal publication is unconfirmed. The map and publication filter merge both NeurIPS tracks into one displayed category; venue metadata is displayed with each paper and does not constrain map coordinates. Files under `research/` are retained verification notes and merge inputs, not another runtime source of truth.

The dataset exhibition shows a 3×3 grid (9 per page); narrow screens use 2×3 pages. Images retain source attribution in the detail pane. The paper map shows all catalog papers without pagination, using time-constrained topology instead of conference rows. `src/components/publication-layout.ts` searches method orderings using shared papers, relaxes and regularizes station heights along chronological method chains, reserves horizontal station platforms, packs name/venue labels and computes forward-only octilinear routes with bend, crossing and overlap penalties. See [design/metro-map.md](design/metro-map.md) for the algorithm and visual contract. This deterministic heuristic is not a globally optimal graph drawing.

Optional `secondaryMethods: [{ cluster, evidence: { url, note } }]` adds a sourced editorial membership to a paper's primary `cluster`. Multiple member routes visit one station, rendered as a double ring. `paperMethods` centralizes membership for geometry, focus, filtering and related papers. Validation rejects unknown, duplicate or unsourced memberships. Paper sources and generated `catalog.md` expose their classification evidence without duplicating paper entries.

Optional `timeline: { month, basis, source }` metadata orders papers within each year: main-conference month for `conference`, arXiv v1 month for `preprint`. Months remain hidden; Q1–Q4 appear from 2025 onward. Empty quarters receive compact slots, unknown months remain in an undated slot, and spacing is schematic. Filters preserve the full-catalog layout. Geometry tests cover chronology, quarter membership, all method memberships at shared stations, venue-independent coordinates, deterministic layout, label clearance, unrelated-node clearance, map bounds and no backward routes. No browser visual checks are required by these scripts.

Details open in an overlay. Shared method membership is an editorial classification, not a citation or inheritance relationship. A crossing without a paper station has no semantic meaning; sourced resource relationships remain in paper details. The paper index shows all filtered entries as compact cards. Historical raster backgrounds remain unused; the current map and exported SVG are entirely vector-based.
