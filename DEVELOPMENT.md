# Development

The atlas uses Vite and TypeScript. Its source of truth is `data/catalog.json`; `src/types.ts` describes the data contract. The Node validation, Markdown generation and tests have no additional runtime dependencies.

## Innovation transit map and artwork

The interface uses a compact metro-style vector map organized around innovation mechanisms and research questions, with warm paper colors and bold transit signage. Each paper's required `mechanism` is a short method label, limited to 24 characters and also included in the generated catalog. Datasets support evaluation context; they do not define the map's main routes. The catalog contains only video anomaly detection and understanding research.

The desktop map keeps all 37 paper stations visible without pagination. Other reading views use same-screen pagination to avoid page scrolling. Keep interface copy short and preserve keyboard access to every station, page and detail.

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

The catalog now contains 37 papers and 9 datasets. Every dataset requires `year`, `venue`, and `thumbnail: { src, alt, sourceUrl, credit }`. Thumbnail paths must be local `/datasets/` assets without traversal; provenance URLs must be HTTP(S), and alt text and credit must be nonempty. Validation is structural and does not confirm image rights or scientific content. Keep the original author/paper figures and their attribution; [public/datasets/README.md](public/datasets/README.md) records the mapping. Images are not generated samples. The Markdown generator emits publication metadata and image-source links without embedding full images.

The literature expansion prioritizes video anomaly understanding, explanation, reasoning and their semantic foundations. Use formal proceedings to establish conference/year, distinguish NeurIPS Datasets and Benchmarks, and preserve workshop or arXiv status where appropriate. Files under `research/` are retained verification notes and merge inputs, not another runtime source of truth.

The dataset exhibition shows a 3×3 grid (9 per page); narrow screens use 2×3 pages to keep labels readable. Images retain source attribution in the detail pane. Search and task filters operate within datasets; paper publication filters remain separate. The metro map renders all 37 papers as stations on five labeled editorial innovation lines, using vector geometry and consistent station markers. No paper pagination or duplicate sidebars are used; venue and year remain on every station label, and details open in an overlay. Line membership is an editorial grouping, not a citation or inheritance relationship; sourced relationships remain in paper details. The paper index also shows all filtered entries as compact cards. The map has no decorative raster backdrop. The former generated backdrop, `src/assets/research-panorama.png`, is retained as a source asset; its prompt and provenance remain in `design/panorama-prompt.md`.
