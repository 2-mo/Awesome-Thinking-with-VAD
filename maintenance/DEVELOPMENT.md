# Development

The atlas uses Vite and TypeScript. Its source of truth is `data/catalog.json`; `src/types.ts` describes the data contract. The Node validation and Markdown generation have no additional runtime dependencies. Geometry regression tests import the TypeScript layout using Node's built-in type stripping.

## Innovation transit map and artwork

The interface uses a compact metro-style vector network: publication time on the horizontal axis, topology-based station heights, conference names under paper labels, and five method trunks, two Y-shaped reading branches, and shared-paper interchanges. Warm paper colors and compact transit signage frame the map. Each paper's required `mechanism` is a short method label, limited to 24 characters and also included in the generated catalog. Datasets support evaluation context; they do not define the map's main routes. The catalog contains only video anomaly detection and understanding research.

The website has one view: the metro-style research route map. It keeps all catalog paper stations visible without pagination, with search, method/venue/year/task filters, pan/zoom and SVG export. Paper details open in a native modal dialog with methods, evaluation resources and source links. Escape closes the dialog and focus returns to the triggering station. Preserve keyboard access and readable content at narrow widths.

The design starts from the image concept at [design/comic-atlas-concept.png](design/comic-atlas-concept.png); [design/README.md](design/README.md) records the prompts and provenance. `src/assets/idea-strip.png` preserves an earlier decorative comic strip; it is no longer imported by the application. These images were generated with the built-in image tool and are visual references, not scientific evidence or a source of catalog facts. The tool did not expose a model selector; no specific model version is asserted. Keep these source assets separate from Vite's generated copies in `docs/`.

## Local development

Use Node.js 24. Install the locked dependencies with `npm ci` (or `npm install` when intentionally changing dependencies), then start the development server:

```sh
npm ci
npm run dev
```

Useful commands:

| Command | Purpose |
| --- | --- |
| `npm run validate` | Validate required fields, source links, scope, entity references and local paper-figure files. |
| `npm run generate` | Regenerate the literature indexes, illustrated `llm4vad.md` cards, `assets/papers/README.md` image provenance and `literature/references.bib`. |
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

Open `http://localhost:8000/`. After editing source data or application code, run `npm run build`, inspect the output, then include `docs/`, `literature/catalog.md`, `llm4vad.md`, `literature/venues.md`, `literature/comparison.md`, `literature/benchmarks.md`, `literature/reading-guide.md`, `literature/citations.md` and `literature/references.bib` with the source changes. Do not edit those generated files by hand.

`.github/workflows/check.yml` installs dependencies with `npm ci`, runs `npm run check` and `npm run build`, and verifies `git diff --exit-code -- docs literature llm4vad.md` plus a status check for those paths so new untracked build output is checked too. GitHub Pages keeps its existing deployment from `docs/`; no repository Pages settings or extra publishing workflow are required by these checks.

## Data and reading documents

Current paper and dataset counts are listed in the generated indexes. Every dataset requires `year` and `venue`; `thumbnail: { src, alt, sourceUrl, credit }` is optional. The map shows associated dataset names and protocols in paper details; the separate dataset gallery has been removed. Any supplied thumbnail still requires complete provenance. Thumbnail paths must be local `/datasets/` assets or existing `assets/papers/` figures without traversal; provenance URLs must be HTTP(S), and alt text and credit must be nonempty. Validation is structural and does not confirm image rights or scientific content. Keep the original author/paper figures and their attribution; [public/datasets/README.md](../public/datasets/README.md) records dataset thumbnails.

Paper `figure` metadata embeds original figures in the year/venue cards in `llm4vad.md`. Keep their files in root-level `assets/papers/`, preserve the original card heading/badge/summary structure, and record provenance in `data/catalog.json`; `assets/papers/README.md` is generated. Missing figures use `figurePending` with checked sources rather than fabricated artwork. The figure-file validator rejects missing files and HTML/error pages saved as images. Publication metadata, cards and venue navigation share the same paper records. For the exact figure schema and update procedure, see [data/README.md](../data/README.md#论文卡片与配图).

The literature expansion prioritizes video anomaly understanding, explanation, reasoning and their semantic foundations. WACV, Findings and workshop papers are grouped under “Other papers” in the READMEs. Supplementary WACV and workshop notes live in `literature/other-papers.md`; existing structured Findings records remain in the generated indexes. Use formal proceedings to establish conference/year, preserve the exact NeurIPS Datasets and Benchmarks track in source data and paper details, and retain arXiv status where formal publication is unconfirmed. The map and publication filter merge both NeurIPS tracks into one displayed category; venue metadata is displayed with each paper and does not constrain map coordinates. Files under `research/` are retained verification notes and merge inputs, not another runtime source of truth.

The map shows eligible catalog papers without pagination; WACV, Findings, workshop and editorially excluded records remain in the reading indexes. Optional `paper.mapExclusion: { note }` records editorial map selection separately from scientific metadata. STEP and TrajVAD are currently excluded; EWAD, SEEK-VAU, CA-Judge and ROAD remain eligible. Search, filter options, detail deep links, map counts and SVG export use the same `isMapPaper` rule, using time-constrained topology instead of conference rows. `src/components/publication-layout.ts` searches method orderings using shared papers, relaxes and regularizes station heights along chronological method chains, fits nearby ordinary station runs to direct 45-degree connections, reserves horizontal platforms at shared stations, packs name/venue labels and computes forward-only octilinear routes with bend, crossing and overlap penalties. Clear horizontal or 45-degree links are preferred without extra platform stubs at ordinary stations. See [design/metro-map.md](design/metro-map.md) for the algorithm and visual contract. This deterministic heuristic is not a globally optimal graph drawing.

Optional `secondaryMethods: [{ cluster, evidence: { url, note } }]` adds a sourced editorial membership to a paper's primary `cluster`. A shared paper remains one station object, label and interaction target, with a separate `platforms[]` point for each member route. Ordinary interchange platforms share the paper x coordinate and are stacked 32 units apart in method order. For a branch, `branchAt: { paperId, evidence }` names a sourced shared paper: its two platforms coincide at one visible fork station, and the branch starts exactly there. The fork paper must precede the branch papers. A neutral outlined connector links the markers; each route passes through its own platform. `stationBounds` reserves the entire station body for labels and unrelated routes, and routing also avoids other platforms of the same shared station. `paperMethods` centralizes membership for geometry, focus, filtering and related papers. Validation rejects unknown, duplicate or unsourced memberships. Paper sources and generated `literature/catalog.md` expose their classification evidence without duplicating paper entries.

Optional `timeline: { month, basis, source }` metadata orders papers within each year: main-conference month for `conference`, sourced online or issue-publication month for `journal` (the source note identifies which), arXiv v1 month for `preprint` (also a first-publication fallback for journals, as with CRCL). Months remain hidden; Q1–Q4 appear from 2025 onward. Empty quarters receive compact slots, unknown months remain in an undated slot, and spacing is schematic. Filters preserve the full-catalog layout. Geometry tests cover chronology, quarter membership, all method memberships at their own platforms, straight and separated interchange approaches, venue-independent coordinates, deterministic layout, label clearance, unrelated-node clearance, map bounds and no backward routes. No browser visual checks are required by these scripts.

Equal-month transfers may precede a fork when this avoids crossing its departing branch; the sourced fork remains before its own branch papers, and different-month chronology stays strict. Direction names are placed beside clear horizontal portions of their own rails after station-label packing. The compact station key then fits existing left-side whitespace, with a below-map fallback for small catalogs. Unplaceable direction names fall back to key entries. Both names and the key use full-catalog geometry, stay fixed during filtering and are reserved before mountain placement.

Dense trunks use 168-unit anchor spacing. Sparse branches beside their parent follow its broad shelves at 96-unit separation while remaining continuous. Long interchange approaches turn near arrival, and ordinary parent/lower-line stops may rise within an interval where an upper branch is not yet active. These layout choices reduce empty corridors without adding paper stations, changing time coordinates or claiming a new research relationship.

Dense months begin with 68-unit station slots; actual platform height differences and longer same-month names expand constrained columns while year and quarter boundaries move with them. Name/venue labels are packed jointly from positions directly above, below or beside each station; ordinary labels stay within 28 units and shared-station labels within 44 units. The map draws no label leaders. Branch parents may themselves be branches, and one line may fork at different papers; validation rejects cycles and every fork retains its own evidence. Layout tests cover repeated and nested forks, while hover/focus highlights all ancestor routes.

Details open in a native modal dialog. Shared method membership is an editorial classification, not a citation or inheritance relationship. A crossing without a paper station has no semantic meaning; dataset protocols and classification sources remain in paper details, with full resource relationships in the generated Markdown. README conference and journal lists are the primary text entry points. The generated year/method documents remain available for existing deep links and require no separate editing. Historical raster backgrounds remain unused; the current map and exported SVG are entirely vector-based.

Local reading geometry uses optional `Cluster.routes` with ordered `paperIds` and editorial evidence. A direction may contain multiple `PublicationLine.tracks`; SVG rendering, legend clearance and decoration inspect each subpath independently. Shared paper IDs anchor same-color branches without duplicating stations. Validation checks route membership, coverage, sources and publication chronology. Interchanges reserve intermediate bands in same-month packing. The early evaluation route leaves Vad-R1 and passes Cue-R1 and FineVAU, then returns to reasoning/verification at Vad-R1-Plus and its later segment begins at the sourced CG-CoE evaluation/verification interchange, clearing both Anom-π approaches. A route starting or ending at an interchange uses the platform itself as its endpoint, without a dangling terminal stub. A returning branch forms a level local corridor: Cue-R1/FineVAU above, URF-ZS-HVAA/TargetVAU on the reasoning trunk, and VAD-DPO on a same-color lower path. The three paths join at Plus; its sourced evaluation membership reflects the staged benchmark contribution. The layout preserves 120-unit lane separation, centers the fan between neighboring routes, reserves the first trunk label after the split, and keeps the trunk flat through its next ordinary stops. Same-color side-path departures/arrivals omit shared horizontal stubs to avoid unnamed overlaps. Routing rejects unnamed touching joins and coincident rails, while ordinary through crossings retain paper-colored casing.

A shared paper with exactly one incoming route end and one outgoing route start uses a single continuation marker (`Station.continuation`). Method memberships remain sourced and filterable. Both route coordinates coincide at the paper, with no terminal stubs; multi-arm interchanges keep separate platforms. LAVIDA uses this treatment, and its tooltip/accessibility text does not call it an interchange.

Vad-R1 and Vad-R1-Plus remain independent dated stations. Their source-backed `extends` relation records the publication lineage without merging the nodes or adding a drawn relationship edge. Both existing mountain groups use one light, unfilled outline style; decoration remains purely visual.

Early supervision is an inline color section: VadCLIP → OVVAD → TPWNG → HAWK follows a level corridor, with ordinary color continuations at OVVAD and TPWNG. TPWNG retains its primary supervision membership and adds sourced alignment membership. Short two-continuation sections follow the adjacent through shelf and do not flip later shelf alternation. The renderer names every direction beside its own rail and exports those labels with the compact station key.
