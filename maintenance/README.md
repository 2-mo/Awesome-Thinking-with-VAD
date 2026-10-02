# Maintainer workflow

Repository updates, source verification, citation metadata and site builds are handled by the maintainer. Reader-facing paper recommendations are described in [CONTRIBUTING.md](../.github/CONTRIBUTING.md).

See [development details](DEVELOPMENT.md) for the application and build. All command examples and code paths are relative to the repository root.

## Repository layout

| Location | Purpose |
| --- | --- |
| `README.md`, `README.zh-CN.md` | Conference/journal lists and the research route map entry |
| `llm4vad.md` | Generated paper lists targeted by the README year links |
| `literature/` | Generated reading indexes and citation exports |
| `archive/` | Historical notes and their original figures |
| `research/` | Surveys and source verification records |
| `maintenance/` | Maintainer workflow, development and design notes |
| `data/`, `src/`, `public/` | Catalog source, application source and public assets |
| `scripts/`, `tests/`, `.github/` | Generation, checks and GitHub configuration |
| `docs/` | Committed GitHub Pages build output |

Root package, TypeScript and Vite configuration files stay in place for the existing build workflow.

## Edit the source data

Edit `data/catalog.json`. Every paper needs its full title, year, venue, scope, cluster, tasks, a short `mechanism` naming its innovation, concise summary, takeaway, editorial reading question (`limitation` for compatibility), paper URL, verification sources with explanatory notes, verification date, and dataset references. Keep `mechanism` within 24 characters so it fits a compact paper tile; describe the method rather than repeating the paper title. Use stable unique IDs across all collections. Omit an unavailable optional code or project link instead of entering an empty string.

The catalog accepts only papers directly addressing video anomaly detection or understanding, with `scope: "core"`. Supported relationships are `uses`, `introduces` and `extends`; an extension must connect two catalog papers. Each relation requires a source URL and evidence note; references in datasets and reading guides must resolve to existing entries.

The validator checks structure, URLs and semantic references locally. It does not check network availability or establish that a paper supports a claim. Contributors must read the cited primary sources and verify titles, venue/year, claims, benchmark protocols and relationship evidence. A successful HTTP request alone is not evidence of correctness.

## Tasks, comparisons and datasets

Use the controlled task vocabulary in `scripts/catalog.mjs`; method, training and scene labels belong in optional `tags`. Comparison dimensions are `outputs`, `training`, `inference`, `futureFrames` and `evaluation`. Each populated fact is `{ values: ["..."], evidence: { url, note } }`. Omit unverified dimensions; the generated table displays them as unverified. Do not infer training-free operation from a frozen backbone or future-frame restrictions from the word “online”.

Dataset records can omit `thumbnail`. If an image is supplied, local path, alt text, provenance and credit are all required. Register annotations and a source-grounded protocol independently of artwork; explicitly distinguish a paper/project entry from a verified data download. Record paper associations only when their source supports them.

## Preserve the research map

Keep publication years on the horizontal axis, method topology on the vertical axis and method schools as the colored routes. Venue metadata belongs with paper labels, not in fixed rows. Datasets provide evaluation context. Use sourced main-conference, journal-publication or arXiv first-submission months for hidden ordering, and preserve forward-only metro geometry. Group WACV, Findings and workshop papers under “Other papers” in the READMEs; maintain supplementary WACV and workshop notes in [other papers](../literature/other-papers.md). Record explicit main-map exclusions in `paper.mapExclusion: { note }`, retaining bibliographic records and distinguishing editorial selection from evidence about methods or quality. STEP and TrajVAD are excluded; retain EWAD and the accepted NeurIPS papers SEEK-VAU, CA-Judge and ROAD. Check versions before adding an extended or renamed paper as a separate entry.

Keep the website focused on the route map. Search and filters change visible stations; a native modal dialog provides paper details and associated dataset protocols. Preserve readable type, keyboard access and scrollable details. Conference/journal reading, comparison and citation exports live in the repository documents.

The image-first visual references and generation prompts are recorded in [design/README.md](design/README.md). The concept image and the unused decorative strip at `src/assets/idea-strip.png` are historical generated artwork, not paper figures or evidence; the current website only renders the vector map. Research facts must come from the catalog sources. Record provenance when replacing artwork; do not claim a particular image-model version unless the generation tool reports it.

## Validate and regenerate

Use Node.js 24 and npm:

```sh
npm ci
npm run validate
npm test
npm run build
npm run check
```

Use `npm install` when initially adding or deliberately updating dependencies; commit the lockfile. Use `npm ci` to reproduce the checked-in dependency set. `npm run dev` opens the development workflow, and `npm run preview` serves the production build locally.

Do not hand-edit `literature/catalog.md`, `llm4vad.md`, `literature/venues.md`, `literature/comparison.md`, `literature/benchmarks.md`, `literature/reading-guide.md`, `literature/citations.md`, `literature/references.bib` or `docs/`. The build regenerates the seven Markdown indexes and the BibTeX library from the data and writes the Vite application to `docs/`. Include all generated outputs in your change. CI verifies the generated catalog, tests, TypeScript and production build, then rejects differences in the committed generated outputs.

In a pull request, state which research claims changed, link their evidence, explain uncertainty, and report validation. Keep existing GitHub Pages deployment from the committed `docs/` folder; this contribution workflow does not change repository Pages settings or add a publishing workflow.

## Release notes by batch

Update `CHANGELOG.md` with each content batch: date and batch number, additions, corrections, and version merges (or explicitly none). Link detailed evidence in `research/`. Keep the README's latest update brief and reader-facing. Record newly verified citation sources and changes in authors, DOI or cited version. A changelog entry is not a GitHub Release; publishing a release is a separate maintainer action.

The root MIT `LICENSE` covers original repository code and documentation. Third-party papers, datasets and figures retain their own terms and attribution.
