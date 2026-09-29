# Contributing

The atlas is a curated research guide for video anomaly understanding. Add papers based on primary sources and record exactly what those sources support. Existing conference notes remain useful additional material, but have not all been reverified for this atlas.

## Edit the source data

Edit `data/catalog.json`. Every paper needs its full title, year, venue, scope, cluster, tasks, a short `mechanism` naming its innovation, concise summary, takeaway, limitation, paper URL, verification sources with explanatory notes, verification date, and dataset references. Keep `mechanism` within 24 characters so it fits a compact paper tile; describe the method rather than repeating the paper title. Use stable unique IDs across all collections. Omit an unavailable optional code or project link instead of entering an empty string.

The catalog accepts only papers directly addressing video anomaly detection or understanding, with `scope: "core"`. Supported relationships are `uses`, `introduces` and `extends`; an extension must connect two catalog papers. Each relation requires a source URL and evidence note; references in datasets and reading guides must resolve to existing entries.

The validator checks structure, URLs and semantic references locally. It does not check network availability or establish that a paper supports a claim. Contributors must read the cited primary sources and verify titles, venue/year, claims, benchmark protocols and relationship evidence. A successful HTTP request alone is not evidence of correctness.

## Preserve the research map

Keep publication years on the horizontal axis, venues on the vertical axis and method schools as the colored routes. Datasets provide evaluation context. Use sourced main-conference or arXiv first-submission months for hidden ordering, and preserve forward-only metro geometry. WACV and workshop papers are outside the curated scope. Check versions before adding an extended or renamed paper as a separate entry.

Use same-screen pagination and short labels to keep the interface compact. Preserve readable type, keyboard access and access to all content; do not clip information to hide overflow.

The image-first visual references and generation prompts are recorded in [design/README.md](design/README.md). The concept image and the decorative strip at `src/assets/idea-strip.png` are generated artwork, not paper figures or evidence. Research facts must come from the catalog sources. Record provenance when replacing artwork; do not claim a particular image-model version unless the generation tool reports it.

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

Do not hand-edit `catalog.md`, `llm4vad.md`, `venues/README.md` or `docs/`. The build regenerates the three Markdown indexes from the data and writes the Vite application to `docs/`. Include all generated outputs in your change. CI verifies the generated catalog, tests, TypeScript and production build, then rejects differences in the committed generated outputs.

In a pull request, state which research claims changed, link their evidence, explain uncertainty, and report validation. Keep existing GitHub Pages deployment from the committed `docs/` folder; this contribution workflow does not change repository Pages settings or add a publishing workflow.
