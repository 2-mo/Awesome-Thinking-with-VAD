# Research method topology

The map keeps publication time horizontally and lets paper stations move vertically according to method topology. Conference names appear beneath each paper name, not on an axis. The former conference-ranked rows are superseded. WACV and workshop papers remain excluded. NeurIPS and Datasets and Benchmarks share a short display label; exact publication metadata remains in details and reading indexes.

## Connections and stations

A paper retains its primary `cluster` and may have `secondaryMethods: [{ cluster, evidence: { url, note } }]`. These are editorial classifications supported by specific method contributions, not citation or inheritance claims. Each paper appears once; every member line visits that same station. MemoVAD connects active decisions and temporal memory; A2Seek-R1 connects active observation and structured reasoning; TD-VAD connects semantic alignment and temporal modeling. Shared stations have a white, dark double-ring marker. Ordinary geometric crossings have a narrow paper-colored casing and no station. Hover and keyboard focus on a shared station emphasize all member lines. Method filtering includes secondary memberships; sources in paper details and the generated catalog explain their evidence.

## Computed layout

`src/components/publication-layout.ts` implements a deterministic, dependency-free staged heuristic:

1. Build the method adjacency graph from shared papers and the chronological paper chains within each method. Search all orderings of the five methods to minimize squared spans between connected lines. For more than seven methods, use bounded adjacent-swap improvement rather than factorial search.
2. Pack papers into hidden month groups. Independent lines can share an x position; papers on any common line occupy successive slots. Every earlier month stays to the left of every later month. Quarters follow these packed groups, not proportional calendar widths. Empty quarters retain narrow slots; unknown months remain outside Q1–Q4.
3. Initialize station heights from method order, averaging the anchors for shared stations. Apply 32 barycentric relaxation passes over neighboring papers, with soft method anchors. Stations therefore bend toward their actual connections without a fixed vertical-axis meaning.
4. Place two-line name/venue labels above or below stations, reserving nearby nodes and previously packed labels. Transfers are placed first. Labels remain in their year; space is allocated for long names.
5. Route with horizontal, vertical and 45-degree segments, avoiding labels and unrelated stations. Costs penalize length, bends, incidental crossings and shared/closely parallel segments. All paths stay within map bounds, move forward in time and leave stations toward the right to prevent vertical retracing. The SVG renderer rounds corners without cutting through labels.

This is an aesthetic heuristic, not a claim of a globally optimal embedding or scientific distance. Conference changes cannot alter station coordinates. Input order is normalized, so catalog reordering cannot change the result. Search and filtering use the full-catalog layout.

## Presentation

Years and recent quarters remain on the top axis. Paper names use 20-unit type, conference names 14-unit type, with no explanatory text in the map. The five colored method names form a compact bottom legend. Ordinary stations use colored rings; transfers use larger double rings. Full title, venue track and method evidence are available in the detail overlay. Every paper remains visible in the single-screen overview; pan/zoom and SVG export are retained.

Visual vocabulary is informed by the user-provided Agile Alliance subway map and the [official TfL Tube map](https://tfl.gov.uk/cdn/static/cms/documents/standard-tube-map.pdf): long trunks, rounded bends, labeled paper stations and actual shared-method connections. No reference artwork, marks or geometry is copied. All runtime drawing and labels are vector-based and included in SVG export. Historical raster backdrops remain unused source assets.
