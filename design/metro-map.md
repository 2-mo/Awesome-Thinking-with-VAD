# Research publication network

The diagram has three explicit dimensions: publication year on the horizontal axis, displayed publication venue on the vertical axis, and five colored method-school routes through the papers. The current catalog supplies 32 stations, four year zones and 11 displayed venue categories. WACV and workshop papers are excluded from the curated paper catalog. NeurIPS and NeurIPS Datasets and Benchmarks share one NeurIPS row and publication-filter category; exact track metadata is preserved in the catalog and paper details. arXiv retains its own row.

Stations use compact paper names; the axes supply venue and year, and the interactive title and detail panel preserve full publication metadata. Route colors identify editorial method families. Crossings without a station do not represent method fusion, citation or inheritance. There are no invented interchange stations. Within each year, horizontal spacing serves label packing, not precise publication dates.

`src/components/publication-layout.ts` separates layout from scientific data. It packs full-catalog stations into year/venue bands, allocates distinct method baselines in dense cells, and routes around label boxes and unrelated paper nodes using horizontal, vertical and 45-degree segments. The renderer rounds bends and uses a narrow paper-colored casing to make crossings legible. Hover or keyboard focus emphasizes the corresponding method route. Search preserves the full-catalog geometry.

Visual vocabulary was informed by the [official Transport for London Tube map](https://tfl.gov.uk/cdn/static/cms/documents/standard-tube-map.pdf): colored lines, station markers and separate labels. No TfL artwork, fonts, marks or map geometry is copied. The earlier free-position innovation routes and generated panorama are superseded by this coordinate-based network; decorative source images remain historical design assets.

All drawing and text is vector-based and included in exported SVGs. The overview stays on one screen with pan/zoom and an on-demand detail overlay.
