# Research transit map

The map uses a transit schematic: five numbered innovation lines, one station per paper, chronological order along each line, and publication metadata directly beside each station. Line bends use horizontal and 45-degree segments. The 37-paper overview uses restrained warm paper colors, no raster backdrop, no duplicated sidebars, and no pagination.

Visual reference: [Transport for London — official Tube map](https://tfl.gov.uk/cdn/static/cms/documents/standard-tube-map.pdf), consulted on 2026-09-30. This informed the colored-line, station-marker and separate-label vocabulary. No TfL artwork, marks, fonts or map geometry is copied into the application.

Geometry is authored in `src/components/metro-layout.ts`, separate from the scientific catalog. Positions remain stable while filtering. Paper ordering uses publication year and then name; this is an editorial reading order, not a citation, influence or inheritance claim. Crossings have a paper-colored casing and no interchange symbol. Data-backed relationships remain in the details panel.

The route layout is tuned for the current 37 papers. Additional papers have a horizontal-segment placement fallback, but publication growth should be accompanied by an editorial spacing review. The map and exported SVG use live text and vector paths. The older generated panorama is retained as a historical design asset and is no longer loaded by the map.
