# Metro geometry references

Style research, 2026-09-30. Only the drawing vocabulary is adapted; no map artwork, station identities, logos or topology are copied into the website.

| Official reference | Observed visual qualities | Application to the research map |
| --- | --- | --- |
| [Hong Kong MTR system map](https://www.mtr.com.hk/archive/en/services/routemap.pdf) | Long horizontal/vertical trunks, occasional diagonals, bends concentrated between straight station sequences, clearly separated parallel lines | Primary direction: align station runs, place bends away from station centers and keep a consistent local corner radius |
| [Singapore MRT/LRT system map](https://www.lta.gov.sg/content/dam/ltagov/getting_around/public_transport/rail_network/pdf/SM_Eng_%28Ver280225%29_Hume.pdf) | Distinct line colors, compact multi-line interchange labels, diagonal corridors and large circular forms | Retain distinct line colors and shared-station identity; do not introduce geographic rings or loops into the time axis |

Both reference PDFs were rendered and visually inspected. Additional official reference entry points found during the search: [London TfL](https://tfl.gov.uk/maps/track/tube) and [Tokyo Metro](https://www.tokyometro.jp/en/subwaymap/index.html).

## Geometry choices

- Keep the existing five method colors, time axis, compact conference labels and three sourced shared stations.
- Regularize free station heights on a 24-unit grid, suppressing minor deviations from a method trunk. This removes small ripples caused by continuous barycentric relaxation.
- Approach and leave stations on short horizontal platforms. The current full catalog has straight segments extending at least 10 units either side of all 54 line visits to its 51 stations.
- Use true circular fillets, normally radius 12, at 45°/90° bends. The older renderer used a quadratic curve with a trim distance as large as 28, whose apparent radius varied with turning angle.
- Leave at least 14 units of straight track between adjacent fillets; shrink or omit a fillet if a short segment or a label requires it. Keep station centers on their route.
- Maintain label/node clearance, forward chronology and full-catalog filter stability. This is a local aesthetic heuristic, not a globally optimal subway drawing.

For the current catalog, simplified route skeletons have 27 bends, down from 37 before this adjustment. These are geometry checks; the application was not repeatedly opened for visual inspection.
