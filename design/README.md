# Comic atlas visual direction

Generated with the built-in `image_gen` tool. The tool does not expose a model selector, so this is not a claim that GPT Image 2.5 was used.

`comic-atlas-concept.png` is an exploratory visual reference, not a source of research facts. Its illustrative text, dates and groupings are not the catalog. The implemented interface uses the verified `data/catalog.json` and organizes papers by innovation mechanism.

The initial concept includes ideas that were later removed: cross-domain PI content and the Holmes-VAU featured card. Neither is part of the final interface. Publication venue and readable type take priority over fitting every panel into a small viewport.

## Concept prompt

```text
Use case: ui-mockup
Asset type: high-fidelity desktop web application visual concept, one complete 16:10 screen.
Primary request: Redesign a real academic Video Anomaly Understanding research atlas into a delightful, dense, comic-book-inspired research workbench. Chinese interface with precise small headings and real research paper labels. It must look like a usable scholarly application, packed with organized information, no giant hero, no vertical or horizontal scroll bars, no decorative empty acreage.
Style: sophisticated European graphic novel / indie science comic printed on ivory paper. Confident black ink contours 2 px, square comic panels, warm butter yellow, periwinkle blue, coral red and mint accents, subtle halftone print dots only in margins, hard offset black shadows, expressive hand-drawn arrows and small speech bubbles. Playful but highly legible professional academic UI. NO 3D, no gradients, no glassmorphism, no generic SaaS dashboard.
Composition: exact edge-to-edge application screenshot without device or browser frame. Slim top masthead (8% height): small illustrated video-camera detective insignia, bold "VAU / IDEA ATLAS", sublabel "视频异常理解 · 创新路线图", compact stats "17 主线论文 · 5 创新方向", utility buttons. Slim secondary row search and toggles. Left rail 14% width, five numbered innovation filters with brief research questions, PI inspiration switch, compact year/task controls. Center 62% width: information-rich comic panel map with five numbered innovation regions and 17 legible paper tiles. Right 24% width: always-present fixed research reading panel, active paper "Holmes-VAU", concise problem/idea/evidence summary, three small tabs "思路 / 证据 / 来源", paper and code buttons, related works at bottom. Map+detail fill remaining viewport.
Center regions are based on innovation ideas, NEVER datasets. Region 01 "语义对齐" papers "TEVAD" and "VADCLIP"; Region 02 "语言解释与规则" papers "LAVAD", "AnomalyRuler", "VERA", "Ex-VAD"; Region 03 "显式推理与验证" papers "VADER", "SRVAU-R1", "Anom-π"; Region 04 "主动取证" papers "PANDA", "MoniTor", "EventVAD"; Region 05 "多粒度时序理解" papers "Holmes-VAU", "CUVA", "HAWK", "FineVAU", "VALU". Region arrangement follows staggered comic panels, an upper row and lower row. Small directional pencil arrows between conceptual regions labeled with research challenges (not scientific citation relations). Each paper tile contains year and short method phrase, color-coded tiny corner square, dense packing with disciplined alignment. Selected paper outlined in heavy black with yellow fill. Only small editorial illustrations occupy corners: video frame, magnifier, thought bubble, clock. No dataset nodes or dataset network. Compact footer strip legend explains "按创新机制组织 · 箭头为阅读路径". Fully legible typography, bold sans serif headings, regular Chinese body type.
The mockup should communicate information density, joyful comic character and innovation-driven navigation. Preserve research focus; avoid superheroes, anime characters, random labels, huge mascots, empty large cards and low-contrast text.
```
