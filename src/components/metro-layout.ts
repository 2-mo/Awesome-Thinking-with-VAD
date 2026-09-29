import type { Paper } from "../types";

export type Point = { x: number; y: number };
export type Station = Point & { below: boolean };
export type MetroLine = {
  id: string;
  number: string;
  color: string;
  track: Point[];
  slots: Station[];
};
const points = (pairs: number[][]): Point[] =>
  pairs.map(([x, y]) => ({ x, y }));
const slots = (triples: number[][]): Station[] =>
  triples.map(([x, y, below = 0]) => ({ x, y, below: !!below }));

// Schematic editorial routes, not citations or a shared time axis.
// Stations remain on horizontal track segments; crossings are not interchanges.
export const METRO_LINES: MetroLine[] = [
  {
    id: "alignment",
    number: "01",
    color: "#267bad",
    track: points([
      [72, 184],
      [826, 184],
      [986, 344],
      [1528, 344],
    ]),
    slots: slots([
      [144, 184],
      [284, 184, 1],
      [424, 184],
      [564, 184],
      [704, 184, 1],
      [824, 184],
      [1160, 344],
      [1316, 344, 1],
      [1472, 344],
    ]),
  },
  {
    id: "explanation",
    number: "02",
    color: "#cf5845",
    track: points([
      [72, 328],
      [508, 328],
      [732, 104],
      [1528, 104],
    ]),
    slots: slots([
      [150, 328],
      [338, 328, 1],
      [896, 104],
      [990, 104, 1],
      [1160, 104, 1],
      [1330, 104],
      [1490, 104, 1],
    ]),
  },
  {
    id: "understanding",
    number: "03",
    color: "#8160ad",
    track: points([
      [72, 468],
      [406, 468],
      [594, 280],
      [1528, 280],
    ]),
    slots: slots([
      [150, 468],
      [330, 468, 1],
      [684, 280, 1],
      [856, 280, 1],
      [1080, 280],
      [1280, 280],
      [1480, 280],
    ]),
  },
  {
    id: "evidence",
    number: "04",
    color: "#b38216",
    track: points([
      [72, 606],
      [744, 606],
      [916, 434],
      [1528, 434],
    ]),
    slots: slots([
      [170, 606],
      [474, 606],
      [980, 434, 1],
      [1420, 434, 1],
    ]),
  },
  {
    id: "reasoning",
    number: "05",
    color: "#36816c",
    track: points([
      [72, 698],
      [1034, 698],
      [1178, 554],
      [1528, 554],
    ]),
    slots: slots([
      [144, 698],
      [284, 698, 1],
      [424, 698],
      [564, 698, 1],
      [704, 698],
      [844, 698, 1],
      [984, 698],
      [1218, 554],
      [1358, 554, 1],
      [1498, 554],
    ]),
  },
];

export const MAP_WIDTH = 1600;
export const MAP_HEIGHT = 780;
export const LABEL_WIDTH = 164;
export const LABEL_HEIGHT = 42;

export const labelWidth = (text: string, size: number) =>
  [...text].reduce(
    (width, char) =>
      width +
      size *
        (/[^\u0000-\u00ff]/.test(char)
          ? 1
          : /[MW@]/.test(char)
            ? 0.95
            : /[A-Z]/.test(char)
              ? 0.72
              : /[a-z0-9]/.test(char)
                ? 0.6
                : 0.4),
    0,
  );
export const publicationLabel = (paper: Paper) =>
  `${paper.venue.replace(/\bDatasets and Benchmarks\b/gi, "D&B").replace(/\s+Workshops?\b/gi, "W")} · ${paper.year}`;

export function labelBox(station: Station, paper: Paper) {
  const width = Math.min(
    LABEL_WIDTH,
    Math.max(
      112,
      labelWidth(paper.shortTitle, 16) + 16,
      labelWidth(publicationLabel(paper), 12.5) + 16,
    ),
  );
  return {
    x: Math.min(MAP_WIDTH - width - 12, Math.max(12, station.x - width / 2)),
    y: station.y + (station.below ? 18 : -57),
    width,
    height: LABEL_HEIGHT,
  };
}

export function createMetroStations(papers: Paper[]) {
  const stations = new Map<string, Station>();
  for (const line of METRO_LINES) {
    const members = papers
      .filter((p) => p.cluster === line.id)
      .sort(
        (a, b) => a.year - b.year || a.shortTitle.localeCompare(b.shortTitle),
      );
    if (members.length <= line.slots.length) {
      members.forEach((paper, i) => {
        const index =
          members.length < 2
            ? 0
            : Math.round((i * (line.slots.length - 1)) / (members.length - 1));
        stations.set(paper.id, line.slots[index]);
      });
    } else {
      // Additional catalog entries are distributed over the same safe horizontal
      // segments, never interpolated through diagonals or silently omitted.
      const runs: Station[][] = [];
      for (const station of line.slots) {
        const run = runs[runs.length - 1];
        if (run && run[0].y === station.y) run.push(station);
        else runs.push([station]);
      }
      let offset = 0;
      runs.forEach((run, index) => {
        const count =
          index === runs.length - 1
            ? members.length - offset
            : Math.round((members.length * run.length) / line.slots.length);
        for (let i = 0; i < count; i++) {
          const first = run[0],
            last = run[run.length - 1];
          stations.set(members[offset++].id, {
            x: first.x + ((last.x - first.x) * i) / Math.max(1, count - 1),
            y: first.y,
            below: i % 2 === 1,
          });
        }
      });
    }
  }
  return stations;
}
