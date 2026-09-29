import type { Paper } from "../types";

export type Point = { x: number; y: number };
export type Box = Point & { width: number; height: number };
export type Station = Point & { paperId: string; lineId: string; label: Box };
export type PublicationLine = {
  id: string;
  label: string;
  color: string;
  track: Point[];
  labelPosition: Point;
  labelAnchor?: "start" | "end";
};
export type PublicationLayout = {
  width: number;
  height: number;
  plotBounds: Box;
  years: { year: number; x: number; width: number; count: number }[];
  venues: {
    venue: string;
    label: string;
    y: number;
    height: number;
    count: number;
  }[];
  lines: PublicationLine[];
  stations: Map<string, Station>;
};

const SCHOOLS = [
  { id: "alignment", label: "语义对齐与融合", color: "#267bad" },
  { id: "explanation", label: "语言判据与提示优化", color: "#cf5845" },
  { id: "understanding", label: "时序分层与记忆", color: "#8160ad" },
  { id: "evidence", label: "主动观察与工具决策", color: "#b38216" },
  { id: "reasoning", label: "结构化推理与验证", color: "#438778" },
];
const VENUES = [
  "AAAI",
  "CVPR",
  "CVPR Workshops",
  "ICCV",
  "ECCV",
  "WACV",
  "NeurIPS",
  "NeurIPS Datasets and Benchmarks",
  "ICML",
  "ICLR",
  "ACM MM",
  "ACL",
  "IJCAI",
  "arXiv",
];
export const publicationLabel = (paper: Paper): string =>
  `${paper.venue === "NeurIPS Datasets and Benchmarks" ? "NeurIPS D&B" : paper.venue === "CVPR Workshops" ? "CVPRW" : paper.venue} · ${paper.year}`;
// Conservative glyph estimates keep packing deterministic before fonts load.
export function labelWidth(text: string, size = 14): number {
  return Math.ceil(
    [...text].reduce(
      (sum, char) =>
        sum +
        (/[^\x00-\x7f]/.test(char)
          ? 1
          : /[MW@]/.test(char)
            ? 0.91
            : /[ilI1 .\-/]/.test(char)
              ? 0.36
              : 0.64),
      0,
    ) *
      size +
      6,
  );
}
const same = (a: Point, b: Point) => a.x === b.x && a.y === b.y;
function simplify(points: Point[]): Point[] {
  const result: Point[] = [];
  for (const point of points) {
    if (result.length && same(result[result.length - 1], point)) continue;
    while (result.length > 1) {
      const a = result[result.length - 2],
        b = result[result.length - 1];
      if (
        (a.x === b.x &&
          b.x === point.x &&
          (b.y - a.y) * (point.y - b.y) >= 0) ||
        (a.y === b.y && b.y === point.y && (b.x - a.x) * (point.x - b.x) >= 0)
      )
        result.pop();
      else break;
    }
    result.push(point);
  }
  return result;
}
function crosses(a: Point, b: Point, box: Box): boolean {
  if (a.y === b.y)
    return (
      a.y > box.y &&
      a.y < box.y + box.height &&
      Math.max(a.x, b.x) > box.x &&
      Math.min(a.x, b.x) < box.x + box.width
    );
  return (
    a.x > box.x &&
    a.x < box.x + box.width &&
    Math.max(a.y, b.y) > box.y &&
    Math.min(a.y, b.y) < box.y + box.height
  );
}

// Orthogonal visibility routing protects every title and every unrelated stop.
// Tracks express method membership, never paper-to-paper citation relations.
function route(
  start: Point,
  end: Point,
  obstacles: Box[],
  gutter: number,
): Point[] {
  const clear = (a: Point, b: Point) =>
    !obstacles.some((box) => crosses(a, b, box));
  const preferred = [
    start,
    { x: gutter, y: start.y },
    { x: gutter, y: end.y },
    end,
  ];
  if (preferred.slice(1).every((point, i) => clear(preferred[i], point)))
    return simplify(preferred);
  const xs = [
    ...new Set([
      start.x,
      end.x,
      gutter,
      ...obstacles.flatMap((b) => [b.x, b.x + b.width]),
    ]),
  ].sort((a, b) => a - b);
  const ys = [
    ...new Set([
      start.y,
      end.y,
      ...obstacles.flatMap((b) => [b.y, b.y + b.height]),
    ]),
  ].sort((a, b) => a - b);
  const nx = xs.length,
    key = (x: number, y: number) => y * nx + x;
  const first = key(xs.indexOf(start.x), ys.indexOf(start.y));
  const last = key(xs.indexOf(end.x), ys.indexOf(end.y));
  const point = (id: number): Point => ({
    x: xs[id % nx],
    y: ys[Math.floor(id / nx)],
  });
  const scores = new Map<number, number>([[first, 0]]),
    previous = new Map<number, number>();
  const heap: { id: number; score: number }[] = [];
  const push = (item: { id: number; score: number }) => {
    heap.push(item);
    let i = heap.length - 1;
    while (i > 0) {
      const parent = (i - 1) >> 1;
      if (heap[parent].score <= item.score) break;
      heap[i] = heap[parent];
      i = parent;
    }
    heap[i] = item;
  };
  const pop = () => {
    const top = heap[0],
      tail = heap.pop()!;
    if (heap.length) {
      let i = 0;
      while (i * 2 + 1 < heap.length) {
        let child = i * 2 + 1;
        if (
          child + 1 < heap.length &&
          heap[child + 1].score < heap[child].score
        )
          child++;
        if (heap[child].score >= tail.score) break;
        heap[i] = heap[child];
        i = child;
      }
      heap[i] = tail;
    }
    return top;
  };
  const visited = new Set<number>();
  push({ id: first, score: 0 });
  while (heap.length) {
    const { id } = pop();
    if (visited.has(id)) continue;
    visited.add(id);
    if (id === last) {
      const path = [point(id)];
      let cursor = id;
      while (previous.has(cursor)) {
        cursor = previous.get(cursor)!;
        path.push(point(cursor));
      }
      return simplify(path.reverse());
    }
    const x = id % nx,
      y = Math.floor(id / nx),
      a = point(id);
    const next = [
      x > 0 ? id - 1 : -1,
      x + 1 < nx ? id + 1 : -1,
      y > 0 ? id - nx : -1,
      y + 1 < ys.length ? id + nx : -1,
    ];
    for (const neighbor of next) {
      if (neighbor < 0 || visited.has(neighbor)) continue;
      const b = point(neighbor);
      if (!clear(a, b)) continue;
      const distance = Math.abs(a.x - b.x) + Math.abs(a.y - b.y);
      const cost = scores.get(id)! + distance + 0.05;
      if (cost >= (scores.get(neighbor) ?? Infinity)) continue;
      scores.set(neighbor, cost);
      previous.set(neighbor, id);
      push({
        id: neighbor,
        score: cost + Math.abs(b.x - end.x) + Math.abs(b.y - end.y),
      });
    }
  }
  return preferred;
}

export function createPublicationLayout(papers: Paper[]): PublicationLayout {
  const yearValues = [...new Set(papers.map((p) => p.year))].sort(
    (a, b) => a - b,
  );
  const venueValues = [
    ...VENUES.filter((v) => papers.some((p) => p.venue === v)),
    ...[...new Set(papers.map((p) => p.venue))]
      .filter((v) => !VENUES.includes(v))
      .sort(),
  ];
  const margin = 224,
    top = 75;
  const rawWidths = yearValues.map((year) => (year === 2023 ? 184 : 418));
  const scale =
    1438 /
    Math.max(
      1438,
      rawWidths.reduce((sum, w) => sum + w, 0),
    );
  let x = margin;
  const years = yearValues.map((year, index) => {
    const width = rawWidths[index] * scale;
    const item = {
      year,
      x,
      width,
      count: papers.filter((p) => p.year === year).length,
    };
    x += width;
    return item;
  });
  const stations = new Map<string, Station>();
  let y = top;
  const venues = venueValues.map((venue) => {
    const cells = years.map((year) => {
      const entries = papers
        .filter((p) => p.venue === venue && p.year === year.year)
        .sort((a, b) => a.shortTitle.localeCompare(b.shortTitle));
      const rows: Paper[][] = [[]];
      let used = 0;
      for (const paper of entries) {
        const width = labelWidth(paper.shortTitle, 14) + 22;
        if (used && used + width > year.width - 44) {
          rows.push([]);
          used = 0;
        }
        rows[rows.length - 1].push(paper);
        used += width;
      }
      return { year, rows };
    });
    const tierCount = Math.max(1, ...cells.map((c) => c.rows.length));
    const methodOrder = SCHOOLS.map((s) => s.id);
    const rowMethods = (row: Paper[]) =>
      [...new Set(row.map((p) => p.cluster))].sort(
        (a, b) => methodOrder.indexOf(a) - methodOrder.indexOf(b),
      );
    const methodSpread =
      Math.max(
        0,
        ...cells.flatMap((c) =>
          c.rows.map((row) => rowMethods(row).length - 1),
        ),
      ) * 7;
    const tierHeight = 47 + methodSpread;
    const height = tierCount * tierHeight + 2;
    for (const { year, rows } of cells)
      rows.forEach((row, tier) => {
        const total = row.reduce(
          (sum, p) => sum + labelWidth(p.shortTitle, 14) + 22,
          0,
        );
        let cursor = year.x + 22 + Math.max(0, (year.width - 44 - total) / 2);
        for (const paper of row) {
          const width = labelWidth(paper.shortTitle, 14) + 12;
          stations.set(paper.id, {
            paperId: paper.id,
            lineId: paper.cluster,
            x: cursor + 7,
            y:
              y +
              tier * tierHeight +
              36 +
              rowMethods(row).indexOf(paper.cluster) * 7,
            label: {
              x: cursor,
              y: y + tier * tierHeight + 7,
              width,
              height: 23,
            },
          });
          cursor += width + 10;
        }
      });
    const item = {
      venue,
      label: venue,
      y,
      height,
      count: papers.filter((p) => p.venue === venue).length,
    };
    y += height;
    return item;
  });
  const schools = [
    ...SCHOOLS,
    ...[...new Set(papers.map((p) => p.cluster))]
      .filter((id) => !SCHOOLS.some((s) => s.id === id))
      .map((id) => ({ id, label: id, color: "#617282" })),
  ];
  const labels: Box[] = [...stations.values()].map((s) => ({
    x: s.label.x - 5,
    y: s.label.y - 3,
    width: s.label.width + 10,
    height: s.label.height + 6,
  }));
  const lines = schools
    .map((school, schoolIndex): PublicationLine => {
      const ordered = papers
        .filter((p) => p.cluster === school.id)
        .sort(
          (a, b) =>
            a.year - b.year ||
            stations.get(a.id)!.y - stations.get(b.id)!.y ||
            stations.get(a.id)!.x - stations.get(b.id)!.x,
        );
      const stops = ordered.map((p) => stations.get(p.id)!);
      const track: Point[] = [];
      stops.forEach((stop, i) => {
        if (!i) {
          track.push({ x: stop.x - 10, y: stop.y }, { x: stop.x, y: stop.y });
          return;
        }
        const before = stops[i - 1];
        const unrelated = [...stations.values()]
          .filter(
            (s) => s.paperId !== stop.paperId && s.paperId !== before.paperId,
          )
          .map((s) => ({ x: s.x - 8, y: s.y - 8, width: 16, height: 16 }));
        const zone = years.find((z) => z.year === ordered[i - 1].year)!;
        const sameYear = ordered[i - 1].year === ordered[i].year;
        const gutter =
          sameYear && i % 2 === 0
            ? zone.x + 8 + schoolIndex * 8
            : zone.x + zone.width - 40 + schoolIndex * 8;
        track.push(
          ...route(before, stop, [...labels, ...unrelated], gutter).slice(1),
        );
      });
      if (stops.length)
        track.push({
          x: stops[stops.length - 1].x + 10,
          y: stops[stops.length - 1].y,
        });
      return {
        ...school,
        track: simplify(track),
        labelPosition: { x: margin + schoolIndex * 265, y: y + 30 },
      };
    })
    .filter((line) => line.track.length);
  return {
    width: Math.max(1700, x + 38),
    height: y + 64,
    years,
    venues,
    stations,
    lines,
    plotBounds: { x: margin, y: top, width: x - margin, height: y - top },
  };
}
