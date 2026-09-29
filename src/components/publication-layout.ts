import type { Paper } from "../types";
import { publicationVenue } from "../publication.ts";

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
  "ICCV",
  "ECCV",
  "NeurIPS",
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
      const ux = b.x - a.x,
        uy = b.y - a.y;
      const vx = point.x - b.x,
        vy = point.y - b.y;
      if (Math.abs(ux * vy - uy * vx) < 0.001 && ux * vx + uy * vy >= 0)
        result.pop();
      else break;
    }
    result.push(point);
  }
  return result;
}

// Slab intersection works for diagonal tracks as well as horizontal/vertical ones.
function crosses(a: Point, b: Point, box: Box): boolean {
  let lo = 0,
    hi = 1;
  for (const [origin, delta, min, max] of [
    [a.x, b.x - a.x, box.x + 0.01, box.x + box.width - 0.01],
    [a.y, b.y - a.y, box.y + 0.01, box.y + box.height - 0.01],
  ]) {
    if (Math.abs(delta) < 0.001) {
      if (origin <= min || origin >= max) return false;
    } else {
      const t1 = (min - origin) / delta,
        t2 = (max - origin) / delta;
      lo = Math.max(lo, Math.min(t1, t2));
      hi = Math.min(hi, Math.max(t1, t2));
      if (lo >= hi) return false;
    }
  }
  return hi > 0 && lo < 1;
}
type Segment = { a: Point; b: Point };
const segments = (path: Point[]): Segment[] =>
  path.slice(1).map((b, i) => ({ a: path[i], b }));
const length = (a: Point, b: Point) => Math.hypot(b.x - a.x, b.y - a.y);

// Two shortest octilinear alternatives: put the diagonal first or last.
function elbows(a: Point, b: Point): Point[][] {
  const dx = b.x - a.x,
    dy = b.y - a.y;
  const d = Math.min(Math.abs(dx), Math.abs(dy));
  return [
    simplify([
      a,
      { x: a.x + Math.sign(dx) * d, y: a.y + Math.sign(dy) * d },
      b,
    ]),
    simplify([
      a,
      { x: b.x - Math.sign(dx) * d, y: b.y - Math.sign(dy) * d },
      b,
    ]),
  ];
}
function chamfer(path: Point[], radius: number): Point[] {
  const points = [path[0]];
  for (let i = 1; i < path.length - 1; i++) {
    const a = path[i - 1],
      b = path[i],
      c = path[i + 1];
    const r = Math.min(radius, length(a, b) / 2, length(b, c) / 2);
    if (r < 14) {
      points.push(b);
      continue;
    }
    points.push(
      { x: b.x + Math.sign(a.x - b.x) * r, y: b.y + Math.sign(a.y - b.y) * r },
      { x: b.x + Math.sign(c.x - b.x) * r, y: b.y + Math.sign(c.y - b.y) * r },
    );
  }
  return simplify([...points, path[path.length - 1]]);
}
function parallelPenalty(a: Point, b: Point, occupied: Segment[]): number {
  const size = length(a, b);
  if (!size) return 0;
  const ux = (b.x - a.x) / size,
    uy = (b.y - a.y) / size;
  let penalty = 0;
  for (const other of occupied) {
    const vx = other.b.x - other.a.x,
      vy = other.b.y - other.a.y;
    if (Math.abs(ux * vy - uy * vx) > 0.01) continue;
    const distance = Math.abs((other.a.x - a.x) * uy - (other.a.y - a.y) * ux);
    if (distance >= 12) continue;
    const p = (other.a.x - a.x) * ux + (other.a.y - a.y) * uy;
    const q = (other.b.x - a.x) * ux + (other.b.y - a.y) * uy;
    const overlap =
      Math.min(size, Math.max(p, q)) - Math.max(0, Math.min(p, q));
    if (overlap > 0) penalty += overlap * (12 - distance) * 30;
  }
  return penalty;
}

// Prefer a small vocabulary of long, decisive metro bends. Visibility search is
// only a fallback for crowded cells; its edges are native octilinear paths.
function route(
  start: Point,
  end: Point,
  obstacles: Box[],
  occupied: Segment[],
): Point[] {
  const clear = (path: Point[]) =>
    segments(path).every(
      ({ a, b }) => !obstacles.some((box) => crosses(a, b, box)),
    );
  const cost = (path: Point[]) =>
    segments(path).reduce(
      (sum, { a, b }) => sum + length(a, b) + parallelPenalty(a, b, occupied),
      0,
    ) +
    (path.length - 2) * 60;
  const candidates = elbows(start, end);
  const xs = new Set([start.x, end.x, (start.x + end.x) / 2]);
  const ys = new Set([start.y, end.y, (start.y + end.y) / 2]);
  for (const box of obstacles) {
    for (const offset of [0, 12, 24, 36, 48]) {
      xs.add(box.x - offset);
      xs.add(box.x + box.width + offset);
      ys.add(box.y - offset);
      ys.add(box.y + box.height + offset);
    }
  }
  for (const x of xs)
    for (const radius of [24, 48, 80])
      candidates.push(
        chamfer([start, { x, y: start.y }, { x, y: end.y }, end], radius),
      );
  for (const y of ys)
    for (const radius of [24, 48, 80])
      candidates.push(
        chamfer([start, { x: start.x, y }, { x: end.x, y }, end], radius),
      );
  let best: Point[] | undefined,
    bestCost = Infinity;
  for (const candidate of candidates) {
    const score = cost(candidate);
    if (score < bestCost && clear(candidate)) {
      best = candidate;
      bestCost = score;
    }
  }
  if (best) return best;

  const nodes = [
    start,
    end,
    ...obstacles.flatMap((box) => [
      { x: box.x, y: box.y },
      { x: box.x + box.width, y: box.y },
      { x: box.x, y: box.y + box.height },
      { x: box.x + box.width, y: box.y + box.height },
    ]),
  ];
  const scores = nodes.map(() => Infinity),
    done = new Set<number>();
  const previous = new Map<number, { node: number; path: Point[] }>();
  scores[0] = 0;
  while (done.size < nodes.length) {
    let current = -1,
      score = Infinity;
    nodes.forEach((_, i) => {
      if (!done.has(i) && scores[i] < score) {
        current = i;
        score = scores[i];
      }
    });
    if (current < 0) break;
    if (current === 1) {
      const parts: Point[][] = [];
      while (current !== 0) {
        const step = previous.get(current)!;
        parts.unshift(step.path.slice(1));
        current = step.node;
      }
      return simplify([start, ...parts.flat()]);
    }
    done.add(current);
    nodes.forEach((node, next) => {
      if (done.has(next) || same(nodes[current], node)) return;
      for (const path of elbows(nodes[current], node)) {
        const nextCost = score + cost(path) + 90;
        if (nextCost < scores[next] && clear(path)) {
          scores[next] = nextCost;
          previous.set(next, { node: current, path });
        }
      }
    });
  }
  // With finite rectangular obstacles the visibility graph has an exterior path.
  throw new Error("Unable to route a publication metro segment");
}

export function createPublicationLayout(papers: Paper[]): PublicationLayout {
  const yearValues = [...new Set(papers.map((p) => p.year))].sort(
    (a, b) => a - b,
  );
  const venueValues = [
    ...VENUES.filter((v) =>
      papers.some((p) => publicationVenue(p.venue) === v),
    ),
    ...[...new Set(papers.map((p) => publicationVenue(p.venue)))]
      .filter((v) => !VENUES.includes(v))
      .sort(),
  ];
  const margin = 224,
    top = 75;
  // Size each year from its busiest merged venue cell so labels fit in one
  // tier; sparse years give their space to the denser publication columns.
  const rawWidths = yearValues.map((year) =>
    Math.max(
      124,
      ...venueValues.map((venue) =>
        papers
          .filter((p) => p.year === year && publicationVenue(p.venue) === venue)
          .reduce((sum, p) => sum + labelWidth(p.shortTitle, 14) + 22, 44),
      ),
    ),
  );
  let x = margin;
  const years = yearValues.map((year, index) => {
    const width = rawWidths[index];
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
        .filter(
          (p) => publicationVenue(p.venue) === venue && p.year === year.year,
        )
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
      ) * 12;
    const tierHeight = 56 + methodSpread;
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
              44 +
              rowMethods(row).indexOf(paper.cluster) * 12,
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
      count: papers.filter((p) => publicationVenue(p.venue) === venue).length,
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
  const occupied: Segment[] = [];
  const lines = schools
    .map((school, schoolIndex): PublicationLine => {
      // Within a year the order is spatial, not a publication sequence. Enter
      // each column from its closer end to avoid repeated full-height returns.
      const ordered: Paper[] = [];
      for (const year of yearValues) {
        const members = papers
          .filter((p) => p.cluster === school.id && p.year === year)
          .sort(
            (a, b) =>
              stations.get(a.id)!.y - stations.get(b.id)!.y ||
              stations.get(a.id)!.x - stations.get(b.id)!.x,
          );
        const previous = ordered.length
          ? stations.get(ordered[ordered.length - 1].id)!
          : undefined;
        if (
          previous &&
          members.length > 1 &&
          length(previous, stations.get(members[members.length - 1].id)!) <
            length(previous, stations.get(members[0].id)!)
        )
          members.reverse();
        ordered.push(...members);
      }
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
        track.push(
          ...route(before, stop, [...labels, ...unrelated], occupied).slice(1),
        );
      });
      if (stops.length)
        track.push({
          x: stops[stops.length - 1].x + 10,
          y: stops[stops.length - 1].y,
        });
      occupied.push(...segments(track));
      return {
        ...school,
        track: simplify(track),
        labelPosition: { x: margin + schoolIndex * 265, y: y + 30 },
      };
    })
    .filter((line) => line.track.length);
  return {
    width: Math.max(1700, x + 38),
    height: y + 50,
    years,
    venues,
    stations,
    lines,
    plotBounds: { x: margin, y: top, width: x - margin, height: y - top },
  };
}
