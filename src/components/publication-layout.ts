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
  years: {
    year: number;
    x: number;
    width: number;
    count: number;
    quarters: { quarter: number | null; x: number; width: number; count: number }[];
  }[];
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
// Preserve the familiar order only as a deterministic tie-breaker.
const VENUE_TIE_ORDER = [
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
// Combined benchmark/model titles keep their model name at the map station;
// the complete title remains in the index, accessible name and detail panel.
export const stationName = (paper: Paper): string =>
  paper.shortTitle.split(" / ").at(-1)!;
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

// Time never runs backwards: every candidate and visibility edge must move
// rightwards. This rules out the tiny U-turns caused by label avoidance.
function route(
  start: Point,
  end: Point,
  obstacles: Box[],
  occupied: Segment[],
): Point[] {
  const forward = (path: Point[]) =>
    segments(path).every(({ a, b }) => b.x >= a.x);
  const clear = (path: Point[]) =>
    forward(path) &&
    // Move right immediately when leaving a station, so a route arriving
    // vertically cannot double back along the same segment at a peak/valley.
    (!same(path[0], start) || path[1]?.x > start.x) &&
    segments(path).every(
      ({ a, b }) => !obstacles.some((box) => crosses(a, b, box)),
    );
  const cost = (path: Point[]) =>
    segments(path).reduce(
      (sum, { a, b }) => sum + length(a, b) + parallelPenalty(a, b, occupied),
      0,
    ) +
    (path.length - 2) * 100;
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
    ]).filter((point) => point.x >= start.x && point.x <= end.x),
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
      if (done.has(next) || same(nodes[current], node) || node.x < nodes[current].x) return;
      for (const path of elbows(nodes[current], node)) {
        const nextCost = score + cost(path) + 90;
        if (nextCost < scores[next] && clear(path)) {
          scores[next] = nextCost;
          previous.set(next, { node: current, path });
        }
      }
    });
  }
  throw new Error(`Unable to route a forward publication metro segment: ${JSON.stringify(start)} → ${JSON.stringify(end)}`);
}

export function createPublicationLayout(papers: Paper[]): PublicationLayout {
  // Equal-cost route candidates must not depend on catalog input order.
  papers = [...papers].sort((a, b) => a.id.localeCompare(b.id));
  const yearValues = [...new Set(papers.map((p) => p.year))].sort(
    (a, b) => a - b,
  );
  const venueCounts = new Map<string, number>();
  for (const paper of papers) {
    const venue = publicationVenue(paper.venue);
    venueCounts.set(venue, (venueCounts.get(venue) ?? 0) + 1);
  }
  const venueValues = [
    ...VENUE_TIE_ORDER.filter((venue) => venueCounts.has(venue)),
    ...[...venueCounts.keys()]
      .filter((venue) => !VENUE_TIE_ORDER.includes(venue))
      .sort(),
  ].sort((a, b) =>
    // Rank the full catalog's conference rows by volume; preprints stay last.
    Number(a === "arXiv") - Number(b === "arXiv") ||
    venueCounts.get(b)! - venueCounts.get(a)!,
  );
  const margin = 140,
    top = 99;
  const methodOrder = SCHOOLS.map((school) => school.id);
  const month = (paper: Paper) => paper.timeline?.month ?? 13;
  const compare = (a: Paper, b: Paper) =>
    a.year - b.year || month(a) - month(b) ||
    methodOrder.indexOf(a.cluster) - methodOrder.indexOf(b.cluster) ||
    a.shortTitle.localeCompare(b.shortTitle) || a.id.localeCompare(b.id);
  const stationXs = new Map<string, number>();
  let x = margin;
  // Hidden month groups establish a consistent order across *all* venue rows.
  // Equal-month papers spread into short slots; widths are schematic, not a
  // proportional calendar. Dense labels stack instead of stretching the map.
  const years = yearValues.map((year) => {
    const members = papers.filter((paper) => paper.year === year);
    const months = [...new Set(members.map(month))].sort((a, b) => a - b);
    let cursor = x + 24;
    let right = cursor;
    const packMonth = (value: number) => {
      const group = members.filter((paper) => month(paper) === value);
      const entries = group.sort((a, b) =>
        venueValues.indexOf(publicationVenue(a.venue)) - venueValues.indexOf(publicationVenue(b.venue)) || compare(a, b));
      // Leave room for left-facing labels at the year edge. Clamping those
      // labels to the edge would otherwise block a route climbing to a new row.
      cursor = Math.max(cursor, ...entries.map((paper, index) =>
        x + labelWidth(stationName(paper)) + 15 - index * 58,
      ));
      const span = Math.max(0, entries.length - 1) * 58;
      entries.forEach((paper, index) => {
        const stationX = cursor + index * 58;
        stationXs.set(paper.id, stationX);
        right = Math.max(right, stationX + labelWidth(stationName(paper)) + 12);
      });
      cursor += span + 44;
    };
    const quarters: PublicationLayout["years"][number]["quarters"] = [];
    if (year >= 2025) {
      // Quarter dividers follow the packed month groups, not equal-width
      // calendar slices. Reserve a small slot even when a quarter has no papers.
      const values: (number | null)[] = [1, 2, 3, 4];
      if (months.includes(13)) values.push(null);
      for (const quarter of values) {
        const left = quarters.length ? cursor - 22 : x;
        const quarterMonths = months.filter((value) =>
          quarter === null ? value === 13 : value <= 12 && Math.ceil(value / 3) === quarter,
        );
        quarterMonths.forEach(packMonth);
        if (!quarterMonths.length) cursor = left + 52 + 22;
        quarters.push({
          quarter,
          x: left,
          width: cursor - 22 - left,
          count: members.filter((paper) => quarterMonths.includes(month(paper))).length,
        });
      }
    } else {
      months.forEach(packMonth);
    }
    const width = Math.max(96, cursor - x + 12, right - x + 22);
    if (quarters.length) quarters.at(-1)!.width = x + width - quarters.at(-1)!.x;
    const item = { year, x, width, count: members.length, quarters };
    x += width;
    return item;
  });
  const labelXs = new Map<string, number>();
  const labelsBelow = new Set<string>();
  for (const paper of papers) {
    const stationX = stationXs.get(paper.id)!;
    const route = papers.filter((other) => other.cluster === paper.cluster)
      .sort((a, b) => stationXs.get(a.id)! - stationXs.get(b.id)!);
    const index = route.findIndex((other) => other.id === paper.id);
    const previous = route[index - 1], next = route[index + 1];
    const row = (item: Paper) => venueValues.indexOf(publicationVenue(item.venue));
    const leavesUp = next && row(next) < row(paper);
    const valley = previous && next && row(previous) < row(paper) && leavesUp;
    if (valley) labelsBelow.add(paper.id);
    const year = years.find((column) => column.year === paper.year)!;
    labelXs.set(paper.id, Math.max(year.x + 10,
      leavesUp && !valley ? stationX - labelWidth(stationName(paper)) - 5 : stationX - 7));
  }
  const stations = new Map<string, Station>();
  let y = top;
  const venues = venueValues.map((venue) => {
    const members = papers.filter((paper) => publicationVenue(paper.venue) === venue);
    const cells = years.map((year) => {
      const entries = members.filter((paper) => paper.year === year.year);
      const methods = [...new Set(entries.map((paper) => paper.cluster))]
        .sort((a, b) => methodOrder.indexOf(a) - methodOrder.indexOf(b));
      const tierStarts: number[][] = [[], []];
      const tiers = new Map<string, number>();
      // Later labels occupy the upper tier; valley labels sit below the track.
      for (const paper of [...entries].sort((a, b) => labelXs.get(b.id)! - labelXs.get(a.id)!)) {
        const left = labelXs.get(paper.id)!;
        const right = left + labelWidth(stationName(paper)) + 12;
        const starts = tierStarts[labelsBelow.has(paper.id) ? 1 : 0];
        let tier = starts.findIndex((start) => right + 10 <= start);
        if (tier < 0) tier = starts.length;
        starts[tier] = left;
        tiers.set(paper.id, tier);
      }
      const aboveHeight = tierStarts[0].length * 27;
      const belowHeight = tierStarts[1].length * 27;
      const trackHeight = 31 + Math.max(0, methods.length - 1) * 12;
      return { entries, methods, tiers, aboveHeight, trackHeight,
        height: aboveHeight + trackHeight + belowHeight };
    });
    const height = Math.max(58, ...cells.map((cell) => cell.height));
    for (const cell of cells) {
      const base = y + (height - cell.height) / 2;
      for (const paper of cell.entries) {
        stations.set(paper.id, {
          paperId: paper.id,
          lineId: paper.cluster,
          x: stationXs.get(paper.id)!,
          y: base + cell.aboveHeight + 19 + cell.methods.indexOf(paper.cluster) * 12,
          label: {
            x: labelXs.get(paper.id)!,
            y: base + 4 + (labelsBelow.has(paper.id) ? cell.aboveHeight + cell.trackHeight : 0) + cell.tiers.get(paper.id)! * 27,
            width: labelWidth(stationName(paper)) + 12,
            height: 23,
          },
        });
      }
    }
    const item = { venue, label: venue, y, height, count: members.length };
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
      const ordered = papers
        .filter((paper) => paper.cluster === school.id)
        .sort((a, b) => stations.get(a.id)!.x - stations.get(b.id)!.x);
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
