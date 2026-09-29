import type { Paper } from "../types";
import { paperMethods, publicationVenue } from "../publication.ts";

export type Point = { x: number; y: number };
export type Box = Point & { width: number; height: number };
export type Station = Point & { paperId: string; lineId: string; lineIds: string[]; label: Box };
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
  methodOrder: string[];
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

function crossingPenalty(a: Point, b: Point, occupied: Segment[]): number {
  const dx = b.x - a.x, dy = b.y - a.y;
  let penalty = 0;
  for (const other of occupied) {
    const ex = other.b.x - other.a.x, ey = other.b.y - other.a.y;
    const cross = dx * ey - dy * ex;
    if (Math.abs(cross) < .001) continue;
    const ox = other.a.x - a.x, oy = other.a.y - a.y;
    const t = (ox * ey - oy * ex) / cross;
    const u = (ox * dy - oy * dx) / cross;
    // Shared station endpoints are connections, not incidental crossings.
    if (t > .001 && t < .999 && u > .001 && u < .999) penalty += 320;
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
  bounds: Box,
): Point[] {
  obstacles = obstacles.filter((box) => box.x < end.x && box.x + box.width > start.x);
  occupied = occupied.filter(({ a, b }) => a.x <= end.x && b.x >= start.x);
  const forward = (path: Point[]) =>
    segments(path).every(({ a, b }) => b.x >= a.x);
  const clear = (path: Point[]) =>
    forward(path) &&
    path.every((point) => point.x >= bounds.x && point.x <= bounds.x + bounds.width &&
      point.y >= bounds.y && point.y <= bounds.y + bounds.height) &&
    // Move right immediately when leaving a station, so a route arriving
    // vertically cannot double back along the same segment at a peak/valley.
    (!same(path[0], start) || path[1]?.x > start.x) &&
    segments(path).every(
      ({ a, b }) => !obstacles.some((box) => crosses(a, b, box)),
    );
  const cost = (path: Point[]) =>
    segments(path).reduce(
      (sum, { a, b }) => sum + length(a, b) + parallelPenalty(a, b, occupied) + crossingPenalty(a, b, occupied),
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

// Shared-method topology determines neighboring lines. Five methods need only
// 120 orderings; the fallback keeps future, larger catalogs bounded.
export function methodOrder(papers: Paper[], ids: string[]): string[] {
  const links = papers.flatMap((paper) => {
    const methods = paperMethods(paper);
    return methods.flatMap((a, i) => methods.slice(i + 1).map((b) => [a, b]));
  });
  const score = (order: string[]) => links.reduce((sum, [a, b]) =>
    sum + (order.indexOf(a) - order.indexOf(b)) ** 2, 0);
  let best = [...ids], bestScore = score(best);
  const visit = (prefix: string[], remaining: string[]) => {
    if (!remaining.length) {
      const value = score(prefix);
      if (value < bestScore) { best = prefix; bestScore = value; }
      return;
    }
    remaining.forEach((id, i) => visit([...prefix, id], remaining.filter((_, j) => i !== j)));
  };
  if (ids.length <= 7) visit([], ids);
  else {
    for (let pass = 0; pass < ids.length; pass++) {
      let improved = false;
      for (let i = 1; i < best.length; i++) {
        const candidate = [...best];
        [candidate[i - 1], candidate[i]] = [candidate[i], candidate[i - 1]];
        if (score(candidate) < bestScore) {
          best = candidate; bestScore = score(best); improved = true;
        }
      }
      if (!improved) break;
    }
  }
  return best;
}

const overlaps = (a: Box, b: Box, gap = 0) =>
  a.x < b.x + b.width + gap && a.x + a.width + gap > b.x &&
  a.y < b.y + b.height + gap && a.y + a.height + gap > b.y;

export function createPublicationLayout(papers: Paper[]): PublicationLayout {
  const month = (paper: Paper) => paper.timeline?.month ?? 13;
  papers = [...papers].sort((a, b) =>
    a.year - b.year || month(a) - month(b) || a.id.localeCompare(b.id));
  const ids = [...new Set(papers.flatMap(paperMethods))];
  const schools = [
    ...SCHOOLS.filter((school) => ids.includes(school.id)),
    ...ids.filter((id) => !SCHOOLS.some((school) => school.id === id)).sort()
      .map((id) => ({ id, label: id, color: "#617282" })),
  ];
  const order = methodOrder(papers, schools.map((school) => school.id));
  const rank = (paper: Paper) => paperMethods(paper)
    .reduce((sum, id) => sum + order.indexOf(id), 0) / paperMethods(paper).length;
  const margin = 24, top = 104, laneGap = 148;
  const stationXs = new Map<string, number>();
  const yearValues = [...new Set(papers.map((paper) => paper.year))];
  let x = margin;
  const years = yearValues.map((year) => {
    const members = papers.filter((paper) => paper.year === year);
    const months = [...new Set(members.map(month))].sort((a, b) => a - b);
    let cursor = x + 40;
    const packMonth = (value: number) => {
      const entries = members.filter((paper) => month(paper) === value)
        .sort((a, b) => rank(a) - rank(b) || a.id.localeCompare(b.id));
      const nextSlot = new Map<string, number>();
      let lastSlot = 0;
      for (const paper of entries) {
        const methods = paperMethods(paper);
        const slot = Math.max(0, ...methods.map((id) => nextSlot.get(id) ?? 0));
        stationXs.set(paper.id, cursor + slot * 64);
        methods.forEach((id) => nextSlot.set(id, slot + 1));
        lastSlot = Math.max(lastSlot, slot);
      }
      cursor += lastSlot * 64 + 72;
    };
    const quarters: PublicationLayout["years"][number]["quarters"] = [];
    if (year >= 2025) {
      const values: (number | null)[] = [1, 2, 3, 4];
      if (months.includes(13)) values.push(null);
      for (const quarter of values) {
        const left = quarters.length ? cursor - 36 : x;
        const quarterMonths = months.filter((value) =>
          quarter === null ? value === 13 : value <= 12 && Math.ceil(value / 3) === quarter);
        quarterMonths.forEach(packMonth);
        if (!quarterMonths.length) cursor = left + 48 + 36;
        quarters.push({ quarter, x: left, width: cursor - 36 - left,
          count: members.filter((paper) => quarterMonths.includes(month(paper))).length });
      }
    } else months.forEach(packMonth);
    const widestTail = Math.max(0, ...members.map((paper) =>
      stationXs.get(paper.id)! - x + labelWidth(stationName(paper), 20) / 2 + 16));
    const width = Math.max(140, cursor - x, widestTail, ...members.map((paper) =>
      Math.max(labelWidth(stationName(paper), 20), labelWidth(publicationVenue(paper.venue), 14)) + 22));
    if (quarters.length) quarters.at(-1)!.width = x + width - quarters.at(-1)!.x;
    const item = { year, x, width, count: members.length, quarters };
    x += width;
    return item;
  });

  const neighbors = new Map(papers.map((paper) => [paper.id, [] as string[]]));
  const lineMembers = new Map(schools.map((school) => {
    const members = papers.filter((paper) => paperMethods(paper).includes(school.id))
      .sort((a, b) => stationXs.get(a.id)! - stationXs.get(b.id)! || a.id.localeCompare(b.id));
    for (let i = 1; i < members.length; i++) {
      neighbors.get(members[i].id)!.push(members[i - 1].id);
      neighbors.get(members[i - 1].id)!.push(members[i].id);
    }
    return [school.id, members];
  }));
  const home = new Map(papers.map((paper) => [paper.id, top + 118 + rank(paper) * laneGap]));
  let ys = new Map(home);
  // Barycentric relaxation follows each paper's actual neighboring stations.
  // Soft method anchors preserve long trunks without imposing vertical rows.
  for (let iteration = 0; iteration < 32; iteration++) {
    const next = new Map<string, number>();
    for (const paper of papers) {
      const adjacent = neighbors.get(paper.id)!;
      const weight = paperMethods(paper).length > 1 ? 1.8 : 3.5;
      next.set(paper.id, (home.get(paper.id)! * weight +
        adjacent.reduce((sum, id) => sum + ys.get(id)!, 0)) / (weight + adjacent.length));
    }
    ys = next;
  }
  const plotBottom = top + Math.max(1, schools.length - 1) * laneGap + 244;
  const plotBounds = { x: margin, y: top, width: x - margin, height: plotBottom - top };
  const stations = new Map<string, Station>();
  for (const paper of papers) {
    stations.set(paper.id, { paperId: paper.id, lineId: paper.cluster,
      lineIds: paperMethods(paper), x: stationXs.get(paper.id)!,
      y: Math.round(ys.get(paper.id)! / 4) * 4,
      label: { x: 0, y: 0, width: Math.max(labelWidth(stationName(paper), 20),
        labelWidth(publicationVenue(paper.venue), 14)) + 14, height: 48 } });
  }
  // A label is kept near its paper, not in a conference band. Reserve stations
  // and already placed labels; high-degree transfer stations get first choice.
  const placed: Box[] = [];
  const nodes = [...stations.values()].map((station) => ({
    x: station.x - 15, y: station.y - 15, width: 30, height: 30 }));
  const labelOrder = [...papers].sort((a, b) => paperMethods(b).length - paperMethods(a).length ||
    stationXs.get(a.id)! - stationXs.get(b.id)! || a.id.localeCompare(b.id));
  for (const paper of labelOrder) {
    const station = stations.get(paper.id)!;
    const year = years.find((item) => item.year === paper.year)!;
    const candidates: { box: Box; score: number }[] = [];
    for (let tier = 0; tier < 6; tier++) {
      for (const side of [-1, 1]) {
        for (const align of [0, -1, 1]) {
          const box = { ...station.label,
            x: Math.max(year.x + 4, Math.min(year.x + year.width - station.label.width - 4,
              station.x - station.label.width / 2 + align * station.label.width / 2)),
            y: side < 0 ? station.y - 68 - tier * 56 : station.y + 20 + tier * 56 };
          if (box.y < top + 10 || box.y + box.height > plotBottom - 12 ||
              placed.some((other) => overlaps(box, other, 9)) ||
              nodes.some((node) => overlaps(box, node, 4))) continue;
          const drift = Math.abs(box.x + box.width / 2 - station.x);
          candidates.push({ box, score: tier * 100 + drift * .3 + (side > 0 ? 6 : 0) });
        }
      }
    }
    candidates.sort((a, b) => a.score - b.score);
    if (!candidates.length) throw new Error(`Unable to place topology label: ${paper.id}`);
    station.label = candidates[0].box;
    placed.push(station.label);
  }
  const labels = placed.map((box) => ({ x: box.x - 4, y: box.y - 4,
    width: box.width + 8, height: box.height + 8 }));
  const occupied: Segment[] = [];
  const lines = order.map((id): PublicationLine => {
    const school = schools.find((item) => item.id === id)!;
    const stops = lineMembers.get(id)!.map((paper) => stations.get(paper.id)!);
    const track: Point[] = [];
    stops.forEach((stop, i) => {
      if (!i) {
        track.push({ x: stop.x - 18, y: stop.y }, { x: stop.x, y: stop.y });
        return;
      }
      const before = stops[i - 1];
      const unrelated = [...stations.values()]
        .filter((station) => station.paperId !== stop.paperId && station.paperId !== before.paperId)
        .map((station) => ({ x: station.x - 12, y: station.y - 12, width: 24, height: 24 }));
      track.push(...route(before, stop, [...labels, ...unrelated], occupied, plotBounds).slice(1));
    });
    if (stops.length) track.push({ x: stops.at(-1)!.x + 18, y: stops.at(-1)!.y });
    const simplified = simplify(track);
    occupied.push(...segments(simplified));
    return { ...school, track: simplified,
      labelPosition: { x: 24 + order.indexOf(id) * 300, y: plotBottom + 32 } };
  });
  return { width: Math.max(1700, x + 24), height: plotBottom + 60,
    years, methodOrder: order, lines, stations,
    plotBounds };
}
