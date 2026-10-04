import type { Cluster, Paper } from "../types";
import { clusterName, isMapPaper, paperMethods, publicationVenue, timelineYear } from "../publication.ts";
import { MAP_FONT_SIZE, mapTextWidth } from "./map-typography.ts";

export type Point = { x: number; y: number };
export type Box = Point & { width: number; height: number };
export type Platform = Point & { lineId: string };
export type Station = Point & {
  paperId: string; lineId: string; lineIds: string[]; platforms: Platform[]; label: Box;
  fork?: { parentId: string; branchId: string };
  // A returning branch meets its parent at one ordinary station.
  merge?: { parentId: string; branchId: string };
  // A two-direction paper with only one incoming and one outgoing rail uses
  // one ordinary marker. Its method memberships remain available in details.
  continuation?: boolean;
};
export const isInterchangeStation = (station: Station): boolean =>
  station.platforms.some(platform => platform.x !== station.x || platform.y !== station.y);
export function stationBounds(station: Station, padding = 12): Box {
  const top = Math.min(...station.platforms.map((platform) => platform.y));
  const bottom = Math.max(...station.platforms.map((platform) => platform.y));
  return { x: station.x - padding, y: top - padding, width: padding * 2, height: bottom - top + padding * 2 };
}
export type PublicationLine = {
  id: string;
  label: string;
  color: string;
  branchOf?: string;
  branchAt?: Cluster["branchAt"];
  routes?: Cluster["routes"];
  // The actual station sequences used to draw each track, including the
  // chronological fallback for directions without editorial route records.
  paperRoutes: string[][];
  tracks: Point[][];
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
  }[];
  methodOrder: string[];
  lines: PublicationLine[];
  junctions: (Point & { paperId: string; parentId: string; branchId: string })[];
  stations: Map<string, Station>;
};

export const publicationLabel = (paper: Paper): string =>
  `${paper.venue === "NeurIPS Datasets and Benchmarks" ? "NeurIPS D&B" : paper.venue === "NeurIPS Evaluations and Datasets" ? "NeurIPS E&D" : paper.venue === "CVPR Workshops" ? "CVPRW" : paper.venue} · ${paper.year}`;
// Combined benchmark/model titles keep their model name at the map station;
// the complete title remains in the index, accessible name and detail panel.
export const stationName = (paper: Paper): string => paper.shortTitle.split(" / ").at(-1)!;
export const STATION_NAME_SIZE = MAP_FONT_SIZE.primary;
export const STATION_ICON_SIZE = 22;
export const stationNameWidth = (paper: Paper): number => labelWidth(stationName(paper), STATION_NAME_SIZE);
export const stationVenue = (paper: Paper): string => publicationVenue(paper.venue);
export const stationVenueSize = (): number => MAP_FONT_SIZE.secondary;
// Reserve the same subtitle footprint across the catalog's venue styles, so
// changing publication metadata does not move a paper's date coordinate.
export const stationSubtitleWidth = (paper: Paper): number =>
  Math.max(labelWidth(stationVenue(paper), stationVenueSize()), labelWidth("NeurIPS", stationVenueSize()),
    labelWidth("ACM MM", stationVenueSize())) + (paper.mapIcon ? STATION_ICON_SIZE + 16 : 0);
// Use the same font metrics as the exported SVG, with a small safety margin.
export function labelWidth(text: string, size = 14): number {
  return mapTextWidth(text, size);
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

// Measure the whole name block, including its ends, against a nearby rail.
// Center-only measurements miss long names that almost touch another line.
export function railLabelDistance(box: Box, a: Point, b: Point): number {
  if (crosses(a, b, box)) return 0;
  const pointToBox = (p: Point) => Math.hypot(
    Math.max(box.x - p.x, p.x - box.x - box.width, 0),
    Math.max(box.y - p.y, p.y - box.y - box.height, 0));
  const dx = b.x - a.x, dy = b.y - a.y, squared = dx * dx + dy * dy;
  return Math.min(pointToBox(a), pointToBox(b), ...[
    { x: box.x, y: box.y }, { x: box.x + box.width, y: box.y },
    { x: box.x, y: box.y + box.height }, { x: box.x + box.width, y: box.y + box.height },
  ].map(p => {
    const t = squared ? Math.max(0, Math.min(1, ((p.x - a.x) * dx + (p.y - a.y) * dy) / squared)) : 0;
    return Math.hypot(p.x - a.x - t * dx, p.y - a.y - t * dy);
  }));
}

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

// A clean crossing has two continuous rails and gets SVG paper casing. A
// touching bend or overlapping segment would look like an unnamed branch.
function unnamedJoin(a: Point, b: Point, occupied: Segment[], connections: Point[]): boolean {
  const dx = b.x - a.x, dy = b.y - a.y;
  return occupied.some(other => {
    const ex = other.b.x - other.a.x, ey = other.b.y - other.a.y;
    const ox = other.a.x - a.x, oy = other.a.y - a.y;
    const cross = dx * ey - dy * ex;
    if (Math.abs(cross) < .001) {
      if (Math.abs(dx * oy - dy * ox) >= .001) return false;
      const size = dx * dx + dy * dy;
      if (!size) return false;
      const p = (ox * dx + oy * dy) / size;
      const q = ((other.b.x - a.x) * dx + (other.b.y - a.y) * dy) / size;
      return Math.min(1, Math.max(p, q)) - Math.max(0, Math.min(p, q)) > .001;
    }
    const t = (ox * ey - oy * ex) / cross, u = (ox * dy - oy * dx) / cross;
    if (t < -.001 || t > 1.001 || u < -.001 || u > 1.001 ||
      t > .001 && t < .999 && u > .001 && u < .999) return false;
    const point = { x: a.x + t * dx, y: a.y + t * dy };
    return !connections.some(connection => length(connection, point) < .001);
  });
}

// Time never runs backwards: every candidate and visibility edge must move
// rightwards. This rules out the tiny U-turns caused by label avoidance.
function routeBetween(
  start: Point,
  end: Point,
  obstacles: Box[],
  occupied: Segment[],
  bounds: Box,
  bendLate = false,
  connections: Point[] = [start, end],
  bendEarly = false,
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
    (!same(path[0], start) || path[1]?.x > start.x &&
      (!bendEarly || end.y === start.y || (path[1].y - start.y) * (end.y - start.y) > 0)) &&
    segments(path).every(
      ({ a, b }) => !obstacles.some((box) => crosses(a, b, box)) && !unnamedJoin(a, b, occupied, connections),
    );
  const cost = (path: Point[]) =>
    segments(path).reduce(
      (sum, { a, b }) => sum + length(a, b) + parallelPenalty(a, b, occupied) + crossingPenalty(a, b, occupied) +
        (bendLate ? Math.abs(b.y - a.y) * Math.max(0, end.x - (a.x + b.x) / 2) / Math.max(1, end.x - start.x) : 0) +
        (bendEarly ? Math.abs(b.y - a.y) * Math.max(0, (a.x + b.x) / 2 - start.x) / Math.max(1, end.x - start.x) * 4 : 0),
      0,
    ) +
    (path.length - 2) * 100;
  const dx = end.x - start.x, dy = Math.abs(end.y - start.y);
  if (dx > 0 && (dy < .001 || Math.abs(dx - dy) < .001) &&
      clear([start, end]) && parallelPenalty(start, end, occupied) === 0 &&
      crossingPenalty(start, end, occupied) === 0) return [start, end];
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

function route(start: Point, end: Point, obstacles: Box[], occupied: Segment[], bounds: Box,
  platforms: { start: number; end: number }, bendLate = false, bendEarly = false): Point[] {
  // Only shared stations need horizontal approaches to their separate
  // platforms. Ordinary stations may sit directly on a diagonal or its end.
  if (!platforms.start && !platforms.end) return routeBetween(start, end, obstacles, occupied, bounds, bendLate, [start, end], bendEarly);
  for (const scale of [1, .75]) {
    if (end.x - start.x <= scale * (platforms.start + platforms.end)) continue;
    const departure = { x: start.x + platforms.start * scale, y: start.y };
    const arrival = { x: end.x - platforms.end * scale, y: end.y };
    if (obstacles.some((box) => crosses(start, departure, box) || crosses(arrival, end, box))) continue;
    try {
      return simplify([start, ...routeBetween(departure, arrival, obstacles, occupied, bounds, bendLate, [start, end], bendEarly), end]);
    } catch {
      // A tight label corridor can require a shorter platform or a direct route.
    }
  }
  return routeBetween(start, end, obstacles, occupied, bounds, bendLate, [start, end], bendEarly);
}

// Shared-paper and parent/branch topology determine neighboring lines.
// Bound the search for larger catalogs.
export function methodOrder(papers: Paper[], ids: string[], branches: string[][] = [], nearby: string[][] = []): string[] {
  const links = papers.flatMap((paper) => {
    const methods = paperMethods(paper);
    return methods.flatMap((a, i) => methods.slice(i + 1).map((b) => [a, b]));
  // A fork's shared membership already contributes its parent/branch edge.
  // Counting it again as a transfer would overpower placement preferences.
  }).filter(pair => !branches.some(branch => branch.every(id => pair.includes(id))));
  const score = (order: string[]) => links.reduce((sum, [a, b]) =>
    sum + 2 * (order.indexOf(a) - order.indexOf(b)) ** 2, 0) + branches.reduce((sum, [a, b]) =>
    sum + (order.indexOf(a) - order.indexOf(b)) ** 2, 0) + nearby.reduce((sum, [a, b]) =>
    sum + 4 * (order.indexOf(a) - order.indexOf(b)) ** 2, 0) + nearby.reduce((sum, [a, b, side]) =>
    sum + (side === "above" && order.indexOf(a) > order.indexOf(b) ||
      side === "below" && order.indexOf(a) < order.indexOf(b) ? 400 : 0), 0);
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

type LabelCandidate = { box: Box; score: number };
// Choose neighboring labels together. Forward checking lets a long title give
// a compact position back to a constrained neighbor instead of pushing that
// neighbor several rows away and drawing a long leader to it.
function packLabels(options: Map<string, LabelCandidate[]>): Map<string, Box> {
  const result = new Map<string, Box>();
  let visits = 0;
  let blocked = "";
  const search = (remaining: Map<string, LabelCandidate[]>): boolean => {
    if (!remaining.size) return true;
    if (++visits > 12000) return false;
    const [id, candidates] = [...remaining].sort((a, b) =>
      a[1].length - b[1].length || a[0].localeCompare(b[0]))[0];
    if (!candidates.length) { blocked = id; return false; }
    for (const { box } of candidates) {
      const next = new Map([...remaining].filter(([key]) => key !== id).map(([key, values]) =>
        [key, values.filter(candidate => !overlaps(box, candidate.box, 8))]));
      const empty = [...next].find(([, values]) => !values.length);
      if (empty) { blocked = empty[0]; continue; }
      result.set(id, box);
      if (search(next)) return true;
      result.delete(id);
    }
    return false;
  };
  // Independent label neighborhoods cannot constrain each other. Solving
  // them separately prevents a tight fork from backtracking through dozens
  // of unrelated names on the other side of the map.
  const envelopes = new Map([...options].map(([id, candidates]) => {
    const boxes = candidates.map(candidate => candidate.box);
    const x = Math.min(...boxes.map(box => box.x)), y = Math.min(...boxes.map(box => box.y));
    return [id, { x, y, width: Math.max(...boxes.map(box => box.x + box.width)) - x,
      height: Math.max(...boxes.map(box => box.y + box.height)) - y }];
  }));
  const pending = new Set(options.keys());
  while (pending.size) {
    const group = [pending.values().next().value!];
    pending.delete(group[0]);
    for (const id of group) for (const other of pending) {
      if (!overlaps(envelopes.get(id)!, envelopes.get(other)!, 8)) continue;
      group.push(other); pending.delete(other);
    }
    visits = 0;
    if (!search(new Map(group.map(id => [id, options.get(id)!]))))
      throw new Error(`Unable to place adjacent topology labels near ${blocked} (${group.join(", ")})`);
  }
  return result;
}

export function createPublicationLayout(papers: Paper[], clusters: Cluster[] = [], options: { width?: number } = {}): PublicationLayout {
  const month = (paper: Paper) => paper.timeline?.month ?? 13;
  papers = papers.filter(isMapPaper).sort((a, b) =>
    timelineYear(a) - timelineYear(b) || month(a) - month(b) || a.id.localeCompare(b.id));
  const ids = [...new Set(papers.flatMap(paperMethods))];
  const schools = [
    ...clusters.filter((school) => ids.includes(school.id)).map(c => ({ ...c, label: clusterName(c) })),
    ...ids.filter((id) => !clusters.some((school) => school.id === id)).sort()
      .map((id) => ({ id, label: id, color: "#617282" })),
  ] as Pick<PublicationLine, "id" | "label" | "color" | "branchOf" | "branchAt" | "routes">[];
  const forks = new Map(schools.filter(s => s.branchOf && s.branchAt &&
    papers.some(p => p.id === s.branchAt!.paperId)).map(s => [s.branchAt!.paperId, s]));
  // Local return routes and one-way branches leave at an ordinary shared
  // station. A branch root does not need a separate interchange platform.
  for (const school of schools) for (const route of school.routes ?? []) {
    if (!school.branchOf || route.paperIds.length < 3) continue;
    const members = route.paperIds.map(id => papers.find(p => p.id === id));
    if (members.some(p => !p)) continue;
    const [root, terminal] = [members[0]!, members.at(-1)!];
    if (!paperMethods(root).includes(school.branchOf!) ||
      members.slice(1, -1).some(p => paperMethods(p!).length > 1)) continue;
    const returns = paperMethods(terminal).includes(school.branchOf!);
    const departureOnly = !school.routes?.some(other => other.paperIds.slice(1).includes(root.id));
    if (!returns && !departureOnly) continue;
    const trunk = schools.find(s => s.id === school.branchOf)?.routes?.find(r =>
      r.paperIds.includes(root.id) && (returns
        ? r.paperIds.indexOf(terminal.id) > r.paperIds.indexOf(root.id)
        : r.paperIds.indexOf(root.id) < r.paperIds.length - 1));
    if (!trunk) continue;
    forks.set(root.id, school);
  }
  const nearby = clusters.filter(c => ids.includes(c.id) && c.layoutNear && ids.includes(c.layoutNear))
    .map(c => [c.id, c.layoutNear!, c.layoutSide ?? ""]);
  const order = methodOrder(papers, schools.map((school) => school.id),
    schools.filter(s => s.branchOf && ids.includes(s.branchOf)).map(s => [s.id, s.branchOf!]), nearby);
  const rank = (paper: Paper) => forks.has(paper.id) ? order.indexOf(forks.get(paper.id)!.branchOf!) : paperMethods(paper)
    .reduce((sum, id) => sum + order.indexOf(id), 0) / paperMethods(paper).length;
  const margin = 24, top = 112, laneGap = 336, slotWidth = 72, origin = top + (ids.includes("synthesis") && ids.includes("alignment") ? 372 : 108);
  // Short branches share a tighter band with their neighbors. Dense trunks
  // retain the room needed by two-sided labels and interchange platforms.
  const compactBranch = (id: string) => schools.some(s => s.id === id && s.branchOf) &&
    papers.filter(paper => paperMethods(paper).includes(id)).length <= 10;
  // A branch explicitly grouped beside its parent needs room for transfer
  // approaches on the neighboring trunk. Other short branches stay compact.
  const groupedBranch = (id: string) => compactBranch(id) && nearby.some(pair => pair.includes(id));
  // Synthesis shares the blue family and occupies the shelf above its trunk.
  // Keep enough space between the remaining families for two-sided labels.
  const corridorGaps = new Map([
    ["synthesis:alignment", 240], ["alignment:understanding", 336],
    ["understanding:evidence", 384], ["evidence:evaluation", 264],
    ["evaluation:reasoning", 312], ["reasoning:explanation", 432],
  ]);
  const levels = new Map<string, number>();
  order.forEach((id, index) => levels.set(id, index === 0 ? 0 : levels.get(order[index - 1])! +
    (corridorGaps.get(`${order[index - 1]}:${id}`) ?? (compactBranch(id) || compactBranch(order[index - 1]) ?
      (groupedBranch(id) || groupedBranch(order[index - 1]) ? laneGap : 240) : laneGap))));
  // A direction does not reserve an empty horizontal band for every year.
  // Earlier trunks use the space before later branches become active; sparse
  // criteria segments sit closer to reasoning before spreading out in 2026.
  const heightProfiles: Record<string, [number, number][]> = {
    alignment: [[2023, -96], [2025.5, -96], [2026.5, 0], [2027, 0]],
    understanding: [[2023, -96], [2025.5, -96], [2026.5, 0], [2027, 0]],
    evidence: [[2023, -96], [2025.5, -96], [2026.5, 0], [2027, 0]],
    evaluation: [[2023, -144], [2025.5, -144], [2026.5, -48], [2027, 0]],
    reasoning: [[2023, -144], [2025.5, -144], [2026.5, -48], [2027, 0]],
    explanation: [[2023, -240], [2025.5, -288], [2026.5, -144], [2027, 0]],
  };
  const profileOffset = (id: string, paper: Paper): number => {
    const profile = heightProfiles[id];
    if (!profile) return 0;
    const minimum = -levels.get(id)!;
    const time = timelineYear(paper) + (month(paper) - 1) / 12;
    const next = profile.findIndex(([date]) => date >= time);
    if (next < 0) return Math.max(minimum, profile.at(-1)![1]);
    if (!next) return Math.max(minimum, profile[0][1]);
    const [a, b] = [profile[next - 1], profile[next]];
    return Math.max(minimum, a[1] + (b[1] - a[1]) * (time - a[0]) / (b[0] - a[0]));
  };
  const level = (paper: Paper) => {
    const methods = forks.has(paper.id) ? [forks.get(paper.id)!.branchOf!] : paperMethods(paper);
    return methods.reduce((sum, id) => sum + levels.get(id)! + profileOffset(id, paper), 0) / methods.length;
  };
  const stationXs = new Map<string, number>();
  // Seed the vertical topology in publication order. These provisional month
  // columns are relaxed independently after branch heights are established.
  const routeSlots = new Map(schools.map(school => [school.id,
    (school.routes ?? []).map((route, index) => ({ key: `${school.id}:${index}`, ids: route.paperIds }))]));
  const yearValues = [...new Set(papers.map(timelineYear))];
  let x = margin;
  const years = yearValues.map((year) => {
    const members = papers.filter((paper) => timelineYear(paper) === year);
    const months = [...new Set(members.map(month))].sort((a, b) => a - b);
    // Let the first date column carry centered names inside its year band.
    let cursor = x + Math.max(40, ...members.filter(paper => month(paper) === months[0])
      .map(paper => Math.max(stationNameWidth(paper), stationSubtitleWidth(paper)) / 2 + 12));
    const packMonth = (value: number) => {
      const pending = members.filter((paper) => month(paper) === value)
        .sort((a, b) => Number(forks.has(b.id)) - Number(forks.has(a.id)) || rank(a) - rank(b) || a.id.localeCompare(b.id));
      // A same-month transfer across a branch's band belongs before its fork:
      // otherwise the parent has to climb through the departing branch and
      // return again. Only the sourced fork must precede its own branch stops.
      const predecessors = new Map(pending.map(paper => [paper.id, new Set<string>()]));
      const forkPredecessors = new Map(pending.map(paper => [paper.id, new Set<string>()]));
      for (const paper of pending) {
        const fork = forks.get(paper.id);
        if (!fork) continue;
        const parentRank = order.indexOf(fork.branchOf!);
        const branchRank = order.indexOf(fork.id);
        for (const other of pending) {
          if (other.id === paper.id) continue;
          const methods = paperMethods(other);
          if (methods.includes(fork.id)) {
            predecessors.get(other.id)!.add(paper.id);
            forkPredecessors.get(other.id)!.add(paper.id);
          }
          else if (methods.includes(fork.branchOf!) && methods.some(id => {
            const otherRank = order.indexOf(id);
            return (otherRank - branchRank) * (parentRank - branchRank) < 0;
          })) predecessors.get(paper.id)!.add(other.id);
        }
      }
      // A local branch starts at its listed paper, including same-month ties.
      for (const school of schools) for (const local of school.routes ?? []) {
        for (let i = 1; i < local.paperIds.length; i++) {
          const a = local.paperIds[i - 1], b = local.paperIds[i];
          if (predecessors.has(a) && predecessors.has(b)) {
            predecessors.get(b)!.add(a);
            forkPredecessors.get(b)!.add(a);
          }
        }
      }
      const entries: Paper[] = [];
      while (pending.length) {
        let next = pending.findIndex(paper => !predecessors.get(paper.id)!.size);
        // Conflicting transfer preferences are soft. The actual fork must
        // still precede its branch, even when several forks share a month.
        if (next < 0) next = pending.findIndex(paper => !forkPredecessors.get(paper.id)!.size);
        const [paper] = pending.splice(next < 0 ? 0 : next, 1);
        entries.push(paper);
        for (const ids of predecessors.values()) ids.delete(paper.id);
        for (const ids of forkPredecessors.values()) ids.delete(paper.id);
      }
      const nextSlot = new Map<string, number>();
      let lastSlot = 0;
      for (const paper of entries) {
        const methods = paperMethods(paper);
        // A linked-platform station also occupies the bands between its lines.
        // Other same-month papers must not land inside its connector.
        const indices = methods.map(id => order.indexOf(id));
        const occupiedMethods = order.slice(Math.min(...indices), Math.max(...indices) + 1);
        const occupiedSlots = occupiedMethods.flatMap(id => {
          const routes = routeSlots.get(id)!;
          const local = routes.filter(route => route.ids.includes(paper.id));
          return routes.length ? (local.length === 1 && methods.length === 1 ? local : routes).map(route => route.key) : [id];
        });
        const slot = Math.max(0, ...occupiedSlots.map((id) => nextSlot.get(id) ?? 0));
        stationXs.set(paper.id, cursor + slot * slotWidth);
        occupiedSlots.forEach((id) => nextSlot.set(id, slot + 1));
        // Reserve time-axis space after the actual fork paper, so an immediate
        // same-month branch can leave it diagonally rather than vertically.
        const fork = forks.get(paper.id);
        if (fork) {
          const keys = routeSlots.get(fork.id)!.map(route => route.key);
          for (const key of keys.length ? keys : [fork.id]) nextSlot.set(key, slot + Math.ceil((laneGap + 32) / slotWidth));
        }
        lastSlot = Math.max(lastSlot, slot);
      }
      cursor += lastSlot * slotWidth + 44;
    };
    months.forEach(packMonth);
    const widestTail = Math.max(0, ...members.map((paper) =>
      stationXs.get(paper.id)! - x + stationNameWidth(paper) * (forks.has(paper.id) ? 1 : .5) + 40));
    const width = Math.max(140, cursor - x, widestTail, ...members.map((paper) =>
      Math.max(stationNameWidth(paper), stationSubtitleWidth(paper)) + 22));
    const item = { year, x, width, count: members.length };
    x += width;
    return item;
  });

  const neighbors = new Map(papers.map((paper) => [paper.id, [] as string[]]));
  const lineMembers = new Map(schools.map(school => [school.id,
    papers.filter(paper => paperMethods(paper).includes(school.id))
      .sort((a, b) => stationXs.get(a.id)! - stationXs.get(b.id)! || a.id.localeCompare(b.id)),
  ]));
  const routeMembers = new Map(schools.map(school => {
    const members = lineMembers.get(school.id)!;
    const local = school.routes ? school.routes.map(route => route.paperIds
      .map(id => members.find(paper => paper.id === id)).filter((paper): paper is Paper => !!paper))
      .filter(route => route.length) : [members];
    // New papers not yet assigned a reviewed route remain visible as roots.
    const covered = new Set(local.flat().map(paper => paper.id));
    local.push(...members.filter(paper => !covered.has(paper.id)).map(paper => [paper]));
    for (const route of local) for (let i = 1; i < route.length; i++) {
      neighbors.get(route[i].id)!.push(route[i - 1].id);
      neighbors.get(route[i - 1].id)!.push(route[i].id);
    }
    return [school.id, local];
  }));
  const continuations = new Set(papers.filter(paper => {
    const methods = paperMethods(paper);
    if (methods.length !== 2 || forks.has(paper.id)) return false;
    const degrees = methods.map(id => {
      const incoming = new Set<string>(), outgoing = new Set<string>();
      for (const route of routeMembers.get(id)!) {
        const index = route.findIndex(member => member.id === paper.id);
        if (index > 0) incoming.add(route[index - 1].id);
        if (index >= 0 && index < route.length - 1) outgoing.add(route[index + 1].id);
      }
      return { incoming: incoming.size, outgoing: outgoing.size };
    });
    return degrees.some(d => d.incoming === 1 && d.outgoing === 0) &&
      degrees.some(d => d.incoming === 0 && d.outgoing === 1);
  }).map(paper => paper.id));
  // Between the last transfer and a later fork, an as-yet inactive branch
  // does not need its full band. Lift ordinary parent/lower-line stops in
  // that interval. Shared papers are exempt from this direct height shift.
  const inactiveBands = schools.flatMap(school => {
    if (!school.branchOf || !school.branchAt || !stationXs.has(school.branchAt.paperId)) return [];
    const end = stationXs.get(school.branchAt.paperId)!;
    const previous = lineMembers.get(school.branchOf)?.filter(paper =>
      stationXs.get(paper.id)! < end - 320 && paperMethods(paper).length > 1 && !forks.has(paper.id)).at(-1);
    if (!previous) return [];
    const parentRank = order.indexOf(school.branchOf), branchRank = order.indexOf(school.id);
    if (branchRank !== parentRank - 1) return [];
    return [{ start: stationXs.get(previous.id)! + 80, end: end - 80, parentRank,
      amount: Math.max(0, Math.min(96, levels.get(school.branchOf)! - levels.get(school.id)! - 48)) }];
  });
  const anchorLevel = (paper: Paper) => level(paper) -
    (paperMethods(paper).length === 1 ? inactiveBands.filter(band =>
      stationXs.get(paper.id)! > band.start && stationXs.get(paper.id)! < band.end &&
      order.indexOf(paper.cluster) >= band.parentRank).reduce((sum, band) => sum + band.amount, 0) : 0);
  const home = new Map(papers.map((paper) => {
    return [paper.id, origin + anchorLevel(paper)];
  }));
  let ys = new Map(home);
  // Barycentric relaxation follows each paper's actual neighboring stations.
  // Soft method anchors preserve long trunks without imposing vertical rows.
  for (let iteration = 0; iteration < 32; iteration++) {
    const next = new Map<string, number>();
    for (const paper of papers) {
      const adjacent = neighbors.get(paper.id)!;
      const weight = paperMethods(paper).length > 1 && !forks.has(paper.id) ? 1.8 : 8;
      next.set(paper.id, (home.get(paper.id)! * weight +
        adjacent.reduce((sum, id) => sum + ys.get(id)!, 0)) / (weight + adjacent.length));
    }
    ys = next;
  }
  const plotBottom = origin + Math.max(laneGap, ...levels.values()) + 96;
  const plotBounds = { x: margin, y: top, width: x - margin, height: plotBottom - top };
  const stations = new Map<string, Station>();
  for (const paper of papers) {
    const relaxed = ys.get(paper.id)!;
    const anchored = (paperMethods(paper).length === 1 || forks.has(paper.id)) && Math.abs(relaxed - home.get(paper.id)!) < 24
      ? home.get(paper.id)! : relaxed;
    const stationX = stationXs.get(paper.id)!;
    const stationY = origin + Math.round((anchored - origin) / 24) * 24;
    const methods = paperMethods(paper);
    const fork = forks.get(paper.id);
    const continuation = continuations.has(paper.id);
    const platforms = [...methods].sort((a, b) => order.indexOf(a) - order.indexOf(b))
      .map((lineId, index) => ({ lineId, x: stationX, y: stationY + (fork || continuation ? 0 : (index - (methods.length - 1) / 2) * 32) }));
    stations.set(paper.id, { paperId: paper.id, lineId: paper.cluster,
      lineIds: methods, platforms, x: stationX, y: stationY,
      ...(fork ? { fork: { parentId: fork.branchOf!, branchId: fork.id } } : {}),
      ...(continuation ? { continuation: true } : {}),
      label: { x: 0, y: 0, width: Math.max(stationNameWidth(paper),
        stationSubtitleWidth(paper)) + 14, height: MAP_FONT_SIZE.primary + MAP_FONT_SIZE.secondary + 18 } });
  }
  // The 24-unit rail grid and month spacing need not have identical steps.
  // Fit nearby level runs to a true 45° connection before placing labels.
  // Move an entire ordinary run by at most one grid step, preserving its
  // horizontal neighbors and leaving dates, forks and interchanges fixed.
  const originalYs = new Map([...stations.values()].map(s => [s.paperId, s.y]));
  for (const members of lineMembers.values()) {
    const runs: Station[][] = [];
    for (const paper of members) {
      const station = stations.get(paper.id)!;
      const last = runs.at(-1);
      if (station.platforms.length === 1 && last?.at(-1)!.platforms.length === 1 &&
          station.y === last[0].y) last.push(station);
      else runs.push([station]);
    }
    for (let i = 1; i < runs.length; i++) {
      const a = runs[i - 1].at(-1)!, b = runs[i][0];
      if (a.platforms.length > 1 || b.platforms.length > 1) continue;
      const dx = b.x - a.x, dy = b.y - a.y;
      if (dx <= 0 || dx > 96 || Math.abs(dy) < 24 || Math.abs(Math.abs(dy) - dx) > 24) continue;
      const target = a.y + Math.sign(dy) * dx;
      if (runs[i].some(s => Math.abs(target - originalYs.get(s.paperId)!) > 24)) continue;
      runs[i].forEach(s => { s.y = target; s.platforms[0].y = target; });
    }
  }
  const moveStation = (station: Station, y: number) => {
    const offset = y - station.y;
    station.y = y;
    station.platforms.forEach(platform => { platform.y += offset; });
  };
  const labelSides = new Map<string, number>();
  const inlineStations = new Set<string>();
  // A short color section between two ordinary continuations belongs on
  // the through corridor. Membership alone must not create a branch or hump.
  const alignContinuations = () => {
    for (const school of schools) for (const members of routeMembers.get(school.id)!) {
      if (members.length !== 2 || members.some(p => !continuations.has(p.id))) continue;
      const first = stations.get(members[0].id)!, last = stations.get(members[1].id)!;
      const adjacent = schools.filter(other => other.id !== school.id).flatMap(other =>
        routeMembers.get(other.id)!.flatMap(route => {
          const a = route.findIndex(p => p.id === first.paperId), b = route.findIndex(p => p.id === last.paperId);
          return [a > 0 ? stations.get(route[a - 1].id)! : null,
            b >= 0 && b < route.length - 1 ? stations.get(route[b + 1].id)! : null]
            .filter((s): s is Station => !!s);
        }));
      if (adjacent.length !== 2) continue;
      adjacent.sort((a, b) => a.x - b.x);
      const y = adjacent[1].y;
      moveStation(first, y); moveStation(last, y);
      inlineStations.add(first.paperId); inlineStations.add(last.paperId);
      labelSides.set(first.paperId, 1); labelSides.set(last.paperId, -1);
      // A one-stop incoming lead can share this shelf too; it has no earlier
      // bend to preserve. This avoids two tiny steps around the color section.
      const lead = adjacent[0];
      const isLead = [...routeMembers.values()].some(routes => routes.some(route =>
        route.length === 2 && route[0].id === lead.paperId && route[1].id === first.paperId));
      if (isLead && lead.platforms.length === 1) moveStation(lead, y);
      // A longer entry can share the same through shelf too. Keep its ordinary
      // stops level instead of inserting a tiny height change before the join.
      for (const routes of routeMembers.values()) for (const route of routes) {
        if (route.at(-1)?.id !== first.paperId || route.slice(0, -1).some(p =>
          stations.get(p.id)!.platforms.length !== 1)) continue;
        route.slice(0, -1).forEach(p => moveStation(stations.get(p.id)!, y));
      }
    }
  };
  alignContinuations();
  // A branch that returns to its parent is a local parallel corridor. Keep
  // the trunk level through the return and use straight shelves for the
  // intermediate papers instead of alternating station-by-station heights.
  const parallelPairs = new Set<string>();
  const directDepartures = new Set<string>(), directArrivals = new Set<string>();
  const fanClearance = new Map<string, number>();
  const convergingPairs = new Set<string>();
  const levelCorridors = new Set<string>();
  for (const school of schools) for (const branch of routeMembers.get(school.id)!) {
    if (!school.branchOf || branch.length < 3 || forks.get(branch[0].id)?.id !== school.id) continue;
    const root = stations.get(branch[0].id)!, terminal = stations.get(branch.at(-1)!.id)!;
    if (!terminal.lineIds.includes(school.branchOf) || branch.slice(1, -1).some(p => paperMethods(p).length > 1)) continue;
    const parentRoutes = routeMembers.get(school.branchOf)!;
    const trunk = parentRoutes.find(route => route.findIndex(p => p.id === root.paperId) >= 0 &&
      route.findIndex(p => p.id === terminal.paperId) > route.findIndex(p => p.id === root.paperId));
    if (!trunk) continue;
    const start = trunk.findIndex(p => p.id === root.paperId), end = trunk.findIndex(p => p.id === terminal.paperId);
    const continuesBranch = routeMembers.get(school.id)!.some(route => {
      const index = route.findIndex(paper => paper.id === terminal.paperId);
      return index >= 0 && index < route.length - 1;
    });
    if (terminal.lineIds.length === 2 && !continuesBranch) {
      terminal.y = terminal.platforms.find(p => p.lineId === school.branchOf)!.y;
      for (const platform of terminal.platforms) platform.y = terminal.y;
      terminal.merge = { parentId: school.branchOf, branchId: school.id };
      directArrivals.add(`${school.id}:${branch.at(-2)!.id}:${terminal.paperId}`);
    }
    if (trunk.slice(start + 1, end).some(p => paperMethods(p).length > 1)) continue;
    let height = root.platforms.find(p => p.lineId === school.branchOf)!.y;
    const sideRoutes = parentRoutes.filter(route => route !== trunk && route[0]?.id === root.paperId &&
      route.at(-1)?.id === terminal.paperId && route.slice(1, -1).every(p => paperMethods(p).length === 1));
    if (sideRoutes.length > 1) continue;
    fanClearance.set(`${root.paperId}:${trunk[start + 1].id}`, 128 + stations.get(trunk[start + 1].id)!.label.width / 2 + 24);
    // Center a three-path fan between the neighboring routes, keeping room
    // outside both branches as well as inside the fan for station labels.
    if (sideRoutes.length) {
      const middle = (root.x + terminal.x) / 2;
      const neighbors = [...lineMembers.entries()].filter(([id]) => id !== school.id && id !== school.branchOf)
        .map(([, members]) => [...members].sort((a,b) => Math.abs(stations.get(a.id)!.x - middle) - Math.abs(stations.get(b.id)!.x - middle))[0])
        .map(paper => stations.get(paper.id)!.y);
      const above = Math.max(...neighbors.filter(y => y < height)), below = Math.min(...neighbors.filter(y => y > height));
      if (Number.isFinite(above) && Number.isFinite(below) && below - above >= 416) {
        height = Math.round((above + below) / 8) * 4;
        moveStation(root, height);
      }
    }
    const side = order.indexOf(school.id) < order.indexOf(school.branchOf) ? -1 : 1;
    // The returning fan stays flat only through its own return. Later papers
    // are free to use the vertical space beside the next junction.
    for (const paper of trunk.slice(start, end + 1)) levelCorridors.add(paper.id);
    for (const paper of trunk.slice(start + 1, end + 1)) {
      const station = stations.get(paper.id)!;
      moveStation(station, station.y + height - station.platforms.find(p => p.lineId === school.branchOf)!.y);
    }
    if (terminal.merge) {
      // Incoming arms occupy both sides of the return label, so give its
      // first continuing neighbor enough room for two adjacent names.
      const next = trunk[end + 1] && stations.get(trunk[end + 1].id);
      if (next) fanClearance.set(`${terminal.paperId}:${next.paperId}`,
        Math.ceil(((terminal.label.width + next.label.width) / 2 + 24) / 8) * 8);
    }
    const shelf = (members: Paper[], y: number, outward: number) => {
      for (const paper of members) { moveStation(stations.get(paper.id)!, y); labelSides.set(paper.id, outward); }
      for (let i = 1; i < members.length; i++) parallelPairs.add(`${members[i - 1].id}:${members[i].id}`);
    };
    shelf(branch.slice(1, -1), height + side * 192, side);
    for (const route of sideRoutes) {
      shelf(route.slice(1, -1), height - side * 192, -side);
      directDepartures.add(`${school.branchOf}:${root.paperId}:${route[1].id}`);
      directArrivals.add(`${school.branchOf}:${route.at(-2)!.id}:${terminal.paperId}`);
    }
  }
  // Same-color reading branches reuse named stations on a longer trunk.
  // Give their interiors a separate shelf on the roomier side, and leave
  // the trunk flat so the fork reads as two paths rather than a sharp peak.
  for (const [lineId, routes] of routeMembers) for (const branch of routes) {
    if (branch.length < 3 || branch.slice(1, -1).some(p => paperMethods(p).length !== 1)) continue;
    const root = stations.get(branch[0].id)!, terminal = stations.get(branch.at(-1)!.id)!;
    const trunk = routes.find(other => other !== branch && other.length > branch.length &&
      other.findIndex(p => p.id === root.paperId) >= 0 &&
      other.findIndex(p => p.id === terminal.paperId) > other.findIndex(p => p.id === root.paperId));
    if (!trunk || levelCorridors.has(root.paperId) && levelCorridors.has(terminal.paperId)) continue;
    const start = trunk.findIndex(p => p.id === root.paperId), end = trunk.findIndex(p => p.id === terminal.paperId);
    const middle = trunk.slice(start + 1, end);
    if (!middle.length || middle.some(p => paperMethods(p).length !== 1) ||
        branch.slice(1, -1).some(p => trunk.some(other => other.id === p.id))) continue;
    const rootHeight = root.platforms.find(p => p.lineId === lineId)!.y;
    const terminalHeight = terminal.platforms.find(p => p.lineId === lineId)!.y;
    const height = isInterchangeStation(terminal) ? Math.round((rootHeight + terminalHeight) / 48) * 24 : rootHeight;
    moveStation(root, root.y + height - rootHeight);
    moveStation(terminal, terminal.y + height - terminalHeight);
    const midpoint = (root.x + terminal.x) / 2;
    const neighbors = [...routeMembers.entries()].filter(([id]) => id !== lineId)
      .flatMap(([, localRoutes]) => localRoutes.flatMap(members => {
        // Disconnected reading segments do not occupy their empty time gap.
        if (stations.get(members[0].id)!.x > terminal.x || stations.get(members.at(-1)!.id)!.x < root.x) return [];
        // Include long passing rails whose endpoints fall outside this fan.
        const active = members.filter(p => !paperMethods(p).includes(lineId));
        active.sort((a, b) => Math.abs(stations.get(a.id)!.x - midpoint) - Math.abs(stations.get(b.id)!.x - midpoint));
        return active.length ? [stations.get(active[0].id)!.y] : [];
      }));
    const above = Math.max(top + 72, ...neighbors.filter(y => y < height));
    const below = Math.min(plotBottom - 72, ...neighbors.filter(y => y > height));
    // Keep a returning arm opposite the other line at its transfer, so the
    // two incoming approaches do not compete for the same diagonal corridor.
    const transferSide = terminal.platforms.find(p => p.lineId !== lineId && p.y !== height);
    const side = transferSide ? -Math.sign(transferSide.y - height)
      : below - height >= height - above ? 1 : -1;
    // A tighter blue return keeps same-month forks within the fixed canvas.
    const shelf = height + side * (lineId === "alignment" ? 208 : 216);
    for (const paper of middle) {
      moveStation(stations.get(paper.id)!, height);
      labelSides.set(paper.id, -side);
    }
    if (terminal.platforms.length === 1) moveStation(terminal, height);
    for (const paper of branch.slice(1, -1)) {
      moveStation(stations.get(paper.id)!, shelf);
      labelSides.set(paper.id, side);
    }
    for (const paper of [...trunk.slice(start, end + 1), ...branch]) levelCorridors.add(paper.id);
    for (const arm of [middle, branch.slice(1, -1)]) {
      for (let i = 1; i < arm.length; i++) parallelPairs.add(`${arm[i - 1].id}:${arm[i].id}`);
    }
    directDepartures.add(`${lineId}:${root.paperId}:${branch[1].id}`);
    directArrivals.add(`${lineId}:${branch.at(-2)!.id}:${terminal.paperId}`);
    convergingPairs.add(`${branch.at(-2)!.id}:${terminal.paperId}`);
    // Center labels at the two named junctions without crowding either arm.
    const lead = start > 0 ? stations.get(trunk[start - 1].id)! : undefined;
    if (lead) {
      const yearStart = years.find(year => year.year === timelineYear(trunk[start - 1]))!.x;
      if (lead.x - yearStart < lead.label.width / 2 + 4)
        fanClearance.set(`${lead.paperId}:${root.paperId}`,
          Math.max((lead.label.width + root.label.width) / 2 + 24, lead.label.width + 40));
    }
    fanClearance.set(`${root.paperId}:${middle[0].id}`, Math.max(root.label.width, stations.get(middle[0].id)!.label.width) * .6 + 24);
    fanClearance.set(`${middle.at(-1)!.id}:${terminal.paperId}`, Math.max(terminal.label.width, stations.get(middle.at(-1)!.id)!.label.width) * .6 + 24);
  }
  // A domain-specific reading arm may finish independently. It does not
  // need a decorative return into an unrelated method on the main route.
  for (const [lineId, routes] of routeMembers) for (const branch of routes) {
    if (branch.length < 3 || branch.slice(1).some(p => paperMethods(p).length !== 1)) continue;
    const root = stations.get(branch[0].id)!;
    const trunk = routes.find(other => other !== branch && other.some(p => p.id === root.paperId));
    if (!trunk || branch.slice(1).some(p => routes.some(other => other !== branch && other.some(q => q.id === p.id)))) continue;
    const height = root.platforms.find(p => p.lineId === lineId)!.y;
    const side = ["alignment", "detection", "understanding"].includes(lineId) ? -1 : 1;
    const offset = lineId === "explanation" ? 192 : 216;
    const shelf = height + side * offset;
    if (shelf + stations.get(branch.at(-1)!.id)!.label.height + 36 >= plotBottom) continue;
    for (const paper of branch.slice(1)) {
      moveStation(stations.get(paper.id)!, shelf);
      labelSides.set(paper.id, side);
      levelCorridors.add(paper.id);
    }
    for (let i = 2; i < branch.length; i++) parallelPairs.add(`${branch[i - 1].id}:${branch[i].id}`);
    directDepartures.add(`${lineId}:${root.paperId}:${branch[1].id}`);
    fanClearance.set(`${root.paperId}:${branch[1].id}`, offset + 48);
  }
  // Two independent entry routes meet at one ordinary station before the
  // through route. Keep their heads on opposite shelves and converge only
  // at the named paper, with no shared incoming stub or extra transfer mark.
  for (const routes of routeMembers.values()) for (const through of routes) {
    if (through.length < 2) continue;
    const terminal = stations.get(through[0].id)!;
    if (terminal.platforms.length !== 1) continue;
    const leads = routes.filter(route => route !== through && route.length === 2 &&
      route[1].id === terminal.paperId && stations.get(route[0].id)!.platforms.length === 1 &&
      !routes.some(other => other !== route && other.some(paper => paper.id === route[0].id)));
    if (leads.length !== 2) continue;
    leads.forEach((lead, index) => {
      const root = stations.get(lead[0].id)!, side = index === 0 ? -1 : 1;
      moveStation(root, terminal.y + side * 108);
      labelSides.set(root.paperId, side);
      fanClearance.set(`${root.paperId}:${terminal.paperId}`, 188);
      convergingPairs.add(`${root.paperId}:${terminal.paperId}`);
    });
  }
  // A short local offshoot should not make its trunk peak at the fork. Keep
  // that root on the incoming shelf, then align an interchange terminal with
  // its continuing route instead of adding another tiny departure bend.
  for (const [lineId, routes] of routeMembers) for (const members of routes) {
    if (members.length !== 2) continue;
    const root = stations.get(members[0].id)!, terminal = stations.get(members[1].id)!;
    const trunk = routes.find(other => other !== members && other.some(paper => paper.id === root.paperId));
    if (!trunk) continue;
    directDepartures.add(`${lineId}:${root.paperId}:${terminal.paperId}`);
    const index = trunk.findIndex(paper => paper.id === root.paperId);
    if (index > 0 && index < trunk.length - 1 && root.platforms.length === 1) {
      const before = stations.get(trunk[index - 1].id)!, after = stations.get(trunk[index + 1].id)!;
      const previousY = before.platforms.find(platform => platform.lineId === lineId)!.y;
      const nextY = after.platforms.find(platform => platform.lineId === lineId)!.y;
      if ((root.y - previousY) * (root.y - nextY) > 0) moveStation(root, previousY);
    }
    // An isolated one-stop branch needs a visible diagonal and a short level
    // run, so its endpoint reads as a branch rather than a displaced marker.
    if (terminal.platforms.length === 1 && !routes.some(other =>
      other !== members && other.some(paper => paper.id === terminal.paperId))) {
      const passingArm = routes.filter(other => other !== members &&
        !other.some(p => p.id === root.paperId) &&
        stations.get(other[0].id)!.x < root.x && stations.get(other.at(-1)!.id)!.x > root.x)
        .flatMap(other => other.slice(1, -1).map(p => stations.get(p.id)!))
        .filter(s => Math.abs(s.y - root.y) >= 72)
        .sort((a, b) => Math.abs(a.y - root.y) - Math.abs(b.y - root.y))[0];
      // A terminal spur belongs opposite a nearby returning arm. Otherwise
      // two same-color branches cross despite having no shared paper there.
      const side = passingArm ? -Math.sign(passingArm.y - root.y) : Math.sign(terminal.y - root.y) || -1;
      const rise = passingArm ? 168 : Math.max(120, Math.abs(terminal.y - root.y));
      moveStation(terminal, root.y + side * rise);
      fanClearance.set(`${root.paperId}:${terminal.paperId}`, Math.abs(terminal.y - root.y) + 80);
      labelSides.set(terminal.paperId, side);
    }
    for (const platform of terminal.platforms.filter(platform => platform.lineId !== lineId)) {
      const continuation = routeMembers.get(platform.lineId)?.find(other => other[0]?.id === terminal.paperId && other.length > 1);
      if (!continuation) continue;
      const next = stations.get(continuation[1].id)!.platforms.find(p => p.lineId === platform.lineId)!;
      moveStation(terminal, terminal.y + next.y - platform.y);
      break;
    }
  }
  // When an active third route passes between a transfer's method bands,
  // leave a broad level shoulder. Crossings then fall away from the shared
  // markers rather than forming a cramped triangular tip around them.
  const platformRoom = new Map([...stations.values()].map(station => {
    if (!isInterchangeStation(station)) return [station.paperId, 0];
    // A diagonal reading arm leaves directly from one platform. Keep the
    // neighboring platform's shoulder short enough that its parallel exit
    // stays visibly separate instead of hiding beneath the arm's SVG casing.
    if (station.platforms.length === 2 && station.lineIds.some(id =>
      [...directDepartures].some(key => key.startsWith(`${id}:${station.paperId}:`))))
      return [station.paperId, 12];
    const indices = station.lineIds.map(id => order.indexOf(id));
    const between = order.slice(Math.min(...indices) + 1, Math.max(...indices));
    const activeBetween = !station.fork && between.some(id =>
      routeMembers.get(id)?.some(members => members.some(p => stationXs.get(p.id)! < station.x) &&
        members.some(p => stationXs.get(p.id)! > station.x)));
    return [station.paperId, Math.max(activeBetween ? 80 : 16, station.label.width / 2 - 40)];
  }));
  // Keep a level passing route clear of the widened transfer shelf, including
  // its two markers. Adjust ordinary endpoints together to retain a flat run.
  for (const station of stations.values()) {
    if (platformRoom.get(station.paperId)! <= 16) continue;
    for (const [lineId, routes] of routeMembers) {
      if (station.lineIds.includes(lineId)) continue;
      for (const members of routes) for (let i = 1; i < members.length; i++) {
        const a = stations.get(members[i - 1].id)!, b = stations.get(members[i].id)!;
        if (a.x >= station.x || b.x <= station.x || a.platforms.length !== 1 ||
            b.platforms.length !== 1 || a.y !== b.y) continue;
        const nearest = station.platforms.reduce((best, p) => Math.abs(p.y - a.y) < Math.abs(best.y - a.y) ? p : best);
        const distance = a.y - nearest.y;
        if (Math.abs(distance) >= 64 || distance === 0) continue;
        const y = nearest.y + Math.sign(distance) * 64;
        moveStation(a, y); moveStation(b, y);
      }
    }
  }
  // Put ordinary runs on one clear shelf and stagger their names instead of
  // adding one-stop rail peaks. Named forks, returns and transfers stay fixed.
  for (const [lineId, routes] of routeMembers) {
    const repeated = new Set(routes.flat().filter((paper, index, all) =>
      all.findIndex(other => other.id === paper.id) !== index).map(paper => paper.id));
    for (const members of routes) {
      let run: Station[] = [];
      const placeRun = () => {
        if (run.length >= 2) {
          const first = members.findIndex(paper => paper.id === run[0].paperId);
          const before = first > 0 ? stations.get(members[first - 1].id) : undefined;
          const y = before?.continuation ? before.platforms.find(p => p.lineId === lineId)!.y
            : Math.round(run.reduce((sum, station) => sum + station.y, 0) / run.length / 24) * 24;
          run.forEach(station => moveStation(station, y));
        }
        run = [];
      };
      for (const paper of members) {
        const station = stations.get(paper.id)!;
        if (station.platforms.length !== 1 || repeated.has(paper.id) || inlineStations.has(paper.id) ||
            levelCorridors.has(paper.id)) { placeRun(); continue; }
        run.push(station);
      }
      placeRun();
    }
  }
  // Later reasoning stays on the Plus corridor. Only genuine transfers
  // leave this shelf; video reasoning papers continue on the same trunk.
  const reasoningShelf = stations.get("vad-r1-plus")?.platforms.find(p => p.lineId === "reasoning")?.y;
  if (reasoningShelf !== undefined) for (const id of ["srvau-r1", "adversa", "las-vad", "stch", "cg-coe", "clue-vad", "avar", "o-vad"]) {
    const station = stations.get(id), platform = station?.platforms.find(p => p.lineId === "reasoning");
    if (station && platform && !isInterchangeStation(station)) moveStation(station, station.y + reasoningShelf - platform.y);
  }
  // The sparse criteria route reads more clearly as level sections than as
  // alternating one-stop peaks. Preserve the shared AnomalyRuler platform
  // and let the later sections descend in a few deliberate, shallow steps.
  const criteriaAnchor = stations.get("anomalyruler")?.platforms.find(p => p.lineId === "explanation");
  if (criteriaAnchor) {
    const criteriaHasBranch = (routeMembers.get("explanation")?.length ?? 0) > 1;
    const criteriaBase = Math.min(criteriaAnchor.y, plotBottom - 432 - 80 - 48);
    for (const [offset, ids] of [
      [144, ["eval", "lavad", "log-sad", "vera", "promptvad"]],
      [240, ["lagovad", "lrpo", "prime-vad", "road"]],
      [criteriaHasBranch ? 432 : 240, ["probe-vad", "ca-judge"]],
    ] as const) for (const id of ids) {
      const station = stations.get(id);
      if (station?.lineIds.length === 1 && station.lineId === "explanation")
        moveStation(station, criteriaBase + offset);
    }
  }
  // The memory branch rises immediately; its continuing trunk keeps the
  // shared platform height instead of following it on a close parallel rail.
  const earlyMemory = stations.get("scene-dependent-vaa"), memoryFork = stations.get("holmes-vau");
  if (earlyMemory?.lineId === "understanding" && memoryFork && !isInterchangeStation(earlyMemory))
    moveStation(earlyMemory, memoryFork.y);
  // The late memory corridor climbs into the detection transfer, then keeps
  // its height through the graph and event-refinement papers.
  const memoryRoot = stations.get("reactvau")?.platforms.find(p => p.lineId === "understanding")
    ?? stations.get("memovad")?.platforms.find(p => p.lineId === "understanding");
  if (memoryRoot) for (const id of ["s2mgraph-vad", "peer-vad"]) {
    const station = stations.get(id);
    if (station?.lineId === "understanding" && !isInterchangeStation(station)) moveStation(station, memoryRoot.y);
  }
  // Representation resumes on one stable shelf after VA-GPT. Its two
  // short comparison arms use the open upper band instead of shifting the
  // trunk whenever another ordinary stop is added.
  const representation = routeMembers.get("alignment")?.find(route =>
    route.some(p => p.id === "hiprobe-vad") && route.some(p => p.id === "spherevad"));
  if (representation) {
    const baseline = origin + levels.get("alignment")! - 96;
    for (const paper of representation.slice(representation.findIndex(p => p.id === "hiprobe-vad"))) {
      const station = stations.get(paper.id)!;
      if (!isInterchangeStation(station)) moveStation(station, baseline);
    }
    // The streaming departure needs a local lower platform so its label
    // does not sit under LAVIDA's neighboring synthesis fork.
    const streaming = stations.get("td-vad");
    if (streaming && !isInterchangeStation(streaming)) moveStation(streaming, baseline + 120);
    for (const id of ["piercingeye", "copra"]) {
      const station = stations.get(id);
      if (station && !isInterchangeStation(station)) moveStation(station, baseline - 120);
    }
  }
  // Put the generation arm above the continuous blue trunk. It leaves at
  // LAVIDA, while the later streaming arm leaves at TD-VAD toward memory.
  const synthesisFork = stations.get("lavida"), blueTrunk = stations.get("steervad");
  if (synthesisFork && blueTrunk && synthesisFork.x > blueTrunk.x) {
    moveStation(synthesisFork, blueTrunk.y);
    for (const id of ["anomalycraft", "pa-vad", "cavge"]) {
      const station = stations.get(id);
      if (station) moveStation(station, synthesisFork.y - 192);
    }
  }
  // Keep the late observation and evaluation runs separated by a full
  // label corridor. Their old opposing zigzags pinched names between rails.
  for (const ids of [["agenticvau", "vibes", "vto", "seek-vau"],
    ["pistachio", "ecva-anomshield", "tau-bench", "tar-bench"]]) {
    const members = ids.map(id => stations.get(id)).filter((s): s is Station => !!s);
    if (members.length < 2 || members.some(isInterchangeStation)) continue;
    const y = ids.includes("tau-bench") && reasoningShelf !== undefined ? reasoningShelf - 192
      : Math.round(members.reduce((sum, s) => sum + s.y, 0) / members.length / 24) * 24;
    members.forEach(s => moveStation(s, y));
  }
  // Entry arms settle onto their continuing trunk at the named merge. Moving
  // the complete head group removes a tiny dip immediately after the merge.
  for (const routes of routeMembers.values()) for (const through of routes) {
    if (through.length < 2) continue;
    const terminal = stations.get(through[0].id)!, next = stations.get(through[1].id)!;
    if (terminal.platforms.length !== 1 || next.platforms.length !== 1) continue;
    const heads = routes.filter(route => route.length === 2 && route[1].id === terminal.paperId &&
      stations.get(route[0].id)!.platforms.length === 1);
    if (heads.length !== 2) continue;
    const shift = next.y - terminal.y;
    for (const head of heads) {
      const station = stations.get(head[0].id)!;
      moveStation(station, station.y + shift);
    }
    moveStation(terminal, next.y);
  }
  alignContinuations();
  // Keep enough room above the topmost rail for a full two-line name.
  for (const station of stations.values()) {
    if (!isInterchangeStation(station) && station.y < top + station.label.height + 36)
      moveStation(station, top + station.label.height + 36);
  }
  // These named labels have an intentional reading side. Keep collision
  // handling within that side when space exists instead of flipping a label
  // merely to save a few pixels of horizontal alignment.
  const preferredLabelSides = new Map<string, number>([
    ["cuebench", -1], ["vad-dpo", 1],
    ["anomalycraft", 1], ["pa-vad", 1], ["cavge", -1],
    ["eval", 1], ["lavad", -1], ["log-sad", 1], ["vera", -1],
    ["lagovad", -1], ["lrpo", -1], ["probe-vad", 1], ["prime-vad", -1],
    ["ca-judge", 1], ["road", 1], ["promptvad", -1],
  ]);
  // Alternate names, not the rail itself. Two neighbors can share horizontal
  // space while each name stays tucked against its own station. Reserve the
  // full width only between every other name on the same reading side.
  const staggeredPairs = new Set<string>();
  const staggeredLabelSides = new Map<string, number>();
  for (const routes of routeMembers.values()) for (const members of routes) {
    let previousSide = 0;
    for (let i = 0; i < members.length; i++) {
      const station = stations.get(members[i].id)!;
      const before = i ? stations.get(members[i - 1].id)! : undefined;
      const levelPair = before && !isInterchangeStation(before) && !isInterchangeStation(station) &&
        before.y === station.y && station.y > top + 108 && station.y < plotBottom - 108;
      const side = preferredLabelSides.get(station.paperId) ??
        (levelPair && previousSide ? -previousSide : labelSides.get(station.paperId) ?? -1);
      if (levelPair && side !== previousSide) {
        staggeredPairs.add(`${before.paperId}:${station.paperId}`);
        staggeredLabelSides.set(before.paperId, previousSide);
        staggeredLabelSides.set(station.paperId, side);
      }
      previousSide = side;
    }
  }
  // Reserve route clearances from actual platform heights, label widths and
  // named forks. These constraints apply to individual stops, not date columns.
  const incoming = new Map<string, { from: Station; distance: number }[]>();
  for (const [lineId, routes] of routeMembers) for (const members of routes) {
    for (let i = 1; i < members.length; i++) {
      const a = stations.get(members[i - 1].id)!, b = stations.get(members[i].id)!;
      const pa = a.platforms.find(p => p.lineId === lineId)!, pb = b.platforms.find(p => p.lineId === lineId)!;
      const departure = a.fork?.branchId !== lineId && !directDepartures.has(`${lineId}:${a.paperId}:${b.paperId}`) ? platformRoom.get(a.paperId)! : 0;
      const arrival = directArrivals.has(`${lineId}:${a.paperId}:${b.paperId}`) ? 0 : platformRoom.get(b.paperId)!;
      const entries = incoming.get(b.paperId) ?? [];
      // Alternating labels can share horizontal space. Wider same-period
      // names still need more room than the compact base column interval.
      const samePeriod = timelineYear(members[i - 1]) === timelineYear(members[i]) && month(members[i - 1]) === month(members[i]);
      const nameWidths = members.slice(i - 1, i + 1).map(paper => stationNameWidth(paper) + 14);
      const paired = parallelPairs.has(`${a.paperId}:${b.paperId}`);
      // At the plot edge, labels cannot alternate above and below the rail.
      // Reserve a full adjacent pair, including compact context icons.
      const labelRoom = Math.max(a.label.height, b.label.height) + 20;
      const edgeShelf = pa.y === pb.y && (pa.y + labelRoom > plotBottom - 12 || pa.y - labelRoom < top + 10);
      const closeShelf = Math.abs(pb.y - pa.y) <= 64 && b.x - a.x <= 96;
      const topEntry = pa.y - labelRoom < top + 24 && pb.y > pa.y;
      const staggered = staggeredPairs.has(`${a.paperId}:${b.paperId}`);
      const afterMerge = a.merge?.parentId === lineId;
      const nearMerge = afterMerge || members.slice(Math.max(0, i - 3), i).some(p =>
        stations.get(p.id)!.merge?.parentId === lineId);
      const labelGap = edgeShelf || topEntry ? Math.ceil(((a.label.width + b.label.width) / 2 + 16) / 8) * 8
        : nearMerge && staggered ? Math.ceil((Math.max(...nameWidths) * .56 + 28) / 8) * 8
        : staggered ? Math.ceil((Math.max(...nameWidths) * .42 + 20) / 8) * 8
        : paired ? Math.ceil(((nameWidths[0] + nameWidths[1]) / 2 + 24) / 8) * 8
        : closeShelf ? Math.ceil(((nameWidths[0] + nameWidths[1]) * .34 + 16) / 8) * 8
        : samePeriod && Math.abs(pb.y - pa.y) < 64 ? Math.ceil(((nameWidths[0] + nameWidths[1]) * .28) / 8) * 8 : 0;
      const colorChangeRoom = a.continuation && pa.y === pb.y &&
        a.lineIds.some(id => !b.lineIds.includes(id)) ? Math.max(...nameWidths) / 2 + 40 : 0;
      entries.push({ from: a, distance: Math.max(Math.abs(pb.y - pa.y) + departure + arrival, labelGap,
        colorChangeRoom, fanClearance.get(`${a.paperId}:${b.paperId}`) ?? 0) });
      if (i > 1 && staggered && staggeredPairs.has(`${members[i - 2].id}:${a.paperId}`)) {
        const earlier = stations.get(members[i - 2].id)!;
        entries.push({ from: earlier, distance: (stationNameWidth(members[i - 2]) + stationNameWidth(members[i]) + 28) / 2 + 20 });
      }
      incoming.set(b.paperId, entries);
    }
  }
  // Years are hard boundaries; months only suggest a position within them.
  // Each rail can spread its own stops instead of sharing rigid date columns.
  const byId = new Map(papers.map(paper => [paper.id, paper]));
  const ordered = [...stations.values()].sort((a, b) => a.x - b.x || a.paperId.localeCompare(b.paperId));
  const seedX = new Map(ordered.map(station => [station.paperId, station.x]));
  const seedYears = new Map(years.map(year => [year.year, { ...year }]));
  const addConstraint = (a: Station, b: Station, distance: number) => {
    const entries = incoming.get(b.paperId) ?? [];
    const existing = entries.find(edge => edge.from.paperId === a.paperId);
    if (existing) existing.distance = Math.max(existing.distance, distance);
    else entries.push({ from: a, distance });
    incoming.set(b.paperId, entries);
  };
  for (let i = 0; i < ordered.length; i++) for (let j = i + 1; j < ordered.length; j++) {
    const a = ordered[i], b = ordered[j];
    const pa = byId.get(a.paperId)!, pb = byId.get(b.paperId)!;
    if (timelineYear(pa) !== timelineYear(pb)) continue;
    // Keep clearly early work before clearly late work, while nearby months
    // on independent routes can trade places to make room for names and bends.
    if (pa.timeline && pb.timeline && month(pb) - month(pa) >= 6) addConstraint(a, b, 24);
    const ba = stationBounds(a, 28), bb = stationBounds(b, 28);
    if (ba.y < bb.y + bb.height && bb.y < ba.y + ba.height) addConstraint(a, b, 72);
  }
  const outgoing = new Map(ordered.map(station => [station.paperId, [] as { to: Station; distance: number }[]]));
  for (const [id, entries] of incoming) for (const edge of entries)
    outgoing.get(edge.from.paperId)!.push({ to: stations.get(id)!, distance: edge.distance });
  const edgePadding = (station: Station) => station.label.width / 2 + 24;
  // The minimum width comes from actual route/label geometry, not month slots.
  const earliest = new Map<string, number>();
  let minimumEnd = margin;
  for (const year of years) {
    const members = ordered.filter(station => timelineYear(byId.get(station.paperId)!) === year.year);
    year.x = minimumEnd;
    for (const station of members) earliest.set(station.paperId, Math.max(year.x + edgePadding(station),
      ...(incoming.get(station.paperId) ?? []).map(edge => earliest.get(edge.from.paperId)! + edge.distance)));
    year.width = Math.max(140, ...members.map(station => earliest.get(station.paperId)! + edgePadding(station) - year.x));
    minimumEnd += year.width;
  }
  // Unconstrained small diagrams also need room for label placement around
  // junctions; the minimum route geometry alone is too tight for two-line names.
  const targetEnd = options.width === undefined ? margin + (minimumEnd - margin) * 1.2 : options.width - margin;
  if (!Number.isFinite(targetEnd) || targetEnd < minimumEnd - .01)
    throw new Error(`Map needs ${Math.ceil(minimumEnd + margin)} units within a fixed ${options.width}-unit width; split a crowded reading route into a vertical branch.`);
  const minimumSpan = minimumEnd - margin;
  let yearStart = margin;
  for (const year of years) {
    year.x = yearStart;
    year.width += minimumSpan ? (targetEnd - minimumEnd) * year.width / minimumSpan : 0;
    yearStart += year.width;
  }
  const yearById = new Map(years.map(year => [year.year, year]));
  const yearFor = (station: Station) => yearById.get(timelineYear(byId.get(station.paperId)!))!;
  const latest = new Map<string, number>();
  for (const station of ordered) {
    const year = yearFor(station);
    earliest.set(station.paperId, Math.max(year.x + edgePadding(station),
      ...(incoming.get(station.paperId) ?? []).map(edge => earliest.get(edge.from.paperId)! + edge.distance)));
  }
  for (const station of [...ordered].reverse()) {
    const year = yearFor(station);
    latest.set(station.paperId, Math.min(year.x + year.width - edgePadding(station),
      ...outgoing.get(station.paperId)!.map(edge => latest.get(edge.to.paperId)! - edge.distance)));
  }
  const preferredX = new Map(ordered.map(station => {
    const year = yearFor(station), seed = seedYears.get(year.year)!;
    const fraction = (seedX.get(station.paperId)! - seed.x) / seed.width;
    return [station.paperId, year.x + fraction * year.width];
  }));
  const positions = new Map(earliest);
  // Relax each stop toward even spacing along its actual reading neighbors.
  // A light date preference preserves the rough timeline without enforcing
  // shared month columns. Forward/backward sweeps preserve all hard clearances.
  for (let pass = 0; pass < 160; pass++) {
    for (const station of pass % 2 ? [...ordered].reverse() : ordered) {
      const id = station.paperId;
      const before = incoming.get(id) ?? [], after = outgoing.get(id)!;
      const lower = Math.max(earliest.get(id)!, ...before.map(edge => positions.get(edge.from.paperId)! + edge.distance));
      const upper = Math.min(latest.get(id)!, ...after.map(edge => positions.get(edge.to.paperId)! - edge.distance));
      const adjacent = [...new Set(neighbors.get(id)!)].map(other => positions.get(other)!);
      // Keep endpoints near their date hint so relaxation cannot collapse a
      // whole terminal run into its predecessor. Interior stops spread freely.
      const dateWeight = adjacent.length === 1 ? 8 : adjacent.length > 2 ? 2 : 0.35;
      const target = (dateWeight * preferredX.get(id)! + adjacent.reduce((sum, value) => sum + value, 0)) /
        (dateWeight + adjacent.length);
      positions.set(id, Math.max(lower, Math.min(upper, target)));
    }
  }
  for (const station of stations.values()) {
    station.x = positions.get(station.paperId)!;
    station.platforms.forEach(platform => { platform.x = station.x; });
  }
  x = targetEnd;
  plotBounds.width = x - margin;
  // Both tracks meet at the sourced paper's one station. There are no
  // synthetic junctions between papers and no layout-driven anchor selection.
  type Stop = Point & { paperId: string };
  const routesByLine = new Map(schools.map(school => [school.id,
    routeMembers.get(school.id)!.map(members => members.map((paper): Stop => ({
      paperId: paper.id, ...stations.get(paper.id)!.platforms.find(p => p.lineId === school.id)!,
    }))),
  ]));
  const junctions: PublicationLayout["junctions"] = [...stations.values()].filter(s => s.fork)
    .map(s => ({ x: s.x, y: s.y, paperId: s.paperId, ...s.fork! }));
  // Keep labels next to their paper. Station bodies and planned tracks define
  // local candidates; the packer resolves conflicts between neighboring names.
  const nodes = [...stations.values()].map((station) => stationBounds(station, 13));
  // Reserve the planned bend corridors before placing labels. A name must not
  // occupy the only forward path between two closely spaced stops.
  const needsPlatform = (stop: Stop) => stations.get(stop.paperId)!.platforms.length > 1 &&
    !stations.get(stop.paperId)!.continuation;
  const bendLate = (id: string, a: Stop, b: Stop) => stations.get(a.paperId)!.fork?.parentId === id ||
    [...directDepartures].some(key => key.startsWith(`${id}:${a.paperId}:`)) ||
    directArrivals.has(`${id}:${a.paperId}:${b.paperId}`) ||
    convergingPairs.has(`${a.paperId}:${b.paperId}`) ||
    b.x - a.x > 240 && needsPlatform(b) && !stations.get(b.paperId)!.fork;
  const bendEarly = (id: string, a: Stop, b: Stop) =>
    stations.get(a.paperId)!.fork?.branchId === id || directDepartures.has(`${id}:${a.paperId}:${b.paperId}`);
  const stationRoutes = new Map(papers.map(paper => [paper.id, new Set<string>()]));
  for (const [id, routes] of routesByLine) routes.forEach((members, index) =>
    members.forEach(stop => stationRoutes.get(stop.paperId)!.add(`${id}:${index}`)));
  const corridors = [...routesByLine.entries()].flatMap(([id, routes]) =>
    routes.flatMap((members, routeIndex) => members.slice(1).flatMap((b, index) => {
      const a = members[index];
      const leavesFork = stations.get(a.paperId)!.fork?.branchId === id;
      const departure = { x: a.x + (leavesFork || directDepartures.has(`${id}:${a.paperId}:${b.paperId}`) ? 0 : platformRoom.get(a.paperId)!), y: a.y };
      const arrival = { x: b.x - (directArrivals.has(`${id}:${a.paperId}:${b.paperId}`) ? 0 : platformRoom.get(b.paperId)!), y: b.y };
      const early = bendEarly(id, a, b), late = !early && bendLate(id, a, b);
      const planned = [a, ...elbows(departure, arrival)[late ? 1 : 0], b];
      const otherNodes = [...stations.values()].filter(station =>
        station.paperId !== a.paperId && station.paperId !== b.paperId)
        .map(station => stationBounds(station, 20));
      if (segments(planned).some(({ a, b }) => otherNodes.some(box => crosses(a, b, box)))) {
        return segments([a, ...routeBetween(departure, arrival, otherNodes, [], plotBounds, late, [a, b], early), b])
          .map(segment => ({ ...segment, routeKey: `${id}:${routeIndex}` }));
      }
      return segments(planned).map(segment => ({ ...segment, routeKey: `${id}:${routeIndex}` }));
    })));
  const labelOptions = new Map<string, LabelCandidate[]>();
  for (const paper of papers) {
    const station = stations.get(paper.id)!;
    const interchange = isInterchangeStation(station);
    const railClearance = interchange ? 14 : 8;
    const year = years.find((item) => item.year === timelineYear(paper))!;
    const outward = labelSides.get(paper.id) ?? (home.get(paper.id)! < origin + anchorLevel(paper) ? -1 : 1);
    const candidates: LabelCandidate[] = [];
    const clearLabel = (box: Box) => box.x >= year.x + 4 && box.x + box.width <= year.x + year.width - 4 &&
      box.y >= top + 10 && box.y + box.height <= plotBottom - 12 &&
      !corridors.some(({ a, b }) => crosses(a, b, { x: box.x - railClearance, y: box.y - railClearance,
        width: box.width + railClearance * 2, height: box.height + railClearance * 2 })) &&
      !nodes.some(node => overlaps(box, node, 2));
    for (const gap of station.platforms.length > 1 && !station.continuation ? [20, 28, 36, 44, 52, 60, 68]
      : [16, 24, 32]) {
      for (const side of [-1, 1]) {
        for (const align of [0, -.25, .25, -.5, .5, -.75, .75, -1, 1]) {
          const box = { ...station.label,
            x: Math.max(year.x + 4, Math.min(year.x + year.width - station.label.width - 4,
              station.x - station.label.width / 2 + align * (station.label.width / 2 - 8))),
            y: side < 0 ? station.platforms[0].y - station.label.height - gap : station.platforms.at(-1)!.y + gap };
          if (station.x < box.x + 8 || station.x > box.x + box.width - 8 || !clearLabel(box)) continue;
          const drift = Math.abs(box.x + box.width / 2 - station.x);
          if (inlineStations.has(paper.id) && [...stations.values()].some(other =>
            other.paperId !== paper.id && Math.abs(other.y - station.y) < 16 &&
            other.lineIds.some(id => station.lineIds.includes(id)) &&
            Math.abs(box.x + box.width / 2 - other.x) < drift + 8)) continue;
          candidates.push({ box, score: gap * 4 + drift + (side !== outward ? 20 : 0) });
        }
      }
      for (const side of [-1, 1]) for (const align of [0, -.5, .5]) {
        if (gap < 24) continue; // Keep horizontal terminal caps outside the name.
        const box = { ...station.label,
          x: side < 0 ? station.x - gap - station.label.width : station.x + gap,
          y: station.y - station.label.height / 2 + align * station.label.height / 2 };
        if (clearLabel(box)) candidates.push({ box, score: gap * 4 + Math.abs(align) * 24 + 60 });
      }
    }
    // A transfer name belongs on the open side of the fan. Favor centered
    // names and clear surroundings over squeezing into the nearest wedge.
    if (interchange) for (const candidate of candidates) {
      const box = candidate.box;
      const halo = { x: box.x - 28, y: box.y - 28, width: box.width + 56, height: box.height + 56 };
      candidate.score += Math.abs(box.x + box.width / 2 - station.x) +
        corridors.filter(({ a, b }) => crosses(a, b, halo)).length * 80;
    }
    const unrelatedCorridors = corridors.filter(segment => !stationRoutes.get(paper.id)!.has(segment.routeKey));
    const associated: LabelCandidate[] = [];
    for (const candidate of candidates) {
      const ownDistance = Math.min(...station.platforms.map(p => railLabelDistance(candidate.box, p, p)));
      const otherDistance = Math.min(...unrelatedCorridors
        .map(({ a, b }) => railLabelDistance(candidate.box, a, b)));
      candidate.score += Math.max(0, ownDistance + 24 - otherDistance) * 16;
      if (otherDistance >= ownDistance + 12) associated.push(candidate);
      const side = staggeredLabelSides.get(paper.id);
      if (side && (side < 0 ? candidate.box.y >= station.y : candidate.box.y < station.y)) candidate.score += 120;
    }
    const nearbyCandidates = associated.length ? associated : candidates;
    const preferredSide = preferredLabelSides.get(paper.id);
    const preferred = preferredSide === undefined ? [] : nearbyCandidates.filter(({ box }) => preferredSide < 0
      ? box.y + box.height < station.platforms[0].y : box.y > station.platforms.at(-1)!.y);
    const options = preferred.length ? preferred : nearbyCandidates;
    options.sort((a, b) => a.score - b.score);
    labelOptions.set(paper.id, [...new Map(options.map(option => [JSON.stringify(option.box), option])).values()]);
  }
  const packedLabels = packLabels(labelOptions);
  for (const [id, box] of packedLabels) stations.get(id)!.label = box;
  const labels = [...packedLabels.values()].map((box) => ({ x: box.x - 4, y: box.y - 4,
    width: box.width + 8, height: box.height + 8 }));
  const occupied: Segment[] = [];
  const lines = order.map((id): PublicationLine => {
    const school = schools.find((item) => item.id === id)!;
    const routes = routesByLine.get(id)!;
    const startsAtNode = (stop: Stop) => stations.get(stop.paperId)!.platforms.length > 1 || school.branchAt?.paperId === stop.paperId ||
      routes.some(other => other.slice(1).some(point => point.paperId === stop.paperId));
    const tracks = routes.map(stops => {
      const track: Point[] = [];
      stops.forEach((stop, i) => {
        if (!i) {
          if (!startsAtNode(stop)) track.push({ x: stop.x - 18, y: stop.y });
          track.push({ x: stop.x, y: stop.y });
          return;
        }
        const before = stops[i - 1];
        const unrelated = [...stations.values()].flatMap((station) =>
          station.paperId !== stop.paperId && station.paperId !== before.paperId
            ? [stationBounds(station)]
            : station.platforms.filter((platform) => platform.lineId !== id && isInterchangeStation(station))
                .map((platform) => ({ x: platform.x - 11, y: platform.y - 11, width: 22, height: 22 })));
        const obstacles = [...labels, ...unrelated];
        const path = route(before, stop, obstacles, occupied, plotBounds, {
          start: stations.get(before.paperId)!.fork?.branchId === id || directDepartures.has(`${id}:${before.paperId}:${stop.paperId}`) ? 0 : platformRoom.get(before.paperId)!,
          end: directArrivals.has(`${id}:${before.paperId}:${stop.paperId}`) ? 0 : platformRoom.get(stop.paperId)!,
        }, !bendEarly(id, before, stop) && bendLate(id, before, stop), bendEarly(id, before, stop));
        track.push(...path.slice(1));
      });
      const last = stops.at(-1);
      if (last && stations.get(last.paperId)!.platforms.length === 1 &&
          !routes.some(other => other !== stops && other.some(stop => stop.paperId === last.paperId))) {
        track.push({ x: last.x + 18, y: last.y });
      }
      const simplified = simplify(track);
      occupied.push(...segments(simplified));
      return simplified;
    });
    return { ...school, tracks, paperRoutes: routes.map(stops => stops.map(stop => stop.paperId)),
      labelPosition: { x: 24 + order.indexOf(id) * 300, y: plotBottom + 32 } };
  });
  return { width: options.width ?? Math.max(1700, x + 24), height: plotBottom + 96,
    years, methodOrder: order, lines, stations, junctions,
    plotBounds };
}
