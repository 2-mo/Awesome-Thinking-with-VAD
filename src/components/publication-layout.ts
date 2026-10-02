import type { Cluster, Paper } from "../types";
import { clusterName, isMapPaper, paperMethods, publicationVenue } from "../publication.ts";

export type Point = { x: number; y: number };
export type Box = Point & { width: number; height: number };
export type Platform = Point & { lineId: string };
export type Station = Point & {
  paperId: string; lineId: string; lineIds: string[]; platforms: Platform[]; label: Box;
  fork?: { parentId: string; branchId: string };
  // A two-direction paper with only one incoming and one outgoing rail uses
  // one ordinary marker. Its method memberships remain available in details.
  continuation?: boolean;
};
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
    quarters: { quarter: number | null; x: number; width: number; count: number }[];
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
export const stationVenue = (paper: Paper, fork = false): string =>
  `${publicationVenue(paper.venue)}${fork ? " · Fork" : ""}`;
export const stationVenueSize = (fork = false): number => fork ? 12 : 14;
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
      ({ a, b }) => !obstacles.some((box) => crosses(a, b, box)) && !unnamedJoin(a, b, occupied, connections),
    );
  const cost = (path: Point[]) =>
    segments(path).reduce(
      (sum, { a, b }) => sum + length(a, b) + parallelPenalty(a, b, occupied) + crossingPenalty(a, b, occupied) +
        (bendLate ? Math.abs(b.y - a.y) * Math.max(0, end.x - (a.x + b.x) / 2) / Math.max(1, end.x - start.x) : 0),
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
  platforms: { start: number; end: number }, bendLate = false): Point[] {
  // Only shared stations need horizontal approaches to their separate
  // platforms. Ordinary stations may sit directly on a diagonal or its end.
  if (!platforms.start && !platforms.end) return routeBetween(start, end, obstacles, occupied, bounds, bendLate);
  for (const scale of [1, .75]) {
    if (end.x - start.x <= scale * (platforms.start + platforms.end)) continue;
    const departure = { x: start.x + platforms.start * scale, y: start.y };
    const arrival = { x: end.x - platforms.end * scale, y: end.y };
    if (obstacles.some((box) => crosses(start, departure, box) || crosses(arrival, end, box))) continue;
    try {
      return simplify([start, ...routeBetween(departure, arrival, obstacles, occupied, bounds, bendLate, [start, end]), end]);
    } catch {
      // A tight label corridor can require a shorter platform or a direct route.
    }
  }
  return routeBetween(start, end, obstacles, occupied, bounds, bendLate);
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
  if (!search(options)) throw new Error(`Unable to place adjacent topology labels near ${blocked}`);
  return result;
}

export function createPublicationLayout(papers: Paper[], clusters: Cluster[] = []): PublicationLayout {
  const month = (paper: Paper) => paper.timeline?.month ?? 13;
  papers = papers.filter(isMapPaper).sort((a, b) =>
    a.year - b.year || month(a) - month(b) || a.id.localeCompare(b.id));
  const ids = [...new Set(papers.flatMap(paperMethods))];
  const schools = [
    ...clusters.filter((school) => ids.includes(school.id)).map(c => ({ ...c, label: clusterName(c) })),
    ...ids.filter((id) => !clusters.some((school) => school.id === id)).sort()
      .map((id) => ({ id, label: id, color: "#617282" })),
  ] as Pick<PublicationLine, "id" | "label" | "color" | "branchOf" | "branchAt" | "routes">[];
  const forks = new Map(schools.filter(s => s.branchOf && s.branchAt &&
    papers.some(p => p.id === s.branchAt!.paperId)).map(s => [s.branchAt!.paperId, s]));
  const nearby = clusters.filter(c => ids.includes(c.id) && c.layoutNear && ids.includes(c.layoutNear))
    .map(c => [c.id, c.layoutNear!, c.layoutSide ?? ""]);
  const order = methodOrder(papers, schools.map((school) => school.id),
    schools.filter(s => s.branchOf && ids.includes(s.branchOf)).map(s => [s.id, s.branchOf!]), nearby);
  const rank = (paper: Paper) => forks.has(paper.id) ? order.indexOf(forks.get(paper.id)!.branchOf!) : paperMethods(paper)
    .reduce((sum, id) => sum + order.indexOf(id), 0) / paperMethods(paper).length;
  const margin = 24, top = 88, laneGap = 168, slotWidth = 68, origin = top + 108;
  // Short branches share a tighter band with their neighbors. Dense trunks
  // retain the room needed by two-sided labels and interchange platforms.
  const compactBranch = (id: string) => schools.some(s => s.id === id && s.branchOf) &&
    papers.filter(paper => paperMethods(paper).includes(id)).length <= 10;
  // A branch explicitly grouped beside its parent needs room for transfer
  // approaches on the neighboring trunk. Other short branches stay compact.
  const groupedBranch = (id: string) => compactBranch(id) && nearby.some(pair => pair.includes(id));
  const levels = new Map<string, number>();
  order.forEach((id, index) => levels.set(id, index === 0 ? 0 : levels.get(order[index - 1])! +
    (compactBranch(id) || compactBranch(order[index - 1]) ?
      (groupedBranch(id) || groupedBranch(order[index - 1]) ? 168 : 144) :
      nearby.some(pair => pair.includes(id) && pair.includes(order[index - 1])) ? 168 : laneGap)));
  const level = (paper: Paper) => {
    const methods = forks.has(paper.id) ? [forks.get(paper.id)!.branchOf!] : paperMethods(paper);
    return methods.reduce((sum, id) => sum + levels.get(id)!, 0) / methods.length;
  };
  const stationXs = new Map<string, number>();
  const yearValues = [...new Set(papers.map((paper) => paper.year))];
  let x = margin;
  const years = yearValues.map((year) => {
    const members = papers.filter((paper) => paper.year === year);
    const months = [...new Set(members.map(month))].sort((a, b) => a - b);
    let cursor = x + 40;
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
        const slot = Math.max(0, ...occupiedMethods.map((id) => nextSlot.get(id) ?? 0));
        stationXs.set(paper.id, cursor + slot * slotWidth);
        occupiedMethods.forEach((id) => nextSlot.set(id, slot + 1));
        // Reserve time-axis space after the actual fork paper, so an immediate
        // same-month branch can leave it diagonally rather than vertically.
        const fork = forks.get(paper.id);
        if (fork) nextSlot.set(fork.id, slot + Math.ceil((laneGap + 32) / slotWidth));
        lastSlot = Math.max(lastSlot, slot);
      }
      cursor += lastSlot * slotWidth + 52;
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
      stationXs.get(paper.id)! - x + labelWidth(stationName(paper), 20) / 2 + 40));
    const width = Math.max(140, cursor - x, widestTail, ...members.map((paper) =>
      Math.max(labelWidth(stationName(paper), 20), labelWidth(publicationVenue(paper.venue), 14)) + 22));
    if (quarters.length) quarters.at(-1)!.width = x + width - quarters.at(-1)!.x;
    const item = { year, x, width, count: members.length, quarters };
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
  // Inserting a short color section into a through route should not flip
  // the alternating shelves of every later station on that route.
  const inlineTails = new Set<string>();
  for (const school of schools) for (const members of routeMembers.get(school.id)!) {
    if (members.length !== 2 || members.some(p => !continuations.has(p.id))) continue;
    for (const id of paperMethods(members[0])) {
      if (id !== school.id && paperMethods(members[1]).includes(id)) inlineTails.add(`${id}:${members[1].id}`);
    }
  }
  // Pair neighboring stops on alternating shelves. The resulting long bends
  // make room for station names without stretching every dense month sideways.
  const shelves = new Map(schools.map((school) => {
    let index = 0;
    const values = new Map(lineMembers.get(school.id)!.map(paper => {
      const shelf = (Math.floor(index / 2) % 2 ? 1 : -1) * 24;
      if (!inlineTails.has(`${school.id}:${paper.id}`)) index++;
      return [paper.id, shelf];
    }));
    return [school.id, values];
  }));
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
    const methods = forks.has(paper.id) ? [forks.get(paper.id)!.branchOf!] : paperMethods(paper);
    const shelf = methods.reduce((sum, id) => sum + shelves.get(id)!.get(paper.id)!, 0) / methods.length;
    return [paper.id, origin + anchorLevel(paper) + shelf];
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
      label: { x: 0, y: 0, width: Math.max(labelWidth(stationName(paper), 20),
        labelWidth(stationVenue(paper, !!fork), stationVenueSize(!!fork))) + 14, height: 48 } });
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
  }
  // A branch that returns to its parent is a local parallel corridor. Keep
  // the trunk level through the return and use straight shelves for the
  // intermediate papers instead of alternating station-by-station heights.
  const parallelPairs = new Set<string>();
  const directDepartures = new Set<string>(), directArrivals = new Set<string>();
  const returningForks = new Set<string>();
  const fanClearance = new Map<string, number>();
  for (const school of schools) for (const branch of routeMembers.get(school.id)!) {
    if (!school.branchOf || branch.length < 3 || branch[0].id !== school.branchAt?.paperId) continue;
    const root = stations.get(branch[0].id)!, terminal = stations.get(branch.at(-1)!.id)!;
    if (!terminal.lineIds.includes(school.branchOf) || branch.slice(1, -1).some(p => paperMethods(p).length > 1)) continue;
    const parentRoutes = routeMembers.get(school.branchOf)!;
    const trunk = parentRoutes.find(route => route.findIndex(p => p.id === root.paperId) >= 0 &&
      route.findIndex(p => p.id === terminal.paperId) > route.findIndex(p => p.id === root.paperId));
    if (!trunk) continue;
    const start = trunk.findIndex(p => p.id === root.paperId), end = trunk.findIndex(p => p.id === terminal.paperId);
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
    let last = end;
    while (last + 1 < trunk.length && paperMethods(trunk[last + 1]).length === 1) last++;
    for (const paper of trunk.slice(start + 1, last + 1)) {
      const station = stations.get(paper.id)!;
      moveStation(station, station.y + height - station.platforms.find(p => p.lineId === school.branchOf)!.y);
    }
    const shelf = (members: Paper[], y: number, outward: number) => {
      for (const paper of members) { moveStation(stations.get(paper.id)!, y); labelSides.set(paper.id, outward); }
      for (let i = 1; i < members.length; i++) parallelPairs.add(`${members[i - 1].id}:${members[i].id}`);
    };
    shelf(branch.slice(1, -1), height + side * 120, side);
    for (const route of sideRoutes) {
      shelf(route.slice(1, -1), height - side * 120, side);
      directDepartures.add(`${school.branchOf}:${root.paperId}:${route[1].id}`);
      directArrivals.add(`${school.branchOf}:${route.at(-2)!.id}:${terminal.paperId}`);
    }
    returningForks.add(`${school.id}:${root.paperId}`);
  }
  // A short local offshoot should not make its trunk peak at the fork. Keep
  // that root on the incoming shelf, then align an interchange terminal with
  // its continuing route instead of adding another tiny departure bend.
  for (const [lineId, routes] of routeMembers) for (const members of routes) {
    if (members.length !== 2) continue;
    const root = stations.get(members[0].id)!, terminal = stations.get(members[1].id)!;
    const trunk = routes.find(other => other !== members && other.some(paper => paper.id === root.paperId));
    if (!trunk) continue;
    const index = trunk.findIndex(paper => paper.id === root.paperId);
    if (index > 0 && index < trunk.length - 1 && root.platforms.length === 1) {
      const before = stations.get(trunk[index - 1].id)!, after = stations.get(trunk[index + 1].id)!;
      const previousY = before.platforms.find(platform => platform.lineId === lineId)!.y;
      const nextY = after.platforms.find(platform => platform.lineId === lineId)!.y;
      if ((root.y - previousY) * (root.y - nextY) > 0) moveStation(root, previousY);
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
    if (station.platforms.length < 2 || station.continuation) return [station.paperId, 0];
    const indices = station.lineIds.map(id => order.indexOf(id));
    const between = order.slice(Math.min(...indices) + 1, Math.max(...indices));
    const activeBetween = !station.fork && between.some(id =>
      routeMembers.get(id)?.some(members => members.some(p => stationXs.get(p.id)! < station.x) &&
        members.some(p => stationXs.get(p.id)! > station.x)));
    return [station.paperId, activeBetween ? 80 : 16];
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
  // Reserve horizontal space from the actual platform heights, rather than
  // counting the number of method bands a transfer crosses. Stretch only the
  // constrained columns; every independent paper in that column moves with it.
  // The same monotone transform updates the year and quarter boundaries.
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
      const samePeriod = members[i - 1].year === members[i].year && month(members[i - 1]) === month(members[i]);
      const nameWidths = members.slice(i - 1, i + 1).map(paper => labelWidth(stationName(paper), 20) + 14);
      const paired = parallelPairs.has(`${a.paperId}:${b.paperId}`);
      const labelGap = samePeriod || paired ? Math.ceil(((nameWidths[0] + nameWidths[1]) * (paired ? .5 : .28) + (paired ? 24 : 0)) / 8) * 8 : 0;
      entries.push({ from: a, distance: Math.max(Math.abs(pb.y - pa.y) + departure + arrival, labelGap, fanClearance.get(`${a.paperId}:${b.paperId}`) ?? 0) });
      incoming.set(b.paperId, entries);
    }
  }
  const columns = [...new Set([...stations.values()].map(station => station.x))].sort((a, b) => a - b);
  const originalXs = new Map([...stations.values()].map(station => [station.paperId, station.x]));
  const movedXs = new Map<string, number>();
  const shifts: { x: number; amount: number }[] = [{ x: margin, amount: 0 }];
  let shift = 0;
  for (const column of columns) {
    const group = [...stations.values()].filter(station => station.x === column);
    const position = Math.max(column + shift, ...group.flatMap(station =>
      (incoming.get(station.paperId) ?? []).map(edge =>
        (movedXs.get(edge.from.paperId) ?? originalXs.get(edge.from.paperId)!) + edge.distance)));
    shift = position - column;
    shifts.push({ x: column, amount: shift });
    group.forEach(station => movedXs.set(station.paperId, position));
  }
  const warpX = (value: number): number => {
    const next = shifts.findIndex(point => point.x >= value);
    if (next < 0) return value + shift;
    if (!next) return value;
    const a = shifts[next - 1], b = shifts[next];
    return value + a.amount + (b.amount - a.amount) * (value - a.x) / (b.x - a.x);
  };
  for (const station of stations.values()) {
    station.x = movedXs.get(station.paperId)!;
    station.platforms.forEach(platform => { platform.x = station.x; });
  }
  for (const year of years) {
    for (const quarter of year.quarters) {
      quarter.width = warpX(quarter.x + quarter.width) - warpX(quarter.x);
      quarter.x = warpX(quarter.x);
    }
    year.width = warpX(year.x + year.width) - warpX(year.x);
    year.x = warpX(year.x);
  }
  x = warpX(x);
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
  const nodes = [...stations.values()].map((station) => stationBounds(station, 15));
  // Reserve the planned bend corridors before placing labels. A name must not
  // occupy the only forward path between two closely spaced stops.
  const needsPlatform = (stop: Stop) => stations.get(stop.paperId)!.platforms.length > 1 &&
    !stations.get(stop.paperId)!.continuation;
  const bendLate = (a: Stop, b: Stop) => b.x - a.x > 240 && needsPlatform(b) && !stations.get(b.paperId)!.fork;
  const corridors = [...routesByLine.entries()].flatMap(([id, routes]) =>
    routes.flatMap(members => members.slice(1).flatMap((b, index) => {
      const a = members[index];
      const leavesFork = schools.find(s => s.id === id)?.branchAt?.paperId === a.paperId;
      const departure = { x: a.x + (leavesFork || directDepartures.has(`${id}:${a.paperId}:${b.paperId}`) ? 0 : platformRoom.get(a.paperId)!), y: a.y };
      const arrival = { x: b.x - (directArrivals.has(`${id}:${a.paperId}:${b.paperId}`) ? 0 : platformRoom.get(b.paperId)!), y: b.y };
      if (leavesFork && !returningForks.has(`${id}:${a.paperId}`) && b.x - a.x > Math.abs(b.y - a.y) + 160) {
        // For a sparse branch, leave the paper on a short diagonal and reserve
        // the large change in height nearer its first station. This keeps the
        // long station-free interval beside the trunk instead of diving away.
        const offset = Math.min(48, Math.abs(b.y - a.y));
        const lead = { x: a.x + offset, y: a.y + Math.sign(b.y - a.y) * offset };
        return segments([a, ...elbows(lead, arrival)[1], b]);
      }
      const planned = [a, ...elbows(departure, arrival)[bendLate(a, b) ? 1 : 0], b];
      const otherNodes = [...stations.values()].filter(station =>
        station.paperId !== a.paperId && station.paperId !== b.paperId)
        .map(station => stationBounds(station, 20));
      if (segments(planned).some(({ a, b }) => otherNodes.some(box => crosses(a, b, box)))) {
        return segments([a, ...routeBetween(departure, arrival, otherNodes, [], plotBounds, bendLate(a, b)), b]);
      }
      return segments(planned);
    })));
  const labelOptions = new Map<string, LabelCandidate[]>();
  for (const paper of papers) {
    const station = stations.get(paper.id)!;
    const year = years.find((item) => item.year === paper.year)!;
    const outward = labelSides.get(paper.id) ?? (home.get(paper.id)! < origin + anchorLevel(paper) ? -1 : 1);
    const candidates: LabelCandidate[] = [];
    const clearLabel = (box: Box) => box.x >= year.x + 4 && box.x + box.width <= year.x + year.width - 4 &&
      box.y >= top + 10 && box.y + box.height <= plotBottom - 12 &&
      !corridors.some(({ a, b }) => crosses(a, b, { x: box.x - 8, y: box.y - 8,
        width: box.width + 16, height: box.height + 16 })) &&
      !nodes.some(node => overlaps(box, node, 4));
    for (const gap of station.platforms.length > 1 && !station.continuation ? [20, 28, 36, 44] : [20, 28]) {
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
        const box = { ...station.label,
          x: side < 0 ? station.x - gap - station.label.width : station.x + gap,
          y: station.y - station.label.height / 2 + align * station.label.height / 2 };
        if (clearLabel(box)) candidates.push({ box, score: gap * 4 + Math.abs(align) * 24 + 60 });
      }
    }
    candidates.sort((a, b) => a.score - b.score);
    labelOptions.set(paper.id, candidates);
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
            : station.platforms.filter((platform) => platform.lineId !== id && !station.fork && !station.continuation)
                .map((platform) => ({ x: platform.x - 11, y: platform.y - 11, width: 22, height: 22 })));
        const obstacles = [...labels, ...unrelated];
        const path = route(before, stop, obstacles, occupied, plotBounds, {
          start: stations.get(before.paperId)!.fork?.branchId === id || directDepartures.has(`${id}:${before.paperId}:${stop.paperId}`) ? 0 : platformRoom.get(before.paperId)!,
          end: directArrivals.has(`${id}:${before.paperId}:${stop.paperId}`) ? 0 : platformRoom.get(stop.paperId)!,
        }, bendLate(before, stop));
        track.push(...path.slice(1));
      });
      const last = stops.at(-1);
      if (last && stations.get(last.paperId)!.platforms.length === 1 &&
          !routes.some(other => other !== stops && other[0]?.paperId === last.paperId)) {
        track.push({ x: last.x + 18, y: last.y });
      }
      const simplified = simplify(track);
      occupied.push(...segments(simplified));
      return simplified;
    });
    return { ...school, tracks,
      labelPosition: { x: 24 + order.indexOf(id) * 300, y: plotBottom + 32 } };
  });
  return { width: Math.max(1700, x + 24), height: plotBottom + 96,
    years, methodOrder: order, lines, stations, junctions,
    plotBounds };
}
