import type { Box, Point, PublicationLayout } from "./publication-layout";
import { stationBounds } from "./publication-layout.ts";

const overlaps = (a: Box, b: Box, gap = 0) =>
  a.x < b.x + b.width + gap && a.x + a.width + gap > b.x &&
  a.y < b.y + b.height + gap && a.y + a.height + gap > b.y;

function crossesBox(a: Point, b: Point, box: Box): boolean {
  let lo = 0, hi = 1;
  for (const [origin, delta, min, max] of [[a.x, b.x - a.x, box.x, box.x + box.width],
    [a.y, b.y - a.y, box.y, box.y + box.height]]) {
    if (Math.abs(delta) < .001) { if (origin < min || origin > max) return false; }
    else {
      const first = (min - origin) / delta, last = (max - origin) / delta;
      lo = Math.max(lo, Math.min(first, last)); hi = Math.min(hi, Math.max(first, last));
      if (lo > hi) return false;
    }
  }
  return true;
}

const CELL = 24;
const MIN_POCKET_WIDTH = 900;
const MIN_POCKET_HEIGHT = 216;
const MIN_POCKET_AREA = 400_000;

// Discover broad empty rectangles between the actual rails. A histogram sweep
// finds pockets of any size, without predetermined positions or a motif count.
function whitespacePockets(network: PublicationLayout, occupied: Box[]): Box[] {
  const plot = network.plotBounds;
  const columns = Math.floor(plot.width / CELL), rows = Math.floor(plot.height / CELL);
  const tracks = network.lines.flatMap(line => line.tracks.flatMap(track =>
    track.slice(1).map((b, i) => ({ a: track[i], b }))));
  const blocked = Array.from({ length: rows }, () => new Uint8Array(columns));
  for (let column = 0; column < columns; column++) {
    const x = plot.x + column * CELL;
    const center = x + CELL / 2;
    const levels = tracks.flatMap(({ a, b }) => {
      if (a.x === b.x || center < Math.min(a.x, b.x) || center > Math.max(a.x, b.x)) return [];
      return [a.y + (b.y - a.y) * (center - a.x) / (b.x - a.x)];
    });
    const top = Math.min(...levels) + 36, bottom = Math.max(...levels) - 36;
    const nearbyTracks = tracks.filter(({ a, b }) =>
      Math.min(a.x, b.x) <= x + CELL + 36 && Math.max(a.x, b.x) >= x - 36);
    const nearbyBoxes = occupied.filter(box => box.x <= x + CELL + 36 && box.x + box.width >= x - 36);
    for (let row = 0; row < rows; row++) {
      const y = plot.y + row * CELL;
      const cell = { x, y, width: CELL, height: CELL };
      const clearance = { x: x - 36, y: y - 36, width: CELL + 72, height: CELL + 72 };
      blocked[row][column] = Number(y < top || y + CELL > bottom ||
        nearbyBoxes.some(box => overlaps(cell, box, 36)) ||
        nearbyTracks.some(({ a, b }) => crossesBox(a, b, clearance)));
    }
  }
  const heights = new Uint32Array(columns);
  const pockets: Box[] = [];
  for (let row = 0; row < rows; row++) {
    for (let column = 0; column < columns; column++)
      heights[column] = blocked[row][column] ? 0 : heights[column] + 1;
    const stack: { start: number; height: number }[] = [];
    for (let column = 0; column <= columns; column++) {
      const height = column === columns ? 0 : heights[column];
      let start = column;
      while (stack.length && stack.at(-1)!.height > height) {
        const previous = stack.pop()!;
        start = previous.start;
        const width = (column - start) * CELL, pocketHeight = previous.height * CELL;
        if (width >= MIN_POCKET_WIDTH && pocketHeight >= MIN_POCKET_HEIGHT && width * pocketHeight >= MIN_POCKET_AREA)
          pockets.push({ x: plot.x + start * CELL, y: plot.y + (row + 1) * CELL - pocketHeight,
            width, height: pocketHeight });
      }
      if (height && (!stack.length || stack.at(-1)!.height < height)) stack.push({ start, height });
    }
  }
  return pockets.sort((a, b) => b.width * b.height - a.width * a.height || a.y - b.y || a.x - b.x);
}

// Decoration follows the completed network and never moves a route or label.
export function createResearchBackdrop(network: PublicationLayout, reserved: Box[] = []): { mountains: Box[] } {
  const plot = network.plotBounds;
  const occupied: Box[] = [...reserved,
    ...network.years.map(year => ({ x: year.x - 6, y: plot.y, width: 12, height: plot.height })),
    ...[...network.stations.values()].flatMap(s => [s.label, stationBounds(s, 20)])];
  const mountains: Box[] = [];
  const usedPockets: Box[] = [];
  for (const pocket of whitespacePockets(network, occupied)) {
    // Keep one centered accent per spacious pocket, with breathing room between
    // neighboring pockets. Narrow fragments remain intentionally undecorated.
    if (usedPockets.some(other => overlaps(pocket, other, 72))) continue;
    const width = Math.min(460, pocket.width * .5, pocket.height * .5 * 172 / 52,
      Math.sqrt(pocket.width * pocket.height) * .75);
    const height = width * 52 / 172;
    mountains.push({ x: pocket.x + (pocket.width - width) / 2,
      y: pocket.y + (pocket.height - height) / 2, width, height });
    usedPockets.push(pocket);
  }
  return { mountains: mountains.sort((a, b) => a.x - b.x || a.y - b.y) };
}
