import type { Box, Point, PublicationLayout } from "./publication-layout";

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

// Decoration is placed only after the compact network is complete. It never
// reserves a region, shifts a route or assigns research meaning to blank space.
export function createResearchBackdrop(network: PublicationLayout, reserved: Box[] = []): { mountains: Box[] } {
  const plot = network.plotBounds;
  const occupied: Box[] = [...reserved, ...[...network.stations.values()].flatMap(s => [s.label,
    { x: s.x - 20, y: s.platforms[0].y - 20, width: 40,
      height: s.platforms.at(-1)!.y - s.platforms[0].y + 40 }])];
  const tracks = network.lines.flatMap(line => line.tracks.flatMap(track => track.slice(1).map((b, i) => ({ a: track[i], b }))));
  const mountains: Box[] = [];
  for (const [fractionX, fractionY] of [[.3, .35], [.72, .72]]) {
    const candidates: { box: Box; score: number }[] = [];
    for (let x = plot.x + 24; x + 180 < plot.x + plot.width - 24; x += 48) {
      for (let y = plot.y + 24; y + 54 < plot.y + plot.height - 24; y += 24) {
        const box = { x, y, width: 180, height: 54 };
        candidates.push({ box, score: Math.abs(x + 90 - (plot.x + plot.width * fractionX)) +
          Math.abs(y + 27 - (plot.y + plot.height * fractionY)) * 2 });
      }
    }
    candidates.sort((a, b) => a.score - b.score);
    const chosen = candidates.find(({ box }) => !occupied.some(other => overlaps(box, other, 18)) &&
      !tracks.some(({ a, b }) => crossesBox(a, b, { x: box.x - 12, y: box.y - 12,
        width: box.width + 24, height: box.height + 24 })));
    if (chosen) { mountains.push(chosen.box); occupied.push(chosen.box); }
  }
  return { mountains };
}
