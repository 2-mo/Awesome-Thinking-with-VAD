import type { Box, Point, PublicationLayout } from "./publication-layout";
import { labelWidth, railLabelDistance, stationBounds } from "./publication-layout.ts";
import { MAP_FONT_SIZE } from "./map-typography.ts";

export type RouteLabel = Box & { lineId: string; text: string; color: string };
export const ROUTE_LABEL_SIZE = MAP_FONT_SIZE.primary;
const overlaps = (a: Box, b: Box, gap: number) =>
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

// Names live beside their own rails. Position them after paper labels, using
// the full catalog so filtering and exported SVGs share the same annotation.
export function createMapRouteLabels(network: PublicationLayout): RouteLabel[] {
  const occupied = [...network.stations.values()].flatMap(s => [s.label, stationBounds(s, 20)]);
  const segments = network.lines.flatMap(line => line.tracks.flatMap(track =>
    track.slice(1).map((b, i) => ({ a: track[i], b, lineId: line.id }))));
  const plot = network.plotBounds;
  const labels: RouteLabel[] = [];
  // Soft positions distribute direction names over the clear middle of their
  // own corridors, without anchoring every annotation at the left edge.
  const preferredCenters: Record<string, number> = {
    synthesis: .82, alignment: .44, detection: .49, understanding: .48, evidence: .42,
    evaluation: .23, reasoning: .39, explanation: .43,
  };
  const choices = network.lines.map(line => {
    const width = labelWidth(line.label, ROUTE_LABEL_SIZE) + 12, height = ROUTE_LABEL_SIZE + 6;
    const candidates: { box: Box; score: number }[] = [];
    for (const [trackIndex, track] of line.tracks.entries()) {
      for (let i = 1; i < track.length; i++) {
        const a = track[i - 1], b = track[i];
        if (a.y !== b.y || b.x - a.x < 32) continue;
        // On a short branch, slide the name around its level segment instead
        // of testing only its center. Collision and nearest-rail checks still
        // apply to the whole label, with at least 128 units of rail overlap.
        const overlap = Math.min(128, b.x - a.x);
        const left = b.x - a.x >= width + 24 ? a.x : Math.max(plot.x + 12, a.x - width + overlap);
        const right = b.x - a.x >= width + 24 ? b.x - width : Math.min(plot.x + plot.width - width - 12, b.x - overlap);
        for (let x = left; x <= right + .01; x += 8) for (const side of [-1, 1]) {
          for (const gap of [12, 18, 24, 36]) {
            const y = side < 0 ? a.y - gap - height : a.y + gap;
            const box = { x, y, width, height };
            if (x < plot.x + 12 || x + width > plot.x + plot.width - 12 ||
                y < plot.y + 12 || y + height > plot.y + plot.height - 12) continue;
            if (occupied.some(other => overlaps(box, other, 14))) continue;
            const clearance = { x: x - 10, y: y - 10, width: width + 20, height: height + 20 };
            if (segments.some(({ a, b }) => crossesBox(a, b, clearance))) continue;
            const otherDistance = Math.min(...segments.filter(segment => segment.lineId !== line.id)
              .map(({ a, b }) => railLabelDistance(box, a, b)));
            // A colored name should read as attached to just one rail, even
            // where two directions run alongside each other.
            if (otherDistance < gap + 28) continue;
            candidates.push({ box, score: (gap - 12) * 6 + (side > 0 ? 8 : 0) + trackIndex * 10 +
              Math.max(0, 64 - otherDistance) * 2 +
              Math.abs(x + width / 2 - (a.x + b.x) / 2) * .02 +
              Math.abs(x + width / 2 - (plot.x + plot.width * (preferredCenters[line.id] ?? .5))) * .08 +
              Math.max(0, width + 80 - (b.x - a.x)) * .5 });
          }
        }
      }
    }
    candidates.sort((a, b) => a.score - b.score || a.box.x - b.box.x || a.box.y - b.box.y);
    return { line, candidates };
  });
  // Constrained directions choose first; a name that cannot fit is retained
  // in the compact key instead of being hidden or drawn over a paper.
  choices.sort((a, b) => a.candidates.length - b.candidates.length || a.line.id.localeCompare(b.line.id));
  for (const { line, candidates } of choices) {
    const chosen = candidates.find(({ box }) => !labels.some(other => overlaps(box, other, 24)));
    if (chosen) labels.push({ ...chosen.box, lineId: line.id, text: line.label, color: line.color });
  }
  return labels.sort((a, b) => network.methodOrder.indexOf(a.lineId) - network.methodOrder.indexOf(b.lineId));
}
