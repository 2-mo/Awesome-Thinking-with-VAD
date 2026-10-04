import type { Box, PublicationLayout } from "./publication-layout";
import { labelWidth, stationBounds } from "./publication-layout.ts";
import { createMapRouteLabels, ROUTE_LABEL_SIZE } from "./map-route-labels.ts";
import type { RouteLabel } from "./map-route-labels";
import { contributionLabels } from "../publication.ts";
import { MAP_FONT_SIZE } from "./map-typography.ts";

export type MapLegend = Box & { embedded: boolean; fallbackLineIds: string[] };
export const LEGEND_ROUTE_TOP = 54;
export const LEGEND_ROUTE_STEP = 42;
export const legendDivider = (lineCount: number) => lineCount ? LEGEND_ROUTE_TOP + lineCount * LEGEND_ROUTE_STEP - 8 : 0;
const LEGEND_PADDING = 16;
let itemX = LEGEND_PADDING + labelWidth("STATION TYPES", MAP_FONT_SIZE.secondary) + 40;
export const LEGEND_ITEMS = (["method", "resource", "hybrid"] as const).map(kind => {
  const item = { kind, x: itemX };
  itemX += 28 + labelWidth(contributionLabels[kind], MAP_FONT_SIZE.primary) + 40;
  return item;
});
const LEGEND_ROW_WIDTH = itemX - 40 + LEGEND_PADDING;
export const LEGEND_ROW_CENTER = 32;
const LEGEND_ROW_HEIGHT = 64;

const overlaps = (a: Box, b: Box, gap: number) =>
  a.x < b.x + b.width + gap && a.x + a.width + gap > b.x &&
  a.y < b.y + b.height + gap && a.y + a.height + gap > b.y;

function crossesBox(a: { x: number; y: number }, b: { x: number; y: number }, box: Box): boolean {
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

// Fit one horizontal key into the open band below the criteria trunk. Filtering
// uses the same full network, so the key never moves or hides a paper.
export function createMapLegend(network: PublicationLayout, routeLabels: RouteLabel[] = createMapRouteLabels(network)): MapLegend {
  const plot = network.plotBounds;
  const fallbackLines = network.lines.filter(line => !routeLabels.some(label => label.lineId === line.id));
  const fallbackLineIds = fallbackLines.map(line => line.id);
  const width = Math.max(LEGEND_ROW_WIDTH, ...fallbackLines.map(line =>
    labelWidth(line.label, ROUTE_LABEL_SIZE) + (line.branchOf ? 80 : 0) + 76));
  const height = legendDivider(fallbackLines.length) + LEGEND_ROW_HEIGHT;
  const criteriaTrunk = network.lines.find(line => line.id === "explanation")?.tracks[0];
  const preferredY = criteriaTrunk?.length
    ? Math.max(...criteriaTrunk.map(point => point.y)) + 80
    : plot.y + plot.height * .88;
  const occupied = [...routeLabels, ...[...network.stations.values()].flatMap(station =>
    [station.label, stationBounds(station, 20)])];
  const tracks = network.lines.flatMap(line => line.tracks.flatMap(track => track.slice(1).map((b, i) => ({ a: track[i], b }))));
  const candidates: { box: Box; score: number }[] = [];
  for (let x = plot.x + 24; x + width <= plot.x + Math.min(plot.width, network.width * .34); x += 24) {
    for (let y = plot.y + plot.height - height - 24; y >= plot.y + plot.height * .75; y -= 12) {
      const box = { x, y, width, height };
      const clearance = { x: x - 20, y: y - 20, width: width + 40, height: height + 40 };
      if (occupied.some(other => overlaps(box, other, 24)) ||
          tracks.some(({ a, b }) => crossesBox(a, b, clearance))) continue;
      candidates.push({ box, score: (x - plot.x) * 3 + Math.abs(y - preferredY) });
    }
  }
  candidates.sort((a, b) => a.score - b.score || a.box.y - b.box.y);
  if (candidates.length) return { ...candidates[0].box, embedded: true, fallbackLineIds };
  // Small/custom catalogs may have no sufficiently large in-map pocket.
  return { x: plot.x + 24, y: plot.y + plot.height + 24, width, height, embedded: false, fallbackLineIds };
}
