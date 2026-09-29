import test from 'node:test';
import assert from 'node:assert/strict';
import { readCatalog } from '../scripts/catalog.mjs';
import { publicationVenue } from '../src/publication.ts';
import { createPublicationLayout } from '../src/components/publication-layout.ts';

const { papers } = await readCatalog();
const layout = createPublicationLayout(papers);
const epsilon = 0.001;
const segments = line => line.track.slice(1).map((b, i) => [line.track[i], b]);
const contains = (box, point) => point.x >= box.x - epsilon && point.x <= box.x + box.width + epsilon
  && point.y >= box.y - epsilon && point.y <= box.y + box.height + epsilon;
const onSegment = (p, a, b) => Math.abs((b.x - a.x) * (p.y - a.y) - (b.y - a.y) * (p.x - a.x)) < epsilon
  && p.x >= Math.min(a.x, b.x) - epsilon && p.x <= Math.max(a.x, b.x) + epsilon
  && p.y >= Math.min(a.y, b.y) - epsilon && p.y <= Math.max(a.y, b.y) + epsilon;

test('every station and label stay in their publication year and venue', () => {
  assert.equal(layout.stations.size, papers.length);
  for (const paper of papers) {
    const station = layout.stations.get(paper.id);
    const year = layout.years.find(item => item.year === paper.year);
    const venue = layout.venues.find(item => item.venue === publicationVenue(paper.venue));
    const cell = { x: year.x, width: year.width, y: venue.y, height: venue.height };
    assert.ok(contains(cell, station), paper.id);
    assert.ok(contains(cell, station.label), `${paper.id} label start`);
    assert.ok(contains(cell, { x: station.label.x + station.label.width, y: station.label.y + station.label.height }), `${paper.id} label end`);
    const line = layout.lines.find(item => item.id === station.lineId);
    assert.ok(segments(line).some(([a, b]) => onSegment(station, a, b)), `${paper.id} is on its method route`);
  }
});

test('hidden month order is consistent across venue rows', () => {
  for (const a of papers) for (const b of papers) {
    if (a.year < b.year || (a.year === b.year && a.timeline?.month < b.timeline?.month)) {
      assert.ok(layout.stations.get(a.id).x < layout.stations.get(b.id).x, `${a.id} precedes ${b.id}`);
    }
  }
});

test('metro routes never turn backwards and use only 0, 45 or 90 degree segments', () => {
  for (const line of layout.lines) {
    const parts = segments(line);
    for (const [index, [a, b]] of parts.entries()) {
      const dx = b.x - a.x, dy = b.y - a.y;
      assert.ok(dx >= -epsilon, `${line.id} backtracks at segment ${index}`);
      assert.ok(Math.abs(dx) < epsilon || Math.abs(dy) < epsilon || Math.abs(Math.abs(dx) - Math.abs(dy)) < epsilon, `${line.id} non-metro angle`);
      assert.ok(contains(layout.plotBounds, a) && contains(layout.plotBounds, b), `${line.id} stays in the map`);
      if (index && Math.abs(dx) < epsilon && Math.abs(parts[index - 1][1].x - parts[index - 1][0].x) < epsilon) {
        assert.ok(dy * (parts[index - 1][1].y - parts[index - 1][0].y) >= 0, `${line.id} vertical U-turn`);
      }
    }
  }
  const boxes = [...layout.stations.values()].map(station => station.label);
  for (let i = 0; i < boxes.length; i++) for (let j = i + 1; j < boxes.length; j++) {
    const a = boxes[i], b = boxes[j];
    assert.ok(a.x + a.width <= b.x || b.x + b.width <= a.x || a.y + a.height <= b.y || b.y + b.height <= a.y, 'station labels do not overlap');
  }
});

test('layout is deterministic and handles a lone station or empty catalog', () => {
  const reversed = createPublicationLayout([...papers].reverse());
  assert.deepEqual(reversed.lines, layout.lines);
  for (const paper of papers) assert.deepEqual(reversed.stations.get(paper.id), layout.stations.get(paper.id));
  assert.equal(createPublicationLayout([papers[0]]).stations.size, 1);
  assert.equal(createPublicationLayout([]).stations.size, 0);
});
