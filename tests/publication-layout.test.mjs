import test from 'node:test';
import assert from 'node:assert/strict';
import { readCatalog } from '../scripts/catalog.mjs';
import { paperMethods } from '../src/publication.ts';
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

test('each paper stays in its publication year and on all of its method routes', () => {
  assert.equal(layout.stations.size, papers.length);
  for (const paper of papers) {
    const station = layout.stations.get(paper.id);
    const year = layout.years.find(item => item.year === paper.year);
    const cell = { x: year.x, width: year.width, y: layout.plotBounds.y, height: layout.plotBounds.height };
    assert.ok(contains(cell, station), paper.id);
    assert.ok(contains(cell, station.label), `${paper.id} label start`);
    assert.ok(contains(cell, { x: station.label.x + station.label.width, y: station.label.y + station.label.height }), `${paper.id} label end`);
    assert.deepEqual(station.lineIds, paperMethods(paper));
    for (const id of station.lineIds) {
      const line = layout.lines.find(item => item.id === id);
      assert.ok(segments(line).some(([a, b]) => onSegment(station, a, b)), `${paper.id} is on ${id}`);
    }
  }
});

test('hidden month order is consistent across the topology', () => {
  for (const a of papers) for (const b of papers) {
    if (a.year < b.year || (a.year === b.year && a.timeline?.month < b.timeline?.month)) {
      assert.ok(layout.stations.get(a.id).x < layout.stations.get(b.id).x, `${a.id} precedes ${b.id}`);
    }
  }
});

test('quarter bands cover recent years and contain the correct paper months', () => {
  for (const year of layout.years.filter(item => item.year >= 2025)) {
    assert.deepEqual(year.quarters.slice(0, 4).map(item => item.quarter), [1, 2, 3, 4]);
    assert.equal(year.quarters.reduce((sum, item) => sum + item.count, 0), year.count);
    assert.equal(year.quarters[0].x, year.x);
    const last = year.quarters.at(-1);
    assert.equal(last.x + last.width, year.x + year.width);
    for (const [index, quarter] of year.quarters.entries()) {
      assert.ok(quarter.width > 0);
      if (index) assert.equal(year.quarters[index - 1].x + year.quarters[index - 1].width, quarter.x);
      const members = papers.filter(paper => paper.year === year.year &&
        (paper.timeline ? Math.ceil(paper.timeline.month / 3) : null) === quarter.quarter);
      assert.equal(quarter.count, members.length);
      for (const paper of members) {
        const station = layout.stations.get(paper.id);
        assert.ok(station.x > quarter.x && station.x < quarter.x + quarter.width, `${paper.id} quarter`);
      }
    }
  }
});

test('unknown months remain outside the four labeled quarters', () => {
  const paper = { ...papers[0], year: 2026, timeline: undefined };
  const network = createPublicationLayout([paper]);
  const quarters = network.years[0].quarters;
  assert.deepEqual(quarters.map(item => item.quarter), [1, 2, 3, 4, null]);
  assert.equal(quarters.slice(0, 4).reduce((sum, item) => sum + item.count, 0), 0);
  const unknown = quarters.at(-1);
  assert.equal(unknown.count, 1);
  assert.ok(network.stations.get(paper.id).x > unknown.x);
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

// Venue metadata must travel with the paper, not constrain its vertical rank.
test('station coordinates are independent of conference assignments', () => {
  const changed = createPublicationLayout(papers.map(paper => ({ ...paper, venue: 'arXiv' })));
  for (const paper of papers) {
    const a = layout.stations.get(paper.id), b = changed.stations.get(paper.id);
    assert.equal(a.x, b.x, `${paper.id} time`);
    assert.equal(a.y, b.y, `${paper.id} height`);
  }
});

test('shared methods form genuine single-paper interchange stations', () => {
  const transfers = papers.filter(paper => paperMethods(paper).length > 1);
  assert.deepEqual(transfers.map(paper => paper.id).sort(), ['a2seek', 'memovad', 'td-vad']);
  for (const paper of transfers) {
    const [a, b] = paperMethods(paper).map(id => layout.methodOrder.indexOf(id));
    assert.equal(Math.abs(a - b), 1, `${paper.id} connects neighboring trunks`);
  }
  assert.equal(layout.stations.size, papers.length, 'interchanges do not duplicate papers');
});

const crossesBox = (a, b, box) => {
  let lo = 0, hi = 1;
  for (const [origin, delta, min, max] of [[a.x, b.x - a.x, box.x + .01, box.x + box.width - .01],
    [a.y, b.y - a.y, box.y + .01, box.y + box.height - .01]]) {
    if (Math.abs(delta) < epsilon) { if (origin <= min || origin >= max) return false; }
    else {
      const u = (min - origin) / delta, v = (max - origin) / delta;
      lo = Math.max(lo, Math.min(u, v)); hi = Math.min(hi, Math.max(u, v));
      if (lo >= hi) return false;
    }
  }
  return hi > 0 && lo < 1;
};
test('tracks avoid paper labels and unrelated station markers', () => {
  for (const line of layout.lines) for (const [a, b] of segments(line)) {
    for (const station of layout.stations.values()) {
      assert.ok(!crossesBox(a, b, station.label), `${line.id} crosses ${station.paperId} label`);
      if (!station.lineIds.includes(line.id)) {
        assert.ok(!crossesBox(a, b, { x: station.x - 10, y: station.y - 10, width: 20, height: 20 }),
          `${line.id} passes through unrelated ${station.paperId}`);
      }
    }
  }
});
