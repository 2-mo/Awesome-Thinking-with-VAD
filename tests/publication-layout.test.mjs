import test from 'node:test';
import assert from 'node:assert/strict';
import { readCatalog } from '../scripts/catalog.mjs';
import { isMapPaper, paperMethods, publicationKind, timelineYear } from '../src/publication.ts';
import { createPublicationLayout as createLayout, isInterchangeStation, railLabelDistance, stationBounds, stationVenue } from '../src/components/publication-layout.ts';
import { createMapLegend } from '../src/components/map-legend.ts';
import { createMapRouteLabels } from '../src/components/map-route-labels.ts';
import { createResearchBackdrop } from '../src/components/research-regions.ts';

const { papers: catalogPapers, clusters } = await readCatalog();
const papers = catalogPapers.filter(isMapPaper);
const createPublicationLayout = papers => createLayout(papers, clusters);
const layout = createPublicationLayout(papers);
const epsilon = 0.001;
const segments = line => line.tracks.flatMap(track => track.slice(1).map((b, i) => [track[i], b]));
const contains = (box, point) => point.x >= box.x - epsilon && point.x <= box.x + box.width + epsilon
  && point.y >= box.y - epsilon && point.y <= box.y + box.height + epsilon;
const onSegment = (p, a, b) => Math.abs((b.x - a.x) * (p.y - a.y) - (b.y - a.y) * (p.x - a.x)) < epsilon
  && p.x >= Math.min(a.x, b.x) - epsilon && p.x <= Math.max(a.x, b.x) + epsilon
  && p.y >= Math.min(a.y, b.y) - epsilon && p.y <= Math.max(a.y, b.y) + epsilon;
const railHeight = (line, x) => {
  const [a, b] = segments(line).find(([a, b]) => b.x > a.x && a.x <= x && b.x >= x);
  return a.y + (b.y - a.y) * (x - a.x) / (b.x - a.x);
};

test('criteria and reasoning remain separate adjacent directions without turning either into a branch', () => {
  assert.equal(Math.abs(layout.methodOrder.indexOf('explanation') - layout.methodOrder.indexOf('reasoning')), 1);
  assert.ok(layout.lines.some(line => line.id === 'explanation' && !line.branchOf));
  assert.ok(layout.lines.some(line => line.id === 'reasoning' && !line.branchOf));
  assert.ok(!layout.junctions.some(j => j.parentId === 'explanation' || j.branchId === 'explanation'));
});

test('the evaluation branch leaves Vad-R1 toward Cue-R1 above reasoning while criteria stays below', () => {
  const parent = layout.methodOrder.indexOf('reasoning');
  assert.equal(layout.methodOrder.indexOf('evaluation'), parent - 1);
  assert.equal(layout.methodOrder.indexOf('explanation'), parent + 1);
  const branch = layout.lines.find(line => line.id === 'evaluation');
  const fork = layout.stations.get('vad-r1');
  assert.deepEqual(branch.tracks[0][0], { x: fork.x, y: fork.y });
  assert.ok(branch.tracks[0][1].x > fork.x && branch.tracks[0][1].y < fork.y, 'the branch leaves to the upper right');
  const stops = [...layout.stations.values()].filter(s => s.lineIds.includes('evaluation') && s.x >= fork.x).sort((a, b) => a.x - b.x);
  assert.deepEqual(stops.slice(0, 2).map(s => s.paperId), ['vad-r1', 'cuebench']);
  assert.deepEqual([...layout.stations.get('cuva').lineIds].sort(), ['evaluation', 'reasoning']);
});

test('three local paths keep level shelves and rejoin at Vad-R1-Plus', () => {
  const root = layout.stations.get('vad-r1'), plus = layout.stations.get('vad-r1-plus');
  const cue = layout.stations.get('cuebench'), fine = layout.stations.get('finevau');
  const lower = layout.stations.get('vad-dpo');
  const arrival = plus.platforms.find(p => p.lineId === 'evaluation');
  const trunk = plus.platforms.find(p => p.lineId === 'reasoning');
  assert.ok(arrival && trunk);
  assert.deepEqual(fine.lineIds, ['evaluation'], 'FineVAU remains an ordinary branch stop');
  assert.equal(cue.y, fine.y);
  assert.ok(root.y - cue.y >= 112 && lower.y - root.y >= 112, 'three paths have clear vertical separation');
  for (const id of ['urf-zs-hvaa', 'targetvau']) {
    assert.equal(layout.stations.get(id).y, root.y, `${id} stays on the level trunk`);
  }
  assert.equal(trunk.y, root.y);
  assert.ok(fine.x - cue.x >= (cue.label.width + fine.label.width) / 2 + 16);
  const evaluation = layout.lines.find(line => line.id === 'evaluation').tracks[0];
  const reasoning = layout.lines.find(line => line.id === 'reasoning');
  const side = reasoning.tracks.find(track => track[0].x === root.x && track[0].y === root.y);
  assert.ok(side, 'same-color side path leaves the original paper');
  for (const [track, endpoint] of [[evaluation, arrival], [side, trunk]]) {
    assert.equal(track.at(-1).x, endpoint.x, 'return is exactly at the shared paper');
    assert.equal(track.at(-1).y, endpoint.y);
  }
  assert.ok(segments({ tracks: [reasoning.tracks[0]] }).some(([a,b]) =>
    a.x <= root.x && b.x >= plus.x && a.y === root.y && b.y === root.y),
    'one straight trunk crosses the entire local corridor');
});

test('dense reading groups use clearly separated returning shelves', () => {
  for (const [lineId, rootId, endId, interior] of [
    ['understanding', 'where-what', 'panda', ['eventvad', 'vadtree']],
  ]) {
    const root = layout.stations.get(rootId).platforms.find(p => p.lineId === lineId);
    const end = layout.stations.get(endId).platforms.find(p => p.lineId === lineId);
    const arm = interior.map(id => layout.stations.get(id));
    assert.equal(root.y, end.y, 'fork and return retain a level trunk');
    assert.ok(arm.every(s => s.y === arm[0].y && Math.abs(s.y - root.y) >= 144));
    assert.ok(root.x < arm[0].x && arm.at(-1).x < end.x);
    const track = layout.lines.find(line => line.id === lineId).tracks.find(track =>
      track[0].x === root.x && track[0].y === root.y && track.at(-1).x === end.x && track.at(-1).y === end.y);
    assert.ok(track, 'the side path leaves and returns at the named platforms without a tail stub');
    assert.equal(isInterchangeStation(layout.stations.get(rootId)), ['va-gpt', 'memovad'].includes(rootId),
      'branching does not change the actual method interchange semantics');
  }
});

test('industrial reasoning ends independently and the later branches keep open terminals', () => {
  const plus = layout.stations.get('vad-r1-plus');
  const industrial = ['iad-r1', 'judo', 'o-vad'].map(id => layout.stations.get(id));
  assert.ok(industrial.every(s => s.y === industrial[0].y && s.y > plus.y));
  const line = layout.lines.find(line => line.id === 'reasoning');
  const arm = line.tracks.find(track => track[0].x === plus.x && track[0].y === plus.y);
  assert.ok(arm && arm.at(-1).x === industrial.at(-1).x + 18);
  for (const id of ['las-vad', 'stch', 'cg-coe', 'clue-vad', 'avar'])
    assert.equal(layout.stations.get(id).y, plus.y, 'ordinary reasoning stops retain a clear main corridor');
  for (const [lineId, rootId, terminalId] of [
    ['detection', 'td-vad', 'deal-vad'],
    ['explanation', 'lrpo', 'ca-judge'],
  ]) {
    const root = layout.stations.get(rootId).platforms.find(p => p.lineId === lineId);
    const terminal = layout.stations.get(terminalId);
    const track = layout.lines.find(line => line.id === lineId).tracks.find(track =>
      track[0].x === root.x && track[0].y === root.y && track.at(-1).x === terminal.x + 18);
    assert.ok(track, `${terminalId} ends independently with a terminal stub`);
    assert.equal(track.at(-1).y, terminal.y);
  }
});

test('Vad-R1 and its Plus extension retain separate dated stations', () => {
  const original = layout.stations.get('vad-r1'), extension = layout.stations.get('vad-r1-plus');
  assert.ok(original && extension);
  assert.ok(original.x < extension.x);
  assert.equal(layout.stations.size, catalogPapers.filter(isMapPaper).length);
  assert.ok(layout.stations.get('crcl').x < original.x, 'CRCL stays before the later R1 papers');
});

test('the labels after the R1 return have breathing room without changing the level trunk', () => {
  const plus = layout.stations.get('vad-r1-plus'), next = layout.stations.get('srvau-r1');
  const later = layout.stations.get('las-vad');
  assert.ok(next.label.x - (plus.label.x + plus.label.width) >= 40,
    'the returning fan label clears the first continuing name');
  assert.ok(later.label.x - (next.label.x + next.label.width) >= 56,
    'same-side labels remain separated across the staggered run');
  for (const id of ['srvau-r1', 'adversa', 'las-vad']) {
    assert.equal(layout.stations.get(id).y, plus.y, 'only local horizontal spacing changes');
  }
});

test('each paper stays in its publication year and on all of its method routes', () => {
  assert.equal(layout.stations.size, papers.length);
  for (const paper of papers) {
    const station = layout.stations.get(paper.id);
    const year = layout.years.find(item => item.year === timelineYear(paper));
    const cell = { x: year.x, width: year.width, y: layout.plotBounds.y, height: layout.plotBounds.height };
    assert.ok(contains(cell, station), paper.id);
    assert.ok(contains(cell, station.label), `${paper.id} label start`);
    assert.ok(contains(cell, { x: station.label.x + station.label.width, y: station.label.y + station.label.height }), `${paper.id} label end`);
    assert.deepEqual(station.lineIds, paperMethods(paper));
    for (const id of station.lineIds) {
      const line = layout.lines.find(item => item.id === id);
      const platform = station.platforms.find(item => item.lineId === id);
      assert.ok(contains(cell, platform), `${paper.id} platform stays in year`);
      assert.ok(segments(line).some(([a, b]) => onSegment(platform, a, b)), `${paper.id} is on ${id}`);
    }
  }
});

test('hidden month order is consistent across the topology', () => {
  for (const a of papers) for (const b of papers) {
    if (timelineYear(a) < timelineYear(b) || (timelineYear(a) === timelineYear(b) && a.timeline?.month < b.timeline?.month)) {
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
      const members = papers.filter(paper => timelineYear(paper) === year.year &&
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
  const transfers = papers.filter(paper => isInterchangeStation(layout.stations.get(paper.id)));
  assert.deepEqual(transfers.map(paper => paper.id).sort(), ['a2seek', 'anom-pi', 'anomalyruler', 'ex-vad', 'memovad', 'panda', 'reactvau', 'va-gpt']);
  for (const paper of transfers) {
    const station = layout.stations.get(paper.id);
    // A line can leave its nominal band to meet a shared paper. Check the
    // actual connector, rather than treating vertical method order as rails.
    const body = stationBounds(station, 20);
    for (const line of layout.lines.filter(line => !station.lineIds.includes(line.id))) {
      assert.ok(segments(line).every(([a, b]) => !crossesBox(a, b, body)),
        `${paper.id} connector clears the unrelated ${line.id} route`);
    }
  }
  assert.equal(layout.stations.size, papers.length, 'interchanges do not duplicate papers');
});

test('LRPO belongs only to the red route and TTHF stays on the separate detection line', () => {
  const lrpo = layout.stations.get('lrpo');
  assert.ok(!isInterchangeStation(lrpo));
  assert.deepEqual(lrpo.lineIds, ['explanation']);
  assert.ok(lrpo.platforms.every(p => p.x === lrpo.x && p.y === lrpo.y));
  for (const id of lrpo.lineIds) {
    assert.ok(segments(layout.lines.find(line => line.id === id)).some(([a, b]) => onSegment(lrpo, a, b)));
  }
  assert.ok(!segments(layout.lines.find(line => line.id === 'reasoning')).some(([a, b]) => onSegment(lrpo, a, b)),
    'the green route bypasses LRPO');
  const tthf = layout.stations.get('tthf'), hawk = layout.stations.get('hawk');
  assert.deepEqual(tthf.lineIds, ['detection']);
  assert.deepEqual(hawk.lineIds, ['alignment']);
  assert.ok(Math.abs(tthf.y - hawk.y) >= 144, 'detection and understanding occupy distinct corridors');
});

test('a returning evaluation branch meets Vad-R1-Plus at one ordinary station', () => {
  const station = layout.stations.get('vad-r1-plus');
  assert.deepEqual(station.lineIds, ['reasoning', 'evaluation']);
  assert.deepEqual(station.merge, { parentId: 'reasoning', branchId: 'evaluation' });
  assert.ok(!isInterchangeStation(station));
  assert.ok(station.platforms.every(platform => platform.x === station.x && platform.y === station.y));
  const evaluation = layout.lines.find(line => line.id === 'evaluation').tracks[0];
  assert.equal(evaluation.at(-1).x, station.x);
  assert.equal(evaluation.at(-1).y, station.y);
  assert.ok(evaluation.at(-2).x < station.x && evaluation.at(-2).y < station.y,
    'the upper branch meets the trunk directly without a duplicate horizontal approach');
  const root = layout.stations.get('vad-r1');
  assert.ok(!isInterchangeStation(root));
  assert.equal(stationVenue(papers.find(paper => paper.id === root.paperId)), 'NeurIPS');
});

test('the early evaluation route leaves CUVA and returns at HoloTrace using ordinary nodes', () => {
  const root = layout.stations.get('cuva'), terminal = layout.stations.get('holotrace');
  const track = layout.lines.find(line => line.id === 'evaluation').tracks[2];
  assert.deepEqual(root.fork, { parentId: 'reasoning', branchId: 'evaluation' });
  assert.deepEqual(terminal.merge, root.fork);
  for (const station of [root, terminal]) {
    assert.ok(!isInterchangeStation(station));
    assert.ok(station.platforms.every(p => p.x === station.x && p.y === station.y));
  }
  assert.deepEqual(track[0], { x: root.x, y: root.y });
  assert.equal(track.at(-1).x, terminal.x);
  assert.equal(track.at(-1).y, terminal.y);
  const upperApproach = track.findLast(point => point.y < terminal.y);
  assert.ok(upperApproach && upperApproach.x < terminal.x && track.every(point => point.x <= terminal.x),
    'the expanded evaluation corridor returns from above without overshooting HoloTrace');
});

test('PANDA connects memory and active observation at distinct platforms on both routes', () => {
  const station = layout.stations.get('panda');
  assert.deepEqual([...station.lineIds].sort(), ['evidence', 'understanding']);
  assert.ok(isInterchangeStation(station));
  for (const platform of station.platforms) {
    const line = layout.lines.find(line => line.id === platform.lineId);
    assert.ok(segments(line).some(([a, b]) => onSegment(platform, a, b)));
  }
});

test('publication styling distinguishes journals and preprints without misclassifying AAAI proceedings', () => {
  for (const id of ['vadclip', 'finevau', 'targetvau', 'judo']) {
    assert.equal(publicationKind(papers.find(paper => paper.id === id)), 'conference', id);
  }
  for (const id of ['pel', 'mpgdfl', 'crcl', 'adversa', 'ecva-anomshield', 'promptvad']) {
    assert.equal(publicationKind(catalogPapers.find(paper => paper.id === id)), 'journal', id);
  }
  for (const paper of papers.filter(paper => paper.venue === 'arXiv')) {
    assert.equal(publicationKind(paper), 'preprint', paper.id);
  }
});

test('LAVIDA splits synthesis from the continuing detection route at one sourced station', () => {
  const paper = papers.find(paper => paper.id === 'lavida');
  const station = layout.stations.get(paper.id);
  assert.equal(paper.cluster, 'synthesis');
  assert.deepEqual(paperMethods(paper), ['synthesis', 'detection']);
  assert.ok(paper.secondaryMethods[0].evidence.url.includes('2602.19248'));
  assert.equal(station.platforms.length, 2);
  assert.ok(!isInterchangeStation(station), 'a named fork uses one ordinary marker');
  assert.ok(station.platforms.every(p => p.x === station.x && p.y === station.y));
  assert.deepEqual(station.fork, { parentId: 'detection', branchId: 'synthesis' });
  assert.ok(layout.junctions.some(junction => junction.paperId === 'lavida'));
  for (const platform of station.platforms) {
    assert.ok(segments(layout.lines.find(line => line.id === platform.lineId))
      .some(([a, b]) => onSegment(platform, a, b)));
  }
  assert.equal([...layout.stations.keys()].filter(id => id === paper.id).length, 1);
  const alignment = layout.lines.find(line => line.id === 'detection');
  const synthesis = layout.lines.find(line => line.id === 'synthesis');
  const incoming = alignment.tracks.find(track => segments({ tracks: [track] }).some(([a, b]) => onSegment(station, a, b)));
  const outgoing = synthesis.tracks.find(track => track[0].x === station.x && track[0].y === station.y);
  assert.ok(incoming && outgoing, 'the two colors meet exactly at the station without terminal stubs');
  assert.ok(incoming[0].x < station.x && incoming.at(-1).x > station.x && outgoing[1].x > station.x);
  const arrival = layout.stations.get('reactvau').platforms.find(p => p.lineId === 'detection');
  assert.equal(incoming.at(-1).x, arrival.x);
  assert.equal(incoming.at(-1).y, arrival.y);
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
        assert.ok(!crossesBox(a, b, stationBounds(station, 10)),
          `${line.id} passes through unrelated ${station.paperId}`);
      } else {
        for (const platform of station.platforms.filter(item => item.lineId !== line.id && isInterchangeStation(station))) {
          assert.ok(!crossesBox(a, b, { x: platform.x - 9, y: platform.y - 9, width: 18, height: 18 }),
            `${line.id} uses the wrong platform at ${station.paperId}`);
        }
      }
    }
  }
});

test('same-month transfers precede forks that would otherwise cross their approach', () => {
  const transfer = papers.find(paper => paper.id === 'a2seek');
  const fork = papers.find(paper => paper.id === 'vad-r1');
  assert.equal(transfer.year, fork.year);
  assert.equal(transfer.timeline.month, fork.timeline.month);
  assert.ok(layout.stations.get(transfer.id).x < layout.stations.get(fork.id).x);
  const routes = layout.lines.filter(line => ['evidence', 'reasoning', 'evaluation'].includes(line.id));
  const left = layout.stations.get('holotrace').x;
  const right = layout.stations.get('cuebench').x;
  for (let i = 0; i < routes.length; i++) for (let j = i + 1; j < routes.length; j++) {
    for (const [a, b] of segments(routes[i])) for (const [c, d] of segments(routes[j])) {
      const dx = b.x - a.x, dy = b.y - a.y, ex = d.x - c.x, ey = d.y - c.y;
      const cross = dx * ey - dy * ex;
      if (Math.abs(cross) < epsilon) continue;
      const ox = c.x - a.x, oy = c.y - a.y;
      const t = (ox * ey - oy * ex) / cross, u = (ox * dy - oy * dx) / cross;
      const x = a.x + t * dx;
      assert.ok(x <= left || x >= right || t <= epsilon || t >= 1 - epsilon ||
        u <= epsilon || u >= 1 - epsilon, `${routes[i].id} crosses ${routes[j].id} around the same-month fork`);
    }
  }
});

test('early supervision is a level through section at OVVAD and TPWNG without an empty cross-year rail', () => {
  const line = layout.lines.find(line => line.id === 'synthesis');
  assert.equal(line.tracks.length, 2);
  const tpwng = layout.stations.get('tpwng'), lavida = layout.stations.get('lavida');
  assert.ok(line.tracks[0].at(-1).x < tpwng.x + 20);
  const platform = lavida.platforms.find(p => p.lineId === 'synthesis');
  assert.ok(segments({ tracks: [line.tracks[1]] }).some(([a,b]) => onSegment(platform,a,b)));
  assert.ok(line.tracks[1].at(-1).x >= layout.stations.get('cavge').x);
  assert.ok(!segments(line).some(([a, b]) => a.x <= tpwng.x + 40 && b.x >= lavida.x - 40));
  const early = ['vadclip', 'ovvad', 'tpwng', 'tthf'].map(id => layout.stations.get(id));
  assert.equal(layout.stations.get('ovvad').y, tpwng.y, 'the supervision color section stays level');
  for (const id of ['ovvad', 'tpwng']) {
    const station = layout.stations.get(id);
    assert.ok(station.continuation && !station.fork);
    assert.ok(station.platforms.every(p => p.y === station.y));
    const center = station.label.x + station.label.width / 2;
    assert.ok(early.filter(other => other !== station).every(other =>
      Math.abs(center - station.x) < Math.abs(center - other.x)), 'the label identifies its own stop');
    const rail = segments(layout.lines.find(l => l.id === 'detection'));
    assert.ok(rail.some(([a, b]) => onSegment(station, a, b)));
  }
});

test('NWPU Campus and DoTA converge independently at one ordinary UCA station', () => {
  const line = layout.lines.find(line => line.id === 'understanding');
  const uca = layout.stations.get('uca-paper');
  const campus = layout.stations.get('nwpu-campus-paper'), dota = layout.stations.get('dota-paper');
  assert.ok(campus.y < uca.y && dota.y > uca.y, 'the two entry routes occupy opposite shelves');
  assert.equal(uca.platforms.length, 1);
  assert.ok(!isInterchangeStation(uca));
  for (const [index, root] of [campus, dota].entries()) {
    const track = line.tracks[index];
    assert.ok(segments({ tracks: [track] }).some(([a, b]) => onSegment(root, a, b)));
    assert.equal(track.at(-1).x, uca.x);
    assert.equal(track.at(-1).y, uca.y);
    assert.ok((track.at(-2).y - uca.y) * (root.y - uca.y) > 0,
      'each incoming diagonal reaches UCA from its own side');
  }
  assert.equal(line.tracks[2][0].x, uca.x, 'the through route starts exactly at UCA');
  assert.equal(line.tracks[2][0].y, uca.y);
});

test('local LAVIDA and COPRA branches depart from named paper platforms', () => {
  const line = layout.lines.find(line => line.id === 'detection');
  for (const [anchorId, leafId] of [['alert-clip', 'reactvau'], ['upr-vad', 'copra']]) {
    const anchor = layout.stations.get(anchorId).platforms.find(p => p.lineId === line.id);
    const leaf = layout.stations.get(leafId).platforms.find(p => p.lineId === line.id);
    const branch = line.tracks.find(track => track[0].x === anchor.x && track[0].y === anchor.y);
    assert.ok(branch, `${leafId} starts at ${anchorId}`);
    assert.ok(branch.slice(1).some((b, i) => onSegment(leaf, branch[i], b)));
    assert.ok(branch.at(-1).x <= leaf.x + 18, `${leafId} terminates locally`);
    assert.ok(line.tracks.filter(other => other !== branch).some(other =>
      segments({ tracks: [other] }).some(([a, b]) => onSegment(anchor, a, b))));
  }
});

test('local offshoots avoid artificial trunk peaks and keep a level LAVIDA continuation', () => {
  for (const [before, root, after] of [['piercingeye', 'alert-clip', 'td-vad'], ['scene-dependent-vad', 'upr-vad', 'spherevad']]) {
    const y = layout.stations.get(root).y;
    assert.ok((y - layout.stations.get(before).y) * (y - layout.stations.get(after).y) <= 0,
      `${root} stays on a level or monotone incoming corridor`);
  }
  const line = layout.lines.find(line => line.id === 'synthesis');
  const lavida = layout.stations.get('lavida').platforms.find(p => p.lineId === line.id);
  const next = layout.stations.get('anomalycraft');
  assert.equal(lavida.y, next.y);
  for (const [a, b] of segments(line).filter(([a, b]) => a.x >= lavida.x && b.x <= next.x)) {
    assert.equal(a.y, lavida.y);
    assert.equal(b.y, lavida.y, 'the continuation has no extra dip');
  }
});

test('the current map gives sloping connections enough room to avoid vertical drops', () => {
  for (const line of layout.lines) for (const [a, b] of segments(line)) {
    assert.ok(b.x > a.x, `${line.id} has a cramped vertical segment`);
  }
});

test('the later evaluation segment reconnects at CG-CoE without crossing Anom-π approaches', () => {
  const station = layout.stations.get('cg-coe');
  const paper = papers.find(p => p.id === station.paperId);
  assert.equal(paper.cluster, 'evaluation');
  assert.deepEqual(station.lineIds, ['evaluation', 'reasoning']);
  assert.ok(!isInterchangeStation(station), 'CG-CoE uses an ordinary branch marker');
  assert.ok(station.platforms.every(platform => platform.x === station.x && platform.y === station.y));
  assert.match(paper.secondaryMethods[0].evidence.note, /事件抽取.*匹配链/);
  const evaluation = layout.lines.find(line => line.id === 'evaluation');
  const platform = station.platforms.find(p => p.lineId === evaluation.id);
  assert.deepEqual(evaluation.tracks[1][0], { x: platform.x, y: platform.y });
  const plus = layout.stations.get('vad-r1-plus');
  assert.equal(evaluation.tracks[0].at(-1).x, plus.x, 'the early segment returns at Vad-R1-Plus');
  const left = layout.stations.get('stch').x, right = station.x + 80;
  assert.ok(layout.stations.get('anom-pi').x < station.x,
    'same-month ordering gives each shared station its own approach');
  const routes = layout.lines.filter(line => ['reasoning', 'evaluation', 'evidence'].includes(line.id));
  for (let i = 0; i < routes.length; i++) for (let j = i + 1; j < routes.length; j++) {
    for (const [a, b] of segments(routes[i])) for (const [c, d] of segments(routes[j])) {
      const dx = b.x - a.x, dy = b.y - a.y, ex = d.x - c.x, ey = d.y - c.y;
      const cross = dx * ey - dy * ex;
      if (Math.abs(cross) < epsilon) continue;
      const ox = c.x - a.x, oy = c.y - a.y;
      const t = (ox * ey - oy * ex) / cross, u = (ox * dy - oy * dx) / cross;
      const x = a.x + t * dx;
      assert.ok(x <= left || x >= right || t <= epsilon || t >= 1 - epsilon ||
        u <= epsilon || u >= 1 - epsilon, `${routes[i].id} crosses ${routes[j].id} around Anom-π`);
    }
  }
});

test('Anom-π connects active observation and structured reasoning once, clearing evaluation roots', () => {
  const paper = papers.find(p => p.id === 'anom-pi');
  const station = layout.stations.get(paper.id);
  assert.deepEqual(station.lineIds, ['evidence', 'reasoning']);
  assert.match(paper.secondaryMethods[0].evidence.note, /结构化假设/);
  assert.equal([...layout.stations.values()].filter(s => s.paperId === paper.id).length, 1);
  assert.ok(layout.stations.get('cg-coe').x > station.x, 'the same-month resource clears the connector');
  const evaluation = layout.lines.find(line => line.id === 'evaluation');
  assert.equal(evaluation.tracks.length, 3);
  assert.ok(evaluation.tracks[0].at(-1).x < station.x);
  assert.ok(evaluation.tracks[1][0].x > station.x, 'the later evaluation route starts after Anom-π');
  const cuva = layout.stations.get('cuva').platforms.find(p => p.lineId === 'evaluation');
  assert.deepEqual(evaluation.tracks[2][0], { x: cuva.x, y: cuva.y });
  for (const id of ['mmad', 'phys-ad']) {
    const earlier = layout.stations.get(id);
    assert.ok(earlier.x < layout.stations.get('vad-r1').x);
    assert.ok(segments({ tracks: [evaluation.tracks[2]] }).some(([a, b]) => onSegment(earlier, a, b)));
  }
});

test('all route joins occur at paper platforms, with no shared unnamed segments', () => {
  const paths = layout.lines.flatMap(line => line.tracks.map(track => ({ id: line.id, track })));
  const isStation = point => [...layout.stations.values()].some(station => station.platforms
    .some(platform => Math.hypot(point.x - platform.x, point.y - platform.y) < epsilon));
  for (let i = 0; i < paths.length; i++) for (let j = i + 1; j < paths.length; j++) {
    for (const [a, b] of segments({ tracks: [paths[i].track] }))
      for (const [c, d] of segments({ tracks: [paths[j].track] })) {
        const dx = b.x - a.x, dy = b.y - a.y, ex = d.x - c.x, ey = d.y - c.y;
        const cross = dx * ey - dy * ex;
        const ox = c.x - a.x, oy = c.y - a.y;
        if (Math.abs(cross) < epsilon) {
          if (Math.abs(dx * oy - dy * ox) >= epsilon) continue;
          const size = dx * dx + dy * dy;
          const p = (ox * dx + oy * dy) / size;
          const q = ((d.x - a.x) * dx + (d.y - a.y) * dy) / size;
          const lo = Math.max(0, Math.min(p, q)), hi = Math.min(1, Math.max(p, q));
          if (hi < lo - epsilon) continue;
          assert.ok(hi - lo < epsilon, `${paths[i].id}/${paths[j].id} share an unnamed segment`);
          assert.ok(isStation({ x: a.x + lo * dx, y: a.y + lo * dy }), 'collinear contact is a paper');
        } else {
          const t = (ox * ey - oy * ex) / cross, u = (ox * dy - oy * dx) / cross;
          if (t < -epsilon || t > 1 + epsilon || u < -epsilon || u > 1 + epsilon) continue;
          // Interior crossings are separate rails with SVG paper casing. A
          // junction or touching endpoint must always be a named platform.
          if (t > epsilon && t < 1 - epsilon && u > epsilon && u < 1 - epsilon) continue;
          assert.ok(isStation({ x: a.x + t * dx, y: a.y + t * dy }),
            `${paths[i].id}/${paths[j].id} join outside a paper station`);
        }
      }
  }
});

test('TD-VAD stays on detection without lifting the temporal line', () => {
  const td = layout.stations.get('td-vad');
  assert.deepEqual(td.lineIds, ['detection']);
  assert.ok(!isInterchangeStation(td));
  assert.equal(td.y, layout.stations.get('fine-vad').y, 'the detection trunk stays level');
  const temporal = layout.lines.find(line => line.id === 'understanding');
  assert.ok(!segments(temporal).some(([a, b]) => onSegment(td, a, b)));
  const panda = layout.stations.get('panda').platforms.find(p => p.lineId === 'understanding');
  const valu = layout.stations.get('valu');
  for (const [a, b] of segments(temporal).filter(([a, b]) => a.x >= panda.x && b.x <= valu.x)) {
    assert.ok(Math.min(a.y, b.y) >= Math.min(panda.y, valu.y) - 24,
      'the temporal run no longer climbs into the empty detection corridor');
  }
});

test('late synthesis and streaming detection reuse the corridor after the blue terminus', () => {
  const trunk = layout.stations.get('alert-clip');
  const synthesis = layout.stations.get('lavida');
  const events = layout.stations.get('ewad');
  const blueEnd = layout.stations.get('steervad');
  const react = layout.stations.get('reactvau');
  assert.ok(events.y <= trunk.y - 192, 'event-camera branch remains above detection');
  assert.ok(synthesis.y >= trunk.y + 288, 'synthesis occupies the interior corridor');
  assert.ok(blueEnd.x < synthesis.x, 'the synthesis lane opens after the earlier blue route ends');
  assert.ok(synthesis.y < react.platforms[0].y - 120, 'streaming detection stays below synthesis');
  assert.equal(layout.methodOrder.indexOf('synthesis'), layout.methodOrder.indexOf('detection') + 1);
  const memoryY = react.platforms.find(p => p.lineId === 'understanding').y;
  for (const id of ['s2mgraph-vad', 'peer-vad'])
    assert.equal(layout.stations.get(id).y, memoryY, 'late memory continues on one shelf');
  const detection = layout.lines.find(line => line.id === 'detection');
  const bridge = detection.tracks.find(track => track.at(-1).x === react.x);
  assert.ok(bridge && segments({ tracks: [bridge] }).some(([a, b]) => onSegment(synthesis, a, b)));
});

test('VA-GPT exits diverge before the temporal branch opens at Where and What', () => {
  const root = layout.stations.get('va-gpt');
  const fork = layout.stations.get('where-what');
  const blue = layout.lines.find(line => line.id === 'alignment');
  const purple = layout.lines.find(line => line.id === 'understanding');
  assert.ok(isInterchangeStation(root), 'VA-GPT retains its genuine method transfer');
  assert.ok(fork.x > root.x, 'the reading branch opens at the next named station');
  const x = Math.min(root.x + 96, (root.x + fork.x) / 2);
  const blueY = railHeight(blue, x), purpleY = railHeight(purple, x);
  assert.ok(blueY < root.platforms[0].y - 48, 'blue climbs away from the transfer');
  assert.ok(purpleY > root.platforms.at(-1).y + 48, 'purple descends before branching');
  assert.ok(purpleY - blueY >= 128, 'the exits open a visible gap instead of tightly tracking each other');
  const event = layout.stations.get('eventvad');
  const arm = purple.tracks.find(track => track[0].x === fork.x && track[0].y === fork.y);
  assert.ok(arm && segments({ tracks: [arm] }).some(([a, b]) => onSegment(event, a, b)));
});

test('every method route belongs to one connected network through sourced paper stations', () => {
  const reached = new Set([layout.methodOrder[0]]);
  for (let pass = 0; pass < layout.lines.length; pass++) for (const station of layout.stations.values()) {
    if (station.lineIds.some(id => reached.has(id))) station.lineIds.forEach(id => reached.add(id));
  }
  assert.deepEqual([...reached].sort(), layout.lines.map(line => line.id).sort());
  for (const [id, methods] of [['va-gpt', ['alignment', 'understanding']], ['reactvau', ['understanding', 'detection']]]) {
    const station = layout.stations.get(id);
    assert.deepEqual(station.lineIds, methods);
    assert.ok(isInterchangeStation(station));
    const paper = papers.find(p => p.id === id);
    assert.ok(paper.secondaryMethods.every(method => method.evidence.url && method.evidence.note));
  }
});

test('the early evaluation corridor clears the reasoning trunk before the later fork', () => {
  const fork = layout.stations.get('vad-r1');
  const trunk = layout.stations.get('vau-r1');
  for (const id of ['mmad', 'phys-ad']) {
    const station = layout.stations.get(id);
    assert.ok(station.x < fork.x);
    assert.ok(station.y < trunk.y, `${id} stays above the active reasoning trunk`);
  }
  assert.ok(layout.stations.get('holotrace').x < fork.x, 'the early reading route returns before the later fork');
});

test('the horizontal station key fits bottom-left whitespace with room around tracks and names', () => {
  const legend = createMapLegend(layout);
  const routeLabels = createMapRouteLabels(layout);
  assert.deepEqual(legend.fallbackLineIds, []);
  assert.ok(legend.height <= 72, "the three station types share one compact row");
  assert.ok(legend.y >= layout.plotBounds.y + layout.plotBounds.height * .75, "legend stays in the bottom quarter");
  const criteriaY = layout.stations.get('lrpo').y;
  assert.ok(legend.y >= criteriaY + 48 && legend.y + legend.height <= criteriaY + 192,
    'legend occupies the open band just below the red trunk');
  assert.ok(legend.embedded);
  assert.ok(legend.x + legend.width < layout.width * .4, 'legend stays on the left');
  assert.ok(contains(layout.plotBounds, legend));
  assert.ok(contains(layout.plotBounds, { x: legend.x + legend.width, y: legend.y + legend.height }));
  const clearance = { x: legend.x - 20, y: legend.y - 20, width: legend.width + 40, height: legend.height + 40 };
  for (const line of layout.lines) for (const [a, b] of segments(line)) {
    assert.ok(!crossesBox(a, b, clearance), `${line.id} keeps space around the legend`);
  }
  for (const station of layout.stations.values()) for (const box of [station.label, stationBounds(station)]) {
    assert.ok(box.x + box.width <= clearance.x || clearance.x + clearance.width <= box.x ||
      box.y + box.height <= clearance.y || clearance.y + clearance.height <= box.y,
      `legend clears ${station.paperId}`);
  }
  const backdrop = createResearchBackdrop(layout, [legend, ...routeLabels]);
  for (const box of backdrop.mountains) {
    assert.ok(box.x + box.width <= clearance.x || clearance.x + clearance.width <= box.x ||
      box.y + box.height <= clearance.y || clearance.y + clearance.height <= box.y, 'decoration clears the legend');
    const padded = { x: box.x - 20, y: box.y - 20, width: box.width + 40, height: box.height + 40 };
    for (const line of layout.lines) for (const [a, b] of segments(line)) {
      assert.ok(!crossesBox(a, b, padded), `decoration clears the ${line.id} route`);
    }
    for (const station of layout.stations.values()) for (const other of [station.label, stationBounds(station)]) {
      assert.ok(other.x + other.width <= padded.x || padded.x + padded.width <= other.x ||
        other.y + other.height <= padded.y || padded.y + padded.height <= other.y,
        `decoration clears ${station.paperId}`);
    }
  }
  assert.deepEqual(createMapLegend(createPublicationLayout([...papers].reverse())), legend);
});

test('every direction is named beside its own rail without covering stations, tracks or other names', () => {
  const labels = createMapRouteLabels(layout);
  assert.deepEqual(labels.map(l => l.lineId).sort(), layout.lines.map(l => l.id).sort());
  const legend = createMapLegend(layout, labels);
  const backdrop = createResearchBackdrop(layout, [legend, ...labels]);
  const disjoint = (a, b) => a.x + a.width <= b.x || b.x + b.width <= a.x ||
    a.y + a.height <= b.y || b.y + b.height <= a.y;
  for (const label of labels) {
    assert.ok(contains(layout.plotBounds, label));
    assert.ok(contains(layout.plotBounds, { x: label.x + label.width, y: label.y + label.height }));
    const own = layout.lines.find(l => l.id === label.lineId);
    assert.equal(label.text, own.label);
    assert.ok(segments(own).some(([a, b]) => a.y === b.y && b.x > label.x && a.x < label.x + label.width &&
      Math.min(Math.abs(a.y - label.y), Math.abs(a.y - label.y - label.height)) <= 42), 'name stays next to its rail');
    const ownDistance = Math.min(...segments(own).map(([a, b]) => railLabelDistance(label, a, b)));
    for (const other of layout.lines.filter(line => line !== own)) {
      assert.ok(segments(other).every(([a, b]) => railLabelDistance(label, a, b) >= ownDistance + 28),
        `${label.lineId} name clearly belongs to its own rail rather than ${other.id}`);
    }
    const padded = { x: label.x - 10, y: label.y - 10, width: label.width + 20, height: label.height + 20 };
    for (const line of layout.lines) for (const [a, b] of segments(line)) assert.ok(!crossesBox(a, b, padded));
    const occupied = [...labels.filter(other => other !== label), legend, ...backdrop.mountains,
      ...[...layout.stations.values()].flatMap(s => [s.label, stationBounds(s, 20)])];
    assert.ok(occupied.every(box => disjoint(padded, box)), `${label.lineId} clears all map content`);
  }
  assert.deepEqual(createMapRouteLabels(createPublicationLayout([...papers].reverse())), labels);
});

test('a small catalog puts the legend below the map when its left gaps are too narrow', () => {
  for (const entries of [[], [papers[0]]]) {
    const network = createPublicationLayout(entries);
    const legend = createMapLegend(network);
    assert.ok(!legend.embedded);
    assert.ok(legend.y > network.plotBounds.y + network.plotBounds.height);
    assert.ok(legend.x + legend.width < network.width);
  }
});

test('interchange routes use separate straight platforms with one shared label', () => {
  for (const station of layout.stations.values()) {
    const bounds = stationBounds(station);
    for (const other of layout.stations.values()) {
      const label = other.label;
      assert.ok(label.x + label.width <= bounds.x || bounds.x + bounds.width <= label.x ||
        label.y + label.height <= bounds.y || bounds.y + bounds.height <= label.y,
        `${other.paperId} label avoids ${station.paperId} station body`);
    }
    if (!isInterchangeStation(station)) continue;
    for (const [index, platform] of station.platforms.entries()) {
      assert.equal(platform.x, station.x, 'all platforms keep the paper date');
      if (index) assert.ok(platform.y - station.platforms[index - 1].y >= 24, 'platform markers remain separate');
      const line = layout.lines.find(item => item.id === platform.lineId);
      const isRoot = line.tracks.some(track => track[0].x === platform.x && track[0].y === platform.y);
      const isEnd = line.tracks.some(track => track.at(-1).x === platform.x && track.at(-1).y === platform.y);
      assert.ok(segments(line).some(([a, b]) => Math.abs(a.y - platform.y) < epsilon &&
        Math.abs(b.y - platform.y) < epsilon && a.x <= platform.x - (isRoot ? 0 : 10) && b.x >= platform.x + (isEnd ? 0 : 10)),
        `${station.paperId}: ${line.id} stays straight through its platform`);
    }
  }
});

test('labels stay close enough to identify their own station in dense months', () => {
  for (const station of layout.stations.values()) {
    const box = station.label;
    const x = Math.max(box.x, Math.min(station.x, box.x + box.width));
    const y = Math.max(box.y, Math.min(station.y, box.y + box.height));
    const distance = Math.min(...station.platforms.map(p => Math.hypot(p.x - x, p.y - y)),
      Math.hypot(station.x - x, station.y - y));
    assert.ok(distance <= (station.platforms.length > 1 && !station.continuation ? 68 : 52),
      `${station.paperId} label drifts away from its station`);
    assert.ok(x === station.x || y === station.y, `${station.paperId} has a diagonal callout`);
    for (const other of layout.lines) for (const track of other.tracks) {
      const parts = segments({ tracks: [track] });
      const ownsStation = station.platforms.some(p => p.lineId === other.id &&
        parts.some(([a, b]) => onSegment(p, a, b)));
      if (ownsStation) continue;
      assert.ok(parts.every(([a, b]) => railLabelDistance(box, a, b) >= distance + 12),
        `${station.paperId} name stays closer to its station than to an unrelated ${other.id} arm`);
    }
  }
});

test('dense late-year labels stay beside their stations instead of stacking far away', () => {
  for (const id of ['scene-dependent-vad', 'upr-vad', 'peer-vad', 's2mgraph-vad']) {
    const station = layout.stations.get(id), box = station.label;
    const verticalGap = Math.max(box.y - station.y, station.y - box.y - box.height, 0);
    const horizontalGap = Math.max(box.x - station.x, station.x - box.x - box.width, 0);
    assert.ok(Math.hypot(verticalGap, horizontalGap) <= 52, `${id} stays adjacent`);
  }
});

test('detection has its own color and keeps all requested foundations without deleting papers', () => {
  const line = layout.lines.find(line => line.id === 'detection');
  assert.notEqual(line.color, layout.lines.find(line => line.id === 'alignment').color);
  assert.equal(layout.lines.length, clusters.length);
  assert.equal(layout.stations.size, papers.length);
  for (const id of ['ovvad', 'anomize', 'mpgdfl', 'piercingeye', 'ewad', 'spherevad'])
    assert.ok(layout.stations.get(id).lineIds.includes('detection'), id);
  for (const id of ['anomalygpt', 'anomaly-ov', 'hawk', 'echotraffic', 'hiprobe-vad', 'headhunt-vad', 'steervad'])
    assert.deepEqual(layout.stations.get(id).lineIds, ['alignment'], id);
  const representationSequence = ['va-gpt', 'hiprobe-vad', 'headhunt-vad', 'steervad'];
  const representationRoute = clusters.find(c => c.id === 'alignment').routes.find(r => r.paperIds.includes('va-gpt'));
  const start = representationRoute.paperIds.indexOf('va-gpt');
  assert.deepEqual(representationRoute.paperIds.slice(start, start + 4), representationSequence);
  assert.deepEqual(layout.stations.get('ex-vad').lineIds, ['alignment', 'detection']);
  const copra = layout.stations.get('copra');
  assert.deepEqual(copra.lineIds, ['detection']);
  assert.ok(!isInterchangeStation(copra), 'COPRA is an ordinary detection endpoint');
  assert.ok(!segments(layout.lines.find(line => line.id === 'alignment')).some(([a, b]) => onSegment(copra, a, b)),
    'the understanding route does not connect to COPRA');
});

test('Y branches start at sourced paper stations with shared platform centers', () => {
  assert.deepEqual(layout.junctions.map(j => j.paperId).sort(), ['cg-coe', 'cuva', 'lavida', 'vad-r1']);
  for (const junction of layout.junctions) {
    const parent = layout.lines.find(l => l.id === junction.parentId);
    const branch = layout.lines.find(l => l.id === junction.branchId);
    const track = branch.tracks.find(track => track[0].x === junction.x && track[0].y === junction.y);
    assert.ok(track, 'a reviewed branch route starts at the junction');
    const trunk = segments(parent).find(([a, b]) => onSegment(junction, a, b));
    assert.ok(trunk, 'branch starts on its parent');
    assert.ok(parent.tracks.some(track => track[0].x <= junction.x && track.at(-1).x > junction.x &&
      segments({ tracks: [track] }).some(([a, b]) => onSegment(junction, a, b))),
      'the parent continues beyond the junction, which may also start the trunk');
    const [a, b] = trunk, arm = track[1];
    assert.ok(Math.abs((b.x - a.x) * (arm.y - junction.y) - (b.y - a.y) * (arm.x - junction.x)) > epsilon,
      'the new arm diverges immediately instead of retracing the parent');
    const station = layout.stations.get(junction.paperId);
    assert.equal(station.x, junction.x);
    assert.equal(station.y, junction.y);
    const cluster = clusters.find(c => c.id === junction.branchId);
    assert.ok(cluster.branchAt.paperId === station.paperId ||
      cluster.routes.some(route => route.paperIds[0] === station.paperId && route.evidence?.url),
      'each fork comes from the primary anchor or a sourced local route');
    assert.equal(station.platforms.length, 2);
    assert.ok(station.platforms.every(p => p.x === station.x && p.y === station.y));
  }
});

test('one route can fork twice and an existing branch can fork again', () => {
  const evidence = { url: 'https://example.org/fork', note: 'Fixture shared-method evidence.' };
  const methods = ['root', 'first', 'second', 'nested'].map((id, i) => ({
    id, name: id, color: ['#1466a8', '#b88a19', '#825cba', '#368e81'][i],
    ...(i ? { branchOf: id === 'nested' ? 'first' : 'root',
      branchAt: { paperId: `${id}-fork`, evidence } } : {}),
  }));
  const definitions = [
    ['root-start', 'root', 2023, 1], ['first-fork', 'root', 2023, 6, 'first'],
    ['nested-fork', 'first', 2024, 6, 'nested'], ['root-middle', 'root', 2024, 6],
    ['second-fork', 'root', 2025, 6, 'second'], ['first-end', 'first', 2025, 11],
    ['nested-end', 'nested', 2025, 11], ['root-end', 'root', 2026, 6],
    ['second-end', 'second', 2026, 11],
  ];
  const entries = definitions.map(([id, cluster, year, month, secondary], i) => ({
    ...papers[0], id, cluster, year, shortTitle: `Paper ${i + 1}`,
    timeline: { month, basis: 'conference', source: evidence },
    secondaryMethods: secondary ? [{ cluster: secondary, evidence }] : undefined,
  }));
  const network = createLayout(entries, methods);
  assert.equal(network.junctions.length, 3);
  assert.equal(network.stations.size, entries.length);
  for (const junction of network.junctions) {
    const parent = network.lines.find(l => l.id === junction.parentId);
    const branch = network.lines.find(l => l.id === junction.branchId);
    assert.deepEqual(branch.tracks[0][0], { x: junction.x, y: junction.y });
    assert.ok(segments(parent).some(([a, b]) => onSegment(junction, a, b)));
    assert.ok(parent.tracks[0][0].x < junction.x && parent.tracks[0].at(-1).x > junction.x);
    assert.ok(network.stations.get(junction.paperId).platforms.every(p => p.x === junction.x && p.y === junction.y));
  }
  assert.equal(network.junctions.filter(j => j.parentId === 'root').length, 2);
  assert.equal(network.junctions.find(j => j.branchId === 'nested').parentId, 'first');
});

test('local branches open diagonally before settling onto their own shelf', () => {
  for (const [lineId, rootId, firstId] of [
    ['detection', 'alert-clip', 'lavida'],
    ['detection', 'td-vad', 'ewad'], ['detection', 'upr-vad', 'copra'],
    ['understanding', 'where-what', 'eventvad'],
    ['reasoning', 'vad-r1-plus', 'iad-r1'], ['explanation', 'lrpo', 'probe-vad'],
    ['evaluation', 'cuva', 'mmad'], ['evaluation', 'vad-r1', 'cuebench'],
    ['evaluation', 'cg-coe', 'pistachio'],
  ]) {
    const root = layout.stations.get(rootId).platforms.find(p => p.lineId === lineId);
    const first = layout.stations.get(firstId).platforms.find(p => p.lineId === lineId);
    const track = layout.lines.find(line => line.id === lineId).tracks.find(track =>
      track[0].x === root.x && track[0].y === root.y &&
      segments({ tracks: [track] }).some(([a, b]) => onSegment(first, a, b)));
    assert.ok(track, `${rootId} has its own outgoing branch`);
    assert.ok(track[1].x > root.x && (track[1].y - root.y) * (first.y - root.y) > 0,
      `${rootId} leaves diagonally at the station`);
    const horizontal = segments({ tracks: [track] }).find(([a, b]) => a.y === b.y && a.x < first.x);
    if (horizontal) assert.ok(Math.abs(horizontal[0].y - root.y) >= Math.min(120, Math.abs(first.y - root.y)),
      `${rootId} opens enough room before running parallel`);
    for (const trunk of layout.lines.find(line => line.id === lineId).tracks.filter(other => other !== track)) {
      if (!segments({ tracks: [trunk] }).some(([a, b]) => onSegment(root, a, b))) continue;
      const x = root.x + Math.min(80, (first.x - root.x) / 2);
      const at = parts => {
        const pair = parts.find(([a, b]) => a.x <= x && b.x >= x && b.x > a.x);
        return pair && pair[0].y + (pair[1].y - pair[0].y) * (x - pair[0].x) / (pair[1].x - pair[0].x);
      };
      const branchY = at(segments({ tracks: [track] })), trunkY = at(segments({ tracks: [trunk] }));
      if (branchY !== undefined && trunkY !== undefined)
        assert.ok(Math.abs(branchY - trunkY) >= 40, `${rootId} arms separate rather than following close parallel diagonals`);
    }
  }
});

test('confirmed papers with pending citations still appear exactly once on their routes', () => {
  for (const id of ['seek-vau', 'ca-judge', 'road']) {
    const paper = papers.find(p => p.id === id);
    assert.equal(paper.citation.version, 'pending');
    assert.equal(paper.classification.basis, 'title');
    assert.equal([...layout.stations.values()].filter(s => s.paperId === id).length, 1);
  }
  assert.equal(layout.stations.get('seek-vau').lineId, 'evidence');
  const branchWithoutAnchor = papers.filter(p => p.cluster === 'synthesis' && !['ovvad', 'lavida'].includes(p.id));
  const onlyBranch = createPublicationLayout(branchWithoutAnchor);
  assert.equal(onlyBranch.junctions.length, 0);
  assert.equal(onlyBranch.stations.size, branchWithoutAnchor.length);
});

test('WACV, Findings and workshop records never enter map geometry', () => {
  const supplementary = ['WACV', 'NAACL Findings', 'Findings of ACL', 'EMNLP Findings', 'CVPR Workshops', 'ICCV Workshop', 'CVPRW', 'ECCVW']
    .map((venue, i) => ({ ...papers[0], id: `other-${i}`, venue }));
  assert.ok(supplementary.every(p => !isMapPaper(p)));
  assert.ok(['CVPR', 'ICCV', 'NeurIPS Evaluations and Datasets', 'TPAMI', 'arXiv'].every(venue => isMapPaper({ venue })));
  const network = createPublicationLayout([...catalogPapers, ...supplementary]);
  assert.equal(network.stations.size, papers.length);
  assert.deepEqual(network.lines, layout.lines);
  assert.deepEqual(network.years, layout.years);
  assert.ok(catalogPapers.some(p => p.id === 'vane-bench'), 'the source catalog retains the paper');
  assert.ok(!network.stations.has('vane-bench'));
  assert.ok(supplementary.every(p => !network.stations.has(p.id)));
});

test('editorial map selection excludes STEP and TrajVAD while retaining EWAD and the accepted NeurIPS papers', () => {
  const network = createPublicationLayout(catalogPapers);
  for (const id of ['step-vad', 'trajvad']) {
    const paper = catalogPapers.find(p => p.id === id);
    assert.ok(paper, `${id} retains its bibliographic record`);
    assert.ok(paper.mapExclusion.note);
    assert.equal(isMapPaper(paper), false);
    assert.equal(network.stations.has(id), false);
  }
  for (const id of ['ewad', 'seek-vau', 'ca-judge', 'road']) {
    assert.ok(isMapPaper(catalogPapers.find(p => p.id === id)));
    assert.ok(network.stations.has(id));
  }
});
