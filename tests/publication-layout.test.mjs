import test from 'node:test';
import assert from 'node:assert/strict';
import { readCatalog } from '../scripts/catalog.mjs';
import { isMapPaper, paperMethods, publicationKind, timelineYear } from '../src/publication.ts';
import { createPublicationLayout as createLayout, isInterchangeStation, railLabelDistance, stationBounds, stationVenue } from '../src/components/publication-layout.ts';
import { MAP_CANVAS_WIDTH, MAP_FONT_SIZE, MAP_RAIL_WIDTH } from '../src/components/map-typography.ts';
import { createMapLegend } from '../src/components/map-legend.ts';
import { createMapRouteLabels } from '../src/components/map-route-labels.ts';
import { createResearchBackdrop } from '../src/components/research-regions.ts';

const { papers: catalogPapers, clusters } = await readCatalog();
const papers = catalogPapers.filter(isMapPaper);
const createPublicationLayout = entries => createLayout(entries, clusters,
  entries.length >= papers.length ? { width: MAP_CANVAS_WIDTH } : {});
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
  assert.ok(cue.label.x + cue.label.width + 8 <= fine.label.x ||
    cue.label.y + cue.label.height + 8 <= fine.label.y || fine.label.y + fine.label.height + 8 <= cue.label.y,
    'parallel-arm names can stagger vertically while retaining a visible gap');
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

test('event organization follows one continuous temporal trunk without the old detection detour', () => {
  const temporal = layout.lines.find(line => line.id === 'understanding');
  assert.equal(temporal.tracks.length, 1);
  assert.deepEqual(temporal.paperRoutes[0].slice(0, 9),
    ['uca-paper', 'holmes-vau', 'va-gpt', 'where-what', 'eventvad', 'vadtree', 'monitor', 'panda', 'valu']);
  for (const id of ['nwpu-campus-paper', 'dota-paper', 'scene-dependent-vaa'])
    assert.ok(!layout.stations.has(id));
});

test('industrial video reasoning stays on the trunk after removing the image-only arm', () => {
  const line = layout.lines.find(line => line.id === 'reasoning');
  assert.equal(line.tracks.length, 2, 'only the R1 reliability return remains');
  const trunk = line.paperRoutes[0];
  assert.deepEqual(trunk.slice(trunk.indexOf('cg-coe')), ['cg-coe', 'o-vad', 'clue-vad', 'avar']);
  for (const id of ['las-vad', 'stch', 'cg-coe', 'o-vad', 'clue-vad', 'avar'])
    assert.equal(layout.stations.get(id).y, layout.stations.get('vad-r1-plus').y);
  for (const id of ['iad-r1', 'judo']) assert.ok(!layout.stations.has(id));
  const criteria = layout.lines.find(line => line.id === 'explanation');
  assert.equal(criteria.tracks.length, 2);
  assert.equal(criteria.paperRoutes[0].at(-1), 'road');
  assert.deepEqual(criteria.paperRoutes[1], ['lrpo', 'probe-vad', 'ca-judge']);
});

test('Vad-R1 and its Plus extension retain separate dated stations', () => {
  const original = layout.stations.get('vad-r1'), extension = layout.stations.get('vad-r1-plus');
  assert.ok(original && extension);
  assert.ok(original.x < extension.x);
  assert.equal(layout.stations.size, catalogPapers.filter(isMapPaper).length);
  assert.ok(layout.stations.get('vau-r1').x < original.x, 'VAU-R1 stays before the later R1 papers');
});

test('the labels after the R1 return have breathing room without changing the level trunk', () => {
  const plus = layout.stations.get('vad-r1-plus'), next = layout.stations.get('srvau-r1');
  const later = layout.stations.get('las-vad');
  assert.ok(plus.label.y >= plus.y + 16 && next.label.y + next.label.height <= next.y - 16,
    'the return label and first continuing name occupy opposite sides of the level trunk');
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

test('year order and broad early-to-late order remain while nearby months can move independently', () => {
  for (const a of papers) for (const b of papers) {
    if (timelineYear(a) < timelineYear(b) || (timelineYear(a) === timelineYear(b) &&
      a.timeline && b.timeline && b.timeline.month - a.timeline.month >= 6)) {
      assert.ok(layout.stations.get(a.id).x < layout.stations.get(b.id).x, `${a.id} precedes ${b.id}`);
    }
  }
  for (const line of layout.lines) for (const route of line.paperRoutes) {
    for (let i = 1; i < route.length; i++)
      assert.ok(layout.stations.get(route[i - 1]).x < layout.stations.get(route[i]).x, `${line.id} reading order`);
  }
  assert.ok(papers.some(a => papers.some(b => timelineYear(a) === timelineYear(b) &&
    a.timeline?.month < b.timeline?.month && layout.stations.get(a.id).x > layout.stations.get(b.id).x)),
    'independent nearby months are no longer locked to global date columns');
});

test('year bands cover the schematic without implying a precise quarter scale', () => {
  for (const [index, year] of layout.years.entries()) {
    assert.ok(year.width > 0);
    assert.equal(year.count, papers.filter(p => timelineYear(p) === year.year).length);
    assert.equal(year.quarters, undefined);
    if (index) assert.ok(Math.abs(layout.years[index - 1].x + layout.years[index - 1].width - year.x) < epsilon);
  }
  assert.equal(layout.years[0].x, layout.plotBounds.x);
  const last = layout.years.at(-1);
  assert.ok(Math.abs(last.x + last.width - layout.plotBounds.x - layout.plotBounds.width) < epsilon);
});

test('unknown months keep their year without inventing a month or quarter', () => {
  const paper = { ...papers[0], year: 2026, timeline: undefined };
  const network = createPublicationLayout([paper]);
  const year = network.years[0], station = network.stations.get(paper.id);
  assert.equal(year.year, 2026);
  assert.ok(station.x > year.x && station.x < year.x + year.width);
  assert.equal(paper.timeline, undefined);
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
  assert.deepEqual(transfers.map(paper => paper.id).sort(), ['a2seek', 'anom-pi', 'anomalyruler', 'memovad', 'panda', 'reactvau', 'va-gpt']);
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

test('LRPO stays on criteria and the retained detection precursor stays separate from understanding', () => {
  const lrpo = layout.stations.get('lrpo');
  assert.ok(!isInterchangeStation(lrpo));
  assert.deepEqual(lrpo.lineIds, ['explanation']);
  assert.ok(lrpo.platforms.every(p => p.x === lrpo.x && p.y === lrpo.y));
  for (const id of lrpo.lineIds) {
    assert.ok(segments(layout.lines.find(line => line.id === id)).some(([a, b]) => onSegment(lrpo, a, b)));
  }
  assert.ok(!segments(layout.lines.find(line => line.id === 'reasoning')).some(([a, b]) => onSegment(lrpo, a, b)),
    'the green route bypasses LRPO');
  const precursor = layout.stations.get('vadclip'), hawk = layout.stations.get('hawk');
  assert.deepEqual(precursor.lineIds, ['alignment']);
  assert.deepEqual(hawk.lineIds, ['alignment']);
  assert.equal(precursor.y, hawk.y, 'semantic precursors share the continuous blue shelf');
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
    assert.equal(publicationKind(catalogPapers.find(paper => paper.id === id)), 'conference', id);
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
  assert.deepEqual(paperMethods(paper), ['synthesis', 'alignment']);
  assert.ok(paper.secondaryMethods[0].evidence.url.includes('2602.19248'));
  assert.equal(station.platforms.length, 2);
  assert.ok(!isInterchangeStation(station), 'a named fork uses one ordinary marker');
  assert.ok(station.platforms.every(p => p.x === station.x && p.y === station.y));
  assert.deepEqual(station.fork, { parentId: 'alignment', branchId: 'synthesis' });
  assert.ok(layout.junctions.some(junction => junction.paperId === 'lavida'));
  for (const platform of station.platforms) {
    assert.ok(segments(layout.lines.find(line => line.id === platform.lineId))
      .some(([a, b]) => onSegment(platform, a, b)));
  }
  assert.equal([...layout.stations.keys()].filter(id => id === paper.id).length, 1);
  const alignment = layout.lines.find(line => line.id === 'alignment');
  const synthesis = layout.lines.find(line => line.id === 'synthesis');
  const incoming = alignment.tracks.find(track => segments({ tracks: [track] }).some(([a, b]) => onSegment(station, a, b)));
  const outgoing = synthesis.tracks.find(track => track[0].x === station.x && track[0].y === station.y);
  assert.ok(incoming && outgoing, 'the two routes meet exactly at the station without terminal stubs');
  assert.ok(incoming[0].x < station.x && incoming.at(-1).x > station.x && outgoing[1].x > station.x);

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

test('supervision connects the early blue stations and branches locally at LAVIDA', () => {
  const line = layout.lines.find(line => line.id === 'synthesis');
  assert.deepEqual(line.paperRoutes, [['ovvad', 'tpwng'], ['lavida', 'anomalycraft', 'pa-vad', 'cavge']]);
  assert.equal(line.tracks.length, 2);
  assert.equal(line.tracks[1][0].x, layout.stations.get('lavida').x);
  assert.ok(line.tracks[1].every(point => point.x >= layout.stations.get('lavida').x));
});

test('the video-language resource UCA is the temporal entry without background dataset branches', () => {
  const line = layout.lines.find(line => line.id === 'understanding');
  const uca = layout.stations.get('uca-paper');
  assert.equal(uca.platforms.length, 1);
  assert.ok(!isInterchangeStation(uca));
  assert.equal(line.paperRoutes[0][0], 'uca-paper');
  assert.ok(segments(line).some(([a, b]) => onSegment(uca, a, b)));
});

test('every paper is reachable through actual route edges, including the early blue connector', () => {
  const edges = layout.lines.flatMap(line => line.paperRoutes.flatMap(route =>
    route.slice(1).map((id, i) => [route[i], id])));
  const reached = new Set(['vadclip']);
  for (let pass = 0; pass < layout.stations.size; pass++) for (const [a, b] of edges) {
    if (reached.has(a)) reached.add(b);
    if (reached.has(b)) reached.add(a);
  }
  assert.deepEqual([...reached].sort(), [...layout.stations.keys()].sort());
  const blue = layout.lines.find(line => line.id === 'alignment');
  assert.ok(blue.paperRoutes.some(route => route[0] === 'td-vad' && route.at(-1) === 'reactvau'));
});

test('the generation arm leaves the named fork upward and continues level', () => {
  const line = layout.lines.find(line => line.id === 'synthesis');
  const fork = layout.stations.get('lavida'), first = layout.stations.get('anomalycraft');
  assert.ok(first.y < fork.y);
  assert.equal(first.y, layout.stations.get('cavge').y);
  assert.ok(line.tracks[1][1].y < fork.y);
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
  for (const id of ['black-swan', 'phys-ad']) {
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



test('the late fork reuses the blue corridor with generation above and streaming below', () => {
  const synthesis = layout.stations.get('lavida');
  const blueEnd = layout.stations.get('steervad');
  const react = layout.stations.get('reactvau');
  assert.ok(blueEnd.x < synthesis.x);
  assert.equal(synthesis.y, blueEnd.y);
  assert.ok(layout.stations.get('anomalycraft').y < synthesis.y);
  assert.ok(react.platforms[0].y > synthesis.y);
  for (const id of ['s2mgraph-vad', 'peer-vad'])
    assert.equal(layout.stations.get(id).y, react.platforms.find(p => p.lineId === 'understanding').y);
});

test('VA-GPT exits separate directly toward representation and event understanding', () => {
  const root = layout.stations.get('va-gpt');
  const blue = layout.lines.find(line => line.id === 'alignment');
  const purple = layout.lines.find(line => line.id === 'understanding');
  assert.ok(isInterchangeStation(root));
  const x = Math.min(root.x + 96, (root.x + layout.stations.get('eventvad').x) / 2);
  assert.ok(railHeight(purple, x) - railHeight(blue, x) >= 128);
});

test('every method route belongs to one connected network through sourced paper stations', () => {
  const reached = new Set([layout.methodOrder[0]]);
  for (let pass = 0; pass < layout.lines.length; pass++) for (const station of layout.stations.values()) {
    if (station.lineIds.some(id => reached.has(id))) station.lineIds.forEach(id => reached.add(id));
  }
  assert.deepEqual([...reached].sort(), layout.lines.map(line => line.id).sort());
  for (const [id, methods] of [['va-gpt', ['alignment', 'understanding']], ['reactvau', ['understanding', 'alignment']]]) {
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
  for (const id of ['black-swan', 'phys-ad']) {
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
  const criteriaY = Math.max(...layout.lines.find(l => l.id === 'explanation').tracks[0].map(p => p.y));
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
  for (const id of ['agenticvau', 'vibes', 'peer-vad', 's2mgraph-vad']) {
    const station = layout.stations.get(id), box = station.label;
    const verticalGap = Math.max(box.y - station.y, station.y - box.y - box.height, 0);
    const horizontalGap = Math.max(box.x - station.x, station.x - box.x - box.width, 0);
    assert.ok(Math.hypot(verticalGap, horizontalGap) <= 52, `${id} stays adjacent`);
  }
});

test('the focused map keeps understanding and bridge papers while retaining excluded bibliography', () => {
  assert.equal(catalogPapers.length, 127);
  assert.equal(layout.stations.size, 87);
  assert.equal(layout.lines.length, 7, 'five trunks and two branch directions');
  assert.equal(layout.lines.find(l => l.id === 'synthesis').color, layout.lines.find(l => l.id === 'alignment').color);
  assert.equal(new Set(layout.lines.map(l => l.color)).size, 6);
  for (const id of ['cuva', 'hawk', 'a2seek', 'anom-pi', 'o-vad', 'phys-ad', 'echotraffic', 'panda', 'memovad'])
    assert.ok(layout.stations.has(id), id);
  for (const id of ['anomalygpt', 'anomaly-ov', 'mmad', 'judo', 'iad-r1', 'adseeker', 'log-sad',
    'nwpu-campus-paper', 'dota-paper', 'scene-dependent-vaa', 'cmcir', 'fedvad', 'stprompt', 'd2mil', 'upr-vad', 'tthf']) {
    const paper = catalogPapers.find(p => p.id === id);
    assert.ok(paper.citation && paper.mapExclusion.note, id);
    assert.ok(!layout.stations.has(id), id);
  }
  assert.equal(layout.width, MAP_CANVAS_WIDTH, 'the canvas does not grow to accommodate more stations');
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
    ['synthesis', 'lavida', 'anomalycraft'],
    ['reasoning', 'vad-r1', 'vad-dpo'],
    ['evaluation', 'cuva', 'black-swan'], ['evaluation', 'vad-r1', 'cuebench'],
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

test('relevant title-based entries stay visible with their provisional classification intact', () => {
  for (const id of ['seek-vau', 'ca-judge', 'road']) {
    const paper = catalogPapers.find(p => p.id === id);
    assert.equal(paper.citation.version, 'pending');
    assert.equal(paper.classification.basis, 'title');
    assert.equal(paper.mapExclusion, undefined);
    assert.ok(layout.stations.has(id));
  }
  const branchWithoutAnchor = papers.filter(p => p.cluster === 'synthesis' && p.id !== 'lavida');
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

test('editorial exclusions stay outside geometry while direct video understanding remains visible', () => {
  const network = createPublicationLayout(catalogPapers);
  for (const id of ['step-vad', 'trajvad', 'cmcir']) {
    const paper = catalogPapers.find(p => p.id === id);
    assert.ok(paper, `${id} retains its bibliographic record`);
    assert.ok(paper.mapExclusion.note);
    assert.equal(isMapPaper(paper), false);
    assert.equal(network.stations.has(id), false);
  }
  for (const id of ['o-vad', 'phys-ad', 'echotraffic', 'headhunt-vad', 'seek-vau', 'ca-judge', 'road', 'fine-vad', 'pi-vad']) {
    assert.ok(isMapPaper(catalogPapers.find(p => p.id === id)));
    assert.ok(network.stations.has(id));
  }
});


test('the fixed overview width preserves readable text, rails and the reference aspect ratio', () => {
  assert.equal(layout.width, 7400);
  assert.ok(layout.width / layout.height >= 2.75 && layout.width / layout.height <= 3);
  assert.ok(MAP_FONT_SIZE.primary * 2048 / layout.width >= 11, 'paper names remain readable at overview width');
  assert.ok(MAP_FONT_SIZE.secondary * 2048 / layout.width >= 7.5);
  assert.ok(MAP_RAIL_WIDTH * 2048 / layout.width >= 2);
  assert.throws(() => createLayout(papers, clusters, { width: 1000 }), /fixed 1000-unit width; split/);
  const blue = layout.lines.find(l => l.id === 'alignment');
  const trunk = blue.paperRoutes.find(route => route.includes('hiprobe-vad'));
  for (const id of trunk.slice(trunk.indexOf('hiprobe-vad')).filter(id => id !== 'td-vad'))
    assert.equal(layout.stations.get(id).y, layout.stations.get('headhunt-vad').y, 'ordinary representation stops share a stable trunk');
  assert.ok(layout.stations.get('anomalycraft').y < layout.stations.get('lavida').y);
});

test('all 57 lower understanding papers remain while blue comparison work uses vertical branches', () => {
  const retained = ["a2seek","adversa","adversa-sd","agenticvau","anom-pi","anomalyruler","avar","black-swan","ca-judge","cg-coe","clue-vad","crcl","cuebench","cuva","ecva-anomshield","eval","eventvad","finevau","holmes-vau","holotrace","lagovad","las-vad","lavad","lrpo","memovad","monitor","o-vad","panda","peer-vad","phys-ad","pistachio","prime-vad","probe-vad","promptvad","reactvau","road","s2mgraph-vad","seek-vau","slowfastvad","srvau-r1","stch","tar-bench","targetvau","tau-bench","uca-paper","urf-zs-hvaa","vad-dpo","vad-r1","vad-r1-plus","vadtree","vagu-gts","valu","vau-r1","vera","vibes","vto","where-what"];
  assert.deepEqual(papers.filter(p => !['alignment', 'synthesis'].includes(p.cluster)).map(p => p.id).sort(), retained);
  for (const id of retained) assert.ok(layout.stations.has(id), id);
  for (const id of ['dsrl', 'anomize', 'lec-vad'])
    assert.ok(layout.stations.get(id).y < layout.stations.get('hawk').y);
  assert.ok(layout.stations.get('piercingeye').y < layout.stations.get('headhunt-vad').y);
});
