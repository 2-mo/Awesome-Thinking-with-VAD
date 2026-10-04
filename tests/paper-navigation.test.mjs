import test from 'node:test';
import assert from 'node:assert/strict';
import { readCatalog } from '../scripts/catalog.mjs';
import { createPublicationLayout } from '../src/components/publication-layout.ts';
import { MAP_CANVAS_WIDTH } from '../src/components/map-typography.ts';
import { adjacentPapers } from '../src/components/paper-navigation.ts';

const catalog = await readCatalog();
const layout = createPublicationLayout(catalog.papers, catalog.clusters, { width: MAP_CANVAS_WIDTH });
const neighbors = (lineId, id) => adjacentPapers(layout.lines.find(line => line.id === lineId), id);

test('a reading fork offers its actual arms rather than a chronological shortcut', () => {
  assert.deepEqual(neighbors('reasoning', 'vad-r1'), {
    previous: ['a2seek'], next: ['urf-zs-hvaa', 'vad-dpo'],
  });
  assert.deepEqual(neighbors('reasoning', 'vad-r1-plus'), {
    previous: ['targetvau', 'vad-dpo'], next: ['srvau-r1'],
  });
});

test('same-color disconnected reading segments have independent endpoints', () => {
  assert.deepEqual(neighbors('evaluation', 'vad-r1-plus'), { previous: ['finevau'], next: [] });
  assert.deepEqual(neighbors('evaluation', 'holotrace'), { previous: ['phys-ad'], next: [] });
  assert.deepEqual(neighbors('evaluation', 'cg-coe'), { previous: [], next: ['pistachio'] });
});

test('switching direction at a shared paper follows that direction only', () => {
  assert.deepEqual(neighbors('alignment', 'va-gpt'), { previous: ['ex-vad'], next: ['hiprobe-vad'] });
  assert.deepEqual(neighbors('understanding', 'va-gpt'), { previous: ['holmes-vau'], next: ['where-what'] });
  assert.deepEqual(neighbors('evaluation', 'vad-r1'), { previous: [], next: ['cuebench'] });
});

test('simplified temporal navigation follows the retained event and memory sequence', () => {
  assert.deepEqual(neighbors('understanding', 'panda'), { previous: ['monitor'], next: ['valu'] });
  assert.deepEqual(neighbors('understanding', 'uca-paper'), {
    previous: [], next: ['holmes-vau'],
  });
});

test('automatic routes follow displayed station order and all navigation edges are reversible', () => {
  const line = layout.lines.find(line => line.id === 'evidence');
  const expected = [...layout.stations.values()].filter(s => s.lineIds.includes(line.id))
    .sort((a, b) => a.x - b.x).map(s => s.paperId);
  assert.deepEqual(line.paperRoutes, [expected]);
  for (const line of layout.lines) for (const route of line.paperRoutes) for (const id of route) {
    assert.ok(layout.stations.get(id).lineIds.includes(line.id));
    for (const next of adjacentPapers(line, id).next) {
      assert.ok(adjacentPapers(line, next).previous.includes(id));
    }
  }
});

test('duplicate local route definitions do not duplicate choices and missing papers have none', () => {
  const line = { paperRoutes: [['a', 'b', 'c'], ['b', 'c']] };
  assert.deepEqual(adjacentPapers(line, 'b'), { previous: ['a'], next: ['c'] });
  assert.deepEqual(adjacentPapers(line, 'absent'), { previous: [], next: [] });
});

test('restored model mechanisms and provisional understanding papers follow their intended routes', () => {
  assert.deepEqual(neighbors('alignment', 'headhunt-vad'), { previous: ['mpgdfl'], next: ['steervad', 'piercingeye'] });
  assert.deepEqual(neighbors('evidence', 'seek-vau'), { previous: ['vto'], next: [] });
  assert.deepEqual(neighbors('explanation', 'road'), { previous: ['prime-vad'], next: [] });
  assert.deepEqual(neighbors('explanation', 'ca-judge'), { previous: ['probe-vad'], next: [] });
});


test('blue comparison branches offer their actual forks and returns', () => {
  assert.deepEqual(neighbors('alignment', 'hawk'), { previous: ['tpwng'], next: ['varcmp', 'dsrl'] });
  assert.deepEqual(neighbors('alignment', 'ex-vad'), { previous: ['pi-vad', 'lec-vad'], next: ['va-gpt'] });
  assert.deepEqual(neighbors('alignment', 'piercingeye'), { previous: ['headhunt-vad'], next: [] });
});
