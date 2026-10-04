import test from 'node:test';
import assert from 'node:assert/strict';
import { createResearchBackdrop } from '../src/components/research-regions.ts';

const network = (width, levels) => ({
  width, height: Math.max(0, ...levels) + 120,
  plotBounds: { x: 0, y: 0, width, height: Math.max(0, ...levels) + 120 },
  years: [], stations: new Map(), methodOrder: [], junctions: [],
  lines: levels.map((y, index) => ({ id: String(index), label: '', color: '',
    labelPosition: { x: 0, y }, tracks: [[{ x: 0, y }, { x: width, y }]] })),
});

test('mountain count follows broad spaces between rails', () => {
  assert.equal(createResearchBackdrop(network(2400, [100, 980])).mountains.length, 1);
  const split = network(2400, [100, 540, 980]);
  const mountains = createResearchBackdrop(split).mountains;
  assert.equal(mountains.length, 2);
  for (const [index, box] of mountains.entries()) {
    const top = [100, 540][index], bottom = [540, 980][index];
    assert.ok(box.y > top && box.y + box.height < bottom);
  }
  assert.deepEqual(createResearchBackdrop({ ...split, lines: [...split.lines].reverse() }).mountains, mountains);
});

test('empty maps, outer margins and narrow rail gaps receive no mountains', () => {
  for (const levels of [[], [200], [100, 220, 340, 460, 580]])
    assert.deepEqual(createResearchBackdrop(network(2400, levels)).mountains, []);
});

test('reserved content divides or removes otherwise open pockets', () => {
  const map = network(2400, [100, 980]);
  const divider = { x: 1100, y: 0, width: 200, height: 1100 };
  const mountains = createResearchBackdrop(map, [divider]).mountains;
  assert.equal(mountains.length, 2);
  assert.ok(mountains[0].x + mountains[0].width < divider.x);
  assert.ok(mountains[1].x > divider.x + divider.width);
  assert.deepEqual(createResearchBackdrop(map, [map.plotBounds]).mountains, []);
});

test('a larger pocket receives a larger ridge without distorting its aspect ratio', () => {
  const small = createResearchBackdrop(network(2400, [100, 440])).mountains[0];
  const large = createResearchBackdrop(network(2400, [200, 800])).mountains[0];
  assert.ok(small && large);
  assert.ok(large.width > small.width && large.height > small.height);
  for (const box of [small, large]) assert.ok(Math.abs(box.width / box.height - 172 / 52) < .001);
});
