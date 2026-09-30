import test from 'node:test';
import assert from 'node:assert/strict';
import { copyFile, mkdir, mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { spawnSync } from 'node:child_process';
import { readCatalog, validateCatalog } from '../scripts/catalog.mjs';
import { renderCatalog, renderLiterature, renderVenueIndex, renderComparison, renderBenchmarks, renderGuides, renderBibEntry, renderBibliography, renderCitations } from '../scripts/generate-catalog.mjs';

function fixture() {
  const source = { url: 'https://example.org/paper', note: 'Primary paper describes the method and evaluation.' };
  const paper = {
    citation: { key: 'coreA2024', type: 'inproceedings', version: 'published', title: 'A core paper', authors: ['Doe, Jane'], year: 2024, publication: 'Example Conference', url: source.url, sources: [source], verifiedAt: '2026-09-29' },
    id: 'core-a', shortTitle: 'Core A', title: 'A core paper', year: 2024, venue: 'Example Conference',
    scope: 'core', cluster: 'reasoning', tasks: ['异常推理'], summary: 'Grounded anomaly interpretation.',
    mechanism: '证据约束解释', takeaway: 'Evaluate grounded explanations.', limitation: 'Limited evaluation scale.',
    links: { paper: source.url }, sources: [source], verifiedAt: '2026-09-29', datasetIds: ['dataset-a'],
  };
  return {
    version: 1, updatedAt: '2026-09-29',
    clusters: [{ id: 'reasoning', name: 'Reasoning', description: 'Anomaly reasoning.', question: 'What happened?', color: '#667799', position: { x: 10, y: 20 } }],
    papers: [paper, { ...structuredClone(paper), id: 'core-b', title: 'Another core paper', citation: { ...structuredClone(paper.citation), key: 'coreB2024' } }],
    datasets: [{ id: 'dataset-a', name: 'Dataset A', year: 2024, venue: 'Example Conference', thumbnail: { src: '/datasets/example.png', alt: 'Example dataset figure', sourceUrl: 'https://example.org/figure.png', credit: 'Dataset authors' }, description: 'A research benchmark.', tasks: ['异常推理'], modalities: ['video'], annotations: ['events'], protocol: 'Use the official split.', links: { website: 'https://example.org/dataset' }, sources: [source] }],
    relations: [{ id: 'relation-a', source: 'core-a', target: 'dataset-a', type: 'uses', evidence: source }],
    guides: [{ id: 'guide-a', title: 'Start here', description: 'Read the core method.', steps: [{ paperId: 'core-a', note: 'Understand the evaluation.' }] }],
  };
}

test('valid structured fixture passes', () => assert.deepEqual(validateCatalog(fixture()), []));
test('the checked-in catalog passes semantic validation', async () => assert.deepEqual(validateCatalog(await readCatalog()), []));

const corruptions = [
  ['malformed dataset thumbnail', data => { data.datasets[0].thumbnail = null; }, /thumbnail.*local artwork/],
  ['remote thumbnail path', data => { data.datasets[0].thumbnail.src = 'https://example.org/image.png'; }, /thumbnail.src.*local/],
  ['thumbnail path traversal', data => { data.datasets[0].thumbnail.src = '/datasets/../image.png'; }, /thumbnail.src.*traversal/],
  ['encoded thumbnail traversal', data => { data.datasets[0].thumbnail.src = '/datasets/%2e%2e/image.png'; }, /thumbnail.src.*local/],
  ['thumbnail outside datasets', data => { data.datasets[0].thumbnail.src = '/assets/image.png'; }, /thumbnail.src.*local/],
  ['missing image credit', data => { data.datasets[0].thumbnail.credit = ''; }, /thumbnail.credit.*non-empty/],
  ['missing image alt', data => { delete data.datasets[0].thumbnail.alt; }, /thumbnail.alt.*non-empty/],
  ['invalid image provenance URL', data => { data.datasets[0].thumbnail.sourceUrl = 'file:///image.png'; }, /thumbnail.sourceUrl.*absolute HTTP/],
  ['missing dataset publication year', data => { delete data.datasets[0].year; }, /datasets.*year.*publication year/],
  ['missing dataset venue', data => { delete data.datasets[0].venue; }, /datasets.*venue.*non-empty/],
  ['malformed secondary methods', data => { data.papers[0].secondaryMethods = {}; }, /secondaryMethods.*array/],
  ['unknown secondary method', data => { data.papers[0].secondaryMethods = [{ cluster: 'missing', evidence: data.papers[0].sources[0] }]; }, /secondaryMethods.*unknown cluster/],
  ['duplicate method membership', data => { data.papers[0].secondaryMethods = [{ cluster: 'reasoning', evidence: data.papers[0].sources[0] }]; }, /duplicate method membership/],
  ['unsourced method membership', data => { data.clusters.push({ ...data.clusters[0], id: 'memory' }); data.papers[0].secondaryMethods = [{ cluster: 'memory' }]; }, /secondaryMethods.*evidence/],
  ['non-core scope', data => { data.papers[0].scope = 'external'; }, /scope.*must be core/],
  ['unsupported relationship type', data => { data.relations[0].type = 'similar'; }, /unknown relation type/],
  ['duplicate entity IDs', data => { data.papers[1].id = data.papers[0].id; }, /duplicate id/],
  ['dangling relationship', data => { data.relations[0].target = 'missing'; }, /unknown relation endpoint/],
  ['empty paper URL', data => { data.papers[0].links.paper = ''; }, /absolute HTTP/],
  ['empty optional URL', data => { data.papers[0].links.code = ''; }, /absolute HTTP/],
  ['non-web source URL', data => { data.relations[0].evidence.url = 'javascript:alert(1)'; }, /absolute HTTP/],
  ['dataset presented as a paper extension', data => { data.relations[0].type = 'extends'; }, /extends must connect two core/],
  ['invalid guide paper reference', data => { data.guides[0].steps[0].paperId = 'missing'; }, /unknown paper/],
  ['invalid dataset reference', data => { data.papers[0].datasetIds = ['missing']; }, /unknown dataset/],
  ['missing core verification sources', data => { data.papers[0].sources = []; }, /verification source/],
  ['missing evidence note', data => { data.relations[0].evidence.note = ''; }, /non-empty string/],
  ['invalid year', data => { data.papers[0].year = 3000; }, /publication year/],
  ['invalid hidden timeline month', data => { data.papers[0].timeline = { month: 13, basis: 'conference', source: data.papers[0].sources[0] }; }, /timeline.month.*1 to 12/],
  ['unsourced hidden timeline month', data => { data.papers[0].timeline = { month: 6, basis: 'conference' }; }, /timeline.source.*URL and evidence/],
  ['invalid calendar date', data => { data.updatedAt = '2026-02-30'; }, /valid YYYY-MM-DD/],
  ['missing required summary', data => { delete data.papers[0].summary; }, /summary/],
  ['missing innovation mechanism', data => { delete data.papers[0].mechanism; }, /mechanism.*non-empty string/],
  ['overlong innovation mechanism', data => { data.papers[0].mechanism = '机制'.repeat(13); }, /mechanism.*24 characters/],
];
for (const [name, mutate, expected] of corruptions) test(`rejects ${name}`, () => {
  const data = fixture();
  mutate(data);
  assert.match(validateCatalog(data).join('\n'), expected);
});

test('malformed collections produce errors instead of throwing', () => {
  const data = fixture();
  data.papers = [null, 42];
  data.guides = {};
  assert.ok(validateCatalog(data).length > 0);
  assert.ok(validateCatalog(null).length > 0);
});

test('generated catalog includes grouped core research and provenance context', () => {
  const markdown = renderCatalog(fixture());
  assert.ok(markdown.indexOf('## Reasoning') < markdown.indexOf('### A core paper'));
  assert.match(markdown, /### Another core paper/);
  assert.match(markdown, /尚未全面复核/);
  assert.match(markdown, /创新：证据约束解释/);
  assert.match(markdown, /### Dataset A/);
  assert.match(markdown, /2024 · Example Conference/);
  assert.match(markdown, /https:\/\/example.org\/figure.png/);
  assert.match(markdown, /Dataset authors/);
  assert.doesNotMatch(markdown, /!\[/);
  assert.equal(markdown, renderCatalog(fixture()));
});

test('generator CLI rejects missing or stale output without writing and rejects invalid data before generation', async t => {
  const dir = await mkdtemp(join(tmpdir(), 'vau-catalog-test-'));
  t.after(() => rm(dir, { recursive: true, force: true }));
  await mkdir(join(dir, 'scripts'));
  await mkdir(join(dir, 'data'));
  for (const name of ['catalog.mjs', 'generate-catalog.mjs']) {
    await copyFile(new URL(`../scripts/${name}`, import.meta.url), join(dir, 'scripts', name));
  }
  const dataPath = join(dir, 'data', 'catalog.json');
  const outputPath = join(dir, 'literature', 'catalog.md');
  await writeFile(dataPath, JSON.stringify(fixture()));
  const run = (...args) => spawnSync(process.execPath, [join(dir, 'scripts', 'generate-catalog.mjs'), ...args], { encoding: 'utf8' });

  assert.equal(run('--check').status, 1);
  await assert.rejects(readFile(outputPath), { code: 'ENOENT' });

  assert.equal(run().status, 0);
  assert.equal(run('--check').status, 0);
  const generated = await readFile(outputPath, 'utf8');
  assert.match(generated, /A core paper/);

  const literaturePath = join(dir, 'llm4vad.md');
  const literature = await readFile(literaturePath, 'utf8');
  await writeFile(literaturePath, `${literature}\nStale literature index\n`);
  assert.equal(run('--check').status, 1);
  assert.match(await readFile(literaturePath, 'utf8'), /Stale literature index/);
  await writeFile(literaturePath, literature);
  const venuePath = join(dir, 'literature', 'venues.md');
  await rm(venuePath);
  assert.equal(run('--check').status, 1);
  await assert.rejects(readFile(venuePath), { code: 'ENOENT' });
  assert.equal(run().status, 0);

  const stale = `${generated}\nHand-edited content\n`;
  await writeFile(outputPath, stale);
  assert.equal(run('--check').status, 1);
  assert.equal(await readFile(outputPath, 'utf8'), stale);

  const broken = fixture();
  broken.papers[0].links.paper = '';
  await writeFile(dataPath, JSON.stringify(broken));
  const rejected = run();
  assert.equal(rejected.status, 1);
  assert.match(rejected.stderr, /absolute HTTP/);
  assert.equal(await readFile(outputPath, 'utf8'), stale);
});

test('generated literature and venue indexes share the catalog and preserve publication status', () => {
  const data = fixture();
  data.papers[0].venue = 'NeurIPS Datasets and Benchmarks';
  data.papers[1].venue = 'arXiv';
  data.papers[1].year = 2025;
  data.papers[1].citation = { ...data.papers[1].citation, type: 'misc', version: 'preprint', year: 2025, arxivId: '2501.01234', publication: undefined };
  const literature = renderLiterature(data);
  assert.match(literature, /2 篇论文/);
  assert.ok(literature.indexOf('## 2025') < literature.indexOf('## 2024'));
  assert.match(literature, /### arXiv · 预印本/);
  assert.match(literature, /Datasets and Benchmarks/);
  assert.match(literature, /A core paper/);
  assert.match(literature, /Another core paper/);
  assert.doesNotMatch(literature, /\]\(\)/);
  assert.match(renderVenueIndex(data), /\| NeurIPS \| 1 \| \[2024\]\(\.\.\/llm4vad.md#year-2024-neurips\)/);
});

test('additional method memberships preserve their classification evidence in the catalog', () => {
  const data = fixture();
  data.clusters.push({ ...data.clusters[0], id: 'memory', name: 'Memory' });
  data.papers[0].secondaryMethods = [{ cluster: 'memory', evidence: { url: 'https://example.org/memory', note: 'Semantic memory is a core mechanism.' } }];
  assert.deepEqual(validateCatalog(data), []);
  assert.match(renderCatalog(data), /兼属方法.*Memory/);
  assert.match(renderCatalog(data), /https:\/\/example.org\/memory/);
  assert.match(renderCatalog(data), /Semantic memory is a core mechanism/);
});

// These guard the new source contract: missing facts must not become negative claims,
// and registering a benchmark must not depend on having publication artwork.
test('text-only datasets remain valid and render with their protocol and sources', () => {
  const data = fixture();
  delete data.datasets[0].thumbnail;
  assert.deepEqual(validateCatalog(data), []);
  assert.match(renderCatalog(data), /Use the official split/);
  assert.match(renderBenchmarks(data), /dataset-dataset-a/);
  assert.match(renderBenchmarks(data), /catalog.md#paper-core-a/);
  assert.doesNotMatch(renderCatalog(data), /图片：/);
});

test('comparison distinguishes unverified fields from evidenced claims', () => {
  const data = fixture();
  const paper = data.papers[0];
  paper.comparison = { training: { values: ['冻结主模型；训练评分器'], evidence: paper.sources[0] } };
  assert.deepEqual(validateCatalog(data), []);
  assert.match(renderComparison(data), /待核验/);
  assert.match(renderComparison(data), /冻结主模型；训练评分器/);
  delete paper.comparison.training.evidence;
  assert.match(validateCatalog(data).join('\n'), /comparison.training.evidence/);
});

test('task synonyms and method labels cannot re-enter the task filter', () => {
  const data = fixture();
  data.papers[0].tasks = ['基准评估', '强化学习'];
  assert.match(validateCatalog(data).join('\n'), /unknown task 基准评估/);
  assert.match(validateCatalog(data).join('\n'), /unknown task 强化学习/);
});

test('reading guides render linked steps and reading questions without a website', () => {
  const markdown = renderGuides(fixture());
  assert.match(markdown, /guide-guide-a/);
  assert.match(markdown, /catalog.md#paper-core-a/);
  assert.match(markdown, /Understand the evaluation/);
  assert.doesNotMatch(renderCatalog(fixture()), /- 局限：/);
});


test('BibTeX escapes source text, protects titles and preserves full author order', () => {
  const data = fixture();
  data.papers[0].citation.title = 'VAGU & CLIP_2: 50%';
  data.papers[0].citation.authors = ['Pereira, João', 'Doe, Jane'];
  data.papers[0].citation.pages = '12-20';
  const bib = renderBibEntry(data.papers[0]);
  assert.ok(bib.includes('title = {{VAGU \\& CLIP\\_2: 50\\%}}'));
  assert.ok(bib.includes('Pereira, Jo{\\~{a}}o and Doe, Jane'));
  assert.match(bib, /pages = \{12--20\}/);
  assert.equal((renderBibliography(data).match(/@inproceedings\{/g) ?? []).length, 2);
});

test('citation validation rejects unsourced exports and mixed publication versions', () => {
  for (const [change, error] of [
    [c => { c.sources = []; }, /verification source/],
    [c => { c.authors = []; }, /authors.*must not be empty/],
    [c => { c.doi = 'https:\/\/doi.org/10.1234/test'; }, /DOI identifier/],
    [c => { c.doi = '10.48550/arXiv.2501.01234'; }, /published citations cannot/],
    [c => { c.year = 2023; }, /publication status and year/],
    [c => { c.doi = {}; }, /DOI identifier/],
  ]) {
    const data = fixture(); change(data.papers[0].citation);
    assert.match(validateCatalog(data).join('\n'), error);
  }
  const duplicate = fixture(); duplicate.papers[1].citation.key = duplicate.papers[0].citation.key;
  assert.match(validateCatalog(duplicate).join('\n'), /duplicate citation key/);
});

test('preprint fallback exports its own year and identifiers with an explicit reader note', () => {
  const data = fixture();
  data.papers[0].citation = { ...data.papers[0].citation, type: 'misc', version: 'preprint', year: 2023, publication: undefined, arxivId: '2301.01234', doi: '10.48550/arXiv.2301.01234' };
  assert.deepEqual(validateCatalog(data), []);
  const bib = renderBibEntry(data.papers[0]);
  assert.match(bib, /year = \{2023\}/);
  assert.match(bib, /archivePrefix = \{arXiv\}/);
  assert.doesNotMatch(bib, /booktitle/);
  assert.match(renderCitations(data), /正式书目信息待补/);
  data.papers[0].citation.doi = '10.1234/published';
  assert.match(validateCatalog(data).join('\n'), /same preprint/);
});
