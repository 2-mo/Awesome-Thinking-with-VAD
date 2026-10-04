import test from 'node:test';
import assert from 'node:assert/strict';
import { copyFile, mkdir, mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { spawnSync } from 'node:child_process';
import { pathToFileURL } from 'node:url';
import { readCatalog, validateCatalog, validatePaperFigureFiles } from '../scripts/catalog.mjs';
import { renderCatalog, renderLiterature, renderFigureSources, renderVenueIndex, renderComparison, renderBenchmarks, renderGuides, renderBibEntry, renderBibliography, renderCitations } from '../scripts/generate-catalog.mjs';

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
    papers: [paper, { ...structuredClone(paper), id: 'core-b', title: 'Another core paper', citation: { ...structuredClone(paper.citation), key: 'coreB2024', title: 'Another core paper' } }],
    datasets: [{ id: 'dataset-a', name: 'Dataset A', year: 2024, venue: 'Example Conference', thumbnail: { src: '/datasets/example.png', alt: 'Example dataset figure', sourceUrl: 'https://example.org/figure.png', credit: 'Dataset authors' }, description: 'A research benchmark.', tasks: ['异常推理'], modalities: ['video'], annotations: ['events'], protocol: 'Use the official split.', links: { website: 'https://example.org/dataset' }, sources: [source] }],
    relations: [{ id: 'relation-a', source: 'core-a', target: 'dataset-a', type: 'uses', evidence: source }],
    guides: [{ id: 'guide-a', title: 'Start here', description: 'Read the core method.', steps: [{ paperId: 'core-a', note: 'Understand the evaluation.' }] }],
  };
}

test('valid structured fixture passes', () => assert.deepEqual(validateCatalog(fixture()), []));
test('the checked-in catalog passes semantic validation', async () => assert.deepEqual(validateCatalog(await readCatalog()), []));

const sampleFigure = () => ({ src: 'assets/papers/core-a.png', alt: 'Original framework', caption: 'Figure 2: framework.', sourceUrl: 'https://example.org/figure.png', sourcePageUrl: 'https://example.org/paper', credit: 'Doe et al.', verifiedAt: '2026-10-03' });

test('paper figures require local paths, provenance and explicit pending status', () => {
  const data = fixture();
  data.papers[0].figure = sampleFigure();
  assert.deepEqual(validateCatalog(data), []);
  for (const src of ['https://example.org/figure.png', 'assets/papers/../image.png', 'assets/papers/%2e%2e.png', 'assets/papers/x.svg']) {
    data.papers[0].figure.src = src;
    assert.match(validateCatalog(data).join('\n'), /figure.src.*local/);
  }
  data.papers[0].figure = sampleFigure();
  delete data.papers[0].figure.sourceUrl;
  assert.match(validateCatalog(data).join('\n'), /figure.sourceUrl.*HTTP/);
  data.papers[0].figure = sampleFigure();
  data.papers[0].figurePending = { note: 'Publisher requests fail; author copy not yet located.', sources: data.papers[0].sources, verifiedAt: '2026-10-03' };
  assert.match(validateCatalog(data).join('\n'), /cannot accompany an available figure/);
  delete data.papers[0].figure;
  assert.deepEqual(validateCatalog(data), []);
});

test('paper cards retain original badges, summaries, image sources and stable anchors', () => {
  const data = fixture();
  data.papers[0].figure = sampleFigure();
  data.papers[0].links.code = 'https://github.com/example/core';
  const markdown = renderLiterature(data);
  assert.match(markdown, /id="paper-core-a"/);
  assert.match(markdown, /#### A core paper/);
  assert.match(markdown, /img.shields.io\/github\/stars\/example\/core/);
  assert.match(markdown, /> Grounded anomaly interpretation/);
  assert.match(markdown, /!\[Original framework\]\(assets\/papers\/core-a.png\)/);
  assert.match(markdown, /assets\/papers\/README.md#figure-core-a/);
  assert.match(markdown, /Doe et al/);
  assert.match(markdown, /已配原论文图片 1 \/ 2/);
  assert.match(markdown, /配图待补/);
  assert.match(renderFigureSources(data), /core-a.png/);
  assert.match(renderFigureSources(data), /id="figure-core-a"/);
  assert.match(renderFigureSources(data), /https:\/\/example.org\/figure.png/);
  assert.match(renderFigureSources(data), /https:\/\/example.org\/paper/);
  assert.match(renderFigureSources(data), /待补原图/);
});

test('figure files reject missing assets and HTML downloads disguised as images', async t => {
  const root = await mkdtemp(join(tmpdir(), 'paper-figure-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  const data = fixture();
  data.papers[0].figure = sampleFigure();
  const base = pathToFileURL(`${root}/`);
  assert.match((await validatePaperFigureFiles(data, base)).join('\n'), /missing or unreadable/);
  await mkdir(join(root, 'assets/papers'), { recursive: true });
  await writeFile(join(root, data.papers[0].figure.src), '<html>Access denied</html>');
  assert.match((await validatePaperFigureFiles(data, base)).join('\n'), /bytes do not match/);
  assert.deepEqual(await validatePaperFigureFiles(await readCatalog()), []);
});

test('journal ordering months retain their source and reject unknown date bases', () => {
  const data = fixture();
  data.papers[0].timeline = { month: 8, basis: 'journal', source: data.papers[0].sources[0] };
  assert.deepEqual(validateCatalog(data), []);
  data.papers[0].timeline.basis = 'guessed';
  assert.match(validateCatalog(data).join('\n'), /timeline.basis.*conference, journal or preprint/);
});

test('editorial exclusions preserve reading entries and cannot remain in explicit map routes', () => {
  const data = fixture();
  data.papers[0].mapExclusion = { note: 'Editorial selection for the main map.' };
  data.clusters[0].routes = [{ paperIds: ['core-b'], evidence: data.papers[0].sources[0] }];
  assert.deepEqual(validateCatalog(data), []);
  assert.match(renderLiterature(data), /A core paper/);
  assert.match(renderBibliography(data), /coreA2024/);
  data.clusters[0].routes[0].paperIds.unshift('core-a');
  assert.match(validateCatalog(data).join('\n'), /map-eligible papers/);
});

test('local reading routes require sourced, complete, chronological paper-node definitions', () => {
  const valid = fixture();
  valid.clusters[0].routes = [{ paperIds: ['core-a', 'core-b'], evidence: valid.papers[0].sources[0] }];
  assert.deepEqual(validateCatalog(valid), []);
  const cases = [
    [data => { data.clusters[0].routes = {}; }, /routes.*nonempty array/],
    [data => { data.clusters[0].routes[0].paperIds = []; }, /paperIds.*empty/],
    [data => { data.clusters[0].routes[0].paperIds = ['missing']; }, /map-eligible papers/],
    [data => { data.clusters[0].routes[0].paperIds = ['core-a']; }, /missing route paper core-b/],
    [data => { data.clusters[0].routes[0].paperIds.push('core-a'); }, /repeat a paper/],
    [data => { delete data.clusters[0].routes[0].evidence; }, /routes.*evidence/],
    [data => { data.papers[0].timeline = { month: 12, basis: 'conference', source: data.papers[0].sources[0] }; data.papers[1].timeline = { ...data.papers[0].timeline, month: 1 }; }, /publication time/],
    [data => { data.clusters.push({ ...data.clusters[0], id: 'other', routes: undefined }); data.papers[1].cluster = 'other'; }, /belong to the route direction/],
  ];
  for (const [mutate, expected] of cases) {
    const data = structuredClone(valid);
    mutate(data);
    assert.match(validateCatalog(data).join('\n'), expected);
  }
});

const corruptions = [
  ['malformed map exclusion', data => { data.papers[0].mapExclusion = true; }, /mapExclusion.*editorial note/],
  ['empty map exclusion note', data => { data.papers[0].mapExclusion = { note: '' }; }, /mapExclusion.note.*non-empty/],
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
test('dataset cards support optional artwork and retain protocol and source access', () => {
  const data = fixture();
  delete data.datasets[0].thumbnail;
  assert.deepEqual(validateCatalog(data), []);
  assert.match(renderCatalog(data), /Use the official split/);
  assert.match(renderBenchmarks(data), /dataset-dataset-a/);
  assert.match(renderBenchmarks(data), /catalog.md#paper-core-a/);
  assert.doesNotMatch(renderCatalog(data), /图片：/);
  data.datasets[0].thumbnail = { src: 'assets/papers/core-a.png', alt: 'Dataset annotation example', sourceUrl: 'https://example.org/figure.png', credit: 'Dataset authors' };
  assert.deepEqual(validateCatalog(data), []);
  assert.match(renderBenchmarks(data), /!\[Dataset annotation example\]\(\.\.\/assets\/papers\/core-a.png\)/);
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

test('accepted conference papers export a status note without inventing proceedings or DOI', () => {
  const data = fixture();
  data.papers[0].citation = { ...data.papers[0].citation, type: 'misc', version: 'accepted', publication: 'ACM Multimedia 2026', doi: undefined };
  assert.deepEqual(validateCatalog(data), []);
  const bib = renderBibEntry(data.papers[0]);
  assert.match(bib, /note = \{Accepted to ACM Multimedia 2026\}/);
  assert.doesNotMatch(bib, /booktitle|doi =/);
  assert.match(renderCitations(data), /已录用记录/);
  data.papers[0].citation.doi = '10.1145/1234567';
  assert.match(validateCatalog(data).join('\n'), /accepted citations must/);
});

test('incomplete citations keep accepted papers in indexes without fabricating BibTeX', () => {
  const data = fixture(), paper = data.papers[0];
  paper.classification = { basis: 'title', evidence: paper.sources[0] };
  paper.citation = { version: 'pending', title: paper.title, year: paper.year,
    url: paper.links.paper, sources: paper.sources, verifiedAt: '2026-09-29', note: '作者待核验。' };
  assert.deepEqual(validateCatalog(data), []);
  assert.match(renderLiterature(data), /A core paper/);
  assert.match(renderCatalog(data), /按题名暂定/);
  assert.match(renderCitations(data), /cite-core-a/);
  assert.match(renderCitations(data), /2 篇论文，1 条完整引用/);
  assert.match(renderCitations(data), /已录用，书目待补/);
  assert.equal(renderBibEntry(paper), '');
  assert.doesNotMatch(renderBibliography(data), /A core paper|undefined/);
  assert.match(renderBibliography(data), /coreB2024/);
  paper.citation.authors = ['Unknown'];
  assert.match(validateCatalog(data).join('\n'), /pending citations cannot/);
  delete paper.classification.evidence;
  assert.match(validateCatalog(data).join('\n'), /classification.evidence/);
});

test('branches reference method lines without forming cycles', () => {
  const data = fixture();
  data.papers[0].secondaryMethods = [{ cluster: 'branch', evidence: data.papers[0].sources[0] }];
  data.clusters.push({ ...data.clusters[0], id: 'branch', branchOf: 'reasoning',
    branchAt: { paperId: 'core-a', evidence: data.papers[0].sources[0] } });
  assert.deepEqual(validateCatalog(data), []);
  data.clusters[0].branchOf = 'branch';
  assert.match(validateCatalog(data).join('\n'), /branch cycle/);
  delete data.clusters[0].branchOf;
  data.clusters[1].branchOf = 'missing';
  assert.match(validateCatalog(data).join('\n'), /another method line/);
});

test('a branch can fork again at a later sourced paper', () => {
  const data = fixture();
  const evidence = data.papers[0].sources[0];
  data.papers[0].timeline = { month: 5, basis: 'conference', source: evidence };
  data.papers[1].timeline = { month: 11, basis: 'conference', source: evidence };
  data.papers[0].secondaryMethods = [{ cluster: 'branch', evidence }];
  data.papers[1].cluster = 'branch';
  data.papers[1].secondaryMethods = [{ cluster: 'subbranch', evidence }];
  data.clusters.push(
    { ...data.clusters[0], id: 'branch', branchOf: 'reasoning', branchAt: { paperId: 'core-a', evidence } },
    { ...data.clusters[0], id: 'subbranch', branchOf: 'branch', branchAt: { paperId: 'core-b', evidence } },
  );
  assert.deepEqual(validateCatalog(data), []);
  data.clusters[1].branchOf = 'subbranch';
  assert.match(validateCatalog(data).join('\n'), /branch cycle/);
});

test('a Y branch requires a sourced shared paper before the branch papers', () => {
  const valid = fixture();
  valid.papers[1].cluster = 'branch';
  valid.papers[0].secondaryMethods = [{ cluster: 'branch', evidence: valid.papers[0].sources[0] }];
  valid.clusters.push({ ...valid.clusters[0], id: 'branch', branchOf: 'reasoning',
    branchAt: { paperId: 'core-a', evidence: valid.papers[0].sources[0] } });
  for (const [mutate, expected] of [
    [d => { delete d.clusters[1].branchAt; }, /real fork paper/],
    [d => { d.clusters[1].branchAt.paperId = 'missing'; }, /real fork paper/],
    [d => { delete d.clusters[1].branchAt.evidence; }, /branchAt.evidence/],
    [d => { delete d.papers[0].secondaryMethods; }, /both method directions/],
    [d => { d.papers[0].venue = 'NAACL Findings'; }, /eligible for the map/],
    [d => { d.papers[1].cluster = 'branch'; d.papers[1].year = 2023; }, /precede its branch papers/],
    [d => { delete d.clusters[1].branchOf; }, /requires a parent/],
  ]) {
    const data = structuredClone(valid); mutate(data);
    assert.match(validateCatalog(data).join('\n'), expected);
  }
  assert.match(renderCatalog(valid), /分叉节点：\[Core A\]\(<#paper-core-a>\)/);
});

test('earlier supplementary papers do not constrain a map fork date', () => {
  const data = fixture();
  const evidence = data.papers[0].sources[0];
  data.papers[0].secondaryMethods = [{ cluster: 'branch', evidence }];
  data.clusters.push({ ...data.clusters[0], id: 'branch', branchOf: 'reasoning',
    branchAt: { paperId: 'core-a', evidence } });
  const earlier = data.papers[1];
  earlier.cluster = 'branch';
  earlier.year = earlier.citation.year = 2023;
  for (const venue of ['NAACL Findings', 'CVPR Workshop', 'ECCVW']) {
    earlier.venue = venue;
    assert.deepEqual(validateCatalog(data), [], venue);
  }
  earlier.venue = 'CVPR';
  assert.match(validateCatalog(data).join('\n'), /precede its branch papers/);
});

test('an earlier reading segment needs its own shared anchor and must end before the later fork', () => {
  const data = fixture(), evidence = data.papers[0].sources[0];
  data.papers[0].secondaryMethods = [{ cluster: 'branch', evidence }];
  data.papers[1].cluster = 'branch';
  const earlier = structuredClone(data.papers[0]);
  earlier.id = 'early-anchor'; earlier.year = 2023;
  earlier.citation.key = 'early2023'; earlier.citation.year = 2023;
  const leaf = structuredClone(earlier);
  leaf.id = 'early-leaf'; leaf.cluster = 'branch'; leaf.citation.key = 'leaf2023';
  delete leaf.secondaryMethods;
  data.papers.push(earlier, leaf);
  data.clusters.push({ ...data.clusters[0], id: 'branch', branchOf: 'reasoning',
    branchAt: { paperId: 'core-a', evidence }, routes: [
      { paperIds: ['core-a', 'core-b'], evidence },
      { paperIds: ['early-anchor', 'early-leaf'], evidence },
    ] });
  assert.deepEqual(validateCatalog(data), []);
  const unanchored = structuredClone(data);
  unanchored.papers.find(p => p.id === 'early-anchor').cluster = 'branch';
  delete unanchored.papers.find(p => p.id === 'early-anchor').secondaryMethods;
  assert.match(validateCatalog(unanchored).join('\n'), /precede its branch papers/);
  const spanning = structuredClone(data);
  spanning.clusters[1].routes[1].paperIds.push('core-b');
  assert.match(validateCatalog(spanning).join('\n'), /precede its branch papers/);
});

test('dataset cards do not imply downloads when availability is unknown', () => {
  const data = fixture();
  assert.deepEqual(validateCatalog(data), []);
  const markdown = renderBenchmarks(data);
  assert.ok(markdown.includes(`Data-${encodeURIComponent('项目入口')}`));
  assert.ok(!markdown.includes(`Data-${encodeURIComponent('下载')}`));
  assert.match(markdown, /id="understanding-data"/);
  assert.doesNotMatch(markdown, /id="detection-data"|id="retrieval-data"/);
});

test('derived datasets link to base records across groups without duplicating details', () => {
  const data = fixture();
  const source = data.datasets[0].sources[0];
  const base = data.datasets[0];
  base.tasks = ['异常检测'];
  base.composition = { kind: 'original', note: 'Collected source videos.', evidence: source };
  base.availability = { status: 'available', note: 'Official files released.', evidence: source, verifiedAt: '2026-10-03' };
  data.datasets.push({ ...structuredClone(base), id: 'dataset-derived', name: 'Derived A',
    tasks: ['异常推理'], usageNote: 'Only annotations released.',
    composition: { kind: 'annotation', note: 'Adds reasoning annotations.', evidence: source, baseDatasetIds: [base.id] },
    availability: { status: 'partial', note: 'Only annotations released.', evidence: source, verifiedAt: '2026-10-02' } });
  assert.deepEqual(validateCatalog(data), []);
  const markdown = renderBenchmarks(data);
  assert.match(markdown, /id="detection-data"/);
  assert.match(markdown, /id="understanding-data"/);
  assert.match(markdown, /基础数据：\[Dataset A\]\(<#dataset-dataset-a>\)/);
  assert.match(markdown, /Only annotations released/);
  assert.ok(markdown.includes(`Data-${encodeURIComponent('部分开放')}`));
  assert.match(markdown, /\[1\]\(<https:\/\/example.org/);
  for (const id of ['dataset-a', 'dataset-derived']) assert.equal(markdown.split(`id="dataset-${id}"`).length - 1, 1);
  for (const kind of ['resplit', 'mixed']) {
    data.datasets[1].composition.kind = kind;
    data.datasets[1].availability.status = 'pending';
    assert.deepEqual(validateCatalog(data), []);
    assert.ok(renderBenchmarks(data).includes(`Data-${encodeURIComponent('待发布')}`));
  }
});

for (const [name, mutate, expected] of [
  ['null composition', d => { d.composition = null; }, /composition:.*kind/],
  ['unknown kind', d => { d.composition.kind = 'new'; }, /composition.kind/],
  ['empty composition note', d => { d.composition.note = ''; }, /composition.note/],
  ['missing composition evidence', d => { delete d.composition.evidence; }, /composition.evidence/],
  ['unknown base dataset', d => { d.composition.baseDatasetIds = ['missing']; }, /baseDatasetIds: unknown dataset/],
  ['self base dataset', d => { d.composition.baseDatasetIds = [d.id]; }, /baseDatasetIds: must not reference itself/],
  ['non-array base datasets', d => { d.composition.baseDatasetIds = 'dataset-a'; }, /baseDatasetIds: must be an array/],
  ['null availability', d => { d.availability = null; }, /availability:.*status/],
  ['unknown availability', d => { d.availability.status = 'open'; }, /availability.status/],
  ['empty availability note', d => { d.availability.note = ''; }, /availability.note/],
  ['invalid availability source', d => { d.availability.evidence = { url: 'file:///data', note: 'Source' }; }, /availability.evidence.url/],
  ['invalid availability date', d => { d.availability.verifiedAt = '2026-02-30'; }, /availability.verifiedAt/],
]) test(`rejects dataset ${name}`, () => {
  const data = fixture();
  const d = data.datasets[0];
  d.composition = { kind: 'mixed', note: 'Combined sources.', evidence: d.sources[0] };
  d.availability = { status: 'unverified', note: 'Access not established.', evidence: d.sources[0], verifiedAt: '2026-10-03' };
  mutate(d);
  assert.match(validateCatalog(data).join('\n'), expected);
});
