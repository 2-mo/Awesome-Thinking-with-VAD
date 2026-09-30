import { readFile } from 'node:fs/promises';
import { realpathSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

export const catalogPath = fileURLToPath(new URL('../data/catalog.json', import.meta.url));
export const readCatalog = async () => JSON.parse(await readFile(catalogPath, 'utf8'));
export const taskVocabulary = ['异常检测', '异常定位', '空间定位', '异常解释', '异常推理', '异常预判', '视频问答', '基准评测'];
export const comparisonFields = { outputs: '输出／评测对象', training: '训练与适配', inference: '运行设置', futureFrames: '未来帧访问', evaluation: '验证方式' };

/** Validate editorial structure and references; this intentionally does not make network requests. */
export function validateCatalog(catalog) {
  const errors = [];
  const fail = (path, message) => errors.push(`${path}: ${message}`);
  const object = value => value !== null && typeof value === 'object' && !Array.isArray(value);
  const text = (value, path) => {
    if (typeof value !== 'string' || !value.trim()) fail(path, 'must be a non-empty string');
  };
  const url = (value, path) => {
    try {
      const parsed = new URL(value);
      if (typeof value !== 'string' || value.trim() !== value || !['https:', 'http:'].includes(parsed.protocol) || !parsed.hostname) throw new Error();
    } catch { fail(path, 'must be an absolute HTTP(S) URL'); }
  };
  const date = (value, path) => {
    if (typeof value !== 'string' || !/^\d{4}-\d{2}-\d{2}$/.test(value) || !Number.isFinite(Date.parse(value)) || new Date(value).toISOString().slice(0, 10) !== value) fail(path, 'must be a valid YYYY-MM-DD date');
  };
  const strings = (value, path, nonempty = true) => {
    if (!Array.isArray(value)) return fail(path, 'must be an array');
    if (nonempty && !value.length) fail(path, 'must not be empty');
    value.forEach((item, i) => text(item, `${path}[${i}]`));
  };
  const source = (value, path) => {
    if (!object(value)) return fail(path, 'must include a URL and evidence note');
    url(value.url, `${path}.url`);
    text(value.note, `${path}.note`);
  };
  const sources = (value, path) => {
    if (!Array.isArray(value) || !value.length) return fail(path, 'must contain at least one verification source');
    value.forEach((item, i) => source(item, `${path}[${i}]`));
  };
  const links = (value, path, required, allowed) => {
    if (!object(value)) return fail(path, 'must be a links object');
    url(value[required], `${path}.${required}`);
    for (const key of allowed) if (key !== required && key in value) url(value[key], `${path}.${key}`);
  };
  if (!object(catalog)) return ['catalog: must be an object'];
  if (!Number.isInteger(catalog.version) || catalog.version < 1) fail('version', 'must be a positive integer');
  date(catalog.updatedAt, 'updatedAt');
  const collections = {};
  const ids = new Map();
  for (const name of ['clusters', 'papers', 'datasets', 'relations', 'guides']) {
    const list = catalog[name];
    collections[name] = Array.isArray(list) ? list.filter(object) : [];
    if (!Array.isArray(list)) { fail(name, 'must be an array'); continue; }
    if (!list.length) fail(name, 'must not be empty');
    list.forEach((item, i) => {
      const path = `${name}[${i}]`;
      if (!object(item)) return fail(path, 'must be an object');
      text(item.id, `${path}.id`);
      if (typeof item.id === 'string' && item.id.trim()) {
        if (ids.has(item.id)) fail(`${path}.id`, `duplicate id ${item.id} (also ${ids.get(item.id)})`);
        else ids.set(item.id, path);
      }
    });
  }
  const { clusters, papers, datasets, relations, guides } = collections;
  const clusterIds = new Set(clusters.map(item => item.id));
  const paperById = new Map(papers.map(item => [item.id, item]));
  const datasetIds = new Set(datasets.map(item => item.id));
  const entityIds = new Set([...paperById.keys(), ...datasetIds]);
  const citationKeys = new Set();
  clusters.forEach((cluster, i) => {
    const path = `clusters[${i}]`;
    for (const key of ['name', 'description', 'question', 'color']) text(cluster[key], `${path}.${key}`);
    for (const key of ['x', 'y']) if (!Number.isFinite(cluster.position?.[key])) fail(`${path}.position.${key}`, 'must be a finite number');
  });
  papers.forEach((paper, i) => {
    const path = `papers[${i}]`;
    for (const key of ['shortTitle', 'title', 'venue', 'mechanism', 'summary', 'takeaway', 'limitation']) text(paper[key], `${path}.${key}`);
    if (typeof paper.mechanism === 'string' && [...paper.mechanism.trim()].length > 24) fail(`${path}.mechanism`, 'keep the innovation mechanism within 24 characters for compact cards');
    if (!Number.isInteger(paper.year) || paper.year < 1900 || paper.year > new Date().getUTCFullYear() + 1) fail(`${path}.year`, 'must be a plausible publication year');
    if (paper.timeline !== undefined) {
      if (!object(paper.timeline)) fail(`${path}.timeline`, 'must include month, basis and source');
      else {
        if (!Number.isInteger(paper.timeline.month) || paper.timeline.month < 1 || paper.timeline.month > 12) fail(`${path}.timeline.month`, 'must be an integer from 1 to 12');
        if (!['conference', 'preprint'].includes(paper.timeline.basis)) fail(`${path}.timeline.basis`, 'must be conference or preprint');
        source(paper.timeline.source, `${path}.timeline.source`);
      }
    }
    if (paper.scope !== 'core') fail(`${path}.scope`, 'must be core');
    if (!clusterIds.has(paper.cluster)) fail(`${path}.cluster`, `unknown cluster ${paper.cluster}`);
    if (paper.secondaryMethods !== undefined) {
      if (!Array.isArray(paper.secondaryMethods)) fail(`${path}.secondaryMethods`, 'must be an array');
      else {
        const seen = new Set([paper.cluster]);
        paper.secondaryMethods.forEach((method, index) => {
          const at = `${path}.secondaryMethods[${index}]`;
          if (!object(method)) return fail(at, 'must include cluster and evidence');
          if (!clusterIds.has(method.cluster)) fail(`${at}.cluster`, `unknown cluster ${method.cluster}`);
          if (seen.has(method.cluster)) fail(`${at}.cluster`, 'duplicate method membership');
          seen.add(method.cluster);
          source(method.evidence, `${at}.evidence`);
        });
      }
    }
    strings(paper.tasks, `${path}.tasks`);
    const c = paper.citation;
    if (!object(c)) fail(`${path}.citation`, 'must contain verified citation metadata');
    else {
      const at = `${path}.citation`;
      for (const key of ['key', 'title']) text(c[key], `${at}.${key}`);
      if (typeof c.key !== 'string' || !/^[a-zA-Z][a-zA-Z0-9:_-]*$/.test(c.key)) fail(`${at}.key`, 'must be a safe BibTeX key');
      if (citationKeys.has(c.key)) fail(`${at}.key`, 'duplicate citation key');
      citationKeys.add(c.key);
      strings(c.authors, `${at}.authors`);
      if (!['article', 'inproceedings', 'misc'].includes(c.type)) fail(`${at}.type`, 'unsupported BibTeX type');
      if (!['published', 'preprint'].includes(c.version)) fail(`${at}.version`, 'must be published or preprint');
      if (!Number.isInteger(c.year) || c.year < 1900 || c.year > new Date().getUTCFullYear() + 1) fail(`${at}.year`, 'must be a plausible citation year');
      if (c.version === 'published') {
        text(c.publication, `${at}.publication`);
        if (c.type === 'misc' || c.arxivId || (typeof c.doi === 'string' && c.doi.toLowerCase().startsWith('10.48550/arxiv.'))) fail(at, 'published citations cannot use preprint identifiers or type');
        if (paper.venue === 'arXiv' || c.year !== paper.year) fail(at, 'published citation must match catalog publication status and year');
      }
      if (c.version === 'preprint') {
        if (c.type !== 'misc' || c.publication) fail(at, 'preprints must use misc without a publication venue');
        if (!/^\d{4}\.\d{4,5}(v\d+)?$/.test(c.arxivId ?? '')) fail(`${at}.arxivId`, 'must identify the cited arXiv preprint');
        if (typeof c.doi === 'string' && (typeof c.arxivId !== 'string' || c.doi.toLowerCase() !== `10.48550/arxiv.${c.arxivId.replace(/v\d+$/, '')}`.toLowerCase())) fail(`${at}.doi`, 'must identify the same preprint, not a published version');
      }
      if (c.doi !== undefined && (typeof c.doi !== 'string' || !/^10\.\d{4,9}\/\S+$/.test(c.doi))) fail(`${at}.doi`, 'must be a DOI identifier, not a URL');
      for (const key of ['volume', 'number', 'pages']) if (c[key] !== undefined) text(c[key], `${at}.${key}`);
      url(c.url, `${at}.url`);
      sources(c.sources, `${at}.sources`);
      date(c.verifiedAt, `${at}.verifiedAt`);
    }
    if (Array.isArray(paper.tasks)) paper.tasks.forEach(task => {
      if (!taskVocabulary.includes(task)) fail(`${path}.tasks`, `unknown task ${task}; use the controlled vocabulary and move method/context labels to tags`);
    });
    if (paper.tags !== undefined) strings(paper.tags, `${path}.tags`, false);
    if (paper.comparison !== undefined) {
      if (!object(paper.comparison)) fail(`${path}.comparison`, 'must be an object');
      else for (const [key, fact] of Object.entries(paper.comparison)) {
        const at = `${path}.comparison.${key}`;
        if (!Object.hasOwn(comparisonFields, key)) fail(at, 'unknown comparison field');
        if (!object(fact)) { fail(at, 'must include values and evidence'); continue; }
        strings(fact.values, `${at}.values`);
        source(fact.evidence, `${at}.evidence`);
      }
    }
    links(paper.links, `${path}.links`, 'paper', ['paper', 'code', 'project']);
    sources(paper.sources, `${path}.sources`);
    date(paper.verifiedAt, `${path}.verifiedAt`);
    strings(paper.datasetIds, `${path}.datasetIds`, false);
    if (Array.isArray(paper.datasetIds)) paper.datasetIds.forEach(id => {
      if (!datasetIds.has(id)) fail(`${path}.datasetIds`, `unknown dataset ${id}`);
    });
  });
  datasets.forEach((dataset, i) => {
    const path = `datasets[${i}]`;
    for (const key of ['name', 'description', 'protocol']) text(dataset[key], `${path}.${key}`);
    if (!Number.isInteger(dataset.year) || dataset.year < 1900 || dataset.year > new Date().getUTCFullYear() + 1) fail(`${path}.year`, 'must be a plausible publication year');
    text(dataset.venue, `${path}.venue`);
    if (dataset.thumbnail !== undefined && !object(dataset.thumbnail)) fail(`${path}.thumbnail`, 'must include local artwork and provenance');
    else if (dataset.thumbnail !== undefined) {
      const thumbnail = dataset.thumbnail;
      if (typeof thumbnail.src !== 'string' || !/^\/datasets\/[A-Za-z0-9][A-Za-z0-9_./-]*$/.test(thumbnail.src) || thumbnail.src.includes('..') || thumbnail.src.endsWith('/') || thumbnail.src.includes('//')) fail(`${path}.thumbnail.src`, 'must be a local /datasets/ path without traversal');
      text(thumbnail.alt, `${path}.thumbnail.alt`);
      text(thumbnail.credit, `${path}.thumbnail.credit`);
      url(thumbnail.sourceUrl, `${path}.thumbnail.sourceUrl`);
    }
    for (const key of ['tasks', 'modalities', 'annotations']) strings(dataset[key], `${path}.${key}`);
    links(dataset.links, `${path}.links`, 'website', ['website', 'paper']);
    sources(dataset.sources, `${path}.sources`);
  });
  relations.forEach((relation, i) => {
    const path = `relations[${i}]`;
    for (const key of ['source', 'target']) if (!entityIds.has(relation[key])) fail(`${path}.${key}`, `unknown relation endpoint ${relation[key]}`);
    if (relation.source === relation.target) fail(path, 'relation endpoints must differ');
    if (!['uses', 'introduces', 'extends'].includes(relation.type)) fail(`${path}.type`, 'unknown relation type');
    if (relation.type === 'extends' && [relation.source, relation.target].some(id => paperById.get(id)?.scope !== 'core')) fail(path, 'extends must connect two core papers');
    source(relation.evidence, `${path}.evidence`);
  });
  guides.forEach((guide, i) => {
    const path = `guides[${i}]`;
    for (const key of ['title', 'description']) text(guide[key], `${path}.${key}`);
    if (!Array.isArray(guide.steps) || !guide.steps.length) return fail(`${path}.steps`, 'must contain at least one reading step');
    guide.steps.forEach((step, j) => {
      if (!object(step)) return fail(`${path}.steps[${j}]`, 'must be an object');
      if (!paperById.has(step.paperId)) fail(`${path}.steps[${j}].paperId`, `unknown paper ${step.paperId}`);
      text(step.note, `${path}.steps[${j}].note`);
    });
  });
  return errors;
}

if (process.argv[1] && realpathSync(process.argv[1]) === fileURLToPath(import.meta.url)) {
  try {
    const catalog = await readCatalog();
    const errors = validateCatalog(catalog);
    if (errors.length) { console.error(errors.join('\n')); process.exitCode = 1; }
    else console.log(`Catalog valid: ${catalog.papers.length} papers, ${catalog.datasets.length} datasets, ${catalog.relations.length} relations.`);
  } catch (error) { console.error(`Cannot validate catalog: ${error.message}`); process.exitCode = 1; }
}
