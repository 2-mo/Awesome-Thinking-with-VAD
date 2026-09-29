import { readFile, writeFile } from 'node:fs/promises';
import { realpathSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { readCatalog, validateCatalog } from './catalog.mjs';

const outputPath = fileURLToPath(new URL('../catalog.md', import.meta.url));
const escape = value => value.replace(/[\\`*_{}\[\]<>|]/g, '\\$&').replace(/\s+/g, ' ').trim();
const link = (label, url) => `[${label}](<${url}>)`;

export function renderCatalog(catalog) {
  const lines = [
    '# Video Anomaly Understanding — 精选核验目录', '',
    '这是从结构化数据生成的精选核验目录，并非完整文献综述。原仓库的会议笔记作为额外资料保留，尚未全面复核，不应视作本目录的核验条目。', '',
    `数据更新时间：${catalog.updatedAt}。核验来源与日期、数据集、证据关系和阅读路线请见 [data/catalog.json](data/catalog.json) 与交互式研究地图。`, '',
    '> 自动生成：请编辑 `data/catalog.json` 后运行 `npm run generate`，不要手工修改此文件。', '',
    '## Core research / 核心研究', '',
  ];
  const paperLines = paper => {
    const refs = Object.entries(paper.links).map(([key, url]) => link(key, url)).join(' · ');
    return [`### ${escape(paper.title)}`, '', `**${paper.year} · ${escape(paper.venue)}** · ${refs}`, '', `**创新抓手：${escape(paper.mechanism)}**`, '', escape(paper.summary), '', `- 核心启示：${escape(paper.takeaway)}`, `- 局限：${escape(paper.limitation)}`, `- 核验：${paper.verifiedAt}；${paper.sources.map((source, i) => link(`来源 ${i + 1}`, source.url)).join(' · ')}`, ''];
  };
  for (const cluster of catalog.clusters) {
    const papers = catalog.papers.filter(paper => paper.scope === 'core' && paper.cluster === cluster.id);
    if (!papers.length) continue;
    lines.push(`## ${escape(cluster.name)}`, '', escape(cluster.description), '', `研究问题：${escape(cluster.question)}`, '');
    for (const paper of papers) lines.push(...paperLines(paper));
  }
  return `${lines.join('\n').trimEnd()}\n`;
}

if (process.argv[1] && realpathSync(process.argv[1]) === fileURLToPath(import.meta.url)) {
  try {
    const catalog = await readCatalog();
    const errors = validateCatalog(catalog);
    if (errors.length) throw new Error(errors.join('\n'));
    const generated = renderCatalog(catalog);
    if (process.argv.includes('--check')) {
      const existing = await readFile(outputPath, 'utf8').catch(() => null);
      if (existing !== generated) throw new Error('catalog.md is missing or stale. Run npm run generate and commit the result.');
      console.log('catalog.md is up to date.');
    } else {
      await writeFile(outputPath, generated);
      console.log('Generated catalog.md.');
    }
  } catch (error) { console.error(error.message); process.exitCode = 1; }
}
