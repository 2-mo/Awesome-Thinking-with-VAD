import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { realpathSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname } from 'node:path';
import { readCatalog, validateCatalog } from './catalog.mjs';

const outputPath = name => fileURLToPath(new URL(`../${name}`, import.meta.url));
const escape = value => value.replace(/[\\`*_{}\[\]<>|]/g, '\\$&').replace(/\s+/g, ' ').trim();
const link = (label, url) => `[${label}](<${url}>)`;
const displayVenue = venue => venue === 'NeurIPS Datasets and Benchmarks' ? 'NeurIPS' : venue;
const venueId = venue => venue.toLowerCase().replace(/[^a-z0-9]+/g, '-');
const venues = catalog => [...new Set(catalog.papers.map(paper => displayVenue(paper.venue)))].sort((a, b) => (a === 'arXiv') - (b === 'arXiv') || a.localeCompare(b));
const years = catalog => [...new Set(catalog.papers.map(paper => paper.year))].sort((a, b) => b - a);

/** A compact year/venue reading index, generated from the same verified papers. */
export function renderLiterature(catalog) {
  const lines = [
    '# 视频异常理解 · 论文年表', '',
    '[研究地图](https://2-mo.github.io/Awesome-Thinking-with-VAD/) · [按创新思路阅读](catalog.md) · [按会议查找](venues/README.md)', '',
    `更新：${catalog.updatedAt} · ${catalog.papers.length} 篇论文 · ${catalog.clusters.length} 个方法方向。`, '',
    '聚焦视频异常解释、推理、时序定位与理解评估，以及直接支撑这些目标的语义表征方法。会议与年份采用已核验的正式发表信息；未确认录用的论文保留 arXiv。', '',
    '> 自动生成：编辑 `data/catalog.json` 后运行 `npm run generate`。完整摘要、阅读关注与核验来源见 [catalog.md](catalog.md)。', '',
    years(catalog).map(year => `[${year}](#year-${year})`).join(' · '), '',
  ];
  for (const year of years(catalog)) {
    lines.push(`<a id="year-${year}"></a>`, '', `## ${year}`, '');
    for (const venue of venues(catalog)) {
      const papers = catalog.papers.filter(paper => paper.year === year && displayVenue(paper.venue) === venue)
        .sort((a, b) => a.shortTitle.localeCompare(b.shortTitle));
      if (!papers.length) continue;
      lines.push(`<a id="year-${year}-${venueId(venue)}"></a>`, '', `### ${escape(venue)}${venue === 'arXiv' ? ' · 预印本' : ''}`, '',
        '| 论文 | 创新抓手 | 资源 |', '| --- | --- | --- |');
      for (const paper of papers) {
        const refs = Object.entries(paper.links).filter(([key]) => key !== 'paper')
          .map(([key, url]) => link(key === 'code' ? '代码' : '项目', url));
        const source = paper.sources.find(item => !Object.values(paper.links).includes(item.url));
        if (source) refs.push(link('核验', source.url));
        const track = paper.venue === 'NeurIPS Datasets and Benchmarks' ? ' · Datasets and Benchmarks' : '';
        lines.push(`| **${escape(paper.shortTitle)}**${track}<br>${link(escape(paper.title), paper.links.paper)} | ${escape(paper.mechanism)} | ${refs.join(' · ') || '—'} |`);
      }
      lines.push('');
    }
  }
  return `${lines.join('\n').trimEnd()}\n`;
}

export function renderVenueIndex(catalog) {
  const legacy = { AAAI: 'aaai', CVPR: 'cvpr', ICCV: 'iccv', ECCV: 'eccv', NeurIPS: 'neurips', ICML: 'icml', ICLR: 'iclr', 'ACM MM': 'acmmm', IJCAI: 'ijcai' };
  const lines = [
    '# 视频异常理解 · 发表索引', '',
    `更新：${catalog.updatedAt}。从 [结构化目录](../data/catalog.json) 自动生成；年份链接进入当前核验年表。`, '',
    '| 发表场所 | 篇数 | 年份 | 历史笔记 |', '| --- | ---: | --- | --- |',
  ];
  for (const venue of venues(catalog)) {
    const entries = catalog.papers.filter(paper => displayVenue(paper.venue) === venue);
    const links = years(catalog).filter(year => entries.some(paper => paper.year === year))
      .map(year => `[${year}](../llm4vad.md#year-${year}-${venueId(venue)})`);
    lines.push(`| ${escape(venue)}${venue === 'arXiv' ? '（预印本）' : ''} | ${entries.length} | ${links.join(' · ')} | ${Object.hasOwn(legacy, venue) ? `[旧笔记](${legacy[venue]}.md)` : '—'} |`);
  }
  lines.push('', 'NeurIPS 两个分轨合并索引，条目仍保留准确分轨。历史笔记保留原收集范围，尚未全面复核，不等同于当前核心目录。', '',
    '> 自动生成：请编辑 `data/catalog.json` 后运行 `npm run generate`。');
  return `${lines.join('\n')}\n`;
}

export function renderCatalog(catalog) {
  const lines = [
    '# Video Anomaly Understanding — 精选核验目录', '',
    '这是从结构化数据生成的精选核验目录，并非完整文献综述。原仓库的会议笔记作为额外资料保留，尚未全面复核，不应视作本目录的核验条目。', '',
    `数据更新时间：${catalog.updatedAt} · ${catalog.papers.length} 篇论文。另见 [按年份阅读](llm4vad.md)、[发表索引](venues/README.md)及 [data/catalog.json](data/catalog.json) 中的核验来源、数据集、证据关系和阅读路线。`, '',
    '> 自动生成：请编辑 `data/catalog.json` 后运行 `npm run generate`，不要手工修改此文件。', '',
    '## Core research / 核心研究', '',
  ];
  const paperLines = paper => {
    const refs = Object.entries(paper.links).map(([key, url]) => link(key, url)).join(' · ');
    return [`### ${escape(paper.title)}`, '', `**${paper.year} · ${escape(paper.venue)}** · ${refs}`, '', `**创新抓手：${escape(paper.mechanism)}**`, '', escape(paper.summary), '', ...(paper.secondaryMethods ?? []).flatMap(method => [
      `- 兼属方法：${escape(catalog.clusters.find(cluster => cluster.id === method.cluster)?.name ?? method.cluster)}；${link('归类依据', method.evidence.url)} — ${escape(method.evidence.note)}`,
    ]), `- 核心启示：${escape(paper.takeaway)}`, `- 局限：${escape(paper.limitation)}`, `- 核验：${paper.verifiedAt}；${paper.sources.map((source, i) => link(`来源 ${i + 1}`, source.url)).join(' · ')}`, ''];
  };
  for (const cluster of catalog.clusters) {
    const papers = catalog.papers.filter(paper => paper.scope === 'core' && paper.cluster === cluster.id);
    if (!papers.length) continue;
    lines.push(`## ${escape(cluster.name)}`, '', escape(cluster.description), '', `研究问题：${escape(cluster.question)}`, '');
    for (const paper of papers) lines.push(...paperLines(paper));
  }
  lines.push('## Datasets / 数据资源', '', '图片为作者项目或论文原图的本地副本；此目录只链接出处，不嵌入大图。', '');
  for (const dataset of catalog.datasets) {
    lines.push(`### ${escape(dataset.name)}`, '', `**${dataset.year} · ${escape(dataset.venue)}** · ${link('资源', dataset.links.website)}`, '', escape(dataset.description), '', `- 评估协议：${escape(dataset.protocol)}`, `- 图片：${link(escape(dataset.thumbnail.alt), dataset.thumbnail.sourceUrl)}；署名：${escape(dataset.thumbnail.credit)}`, `- 核验：${dataset.sources.map((source, i) => link(`来源 ${i + 1}`, source.url)).join(' · ')}`, '');
  }
  return `${lines.join('\n').trimEnd()}\n`;
}

if (process.argv[1] && realpathSync(process.argv[1]) === fileURLToPath(import.meta.url)) {
  try {
    const catalog = await readCatalog();
    const errors = validateCatalog(catalog);
    if (errors.length) throw new Error(errors.join('\n'));
    const outputs = [
      ['catalog.md', renderCatalog(catalog)],
      ['llm4vad.md', renderLiterature(catalog)],
      ['venues/README.md', renderVenueIndex(catalog)],
    ];
    if (process.argv.includes('--check')) {
      const stale = [];
      for (const [name, content] of outputs) {
        const existing = await readFile(outputPath(name), 'utf8').catch(() => null);
        if (existing !== content) stale.push(name);
      }
      if (stale.length) throw new Error(`${stale.join(', ')} missing or stale. Run npm run generate and commit the result.`);
      console.log('Generated research indexes are up to date.');
    } else {
      for (const [name, content] of outputs) {
        await mkdir(dirname(outputPath(name)), { recursive: true });
        await writeFile(outputPath(name), content);
      }
      console.log(`Generated ${outputs.map(([name]) => name).join(', ')}.`);
    }
  } catch (error) { console.error(error.message); process.exitCode = 1; }
}
