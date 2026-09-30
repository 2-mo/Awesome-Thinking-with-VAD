import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { realpathSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname } from 'node:path';
import { readCatalog, validateCatalog, comparisonFields } from './catalog.mjs';

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
    '[按创新思路阅读](literature/catalog.md) · [方法比较](literature/comparison.md) · [数据集与评测](literature/benchmarks.md) · [阅读路线](literature/reading-guide.md) · [按会议查找](literature/venues.md) · [研究地图](https://2-mo.github.io/Awesome-Thinking-with-VAD/)', '',
    `更新：${catalog.updatedAt} · ${catalog.papers.length} 篇论文 · ${catalog.clusters.length} 个方法方向。`, '',
    '聚焦视频异常解释、推理、时序定位与理解评估，以及直接支撑这些目标的语义表征方法。会议与年份采用已核验的正式发表信息；未确认录用的论文保留 arXiv。', '',
    '完整摘要与阅读关注见 [研究目录](literature/catalog.md)，作者、DOI 与 BibTeX 见 [引用导出](literature/citations.md)。', '',
    years(catalog).map(year => `[${year}](#year-${year})`).join(' · '), '',
  ];
  for (const year of years(catalog)) {
    lines.push(`<a id="year-${year}"></a>`, '', `## ${year}`, '');
    for (const venue of venues(catalog)) {
      const papers = catalog.papers.filter(paper => paper.year === year && displayVenue(paper.venue) === venue)
        .sort((a, b) => a.shortTitle.localeCompare(b.shortTitle));
      if (!papers.length) continue;
      lines.push(`<a id="year-${year}-${venueId(venue)}"></a>`, '', `### ${escape(venue)}${venue === 'arXiv' ? ' · 预印本' : ''}`, '',
        '| 论文 | 创新 | 资源 |', '| --- | --- | --- |');
      for (const paper of papers) {
        const refs = Object.entries(paper.links).filter(([key]) => key !== 'paper')
          .map(([key, url]) => link(key === 'code' ? '代码' : '项目', url));
        const source = paper.sources.find(item => !Object.values(paper.links).includes(item.url));
        if (source) refs.push(link('核验', source.url));
        refs.push(link('引用', `literature/citations.md#cite-${paper.id}`));
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
    lines.push(`| ${escape(venue)}${venue === 'arXiv' ? '（预印本）' : ''} | ${entries.length} | ${links.join(' · ')} | ${Object.hasOwn(legacy, venue) ? `[旧笔记](../archive/venues/${legacy[venue]}.md)` : '—'} |`);
  }
  lines.push('', 'NeurIPS 两个分轨合并索引，条目仍保留准确分轨。历史笔记保留原收集范围，尚未全面复核，不等同于当前核心目录。', '',
    '[引用导出](citations.md) · [更新记录](../CHANGELOG.md)');
  return `${lines.join('\n')}\n`;
}

export function renderCatalog(catalog) {
  const lines = [
    '# Video Anomaly Understanding — 精选核验目录', '',
    '这是从结构化数据生成的精选核验目录，并非完整文献综述。原仓库的会议笔记作为额外资料保留，尚未全面复核，不应视作本目录的核验条目。', '',
    `数据更新时间：${catalog.updatedAt} · ${catalog.papers.length} 篇论文。另见 [按年份阅读](../llm4vad.md)、[发表索引](venues.md)、[阅读路线](reading-guide.md)与[引用导出](citations.md)。`, '',
    '## Core research / 核心研究', '',
  ];
  const paperLines = paper => {
    const refs = [...Object.entries(paper.links).map(([key, url]) => link(key, url)), link('引用 / BibTeX', `citations.md#cite-${paper.id}`)].join(' · ');
    return [`<a id="paper-${paper.id}"></a>`, '', `### ${escape(paper.title)}`, '', `**${paper.year} · ${escape(paper.venue)}** · ${refs}`, '', `**创新：${escape(paper.mechanism)}**`, '', escape(paper.summary), '', `- 任务：${paper.tasks.map(escape).join('、')}`, ...(paper.tags?.length ? [`- 方法与场景标签：${paper.tags.map(escape).join('、')}`] : []), ...(paper.secondaryMethods ?? []).flatMap(method => [
      `- 兼属方法：${escape(catalog.clusters.find(cluster => cluster.id === method.cluster)?.name ?? method.cluster)}；${link('归类依据', method.evidence.url)} — ${escape(method.evidence.note)}`,
    ]), `- 核心启示：${escape(paper.takeaway)}`, `- 阅读关注：${escape(paper.limitation.replace(/^阅读关注：/, ''))}`, `- 核验：${paper.verifiedAt}；${paper.sources.map((source, i) => link(`来源 ${i + 1}`, source.url)).join(' · ')}`, ''];
  };
  for (const cluster of catalog.clusters) {
    const papers = catalog.papers.filter(paper => paper.scope === 'core' && paper.cluster === cluster.id);
    if (!papers.length) continue;
    lines.push(`## ${escape(cluster.name)}`, '', escape(cluster.description), '', `研究问题：${escape(cluster.question)}`, '');
    for (const paper of papers) lines.push(...paperLines(paper));
  }
  lines.push('## Datasets / 数据资源', '', '完整协议与关联论文见 [数据集索引](benchmarks.md)。数据登记不要求图片，也不代表资源已开放下载。已有图片保留作者署名。', '');
  for (const dataset of catalog.datasets) {
    lines.push(`### ${escape(dataset.name)}`, '', `**${dataset.year} · ${escape(dataset.venue)}** · ${link('来源入口', dataset.links.website)}`, '', escape(dataset.description), '', `- 评估协议：${escape(dataset.protocol)}`, ...(dataset.thumbnail ? [`- 图片：${link(escape(dataset.thumbnail.alt), dataset.thumbnail.sourceUrl)}；署名：${escape(dataset.thumbnail.credit)}`] : []), `- 核验：${dataset.sources.map((source, i) => link(`来源 ${i + 1}`, source.url)).join(' · ')}`, '');
  }
  return `${lines.join('\n').trimEnd()}\n`;
}

const generatedNote = '[引用导出](citations.md) · [更新记录](../CHANGELOG.md)';
const paperRef = paper => link(escape(paper.shortTitle), `catalog.md#paper-${paper.id}`);

export function renderComparison(catalog) {
  const lines = ['# 视频异常理解 · 方法比较', '', '[论文年表](../llm4vad.md) · [数据集与评测](benchmarks.md) · [阅读路线](reading-guide.md)', '', generatedNote, '',
    '按方法方向比较输出、训练与适配、运行设置和验证方式。每个已填写单元格链接到证据；待核验表示尚未完成该维度核对，不代表论文没有该能力。基准论文记录其评测对象。', '',
    '冻结主模型不等于整个流程无需训练；提示搜索、轻量模块训练和权重微调分别记录。在线／流式标签不能单独证明不访问未来帧。不同数据划分、输入模态与评测协议下的数值不作统一排名。', ''];
  for (const cluster of catalog.clusters) {
    lines.push(`## ${escape(cluster.name)}`, '', `| 论文 | ${Object.values(comparisonFields).join(' | ')} |`, `| --- | ${Object.keys(comparisonFields).map(() => '---').join(' | ')} |`);
    for (const paper of catalog.papers.filter(p => p.cluster === cluster.id)) {
      const cells = Object.keys(comparisonFields).map(key => {
        const fact = paper.comparison?.[key];
        return fact ? link(escape(fact.values.join('；')), fact.evidence.url) : '待核验';
      });
      lines.push(`| ${paperRef(paper)} | ${cells.join(' | ')} |`);
    }
    lines.push('');
  }
  return `${lines.join('\n').trimEnd()}\n`;
}

export function renderBenchmarks(catalog) {
  const lines = ['# 视频异常理解 · 数据集与评测', '', '[论文年表](../llm4vad.md) · [方法比较](comparison.md) · [阅读路线](reading-guide.md)', '', generatedNote, '',
    `${catalog.datasets.length} 个数据资源记录。登记依据论文与作者来源，不以缩略图为前提，也不等同于已发布可下载数据。关联论文只包含已核验关系；没有关联不表示没有使用。`, '',
    '| 数据集 | 年份／发表 | 任务 | 标注 |', '| --- | --- | --- | --- |',
    ...catalog.datasets.map(d => `| [${escape(d.name)}](#dataset-${d.id}) | ${d.year} · ${escape(d.venue)} | ${d.tasks.map(escape).join('、')} | ${d.annotations.map(escape).join('、')} |`), ''];
  for (const d of catalog.datasets) {
    const papers = catalog.papers.filter(p => p.datasetIds.includes(d.id));
    lines.push(`<a id="dataset-${d.id}"></a>`, '', `## ${escape(d.name)}`, '', escape(d.description), '',
      `- 模态：${d.modalities.map(escape).join('、')}`, `- 评测协议／阅读关注：${escape(d.protocol)}`,
      `- 来源入口：${link('作者／论文', d.links.website)}${d.links.paper && d.links.paper !== d.links.website ? ` · ${link('论文', d.links.paper)}` : ''}`,
      `- 已关联论文：${papers.map(paperRef).join(' · ') || '待核验'}`,
      `- 核验依据：${d.sources.map(s => `${link('来源', s.url)} — ${escape(s.note)}`).join('；')}`, '');
  }
  return `${lines.join('\n').trimEnd()}\n`;
}

export function renderGuides(catalog) {
  const lines = ['# 视频异常理解 · 阅读路线', '', '[论文年表](../llm4vad.md) · [方法比较](comparison.md) · [数据集与评测](benchmarks.md) · [背景综述](../research/README.md)', '', generatedNote, '',
    '按研究问题选择路线。以下顺序是编辑阅读建议，不表示论文之间存在引用或继承关系。', '',
    ...catalog.guides.map(g => `- [${escape(g.title)}](#guide-${g.id})`), ''];
  for (const g of catalog.guides) {
    lines.push(`<a id="guide-${g.id}"></a>`, '', `## ${escape(g.title)}`, '', escape(g.description), '');
    g.steps.forEach((s, i) => lines.push(`${i + 1}. ${paperRef(catalog.papers.find(p => p.id === s.paperId))}：${escape(s.note)}`));
    lines.push('');
  }
  return `${lines.join('\n').trimEnd()}\n`;
}

// Protect literal source text and acronym capitalization in traditional BibTeX.
const tex = value => value.replace(/[\\{}&%$#_~^]/g, c => ({
  '\\': '\\textbackslash{}', '{': '\\{', '}': '\\}', '&': '\\&', '%': '\\%',
  '$': '\\$', '#': '\\#', '_': '\\_', '~': '\\textasciitilde{}', '^': '\\textasciicircum{}',
}[c])).normalize('NFD').replace(/([A-Za-z])([\u0300-\u036f])/g, (match, letter, mark) => {
  const accent = { '\u0300': '`', '\u0301': "'", '\u0302': '^', '\u0303': '~', '\u0308': '"', '\u0327': 'c' }[mark];
  return accent ? `{\\${accent}{${letter}}}` : match.normalize('NFC');
}).replace(/[–—]/g, '--').replace(/[’‘]/g, "'");

export function renderBibEntry(paper) {
  const c = paper.citation;
  const fields = [
    ['author', c.authors.map(tex).join(' and ')],
    ['title', `{${tex(c.title)}}`], ['year', String(c.year)],
  ];
  if (c.publication) fields.push([c.type === 'article' ? 'journal' : 'booktitle', tex(c.publication)]);
  for (const key of ['volume', 'number', 'pages']) if (c[key]) fields.push([key, key === 'pages' ? c[key].replace(/[–—]|(?<!-)-(?!-)/g, '--') : tex(c[key])]);
  if (c.arxivId) fields.push(['eprint', c.arxivId], ['archivePrefix', 'arXiv']);
  if (c.doi) fields.push(['doi', c.doi]);
  fields.push(['url', c.url]);
  return `@${c.type}{${c.key},\n${fields.map(([k, v]) => `  ${k} = {${v}}`).join(',\n')}\n}`;
}

export function renderBibliography(catalog) {
  return `% Paper citations for Awesome Thinking with VAD. Verified ${catalog.updatedAt}.\n% Cite the original papers; see citations.md for version and source details.\n\n${catalog.papers.map(renderBibEntry).join('\n\n')}\n`;
}

export function renderCitations(catalog) {
  const published = catalog.papers.filter(p => p.citation.version === 'published').length;
  const lines = ['# 论文引用 / Citations', '', '[论文年表](../llm4vad.md) · [更新记录](../CHANGELOG.md) · [查看完整 BibTeX](references.bib) · [下载 references.bib](https://raw.githubusercontent.com/2-mo/Awesome-Thinking-with-VAD/main/literature/references.bib)', '',
    `${catalog.papers.length} 篇论文的作者与 BibTeX；其中 ${published} 条引用正式发表版本，${catalog.papers.length - published} 条引用预印本。核验日期：${catalog.updatedAt}。`, '',
    '复制下方单篇 BibTeX，或下载整库加入文献管理器。优先引用正式版本；尚未取得完整正式书目信息时，明确导出预印本。DOI 未核验时不填写；arXiv DOI 仅用于预印本，不代替会议／期刊 DOI。引用的是原始论文，不是本仓库。', '',
    '| 论文 | 引用版本 | DOI |', '| --- | --- | --- |'];
  for (const p of catalog.papers) {
    const c = p.citation;
    lines.push(`| [${escape(p.shortTitle)}](#cite-${p.id}) | ${c.year} · ${c.version === 'published' ? escape(p.venue) : 'arXiv 预印本'} | ${c.doi ? link(escape(c.doi), `https://doi.org/${c.doi}`) : '未核验'} |`);
  }
  for (const p of catalog.papers) {
    const c = p.citation;
    lines.push('', `<a id="cite-${p.id}"></a>`, '', `## ${escape(p.shortTitle)}`, '', `**${escape(c.title)}**`, '',
      `作者（原顺序）：${c.authors.map(escape).join('；')}`, '',
      `引用版本：${c.year} · ${c.version === 'published' ? escape(c.publication) : 'arXiv 预印本'}。${c.version === 'preprint' && p.venue !== 'arXiv' ? `目录发表身份为 ${escape(p.venue)} ${p.year}；此处导出预印本，正式书目信息待补。` : ''}`, '',
      `来源：${c.sources.map(s => `${link('核验依据', s.url)} — ${escape(s.note)}`).join('；')}`, '',
      '```bibtex', renderBibEntry(p), '```');
  }
  return `${lines.join('\n')}\n`;
}

if (process.argv[1] && realpathSync(process.argv[1]) === fileURLToPath(import.meta.url)) {
  try {
    const catalog = await readCatalog();
    const errors = validateCatalog(catalog);
    if (errors.length) throw new Error(errors.join('\n'));
    const outputs = [
      ['literature/catalog.md', renderCatalog(catalog)],
      ['llm4vad.md', renderLiterature(catalog)],
      ['literature/venues.md', renderVenueIndex(catalog)],
      ['literature/comparison.md', renderComparison(catalog)],
      ['literature/benchmarks.md', renderBenchmarks(catalog)],
      ['literature/reading-guide.md', renderGuides(catalog)],
      ['literature/citations.md', renderCitations(catalog)],
      ['literature/references.bib', renderBibliography(catalog)],
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
