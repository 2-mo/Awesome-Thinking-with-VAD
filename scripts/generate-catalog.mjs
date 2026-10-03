import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { realpathSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname } from 'node:path';
import { readCatalog, validateCatalog, comparisonFields } from './catalog.mjs';

const outputPath = name => fileURLToPath(new URL(`../${name}`, import.meta.url));
const escape = value => value.replace(/[\\`*_{}\[\]<>|]/g, '\\$&').replace(/\s+/g, ' ').trim();
const link = (label, url) => `[${label}](<${url}>)`;
const displayVenue = venue => venue === 'NeurIPS Datasets and Benchmarks' || venue === 'NeurIPS Evaluations and Datasets' ? 'NeurIPS' : venue;
const venueId = venue => venue.toLowerCase().replace(/[^a-z0-9]+/g, '-');
const venues = catalog => [...new Set(catalog.papers.map(paper => displayVenue(paper.venue)))].sort((a, b) => (a === 'arXiv') - (b === 'arXiv') || a.localeCompare(b));
const years = catalog => [...new Set(catalog.papers.map(paper => paper.year))].sort((a, b) => b - a);
// Reuse the original llm4vad.md cards: heading, publication/code badges, quote, figure.
const venueColors = { CVPR: '1E90FF', ICCV: '00CED1', ECCV: '0B84FE', NeurIPS: '2DB55D', ICML: 'FF6B6B', ICLR: '4B0082', AAAI: '000080', 'ACM MM': 'FF69B4', arXiv: 'b31b1b' };
const badgePart = value => encodeURIComponent(String(value).replace(/-/g, '--').replace(/_/g, '__').replace(/ /g, '_'));
const badge = (label, message, color, target, logo = '') => link(`![${escape(label)}](https://img.shields.io/badge/${badgePart(label)}-${badgePart(message)}-${color}${logo ? `?logo=${logo}` : ''})`, target);

export function renderPaperCard(paper) {
  const venue = displayVenue(paper.venue);
  const badges = [badge(venue, paper.year, venueColors[venue] ?? '537A7A', paper.links.paper, venue === 'arXiv' ? 'arxiv' : '')];
  if (paper.links.code) {
    const match = paper.links.code.match(/^https:\/\/github\.com\/([\w.-]+\/[\w.-]+)(?:\/|$)/);
    badges.push(match ? link(`![Code](https://img.shields.io/github/stars/${match[1]}?style=social&label=Code&logo=github)`, paper.links.code)
      : badge('Code', 'GitHub', 'black', paper.links.code, 'github'));
  }
  if (paper.links.project) badges.push(badge('Project', 'Website', '537A7A', paper.links.project));
  const track = paper.venue.startsWith('NeurIPS ') ? `${escape(paper.venue.slice('NeurIPS '.length))} · ` : '';
  const refs = [link('论文', paper.links.paper), link('阅读笔记', `literature/catalog.md#paper-${paper.id}`), link('引用 / BibTeX', `literature/citations.md#cite-${paper.id}`)];
  const lines = [`<a id="paper-${paper.id}"></a>`, '', `#### ${escape(paper.title)}`, '', ...badges, '',
    `${track}${refs.join(' · ')}`, '', `> **${escape(paper.shortTitle)} · ${escape(paper.mechanism)}**`, `>`, `> ${escape(paper.summary)}`, ''];
  if (paper.classification) lines.push(`方法归类按题名暂定，${link('核验依据', paper.classification.evidence.url)}。`, '');
  if (paper.figure) {
    const f = paper.figure;
    lines.push(`[![${escape(f.alt)}](${f.src})](${f.src})`, '',
      `*${escape(f.caption)} ${escape(f.credit)} · ${link('来源', `assets/papers/README.md#figure-${paper.id}`)}*`, '');
  } else lines.push(`*配图待补：尚未取得可核验的论文原图。${link('查看检索记录', `assets/papers/README.md#missing-${paper.id}`)}*`, '');
  lines.push('---', '');
  return lines;
}

/** Illustrated cards grouped by year/venue, generated from the same verified papers. */
export function renderLiterature(catalog) {
  const lines = [
    '# 视频异常理解 · 论文年表', '',
    '[按创新思路阅读](literature/catalog.md) · [方法比较](literature/comparison.md) · [数据集与评测](literature/benchmarks.md) · [阅读路线](literature/reading-guide.md) · [按会议查找](literature/venues.md) · [研究地图](https://2-mo.github.io/Awesome-Thinking-with-VAD/)', '',
    `更新：${catalog.updatedAt} · ${catalog.papers.length} 篇论文 · ${catalog.clusters.length} 个方法方向。`, '',
    '聚焦视频异常解释、推理、时序定位与理解评估，以及直接支撑这些目标的语义表征方法。会议与年份采用已核验的正式发表信息；未确认录用的论文保留 arXiv。', '',
    `已配原论文图片 ${catalog.papers.filter(p => p.figure).length} / ${catalog.papers.length} 篇；点击图片查看大图。[图片来源与待补记录](assets/papers/README.md)。完整阅读关注见 [研究目录](literature/catalog.md)，作者、DOI 与 BibTeX 见 [引用导出](literature/citations.md)。`, '',
    years(catalog).map(year => `[${year}](#year-${year})`).join(' · '), '',
  ];
  for (const year of years(catalog)) {
    lines.push(`<a id="year-${year}"></a>`, '', `## ${year}`, '');
    for (const venue of venues(catalog)) {
      const papers = catalog.papers.filter(paper => paper.year === year && displayVenue(paper.venue) === venue)
        .sort((a, b) => a.shortTitle.localeCompare(b.shortTitle));
      if (!papers.length) continue;
      lines.push(`<a id="year-${year}-${venueId(venue)}"></a>`, '', `### ${escape(venue)}${venue === 'arXiv' ? ' · 预印本' : ''}`, '');
      for (const paper of papers) {
        lines.push(...renderPaperCard(paper));
      }
      lines.push('');
    }
  }
  return `${lines.join('\n').trimEnd()}\n`;
}

export function renderFigureSources(catalog) {
  const illustrated = catalog.papers.filter(p => p.figure);
  const missing = catalog.papers.filter(p => !p.figure);
  const lines = ['# 论文配图来源', '',
    `核验：${catalog.updatedAt} · 已配图 ${illustrated.length} / ${catalog.papers.length} 篇。由 data/catalog.json 自动生成。`, '',
    '图片用于原论文导读，版权归原作者／出版方；不纳入本仓库文字与代码的许可。优先保留作者原图；PDF 图区仅作原图渲染，不重绘、补造或更改科研内容。这里的预印本／作者图版本可能早于正式发表版本，发表信息与图片来源分别记录。', '',
    '[返回论文年表](../../llm4vad.md)', '', '| 论文 | 本地图 | 内容 | 原始文件 | 论文／作者页面 | 署名 | 核验日期 |', '| --- | --- | --- | --- | --- | --- | --- |'];
  for (const p of illustrated) {
    const f = p.figure;
    lines.push(`| <a id="figure-${p.id}"></a>${link(escape(p.shortTitle), `../../llm4vad.md#paper-${p.id}`)} | ${link('图片', f.src.replace('assets/papers/', ''))} | ${escape(f.caption)} | ${link('原图 / PDF', f.sourceUrl)} | ${link('来源页面', f.sourcePageUrl)} | ${escape(f.credit)} | ${f.verifiedAt} |`);
  }
  if (missing.length) {
    lines.push('', '## 待补原图', '', '未取得图片不等同于论文没有框架图或未公开全文。', '');
    for (const p of missing) lines.push(`<a id="missing-${p.id}"></a>`, '', `### ${escape(p.shortTitle)}`, '',
      escape(p.figurePending?.note ?? '尚未取得可核验原图。'), '',
      (p.figurePending?.sources ?? p.sources).map((s, i) => link(`已查来源 ${i + 1}`, s.url)).join(' · '), '');
  }
  return `${lines.join('\n').trimEnd()}\n`;
}

export function renderVenueIndex(catalog) {
  const legacy = { AAAI: 'aaai', CVPR: 'cvpr', ICCV: 'iccv', ECCV: 'eccv', NeurIPS: 'neurips', ICML: 'icml', ICLR: 'iclr', 'ACM MM': 'acmmm', IJCAI: 'ijcai' };
  const journals = { TPAMI: 'tpami', IJCV: 'ijcv', TIP: 'tip', TNNLS: 'tnnls' };
  // Editorial influence priority for count ties in this research area.
  const influenceOrder = ['TPAMI', 'IJCV', 'CVPR', 'ICCV', 'NeurIPS', 'ICML', 'ICLR', 'ECCV', 'TIP', 'ACL', 'AAAI', 'IJCAI', 'ACM MM', 'TNNLS', 'NAACL Findings', 'WACV', 'arXiv'];
  const influenceRank = venue => {
    const rank = influenceOrder.indexOf(venue);
    return rank < 0 ? influenceOrder.length : rank;
  };
  const groups = venues(catalog).map(venue => ({
    venue,
    entries: catalog.papers.filter(paper => displayVenue(paper.venue) === venue),
  })).sort((a, b) => b.entries.length - a.entries.length
    || influenceRank(a.venue) - influenceRank(b.venue)
    || a.venue.localeCompare(b.venue));
  const lines = [
    '# 视频异常理解 · 发表索引', '',
    `更新：${catalog.updatedAt}。从 [结构化目录](../data/catalog.json) 自动生成；年份链接进入当前核验年表。`, '',
    '| 发表场所 | 篇数 | 年份 | 历史笔记 |', '| --- | ---: | --- | --- |',
  ];
  for (const { venue, entries } of groups) {
    const links = years(catalog).filter(year => entries.some(paper => paper.year === year))
      .map(year => `[${year}](../llm4vad.md#year-${year}-${venueId(venue)})`);
    const notes = Object.hasOwn(legacy, venue) ? `[旧笔记](../archive/venues/${legacy[venue]}.md)`
      : Object.hasOwn(journals, venue) ? `[期刊笔记](../archive/journals/${journals[venue]}.md)` : '—';
    lines.push(`| ${escape(venue)}${venue === 'arXiv' ? '（预印本）' : ''} | ${entries.length} | ${links.join(' · ')} | ${notes} |`);
  }
  lines.push('', 'NeurIPS 各分轨合并索引，条目仍保留准确分轨。历史笔记保留原收集范围，尚未全面复核，不等同于当前核心目录。', '',
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
    return [`<a id="paper-${paper.id}"></a>`, '', `### ${escape(paper.title)}`, '', `**${paper.year} · ${escape(paper.venue)}** · ${refs}`, '', `**创新：${escape(paper.mechanism)}**`, '', escape(paper.summary), '', `- 任务：${paper.tasks.map(escape).join('、')}`, ...(paper.tags?.length ? [`- 方法与场景标签：${paper.tags.map(escape).join('、')}`] : []), ...(paper.classification ? [`- 归类状态：按题名暂定；${link('依据', paper.classification.evidence.url)} — ${escape(paper.classification.evidence.note)}`] : []), ...(paper.secondaryMethods ?? []).flatMap(method => [
      `- 兼属方法：${escape(catalog.clusters.find(cluster => cluster.id === method.cluster)?.name ?? method.cluster)}；${link('归类依据', method.evidence.url)} — ${escape(method.evidence.note)}`,
    ]), `- 核心启示：${escape(paper.takeaway)}`, `- 阅读关注：${escape(paper.limitation.replace(/^阅读关注：/, ''))}`, `- 核验：${paper.verifiedAt}；${paper.sources.map((source, i) => link(`来源 ${i + 1}`, source.url)).join(' · ')}`, ''];
  };
  for (const cluster of catalog.clusters) {
    const papers = catalog.papers.filter(paper => paper.scope === 'core' && paper.cluster === cluster.id);
    if (!papers.length) continue;
    lines.push(`## ${escape(cluster.name)}`, '', escape(cluster.description), '', `研究问题：${escape(cluster.question)}`, '');
    if (cluster.branchAt) {
      const anchor = catalog.papers.find(p => p.id === cluster.branchAt.paperId);
      lines.push(`分叉节点：${link(escape(anchor.shortTitle), `#paper-${anchor.id}`)}；${link('分叉依据', cluster.branchAt.evidence.url)} — ${escape(cluster.branchAt.evidence.note)}`, '');
    }
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
  const isRetrieval = d => d.tasks.includes('异常检索');
  const isUnderstanding = d => !isRetrieval(d) && d.tasks.some(task => ['异常解释', '异常推理', '视频问答', '基准评测'].includes(task));
  const groups = [
    { id: 'detection-data', title: '异常检测', matches: d => !isRetrieval(d) && !isUnderstanding(d) },
    { id: 'understanding-data', title: '理解与推理', matches: isUnderstanding },
    { id: 'retrieval-data', title: '异常检索', matches: isRetrieval },
  ].map(group => ({ ...group, datasets: catalog.datasets.filter(group.matches).sort((a, b) => a.year - b.year || a.name.localeCompare(b.name)) })).filter(group => group.datasets.length);
  const lines = ['# 视频异常理解 · 数据集与评测', '', '[论文年表](../llm4vad.md) · [方法比较](comparison.md) · [阅读路线](reading-guide.md)', '',
    `${catalog.datasets.length} 个数据集与评测资源，按任务浏览。点击徽章进入论文或作者发布页。`, '',
    groups.map(group => link(group.title, `#${group.id}`)).join(' · '), '',
    '<details>', '<summary>快速跳转</summary>', '',
    ...groups.map(group => `- **${group.title}**：${group.datasets.map(d => link(escape(d.name), `#dataset-${d.id}`)).join(' · ')}`), '', '</details>', ''];
  for (const group of groups) {
    lines.push(`<a id="${group.id}"></a>`, '', `## ${group.title}`, '');
    for (const d of group.datasets) {
      const papers = catalog.papers.filter(p => p.datasetIds.includes(d.id));
      const venue = displayVenue(d.venue);
      const paperUrl = d.links.paper ?? d.links.website;
      const publication = /^(TPAMI|TIP|TNNLS|TCYB|TIFS|IJCV)$/.test(venue)
        ? `${link(`![${escape(venue)}](https://img.shields.io/badge/${badgePart(venue)}-537A7A?style=flat)`, paperUrl)} · ${d.year}`
        : badge(venue, d.year, venueColors[venue] ?? '537A7A', paperUrl);
      const a = d.availability;
      const resource = {
        available: ['下载', '537A7A'], partial: ['部分开放', 'A87938'], pending: ['待发布', '8A8A8A'], unverified: ['项目入口', '537A7A'],
      }[a?.status ?? 'unverified'];
      lines.push(`<a id="dataset-${d.id}"></a>`, '', `### ${escape(d.name)}`, '',
        `${publication} ${badge('Data', resource[0], resource[1], a?.evidence.url ?? d.links.website)}`, '',
        `> ${escape(d.description)}`, '',
        `**标注** · ${d.annotations.map(escape).join(' · ')}`, '');
      if (d.usageNote) lines.push(`**使用说明** · ${escape(d.usageNote)}`, '');
      if (d.thumbnail) {
        const f = d.thumbnail;
        const src = f.src.startsWith('/datasets/') ? `../public${f.src}` : `../${f.src}`;
        lines.push(`[![${escape(f.alt)}](${src})](${src})`, '',
          `*${f.caption ? `${escape(f.caption)} ` : ''}${escape(f.credit)} · ${link('图片来源', f.sourceUrl)}*`, '');
      }
      lines.push('<details>', '<summary>划分与相关工作</summary>', '', escape(d.protocol), '');
      if (a?.note) lines.push(`获取：${escape(a.note)}`, '');
      if (d.composition?.baseDatasetIds?.length) lines.push(`基础数据：${d.composition.baseDatasetIds.map(id => link(escape(catalog.datasets.find(base => base.id === id).name), `#dataset-${id}`)).join(' · ')}`, '');
      if (papers.length) lines.push(`相关工作：${papers.map(paperRef).join(' · ')}`, '');
      const sources = [...new Set(d.sources.map(s => s.url))];
      lines.push(`来源：${sources.map((url, i) => link(String(i + 1), url)).join(' · ')}`, '', '</details>', '', '---', '');
    }
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
  if (c.version === 'pending') return '';
  const fields = [
    ['author', c.authors.map(tex).join(' and ')],
    ['title', `{${tex(c.title)}}`], ['year', String(c.year)],
  ];
  if (c.publication) fields.push(c.version === 'accepted' ? ['note', `Accepted to ${tex(c.publication)}`] : [c.type === 'article' ? 'journal' : 'booktitle', tex(c.publication)]);
  for (const key of ['volume', 'number', 'pages']) if (c[key]) fields.push([key, key === 'pages' ? c[key].replace(/[–—]|(?<!-)-(?!-)/g, '--') : tex(c[key])]);
  if (c.arxivId) fields.push(['eprint', c.arxivId], ['archivePrefix', 'arXiv']);
  if (c.doi) fields.push(['doi', c.doi]);
  fields.push(['url', c.url]);
  return `@${c.type}{${c.key},\n${fields.map(([k, v]) => `  ${k} = {${v}}`).join(',\n')}\n}`;
}

export function renderBibliography(catalog) {
  return `% Paper citations for Awesome Thinking with VAD. Verified ${catalog.updatedAt}.\n% Cite the original papers; see citations.md for version and source details.\n\n${catalog.papers.map(renderBibEntry).filter(Boolean).join('\n\n')}\n`;
}

export function renderCitations(catalog) {
  const published = catalog.papers.filter(p => p.citation.version === 'published').length;
  const accepted = catalog.papers.filter(p => p.citation.version === 'accepted').length;
  const pending = catalog.papers.filter(p => p.citation.version === 'pending').length;
  const citationLabel = p => p.citation.version === 'published' ? escape(p.citation.publication)
    : p.citation.version === 'accepted' ? `${escape(p.citation.publication)} · 已录用`
    : p.citation.version === 'pending' ? `${escape(p.venue)} · 已录用，书目待补` : 'arXiv 预印本';
  const lines = ['# 论文引用 / Citations', '', '[论文年表](../llm4vad.md) · [更新记录](../CHANGELOG.md) · [查看完整 BibTeX](references.bib) · [下载 references.bib](https://raw.githubusercontent.com/2-mo/Awesome-Thinking-with-VAD/main/literature/references.bib)', '',
    `${catalog.papers.length} 篇论文，${catalog.papers.length - pending} 条完整引用；其中 ${published} 条引用正式发表版本，${accepted} 条为会议已录用记录，${catalog.papers.length - published - accepted - pending} 条引用预印本。${pending ? `另有 ${pending} 篇已录用论文待补书目，暂不导出 BibTeX。` : ''}核验日期：${catalog.updatedAt}。`, '',
    '复制下方单篇 BibTeX，或下载整库加入文献管理器。优先引用正式版本；尚未取得完整正式书目信息时，区分已录用记录与预印本。DOI 未核验时不填写；arXiv DOI 仅用于预印本，不代替会议／期刊 DOI。引用的是原始论文，不是本仓库。', '',
    '| 论文 | 引用版本 | DOI |', '| --- | --- | --- |'];
  for (const p of catalog.papers) {
    const c = p.citation;
    lines.push(`| [${escape(p.shortTitle)}](#cite-${p.id}) | ${c.year} · ${citationLabel(p)} | ${c.doi ? link(escape(c.doi), `https://doi.org/${c.doi}`) : '未核验'} |`);
  }
  for (const p of catalog.papers) {
    const c = p.citation;
    if (c.version === 'pending') {
      lines.push('', `<a id="cite-${p.id}"></a>`, '', `## ${escape(p.shortTitle)}`, '', `**${escape(c.title)}**`, '',
        `状态：${c.year} · ${citationLabel(p)}。${escape(c.note)}`, '',
        `来源：${c.sources.map(s => `${link('核验依据', s.url)} — ${escape(s.note)}`).join('；')}`);
      continue;
    }
    lines.push('', `<a id="cite-${p.id}"></a>`, '', `## ${escape(p.shortTitle)}`, '', `**${escape(c.title)}**`, '',
      `作者（原顺序）：${c.authors.map(escape).join('；')}`, '',
      `引用版本：${c.year} · ${citationLabel(p)}。${c.version === 'preprint' && p.venue !== 'arXiv' ? `目录发表身份为 ${escape(p.venue)} ${p.year}；此处导出预印本，正式书目信息待补。` : ''}`, '',
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
      ['assets/papers/README.md', renderFigureSources(catalog)],
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
