import { useEffect, useMemo, useRef, useState } from "react";
import rawCatalog from "../data/catalog.json";
import type { Catalog, Paper } from "./types";
import ResearchMap from "./components/ResearchMap";
import PaperFigure from "./components/PaperFigure";
import PaperNavigation from "./components/PaperNavigation";
import { MAP_CANVAS_WIDTH } from "./components/map-typography";
import { createPublicationLayout } from "./components/publication-layout";
import { clusterName, contributionLabels, isMapPaper, paperContribution, paperMethods, publicationVenue } from "./publication";

const catalog = rawCatalog as Catalog;
const mapPapers = catalog.papers.filter(isMapPaper);
const paperById = new Map(mapPapers.map(paper => [paper.id, paper]));
const mapLayout = createPublicationLayout(mapPapers, catalog.clusters, { width: MAP_CANVAS_WIDTH });
const REPO = "https://github.com/2-mo/Awesome-Thinking-with-VAD";
const initial = new URLSearchParams(window.location.search);
const venues = [...new Set(mapPapers.map(p => publicationVenue(p.venue)))].sort();
const years = [...new Set(mapPapers.map(p => String(p.year)))].sort().reverse();
const tasks = [...new Set(mapPapers.flatMap(p => p.tasks))];
const initialValue = (key: string, values: string[]) => {
  const value = initial.get(key) || "";
  return values.includes(value) ? value : "";
};
const comparisonLabels = {
  outputs: "输出／评测对象", training: "训练与适配", inference: "运行设置",
  futureFrames: "未来帧访问", evaluation: "验证方式",
};
const datasetKindLabels = { original: "原始数据", annotation: "派生标注", resplit: "重新划分", mixed: "混合来源" };
const datasetAvailabilityLabels = { available: "已开放", partial: "部分开放", pending: "待发布", unverified: "待核验" };

export default function App() {
  const [query, setQuery] = useState(initial.get("q") || "");
  const [cluster, setCluster] = useState(initialValue("cluster", catalog.clusters.map(c => c.id)));
  const [venue, setVenue] = useState(() => {
    const value = publicationVenue(initial.get("venue") || "");
    return venues.includes(value) ? value : "";
  });
  const [year, setYear] = useState(initialValue("year", years));
  const [task, setTask] = useState(initialValue("task", tasks));
  const [selectedId, setSelectedId] = useState(initialValue("paper", mapPapers.map(p => p.id)));
  const [readingLineId, setReadingLineId] = useState(cluster);
  const dialog = useRef<HTMLDialogElement>(null);
  const paperTitle = useRef<HTMLHeadingElement>(null);
  const navigating = useRef(false);
  const selected = mapPapers.find(p => p.id === selectedId);
  const visiblePapers = useMemo(() => mapPapers.filter(p => {
    const text = `${p.shortTitle} ${p.title} ${p.summary} ${p.mechanism} ${p.venue} ${p.tasks.join(" ")} ${(p.tags || []).join(" ")}`.toLowerCase();
    return (!cluster || paperMethods(p).includes(cluster))
      && (!venue || publicationVenue(p.venue) === venue)
      && (!year || String(p.year) === year)
      && (!task || p.tasks.includes(task))
      && query.toLowerCase().trim().split(/\s+/).every(word => text.includes(word));
  }), [query, cluster, venue, year, task]);

  useEffect(() => {
    const params = new URLSearchParams();
    for (const [key, value] of Object.entries({ q: query, cluster, venue, year, task, paper: selectedId })) {
      if (value) params.set(key, value);
    }
    window.history.replaceState(null, "", `${window.location.pathname}${params.size ? `?${params}` : ""}`);
  }, [query, cluster, venue, year, task, selectedId]);

  useEffect(() => {
    if (selected && !dialog.current?.open) dialog.current?.showModal();
    else if (!selected && dialog.current?.open) dialog.current.close();
    if (selected && navigating.current) {
      paperTitle.current?.focus({ preventScroll: true });
      if (dialog.current) dialog.current.scrollTop = 0;
      navigating.current = false;
    }
  }, [selected]);

  const openPaper = (id: string) => {
    setReadingLineId(cluster || paperById.get(id)!.cluster);
    setSelectedId(id);
  };
  const navigatePaper = (id: string, lineId: string) => {
    navigating.current = true;
    setReadingLineId(lineId);
    setSelectedId(id);
  };

  const resetFilters = () => {
    setQuery(""); setCluster(""); setVenue(""); setYear(""); setTask("");
  };

  return (
    <div className="map-page">
      <a className="skip-link" href="#main">跳到线路图</a>
      <header className="map-header">
        <div>
          <span className="eyebrow">THINKING WITH VAD</span>
          <h1>异常理解 · 研究线路图</h1>
        </div>
        <nav aria-label="阅读资源">
          <a href={`${REPO}#conference-papers`}>会议与期刊论文 ↗</a>
          <a href={`${REPO}/blob/main/literature/references.bib`}>BibTeX ↗</a>
        </nav>
      </header>

      <form className="map-filters" aria-label="筛选论文" onSubmit={event => event.preventDefault()}>
        <label className="search-field"><span>搜索</span>
          <input type="search" placeholder="论文、方法或关键词" value={query} onChange={event => setQuery(event.target.value)} />
        </label>
        <label><span>Direction</span><select value={cluster} onChange={event => setCluster(event.target.value)}>
          <option value="">All directions</option>
          {catalog.clusters.map(c => <option key={c.id} value={c.id}>{clusterName(c)}{c.branchOf ? " · Branch" : ""}</option>)}
        </select></label>
        <label><span>发表</span><select value={venue} onChange={event => setVenue(event.target.value)}>
          <option value="">全部会议与期刊</option>
          {venues.map(v => <option key={v}>{v}</option>)}
        </select></label>
        <label><span>年份</span><select value={year} onChange={event => setYear(event.target.value)}>
          <option value="">全部年份</option>
          {years.map(y => <option key={y}>{y}</option>)}
        </select></label>
        <label><span>任务</span><select value={task} onChange={event => setTask(event.target.value)}>
          <option value="">全部任务</option>
          {tasks.map(t => <option key={t}>{t}</option>)}
        </select></label>
        <button type="button" onClick={resetFilters} disabled={!(query || cluster || venue || year || task)}>清除筛选</button>
      </form>

      <main id="main" tabIndex={-1} className="map-frame">
        <ResearchMap papers={visiblePapers} allPapers={mapPapers} clusters={catalog.clusters}
          network={mapLayout} selectedId={selectedId || null} onSelect={openPaper} activeCluster={cluster || "all"} onReset={resetFilters} />
      </main>
      <footer className="map-footer">
        <span>点击站点查看论文 · 拖动平移</span>
        <span>更新 {catalog.updatedAt} · {mapPapers.length} 篇论文</span>
      </footer>

      <dialog ref={dialog} className="paper-dialog" aria-labelledby="paper-title" onClose={() => setSelectedId("")}
        onClick={event => { if (event.target === event.currentTarget) dialog.current?.close(); }}>
        {selected && <>
          <div className="paper-heading">
            <span>{contributionLabels[paperContribution(selected)]} · {selected.venue} · {selected.year}</span>
            <button type="button" autoFocus aria-label="关闭论文详情" onClick={() => dialog.current?.close()}>关闭 ×</button>
          </div>
          <h2 id="paper-title" ref={paperTitle} tabIndex={-1}>{selected.shortTitle}</h2>
          <p className="full-title">{selected.title}</p>
          <div className="paper-links">
            <a href={selected.links.paper} target="_blank" rel="noreferrer">论文 ↗</a>
            {selected.links.code && <a href={selected.links.code} target="_blank" rel="noreferrer">代码 ↗</a>}
            {selected.links.project && <a href={selected.links.project} target="_blank" rel="noreferrer">项目 ↗</a>}
            <a href={`${REPO}/blob/main/literature/citations.md#cite-${selected.id}`}>引用信息 ↗</a>
          </div>
          <div className="method-tags">{paperMethods(selected).map(id => {
            const method = catalog.clusters.find(c => c.id === id)!;
            return <span key={id} style={{ borderColor: method.color }}>{clusterName(method)}</span>;
          })}</div>
          <PaperNavigation paper={selected} papers={paperById} lines={mapLayout.lines} lineId={readingLineId}
            onLineChange={setReadingLineId} onNavigate={navigatePaper} />
          {selected.classification?.basis === "title" && <p className="classification-note">
            按题名暂定归类 · 方法细节待正文核验
          </p>}
          {selected.citation.version === "pending" && <p className="verification-date">{selected.citation.note}</p>}
          <h3>{selected.mechanism}</h3>
          <p>{selected.summary}</p>
          {selected.figure && <PaperFigure key={selected.id} figure={selected.figure} />}
          <p>{selected.takeaway}</p>
          <PaperEvidence key={`evidence-${selected.id}`} paper={selected} />
        </>}
      </dialog>
    </div>
  );
}

function PaperEvidence({ paper }: { paper: Paper }) {
  const datasets = catalog.datasets.filter(d => paper.datasetIds.includes(d.id));
  const sources = [...paper.sources, ...(paper.secondaryMethods || []).map(m => m.evidence),
    ...catalog.clusters.filter(c => c.branchAt?.paperId === paper.id).map(c => c.branchAt!.evidence),
    ...(paper.contribution ? [paper.contribution.evidence] : []),
    ...(paper.classification ? [paper.classification.evidence] : [])];
  return <>
    {paper.comparison && <details><summary>方法设置与评测</summary>
      <dl>{Object.entries(comparisonLabels).map(([key, label]) => {
        const fact = paper.comparison?.[key as keyof typeof comparisonLabels];
        return <div key={key}><dt>{label}</dt><dd>{fact
          ? <a href={fact.evidence.url} title={fact.evidence.note} target="_blank" rel="noreferrer">{fact.values.join("；")}</a>
          : "待核验"}</dd></div>;
      })}</dl>
    </details>}
    {!!datasets.length && <details><summary>数据与评测资源</summary>
      {datasets.map(d => <div key={d.id}>
        <h4><a href={d.links.website} target="_blank" rel="noreferrer">{d.name} ↗</a></h4>
        {d.composition && <p>数据性质：{datasetKindLabels[d.composition.kind]} · {d.composition.note}</p>}
        {d.availability && <p>开放状态：<a href={d.availability.evidence.url} target="_blank" rel="noreferrer">{datasetAvailabilityLabels[d.availability.status]}</a> · {d.availability.note}</p>}
        <p>{d.protocol}</p>
      </div>)}
    </details>}
    <details><summary>阅读关注与来源</summary>
      <p>{paper.limitation}</p>
      <ul>{sources.map((source, i) => <li key={`${source.url}-${i}`}><a href={source.url} target="_blank" rel="noreferrer">{source.note}</a></li>)}</ul>
      <p className="verification-date">核验日期 {paper.verifiedAt}</p>
    </details>
  </>;
}
