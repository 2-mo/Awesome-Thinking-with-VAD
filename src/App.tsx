import { useEffect, useMemo, useRef, useState } from "react";
import {
  ArrowDownToLine,
  ArrowRight,
  ArrowUpRight,
  BookOpen,
  Check,
  ChevronLeft,
  ChevronRight,
  Code,
  Copy,
  Database,
  FileText,
  GitBranch,
  Lightbulb,
  List,
  Map,
  Search,
  Share2,
  SlidersHorizontal,
  X,
} from "lucide-react";
import rawCatalog from "../data/catalog.json";
import type { Catalog, Dataset, Paper } from "./types";
import ResearchMap from "./components/ResearchMap";
import DatasetGallery, { datasetImageUrl } from "./components/DatasetGallery";
import stripUrl from "./assets/idea-strip.png";

const catalog = rawCatalog as Catalog;
const REPO = "https://github.com/2-mo/Awesome-Thinking-with-VAD";
const initial = new URLSearchParams(window.location.search);
const initialPaper = catalog.papers.find((p) => p.id === initial.get("paper"));
const initialDataset = catalog.datasets.find(
  (d) => d.id === initial.get("resource"),
);
type View = "map" | "datasets" | "guides";
type Layout = "map" | "timeline" | "list";
type DetailTab = "idea" | "question" | "evidence" | "sources";
const detailTabs: [DetailTab, string][] = [
  ["idea", "创新"],
  ["question", "追问"],
  ["evidence", "证据"],
  ["sources", "来源"],
];
const artIndex: Record<string, number> = {
  alignment: 0,
  explanation: 1,
  reasoning: 2,
  evidence: 3,
  understanding: 4,
};

function ResourceLink({
  href,
  children,
}: {
  href: string;
  children: React.ReactNode;
}) {
  return (
    <a href={href} target="_blank" rel="noreferrer">
      {children}
      <ArrowUpRight size={13} />
    </a>
  );
}
function Art({
  cluster,
  className = "",
}: {
  cluster: string;
  className?: string;
}) {
  return (
    <span
      className={`comic-art ${className}`}
      aria-hidden="true"
      style={{
        backgroundImage: `url(${stripUrl})`,
        backgroundPosition: `${(artIndex[cluster] ?? 0) * 25}% 50%`,
      }}
    />
  );
}
function Pager({
  page,
  total,
  onChange,
  label,
}: {
  page: number;
  total: number;
  onChange: (page: number) => void;
  label: string;
}) {
  return (
    <div className="pager" aria-label={label}>
      <button
        disabled={page <= 0}
        onClick={() => onChange(page - 1)}
        aria-label={`${label}上一页`}
      >
        <ChevronLeft size={15} />
      </button>
      <span>
        {page + 1} / {Math.max(1, total)}
      </span>
      <button
        disabled={page + 1 >= total}
        onClick={() => onChange(page + 1)}
        aria-label={`${label}下一页`}
      >
        <ChevronRight size={15} />
      </button>
    </div>
  );
}

export default function App() {
  const [view, setView] = useState<View>(
    ["datasets", "guides"].includes(initial.get("view") || "")
      ? (initial.get("view") as View)
      : "map",
  );
  const [query, setQuery] = useState(initial.get("q") || "");
  const [cluster, setCluster] = useState(
    catalog.clusters.some((c) => c.id === initial.get("cluster"))
      ? initial.get("cluster")!
      : "all",
  );
  const [venue, setVenue] = useState(
    catalog.papers.some((p) => p.venue === initial.get("venue"))
      ? initial.get("venue")!
      : "all",
  );
  const [year, setYear] = useState(
    catalog.papers.some((p) => String(p.year) === initial.get("year"))
      ? initial.get("year")!
      : "all",
  );
  const [task, setTask] = useState(
    catalog.papers.some((p) => p.tasks.includes(initial.get("task") || ""))
      ? initial.get("task")!
      : "all",
  );
  const [datasetId, setDatasetId] = useState(
    catalog.datasets.some((d) => d.id === initial.get("dataset"))
      ? initial.get("dataset")!
      : "",
  );
  const [selectedId, setSelectedId] = useState<string | null>(
    initialPaper?.id || null,
  );
  const [selectedDataset, setSelectedDataset] = useState(
    initialDataset?.id || catalog.datasets[0].id,
  );
  const [layout, setLayout] = useState<Layout>(
    ["timeline", "list"].includes(initial.get("layout") || "")
      ? (initial.get("layout") as Layout)
      : "map",
  );
  const [showFilters, setShowFilters] = useState(false);
  const [detailTab, setDetailTab] = useState<DetailTab>("idea");
  const [detailPage, setDetailPage] = useState(0);
  const [guidePage, setGuidePage] = useState(0);
  const [guideStepPage, setGuideStepPage] = useState(0);
  const [datasetQuery, setDatasetQuery] = useState(initial.get("dq") || "");
  const [datasetRelatedPage, setDatasetRelatedPage] = useState(0);
  const [datasetReading, setDatasetReading] = useState(
    initial.get("view") === "datasets" && !!initialDataset,
  );
  const [notice, setNotice] = useState("");
  const noticeTimer = useRef<ReturnType<typeof setTimeout> | undefined>(
    undefined,
  );
  const previousFocus = useRef<HTMLElement | null>(null);
  const selected = catalog.papers.find((p) => p.id === selectedId);
  const selectedCluster = catalog.clusters.find(
    (c) => c.id === selected?.cluster,
  );
  const venues = [...new Set(catalog.papers.map((p) => p.venue))].sort((a, b) =>
    a.localeCompare(b),
  );
  const years = [...new Set(catalog.papers.map((p) => p.year))].sort(
    (a, b) => b - a,
  );
  const coreCount = catalog.papers.filter((p) => p.scope === "core").length;
  const hasFilters =
    !!query ||
    cluster !== "all" ||
    year !== "all" ||
    venue !== "all" ||
    task !== "all" ||
    !!datasetId;
  const visiblePapers = useMemo(
    () =>
      catalog.papers.filter((p) => {
        const text =
          `${p.shortTitle} ${p.title} ${p.summary} ${p.mechanism} ${p.takeaway} ${p.venue} ${p.tasks.join(" ")}`.toLowerCase();
        return (
          (cluster === "all" || p.cluster === cluster) &&
          (year === "all" || String(p.year) === year) &&
          (venue === "all" || p.venue === venue) &&
          (task === "all" || p.tasks.includes(task)) &&
          (!datasetId || p.datasetIds.includes(datasetId)) &&
          query
            .toLowerCase()
            .trim()
            .split(/\s+/)
            .every((word) => text.includes(word))
        );
      }),
    [query, cluster, year, venue, task, datasetId],
  );
  const selectedIndex = visiblePapers.findIndex((p) => p.id === selectedId);
  const selectionRelations = catalog.relations.filter(
    (r) => r.source === selectedId || r.target === selectedId,
  );

  useEffect(() => {
    const params = new URLSearchParams();
    if (view !== "map") params.set("view", view);
    if (query) params.set("q", query);
    if (cluster !== "all") params.set("cluster", cluster);
    if (year !== "all") params.set("year", year);
    if (venue !== "all") params.set("venue", venue);
    if (view === "datasets" && datasetQuery) params.set("dq", datasetQuery);
    if (view === "datasets" && datasetReading)
      params.set("resource", selectedDataset);
    if (task !== "all") params.set("task", task);
    if (datasetId) params.set("dataset", datasetId);
    if (selectedId) params.set("paper", selectedId);
    if (layout !== "map") params.set("layout", layout);
    window.history.replaceState(
      null,
      "",
      `${window.location.pathname}${params.size ? `?${params}` : ""}`,
    );
  }, [
    view,
    query,
    cluster,
    year,
    venue,
    task,
    datasetId,
    selectedId,
    layout,
    datasetQuery,
    datasetReading,
    selectedDataset,
  ]);
  useEffect(() => {
    if (selectedId && !visiblePapers.some((p) => p.id === selectedId))
      setSelectedId(null);
  }, [visiblePapers]);
  useEffect(() => {
    setDetailPage(0);
  }, [selectedId, detailTab]);
  useEffect(() => {
    setDatasetRelatedPage(0);
  }, [selectedDataset]);
  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        setSelectedId(null);
        setDatasetReading(false);
        setShowFilters(false);
        previousFocus.current?.focus();
      }
      if (
        event.key === "/" &&
        !["INPUT", "TEXTAREA", "SELECT"].includes(
          (event.target as HTMLElement).tagName,
        )
      ) {
        event.preventDefault();
        document.getElementById("paper-search")?.focus();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);
  useEffect(() => () => clearTimeout(noticeTimer.current), []);
  function announce(message: string) {
    clearTimeout(noticeTimer.current);
    setNotice(message);
    noticeTimer.current = setTimeout(() => setNotice(""), 3000);
  }
  async function copy(text: string, message: string) {
    try {
      await navigator.clipboard.writeText(text);
      announce(message);
    } catch {
      announce("复制未成功，可从地址栏或论文原文复制。");
    }
  }
  function resetFilters() {
    setQuery("");
    setCluster("all");
    setYear("all");
    setVenue("all");
    setTask("all");
    setDatasetId("");
  }
  function selectPaper(id: string) {
    previousFocus.current = document.activeElement as HTMLElement;
    if (!visiblePapers.some((p) => p.id === id)) resetFilters();
    setSelectedId(id);
    setDetailTab("idea");
    setView("map");
    setShowFilters(false);
  }
  function changeView(next: View) {
    setView(next);
    setShowFilters(false);
    setDatasetReading(false);
  }
  function filterCluster(id: string) {
    setCluster(cluster === id ? "all" : id);
    setView("map");
    setShowFilters(false);
  }
  function showDatasetPapers(id: string) {
    resetFilters();
    setDatasetId(id);
    setSelectedId(null);
    setView("map");
  }
  function exportData() {
    const ids = new Set(visiblePapers.map((p) => p.id));
    const datasets = catalog.datasets.filter((d) =>
      visiblePapers.some((p) => p.datasetIds.includes(d.id)),
    );
    const entityIds = new Set([...ids, ...datasets.map((d) => d.id)]);
    const blob = new Blob(
      [
        JSON.stringify(
          {
            updatedAt: catalog.updatedAt,
            papers: visiblePapers,
            datasets,
            relations: catalog.relations.filter(
              (r) => entityIds.has(r.source) && entityIds.has(r.target),
            ),
          },
          null,
          2,
        ),
      ],
      { type: "application/json" },
    );
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "vau-selected-papers.json";
    a.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
    announce("已导出当前文献与关系来源");
  }
  function closeDetail() {
    setSelectedId(null);
    previousFocus.current?.focus();
  }
  const activeDataset = catalog.datasets.find((d) => d.id === selectedDataset)!;
  const currentGuide = catalog.guides[guidePage];

  function PaperInspector({ paper }: { paper: Paper }) {
    const related = catalog.papers.filter(
      (p) => p.cluster === paper.cluster && p.id !== paper.id,
    );
    const pageCount =
      detailTab === "sources"
        ? Math.ceil(paper.sources.length / 2)
        : detailTab === "evidence"
          ? selectionRelations.length
          : detailTab === "question"
            ? Math.ceil(related.length / 6)
            : 1;
    return (
      <>
        <div className="inspector-heading">
          <BookOpen size={18} />
          <strong>论文研读</strong>
          <button
            className="icon-button"
            aria-label="关闭论文详情"
            onClick={closeDetail}
          >
            <X size={18} />
          </button>
        </div>
        <div
          className="paper-id"
          style={{ background: selectedCluster?.color }}
        >
          <div className="paper-id-top">
            <span>{selectedCluster?.name}</span>
            <b>{paper.year}</b>
          </div>
          <h2>{paper.shortTitle}</h2>
          <p className="paper-full-title" title={paper.title}>
            {paper.title}
          </p>
          <small>{paper.venue}</small>
        </div>
        <div className="detail-tabs" role="tablist" aria-label="论文详情内容">
          {detailTabs.map(([id, name]) => (
            <button
              key={id}
              role="tab"
              aria-selected={detailTab === id}
              aria-controls="detail-content"
              className={detailTab === id ? "active" : ""}
              onClick={() => setDetailTab(id)}
            >
              {name}
            </button>
          ))}
        </div>
        <div
          id="detail-content"
          className="detail-content"
          role="tabpanel"
          aria-label={detailTabs.find(([id]) => id === detailTab)?.[1]}
        >
          {detailTab === "idea" && (
            <>
              <div className="detail-block">
                <h3>
                  <Lightbulb size={14} />
                  创新抓手
                </h3>
                <strong className="mechanism-label">{paper.mechanism}</strong>
                <p>{paper.summary}</p>
              </div>
              <div className="detail-block">
                <h3>为什么值得读</h3>
                <p>{paper.takeaway}</p>
              </div>
              <div className="task-tags">
                {paper.tasks.map((t) => (
                  <button key={t} onClick={() => setTask(t)}>
                    {t}
                  </button>
                ))}
              </div>
            </>
          )}
          {detailTab === "question" && (
            <>
              <div className="detail-block">
                <h3>带着这个问题读</h3>
                <p className="reading-question">
                  {paper.limitation.replace("阅读关注：", "")}
                </p>
              </div>
              <div className="detail-block">
                <h3>同一创新方向</h3>
                <p>{selectedCluster?.question}</p>
                <div className="related-papers">
                  {related
                    .slice(detailPage * 6, detailPage * 6 + 6)
                    .map((p) => (
                      <button key={p.id} onClick={() => selectPaper(p.id)}>
                        {p.shortTitle}
                        <ArrowUpRight size={12} />
                      </button>
                    ))}
                </div>
              </div>
            </>
          )}
          {detailTab === "evidence" && (
            <>
              <div className="detail-block">
                <h3>数据与可追溯关系</h3>
              </div>
              {selectionRelations.slice(detailPage, detailPage + 1).map((r) => {
                const otherId = r.source === paper.id ? r.target : r.source;
                const other = catalog.papers.find((p) => p.id === otherId);
                return (
                  <div className="evidence-note" key={r.id}>
                    <b>
                      {other?.shortTitle ||
                        catalog.datasets.find((d) => d.id === otherId)?.name}
                    </b>
                    <span className="relation-type">
                      {r.type === "introduces"
                        ? "提出资源"
                        : r.type === "uses"
                          ? "使用资源"
                          : "方法关联"}
                    </span>
                    <p>{r.evidence.note}</p>
                    <ResourceLink href={r.evidence.url}>查看依据</ResourceLink>
                  </div>
                );
              })}
              {!selectionRelations.length && (
                <p className="small-notice">暂无额外关系</p>
              )}
            </>
          )}
          {detailTab === "sources" && (
            <>
              <p className="small-notice">核验日期 {paper.verifiedAt}</p>
              {paper.sources
                .slice(detailPage * 2, detailPage * 2 + 2)
                .map((s, i) => (
                  <div className="source-note" key={s.url + i}>
                    <p>{s.note}</p>
                    <ResourceLink href={s.url}>
                      {new URL(s.url).hostname}
                    </ResourceLink>
                  </div>
                ))}
              <button
                className="text-action"
                onClick={() =>
                  copy(
                    `${paper.title}. ${paper.venue}, ${paper.year}. ${paper.links.paper}`,
                    "论文引用已复制",
                  )
                }
              >
                <Copy size={13} />
                复制引用
              </button>
            </>
          )}
          {pageCount > 1 && (
            <Pager
              page={detailPage}
              total={pageCount}
              onChange={setDetailPage}
              label="详情分页"
            />
          )}
        </div>
        <div className="inspector-bottom">
          <div className="detail-resources">
            <ResourceLink href={paper.links.paper}>
              <FileText size={14} />
              论文
            </ResourceLink>
            {paper.links.code ? (
              <ResourceLink href={paper.links.code}>
                <Code size={14} />
                代码
              </ResourceLink>
            ) : (
              paper.links.project && (
                <ResourceLink href={paper.links.project}>项目页</ResourceLink>
              )
            )}
          </div>
          <div className="paper-navigation">
            <button
              disabled={selectedIndex <= 0}
              onClick={() => selectPaper(visiblePapers[selectedIndex - 1].id)}
              aria-label="上一篇论文"
            >
              <ChevronLeft size={15} />
            </button>
            <span>
              {selectedIndex + 1} / {visiblePapers.length} 篇
            </span>
            <button
              disabled={
                selectedIndex < 0 || selectedIndex >= visiblePapers.length - 1
              }
              onClick={() => selectPaper(visiblePapers[selectedIndex + 1].id)}
              aria-label="下一篇论文"
            >
              <ChevronRight size={15} />
            </button>
          </div>
        </div>
      </>
    );
  }
  function DatasetInspector({ dataset }: { dataset: Dataset }) {
    const related = catalog.papers.filter((p) =>
      p.datasetIds.includes(dataset.id),
    );
    return (
      <>
        <div className="inspector-heading">
          <Database size={18} />
          <strong>评测语境</strong>
          <button
            className="icon-button"
            onClick={() => setDatasetReading(false)}
            aria-label="返回数据资源"
          >
            <X size={18} />
          </button>
        </div>
        <a
          className="dataset-detail-image"
          href={dataset.thumbnail.sourceUrl}
          target="_blank"
          rel="noreferrer"
        >
          <img
            src={datasetImageUrl(dataset.thumbnail.src)}
            alt={dataset.thumbnail.alt}
          />
          <span>
            {dataset.thumbnail.credit}
            <ArrowUpRight size={12} />
          </span>
        </a>
        <div className="paper-id dataset-id">
          <span className="overline">
            {dataset.venue} · {dataset.year}
          </span>
          <h2>{dataset.name}</h2>
          <p>{dataset.modalities.join(" · ")}</p>
        </div>
        <div className="detail-content">
          <div className="detail-block">
            <h3>标注提供什么</h3>
            <div className="task-tags">
              {dataset.annotations.map((a) => (
                <span key={a}>{a}</span>
              ))}
            </div>
          </div>
          <div className="detail-block">
            <h3>评测协议</h3>
            <p>{dataset.protocol}</p>
          </div>
          <div className="detail-block">
            <h3>关联论文</h3>
            <div className="related-papers">
              {related
                .slice(datasetRelatedPage * 6, datasetRelatedPage * 6 + 6)
                .map((p) => (
                  <button key={p.id} onClick={() => selectPaper(p.id)}>
                    {p.shortTitle}
                    <ArrowUpRight size={12} />
                  </button>
                ))}
            </div>
          </div>
        </div>
        {related.length > 6 && (
          <Pager
            page={datasetRelatedPage}
            total={Math.ceil(related.length / 6)}
            onChange={setDatasetRelatedPage}
            label="数据集关联论文"
          />
        )}
        <div className="inspector-bottom detail-resources">
          <ResourceLink href={dataset.links.website}>官方资源</ResourceLink>
          <button onClick={() => showDatasetPapers(dataset.id)}>
            筛选论文
            <ArrowRight size={14} />
          </button>
        </div>
      </>
    );
  }

  return (
    <div
      className={`app-shell ${view === "map" && layout === "map" ? "is-metro" : ""}`}
    >
      <a className="skip-link" href="#main">
        跳到主要内容
      </a>
      <header className="masthead">
        <a className="brand" href="./" aria-label="视频异常理解创新地图首页">
          <Art cluster="evidence" />
          <span>
            <strong>
              VAU <i>/</i> IDEA ATLAS
            </strong>
            <small>视频异常理解 · 创新路线图</small>
          </span>
        </a>
        <div className="masthead-note">
          <b>{coreCount}</b>
          <span>主线论文</span>
          <i /> <b>{catalog.clusters.length}</b>
          <span>创新方向</span>
        </div>
        <nav aria-label="主导航">
          {(
            [
              ["map", "地图", Map],
              ["papers", "论文", FileText],
              ["guides", "路线", BookOpen],
              ["datasets", "数据", Database],
            ] as const
          ).map(([id, name, Icon]) => (
            <button
              key={id}
              className={
                (
                  id === "papers"
                    ? view === "map" && layout === "list"
                    : id === "map"
                      ? view === "map" && layout !== "list"
                      : view === id
                )
                  ? "active"
                  : ""
              }
              onClick={() => {
                if (id === "papers") {
                  changeView("map");
                  setLayout("list");
                } else {
                  changeView(id);
                  if (id === "map") setLayout("map");
                }
              }}
              aria-pressed={
                id === "papers"
                  ? view === "map" && layout === "list"
                  : id === "map"
                    ? view === "map" && layout !== "list"
                    : view === id
              }
            >
              <Icon size={15} />
              <span>{name}</span>
            </button>
          ))}
        </nav>
        <a
          className="repo-link"
          href={REPO}
          target="_blank"
          rel="noreferrer"
          aria-label="GitHub 项目"
        >
          <Code size={19} />
          <span>GitHub</span>
          <ArrowUpRight size={13} />
        </a>
      </header>
      <div className="command-bar">
        {view === "map" && (
          <button
            className="mobile-filter-toggle"
            aria-label="展开筛选"
            aria-expanded={showFilters}
            onClick={() => setShowFilters(!showFilters)}
          >
            <SlidersHorizontal size={17} />
          </button>
        )}
        <label className="search-field">
          <Search size={16} />
          <input
            id="paper-search"
            aria-label={
              view === "datasets" ? "搜索数据集" : "搜索论文、机制或关键词"
            }
            placeholder={
              view === "datasets"
                ? "找数据集、任务、标注…"
                : "找论文、创新机制、研究问题…"
            }
            value={view === "datasets" ? datasetQuery : query}
            onChange={(e) => {
              if (view === "datasets") setDatasetQuery(e.target.value);
              else {
                setQuery(e.target.value);
                setView("map");
              }
            }}
          />
          <kbd>/</kbd>
          {(view === "datasets" ? datasetQuery : query) && (
            <button
              onClick={() =>
                view === "datasets" ? setDatasetQuery("") : setQuery("")
              }
              aria-label="清空搜索"
            >
              <X size={14} />
            </button>
          )}
        </label>
        {view === "map" && (
          <div
            className={`header-filters ${showFilters ? "is-open" : ""}`}
            aria-label="论文搜索条件"
          >
            <select
              aria-label="发表出处"
              value={venue}
              title={venue === "all" ? "发表出处" : venue}
              onChange={(e) => setVenue(e.target.value)}
            >
              <option value="all">全部出处</option>
              {venues.map((v) => (
                <option key={v}>{v}</option>
              ))}
            </select>
            <select
              aria-label="发表年份"
              value={year}
              onChange={(e) => setYear(e.target.value)}
            >
              <option value="all">全部年份</option>
              {years.map((y) => (
                <option key={y}>{y}</option>
              ))}
            </select>
            {hasFilters && (
              <button
                onClick={resetFilters}
                aria-label="清除所有筛选"
                title="清除筛选"
              >
                <X size={13} />
                清除
              </button>
            )}
          </div>
        )}
        <div className="command-actions">
          <button
            title="分享当前视图"
            aria-label="分享当前视图"
            onClick={() => copy(window.location.href, "视图链接已复制")}
          >
            <Share2 size={15} />
            <span>分享</span>
          </button>
          <button
            title="导出当前文献 JSON"
            aria-label="导出当前文献 JSON"
            onClick={exportData}
          >
            <ArrowDownToLine size={15} />
            <span>数据</span>
          </button>
          <a
            href={`${REPO}/blob/main/catalog.md`}
            target="_blank"
            rel="noreferrer"
          >
            <FileText size={15} />
            <span>文档</span>
          </a>
        </div>
      </div>
      <main
        id="main"
        className={`desk ${view === "map" && selected ? "has-selection" : view === "datasets" && datasetReading ? "has-dataset-selection" : ""} view-${view}`}
      >
        <section className="workspace" aria-label="研究工作区">
          <div className="workspace-toolbar">
            {view === "map" ? (
              <div className="layout-tabs" aria-label="论文浏览方式">
                {(
                  [
                    ["map", "研究线路", Map],
                    ["timeline", "时间线", GitBranch],
                    ["list", "索引", List],
                  ] as const
                ).map(([id, name, Icon]) => (
                  <button
                    key={id}
                    className={layout === id ? "active" : ""}
                    onClick={() => setLayout(id)}
                    aria-pressed={layout === id}
                  >
                    <Icon size={13} />
                    {name}
                  </button>
                ))}
              </div>
            ) : (
              <h1>{view === "guides" ? "阅读路线" : "数据与评测"}</h1>
            )}
            <span className="workspace-count" aria-live="polite">
              {view === "map"
                ? layout === "map"
                  ? `${visiblePapers.length} / ${catalog.papers.length} 篇论文`
                  : `${visiblePapers.length} 篇可见`
                : view === "guides"
                  ? "3 条编辑路线"
                  : `${catalog.datasets.length} 个数据资源`}
            </span>
          </div>
          <div className="workspace-body">
            {view === "map" && layout !== "list" && (
              <ResearchMap
                papers={visiblePapers}
                allPapers={catalog.papers}
                clusters={catalog.clusters}
                selectedId={selectedId}
                onSelect={selectPaper}
                onCluster={filterCluster}
                layout={layout}
                activeCluster={cluster}
                onReset={resetFilters}
              />
            )}
            {view === "map" && layout === "list" && (
              <section className="paper-overview" aria-label="全部论文卡片">
                <div className="paper-grid">
                  {[...visiblePapers]
                    .sort(
                      (a, b) =>
                        b.year - a.year ||
                        a.shortTitle.localeCompare(b.shortTitle),
                    )
                    .map((p) => (
                      <button
                        key={p.id}
                        className={`paper-chip ${selectedId === p.id ? "selected" : ""}`}
                        onClick={() => selectPaper(p.id)}
                        style={
                          {
                            "--paper-color": catalog.clusters.find(
                              (c) => c.id === p.cluster,
                            )?.color,
                          } as React.CSSProperties
                        }
                        title={`${p.title} — ${p.venue}, ${p.year}`}
                      >
                        <strong>{p.shortTitle}</strong>
                        <span>
                          <span>
                            {p.venue
                              .replace(/Datasets and Benchmarks/g, "D&B")
                              .replace(/ Workshops?/g, "W")}
                          </span>
                          <b>{p.year}</b>
                        </span>
                      </button>
                    ))}
                </div>
                {!visiblePapers.length && (
                  <div className="empty-state">
                    <Search size={24} />
                    <b>没有匹配的论文</b>
                    <button onClick={resetFilters}>清除筛选</button>
                  </div>
                )}
              </section>
            )}
            {view === "guides" && (
              <div className="guides-view">
                <div
                  className="guide-selector"
                  role="tablist"
                  aria-label="阅读路线"
                >
                  {catalog.guides.map((g, i) => (
                    <button
                      key={g.id}
                      role="tab"
                      aria-selected={guidePage === i}
                      className={guidePage === i ? "active" : ""}
                      onClick={() => {
                        setGuidePage(i);
                        setGuideStepPage(0);
                      }}
                    >
                      0{i + 1}
                      <span>{g.title}</span>
                    </button>
                  ))}
                </div>
                <div className="guide-intro">
                  <h2>{currentGuide.title}</h2>
                  <p>{currentGuide.description}</p>
                </div>
                <ol className="guide-steps">
                  {currentGuide.steps
                    .slice(guideStepPage * 3, guideStepPage * 3 + 3)
                    .map((step, i) => {
                      const p = catalog.papers.find(
                        (p) => p.id === step.paperId,
                      )!;
                      return (
                        <li key={step.paperId}>
                          <span>0{guideStepPage * 3 + i + 1}</span>
                          <button onClick={() => selectPaper(p.id)}>
                            <strong>
                              {p.shortTitle}
                              <small>
                                {p.venue} · {p.year}
                              </small>
                              <ArrowUpRight size={16} />
                            </strong>
                            <p>{step.note}</p>
                          </button>
                        </li>
                      );
                    })}
                </ol>
                <Pager
                  page={guideStepPage}
                  total={Math.ceil(currentGuide.steps.length / 3)}
                  onChange={setGuideStepPage}
                  label="阅读步骤"
                />
              </div>
            )}
            {view === "datasets" && (
              <DatasetGallery
                datasets={catalog.datasets}
                query={datasetQuery}
                selectedId={datasetReading ? selectedDataset : null}
                onSelect={(id) => {
                  setSelectedDataset(id);
                  setDatasetReading(true);
                }}
              />
            )}
          </div>
        </section>
        {((view === "map" && selected) ||
          (view === "datasets" && datasetReading)) && (
          <aside
            className={`inspector ${view === "map" ? "is-reading" : ""}`}
            aria-label={
              view === "datasets"
                ? `${activeDataset.name} 数据集详情`
                : `${selected?.shortTitle} 论文详情`
            }
          >
            {view === "datasets" ? (
              <DatasetInspector dataset={activeDataset} />
            ) : (
              selected && <PaperInspector paper={selected} />
            )}
          </aside>
        )}
      </main>
      <footer className="statusbar">
        <span>
          <i />
          视频异常理解主线
        </span>
        <span>核验 {catalog.updatedAt}</span>
        <a
          href={`${REPO}/blob/main/CONTRIBUTING.md`}
          target="_blank"
          rel="noreferrer"
        >
          补充 / 纠错
          <ArrowUpRight size={11} />
        </a>
      </footer>
      <div className={`toast ${notice ? "visible" : ""}`} role="status">
        {notice && (
          <>
            <Check size={16} />
            {notice}
          </>
        )}
      </div>
    </div>
  );
}
