import { useEffect, useId, useMemo, useRef, useState } from "react";
import type { PointerEvent as ReactPointerEvent } from "react";
import type { Cluster, Paper, Relation } from "../types";
import ideaStrip from "../assets/idea-strip.png";
import "./research-map.css";

interface Props {
  papers: Paper[];
  allPapers?: Paper[];
  clusters: Cluster[];
  relations?: Relation[];
  selectedId: string | null;
  onSelect: (id: string) => void;
  onCluster: (id: string) => void;
  layout: "map" | "timeline";
  activeCluster: string;
  onReset: () => void;
}
type Point = { x: number; y: number };
type Box = Point & { width: number; height: number };
const INK = "#20201e";
const CREAM = "#fff8e9";
const ORDER = [
  "alignment",
  "explanation",
  "understanding",
  "evidence",
  "reasoning",
];
const COLORS: Record<string, string> = {
  alignment: "#badff5",
  explanation: "#ffc3b5",
  reasoning: "#c5e2c1",
  evidence: "#ffdf70",
  understanding: "#d9cef1",
};
const QUESTIONS: Record<string, string> = {
  alignment: "视觉与语言，如何对齐异常？",
  explanation: "把异常判断变成可检查的判据",
  understanding: "长视频，如何保留关键上下文？",
  evidence: "证据不够，下一步该看哪里？",
  reasoning: "如何构造、验证并修正推理？",
};
const clamp = (n: number, min: number, max: number) =>
  Math.min(max, Math.max(min, n));

export default function ResearchMap({
  papers,
  allPapers,
  clusters,
  selectedId,
  onSelect,
  onCluster,
  layout,
  activeCluster,
  onReset,
}: Props) {
  const svgRef = useRef<SVGSVGElement>(null);
  const sectionRef = useRef<HTMLElement>(null);
  const drag = useRef<{
    id: number;
    x: number;
    y: number;
    origin: Point;
    moved: boolean;
  } | null>(null);
  const suppressClick = useRef(false);
  const [pan, setPan] = useState<Point>({ x: 0, y: 0 });
  const [zoom, setZoom] = useState(1);
  const [dragging, setDragging] = useState(false);
  const [exporting, setExporting] = useState(false);
  const [exportFailed, setExportFailed] = useState(false);
  const [mobile, setMobile] = useState(() => window.innerWidth < 800);
  const [page, setPage] = useState(0);
  const [yearPage, setYearPage] = useState(0);
  const [paperPage, setPaperPage] = useState(0);
  const id = useId().replace(/[^a-zA-Z0-9_-]/g, "");
  const patternId = `comic-dots-${id}`;
  const arrowId = `comic-arrow-${id}`;
  const artId = `comic-art-${id}`;
  const sourcePapers = allPapers || papers;
  const orderedClusters = useMemo(
    () =>
      [...clusters].sort((a, b) => ORDER.indexOf(a.id) - ORDER.indexOf(b.id)),
    [clusters],
  );
  useEffect(() => {
    const section = sectionRef.current;
    if (!section) return;
    const update = (width: number, height: number) =>
      setMobile(width < 800 || height - 44 < 480);
    const bounds = section.getBoundingClientRect();
    update(bounds.width, bounds.height);
    const observer = new ResizeObserver((entries) => {
      const entry = entries[0];
      if (entry) update(entry.contentRect.width, entry.contentRect.height);
    });
    observer.observe(section);
    return () => observer.disconnect();
  }, []);
  useEffect(() => {
    const next = orderedClusters.findIndex(
      (cluster) => cluster.id === activeCluster,
    );
    if (next >= 0) setPage(next);
  }, [activeCluster, orderedClusters]);
  useEffect(() => {
    setPan({ x: 0, y: 0 });
    setZoom(1);
  }, [layout, mobile, page, yearPage, paperPage]);

  const paperGroups = useMemo(() => {
    const groups = new Map<string, Paper[]>();
    for (const paper of papers) {
      const key = layout === "map" ? paper.cluster : String(paper.year);
      const members = groups.get(key) || [];
      members.push(paper);
      groups.set(key, members);
    }
    for (const members of groups.values())
      members.sort((a, b) =>
        layout === "map"
          ? a.year - b.year || a.id.localeCompare(b.id)
          : ORDER.indexOf(a.cluster) - ORDER.indexOf(b.cluster) ||
            a.id.localeCompare(b.id),
      );
    return groups;
  }, [papers, layout]);
  const papersPerPage = layout === "map" ? 4 : 5;
  const groupPageCount = (key: string) =>
    Math.max(1, Math.ceil((paperGroups.get(key)?.length || 0) / papersPerPage));
  const groupPage = (key: string) =>
    Math.min(paperPage, groupPageCount(key) - 1);
  const pageMembers = (key: string) => {
    const start = groupPage(key) * papersPerPage;
    return (paperGroups.get(key) || []).slice(start, start + papersPerPage);
  };

  const geometry = useMemo(() => {
    const width = mobile ? 420 : 1200;
    const height = mobile ? (layout === "timeline" ? 600 : 440) : 720;
    const panels = new Map<string, Box>();
    const nodes = new Map<string, Box>();
    const years = [...new Set(sourcePapers.map((paper) => paper.year))].sort(
      (a, b) => a - b,
    );
    const minYear = years[0] || new Date().getFullYear();
    const maxYear = years[years.length - 1] || minYear;
    const yearWidth = mobile ? 394 : 1160 / (maxYear - minYear + 1);
    const yearColumns: { year: number; x: number; width: number }[] = [];
    for (let year = minYear; year <= maxYear; year++)
      yearColumns.push({
        year,
        x: mobile ? 20 : 20 + (year - minYear) * yearWidth,
        width: yearWidth - 14,
      });
    if (layout === "timeline") {
      yearColumns.forEach((column) => {
        const members = pageMembers(String(column.year));
        const cardHeight = 77;
        members.forEach((paper, index) =>
          nodes.set(paper.id, {
            x: column.x + 4,
            y: 93 + index * (cardHeight + 9),
            width: column.width - 8,
            height: cardHeight,
          }),
        );
      });
    } else {
      orderedClusters.forEach((cluster, index) => {
        const panel: Box = mobile
          ? { x: 8, y: 8, width: 404, height: 415 }
          : index < 3
            ? { x: 10 + index * 398, y: 12, width: 384, height: 340 }
            : { x: index === 3 ? 610 : 10, y: 382, width: 584, height: 275 };
        panels.set(cluster.id, panel);
        const members = pageMembers(cluster.id);
        const cardWidth = mobile ? panel.width - 32 : (panel.width - 45) / 2;
        const cardHeight = mobile ? 68 : index < 3 ? 102 : 75;
        const gap = mobile ? 10 : 12;
        members.forEach((paper, position) =>
          nodes.set(paper.id, {
            x: panel.x + 16 + (mobile ? 0 : position % 2) * (cardWidth + 12),
            y:
              panel.y +
              105 +
              (mobile ? position : Math.floor(position / 2)) *
                (cardHeight + gap),
            width: cardWidth,
            height: cardHeight,
          }),
        );
      });
    }
    return { width, height, panels, nodes, yearColumns };
  }, [sourcePapers, orderedClusters, mobile, layout, paperGroups, paperPage]);
  const mobilePage = orderedClusters[page];
  const mobileYear = geometry.yearColumns[yearPage]?.year;
  const displayedPapers = papers.filter(
    (paper) =>
      geometry.nodes.has(paper.id) &&
      (!mobile ||
        (layout === "map"
          ? paper.cluster === mobilePage?.id
          : paper.year === mobileYear)),
  );
  const visibleGroupKeys = mobile
    ? [layout === "map" ? mobilePage?.id || "" : String(mobileYear)]
    : [...paperGroups.keys()];
  const paperPageCount = Math.max(1, ...visibleGroupKeys.map(groupPageCount));
  const currentPaperPage = Math.min(paperPage, paperPageCount - 1);
  useEffect(() => {
    setPaperPage((current) => Math.min(current, paperPageCount - 1));
  }, [paperPageCount]);
  useEffect(() => {
    const selected = papers.find((paper) => paper.id === selectedId);
    if (!selected) return;
    const key = layout === "map" ? selected.cluster : String(selected.year);
    const index =
      paperGroups.get(key)?.findIndex((paper) => paper.id === selectedId) ?? -1;
    if (index >= 0) setPaperPage(Math.floor(index / papersPerPage));
  }, [selectedId, paperGroups, papersPerPage, papers, layout]);
  const displayedClusters =
    mobile && layout === "map"
      ? orderedClusters.filter((_, index) => index === page)
      : orderedClusters;
  useEffect(() => {
    if (!mobile || !papers.length) return;
    if (layout === "timeline") {
      setYearPage((current) =>
        papers.some(
          (paper) => paper.year === geometry.yearColumns[current]?.year,
        )
          ? current
          : Math.max(
              0,
              geometry.yearColumns.findIndex((column) =>
                papers.some((paper) => paper.year === column.year),
              ),
            ),
      );
    } else if (activeCluster === "all") {
      setPage((current) =>
        papers.some((paper) => paper.cluster === orderedClusters[current]?.id)
          ? current
          : Math.max(
              0,
              orderedClusters.findIndex((cluster) =>
                papers.some((paper) => paper.cluster === cluster.id),
              ),
            ),
      );
    }
  }, [papers, layout, mobile, activeCluster, orderedClusters, sourcePapers]);
  useEffect(() => {
    const paper = sourcePapers.find((item) => item.id === selectedId);
    if (!mobile || !paper) return;
    setPage(
      Math.max(
        0,
        orderedClusters.findIndex((cluster) => cluster.id === paper.cluster),
      ),
    );
    setYearPage(
      Math.max(
        0,
        geometry.yearColumns.findIndex((column) => column.year === paper.year),
      ),
    );
  }, [selectedId, mobile, sourcePapers, orderedClusters]);
  const changePage = (direction: number) => {
    if (layout === "timeline") {
      setYearPage((value) =>
        clamp(value + direction, 0, geometry.yearColumns.length - 1),
      );
      return;
    }
    const next = clamp(page + direction, 0, orderedClusters.length - 1);
    setPage(next);
    if (activeCluster !== "all" && orderedClusters[next])
      onCluster(orderedClusters[next].id);
  };
  const resetView = () => {
    setPan({ x: 0, y: 0 });
    setZoom(1);
  };
  const changeZoom = (amount: number) =>
    setZoom((value) => clamp(Number((value + amount).toFixed(2)), 0.7, 2.5));
  const onPointerDown = (event: ReactPointerEvent<SVGSVGElement>) => {
    if (event.button !== 0) return;
    drag.current = {
      id: event.pointerId,
      x: event.clientX,
      y: event.clientY,
      origin: pan,
      moved: false,
    };
    suppressClick.current = false;
  };
  const onPointerMove = (event: ReactPointerEvent<SVGSVGElement>) => {
    const start = drag.current;
    if (!start || start.id !== event.pointerId) return;
    const dx = event.clientX - start.x;
    const dy = event.clientY - start.y;
    if (Math.abs(dx) + Math.abs(dy) < 5 && !start.moved) return;
    start.moved = true;
    suppressClick.current = true;
    setDragging(true);
    if (!event.currentTarget.hasPointerCapture(event.pointerId))
      event.currentTarget.setPointerCapture(event.pointerId);
    const bounds = event.currentTarget.getBoundingClientRect();
    const scale = Math.min(
      bounds.width / geometry.width,
      bounds.height / geometry.height,
    );
    setPan({ x: start.origin.x + dx / scale, y: start.origin.y + dy / scale });
  };
  const endDrag = (event: ReactPointerEvent<SVGSVGElement>) => {
    if (event.currentTarget.hasPointerCapture(event.pointerId))
      event.currentTarget.releasePointerCapture(event.pointerId);
    drag.current = null;
    setDragging(false);
  };
  const exportSvg = async () => {
    if (!svgRef.current || exporting) return;
    setExporting(true);
    setExportFailed(false);
    try {
      const clone = svgRef.current.cloneNode(true) as SVGSVGElement;
      clone.setAttribute("xmlns", "http://www.w3.org/2000/svg");
      clone.setAttribute("width", String(geometry.width));
      clone.setAttribute("height", String(geometry.height));
      clone.querySelector("[data-map-content]")?.removeAttribute("transform");
      const response = await fetch(ideaStrip);
      if (!response.ok) throw new Error("Artwork unavailable");
      const art = await response.blob();
      const encodedArt = await new Promise<string>((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => resolve(String(reader.result));
        reader.onerror = () => reject(reader.error);
        reader.readAsDataURL(art);
      });
      clone.querySelector("image")?.setAttribute("href", encodedArt);
      const xml = new XMLSerializer().serializeToString(clone);
      const url = URL.createObjectURL(
        new Blob([xml], { type: "image/svg+xml;charset=utf-8" }),
      );
      const anchor = document.createElement("a");
      anchor.href = url;
      anchor.download = `vau-idea-atlas-${layout}${mobile ? `-${layout === "map" ? mobilePage?.id : mobileYear}` : ""}.svg`;
      anchor.click();
      window.setTimeout(() => URL.revokeObjectURL(url), 1000);
    } catch {
      setExportFailed(true);
    } finally {
      setExporting(false);
    }
  };

  return (
    <section
      ref={sectionRef}
      className={`research-map comic-map${mobile ? " is-paged" : ""}${dragging ? " is-dragging" : ""}`}
      aria-label={layout === "map" ? "创新机制漫画地图" : "论文发表时间排列"}
    >
      <svg
        ref={svgRef}
        className="research-map-canvas"
        viewBox={`0 0 ${geometry.width} ${geometry.height}`}
        aria-label="选择论文卡阅读机制与证据；箭头为编辑阅读顺序，不表示论文继承"
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={endDrag}
        onPointerCancel={endDrag}
        style={{
          fontFamily:
            '-apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif',
        }}
      >
        <title>
          视频异常理解 ·{" "}
          {layout === "map" ? "创新机制漫画地图" : "论文发表时间排列"}
        </title>
        <desc>
          论文按照创新机制归类。分镜间箭头只表示编辑建议的问题进阶顺序，不表示论文引用、继承或经验证的因果关系。
        </desc>
        <defs>
          <image id={artId} href={ideaStrip} width="2172" height="724" />
          <pattern
            id={patternId}
            width="8"
            height="8"
            patternUnits="userSpaceOnUse"
          >
            <circle cx="2" cy="2" r="0.7" fill={INK} opacity="0.17" />
          </pattern>
          <marker
            id={arrowId}
            markerWidth="8"
            markerHeight="8"
            refX="6"
            refY="4"
            orient="auto"
          >
            <path d="M1 1l5 3-5 3" fill="none" stroke={INK} strokeWidth="1.5" />
          </marker>
        </defs>
        <rect width={geometry.width} height={geometry.height} fill={CREAM} />
        <g
          data-map-content="true"
          transform={`translate(${pan.x + geometry.width / 2} ${pan.y + geometry.height / 2}) scale(${zoom}) translate(${-geometry.width / 2} ${-geometry.height / 2})`}
        >
          {layout === "map" ? (
            <>
              {displayedClusters.map((cluster) => {
                const panel = geometry.panels.get(cluster.id)!;
                const index = orderedClusters.findIndex(
                  (item) => item.id === cluster.id,
                );
                const count = papers.filter(
                  (paper) => paper.cluster === cluster.id,
                ).length;
                const active = activeCluster === cluster.id;
                return (
                  <g key={cluster.id}>
                    <rect
                      x={panel.x + 4}
                      y={panel.y + 5}
                      width={panel.width}
                      height={panel.height}
                      rx="2"
                      fill={INK}
                    />
                    <rect
                      x={panel.x}
                      y={panel.y}
                      width={panel.width}
                      height={panel.height}
                      rx="2"
                      fill={COLORS[cluster.id] || "#badff5"}
                      stroke={INK}
                      strokeWidth={active ? 5 : 3}
                    />
                    <path
                      d={`M${panel.x + panel.width - 83} ${panel.y + 1}h82v${panel.height - 2}h-82z`}
                      fill={`url(#${patternId})`}
                    />
                    <g
                      className="comic-cluster"
                      role="button"
                      tabIndex={0}
                      aria-label={`筛选${cluster.name}，当前${count}篇论文`}
                      aria-pressed={active}
                      onClick={() => {
                        if (!suppressClick.current) onCluster(cluster.id);
                      }}
                      onKeyDown={(event) => {
                        if (event.key === "Enter" || event.key === " ") {
                          event.preventDefault();
                          onCluster(cluster.id);
                        }
                      }}
                    >
                      <rect
                        className="comic-cluster-hit"
                        x={panel.x + 6}
                        y={panel.y + 7}
                        width={panel.width - 70}
                        height="81"
                        fill="transparent"
                        rx="2"
                      />
                      <text
                        x={panel.x + 14}
                        y={panel.y + 52}
                        fill={INK}
                        fontFamily="Impact, 'Arial Black', sans-serif"
                        fontSize="51"
                        fontStyle="italic"
                        fontWeight="900"
                      >
                        0{index + 1}
                      </text>
                      <text
                        x={panel.x + 78}
                        y={panel.y + 34}
                        fill={INK}
                        fontSize={mobile ? 22 : 21}
                        fontWeight="900"
                        letterSpacing="-0.8"
                      >
                        {cluster.name}
                      </text>
                      <text
                        x={panel.x + 79}
                        y={panel.y + 59}
                        fill={INK}
                        fontSize={mobile ? 14.5 : 14}
                        fontWeight="500"
                      >
                        {QUESTIONS[cluster.id]}
                      </text>
                      <path
                        d={`M${panel.x + 17} ${panel.y + 80}h${panel.width - 110}`}
                        stroke={INK}
                        strokeWidth="2"
                      />
                    </g>
                    <svg
                      x={panel.x + panel.width - 72}
                      y={panel.y + 8}
                      width="64"
                      height="87"
                      viewBox="0 0 434.4 590"
                      aria-hidden="true"
                    >
                      <use
                        href={`#${artId}`}
                        x={
                          -[
                            "alignment",
                            "explanation",
                            "reasoning",
                            "evidence",
                            "understanding",
                          ].indexOf(cluster.id) * 434.4
                        }
                        y="-75"
                      />
                    </svg>
                    <rect
                      x={panel.x + panel.width - 72}
                      y={panel.y + 8}
                      width="64"
                      height="87"
                      fill="none"
                      stroke={INK}
                      strokeWidth="1.5"
                      aria-hidden="true"
                    />
                    {groupPageCount(cluster.id) > 1 && (
                      <text
                        x={panel.x + 18}
                        y={panel.y + 97}
                        fill={INK}
                        fontSize="12"
                        fontWeight="650"
                      >
                        {`${groupPage(cluster.id) + 1} / ${groupPageCount(cluster.id)} 页 · ${count} 篇`}
                      </text>
                    )}
                    {count === 0 && (
                      <text
                        x={panel.x + panel.width / 2}
                        y={panel.y + 176}
                        textAnchor="middle"
                        fill={INK}
                        fontSize="17"
                      >
                        当前筛选下暂无论文
                      </text>
                    )}
                    {cluster.id === "evidence" &&
                      !mobile &&
                      pageMembers(cluster.id).length <= 2 && (
                        <g
                          transform={`translate(${panel.x + 20} ${panel.y + 232})`}
                        >
                          <path
                            d="M0-14h363l10 12-10 12H0z"
                            fill={CREAM}
                            stroke={INK}
                            strokeWidth="1.6"
                          />
                          <text
                            x="13"
                            y="4"
                            fill={INK}
                            fontSize="15"
                            fontWeight="600"
                          >
                            提出问题 → 调用工具 → 回看证据
                          </text>
                          <text
                            x="398"
                            y="4"
                            fill={INK}
                            fontSize="13"
                            fontWeight="600"
                          >
                            机制导览 ↗
                          </text>
                        </g>
                      )}
                  </g>
                );
              })}
              {!mobile && (
                <g
                  aria-label="编辑阅读路径，非论文继承关系"
                  stroke={INK}
                  strokeWidth="2.4"
                  fill="none"
                  markerEnd={`url(#${arrowId})`}
                >
                  <path d="M384 165h28" />
                  <path d="M782 165h28" />
                  <path d="M1118 354v24" />
                  <path d="M613 556h-27" />
                </g>
              )}
            </>
          ) : (
            <>
              {geometry.yearColumns
                .filter((_, index) => !mobile || index === yearPage)
                .map((column) => (
                  <g key={column.year}>
                    <rect
                      x={column.x}
                      y="14"
                      width={column.width}
                      height={mobile ? 570 : 658}
                      fill="#fffdf6"
                      stroke={INK}
                      strokeWidth="2"
                    />
                    <path
                      d={`M${column.x} 14h${column.width}v60h-${column.width}z`}
                      fill="#ffdf70"
                      stroke={INK}
                      strokeWidth="2"
                    />
                    <text
                      x={column.x + 15}
                      y="57"
                      fill={INK}
                      fontFamily="Impact, 'Arial Black', sans-serif"
                      fontSize="42"
                      fontWeight="900"
                    >
                      {column.year}
                    </text>
                    <text
                      x={column.x + column.width - 13}
                      y="51"
                      fill={INK}
                      textAnchor="end"
                      fontSize="15"
                      fontWeight="700"
                    >
                      {
                        papers.filter((paper) => paper.year === column.year)
                          .length
                      }{" "}
                      篇
                      {groupPageCount(String(column.year)) > 1
                        ? ` · ${groupPage(String(column.year)) + 1}/${groupPageCount(String(column.year))} 页`
                        : ""}
                    </text>
                  </g>
                ))}
            </>
          )}
          {displayedPapers.map((paper) => {
            const box = geometry.nodes.get(paper.id);
            if (!box) return null;
            const selected = selectedId === paper.id;
            const compact = box.height < 90;
            const tiny = box.height < 50;
            const color = COLORS[paper.cluster] || "#badff5";
            const titleSize = tiny ? 20 : mobile ? 23 : 21;
            const mechanism = paper.mechanism;
            const mechanismLines =
              !compact && box.width < 200 && mechanism.length > 8
                ? [mechanism.slice(0, 8), mechanism.slice(8)]
                : [mechanism];
            const venue = paper.venue
              .replace(/\bDatasets and Benchmarks\b/gi, "D&B")
              .replace(/\s+Workshops?\b/gi, "W");
            // Keep named tracks intact; narrow overview cards expose the year
            // through their full title and accessible label rather than squeezing type.
            const venueLabel =
              box.width < 200 && venue.length > 10
                ? venue
                : `${venue} · ${paper.year}`;
            return (
              <g
                key={paper.id}
                role="button"
                tabIndex={0}
                aria-label={`${paper.shortTitle}，${paper.venue}，${paper.year}，${mechanism}，视频异常理解，查看详情`}
                aria-pressed={selected}
                className={`comic-paper${selected ? " is-selected" : ""}`}
                transform={`translate(${box.x} ${box.y})`}
                onClick={() => {
                  if (!suppressClick.current) onSelect(paper.id);
                }}
                onKeyDown={(event) => {
                  if (event.key === "Enter" || event.key === " ") {
                    event.preventDefault();
                    onSelect(paper.id);
                  }
                }}
              >
                <title>{`${paper.title} — ${paper.venue}, ${paper.year}`}</title>
                <rect
                  className="comic-card-shadow"
                  x={selected ? 4 : 2}
                  y={selected ? 5 : 3}
                  width={box.width}
                  height={box.height}
                  rx="3"
                  fill={INK}
                />
                <rect
                  className="comic-card-body"
                  width={box.width}
                  height={box.height}
                  rx="3"
                  fill={selected ? "#ffdf70" : "#fffefa"}
                  stroke={INK}
                  strokeWidth={selected ? 3.5 : 2}
                />
                {layout === "timeline" && (
                  <rect
                    x="1.5"
                    y="1.5"
                    width="5"
                    height={box.height - 3}
                    fill={color}
                  />
                )}
                <text
                  x="10"
                  y={tiny ? 19 : 25}
                  fill={INK}
                  fontSize={titleSize}
                  fontWeight="800"
                  letterSpacing="-0.5"
                  textLength={
                    paper.shortTitle.length > 15 ? box.width - 22 : undefined
                  }
                  lengthAdjust="spacingAndGlyphs"
                >
                  {paper.shortTitle}
                </text>
                {!tiny && (
                  <text
                    x="10"
                    y={compact ? 46 : 50}
                    fill="#35352f"
                    fontSize="18"
                    fontWeight="500"
                  >
                    {mechanismLines.map((line, index) => (
                      <tspan key={index} x="10" dy={index ? 21 : 0}>
                        {line}
                      </tspan>
                    ))}
                  </text>
                )}
                <text
                  x="10"
                  y={tiny ? 39 : box.height - 6}
                  fill="#333126"
                  fontSize="18"
                  fontWeight="650"
                >
                  {venueLabel}
                </text>
              </g>
            );
          })}
          {!mobile && (
            <g>
              <path
                d="M13 686h21l7 9-7 9H13z"
                fill="#ffdf70"
                stroke={INK}
                strokeWidth="1.6"
              />
              <text x="51" y="700" fill={INK} fontSize="14" fontWeight="600">
                {layout === "map" ? "阅读路径" : "论文发表年份"}
              </text>
              <text x="1182" y="700" textAnchor="end" fill={INK} fontSize="13">
                {papers.length} 篇论文
              </text>
            </g>
          )}
        </g>
      </svg>
      {papers.length === 0 && (
        <div className="comic-empty">
          <strong>暂无匹配论文</strong>
          <button type="button" onClick={onReset}>
            清除筛选 ↗
          </button>
        </div>
      )}
      <div className="comic-map-bottom">
        <div className="comic-navigation">
          {mobile ? (
            <div
              className="comic-pages"
              aria-label={layout === "map" ? "切换创新机制分镜" : "切换年份"}
            >
              <button
                type="button"
                disabled={(layout === "map" ? page : yearPage) === 0}
                onClick={() => changePage(-1)}
                aria-label={layout === "map" ? "上一个创新机制" : "上一年"}
              >
                ←
              </button>
              <span aria-live="polite">
                {layout === "map"
                  ? `${String(page + 1).padStart(2, "0")} / ${String(orderedClusters.length).padStart(2, "0")}`
                  : mobileYear}
              </span>
              <button
                type="button"
                disabled={
                  layout === "map"
                    ? page >= orderedClusters.length - 1
                    : yearPage >= geometry.yearColumns.length - 1
                }
                onClick={() => changePage(1)}
                aria-label={layout === "map" ? "下一个创新机制" : "下一年"}
              >
                →
              </button>
            </div>
          ) : (
            <span className="comic-map-footnote" />
          )}
          <div className="comic-pages comic-paper-pages" aria-label="文献分页">
            <button
              type="button"
              disabled={currentPaperPage === 0}
              onClick={() => setPaperPage(currentPaperPage - 1)}
              aria-label="上一页文献"
            >
              ‹
            </button>
            <span aria-live="polite">
              文献 {currentPaperPage + 1}/{paperPageCount}
            </span>
            <button
              type="button"
              disabled={currentPaperPage >= paperPageCount - 1}
              onClick={() => setPaperPage(currentPaperPage + 1)}
              aria-label="下一页文献"
            >
              ›
            </button>
          </div>
        </div>
        <div className="comic-controls" aria-label="地图视图控制">
          <button
            type="button"
            onClick={() => changeZoom(-0.2)}
            disabled={zoom <= 0.7}
            aria-label="缩小"
          >
            −
          </button>
          <button
            type="button"
            className="comic-zoom"
            onClick={resetView}
            aria-label="复位到完整地图"
            title="复位视角"
          >
            {Math.round(zoom * 100)}%
          </button>
          <button
            type="button"
            onClick={() => changeZoom(0.2)}
            disabled={zoom >= 2.5}
            aria-label="放大"
          >
            +
          </button>
          <button
            type="button"
            className="comic-export"
            onClick={exportSvg}
            disabled={exporting}
            title="导出当前可见文献页 SVG"
          >
            {exporting ? "…" : exportFailed ? "重试 SVG" : "SVG ↗"}
          </button>
        </div>
      </div>
    </section>
  );
}
