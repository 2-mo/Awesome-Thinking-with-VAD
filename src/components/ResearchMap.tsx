import { useEffect, useId, useMemo, useRef, useState } from "react";
import type { PointerEvent as ReactPointerEvent } from "react";
import type { Cluster, Paper, Relation } from "../types";
import panorama from "../assets/research-panorama.png";
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
const ROUTE_BENDS = [
  [0, 24, 12, -12, -24, -8, 16, 26, 0],
  [0, -16, -22, 4, 24, 8, 0],
  [0, 22, 8, -24, -12, 26, 0],
  [0, 20, -28, 0],
  [0, -18, 20, 28, -10, -26, 6, 18, -12, 0],
];
const COLORS: Record<string, string> = {
  alignment: "#badff5",
  explanation: "#ffc3b5",
  reasoning: "#c5e2c1",
  evidence: "#ffdf70",
  understanding: "#d9cef1",
};
const clamp = (n: number, min: number, max: number) =>
  Math.min(max, Math.max(min, n));
// A conservative width estimate bounds every label, including mixed CJK/Latin text.
const labelWidth = (text: string, size: number) =>
  [...text].reduce(
    (width, char) =>
      width +
      size *
        (/[^\u0000-\u00ff]/.test(char)
          ? 1
          : /[MW@]/.test(char)
            ? 0.95
            : /[A-Z]/.test(char)
              ? 0.72
              : /[a-z0-9]/.test(char)
                ? 0.6
                : 0.4),
    0,
  );
const fitLabel = (text: string, size: number, available: number) =>
  labelWidth(text, size) > available ? available : undefined;

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
  const id = useId().replace(/[^a-zA-Z0-9_-]/g, "");
  const patternId = `comic-dots-${id}`;
  const sourcePapers = allPapers || papers;
  const orderedClusters = useMemo(
    () =>
      [...clusters].sort((a, b) => ORDER.indexOf(a.id) - ORDER.indexOf(b.id)),
    [clusters],
  );
  useEffect(() => {
    setPan({ x: 0, y: 0 });
    setZoom(1);
  }, [layout]);

  const geometry = useMemo(() => {
    const width = 1600;
    const height = 700;
    const panels = new Map<string, Box>();
    const nodes = new Map<string, Box>();
    const stations = new Map<string, Point>();
    const routes = new Map<string, Point[]>();
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
    const placeCards = (members: Paper[], panel: Box, columns: number) => {
      const rows = Math.max(1, Math.ceil(members.length / columns));
      const gap = 10;
      const cardWidth = (panel.width - 28 - (columns - 1) * gap) / columns;
      const cardHeight = Math.min(
        46,
        (panel.height - 70 - (rows - 1) * gap) / rows,
      );
      members.forEach((paper, index) =>
        nodes.set(paper.id, {
          x: panel.x + 12 + (index % columns) * (cardWidth + gap),
          y: panel.y + 54 + Math.floor(index / columns) * (cardHeight + gap),
          width: cardWidth,
          height: cardHeight,
        }),
      );
    };
    const years = [...new Set(sourcePapers.map((paper) => paper.year))].sort(
      (a, b) => a - b,
    );
    const yearUnits = years.map((year) =>
      Math.max(
        1,
        Math.ceil(sourcePapers.filter((p) => p.year === year).length / 10),
      ),
    );
    const unitWidth =
      (1580 - (years.length - 1) * 12) /
      Math.max(
        1,
        yearUnits.reduce((sum, n) => sum + n, 0),
      );
    let yearX = 8;
    const yearColumns = years.map((year, index) => {
      const panel = {
        x: yearX,
        y: 8,
        width: unitWidth * yearUnits[index],
        height: 656,
      };
      yearX += panel.width + 12;
      if (layout === "timeline")
        placeCards(groups.get(String(year)) || [], panel, yearUnits[index]);
      return { ...panel, year };
    });
    if (layout === "map")
      orderedClusters.forEach((cluster, index) => {
        const baseline = 82 + index * 130;
        const members = sourcePapers
          .filter((paper) => paper.cluster === cluster.id)
          .sort((a, b) => a.year - b.year || a.id.localeCompare(b.id));
        const spacing = 1266 / Math.max(1, members.length - 1);
        const reverse = index % 2 === 1;
        const points = members.map((paper, position) => {
          const x = reverse
            ? 1516 - position * spacing
            : 250 + position * spacing;
          const bends = ROUTE_BENDS[index] || [0];
          const bend =
            bends[
              Math.round(
                (position / Math.max(1, members.length - 1)) *
                  (bends.length - 1),
              )
            ];
          const station = { x, y: baseline + bend };
          stations.set(paper.id, station);
          nodes.set(paper.id, {
            x: station.x - 68,
            y: station.y + (position % 2 ? 12 : -55),
            width: 136,
            height: 42,
          });
          return station;
        });
        routes.set(cluster.id, [
          { x: reverse ? 1562 : 194, y: baseline },
          ...points,
          { x: reverse ? 194 : 1562, y: baseline },
        ]);
        panels.set(cluster.id, {
          x: 12,
          y: baseline - 29,
          width: 170,
          height: 58,
        });
      });
    return { width, height, panels, nodes, stations, routes, yearColumns };
  }, [papers, sourcePapers, orderedClusters, layout]);
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
      const response = await fetch(panorama);
      if (!response.ok) throw new Error("Background unavailable");
      const artwork = await response.blob();
      const encoded = await new Promise<string>((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => resolve(String(reader.result));
        reader.onerror = () => reject(reader.error);
        reader.readAsDataURL(artwork);
      });
      clone.querySelector("image")?.setAttribute("href", encoded);
      const xml = new XMLSerializer().serializeToString(clone);
      const url = URL.createObjectURL(
        new Blob([xml], { type: "image/svg+xml;charset=utf-8" }),
      );
      const anchor = document.createElement("a");
      anchor.href = url;
      anchor.download = `vau-idea-atlas-${layout}.svg`;
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
      className={`research-map comic-map${dragging ? " is-dragging" : ""}`}
      aria-label={layout === "map" ? "创新机制路线地图" : "论文发表时间排列"}
    >
      <svg
        ref={svgRef}
        className="research-map-canvas"
        viewBox={`0 0 ${geometry.width} ${geometry.height}`}
        aria-label={`${papers.length}篇论文，选择卡片查看详情`}
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
          {layout === "map" ? "创新机制路线地图" : "论文发表时间排列"}
        </title>
        <defs>
          <pattern
            id={patternId}
            width="8"
            height="8"
            patternUnits="userSpaceOnUse"
          >
            <circle cx="2" cy="2" r="0.7" fill={INK} opacity="0.17" />
          </pattern>
        </defs>
        <rect width={geometry.width} height={geometry.height} fill={CREAM} />

        <g
          data-map-content="true"
          transform={`translate(${pan.x + geometry.width / 2} ${pan.y + geometry.height / 2}) scale(${zoom}) translate(${-geometry.width / 2} ${-geometry.height / 2})`}
        >
          <image
            href={panorama}
            width={geometry.width}
            height={geometry.height}
            preserveAspectRatio="xMidYMid slice"
            opacity=".34"
          />
          <rect
            width={geometry.width}
            height={geometry.height}
            fill={`url(#${patternId})`}
            opacity=".2"
          />
          {layout === "map" ? (
            <>
              {[0, 1, 2, 3].map((index) => {
                const right = index % 2 === 0;
                const x = right ? 1562 : 194;
                const bend = right ? 1590 : 165;
                const y = 82 + index * 130;
                return (
                  <path
                    key={index}
                    d={`M${x} ${y}C${bend} ${y + 25} ${bend} ${y + 105} ${x} ${y + 130}`}
                    fill="none"
                    stroke={INK}
                    strokeWidth="3"
                    strokeDasharray="4 6"
                    opacity=".38"
                  />
                );
              })}
              {orderedClusters.map((cluster, index) => {
                const panel = geometry.panels.get(cluster.id)!;
                const count = papers.filter(
                  (p) => p.cluster === cluster.id,
                ).length;
                const active = activeCluster === cluster.id;
                const points = geometry.routes.get(cluster.id)!;
                const path = points
                  .map((point, position) => {
                    if (!position) return `M${point.x} ${point.y}`;
                    const previous = points[position - 1];
                    const middle = (previous.x + point.x) / 2;
                    return `C${middle} ${previous.y} ${middle} ${point.y} ${point.x} ${point.y}`;
                  })
                  .join(" ");
                return (
                  <g key={cluster.id}>
                    <path
                      d={path}
                      fill="none"
                      stroke={INK}
                      strokeWidth={active ? 12 : 10}
                      strokeLinejoin="round"
                      strokeLinecap="round"
                    />
                    <path
                      d={path}
                      fill="none"
                      stroke={COLORS[cluster.id]}
                      strokeWidth={active ? 8 : 6}
                      strokeLinejoin="round"
                      strokeLinecap="round"
                    />
                    <circle
                      cx={points[0].x}
                      cy={panel.y + 29}
                      r="7"
                      fill={COLORS[cluster.id]}
                      stroke={INK}
                      strokeWidth="2"
                    />
                    <g
                      className="comic-cluster"
                      role="button"
                      tabIndex={0}
                      aria-label={`筛选${cluster.name}，${count}篇论文`}
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
                        x={panel.x}
                        y={panel.y}
                        width={panel.width}
                        height={panel.height}
                        rx="4"
                        fill="transparent"
                      />
                      <path
                        d={`M${panel.x + 3} ${panel.y + 5}h27l-3 22h-27z`}
                        fill={COLORS[cluster.id]}
                        stroke={INK}
                        strokeWidth="1.8"
                      />
                      <text
                        x={panel.x + 15}
                        y={panel.y + 21}
                        textAnchor="middle"
                        fill={INK}
                        fontSize="15"
                        fontStyle="italic"
                        fontWeight="900"
                      >
                        {index + 1}
                      </text>
                      <text
                        x={panel.x + 39}
                        y={panel.y + 21}
                        fill="#585348"
                        fontSize="12"
                        fontWeight="700"
                      >
                        {count} 篇
                      </text>
                      <text
                        x={panel.x + 1}
                        y={panel.y + 48}
                        fill={INK}
                        fontSize="17"
                        fontWeight="850"
                        textLength={fitLabel(cluster.name, 17, panel.width - 8)}
                        lengthAdjust="spacingAndGlyphs"
                      >
                        {cluster.name}
                      </text>
                    </g>
                  </g>
                );
              })}
            </>
          ) : (
            geometry.yearColumns.map((column) => (
              <g key={column.year}>
                <rect
                  x={column.x + 4}
                  y={column.y + 5}
                  width={column.width}
                  height={column.height}
                  fill={INK}
                />
                <rect
                  x={column.x}
                  y={column.y}
                  width={column.width}
                  height={column.height}
                  fill="#fffdf6"
                  stroke={INK}
                  strokeWidth="2.5"
                />
                <path
                  d={`M${column.x} ${column.y}h${column.width}v44h-${column.width}z`}
                  fill="#ffdf70"
                  stroke={INK}
                  strokeWidth="2"
                />
                <text
                  x={column.x + 12}
                  y={column.y + 33}
                  fill={INK}
                  fontFamily="Impact, 'Arial Black', sans-serif"
                  fontSize="31"
                  fontWeight="900"
                >
                  {column.year}
                </text>
                <text
                  x={column.x + column.width - 12}
                  y={column.y + 29}
                  fill={INK}
                  textAnchor="end"
                  fontSize="13"
                  fontWeight="700"
                >
                  {papers.filter((p) => p.year === column.year).length} 篇
                </text>
              </g>
            ))
          )}
          {papers.map((paper) => {
            const box = geometry.nodes.get(paper.id);
            if (!box) return null;
            const selected = selectedId === paper.id;
            const station = geometry.stations.get(paper.id);
            const venue = paper.venue
              .replace(/\bDatasets and Benchmarks\b/gi, "D&B")
              .replace(/\s+Workshops?\b/gi, "W");
            const venueLabel = `${venue} · ${paper.year}`;
            return (
              <g
                key={paper.id}
                role="button"
                tabIndex={0}
                aria-label={`${paper.title}，${paper.venue}，${paper.year}，查看详情`}
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
                {station && (
                  <g aria-hidden="true">
                    <path
                      d={`M${station.x - box.x} ${station.y - box.y}V${station.y < box.y ? 0 : box.height}`}
                      fill="none"
                      stroke={INK}
                      strokeWidth="1.8"
                    />
                    <circle
                      cx={station.x - box.x}
                      cy={station.y - box.y}
                      r={selected ? 7 : 5.5}
                      fill={selected ? "#ffdf70" : CREAM}
                      stroke={INK}
                      strokeWidth="2"
                    />
                  </g>
                )}
                <path
                  className="comic-card-shadow"
                  transform={`translate(${selected ? 3 : 2} ${selected ? 4 : 2})`}
                  d={`M0 0H${box.width}l-5 ${box.height / 2} 5 ${box.height / 2}H0Z`}
                  fill={INK}
                />
                <path
                  className="comic-card-body"
                  d={`M0 0H${box.width}l-5 ${box.height / 2} 5 ${box.height / 2}H0Z`}
                  fill={selected ? "#ffdf70" : "#fffefa"}
                  stroke={INK}
                  strokeWidth={selected ? 3 : 1.6}
                />
                {layout === "timeline" && (
                  <rect
                    x="1.5"
                    y="1.5"
                    width="4"
                    height={box.height - 3}
                    fill={COLORS[paper.cluster]}
                  />
                )}
                <text
                  x="9"
                  y="19"
                  fill={INK}
                  fontSize="16"
                  fontWeight="800"
                  textLength={fitLabel(paper.shortTitle, 16, box.width - 23)}
                  lengthAdjust="spacingAndGlyphs"
                >
                  {paper.shortTitle}
                </text>
                <text
                  x="9"
                  y={box.height - 8}
                  fill="#333126"
                  fontSize="13"
                  fontWeight="650"
                  textLength={fitLabel(venueLabel, 13, box.width - 23)}
                  lengthAdjust="spacingAndGlyphs"
                >
                  {venueLabel}
                </text>
              </g>
            );
          })}
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
        <span className="comic-map-count">
          {papers.length} 篇{layout === "map" ? " · 编辑阅读路线" : ""}
        </span>
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
            title="导出完整地图 SVG"
          >
            {exporting ? "…" : exportFailed ? "重试 SVG" : "SVG ↗"}
          </button>
        </div>
      </div>
    </section>
  );
}
