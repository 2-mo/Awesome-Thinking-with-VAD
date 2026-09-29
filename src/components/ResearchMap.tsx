import { useEffect, useId, useMemo, useRef, useState } from "react";
import type { PointerEvent as ReactPointerEvent } from "react";
import type { Cluster, Paper, Relation } from "../types";
import {
  createPublicationLayout,
  labelWidth,
  publicationLabel,
  stationName,
} from "./publication-layout";
import type { Point } from "./publication-layout";
import { publicationVenue } from "../publication";
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
type Box = Point & { width: number; height: number };
const INK = "#243b3b";
const PAPER = "#fffcf5";
const clamp = (n: number, min: number, max: number) =>
  Math.min(max, Math.max(min, n));
const fitLabel = (text: string, size: number, available: number) =>
  labelWidth(text, size) > available ? available : undefined;

// Round the metro skeleton without moving its stations. At station bends the
// small curve stays underneath the opaque marker; other bends have more room.
function routePath(points: Point[], stations: Point[], labels: Box[]) {
  if (!points.length) return "";
  let path = `M${points[0].x} ${points[0].y}`;
  for (let i = 1; i < points.length - 1; i++) {
    const a = points[i - 1],
      p = points[i],
      b = points[i + 1];
    const atStation = stations.some(
      (station) =>
        Math.abs(station.x - p.x) < 0.001 &&
        Math.abs(station.y - p.y) < 0.001,
    );
    const before = Math.hypot(p.x - a.x, p.y - a.y),
      after = Math.hypot(b.x - p.x, b.y - p.y);
    if (!before || !after) continue;
    let radius = Math.min(atStation ? 12 : 28, before * 0.45, after * 0.45);
    const trim = (point: Point, distance: number): Point => ({
      x: p.x + ((point.x - p.x) * radius) / distance,
      y: p.y + ((point.y - p.y) * radius) / distance,
    });
    let start = trim(a, before),
      end = trim(b, after);
    // The curve lies inside this control triangle. Preserve the label clearance
    // of the straight route instead of rounding through a nearby paper name.
    const overlapsLabel = () => {
      const left = Math.min(start.x, p.x, end.x),
        right = Math.max(start.x, p.x, end.x),
        top = Math.min(start.y, p.y, end.y),
        bottom = Math.max(start.y, p.y, end.y);
      return labels.some((box) =>
        left < box.x + box.width + 3 && right > box.x - 3 &&
        top < box.y + box.height + 3 && bottom > box.y - 3,
      );
    };
    while (radius > 1 && overlapsLabel()) {
      radius *= 0.65;
      start = trim(a, before);
      end = trim(b, after);
    }
    path += `L${start.x} ${start.y}Q${p.x} ${p.y} ${end.x} ${end.y}`;
  }
  const last = points[points.length - 1];
  return `${path}L${last.x} ${last.y}`;
}

export default function ResearchMap({
  papers,
  allPapers,
  clusters,
  selectedId,
  onSelect,
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
  const [hoveredId, setHoveredId] = useState<string | null>(null);
  const [exportFailed, setExportFailed] = useState(false);
  const id = useId().replace(/[^a-zA-Z0-9_-]/g, "");
  const patternId = `publication-grid-${id}`;
  const sourcePapers = allPapers || papers;
  const visibleIds = useMemo(() => new Set(papers.map((p) => p.id)), [papers]);
  const network = useMemo(
    () => createPublicationLayout(sourcePapers),
    [sourcePapers],
  );
  const { width, height } = network;
  const focusedLine =
    sourcePapers.find((paper) => paper.id === (hoveredId || selectedId))
      ?.cluster || (activeCluster === "all" ? null : activeCluster);
  const timeline = useMemo(() => {
    const years = [...new Set(sourcePapers.map((p) => p.year))].sort(
      (a, b) => a - b,
    );
    const units = years.map((year) =>
      Math.max(
        1,
        Math.ceil(sourcePapers.filter((p) => p.year === year).length / 10),
      ),
    );
    const unitWidth =
      (width - 48 - (years.length - 1) * 16) /
      Math.max(
        1,
        units.reduce((sum, n) => sum + n, 0),
      );
    const nodes = new Map<string, Box>();
    let x = 24;
    const columns = years.map((year, index) => {
      const column = { year, x, width: unitWidth * units[index] };
      const members = papers
        .filter((p) => p.year === year)
        .sort(
          (a, b) =>
            a.venue.localeCompare(b.venue) ||
            a.shortTitle.localeCompare(b.shortTitle),
        );
      members.forEach((paper, i) =>
        nodes.set(paper.id, {
          x: x + 12 + (i % units[index]) * (column.width / units[index]),
          y: 100 + Math.floor(i / units[index]) * 61,
          width: column.width / units[index] - 24,
          height: 49,
        }),
      );
      x += column.width + 16;
      return column;
    });
    return { nodes, columns };
  }, [sourcePapers, papers, width]);
  const left = network.years[0]?.x ?? 190;
  const plotTop = network.venues[0]?.y ?? 90;
  const lastVenue = network.venues[network.venues.length - 1];
  const plotBottom = lastVenue ? lastVenue.y + lastVenue.height : height - 70;

  useEffect(() => {
    setPan({ x: 0, y: 0 });
    setZoom(1);
  }, [layout]);
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
    const dx = event.clientX - start.x,
      dy = event.clientY - start.y;
    if (Math.abs(dx) + Math.abs(dy) < 5 && !start.moved) return;
    start.moved = true;
    suppressClick.current = true;
    setDragging(true);
    if (!event.currentTarget.hasPointerCapture(event.pointerId))
      event.currentTarget.setPointerCapture(event.pointerId);
    const bounds = event.currentTarget.getBoundingClientRect();
    const scale = Math.min(bounds.width / width, bounds.height / height);
    setPan({ x: start.origin.x + dx / scale, y: start.origin.y + dy / scale });
  };
  const endDrag = (event: ReactPointerEvent<SVGSVGElement>) => {
    if (event.currentTarget.hasPointerCapture(event.pointerId))
      event.currentTarget.releasePointerCapture(event.pointerId);
    drag.current = null;
    setDragging(false);
  };
  const exportSvg = () => {
    if (!svgRef.current) return;
    setExportFailed(false);
    try {
      const clone = svgRef.current.cloneNode(true) as SVGSVGElement;
      clone.setAttribute("xmlns", "http://www.w3.org/2000/svg");
      clone.setAttribute("width", String(width));
      clone.setAttribute("height", String(height));
      clone.querySelector("[data-map-content]")?.removeAttribute("transform");
      const url = URL.createObjectURL(
        new Blob([new XMLSerializer().serializeToString(clone)], {
          type: "image/svg+xml;charset=utf-8",
        }),
      );
      const anchor = document.createElement("a");
      anchor.href = url;
      anchor.download = `vau-publication-${layout}.svg`;
      anchor.click();
      window.setTimeout(() => URL.revokeObjectURL(url), 1000);
    } catch {
      setExportFailed(true);
    }
  };

  return (
    <section
      className={`research-map metro-map publication-map${dragging ? " is-dragging" : ""}`}
      aria-label={
        layout === "map" ? "年份、会议与方法派别研究线路图" : "论文发表时间排列"
      }
    >
      <svg
        ref={svgRef}
        className="research-map-canvas"
        viewBox={`0 0 ${width} ${height}`}
        aria-label={
          layout === "map"
            ? `${papers.length}篇论文，横轴年份与季度，纵轴会议，彩色线路表示方法派别`
            : `${papers.length}篇论文，按发表年份排列`
        }
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
          {layout === "map" ? "年份 × 会议 · 方法线路" : "发表时间线"}
        </title>
        <defs>
          <pattern
            id={patternId}
            width="20"
            height="20"
            patternUnits="userSpaceOnUse"
          >
            <circle cx="10" cy="10" r=".55" fill="#55706b" opacity=".1" />
          </pattern>
        </defs>
        <rect width={width} height={height} fill={PAPER} />
        <g
          data-map-content="true"
          transform={`translate(${pan.x + width / 2} ${pan.y + height / 2}) scale(${zoom}) translate(${-width / 2} ${-height / 2})`}
        >
          {layout === "map" ? (
            <>
              <rect
                x={left}
                y={plotTop}
                width={width - left - 24}
                height={plotBottom - plotTop}
                fill={`url(#${patternId})`}
              />
              {network.years.map((year, index) => (
                <g key={year.year}>
                  <rect
                    x={year.x}
                    y={plotTop}
                    width={year.width}
                    height={plotBottom - plotTop}
                    fill={index % 2 ? "#e9eee7" : PAPER}
                    fillOpacity={index % 2 ? 0.2 : 0.1}
                  />
                  <path
                    d={`M${year.x} ${plotTop - 10}V${plotBottom}`}
                    stroke="#cad3c8"
                    strokeWidth="1"
                    strokeDasharray="3 6"
                  />
                  <text
                    x={year.x + year.width / 2}
                    y={plotTop - 49}
                    textAnchor="middle"
                    fill={INK}
                    fontSize="27"
                    fontWeight="850"
                    letterSpacing="-.8"
                  >
                    {year.year}
                  </text>
                  <path
                    d={`M${year.x + 20} ${plotTop - 36}h${year.width - 40}`}
                    stroke={INK}
                    strokeWidth="2"
                  />
                  {year.quarters.map((quarter, quarterIndex) => (
                    <g key={quarter.quarter ?? "unknown"}>
                      {quarterIndex > 0 && (
                        <path
                          d={`M${quarter.x} ${plotTop - 25}V${plotBottom}`}
                          stroke="#c3cdc2"
                          strokeWidth=".8"
                          strokeDasharray="2 8"
                          opacity=".55"
                        />
                      )}
                      <text
                        x={quarter.x + quarter.width / 2}
                        y={plotTop - 13}
                        textAnchor="middle"
                        fill={quarter.count ? "#52685e" : "#8b978d"}
                        fontSize="14"
                        fontWeight="750"
                      >
                        {quarter.quarter === null ? "待定" : `Q${quarter.quarter}`}
                      </text>
                    </g>
                  ))}
                </g>
              ))}
              <text
                x="25"
                y={plotTop - 49}
                fill={INK}
                fontSize="13"
                fontWeight="800"
                letterSpacing="1"
              >
                会议 / 时间 →
              </text>
              {network.venues.map((venue) => {
                const selectedVenue =
                  publicationVenue(
                    sourcePapers.find((paper) => paper.id === selectedId)
                      ?.venue || "",
                  ) === venue.venue;
                const label = venue.label;
                const parts =
                  venue.venue === "arXiv" ? ["arXiv", "预印本"] : [label];
                return (
                  <g key={venue.venue}>
                    <rect
                      x="16"
                      y={venue.y + 3}
                      width={left - 27}
                      height={venue.height - 6}
                      rx="4"
                      fill={selectedVenue ? "#fce8a2" : "transparent"}
                    />
                    <path
                      d={`M${left} ${venue.y + venue.height}H${width - 24}`}
                      stroke="#d8ded3"
                      strokeWidth=".8"
                      opacity=".3"
                    />
                    <text
                      x="29"
                      y={
                        venue.y + venue.height / 2 - (parts.length > 1 ? 2 : -5)
                      }
                      fill={INK}
                      fontSize="14"
                      fontWeight="750"
                    >
                      {parts.map((part, index) => (
                        <tspan
                          key={part}
                          x="29"
                          dy={index ? 14 : 0}
                          fontSize={index ? 11 : 14}
                        >
                          {part}
                        </tspan>
                      ))}
                    </text>
                    <text
                      x={left - 23}
                      y={venue.y + venue.height / 2 + 5}
                      textAnchor="end"
                      fill="#768379"
                      fontSize="11"
                      fontWeight="650"
                    >
                      {
                        papers.filter(
                          (paper) =>
                            publicationVenue(paper.venue) === venue.venue,
                        ).length
                      }
                    </text>
                  </g>
                );
              })}
              {network.lines.map((line) => {
                const focus = !focusedLine || line.id === focusedLine;
                const path = routePath(
                  line.track,
                  [...network.stations.values()].filter(
                    (station) => station.lineId === line.id,
                  ),
                  [...network.stations.values()].map((station) => station.label),
                );
                return (
                  <g
                    key={line.id}
                    className="publication-route"
                    opacity={focus ? 0.95 : 0.17}
                  >
                    <path
                      d={path}
                      fill="none"
                      stroke={PAPER}
                      strokeWidth={focus && focusedLine ? 11 : 9}
                      strokeLinejoin="round"
                      strokeLinecap="round"
                    />
                    <path
                      d={path}
                      fill="none"
                      stroke={line.color}
                      strokeWidth={focus && focusedLine ? 6 : 4.8}
                      strokeLinejoin="round"
                      strokeLinecap="round"
                    />
                  </g>
                );
              })}
              {sourcePapers
                .filter((paper) => !visibleIds.has(paper.id))
                .map((paper) => {
                  const point = network.stations.get(paper.id);
                  return (
                    point && (
                      <circle
                        key={paper.id}
                        cx={point.x}
                        cy={point.y}
                        r="3.5"
                        fill={PAPER}
                        stroke="#b1b9ae"
                        strokeWidth="1.5"
                      />
                    )
                  );
                })}
              {network.lines.map((line, index) => {
                const x = 26 + index * ((width - 52) / network.lines.length);
                return (
                  <g key={line.id} transform={`translate(${x} ${height - 28})`}>
                    <path
                      d="M0 0h30"
                      stroke={line.color}
                      strokeWidth="4"
                      strokeLinecap="round"
                    />
                    <circle
                      cx="15"
                      r="6.5"
                      fill={PAPER}
                      stroke={line.color}
                      strokeWidth="2.8"
                    />
                    <circle cx="15" r="2" fill={line.color} />
                    <text
                      x="41"
                      y="5"
                      fill={INK}
                      fontSize="14"
                      fontWeight="700"
                    >
                      {clusters.find((cluster) => cluster.id === line.id)
                        ?.name || line.label}
                    </text>
                  </g>
                );
              })}
            </>
          ) : (
            timeline.columns.map((column) => (
              <g key={column.year}>
                <rect
                  x={column.x}
                  y="30"
                  width={column.width}
                  height={height - 68}
                  rx="7"
                  fill={PAPER}
                  stroke="#d7dcd0"
                />
                <text
                  x={column.x + 14}
                  y="72"
                  fill={INK}
                  fontSize="30"
                  fontWeight="850"
                >
                  {column.year}
                </text>
                <text
                  x={column.x + column.width - 14}
                  y="69"
                  textAnchor="end"
                  fill="#52605a"
                  fontSize="13"
                >
                  {papers.filter((p) => p.year === column.year).length} 篇
                </text>
                <path
                  d={`M${column.x + 14} 85h${column.width - 28}`}
                  stroke={INK}
                  strokeWidth="2"
                />
              </g>
            ))
          )}
          {papers.map((paper) => {
            const station =
              layout === "map" ? network.stations.get(paper.id) : undefined;
            const box = station ? station.label : timeline.nodes.get(paper.id);
            if (!box) return null;
            const selected = selectedId === paper.id;
            const active = selected || hoveredId === paper.id;
            const line = network.lines.find(
              (item) => item.id === paper.cluster,
            );
            const color = line?.color || INK;
            const label = publicationLabel(paper);
            const fontSize = station ? 14 : 16;
            const name = station ? stationName(paper) : paper.shortTitle;
            const tx = station ? box.x + box.width / 2 : box.x + 13;
            const available = box.width - (station ? 10 : 26);
            const labelY = station ? box.y + box.height / 2 + 5 : box.y + 18;
            const anchor = station
              ? {
                  x: clamp(station.x, box.x, box.x + box.width),
                  y: clamp(station.y, box.y, box.y + box.height),
                }
              : null;
            return (
              <g
                key={paper.id}
                role="button"
                tabIndex={0}
                aria-label={`${paper.title}，${paper.venue}，${paper.year}，${clusters.find((cluster) => cluster.id === paper.cluster)?.name || ""}，查看详情`}
                aria-pressed={selected}
                className={`metro-station${selected ? " is-selected" : ""}`}
                onMouseEnter={() => setHoveredId(paper.id)}
                onMouseLeave={() => setHoveredId(null)}
                onFocus={() => setHoveredId(paper.id)}
                onBlur={() => setHoveredId(null)}
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
                {station && anchor && (
                  <path
                    d={`M${station.x} ${station.y}L${anchor.x} ${anchor.y}`}
                    stroke={color}
                    strokeWidth="1.4"
                  />
                )}
                <rect
                  className="metro-station-label"
                  x={box.x}
                  y={box.y}
                  width={box.width}
                  height={box.height}
                  rx="3"
                  fill={selected ? "#fce8a2" : PAPER}
                  stroke={selected ? color : PAPER}
                  strokeWidth={selected ? 1.5 : 3}
                />
                {station ? (
                  <>
                    <circle
                      className="metro-station-halo"
                      cx={station.x}
                      cy={station.y}
                      r={active ? 16 : 11}
                      fill={color}
                      fillOpacity={active ? 0.16 : 0}
                    />
                    <circle
                      cx={station.x}
                      cy={station.y}
                      r={active ? 12.5 : 11}
                      fill={PAPER}
                    />
                    <circle
                      className="metro-station-dot"
                      cx={station.x}
                      cy={station.y}
                      r={active ? 9.5 : 8}
                      fill={active ? color : PAPER}
                      stroke={color}
                      strokeWidth="3.2"
                    />
                    <circle
                      cx={station.x}
                      cy={station.y}
                      r={active ? 3 : 2.3}
                      fill={active ? PAPER : color}
                    />
                    <circle
                      cx={station.x}
                      cy={station.y}
                      r="16"
                      fill="transparent"
                    />
                  </>
                ) : (
                  <rect
                    x={box.x}
                    y={box.y + 7}
                    width="4"
                    height={box.height - 14}
                    rx="2"
                    fill={color}
                  />
                )}
                <text
                  x={tx}
                  y={labelY}
                  textAnchor={station ? "middle" : "start"}
                  fill={INK}
                  fontSize={fontSize}
                  fontWeight="750"
                  textLength={fitLabel(name, fontSize, available)}
                  lengthAdjust="spacingAndGlyphs"
                >
                  {name}
                </text>
                {!station && (
                  <text
                    x={tx}
                    y={box.y + 35}
                    fill="#52605a"
                    fontSize="12.5"
                    fontWeight="550"
                    textLength={fitLabel(label, 12.5, available)}
                    lengthAdjust="spacingAndGlyphs"
                  >
                    {label}
                  </text>
                )}
              </g>
            );
          })}
        </g>
      </svg>
      {papers.length === 0 && (
        <div className="metro-empty">
          <strong>暂无匹配论文</strong>
          <button type="button" onClick={onReset}>
            清除筛选 ↗
          </button>
        </div>
      )}
      <div className="metro-map-bottom">
        <span className="metro-map-count">{papers.length} 篇论文</span>
        <div className="metro-controls" aria-label="地图视图控制">
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
            className="metro-zoom"
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
            className="metro-export"
            onClick={exportSvg}
            title="导出完整地图 SVG"
          >
            {exportFailed ? "重试 SVG" : "SVG ↗"}
          </button>
        </div>
      </div>
    </section>
  );
}
