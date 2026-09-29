import { useEffect, useId, useMemo, useRef, useState } from "react";
import type { PointerEvent as ReactPointerEvent } from "react";
import type { Cluster, Paper, Relation } from "../types";
import {
  createMetroStations,
  labelBox,
  labelWidth,
  publicationLabel,
  MAP_HEIGHT,
  MAP_WIDTH,
  METRO_LINES,
} from "./metro-layout";
import type { Point } from "./metro-layout";
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
const INK = "#263836";
const PAPER = "#fffcf5";
const clamp = (n: number, min: number, max: number) =>
  Math.min(max, Math.max(min, n));
const fitLabel = (text: string, size: number, available: number) =>
  labelWidth(text, size) > available ? available : undefined;

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
  const [exportFailed, setExportFailed] = useState(false);
  const id = useId().replace(/[^a-zA-Z0-9_-]/g, "");
  const patternId = `metro-grid-${id}`;
  const sourcePapers = allPapers || papers;
  const visibleIds = useMemo(() => new Set(papers.map((p) => p.id)), [papers]);
  const stations = useMemo(
    () => createMetroStations(sourcePapers),
    [sourcePapers],
  );
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
      (MAP_WIDTH - 48 - (years.length - 1) * 16) /
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
            METRO_LINES.findIndex((line) => line.id === a.cluster) -
              METRO_LINES.findIndex((line) => line.id === b.cluster) ||
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
  }, [sourcePapers, papers]);

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
    const scale = Math.min(
      bounds.width / MAP_WIDTH,
      bounds.height / MAP_HEIGHT,
    );
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
      clone.setAttribute("width", String(MAP_WIDTH));
      clone.setAttribute("height", String(MAP_HEIGHT));
      clone.querySelector("[data-map-content]")?.removeAttribute("transform");
      const url = URL.createObjectURL(
        new Blob([new XMLSerializer().serializeToString(clone)], {
          type: "image/svg+xml;charset=utf-8",
        }),
      );
      const anchor = document.createElement("a");
      anchor.href = url;
      anchor.download = `vau-metro-${layout}.svg`;
      anchor.click();
      window.setTimeout(() => URL.revokeObjectURL(url), 1000);
    } catch {
      setExportFailed(true);
    }
  };

  return (
    <section
      className={`research-map metro-map${dragging ? " is-dragging" : ""}`}
      aria-label={
        layout === "map" ? "视频异常理解研究线路图" : "论文发表时间排列"
      }
    >
      <svg
        ref={svgRef}
        className="research-map-canvas"
        viewBox={`0 0 ${MAP_WIDTH} ${MAP_HEIGHT}`}
        aria-label={`${papers.length}篇论文，选择站点查看详情`}
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
          视频异常理解 · {layout === "map" ? "研究线路图" : "发表时间线"}
        </title>
        <defs>
          <pattern
            id={patternId}
            width="24"
            height="24"
            patternUnits="userSpaceOnUse"
          >
            <circle cx="12" cy="12" r=".65" fill="#506b65" opacity=".12" />
          </pattern>
        </defs>
        <rect width={MAP_WIDTH} height={MAP_HEIGHT} fill={PAPER} />
        <g
          data-map-content="true"
          transform={`translate(${pan.x + MAP_WIDTH / 2} ${pan.y + MAP_HEIGHT / 2}) scale(${zoom}) translate(${-MAP_WIDTH / 2} ${-MAP_HEIGHT / 2})`}
        >
          <rect
            width={MAP_WIDTH}
            height={MAP_HEIGHT}
            fill={`url(#${patternId})`}
          />
          {layout === "map" ? (
            <>
              <path
                d="M24 66V8H1576V66 M24 728V760H1576V728"
                fill="none"
                stroke="#d7dcd0"
                strokeWidth="1"
              />
              {METRO_LINES.map((line, index) => {
                const name =
                  clusters.find((cluster) => cluster.id === line.id)?.name ||
                  line.id;
                const muted =
                  activeCluster !== "all" && activeCluster !== line.id;
                const legendX = 47 + index * 310;
                const path = line.track
                  .map((point, i) => `${i ? "L" : "M"}${point.x} ${point.y}`)
                  .join(" ");
                return (
                  <g key={line.id} opacity={muted ? 0.2 : 1}>
                    <g aria-label={`${line.number}号线，${name}`}>
                      <rect
                        x={legendX}
                        y="18"
                        width="33"
                        height="25"
                        rx="5"
                        fill={line.color}
                      />
                      <text
                        x={legendX + 16.5}
                        y="35.5"
                        textAnchor="middle"
                        fill="#fff"
                        fontSize="13"
                        fontWeight="850"
                      >
                        {line.number}
                      </text>
                      <text
                        x={legendX + 44}
                        y="36"
                        fill={INK}
                        fontSize="16"
                        fontWeight="750"
                      >
                        {name}
                      </text>
                    </g>
                    <path
                      d={path}
                      fill="none"
                      stroke={PAPER}
                      strokeWidth="16"
                      strokeLinejoin="round"
                      strokeLinecap="round"
                    />
                    <path
                      d={path}
                      fill="none"
                      stroke={line.color}
                      strokeWidth="8"
                      strokeLinejoin="round"
                      strokeLinecap="round"
                    />
                    {line.track
                      .filter((_, i) => i === 0 || i === line.track.length - 1)
                      .map((point, i) => (
                        <g key={i}>
                          <rect
                            x={point.x - 3}
                            y={point.y - 10}
                            width="6"
                            height="20"
                            rx="2"
                            fill={line.color}
                          />
                          {i === 0 && (
                            <text
                              x={point.x - 17}
                              y={point.y + 5}
                              textAnchor="end"
                              fill={line.color}
                              fontSize="15"
                              fontWeight="850"
                            >
                              {line.number}
                            </text>
                          )}
                        </g>
                      ))}
                  </g>
                );
              })}
              {sourcePapers
                .filter((paper) => !visibleIds.has(paper.id))
                .map((paper) => {
                  const point = stations.get(paper.id);
                  return (
                    point && (
                      <circle
                        key={paper.id}
                        cx={point.x}
                        cy={point.y}
                        r="4.5"
                        fill={PAPER}
                        stroke="#aaaFA6"
                        strokeWidth="2"
                        opacity=".55"
                      />
                    )
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
                  height="715"
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
              layout === "map" ? stations.get(paper.id) : undefined;
            const box = station
              ? labelBox(station, paper)
              : timeline.nodes.get(paper.id);
            if (!box) return null;
            const selected = selectedId === paper.id;
            const line = METRO_LINES.find((item) => item.id === paper.cluster)!;
            const label = publicationLabel(paper);
            const tx = station ? box.x + box.width / 2 : box.x + 13;
            const available = box.width - (station ? 10 : 26);
            return (
              <g
                key={paper.id}
                role="button"
                tabIndex={0}
                aria-label={`${paper.title}，${paper.venue}，${paper.year}，查看详情`}
                aria-pressed={selected}
                className={`metro-station${selected ? " is-selected" : ""}`}
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
                  className="metro-station-label"
                  x={box.x}
                  y={box.y}
                  width={box.width}
                  height={box.height}
                  rx="5"
                  fill={selected ? "#fce8a2" : "transparent"}
                  stroke={selected ? line.color : "transparent"}
                  strokeWidth="1.5"
                />
                {station ? (
                  <>
                    <path
                      d={`M${station.x} ${station.y}v${station.below ? 15 : -15}`}
                      stroke={line.color}
                      strokeWidth="2"
                    />
                    <circle
                      className="metro-station-halo"
                      cx={station.x}
                      cy={station.y}
                      r="12"
                      fill={selected ? line.color : "transparent"}
                      fillOpacity=".18"
                    />
                    <circle
                      className="metro-station-dot"
                      cx={station.x}
                      cy={station.y}
                      r={selected ? 7.5 : 6}
                      fill={PAPER}
                      stroke={line.color}
                      strokeWidth={selected ? 4 : 3}
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
                    fill={line.color}
                  />
                )}
                <text
                  x={tx}
                  y={box.y + 18}
                  textAnchor={station ? "middle" : "start"}
                  fill={INK}
                  fontSize="16"
                  fontWeight="750"
                  textLength={fitLabel(paper.shortTitle, 16, available)}
                  lengthAdjust="spacingAndGlyphs"
                >
                  {paper.shortTitle}
                </text>
                <text
                  x={tx}
                  y={box.y + 35}
                  textAnchor={station ? "middle" : "start"}
                  fill="#52605a"
                  fontSize="12.5"
                  fontWeight="550"
                  textLength={fitLabel(label, 12.5, available)}
                  lengthAdjust="spacingAndGlyphs"
                >
                  {label}
                </text>
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
        <span className="metro-map-count">
          {layout === "map"
            ? `VAU / ${METRO_LINES.length} 条线路 / ${papers.length} 篇论文`
            : `${papers.length} 篇论文`}
        </span>
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
