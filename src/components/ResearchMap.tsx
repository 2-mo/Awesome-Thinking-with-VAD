import { useId, useMemo, useRef, useState } from "react";
import type { PointerEvent as ReactPointerEvent } from "react";
import type { Cluster, Paper } from "../types";
import {
  createPublicationLayout,
  labelWidth,
  stationName,
  stationVenue,
  stationVenueSize,
} from "./publication-layout";
import type { Point, Station } from "./publication-layout";
import { clusterName, contributionLabels, paperContribution, paperMethods } from "../publication";
import type { ContributionKind } from "../publication";
import { metroPath } from "./metro-path";
import { createResearchBackdrop } from "./research-regions";
import { createMapLegend, LEGEND_ROUTE_TOP, LEGEND_ROUTE_STEP, legendDivider } from "./map-legend";
import { createMapRouteLabels, ROUTE_LABEL_SIZE } from "./map-route-labels";
import "./research-map.css";

interface Props {
  papers: Paper[];
  allPapers?: Paper[];
  clusters: Cluster[];
  selectedId: string | null;
  onSelect: (id: string) => void;
  activeCluster: string;
  onReset: () => void;
}
const INK = "#243b3b";
const PAPER = "#fffcf5";
const clamp = (n: number, min: number, max: number) =>
  Math.min(max, Math.max(min, n));
const fitLabel = (text: string, size: number, available: number) =>
  labelWidth(text, size) > available ? available : undefined;

function StationShape({ kind, x = 0, y = 0, radius, fill, stroke, strokeWidth, className }: {
  kind: ContributionKind; x?: number; y?: number; radius: number;
  fill: string; stroke?: string; strokeWidth?: number; className?: string;
}) {
  return kind === "method"
    ? <circle className={className} cx={x} cy={y} r={radius} fill={fill} stroke={stroke} strokeWidth={strokeWidth} />
    : <rect className={className} x={x - radius} y={y - radius} width={radius * 2} height={radius * 2}
      rx={radius * .16} fill={fill} stroke={stroke} strokeWidth={strokeWidth} />;
}

function StationMarker({ station, colors, kind, active = false, muted = false }: {
  station: Station; colors: Map<string, string>; kind: ContributionKind; active?: boolean; muted?: boolean;
}) {
  const color = colors.get(station.lineId) || INK;
  const first = station.platforms[0], last = station.platforms.at(-1)!;
  const interchange = station.platforms.length > 1 && !station.fork && !station.continuation;
  const connector = `M${station.x} ${first.y}V${last.y}`;
  return (
    <g className={station.fork ? "metro-fork" : interchange ? "metro-interchange" : undefined}>
      {station.fork ? (
        <g transform={`translate(${station.x} ${station.y})`}>
          {active && <circle r="18" fill={color} opacity=".16" />}
          <StationShape kind={kind} radius={muted ? 6 : 11} fill={PAPER} stroke={muted ? "#b1b9ae" : INK} strokeWidth={muted ? 2 : 3} />
          {!muted && <>
            <path d="M-5 0H5" stroke={colors.get(station.fork.parentId)} strokeWidth="2.5" strokeLinecap="round" />
            <path d={`M0 0l5 ${[...colors.keys()].indexOf(station.fork.branchId) < [...colors.keys()].indexOf(station.fork.parentId) ? -5 : 5}`}
              stroke={colors.get(station.fork.branchId)} strokeWidth="2.5" strokeLinecap="round" />
            {kind === "hybrid" && <circle r="2.5" fill={INK} />}
          </>}
        </g>
      ) : interchange ? (
        <>
          {active && <path d={connector} stroke={color} strokeWidth="36" strokeLinecap="round" opacity=".14" />}
          <path d={connector} stroke={PAPER} strokeWidth="22" strokeLinecap="round" />
          <path d={connector} stroke={muted ? "#b1b9ae" : INK} strokeWidth="7" strokeLinecap="round" />
          <path d={connector} stroke={PAPER} strokeWidth="3" strokeLinecap="round" />
          {station.platforms.map((platform) => (
            <g key={platform.lineId}>
              <StationShape kind={kind} x={platform.x} y={platform.y} radius={muted ? 5 : 8}
                fill={PAPER} stroke={muted ? "#b1b9ae" : INK} strokeWidth={muted ? 2 : 2.8} />
              {!muted && kind !== "resource" && <circle cx={platform.x} cy={platform.y} r={active ? 3.6 : 3}
                fill={colors.get(platform.lineId) || INK} />}
            </g>
          ))}
        </>
      ) : muted ? (
        <g>
          <StationShape kind={kind} x={station.x} y={station.y} radius={3.5} fill={PAPER} stroke="#b1b9ae" strokeWidth={1.5} />
          {kind === "hybrid" && <circle cx={station.x} cy={station.y} r="1.3" fill="#b1b9ae" />}
        </g>
      ) : (
        <>
          <circle className="metro-station-halo" cx={station.x} cy={station.y}
            r={active ? 16 : 11} fill={color} fillOpacity={active ? .16 : 0} />
          <StationShape kind={kind} x={station.x} y={station.y} radius={active ? 12.5 : 11} fill={PAPER} />
          <StationShape kind={kind} className="metro-station-dot" x={station.x} y={station.y}
            radius={active ? 9.5 : 8} fill={active && kind !== "resource" ? color : PAPER} stroke={color} strokeWidth={3.2} />
          {kind !== "resource" && <circle cx={station.x} cy={station.y}
            r={kind === "hybrid" ? 3.2 : active ? 3 : 2.3} fill={active ? PAPER : color} />}
        </>
      )}
      {!muted && <rect x={station.x - 18} y={first.y - 18} width="36"
        height={last.y - first.y + 36} rx="18" fill="transparent" />}
    </g>
  );
}

export default function ResearchMap({
  papers,
  allPapers,
  clusters,
  selectedId,
  onSelect,
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
  const regionsClipId = `research-regions-${id}`;
  const sourcePapers = allPapers || papers;
  const visibleIds = useMemo(() => new Set(papers.map((p) => p.id)), [papers]);
  const network = useMemo(
    () => createPublicationLayout(sourcePapers, clusters),
    [sourcePapers, clusters],
  );
  const { width } = network;
  const routeLabels = useMemo(() => createMapRouteLabels(network), [network]);
  const legend = useMemo(() => createMapLegend(network, routeLabels), [network, routeLabels]);
  const height = Math.max(network.plotBounds.y + network.plotBounds.height + 28, legend.y + legend.height + 24);
  const backdrop = useMemo(() => createResearchBackdrop(network, [legend, ...routeLabels]), [network, legend, routeLabels]);
  const lineColors = new Map(network.lines.map((line) => [line.id, line.color]));
  const focusedPaper = sourcePapers.find((paper) => paper.id === (hoveredId || selectedId));
  const focusedLines = new Set(focusedPaper ? paperMethods(focusedPaper) :
    activeCluster === "all" ? [] : [activeCluster]);
  for (const method of [...focusedLines]) {
    const ancestors = new Set([method]);
    let parent = clusters.find(c => c.id === method)?.branchOf;
    while (parent && !ancestors.has(parent)) {
      ancestors.add(parent);
      focusedLines.add(parent);
      parent = clusters.find(c => c.id === parent)?.branchOf;
    }
  }
  const left = network.years[0]?.x ?? 190;
  const plotTop = network.plotBounds.y;
  const plotBottom = plotTop + network.plotBounds.height;

  const resetView = () => {
    setPan({ x: 0, y: 0 });
    setZoom(1);
  };
  const changeZoom = (factor: number) =>
    setZoom((value) => clamp(Number((value * factor).toFixed(2)), 0.7, 6));
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
      anchor.download = "vau-research-route-map.svg";
      anchor.click();
      window.setTimeout(() => URL.revokeObjectURL(url), 1000);
    } catch {
      setExportFailed(true);
    }
  };

  return (
    <section
      className={`research-map metro-map publication-map${dragging ? " is-dragging" : ""}`}
      aria-label="Research route map by time and method"
    >
      <svg
        ref={svgRef}
        className="research-map-canvas"
        viewBox={`0 0 ${width} ${height}`}
        aria-label={`${papers.length} papers by year and quarter. Circles: methods. Squares: datasets or benchmarks. Squares with dots: combined contributions. Y junctions: forks. Linked platforms: multiple research directions.`}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={endDrag}
        onPointerCancel={endDrag}
        style={{
          fontFamily:
            '-apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif',
        }}
      >
        <title>Video Anomaly Understanding · Research Route Map</title>
        <desc>Research directions are named beside their colored routes. Circles mark methods, squares mark datasets or benchmarks, and squares with dots mark combined contributions. {legend.embedded ? "The station key occupies clear space on the left." : "The station key appears below the map."}</desc>
        <defs>
          <clipPath id={regionsClipId}>
            <rect x={network.plotBounds.x} y={plotTop} width={network.plotBounds.width} height={network.plotBounds.height} />
          </clipPath>
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
          <>
              <g className="research-regions" pointerEvents="none" clipPath={`url(#${regionsClipId})`}>
                <g data-terrain="mountains" aria-hidden="true" fill="none" stroke="#c0c7b8" strokeWidth="1.1" opacity=".72"
                  strokeLinecap="round" strokeLinejoin="round">
                  {backdrop.mountains.map((point, index) => (
                    <g key={index} transform={`translate(${point.x} ${point.y}) scale(${point.width / 172} ${point.height / 52})`}>
                        <path d="M0 44l25-24 19 15L74 2l29 35 22-19 41 29" />
                        <path d="M57 21l10 3 7-8 8 7 8-2M74 16l-5 28M115 27l8 5 9-5" />
                        <path d="M15 50q20-8 39-1m42 1q24-5 53 1" />
                    </g>
                  ))}
                </g>
              </g>
              <rect
                x={left}
                y={plotTop}
                width={width - left - 24}
                height={plotBottom - plotTop}
                fill={`url(#${patternId})`}
              />
              {network.years.map((year) => (
                <g key={year.year}>
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
                        {quarter.quarter === null ? "TBD" : `Q${quarter.quarter}`}
                      </text>
                    </g>
                  ))}
                </g>
              ))}
              {network.lines.map((line) => {
                const focus = !focusedLines.size || focusedLines.has(line.id);
                const path = line.tracks.map(track => metroPath(
                  track,
                  [...network.junctions, ...[...network.stations.values()].flatMap(
                    (station) => station.platforms.filter((platform) => platform.lineId === line.id),
                  )],
                  [...network.stations.values()].map((station) => station.label),
                )).join(" ");
                return (
                  <g
                    key={line.id}
                    className="publication-route"
                    data-line={line.id}
                    data-branch-of={line.branchOf}
                    opacity={focus ? 0.95 : 0.17}
                  >
                    <path
                      d={path}
                      fill="none"
                      stroke={PAPER}
                      strokeWidth={focus && focusedLines.size ? 16 : 14}
                      strokeLinejoin="round"
                      strokeLinecap="round"
                    />
                    <path
                      d={path}
                      fill="none"
                      stroke={line.color}
                      strokeWidth={focus && focusedLines.size ? 6 : 4.8}
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
                  return point && (
                    <g key={paper.id} pointerEvents="none">
                      <StationMarker station={point} colors={lineColors} kind={paperContribution(paper)} muted />
                    </g>
                  );
                })}
              {routeLabels.map(label => <text key={label.lineId} data-route-label={label.lineId}
                x={label.x} y={label.y + 18} fill={label.color} fontSize={ROUTE_LABEL_SIZE} fontWeight="600"
                opacity={!focusedLines.size || focusedLines.has(label.lineId) ? 1 : .22} pointerEvents="none">
                {label.text}
              </text>)}
              <g data-map-legend="true" data-embedded={legend.embedded} transform={`translate(${legend.x} ${legend.y})`} pointerEvents="none">
                <rect width={legend.width} height={legend.height} rx="8" fill={PAPER} />
                {!!legend.fallbackLineIds.length && <text x="16" y="20" fill="#65726a" fontSize="14" fontWeight="750" letterSpacing="1.3">RESEARCH DIRECTIONS</text>}
              {network.lines.filter(line => legend.fallbackLineIds.includes(line.id)).map((line, index) => {
                const x = 16;
                const y = LEGEND_ROUTE_TOP + index * LEGEND_ROUTE_STEP;
                return (
                  <g key={line.id} transform={`translate(${x} ${y})`}>
                    <path
                      d={line.branchOf ? "M0 5h9l10 -10h11" : "M0 0h30"}
                      stroke={line.color}
                      strokeWidth="4"
                      strokeLinecap="round"
                    />
                    {!line.branchOf && <circle
                      cx="15"
                      r="6.5"
                      fill={PAPER}
                      stroke={line.color}
                      strokeWidth="2.8"
                    />}
                    {!line.branchOf && <circle cx="15" r="2" fill={line.color} />}
                    <text
                      x="41"
                      y="5"
                      fill={INK}
                      fontSize="18"
                      fontWeight="700"
                    >
                      {line.label}{line.branchOf && <tspan fontSize="14" fill="#65726a"> · Branch</tspan>}
                    </text>
                  </g>
                );
              })}
              <g data-contribution-legend="true" transform={`translate(16 ${legendDivider(legend.fallbackLineIds.length)})`}>
                {!!legend.fallbackLineIds.length && <path d={`M0 0H${legend.width - 32}`} stroke="#d9dfd3" strokeWidth="1" />}
                <text x="0" y="24" fill="#65726a" fontSize="14" fontWeight="750" letterSpacing="1.3">STATION TYPES</text>
                {(["method", "resource", "hybrid"] as const).map((kind, index) => (
                  <g key={kind} transform={`translate(${index === 1 ? 300 : 8} ${index === 2 ? 86 : 54})`}>
                    <StationShape kind={kind} radius={8} fill={PAPER} stroke={INK} strokeWidth={2.6} />
                    {kind !== "resource" && <circle r={kind === "hybrid" ? 3.2 : 2.3} fill={INK} />}
                    <text x="20" y="5" fill={INK} fontSize="16" fontWeight="650">{contributionLabels[kind]}</text>
                  </g>
                ))}
                <g transform="translate(300 86)">
                  <path d="M0 -7V7" stroke={INK} strokeWidth="5" strokeLinecap="round" />
                  <path d="M0 -7V7" stroke={PAPER} strokeWidth="2" strokeLinecap="round" />
                  {[-7, 7].map(y => <g key={y}>
                    <circle cy={y} r="5" fill={PAPER} stroke={INK} strokeWidth="2" />
                    <circle cy={y} r="1.7" fill={INK} />
                  </g>)}
                  <text x="20" y="5" fill={INK} fontSize="16" fontWeight="650">Interchange</text>
                </g>
              </g>
              </g>
            </>
          {papers.map((paper) => {
            const station = network.stations.get(paper.id);
            if (!station) return null;
            const box = station.label;
            const selected = selectedId === paper.id;
            const publicationDetails = `${paper.title}, ${paper.venue}, ${paper.year}`;
            const active = selected || hoveredId === paper.id;
            const line = network.lines.find(
              (item) => item.id === paper.cluster,
            );
            const color = line?.color || INK;
            const fontSize = 20;
            const interchange = station.lineIds.length > 1 && !station.fork && !station.continuation;
            const name = stationName(paper);
            const kind = paperContribution(paper);
            const tx = box.x + box.width / 2;
            const available = box.width - 10;
            const labelY = box.y + 21;
            return (
              <g
                key={paper.id}
                role="button"
                tabIndex={0}
                aria-label={`${publicationDetails}, ${contributionLabels[kind]}, ${paperMethods(paper).map((id) => { const cluster = clusters.find((item) => item.id === id); return cluster ? clusterName(cluster) : id; }).join(", ")}${station.fork ? ", fork" : interchange ? ", interchange" : ""}. View paper details.`}
                aria-pressed={selected}
                data-paper={paper.id}
                data-contribution={kind}
                data-fork={station.fork?.branchId}
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
                <title>{`${publicationDetails} — ${contributionLabels[kind]}${station.fork ? " · Fork" : interchange ? " · Interchange" : ""}`}</title>
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
                <StationMarker station={station} colors={lineColors} kind={kind} active={active} />
                <text
                  x={tx}
                  y={labelY}
                  textAnchor="middle"
                  fill={INK}
                  fontSize={fontSize}
                  fontWeight="750"
                  textLength={fitLabel(name, fontSize, available)}
                  lengthAdjust="spacingAndGlyphs"
                >
                  {name}
                </text>
                <text
                    x={tx}
                    y={box.y + 41}
                    textAnchor="middle"
                    fill="#52605a"
                    fontSize={stationVenueSize(!!station.fork)}
                    fontWeight="650"
                    textLength={fitLabel(stationVenue(paper, !!station.fork), stationVenueSize(!!station.fork), available)}
                    lengthAdjust="spacingAndGlyphs"
                  >
                    {stationVenue(paper, !!station.fork)}
                  </text>
              </g>
            );
          })}
        </g>
      </svg>
      {papers.length === 0 && (
        <div className="metro-empty">
          <strong>No matching papers</strong>
          <button type="button" onClick={onReset}>
            Clear filters ↗
          </button>
        </div>
      )}
      <div className="metro-map-bottom">
        <span className="metro-map-count">{papers.length} papers</span>
        <div className="metro-controls" aria-label="Map controls">
          <button
            type="button"
            onClick={() => changeZoom(1 / 1.25)}
            disabled={zoom <= 0.7}
            aria-label="Zoom out"
          >
            −
          </button>
          <button
            type="button"
            className="metro-zoom"
            onClick={resetView}
            aria-label="Reset to the full map"
            title="Reset view"
          >
            {Math.round(zoom * 100)}%
          </button>
          <button
            type="button"
            onClick={() => changeZoom(1.25)}
            disabled={zoom >= 6}
            aria-label="Zoom in"
          >
            +
          </button>
          <button
            type="button"
            className="metro-export"
            onClick={exportSvg}
            title="Export the full map as SVG"
          >
            {exportFailed ? "Retry SVG" : "SVG ↗"}
          </button>
        </div>
      </div>
    </section>
  );
}
