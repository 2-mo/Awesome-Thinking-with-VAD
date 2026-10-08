import { useId, useMemo, useRef, useState } from "react";
import type { PointerEvent as ReactPointerEvent } from "react";
import type { Cluster, Paper } from "../types";
import {
  isInterchangeStation,
  labelWidth,
  stationName,
  stationVenue,
  stationVenueSize,
  STATION_NAME_SIZE,
  STATION_ICON_SIZE,
} from "./publication-layout";
import type { Point, Station, PublicationLayout } from "./publication-layout";
import { clusterName, contributionLabels, mapIconLabels, paperContribution, paperMethods, publicationKind } from "../publication";
import type { ContributionKind, PublicationKind } from "../publication";
import { metroPath } from "./metro-path";
import { createResearchBackdrop } from "./research-regions";
import { createMapLegend, LEGEND_ROUTE_TOP, LEGEND_ROUTE_STEP, LEGEND_ITEMS, LEGEND_ROW_CENTER, legendDivider } from "./map-legend";
import { createMapRouteLabels, ROUTE_LABEL_SIZE } from "./map-route-labels";
import { MAP_FONT_SIZE, MAP_FONT_FAMILY, MAP_RAIL_WIDTH } from "./map-typography";
import "./research-map.css";

interface Props {
  network: PublicationLayout;
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
// IEEE titles share cool tints; Springer IJCV uses a warmer family.
const JOURNAL_TINTS: Record<string, string> = {
  TPAMI: "#e5edf6", TIP: "#e9f1f7", TNNLS: "#ececf7",
  TCSVT: "#e8eef9", TCYB: "#e8edf5", TIFS: "#e5f0f4",
  IJCV: "#f3e8da",
};
const clamp = (n: number, min: number, max: number) =>
  Math.min(max, Math.max(min, n));
const fitLabel = (text: string, size: number, available: number) =>
  labelWidth(text, size) > available ? available : undefined;

function mapSvgBlob(svg: SVGSVGElement, width: number, height: number) {
  const clone = svg.cloneNode(true) as SVGSVGElement;
  clone.setAttribute("xmlns", "http://www.w3.org/2000/svg");
  clone.setAttribute("width", String(width));
  clone.setAttribute("height", String(height));
  clone.querySelector("[data-map-content]")?.removeAttribute("transform");
  return new Blob([new XMLSerializer().serializeToString(clone)], {
    type: "image/svg+xml;charset=utf-8",
  });
}

function downloadMap(blob: Blob, extension: "svg" | "png") {
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  try {
    anchor.href = url;
    anchor.download = `vau-research-route-map.${extension}`;
    document.body.append(anchor);
    anchor.click();
  } finally {
    anchor.remove();
    window.setTimeout(() => URL.revokeObjectURL(url), 1000);
  }
}

function PaperContextIcon({ kind, x, y }: { kind: NonNullable<Paper["mapIcon"]>["kind"]; x: number; y: number }) {
  return <g data-context-icon={kind} transform={`translate(${x - STATION_ICON_SIZE / 2} ${y - STATION_ICON_SIZE / 2}) scale(${STATION_ICON_SIZE / 24})`}
    fill="none" stroke="#89918c" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
    <title>{mapIconLabels[kind]}</title>
    {kind === "industry" ? <>
      <path d={`${Array.from({ length: 32 }, (_, i) => {
        const radius = i % 4 < 2 ? 11 : 8.6, angle = (i - .5) * Math.PI / 16;
        return `${i ? "L" : "M"}${12 + radius * Math.cos(angle)} ${12 + radius * Math.sin(angle)}`;
      }).join(" ")}Z M15.4 12a3.4 3.4 0 1 0-6.8 0 3.4 3.4 0 1 0 6.8 0Z`}
        fill="none" />
    </> : kind === "road" ? <>
      <path d="M7 3 3 21M17 3l4 18M12 3v3m0 4v4m0 4v3" />
    </> : <>
      <rect x="2" y="4" width="20" height="16" rx="3" />
      <path d="m10 8 6 4-6 4Z" fill="#89918c" stroke="none" />
    </>}
  </g>;
}

function StationShape({ kind, x = 0, y = 0, radius, fill, stroke, strokeWidth, className }: {
  kind: ContributionKind; x?: number; y?: number; radius: number;
  fill: string; stroke?: string; strokeWidth?: number; className?: string;
}) {
  return kind === "method"
    ? <circle className={className} cx={x} cy={y} r={radius} fill={fill} stroke={stroke} strokeWidth={strokeWidth} />
    : <rect className={className} x={x - radius} y={y - radius} width={radius * 2} height={radius * 2}
      rx={radius * .16} fill={fill} stroke={stroke} strokeWidth={strokeWidth} />;
}

function VenueLabel({ venue, kind, x, y, available = Infinity }: {
  venue: string; kind: PublicationKind; x: number; y: number; available?: number;
}) {
  const size = stationVenueSize();
  const width = Math.min(labelWidth(venue, size) + 8, available);
  return <g data-publication-venue={kind}>
    <title>{kind === "journal" ? "Journal" : kind === "preprint" ? "Preprint" : "Conference"}</title>
    {kind === "journal" && <rect x={x - width / 2} y={y - size - 1} width={width} height={size + 6} rx="3"
      fill={JOURNAL_TINTS[venue] ?? "#edf1ed"} stroke="none" />}
    <text x={x} y={y} textAnchor="middle" fill={kind === "preprint" ? "#747474" : "#52605a"}
      fontSize={size} fontWeight={kind === "conference" ? "650" : kind === "journal" ? "500" : "400"}
      fontStyle={kind === "preprint" ? "italic" : undefined}
      textLength={fitLabel(venue, size, kind === "journal" ? width - 8 : available)} lengthAdjust="spacingAndGlyphs">
      {venue}
    </text>
  </g>;
}

function StationMarker({ station, colors, kind, active = false, muted = false }: {
  station: Station; colors: Map<string, string>; kind: ContributionKind; active?: boolean; muted?: boolean;
}) {
  const color = colors.get(station.lineId) || INK;
  const first = station.platforms[0], last = station.platforms.at(-1)!;
  const interchange = isInterchangeStation(station);
  const connector = `M${station.x} ${first.y}V${last.y}`;
  return (
    <g className={interchange ? "metro-interchange" : undefined}>
      {interchange ? (
        <>
          {active && <path d={connector} stroke={color} strokeWidth="36" strokeLinecap="round" opacity=".14" />}
          <path d={connector} stroke={PAPER} strokeWidth="22" strokeLinecap="round" />
          <path d={connector} stroke={muted ? "#b1b9ae" : INK} strokeWidth="7" strokeLinecap="round" />
          <path d={connector} stroke={PAPER} strokeWidth="3" strokeLinecap="round" />
          {station.platforms.map((platform) => (
            <g key={platform.lineId}>
              <StationShape kind={kind} x={platform.x} y={platform.y} radius={muted ? 5 : 10}
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
          <StationShape kind={kind} x={station.x} y={station.y} radius={active ? 15 : 13} fill={PAPER} />
          <StationShape kind={kind} className="metro-station-dot" x={station.x} y={station.y}
            radius={active ? 12 : 10} fill={active && kind !== "resource" ? color : PAPER} stroke={color} strokeWidth={3.8} />
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
  network,
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
  const [exportingPng, setExportingPng] = useState(false);
  const [pngExportFailed, setPngExportFailed] = useState(false);
  const id = useId().replace(/[^a-zA-Z0-9_-]/g, "");
  const patternId = `publication-grid-${id}`;
  const regionsClipId = `research-regions-${id}`;
  const sourcePapers = allPapers || papers;
  const visibleIds = useMemo(() => new Set(papers.map((p) => p.id)), [papers]);
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
      downloadMap(mapSvgBlob(svgRef.current, width, height), "svg");
    } catch {
      setExportFailed(true);
    }
  };
  const exportPng = async () => {
    if (!svgRef.current || exportingPng) return;
    setExportingPng(true);
    setPngExportFailed(false);
    let url: string | undefined;
    try {
      url = URL.createObjectURL(mapSvgBlob(svgRef.current, width, height));
      const image = new Image();
      await new Promise<void>((resolve, reject) => {
        image.onload = () => resolve();
        image.onerror = () => reject(new Error("Could not load the map image"));
        image.src = url!;
      });
      const canvas = document.createElement("canvas");
      canvas.width = Math.ceil(width);
      canvas.height = Math.ceil(height);
      const context = canvas.getContext("2d");
      if (!context) throw new Error("Canvas is unavailable");
      context.drawImage(image, 0, 0, canvas.width, canvas.height);
      const blob = await new Promise<Blob>((resolve, reject) => {
        canvas.toBlob((result) => {
          if (result) resolve(result);
          else reject(new Error("Could not encode the map as PNG"));
        }, "image/png");
      });
      downloadMap(blob, "png");
    } catch {
      setPngExportFailed(true);
    } finally {
      if (url) URL.revokeObjectURL(url);
      setExportingPng(false);
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
        aria-label={`${papers.length} papers arranged by year and approximate publication order; spacing follows research routes, not a month scale. Circles: methods. Squares: datasets or benchmarks. Squares with dots: combined contributions. Linked platforms: multiple research directions. Venue text: semibold conferences, journals on a pale background, and gray italic preprints.`}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={endDrag}
        onPointerCancel={endDrag}
        style={{
          fontFamily: MAP_FONT_FAMILY,
        }}
      >
        <title>Visual Anomaly Understanding · Research Route Map</title>
        <desc>Year bands preserve publication years; positions within each year show approximate order with spacing chosen for readability. Research directions are named beside their colored routes. Circles mark methods, squares mark datasets or benchmarks, and squares with dots mark combined contributions. Branch departures and returns use ordinary markers. Venue names have no borders: semibold text for conferences, regular text on a pale background for journals, and gray italic text for preprints. {legend.embedded ? "The horizontal station key occupies clear space in the bottom-left corner." : "The station key appears below the map."}</desc>
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
                <g data-terrain="mountains" aria-hidden="true" fill="none" stroke="#b1bca9" strokeWidth="1" opacity=".42"
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
                    fontSize={MAP_FONT_SIZE.primary}
                    fontWeight="700"
                    letterSpacing="-.8"
                  >
                    {year.year}
                  </text>
                  <path
                    d={`M${year.x + 20} ${plotTop - 36}h${year.width - 40}`}
                    stroke={INK}
                    strokeWidth="2"
                  />
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
                      strokeWidth={focus && focusedLines.size ? MAP_RAIL_WIDTH + 10 : MAP_RAIL_WIDTH + 8}
                      strokeLinejoin="round"
                      strokeLinecap="round"
                    />
                    <path
                      d={path}
                      fill="none"
                      stroke={line.color}
                      strokeWidth={focus && focusedLines.size ? MAP_RAIL_WIDTH + 1.5 : MAP_RAIL_WIDTH}
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
                x={label.x} y={label.y + ROUTE_LABEL_SIZE} fill={label.color} fontSize={ROUTE_LABEL_SIZE} fontWeight="600"
                opacity={!focusedLines.size || focusedLines.has(label.lineId) ? 1 : .22} pointerEvents="none">
                {label.text}
              </text>)}
              <g data-map-legend="true" data-embedded={legend.embedded} transform={`translate(${legend.x} ${legend.y})`} pointerEvents="none">
                <rect width={legend.width} height={legend.height} rx="8" fill={PAPER} />
                {!!legend.fallbackLineIds.length && <text x="16" y="26" fill="#65726a" fontSize={MAP_FONT_SIZE.secondary} fontWeight="700" letterSpacing="1.3">RESEARCH DIRECTIONS</text>}
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
                      fontSize={ROUTE_LABEL_SIZE}
                      fontWeight="700"
                    >
                      {line.label}{line.branchOf && <tspan fontSize={MAP_FONT_SIZE.secondary} fill="#65726a"> · Branch</tspan>}
                    </text>
                  </g>
                );
              })}
              <g data-contribution-legend="true" transform={`translate(0 ${legendDivider(legend.fallbackLineIds.length)})`}>
                {!!legend.fallbackLineIds.length && <path d={`M16 0H${legend.width - 16}`} stroke="#d9dfd3" strokeWidth="1" />}
                <text x="16" y={LEGEND_ROW_CENTER + 7} fill="#8b948b" fontSize={24} fontWeight="500">STATION TYPES</text>
                {LEGEND_ITEMS.map(({ kind, x }) => (
                  <g key={kind} transform={`translate(${x} ${LEGEND_ROW_CENTER})`}>
                    <StationShape kind={kind} radius={7} fill={PAPER} stroke="#7b857f" strokeWidth={2} />
                    {kind !== "resource" && <circle r={kind === "hybrid" ? 2.8 : 2} fill="#7b857f" />}
                    <text x="20" y="10" fill="#7b857f" fontSize={32} fontWeight="500">{contributionLabels[kind]}</text>
                  </g>
                ))}
              </g>
              </g>
            </>
          {papers.map((paper) => {
            const station = network.stations.get(paper.id);
            if (!station) return null;
            const box = station.label;
            const selected = selectedId === paper.id;
            const publicationDetails = `${paper.title}, ${paper.venue}, ${paper.year}${paper.mapIcon ? `, ${mapIconLabels[paper.mapIcon.kind]}` : ""}`;
            const active = selected || hoveredId === paper.id;
            const line = network.lines.find(
              (item) => item.id === paper.cluster,
            );
            const color = line?.color || INK;
            const fontSize = STATION_NAME_SIZE;
            const interchange = isInterchangeStation(station);
            const name = stationName(paper);
            const kind = paperContribution(paper);
            const tx = box.x + box.width / 2;
            const available = box.width - 10;
            const iconSpace = paper.mapIcon ? STATION_ICON_SIZE + 8 : 0;
            const venueWidth = Math.min(labelWidth(stationVenue(paper), stationVenueSize()) + 8, available - iconSpace);
            const labelY = box.y + fontSize + 1;
            const venueY = box.y + box.height - 7;
            return (
              <g
                key={paper.id}
                role="button"
                tabIndex={0}
                aria-label={`${publicationDetails}, ${publicationKind(paper)}, ${contributionLabels[kind]}, ${paperMethods(paper).map((id) => { const cluster = clusters.find((item) => item.id === id); return cluster ? clusterName(cluster) : id; }).join(", ")}${interchange ? ", interchange" : ""}. View paper details.`}
                aria-pressed={selected}
                data-paper={paper.id}
                data-contribution={kind}
                data-publication-kind={publicationKind(paper)}
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
                <title>{`${publicationDetails} — ${contributionLabels[kind]}${interchange ? " · Interchange" : ""}\n${paper.mechanism}`}</title>
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
                  fontWeight="700"
                  textLength={fitLabel(name, fontSize, available)}
                  lengthAdjust="spacingAndGlyphs"
                >
                  {name}
                </text>
                <VenueLabel venue={stationVenue(paper)} kind={publicationKind(paper)} x={tx - iconSpace / 2} y={venueY} available={available - iconSpace} />
                {paper.mapIcon && <PaperContextIcon kind={paper.mapIcon.kind} x={tx + venueWidth / 2 + 4} y={venueY - stationVenueSize() * .375} />}
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
          <button
            type="button"
            className="metro-export"
            onClick={exportPng}
            disabled={exportingPng}
            aria-busy={exportingPng}
            title={pngExportFailed ? "PNG export failed. Click to retry." : "Export the full map as PNG"}
          >
            {exportingPng ? "Exporting…" : pngExportFailed ? "Retry PNG" : "PNG ↗"}
          </button>
        </div>
      </div>
    </section>
  );
}
