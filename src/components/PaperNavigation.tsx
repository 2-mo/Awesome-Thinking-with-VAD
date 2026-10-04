import type { Paper } from "../types";
import type { PublicationLine } from "./publication-layout";
import { adjacentPapers } from "./paper-navigation";

interface Props {
  paper: Paper;
  papers: Map<string, Paper>;
  lines: PublicationLine[];
  lineId: string;
  onLineChange: (id: string) => void;
  onNavigate: (id: string, lineId: string) => void;
}

export default function PaperNavigation({ paper, papers, lines, lineId, onLineChange, onNavigate }: Props) {
  const available = lines.filter(line => line.paperRoutes.some(route => route.includes(paper.id)));
  const line = available.find(line => line.id === lineId)
    ?? available.find(line => line.id === paper.cluster) ?? available[0];
  if (!line) return null;
  const adjacent = adjacentPapers(line, paper.id);
  return <nav className="paper-navigation" aria-label="沿线路阅读" style={{ borderLeftColor: line.color }}>
    <div className="paper-navigation-heading">
      <span>沿线路阅读</span>
      {available.length > 1
        ? <select aria-label="阅读方向" value={line.id} onChange={event => onLineChange(event.target.value)}>
          {available.map(item => <option key={item.id} value={item.id}>{item.label}</option>)}
        </select>
        : <strong>{line.label}</strong>}
    </div>
    <div className="paper-navigation-steps">
      {(["previous", "next"] as const).map(direction => {
        const choices = adjacent[direction].map(id => papers.get(id)).filter((p): p is Paper => !!p);
        const label = direction === "previous" ? "上一篇" : "下一篇";
        return choices.length > 1
          ? <label className="paper-navigation-branch" key={direction}>
            <span>{direction === "previous" ? "← 上一篇" : "下一篇 →"}</span>
            <select aria-label={`选择${label}`} value="" onChange={event => onNavigate(event.target.value, line.id)}>
              <option value="" disabled>选择分支（{choices.length}）</option>
              {choices.map(choice => <option key={choice.id} value={choice.id}>{choice.shortTitle}</option>)}
            </select>
          </label>
          : <button type="button" key={direction} disabled={!choices.length}
            aria-label={choices.length ? `${label}：${choices[0].shortTitle}` : `${label}：已到本段${direction === "previous" ? "起点" : "终点"}`}
            title={choices[0]?.title} onClick={() => onNavigate(choices[0].id, line.id)}>
            <span>{direction === "previous" ? "← 上一篇" : "下一篇 →"}</span>
            <strong>{choices[0]?.shortTitle ?? `已到本段${direction === "previous" ? "起点" : "终点"}`}</strong>
          </button>;
      })}
    </div>
  </nav>;
}
