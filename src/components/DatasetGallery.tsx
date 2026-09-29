import { useEffect, useMemo, useState } from "react";
import {
  ArrowUpRight,
  ChevronLeft,
  ChevronRight,
  Film,
  Search,
} from "lucide-react";
import type { Dataset } from "../types";
import "./dataset-gallery.css";

interface Props {
  datasets: Dataset[];
  query: string;
  selectedId: string | null;
  onSelect: (id: string) => void;
}

export function datasetImageUrl(src: string) {
  return `${import.meta.env.BASE_URL}${src.replace(/^\/+/, "")}`;
}

export default function DatasetGallery({
  datasets,
  query,
  selectedId,
  onSelect,
}: Props) {
  const [task, setTask] = useState("all");
  const [page, setPage] = useState(0);
  const [compact, setCompact] = useState(() => window.innerWidth <= 760);
  const [unavailable, setUnavailable] = useState<Set<string>>(() => new Set());
  const perPage = compact ? 6 : 9;
  useEffect(() => {
    const media = window.matchMedia("(max-width: 760px)");
    const update = () => setCompact(media.matches);
    update();
    media.addEventListener("change", update);
    return () => media.removeEventListener("change", update);
  }, []);
  const tasks = useMemo(
    () => [...new Set(datasets.flatMap((d) => d.tasks))],
    [datasets],
  );
  const filtered = useMemo(
    () =>
      datasets.filter((d) => {
        const text =
          `${d.name} ${d.description} ${d.venue} ${d.year} ${d.tasks.join(" ")} ${d.annotations.join(" ")}`.toLowerCase();
        return (
          (task === "all" || d.tasks.includes(task)) &&
          query
            .trim()
            .toLowerCase()
            .split(/\s+/)
            .every((word) => text.includes(word))
        );
      }),
    [datasets, task, query],
  );
  useEffect(() => {
    setPage(0);
  }, [query, task, perPage]);
  const pages = Math.max(1, Math.ceil(filtered.length / perPage));
  const currentPage = Math.min(page, pages - 1);
  const visible = filtered.slice(
    currentPage * perPage,
    (currentPage + 1) * perPage,
  );
  return (
    <section className="dataset-exhibition" aria-label="数据集展览">
      <div className="exhibition-bar">
        <span className="exhibition-edition">
          DATA EXHIBITION <b>{String(filtered.length).padStart(2, "0")}</b>
        </span>
        <label>
          任务
          <select
            value={task}
            onChange={(event) => setTask(event.target.value)}
            aria-label="筛选数据集任务"
          >
            <option value="all">全部</option>
            {tasks.map((t) => (
              <option key={t}>{t}</option>
            ))}
          </select>
        </label>
      </div>
      <div className="exhibition-grid">
        {visible.map((dataset, i) => (
          <button
            key={dataset.id}
            className={`exhibit ${selectedId === dataset.id ? "is-selected" : ""}`}
            onClick={() => onSelect(dataset.id)}
            aria-label={`${dataset.name}，${dataset.venue} ${dataset.year}，查看数据集`}
          >
            <div className="exhibit-image">
              {!unavailable.has(dataset.id) ? (
                <img
                  src={datasetImageUrl(dataset.thumbnail.src)}
                  alt={dataset.thumbnail.alt}
                  loading="lazy"
                  decoding="async"
                  onError={() =>
                    setUnavailable(
                      (previous) => new Set([...previous, dataset.id]),
                    )
                  }
                />
              ) : (
                <span className="exhibit-placeholder">
                  <Film size={30} />
                  <span>{dataset.name}</span>
                </span>
              )}
              <span className="exhibit-number">
                {String(currentPage * perPage + i + 1).padStart(2, "0")}
              </span>
              <span className="exhibit-open">
                <ArrowUpRight size={17} />
              </span>
            </div>
            <div className="exhibit-label">
              <div className="exhibit-title">
                <h2>{dataset.name}</h2>
                <span>{dataset.year}</span>
              </div>
              <p className="exhibit-venue">{dataset.venue}</p>
              <div className="exhibit-tags">
                {dataset.tasks.slice(0, 2).map((t) => (
                  <span key={t}>{t}</span>
                ))}
              </div>
            </div>
          </button>
        ))}
      </div>
      {!filtered.length && (
        <div className="exhibition-empty">
          <Search size={28} />
          <span>暂无匹配的数据集</span>
          {task !== "all" && (
            <button onClick={() => setTask("all")}>全部任务</button>
          )}
        </div>
      )}
      <div className="exhibition-pagination">
        <span>
          {filtered.length
            ? `${currentPage * perPage + 1}–${Math.min((currentPage + 1) * perPage, filtered.length)} / ${filtered.length}`
            : "0 / 0"}
        </span>
        <div aria-label="数据展览分页">
          <button
            disabled={currentPage === 0}
            onClick={() => setPage(currentPage - 1)}
            aria-label="数据展览上一页"
          >
            <ChevronLeft size={17} />
          </button>
          <b>
            {currentPage + 1} / {pages}
          </b>
          <button
            disabled={currentPage + 1 >= pages}
            onClick={() => setPage(currentPage + 1)}
            aria-label="数据展览下一页"
          >
            <ChevronRight size={17} />
          </button>
        </div>
        <span>{perPage} / 页</span>
      </div>
    </section>
  );
}
