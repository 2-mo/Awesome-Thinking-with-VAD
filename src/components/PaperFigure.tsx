import { useEffect, useRef, useState } from "react";
import type { Paper } from "../types";

// Import URLs only: Vite emits separate files, never image bytes in the JS bundle.
const figureUrls = import.meta.glob<string>("../../assets/papers/*.{png,jpg,jpeg,webp}", {
  eager: true,
  query: "?url&no-inline",
  import: "default",
});

export default function PaperFigure({ figure }: { figure: NonNullable<Paper["figure"]> }) {
  const frame = useRef<HTMLDivElement>(null);
  const [nearViewport, setNearViewport] = useState(false);
  const [status, setStatus] = useState<"loading" | "loaded" | "error">("loading");
  const [attempt, setAttempt] = useState(0);
  const src = figureUrls[`../../${figure.src}`];

  useEffect(() => {
    const element = frame.current;
    if (!element) return;
    if (!("IntersectionObserver" in window)) {
      setNearViewport(true);
      return;
    }
    const observer = new IntersectionObserver(entries => {
      if (entries.some(entry => entry.isIntersecting)) {
        setNearViewport(true);
        observer.disconnect();
      }
    }, { root: element.closest("dialog"), rootMargin: "160px 0px" });
    observer.observe(element);
    return () => observer.disconnect();
  }, []);

  const failed = !src || status === "error";
  return <figure className="paper-figure">
    <div ref={frame} className="paper-figure-frame" aria-busy={nearViewport && !failed && status === "loading"}>
      {nearViewport && !failed && <a className="paper-figure-image" href={src} target="_blank" rel="noreferrer"
        aria-label="查看论文大图（新窗口）">
        <img key={attempt} src={src} alt={figure.alt} loading="lazy" decoding="async" fetchPriority="low"
          className={status === "loaded" ? "is-loaded" : undefined}
          onLoad={() => setStatus("loaded")} onError={() => setStatus("error")} />
      </a>}
      {status !== "loaded" && <div className="paper-figure-placeholder" role="status">
        <span>{failed ? "图片暂时无法加载" : nearViewport ? "正在加载论文图…" : "论文原图"}</span>
        {failed && src && <button type="button" onClick={() => {
          setAttempt(value => value + 1);
          setStatus("loading");
        }}>重新加载</button>}
        {failed && <a href={figure.sourceUrl} target="_blank" rel="noreferrer">查看原文图 ↗</a>}
      </div>}
    </div>
    <figcaption>
      <span>{figure.caption}</span>
      <span className="paper-figure-credit">{figure.credit}</span>
      <span className="paper-figure-links">
        {src && <a href={src} target="_blank" rel="noreferrer">查看大图 ↗</a>}
        <a href={figure.sourceUrl} target="_blank" rel="noreferrer">原图来源 ↗</a>
      </span>
    </figcaption>
  </figure>;
}
