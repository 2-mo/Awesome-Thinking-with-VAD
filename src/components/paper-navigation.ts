import type { PublicationLine } from "./publication-layout";

/** Follow drawn edges only: disconnected segments of one color stay separate. */
export function adjacentPapers(line: Pick<PublicationLine, "paperRoutes">, paperId: string) {
  const previous = new Set<string>(), next = new Set<string>();
  for (const route of line.paperRoutes) {
    const index = route.indexOf(paperId);
    if (index > 0) previous.add(route[index - 1]);
    if (index >= 0 && index < route.length - 1) next.add(route[index + 1]);
  }
  return { previous: [...previous], next: [...next] };
}
