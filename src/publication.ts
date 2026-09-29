import type { Paper } from "./types";

/** Combine publication tracks for navigation while retaining exact paper metadata. */
export function publicationVenue(venue: string): string {
  return venue === "NeurIPS Datasets and Benchmarks" ? "NeurIPS" : venue;
}

/** Shared stations retain one primary method and sourced secondary memberships. */
export function paperMethods(paper: Paper): string[] {
  return [paper.cluster, ...(paper.secondaryMethods ?? []).map((method) => method.cluster)];
}
