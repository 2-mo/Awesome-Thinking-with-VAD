import type { Cluster, Paper } from "./types";

export type ContributionKind = "method" | "resource" | "hybrid";
export const contributionLabels: Record<ContributionKind, string> = {
  method: "Method",
  resource: "Dataset / Benchmark",
  hybrid: "Method + Data / Benchmark",
};
export const clusterName = (cluster: Pick<Cluster, "name" | "nameEn">): string =>
  cluster.nameEn || cluster.name;
export function paperContribution(paper: Pick<Paper, "contribution">): ContributionKind {
  return paper.contribution?.kind ?? "method";
}

/** Editorial exclusions and supplementary venues stay in the reading indexes. */
export function isMapPaper(paper: Pick<Paper, "venue" | "mapExclusion">): boolean {
  return paper.mapExclusion === undefined && !/\bfindings?\b|\bworkshops?\b|\bWACV\b|\b(?:CVPR|ICCV|ECCV|WACV)W\b/i.test(paper.venue);
}

/** Combine publication tracks for navigation while retaining exact paper metadata. */
export function publicationVenue(venue: string): string {
  return venue === "NeurIPS Datasets and Benchmarks" || venue === "NeurIPS Evaluations and Datasets" ? "NeurIPS" : venue;
}

/** Shared stations retain one primary method and sourced secondary memberships. */
export function paperMethods(paper: Paper): string[] {
  return [paper.cluster, ...(paper.secondaryMethods ?? []).map((method) => method.cluster)];
}
