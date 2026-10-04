import type { Cluster, Paper } from "./types";

export type ContributionKind = "method" | "resource" | "hybrid";
export const mapIconLabels = {
  industry: "Industrial image anomalies",
  road: "Road / traffic anomaly research",
  video: "Video anomaly detection",
};
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

export type PublicationKind = "conference" | "journal" | "preprint";
/** AAAI proceedings use BibTeX article entries, so citation.type is not a venue type. */
export function publicationKind(paper: Pick<Paper, "venue" | "timeline">): PublicationKind {
  if (/^arxiv$/i.test(paper.venue)) return "preprint";
  return /^(TPAMI|IJCV|TIP|TNNLS|TCSVT|TCYB|TIFS|TMLR)$/.test(paper.venue) || paper.timeline?.basis === "journal"
    ? "journal" : "conference";
}

/** Shared stations retain one primary method and sourced secondary memberships. */
export function paperMethods(paper: Paper): string[] {
  return [paper.cluster, ...(paper.secondaryMethods ?? []).map((method) => method.cluster)];
}

/** Map chronology may use an earlier online publication year than the journal issue. */
export const timelineYear = (paper: Pick<Paper, "year" | "timeline">): number => paper.timeline?.year ?? paper.year;
