export interface Source {
  url: string;
  note: string;
}
export interface Paper {
  id: string;
  shortTitle: string;
  title: string;
  year: number;
  venue: string;
  // Hidden ordering only: conference main-session month, or arXiv v1 month.
  timeline?: { month: number; basis: "conference" | "preprint"; source: Source };
  scope: "core";
  cluster: string;
  tasks: string[];
  summary: string;
  mechanism: string;
  takeaway: string;
  limitation: string;
  links: { paper: string; code?: string; project?: string };
  sources: Source[];
  verifiedAt: string;
  datasetIds: string[];
}
export interface Dataset {
  year: number;
  venue: string;
  thumbnail: { src: string; alt: string; sourceUrl: string; credit: string };
  id: string;
  name: string;
  description: string;
  tasks: string[];
  modalities: string[];
  annotations: string[];
  protocol: string;
  links: { website: string; paper?: string };
  sources: Source[];
}
export interface Cluster {
  id: string;
  name: string;
  description: string;
  question: string;
  color: string;
  position: { x: number; y: number };
}
export interface Relation {
  id: string;
  source: string;
  target: string;
  type: "uses" | "introduces" | "extends";
  evidence: Source;
}
export interface Guide {
  id: string;
  title: string;
  description: string;
  steps: { paperId: string; note: string }[];
}
export interface Catalog {
  version: number;
  updatedAt: string;
  clusters: Cluster[];
  papers: Paper[];
  datasets: Dataset[];
  relations: Relation[];
  guides: Guide[];
}
