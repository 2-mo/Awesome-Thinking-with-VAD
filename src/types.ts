export interface Source {
  url: string;
  note: string;
}
export interface ComparisonFact {
  values: string[];
  evidence: Source;
}
export interface Citation {
  key: string;
  type: "article" | "inproceedings" | "misc";
  version: "published" | "accepted" | "preprint";
  title: string;
  authors: string[];
  year: number;
  publication?: string;
  doi?: string;
  arxivId?: string;
  volume?: string;
  number?: string;
  pages?: string;
  url: string;
  sources: Source[];
  verifiedAt: string;
}
export interface PendingCitation {
  version: "pending";
  title: string;
  year: number;
  url: string;
  note: string;
  sources: Source[];
  verifiedAt: string;
}
export interface Paper {
  id: string;
  shortTitle: string;
  title: string;
  citation: Citation | PendingCitation;
  year: number;
  venue: string;
  // Hidden ordering: conference, journal publication, or arXiv v1 month.
  timeline?: { year?: number; month: number; basis: "conference" | "journal" | "preprint"; source: Source };
  scope: "core";
  // Editorial map selection only; keep the bibliographic record in reading indexes.
  mapExclusion?: { note: string };
  // A compact scene/task hint; video anomaly understanding remains unmarked.
  mapIcon?: { kind: "industry" | "road" | "video"; evidence: Source };
  cluster: string;
  classification?: { basis: "title"; evidence: Source };
  // Resource-focused or combined contributions; ordinary method papers omit this.
  contribution?: { kind: "resource" | "hybrid"; evidence: Source };
  // Additional method memberships are editorial classifications with evidence.
  secondaryMethods?: { cluster: string; evidence: Source }[];
  tasks: string[];
  tags?: string[];
  comparison?: Partial<Record<"outputs" | "training" | "inference" | "futureFrames" | "evaluation", ComparisonFact>>;
  summary: string;
  mechanism: string;
  takeaway: string;
  limitation: string;
  links: { paper: string; code?: string; project?: string };
  sources: Source[];
  verifiedAt: string;
  datasetIds: string[];
  // Author-provided artwork or a rendered region of the original paper PDF.
  figure?: {
    src: string;
    alt: string;
    caption: string;
    sourceUrl: string;
    sourcePageUrl: string;
    credit: string;
    verifiedAt: string;
  };
  figurePending?: { note: string; sources: Source[]; verifiedAt: string };
}
export interface Dataset {
  composition?: {
    kind: "original" | "annotation" | "resplit" | "mixed";
    note: string;
    evidence: Source;
    baseDatasetIds?: string[];
  };
  availability?: {
    status: "available" | "partial" | "pending" | "unverified";
    note: string;
    evidence: Source;
    verifiedAt: string;
  };
  year: number;
  venue: string;
  thumbnail?: { src: string; alt: string; caption?: string; sourceUrl: string; credit: string };
  id: string;
  name: string;
  description: string;
  tasks: string[];
  modalities: string[];
  annotations: string[];
  // Concise reader-facing caveat; full protocol and evidence remain below.
  usageNote?: string;
  protocol: string;
  links: { website: string; paper?: string };
  sources: Source[];
}
export interface Cluster {
  id: string;
  name: string;
  nameEn?: string;
  // Editorial placement preference only; this does not create a connection.
  layoutNear?: string;
  layoutSide?: "above" | "below";
  // Explicit local reading routes; shared paper IDs are real branch anchors.
  // These editorial associations do not assert scientific inheritance.
  routes?: { paperIds: string[]; evidence: Source }[];
  description: string;
  question: string;
  color: string;
  // A sourced paper must connect the two method directions at the fork.
  branchOf?: string;
  branchAt?: { paperId: string; evidence: Source };
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
