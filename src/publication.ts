/** Combine publication tracks for navigation while retaining exact paper metadata. */
export function publicationVenue(venue: string): string {
  return venue === "NeurIPS Datasets and Benchmarks" ? "NeurIPS" : venue;
}
