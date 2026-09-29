import type { Box, Point } from "./publication-layout";

// Circular fillets have the same visible radius at both 45° and 90° turns.
// Leave a straight run between neighboring fillets instead of making S-curves.
export function metroPath(points: Point[], stations: Point[], labels: Box[]): string {
  if (!points.length) return "";
  const value = (n: number) => Number(n.toFixed(3));
  const xy = (p: Point) => `${value(p.x)} ${value(p.y)}`;
  let path = `M${xy(points[0])}`;
  for (let i = 1; i < points.length - 1; i++) {
    const a = points[i - 1], p = points[i], b = points[i + 1];
    const before = Math.hypot(p.x - a.x, p.y - a.y);
    const after = Math.hypot(b.x - p.x, b.y - p.y);
    if (!before || !after) continue;
    const u = { x: (p.x - a.x) / before, y: (p.y - a.y) / before };
    const v = { x: (b.x - p.x) / after, y: (b.y - p.y) / after };
    const angle = Math.acos(Math.max(-1, Math.min(1, u.x * v.x + u.y * v.y)));
    const atStation = stations.some((s) => Math.hypot(s.x - p.x, s.y - p.y) < .001);
    if (atStation || angle < .001 || Math.PI - angle < .001) {
      path += `L${xy(p)}`;
      continue;
    }
    const tangent = Math.tan(angle / 2);
    let trim = Math.min(12 * tangent, Math.max(0, (before - 14) / 2), Math.max(0, (after - 14) / 2));
    const endpoints = () => ({
      start: { x: p.x - u.x * trim, y: p.y - u.y * trim },
      end: { x: p.x + v.x * trim, y: p.y + v.y * trim },
    });
    let { start, end } = endpoints();
    const overlapsLabel = () => {
      const left = Math.min(start.x, p.x, end.x), right = Math.max(start.x, p.x, end.x);
      const top = Math.min(start.y, p.y, end.y), bottom = Math.max(start.y, p.y, end.y);
      return labels.some((box) => left < box.x + box.width + 3 && right > box.x - 3 &&
        top < box.y + box.height + 3 && bottom > box.y - 3);
    };
    while (trim > .5 && overlapsLabel()) {
      trim *= .5;
      ({ start, end } = endpoints());
    }
    const radius = trim / tangent;
    if (radius < 1 || overlapsLabel()) path += `L${xy(p)}`;
    else path += `L${xy(start)}A${value(radius)} ${value(radius)} 0 0 ${u.x * v.y - u.y * v.x > 0 ? 1 : 0} ${xy(end)}`;
  }
  return `${path}L${xy(points[points.length - 1])}`;
}
