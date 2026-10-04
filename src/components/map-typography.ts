// Shared by SVG rendering and deterministic label measurement.
export const MAP_FONT_SIZE = { primary: 40, secondary: 28 } as const;
export const MAP_FONT_FAMILY = 'Arial, Helvetica, "PingFang SC", "Microsoft YaHei", sans-serif';
export const MAP_RAIL_WIDTH = 8;
export const MAP_CANVAS_WIDTH = 7400;

// Conservative unkerned advance widths for printable ASCII in Arial regular
// and bold, in ems (larger of the two). This avoids padding every lowercase
// letter like a wide capital while keeping server and browser layout equal.
const ASCII_ADVANCES = [0.278, 0.333, 0.474, 0.556, 0.556, 0.889, 0.722, 0.238, 0.333, 0.333, 0.389, 0.584, 0.278, 0.333, 0.278, 0.278, 0.556, 0.556, 0.556, 0.556, 0.556, 0.556, 0.556, 0.556, 0.556, 0.556, 0.333, 0.333, 0.584, 0.584, 0.584, 0.611, 1.015, 0.722, 0.722, 0.722, 0.722, 0.667, 0.611, 0.778, 0.722, 0.278, 0.556, 0.722, 0.611, 0.833, 0.722, 0.778, 0.667, 0.778, 0.722, 0.667, 0.611, 0.722, 0.667, 0.944, 0.667, 0.667, 0.611, 0.333, 0.278, 0.333, 0.584, 0.556, 0.333, 0.556, 0.611, 0.556, 0.611, 0.556, 0.333, 0.611, 0.611, 0.278, 0.278, 0.556, 0.278, 0.889, 0.611, 0.611, 0.611, 0.611, 0.389, 0.556, 0.333, 0.611, 0.556, 0.778, 0.556, 0.556, 0.5, 0.389, 0.28, 0.389, 0.584];
export function mapTextWidth(text: string, size: number): number {
  const advance = [...text].reduce((sum, char) => {
    const code = char.codePointAt(0)!;
    return sum + (code >= 32 && code <= 126 ? ASCII_ADVANCES[code - 32] : 1);
  }, 0);
  return Math.ceil(advance * size * 1.02 + 6);
}
