/**
 * Marching squares: extracts the level set {f = level} from a regular grid of
 * samples. Used to draw the decision boundary (the curve where the network
 * outputs exactly 0.5) as a crisp line rather than a colour change.
 */

export type Segment = [number, number, number, number];

/**
 * `values` is row-major with `cols` entries per row. Coordinates in the result
 * are grid indices (fractional), so the caller scales them to pixels.
 */
export function marchingSquares(
  values: Float64Array | number[],
  cols: number,
  rows: number,
  level = 0.5,
): Segment[] {
  const segments: Segment[] = [];
  const at = (r: number, c: number) => values[r * cols + c] - level;

  // Linear interpolation of the crossing point along a cell edge.
  const interp = (a: number, b: number) => {
    const denom = a - b;
    return Math.abs(denom) < 1e-12 ? 0.5 : a / denom;
  };

  for (let r = 0; r < rows - 1; r++) {
    for (let c = 0; c < cols - 1; c++) {
      const tl = at(r, c);
      const tr = at(r, c + 1);
      const br = at(r + 1, c + 1);
      const bl = at(r + 1, c);

      let code = 0;
      if (tl > 0) code |= 8;
      if (tr > 0) code |= 4;
      if (br > 0) code |= 2;
      if (bl > 0) code |= 1;
      if (code === 0 || code === 15) continue;

      const top: [number, number] = [c + interp(tl, tr), r];
      const right: [number, number] = [c + 1, r + interp(tr, br)];
      const bottom: [number, number] = [c + interp(bl, br), r + 1];
      const left: [number, number] = [c, r + interp(tl, bl)];

      const push = (a: [number, number], b: [number, number]) =>
        segments.push([a[0], a[1], b[0], b[1]]);

      switch (code) {
        case 1:
        case 14:
          push(left, bottom);
          break;
        case 2:
        case 13:
          push(bottom, right);
          break;
        case 3:
        case 12:
          push(left, right);
          break;
        case 4:
        case 11:
          push(top, right);
          break;
        case 6:
        case 9:
          push(top, bottom);
          break;
        case 7:
        case 8:
          push(left, top);
          break;
        // Saddle cases: connect both crossings.
        case 5:
          push(left, top);
          push(bottom, right);
          break;
        case 10:
          push(top, right);
          push(left, bottom);
          break;
      }
    }
  }
  return segments;
}
