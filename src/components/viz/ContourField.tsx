import { useMemo } from 'react';
import type { PlotScales } from './Plot.tsx';
import { marchingSquares } from '../../lib/contour.ts';
import { useThemeSignal } from '../../hooks/useThemeSignal.ts';

interface ContourFieldProps {
  f: (x: number, y: number) => number;
  scales: PlotScales;
  resolution?: number;
  /** Explicit iso-levels; by default they are spaced on a square-root scale. */
  levels?: number[];
  levelCount?: number;
}

/**
 * Renders a scalar field as a raster image plus iso-lines, inside a Plot frame.
 * The raster is generated once per (f, domain) pair and drawn as an <image>,
 * which keeps the SVG small while the contours stay crisp.
 */
export function ContourField({
  f,
  scales,
  resolution = 110,
  levels,
  levelCount = 9,
}: ContourFieldProps) {
  const themeVersion = useThemeSignal();
  const { href, isoPaths } = useMemo(() => {
    const isDark = document.documentElement.dataset.theme === 'dark';
    const [x0, x1] = scales.x.domain() as [number, number];
    const [y0, y1] = scales.y.domain() as [number, number];
    const grid = new Float64Array(resolution * resolution);
    let min = Infinity;
    let max = -Infinity;

    for (let r = 0; r < resolution; r++) {
      const y = y1 - ((y1 - y0) * r) / (resolution - 1);
      for (let c = 0; c < resolution; c++) {
        const x = x0 + ((x1 - x0) * c) / (resolution - 1);
        const value = f(x, y);
        grid[r * resolution + c] = value;
        if (value < min) min = value;
        if (value > max) max = value;
      }
    }

    const canvas = document.createElement('canvas');
    canvas.width = resolution;
    canvas.height = resolution;
    const ctx = canvas.getContext('2d');
    let url = '';
    if (ctx) {
      const image = ctx.createImageData(resolution, resolution);
      const span = Math.max(1e-12, max - min);
      for (let i = 0; i < grid.length; i++) {
        // Square-root compression keeps detail near the minimum visible.
        const t = Math.sqrt((grid[i] - min) / span);
        image.data[i * 4] = 59;
        image.data[i * 4 + 1] = 130;
        image.data[i * 4 + 2] = 246;
        image.data[i * 4 + 3] = Math.round((isDark ? 14 : 18) + t * (isDark ? 110 : 150));
      }
      ctx.putImageData(image, 0, 0);
      url = canvas.toDataURL();
    }

    const chosen =
      levels ??
      Array.from({ length: levelCount }, (_, i) => {
        const t = (i + 1) / (levelCount + 1);
        return min + (max - min) * t * t;
      });

    const paths = chosen.map((level) => {
      const segments = marchingSquares(grid, resolution, resolution, level);
      const cellX = scales.innerWidth / (resolution - 1);
      const cellY = scales.innerHeight / (resolution - 1);
      let d = '';
      for (const [ax, ay, bx, by] of segments) {
        d += `M${(ax * cellX).toFixed(2)} ${(ay * cellY).toFixed(2)}L${(bx * cellX).toFixed(2)} ${(by * cellY).toFixed(2)}`;
      }
      return { level, d };
    });

    return { href: url, isoPaths: paths };
  }, [f, scales, resolution, levels, levelCount, themeVersion]);

  return (
    <g>
      {href ? (
        <image
          href={href}
          x={0}
          y={0}
          width={scales.innerWidth}
          height={scales.innerHeight}
          preserveAspectRatio="none"
        />
      ) : null}
      {isoPaths.map((path) => (
        <path
          key={path.level}
          d={path.d}
          fill="none"
          stroke="var(--text-faint)"
          strokeWidth={0.7}
          opacity={0.45}
        />
      ))}
    </g>
  );
}
