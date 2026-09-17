import type { ReactNode } from 'react';
import { useCallback, useMemo, useRef } from 'react';
import { scaleLinear } from 'd3-scale';
import type { ScaleLinear } from 'd3-scale';
import { line as d3line } from 'd3-shape';

export interface PlotScales {
  x: ScaleLinear<number, number>;
  y: ScaleLinear<number, number>;
  /** Inner plot area in viewBox units. */
  innerWidth: number;
  innerHeight: number;
}

interface PlotProps {
  xDomain: [number, number];
  yDomain: [number, number];
  width?: number;
  height?: number;
  margin?: { top: number; right: number; bottom: number; left: number };
  xLabel?: string;
  yLabel?: string;
  xTicks?: number;
  yTicks?: number;
  /** Draw a line at y = 0 and x = 0 when inside the domain. */
  showZeroLines?: boolean;
  grid?: boolean;
  children: (scales: PlotScales) => ReactNode;
  onPointerData?: (point: { x: number; y: number }) => void;
  ariaLabel?: string;
  className?: string;
}

const DEFAULT_MARGIN = { top: 12, right: 14, bottom: 30, left: 42 };

/**
 * A minimal Cartesian plot frame: linear scales, grid, axes and ticks.
 * Children draw inside the plot area using the supplied scales, so every
 * visualisation on the site shares one coordinate convention.
 */
export function Plot({
  xDomain,
  yDomain,
  width = 480,
  height = 300,
  margin = DEFAULT_MARGIN,
  xLabel,
  yLabel,
  xTicks = 6,
  yTicks = 5,
  showZeroLines = true,
  grid = true,
  children,
  onPointerData,
  ariaLabel,
  className,
}: PlotProps) {
  const svgRef = useRef<SVGSVGElement>(null);
  const innerWidth = width - margin.left - margin.right;
  const innerHeight = height - margin.top - margin.bottom;

  const scales = useMemo<PlotScales>(() => {
    const x = scaleLinear().domain(xDomain).range([0, innerWidth]);
    const y = scaleLinear().domain(yDomain).range([innerHeight, 0]);
    return { x, y, innerWidth, innerHeight };
  }, [xDomain, yDomain, innerWidth, innerHeight]);

  const xTickValues = useMemo(() => scales.x.ticks(xTicks), [scales, xTicks]);
  const yTickValues = useMemo(() => scales.y.ticks(yTicks), [scales, yTicks]);

  const handlePointer = useCallback(
    (event: React.PointerEvent<SVGSVGElement>) => {
      if (!onPointerData || !svgRef.current) return;
      const rect = svgRef.current.getBoundingClientRect();
      if (rect.width === 0) return;
      const scale = width / rect.width;
      const vx = (event.clientX - rect.left) * scale - margin.left;
      const vy = (event.clientY - rect.top) * scale - margin.top;
      onPointerData({ x: scales.x.invert(vx), y: scales.y.invert(vy) });
    },
    [onPointerData, margin.left, margin.top, scales, width],
  );

  return (
    <svg
      ref={svgRef}
      className={className ? `viz ${className}` : 'viz'}
      viewBox={`0 0 ${width} ${height}`}
      role="img"
      aria-label={ariaLabel}
      style={onPointerData ? { cursor: 'crosshair' } : undefined}
      onPointerDown={(event) => {
        if (onPointerData) event.currentTarget.setPointerCapture(event.pointerId);
        handlePointer(event);
      }}
      onPointerMove={(event) => {
        if (event.buttons === 1) handlePointer(event);
      }}
    >
      <g transform={`translate(${margin.left}, ${margin.top})`}>
        {grid
          ? yTickValues.map((tick) => (
              <line
                key={`gy-${tick}`}
                className="grid-line"
                x1={0}
                x2={innerWidth}
                y1={scales.y(tick)}
                y2={scales.y(tick)}
              />
            ))
          : null}
        {grid
          ? xTickValues.map((tick) => (
              <line
                key={`gx-${tick}`}
                className="grid-line"
                y1={0}
                y2={innerHeight}
                x1={scales.x(tick)}
                x2={scales.x(tick)}
              />
            ))
          : null}

        {showZeroLines && yDomain[0] < 0 && yDomain[1] > 0 ? (
          <line
            className="axis-line"
            x1={0}
            x2={innerWidth}
            y1={scales.y(0)}
            y2={scales.y(0)}
          />
        ) : null}
        {showZeroLines && xDomain[0] < 0 && xDomain[1] > 0 ? (
          <line className="axis-line" y1={0} y2={innerHeight} x1={scales.x(0)} x2={scales.x(0)} />
        ) : null}

        {children(scales)}

        <line className="axis-line" x1={0} x2={innerWidth} y1={innerHeight} y2={innerHeight} />
        <line className="axis-line" x1={0} x2={0} y1={0} y2={innerHeight} />

        {xTickValues.map((tick) => (
          <text key={`tx-${tick}`} x={scales.x(tick)} y={innerHeight + 15} textAnchor="middle">
            {formatTick(tick)}
          </text>
        ))}
        {yTickValues.map((tick) => (
          <text key={`ty-${tick}`} x={-8} y={scales.y(tick) + 3.5} textAnchor="end">
            {formatTick(tick)}
          </text>
        ))}

        {xLabel ? (
          <text x={innerWidth / 2} y={innerHeight + 28} textAnchor="middle">
            {xLabel}
          </text>
        ) : null}
        {yLabel ? (
          <text
            transform={`translate(${-margin.left + 11}, ${innerHeight / 2}) rotate(-90)`}
            textAnchor="middle"
          >
            {yLabel}
          </text>
        ) : null}
      </g>
    </svg>
  );
}

function formatTick(value: number): string {
  if (Object.is(value, -0) || value === 0) return '0';
  const abs = Math.abs(value);
  if (abs >= 1000) return value.toExponential(0).replace('-', '−');
  if (abs < 0.001) return value.toExponential(0).replace('-', '−');
  const text = Number(value.toPrecision(4)).toString();
  return text.replace('-', '−');
}

interface CurveProps {
  f: (x: number) => number;
  scales: PlotScales;
  domain?: [number, number];
  samples?: number;
  color: string;
  dash?: string;
  width?: number;
  opacity?: number;
  /**
   * x values where f jumps. The path is drawn in separate pieces so a step
   * discontinuity is not rendered as a vertical line through values the
   * function never takes.
   */
  breaks?: number[];
}

/** Samples f over the x domain and renders it as a path. */
export function Curve({
  f,
  scales,
  domain,
  samples = 240,
  color,
  dash,
  width = 2,
  opacity = 1,
  breaks,
}: CurveProps) {
  const path = useMemo(() => {
    const [lo, hi] = domain ?? (scales.x.domain() as [number, number]);
    const generator = d3line<[number, number]>()
      .x((p) => scales.x(p[0]))
      .y((p) => scales.y(p[1]));

    const cuts = (breaks ?? []).filter((b) => b > lo && b < hi).sort((a, b) => a - b);
    const bounds = [lo, ...cuts, hi];
    const epsilon = (hi - lo) * 1e-6;
    let d = '';
    for (let piece = 0; piece < bounds.length - 1; piece++) {
      const start = piece === 0 ? bounds[piece] : bounds[piece] + epsilon;
      const end = piece === bounds.length - 2 ? bounds[piece + 1] : bounds[piece + 1] - epsilon;
      const pieceSamples = Math.max(2, Math.round((samples * (end - start)) / (hi - lo)));
      const points: Array<[number, number]> = [];
      for (let i = 0; i <= pieceSamples; i++) {
        const x = start + ((end - start) * i) / pieceSamples;
        const y = f(x);
        if (Number.isFinite(y)) points.push([x, y]);
      }
      d += generator(points) ?? '';
    }
    return d;
  }, [f, domain, samples, scales, breaks]);

  return (
    <path
      className="curve"
      d={path}
      stroke={color}
      strokeWidth={width}
      strokeDasharray={dash}
      opacity={opacity}
    />
  );
}

interface PolylineProps {
  points: Array<[number, number]>;
  scales: PlotScales;
  color: string;
  width?: number;
  dash?: string;
  opacity?: number;
}

/** Renders an explicit sequence of data points as a path. */
export function Polyline({ points, scales, color, width = 2, dash, opacity = 1 }: PolylineProps) {
  const path = useMemo(() => {
    const generator = d3line<[number, number]>()
      .x((p) => scales.x(p[0]))
      .y((p) => scales.y(p[1]));
    return generator(points.filter((p) => Number.isFinite(p[1]))) ?? '';
  }, [points, scales]);
  return (
    <path
      className="curve"
      d={path}
      stroke={color}
      strokeWidth={width}
      strokeDasharray={dash}
      opacity={opacity}
    />
  );
}

interface MarkerProps {
  x: number;
  y: number;
  scales: PlotScales;
  color: string;
  label?: string;
  radius?: number;
  guides?: boolean;
}

/** A highlighted point with optional dashed guides to the axes. */
export function Marker({ x, y, scales, color, label, radius = 5, guides = true }: MarkerProps) {
  const px = scales.x(x);
  const py = scales.y(y);
  if (!Number.isFinite(px) || !Number.isFinite(py)) return null;
  return (
    <g>
      {guides ? (
        <>
          <line
            x1={0}
            x2={px}
            y1={py}
            y2={py}
            stroke={color}
            strokeWidth={1}
            strokeDasharray="3 3"
            opacity={0.55}
          />
          <line
            x1={px}
            x2={px}
            y1={py}
            y2={scales.innerHeight}
            stroke={color}
            strokeWidth={1}
            strokeDasharray="3 3"
            opacity={0.55}
          />
        </>
      ) : null}
      <circle cx={px} cy={py} r={radius} fill={color} stroke="var(--bg-panel)" strokeWidth={1.5} />
      {label ? (
        <text x={px + 9} y={py - 8} fill={color} style={{ fontWeight: 600 }}>
          {label}
        </text>
      ) : null}
    </g>
  );
}
