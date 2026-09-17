import { useMemo } from 'react';
import { Plot, Polyline } from './Plot.tsx';

export interface Series {
  label: string;
  color: string;
  values: Array<[number, number]>;
  dash?: string;
}

interface MetricChartProps {
  series: Series[];
  /** viewBox width; smaller values make the tick labels relatively larger. */
  width?: number;
  height?: number;
  xLabel?: string;
  yLabel?: string;
  /** Force the y axis to start at 0 (losses) or fit the data (accuracy). */
  yMin?: number;
  yMax?: number;
  /** Vertical rule, e.g. the early-stopping epoch. */
  markerX?: number | null;
  markerLabel?: string;
  ariaLabel?: string;
}

/** Downsamples long series so that redraws stay cheap while training runs. */
function thin(values: Array<[number, number]>, limit = 320): Array<[number, number]> {
  if (values.length <= limit) return values;
  const stride = Math.ceil(values.length / limit);
  const out: Array<[number, number]> = [];
  for (let i = 0; i < values.length; i += stride) out.push(values[i]);
  const last = values[values.length - 1];
  if (out[out.length - 1] !== last) out.push(last);
  return out;
}

export function MetricChart({
  series,
  width = 420,
  height = 220,
  xLabel = 'epoch',
  yLabel,
  yMin,
  yMax,
  markerX,
  markerLabel,
  ariaLabel,
}: MetricChartProps) {
  const thinned = useMemo(() => series.map((s) => ({ ...s, values: thin(s.values) })), [series]);

  const [xDomain, yDomain] = useMemo(() => {
    let maxX = 1;
    let minY = Infinity;
    let maxY = -Infinity;
    for (const s of thinned) {
      for (const [x, y] of s.values) {
        if (x > maxX) maxX = x;
        if (Number.isFinite(y)) {
          if (y < minY) minY = y;
          if (y > maxY) maxY = y;
        }
      }
    }
    if (!Number.isFinite(minY)) {
      minY = 0;
      maxY = 1;
    }
    const lo = yMin ?? Math.max(0, minY - (maxY - minY) * 0.1);
    const hi = yMax ?? maxY + Math.max(1e-6, (maxY - minY) * 0.12);
    return [
      [0, maxX] as [number, number],
      [lo, hi === lo ? lo + 1 : hi] as [number, number],
    ];
  }, [thinned, yMin, yMax]);

  return (
    <Plot
      xDomain={xDomain}
      yDomain={yDomain}
      width={width}
      height={height}
      xLabel={xLabel}
      yLabel={yLabel}
      showZeroLines={false}
      ariaLabel={ariaLabel}
    >
      {(scales) => (
        <>
          {markerX !== null && markerX !== undefined ? (
            <g>
              <line
                x1={scales.x(markerX)}
                x2={scales.x(markerX)}
                y1={0}
                y2={scales.innerHeight}
                stroke="var(--text-faint)"
                strokeDasharray="4 3"
              />
              {markerLabel ? (
                <text x={scales.x(markerX) + 4} y={11}>
                  {markerLabel}
                </text>
              ) : null}
            </g>
          ) : null}
          {thinned.map((s) => (
            <Polyline
              key={s.label}
              points={s.values}
              scales={scales}
              color={s.color}
              width={1.8}
              dash={s.dash}
            />
          ))}
        </>
      )}
    </Plot>
  );
}
