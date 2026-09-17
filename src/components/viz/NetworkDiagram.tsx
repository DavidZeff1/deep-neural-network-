import { useMemo } from 'react';
import type { ForwardTrace, Gradients, MLP } from '../../lib/network.ts';
import { fmt } from '../../lib/format.ts';

export type DiagramSelection =
  | { kind: 'node'; layer: number; index: number }
  | { kind: 'edge'; layer: number; from: number; to: number };

/** Fixed hues, legible on both themes. Blue = positive, orange = negative. */
export const POSITIVE_COLOR = '#3b82f6';
export const NEGATIVE_COLOR = '#e0761f';

interface NetworkDiagramProps {
  network: MLP;
  trace?: ForwardTrace | null;
  gradients?: Gradients | null;
  selection?: DiagramSelection | null;
  onSelect?: (selection: DiagramSelection | null) => void;
  /** Weight-layer index currently being computed; its edges animate. */
  activeLayer?: number | null;
  /** 'forward' animates left to right, 'backward' right to left. */
  direction?: 'forward' | 'backward';
  showValues?: boolean;
  /** Show ∂L/∂w on edges and δ in nodes instead of activations. */
  showGradients?: boolean;
  inputLabels?: string[];
  outputLabels?: string[];
  height?: number;
  maxNodes?: number;
}

interface NodePosition {
  x: number;
  y: number;
  layer: number;
  index: number;
}

export function NetworkDiagram({
  network,
  trace,
  gradients,
  selection,
  onSelect,
  activeLayer = null,
  direction = 'forward',
  showValues = true,
  showGradients = false,
  inputLabels,
  outputLabels,
  height = 300,
  maxNodes = 12,
}: NetworkDiagramProps) {
  const width = 620;
  const padX = 62;
  const padTop = 22;
  const padBottom = 42;

  const layout = useMemo(() => {
    const sizes = network.sizes;
    const shown = sizes.map((n) => Math.min(n, maxNodes));
    const columns = sizes.length;
    const usableW = width - padX * 2;
    const requestedInner = height - padTop - padBottom;
    const maxShown = Math.max(...shown);
    const radius = Math.max(8, Math.min(15, requestedInner / (maxShown * 2.5)));
    const maxSpacing = radius * 3.4;

    // Shrink the viewBox when the widest layer does not need the full height,
    // so small networks are not surrounded by empty space.
    const neededInner = maxShown > 1 ? maxSpacing * (maxShown - 1) + radius * 2 : radius * 2;
    const innerHeight = Math.min(requestedInner, Math.max(neededInner, radius * 2));
    const viewHeight = innerHeight + padTop + padBottom;

    const positions: NodePosition[][] = [];
    for (let l = 0; l < columns; l++) {
      const count = shown[l];
      const x = columns === 1 ? width / 2 : padX + (usableW * l) / (columns - 1);
      const spacing = count > 1 ? Math.min((innerHeight - radius * 2) / (count - 1), maxSpacing) : 0;
      const totalHeight = spacing * (count - 1);
      const top = padTop + (innerHeight - totalHeight) / 2;
      const column: NodePosition[] = [];
      for (let i = 0; i < count; i++) {
        column.push({ x, y: top + spacing * i, layer: l, index: i });
      }
      positions.push(column);
    }
    return { positions, shown, radius, sizes, viewHeight };
  }, [network.sizes, height, maxNodes]);

  const viewHeight = layout.viewHeight;

  const maxWeight = useMemo(() => {
    let max = 1e-6;
    for (const layer of network.W) for (const row of layer) for (const w of row) max = Math.max(max, Math.abs(w));
    return max;
  }, [network.W]);

  const maxGradient = useMemo(() => {
    if (!gradients) return 1;
    let max = 1e-9;
    for (const layer of gradients.dW)
      for (const row of layer) for (const g of row) max = Math.max(max, Math.abs(g));
    return max;
  }, [gradients]);

  const activationAt = (layer: number, index: number): number | null => {
    if (!trace) return null;
    if (layer === 0) return trace.input[index] ?? null;
    return trace.layers[layer - 1]?.a[index] ?? null;
  };

  const deltaAt = (layer: number, index: number): number | null => {
    if (!gradients || layer === 0) return null;
    return gradients.delta[layer - 1]?.[index] ?? null;
  };

  const maxActivation = useMemo(() => {
    if (!trace) return 1;
    let max = 1e-6;
    for (const v of trace.input) max = Math.max(max, Math.abs(v));
    for (const layer of trace.layers) for (const v of layer.a) max = Math.max(max, Math.abs(v));
    return max;
  }, [trace]);

  const maxDelta = useMemo(() => {
    if (!gradients) return 1;
    let max = 1e-9;
    for (const d of gradients.delta) for (const v of d) max = Math.max(max, Math.abs(v));
    return max;
  }, [gradients]);

  const isSelectedNode = (layer: number, index: number) =>
    selection?.kind === 'node' && selection.layer === layer && selection.index === index;

  const isSelectedEdge = (layer: number, from: number, to: number) =>
    selection?.kind === 'edge' &&
    selection.layer === layer &&
    selection.from === from &&
    selection.to === to;

  const edgeTouchesSelection = (layer: number, from: number, to: number) => {
    if (selection?.kind !== 'node') return false;
    if (selection.layer === layer && selection.index === from) return true;
    if (selection.layer === layer + 1 && selection.index === to) return true;
    return false;
  };

  const layerName = (l: number): string => {
    if (l === 0) return 'Input';
    if (l === layout.sizes.length - 1) return 'Output';
    return `Hidden ${l}`;
  };

  return (
    <svg
      className="viz"
      viewBox={`0 0 ${width} ${viewHeight}`}
      role="img"
      aria-label={`Network with layer sizes ${network.sizes.join(', ')}`}
      onClick={(event) => {
        if (event.target === event.currentTarget) onSelect?.(null);
      }}
    >
      {/* Edges are drawn first so nodes sit on top. */}
      <g>
        {layout.positions.slice(0, -1).map((column, l) =>
          column.map((source) =>
            layout.positions[l + 1].map((target) => {
              const weight = network.W[l]?.[target.index]?.[source.index] ?? 0;
              const gradient = gradients?.dW[l]?.[target.index]?.[source.index] ?? 0;
              const value = showGradients ? gradient : weight;
              const scale = showGradients ? maxGradient : maxWeight;
              const magnitude = Math.min(1, Math.abs(value) / scale);
              const selected = isSelectedEdge(l, source.index, target.index);
              const related = edgeTouchesSelection(l, source.index, target.index);
              const dimmed = selection !== null && selection !== undefined && !selected && !related;
              const color = value >= 0 ? POSITIVE_COLOR : NEGATIVE_COLOR;
              const flowing = activeLayer === l;
              return (
                <line
                  key={`e-${l}-${source.index}-${target.index}`}
                  className={`edge${flowing ? ' edge--flow' : ''}`}
                  x1={direction === 'backward' ? target.x : source.x}
                  y1={direction === 'backward' ? target.y : source.y}
                  x2={direction === 'backward' ? source.x : target.x}
                  y2={direction === 'backward' ? source.y : target.y}
                  stroke={color}
                  strokeWidth={selected ? 3.4 : 0.6 + magnitude * 2.6}
                  opacity={dimmed ? 0.09 : selected ? 1 : 0.16 + magnitude * 0.62}
                  onClick={(event) => {
                    event.stopPropagation();
                    onSelect?.(
                      selected ? null : { kind: 'edge', layer: l, from: source.index, to: target.index },
                    );
                  }}
                >
                  <title>
                    {showGradients
                      ? `∂L/∂w[${l}][${target.index}][${source.index}] = ${fmt(gradient, 4)}`
                      : `w[${l}][${target.index}][${source.index}] = ${fmt(weight, 3)}`}
                  </title>
                </line>
              );
            }),
          ),
        )}
      </g>

      <g>
        {layout.positions.map((column, l) => (
          <g key={`layer-${l}`}>
            {column.map((node) => {
              const raw = showGradients && l > 0 ? deltaAt(l, node.index) : activationAt(l, node.index);
              const scale = showGradients && l > 0 ? maxDelta : maxActivation;
              const magnitude = raw === null ? 0 : Math.min(1, Math.abs(raw) / scale);
              const color = raw === null ? POSITIVE_COLOR : raw >= 0 ? POSITIVE_COLOR : NEGATIVE_COLOR;
              const selected = isSelectedNode(l, node.index);
              const dimmed =
                selection?.kind === 'node' && !selected
                  ? Math.abs(selection.layer - l) > 1
                  : false;
              const r = layout.radius;
              return (
                <g
                  className="node-group"
                  key={`n-${l}-${node.index}`}
                  opacity={dimmed ? 0.4 : 1}
                  onClick={(event) => {
                    event.stopPropagation();
                    onSelect?.(selected ? null : { kind: 'node', layer: l, index: node.index });
                  }}
                >
                  <circle
                    className="node"
                    cx={node.x}
                    cy={node.y}
                    r={r}
                    fill="var(--bg-panel)"
                    stroke={selected ? color : 'var(--border-strong)'}
                    strokeWidth={selected ? 2.4 : 1.2}
                  />
                  <circle
                    cx={node.x}
                    cy={node.y}
                    r={r - 1}
                    fill={color}
                    fillOpacity={magnitude * 0.72}
                    pointerEvents="none"
                    style={{ transition: 'fill-opacity 0.25s var(--ease)' }}
                  />
                  {showValues && raw !== null && r >= 11 ? (
                    <text
                      x={node.x}
                      y={node.y + 3.2}
                      textAnchor="middle"
                      style={{ fontSize: 9, fill: 'var(--text)', fontWeight: 600 }}
                    >
                      {fmt(raw, Math.abs(raw) >= 10 ? 0 : 2).replace('0.', '.')}
                    </text>
                  ) : null}
                  <title>
                    {`${layerName(l)} unit ${node.index + 1}${raw === null ? '' : ` = ${fmt(raw, 4)}`}`}
                  </title>
                </g>
              );
            })}
            {layout.shown[l] < layout.sizes[l] ? (
              <text
                x={column[0].x}
                y={column[column.length - 1].y + layout.radius + 12}
                textAnchor="middle"
              >
                +{layout.sizes[l] - layout.shown[l]}
              </text>
            ) : null}
            <text
              className={`layer-label${activeLayer !== null && (activeLayer === l - 1) ? ' layer-label--active' : ''}`}
              x={column[0].x}
              y={viewHeight - 15}
              textAnchor="middle"
            >
              {layerName(l)}
            </text>
            <text x={column[0].x} y={viewHeight - 3} textAnchor="middle" style={{ fontSize: 9.5 }}>
              {layout.sizes[l]} {layout.sizes[l] === 1 ? 'unit' : 'units'}
            </text>
          </g>
        ))}
      </g>

      {/* Input and output annotations */}
      <g>
        {inputLabels
          ? layout.positions[0].map((node, i) =>
              inputLabels[i] ? (
                <text
                  key={`il-${i}`}
                  x={node.x - layout.radius - 8}
                  y={node.y + 3.5}
                  textAnchor="end"
                  style={{ fill: 'var(--text-muted)' }}
                >
                  {inputLabels[i]}
                </text>
              ) : null,
            )
          : null}
        {outputLabels
          ? layout.positions[layout.positions.length - 1].map((node, i) =>
              outputLabels[i] ? (
                <text
                  key={`ol-${i}`}
                  x={node.x + layout.radius + 8}
                  y={node.y + 3.5}
                  textAnchor="start"
                  style={{ fill: 'var(--text-muted)' }}
                >
                  {outputLabels[i]}
                </text>
              ) : null,
            )
          : null}
      </g>
    </svg>
  );
}
