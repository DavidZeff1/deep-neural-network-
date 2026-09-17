import { useMemo, useState } from 'react';
import { Section } from '../components/layout/Section.tsx';
import type { SectionProps } from './registry.ts';
import type { NetworkConfig } from '../lib/network.ts';
import { useMutableNetwork } from '../hooks/useMutableNetwork.ts';
import { NetworkDiagram, NEGATIVE_COLOR, POSITIVE_COLOR } from '../components/viz/NetworkDiagram.tsx';
import type { DiagramSelection } from '../components/viz/NetworkDiagram.tsx';
import { Panel, Note, Stats } from '../components/ui/layout.tsx';
import { Button, Slider } from '../components/ui/controls.tsx';
import { Equation, M } from '../components/ui/Math.tsx';
import { fmt } from '../lib/format.ts';

const CONFIG: NetworkConfig = {
  inputSize: 2,
  hiddenUnits: [3],
  outputSize: 1,
  hiddenActivation: 'tanh',
  outputActivation: 'sigmoid',
  loss: 'bce',
  seed: 42,
};

/** Clickable heat map of one weight matrix, with the bias column alongside. */
function MatrixGrid({
  W,
  b,
  layer,
  selection,
  onSelect,
}: {
  W: number[][];
  b: number[];
  layer: number;
  selection: DiagramSelection | null;
  onSelect: (selection: DiagramSelection | null) => void;
}) {
  const max = Math.max(1e-6, ...W.flat().map(Math.abs), ...b.map(Math.abs));
  const cell = 46;
  const gap = 4;
  const cols = W[0].length;
  const rows = W.length;
  const width = (cols + 1.6) * (cell + gap) + 30;
  const height = rows * (cell + gap) + 26;

  return (
    <svg className="viz" viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Weight matrix">
      {W.map((row, j) =>
        row.map((value, i) => {
          const selected =
            selection?.kind === 'edge' &&
            selection.layer === layer &&
            selection.from === i &&
            selection.to === j;
          const colour = value >= 0 ? POSITIVE_COLOR : NEGATIVE_COLOR;
          return (
            <g
              key={`${j}-${i}`}
              onClick={() =>
                onSelect(selected ? null : { kind: 'edge', layer, from: i, to: j })
              }
              style={{ cursor: 'pointer' }}
            >
              <rect
                x={30 + i * (cell + gap)}
                y={20 + j * (cell + gap)}
                width={cell}
                height={cell}
                rx={4}
                fill={colour}
                fillOpacity={Math.min(1, Math.abs(value) / max) * 0.55}
                stroke={selected ? colour : 'var(--border)'}
                strokeWidth={selected ? 2 : 1}
              />
              <text
                x={30 + i * (cell + gap) + cell / 2}
                y={20 + j * (cell + gap) + cell / 2 + 4}
                textAnchor="middle"
                style={{ fill: 'var(--text)', fontSize: 11 }}
              >
                {fmt(value, 2)}
              </text>
            </g>
          );
        }),
      )}
      {b.map((value, j) => (
        <g key={`b-${j}`}>
          <rect
            x={30 + cols * (cell + gap) + 16}
            y={20 + j * (cell + gap)}
            width={cell}
            height={cell}
            rx={4}
            fill={value >= 0 ? POSITIVE_COLOR : NEGATIVE_COLOR}
            fillOpacity={Math.min(1, Math.abs(value) / max) * 0.55}
            stroke="var(--border)"
            strokeDasharray="3 3"
          />
          <text
            x={30 + cols * (cell + gap) + 16 + cell / 2}
            y={20 + j * (cell + gap) + cell / 2 + 4}
            textAnchor="middle"
            style={{ fill: 'var(--text)', fontSize: 11 }}
          >
            {fmt(value, 2)}
          </text>
        </g>
      ))}
      {W.map((_, j) => (
        <text key={`r-${j}`} x={24} y={20 + j * (cell + gap) + cell / 2 + 4} textAnchor="end">
          {j + 1}
        </text>
      ))}
      {W[0].map((_, i) => (
        <text key={`c-${i}`} x={30 + i * (cell + gap) + cell / 2} y={14} textAnchor="middle">
          {i + 1}
        </text>
      ))}
      <text x={30 + cols * (cell + gap) + 16 + cell / 2} y={14} textAnchor="middle">
        b
      </text>
    </svg>
  );
}

export function WeightsSection({ id, index }: SectionProps) {
  const config = useMemo(() => CONFIG, []);
  const { network, version, mutate, reset } = useMutableNetwork(config);
  const [selection, setSelection] = useState<DiagramSelection | null>({
    kind: 'edge',
    layer: 0,
    from: 0,
    to: 0,
  });
  const [input, setInput] = useState<[number, number]>([0.6, -0.4]);
  const [layerIndex, setLayerIndex] = useState(0);

  const trace = useMemo(
    () => network.forward(input),
    // `version` is the dependency that matters: the network is mutated in place.
    [network, input, version],
  );

  const selectedWeight =
    selection?.kind === 'edge' ? network.W[selection.layer][selection.to][selection.from] : null;

  const selectedBias = selection?.kind === 'node' && selection.layer > 0
    ? network.b[selection.layer - 1][selection.index]
    : null;

  const weightNorm = useMemo(() => {
    let sum = 0;
    for (const layer of network.W) for (const row of layer) for (const w of row) sum += w * w;
    return Math.sqrt(sum);
    // `version` is the dependency that matters: the network is mutated in place.
  }, [network, version]);

  const scaleAll = (factor: number) =>
    mutate((net) => {
      for (const layer of net.W) for (const row of layer) for (let i = 0; i < row.length; i++) row[i] *= factor;
    });

  return (
    <Section
      id={id}
      index={index}
      title="Weights & biases"
      lede={
        <>
          The weights are the entries of <M>{'W^{(l)}'}</M>; row <M>{'j'}</M> holds the weight
          vector of unit <M>{'j'}</M>. The biases are the entries of <M>{'\\mathbf{b}^{(l)}'}</M>.
          These are the only quantities training changes.
        </>
      }
    >
      <div className="grid grid--side">
        <Panel
          title="2 → 3 → 1 network"
          hint="click a connection or a unit"
          caption="Selecting an edge in the diagram selects the same entry in the matrix below, and the reverse. Changing it recomputes the forward pass immediately."
        >
          <NetworkDiagram
            network={network}
            trace={trace}
            selection={selection}
            onSelect={setSelection}
            height={250}
            inputLabels={['x₁', 'x₂']}
            outputLabels={['ŷ']}
          />
        </Panel>

        <div className="stack">
          <Panel title={layerIndex === 0 ? 'W⁽¹⁾ — shape 3 × 2' : 'W⁽²⁾ — shape 1 × 3'}>
            <div className="btn-row" style={{ marginBottom: 12 }}>
              <Button onClick={() => setLayerIndex(0)} variant={layerIndex === 0 ? 'primary' : 'default'}>
                Layer 1
              </Button>
              <Button onClick={() => setLayerIndex(1)} variant={layerIndex === 1 ? 'primary' : 'default'}>
                Layer 2
              </Button>
            </div>
            <MatrixGrid
              W={network.W[layerIndex]}
              b={network.b[layerIndex]}
              layer={layerIndex}
              selection={selection}
              onSelect={setSelection}
            />
          </Panel>

          <Panel title="Edit the selected parameter">
            {selection?.kind === 'edge' && selectedWeight !== null ? (
              <div className="stack stack--sm">
                <Slider
                  label={
                    <>
                      w<sup>({selection.layer + 1})</sup>
                      <sub>
                        {selection.to + 1},{selection.from + 1}
                      </sub>
                    </>
                  }
                  min={-3}
                  max={3}
                  value={selectedWeight}
                  onChange={(value) =>
                    mutate((net) => {
                      net.W[selection.layer][selection.to][selection.from] = value;
                    })
                  }
                  display={fmt(selectedWeight, 3)}
                />
                <p className="faint" style={{ fontSize: 12.5, margin: 0 }}>
                  Connects unit {selection.from + 1} of layer {selection.layer} to unit{' '}
                  {selection.to + 1} of layer {selection.layer + 1}.
                </p>
              </div>
            ) : selection?.kind === 'node' && selectedBias !== null ? (
              <div className="stack stack--sm">
                <Slider
                  label={
                    <>
                      b<sup>({selection.layer})</sup>
                      <sub>{selection.index + 1}</sub>
                    </>
                  }
                  min={-3}
                  max={3}
                  value={selectedBias}
                  onChange={(value) =>
                    mutate((net) => {
                      net.b[selection.layer - 1][selection.index] = value;
                    })
                  }
                  display={fmt(selectedBias, 3)}
                />
                <p className="faint" style={{ fontSize: 12.5, margin: 0 }}>
                  Bias of unit {selection.index + 1} in layer {selection.layer}. It shifts that
                  unit's pre-activation without changing how it weighs its inputs.
                </p>
              </div>
            ) : (
              <p className="faint" style={{ fontSize: 13, margin: 0 }}>
                Select a connection or a hidden/output unit to edit its value.
              </p>
            )}
          </Panel>
        </div>
      </div>

      <div className="grid grid--side">
        <div className="prose-block">
          <h3 className="subhead">Reading the matrix</h3>
          <p>
            Column <M>{'i'}</M> of <M>{'W^{(l)}'}</M> collects every weight leaving unit{' '}
            <M>{'i'}</M> of the previous layer. Row <M>{'j'}</M> collects every weight entering unit{' '}
            <M>{'j'}</M> of this layer. The entry <M>{'W^{(l)}_{ji}'}</M> appears in exactly one
            product in the forward pass:
          </p>
          <Equation>
            {'z^{(l)}_j = \\sum_i W^{(l)}_{ji}\\, a^{(l-1)}_i + b^{(l)}_j'}
          </Equation>
          <p>
            Because <M>{'W^{(l)}_{ji}'}</M> multiplies only <M>{'a^{(l-1)}_i'}</M>, its partial
            derivative is <M>{'\\partial z^{(l)}_j / \\partial W^{(l)}_{ji} = a^{(l-1)}_i'}</M>.
            That single fact is what makes backpropagation cheap, and it returns in section 08.
          </p>

          <h3 className="subhead">Initialisation is not arbitrary</h3>
          <p>
            Setting every weight to the same value makes every unit in a layer compute the same
            thing, receive the same gradient, and stay identical for ever. Random initialisation
            breaks that symmetry. The scale matters too: weights that are too large push{' '}
            <M>{'|z|'}</M> into the flat regions of the activation, where <M>{"f'(z) \\approx 0"}</M>{' '}
            and learning stalls.
          </p>
          <p>
            The buttons rescale every weight in the network. Watch the hidden activations in the
            diagram: press ×2 twice and they saturate towards <M>{'\\pm 1'}</M>, where tanh is flat;
            press ÷2 a few times and they collapse towards 0, where every unit carries almost the
            same signal.
          </p>
        </div>

        <div className="stack">
          <Panel title="Experiments">
            <div className="btn-row">
              <Button onClick={() => reset(Math.floor(Math.random() * 100000))}>Re-initialise</Button>
              <Button onClick={() => scaleAll(2)}>Scale ×2</Button>
              <Button onClick={() => scaleAll(0.5)}>Scale ÷2</Button>
              <Button
                onClick={() =>
                  mutate((net) => {
                    for (const layer of net.W) for (const row of layer) row.fill(0);
                  })
                }
              >
                All weights → 0
              </Button>
            </div>
            <div className="stack stack--sm" style={{ marginTop: 14 }}>
              <Slider
                label="x₁"
                min={-2}
                max={2}
                value={input[0]}
                onChange={(v) => setInput([v, input[1]])}
                display={fmt(input[0])}
              />
              <Slider
                label="x₂"
                min={-2}
                max={2}
                value={input[1]}
                onChange={(v) => setInput([input[0], v])}
                display={fmt(input[1])}
              />
            </div>
          </Panel>

          <Stats
            items={[
              { label: 'Hidden a₁', value: fmt(trace.layers[0].a[0], 3) },
              { label: 'Hidden a₂', value: fmt(trace.layers[0].a[1], 3) },
              { label: 'Hidden a₃', value: fmt(trace.layers[0].a[2], 3) },
              { label: 'Output ŷ', value: fmt(trace.output[0], 3), accent: true },
              { label: '‖W‖₂', value: fmt(weightNorm, 2) },
            ]}
          />
        </div>
      </div>

      <Note title="Zero weights" accent>
        <p>
          With every weight at 0 the hidden pre-activations are just the biases, the output is
          constant, and — as section 08 shows — the gradient with respect to{' '}
          <M>{'W^{(2)}'}</M> is proportional to the hidden activations while the gradient with
          respect to <M>{'W^{(1)}'}</M> is proportional to <M>{'W^{(2)}'}</M>. Both are degenerate,
          so the network cannot escape by gradient descent. This is a real failure mode, not a
          hypothetical one.
        </p>
      </Note>
    </Section>
  );
}
