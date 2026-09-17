import { useMemo, useState } from 'react';
import { Section } from '../components/layout/Section.tsx';
import type { SectionProps } from './registry.ts';
import type { NetworkConfig } from '../lib/network.ts';
import { MLP } from '../lib/network.ts';
import { ACTIVATIONS, HIDDEN_ACTIVATIONS } from '../lib/activations.ts';
import type { ActivationName } from '../lib/activations.ts';
import { makeGaussian, makeRng } from '../lib/rng.ts';
import { Plot, Polyline } from '../components/viz/Plot.tsx';
import { useMutableNetwork } from '../hooks/useMutableNetwork.ts';
import { NetworkDiagram, NEGATIVE_COLOR, POSITIVE_COLOR } from '../components/viz/NetworkDiagram.tsx';
import type { DiagramSelection } from '../components/viz/NetworkDiagram.tsx';
import { Panel, Note, Stats, Legend } from '../components/ui/layout.tsx';
import { Detail, InWords } from '../components/ui/Detail.tsx';
import { Button, Segmented, Slider } from '../components/ui/controls.tsx';
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


const ACT_COLOR = '#3b82f6';
const GRAD_COLOR = '#e0761f';

interface LayerScale {
  layer: number;
  activationStd: number;
  gradientStd: number;
}

/**
 * Initialises a deep network with weight standard deviation gain/sqrt(fan_in),
 * runs a batch of standard-normal inputs through it, and measures how the
 * spread of activations and of pre-activation gradients changes with depth.
 */
function probeInitialisation(
  depth: number,
  width: number,
  gain: number,
  activation: ActivationName,
  samples = 64,
): LayerScale[] {
  const net = new MLP({
    inputSize: width,
    hiddenUnits: Array.from({ length: depth }, () => width),
    outputSize: 1,
    hiddenActivation: activation,
    outputActivation: 'linear',
    loss: 'mse',
    seed: 99,
  });

  const rng = makeRng(4242);
  const gaussian = makeGaussian(rng);
  for (let l = 0; l < net.W.length; l++) {
    const std = gain / Math.sqrt(net.sizes[l]);
    for (const row of net.W[l]) {
      for (let i = 0; i < row.length; i++) row[i] = gaussian() * std;
    }
    net.b[l].fill(0);
  }

  const activationSums = new Array(depth).fill(0);
  const gradientSums = new Array(depth).fill(0);
  let counted = 0;

  for (let s = 0; s < samples; s++) {
    const x = Array.from({ length: width }, () => gaussian());
    const trace = net.forward(x);
    // Fix the output gradient at 1 so the plot isolates the backward dynamics.
    const grads = net.backward(trace, [trace.output[0] - 0.5]);
    for (let l = 0; l < depth; l++) {
      for (const v of trace.layers[l].a) activationSums[l] += v * v;
      for (const v of grads.delta[l]) gradientSums[l] += v * v;
    }
    counted += 1;
  }

  return Array.from({ length: depth }, (_, l) => ({
    layer: l + 1,
    activationStd: Math.sqrt(activationSums[l] / (counted * width)),
    gradientStd: Math.sqrt(gradientSums[l] / (counted * width)),
  }));
}

const LOG_FLOOR = 1e-12;
const safeLog = (v: number) => Math.log10(Math.max(LOG_FLOOR, v));

function InitialisationProbe() {
  const [gain, setGain] = useState(1.41);
  const [activation, setActivation] = useState<ActivationName>('relu');
  const depth = 10;
  const width = 48;

  const scales = useMemo(
    () => probeInitialisation(depth, width, gain, activation),
    [gain, activation],
  );

  const activationPoints = scales.map((s) => [s.layer, safeLog(s.activationStd)] as [number, number]);
  const gradientPoints = scales.map((s) => [s.layer, safeLog(s.gradientStd)] as [number, number]);
  const recommended = activation === 'relu' || activation === 'leakyRelu' ? Math.SQRT2 : 1;
  const last = scales[scales.length - 1];
  const first = scales[0];

  return (
    <div className="grid grid--side">
      <Panel
        title={`${depth} layers of ${width} units, weights ~ N(0, gain²/fan_in)`}
        hint="log scale"
        caption="Blue: the standard deviation of the activations entering each layer. Orange: the standard deviation of δ at each layer, measured with the output gradient fixed at 1. A flat line means the scale is preserved; a sloping line means it compounds geometrically with depth."
      >
        <Plot
          xDomain={[1, depth]}
          yDomain={[-8, 4]}
          height={280}
          xLabel="layer"
          yLabel="log₁₀ std"
          xTicks={depth}
          showZeroLines={false}
          ariaLabel="Activation and gradient scale by layer"
        >
          {(plotScales) => (
            <>
              <line
                x1={0}
                x2={plotScales.innerWidth}
                y1={plotScales.y(0)}
                y2={plotScales.y(0)}
                stroke="var(--border-strong)"
                strokeDasharray="4 4"
              />
              <text x={plotScales.innerWidth - 4} y={plotScales.y(0) - 5} textAnchor="end">
                std = 1
              </text>
              <Polyline points={activationPoints} scales={plotScales} color={ACT_COLOR} width={2.2} />
              <Polyline points={gradientPoints} scales={plotScales} color={GRAD_COLOR} width={2.2} dash="5 4" />
              {activationPoints.map(([x, y]) => (
                <circle key={`a${x}`} cx={plotScales.x(x)} cy={plotScales.y(y)} r={3} fill={ACT_COLOR} />
              ))}
              {gradientPoints.map(([x, y]) => (
                <circle key={`g${x}`} cx={plotScales.x(x)} cy={plotScales.y(y)} r={3} fill={GRAD_COLOR} />
              ))}
            </>
          )}
        </Plot>
        <div style={{ marginTop: 10 }}>
          <Legend
            items={[
              { color: ACT_COLOR, label: 'activation std' },
              { color: GRAD_COLOR, label: 'gradient std', dashed: true },
            ]}
          />
        </div>
      </Panel>

      <div className="stack">
        <Panel title="Initialisation scale">
          <div className="stack stack--sm">
            <Slider
              label="gain"
              min={0.2}
              max={3}
              step={0.01}
              value={gain}
              onChange={setGain}
              display={fmt(gain, 2)}
            />
            <Segmented
              label="Activation"
              value={activation}
              options={HIDDEN_ACTIVATIONS.map((n) => ({ value: n, label: ACTIVATIONS[n].label }))}
              onChange={setActivation}
            />
          </div>
          <div className="btn-row" style={{ marginTop: 12 }}>
            <Button onClick={() => setGain(Number(recommended.toFixed(2)))}>
              Use the recommended gain ({fmt(recommended, 2)})
            </Button>
            <Button onClick={() => setGain(0.7)}>Too small</Button>
            <Button onClick={() => setGain(2.2)}>Too large</Button>
          </div>
        </Panel>

        <Stats
          items={[
            { label: 'a std, layer 1', value: fmt(first.activationStd, 3) },
            { label: `a std, layer ${depth}`, value: fmt(last.activationStd, 3), accent: true },
            { label: 'δ std, layer 1', value: fmt(first.gradientStd, 6) },
            { label: `δ std, layer ${depth}`, value: fmt(last.gradientStd, 6) },
            {
              label: 'ratio per layer',
              value: fmt(Math.pow(last.activationStd / Math.max(1e-12, first.activationStd), 1 / (depth - 1)), 3),
            },
          ]}
        />

        <Note>
          <p style={{ fontSize: 14 }}>
            At gain = {fmt(recommended, 2)} both lines are close to flat. Move the slider to 0.7 and
            the activations fall by roughly a factor of {fmt(0.7 / recommended, 2)} per layer, so
            after ten layers they are {fmt(Math.pow(0.7 / recommended, 9), 4)} times their original
            size. Move it to 2.2 and the same factor acts upward until the numbers overflow. The
            gradient line moves with it, because the backward pass multiplies by the same matrices.
          </p>
        </Note>
      </div>
    </div>
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
          <InWords>
            <p>
              Unit <M>{'j'}</M> of this layer walks along row <M>{'j'}</M> of the grid, multiplies
              each weight by the matching activation from the previous layer, adds the products up,
              and adds its own bias.
            </p>
            <p>
              The key detail for later: the weight <M>{'W_{ji}'}</M> is used exactly once, in exactly
              one product. Nothing else in the entire network touches it. That is why working out
              how much it affects the error turns out to be easy.
            </p>
          </InWords>
          <p>
            Because <M>{'W^{(l)}_{ji}'}</M> multiplies only <M>{'a^{(l-1)}_i'}</M>, its partial
            derivative is <M>{'\\partial z^{(l)}_j / \\partial W^{(l)}_{ji} = a^{(l-1)}_i'}</M>.
            That single fact is what makes backpropagation cheap, and it returns in section 08.
          </p>

          <h3 className="subhead">Initialisation is not arbitrary</h3>
          <p>
            Two things can go wrong before training even starts. The first is symmetry: if two units
            in a layer begin with identical incoming weights and biases, they compute identical
            values, receive identical gradients, and remain identical for every subsequent update.
            A layer of <M>{'n'}</M> such units has the expressive power of one unit, permanently.
          </p>
          <p>
            The second is scale. The forward pass multiplies by <M>{'W'}</M> at every layer, and the
            backward pass multiplies by <M>{'W^{\\top}'}</M> at every layer. If those
            multiplications shrink the typical magnitude by a factor <M>{'r < 1'}</M>, then after{' '}
            <M>{'L'}</M> layers the signal is <M>{'r^{L}'}</M> times its original size. With{' '}
            <M>{'r = 0.5'}</M> and <M>{'L = 20'}</M> that is <M>{'10^{-6}'}</M>. With{' '}
            <M>{'r = 2'}</M> it is <M>{'10^{6}'}</M>. Neither is usable, and both happen easily.
          </p>
          <p>
            The buttons rescale every weight in the network. Watch the hidden activations in the
            diagram: press ×2 twice and they saturate towards <M>{'\\pm 1'}</M>, where tanh is flat;
            press ÷2 a few times and they collapse towards 0, where every unit carries almost the
            same signal.
          </p>

          <Detail kicker="argument" title="Why identical weights stay identical">
            <p>
              Take units <M>{'j'}</M> and <M>{'k'}</M> in the same layer with{' '}
              <M>{'W^{(l)}_{j\\cdot} = W^{(l)}_{k\\cdot}'}</M> and{' '}
              <M>{'b^{(l)}_j = b^{(l)}_k'}</M>. They receive the same input vector{' '}
              <M>{'\\mathbf{a}^{(l-1)}'}</M>, so <M>{'z_j = z_k'}</M> and{' '}
              <M>{'a_j = a_k'}</M> for every example.
            </p>
            <p>
              Now look at the gradients. Section 08 derives{' '}
              <M>{"\\delta^{(l)}_j = f'(z^{(l)}_j)\\sum_{p} W^{(l+1)}_{pj}\\delta^{(l+1)}_p"}</M>.
              The factor <M>{"f'(z_j)"}</M> equals <M>{"f'(z_k)"}</M> because the pre-activations are
              equal. The sum differs only through the outgoing weights{' '}
              <M>{'W^{(l+1)}_{pj}'}</M> and <M>{'W^{(l+1)}_{pk}'}</M> — so if those columns are also
              equal (as they are when the whole matrix is initialised to one constant), then{' '}
              <M>{'\\delta_j = \\delta_k'}</M>, and therefore{' '}
              <M>{'\\partial L/\\partial W_{ji} = \\delta_j a_i = \\delta_k a_i = \\partial L/\\partial W_{ki}'}</M>.
            </p>
            <p>
              The two rows receive the same update, so they are still equal after the step. By
              induction they are equal for ever. Breaking the tie requires the initial values to
              differ; random draws from a continuous distribution do so with probability 1.
            </p>
            <p>
              Biases can safely start at zero — the weights alone break the symmetry — which is why
              the standard practice is random weights and zero biases. This site follows it.
            </p>
          </Detail>

          <Detail title="Where 2/fan_in and 2/(fan_in + fan_out) come from">
            <p>
              The goal is to pick a spread for the random initial weights such that the numbers
              flowing through the network keep roughly the same spread from one layer to the next.
              Too small and they shrink towards zero; too large and they blow up. The calculation
              below finds the value that keeps them steady.
            </p>
            <p>
              Three facts about variance are all that is needed. Writing{' '}
              <M>{'\\operatorname{Var}(X)'}</M> for how spread out a quantity is (section 00):
            </p>
            <ul>
              <li>
                Scaling multiplies the spread by the square:{' '}
                <M>{'\\operatorname{Var}(cX) = c^2\\operatorname{Var}(X)'}</M>. Doubling every
                number quadruples the variance.
              </li>
              <li>
                Adding independent quantities adds their variances:{' '}
                <M>{'\\operatorname{Var}(X+Y) = \\operatorname{Var}(X) + \\operatorname{Var}(Y)'}</M>.
              </li>
              <li>
                For a quantity with average zero, variance and mean-square are the same thing:{' '}
                <M>{'\\operatorname{Var}(X) = \\mathbb{E}[X^2]'}</M>.
              </li>
            </ul>
            <p>
              Now take one unit's pre-activation, which is a sum of{' '}
              <M>{'n_{\\text{in}}'}</M> products, with the bias starting at zero:
            </p>
            <Equation plain>{'z_j = \\sum_{i=1}^{n_{\\text{in}}} w_{ji} a_i'}</Equation>
            <p>
              Assume the weights are drawn independently with average 0 and spread{' '}
              <M>{'\\sigma_w^2'}</M>, and that they are unrelated to the incoming activations.
              Applying the second fact to the sum, then the first to each product:
            </p>
            <Equation plain>
              {'\\operatorname{Var}(z_j) = \\sum_{i} \\operatorname{Var}(w_{ji} a_i) = n_{\\text{in}}\\,\\sigma_w^2\\,\\mathbb{E}[a^2]'}
            </Equation>
            <p>
              Read that result plainly: the spread of a unit's output is the spread of one weight,
              multiplied by how many inputs it has, multiplied by the typical squared size of those
              inputs. The fan-in appears because more inputs means more terms added together, and
              adding independent terms accumulates variance.
            </p>
            <p>
              For the spread to stay the same from layer to layer we need the multiplier{' '}
              <M>{'n_{\\text{in}}\\sigma_w^2'}</M> to equal 1, which means
            </p>
            <Equation plain>{'\\sigma_w^2 = \\frac{1}{n_{\\text{in}}}'}</Equation>
            <p>
              In words: a unit with 100 inputs should have weights one tenth the size of a unit with
              1 input, because <M>{'\\sigma_w = 1/\\sqrt{n_{\\text{in}}}'}</M>. That is the
              right answer for an activation that passes its input through roughly unchanged near
              zero, such as tanh.
            </p>
            <p>
              ReLU changes the accounting, because it throws half the signal away. If{' '}
              <M>{'z'}</M> is equally likely to be positive or negative, then{' '}
              <M>{'a = \\max(0, z)'}</M> is exactly zero half the time and equal to{' '}
              <M>{'z'}</M> the other half. Averaging <M>{'a^2'}</M> over both cases gives half of
              what averaging <M>{'z^2'}</M> would:
            </p>
            <Equation plain>
              {'\\mathbb{E}[a^2] = \\underbrace{\\tfrac{1}{2}\\cdot 0}_{\\text{negative half}} + \\underbrace{\\tfrac{1}{2}\\,\\mathbb{E}[z^2]}_{\\text{positive half}} = \\tfrac{1}{2}\\operatorname{Var}(z)'}
            </Equation>
            <p>
              Feeding that factor of one half back into the previous result, each layer now shrinks
              the spread by an extra half unless the weights compensate. Doubling{' '}
              <M>{'\\sigma_w^2'}</M> exactly cancels it:
            </p>
            <Equation plain>{'\\sigma_w^2 = \\frac{2}{n_{\\text{in}}} \\quad \\text{(He)}'}</Equation>
            <p>
              The backward pass gives a second condition. The gradient recursion multiplies by{' '}
              <M>{'W^{\\top}'}</M>, where the sum now runs over the <M>{'n_{\\text{out}}'}</M>{' '}
              units of the next layer, so preserving the gradient scale asks for{' '}
              <M>{'\\sigma_w^2 = 1/n_{\\text{out}}'}</M>. The two conditions disagree whenever the
              layer changes width. Glorot initialisation takes the harmonic compromise
            </p>
            <Equation plain>
              {'\\sigma_w^2 = \\frac{2}{n_{\\text{in}} + n_{\\text{out}}} \\quad \\text{(Glorot)}'}
            </Equation>
            <InWords tag="in practice">
              <p>
                You never type these numbers yourself — every framework has{' '}
                <code>he_normal</code> and <code>glorot_uniform</code> built in. What matters is
                knowing that the right answer depends on the layer's width and on which activation
                follows it, and that getting it wrong makes a deep network untrainable rather than
                merely slower.
              </p>
            </InWords>
            <p>
              which satisfies neither exactly and both approximately. In terms of the{' '}
              <em>gain</em> in the plot below, writing{' '}
              <M>{'\\sigma_w = g/\\sqrt{n_{\\text{in}}}'}</M>: He is{' '}
              <M>{'g = \\sqrt{2} \\approx 1.41'}</M> and the tanh-appropriate value is{' '}
              <M>{'g = 1'}</M>.
            </p>
          </Detail>
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

      <h3 className="subhead">Initialisation scale, measured</h3>
      <p className="prose-block">
        The derivation above predicts a specific gain. The panel below checks it: a 10-layer network
        of 48 units, weights drawn from{' '}
        <M>{'\\mathcal{N}(0, g^2/n_{\\text{in}})'}</M>, fed standard-normal inputs. It reports the
        standard deviation of the activations and of <M>{'\\delta'}</M> at every layer, on a log
        scale, so a constant scale is a horizontal line and a compounding one is a straight slope.
      </p>

      <InitialisationProbe />

      <div className="grid grid--2">
        <div className="prose-block">
          <h3 className="subhead">What the plot shows</h3>
          <ul>
            <li>
              <strong>ReLU at gain 1.41:</strong> both lines are nearly flat. This is He
              initialisation, and it is what this site uses for ReLU-family activations.
            </li>
            <li>
              <strong>ReLU at gain 1.0:</strong> the activations decay by a factor of{' '}
              <M>{'1/\\sqrt{2} \\approx 0.71'}</M> per layer, exactly as the{' '}
              <M>{'\\mathbb{E}[a^2] = \\tfrac12 \\operatorname{Var}(z)'}</M> term predicts.
            </li>
            <li>
              <strong>Tanh at gain 1.41:</strong> the activations do not explode, because tanh
              saturates — but the gradient line falls, because{' '}
              <M>{"\\tanh'(z) = 1 - \\tanh^2(z)"}</M> is small wherever the unit is saturated.
              Saturation converts an exploding forward pass into a vanishing backward one.
            </li>
            <li>
              <strong>Any activation at gain 0.4:</strong> both lines fall steeply. The last layers
              still train, but the first layers receive gradients many orders of magnitude smaller
              than the last, so they barely move.
            </li>
          </ul>
        </div>
        <div className="prose-block">
          <h3 className="subhead">Permutation symmetry</h3>
          <p>
            Random initialisation breaks the symmetry between units, but one symmetry remains and
            cannot be removed. Permuting the units of a hidden layer — reordering the rows of{' '}
            <M>{'W^{(l)}'}</M> and <M>{'\\mathbf{b}^{(l)}'}</M>, and the columns of{' '}
            <M>{'W^{(l+1)}'}</M> in the same way — leaves the function computed by the network
            exactly unchanged.
          </p>
          <p>
            A layer of <M>{'n'}</M> units therefore admits <M>{'n!'}</M> parameter settings that all
            compute the same function, and a network with hidden widths{' '}
            <M>{'n_1, \\ldots, n_{L-1}'}</M> has at least <M>{'\\prod_l n_l!'}</M> copies of every
            minimum. For a modest 8-8 network that is <M>{'40320^2 \\approx 1.6\\times 10^{9}'}</M>{' '}
            equivalent points.
          </p>
          <p>
            This is why comparing two trained networks weight-by-weight is meaningless, and why
            "the" minimum found by training is never unique. It also means the loss surface is
            highly non-convex by construction, independent of the data.
          </p>
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
