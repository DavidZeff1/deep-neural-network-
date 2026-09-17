import { useMemo, useState } from 'react';
import { Section } from '../components/layout/Section.tsx';
import type { SectionProps } from './registry.ts';
import { ACTIVATIONS, softmax } from '../lib/activations.ts';
import type { ActivationName } from '../lib/activations.ts';
import { Panel, Note, Stats, Legend } from '../components/ui/layout.tsx';
import { Segmented, Slider, Button } from '../components/ui/controls.tsx';
import { Equation, M } from '../components/ui/Math.tsx';
import { Curve, Marker, Plot } from '../components/viz/Plot.tsx';
import { fmt } from '../lib/format.ts';

const ORDER: ActivationName[] = ['relu', 'sigmoid', 'tanh', 'leakyRelu'];

const CURVE_COLORS: Record<ActivationName, string> = {
  relu: '#3b82f6',
  sigmoid: '#e0761f',
  tanh: '#16a34a',
  leakyRelu: '#8b5cf6',
  linear: '#64748b',
};

function SoftmaxPanel() {
  const [logits, setLogits] = useState([2.0, 1.0, 0.1]);
  const probabilities = softmax(logits);
  const max = Math.max(...logits);
  const exps = logits.map((z) => Math.exp(z - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  const width = 480;
  const height = 190;
  const barWidth = 74;

  const setLogit = (index: number, value: number) => {
    const next = logits.slice();
    next[index] = value;
    setLogits(next);
  };

  return (
    <div className="grid grid--side">
      <Panel
        title="Softmax over 3 logits"
        hint="probabilities sum to 1"
        caption="Bars show the output probabilities. The dashed line marks the largest logit; softmax is invariant to adding a constant to every logit, which is why implementations subtract the maximum before exponentiating."
      >
        <svg className="viz" viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Softmax probabilities">
          {probabilities.map((p, i) => {
            const x = 62 + i * (barWidth + 46);
            const h = p * 118;
            return (
              <g key={i}>
                <rect
                  x={x}
                  y={140 - h}
                  width={barWidth}
                  height={Math.max(1, h)}
                  rx={3}
                  fill="#3b82f6"
                  fillOpacity={0.28 + p * 0.55}
                  stroke="#3b82f6"
                  style={{ transition: 'all 0.18s var(--ease)' }}
                />
                <text x={x + barWidth / 2} y={140 - h - 8} textAnchor="middle" style={{ fill: 'var(--text)', fontSize: 12, fontWeight: 600 }}>
                  {fmt(p, 3)}
                </text>
                <text x={x + barWidth / 2} y={156} textAnchor="middle">
                  class {i + 1}
                </text>
                <text x={x + barWidth / 2} y={170} textAnchor="middle" style={{ fill: 'var(--text-muted)' }}>
                  z = {fmt(logits[i], 2)}
                </text>
              </g>
            );
          })}
          <line x1={40} y1={140} x2={width - 20} y2={140} className="axis-line" />
          <line x1={40} y1={22} x2={width - 20} y2={22} stroke="var(--border-strong)" strokeDasharray="4 4" />
          <text x={34} y={26} textAnchor="end">
            1.0
          </text>
          <text x={34} y={144} textAnchor="end">
            0
          </text>
        </svg>
      </Panel>

      <div className="stack">
        <Panel title="Logits">
          <div className="stack stack--sm">
            {logits.map((value, i) => (
              <Slider
                key={i}
                label={`z${i + 1}`}
                min={-4}
                max={4}
                value={value}
                onChange={(next) => setLogit(i, next)}
                display={fmt(value, 2)}
              />
            ))}
          </div>
          <div className="btn-row" style={{ marginTop: 12 }}>
            <Button onClick={() => setLogits([2, 1, 0.1])}>Reset</Button>
            <Button onClick={() => setLogits([0, 0, 0])}>All equal</Button>
            <Button onClick={() => setLogits([6, 1, 0.1])}>One dominant</Button>
          </div>
        </Panel>

        <Panel title="The calculation">
          <div className="calc">
            <div className="calc__line">
              <span className="calc__label">shift: </span>
              zᵢ − max(z) = {logits.map((z) => fmt(z - max, 2)).join(', ')}
            </div>
            <div className="calc__line">
              <span className="calc__label">exp:   </span>
              {exps.map((e) => fmt(e, 4)).join(', ')}
            </div>
            <div className="calc__line">
              <span className="calc__label">sum:   </span>
              {fmt(sum, 4)}
            </div>
            <div className="calc__line">
              <span className="calc__label">p:     </span>
              <span className="calc__result">{probabilities.map((p) => fmt(p, 4)).join(', ')}</span>
            </div>
            <div className="calc__line">
              <span className="calc__label">Σp:    </span>
              {fmt(probabilities.reduce((a, b) => a + b, 0), 6)}
            </div>
          </div>
        </Panel>
      </div>
    </div>
  );
}

export function ActivationsSection({ id, index }: SectionProps) {
  const [name, setName] = useState<ActivationName>('relu');
  const [z, setZ] = useState(0.85);
  const [showAll, setShowAll] = useState(false);

  const act = ACTIVATIONS[name];
  const value = act.f(z);
  const slope = act.df(z);

  const yDomain = useMemo<[number, number]>(() => {
    if (name === 'relu' || name === 'leakyRelu') return [-1.2, 4.2];
    if (name === 'sigmoid') return [-0.25, 1.25];
    return [-1.35, 1.35];
  }, [name]);

  return (
    <Section
      id={id}
      index={index}
      title="Activation functions"
      lede={
        <>
          The activation <M>{'f'}</M> is applied elementwise to the pre-activations. Without it the
          whole network reduces to one linear map, so <M>{'f'}</M> is what makes depth meaningful.
          Its derivative decides how much gradient survives the trip back through the layer.
        </>
      }
    >
      <div className="grid grid--side">
        <Panel
          title="f(z) and its derivative"
          hint="drag the slider or the plot"
          caption="Solid line: f(z). Dashed line: f′(z). The marker is the current input. Flat regions of f are exactly the regions where f′ ≈ 0."
        >
          <Plot
            xDomain={[-4, 4]}
            yDomain={yDomain}
            height={300}
            xLabel="z"
            yLabel="f(z)"
            onPointerData={(point) => setZ(Math.max(-4, Math.min(4, point.x)))}
            ariaLabel={`${act.label} activation function`}
          >
            {(scales) => (
              <>
                {showAll
                  ? ORDER.filter((other) => other !== name).map((other) => (
                      <Curve
                        key={other}
                        f={ACTIVATIONS[other].f}
                        scales={scales}
                        color={CURVE_COLORS[other]}
                        width={1.2}
                        opacity={0.45}
                      />
                    ))
                  : null}
                <Curve
                  f={act.df}
                  scales={scales}
                  color={CURVE_COLORS[name]}
                  dash="4 4"
                  width={1.6}
                  opacity={0.75}
                  breaks={name === 'relu' || name === 'leakyRelu' ? [0] : undefined}
                />
                <Curve f={act.f} scales={scales} color={CURVE_COLORS[name]} width={2.4} />
                <Marker x={z} y={value} scales={scales} color={CURVE_COLORS[name]} />
                <Marker x={z} y={slope} scales={scales} color="var(--text-faint)" radius={3.4} guides={false} />
              </>
            )}
          </Plot>
        </Panel>

        <div className="stack">
          <Panel title="Function">
            <Segmented
              value={name}
              options={ORDER.map((n) => ({ value: n, label: ACTIVATIONS[n].label }))}
              onChange={setName}
              ariaLabel="Activation function"
            />
            <div style={{ marginTop: 16 }}>
              <Equation plain>{act.formula}</Equation>
              <Equation plain>{act.derivative}</Equation>
            </div>
            <div style={{ marginTop: 10 }}>
              <Slider
                label="Input z"
                min={-4}
                max={4}
                step={0.01}
                value={z}
                onChange={setZ}
                display={fmt(z, 2)}
              />
            </div>
            <div className="btn-row" style={{ marginTop: 10 }}>
              <Button onClick={() => setShowAll(!showAll)}>
                {showAll ? 'Hide other functions' : 'Overlay other functions'}
              </Button>
            </div>
            {showAll ? (
              <div style={{ marginTop: 10 }}>
                <Legend
                  items={ORDER.map((n) => ({ color: CURVE_COLORS[n], label: ACTIVATIONS[n].label }))}
                />
              </div>
            ) : null}
          </Panel>

          <Stats
            items={[
              { label: 'input z', value: fmt(z, 3) },
              { label: 'output f(z)', value: fmt(value, 3), accent: true },
              { label: "slope f'(z)", value: fmt(slope, 3) },
            ]}
          />

          <Note>
            <p style={{ fontSize: 14 }}>{act.notes}</p>
          </Note>
        </div>
      </div>

      <div className="grid grid--2">
        <div className="prose-block">
          <h3 className="subhead">Why the derivative is the quantity that matters</h3>
          <p>
            During backpropagation the gradient arriving at a unit is multiplied by{' '}
            <M>{"f'(z)"}</M> before it continues to the previous layer. Across{' '}
            <M>{'L'}</M> layers the factors multiply:
          </p>
          <Equation>
            {"\\frac{\\partial L}{\\partial \\mathbf{a}^{(0)}} \\propto \\prod_{l=1}^{L} f'\\!\\left(z^{(l)}\\right) \\cdot W^{(l)}"}
          </Equation>
          <p>
            With sigmoid, <M>{"f'(z) \\le 0.25"}</M> everywhere. Ten such layers contribute a factor
            of at most <M>{'0.25^{10} \\approx 10^{-6}'}</M>, so the early layers receive almost no
            gradient. This is the vanishing-gradient problem, and it is the practical reason ReLU
            replaced sigmoid in hidden layers: for <M>{'z > 0'}</M> its derivative is exactly 1, so
            the product does not shrink.
          </p>
        </div>

        <div className="prose-block">
          <h3 className="subhead">Choosing an output activation</h3>
          <ul>
            <li>
              <strong>Regression:</strong> linear. The output must be able to take any real value.
            </li>
            <li>
              <strong>Binary classification:</strong> sigmoid, giving one number in{' '}
              <M>{'(0,1)'}</M> read as <M>{'P(y = 1 \\mid \\mathbf{x})'}</M>.
            </li>
            <li>
              <strong>K mutually exclusive classes:</strong> softmax, giving <M>{'K'}</M> positive
              numbers that sum to 1.
            </li>
            <li>
              <strong>K independent labels:</strong> K separate sigmoids, not softmax — the labels
              are not competing for a shared budget of probability.
            </li>
          </ul>
          <p>
            Hidden layers are a separate choice; ReLU is the standard default, with tanh preferred
            in small networks where zero-centred activations help.
          </p>
        </div>
      </div>

      <h3 className="subhead">Softmax</h3>
      <p className="prose-block">
        Softmax maps a vector of <M>{'K'}</M> real numbers to a probability distribution. Unlike the
        functions above it is not elementwise: every output depends on every input.
      </p>
      <Equation caption="Exponentiation makes every entry positive; dividing by the sum makes them add to 1.">
        {'\\mathrm{softmax}(\\mathbf{z})_k = \\frac{e^{z_k}}{\\sum_{j=1}^{K} e^{z_j}}'}
      </Equation>

      <SoftmaxPanel />

      <Note title="Two properties worth knowing" accent>
        <p>
          <strong>Shift invariance:</strong> adding a constant <M>{'c'}</M> to every logit leaves
          the output unchanged, because <M>{'e^{z_k + c} = e^{c} e^{z_k}'}</M> cancels between
          numerator and denominator. Set all three sliders to the same value to see it.
        </p>
        <p>
          <strong>Jacobian:</strong> since each output depends on all inputs, the derivative is a
          matrix, <M>{'\\partial p_i / \\partial z_j = p_i(\\delta_{ij} - p_j)'}</M>. Composed with
          categorical cross-entropy it collapses to <M>{'\\partial L/\\partial z_k = p_k - y_k'}</M>,
          which is why the two are almost always implemented together.
        </p>
      </Note>
    </Section>
  );
}
