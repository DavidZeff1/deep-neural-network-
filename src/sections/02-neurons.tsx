import { useMemo, useState } from 'react';
import { Section } from '../components/layout/Section.tsx';
import type { SectionProps } from './registry.ts';
import { ACTIVATIONS, HIDDEN_ACTIVATIONS } from '../lib/activations.ts';
import type { ActivationName } from '../lib/activations.ts';
import { fmt, clamp } from '../lib/format.ts';
import { Panel, Note, Stats } from '../components/ui/layout.tsx';
import { Detail, InWords } from '../components/ui/Detail.tsx';
import { Segmented, Slider, Button } from '../components/ui/controls.tsx';
import { Equation, M } from '../components/ui/Math.tsx';
import { Exercise } from '../components/ui/Exercise.tsx';
import { DecisionBoundary } from '../components/viz/DecisionBoundary.tsx';
import { NEGATIVE_COLOR, POSITIVE_COLOR } from '../components/viz/NetworkDiagram.tsx';

interface NeuronDiagramProps {
  inputs: number[];
  weights: number[];
  bias: number;
  z: number;
  a: number;
  activationLabel: string;
}

function NeuronDiagram({ inputs, weights, bias, z, a, activationLabel }: NeuronDiagramProps) {
  const width = 520;
  const height = 210;
  const inputX = 74;
  const bodyX = 300;
  const outX = 452;
  const rows = inputs.map((_, i) => 58 + i * 58);
  const biasY = rows[rows.length - 1] + 52;
  const maxWeight = Math.max(1e-6, ...weights.map(Math.abs), Math.abs(bias));

  return (
    <svg className="viz" viewBox={`0 0 ${width} ${height}`} role="img" aria-label="A single neuron">
      {inputs.map((value, i) => {
        const weight = weights[i];
        const color = weight >= 0 ? POSITIVE_COLOR : NEGATIVE_COLOR;
        const magnitude = Math.min(1, Math.abs(weight) / maxWeight);
        // Place the label along the edge, offset perpendicular to it.
        const ax = inputX + 22;
        const ay = rows[i];
        const bx = bodyX - 34;
        const by = height / 2 - 6;
        const length = Math.hypot(bx - ax, by - ay) || 1;
        const normalX = -(by - ay) / length;
        const normalY = (bx - ax) / length;
        const labelX = ax + (bx - ax) * 0.45 + normalX * 12;
        const labelY = ay + (by - ay) * 0.45 + normalY * 12 + 3;
        return (
          <g key={i}>
            <line
              x1={inputX + 22}
              y1={rows[i]}
              x2={bodyX - 34}
              y2={height / 2 - 6}
              stroke={color}
              strokeWidth={0.8 + magnitude * 3}
              opacity={0.25 + magnitude * 0.6}
            />
            <circle cx={inputX} cy={rows[i]} r={20} fill="var(--bg-inset)" stroke="var(--border-strong)" />
            <text x={inputX} y={rows[i] + 4} textAnchor="middle" style={{ fill: 'var(--text)', fontSize: 12 }}>
              {fmt(value, 2)}
            </text>
            <text x={inputX - 28} y={rows[i] + 4} textAnchor="end" style={{ fill: 'var(--text-muted)' }}>
              x{i + 1}
            </text>
            <text
              x={labelX}
              y={labelY}
              textAnchor="middle"
              style={{
                fill: color,
                fontWeight: 600,
                paintOrder: 'stroke',
                stroke: 'var(--bg-panel)',
                strokeWidth: 3.5,
                strokeLinejoin: 'round',
              }}
            >
              w{i + 1} = {fmt(weight, 2)}
            </text>
          </g>
        );
      })}

      {/* bias */}
      <line
        x1={inputX + 22}
        y1={biasY}
        x2={bodyX - 34}
        y2={height / 2 + 10}
        stroke={bias >= 0 ? POSITIVE_COLOR : NEGATIVE_COLOR}
        strokeWidth={0.8 + Math.min(1, Math.abs(bias) / maxWeight) * 3}
        opacity={0.5}
        strokeDasharray="4 4"
      />
      <circle cx={inputX} cy={biasY} r={20} fill="var(--bg-inset)" stroke="var(--border-strong)" strokeDasharray="3 3" />
      <text x={inputX} y={biasY + 4} textAnchor="middle" style={{ fill: 'var(--text)', fontSize: 12 }}>
        1
      </text>
      <text x={inputX - 28} y={biasY + 4} textAnchor="end" style={{ fill: 'var(--text-muted)' }}>
        bias
      </text>
      <text
        x={(inputX + bodyX) / 2 + 14}
        y={biasY - 4}
        textAnchor="middle"
        style={{
          fill: 'var(--text-muted)',
          paintOrder: 'stroke',
          stroke: 'var(--bg-panel)',
          strokeWidth: 3.5,
          strokeLinejoin: 'round',
        }}
      >
        b = {fmt(bias, 2)}
      </text>

      {/* body */}
      <rect
        x={bodyX - 34}
        y={height / 2 - 40}
        width={112}
        height={80}
        rx={10}
        fill="var(--bg-panel)"
        stroke="var(--border-strong)"
      />
      <line x1={bodyX + 22} y1={height / 2 - 40} x2={bodyX + 22} y2={height / 2 + 40} stroke="var(--border)" />
      <text x={bodyX - 6} y={height / 2 - 18} textAnchor="middle" style={{ fill: 'var(--text-faint)', fontSize: 9.5 }}>
        Σ
      </text>
      <text x={bodyX - 6} y={height / 2 + 8} textAnchor="middle" style={{ fill: 'var(--text)', fontSize: 13, fontWeight: 600 }}>
        {fmt(z, 2)}
      </text>
      <text x={bodyX - 6} y={height / 2 + 26} textAnchor="middle" style={{ fill: 'var(--text-faint)', fontSize: 9 }}>
        z
      </text>
      <text x={bodyX + 50} y={height / 2 - 18} textAnchor="middle" style={{ fill: 'var(--text-faint)', fontSize: 9.5 }}>
        {activationLabel}
      </text>
      <text x={bodyX + 50} y={height / 2 + 8} textAnchor="middle" style={{ fill: 'var(--text)', fontSize: 13, fontWeight: 600 }}>
        {fmt(a, 2)}
      </text>
      <text x={bodyX + 50} y={height / 2 + 26} textAnchor="middle" style={{ fill: 'var(--text-faint)', fontSize: 9 }}>
        a
      </text>

      <line x1={bodyX + 78} y1={height / 2} x2={outX - 24} y2={height / 2} stroke="var(--border-strong)" strokeWidth={1.5} />
      <polygon
        points={`${outX - 24},${height / 2} ${outX - 32},${height / 2 - 4} ${outX - 32},${height / 2 + 4}`}
        fill="var(--border-strong)"
      />
      <circle cx={outX} cy={height / 2} r={22} fill="var(--bg-inset)" stroke="var(--border-strong)" />
      <text x={outX} y={height / 2 + 4} textAnchor="middle" style={{ fill: 'var(--text)', fontSize: 12, fontWeight: 600 }}>
        {fmt(a, 2)}
      </text>
      <text x={outX} y={height / 2 + 40} textAnchor="middle" style={{ fill: 'var(--text-muted)' }}>
        output
      </text>
    </svg>
  );
}

export function NeuronsSection({ id, index }: SectionProps) {
  const [x1, setX1] = useState(1.0);
  const [x2, setX2] = useState(0.5);
  const [w1, setW1] = useState(0.8);
  const [w2, setW2] = useState(-0.3);
  const [bias, setBias] = useState(0.2);
  const [activation, setActivation] = useState<ActivationName>('relu');

  const act = ACTIVATIONS[activation];
  const z = w1 * x1 + w2 * x2 + bias;
  const a = act.f(z);

  const predict = useMemo(() => {
    return (px: number, py: number) => {
      const value = w1 * px + w2 * py + bias;
      return 1 / (1 + Math.exp(-3 * value));
    };
  }, [w1, w2, bias]);

  const term = (w: number, x: number) => `(${fmt(w, 2)})(${fmt(x, 2)})`;

  return (
    <Section
      id={id}
      index={index}
      title="Neurons"
      lede={
        <>
          A neuron computes a weighted sum of its inputs, adds a bias, and applies an activation
          function to the result. Two numbers describe its state: the pre-activation{' '}
          <M>{'z'}</M> and the activation <M>{'a = f(z)'}</M>.
        </>
      }
    >
      <Equation caption="For a neuron with n inputs. In matrix form, a whole layer of these is a = f(Wx + b).">
        {'z = \\sum_{i=1}^{n} w_i x_i + b, \\qquad a = f(z)'}
      </Equation>
      <InWords>
        <p>
          Multiply each input by its own weight, add all those products together, then add one more
          number called the bias. Call the total <M>{'z'}</M>. Finally pass{' '}
          <M>{'z'}</M> through a function <M>{'f'}</M> to get the neuron's output{' '}
          <M>{'a'}</M>.
        </p>
        <p>
          With two inputs that is literally{' '}
          <M>{'z = w_1x_1 + w_2x_2 + b'}</M> — the worked example below uses exactly those numbers.
        </p>
      </InWords>

      <div className="grid grid--side">
        <Panel
          title="One neuron, two inputs"
          hint="every value below is live"
          caption="Solid edges carry inputs weighted by w. The dashed edge is the bias: a constant input of 1 with its own weight b."
        >
          <NeuronDiagram
            inputs={[x1, x2]}
            weights={[w1, w2]}
            bias={bias}
            z={z}
            a={a}
            activationLabel={act.label}
          />
        </Panel>

        <div className="stack">
          <Panel title="Inputs and parameters">
            <div className="controls">
              <Slider label={<>x₁ input</>} min={-2} max={2} value={x1} onChange={setX1} display={fmt(x1)} />
              <Slider label={<>x₂ input</>} min={-2} max={2} value={x2} onChange={setX2} display={fmt(x2)} />
              <Slider label={<>w₁ weight</>} min={-2} max={2} value={w1} onChange={setW1} display={fmt(w1)} />
              <Slider label={<>w₂ weight</>} min={-2} max={2} value={w2} onChange={setW2} display={fmt(w2)} />
              <Slider label={<>b bias</>} min={-2} max={2} value={bias} onChange={setBias} display={fmt(bias)} />
            </div>
            <div style={{ marginTop: 14 }}>
              <Segmented
                label="Activation f"
                value={activation}
                options={HIDDEN_ACTIVATIONS.map((name) => ({
                  value: name,
                  label: ACTIVATIONS[name].label,
                }))}
                onChange={setActivation}
              />
            </div>
          </Panel>

          <Panel title="The calculation">
            <div className="calc">
              <div className="calc__line">
                <span className="calc__label">z = </span>w₁x₁ + w₂x₂ + b
              </div>
              <div className="calc__line">
                <span className="calc__label">z = </span>
                {term(w1, x1)} + {term(w2, x2)} + {fmt(bias, 2)}
              </div>
              <div className="calc__line">
                <span className="calc__label">z = </span>
                {fmt(w1 * x1, 3)} + {fmt(w2 * x2, 3)} + {fmt(bias, 2)}
              </div>
              <div className="calc__line">
                <span className="calc__label">z = </span>
                <span className="calc__result">{fmt(z, 3)}</span>
              </div>
              <div className="calc__line" style={{ marginTop: 6 }}>
                <span className="calc__label">a = </span>
                {act.label}({fmt(z, 3)}) = <span className="calc__result">{fmt(a, 3)}</span>
              </div>
            </div>
          </Panel>
        </div>
      </div>

      <div className="grid grid--side">
        <div className="prose-block">
          <h3 className="subhead">What a single neuron can represent</h3>
          <p>
            The set of points where <M>{'z = 0'}</M> is the line{' '}
            <M>{'w_1 x_1 + w_2 x_2 + b = 0'}</M>. The weight vector{' '}
            <M>{'\\mathbf{w} = (w_1, w_2)'}</M> is perpendicular to that line and points towards
            increasing <M>{'z'}</M>; the bias shifts the line without rotating it. With{' '}
            <M>{'n'}</M> inputs the same expression defines a hyperplane in{' '}
            <M>{'\\mathbb{R}^n'}</M>.
          </p>
          <p>
            The activation decides what happens on either side of that boundary. It does not move
            the boundary: <M>{'f'}</M> is applied after the sum, and every activation used here is
            monotonic, so the sign of <M>{'z'}</M> determines which branch of <M>{'f'}</M> applies.
            One neuron is therefore a linear boundary followed by a fixed non-linear response.
          </p>
          <p>
            Set <M>{'w_1 = 1'}</M>, <M>{'w_2 = 1'}</M>, <M>{'b = 0'}</M> and the line is the
            anti-diagonal. Increase <M>{'b'}</M> and it slides towards the lower left. This is the
            entire expressive power of a single unit, and the reason networks stack many of them.
          </p>

          <h3 className="subhead">z measures distance, scaled by ‖w‖</h3>
          <p>
            The pre-activation is not an arbitrary number: it is the signed distance from{' '}
            <M>{'\\mathbf{x}'}</M> to the boundary, multiplied by{' '}
            <M>{'\\lVert \\mathbf{w} \\rVert'}</M>.
          </p>
          <Equation caption="d is positive on the side the weight vector points towards, negative on the other.">
            {'z(\\mathbf{x}) = \\lVert \\mathbf{w} \\rVert \\cdot d(\\mathbf{x}), \\qquad d(\\mathbf{x}) = \\frac{\\mathbf{w}\\cdot\\mathbf{x} + b}{\\lVert \\mathbf{w} \\rVert}'}
          </Equation>
          <InWords tag="notation">
            <p>
              <M>{'\\lVert \\mathbf{w} \\rVert'}</M> is the length of the weight vector — for two
              weights it is <M>{'\\sqrt{w_1^2 + w_2^2}'}</M>, straight from Pythagoras.{' '}
              <M>{'\\mathbf{w}\\cdot\\mathbf{x}'}</M> is the sum of products{' '}
              <M>{'w_1x_1 + w_2x_2'}</M> from the top of this section.
            </p>
            <p>
              So the statement is: <M>{'z'}</M> equals how far the point sits from the line, times a
              fixed number that depends only on the weights. Points twice as far from the line get
              twice the <M>{'z'}</M>.
            </p>
          </InWords>
          <p>
            Two consequences follow. Scaling <M>{'\\mathbf{w}'}</M> and <M>{'b'}</M> by the same
            positive constant leaves the boundary fixed but multiplies every <M>{'z'}</M> by that
            constant — so the direction of <M>{'\\mathbf{w}'}</M> sets <em>where</em> the
            boundary is, and its magnitude sets <em>how sharply</em> the activation switches across
            it. And the boundary's distance from the origin is{' '}
            <M>{'|b| / \\lVert \\mathbf{w} \\rVert'}</M>, which is why a large bias relative to
            the weights pushes the boundary out of the region where the data lives.
          </p>

          <Detail title="z as a scaled distance">
            <p>
              Let <M>{'\\mathbf{x}_0'}</M> be any point on the boundary, so{' '}
              <M>{'\\mathbf{w}\\cdot\\mathbf{x}_0 + b = 0'}</M>. For an arbitrary{' '}
              <M>{'\\mathbf{x}'}</M>, decompose the displacement into a component along the unit
              normal <M>{'\\hat{\\mathbf{n}} = \\mathbf{w}/\\lVert\\mathbf{w}\\rVert'}</M>{' '}
              and a component in the boundary:
            </p>
            <Equation plain>
              {'\\mathbf{x} - \\mathbf{x}_0 = d\\,\\hat{\\mathbf{n}} + \\mathbf{t}, \\qquad \\mathbf{w}\\cdot\\mathbf{t} = 0'}
            </Equation>
            <p>Taking the inner product with <M>{'\\mathbf{w}'}</M>:</p>
            <Equation plain>
              {'\\mathbf{w}\\cdot\\mathbf{x} - \\mathbf{w}\\cdot\\mathbf{x}_0 = d\\,\\lVert\\mathbf{w}\\rVert'}
            </Equation>
            <p>
              Substituting <M>{'\\mathbf{w}\\cdot\\mathbf{x}_0 = -b'}</M> gives{' '}
              <M>{'\\mathbf{w}\\cdot\\mathbf{x} + b = d\\,\\lVert\\mathbf{w}\\rVert'}</M>,
              which is <M>{'z = \\lVert\\mathbf{w}\\rVert\\, d'}</M>. Setting{' '}
              <M>{'\\mathbf{x} = \\mathbf{0}'}</M> gives the distance from the origin to the
              boundary as <M>{'|b|/\\lVert\\mathbf{w}\\rVert'}</M>.
            </p>
            <p>
              Check it with the worked example: <M>{'\\mathbf{w} = (0.8, -0.3)'}</M>,{' '}
              <M>{'b = 0.2'}</M>, so <M>{'\\lVert\\mathbf{w}\\rVert = 0.854'}</M>. At{' '}
              <M>{'\\mathbf{x} = (1.0, 0.5)'}</M> we found <M>{'z = 0.85'}</M>, so the point sits{' '}
              <M>{'0.85/0.854 = 0.995'}</M> units from the boundary, on the positive side. The
              boundary itself passes <M>{'0.2/0.854 = 0.234'}</M> units from the origin.
            </p>
          </Detail>

          <h3 className="subhead">What one neuron cannot do</h3>
          <p>
            Because the level sets of <M>{'z'}</M> are parallel hyperplanes and <M>{'f'}</M> is
            monotonic, the set of inputs a single unit assigns to any output threshold is always a
            half-space. Any labelling that is not linearly separable is out of reach — no choice of{' '}
            <M>{'\\mathbf{w}'}</M> and <M>{'b'}</M> works, and gradient descent cannot help,
            because the problem is with the function family rather than with the search.
          </p>

          <Detail kicker="proof" title="No single neuron computes XOR">
            <p>
              XOR labels the four corners of the unit square:{' '}
              <M>{'(0,0) \\mapsto 0'}</M>, <M>{'(0,1) \\mapsto 1'}</M>,{' '}
              <M>{'(1,0) \\mapsto 1'}</M>, <M>{'(1,1) \\mapsto 0'}</M>. Suppose some{' '}
              <M>{'\\mathbf{w}, b'}</M> and monotonically increasing <M>{'f'}</M> reproduced it
              with the rule "output <M>{'\\ge'}</M> threshold means class 1". Monotonicity means
              the rule is equivalent to <M>{'z \\ge c'}</M> for some constant <M>{'c'}</M>. Write{' '}
              <M>{'z(x_1, x_2) = w_1x_1 + w_2x_2 + b'}</M>. The four requirements are:
            </p>
            <Equation plain>
              {'\\begin{aligned} b &< c \\\\ w_2 + b &\\ge c \\\\ w_1 + b &\\ge c \\\\ w_1 + w_2 + b &< c \\end{aligned}'}
            </Equation>
            <p>
              Add the second and third inequalities: <M>{'w_1 + w_2 + 2b \\ge 2c'}</M>. Add the
              first and fourth: <M>{'w_1 + w_2 + 2b < 2c'}</M>. The same quantity is both at least{' '}
              <M>{'2c'}</M> and strictly less than <M>{'2c'}</M>, a contradiction. No such{' '}
              <M>{'\\mathbf{w}, b, c'}</M> exist.
            </p>
            <p>
              The proof used only linearity and monotonicity, so it rules out every activation on
              this page. What it does not rule out is two neurons: the sum{' '}
              <M>{'x_1 + x_2'}</M> distinguishes the corners into three groups{' '}
              <M>{'\\{0\\}, \\{1\\}, \\{2\\}'}</M>, and one hidden layer that computes two
              different linear functions can separate the middle group from the outer two. Section
              11 shows exactly that, trained.
            </p>
          </Detail>

          <h3 className="subhead">The bias-as-input trick</h3>
          <p>
            Appending a constant 1 to the input and a column <M>{'b'}</M> to the weight matrix makes
            the bias an ordinary weight: <M>{'\\tilde{\\mathbf{x}} = [\\mathbf{x}; 1]'}</M> and{' '}
            <M>{'\\tilde{\\mathbf{w}} = [\\mathbf{w}; b]'}</M> give{' '}
            <M>{'z = \\tilde{\\mathbf{w}}\\cdot\\tilde{\\mathbf{x}}'}</M>. This is the
            dashed edge in the diagram above. It simplifies the algebra — everything becomes one
            inner product — but implementations usually keep the bias separate, because it is
            excluded from weight decay and because the batched form adds it by broadcasting rather
            than by materialising a column of ones.
          </p>
        </div>

        <Panel
          title="Input plane"
          hint="click to set x"
          caption="Shading shows the sign and magnitude of z across the input plane; the dark curve is the set z = 0. The marker is the current (x₁, x₂)."
        >
          <DecisionBoundary
            predict={predict}
            train={[]}
            version={w1 * 1000 + w2 * 100 + bias}
            domain={[-2, 2]}
            resolution={72}
            highlight={[x1, x2]}
            onPick={(px, py) => {
              setX1(clamp(Number(px.toFixed(2)), -2, 2));
              setX2(clamp(Number(py.toFixed(2)), -2, 2));
            }}
            hint="x₁ →, x₂ ↑"
          />
          <div style={{ marginTop: 12 }}>
            <Stats
              items={[
                { label: 'z', value: fmt(z, 3), accent: true },
                { label: 'a = f(z)', value: fmt(a, 3) },
                { label: "f'(z)", value: fmt(act.df(z), 3) },
              ]}
            />
          </div>
          <div className="btn-row" style={{ marginTop: 12 }}>
            <Button
              onClick={() => {
                setW1(0.8);
                setW2(-0.3);
                setBias(0.2);
                setX1(1);
                setX2(0.5);
              }}
            >
              Reset to the worked example
            </Button>
          </div>
        </Panel>
      </div>

      <Note title="Why the bias is a separate parameter" accent>
        <p>
          Without <M>{'b'}</M> the boundary <M>{'\\mathbf{w}\\cdot\\mathbf{x} = 0'}</M> always
          passes through the origin, so a neuron could only represent boundaries through that one
          point. Implementations often absorb the bias into the weight matrix by appending a
          constant input of 1 — that is what the dashed edge above shows — but it remains a
          trainable parameter either way.
        </p>
      </Note>

      <Exercise
        notebook="02-neurons.ipynb"
        count={3}
        tasks={[
          <>Implement <code>neuron</code> and reproduce z = 0.85 from the worked example</>,
          <>Write <code>signed_distance</code> and confirm ‖w‖ · d equals z</>,
          <>Sweep 68,921 combinations of (w₁, w₂, b) and find that none solves XOR</>,
          <>Then verify two hand-chosen hidden units do solve it</>,
        ]}
      >
        One neuron, its geometry, and a brute-force demonstration of what it cannot do.
      </Exercise>
    </Section>
  );
}
