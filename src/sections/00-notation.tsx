import { useMemo, useState } from 'react';
import { Section } from '../components/layout/Section.tsx';
import type { SectionProps } from './registry.ts';
import { Panel, Note, Stats } from '../components/ui/layout.tsx';
import { Detail, InWords, Notation, Readout } from '../components/ui/Detail.tsx';
import { Button, Segmented, Slider } from '../components/ui/controls.tsx';
import { Equation, M } from '../components/ui/Math.tsx';
import { Curve, Marker, Plot, Polyline } from '../components/viz/Plot.tsx';
import { fmt, sub } from '../lib/format.ts';

const ACCENT = '#3b82f6';
const WARM = '#e0761f';

/* ---------------------------------------------------------------------------
   1. Sigma notation
   --------------------------------------------------------------------------- */

function SumExplorer() {
  const [count, setCount] = useState(3);
  const allWeights = [0.5, -1.2, 2.0, 0.8, -0.4];
  const allInputs = [4, 3, 2, 5, 1];
  const weights = allWeights.slice(0, count);
  const inputs = allInputs.slice(0, count);
  const products = weights.map((w, i) => w * inputs[i]);
  const total = products.reduce((a, b) => a + b, 0);
  const [highlight, setHighlight] = useState<number | null>(null);

  let running = 0;

  return (
    <div className="grid grid--side">
      <Panel
        title="Adding up a list of products"
        hint="hover a term"
        caption="The symbol Σ is an instruction: repeat the expression to its right once for each value of the counter, then add all the results together."
      >
        <Equation plain>
          {`\\sum_{i=1}^{${count}} w_i x_i`}
        </Equation>
        <div className="calc" style={{ marginTop: 8 }}>
          <div className="calc__line">
            <span className="calc__label">means </span>
            {weights.map((_, i) => (
              <span
                key={i}
                onMouseEnter={() => setHighlight(i)}
                onMouseLeave={() => setHighlight(null)}
                style={{
                  background: highlight === i ? 'var(--accent-soft)' : 'transparent',
                  borderRadius: 3,
                  padding: '1px 2px',
                  cursor: 'default',
                }}
              >
                {i > 0 ? ' + ' : ''}w{sub(i + 1)}x{sub(i + 1)}
              </span>
            ))}
          </div>
          <div className="calc__line">
            <span className="calc__label">with  </span>
            {weights.map((w, i) => (
              <span
                key={i}
                onMouseEnter={() => setHighlight(i)}
                onMouseLeave={() => setHighlight(null)}
                style={{
                  background: highlight === i ? 'var(--accent-soft)' : 'transparent',
                  borderRadius: 3,
                  padding: '1px 2px',
                }}
              >
                {i > 0 ? ' + ' : ''}({fmt(w, 1)})({fmt(inputs[i], 0)})
              </span>
            ))}
          </div>
          <div className="calc__line">
            <span className="calc__label">=     </span>
            {products.map((p, i) => (
              <span
                key={i}
                onMouseEnter={() => setHighlight(i)}
                onMouseLeave={() => setHighlight(null)}
                style={{
                  background: highlight === i ? 'var(--accent-soft)' : 'transparent',
                  borderRadius: 3,
                  padding: '1px 2px',
                }}
              >
                {i > 0 ? ' + ' : ''}
                {fmt(p, 1)}
              </span>
            ))}
          </div>
          <div className="calc__line">
            <span className="calc__label">=     </span>
            <span className="calc__result">{fmt(total, 1)}</span>
          </div>
        </div>

        <table className="data" style={{ marginTop: 14 }}>
          <thead>
            <tr>
              <th>i</th>
              <th>
                w<sub>i</sub>
              </th>
              <th>
                x<sub>i</sub>
              </th>
              <th>
                w<sub>i</sub>x<sub>i</sub>
              </th>
              <th>running total</th>
            </tr>
          </thead>
          <tbody>
            {products.map((p, i) => {
              running += p;
              return (
                <tr
                  key={i}
                  className={highlight === i ? 'is-current' : undefined}
                  onMouseEnter={() => setHighlight(i)}
                  onMouseLeave={() => setHighlight(null)}
                >
                  <td>{i + 1}</td>
                  <td>{fmt(weights[i], 1)}</td>
                  <td>{fmt(inputs[i], 0)}</td>
                  <td>{fmt(p, 1)}</td>
                  <td>{fmt(running, 1)}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </Panel>

      <div className="stack">
        <Panel title="How many terms">
          <Slider
            label="n — the number of terms"
            min={2}
            max={5}
            step={1}
            value={count}
            onChange={setCount}
            display={count}
          />
          <p className="faint" style={{ fontSize: 13, marginTop: 12, marginBottom: 0 }}>
            The three pieces of the symbol: <span className="mono">i = 1</span> underneath says the
            counter starts at 1, <span className="mono">n</span> on top says where it stops, and the
            expression to the right is what gets repeated and added.
          </p>
        </Panel>

        <InWords>
          <p>
            <M>{'\\sum_{i=1}^{n} w_i x_i'}</M> is shorthand for: take the first weight times the
            first input, add the second weight times the second input, keep going to the{' '}
            <M>{'n'}</M>-th, and report the total.
          </p>
          <p>
            Nothing more is happening. The notation exists so the same expression can describe 2
            terms or 2 million without being rewritten.
          </p>
        </InWords>

        <Note title="Where you will meet it">
          <p style={{ fontSize: 14 }}>
            This exact sum is what one neuron computes. In section 02 the weights come from the
            network, the inputs come from the data, and the total gets a name:{' '}
            <M>{'z'}</M>.
          </p>
        </Note>
      </div>
    </div>
  );
}

/* ---------------------------------------------------------------------------
   2. Matrix times vector
   --------------------------------------------------------------------------- */

const DEMO_W = [
  [0.5, -1.0],
  [2.0, 0.5],
  [-0.5, 1.5],
];

function MatrixVectorExplorer() {
  const [x, setX] = useState<[number, number]>([4, 2]);
  const [step, setStep] = useState(0);
  const results = DEMO_W.map((row) => row[0] * x[0] + row[1] * x[1]);
  const cell = 54;
  const gap = 6;

  return (
    <div className="grid grid--side">
      <Panel
        title="A matrix multiplied by a vector"
        hint={step === 0 ? 'press Next row' : `row ${step} of 3`}
        caption="One row at a time: pair each number in the row with the number in the same position of the vector, multiply the pairs, add the results. That single number becomes one entry of the output."
      >
        <svg className="viz" viewBox="0 0 400 230" role="img" aria-label="Matrix times vector">
          {/* W */}
          <text x={30 + cell} y={22} textAnchor="middle" style={{ fill: 'var(--text-muted)' }}>
            W — 3 rows, 2 columns
          </text>
          {DEMO_W.map((row, j) =>
            row.map((value, i) => {
              const active = step === j + 1;
              return (
                <g key={`${j}-${i}`}>
                  <rect
                    x={30 + i * (cell + gap)}
                    y={36 + j * (cell + gap)}
                    width={cell}
                    height={cell}
                    rx={4}
                    fill={active ? 'var(--accent-soft)' : 'var(--bg-inset)'}
                    stroke={active ? ACCENT : 'var(--border)'}
                    strokeWidth={active ? 2 : 1}
                    style={{ transition: 'all 0.2s var(--ease)' }}
                  />
                  <text
                    x={30 + i * (cell + gap) + cell / 2}
                    y={36 + j * (cell + gap) + cell / 2 + 4}
                    textAnchor="middle"
                    style={{ fill: 'var(--text)', fontSize: 13 }}
                  >
                    {fmt(value, 1)}
                  </text>
                </g>
              );
            }),
          )}

          <text x={172} y={116} textAnchor="middle" style={{ fill: 'var(--text-faint)', fontSize: 16 }}>
            ×
          </text>

          {/* x */}
          <text x={222} y={22} textAnchor="middle" style={{ fill: 'var(--text-muted)' }}>
            x — 2 numbers
          </text>
          {x.map((value, i) => (
            <g key={`x-${i}`}>
              <rect
                x={195}
                y={64 + i * (cell + gap)}
                width={cell}
                height={cell}
                rx={4}
                fill={step > 0 ? 'rgba(224, 118, 31, 0.16)' : 'var(--bg-inset)'}
                stroke={step > 0 ? WARM : 'var(--border)'}
                strokeWidth={step > 0 ? 2 : 1}
                style={{ transition: 'all 0.2s var(--ease)' }}
              />
              <text
                x={195 + cell / 2}
                y={64 + i * (cell + gap) + cell / 2 + 4}
                textAnchor="middle"
                style={{ fill: 'var(--text)', fontSize: 13 }}
              >
                {fmt(value, 1)}
              </text>
            </g>
          ))}

          <text x={290} y={116} textAnchor="middle" style={{ fill: 'var(--text-faint)', fontSize: 16 }}>
            =
          </text>

          {/* result */}
          <text x={345} y={22} textAnchor="middle" style={{ fill: 'var(--text-muted)' }}>
            result — 3 numbers
          </text>
          {results.map((value, j) => {
            const revealed = step > j;
            const active = step === j + 1;
            return (
              <g key={`r-${j}`}>
                <rect
                  x={318}
                  y={36 + j * (cell + gap)}
                  width={cell}
                  height={cell}
                  rx={4}
                  fill={active ? 'var(--accent-soft)' : 'var(--bg-inset)'}
                  stroke={active ? ACCENT : 'var(--border)'}
                  strokeWidth={active ? 2 : 1}
                  strokeDasharray={revealed ? undefined : '3 3'}
                  style={{ transition: 'all 0.2s var(--ease)' }}
                />
                <text
                  x={318 + cell / 2}
                  y={36 + j * (cell + gap) + cell / 2 + 4}
                  textAnchor="middle"
                  style={{ fill: revealed ? 'var(--text)' : 'var(--text-faint)', fontSize: 13 }}
                >
                  {revealed ? fmt(value, 1) : '?'}
                </text>
              </g>
            );
          })}

        </svg>

        <div className="calc" style={{ marginTop: 6 }}>
          {step === 0 ? (
            <div className="calc__line calc__label">
              Press "Next row" to work through the rows one at a time.
            </div>
          ) : (
            <>
              <div className="calc__line">
                <span className="calc__label">row {step} </span>[{DEMO_W[step - 1].map((v) => fmt(v, 1)).join(', ')}] paired with x = [
                {x.map((v) => fmt(v, 1)).join(', ')}]
              </div>
              <div className="calc__line">
                <span className="calc__label">       </span>({fmt(DEMO_W[step - 1][0], 1)})({fmt(x[0], 1)}) + (
                {fmt(DEMO_W[step - 1][1], 1)})({fmt(x[1], 1)})
              </div>
              <div className="calc__line">
                <span className="calc__label">       </span>= {fmt(DEMO_W[step - 1][0] * x[0], 2)} +{' '}
                {fmt(DEMO_W[step - 1][1] * x[1], 2)} ={' '}
                <span className="calc__result">{fmt(results[step - 1], 2)}</span>
              </div>
            </>
          )}
        </div>

        <div className="btn-row" style={{ marginTop: 10 }}>
          <Button onClick={() => setStep(0)} disabled={step === 0}>
            Reset
          </Button>
          <Button variant="primary" onClick={() => setStep((s) => Math.min(3, s + 1))} disabled={step === 3}>
            Next row →
          </Button>
          <Button onClick={() => setStep(3)} disabled={step === 3}>
            Show all
          </Button>
        </div>
      </Panel>

      <div className="stack">
        <Panel title="The input vector">
          <div className="stack stack--sm">
            <Slider label="x₁" min={-5} max={5} step={0.5} value={x[0]} onChange={(v) => setX([v, x[1]])} display={fmt(x[0], 1)} />
            <Slider label="x₂" min={-5} max={5} step={0.5} value={x[1]} onChange={(v) => setX([x[0], v])} display={fmt(x[1], 1)} />
          </div>
        </Panel>

        <InWords>
          <p>
            A <strong>vector</strong> is just an ordered list of numbers. A{' '}
            <strong>matrix</strong> is a grid of numbers arranged in rows and columns.
          </p>
          <p>
            Multiplying a matrix by a vector produces one output number per row. Each output is the
            sum-of-products from the previous panel, using that row's numbers as the weights.
          </p>
          <p>
            So <M>{'W\\mathbf{x}'}</M> is not a new idea — it is the same sum repeated once per row,
            written compactly.
          </p>
        </InWords>

        <Note title="Why the shapes have to match" accent>
          <p style={{ fontSize: 14 }}>
            Each row of <M>{'W'}</M> has 2 numbers and <M>{'\\mathbf{x}'}</M> has 2 numbers, so they
            pair up exactly. If <M>{'\\mathbf{x}'}</M> had 3 numbers there would be nothing to pair
            the third with, and the operation would be undefined.
          </p>
          <p style={{ fontSize: 14 }}>
            The rule: a matrix with <M>{'r'}</M> rows and <M>{'c'}</M> columns can multiply a vector
            of length <M>{'c'}</M>, and produces a vector of length <M>{'r'}</M>. Here 3×2 times
            length-2 gives length-3.
          </p>
        </Note>
      </div>
    </div>
  );
}

/* ---------------------------------------------------------------------------
   3. Derivative as slope
   --------------------------------------------------------------------------- */

const SLOPE_FUNCTIONS = {
  quadratic: {
    label: 'f(x) = x²',
    f: (x: number) => x * x,
    df: (x: number) => 2 * x,
    latex: 'f(x) = x^2',
    dlatex: "f'(x) = 2x",
    domain: [-3, 3] as [number, number],
    range: [-1, 9] as [number, number],
  },
  cubicish: {
    label: 'f(x) = x³/3 − x',
    f: (x: number) => (x * x * x) / 3 - x,
    df: (x: number) => x * x - 1,
    latex: 'f(x) = \\tfrac{1}{3}x^3 - x',
    dlatex: "f'(x) = x^2 - 1",
    domain: [-2.6, 2.6] as [number, number],
    range: [-2.2, 2.2] as [number, number],
  },
  sigmoid: {
    label: 'f(x) = 1/(1+e⁻ˣ)',
    f: (x: number) => 1 / (1 + Math.exp(-x)),
    df: (x: number) => {
      const s = 1 / (1 + Math.exp(-x));
      return s * (1 - s);
    },
    latex: 'f(x) = \\dfrac{1}{1+e^{-x}}',
    dlatex: "f'(x) = f(x)\\bigl(1-f(x)\\bigr)",
    domain: [-5, 5] as [number, number],
    range: [-0.2, 1.2] as [number, number],
  },
};

type SlopeKey = keyof typeof SLOPE_FUNCTIONS;

function SlopeExplorer() {
  const [key, setKey] = useState<SlopeKey>('quadratic');
  const [x0, setX0] = useState(1);
  const [logH, setLogH] = useState(-0.3); // h = 10^logH

  const spec = SLOPE_FUNCTIONS[key];
  const h = Math.pow(10, logH);
  const f0 = spec.f(x0);
  const f1 = spec.f(x0 + h);
  const rise = f1 - f0;
  const slope = rise / h;
  const exact = spec.df(x0);
  const error = Math.abs(slope - exact);

  return (
    <div className="grid grid--side">
      <Panel
        title="How fast the output changes when you nudge the input"
        hint="shrink h to close the gap"
        caption="The dashed line joins the two marked points. Its steepness is rise ÷ run. As the run h gets smaller the line settles onto the curve's true steepness at that point, which is what the derivative reports."
      >
        <Plot
          xDomain={spec.domain}
          yDomain={spec.range}
          height={300}
          xLabel="x"
          yLabel="f(x)"
          onPointerData={(point) =>
            setX0(Math.max(spec.domain[0] + 0.1, Math.min(spec.domain[1] - 0.6, point.x)))
          }
          ariaLabel="Curve with a secant line"
        >
          {(scales) => (
            <>
              <Curve f={spec.f} scales={scales} color={ACCENT} width={2.2} />
              {/* run and rise */}
              <line
                x1={scales.x(x0)}
                x2={scales.x(x0 + h)}
                y1={scales.y(f0)}
                y2={scales.y(f0)}
                stroke={WARM}
                strokeWidth={1.6}
              />
              <line
                x1={scales.x(x0 + h)}
                x2={scales.x(x0 + h)}
                y1={scales.y(f0)}
                y2={scales.y(f1)}
                stroke={WARM}
                strokeWidth={1.6}
              />
              <text
                x={(scales.x(x0) + scales.x(x0 + h)) / 2}
                y={scales.y(f0) + 15}
                textAnchor="middle"
                style={{
                  fill: WARM,
                  paintOrder: 'stroke',
                  stroke: 'var(--bg-panel)',
                  strokeWidth: 4,
                  strokeLinejoin: 'round',
                }}
              >
                run = {fmt(h, 3)}
              </text>
              <text
                x={scales.x(x0 + h) + 7}
                y={(scales.y(f0) + scales.y(f1)) / 2}
                style={{
                  fill: WARM,
                  paintOrder: 'stroke',
                  stroke: 'var(--bg-panel)',
                  strokeWidth: 4,
                  strokeLinejoin: 'round',
                }}
              >
                rise = {fmt(rise, 3)}
              </text>
              {/* secant, extended */}
              <Polyline
                points={[
                  [spec.domain[0], f0 + slope * (spec.domain[0] - x0)],
                  [spec.domain[1], f0 + slope * (spec.domain[1] - x0)],
                ]}
                scales={scales}
                color="var(--text-muted)"
                width={1.3}
                dash="5 4"
              />
              <Marker x={x0} y={f0} scales={scales} color={ACCENT} guides={false} />
              <Marker x={x0 + h} y={f1} scales={scales} color={WARM} guides={false} radius={4} />
            </>
          )}
        </Plot>
      </Panel>

      <div className="stack">
        <Panel title="Controls">
          <div className="stack stack--sm">
            <Segmented
              label="Function"
              value={key}
              options={(Object.keys(SLOPE_FUNCTIONS) as SlopeKey[]).map((k) => ({
                value: k,
                label: SLOPE_FUNCTIONS[k].label,
              }))}
              onChange={(v) => setKey(v)}
            />
            <Slider
              label="x — where on the curve"
              min={spec.domain[0] + 0.1}
              max={spec.domain[1] - 0.6}
              step={0.01}
              value={x0}
              onChange={setX0}
              display={fmt(x0, 2)}
            />
            <Slider
              label="h — how far you nudge"
              min={-3}
              max={0.3}
              step={0.01}
              value={logH}
              onChange={setLogH}
              display={fmt(h, 4)}
            />
          </div>
          <div className="btn-row" style={{ marginTop: 12 }}>
            <Button onClick={() => setLogH(0.3)}>Big nudge</Button>
            <Button onClick={() => setLogH(-3)}>Tiny nudge</Button>
          </div>
        </Panel>

        <Panel title="The arithmetic">
          <Readout
            rows={[
              { label: 'f(x)', value: fmt(f0, 5) },
              { label: 'f(x + h)', value: fmt(f1, 5) },
              { label: 'rise = f(x+h) − f(x)', value: fmt(rise, 5) },
              { label: 'run  = h', value: fmt(h, 5) },
              { label: 'rise ÷ run', value: fmt(slope, 5), accent: true },
              { label: "exact slope f'(x)", value: fmt(exact, 5) },
              { label: 'difference', value: fmt(error, 6) },
            ]}
          />
        </Panel>

        <InWords>
          <p>
            A <strong>derivative</strong> answers one question: if I increase the input by a tiny
            amount, how much does the output change, per unit of input?
          </p>
          <p>
            You measure it exactly as you would measure the steepness of a hill — go a short distance
            sideways (the run), see how far you rose (the rise), divide. The derivative is what that
            ratio settles down to as the sideways step shrinks to nothing.
          </p>
          <p>
            The notation <M>{"f'(x)"}</M> and{' '}
            <M>{'\\dfrac{df}{dx}'}</M> mean the same thing. The second one is written as a fraction
            because that is what it is: a change in <M>{'f'}</M> divided by a change in{' '}
            <M>{'x'}</M>.
          </p>
        </InWords>

        <Note title="Why the difference never quite reaches zero" accent>
          <p style={{ fontSize: 14 }}>
            Drag <M>{'h'}</M> to its smallest value and the difference gets very small but not
            exactly zero. Two reasons: the ratio only equals the derivative in the limit, and the
            computer stores numbers with about 16 digits of precision, so subtracting two nearly
            equal values loses accuracy. Around <M>{'h \\approx 10^{-8}'}</M> those two effects
            balance, which is why the gradient checks elsewhere on this site use{' '}
            <M>{'h = 10^{-5}'}</M> rather than something smaller.
          </p>
        </Note>
      </div>
    </div>
  );
}

/* ---------------------------------------------------------------------------
   4. Chain rule
   --------------------------------------------------------------------------- */

function ChainRuleExplorer() {
  const [x, setX] = useState(1.5);
  const [w, setW] = useState(0.8);
  const [b, setB] = useState(0.2);
  const [target, setTarget] = useState(1);
  const [logH, setLogH] = useState(-2);
  const [stage, setStage] = useState(3);

  const h = Math.pow(10, logH);
  const sigmoid = (v: number) => 1 / (1 + Math.exp(-v));

  const evaluate = (xv: number) => {
    const z = w * xv + b;
    const a = sigmoid(z);
    const loss = (a - target) ** 2;
    return { z, a, loss };
  };

  const base = evaluate(x);
  const nudged = evaluate(x + h);

  const dzdx = w;
  const dadz = base.a * (1 - base.a);
  const dLda = 2 * (base.a - target);
  const product = dLda * dadz * dzdx;
  const measured = (nudged.loss - base.loss) / h;

  const stages = [
    { name: 'input x', value: x, delta: h },
    { name: 'z = w·x + b', value: base.z, delta: nudged.z - base.z },
    { name: 'a = σ(z)', value: base.a, delta: nudged.a - base.a },
    { name: 'L = (a − y)²', value: base.loss, delta: nudged.loss - base.loss },
  ];

  return (
    <div className="stack">
      <Panel
        title="One nudge, travelling through three steps"
        hint={`nudge x by ${fmt(h, 4)}`}
        caption="Each box shows its current value and, underneath, how much that value moves when x alone is nudged. The change shrinks or grows at each step by that step's own rate."
      >
        <div className="pipeline">
          {stages.map((s, i) => (
            <div key={s.name} style={{ display: 'contents' }}>
              {i > 0 ? <div className="pipeline__arrow">→</div> : null}
              <div className={i <= stage ? 'pipeline__stage pipeline__stage--active' : 'pipeline__stage'}>
                <div className="pipeline__name">{s.name}</div>
                <div className="pipeline__value">{fmt(s.value, 4)}</div>
                <div className="pipeline__delta">
                  change {s.delta >= 0 ? '+' : ''}
                  {fmt(s.delta, 6)}
                </div>
              </div>
            </div>
          ))}
        </div>

        <div className="btn-row" style={{ marginTop: 14 }}>
          <Button onClick={() => setStage(0)} disabled={stage === 0}>
            Reset
          </Button>
          <Button variant="primary" onClick={() => setStage((s) => Math.min(3, s + 1))} disabled={stage === 3}>
            Follow the nudge →
          </Button>
        </div>
      </Panel>

      <div className="grid grid--side">
        <Panel title="The rate at each step, and their product">
          <table className="data">
            <thead>
              <tr>
                <th>step</th>
                <th>rate of change</th>
                <th>value here</th>
              </tr>
            </thead>
            <tbody>
              <tr className={stage >= 1 ? undefined : 'faint'}>
                <td>x changes z</td>
                <td>
                  <M>{'\\partial z/\\partial x = w'}</M>
                </td>
                <td>{fmt(dzdx, 4)}</td>
              </tr>
              <tr className={stage >= 2 ? undefined : 'faint'}>
                <td>z changes a</td>
                <td>
                  <M>{"\\partial a/\\partial z = \\sigma'(z)"}</M>
                </td>
                <td>{fmt(dadz, 4)}</td>
              </tr>
              <tr className={stage >= 3 ? undefined : 'faint'}>
                <td>a changes L</td>
                <td>
                  <M>{'\\partial L/\\partial a = 2(a - y)'}</M>
                </td>
                <td>{fmt(dLda, 4)}</td>
              </tr>
              <tr className="is-current">
                <td>all three multiplied</td>
                <td>
                  <M>{'\\partial L/\\partial x'}</M>
                </td>
                <td>{fmt(product, 6)}</td>
              </tr>
              <tr>
                <td>measured by nudging</td>
                <td>(L(x+h) − L(x)) / h</td>
                <td>{fmt(measured, 6)}</td>
              </tr>
            </tbody>
          </table>
          <p className="faint" style={{ fontSize: 13, marginTop: 12, marginBottom: 0 }}>
            The last two rows agree to {Math.max(0, -Math.floor(Math.log10(Math.max(1e-12, Math.abs(product - measured)))))}{' '}
            decimal places. Shrink <M>{'h'}</M> and they agree further; this is the chain rule being
            correct, not an approximation that happens to work.
          </p>
        </Panel>

        <div className="stack">
          <Panel title="Controls">
            <div className="stack stack--sm">
              <Slider label="x — the input" min={-3} max={3} step={0.05} value={x} onChange={setX} display={fmt(x, 2)} />
              <Slider label="w — the weight" min={-3} max={3} step={0.05} value={w} onChange={setW} display={fmt(w, 2)} />
              <Slider label="b — the bias" min={-2} max={2} step={0.05} value={b} onChange={setB} display={fmt(b, 2)} />
              <Segmented
                label="y — the target"
                value={String(target)}
                options={[
                  { value: '1', label: 'y = 1' },
                  { value: '0', label: 'y = 0' },
                ]}
                onChange={(v) => setTarget(Number(v))}
              />
              <Slider
                label="h — size of the nudge"
                min={-4}
                max={-1}
                step={0.05}
                value={logH}
                onChange={setLogH}
                display={fmt(h, 5)}
              />
            </div>
          </Panel>

          <InWords>
            <p>
              The <strong>chain rule</strong> says: when a change has to travel through several
              steps, multiply the rates.
            </p>
            <p>
              If moving <M>{'x'}</M> by 1 moves <M>{'z'}</M> by 0.8, and moving{' '}
              <M>{'z'}</M> by 1 moves <M>{'a'}</M> by 0.23, then moving{' '}
              <M>{'x'}</M> by 1 moves <M>{'a'}</M> by 0.8 × 0.23 = 0.184. Add a third step and you
              multiply by a third rate.
            </p>
            <p>
              That is all backpropagation does. A network is a long chain of steps, and the gradient
              of any weight is the product of the rates along the path from that weight to the loss.
            </p>
          </InWords>
        </div>
      </div>
    </div>
  );
}

/* ---------------------------------------------------------------------------
   Section
   --------------------------------------------------------------------------- */

export function NotationSection({ id, index }: SectionProps) {
  const glossary = useMemo(
    () => [
      { symbol: <M>{'x, y, z'}</M>, meaning: 'Single numbers. Italic lower case.' },
      {
        symbol: <M>{'\\mathbf{x}, \\mathbf{a}'}</M>,
        meaning: 'A list of numbers (a vector). Bold lower case.',
      },
      { symbol: <M>{'W'}</M>, meaning: 'A grid of numbers (a matrix). Capital letter.' },
      {
        symbol: <M>{'x_i'}</M>,
        meaning: (
          <>
            The <M>{'i'}</M>-th number in the list <M>{'\\mathbf{x}'}</M>. A subscript picks an item
            out.
          </>
        ),
      },
      {
        symbol: <M>{'W_{ji}'}</M>,
        meaning: (
          <>
            The number in row <M>{'j'}</M>, column <M>{'i'}</M> of the grid. Row first, then column.
          </>
        ),
      },
      {
        symbol: <M>{'\\mathbf{a}^{(l)}'}</M>,
        meaning: (
          <>
            A superscript in brackets is a label, not a power. This is the vector belonging to layer{' '}
            <M>{'l'}</M> — it does not mean raised to the power <M>{'l'}</M>.
          </>
        ),
      },
      {
        symbol: <M>{'\\sum'}</M>,
        meaning: 'Add up a list of things. The first panel above shows exactly how.',
      },
      { symbol: <M>{'\\prod'}</M>, meaning: 'The same idea, but multiply instead of add.' },
      {
        symbol: <M>{"f'(x)"}</M>,
        meaning: 'The derivative: how fast f changes as x changes. Also written df/dx.',
      },
      {
        symbol: <M>{'\\partial'}</M>,
        meaning:
          'A derivative when several inputs exist: change one of them, hold the rest still. Read "partial".',
      },
      {
        symbol: <M>{'\\nabla L'}</M>,
        meaning:
          'All the partial derivatives of L collected into one list, one per parameter. Read "grad L".',
      },
      {
        symbol: <M>{'W^{\\top}'}</M>,
        meaning: 'The grid with rows and columns swapped. Read "W transpose".',
      },
      {
        symbol: <M>{'\\odot'}</M>,
        meaning: 'Multiply two equal-length lists position by position: first × first, second × second.',
      },
      {
        symbol: <M>{'\\approx'}</M>,
        meaning: 'Approximately equal — close enough for the point being made.',
      },
      {
        symbol: <M>{'\\propto'}</M>,
        meaning: 'Proportional to — equal up to some constant factor that does not matter here.',
      },
      {
        symbol: <M>{'\\mathbb{E}[\\cdot]'}</M>,
        meaning: 'The average value of a quantity over many random draws. Read "expected value".',
      },
      {
        symbol: <M>{'\\operatorname{Var}(\\cdot)'}</M>,
        meaning: 'How spread out a quantity is: the average of its squared distance from its mean.',
      },
    ],
    [],
  );

  return (
    <Section
      id={id}
      index={index}
      eyebrow="Before you start"
      title="Reading the mathematics"
      lede={
        <>
          The rest of this course uses a small amount of notation, over and over. This section
          introduces all of it, one symbol at a time, with something you can move. Nothing here is
          specific to neural networks — it is the vocabulary the later sections are written in.
        </>
      }
    >
      <Note title="You can skip this" accent>
        <p>
          If <M>{'\\sum_i w_i x_i'}</M>, <M>{'\\partial L/\\partial w'}</M> and{' '}
          <M>{'W^{\\top}\\mathbf{v}'}</M> are already familiar, go straight to section 01. If any of
          them look like noise, spend ten minutes here first — every later section assumes exactly
          these four ideas and nothing more.
        </p>
      </Note>

      <h3 className="subhead">1. Σ — add up a list</h3>
      <p className="prose-block">
        A neuron's job is to combine many numbers into one. The symbol{' '}
        <M>{'\\sum'}</M> (a capital Greek sigma) is the instruction to do that combining.
      </p>
      <SumExplorer />

      <h3 className="subhead">2. Vectors, matrices, and what multiplying them means</h3>
      <p className="prose-block">
        A layer of a network is many neurons, each doing the sum above with its own weights. Writing
        that out one neuron at a time gets long, so the weights are stacked into a grid and the
        whole layer is written as one multiplication.
      </p>
      <MatrixVectorExplorer />

      <div className="grid grid--2">
        <div className="prose-block">
          <h3 className="subhead">Transpose</h3>
          <p>
            Flipping a grid so its rows become columns is called transposing it, written{' '}
            <M>{'W^{\\top}'}</M>. If <M>{'W'}</M> has 3 rows and 2 columns, then{' '}
            <M>{'W^{\\top}'}</M> has 2 rows and 3 columns, and the entry at row{' '}
            <M>{'j'}</M>, column <M>{'i'}</M> of one is the entry at row <M>{'i'}</M>, column{' '}
            <M>{'j'}</M> of the other.
          </p>
          <Equation plain>
            {'W = \\begin{bmatrix} 0.5 & -1.0 \\\\ 2.0 & 0.5 \\\\ -0.5 & 1.5 \\end{bmatrix} \\qquad W^{\\top} = \\begin{bmatrix} 0.5 & 2.0 & -0.5 \\\\ -1.0 & 0.5 & 1.5 \\end{bmatrix}'}
          </Equation>
          <InWords>
            <p>
              It appears in exactly one place in this course. Information flows forward through{' '}
              <M>{'W'}</M>; when the gradient flows backward it travels the same connections in the
              opposite direction, and reversing the direction is what the transpose expresses.
            </p>
          </InWords>
        </div>
        <div className="prose-block">
          <h3 className="subhead">Elementwise operations</h3>
          <p>
            Some operations act on whole lists but treat each position independently. Applying a
            function <M>{'f'}</M> to a vector means applying it to each entry separately:
          </p>
          <Equation plain>
            {'f\\!\\left(\\begin{bmatrix} 2.0 \\\\ -1.0 \\\\ 0.5 \\end{bmatrix}\\right) = \\begin{bmatrix} f(2.0) \\\\ f(-1.0) \\\\ f(0.5) \\end{bmatrix}'}
          </Equation>
          <p>
            The symbol <M>{'\\odot'}</M> means multiply two lists position by position — not the
            matrix multiplication above:
          </p>
          <Equation plain>
            {'\\begin{bmatrix} 2 \\\\ 3 \\end{bmatrix} \\odot \\begin{bmatrix} 5 \\\\ 10 \\end{bmatrix} = \\begin{bmatrix} 10 \\\\ 30 \\end{bmatrix}'}
          </Equation>
          <InWords>
            <p>
              Both appear in the backpropagation formula, where each unit's gradient is scaled by
              that unit's own slope — a position-by-position multiplication.
            </p>
          </InWords>
        </div>
      </div>

      <h3 className="subhead">3. Derivatives — how fast something changes</h3>
      <p className="prose-block">
        Training a network means adjusting numbers to make an error smaller. To do that you need to
        know, for each number, whether increasing it would make the error go up or down, and by how
        much. That quantity is a derivative.
      </p>
      <SlopeExplorer />

      <div className="grid grid--2">
        <div className="prose-block">
          <h3 className="subhead">Partial derivatives</h3>
          <p>
            A network's error depends on thousands of numbers at once. The derivative with respect
            to one of them — changing that one and holding every other still — is called a{' '}
            <strong>partial derivative</strong> and written with a curly{' '}
            <M>{'\\partial'}</M> instead of a straight <M>{'d'}</M>:
          </p>
          <Equation plain>{'\\frac{\\partial L}{\\partial w_3}'}</Equation>
          <InWords>
            <p>
              Read it as: "if I increase <M>{'w_3'}</M> by a tiny amount and change nothing else, how
              much does <M>{'L'}</M> change, per unit of <M>{'w_3'}</M>?"
            </p>
            <p>
              A positive value means increasing <M>{'w_3'}</M> increases the error, so training will
              decrease it. A negative value means the opposite. A value near zero means that
              particular number barely matters right now.
            </p>
          </InWords>
          <p>
            Collecting one partial derivative per parameter into a single list gives the{' '}
            <strong>gradient</strong>, written <M>{'\\nabla L'}</M>. It has exactly as many entries
            as the network has parameters. Computing it is what section 08 is about; using it is
            what section 07 is about.
          </p>
        </div>

        <div className="prose-block">
          <h3 className="subhead">Averages and spread</h3>
          <p>
            Two statistical terms appear in a handful of places. Both describe a quantity that varies
            — across examples in a dataset, or across random draws.
          </p>
          <p>
            <M>{'\\mathbb{E}[X]'}</M>, the <strong>expected value</strong>, is the average of{' '}
            <M>{'X'}</M> over all those draws. For the list 2, 4, 9 the expected value is{' '}
            <M>{'(2+4+9)/3 = 5'}</M>.
          </p>
          <p>
            <M>{'\\operatorname{Var}(X)'}</M>, the <strong>variance</strong>, measures how spread out
            the values are: take each value's distance from the average, square it, and average those
            squares. For 2, 4, 9 that is{' '}
            <M>{'\\bigl((-3)^2 + (-1)^2 + 4^2\\bigr)/3 = 8.67'}</M>. The square root, 2.94, is the{' '}
            <strong>standard deviation</strong>, which is in the same units as the original numbers
            and is usually the friendlier number to report.
          </p>
          <InWords>
            <p>
              Where they show up: the spread of the numbers flowing through a network decides whether
              it trains at all (section 03), and the spread of gradients across a mini-batch decides
              how noisy each training step is (section 09).
            </p>
          </InWords>
        </div>
      </div>

      <h3 className="subhead">4. The chain rule — following a change through several steps</h3>
      <p className="prose-block">
        This is the one piece of calculus the whole course rests on. A network computes its output in
        stages, so changing a weight changes the next stage, which changes the one after, all the way
        to the error. The chain rule says how to combine those stages into a single rate.
      </p>
      <ChainRuleExplorer />

      <Detail kicker="worked example" title="The chain rule with numbers you can check by hand">
        <p>
          Take <M>{'w = 2'}</M> and two steps: first double the input, then square the result. So{' '}
          <M>{'z = 2x'}</M> and <M>{'a = z^2'}</M>.
        </p>
        <p>
          Start at <M>{'x = 3'}</M>. Then <M>{'z = 6'}</M> and <M>{'a = 36'}</M>.
        </p>
        <p>
          Now nudge: <M>{'x = 3.001'}</M> gives <M>{'z = 6.002'}</M> and{' '}
          <M>{'a = 36.024004'}</M>. The input moved by 0.001 and the output moved by 0.024004, so the
          measured rate is <M>{'0.024004 / 0.001 = 24.004'}</M>.
        </p>
        <p>Now the chain rule. The two individual rates are:</p>
        <ul>
          <li>
            <M>{'\\partial z/\\partial x = 2'}</M> — doubling means every unit of{' '}
            <M>{'x'}</M> produces two units of <M>{'z'}</M>.
          </li>
          <li>
            <M>{'\\partial a/\\partial z = 2z = 12'}</M> — at <M>{'z = 6'}</M>, the squaring function
            has slope 12.
          </li>
        </ul>
        <p>
          Multiply them: <M>{'2 \\times 12 = 24'}</M>. That matches the measured 24.004, and the
          small excess is the second-order term that vanishes as the nudge shrinks. Try{' '}
          <M>{'x = 3.0001'}</M> and the measurement becomes 24.0004.
        </p>
        <p>
          The rule generalises to any number of stages: multiply the rate of every stage along the
          path. With three stages you multiply three numbers; with fifty layers you multiply fifty.
          That last sentence is why section 08 spends so long on what happens when those fifty
          numbers are each slightly below 1.
        </p>
      </Detail>

      <div className="grid grid--side">
        <div className="prose-block">
          <h3 className="subhead">Glossary</h3>
          <p>
            Everything the later sections use. Come back here whenever a symbol is unfamiliar rather
            than guessing at it.
          </p>
          <Notation entries={glossary} />
        </div>
        <div className="stack">
          <Stats
            items={[
              { label: 'Symbols to learn', value: '17' },
              { label: 'Calculus rules used', value: '1', accent: true },
              { label: 'Sections that need more', value: '0' },
            ]}
          />
          <Note title="One rule, used everywhere">
            <p style={{ fontSize: 14 }}>
              Backpropagation looks intimidating written out, but it applies the chain rule and
              nothing else. Every formula in section 08 is that one rule applied to a specific pair
              of connected quantities. If the chain-rule panel above makes sense, section 08 is
              bookkeeping.
            </p>
          </Note>
          <Note title="A note on superscripts" accent>
            <p style={{ fontSize: 14 }}>
              The single most common misreading: <M>{'\\mathbf{a}^{(2)}'}</M> is "the activations of
              layer 2", not "activations squared". The brackets mark it as a label. Powers are
              written without brackets, as in <M>{'z^2'}</M>.
            </p>
          </Note>
        </div>
      </div>
    </Section>
  );
}
