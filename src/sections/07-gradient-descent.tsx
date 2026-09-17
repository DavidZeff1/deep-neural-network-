import { useEffect, useMemo, useState } from 'react';
import { Section } from '../components/layout/Section.tsx';
import type { SectionProps } from './registry.ts';
import {
  LANDSCAPE_KEYS,
  LOSS_LANDSCAPES,
  REGRESSION_POINTS,
  gradientDescent1D,
  gradientDescent2D,
  regressionLoss,
  regressionOptimum,
} from '../lib/optimisation.ts';
import { Panel, Note, Stats } from '../components/ui/layout.tsx';
import { Button, Segmented, Slider } from '../components/ui/controls.tsx';
import { Equation, M } from '../components/ui/Math.tsx';
import { Curve, Marker, Plot, Polyline } from '../components/viz/Plot.tsx';
import { ContourField } from '../components/viz/ContourField.tsx';
import { fmt } from '../lib/format.ts';

const ACCENT = '#3b82f6';
const PATH_COLOR = '#e0761f';

function Descent1D() {
  const [key, setKey] = useState('quadratic');
  const [start, setStart] = useState(1.8);
  const [learningRate, setLearningRate] = useState(0.1);
  const [iterations, setIterations] = useState(30);
  const [cursor, setCursor] = useState(0);
  const [playing, setPlaying] = useState(false);

  const landscape = LOSS_LANDSCAPES[key];
  const steps = useMemo(
    () => gradientDescent1D(landscape, start, learningRate, iterations),
    [landscape, start, learningRate, iterations],
  );
  const clampedCursor = Math.min(cursor, steps.length - 1);
  const current = steps[clampedCursor];

  useEffect(() => {
    setCursor(0);
    setPlaying(false);
  }, [key, start, learningRate, iterations]);

  useEffect(() => {
    if (!playing) return;
    const handle = window.setTimeout(() => {
      setCursor((c) => {
        if (c >= steps.length - 1) {
          setPlaying(false);
          return c;
        }
        return c + 1;
      });
    }, 130);
    return () => window.clearTimeout(handle);
  }, [playing, clampedCursor, steps.length]);

  const yDomain = useMemo<[number, number]>(() => {
    const [lo, hi] = landscape.domain;
    let min = Infinity;
    let max = -Infinity;
    for (let i = 0; i <= 200; i++) {
      const value = landscape.loss(lo + ((hi - lo) * i) / 200);
      min = Math.min(min, value);
      max = Math.max(max, value);
    }
    const pad = (max - min) * 0.12;
    return [min - pad, max + pad];
  }, [landscape]);

  const visited = steps
    .slice(0, clampedCursor + 1)
    .map((s) => [s.theta, s.loss] as [number, number]);

  const diverged = Math.abs(current.theta) > Math.abs(landscape.domain[1]) * 3;

  return (
    <>
      <div className="grid grid--side">
        <Panel
          title={landscape.label}
          hint={`iteration ${current.iteration} of ${iterations}`}
          caption="Orange points are the iterates; the line joins consecutive values of θ. The tangent at the current point has slope ∇L(θ) — the step taken is proportional to it and points downhill."
        >
          <Plot
            xDomain={landscape.domain}
            yDomain={yDomain}
            height={320}
            xLabel="θ"
            yLabel="L(θ)"
            onPointerData={(point) => setStart(Number(point.x.toFixed(2)))}
            ariaLabel="Loss landscape with gradient descent trajectory"
          >
            {(scales) => (
              <>
                <Curve f={landscape.loss} scales={scales} color={ACCENT} width={2.2} />
                <line
                  x1={scales.x(landscape.minimum)}
                  x2={scales.x(landscape.minimum)}
                  y1={0}
                  y2={scales.innerHeight}
                  stroke="var(--text-faint)"
                  strokeDasharray="3 4"
                  opacity={0.8}
                />
                <text x={scales.x(landscape.minimum) + 5} y={12}>
                  minimum
                </text>
                <Polyline points={visited} scales={scales} color={PATH_COLOR} width={1.3} opacity={0.75} />
                {visited.map((point, i) => (
                  <circle
                    key={i}
                    cx={scales.x(point[0])}
                    cy={scales.y(point[1])}
                    r={i === visited.length - 1 ? 0 : 3}
                    fill={PATH_COLOR}
                    opacity={0.25 + (0.6 * i) / Math.max(1, visited.length)}
                  />
                ))}
                {/* Tangent line at the current iterate. */}
                {Number.isFinite(current.gradient) ? (
                  <Polyline
                    points={[
                      [current.theta - 0.55, current.loss - 0.55 * current.gradient],
                      [current.theta + 0.55, current.loss + 0.55 * current.gradient],
                    ]}
                    scales={scales}
                    color="var(--text-muted)"
                    width={1.2}
                    dash="4 3"
                  />
                ) : null}
                <Marker x={current.theta} y={current.loss} scales={scales} color={PATH_COLOR} guides={false} radius={6} />
              </>
            )}
          </Plot>
        </Panel>

        <div className="stack">
          <Panel title="Controls">
            <div className="stack stack--sm">
              <Segmented
                label="Loss landscape"
                value={key}
                options={LANDSCAPE_KEYS.map((k) => ({
                  value: k,
                  label: LOSS_LANDSCAPES[k].label.split(' — ')[0],
                }))}
                onChange={setKey}
              />
              <Slider
                label="Starting θ₀"
                min={landscape.domain[0]}
                max={landscape.domain[1]}
                step={0.01}
                value={start}
                onChange={setStart}
                display={fmt(start, 2)}
              />
              <Slider
                label="Learning rate η"
                min={0.001}
                max={1.2}
                step={0.001}
                value={learningRate}
                onChange={setLearningRate}
                display={fmt(learningRate, 3)}
              />
              <Slider
                label="Iterations"
                min={1}
                max={120}
                step={1}
                value={iterations}
                onChange={setIterations}
                display={iterations}
              />
              <Slider
                label="Scrub to iteration"
                min={0}
                max={steps.length - 1}
                step={1}
                value={clampedCursor}
                onChange={setCursor}
                display={clampedCursor}
              />
            </div>
            <div className="btn-row" style={{ marginTop: 12 }}>
              <Button variant="primary" onClick={() => setPlaying(!playing)}>
                {playing ? 'Pause' : 'Run'}
              </Button>
              <Button onClick={() => setCursor((c) => Math.min(steps.length - 1, c + 1))}>Step</Button>
              <Button onClick={() => setCursor(0)}>Reset</Button>
            </div>
          </Panel>

          <Stats
            items={[
              { label: 'Iteration', value: current.iteration },
              { label: 'Parameter θ', value: fmt(current.theta, 3), accent: true },
              { label: 'Loss', value: fmt(current.loss, 4) },
              { label: 'Gradient', value: fmt(current.gradient, 4) },
              { label: 'Learning rate', value: fmt(learningRate, 3) },
            ]}
          />

          <Panel title="This step">
            <div className="calc">
              <div className="calc__line">θ ← θ − η · ∇L(θ)</div>
              <div className="calc__line">
                θ ← {fmt(current.theta, 4)} − {fmt(learningRate, 3)} × {fmt(current.gradient, 4)}
              </div>
              <div className="calc__line">
                θ ← <span className="calc__result">{fmt(current.theta + current.update, 4)}</span>
              </div>
            </div>
            {diverged ? (
              <p style={{ color: 'var(--warn)', fontSize: 13, marginTop: 10, marginBottom: 0 }}>
                The iterates are growing without bound: η is above the stability limit for this
                curvature. Each step overshoots the minimum by more than it started away from it.
              </p>
            ) : null}
          </Panel>
        </div>
      </div>

      <div className="grid grid--side">
        <div className="prose-block">
          <h3 className="subhead">{landscape.label}</h3>
          <Equation plain>{landscape.formula}</Equation>
          <Equation plain>{landscape.gradientFormula}</Equation>
          <p>{landscape.notes}</p>
        </div>

        <Panel title="Iteration log" flush>
          <div className="scroll-y">
            <table className="data">
              <thead>
                <tr>
                  <th>i</th>
                  <th>θ</th>
                  <th>L(θ)</th>
                  <th>∇L</th>
                  <th>−η∇L</th>
                </tr>
              </thead>
              <tbody>
                {steps.map((s) => (
                  <tr key={s.iteration} className={s.iteration === clampedCursor ? 'is-current' : undefined}>
                    <td>{s.iteration}</td>
                    <td>{fmt(s.theta, 4)}</td>
                    <td>{fmt(s.loss, 4)}</td>
                    <td>{fmt(s.gradient, 4)}</td>
                    <td>{fmt(s.update, 4)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Panel>
      </div>
    </>
  );
}

function Descent2D() {
  const [learningRate, setLearningRate] = useState(0.15);
  const [iterations, setIterations] = useState(40);
  const [startW, setStartW] = useState(-1.4);
  const [startB, setStartB] = useState(1.5);
  const [cursor, setCursor] = useState(0);
  const [playing, setPlaying] = useState(false);

  const steps = useMemo(
    () => gradientDescent2D({ w: startW, b: startB }, learningRate, iterations),
    [startW, startB, learningRate, iterations],
  );
  const clamped = Math.min(cursor, steps.length - 1);
  const current = steps[clamped];
  const optimum = useMemo(() => regressionOptimum(), []);

  useEffect(() => {
    setCursor(0);
    setPlaying(false);
  }, [startW, startB, learningRate, iterations]);

  useEffect(() => {
    if (!playing) return;
    const handle = window.setTimeout(() => {
      setCursor((c) => {
        if (c >= steps.length - 1) {
          setPlaying(false);
          return c;
        }
        return c + 1;
      });
    }, 110);
    return () => window.clearTimeout(handle);
  }, [playing, clamped, steps.length]);

  const field = useMemo(() => (w: number, b: number) => Math.min(6, regressionLoss({ w, b })), []);
  const path = steps.slice(0, clamped + 1).map((s) => [s.w, s.b] as [number, number]);

  return (
    <div className="grid grid--side">
      <Panel
        title="Loss surface over two parameters"
        hint="click to move the starting point"
        caption="Contours of L(w, b) for a straight-line fit. The gradient is perpendicular to the contour through the current point, so each step crosses contours at right angles."
      >
        <Plot
          xDomain={[-2, 2.6]}
          yDomain={[-1.6, 2]}
          height={320}
          xLabel="w"
          yLabel="b"
          grid={false}
          showZeroLines={false}
          onPointerData={(point) => {
            setStartW(Number(point.x.toFixed(2)));
            setStartB(Number(point.y.toFixed(2)));
          }}
          ariaLabel="Contour plot of the regression loss"
        >
          {(scales) => (
            <>
              <ContourField f={field} scales={scales} levelCount={10} />
              <Polyline points={path} scales={scales} color={PATH_COLOR} width={1.6} />
              {path.map((point, i) => (
                <circle
                  key={i}
                  cx={scales.x(point[0])}
                  cy={scales.y(point[1])}
                  r={2.4}
                  fill={PATH_COLOR}
                  opacity={0.3 + (0.6 * i) / Math.max(1, path.length)}
                />
              ))}
              <g>
                <line
                  x1={scales.x(optimum.w) - 6}
                  x2={scales.x(optimum.w) + 6}
                  y1={scales.y(optimum.b)}
                  y2={scales.y(optimum.b)}
                  stroke="var(--text)"
                  strokeWidth={1.4}
                />
                <line
                  x1={scales.x(optimum.w)}
                  x2={scales.x(optimum.w)}
                  y1={scales.y(optimum.b) - 6}
                  y2={scales.y(optimum.b) + 6}
                  stroke="var(--text)"
                  strokeWidth={1.4}
                />
              </g>
              <Marker x={current.w} y={current.b} scales={scales} color={PATH_COLOR} guides={false} radius={5.5} />
            </>
          )}
        </Plot>
      </Panel>

      <div className="stack">
        <Panel title="Fit">
          <Plot
            xDomain={[-1.6, 1.8]}
            yDomain={[-1.6, 2]}
            height={200}
            xLabel="x"
            yLabel="y"
            ariaLabel="Data points and current line"
          >
            {(scales) => (
              <>
                <Polyline
                  points={[
                    [-1.6, current.w * -1.6 + current.b],
                    [1.8, current.w * 1.8 + current.b],
                  ]}
                  scales={scales}
                  color={PATH_COLOR}
                  width={2}
                />
                {REGRESSION_POINTS.map(([x, y]) => (
                  <g key={`${x}-${y}`}>
                    <line
                      x1={scales.x(x)}
                      x2={scales.x(x)}
                      y1={scales.y(y)}
                      y2={scales.y(current.w * x + current.b)}
                      stroke="var(--text-faint)"
                      strokeWidth={1}
                      strokeDasharray="2 2"
                    />
                    <circle cx={scales.x(x)} cy={scales.y(y)} r={4} fill={ACCENT} />
                  </g>
                ))}
              </>
            )}
          </Plot>
        </Panel>

        <Panel title="Controls">
          <div className="stack stack--sm">
            <Slider label="Learning rate η" min={0.01} max={0.6} step={0.005} value={learningRate} onChange={setLearningRate} display={fmt(learningRate, 3)} />
            <Slider label="Iterations" min={1} max={120} step={1} value={iterations} onChange={setIterations} display={iterations} />
            <Slider label="Scrub" min={0} max={steps.length - 1} step={1} value={clamped} onChange={setCursor} display={clamped} />
          </div>
          <div className="btn-row" style={{ marginTop: 12 }}>
            <Button variant="primary" onClick={() => setPlaying(!playing)}>
              {playing ? 'Pause' : 'Run'}
            </Button>
            <Button onClick={() => setCursor((c) => Math.min(steps.length - 1, c + 1))}>Step</Button>
            <Button onClick={() => setCursor(0)}>Reset</Button>
          </div>
        </Panel>

        <Stats
          items={[
            { label: 'Iteration', value: current.iteration },
            { label: 'w', value: fmt(current.w, 3), accent: true },
            { label: 'b', value: fmt(current.b, 3), accent: true },
            { label: 'Loss', value: fmt(current.loss, 4) },
            { label: '∂L/∂w', value: fmt(current.gradW, 3) },
            { label: '∂L/∂b', value: fmt(current.gradB, 3) },
          ]}
        />
      </div>
    </div>
  );
}

export function GradientDescentSection({ id, index }: SectionProps) {
  const [tab, setTab] = useState<'one' | 'two'>('one');

  return (
    <Section
      id={id}
      index={index}
      title="Gradient descent"
      lede={
        <>
          Gradient descent repeatedly moves each parameter a small distance in the direction that
          decreases the loss fastest. The gradient <M>{'\\nabla L(\\theta)'}</M> points in the
          direction of steepest increase, so the step is taken against it.
        </>
      }
    >
      <Equation caption="η is the learning rate: the fraction of the gradient applied at each step.">
        {'\\theta \\leftarrow \\theta - \\eta\\, \\nabla L(\\theta)'}
      </Equation>

      <p className="prose-block">
        The rule uses only local information — the value of the gradient at the current point. It
        does not know where the minimum is, how far away it is, or whether a lower one exists
        elsewhere.
      </p>

      <Segmented
        value={tab}
        options={[
          { value: 'one' as const, label: 'One parameter' },
          { value: 'two' as const, label: 'Two parameters' },
        ]}
        onChange={setTab}
        ariaLabel="Number of parameters"
      />

      {tab === 'one' ? <Descent1D /> : <Descent2D />}

      <div className="grid grid--2">
        <div className="prose-block">
          <h3 className="subhead">The learning rate</h3>
          <p>
            For a quadratic loss with curvature <M>{"L'' = c"}</M>, one step multiplies the distance
            to the minimum by <M>{'|1 - \\eta c|'}</M>. Three regimes follow directly:
          </p>
          <ul>
            <li>
              <M>{'\\eta < 1/c'}</M>: monotone convergence, slower as <M>{'\\eta'}</M> shrinks.
            </li>
            <li>
              <M>{'1/c < \\eta < 2/c'}</M>: the iterates overshoot and alternate sides, but still
              converge.
            </li>
            <li>
              <M>{'\\eta > 2/c'}</M>: each step lands further away than the last, and the loss
              diverges.
            </li>
          </ul>
          <p>
            Real networks have a different curvature in every direction, so no single{' '}
            <M>{'\\eta'}</M> is ideal for all parameters. This is why learning-rate schedules and
            adaptive methods exist.
          </p>
        </div>
        <div className="prose-block">
          <h3 className="subhead">Batch, stochastic, mini-batch</h3>
          <p>
            The objective is an average over the dataset. Computing <M>{'\\nabla J'}</M> exactly
            requires a pass over all <M>{'m'}</M> examples. Three variants trade accuracy for speed:
          </p>
          <ul>
            <li>
              <strong>Batch:</strong> one update per full pass. Exact gradient, expensive.
            </li>
            <li>
              <strong>Stochastic:</strong> one update per example. Very noisy, very cheap.
            </li>
            <li>
              <strong>Mini-batch:</strong> one update per group of <M>{'B'}</M> examples, typically
              16–256. The gradient estimate has standard error proportional to{' '}
              <M>{'1/\\sqrt{B}'}</M>.
            </li>
          </ul>
          <p>
            The training sections below use mini-batch gradient descent, and the batch size is a
            control you can change.
          </p>
        </div>
      </div>

      <Note title="What the gradient is not" accent>
        <p>
          <M>{'\\nabla L'}</M> gives a direction and a local rate of change, not a distance to the
          minimum. A large gradient does not mean the minimum is far away, and a gradient near zero
          identifies a stationary point — a minimum, a maximum, or a saddle. In high dimensions
          saddles vastly outnumber local minima, which is why they, not local minima, dominate the
          difficulty of training deep networks.
        </p>
      </Note>
    </Section>
  );
}
