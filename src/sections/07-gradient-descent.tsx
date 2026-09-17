import { useEffect, useMemo, useState } from 'react';
import { Section } from '../components/layout/Section.tsx';
import type { SectionProps } from './registry.ts';
import {
  LANDSCAPE_KEYS,
  LOSS_LANDSCAPES,
  gradientDescent1D,
  gradientDescent2D,
  regressionCurvature,
  regressionLoss,
  regressionOptimum,
  scaledPoints,
} from '../lib/optimisation.ts';
import { Panel, Note, Stats } from '../components/ui/layout.tsx';
import { Detail } from '../components/ui/Detail.tsx';
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
  const [momentum, setMomentum] = useState(0);
  const [cursor, setCursor] = useState(0);
  const [playing, setPlaying] = useState(false);

  const landscape = LOSS_LANDSCAPES[key];
  const steps = useMemo(
    () => gradientDescent1D(landscape, start, learningRate, iterations, momentum),
    [landscape, start, learningRate, iterations, momentum],
  );
  const clampedCursor = Math.min(cursor, steps.length - 1);
  const current = steps[clampedCursor];

  useEffect(() => {
    setCursor(0);
    setPlaying(false);
  }, [key, start, learningRate, iterations, momentum]);

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
                label="Momentum β"
                min={0}
                max={0.95}
                step={0.01}
                value={momentum}
                onChange={setMomentum}
                display={fmt(momentum, 2)}
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
              { label: 'Velocity v', value: fmt(current.velocity, 4) },
              { label: 'Learning rate', value: fmt(learningRate, 3) },
            ]}
          />

          <Panel title="This step">
            <div className="calc">
              {momentum > 0 ? (
                <>
                  <div className="calc__line">v ← β · v + ∇L(θ)</div>
                  <div className="calc__line">
                    v ← {fmt(momentum, 2)} × {fmt((current.velocity - current.gradient) / Math.max(1e-9, momentum), 4)} +{' '}
                    {fmt(current.gradient, 4)} = <span className="calc__result">{fmt(current.velocity, 4)}</span>
                  </div>
                  <div className="calc__line">θ ← θ − η · v</div>
                  <div className="calc__line">
                    θ ← {fmt(current.theta, 4)} − {fmt(learningRate, 3)} × {fmt(current.velocity, 4)}
                  </div>
                </>
              ) : (
                <>
                  <div className="calc__line">θ ← θ − η · ∇L(θ)</div>
                  <div className="calc__line">
                    θ ← {fmt(current.theta, 4)} − {fmt(learningRate, 3)} × {fmt(current.gradient, 4)}
                  </div>
                </>
              )}
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
  const [momentum, setMomentum] = useState(0);
  const [featureScale, setFeatureScale] = useState(1);
  const [startW, setStartW] = useState(-1.4);
  const [startB, setStartB] = useState(1.5);
  const [cursor, setCursor] = useState(0);
  const [playing, setPlaying] = useState(false);

  const steps = useMemo(
    () => gradientDescent2D({ w: startW, b: startB }, learningRate, iterations, momentum, featureScale),
    [startW, startB, learningRate, iterations, momentum, featureScale],
  );
  const clamped = Math.min(cursor, steps.length - 1);
  const current = steps[clamped];
  const optimum = useMemo(() => regressionOptimum(featureScale), [featureScale]);
  const curvature = useMemo(() => regressionCurvature(featureScale), [featureScale]);
  const points = useMemo(() => scaledPoints(featureScale), [featureScale]);

  useEffect(() => {
    setCursor(0);
    setPlaying(false);
  }, [startW, startB, learningRate, iterations, momentum, featureScale]);

  // Rescaling the feature moves the optimum, so re-place the start relative to it.
  useEffect(() => {
    const target = regressionOptimum(featureScale);
    setStartW(Number((target.w - 2.2).toFixed(2)));
    setStartB(1.5);
  }, [featureScale]);

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

  const field = useMemo(
    () => (w: number, b: number) => Math.min(6, regressionLoss({ w, b }, featureScale)),
    [featureScale],
  );
  const path = steps.slice(0, clamped + 1).map((s) => [s.w, s.b] as [number, number]);

  return (
    <div className="grid grid--side">
      <Panel
        title="Loss surface over two parameters"
        hint="click to move the starting point"
        caption={
          momentum > 0
            ? 'Contours of L(w, b) for a straight-line fit. With momentum the step follows the accumulated velocity rather than the current gradient, so the path no longer crosses contours at right angles — and can carry past the minimum before reversing.'
            : 'Contours of L(w, b) for a straight-line fit. The gradient is perpendicular to the contour through the current point, so each step crosses contours at right angles.'
        }
      >
        <Plot
          xDomain={[optimum.w - 2.6, optimum.w + 1.9]}
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
            xDomain={[-1.6 * featureScale, 1.8 * featureScale]}
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
                    [-1.6 * featureScale, current.w * -1.6 * featureScale + current.b],
                    [1.8 * featureScale, current.w * 1.8 * featureScale + current.b],
                  ]}
                  scales={scales}
                  color={PATH_COLOR}
                  width={2}
                />
                {points.map(([x, y]) => (
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
            <Slider
              label="Feature scale s"
              min={0.15}
              max={3}
              step={0.05}
              value={featureScale}
              onChange={setFeatureScale}
              display={fmt(featureScale, 2)}
            />
            <Slider label="Learning rate η" min={0.01} max={1.2} step={0.005} value={learningRate} onChange={setLearningRate} display={fmt(learningRate, 3)} />
            <Slider label="Momentum β" min={0} max={0.95} step={0.01} value={momentum} onChange={setMomentum} display={fmt(momentum, 2)} />
            <Slider label="Iterations" min={1} max={200} step={1} value={iterations} onChange={setIterations} display={iterations} />
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
            { label: 'κ (condition no.)', value: fmt(curvature.conditionNumber, 2) },
            { label: 'η stability limit', value: fmt(curvature.maxStableRate, 3) },
          ]}
        />

        <Note>
          <p style={{ fontSize: 14 }}>
            The Hessian here is constant, so its eigenvalues{' '}
            <span className="mono">
              {fmt(curvature.eigenvalues[0], 2)}, {fmt(curvature.eigenvalues[1], 2)}
            </span>{' '}
            describe the whole surface. Learning rates at or above{' '}
            <span className="mono">{fmt(curvature.maxStableRate, 3)}</span> diverge. The best
            achievable contraction per step is{' '}
            <span className="mono">{fmt(curvature.bestContraction, 3)}</span>, so reaching a given
            accuracy takes roughly κ = {fmt(curvature.conditionNumber, 1)} times as many steps as it
            would on a perfectly round surface.
          </p>
        </Note>
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

      <div className="prose-block">
        <Detail kicker="derivation" title="Why the step is against the gradient, and why η has to be small">
          <p>
            Expand the loss around the current point to first order. For a small displacement{' '}
            <M>{'\\mathbf{d}'}</M>:
          </p>
          <Equation plain>
            {'L(\\theta + \\mathbf{d}) = L(\\theta) + \\nabla L(\\theta)\\cdot\\mathbf{d} + O(\\lVert \\mathbf{d}\\rVert^2)'}
          </Equation>
          <p>
            To decrease the loss as fast as possible we want the most negative first-order term, but
            that term can be made arbitrarily negative by taking <M>{'\\mathbf{d}'}</M> longer. So
            the question has to be asked at a fixed step length: among all{' '}
            <M>{'\\mathbf{d}'}</M> with <M>{'\\lVert\\mathbf{d}\\rVert = \\epsilon'}</M>,
            which minimises <M>{'\\nabla L \\cdot \\mathbf{d}'}</M>?
          </p>
          <p>
            By the Cauchy–Schwarz inequality,{' '}
            <M>{'\\nabla L\\cdot\\mathbf{d} \\ge -\\lVert\\nabla L\\rVert\\,\\epsilon'}</M>,
            with equality exactly when <M>{'\\mathbf{d}'}</M> points opposite to{' '}
            <M>{'\\nabla L'}</M>. So the steepest descent direction is{' '}
            <M>{'-\\nabla L/\\lVert\\nabla L\\rVert'}</M>, and writing the step as{' '}
            <M>{'-\\eta\\nabla L'}</M> folds the length and the normalisation into one constant.
          </p>
          <p>
            The <M>{'O(\\lVert\\mathbf{d}\\rVert^2)'}</M> term is what limits{' '}
            <M>{'\\eta'}</M>. The linear approximation is only trustworthy while the step is small
            enough that curvature has not changed the picture. Keep the second-order term:
          </p>
          <Equation plain>
            {'L(\\theta - \\eta\\nabla L) \\approx L(\\theta) - \\eta\\lVert\\nabla L\\rVert^2 + \\tfrac{1}{2}\\eta^2\\,\\nabla L^{\\top}H\\,\\nabla L'}
          </Equation>
          <p>
            The first-order term is linear in <M>{'\\eta'}</M> and helps; the second-order term is
            quadratic in <M>{'\\eta'}</M> and hurts. Beyond some <M>{'\\eta'}</M> the second wins
            and the step increases the loss. Bounding{' '}
            <M>{'\\nabla L^{\\top}H\\nabla L \\le \\lambda_{\\max}\\lVert\\nabla L\\rVert^2'}</M>{' '}
            gives a decrease whenever <M>{'\\eta < 2/\\lambda_{\\max}'}</M> — the stability bound
            that appears again below.
          </p>
        </Detail>

        <Detail title="Exact convergence rate on a quadratic">
          <p>
            Take <M>{'L(\\theta) = \\tfrac{1}{2}\\theta^{\\top}H\\theta'}</M> with{' '}
            <M>{'H'}</M> symmetric positive definite, so <M>{'\\nabla L = H\\theta'}</M> and the
            minimum is at the origin. The update is
          </p>
          <Equation plain>{'\\theta_{t+1} = \\theta_t - \\eta H\\theta_t = (I - \\eta H)\\theta_t'}</Equation>
          <p>
            Write <M>{'\\theta_t'}</M> in the eigenbasis of <M>{'H'}</M>. Each coordinate evolves
            independently: the component along the eigenvector with eigenvalue{' '}
            <M>{'\\lambda_i'}</M> is multiplied by <M>{'(1 - \\eta\\lambda_i)'}</M> every step,
            so after <M>{'t'}</M> steps it is <M>{'(1-\\eta\\lambda_i)^t'}</M> times its initial
            value.
          </p>
          <p>
            Convergence therefore requires <M>{'|1 - \\eta\\lambda_i| < 1'}</M> for every{' '}
            <M>{'i'}</M>, that is <M>{'0 < \\eta < 2/\\lambda_{\\max}'}</M>. The slowest
            coordinate is the one with the factor closest to 1, so the rate is governed by
          </p>
          <Equation plain>{'\\rho(\\eta) = \\max_i |1 - \\eta\\lambda_i|'}</Equation>
          <p>
            Minimising <M>{'\\rho'}</M> over <M>{'\\eta'}</M> balances the two extremes:{' '}
            <M>{'1 - \\eta\\lambda_{\\min} = \\eta\\lambda_{\\max} - 1'}</M>, giving{' '}
            <M>{'\\eta^{*} = 2/(\\lambda_{\\max}+\\lambda_{\\min})'}</M> and
          </p>
          <Equation plain>
            {'\\rho^{*} = \\frac{\\lambda_{\\max}-\\lambda_{\\min}}{\\lambda_{\\max}+\\lambda_{\\min}} = \\frac{\\kappa-1}{\\kappa+1}, \\qquad \\kappa = \\frac{\\lambda_{\\max}}{\\lambda_{\\min}}'}
          </Equation>
          <p>
            The number of steps to reduce the error by a fixed factor is{' '}
            <M>{'\\Theta(\\kappa)'}</M>. At <M>{'\\kappa = 1'}</M> one step suffices; at{' '}
            <M>{'\\kappa = 100'}</M> it takes roughly a hundred times as many. This is why the
            condition number, not the gradient magnitude, predicts how slow training will be — and
            why the two-parameter demo below lets you change it directly.
          </p>
        </Detail>
      </div>

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
          <h3 className="subhead">Momentum</h3>
          <p>
            Plain gradient descent forgets everything between steps. Momentum keeps a running
            average of past gradients and steps along that instead:
          </p>
          <Equation plain>
            {'\\mathbf{v} \\leftarrow \\beta\\mathbf{v} + \\nabla L(\\theta), \\qquad \\theta \\leftarrow \\theta - \\eta\\mathbf{v}'}
          </Equation>
          <p>
            Unrolling the recursion gives{' '}
            <M>{'\\mathbf{v}_t = \\sum_{k=0}^{t}\\beta^{k}\\nabla L(\\theta_{t-k})'}</M> — an
            exponentially weighted sum with effective horizon{' '}
            <M>{'1/(1-\\beta)'}</M> steps. At <M>{'\\beta = 0.9'}</M> that is about ten steps.
          </p>
          <p>
            Two effects follow. Components of the gradient that keep the same sign accumulate, so
            progress along a consistent slope is amplified by up to{' '}
            <M>{'1/(1-\\beta)'}</M>. Components that alternate sign — the oscillation across a
            narrow valley — cancel. Momentum therefore helps most when <M>{'\\kappa'}</M> is large,
            which is exactly when plain descent is slow.
          </p>
          <p>
            It is not free. Set the feature scale to 0.2 above, leave{' '}
            <M>{'\\eta'}</M> well below its limit, and raise <M>{'\\beta'}</M>: the path
            straightens and convergence accelerates sharply. Then push <M>{'\\beta'}</M> past 0.95
            and the iterates overshoot and ring, because the accumulated velocity carries the
            parameters past the minimum before the gradient can reverse it. The stable region is
            over the pair <M>{'(\\eta, \\beta)'}</M> jointly, not over either alone.
          </p>

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
          <p>
            The two standard responses. A <strong>schedule</strong> lowers{' '}
            <M>{'\\eta'}</M> over training — a large rate early, when the parameters are far from
            any minimum and the linear approximation is crude, and a small one late, when precision
            matters. Common forms are step decay, <M>{'\\eta_t = \\eta_0/(1+\\gamma t)'}</M>,
            and cosine decay. An <strong>adaptive</strong> method keeps a separate effective rate
            per parameter, estimated from the recent magnitude of that parameter's gradients: RMSProp
            divides by a running root-mean-square, and Adam combines that with momentum. Both attack
            the same problem — one <M>{'\\eta'}</M> cannot suit every curvature at once.
          </p>

          <h3 className="subhead">Conditioning is something you control</h3>
          <p>
            The condition number is not a fixed property of the problem. It depends on how the
            inputs are represented. Multiply one input feature by{' '}
            <M>{'s'}</M> and the corresponding row and column of the Hessian scale by{' '}
            <M>{'s'}</M> and <M>{'s^2'}</M>, stretching the loss surface along that axis. The
            feature-scale slider in the two-parameter demo does exactly this: at{' '}
            <M>{'s = 1'}</M> the surface is nearly round and <M>{'\\kappa \\approx 1.5'}</M>; at{' '}
            <M>{'s = 0.2'}</M> it becomes a long narrow valley with{' '}
            <M>{'\\kappa \\approx 32'}</M>, and the same learning rate now zig-zags.
          </p>
          <p>
            Standardising each input to zero mean and unit variance is the cheapest possible
            preconditioner: it makes the diagonal of the Hessian roughly uniform, which usually cuts{' '}
            <M>{'\\kappa'}</M> by orders of magnitude. It costs two numbers per feature and often
            matters more than any optimiser choice.
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
