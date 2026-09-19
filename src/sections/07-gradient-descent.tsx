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
import { Detail, InWords } from '../components/ui/Detail.tsx';
import { Button, Segmented, Slider } from '../components/ui/controls.tsx';
import { Equation, M } from '../components/ui/Math.tsx';
import { Exercise } from '../components/ui/Exercise.tsx';
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
      <InWords>
        <p>
          The arrow means "replace with". For every parameter: work out how much increasing it would
          increase the error — that is its entry of <M>{'\\nabla L'}</M> — multiply by a small
          number <M>{'\\eta'}</M>, and subtract the result.
        </p>
        <p>
          Subtracting is what sends it downhill. If raising a parameter raises the error its
          derivative is positive, so subtracting lowers the parameter. If raising it lowers the error
          the derivative is negative, and subtracting a negative raises the parameter. Either way the
          error goes down.
        </p>
        <p>
          <M>{'\\eta'}</M> — the Greek letter eta — controls how big a step to take. It is the
          single most consequential number in training, and the demo below is mostly about why.
        </p>
      </InWords>

      <p className="prose-block">
        The rule uses only local information — the value of the gradient at the current point. It
        does not know where the minimum is, how far away it is, or whether a lower one exists
        elsewhere.
      </p>

      <div className="prose-block">
        <Detail kicker="derivation" title="Why the step is against the gradient, and why η has to be small">
          <p>
            Two separate questions: which direction should the step go, and how long should it be?
          </p>
          <p>
            <strong>Direction.</strong> Zoom in close enough and the error surface looks like a flat
            slope. Moving by a small displacement <M>{'\\mathbf{d}'}</M> changes the loss by
            roughly
          </p>
          <Equation plain>{'\\Delta L \\approx \\nabla L \\cdot \\mathbf{d}'}</Equation>
          <p>
            That is the chain rule from section 00 with one term per parameter: how fast the loss
            responds to each parameter, times how far that parameter moved, added up.
          </p>
          <p>
            A product of two vectors like this can be rewritten using the angle{' '}
            <M>{'\\vartheta'}</M> between them:
          </p>
          <Equation plain>
            {'\\nabla L \\cdot \\mathbf{d} = \\lVert \\nabla L \\rVert \\; \\lVert \\mathbf{d} \\rVert \\; \\cos\\vartheta'}
          </Equation>
          <p>
            Fix the step length. The only quantity left to choose is the angle, and{' '}
            <M>{'\\cos\\vartheta'}</M> is most negative — giving the biggest decrease — at{' '}
            <M>{'\\vartheta = 180^{\\circ}'}</M>. So the best direction is exactly opposite the
            gradient. Anything more than 90° away from that increases the loss instead.
          </p>
          <p>
            That is the whole reason for the minus sign. Writing the step as{' '}
            <M>{'-\\eta\\nabla L'}</M> rather than dividing by the gradient's length folds the
            length into <M>{'\\eta'}</M>, with the side effect that steps are longer where the
            surface is steeper.
          </p>
          <p>
            <strong>Length.</strong> The approximation above is only good for small steps, because
            the slope itself changes as you move. Keeping one more term of accuracy:
          </p>
          <Equation plain>
            {'L(\\theta - \\eta\\nabla L) \\approx L(\\theta) - \\underbrace{\\eta\\lVert\\nabla L\\rVert^2}_{\\text{gain, grows like }\\eta} + \\underbrace{\\tfrac{1}{2}\\eta^2 c \\lVert\\nabla L\\rVert^2}_{\\text{penalty, grows like }\\eta^2}'}
          </Equation>
          <p>
            <M>{'c'}</M> measures how quickly the slope itself is changing — the curvature. The
            middle term is the decrease you wanted and grows in proportion to{' '}
            <M>{'\\eta'}</M>. The last term is the correction for the slope not staying put, and it
            grows with <M>{'\\eta'}</M> squared, so past some point it overwhelms the gain.
          </p>
          <p>Setting the two equal finds where a step stops helping at all:</p>
          <Equation plain>
            {'\\eta\\lVert\\nabla L\\rVert^2 = \\tfrac{1}{2}\\eta^2 c\\lVert\\nabla L\\rVert^2 \\;\\Longrightarrow\\; \\eta = \\frac{2}{c}'}
          </Equation>
          <p>
            Below <M>{'2/c'}</M> each step reduces the loss; above it each step increases the loss
            and the iterates grow without bound. Note what this says on its own: the flatter the
            surface, the larger the step you are allowed to take.
          </p>
        </Detail>

        <Detail title="How many steps convergence takes, and what decides it">
          <p>
            Start with one parameter and the simplest bowl,{' '}
            <M>{'L(\\theta) = \\tfrac{1}{2}c\\,\\theta^2'}</M>, whose minimum is at{' '}
            <M>{'\\theta = 0'}</M>. Its derivative is <M>{'c\\theta'}</M>, so one step is
          </p>
          <Equation plain>
            {'\\theta_{t+1} = \\theta_t - \\eta\\, c\\, \\theta_t = (1 - \\eta c)\\,\\theta_t'}
          </Equation>
          <p>
            Every step multiplies the distance to the minimum by the same fixed number{' '}
            <M>{'(1-\\eta c)'}</M>. After <M>{'t'}</M> steps the distance is{' '}
            <M>{'(1-\\eta c)^{t}'}</M> times what it started as. Three cases, all reproducible in
            the demo above:
          </p>
          <ul>
            <li>
              <M>{'0 < \\eta c < 1'}</M> — the multiplier is between 0 and 1, so the distance
              shrinks steadily and the iterates approach from one side.
            </li>
            <li>
              <M>{'1 < \\eta c < 2'}</M> — the multiplier is between −1 and 0, so the sign flips
              each step. The iterates overshoot and alternate sides, but shrink.
            </li>
            <li>
              <M>{'\\eta c > 2'}</M> — the multiplier is below −1 and the distance grows every step.
              Same <M>{'2/c'}</M> limit as before.
            </li>
          </ul>
          <p>
            <strong>Now more than one parameter.</strong> A bowl in two dimensions is generally not
            round: it is an ellipse, steep across the narrow direction and shallow along the long
            one. The useful fact is that such a surface always splits into a set of perpendicular
            directions that do not interfere with each other, each behaving exactly like the
            one-parameter case above with its own curvature.
          </p>
          <p>
            Those special directions are called <strong>eigenvectors</strong> and each one's
            curvature is its <strong>eigenvalue</strong>, written <M>{'\\lambda'}</M>. You do not
            need to compute them to use the result; the point is only that a multi-parameter problem
            is several single-parameter problems running side by side. In the two-parameter demo
            below, the long axis of the ellipse is the small-<M>{'\\lambda'}</M> direction and the
            short axis the large-<M>{'\\lambda'}</M> one.
          </p>
          <p>
            One learning rate has to serve all of them. It must stay below{' '}
            <M>{'2/\\lambda_{\\max}'}</M> or the steepest direction diverges. But then the
            shallowest direction shrinks by only <M>{'(1-\\eta\\lambda_{\\min})'}</M> per step,
            which is close to 1 when <M>{'\\lambda_{\\min}'}</M> is small. Overall progress is set
            by whichever direction is worst:
          </p>
          <Equation plain>
            {'\\text{shrink factor per step} = \\max_i \\left|1 - \\eta\\lambda_i\\right|'}
          </Equation>
          <p>
            The best <M>{'\\eta'}</M> balances the two extremes — fast enough for the shallow
            direction, slow enough for the steep one — and gives
          </p>
          <Equation plain>
            {'\\text{best shrink factor} = \\frac{\\kappa - 1}{\\kappa + 1}, \\qquad \\kappa = \\frac{\\lambda_{\\max}}{\\lambda_{\\min}}'}
          </Equation>
          <p>
            <M>{'\\kappa'}</M> is the <strong>condition number</strong> — the ratio of steepest
            curvature to shallowest, which is just how elongated the bowl is. Putting numbers in:
          </p>
          <table className="data">
            <thead>
              <tr>
                <th>κ</th>
                <th>shrink per step</th>
                <th>steps to shrink the error 1000×</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td>1 (perfectly round)</td>
                <td>0</td>
                <td>1</td>
              </tr>
              <tr>
                <td>3</td>
                <td>0.50</td>
                <td>10</td>
              </tr>
              <tr>
                <td>10</td>
                <td>0.82</td>
                <td>35</td>
              </tr>
              <tr>
                <td>100</td>
                <td>0.98</td>
                <td>344</td>
              </tr>
              <tr>
                <td>1000</td>
                <td>0.998</td>
                <td>3453</td>
              </tr>
            </tbody>
          </table>
          <p>
            Roughly, the work is proportional to <M>{'\\kappa'}</M>. This is the most useful single
            fact about gradient descent: its speed depends on the <em>shape</em> of the surface, not
            on how steep it is or how far away the minimum is. The two-parameter demo lets you change{' '}
            <M>{'\\kappa'}</M> directly and watch the cost.
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
          <InWords>
            <p>
              Keep a running quantity <M>{'\\mathbf{v}'}</M>. Each step, shrink it a little — that
              is the <M>{'\\beta'}</M> factor, typically 0.9 — then add the current gradient to it.
              Step along <M>{'\\mathbf{v}'}</M> instead of along the gradient.
            </p>
            <p>
              The effect: a direction the gradient keeps pointing in builds up, while a direction
              that flips back and forth cancels itself out. At <M>{'\\beta = 0'}</M> nothing is kept
              and this is ordinary gradient descent again.
            </p>
          </InWords>
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

      <Exercise
        notebook="07-gradient-descent.ipynb"
        count={4}
        tasks={[
          <>Write <code>gradient_descent</code> and return the whole trajectory</>,
          <>Search for the largest learning rate that converges, and compare it to 2/c</>,
          <>Compute a condition number from the Hessian and watch κ rise as a feature is rescaled</>,
          <>Implement momentum, and find where too much of it starts to hurt</>,
        ]}
      >
        The update rule, its stability limit found by search, and momentum.
      </Exercise>
    </Section>
  );
}
