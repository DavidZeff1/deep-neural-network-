/**
 * Gradient descent on explicit loss surfaces.
 *
 * These landscapes are not neural networks: they isolate the update rule
 * θ ← θ − η ∇L(θ) so that step size, curvature and convergence can be read off
 * a single curve.
 */

export interface Landscape1D {
  key: string;
  label: string;
  formula: string;
  gradientFormula: string;
  loss: (theta: number) => number;
  gradient: (theta: number) => number;
  domain: [number, number];
  /** Global arg min, found by a dense scan with local refinement. */
  minimum: number;
  notes: string;
}

function findMinimum(loss: (t: number) => number, [lo, hi]: [number, number]): number {
  let best = lo;
  let bestValue = Infinity;
  const steps = 4000;
  for (let i = 0; i <= steps; i++) {
    const t = lo + ((hi - lo) * i) / steps;
    const v = loss(t);
    if (v < bestValue) {
      bestValue = v;
      best = t;
    }
  }
  // Golden-section refinement around the best grid point.
  let a = best - (hi - lo) / steps;
  let b = best + (hi - lo) / steps;
  const phi = (Math.sqrt(5) - 1) / 2;
  for (let i = 0; i < 80; i++) {
    const c = b - phi * (b - a);
    const d = a + phi * (b - a);
    if (loss(c) < loss(d)) b = d;
    else a = c;
  }
  return (a + b) / 2;
}

function landscape(
  spec: Omit<Landscape1D, 'minimum'> & { minimum?: number },
): Landscape1D {
  return { ...spec, minimum: spec.minimum ?? findMinimum(spec.loss, spec.domain) };
}

export const LOSS_LANDSCAPES: Record<string, Landscape1D> = {
  quadratic: landscape({
    key: 'quadratic',
    label: 'Convex — L(θ) = θ²',
    formula: 'L(\\theta) = \\theta^{2}',
    gradientFormula: '\\nabla L(\\theta) = 2\\theta',
    loss: (t) => t * t,
    gradient: (t) => 2 * t,
    domain: [-3, 3],
    minimum: 0,
    notes:
      'One minimum, constant curvature L″ = 2. The update becomes θ ← θ(1 − 2η), so the distance to the minimum shrinks by the factor |1 − 2η| every step. Convergence requires η < 1; at η = 0.5 it lands exactly on the minimum in one step; above η = 1 the iterates grow without bound.',
  }),
  shifted: landscape({
    key: 'shifted',
    label: 'Shifted — L(θ) = 0.4(θ − 1.2)² + 0.15',
    formula: 'L(\\theta) = 0.4(\\theta - 1.2)^{2} + 0.15',
    gradientFormula: '\\nabla L(\\theta) = 0.8(\\theta - 1.2)',
    loss: (t) => 0.4 * (t - 1.2) ** 2 + 0.15,
    gradient: (t) => 0.8 * (t - 1.2),
    domain: [-3, 4],
    minimum: 1.2,
    notes:
      'Lower curvature (L″ = 0.8) shifts the stability limit to η < 2.5. Flatter surfaces tolerate — and need — larger steps. The minimum loss is 0.15, not 0: a non-zero loss at convergence does not mean training failed.',
  }),
  doubleWell: landscape({
    key: 'doubleWell',
    label: 'Two minima — quartic',
    formula: 'L(\\theta) = 0.08\\theta^{4} - 0.5\\theta^{2} + 0.15\\theta + 1.2',
    gradientFormula: '\\nabla L(\\theta) = 0.32\\theta^{3} - \\theta + 0.15',
    loss: (t) => 0.08 * t ** 4 - 0.5 * t * t + 0.15 * t + 1.2,
    gradient: (t) => 0.32 * t ** 3 - t + 0.15,
    domain: [-3.2, 3.2],
    notes:
      'Non-convex. Gradient descent is a local rule: it follows the slope beneath the current θ and settles in whichever basin it started in. The left minimum is slightly lower than the right one, but no step of the algorithm compares them.',
  }),
  wavy: landscape({
    key: 'wavy',
    label: 'Many minima — quadratic + sine',
    formula: 'L(\\theta) = 0.35\\theta^{2} + 0.55\\sin(3\\theta) + 1',
    gradientFormula: '\\nabla L(\\theta) = 0.7\\theta + 1.65\\cos(3\\theta)',
    loss: (t) => 0.35 * t * t + 0.55 * Math.sin(3 * t) + 1,
    gradient: (t) => 0.7 * t + 1.65 * Math.cos(3 * t),
    domain: [-3.4, 3.4],
    notes:
      'Several local minima along a broad convex trend. A small learning rate stops at the first flat point encountered; a larger one steps over narrow basins. Neither behaviour is a bug — both are the same rule applied to different step sizes.',
  }),
};

export const LANDSCAPE_KEYS = ['quadratic', 'shifted', 'doubleWell', 'wavy'];

export interface DescentStep {
  iteration: number;
  theta: number;
  loss: number;
  gradient: number;
  /** −η ∇L(θ), the displacement applied to reach the next iterate. */
  update: number;
}

/**
 * Runs `iterations` updates and returns `iterations + 1` records, starting from
 * the initial point. Diverging runs are stopped once θ leaves a safe range so
 * the values stay finite and plottable.
 */
export function gradientDescent1D(
  landscapeSpec: Landscape1D,
  start: number,
  learningRate: number,
  iterations: number,
): DescentStep[] {
  const steps: DescentStep[] = [];
  let theta = start;
  const limit = 1e6;
  for (let i = 0; i <= iterations; i++) {
    const gradient = landscapeSpec.gradient(theta);
    const update = -learningRate * gradient;
    steps.push({ iteration: i, theta, loss: landscapeSpec.loss(theta), gradient, update });
    if (i === iterations) break;
    theta = theta + update;
    if (!Number.isFinite(theta) || Math.abs(theta) > limit) {
      steps.push({
        iteration: i + 1,
        theta: Math.sign(theta) * limit,
        loss: landscapeSpec.loss(Math.sign(theta) * limit),
        gradient: Number.NaN,
        update: Number.NaN,
      });
      break;
    }
  }
  return steps;
}

// --- Two parameters: least squares for a line -------------------------------

/** Six fixed points; the loss surface over (w, b) is an exact paraboloid. */
export const REGRESSION_POINTS: Array<[number, number]> = [
  [-1.2, -0.9],
  [-0.6, -0.2],
  [-0.1, 0.25],
  [0.4, 0.55],
  [0.9, 1.3],
  [1.4, 1.5],
];

export interface Point2D {
  w: number;
  b: number;
}

/** L(w, b) = (1/n) Σ (w xᵢ + b − yᵢ)² */
export function regressionLoss({ w, b }: Point2D): number {
  let sum = 0;
  for (const [x, y] of REGRESSION_POINTS) {
    const e = w * x + b - y;
    sum += e * e;
  }
  return sum / REGRESSION_POINTS.length;
}

/** ∂L/∂w = (2/n) Σ xᵢ(w xᵢ + b − yᵢ),  ∂L/∂b = (2/n) Σ (w xᵢ + b − yᵢ) */
export function regressionGradient({ w, b }: Point2D): Point2D {
  let gw = 0;
  let gb = 0;
  for (const [x, y] of REGRESSION_POINTS) {
    const e = w * x + b - y;
    gw += 2 * x * e;
    gb += 2 * e;
  }
  const n = REGRESSION_POINTS.length;
  return { w: gw / n, b: gb / n };
}

export interface DescentStep2D {
  iteration: number;
  w: number;
  b: number;
  loss: number;
  gradW: number;
  gradB: number;
}

export function gradientDescent2D(
  start: Point2D,
  learningRate: number,
  iterations: number,
): DescentStep2D[] {
  const steps: DescentStep2D[] = [];
  let point = { ...start };
  for (let i = 0; i <= iterations; i++) {
    const grad = regressionGradient(point);
    steps.push({
      iteration: i,
      w: point.w,
      b: point.b,
      loss: regressionLoss(point),
      gradW: grad.w,
      gradB: grad.b,
    });
    if (i === iterations) break;
    point = { w: point.w - learningRate * grad.w, b: point.b - learningRate * grad.b };
    if (!Number.isFinite(point.w) || Math.abs(point.w) > 1e4 || Math.abs(point.b) > 1e4) break;
  }
  return steps;
}

/** Closed-form least-squares solution, used to mark the true minimum. */
export function regressionOptimum(): Point2D {
  const n = REGRESSION_POINTS.length;
  let sx = 0;
  let sy = 0;
  let sxx = 0;
  let sxy = 0;
  for (const [x, y] of REGRESSION_POINTS) {
    sx += x;
    sy += y;
    sxx += x * x;
    sxy += x * y;
  }
  const w = (n * sxy - sx * sy) / (n * sxx - sx * sx);
  const b = (sy - w * sx) / n;
  return { w, b };
}
