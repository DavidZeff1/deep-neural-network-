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
  /** The velocity used for this step; equals the gradient when β = 0. */
  velocity: number;
  /** −η v, the displacement applied to reach the next iterate. */
  update: number;
}

/**
 * Runs `iterations` updates and returns `iterations + 1` records, starting from
 * the initial point.
 *
 * With `momentum` β > 0 the rule becomes the standard heavy-ball form
 *   v ← β v + ∇L(θ),  θ ← θ − η v
 * which reduces to plain gradient descent at β = 0.
 *
 * Diverging runs are stopped once θ leaves a safe range so the values stay
 * finite and plottable.
 */
export function gradientDescent1D(
  landscapeSpec: Landscape1D,
  start: number,
  learningRate: number,
  iterations: number,
  momentum = 0,
): DescentStep[] {
  const steps: DescentStep[] = [];
  let theta = start;
  let velocity = 0;
  const limit = 1e6;
  for (let i = 0; i <= iterations; i++) {
    const gradient = landscapeSpec.gradient(theta);
    velocity = momentum * velocity + gradient;
    const update = -learningRate * velocity;
    steps.push({
      iteration: i,
      theta,
      loss: landscapeSpec.loss(theta),
      gradient,
      velocity,
      update,
    });
    if (i === iterations) break;
    theta = theta + update;
    if (!Number.isFinite(theta) || Math.abs(theta) > limit) {
      const clamped = Math.sign(theta) * limit;
      steps.push({
        iteration: i + 1,
        theta: clamped,
        loss: landscapeSpec.loss(clamped),
        gradient: Number.NaN,
        velocity: Number.NaN,
        update: Number.NaN,
      });
      break;
    }
  }
  return steps;
}

// --- Two parameters: least squares for a line -------------------------------

/**
 * Six fixed points. The loss surface over (w, b) is an exact paraboloid, so its
 * curvature can be computed in closed form.
 *
 * `featureScale` multiplies every x value. It does not change the problem in
 * any meaningful sense — the optimal line still passes through the same points —
 * but it stretches one axis of the loss surface relative to the other, which is
 * exactly what unnormalised features do to a real objective.
 */
export const REGRESSION_POINTS: Array<[number, number]> = [
  [-1.2, -0.9],
  [-0.6, -0.2],
  [-0.1, 0.25],
  [0.4, 0.55],
  [0.9, 1.3],
  [1.4, 1.5],
];

export function scaledPoints(featureScale = 1): Array<[number, number]> {
  return REGRESSION_POINTS.map(([x, y]) => [x * featureScale, y]);
}

export interface Point2D {
  w: number;
  b: number;
}

/** L(w, b) = (1/n) Σ (w xᵢ + b − yᵢ)² */
export function regressionLoss({ w, b }: Point2D, featureScale = 1): number {
  let sum = 0;
  for (const [x, y] of REGRESSION_POINTS) {
    const e = w * (x * featureScale) + b - y;
    sum += e * e;
  }
  return sum / REGRESSION_POINTS.length;
}

/** ∂L/∂w = (2/n) Σ xᵢ(w xᵢ + b − yᵢ),  ∂L/∂b = (2/n) Σ (w xᵢ + b − yᵢ) */
export function regressionGradient({ w, b }: Point2D, featureScale = 1): Point2D {
  let gw = 0;
  let gb = 0;
  for (const [x0, y] of REGRESSION_POINTS) {
    const x = x0 * featureScale;
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
  velocityW: number;
  velocityB: number;
}

export function gradientDescent2D(
  start: Point2D,
  learningRate: number,
  iterations: number,
  momentum = 0,
  featureScale = 1,
): DescentStep2D[] {
  const steps: DescentStep2D[] = [];
  let point = { ...start };
  let velocity: Point2D = { w: 0, b: 0 };
  for (let i = 0; i <= iterations; i++) {
    const grad = regressionGradient(point, featureScale);
    velocity = {
      w: momentum * velocity.w + grad.w,
      b: momentum * velocity.b + grad.b,
    };
    steps.push({
      iteration: i,
      w: point.w,
      b: point.b,
      loss: regressionLoss(point, featureScale),
      gradW: grad.w,
      gradB: grad.b,
      velocityW: velocity.w,
      velocityB: velocity.b,
    });
    if (i === iterations) break;
    point = { w: point.w - learningRate * velocity.w, b: point.b - learningRate * velocity.b };
    if (!Number.isFinite(point.w) || Math.abs(point.w) > 1e4 || Math.abs(point.b) > 1e4) break;
  }
  return steps;
}

/** Closed-form least-squares solution, used to mark the true minimum. */
export function regressionOptimum(featureScale = 1): Point2D {
  const n = REGRESSION_POINTS.length;
  let sx = 0;
  let sy = 0;
  let sxx = 0;
  let sxy = 0;
  for (const [x0, y] of REGRESSION_POINTS) {
    const x = x0 * featureScale;
    sx += x;
    sy += y;
    sxx += x * x;
    sxy += x * y;
  }
  const w = (n * sxy - sx * sy) / (n * sxx - sx * sx);
  const b = (sy - w * sx) / n;
  return { w, b };
}

export interface Curvature {
  /** Eigenvalues of the Hessian, largest first. */
  eigenvalues: [number, number];
  /** λ_max / λ_min. Gradient descent needs ~κ steps per unit of progress. */
  conditionNumber: number;
  /** Learning rates at or above 2/λ_max diverge. */
  maxStableRate: number;
  /** Worst-case contraction factor per step at the best learning rate. */
  bestContraction: number;
}

/**
 * The least-squares Hessian is constant: H = (2/n) Σ [xᵢ, 1][xᵢ, 1]ᵀ. Its
 * eigenvalues determine both the stability limit and the convergence rate.
 */
export function regressionCurvature(featureScale = 1): Curvature {
  const n = REGRESSION_POINTS.length;
  let sxx = 0;
  let sx = 0;
  for (const [x0] of REGRESSION_POINTS) {
    const x = x0 * featureScale;
    sxx += x * x;
    sx += x;
  }
  const a = (2 / n) * sxx;
  const b = (2 / n) * sx;
  const d = 2;
  const trace = a + d;
  const det = a * d - b * b;
  const discriminant = Math.sqrt(Math.max(0, (trace / 2) ** 2 - det));
  const lambdaMax = trace / 2 + discriminant;
  const lambdaMin = Math.max(1e-12, trace / 2 - discriminant);
  const kappa = lambdaMax / lambdaMin;
  return {
    eigenvalues: [lambdaMax, lambdaMin],
    conditionNumber: kappa,
    maxStableRate: 2 / lambdaMax,
    bestContraction: (kappa - 1) / (kappa + 1),
  };
}
