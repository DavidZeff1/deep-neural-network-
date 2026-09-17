/**
 * Activation functions.
 *
 * Each entry carries the scalar function, its derivative with respect to the
 * pre-activation z, LaTeX for both, and the plotting range used by the graphs.
 * Softmax is kept separate because it maps a vector to a vector and its
 * derivative is a Jacobian rather than a scalar.
 */

export type ActivationName = 'relu' | 'leakyRelu' | 'sigmoid' | 'tanh' | 'linear';

export interface Activation {
  name: ActivationName;
  label: string;
  /** f(z) */
  f: (z: number) => number;
  /** f'(z), expressed in terms of z */
  df: (z: number) => number;
  formula: string;
  derivative: string;
  /** Output range, used to scale plots and node colouring. */
  range: [number, number];
  notes: string;
}

export const LEAKY_SLOPE = 0.1;

export const ACTIVATIONS: Record<ActivationName, Activation> = {
  relu: {
    name: 'relu',
    label: 'ReLU',
    f: (z) => (z > 0 ? z : 0),
    df: (z) => (z > 0 ? 1 : 0),
    formula: 'f(z) = \\max(0, z)',
    derivative: "f'(z) = \\begin{cases} 1 & z > 0 \\\\ 0 & z < 0 \\end{cases}",
    range: [0, 4],
    notes:
      'Piecewise linear. The derivative is exactly 1 for positive z, so gradients pass through deep stacks without shrinking. For z < 0 the derivative is 0 and the unit stops learning while it stays there. The derivative at z = 0 is undefined; implementations pick 0 or 1 by convention.',
  },
  leakyRelu: {
    name: 'leakyRelu',
    label: 'Leaky ReLU',
    f: (z) => (z > 0 ? z : LEAKY_SLOPE * z),
    df: (z) => (z > 0 ? 1 : LEAKY_SLOPE),
    formula: 'f(z) = \\max(\\alpha z, z), \\quad \\alpha = 0.1',
    derivative: "f'(z) = \\begin{cases} 1 & z > 0 \\\\ \\alpha & z < 0 \\end{cases}",
    range: [-1, 4],
    notes:
      'ReLU with a small non-zero slope for negative z. The gradient never reaches exactly zero, so a unit pushed into the negative region can still recover.',
  },
  sigmoid: {
    name: 'sigmoid',
    label: 'Sigmoid',
    f: (z) => 1 / (1 + Math.exp(-z)),
    df: (z) => {
      const s = 1 / (1 + Math.exp(-z));
      return s * (1 - s);
    },
    formula: '\\sigma(z) = \\dfrac{1}{1 + e^{-z}}',
    derivative: "\\sigma'(z) = \\sigma(z)\\,\\bigl(1 - \\sigma(z)\\bigr)",
    range: [0, 1],
    notes:
      'Maps the real line to (0, 1), which makes it usable as a probability for a single binary outcome. Its derivative peaks at 0.25 when z = 0 and decays towards 0 for large |z|, so stacking many sigmoid layers multiplies numbers below 0.25 together and the gradient vanishes.',
  },
  tanh: {
    name: 'tanh',
    label: 'Tanh',
    f: (z) => Math.tanh(z),
    df: (z) => {
      const t = Math.tanh(z);
      return 1 - t * t;
    },
    formula: '\\tanh(z) = \\dfrac{e^{z} - e^{-z}}{e^{z} + e^{-z}}',
    derivative: "\\tanh'(z) = 1 - \\tanh^2(z)",
    range: [-1, 1],
    notes:
      'A rescaled sigmoid: tanh(z) = 2σ(2z) − 1. It is zero-centred, so activations entering the next layer have mean near zero, which usually trains faster than sigmoid. Its derivative peaks at 1, but still saturates for large |z|.',
  },
  linear: {
    name: 'linear',
    label: 'Linear',
    f: (z) => z,
    df: () => 1,
    formula: 'f(z) = z',
    derivative: "f'(z) = 1",
    range: [-4, 4],
    notes:
      'The identity. Used for regression outputs. A network whose hidden layers are all linear collapses to a single linear map: W₂(W₁x + b₁) + b₂ = (W₂W₁)x + (W₂b₁ + b₂). Depth adds nothing without a non-linear f.',
  },
};

export const HIDDEN_ACTIVATIONS: ActivationName[] = ['relu', 'tanh', 'sigmoid', 'leakyRelu'];

/** Numerically stable softmax. */
export function softmax(logits: number[]): number[] {
  if (logits.length === 0) return [];
  const max = Math.max(...logits);
  const exps = logits.map((z) => Math.exp(z - max));
  const sum = exps.reduce((acc, e) => acc + e, 0);
  return exps.map((e) => e / sum);
}

/**
 * Jacobian of softmax: ∂pᵢ/∂zⱼ = pᵢ(δᵢⱼ − pⱼ).
 * Returned as jac[i][j].
 */
export function softmaxJacobian(probabilities: number[]): number[][] {
  const n = probabilities.length;
  const jac: number[][] = [];
  for (let i = 0; i < n; i++) {
    const row: number[] = [];
    for (let j = 0; j < n; j++) {
      row.push(probabilities[i] * ((i === j ? 1 : 0) - probabilities[j]));
    }
    jac.push(row);
  }
  return jac;
}
