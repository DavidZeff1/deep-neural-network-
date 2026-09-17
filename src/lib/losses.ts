/**
 * Loss functions.
 *
 * `value` is the loss for a single example. `gradOutput` is ∂L/∂ŷ, the
 * derivative with respect to the network's output activation.
 */

export type LossName = 'mse' | 'bce' | 'cce';

export interface LossFunction {
  name: LossName;
  label: string;
  formula: string;
  gradientFormula: string;
  /** Loss for one example. */
  value: (predicted: number[], target: number[]) => number;
  /** ∂L/∂ŷ for one example. */
  gradOutput: (predicted: number[], target: number[]) => number[];
  notes: string;
}

/** Keeps logarithms finite when a prediction saturates at 0 or 1. */
export const EPS = 1e-12;

const clip = (p: number) => Math.min(1 - EPS, Math.max(EPS, p));

export const LOSSES: Record<LossName, LossFunction> = {
  mse: {
    name: 'mse',
    label: 'Mean squared error',
    formula: 'L = \\dfrac{1}{n}\\sum_{i=1}^{n} (\\hat{y}_i - y_i)^2',
    gradientFormula: '\\dfrac{\\partial L}{\\partial \\hat{y}_i} = \\dfrac{2}{n}(\\hat{y}_i - y_i)',
    value: (predicted, target) => {
      let sum = 0;
      for (let i = 0; i < predicted.length; i++) {
        const d = predicted[i] - target[i];
        sum += d * d;
      }
      return sum / predicted.length;
    },
    gradOutput: (predicted, target) =>
      predicted.map((p, i) => (2 * (p - target[i])) / predicted.length),
    notes:
      'Used for regression. The penalty grows quadratically with the error, so a single large mistake dominates many small ones. Paired with a sigmoid output it produces a flat gradient when the prediction is confidently wrong, which is why classification uses cross-entropy instead.',
  },
  bce: {
    name: 'bce',
    label: 'Binary cross-entropy',
    formula: 'L = -\\bigl[\\, y\\log \\hat{y} + (1-y)\\log(1-\\hat{y}) \\,\\bigr]',
    gradientFormula:
      '\\dfrac{\\partial L}{\\partial \\hat{y}} = \\dfrac{\\hat{y} - y}{\\hat{y}(1 - \\hat{y})}',
    value: (predicted, target) => {
      let sum = 0;
      for (let i = 0; i < predicted.length; i++) {
        const p = clip(predicted[i]);
        sum += -(target[i] * Math.log(p) + (1 - target[i]) * Math.log(1 - p));
      }
      return sum / predicted.length;
    },
    gradOutput: (predicted, target) =>
      predicted.map((raw, i) => {
        const p = clip(raw);
        return (p - target[i]) / (p * (1 - p) * predicted.length);
      }),
    notes:
      'Used for a single binary outcome with a sigmoid output. The loss is unbounded: a confident wrong prediction costs −log(ε), which is large, so the gradient stays informative exactly where MSE goes flat.',
  },
  cce: {
    name: 'cce',
    label: 'Categorical cross-entropy',
    formula: 'L = -\\sum_{k=1}^{K} y_k \\log \\hat{y}_k',
    gradientFormula: '\\dfrac{\\partial L}{\\partial z_k} = \\hat{y}_k - y_k \\quad (\\text{with softmax})',
    value: (predicted, target) => {
      let sum = 0;
      for (let i = 0; i < predicted.length; i++) {
        sum += -target[i] * Math.log(clip(predicted[i]));
      }
      return sum;
    },
    gradOutput: (predicted, target) => predicted.map((p, i) => -target[i] / clip(p)),
    notes:
      'Used for K mutually exclusive classes with a softmax output. With a one-hot target only one term survives: L = −log ŷ_c for the correct class c. Composing softmax with this loss collapses the Jacobian to the difference ŷ − y.',
  },
};
