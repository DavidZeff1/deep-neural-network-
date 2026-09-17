/**
 * A dense feed-forward network (multilayer perceptron) with explicit forward
 * and backward passes.
 *
 * The implementation is deliberately written out in loops rather than hidden
 * behind a tensor library: every quantity the site visualises — zᴸ, aᴸ, δᴸ,
 * ∂L/∂W, ∂L/∂b — is a named value here.
 *
 * Index convention: W[l][j][i] is the weight from unit i of layer l to unit j
 * of layer l+1. Layer 0 is the input.
 */

import type { ActivationName } from './activations.ts';
import { ACTIVATIONS, softmax } from './activations.ts';
import type { LossName } from './losses.ts';
import { LOSSES } from './losses.ts';
import { makeGaussian, makeRng, shuffleInPlace } from './rng.ts';

export type OutputActivation = 'sigmoid' | 'softmax' | 'linear';

export interface NetworkConfig {
  inputSize: number;
  /** Units per hidden layer, e.g. [8, 6] for two hidden layers. */
  hiddenUnits: number[];
  outputSize: number;
  hiddenActivation: ActivationName;
  outputActivation: OutputActivation;
  loss: LossName;
  seed: number;
  /** L2 coefficient λ in L_total = L_data + (λ/2)Σ w². */
  l2?: number;
  /** Probability of dropping a hidden unit during training. */
  dropout?: number;
}

export interface LayerTrace {
  /** Pre-activations z = W a_prev + b. */
  z: number[];
  /** Activations a = f(z). */
  a: number[];
  /** Inverted-dropout mask actually applied to `a`, when training. */
  mask?: number[];
}

export interface ForwardTrace {
  input: number[];
  layers: LayerTrace[];
  output: number[];
}

export interface Gradients {
  /** dW[l][j][i] = ∂L/∂W[l][j][i] */
  dW: number[][][];
  /** db[l][j] = ∂L/∂b[l][j] */
  db: number[][];
  /** delta[l][j] = ∂L/∂z for unit j of layer l+1 (the layer's pre-activation). */
  delta: number[][];
  /** dA[l][i] = ∂L/∂a for unit i of layer l+1's *input* activations. */
  dA: number[][];
}

export interface Sample {
  x: number[];
  y: number[];
}

const zeros = (n: number): number[] => new Array<number>(n).fill(0);
const zeroMatrix = (rows: number, cols: number): number[][] =>
  Array.from({ length: rows }, () => zeros(cols));

export class MLP {
  readonly config: Required<NetworkConfig>;
  readonly sizes: number[];
  /** W[l] has shape [sizes[l+1], sizes[l]]. */
  W: number[][][];
  b: number[][];
  private rng: () => number;
  private gaussian: () => number;

  constructor(config: NetworkConfig) {
    this.config = { l2: 0, dropout: 0, ...config };
    this.sizes = [config.inputSize, ...config.hiddenUnits, config.outputSize];
    this.rng = makeRng(config.seed);
    this.gaussian = makeGaussian(this.rng);
    this.W = [];
    this.b = [];
    this.initialise();
  }

  /**
   * He initialisation for ReLU-family units (variance 2/fan_in), Glorot for
   * saturating units (variance 2/(fan_in + fan_out)). Both keep the variance of
   * the activations roughly constant as signals move through the layers.
   */
  initialise(seed?: number): void {
    if (seed !== undefined) {
      this.rng = makeRng(seed);
      this.gaussian = makeGaussian(this.rng);
    }
    this.W = [];
    this.b = [];
    for (let l = 0; l < this.sizes.length - 1; l++) {
      const fanIn = this.sizes[l];
      const fanOut = this.sizes[l + 1];
      const isHidden = l < this.sizes.length - 2;
      const activation = isHidden ? this.config.hiddenActivation : this.config.outputActivation;
      const reluFamily = activation === 'relu' || activation === 'leakyRelu';
      const std = reluFamily ? Math.sqrt(2 / fanIn) : Math.sqrt(2 / (fanIn + fanOut));
      const layerW: number[][] = [];
      for (let j = 0; j < fanOut; j++) {
        const row: number[] = [];
        for (let i = 0; i < fanIn; i++) row.push(this.gaussian() * std);
        layerW.push(row);
      }
      this.W.push(layerW);
      this.b.push(zeros(fanOut));
    }
  }

  get layerCount(): number {
    return this.W.length;
  }

  /** Activation applied by layer index l (0-based over weight layers). */
  activationAt(l: number): ActivationName | 'softmax' {
    const isOutput = l === this.W.length - 1;
    if (!isOutput) return this.config.hiddenActivation;
    return this.config.outputActivation === 'softmax' ? 'softmax' : this.config.outputActivation;
  }

  private applyActivation(l: number, z: number[]): number[] {
    const act = this.activationAt(l);
    if (act === 'softmax') return softmax(z);
    return z.map(ACTIVATIONS[act].f);
  }

  /**
   * Forward pass. With `training` true and dropout > 0, hidden activations are
   * masked and rescaled by 1/(1−p) (inverted dropout), so the expected value of
   * each activation is unchanged and inference needs no rescaling.
   */
  forward(x: number[], training = false): ForwardTrace {
    const layers: LayerTrace[] = [];
    let a = x;
    const p = this.config.dropout;
    for (let l = 0; l < this.W.length; l++) {
      const z = zeros(this.sizes[l + 1]);
      for (let j = 0; j < z.length; j++) {
        let sum = this.b[l][j];
        const row = this.W[l][j];
        for (let i = 0; i < a.length; i++) sum += row[i] * a[i];
        z[j] = sum;
      }
      let out = this.applyActivation(l, z);
      const isHidden = l < this.W.length - 1;
      let mask: number[] | undefined;
      if (training && isHidden && p > 0) {
        mask = out.map(() => (this.rng() < p ? 0 : 1 / (1 - p)));
        out = out.map((v, i) => v * mask![i]);
      }
      layers.push({ z, a: out, mask });
      a = out;
    }
    return { input: x, layers, output: a };
  }

  /**
   * Backward pass for a single example.
   *
   * δᴸ = ∂L/∂zᴸ is formed at the output and propagated backwards with
   * δˡ = (Wˡ⁺¹)ᵀ δˡ⁺¹ ⊙ f'(zˡ). Gradients follow directly:
   * ∂L/∂Wˡ = δˡ (aˡ⁻¹)ᵀ and ∂L/∂bˡ = δˡ.
   */
  backward(trace: ForwardTrace, target: number[]): Gradients {
    const L = this.W.length;
    const dW: number[][][] = [];
    const db: number[][] = [];
    const delta: number[][] = new Array(L);
    const dA: number[][] = new Array(L);

    const loss = LOSSES[this.config.loss];
    const outAct = this.config.outputActivation;
    const output = trace.output;
    const lastZ = trace.layers[L - 1].z;

    // Output-layer δ. Softmax+CCE and sigmoid+BCE both simplify to ŷ − y;
    // everything else goes through ∂L/∂ŷ · f'(z) explicitly.
    let deltaOut: number[];
    if (outAct === 'softmax' && this.config.loss === 'cce') {
      deltaOut = output.map((p, i) => p - target[i]);
    } else if (outAct === 'sigmoid' && this.config.loss === 'bce') {
      deltaOut = output.map((p, i) => (p - target[i]) / output.length);
    } else {
      const dY = loss.gradOutput(output, target);
      if (outAct === 'softmax') {
        // Full Jacobian product: δ_k = Σ_i dY_i · p_i (δ_ik − p_k)
        deltaOut = lastZ.map((_, k) => {
          let sum = 0;
          for (let i = 0; i < output.length; i++) {
            sum += dY[i] * output[i] * ((i === k ? 1 : 0) - output[k]);
          }
          return sum;
        });
      } else {
        const act = ACTIVATIONS[outAct];
        deltaOut = dY.map((g, i) => g * act.df(lastZ[i]));
      }
    }
    delta[L - 1] = deltaOut;

    for (let l = L - 1; l >= 0; l--) {
      const aPrev = l === 0 ? trace.input : trace.layers[l - 1].a;
      const d = delta[l];
      const layerDW = zeroMatrix(this.sizes[l + 1], this.sizes[l]);
      for (let j = 0; j < d.length; j++) {
        for (let i = 0; i < aPrev.length; i++) layerDW[j][i] = d[j] * aPrev[i];
      }
      dW[l] = layerDW;
      db[l] = d.slice();

      if (l > 0) {
        // ∂L/∂a for the previous layer's (post-dropout) activations.
        const prevSize = this.sizes[l];
        const gradA = zeros(prevSize);
        for (let i = 0; i < prevSize; i++) {
          let sum = 0;
          for (let j = 0; j < d.length; j++) sum += this.W[l][j][i] * d[j];
          gradA[i] = sum;
        }
        dA[l - 1] = gradA;
        const prevLayer = trace.layers[l - 1];
        const act = ACTIVATIONS[this.config.hiddenActivation];
        delta[l - 1] = gradA.map((g, i) => {
          const maskScale = prevLayer.mask ? prevLayer.mask[i] : 1;
          return g * maskScale * act.df(prevLayer.z[i]);
        });
      }
    }
    return { dW, db, delta, dA };
  }

  /** Loss for one example, excluding the regularisation term. */
  sampleLoss(predicted: number[], target: number[]): number {
    return LOSSES[this.config.loss].value(predicted, target);
  }

  /** (λ/2) Σ w² over all weight matrices. Biases are not regularised. */
  l2Penalty(): number {
    const lambda = this.config.l2;
    if (lambda === 0) return 0;
    let sum = 0;
    for (const layer of this.W) for (const row of layer) for (const w of row) sum += w * w;
    return (lambda / 2) * sum;
  }

  /** Mean data loss over a dataset (no dropout, no regularisation term). */
  evaluateLoss(samples: Sample[]): number {
    if (samples.length === 0) return 0;
    let total = 0;
    for (const s of samples) total += this.sampleLoss(this.forward(s.x).output, s.y);
    return total / samples.length;
  }

  /** Fraction of examples classified correctly. */
  evaluateAccuracy(samples: Sample[]): number {
    if (samples.length === 0) return 0;
    let correct = 0;
    for (const s of samples) {
      const out = this.forward(s.x).output;
      if (out.length === 1) {
        const predicted = out[0] >= 0.5 ? 1 : 0;
        if (predicted === (s.y[0] >= 0.5 ? 1 : 0)) correct++;
      } else {
        let argmax = 0;
        let argmaxTarget = 0;
        for (let i = 1; i < out.length; i++) if (out[i] > out[argmax]) argmax = i;
        for (let i = 1; i < s.y.length; i++) if (s.y[i] > s.y[argmaxTarget]) argmaxTarget = i;
        if (argmax === argmaxTarget) correct++;
      }
    }
    return correct / samples.length;
  }

  /**
   * One pass over `samples` in mini-batches. Returns the mean data loss seen
   * during the pass (measured before each update, which is the standard
   * "training loss for this epoch" reported by frameworks).
   */
  trainEpoch(samples: Sample[], learningRate: number, batchSize: number): number {
    if (samples.length === 0) return 0;
    const order = shuffleInPlace(
      samples.map((_, i) => i),
      this.rng,
    );
    const size = Math.max(1, Math.min(batchSize, samples.length));
    let totalLoss = 0;

    for (let start = 0; start < order.length; start += size) {
      const batch = order.slice(start, start + size);
      const accW = this.W.map((layer) => zeroMatrix(layer.length, layer[0].length));
      const accB = this.b.map((layer) => zeros(layer.length));

      for (const index of batch) {
        const sample = samples[index];
        const trace = this.forward(sample.x, true);
        totalLoss += this.sampleLoss(trace.output, sample.y);
        const grads = this.backward(trace, sample.y);
        for (let l = 0; l < this.W.length; l++) {
          for (let j = 0; j < accW[l].length; j++) {
            accB[l][j] += grads.db[l][j];
            for (let i = 0; i < accW[l][j].length; i++) accW[l][j][i] += grads.dW[l][j][i];
          }
        }
      }

      const scale = 1 / batch.length;
      const lambda = this.config.l2;
      for (let l = 0; l < this.W.length; l++) {
        for (let j = 0; j < this.W[l].length; j++) {
          this.b[l][j] -= learningRate * accB[l][j] * scale;
          for (let i = 0; i < this.W[l][j].length; i++) {
            const grad = accW[l][j][i] * scale + lambda * this.W[l][j][i];
            this.W[l][j][i] -= learningRate * grad;
          }
        }
      }
    }
    return totalLoss / samples.length;
  }

  /** Output for many inputs at once — used to sweep decision-boundary grids. */
  predictBatch(inputs: number[][]): number[][] {
    return inputs.map((x) => this.forward(x).output);
  }

  /** Total number of trainable parameters. */
  parameterCount(): number {
    let count = 0;
    for (let l = 0; l < this.W.length; l++) count += this.sizes[l] * this.sizes[l + 1] + this.sizes[l + 1];
    return count;
  }

  /** Regularisation strength can change between epochs without rebuilding. */
  setRegularisation(options: { l2?: number; dropout?: number }): void {
    if (options.l2 !== undefined) this.config.l2 = options.l2;
    if (options.dropout !== undefined) this.config.dropout = options.dropout;
  }

  snapshot(): { W: number[][][]; b: number[][] } {
    return {
      W: this.W.map((layer) => layer.map((row) => row.slice())),
      b: this.b.map((row) => row.slice()),
    };
  }

  restore(state: { W: number[][][]; b: number[][] }): void {
    this.W = state.W.map((layer) => layer.map((row) => row.slice()));
    this.b = state.b.map((row) => row.slice());
  }
}
