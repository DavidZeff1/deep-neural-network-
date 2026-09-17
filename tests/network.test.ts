import { strict as assert } from 'node:assert';
import test from 'node:test';

import { ACTIVATIONS, softmax, softmaxJacobian } from '../src/lib/activations.ts';
import { LOSSES } from '../src/lib/losses.ts';
import { MLP } from '../src/lib/network.ts';
import type { NetworkConfig, OutputActivation } from '../src/lib/network.ts';
import type { ActivationName } from '../src/lib/activations.ts';
import type { LossName } from '../src/lib/losses.ts';
import { DATASET_NAMES, generateDataset, splitDataset } from '../src/lib/datasets.ts';
import {
  gradientDescent1D,
  gradientDescent2D,
  regressionCurvature,
  regressionLoss,
  regressionOptimum,
  LOSS_LANDSCAPES,
} from '../src/lib/optimisation.ts';

// --- activations -----------------------------------------------------------

test('activation derivatives match finite differences', () => {
  const h = 1e-6;
  for (const act of Object.values(ACTIVATIONS)) {
    for (const z of [-2.7, -1.1, -0.3, 0.4, 1.3, 3.1]) {
      const numeric = (act.f(z + h) - act.f(z - h)) / (2 * h);
      assert.ok(
        Math.abs(numeric - act.df(z)) < 1e-5,
        `${act.name} at z=${z}: analytic ${act.df(z)} vs numeric ${numeric}`,
      );
    }
  }
});

test('softmax sums to one and is shift invariant', () => {
  const logits = [2.0, 1.0, 0.1];
  const p = softmax(logits);
  assert.ok(Math.abs(p.reduce((a, b) => a + b, 0) - 1) < 1e-12);
  const shifted = softmax(logits.map((z) => z + 100));
  for (let i = 0; i < p.length; i++) assert.ok(Math.abs(p[i] - shifted[i]) < 1e-12);
  // Known values for [2, 1, 0.1]
  assert.ok(Math.abs(p[0] - 0.6590011388859679) < 1e-9);
  assert.ok(Math.abs(p[1] - 0.2424329707047139) < 1e-9);
  assert.ok(Math.abs(p[2] - 0.09856589040931818) < 1e-9);
});

test('softmax jacobian matches finite differences', () => {
  const logits = [0.7, -0.4, 1.6];
  const jac = softmaxJacobian(softmax(logits));
  const h = 1e-6;
  for (let i = 0; i < 3; i++) {
    for (let j = 0; j < 3; j++) {
      const up = logits.slice();
      const down = logits.slice();
      up[j] += h;
      down[j] -= h;
      const numeric = (softmax(up)[i] - softmax(down)[i]) / (2 * h);
      assert.ok(Math.abs(numeric - jac[i][j]) < 1e-6, `jac[${i}][${j}]`);
    }
  }
});

// --- losses ----------------------------------------------------------------

test('loss values match hand computation', () => {
  assert.ok(Math.abs(LOSSES.mse.value([0.8], [1]) - 0.04) < 1e-12);
  assert.ok(Math.abs(LOSSES.mse.value([0.5, 0.2], [1, 0]) - (0.25 + 0.04) / 2) < 1e-12);
  assert.ok(Math.abs(LOSSES.bce.value([0.8], [1]) - -Math.log(0.8)) < 1e-12);
  assert.ok(Math.abs(LOSSES.bce.value([0.3], [0]) - -Math.log(0.7)) < 1e-12);
  assert.ok(Math.abs(LOSSES.cce.value([0.7, 0.2, 0.1], [1, 0, 0]) - -Math.log(0.7)) < 1e-12);
});

test('loss gradients match finite differences', () => {
  const h = 1e-6;
  const cases: Array<[LossName, number[], number[]]> = [
    ['mse', [0.3, 0.9], [1, 0]],
    ['bce', [0.62], [1]],
    ['bce', [0.21], [0]],
    ['cce', [0.5, 0.3, 0.2], [0, 1, 0]],
  ];
  for (const [name, predicted, target] of cases) {
    const loss = LOSSES[name];
    const analytic = loss.gradOutput(predicted, target);
    for (let i = 0; i < predicted.length; i++) {
      const up = predicted.slice();
      const down = predicted.slice();
      up[i] += h;
      down[i] -= h;
      const numeric = (loss.value(up, target) - loss.value(down, target)) / (2 * h);
      assert.ok(
        Math.abs(numeric - analytic[i]) < 1e-4,
        `${name}[${i}]: analytic ${analytic[i]} vs numeric ${numeric}`,
      );
    }
  }
});

// --- backpropagation -------------------------------------------------------

function gradientCheck(config: NetworkConfig, x: number[], y: number[]): number {
  const net = new MLP(config);
  const grads = net.backward(net.forward(x), y);
  const h = 1e-5;
  let worst = 0;
  for (let l = 0; l < net.W.length; l++) {
    for (let j = 0; j < net.W[l].length; j++) {
      for (let i = 0; i < net.W[l][j].length; i++) {
        const original = net.W[l][j][i];
        net.W[l][j][i] = original + h;
        const up = net.sampleLoss(net.forward(x).output, y);
        net.W[l][j][i] = original - h;
        const down = net.sampleLoss(net.forward(x).output, y);
        net.W[l][j][i] = original;
        const numeric = (up - down) / (2 * h);
        const analytic = grads.dW[l][j][i];
        const denom = Math.max(1e-8, Math.abs(numeric) + Math.abs(analytic));
        worst = Math.max(worst, Math.abs(numeric - analytic) / denom);
      }
      const originalB = net.b[l][j];
      net.b[l][j] = originalB + h;
      const upB = net.sampleLoss(net.forward(x).output, y);
      net.b[l][j] = originalB - h;
      const downB = net.sampleLoss(net.forward(x).output, y);
      net.b[l][j] = originalB;
      const numericB = (upB - downB) / (2 * h);
      const analyticB = grads.db[l][j];
      const denomB = Math.max(1e-8, Math.abs(numericB) + Math.abs(analyticB));
      worst = Math.max(worst, Math.abs(numericB - analyticB) / denomB);
    }
  }
  return worst;
}

test('backpropagation matches numerical gradients across configurations', () => {
  const hidden: ActivationName[] = ['relu', 'leakyRelu', 'sigmoid', 'tanh'];
  const heads: Array<[OutputActivation, LossName, number]> = [
    ['sigmoid', 'bce', 1],
    ['sigmoid', 'mse', 1],
    ['linear', 'mse', 2],
    ['softmax', 'cce', 3],
    ['softmax', 'mse', 3],
  ];
  for (const h of hidden) {
    for (const [outputActivation, loss, outputSize] of heads) {
      const config: NetworkConfig = {
        inputSize: 3,
        hiddenUnits: [5, 4],
        outputSize,
        hiddenActivation: h,
        outputActivation,
        loss,
        seed: 21,
      };
      const x = [0.4, -0.9, 1.2];
      const y = outputSize === 1 ? [1] : outputSize === 2 ? [0.3, -0.7] : [0, 1, 0];
      const worst = gradientCheck(config, x, y);
      assert.ok(worst < 1e-5, `${h}/${outputActivation}/${loss}: relative error ${worst}`);
    }
  }
});

test('backpropagation is correct for a network with no hidden layers', () => {
  const worst = gradientCheck(
    {
      inputSize: 2,
      hiddenUnits: [],
      outputSize: 1,
      hiddenActivation: 'relu',
      outputActivation: 'sigmoid',
      loss: 'bce',
      seed: 3,
    },
    [0.6, -0.2],
    [0],
  );
  assert.ok(worst < 1e-6, `relative error ${worst}`);
});

test('backpropagation is correct for a deep network', () => {
  const worst = gradientCheck(
    {
      inputSize: 2,
      hiddenUnits: [6, 6, 6, 6],
      outputSize: 1,
      hiddenActivation: 'tanh',
      outputActivation: 'sigmoid',
      loss: 'bce',
      seed: 99,
    },
    [-0.3, 0.8],
    [1],
  );
  assert.ok(worst < 1e-5, `relative error ${worst}`);
});

// --- training --------------------------------------------------------------

test('training drives the loss down and solves XOR', () => {
  const data = generateDataset('xor', { count: 200, noise: 0.06, seed: 5 });
  const net = new MLP({
    inputSize: 2,
    hiddenUnits: [8, 8],
    outputSize: 1,
    hiddenActivation: 'tanh',
    outputActivation: 'sigmoid',
    loss: 'bce',
    seed: 11,
  });
  const before = net.evaluateLoss(data);
  for (let epoch = 0; epoch < 260; epoch++) net.trainEpoch(data, 0.35, 16);
  const after = net.evaluateLoss(data);
  assert.ok(after < before, `loss did not decrease: ${before} -> ${after}`);
  assert.ok(net.evaluateAccuracy(data) > 0.95, `accuracy ${net.evaluateAccuracy(data)}`);
});

test('a network with no hidden layer cannot solve XOR but solves the linear set', () => {
  const build = () =>
    new MLP({
      inputSize: 2,
      hiddenUnits: [],
      outputSize: 1,
      hiddenActivation: 'relu',
      outputActivation: 'sigmoid',
      loss: 'bce',
      seed: 4,
    });
  const xor = generateDataset('xor', { count: 200, noise: 0.05, seed: 5 });
  const linear = generateDataset('linear', { count: 200, noise: 0.05, seed: 5 });

  const a = build();
  for (let e = 0; e < 400; e++) a.trainEpoch(xor, 0.5, 16);
  assert.ok(a.evaluateAccuracy(xor) < 0.75, `XOR accuracy unexpectedly high: ${a.evaluateAccuracy(xor)}`);

  const b = build();
  for (let e = 0; e < 400; e++) b.trainEpoch(linear, 0.5, 16);
  assert.ok(b.evaluateAccuracy(linear) > 0.95, `linear accuracy ${b.evaluateAccuracy(linear)}`);
});

test('L2 shrinks weights relative to an unregularised run', () => {
  const data = generateDataset('moons', { count: 120, noise: 0.2, seed: 8 });
  const norm = (net: MLP) => {
    let s = 0;
    for (const layer of net.W) for (const row of layer) for (const w of row) s += w * w;
    return s;
  };
  const base = new MLP({
    inputSize: 2,
    hiddenUnits: [12, 12],
    outputSize: 1,
    hiddenActivation: 'relu',
    outputActivation: 'sigmoid',
    loss: 'bce',
    seed: 31,
  });
  const regularised = new MLP({ ...base.config, l2: 0.05 });
  for (let e = 0; e < 150; e++) {
    base.trainEpoch(data, 0.2, 16);
    regularised.trainEpoch(data, 0.2, 16);
  }
  assert.ok(norm(regularised) < norm(base), `${norm(regularised)} vs ${norm(base)}`);
});

test('dropout masks are applied only to hidden layers during training', () => {
  const net = new MLP({
    inputSize: 2,
    hiddenUnits: [40],
    outputSize: 1,
    hiddenActivation: 'relu',
    outputActivation: 'sigmoid',
    loss: 'bce',
    seed: 2,
    dropout: 0.5,
  });
  const trace = net.forward([0.5, -0.5], true);
  assert.ok(trace.layers[0].mask, 'hidden layer should carry a mask');
  assert.equal(trace.layers[1].mask, undefined, 'output layer must not be dropped');
  const dropped = trace.layers[0].mask!.filter((m) => m === 0).length;
  assert.ok(dropped > 5 && dropped < 35, `unexpected drop count ${dropped}`);
  assert.equal(net.forward([0.5, -0.5], false).layers[0].mask, undefined);
});

test('parameter count matches the architecture', () => {
  const net = new MLP({
    inputSize: 2,
    hiddenUnits: [4, 3],
    outputSize: 1,
    hiddenActivation: 'relu',
    outputActivation: 'sigmoid',
    loss: 'bce',
    seed: 1,
  });
  // (2*4+4) + (4*3+3) + (3*1+1) = 12 + 15 + 4 = 31
  assert.equal(net.parameterCount(), 31);
});

// --- datasets --------------------------------------------------------------

test('datasets are in range, balanced and reproducible', () => {
  for (const name of DATASET_NAMES) {
    const a = generateDataset(name, { count: 160, noise: 0.05, seed: 12 });
    const b = generateDataset(name, { count: 160, noise: 0.05, seed: 12 });
    assert.deepEqual(a, b, `${name} is not reproducible`);
    assert.equal(a.length, 160);
    const positives = a.filter((s) => s.y[0] === 1).length;
    assert.ok(positives > 60 && positives < 100, `${name} unbalanced: ${positives}`);
    for (const s of a) {
      assert.ok(Math.abs(s.x[0]) < 1.6 && Math.abs(s.x[1]) < 1.6, `${name} out of range`);
    }
  }
});

test('split is disjoint and covers the data', () => {
  const data = generateDataset('circles', { count: 100, noise: 0.05, seed: 3 });
  const { train, validation } = splitDataset(data, 0.7);
  assert.equal(train.length + validation.length, 100);
  assert.equal(train.length, 70);
  for (const v of validation) assert.ok(!train.includes(v));
});

// --- gradient descent ------------------------------------------------------

test('gradient descent converges on a quadratic and diverges past the stability bound', () => {
  const quadratic = LOSS_LANDSCAPES.quadratic;
  // L(θ) = θ², stable while η < 1/a where L'' = 2a; here L'' = 2 so η < 1.
  const converging = gradientDescent1D(quadratic, 1.8, 0.1, 60);
  const last = converging[converging.length - 1];
  assert.ok(Math.abs(last.theta - quadratic.minimum) < 1e-3, `theta ${last.theta}`);
  assert.ok(last.loss < converging[0].loss);

  const diverging = gradientDescent1D(quadratic, 1.8, 1.05, 40);
  const end = diverging[diverging.length - 1];
  assert.ok(Math.abs(end.theta) > 1.8, `expected divergence, got ${end.theta}`);
});

test('gradient descent step follows theta <- theta - eta * grad', () => {
  const landscape = LOSS_LANDSCAPES.quadratic;
  const steps = gradientDescent1D(landscape, 2, 0.25, 3);
  for (let i = 1; i < steps.length; i++) {
    const expected = steps[i - 1].theta - 0.25 * steps[i - 1].gradient;
    assert.ok(Math.abs(steps[i].theta - expected) < 1e-12);
  }
});

test('landscape gradients match finite differences', () => {
  const h = 1e-6;
  for (const landscape of Object.values(LOSS_LANDSCAPES)) {
    for (const theta of [-2, -0.6, 0.3, 1.4, 2.5]) {
      const numeric = (landscape.loss(theta + h) - landscape.loss(theta - h)) / (2 * h);
      assert.ok(
        Math.abs(numeric - landscape.gradient(theta)) < 1e-4,
        `${landscape.label} at ${theta}: ${landscape.gradient(theta)} vs ${numeric}`,
      );
    }
  }
});

// --- momentum --------------------------------------------------------------

test('momentum reduces to plain gradient descent at beta = 0', () => {
  const plain = gradientDescent1D(LOSS_LANDSCAPES.quadratic, 1.8, 0.1, 10);
  const zeroMomentum = gradientDescent1D(LOSS_LANDSCAPES.quadratic, 1.8, 0.1, 10, 0);
  for (let i = 0; i < plain.length; i++) {
    assert.ok(Math.abs(plain[i].theta - zeroMomentum[i].theta) < 1e-12);
    assert.ok(Math.abs(plain[i].velocity - plain[i].gradient) < 1e-12);
  }
});

test('momentum follows v <- beta*v + g and theta <- theta - eta*v', () => {
  const beta = 0.8;
  const eta = 0.05;
  const steps = gradientDescent1D(LOSS_LANDSCAPES.quadratic, 2, eta, 5, beta);
  let velocity = 0;
  for (let i = 0; i < steps.length; i++) {
    velocity = beta * velocity + steps[i].gradient;
    assert.ok(Math.abs(steps[i].velocity - velocity) < 1e-12, `velocity at ${i}`);
    if (i + 1 < steps.length) {
      const expected = steps[i].theta - eta * velocity;
      assert.ok(Math.abs(steps[i + 1].theta - expected) < 1e-12, `theta at ${i + 1}`);
    }
  }
});

test('feature scaling changes the condition number, not the fit', () => {
  // At scale 1 the two curvature directions are nearly equal; shrinking the
  // feature stretches the surface and raises the condition number sharply.
  const balanced = regressionCurvature(1);
  const stretched = regressionCurvature(0.2);
  assert.ok(balanced.conditionNumber < 2, `kappa(1) = ${balanced.conditionNumber}`);
  assert.ok(stretched.conditionNumber > 10, `kappa(0.2) = ${stretched.conditionNumber}`);

  // The fitted line is the same function of the original x either way.
  const a = regressionOptimum(1);
  const b = regressionOptimum(0.2);
  assert.ok(Math.abs(a.w - b.w * 0.2) < 1e-9, `${a.w} vs ${b.w * 0.2}`);
  assert.ok(Math.abs(a.b - b.b) < 1e-9);
  assert.ok(Math.abs(regressionLoss(a, 1) - regressionLoss(b, 0.2)) < 1e-12);
});

test('momentum converges faster at a conservative learning rate on an ill-conditioned surface', () => {
  const scale = 0.2; // condition number ≈ 32
  const start = { w: 0, b: 0 };
  const optimum = regressionOptimum(scale);
  const distance = (s: { w: number; b: number }) => Math.hypot(s.w - optimum.w, s.b - optimum.b);
  const eta = regressionCurvature(scale).maxStableRate * 0.5;
  const plain = gradientDescent2D(start, eta, 120, 0, scale);
  const withMomentum = gradientDescent2D(start, eta, 120, 0.8, scale);
  const plainDistance = distance(plain[plain.length - 1]);
  const momentumDistance = distance(withMomentum[withMomentum.length - 1]);
  assert.ok(
    momentumDistance < plainDistance / 100,
    `momentum ${momentumDistance} vs plain ${plainDistance}`,
  );
});

test('too much momentum overshoots even below the plain stability limit', () => {
  const scale = 0.2;
  const start = { w: 0, b: 0 };
  const optimum = regressionOptimum(scale);
  const distance = (s: { w: number; b: number }) => Math.hypot(s.w - optimum.w, s.b - optimum.b);
  const eta = regressionCurvature(scale).maxStableRate * 0.5;
  const tuned = gradientDescent2D(start, eta, 120, 0.8, scale);
  const excessive = gradientDescent2D(start, eta, 120, 0.97, scale);
  assert.ok(
    distance(excessive[excessive.length - 1]) > distance(tuned[tuned.length - 1]),
    'beta = 0.97 should be worse than beta = 0.8 at this learning rate',
  );
});

test('the stability bound 2/lambda_max separates convergence from divergence', () => {
  for (const scale of [0.2, 1, 2.5]) {
    const { maxStableRate, eigenvalues, conditionNumber } = regressionCurvature(scale);
    assert.ok(eigenvalues[0] >= eigenvalues[1]);
    assert.ok(conditionNumber >= 1);
    const start = { w: 0, b: 0 };
    const stable = gradientDescent2D(start, maxStableRate * 0.9, 300, 0, scale);
    const unstable = gradientDescent2D(start, maxStableRate * 1.1, 80, 0, scale);
    const stableEnd = stable[stable.length - 1];
    const unstableEnd = unstable[unstable.length - 1];
    assert.ok(stableEnd.loss < stable[0].loss, `scale ${scale}: stable run did not descend`);
    assert.ok(
      unstableEnd.loss > stable[0].loss,
      `scale ${scale}: eta above the bound should not converge (${unstableEnd.loss})`,
    );
  }
});
