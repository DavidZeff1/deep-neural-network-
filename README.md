# Deep Neural Networks — an interactive course

A visualisation-first course on deep neural networks. Every figure is a live
network: weights are initialised, forward passes are computed, gradients are
derived by backpropagation and training runs in the browser. There are no
pre-recorded animations and no placeholder controls.

![Twelve sections, from network structure to a full playground](docs/overview.png)

## Contents

| # | Section | What it does |
|---|---------|--------------|
| 01 | Network structure | Reshape an architecture and watch the layer shapes and parameter count follow |
| 02 | Neurons | One neuron, two inputs; the weighted sum, the bias and the line `z = 0` in the input plane |
| 03 | Weights & biases | Edit any entry of `W` or `b` from the diagram or the matrix and see the forward pass change |
| 04 | Activation functions | ReLU, sigmoid, tanh, leaky ReLU with their derivatives; softmax over adjustable logits |
| 05 | Forward propagation | Step through `z⁽¹⁾ → a⁽¹⁾ → z⁽²⁾ → ŷ` with the real numbers and the matrix form side by side |
| 06 | Loss functions | MSE, binary cross-entropy and categorical cross-entropy as functions of the prediction |
| 07 | Gradient descent | `θ ← θ − η ∇L(θ)` on four 1-D landscapes and on a 2-D least-squares surface |
| 08 | Backpropagation | Step the backward pass, expand the chain rule for any weight, verify it against finite differences |
| 09 | Training | Mini-batch training on five datasets with a live decision boundary and metric curves |
| 10 | Overfitting & regularisation | Train/validation split, L2, dropout and early stopping |
| 11 | Why depth matters | Four depths trained in lockstep, hidden-unit responses, linear-region counts |
| 12 | Playground | Everything combined, with a probe point and a per-unit / per-weight inspector |

## Running it

```bash
npm install
npm run dev          # development server
npm run build        # production build into dist/
npm run preview      # serve the production build on :4173
```

The built site is static and has no network dependencies at runtime.

## Tests

```bash
npm test             # numerical correctness of the network engine
npm run test:browser # every interactive control, against a running preview server
```

`npm test` is the important one. It gradient-checks backpropagation against
central finite differences for every combination of hidden activation
(`relu`, `leakyRelu`, `sigmoid`, `tanh`), output head (`sigmoid`, `softmax`,
`linear`) and loss (`mse`, `bce`, `cce`), for networks with zero, one, two and
four hidden layers, and requires a relative error below `1e-5`. It also checks
the softmax Jacobian, the loss derivatives, the update rule, dataset balance and
reproducibility, and that a network with no hidden layer fails on XOR while
solving a linearly separable set.

`npm run test:browser` needs a preview server on port 4173 and Playwright's
Chromium. It drives 73 checks across all twelve sections — sliders, buttons,
selects, canvas clicks, training runs, navigation, theme switching and mobile
layout — and asserts on the values the page displays (for example that
`BCE(0.8, y=1)` reads `0.2231`, that one descent step equals `θ − η∇L`, and that
the on-screen chain-rule product equals the value backpropagation produced).
Set `CHROMIUM_PATH` to use a system Chromium instead of a downloaded one.

## Layout

```
src/
  lib/                 the mathematics, with no UI dependencies
    activations.ts     f(z) and f'(z) for each activation; softmax and its Jacobian
    losses.ts          MSE, BCE, CCE with their derivatives
    network.ts         the MLP: forward, backward, mini-batch training, L2, dropout
    optimisation.ts    1-D loss landscapes and least-squares gradient descent
    datasets.ts        linear, XOR, circles, moons, spiral
    contour.ts         marching squares, used for decision boundaries and contours
    rng.ts             seeded RNG so every figure is reproducible
    format.ts          number formatting
  components/
    ui/                controls, panels, stat tiles, KaTeX wrappers
    viz/               Plot frame, network diagram, decision boundary, contours, metric charts
    layout/            section shell
  hooks/               mutable network, trainer, ensemble trainer, scroll spy, theme
  sections/            one file per section, plus the registry that orders them
tests/
  network.test.ts      numerical tests
  browser/             end-to-end interaction tests
```

`src/lib` is plain TypeScript with no React imports, so the engine can be read,
tested and reused on its own.

## Implementation notes

**Index convention.** `W[l][j][i]` is the weight from unit `i` of layer `l` to
unit `j` of layer `l+1`. Layer 0 is the input and holds no parameters.

**Backpropagation.** `δ⁽ᴸ⁾ = ∂L/∂z⁽ᴸ⁾` is formed at the output and propagated with
`δ⁽ˡ⁾ = (W⁽ˡ⁺¹⁾)ᵀ δ⁽ˡ⁺¹⁾ ⊙ f′(z⁽ˡ⁾)`; parameter gradients are
`∂L/∂W⁽ˡ⁾ = δ⁽ˡ⁾ (a⁽ˡ⁻¹⁾)ᵀ` and `∂L/∂b⁽ˡ⁾ = δ⁽ˡ⁾`. Sigmoid+BCE and softmax+CCE
use the simplification `δ⁽ᴸ⁾ = ŷ − y`; every other pairing goes through the
explicit `∂L/∂ŷ · f′(z)` path, including the full softmax Jacobian.

**Initialisation.** He (`σ² = 2/fan_in`) for ReLU-family units, Glorot
(`σ² = 2/(fan_in + fan_out)`) for saturating units.

**Regularisation.** L2 is `(λ/2)Σw²`, contributing `λW` to the gradient; biases
are not regularised. Dropout is inverted dropout: hidden activations are masked
and divided by `1 − p` during training only. Both can be changed mid-run without
resetting the weights.

**Training loop.** Mini-batch gradient descent, one or more epochs per animation
frame, with the learning rate and batch size read from refs so they take effect
immediately. Reported losses are recomputed on the full set after each epoch with
dropout disabled.

**Decision boundaries.** The output is swept over a grid, drawn as a raster, and
the `p = 0.5` level set is extracted with marching squares so the boundary is a
line rather than a colour change.

## Libraries

`react` for components, `katex` for mathematical notation, `d3-scale` and
`d3-shape` for scales and path generation. The diagrams, plots, contours and
decision boundaries are written directly in SVG and canvas — a charting library
would not draw a network with per-weight interaction or a probability field with
an extracted level set.
