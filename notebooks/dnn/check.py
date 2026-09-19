"""
Correctness checks for the exercises.

Every check verifies a *property* of your answer rather than comparing it to a
stored solution: derivatives are checked against finite differences of your own
code, batched versions are checked against your own single-example version, and
so on. Passing a check means the thing is actually right, not that it matched
something hidden in this file.

Conventions used throughout the notebooks
-----------------------------------------
``X``       array of shape (m, n_in) — one row per example
``y``       array of shape (m,) holding 0.0 or 1.0
``params``  list of dicts, one per weighted layer: {'W': (n_in, n_out), 'b': (n_out,)}
``Z``       pre-activations, ``X @ W + b``, shape (m, n_out)
"""

from __future__ import annotations

import numpy as np


class CheckFailed(AssertionError):
    """Raised when an exercise answer is wrong. The message says how."""


class _Check:
    def __init__(self, name: str):
        self.name = name
        self.count = 0

    def that(self, condition, message: str, hint: str | None = None):
        self.count += 1
        if not condition:
            text = f'{self.name}: {message}'
            if hint:
                text += f'\n\n  hint: {hint}'
            raise CheckFailed(text)
        return self

    def close(self, note: str = ''):
        suffix = f'  ({note})' if note else ''
        print(f'✓ {self.name} — {self.count} checks passed{suffix}')


def _call(fn, *args, **kwargs):
    """Call a student function, turning the unfilled stub into a clear message."""
    try:
        return fn(*args, **kwargs)
    except NotImplementedError as exc:
        raise CheckFailed(
            f'{getattr(fn, "__name__", "the function")} is not implemented yet — '
            f'replace the `raise NotImplementedError` line with your code.'
        ) from exc


def _close(a, b, tol=1e-9):
    return np.allclose(np.asarray(a, dtype=float), np.asarray(b, dtype=float), rtol=tol, atol=tol)


def _relative_error(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    denom = np.maximum(1e-12, np.abs(a) + np.abs(b))
    return float(np.max(np.abs(a - b) / denom))


# ---------------------------------------------------------------------------
# 00 — reading the mathematics
# ---------------------------------------------------------------------------

def weighted_sum(fn):
    c = _Check('weighted_sum')
    rng = np.random.default_rng(0)
    for n in (1, 2, 5, 40):
        w, x = rng.normal(size=n), rng.normal(size=n)
        got = _call(fn, w, x)
        c.that(np.isscalar(got) or np.ndim(got) == 0,
               f'returned an array for length-{n} inputs; it should be a single number',
               'sum() over the products collapses them to one value.')
        c.that(_close(got, float(np.dot(w, x))),
               f'gave {float(got):.6f} for a length-{n} case, expected {float(np.dot(w, x)):.6f}',
               'multiply matching entries, then add all the products together.')
    c.that(_close(_call(fn, np.array([0.5, -1.2, 2.0]), np.array([4.0, 3.0, 2.0])), 2.4),
           'the worked example from section 00 should give 2.4')
    c.close()


def matvec(fn):
    c = _Check('matvec')
    rng = np.random.default_rng(1)
    for rows, cols in ((3, 2), (1, 4), (5, 5)):
        W, x = rng.normal(size=(rows, cols)), rng.normal(size=cols)
        got = np.asarray(_call(fn, W, x), dtype=float)
        c.that(got.shape == (rows,),
               f'returned shape {got.shape} for a {rows}x{cols} matrix; expected ({rows},)',
               'one output number per row of the matrix.')
        c.that(_close(got, W @ x), f'values are wrong for a {rows}x{cols} matrix',
               'output[j] is the weighted sum of row j with x.')
    W = np.array([[0.5, -1.0], [2.0, 0.5], [-0.5, 1.5]])
    c.that(_close(_call(fn, W, np.array([4.0, 2.0])), [0.0, 9.0, 1.0]),
           'the worked example from section 00 should give [0, 9, 1]')
    c.close()


def numerical_derivative(fn):
    c = _Check('numerical_derivative')
    cases = [
        (lambda t: t ** 2, lambda t: 2 * t, 1.0, 'x squared'),
        (lambda t: t ** 3, lambda t: 3 * t ** 2, -2.0, 'x cubed'),
        (np.sin, np.cos, 0.7, 'sine'),
        (np.exp, np.exp, 0.3, 'exp'),
    ]
    for f, df, x, label in cases:
        got = float(_call(fn, f, x, 1e-5))
        c.that(abs(got - df(x)) < 1e-6,
               f'for {label} at x={x} you gave {got:.8f}, expected about {df(x):.8f}',
               'use the centred form (f(x+h) - f(x-h)) / (2h) — it is far more accurate '
               'than the one-sided version.')
    # x**2 is a special case: the centred difference is exact for it, so use a
    # cubic, where the error is proportional to h**2.
    coarse = abs(float(_call(fn, lambda t: t ** 3, 1.0, 1e-1)) - 3.0)
    fine = abs(float(_call(fn, lambda t: t ** 3, 1.0, 1e-4)) - 3.0)
    c.that(fine < coarse / 100,
           f'a smaller h should be much more accurate: h=0.1 gave an error of {coarse:.2e} '
           f'and h=0.0001 gave {fine:.2e}',
           'the centred difference error shrinks like h squared; a one-sided difference '
           'only shrinks like h.')
    c.close('centred difference confirmed')


def chain_rule(fn):
    c = _Check('chain_rule')
    rng = np.random.default_rng(2)
    for _ in range(6):
        x, w, b = rng.normal(), rng.normal(), rng.normal()
        y = float(rng.integers(0, 2))
        got = float(_call(fn, x, w, b, y))

        def loss(xv):
            a = 1.0 / (1.0 + np.exp(-(w * xv + b)))
            return (a - y) ** 2

        h = 1e-6
        numeric = (loss(x + h) - loss(x - h)) / (2 * h)
        c.that(abs(got - numeric) < 1e-5,
               f'at x={x:.3f}, w={w:.3f}, b={b:.3f}, y={y} you gave {got:.8f} '
               f'but nudging x gives {numeric:.8f}',
               'multiply the three rates: dz/dx = w, da/dz = a(1-a), dL/da = 2(a-y).')
    c.close('matches a direct measurement')


def shapes(answers: dict):
    c = _Check('shapes')
    expected = {
        'X @ W1': (32, 8),
        'Z1 + b1': (32, 8),
        'A1 @ W2': (32, 1),
        'W1.T': (8, 2),
    }
    for key, want in expected.items():
        c.that(key in answers, f'no answer given for {key!r}')
        got = tuple(answers[key])
        c.that(got == want, f'for {key} you said {got}, but it is {want}',
               'a (m, n) array times an (n, k) array gives (m, k); adding a length-k '
               'vector broadcasts across the rows.')
    c.close()


# ---------------------------------------------------------------------------
# 01 — network structure
# ---------------------------------------------------------------------------

def layer_shapes(fn):
    c = _Check('layer_shapes')
    for sizes, want in (
        ([2, 3, 1], [((2, 3), (3,)), ((3, 1), (1,))]),
        ([3, 5, 4, 2], [((3, 5), (5,)), ((5, 4), (4,)), ((4, 2), (2,))]),
        ([2, 1], [((2, 1), (1,))]),
    ):
        got = _call(fn, list(sizes))
        got = [(tuple(w), tuple(b)) for w, b in got]
        c.that(got == want, f'for sizes {sizes} you gave {got}, expected {want}',
               'there is one weighted layer per adjacent pair of sizes; W is '
               '(inputs, outputs) and b has one entry per output.')
    c.close()


def count_parameters(fn):
    c = _Check('count_parameters')
    for sizes, want in (([2, 3, 1], 13), ([2, 4, 3, 1], 31), ([3, 5, 4, 2], 54), ([2, 1], 3)):
        got = _call(fn, list(sizes))
        c.that(int(got) == want, f'for sizes {sizes} you gave {got}, expected {want}',
               'each layer contributes n_in * n_out weights plus n_out biases.')
    c.close()


def init_params(fn):
    c = _Check('init_params')
    sizes = [2, 6, 4, 1]
    params = _call(fn, sizes, 0)
    c.that(len(params) == 3, f'expected 3 layers for sizes {sizes}, got {len(params)}')
    for i, layer in enumerate(params):
        c.that('W' in layer and 'b' in layer, f"layer {i} should be a dict with keys 'W' and 'b'")
        c.that(layer['W'].shape == (sizes[i], sizes[i + 1]),
               f"layer {i} W has shape {layer['W'].shape}, expected {(sizes[i], sizes[i + 1])}")
        c.that(layer['b'].shape == (sizes[i + 1],),
               f"layer {i} b has shape {layer['b'].shape}, expected {(sizes[i + 1],)}")
        c.that(np.allclose(layer['b'], 0.0), f'layer {i} biases should start at zero')
        c.that(not np.allclose(layer['W'], 0.0), f'layer {i} weights must not all be zero',
               'identical weights make every unit in the layer compute the same thing for ever.')
        rows = layer['W'].T
        if len(rows) > 1:
            c.that(not np.allclose(rows[0], rows[1]),
                   f'layer {i}: two units got identical weights — the symmetry is not broken')
    again = _call(fn, sizes, 0)
    c.that(all(np.allclose(a['W'], b['W']) for a, b in zip(params, again)),
           'the same seed should produce the same weights')
    other = _call(fn, sizes, 1)
    c.that(not np.allclose(params[0]['W'], other[0]['W']),
           'a different seed should produce different weights')
    c.close()


# ---------------------------------------------------------------------------
# 02 — neurons
# ---------------------------------------------------------------------------

def neuron(fn):
    c = _Check('neuron')
    relu = lambda t: np.maximum(0.0, t)
    got = float(_call(fn, np.array([1.0, 0.5]), np.array([0.8, -0.3]), 0.2, relu))
    c.that(_close(got, 0.85, 1e-6),
           f'the worked example should give 0.85, you gave {got:.4f}',
           'z = (0.8)(1.0) + (-0.3)(0.5) + 0.2 = 0.85, and ReLU leaves a positive z alone.')
    got = float(_call(fn, np.array([1.0, 0.5]), np.array([-0.8, -0.3]), 0.2, relu))
    c.that(_close(got, 0.0, 1e-6),
           f'a negative z should give 0 through ReLU, you gave {got:.4f}')
    sigmoid = lambda t: 1.0 / (1.0 + np.exp(-t))
    got = float(_call(fn, np.array([1.0, 0.5]), np.array([0.8, -0.3]), 0.2, sigmoid))
    c.that(_close(got, 1 / (1 + np.exp(-0.85)), 1e-6),
           'the activation argument must actually be used')
    rng = np.random.default_rng(3)
    for n in (1, 3, 7):
        x, w, b = rng.normal(size=n), rng.normal(size=n), rng.normal()
        c.that(_close(_call(fn, x, w, b, relu), relu(np.dot(w, x) + b)),
               f'wrong for {n} inputs')
    c.close()


def signed_distance(fn):
    c = _Check('signed_distance')
    rng = np.random.default_rng(4)
    for _ in range(6):
        w, b, x = rng.normal(size=2), rng.normal(), rng.normal(size=2)
        want = (np.dot(w, x) + b) / np.linalg.norm(w)
        got = float(_call(fn, x, w, b))
        c.that(_close(got, want, 1e-9),
               f'you gave {got:.6f}, expected {want:.6f}',
               'divide z by the length of w; np.linalg.norm(w) gives that length.')
    w, b = np.array([0.8, -0.3]), 0.2
    c.that(_close(_call(fn, np.array([1.0, 0.5]), w, b), 0.85 / np.linalg.norm(w), 1e-9),
           'the worked example should give 0.85 / ||w|| = 0.995')
    c.close('sign and magnitude both correct')


def xor_is_impossible(best_accuracy: float):
    c = _Check('xor_is_impossible')
    c.that(best_accuracy is not None, 'report the best accuracy your search found')
    c.that(best_accuracy < 1.0,
           f'your search reported {best_accuracy:.3f}; a single neuron cannot reach 1.0 on XOR',
           'if you got 1.0, check that you are testing all four corners with the right labels.')
    c.that(best_accuracy >= 0.75,
           f'{best_accuracy:.3f} is lower than expected — three of the four corners '
           'are always achievable')
    c.close(f'best possible is {best_accuracy:.2f}, never 1.0')


# ---------------------------------------------------------------------------
# 03 — weights, biases, initialisation
# ---------------------------------------------------------------------------

def init_scale(he, glorot):
    c = _Check('init_scale')
    for fan_in, fan_out in ((64, 32), (128, 128), (16, 256)):
        W = _call(he, fan_in, fan_out, 0)
        c.that(W.shape == (fan_in, fan_out), f'he_init gave shape {W.shape}, expected {(fan_in, fan_out)}')
        want = np.sqrt(2.0 / fan_in)
        got = float(W.std())
        c.that(abs(got - want) / want < 0.12,
               f'he_init({fan_in}, {fan_out}) has std {got:.4f}, expected about {want:.4f}',
               'He initialisation draws from a normal with variance 2 / fan_in.')
        c.that(abs(float(W.mean())) < 0.05 * want + 1e-3, 'he_init should be centred on zero')

        G = _call(glorot, fan_in, fan_out, 0)
        want_g = np.sqrt(2.0 / (fan_in + fan_out))
        got_g = float(G.std())
        c.that(abs(got_g - want_g) / want_g < 0.12,
               f'glorot_init({fan_in}, {fan_out}) has std {got_g:.4f}, expected about {want_g:.4f}',
               'Glorot uses variance 2 / (fan_in + fan_out).')
    c.close('empirical spread matches the formulas')


def layer_stds(fn):
    c = _Check('layer_stds')
    depth = 10
    flat = np.asarray(_call(fn, depth, 48, float(np.sqrt(2)), 'relu'))
    c.that(flat.shape == (depth,), f'expected {depth} numbers, got shape {flat.shape}')
    ratio = float(flat[-1] / max(flat[0], 1e-12))
    c.that(0.4 < ratio < 2.5,
           f'with ReLU at gain sqrt(2) the scale should stay roughly constant, '
           f'but layer {depth} is {ratio:.3g} times layer 1',
           'that is exactly what He initialisation is designed to achieve.')
    small = np.asarray(_call(fn, depth, 48, 0.7, 'relu'))
    c.that(small[-1] < flat[-1] / 10,
           f'a gain of 0.7 should make the activations vanish, but layer {depth} '
           f'is {small[-1]:.3g} versus {flat[-1]:.3g} at the correct gain')
    big = np.asarray(_call(fn, depth, 48, 2.2, 'relu'))
    c.that(big[-1] > flat[-1] * 5,
           'a gain of 2.2 should make the activations explode')
    c.close('flat at sqrt(2), vanishing below it, exploding above it')


def symmetry(before, after):
    c = _Check('symmetry')
    before, after = np.asarray(before), np.asarray(after)
    c.that(np.allclose(before[0], before[1]), 'the two units should start identical')
    c.that(np.allclose(after[0], after[1]),
           'after a gradient step the two units differ, but identical units receive '
           'identical gradients and must stay identical',
           'check that both units really started with the same weights and the same bias.')
    c.close('identical units stay identical — which is why we randomise')


# ---------------------------------------------------------------------------
# 04 — activation functions
# ---------------------------------------------------------------------------

def activations(relu, leaky_relu, sigmoid, tanh):
    c = _Check('activations')
    z = np.array([-2.0, -0.5, 0.0, 0.5, 2.0])

    r = np.asarray(_call(relu, z))
    c.that(r.shape == z.shape, f'relu changed the shape: {r.shape} vs {z.shape}',
           'use a vectorised numpy operation so it works on whole arrays.')
    c.that(_close(r, [0, 0, 0, 0.5, 2.0]), f'relu gave {r}, expected [0, 0, 0, 0.5, 2]')

    lr = np.asarray(_call(leaky_relu, z, 0.1))
    c.that(_close(lr, [-0.2, -0.05, 0.0, 0.5, 2.0]),
           f'leaky_relu gave {lr}, expected [-0.2, -0.05, 0, 0.5, 2]',
           'negative inputs are multiplied by alpha instead of being zeroed.')

    s = np.asarray(_call(sigmoid, z))
    c.that(_close(s, 1 / (1 + np.exp(-z)), 1e-9), 'sigmoid values are wrong')
    c.that(_close(_call(sigmoid, np.array([0.0])), [0.5]), 'sigmoid(0) should be 0.5')
    big = np.asarray(_call(sigmoid, np.array([-800.0, 800.0])))
    c.that(np.all(np.isfinite(big)),
           'sigmoid overflowed on large inputs and produced nan or inf',
           'np.exp(800) is infinite. Handle positive and negative z separately, or '
           'use np.where with a clipped exponent.')

    t = np.asarray(_call(tanh, z))
    c.that(_close(t, np.tanh(z), 1e-9), 'tanh values are wrong')
    c.close('values, shapes and overflow all handled')


def activation_gradients(pairs):
    """``pairs`` maps a name to ``(f, df)``."""
    c = _Check('activation_gradients')
    zs = np.array([-3.0, -1.4, -0.3, 0.4, 1.1, 2.7])
    for name, (f, df) in pairs.items():
        analytic = np.asarray(_call(df, zs), dtype=float)
        c.that(analytic.shape == zs.shape, f'{name}: derivative changed the shape')
        h = 1e-6
        numeric = (np.asarray(f(zs + h), float) - np.asarray(f(zs - h), float)) / (2 * h)
        err = _relative_error(analytic, numeric)
        c.that(err < 1e-5,
               f'{name}: analytic derivative {analytic} does not match the measured '
               f'{numeric} (relative error {err:.2e})',
               'differentiate by hand, then express the result in numpy. For sigmoid and '
               'tanh the derivative can reuse the function value itself.')
    c.close(f'{len(pairs)} derivatives verified against finite differences')


def softmax(fn):
    c = _Check('softmax')
    z = np.array([2.0, 1.0, 0.1])
    p = np.asarray(_call(fn, z), dtype=float)
    c.that(p.shape == z.shape, f'shape changed: {p.shape} vs {z.shape}')
    c.that(_close(p.sum(), 1.0, 1e-9), f'the outputs sum to {p.sum():.6f}, they must sum to 1')
    c.that(_close(p, [0.65900114, 0.24243297, 0.09856589], 1e-6),
           f'values are wrong: got {p}')
    shifted = np.asarray(_call(fn, z + 100.0), dtype=float)
    c.that(np.all(np.isfinite(shifted)),
           'adding 100 to every logit produced nan or inf',
           'subtract the maximum before calling np.exp — it does not change the answer '
           'but keeps every exponent at or below zero.')
    c.that(_close(shifted, p, 1e-9),
           'adding a constant to every logit changed the answer; softmax is shift invariant')
    huge = np.asarray(_call(fn, np.array([1000.0, 999.0, 998.0])), dtype=float)
    c.that(np.all(np.isfinite(huge)) and _close(huge.sum(), 1.0, 1e-9),
           'logits around 1000 still overflow')
    batched = np.asarray(_call(fn, np.array([[2.0, 1.0, 0.1], [0.0, 0.0, 0.0]])), dtype=float)
    if batched.shape == (2, 3):
        c.that(_close(batched.sum(axis=1), [1.0, 1.0], 1e-9),
               'when given a batch, each row must sum to 1 (use axis=-1, keepdims=True)')
        c.that(_close(batched[1], [1 / 3, 1 / 3, 1 / 3], 1e-9), 'equal logits should give equal probabilities')
    c.close('stable, shift invariant, sums to one')


def log_softmax(fn, softmax_fn):
    c = _Check('log_softmax')
    z = np.array([2.0, 1.0, 0.1])
    got = np.asarray(_call(fn, z), dtype=float)
    c.that(_close(got, np.log(np.asarray(softmax_fn(z), float)), 1e-9),
           'log_softmax should equal log(softmax(z)) where that is computable')
    extreme = np.asarray(_call(fn, np.array([0.0, -1000.0])), dtype=float)
    c.that(np.all(np.isfinite(extreme)),
           'log_softmax returned -inf where taking the log of a softmax would underflow',
           'use z - max - log(sum(exp(z - max))) and never form the probability first.')
    c.close('finite even where log(softmax(z)) underflows')


# ---------------------------------------------------------------------------
# 05 — forward propagation
# ---------------------------------------------------------------------------

def forward(fn, init_fn):
    c = _Check('forward')
    sizes = [2, 4, 3, 1]
    params = init_fn(sizes, 0)
    rng = np.random.default_rng(5)
    X = rng.normal(size=(7, 2))
    out, cache = _call(fn, X, params)
    out = np.asarray(out, dtype=float)
    c.that(out.shape == (7, 1), f'output shape {out.shape}, expected (7, 1)')
    c.that(np.all((out >= 0) & (out <= 1)),
           'the output should be a probability in [0, 1] — apply the sigmoid at the last layer')
    c.that(len(cache) == len(params),
           f'the cache has {len(cache)} entries but there are {len(params)} layers; '
           'backpropagation needs one per layer')
    for i, entry in enumerate(cache):
        for key in ('a_prev', 'z', 'a'):
            c.that(key in entry, f"cache entry {i} is missing '{key}'")
        c.that(entry['z'].shape == (7, sizes[i + 1]),
               f"cache[{i}]['z'] has shape {entry['z'].shape}, expected {(7, sizes[i + 1])}")
        c.that(np.allclose(entry['a_prev'], X if i == 0 else cache[i - 1]['a']),
               f"cache[{i}]['a_prev'] should be the activations entering this layer")

    single = np.stack([np.asarray(_call(fn, X[j:j + 1], params)[0], float)[0] for j in range(7)])
    c.that(_close(single, out, 1e-9),
           'running the batch gives a different answer from running the rows one at a time',
           'each row must be processed independently — check you are not mixing the batch '
           'dimension into a sum.')

    W, b = params[0]['W'], params[0]['b']
    c.that(_close(cache[0]['z'], X @ W + b, 1e-9),
           'the first layer pre-activations are not X @ W + b')
    c.close('shapes, cache, and batch-equals-loop all correct')


# ---------------------------------------------------------------------------
# 06 — loss functions
# ---------------------------------------------------------------------------

def losses(mse, bce):
    c = _Check('losses')
    c.that(_close(_call(mse, np.array([0.8]), np.array([1.0])), 0.04, 1e-9),
           'mse(0.8, 1) should be 0.04')
    c.that(_close(_call(mse, np.array([0.5, 0.2]), np.array([1.0, 0.0])), 0.145, 1e-9),
           'mse should average over the examples: (0.25 + 0.04) / 2 = 0.145')
    c.that(_close(_call(bce, np.array([0.8]), np.array([1.0])), -np.log(0.8), 1e-9),
           f'bce(0.8, 1) should be {-np.log(0.8):.4f}')
    c.that(_close(_call(bce, np.array([0.3]), np.array([0.0])), -np.log(0.7), 1e-9),
           f'bce(0.3, 0) should be {-np.log(0.7):.4f}')
    extreme = float(_call(bce, np.array([0.0]), np.array([1.0])))
    c.that(np.isfinite(extreme),
           'bce(0, 1) returned inf or nan',
           'clip the prediction into [eps, 1-eps] before taking the log.')
    c.that(extreme > 10, 'a confidently wrong prediction should still cost a lot')
    c.close('values correct and log(0) handled')


def loss_gradients(pairs):
    """``pairs`` maps a name to ``(loss_fn, grad_fn)`` operating on (y_hat, y)."""
    c = _Check('loss_gradients')
    rng = np.random.default_rng(6)
    for name, (loss, grad) in pairs.items():
        y_hat = rng.uniform(0.05, 0.95, size=5)
        y = rng.integers(0, 2, size=5).astype(float)
        analytic = np.asarray(_call(grad, y_hat, y), dtype=float)
        c.that(analytic.shape == y_hat.shape,
               f'{name}: gradient shape {analytic.shape}, expected {y_hat.shape} — '
               'one derivative per prediction')
        h = 1e-6
        numeric = np.empty_like(y_hat)
        for i in range(len(y_hat)):
            up, down = y_hat.copy(), y_hat.copy()
            up[i] += h
            down[i] -= h
            numeric[i] = (loss(up, y) - loss(down, y)) / (2 * h)
        err = _relative_error(analytic, numeric)
        c.that(err < 1e-4,
               f'{name}: your gradient {analytic} does not match the measured {numeric} '
               f'(relative error {err:.2e})',
               'remember the 1/m from the average if your loss divides by the number of examples.')
    c.close(f'{len(pairs)} loss gradients verified against finite differences')


def sigmoid_bce_delta(fn):
    c = _Check('sigmoid_bce_delta')
    rng = np.random.default_rng(7)
    z = rng.normal(size=6)
    y = rng.integers(0, 2, size=6).astype(float)
    y_hat = 1 / (1 + np.exp(-z))
    got = np.asarray(_call(fn, z, y), dtype=float)
    c.that(_close(got, y_hat - y, 1e-9),
           f'expected y_hat - y = {y_hat - y}, got {got}',
           'the 1/(y_hat(1-y_hat)) from the loss derivative cancels against '
           "sigma'(z) = y_hat(1-y_hat).")
    h = 1e-6
    for i in range(len(z)):
        up, down = z.copy(), z.copy()
        up[i] += h
        down[i] -= h

        def bce_of(zz):
            p = np.clip(1 / (1 + np.exp(-zz)), 1e-12, 1 - 1e-12)
            return float(np.sum(-(y * np.log(p) + (1 - y) * np.log(1 - p))))

        numeric = (bce_of(up) - bce_of(down)) / (2 * h)
        c.that(abs(numeric - got[i]) < 1e-5,
               f'entry {i}: measured {numeric:.8f} but you gave {got[i]:.8f}')
    c.close('the cancellation is exact')


# ---------------------------------------------------------------------------
# 07 — gradient descent
# ---------------------------------------------------------------------------

def gradient_descent(fn):
    c = _Check('gradient_descent')
    f = lambda t: t ** 2
    df = lambda t: 2 * t
    path = np.asarray(_call(fn, df, 1.8, 0.1, 5), dtype=float)
    c.that(path.shape == (6,),
           f'expected 6 values for 5 steps from a starting point, got shape {path.shape}',
           'include the starting point, then one value per step.')
    c.that(_close(path[0], 1.8), 'the first value should be the starting point')
    for i in range(5):
        want = path[i] - 0.1 * df(path[i])
        c.that(_close(path[i + 1], want, 1e-12),
               f'step {i + 1}: from {path[i]:.6f} the rule gives {want:.6f}, you gave {path[i + 1]:.6f}',
               'theta <- theta - learning_rate * gradient(theta)')
    converged = np.asarray(_call(fn, df, 1.8, 0.1, 80), dtype=float)
    c.that(abs(converged[-1]) < 1e-3, 'after 80 steps at lr=0.1 you should be very near 0')
    diverged = np.asarray(_call(fn, df, 1.8, 1.05, 40), dtype=float)
    c.that(abs(diverged[-1]) > 1.8,
           'at lr=1.05 the iterates should grow, not shrink — that is above the stability limit')
    c.that(f(converged[-1]) < f(converged[0]), 'the loss should be lower at the end')
    c.close('update rule, convergence and divergence all correct')


def stability_limit(reported: float, curvature: float = 2.0):
    c = _Check('stability_limit')
    want = 2.0 / curvature
    c.that(reported is not None, 'report the largest learning rate that still converged')
    c.that(abs(reported - want) < 0.08 * want + 0.02,
           f'you found {reported:.4f}, but the theory says 2 / {curvature} = {want:.4f}',
           'search a fine grid of learning rates and find where the behaviour flips.')
    c.close(f'measured {reported:.3f} vs predicted {want:.3f}')


def momentum(fn):
    c = _Check('momentum')
    df = lambda t: 2 * t
    beta, lr = 0.8, 0.05
    path = np.asarray(_call(fn, df, 2.0, lr, 6, beta), dtype=float)
    c.that(path.shape == (7,), f'expected 7 values for 6 steps, got {path.shape}')
    v = 0.0
    theta = 2.0
    for i in range(6):
        v = beta * v + df(theta)
        theta = theta - lr * v
        c.that(_close(path[i + 1], theta, 1e-12),
               f'step {i + 1}: the rule v <- beta*v + grad, theta <- theta - lr*v gives '
               f'{theta:.8f}, you gave {path[i + 1]:.8f}')
    plain = np.asarray(_call(fn, df, 2.0, lr, 6, 0.0), dtype=float)
    reference = np.asarray(_call(fn, df, 2.0, lr, 6, 0.0), dtype=float)
    c.that(_close(plain, reference), 'the function should be deterministic')
    c.that(abs(plain[-1]) > abs(path[-1]),
           'with beta=0.8 you should get closer to the minimum than with beta=0 in the same '
           'number of steps')
    c.close('velocity recursion verified step by step')


def condition_number(fn):
    c = _Check('condition_number')
    X = np.array([[-1.2], [-0.6], [-0.1], [0.4], [0.9], [1.4]])
    for scale, expect_bigger in ((1.0, False), (0.2, True)):
        kappa = float(_call(fn, X * scale))
        c.that(kappa >= 1.0, f'a condition number cannot be below 1, you gave {kappa:.3f}')
    k1 = float(_call(fn, X))
    k2 = float(_call(fn, X * 0.2))
    c.that(k1 < 2.5, f'at scale 1 the surface is nearly round, so kappa should be small; got {k1:.2f}')
    c.that(k2 > 10,
           f'shrinking the feature by 5x should stretch the surface a lot; got kappa={k2:.2f}',
           'build the Hessian (2/m) * A.T @ A where A has the feature column and a column of '
           'ones, then take the ratio of its eigenvalues.')
    c.close(f'kappa goes from {k1:.2f} to {k2:.1f} when the feature is rescaled')


# ---------------------------------------------------------------------------
# 08 — backpropagation
# ---------------------------------------------------------------------------

def backward(forward_fn, backward_fn, init_fn, loss_fn):
    """
    The central check of the whole course: compare every gradient your backward
    pass produces against a finite-difference measurement of your own forward
    pass. If these agree, your derivation is right.
    """
    c = _Check('backward')
    rng = np.random.default_rng(8)
    sizes = [3, 5, 4, 1]
    params = init_fn(sizes, 11)
    X = rng.normal(size=(6, 3))
    y = rng.integers(0, 2, size=6).astype(float)

    out, cache = forward_fn(X, params)
    grads = _call(backward_fn, cache, y, params)
    c.that(len(grads) == len(params),
           f'expected one gradient entry per layer ({len(params)}), got {len(grads)}')

    h = 1e-5
    worst = 0.0
    worst_where = ''
    for l, (layer, grad) in enumerate(zip(params, grads)):
        for key in ('dW', 'db'):
            c.that(key in grad, f"layer {l} gradient is missing '{key}'")
        c.that(grad['dW'].shape == layer['W'].shape,
               f"layer {l}: dW has shape {grad['dW'].shape}, but W is {layer['W'].shape} — "
               'the gradient must have the same shape as the thing it differentiates')
        c.that(grad['db'].shape == layer['b'].shape,
               f"layer {l}: db has shape {grad['db'].shape}, but b is {layer['b'].shape}")

        for name, array, darray in (('W', layer['W'], grad['dW']), ('b', layer['b'], grad['db'])):
            it = np.ndindex(array.shape)
            for idx in it:
                original = array[idx]
                array[idx] = original + h
                up = loss_fn(forward_fn(X, params)[0].ravel(), y)
                array[idx] = original - h
                down = loss_fn(forward_fn(X, params)[0].ravel(), y)
                array[idx] = original
                numeric = (up - down) / (2 * h)
                err = abs(numeric - darray[idx]) / max(1e-9, abs(numeric) + abs(darray[idx]))
                if err > worst:
                    worst, worst_where = err, f'layer {l}, {name}{idx}'

    c.that(worst < 1e-5,
           f'the largest disagreement is {worst:.3e} at {worst_where}. Your analytic gradient '
           'and a direct measurement of your own forward pass do not match.',
           'work backwards one layer at a time. Check the output delta first, then the '
           'recursion, then the per-layer dW = a_prev.T @ delta / m.')
    c.close(f'largest relative error {worst:.2e} across {sum(p["W"].size + p["b"].size for p in params)} parameters')


def output_delta(fn):
    c = _Check('output_delta')
    rng = np.random.default_rng(9)
    y_hat = rng.uniform(0.05, 0.95, size=(5, 1))
    y = rng.integers(0, 2, size=5).astype(float)
    got = np.asarray(_call(fn, y_hat, y), dtype=float)
    c.that(got.shape == y_hat.shape,
           f'expected shape {y_hat.shape}, got {got.shape}')
    c.that(_close(got, (y_hat - y.reshape(-1, 1)) / len(y), 1e-9),
           'for a sigmoid output with binary cross-entropy averaged over m examples, '
           'the delta is (y_hat - y) / m',
           'the 1/m comes from the loss being a mean rather than a sum.')
    c.close()


def gradient_flow(norms_relu, norms_sigmoid):
    c = _Check('gradient_flow')
    relu = np.asarray(norms_relu, dtype=float)
    sig = np.asarray(norms_sigmoid, dtype=float)
    c.that(relu.shape == sig.shape and relu.ndim == 1,
           'expected one gradient norm per layer for each activation')
    c.that(len(relu) >= 6, 'use a network deep enough for the effect to show — at least 6 layers')
    decay_relu = relu[0] / max(relu[-1], 1e-30)
    decay_sig = sig[0] / max(sig[-1], 1e-30)
    c.that(decay_sig < decay_relu,
           f'sigmoid should lose far more gradient over the depth than ReLU, but you measured '
           f'a factor of {decay_sig:.2e} for sigmoid and {decay_relu:.2e} for ReLU',
           'the norms are ordered from the first layer to the last; sigmoid multiplies by at '
           'most 0.25 per layer.')
    c.that(decay_sig < 0.05,
           f'the first sigmoid layer should receive a tiny fraction of the last layer\'s '
           f'gradient; you measured {decay_sig:.3g}')
    c.close(f'sigmoid loses {1 / max(decay_sig, 1e-30):.0f}x more gradient than ReLU over the depth')


# ---------------------------------------------------------------------------
# 09 — training
# ---------------------------------------------------------------------------

def batches(fn):
    c = _Check('batches')
    rng = np.random.default_rng(10)
    X = rng.normal(size=(10, 2))
    y = rng.integers(0, 2, size=10).astype(float)
    out = list(_call(fn, X, y, 4, rng))
    c.that(len(out) == 3, f'10 examples in batches of 4 should give 3 batches, you gave {len(out)}',
           'the last batch is smaller than the rest; do not drop it.')
    sizes = [len(bx) for bx, _ in out]
    c.that(sizes == [4, 4, 2], f'batch sizes {sizes}, expected [4, 4, 2]')
    seen = np.concatenate([bx for bx, _ in out])
    c.that(len(seen) == 10, 'every example should appear exactly once per epoch')
    matched = sorted(tuple(np.round(r, 9)) for r in seen) == sorted(tuple(np.round(r, 9)) for r in X)
    c.that(matched, 'the batches should contain the original rows, just reordered')
    for bx, by in out:
        for row, label in zip(bx, by):
            idx = int(np.argmin(np.abs(X - row).sum(axis=1)))
            c.that(_close(y[idx], label), 'an example got separated from its label during shuffling',
                   'shuffle an index array and use it to index both X and y.')
    orders = [np.concatenate([bx for bx, _ in _call(fn, X, y, 4, np.random.default_rng(s))])
              for s in (1, 2)]
    c.that(not np.allclose(orders[0], orders[1]), 'the order should differ between epochs')
    c.close('shuffled, paired, and the short final batch kept')


def training(history, min_accuracy=0.9):
    c = _Check('training')
    c.that('train_loss' in history, "your history should record 'train_loss'")
    losses_ = np.asarray(history['train_loss'], dtype=float)
    c.that(len(losses_) >= 10, 'train for at least 10 epochs so the curve says something')
    c.that(np.all(np.isfinite(losses_)), 'the loss became nan or inf — the learning rate is too high')
    c.that(losses_[-1] < losses_[0],
           f'the loss went from {losses_[0]:.4f} to {losses_[-1]:.4f}; training should reduce it')
    if 'train_acc' in history:
        acc = float(np.asarray(history['train_acc'], dtype=float)[-1])
        c.that(acc >= min_accuracy,
               f'final accuracy {acc:.3f} is below {min_accuracy:.2f}. Try more epochs, a '
               'different learning rate, or more hidden units.')
    c.close(f'loss {losses_[0]:.4f} -> {losses_[-1]:.4f}')


# ---------------------------------------------------------------------------
# 10 — overfitting and regularisation
# ---------------------------------------------------------------------------

def l2_gradient(fn):
    c = _Check('l2_gradient')
    rng = np.random.default_rng(11)
    W = rng.normal(size=(3, 4))
    base = rng.normal(size=(3, 4))
    got = np.asarray(_call(fn, base, W, 0.05), dtype=float)
    c.that(got.shape == W.shape, f'shape {got.shape}, expected {W.shape}')
    c.that(_close(got, base + 0.05 * W, 1e-9),
           'with the penalty (lambda/2) * sum(W**2), the extra gradient term is lambda * W',
           'add lambda * W to the data gradient; do not scale the data gradient itself.')
    c.that(_close(_call(fn, base, W, 0.0), base, 1e-9),
           'at lambda = 0 the gradient must be unchanged')
    c.close()


def dropout(fn):
    c = _Check('dropout')
    rng = np.random.default_rng(12)
    A = np.ones((2000, 20))
    out, mask = _call(fn, A, 0.5, rng, True)
    out = np.asarray(out, dtype=float)
    c.that(out.shape == A.shape, f'shape {out.shape}, expected {A.shape}')
    zero_fraction = float(np.mean(out == 0))
    c.that(0.42 < zero_fraction < 0.58,
           f'{zero_fraction:.3f} of the activations were dropped, expected about 0.5')
    kept = out[out != 0]
    c.that(_close(kept.mean(), 2.0, 1e-6),
           f'surviving activations average {kept.mean():.4f}; with p=0.5 they should be '
           'scaled by 1/(1-p) = 2',
           'this is inverted dropout: divide the survivors by (1 - p) so the expected value '
           'entering the next layer is unchanged.')
    c.that(abs(float(out.mean()) - 1.0) < 0.05,
           f'the overall mean is {out.mean():.4f}; the rescaling should keep it near the '
           'original 1.0')
    eval_out, _ = _call(fn, A, 0.5, rng, False)
    c.that(_close(eval_out, A, 1e-12),
           'at evaluation time nothing should be dropped and nothing rescaled')
    c.that(mask is not None and np.asarray(mask).shape == A.shape,
           'return the mask as well — the backward pass needs the same one')
    c.close('inverted dropout, off at evaluation time')


def early_stopping(fn):
    c = _Check('early_stopping')
    losses_ = [1.0, 0.8, 0.6, 0.55, 0.61, 0.66, 0.70, 0.75]
    best = _call(fn, losses_)
    c.that(int(best) == 3,
           f'the lowest validation loss is 0.55 at epoch 3, you returned {best}',
           'epochs are counted from 0, and ties should keep the earliest.')
    c.that(int(_call(fn, [0.5, 0.5, 0.9])) == 0, 'with a tie, keep the earlier epoch')
    c.that(int(_call(fn, [0.9, 0.8, 0.7])) == 2, 'if the loss never rises, the best epoch is the last')
    c.close()


def regularisation_effect(no_reg, with_reg):
    """
    Compare an unregularised run with a regularised one.

    The honest claim is not that L2 lowers the *best* validation loss — taking
    the minimum over epochs is already a form of early stopping, which is itself
    a regulariser. What L2 reliably does is shrink the weights and close the gap
    between the final training and validation losses.
    """
    c = _Check('regularisation_effect')
    for key in ('final_train_loss', 'final_val_loss', 'final_weight_norm'):
        for name, run in (('no_reg', no_reg), ('with_reg', with_reg)):
            c.that(key in run, f"{name} is missing '{key}'")

    c.that(with_reg['final_weight_norm'] < no_reg['final_weight_norm'],
           f"L2 should shrink the weights, but the norm went from "
           f"{no_reg['final_weight_norm']:.3f} to {with_reg['final_weight_norm']:.3f}")

    gap_without = no_reg['final_val_loss'] - no_reg['final_train_loss']
    gap_with = with_reg['final_val_loss'] - with_reg['final_train_loss']
    c.that(gap_without > 0.2,
           f'the unregularised run should overfit visibly, but its train/validation '
           f'gap is only {gap_without:.4f}. Train it longer, or add more noise to the data.')
    c.that(gap_with < gap_without / 2,
           f'L2 should close most of the gap: {gap_without:.4f} without it, '
           f'{gap_with:.4f} with it',
           'if the gap barely moved, lambda is too small for this much noise.')
    c.that(with_reg['final_val_loss'] < no_reg['final_val_loss'],
           f"the regularised run should end with a lower validation loss "
           f"({no_reg['final_val_loss']:.4f} -> {with_reg['final_val_loss']:.4f})")
    c.close(f"gap {gap_without:.2f} -> {gap_with:.2f}, "
            f"||W|| {no_reg['final_weight_norm']:.1f} -> {with_reg['final_weight_norm']:.1f}")


# ---------------------------------------------------------------------------
# 11 — depth
# ---------------------------------------------------------------------------

def linear_regions(shallow_fn, deep_fn):
    c = _Check('linear_regions')
    for n, want in ((2, 4), (3, 7), (8, 37), (16, 137)):
        got = int(_call(shallow_fn, n, 2))
        c.that(got == want,
               f'one hidden layer of {n} units in 2 inputs cuts the plane into {want} regions, '
               f'you gave {got}',
               'sum the binomial coefficients C(n, 0) + C(n, 1) + ... + C(n, d).')
    c.that(int(_call(shallow_fn, 5, 1)) == 6, 'in one input dimension, n cuts give n + 1 pieces')
    one = int(_call(deep_fn, 8, 1, 2))
    c.that(one == int(_call(shallow_fn, 8, 2)),
           'with a single hidden layer the deep bound should reduce to the shallow count')
    growth = [int(_call(deep_fn, 8, layers, 2)) for layers in (1, 2, 3, 4)]
    c.that(all(b > a for a, b in zip(growth, growth[1:])), 'the bound should grow with depth')
    ratios = [b / a for a, b in zip(growth, growth[1:])]
    c.that(max(ratios) / min(ratios) < 1.01,
           f'each extra layer should multiply the count by the same factor, but the ratios '
           f'are {[round(r, 2) for r in ratios]}',
           'the factor is floor(n / d) ** d, applied once per extra layer.')
    c.close(f'8 units: {growth[0]} regions at depth 1, {growth[-1]:,} at depth 4')


def depth_comparison(results: dict):
    c = _Check('depth_comparison')
    c.that(len(results) >= 3, 'compare at least three depths')
    c.that(0 in results or '0' in results or 'none' in results or 1 in results,
           'include a network with no hidden layer or a single one as the baseline')
    losses_ = [v for v in results.values()]
    c.that(all(np.isfinite(v) for v in losses_), 'one of the runs produced a non-finite loss')
    keys = sorted(results, key=lambda k: int(k) if str(k).isdigit() else 0)
    shallowest, deepest = results[keys[0]], results[keys[-1]]
    c.that(deepest < shallowest,
           f'on a dataset that needs curvature, the deeper network should reach a lower loss; '
           f'you got {shallowest:.4f} for the shallowest and {deepest:.4f} for the deepest',
           'train them all for the same number of epochs with the same learning rate.')
    c.close(f'{shallowest:.4f} -> {deepest:.4f} from shallow to deep')


# ---------------------------------------------------------------------------
# 12 — capstone
# ---------------------------------------------------------------------------

def network_class(cls):
    c = _Check('NeuralNetwork')
    rng = np.random.default_rng(13)
    net = cls([2, 8, 8, 1], activation='tanh', seed=3)
    c.that(hasattr(net, 'fit') and hasattr(net, 'predict'),
           'the class needs fit() and predict() methods')

    X = rng.normal(size=(5, 2))
    p = np.asarray(net.predict(X), dtype=float).ravel()
    c.that(p.shape == (5,), f'predict returned shape {p.shape}, expected (5,)')
    c.that(np.all((p >= 0) & (p <= 1)), 'predict should return probabilities in [0, 1]')

    # Gradients must match finite differences before we trust the training loop.
    y = rng.integers(0, 2, size=5).astype(float)
    grads = net.gradients(X, y)
    h = 1e-5
    worst = 0.0
    for layer, grad in zip(net.params, grads):
        for name, arr, darr in (('W', layer['W'], grad['dW']), ('b', layer['b'], grad['db'])):
            for idx in np.ndindex(arr.shape):
                original = arr[idx]
                arr[idx] = original + h
                up = net.loss(X, y)
                arr[idx] = original - h
                down = net.loss(X, y)
                arr[idx] = original
                numeric = (up - down) / (2 * h)
                worst = max(worst, abs(numeric - darr[idx]) /
                            max(1e-9, abs(numeric) + abs(darr[idx])))
    c.that(worst < 1e-5, f'gradients disagree with finite differences by {worst:.2e}')

    from .data import make_dataset
    Xc, yc = make_dataset('circles', n=200, noise=0.06, seed=1)
    net2 = cls([2, 8, 8, 1], activation='tanh', seed=5)
    history = net2.fit(Xc, yc, learning_rate=0.3, batch_size=16, epochs=250)
    acc = float(np.mean((np.asarray(net2.predict(Xc)).ravel() >= 0.5) == (yc >= 0.5)))
    c.that(acc > 0.92,
           f'after 250 epochs on circles the accuracy is {acc:.3f}; it should exceed 0.92',
           'check the learning rate and that the loss in your history is falling.')
    c.that('train_loss' in history and len(history['train_loss']) == 250,
           'fit() should return a history with one train_loss entry per epoch')
    c.close(f'gradients exact ({worst:.1e}) and {acc:.1%} accuracy on circles')
