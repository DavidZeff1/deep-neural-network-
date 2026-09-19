# %% [markdown]
# # 03 — Weights, biases, and why the starting values matter
#
# Companion to section 03 of the site.
#
# Two things can go wrong before a single training step has happened.
#
# 1. **Symmetry.** If two units in a layer start identical, they stay identical
#    for ever. You will demonstrate this rather than take it on trust.
# 2. **Scale.** The forward pass multiplies by `W` at every layer. If that
#    multiplication shrinks the numbers by a factor of 0.7, then after 10 layers
#    they are 0.7¹⁰ ≈ 0.028 of their original size. You will measure this and
#    then derive the scale that prevents it.

# %%
import subprocess
import sys
from pathlib import Path

# Find the `dnn` helper package. It sits next to these notebooks, so this works
# from the repo root, from notebooks/, or from notebooks/solutions/. On a hosted
# runtime such as Colab, where only this one file was fetched, clone the repo.
REPO = 'https://github.com/DavidZeff1/deep-neural-network-.git'
root = next((c for c in (Path.cwd(), Path.cwd() / 'notebooks', Path.cwd().parent)
             if (c / 'dnn' / '__init__.py').exists()), None)
if root is None:
    subprocess.run(['git', 'clone', '--depth', '1', REPO, 'dnn-course'], check=True)
    root = Path('dnn-course') / 'notebooks'
sys.path.insert(0, str(root))

import numpy as np
import matplotlib.pyplot as plt

from dnn import check

np.set_printoptions(precision=4, suppress=True)

# %% [markdown]
# ## 1. Identical units stay identical
#
# The claim: two units in the same layer that start with the same weights and
# the same bias receive the same gradient, so a gradient step leaves them still
# equal.
#
# To check it we need gradients, and backpropagation is not until notebook 08.
# That is fine — notebook 00 built a perfectly good gradient tool already. It is
# slow, but it needs no theory at all.

# %%
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def tiny_network_loss(W1, b1, W2, b2, X, y):
    """A 2 -> 2 -> 1 network scored by mean squared error."""
    A1 = np.tanh(X @ W1 + b1)
    out = sigmoid(A1 @ W2 + b2).ravel()
    return float(np.mean((out - y) ** 2))


def numeric_gradient(array, loss_fn, h=1e-6):
    """Finite-difference gradient of loss_fn with respect to every entry of array."""
    grad = np.zeros_like(array)
    for idx in np.ndindex(array.shape):
        original = array[idx]
        array[idx] = original + h
        up = loss_fn()
        array[idx] = original - h
        down = loss_fn()
        array[idx] = original
        grad[idx] = (up - down) / (2 * h)
    return grad


rng = np.random.default_rng(0)
X = rng.normal(size=(20, 2))
y = (X[:, 0] + X[:, 1] > 0).astype(float)

# Both hidden units start with exactly the same incoming weights.
W1 = np.tile(np.array([[0.4], [-0.7]]), (1, 2))   # shape (2, 2), identical columns
b1 = np.zeros(2)
W2 = np.array([[0.5], [0.5]])
b2 = np.zeros(1)

print('the two hidden units start with weights')
print(W1.T)

# %%
before = W1.T.copy()

grad_W1 = numeric_gradient(W1, lambda: tiny_network_loss(W1, b1, W2, b2, X, y))
print('the gradients they receive:')
print(grad_W1.T)

W1 = W1 - 0.5 * grad_W1
after = W1.T.copy()

print()
print('after one gradient step:')
print(after)

check.symmetry(before, after)

# %% [markdown]
# The two rows are still equal to the last digit. Nothing in the training loop
# can separate them, because every quantity that feeds their gradient is the
# same for both. A layer of 100 identical units has the expressive power of one
# unit, permanently.
#
# The fix is not subtle: draw the weights at random. Any continuous distribution
# separates them with probability 1. Biases can start at zero, because the
# weights have already broken the tie — and that is exactly what
# `init_params` did in notebook 01.

# %% [markdown]
# ## 2. Where the initialisation scale comes from
#
# Take one unit's pre-activation, a sum over `n_in` products:
#
# $$z_j = \sum_{i=1}^{n_{\text{in}}} w_{ji} a_i$$
#
# Three facts about variance (the average squared distance from the mean) are
# all that is needed:
#
# * Scaling multiplies the spread by the square: $\operatorname{Var}(cX) = c^2\operatorname{Var}(X)$
# * Independent quantities add their variances
# * For a zero-mean quantity, variance and mean-square are the same thing
#
# Putting them together:
#
# $$\operatorname{Var}(z) = n_{\text{in}}\,\sigma_w^2\,\mathbb{E}[a^2]$$
#
# For the spread to stay constant from layer to layer we need
# $n_{\text{in}}\sigma_w^2 = 1$. **But ReLU throws away half the signal** — it is
# zero whenever $z$ is negative — so $\mathbb{E}[a^2]$ is only half of
# $\operatorname{Var}(z)$, and the weights have to be twice as large to
# compensate:
#
# $$\sigma_w^2 = \frac{2}{n_{\text{in}}} \quad \text{(He)}
#   \qquad\qquad
#   \sigma_w^2 = \frac{2}{n_{\text{in}} + n_{\text{out}}} \quad \text{(Glorot)}$$
#
# Glorot is the compromise between keeping the forward scale constant
# ($1/n_{\text{in}}$) and keeping the backward scale constant
# ($1/n_{\text{out}}$), used for saturating activations like tanh.

# %%
def he_init(fan_in, fan_out, seed=0):
    """Draw a (fan_in, fan_out) weight matrix with variance 2 / fan_in."""
    rng = np.random.default_rng(seed)
    ### PROMPT: Draw from a normal with the standard deviation He prescribes.
    ### HINT: standard deviation is the square root of the variance.
    ### BEGIN SOLUTION
    return rng.normal(scale=np.sqrt(2.0 / fan_in), size=(fan_in, fan_out))
    ### END SOLUTION


def glorot_init(fan_in, fan_out, seed=0):
    """Draw a (fan_in, fan_out) weight matrix with variance 2 / (fan_in + fan_out)."""
    rng = np.random.default_rng(seed)
    ### PROMPT: Same idea, with the Glorot variance.
    ### BEGIN SOLUTION
    return rng.normal(scale=np.sqrt(2.0 / (fan_in + fan_out)), size=(fan_in, fan_out))
    ### END SOLUTION


check.init_scale(he_init, glorot_init)

# %%
for fan_in, fan_out in ((16, 16), (128, 128), (1024, 64)):
    W = he_init(fan_in, fan_out, seed=1)
    print(f'he_init({fan_in:>5}, {fan_out:>4})  measured std {W.std():.4f}   '
          f'predicted {np.sqrt(2 / fan_in):.4f}')

# %% [markdown]
# ## 3. Measuring what actually happens over ten layers
#
# The formula predicts something specific. Check it: build a deep network with a
# chosen scale, push standard-normal inputs through it, and record how spread out
# the activations are at each layer.
#
# The `gain` is the multiplier on $1/\sqrt{n_{\text{in}}}$, so `gain = sqrt(2)`
# is He and `gain = 1` is the value that suits tanh.

# %%
def layer_stds(depth, width, gain, activation='relu', samples=256, seed=0):
    """
    Push random inputs through `depth` layers and return the standard deviation
    of the activations at each one.

    Weights are drawn with standard deviation gain / sqrt(width).
    """
    rng = np.random.default_rng(seed)
    f = {'relu': lambda t: np.maximum(0.0, t), 'tanh': np.tanh}[activation]
    A = rng.normal(size=(samples, width))
    stds = []
    ### PROMPT: For each layer, draw W, compute A = f(A @ W), and record A.std().
    ### BEGIN SOLUTION
    for _ in range(depth):
        W = rng.normal(scale=gain / np.sqrt(width), size=(width, width))
        A = f(A @ W)
        stds.append(float(A.std()))
    ### END SOLUTION
    return np.array(stds)


check.layer_stds(layer_stds)

# %%
depth, width = 10, 64
gains = [(0.7, 'gain 0.70 — too small'),
         (np.sqrt(2), 'gain 1.41 — He, correct for ReLU'),
         (2.2, 'gain 2.20 — too large')]

fig, ax = plt.subplots(figsize=(6.0, 3.6))
for gain, label in gains:
    stds = layer_stds(depth, width, gain, 'relu')
    ax.semilogy(range(1, depth + 1), stds, marker='o', ms=4, label=label)
    print(f'{label:<34} layer 1 {stds[0]:>10.4g}   layer {depth} {stds[-1]:>10.4g}   '
          f'ratio per layer {(stds[-1] / stds[0]) ** (1 / (depth - 1)):.3f}')

ax.axhline(1.0, color='#cbd2dd', ls='--', lw=1)
ax.set_xlabel('layer')
ax.set_ylabel('std of activations')
ax.set_title(f'{depth} layers of {width} ReLU units', fontsize=10)
ax.legend(fontsize=8)
ax.grid(alpha=0.25, which='both')
plt.tight_layout()
plt.show()

# %% [markdown]
# The correct gain gives a flat line. Below it the activations decay
# geometrically; above it they grow. Neither is survivable at real depth — and
# the same factor applies to the gradients on the way back, because the backward
# pass multiplies by the same matrices.

# %% [markdown]
# ## 4. Saturation turns an exploding forward pass into a vanishing backward one
#
# ReLU is unbounded, so too large a gain makes its activations explode. Tanh is
# bounded, so they cannot — but that is not good news. A saturated tanh unit sits
# where its slope is almost zero, and a unit with no slope passes no gradient.

# %%
fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
for ax, activation in zip(axes, ('relu', 'tanh')):
    for gain in (0.7, np.sqrt(2), 2.2):
        stds = layer_stds(depth, width, gain, activation)
        ax.semilogy(range(1, depth + 1), stds, marker='o', ms=3, label=f'gain {gain:.2f}')
    ax.axhline(1.0, color='#cbd2dd', ls='--', lw=1)
    ax.set_title(activation, fontsize=10)
    ax.set_xlabel('layer')
    ax.set_ylabel('std of activations')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25, which='both')
plt.tight_layout()
plt.show()

# Fraction of tanh units sitting in the flat region at a too-large gain.
rng = np.random.default_rng(3)
A = rng.normal(size=(512, width))
for _ in range(depth):
    A = np.tanh(A @ rng.normal(scale=2.2 / np.sqrt(width), size=(width, width)))
slope = 1 - A ** 2
print(f'with gain 2.2 and tanh, {np.mean(slope < 0.01):.1%} of units have slope below 0.01')
print(f'the average slope is {slope.mean():.4f} — a gradient crossing 10 such layers '
      f'is multiplied by about {slope.mean() ** depth:.2e}')

# %% [markdown]
# ## Try this
#
# * Set `activation='tanh'` and find the gain that keeps the tanh line flattest.
#   It is near 1.0, not 1.41 — tanh does not discard half the signal, so it does
#   not need the factor of two.
# * Raise `depth` to 40 with gain 0.7 and watch the activations reach the
#   smallest number a float can hold.
#
# **Next:** `04-activations.ipynb` — the functions themselves, and their
# derivatives.
