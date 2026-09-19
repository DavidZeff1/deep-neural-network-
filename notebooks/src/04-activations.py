# %% [markdown]
# # 04 — Activation functions
#
# Companion to section 04 of the site.
#
# You will implement four activations and their derivatives, then verify every
# derivative against a direct measurement. Then softmax, where the interesting
# problem is not the formula but making it survive large inputs.
#
# The derivative is the part that matters. During backpropagation the gradient
# arriving at a unit gets multiplied by $f'(z)$ before continuing, so across
# $L$ layers those factors multiply together.

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
# ## 1. Four activations
#
# | | definition | range |
# |---|---|---|
# | ReLU | $\max(0, z)$ | $[0, \infty)$ |
# | Leaky ReLU | $\max(\alpha z, z)$ | $(-\infty, \infty)$ |
# | Sigmoid | $1 / (1 + e^{-z})$ | $(0, 1)$ |
# | Tanh | $(e^{z} - e^{-z}) / (e^{z} + e^{-z})$ | $(-1, 1)$ |
#
# All four must work on whole arrays, not one number at a time.
#
# One trap: `np.exp(800)` overflows to infinity. A naive sigmoid then computes
# `1 / inf` for large negative `z`, which is fine, but `1 / (1 + inf)` produces a
# warning and `np.exp(-(-800))` produces `inf` outright. Handle both tails.

# %%
def relu(z):
    ### PROMPT: Zero for negative z, unchanged for positive z.
    ### BEGIN SOLUTION
    return np.maximum(0.0, z)
    ### END SOLUTION


def leaky_relu(z, alpha=0.1):
    ### PROMPT: Like ReLU, but negative z is multiplied by alpha instead of zeroed.
    ### HINT: np.where(condition, if_true, if_false) works elementwise.
    ### BEGIN SOLUTION
    return np.where(z > 0, z, alpha * z)
    ### END SOLUTION


def sigmoid(z):
    ### PROMPT: 1 / (1 + exp(-z)), written so that z = -800 does not overflow.
    ### HINT: For negative z, exp(z) / (1 + exp(z)) is algebraically identical
    ### HINT: and keeps every exponent at or below zero.
    ### BEGIN SOLUTION
    z = np.asarray(z, dtype=float)
    positive = z >= 0
    out = np.empty_like(z)
    out[positive] = 1.0 / (1.0 + np.exp(-z[positive]))
    exp_z = np.exp(z[~positive])
    out[~positive] = exp_z / (1.0 + exp_z)
    return out
    ### END SOLUTION


def tanh(z):
    ### PROMPT: numpy already has this one.
    ### BEGIN SOLUTION
    return np.tanh(z)
    ### END SOLUTION


check.activations(relu, leaky_relu, sigmoid, tanh)

# %%
print('sigmoid at the extremes:', sigmoid(np.array([-800.0, -20.0, 0.0, 20.0, 800.0])))

# %% [markdown]
# ## 2. The derivatives
#
# Differentiate each one by hand first, then write it down.
#
# * $\text{ReLU}'(z) = 1$ for $z > 0$, else $0$
# * $\text{LeakyReLU}'(z) = 1$ for $z > 0$, else $\alpha$
# * $\sigma'(z) = \sigma(z)\bigl(1 - \sigma(z)\bigr)$
# * $\tanh'(z) = 1 - \tanh^2(z)$
#
# The last two are worth noting: both express the derivative in terms of the
# function's own output, so a forward pass that already computed $a$ gets the
# derivative almost free.

# %%
def drelu(z):
    ### PROMPT: 1 where z is positive, 0 elsewhere.
    ### BEGIN SOLUTION
    return np.where(z > 0, 1.0, 0.0)
    ### END SOLUTION


def dleaky_relu(z, alpha=0.1):
    ### BEGIN SOLUTION
    return np.where(z > 0, 1.0, alpha)
    ### END SOLUTION


def dsigmoid(z):
    ### PROMPT: Reuse sigmoid(z) rather than differentiating from scratch.
    ### BEGIN SOLUTION
    s = sigmoid(z)
    return s * (1 - s)
    ### END SOLUTION


def dtanh(z):
    ### BEGIN SOLUTION
    return 1 - np.tanh(z) ** 2
    ### END SOLUTION


check.activation_gradients({
    'relu': (relu, drelu),
    'leaky_relu': (leaky_relu, dleaky_relu),
    'sigmoid': (sigmoid, dsigmoid),
    'tanh': (tanh, dtanh),
})

# %% [markdown]
# Those were not checked against stored answers. Each one was compared to
# $(f(z+h) - f(z-h)) / 2h$ computed from *your* forward function. If they agree,
# your derivative is right.

# %%
zs = np.linspace(-4, 4, 400)
fig, axes = plt.subplots(1, 4, figsize=(13, 3.0), sharex=True)
for ax, (name, f, df) in zip(axes, [
    ('ReLU', relu, drelu),
    ('Leaky ReLU', leaky_relu, dleaky_relu),
    ('Sigmoid', sigmoid, dsigmoid),
    ('Tanh', tanh, dtanh),
]):
    ax.plot(zs, f(zs), color='#3b82f6', lw=2, label='f(z)')
    ax.plot(zs, df(zs), color='#e0761f', lw=1.6, ls='--', label="f'(z)")
    ax.axhline(0, color='#cbd2dd', lw=1)
    ax.axvline(0, color='#cbd2dd', lw=1)
    ax.set_title(f'{name}   max f\' = {df(zs).max():.2f}', fontsize=10)
    ax.set_xlabel('z')
    ax.grid(alpha=0.25)
axes[0].legend(fontsize=8)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 3. Why `max f'` is the number to look at
#
# Backpropagation multiplies by $f'(z)$ once per layer. Sigmoid's derivative
# never exceeds 0.25, so ten sigmoid layers contribute a factor of at most
# $0.25^{10}$.

# %%
print(f'{"per-layer factor":>18} {"after 10":>12} {"after 30":>12}')
for factor, label in ((0.25, 'sigmoid (max)'), (0.9, '0.9'), (1.0, 'ReLU, z > 0'), (1.1, '1.1')):
    print(f'{label:>18} {factor ** 10:>12.2e} {factor ** 30:>12.2e}')

print()
print(f'a 32-bit float underflows to zero below about {np.finfo(np.float32).tiny:.1e},')
print(f'so a sigmoid stack reaches exactly zero gradient at depth '
      f'{int(np.log(np.finfo(np.float32).tiny) / np.log(0.25))}')

# %% [markdown]
# ## 4. Dead ReLU units
#
# A ReLU unit whose $z$ is negative for every training example outputs zero for
# all of them and has derivative zero for all of them. Its gradient is exactly
# zero, so no update ever changes it. It is not slow — it is dead.

# %%
w, b = np.array([0.4, -0.2]), -3.0
corners = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]], dtype=float)
z_values = corners @ w + b

print('a unit with w = (0.4, -0.2), b = -3.0, on data inside [-1, 1] x [-1, 1]:')
for corner, z in zip(corners, z_values):
    print(f'  x = {corner}   z = {z:+.2f}   ReLU(z) = {relu(z):.2f}   ReLU\'(z) = {drelu(z):.2f}')
print()
print(f'largest achievable z = {z_values.max():.2f} — always negative, so the unit never fires')
print(f'with leaky ReLU the slope would be {dleaky_relu(z_values.max()):.2f} instead of '
      f'{drelu(z_values.max()):.2f}, and it could recover')

# %% [markdown]
# ## 5. Softmax
#
# $$\mathrm{softmax}(\mathbf{z})_k = \frac{e^{z_k}}{\sum_{j} e^{z_j}}$$
#
# Two steps: exponentiate, which makes everything positive; then divide by the
# total, which makes the results sum to 1.
#
# The formula is shift invariant — adding the same constant to every logit
# cancels between the numerator and denominator. That is not just a curiosity,
# it is the fix for overflow: subtract the maximum first and no exponent is ever
# above zero.

# %%
def softmax(z):
    """Stable softmax. Works on a 1-D vector or on each row of a 2-D array."""
    ### PROMPT: Subtract the max along the last axis before exponentiating.
    ### HINT: keepdims=True keeps the shape right for both 1-D and 2-D inputs.
    ### BEGIN SOLUTION
    z = np.asarray(z, dtype=float)
    shifted = z - np.max(z, axis=-1, keepdims=True)
    exps = np.exp(shifted)
    return exps / np.sum(exps, axis=-1, keepdims=True)
    ### END SOLUTION


check.softmax(softmax)

# %%
logits = np.array([2.0, 1.0, 0.1])
print('logits          ', logits)
print('softmax         ', softmax(logits))
print('sums to         ', softmax(logits).sum())
print()
print('same logits + 100:', softmax(logits + 100))
print('identical, and no overflow — that is the shift invariance doing the work')
print()
naive = np.exp(logits + 800) / np.exp(logits + 800).sum()
print('the naive formula at +800:', naive, '  <- nan')

# %% [markdown]
# ## 6. log_softmax
#
# The loss needs $\log p_k$, not $p_k$. Forming the probability first and then
# taking its log throws away precision, and underflows to $-\infty$ when a
# probability rounds to zero. Compute the log directly:
#
# $$\log p_k = z_k - m - \log\!\sum_j e^{z_j - m}, \qquad m = \max_j z_j$$

# %%
def log_softmax(z):
    """log(softmax(z)), computed without ever forming the probability."""
    ### PROMPT: Use the identity above.
    ### BEGIN SOLUTION
    z = np.asarray(z, dtype=float)
    m = np.max(z, axis=-1, keepdims=True)
    shifted = z - m
    return shifted - np.log(np.sum(np.exp(shifted), axis=-1, keepdims=True))
    ### END SOLUTION


check.log_softmax(log_softmax, softmax)

# %%
extreme = np.array([0.0, -1000.0])
print('softmax        ', softmax(extreme))
print('log(softmax)   ', np.log(softmax(extreme)), '  <- -inf, information lost')
print('log_softmax    ', log_softmax(extreme), '  <- exact')

# %% [markdown]
# ## 7. Temperature
#
# Dividing the logits by a constant $T$ before the softmax changes how sharply
# the distribution concentrates, without ever changing which class is largest.

# %%
temperatures = [0.25, 0.5, 1.0, 2.0, 8.0]
fig, ax = plt.subplots(figsize=(6.2, 3.2))
width = 0.15
for i, T in enumerate(temperatures):
    p = softmax(logits / T)
    entropy = -np.sum(p * np.log(np.maximum(p, 1e-12)))
    ax.bar(np.arange(3) + i * width - 2 * width, p, width, label=f'T = {T}  (H = {entropy:.2f})')
ax.set_xticks(range(3))
ax.set_xticklabels([f'class {i + 1}\nz = {z}' for i, z in enumerate(logits)])
ax.set_ylabel('probability')
ax.legend(fontsize=8)
ax.set_title('softmax at different temperatures', fontsize=10)
plt.tight_layout()
plt.show()

print(f'maximum possible entropy for 3 classes: ln(3) = {np.log(3):.4f}')
print('low T concentrates on the argmax; high T approaches the uniform distribution')

# %% [markdown]
# ## Try this
#
# * Write a non-stable softmax and call it on `logits + 800`. Compare.
# * Plot `dsigmoid` and read off its peak. Everything about vanishing gradients
#   in deep sigmoid networks follows from that single number.
#
# **Next:** `05-forward.ipynb` — chaining layers into a full forward pass.
