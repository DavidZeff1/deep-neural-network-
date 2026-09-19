# %% [markdown]
# # 02 — Neurons
#
# Companion to section 02 of the site.
#
# One neuron: multiply each input by its weight, add the products, add a bias,
# pass the total through a function. Two exercises build it. The third proves,
# by exhaustive search, that a single neuron cannot solve XOR — which is the
# whole reason the rest of this course exists.

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

from dnn import check, plotting

np.set_printoptions(precision=4, suppress=True)

# %% [markdown]
# ## 1. One neuron
#
# $$z = \sum_{i=1}^{n} w_i x_i + b \qquad a = f(z)$$
#
# The worked example from the site: $x = (1.0, 0.5)$, $w = (0.8, -0.3)$,
# $b = 0.2$, so
#
# $$z = (0.8)(1.0) + (-0.3)(0.5) + 0.2 = 0.85$$

# %%
def relu(z):
    return np.maximum(0.0, z)


def neuron(x, w, b, f):
    """Compute one neuron's output: weighted sum, plus bias, through f."""
    ### PROMPT: One line: the weighted sum plus the bias, passed through f.
    ### BEGIN SOLUTION
    return f(np.dot(w, x) + b)
    ### END SOLUTION


check.neuron(neuron)

# %%
x = np.array([1.0, 0.5])
w = np.array([0.8, -0.3])
b = 0.2

z = np.dot(w, x) + b
print(f'z = ({w[0]})({x[0]}) + ({w[1]})({x[1]}) + {b} = {z:.4f}')
print(f'ReLU(z) = {neuron(x, w, b, relu):.4f}')

# %% [markdown]
# ## 2. The neuron is a line
#
# The set of inputs where $z = 0$ is the line $w_1x_1 + w_2x_2 + b = 0$. On one
# side $z$ is positive, on the other it is negative, and the activation only
# decides what happens on each side — it never moves the line.
#
# There is more structure than that. The pre-activation is the **signed distance
# to the line, scaled by the length of the weight vector**:
#
# $$z(\mathbf{x}) = \lVert \mathbf{w} \rVert \cdot d(\mathbf{x})
#   \qquad\text{so}\qquad
#   d(\mathbf{x}) = \frac{\mathbf{w}\cdot\mathbf{x} + b}{\lVert \mathbf{w} \rVert}$$

# %%
def signed_distance(x, w, b):
    """How far x sits from the line z = 0, positive on the side w points to."""
    ### PROMPT: Divide the pre-activation by the length of w.
    ### HINT: np.linalg.norm(w) gives the length of the weight vector.
    ### BEGIN SOLUTION
    return (np.dot(w, x) + b) / np.linalg.norm(w)
    ### END SOLUTION


check.signed_distance(signed_distance)

# %%
print(f'||w||                     = {np.linalg.norm(w):.4f}')
print(f'distance from x to line   = {signed_distance(x, w, b):.4f}')
print(f'distance from origin      = {abs(b) / np.linalg.norm(w):.4f}')
print()
print('check: ||w|| * distance should equal z')
print(f'  {np.linalg.norm(w):.4f} * {signed_distance(x, w, b):.4f} = '
      f'{np.linalg.norm(w) * signed_distance(x, w, b):.4f},  z = {z:.4f}')

# %%
# Draw the neuron's output across the plane, with the z = 0 line marked.
def field(points):
    zs = points @ w + b
    return 1 / (1 + np.exp(-3 * zs))     # squashed only so the colours read well


ax = plotting.plot_decision_boundary(field, limits=(-2, 2), title='one neuron: the line z = 0')
ax.plot([x[0]], [x[1]], 'o', color='#14171f', ms=8, mfc='none', mew=2)
ax.annotate(f'x, z = {z:.2f}', xy=(x[0], x[1]), xytext=(x[0] - 1.1, x[1] + 0.5), fontsize=9)
ax.arrow(0, 0, w[0], w[1], color='#14171f', width=0.02, length_includes_head=True)
ax.annotate('w', xy=(w[0], w[1]), xytext=(w[0] + 0.08, w[1] - 0.18), fontsize=10)
plt.show()

# %% [markdown]
# The arrow is $\mathbf{w}$. It is perpendicular to the line and points towards
# increasing $z$. Change `w` and `b` in the cells above and re-run — the line
# rotates with the direction of $\mathbf{w}$ and slides with $b$.

# %% [markdown]
# ## 3. Proving a single neuron cannot solve XOR
#
# XOR labels the four corners of a square: opposite corners share a label.
#
# | $x_1$ | $x_2$ | label |
# |---|---|---|
# | 0 | 0 | 0 |
# | 0 | 1 | 1 |
# | 1 | 0 | 1 |
# | 1 | 1 | 0 |
#
# The site proves by contradiction that no weights can reproduce this. Here you
# will check it by brute force: sweep a large grid of $(w_1, w_2, b)$ and record
# the best accuracy any of them achieves.
#
# Because the activation is monotonic, "output above a threshold" is the same as
# "$z$ above some constant", so classifying by `z > 0` loses no generality once
# $b$ is free to move.

# %%
XOR_X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
XOR_Y = np.array([0.0, 1.0, 1.0, 0.0])


def best_single_neuron_accuracy(grid=41, limit=4.0):
    """
    Sweep w1, w2 and b over a grid and return the best accuracy on XOR.

    Classify a corner as 1 when z > 0.
    """
    values = np.linspace(-limit, limit, grid)
    best = 0.0
    ### PROMPT: Try every combination of w1, w2, b and keep the best accuracy.
    ### HINT: z = XOR_X @ np.array([w1, w2]) + b gives all four corners at once.
    ### BEGIN SOLUTION
    for w1 in values:
        for w2 in values:
            for bias in values:
                z = XOR_X @ np.array([w1, w2]) + bias
                accuracy = np.mean((z > 0) == (XOR_Y > 0.5))
                best = max(best, accuracy)
    ### END SOLUTION
    return best


best = best_single_neuron_accuracy()
print(f'best accuracy over the whole grid: {best:.3f}  ({best * 4:.0f} of 4 corners)')
check.xor_is_impossible(best)

# %% [markdown]
# 68,921 combinations tried, and not one of them gets all four corners. That is
# not a failure of the search — the site's proof shows no such weights exist at
# any precision.
#
# The reason is visible as soon as you plot it. A single neuron splits the plane
# with one straight line, and no straight line puts the two blue corners on one
# side and the two orange corners on the other.

# %%
fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
for ax, (w1, w2, bias, label) in zip(axes, [
    (1.0, 1.0, -0.5, 'w = (1, 1), b = -0.5'),
    (1.0, -1.0, 0.0, 'w = (1, -1), b = 0'),
    (-1.0, 1.0, -0.5, 'w = (-1, 1), b = -0.5'),
]):
    wv = np.array([w1, w2])
    plotting.plot_decision_boundary(
        lambda pts, wv=wv, bias=bias: 1 / (1 + np.exp(-4 * (pts @ wv + bias))),
        limits=(-0.5, 1.5), ax=ax, title=label)
    plotting.plot_dataset(XOR_X, XOR_Y, ax=ax)
    correct = int(np.sum(((XOR_X @ wv + bias) > 0) == (XOR_Y > 0.5)))
    ax.set_xlabel(f'{correct} of 4 correct')
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 4. Two neurons are enough
#
# Nothing here is hand-wavy about why the fix works. XOR becomes separable once
# you look at *two* linear functions instead of one. Take
#
# $$h_1 = \text{ReLU}(x_1 + x_2 - 0.5) \qquad h_2 = \text{ReLU}(x_1 + x_2 - 1.5)$$
#
# and then $y = h_1 - 2h_2$ classifies all four corners correctly.

# %%
def hand_built_xor(points):
    h1 = relu(points @ np.array([1.0, 1.0]) - 0.5)
    h2 = relu(points @ np.array([1.0, 1.0]) - 1.5)
    return h1 - 2 * h2


outputs = hand_built_xor(XOR_X)
for point, target, out in zip(XOR_X, XOR_Y, outputs):
    print(f'x = {point}  target {target:.0f}  ->  {out:+.2f}  '
          f'{"correct" if (out > 0.25) == (target > 0.5) else "WRONG"}')

print()
print(f'accuracy: {np.mean((outputs > 0.25) == (XOR_Y > 0.5)):.3f}')

# %%
plotting.plot_decision_boundary(
    lambda pts: np.clip(hand_built_xor(pts), 0, 1),
    limits=(-0.5, 1.5), title='two hidden units, chosen by hand')
plotting.plot_dataset(XOR_X, XOR_Y, ax=plt.gca())
plt.show()

# %% [markdown]
# Those weights were chosen by hand. The rest of the course is about finding
# them automatically — which needs a way to measure how wrong the network is
# (notebook 06) and a way to improve it (notebooks 07 and 08).
#
# **Next:** `03-weights.ipynb` — why the starting values of the weights decide
# whether a deep network trains at all.
