# %% [markdown]
# # 06 — Loss functions
#
# Companion to section 06 of the site.
#
# A loss function reduces a prediction and its target to one number saying how
# wrong the prediction is. Training minimises the average of that number.
#
# You will implement three losses and their derivatives, verify every derivative
# against a measurement, and then see the single most important consequence of
# the choice: with cross-entropy the gradient stays useful when the model is
# confidently wrong, and with squared error it does not.

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

EPS = 1e-12


def sigmoid(z):
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    e = np.exp(z[~pos])
    out[~pos] = e / (1.0 + e)
    return out

# %% [markdown]
# ## 1. Two losses
#
# **Mean squared error**, for predicting a real-valued target:
#
# $$L = \frac{1}{m}\sum_{i=1}^{m} (\hat{y}_i - y_i)^2$$
#
# **Binary cross-entropy**, for predicting a 0/1 label with a sigmoid output:
#
# $$L = -\frac{1}{m}\sum_{i=1}^{m}\Bigl[\, y_i\log \hat{y}_i + (1-y_i)\log(1-\hat{y}_i) \,\Bigr]$$
#
# Cross-entropy takes a logarithm, so a prediction of exactly 0 or 1 gives
# $-\infty$. Clip into $[\varepsilon, 1-\varepsilon]$ first.

# %%
def mse(y_hat, y):
    """Mean squared error over the examples."""
    ### PROMPT: Average of the squared differences.
    ### BEGIN SOLUTION
    return float(np.mean((y_hat - y) ** 2))
    ### END SOLUTION


def bce(y_hat, y):
    """Binary cross-entropy, averaged over the examples."""
    ### PROMPT: Clip first, then apply the formula above.
    ### HINT: np.clip(y_hat, EPS, 1 - EPS) keeps the logarithms finite.
    ### BEGIN SOLUTION
    p = np.clip(y_hat, EPS, 1 - EPS)
    return float(np.mean(-(y * np.log(p) + (1 - y) * np.log(1 - p))))
    ### END SOLUTION


check.losses(mse, bce)

# %%
print(f'{"prediction":>12} {"BCE (y=1)":>12} {"MSE (y=1)":>12}')
for p in (0.99, 0.9, 0.8, 0.5, 0.2, 0.01):
    print(f'{p:>12.2f} {bce(np.array([p]), np.array([1.0])):>12.4f} '
          f'{mse(np.array([p]), np.array([1.0])):>12.4f}')

print()
print(f'a coin flip on a balanced problem scores ln(2) = {np.log(2):.4f}')
print('anything at or above that has learned nothing usable')

# %% [markdown]
# ## 2. The derivatives
#
# Differentiate with respect to each prediction. Because both losses average
# over $m$ examples, the $1/m$ carries into the derivative.
#
# $$\frac{\partial L}{\partial \hat{y}_i} = \frac{2}{m}(\hat{y}_i - y_i)
#   \qquad\qquad
#   \frac{\partial L}{\partial \hat{y}_i} = \frac{1}{m}\cdot\frac{\hat{y}_i - y_i}{\hat{y}_i(1-\hat{y}_i)}$$

# %%
def dmse(y_hat, y):
    ### PROMPT: One derivative per prediction. Do not forget the 1/m.
    ### BEGIN SOLUTION
    return 2.0 * (y_hat - y) / len(y_hat)
    ### END SOLUTION


def dbce(y_hat, y):
    ### PROMPT: Clip the same way the loss does, then apply the formula.
    ### BEGIN SOLUTION
    p = np.clip(y_hat, EPS, 1 - EPS)
    return (p - y) / (p * (1 - p) * len(y_hat))
    ### END SOLUTION


check.loss_gradients({'mse': (mse, dmse), 'bce': (bce, dbce)})

# %% [markdown]
# ## 3. Why classification uses cross-entropy
#
# The loss is not what reaches the weights — the gradient is. And what actually
# matters is the derivative with respect to the **logit** $z$, because that is
# what backpropagation starts from.
#
# With a sigmoid output, $\partial \hat{y} / \partial z = \hat{y}(1-\hat{y})$,
# so by the chain rule:
#
# * MSE: $\dfrac{\partial L}{\partial z} = 2(\hat{y}-y)\cdot\hat{y}(1-\hat{y})$
# * BCE: $\dfrac{\partial L}{\partial z} = \dfrac{\hat{y}-y}{\hat{y}(1-\hat{y})}\cdot\hat{y}(1-\hat{y}) = \hat{y}-y$
#
# The saturating factor cancels exactly for cross-entropy. Implement it and
# confirm.

# %%
def sigmoid_bce_delta(z, y):
    """dL/dz for a sigmoid output scored by binary cross-entropy (summed, not averaged)."""
    ### PROMPT: After the cancellation this is remarkably short.
    ### BEGIN SOLUTION
    return sigmoid(z) - y
    ### END SOLUTION


check.sigmoid_bce_delta(sigmoid_bce_delta)

# %%
print(f'{"y_hat":>8} {"dL/dz (BCE)":>14} {"dL/dz (MSE)":>14} {"ratio":>10}')
for p in (0.99, 0.9, 0.5, 0.1, 0.01, 0.001):
    d_bce = p - 1.0
    d_mse = 2 * (p - 1.0) * p * (1 - p)
    print(f'{p:>8.3f} {d_bce:>14.6f} {d_mse:>14.6f} {abs(d_bce / d_mse):>10.1f}x')

print()
print('At y_hat = 0.001 — confidently wrong — cross-entropy still delivers a')
print('gradient of -0.999, while squared error delivers -0.002. The MSE model')
print('barely moves, precisely when it most needs to.')

# %%
ps = np.linspace(0.001, 0.999, 500)
fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.4))

axes[0].plot(ps, [bce(np.array([p]), np.array([1.0])) for p in ps], color='#3b82f6', lw=2, label='BCE')
axes[0].plot(ps, [mse(np.array([p]), np.array([1.0])) for p in ps], color='#e0761f', lw=2, ls='--', label='MSE')
axes[0].set_title('loss when the target is 1', fontsize=10)
axes[0].set_xlabel('prediction $\\hat{y}$')
axes[0].set_ylabel('loss')
axes[0].set_ylim(0, 5)

axes[1].plot(ps, ps - 1.0, color='#3b82f6', lw=2, label='BCE')
axes[1].plot(ps, 2 * (ps - 1.0) * ps * (1 - ps), color='#e0761f', lw=2, ls='--', label='MSE')
axes[1].set_title('gradient reaching the logit, $\\partial L / \\partial z$', fontsize=10)
axes[1].set_xlabel('prediction $\\hat{y}$')
axes[1].set_ylabel('$\\partial L/\\partial z$')

for ax in axes:
    ax.axvline(0.5, color='#cbd2dd', lw=1)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
plt.tight_layout()
plt.show()

# %% [markdown]
# Read the right-hand plot at $\hat{y}$ near 0. The blue line is at its steepest
# — maximum signal when the model is most wrong. The orange line is flat against
# zero, which is the failure.

# %% [markdown]
# ## 4. Where cross-entropy comes from
#
# It is not an arbitrary choice of "distance". Model the label as a coin flip
# whose bias the network predicts:
#
# $$P(y \mid \mathbf{x}) = \hat{y}^{\,y}(1-\hat{y})^{\,1-y}$$
#
# Check the two cases: at $y=1$ this is $\hat{y}$, at $y=0$ it is $1-\hat{y}$.
# The probability of the whole dataset is the product over examples, and
# maximising that is the same as maximising its logarithm — which turns the
# product into a sum.
#
# Negate, divide by $m$, and you have binary cross-entropy exactly.

# %%
rng = np.random.default_rng(0)
y_hat = rng.uniform(0.05, 0.95, size=8)
y = rng.integers(0, 2, size=8).astype(float)

likelihood = np.prod(y_hat ** y * (1 - y_hat) ** (1 - y))
neg_log_likelihood = -np.log(likelihood) / len(y)

print(f'probability of this dataset under the model : {likelihood:.3e}')
print(f'-log(that) / m                              : {neg_log_likelihood:.6f}')
print(f'bce(y_hat, y)                               : {bce(y_hat, y):.6f}')
print()
print('Identical. Minimising cross-entropy *is* maximising the likelihood.')

# %% [markdown]
# The same argument with a Gaussian instead of a coin flip gives mean squared
# error. So the pairing is not a convention: MSE assumes a real-valued target
# with additive noise, cross-entropy assumes a categorical outcome. Using MSE on
# a classification problem is fitting the wrong noise model, and the flat
# gradient above is the symptom.

# %% [markdown]
# ## 5. Categorical cross-entropy
#
# For $K$ mutually exclusive classes with a softmax output and a one-hot target,
# only one term of the sum survives:
#
# $$L = -\sum_{k=1}^{K} y_k \log \hat{y}_k = -\log \hat{y}_c$$
#
# where $c$ is the correct class. Use `log_softmax` from notebook 04 rather than
# taking a log of a probability.

# %%
def log_softmax(z):
    z = np.asarray(z, dtype=float)
    m = np.max(z, axis=-1, keepdims=True)
    shifted = z - m
    return shifted - np.log(np.sum(np.exp(shifted), axis=-1, keepdims=True))


def categorical_cross_entropy(logits, class_indices):
    """Mean -log p(correct class), computed from logits for numerical safety."""
    ### PROMPT: Take log_softmax, pick out the correct class, negate, average.
    ### HINT: log_probs[np.arange(len(idx)), idx] selects one entry per row.
    ### BEGIN SOLUTION
    log_probs = log_softmax(logits)
    picked = log_probs[np.arange(len(class_indices)), class_indices]
    return float(-np.mean(picked))
    ### END SOLUTION


logits = np.array([[2.0, 1.0, 0.1],
                   [0.1, 3.0, 0.2],
                   [1.0, 1.0, 1.0]])
targets = np.array([0, 1, 2])

probs = np.exp(log_softmax(logits))
for row, target, p in zip(logits, targets, probs):
    print(f'logits {row}  true class {target}  p(true) = {p[target]:.4f}  '
          f'loss = {-np.log(p[target]):.4f}')

print()
print(f'mean loss: {categorical_cross_entropy(logits, targets):.6f}')
print(f'baseline for 3 classes, ln(3) = {np.log(3):.6f}')
print(f'perplexity e^L = {np.exp(categorical_cross_entropy(logits, targets)):.3f} '
      f'(effective number of classes still in play)')

# %% [markdown]
# ## Try this
#
# * Remove the clipping from `bce` and evaluate it at `y_hat = 0.0` with
#   `y = 1.0`. You get `inf`, and one such example makes the whole average `inf`
#   for the rest of training.
# * Re-run the gradient check after deleting the `/ len(y_hat)` from `dmse`. The
#   check catches it and tells you the factor you dropped.
#
# **Next:** `07-gradient-descent.ipynb` — using these gradients to actually
# change the parameters.
