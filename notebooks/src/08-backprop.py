# %% [markdown]
# # 08 — Backpropagation
#
# Companion to section 08 of the site.
#
# This is the notebook the course has been building towards. You already have
# every piece: the chain rule from notebook 00, the forward pass with its cache
# from notebook 05, and the loss derivative from notebook 06. Backpropagation
# assembles them.
#
# When you finish, a check will compare **every** gradient your code produces
# against a finite-difference measurement of **your own** forward pass. If they
# agree to six digits, your derivation is right. There is no ambiguity about it.

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

# %% [markdown]
# ## Setup: your earlier work
#
# Straight from notebooks 01, 04, 05 and 06.

# %%
def relu(z):
    return np.maximum(0.0, z)


def drelu(z):
    return np.where(z > 0, 1.0, 0.0)


def dtanh(z):
    return 1 - np.tanh(z) ** 2


def sigmoid(z):
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    e = np.exp(z[~pos])
    out[~pos] = e / (1.0 + e)
    return out


ACTIVATIONS = {'relu': (relu, drelu), 'tanh': (np.tanh, dtanh),
               'sigmoid': (sigmoid, lambda z: sigmoid(z) * (1 - sigmoid(z)))}


def init_params(sizes, seed=0):
    rng = np.random.default_rng(seed)
    return [{'W': rng.normal(scale=np.sqrt(2.0 / n_in), size=(n_in, n_out)),
             'b': np.zeros(n_out)}
            for n_in, n_out in zip(sizes, sizes[1:])]


def forward(X, params, activation='relu'):
    f = ACTIVATIONS[activation][0]
    cache, A = [], X
    for index, layer in enumerate(params):
        Z = A @ layer['W'] + layer['b']
        A = sigmoid(Z) if index == len(params) - 1 else f(Z)
        cache.append({'a_prev': cache[-1]['a'] if cache else X, 'z': Z, 'a': A})
    return A, cache


def bce(y_hat, y):
    p = np.clip(np.asarray(y_hat).ravel(), EPS, 1 - EPS)
    return float(np.mean(-(y * np.log(p) + (1 - y) * np.log(1 - p))))

# %% [markdown]
# ## 1. Start at the output
#
# Backpropagation carries one quantity backwards: $\delta$, one number per unit,
# meaning *how much the loss changes if this unit's total changes*.
#
# $$\delta^{(l)} = \frac{\partial L}{\partial \mathbf{z}^{(l)}}$$
#
# At the output layer, notebook 06 already did the work. With a sigmoid output
# scored by binary cross-entropy, the saturating factor cancels and what remains
# is the error itself — divided by $m$, because the loss is a mean:
#
# $$\delta^{(L)} = \frac{\hat{\mathbf{y}} - \mathbf{y}}{m}$$

# %%
def output_delta(y_hat, y):
    """dL/dz at the output layer, for sigmoid + mean binary cross-entropy."""
    ### PROMPT: The cancellation from notebook 06, divided by the number of examples.
    ### HINT: y has shape (m,) and y_hat has shape (m, 1) — reshape y to match.
    ### BEGIN SOLUTION
    m = len(y)
    return (y_hat - y.reshape(-1, 1)) / m
    ### END SOLUTION


check.output_delta(output_delta)

# %% [markdown]
# ## 2. Two rules, applied once per layer
#
# Given $\delta$ for a layer, everything else is immediate.
#
# **The parameter gradients.** The weight $W_{ji}$ appears in exactly one place
# in the whole network: the product $W_{ji}\,a_i$ inside $z_j$. So the chain-rule
# sum collapses to a single term, and
#
# $$\frac{\partial L}{\partial W} = A_{\text{prev}}^{\top}\,\delta
#   \qquad
#   \frac{\partial L}{\partial \mathbf{b}} = \sum_{\text{rows}} \delta$$
#
# The bias gradient is just $\delta$ summed over the batch, because a bias is
# added directly with no input to multiply by.
#
# **The recursion.** To get $\delta$ for the previous layer, send this one's
# backwards through the same weights that carried the signal forwards, then scale
# by how responsive each unit's activation was:
#
# $$\delta^{(l-1)} = \left(\delta^{(l)} W^{(l)\top}\right) \odot f'\!\left(\mathbf{z}^{(l-1)}\right)$$
#
# The transpose is what reverses the direction. The $\odot$ multiplies position
# by position.

# %%
def backward(cache, y, params, activation='relu'):
    """
    Compute dL/dW and dL/db for every layer.

    Returns a list of {'dW', 'db'}, in the same order as `params`.
    """
    df = ACTIVATIONS[activation][1]
    grads = [None] * len(params)
    delta = output_delta(cache[-1]['a'], y)

    ### PROMPT: Walk backwards. Record dW and db, then propagate delta one layer down.
    ### HINT: for l in reversed(range(len(params))) walks from the output to the input.
    ### HINT: Only propagate delta when there is a previous layer to send it to.
    ### BEGIN SOLUTION
    for l in reversed(range(len(params))):
        grads[l] = {
            'dW': cache[l]['a_prev'].T @ delta,
            'db': delta.sum(axis=0),
        }
        if l > 0:
            delta = (delta @ params[l]['W'].T) * df(cache[l - 1]['z'])
    ### END SOLUTION
    return grads


check.backward(
    forward_fn=lambda X, p: forward(X, p, 'relu'),
    backward_fn=lambda cache, y, p: backward(cache, y, p, 'relu'),
    init_fn=init_params,
    loss_fn=bce,
)

# %% [markdown]
# That check perturbed every single weight and bias of a 3→5→4→1 network by
# $10^{-5}$, measured how the loss moved, and compared the result to what your
# `backward` claimed. 61 parameters, all agreeing.
#
# This is the standard way to debug a gradient, and it is worth internalising:
# **if you ever write a derivative by hand, check it numerically before trusting
# it.**

# %% [markdown]
# ## 3. Seeing it work
#
# Gradients are only useful if following them reduces the loss. One step should
# do that, for a small enough learning rate.

# %%
rng = np.random.default_rng(0)
params = init_params([2, 5, 1], seed=4)
X = rng.normal(size=(40, 2))
y = (X[:, 0] * X[:, 1] > 0).astype(float)

out, cache = forward(X, params)
print(f'loss before   : {bce(out, y):.6f}')

grads = backward(cache, y, params)
for layer, grad in zip(params, grads):
    layer['W'] -= 0.5 * grad['dW']
    layer['b'] -= 0.5 * grad['db']

print(f'loss after one step: {bce(forward(X, params)[0], y):.6f}')

# %%
# And repeatedly.
params = init_params([2, 8, 8, 1], seed=4)
losses = []
for step in range(400):
    out, cache = forward(X, params, 'tanh')
    losses.append(bce(out, y))
    grads = backward(cache, y, params, 'tanh')
    for layer, grad in zip(params, grads):
        layer['W'] -= 0.5 * grad['dW']
        layer['b'] -= 0.5 * grad['db']

fig, ax = plt.subplots(figsize=(5.6, 3.2))
ax.plot(losses, color='#3b82f6')
ax.set_xlabel('step')
ax.set_ylabel('loss')
ax.set_title('400 full-batch gradient steps', fontsize=10)
ax.grid(alpha=0.25)
plt.tight_layout()
plt.show()

accuracy = np.mean((forward(X, params, 'tanh')[0].ravel() >= 0.5) == (y >= 0.5))
print(f'loss {losses[0]:.4f} -> {losses[-1]:.4f},  accuracy {accuracy:.3f}')

# %% [markdown]
# That is a complete training loop. Notebook 09 adds mini-batches and metrics,
# but nothing more fundamental.

# %% [markdown]
# ## 4. Why backwards, and not forwards
#
# Both directions compute exact derivatives. They differ in what one pass gives
# you.
#
# * **Forward mode** propagates the derivative with respect to one chosen input.
#   One pass gives the derivative of everything with respect to *one* parameter.
# * **Reverse mode** — backpropagation — propagates the derivative of one chosen
#   output. One pass gives the derivative of *one* scalar with respect to
#   everything.
#
# Training needs the second shape: one scalar loss, many parameters. Measure the
# difference.

# %%
import time

params = init_params([10, 40, 40, 1], seed=0)
X = rng.normal(size=(64, 10))
y = rng.integers(0, 2, size=64).astype(float)
n_params = sum(l['W'].size + l['b'].size for l in params)

start = time.time()
out, cache = forward(X, params)
backward(cache, y, params)
backprop_time = time.time() - start

# Finite differences need two forward passes per parameter. Time 200 of them.
sample = 200
start = time.time()
count = 0
for layer in params:
    for idx in np.ndindex(layer['W'].shape):
        original = layer['W'][idx]
        layer['W'][idx] = original + 1e-5
        bce(forward(X, params)[0], y)
        layer['W'][idx] = original - 1e-5
        bce(forward(X, params)[0], y)
        layer['W'][idx] = original
        count += 1
        if count >= sample:
            break
    if count >= sample:
        break
per_param = (time.time() - start) / sample

print(f'{n_params:,} parameters')
print(f'backpropagation, all gradients : {backprop_time * 1000:>10.2f} ms')
print(f'finite differences, all of them: {per_param * n_params * 1000:>10.2f} ms')
print(f'                        factor : {per_param * n_params / backprop_time:>10.0f}x')
print()
print('That factor grows with the parameter count. At a million parameters it is')
print('the difference between 3 milliseconds and about 17 minutes — per gradient.')
print('This is why neural networks are trained at all, and why the check above is')
print('a test rather than a method.')

# %% [markdown]
# ## 5. Where gradients vanish
#
# Unrolling the recursion from the loss back to layer $l$ leaves a product with
# one factor per layer in between — the weights, and the activation slopes.
# Anything other than 1 compounds.
#
# Sigmoid's derivative never exceeds 0.25. Measure what that does over ten
# layers.

# %%
def gradient_norms_by_layer(activation, depth=10, width=32, seed=0):
    """Norm of dL/dW at each layer, ordered from the first layer to the last."""
    rng = np.random.default_rng(seed)
    sizes = [width] * (depth + 1) + [1]
    params = init_params(sizes, seed=seed)
    X = rng.normal(size=(64, width))
    y = rng.integers(0, 2, size=64).astype(float)
    ### PROMPT: Run forward and backward, then take the norm of each layer's dW.
    ### HINT: np.linalg.norm(grad['dW']) collapses a matrix to a single number.
    ### BEGIN SOLUTION
    _, cache = forward(X, params, activation)
    grads = backward(cache, y, params, activation)
    return np.array([np.linalg.norm(g['dW']) for g in grads])
    ### END SOLUTION


norms_relu = gradient_norms_by_layer('relu')
norms_sigmoid = gradient_norms_by_layer('sigmoid')
check.gradient_flow(norms_relu, norms_sigmoid)

# %%
fig, ax = plt.subplots(figsize=(6.2, 3.6))
ax.semilogy(range(1, len(norms_relu) + 1), norms_relu, 'o-', color='#3b82f6', label='ReLU')
ax.semilogy(range(1, len(norms_sigmoid) + 1), norms_sigmoid, 'o--', color='#e0761f', label='sigmoid')
ax.set_xlabel('layer (1 = closest to the input)')
ax.set_ylabel(r'$\|\partial L / \partial W\|$')
ax.set_title('gradient magnitude by depth', fontsize=10)
ax.legend(fontsize=9)
ax.grid(alpha=0.25, which='both')
plt.tight_layout()
plt.show()

print(f'ReLU    : layer 1 receives {norms_relu[0] / norms_relu[-1]:.3g} of the last layer')
print(f'sigmoid : layer 1 receives {norms_sigmoid[0] / norms_sigmoid[-1]:.3g} of the last layer')
print()
print('The last layers of the sigmoid network train fine. The first ones barely')
print('move, so the features they compute stay close to their random starting')
print('values however long you train.')

# %% [markdown]
# ## 6. Your implementation, kept
#
# Notebooks 09 to 12 build on this. So that they can stay focused on training
# rather than re-deriving gradients, a reference version of everything you just
# wrote lives in `dnn/mlp.py`. It uses the identical conventions.
#
# Open it now if you like — but only now, not before.

# %%
from dnn import mlp

reference = mlp.MLP([2, 5, 1], activation='relu', seed=4)
params = init_params([2, 5, 1], seed=4)

X = rng.normal(size=(12, 2))
y = rng.integers(0, 2, size=12).astype(float)

mine_out, mine_cache = forward(X, params, 'relu')
mine_grads = backward(mine_cache, y, params, 'relu')
their_grads = reference.gradients(X, y)

print('your forward matches the reference :',
      bool(np.allclose(mine_out.ravel(), reference.predict(X).ravel())))
print('your gradients match the reference :',
      bool(all(np.allclose(a['dW'], b['dW']) and np.allclose(a['db'], b['db'])
               for a, b in zip(mine_grads, their_grads))))

# %% [markdown]
# ## Try this
#
# * Delete the `* df(cache[l - 1]['z'])` from the recursion and re-run the check.
#   It fails and names the layer where the disagreement is largest — dropping
#   the activation slope is the single most common backpropagation bug.
# * Replace `delta.sum(axis=0)` with `delta.mean(axis=0)` for the bias. The check
#   catches the factor of $m$.
# * Set `depth=30` in `gradient_norms_by_layer('sigmoid')`. The first layer's
#   gradient reaches the smallest number a float can represent.
#
# **Next:** `09-training.ipynb` — mini-batches, epochs, and watching a real
# decision boundary form.
