# %% [markdown]
# # 05 — Forward propagation
#
# Companion to section 05 of the site.
#
# Forward propagation evaluates the network on an input. Each layer takes the
# previous layer's output, multiplies by its weights, adds its bias, applies the
# activation, and passes the result on.
#
# The one subtlety is that you must **keep the intermediate values**.
# Backpropagation needs them, and recomputing them would double the work. That
# storage is the reason training uses several times the memory of inference.

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

from dnn import check, data, plotting

np.set_printoptions(precision=4, suppress=True)

# %% [markdown]
# ## Setup: what you built earlier
#
# These are notebook 01's `init_params` and notebook 04's activations. Nothing
# new.

# %%
def init_params(sizes, seed=0):
    rng = np.random.default_rng(seed)
    return [{'W': rng.normal(scale=np.sqrt(2.0 / n_in), size=(n_in, n_out)),
             'b': np.zeros(n_out)}
            for n_in, n_out in zip(sizes, sizes[1:])]


def relu(z):
    return np.maximum(0.0, z)


def tanh(z):
    return np.tanh(z)


def sigmoid(z):
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    e = np.exp(z[~pos])
    out[~pos] = e / (1.0 + e)
    return out


ACTIVATIONS = {'relu': relu, 'tanh': tanh, 'sigmoid': sigmoid}

# %% [markdown]
# ## 1. One layer
#
# With one example per row, a layer is
#
# $$Z = XW + \mathbf{b} \qquad A = f(Z)$$
#
# `X` is `(m, n_in)`, `W` is `(n_in, n_out)`, so `Z` is `(m, n_out)`. The bias
# has `n_out` entries and is broadcast across all `m` rows.

# %%
def layer_forward(A_prev, W, b, f):
    """Return (Z, A) for one layer."""
    ### PROMPT: Two lines: the affine part, then the activation.
    ### BEGIN SOLUTION
    Z = A_prev @ W + b
    return Z, f(Z)
    ### END SOLUTION


rng = np.random.default_rng(0)
X = rng.normal(size=(4, 2))
W, b = rng.normal(size=(2, 3)), np.array([0.1, -0.2, 0.3])
Z, A = layer_forward(X, W, b, relu)
print('X', X.shape, ' W', W.shape, ' ->  Z', Z.shape, ' A', A.shape)
print()
print('Z:'); print(Z)
print('A = ReLU(Z):'); print(A)

# %% [markdown]
# ## 2. The whole network
#
# Chain the layers. Hidden layers use `f`; the final layer uses sigmoid, so the
# output reads as a probability.
#
# Return a **cache** alongside the output: for each layer, the activations that
# entered it, the pre-activations it computed, and the activations it produced.
# Notebook 08 consumes exactly this.

# %%
def forward(X, params, activation='relu'):
    """
    Run the network and return (output, cache).

    output: (m, n_out) probabilities from a sigmoid final layer
    cache:  list of {'a_prev', 'z', 'a'}, one per layer
    """
    f = ACTIVATIONS[activation]
    cache = []
    A = X
    ### PROMPT: Loop over the layers, recording a_prev, z and a for each one.
    ### HINT: The last layer is the only one that uses sigmoid.
    ### BEGIN SOLUTION
    for index, layer in enumerate(params):
        is_last = index == len(params) - 1
        Z, A_next = layer_forward(A, layer['W'], layer['b'], sigmoid if is_last else f)
        cache.append({'a_prev': A, 'z': Z, 'a': A_next})
        A = A_next
    ### END SOLUTION
    return A, cache


check.forward(lambda X, params: forward(X, params, 'relu'), init_params)

# %% [markdown]
# Notice what that check verified. It compared running a batch of 7 examples
# against running the 7 rows one at a time, and they matched. Each row must be
# processed independently — nothing is allowed to leak across the batch.

# %%
params = init_params([2, 4, 3, 1], seed=1)
X = rng.normal(size=(6, 2))
out, cache = forward(X, params)

print(f'output {out.shape}, all in [0, 1]: {bool(np.all((out >= 0) & (out <= 1)))}')
print(out.ravel())
print()
for i, entry in enumerate(cache):
    print(f"layer {i}:  a_prev {entry['a_prev'].shape}  ->  z {entry['z'].shape}  "
          f"->  a {entry['a'].shape}")

# %% [markdown]
# ## 3. Following one example through by hand
#
# Take a single row and reproduce the network's answer with explicit arithmetic.
# If your `forward` is right, the two agree exactly.

# %%
x = X[0]
print(f'input x = {x}')
print()

a = x
for i, layer in enumerate(params):
    z = a @ layer['W'] + layer['b']
    f = sigmoid if i == len(params) - 1 else relu
    a = f(z)
    print(f'layer {i}')
    print(f'  z = {z}')
    print(f'  a = {a}')

print()
print(f'by hand : {a[0]:.10f}')
print(f'forward : {out[0, 0]:.10f}')
assert abs(a[0] - out[0, 0]) < 1e-12

# %% [markdown]
# ## 4. What the untrained network looks like
#
# The weights are random, so the output is a meaningless but perfectly
# well-defined function of the input. Plotting it over the plane shows what
# training starts from.

# %%
Xd, yd = data.make_dataset('circles', n=200, noise=0.06, seed=0)

fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.8))
for ax, seed in zip(axes, (0, 1, 2)):
    p = init_params([2, 8, 8, 1], seed=seed)
    plotting.plot_decision_boundary(
        lambda pts, p=p: forward(pts, p, 'tanh')[0].ravel(),
        Xd, yd, ax=ax, title=f'untrained, seed {seed}')
plt.tight_layout()
plt.show()

print('Three different random initialisations, three different arbitrary boundaries.')
print('Accuracy right now:')
for seed in (0, 1, 2):
    p = init_params([2, 8, 8, 1], seed=seed)
    acc = np.mean((forward(Xd, p, 'tanh')[0].ravel() >= 0.5) == (yd >= 0.5))
    print(f'  seed {seed}: {acc:.3f}')

# %% [markdown]
# ## 5. Batching is free
#
# The parameters do not depend on how many examples you pass. One matrix
# multiplication handles the whole batch, and that is much faster than looping.

# %%
import time

big = rng.normal(size=(4096, 2))
wide = init_params([2, 256, 256, 1], seed=0)

start = time.time()
forward(big, wide)
batched = time.time() - start

start = time.time()
for row in big[:256]:
    forward(row[None, :], wide)
looped_256 = time.time() - start

print(f'4096 examples in one call : {batched * 1000:.1f} ms')
print(f'256 examples one at a time: {looped_256 * 1000:.1f} ms')
print(f'per example, batching is about {(looped_256 / 256) / (batched / 4096):.0f}x faster')

# %% [markdown]
# ## 6. What the cache costs
#
# Every entry of the cache is held until the backward pass reaches it. That is
# the memory difference between running a network and training one.

# %%
def cache_bytes(cache):
    return sum(entry['z'].nbytes + entry['a'].nbytes for entry in cache)


for m in (1, 32, 512):
    _, c = forward(rng.normal(size=(m, 2)), wide)
    param_bytes = sum(l['W'].nbytes + l['b'].nbytes for l in wide)
    print(f'batch {m:>4}: parameters {param_bytes / 1024:>8.1f} KiB   '
          f'cached activations {cache_bytes(c) / 1024:>8.1f} KiB')

print()
print('Parameters do not grow with the batch. Activations do, linearly.')
print('That is why doubling the batch size can run out of memory when doubling')
print('the model would not have.')

# %% [markdown]
# ## Try this
#
# * Delete `'a_prev'` from the cache and re-run the check. It fails, and the
#   message says why: notebook 08 cannot compute `dW` without it.
# * Change the final layer to `relu` instead of `sigmoid` and re-run. The check
#   objects that the output is no longer a probability.
#
# **Next:** `06-loss.ipynb` — turning a prediction into a single number that
# says how wrong it is.
