# %% [markdown]
# # 01 — Network structure
#
# Companion to section 01 of the site.
#
# A feed-forward network is fully described by one thing: a list of layer sizes.
# Everything else — the shape of every weight matrix, the number of parameters,
# the cost of a forward pass — follows from that list by arithmetic.
#
# By the end of this notebook you will have a function that turns `[2, 6, 4, 1]`
# into a ready-to-train set of parameters, and you will know exactly how many
# numbers that network contains and why.

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
# ## 1. From sizes to shapes
#
# `sizes = [2, 6, 4, 1]` means: 2 inputs, a hidden layer of 6, a hidden layer of
# 4, and 1 output. Adjacent pairs become weighted layers, so this list describes
# **three** of them — the input is not a weighted layer, because it holds no
# parameters.
#
# Using the numpy convention from notebook 00 (`X @ W + b`, one example per row),
# the layer connecting `n_in` units to `n_out` units has
#
# * `W` of shape `(n_in, n_out)`
# * `b` of shape `(n_out,)`

# %%
def layer_shapes(sizes):
    """
    Return [(W_shape, b_shape), ...], one tuple per weighted layer.

    layer_shapes([2, 3, 1]) -> [((2, 3), (3,)), ((3, 1), (1,))]
    """
    ### PROMPT: Walk over adjacent pairs of sizes and build the shapes.
    ### HINT: zip(sizes, sizes[1:]) gives you each adjacent pair.
    ### BEGIN SOLUTION
    return [((n_in, n_out), (n_out,)) for n_in, n_out in zip(sizes, sizes[1:])]
    ### END SOLUTION


check.layer_shapes(layer_shapes)

# %%
for shape in layer_shapes([2, 6, 4, 1]):
    print(shape)

# %% [markdown]
# ## 2. Counting the parameters
#
# Each layer contributes `n_in * n_out` weights and `n_out` biases:
#
# $$\#\text{params} = \sum_{l=1}^{L} \left( n_l\, n_{l-1} + n_l \right)$$
#
# These are the numbers training will change. Nothing else about the network
# moves.

# %%
def count_parameters(sizes):
    """Total number of trainable numbers in a network with these layer sizes."""
    ### PROMPT: Add up the weights and biases of every layer.
    ### BEGIN SOLUTION
    return sum(n_in * n_out + n_out for n_in, n_out in zip(sizes, sizes[1:]))
    ### END SOLUTION


check.count_parameters(count_parameters)

# %%
for sizes in ([2, 3, 1], [2, 8, 1], [2, 8, 8, 1], [2, 64, 1], [784, 128, 64, 10]):
    print(f'{str(sizes):<22} {count_parameters(sizes):>8,} parameters')

# %% [markdown]
# Two things worth noticing in that table.
#
# `[2, 8, 8, 1]` has 105 parameters and `[2, 64, 1]` has 257 — the wide shallow
# network is more than twice the size. Notebook 11 shows which one actually fits
# a hard dataset better.
#
# And `[784, 128, 64, 10]` — a small image classifier — already has over 109,000
# parameters, almost all of them in the first layer. Width in the first layer is
# expensive when the input is large.

# %% [markdown]
# ## 3. Creating the parameters
#
# Now build them. Two rules, both from section 03 of the site, which you will
# derive properly in notebook 03:
#
# * **Weights must be random.** If two units in a layer start identical they
#   compute the same thing, receive the same gradient, and stay identical for
#   ever.
# * **Biases start at zero.** The weights already break the symmetry, so there
#   is nothing for a random bias to add.
#
# For the scale, use $\sigma = \sqrt{2 / n_{\text{in}}}$ — He initialisation.
# Notebook 03 shows where that number comes from.

# %%
def init_params(sizes, seed=0):
    """
    Build the parameters for a network with these layer sizes.

    Returns a list of dicts, one per weighted layer: {'W': ..., 'b': ...}.
    """
    rng = np.random.default_rng(seed)
    ### PROMPT: Build one dict per layer with random W and zero b.
    ### HINT: rng.normal(scale=sigma, size=(n_in, n_out)) draws the weights.
    ### BEGIN SOLUTION
    params = []
    for n_in, n_out in zip(sizes, sizes[1:]):
        sigma = np.sqrt(2.0 / n_in)
        params.append({
            'W': rng.normal(scale=sigma, size=(n_in, n_out)),
            'b': np.zeros(n_out),
        })
    return params
    ### END SOLUTION


check.init_params(init_params)

# %%
params = init_params([2, 4, 1], seed=0)
for i, layer in enumerate(params):
    print(f"layer {i}: W {layer['W'].shape}, b {layer['b'].shape}")
    print(layer['W'])
    print()

total = sum(layer['W'].size + layer['b'].size for layer in params)
print(f'{total} parameters, and count_parameters says {count_parameters([2, 4, 1])}')

# %% [markdown]
# ## 4. What a network costs
#
# Layer `l` performs `n_in * n_out` multiply-adds per example. A forward pass is
# therefore about `2 * sum(n_in * n_out)` floating-point operations — roughly
# twice the parameter count. Backpropagation costs about twice that again,
# because it computes both the parameter gradients and the signal passed back.
#
# A useful rule of thumb: **one training step on one example costs about
# `6 × #params` FLOPs.**

# %%
def forward_flops(sizes):
    """Approximate floating-point operations for one example's forward pass."""
    ### PROMPT: Two operations (one multiply, one add) per weight.
    ### BEGIN SOLUTION
    return 2 * sum(n_in * n_out for n_in, n_out in zip(sizes, sizes[1:]))
    ### END SOLUTION


for sizes in ([2, 8, 8, 1], [784, 128, 64, 10]):
    p = count_parameters(sizes)
    print(f'{str(sizes):<22} {p:>8,} params   {forward_flops(sizes):>10,} FLOPs forward   '
          f'{6 * p:>11,} FLOPs per training step')

# %% [markdown]
# ## 5. Where the parameters go
#
# Parameter count is not spread evenly. Plot it per layer and the imbalance is
# obvious — and it tells you where a memory or speed problem will come from.

# %%
sizes = [784, 128, 64, 10]
per_layer = [n_in * n_out + n_out for n_in, n_out in zip(sizes, sizes[1:])]

fig, ax = plt.subplots(figsize=(5.4, 3.0))
ax.bar(range(len(per_layer)), per_layer, color='#3b82f6')
ax.set_xticks(range(len(per_layer)))
ax.set_xticklabels([f'{a}→{b}' for a, b in zip(sizes, sizes[1:])])
ax.set_ylabel('parameters')
ax.set_title(f'{sizes} — {sum(per_layer):,} parameters in total', fontsize=10)
for i, v in enumerate(per_layer):
    ax.text(i, v, f'{v:,}', ha='center', va='bottom', fontsize=9)
ax.margins(y=0.18)
plt.tight_layout()

print(f'the first layer holds {per_layer[0] / sum(per_layer):.0%} of the parameters')

# %% [markdown]
# ## Try this
#
# * Change `sizes` above to `[2, 64, 1]` and to `[2, 8, 8, 8, 1]`. Both are
#   small; the distribution is completely different.
# * What happens to `count_parameters` if you double every hidden width? Try it.
#   The growth is quadratic in the width of the *interior* layers, because both
#   the rows and the columns of those matrices grow.
#
# **Next:** `02-neurons.ipynb` — what a single one of these units actually does.
