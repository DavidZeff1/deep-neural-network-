# %% [markdown]
# # 11 — Why depth matters
#
# Companion to section 11 of the site.
#
# The claim to test: for the same budget, spending parameters on **depth** buys
# more than spending them on **width**. You will train four networks that differ
# only in the number of hidden layers, count the regions a ReLU network can
# carve the plane into, and then find the cost that comes with depth.

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
from dnn.mlp import MLP

np.set_printoptions(precision=4, suppress=True)

# %% [markdown]
# ## 1. Four depths, everything else identical
#
# Same dataset, same width, same learning rate, same number of epochs. Only the
# number of hidden layers changes.

# %%
X, y = data.make_dataset('spiral', n=240, noise=0.05, seed=2)
WIDTH, EPOCHS, LR = 6, 400, 0.3

architectures = {
    0: [2, 1],
    1: [2, WIDTH, 1],
    2: [2, WIDTH, WIDTH, 1],
    3: [2, WIDTH, WIDTH, WIDTH, 1],
}

trained, final_losses = {}, {}
for layers, sizes in architectures.items():
    net = MLP(sizes, activation='relu', seed=808)
    h = net.fit(X, y, learning_rate=LR, batch_size=16, epochs=EPOCHS)
    trained[layers] = net
    final_losses[layers] = h['train_loss'][-1]
    print(f'{layers} hidden layers  {str(sizes):<22} {net.n_parameters:>4} params   '
          f'loss {h["train_loss"][-1]:.4f}   accuracy {h["train_acc"][-1]:.3f}')

check.depth_comparison(final_losses)

# %%
fig, axes = plt.subplots(1, 4, figsize=(15, 3.6))
for ax, (layers, net) in zip(axes, trained.items()):
    acc = float(np.mean(net.predict_classes(X) == y))
    plotting.plot_decision_boundary(
        net.predict, X, y, ax=ax,
        title=f'{layers} hidden layers\n{net.n_parameters} params, {acc:.0%}')
plt.tight_layout()
plt.show()

# %% [markdown]
# With no hidden layer the boundary is a single straight line and the accuracy
# is barely above chance — a spiral is not linearly separable. Each added layer
# lets the boundary bend more, and the arms start to separate.

# %% [markdown]
# ## 2. Depth versus width at a matched budget
#
# The comparison above is not quite fair: the deeper networks have more
# parameters. Fix that. Choose widths so that a one-layer and a two-layer
# network have roughly the same parameter count, and train both.

# %%
def params_for(sizes):
    return sum(a * b + b for a, b in zip(sizes, sizes[1:]))


candidates = [
    ('1 layer,  wide',  [2, 24, 1]),
    ('2 layers, medium', [2, 12, 12, 1]),
    ('3 layers, narrow', [2, 9, 9, 9, 1]),
]
for label, sizes in candidates:
    print(f'{label:<18} {str(sizes):<20} {params_for(sizes):>4} parameters')

# %%
fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
for ax, (label, sizes) in zip(axes, candidates):
    net = MLP(sizes, activation='relu', seed=808)
    h = net.fit(X, y, learning_rate=LR, batch_size=16, epochs=EPOCHS)
    plotting.plot_decision_boundary(
        net.predict, X, y, ax=ax,
        title=f'{label}\n{net.n_parameters} params, loss {h["train_loss"][-1]:.3f}')
    print(f'{label:<18} loss {h["train_loss"][-1]:.4f}   accuracy {h["train_acc"][-1]:.3f}')
plt.tight_layout()
plt.show()

print()
print('Roughly equal parameter counts. The deeper ones fit the spiral better.')

# %% [markdown]
# ## 3. Counting what a ReLU network can represent
#
# With ReLU, every layer is piecewise linear, so the whole network is piecewise
# linear: the input space is cut into regions, and on each region the network is
# exactly an affine function. Counting those regions measures what the
# architecture can express.
#
# **One hidden layer.** Each unit's boundary is a hyperplane. The number of
# pieces $n$ hyperplanes cut $d$-dimensional space into is
#
# $$r(n, d) = \sum_{j=0}^{d} \binom{n}{j}$$
#
# where $\binom{n}{k}$ — "n choose k" — counts the ways of picking $k$ items
# from $n$. For $d = 2$ that is $1 + n + n(n-1)/2$.

# %%
from math import comb


def shallow_regions(n_units, n_inputs=2):
    """Maximum linear regions of a one-hidden-layer ReLU network."""
    ### PROMPT: Sum the binomial coefficients from j = 0 to n_inputs.
    ### HINT: comb(n, j) from the math module is the binomial coefficient.
    ### BEGIN SOLUTION
    return sum(comb(n_units, j) for j in range(n_inputs + 1))
    ### END SOLUTION


def deep_regions(n_units, n_layers, n_inputs=2):
    """
    Lower bound on the linear regions of an n_layers-deep ReLU network.

    Each extra layer folds the space, multiplying the count by
    floor(n_units / n_inputs) ** n_inputs.
    """
    ### PROMPT: The shallow count, multiplied by the folding factor once per extra layer.
    ### BEGIN SOLUTION
    fold = (n_units // n_inputs) ** n_inputs
    return fold ** (n_layers - 1) * shallow_regions(n_units, n_inputs)
    ### END SOLUTION


check.linear_regions(shallow_regions, deep_regions)

# %%
n = 8
print(f'{"hidden layers":>14} {"parameters":>11} {"linear regions":>20}')
for layers in (1, 2, 3, 4):
    sizes = [2] + [n] * layers + [1]
    print(f'{layers:>14} {params_for(sizes):>11} {deep_regions(n, layers):>20,}')

print()
print('Parameters grow linearly with depth. The region count grows by a constant')
print(f'factor of {(n // 2) ** 2} per layer, so it grows exponentially.')
print()
print('To reach the 4-layer count with one hidden layer you would need width:')
target = deep_regions(n, 4)
width = next(w for w in range(1, 4000) if shallow_regions(w) >= target)
print(f'  {width:,} units -> {params_for([2, width, 1]):,} parameters, '
      f'versus {params_for([2] + [n] * 4 + [1])} for the deep one')

# %%
widths = np.arange(2, 33)
fig, ax = plt.subplots(figsize=(6.4, 3.8))
for layers, colour in ((1, '#94a3b8'), (2, '#e0761f'), (3, '#3b82f6'), (4, '#16a34a')):
    ax.semilogy(widths, [deep_regions(int(w), layers) for w in widths],
                color=colour, lw=2, label=f'{layers} hidden layer{"s" if layers > 1 else ""}')
ax.set_xlabel('units per layer')
ax.set_ylabel('linear regions (log scale)')
ax.legend(fontsize=8)
ax.grid(alpha=0.25, which='both')
ax.set_title('regions a ReLU network can carve the plane into', fontsize=10)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 4. What the hidden units actually learn
#
# The region count says what is *possible*. Look at what training produced: each
# hidden unit's activation across the plane, which is the feature the next layer
# combines.

# %%
net = trained[2]
probe = np.stack(np.meshgrid(np.linspace(-1.3, 1.3, 90),
                             np.linspace(-1.3, 1.3, 90)), axis=-1).reshape(-1, 2)
_, cache = net.forward(probe)

fig, axes = plt.subplots(2, WIDTH, figsize=(2.0 * WIDTH, 4.4))
for layer_index in (0, 1):
    activations = cache[layer_index]['a']
    for unit in range(WIDTH):
        ax = axes[layer_index, unit]
        field = activations[:, unit].reshape(90, 90)
        ax.imshow(field, origin='lower', extent=(-1.3, 1.3, -1.3, 1.3), cmap='Blues')
        ax.contour(np.linspace(-1.3, 1.3, 90), np.linspace(-1.3, 1.3, 90),
                   field, levels=[1e-6], colors='#14171f', linewidths=0.8)
        ax.set_xticks([]); ax.set_yticks([])
        if unit == 0:
            ax.set_ylabel(f'hidden {layer_index + 1}', fontsize=9)
plt.tight_layout()
plt.show()

print('Top row: every unit in the first hidden layer is a straight boundary — a')
print('single ReLU can only be that. Bottom row: the second layer combines them,')
print('and its units already have bent, multi-sided regions. That is the folding')
print('argument, visible.')

# %% [markdown]
# ## 5. What depth costs
#
# Depth is not free. Every layer adds a factor of $f'(z)$ and $W^{\top}$ to the
# backward product, so gradients have further to travel and more chances to
# shrink. With a saturating activation the effect is severe enough to reverse
# the benefit entirely.

# %%
print(f'{"activation":>10} {"layers":>7} {"final loss":>12} {"accuracy":>10}')
for activation in ('relu', 'sigmoid'):
    for layers in (1, 3):
        sizes = [2] + [WIDTH] * layers + [1]
        net = MLP(sizes, activation=activation, seed=808)
        h = net.fit(X, y, learning_rate=LR, batch_size=16, epochs=EPOCHS)
        print(f'{activation:>10} {layers:>7} {h["train_loss"][-1]:>12.4f} '
              f'{h["train_acc"][-1]:>10.3f}')

print()
print('With ReLU, going deeper helps. With sigmoid, the three-layer network is')
print('no better than the one-layer network despite being strictly more')
print('expressive — the gradient reaching its first layer is too small to use.')
print('Notebook 08 measured exactly that decay.')

# %% [markdown]
# ## 6. What the universal approximation theorem does and does not say
#
# It says: for any continuous function on a compact set, any non-polynomial
# activation, and any tolerance, there **exists** a one-hidden-layer network
# within that tolerance.
#
# It does not say how wide — the required width can be exponential in the input
# dimension. It does not say gradient descent can find those weights. And it
# says nothing about generalising from samples.
#
# So depth is not about what is representable in principle. It is about how many
# units you need, and whether training can find them.

# %%
wide = MLP([2, 64, 1], activation='relu', seed=808)
deep = MLP([2, 8, 8, 8, 1], activation='relu', seed=808)
for name, net in (('1 layer of 64', wide), ('3 layers of 8', deep)):
    h = net.fit(X, y, learning_rate=LR, batch_size=16, epochs=EPOCHS)
    print(f'{name:<16} {net.n_parameters:>4} params   loss {h["train_loss"][-1]:.4f}   '
          f'accuracy {h["train_acc"][-1]:.3f}')

print()
print('The wide network has more than twice the parameters and does not do better.')
print('Given enough units and enough epochs it would get there — the theorem')
print('guarantees that. The point is the cost.')

# %% [markdown]
# ## Try this
#
# * Change the dataset to `'linear'` and re-run section 1. Depth makes no
#   difference at all, because one straight line is already optimal. Depth buys
#   you nothing you did not need.
# * Raise `WIDTH` to 16 and re-run. The one-layer network improves a lot — width
#   does help, it is just less efficient per parameter.
#
# **Next:** `12-capstone.ipynb` — build the whole thing yourself, from scratch.
