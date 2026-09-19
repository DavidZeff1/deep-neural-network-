# %% [markdown]
# # 09 — Training
#
# Companion to section 09 of the site.
#
# Everything is in place. Training is a loop: shuffle, split into mini-batches,
# and for each batch run forward, compute the loss, run backward, update.
#
# This notebook builds that loop, then uses it to measure the two decisions that
# matter most — the learning rate and the batch size.

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
from dnn.mlp import MLP, binary_cross_entropy

np.set_printoptions(precision=4, suppress=True)

# %% [markdown]
# From here on, `MLP` from `dnn/mlp.py` provides the forward and backward passes
# you wrote in notebooks 05 and 08. `net.step(X, y, lr)` does one update on one
# batch and returns that batch's loss — that is the only method the loop below
# needs.

# %%
net = MLP([2, 8, 8, 1], activation='tanh', seed=0)
print(net)

# %% [markdown]
# ## 1. Mini-batches
#
# Three requirements, and each one matters:
#
# * **Shuffle every epoch.** Without it the batches are identical each pass, so
#   the sequence of updates is periodic and the model can fit the batch
#   boundaries rather than the data.
# * **Keep `X` and `y` together.** Shuffle an index array and use it for both.
# * **Do not drop the last short batch.** It is real data.

# %%
def make_batches(X, y, batch_size, rng):
    """Yield (X_batch, y_batch) pairs covering the whole dataset, shuffled."""
    ### PROMPT: Permute the indices, then slice them into chunks of batch_size.
    ### HINT: rng.permutation(len(X)) gives a shuffled index array.
    ### BEGIN SOLUTION
    order = rng.permutation(len(X))
    for start in range(0, len(order), batch_size):
        batch = order[start:start + batch_size]
        yield X[batch], y[batch]
    ### END SOLUTION


check.batches(make_batches)

# %%
Xd, yd = data.make_dataset('circles', n=10, noise=0.05, seed=0)
for i, (bx, by) in enumerate(make_batches(Xd, yd, 4, np.random.default_rng(0))):
    print(f'batch {i}: {len(bx)} examples, labels {by}')

# %% [markdown]
# ## 2. The training loop
#
# One epoch is one pass over every batch. After each epoch, record the metrics —
# measured on the whole dataset, with dropout disabled, so the numbers are
# comparable across epochs.

# %%
def train(net, X, y, learning_rate=0.2, batch_size=16, epochs=100, seed=0):
    """Train by mini-batch gradient descent. Return a history dict."""
    rng = np.random.default_rng(seed)
    history = {'train_loss': [], 'train_acc': []}
    ### PROMPT: For each epoch, step over every batch, then record the metrics.
    ### HINT: net.step(bx, by, learning_rate) performs one update.
    ### HINT: net.predict(X) gives probabilities; net.predict_classes(X) gives 0/1.
    ### BEGIN SOLUTION
    for _ in range(epochs):
        for bx, by in make_batches(X, y, batch_size, rng):
            net.step(bx, by, learning_rate)
        history['train_loss'].append(binary_cross_entropy(net.predict(X), y))
        history['train_acc'].append(float(np.mean(net.predict_classes(X) == y)))
    ### END SOLUTION
    return history


X, y = data.make_dataset('circles', n=200, noise=0.08, seed=0)
net = MLP([2, 8, 8, 1], activation='tanh', seed=0)
history = train(net, X, y, learning_rate=0.3, batch_size=16, epochs=200)

check.training(history, min_accuracy=0.9)

# %%
fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
plotting.plot_decision_boundary(net.predict, X, y, ax=axes[0], title='after training')
axes[1].plot(history['train_loss'], color='#3b82f6')
axes[1].set_xlabel('epoch'); axes[1].set_ylabel('loss'); axes[1].grid(alpha=0.25)
axes[1].set_title('loss', fontsize=10)
axes[2].plot(history['train_acc'], color='#16a34a')
axes[2].set_xlabel('epoch'); axes[2].set_ylabel('accuracy'); axes[2].grid(alpha=0.25)
axes[2].set_ylim(0.4, 1.02); axes[2].set_title('accuracy', fontsize=10)
plt.tight_layout()
plt.show()

print(f'{len(X)} examples, batch size 16 -> {int(np.ceil(len(X) / 16))} updates per epoch')
print(f'200 epochs -> {200 * int(np.ceil(len(X) / 16)):,} parameter updates in total')

# %% [markdown]
# ## 3. Watching the boundary form
#
# Train in stages and plot as you go. The network starts with an arbitrary split
# of the plane and bends it into shape.

# %%
net = MLP([2, 8, 8, 1], activation='tanh', seed=0)
checkpoints = [0, 5, 20, 60, 200]

fig, axes = plt.subplots(1, len(checkpoints), figsize=(15, 3.2))
done = 0
for ax, target in zip(axes, checkpoints):
    if target > done:
        train(net, X, y, learning_rate=0.3, batch_size=16, epochs=target - done, seed=done)
        done = target
    acc = float(np.mean(net.predict_classes(X) == y))
    plotting.plot_decision_boundary(net.predict, X, y, ax=ax,
                                    title=f'epoch {target} — {acc:.0%}')
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 4. The learning rate
#
# Section 07 showed that too large a step increases the loss. Here is what that
# looks like on a real network.

# %%
rates = [0.01, 0.1, 0.5, 2.0, 10.0]
fig, ax = plt.subplots(figsize=(6.4, 3.8))
for rate in rates:
    net = MLP([2, 8, 8, 1], activation='tanh', seed=0)
    h = train(net, X, y, learning_rate=rate, batch_size=16, epochs=120)
    final = h['train_loss'][-1]
    ax.plot(h['train_loss'], label=f'$\\eta$ = {rate}  (final {final:.3f})')
    print(f'eta = {rate:>5}: loss {h["train_loss"][0]:.4f} -> {final:.4f}, '
          f'accuracy {h["train_acc"][-1]:.3f}')
ax.set_xlabel('epoch'); ax.set_ylabel('loss')
ax.set_ylim(0, 1.2); ax.legend(fontsize=8); ax.grid(alpha=0.25)
ax.set_title('the same network at five learning rates', fontsize=10)
plt.tight_layout()
plt.show()

# %% [markdown]
# Too small and it barely moves in the budget available. Too large and the loss
# jumps around or stops improving. There is a broad middle where it works, which
# is why a coarse search over powers of ten is usually enough to find a
# workable value.

# %% [markdown]
# ## 5. Batch size
#
# A mini-batch gradient is an unbiased estimate of the full gradient, and its
# noise falls as $1/\sqrt{B}$. Measure that directly: compute the full-dataset
# gradient, then many batch gradients, and see how far they scatter.

# %%
net = MLP([2, 8, 8, 1], activation='tanh', seed=1)
full = net.gradients(X, y)
full_flat = np.concatenate([g['dW'].ravel() for g in full])

rng = np.random.default_rng(0)
print(f'{"batch size":>11} {"mean distance from the full gradient":>38} {"predicted":>11}')
baseline = None
for B in (1, 4, 16, 64, 200):
    distances = []
    for _ in range(60):
        idx = rng.choice(len(X), size=B, replace=False)
        g = net.gradients(X[idx], y[idx])
        flat = np.concatenate([gi['dW'].ravel() for gi in g])
        distances.append(np.linalg.norm(flat - full_flat))
    mean_distance = float(np.mean(distances))
    if baseline is None:
        baseline = mean_distance
    print(f'{B:>11} {mean_distance:>38.6f} {baseline / np.sqrt(B):>11.6f}')

print()
print('The measured scatter tracks 1/sqrt(B) closely. Going from B=1 to B=4')
print('halves the noise at four times the cost; from B=64 to B=256 halves it')
print('again at four times the cost, but by then the noise is already small.')
print('That diminishing return is why batch sizes sit in the tens to hundreds.')

# %%
fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
for B, colour in ((1, '#e0761f'), (64, '#3b82f6')):
    net = MLP([2, 8, 8, 1], activation='tanh', seed=0)
    h = train(net, X, y, learning_rate=0.3, batch_size=B, epochs=120)
    axes[0].plot(h['train_loss'], color=colour, lw=1.2, label=f'batch {B}')
    updates = int(np.ceil(len(X) / B)) * 120
    print(f'batch {B:>3}: {updates:>6,} updates, final loss {h["train_loss"][-1]:.4f}, '
          f'accuracy {h["train_acc"][-1]:.3f}')
axes[0].set_xlabel('epoch'); axes[0].set_ylabel('loss')
axes[0].legend(fontsize=8); axes[0].grid(alpha=0.25)
axes[0].set_title('batch size 1 is noisy but covers more ground per epoch', fontsize=9)

# Compensating with the linear scaling rule.
for B, lr, colour in ((16, 0.3, '#3b82f6'), (128, 0.3, '#94a3b8'), (128, 2.4, '#16a34a')):
    net = MLP([2, 8, 8, 1], activation='tanh', seed=0)
    h = train(net, X, y, learning_rate=lr, batch_size=B, epochs=120)
    axes[1].plot(h['train_loss'], color=colour, lw=1.4, label=f'B={B}, $\\eta$={lr}')
axes[1].set_xlabel('epoch'); axes[1].set_ylabel('loss')
axes[1].legend(fontsize=8); axes[1].grid(alpha=0.25)
axes[1].set_title('a bigger batch needs a bigger step', fontsize=9)
plt.tight_layout()
plt.show()

print()
print('Eight times the batch with the same learning rate trains far slower.')
print('Multiply the learning rate by eight as well and most of the gap closes.')
print('That is the linear scaling rule: keep eta / B roughly fixed.')

# %% [markdown]
# ## 6. Reading a training curve
#
# | what you see | most likely cause |
# |---|---|
# | loss rises or swings wildly | learning rate above the stability limit |
# | loss falls then plateaus high | model too small, or dead units |
# | loss barely moves at all | learning rate far too small, or vanishing gradients |
# | loss becomes `nan` | overflow — learning rate far too large, or `log(0)` |
# | jagged but descending | normal for a small batch size |
#
# A good first move for any of the top three: divide the learning rate by ten
# and see which symptom changes. That separates an optimisation problem from a
# capacity problem in one experiment.

# %%
for label, kwargs, sizes in (
    ('healthy',            dict(learning_rate=0.3,  batch_size=16), [2, 8, 8, 1]),
    ('lr far too large',   dict(learning_rate=20.0, batch_size=16), [2, 8, 8, 1]),
    ('lr far too small',   dict(learning_rate=1e-4, batch_size=16), [2, 8, 8, 1]),
    ('model too small',    dict(learning_rate=0.3,  batch_size=16), [2, 1]),
):
    net = MLP(sizes, activation='tanh', seed=0)
    h = train(net, X, y, epochs=120, **kwargs)
    losses = h['train_loss']
    print(f'{label:<18} loss {losses[0]:.4f} -> {losses[-1]:.4f}   '
          f'accuracy {h["train_acc"][-1]:.3f}')

# %% [markdown]
# ## Try this
#
# * Switch the dataset to `'spiral'` and see how many epochs and units it needs.
# * Remove the shuffle from `make_batches` — pass `np.arange(len(X))` instead of
#   a permutation — and compare the curves. On this dataset the labels alternate,
#   so the damage is small; sort by label first and it is severe.
#
# **Next:** `10-overfitting.ipynb` — what happens when the network fits the
# training data *too* well.
