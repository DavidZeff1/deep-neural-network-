# %% [markdown]
# # 10 — Overfitting and regularisation
#
# Companion to section 10 of the site.
#
# A network with enough parameters can drive the training loss to nearly zero by
# memorising the training points, noise included. The only way to see that
# happening is to measure the loss on data the updates never touched.
#
# You will produce the overfitting curve, then implement the three standard
# responses and measure what each one does.

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
# ## 1. A dataset noisy enough to overfit
#
# Few points and a lot of noise. Split it: the training half contributes
# gradients, the validation half is only ever measured.

# %%
X_all, y_all = data.make_dataset('moons', n=160, noise=0.28, seed=3)
X_train, y_train, X_val, y_val = data.train_val_split(X_all, y_all, 0.5, seed=1)

print(f'{len(X_train)} training points, {len(X_val)} validation points')

fig, ax = plt.subplots(figsize=(4.6, 4.6))
plotting.plot_dataset(X_train, y_train, ax=ax)
ax.scatter(X_val[:, 0], X_val[:, 1], facecolors='none',
           edgecolors=np.where(y_val >= 0.5, '#3b82f6', '#e0761f'), s=42, linewidths=1.4)
ax.set_title('filled = training, hollow = validation', fontsize=10)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 2. Watching the gap open
#
# Train a large network for a long time and record both losses every epoch.

# %%
net = MLP([2, 40, 40, 40, 1], activation='relu', seed=7)
print(net)
print(f'{net.n_parameters} parameters fitted to {len(X_train)} training points')

history = net.fit(X_train, y_train, learning_rate=0.05, batch_size=8, epochs=600,
                  validation=(X_val, y_val))

best_epoch = int(np.argmin(history['val_loss']))
print()
print(f'lowest validation loss : {history["val_loss"][best_epoch]:.4f} at epoch {best_epoch}')
print(f'final validation loss  : {history["val_loss"][-1]:.4f}')
print(f'final training loss    : {history["train_loss"][-1]:.4f}')

# %%
fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
axes[0].plot(history['train_loss'], color='#3b82f6', label='training')
axes[0].plot(history['val_loss'], color='#e0761f', ls='--', label='validation')
axes[0].axvline(best_epoch, color='#94a3b8', ls=':', lw=1.4)
axes[0].annotate('best validation', xy=(best_epoch, history['val_loss'][best_epoch]),
                 xytext=(best_epoch + 60, history['val_loss'][best_epoch] + 0.25), fontsize=8)
axes[0].set_xlabel('epoch'); axes[0].set_ylabel('loss')
axes[0].legend(fontsize=8); axes[0].grid(alpha=0.25)
axes[0].set_title('the training loss keeps falling; the validation loss turns', fontsize=9)

plotting.plot_decision_boundary(net.predict, X_train, y_train, ax=axes[1],
                                validation=(X_val, y_val), title='after 600 epochs')
plt.tight_layout()
plt.show()

# %% [markdown]
# That is the whole phenomenon. The training loss goes on falling because the
# network is still finding ways to fit individual points. The validation loss
# turned upward because those ways do not generalise — look at the pockets the
# boundary has grown around isolated training points.

# %% [markdown]
# ## 3. Where the variance comes from
#
# "Overfitting" has a precise decomposition. Train the same architecture on
# different samples of data and the fitted models differ. That spread is the
# **variance** term, and it is what regularisation reduces.

# %%
fig, axes = plt.subplots(1, 4, figsize=(14, 3.4))
for ax, seed in zip(axes, (0, 1, 2, 3)):
    Xs, ys = data.make_dataset('moons', n=80, noise=0.28, seed=20 + seed)
    m = MLP([2, 40, 40, 1], activation='relu', seed=7)
    m.fit(Xs, ys, learning_rate=0.05, batch_size=8, epochs=400)
    plotting.plot_decision_boundary(m.predict, Xs, ys, ax=ax, title=f'sample {seed}')
plt.tight_layout()
plt.show()

print('Same architecture, same initialisation, same hyperparameters.')
print('Four different samples of the same underlying data, four different boundaries.')
print('That disagreement is the variance. A simpler model would produce four')
print('similar boundaries — lower variance, but further from the truth on average,')
print('which is the bias you pay for it.')

# %% [markdown]
# ## 4. L2 regularisation
#
# Add the squared size of the weights to the objective:
#
# $$J = \frac{1}{m}\sum_i L_i + \frac{\lambda}{2}\sum_{l}\lVert W^{(l)} \rVert^2$$
#
# Differentiating the extra term gives $\lambda W$, so it simply adds to the
# gradient. Biases are left out: they shift the function without making it
# wigglier.

# %%
def add_l2_gradient(data_gradient, W, lam):
    """Add the L2 penalty's contribution to one layer's weight gradient."""
    ### PROMPT: The penalty (lambda/2) * sum(W**2) differentiates to lambda * W.
    ### BEGIN SOLUTION
    return data_gradient + lam * W
    ### END SOLUTION


check.l2_gradient(add_l2_gradient)

# %%
print('Rewriting the update shows what it does:')
print()
print('   W <- W - eta * (dL/dW + lambda * W)')
print('   W <- (1 - eta * lambda) * W  -  eta * dL/dW')
print()
for eta, lam in ((0.05, 0.01), (0.05, 0.1), (0.2, 0.05)):
    print(f'   eta={eta}, lambda={lam}: every weight is multiplied by '
          f'{1 - eta * lam:.4f} each step before the data term is applied')
print()
print('Hence the other name for it: weight decay.')

# %%
results = {}
for lam in (0.0, 0.001, 0.01, 0.05, 0.3):
    m = MLP([2, 40, 40, 40, 1], activation='relu', seed=7, l2=lam)
    h = m.fit(X_train, y_train, learning_rate=0.05, batch_size=8, epochs=600,
              validation=(X_val, y_val))
    results[lam] = {
        'final_train_loss': h['train_loss'][-1],
        'final_val_loss': h['val_loss'][-1],
        'best_val_loss': float(np.min(h['val_loss'])),
        'final_val_acc': h['val_acc'][-1],
        'final_weight_norm': m.weight_norm(),
        'net': m,
        'history': h,
    }
    r = results[lam]
    print(f'lambda = {lam:<6} train {r["final_train_loss"]:.4f}   '
          f'val {r["final_val_loss"]:.4f}   '
          f'gap {r["final_val_loss"] - r["final_train_loss"]:>7.4f}   '
          f'val acc {r["final_val_acc"]:.3f}   ||W|| {r["final_weight_norm"]:>6.2f}')

check.regularisation_effect(results[0.0], results[0.01])

# %%
fig, axes = plt.subplots(1, 5, figsize=(16, 3.4))
for ax, (lam, r) in zip(axes, results.items()):
    plotting.plot_decision_boundary(r['net'].predict, X_train, y_train, ax=ax,
                                    validation=(X_val, y_val),
                                    title=f'$\\lambda$ = {lam}\nval acc {r["final_val_acc"]:.2f}')
plt.tight_layout()
plt.show()

print('Read the gap column. At lambda = 0 the network fits the training half')
print('almost perfectly and the validation half badly — a gap of about 0.78.')
print('At lambda = 0.01 the gap is essentially zero and the validation accuracy')
print('is the best of the five.')
print()
print('At lambda = 0.3 something worth seeing has happened: ||W|| is 0.00.')
print('The penalty overwhelmed the data term entirely, every weight decayed to')
print('zero, and the network now predicts the same value everywhere. Accuracy')
print('falls to chance. Too much regularisation is not a milder version of the')
print('right amount — it destroys the model.')

# %% [markdown]
# ## 5. Dropout
#
# During training, set each hidden unit to zero with probability $p$,
# independently for every example. Divide the survivors by $1 - p$ so the
# expected value entering the next layer is unchanged — that is **inverted
# dropout**, and it means evaluation needs no rescaling at all.

# %%
def dropout_forward(A, p, rng, training):
    """
    Apply inverted dropout to activations A.

    Returns (A_out, mask). At evaluation time nothing is dropped.
    """
    ### PROMPT: Build a 0/1 mask, scale it by 1/(1-p), and multiply.
    ### HINT: rng.random(A.shape) >= p keeps each unit with probability 1 - p.
    ### BEGIN SOLUTION
    if not training or p <= 0:
        return A, np.ones_like(A)
    mask = (rng.random(A.shape) >= p) / (1.0 - p)
    return A * mask, mask
    ### END SOLUTION


check.dropout(dropout_forward)

# %%
rng = np.random.default_rng(0)
A = np.ones((6, 8))
out, mask = dropout_forward(A, 0.5, rng, training=True)
print('activations entering the layer (all 1.0), after dropout at p = 0.5:')
print(out)
print()
print(f'{np.mean(out == 0):.1%} dropped, survivors scaled to {out[out != 0][0]:.1f}, '
      f'mean still {out.mean():.3f}')

# %%
for p in (0.0, 0.1, 0.3, 0.5):
    m = MLP([2, 40, 40, 40, 1], activation='relu', seed=7, dropout=p)
    h = m.fit(X_train, y_train, learning_rate=0.05, batch_size=8, epochs=600,
              validation=(X_val, y_val))
    print(f'dropout = {p:<5} train {h["train_loss"][-1]:.4f}   '
          f'best val {np.min(h["val_loss"]):.4f}   val acc {h["val_acc"][-1]:.3f}')

print()
print('Note the training loss rises with p. That is expected, not a failure:')
print('it is measured on the full network while training happened on thinned')
print('ones. Dropout removes capacity, so it helps a model that is overfitting')
print('and hurts one that is already too small.')

# %% [markdown]
# ## 6. Early stopping
#
# The cheapest of the three. Track the validation loss and keep the parameters
# from the epoch where it was lowest.

# %%
def best_epoch_index(val_losses):
    """Index of the lowest validation loss. Ties keep the earliest epoch."""
    ### PROMPT: One numpy call does this.
    ### BEGIN SOLUTION
    return int(np.argmin(val_losses))
    ### END SOLUTION


check.early_stopping(best_epoch_index)

# %%
# Train again, snapshotting whenever the validation loss improves.
net = MLP([2, 40, 40, 40, 1], activation='relu', seed=7)
best_loss, best_state, best_at = np.inf, None, 0
curve = {'train_loss': [], 'val_loss': []}

for epoch in range(600):
    net.fit(X_train, y_train, learning_rate=0.05, batch_size=8, epochs=1)
    from dnn.mlp import binary_cross_entropy
    val_loss = binary_cross_entropy(net.predict(X_val), y_val)
    curve['train_loss'].append(binary_cross_entropy(net.predict(X_train), y_train))
    curve['val_loss'].append(val_loss)
    if val_loss < best_loss:
        best_loss, best_state, best_at = val_loss, net.snapshot(), epoch

final_acc = float(np.mean(net.predict_classes(X_val) == y_val))
final_boundary = net.predict

net.restore(best_state)
restored_acc = float(np.mean(net.predict_classes(X_val) == y_val))

print(f'best validation loss {best_loss:.4f} at epoch {best_at}')
print(f'validation accuracy at epoch 600 : {final_acc:.3f}')
print(f'validation accuracy when restored: {restored_acc:.3f}')

# %%
fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8))
axes[0].plot(curve['train_loss'], color='#3b82f6', label='training')
axes[0].plot(curve['val_loss'], color='#e0761f', ls='--', label='validation')
axes[0].axvline(best_at, color='#94a3b8', ls=':', lw=1.4)
axes[0].set_xlabel('epoch'); axes[0].set_ylabel('loss')
axes[0].legend(fontsize=8); axes[0].grid(alpha=0.25)
axes[0].set_title(f'stop at epoch {best_at}', fontsize=9)
plotting.plot_decision_boundary(final_boundary, X_train, y_train, ax=axes[1],
                                validation=(X_val, y_val), title='epoch 600')
plotting.plot_decision_boundary(net.predict, X_train, y_train, ax=axes[2],
                                validation=(X_val, y_val), title=f'restored to epoch {best_at}')
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 7. Train, validation, test — why three
#
# The validation set was used to *choose* things: the value of $\lambda$, the
# stopping epoch. Selection is a form of fitting. If you try 50 configurations
# and keep the best, that minimum is optimistic — you have partly fitted the
# validation set through the choice.
#
# Measure the effect. Score 50 identical-quality models on a small validation
# set and see how much the best one beats their true average by luck alone.

# %%
rng = np.random.default_rng(0)
true_quality = 0.30
trials = 50
for val_size in (20, 80, 400):
    best_of_each = []
    for _ in range(400):
        noise = rng.normal(scale=1.0 / np.sqrt(val_size), size=trials)
        best_of_each.append(np.min(true_quality + noise))
    gap = true_quality - np.mean(best_of_each)
    print(f'validation set of {val_size:>4}: the best of {trials} models looks '
          f'{gap:.4f} better than it is')

print()
print('The smaller the validation set, the bigger the illusion. This is why a')
print('number you selected on cannot also be the number you report, and why a')
print('real experiment keeps a third split that is looked at exactly once.')

# %% [markdown]
# ## Try this
#
# * Set the dataset noise to 0.05 instead of 0.28 and re-run section 2. The gap
#   almost vanishes: with clean data there is little noise to memorise.
# * Compare the `best val` and `final val` columns at lambda = 0. The minimum
#   over epochs is far better than the end — taking that minimum is early
#   stopping, and it is already regularising even when you did not ask for it.
# * Combine `l2=0.01` with `dropout=0.2`. They stack, and the combination is
#   often better than either alone.
#
# **Next:** `11-depth.ipynb` — what extra layers actually buy.
