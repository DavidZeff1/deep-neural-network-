# %% [markdown]
# # 12 — Capstone: build the whole thing
#
# Companion to section 12 of the site.
#
# No new theory. This notebook asks you to assemble everything into one class,
# from an empty file, with nothing imported from `dnn.mlp`.
#
# The check is the strictest in the course. It will:
#
# 1. confirm `predict` returns probabilities of the right shape,
# 2. perturb **every** parameter and compare your gradients to finite
#    differences of your own loss,
# 3. train your network from scratch on `circles` and require better than 92%
#    accuracy.
#
# If that passes, you have written a working neural network library.

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
EPS = 1e-12

# %% [markdown]
# ## The specification
#
# ```
# NeuralNetwork(sizes, activation='relu', seed=0, l2=0.0)
#
#   .params            list of {'W': (n_in, n_out), 'b': (n_out,)}, one per layer
#   .forward(X)        -> (output, cache) with cache[i] = {'a_prev', 'z', 'a'}
#   .predict(X)        -> probabilities, shape (m,)
#   .loss(X, y)        -> mean binary cross-entropy, plus the L2 penalty
#   .gradients(X, y)   -> list of {'dW', 'db'}, same order as .params
#   .fit(X, y, learning_rate, batch_size, epochs) -> history dict
# ```
#
# Conventions, unchanged from the rest of the course: one example per row, `W`
# is `(n_in, n_out)`, the last layer is always a sigmoid, the loss is mean
# binary cross-entropy.
#
# Everything you need is in notebooks 01, 04, 05, 06, 08 and 09. Try to write it
# without looking first — and check `dnn/mlp.py` only once you are done.

# %%
def sigmoid(z):
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    e = np.exp(z[~pos])
    out[~pos] = e / (1.0 + e)
    return out


ACTIVATIONS = {
    'relu': (lambda z: np.maximum(0.0, z), lambda z: np.where(z > 0, 1.0, 0.0)),
    'tanh': (np.tanh, lambda z: 1 - np.tanh(z) ** 2),
    'sigmoid': (sigmoid, lambda z: sigmoid(z) * (1 - sigmoid(z))),
}


class NeuralNetwork:
    """A dense feed-forward network trained by mini-batch gradient descent."""

    def __init__(self, sizes, activation='relu', seed=0, l2=0.0):
        self.sizes = list(sizes)
        self.activation = activation
        self.l2 = float(l2)
        self.rng = np.random.default_rng(seed)
        ### PROMPT: Initialise params: random W (He scale), zero b, one dict per layer.
        ### HINT: Use a separate Generator seeded the same way, so fit() shuffling
        ### HINT: does not change the initial weights.
        ### BEGIN SOLUTION
        init_rng = np.random.default_rng(seed)
        self.params = [
            {'W': init_rng.normal(scale=np.sqrt(2.0 / n_in), size=(n_in, n_out)),
             'b': np.zeros(n_out)}
            for n_in, n_out in zip(self.sizes, self.sizes[1:])
        ]
        ### END SOLUTION

    # -- forward ------------------------------------------------------------

    def forward(self, X):
        """Return (output, cache). The last layer uses sigmoid."""
        f = ACTIVATIONS[self.activation][0]
        ### PROMPT: Chain the layers, recording a_prev, z and a for each.
        ### BEGIN SOLUTION
        cache, A = [], np.asarray(X, dtype=float)
        last = len(self.params) - 1
        for index, layer in enumerate(self.params):
            Z = A @ layer['W'] + layer['b']
            A = sigmoid(Z) if index == last else f(Z)
            cache.append({'a_prev': cache[-1]['a'] if cache else np.asarray(X, dtype=float),
                          'z': Z, 'a': A})
        return A, cache
        ### END SOLUTION

    def predict(self, X):
        return self.forward(X)[0].ravel()

    def predict_classes(self, X):
        return (self.predict(X) >= 0.5).astype(float)

    # -- loss ---------------------------------------------------------------

    def loss(self, X, y):
        """Mean binary cross-entropy plus the L2 penalty."""
        y = np.asarray(y, dtype=float).ravel()
        ### PROMPT: Clip the prediction, apply the BCE formula, add (l2/2)*sum(W**2).
        ### BEGIN SOLUTION
        p = np.clip(self.predict(X), EPS, 1 - EPS)
        data_loss = float(np.mean(-(y * np.log(p) + (1 - y) * np.log(1 - p))))
        penalty = 0.5 * self.l2 * sum(float(np.sum(l['W'] ** 2)) for l in self.params)
        return data_loss + penalty
        ### END SOLUTION

    # -- gradients ----------------------------------------------------------

    def gradients(self, X, y):
        """Return [{'dW', 'db'}, ...] for the current parameters."""
        df = ACTIVATIONS[self.activation][1]
        _, cache = self.forward(X)
        y = np.asarray(y, dtype=float).reshape(-1, 1)
        ### PROMPT: Output delta, then walk backwards recording dW and db.
        ### HINT: Do not forget the l2 * W term in dW if you support regularisation.
        ### BEGIN SOLUTION
        grads = [None] * len(self.params)
        delta = (cache[-1]['a'] - y) / len(y)
        for l in reversed(range(len(self.params))):
            grads[l] = {
                'dW': cache[l]['a_prev'].T @ delta + self.l2 * self.params[l]['W'],
                'db': delta.sum(axis=0),
            }
            if l > 0:
                delta = (delta @ self.params[l]['W'].T) * df(cache[l - 1]['z'])
        return grads
        ### END SOLUTION

    # -- training -----------------------------------------------------------

    def fit(self, X, y, learning_rate=0.2, batch_size=16, epochs=100, validation=None):
        """Mini-batch gradient descent. Returns a history dict."""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        history = {'train_loss': [], 'train_acc': []}
        if validation is not None:
            history['val_loss'], history['val_acc'] = [], []
        ### PROMPT: Shuffle, step over the batches, record the metrics once per epoch.
        ### BEGIN SOLUTION
        for _ in range(epochs):
            order = self.rng.permutation(len(X))
            for start in range(0, len(order), max(1, batch_size)):
                batch = order[start:start + max(1, batch_size)]
                grads = self.gradients(X[batch], y[batch])
                for layer, grad in zip(self.params, grads):
                    layer['W'] -= learning_rate * grad['dW']
                    layer['b'] -= learning_rate * grad['db']
            history['train_loss'].append(self.loss(X, y))
            history['train_acc'].append(float(np.mean(self.predict_classes(X) == y)))
            if validation is not None:
                Xv, yv = validation
                yv = np.asarray(yv, dtype=float).ravel()
                history['val_loss'].append(self.loss(Xv, yv))
                history['val_acc'].append(float(np.mean(self.predict_classes(Xv) == yv)))
        ### END SOLUTION
        return history

    @property
    def n_parameters(self):
        return sum(l['W'].size + l['b'].size for l in self.params)

    def __repr__(self):
        return f'NeuralNetwork({self.sizes}, {self.activation!r}) - {self.n_parameters} parameters'


check.network_class(NeuralNetwork)

# %% [markdown]
# ## Use it
#
# Every dataset from the course, one network each.

# %%
fig, axes = plt.subplots(1, 5, figsize=(16, 3.6))
for ax, name in zip(axes, data.DATASETS):
    X, y = data.make_dataset(name, n=220, noise=0.07, seed=1)
    net = NeuralNetwork([2, 12, 12, 1], activation='tanh', seed=4)
    h = net.fit(X, y, learning_rate=0.3, batch_size=16, epochs=400)
    plotting.plot_decision_boundary(
        net.predict, X, y, ax=ax,
        title=f'{name}\nloss {h["train_loss"][-1]:.3f}, acc {h["train_acc"][-1]:.0%}')
    print(f'{name:<9} loss {h["train_loss"][-1]:.4f}   accuracy {h["train_acc"][-1]:.3f}')
plt.tight_layout()
plt.show()

# %% [markdown]
# ## The full picture on one problem
#
# Training and validation curves, the boundary, and early stopping — everything
# the site's playground shows, produced by your own code.

# %%
X_all, y_all = data.make_dataset('moons', n=300, noise=0.22, seed=5)
X_tr, y_tr, X_va, y_va = data.train_val_split(X_all, y_all, 0.75, seed=2)

net = NeuralNetwork([2, 16, 16, 1], activation='tanh', seed=1, l2=0.002)
history = net.fit(X_tr, y_tr, learning_rate=0.25, batch_size=16, epochs=500,
                  validation=(X_va, y_va))

best = int(np.argmin(history['val_loss']))
print(net)
print(f'best validation loss {history["val_loss"][best]:.4f} at epoch {best}')
print(f'final train acc {history["train_acc"][-1]:.3f}, val acc {history["val_acc"][-1]:.3f}')

fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8))
plotting.plot_decision_boundary(net.predict, X_tr, y_tr, ax=axes[0],
                                validation=(X_va, y_va), title='decision boundary')
axes[1].plot(history['train_loss'], color='#3b82f6', label='training')
axes[1].plot(history['val_loss'], color='#e0761f', ls='--', label='validation')
axes[1].axvline(best, color='#94a3b8', ls=':', lw=1.3)
axes[1].set_xlabel('epoch'); axes[1].set_ylabel('loss')
axes[1].legend(fontsize=8); axes[1].grid(alpha=0.25); axes[1].set_title('loss', fontsize=10)
axes[2].plot(history['train_acc'], color='#3b82f6', label='training')
axes[2].plot(history['val_acc'], color='#e0761f', ls='--', label='validation')
axes[2].set_xlabel('epoch'); axes[2].set_ylabel('accuracy')
axes[2].legend(fontsize=8); axes[2].grid(alpha=0.25); axes[2].set_title('accuracy', fontsize=10)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Probing a single point
#
# The site's playground lets you click a point and inspect every activation. The
# same thing, with your cache:

# %%
probe = np.array([[0.4, 0.3]])
out, cache = net.forward(probe)

print(f'input {probe[0]}')
for i, entry in enumerate(cache):
    kind = 'output' if i == len(cache) - 1 else f'hidden {i + 1}'
    print(f'\n{kind}')
    print(f'  z = {entry["z"][0]}')
    print(f'  a = {entry["a"][0]}')
print(f'\nprediction {out[0, 0]:.4f} -> class {int(out[0, 0] >= 0.5)}')

# %%
# The arithmetic behind one hidden unit, spelled out.
unit = 0
W, b = net.params[0]['W'], net.params[0]['b']
terms = ' + '.join(f'({W[i, unit]:+.3f})({probe[0, i]:+.3f})' for i in range(len(probe[0])))
z = float(cache[0]['z'][0, unit])
print(f'hidden unit {unit + 1}')
print(f'  z = {terms} + {b[unit]:+.3f}')
print(f'  z = {z:.6f}')
print(f'  a = tanh({z:.6f}) = {np.tanh(z):.6f}')
print(f'  matches the cache: {np.isclose(np.tanh(z), cache[0]["a"][0, unit])}')

# %% [markdown]
# ## Where to go next
#
# Your network is missing things that real libraries have, and each one is a
# small, self-contained addition you now have the background for:
#
# * **A better optimiser.** Momentum is six lines (notebook 07). Adam is about
#   fifteen: keep a running average of the gradient *and* of its square, and
#   divide one by the square root of the other.
# * **Multi-class output.** Replace the sigmoid with the softmax from notebook 04
#   and the loss with categorical cross-entropy. The output delta stays
#   `p - y` — that is the whole change.
# * **Dropout.** Notebook 10 has it. The only subtlety is applying the same mask
#   in the backward pass.
# * **Other layer types.** A convolutional layer is this same equation with
#   weights tied together and distant entries fixed at zero. Everything you
#   derived still applies.
#
# ## What you built
#
# | notebook | what it produced |
# |---|---|
# | 00 | the chain rule, verified against a measurement |
# | 01 | parameters from a list of layer sizes |
# | 02 | one neuron, and a proof it cannot do XOR |
# | 03 | why initialisation scale decides whether a deep net trains |
# | 04 | four activations, their derivatives, and a stable softmax |
# | 05 | the forward pass with its cache |
# | 06 | three losses and the sigmoid/cross-entropy cancellation |
# | 07 | gradient descent, its stability limit, momentum, conditioning |
# | 08 | backpropagation, gradient-checked parameter by parameter |
# | 09 | mini-batches, and what the learning rate and batch size do |
# | 10 | overfitting, L2, dropout, early stopping |
# | 11 | what depth buys and what it costs |
# | 12 | all of it, in one class |
#
# Nothing here was taken on trust. Every derivative was checked against a
# measurement, and every claim about training was run.
