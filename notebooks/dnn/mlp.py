"""
A complete reference multilayer perceptron.

This is the finished version of what notebooks 01 to 08 build up. Notebooks 09
onwards import it so that they can concentrate on training, regularisation and
depth rather than re-deriving gradients every time.

It is deliberately plain numpy, with the same conventions used throughout:

* ``X`` has shape ``(m, n_in)`` — one row per example
* ``W`` has shape ``(n_in, n_out)`` and ``b`` has shape ``(n_out,)``
* a layer computes ``Z = A_prev @ W + b`` then ``A = f(Z)``
* the last layer always uses a sigmoid, so the output reads as a probability
* the loss is mean binary cross-entropy

If you are working through notebook 08, write your own version first.
"""

from __future__ import annotations

import numpy as np

EPS = 1e-12


def sigmoid(z):
    """Logistic function, written so that large |z| does not overflow."""
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z)
    positive = z >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-z[positive]))
    exp_z = np.exp(z[~positive])
    out[~positive] = exp_z / (1.0 + exp_z)
    return out


ACTIVATIONS = {
    'relu': (lambda z: np.maximum(0.0, z), lambda z: np.where(z > 0, 1.0, 0.0)),
    'leaky_relu': (lambda z: np.where(z > 0, z, 0.1 * z),
                   lambda z: np.where(z > 0, 1.0, 0.1)),
    'tanh': (np.tanh, lambda z: 1 - np.tanh(z) ** 2),
    'sigmoid': (sigmoid, lambda z: sigmoid(z) * (1 - sigmoid(z))),
}


def binary_cross_entropy(y_hat, y):
    """Mean binary cross-entropy over the examples."""
    p = np.clip(np.asarray(y_hat).ravel(), EPS, 1 - EPS)
    y = np.asarray(y).ravel()
    return float(np.mean(-(y * np.log(p) + (1 - y) * np.log(1 - p))))


class MLP:
    """
    A dense feed-forward network trained by mini-batch gradient descent.

    Parameters
    ----------
    sizes
        Layer widths, input first. ``[2, 8, 8, 1]`` is two inputs, two hidden
        layers of eight units, and one output.
    activation
        Hidden-layer activation: one of ``ACTIVATIONS``. The output layer is
        always a sigmoid.
    seed
        Seed for the weight initialisation and the mini-batch shuffling.
    l2
        Coefficient of the ``(lambda / 2) * sum(W ** 2)`` penalty. Biases are
        not regularised.
    dropout
        Probability of dropping each hidden unit during training. Inverted
        dropout: survivors are divided by ``1 - dropout``.
    """

    def __init__(self, sizes, activation='relu', seed=0, l2=0.0, dropout=0.0):
        if activation not in ACTIVATIONS:
            raise ValueError(f'unknown activation {activation!r}')
        self.sizes = list(sizes)
        self.activation = activation
        self.l2 = float(l2)
        self.dropout = float(dropout)
        self.rng = np.random.default_rng(seed)
        self.params = self._initialise(seed)

    # -- setup --------------------------------------------------------------

    def _initialise(self, seed):
        rng = np.random.default_rng(seed)
        params = []
        for index, (n_in, n_out) in enumerate(zip(self.sizes, self.sizes[1:])):
            is_last = index == len(self.sizes) - 2
            # He for ReLU-family hidden layers, Glorot for saturating ones.
            if is_last or self.activation in ('tanh', 'sigmoid'):
                scale = np.sqrt(2.0 / (n_in + n_out))
            else:
                scale = np.sqrt(2.0 / n_in)
            params.append({'W': rng.normal(scale=scale, size=(n_in, n_out)),
                           'b': np.zeros(n_out)})
        return params

    @property
    def n_parameters(self) -> int:
        return sum(layer['W'].size + layer['b'].size for layer in self.params)

    # -- forward ------------------------------------------------------------

    def forward(self, X, training=False):
        """Return ``(output, cache)``. The cache holds what the backward pass needs."""
        f = ACTIVATIONS[self.activation][0]
        cache = []
        A = np.asarray(X, dtype=float)
        last = len(self.params) - 1
        for index, layer in enumerate(self.params):
            Z = A @ layer['W'] + layer['b']
            A_next = sigmoid(Z) if index == last else f(Z)
            mask = None
            if training and self.dropout > 0 and index != last:
                mask = (self.rng.random(A_next.shape) >= self.dropout) / (1 - self.dropout)
                A_next = A_next * mask
            cache.append({'a_prev': A, 'z': Z, 'a': A_next, 'mask': mask})
            A = A_next
        return A, cache

    def predict(self, X):
        """Output probabilities, with dropout disabled."""
        return self.forward(X, training=False)[0].ravel()

    def predict_classes(self, X):
        return (self.predict(X) >= 0.5).astype(float)

    # -- loss and gradients -------------------------------------------------

    def loss(self, X, y):
        """Mean binary cross-entropy plus the L2 penalty."""
        return binary_cross_entropy(self.predict(X), y) + self.l2_penalty()

    def l2_penalty(self):
        if self.l2 == 0.0:
            return 0.0
        return 0.5 * self.l2 * sum(float(np.sum(layer['W'] ** 2)) for layer in self.params)

    def backward(self, cache, y):
        """Gradients of the loss with respect to every parameter."""
        df = ACTIVATIONS[self.activation][1]
        y = np.asarray(y, dtype=float).reshape(-1, 1)
        grads = [None] * len(self.params)
        delta = (cache[-1]['a'] - y) / len(y)

        for l in reversed(range(len(self.params))):
            grads[l] = {
                'dW': cache[l]['a_prev'].T @ delta + self.l2 * self.params[l]['W'],
                'db': delta.sum(axis=0),
            }
            if l > 0:
                delta = delta @ self.params[l]['W'].T
                if cache[l - 1]['mask'] is not None:
                    delta = delta * cache[l - 1]['mask']
                delta = delta * df(cache[l - 1]['z'])
        return grads

    def gradients(self, X, y, training=False):
        """Convenience: forward then backward in one call."""
        _, cache = self.forward(X, training=training)
        return self.backward(cache, y)

    # -- training -----------------------------------------------------------

    def step(self, X, y, learning_rate):
        """One gradient-descent update on this batch. Returns the batch loss."""
        out, cache = self.forward(X, training=True)
        grads = self.backward(cache, y)
        for layer, grad in zip(self.params, grads):
            layer['W'] -= learning_rate * grad['dW']
            layer['b'] -= learning_rate * grad['db']
        return binary_cross_entropy(out, y)

    def fit(self, X, y, learning_rate=0.2, batch_size=16, epochs=100,
            validation=None, verbose=False):
        """
        Train by mini-batch gradient descent.

        Returns a history dict with ``train_loss`` and ``train_acc`` per epoch,
        plus ``val_loss`` and ``val_acc`` when ``validation=(X_val, y_val)``.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        history = {'train_loss': [], 'train_acc': []}
        if validation is not None:
            history['val_loss'] = []
            history['val_acc'] = []

        for epoch in range(epochs):
            order = self.rng.permutation(len(X))
            for start in range(0, len(order), max(1, batch_size)):
                batch = order[start:start + max(1, batch_size)]
                self.step(X[batch], y[batch], learning_rate)

            history['train_loss'].append(self.loss(X, y))
            history['train_acc'].append(float(np.mean(self.predict_classes(X) == y)))
            if validation is not None:
                Xv, yv = validation
                yv = np.asarray(yv, dtype=float).ravel()
                history['val_loss'].append(binary_cross_entropy(self.predict(Xv), yv))
                history['val_acc'].append(float(np.mean(self.predict_classes(Xv) == yv)))
            if verbose and (epoch + 1) % max(1, epochs // 10) == 0:
                extra = ''
                if validation is not None:
                    extra = f"  val {history['val_loss'][-1]:.4f}"
                print(f"epoch {epoch + 1:>5}  loss {history['train_loss'][-1]:.4f}"
                      f"  acc {history['train_acc'][-1]:.3f}{extra}")
        return history

    # -- misc ---------------------------------------------------------------

    def snapshot(self):
        """A deep copy of the current parameters, for early stopping."""
        return [{'W': layer['W'].copy(), 'b': layer['b'].copy()} for layer in self.params]

    def restore(self, snapshot):
        self.params = [{'W': layer['W'].copy(), 'b': layer['b'].copy()} for layer in snapshot]

    def weight_norm(self):
        return float(np.sqrt(sum(float(np.sum(layer['W'] ** 2)) for layer in self.params)))

    def __repr__(self):
        return (f'MLP({self.sizes}, activation={self.activation!r}, '
                f'l2={self.l2}, dropout={self.dropout}) '
                f'- {self.n_parameters} parameters')
