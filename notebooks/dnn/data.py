"""
The same five two-dimensional datasets the website uses.

Every generator returns ``X`` with shape ``(n, 2)`` and ``y`` with shape
``(n,)`` holding 0.0 or 1.0. Points lie roughly inside the square
[-1, 1] x [-1, 1], so one set of axis limits works for all of them.
"""

from __future__ import annotations

import numpy as np

DATASETS = ('linear', 'xor', 'circles', 'moons', 'spiral')


def make_dataset(name: str, n: int = 200, noise: float = 0.08, seed: int = 0):
    """Return ``(X, y)`` for one of the datasets in :data:`DATASETS`."""
    if name not in DATASETS:
        raise ValueError(f'unknown dataset {name!r}; choose from {DATASETS}')
    rng = np.random.default_rng(seed)
    label = np.arange(n) % 2

    if name == 'linear':
        along = rng.uniform(-0.75, 0.75, n)
        offset = np.where(label == 1, 0.45, -0.45)
        x1 = offset * 0.85 + along * 0.7
        x2 = offset * 0.85 - along * 0.7
        X = np.stack([x1, x2], axis=1)

    elif name == 'xor':
        sx = np.where(np.arange(n) % 2 == 0, 1.0, -1.0)
        sy = np.where(np.arange(n) % 4 < 2, 1.0, -1.0)
        X = np.stack([sx * rng.uniform(0.12, 0.9, n), sy * rng.uniform(0.12, 0.9, n)], axis=1)
        label = (sx * sy > 0).astype(int)

    elif name == 'circles':
        radius = np.where(label == 1, rng.uniform(0, 0.36, n), rng.uniform(0.62, 0.9, n))
        angle = rng.uniform(0, 2 * np.pi, n)
        X = np.stack([radius * np.cos(angle), radius * np.sin(angle)], axis=1)

    elif name == 'moons':
        angle = rng.uniform(0, np.pi, n)
        x1 = np.where(label == 1, np.cos(angle) * 0.8 - 0.35, np.cos(angle) * 0.8 + 0.35)
        x2 = np.where(label == 1, np.sin(angle) * 0.8 - 0.25, -np.sin(angle) * 0.8 + 0.25)
        X = np.stack([x1, x2], axis=1)

    else:  # spiral
        t = np.arange(n) / n * 2.6 + 0.35
        angle = t * 2.2 + label * np.pi
        radius = t * 0.34
        X = np.stack([radius * np.cos(angle), radius * np.sin(angle)], axis=1)

    X = X + rng.normal(0.0, noise, X.shape)
    order = rng.permutation(n)
    return X[order], label[order].astype(float)


def train_val_split(X, y, train_fraction: float = 0.75, seed: int = 7):
    """Split into training and validation sets. Returns ``(Xtr, ytr, Xva, yva)``."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(X))
    cut = max(1, int(round(len(X) * train_fraction)))
    tr, va = order[:cut], order[cut:]
    return X[tr], y[tr], X[va], y[va]
