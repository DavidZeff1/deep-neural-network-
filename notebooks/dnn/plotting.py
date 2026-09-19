"""
Plot helpers shared by the notebooks.

These deliberately mirror the visualisations on the website, so that a figure
produced here and the corresponding panel there show the same thing.
"""

from __future__ import annotations

import numpy as np

CLASS_1 = '#3b82f6'
CLASS_0 = '#e0761f'


def _axes(ax):
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(4.4, 4.4))
    return ax


def plot_dataset(X, y, ax=None, title=None, alpha=1.0):
    """Scatter a 2-D dataset, colouring class 1 blue and class 0 orange."""
    ax = _axes(ax)
    mask = y >= 0.5
    ax.scatter(X[mask, 0], X[mask, 1], c=CLASS_1, s=18, edgecolors='white',
               linewidths=0.5, label='class 1', alpha=alpha)
    ax.scatter(X[~mask, 0], X[~mask, 1], c=CLASS_0, s=18, edgecolors='white',
               linewidths=0.5, label='class 0', alpha=alpha)
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_aspect('equal')
    if title:
        ax.set_title(title, fontsize=10)
    return ax


def plot_decision_boundary(predict, X=None, y=None, ax=None, title=None,
                           limits=(-1.3, 1.3), resolution=120, validation=None):
    """
    Shade the plane by ``predict(grid)`` and draw the p = 0.5 contour.

    ``predict`` takes an array of shape (m, 2) and returns m probabilities.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    ax = _axes(ax)
    lo, hi = limits
    grid_x, grid_y = np.meshgrid(np.linspace(lo, hi, resolution),
                                 np.linspace(lo, hi, resolution))
    points = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1)
    probs = np.asarray(predict(points)).reshape(grid_x.shape)

    cmap = LinearSegmentedColormap.from_list('dnn', [CLASS_0, '#f2f4f7', CLASS_1])
    ax.contourf(grid_x, grid_y, probs, levels=32, cmap=cmap, vmin=0, vmax=1, alpha=0.75)
    ax.contour(grid_x, grid_y, probs, levels=[0.5], colors='#14171f', linewidths=1.2)

    if X is not None and y is not None:
        plot_dataset(X, y, ax=ax)
    if validation is not None:
        Xv, yv = validation
        mask = yv >= 0.5
        ax.scatter(Xv[mask, 0], Xv[mask, 1], facecolors='none', edgecolors=CLASS_1, s=28, linewidths=1.3)
        ax.scatter(Xv[~mask, 0], Xv[~mask, 1], facecolors='none', edgecolors=CLASS_0, s=28, linewidths=1.3)

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect('equal')
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    if title:
        ax.set_title(title, fontsize=10)
    plt.tight_layout()
    return ax


def plot_history(history, keys=('train_loss', 'val_loss'), ax=None, title=None, ylabel='loss'):
    """Plot one or more metric series recorded during training."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(5.2, 3.2))
    styles = {'train_loss': ('-', CLASS_1), 'val_loss': ('--', CLASS_0),
              'train_acc': ('-', CLASS_1), 'val_acc': ('--', CLASS_0)}
    for key in keys:
        if key not in history:
            continue
        style, colour = styles.get(key, ('-', '#64748b'))
        ax.plot(history[key], style, color=colour, label=key.replace('_', ' '))
    ax.set_xlabel('epoch')
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    if title:
        ax.set_title(title, fontsize=10)
    plt.tight_layout()
    return ax


def plot_function(f, df=None, limits=(-4, 4), ax=None, title=None, marker_x=None):
    """Plot a scalar function and, optionally, its derivative."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(5.2, 3.4))
    xs = np.linspace(limits[0], limits[1], 400)
    ax.plot(xs, f(xs), color=CLASS_1, lw=2, label='f(z)')
    if df is not None:
        ax.plot(xs, df(xs), color=CLASS_1, lw=1.4, ls='--', alpha=0.75, label="f'(z)")
    if marker_x is not None:
        ax.plot([marker_x], [f(np.array([marker_x]))[0]], 'o', color=CLASS_0, ms=7)
    ax.axhline(0, color='#cbd2dd', lw=1)
    ax.axvline(0, color='#cbd2dd', lw=1)
    ax.set_xlabel('z')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    if title:
        ax.set_title(title, fontsize=10)
    plt.tight_layout()
    return ax
