# %% [markdown]
# # 07 — Gradient descent
#
# Companion to section 07 of the site.
#
# $$\theta \leftarrow \theta - \eta\, \nabla L(\theta)$$
#
# For every parameter: work out how much increasing it would increase the error,
# multiply by a small number, subtract. That is the whole algorithm.
#
# The interesting part is the learning rate $\eta$. This notebook measures its
# stability limit rather than quoting it, shows why the *shape* of the loss
# surface — not its steepness — decides how slow training is, and implements
# momentum.

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
# ## 1. The update rule
#
# Return the whole trajectory, starting point included, so there is something to
# plot and to check.

# %%
def gradient_descent(gradient, theta0, learning_rate, steps):
    """
    Run `steps` updates and return every value of theta, including theta0.

    `gradient` is a function of theta returning dL/dtheta.
    """
    ### PROMPT: Start with [theta0], then repeatedly subtract lr * gradient.
    ### BEGIN SOLUTION
    theta = float(theta0)
    path = [theta]
    for _ in range(steps):
        theta = theta - learning_rate * gradient(theta)
        path.append(theta)
    return np.array(path)
    ### END SOLUTION


check.gradient_descent(gradient_descent)

# %%
loss = lambda t: t ** 2
grad = lambda t: 2 * t

path = gradient_descent(grad, 1.8, 0.1, 6)
print(f'{"step":>5} {"theta":>10} {"loss":>10} {"gradient":>10} {"-lr*grad":>10}')
for i, theta in enumerate(path):
    g = grad(theta)
    print(f'{i:>5} {theta:>10.5f} {loss(theta):>10.5f} {g:>10.5f} {-0.1 * g:>10.5f}')

# %% [markdown]
# ## 2. The learning rate has a hard limit, and you can find it
#
# For $L(\theta) = \tfrac{1}{2}c\,\theta^2$ each step multiplies the distance to
# the minimum by $(1 - \eta c)$. Convergence needs $|1 - \eta c| < 1$, so
# $\eta < 2/c$.
#
# Here $L(\theta) = \theta^2$, which is $c = 2$, predicting a limit of exactly
# $\eta = 1$. Do not take that on trust — search for it.

# %%
def find_stability_limit(gradient, theta0=1.8, steps=60, lo=0.01, hi=2.0, samples=400):
    """Return the largest learning rate that still converges towards zero."""
    rates = np.linspace(lo, hi, samples)
    best = lo
    ### PROMPT: For each rate, run the descent and keep it if the iterates shrank.
    ### HINT: A run converged if the final |theta| is smaller than the starting one.
    ### BEGIN SOLUTION
    for rate in rates:
        final = gradient_descent(gradient, theta0, rate, steps)[-1]
        if np.isfinite(final) and abs(final) < abs(theta0):
            best = rate
    ### END SOLUTION
    return best


limit = find_stability_limit(grad)
print(f'largest learning rate that converged : {limit:.4f}')
print(f'theory says 2 / c = 2 / 2            : {2 / 2:.4f}')
check.stability_limit(limit, curvature=2.0)

# %%
fig, axes = plt.subplots(1, 4, figsize=(13, 3.0), sharey=True)
xs = np.linspace(-2.2, 2.2, 200)
for ax, lr in zip(axes, (0.1, 0.5, 0.95, 1.05)):
    p = gradient_descent(grad, 1.8, lr, 14)
    p = np.clip(p, -2.2, 2.2)
    ax.plot(xs, loss(xs), color='#3b82f6', lw=2)
    ax.plot(p, loss(p), 'o-', color='#e0761f', ms=4, lw=1)
    end = gradient_descent(grad, 1.8, lr, 60)[-1]
    verdict = 'converges' if abs(end) < 1e-3 else ('exact' if abs(end) < 1e-12 else 'diverges')
    ax.set_title(f'$\\eta$ = {lr}  ({verdict})', fontsize=10)
    ax.set_xlabel(r'$\theta$')
    ax.grid(alpha=0.25)
axes[0].set_ylabel(r'$L(\theta)$')
plt.tight_layout()
plt.show()

print('eta = 0.10  approaches from one side')
print('eta = 0.50  lands exactly on the minimum in one step, since 1 - eta*c = 0')
print('eta = 0.95  overshoots and alternates sides, but still shrinks')
print('eta = 1.05  overshoots by more than it started away — it grows for ever')

# %% [markdown]
# ## 3. The shape of the surface, not its steepness
#
# Real problems have many parameters, and the curvature differs by direction.
# The **condition number** $\kappa = \lambda_{\max} / \lambda_{\min}$ measures
# how elongated the bowl is.
#
# Take least squares on six points. The loss over $(w, b)$ is an exact
# paraboloid, so its curvature is a constant matrix — the Hessian — that you can
# write down:
#
# $$H = \frac{2}{m} A^{\top}A \qquad
#   A = \begin{bmatrix} x_1 & 1 \\ \vdots & \vdots \\ x_m & 1 \end{bmatrix}$$

# %%
POINTS_X = np.array([-1.2, -0.6, -0.1, 0.4, 0.9, 1.4])
POINTS_Y = np.array([-0.9, -0.2, 0.25, 0.55, 1.3, 1.5])


def condition_number(feature_column):
    """Condition number of the least-squares Hessian for this feature."""
    x = np.asarray(feature_column).reshape(-1, 1)
    ### PROMPT: Build A with a column of ones, form H, take the eigenvalue ratio.
    ### HINT: np.linalg.eigvalsh(H) returns the eigenvalues of a symmetric matrix.
    ### BEGIN SOLUTION
    A = np.hstack([x, np.ones_like(x)])
    H = 2.0 / len(x) * (A.T @ A)
    eigenvalues = np.linalg.eigvalsh(H)
    return float(eigenvalues.max() / eigenvalues.min())
    ### END SOLUTION


check.condition_number(condition_number)

# %%
for scale in (3.0, 1.0, 0.5, 0.2, 0.1):
    k = condition_number(POINTS_X * scale)
    print(f'feature scaled by {scale:>4}:  kappa = {k:>8.2f}   '
          f'best shrink per step = {(k - 1) / (k + 1):.4f}   '
          f'steps for 1000x = {int(np.ceil(np.log(1e-3) / np.log((k - 1) / (k + 1)))) if k > 1.001 else 1}')

print()
print('Rescaling one input feature does not change the problem — the best-fitting')
print('line is the same. It only changes how hard the optimiser has to work.')
print('That is what standardising your inputs actually buys you.')

# %%
def least_squares_loss(w, b, scale=1.0):
    pred = w * (POINTS_X * scale) + b
    return np.mean((pred - POINTS_Y) ** 2)


def least_squares_grad(w, b, scale=1.0):
    x = POINTS_X * scale
    err = w * x + b - POINTS_Y
    return 2 * np.mean(err * x), 2 * np.mean(err)


def descend_2d(scale, lr, steps, momentum=0.0, start=(0.0, 0.0)):
    w, b = start
    vw = vb = 0.0
    path = [(w, b)]
    for _ in range(steps):
        gw, gb = least_squares_grad(w, b, scale)
        vw, vb = momentum * vw + gw, momentum * vb + gb
        w, b = w - lr * vw, b - lr * vb
        path.append((w, b))
    return np.array(path)


fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
for ax, scale in zip(axes, (1.0, 0.2)):
    k = condition_number(POINTS_X * scale)
    optimum_w = np.polyfit(POINTS_X * scale, POINTS_Y, 1)[0]
    grid_w = np.linspace(optimum_w - 2.6, optimum_w + 2.0, 160)
    grid_b = np.linspace(-1.4, 2.0, 160)
    WW, BB = np.meshgrid(grid_w, grid_b)
    ZZ = np.array([[least_squares_loss(w, b, scale) for w in grid_w] for b in grid_b])

    ax.contour(WW, BB, ZZ, levels=22, colors='#94a3b8', linewidths=0.6)
    path = descend_2d(scale, 0.45, 60)
    ax.plot(path[:, 0], path[:, 1], 'o-', color='#e0761f', ms=2.5, lw=1.1)
    ax.plot([optimum_w], [np.polyfit(POINTS_X * scale, POINTS_Y, 1)[1]], '+',
            color='#14171f', ms=12, mew=2)
    ax.set_title(f'feature scale {scale}:  $\\kappa$ = {k:.1f}', fontsize=10)
    ax.set_xlabel('w')
    ax.set_ylabel('b')
plt.tight_layout()
plt.show()

print('Same learning rate, same number of steps. On the round surface it walks')
print('straight in. On the stretched one it crawls along the flat direction.')

# %% [markdown]
# ## 4. Momentum
#
# Plain gradient descent forgets everything between steps. Momentum keeps a
# running quantity, shrinks it a little each step, and adds the current
# gradient:
#
# $$\mathbf{v} \leftarrow \beta\mathbf{v} + \nabla L(\theta)
#   \qquad
#   \theta \leftarrow \theta - \eta\mathbf{v}$$
#
# A direction the gradient keeps pointing in accumulates. A direction that flips
# back and forth cancels. At $\beta = 0$ this is ordinary gradient descent.

# %%
def momentum_descent(gradient, theta0, learning_rate, steps, beta=0.9):
    """Gradient descent with momentum. Return the trajectory including theta0."""
    ### PROMPT: Track a velocity v, update it, then step along it.
    ### BEGIN SOLUTION
    theta = float(theta0)
    velocity = 0.0
    path = [theta]
    for _ in range(steps):
        velocity = beta * velocity + gradient(theta)
        theta = theta - learning_rate * velocity
        path.append(theta)
    return np.array(path)
    ### END SOLUTION


check.momentum(momentum_descent)

# %%
# Momentum earns its keep exactly where plain descent is slow: a stretched surface.
scale = 0.2
k = condition_number(POINTS_X * scale)
optimum = np.polyfit(POINTS_X * scale, POINTS_Y, 1)

print(f'condition number {k:.1f}, learning rate 0.45, 60 steps\n')
print(f'{"beta":>6} {"final loss":>14} {"distance to optimum":>22}')
for beta in (0.0, 0.5, 0.8, 0.9, 0.97):
    path = descend_2d(scale, 0.45, 60, momentum=beta)
    w, b = path[-1]
    distance = np.hypot(w - optimum[0], b - optimum[1])
    print(f'{beta:>6.2f} {least_squares_loss(w, b, scale):>14.8f} {distance:>22.2e}')

print()
print('beta = 0.8 gets orders of magnitude closer than beta = 0 in the same budget.')
print('beta = 0.97 is worse again: the accumulated velocity carries it past the')
print('minimum faster than the gradient can turn it round. The stable region is')
print('over the pair (eta, beta) together, not over either one alone.')

# %%
fig, ax = plt.subplots(figsize=(6.4, 4.2))
grid_w = np.linspace(optimum[0] - 2.6, optimum[0] + 2.0, 160)
grid_b = np.linspace(-1.4, 2.0, 160)
WW, BB = np.meshgrid(grid_w, grid_b)
ZZ = np.array([[least_squares_loss(w, b, scale) for w in grid_w] for b in grid_b])
ax.contour(WW, BB, ZZ, levels=22, colors='#94a3b8', linewidths=0.6)
for beta, colour in ((0.0, '#e0761f'), (0.8, '#3b82f6')):
    path = descend_2d(scale, 0.45, 60, momentum=beta)
    ax.plot(path[:, 0], path[:, 1], 'o-', ms=2.5, lw=1.2, color=colour, label=f'$\\beta$ = {beta}')
ax.plot([optimum[0]], [optimum[1]], '+', color='#14171f', ms=12, mew=2)
ax.set_xlabel('w')
ax.set_ylabel('b')
ax.set_title(f'the same 60 steps, $\\kappa$ = {k:.0f}', fontsize=10)
ax.legend(fontsize=9)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 5. Gradient descent is a local rule
#
# It follows the slope under the current point. It does not know where the
# minimum is, how far away it is, or whether a lower one exists elsewhere.
# On a surface with several minima, where you start decides where you end.

# %%
wavy = lambda t: 0.35 * t ** 2 + 0.55 * np.sin(3 * t) + 1
dwavy = lambda t: 0.7 * t + 1.65 * np.cos(3 * t)

fig, ax = plt.subplots(figsize=(6.8, 3.6))
xs = np.linspace(-3.4, 3.4, 400)
ax.plot(xs, wavy(xs), color='#3b82f6', lw=2)
for start, colour in ((-3.0, '#e0761f'), (-0.5, '#16a34a'), (2.6, '#8b5cf6')):
    p = gradient_descent(dwavy, start, 0.08, 120)
    ax.plot(p, wavy(p), 'o-', ms=2.5, lw=0.9, color=colour, alpha=0.8)
    ax.plot([p[-1]], [wavy(p[-1])], 'o', ms=8, color=colour,
            label=f'start {start:+.1f} -> {p[-1]:+.2f}')
ax.set_xlabel(r'$\theta$')
ax.set_ylabel(r'$L(\theta)$')
ax.legend(fontsize=8)
ax.grid(alpha=0.25)
plt.tight_layout()
plt.show()

print('Three identical runs of the identical rule, three different answers.')
print('Nothing compared them. Each one only ever saw the slope beneath itself.')

# %% [markdown]
# ## Try this
#
# * Change `loss` to `0.4 * (t - 1.2)**2` (so `c = 0.8`) and re-run
#   `find_stability_limit` with `hi=4.0`. The limit moves to `2/0.8 = 2.5`.
#   Flatter surfaces both tolerate and need larger steps.
# * In the momentum table, drop the learning rate to 0.1 and re-run. Momentum's
#   advantage gets *larger*, because plain descent is further from its limit.
#
# **Next:** `08-backprop.ipynb` — where the gradients for a real network come
# from.
