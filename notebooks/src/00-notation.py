# %% [markdown]
# # 00 — Reading the mathematics, in code
#
# Companion to section 00 of the site.
#
# The site shows you what the notation means. This notebook makes you write it.
# Every symbol used in the rest of the course — the sum, the matrix product, the
# derivative, the chain rule — appears here as a few lines of numpy that you
# implement and then verify.
#
# **How this works.** Each exercise is a function with its body removed. Replace
# the `raise NotImplementedError(...)` line with your code and run the cell
# underneath it. The check does not compare your answer to a stored solution —
# it tests properties. If it passes, the thing is actually right.
#
# Stuck? The completed version of this notebook is in `notebooks/solutions/`.

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
print('ready')

# %% [markdown]
# ## 1. Σ — add up a list of products
#
# $$\sum_{i=1}^{n} w_i x_i$$
#
# The three parts: `i = 1` says the counter starts at 1, `n` says where it
# stops, and the expression on the right is what gets repeated and added.
#
# Write it twice. First with an explicit loop, so there is no doubt about what
# the symbol asks for.

# %%
def weighted_sum_loop(w, x):
    """Return the sum of w[i] * x[i], written as an explicit loop."""
    ### PROMPT: Loop over the entries of w and x, multiplying and accumulating.
    ### BEGIN SOLUTION
    total = 0.0
    for i in range(len(w)):
        total += w[i] * x[i]
    return total
    ### END SOLUTION


check.weighted_sum(weighted_sum_loop)

# %% [markdown]
# Now the same thing the way you will actually write it. `np.dot` — or the `@`
# operator — performs exactly this loop in compiled code.

# %%
def weighted_sum(w, x):
    """Return the sum of w[i] * x[i] using numpy instead of a loop."""
    ### PROMPT: One call to np.dot, or (w * x).sum().
    ### HINT: np.dot(w, x) multiplies matching entries and adds the products.
    ### BEGIN SOLUTION
    return np.dot(w, x)
    ### END SOLUTION


check.weighted_sum(weighted_sum)

# %%
# The two agree, and the numpy one is far faster on anything large.
rng = np.random.default_rng(0)
w, x = rng.normal(size=100_000), rng.normal(size=100_000)
print('loop  ', weighted_sum_loop(w, x))
print('numpy ', weighted_sum(w, x))

# %% [markdown]
# ## 2. Matrices — the same sum, once per row
#
# A **vector** is an ordered list of numbers. A **matrix** is a grid of them.
# Multiplying a matrix by a vector produces one output per row: each output is
# the weighted sum above, using that row as the weights.
#
# A layer of a neural network is exactly this — one row of weights per unit.

# %%
def matvec(W, x):
    """
    Multiply matrix W by vector x, without using @ or np.dot on the whole thing.

    W has shape (rows, cols) and x has length cols, so the answer has length rows.
    """
    ### PROMPT: Build the output one row at a time using weighted_sum.
    ### HINT: np.array([...]) turns a list of numbers into an array.
    ### BEGIN SOLUTION
    return np.array([weighted_sum(row, x) for row in W])
    ### END SOLUTION


check.matvec(matvec)

# %%
# numpy writes the same operation as W @ x.
W = np.array([[0.5, -1.0],
              [2.0, 0.5],
              [-0.5, 1.5]])
x = np.array([4.0, 2.0])

print('by hand:', matvec(W, x))
print('W @ x  :', W @ x)
print()
print('W is', W.shape, 'and x is', x.shape, '-> result is', (W @ x).shape)

# %% [markdown]
# ### A convention worth pinning down now
#
# The website writes one example as a column and puts $W$ in shape
# (units, inputs), so a layer is $W\mathbf{x}$. Numpy code almost always does it
# the other way round: **one example per row**, $W$ in shape (inputs, units), so
# a layer is `X @ W + b`.
#
# Same arithmetic, transposed layout. Every notebook from here on uses the numpy
# convention:
#
# | quantity | shape |
# |---|---|
# | `X` — a batch of examples | `(m, n_in)` |
# | `W` — one layer's weights | `(n_in, n_out)` |
# | `b` — one layer's biases | `(n_out,)` |
# | `X @ W + b` | `(m, n_out)` |
#
# Predict the shapes before you run the cell.

# %%
m, n_in, n_hidden, n_out = 32, 2, 8, 1
X = rng.normal(size=(m, n_in))
W1, b1 = rng.normal(size=(n_in, n_hidden)), np.zeros(n_hidden)
W2, b2 = rng.normal(size=(n_hidden, n_out)), np.zeros(n_out)

Z1 = X @ W1
A1 = np.tanh(Z1 + b1)

answers = {
    ### PROMPT: Fill in each shape as a tuple, e.g. (32, 8).
    ### BEGIN SOLUTION
    'X @ W1': (32, 8),
    'Z1 + b1': (32, 8),
    'A1 @ W2': (32, 1),
    'W1.T': (8, 2),
    ### END SOLUTION
}

check.shapes(answers)
print('actual:', (X @ W1).shape, (Z1 + b1).shape, (A1 @ W2).shape, W1.T.shape)

# %% [markdown]
# `b1` has 8 entries but `Z1` is 32×8. Adding them works because numpy
# **broadcasts**: the length-8 vector is added to every one of the 32 rows. That
# is exactly what a bias should do — the same offset for every example.

# %% [markdown]
# ## 3. Derivatives — how fast the output moves
#
# A derivative answers one question: if the input increases by a tiny amount,
# how much does the output change, per unit of input?
#
# Measure it the way you would measure the steepness of a hill. Step a short
# distance sideways (`h`), see how far you rose, divide. Use the **centred**
# form, which measures on both sides of the point and is much more accurate than
# stepping only forwards:
#
# $$f'(x) \approx \frac{f(x+h) - f(x-h)}{2h}$$

# %%
def numerical_derivative(f, x, h=1e-5):
    """Estimate f'(x) by nudging x in both directions."""
    ### PROMPT: Implement the centred difference formula above.
    ### BEGIN SOLUTION
    return (f(x + h) - f(x - h)) / (2 * h)
    ### END SOLUTION


check.numerical_derivative(numerical_derivative)

# %%
# Watch the estimate converge, then fall apart.
x0 = 1.0
hs = np.logspace(-1, -12, 40)
errors = [abs(numerical_derivative(np.sin, x0, h) - np.cos(x0)) for h in hs]

fig, ax = plt.subplots(figsize=(5.6, 3.4))
ax.loglog(hs, np.maximum(errors, 1e-18), color='#3b82f6', marker='o', ms=3)
ax.set_xlabel('h')
ax.set_ylabel("error in the estimate of $f'(1)$")
ax.set_title('centred difference on $f(x) = \\sin x$', fontsize=10)
ax.invert_xaxis()
ax.grid(alpha=0.3, which='both')
plt.tight_layout()

best = int(np.argmin(errors))
print(f'best h   ~ {hs[best]:.1e}')
print(f'best err   {errors[best]:.2e}')
print(f'at h=1e-1  {errors[0]:.2e}')
print(f'at h=1e-12 {errors[-1]:.2e}')

# %% [markdown]
# The error falls as `h` shrinks, bottoms out around `h = 1e-5`, then gets worse
# again. Two effects are fighting: the formula is only exact in the limit, and
# subtracting two nearly equal floating-point numbers loses precision. This is
# why the gradient checks later in the course use `h = 1e-5` and not something
# smaller.

# %% [markdown]
# ## 4. The chain rule — following a change through several steps
#
# This is the one piece of calculus the whole course rests on. When a change has
# to travel through several steps, **multiply the rates**.
#
# Here is a chain with three links — it is a single neuron with one input,
# scored against a target:
#
# $$z = wx + b \qquad a = \sigma(z) \qquad L = (a - y)^2$$
#
# The three rates are
#
# $$\frac{\partial z}{\partial x} = w \qquad
#   \frac{\partial a}{\partial z} = a(1-a) \qquad
#   \frac{\partial L}{\partial a} = 2(a - y)$$
#
# and the chain rule says $\dfrac{\partial L}{\partial x}$ is their product.

# %%
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def dL_dx(x, w, b, y):
    """Return dL/dx for z = wx + b, a = sigmoid(z), L = (a - y)**2."""
    ### PROMPT: Compute z and a, then multiply the three rates together.
    ### BEGIN SOLUTION
    z = w * x + b
    a = sigmoid(z)
    dz_dx = w
    da_dz = a * (1 - a)
    dL_da = 2 * (a - y)
    return dL_da * da_dz * dz_dx
    ### END SOLUTION


check.chain_rule(dL_dx)

# %% [markdown]
# The check confirmed your answer against a direct measurement: it nudged `x`,
# watched `L` move, and divided. The chain rule is not an approximation that
# happens to work — it is exact, and the only error is the one from the nudge.
#
# Compare the two yourself:

# %%
x0, w0, b0, y0 = 1.5, 0.8, 0.2, 1.0


def loss_of_x(xv):
    return (sigmoid(w0 * xv + b0) - y0) ** 2


print(f'chain rule : {dL_dx(x0, w0, b0, y0):.9f}')
print(f'measured   : {numerical_derivative(loss_of_x, x0):.9f}')
print()
print('the rates, one per link:')
z0 = w0 * x0 + b0
a0 = sigmoid(z0)
print(f'  dz/dx = w        = {w0:.6f}')
print(f'  da/dz = a(1-a)   = {a0 * (1 - a0):.6f}')
print(f'  dL/da = 2(a - y) = {2 * (a0 - y0):.6f}')
print(f'  product          = {w0 * a0 * (1 - a0) * 2 * (a0 - y0):.9f}')

# %% [markdown]
# ## What you built
#
# * `weighted_sum` — one neuron's core operation
# * `matvec` — a whole layer's core operation
# * `numerical_derivative` — the tool that will verify every gradient you write
# * `dL_dx` — backpropagation through a one-unit network, by hand
#
# The last one is not a toy. Notebook 08 does the same thing for an arbitrary
# number of layers, and the code is barely longer.
#
# **Next:** `01-structure.ipynb` — describing a whole network with a list of
# layer sizes.
