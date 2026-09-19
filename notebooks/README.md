# Exercise notebooks

Thirteen Jupyter notebooks that run alongside the website, one per section. The
site shows you what happens; these make you build it.

**53 exercises.** Each is a function with its body removed. You fill it in and
run the check underneath. Nothing is taken on trust — the checks verify
properties rather than comparing against a stored answer, so your derivatives
are tested against finite differences of your own code, and your batched
implementations against your own single-example versions.

By the end you will have written a working neural network library from an empty
file: forward propagation, backpropagation, mini-batch training, L2, dropout and
early stopping, all in plain numpy.

## Running them

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r notebooks/requirements.txt jupyterlab
jupyter lab notebooks/
```

Only numpy and matplotlib are needed to do the exercises; JupyterLab is just one
way to open them. Each notebook also works in VS Code, in PyCharm, or in Google
Colab — the first cell clones the repository automatically when the helper
package is not already present.

## The notebooks

| # | Notebook | What you build | Exercises |
|---|---|---|---|
| 00 | `00-notation.ipynb` | Σ, matrix products, derivatives, the chain rule | 6 |
| 01 | `01-structure.ipynb` | Layer shapes, parameter counting, initialisation | 4 |
| 02 | `02-neurons.ipynb` | One neuron; a brute-force proof it cannot do XOR | 3 |
| 03 | `03-weights.ipynb` | He and Glorot scaling, measured over ten layers | 3 |
| 04 | `04-activations.ipynb` | Four activations, their derivatives, a stable softmax | 10 |
| 05 | `05-forward.ipynb` | The forward pass and the cache backprop needs | 2 |
| 06 | `06-loss.ipynb` | MSE, cross-entropy, and why classification uses the latter | 6 |
| 07 | `07-gradient-descent.ipynb` | The update rule, its stability limit, momentum, conditioning | 4 |
| 08 | `08-backprop.ipynb` | Backpropagation, gradient-checked parameter by parameter | 3 |
| 09 | `09-training.ipynb` | Mini-batches, the training loop, learning rate and batch size | 2 |
| 10 | `10-overfitting.ipynb` | The overfitting curve, L2, dropout, early stopping | 3 |
| 11 | `11-depth.ipynb` | Depth versus width, and counting linear regions | 2 |
| 12 | `12-capstone.ipynb` | The whole library, in one class, from scratch | 5 |

Work through them in order. Each notebook restates what it needs from the
previous ones, so you can also drop into any of them on its own.

## Stuck?

`notebooks/solutions/` holds a completed, executed copy of every notebook. The
checks are designed to say something useful when they fail — they name the
specific case that went wrong and usually suggest what to look at — so read the
failure before reaching for the solution.

## What is in `dnn/`

Helpers, none of which solve an exercise:

| module | contents |
|---|---|
| `dnn/data.py` | The same five 2-D datasets the website uses |
| `dnn/plotting.py` | Decision boundaries, scatter plots, training curves |
| `dnn/check.py` | The correctness checks |
| `dnn/mlp.py` | A complete reference network, used by notebooks 09–12 |

`dnn/mlp.py` is the finished version of what notebooks 01–08 build. Notebooks
09 onwards import it so they can concentrate on training dynamics rather than
re-deriving gradients. If you are working through notebook 08, write your own
first.

## For maintainers

Notebooks are generated, not edited by hand. The source of truth is
`notebooks/src/*.py`, written in the "percent" cell format with solutions marked:

```python
### PROMPT: What the student should do.
### HINT: An optional nudge, kept in both versions.
### BEGIN SOLUTION
return np.maximum(0.0, z)
### END SOLUTION
```

From one source the build produces the student notebook (solution bodies
replaced by a stub) and the solution notebook (complete).

```bash
python tools/build_notebooks.py           # regenerate both sets
python tools/build_notebooks.py --check   # fail if the committed ones are stale
python tools/run_notebooks.py             # execute every solution notebook
python tools/run_notebooks.py 08 09       # just those
```

`run_notebooks.py` is what proves the exercises are solvable: it executes each
solution top to bottom and fails if any cell raises, which includes every
`check.*` call inside them.
