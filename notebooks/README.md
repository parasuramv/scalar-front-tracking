# Interactive front tracking

Start with **[Ready-to-run examples](02_ready_to_run_examples.ipynb)**. Run every
cell to compute four small experiments and activate their time controls.

Use **[Front tracking playground](01_front_tracking_playground.ipynb)** for your
own flux function, initial data, state grid, observation window and times. The
editable parameter cell contains a complete working example and alternatives.

## Run in Colab

Use the *Open in Colab* badges in the [main README](../README.md), then
**Runtime → Run all**. A CPU runtime is sufficient. Missing dependencies are
installed by the first code cell. You can also upload a downloaded `.ipynb`
with **File → Upload notebook**.

Notebooks 01 and 02 bundle the actual solver source, so neither depends on a
local checkout or on this repository being reachable. The bundled source cell
is collapsed because it is implementation, not experiment configuration. It is
readable Python, not a remote download.

### Theory notebooks (03 onward)

**[03 · The Riemann problem](03_riemann_problem.ipynb)**,
**[04 · Interactions](04_interactions.ipynb)** (the event loop, collisions,
front and event counts), **[05 · Convergence](05_convergence.ipynb)**
(rates as δ → 0 in L¹ and for the primitive, projection of initial data,
roundoff) and **[06 · Stability](06_stability.ipynb)** (L¹ contraction and
where it comes from, total variation, front counts, failure of L² and L∞
stability) do *not* bundle the solver. Their first cell uses `fronttrack` from
the checkout if present, and otherwise installs it from this repository with
`pip`; in Colab this happens automatically.

## Run locally

In the project root with Python 3.10+:

```bash
python -m pip install -e ".[notebook]"
python -m jupyterlab notebooks/
```

Select Python 3 and run all cells. In a checkout the notebooks use the local
`src/fronttrack` package. Restart the kernel when switching source versions.
GitHub itself displays static output; it does not run Python or time sliders.

## What to change

- **Flux and state grid:** change the numerical polygonal flux.
- **Initial jumps and values:** change the initial step function; first/last
  values extend to infinity. A sampled-function example is included.
- **Observation window:** crops the view; it imposes no boundary conditions.
- **Snapshot times:** preserve exact step geometry at those times.
- **Space samples and time frames:** improve the colour raster only.

The playground's additional-time cell independently evolves a fresh tracker and
can generate a snapshot at any finite nonnegative time. The slider selects
cached frames and does not rerun the solver.

Exports appear in `front_tracking_exports/` next to the running notebook. Colab's
Files panel lets readers download PNG/PDF figures and compressed NumPy data.

## Rebuild after source changes

```bash
python tools/build_notebooks.py
python tools/check_notebooks.py --write
python tools/build_theory_notebooks.py --execute
```

The first command refreshes the source snapshot and clears outputs. The second
executes 01 and 02 in fresh kernels outside the checkout, exercises the
widget callbacks, and saves output figures. The third rebuilds the theory notebooks (03 onward)
from their build script (edit that script, never the `.ipynb`). All three require the
notebook dependencies. The embedded source fingerprint records the solver
snapshot used at generation.

Saved notebooks contain no widget state, execution timestamps or random cell
ids, so rebuilding unchanged sources in the same environment reproduces them
byte for byte, and `git diff` after a rebuild shows only real changes. Outside a
live kernel, widget cells show a note (01, 02) or a static preview (03 onward).
