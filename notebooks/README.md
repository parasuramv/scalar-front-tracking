# Interactive front tracking

Start with **[Ready-to-run examples](02_ready_to_run_examples.ipynb)**. Run every
cell to compute four small experiments and activate their time controls.

Use **[Front tracking playground](01_front_tracking_playground.ipynb)** for your
own flux function, initial data, state grid, observation window and times. The
editable parameter cell contains a complete working example and alternatives.

## Run in Colab

Open [Google Colab](https://colab.research.google.com/), choose **File → Upload
notebook**, select either `.ipynb`, then **Runtime → Run all**. A CPU runtime is
sufficient. Missing dependencies are installed by the first code cell. Once the
files are on GitHub, Colab can also open their public GitHub URLs.

The notebooks bundle the actual solver source. Neither depends on access to a
local checkout, a sibling project, a particular GitHub repository or a private
package. The bundled source cell is collapsed because it is implementation,
not experiment configuration. It is readable Python, not a remote download.

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
```

The first command refreshes the source snapshot and clears outputs. The second
executes both notebooks in fresh kernels outside the checkout, exercises the
widget callbacks, and saves output figures. It requires the notebook dependencies.
The embedded source fingerprint records the solver snapshot used at generation.
