# Upload-ready project

The two notebooks are standalone: upload either `.ipynb` alone, or upload this
project with its `src/`, notebooks, README, dependency configuration and tests.
There is no hard-coded GitHub owner, repository, branch or private download URL.

For a GitHub repository, keep the relative folder layout so the README links
work. GitHub renders saved figures; Colab/Jupyter runs the interactive controls.
After upload, add a Colab badge pointing at the actual public notebook URL. No
publication or remote-repository configuration has been performed here.

Do not upload `.venv/`, `__pycache__/`, local editor metadata or generated export
folders. These are excluded by `.gitignore`. The public notebooks' saved outputs
contain plots and example diagnostics, not machine-specific paths or credentials.

## Maintaining standalone notebooks

1. Edit the canonical package in `src/fronttrack/`.
2. Run `python tools/build_notebooks.py` to update both embedded source snapshots.
3. Run `python tools/check_notebooks.py --write` to execute and save previews.
4. Run `python -m unittest discover -v` with plotting dependencies installed.

The notebook runner tests a fresh, isolated environment directory with the current
Python interpreter. It is not a claim that the live Colab service was tested.
Matplotlib's [pcolormesh documentation](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.pcolormesh.html)
describes the raster shading convention; the notebooks use the official
[Colab widget manager](https://github.com/googlecolab/colabtools/blob/main/google/colab/output/_widgets.py)
when running in Colab.

## Project boundaries

This folder remains independently usable. Compact Domain Schemes and Splitting
Schemes gained plotting APIs and CLI figures, with usage in their own READMEs;
they do not yet have notebooks. Their plots require this updated `fronttrack`
package.
