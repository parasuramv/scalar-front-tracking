# Public notebook and visualization validation — 27 September 2026

- Homo Front Tracking: 26 tests pass (21 existing numerical tests, five visualization regressions).
- Compact Domain Schemes: 22 tests pass, including new checks for recorded paths crossing the periodic seam in both directions over multiple circuits.
- Splitting Schemes: all 26 existing tests pass.
- Both notebooks execute from start to finish in separate fresh Jupyter kernels, using the bundled source from temporary directories outside the checkout. The checker also changes time, colour map, path visibility and each preloaded example through widget callbacks. Executed outputs are saved in the notebooks.
- The final embedded source fingerprint begins `9a8c2346c6a57882`; rebuild notebooks after changing the package.
- Static figures were visually inspected for the notebook gallery, real-line pulse, periodic transport and split cell-average snapshots/slices. Boundary hatching is restricted to the invalid cells; a regression checks the hatch geometry.
- Plot checks cover exact jump locations versus coarse sampling, narrow states missed by the display raster, constant solutions, single-time histories, nonuniform grid cells, slice orientation, consistent colours and observation-input validation.
- CLI smoke runs produced arrays and figures for the real-line pulse, periodic transport, 2D torus advection and 2D splitting advection/decay. Zero-time runs were checked for real-line, circle and splitting views.
- This is local execution validation with the `claw` Python environment. Live Colab rendering was not tested. No GitHub upload or deployment was performed.

The checks above validate implementation behaviour and displayed geometry; they do not constitute a general convergence proof.

---

# Validation — 26 September 2026

- Restructured into the installable `fronttrack` package (`src/fronttrack/`, `pyproject.toml`); root-level `flux.py` etc. are unused compatibility shims. Code moved verbatim apart from relative imports.
- All 21 tests pass, with and without `pip install -e .`. CLI and convergence outputs are identical to before the move (nonconvex example: 60 interactions; rarefaction L1 errors 0.1 → 0.00625, order 1).
- An editable install exposes only `fronttrack` (checked: `flux`, `solver`, `main`, `_srcpath` are not importable elsewhere).
- Independent checks: 300 random trials (four fluxes including nonconvex, random nonuniform state grids) preserved mass, nonincreasing total variation, front ordering and state continuity; the u^3 example's L1 distance to a fine Godunov run decreased under state refinement (0.044, 0.022, 0.017).
