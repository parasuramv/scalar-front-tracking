"""Measure exact L1 error against a Burgers rarefaction at t=1.

The exact solution is clip(x,-1,1). Outside [-1,1] the staircase agrees with it,
so only the fan needs integration. This is a state-grid refinement experiment;
there is no sampled spatial quadrature error in the reported L1 norm.
"""

import argparse
from pathlib import Path

import numpy as np

import _srcpath  # noqa: F401  (finds src/fronttrack without installation)
from fronttrack.flux import DiscreteFlux
from fronttrack.solver import FrontTracker


def rarefaction_error(nodes: int) -> tuple[float, float]:
    """Return (maximum state spacing, integrated absolute error) for one grid."""
    flux = DiscreteFlux(lambda u: u*u/2, np.linspace(-1, 1, nodes))
    tracker = FrontTracker.from_values(
        flux, [0], [-1, 1], record_history=False,
    ).run(1)
    edges = np.array([-1.0] + [front.x for front in tracker.fronts] + [1.0])
    left, right = edges[:-1], edges[1:]

    # Sample ALL interval states in one call. Reconstructing the full front
    # array once per interval used to introduce quadratic work here.
    values = tracker.sample((left + right) / 2)
    crosses_zero = (left <= values) & (values <= right)
    # On an interval, the integrand is |c-x|. If x=c lies inside it, integrate
    # the two triangles separately; otherwise its signed integral has one sign.
    errors = np.where(
        crosses_zero,
        ((values - left)**2 + (right - values)**2) / 2,
        np.abs(values * (right - left) - (right**2 - left**2) / 2),
    )
    return float(np.max(np.diff(flux.u_grid))), float(np.sum(errors))


def main():
    """Print successive error ratios and save the same table as a CSV."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/convergence.csv"))
    args = parser.parse_args()
    rows = []
    previous = None
    for nodes in (11, 21, 41, 81, 161):
        spacing, error = rarefaction_error(nodes)
        # E(h) ~ C*h^p implies p = log(E_old/E_new)/log(h_old/h_new).
        # The first grid has no predecessor, so its order is undefined (NaN).
        order = np.nan
        if previous is not None:
            order = np.log(previous[1] / error) / np.log(previous[0] / spacing)
        rows.append((nodes, spacing, error, order))
        previous = spacing, error
        print(f"nodes={nodes:3d}  h={spacing:.5f}  L1={error:.6f}  order={order:.3f}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(
        args.output, rows, delimiter=",",
        header="nodes,max_state_spacing,L1_error,order", comments="",
    )


if __name__ == "__main__":
    main()
