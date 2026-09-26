"""Small interactive tour of flux discretisation and Riemann wave geometry.

This file is intentionally separate from ``main.py``: it is a teaching aid,
not part of the event-driven solver or its output pipeline.  Running
``python explore.py`` draws a smooth Burgers flux beside its polygonal
approximation, then prints the wave decomposition of one shock and one
rarefaction.  Importing the module has no plotting or printing side effects,
so its helpers remain safe to reuse in a notebook or a test.
"""

import numpy as np

import _srcpath  # noqa: F401  (finds src/fronttrack without installation)
from fronttrack.flux import DiscreteFlux
from fronttrack.riemann import solve_riemann


def burgers(u):
    """The convex flux f(u)=u^2/2 used by the introductory examples."""
    return u**2 / 2


def plot_polygonal_flux():
    """Show how five state nodes turn the smooth Burgers flux into f_h.

    Matplotlib is imported here rather than at module scope.  The numerical
    library therefore has no plotting dependency until this optional visual
    demonstration is explicitly requested.
    """
    import matplotlib.pyplot as plt

    # These are state-space nodes, not a spatial mesh.  DiscreteFlux samples
    # f at them once and linearly interpolates between neighbouring samples.
    flux = DiscreteFlux(burgers, [-1, -0.5, 0, 0.5, 1])
    # Dense points are solely for drawing the original parabola smoothly.
    u = np.linspace(-1, 1, 401)

    fig, ax = plt.subplots()
    ax.plot(u, burgers(u), label="Original flux", linewidth=2)
    ax.plot(u, flux.value(u), "--", label="Polygonal flux", linewidth=2)
    ax.scatter(flux.u_grid, flux.f_values, color="black",
               label="Flux nodes", zorder=3)
    ax.set(xlabel="State u", ylabel="Flux f(u)",
           title="Original and polygonal fluxes")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.show()


def print_burgers_waves():
    """Print the expected one-shock and two-front Burgers Riemann solutions."""
    flux = DiscreteFlux(burgers, [-1, 0, 1])
    # A decreasing jump is a shock; an increasing jump is a two-step fan on
    # this three-node grid.  Each tuple is (left node, right node, speed).
    print("Downward jump:", solve_riemann(2, 0, flux))
    print("Upward jump:", solve_riemann(0, 2, flux))


def main():
    """Run both independent exploration pieces when invoked as a script."""
    plot_polygonal_flux()
    print_burgers_waves()


if __name__ == "__main__":
    main()
