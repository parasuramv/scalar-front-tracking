"""Run reproducible numerical examples: python main.py --help.

This module handles example selection, observation times, export, and plotting.
The numerical evolution lives in solver.py and does not depend on Matplotlib.
"""

import argparse
from pathlib import Path

import numpy as np

import _srcpath  # noqa: F401  (finds src/fronttrack without installation)
from fronttrack.flux import DiscreteFlux
from fronttrack.solver import FrontTracker


# Each entry contains jump positions and physical values on the N+1 regions.
# Both exterior regions extend to infinity; plot limits do not impose boundaries.
_BURGERS_DATA = {
    "shock": ([0], [1, -1]),
    "rarefaction": ([0], [-1, 1]),
    "collision": ([-0.5, 0.5], [1, 0, -1]),
    "pulse": ([0, 1], [0, 1, 0]),
}


def example(name: str, nodes: int, *, record_history=True) -> FrontTracker:
    """Construct a named example at time zero, with an odd state-node count."""
    if nodes < 3 or nodes % 2 != 1:
        raise ValueError("Use an odd number of nodes >= 3 so zero is a grid node")
    grid = np.linspace(-1, 1, nodes)
    if name == "nonconvex":
        flux = DiscreteFlux(lambda u: u**3, grid)
        positions = np.linspace(-1, 1, 41)
        centers = (positions[:-1] + positions[1:]) / 2
        # Midpoint samples approximate a sine profile; append constant exteriors.
        values = np.r_[0.0, 0.9 * np.sin(np.pi * centers), 0.0]
    elif name in _BURGERS_DATA:
        flux = DiscreteFlux(lambda u: 0.5 * u**2, grid)
        positions, values = _BURGERS_DATA[name]
    else:
        raise ValueError(f"Unknown example: {name}")
    return FrontTracker.from_values(
        flux, positions, values, record_history=record_history,
    )


def parse_args(argv=None):
    """Parse and validate CLI settings before allocating a simulation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--example", choices=[*_BURGERS_DATA, "nonconvex"], default="pulse")
    parser.add_argument("--nodes", type=int, default=81, help="Number of flux state nodes (odd)")
    parser.add_argument("--time", type=float, default=3.0, help="Final absolute time")
    parser.add_argument("--frames", type=int, default=101, help="Number of observation times")
    parser.add_argument("--samples", type=int, default=1001, help="Plot/sample grid size")
    parser.add_argument("--xmin", type=float, default=-2.0)
    parser.add_argument("--xmax", type=float, default=4.0)
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--no-plot", action="store_true", help="Skip plotting and trajectory history")
    args = parser.parse_args(argv)
    if not np.isfinite(args.time) or args.time < 0 or args.frames < 2 or args.samples < 2:
        parser.error("Require finite time >= 0, frames >= 2 and samples >= 2")
    if not np.isfinite([args.xmin, args.xmax]).all() or args.xmin >= args.xmax:
        parser.error("Require finite xmin < xmax")
    if args.nodes < 3 or args.nodes % 2 != 1:
        parser.error("Use an odd number of nodes >= 3 so zero is a grid node")
    return args


def collect_outputs(tracker, x, times):
    """Observe one evolving tracker; return profiles, diagnostics, and front rows.

    The fixed-size profile and diagnostic tables are allocated once. Front count
    changes at collisions, so the front geometry table is accumulated as a list.
    """
    profiles = np.empty((len(times), len(x)))
    diagnostics = np.empty((len(times), 5))
    front_rows = []
    for frame, time in enumerate(times):
        tracker.run(float(time))
        profiles[frame] = tracker.sample(x)
        diagnostics[frame] = (
            time, len(tracker.fronts), tracker.event_count,
            tracker.total_variation(), tracker.integral(x[0], x[-1]),
        )
        front_rows.extend(
            (time, front.x, front.iL, front.iR, front.speed)
            for front in tracker.fronts
        )
    # A constant solution has no fronts; still return a table with five columns.
    return profiles, diagnostics, np.asarray(front_rows).reshape(-1, 5)


def save_outputs(prefix, tracker, x, times, profiles, diagnostics, front_rows):
    """Save sampled arrays and exact front geometry; overwrite this prefix's files."""
    np.savez(
        str(prefix) + ".npz", x=x, times=times, u=profiles,
        u_grid=tracker.flux.u_grid, f_values=tracker.flux.f_values,
    )
    np.savetxt(
        str(prefix) + "_diagnostics.csv", diagnostics, delimiter=",",
        header="time,fronts,events,total_variation,window_integral", comments="",
    )
    np.savetxt(
        str(prefix) + "_fronts.csv", front_rows, delimiter=",",
        header="time,x,iL,iR,speed", comments="",
    )


def plot_results(prefix, name, tracker, x, times, profiles):
    """Render profiles and recorded front trajectories to a PNG without a GUI."""
    # Lazy imports keep Matplotlib optional for API users and --no-plot runs.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from fronttrack.visualization import History, plot_spacetime

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    try:
        for k in sorted(set(np.linspace(0, len(times)-1, min(5, len(times)), dtype=int))):
            time, values = times[k], profiles[k]
            axes[0].step(x, values, where="post", label=f"t={time:g}")
        axes[0].set(xlabel="x", ylabel="u", title=name)
        axes[0].legend(fontsize=8)
        if times[-1] > times[0]:
            history = History(x, times, profiles, tracker.segments)
            plot_spacetime(history, ax=axes[1])
        else:
            axes[1].text(.5, .5, "No evolution: final time is zero", ha="center")
            axes[1].set_axis_off()
        fig.tight_layout()
        fig.savefig(str(prefix) + ".png", dpi=160)
    finally:
        plt.close(fig)  # Release figure memory even if saving raises an error.


def main(argv=None):
    """Connect input selection, numerical evolution, and output generation."""
    args = parse_args(argv)
    tracker = example(args.example, args.nodes, record_history=not args.no_plot)
    args.output.mkdir(parents=True, exist_ok=True)
    x = np.linspace(args.xmin, args.xmax, args.samples)
    times = np.linspace(0, args.time, args.frames)
    profiles, diagnostics, front_rows = collect_outputs(tracker, x, times)
    prefix = args.output / args.example
    save_outputs(prefix, tracker, x, times, profiles, diagnostics, front_rows)
    if not args.no_plot:
        plot_results(prefix, args.example, tracker, x, times, profiles)
    print(f"t={tracker.time:g}: {len(tracker.fronts)} fronts, {tracker.event_count} interactions")
    print(f"Results: {prefix.resolve()}.*")


if __name__ == "__main__":
    # Importing this file exposes helpers without starting an example.
    main()
