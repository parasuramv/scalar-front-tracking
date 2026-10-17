"""Run periodic examples on the circle S^1; save profiles, fronts and diagnostics.

python periodic_main.py --help. The whole-line examples are in main.py.
"""
import argparse
import csv
from pathlib import Path
import numpy as np
import _srcpath  # noqa: F401  (finds src/fronttrack without installation)
from fronttrack.flux import DiscreteFlux
from fronttrack.periodic import Circle, PeriodicFrontTracker
from fronttrack.visualization import History, step_profile, plot_profile, plot_spacetime


def main(argv=None):
    """Parse options, build one example, observe it at output times, save."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--example', choices=['pulse', 'sine', 'transport', 'nonconvex'], default='pulse')
    parser.add_argument('--length', type=float, default=1.0)
    parser.add_argument('--time', type=float, default=2.0)
    parser.add_argument('--nodes', type=int, default=81)
    parser.add_argument('--cells', type=int, default=64)
    parser.add_argument('--samples', type=int, default=600)
    parser.add_argument('--frames', type=int, default=81)
    parser.add_argument('--max-events', type=int, default=100000)
    parser.add_argument('--output', type=Path, default=Path('results'))
    parser.add_argument('--no-plot', action='store_true')
    args = parser.parse_args(argv)
    if not np.isfinite(args.time) or args.time < 0:
        parser.error('--time must be finite and nonnegative')
    if args.nodes < 3 or args.nodes % 2 == 0:
        parser.error('--nodes must be odd and at least 3')
    if min(args.cells, args.samples, args.max_events) < 1 or args.frames < 2:
        parser.error('cells, samples and max-events must be positive; frames must be at least 2')
    domain = Circle(args.length)
    # transport: f(u)=-u moves data left at unit speed (exact, no events);
    # nonconvex: f(u)=u^3; pulse and sine: Burgers f(u)=u^2/2.
    f = {'transport': lambda u: -u, 'nonconvex': lambda u: u**3}.get(
        args.example, lambda u: 0.5 * u**2)
    flux = DiscreteFlux(f, np.linspace(-1, 1, args.nodes))
    if args.example in ('pulse', 'transport'):
        tracker = PeriodicFrontTracker.from_values(
            flux, [0, 0.25 * args.length, 0.75 * args.length], [0, 1, 0],
            domain=domain, max_events=args.max_events, record_history=not args.no_plot)
    else:
        tracker = PeriodicFrontTracker.from_function(
            flux, lambda x: np.sin(2 * np.pi * x / args.length),
            domain=domain, cells=args.cells, max_events=args.max_events, record_history=not args.no_plot)
    # Export the half-open periodic chart; plotting adds the identified endpoint.
    x = np.linspace(0, args.length, args.samples, endpoint=False)
    times = np.linspace(0, args.time, args.frames)
    profiles = np.empty((len(times), len(x)))
    diagnostics, front_rows, snapshots = [], [], []
    for k, time in enumerate(times):
        tracker.run(float(time))  # absolute time: continues the same solution
        profiles[k] = tracker.sample(x)
        snapshots.append(step_profile(tracker, (0, args.length)))
        diagnostics.append((time, len(tracker.fronts), tracker.event_count,
                            tracker.integral(), tracker.total_variation()))
        # Fronts live on a lift; wrap for export. iL/iR are node INDICES.
        front_rows.extend((time, domain.wrap(f.x), f.iL, f.iR, f.speed)
                          for f in tracker.fronts)
    args.output.mkdir(parents=True, exist_ok=True)
    stem = args.output / args.example
    np.savez(stem.with_suffix('.npz'), x=x, times=times, u=profiles,
             length=domain.length, origin=domain.origin,
             u_grid=flux.u_grid, f_values=flux.f_values)
    for suffix, header, rows in [
        ('diagnostics', ['time', 'fronts', 'events', 'mass', 'total_variation'], diagnostics),
        ('fronts', ['time', 'x', 'iL', 'iR', 'speed'], front_rows),
    ]:
        with (args.output / f'{args.example}_{suffix}.csv').open('w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(header)
            writer.writerows(rows)
    if not args.no_plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
        plot_x = np.r_[x, args.length]
        plot_values = np.c_[profiles, profiles[:, 0]]
        if times[-1] > 0:
            history = History(plot_x, times, plot_values, tracker.segments, snapshots)
        else:
            history = History(plot_x, times[:1], plot_values[:1], profiles=snapshots[:1])
        for k in sorted(set([0, len(history.times)//2, len(history.times)-1])):
            e, v = history.profiles[k]
            axes[0].stairs(v, e, label=f't={history.times[k]:g}')
        axes[0].set(xlabel='Periodic coordinate x', ylabel='u', xlim=(0, args.length))
        axes[0].legend()
        if len(history.times) > 1:
            plot_spacetime(history, ax=axes[1])
        else:
            plot_profile(history, ax=axes[1])
        fig.suptitle(f'{args.example} on S¹, circumference {args.length:g}')
        fig.savefig(stem.with_suffix('.png'), dpi=160)
        plt.close(fig)
    mass = np.array(diagnostics)[:, 3]
    print(f'Saved {args.example} to {args.output.resolve()}')
    print(f'Events: {tracker.event_count}; maximum mass drift: {np.max(np.abs(mass - mass[0])):.3g}')


if __name__ == '__main__':
    main()
