"""Optional Matplotlib/ipywidgets views; importing this module needs only NumPy.

Colours represent u, never a front index or speed. Time rasters are sampled
observations (nearest-frame display), not additional evolution discretizations.
"""
from dataclasses import dataclass, field
import numpy as np


def _axis(values, name, minimum=2):
    a = np.array(values, dtype=float, copy=True)
    if a.ndim != 1 or len(a) < minimum or not np.isfinite(a).all() or np.any(np.diff(a) <= 0):
        raise ValueError(f'{name} must contain at least {minimum} finite, strictly increasing values')
    return a


@dataclass
class History:
    """Observed 1D profiles. Exact step snapshots are optional (edges, values)."""
    x: np.ndarray
    times: np.ndarray
    values: np.ndarray
    segments: list = field(default_factory=list)
    profiles: list = field(default_factory=list)
    cell_edges: object = None
    valid: object = None
    coordinate_label: str = "x"

    def __post_init__(self):
        self.x = _axis(self.x, 'x')
        self.times = _axis(self.times, 'times', 1)
        self.values = np.array(self.values, dtype=float, copy=True)
        if self.values.shape != (len(self.times), len(self.x)) or not np.isfinite(self.values).all():
            raise ValueError('values must be finite with shape (len(times), len(x))')


def step_profile(tracker, window):
    """Exact jump locations in a window, including periodic chart seams."""
    a, b = _axis(window, 'window')
    positions = np.array([f.x for f in tracker.fronts])
    if hasattr(tracker, 'domain'):
        domain = tracker.domain
        if a < domain.origin or b > domain.origin + domain.length:
            raise ValueError('Use one periodic chart for exact profiles')
        positions = domain.wrap(positions)
    edges = np.unique(np.r_[a, positions[(positions > a) & (positions < b)], b])
    # Right-continuous samples at left edges avoid midpoint rounding at tiny cells.
    return edges, np.asarray(tracker.sample(edges[:-1]))


def observe_tracker(tracker, x, times):
    """Advance a real-line/circle tracker and retain exact snapshots and samples.

    This consumes the tracker up to times[-1]; use a fresh tracker to start over.
    All input validation precedes advancement. History recording is optional.
    """
    x, times = _axis(x, 'x'), _axis(times, 'times', 1)
    if times[0] < tracker.time:
        raise ValueError('Observation times precede the current tracker time')
    step_profile(tracker, (x[0], x[-1]))  # Validate chart limits before advancing.
    profiles, values = [], []
    for t in times:
        tracker.run(float(t))
        values.append(tracker.sample(x))
        profiles.append(step_profile(tracker, (x[0], x[-1])))
    return History(x, times, values, list(getattr(tracker, 'segments', [])), profiles)


def color_norm(values, limits=None):
    """A stable colour scale over the entire run, including constant solutions."""
    from matplotlib.colors import Normalize
    if isinstance(values, History):
        samples = [values.values.ravel()]
        samples.extend(np.asarray(v) for _, v in values.profiles)
        values = np.concatenate(samples)
    lo, hi = (float(np.min(values)), float(np.max(values))) if limits is None else limits
    if not np.isfinite([lo, hi]).all() or lo > hi:
        raise ValueError('Colour limits must be finite and ordered')
    if lo == hi:
        pad = max(1, abs(lo)) * .01
        lo, hi = lo - pad, hi + pad
    return Normalize(lo, hi)


def plot_profile(history, index=-1, *, ax=None, cmap='viridis', norm=None):
    """Render a snapshot, using exact front positions when available."""
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    ax = plt.subplots(figsize=(7, 3.5))[1] if ax is None else ax
    norm = color_norm(history) if norm is None else norm
    if history.profiles:
        edges, values = history.profiles[index]
    else:
        edges, values = history.x, history.values[index, :-1]
    ax.stairs(values, edges, color='0.25', linewidth=.8)
    lines = [[(a, u), (b, u)] for a, b, u in zip(edges[:-1], edges[1:], values)]
    collection = LineCollection(lines, cmap=cmap, norm=norm, linewidths=2.5)
    collection.set_array(np.asarray(values))
    ax.add_collection(collection)
    if history.valid is not None:
        for a, b, ok in zip(edges[:-1], edges[1:], history.valid[index]):
            if not ok:
                ax.axvspan(a, b, facecolor='none', edgecolor='0.5', hatch='///', linewidth=0)
    pad = .08 * (norm.vmax - norm.vmin)
    ax.set(xlim=(history.x[0], history.x[-1]), ylim=(norm.vmin-pad, norm.vmax+pad),
           xlabel=history.coordinate_label, ylabel='u', title=f'Profile at t = {history.times[index]:g}')
    return ax


def _hatch_cells(ax, xedges, yedges, invalid):
    """Hatch only the specified cells; keep masking independent of colour arrays."""
    from matplotlib.collections import PolyCollection
    rows, cols = np.nonzero(invalid)
    quads = [[(xedges[j], yedges[i]), (xedges[j+1], yedges[i]),
              (xedges[j+1], yedges[i+1]), (xedges[j], yedges[i+1])]
             for i, j in zip(rows, cols)]
    overlay = PolyCollection(quads, facecolors='none', edgecolors='0.45',
                             linewidths=0, hatch='///')
    ax.add_collection(overlay)


def plot_spacetime(history, *, ax=None, cmap='viridis', norm=None, fronts=True, colorbar=True):
    """Sampled colour field with exact recorded front segments overlaid."""
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    if len(history.times) < 2:
        raise ValueError('Space–time rendering needs at least two distinct times')
    ax = plt.subplots(figsize=(7, 4.5))[1] if ax is None else ax
    norm = color_norm(history) if norm is None else norm
    if history.cell_edges is None:
        mesh = ax.pcolormesh(history.x, history.times, history.values,
                            shading='nearest', cmap=cmap, norm=norm, rasterized=True)
    else:
        t = history.times
        te = np.r_[t[0], (t[:-1]+t[1:])/2, t[-1]]
        mesh = ax.pcolormesh(history.cell_edges, te, history.values[:, :-1],
                            shading='flat', cmap=cmap, norm=norm, rasterized=True)
        if history.valid is not None and not history.valid.all():
            invalid = ~history.valid
            _hatch_cells(ax, history.cell_edges, te, invalid)
    if fronts and history.segments:
        ax.add_collection(LineCollection(np.asarray(history.segments).reshape(-1, 2, 2),
                                        colors='0.15', linewidths=.45, alpha=.65))
    ax.set(xlim=(history.x[0], history.x[-1]), ylim=(history.times[0], history.times[-1]),
           xlabel=history.coordinate_label, ylabel='t', title='Space–time: conserved quantity u')
    if colorbar:
        ax.figure.colorbar(mesh, ax=ax, label='u')
    return ax


def dashboard(history, *, cmap='viridis', fronts=True):
    """Time slider, play/pause, colour-map selector and front visibility control.

    Only cached observations are displayed; changing the slider never reruns the
    solver. Static plots remain available when widgets are not installed.
    """
    import ipywidgets as widgets
    import matplotlib.pyplot as plt
    from IPython.display import display
    frame = widgets.IntSlider(min=0, max=len(history.times)-1, value=0,
                              description='Frame', continuous_update=False)
    play = widgets.Play(min=0, max=len(history.times)-1, interval=150)
    link = widgets.jslink((play, 'value'), (frame, 'value'))
    palette = widgets.Dropdown(options=['viridis', 'cividis', 'coolwarm', 'magma'],
                               value=cmap, description='Colours')
    paths = widgets.Checkbox(value=fronts, description='Front paths')
    output = widgets.Output()
    norm = color_norm(history)

    def draw(change=None):
        with output:
            output.clear_output(wait=True)
            fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
            plot_profile(history, frame.value, ax=axes[0], cmap=palette.value, norm=norm)
            if len(history.times) > 1:
                plot_spacetime(history, ax=axes[1], cmap=palette.value, norm=norm, fronts=paths.value)
                axes[1].axhline(history.times[frame.value], color='white', linewidth=1)
            else:
                axes[1].set_visible(False)
            display(fig)
            plt.close(fig)
    for widget in (frame, palette, paths):
        widget.observe(draw, names='value')
    panel = widgets.VBox([widgets.HBox([play, frame]), widgets.HBox([palette, paths]), output])
    panel._fronttrack_link = link  # Retain the widget link for this panel's lifetime.
    draw()
    return panel


@dataclass
class GridHistory:
    """1D/2D cell averages, edges and boundary-validity masks at saved times."""
    edges: tuple
    times: np.ndarray
    values: np.ndarray
    valid: np.ndarray

    def line(self, *, axis=0, coordinate=None):
        """A 1D history or a fixed-coordinate slice of a 2D history (no averaging)."""
        if axis not in range(len(self.edges)):
            raise ValueError('Invalid slice axis')
        if len(self.edges) == 1:
            values = self.values
            valid = self.valid
        else:
            other = 1-axis
            e = self.edges[other]
            coordinate = (e[0]+e[-1])/2 if coordinate is None else coordinate
            if not np.isfinite(coordinate) or not e[0] <= coordinate <= e[-1]:
                raise ValueError('Slice coordinate lies outside the domain')
            j = min(len(e)-2, np.searchsorted(e, coordinate, side='right')-1)
            values = self.values[:, :, j] if axis == 0 else self.values[:, j, :]
            valid = self.valid[:, :, j] if axis == 0 else self.valid[:, j, :]
        e = self.edges[axis]
        # Repeat last cell at final edge solely for display, not a boundary state.
        return History(e, self.times, np.c_[values, values[:, -1]],
                       profiles=[(e, row) for row in values], cell_edges=e, valid=valid,
                       coordinate_label="x" if axis == 0 else "y")


def observe_grid(solver, times, **run_options):
    """Record a splitting/torus solver in 1D or 2D, advancing it in place.

    Output times may shorten macro steps and change remapping error. They are
    part of a splitting experiment, unlike observation of event-driven fronts.
    """
    times = _axis(times, 'times', 1)
    if solver.grid.ndim not in (1, 2) or times[0] < solver.time:
        raise ValueError('Require a 1D/2D solver and times >= its current time')
    values, valid = [], []
    for t in times:
        solver.run(float(t), **run_options)
        values.append(solver.values)
        valid.append(solver.valid_mask)
    return GridHistory(tuple(e.copy() for e in solver.grid.edges), times,
                       np.asarray(values), np.asarray(valid))


def plot_grid_snapshot(history, index=-1, *, ax=None, cmap='viridis', norm=None):
    """Cell-average field; hatch cells whose dependence reaches artificial edges."""
    import matplotlib.pyplot as plt
    norm = color_norm(history.values) if norm is None else norm
    if len(history.edges) == 1:
        return plot_profile(history.line(), index, ax=ax, cmap=cmap, norm=norm)
    ax = plt.subplots(figsize=(5, 4))[1] if ax is None else ax
    x, y = history.edges
    mesh = ax.pcolormesh(x, y, history.values[index].T, cmap=cmap, norm=norm, shading='flat')
    invalid = (~history.valid[index]).T
    if invalid.any():
        _hatch_cells(ax, x, y, invalid)
    ax.set(xlabel='x', ylabel='y', title=f'Cell averages at t = {history.times[index]:g}')
    ax.set_aspect('equal')
    ax.figure.colorbar(mesh, ax=ax, label='u')
    return ax
