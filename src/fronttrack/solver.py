"""Event-driven scalar front tracking on R with constant exterior states.

The evolving solution is represented entirely by ordered fronts and a left
far-field state. There is no spatial evolution mesh or CFL time step. Sampling
and plotting grids observe the solution without changing its discretization.

The event loop itself (anchored fronts and a priority queue of predicted
collisions) lives in events.py and is shared with the circle driver.
"""

from dataclasses import replace
import math

import numpy as np

from .events import AnchoredEventDriver
from .fronts import initial_fronts
from .interactions import resolve_interaction

# Relative allowance for roundoff when identifying a multi-front collision.
_ROUNDOFF_FACTOR = 16 * np.finfo(float).eps


class FrontTracker(AnchoredEventDriver):
    """Evolve node-valued step data by repeatedly solving local Riemann problems.

    Supply N jump positions and N+1 integer state indices, or use ``from_values``
    to quantize physical values. ``run(T)`` advances in place to absolute time T.
    ``position_tol`` is in spatial units; nearby fronts at a triggered collision
    are combined. ``max_events`` limits cumulative collision clusters.
    """

    def __init__(self, flux, positions, states, *, position_tol=1e-12,
                 max_events=100000, record_history=True):
        self.flux = flux
        states = list(states)
        self.collection = initial_fronts(positions, states, flux)
        self.left_state, self.right_state = states[0], states[-1]
        if not math.isfinite(position_tol) or position_tol < 0:
            raise ValueError("position_tol must be finite and nonnegative")
        if not isinstance(max_events, int) or max_events < 1:
            raise ValueError("max_events must be a positive integer")
        self.position_tol = position_tol
        self.max_events = max_events
        self.record_history = record_history
        self.time = 0.0
        self.event_count = 0
        # Entries are (x_start, t_start, x_end, t_end), suitable for plotting:
        # one per straight piece of a path, cut at its collision and at the
        # end of each run() call. Disable when only final states are needed.
        self.segments = []

    @classmethod
    def from_values(cls, flux, positions, values, **kwargs):
        """Map physical values to nearest flux nodes and initialize the tracker."""
        return cls(flux, positions, flux.state_index(values), **kwargs)

    def _pair(self, front):
        """(front, right neighbour, gap now) for the pair front owns, or None."""
        right = front.next
        if right is None:
            return None
        return front, right, self._position(right) - self._position(front)

    def _record_path(self, x0, t0, speed, t1):
        """One straight history segment (x_start, t_start, x_end, t_end)."""
        self.segments.append((x0, t0, x0 + speed * (t1 - t0), t1))

    def _resolve_cluster(self, front) -> None:
        """Replace the full cluster around front and its right neighbour."""
        position = self._position
        x = 0.5 * (position(front) + position(front.next))
        first, last = front, front.next
        tolerance = self.position_tol + _ROUNDOFF_FACTOR * max(1.0, abs(x))

        # Compare every member to the event point, not just its neighbor, to
        # avoid chaining many small gaps into an arbitrarily wide cluster.
        while first.prev is not None and abs(position(first.prev) - x) <= tolerance:
            first = first.prev
        while last.next is not None and abs(position(last.next) - x) <= tolerance:
            last = last.next

        outgoing = resolve_interaction(first, last, self.flux)
        for new in outgoing:
            # All waves in the new Riemann fan share one origin and anchor.
            new.x, new.t0 = x, self.time
        before, after = first.prev, last.next
        cluster = self._span(first, last)
        for old in cluster:
            self._retire(old)
        # Relink only around the cluster; no other front moves.
        self._chain([before, *outgoing, after])
        if before is None:
            self._head = outgoing[0] if outgoing else after
        self._count += len(outgoing) - len(cluster)
        self.event_count += 1
        # Only the new fronts and the left outer neighbour have new right
        # neighbours; every other queued prediction is still valid.
        self._refresh([before, *outgoing])

    def snapshot(self):
        """Independent copies of front geometry; this is not a full restart file."""
        return [replace(front) for front in self.fronts]

    def sample(self, x):
        """Right-continuous point values, with the shape of x; no time advancement."""
        x = np.asarray(x, dtype=float)
        if not np.all(np.isfinite(x)):
            raise ValueError("Sample coordinates must be finite")
        positions = [front.x for front in self.fronts]
        states = [self.left_state] + [front.iR for front in self.fronts]
        # The insertion index counts jumps at or left of x. side='right' also
        # crosses every coincident initial front, selecting the outer right state.
        indices = np.searchsorted(positions, x, side="right")
        values = self.flux.u_grid[np.asarray(states)[indices]]
        return float(values) if x.ndim == 0 else values

    def integral(self, a: float, b: float) -> float:
        """Integrate the step solution on [a,b] in one pass over its fronts.

        Each region contributes width * state. No midpoint sampling, binary
        searches, or quadrature grid is needed. Coincident fronts contribute
        zero width and still update the state correctly.
        """
        if not math.isfinite(a) or not math.isfinite(b) or a >= b:
            raise ValueError("Require finite a < b")

        def contributions():
            position, state = a, self.left_state
            grid = self.flux.u_grid
            for front in self.fronts:
                if front.x <= a:
                    state = front.iR  # Find the state at the left window edge.
                    continue
                if front.x >= b:
                    break
                yield (front.x - position) * grid[state]
                position, state = front.x, front.iR
            yield (b - position) * grid[state]

        # fsum reduces cancellation for profiles with both positive and negative
        # mass, without allocating an array of all interval contributions.
        return math.fsum(contributions())

    def total_variation(self) -> float:
        """Sum absolute jump strengths, including jumps to exterior states."""
        grid = self.flux.u_grid
        return math.fsum(abs(grid[front.iR] - grid[front.iL]) for front in self.fronts)
