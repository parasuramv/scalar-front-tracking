"""Event-driven scalar front tracking on R with constant exterior states.

The evolving solution is represented entirely by ordered fronts and a left
far-field state. There is no spatial evolution mesh or CFL time step. Sampling
and plotting grids observe the solution without changing its discretization.
"""

from dataclasses import replace
import math

import numpy as np

from .fronts import initial_fronts
from .interactions import next_interaction, resolve_interaction

# Relative allowance for roundoff when identifying a multi-front collision.
_ROUNDOFF_FACTOR = 16 * np.finfo(float).eps


class FrontTracker:
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
        # Entries are (x_start, t_start, x_end, t_end), suitable for plotting.
        # Disable recording for long runs when only final states are required.
        self.segments = []

    @classmethod
    def from_values(cls, flux, positions, values, **kwargs):
        """Map physical values to nearest flux nodes and initialize the tracker."""
        return cls(flux, positions, flux.state_index(values), **kwargs)

    @property
    def fronts(self):
        """Live spatially ordered front list. Use snapshot() for an editable copy."""
        return self.collection.fronts

    def _advance(self, dt: float) -> None:
        """Advance by an internally validated increment, optionally recording paths."""
        if dt == 0:
            return  # Simultaneous events require no motion or history entries.
        end_time = self.time + dt
        if self.record_history:
            # Record and translate in the same pass, calculating each endpoint
            # once. This avoids the former second traversal of all fronts.
            append = self.segments.append
            for front in self.fronts:
                end_x = front.x + front.speed * dt
                append((front.x, self.time, end_x, end_time))
                front.x = end_x
        else:
            self.collection.advance(dt)
        self.time = end_time

    def _resolve_cluster(self, pair_index: int) -> None:
        """Replace the full cluster surrounding an already advanced colliding pair."""
        fronts = self.fronts
        x = 0.5 * (fronts[pair_index].x + fronts[pair_index + 1].x)
        first, last = pair_index, pair_index + 1
        tolerance = self.position_tol + _ROUNDOFF_FACTOR * max(1.0, abs(x))

        # Compare every member to the event point, not just its neighbor, to
        # avoid chaining many small gaps into an arbitrarily wide cluster.
        while first > 0 and abs(fronts[first - 1].x - x) <= tolerance:
            first -= 1
        while last + 1 < len(fronts) and abs(fronts[last + 1].x - x) <= tolerance:
            last += 1

        outgoing = resolve_interaction(fronts[first], fronts[last], self.flux)
        for front in outgoing:
            front.x = x  # All waves in the new Riemann fan share one origin.
        fronts[first:last + 1] = outgoing
        self.event_count += 1

    def run(self, final_time: float):
        """Advance to absolute final_time, resolving events exactly at that time.

        Returns self for chaining. Calling again continues the current solution;
        going backward is rejected. An event-limit error leaves the state at
        its last completed event so it can be inspected or resumed.
        """
        if not math.isfinite(final_time) or final_time < self.time:
            raise ValueError("final_time must be finite and at least the current time")
        while True:
            dt, pair_index = next_interaction(self.fronts)
            remaining = final_time - self.time
            if pair_index is None or dt > remaining:
                # No interaction before the observation time: pure translation.
                self._advance(remaining)
                self.time = float(final_time)
                return self
            if self.event_count >= self.max_events:
                raise RuntimeError("Event limit reached; inspect data or raise max_events")

            self._advance(dt)
            self._resolve_cluster(pair_index)
            # Re-scan because the replacement fronts have new neighbors/speeds.
            # A separate simultaneous collision has dt=0 on the next iteration.

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
