"""Event-driven scalar front tracking on the circle S^1 = R/(L Z).

Same equation as solver.py, u_t + f(u)_x = 0, but on a compact domain with no
boundary: a front leaving one end of the chart [origin, origin + L) re-enters
at the other and can collide with fronts there. This is intrinsic 1D evolution,
not a 2D PDE restricted to a curve, and there are no far-field states.

Fronts live on an ordered lift spanning at most one circumference. A common
translation by an integer number of periods keeps coordinates small; individual
fronts are never independently wrapped and sorted during evolution. The event
loop (anchored fronts, priority queue) is shared with solver.py via events.py.
"""
from dataclasses import dataclass, replace
import math

import numpy as np

from .events import AnchoredEventDriver
from .fronts import Front, FrontCollection
from .interactions import resolve_interaction
from .riemann import solve_riemann


@dataclass(frozen=True)
class Circle:
    """Flat circle R/(length Z), represented on [origin, origin + length)."""
    length: float = 1.0
    origin: float = 0.0

    def __post_init__(self):
        if not math.isfinite(self.length) or self.length <= 0:
            raise ValueError("length must be positive and finite")
        if (not math.isfinite(self.origin) or not math.isfinite(self.origin + self.length)
                or self.origin + self.length <= self.origin):
            raise ValueError("origin and endpoint must be finite and distinguishable")

    @property
    def dimension(self):
        return 1

    @property
    def volume(self):
        return self.length

    def wrap(self, points):
        """Map finite coordinates into [origin, origin+length), same shape."""
        points = np.asarray(points, dtype=float)
        if not np.all(np.isfinite(points)):
            raise ValueError("coordinates must be finite")
        result = self.origin + np.remainder(points - self.origin, self.length)
        # remainder can return exactly `length` for tiny negative inputs
        # (e.g. -1e-17 % 1 == 1.0 in floating point); fold that onto the origin.
        result = np.where(result >= self.origin + self.length, self.origin, result)
        return float(result) if points.ndim == 0 else result


class PeriodicFrontTracker(AnchoredEventDriver):
    """Evolve u_t + f(u)_x = 0 on Circle.

    N breakpoints require N states: state[j] occupies the positively oriented
    arc from positions[j] to positions[j+1], including the final wraparound arc.
    For a constant solution, use no breakpoints and one state. All states are
    flux-node indices; from_values quantizes physical values. run(T) uses
    absolute time. Live fronts are internal; snapshot() returns copies.
    """
    _event_limit_message = "Event limit reached; inspect or increase max_events"
    _cyclic = True

    def __init__(self, flux, positions, states, *, domain=None,
                 position_tol=1e-12, max_events=100000, record_history=False):
        self.domain = Circle() if domain is None else domain
        if not isinstance(self.domain, Circle):
            raise TypeError("PeriodicFrontTracker currently supports Circle only")
        self.flux = flux
        p = np.asarray(positions, dtype=float)
        states = [flux.validate_index(s) for s in states]
        if p.ndim != 1 or not np.all(np.isfinite(p)) or np.any(np.diff(p) <= 0):
            raise ValueError("positions must be finite and strictly increasing")
        if np.any(p < self.domain.origin) or np.any(p >= self.domain.origin + self.domain.length):
            raise ValueError("positions must lie in the half-open domain interval")
        if len(states) != (len(p) if len(p) else 1):
            raise ValueError("Need N positions and N states, or no positions and one constant state")
        if not math.isfinite(position_tol) or not 0 <= position_tol < self.domain.length / 4:
            raise ValueError("position_tol must be finite and in [0, length/4)")
        if isinstance(max_events, bool) or not isinstance(max_events, int) or max_events < 1:
            raise ValueError("max_events must be a positive integer")
        self.position_tol, self.max_events = position_tol, max_events
        self.time, self.event_count = 0.0, 0
        self.record_history = record_history
        self.segments = []
        # With no fronts the circle carries one constant; remember which. It is
        # refreshed at every collision so full annihilation leaves the right value.
        self._constant_state = states[0]
        # Breakpoint j separates states[j-1] (its left arc) from states[j]. For
        # j=0, Python's states[-1] is the final arc wrapping through the seam,
        # so every breakpoint, including the first, gets a genuine left state.
        # Fans from one breakpoint share its position, ordered by speed.
        self.collection = FrontCollection([
            Front(float(x), a, b, speed)
            for j, x in enumerate(p)
            for a, b, speed in solve_riemann(states[j - 1], states[j], flux)
        ])

    @classmethod
    def from_values(cls, flux, positions, values, **kwargs):
        """Quantize physical values to nearest flux nodes (no clipping)."""
        return cls(flux, positions, flux.state_index(values), **kwargs)

    @classmethod
    def from_function(cls, flux, function, *, cells=100, domain=None, **kwargs):
        """Midpoint sample and quantize a function; these are not cell averages."""
        domain = Circle() if domain is None else domain
        if not isinstance(domain, Circle):
            raise TypeError("PeriodicFrontTracker currently supports Circle only")
        if isinstance(cells, bool) or not isinstance(cells, int) or cells < 1:
            raise ValueError("cells must be a positive integer")
        # Left cell edges only: the last cell ends at the seam, which is the
        # first edge again, so the endpoint must not be repeated.
        edges = domain.origin + np.arange(cells) * domain.length / cells
        values = np.asarray(function(edges + domain.length / (2 * cells)), dtype=float)
        if values.ndim == 0:
            values = np.full(cells, values)
        return cls.from_values(flux, edges, values, domain=domain, **kwargs)

    @property
    def fronts(self):
        """Live fronts in cyclic order on the current lift. Do not modify.

        During ``run`` this is a fresh list built from the links (see events.py).
        """
        return super().fronts

    def _pair(self, front):
        """(front, right neighbour, gap now) for the pair front owns, or None.

        Same closing-speed test as the real-line driver, but adjacency is
        cyclic: the last front's right neighbour is the first front (the
        head), one circumference further along. Only that wraparound pair
        needs +L: the lift is ordered and spans at most one circumference.
        """
        if self._count < 2:
            return None
        right = front.next
        gap = self._position(right) - self._position(front)
        return front, right, gap + (self.domain.length if right is self._head else 0.0)

    def _record_path(self, x0, t0, speed, t1):
        """Record a straight path, split at chart seams (no false diagonals)."""
        o, length, dt = self.domain.origin, self.domain.length, t1 - t0
        end = x0 + speed * dt
        cuts = [0.0, dt]
        if speed != 0:
            lo, hi = sorted((x0, end))
            for k in range(math.floor((lo - o) / length) + 1, math.ceil((hi - o) / length)):
                crossing = (o + k * length - x0) / speed
                if 0 < crossing < dt:
                    cuts.append(crossing)
        cuts.sort()
        for a, b in zip(cuts[:-1], cuts[1:]):
            shift = math.floor((x0 + speed * (a + b) / 2 - o) / length) * length
            self.segments.append((x0 + speed * a - shift, t0 + a,
                                  x0 + speed * b - shift, t0 + b))

    def _normalize(self):
        """Shift the whole lift by whole periods so front 0 is in the chart.

        The shift is the same for every front (an integer number of periods up
        to roundoff), so cyclic order and all separations are preserved; it
        only stops coordinates growing without bound over many circuits.
        """
        if self.fronts:
            first = self.fronts[0].x
            shift = first - self.domain.wrap(first)
            if shift:
                for front in self.fronts:
                    front.x -= shift

    def _resolve_cluster(self, front):
        """Replace the collision cluster containing front and its right neighbour.

        Runs at the collision time. As on the line, the new Riemann problem
        uses only the cluster's two exterior states. Positions are compared on
        one lift: stepping right from the last front onto the head adds L,
        stepping left from the head onto the last front subtracts L.

        A cluster that crosses the seam is expressed on the lift of its first
        (leftmost) front, and the head moves to the first front after it, so
        the new fan sits at the end of the cyclic order. That keeps the lift
        ordered without shifting any other front.
        """
        length, position, head, n = self.domain.length, self._position, self._head, self._count
        first, last = front, front.next
        first_shift, last_shift = 0.0, (length if last is head else 0.0)
        x = ((position(first) + first_shift) + (position(last) + last_shift)) / 2
        # The grouping tolerance is local to the circle, independent of how
        # many periods the fronts have travelled together.
        tol = self.position_tol + 16 * np.finfo(float).eps * length
        size = 2
        while size < n:
            shift = first_shift - length if first is head else first_shift
            if abs((position(first.prev) + shift) - x) > tol:
                break
            first, first_shift, size = first.prev, shift, size + 1
        while size < n:
            shift = last_shift + length if last.next is head else last_shift
            if abs((position(last.next) + shift) - x) > tol:
                break
            last, last_shift, size = last.next, shift, size + 1

        crosses_seam = first_shift != 0.0 or last_shift != 0.0
        if first_shift:
            x += length  # Re-express the event on the first front's lift.
        self._constant_state = first.iL
        outgoing = resolve_interaction(first, last, self.flux)
        for new in outgoing:
            new.x, new.t0 = x, self.time
        before, after = first.prev, last.next
        cluster = self._span(first, last)
        if crosses_seam:
            # Retire the head side first, then the tail side, in list order.
            k = next(i for i, old in enumerate(cluster) if old is head)
            cluster = cluster[k:] + cluster[:k]
        for old in cluster:
            self._retire(old)
        self._count += len(outgoing) - size
        self.event_count += 1
        if size == n:
            # The whole circle collided: the fan (possibly empty) is all there is.
            self._chain([*outgoing, *outgoing[:1]])
            self._head = outgoing[0] if outgoing else None
            self._refresh(outgoing)
            return
        self._chain([before, *outgoing, after])
        if crosses_seam:
            self._head = after
        elif first is head:
            self._head = outgoing[0] if outgoing else after
        # New fronts and the cyclic left neighbour own changed pairs.
        self._refresh([before, *outgoing])

    def sample(self, x):
        """Right-continuous values at any finite coordinates (wrapped), same shape."""
        x = np.asarray(self.domain.wrap(x))
        if not self.fronts:
            result = np.full(x.shape, self.flux.u_grid[self._constant_state])
        else:
            first = self.fronts[0].x
            # Represent each query on the fronts' lift [first, first+L). Then
            # front 0 is at or left of every query; the state right of the last
            # front at or left of the query is the answer. side='right' crosses
            # coincident fans, selecting their outermost right state.
            lifted = first + np.remainder(x - first, self.domain.length)
            positions = [front.x for front in self.fronts]
            indices = np.searchsorted(positions, lifted, side="right") - 1
            result = self.flux.u_grid[np.array([f.iR for f in self.fronts])[indices]]
        return float(result) if x.ndim == 0 else result

    def snapshot(self):
        """Copied fronts on the current ordered lift (not necessarily in [0,L))."""
        return [replace(front) for front in self.fronts]

    def integral(self):
        """Exact step-function mass over the whole circle, up to roundoff."""
        if not self.fronts:
            return float(self.domain.length * self.flux.u_grid[self._constant_state])
        fronts = self.fronts
        # Arc from each front to the next carries that front's right state;
        # the last arc wraps to the first front one circumference later.
        return math.fsum(
            ((fronts[i + 1].x if i + 1 < len(fronts) else fronts[0].x + self.domain.length) - f.x)
            * self.flux.u_grid[f.iR] for i, f in enumerate(fronts)
        )

    def total_variation(self):
        """Sum of |jumps| around the whole circle (nonincreasing in exact arithmetic)."""
        return math.fsum(abs(self.flux.u_grid[f.iR] - self.flux.u_grid[f.iL]) for f in self.fronts)
