"""Resolve one initial jump using an entropy envelope of the polygonal flux.

An increasing jump uses the lower convex envelope; a decreasing jump uses the
upper concave envelope. Envelope edges become moving fronts. In both cases,
the returned fronts are ordered from left to right in physical space.
"""

import math
from itertools import pairwise

import numpy as np

from .flux import DiscreteFlux

# Each wave stores node indices, followed by its physical propagation speed.
Wave = tuple[int, int, float]

# Collinearity allowance, relative to the magnitudes entering the test. Nodes
# within this vertical distance of a chord are treated as lying on it. Measured
# worst case for mathematically collinear u**3 nodes is about 1.3 eps; genuine
# gaps on grids up to 2001 nodes are above 1e9 eps.
_COLLINEAR_FACTOR = 16 * np.finfo(float).eps


def _local_gaps(flux: DiscreteFlux):
    """Vertical offset of each interior node from its neighbours' chord, and
    the roundoff allowance for that offset (same formula as the hull stack)."""
    u, f = flux.u_grid, flux.f_values
    a, b, c = slice(None, -2), slice(1, -1), slice(2, None)
    rise, run = f[c] - f[a], u[c] - u[a]
    gap = (f[b] - f[a]) - rise * ((u[b] - u[a]) / run)
    u_scale = max(abs(u[0]), abs(u[-1]))
    scale = (np.maximum(np.maximum(np.abs(f[a]), np.abs(f[b])), np.abs(f[c]))
             + np.abs(rise / run) * u_scale)
    return gap, _COLLINEAR_FACTOR * scale


def _hull_candidates(flux: DiscreteFlux):
    """Interior nodes that can be vertices of a lower/upper envelope.

    A vertex of the lower convex envelope of any node range lies strictly
    below the chord joining ANY two range nodes on either side of it, in
    particular its two grid neighbours. So a node clearly above the chord of
    its neighbours (locally concave by more than the roundoff allowance) is
    never a lower-envelope vertex, and symmetrically for the upper envelope.
    Nodes within the allowance of collinear stay candidates; the stack in
    solve_riemann applies its own allowance to them exactly as before.

    Returns two sorted index arrays (lower, upper), computed once per flux
    in O(n) and cached on it. For a convex flux the upper array is empty, so
    a shock between any two states costs O(log n) instead of O(n).
    """
    cached = getattr(flux, "_hull_candidate_cache", None)
    if cached is not None:
        return cached
    gap, allowance = _local_gaps(flux)
    interior = np.arange(1, len(flux.u_grid) - 1)
    cached = (interior[gap <= allowance], interior[gap >= -allowance])
    flux._hull_candidate_cache = cached
    return cached


def _strict_runs(flux: DiscreteFlux):
    """Where runs of strictly convex (lower) / strictly concave (upper) nodes end.

    A node is strict when the hull stack would keep it between its two grid
    neighbours: it is beyond their chord by more than the allowance. Returns
    (lower_end, upper_end): lists where entry i is the last index of the
    maximal run of consecutive strict nodes containing node i, or -1 if node
    i is not strict. Also caches the grid as Python floats for the stack.
    Computed once per flux in O(n).
    """
    cached = getattr(flux, "_strict_run_cache", None)
    if cached is not None:
        return cached
    gap, allowance = _local_gaps(flux)
    n = len(flux.u_grid)
    index = np.arange(n)
    ends = []
    for strict in (gap < -allowance, gap > allowance):
        mask = np.zeros(n, dtype=bool)
        mask[1:-1] = strict
        # First non-strict index at or after each node, by a reversed minimum.
        stop = np.minimum.accumulate(np.where(mask, n, index)[::-1])[::-1]
        ends.append(np.where(mask, stop - 1, -1).tolist())
    cached = (ends[0], ends[1], flux.u_grid.tolist(), flux.f_values.tolist())
    flux._strict_run_cache = cached
    return cached


# Strict runs at least this long are inserted with binary searches instead of
# node by node. Shorter runs gain nothing from the extra bookkeeping.
_MIN_RUN = 8


def solve_riemann(iL, iR, flux: DiscreteFlux) -> list[Wave]:
    """Return (left index, right index, RH speed) waves in spatial order.

    The envelope is built by a monotone stack. A node is removed unless it
    lies strictly on the envelope side of the chord joining its neighbours,
    by more than a roundoff allowance. Hence collinear nodes never split one
    chord into several co-moving fronts, and returned speeds strictly
    increase. Only the endpoints and the candidate nodes between them are
    visited (see _hull_candidates).

    The stack holds runs of consecutive grid nodes, not single nodes. On a
    run of strictly convex nodes (strictly concave for the upper envelope)
    the envelope touches the flux along one contiguous piece, joined to the
    rest by chords. The ends of that piece are found by binary search, so a
    run costs O(log^2 n) instead of one push and pop per node; the result is
    the same as pushing the nodes one at a time. A solve costs about
    O(r log^2 n + k) for r runs and k returned waves: O(log n) for a convex
    shock, and no longer O(n) for a shock that bridges an inflection.
    """
    iL, iR = flux.validate_index(iL), flux.validate_index(iR)
    if iL == iR:
        return []  # Identical states have no discontinuity.

    # +1: keep nodes strictly below chords (lower convex envelope, iL < iR).
    # -1: keep nodes strictly above chords (upper concave envelope).
    side = 1.0 if iL < iR else -1.0
    lower_end, upper_end, states, values = _strict_runs(flux)
    run_end = lower_end if side > 0 else upper_end
    # The grid is sorted, so its largest magnitude sits at an end. Node
    # placement errors (for example from linspace) scale with it.
    u_scale = max(abs(states[0]), abs(states[-1]))
    tolerance = -_COLLINEAR_FACTOR

    def keep(a, b, c):
        """True if node b stays between a and c (a < b < c in state)."""
        rise, run = values[c] - values[a], states[c] - states[a]
        # Vertical offset of node b from the chord a-c (in flux units),
        # i.e. the orientation of (a, b, c) divided by the positive run.
        gap = (values[b] - values[a]) - rise * ((states[b] - states[a]) / run)
        scale = (max(abs(values[a]), abs(values[b]), abs(values[c]))
                 + abs(rise / run) * u_scale)
        return side * gap < tolerance * scale

    # The envelope so far: runs starts[k]..ends[k] of consecutive grid nodes,
    # every node of every run being a current vertex.
    starts: list[int] = []
    ends: list[int] = []

    def tangent(c):
        """Where the stack would be cut before pushing c: (run k, new end).

        Same decisions as popping one node at a time; inside a run the kept
        nodes form a prefix (the run is convex), found by binary search.
        """
        k = len(ends) - 1
        while True:
            s, e = starts[k], ends[k]
            if e > s:
                if keep(e - 1, e, c):
                    return k, e
                if e - 1 > s and keep(s, s + 1, c):
                    lo, hi = s + 1, e - 1  # keep() is true at lo, false past hi
                    while lo < hi:
                        mid = (lo + hi + 1) // 2
                        if keep(mid - 1, mid, c):
                            lo = mid
                        else:
                            hi = mid - 1
                    return k, lo
            if k == 0:
                return 0, s  # A single vertex is never removed.
            if keep(ends[k - 1], s, c):
                return k, s
            k -= 1

    def cut(k, end):
        del starts[k + 1:], ends[k + 1:]
        ends[k] = end

    def push(c):
        if starts:
            cut(*tangent(c))
        starts.append(c)
        ends.append(c)

    def push_run(p, q):
        """Push the strict nodes p..q, as the one-at-a-time stack would.

        Node j < q survives the arrival of j + 1 exactly when it is kept
        between its own tangent point on the stack and j + 1. That holds
        from some first node J on (monotone along a convex run); nodes
        before J are each removed by their successor. The stack then ends
        with the cut for J followed by the whole run J..q.
        """
        lo, hi = p, q
        while lo < hi:
            mid = (lo + hi) // 2
            if keep(tangent(mid)[1], mid, mid + 1):
                hi = mid
            else:
                lo = mid + 1
        cut(*tangent(lo))
        starts.append(lo)
        ends.append(q)

    # Always build in increasing STATE order; reverse for a downward jump later.
    # The endpoints were validated above and all visited nodes lie between
    # them, so indices are not re-validated inside this loop.
    lo, hi = min(iL, iR), max(iL, iR)
    candidates = _hull_candidates(flux)[0 if side > 0 else 1]
    position = int(np.searchsorted(candidates, lo, side="right"))
    last = int(np.searchsorted(candidates, hi, side="left"))
    push(lo)
    while position < last:
        c = int(candidates[position])
        q = min(run_end[c], hi - 1)
        if q - c + 1 >= _MIN_RUN:
            # Strict nodes are candidates, so the run is contiguous here too.
            push_run(c, q)
            position += q - c + 1
        else:
            push(c)
            position += 1
    push(hi)

    hull = [node for s, e in zip(starts, ends) for node in range(s, e + 1)]
    # RH speeds, always computed in increasing state order so a chord's speed
    # does not depend on the direction of the jump.
    speeds = [float((values[b] - values[a]) / (states[b] - states[a]))
              for a, b in pairwise(hull)]
    # On an upper hull, speeds decrease in state order. Reversing gives
    # increasing speeds in physical space and the requested iL -> iR jump.
    if side < 0:
        hull.reverse()
        speeds.reverse()
    return [(a, b, speed) for (a, b), speed in zip(pairwise(hull), speeds)]


def wave_direction(iL, iR) -> int:
    """Sign of the state jump (+1 increasing, -1 decreasing, 0 absent)."""
    return int(iR > iL) - int(iR < iL)


def enforce_entropy(waves):
    """Validate wave continuity and ordering; return the unchanged wave list.

    Kept for compatibility. This does not construct or repair an entropy hull:
    increasing speeds alone cannot establish admissibility for a nonconvex flux.
    Use ``solve_riemann`` for the actual entropy construction.
    """
    for left, right in pairwise(waves):
        if left[1] != right[0] or left[2] > right[2]:
            raise ValueError("Waves must be connected with nondecreasing speeds")
    if any(not math.isfinite(wave[2]) for wave in waves):
        raise ValueError("Wave speeds must be finite")
    return waves
