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
    u, f = flux.u_grid, flux.f_values
    a, b, c = slice(None, -2), slice(1, -1), slice(2, None)
    rise, run = f[c] - f[a], u[c] - u[a]
    gap = (f[b] - f[a]) - rise * ((u[b] - u[a]) / run)
    u_scale = max(abs(u[0]), abs(u[-1]))
    scale = (np.maximum(np.maximum(np.abs(f[a]), np.abs(f[b])), np.abs(f[c]))
             + np.abs(rise / run) * u_scale)
    allowance = _COLLINEAR_FACTOR * scale
    interior = np.arange(1, len(u) - 1)
    cached = (interior[gap <= allowance], interior[gap >= -allowance])
    flux._hull_candidate_cache = cached
    return cached


def solve_riemann(iL, iR, flux: DiscreteFlux) -> list[Wave]:
    """Return (left index, right index, RH speed) waves in spatial order.

    The hull is a stack: each node is appended once and removed at most once.
    A node is removed unless it lies strictly on the envelope side of the
    chord joining its neighbours, by more than a roundoff allowance. Hence
    collinear nodes never split one chord into several co-moving fronts, and
    returned speeds strictly increase. Only the endpoints and the candidate
    nodes between them are visited (see _hull_candidates): O(log n + k) work
    for k candidates, instead of O(abs(iR-iL)).
    """
    iL, iR = flux.validate_index(iL), flux.validate_index(iR)
    if iL == iR:
        return []  # Identical states have no discontinuity.

    # +1: keep nodes strictly below chords (lower convex envelope, iL < iR).
    # -1: keep nodes strictly above chords (upper concave envelope).
    side = 1.0 if iL < iR else -1.0
    states, values = flux.u_grid, flux.f_values
    # The grid is sorted, so its largest magnitude sits at an end. Node
    # placement errors (for example from linspace) scale with it.
    u_scale = max(abs(states[0]), abs(states[-1]))
    hull: list[int] = []

    # Always build in increasing STATE order; reverse for a downward jump later.
    # The endpoints were validated above and all visited nodes lie between
    # them, so indices are not re-validated inside this loop.
    lo, hi = min(iL, iR), max(iL, iR)
    candidates = _hull_candidates(flux)[0 if side > 0 else 1]
    first = int(np.searchsorted(candidates, lo, side="right"))
    last = int(np.searchsorted(candidates, hi, side="left"))
    for c in [lo, *candidates[first:last].tolist(), hi]:
        while len(hull) >= 2:
            a, b = hull[-2], hull[-1]
            rise, run = values[c] - values[a], states[c] - states[a]
            # Vertical offset of node b from the chord a-c (in flux units),
            # i.e. the orientation of (a, b, c) divided by the positive run.
            gap = (values[b] - values[a]) - rise * ((states[b] - states[a]) / run)
            scale = (max(abs(values[a]), abs(values[b]), abs(values[c]))
                     + abs(rise / run) * u_scale)
            if side * gap < -_COLLINEAR_FACTOR * scale:
                break  # b is strictly on the envelope side: keep it.
            # b lies on or beyond the chord a-c: drop it and retry from a.
            hull.pop()
        hull.append(c)

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
