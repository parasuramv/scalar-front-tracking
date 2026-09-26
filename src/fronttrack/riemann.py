"""Resolve one initial jump using an entropy envelope of the polygonal flux.

An increasing jump uses the lower convex envelope; a decreasing jump uses the
upper concave envelope. Envelope edges become moving fronts. In both cases,
the returned fronts are ordered from left to right in physical space.
"""

import math
from itertools import pairwise

from .flux import DiscreteFlux

# Each wave stores node indices, followed by its physical propagation speed.
Wave = tuple[int, int, float]


def solve_riemann(iL, iR, flux: DiscreteFlux) -> list[Wave]:
    """Return (left index, right index, RH speed) waves in spatial order.

    The hull is a stack: each node is appended once and removed at most once,
    giving O(abs(iR-iL)) work. A second stack stores accepted edge slopes so
    unchanged edges do not need repeated Rankine–Hugoniot calculations.
    """
    iL, iR = flux.validate_index(iL), flux.validate_index(iR)
    if iL == iR:
        return []  # Identical states have no discontinuity.

    convex = iL < iR
    states, values = flux.u_grid, flux.f_values
    hull: list[int] = []
    slopes: list[float] = []  # slopes[j] joins hull[j] to hull[j+1].

    # Always build in increasing STATE order; reverse for a downward jump later.
    for node in range(min(iL, iR), max(iL, iR) + 1):
        while hull:
            previous = hull[-1]
            # The endpoints were validated above and all visited nodes lie
            # between them. Avoid re-validating indices inside this hot loop.
            speed = float(
                (values[node] - values[previous])
                / (states[node] - states[previous])
            )
            if not slopes:
                break
            violates_envelope = slopes[-1] >= speed if convex else slopes[-1] <= speed
            if not violates_envelope:
                break
            # The last node lies above the lower hull (or below the upper hull).
            # Drop it and retry the chord from the preceding node to this node.
            # Equality also removes collinear nodes, yielding a single contact.
            hull.pop()
            slopes.pop()
        if hull:
            slopes.append(speed)
        hull.append(node)

    # On an upper hull, slopes decrease in state order. Reversing both stacks
    # gives increasing speeds in physical space and the requested iL -> iR jump.
    if not convex:
        hull.reverse()
        slopes.reverse()
    return [(a, b, speed) for (a, b), speed in zip(pairwise(hull), slopes)]


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
