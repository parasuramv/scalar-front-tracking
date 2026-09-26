"""Represent moving jumps and expand piecewise constant initial data into fronts.

A single initial jump can generate several fronts (a rarefaction staircase),
one front (for example a shock), or none when its two states are equal.
"""

from dataclasses import dataclass
from itertools import pairwise
import math

import numpy as np

from .riemann import solve_riemann


@dataclass(slots=True)
class Front:
    """One moving discontinuity with a constant speed until its next collision.

    ``x`` is its current position. ``iL``/``iR`` are flux-node indices on its
    left/right. Slots reduce per-front memory; only position changes in flight.
    """

    x: float
    iL: int
    iR: int
    speed: float


class FrontCollection:
    """Spatially ordered fronts; ownership and collision handling live in the driver."""

    def __init__(self, fronts):
        self.fronts = list(fronts)

    def advance(self, dt: float) -> None:
        """Translate all fronts by speed * dt; caller must prevent crossings."""
        if not math.isfinite(dt) or dt < 0:
            raise ValueError("dt must be finite and nonnegative")
        for front in self.fronts:
            front.x += front.speed * dt


def initial_fronts(positions, states, flux) -> FrontCollection:
    """Expand N strictly ordered jumps and N+1 integer state indices.

    The first and last states extend to infinity. Multiple outgoing fronts from
    one jump start at the same point, already ordered by their Riemann speeds.
    """
    positions = np.asarray(positions, dtype=float)
    states = [flux.validate_index(i) for i in states]
    if positions.ndim != 1 or len(states) != len(positions) + 1:
        raise ValueError("Need N jump positions and N+1 state indices")
    if not np.all(np.isfinite(positions)) or np.any(np.diff(positions) <= 0):
        raise ValueError("Jump positions must be finite and strictly increasing")

    # Keep the initial spatial order. Sorting by state or by speed globally
    # would destroy the ordering of the piecewise constant solution.
    fronts = []
    for x, (iL, iR) in zip(positions, pairwise(states)):
        for left, right, speed in solve_riemann(iL, iR, flux):
            fronts.append(Front(float(x), left, right, speed))
    return FrontCollection(fronts)
