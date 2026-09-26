"""Predict adjacent collisions and solve the Riemann problem they create.

Only neighbors need checking: a front cannot reach a non-neighbor without
first meeting the front between them. Prediction remains a readable O(N) scan.
"""

from itertools import pairwise
import math

from .fronts import Front, FrontCollection
from .riemann import solve_riemann


def next_interaction(fronts) -> tuple[float, int | None]:
    """Return (relative waiting time, left pair index), or (inf, None).

    Accept either a front list or FrontCollection. Ties select the first pair;
    the driver subsequently handles any other event at the same absolute time.
    """
    fronts = fronts.fronts if isinstance(fronts, FrontCollection) else fronts
    best_time, index = math.inf, None
    # pairwise iterates without allocating the full fronts[1:] list each event.
    for i, (left, right) in enumerate(pairwise(fronts)):
        closing_speed = left.speed - right.speed
        if closing_speed > 0:
            # Equal or separating speeds never collide, even at the same point.
            # Clamp tiny negative gaps due to roundoff to an immediate event.
            dt = max(0.0, (right.x - left.x) / closing_speed)
            if dt < best_time:
                best_time, index = dt, i
    return best_time, index


def resolve_interaction(front_i, front_j, flux) -> list[Front]:
    """Replace a collided cluster using only its two exterior states.

    Caller advances incoming fronts to the collision point first. Arguments
    may be the first and last fronts of a multiway cluster. Intermediate states
    occupy zero width at the event and disappear from the new local problem.
    """
    x = 0.5 * (front_i.x + front_j.x)
    return [Front(x, left, right, speed)
            for left, right, speed in solve_riemann(front_i.iL, front_j.iR, flux)]
