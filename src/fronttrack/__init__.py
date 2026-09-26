"""Dafermos polygonal-flux front tracking for scalar conservation laws on R.

Public API (read the modules in this order):

- ``DiscreteFlux``   (flux.py)          polygonal flux on a state grid
- ``solve_riemann``  (riemann.py)       entropy envelope of one jump
- ``Front``, ``FrontCollection``, ``initial_fronts`` (fronts.py)
- ``next_interaction``, ``resolve_interaction``      (interactions.py)
- ``FrontTracker``   (solver.py)        event-driven evolution driver

Install once with ``python -m pip install -e .`` from the repository root, or
let the repository's scripts add ``src/`` to the path themselves.
"""
from .flux import DiscreteFlux
from .riemann import solve_riemann
from .fronts import Front, FrontCollection, initial_fronts
from .interactions import next_interaction, resolve_interaction
from .solver import FrontTracker

__version__ = "0.2.0"
__all__ = [
    "DiscreteFlux", "solve_riemann", "Front", "FrontCollection", "initial_fronts",
    "next_interaction", "resolve_interaction", "FrontTracker",
]
