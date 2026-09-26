"""Approximate f(u) by the polygonal line through a fixed set of state nodes.

Physical states (for example u=0.5) and node indices (for example i=20) are
separate concepts: ``value``/``state_index`` take states, while ``rh``/``slope``
take indices. Fronts store indices so interaction states stay on this grid.
"""

import operator

import numpy as np


class DiscreteFlux:
    """Continuous piecewise linear flux on a strictly increasing state grid.

    ``f`` must accept a NumPy array and return finite values of the same shape,
    or a scalar for a constant flux. Nonuniform grids are supported. Arrays are
    copied and made read-only so precomputed interval slopes remain consistent.
    """

    def __init__(self, f, u_grid):
        # Validate once at construction; every Riemann solve reuses these nodes.
        grid = np.array(u_grid, dtype=float, copy=True)
        if grid.ndim != 1 or len(grid) < 2 or not np.all(np.isfinite(grid)):
            raise ValueError("u_grid must contain at least two finite values")
        if np.any(np.diff(grid) <= 0):
            raise ValueError("u_grid must be strictly increasing")

        values = np.asarray(f(grid), dtype=float)
        if values.ndim == 0:
            values = np.full_like(grid, values)
        if values.shape != grid.shape or not np.all(np.isfinite(values)):
            raise ValueError("f must return finite values matching u_grid")

        self.f = f  # Keep the original callable for comparisons with f_h.
        self.u_grid = grid
        self.f_values = values.copy()
        self._slopes = np.diff(self.f_values) / np.diff(grid)
        for array in (self.u_grid, self.f_values, self._slopes):
            array.flags.writeable = False

    def validate_index(self, i) -> int:
        """Accept Python/NumPy integers; reject fractional or out-of-range indices."""
        i = operator.index(i)
        if not 0 <= i < len(self.u_grid):
            raise IndexError("State index out of bounds")
        return i

    def slope(self, i) -> float:
        """Return the precomputed slope on [u_grid[i], u_grid[i+1]]."""
        i = self.validate_index(i)
        if i == len(self.u_grid) - 1:
            raise IndexError("Slope index out of bounds")
        return float(self._slopes[i])

    def rh(self, iL, iR) -> float:
        """Rankine–Hugoniot speed: flux difference divided by state difference."""
        iL, iR = self.validate_index(iL), self.validate_index(iR)
        if iL == iR:
            raise ValueError("RH speed undefined for equal states")
        return float(
            (self.f_values[iR] - self.f_values[iL])
            / (self.u_grid[iR] - self.u_grid[iL])
        )

    def _states(self, u):
        """Normalize scalar/array input and reject extrapolation outside the grid."""
        u = np.asarray(u, dtype=float)
        if (not np.all(np.isfinite(u)) or np.any(u < self.u_grid[0])
                or np.any(u > self.u_grid[-1])):
            raise ValueError("State outside discretised flux range")
        return u

    def _interval_indices(self, u):
        """Locate already validated states without repeating range checks."""
        # side='right' chooses the interval beginning at an interior node.
        # At the final node there is no right interval, so use the last one.
        return np.clip(
            np.searchsorted(self.u_grid, u, side="right") - 1,
            0, len(self.u_grid) - 2,
        )

    def interval_index(self, u):
        """Right interval at an interior knot; last interval at the endpoint."""
        u = self._states(u)
        indices = self._interval_indices(u)
        return int(indices) if u.ndim == 0 else indices

    def value(self, u):
        """Evaluate f_h, preserving array shape or returning a float for a scalar."""
        u = self._states(u)
        indices = self._interval_indices(u)
        # Interpolation reuses slopes instead of dividing again for every query.
        value = self.f_values[indices] + (u - self.u_grid[indices]) * self._slopes[indices]
        return float(value) if u.ndim == 0 else value

    def state_index(self, u):
        """Quantize to the nearest node (ties choose the lower node). No clipping."""
        u = self._states(u)
        # Bracket each state, then compare distances to the two candidate nodes.
        hi = np.clip(np.searchsorted(self.u_grid, u), 1, len(self.u_grid) - 1)
        lo = hi - 1
        indices = np.where(u - self.u_grid[lo] <= self.u_grid[hi] - u, lo, hi)
        return int(indices) if u.ndim == 0 else indices
