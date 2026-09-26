"""Analytical and geometric checks on the scalar entropy Riemann construction."""

import unittest
import numpy as np
import _srcpath  # noqa: F401  (finds src/fronttrack without installation)
from fronttrack.flux import DiscreteFlux
from fronttrack.riemann import solve_riemann, enforce_entropy


class RiemannTests(unittest.TestCase):
    def test_burgers(self):
        """Distinguish a downward shock from an upward rarefaction staircase."""
        flux = DiscreteFlux(lambda u: u*u/2, [-1, 0, 1])
        self.assertEqual(solve_riemann(2, 0, flux), [(2, 0, 0.0)])
        self.assertEqual(solve_riemann(0, 2, flux), [(0, 1, -0.5), (1, 2, 0.5)])
        self.assertEqual(solve_riemann(1, 1, flux), [])
        with self.assertRaises(ValueError):
            enforce_entropy([(0, 1, 1), (1, 2, -1)])

    def test_affine_contacts_and_concave(self):
        """Affine waves coalesce; changing convexity swaps shock/fan orientations."""
        flux = DiscreteFlux(lambda u: 2*u+1, [-1, 0, 1])
        self.assertEqual(solve_riemann(0, 2, flux), [(0, 2, 2.0)])
        self.assertEqual(solve_riemann(2, 0, flux), [(2, 0, 2.0)])
        flux = DiscreteFlux(lambda u: -u*u/2, [-1, 0, 1])
        self.assertEqual(len(solve_riemann(2, 0, flux)), 2)
        self.assertEqual(len(solve_riemann(0, 2, flux)), 1)

    def test_nonconvex_envelope_admissibility(self):
        """Check chord placement, not merely speed ordering, for every state pair."""
        flux = DiscreteFlux(lambda u: u**3 + 0.15*np.sin(7*u), np.linspace(-2, 2, 31))
        for left in range(31):
            for right in range(31):
                waves = solve_riemann(left, right, flux)
                enforce_entropy(waves)
                if left == right:
                    self.assertEqual(waves, [])
                    continue
                self.assertEqual(waves[0][0], left)
                self.assertEqual(waves[-1][1], right)
                for a, b, speed in waves:
                    k = np.arange(min(a, b), max(a, b)+1)
                    chord = flux.f_values[a] + speed*(flux.u_grid[k]-flux.u_grid[a])
                    gap = flux.f_values[k] - chord
                    self.assertTrue(np.all(gap >= -1e-12) if left < right else np.all(gap <= 1e-12))


if __name__ == "__main__":
    unittest.main()
