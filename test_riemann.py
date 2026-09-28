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


class CollinearRoundoffTests(unittest.TestCase):
    """Exactly collinear flux nodes must give ONE front, whatever roundoff does.

    For f = u**3 the chord slope from u_a to u_b is u_a**2 + u_a*u_b + u_b**2, so
    on np.linspace(-1, 1, 21) the nodes -0.3, 0.1, 0.2 (indices 7, 11, 12) are
    collinear with slope 0.07. Rounding can make the middle node look slightly
    convex, which used to split the chord into two fronts 1 ulp apart in speed.
    """

    def assertSingleFront(self, waves, iL, iR, speed):
        self.assertEqual(len(waves), 1, waves)
        self.assertEqual(waves[0][:2], (iL, iR))
        self.assertAlmostEqual(waves[0][2], speed, places=12)

    def test_cubic_chord_is_one_front(self):
        flux = DiscreteFlux(lambda u: u**3, np.linspace(-1, 1, 21))
        self.assertSingleFront(solve_riemann(7, 12, flux), 7, 12, 0.07)

    def test_concave_branch(self):
        """A decreasing jump for -u**3 uses the upper envelope: same chord."""
        flux = DiscreteFlux(lambda u: -u**3, np.linspace(-1, 1, 21))
        self.assertSingleFront(solve_riemann(12, 7, flux), 12, 7, -0.07)

    def test_nonuniform_grid(self):
        flux = DiscreteFlux(lambda u: u**3, [-1, -0.55, -0.3, -0.12, 0.1, 0.2, 0.47, 1])
        self.assertSingleFront(solve_riemann(2, 5, flux), 2, 5, 0.07)

    def test_affine_terms_do_not_matter(self):
        """f + c + s*u has the same fronts, with speeds shifted by s.

        s = -0.07 makes the chord's speed zero, so a tolerance relative to the
        slope would vanish there; c = 1000 makes the rounding errors large.
        """
        for c, s in [(0, -0.07), (1000, 0), (0, 5)]:
            with self.subTest(c=c, s=s):
                flux = DiscreteFlux(lambda u: u**3 + c + s*u, np.linspace(-1, 1, 21))
                self.assertSingleFront(solve_riemann(7, 12, flux), 7, 12, 0.07 + s)

    def test_fans_strictly_separate(self):
        """Every fan has strictly increasing speeds, with no co-moving neighbours.

        Strictness is what keeps the event loop finite: fronts leaving one
        collision point can never be predicted to meet again with dt = 0.
        """
        fluxes = [lambda u: u**3, lambda u: u**3 - 0.07*u,
                  lambda u: np.sin(3*u), lambda u: u*u/2]
        for f in fluxes:
            for n in (21, 41):
                flux = DiscreteFlux(f, np.linspace(-1, 1, n))
                for left in range(n):
                    for right in range(n):
                        speeds = [w[2] for w in solve_riemann(left, right, flux)]
                        gaps = np.diff(speeds)
                        self.assertTrue(np.all(gaps > 1e-9), (left, right, speeds))

    def test_fine_rarefaction_keeps_every_node(self):
        """The allowance must not merge genuinely convex nodes, even tiny cells."""
        h = np.geomspace(1e-6, 1, 40)
        grid = np.concatenate([-h[::-1], [0], h])
        flux = DiscreteFlux(lambda u: u*u/2, grid)
        self.assertEqual(len(solve_riemann(0, len(grid) - 1, flux)), len(grid) - 1)


if __name__ == "__main__":
    unittest.main()
