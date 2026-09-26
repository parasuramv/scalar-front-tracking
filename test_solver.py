"""End-to-end conservation, collision, and convergence checks for the event driver."""

import unittest
import numpy as np
import _srcpath  # noqa: F401  (finds src/fronttrack without installation)
from fronttrack.flux import DiscreteFlux
from fronttrack.fronts import Front
from fronttrack.interactions import next_interaction, resolve_interaction
from fronttrack.solver import FrontTracker


def burgers(n=81):
    return DiscreteFlux(lambda u: u*u/2, np.linspace(-1, 1, n))


class SolverTests(unittest.TestCase):
    def test_shock_and_collision_at_final_time(self):
        """The two shocks meet at t=1 and must already be resolved on return."""
        sim = FrontTracker.from_values(burgers(), [-0.5, 0.5], [1, 0, -1])
        self.assertEqual(next_interaction(sim.fronts), (1.0, 0))
        sim.run(1)
        self.assertEqual(sim.event_count, 1)
        self.assertEqual(len(sim.fronts), 1)
        self.assertAlmostEqual(sim.fronts[0].x, 0)
        sim.run(10)
        np.testing.assert_array_equal(sim.sample([-1, 0, 1]), [1, -1, -1])

    def test_constant_and_annihilation(self):
        """No-jump solutions remain constant; equal exterior states cancel locally."""
        sim = FrontTracker.from_values(burgers(), [], [0.3]).run(100)
        self.assertEqual(len(sim.fronts), 0)
        self.assertAlmostEqual(sim.integral(-1, 1), 0.6)
        self.assertEqual(resolve_interaction(Front(0, 0, 1, 1), Front(0, 1, 0, -1), burgers()), [])

    def test_three_way_collision(self):
        """Three incoming shocks meet at one point and use one exterior Riemann solve."""
        flux = DiscreteFlux(lambda u: u*u/2, [0, 1, 2, 3])
        sim = FrontTracker(flux, [-2.5, -1.5, -0.5], [3, 2, 1, 0]).run(1)
        self.assertEqual(len(sim.fronts), 1)
        self.assertEqual(sim.event_count, 1)
        self.assertAlmostEqual(sim.fronts[0].x, 0)
        self.assertAlmostEqual(sim.fronts[0].speed, 1.5)

    def test_simultaneous_separate_collisions(self):
        """Equal event times at different positions are independent collision clusters."""
        flux = DiscreteFlux(lambda u: u*u/2, [0, 1, 2, 3, 4])
        sim = FrontTracker(flux, [0, 1, 10, 11], [4, 3, 2, 1, 0]).run(1)
        self.assertEqual(sim.event_count, 2)
        self.assertEqual(len(sim.fronts), 2)

    def test_rarefaction_refinement(self):
        """Halving state spacing should approximately halve the sampled L1 error."""
        errors = []
        x = np.linspace(-1.5, 1.5, 60001)
        for n in (11, 21, 41):
            sim = FrontTracker.from_values(burgers(n), [0], [-1, 1]).run(1)
            errors.append(np.mean(np.abs(sim.sample(x) - np.clip(x, -1, 1))))
            self.assertEqual(sim.event_count, 0)
        self.assertLess(errors[1], 0.55*errors[0])
        self.assertLess(errors[2], 0.55*errors[1])

    def test_pulse_mass_tv_and_restart(self):
        """Interactions conserve mass and reduce TV, independently of output times."""
        flux = burgers()
        sim = FrontTracker.from_values(flux, [0, 1], [0, 1, 0])
        tv = sim.total_variation()
        old = sim.snapshot()
        for t in np.linspace(0, 5, 51):
            sim.run(float(t))
            self.assertAlmostEqual(sim.integral(-10, 10), 1, places=10)
            self.assertLessEqual(sim.total_variation(), tv + 1e-12)
            tv = sim.total_variation()
        direct = FrontTracker.from_values(flux, [0, 1], [0, 1, 0]).run(5)
        np.testing.assert_allclose([f.x for f in sim.fronts], [f.x for f in direct.fronts], atol=1e-12)
        self.assertEqual(old[-1].x, 1)
        self.assertGreater(sim.event_count, 0)

    def test_nonconvex_mass_and_structure(self):
        """Randomized fixed-seed data exercise mixed waves and ordered state continuity."""
        flux = DiscreteFlux(lambda u: u**3, np.linspace(-1, 1, 41))
        rng = np.random.default_rng(14)
        values = np.r_[0, rng.uniform(-1, 1, 15), 0]
        sim = FrontTracker.from_values(flux, np.linspace(-1, 1, 16), values)
        mass, tv = sim.integral(-20, 20), sim.total_variation()
        for t in np.linspace(0, 3, 31):
            sim.run(float(t))
            self.assertAlmostEqual(sim.integral(-20, 20), mass, places=10)
            self.assertLessEqual(sim.total_variation(), tv + 1e-12)
            tv = sim.total_variation()
            self.assertTrue(all(a.x <= b.x + 1e-12 and a.iR == b.iL
                                for a, b in zip(sim.fronts, sim.fronts[1:])))

    def test_boundary_flux_balance(self):
        """Unequal boundary fluxes change window mass by f(left)-f(right)."""
        sim = FrontTracker.from_values(burgers(), [0], [1, 0])
        initial = sim.integral(-2, 2)
        sim.run(1)
        self.assertAlmostEqual(sim.integral(-2, 2) - initial, 0.5)
        self.assertAlmostEqual(sim.fronts[0].x, 0.5)

    def test_affine_translation(self):
        """Linear transport translates the entire profile without creating interactions."""
        flux = DiscreteFlux(lambda u: -2*u + 1, [-1, 0, 1])
        sim = FrontTracker.from_values(flux, [0, 1], [0, 1, 0]).run(2)
        np.testing.assert_allclose([f.x for f in sim.fronts], [-4, -3])
        self.assertEqual(sim.event_count, 0)
        self.assertAlmostEqual(sim.integral(-5, 2), 1)

    def test_integral_at_boundaries_and_narrow_intervals(self):
        """Use exact region widths even when a floating-point midpoint rounds away."""
        sim = FrontTracker.from_values(burgers(), [0, 1], [0, 1, 0])
        for a, b, expected in [(-2, 0, 0), (0, 1, 1), (1, 2, 0),
                               (0.25, 0.75, 0.5), (-1, 2, 1)]:
            self.assertAlmostEqual(sim.integral(a, b), expected)
        a = np.nextafter(1.0, 0.0)
        self.assertEqual(sim.integral(a, 1.0), 1.0 - a)

    def test_history_does_not_change_evolution(self):
        """Both advancement paths must produce identical fronts and diagnostics."""
        recorded = FrontTracker.from_values(burgers(), [0, 1], [0, 1, 0])
        plain = FrontTracker.from_values(
            burgers(), [0, 1], [0, 1, 0], record_history=False,
        )
        for time in (0, 0.5, 2, 3):
            recorded.run(time)
            plain.run(time)
        self.assertEqual(recorded.snapshot(), plain.snapshot())
        self.assertEqual(recorded.event_count, plain.event_count)
        self.assertTrue(recorded.segments)
        self.assertEqual(plain.segments, [])
        # History entries must connect straight trajectories over positive time.
        self.assertTrue(all(end_t > start_t for _, start_t, _, end_t in recorded.segments))

    def test_invalid_data_and_limits(self):
        """Malformed data and an exhausted event budget must raise explicit errors."""
        flux = burgers()
        for positions, states in [([0], [1]), ([1, 0], [0, 1, 2]), ([0, 0], [0, 1, 2])]:
            with self.assertRaises(ValueError):
                FrontTracker(flux, positions, states)
        sim = FrontTracker.from_values(flux, [0, 1], [0, 1, 0], max_events=1)
        with self.assertRaises(RuntimeError):
            sim.run(5)
        with self.assertRaises(ValueError):
            sim.run(-1)


if __name__ == "__main__":
    unittest.main()
