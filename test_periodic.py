"""Front tracking on the circle: cyclic adjacency, seam crossings, invariants."""
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np
import _srcpath  # noqa: F401  (finds src/fronttrack without installation)
from fronttrack.flux import DiscreteFlux
from fronttrack.periodic import Circle, PeriodicFrontTracker as Tracker
from fronttrack.solver import FrontTracker
from fronttrack.visualization import observe_tracker
class PeriodicTests(unittest.TestCase):
    def setUp(self):
        self.flux = DiscreteFlux(lambda u: u*u/2, np.linspace(-1, 1, 21))

    def check_invariants(self, s):
        for left, right in zip(s.fronts, s.fronts[1:] + s.fronts[:1]):
            self.assertEqual(left.iR, right.iL)
        if s.fronts:
            self.assertGreaterEqual(min(np.diff([f.x for f in s.fronts]), default=0), -1e-12)
            self.assertLessEqual(s.fronts[-1].x - s.fronts[0].x, s.domain.length + 1e-12)

    def test_constant_and_empty_fronts(self):
        for positions, values in [([], [0.5]), ([0, 0.5], [0.5, 0.5])]:
            s = Tracker.from_values(self.flux, positions, values, domain=Circle(3, -1))
            s.run(100)
            self.assertEqual(s.sample(-50), 0.5)
            self.assertEqual(s.integral(), 1.5)
            self.assertEqual(s.total_variation(), 0)

    def test_affine_transport_many_wraps_both_directions(self):
        for speed in [-3.0, 0.0, 2.0]:
            flux = DiscreteFlux(lambda u: speed*u + 2, [-1, 0, 1])
            s = Tracker.from_values(flux, [-2, -1.7, -0.8], [0, 1, -1], domain=Circle(2, -2))
            initial = Tracker.from_values(flux, [-2, -1.7, -0.8], [0, 1, -1], domain=Circle(2, -2))
            x = np.linspace(-4.9997, 4.9997, 600)
            mass = s.integral()
            s.run(23.125)
            np.testing.assert_allclose(s.sample(x), initial.sample(x - speed*23.125))
            self.assertAlmostEqual(s.integral(), mass, places=12)
            self.assertEqual(s.event_count, 0)
            self.check_invariants(s)

    def test_right_continuity_at_seam_and_periodic_samples(self):
        s = Tracker.from_values(self.flux, [0, 0.5], [1, -1])
        np.testing.assert_equal(s.sample([0, 1, -1, 0.5]), [1, 1, 1, -1])
        s.run(0.2)
        x = np.array([[0.123, 0.345], [0.671, 0.894]])
        np.testing.assert_allclose(s.sample(x), s.sample(x + 4))

    def test_seam_collision_and_rotation_equivariance(self):
        # Two shocks approach across the seam at t=0.4; the middle fan
        # is still separated. Translating coordinates must commute with evolution.
        s = Tracker.from_values(self.flux, [0.2, 0.5, 0.8], [-1, 1, 0])
        shifted = Tracker.from_values(self.flux, [0.17, 0.57, 0.87], [0, -1, 1])
        s.run(0.4)
        shifted.run(0.4)
        self.assertEqual(s.event_count, 1)
        x = np.linspace(0.0013, 0.9991, 777)
        np.testing.assert_allclose(s.sample(x), shifted.sample(x + 0.37))
        self.assertAlmostEqual(s.integral(), 0)
        self.check_invariants(s)

    def test_mass_tv_and_resume_convex_and_nonconvex(self):
        rng = np.random.default_rng(103)
        for function in [lambda u: u*u/2, lambda u: u**3 - u, lambda u: -u*u/2]:
            flux = DiscreteFlux(function, np.linspace(-1, 1, 31))
            states = rng.integers(0, 31, 15)
            p = np.arange(15)/15
            s, direct = Tracker(flux, p, states), Tracker(flux, p, states)
            mass, tv = s.integral(), s.total_variation()
            for time in np.linspace(0, 4, 31):
                s.run(float(time))
                self.assertAlmostEqual(s.integral(), mass, places=10)
                self.assertLessEqual(s.total_variation(), tv + 1e-10)
                tv = s.total_variation()
                self.check_invariants(s)
            direct.run(4)
            np.testing.assert_allclose(s.sample(np.linspace(0.000123, 0.999123, 400)),
                                       direct.sample(np.linspace(0.000123, 0.999123, 400)))

    def test_simultaneous_separated_collisions(self):
        a = Tracker.from_values(self.flux, [0.1, 0.25, 0.4, 0.6, 0.75, 0.9], [-1, 1, 0, -1, 1, 0])
        a.run(0.2)
        self.assertEqual(a.event_count, 2)
        self.check_invariants(a)

    def test_multiway_annihilation(self):
        # Polygonal tent flux has converging contacts around a constant plateau.
        flux = DiscreteFlux(lambda u: 1-np.abs(u), [-1, 0, 1])
        s = Tracker.from_values(flux, [0, 0.25, 0.75], [-1, 0, 1])
        mass = s.integral()
        s.run(2)
        self.assertAlmostEqual(s.integral(), mass)
        self.assertEqual(s.fronts, [])
        self.assertEqual(s.sample(0.3), 0)
        self.check_invariants(s)

    def test_event_limit_validation_snapshot(self):
        s = Tracker.from_values(self.flux, [0, 0.5], [1, 0], max_events=1)
        snap = s.snapshot()
        snap[0].x = 100
        self.assertNotEqual(s.fronts[0].x, 100)
        with self.assertRaises(RuntimeError):
            s.run(10)
        s.max_events = 10000
        s.run(10)
        with self.assertRaises(ValueError):
            s.run(9)
        for positions, values in [([0, 1], [0, 1]), ([0.5, 0.2], [0, 1]), ([0], [0, 1]), ([], [])]:
            with self.assertRaises(ValueError):
                Tracker.from_values(self.flux, positions, values)
        for length in [0, -1, float('nan'), float('inf')]:
            with self.assertRaises(ValueError):
                Circle(length)
        with self.assertRaises(ValueError):
            s.sample(float('nan'))
        with self.assertRaises(TypeError):
            Tracker(self.flux, [], [0], domain=object())

    def test_midpoint_function(self):
        s = Tracker.from_function(self.flux, lambda x: np.sin(2*np.pi*x), cells=20)
        centers = (np.arange(20)+0.5)/20
        np.testing.assert_equal(s.sample(centers), self.flux.u_grid[self.flux.state_index(np.sin(2*np.pi*centers))])

    def test_against_original_solver_on_repeated_cover(self):
        # Independent real-line driver, with enough copies that exterior waves
        # cannot reach [0,1] during this test (all speeds have magnitude <= 1).
        f = DiscreteFlux(lambda u: u**3/3, np.linspace(-1, 1, 21))
        p = np.array([k+x for k in range(-4, 5) for x in [0.1, 0.4, 0.8]])
        values = [-0.5] + [v for k in range(-4, 5) for v in [1, 0, -0.5]]
        x = np.linspace(0.00123, 0.99987, 1000)
        reference = FrontTracker.from_values(f, p, values).run(1.3).sample(x)
        flux = DiscreteFlux(lambda u: u**3/3, np.linspace(-1, 1, 21))
        s = Tracker.from_values(flux, [0.1, 0.4, 0.8], [1, 0, -0.5]).run(1.3)
        np.testing.assert_allclose(s.sample(x), reference)

    def test_cli_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            subprocess.run([sys.executable, str(Path(__file__).resolve().parent/'periodic_main.py'),
                            '--example', 'transport', '--time', '2', '--frames', '5',
                            '--samples', '40', '--no-plot', '--output', tmp], check=True,
                           capture_output=True)
            data = np.load(Path(tmp)/'transport.npz')
            self.assertEqual(data['u'].shape, (5, 40))
            np.testing.assert_equal(data['u'][0], data['u'][-1])
            self.assertTrue((Path(tmp)/'transport_diagnostics.csv').exists())


class PeriodicHistoryTests(unittest.TestCase):
    """Periodic path recording must preserve the solution and split chart seams."""
    def test_both_directions_multiple_wraps(self):
        for speed in (-2., 2.):
            flux = DiscreteFlux(lambda u: speed*u, [0., 1.])
            domain = Circle(1., -0.5)
            kwargs = dict(flux=flux, positions=[-.3, .2], values=[0, 1], domain=domain)
            plain = Tracker.from_values(**kwargs).run(1.7)
            recorded = Tracker.from_values(**kwargs, record_history=True)
            history = observe_tracker(recorded, np.linspace(-.5, .5, 101), [0, 1.7])
            np.testing.assert_array_equal(recorded.sample(history.x), plain.sample(history.x))
            self.assertEqual(recorded.event_count, plain.event_count)
            self.assertEqual(plain.segments, [])
            for x0, t0, x1, t1 in history.segments:
                self.assertGreater(t1, t0)
                self.assertGreaterEqual(min(x0, x1), -.5-1e-12)
                self.assertLessEqual(max(x0, x1), .5+1e-12)
                self.assertAlmostEqual(x1-x0, speed*(t1-t0))
            self.assertAlmostEqual(sum(b-a for _, a, _, b in history.segments), 3.4)
            edges, values = history.profiles[-1]
            self.assertAlmostEqual(np.dot(np.diff(edges), values), plain.integral())


if __name__ == '__main__':
    unittest.main()
