"""End-to-end conservation, collision, and convergence checks for the event driver."""

import itertools
import unittest
import numpy as np
import _srcpath  # noqa: F401  (finds src/fronttrack without installation)
from fronttrack.flux import DiscreteFlux
from fronttrack.fronts import Front, initial_fronts
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

    def test_rightmost_annihilation_invalidates_predecessor_prediction(self):
        flux = DiscreteFlux(lambda u: u*u/2, [0, 1, 2, 3])
        sim = FrontTracker(flux, [-1, 0, 1e-13], [3, 0, 2, 0]).run(2)
        self.assertEqual(sim.event_count, 1)
        self.assertEqual(len(sim.fronts), 1)
        self.assertAlmostEqual(sim.fronts[0].x, 2)

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

    def test_collision_into_collinear_chord_leaves_one_front(self):
        """States -0.9 | -1 | 0.5 for u**3: the two fronts meet, and the outgoing
        envelope from -0.9 to 0.5 is one chord. Its continuous tangency point is
        0.45, so the nodes 0.4 and 0.5 are collinear with -0.9 (slope 0.61).
        """
        flux = DiscreteFlux(lambda u: u**3, np.linspace(-1, 1, 21))
        sim = FrontTracker(flux, [0.0, 0.5], [1, 0, 15]).run(20)
        self.assertEqual(sim.event_count, 1)
        self.assertEqual([(f.iL, f.iR) for f in sim.fronts], [(1, 15)])
        self.assertAlmostEqual(sim.fronts[0].speed, 0.61)

    def test_queue_matches_full_rescan(self):
        """The priority queue must reproduce the scan-every-pair driver.

        The reference below is the textbook loop: find the earliest adjacent
        collision by scanning all pairs, move every front, resolve, repeat.
        """
        def reference(flux, positions, states, T):
            fronts = initial_fronts(positions, states, flux).fronts
            t = 0.0
            while True:
                dt, i = next_interaction(fronts)
                if i is None or dt > T - t:
                    for f in fronts:
                        f.x += f.speed * (T - t)
                    return fronts
                for f in fronts:
                    f.x += f.speed * dt
                t += dt
                x = 0.5 * (fronts[i].x + fronts[i + 1].x)
                tol = 1e-12 + 16 * np.finfo(float).eps * max(1.0, abs(x))
                first, last = i, i + 1
                while first > 0 and abs(fronts[first - 1].x - x) <= tol:
                    first -= 1
                while last + 1 < len(fronts) and abs(fronts[last + 1].x - x) <= tol:
                    last += 1
                out = resolve_interaction(fronts[first], fronts[last], flux)
                for f in out:
                    f.x = x
                fronts[first:last + 1] = out

        rng = np.random.default_rng(21)
        for f in (lambda u: u*u/2, lambda u: u**3, lambda u: np.sin(3*u)):
            flux = DiscreteFlux(f, np.linspace(-1, 1, 41))
            for _ in range(20):
                x = np.sort(rng.uniform(-3, 3, 25))
                states = rng.integers(0, 41, 26)
                expected = reference(flux, x, states, 4.0)
                sim = FrontTracker(flux, x, states, record_history=False)
                for t in (0.5, 2.0, 4.0):  # resuming must not matter
                    sim.run(t)
                self.assertEqual([(f.iL, f.iR) for f in sim.fronts],
                                 [(f.iL, f.iR) for f in expected])
                np.testing.assert_allclose([f.x for f in sim.fronts],
                                           [f.x for f in expected], atol=1e-12)

    def test_positions_current_after_event_limit(self):
        """Front.x is the position at the tracker's clock even after an error."""
        sim = FrontTracker.from_values(burgers(), [-1, 0, 1, 2], [1, 0.5, 0, 0.5, 0],
                                       max_events=1, record_history=False)
        with self.assertRaises(RuntimeError):
            sim.run(10)
        self.assertGreater(sim.time, 0)
        free = FrontTracker.from_values(burgers(), [-1, 0, 1, 2], [1, 0.5, 0, 0.5, 0],
                                        record_history=False).run(sim.time)
        self.assertEqual(free.event_count, 1)
        np.testing.assert_allclose([f.x for f in sim.fronts], [f.x for f in free.fronts])
        self.assertEqual(sim.snapshot(), free.snapshot())

    def test_no_zero_time_event_loop(self):
        """Many u**3 runs with collinear chords finish well within max_events."""
        flux = DiscreteFlux(lambda u: u**3, np.linspace(-1, 1, 21))
        for states in itertools.product(range(0, 21, 5), repeat=4):
            sim = FrontTracker(flux, [0.0, 0.05, 0.13], states,
                               max_events=200, record_history=False)
            sim.run(10.0)  # raises RuntimeError on a livelock


class CountingList(list):
    """A list that counts how often it is touched, to show drivers stay off
    the public front list while ``run`` resolves events (they use links)."""

    ops = 0

    def _touch(self):
        self.ops += 1

    def __getitem__(self, i):
        self._touch()
        return super().__getitem__(i)

    def __setitem__(self, i, v):
        self._touch()
        super().__setitem__(i, v)

    def __delitem__(self, i):
        self._touch()
        super().__delitem__(i)

    def __iter__(self):
        self._touch()
        return super().__iter__()

    def __len__(self):
        self._touch()
        return super().__len__()


def unlinked(fronts):
    return all(f.prev is None and f.next is None for f in fronts)


class LinkedFrontTests(unittest.TestCase):
    """During run() fronts are a linked list; outside it, only the list exists."""

    def setUp(self):
        rng = np.random.default_rng(3)
        self.flux = DiscreteFlux(lambda u: u**3, np.linspace(-1, 1, 65))
        self.positions = np.sort(rng.uniform(-3, 3, 60))
        self.states = rng.integers(0, 65, 61)

    def make(self, **kwargs):
        return FrontTracker(self.flux, self.positions, self.states,
                            record_history=False, **kwargs)

    def test_no_list_work_per_event(self):
        sim = self.make()
        sim.collection.fronts = counting = CountingList(sim.fronts)
        sim.run(3.0)
        self.assertGreater(sim.event_count, 100)
        # A handful of list operations per run(), independent of the events.
        self.assertLess(counting.ops, 20)
        self.assertIs(sim.fronts, counting)  # rebuilt in place, same object
        self.assertTrue(unlinked(sim.fronts))
        self.assertTrue(unlinked(sim.snapshot()))

    def test_hooks_see_current_fronts_during_run(self):
        """A subclass hook reading fronts mid-run sees the current fronts."""
        log = []

        class Recording(FrontTracker):
            def _resolve_cluster(self, front):
                before = self.total_variation()
                super()._resolve_cluster(front)
                fronts = self.fronts
                log.append((before, self.total_variation(),
                            all(a.iR == b.iL for a, b in zip(fronts, fronts[1:]))))

        sim = Recording(self.flux, self.positions, self.states, record_history=False)
        sim.run(3.0)
        self.assertEqual(len(log), sim.event_count)
        self.assertTrue(all(chained for _, _, chained in log))
        self.assertTrue(all(after <= before + 1e-12 for before, after, _ in log))
        self.assertGreater(sum(after < before - 1e-12 for before, after, _ in log), len(log) // 2)
        self.assertTrue(unlinked(sim.fronts))

    def test_edits_between_runs_are_relinked(self):
        edited, twin = self.make(), self.make()
        edited.run(0.5)
        twin.run(0.5)
        edited.fronts[:] = edited.snapshot()  # fresh, unlinked copies
        edited.run(3.0)
        twin.run(3.0)
        self.assertEqual(edited.snapshot(), twin.snapshot())
        self.assertEqual(edited.event_count, twin.event_count)

    def test_unlinked_after_event_limit(self):
        sim = self.make(max_events=7)
        with self.assertRaises(RuntimeError):
            sim.run(3.0)
        self.assertEqual(sim.event_count, 7)
        self.assertTrue(unlinked(sim.fronts))
        sim.max_events = 10**6
        sim.run(3.0)
        # Resuming re-anchors at the pause, so positions agree up to roundoff.
        free = self.make().run(3.0)
        self.assertEqual([(f.iL, f.iR, f.speed) for f in sim.fronts],
                         [(f.iL, f.iR, f.speed) for f in free.fronts])
        np.testing.assert_allclose([f.x for f in sim.fronts], [f.x for f in free.fronts],
                                   rtol=0, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
