"""Check that views preserve physical states, geometry and observation semantics."""
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/fronttrack-mpl')
import unittest
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import _srcpath
from fronttrack import DiscreteFlux, FrontTracker
from fronttrack.visualization import (
    History, GridHistory, observe_tracker, step_profile, plot_profile,
    plot_spacetime, plot_grid_snapshot, color_norm,
)


class VisualizationTests(unittest.TestCase):
    def tearDown(self):
        plt.close('all')

    def test_exact_profile_not_raster_approximation(self):
        flux = DiscreteFlux(lambda u: u, [0, 2])
        tracker = FrontTracker.from_values(flux, [.123], [0, 2])
        h = observe_tracker(tracker, [-1, 1], [0, .25])
        np.testing.assert_allclose(h.profiles[1][0], [-1, .373, 1])
        np.testing.assert_array_equal(h.profiles[1][1], [0, 2])
        ax = plot_profile(h)
        np.testing.assert_array_equal(ax.collections[0].get_array(), [0, 2])
        norm = color_norm(h.values)
        field = plot_spacetime(h, norm=norm)
        self.assertIs(field.collections[0].norm, norm)
        np.testing.assert_array_equal(field.collections[0].get_array().reshape(2, 2), h.values)

    def test_colour_range_includes_fronts_missed_by_raster(self):
        flux = DiscreteFlux(lambda u: u, [0, 2])
        tracker = FrontTracker.from_values(flux, [.1, .2], [0, 2, 0])
        h = observe_tracker(tracker, [-1, 1], [0, .1])
        self.assertEqual(h.values.max(), 0)
        self.assertEqual(color_norm(h).vmax, 2)
        ax = plot_profile(h)
        self.assertGreater(ax.get_ylim()[1], 2)

    def test_sampling_does_not_change_final_state(self):
        flux = DiscreteFlux(lambda u: .5*u*u, np.linspace(0, 1, 21))
        one = FrontTracker.from_values(flux, [0, 1], [0, 1, 0]).run(3)
        other = FrontTracker.from_values(flux, [0, 1], [0, 1, 0])
        observe_tracker(other, np.linspace(-1, 4, 51), np.linspace(0, 3, 71))
        # Point samples exactly on jumps can differ after roundoff-sized shifts.
        np.testing.assert_allclose([f.x for f in one.fronts], [f.x for f in other.fronts], atol=1e-12)
        self.assertEqual([(f.iL, f.iR) for f in one.fronts],
                         [(f.iL, f.iR) for f in other.fronts])
        self.assertAlmostEqual(one.integral(-1, 4), other.integral(-1, 4))

    def test_constant_single_time_and_validation(self):
        tracker = FrontTracker.from_values(DiscreteFlux(lambda u: u, [-1, 1]), [], [1])
        h = observe_tracker(tracker, [-1, 1], [0])
        plot_profile(h)
        with self.assertRaises(ValueError):
            plot_spacetime(h)
        self.assertLess(color_norm(h.values).vmin, 1)
        for times in ([0, 0], [-1, 1], [0, np.nan]):
            with self.assertRaises(ValueError):
                observe_tracker(tracker, [-1, 1], times)
            self.assertEqual(tracker.time, 0)
        with self.assertRaises(ValueError):
            History([0, 1], [0, 1], [[0, 1]])

    def test_nonuniform_grid_slice_orientation_and_mask(self):
        e = (np.array([0, 1, 3]), np.array([-1, 0, 2, 4]))
        values = np.arange(12).reshape(2, 2, 3)
        valid = np.ones_like(values, dtype=bool)
        valid[1, 1, 0] = False
        h = GridHistory(e, np.array([0, 1]), values, valid)
        xline = h.line(coordinate=-.5)
        np.testing.assert_array_equal(xline.profiles[1][1], [6, 9])
        np.testing.assert_array_equal(xline.valid[1], [True, False])
        yline = h.line(axis=1, coordinate=2)
        np.testing.assert_array_equal(yline.profiles[1][1], [9, 10, 11])
        self.assertEqual(plot_profile(yline).get_xlabel(), 'y')
        ax = plot_spacetime(xline)
        np.testing.assert_array_equal(ax.collections[0].get_coordinates()[0, :, 0], e[0])
        self.assertEqual(len(ax.collections[1].get_paths()), 1)
        ax = plot_grid_snapshot(h)
        np.testing.assert_array_equal(ax.collections[0].get_array().reshape(3, 2), values[1].T)
        self.assertEqual(len(ax.collections[1].get_paths()), 1)
        with self.assertRaises(ValueError):
            h.line(coordinate=7)


if __name__ == '__main__':
    unittest.main()
