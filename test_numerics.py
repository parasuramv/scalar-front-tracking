"""Check exported numerical tables and the independent refinement calculation."""

from pathlib import Path
import tempfile
import unittest

import numpy as np

from convergence import rarefaction_error
from main import collect_outputs, example, save_outputs


class NumericsTests(unittest.TestCase):
    def test_exact_rarefaction_error(self):
        """For this uniform Burgers fan, integrating the triangles gives E=h/2."""
        for nodes in (11, 21, 41, 81):
            spacing, error = rarefaction_error(nodes)
            self.assertAlmostEqual(error, spacing / 2)

    def test_export_round_trip(self):
        """NPZ profiles and CSV diagnostics must retain the computed values."""
        tracker = example("collision", 21, record_history=False)
        x, times = np.linspace(-2, 2, 101), np.array([0.0, 1.0, 2.0])
        profiles, diagnostics, fronts = collect_outputs(tracker, x, times)
        # At and after t=1, the two shocks have merged into one stationary shock.
        np.testing.assert_array_equal(diagnostics[:, 1], [2, 1, 1])
        np.testing.assert_array_equal(diagnostics[:, 2], [0, 1, 1])
        with tempfile.TemporaryDirectory() as directory:
            prefix = Path(directory) / "collision"
            save_outputs(prefix, tracker, x, times, profiles, diagnostics, fronts)
            with np.load(str(prefix) + ".npz") as saved:
                np.testing.assert_array_equal(saved["u"], profiles)
                np.testing.assert_array_equal(saved["times"], times)
            saved = np.loadtxt(str(prefix) + "_diagnostics.csv", delimiter=",", skiprows=1)
            np.testing.assert_allclose(saved, diagnostics)
            saved_fronts = np.loadtxt(str(prefix) + "_fronts.csv", delimiter=",", skiprows=1)
            np.testing.assert_allclose(saved_fronts, fronts)


if __name__ == "__main__":
    unittest.main()
