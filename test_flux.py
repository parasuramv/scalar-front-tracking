"""Regression tests for physical-state conversion and polygonal flux arithmetic."""

import unittest
import numpy as np
import _srcpath  # noqa: F401  (finds src/fronttrack without installation)
from fronttrack.flux import DiscreteFlux


class FluxTests(unittest.TestCase):
    def test_interpolation_and_endpoints(self):
        """Check polygonal values, endpoint inclusion, and lower-node tie breaking."""
        flux = DiscreteFlux(lambda u: u**2, [-2, 0, 1])
        np.testing.assert_allclose(flux.value([-2, -1, 0, 0.5, 1]), [4, 2, 0, 0.5, 1])
        self.assertEqual(flux.interval_index(-2), 0)
        self.assertEqual(flux.interval_index(1), 1)
        np.testing.assert_array_equal(flux.state_index([-2, -1, 0.6, 1]), [0, 0, 2, 2])

    def test_burgers_speeds(self):
        """For Burgers, every chord speed is the mean of its two endpoint states."""
        flux = DiscreteFlux(lambda u: 0.5*u**2, np.linspace(-2, 2, 15))
        for i in range(15):
            for j in range(15):
                if i != j:
                    self.assertAlmostEqual(flux.rh(i, j), (flux.u_grid[i] + flux.u_grid[j])/2)
        for i in range(14):
            self.assertEqual(flux.slope(i), flux.rh(i, i+1))

    def test_array_shape_and_owned_grid(self):
        """Caller mutations cannot stale cached slopes; array evaluation keeps shape."""
        grid = np.array([-2.0, 0.0, 1.0])
        flux = DiscreteFlux(lambda u: u*u, grid)
        grid[0] = -100
        np.testing.assert_allclose(flux.value([[-2, -1], [0, 1]]), [[4, 2], [0, 1]])
        self.assertEqual(flux.slope(0), -2)
        with self.assertRaises(ValueError):
            flux.u_grid[0] = -100

    def test_invalid_inputs(self):
        """Reject malformed grids and states instead of extrapolating silently."""
        for grid in ([1], [0, 0], [1, 0], [0, np.nan], [[0, 1]]):
            with self.assertRaises(ValueError):
                DiscreteFlux(lambda u: u, grid)
        flux = DiscreteFlux(lambda u: u, [0, 1])
        for value in (-0.1, 1.1, np.nan):
            with self.assertRaises(ValueError):
                flux.value(value)
        with self.assertRaises(IndexError):
            flux.rh(-1, 0)
        with self.assertRaises(ValueError):
            flux.rh(0, 0)
        with self.assertRaises(TypeError):
            flux.rh(0.5, 1)


if __name__ == "__main__":
    unittest.main()
