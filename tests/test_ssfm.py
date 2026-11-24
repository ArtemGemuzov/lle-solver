import unittest

import numpy as np

from lle.numerics.grid import PeriodicGrid
from lle.numerics.split_schemes import LieTrotterSplitStep
from lle.numerics.ssfm import solve


class SolveOnGridTest(unittest.TestCase):
    def test_complex_field_and_continuation(self):
        grid = PeriodicGrid.from_theta(np.arange(8, dtype=float) * 0.25)
        initial = np.exp(2j * np.pi * grid.theta / grid.length)

        def rotate(field, dt):
            field *= np.exp(2j * dt)
            return field

        scheme = LieTrotterSplitStep(rotate, lambda field, dt: field * np.exp(-dt))

        def run_steps(state, steps):
            return solve(
                scheme, grid, state, num_steps=steps, step_size=0.125, record_every=2
            )

        full = run_steps(initial.copy(), 5)
        np.testing.assert_array_equal(full.theta, grid.theta)
        np.testing.assert_array_equal(full.times, [0, 0.25, 0.5, 0.625])
        expected = initial * np.exp((-1 + 2j) * full.times[:, None])
        np.testing.assert_allclose(full.values, expected, rtol=1e-14, atol=1e-14)

        first = run_steps(initial, 2)
        second = run_steps(first[-1].values, 3)
        np.testing.assert_array_equal(first.values, full.values[:2])
        np.testing.assert_array_equal(second.values, full.values[1:])
        np.testing.assert_array_equal(second.times + first.times[-1], full.times[1:])
        np.testing.assert_array_equal(run_steps(initial, 0).values, full.values[:1])
