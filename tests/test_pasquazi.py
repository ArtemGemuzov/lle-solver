import unittest

import numpy as np
from scipy.integrate import solve_ivp

from lle.numerics.grid import PeriodicGrid
from lle.numerics.pasquazi import solve
from lle.physics.pasquazi import PasquaziParameters


class PasquaziTest(unittest.TestCase):
    def test_rotating_uniform_state(self):
        grid = PeriodicGrid(8, 1)
        omega = 0.25
        p = PasquaziParameters(kappa=1, gain=0.5, detuning=0.75 / (2 * np.pi))
        initial = np.sqrt(0.75) * np.ones((2, 8), dtype=complex)
        initial[1] *= 1 + 1j
        result = solve(
            p, grid, initial, time_step=0.001, time_end=0.1, recording_interval=0.02
        )
        phase = np.exp(1j * omega * result.ring.times[:, None])
        np.testing.assert_allclose(
            result.ring.values, phase * initial[0], atol=2e-8, rtol=0
        )
        np.testing.assert_allclose(
            result.loop[:, 0], phase * initial[1], atol=2e-8, rtol=0
        )

    def test_travelling_harmonic_with_side_modes(self):
        grid = PeriodicGrid(16, 1)
        p = PasquaziParameters(n=1, fsr_mismatch=0.08)
        k = 4 * np.pi
        wave = np.exp(1j * k * grid.theta)
        initial = np.array([1 + 0.3j, -0.2j, 0.4, 0.3j])

        def rhs(t, amplitudes):
            a, b = amplitudes[0], amplitudes[1:]
            da = (0.5j * p.zeta_a * k**2 - 1j * abs(a) ** 2 - p.kappa) * a + np.sqrt(
                p.kappa
            ) * sum(b)
            db = (
                (
                    (0.5j * p.zeta_b - p.sigma) * k**2
                    - 1j * k * p.fsr_mismatch
                    + 2j * np.pi * (p.detuning - np.arange(-1, 2))
                    + p.gain
                )
                * b
                - sum(b)
                + np.sqrt(p.kappa) * a
            )
            return np.r_[da, db]

        ref = solve_ivp(rhs, (0, 0.04), initial, rtol=1e-12, atol=1e-14)
        self.assertTrue(ref.success)
        expected = ref.y[:, -1, None] * wave
        errors = []
        for dt in (0.002, 0.001):
            result = solve(
                p,
                grid,
                initial[:, None] * wave,
                time_step=dt,
                time_end=0.04,
                recording_interval=0.04,
            )
            actual = np.vstack((result.ring.values[-1], result.loop[-1]))
            errors.append(np.max(np.abs(actual - expected)))
        self.assertLess(errors[1], 1e-6)
        self.assertAlmostEqual(errors[0] / errors[1], 4, delta=0.1)
