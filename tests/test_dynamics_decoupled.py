import unittest

import numpy as np
from scipy.integrate import solve_ivp

from printer_model.dynamics import dynamics, dynamics_decoupled
from printer_model.parameters import param


class DecoupledDynamicsTests(unittest.TestCase):
    def test_transformed_coupled_derivatives_match_decoupled_model(self):
        r = param["r_m"]
        transform = np.array([
            [r / 2, r / 2, 0, 0, 0, 0],
            [0, 0, r / 2, r / 2, 0, 0],
            [0, 0, 0, 0, 1, 1],
            [r / 2, -r / 2, 0, 0, 0, 0],
            [0, 0, r / 2, -r / 2, 0, 0],
            [0, 0, 0, 0, 1, -1],
        ])
        rng = np.random.default_rng(17)
        for _ in range(100):
            state = rng.normal(size=6)
            voltage = rng.normal(size=2)
            signal = lambda t: voltage
            expected = transform @ dynamics(0, state, signal, param)
            actual = dynamics_decoupled(0, transform @ state, signal, param)
            np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-10)

    def test_equal_and_opposite_motor_inputs_drive_separate_axes(self):
        for voltage, driven, stationary in (
            ((1.0, 1.0), [0, 1, 2], [3, 4, 5]),
            ((1.0, -1.0), [3, 4, 5], [0, 1, 2]),
        ):
            with self.subTest(voltage=voltage):
                solution = solve_ivp(
                    dynamics_decoupled, (0, 1), np.zeros(6),
                    args=(lambda t: voltage, param), method="Radau",
                    rtol=1e-8, atol=1e-11,
                )
                self.assertTrue(solution.success, solution.message)
                np.testing.assert_allclose(solution.y[stationary], 0, atol=1e-12)
                self.assertTrue(np.all(solution.y[driven, -1] > 0))


if __name__ == "__main__":
    unittest.main()
