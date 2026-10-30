import unittest

import numpy as np
from scipy.integrate import solve_ivp

from printer_model.dynamics import dynamics, dynamics_decoupled
from printer_model.parameters import param
from simulations.q1_4_draw_shapes import (
    a, b, center, flame_duration, flame_initial_state,
    flame_motion, flame_path, flame_size_scale, flame_voltage,
)


class FlameInputTests(unittest.TestCase):
    def test_outline_is_smooth_closed_and_fits_deck(self):
        for order in range(4):
            np.testing.assert_allclose(
                flame_path(0.0, nu=order), flame_path(1.0, nu=order), atol=1e-10
            )
        positions = flame_path(np.linspace(0.0, 1.0, 10001))
        half_size = flame_size_scale * np.array([a, b])
        self.assertTrue(np.all(positions >= center - half_size - 1e-8))
        self.assertTrue(np.all(positions <= center + half_size + 1e-8))
        self.assertTrue(np.all(positions >= 0.0))
        self.assertTrue(np.all(positions <= [param["dx"], param["dy"]]))
        np.testing.assert_allclose(
            np.ptp(positions[:, 1]), 2 * b * flame_size_scale, atol=1e-8
        )

    def test_starts_and_finishes_at_rest(self):
        for t in (0.0, flame_duration):
            position, velocity, acceleration, _ = flame_motion(t)
            np.testing.assert_allclose(position, flame_initial_state[[0, 3]], atol=1e-12)
            np.testing.assert_allclose(velocity, 0.0, atol=1e-12)
            np.testing.assert_allclose(acceleration, 0.0, atol=1e-12)
        np.testing.assert_array_equal(flame_initial_state[[1, 2, 4, 5]], 0.0)

    def test_both_models_follow_flame_and_finish_at_rest(self):
        r = param["r_m"]
        transform = np.array([
            [r / 2, r / 2, 0, 0, 0, 0],
            [0, 0, r / 2, r / 2, 0, 0],
            [0, 0, 0, 0, 1, 1],
            [r / 2, -r / 2, 0, 0, 0, 0],
            [0, 0, r / 2, -r / 2, 0, 0],
            [0, 0, 0, 0, 1, -1],
        ])
        times = np.linspace(0.0, flame_duration, 2001)
        expected = np.array([flame_motion(t)[0] for t in times]).T
        for model in (dynamics_decoupled, dynamics):
            with self.subTest(model=model.__name__):
                start = (
                    flame_initial_state if model is dynamics_decoupled
                    else np.linalg.solve(transform, flame_initial_state)
                )
                solution = solve_ivp(
                    model, (0.0, flame_duration), start,
                    args=(flame_voltage, param), t_eval=times,
                    method="Radau", rtol=1e-8, atol=1e-11,
                )
                self.assertTrue(solution.success, solution.message)
                state = (
                    solution.y if model is dynamics_decoupled
                    else transform @ solution.y
                )
                np.testing.assert_allclose(state[[0, 3]], expected, atol=2e-7, rtol=0)
                np.testing.assert_allclose(state[[1, 2, 4, 5], -1], 0.0, atol=2e-7)


if __name__ == "__main__":
    unittest.main()
