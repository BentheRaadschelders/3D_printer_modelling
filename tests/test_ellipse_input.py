import unittest

import numpy as np
from scipy.integrate import solve_ivp

from printer_model.dynamics import dynamics, dynamics_decoupled
from printer_model.parameters import param
from simulations.q1_4_draw_shapes import (
    a, b, center, duration, initial_state_from_motion, ellipse_motion, ellipse_voltage,
)


class EllipseInputTests(unittest.TestCase):
    def test_reference_and_initial_state_match_periodic_motion(self):
        phase_rate = 2 * np.pi / duration
        r = param["r_m"]
        current_x = r / param["K_m"] * (
            param["m_head"] + 2 * param["J_m"] / r**2
        ) * a * phase_rate**2
        current_y = -r / param["K_m"] * (
            2 * param["mu_ly"] + 4 * param["mu_r"]
        ) * b * phase_rate
        np.testing.assert_allclose(
            initial_state_from_motion(ellipse_motion),
            [center[0] - a, 0, current_x, center[1], -b * phase_rate, current_y],
            atol=1e-13,
        )
        for t in (0.0, duration):
            position, velocity, acceleration, _ = ellipse_motion(t)
            np.testing.assert_allclose(position, center + [-a, 0], atol=1e-14)
            np.testing.assert_allclose(velocity, [0, -b * phase_rate], atol=1e-14)
            np.testing.assert_allclose(acceleration, [a * phase_rate**2, 0], atol=1e-13)

    def test_derivatives_match_finite_differences(self):
        h = 1e-6
        for t in np.linspace(0, duration, 9):
            actual = ellipse_motion(t)
            before, after = ellipse_motion(t - h), ellipse_motion(t + h)
            for order in (1, 2, 3):
                numerical = (after[order - 1] - before[order - 1]) / (2 * h)
                np.testing.assert_allclose(actual[order], numerical, rtol=1e-7, atol=1e-7)

    def test_both_models_follow_two_revolutions_and_return_to_initial_state(self):
        initial_state = initial_state_from_motion(ellipse_motion)
        r = param["r_m"]
        transform = np.array([
            [r / 2, r / 2, 0, 0, 0, 0],
            [0, 0, r / 2, r / 2, 0, 0],
            [0, 0, 0, 0, 1, 1],
            [r / 2, -r / 2, 0, 0, 0, 0],
            [0, 0, r / 2, -r / 2, 0, 0],
            [0, 0, 0, 0, 1, -1],
        ])
        times = np.linspace(0.0, 2 * duration, 2001)
        expected = np.array([ellipse_motion(t)[0] for t in times]).T
        for model in (dynamics_decoupled, dynamics):
            with self.subTest(model=model.__name__):
                start = (
                    initial_state if model is dynamics_decoupled
                    else np.linalg.solve(transform, initial_state)
                )
                solution = solve_ivp(
                    model, (0.0, 2 * duration), start, args=(ellipse_voltage, param),
                    t_eval=times, method="Radau", rtol=1e-8, atol=1e-11,
                )
                self.assertTrue(solution.success, solution.message)
                state = (
                    solution.y if model is dynamics_decoupled
                    else transform @ solution.y
                )
                np.testing.assert_allclose(state[[0, 3]], expected, atol=1e-8, rtol=0)
                np.testing.assert_allclose(state[:, -1], initial_state, atol=1e-8, rtol=0)
        np.testing.assert_allclose(
            ellipse_voltage(0.23 * duration), ellipse_voltage(1.23 * duration), atol=1e-12
        )


if __name__ == "__main__":
    unittest.main()
