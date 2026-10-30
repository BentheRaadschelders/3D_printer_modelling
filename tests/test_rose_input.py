import unittest

import numpy as np
from scipy.integrate import solve_ivp

from printer_model.dynamics import dynamics, dynamics_decoupled
from printer_model.parameters import param
from simulations.q1_4_draw_shapes import (
    center, rose_duration, initial_state_from_motion, rose_motion, rose_radius, rose_voltage,
)


class RoseInputTests(unittest.TestCase):
    def test_petal_count_size_periodicity_and_initial_state(self):
        radius, duration = rose_radius, rose_duration
        for petals in (1, 3, 4, 5, 7, 8, 12):
            with self.subTest(petals=petals):
                k = petals if petals % 2 else petals / 2
                phase_span = np.pi if petals % 2 else 2 * np.pi
                phase_rate = phase_span / duration
                tips = []
                for petal in range(petals):
                    tip_time = petal * np.pi / (k * phase_rate)
                    tip = rose_motion(tip_time, petals, radius, duration)[0]
                    np.testing.assert_allclose(np.linalg.norm(tip - center), radius, atol=1e-12)
                    tips.append(tip)
                self.assertEqual(len(np.unique(np.round(tips, 10), axis=0)), petals)
                np.testing.assert_allclose(
                    rose_motion(duration, petals, radius, duration),
                    rose_motion(0, petals, radius, duration), atol=1e-10
                )
                r = param["r_m"]
                current_x = r / param["K_m"] * (
                    param["m_head"] + 2 * param["J_m"] / r**2
                ) * radius * (k**2 + 1) * phase_rate**2
                current_y = -r / param["K_m"] * (
                    2 * param["mu_ly"] + 4 * param["mu_r"]
                ) * radius * phase_rate
                np.testing.assert_allclose(
                    initial_state_from_motion(lambda t: rose_motion(t, petals, radius, duration)),
                    [center[0] - radius, 0, current_x, center[1], -radius * phase_rate, current_y],
                    atol=1e-12,
                )
                positions = np.array([
                    rose_motion(t, petals, radius, duration)[0]
                    for t in np.linspace(0, duration, 401)
                ])
                self.assertTrue(np.all(positions >= 0))
                self.assertTrue(np.all(positions <= [param["dx"], param["dy"]]))

    def test_derivatives_match_finite_differences(self):
        h = 1e-6
        for petals in (3, 5, 8, 12):
            for fraction in (0.17, 0.38, 0.64, 0.89):
                t = (1 + fraction) * rose_duration / petals
                actual = rose_motion(t, petals)
                before = rose_motion(t - h, petals)
                after = rose_motion(t + h, petals)
                for order in (1, 2, 3):
                    numerical = (after[order - 1] - before[order - 1]) / (2 * h)
                    np.testing.assert_allclose(
                        actual[order], numerical, rtol=2e-6, atol=2e-7
                    )

    def test_both_models_track_two_periods_of_odd_and_even_roses(self):
        duration = rose_duration
        times = np.linspace(0, 2 * duration, 1601)
        r = param["r_m"]
        transform = np.array([
            [r / 2, r / 2, 0, 0, 0, 0],
            [0, 0, r / 2, r / 2, 0, 0],
            [0, 0, 0, 0, 1, 1],
            [r / 2, -r / 2, 0, 0, 0, 0],
            [0, 0, r / 2, -r / 2, 0, 0],
            [0, 0, 0, 0, 1, -1],
        ])
        for petals in (5, 8):
            initial_state = initial_state_from_motion(
                lambda t: rose_motion(t, petals=petals, traversal_time=duration))
            expected = np.array([
                rose_motion(t, petals, traversal_time=duration)[0] for t in times
            ]).T

            for model in (dynamics_decoupled, dynamics):
                with self.subTest(petals=petals, model=model.__name__):
                    start = (
                        initial_state if model is dynamics_decoupled
                        else np.linalg.solve(transform, initial_state)
                    )

                    def voltage(t):
                        return rose_voltage(t, petals=petals, traversal_time=duration)

                    solution = solve_ivp(
                        model, (0, 2 * duration), start,
                        args=(voltage, param), t_eval=times,
                        method="Radau", rtol=1e-8, atol=1e-11,
                    )
                    self.assertTrue(solution.success, solution.message)
                    state = solution.y if model is dynamics_decoupled else transform @ solution.y
                    np.testing.assert_allclose(state[[0, 3]], expected, atol=2e-7, rtol=0)
                    np.testing.assert_allclose(state[:, -1], initial_state, atol=2e-7, rtol=0)
                    np.testing.assert_allclose(voltage(0.37), voltage(duration + 0.37), atol=1e-12)

    def test_invalid_parameters(self):
        for petals in (0, -2, 2.5, True, 2, 6, 10):
            with self.assertRaises(ValueError):
                rose_motion(0, petals=petals)
        for radius in (0, -0.1, np.nan):
            with self.assertRaises(ValueError):
                rose_motion(0, radius=radius)
        for duration in (0, -1, np.inf):
            with self.assertRaises(ValueError):
                rose_motion(0, traversal_time=duration)


if __name__ == "__main__":
    unittest.main()
