import unittest

import numpy as np

from printer_model.parameters import param
from printer_model.square import (
    SquareMode, SquareParameters, encoder_kinematics,
    simulate_square, square_arrival_guard, square_motor_voltage,
)


class SquareTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.settings = SquareParameters()
        cls.result = simulate_square(cls.settings)

    def test_four_straight_equal_sides_and_closure(self):
        run, settings = self.result, self.settings
        points = run.position
        for index in range(4):
            side = points[run.mode == index]
            fixed_axis, moving_axis = (1, 0) if index % 2 == 0 else (0, 1)
            np.testing.assert_allclose(side[:, fixed_axis], settings.vertices[index, fixed_axis],
                                       atol=1.01 * settings.position_tolerance, rtol=0)
            np.testing.assert_allclose(side[0], settings.vertices[index],
                                       atol=1.01 * settings.position_tolerance, rtol=0)
            np.testing.assert_allclose(
                side[-1], settings.vertices[index + 1], atol=1.01 * settings.position_tolerance, rtol=0)
            self.assertLess(
                abs(np.linalg.norm(side[-1] - side[0]) - settings.side_length), 2 * settings.position_tolerance)
            sign = 1 if index < 2 else -1
            self.assertTrue(np.all(sign * np.diff(side[:, moving_axis]) >= -1e-9))
        np.testing.assert_allclose(points[-1], points[0], atol=settings.position_tolerance, rtol=0)

    def test_corner_guards_preserve_every_physical_state_without_a_clock(self):
        self.assertEqual([event["to"] for event in self.result.events],
                         ["Up", "Left", "Down", "Done"])
        for index, event in enumerate(self.result.events):
            before, after = np.array(event["before"]), np.array(event["after"])
            self.assertEqual(len(before), 9)
            np.testing.assert_array_equal(before, after)
            self.assertLessEqual(
                np.max(np.abs(before[2:4])), self.settings.angular_speed_tolerance * (1 + 1e-8))
            self.assertLessEqual(square_arrival_guard(before[:2], before[2:4],
                                                      SquareMode(index), self.settings), 1e-8)
        self.assertTrue(np.all(np.diff([0, *[event["time"] for event in self.result.events]]) > 0))

    def test_arrival_requires_both_corner_position_and_rest(self):
        settings = self.settings
        angles = self.result.state[0, :2]
        self.assertGreater(square_arrival_guard(angles, np.zeros(2), SquareMode.RIGHT, settings), 0)
        target = settings.vertices[1]
        angles = np.array([sum(target), target[0] - target[1]]) / param["r_m"]
        self.assertGreater(square_arrival_guard(angles, np.ones(2), SquareMode.RIGHT, settings), 0)
        self.assertLessEqual(square_arrival_guard(
            angles, np.zeros(2), SquareMode.RIGHT, settings), 0)

    def test_controller_and_guard_need_only_angular_measurements(self):
        # Reconstruct every applied input using encoders and the next corner.
        run = self.result
        for index in np.linspace(0, len(run.time) - 1, 37, dtype=int):
            state, mode = run.state[index], SquareMode(run.mode[index])
            voltage = square_motor_voltage(state[:2], state[2:4], mode, self.settings)
            np.testing.assert_allclose(voltage, run.voltage[index], atol=1e-12)
        position, velocity = encoder_kinematics([2, 1], [3, -1])
        np.testing.assert_allclose(position, [1.5 * param["r_m"], 0.5 * param["r_m"]])
        np.testing.assert_allclose(velocity, [param["r_m"], 2 * param["r_m"]])

    def test_extrusion_disabled_after_completion_and_nozzle_is_dynamic(self):
        run = self.result
        np.testing.assert_array_equal(run.extrude, run.mode != SquareMode.DONE)
        np.testing.assert_allclose(run.state[:, 6:9],
                                   np.tile(self.settings.nozzle.equilibrium, (len(run.time), 1)), atol=1e-8)
        self.assertTrue(np.all(run.position > 0))
        self.assertTrue(np.all(run.position < [param["dx"], param["dy"]]))

    def test_invalid_size_and_duration(self):
        for kwargs in ({"side_length": 0}, {"kd_voltage": -1}, {"side_length": 0.4},
                       {"lower_left": (0, 0.05)}, {"lower_left": (np.nan, 0.05)}):
            with self.assertRaises(ValueError):
                SquareParameters(**kwargs)


if __name__ == "__main__":
    unittest.main()
