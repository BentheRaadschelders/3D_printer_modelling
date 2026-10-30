import unittest

import numpy as np

from printer_model.dynamics import dynamics
from printer_model.parameters import param


class DynamicsTests(unittest.TestCase):
    def test_rest_is_an_equilibrium_without_voltage(self):
        state = np.array([1.0, -2.0, 0.0, 0.0, 0.0, 0.0])
        np.testing.assert_array_equal(dynamics(0, state, lambda t: (0, 0), param), 0)

    def test_electrical_input_power_equals_stored_power_and_losses(self):
        rng = np.random.default_rng(42)
        r = param["r_m"]
        for _ in range(30):
            state = rng.normal(size=6)
            voltage = rng.normal(size=2)
            derivative = dynamics(0, state, lambda t: voltage, param)
            omega, current = state[2:4], state[4:6]
            acceleration = derivative[2:4]
            vx, vy = r / 2 * np.array([omega.sum(), omega[0] - omega[1]])
            ax, ay = r / 2 * np.array([
                acceleration.sum(), acceleration[0] - acceleration[1]])
            stored_power = (
                param["J_m"] * omega @ acceleration
                + param["m_head"] * vx * ax
                + (param["m_head"] + param["m_rails"]) * vy * ay
                + param["L_m"] * current @ derivative[4:6]
            )
            loss = (
                param["R_m"] * current @ current
                + (2 * param["mu_lx"] + 8 * param["mu_r"]) * vx**2
                + (2 * param["mu_ly"] + 4 * param["mu_r"]) * vy**2
            )
            self.assertAlmostEqual(stored_power + loss, voltage @ current, places=11)


if __name__ == "__main__":
    unittest.main()
