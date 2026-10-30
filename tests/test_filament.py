"""Analytical and energy-balance checks for the Q1.6 discretization."""

from dataclasses import replace
import unittest

import numpy as np
from numpy.testing import assert_allclose

from printer_model.filament import (
    FilamentParameters, advance_temperature, ellipse_mesh,
    cell_heat_balance, filament_dynamics, simulate_filament,
    settled_nozzle_temperature, settled_nozzle_voltage,
    stability_diagnostics, thermal_matrix,
)


class TestFilament(unittest.TestCase):
    def test_conduction_conserves_energy_and_uniform_temperature(self):
        settings = FilamentParameters(air_coefficient=0, bed_coefficient=0)
        initial = np.array([310., 500., 370., 410., 330.])
        for closed in (False, True):
            matrix = thermal_matrix(5, 0.001, settings, closed=closed)
            evolved = advance_temperature(initial, 3., matrix, 0.)
            assert_allclose(evolved.sum(), initial.sum(), rtol=1e-12)
            assert_allclose(matrix @ np.ones(5), 0., atol=1e-15)
            self.assertGreaterEqual(evolved.min(), initial.min() - 1e-10)
            self.assertLessEqual(evolved.max(), initial.max() + 1e-10)

    def test_uniform_solution_matches_lumped_analytical_solution(self):
        settings = FilamentParameters()
        equilibrium = settings.equilibrium_temperature
        expected = equilibrium + (500 - equilibrium) * np.exp(-settings.decay_rate * 7)
        for closed in (False, True):
            matrix = thermal_matrix(12, 0.0005, settings, closed=closed)
            actual = advance_temperature(np.full(12, 500.), 7., matrix, equilibrium)
            assert_allclose(actual, expected, rtol=1e-12)

    def test_growth_matches_independent_cell_cooling_when_conductivity_is_zero(self):
        # A different operating point ensures deposition uses the supplied
        # settled temperature rather than a fixed deposition boundary value.
        nozzle_temperature = settled_nozzle_temperature(12., 300., 8., 9.)
        settings = replace(FilamentParameters(), conductivity=0.,
                           deposition_temperature=nozzle_temperature)
        result = simulate_filament(settings, n_cells=30, print_samples=20, cooling_samples=10)
        age = result["time"][:, None] - result["birth_times"][None, :]
        expected = settings.equilibrium_temperature + (
            settings.deposition_temperature - settings.equilibrium_temperature
        ) * np.exp(-settings.decay_rate * np.maximum(age, 0))
        active = age >= 0
        self.assertTrue(np.all(np.isnan(result["temperature"][~active])))
        assert_allclose(result["temperature"][active], expected[active], atol=2e-10)
        assert_allclose(np.ptp(result["final_temperature"]), 1., atol=1e-7)

    def test_settled_nozzle_satisfies_electrical_and_thermal_power_balance(self):
        for voltage in (0., 12., -12.):
            temperature = settled_nozzle_temperature(voltage, 300., 8., 9.)
            electrical_power = voltage**2 / 8.
            thermal_loss = (temperature - 300.) / 9.
            assert_allclose(thermal_loss, electrical_power, atol=1e-12)
        assert_allclose(settled_nozzle_temperature(12., 300., 8., 9.), 462.)
        # Selected operating point: 14 V, T_e=297 K, R_n=10 ohm,
        # R_t,n=11 K/W. Heater power and thermal loss must both be 19.6 W.
        selected_temperature = settled_nozzle_temperature(14., 297., 10., 11.)
        assert_allclose(selected_temperature, 512.6)
        assert_allclose((selected_temperature - 297.) / 11., 19.6)
        assert_allclose(FilamentParameters().deposition_temperature, selected_temperature)

    def test_selected_nozzle_temperature_and_voltage_give_same_equilibrium(self):
        for temperature in (300., 473.15, 513.15):
            voltage = settled_nozzle_voltage(temperature, 300., 8., 9.)
            self.assertGreaterEqual(voltage, 0.)
            assert_allclose(settled_nozzle_temperature(voltage, 300., 8., 9.), temperature)
        with self.assertRaises(ValueError):
            settled_nozzle_voltage(290., 300., 8., 9.)
        with self.assertRaises(ValueError):
            settled_nozzle_temperature(12., 300., 0., 9.)

    def test_mean_of_nonuniform_loop_satisfies_global_energy_balance(self):
        settings = FilamentParameters()
        initial = np.array([310., 500., 370., 410., 330.])
        matrix = thermal_matrix(5, 0.001, settings, closed=True)
        actual = advance_temperature(initial, 8., matrix, settings.equilibrium_temperature)
        expected_mean = settings.equilibrium_temperature + (
            initial.mean() - settings.equilibrium_temperature
        ) * np.exp(-settings.decay_rate * 8.)
        assert_allclose(actual.mean(), expected_mean, rtol=1e-12)

    def test_deposition_energy_has_no_phantom_material(self):
        settings = FilamentParameters(air_coefficient=0, bed_coefficient=0)
        # Isothermal deposition with no lateral heat exchange requires no wait.
        result = simulate_filament(settings, n_cells=20, print_samples=8)
        mass_per_cell = settings.density * settings.width**2 * result["cell_length"]
        reference = settings.ambient_temperature
        for row in result["temperature"]:
            active = np.isfinite(row)
            energy = np.nansum(row - reference) * mass_per_cell * settings.specific_heat
            injected = active.sum() * mass_per_cell * settings.specific_heat * (
                settings.deposition_temperature - reference)
            assert_allclose(energy, injected, rtol=1e-12)
        self.assertEqual(result["wait_time"], 0.)

    def test_circle_arc_length_and_birth_schedule(self):
        mesh = ellipse_mesh(40, print_time=20., a=0.1, b=0.1)
        assert_allclose(mesh["length"], 2 * np.pi * 0.1, rtol=1e-11)
        assert_allclose(mesh["birth_times"], np.arange(40) * 20. / 40, atol=1e-10)

    def test_seam_is_connected_only_after_closure(self):
        settings = FilamentParameters()
        opened = thermal_matrix(10, 0.001, settings)
        closed = thermal_matrix(10, 0.001, settings, closed=True)
        self.assertEqual(opened[0, -1], 0.)
        self.assertGreater(closed[0, -1], 0.)
        assert_allclose(np.asarray(closed.sum(axis=1)).ravel(), -settings.decay_rate)

    def test_face_flows_and_sources_satisfy_cell_and_total_energy_balance(self):
        settings = FilamentParameters()
        for count in (1, 2, 5):
            initial = np.linspace(310., 500., count)
            for closed in (False, True):
                with self.subTest(count=count, closed=closed):
                    balance = cell_heat_balance(initial, 0.001, settings, closed)
                    faces = balance["face_flow_W"]
                    # With a rising temperature, interior conductive heat
                    # must flow towards the colder, lower-index cells.
                    self.assertTrue(np.all(faces[1:-1] <= 0))
                    if closed:
                        assert_allclose(faces[0], faces[-1])
                    else:
                        assert_allclose(faces[[0, -1]], 0.)
                    assert_allclose(balance["net_power_W"].sum(),
                                    balance["source_power_W"].sum(), atol=1e-14)
                    matrix = thermal_matrix(count, 0.001, settings, closed)
                    matrix_rate = matrix @ (initial - settings.equilibrium_temperature)
                    assert_allclose(filament_dynamics(0., initial, 0.001, settings, closed),
                                    matrix_rate, atol=1e-12)

    def test_explicit_bound_preserves_nonnegative_weights_and_expm_needs_no_cfl(self):
        settings = FilamentParameters(ambient_temperature=300., bed_temperature=300.)
        matrix = thermal_matrix(5, 0.001, settings, closed=True)
        diagnostics = stability_diagnostics(matrix)
        step = diagnostics["forward_euler_nonnegative_weight_limit_s"]
        initial = np.array([300., 300., 500., 300., 300.])
        euler = initial + step * filament_dynamics(0., initial, 0.001, settings, closed=True)
        self.assertGreaterEqual(euler.min(), 300. - 1e-10)
        self.assertLessEqual(euler.max(), 500. + 1e-10)
        too_large = initial + 1.01 * step * filament_dynamics(
            0., initial, 0.001, settings, closed=True)
        self.assertLess(too_large.min(), 300.)
        # The exponential stays bounded even far beyond the explicit limit.
        evolved = advance_temperature(initial, 20 * step, matrix, 300.)
        self.assertGreaterEqual(evolved.min(), 300. - 1e-10)
        self.assertLessEqual(evolved.max(), 500. + 1e-10)
        eigenvalues = np.linalg.eigvalsh(matrix.toarray())
        self.assertGreaterEqual(
            eigenvalues.min(), diagnostics["eigenvalue_lower_bound_s_inv"] - 1e-12)
        self.assertLessEqual(eigenvalues.max(), diagnostics["eigenvalue_upper_bound_s_inv"] + 1e-12)
        self.assertLessEqual(eigenvalues.max(), 0.)


if __name__ == "__main__":
    unittest.main()
