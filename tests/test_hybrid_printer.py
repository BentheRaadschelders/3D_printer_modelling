import unittest

import numpy as np

from printer_model.hybrid_printer import (
    PHASE, PRINTED, Context, ExternalSensorEvent, HybridParameters, Mode, commands,
    at_rest, homing_command, motion_reference, parked_transition, position_reached,
    rose_geometry, simulate_replacement, hybrid_dynamics,
)
from printer_model.nozzle import NozzleParameters, nozzle_dynamics
from simulations.q1_4_draw_shapes import rose_motion


class NozzleTests(unittest.TestCase):
    def test_constant_voltage_equilibrium_and_initial_cooling(self):
        settings = NozzleParameters()
        self.assertAlmostEqual(settings.equilibrium[0], 512.6)
        np.testing.assert_allclose(nozzle_dynamics(
            settings.equilibrium, 14, settings), 0, atol=1e-12)
        # Switching the supply off does not instantly remove stored circuit energy.
        state_dot = nozzle_dynamics(settings.equilibrium, 0, settings)
        self.assertAlmostEqual(state_dot[0], 0)
        self.assertLess(state_dot[1], 0)
        self.assertAlmostEqual(state_dot[2], 0)

    def test_invalid_parameter_sets(self):
        for kwargs in ({"capacitance": 0}, {"resistance": -1}, {"thermal_capacity": np.nan}):
            with self.assertRaises(ValueError):
                NozzleParameters(**kwargs)


class ParkedDetectorTests(unittest.TestCase):
    def test_detection_and_revocation_of_boundary_knowledge(self):
        state = "Unknown"
        for symbol in (("V", "+"), ("V", "0"), ("H", "+"), ("H", "0")):
            state = parked_transition(state, symbol)
        self.assertEqual(state, "Parked")
        self.assertEqual(parked_transition(state, ("V", "+")), "Left")
        self.assertEqual(parked_transition(state, ("H", "+")), "Down")

    def test_homing_command_depends_on_existing_detector_state(self):
        settings = HybridParameters()
        state = np.zeros(11)
        # Even at the bottom, the controller awaits the encoder event before
        # switching command. Physical position does not bypass Q2.2.
        context = Context(mode=Mode.MOVING_TO_ORIGIN)
        for detector, expected in (("Unknown", [0, settings.homing_voltage]),
                                   ("Down", [settings.homing_voltage, 0]),
                                   ("Left", [0, settings.homing_voltage]),
                                   ("Parked", [0, 0])):
            context.parked_state = detector
            ul, ur = commands(state, context, settings)[0]
            np.testing.assert_allclose([ul + ur, ul - ur], expected)

    def test_boundary_reaction_uses_position_and_velocity(self):
        settings = HybridParameters()
        context = Context(mode=Mode.MOVING_TO_ORIGIN)
        state = np.zeros(11)
        state[0] = 0.15
        state[6:9] = settings.nozzle.equilibrium
        state[5] = -0.1  # Inward y force while at rest on the bottom boundary.
        state_dot = hybrid_dynamics(0, state, context, settings)
        np.testing.assert_array_equal(state_dot[[3, 4]], 0)
        state[5] = 0.1  # Outward force releases contact without changing memory.
        self.assertGreater(hybrid_dynamics(0, state, context, settings)[4], 0)
        state[3:6] = [1e-14, -0.02, -0.1]
        # An approaching head must reach its impact event, not be frozen early.
        self.assertEqual(hybrid_dynamics(0, state, context, settings)[3], -0.02)


class HybridPrinterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Shorten only the thermal waits in tests, retaining the full ODE circuit.
        cls.settings = HybridParameters(nozzle=NozzleParameters(thermal_capacity=0.2))
        cls.sensor_events = [ExternalSensorEvent(9.375, "filament_finished"),
                             ExternalSensorEvent(17.0, "wipe_complete"),
                             ExternalSensorEvent(23.0, "loaded")]
        cls.result = simulate_replacement(cls.settings, sensor_events=cls.sensor_events)

    def test_geometry_agrees_with_q1_4(self):
        for phase in (0.0, 0.5, 1.9, 4.0, 2 * np.pi):
            geometry = rose_geometry(phase)
            q14 = rose_motion(phase * 3 / (2 * np.pi), traversal_time=3)
            for order in range(4):
                np.testing.assert_allclose(geometry[order] * (2 * np.pi / 3)**order,
                                           q14[order], atol=1e-10)

    def test_complete_state_sequence_and_thermal_guards(self):
        switches = [event for event in self.result.events if event["from"] != event["to"]]
        self.assertEqual([event["to"] for event in switches],
                         ["Stopped", "Moving to origin", "Wiping", "Heat off",
                          "Loading filament", "Heat on", "Returning", "Printing", "Complete"])
        loading = next(event for event in switches if event["to"] == "Loading filament")
        returning = next(event for event in switches if event["to"] == "Returning")
        self.assertAlmostEqual(loading["before"][6], self.settings.cold_temperature, places=7)
        self.assertAlmostEqual(returning["before"][6], self.settings.warm_temperature, places=7)
        self.assertTrue(np.all(np.diff([event["time"] for event in switches]) > 0))

    def test_continuous_states_preserved_at_fsm_switches(self):
        for event in self.result.events:
            before, after = np.array(event["before"]), np.array(event["after"])
            if "impact" in event["event"]:
                # Inelastic impact: normal speed alone jumps, currents stay continuous.
                allowed = [3, 4] if event["event"].startswith("V") else [0, 1]
                keep = [i for i in range(11) if i not in allowed]
                np.testing.assert_array_equal(before[keep], after[keep])
                self.assertEqual(after[allowed[1]], 0)
            else:
                np.testing.assert_array_equal(before[:9], after[:9])
                self.assertEqual(before[PHASE], after[PHASE])
                self.assertEqual(before[PRINTED], after[PRINTED])
                np.testing.assert_array_equal(before, after)

    def test_saved_phase_prevents_resuming_wrong_petal(self):
        run = self.result
        self.assertAlmostEqual(run.saved_phase, 5 * np.pi / 8, places=6)
        np.testing.assert_allclose(run.saved_position, [.15, .15], atol=1e-7)
        paused = run.mode != Mode.PRINTING
        np.testing.assert_allclose(run.state[paused, PHASE], run.saved_phase, atol=1e-10)
        resumed = next(event["time"] for event in run.events if event["from"] == "Returning")
        after = run.state[(run.mode == Mode.PRINTING) & (run.time >= resumed)]
        self.assertAlmostEqual(after[0, PHASE], run.saved_phase, places=7)
        self.assertTrue(np.all(np.diff(after[:, PHASE]) >= -1e-10))
        self.assertAlmostEqual(after[-1, PHASE], 2 * np.pi, places=7)
        elapsed = run.time[(run.mode == Mode.PRINTING) & (run.time >= resumed)] - resumed
        np.testing.assert_allclose(after[:, PHASE], run.saved_phase + \
                                   self.settings.phase_rate * elapsed, atol=1e-9)
        desired = np.array([rose_geometry(s[PHASE])[0] for s in after])
        error = np.linalg.norm(after[:, [0, 3]] - desired, axis=1)
        # The original moving reference restarts from a stationary physical
        # state; retain its finite tracking transient rather than hiding it.
        self.assertGreater(error.max(), 1e-5)
        self.assertLess(error.max(), 0.005)
        self.assertLess(error[elapsed >= 1.0].max(), 2e-6)

    def test_return_uses_pd_voltage_and_arrival_not_fixed_duration(self):
        run, settings = self.result, self.settings
        mask = run.mode == Mode.RETURNING
        state = run.state[mask]
        expected = settings.return_kp * \
            (run.saved_position - state[:, [0, 3]]) - settings.return_kd * state[:, [1, 4]]
        motor = run.motor_voltage[mask]
        actual = np.column_stack((motor[:, 0] + motor[:, 1], motor[:, 0] - motor[:, 1]))
        np.testing.assert_allclose(actual, expected, atol=1e-12)
        endpoint = next(event for event in run.events if event["from"] == "Returning")
        before = np.array(endpoint["before"])
        self.assertLessEqual(position_reached(before, run.saved_position, settings), 1e-8)
        # The reference now moves at the original rate; actual states are
        # retained at rest and only voltages change at the mode switch.
        context = Context(saved_position=run.saved_position, saved_phase=run.saved_phase)
        reference = motion_reference(np.array(endpoint["after"]), context, settings)
        self.assertGreater(np.linalg.norm(reference[1]), 0.05)
        self.assertLessEqual(np.max(abs(before[[1, 4]])), settings.speed_tolerance * (1 + 1e-8))

    def test_return_control_does_not_use_motor_current_feedback(self):
        settings = self.settings
        context = Context(mode=Mode.RETURNING, saved_position=np.array([0.15, 0.15]))
        state = np.zeros(11)
        voltage = commands(state, context, settings)[0]
        state[[2, 5]] = [1.0, -1.0]
        np.testing.assert_array_equal(commands(state, context, settings)[0], voltage)

    def test_zero_voltage_stops_without_requiring_the_saved_position(self):
        run, settings = self.result, self.settings
        stationary_modes = [Mode.STOPPED, Mode.WIPING, Mode.HEAT_OFF,
                            Mode.LOADING, Mode.HEAT_ON]
        np.testing.assert_array_equal(
            run.motor_voltage[np.isin(run.mode, stationary_modes)], 0)
        stopped = next(event for event in run.events if event["from"] == "Stopped")
        state = np.array(stopped["before"])
        self.assertLessEqual(at_rest(state, settings), 1e-8)
        # Passive braking travels beyond the interrupted print point. A
        # position-error guard would prevent this transition from occurring.
        self.assertGreater(np.linalg.norm(state[[0, 3]] - run.saved_position), 0.001)
        interrupted = next(event for event in run.events if event["to"] == "Stopped")
        np.testing.assert_array_equal(run.saved_position,
                                      np.array(interrupted["before"])[[0, 3]])
        self.assertGreater(np.linalg.norm(np.array(interrupted["before"])[[1, 4]]),
                           settings.speed_tolerance)

    def test_runout_wiping_and_loading_use_external_sensor_pulses(self):
        external = [event for event in self.result.events
                    if event.get("source") == "external_sensor"]
        self.assertEqual([(event["external_signal"], event["time"]) for event in external],
                         [("filament_finished", 9.375), ("wipe_complete", 17.0), ("loaded", 23.0)])
        for event in external:
            # No timer or spool-budget state is reset by these input pulses.
            before, after = np.array(event["before"]), np.array(event["after"])
            self.assertEqual(len(before), 11)
            np.testing.assert_array_equal(before, after)

    def test_replacement_is_not_triggered_without_a_runout_sensor_input(self):
        with self.assertRaisesRegex(ValueError, "external 'filament_finished'"):
            simulate_replacement(self.settings, sensor_events=[])

    def test_absent_or_early_sensor_pulses_do_not_complete_wiping(self):
        for supplied in ([], [ExternalSensorEvent(1.0, "wipe_complete")]):
            with self.subTest(sensor_events=supplied):
                with self.assertRaisesRegex(RuntimeError, "Wiping is waiting for external"):
                    simulate_replacement(self.settings, sensor_events=[
                        ExternalSensorEvent(9.375, "filament_finished"), *supplied])

    def test_no_drawing_during_travel_and_homing_accepts_real_stalls(self):
        run = self.result
        expected = np.isin(run.mode, [Mode.PRINTING, Mode.WIPING])
        np.testing.assert_array_equal(run.extrude, expected)
        heating_off = np.isin(run.mode, [Mode.HEAT_OFF, Mode.LOADING])
        np.testing.assert_array_equal(run.heater_voltage[heating_off], 0)
        np.testing.assert_array_equal(run.heater_voltage[~heating_off], 14)
        self.assertEqual(run.sensor_samples[-1]["state"], "Parked")
        zero_samples = [s for s in run.sensor_samples if s["symbol"][1] == "0"]
        self.assertEqual([s["symbol"][0] for s in zero_samples], ["V", "H"])
        self.assertTrue(all(s["axis_speed"] <= self.settings.speed_tolerance for s in zero_samples))
        for sample in run.sensor_samples:
            self.assertEqual(homing_command(sample["detector_before"]), sample["symbol"][0])
            self.assertEqual(parked_transition(sample["detector_before"], tuple(sample["symbol"])),
                             sample["state"])
            index = np.argmin(abs(run.time - sample["time"]))
            ul, ur = run.motor_voltage[index]
            ux, uy = ul + ur, ul - ur
            expected = (0, self.settings.homing_voltage) if sample["symbol"][0] == "V" else (
                self.settings.homing_voltage, 0)
            np.testing.assert_allclose([ux, uy], expected, atol=1e-12)
        self.assertTrue(np.all(run.state[:, [0, 3]] >= -1e-8))
        # Travel must not increase the diagnostic printed-path length.
        for mode in Mode:
            if mode == Mode.PRINTING:
                continue
            printed = run.state[run.mode == mode, PRINTED]
            np.testing.assert_allclose(printed, printed[0], atol=1e-10)


if __name__ == "__main__":
    unittest.main()
