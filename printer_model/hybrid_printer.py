from dataclasses import dataclass, field
from enum import IntEnum

import numpy as np
from scipy.integrate import solve_ivp

from printer_model.dynamics import dynamics_decoupled
from printer_model.nozzle import NozzleParameters, nozzle_dynamics
from printer_model.parameters import param


class Mode(IntEnum):
    PRINTING = 0
    STOPPED = 1
    MOVING_TO_ORIGIN = 2
    WIPING = 3
    HEAT_OFF = 4
    LOADING = 5
    HEAT_ON = 6
    RETURNING = 7


MODE_LABELS = ["Printing", "Stopped", "Moving to origin", "Wiping",
               "Heat off", "Loading filament", "Heat on", "Returning"]

# [x, vx, ix, y, vy, iy, Tn, iLn, uRn, phi, printed]
POSITION = [0, 3]
VELOCITY = [1, 4]
CURRENT = [2, 5]
PHASE, PRINTED = 9, 10


@dataclass(frozen=True)
class ExternalSensorEvent:
    """An externally supplied sensor pulse at absolute simulation time [s]."""

    time: float
    signal: str

    def __post_init__(self):
        if not np.isfinite(self.time) or self.time < 0:
            raise ValueError("External sensor times must be finite and nonnegative.")
        if self.signal not in ("filament_finished", "wipe_complete", "loaded"):
            raise ValueError("Expected a filament_finished, wipe_complete or loaded sensor pulse.")


@dataclass(frozen=True)
class HybridParameters:
    nozzle: NozzleParameters = field(default_factory=NozzleParameters)
    radius: float = 0.12
    constant_rate_period: float = 30.0
    return_kp: float = 40.0  # V/m: drive towards the saved position.
    return_kd: float = 2.0  # V s/m: brake as the head approaches it.
    cold_temperature: float = 353.15
    warm_margin: float = 2.0
    homing_voltage: float = -0.6  # Decoupled axis voltage, always towards origin.
    sensor_period: float = 0.05
    position_tolerance: float = 2e-5
    speed_tolerance: float = 1e-4
    current_tolerance: float = 2e-3
    kp: float = 400.0
    kd: float = 40.0
    ki: float = 600.0

    def __post_init__(self):
        positive = [self.radius, self.constant_rate_period,
                    self.return_kp, self.return_kd, self.warm_margin,
                    self.sensor_period, self.position_tolerance,
                    self.speed_tolerance, self.current_tolerance,
                    self.kp, self.kd, self.ki]
        if not np.all(np.isfinite(positive)) or min(positive) <= 0:
            raise ValueError("Durations, tolerances, gains and size must be positive.")
        if not self.nozzle.ambient_temperature < self.cold_temperature < self.warm_temperature:
            raise ValueError("Require ambient < cold threshold < warm threshold.")
        if not np.isfinite(self.homing_voltage) or self.homing_voltage >= 0:
            raise ValueError("Homing voltage must be negative (left/down).")
        if self.radius >= min(param["dx"], param["dy"]) / 2:
            raise ValueError("The rose must fit strictly inside the bed.")

    @property
    def phase_rate(self):
        return 2 * np.pi / self.constant_rate_period

    @property
    def warm_temperature(self):
        return self.nozzle.equilibrium[0] - self.warm_margin


def rose_geometry(phase, radius=0.12):
    """Eight-petal rose: position and its first three phase derivatives.

    Identical geometry/orientation to Q1.4; phase zero is the leftmost tip.
    """
    k = 4
    r = radius * np.cos(k * phase)
    dr = -radius * k * np.sin(k * phase)
    ddr, dddr = -k**2 * r, -k**2 * dr
    orientation = np.pi + phase
    radial = np.array([np.cos(orientation), np.sin(orientation)])
    tangent = np.array([-np.sin(orientation), np.cos(orientation)])
    center = np.array([param["dx"], param["dy"]]) / 2
    return (center + r * radial,
            dr * radial + r * tangent,
            (ddr - r) * radial + 2 * dr * tangent,
            (dddr - 3 * dr) * radial + (3 * ddr - r) * tangent)


def motor_coefficients():
    r = param["r_m"]
    mass = np.array([param["m_head"], param["m_head"] + param["m_rails"]])
    mass += 2 * param["J_m"] / r**2
    damping = np.array([2 * param["mu_lx"] + 8 * param["mu_r"],
                        2 * param["mu_ly"] + 4 * param["mu_r"]])
    return mass, damping


def parked_transition(state, symbol):
    """Q2.2 automaton; H is leftward, V is downward, '+' means moving."""
    alphabet = [("H", "+"), ("H", "0"), ("V", "+"), ("V", "0")]
    table = {"Unknown": ["Unknown", "Left", "Unknown", "Down"],
             "Down": ["Down", "Parked", "Unknown", "Down"],
             "Left": ["Unknown", "Left", "Left", "Parked"],
             "Parked": ["Down", "Parked", "Left", "Parked"]}
    return table[state][alphabet.index(symbol)]


def homing_command(parked_state):
    """Choose the homing direction from the Q2.2 detector state."""
    return {"Unknown": "V", "Down": "H", "Left": "V", "Parked": None}[parked_state]


@dataclass
class Context:
    mode: Mode = Mode.PRINTING
    saved_position: np.ndarray = field(default_factory=lambda: np.zeros(2))
    saved_phase: float = 0.0
    # Q2.2 parked-detector state.
    parked_state: str = "Unknown"


def motion_reference(state, context, settings):
    """Return position, velocity, acceleration, jerk, and phase rate."""
    mode = context.mode
    if mode == Mode.PRINTING:
        # Resume the rose at its original phase rate.
        rate = settings.phase_rate
        position, dp, ddp, dddp = rose_geometry(state[PHASE], settings.radius)
        return position, dp * rate, ddp * rate**2, dddp * rate**3, rate
    target = (context.saved_position if mode in (Mode.STOPPED, Mode.RETURNING)
              else np.zeros(2))
    return target, np.zeros(2), np.zeros(2), np.zeros(2), 0.0


def commands(state, context, settings):
    """Physical voltages and logical extrusion command in this mode."""
    position, velocity, current = state[POSITION], state[VELOCITY], state[CURRENT]
    if context.mode == Mode.RETURNING:
        # Drive towards the saved point and brake using velocity feedback.
        position_error = context.saved_position - position
        axis_voltage = (
            settings.return_kp * position_error
            - settings.return_kd * velocity
        )
    elif context.mode == Mode.MOVING_TO_ORIGIN:
        # Keep driving until an encoder sample confirms the end-stop.
        axis_voltage = np.zeros(2)
        motion = homing_command(context.parked_state)
        if motion is not None:
            axis_voltage[1 if motion == "V" else 0] = settings.homing_voltage
    elif context.mode == Mode.PRINTING:
        position_ref, velocity_ref, acceleration_ref, jerk_ref, _ = motion_reference(
            state, context, settings
        )
        mass, damping = motor_coefficients()
        motor_force = param["K_m"] * current / param["r_m"]
        friction_force = damping * velocity
        acceleration = (motor_force - friction_force) / mass
        for axis in (0, 1):
            if position[axis] <= 1e-12 and abs(velocity[axis]) <= 1e-12:
                acceleration[axis] = max(0.0, acceleration[axis])
        position_error = position_ref - position
        velocity_error = velocity_ref - velocity
        acceleration_error = acceleration_ref - acceleration

        acceleration_cmd = (
            acceleration_ref
            + settings.kp * position_error
            + settings.kd * velocity_error
        )
        acceleration_cmd_dot = (
            jerk_ref
            + settings.kp * velocity_error
            + settings.kd * acceleration_error
        )

        desired_current = param["r_m"] / param["K_m"] * (
            mass * acceleration_cmd + damping * velocity
        )
        desired_current_dot = param["r_m"] / param["K_m"] * (
            mass * acceleration_cmd_dot + damping * acceleration
        )
        current_error = desired_current - current
        axis_voltage = (
            param["R_m"] * current
            + 2 * param["K_e"] / param["r_m"] * velocity
            + param["L_m"] * (desired_current_dot + settings.ki * current_error)
        )
    else:
        # Zero motor voltage allows back EMF and friction to brake the head.
        axis_voltage = np.zeros(2)
    motor_voltage = np.array([sum(axis_voltage) / 2,
                              (axis_voltage[0] - axis_voltage[1]) / 2])
    heater_voltage = (0.0 if context.mode in (Mode.HEAT_OFF, Mode.LOADING)
                      else settings.nozzle.on_voltage)
    extrude = context.mode in (Mode.PRINTING, Mode.WIPING)
    return motor_voltage, heater_voltage, extrude


def hybrid_dynamics(t, state, context, settings):
    """Motor, nozzle and path-state derivatives for the current FSM mode."""
    motor_voltage, heater_voltage, _ = commands(state, context, settings)
    state_dot = np.zeros_like(state)
    state_dot[:6] = dynamics_decoupled(t, state[:6], lambda _: motor_voltage, param)
    # At rest against an end-stop, its reaction cancels force into the stop.
    for axis in (0, 1):
        if (state[3 * axis] <= 1e-12 and abs(state[3 * axis + 1]) <= 1e-12
                and state_dot[3 * axis + 1] < 0):
            state_dot[3 * axis] = 0.0
            state_dot[3 * axis + 1] = 0.0
    state_dot[6:9] = nozzle_dynamics(state[6:9], heater_voltage, settings.nozzle)
    state_dot[PHASE] = motion_reference(state, context, settings)[-1]
    if context.mode == Mode.PRINTING:
        state_dot[PRINTED] = np.linalg.norm(state[VELOCITY])
    return state_dot


def at_rest(state, settings):
    """Settled speed and current; passive braking need not stop at a target."""
    speed_guard = np.max(np.abs(state[VELOCITY])) / settings.speed_tolerance
    current_guard = np.max(np.abs(state[CURRENT])) / settings.current_tolerance
    return max(speed_guard, current_guard) - 1.0


def position_reached(state, target, settings):
    """Return-motion guard using position and speed, without current sensing."""
    position_error = np.max(np.abs(state[POSITION] - target))
    position_guard = position_error / settings.position_tolerance
    speed_guard = np.max(np.abs(state[VELOCITY])) / settings.speed_tolerance
    return max(position_guard, speed_guard) - 1.0


@dataclass
class Run:
    parameters: HybridParameters
    time: np.ndarray
    state: np.ndarray
    mode: np.ndarray
    motor_voltage: np.ndarray
    heater_voltage: np.ndarray
    extrude: np.ndarray
    events: list
    sensor_samples: list
    saved_position: np.ndarray
    saved_phase: float
    external_sensor_events: tuple


def simulate_replacement(settings=None, *, sensor_events):
    """Run one replacement cycle using external sensor pulses.

    Pulses are accepted only in their matching modes. A missing completion
    pulse raises a waiting message.
    """
    settings = settings or HybridParameters()
    sensor_events = tuple(sorted(sensor_events, key=lambda event: event.time))
    finished = None
    for event in sensor_events:
        if event.signal == "filament_finished":
            finished = event
            break
    if finished is None or finished.time >= settings.constant_rate_period:
        raise ValueError("This replacement example needs an external 'filament_finished' "
                         "pulse before the first rose traversal completes.")
    context = Context()
    state = np.zeros(11)
    position, dp, ddp, _ = rose_geometry(0, settings.radius)
    mass, damping = motor_coefficients()
    velocity, acceleration = dp * settings.phase_rate, ddp * settings.phase_rate**2
    state[POSITION], state[VELOCITY] = position, velocity
    state[CURRENT] = param["r_m"] / param["K_m"] * (mass * acceleration + damping * velocity)
    state[6:9] = settings.nozzle.equilibrium
    time = 0.0
    blocks, events, sensors = [], [], []

    def advance(end_time, event=None, spacing=0.03):
        nonlocal state, time
        kwargs = {}
        if event is not None:
            event.terminal = True
            event.direction = -1
            kwargs["events"] = event
        solution = solve_ivp(hybrid_dynamics, (time, end_time), state,
                             args=(context, settings), method="Radau", dense_output=True,
                             rtol=2e-8, atol=1e-10, max_step=1.0, **kwargs)
        if not solution.success:
            raise RuntimeError(solution.message)
        stop = solution.t[-1]
        times = np.linspace(time, stop, max(2, int(np.ceil((stop - time) / spacing)) + 1))
        values = solution.sol(times).T
        outputs = [commands(state, context, settings) for state in values]
        blocks.append((times, values, np.full(len(times), int(context.mode)),
                       np.array([o[0] for o in outputs]),
                       np.array([o[1] for o in outputs]),
                       np.array([o[2] for o in outputs])))
        state, time = solution.y[:, -1].copy(), stop
        return event is not None and len(solution.t_events[0]) > 0

    def jump(new_mode, reason, external_signal=None):
        before = state.copy()
        old_mode = context.mode
        if new_mode == Mode.STOPPED:
            context.saved_position = state[POSITION].copy()
            context.saved_phase = float(state[PHASE])
        if new_mode == Mode.MOVING_TO_ORIGIN:
            context.parked_state = "Unknown"
        context.mode = new_mode
        events.append({"time": time, "from": MODE_LABELS[old_mode],
                       "to": MODE_LABELS[new_mode], "event": reason,
                       "source": "external_sensor" if external_signal else "state_guard",
                       "external_signal": external_signal,
                       "before": before.tolist(), "after": state.tolist()})

    def wait_for_sensor(signal, new_mode, reason):
        # Ignore pulses that arrived before this mode became active.
        event = None
        for candidate in sensor_events:
            if candidate.signal == signal and candidate.time >= time:
                event = candidate
                break
        if event is None:
            raise RuntimeError(f"{MODE_LABELS[context.mode]} is waiting for external "
                               f"sensor signal '{signal}'; no future pulse was supplied.")
        if event.time > time:
            advance(event.time, spacing=0.03 if context.mode == Mode.PRINTING else 0.25)
        jump(new_mode, reason, external_signal=signal)

    # Runout is triggered by the external F pulse.
    wait_for_sensor("filament_finished", Mode.STOPPED, "F: filament-finished sensor")

    def stopped(t, state, *_):
        return at_rest(state, settings)
    if not advance(time + 10, stopped):
        raise RuntimeError("The head did not settle under zero motor voltage.")
    jump(Mode.MOVING_TO_ORIGIN, "Location saved and head stopped")

    # Use the Q2.2 detector state to choose the homing direction.
    deadline = time + 60.0
    while context.parked_state != "Parked" and time < deadline:
        motion = homing_command(context.parked_state)
        axis = 1 if motion == "V" else 0
        next_sample = time + settings.sensor_period

        def hit_stop(t, state, *_):
            return state[3 * axis]
        # After impact, continue to the next encoder sample without retriggering.
        event = (hit_stop if state[3 * axis] > 1e-12
                 or abs(state[3 * axis + 1]) > 1e-12 else None)
        hit = advance(next_sample, event, spacing=settings.sensor_period)
        if hit:
            before = state.copy()
            state[3 * axis], state[3 * axis + 1] = 0.0, 0.0
            events.append({"time": time, "event": f"{motion} end-stop impact",
                           "from": MODE_LABELS[context.mode],
                           "to": MODE_LABELS[context.mode],
                           "before": before.tolist(), "after": state.tolist()})
            if time < next_sample:
                advance(next_sample, spacing=settings.sensor_period)
        speed = abs(state[3 * axis + 1])
        symbol = (motion, "+" if speed > settings.speed_tolerance else "0")
        detector_before = context.parked_state
        context.parked_state = parked_transition(detector_before, symbol)
        sensors.append({"time": time, "symbol": list(symbol),
                        "detector_before": detector_before,
                        "state": context.parked_state, "axis_speed": speed})
    if context.parked_state != "Parked":
        raise RuntimeError("The parked detector did not accept the homing sequence.")
    jump(Mode.WIPING, "Q2.2 detector accepted: reached origin")

    wait_for_sensor("wipe_complete", Mode.HEAT_OFF, "Excess filament removed / wiping complete")

    def cold(t, state, *_):
        return state[6] - settings.cold_temperature
    if not advance(time + 20 * settings.nozzle.thermal_capacity * settings.nozzle.thermal_resistance,
                   cold, spacing=0.5):
        raise RuntimeError("The nozzle did not cool to the loading threshold.")
    jump(Mode.LOADING, "Cooled down")
    wait_for_sensor("loaded", Mode.HEAT_ON, "Loaded: new spool installed")

    def warm(t, state, *_):
        return settings.warm_temperature - state[6]
    if not advance(time + 20 * settings.nozzle.thermal_capacity * settings.nozzle.thermal_resistance,
                   warm, spacing=0.5):
        raise RuntimeError("The nozzle did not reach the warm threshold.")
    jump(Mode.RETURNING, "Warmed up")

    def returned(t, state, *_):
        return position_reached(state, context.saved_position, settings)
    if not advance(time + 30, returned):
        raise RuntimeError("The head did not reach the saved location.")
    jump(Mode.PRINTING, "Saved location reached; resume saved phase")

    def completed(t, state, *_):
        return 2 * np.pi - state[PHASE]
    if not advance(time + settings.constant_rate_period, completed):
        raise RuntimeError("The remaining rose traversal did not complete.")
    events.append({"time": time, "from": "Printing", "to": "Complete",
                   "event": "Rose complete", "before": state.tolist(),
                   "after": state.tolist()})
    combined = [np.concatenate([block[i] for block in blocks]) for i in range(6)]
    return Run(settings, *combined, events, sensors, context.saved_position.copy(),
               context.saved_phase, sensor_events)
