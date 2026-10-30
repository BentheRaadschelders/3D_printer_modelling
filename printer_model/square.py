from dataclasses import dataclass, field
from enum import IntEnum

import numpy as np
from scipy.integrate import solve_ivp

from printer_model.dynamics import dynamics
from printer_model.nozzle import NozzleParameters, nozzle_dynamics
from printer_model.parameters import param


class SquareMode(IntEnum):
    RIGHT = 0
    UP = 1
    LEFT = 2
    DOWN = 3
    DONE = 4


SQUARE_LABELS = ["Right", "Up", "Left", "Down", "Done"]
# [theta_l, theta_r, omega_l, omega_r, i_l, i_r, Tn, iLn, uRn]


@dataclass(frozen=True)
class SquareParameters:
    nozzle: NozzleParameters = field(default_factory=NozzleParameters)
    lower_left: tuple = (0.05, 0.05)
    side_length: float = 0.20
    position_tolerance: float = 1e-6
    angular_speed_tolerance: float = 1e-4
    kp_voltage: float = 40.0
    kd_voltage: float = 4.0
    final_hold_time: float = 0.5

    def __post_init__(self):
        positive = [self.side_length, self.position_tolerance,
                    self.angular_speed_tolerance, self.kp_voltage,
                    self.kd_voltage, self.final_hold_time]
        if not np.all(np.isfinite(positive)) or min(positive) <= 0:
            raise ValueError("Square size, durations, gains and tolerances must be positive.")
        start = np.asarray(self.lower_left)
        if (start.shape != (2,) or not np.all(np.isfinite(start))
                or np.any(start <= 0)
                or np.any(start + self.side_length >= [param["dx"], param["dy"]])):
            raise ValueError("The complete square must lie strictly inside the bed.")

    @property
    def vertices(self):
        x, y = self.lower_left
        length = self.side_length
        return np.array([[x, y], [x + length, y], [x + length, y + length],
                         [x, y + length], [x, y]])


def encoder_kinematics(angles, angular_velocities):
    """Cartesian estimates using calibrated motor encoders, without XY sensors."""
    transform = param["r_m"] / 2 * np.array([[1.0, 1.0], [1.0, -1.0]])
    return transform @ angles, transform @ angular_velocities


def square_target(mode, settings):
    """The next corner is a fixed position target, with no timed trajectory."""
    if mode == SquareMode.DONE:
        return settings.vertices[-1]
    return settings.vertices[int(mode) + 1]


def square_motor_voltage(angles, angular_velocities, mode, settings):
    """Position error and velocity damping, as in the Q3.1 return controller."""
    position, velocity = encoder_kinematics(angles, angular_velocities)
    target = square_target(mode, settings)
    position_error = target - position
    axis_voltage = (
        settings.kp_voltage * position_error
        - settings.kd_voltage * velocity
    )
    ux, uy = axis_voltage
    return np.array([(ux + uy) / 2, (ux - uy) / 2])


def square_arrival_guard(angles, angular_velocities, mode, settings):
    """Nonpositive iff encoder position and speed are within arrival tolerances."""
    position, _ = encoder_kinematics(angles, angular_velocities)
    target = settings.vertices[int(mode) + 1]
    position_error = np.max(np.abs(position - target))
    angular_speed = np.max(np.abs(angular_velocities))
    position_guard = position_error / settings.position_tolerance - 1
    speed_guard = angular_speed / settings.angular_speed_tolerance - 1
    return max(position_guard, speed_guard)


def square_dynamics(t, state, mode, settings):
    """Combined motor/nozzle dynamics, with the current side as a discrete mode."""
    voltage = square_motor_voltage(state[:2], state[2:4], mode, settings)
    state_dot = np.zeros_like(state)
    state_dot[:6] = dynamics(t, state[:6], lambda _: voltage, param)
    state_dot[6:9] = nozzle_dynamics(state[6:9], settings.nozzle.on_voltage, settings.nozzle)
    return state_dot


@dataclass
class SquareRun:
    parameters: SquareParameters
    time: np.ndarray
    state: np.ndarray
    mode: np.ndarray
    voltage: np.ndarray
    extrude: np.ndarray
    events: list

    @property
    def position(self):
        return np.array([encoder_kinematics(state[:2], state[2:4])[0] for state in self.state])

    @property
    def velocity(self):
        return np.array([encoder_kinematics(state[:2], state[2:4])[1] for state in self.state])


def simulate_square(settings=None):
    settings = settings or SquareParameters()
    state = np.zeros(9)
    transform = param["r_m"] / 2 * np.array([[1.0, 1.0], [1.0, -1.0]])
    state[:2] = np.linalg.solve(transform, settings.vertices[0])
    state[6:9] = settings.nozzle.equilibrium
    time = 0.0
    blocks, events = [], []

    def advance(mode, end, event=None):
        nonlocal time, state
        kwargs = {}
        if event is not None:
            event.terminal = True
            event.direction = -1
            kwargs["events"] = event
        solution = solve_ivp(square_dynamics, (time, end), state, args=(mode, settings),
                             method="Radau", rtol=1e-9, atol=1e-11,
                             max_step=0.05, dense_output=True, **kwargs)
        if not solution.success:
            raise RuntimeError(solution.message)
        end = solution.t[-1]
        samples = np.linspace(time, end, max(2, int(np.ceil((end - time) / 0.01)) + 1))
        values = solution.sol(samples).T
        voltages = np.array([square_motor_voltage(state[:2], state[2:4], mode, settings)
                             for state in values])
        blocks.append((samples, values, np.full(len(samples), int(mode)), voltages,
                       np.full(len(samples), mode != SquareMode.DONE)))
        time, state = end, solution.y[:, -1].copy()
        return event is not None and len(solution.t_events[0]) > 0

    for mode in (SquareMode.RIGHT, SquareMode.UP, SquareMode.LEFT, SquareMode.DOWN):
        def arrived(t, state, *_):
            return square_arrival_guard(state[:2], state[2:4], mode, settings)
        # A solver safety limit, not a motion duration or transition guard.
        if not advance(mode, time + 30.0, arrived):
            raise RuntimeError(f"Square side {SQUARE_LABELS[mode]} failed its arrival guard.")
        before = state.copy()
        position, _ = encoder_kinematics(state[:2], state[2:4])
        events.append({"time": time, "from": SQUARE_LABELS[mode],
                       "to": SQUARE_LABELS[int(mode) + 1], "event": "Encoder arrival and rest",
                       "position_m": position.tolist(),
                       "angular_velocity_rad_s": state[2:4].tolist(),
                       "before": before.tolist(), "after": state.tolist()})
    advance(SquareMode.DONE, time + settings.final_hold_time)
    combined = [np.concatenate([block[i] for block in blocks]) for i in range(5)]
    return SquareRun(settings, *combined, events)
