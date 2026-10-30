from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import BSpline

from printer_model.dynamics import dynamics_decoupled
from printer_model.parameters import param
from simulations.utils import (
    inverse_dynamics_voltage, simulate_dynamics, plot_solution, plot_trajectory,
)

duration = 0.5
a = 0.1
b = 0.06
center = np.array([0.15, 0.15])

rose_petals = 8
rose_radius = 0.12
rose_duration = 3.0

# Indicative outline, hand-traced from the TU Delft flame in arbitrary units.
# Reference: https://quspinlab.tudelft.nl/images/logopic/TUDelft_logo.png
# A periodic quintic B-spline rounds the tips and keeps jerk continuous.
flame_control_points = np.array([
    [0.04, 0.08], [0.00, 0.20], [0.00, 0.34], [0.04, 0.47],
    [0.13, 0.59], [0.28, 0.70], [0.48, 0.79], [0.69, 0.86],
    [0.85, 0.92], [0.96, 0.98], [0.99, 0.98], [1.00, 0.96],
    [0.97, 0.91], [0.91, 0.86], [0.79, 0.76], [0.66, 0.65],
    [0.60, 0.55], [0.61, 0.47], [0.69, 0.43], [0.79, 0.44],
    [0.87, 0.49], [0.91, 0.53], [0.91, 0.49], [0.87, 0.38],
    [0.80, 0.25], [0.68, 0.14], [0.53, 0.05], [0.36, 0.00],
    [0.32, 0.00], [0.38, 0.04], [0.46, 0.11], [0.53, 0.22],
    [0.56, 0.33], [0.57, 0.40], [0.56, 0.40], [0.51, 0.34],
    [0.44, 0.29], [0.37, 0.28], [0.32, 0.30], [0.30, 0.35],
    [0.32, 0.41], [0.38, 0.51], [0.43, 0.61], [0.44, 0.65],
    [0.42, 0.64], [0.36, 0.57], [0.27, 0.49], [0.17, 0.40],
    [0.09, 0.30], [0.05, 0.19], [0.05, 0.09],
])
flame_duration = 5.0
flame_size_scale = 2.0


def ellipse_motion(t):
    phase_rate = 2 * np.pi / duration
    phase = np.pi + phase_rate * t

    position = center + np.array([a * np.cos(phase), b * np.sin(phase)])
    dposition = np.array([-a * np.sin(phase), b * np.cos(phase)])
    ddposition = np.array([-a * np.cos(phase), -b * np.sin(phase)])

    velocity = dposition * phase_rate
    acceleration = ddposition * phase_rate**2
    jerk = -dposition * phase_rate**3
    return position, velocity, acceleration, jerk


def ellipse_voltage(t):
    _, velocity, acceleration, jerk = ellipse_motion(t)
    return inverse_dynamics_voltage(velocity, acceleration, jerk, param)


def initial_state_from_motion(motion):
    position, velocity, acceleration, _ = motion(0.0)
    r = param["r_m"]
    mass = np.array([
        param["m_head"], param["m_head"] + param["m_rails"],
    ]) + 2 * param["J_m"] / r**2
    damping = np.array([
        2 * param["mu_lx"] + 8 * param["mu_r"],
        2 * param["mu_ly"] + 4 * param["mu_r"],
    ])
    current = r / param["K_m"] * (mass * acceleration + damping * velocity)
    return np.array([
        position[0], velocity[0], current[0],
        position[1], velocity[1], current[1],
    ])


def rose_motion(t, petals=rose_petals, radius=rose_radius, traversal_time=rose_duration):
    """Rose with an odd number of petals or a petal count divisible by four."""
    if (isinstance(petals, (bool, np.bool_))
            or not isinstance(petals, (int, np.integer)) or petals <= 0):
        raise ValueError("Petals must be a positive integer.")
    if petals % 2 == 0 and petals % 4 != 0:
        raise ValueError("This rose formula supports odd petal counts or multiples of four.")
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError("Radius must be positive and finite.")
    if not np.isfinite(traversal_time) or traversal_time <= 0:
        raise ValueError("Traversal time must be positive and finite.")
    k = petals if petals % 2 else petals / 2
    phase_span = np.pi if petals % 2 else 2 * np.pi
    phase_rate = phase_span / traversal_time
    angle = phase_rate * t

    r = radius * np.cos(k * angle)
    dr = -radius * k * np.sin(k * angle)
    ddr = -k**2 * r
    dddr = -k**2 * dr
    orientation = np.pi + angle
    radial = np.array([np.cos(orientation), np.sin(orientation)])
    tangent = np.array([-np.sin(orientation), np.cos(orientation)])
    dposition = dr * radial + r * tangent
    ddposition = (ddr - r) * radial + 2 * dr * tangent
    dddposition = (dddr - 3 * dr) * radial + (3 * ddr - r) * tangent

    position = center + r * radial
    velocity = dposition * phase_rate
    acceleration = ddposition * phase_rate**2
    jerk = dddposition * phase_rate**3
    return position, velocity, acceleration, jerk


def rose_voltage(t, petals=rose_petals, radius=rose_radius, traversal_time=rose_duration):
    _, velocity, acceleration, jerk = rose_motion(t, petals, radius, traversal_time)
    return inverse_dynamics_voltage(velocity, acceleration, jerk, param)


def make_flame_path():
    degree = 5
    count = len(flame_control_points)
    knots = np.arange(-degree, count + degree + 1, dtype=float) / count
    coefficients = np.vstack((flame_control_points, flame_control_points[:degree]))
    raw_path = BSpline(knots, coefficients, degree, extrapolate="periodic")

    samples = raw_path(np.linspace(0.0, 1.0, 10001))
    lower, upper = samples.min(axis=0), samples.max(axis=0)
    scale = flame_size_scale * np.min(np.array([2 * a, 2 * b]) / (upper - lower))
    coefficients = center + scale * (coefficients - (lower + upper) / 2)
    return BSpline(knots, coefficients, degree, extrapolate="periodic")


def flame_motion(t):
    flame_path = make_flame_path()
    normalized_time = t / flame_duration
    progress = 10 * normalized_time**3 - 15 * normalized_time**4 + 6 * normalized_time**5
    progress_dot = (30 * normalized_time**2 - 60 * normalized_time **
                    3 + 30 * normalized_time**4) / flame_duration
    progress_ddot = (60 * normalized_time - 180 * normalized_time **
                     2 + 120 * normalized_time**3) / flame_duration**2
    progress_dddot = (60 - 360 * normalized_time + 360 * normalized_time**2) / flame_duration**3

    position = flame_path(progress)
    dposition = flame_path(progress, nu=1)
    ddposition = flame_path(progress, nu=2)
    dddposition = flame_path(progress, nu=3)
    velocity = dposition * progress_dot
    acceleration = ddposition * progress_dot**2 + dposition * progress_ddot
    jerk = (
        dddposition * progress_dot**3
        + 3 * ddposition * progress_dot * progress_ddot
        + dposition * progress_dddot
    )
    return position, velocity, acceleration, jerk


def flame_voltage(t):
    """Actual left/right motor voltages for the rounded flame trajectory."""
    if t < 0 or t > flame_duration:
        return 0.0, 0.0
    _, velocity, acceleration, jerk = flame_motion(t)
    return inverse_dynamics_voltage(velocity, acceleration, jerk, param)


if __name__ == "__main__":
    output_dir = Path(__file__).resolve().parents[1] / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    initial_state = initial_state_from_motion(ellipse_motion)
    flame_start = flame_motion(0.0)[0]
    flame_initial_state = np.array([flame_start[0], 0.0, 0.0, flame_start[1], 0.0, 0.0])
    rose_start_state = initial_state_from_motion(
        lambda t: rose_motion(t, rose_petals, rose_radius, rose_duration))
    for name, filename, voltage_signal, start, traversal_time in (
        ("Ellipse", "Figure_ellipse1.png", ellipse_voltage, initial_state, duration),
        ("Rose", "Figure_flower1.png", rose_voltage, rose_start_state, rose_duration),
        ("TU Delft flame", "Figure_flame1.png", flame_voltage, flame_initial_state, flame_duration)
    ):
        solution = simulate_dynamics(
            dynamics_decoupled, voltage_signal, start, param,
            t_span=(0.0, traversal_time),
        )
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle(name)
        plot_solution(
            solution, labels=["x", "y"],
            ylabel=["Position [m]", "Velocity [m/s]", "Current [A]"],
            axes=axes.flat[:3], show=False,
        )
        plot_trajectory(
            solution,
            title=f"{name} trajectory",
            deck_dimensions=(param["dx"], param["dy"]),
            ax=axes[1, 1], show=False,
        )
        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=100)
    print(f"Figures saved in {output_dir}")
    plt.show()
