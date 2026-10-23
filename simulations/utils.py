import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.integrate import solve_ivp


def inverse_dynamics_voltage(velocity, acceleration, jerk, param):
    r = param["r_m"]
    mass = np.array([
        param["m_head"],
        param["m_head"] + param["m_rails"],
    ]) + 2 * param["J_m"] / r**2
    damping = np.array([
        2 * param["mu_lx"] + 8 * param["mu_r"],
        2 * param["mu_ly"] + 4 * param["mu_r"],
    ])

    current = r / param["K_m"] * (mass * acceleration + damping * velocity)
    current_dot = r / param["K_m"] * (mass * jerk + damping * acceleration)
    u_x, u_y = (
        param["L_m"] * current_dot
        + param["R_m"] * current
        + 2 * param["K_e"] / r * velocity
    )
    return 0.5 * (u_x + u_y), 0.5 * (u_x - u_y)


def simulate_dynamics(
    dynamics_function, voltage_signal, initial_state, param, t_span=(0.0, 1.0)
):
    solution = solve_ivp(
        dynamics_function,
        t_span=t_span,
        y0=initial_state,
        args=(voltage_signal, param),
        method="Radau",
        rtol=1e-6,
        atol=1e-9,
    )
    if not solution.success:
        raise RuntimeError(solution.message)
    return solution

def plot_solution(solution, labels, ylabel, axes=None, show=True):
    if axes is None:
        fig, axes = plt.subplots(3, 1, sharex=True, figsize=(8, 8))
    else:
        fig = axes[0].figure

    axes[0].plot(solution.t, solution.y[0], label=labels[0])
    axes[0].plot(solution.t, solution.y[3], "--", label=labels[1])
    axes[0].set_ylabel(ylabel[0])

    axes[1].plot(solution.t, solution.y[1], label=labels[0])
    axes[1].plot(solution.t, solution.y[4], "--", label=labels[1])
    axes[1].set_ylabel(ylabel[1])

    axes[2].plot(solution.t, solution.y[2], label=labels[0])
    axes[2].plot(solution.t, solution.y[5], "--", label=labels[1])
    axes[2].set_ylabel(ylabel[2])
    axes[2].set_xlabel("Time [s]")

    for ax in axes:
        ax.grid(True)
        ax.legend()

    if show:
        fig.tight_layout()
        plt.show()
    return fig, axes


def plot_trajectory(
    solution, title="Tool-head trajectory", deck_dimensions=None, ax=None, show=True
):
    x = solution.y[0]
    y = solution.y[3]

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    else:
        fig = ax.figure
    if deck_dimensions is not None:
        deck_width, deck_depth = deck_dimensions
        ax.add_patch(Rectangle(
            (0, 0), deck_width, deck_depth,
            facecolor="none", edgecolor="gray", hatch="///",
            linewidth=0, zorder=0,
        ))
        ax.plot(
            [0, deck_width, deck_width, 0, 0],
            [0, 0, deck_depth, deck_depth, 0],
            "-",
            color="black",
        )

    ax.plot(x, y, label="Trajectory", color="fuchsia")
    ax.plot(x[0], y[0], "go", label="Start")
    ax.plot(x[-1], y[-1], "rx", label="End")

    if deck_dimensions is not None:
        margin = 0.05 * max(deck_width, deck_depth)
        ax.set_xlim(-margin, deck_width + margin)
        ax.set_ylim(-margin, deck_depth + margin)

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True)
    ax.legend()

    if show:
        fig.tight_layout()
        plt.show()
    return fig, ax
