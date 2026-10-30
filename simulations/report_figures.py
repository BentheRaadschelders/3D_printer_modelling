from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np

from printer_model.hybrid_printer import MODE_LABELS, Mode, PHASE, rose_geometry
from printer_model.parameters import param
from printer_model.square import SQUARE_LABELS
from simulations.utils import plot_solution, plot_trajectory


STYLE = {"font.size": 12, "axes.labelsize": 12, "axes.titlesize": 13,
         "xtick.labelsize": 11, "ytick.labelsize": 11, "legend.fontsize": 10}


def _finish(fig, path):
    fig.tight_layout(pad=0.7)
    fig.savefig(path, dpi=200)


def plot_replacement_details(run, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    resumed = next(event["time"] for event in run.events if event["from"] == "Returning")
    interrupted = next(event["time"] for event in run.events if event["to"] == "Stopped")
    parked = next(event["time"] for event in run.events if event["to"] == "Wiping")
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(5.2, 4.8))
        plot_trajectory(SimpleNamespace(y=run.state[:, :6].T), title="Rose trajectory",
                        deck_dimensions=(param["dx"], param["dy"]), ax=ax, show=False)
        for line in ax.lines:
            if line.get_label() == "Trajectory":
                line.set_visible(False)
            if line.get_label() in ("Trajectory", "Start", "End"):
                line.set_label("_nolegend_")
        for mask, style, label in (
            ((run.mode == Mode.PRINTING) & (run.time < resumed), "-", "Spool 1"),
            ((run.mode == Mode.PRINTING) & (run.time >= resumed), "-", "Spool 2"),
            (run.mode == Mode.STOPPED, "--", "Braking"),
            (run.mode == Mode.MOVING_TO_ORIGIN, "--", "Homing"),
            (run.mode == Mode.RETURNING, ":", "Return"),
        ):
            ax.plot(run.state[mask, 0], run.state[mask, 3], style, label=label, linewidth=1.6)
        ax.scatter(*run.saved_position, marker="x", color="black", s=30, zorder=5)
        ax.scatter(0, 0, marker="s", color="purple", s=20, zorder=5)
        ax.set_title("")
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=3,
                  fontsize=10, columnspacing=0.6, handlelength=1.2)
        _finish(fig, output_dir / "q3_1_xy_report.png")

        fig, axes = plt.subplots(2, 1, figsize=(5.2, 4.8), sharex=True)
        mask = (run.time >= interrupted - 0.15) & (run.time <= parked + 0.1)
        time, state = run.time[mask], run.state[mask]
        axes[0].plot(time, (state[:, 1] + state[:, 4]) / param["r_m"], label="Left")
        axes[0].plot(time, (state[:, 1] - state[:, 4]) / param["r_m"], "--", label="Right")
        axes[0].set_title("Stopping and homing")
        axes[0].set_ylabel("Motor speed [rad/s]")
        for index, name in enumerate(("Left", "Right")):
            axes[1].plot(time, run.motor_voltage[mask, index],
                         "-" if index == 0 else "--", label=name)
        axes[1].set_xlabel("Time [s]")
        axes[1].set_ylabel("Motor voltage [V]")
        for ax in axes:
            ax.grid(alpha=0.3)
            ax.legend(loc="best")
            for event in run.events:
                if "impact" in event["event"]:
                    ax.axvline(event["time"], color="0.6", linewidth=0.8)
        _finish(fig, output_dir / "q3_1_homing_report.png")

        fig, axes = plt.subplots(2, 1, figsize=(5.2, 4.8), sharex=True)
        mask = (run.mode == Mode.PRINTING) & (run.time >= resumed) & (run.time <= resumed + 1)
        time = run.time[mask] - resumed
        actual = run.state[mask][:, [0, 3]]
        desired = np.array([rose_geometry(state[PHASE], run.parameters.radius)[0]
                           for state in run.state[mask]])
        for index, label in enumerate(("x", "y")):
            line, = axes[0].plot(time, actual[:, index] * 1000, label=label)
            axes[0].plot(time, desired[:, index] * 1000, "--", color=line.get_color())
        axes[0].set_title("Restart from rest")
        axes[0].set_ylabel("Position [mm]")
        axes[0].legend()
        axes[1].plot(time, np.linalg.norm(actual - desired, axis=1) * 1000, color="tab:red")
        axes[1].set_xlabel("Time since restart [s]")
        axes[1].set_ylabel("Position error [mm]")
        for ax in axes:
            ax.grid(alpha=0.3)
        _finish(fig, output_dir / "q3_1_restart_report.png")

        fig, axes = plt.subplots(4, 1, figsize=(10, 6.6), sharex=True,
                                 gridspec_kw={"height_ratios": [1.8, 1.1, 0.9, 0.9]})
        axes[0].step(run.time, run.mode, where="post")
        axes[0].set_yticks(range(8), MODE_LABELS, fontsize=10)
        axes[0].set_ylabel("Mode")
        axes[1].plot(run.time, run.state[:, 6] - 273.15, color="tab:red")
        for threshold, label, color in ((run.parameters.cold_temperature, "Load", "tab:blue"),
                                        (run.parameters.warm_temperature, "Resume", "tab:orange")):
            axes[1].axhline(threshold - 273.15, linestyle="--", color=color, label=label)
        axes[1].set_ylabel("Nozzle [°C]")
        axes[1].legend(loc="center right", ncol=2)
        axes[2].step(run.time, run.heater_voltage, where="post",
                     color="tab:red", label="Heater [V]")
        axes[2].step(run.time, run.extrude * 5, where="post",
                     color="tab:green", label="Extrusion (×5)")
        axes[2].set_ylabel("Commands")
        axes[2].legend(loc="center right", ncol=2)
        axes[3].plot(run.time, run.state[:, PHASE] / np.pi, color="tab:purple")
        axes[3].set_xlabel("Time [s]")
        axes[3].set_ylabel(r"Phase $\varphi/\pi$")
        for ax in axes:
            ax.grid(alpha=0.25)
            for event in run.events:
                if event["from"] != event["to"]:
                    ax.axvline(event["time"], color="0.75", linewidth=0.6, zorder=0)
        _finish(fig, output_dir / "q3_1_timeline_report.png")


def plot_square_panels(run, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    position, velocity = run.position, run.velocity
    values = np.array([position[:, 0], velocity[:, 0], run.state[:, 4] + run.state[:, 5],
                       position[:, 1], velocity[:, 1], run.state[:, 4] - run.state[:, 5]])
    solution = SimpleNamespace(t=run.time, y=values)
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(3, 1, figsize=(10, 5.8), sharex=True)
        plot_solution(solution, labels=["x", "y"],
                      ylabel=["Position [m]", "Velocity [m/s]", "Current [A]"],
                      axes=axes, show=False)
        axes[0].set_title("Square: position, velocity and current")
        for ax in axes[:-1]:
            ax.get_legend().remove()
        _finish(fig, output_dir / "q3_2_motion_report.png")

        fig, ax = plt.subplots(figsize=(5.2, 4.8))
        plot_trajectory(solution, title="Square trajectory",
                        deck_dimensions=(param["dx"], param["dy"]), ax=ax, show=False)
        vertices = run.parameters.vertices
        for start, end in zip(vertices[:-1], vertices[1:]):
            midpoint, delta = (start + end) / 2, end - start
            ax.annotate("", xy=midpoint + 0.09 * delta, xytext=midpoint - 0.09 * delta,
                        arrowprops={"arrowstyle": "->", "color": "black", "lw": 1.3})
        ax.scatter(*vertices[1:-1].T, color="black", s=12, zorder=5)
        ax.legend(loc="upper left", fontsize=10)
        _finish(fig, output_dir / "q3_2_xy_report.png")

        fig, axes = plt.subplots(2, 1, figsize=(5.2, 4.8), sharex=True)
        for motor, label in enumerate(("Left", "Right")):
            axes[0].plot(run.time, run.voltage[:, motor], "-" if motor == 0 else "--", label=label)
        axes[0].set_title("Motor inputs and modes")
        axes[0].set_ylabel("Motor voltage [V]")
        axes[0].legend()
        axes[1].step(run.time, run.mode, where="post", color="tab:purple")
        axes[1].set_yticks(range(5), SQUARE_LABELS)
        axes[1].set_xlabel("Time [s]")
        axes[1].set_ylabel("Mode")
        for ax in axes:
            ax.grid(alpha=0.3)
            for event in run.events:
                ax.axvline(event["time"], color="0.65", linewidth=0.8)
        _finish(fig, output_dir / "q3_2_controls_report.png")
