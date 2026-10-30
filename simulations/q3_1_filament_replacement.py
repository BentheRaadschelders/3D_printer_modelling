import argparse
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
import warnings

warnings.filterwarnings("ignore", message="Unable to import Axes3D")
import matplotlib.pyplot as plt
import numpy as np

from printer_model.hybrid_printer import (
    PHASE, PRINTED, ExternalSensorEvent, Mode, HybridParameters,
    rose_geometry, simulate_replacement,
)
from printer_model.nozzle import NozzleParameters
from printer_model.parameters import param
from simulations.report_figures import plot_replacement_details
from simulations.utils import plot_solution


def results_summary(run):
    printing = run.mode == Mode.PRINTING
    desired = np.array([rose_geometry(a, run.parameters.radius)[0]
                        for a in run.state[printing, PHASE]])
    errors = np.linalg.norm(run.state[printing][:, [0, 3]] - desired, axis=1)
    print_times = run.time[printing]
    return_start = next(event["time"] for event in run.events if event["to"] == "Returning")
    resumed = next(event["time"] for event in run.events if event["from"] == "Returning")
    transitions = [event for event in run.events if event["from"] != event["to"]]
    interruption = next(event for event in transitions if event["to"] == "Stopped")
    stopped = next(event for event in transitions if event["from"] == "Stopped")
    stopping_displacement = np.array(stopped["before"])[[0, 3]] - run.saved_position
    return {
        "parameters": asdict(run.parameters),
        "external_sensor_schedule": [asdict(event) for event in run.external_sensor_events],
        "motor_parameters": param,
        "settled_nozzle_temperature_K": run.parameters.nozzle.equilibrium[0],
        "saved_phase_rad": run.saved_phase,
        "saved_position_m": run.saved_position.tolist(),
        "stopping_duration_s": stopped["time"] - interruption["time"],
        "stopping_displacement_m": stopping_displacement.tolist(),
        "end_time_s": float(run.time[-1]),
        "total_printed_track_length_m": float(run.state[-1, PRINTED]),
        "maximum_printing_position_error_m": float(np.max(errors)),
        "return_duration_s": resumed - return_start,
        "resumed_printing_time_s": resumed,
        "maximum_error_after_first_restart_second_m": float(np.max(errors[print_times >= resumed + 1])),
        "final_head_velocity_m_s": run.state[-1, [1, 4]].tolist(),
        "final_phase_rad": float(run.state[-1, PHASE]),
        "maximum_physical_state_jump_at_FSM_switch": max(
            float(np.max(np.abs(np.array(event["before"][:9]) - event["after"][:9])))
            for event in transitions),
        "events": run.events,
        "parked_detector_samples": run.sensor_samples,
    }


def plot_motion(run, output_dir):
    """Save the stacked motion signals with the stationary wait omitted."""
    output_dir.mkdir(parents=True, exist_ok=True)
    wait_start = next(event["time"] for event in run.events if event["to"] == "Heat off")
    right_start = min(520.0, return_start_time(run))
    windows = (run.time <= wait_start, run.time >= right_start)

    with plt.rc_context({"font.size": 12, "axes.labelsize": 12,
                         "xtick.labelsize": 11, "ytick.labelsize": 11}):
        fig = plt.figure(figsize=(11, 6.8))
        grid = fig.add_gridspec(3, 2, width_ratios=[wait_start, run.time[-1] - right_start],
                                hspace=0.12, wspace=0.045)
        axes = np.empty((3, 2), dtype=object)
        for row in range(3):
            axes[row, 0] = fig.add_subplot(grid[row, 0],
                                           sharex=axes[0, 0] if row else None)
            axes[row, 1] = fig.add_subplot(grid[row, 1], sharey=axes[row, 0],
                                           sharex=axes[0, 1] if row else None)
        for col, mask in enumerate(windows):
            solution = SimpleNamespace(t=run.time[mask], y=run.state[mask, :6].T)
            plot_solution(solution, labels=["x", "y"],
                          ylabel=["Position [m]", "Velocity [m/s]", "Current [A]"],
                          axes=axes[:, col], show=False)
            for row, ax in enumerate(axes[:, col]):
                ax.set_xlabel("")
                ax.spines["right" if col == 0 else "left"].set_visible(False)
                if col:
                    ax.set_ylabel("")
                    ax.tick_params(axis="y", left=False, labelleft=False)
                if row < 2:
                    ax.tick_params(axis="x", labelbottom=False)
                if row or col:
                    ax.get_legend().remove()
                else:
                    ax.legend(loc="upper right", fontsize=10)
                xbreak = 1 if col == 0 else 0
                for ybreak in (0, 1):
                    ax.plot([xbreak - 0.012, xbreak + 0.012],
                            [ybreak - 0.025, ybreak + 0.025], transform=ax.transAxes,
                            color="black", linewidth=1, clip_on=False)
        axes[0, 0].set_xlim(0, wait_start)
        axes[0, 1].set_xlim(right_start, run.time[-1])
        fig.suptitle("Tool-head position, velocity and current", y=0.98)
        fig.text(0.5, 0.925,
                 f"Stationary thermal wait ({wait_start:.0f}–{right_start:.0f} s) omitted; times remain absolute",
                 ha="center", fontsize=10)
        fig.supxlabel("Time [s]", fontsize=12, y=0.015)
        fig.subplots_adjust(left=0.105, right=0.98, bottom=0.095, top=0.88)
        fig.savefig(output_dir / "q3_1_motion.png", dpi=180)


def plot_run(run, output_dir):
    plot_motion(run, output_dir)
    plot_replacement_details(run, output_dir)


def return_start_time(run):
    return next(event["time"] for event in run.events if event["to"] == "Returning")


def main():
    defaults = NozzleParameters()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--show", action="store_true", help="Also show the generated plots.")
    parser.add_argument("--thermal-capacity", type=float,
                        default=defaults.thermal_capacity, help="C_t,n [J/K].")
    parser.add_argument("--inductance", type=float, default=defaults.inductance, help="L_n [H].")
    parser.add_argument("--capacitance", type=float, default=defaults.capacitance, help="C_n [F].")
    parser.add_argument("--filament-finished-at", type=float, default=9.375,
                        help="Absolute time [s] of the external filament-finished sensor pulse.")
    parser.add_argument("--wipe-complete-at", type=float, default=17.0,
                        help="Absolute time [s] of the external wipe-complete sensor pulse.")
    parser.add_argument("--loaded-at", type=float, default=142.0,
                        help="Absolute time [s] of the external filament-loaded sensor pulse.")
    args = parser.parse_args()
    nozzle = NozzleParameters(thermal_capacity=args.thermal_capacity,
                              inductance=args.inductance, capacitance=args.capacitance)
    sensor_events = [ExternalSensorEvent(args.filament_finished_at, "filament_finished"),
                     ExternalSensorEvent(args.wipe_complete_at, "wipe_complete"),
                     ExternalSensorEvent(args.loaded_at, "loaded")]
    run = simulate_replacement(HybridParameters(nozzle=nozzle), sensor_events=sensor_events)
    output_dir = Path(__file__).resolve().parents[1] / "results"
    plot_run(run, output_dir)
    summary = results_summary(run)
    for event in run.events:
        print(f'{event["time"]:8.3f} s  {event["event"]}')
    print(f'Maximum printing error: {summary["maximum_printing_position_error_m"]:.3e} m')
    print(f"Figures saved in {output_dir}")
    if args.show:
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":
    main()
