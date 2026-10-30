import argparse
from dataclasses import asdict
from pathlib import Path
import warnings

warnings.filterwarnings("ignore", message="Unable to import Axes3D")
import matplotlib.pyplot as plt
import numpy as np

from printer_model.parameters import param
from printer_model.square import SquareParameters, simulate_square
from simulations.report_figures import plot_square_panels


def results_summary(run):
    position = run.position
    corners = np.array([event["position_m"] for event in run.events])
    cross_track_errors = []
    distances_to_sides = []
    side_lengths = []
    for index in range(4):
        points = position[run.mode == index]
        fixed_axis = 1 if index % 2 == 0 else 0
        cross_track_errors.extend(
            abs(points[:, fixed_axis] - run.parameters.vertices[index, fixed_axis]))
        side_lengths.append(float(np.linalg.norm(points[-1] - points[0])))
        start, end = run.parameters.vertices[index:index + 2]
        delta = end - start
        fraction = np.clip((position - start) @ delta / (delta @ delta), 0, 1)
        distances_to_sides.append(np.linalg.norm(
            position - start - fraction[:, None] * delta, axis=1))
    return {
        "parameters": asdict(run.parameters),
        "motor_parameters": param,
        "sensors_used_for_motion": ["theta_l", "theta_r", "omega_l", "omega_r"],
        "motion_completion_time_s": run.events[-1]["time"],
        "maximum_cross_track_error_m": float(max(cross_track_errors)),
        "maximum_deviation_from_square_m": float(np.min(distances_to_sides, axis=0).max()),
        "maximum_corner_error_m": float(np.linalg.norm(corners - run.parameters.vertices[1:], axis=1).max()),
        "closure_error_m": float(np.linalg.norm(corners[-1] - position[0])),
        "side_lengths_m": side_lengths,
        "side_durations_s": np.diff([0.0, *[event["time"] for event in run.events]]).tolist(),
        "maximum_motor_voltage_V": float(np.abs(run.voltage).max()),
        "maximum_corner_angular_speed_rad_s": float(max(
            max(abs(v) for v in event["angular_velocity_rad_s"]) for event in run.events)),
        "events": run.events,
    }


def plot_square(run, output_dir):
    """Save the motion, trajectory and voltage/mode panels used in the report."""
    plot_square_panels(run, output_dir)


def main():
    defaults = SquareParameters()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--side-length", type=float,
                        default=defaults.side_length, help="Square side [m].")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    run = simulate_square(SquareParameters(side_length=args.side_length))
    output_dir = Path(__file__).resolve().parents[1] / "results"
    plot_square(run, output_dir)
    summary = results_summary(run)
    for event in run.events:
        print(f'{event["time"]:7.3f} s  {event["from"]} -> {event["to"]}')
    print(f'Maximum cross-track error: {summary["maximum_cross_track_error_m"]:.3e} m')
    print(f'Closure error: {summary["closure_error_m"]:.3e} m')
    print(f"Figures saved in {output_dir}")
    if args.show:
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":
    main()
