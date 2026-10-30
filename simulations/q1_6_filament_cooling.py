import argparse
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from printer_model.filament import (
    FilamentParameters, settled_nozzle_temperature, settled_nozzle_voltage,
    simulate_filament,
)


def metrics(result):
    final = result["final_temperature"]
    closure = result["closure_temperature"]
    return {
        "n_cells": int(final.size),
        "length_m": result["length"],
        "cell_length_m": result["cell_length"],
        "print_time_s": result["print_time"],
        "additional_cooling_time_s": result["wait_time"],
        "uniform_time_s": result["uniform_time"],
        "closure_min_C": float(closure.min() - 273.15),
        "closure_max_C": float(closure.max() - 273.15),
        "final_min_C": float(final.min() - 273.15),
        "final_max_C": float(final.max() - 273.15),
        "final_mean_C": float(final.mean() - 273.15),
        "final_range_K": float(np.ptp(final)),
        "equilibrium_C": result["equilibrium_temperature"] - 273.15,
        "nozzle_equilibrium_C": result["nozzle_equilibrium_temperature"] - 273.15,
        "numerics": result["numerics"],
        "energy_diagnostics": result["energy_diagnostics"],
    }


def plot_results(result, output_dir):
    import warnings
    warnings.filterwarnings("ignore", message="Unable to import Axes3D")
    import matplotlib.pyplot as plt

    time = result["time"]
    temperature = result["temperature"] - 273.15
    end = result["uniform_time"]
    printing_end = result["print_time"]
    minimum = np.nanmin(temperature, axis=1)
    maximum = np.nanmax(temperature, axis=1)
    mean = np.nanmean(temperature, axis=1)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), layout="constrained")
    # Irregular deposition events: explicitly supply pixel edges.
    time_edges = np.r_[time[0], (time[:-1] + time[1:]) / 2, time[-1]]
    arc_edges = np.linspace(0, result["length"] * 1000, temperature.shape[1] + 1)
    # The plot must follow the chosen operating point rather than assume 240 C.
    lower_temperature = min(float(np.nanmin(temperature)),
                            result["equilibrium_temperature"] - 273.15)
    upper_temperature = max(float(np.nanmax(temperature)),
                            result["equilibrium_temperature"] - 273.15)
    if upper_temperature == lower_temperature:
        upper_temperature += 1.0
    field = axes[0, 0].pcolormesh(
        time_edges, arc_edges, np.ma.masked_invalid(temperature.T),
        cmap="inferno", vmin=lower_temperature, vmax=upper_temperature,
        shading="flat", rasterized=True)
    axes[0, 0].set_xlabel("Time [s]")
    axes[0, 0].set_ylabel("Arc length s [mm]")
    axes[0, 0].set_title("Growing filament: grey means not yet printed")
    axes[0, 0].set_facecolor("#eeeeee")
    fig.colorbar(field, ax=axes[0, 0], label="Temperature [°C]")

    axes[0, 1].plot(time, maximum, label="Maximum", color="#c2410c")
    axes[0, 1].plot(time, mean, label="Mean", color="#7c3aed")
    axes[0, 1].plot(time, minimum, label="Minimum", color="#0369a1")
    axes[0, 1].axhline(result["equilibrium_temperature"] - 273.15,
                       linestyle=":", color="black", label="Equilibrium")
    axes[0, 1].set_xlabel("Time [s]")
    axes[0, 1].set_ylabel("Temperature [°C]")
    axes[0, 1].set_title("Temperature statistics of deposited cells")
    axes[0, 1].legend(fontsize=8)

    closed = time >= printing_end
    axes[1, 0].semilogy(time[closed], (maximum - minimum)[closed], color="#0369a1")
    axes[1, 0].axhline(result["tolerance"], color="black", linestyle="--",
                       label="1 K uniformity criterion")
    axes[1, 0].set_xlabel("Time since printing started [s]")
    axes[1, 0].set_ylabel("Maximum − minimum [K]")
    axes[1, 0].set_title(f"Additional cooling time: {result['wait_time']:.2f} s")
    axes[1, 0].legend(fontsize=8)
    for sample_time in (printing_end, printing_end + 10, end):
        index = int(np.argmin(abs(time - sample_time)))
        axes[1, 1].plot(result["centres"] * 1000, temperature[index],
                        label=f"t = {time[index]:.1f} s")
    axes[1, 1].set_xlabel("Arc length s [mm]")
    axes[1, 1].set_ylabel("Temperature [°C]")
    axes[1, 1].set_title("Temperature around the completed loop")
    axes[1, 1].legend(fontsize=8)
    for ax in axes.flat:
        ax.grid(alpha=0.2)
    for ax in axes[0]:
        ax.axvline(printing_end, color="#0284c7", linestyle="--", linewidth=1)
    fig.savefig(output_dir / "q1_6_cooling.png", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cells", type=int, default=400)
    parser.add_argument("--print-time", type=float, default=20.0)
    parser.add_argument("--quick", action="store_true")
    nozzle_input = parser.add_mutually_exclusive_group()
    nozzle_input.add_argument(
        "--nozzle-temperature", type=float, metavar="CELSIUS",
        help="Alternatively select the settled nozzle temperature directly")
    nozzle_input.add_argument(
        "--nozzle-voltage", type=float, metavar="VOLTS",
        help="Constant nozzle voltage; default operating point: 14 V")
    parser.add_argument("--nozzle-resistance", type=float, default=10.0, metavar="OHMS")
    parser.add_argument("--nozzle-thermal-resistance", type=float, default=11.0, metavar="K_PER_W")
    args = parser.parse_args()
    settings = FilamentParameters()
    try:
        if args.nozzle_temperature is None:
            # The default operating point is chosen by its electrical input.
            nozzle_voltage = 14.0 if args.nozzle_voltage is None else args.nozzle_voltage
            nozzle_temperature = settled_nozzle_temperature(
                nozzle_voltage, settings.ambient_temperature,
                args.nozzle_resistance, args.nozzle_thermal_resistance)
            operating_point_source = "constant nozzle voltage"
        else:
            nozzle_temperature = args.nozzle_temperature + 273.15
            nozzle_voltage = settled_nozzle_voltage(
                nozzle_temperature, settings.ambient_temperature,
                args.nozzle_resistance, args.nozzle_thermal_resistance)
            operating_point_source = "selected settled nozzle temperature"
        settings = replace(settings, deposition_temperature=nozzle_temperature)
    except ValueError as error:
        parser.error(str(error))
    result = simulate_filament(settings, n_cells=args.cells, print_time=args.print_time)
    output_dir = Path(__file__).resolve().parents[1] / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_results(result, output_dir)
    summary = {
        "parameters": asdict(settings),
        "nozzle_operating_point": {
            "source": operating_point_source,
            "settled_temperature_K": nozzle_temperature,
            "settled_temperature_C": nozzle_temperature - 273.15,
            "constant_voltage_V": nozzle_voltage,
            "electrical_resistance_ohm": args.nozzle_resistance,
            "thermal_resistance_K_W": args.nozzle_thermal_resistance,
            "warm_up": "assumed completed before filament printing starts",
        },
        "baseline": metrics(result),
    }
    if not args.quick:
        summary["mesh_refinement"] = []
        for count in (100, 200, 400, 800):
            refined = result if count == args.cells else simulate_filament(
                settings, n_cells=count, print_time=args.print_time,
                print_samples=2, cooling_samples=2)
            summary["mesh_refinement"].append(metrics(refined))
        summary["bed_contact_sensitivity"] = []
        for coefficient in (25.0, 50.0, 100.0):
            changed = result if coefficient == settings.bed_coefficient else simulate_filament(
                replace(settings, bed_coefficient=coefficient),
                n_cells=args.cells, print_time=args.print_time,
                print_samples=2, cooling_samples=2)
            summary["bed_contact_sensitivity"].append({
                "bed_coefficient_W_m2K": coefficient, **metrics(changed)})
    print(f"Nozzle input: {nozzle_voltage:.2f} V")
    print(f"Settled nozzle temperature: {nozzle_temperature:.2f} K")
    print(f"Additional cooling time: {result['wait_time']:.2f} s")
    if not args.quick:
        print("Mesh refinement (cells, additional cooling time [s]):")
        for entry in summary["mesh_refinement"]:
            print(f"  {entry['n_cells']}: {entry['additional_cooling_time_s']:.4f}")
        print("Bed-contact sensitivity (coefficient [W/(m^2 K)], cooling time [s]):")
        for entry in summary["bed_contact_sensitivity"]:
            print(f"  {entry['bed_coefficient_W_m2K']:.0f}: "
                  f"{entry['additional_cooling_time_s']:.4f}")
    print(f"Figure saved in {output_dir}")


if __name__ == "__main__":
    main()
