# 3D Printer Modelling

SC42155 mini-project: physical and hybrid models of a CoreXY 3D printer.

The project covers motor kinematics and dynamics, prescribed drawing paths,
nozzle heating, cooling of deposited filament, filament replacement, and
square drawing using motor-encoder feedback.

## Contents

- [Getting started](#getting-started)
- [Project structure](#project-structure)
- [Simulations](#simulations)
- [Results](#results)
- [Tests](#tests)

## Getting started

### Requirements

- Python 3
- NumPy
- SciPy
- Matplotlib

### Installation

Open a terminal in the repository root. Optionally create and activate a
virtual environment on Linux or macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the dependencies:

```bash
python3 -m pip install numpy scipy matplotlib
```

Run the commands below from the repository root so that the project modules
can be imported.

## Project structure

```text
3D_printer_modelling/
├── printer_model/                 # Physical models and hybrid simulations
│   ├── parameters.py             # Motor and workspace parameters
│   ├── kinematics.py             # Motor-angle to Cartesian transformations
│   ├── dynamics.py               # Coupled and decoupled motor dynamics
│   ├── nozzle.py                 # Electrical and thermal nozzle model
│   ├── filament.py               # Growing-filament cooling model
│   ├── hybrid_printer.py         # Filament-replacement model and controller
│   └── square.py                 # Encoder-controlled square model
├── simulations/                  # Runnable simulations and plotting
│   ├── q1_4_draw_shapes.py
│   ├── q1_6_filament_cooling.py
│   ├── q3_1_filament_replacement.py
│   ├── q3_2_draw_square.py
│   ├── utils.py                  # Shared simulation and plotting functions
│   └── report_figures.py         # Compact panels for the report
├── tests/                        # Model and simulation checks
├── results/                      # Only the figures included in the LaTeX report
└── README.md
```

## Simulations

### Q1.4: drawing shapes

```bash
python3 -m simulations.q1_4_draw_shapes
```

This saves and displays position, velocity, current and XY trajectories for
an ellipse, an eight-petal rose and an optional TU Delft flame. The three
figures are saved in `results/` using the filenames listed below. The script uses the
decoupled motor model and inverse-dynamics voltages. Shape dimensions and
durations are set near the top of `simulations/q1_4_draw_shapes.py`.
Shared simulation and plotting functions are in `simulations/utils.py`.

### Q1.6: cooling of deposited filament

```bash
python3 -m simulations.q1_6_filament_cooling
```

This uses NumPy, SciPy and Matplotlib. It prints an ellipse in 20 seconds,
adds filament cells progressively at the nozzle temperature, connects the
ends when the loop is finished, and continues until the maximum temperature
spread is 1 K. The time parameter refers to this thermal example; the Q1.4
motor simulation is unchanged.

#### Files and outputs

- Model and thermal parameters: `printer_model/filament.py`.
- Generated figure: `results/q1_6_cooling.png`.
- Cooling times, mesh-refinement checks and contact sensitivity are printed
  in the terminal.

#### Options and nozzle operating point

For a faster run without mesh refinement and contact sensitivity, add
`--quick`. This regenerates the baseline figure and skips the extra checks.
`--cells` and `--print-time` change the mesh and duration.

The derivation keeps the settled nozzle temperature `T_n_bar` symbolic.
In code it is the configurable `FilamentParameters.deposition_temperature`
in kelvin. The dynamic nozzle state `T_n(t)` belongs to Q1.5/the full printer
model; this standalone cooling simulation assumes nozzle warm-up is complete.
The default numerical example selects a constant input of 14 V, with
ambient temperature 297 K, nozzle resistance 10 ohms and nozzle thermal
resistance 11 K/W. The resulting settled temperature is 512.6 K
(239.45 C). You can alternatively select the temperature directly:

```bash
python3 -m simulations.q1_6_filament_cooling --nozzle-temperature 220
```

The default voltage-driven operating point can also be specified explicitly,
using the Q1.5 equilibrium `T_n_bar = T_e + (R_t,n / R_n) * u_n_bar**2`:

```bash
python3 -m simulations.q1_6_filament_cooling --nozzle-voltage 14 --nozzle-resistance 10 --nozzle-thermal-resistance 11
```

The temperature option is in degrees Celsius; voltage is in volts,
electrical resistance in ohms, and thermal resistance in K/W. The example
defaults are 10 ohms and 11 K/W. The chosen voltage and calculated equilibrium
temperature are printed in the terminal. These options select
a preheated equilibrium; they do not simulate the warm-up transient.

#### Assumptions and numerical method

The default ABS properties are literature values, while filament width and
air/plate heat-transfer coefficients are illustrative assumptions. In
particular, plate contact has not been calibrated to a printer. The reported
cooling time depends on these choices.

The model exposes `internal_energy_density`, `distributed_heat_source`,
`cell_heat_balance` and `filament_dynamics` to follow the report's energy-balance
derivation directly. `cell_heat_balance` returns energies in joules, face
flows and external powers in watts, and temperature rates in kelvin per
second for the currently existing cells.

The model also calculates energy-balance diagnostics and a sufficient
forward-Euler step limit with nonnegative weights. These remain available
in the returned simulation data. The matrix-exponential solver does not
use this explicit step restriction.

For a fixed set of deposited cells, the shifted temperature satisfies
`d(T - T_inf)/dt = M @ (T - T_inf)`. The matrix is symmetric and
negative semidefinite, so its matrix exponential does not amplify the
shifted temperature's Euclidean norm for a nonnegative time interval.
Adding a new cell supplies energy separately; spatial and deposition
errors still need the mesh-refinement check.

### Q3.1: filament replacement during a rose

```bash
python3 -m simulations.q3_1_filament_replacement
```

This interrupts the eight-petal rose at its third centre crossing when
an external filament-finished sensor pulse arrives at 9.375 s. It saves
both position and phase, stops, homes vertically then horizontally, wipes,
cools, loads, reheats, returns and resumes the remaining petals. Encoder
samples feed the Q2.2 parked detector. The motor and nozzle ODEs are solved
together; the nozzle temperature controls the cool/warm transitions.

#### Files and outputs

- Hybrid model, guards, resets and detector: `printer_model/hybrid_printer.py`.
- Q1.5 nozzle ODE: `printer_model/nozzle.py`.
- Generated figures: `results/q3_1_motion.png`, `results/q3_1_xy_report.png`,
  `results/q3_1_timeline_report.png`, `results/q3_1_homing_report.png` and
  `results/q3_1_restart_report.png`.
- Event times and maximum printing error are printed in the terminal.
  Full state histories and event/sensor data remain in the returned run object.

#### Parameters and assumptions

The numerical example uses the chosen 14 V heater, 297 K ambient, 10 ohm
heater resistance and 11 K/W thermal resistance. The Q1.5 nozzle parameters
are 0.1 mH, 500 microfarads and 8 J/K.
Override them with `--inductance`, `--capacitance` and `--thermal-capacity`,
in H, F and J/K. This example also assumes ideal unsaturated voltage
control, inelastic end-stops, external wiping/loading completion signals and residual filament
for wiping. The thermal thresholds are modelling choices, not material
specifications. Initial nozzle warm-up and the deposited-filament PDE are
outside this replacement example.

#### Motion, homing and restart

The reference holds its phase throughout replacement. Returning uses a
position-and-velocity controller, with no prescribed return trajectory or
duration. Arrival depends on position and speed tolerances. Q2.2 homing
still uses downward then leftward voltage commands and encoder stall
detection. The existing Q2.2 detector state selects these commands
directly; there is no separate homing-stage variable. Boundary reactions
use physical position and velocity, with no stored contact flags.
Initial travel to the flower and initial warm-up are assumed
complete before the simulation starts.

Stopping applies zero voltage to both motors and waits for speed and
current to settle, without requiring the head to stay at the interrupted
printing position. Friction and back EMF brake the head over a finite
distance. The original position and phase remain saved for returning
and resuming. Wiping and thermal/loading waits also use zero motor
voltage; no active holding or external mechanical disturbance is modelled.
The printing tracking controller and return controller remain active in
their respective modes.

Resumed printing uses the original constant phase rate from Q1.4. The
motor states remain continuous, producing a brief tracking transient
after restarting from rest. The restart plot and results quantify this
error. Integration ends when the rose's final phase is reached; final
braking is outside this example. The back-EMF terms are retained.

#### External sensor inputs

Q3.1 has no elapsed-time or current-spool-budget state. `filament_finished`,
`wipe_complete` and `loaded` are discrete
sensor inputs supplied to `simulate_replacement(..., sensor_events=...)`,
separately from the physical parameters. The example launcher supplies
these pulses at absolute simulation times 9.375 s, 17 s and 142 s. Change them
with `--filament-finished-at`, `--wipe-complete-at` and `--loaded-at`; these values are an illustrative
input sequence, not internal wiping/loading timers. A pulse is accepted
only in the matching mode. If no future matching pulse is supplied, the
simulation reports that it is waiting rather than completing that action
automatically. The example schedule must allow the physical cooling guard
to be reached before the `loaded` pulse arrives. This replacement example
requires a filament-finished pulse before the first rose traversal ends.
Total printed path length remains a diagnostic; it does not trigger runout.

Add
`--show` to display the plots as well as saving them.

### Q3.2: square with motor encoders

```bash
python3 -m simulations.q3_2_draw_square
```

The head draws a 0.20 m square, starting at (0.05, 0.05) m. A position-error
and velocity-feedback controller drives directly towards each corner.
Encoder position and speed guards select the next corner only after
arrival and rest. There is no prescribed side duration, timed trajectory,
inverse-dynamics input or clock state. All nine motor/nozzle states stay
continuous when the target changes. The chosen gains are 40 V/m and
4 V s/m; the default square takes about 14.27 s to complete.
Motion feedback uses motor angles and angular velocities, with no XY
position sensors or current measurements. Encoder offsets are assumed
calibrated by prior homing. The nozzle remains preheated at the Q3.1
operating point, and sufficient filament is assumed for the complete path.

#### Files and outputs

- Model and motion controller: `printer_model/square.py`.
- Generated figures: `results/q3_2_motion_report.png`,
  `results/q3_2_xy_report.png` and `results/q3_2_controls_report.png`.
- Corner events and numerical errors are printed in the terminal.
  Full state histories and corner events remain in the returned run object.

#### Options

Add `--show` to display the plots; `--side-length` selects the size in
metres. The motion duration follows from the dynamics and arrival guards.

## Results

`results/` contains only the 14 figures included in the LaTeX report.
Simulation commands save directly into this folder. Numerical summaries
are printed in the terminal, and the physical model data remain available
in memory. The local report sources and working files are excluded from
version control.

| Exercise | Figure files in `results/` |
| --- | --- |
| Q1.4 | `Figure_ellipse1.png`, `Figure_flower1.png`, `Figure_flame1.png` |
| Q1.6 | `q1_6_cooling.png` |
| Q2.1 | `FSM.png` |
| Q2.2 | `automaton.png` |
| Q3.1 | `q3_1_motion.png`, `q3_1_xy_report.png`, `q3_1_timeline_report.png`, `q3_1_homing_report.png`, `q3_1_restart_report.png` |
| Q3.2 | `q3_2_motion_report.png`, `q3_2_xy_report.png`, `q3_2_controls_report.png` |

`FSM.png` and `automaton.png` are manually created diagrams included
alongside the simulation figures. The simulation commands regenerate
the other 12 images.

The Q3 launchers use `simulations/report_figures.py` for the compact panels
with larger labels for side-by-side layouts. Q3.1 keeps the full-width
motion plot with its 17–520 s axis break. Q3.2 has a full-width stacked
motion plot and separate XY and voltage/mode panels. Extra diagnostic
plot layouts are not saved.

## Tests

Run all discovered tests from the repository root:

```bash
python3 -m unittest discover -s tests -v
```

Individual exercise checks can also be run separately.

### Q1.6: analytical and conservation checks

```bash
python3 -m unittest discover -s tests -p test_filament.py -v
```

### Q3.1: hybrid model and sensor events

```bash
python3 -m unittest discover -s tests -p test_hybrid_printer.py -v
```

### Q3.2: square motion and encoder guards

```bash
python3 -m unittest discover -s tests -p test_square.py -v
```

### Existing Q1.4 test issues

The ellipse, rose and flame test modules currently import initial-state
helpers that are not defined in `simulations/q1_4_draw_shapes.py`:
`ellipse_initial_state`, `rose_initial_state` and `flame_initial_state`.
Full test discovery therefore reports three import errors. The simulation
script uses `initial_state_from_motion` to calculate its initial states.
