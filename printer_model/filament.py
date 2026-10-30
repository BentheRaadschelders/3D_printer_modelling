from dataclasses import dataclass

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import brentq
from scipy.sparse import diags, lil_matrix
from scipy.sparse.linalg import expm_multiply


@dataclass(frozen=True)
class FilamentParameters:
    """Numerical example parameters; deposition_temperature represents T_n_bar."""
    density: float = 1040.0                 # kg/m^3
    specific_heat: float = 1670.0           # J/(kg K)
    conductivity: float = 0.173             # W/(m K)
    width: float = 0.0004                   # m, assumed square cross-section
    air_coefficient: float = 10.0           # W/(m^2 K), assumed weak convection
    bed_coefficient: float = 50.0           # W/(m^2 K), uncalibrated contact
    ambient_temperature: float = 297.0      # K, selected simulation ambient
    bed_temperature: float = 378.15         # K = 105 deg C
    deposition_temperature: float = 512.6    # K, from 14 V with R_n=10 ohm, R_t,n=11 K/W

    def __post_init__(self):
        for name in ("density", "specific_heat", "width"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive and finite")
        for name in ("conductivity", "air_coefficient", "bed_coefficient"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be nonnegative and finite")
        for name in ("ambient_temperature", "bed_temperature", "deposition_temperature"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive in kelvin")

    @property
    def diffusivity(self):
        return self.conductivity / (self.density * self.specific_heat)

    @property
    def area(self):
        return self.width**2

    @property
    def volumetric_heat_capacity(self):
        return self.density * self.specific_heat

    @property
    def air_rate(self):
        return 3 * self.air_coefficient / (self.density * self.specific_heat * self.width)

    @property
    def bed_rate(self):
        return self.bed_coefficient / (self.density * self.specific_heat * self.width)

    @property
    def decay_rate(self):
        return self.air_rate + self.bed_rate

    @property
    def equilibrium_temperature(self):
        if self.decay_rate == 0:
            # Arbitrary shift when there is no heat exchange.
            return 0.0
        return (self.air_rate * self.ambient_temperature
                + self.bed_rate * self.bed_temperature) / self.decay_rate


def _validate_nozzle_parameters(ambient_temperature, electrical_resistance, thermal_resistance):
    values = (ambient_temperature, electrical_resistance, thermal_resistance)
    if not np.all(np.isfinite(values)) or min(values) <= 0:
        raise ValueError("Need positive finite ambient temperature [K], R_n [ohm] and R_t,n [K/W]")


def settled_nozzle_temperature(voltage, ambient_temperature,
                               electrical_resistance, thermal_resistance):
    """Q1.5 equilibrium T_n_bar = T_e + (R_t,n / R_n) * u_n_bar**2 [K].

    Assumes a settled electrical circuit, constant resistances and the Q1.5
    lumped thermal-loss model. This does not simulate nozzle warm-up or replace
    the dynamic state T_n(t) needed for a later extrusion guard.
    """
    _validate_nozzle_parameters(ambient_temperature, electrical_resistance, thermal_resistance)
    if not np.isfinite(voltage):
        raise ValueError("Nozzle voltage must be finite")
    return ambient_temperature + thermal_resistance / electrical_resistance * voltage**2


def settled_nozzle_voltage(temperature, ambient_temperature,
                           electrical_resistance, thermal_resistance):
    """Nonnegative steady voltage for a selected nozzle temperature in kelvin."""
    _validate_nozzle_parameters(ambient_temperature, electrical_resistance, thermal_resistance)
    if not np.isfinite(temperature) or temperature < ambient_temperature:
        raise ValueError("A resistive heater needs a target temperature at or above ambient")
    return float(np.sqrt(electrical_resistance / thermal_resistance
                         * (temperature - ambient_temperature)))


def cell_heat_capacity(cell_length, settings):
    """C_f = rho c_p A ds [J/K] of one existing finite volume."""
    if not np.isfinite(cell_length) or cell_length <= 0:
        raise ValueError("cell_length must be positive and finite")
    return settings.volumetric_heat_capacity * settings.area * cell_length


def internal_energy_density(temperature, settings):
    """e_th = rho c_p (T - T_e) [J/m^3], relative to ambient."""
    return settings.volumetric_heat_capacity * (
        np.asarray(temperature, dtype=float) - settings.ambient_temperature)


def distributed_heat_source(temperature, settings):
    """q_th [W/m^3]: heat from three air faces and the bed-contact face.

    These are volumetric sources after reduction to a 1D model. Deposited
    material is stationary, so no material-advection flux is included.
    """
    temperature = np.asarray(temperature, dtype=float)
    return (
        3 * settings.air_coefficient / settings.width
        * (settings.ambient_temperature - temperature)
        + settings.bed_coefficient / settings.width
        * (settings.bed_temperature - temperature)
    )


def cell_heat_balance(temperature, cell_length, settings, closed=False):
    """Cell energies [J], face heat flows [W], sources [W], and temperature rates.

    The n+1 face flows are positive towards increasing arc length. Before
    closure the two end flows are zero. After closure both entries for the
    seam carry the same flow, making the global conductive power sum zero.
    Only existing cells may be passed; unprinted cells have no energy state.
    """
    temperature = np.asarray(temperature, dtype=float)
    if temperature.ndim != 1 or not temperature.size or not np.all(np.isfinite(temperature)):
        raise ValueError("temperature must be a nonempty finite 1D array of active cells")
    capacity = cell_heat_capacity(cell_length, settings)
    volume = settings.area * cell_length
    conductance = settings.conductivity * settings.area / cell_length
    face_flow = np.zeros(temperature.size + 1)
    face_flow[1:-1] = -conductance * np.diff(temperature)
    if closed:
        face_flow[0] = face_flow[-1] = conductance * (temperature[-1] - temperature[0])
    source_density = distributed_heat_source(temperature, settings)
    source_power = volume * source_density
    net_power = face_flow[:-1] - face_flow[1:] + source_power
    return {
        "energy_J": volume * internal_energy_density(temperature, settings),
        "face_flow_W": face_flow,
        "source_density_W_m3": source_density,
        "source_power_W": source_power,
        "net_power_W": net_power,
        "temperature_rate_K_s": net_power / capacity,
    }


def filament_dynamics(time, temperature, cell_length, settings, closed=False):
    """Method-of-lines ODE: dT_i/dt = (W_left - W_right + Q_i) / C_f.

    The time argument allows use with an ODE solver. The standalone simulation
    instead evaluates the equivalent constant-coefficient matrix exponential
    between deposition events. It never steps over a material-addition event.
    """
    return cell_heat_balance(temperature, cell_length, settings, closed)["temperature_rate_K_s"]


def thermal_matrix(n_cells, cell_length, settings, closed=False):
    """Finite-volume generator, with insulated ends or a periodic seam.

    Each edge transfers equal and opposite heat to its two cells. End cells
    have one neighbour until closure; all cells have two afterwards.
    """
    if n_cells < 1 or cell_length <= 0:
        raise ValueError("Need at least one cell and a positive cell length")
    capacity = cell_heat_capacity(cell_length, settings)
    axial_conductance = settings.conductivity * settings.area / cell_length
    air_conductance = 3 * settings.air_coefficient * settings.width * cell_length
    bed_conductance = settings.bed_coefficient * settings.width * cell_length
    # Each conductance is W/K; division by cell capacity gives a rate in 1/s.
    rate = axial_conductance / capacity
    matrix = lil_matrix((n_cells, n_cells), dtype=float)
    for i in range(n_cells - 1):
        matrix[i, i] -= rate
        matrix[i + 1, i + 1] -= rate
        matrix[i, i + 1] += rate
        matrix[i + 1, i] += rate
    if closed and n_cells > 1:
        # For two cells this adds the second shared face of a ring.
        matrix[0, 0] -= rate
        matrix[-1, -1] -= rate
        matrix[0, -1] += rate
        matrix[-1, 0] += rate
    exchange_rate = (air_conductance + bed_conductance) / capacity
    return matrix.tocsr() - diags(np.full(n_cells, exchange_rate))


def stability_diagnostics(matrix):
    """Bounds for the fixed active-cell system; these do not restrict expm.

    M is symmetric with nonnegative off-diagonal entries and row sum -gamma.
    Gershgorin bounds therefore enclose its real, nonpositive eigenvalues.
    dt <= 1/max(-M_ii) makes all forward-Euler temperature weights nonnegative;
    for an interior cell this is 1/(2*kappa'/ds^2 + beta_a + beta_b).
    It is a sufficient monotonicity/stability bound, not the exact Euler limit.
    """
    diagonal = matrix.diagonal()
    off_diagonal_sum = np.asarray(abs(matrix).sum(axis=1)).ravel() - abs(diagonal)
    maximum_loss = float(np.max(-diagonal))
    return {
        "forward_euler_nonnegative_weight_limit_s": (
            1.0 / maximum_loss if maximum_loss > 0 else float("inf")),
        "eigenvalue_lower_bound_s_inv": float(np.min(diagonal - off_diagonal_sum)),
        "eigenvalue_upper_bound_s_inv": float(np.max(diagonal + off_diagonal_sum)),
        "time_integrator": "matrix exponential between deposition events",
        "explicit_step_limit_applies": False,
    }


def advance_temperature(temperature, duration, matrix, equilibrium):
    """Advance active-cell temperatures [K] using the matrix exponential."""
    if duration < 0:
        raise ValueError("duration must be nonnegative")
    if duration == 0:
        return np.asarray(temperature, dtype=float).copy()
    scaled = duration * matrix
    return equilibrium + expm_multiply(
        scaled, np.asarray(temperature) - equilibrium,
        traceA=float(scaled.diagonal().sum()),
    )


def ellipse_mesh(n_cells, print_time, a=0.10, b=0.06):
    """Equal arc-length cells, with constant angular rate and a left start."""
    if n_cells < 3 or int(n_cells) != n_cells or print_time <= 0 or min(a, b) <= 0:
        raise ValueError("Need an integer n_cells >= 3 and positive geometry/timing")
    phase = np.linspace(0.0, 2 * np.pi, 40001)
    arc = cumulative_trapezoid(
        np.sqrt(a**2 * np.sin(phase)**2 + b**2 * np.cos(phase)**2),
        phase, initial=0.0,
    )
    length = float(arc[-1])
    ds = length / n_cells
    left_edges = np.arange(n_cells) * ds
    centres = left_edges + ds / 2
    birth_times = np.interp(left_edges, arc, phase) * print_time / (2 * np.pi)
    centre_phase = np.pi + np.interp(centres, arc, phase)
    return {
        "length": length, "cell_length": ds, "centres": centres,
        "birth_times": birth_times,
        "x": 0.15 + a * np.cos(centre_phase),
        "y": 0.15 + b * np.sin(centre_phase),
    }


def simulate_filament(settings=None, n_cells=400, print_time=20.0,
                      tolerance=1.0, a=0.10, b=0.06,
                      print_samples=201, cooling_samples=241,
                      maximum_cooling_time=600.0):
    """Print an ellipse, close the seam, then stop at max(T)-min(T) <= tolerance.

    New cells are added at the start of their deposition interval, each with
    exactly one cell's thermal energy at the supplied settled nozzle
    temperature T_n_bar (settings.deposition_temperature). Nozzle
    heating is not imposed on previously deposited material. The leading-edge
    activation has a length error of at most one cell, vanishing as ds -> 0.
    """
    settings = settings or FilamentParameters()
    if tolerance <= 0 or min(print_samples, cooling_samples) < 2:
        raise ValueError("Need positive tolerance and at least two output samples")
    mesh = ellipse_mesh(n_cells, print_time, a, b)
    births = mesh["birth_times"]
    equilibrium = settings.equilibrium_temperature
    events = np.unique(np.r_[births, np.linspace(0, print_time, print_samples)])
    temperature = np.empty(0)
    times, history = [], []
    previous_time = 0.0
    matrix = None
    for time in events:
        if temperature.size:
            temperature = advance_temperature(
                temperature, time - previous_time, matrix, equilibrium)
        active = np.searchsorted(births, time, side="right")
        if active > temperature.size:
            temperature = np.r_[temperature, np.full(
                active - temperature.size, settings.deposition_temperature)]
            matrix = thermal_matrix(active, mesh["cell_length"], settings)
        row = np.full(n_cells, np.nan)
        row[:active] = temperature
        times.append(float(time))
        history.append(row)
        previous_time = time

    closure_temperature = temperature.copy()
    closed_matrix = thermal_matrix(n_cells, mesh["cell_length"], settings, closed=True)

    def after_closure(duration):
        return advance_temperature(closure_temperature, duration, closed_matrix, equilibrium)

    initial_range = float(np.ptp(closure_temperature))
    if initial_range <= tolerance:
        wait = 0.0
    else:
        # Uniform lateral exchange alone gives this upper bound. Diffusion
        # further reduces the range, so this avoids unnecessary long solves.
        bound = maximum_cooling_time
        if settings.decay_rate > 0:
            bound = min(bound, np.log(initial_range / tolerance) / settings.decay_rate + 1e-6)
        if np.ptp(after_closure(bound)) > tolerance:
            raise RuntimeError("Uniformity tolerance not reached within maximum_cooling_time")
        wait = brentq(lambda dt: np.ptp(after_closure(dt)) - tolerance,
                      0, bound, xtol=1e-8)

    for duration in np.linspace(0, wait, cooling_samples)[1:]:
        if wait == 0:
            break
        times.append(print_time + float(duration))
        history.append(after_closure(duration))
    temperature = after_closure(wait)
    closure_balance = cell_heat_balance(
        closure_temperature, mesh["cell_length"], settings, closed=True)
    final_balance = cell_heat_balance(temperature, mesh["cell_length"], settings, closed=True)
    return {
        **mesh, "time": np.asarray(times), "temperature": np.asarray(history),
        "closure_temperature": closure_temperature,
        "final_temperature": temperature,
        "print_time": print_time, "wait_time": float(wait),
        "uniform_time": float(print_time + wait), "tolerance": tolerance,
        "equilibrium_temperature": equilibrium,
        "nozzle_equilibrium_temperature": settings.deposition_temperature,
        "numerics": stability_diagnostics(closed_matrix),
        "energy_diagnostics": {
            "cell_heat_capacity_J_K": cell_heat_capacity(mesh["cell_length"], settings),
            "closure_energy_relative_to_ambient_J": float(closure_balance["energy_J"].sum()),
            "final_energy_relative_to_ambient_J": float(final_balance["energy_J"].sum()),
            # On a closed loop conduction only redistributes energy. The net
            # external power must account for the entire total-energy rate.
            "closure_conduction_power_residual_W": float(
                (closure_balance["net_power_W"] - closure_balance["source_power_W"]).sum()),
        },
    }
