from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class NozzleParameters:
    ambient_temperature: float = 297.0
    resistance: float = 10.0
    thermal_resistance: float = 11.0
    on_voltage: float = 14.0
    # Parameter values selected for Q1.5.
    inductance: float = 1e-4
    capacitance: float = 5e-4
    thermal_capacity: float = 8.0

    def __post_init__(self):
        values = np.array(list(vars(self).values()))
        if not np.all(np.isfinite(values)) or np.any(values <= 0):
            raise ValueError("Nozzle parameters must be finite and positive.")

    @property
    def equilibrium(self):
        """[T_n, i_L,n, u_R,n] for the constant on-voltage."""
        return np.array([
            self.ambient_temperature
            + self.thermal_resistance * self.on_voltage**2 / self.resistance,
            self.on_voltage / self.resistance,
            self.on_voltage,
        ])


def nozzle_dynamics(state, voltage, settings):
    """State order: temperature [K], inductor current [A], resistor voltage [V]."""
    T_n, i_L, u_R = state

    dT_n = (
        u_R**2 / settings.resistance
        - (T_n - settings.ambient_temperature) / settings.thermal_resistance
    ) / settings.thermal_capacity

    di_L = (voltage - u_R) / settings.inductance

    du_R = (i_L - u_R / settings.resistance) / settings.capacitance

    return np.array([dT_n, di_L, du_R])
