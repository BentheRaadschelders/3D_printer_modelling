import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

from printer_model.dynamics import dynamics
from printer_model.parameters import param


def voltage_signal(t):
    return 1.0, 1.0


initial_state = np.zeros(6)

solution = solve_ivp(
    dynamics,
    t_span=(0.0, 1.0),
    y0=initial_state,
    args=(voltage_signal, param),
    method="Radau",
    rtol=1e-6,
    atol=1e-9,
)

if not solution.success:
    raise RuntimeError(solution.message)

fig, axes = plt.subplots(3, 1, sharex=True, figsize=(8, 8))

axes[0].plot(solution.t, solution.y[0], label="Left motor")
axes[0].plot(solution.t, solution.y[1], "--", label="Right motor")
axes[0].set_ylabel("Angle")

axes[1].plot(solution.t, solution.y[2], label="Left motor")
axes[1].plot(solution.t, solution.y[3], "--", label="Right motor")
axes[1].set_ylabel("Angular velocity")

axes[2].plot(solution.t, solution.y[4], label="Left motor")
axes[2].plot(solution.t, solution.y[5], "--", label="Right motor")
axes[2].set_ylabel("Current")
axes[2].set_xlabel("Time")

for ax in axes:
    ax.grid(True)
    ax.legend()

fig.tight_layout()
plt.show()
