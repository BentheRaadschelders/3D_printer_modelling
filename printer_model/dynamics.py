import numpy as np

def dynamics(t, x, u_m, param):
    x = np.asarray(x)
    x_dot = np.zeros_like(x)
    theta_dot = x[2:4]
    current = x[4:6]
    u = np.asarray(u_m(t))
    r = param["r_m"]

    G = 0.5 * r * np.array([
        [1.0, 1.0],
        [1.0, -1.0]
    ])

    mass = np.array([
        [param["m_head"], 0],
        [0, param["m_head"] + param["m_rails"]]
    ])

    damping = np.array([
        [2 * param["mu_lx"] + 8 * param["mu_r"], 0],
        [0, 2 * param["mu_ly"] + 4 * param["mu_r"]]
    ])
    D = G.T @ damping @ G

    M = param["J_m"] * np.eye(2) + G.T @ mass @ G

    theta_dot_dot = np.linalg.solve(M, param["K_m"] * current - D @ theta_dot)

    i_dot = (u - param["R_m"] * current - param["K_e"] * theta_dot) / param["L_m"]

    x_dot = np.concatenate((theta_dot, theta_dot_dot, i_dot))
    return x_dot


def dynamics_decoupled(t, xx, u_m, param):
    x, vx, i_x, y, vy, i_y = np.asarray(xx)
    u_l, u_r = np.asarray(u_m(t))
    r = param["r_m"]

    u_x = u_l + u_r
    u_y = u_l - u_r

    motor_mass = 2 * param["J_m"] / r**2
    mass_x = param["m_head"] + motor_mass
    mass_y = param["m_head"] + param["m_rails"] + motor_mass

    damping_x = 2 * param["mu_lx"] + 8 * param["mu_r"]
    damping_y = 2 * param["mu_ly"] + 4 * param["mu_r"]

    ax = (param["K_m"] * i_x / r - damping_x * vx) / mass_x
    ay = (param["K_m"] * i_y / r - damping_y * vy) / mass_y

    di_x = (
        u_x - param["R_m"] * i_x - 2 * param["K_e"] * vx / r
    ) / param["L_m"]
    di_y = (
        u_y - param["R_m"] * i_y - 2 * param["K_e"] * vy / r
    ) / param["L_m"]

    return np.array([vx, ax, di_x, vy, ay, di_y])
