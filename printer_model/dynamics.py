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
