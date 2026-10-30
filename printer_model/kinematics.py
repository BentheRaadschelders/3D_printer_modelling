def forward_position(theta_l, theta_r, r_m):
    x = 0.5 * r_m * (theta_l + theta_r)
    y = 0.5 * r_m * (theta_l - theta_r)
    return x, y


def forward_velocity(omega_l, omega_r, r_m):
    vx = 0.5 * r_m * (omega_l + omega_r)
    vy = 0.5 * r_m * (omega_l - omega_r)
    return vx, vy
