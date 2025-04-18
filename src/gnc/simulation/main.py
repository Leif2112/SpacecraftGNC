import numpy as np
import matplotlib.pyplot as plt
from gnc.attitude.rigid_body import attitude_rhs, axis_angle_to_quaternion
from gnc.integrators.ode113_like import integrate_attitude_ode113_like
from gnc.visualisation.plot_specific_energy import plot_specific_energy

######################### ORBITAL ELEMENTS #########################

coe = np.array([
    7151.6,              # semi-major axis [km]
    0.0008,              # eccentricity
    np.deg2rad(98.39),   # inclination
    np.deg2rad(10),      # RAAN
    np.deg2rad(233),     # argument of periapsis
    0.0                  # true anomaly
])

mu = 398600.4418  # gravitational parameter
#####################################################################


def run(): 
    # Inertia matrix [kg·m²] (example spacecraft)
    J = np.diag([2500.0, 5000.0, 6500.0])

    # Initial angular velocity [rad/s]
    omega0 = np.array([0.001, 0.002, 0.003])

    # Initial quaternion (identity rotation)
    axis = np.array([0, 0, 1])
    angle = 0.0
    q0 = axis_angle_to_quaternion(axis, angle)

    # Combined initial state: [q0, q1, q2, q3, wx, wy, wz]
    y0 = np.concatenate((q0, omega0))

    # Simulation time parameters
    t0, tf = 0.0, 3600.0  # seconds
    t_eval = np.linspace(t0, tf, 1000)

    # Integrate
    sol = integrate_attitude_ode113_like(
        rhs=attitude_rhs,
        y0=y0,
        t_span=(t0, tf),
        t_eval=t_eval,
        J=J
    )

    # Extract results
    q = sol.y[:4, :]
    omega = sol.y[4:, :]

    # Compute quaternion norm error and angular momentum over time
    q_norm_error = np.abs(np.linalg.norm(q, axis=0) - 1.0)
    H = J @ omega
    H_mag = np.linalg.norm(H, axis=0)
    T = 0.5 * np.einsum('ij,ij->j', omega, H)

    """    # Plot quaternion norm error
    plt.figure(figsize=(10, 4))
    plt.plot(sol.t, q_norm_error)
    plt.xlabel("Time [s]")
    plt.ylabel("|norm(q) - 1|")
    plt.title("Quaternion Norm Drift")
    plt.grid(True)
    plt.tight_layout()

    # Plot angular momentum magnitude
    plt.figure(figsize=(10, 4))
    plt.plot(sol.t, H_mag)
    plt.xlabel("Time [s]")
    plt.ylabel("|H| [kg·m²/s]")
    plt.title("Angular Momentum Magnitude")
    plt.grid(True)
    plt.tight_layout()

    # Plot rotational kinetic energy
    plt.figure(figsize=(10, 4))
    plt.plot(sol.t, T)
    plt.xlabel("Time [s]")
    plt.ylabel("Kinetic Energy [J]")
    plt.title("Rotational Kinetic Energy")
    plt.grid(True)
    plt.tight_layout()  """

    

plot_specific_energy(coe, mu)

if __name__ == "__main__":
    run()
