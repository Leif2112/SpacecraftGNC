import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from gnc.dynamics.orbital import coe_to_rv, tbp_eci_rhs

mu_earth = 398600.4418  # km^3/s^2

def run_energy_diagnostics():
    # Initial COEs (circular-ish sun-synchronous orbit)
    coe = np.array([
        7151.6,                # semi-major axis [km]
        0.0008,                # eccentricity
        np.deg2rad(98.39),    # inclination [rad]
        np.deg2rad(10),       # RAAN [rad]
        np.deg2rad(233),      # argument of periapsis [rad]
        np.deg2rad(0)         # true anomaly [rad]
    ])

    state0 = coe_to_rv(coe, mu_earth)
    t_span = (0, 6000)  # seconds
    t_eval = np.linspace(t_span[0], t_span[1], 1000)

    sol = solve_ivp(tbp_eci_rhs, t_span, state0, args=(mu_earth,), t_eval=t_eval, rtol=1e-9, atol=1e-9)
    r_vec = sol.y[:3, :]
    v_vec = sol.y[3:, :]

    r_mag = np.linalg.norm(r_vec, axis=0)
    v_mag = np.linalg.norm(v_vec, axis=0)

    zeta2 = 0.5 * v_mag**2 - mu_earth / r_mag
    zeta1 = np.full_like(zeta2, -mu_earth / (2 * coe[0]))

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(sol.t, zeta1, 'r-', linewidth=1.5, label=r'$\zeta_1 = -\frac{\mu}{2a}$')
    plt.plot(sol.t, zeta2, 'c-', linewidth=1.2, label=r'$\zeta_2 = \frac{v^2}{2} - \frac{\mu}{r}$')
    plt.xlabel(r'time $t$ (s)', fontsize=12)
    plt.ylabel(r'specific energy $\zeta$ $(\mathrm{km}^2/\mathrm{s}^2)$', fontsize=12)
    plt.title('Figure I-4 Specific Energy vs. time', fontsize=13, style='italic')
    plt.legend(loc='upper right', fontsize=12)
    plt.grid(True)
    plt.ylim(-29, -26.5)
    plt.tight_layout()
    plt.savefig("energy_vs_time.png")
    plt.show()

if __name__ == "__main__":
    run_energy_diagnostics()