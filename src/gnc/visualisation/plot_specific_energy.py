import numpy as np
import matplotlib.pyplot as plt
from gnc.dynamics.orbital import coe_to_rv

def plot_specific_energy(coe: np.ndarray, mu: float):
    """
    Plot specific orbital energy as a function of true anomaly.

    Parameters:
        coe : np.ndarray
            Classical orbital elements [a, e, i, RAAN, argp, TA]
        mu : float
            Gravitational parameter (km^3/s^2)
    """
    nu_array = np.linspace(0, 2*np.pi, 500)
    zeta1 = -mu / (2 * coe[0])
    zeta1_arr = np.full_like(nu_array, zeta1)
    zeta2_arr = []

    for nu in nu_array:
        coe_copy = coe.copy()
        coe_copy[-1] = nu
        state = coe_to_rv(coe_copy, mu)
        r = np.linalg.norm(state[:3])
        v = np.linalg.norm(state[3:])
        zeta2 = 0.5 * v**2 - mu / r
        zeta2_arr.append(zeta2)

    zeta2_arr = np.array(zeta2_arr)
    sp_diff = np.abs(zeta2_arr - zeta1_arr)

    # Plot with two y-axes
    fig, ax1 = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("#2e2e2e")
    ax1.set_facecolor("#2e2e2e")
    theta_deg = np.rad2deg(nu_array)

    ax2 = ax1.twinx()
    ax2.set_facecolor("#2e2e2e")
    ax2.plot(theta_deg, zeta2_arr, label=r"$\zeta_2 = \frac{v^2}{2} - \frac{\mu}{r}$", color="#78DCE8", linewidth=1.5)
    ax2.set_ylabel("specific energy $\zeta$ $(km^2/s^2)$", color="#78DCE8")
    ax2.tick_params(axis='y', labelcolor="#78DCE8")
    ax2.spines['right'].set_color('white')

    ax1.plot(theta_deg, zeta1_arr, label=r"$\zeta_1 = -\frac{\mu}{2a}$", color="#FF6188", linewidth=1.5)
    ax1.set_xlabel("True Anomaly $\nu$ (degrees)", color="white")
    ax1.set_ylabel("specific energy $\zeta$ $(km^2/s^2)$", color="#FF6188")
    ax1.tick_params(axis='x', labelcolor="white")
    ax1.tick_params(axis='y', labelcolor="#FF6188")

    # White plot box (spines)
    for spine in ax1.spines.values():
        spine.set_edgecolor('white')
    for spine in ax2.spines.values():
        spine.set_edgecolor('white')

    fig.suptitle("Specific Energy vs True Anomaly", fontsize=14, color='white')
    fig.tight_layout()
    fig.subplots_adjust(top=0.9)
    ax1.grid(True, color="#444444")
    ax1.legend(loc="upper right", facecolor="#2e2e2e", edgecolor="white", labelcolor='white')
    ax2.legend(loc="upper left", facecolor="#2e2e2e", edgecolor="white", labelcolor='white')
    plt.show()
