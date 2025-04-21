import numpy as np
import matplotlib.pyplot as plt

def plot_error_magnitudes(t: np.ndarray, vdiff: np.ndarray, rdiff: np.ndarray):
    """
    Plot velocity and position error magnitudes on dual y-axes.

    Parameters:
        t : np.ndarray
            Time array [s]
        vdiff : np.ndarray
            Velocity error magnitudes [m/s]
        rdiff : np.ndarray
            Position error magnitudes [m]
    """
    fig, ax1 = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("#2e2e2e")
    ax1.set_facecolor("#2e2e2e")

    # Left Y-axis: velocity error
    ax1.set_ylabel("velocity error $\\epsilon_v$ $(m/s)$", color="#FF6188")
    ax1.plot(t, vdiff, color="#FF6188", linewidth=1, label=r"$\epsilon_v$")
    ax1.tick_params(axis='y', labelcolor="#FF6188")
    ax1.tick_params(axis='x', labelcolor="white")
    ax1.set_xlabel("time $t$ $(s)$", color="white")
    ax1.spines['left'].set_color("white")
    ax1.spines['bottom'].set_color("white")

    # Right Y-axis: position error
    ax2 = ax1.twinx()
    ax2.set_ylabel("position error $\\epsilon_r$ $(m)$", color="#78DCE8")
    ax2.plot(t, rdiff, color="#78DCE8", linewidth=1, label=r"$\epsilon_r$")
    ax2.tick_params(axis='y', labelcolor="#78DCE8")
    ax2.spines['right'].set_color("white")

    # Grid, limits, legend
    ax1.grid(True, color="#444444")
    ax1.set_xlim([t.min(), t.max()])
    fig.suptitle("Velocity and Position Error vs Time", color="white", fontsize=14)

    # Combined legend in top right
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right",
               facecolor="#2e2e2e", edgecolor="white", labelcolor='white')

    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.show()
