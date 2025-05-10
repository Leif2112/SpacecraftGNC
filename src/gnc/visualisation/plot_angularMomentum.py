import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np

def plot_angular_momentum(t2: np.ndarray, Hmag: np.ndarray) -> None:
    """
    Plot magnitude of angular momentum vs time.

    Parameters
    ----------
    t2 : np.ndarray
        Time array (1D).
    Hmag : np.ndarray
        Angular momentum magnitude array (1D), same length as t2.
    """
    # Plot settings
    rcParams['text.usetex'] = False
    rcParams['font.size'] = 20
    rcParams['legend.fontsize'] = 15

    color = "#FF6188"

    # Figure size in inches
    picturewidth = 17.6  # cm
    hw_ratio = 0.75
    figsize = (picturewidth / 2.54, (hw_ratio * picturewidth) / 2.54)

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(t2, Hmag, color=color, linewidth=1.5, label="H")

    ax.set_xlabel('time t (s)')
    ax.set_ylabel('angular momentum H (kg m²/s)')
    ax.set_xlim([np.min(t2), np.max(t2)])
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.grid(True)
    ax.spines['left'].set_color('black')
    ax.yaxis.label.set_color('black')

    legend = ax.legend(loc='best', edgecolor='black')
    legend.set_title("")  # remove legend title if any

    ax.yaxis.set_label_coords(-0.08, 0.5)  # vertical centering (optional)
    plt.tight_layout()
    plt.show()
