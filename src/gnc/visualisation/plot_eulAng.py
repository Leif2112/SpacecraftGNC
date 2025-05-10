import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np

def plot_euler_angles(t2: np.ndarray, X_Att: np.ndarray) -> None:
    """
    Plot Yaw, Pitch, Roll Euler angles vs. time in LaTeX style.

    Parameters:
    ----------
    t2 : np.ndarray
        Time array (1D).
    X_Att : np.ndarray
        Attitude state array (6 x N), where first 3 rows are Euler angles [psi; theta; phi].
    """
    # Set LaTeX font and style
    rcParams['text.usetex'] = False
    rcParams['font.size'] = 20
    rcParams['legend.fontsize'] = 15

    # Colours and labels
    colors = ["#FF6188", "#78DCE8", "#AB9DF2"]
    labels = [r"$Yaw$ $\psi$", r"$Pitch$ $\theta$", r"$Roll$ $\phi$"]

    # Create figure with size matching MATLAB (cm → inches)
    picturewidth = 17.6  # cm
    hw_ratio = 0.75
    figsize = (picturewidth / 2.54, (hw_ratio * picturewidth) / 2.54)

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(t2, X_Att[5, :], color=colors[0], linewidth=1.5, label=labels[0])
    ax.plot(t2, X_Att[6, :], color=colors[1], linewidth=1.5, label=labels[1])
    ax.plot(t2, X_Att[7, :], color=colors[2], linewidth=1.5, label=labels[2])

    ax.set_xlabel(r'time $t$ $(s)$')
    ax.set_ylabel(r'Euler angles $\Psi$ $(rad)$')
    ax.set_xlim([np.min(t2), np.max(t2)])
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.grid(True)
    ax.spines['left'].set_color('black')
    ax.yaxis.label.set_color('black')

    legend = ax.legend(loc='best', edgecolor='black')
    plt.tight_layout()
    plt.show()
