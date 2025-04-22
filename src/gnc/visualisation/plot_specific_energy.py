import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

def plot_specific_energy(t, sp_e, sp_e2):
    fig, ax1 = plt.subplots(figsize=(17.6/2.54, 0.75 * 17.6 / 2.54))  # ~(7in, 5in)

    # Background color
    ax1.set_facecolor("white")
    fig.patch.set_facecolor("white")

    # Plot theoretical energy on left axis
    ax1.plot(t, sp_e2, linewidth=2, linestyle='-', color="#FF6188", label=r'$\zeta_1 = -\mu / (2a)$')
    ax1.set_xlabel("Time $t$ (s)", fontsize=13)
    ax1.set_ylabel("Specific Energy $\zeta_1$ (km$^2$/s$^2$)", fontsize=12, color="#FF6188")
    ax1.tick_params(axis='y', labelcolor="#FF6188", labelsize=11)
    ax1.tick_params(axis='x', labelsize=11)
    ax1.spines['left'].set_color("#FF6188")
    ax1.set_ylim(np.min(sp_e2) - 0.2, np.max(sp_e2) + 0.2)

    # Plot computed energy on right axis
    ax2 = ax1.twinx()
    ax2.plot(t, sp_e, linewidth=2, color="#78DCE8", label=r'$\zeta_2 = v^2/2 - \mu/r$')
    ax2.tick_params(axis='y', labelcolor='none', color='none')
    ax2.yaxis.label.set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.set_ylabel("Specific Energy $\zeta_2$ (km$^2$/s$^2$)", fontsize=12, color="#78DCE8")
    ax2.spines['right'].set_color("#78DCE8")
    ax2.set_ylim(np.min(sp_e) - 0.2, np.max(sp_e) + 0.2)

    # Grid
    ax1.grid(True, linestyle='--', linewidth=0.5, color='gray', alpha=0.5)

    # Title
    ax1.set_title("Conservation of Specific Orbital Energy", fontsize=14, weight='bold', pad=15)

    # Combined legend
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc='best', frameon=True, edgecolor='black', fontsize=11)

    # Inset for zoomed-in deviation of ζ2
    ax_inset = inset_axes(ax1, width="30%", height="35%", loc='lower right', borderpad=2)

    delta_e = sp_e - sp_e[0]
    ax_inset.plot(t, delta_e, color="#78DCE8", linewidth=1)
    ax_inset.axhline(0, color="red", linewidth=0.5, linestyle='--')
    ax_inset.set_xlim(t[0], t[-1])
    ax_inset.set_ylim(-1e-11, 1e-11)
    ax_inset.set_yticks([-0.5e-11, 0.0, 0.5e-11])
    ax_inset.set_xticks([])
    ax_inset.tick_params(labelsize=8)
    ax_inset.grid(True, linestyle='--', linewidth=0.4)

    # Overlay zoom label directly in the inset (top-left corner of the inset axes)
    ax_inset.text(
        0.02, 0.95,
        r"Zoom: $\Delta \zeta_2$",
        transform=ax_inset.transAxes,
        fontsize=9,
        verticalalignment='top',
        bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', boxstyle='round,pad=0.2')
    )

    fig.tight_layout()
    plt.show()
