import numpy as np  
import matplotlib.pyplot as plt

def anomalyPlot(t: np.ndarray, E: np.ndarray, TA: np.ndarray, MAt: np.ndarray):
    """
        PLot True, Mean & Eccentric anomaly against time
        
        Parameters:
            t : Time array [s]
            E : Eccentric Anomaly [rad]
            TA : True Anomaly [rad]
            MAt : Mean Anomaly [rad]
            
    """
    Ediff = E - MAt
    TAdiff = TA - MAt
    

    # Plotting
    fig, ax1 = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor("white")
    ax1.set_facecolor("white")

    # Left Y-axis
    ax1.plot(t, E, label="EA", color="#FF6188", linewidth=1.5)
    ax1.plot(t, TA, label="TA", color="#78DCE8", linewidth=1.5)
    ax1.plot(t, MAt, label="MAt", color="#AB9DF2", linewidth=1.5)
    ax1.set_xlabel("time $t$ $(s)$", fontsize=12)
    ax1.set_ylabel("angle $(rad)$", fontsize=12)
    ax1.tick_params(axis='y', labelcolor='black')
    ax1.set_xlim([np.min(t), np.max(t)])
    ax1.grid(True, linestyle='--', color='gray', linewidth=0.5)

    # Right Y-axis
    ax2 = ax1.twinx()
    ax2.plot(t, Ediff, label="EA deviation from MA", linestyle="--", color="#FF6188", linewidth=1.5)
    ax2.plot(t, TAdiff, label="TA deviation from MA", linestyle="-.", color="#78DCE8", linewidth=1.5)
    ax2.set_ylabel("deviation angle $(rad)$", fontsize=12)
    ax2.tick_params(axis='y', labelcolor='black')
    ax2.set_ylim([-5e-3, 5e-3])

    # Combine legends
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="upper left", fontsize=10, frameon=True)

    fig.suptitle("Mean, Eccentric and True Anomalies vs. time", fontsize=14, color='black')
    fig.tight_layout()
    fig.subplots_adjust(top=0.9)
    plt.show()
