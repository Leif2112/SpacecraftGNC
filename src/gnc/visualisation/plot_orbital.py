import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def plot_orbit_eci(X: np.ndarray, Re: float, mu: float):
    """
    Plot the spacecraft orbital trajectory in the ECI frame along with Earth and reference vectors.

    Parameters:
        X : np.ndarray
            State history array with shape (6, N), first 3 rows are position vectors.
        Re : float
            Radius of the Earth in km (default: 6378.0)
    """
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    fig.patch.set_facecolor('white')

    # Create Earth mesh
    u, v = np.mgrid[0:2 * np.pi:50j, 0:np.pi:25j]
    x = Re * np.cos(u) * np.sin(v)
    y = Re * np.sin(u) * np.sin(v)
    z = Re * np.cos(v)
    ax.plot_surface(x, y, z, rstride=1, cstride=1, color='lightgray', edgecolor='k', alpha=0.5)

    # Plot ECI axes
    ax.quiver(0, 0, 0, 1e4, 0, 0, color='k', linewidth=2)
    ax.quiver(0, 0, 0, 0, 1e4, 0, color='k', linewidth=2)
    ax.quiver(0, 0, 0, 0, 0, 1e4, color='k', linewidth=2)
    ax.text(1.22e4, 0, 0, "$I$", color='k')
    ax.text(0, 1.1e4, 0, "$J$", color='k')
    ax.text(0, 0, 1.1e4, "$K$", color='k')

    # Plot spacecraft trajectory
    ax.plot(X[0, :], X[1, :], X[2, :], color="#FF6188", linewidth=2)
    ax.plot([X[0, 0]], [X[1, 0]], [X[2, 0]], 'o', color="#A9DC76")  # Start
    ax.plot([X[0, -1]], [X[1, -1]], [X[2, -1]], 'o', color="#A9DC76")  # End

    # Eccentricity and angular momentum vectors
    r0 = X[0:3, 0]
    v0 = X[3:6, 0]
    h = np.cross(r0, v0)
    e_vec = np.cross(v0, h) / mu - r0 / np.linalg.norm(r0)  

    ih = h / np.linalg.norm(h)
    ie = e_vec / np.linalg.norm(e_vec)
    ip = np.cross(ih, ie) / np.linalg.norm(np.cross(ih, ie))

    ax.quiver(0, 0, 0, *ie * 1e4, color="#78DCE8", linewidth=2)
    ax.quiver(0, 0, 0, *ip * 1e4, color="#78DCE8", linewidth=2)
    ax.quiver(0, 0, 0, *ih * 1e4, color="#78DCE8", linewidth=2)
    ax.text(*ie * 1e4 * np.array([1, 1, 1.2]), "$i_e$", fontsize=14, color="#78DCE8")
    ax.text(*ip * 1e4 * np.array([1, 1, 1.1]), "$i_p$", fontsize=14, color="#78DCE8")
    ax.text(*ih * 1e4 * np.array([1, 1.15, 1]), "$i_h$", fontsize=14, color="#78DCE8")

    ax.set_xlabel("$I_{ECI} (km)$")
    ax.set_ylabel("$J_{ECI} (km)$")
    ax.set_zlabel("$K_{ECI} (km)$")
    ax.set_box_aspect([1, 1, 1])
    ax.view_init(elev=30, azim=130)
    plt.title("Spacecraft ECI Orbital Trajectory")
    plt.tight_layout()
    plt.show()
