import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.geodesic import Geodesic

def plot_groundTrack(X_ECEF: np.ndarray, sat_name="ZHUHAI-1 01") -> None:
    """
    Plots an ASCII-style satellite ground track over a dark map.

    Parameters
    ----------
    X_ECEF : np.ndarray
        Shape (3, N). ECEF coordinates over time.
    sat_name : str
        Label for the active satellite position.
    """

    x, y, z = X_ECEF[:3, :]

    r = np.sqrt(x**2 + y**2 + z**2)
    lat = np.degrees(np.arcsin(z / r))
    lon = np.degrees(np.arctan2(y, x))
    lon = np.unwrap(np.radians(lon))
    lon = np.degrees(lon)

    # Create black figure
    fig = plt.figure(figsize=(16, 8), facecolor='black')
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_global()
    ax.set_facecolor('black')
    fig.patch.set_facecolor('black')

    # Add minimal grey land/coast outlines (dot feel)
    ax.add_feature(cfeature.LAND, facecolor='black', edgecolor='none')
    ax.add_feature(cfeature.COASTLINE, edgecolor='grey', linestyle=':', linewidth=0.2)
    ax.add_feature(cfeature.BORDERS, edgecolor='grey', linestyle=':', linewidth=0.2)

    # Plot orbital path in pixel style
    num_passes = 3
    pts_per_pass = len(lon) // num_passes
    for i in range(num_passes):
        i0 = i * pts_per_pass
        i1 = (i + 1) * pts_per_pass if i < num_passes - 1 else len(lon)
        color = "#78DCE8" if i % 2 == 0 else "#FFD866"
        ax.plot(
            lon[i0:i1], lat[i0:i1],
            linestyle="None",
            marker="s",
            markersize=2,
            color=color,
            transform=ccrs.Geodetic()
        )

    # Current satellite location
    ax.plot(lon[-1], lat[-1], "s", color="red", markersize=4, transform=ccrs.Geodetic())
    ax.text(
        lon[-1] + 2, lat[-1] + 1, f"{sat_name}",
        transform=ccrs.Geodetic(),
        fontsize=10,
        color="red",
        fontweight="bold",
        family="monospace"
    )

    # Simulate a visibility circle (like in reference image)
    gd = Geodesic()
    vis_circle = gd.circle(lon[-1], lat[-1], radius=1200e3, n_samples=120)
    ax.plot(vis_circle[:, 0], vis_circle[:, 1], color="yellow", linestyle=":", linewidth=0.7)

    # Remove axes, ticks, spines
    ax.set_xticks([])
    ax.set_yticks([])
    try:
        ax.spines['geo'].set_visible(False)
    except Exception:
        pass
    ax.gridlines(draw_labels=False)

    # Title with terminal-like style
    plt.title("Satellite Ground Track", color="white", fontfamily="monospace", fontsize=12)
    plt.tight_layout()
    plt.show()
