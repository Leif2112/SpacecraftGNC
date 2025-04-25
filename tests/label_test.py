import pyvista as pv
import numpy as np

# Create plotter
R = 6378

pl = pv.Plotter(window_size=(900, 900))
pl.set_background("white")
pl.enable_terrain_style() 

# Set origin and axis length
origin = np.array([0, 0, 0])
axis_len = 1.5

def latlon_grid(R, n_lon=12, n_lat=6):
    """Return a PyVista PolyData containing latitude/longitude lines."""
    lines = []
    # longitudes (constant λ)
    lon = np.linspace(0, 2*np.pi, n_lon, endpoint=False)
    lat = np.linspace(-np.pi/2, np.pi/2, 181)
    for λ in lon:
        x = R*np.cos(lat)*np.cos(λ)
        y = R*np.cos(lat)*np.sin(λ)
        z = R*np.sin(lat)
        lines.append(pv.Spline(np.c_[x, y, z], 180))
    # latitudes (constant φ)
    lat = np.linspace(-np.pi/2+np.pi/n_lat, np.pi/2-np.pi/n_lat, n_lat-1)
    lon = np.linspace(0, 2*np.pi, 361)
    for φ in lat:
        x = R*np.cos(φ)*np.cos(lon)
        y = R*np.cos(φ)*np.sin(lon)
        z = np.full_like(lon, R*np.sin(φ))
        lines.append(pv.Spline(np.c_[x, y, z], 360))
    return lines

for curve in latlon_grid(R * 1.001, n_lon=64, n_lat=32):
        pl.add_mesh(curve, color="black", line_width=2)

# Define unit vectors and labels
axes = [(np.array([1, 0, 0]), "I"),
        (np.array([0, 1, 0]), "J"),
        (np.array([0, 0, 1]), "K")]

# Add axes, arrows and labels
for vec, lbl in axes:
    endpoint = origin + vec * R*1.5

    # Draw line for the axis
    line = pv.Line(origin, endpoint)
    pl.add_mesh(line, color="black", line_width=4)

    # Add arrow at the end
    arrow = pv.Arrow(
        np.array([endpoint]),
        direction=vec,
        scale=R*0.1,
        shaft_radius=0.01,
        tip_length=0.75,
        tip_radius=0.25
    )
    pl.add_mesh(arrow, color="black")

    # Add label slightly beyond the arrow
    pl.add_point_labels(
        np.array([endpoint * 1.1]),
        [lbl],
        text_color="red",
        font_size=20,
        point_size=0,
        show_points=False,
        shape_opacity=0.0
    )

# Show plot
pl.show()
