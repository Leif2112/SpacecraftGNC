import numpy as np
import pyvista as pv

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

def add_filled_arc(plotter, origin, normal, vec_start, vec_end,
                   radius, color, label,
                   n_pts=120, opacity=0.4, edge_width=2,
                   label_offset=1.05):
    """
    Draw a circular arc *and* a translucent filled sector.

    Parameters
    ----------
    (Same args as before) plus
    opacity : float
        0 = fully transparent, 1 = opaque.
    edge_width : float
        Width of the arc outline.
    """
    n = np.asarray(normal, float)
    n /= np.linalg.norm(n)

    a = np.asarray(vec_start, float)
    a /= np.linalg.norm(a)
    b = np.asarray(vec_end,   float)
    b /= np.linalg.norm(b)

    # signed angle from a to b about n
    ang = np.arctan2(np.dot(n, np.cross(a, b)), np.dot(a, b))
    ts  = np.linspace(0.0, ang, n_pts)

    # Rodrigues rotation
    arc_pts = []
    for t in ts:
        arc_pts.append(
            radius * (a*np.cos(t) +
                      np.cross(n, a)*np.sin(t) +
                      n*np.dot(n, a)*(1-np.cos(t)))
        )
    arc_pts = np.asarray(arc_pts) + origin

    # 1) outline
    outline = pv.Spline(arc_pts, n_pts)
    outline_actor = plotter.add_mesh(outline, color=color, line_width=edge_width)

    # 2) filled sector (triangle-fan: origin-arcPts[i]-arcPts[i+1])
    fan_pts = np.vstack([origin, arc_pts])
    n_tri   = fan_pts.shape[0] - 2
    faces   = np.hstack([[3, 0, i+1, i+2] for i in range(n_tri)])
    sector  = pv.PolyData(fan_pts, faces)
    sector_actor = plotter.add_mesh(sector, color=color, opacity=opacity, style='surface')

    # 3) label at mid-arc
    mid = arc_pts[len(arc_pts)//2] * label_offset
    label_actor = plotter.add_point_labels(
        np.array([mid]), [label],
        text_color="white", font_size=18, point_size=0, show_points=False, shape_color=color, shape_opacity=0.7 
    )
    return [outline_actor, sector_actor, label_actor]
    

def plot_orbit_eci(X, Re, mu):
    """
    Interactive 3-D plot of an ECI orbit with inertial axes and
    perifocal triad.  No texture – just a translucent reference sphere.

    Parameters
    ----------
    X : (6, N) ndarray
        ECI state history [r; v] in km, km·s⁻¹
    Re : float
        Reference radius (Earth) in km
    mu : float
        Gravitational parameter (not used here except for perifocal triad)
    """

    # ────────────────────────────── 1 | Plotter ─────────────────────────────
    pl = pv.Plotter(window_size=(900, 900))
    pl.set_background("white")
    pl.enable_terrain_style() 

    # ────────────────────────────── 2 | Reference sphere ────────────────────
    earth = pv.Sphere(radius=Re, theta_resolution=180, phi_resolution=100)
    pl.add_mesh(
        earth, color="white", opacity=1, name="Earth", lighting = False
    )

    for curve in latlon_grid(Re * 1.001, n_lon=64, n_lat=32):
        pl.add_mesh(curve, color="black", line_width=2)

    # ────────────────────────────── 3 | Orbit line ──────────────────────────
    r_xyz = X[:3] * 1.001                      # tiny lift to avoid z-fight
    orbit = pv.Line(r_xyz[:, 0], r_xyz[:, -1], resolution=r_xyz.shape[1]-1)
    orbit.points = r_xyz.T                    # overwrite with full track
    pl.add_mesh(orbit, color="#FF6188", line_width=4, name="Orbit")

    # start / end points
    pl.add_mesh(pv.PolyData(r_xyz[:, 0]),  color="#A9DC76",
                render_points_as_spheres=True, point_size=10)
    pl.add_mesh(pv.PolyData(r_xyz[:, -1]), color="#A9DC76",
                render_points_as_spheres=True, point_size=10)

    # ────────────────────────────── 4 | Perifocal triad ─────────────────────
    r0, v0 = X[:3, 0], X[3:6, 0]
    h_vec  = np.cross(r0, v0)
    e_vec  = np.cross(v0, h_vec) / mu - r0 / np.linalg.norm(r0)

    i_e = e_vec / np.linalg.norm(e_vec)
    i_h = h_vec / np.linalg.norm(h_vec)
    i_p = np.cross(i_h, i_e)

    origin    = np.zeros(3)
    triad_len = Re * 1.5
    

    for vec, lbl in [(i_e, "i_e"), (i_p, "i_p"), (i_h, "i_h")]:
        endpoint = origin + vec * triad_len

        line = pv.Line(origin, endpoint)
        pl.add_mesh(line, color="#78DCE8", line_width=4)
        pl.add_point_labels(
            np.array([endpoint + vec*Re*0.15]),
            [lbl], text_color="white", font_size=18, point_size=0, shape_opacity=0.75, shape_color="#78DCE8"
        )

        arrow = pv.Arrow(
            np.array([endpoint - vec*Re*0.025]),
            direction=vec,
            scale=Re*0.1,
            shaft_radius=0.04,
            tip_length=0.75,
            tip_radius=0.25
        )
        pl.add_mesh(arrow, color="#78DCE8")

    # ------------------------------ ARCS -----------------------------------
    arc_R = Re * 1.5

    # 1) Inclination  i  (between equatorial +Z and orbital plane)
    #    Draw inside the declination plane defined by node line (i_h)
    node_dir = np.cross([0, 0, 1], i_h)    # line of nodes (ascending)
    node_dir /= np.linalg.norm(node_dir)
    
    arcs = []
    arcs += add_filled_arc(pl, origin=np.zeros(3), normal=node_dir,
            vec_start=[0, 0, 1], vec_end=i_h, radius=arc_R,
            color="#cb0c59", label="incl.")

    # 2) RAAN  Ω  (from +X to ascending node in equatorial plane)
    asc_node = np.cross([0, 0, 1], i_h)
    asc_node /= np.linalg.norm(asc_node)
    arcs += add_filled_arc(pl, origin=np.zeros(3), normal=[0, 0, 1],
            vec_start=[1, 0, 0], vec_end=asc_node, radius=arc_R,
            color="#00f0ff", label="RAAN")

    # 3) Argument of perigee  ω  (from node to periapsis in orbital plane)
    peri_dir = i_e                       # eccentricity vector points to periapsis
    arcs += add_filled_arc(pl, origin=np.zeros(3), normal=i_h,
            vec_start=asc_node, vec_end=peri_dir, radius=arc_R,
            color="#defe47", label="Arg. P")
    
    def toggle_arcs(state):
        for actor in arcs:
            actor.SetVisibility(state)
        

    pl.add_checkbox_button_widget(
        callback=toggle_arcs,
        value=True,
        position=(10, 10),
        size=30,
        color_on='#defe47',
        color_off='gray',
        border_size=4,
        background_color='white'
    )


    # ────────────────────────────── 5 | Inertial I-J-K axes ────────────────

    axes = [(np.array([1, 0, 0]), "I"),
            (np.array([0, 1, 0]), "J"),
            (np.array([0, 0, 1]), "K")]

# Add axes, arrows and labels
    for vec, lbl in axes:
        endpoint = origin + vec * Re*1.5

        # Draw line for the axis
        line = pv.Line(origin, endpoint)
        pl.add_mesh(line, color="black", line_width=4)

        # Add arrow at the end
        arrow = pv.Arrow(
            np.array([endpoint - vec*Re*0.025]),
            direction=vec,
            scale=Re*0.1,
            shaft_radius=0.04,
            tip_length=0.75,
            tip_radius=0.25
        )
        pl.add_mesh(arrow, color="black")

        # Add label slightly beyond the arrow
        pl.add_point_labels(
            np.array([endpoint * 1.1]),
            [lbl],
            text_color="black",
            font_size=20,
            point_size=0,
            show_points=False,
            shape_opacity=0.0
        )

    # ────────────────────────────── 6 | Camera & render ────────────────────

    pl.show_grid(color="lightgrey")      # compatible with every PyVista ≥ 0.32

    pl.show(title="Spacecraft ECI Orbital Trajectory")





def plot_orbit_ecef(X, Re):
    """
    Interactive 3-D plot of an ECEF orbit with inertial axes and
    perifocal triad.  No texture – just a translucent reference sphere.

    Parameters
    ----------
    X : (6, N) ndarray
        ECEF state history [r; v] in km, km·s⁻¹
    Re : float
        Reference radius (Earth) in km
    mu : float
        Gravitational parameter (not used here except for perifocal triad)
    """

    # ────────────────────────────── 1 | Plotter ─────────────────────────────
    pl = pv.Plotter(window_size=(900, 900))
    pl.set_background("white")
    pl.enable_terrain_style() 

    # ────────────────────────────── 2 | Reference sphere ────────────────────
    earth = pv.Sphere(radius=Re, theta_resolution=180, phi_resolution=100)
    pl.add_mesh(
        earth, color="white", opacity=1, name="Earth", lighting = False
    )

    for curve in latlon_grid(Re * 1.001, n_lon=64, n_lat=32):
        pl.add_mesh(curve, color="black", line_width=2)

    # ────────────────────────────── 3 | Orbit line ──────────────────────────
    r_xyz = X[:3] * 1.001                      # tiny lift to avoid z-fight
    orbit = pv.Line(r_xyz[:, 0], r_xyz[:, -1], resolution=r_xyz.shape[1]-1)
    orbit.points = r_xyz.T                    # overwrite with full track
    pl.add_mesh(orbit, color="#78DCE8", line_width=4, name="Orbit")

    # start / end points
    pl.add_mesh(pv.PolyData(r_xyz[:, 0]),  color="#A9DC76",
                render_points_as_spheres=True, point_size=10)
    pl.add_mesh(pv.PolyData(r_xyz[:, -1]), color="red",
                render_points_as_spheres=True, point_size=10)

    # ────────────────────────────── 5 | Inertial I-J-K axes ────────────────

    origin    = np.zeros(3)
    axes = [(np.array([1, 0, 0]), "I"),
            (np.array([0, 1, 0]), "J"),
            (np.array([0, 0, 1]), "K")]

# Add axes, arrows and labels
    for vec, lbl in axes:
        endpoint = origin + vec * Re*1.5

        # Draw line for the axis
        line = pv.Line(origin, endpoint)
        pl.add_mesh(line, color="black", line_width=4)

        # Add arrow at the end
        arrow = pv.Arrow(
            np.array([endpoint - vec*Re*0.025]),
            direction=vec,
            scale=Re*0.1,
            shaft_radius=0.04,
            tip_length=0.75,
            tip_radius=0.25
        )
        pl.add_mesh(arrow, color="black")

        # Add label slightly beyond the arrow
        pl.add_point_labels(
            np.array([endpoint * 1.1]),
            [lbl],
            text_color="black",
            font_size=20,
            point_size=0,
            show_points=False,
            shape_opacity=0.0
        )

    # ────────────────────────────── 6 | Camera & render ────────────────────

    pl.show_grid(color="lightgrey")      # compatible with every PyVista ≥ 0.32

    pl.show(title="Spacecraft ECI Orbital Trajectory")
