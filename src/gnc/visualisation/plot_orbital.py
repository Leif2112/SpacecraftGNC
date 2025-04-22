import numpy as np
import pyvista as pv


def plot_orbit_eci(X, Re, mu):
    """
    Seam‑free 3‑D plot of an ECI orbit around a textured Earth.

    Parameters
    ----------
    X  : (6, N) ndarray
        ECI state history [r; v] in km and km·s⁻¹.
    Re : float
        Earth radius, km.
    mu : float
        Gravitational parameter, km³·s⁻².
    """
    # ──────────────────────────────────────────────────────────────────────────
    # 1.  Plotter
    # ──────────────────────────────────────────────────────────────────────────
    pl = pv.Plotter(window_size=(800, 800))
    pl.set_background("white")

       # ──────────────────────────────────────────────────────────────────────────
    # 2. Seamless Earth sphere with custom UV mapping (GL-style)
    # ──────────────────────────────────────────────────────────────────────────
    earth_tex = pv.read_texture("src/gnc/visualisation/flat_earth.jpg")

    # This trick forces PyVista to duplicate the seam vertices internally
    earth = pv.Sphere(radius=Re, theta_resolution=360, phi_resolution=180,
                      start_theta=270.001, end_theta=270)

    # Manually assign UVs (GLSL equirectangular projection)
    pts = earth.points
    u = 0.5 + np.arctan2(-pts[:, 0], pts[:, 1]) / (2 * np.pi)
    v = 0.5 + np.arcsin(pts[:, 2] / np.linalg.norm(pts, axis=1)) / np.pi
    earth.active_texture_coordinates = np.c_[u, v].astype(np.float32)

    # Apply the seamless texture
    pl.add_mesh(
        earth, texture=earth_tex,
        smooth_shading=True, name="Earth"
    )

    # ──────────────────────────────────────────────────────────────────────────
    # 3.  Orbit line (lifted 0.1 % to avoid z‑fight)
    # ──────────────────────────────────────────────────────────────────────────
    r = X[:3, :] * 1.001                # tiny offset
    orbit = pv.Line(r[:, 0], r[:, -1], resolution=r.shape[1] - 1)
    orbit.points = r.T
    pl.add_mesh(orbit, color="#FF6188", line_width=4, name="Orbit")

    pl.add_mesh(pv.PolyData(r[:, 0]),  color="#A9DC76",
                render_points_as_spheres=True, point_size=12)  # start
    pl.add_mesh(pv.PolyData(r[:, -1]), color="#A9DC76",
                render_points_as_spheres=True, point_size=12)  # end

    # ──────────────────────────────────────────────────────────────────────────
    # 4.  Perifocal triad at t₀
    # ──────────────────────────────────────────────────────────────────────────
    r0, v0 = X[:3, 0], X[3:6, 0]
    h_vec = np.cross(r0, v0)
    e_vec = np.cross(v0, h_vec) / mu - r0 / np.linalg.norm(r0)

    i_e = e_vec / np.linalg.norm(e_vec)
    i_h = h_vec / np.linalg.norm(h_vec)
    i_p = np.cross(i_h, i_e)

    origin = np.array([0.0, 0.0, 0.0])
    arrow_len = Re * 1.5           # long enough to emerge past Earth’s surface

    for vec, lbl in [(i_e, "$i_e$"), (i_p, "$i_p$"), (i_h, "$i_h$")]:
        start = origin
        end   = origin + vec * arrow_len

        line = pv.Line(start, end)
        pl.add_mesh(line, color="#78DCE8", line_width=6, name=lbl)

        pl.add_point_labels(
            np.array([end]),
            [lbl], text_color="#78DCE8", font_size=18, point_size=0
        )


    # ──────────────────────────────────────────────────────────────────────────
    # 5.  Inertial I‑J‑K axes
    # ──────────────────────────────────────────────────────────────────────────
    L = Re * 1.6
    axis_kw = dict(tip_length=Re * 0.15,
                   tip_radius=Re * 0.08,
                   shaft_radius=Re * 0.02)

    for vec, lbl in [((1, 0, 0), "I"), ((0, 1, 0), "J"), ((0, 0, 1), "K")]:
        pl.add_mesh(
            pv.Arrow(start=(0, 0, 0), direction=np.array(vec) * L, **axis_kw),
            color="black"
        )
        pl.add_point_labels(
            np.array(vec) * L * 1.15, [lbl],
            font_size=20, text_color="black", point_size=0
        )

    # ──────────────────────────────────────────────────────────────────────────
    # 6.  Camera & render
    # ──────────────────────────────────────────────────────────────────────────
    pl.camera_position = [
        (Re * 2.0, Re * 2.0, Re * 0.5),   # eye
        (0.0, 0.0, 0.0),                  # focal point
        (0.0, 0.0, 1.0),                  # view‑up
    ]
    pl.enable_anti_aliasing()
    pl.show(title="Spacecraft ECI Orbital Trajectory")
