import numpy as np
import pyvista as pv
from scipy.integrate import solve_ivp
from scipy.spatial.transform import Rotation as R


def add_energy_contours_on_sphere(pl: pv.Plotter,
                                  Im: np.ndarray,
                                  radius: float,
                                  n_levels: int = 12,
                                  color: str = "black",
                                  width: float = 1.2):
    """
    Add iso-energy contour lines for the inertia tensor Im on the momentum sphere.
    """
    # High-res unit sphere
    sphere = pv.Sphere(radius=1.0, theta_resolution=300, phi_resolution=200)
    pts = sphere.points.T  # shape (3, N)

    # ω = I⁻¹ H  and  T = 0.5 ωᵀ I ω
    omega = np.linalg.inv(Im) @ pts
    T_vals = 0.5 * np.sum(omega * (Im @ omega), axis=0)

    sphere["T"] = T_vals
    tmin, tmax = T_vals.min(), T_vals.max()
    isosurfaces = np.linspace(tmin, tmax, n_levels)

    contours = sphere.contour(isosurfaces=isosurfaces, scalars="T")
    contours.points *= radius

    pl.add_mesh(contours, color=color, line_width=width, name=f"Contours_{id(Im)}")


def plot_polhode(H: np.ndarray,
                 Hmag: np.ndarray,
                 Im: np.ndarray,
                 T: np.ndarray) -> None:
    """
    Plot polhode motion with both vertical and horizontal iso-energy contours.

    Parameters
    ----------
    H : (3, N) ndarray
        Angular momentum history in body frame.
    Hmag : (N,) ndarray
        Magnitude of H at each time (constant for torque-free).
    Im : (3,3) ndarray
        Diagonal inertia tensor.
    T : (N,) ndarray
        Rotational kinetic energy over time.
    """
    R0 = Hmag[0]
    origin = np.array([0, 0, 0])

    pl = pv.Plotter(window_size=(900, 900))
    pl.set_background("#f4f4f4")
    pl.enable_eye_dome_lighting()

    # ───────────────────── Momentum sphere ─────────────────────
    sphere = pv.Sphere(radius=R0, theta_resolution=180, phi_resolution=180)
    sphere.compute_normals(inplace=True)
    pl.add_mesh(sphere, color="#d8f1f1", opacity=0.3, smooth_shading=True)

    # ───────────────────── Iso-energy contours ─────────────────────
    # Vertical ellipsoid (original)
    add_energy_contours_on_sphere(pl, Im, radius=R0, n_levels=12, color="black", width=1.2)

    # Horizontal ellipsoid (Im rotated 90° about X-axis)
    rot_x = R.from_euler('x', 90, degrees=True).as_matrix()
    Im_x = rot_x @ Im @ rot_x.T
    add_energy_contours_on_sphere(pl, Im_x, radius=R0, n_levels=12, color="black", width=1.2)

    # ───────────────────── Main polhode curve ─────────────────────
    H_curve = pv.lines_from_points(H.T)
    pl.add_mesh(H_curve, color="#FF6188", line_width=4, name="MainPolhode")

    # ───────────────────── Principal axes ─────────────────────
    axes = [(np.array([1, 0, 0]), "H₁"),
            (np.array([0, 1, 0]), "H₂"),
            (np.array([0, 0, 1]), "H₃")]

    for vec, lbl in axes:
        arrow = pv.Arrow(start=origin,
                         direction=vec,
                         scale=R0 * 1.2,
                         tip_length=0.3,
                         tip_radius=0.1,
                         shaft_radius=0.03)
        pl.add_mesh(arrow, color="black")
        pl.add_point_labels(np.array([vec * R0 * 1.25]), [lbl],
                            font_size=14, text_color="black")

    # ───────────────────── Final render ─────────────────────
    pl.show_grid(color="gray")
    pl.view_vector([1, 1, 0.75])  # akin to MATLAB view(135,20)
    pl.show(title="Polhode Plot with Vertical & Horizontal Contours")