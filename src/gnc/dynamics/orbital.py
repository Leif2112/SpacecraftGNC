import numpy as np

def solve_kepler(ecc: float, M: float, tol: float = 1e-10, max_iter: int = 100) -> float:
    """
    Solve Kepler's equation M = E - e*sin(E) for eccentric anomaly E using Newton-Raphson method.

    Parameters:
        ecc : float
            Orbital eccentricity (0 <= e < 1)
        M : float
            Mean anomaly [rad]
        tol : float
            Tolerance for convergence
        max_iter : int
            Maximum number of iterations

    Returns:
        E : float
            Eccentric anomaly [rad]
    """
    if not (0 <= ecc < 1):
        raise ValueError("Eccentricity must be in range [0, 1).")

    E = M if ecc < 0.8 else np.pi  # good starting guess
    for _ in range(max_iter):
        f = E - ecc * np.sin(E) - M
        f_prime = 1 - ecc * np.cos(E)
        delta = f / f_prime
        E -= delta
        if abs(delta) < tol:
            return E
    raise RuntimeError("Kepler solver did not converge")

def coe_to_rv(coe: np.ndarray, mu: float) -> np.ndarray:
    """
    Convert classical orbital elements to ECI position and velocity vectors.

    Parameters:
        coe : np.ndarray
            Orbital elements [a, e, i, RAAN, arg_periapsis, true_anomaly] in radians
        mu : float
            Gravitational parameter [km^3/s^2]

    Returns:
        rv : np.ndarray
            6x1 state vector [x, y, z, vx, vy, vz] in ECI frame
    """
    a, e, i, RAAN, argp, nu = coe

    # Specific angular momentum
    h = np.sqrt(mu * a * (1 - e ** 2))

    # Orbital radius (magnitude of r)
    r = a * (1 - e ** 2) / (1 + e * np.cos(nu))

    # Perifocal position and velocity
    r_pf = r * np.array([
        np.cos(nu),
        np.sin(nu),
        0.0
    ])
    v_pf = (mu / h) * np.array([
        -np.sin(nu),
        e + np.cos(nu),
        0.0
    ])

    # Rotation matrices (exactly match MATLAB definitions)
    R3_RAAN = np.array([
        [np.cos(RAAN), np.sin(RAAN), 0],
        [-np.sin(RAAN), np.cos(RAAN), 0],
        [0, 0, 1]
    ])
    R1_i = np.array([
        [1, 0, 0],
        [0, np.cos(i), np.sin(i)],
        [0, -np.sin(i), np.cos(i)]
    ])
    R3_argp = np.array([
        [np.cos(argp), np.sin(argp), 0],
        [-np.sin(argp), np.cos(argp), 0],
        [0, 0, 1]
    ])

    # Construct P->ECI DCM as transpose of (R3_argp * R1_i * R3_RAAN)
    Q_pX = (R3_argp @ R1_i @ R3_RAAN).T

    r_eci = Q_pX @ r_pf
    v_eci = Q_pX @ v_pf

    return np.concatenate((r_eci, v_eci))

def tbp_eci_rhs(t: float, state: np.ndarray, mu: float) -> np.ndarray:
    """
    Compute RHS of the two-body problem in ECI frame.

    Parameters:
        t : float
            Time [s]
        state : np.ndarray
            6x1 state vector [x, y, z, vx, vy, vz]
        mu : float
            Gravitational parameter [km^3/s^2]

    Returns:
        dstate_dt : np.ndarray
            Derivative of state vector
    """
    r = state[:3]
    v = state[3:]
    norm_r = np.linalg.norm(r)

    a = -mu / norm_r**3 * r
    return np.concatenate((v, a))

def tbp_ecef_rhs(t: float, state: np.ndarray, mu: float, omega_earth: float) -> np.ndarray:
    """
    Compute RHS of the two-body problem in the ECEF frame (including Coriolis and centrifugal terms).

    Parameters:
        t : float
            Time [s]
        state : np.ndarray
            6x1 state vector [x, y, z, vx, vy, vz] in ECEF frame
        mu : float
            Gravitational parameter [km^3/s^2]
        omega_earth : float
            Earth rotation rate [rad/s]

    Returns:
        dstate_dt : np.ndarray
            Derivative of state vector
    """
    r = state[:3]
    v = state[3:]
    w = np.array([0, 0, omega_earth])

    acc_gravity = -mu / np.linalg.norm(r) ** 3 * r
    acc_coriolis = -2 * np.cross(w, v)
    acc_centrifugal = -np.cross(w, np.cross(w, r))

    a = acc_gravity + acc_coriolis + acc_centrifugal
    return np.concatenate((v, a))
