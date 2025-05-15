import numpy as np
from numba import njit
from pyproj import Transformer

@njit
def solve_kepler(ecc: float, M: float, tol: float = 1e-10) -> tuple[float, int]:
    """
    Solve Kepler's equation M = E - e*sin(E) for eccentric anomaly E using Newton-Raphson method.

    Parameters:
        ecc : Orbital eccentricity (0 <= e < 1)
        M : Mean anomaly [rad]
        tol : Tolerance for convergence
        max_iter : Maximum number of iterations

    Returns:
        E : Eccentric anomaly [rad]
    """
    if not (0 < ecc <= 0.99):
        raise ValueError("Eccentricity must be in range [0, 1).")
    
    En = M                          #Eucated guess
    i = 0                           #initialise interation count
    fEn = En - ecc * np.sin(En) - M    #initialise F(En)

    while np.abs(fEn) > tol:        #iterate while F(En) is higher than threshold value
        i += 1
        fEn = En - ecc * np.sin(En) - M
        fpEn = 1 - ecc * np.cos(En)
        Ens = En - fEn/fpEn
        En = Ens
    
    E = np.mod(En, 2 * np.pi)
    return E, i  
    

def coe_to_rv(coe: np.ndarray, mu: float) -> np.ndarray:
    """
    Convert classical orbital elements to ECI position and velocity vectors.

    Parameters:
        coe : Orbital elements [a, e, i, RAAN, arg_periapsis, true_anomaly] in radians
        mu : Gravitational parameter [km^3/s^2]

    Returns:
        rv : 6x1 state vector [x, y, z, vx, vy, vz] in ECI frame
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

def tbp_eci(t: float, state: np.ndarray, mu: float) -> np.ndarray:
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

def tbp_ecef(t: float, state: np.ndarray, mu: float, we: float) -> np.ndarray:
    """
    Compute RHS of the two-body problem in the ECEF frame (including Coriolis and centrifugal terms).

    Parameters:
        t : Time [s]
        state : 6x1 state vector [x, y, z, vx, vy, vz] in ECEF frame
        mu : Gravitational parameter [km^3/s^2]
        we : Earth rotation rate [rad/s]

    Returns:
        dstate_dt : np.ndarray
            Derivative of state vector
    """
    r = state[:3]
    v = state[3:]
    w = np.array([0, 0, we])

    acc_gravity = -mu / np.linalg.norm(r) ** 3 * r
    acc_coriolis = -2 * np.cross(w, v)
    acc_centrifugal = -np.cross(w, np.cross(w, r))

    a = acc_gravity + acc_coriolis + acc_centrifugal
    return np.concatenate((v, a))


def specific_energy(r_Xout:np.ndarray, v_Xout: np.ndarray, mu: float, a: float):
    """
    Compute the specific energy of the spacecraft at every time step.
    
    Parameters: 
        r_Xout : position mag. of SC @ every time step
        v_Xout : velocity mag. of SC @ every time step
        state : state vector of the SC in ECI frame
        mu : Gravitational Paramter
        a : Orbit semi-major axis

     Returns:
        sp_e : Specific energy at each time step
        sp_e2 : Constant specific energy estimate from orbital mechanics: -mu / (2a)
    """
    
    sp_e = 0.5 * v_Xout**2 - mu / r_Xout        # SC sp_e should be approx. constant 
    sp_e2 = np.full_like(sp_e, -mu / (2 * a))   # filled with the constant value -GM / (2*a) @ every t

    return sp_e, sp_e2

