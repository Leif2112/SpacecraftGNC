import numpy as np

def axis_angle_to_quaternion(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    """ Convert axis-angle to unit quaternion [x, y, z, w] """
    axis = axis / np.linalg.norm(axis)
    sin_half = np.sin(angle_rad / 2)
    cos_half = np.cos(angle_rad / 2)
    q_xyz = axis * sin_half
    q_w = cos_half
    q = np.concatenate((q_xyz, [q_w]))
    return q / np.linalg.norm(q)

def quaternion_to_dcm(q: np.ndarray) -> np.ndarray:
    """ Convert quaternion [x, y, z, w] to DCM """
    x, y, z, w = q
    return np.array([
        [x**2 - y**2 - z**2 + w**2, 2*(x*y + w*z),        2*(x*z - w*y)],
        [2*(x*y - w*z),             -x**2 + y**2 - z**2 + w**2, 2*(y*z + w*x)],
        [2*(x*z + w*y),             2*(y*z - w*x),        -x**2 - y**2 + z**2 + w**2]
    ])

def dcm_to_euler_angles(dcm: np.ndarray) -> np.ndarray:
    """ Convert DCM to ZYX Euler angles [psi, theta, phi] """
    psi = np.arctan2(dcm[0,1], dcm[0,0])
    theta = np.arcsin(-dcm[0,2])
    phi = np.arctan2(dcm[1,2], dcm[2,2])
    return np.array([psi, theta, phi])

def omega_matrix(omega: np.ndarray) -> np.ndarray:
    """ Construct 4x4 Omega matrix for quaternion propagation """
    wx, wy, wz = omega
    return np.array([
        [0, -wx, -wy, -wz],
        [wx,  0,  wz, -wy],
        [wy, -wz, 0,  wx],
        [wz,  wy, -wx, 0]
    ])

def attitude_rhs(t: float, state: np.ndarray, J: np.ndarray, torque: np.ndarray = None) -> np.ndarray:
    """
    Compute RHS for attitude dynamics.
    State = [q0, q1, q2, q3, wx, wy, wz] (quaternion + angular velocity)
    """
    q = state[:4]  # [x, y, z, w]
    omega = state[4:]
    if torque is None:
        torque = np.zeros(3)

    # Quaternion derivative
    Omega = omega_matrix(omega)
    q_dot = 0.5 * Omega @ q

    # Angular velocity derivative
    J_inv = np.linalg.inv(J)
    omega_dot = J_inv @ (torque - np.cross(omega, J @ omega))

    return np.concatenate((q_dot, omega_dot))

def angular_momentum(J: np.ndarray, omega: np.ndarray) -> np.ndarray:
    return J @ omega

def rotational_kinetic_energy(J: np.ndarray, omega: np.ndarray) -> float:
    return 0.5 * omega.T @ J @ omega
