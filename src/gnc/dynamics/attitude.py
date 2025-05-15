import numpy as np

def AttitudeDynamics(t: float, X: np.ndarray, Im: np.ndarray):
    ''''
    Function to integrate torque free attitude dynamics of the spacecraft.
    ⚠ Beware this equation is singular when 𝜃 = 𝜋/2, 3𝜋/2. This will cause a runtime overflow ⚠

    Inputs:
        t : time [s]
        X : Attitude state vector of spacecraft
        Im : Inertia matrix of spacecraft 

    Output: 
        Xdot_Att : attitude dynamics as a function of time --> time rate change of euler angles.
    '''
    x1, x2, x3 = X[0:3]
    AngVel = np.ravel(X[3:6])

    B = np.array([
        [0,          np.sin(x3),                np.cos(x3)],
        [0,          np.cos(x2) * np.cos(x3),  -np.cos(x2) * np.sin(x3)],
        [np.cos(x2), np.sin(x2) * np.sin(x3),   np.sin(x2) * np.cos(x3)]
    ])

    EulAng_dot = (1 / np.cos(x2)) * B @ AngVel      #Compute derivative of Euler Angles
    AngVel_rhs = -(np.cross(AngVel, Im @ AngVel))   #Compute    
    EulEq = AngVel_rhs @ Im

    return np.concatenate((EulAng_dot, EulEq))

def q_AttitudeDynamics(t: float, Q: np.ndarray, Im: np.ndarray, torque_func=None):
    ''''
    Function to integrate attitude dynamics of spacecraft.
    Defining continuous model to be used in our solver.
    
    Inputs:
        t           : time [s]
        Q           : Quaternion attitude state vector of spacecraft [q0, q1, q2, q3, wx, wy, wz]
        Im          : Inertia matrix of spacecraft 
        torque_func : callable or None
            A function torque_func(t, y) → np.ndarray(3,) that returns torque in body frame.
            If None, assumes torque-free.

    Output: 
        qdot_Att : attitude dynamics as a function of time --> time rate change of quaternions.
    '''

    q = Q[0:4]
    omega = Q[4:7]

    q = q / np.linalg.norm(q) # ALWAYS Normalise quaternion before use! May corrupt energy conservation otherwise...

    Omega = np.array([
        [ 0.0,     -omega[0], -omega[1], -omega[2]],
        [ omega[0],     0.0,   omega[2], -omega[1]],
        [ omega[1], -omega[2],     0.0,   omega[0]],
        [ omega[2],  omega[1], -omega[0],     0.0]
    ])
    
    q_dot = 0.5 * Omega @ q

    # External/internal torque (optional)
    if torque_func is not None:
        tau = torque_func(t, Q)  # must return shape (3,)
    else:
        tau = np.zeros(3)

    # Euler's equation with torque
    omega = np.ravel(omega)         # ensures (3,)
    Iomega = Im @ omega             # also (3,)
    gyro_term = np.cross(omega, Iomega)
    omega_dot = np.linalg.solve(Im, tau - gyro_term)
    
    return np.concatenate([q_dot, omega_dot])

def angularMomentum(Q_Att: np.ndarray, Im: np.ndarray, stepwise: bool = False):
    """
    Compute angular momentum vector H and its magnitude over time.

    Inputs: 
        Q_Att    : State history of shape (7, N). Rows 4:7 are angular velocity [wx; wy; wz].
        Im       : Inertia matrix of shape (3, 3).
        stepwise : bool, optional
            If True, compute H step-by-step. 
            If False, use vectorised NumPy operations.

    Outputs:
        H        : Angular momentum vectors, shape (3, N)
        Hmag     : Angular momentum magnitudes, shape (N,)
    """
    omega = Q_Att[4:7, :]  # shape (3, N)
    N = omega.shape[1]
    
    if not stepwise:
        # Vectorised version
        H = Im @ omega                        # shape (3, N)
        Hmag = np.linalg.norm(H, axis=0)     # shape (N,)
    else:
        # Stepwise version
        H = np.zeros((3, N))
        Hmag = np.zeros(N)
        for ii in range(N):
            w = omega[:, ii]                 # shape (3,)
            H[:, ii] = Im @ w
            Hmag[ii] = np.linalg.norm(H[:, ii])
    return H, Hmag

def KineticEnergy(Q_Att: np.ndarray, Im: np.ndarray, stepwise: bool = False):
    omega = Q_Att[4:7, :]  # (3, N)
    N = omega.shape[1]

    if not stepwise:
        # Vectorised version: T = 0.5 * sum(omega * H) over time
        H = Im @ omega                              # (3, N)
        T = 0.5 * np.einsum('ij,ij->j', omega, H)   # scalar product per column
    else:
        # Stepwise version
        T = np.zeros(N)
        for ii in range(N):
            w = omega[:, ii]
            H = Im @ w
            T[ii] = 0.5 * np.dot(w, H)

    return T, T                                     # Tmag is just the scalar value; duplicated for naming consistency

