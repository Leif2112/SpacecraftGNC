import numpy as np 


def rota121(EulerAng: np.ndarray):
    
    """
    Rotation matrices for 1-2-1 sequence of rotations

        -> DCM from Orbital to Body Frame 
    """

    alpha = EulerAng[0]
    beta = EulerAng[1]
    gamma = EulerAng[2]

    R1_alpha = np.array([
        [1, 0, 0],
        [0, np.cos(alpha),  np.sin(alpha)],
        [0, -np.sin(alpha), np.cos(alpha)]
    ])

    R2_beta = np.array([
        [np.cos(beta), 0, -np.sin(beta)],
        [0, 1, 0],
        [np.sin(beta), 0, np.cos(beta)]
    ])

    R1_gamma = np.array([
        [1, 0, 0],
        [0, np.cos(gamma),  np.sin(gamma)],
        [0, -np.sin(gamma), np.cos(gamma)]
    ])

    R_BO = R1_gamma @ R2_beta @ R1_alpha    #DCM from orbital to body frame

    return R_BO

def OrbTo_EulAx(r6: np.array, v6: np.array, R_BO: np.ndarray):
    """
    Determine Euler angles and principle axes from orbital state
    """ 

    O1 = r6 / np.linalg.norm(r6)
    h6 = np.cross(r6, v6)
    O3 = h6 / np.linalg.norm(h6)
    O2 = np.cross (O3, O1)

    #DCM that converts the state of the spacecraft from orbital frame to ECI frame is determined given R_IO = (R_OI)^T --> transpose of R_OI 
    R_OI = np.array([
        [O1[0], O1[1], O1[2]],
        [O2[0], O2[1], O2[2]],
        [O3[0], O3[1], O3[2]]
    ])
    
    #then the DCM from ECI frame to body frame, by definition, is R_BI = R_BO * R_OI
    R_BI = R_BO @ R_OI
   
    EulAng_BO = np.acos((np.trace(R_BI) - 1) / 2)   #Determine Euler angles to rotate from Orbital to Body frame
    EulAX_1 = (R_BI[1, 2] - R_BI[2, 1]) / (2 * np.sin(EulAng_BO))
    EulAX_2 = (R_BI[2, 0] - R_BI[0, 2]) / (2 * np.sin(EulAng_BO))
    EulAX_3 = (R_BI[0, 1] - R_BI[1, 0]) / (2 * np.sin(EulAng_BO))


    EulAx = np.array([EulAX_1, EulAX_2, EulAX_3])

    return EulAx, EulAng_BO

def EulTo_Quat(EulAx: float, EulAng: np.ndarray):
    '''
    Function to convert an Euler rotation to quaternion representation. 
    This will become obsolete once attitude determination is implemented. This is a placeholder. 
    A quaternion can be defined in terms of the principal rotation components as:

    q = 𝒆̂ sin Φ / 2
    q4 = cos Φ / 2
    '''

    q123 = EulAx * np.sin(EulAng/ 2)
    q4 = np.cos(EulAng / 2)
    q = np.append(q123, q4)
    scale = np.linalg.norm(q)
    q /= scale
    
    return q

def QuatTo_DCM(q: np.ndarray):
    """
    DCM for Inertial to Body Frame rotation in quaternion form.
    The rotation matrix can be constructed from the quaternion as : 
    𝑹 = (𝑞42 − 𝒒𝑇)𝑰𝟑 + 𝟐𝒒𝒒𝑇 − 2𝑞4𝒒×

    which develops as follows.
    """
    q1, q2, q3, q4 = q

    return np.array([
        [q1**2 - q2**2 - q3**2 + q4**2, 2*(q1*q2 + q4*q3),              2*(q1*q3 - q4*q2)],
        [2*(q1*q2 - q4*q3),             -q1**2 + q2**2 - q3**2 + q4**2, 2*(q2*q3 + q4*q1)],
        [2*(q1*q3 + q4*q2),             2*(q2*q3 - q4*q1),              -q1**2 - q2**2 + q3**2 + q4**2]
    ])

def dcmTo_Eul(dcm: np.ndarray):
    """
    The DCM to rotate from Inertial to Body Frame can also be performed as a 3-2-1 sequence about [𝜓, 𝜃, 𝜙] 
    """
    psi = np.arctan2(dcm[0,1], dcm[0,0])
    theta = np.arcsin(-dcm[0,2])
    phi = np.arctan2(dcm[1,2], dcm[2,2])

    return psi, theta, phi