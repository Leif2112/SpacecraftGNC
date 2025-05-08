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



