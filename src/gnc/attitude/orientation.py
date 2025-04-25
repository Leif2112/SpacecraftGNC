import numpy as np 


def rota121(EulerAng: np.ndarray):
    
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

    R_BO = R1_gamma @ R2_beta @ R1_alpha

    return R_BO

