import numpy as np
import time
import matplotlib.pyplot as plt
from colorama import init, Fore, Style
from colorama.ansi import AnsiFore
import curses
import argparse

from .sim import display_sim_dashboard

from gnc.dynamics.orbital import solve_kepler, coe_to_rv, tbp_eci, tbp_ecef, specific_energy
from gnc.dynamics.attitude import AttitudeDynamics, q_AttitudeDynamics, angularMomentum, KineticEnergy 

from gnc.attitude.rigid_body import attitude_rhs, axis_angle_to_quaternion
from gnc.attitude.orientation import rota121, OrbTo_EulAx, EulTo_Quat, QuatTo_DCM, dcmTo_Eul, QuatTo_Eul

from gnc.integrators.ode113_like import ode113

from gnc.visualisation.plot_specific_energy import plot_specific_energy
from gnc.visualisation.plot_posVelError import plot_error_magnitudes
from gnc.visualisation.plot_anomalyVStime import anomalyPlot
from gnc.visualisation.plot_orbital import plot_orbit_eci, plot_orbit_ecef
from gnc.visualisation.plot_eulAng import plot_euler_angles
from gnc.visualisation.plot_angVel import plot_angular_velocity
from gnc.visualisation.plot_angularMomentum import plot_angular_momentum
from gnc.visualisation.plot_polhode import plot_polhode
from gnc.visualisation.plot_groundTrack import plot_groundTrack


# Initialize colorama once
init(autoreset=True)

def rgb(r, g, b):
    return f'\033[38;2;{r};{g};{b}m'

green = rgb(88, 176, 140)     
cyan = rgb(120, 220, 232)     # ~ "#78DCE8"
pink = rgb(255, 97, 136)      # ~ "#FF6188"
reset = Style.RESET_ALL

def cli():
    parser = argparse.ArgumentParser(description="Run spacecraft GNC simulation.")
    parser.add_argument("--ascii", action="store_true", help="Render ASCII ground track in terminal")
    args = parser.parse_args()

    run(ascii_plot=args.ascii)

######################### ORBITAL ELEMENTS #########################

coe = np.array([
    7151.16,              # semi-major axis [km]
    0.0008,              # eccentricity
    np.deg2rad(98.39),     # inclination
    np.deg2rad(10),      # RAAN
    np.deg2rad(233),     # argument of periapsis
    np.deg2rad(127)    # true anomaly
])


mu = 398600.4418         # gravitational parameter
Re = 6378.137            # radius of Earth [km]
Te = 86164.100           # Earth rotational period / sidereal day [s]
we = 2 * np.pi / Te      # angular velocity of Earth [rad/s]

Im = np.array([          # Spacecraft inertia matrix [kg·m²] 
    [2500, 0,    0],
    [0,    5000, 0],
    [0,    0,    6500]
])

AngVel = np.array([-3.092e-4, 6.6161e-4, 7.4606e-4])

tol = 10e-10         


#####################################################################

def run(ascii_plot=False): 

    #compute orbital period of spacecraft
    n = np.sqrt(mu / coe[0]**3)                      # mean motion [rad/s]
    P = 2 * np.pi / n                                # orbital period [s]
    

    #create time vector for plotting
    t0 = 0
    t = np.linspace(t0, P + t0, 1000)
    t2 = np.linspace(0, 1000, 100)
    
    E0, i = solve_kepler(coe[1], coe[5], tol=tol)                               # eccentric anomaly at t0 [rad]
    TA0 = 2 * np.arctan(np.sqrt((1 + coe[1]) / (1 - coe[1])) * np.tan(E0 / 2))  # true anomaly at t0 [rad]
    MAt = np.mod(coe[5] + n * (t - t0), 2 * np.pi)                              # mean anomaly [rad]

    E_matrix = np.zeros(len(t))
    TA = np.zeros(len(t))
    COE = np.zeros((6, len(t)))
    X = np.zeros((6, len(t)))

    #solve Kepler's equation for each time step 
    for tt in range(len(t)):
        E, _ = solve_kepler(coe[1], MAt[tt], tol=tol)          # eccentric anomaly as a function of time
        E_matrix[tt] = E                                    # store E in array for plotting    
        
        # True amomaly from eccentric anomaly
        TA[tt] = np.mod(
            2 * np.arctan2(
                np.sqrt(1 + coe[1]) * np.tan(E /2),
                np.sqrt(1 - coe[1])
            ),
            2 * np.pi
        )  
        
        #Store COE and convert to Cartesian coordinates / @COE2RV
        COE[:, tt] = np.array([ coe[0], coe[1], coe[2], coe[3], coe[4], TA[tt] ])
        X[:, tt] = coe_to_rv(COE[:, tt], mu)                # State vector from clasissical orbital elements to RV propagation

    solECI = ode113(
        rhs = tbp_eci,
        y0 = X[:, 0],
        t_span = (t[0], t[-1]),
        t_eval = t,
        args = (mu, )
    )
    Xout = solECI.y         # Equation of motion integration of the ECI TBP  

    v_Xout = np.linalg.norm(Xout[3:6, :], axis=0)  # velocity magnitude at each time step
    r_Xout = np.linalg.norm(Xout[0:3, :], axis=0)  # position magnitude at each time step

    diff = np.abs(X[1:6, :] - Xout[1:6, :])        # Pos and vel difference from COE2RV propagation
    rdiff = np.linalg.norm(diff[0:3, :], axis=0)   #pos error magnitude
    vdiff = np.linalg.norm(diff[3:6, :], axis=0)   #vel error magnitude

    
    display_sim_dashboard(coe, P, TA0, X, i)

    #Compute specific energy of the spacecraft at every time step 
    sp_e, sp_e2 = specific_energy(r_Xout, v_Xout, mu=mu, a=coe[0])


    # --- Compute SC. initial conditions in ECEF frame ---
    Omega = np.array([0.0, 0.0, we])
    FI = np.eye(3)
    r_eci = X[0:3, 0]
    v_eci = X[3:6, 0]

    X_ini_ECEF = np.concatenate((FI @ r_eci, FI @ v_eci - np.cross(Omega, r_eci)))
    
    # -------------- 10 Orbits -----------------------------
    t_span = (0, 10 * P)
    t_eval = np.linspace(t_span[0], t_span[1], 10000)
    # integrate the equations of motion of the satellite in ECEF frame
    solECEF = ode113(
        rhs = tbp_ecef,
        y0 = X_ini_ECEF,
        t_span = t_span,
        t_eval = t_eval,
        args = (mu, we)
    )
    X_ECEF = solECEF.y 

    # -------------- PROTOTYPE ATTITUDE --------------
    ''''This section is a work in progress, I intend on implementing attitude control algos to this section.
        For now, it will remain as is, with a hard coded example while it is being developed.'''

    #initial position and velocity in the ECI frame @ t = 1h into orbit 
    r6 = np.array([6768.27, 870.90, 2153.59])
    v6 = np.array([-2.0519, -1.4150, 7.0323])
    alpha = np.deg2rad(30)
    beta = np.deg2rad(20)
    gamma = np.deg2rad(10)

    
    '''This is where the attitude determination algo should go, removing the need for hard coded example.
        Avoid using anything other than quaternions --> no euler angles please, unless using for plots
        
        In GNC systems, the attitude is not propagated indefinitely, regularly corrected by a filter, such as:
        Extended Kalman Filter
        Unscented Kalman Filter
        Each measurement update phase re-normalises the quaternion --> TODO implement sensor observation.
        '''
    
    EulerAng = [alpha, beta, gamma]                     #initial Euler Angles
    
    R_BO = rota121(EulerAng)                            #compute rotation from orbital to body frame 

    EulAx, EulerAng_BO = OrbTo_EulAx(r6, v6, R_BO)      #compute Euler angles and principle axis to rotate from Orbital to Body Frame
    q = EulTo_Quat(EulAx, EulerAng_BO)                  #compute quaternion representation of Euler rotation
    q_BI = QuatTo_DCM(q)
    Q_ini_Att = np.concatenate((q, AngVel))

    # Propagate quaternions
    solAttQ = ode113(
        rhs= q_AttitudeDynamics,
        y0= Q_ini_Att,
        t_span= (t2[0], t2[-1]),
        t_eval= t2,
        args= (Im, )
    )

    Q_Att = solAttQ.y
    Q_Att[0:4, :] = Q_Att[0:4, :] / np.linalg.norm(Q_Att[0:4, :], axis=0)
    X_Att = QuatTo_Eul(Q_Att)                           #convert quaternions to euler angles for visualisation 
                                                        # --> for plotting & kinetic energy + angular momentum


    #------------- ANGULAR MOMENTUM & ROTATIONAL KINETIC ENERGY -------------
    '''Compute angular momentum & rotational kinetic energy over time using quaternion-integrated angular velocities.
    Set stepwise=True to compute it manually at each timestep (useful for debugging or time-varying inertia).
    Set stepwise=False (default) for faster vectorised computation — recommended for constant inertia.'''

    H, Hmag = angularMomentum(Q_Att, Im, stepwise=True)
    T, Tmag = KineticEnergy(Q_Att, Im, stepwise=False)


    # ------------ PLOTS ------------ 
    #plot_orbit_ecef(X_ECEF, Re)
    response = input(f"\nRun plotting scripts? [{green}{Style.BRIGHT} Y{Style.RESET_ALL} / {Fore.RED}{Style.BRIGHT}N{Style.RESET_ALL} ]: ").strip().lower()
    run_plots = response in ["y", "yes", "true", "1"]

    if run_plots:
        #plot True, Mean & Eccentric anomaly against time
        #anomalyPlot(t, E_matrix, TA, MAt)

        #plot_specific_energy(t, sp_e, sp_e2)
        #plot_error_magnitudes(t, vdiff, rdiff)
   
        #plot ECI orbit 
        #plot_orbit_eci(X, Re, mu)

        #plot attitude of spacecraft over time 
        #plot_euler_angles(t2, X_Att)
        #plot_angular_velocity(t2, Q_Att[4:7, :])

        #plot_angular_momentum(t2, Hmag)
        #plot_polhode(H, Hmag, Im, T) TODO 
        plot_groundTrack(X_ECEF, sat_name="ZHUHAI-1 01 (CAS-4A)")



if __name__ == "__main__":
    cli()
