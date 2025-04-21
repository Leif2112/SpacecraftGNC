import numpy as np
import time
from colorama import init, Fore, Style
from colorama.ansi import AnsiFore



from gnc.attitude.rigid_body import attitude_rhs, axis_angle_to_quaternion
from gnc.integrators.ode113_like import ode113
from gnc.dynamics.orbital import solve_kepler, coe_to_rv, tbp_eci

from gnc.visualisation.plot_specific_energy import plot_specific_energy
from gnc.visualisation.plot_posVelError import plot_error_magnitudes
from gnc.visualisation.plot_anomalyVStime import anomalyPlot
from gnc.visualisation.plot_orbital import plot_orbit_eci


# Initialize colorama once
init(autoreset=True)

def rgb(r, g, b):
    return f'\033[38;2;{r};{g};{b}m'

cyan = rgb(120, 220, 232)     # ~ "#78DCE8"
pink = rgb(255, 97, 136)      # ~ "#FF6188"
reset = Style.RESET_ALL

######################### ORBITAL ELEMENTS #########################

coe = np.array([
    7151.6,              # semi-major axis [km]
    0.0008,              # eccentricity
    np.deg2rad(98.39),   # inclination
    np.deg2rad(10),      # RAAN
    np.deg2rad(233),     # argument of periapsis
    np.deg2rad(127.0)    # true anomaly
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


tol = 10e-10         


#####################################################################

def run(): 


    #Solve Kepler's equation
    
    # --- TIME KEEPER START ---
    #start = time.perf_counter()
    #end = time.perf_counter()
    #print("⏱️  [Performance]")
    #print(f"  Execution time         : {end - start:.6f} seconds\n")
    # --- TIME KEEPER END ---
        #compute orbital period of spacecraft
    n = np.sqrt(mu / coe[0]**3)                      # mean motion [rad/s]
    P = 2 * np.pi / n                                # orbital period [s]

    print(f"\n{Fore.GREEN}{Style.BRIGHT}[Orbit Parameters]")
    print(f"    Semi-major axis       (a)  : {coe[0]:.4f} [km]")
    print(f"    Eccentricity          (e)  : {coe[1]:.4f}")
    print(f"    Inclination           (i)  : {coe[2]:.4f} [rad]")
    print(f"    RAAN                  (Ω)  : {coe[3]:.4f} [rad]")
    print(f"    Argument of Periapsis (ω)  : {coe[4]:.4f} [rad]")
    print(f"    True Anomaly          (M₀) : {coe[5]:.4f}")  
    print(f"    Orbital Period        (P)  : {P:.4f} [s]\n")
    

    #create time vector for plotting
    t0 = 0
    t = np.linspace(t0, P + t0, 1000)   # time vector [s]


    
    print(f"\n{Fore.GREEN}{Style.BRIGHT}[Kepler Solver]")
    
    E0, i = solve_kepler(coe[1], coe[5], tol=tol)  # eccentric anomaly at t0 [rad]
    
       
    print(f"    Converged after        : {i} iterations")
    print(f"    Eccentric Anomaly (E₀) : {E0:.4f} rad")


    
    # input("Press Enter to continue...\n")
    

    #Compute initial Ture Anomaly
    TA0 = 2 * np.arctan(np.sqrt((1 + coe[1]) / (1 - coe[1])) * np.tan(E0 / 2))  # true anomaly at t0 [rad]
    print(f"    True Anomaly (ν₀)      : {TA0:.4f} rad")


    position_label = "Apoapsis" if np.pi/2 < TA0 < 1.5 * np.pi else "Periapsis"
    print(f"    Satellite is near      : {position_label}\n")




    MAt = np.mod(coe[5] + n * (t - t0), 2 * np.pi)  # mean anomaly [rad]


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

    sol = ode113(
        rhs = tbp_eci,
        y0 = X[:6, 0],
        t_span = (t[0], t[-1]),
        t_eval = t,
        mu = mu
    )

    Xout = sol.y         # solution state vector at each time step 

    print(f"{cyan}{Style.BRIGHT}[Initial State Vector]{reset}")
    print("   ", np.array2string(Xout[:, 0], precision=6, suppress_small=True))

    print(f"\n{pink}{Style.BRIGHT}[Final State Vector]")
    print("   ", np.array2string(Xout[:, -1], precision=6, suppress_small=True))

    v_Xout = np.linalg.norm(Xout[3:6, :], axis=0)  # velocity magnitude at each time step
    r_Xout = np.linalg.norm(Xout[0:3, :], axis=0)  # position magnitude at each time step

    diff = np.abs(X[1:6, :] - Xout[1:6, :])        # Pos and vel difference from COE2RV propagation
    rdiff = np.linalg.norm(diff[0:3, :], axis=0)   #pos error magnitude
    vdiff = np.linalg.norm(diff[3:6, :], axis=0)   #vel error magnitude

    

    # Integrate equation of motion

    
    '''
    # Inertia matrix [kg·m²] (example spacecraft)
    J = np.diag([2500.0, 5000.0, 6500.0])

    # Initial angular velocity [rad/s]
    omega0 = np.array([0.001, 0.002, 0.003])

    # Initial quaternion (identity rotation)
    axis = np.array([0, 0, 1])
    angle = 0.0
    q0 = axis_angle_to_quaternion(axis, angle)

    # Combined initial state: [q0, q1, q2, q3, wx, wy, wz]
    y0 = np.concatenate((q0, omega0))

    # Simulation time parameters
    t0, tf = 0.0, 3600.0  # seconds
    t_eval = np.linspace(t0, tf, 1000)

    # Integrate
    sol = integrate_attitude_ode113_like(
        rhs=attitude_rhs,
        y0=y0,
        t_span=(t0, tf),
        t_eval=t_eval,
        J=J
    )

    # Extract results
    q = sol.y[:4, :]
    omega = sol.y[4:, :]

    # Compute quaternion norm error and angular momentum over time
    q_norm_error = np.abs(np.linalg.norm(q, axis=0) - 1.0)
    H = J @ omega
    H_mag = np.linalg.norm(H, axis=0)
    T = 0.5 * np.einsum('ij,ij->j', omega, H) '''

    """    # Plot quaternion norm error
    plt.figure(figsize=(10, 4))
    plt.plot(sol.t, q_norm_error)
    plt.xlabel("Time [s]")
    plt.ylabel("|norm(q) - 1|")
    plt.title("Quaternion Norm Drift")
    plt.grid(True)
    plt.tight_layout()

    # Plot angular momentum magnitude
    plt.figure(figsize=(10, 4))
    plt.plot(sol.t, H_mag)
    plt.xlabel("Time [s]")
    plt.ylabel("|H| [kg·m²/s]")
    plt.title("Angular Momentum Magnitude")
    plt.grid(True)
    plt.tight_layout()

    # Plot rotational kinetic energy
    plt.figure(figsize=(10, 4))
    plt.plot(sol.t, T)
    plt.xlabel("Time [s]")
    plt.ylabel("Kinetic Energy [J]")
    plt.title("Rotational Kinetic Energy")
    plt.grid(True)
    plt.tight_layout()  """

    
    #plot True, Mean & Eccentric anomaly against time
    #anomalyPlot(t, E_matrix, TA, MAt)

    #plot_specific_energy(coe, mu)
    plot_error_magnitudes(t, vdiff, rdiff)

    #plot ECI orbit 
    #plot_orbit_eci(X, Re, mu)
    

if __name__ == "__main__":
    run()
