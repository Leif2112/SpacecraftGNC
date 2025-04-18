import numpy as np
from gnc.dynamics.orbital import solve_kepler, coe_to_rv, tbp_eci_rhs

mu_earth = 398600.4418  # km^3/s^2

def test_solve_kepler():
    ecc = 0.0008
    M = 2.2172  # Mean anomaly [rad]
    E = solve_kepler(ecc, M)
    M_check = E - ecc * np.sin(E)
    error = abs(M_check - M)
    print(f"Kepler residual error: {error:.2e} rad")
    assert error < 1e-10

def test_coe_to_rv():
    coe = np.array([
        7151.6,             # Semi-major axis [km]
        0.0008,             # Eccentricity
        np.deg2rad(98.39),  # Inclination [rad]
        np.deg2rad(10),     # RAAN [rad]
        np.deg2rad(233),    # Argument of periapsis [rad]
        np.deg2rad(127)     # True anomaly [rad]
    ])
    state = coe_to_rv(coe, mu_earth)
    assert state.shape == (6,)

    r = np.linalg.norm(state[:3])
    v = np.linalg.norm(state[3:])
    energy = 0.5 * v**2 - mu_earth / r
    expected_energy = -mu_earth / (2 * coe[0])
    energy_error = abs(energy - expected_energy)
    print(f"Specific energy error: {energy_error:.4e} km^2/s^2")
    assert energy_error < 1e-4

def test_tbp_eci_rhs():
    state = np.array([7000, 0, 0, 0, 7.5, 0])  # position [km], velocity [km/s]
    deriv = tbp_eci_rhs(0, state, mu_earth)
    assert deriv.shape == (6,)

    r = np.linalg.norm(state[:3])
    expected_acc = mu_earth / r**2
    acc_mag = np.linalg.norm(deriv[3:])
    rel_error = abs(acc_mag - expected_acc) / expected_acc
    print(f"Acceleration magnitude error: {rel_error*100:.2f}%")
    assert rel_error < 0.01  # within 1%
