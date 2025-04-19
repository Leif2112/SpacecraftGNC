from scipy.integrate import solve_ivp
import numpy as np

def integrate_attitude_ode113_like(rhs, y0, t_span, t_eval, J, rtol=1e-13, atol=1e-14):
    """
    Integrate ODE using a method similar to MATLAB's ode113 (adaptive, high-order).

    Parameters:
        rhs     : callable
            Function f(t, y, J) that returns dy/dt
        y0      : np.ndarray
            Initial state vector
        t_span  : tuple
            Start and end times (t0, tf)
        t_eval  : np.ndarray
            Array of time points to store the solution
        J       : np.ndarray
            Inertia matrix (or other parameters passed to rhs)
        rtol    : float
            Relative tolerance
        atol    : float
            Absolute tolerance

    Returns:
        sol : OdeResult
            Object with .t, .y, .success, etc.
    """
    return solve_ivp(
        rhs,
        t_span,
        y0,
        method='DOP853',  # 8th-order Dormand–Prince, like ode113 in behaviour
        t_eval=t_eval,
        args=(J,),
        rtol=rtol,
        atol=atol
    )
