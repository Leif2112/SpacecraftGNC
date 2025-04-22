from scipy.integrate import solve_ivp
import numpy as np

def ode113(rhs, y0, t_span, t_eval, args=(), rtol=1e-13, atol=1e-14):
    return solve_ivp(
        rhs,
        t_span,
        y0,
        method='DOP853',
        t_eval=t_eval,
        args=args,
        rtol=rtol,
        atol=atol
    )
