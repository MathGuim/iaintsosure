import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import differential_evolution

from ..utils import lotka_volterra


def objective_fn(params, t_eval, hare_true, lynx_true):
    alpha, beta, gamma, delta, H0, L0 = params
    
    sol = solve_ivp(
        lotka_volterra, 
        t_span=(t_eval[0], t_eval[-1]), 
        y0=[H0, L0], 
        t_eval=t_eval, 
        args=(alpha, beta, gamma, delta),
        method="RK45"
    )
    
    if not sol.success:
        return 1e6
    
    hare_pred, lynx_pred = sol.y
    if len(hare_pred) < len(t_eval): # Handle early solver termination
        return 1e6
        
    hare_res = np.log(np.clip(hare_pred, 1e-2, None)) - np.log(hare_true)
    lynx_res = np.log(np.clip(lynx_pred, 1e-2, None)) - np.log(lynx_true)
    
    return np.mean(hare_res**2 + lynx_res**2)


def fit(years, hare_obs, lynx_obs, bounds):

    result = differential_evolution(
        objective_fn, 
        bounds=bounds, 
        args=(years, hare_obs, lynx_obs),
        seed=42,
        popsize=40,
        polish=True
    )

    return result.x


def predict(theta, y0, t0, tn):

    years_extended = np.arange(t0, tn + 1)
    sol_extended = solve_ivp(
        lotka_volterra, t_span=(t0, tn), y0=y0, 
        t_eval=years_extended, args=theta
    )
    return years_extended, sol_extended.y