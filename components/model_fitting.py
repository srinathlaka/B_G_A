import numpy as np
from scipy.optimize import curve_fit
from utils.growth_models import polynomial_growth, polynomial_func

# Fit growth model
def fit_growth_model(model_type, data_x, data_y):
    if model_type == "Polynomial Growth":
        popt, pcov = curve_fit(polynomial_growth, data_x, data_y)
        return popt, pcov
    elif model_type == "Polynomial Function":
        popt, pcov = curve_fit(polynomial_func, data_x, data_y)
        return popt, pcov
    else:
        raise ValueError("Invalid model type")

# Compute normal confidence intervals for both models
def compute_confidence_intervals(time, params, covariance, alpha, dof, residual_variance, model_type):
    from scipy.stats import t
    t_critical = t.ppf(1 - alpha / 2, dof)
    
    # Initialize the Jacobian matrix based on model type
    J = np.zeros((len(time), len(params)))
    
    if model_type == "Polynomial Growth":
        # Jacobian for polynomial growth
        J[:, 0] = np.power(time, params[1])
        J[:, 1] = params[0] * np.log(time + 1e-8) * np.power(time, params[1])  # Log to avoid log(0)
        J[:, 2] = 1
        fitted_curve = polynomial_growth(time, *params)
        
    elif model_type == "Polynomial Function":
        # Jacobian for polynomial function (quadratic)
        J[:, 0] = time**2  # Partial derivative w.r.t `a`
        J[:, 1] = time     # Partial derivative w.r.t `b`
        J[:, 2] = 1        # Partial derivative w.r.t `c`
        fitted_curve = polynomial_func(time, *params)
    
    # Compute the confidence intervals
    conf_interval = np.zeros(len(time))
    for i in range(len(time)):
        gradient = J[i, :]
        conf_interval[i] = np.sqrt(np.dot(gradient, np.dot(covariance, gradient.T)) + residual_variance)
    
    lower_bound = fitted_curve - t_critical * conf_interval
    upper_bound = fitted_curve + t_critical * conf_interval
    return lower_bound, upper_bound
