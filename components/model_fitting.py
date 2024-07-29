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

# Compute normal confidence intervals
def compute_confidence_intervals(time, params, covariance, alpha, dof, residual_variance):
    from scipy.stats import t
    t_critical = t.ppf(1 - alpha / 2, dof)
    J = np.zeros((len(time), len(params)))
    J[:, 0] = np.power(time, params[1])
    J[:, 1] = params[0] * np.log(time + 1e-8) * np.power(time, params[1])  # Adding small constant to avoid log(0)
    J[:, 2] = 1
    conf_interval = np.zeros(len(time))
    fitted_od_growth = polynomial_growth(time, *params)
    for i in range(len(time)):
        gradient = J[i, :]
        conf_interval[i] = np.sqrt(np.dot(gradient, np.dot(covariance, gradient.T)) + residual_variance)
    lower_bound = fitted_od_growth - t_critical * conf_interval
    upper_bound = fitted_od_growth + t_critical * conf_interval
    return lower_bound, upper_bound
