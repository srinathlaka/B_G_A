import numpy as np
from scipy.stats import t
from scipy.optimize import curve_fit
from utils.growth_models import polynomial_growth, polynomial_func

# Fit growth model
def fit_growth_model(model_type, data_x, data_y):
    if model_type == "Polynomial Growth":
        popt, pcov = curve_fit(polynomial_growth, data_x, data_y)
    elif model_type == "Polynomial Function":
        popt, pcov = curve_fit(polynomial_func, data_x, data_y)
    else:
        raise ValueError("Invalid model type")

    # Calculate standard deviations (errors) for each parameter from the covariance matrix
    if pcov is not None:
        perr = np.sqrt(np.diag(pcov))  # Standard deviations of the parameters
    else:
        perr = None

    return popt, perr, pcov  # Also return pcov to use in CI calculation


# Compute normal confidence intervals for all models

def compute_confidence_intervals(time, params, covariance, alpha, dof, residual_variance, model_type):
    t_critical = t.ppf(1 - alpha / 2, dof)

    # Skip confidence intervals for user-provided models
    if model_type in ['UserProvidedODE', 'UserProvidedFunction']:
        return None, None  # Return None for CIs for user-provided models

    # Initialize the Jacobian matrix
    J = np.zeros((len(time), len(params)))

    # Compute the Jacobian matrix based on the model type
    if model_type == "Exponential":
        J[:, 0] = params[1] * time * np.exp(params[0] * time)  # Derivative w.r.t mu
        J[:, 1] = np.exp(params[0] * time)  # Derivative w.r.t X0

    elif model_type == "Logistic":
        exp_mu_t = np.exp(params[0] * time)
        denom = 1 + (params[1] / params[2]) * (exp_mu_t - 1)
        
        J[:, 0] = (params[1] * time * exp_mu_t) / denom - (params[1] * exp_mu_t * time * exp_mu_t) / denom**2  # Derivative w.r.t mu
        J[:, 1] = exp_mu_t / denom  # Derivative w.r.t X0
        J[:, 2] = -(params[1] * exp_mu_t * (exp_mu_t - 1)) / (params[2]**2 * denom**2)  # Derivative w.r.t K

    elif model_type == "Polynomial Growth":
        J[:, 0] = np.power(time, params[1])  # Derivative w.r.t a
        J[:, 1] = params[0] * np.log(time + 1e-8) * np.power(time, params[1])  # Derivative w.r.t n
        J[:, 2] = 1  # Derivative w.r.t b

    elif model_type == "Baranyi":
        exp_mu_t = np.exp(params[0] * time)
        denom = 1 + params[2]
        
        J[:, 0] = params[1] * params[2] * time * exp_mu_t / denom  # Derivative w.r.t mu
        J[:, 1] = (1 + params[2] * exp_mu_t) / denom  # Derivative w.r.t X0
        J[:, 2] = params[1] * exp_mu_t / denom  # Derivative w.r.t q0

    elif model_type == "Lag-Exponential-Saturation":
        exp_mu_t = np.exp(params[0] * time)
        denom = 1 + params[2] - (params[1] / params[3]) + (params[2] * params[1] / params[3]) * exp_mu_t
        
        J[:, 0] = (params[1] * params[2] * time * exp_mu_t * denom - params[1] * params[2] * (params[1] / params[3]) * time * exp_mu_t * exp_mu_t) / denom**2  # Derivative w.r.t mu
        J[:, 1] = (params[2] * exp_mu_t * denom - (params[1] / params[3]) * denom + (params[2] / params[3]) * exp_mu_t * (params[1] * exp_mu_t)) / denom**2  # Derivative w.r.t X0
        J[:, 2] = (params[1] * exp_mu_t * denom) / denom**2  # Derivative w.r.t q0
        J[:, 3] = -(params[1] * params[2] * exp_mu_t * (params[1] * exp_mu_t - 1)) / (params[3]**2 * denom**2)  # Derivative w.r.t K

    # Calculate confidence intervals based on the Jacobian
    conf_interval = np.zeros(len(time))
    for i in range(len(time)):
        gradient = J[i, :]
        conf_interval[i] = np.sqrt(np.dot(gradient, np.dot(covariance, gradient.T)) + residual_variance)

    # Fitted curve based on model
    fitted_curve = None
    if model_type == "Exponential":
        fitted_curve = params[1] * np.exp(params[0] * time)
    elif model_type == "Logistic":
        fitted_curve = (params[1] * np.exp(params[0] * time)) / (1 + (params[1] / params[2]) * (np.exp(params[0] * time) - 1))
    elif model_type == "Polynomial Growth":
        fitted_curve = params[0] * np.power(time, params[1]) + params[2]
    elif model_type == "Baranyi":
        fitted_curve = params[1] * (1 + params[2] * np.exp(params[0] * time)) / (1 + params[2])
    elif model_type == "Lag-Exponential-Saturation":
        fitted_curve = params[1] * (1 + params[2] * np.exp(params[0] * time)) / (1 + params[2] - (params[1] / params[3]) + (params[2] * params[1] / params[3]) * np.exp(params[0] * time))

    # Calculate the lower and upper bounds
    lower_bound = fitted_curve - t_critical * conf_interval
    upper_bound = fitted_curve + t_critical * conf_interval

    return lower_bound, upper_bound
