import numpy as np

def polynomial_growth(x, a, n, b):
    return a * np.power(x, n) + b

def polynomial_func(t, a, b, c):
    return a * t**2 + b * t + c

def exponential_growth(t, mu, X0):
    return X0 * np.exp(mu * t)

def logistic_growth(t, mu, X0, K):
    return (X0 * np.exp(mu * t)) / (1 + (X0 / K) * (np.exp(mu * t) - 1))

def baranyi_growth(t, mu, X0, q0):
    q_t = q0 * np.exp(mu * t)
    return X0 * (1 + q_t) / (1 + q0)

def lag_exponential_saturation_growth(t, mu, X0, q0, K):
    return X0 * (1 + q0 * np.exp(mu * t)) / (1 + q0 - q0 * (X0 / K) + (q0 * X0 / K) * np.exp(mu * t))
