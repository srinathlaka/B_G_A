import numpy as np

def calculate_metrics(observed, predicted, num_params):
    residuals = observed - predicted
    rss = np.sum(residuals**2)
    r_squared = 1 - (rss / np.sum((observed - np.mean(observed))**2))
    aic = 2 * num_params + len(observed) * np.log(rss / len(observed))
    return rss, r_squared, aic
