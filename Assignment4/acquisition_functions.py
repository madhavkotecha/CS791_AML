### Add necessary imports ###
import numpy as np
import math

# Implement acquisition functions here

def expected_improvement(mu, sigma, f_best, xi=0.01):
    """
    Compute the Expected Improvement acquisition function.
    mu: Predicted means (num_test_samples, 1)
    sigma: Predicted standard deviations (num_test_samples, 1)
    f_best: Best observed function value
    
    Returns:
    ei: Expected Improvement values (num_test_samples, 1)
    """
    erf_vec = np.vectorize(math.erf)

    mu, sigma = np.atleast_1d(mu), np.atleast_1d(sigma)
    imp = mu - f_best - xi
    with np.errstate(divide='warn'):
        z = np.divide(imp, sigma, out=np.zeros_like(imp), where=sigma > 0)
    # cdf = 1/(1 + np.exp(-1.702 * z))
    cdf_z = 0.5 * (1.0 + erf_vec(z / np.sqrt(2.0)))
    pdf_z = np.exp(-0.5 * z**2) / np.sqrt(2 * np.pi)
    ei = imp * cdf_z + sigma * pdf_z
    ei[sigma <= 0.0] = 0.0
    return ei

def probability_of_improvement(mu, sigma, f_best, xi=0.01):
    """
    Compute the Probability of Improvement acquisition function.
    mu: Predicted means (num_test_samples, 1)
    sigma: Predicted standard deviations (num_test_samples, 1)
    f_best: Best observed function value
    xi: Exploration-exploitation trade-off parameter

    Returns:
    pi: Probability of Improvement values (num_test_samples, 1)
    """
    erf_vec = np.vectorize(math.erf)
    
    mu, sigma = np.atleast_1d(mu), np.atleast_1d(sigma)
    with np.errstate(divide='warn'):
        z = np.divide(mu - f_best - xi, sigma, out=np.zeros_like(mu), where=sigma > 0)
    # return 1/(1 + np.exp(-1.702 * z))
    return 0.5 * (1.0 + erf_vec(z / np.sqrt(2.0)))