### Add necessary imports ###
import numpy as np

### Implement various kernel functions here ###

def rbf_kernel(x1, x2, length_scale=1.0, sigma_f=1.0):
    """Compute the RBF (Gaussian) kernel."""
    l2_sqr = np.sum((x1[:,None,:] - x2[None,:,:])**2, axis=2)
    k = sigma_f**2 * np.exp(-(l2_sqr/(2*(length_scale**2))))
    return k

def matern_kernel(x1, x2, length_scale=1.0, sigma_f=1.0, nu=1.5):
    """Compute the Matern kernel."""
    l2_dist = np.sqrt(np.sum((x1[:,None,:] - x2[None,:,:])**2, axis=2))
    dist_len = (np.sqrt(3)*l2_dist) / length_scale
    k = sigma_f**2 * (1 + dist_len) * np.exp(-dist_len)
    return k

def rational_quadratic_kernel(x1, x2, length_scale=1.0, sigma_f=1.0, alpha=1.0):
    """Compute the Rational Quadratic kernel."""
    l2_sqr = np.sum((x1[:,None,:] - x2[None,:,:])**2, axis=2)
    k = sigma_f**2 * (1 + (l2_sqr / (2*alpha * (length_scale**2))))**(-alpha)
    return k