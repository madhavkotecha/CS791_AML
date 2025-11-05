### Add necessary imports ###
import random
import numpy as np
import torch

def seed_everything(seed):
    """Set random seed for all libraries to ensure reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available(): # GPU operation have separate seed
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    random.seed(seed)

def log_marginal_likelihood(x_train, y_train, kernel_func, length_scale, sigma_f, noise=1e-4):
    """
    Compute the log-marginal likelihood.
    x_train: Training inputs (num_train_samples, num_hyperparameters)
    y_train: Training targets (num_train_samples, 1)

    Returns:
    log_likelihood: Scalar log-marginal likelihood value
    """
    K = kernel_func(x_train, x_train, length_scale=length_scale, sigma_f=sigma_f)
    K += noise * np.eye(len(x_train))

    try:
        L = np.linalg.cholesky(K)
    except np.linalg.LinAlgError:
        return -np.inf

    y = y_train.reshape(-1, 1)
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
    log_det_K = 2 * np.sum(np.log(np.diag(L)))
    log_like = -0.5 * np.dot(y.T, alpha) - 0.5 * log_det_K - 0.5 * len(x_train) * np.log(2 * np.pi)
    return log_like.squeeze()

def optimize_hyperparameters(x_train, y_train, kernel_func, noise=1e-4):
    """
    Optimize hyperparameters using grid search.
    x_train: Training inputs (num_train_samples, num_hyperparameters)
    y_train: Training targets (num_train_samples, 1)

    Returns:
    best_length_scale: Optimized length scale
    best_sigma_f: Optimized signal variance
    """
    best_log_like = -np.inf
    best_params = (1.0, 1.0)

    len_scale_range = np.logspace(-2, 2, 5)  #[0.01, 0.1, 1, 10, 100]
    sig_f_range = np.logspace(-2, 2, 5)

    for l in len_scale_range:
        for s in sig_f_range:
            try:
                current_ll = log_marginal_likelihood(x_train, y_train, kernel_func, l, s, noise)
                if current_ll > best_log_like:
                    best_log_like = current_ll
                    best_params = (l, s)
            except Exception:
                continue

    best_length_scale, best_sigma_f = best_params
    return best_length_scale, best_sigma_f

def gaussian_process_predict(x_train, y_train, x_test, kernel_func, length_scale=1.0, sigma_f=1.0, noise=1e-4):
    """
    Perform GP prediction. Return mean and standard deviation of predictions.
    x_train: Training inputs (num_train_samples, num_hyperparameters)
    y_train: Training targets (num_train_samples, 1)
    x_test: Test inputs (num_test_samples, num_hyperparameters)
    kernel_func: Kernel function to use

    Returns:
    mu_s: Predicted means (num_test_samples, 1)
    sigma_s: Predicted standard deviations (num_test_samples, 1)
    """
    K_train = kernel_func(x_train, x_train, length_scale=length_scale, sigma_f=sigma_f) + noise * np.eye(len(x_train))
    K_s = kernel_func(x_train, x_test, length_scale=length_scale, sigma_f=sigma_f)
    K_ss = kernel_func(x_test, x_test, length_scale=length_scale, sigma_f=sigma_f) + 1e-8 * np.eye(len(x_test))

    L = np.linalg.cholesky(K_train)
    y = y_train.reshape(-1, 1)

    v = np.linalg.solve(L, y)
    mean_pred = np.dot(K_s.T, np.linalg.solve(L.T, v))

    temp = np.linalg.solve(L, K_s)
    cov_pred = K_ss - np.dot(temp.T, temp)

    std_pred = np.sqrt(np.clip(np.diag(cov_pred), a_min=1e-9, a_max=None)).reshape(-1, 1)

    return mean_pred, std_pred