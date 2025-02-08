"""TODO: Add docstring."""

import numpy as np
from sklearn.linear_model import LogisticRegression


def fit_binarized_count_data_ising_model(
    x_data: np.ndarray,
    genes: list,
    lambda_val: float = 0.01,
) -> tuple:
    """TODO: Add docstring."""
    x_data_transformed = 2 * x_data - 1

    n_samples, p = x_data_transformed.shape
    intercepts = np.zeros(p)
    coefs = np.zeros((p, p))
    for i, _ in enumerate(genes):
        y = (x_data_transformed[:, i] == 1).astype(int)
        x_other = np.delete(x_data_transformed, i, axis=1)
        model = LogisticRegression(
            penalty="l1",
            C=1.0 / lambda_val,
            solver="liblinear",
            fit_intercept=True,
            max_iter=2000,
        )
        model.fit(x_other, y)
        intercepts[i] = model.intercept_[0]
        coef_i = model.coef_[0]
        idx = 0
        for j in range(p):
            if j == i:
                continue
            coefs[i, j] = coef_i[idx]
            idx += 1
    theta = {}
    for i, gene in enumerate(genes):
        theta[gene] = intercepts[i]
    beta = {}
    for i in range(p):
        for j in range(i + 1, p):
            val = 0.5 * (coefs[i, j] + coefs[j, i])
            beta[(genes[i], genes[j])] = val
    return theta, beta
