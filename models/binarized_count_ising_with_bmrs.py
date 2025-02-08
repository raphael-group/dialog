"""TODO: Add docstring."""

import numpy as np
from sklearn.linear_model import LogisticRegression


def fit_bmr_aware_binarized_count_data_ising_model(
    x_data: np.ndarray,
    genes: list,
    gene_to_bkgd_factor: dict,
    gene_to_somatic_factor: dict,
    lambda_val: float = 0.1,
) -> tuple:
    """TODO: Add docstring."""
    x_data_transformed = 2 * x_data - 1
    n_samples, p = x_data_transformed.shape

    intercepts = np.zeros(p)
    coefs = np.zeros((p, p))
    offset_coefs_bkgd = np.zeros(p)
    offset_coefs_somatic = np.zeros(p)

    for i, gene in enumerate(genes):
        y = (x_data_transformed[:, i] == 1).astype(int)
        x_other = np.delete(x_data_transformed, i, axis=1)

        val_bkgd = gene_to_bkgd_factor[gene]
        val_somatic = gene_to_somatic_factor[gene]

        bkgd_col = np.full((n_samples, 1), val_bkgd)
        somatic_col = np.full((n_samples, 1), val_somatic)

        x_design = np.hstack([x_other, bkgd_col, somatic_col])

        model = LogisticRegression(
            penalty="l1",
            C=1.0 / lambda_val,
            solver="liblinear",
            fit_intercept=True,
            max_iter=2000,
        )
        model.fit(x_design, y)

        intercepts[i] = model.intercept_[0]
        coef_i = model.coef_[0]

        idx = 0
        for j in range(p):
            if j == i:
                continue
            coefs[i, j] = coef_i[idx]
            idx += 1

        offset_coefs_bkgd[i] = coef_i[idx]
        idx += 1
        offset_coefs_somatic[i] = coef_i[idx]

    theta = {}
    for i, gene in enumerate(genes):
        val_bkgd = gene_to_bkgd_factor[gene]
        val_somatic = gene_to_somatic_factor[gene]
        theta[gene] = (
            intercepts[i]
            + offset_coefs_bkgd[i] * val_bkgd
            + offset_coefs_somatic[i] * val_somatic
        )

    beta = {}
    for i in range(p):
        for j in range(i + 1, p):
            val = 0.5 * (coefs[i, j] + coefs[j, i])
            beta[(genes[i], genes[j])] = val

    return theta, beta
