"""TODO: Add docstring."""

import numpy as np
import pandas as pd
from dialog.models import GeneBackgroundPMFs


def run_bmr_inverse_covariance(
    cnt_mtx_df: pd.DataFrame,
    bmr_pmfs: GeneBackgroundPMFs,
) -> np.ndarray:
    """TODO: Add docstring."""
    covariance_mtx = cnt_mtx_df.cov()
    bmr_vars = [
        sum(  # E[X^2]
            (k**2) * bmr_pmfs.mapping[gene][k]
            for k in range(len(bmr_pmfs.mapping[gene]))
        )
        - sum(k * bmr_pmfs.mapping[gene][k] for k in range(len(bmr_pmfs.mapping[gene])))
        ** 2  # E[X]^2
        for gene in cnt_mtx_df.columns
    ]
    driver_covariance_mtx = covariance_mtx - np.diag(bmr_vars)
    return np.linalg.inv(driver_covariance_mtx)
