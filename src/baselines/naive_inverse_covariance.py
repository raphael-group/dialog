"""TODO: Add docstring."""

import numpy as np
import pandas as pd


def run_naive_inverse_covariance(
    cnt_mtx_df: pd.DataFrame,
) -> np.ndarray:
    """TODO: Add docstring."""
    covariance_mtx = cnt_mtx_df.cov()
    return np.linalg.inv(covariance_mtx)
