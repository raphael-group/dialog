"""TODO: Add docstring."""

import pandas as pd

from models.inference import gibbs_sampling, iterated_conditional_modes
from models.initialization import (
    initialize_betas,
    initialize_latent_drivers,
    initialize_thetas,
)
from models.learning import update_theta_and_beta_parameters
from models.structures import GeneBackgroundPMFs


def run_dialog_method(
    cnt_mtx_df: pd.DataFrame,
    bmr_pmfs: GeneBackgroundPMFs,
    num_iter: int = 1_000,
) -> None:
    """TODO: Add docstring."""
    num_samples, num_genes = cnt_mtx_df.shape
    thetas = initialize_thetas(cnt_mtx_df, bmr_pmfs)
    betas = initialize_betas(cnt_mtx_df)
    latent_drivers = initialize_latent_drivers(num_samples, num_genes)
    persistent_chain = latent_drivers.copy()

    for _ in range(num_iter):
        latent_drivers = iterated_conditional_modes(
            latent_drivers,
            cnt_mtx_df,
            bmr_pmfs,
            thetas,
            betas,
        )
        latent_driver_samples = gibbs_sampling(
            latent_drivers,
            cnt_mtx_df,
            bmr_pmfs,
            thetas,
            betas,
        )
        thetas, betas, persistent_chain = update_theta_and_beta_parameters(
            latent_drivers,
            latent_driver_samples,
            thetas,
            betas,
            persistent_chain,
        )
        # save posterior/thetas/betas/latent_drivers/persistent_chain in checkpoint
