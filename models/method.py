"""TODO: Add docstring."""

from pathlib import Path

import numpy as np
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
    out_dir: Path,
    num_iter: int = 1_000,
    num_gibbs_samples: int = 100,
    alpha_learning_rate: float = 1,
    lambda_theta: float = 0.1,
    lambda_beta: float = 0.1,
    momentum: float = 0.9,
) -> None:
    """TODO: Add docstring."""
    num_samples, num_genes = cnt_mtx_df.shape
    thetas = initialize_thetas(cnt_mtx_df, bmr_pmfs)
    betas = initialize_betas(cnt_mtx_df)
    latent_drivers = initialize_latent_drivers(num_samples, num_genes)
    persistent_chain = latent_drivers.copy()

    dout = (
        out_dir / f"NS{num_samples}_NG{num_genes}_NI{num_iter}_NGS{num_gibbs_samples}_"
        f"ALR{alpha_learning_rate}_LT{lambda_theta}_LB{lambda_beta}_M{momentum}"
    )
    dout.mkdir(parents=True, exist_ok=True)

    for it in range(num_iter):
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
            num_gibbs_samples,
        )
        thetas, betas, persistent_chain = update_theta_and_beta_parameters(
            latent_drivers,
            latent_driver_samples,
            thetas,
            betas,
            persistent_chain,
            alpha=alpha_learning_rate,
            lambda_theta=lambda_theta,
            lambda_beta=lambda_beta,
            momentum=momentum,
        )
        checkpoint_path = dout / f"iter_{it}.npy"
        np.save(
            checkpoint_path,
            {
                "thetas": thetas,
                "betas": betas,
                "persistent_chain": persistent_chain,
                "gene_names": cnt_mtx_df.columns.to_numpy(),
            },
        )
