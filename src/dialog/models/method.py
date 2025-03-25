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
from models.learning import compute_objective, update_theta_and_beta_parameters
from models.structures import GeneBackgroundPMFs


def run_dialog_method(
    cnt_mtx_df: pd.DataFrame,
    bmr_pmfs: GeneBackgroundPMFs,
    out_dir: Path,
    num_iter: int,
    num_gibbs_samples: int,
    learning_rate: float,
    lambda_theta: float,
    lambda_beta: float,
    momentum: float,
) -> None:
    """TODO: Add docstring."""
    rng = np.random.default_rng(seed=42)
    num_samples, num_genes = cnt_mtx_df.shape
    thetas = initialize_thetas(cnt_mtx_df, bmr_pmfs)
    betas = initialize_betas(cnt_mtx_df)
    latent_drivers = initialize_latent_drivers(rng, num_samples, num_genes)
    persistent_chain = latent_drivers.copy()
    objective = np.inf

    dout = (
        out_dir / f"ITER{num_iter}_S{num_gibbs_samples}_LR{learning_rate}"
        f"_LT{lambda_theta}_LB{lambda_beta}_M{momentum}"
    )
    dout.mkdir(parents=True, exist_ok=True)

    for it in range(num_iter):
        checkpoint_path = dout / f"iter_{it}.npy"
        np.save(
            checkpoint_path,
            {
                "thetas": thetas,
                "betas": betas,
                "gene_names": cnt_mtx_df.columns.to_numpy(),
                "objective": objective,
            },
        )
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
            rng,
            num_gibbs_samples,
        )
        thetas, betas, persistent_chain = update_theta_and_beta_parameters(
            latent_drivers,
            latent_driver_samples,
            thetas,
            betas,
            persistent_chain,
            learning_rate=learning_rate,
            lambda_theta=lambda_theta,
            lambda_beta=lambda_beta,
            momentum=momentum,
        )
        objective = compute_objective(
            latent_drivers,
            persistent_chain,
            thetas,
            betas,
            lambda_theta,
            lambda_beta,
        )
