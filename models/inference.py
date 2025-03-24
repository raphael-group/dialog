"""TODO: Add docstring."""

import numpy as np
import pandas as pd

from models.structures import GeneBackgroundPMFs


def iterated_conditional_modes(
    latent_drivers: np.ndarray,
    cnt_mtx_df: pd.DataFrame,
    bmr_pmfs: GeneBackgroundPMFs,
    thetas: np.array,
    betas: np.array,
    max_iter: int = 100,
) -> np.ndarray:
    """TODO: Add docstring."""
    num_samples, num_genes = cnt_mtx_df.shape
    genes = cnt_mtx_df.columns
    for _ in range(max_iter):
        # TODO @ashuaibi7: replace -1e10 w/ -np.inf
        log_p_c_given_d_0 = np.full((num_samples, num_genes), -1e10) # P(C | D = 0)
        log_p_c_given_d_1 = np.full((num_samples, num_genes), -1e10) # P(C | D = 1)
        for g, gene in enumerate(genes):
            # TODO @ashuaibi7: replace 1e-8 w/ constant
            log_bmr = np.log(np.asarray(bmr_pmfs.mapping[gene]) + 1e-8) # log P(B)
            mutation_counts = cnt_mtx_df[gene].to_numpy()
            # given D, mutation count c or c - 1 must be contained in bmr_pmf
            valid_d_0 = (mutation_counts >= 0) & (mutation_counts < len(log_bmr))
            valid_d_1 = (mutation_counts >= 1) & (mutation_counts - 1 < len(log_bmr))
            # update P(C | D = 0) and P(C | D = 1) for counts w/ valid D
            log_p_c_given_d_0[valid_d_0, g] = log_bmr[mutation_counts[valid_d_0]]
            log_p_c_given_d_1[valid_d_1, g] = log_bmr[mutation_counts[valid_d_1]]

        np.fill_diagonal(betas, 0) # exclude self interaction g = h
        new_latent_drivers = ((log_p_c_given_d_0) < ( # log P(C | D = 0)
            log_p_c_given_d_1 # log P(C | D = 1)
            + thetas.reshape(1, -1) # θ_g
            + latent_drivers @ betas  # Σ β_gh * D_h
        )).astype(int)
        if np.array_equal(new_latent_drivers, latent_drivers):
            break
        latent_drivers = new_latent_drivers
    return latent_drivers

def gibbs_sampling(
    latent_drivers: np.ndarray,
    cnt_mtx_df: pd.DataFrame,
    gene_to_bmr_pmf: GeneBackgroundPMFs,
    thetas: np.ndarray,
    betas: np.ndarray,
    rng: np.random.Generator,
    num_iter: int = 100,
) -> list:
    """TODO: Add docstring."""
    num_samples, num_genes = cnt_mtx_df.shape
    genes = cnt_mtx_df.columns
    samples = []
    for _ in range(num_iter):
        for g, gene in enumerate(genes):
            log_bmr = np.log(np.asarray(gene_to_bmr_pmf.mapping[gene]) + 1e-8)
            mutation_counts = cnt_mtx_df[gene].to_numpy()
            # given D, mutation count c or c - 1 must be contained in bmr_pmf
            valid_d_0 = (mutation_counts >= 0) & (mutation_counts < len(log_bmr))
            valid_d_1 = (mutation_counts >= 1) & (mutation_counts - 1 < len(log_bmr))
            log_p_c_given_d_0 = np.zeros(num_samples)
            log_p_c_given_d_0[valid_d_0] = log_bmr[mutation_counts[valid_d_0]]
            log_p_c_given_d_1 = np.zeros(num_samples)
            log_p_c_given_d_1[valid_d_1] = log_bmr[mutation_counts[valid_d_1]]

            exp_d_0 = np.exp(log_p_c_given_d_0) # log P(C | D = 0)
            betas[g, g] = 0 # exclude self interaction g = h
            exp_d_1 = np.exp(log_p_c_given_d_1 # log P(C | D = 1)
                             + thetas[g] # θ_g
                             + latent_drivers @ betas[g, :]) # Σ β_gh * D_h
            prob_update = exp_d_1 / (exp_d_0 + exp_d_1)
            rand_vals = rng.random(num_samples) # [0, 1] to determine D_g update
            latent_drivers[:, g] = (rand_vals < prob_update).astype(int)
        samples.append(latent_drivers.copy())
    return samples
