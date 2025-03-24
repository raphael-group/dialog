"""TODO: Add docstring."""

# ------------------------------------------------------------------------------------ #
#                                        IMPORTS                                       #
# ------------------------------------------------------------------------------------ #
import numpy as np
import pandas as pd

from models.structures import GeneBackgroundPMFs

# ------------------------------------------------------------------------------------ #
#                                       CONSTANTS                                      #
# ------------------------------------------------------------------------------------ #
EPSILON = 1e-8


# ------------------------------------------------------------------------------------ #
#                                       FUNCTIONS                                      #
# ------------------------------------------------------------------------------------ #
def initialize_latent_drivers(
    rng: np.random.Generator,
    num_samples: int,
    num_genes: int,
) -> np.ndarray:
    """Randomly initilize binary latent driver indicators for each sample/gene."""
    return rng.integers(0, 2, (num_samples, num_genes))


def initialize_thetas(
    cnt_mtx_df: pd.DataFrame,
    bmr_pmfs: GeneBackgroundPMFs,
) -> np.ndarray:
    """Initialize thetas using the count matrix and background mutation rate PMFs."""
    genes = cnt_mtx_df.columns
    thetas = np.zeros(len(genes))
    for i, gene in enumerate(genes):
        mean_observed_mutations = cnt_mtx_df[gene].mean()
        bmr_pmf = bmr_pmfs.mapping[gene]
        # Compute expected count under background mutation PMF: E[B] = ∑ x * P(B = x)
        exp_background_mutations = sum(k * bmr_pmf[k] for k in range(len(bmr_pmf)))
        # we care when observed > expected (not vice versa) - suggests potential driver
        numerator = max(mean_observed_mutations - exp_background_mutations, 0) + EPSILON
        denominator = exp_background_mutations + EPSILON
        thetas[i] = np.log(numerator / denominator)
    return thetas


def initialize_betas(cnt_mtx_df: pd.DataFrame) -> np.ndarray:
    """Initialize betas using pairwise log-odds ratios from binarized count matrix."""
    bin_cnt_mtx_df = (cnt_mtx_df >= 1).astype(int)
    genes = cnt_mtx_df.columns
    n_genes = len(genes)
    betas = np.zeros((n_genes, n_genes))

    for i in range(n_genes):
        for j in range(i + 1, n_genes):
            cont_table = (
                pd.crosstab(bin_cnt_mtx_df.iloc[:, i], bin_cnt_mtx_df.iloc[:, j])
                .reindex(index=[0, 1], columns=[0, 1], fill_value=0)
                .to_numpy()
            )
            # add epsilon to avoid division by zero
            prob_dist_table = cont_table / cont_table.sum() + EPSILON
            log_odds_ratio = np.log(
                (prob_dist_table[1, 1] * prob_dist_table[0, 0])
                / (prob_dist_table[1, 0] * prob_dist_table[0, 1]),
            )
            # beta matrix is symmetric
            betas[i, j] = log_odds_ratio
            betas[j, i] = log_odds_ratio

    return betas
