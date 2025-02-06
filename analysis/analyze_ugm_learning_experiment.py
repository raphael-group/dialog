"""TODO: Add docstring."""

from argparse import ArgumentParser
from pathlib import Path

import argcomplete
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


# ------------------------------------------------------------------------------------ #
#                                        MODELS                                        #
# ------------------------------------------------------------------------------------ #
def fit_binarized_count_data_ising_model(
    x_data: np.ndarray,
    genes: list,
    lambda_val: float = 0.01,
) -> tuple:
    """TODO: Add docstring."""
    n_samples, p = x_data.shape
    intercepts = np.zeros(p)
    coefs = np.zeros((p, p))
    for i in range(p):
        y = x_data[:, i]
        x_other = np.delete(x_data, i, axis=1)
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

def fit_bmr_aware_binarized_count_data_ising_model(
    x_data: np.ndarray,
    genes: list,
    expected_background_mutations: dict,
    lambda_val: float = 0.01,
) -> tuple:
    """TODO: Add docstring."""
    n_samples, p = x_data.shape
    total_observed_muts = x_data.sum()

    def _compute_bg_fraction(g: str) -> float:
        raw_bg = expected_background_mutations[g] / max(total_observed_muts, 1e-12)
        return max(min(raw_bg, 1.0 - 1e-6), 1e-6)

    bg_offsets = []
    for g in genes:
        b_g = _compute_bg_fraction(g)
        rho_g = 0.5 * np.log(b_g / (1.0 - b_g))
        bg_offsets.append(rho_g)

    intercepts = np.zeros(p)
    coefs = np.zeros((p, p))

    for i in range(p):
        y = (x_data[:, i] > 0).astype(int)
        x_other = np.delete(x_data, i, axis=1)
        offset_col = np.full((n_samples, 1), bg_offsets[i])

        x_design = np.hstack([x_other, offset_col])
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

    theta = {}
    for i, g in enumerate(genes):
        theta[g] = intercepts[i]

    beta = {}
    for i in range(p):
        for j in range(i + 1, p):
            val = 0.5 * (coefs[i, j] + coefs[j, i])
            beta[(genes[i], genes[j])] = val

    return theta, beta


# ------------------------------------------------------------------------------------ #
#                                   HELPER FUNCTIONS                                   #
# ------------------------------------------------------------------------------------ #
def _build_argument_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument("-c", "--cnt_mtx_fn", required=True, type=Path)
    parser.add_argument("-b", "--bmr_pmf_fn", required=True, type=Path)
    parser.add_argument("-o", "--out_dir", required=True, type=Path)
    parser.add_argument("-k", "--num_genes", default=50, type=int)
    argcomplete.autocomplete(parser)
    return parser


def _calculate_expected_background_mutations(
    num_samples: int,
    bmr_pmf_vals: list,
) -> float:
    return num_samples * np.sum([x * prob_x for x, prob_x in enumerate(bmr_pmf_vals)])


def run_sparse_logistic_regression_ising_model(
    x_data: np.ndarray,
    genes: list,
    out_dir: Path,
) -> None:
    """TODO: Add docstring."""
    theta, beta = fit_binarized_count_data_ising_model(
        x_data,
        genes,
        lambda_val=0.01,
    )

    sorted_theta_values = sorted(
        theta.items(),
        key=lambda x: x[1],
        reverse=True,
    )
    sorted_beta_values = sorted(
        beta.items(),
        key=lambda x: x[1],
        reverse=True,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    df_single = pd.DataFrame(sorted_theta_values, columns=["gene", "theta"])
    df_single.to_csv(out_dir / "ising_single_theta_values.csv", index=False)
    df_pairwise = pd.DataFrame(
        [(g1, g2, val) for ((g1, g2), val) in sorted_beta_values],
        columns=["gene_a", "gene_b", "beta_ij"],
    )
    df_pairwise.to_csv(out_dir / "ising_pairwise_beta_values.csv", index=False)


def run_bmr_aware_binarized_count_data_ising_model(
    x_data: np.ndarray,
    genes: list,
    expected_background_mutations: dict,
    out_dir: Path,
) -> None:
    """TODO: Add docstring."""
    theta, beta = fit_bmr_aware_binarized_count_data_ising_model(
        x_data,
        genes,
        expected_background_mutations,
    )
    sorted_theta_values = sorted(theta.items(), key=lambda x: x[1], reverse=True)
    sorted_beta_values = sorted(beta.items(), key=lambda x: x[1], reverse=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    df_single = pd.DataFrame(sorted_theta_values, columns=["gene", "theta"])
    df_single.to_csv(out_dir / "bmr_ising_single_theta_values.csv", index=False)

    df_pairwise = pd.DataFrame(
        [(g1, g2, val) for ((g1, g2), val) in sorted_beta_values],
        columns=["gene_a", "gene_b", "beta_ij"],
    )
    df_pairwise.to_csv(out_dir / "bmr_ising_pairwise_beta_values.csv", index=False)


# ------------------------------------------------------------------------------------ #
#                                     MAIN FUNCTION                                    #
# ------------------------------------------------------------------------------------ #
def main() -> None:
    """TODO: Add docstring."""
    parser = _build_argument_parser()
    args = parser.parse_args()
    # read and process count matrix data
    cnt_mtx_df = pd.read_csv(args.cnt_mtx_fn, index_col=0)
    highest_mutated_genes = (
        cnt_mtx_df.sum().sort_values(ascending=False).head(args.num_genes)
    )
    genes = list(highest_mutated_genes.index)
    sub_cnt_mtx_df = cnt_mtx_df[genes]
    # read and process background mutation rate data
    bmr_pmf_df = pd.read_csv(args.bmr_pmf_fn, index_col=0)
    gene_to_bmr_pmf_raw = bmr_pmf_df.T.to_dict(orient="list")
    gene_to_bmr_pmf = {
        k: [val for val in pmf_vals if not np.isnan(val)]
        for k, pmf_vals in gene_to_bmr_pmf_raw.items()
    }
    expected_background_mutations = {
        gene: _calculate_expected_background_mutations(
            cnt_mtx_df.shape[0],
            bmr_pmf_vals,
        )
        for gene, bmr_pmf_vals in gene_to_bmr_pmf.items()
    }

    run_sparse_logistic_regression_ising_model(
        (sub_cnt_mtx_df.to_numpy() > 0).astype(int),
        genes,
        args.out_dir,
    )

    run_bmr_aware_binarized_count_data_ising_model(
        sub_cnt_mtx_df.to_numpy(),
        genes,
        expected_background_mutations,
        args.out_dir,
    )


if __name__ == "__main__":
    main()
