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
            solver="saga",
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


# ------------------------------------------------------------------------------------ #
#                                     MAIN FUNCTION                                    #
# ------------------------------------------------------------------------------------ #
def main() -> None:
    """TODO: Add docstring."""
    parser = _build_argument_parser()
    args = parser.parse_args()
    cnt_mtx_df = pd.read_csv(args.cnt_mtx_fn, index_col=0)
    highest_mutated_genes = (
        cnt_mtx_df.sum().sort_values(ascending=False).head(args.num_genes)
    )
    genes = list(highest_mutated_genes.index)
    sub_cnt_mtx_df = cnt_mtx_df[genes]
    x_data = (sub_cnt_mtx_df.to_numpy() > 0).astype(int)

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

    args.out_dir.mkdir(parents=True, exist_ok=True)
    df_single = pd.DataFrame(sorted_theta_values, columns=["gene", "theta"])
    df_single.to_csv(args.out_dir / "single_gene_theta_values.csv", index=False)
    df_pairwise = pd.DataFrame(
        [(g1, g2, val) for ((g1, g2), val) in sorted_beta_values],
        columns=["gene_a", "gene_b", "beta_ij"],
    )
    df_pairwise.to_csv(args.out_dir / "pairwise_beta_values.csv", index=False)


if __name__ == "__main__":
    main()
