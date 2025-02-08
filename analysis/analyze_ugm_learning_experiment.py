"""TODO: Add docstring."""

from argparse import ArgumentParser
from pathlib import Path

import argcomplete
import numpy as np
import pandas as pd
import torch
from torch import optim

from models.binarized_count_ising import fit_binarized_count_data_ising_model
from models.bmr_aware_binarized_count_ising import (
    fit_bmr_aware_binarized_count_data_ising_model,
)


# ------------------------------------------------------------------------------------ #
#                                        MODELS                                        #
# ------------------------------------------------------------------------------------ #
def run_mean_field_variational_inference(
    count_data: np.ndarray,
    gene_to_bmr_pmf: dict,
    num_iterations: int = 50,
    lr: float = 1e-2,
) -> tuple:
    """TODO: Add docstring."""
    n_samples, p = count_data.shape
    m_param = torch.zeros(n_samples, p, requires_grad=True)
    theta_param = torch.zeros(p, requires_grad=True)
    beta_param = torch.zeros(p, p, requires_grad=True)
    optimizer = optim.Adam([m_param, theta_param, beta_param], lr=lr)

    def log_prob_counts(x: float, mean_spin: float, bmr_probs: torch.tensor) -> float:
        return -((1 + mean_spin) / 2.0 * (x - bmr_probs.mean()) ** 2).sum()

    def free_energy() -> float:
        m = torch.tanh(m_param)
        energy = 0.0
        energy += (m * theta_param).sum()
        for n in range(n_samples):
            mn = m[n]
            energy += torch.triu(beta_param, diagonal=1).mul(torch.ger(mn, mn)).sum()
        entropy = 0.0
        eps = 1e-7
        m_clamped = torch.clamp(m, -1 + eps, 1 - eps)
        p_plus = 0.5 * (1.0 + m_clamped)
        p_minus = 0.5 * (1.0 - m_clamped)
        entropy -= (p_plus * torch.log(p_plus) + p_minus * torch.log(p_minus)).sum()
        count_likelihood = 0.0
        for i in range(p):
            bmr_probs = torch.tensor(gene_to_bmr_pmf.get(i)).float()
            for n in range(n_samples):
                count_likelihood += log_prob_counts(
                    count_data[n, i],
                    m[n, i],
                    bmr_probs,
                )
        return -(energy + entropy + count_likelihood)

    for _i in range(num_iterations):
        optimizer.zero_grad()
        f = free_energy()
        f.backward()
        optimizer.step()

    with torch.no_grad():
        m_final = torch.tanh(m_param).detach().cpu().numpy()
        theta_final = theta_param.detach().cpu().numpy()
        beta_final = 0.5 * (beta_param + beta_param.T)
        beta_final = beta_final.detach().cpu().numpy()
    return m_final, theta_final, beta_final


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


def _calculate_prob_nonzero_background_mutations(
    bmr_pmf_vals: list,
) -> float:
    return 1 - bmr_pmf_vals[0]


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
    gene_to_exp_sample_bkgd_mutations: dict,
    gene_to_somatic_factor: dict,
    out_dir: Path,
) -> None:
    """TODO: Add docstring."""
    theta, beta = fit_bmr_aware_binarized_count_data_ising_model(
        x_data,
        genes,
        gene_to_exp_sample_bkgd_mutations,
        gene_to_somatic_factor,
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


def create_obs_exp_mutation_table(
    gene_to_bmr_pmf: dict,
    cnt_mtx_df: pd.DataFrame,
) -> pd.DataFrame:
    """TODO: Add docstring."""
    genes = cnt_mtx_df.columns
    expected_mutations = {
        gene: _calculate_expected_background_mutations(
            cnt_mtx_df.shape[0],
            gene_to_bmr_pmf[gene],
        )
        for gene in genes
    }
    observed_mutations = cnt_mtx_df.sum().to_dict()

    result_df = pd.DataFrame(
        {
            "gene": genes,
            "observed": [observed_mutations[gene] for gene in genes],
            "expected": [expected_mutations[gene] for gene in genes],
            "obs_minus_exp": [
                observed_mutations[gene] - expected_mutations[gene] for gene in genes
            ],
        },
    )

    return result_df.sort_values(by="obs_minus_exp", ascending=False)


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
    # create observed vs. expected mutation table
    obs_exp_mut_table = create_obs_exp_mutation_table(gene_to_bmr_pmf, sub_cnt_mtx_df)
    obs_exp_mut_table.to_csv(args.out_dir / "obs_exp_mutation_table.csv", index=False)

    # run mean field variational inference
    arr_counts = sub_cnt_mtx_df.to_numpy()
    m_final, theta_final, beta_final = run_mean_field_variational_inference(
        arr_counts,
        {i: gene_to_bmr_pmf.get(g) for i, g in enumerate(genes)},
    )
    df_theta = pd.DataFrame(
        zip(genes, theta_final),
        columns=["gene", "theta"],
    ).sort_values(by="theta", ascending=True)
    df_theta.to_csv(args.out_dir / "mf_single_theta_values.csv", index=False)
    df_beta = pd.DataFrame(
        [
            (gene_a, gene_b, beta_final[i, j])
            for i, gene_a in enumerate(genes)
            for j, gene_b in enumerate(genes[i + 1 :], start=i + 1)
        ],
        columns=["gene_a", "gene_b", "beta_ij"],
    ).sort_values(by="beta_ij", ascending=True)
    df_beta.to_csv(args.out_dir / "mf_pairwise_beta_values.csv", index=False)


if __name__ == "__main__":
    main()
