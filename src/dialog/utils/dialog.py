"""TODO: Add docstring."""

import json
from argparse import ArgumentParser
from itertools import combinations
from pathlib import Path

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve

# ------------------------------------------------------------------------------------ #
#                                       CONSTANTS                                      #
# ------------------------------------------------------------------------------------ #
font_path = "/u/ashuaibi/.fonts/cmuserif.ttf"
font_prop = fm.FontProperties(fname=font_path)
fm.fontManager.addfont(font_path)
FONT_FAMILY = font_prop.get_name()
plt.rcParams["font.family"] = FONT_FAMILY
FONT_SCALE = 1.5


# ---------------------------------------------------------------------------- #
#                              PLOTTING FUNCTIONS                              #
# ---------------------------------------------------------------------------- #
def style_plot(xlabel: str, ylabel: str, font_scale: float = FONT_SCALE) -> None:
    """TODO: Add docstring."""
    plt.xlabel(xlabel, fontsize=font_scale * 10)
    plt.ylabel(ylabel, fontsize=font_scale * 10)
    plt.xticks(fontsize=font_scale * 8)
    plt.yticks(fontsize=font_scale * 8)

    plt.gca().tick_params(
        axis="both",
        direction="in",
        length=font_scale * 4,
        width=font_scale,
    )
    plt.minorticks_on()

    plt.legend(fontsize=font_scale * 6, frameon=False)

    plt.tight_layout()


# ------------------------------------------------------------------------------------ #
#                                   HELPER FUNCTIONS                                   #
# ------------------------------------------------------------------------------------ #
def _build_argument_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument("-d", "--dir", required=True, type=Path)
    parser.add_argument("-at", "--analysis_type", required=True, choices=["me", "co"])
    parser.add_argument("-k", "--num_genes", default=200, type=int)
    return parser


def initialize_thetas(cnt_mtx_df: pd.DataFrame, gene_to_bmr_pmf: dict) -> np.ndarray:
    """TODO: Add docstring."""
    eps = 1e-6
    genes = cnt_mtx_df.columns
    thetas = np.zeros(len(genes))
    for i, g in enumerate(genes):
        ecg = cnt_mtx_df[g].mean()
        pmf = gene_to_bmr_pmf[g]
        ebg = sum(k * pmf[k] for k in range(len(pmf)))
        num = max(ecg - ebg, 0) + eps
        den = ebg + eps
        thetas[i] = np.log(num / den)
    return thetas


def initialize_betas(cnt_mtx_df: pd.DataFrame) -> np.ndarray:
    """TODO: Add docstring."""
    genes = cnt_mtx_df.columns
    num_genes = len(genes)
    betas = np.zeros((num_genes, num_genes))
    eps = 1e-6
    for i in range(num_genes):
        g = genes[i]
        mask_g = cnt_mtx_df[g] >= 1
        for j in range(i + 1, num_genes):
            h = genes[j]
            mask_h = cnt_mtx_df[h] >= 1
            p_11 = np.mean(mask_g & mask_h) + eps
            p_10 = np.mean(mask_g & ~mask_h) + eps
            p_01 = np.mean(~mask_g & mask_h) + eps
            p_00 = np.mean(~mask_g & ~mask_h) + eps
            beta_ij = np.log((p_11 * p_00) / (p_10 * p_01))
            betas[i, j] = beta_ij
            betas[j, i] = beta_ij
    return betas


def save_results(
    genes: list,
    thetas: np.ndarray,
    betas: np.ndarray,
    out_dir: Path,
    theta_filename: str,
    beta_filename: str,
    theta_col: str = "theta",
    beta_col: str = "beta",
) -> None:
    """TODO: Add docstring."""
    theta_df = pd.DataFrame({"gene": genes, theta_col: thetas})
    beta_df = pd.DataFrame(
        [
            {"gene_a": genes[i], "gene_b": genes[j], beta_col: betas[i, j]}
            for i, j in combinations(range(betas.shape[0]), 2)
        ],
    )
    theta_df.sort_values(by=theta_col).to_csv(out_dir / theta_filename, index=False)
    beta_df.sort_values(by=beta_col).to_csv(out_dir / beta_filename, index=False)
    return theta_df, beta_df


def run_standard_methods(
    cnt_mtx_df: pd.DataFrame,
    gene_to_bmr_pmf: dict,
    out_dir: Path,
) -> tuple:
    """TODO: Add docstring."""
    # ---------------------------------------------------------------------------- #
    #                   GAUSSIAN PRECISION MATRIX FROM COUNT DATA                  #
    # ---------------------------------------------------------------------------- #
    cov_mtx = cnt_mtx_df.cov()
    precision_mtx = np.linalg.inv(cnt_mtx_df.cov())
    precision_mtx_df = pd.DataFrame(
        precision_mtx,
        index=cnt_mtx_df.columns,
        columns=cnt_mtx_df.columns,
    )
    interaction_to_precision = {
        (gene_a, gene_b): precision_mtx_df.loc[gene_a, gene_b]
        for gene_a, gene_b in combinations(cnt_mtx_df.columns, 2)
    }
    precision_df = pd.DataFrame(
        {
            "gene_a": [interaction[0] for interaction in interaction_to_precision],
            "gene_b": [interaction[1] for interaction in interaction_to_precision],
            "precision": list(interaction_to_precision.values()),
        },
    )

    # ---------------------------------------------------------------------------- #
    #               GAUSSIAN PRECISION MATRIX FROM COUNT AND BMR DATA              #
    # ---------------------------------------------------------------------------- #
    diag_vals = np.diag(cov_mtx)
    gene_vars = []
    for i, g in enumerate(cnt_mtx_df.columns):
        p = gene_to_bmr_pmf[g]
        e = sum(k * p[k] for k in range(len(p)))
        e2 = sum((k**2) * p[k] for k in range(len(p)))
        passenger_var = e2 - e**2
        passenger_var_clamped = min(passenger_var, 0.99 * diag_vals[i])
        gene_vars.append(passenger_var_clamped)
    sigma_d = cov_mtx - np.diag(gene_vars)

    precision_bmr_mtx = np.linalg.inv(sigma_d)
    precision_bmr_mtx_df = pd.DataFrame(
        precision_bmr_mtx,
        index=cnt_mtx_df.columns,
        columns=cnt_mtx_df.columns,
    )
    interaction_to_precision_bmr = {
        (a, b): precision_bmr_mtx_df.loc[a, b]
        for a, b in combinations(cnt_mtx_df.columns, 2)
    }
    precision_bmr_df = pd.DataFrame(
        {
            "gene_a": [i[0] for i in interaction_to_precision_bmr],
            "gene_b": [i[1] for i in interaction_to_precision_bmr],
            "precision_bmr": list(interaction_to_precision_bmr.values()),
        },
    )

    # ---------------------------------------------------------------------------- #
    #                         ESTIMATE FROM BINARIZED DATA                         #
    # ---------------------------------------------------------------------------- #
    theta_naughts = initialize_thetas(cnt_mtx_df, gene_to_bmr_pmf)
    beta_naughts = initialize_betas(cnt_mtx_df)
    theta_naught_df, beta_naught_df = save_results(
        genes=cnt_mtx_df.columns.to_list(),
        thetas=theta_naughts,
        betas=beta_naughts,
        out_dir=out_dir,
        theta_filename="theta_naughts.csv",
        beta_filename="beta_naughts.csv",
        theta_col="theta_naught",
        beta_col="beta_naught",
    )

    return precision_df, precision_bmr_df, theta_naught_df, beta_naught_df


def gibbs_sampling(
    d_mtx: np.ndarray,
    cnt_mtx_df: pd.DataFrame,
    gene_to_bmr_pmf: dict,
    thetas: np.ndarray,
    betas: np.ndarray,
    rng: np.random.Generator,
    n_iter: int = 100,
) -> list:
    """TODO: Add docstring."""
    num_samples, num_genes = d_mtx.shape
    d_mtx_samples = []
    c_val_mtx = cnt_mtx_df.to_numpy()
    genes = cnt_mtx_df.columns

    for _ in range(n_iter):
        for g_idx, gene in enumerate(genes):
            # Get PMF as a numpy array for vectorized indexing.
            pmf_array = np.array(gene_to_bmr_pmf[gene])
            max_idx = len(pmf_array)
            # Get the count values for gene g across all samples.
            c_vals = c_val_mtx[:, g_idx]
            # p0: PMF value for count c, with invalid indices set to 0.
            valid0 = (c_vals >= 0) & (c_vals < max_idx)
            p0 = np.zeros(num_samples)
            p0[valid0] = pmf_array[c_vals[valid0].astype(int)]
            # p1: PMF value for count (c-1), with invalid indices set to 0.
            c_minus_1 = c_vals - 1
            valid1 = (c_minus_1 >= 0) & (c_minus_1 < max_idx)
            p1 = np.zeros(num_samples)
            p1[valid1] = pmf_array[c_minus_1[valid1].astype(int)]
            # Compute log probabilities.
            log_p0 = np.log(p0 + 1e-12)
            log_p1 = np.log(p1 + 1e-12) + thetas[g_idx]
            # Compute contribution from neighboring genes:
            # Dot product of current state with the g-th row of betas.
            contribution = d_mtx @ betas[g_idx, :]
            # Exclude self-interaction if needed.
            contribution -= betas[g_idx, g_idx] * d_mtx[:, g_idx]
            log_p1 += contribution
            # Compute the probability that D=1.
            e0 = np.exp(log_p0)
            e1 = np.exp(log_p1)
            prob = e1 / (e0 + e1)
            # Update the state for gene g in all samples.
            rand_vals = rng.random(num_samples)
            d_mtx[:, g_idx] = (rand_vals < prob).astype(int)
        d_mtx_samples.append(d_mtx.copy())
    return d_mtx_samples


def run_icm(
    d_mtx: np.ndarray,
    cnt_mtx_df: pd.DataFrame,
    gene_to_bmr_pmf: dict,
    thetas: np.ndarray,
    betas: np.ndarray,
    max_iter: int = 10,
) -> np.ndarray:
    """TODO: Add docstring."""
    num_samples, num_genes = d_mtx.shape
    genes = cnt_mtx_df.columns
    c_val_mtx = cnt_mtx_df.to_numpy()
    for _ in range(max_iter):
        log_p_c_if_d0_mtx = np.full((num_samples, num_genes), -1e10)
        log_p_c_if_d1_mtx = np.full((num_samples, num_genes), -1e10)
        for g_idx, gene in enumerate(genes):
            pmf = gene_to_bmr_pmf[gene]
            log_pmf = np.log(np.asarray(pmf) + 1e-12)
            max_k = len(pmf)
            vals = c_val_mtx[:, g_idx]
            valid_0 = (vals >= 0) & (vals < max_k)
            valid_1 = (vals - 1 >= 0) & (vals - 1 < max_k)
            log_p_c_if_d0_mtx[valid_0, g_idx] = log_pmf[vals[valid_0]]
            log_p_c_if_d1_mtx[valid_1, g_idx] = log_pmf[vals[valid_1]]
        beta_sum_mtx = d_mtx @ betas.T
        lhs = log_p_c_if_d0_mtx
        rhs = log_p_c_if_d1_mtx + thetas.reshape(1, -1) + beta_sum_mtx
        new_d_mtx = (lhs < rhs).astype(int)
        if np.array_equal(new_d_mtx, d_mtx):
            break
        d_mtx = new_d_mtx
    return d_mtx


def update_theta_beta(
    d_map: np.ndarray,  # Positive phase: e.g., the MAP estimate from ICM
    d_samples: list,  # Negative phase: Gibbs samples from current persistent chain
    thetas: np.ndarray,
    betas: np.ndarray,
    persistent_chain: np.ndarray,  # Persistent chain from previous iterations
    alpha: float = 1.0,
    lambda_theta: float = 0.1,
    lambda_beta: float = 0.1,
    momentum: float = 0.9,  # How much of the old chain to keep
) -> tuple:
    """TODO: Add docstring."""
    # Compute a new model expectation from the current Gibbs samples.
    d_model_current = np.mean(np.stack(d_samples, axis=0), axis=0)
    # Update the persistent chain via momentum to smooth the model expectation.
    persistent_chain_new = (
        momentum * persistent_chain + (1 - momentum) * d_model_current
    )

    n = d_map.shape[0]

    # Positive phase: use the MAP estimate for the data
    empirical_mean = d_map.mean(axis=0)
    # Negative phase: use the persistent chain for the model expectation
    model_mean = persistent_chain_new.mean(axis=0)
    theta_grad = empirical_mean - model_mean

    # Compute the positive (empirical) and negative (model) correlations
    empirical_corr = d_map.T @ d_map / n
    model_corr = persistent_chain_new.T @ persistent_chain_new / n
    beta_grad = empirical_corr - model_corr

    # Update parameters with the computed gradients.
    thetas_new = thetas + alpha * theta_grad
    betas_new = betas + alpha * beta_grad

    def _soft_threshold(x: np.array, thresh: float) -> np.array:
        return np.sign(x) * np.maximum(np.abs(x) - thresh, 0)

    thetas_new = _soft_threshold(thetas_new, alpha * lambda_theta)
    betas_new = _soft_threshold(betas_new, alpha * lambda_beta)

    return thetas_new, betas_new, persistent_chain_new


def run_dialog(
    cnt_mtx_df: pd.DataFrame,
    gene_to_bmr: dict,
    out_dir: Path,
    max_iters: int = 10,
    tol: float = 1e-3,
) -> None:
    """TODO: Add docstring."""
    rng = np.random.default_rng()
    num_samples, num_genes = cnt_mtx_df.shape
    thetas = initialize_thetas(cnt_mtx_df, gene_to_bmr)
    betas = initialize_betas(cnt_mtx_df)
    d_mtx = rng.integers(0, 2, (num_samples, num_genes))
    persistent_chain = d_mtx.copy()  # initialize persistent chain

    for _ in range(max_iters):
        d_mtx = run_icm(d_mtx, cnt_mtx_df, gene_to_bmr, thetas, betas)
        d_samples = gibbs_sampling(d_mtx, cnt_mtx_df, gene_to_bmr, thetas, betas, rng)
        new_thetas, new_betas, persistent_chain = update_theta_beta(
            d_map=d_mtx,
            d_samples=d_samples,
            thetas=thetas,
            betas=betas,
            persistent_chain=persistent_chain,
            alpha=1.0,
            lambda_theta=0.1,
            lambda_beta=0.1,
            momentum=0.9,
        )

        diff_theta = np.linalg.norm(new_thetas - thetas)
        diff_betas = np.linalg.norm(new_betas - betas)
        if diff_theta < tol and diff_betas < tol:
            break

        thetas, betas = new_thetas, new_betas

    theta_df, beta_df = save_results(
        genes=cnt_mtx_df.columns.to_list(),
        thetas=thetas,
        betas=betas,
        out_dir=out_dir,
        theta_filename="thetas.csv",
        beta_filename="betas.csv",
        theta_col="theta",
        beta_col="beta",
    )
    return theta_df, beta_df


def create_precision_recall_curve(
    dfs: pd.DataFrame,
    col_names: str,
    info: Path,
    analysis_type: str,
    fout: str,
    font_scale: float = FONT_SCALE,
    figsize: tuple = (5, 4),
) -> None:
    """TODO: Add docstring."""
    with info.open() as f:
        gt = json.load(f)
    key = "ME Pairs" if analysis_type == "me" else "CO Pairs"
    true_driver_interactions = {tuple(sorted(pair)) for pair in gt.get(key)}
    plt.figure(figsize=figsize)

    for df, col_name in zip(dfs, col_names):
        y_true = [
            tuple(sorted(pair)) in true_driver_interactions
            for pair in zip(df["gene_a"], df["gene_b"])
        ]
        y_score = df[col_name] if col_name == "precision" else -df[col_name]
        if analysis_type == "co":
            y_score = -y_score
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        auc_val = average_precision_score(y_true, y_score)

        plt.plot(
            recall,
            precision,
            label=f"{col_name} (AUC={auc_val:.3f})",
            linewidth=font_scale * 2,
            alpha=0.75,
        )

    total_positives = sum(y_true)
    total_samples = len(y_true)
    baseline = total_positives / total_samples if total_samples > 0 else 0
    plt.axhline(
        y=baseline,
        color="gray",
        label=f"Baseline (AUC={baseline:.3f})",
        linewidth=font_scale * 2,
        alpha=0.75,
    )
    style_plot("Recall", "Precision", font_scale)
    plt.savefig(f"{fout}.svg", dpi=300, transparent=False)


def create_recall_at_k_curve(
    dfs: list,
    col_names: list,
    info: Path,
    analysis_type: str,
    fout: str,
    font_scale: float = FONT_SCALE,
    max_k: int = 500,
    figsize: tuple = (5, 4),
) -> None:
    """TODO: Add docstring."""
    with info.open() as f:
        gt = json.load(f)
    key = "ME Pairs" if analysis_type == "me" else "CO Pairs"
    true_driver_interactions = {tuple(sorted(pair)) for pair in gt.get(key)}

    plt.figure(figsize=figsize)

    def get_recalls_at_k(
        true_pairs: set,
        top_ranked_table: pd.DataFrame,
        num_positive: int,
        max_k: int,
    ) -> list:
        recalls_at_k = []
        for k in range(1, max_k + 1):  # Start from k=1
            top_k_pairs_set = {
                tuple(sorted(pair))
                for pair in top_ranked_table.head(k)[["gene_a", "gene_b"]].to_numpy()
            }
            recalls_at_k.append(len(top_k_pairs_set & true_pairs) / num_positive)
        return recalls_at_k

    for df, col_name in zip(dfs, col_names):
        ascending = col_name != "precision"
        if analysis_type == "co":
            ascending = not ascending
        num_positive = len(true_driver_interactions)
        recall_at_k = get_recalls_at_k(
            true_driver_interactions,
            df.sort_values(by=col_name, ascending=ascending),
            num_positive,
            max_k,
        )

        plt.plot(
            np.arange(1, max_k + 1),
            recall_at_k,
            label=f"{col_name}",
            linewidth=font_scale * 2,
            alpha=0.75,
        )

    k_values = np.arange(1, max_k + 1)
    baseline_recall = k_values / 19900
    plt.plot(
        k_values,
        baseline_recall,
        color="gray",
        label="Baseline",
        linewidth=font_scale * 2,
        alpha=0.75,
    )

    style_plot("Top K Ranked Interactions", "Recall@K", font_scale)
    plt.savefig(f"{fout}.svg", dpi=300, transparent=False)


def main() -> None:
    """TODO: Add docstring."""
    parser = _build_argument_parser()
    args = parser.parse_args()
    cnt_mtx_df = pd.read_csv(args.dir / "count_matrix.csv", index_col=0)
    highest_mutated_genes = (
        cnt_mtx_df.sum().sort_values(ascending=False).head(args.num_genes)
    )
    genes = list(highest_mutated_genes.index)
    sub_cnt_mtx_df = cnt_mtx_df[genes]
    bmr_pmf_df = pd.read_csv(args.dir / "bmr_pmfs.csv", index_col=0)
    gene_to_bmr_pmf_raw = bmr_pmf_df.T.to_dict(orient="list")
    gene_to_bmr_pmf = {
        k: [val for val in pmf_vals if not np.isnan(val)]
        for k, pmf_vals in gene_to_bmr_pmf_raw.items()
    }
    sub_gene_to_bmr_pmf = {gene: gene_to_bmr_pmf[gene] for gene in cnt_mtx_df.columns}

    precision_df, precision_bmr_df, theta_naught_df, beta_naught_df = (
        run_standard_methods(
            sub_cnt_mtx_df,
            sub_gene_to_bmr_pmf,
            args.dir,
        )
    )

    theta_df, beta_df = run_dialog(
        sub_cnt_mtx_df,
        sub_gene_to_bmr_pmf,
        args.dir,
    )

    create_precision_recall_curve(
        [precision_df, beta_naught_df, beta_df],
        ["precision", "beta_naught", "beta"],
        args.dir / "matrix_simulation_info.json",
        args.analysis_type,
        args.dir / "precision_recall_curve",
    )

    create_recall_at_k_curve(
        [precision_df, beta_naught_df, beta_df],
        ["precision", "beta_naught", "beta"],
        args.dir / "matrix_simulation_info.json",
        args.analysis_type,
        args.dir / "recall_at_k_curve",
    )


if __name__ == "__main__":
    main()
