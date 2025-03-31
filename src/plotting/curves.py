"""TODO: Add docstring."""

import json
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve

from plotting.styling import set_plot_font_and_scale, set_plot_style


def draw_dialog_objective_optimization_curve(
    results_dir: Path,
    num_iter: int,
    out_fn: Path,
) -> None:
    """TODO: Add docstring."""
    objective_values = []
    for it in range(num_iter):
        training_history_fn = results_dir / f"iter_{it}.npy"
        if not training_history_fn.exists():
            continue
        training_history = np.load(training_history_fn, allow_pickle=True).item()
        objective_values.append(training_history["objective"])
    set_plot_font_and_scale()
    set_plot_style()
    plt.figure()
    plt.plot(objective_values)
    plt.xlabel("Iteration")
    plt.ylabel("Objective")
    plt.savefig(out_fn)
    plt.close()


def draw_dialog_parameter_convergence_curve(
    results_dir: Path,
    num_iter: int,
    out_fn: Path,
) -> None:
    """TODO: Add docstring."""
    initial_params_fn = results_dir / "iter_0.npy"
    initial_params = np.load(initial_params_fn, allow_pickle=True).item()
    prev_theta, prev_beta = initial_params["thetas"], initial_params["betas"]
    theta_diffs, beta_diffs = [], []
    for it in range(1, num_iter):
        training_history_fn = results_dir / f"iter_{it}.npy"
        if not training_history_fn.exists():
            continue
        training_history = np.load(training_history_fn, allow_pickle=True).item()
        theta, beta = training_history["thetas"], training_history["betas"]
        theta_diffs.append(np.linalg.norm(theta - prev_theta))
        beta_diffs.append(np.linalg.norm(beta - prev_beta))
        prev_theta, prev_beta = theta, beta
    set_plot_font_and_scale()
    set_plot_style()
    plt.figure()
    plt.plot(theta_diffs, label="Theta")
    plt.plot(beta_diffs, label="Beta")
    plt.xlabel("Iteration")
    plt.ylabel("Parameter Difference")
    plt.legend()
    plt.savefig(out_fn)
    plt.close()


def draw_precision_recall_curve(
    simulation_info_fn: Path,
    results_fn: Path,
    out_fn: Path,
    ixn_type: str = "ME",
) -> None:
    """TODO: Add docstring."""
    with simulation_info_fn.open() as f:
        gt = json.load(f)
    key = f"{ixn_type} Pairs"
    true_ixns = {tuple(sorted(pair)) for pair in gt.get(key)}

    init_results_fn = results_fn.with_name("iter_0.npy")

    set_plot_font_and_scale()
    set_plot_style()
    plt.figure()
    for name, res_fn in zip(["Init", "Final"], [init_results_fn, results_fn]):
        results = np.load(res_fn, allow_pickle=True).item()
        genes = results["gene_names"]
        betas = pd.DataFrame(
            [
                {"gene_a": genes[g], "gene_b": genes[h], "beta": results["betas"][g, h]}
                for g, h in combinations(range(len(genes)), 2)
            ],
        )
        y_true = [
            tuple(sorted(pair)) in true_ixns
            for pair in zip(betas["gene_a"], betas["gene_b"])
        ]
        y_score = betas["beta"] if ixn_type == "ME" else -betas["beta"]
        auc_val = average_precision_score(y_true, y_score)
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        plt.plot(recall, precision, label=f"{name} AUC: {auc_val:.3f}")

    # read and plot naive inverse covariance matrix results
    naive_results_fn = results_fn.parents[1] / "naive_precision_matrix.csv"
    naive_results_df = pd.read_csv(naive_results_fn, index_col=0)
    naive_results_arr = naive_results_df.to_numpy()
    naive_results = pd.DataFrame(
        [
            {
                "gene_a": naive_results_df.columns[g],
                "gene_b": naive_results_df.columns[h],
                "inv_cov": naive_results_arr[g, h],
            }
            for g, h in combinations(range(len(naive_results_df.columns)), 2)
        ],
    )
    y_true = [
        tuple(sorted(pair)) in true_ixns
        for pair in zip(naive_results["gene_a"], naive_results["gene_b"])
    ]
    y_score = (
        -naive_results["inv_cov"] if ixn_type == "ME" else naive_results["inv_cov"]
    )
    auc_val = average_precision_score(y_true, y_score)
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    plt.plot(recall, precision, label=f"Naive AUC: {auc_val:.3f}")

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.savefig(out_fn)
    plt.close()


def draw_recall_at_k_curve(
    simulation_info_fn: Path,
    results_fn: Path,
    out_fn: Path,
    ixn_type: str = "ME",
    max_k: int = 100,
) -> None:
    """TODO: Add docstring."""

    def _get_recalls_at_k(
        true_ixns: set,
        ranked_ixns: pd.DataFrame,
        num_positive: int,
        max_k: int,
    ) -> list:
        return [
            len(
                {
                    tuple(sorted(ixn))
                    for ixn in ranked_ixns.head(k)[["gene_a", "gene_b"]].to_numpy()
                }
                & true_ixns,
            )
            / num_positive
            for k in range(1, max_k + 1)
        ]

    with simulation_info_fn.open() as f:
        gt = json.load(f)
    key = f"{ixn_type} Pairs"
    true_ixns = {tuple(sorted(pair)) for pair in gt.get(key)}

    init_results_fn = results_fn.with_name("iter_0.npy")


    set_plot_font_and_scale()
    set_plot_style()
    plt.figure()

    for name, res_fn in zip(["Init", "Final"], [init_results_fn, results_fn]):
        results = np.load(res_fn, allow_pickle=True).item()
        genes = results["gene_names"]
        betas = pd.DataFrame(
            [
                {"gene_a": genes[g], "gene_b": genes[h], "beta": results["betas"][g, h]}
                for g, h in combinations(range(len(genes)), 2)
            ],
        )
        recalls_at_k = _get_recalls_at_k(
            true_ixns=true_ixns,
            ranked_ixns=betas.sort_values(by="beta", ascending=ixn_type != "ME"),
            num_positive=len(true_ixns),
            max_k=max_k,
        )
        plt.plot(range(1, max_k + 1), recalls_at_k, label=f"{name}")

    # read and plot naive inverse covariance matrix results
    naive_results_fn = results_fn.parents[1] / "naive_precision_matrix.csv"
    naive_results_df = pd.read_csv(naive_results_fn, index_col=0)
    naive_results_arr = naive_results_df.to_numpy()
    naive_results = pd.DataFrame(
        [
            {
                "gene_a": naive_results_df.columns[g],
                "gene_b": naive_results_df.columns[h],
                "inv_cov": naive_results_arr[g, h],
            }
            for g, h in combinations(range(len(naive_results_df.columns)), 2)
        ],
    )
    recalls_at_k = _get_recalls_at_k(
        true_ixns=true_ixns,
        ranked_ixns=naive_results.sort_values(by="inv_cov", ascending=ixn_type == "ME"),
        num_positive=len(true_ixns),
        max_k=max_k,
    )
    plt.plot(range(1, max_k + 1), recalls_at_k, label="Naive")

    plt.ylim(0, 1)
    plt.xlabel("Top K Ranked Interactions")
    plt.ylabel("Recall@K")
    plt.legend()
    plt.savefig(out_fn)
    plt.close()
