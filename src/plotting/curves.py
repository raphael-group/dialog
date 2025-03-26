"""TODO: Add docstring."""

import json
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve
from styling import set_plot_font_and_scale, set_plot_style


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
    plt.plot(objective_values)
    plt.xlabel("Iteration")
    plt.ylabel("Objective")
    plt.savefig(out_fn)


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
    plt.plot(theta_diffs, label="Theta")
    plt.plot(beta_diffs, label="Beta")
    plt.xlabel("Iteration")
    plt.ylabel("Parameter Difference")
    plt.legend()
    plt.savefig(out_fn)


def draw_precision_recall_curve(
    simulation_info_fn: Path,
    results_fn: Path,
    out_fn: Path,
) -> None:
    """TODO: Add docstring."""
    with simulation_info_fn.open() as f:
        gt = json.load(f)
    key = "ME Pairs"
    true_ixns = {tuple(sorted(pair)) for pair in gt.get(key)}

    init_results_fn = results_fn.with_name("iter_0.npy")

    set_plot_font_and_scale()
    set_plot_style()
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
        y_score = -betas["beta"]
        auc_val = average_precision_score(y_true, y_score)
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        plt.plot(recall, precision, label=f"{name} AUC: {auc_val:.3f}")

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.savefig(out_fn)
