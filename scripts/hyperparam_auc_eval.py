"""TODO: Add docstring."""

import json
import re
from argparse import ArgumentParser
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


# ------------------------------------------------------------------------------------ #
#                                   HELPER FUNCTIONS                                   #
# ------------------------------------------------------------------------------------ #
def _build_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument("--sim_info", type=Path, required=True)
    parser.add_argument("--results_dir", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    return parser

def _read_simulation_info(simulation_info_fn: Path) -> dict:
    with simulation_info_fn.open() as f:
        return json.load(f)

# ------------------------------------------------------------------------------------ #
#                                    MAIN FUNCTIONS                                    #
# ------------------------------------------------------------------------------------ #
def evaluate_and_find_best_auc_across_iterations(
    simulation_info: dict,
    results_dir: Path,
    out_dir: Path,
) -> None:
    """TODO: Add docstring."""
    true_ixns = {tuple(sorted(pair)) for pair in simulation_info.get("ME Pairs")}
    simulation_name = results_dir.name
    hyperparameters = [
        re.search(r"\d*\.?\d+", x).group() for x in simulation_name.split("_")
    ]
    num_iter, num_samples, learning_rate, lambda_theta, lambda_beta, momentum = (
        hyperparameters
    )
    max_auc_iter, max_auc_val = 0, 0
    for it in range(int(num_iter)):
        training_history_fn = results_dir / f"iter_{it}.npy"
        if not training_history_fn.exists():
            continue
        training_history = np.load(training_history_fn, allow_pickle=True).item()
        genes, betas = training_history["gene_names"], training_history["betas"]
        betas = pd.DataFrame(
            [
                {"gene_a": genes[g], "gene_b": genes[h], "beta": betas[g, h]}
                for g, h in combinations(range(len(genes)), 2)
            ],
        )
        y_true = [
            tuple(sorted(pair)) in true_ixns
            for pair in zip(betas["gene_a"], betas["gene_b"])
        ]
        y_score = -betas["beta"]
        auc_val = average_precision_score(y_true, y_score)
        if auc_val > max_auc_val:
            max_auc_iter, max_auc_val = it, auc_val
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / f"{simulation_name}.csv"
    with output_path.open("w") as f:
        f.write(
            f"{num_iter},"
            f"{num_samples},"
            f"{learning_rate},"
            f"{lambda_theta},"
            f"{lambda_beta},"
            f"{momentum},"
            f"{max_auc_iter},"
            f"{max_auc_val:.5f}\n",
        )


if __name__ == "__main__":
    parser = _build_parser()
    args = parser.parse_args()
    evaluate_and_find_best_auc_across_iterations(
        _read_simulation_info(args.sim_info),
        args.results_dir,
        args.out_dir,
    )
