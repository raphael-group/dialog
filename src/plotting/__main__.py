"""TODO: Add docstring."""

from argparse import ArgumentParser
from pathlib import Path

from plotting import (
    draw_dialog_objective_optimization_curve,
    draw_dialog_parameter_convergence_curve,
    draw_precision_recall_curve,
)


def _build_argument_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument("--simulation_info", type=Path, required=True)
    parser.add_argument("--results_dir", type=Path, required=True)
    parser.add_argument("--num_iter", type=int, required=True)
    parser.add_argument("--best_iter", type=int, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    return parser


if __name__ == "__main__":
    parser = _build_argument_parser()
    args = parser.parse_args()
    draw_dialog_objective_optimization_curve(
        results_dir=args.results_dir,
        num_iter=args.num_iter,
        out_fn=args.out_dir / "optimization_curve.svg",
    )
    draw_dialog_parameter_convergence_curve(
        results_dir=args.results_dir,
        num_iter=args.num_iter,
        out_fn=args.out_dir / "convergence_curve.svg",
    )
    draw_precision_recall_curve(
        simulation_info_fn=args.simulation_info,
        results_fn=args.results_dir / f"iter_{args.best_iter}.npy",
        out_fn=args.out_dir / "precision_recall_curve.svg",
    )
