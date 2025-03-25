"""TODO: Add docstring."""

from argparse import ArgumentParser
from pathlib import Path


def build_dialog_argument_parser() -> ArgumentParser:
    """TODO: Add docstring."""
    parser = ArgumentParser()
    parser.add_argument("-c", "--cnt_mtx", type=Path, required=True)
    parser.add_argument("-b", "--bmr_pmfs", type=Path, required=True)
    parser.add_argument("-o", "--out_dir", type=Path, required=True)
    parser.add_argument("-i", "--num_iter", type=int, default=1_000)
    parser.add_argument("-s", "--num_gibbs_samples", type=int, default=1)
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.1)
    parser.add_argument("-lt", "--lambda_theta", type=float, default=0.1)
    parser.add_argument("-lb", "--lambda_beta", type=float, default=0.1)
    parser.add_argument("-m", "--momentum", type=float, default=0.9)
    return parser
