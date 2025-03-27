"""TODO: Add docstring."""
from argparse import ArgumentParser
from pathlib import Path

import pandas as pd
from naive_inverse_covariance import run_naive_inverse_covariance


def _build_argument_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument("-c", "--cnt_mtx", type=Path, required=True)
    parser.add_argument("-o", "--out_dir", type=Path, required=True)
    return parser

def main() -> None:
    """TODO: Add docstring."""
    parser = _build_argument_parser()
    args = parser.parse_args()
    cnt_mtx_df = pd.read_csv(args.cnt_mtx, index_col=0)
    precision_mtx = run_naive_inverse_covariance(cnt_mtx_df)
    precision_mtx_df = pd.DataFrame(
        precision_mtx,
        index=cnt_mtx_df.columns,
        columns=cnt_mtx_df.columns,
    )
    out_fn = args.out_dir / "naive_precision_matrix.csv"
    precision_mtx_df.to_csv(out_fn)


if __name__ == "__main__":
    main()
