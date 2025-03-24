"""TODO: Add docstring."""

import numpy as np
import pandas as pd

from dialog.models import GeneBackgroundPMFs, run_dialog_method
from dialog.utils import build_dialog_argument_parser


def main() -> None:
    """TODO: Add docstring."""
    parser = build_dialog_argument_parser()
    args = parser.parse_args()
    cnt_mtx_df = pd.read_csv(args.cnt_mtx, index_col=0)
    bmr_pmf_df = pd.read_csv(args.bmr_pmfs, index_col=0)
    gene_to_bmr_pmf_raw = bmr_pmf_df.T.to_dict(orient="list")
    gene_to_bmr_pmf = {
        k: [val for val in pmf_vals if not np.isnan(val)]
        for k, pmf_vals in gene_to_bmr_pmf_raw.items()
    }
    bmr_pmfs = GeneBackgroundPMFs(mapping=gene_to_bmr_pmf)
    run_dialog_method(
        cnt_mtx_df=cnt_mtx_df,
        bmr_pmfs=bmr_pmfs,
        out_dir=args.out_dir,
        num_iter=args.num_iter,
        num_gibbs_samples=args.num_gibbs_samples,
        learning_rate=args.learning_rate,
        lambda_theta=args.lambda_theta,
        lambda_beta=args.lambda_beta,
        momentum=args.momentum,
    )


if __name__ == "__main__":
    main()
