#!/bin/bash
#SBATCH --job-name=hyperparam
#SBATCH --time=01:00:00
#SBATCH --mem=2G
#SBATCH --cpus-per-task=4
#SBATCH --output=output/slurm/%x_%j.out
#SBATCH --error=output/slurm/%x_%j.err

SIM_INFO=$1
RESULTS_DIR=$2
OUT_DIR=$3
IXN_TYPE=$4

python scripts/hyperparam_auc_eval.py --results_dir "$RESULTS_DIR" --sim_info "$SIM_INFO" --out_dir "$OUT_DIR" --ixn_type "$IXN_TYPE"
