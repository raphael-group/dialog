#!/usr/bin/env bash
#SBATCH -N 1
#SBATCH --ntasks-per-node=10
#SBATCH -t 04:00:00
#SBATCH --mem=8GB
#SBATCH --job-name=dialog

# Get parameters from command-line arguments
LR="$1"
M="$2"
LAMBDA="$3"

nice python src/dialog/__main__.py \
  -c data/simulations/UCEC/NS5K_TL0.15_TH0.25_25ME_0CO_150LP/count_matrix.csv \
  -b data/simulations/UCEC/NS5K_TL0.15_TH0.25_25ME_0CO_150LP/bmr_pmfs.csv \
  -o output/NS5K_TL0.15_TH0.25_25ME_0CO_150LP/ \
  -lr "${LR}" -m "${M}" -lt "${LAMBDA}" -lb "${LAMBDA}"
