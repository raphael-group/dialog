#!/usr/bin/env bash
#SBATCH -N 1
#SBATCH --ntasks-per-node=10
#SBATCH -t 04:00:00
#SBATCH --mem=8GB
#SBATCH --job-name=dialog

# Get parameters from command-line arguments
LR="$1"
M="$2"
S="$3"
LAMBDA="$4"
ME_SIM_NAME=NS5K_TL0.15_TH0.25_25ME_0CO_150LP
CO_SIM_NAME=NS5K_TL0.02_TH0.06_0ME_25CO_150LP

for SIM_NAME in "$ME_SIM_NAME" "$CO_SIM_NAME"; do
  nice python src/dialog/__main__.py \
    -c data/simulations/UCEC/${SIM_NAME}/count_matrix.csv \
    -b data/simulations/UCEC/${SIM_NAME}/bmr_pmfs.csv \
    -o output/${SIM_NAME}/ \
    -lr "${LR}" -m "${M}" -s "${S}" -lt "${LAMBDA}" -lb "${LAMBDA}"

  nice python src/baselines/__main__.py \
    -c data/simulations/UCEC/${SIM_NAME}/count_matrix.csv \
    -o output/${SIM_NAME}/
done
