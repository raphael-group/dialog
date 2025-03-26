#!/bin/bash

SIM_ROOT="data/simulations/UCEC/NS5K_TL0.15_TH0.25_25ME_0CO_150LP"
SIM_INFO=$SIM_ROOT/matrix_simulation_info.json
RES_ROOT="output/NS5K_TL0.15_TH0.25_25ME_0CO_150LP"
OUT_DIR="hyperparam_search"

for RES_DIR in "$RES_ROOT"/*; do
    NAME=$(basename "$RES_DIR")
    sbatch scripts/ss_single_hyperparam_eval.sh $SIM_INFO $RES_DIR $OUT_DIR
    sleep 0.05
done
