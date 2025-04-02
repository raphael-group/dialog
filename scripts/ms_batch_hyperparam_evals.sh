#!/bin/bash

ME_SIM_ROOT="data/simulations/UCEC/NS5K_TL0.15_TH0.25_25ME_0CO_150LP"
ME_SIM_INFO=$ME_SIM_ROOT/matrix_simulation_info.json
ME_RES_ROOT="output/NS5K_TL0.15_TH0.25_25ME_0CO_150LP"
ME_OUT_DIR="output/me_hpo_results"
for ME_RES_DIR in "$ME_RES_ROOT"/*; do
    NAME=$(basename "$ME_RES_DIR")
    sbatch scripts/ss_single_hyperparam_eval.sh $ME_SIM_INFO "$ME_RES_DIR" $ME_OUT_DIR ME
    sleep 0.1
done

CO_SIM_ROOT="data/simulations/UCEC/NS5K_TL0.02_TH0.06_0ME_25CO_150LP"
CO_SIM_INFO=$CO_SIM_ROOT/matrix_simulation_info.json
CO_RES_ROOT="output/NS5K_TL0.02_TH0.06_0ME_25CO_150LP"
CO_OUT_DIR="output/co_hpo_results"
for CO_RES_DIR in "$CO_RES_ROOT"/*; do
    NAME=$(basename "$CO_RES_DIR")
    sbatch scripts/ss_single_hyperparam_eval.sh $CO_SIM_INFO "$CO_RES_DIR" $CO_OUT_DIR CO
    sleep 0.1
done
