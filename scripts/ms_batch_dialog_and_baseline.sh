#!/bin/bash
mkdir -p output/slurm

for lr in $(seq 0.08 0.02 0.18); do
  for m in $(seq 0.70 0.02 0.90); do
    for ns in 1 3 5 10; do
      for lambda in 0.0001 0.001 0.01; do
        JOB_NAME="dialog_lr${lr}_m${m}_ns${ns}_lambda${lambda}"
        LOGFILE="output/slurm/${JOB_NAME}.out"
        echo "Submitting ${JOB_NAME}"
        sbatch --job-name="${JOB_NAME}" --output="${LOGFILE}" scripts/ss_batch_dialog_and_baseline.sh "${lr}" "${m}" "${ns}" "${lambda}"
        sleep 0.10
      done
    done
  done
done
