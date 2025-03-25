#!/bin/bash
# Create the log directory if it doesn't exist
mkdir -p slurm

# Loop over learning rate from 0.01 to 0.1 (steps of 0.01)
for lr in $(seq 0.01 0.01 0.1); do
  # Loop over momentum from 0.8 to 0.99 (steps of 0.01)
  for m in $(seq 0.8 0.01 0.99); do
    # Loop over lambda from 0.0001 to 0.1 in 10 increments (~0.0111 per step)
    for lambda in $(seq 0.0001 0.0111 0.1); do
      JOB_NAME="dialog_lr${lr}_m${m}_lambda${lambda}"
      LOGFILE="slurm/${JOB_NAME}.out"
      echo "Submitting ${JOB_NAME}"
      sbatch --job-name="${JOB_NAME}" --output="${LOGFILE}" ssbatch.sh "${lr}" "${m}" "${lambda}"
      sleep 0.1
    done
  done
done
