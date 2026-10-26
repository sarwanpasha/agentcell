#!/bin/bash
# bootstrap.sh "<EPOCHS:SEED list>" [fixed]   -- run once per node
set -euo pipefail
W=$HOME/agentcell_work; mkdir -p $W/logs; cd $W/setup_files
SLOTS=${SLOTS:-3} bash slurm_setup.sh > $W/logs/slurm_setup.log 2>&1
SETUP=$(sbatch --parsable sb_setup.sbatch)
echo "setup job $SETUP"
bash submit_study.sh "$1" $SETUP
if [ "${2:-}" = fixed ]; then
  sbatch --parsable -t 2:00:00 --gres=g0:1 -d afterok:$SETUP --export=ALL,GPU=0 sb_fixed.sbatch
fi
squeue -o '%.5i %.32j %.8T %.4C %.10b %R'
