#!/bin/bash
# bootstrap_one.sh SEED CH EPOCHS  -- 1-GPU node that trains+embeds one channel of a split seed
set -euo pipefail
W=$HOME/agentcell_work; mkdir -p $W/logs; cd $W/setup_files
N=$(( $(nproc) - 2 ))
sed -i "s/^#SBATCH -c 32$/#SBATCH -c $N/; s/--workers 30/--workers $N/" sb_setup.sbatch
SLOTS=1 bash slurm_setup.sh > $W/logs/slurm_setup.log 2>&1
SETUP=$(sbatch --parsable sb_setup.sbatch)
OB=$W/runs/study/ep$3_seed$1
J=$(sbatch --parsable -d afterok:$SETUP -t 23:00:00 --gres=g0:1 -J agentCell_trainembed_ep$3_s$1_$2 \
   -o $W/logs/agentCell_trainembed_ep$3_s$1_$2_%j.out --export=ALL,SEED=$1,CH=$2,EPOCHS=$3,OUTBASE=$OB,GPU=0 sb_train_embed.sbatch)
echo "setup $SETUP trainembed $J"
squeue
