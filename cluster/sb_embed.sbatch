#!/bin/bash
#SBATCH -c 6
#SBATCH --mem=32G
# Embed all checkpoints of one channel. Env: TRAIN_DIR CH RUN GPU [MODEL] [VENV] [DATA_OVERRIDE] [ONLY_FINAL]
set -euo pipefail
source $HOME/agentcell_work/setup_files/env.sh
export CUDA_VISIBLE_DEVICES=$GPU
EXTRA=""; [ -n "${ONLY_FINAL:-}" ] && EXTRA="--only-final"
if [ -n "${MODEL:-}" ]; then
  python $W/setup_files/embed_ckpts.py --model $MODEL --channel $CH --run $RUN --gram-dir $W/grams
else
  python $W/setup_files/embed_ckpts.py --train-dir $TRAIN_DIR --channel $CH --run $RUN --gram-dir $W/grams $EXTRA
fi
