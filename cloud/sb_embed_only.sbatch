#!/bin/bash
#SBATCH -J agentCell_embedone
#SBATCH -c 4
#SBATCH --mem=24G
#SBATCH -o %x_%j.out
# Embed ONE finished channel. Env: CH OUTBASE GPU
set -euo pipefail
source $HOME/agentcell_work/setup_files/env.sh
export CUDA_VISIBLE_DEVICES=$GPU
cd $WSDINO_REPO
python $HOME/agentcell_work/setup_files/embed_one.py --ckpt $OUTBASE/train_${CH}_compound/${CH}_weak_compound_DINO_checkpoint.pth \
  --channel $CH --tag $(basename $OUTBASE) --features-out $OUTBASE/features
