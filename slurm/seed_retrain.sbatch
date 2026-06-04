#!/bin/bash
#SBATCH --job-name=seedretrain
#SBATCH --partition=gh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=06:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# Train all three channels at the fixed best config (compound, 100 ep) with a
# specified random seed, to measure downstream-NSC variance from training
# stochasticity. Pass SEED via --export.

set -euo pipefail
# Set PYTHONUSERBASE / virtualenv as appropriate for your site
REPO=${WSDINO_REPO:-$HOME/WS-DINO}
INIT_CKPT=$REPO/pretrain_full_checkpoint.pth
DATA=${AGENTCELL_DATA:-./data/bbbc021_wsdino}/BBBC021_annotated_wsdino.csv

SEED=${SEED:?set SEED}
OUTBASE=${AGENTCELL_RUNS:-./runs}/seed_runs/seed_${SEED}
cd "$REPO"

for PAIR in "0 DAPI" "1 Tubulin" "2 Actin"; do
  set -- $PAIR
  CIDX=$1; CH=$2
  OUTDIR=$OUTBASE/train_${CH}_compound
  mkdir -p "$OUTDIR"
  cp "$INIT_CKPT" "$OUTDIR/pretrain_full_checkpoint.pth"
  export RANK=0 WORLD_SIZE=1 MASTER_ADDR=127.0.0.1 LOCAL_RANK=0
  export MASTER_PORT=$((29700 + SEED*10 + CIDX))
  echo "=== seed $SEED channel $CH (idx $CIDX) start $(date) ==="
  /usr/bin/python WS-DINO_BBBC021.py \
      --data_path "$DATA" --output_dir "$OUTDIR" \
      --channel_index "$CIDX" --weak_label_header Unique_Compounds \
      --arch vit_small --patch_size 8 --batch_size_per_gpu 16 \
      --epochs 100 --warmup_epochs 10 --warmup_teacher_temp_epochs 10 \
      --lr 0.000004 --min_lr 0.000003 --momentum_teacher 0.99 \
      --weight_decay 0.0 --weight_decay_end 0.0 --clip_grad 0 \
      --global_crops_scale 0.1 0.2 --local_crops_scale 0.04 0.08 \
      --local_crops_number 8 --saveckp_freq 50 --num_workers 8 \
      --seed "$SEED"
done
echo "=== seed $SEED all channels done $(date) ==="
