#!/bin/bash
#SBATCH --job-name=wsdino
#SBATCH --partition=gh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=10:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# WS-DINO training for ONE channel. Parameterized via environment variables so
# both the manual submit wrapper and the agent's tool_train_ssl can drive it.
#
# Required env:
#   CHANNEL_INDEX : 0=DAPI, 1=Tubulin, 2=Actin
#   CHTAG         : DAPI | Tubulin | Actin
# Optional env (with defaults; agent's tool_train_ssl sets these):
#   WSDINO_OUTDIR   : output dir (default /scratch/.../wsdino_runs/<CHTAG>)
#   WSDINO_EPOCHS   : epochs (default 400)
#   WSDINO_WEAKLABEL: compound|moa|treatment|none (default compound)
#   WSDINO_DATA     : annotated CSV (default the standard wsdino CSV)

set -euo pipefail

# Set PYTHONUSERBASE / virtualenv as appropriate for your site
REPO=${WSDINO_REPO:-$HOME/WS-DINO}
INIT_CKPT=$REPO/pretrain_full_checkpoint.pth

CHANNEL_INDEX=${CHANNEL_INDEX:?set CHANNEL_INDEX (0/1/2)}
CHTAG=${CHTAG:?set CHTAG (DAPI/Tubulin/Actin)}

OUTDIR=${WSDINO_OUTDIR:-${AGENTCELL_RUNS:-./runs}/wsdino_runs/${CHTAG}}
EPOCHS=${WSDINO_EPOCHS:-400}
WEAKLABEL=${WSDINO_WEAKLABEL:-compound}
DATA_CSV=${WSDINO_DATA:-${AGENTCELL_DATA:-./data/bbbc021_wsdino}/BBBC021_annotated_wsdino.csv}

case "$WEAKLABEL" in
  compound)  WEAK_HEADER="Unique_Compounds" ;;
  moa)       WEAK_HEADER="Unique_MoA" ;;
  treatment) WEAK_HEADER="Unique_Treatments" ;;
  none)      WEAK_HEADER="Unique_Compounds" ;;
  *)         WEAK_HEADER="Unique_Compounds" ;;
esac

mkdir -p "$OUTDIR"
cp "$INIT_CKPT" "$OUTDIR/pretrain_full_checkpoint.pth"
cd "$REPO"

export RANK=0 WORLD_SIZE=1 MASTER_ADDR=127.0.0.1
export MASTER_PORT=$((29520 + CHANNEL_INDEX + RANDOM % 1000))
export LOCAL_RANK=0

echo "=== WS-DINO training ==="
echo "channel=$CHTAG idx=$CHANNEL_INDEX  epochs=$EPOCHS  weak=$WEAKLABEL ($WEAK_HEADER)"
echo "outdir=$OUTDIR  data=$DATA_CSV"
echo "started: $(date)"

if [ "$WEAKLABEL" = "none" ]; then
  echo "WARNING: weak_label=none requested; WS-DINO samples by a label by design."
  echo "Running with compound header as closest available; a true unsupervised"
  echo "baseline would require DINO_BBBC021.py instead."
fi

/usr/bin/python WS-DINO_BBBC021.py \
    --data_path "$DATA_CSV" \
    --output_dir "$OUTDIR" \
    --channel_index "$CHANNEL_INDEX" \
    --weak_label_header "$WEAK_HEADER" \
    --arch vit_small --patch_size 8 \
    --batch_size_per_gpu 16 \
    --epochs "$EPOCHS" \
    --warmup_epochs 10 \
    --warmup_teacher_temp_epochs 10 \
    --lr 0.000004 --min_lr 0.000003 \
    --momentum_teacher 0.99 \
    --weight_decay 0.0 --weight_decay_end 0.0 \
    --clip_grad 0 \
    --global_crops_scale 0.1 0.2 \
    --local_crops_scale 0.04 0.08 \
    --local_crops_number 8 \
    --saveckp_freq 50 \
    --num_workers 8

echo "finished: $(date)"
