#!/bin/bash
#SBATCH -J agentCell_score
#SBATCH -c 6
#SBATCH --mem=24G
#SBATCH -o %x_%j.out
# Embed with the 3 per-channel FINAL checkpoints of OUTBASE; score plain + per_channel_l2. Env: OUTBASE GPU
# (A clean dir of symlinks to the final checkpoints only: the training dirs also hold
#  pretrain_full_checkpoint.pth and epoch snapshots, which tool_embed's resolver can pick up.)
set -euo pipefail
source $HOME/agentcell_work/setup_files/env.sh
export CUDA_VISIBLE_DEVICES=$GPU
FIN=$OUTBASE/final_ckpts; rm -rf $FIN
for CH in DAPI Tubulin Actin; do
  CK=$OUTBASE/train_${CH}_compound/${CH}_weak_compound_DINO_checkpoint.pth
  [ -f $CK ] || { echo "missing $CK"; exit 1; }
  mkdir -p $FIN/train_${CH}_compound; ln -s $CK $FIN/train_${CH}_compound/
done
python - <<PY
import torch
for ch in ["DAPI","Tubulin","Actin"]:
    c=torch.load("$FIN/train_%s_compound/%s_weak_compound_DINO_checkpoint.pth"%(ch,ch),map_location="cpu",weights_only=False)
    print("using", ch, "epoch", c["epoch"])
PY
T=$(basename $OUTBASE); cd $WSDINO_REPO
python $AGENTCELL_TOOLS/tool_embed.py --data-root $AGENTCELL_DATA --model vit_trained --channels all_three \
  --init-dir $FIN --allow-external --tag $T --features-out $OUTBASE/features --result-out $OUTBASE/embed.json | tail -1 | cut -c1-300
for N in plain per_channel_l2; do
  python $AGENTCELL_TOOLS/tool_score.py --features-dir $OUTBASE/features --tag $T --normalization $N \
    --result-out $OUTBASE/score_$N.json | tail -1
done
