#!/bin/bash
# Resubmit trainings whose job is gone but whose channel has no DONE marker, then re-chain scoring.
# Usage: resubmit_failed.sh "EPOCHS:SEED ..."
set -euo pipefail
cd $HOME/agentcell_work/setup_files
source env.sh
python -c "import torch; torch.hub.list('facebookresearch/xcit:main')" >/dev/null 2>&1 || true
NGPU=$(nvidia-smi -L | wc -l); k=0
for ES in $1; do
  EP=${ES%%:*}; S=${ES##*:}; OB=$HOME/agentcell_work/runs/study/ep${EP}_seed${S}
  TL=$([ $EP -gt 200 ] && echo 23:00:00 || echo 8:00:00)
  DEPS=()
  for CH in DAPI Tubulin Actin; do
    NAME=agentCell_train_ep${EP}_s${S}_${CH}
    LIVE=$(squeue -h -n $NAME -o %i | head -1)
    if [ -n "$LIVE" ]; then DEPS+=($LIVE); continue; fi
    [ -f $OB/train_${CH}_compound/DONE ] && continue
    G=$((k % NGPU)); k=$((k+1))
    DEPS+=($(sbatch --parsable -t $TL --gres=g$G:1 -J $NAME -o $HOME/agentcell_work/logs/${NAME}_%j.out \
      --export=ALL,SEED=$S,CH=$CH,EPOCHS=$EP,OUTBASE=$OB,GPU=$G sb_train_channel.sbatch))
    echo "resubmitted $NAME -> ${DEPS[-1]}"
  done
  SN=agentCell_score_ep${EP}_s${S}
  OLD=$(squeue -h -n $SN -o %i | tr '\n' ' '); [ -n "$OLD" ] && scancel $OLD
  [ -f $OB/score_per_channel_l2.json ] && continue
  D=""; [ ${#DEPS[@]} -gt 0 ] && D="-d afterok:$(IFS=:; echo "${DEPS[*]}")"
  sbatch --parsable -t 2:00:00 --gres=g$((k % NGPU)):1 $D -J $SN -o $HOME/agentcell_work/logs/${SN}_%j.out \
    --export=ALL,OUTBASE=$OB,GPU=$((k % NGPU)) sb_score_seed.sbatch
done
squeue -h -o '%i %j %T %R'
