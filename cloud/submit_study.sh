#!/bin/bash
# Usage: submit_study.sh "EPOCHS:SEED EPOCHS:SEED ..."  [DEP_JOB]
# Spreads channel trainings round-robin over GPUs (gres g<i>), scoring after each seed's 3 channels.
set -euo pipefail
cd $HOME/agentcell_work/setup_files
NGPU=$(nvidia-smi -L | wc -l); k=0
DEP=${2:+-d afterok:$2}
for ES in $1; do
  EP=${ES%%:*}; S=${ES##*:}
  OB=$HOME/agentcell_work/runs/study/ep${EP}_seed${S}
  TL=$([ $EP -gt 200 ] && echo 23:00:00 || echo 8:00:00)
  IDS=()
  for CH in DAPI Tubulin Actin; do
    G=$((k % NGPU)); k=$((k+1))
    IDS+=($(sbatch --parsable $DEP -t $TL --gres=g$G:1 -J agentCell_train_ep${EP}_s${S}_${CH} \
      -o $HOME/agentcell_work/logs/agentCell_train_ep${EP}_s${S}_${CH}_%j.out \
      --export=ALL,SEED=$S,CH=$CH,EPOCHS=$EP,OUTBASE=$OB,GPU=$G sb_train_channel.sbatch))
  done
  G=$((k % NGPU))
  SC=$(sbatch --parsable -t 2:00:00 --gres=g$G:1 -d afterok:$(IFS=:; echo "${IDS[*]}") -J agentCell_score_ep${EP}_s${S} \
       -o $HOME/agentcell_work/logs/agentCell_score_ep${EP}_s${S}_%j.out --export=ALL,OUTBASE=$OB,GPU=$G sb_score_seed.sbatch)
  echo "ep$EP seed$S: train ${IDS[*]} score $SC"
done
