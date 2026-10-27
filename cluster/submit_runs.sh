#!/bin/bash
# Usage: submit_runs.sh SPECFILE [DEPJOB]
# spec line: NAME WEAK EPOCHS SEED SAVEFREQ VENV(sys|251) PREP(cp|um|lincs)
# Trainings take 2 of the 3 gres slots of a GPU, embeddings 1, so each GPU runs one training and one embedding.
set -euo pipefail
W=$HOME/agentcell_work; S=$W/setup_files; cd $S
NGPU=$(nvidia-smi -L | wc -l); k=${K0:-0}
DEP=""; [ -n "${2:-}" ] && DEP="-d afterok:$2"
data_of(){ case $1 in cp) echo $W/data/bbbc021_wsdino;; um) echo $W/data/bbbc021_wsdino_unitmean;; lincs) echo $W/data/lincs_wsdino;; esac; }
while read NAME WEAK EP SEED SF VE PR; do
  [ -z "$NAME" ] && continue; [ "${NAME:0:1}" = "#" ] && continue
  OB=$W/runs/rev/$NAME; DATA=$(data_of $PR)
  VENVP=$W/venv; [ $VE = 251 ] && VENVP=$W/venv251
  TL=$([ $EP -gt 200 ] && echo 60:00:00 || echo 20:00:00)
  for CH in DAPI Tubulin Actin; do
    G=$((k % NGPU)); k=$((k+1))
    if [ "$EP" = 0 ]; then
      sbatch --parsable $DEP -t 3:00:00 --gres=g$G:1 -J agentCell_em_${NAME}_$CH -o $W/logs/agentCell_em_${NAME}_${CH}_%j.out \
        --export=ALL,MODEL=$WEAK,CH=$CH,RUN=$NAME,GPU=$G,VENV=$VENVP,DATA_OVERRIDE=$DATA $S/sb_embed.sbatch >/dev/null
      continue
    fi
    T=$(sbatch --parsable $DEP -t $TL --gres=g$G:2 -J agentCell_tr_${NAME}_$CH -o $W/logs/agentCell_tr_${NAME}_${CH}_%j.out \
       --export=ALL,SEED=$SEED,CH=$CH,EPOCHS=$EP,OUTBASE=$OB,GPU=$G,WEAK=$WEAK,SAVEFREQ=$SF,VENV=$VENVP,DATA_OVERRIDE=$DATA $S/sb_train.sbatch)
    sbatch --parsable -d afterok:$T -t 10:00:00 --gres=g$G:1 -J agentCell_em_${NAME}_$CH -o $W/logs/agentCell_em_${NAME}_${CH}_%j.out \
       --export=ALL,TRAIN_DIR=$OB/train_${CH}_${WEAK},CH=$CH,RUN=$NAME,GPU=$G,VENV=$VENVP,DATA_OVERRIDE=$DATA $S/sb_embed.sbatch >/dev/null
  done
done < $1
squeue -h | wc -l
