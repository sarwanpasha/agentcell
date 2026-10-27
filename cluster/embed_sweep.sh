#!/bin/bash
# Launch embeddings for finished trainings that have no Gram files yet (GPUs given in SWEEP_GPUS).
W=$HOME/agentcell_work; S=$W/setup_files; k=0; read -a GS <<< "${SWEEP_GPUS:-6 7}"
for d in $W/runs/rev/*/train_*; do
  [ -f $d/DONE ] || continue
  run=$(basename $(dirname $d)); ch=$(basename $d | cut -d_ -f2)
  n=$(ls $W/grams/$run/${ch}_e*.npz 2>/dev/null | wc -l)
  lock=$W/logs/sweep_${run}_${ch}.lock
  [ -f $lock ] && continue
  if [ "$n" = 0 ]; then
    G=${GS[$((k % ${#GS[@]}))]}; k=$((k+1))
    VENV=$W/venv; case $run in f_251*) VENV=$W/venv251;; esac
    DATA=$W/data/bbbc021_wsdino; case $run in f_*_um*) DATA=$W/data/bbbc021_wsdino_unitmean;; L_*) DATA=$W/data/lincs_wsdino;; esac
    touch $lock
    TRAIN_DIR=$d CH=$ch RUN=$run GPU=$G VENV=$VENV DATA_OVERRIDE=$DATA nohup bash $S/sb_embed.sbatch > $W/logs/run_em_${run}_${ch}_sweep.out 2>&1 &
    echo "sweep launched $run $ch gpu$G"
  fi
done
