#!/bin/bash
# Preprocessing A/B test: rebuild data with the unit-mean illumination recipe (preprocess_unitmean.py), retrain ep100 seed1 (3 channels), score.
set -euo pipefail
W=$HOME/agentcell_work; S=$W/setup_files; cd $S
for f in sb_train_channel.sbatch sb_score_seed.sbatch; do
  sed 's|source $HOME/agentcell_work/setup_files/env.sh|source $HOME/agentcell_work/setup_files/env.sh; export AGENTCELL_DATA=$HOME/agentcell_work/data/bbbc021_wsdino_unitmean|' $f > unitmean_$f
done
P=$(sbatch --parsable -c 20 --mem=60G -t 1:00:00 -J agentCell_prep_unitmean -o $W/logs/agentCell_prep_unitmean_%j.out --wrap \
  "source $S/env.sh; python $S/preprocess_unitmean.py --raw-dir $W/raw --wsdino-repo $W/WS-DINO --out-dir $W/data/bbbc021_wsdino_unitmean --workers 20")
OB=$W/runs/study_unitmean/ep100_seed1; IDS=(); k=0
for CH in DAPI Tubulin Actin; do
  G=$((k % 2)); k=$((k+1))
  IDS+=($(sbatch --parsable -d afterok:$P -t 6:00:00 --gres=g$G:1 -J agentCell_umtrain_s1_$CH -o $W/logs/agentCell_umtrain_s1_${CH}_%j.out \
     --export=ALL,SEED=1,CH=$CH,EPOCHS=100,OUTBASE=$OB,GPU=$G unitmean_sb_train_channel.sbatch))
done
sbatch --parsable -d afterok:$(IFS=:; echo "${IDS[*]}") -t 1:00:00 --gres=g1:1 -J agentCell_umscore_s1 -o $W/logs/agentCell_umscore_s1_%j.out \
  --export=ALL,OUTBASE=$OB,GPU=1 unitmean_sb_score_seed.sbatch
squeue -h -o '%i %j %T %b %R'
