#!/bin/bash
#SBATCH -J agentCell_train
#SBATCH -c 4
#SBATCH --mem=24G
#SBATCH -o %x_%j.out
# One WS-DINO channel (compound weak label). Env: SEED CH EPOCHS OUTBASE GPU
set -euo pipefail
source $HOME/agentcell_work/setup_files/env.sh
export CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
declare -A IDX=([DAPI]=0 [Tubulin]=1 [Actin]=2)
O=$OUTBASE/train_${CH}_compound; mkdir -p $O
CK=$O/${CH}_weak_compound_DINO_checkpoint.pth
if [ -f $O/DONE ]; then echo "already done"; exit 0; fi
# resume if a partial run exists (checkpoint has epoch), else start from ImageNet init
if [ -f $CK ]; then cp $CK $O/pretrain_full_checkpoint.pth; else cp $WSDINO_REPO/pretrain_full_checkpoint.pth $O/; fi
cd $WSDINO_REPO
export RANK=0 WORLD_SIZE=1 MASTER_ADDR=127.0.0.1 LOCAL_RANK=0
export MASTER_PORT=$((20000 + SLURM_JOB_ID % 20000))
WARM=$(( EPOCHS < 11 ? EPOCHS-1 : 10 ))
echo "seed=$SEED ch=$CH epochs=$EPOCHS gpu=$GPU out=$O start $(date)"
python WS-DINO_BBBC021.py --data_path $AGENTCELL_DATA/BBBC021_annotated_wsdino.csv --output_dir $O \
  --channel_index ${IDX[$CH]} --weak_label_header Unique_Compounds --arch vit_small --patch_size 8 \
  --batch_size_per_gpu 16 --epochs $EPOCHS --warmup_epochs $WARM --warmup_teacher_temp_epochs $WARM \
  --lr 0.000004 --min_lr 0.000003 --momentum_teacher 0.99 --weight_decay 0.0 --weight_decay_end 0.0 \
  --clip_grad 0 --global_crops_scale 0.1 0.2 --local_crops_scale 0.04 0.08 --local_crops_number 8 \
  --saveckp_freq 50 --num_workers 4 --seed $SEED 2>&1 | grep -vE '^Epoch: \[[0-9]+\] +\[ *[0-9]*[1-9]/'
python - <<PY
import torch; c=torch.load("$CK",map_location="cpu",weights_only=False); print("final epoch",c["epoch"]); assert c["epoch"]==$EPOCHS
PY
touch $O/DONE; echo "done $(date)"
